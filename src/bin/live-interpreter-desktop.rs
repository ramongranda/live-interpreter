use async_trait::async_trait;
use base64::{Engine, engine::general_purpose::STANDARD};
use cpal::Sample;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use live_interpreter::asr::AsrEngine;
use live_interpreter::config::Config;
use live_interpreter::desktop::SaveVoiceProfileRequest;
use live_interpreter::desktop::{
    ActionResult, AppStatus, DesktopConfig, VoiceProfile, collect_status, save_voice_profile,
    start_client, start_server, stop_client, stop_server, voice_profile,
};
use live_interpreter::mesh::{
    AudioChunk, AudioTaskResult, LiveInterpreterMesh, MeshAudioProcessor, MeshCommand, MeshConfig,
    MeshRole, NvmlGpuTelemetry,
};
use live_interpreter::translate::Translator;
use live_interpreter::tts::TtsEngine;
use live_interpreter::types::Direction;
use live_interpreter::vram::{app_vram_telemetry, vram_snapshot};
use serde::Serialize;
use tauri::Emitter;
use std::{
    collections::VecDeque,
    env,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex as StdMutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::Duration,
};
use tauri::State;
use tokio::{
    sync::{Mutex, mpsc},
    task::JoinHandle,
    time::timeout,
};
use tracing_subscriber::{EnvFilter, fmt};

struct NativeState {
    config: DesktopConfig,
    http: reqwest::Client,
    lock: Arc<Mutex<()>>,
    mesh: Mutex<Option<MeshRuntime>>,
    mesh_capture: Mutex<Option<MeshCaptureRuntime>>,
    voice_recorder: StdMutex<Option<VoiceRecorder>>,
}

struct MeshRuntime {
    role: MeshRole,
    commands: mpsc::Sender<MeshCommand>,
    task: JoinHandle<anyhow::Result<()>>,
}

struct MeshCaptureRuntime {
    stop: Arc<AtomicBool>,
    task: thread::JoinHandle<()>,
}

#[derive(Clone)]
enum DesktopMeshProcessor {
    Ready(LiveMeshAudioProcessor),
    Unavailable,
}

#[derive(Clone)]
struct LiveMeshAudioProcessor {
    data_dir: PathBuf,
    asr: Arc<AsrEngine>,
    translator: Translator,
    tts: TtsEngine,
}

struct VoiceRecorder {
    stream: cpal::Stream,
    samples: Arc<StdMutex<Vec<f32>>>,
    sample_rate: u32,
}

#[derive(Debug, Serialize)]
struct MeshStatus {
    running: bool,
    role: Option<MeshRole>,
}

#[tauri::command]
async fn app_status(state: State<'_, NativeState>) -> Result<AppStatus, String> {
    Ok(collect_status(&state.config, &state.http).await)
}

#[tauri::command]
async fn server_start(state: State<'_, NativeState>) -> Result<ActionResult, String> {
    let _guard = state.lock.lock().await;
    Ok(start_server(&state.config).await)
}

#[tauri::command]
async fn server_stop(state: State<'_, NativeState>) -> Result<ActionResult, String> {
    let _guard = state.lock.lock().await;
    Ok(stop_server(&state.config).await)
}

#[tauri::command]
async fn client_start(state: State<'_, NativeState>) -> Result<ActionResult, String> {
    let _guard = state.lock.lock().await;
    Ok(start_client(&state.config).await)
}

#[tauri::command]
async fn client_stop(state: State<'_, NativeState>) -> Result<ActionResult, String> {
    let _guard = state.lock.lock().await;
    Ok(stop_client(&state.config).await)
}

#[tauri::command]
async fn mesh_start_consumer(state: State<'_, NativeState>) -> Result<ActionResult, String> {
    start_mesh(state, MeshRole::Consumer).await
}

#[tauri::command]
async fn mesh_start_provider(state: State<'_, NativeState>) -> Result<ActionResult, String> {
    start_mesh(state, MeshRole::GpuProvider).await
}

#[tauri::command]
async fn mesh_stop(state: State<'_, NativeState>) -> Result<ActionResult, String> {
    stop_mesh_capture(&state).await;
    let mut mesh = state.mesh.lock().await;
    let Some(mut runtime) = mesh.take() else {
        return Ok(ActionResult {
            ok: true,
            output: "Mesh no estaba arrancada.".into(),
        });
    };

    let _ = runtime.commands.send(MeshCommand::Shutdown).await;
    match timeout(Duration::from_secs(2), &mut runtime.task).await {
        Ok(Ok(Ok(()))) => Ok(ActionResult {
            ok: true,
            output: "Mesh parada.".into(),
        }),
        Ok(Ok(Err(error))) => Ok(ActionResult {
            ok: false,
            output: error.to_string(),
        }),
        Ok(Err(error)) => Ok(ActionResult {
            ok: false,
            output: error.to_string(),
        }),
        Err(_) => {
            runtime.task.abort();
            Ok(ActionResult {
                ok: true,
                output: "Mesh parada por timeout.".into(),
            })
        }
    }
}

#[tauri::command]
async fn mesh_capture_start(
    state: State<'_, NativeState>,
    direction: Direction,
) -> Result<ActionResult, String> {
    let mut capture = state.mesh_capture.lock().await;
    if capture.is_some() {
        return Ok(ActionResult {
            ok: true,
            output: "Captura Mesh ya esta activa.".into(),
        });
    }

    let commands = {
        let mesh = state.mesh.lock().await;
        let Some(runtime) = mesh.as_ref() else {
            return Ok(ActionResult {
                ok: false,
                output: "Arranca la Mesh como consumidor antes de capturar audio.".into(),
            });
        };
        if runtime.role != MeshRole::Consumer {
            return Ok(ActionResult {
                ok: false,
                output: "La captura Mesh solo se activa en modo consumidor.".into(),
            });
        }
        runtime.commands.clone()
    };

    let root = state.config.root.clone();
    let play_target = state.config.play_target.clone();
    let handle = tokio::runtime::Handle::current();
    let stop = Arc::new(AtomicBool::new(false));
    let task_stop = stop.clone();
    let capture_direction = direction.clone();
    let task = thread::spawn(move || {
        if let Err(error) = run_mesh_audio_capture(
            commands,
            root,
            play_target,
            capture_direction,
            handle,
            task_stop,
        ) {
            eprintln!("mesh audio capture failed: {error}");
        }
    });
    *capture = Some(MeshCaptureRuntime { stop, task });

    Ok(ActionResult {
        ok: true,
        output: format!(
            "Captura Mesh activa en direccion {:?}. Envia frases al mejor proveedor GPU disponible.",
            direction
        ),
    })
}

#[tauri::command]
async fn mesh_capture_stop(state: State<'_, NativeState>) -> Result<ActionResult, String> {
    stop_mesh_capture(&state).await;
    Ok(ActionResult {
        ok: true,
        output: "Captura Mesh parada.".into(),
    })
}

async fn stop_mesh_capture(state: &State<'_, NativeState>) {
    let mut capture = state.mesh_capture.lock().await;
    if let Some(runtime) = capture.take() {
        runtime.stop.store(true, Ordering::Relaxed);
        let _ = runtime.task.join();
    }
}

#[tauri::command]
async fn mesh_status(state: State<'_, NativeState>) -> Result<MeshStatus, String> {
    let mesh = state.mesh.lock().await;
    if mesh
        .as_ref()
        .map(|runtime| runtime.task.is_finished())
        .unwrap_or(false)
    {
        return Ok(MeshStatus {
            running: false,
            role: None,
        });
    }
    Ok(MeshStatus {
        running: mesh.is_some(),
        role: mesh.as_ref().map(|runtime| runtime.role),
    })
}

async fn start_mesh(state: State<'_, NativeState>, role: MeshRole) -> Result<ActionResult, String> {
    let mut slot = state.mesh.lock().await;
    if slot
        .as_ref()
        .map(|runtime| runtime.task.is_finished())
        .unwrap_or(false)
    {
        *slot = None;
    }
    if let Some(runtime) = slot.as_ref() {
        return Ok(ActionResult {
            ok: true,
            output: format!("Mesh ya esta arrancada en modo {:?}.", runtime.role),
        });
    }

    let (commands, rx) =
        LiveInterpreterMesh::<NvmlGpuTelemetry, DesktopMeshProcessor>::command_channel();
    let processor = match role {
        MeshRole::GpuProvider => {
            DesktopMeshProcessor::Ready(LiveMeshAudioProcessor::from_env().await.map_err(
                |error| format!("No se pudo inicializar el procesador Mesh GPU: {error}"),
            )?)
        }
        MeshRole::Consumer => DesktopMeshProcessor::Unavailable,
    };
    let mesh = LiveInterpreterMesh::new(
        MeshConfig {
            local_role: role,
            ..MeshConfig::default()
        },
        NvmlGpuTelemetry,
        processor,
    );
    let task = tokio::spawn(async move { mesh.run(rx).await });
    *slot = Some(MeshRuntime {
        role,
        commands,
        task,
    });

    Ok(ActionResult {
        ok: true,
        output: match role {
            MeshRole::GpuProvider => {
                "Mesh arrancada como proveedor GPU. Publicando salud por mDNS/Gossipsub.".into()
            }
            MeshRole::Consumer => {
                "Mesh arrancada como consumidor. Buscando proveedores GPU en la LAN.".into()
            }
        },
    })
}

impl LiveMeshAudioProcessor {
    async fn from_env() -> anyhow::Result<Self> {
        let config = Config::from_env()?;
        tokio::fs::create_dir_all(&config.data_dir).await?;
        let asr = Arc::new(AsrEngine::load(&config)?);
        let translator =
            Translator::from_env(config.ollama_url.clone(), config.ollama_model.clone())?;
        let tts = TtsEngine::new(&config).await?;
        Ok(Self {
            data_dir: config.data_dir,
            asr,
            translator,
            tts,
        })
    }
}

#[async_trait]
impl MeshAudioProcessor for DesktopMeshProcessor {
    async fn process(&self, chunk: AudioChunk) -> anyhow::Result<AudioTaskResult> {
        match self {
            DesktopMeshProcessor::Ready(processor) => processor.process(chunk).await,
            DesktopMeshProcessor::Unavailable => {
                anyhow::bail!("este nodo Mesh no tiene procesador GPU local")
            }
        }
    }
}

#[async_trait]
impl MeshAudioProcessor for LiveMeshAudioProcessor {
    async fn process(&self, chunk: AudioChunk) -> anyhow::Result<AudioTaskResult> {
        let input = self.mesh_input_path(&chunk);
        write_chunk_wav(&input, &chunk).await?;
        let segments = self
            .asr
            .transcribe_file(&input, chunk.direction.source_lang())
            .await?;
        let transcription = segments
            .iter()
            .map(|segment| segment.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let translation = self
            .translator
            .translate(&transcription, &chunk.direction)
            .await?;
        let audio_path = self
            .tts
            .synthesize(chunk.session_id, &translation, &chunk.direction)
            .await?;
        let (tts_sample_rate_hz, tts_output) = read_wav_as_f32(&audio_path)?;
        Ok(AudioTaskResult {
            session_id: chunk.session_id,
            sequence: chunk.sequence,
            transcription,
            translation,
            tts_sample_rate_hz,
            tts_output,
        })
    }
}

impl LiveMeshAudioProcessor {
    fn mesh_input_path(&self, chunk: &AudioChunk) -> PathBuf {
        self.data_dir
            .join("mesh")
            .join(format!("{}-{}.wav", chunk.session_id, chunk.sequence))
    }
}

async fn write_chunk_wav(path: &Path, chunk: &AudioChunk) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let samples = chunk.samples.clone();
    let sample_rate_hz = chunk.sample_rate_hz;
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: sample_rate_hz,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec)?;
        for sample in samples {
            writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
        }
        writer.finalize()?;
        Ok(())
    })
    .await?
}

fn read_wav_as_f32(path: &Path) -> anyhow::Result<(u32, Vec<f32>)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels.max(1));
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int if spec.bits_per_sample <= 16 => reader
            .samples::<i16>()
            .map(|sample| sample.map(|value| value as f32 / i16::MAX as f32))
            .collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .map(|sample| sample.map(|value| value as f32 / i32::MAX as f32))
            .collect::<Result<Vec<_>, _>>()?,
    };
    let mono = if channels == 1 {
        samples
    } else {
        samples
            .chunks(channels)
            .map(|frame| frame.iter().copied().sum::<f32>() / frame.len() as f32)
            .collect()
    };
    Ok((spec.sample_rate, mono))
}

#[tauri::command]
async fn get_voice_profile(state: State<'_, NativeState>) -> Result<VoiceProfile, String> {
    Ok(voice_profile(&state.config))
}

#[tauri::command]
async fn save_voice(
    state: State<'_, NativeState>,
    request: SaveVoiceProfileRequest,
) -> Result<ActionResult, String> {
    Ok(save_voice_profile(&state.config, request))
}

#[tauri::command]
fn start_voice_recording(state: State<'_, NativeState>) -> Result<ActionResult, String> {
    let mut recorder = state
        .voice_recorder
        .lock()
        .map_err(|error| error.to_string())?;
    if recorder.is_some() {
        return Ok(ActionResult {
            ok: true,
            output: "Ya se esta grabando.".into(),
        });
    }

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| "No se encontro microfono de entrada.".to_string())?;
    let config = device
        .default_input_config()
        .map_err(|error| error.to_string())?;
    let sample_rate = config.sample_rate().0;
    let channels = usize::from(config.channels());
    let samples = Arc::new(StdMutex::new(Vec::new()));
    let err_fn = |error| eprintln!("voice recording stream error: {error}");

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => {
            build_input_stream::<f32>(&device, &config.into(), channels, samples.clone(), err_fn)
        }
        cpal::SampleFormat::I16 => {
            build_input_stream::<i16>(&device, &config.into(), channels, samples.clone(), err_fn)
        }
        cpal::SampleFormat::U16 => {
            build_input_stream::<u16>(&device, &config.into(), channels, samples.clone(), err_fn)
        }
        other => Err(anyhow::anyhow!(
            "Formato de microfono no soportado: {other:?}"
        )),
    }
    .map_err(|error| error.to_string())?;

    stream.play().map_err(|error| error.to_string())?;
    *recorder = Some(VoiceRecorder {
        stream,
        samples,
        sample_rate,
    });

    Ok(ActionResult {
        ok: true,
        output: format!("Grabando desde {}.", device.name().unwrap_or_default()),
    })
}

#[tauri::command]
fn stop_voice_recording(state: State<'_, NativeState>) -> Result<ActionResult, String> {
    let recorder = state
        .voice_recorder
        .lock()
        .map_err(|error| error.to_string())?
        .take();
    let Some(recorder) = recorder else {
        return Ok(ActionResult {
            ok: false,
            output: "No habia grabacion activa.".into(),
        });
    };

    drop(recorder.stream);
    let samples = recorder
        .samples
        .lock()
        .map_err(|error| error.to_string())?
        .clone();
    if samples.len() < recorder.sample_rate as usize {
        return Ok(ActionResult {
            ok: false,
            output: "La muestra es demasiado corta. Graba al menos unos segundos.".into(),
        });
    }

    let wav = encode_wav_24k(&samples, recorder.sample_rate).map_err(|error| error.to_string())?;
    let audio_base64 = STANDARD.encode(wav);
    Ok(ActionResult {
        ok: true,
        output: audio_base64,
    })
}

#[tauri::command]
fn save_recorded_voice(
    state: State<'_, NativeState>,
    audio_base64: String,
    reference_text: String,
) -> Result<ActionResult, String> {
    Ok(save_voice_profile(
        &state.config,
        SaveVoiceProfileRequest {
            audio_base64,
            reference_text,
        },
    ))
}

fn build_input_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    samples: Arc<StdMutex<Vec<f32>>>,
    err_fn: impl FnMut(cpal::StreamError) + Send + 'static,
) -> anyhow::Result<cpal::Stream>
where
    T: cpal::Sample + cpal::SizedSample,
    f32: cpal::FromSample<T>,
{
    Ok(device.build_input_stream(
        config,
        move |data: &[T], _| {
            if let Ok(mut out) = samples.lock() {
                for frame in data.chunks(channels) {
                    let sum = frame
                        .iter()
                        .map(|sample| f32::from_sample(*sample))
                        .sum::<f32>();
                    out.push(sum / channels as f32);
                }
            }
        },
        err_fn,
        None,
    )?)
}

fn encode_wav_24k(samples: &[f32], input_rate: u32) -> anyhow::Result<Vec<u8>> {
    let resampled = resample_linear(samples, input_rate, 24_000);
    let mut cursor = std::io::Cursor::new(Vec::new());
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
        for sample in resampled {
            let clamped = sample.clamp(-1.0, 1.0);
            writer.write_sample((clamped * i16::MAX as f32) as i16)?;
        }
        writer.finalize()?;
    }
    Ok(cursor.into_inner())
}

fn resample_linear(samples: &[f32], input_rate: u32, output_rate: u32) -> Vec<f32> {
    if samples.is_empty() || input_rate == output_rate {
        return samples.to_vec();
    }
    let output_len = ((samples.len() as u64 * output_rate as u64) / input_rate as u64).max(1);
    (0..output_len)
        .map(|index| {
            let src = index as f64 * input_rate as f64 / output_rate as f64;
            let left = src.floor() as usize;
            let right = (left + 1).min(samples.len() - 1);
            let frac = (src - left as f64) as f32;
            samples[left] * (1.0 - frac) + samples[right] * frac
        })
        .collect()
}

fn run_mesh_audio_capture(
    commands: mpsc::Sender<MeshCommand>,
    root: PathBuf,
    play_target: String,
    direction: Direction,
    handle: tokio::runtime::Handle,
    stop: Arc<AtomicBool>,
) -> anyhow::Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("no default input device available"))?;
    let supported_config = device.default_input_config()?;
    let sample_rate = supported_config.sample_rate().0;
    let channels = usize::from(supported_config.channels());
    let stream_config = supported_config.clone().into();
    let (tx, rx) = std::sync::mpsc::channel::<Vec<f32>>();
    let stream = match supported_config.sample_format() {
        cpal::SampleFormat::F32 => {
            build_sender_stream::<f32>(&device, &stream_config, channels, tx.clone())
        }
        cpal::SampleFormat::I16 => {
            build_sender_stream::<i16>(&device, &stream_config, channels, tx.clone())
        }
        cpal::SampleFormat::U16 => {
            build_sender_stream::<u16>(&device, &stream_config, channels, tx.clone())
        }
        other => Err(anyhow::anyhow!(
            "Formato de microfono no soportado: {other:?}"
        )),
    }?;
    stream.play()?;

    let sequence = AtomicU64::new(1);
    let silence_samples = samples_for_ms(sample_rate, 800);
    let min_voice_samples = samples_for_ms(sample_rate, 700);
    let max_utterance_samples = samples_for_ms(sample_rate, 9_000);
    let pre_roll_samples = samples_for_ms(sample_rate, 180);
    let mut pre_roll = VecDeque::<f32>::with_capacity(pre_roll_samples);
    let mut utterance = Vec::<f32>::with_capacity(max_utterance_samples);
    let mut in_speech = false;
    let mut silence = 0usize;
    let mut voiced = 0usize;

    while !stop.load(Ordering::Relaxed) {
        let frame = match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(frame) => frame,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(error) => return Err(error.into()),
        };
        let frame_rms = rms(&frame);
        if !in_speech {
            push_pre_roll(&mut pre_roll, &frame, pre_roll_samples);
            if frame_rms >= 0.018 {
                in_speech = true;
                silence = 0;
                voiced = frame.len();
                utterance.extend(pre_roll.iter().copied());
                utterance.extend(frame);
            }
            continue;
        }

        utterance.extend(&frame);
        if frame_rms >= 0.018 {
            voiced += frame.len();
            silence = 0;
        } else {
            silence += frame.len();
        }

        let should_flush = (silence >= silence_samples && voiced >= min_voice_samples)
            || utterance.len() >= max_utterance_samples;
        if should_flush {
            let speech = trim_tail_silence(std::mem::take(&mut utterance), silence);
            pre_roll.clear();
            in_speech = false;
            silence = 0;
            voiced = 0;

            if speech.len() >= min_voice_samples {
                let seq = sequence.fetch_add(1, Ordering::Relaxed);
                let result = handle.block_on(send_mesh_utterance(
                    commands.clone(),
                    root.clone(),
                    play_target.clone(),
                    direction.clone(),
                    speech,
                    sample_rate,
                    seq,
                ));
                if let Err(error) = result {
                    eprintln!("mesh utterance failed: {error}");
                }
            }
        }
    }

    Ok(())
}

async fn send_mesh_utterance(
    commands: mpsc::Sender<MeshCommand>,
    root: PathBuf,
    play_target: String,
    direction: Direction,
    samples: Vec<f32>,
    sample_rate: u32,
    sequence: u64,
) -> anyhow::Result<()> {
    let (reply, rx) = tokio::sync::oneshot::channel();
    let chunk = AudioChunk {
        session_id: uuid::Uuid::new_v4(),
        sequence,
        sample_rate_hz: sample_rate,
        direction,
        samples,
    };
    commands
        .send(MeshCommand::SubmitAudio { chunk, reply })
        .await?;
    let result = rx.await??;
    let out_dir = root.join("data/mesh/results");
    tokio::fs::create_dir_all(&out_dir).await?;
    let out = out_dir.join(format!("{}-{}.wav", result.session_id, result.sequence));
    write_response_wav(&out, &result.tts_output, result.tts_sample_rate_hz).await?;
    play_mesh_audio(&out, &play_target).await?;
    Ok(())
}

async fn write_response_wav(path: &Path, samples: &[f32], sample_rate: u32) -> anyhow::Result<()> {
    let path = path.to_path_buf();
    let samples = samples.to_vec();
    tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec)?;
        for sample in samples {
            writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
        }
        writer.finalize()?;
        Ok(())
    })
    .await?
}

async fn play_mesh_audio(path: &Path, play_target: &str) -> anyhow::Result<()> {
    let mut command = tokio::process::Command::new("pw-play");
    if !play_target.trim().is_empty() {
        command.arg("--target").arg(play_target);
    }
    let status = command.arg(path).status().await?;
    if !status.success() {
        anyhow::bail!("pw-play failed for Mesh audio output");
    }
    Ok(())
}

fn build_sender_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    tx: std::sync::mpsc::Sender<Vec<f32>>,
) -> anyhow::Result<cpal::Stream>
where
    T: cpal::Sample + cpal::SizedSample,
    f32: cpal::FromSample<T>,
{
    Ok(device.build_input_stream(
        config,
        move |data: &[T], _| {
            let mut frame = Vec::with_capacity(data.len() / channels.max(1));
            for chunk in data.chunks(channels.max(1)) {
                let sum = chunk
                    .iter()
                    .map(|sample| f32::from_sample(*sample))
                    .sum::<f32>();
                frame.push(sum / chunk.len() as f32);
            }
            let _ = tx.send(frame);
        },
        |error| eprintln!("mesh capture stream error: {error}"),
        None,
    )?)
}

fn samples_for_ms(sample_rate: u32, millis: u64) -> usize {
    ((sample_rate as u64 * millis) / 1000).max(1) as usize
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32).sqrt()
}

fn push_pre_roll(pre_roll: &mut VecDeque<f32>, frame: &[f32], limit: usize) {
    for sample in frame {
        pre_roll.push_back(*sample);
        while pre_roll.len() > limit {
            pre_roll.pop_front();
        }
    }
}

fn trim_tail_silence(mut samples: Vec<f32>, silence: usize) -> Vec<f32> {
    let keep = samples.len().saturating_sub(silence / 2);
    samples.truncate(keep);
    samples
}

/// Background monitor: samples NVML every 2s and pushes a `gpu-telemetry` event to the WebView.
/// `vram_snapshot` offloads the blocking NVML call internally. Frontend: `listen('gpu-telemetry', ...)`.
async fn spawn_vram_telemetry(app: tauri::AppHandle, config: DesktopConfig) {
    let mut ticker = tokio::time::interval(Duration::from_secs(2));
    loop {
        ticker.tick().await;
        match vram_snapshot().await {
            Ok(snapshot) => {
                let telemetry = app_vram_telemetry(&config, &snapshot);
                if app.emit("gpu-telemetry", telemetry).is_err() {
                    break;
                }
            }
            Err(error) => {
                let _ = app.emit("gpu-telemetry-error", error.to_string());
            }
        }
    }
}

fn main() {
    fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("info".parse().expect("valid log directive")),
        )
        .init();

    let root = env::current_dir().expect("failed to read current directory");
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .expect("failed to build http client");
    let config = DesktopConfig::from_root(root);
    let telemetry_config = config.clone();
    let state = NativeState {
        config,
        http,
        lock: Arc::new(Mutex::new(())),
        mesh: Mutex::new(None),
        mesh_capture: Mutex::new(None),
        voice_recorder: StdMutex::new(None),
    };

    tauri::Builder::default()
        .manage(state)
        .setup(move |app| {
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(spawn_vram_telemetry(handle, telemetry_config));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            app_status,
            server_start,
            server_stop,
            client_start,
            client_stop,
            mesh_start_consumer,
            mesh_start_provider,
            mesh_stop,
            mesh_capture_start,
            mesh_capture_stop,
            mesh_status,
            get_voice_profile,
            save_voice,
            start_voice_recording,
            stop_voice_recording,
            save_recorded_voice
        ])
        .run(tauri::generate_context!())
        .expect("error while running Live Interpreter desktop application");
}
