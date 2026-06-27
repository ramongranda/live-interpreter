use anyhow::Context;
use base64::{Engine, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    future::Future,
    path::{Path, PathBuf},
    process::Stdio,
};
use tokio::process::Command;
use tokio::sync::mpsc;

#[derive(Clone, Debug)]
pub struct DesktopConfig {
    pub root: PathBuf,
    pub stack_bind: String,
    pub server_url: String,
    pub qwen_url: String,
    pub client_bind: String,
    pub client_url: String,
    pub play_target: String,
    pub min_server_vram_mb: u64,
}

impl DesktopConfig {
    pub fn from_root(root: PathBuf) -> Self {
        let client_bind =
            std::env::var("LI_CLIENT_BIND").unwrap_or_else(|_| "127.0.0.1:8790".into());
        Self {
            root,
            stack_bind: std::env::var("LI_STACK_BIND").unwrap_or_else(|_| "0.0.0.0:8787".into()),
            server_url: std::env::var("LI_SERVER_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8787".into()),
            qwen_url: std::env::var("LI_QWEN_TTS_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8020".into()),
            client_url: format!("http://{client_bind}"),
            client_bind,
            play_target: std::env::var("LI_CLIENT_PLAY_TARGET")
                .unwrap_or_else(|_| "live-interpreter-mic-sink".into()),
            min_server_vram_mb: std::env::var("LI_MIN_SERVER_VRAM_MB")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(8_000),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct AppStatus {
    pub server_running: bool,
    pub qwen_running: bool,
    pub client_running: bool,
    pub mic_bridge_running: bool,
    pub server_health: bool,
    pub qwen_health: bool,
    pub client_health: bool,
    pub gpu_summary: String,
    pub gpu_processes: Vec<GpuProcess>,
    pub gpu_ready: bool,
    pub gpu_gate: String,
    pub min_server_vram_mb: u64,
    pub server_url: String,
    pub client_url: String,
    pub role_hint: String,
    pub last_error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct GpuProcess {
    pub pid: String,
    pub name: String,
    pub memory: String,
}

#[derive(Debug, Serialize)]
pub struct ActionResult {
    pub ok: bool,
    pub output: String,
}

#[derive(Debug, Serialize)]
pub struct VoiceProfile {
    pub configured: bool,
    pub audio_path: String,
    pub text_path: String,
    pub reference_text: String,
}

#[derive(Debug, Deserialize)]
pub struct SaveVoiceProfileRequest {
    pub audio_base64: String,
    pub reference_text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamingBackpressure {
    pub raw_audio_frames: usize,
    pub transcript_fragments: usize,
    pub translated_texts: usize,
    pub generated_audio_buffers: usize,
}

impl Default for StreamingBackpressure {
    fn default() -> Self {
        Self {
            raw_audio_frames: 8,
            transcript_fragments: 8,
            translated_texts: 4,
            generated_audio_buffers: 4,
        }
    }
}

#[derive(Debug)]
pub struct StreamingChannels {
    pub raw_audio_tx: mpsc::Sender<RawAudioFrame>,
    pub raw_audio_rx: mpsc::Receiver<RawAudioFrame>,
    pub transcript_tx: mpsc::Sender<TranscriptFragment>,
    pub transcript_rx: mpsc::Receiver<TranscriptFragment>,
    pub translated_tx: mpsc::Sender<TranslatedText>,
    pub translated_rx: mpsc::Receiver<TranslatedText>,
    pub generated_audio_tx: mpsc::Sender<GeneratedAudio>,
    pub generated_audio_rx: mpsc::Receiver<GeneratedAudio>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RawAudioFrame {
    pub sequence: u64,
    pub sample_rate_hz: u32,
    pub samples: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscriptFragment {
    pub sequence: u64,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranslatedText {
    pub sequence: u64,
    pub source_text: String,
    pub translated_text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneratedAudio {
    pub sequence: u64,
    pub sample_rate_hz: u32,
    pub pcm_s16le: Vec<i16>,
}

pub trait AsrEngine: Send + Sync + 'static {
    fn transcribe(
        &self,
        frame: RawAudioFrame,
    ) -> impl Future<Output = anyhow::Result<Vec<TranscriptFragment>>> + Send;
}

pub trait TranslationEngine: Send + Sync + 'static {
    fn translate(
        &self,
        fragment: TranscriptFragment,
    ) -> impl Future<Output = anyhow::Result<TranslatedText>> + Send;
}

pub trait TtsEngine: Send + Sync + 'static {
    fn synthesize(
        &self,
        text: TranslatedText,
    ) -> impl Future<Output = anyhow::Result<GeneratedAudio>> + Send;
}

pub trait AudioSink: Send + Sync + 'static {
    fn play(&self, audio: GeneratedAudio) -> impl Future<Output = anyhow::Result<()>> + Send;
}

pub fn streaming_channels(config: StreamingBackpressure) -> StreamingChannels {
    let (raw_audio_tx, raw_audio_rx) = mpsc::channel(config.raw_audio_frames);
    let (transcript_tx, transcript_rx) = mpsc::channel(config.transcript_fragments);
    let (translated_tx, translated_rx) = mpsc::channel(config.translated_texts);
    let (generated_audio_tx, generated_audio_rx) = mpsc::channel(config.generated_audio_buffers);

    StreamingChannels {
        raw_audio_tx,
        raw_audio_rx,
        transcript_tx,
        transcript_rx,
        translated_tx,
        translated_rx,
        generated_audio_tx,
        generated_audio_rx,
    }
}

pub async fn run_asr_actor<E>(
    engine: E,
    mut raw_audio_rx: mpsc::Receiver<RawAudioFrame>,
    transcript_tx: mpsc::Sender<TranscriptFragment>,
) -> anyhow::Result<()>
where
    E: AsrEngine,
{
    while let Some(frame) = raw_audio_rx.recv().await {
        for fragment in engine.transcribe(frame).await? {
            if transcript_tx.send(fragment).await.is_err() {
                break;
            }
        }
    }
    Ok(())
}

pub async fn run_translation_actor<E>(
    engine: E,
    mut transcript_rx: mpsc::Receiver<TranscriptFragment>,
    translated_tx: mpsc::Sender<TranslatedText>,
) -> anyhow::Result<()>
where
    E: TranslationEngine,
{
    while let Some(fragment) = transcript_rx.recv().await {
        let translated = engine.translate(fragment).await?;
        if translated_tx.send(translated).await.is_err() {
            break;
        }
    }
    Ok(())
}

pub async fn run_tts_actor<E>(
    engine: E,
    mut translated_rx: mpsc::Receiver<TranslatedText>,
    generated_audio_tx: mpsc::Sender<GeneratedAudio>,
) -> anyhow::Result<()>
where
    E: TtsEngine,
{
    while let Some(text) = translated_rx.recv().await {
        let audio = engine.synthesize(text).await?;
        if generated_audio_tx.send(audio).await.is_err() {
            break;
        }
    }
    Ok(())
}

pub async fn run_audio_sink_actor<S>(
    sink: S,
    mut generated_audio_rx: mpsc::Receiver<GeneratedAudio>,
) -> anyhow::Result<()>
where
    S: AudioSink,
{
    while let Some(audio) = generated_audio_rx.recv().await {
        sink.play(audio).await?;
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuPreflight {
    pub ready: bool,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuInfo {
    pub name: String,
    pub total_mb: u64,
    pub free_mb: u64,
}

pub fn role_hint(server_running: bool, client_running: bool) -> &'static str {
    match (server_running, client_running) {
        (true, false) => "Modo servidor GPU",
        (false, true) => "Modo cliente de llamadas",
        (true, true) => "Servidor y cliente activos",
        (false, false) => "Selecciona servidor o cliente",
    }
}

pub fn pid_alive(path: &Path) -> bool {
    let Ok(pid) = std::fs::read_to_string(path) else {
        return false;
    };
    let Ok(pid) = pid.trim().parse::<u32>() else {
        return false;
    };
    PathBuf::from(format!("/proc/{pid}")).exists()
}

pub fn voice_dir(config: &DesktopConfig) -> PathBuf {
    config.root.join("data/voice")
}

pub fn voice_audio_path(config: &DesktopConfig) -> PathBuf {
    voice_dir(config).join("reference.wav")
}

pub fn voice_text_path(config: &DesktopConfig) -> PathBuf {
    voice_dir(config).join("reference.txt")
}

pub fn voice_profile(config: &DesktopConfig) -> VoiceProfile {
    let audio_path = voice_audio_path(config);
    let text_path = voice_text_path(config);
    let reference_text = fs::read_to_string(&text_path)
        .ok()
        .map(|text| text.trim().to_string())
        .unwrap_or_default();
    VoiceProfile {
        configured: audio_path.exists() && !reference_text.is_empty(),
        audio_path: audio_path.display().to_string(),
        text_path: text_path.display().to_string(),
        reference_text,
    }
}

pub fn save_voice_profile(
    config: &DesktopConfig,
    request: SaveVoiceProfileRequest,
) -> ActionResult {
    let reference_text = request.reference_text.trim();
    if reference_text.is_empty() {
        return ActionResult {
            ok: false,
            output: "Escribe la transcripcion exacta de la muestra de voz.".into(),
        };
    }

    let audio_bytes = match STANDARD.decode(request.audio_base64.trim()) {
        Ok(bytes) => bytes,
        Err(error) => {
            return ActionResult {
                ok: false,
                output: format!("Audio base64 invalido: {error}"),
            };
        }
    };

    if audio_bytes.len() < 1_000 {
        return ActionResult {
            ok: false,
            output: "La muestra de voz es demasiado corta o esta vacia.".into(),
        };
    }

    let dir = voice_dir(config);
    if let Err(error) = fs::create_dir_all(&dir) {
        return ActionResult {
            ok: false,
            output: error.to_string(),
        };
    }

    let audio_path = voice_audio_path(config);
    let text_path = voice_text_path(config);
    if let Err(error) = fs::write(&audio_path, audio_bytes) {
        return ActionResult {
            ok: false,
            output: error.to_string(),
        };
    }
    if let Err(error) = fs::write(&text_path, reference_text) {
        return ActionResult {
            ok: false,
            output: error.to_string(),
        };
    }

    ActionResult {
        ok: true,
        output: format!(
            "Voz guardada. Reinicia el servidor GPU para usar {}.",
            audio_path.display()
        ),
    }
}

pub fn parse_gpu_preflight(stdout: &str, min_vram_mb: u64) -> GpuPreflight {
    let Some(gpu) = best_gpu(stdout) else {
        return GpuPreflight {
            ready: false,
            message: "Servidor GPU bloqueado: nvidia-smi no devolvio GPUs".into(),
        };
    };

    if gpu.total_mb < min_vram_mb {
        return GpuPreflight {
            ready: false,
            message: format!(
                "Servidor GPU bloqueado: {} tiene {} MiB VRAM, minimo requerido {} MiB",
                gpu.name, gpu.total_mb, min_vram_mb
            ),
        };
    }

    GpuPreflight {
        ready: true,
        message: format!(
            "GPU lista: {}, {} MiB VRAM total, {} MiB libres, minimo {} MiB",
            gpu.name, gpu.total_mb, gpu.free_mb, min_vram_mb
        ),
    }
}

pub fn best_gpu(stdout: &str) -> Option<GpuInfo> {
    stdout
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| {
            let mut parts = line.split(',').map(str::trim);
            Some(GpuInfo {
                name: parts.next()?.into(),
                total_mb: parts.next()?.parse().ok()?,
                free_mb: parts.next()?.parse().ok()?,
            })
        })
        .max_by_key(|gpu| gpu.total_mb)
}

pub async fn gpu_preflight(min_vram_mb: u64) -> GpuPreflight {
    let output = match Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .await
    {
        Ok(output) => output,
        Err(error) => {
            return GpuPreflight {
                ready: false,
                message: format!("Servidor GPU bloqueado: nvidia-smi no esta disponible ({error})"),
            };
        }
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return GpuPreflight {
            ready: false,
            message: if stderr.is_empty() {
                "Servidor GPU bloqueado: no se detecta una GPU NVIDIA utilizable".into()
            } else {
                format!("Servidor GPU bloqueado: {stderr}")
            },
        };
    }

    parse_gpu_preflight(&String::from_utf8_lossy(&output.stdout), min_vram_mb)
}

pub async fn gpu_status() -> anyhow::Result<(Vec<GpuProcess>, String, Option<String>)> {
    let summary = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader",
        ])
        .output()
        .await
        .context("failed to run nvidia-smi summary")?;
    let processes = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ])
        .output()
        .await
        .context("failed to run nvidia-smi processes")?;

    let summary_text = String::from_utf8_lossy(&summary.stdout).trim().to_string();
    let processes_text = String::from_utf8_lossy(&processes.stdout);
    let gpu_processes = processes_text
        .lines()
        .filter_map(|line| {
            let mut parts = line.splitn(3, ',').map(str::trim);
            Some(GpuProcess {
                pid: parts.next()?.into(),
                name: parts.next()?.into(),
                memory: parts.next()?.into(),
            })
        })
        .collect();

    Ok((gpu_processes, summary_text, None))
}

pub async fn run_script(
    config: &DesktopConfig,
    script: &str,
    envs: &[(&str, &str)],
) -> ActionResult {
    let mut command = Command::new("bash");
    command
        .arg(config.root.join("scripts").join(script))
        .current_dir(&config.root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (key, value) in envs {
        command.env(key, value);
    }

    match command.output().await {
        Ok(output) => {
            let mut text = String::new();
            text.push_str(&String::from_utf8_lossy(&output.stdout));
            text.push_str(&String::from_utf8_lossy(&output.stderr));
            ActionResult {
                ok: output.status.success(),
                output: text,
            }
        }
        Err(error) => ActionResult {
            ok: false,
            output: error.to_string(),
        },
    }
}

pub async fn collect_status(config: &DesktopConfig, http: &reqwest::Client) -> AppStatus {
    let server_health = health(http, &format!("{}/health", config.server_url)).await;
    let qwen_health = health(http, &format!("{}/health", config.qwen_url)).await;
    let client_health = health(http, &format!("{}/api/state", config.client_url)).await;
    let gpu = gpu_preflight(config.min_server_vram_mb).await;
    let (gpu_processes, gpu_summary, gpu_error) = gpu_status()
        .await
        .unwrap_or_else(|error| (Vec::new(), String::new(), Some(error.to_string())));
    let server_running = pid_alive(&config.root.join("data/logs/live-interpreter.pid"));
    let client_running = pid_alive(&config.root.join("data/logs/live-interpreter-client.pid"));

    AppStatus {
        server_running,
        qwen_running: pid_alive(&config.root.join("data/logs/qwen3-tts.pid")),
        client_running,
        mic_bridge_running: pid_alive(&config.root.join("data/logs/live-interpreter-mic.pid")),
        server_health,
        qwen_health,
        client_health,
        gpu_summary,
        gpu_processes,
        gpu_ready: gpu.ready,
        gpu_gate: gpu.message,
        min_server_vram_mb: config.min_server_vram_mb,
        server_url: config.server_url.clone(),
        client_url: config.client_url.clone(),
        role_hint: role_hint(server_running, client_running).into(),
        last_error: gpu_error,
    }
}

pub async fn start_server(config: &DesktopConfig) -> ActionResult {
    let gpu = crate::vram::gpu_preflight_realtime(config.min_server_vram_mb).await;
    if !gpu.ready {
        return ActionResult {
            ok: false,
            output: gpu.message,
        };
    }

    let profile = voice_profile(config);
    let mut owned_envs = vec![("LI_BIND".to_string(), config.stack_bind.clone())];
    if profile.configured {
        owned_envs.extend([
            (
                "LI_QWEN_MODEL".to_string(),
                "Qwen3-TTS-12Hz-0.6B-Base".to_string(),
            ),
            (
                "LI_QWEN_TTS_MODEL".to_string(),
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base".to_string(),
            ),
            ("LI_VOICE_REF".to_string(), profile.audio_path),
            ("LI_VOICE_REF_TEXT".to_string(), profile.reference_text),
        ]);
    }
    let envs = owned_envs
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect::<Vec<_>>();

    run_script(config, "start-local-stack.sh", &envs).await
}

pub async fn stop_server(config: &DesktopConfig) -> ActionResult {
    run_script(config, "stop-local-stack.sh", &[]).await
}

pub async fn start_client(config: &DesktopConfig) -> ActionResult {
    if pid_alive(&config.root.join("data/logs/live-interpreter-client.pid")) {
        return ActionResult {
            ok: true,
            output: "Cliente ya esta arrancado".into(),
        };
    }

    let bin = config.root.join("target/release/live-interpreter-client");
    let log = config.root.join("data/logs/live-interpreter-client.log");
    let pid = config.root.join("data/logs/live-interpreter-client.pid");
    if let Err(error) = tokio::fs::create_dir_all(config.root.join("data/logs")).await {
        return ActionResult {
            ok: false,
            output: error.to_string(),
        };
    }

    let log_file = match std::fs::File::create(&log) {
        Ok(file) => file,
        Err(error) => {
            return ActionResult {
                ok: false,
                output: error.to_string(),
            };
        }
    };
    let log_file_err = match log_file.try_clone() {
        Ok(file) => file,
        Err(error) => {
            return ActionResult {
                ok: false,
                output: error.to_string(),
            };
        }
    };

    match Command::new(bin)
        .current_dir(&config.root)
        .env("LI_SERVER_URL", &config.server_url)
        .env("LI_CLIENT_BIND", &config.client_bind)
        .env("LI_CLIENT_PLAY_TARGET", &config.play_target)
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err))
        .spawn()
    {
        Ok(child) => {
            let child_pid = child.id().unwrap_or_default();
            let _ = tokio::fs::write(pid, child_pid.to_string()).await;
            ActionResult {
                ok: true,
                output: format!("Cliente arrancado en {}", config.client_url),
            }
        }
        Err(error) => ActionResult {
            ok: false,
            output: error.to_string(),
        },
    }
}

pub async fn stop_client(config: &DesktopConfig) -> ActionResult {
    let pid_path = config.root.join("data/logs/live-interpreter-client.pid");
    let Ok(pid) = tokio::fs::read_to_string(&pid_path).await else {
        return ActionResult {
            ok: true,
            output: "Cliente no estaba arrancado".into(),
        };
    };

    let output = Command::new("kill")
        .arg(pid.trim())
        .output()
        .await
        .map(|output| {
            let mut text = String::from_utf8_lossy(&output.stdout).to_string();
            text.push_str(&String::from_utf8_lossy(&output.stderr));
            ActionResult {
                ok: output.status.success(),
                output: if text.trim().is_empty() {
                    "Cliente parado".into()
                } else {
                    text
                },
            }
        })
        .unwrap_or_else(|error| ActionResult {
            ok: false,
            output: error.to_string(),
        });
    let _ = tokio::fs::remove_file(pid_path).await;
    output
}

async fn health(http: &reqwest::Client, url: &str) -> bool {
    http.get(url)
        .send()
        .await
        .map(|response| response.status().is_success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn role_hint_covers_all_modes() {
        assert_eq!(role_hint(false, false), "Selecciona servidor o cliente");
        assert_eq!(role_hint(true, false), "Modo servidor GPU");
        assert_eq!(role_hint(false, true), "Modo cliente de llamadas");
        assert_eq!(role_hint(true, true), "Servidor y cliente activos");
    }

    #[test]
    fn best_gpu_selects_largest_vram_card() {
        let gpu = best_gpu("Small GPU, 4096, 1024\nLarge GPU, 16384, 12000\n").unwrap();
        assert_eq!(gpu.name, "Large GPU");
        assert_eq!(gpu.total_mb, 16384);
        assert_eq!(gpu.free_mb, 12000);
    }

    #[test]
    fn preflight_allows_sufficient_gpu() {
        let preflight = parse_gpu_preflight("RTX 5060 Ti, 16311, 7281\n", 8000);
        assert!(preflight.ready);
        assert!(preflight.message.contains("GPU lista"));
    }

    #[test]
    fn preflight_blocks_insufficient_gpu() {
        let preflight = parse_gpu_preflight("Small GPU, 4096, 1024\n", 8000);
        assert!(!preflight.ready);
        assert!(preflight.message.contains("bloqueado"));
        assert!(preflight.message.contains("4096 MiB"));
    }

    #[test]
    fn preflight_blocks_empty_gpu_list() {
        let preflight = parse_gpu_preflight("", 8000);
        assert!(!preflight.ready);
        assert_eq!(
            preflight.message,
            "Servidor GPU bloqueado: nvidia-smi no devolvio GPUs"
        );
    }

    #[test]
    fn voice_profile_save_and_read_roundtrip() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("live-interpreter-test-{unique}"));
        let config = DesktopConfig::from_root(root.clone());
        let result = save_voice_profile(
            &config,
            SaveVoiceProfileRequest {
                audio_base64: STANDARD.encode(vec![1u8; 2048]),
                reference_text: "hola mundo".into(),
            },
        );
        assert!(result.ok);
        let profile = voice_profile(&config);
        assert!(profile.configured);
        assert_eq!(profile.reference_text, "hola mundo");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn streaming_channels_are_bounded() {
        let channels = streaming_channels(StreamingBackpressure {
            raw_audio_frames: 1,
            transcript_fragments: 1,
            translated_texts: 1,
            generated_audio_buffers: 1,
        });

        channels
            .raw_audio_tx
            .try_send(RawAudioFrame {
                sequence: 1,
                sample_rate_hz: 16_000,
                samples: vec![0.0; 160],
            })
            .unwrap();

        assert!(
            channels
                .raw_audio_tx
                .try_send(RawAudioFrame {
                    sequence: 2,
                    sample_rate_hz: 16_000,
                    samples: vec![0.0; 160],
                })
                .is_err()
        );
    }

    #[tokio::test]
    async fn actors_move_audio_through_the_streaming_pipeline() {
        let channels = streaming_channels(StreamingBackpressure {
            raw_audio_frames: 2,
            transcript_fragments: 2,
            translated_texts: 2,
            generated_audio_buffers: 2,
        });
        let played = Arc::new(AtomicUsize::new(0));

        let asr = tokio::spawn(run_asr_actor(
            MockAsr,
            channels.raw_audio_rx,
            channels.transcript_tx,
        ));
        let translator = tokio::spawn(run_translation_actor(
            MockTranslator,
            channels.transcript_rx,
            channels.translated_tx,
        ));
        let tts = tokio::spawn(run_tts_actor(
            MockTts,
            channels.translated_rx,
            channels.generated_audio_tx,
        ));
        let sink = tokio::spawn(run_audio_sink_actor(
            CountingSink {
                played: played.clone(),
            },
            channels.generated_audio_rx,
        ));

        channels
            .raw_audio_tx
            .send(RawAudioFrame {
                sequence: 7,
                sample_rate_hz: 16_000,
                samples: vec![0.25; 320],
            })
            .await
            .unwrap();
        drop(channels.raw_audio_tx);

        asr.await.unwrap().unwrap();
        translator.await.unwrap().unwrap();
        tts.await.unwrap().unwrap();
        sink.await.unwrap().unwrap();

        assert_eq!(played.load(Ordering::SeqCst), 1);
    }

    struct MockAsr;

    impl AsrEngine for MockAsr {
        async fn transcribe(
            &self,
            frame: RawAudioFrame,
        ) -> anyhow::Result<Vec<TranscriptFragment>> {
            Ok(vec![TranscriptFragment {
                sequence: frame.sequence,
                text: "hola".into(),
            }])
        }
    }

    struct MockTranslator;

    impl TranslationEngine for MockTranslator {
        async fn translate(&self, fragment: TranscriptFragment) -> anyhow::Result<TranslatedText> {
            Ok(TranslatedText {
                sequence: fragment.sequence,
                source_text: fragment.text,
                translated_text: "hello".into(),
            })
        }
    }

    struct MockTts;

    impl TtsEngine for MockTts {
        async fn synthesize(&self, text: TranslatedText) -> anyhow::Result<GeneratedAudio> {
            Ok(GeneratedAudio {
                sequence: text.sequence,
                sample_rate_hz: 24_000,
                pcm_s16le: vec![0; 480],
            })
        }
    }

    struct CountingSink {
        played: Arc<AtomicUsize>,
    }

    impl AudioSink for CountingSink {
        async fn play(&self, audio: GeneratedAudio) -> anyhow::Result<()> {
            assert_eq!(audio.sequence, 7);
            assert_eq!(audio.sample_rate_hz, 24_000);
            assert_eq!(audio.pcm_s16le.len(), 480);
            self.played.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }
}
