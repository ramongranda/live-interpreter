use anyhow::{Context, bail};
use axum::{
    Json, Router,
    extract::State,
    response::{Html, IntoResponse},
    routing::{get, post},
};
use cpal::{
    SampleFormat, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use futures_util::{SinkExt, StreamExt};
use hound::{SampleFormat as WavSampleFormat, WavSpec, WavWriter};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::{
    env,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::process::Command;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing_subscriber::{EnvFilter, fmt};

#[derive(Clone)]
struct App {
    config: ClientConfig,
    controls: Arc<Controls>,
    state: Arc<Mutex<ClientState>>,
}

#[derive(Clone, Debug)]
struct ClientConfig {
    server_url: String,
    bind: SocketAddr,
    direction: String,
    data_dir: PathBuf,
    play_cmd: String,
    play_target: Option<String>,
    auth_token: Option<String>,
    vad_threshold: f32,
    silence_ms: u64,
    min_voice_ms: u64,
    max_utterance_ms: u64,
    pre_roll_ms: u64,
}

impl ClientConfig {
    fn from_env() -> anyhow::Result<Self> {
        let server_url = env::var("LI_SERVER_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:8787".into())
            .trim_end_matches('/')
            .to_string();
        let bind = env::var("LI_CLIENT_BIND")
            .unwrap_or_else(|_| "127.0.0.1:8790".into())
            .parse()?;
        let direction = env::var("LI_CLIENT_DIRECTION").unwrap_or_else(|_| "es_to_en".into());
        let data_dir =
            PathBuf::from(env::var("LI_CLIENT_DATA_DIR").unwrap_or_else(|_| "data/client".into()));
        let play_cmd = env::var("LI_CLIENT_PLAY_CMD").unwrap_or_else(|_| "pw-play".into());
        let play_target = env::var("LI_CLIENT_PLAY_TARGET")
            .ok()
            .or_else(|| Some("live-interpreter-mic-sink".into()));
        let auth_token = env::var("LI_CLIENT_AUTH_TOKEN")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .or_else(|| {
                env::var("LI_AUTH_TOKEN")
                    .ok()
                    .filter(|value| !value.trim().is_empty())
            });
        let vad_threshold = env::var("LI_CLIENT_VAD_THRESHOLD")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0.012);
        let silence_ms = env::var("LI_CLIENT_SILENCE_MS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(800);
        let min_voice_ms = env::var("LI_CLIENT_MIN_VOICE_MS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(280);
        let max_utterance_ms = env::var("LI_CLIENT_MAX_UTTERANCE_MS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(8500);
        let pre_roll_ms = env::var("LI_CLIENT_PRE_ROLL_MS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(240);

        Ok(Self {
            server_url,
            bind,
            direction,
            data_dir,
            play_cmd,
            play_target,
            auth_token,
            vad_threshold,
            silence_ms,
            min_voice_ms,
            max_utterance_ms,
            pre_roll_ms,
        })
    }
}

#[derive(Default)]
struct Controls {
    paused: AtomicBool,
    input_muted: AtomicBool,
    output_muted: AtomicBool,
    stop: AtomicBool,
    direction_es_to_en: AtomicBool,
}

#[derive(Clone, Debug, Serialize)]
struct ClientState {
    server_url: String,
    status: String,
    direction: String,
    paused: bool,
    input_muted: bool,
    output_muted: bool,
    chunks_sent: u64,
    last_latency_ms: u128,
    current_rms: f32,
    last_transcript: String,
    last_translation: String,
    last_audio_url: Option<String>,
    last_error: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Direction {
    EsToEn,
    EnToEs,
}

#[derive(Clone, Debug, Serialize)]
struct StreamStart {
    direction: Direction,
    synthesize: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
#[allow(dead_code)]
enum StreamEvent {
    Ready,
    Listening,
    Processing {
        id: String,
    },
    TranscriptFinal {
        id: String,
        text: String,
    },
    TranslationFinal {
        id: String,
        text: String,
    },
    AudioStart {
        id: String,
        content_type: String,
        bytes: usize,
    },
    Done {
        id: String,
        latency_ms: u128,
    },
    Error {
        message: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let config = ClientConfig::from_env()?;
    tokio::fs::create_dir_all(&config.data_dir).await?;

    let controls = Arc::new(Controls {
        direction_es_to_en: AtomicBool::new(config.direction == "es_to_en"),
        ..Controls::default()
    });
    let state = Arc::new(Mutex::new(ClientState {
        server_url: config.server_url.clone(),
        status: "starting".into(),
        direction: config.direction.clone(),
        paused: false,
        input_muted: false,
        output_muted: false,
        chunks_sent: 0,
        last_latency_ms: 0,
        current_rms: 0.0,
        last_transcript: String::new(),
        last_translation: String::new(),
        last_audio_url: None,
        last_error: None,
    }));
    let app = App {
        config,
        controls,
        state,
    };

    let audio_app = app.clone();
    std::thread::spawn(move || {
        if let Err(error) = run_audio_capture(audio_app) {
            tracing::error!("audio capture failed: {error:?}");
        }
    });

    let router = Router::new()
        .route("/", get(index))
        .route("/api/state", get(api_state))
        .route("/api/toggle-pause", post(toggle_pause))
        .route("/api/toggle-input", post(toggle_input))
        .route("/api/toggle-output", post(toggle_output))
        .route("/api/swap-direction", post(swap_direction))
        .with_state(app.clone());

    tracing::info!("Live Interpreter client UI: http://{}", app.config.bind);
    let listener = tokio::net::TcpListener::bind(app.config.bind).await?;
    axum::serve(listener, router).await?;
    app.controls.stop.store(true, Ordering::Relaxed);
    Ok(())
}

fn run_audio_capture(app: App) -> anyhow::Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("no default input device available")?;
    let supported_config = device.default_input_config()?;
    let sample_rate = supported_config.sample_rate().0;
    let channels = supported_config.channels() as usize;
    let stream_config: StreamConfig = supported_config.clone().into();
    let (tx, rx) = std::sync::mpsc::channel::<Vec<f32>>();

    let stream = match supported_config.sample_format() {
        SampleFormat::F32 => build_f32_stream(&device, &stream_config, tx, channels)?,
        SampleFormat::I16 => build_i16_stream(&device, &stream_config, tx, channels)?,
        SampleFormat::U16 => build_u16_stream(&device, &stream_config, tx, channels)?,
        other => bail!("unsupported input sample format: {other:?}"),
    };
    stream.play()?;

    update_status(&app, "listening");
    let rt = tokio::runtime::Runtime::new()?;
    let silence_samples = samples_for_ms(sample_rate, app.config.silence_ms);
    let min_voice_samples = samples_for_ms(sample_rate, app.config.min_voice_ms);
    let max_utterance_samples = samples_for_ms(sample_rate, app.config.max_utterance_ms);
    let pre_roll_samples = samples_for_ms(sample_rate, app.config.pre_roll_ms);
    let mut pre_roll = VecDeque::<f32>::with_capacity(pre_roll_samples);
    let mut utterance = Vec::<f32>::with_capacity(max_utterance_samples);
    let mut in_speech = false;
    let mut silence = 0usize;
    let mut voiced = 0usize;

    while !app.controls.stop.load(Ordering::Relaxed) {
        let frame = match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(frame) => frame,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(error) => return Err(error.into()),
        };

        refresh_control_state(&app);
        if app.controls.paused.load(Ordering::Relaxed)
            || app.controls.input_muted.load(Ordering::Relaxed)
        {
            pre_roll.clear();
            utterance.clear();
            in_speech = false;
            silence = 0;
            voiced = 0;
            continue;
        }

        let frame_rms = rms(&frame);
        set_rms(&app, frame_rms);
        if !in_speech {
            push_pre_roll(&mut pre_roll, &frame, pre_roll_samples);
            if frame_rms >= app.config.vad_threshold {
                in_speech = true;
                silence = 0;
                voiced = frame.len();
                utterance.extend(pre_roll.iter().copied());
                utterance.extend(frame);
                update_status(&app, "speaking");
            }
            continue;
        }

        utterance.extend(&frame);
        if frame_rms >= app.config.vad_threshold {
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
                if let Err(error) = rt.block_on(process_utterance(app.clone(), speech, sample_rate))
                {
                    set_error(&app, error);
                }
            }
            update_status(&app, "listening");
        }
    }

    Ok(())
}

fn build_f32_stream(
    device: &cpal::Device,
    config: &StreamConfig,
    tx: std::sync::mpsc::Sender<Vec<f32>>,
    channels: usize,
) -> anyhow::Result<cpal::Stream> {
    let stream = device.build_input_stream(
        config,
        move |data: &[f32], _| {
            let _ = tx.send(mix_to_mono(data, channels, |sample| sample));
        },
        log_stream_error,
        None,
    )?;
    Ok(stream)
}

fn build_i16_stream(
    device: &cpal::Device,
    config: &StreamConfig,
    tx: std::sync::mpsc::Sender<Vec<f32>>,
    channels: usize,
) -> anyhow::Result<cpal::Stream> {
    let stream = device.build_input_stream(
        config,
        move |data: &[i16], _| {
            let _ = tx.send(mix_to_mono(data, channels, |sample| {
                sample as f32 / i16::MAX as f32
            }));
        },
        log_stream_error,
        None,
    )?;
    Ok(stream)
}

fn build_u16_stream(
    device: &cpal::Device,
    config: &StreamConfig,
    tx: std::sync::mpsc::Sender<Vec<f32>>,
    channels: usize,
) -> anyhow::Result<cpal::Stream> {
    let stream = device.build_input_stream(
        config,
        move |data: &[u16], _| {
            let _ = tx.send(mix_to_mono(data, channels, |sample| {
                (sample as f32 / u16::MAX as f32) * 2.0 - 1.0
            }));
        },
        log_stream_error,
        None,
    )?;
    Ok(stream)
}

fn mix_to_mono<T>(data: &[T], channels: usize, convert: impl Fn(T) -> f32) -> Vec<f32>
where
    T: Copy,
{
    data.chunks(channels)
        .map(|frame| frame.iter().copied().map(&convert).sum::<f32>() / frame.len() as f32)
        .collect()
}

fn log_stream_error(error: cpal::StreamError) {
    tracing::error!("audio stream error: {error}");
}

async fn process_utterance(app: App, samples: Vec<f32>, sample_rate: u32) -> anyhow::Result<()> {
    set_status(&app, "streaming").await;
    let started = Instant::now();
    let direction = current_direction(&app);
    let wav_path = app.config.data_dir.join("chunk.wav");
    write_wav(&wav_path, &samples, sample_rate)?;

    let bytes = tokio::fs::read(&wav_path).await?;
    let url = websocket_url(&app.config.server_url, app.config.auth_token.as_deref());
    let (mut socket, _) = connect_async(&url)
        .await
        .with_context(|| format!("failed to connect websocket {url}"))?;
    let start = StreamStart {
        direction: parse_direction(&direction),
        synthesize: true,
    };
    socket
        .send(Message::Text(serde_json::to_string(&start)?.into()))
        .await?;
    socket.send(Message::Binary(bytes.into())).await?;

    let mut transcript = String::new();
    let mut translation = String::new();
    let mut audio_url = None;
    let out_path = app.config.data_dir.join("translated.wav");

    while let Some(message) = socket.next().await {
        match message? {
            Message::Text(text) => {
                let event: StreamEvent = serde_json::from_str(&text)?;
                match event {
                    StreamEvent::Ready | StreamEvent::Listening => {}
                    StreamEvent::Processing { .. } => {
                        set_status(&app, "processing").await;
                    }
                    StreamEvent::TranscriptFinal { text, .. } => {
                        transcript = text;
                        update_last_text(&app, &transcript, &translation, audio_url.clone());
                    }
                    StreamEvent::TranslationFinal { text, .. } => {
                        translation = text;
                        update_last_text(&app, &transcript, &translation, audio_url.clone());
                    }
                    StreamEvent::AudioStart { .. } => {
                        set_status(&app, "receiving_audio").await;
                    }
                    StreamEvent::Done { .. } => break,
                    StreamEvent::Error { message } => bail!(message),
                }
            }
            Message::Binary(audio) => {
                audio_url = Some("websocket://last-audio".to_string());
                tokio::fs::write(&out_path, audio).await?;
                if !app.controls.output_muted.load(Ordering::Relaxed) {
                    play_audio(&app, &out_path).await?;
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    let elapsed = started.elapsed().as_millis();
    let mut state = app.state.lock().expect("client state mutex poisoned");
    state.status = "listening".into();
    state.direction = direction;
    state.chunks_sent += 1;
    state.last_latency_ms = elapsed;
    state.last_transcript = transcript;
    state.last_translation = translation;
    state.last_audio_url = audio_url;
    state.last_error = None;
    Ok(())
}

fn websocket_url(server_url: &str, auth_token: Option<&str>) -> String {
    let base = server_url.trim_end_matches('/');
    let mut url = if let Some(rest) = base.strip_prefix("https://") {
        format!("wss://{rest}/v1/stream/meeting")
    } else if let Some(rest) = base.strip_prefix("http://") {
        format!("ws://{rest}/v1/stream/meeting")
    } else {
        format!("ws://{base}/v1/stream/meeting")
    };
    if let Some(token) = auth_token {
        url.push_str("?token=");
        url.push_str(token);
    }
    url
}

fn parse_direction(value: &str) -> Direction {
    match value {
        "en_to_es" => Direction::EnToEs,
        _ => Direction::EsToEn,
    }
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

fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> anyhow::Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: WavSampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)?;
    for sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        writer.write_sample((clamped * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

async fn play_audio(app: &App, path: &Path) -> anyhow::Result<()> {
    if app.config.play_cmd.trim().is_empty() {
        return Ok(());
    }

    let mut command = Command::new(&app.config.play_cmd);
    if let Some(target) = app.config.play_target.as_deref() {
        command.arg("--target").arg(target);
    }
    command.arg(path);
    let status = command.status().await?;
    if !status.success() {
        bail!("play command failed: {}", app.config.play_cmd);
    }
    Ok(())
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn api_state(State(app): State<App>) -> Json<ClientState> {
    refresh_control_state(&app);
    Json(
        app.state
            .lock()
            .expect("client state mutex poisoned")
            .clone(),
    )
}

async fn toggle_pause(State(app): State<App>) -> impl IntoResponse {
    toggle(&app.controls.paused);
    refresh_control_state(&app);
    Json(serde_json::json!({"ok": true}))
}

async fn toggle_input(State(app): State<App>) -> impl IntoResponse {
    toggle(&app.controls.input_muted);
    refresh_control_state(&app);
    Json(serde_json::json!({"ok": true}))
}

async fn toggle_output(State(app): State<App>) -> impl IntoResponse {
    toggle(&app.controls.output_muted);
    refresh_control_state(&app);
    Json(serde_json::json!({"ok": true}))
}

async fn swap_direction(State(app): State<App>) -> impl IntoResponse {
    toggle(&app.controls.direction_es_to_en);
    refresh_control_state(&app);
    Json(serde_json::json!({"ok": true}))
}

fn toggle(value: &AtomicBool) {
    let current = value.load(Ordering::Relaxed);
    value.store(!current, Ordering::Relaxed);
}

fn current_direction(app: &App) -> String {
    if app.controls.direction_es_to_en.load(Ordering::Relaxed) {
        "es_to_en".into()
    } else {
        "en_to_es".into()
    }
}

fn refresh_control_state(app: &App) {
    let paused = app.controls.paused.load(Ordering::Relaxed);
    let input_muted = app.controls.input_muted.load(Ordering::Relaxed);
    let output_muted = app.controls.output_muted.load(Ordering::Relaxed);
    let direction = current_direction(app);
    let mut state = app.state.lock().expect("client state mutex poisoned");
    state.paused = paused;
    state.input_muted = input_muted;
    state.output_muted = output_muted;
    state.direction = direction;
}

fn set_rms(app: &App, rms: f32) {
    app.state
        .lock()
        .expect("client state mutex poisoned")
        .current_rms = rms;
}

fn update_last_text(app: &App, transcript: &str, translation: &str, audio_url: Option<String>) {
    let mut state = app.state.lock().expect("client state mutex poisoned");
    state.last_transcript = transcript.into();
    state.last_translation = translation.into();
    state.last_audio_url = audio_url;
}

fn update_status(app: &App, status: &str) {
    app.state
        .lock()
        .expect("client state mutex poisoned")
        .status = status.into();
}

fn set_error(app: &App, error: anyhow::Error) {
    let mut state = app.state.lock().expect("client state mutex poisoned");
    state.status = "error".into();
    state.last_error = Some(error.to_string());
}

async fn set_status(app: &App, status: &str) {
    app.state
        .lock()
        .expect("client state mutex poisoned")
        .status = status.into();
}

const INDEX_HTML: &str = r#"<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Live Interpreter Client</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; background:#101214; color:#eef1f4; }
    body { margin:0; }
    main { max-width: 1120px; margin: 0 auto; padding: 24px; }
    header { display:flex; justify-content:space-between; align-items:center; gap:16px; margin-bottom:20px; }
    h1 { font-size:24px; margin:0; font-weight:650; }
    .status { font-size:14px; color:#aab3bd; }
    .bar { display:flex; flex-wrap:wrap; gap:10px; margin-bottom:20px; }
    button { background:#242a30; color:#f4f7fa; border:1px solid #3c4650; padding:10px 14px; border-radius:6px; cursor:pointer; }
    button.active { background:#73342f; border-color:#a84d45; }
    .grid { display:grid; grid-template-columns: 1fr 1fr; gap:16px; }
    section { background:#171b1f; border:1px solid #2d343b; border-radius:8px; padding:16px; min-height:220px; }
    h2 { font-size:14px; color:#aab3bd; margin:0 0 12px; text-transform:uppercase; letter-spacing:.04em; }
    .text { font-size:24px; line-height:1.35; white-space:pre-wrap; }
    .meta { display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:10px; margin-bottom:16px; }
    .metric { background:#171b1f; border:1px solid #2d343b; border-radius:8px; padding:12px; }
    .metric span { display:block; font-size:12px; color:#aab3bd; margin-bottom:4px; }
    .metric strong { font-size:18px; }
    @media (max-width: 820px) { .grid, .meta { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <header>
    <h1>Live Interpreter Client</h1>
    <div class="status" id="server"></div>
  </header>
  <div class="bar">
    <button id="pause" onclick="act('/api/toggle-pause')">Pausa</button>
    <button id="input" onclick="act('/api/toggle-input')">Mutear entrada</button>
    <button id="output" onclick="act('/api/toggle-output')">Mutear salida</button>
    <button id="direction" onclick="act('/api/swap-direction')">Cambiar dirección</button>
  </div>
  <div class="meta">
    <div class="metric"><span>Estado</span><strong id="status"></strong></div>
    <div class="metric"><span>Dirección</span><strong id="dir"></strong></div>
    <div class="metric"><span>Latencia</span><strong id="latency"></strong></div>
    <div class="metric"><span>Chunks</span><strong id="chunks"></strong></div>
  </div>
  <div class="grid">
    <section><h2>Transcripción</h2><div class="text" id="transcript"></div></section>
    <section><h2>Traducción</h2><div class="text" id="translation"></div></section>
  </div>
</main>
<script>
async function act(url) { await fetch(url, {method:'POST'}); await tick(); }
function cls(id, on) { document.getElementById(id).classList.toggle('active', on); }
async function tick() {
  const s = await fetch('/api/state').then(r => r.json());
  document.getElementById('server').textContent = s.server_url;
  document.getElementById('status').textContent = s.status;
  document.getElementById('dir').textContent = s.direction;
  document.getElementById('latency').textContent = s.last_latency_ms + ' ms';
  document.getElementById('chunks').textContent = s.chunks_sent;
  document.getElementById('transcript').textContent = s.last_transcript || '';
  document.getElementById('translation').textContent = s.last_translation || s.last_error || '';
  cls('pause', s.paused); cls('input', s.input_muted); cls('output', s.output_muted);
}
setInterval(tick, 500); tick();
</script>
</body>
</html>
"#;
