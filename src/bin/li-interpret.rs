//! Real voice interpreter (R6-bin): mic capture + VAD → Whisper ASR → Ollama
//! translate → cloned voice (Qwen3-TTS) → PipeWire virtual mic. All wired through
//! `pipeline::interpret_utterance`, the trait-tested orchestration core.
//!
//! Speak Spanish → the `live-interpreter-mic-source` device emits the English
//! translation in your timbre. Needs: Whisper ggml model (`LI_WHISPER_MODEL`),
//! Ollama (`translator:latest`), the Qwen3-TTS service on :8020, and PipeWire.
//!
//! ```bash
//! cargo run --features native-audio --bin li-interpret
//! # clone path: export LI_VOICE_REF=data/voice/reference.wav LI_VOICE_REF_TEXT="..."
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::{Html, IntoResponse};
use axum::routing::get;
use axum::{Json, Router};
use live_interpreter::asr::AsrEngine;
use live_interpreter::capture::{self, CaptureConfig};
use live_interpreter::config::Config;
use live_interpreter::events::EventHub;
use live_interpreter::pipeline::interpret_utterance_chunked;
use live_interpreter::runtime::assemble_app_status;
use live_interpreter::translate::Translator;
use live_interpreter::types::{AppStatus, AudioSpec, Direction, Lane, Liveness, PipelineEvent};
use live_interpreter::virtual_mic::PipewireVirtualMic;
use live_interpreter::voice::{HttpQwenBackend, VoiceIdentity, VoiceProfile, VoiceSample};
use tokio::net::TcpListener;
use uuid::Uuid;

#[derive(Clone)]
struct UiState {
    hub: Arc<EventHub>,
    voice_configured: bool,
    delay: Arc<AtomicU64>,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let config = Config::from_env()?;
    let asr = AsrEngine::load(&config).context("Whisper model not found (set LI_WHISPER_MODEL)")?;
    let translator = Translator::from_env(config.ollama_url.clone(), config.ollama_model.clone())?;
    let voice = HttpQwenBackend::from_env();
    let mic =
        PipewireVirtualMic::spawn(AudioSpec::mono_s16le(24_000), "live-interpreter-mic-source")
            .context("failed to start PipeWire virtual mic")?;

    let (profile, identity) = match std::env::var("LI_VOICE_REF") {
        Ok(path) if PathBuf::from(&path).exists() => {
            tracing::info!("voice profile loaded → cloning your timbre");
            (clone_profile(path.into()), VoiceIdentity::MyProfile)
        }
        _ => {
            tracing::info!("no LI_VOICE_REF → neutral voice");
            (placeholder_profile(), VoiceIdentity::Neutral)
        }
    };

    // Event hub + console UI server (serves the reactive FSM UI + live bubbles).
    let hub = Arc::new(EventHub::new(Uuid::new_v4(), 256));
    let delay = Arc::new(AtomicU64::new(0));
    let voice_configured = identity == VoiceIdentity::MyProfile;
    {
        let ui = UiState {
            hub: hub.clone(),
            voice_configured,
            delay: delay.clone(),
        };
        let bind = std::env::var("LI_UI_BIND").unwrap_or_else(|_| "127.0.0.1:8799".into());
        tokio::spawn(async move {
            let router = Router::new()
                .route("/", get(ui_index))
                .route("/api/status", get(ui_status))
                .route("/ws", get(ws_handler))
                .with_state(ui);
            match TcpListener::bind(&bind).await {
                Ok(listener) => {
                    tracing::info!("console UI on http://{bind}");
                    let _ = axum::serve(listener, router).await;
                }
                Err(error) => tracing::warn!("console UI disabled: {error}"),
            }
        });
    }

    // Shared cpal capture + energy-VAD utterance segmentation.
    let cap = capture::start_capture(CaptureConfig::default())?;
    let sample_rate = cap.sample_rate;
    let _capture_stream = cap.stream; // hold to keep the device open
    let mut utt_rx = cap.utterances;
    tracing::info!(
        "listening on default mic ({sample_rate} Hz, {} ch) → speak Spanish; \
         output on 'live-interpreter-mic-source'",
        cap.channels
    );

    let uploads = config.data_dir.join("uploads");
    tokio::fs::create_dir_all(&uploads).await.ok();

    while let Some(utterance) = utt_rx.recv().await {
        let id = Uuid::new_v4();
        let wav_path = uploads.join(format!("{id}-capture.wav"));
        if let Err(error) = capture::write_wav_16le(&wav_path, &utterance, sample_rate) {
            tracing::error!("wav write failed: {error:#}");
            continue;
        }
        let started = Instant::now();
        match interpret_utterance_chunked(
            &asr,
            &translator,
            &voice,
            &mic,
            &profile,
            &wav_path,
            Direction::EsToEn,
            identity,
            Lane::Local,
            id,
            0,
        )
        .await
        {
            Ok(events) => {
                let elapsed = started.elapsed().as_millis() as u64;
                delay.store(elapsed, Ordering::Relaxed);
                let ts = now_ms();
                for event in &events {
                    hub.publish(event.clone(), ts);
                }
                log_events(&events, elapsed);
            }
            Err(error) => tracing::error!("pipeline error: {error:#}"),
        }
        let _ = tokio::fs::remove_file(&wav_path).await;
    }
    Ok(())
}

async fn ui_index() -> Html<&'static str> {
    Html(include_str!("../../static/fsm-ui.html"))
}

async fn ui_status(State(state): State<UiState>) -> Json<AppStatus> {
    let snapshot = live_interpreter::vram::vram_snapshot().await.ok();
    let gpu = live_interpreter::vram::build_gpu_status(snapshot.as_ref(), 8_000);
    // The interpreter is the active console: it captures locally → ActiveClient.
    let live = Liveness {
        client: true,
        mic: true,
        ..Default::default()
    };
    Json(assemble_app_status(
        &live,
        false,
        false,
        gpu,
        state.voice_configured,
        1,
        state.delay.load(Ordering::Relaxed),
        None,
    ))
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<UiState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| ws_loop(socket, state.hub.clone()))
}

async fn ws_loop(mut socket: WebSocket, hub: Arc<EventHub>) {
    let mut rx = hub.subscribe();
    while let Ok(envelope) = rx.recv().await {
        match serde_json::to_string(&envelope) {
            Ok(json) => {
                if socket.send(Message::Text(json.into())).await.is_err() {
                    break;
                }
            }
            Err(_) => break,
        }
    }
}

fn log_events(events: &[PipelineEvent], latency_ms: u64) {
    for event in events {
        match event {
            PipelineEvent::Transcript { text, .. } => tracing::info!("· ES: {text}"),
            PipelineEvent::Translation { text, .. } => tracing::info!("· EN: {text}"),
            _ => {}
        }
    }
    let frames = events
        .iter()
        .filter(|e| matches!(e, PipelineEvent::AudioFrame { .. }))
        .count();
    tracing::info!("→ {frames} audio chunk(s) to virtual mic ({latency_ms} ms)");
}

fn clone_profile(path: PathBuf) -> VoiceProfile {
    VoiceProfile {
        id: Uuid::new_v4(),
        name: "personal".into(),
        owner: "self".into(),
        consent_confirmed: true,
        samples: vec![VoiceSample {
            path,
            transcript: std::env::var("LI_VOICE_REF_TEXT").ok(),
            lang: live_interpreter::types::Lang::Es,
            duration_ms: 0,
            sample_rate: 24_000,
        }],
        embedding_path: None,
        default_lang: live_interpreter::types::Lang::Es,
        quality_score: 1.0,
        created_at: chrono::Utc::now(),
    }
}

fn placeholder_profile() -> VoiceProfile {
    VoiceProfile {
        id: Uuid::nil(),
        name: "neutral".into(),
        owner: "demo".into(),
        consent_confirmed: false,
        samples: Vec::new(),
        embedding_path: None,
        default_lang: live_interpreter::types::Lang::En,
        quality_score: 0.0,
        created_at: chrono::Utc::now(),
    }
}
