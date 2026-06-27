//! Mesh node (R10): the translated voice travels to another node.
//!
//! Two roles, selected by `LI_ROLE`:
//!
//! * `provider` — owns the GPU pipeline (Whisper + Ollama + Qwen3-TTS). Joins the
//!   libp2p mesh, advertises VRAM over gossipsub, and answers audio tasks: it
//!   transcribes → translates → synthesizes the incoming `f32` samples and
//!   returns the translated voice.
//! * `consumer` (default) — captures the local mic, VAD-segments utterances,
//!   ships each to the best provider on the LAN, and plays the returned
//!   translated voice into the `live-interpreter-mic-source` virtual mic.
//!
//! Discovery is mDNS (same LAN); the consumer needs no address. Direction
//! defaults to `es_to_en` (`LI_DIRECTION=en_to_es` to flip).
//!
//! ```bash
//! # box A (has the GPU + models + Qwen3-TTS on :8020)
//! LI_ROLE=provider cargo run --features native-audio --bin li-mesh
//! # box B (no GPU needed)
//! LI_ROLE=consumer cargo run --features native-audio --bin li-mesh
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use live_interpreter::asr::AsrEngine;
use live_interpreter::capture::{self, CaptureConfig};
use live_interpreter::config::Config;
use live_interpreter::mesh::{
    AudioChunk, LiveInterpreterMesh, MeshCommand, MeshConfig, MeshRole, NoopGpuTelemetry,
    NvmlGpuTelemetry, RejectingAudioProcessor,
};
use live_interpreter::mesh_pipeline::{PipelineMeshProcessor, f32_to_pcm_s16le};
use live_interpreter::translate::Translator;
use live_interpreter::types::{AudioSpec, Direction, Lang};
use live_interpreter::virtual_mic::{AudioOutput, PipewireVirtualMic};
use live_interpreter::voice::{
    AudioFrame, HttpQwenBackend, VoiceIdentity, VoiceProfile, VoiceSample,
};
use tokio::sync::oneshot;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let role = std::env::var("LI_ROLE").unwrap_or_else(|_| "consumer".into());
    match role.as_str() {
        "provider" => run_provider().await,
        "consumer" => run_consumer().await,
        other => anyhow::bail!("unknown LI_ROLE '{other}' (expected 'provider' or 'consumer')"),
    }
}

/// GPU provider: run the real pipeline for mesh audio tasks.
async fn run_provider() -> Result<()> {
    let config = Config::from_env()?;
    let asr = AsrEngine::load(&config).context("Whisper model not found (set LI_WHISPER_MODEL)")?;
    let translator = Translator::from_env(config.ollama_url.clone(), config.ollama_model.clone())?;
    let voice = HttpQwenBackend::from_env();
    let (profile, identity) = voice_identity_from_env();
    let uploads = config.data_dir.join("uploads");

    let processor = PipelineMeshProcessor::new(
        Arc::new(asr),
        Arc::new(translator),
        Arc::new(voice),
        profile,
        identity,
        uploads,
    );
    let mesh = LiveInterpreterMesh::new(
        MeshConfig {
            local_role: MeshRole::GpuProvider,
            ..MeshConfig::default()
        },
        NvmlGpuTelemetry,
        processor,
    );
    let (_commands, rx) =
        LiveInterpreterMesh::<NvmlGpuTelemetry, PipelineMeshProcessor>::command_channel();

    tracing::info!(
        "mesh provider ready — advertising GPU, waiting for audio tasks (Ctrl-C to quit)"
    );
    let handle = tokio::spawn(async move {
        if let Err(error) = mesh.run(rx).await {
            tracing::error!("mesh provider stopped: {error:#}");
        }
    });
    tokio::signal::ctrl_c().await.ok();
    handle.abort();
    Ok(())
}

/// Consumer: capture → mesh → play the translated voice into the virtual mic.
async fn run_consumer() -> Result<()> {
    let direction = direction_from_env();
    let mic =
        PipewireVirtualMic::spawn(AudioSpec::mono_s16le(24_000), "live-interpreter-mic-source")
            .context("failed to start PipeWire virtual mic")?;

    let mesh = LiveInterpreterMesh::new(
        MeshConfig::default(), // local_role = Consumer
        NoopGpuTelemetry,
        RejectingAudioProcessor,
    );
    let (commands, rx) =
        LiveInterpreterMesh::<NoopGpuTelemetry, RejectingAudioProcessor>::command_channel();
    tokio::spawn(async move {
        if let Err(error) = mesh.run(rx).await {
            tracing::error!("mesh consumer stopped: {error:#}");
        }
    });

    let cap = capture::start_capture(CaptureConfig::default())?;
    let sample_rate = cap.sample_rate;
    let _capture_stream = cap.stream; // hold to keep the device open
    let mut utterances = cap.utterances;
    let session_id = Uuid::new_v4();
    let sequence = AtomicU64::new(0);
    tracing::info!(
        "mesh consumer listening ({sample_rate} Hz, {} ch) → {direction:?}; output on \
         'live-interpreter-mic-source'. Needs a provider on the LAN.",
        cap.channels
    );

    while let Some(utterance) = utterances.recv().await {
        let chunk = AudioChunk {
            session_id,
            sequence: sequence.fetch_add(1, Ordering::Relaxed),
            sample_rate_hz: sample_rate,
            direction,
            samples: utterance,
        };
        let (reply_tx, reply_rx) = oneshot::channel();
        if commands
            .send(MeshCommand::SubmitAudio {
                chunk,
                reply: reply_tx,
            })
            .await
            .is_err()
        {
            tracing::error!("mesh task loop is gone; stopping");
            break;
        }
        match reply_rx.await {
            Ok(Ok(result)) => {
                tracing::info!("· src: {}", result.transcription);
                tracing::info!("· dst: {}", result.translation);
                let frame = AudioFrame {
                    spec: AudioSpec::mono_s16le(result.tts_sample_rate_hz),
                    pcm: f32_to_pcm_s16le(&result.tts_output),
                };
                if let Err(error) = mic.submit(&frame) {
                    tracing::error!("virtual mic submit failed: {error:#}");
                } else {
                    tracing::info!("→ translated voice to virtual mic");
                }
            }
            Ok(Err(error)) => tracing::warn!("mesh task failed: {error:#}"),
            Err(_) => tracing::warn!("mesh task dropped without a reply"),
        }
    }
    Ok(())
}

/// Direction from `LI_DIRECTION` (`en_to_es` flips; default `es_to_en`).
fn direction_from_env() -> Direction {
    match std::env::var("LI_DIRECTION").as_deref() {
        Ok("en_to_es") => Direction::EnToEs,
        _ => Direction::EsToEn,
    }
}

/// Provider voice: clone with `LI_VOICE_REF` if present, else the neutral voice.
fn voice_identity_from_env() -> (VoiceProfile, VoiceIdentity) {
    match std::env::var("LI_VOICE_REF") {
        Ok(path) if PathBuf::from(&path).exists() => {
            tracing::info!("voice profile loaded → provider clones the timbre");
            (clone_profile(path.into()), VoiceIdentity::MyProfile)
        }
        _ => {
            tracing::info!("no LI_VOICE_REF → provider renders the neutral voice");
            (placeholder_profile(), VoiceIdentity::Neutral)
        }
    }
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
            lang: Lang::Es,
            duration_ms: 0,
            sample_rate: 24_000,
        }],
        embedding_path: None,
        default_lang: Lang::Es,
        quality_score: 1.0,
        created_at: chrono::Utc::now(),
    }
}

fn placeholder_profile() -> VoiceProfile {
    VoiceProfile {
        id: Uuid::nil(),
        name: "neutral".into(),
        owner: "node".into(),
        consent_confirmed: false,
        samples: Vec::new(),
        embedding_path: None,
        default_lang: Lang::En,
        quality_score: 0.0,
        created_at: chrono::Utc::now(),
    }
}
