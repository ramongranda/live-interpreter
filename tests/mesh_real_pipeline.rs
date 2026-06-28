//! End-to-end validation of the **real** interpretation pipeline over the mesh:
//! two real libp2p nodes where the provider runs the actual
//! Whisper (ASR) → Ollama (translate) → Qwen3-TTS (synthesize) pipeline and
//! streams the translated voice back **clause-by-clause** (R11.7) to the consumer.
//!
//! Unlike `mesh_roundtrip.rs` (mock echo processor), this exercises the whole
//! production stack — so it needs the real services running (Ollama on :11434,
//! Qwen3-TTS on :8020), a Whisper ggml model, and a sample WAV. It is therefore
//! `#[ignore]` (run explicitly). Build with `--features cuda` to validate the
//! GPU path (Whisper turbo); without it, Whisper runs on CPU (slower, same result
//! shape).
//!
//! Run:
//! ```bash
//! LI_WHISPER_MODEL=data/models/ggml-large-v3-turbo.bin \
//!   cargo test --features cuda --test mesh_real_pipeline -- --ignored --nocapture
//! ```

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use chrono::{DateTime, Utc};
use live_interpreter::asr::AsrEngine;
use live_interpreter::capture::read_wav_f32;
use live_interpreter::config::Config;
use live_interpreter::mesh::{
    AudioChunk, AudioSegment, LiveInterpreterMesh, MeshCommand, MeshConfig, MeshRole,
    NoopGpuTelemetry, NvmlGpuTelemetry, RejectingAudioProcessor,
};
use live_interpreter::mesh_pipeline::PipelineMeshProcessor;
use live_interpreter::translate::Translator;
use live_interpreter::types::{Direction, Lang};
use live_interpreter::voice::{HttpQwenBackend, VoiceIdentity, VoiceProfile};
use tokio::sync::mpsc;
use uuid::Uuid;

/// A consent-less, sample-less profile → routes to the neutral voice (no clone),
/// so the test needs no reference audio.
fn neutral_profile() -> VoiceProfile {
    VoiceProfile {
        id: Uuid::nil(),
        name: "neutral".into(),
        owner: "test".into(),
        consent_confirmed: false,
        samples: Vec::new(),
        embedding_path: None,
        default_lang: Lang::En,
        quality_score: 0.0,
        created_at: DateTime::<Utc>::from_timestamp(0, 0).unwrap(),
    }
}

fn fast_config(role: MeshRole) -> MeshConfig {
    MeshConfig {
        local_role: role,
        health_interval: Duration::from_millis(500),
        ..MeshConfig::default()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs Ollama + Qwen3-TTS + a Whisper model + mDNS multicast; run with --ignored"]
async fn real_pipeline_streams_translation_over_mesh() {
    let config = Config::from_env().expect("config from env");
    let sample_wav = std::path::Path::new("data/voice/reference.wav");
    assert!(
        sample_wav.exists(),
        "sample WAV not found at {} (record a reference voice first)",
        sample_wav.display()
    );
    let (samples, sample_rate_hz) = read_wav_f32(sample_wav).expect("read sample wav");

    // Provider node: the real pipeline behind the mesh.
    let asr = AsrEngine::load(&config).expect("load Whisper model (set LI_WHISPER_MODEL)");
    let translator = Translator::from_env(config.ollama_url.clone(), config.ollama_model.clone())
        .expect("translator");
    let voice = HttpQwenBackend::from_env();
    let processor = PipelineMeshProcessor::new(
        Arc::new(asr),
        Arc::new(translator),
        Arc::new(voice),
        neutral_profile(),
        VoiceIdentity::Neutral,
        config.data_dir.join("uploads"),
        config.data_dir.join("voice"),
    );
    let provider = LiveInterpreterMesh::new(
        fast_config(MeshRole::GpuProvider),
        NvmlGpuTelemetry,
        processor,
    );
    let (_p_cmd, p_rx) =
        LiveInterpreterMesh::<NvmlGpuTelemetry, PipelineMeshProcessor>::command_channel();
    tokio::spawn(async move {
        let _ = provider.run(p_rx).await;
    });

    // Consumer node.
    let consumer = LiveInterpreterMesh::new(
        fast_config(MeshRole::Consumer),
        NoopGpuTelemetry,
        RejectingAudioProcessor,
    );
    let (c_cmd, c_rx) =
        LiveInterpreterMesh::<NoopGpuTelemetry, RejectingAudioProcessor>::command_channel();
    tokio::spawn(async move {
        let _ = consumer.run(c_rx).await;
    });

    let chunk = AudioChunk {
        session_id: Uuid::new_v4(),
        sequence: 0,
        sample_rate_hz,
        direction: Direction::EsToEn,
        samples,
        voice_ref: None,
        auth_token: None,
    };

    // Poll until the provider is discovered, then collect the streamed clauses.
    let segments: Vec<AudioSegment> = tokio::time::timeout(Duration::from_secs(90), async {
        loop {
            let (seg_tx, mut seg_rx) = mpsc::channel(32);
            c_cmd
                .send(MeshCommand::SubmitAudio {
                    chunk: chunk.clone(),
                    segments: seg_tx,
                })
                .await
                .expect("consumer alive");
            // First clause may take seconds (ASR + translate + first TTS clause).
            match tokio::time::timeout(Duration::from_secs(30), seg_rx.recv()).await {
                Ok(Some(first)) => {
                    let mut collected = vec![first];
                    while let Some(segment) = seg_rx.recv().await {
                        let last = segment.last;
                        collected.push(segment);
                        if last {
                            break;
                        }
                    }
                    break collected;
                }
                _ => tokio::time::sleep(Duration::from_millis(500)).await,
            }
        }
    })
    .await
    .expect("real pipeline streamed within 90s (services up? mDNS allowed?)");

    // The real stack produced ordered, terminated, non-empty audio clauses.
    assert!(!segments.is_empty(), "at least one clause");
    assert!(segments.last().unwrap().last, "stream is terminated");
    for (i, segment) in segments.iter().enumerate() {
        assert_eq!(segment.clause_index, i as u32, "clauses arrive in order");
    }
    assert!(
        !segments[0].transcription.trim().is_empty(),
        "Whisper produced a transcription"
    );
    assert!(
        segments.iter().any(|s| !s.translation.trim().is_empty()),
        "translation is non-empty"
    );
    assert!(
        segments.iter().any(|s| !s.tts_output.is_empty()),
        "TTS produced audio"
    );

    eprintln!(
        "real pipeline e2e: {} clause(s); src=\"{}\"",
        segments.len(),
        segments[0].transcription.trim()
    );
    for s in &segments {
        eprintln!(
            "  clause {}: \"{}\" ({} samples @ {} Hz)",
            s.clause_index,
            s.translation.trim(),
            s.tts_output.len(),
            s.tts_sample_rate_hz
        );
    }
}
