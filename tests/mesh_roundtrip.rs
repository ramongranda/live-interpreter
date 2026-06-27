//! R10 live mesh verification: two real libp2p nodes (provider + consumer) on
//! loopback discover each other via mDNS, and an audio task submitted by the
//! consumer round-trips to the provider and back.
//!
//! This exercises the actual transport (mDNS discovery → gossipsub health →
//! request-response bincode), not a mock — the layer `li-mesh` depends on. The
//! pipeline itself is stubbed with an echo processor so the test needs no GPU,
//! models, or external services. mDNS uses UDP multicast; if the environment
//! blocks it, discovery never happens and the test times out (see `#[ignore]`).

use anyhow::Result;
use async_trait::async_trait;
use live_interpreter::mesh::{
    AudioChunk, AudioTaskResult, GpuTelemetry, GpuTelemetrySnapshot, LiveInterpreterMesh,
    MeshAudioProcessor, MeshCommand, MeshConfig, MeshRole, NoopGpuTelemetry,
    RejectingAudioProcessor,
};
use live_interpreter::types::Direction;
use std::time::Duration;
use tokio::sync::oneshot;
use uuid::Uuid;

struct FixedTelemetry;

#[async_trait]
impl GpuTelemetry for FixedTelemetry {
    async fn read(&self) -> Result<GpuTelemetrySnapshot> {
        Ok(GpuTelemetrySnapshot {
            free_vram_mb: 16_000,
            total_vram_mb: 16_000,
            active_sessions: 0,
        })
    }
}

/// Stands in for the real pipeline: echoes a canned translated result.
struct EchoProcessor;

#[async_trait]
impl MeshAudioProcessor for EchoProcessor {
    async fn process(&self, chunk: AudioChunk) -> Result<AudioTaskResult> {
        Ok(AudioTaskResult {
            session_id: chunk.session_id,
            sequence: chunk.sequence,
            transcription: "hola mundo".into(),
            translation: "hello world".into(),
            tts_sample_rate_hz: 24_000,
            tts_output: vec![0.1, 0.2, 0.3],
        })
    }
}

fn fast_config(role: MeshRole) -> MeshConfig {
    MeshConfig {
        local_role: role,
        health_interval: Duration::from_millis(500),
        ..MeshConfig::default()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs UDP-multicast mDNS on loopback; run with --ignored"]
async fn audio_task_round_trips_provider_to_consumer() {
    // Provider node: advertises VRAM, answers audio tasks with the echo pipeline.
    let provider = LiveInterpreterMesh::new(
        fast_config(MeshRole::GpuProvider),
        FixedTelemetry,
        EchoProcessor,
    );
    let (_p_cmd, p_rx) = LiveInterpreterMesh::<FixedTelemetry, EchoProcessor>::command_channel();
    tokio::spawn(async move {
        let _ = provider.run(p_rx).await;
    });

    // Consumer node: discovers the provider, submits audio, plays nothing here.
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
        sample_rate_hz: 16_000,
        direction: Direction::EsToEn,
        samples: vec![0.0; 1600],
        voice_ref: None,
    };

    // Poll: SubmitAudio fails fast while no provider is known yet, so retry until
    // mDNS + gossipsub have populated the provider table (or we give up).
    let result = tokio::time::timeout(Duration::from_secs(40), async {
        loop {
            let (reply_tx, reply_rx) = oneshot::channel();
            c_cmd
                .send(MeshCommand::SubmitAudio {
                    chunk: chunk.clone(),
                    reply: reply_tx,
                })
                .await
                .expect("consumer mesh task alive");
            match reply_rx.await.expect("reply channel") {
                Ok(result) => break result,
                Err(_) => tokio::time::sleep(Duration::from_millis(500)).await,
            }
        }
    })
    .await
    .expect("mesh round-trip within 40s (mDNS may be blocked in this environment)");

    assert_eq!(result.transcription, "hola mundo");
    assert_eq!(result.translation, "hello world");
    assert_eq!(result.tts_sample_rate_hz, 24_000);
    assert_eq!(result.tts_output, vec![0.1, 0.2, 0.3]);
}
