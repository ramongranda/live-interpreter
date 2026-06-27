//! End-to-end demo: text → `HttpQwenBackend` (cloning TTS) → PipeWire virtual mic.
//!
//! Proves the Candle-native thesis with the pieces that exist today: the runtime
//! produces voice in-process and injects it into a system microphone, no `pw-play`,
//! no Bash. Needs the Qwen3-TTS service on :8020 and a running PipeWire session.
//!
//! ```bash
//! cargo run --features native-audio --bin li-voice-demo -- "Hello, this is my voice"
//! # neutral default voice; set LI_VOICE_REF=data/voice/reference.wav (+ LI_VOICE_REF_TEXT)
//! # and pass --clone to render with the personalized voice profile.
//! ```

use std::time::Duration;

use anyhow::{Context, Result};
use futures_util::StreamExt;
use live_interpreter::types::Lang;
use live_interpreter::virtual_mic::{AudioOutput, PipewireVirtualMic};
use live_interpreter::voice::{
    HttpQwenBackend, VoiceProfile, VoiceSample, VoiceSynthesisBackend, VoiceSynthesisRequest,
};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let clone = args.iter().any(|a| a == "--clone");
    let text = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "En un lugar de la Mancha, de cuyo nombre no quiero acordarme".into());

    let (profile, neutral) = if clone {
        (load_profile()?, false)
    } else {
        (neutral_profile(), true)
    };

    let backend = HttpQwenBackend::from_env();
    let request = VoiceSynthesisRequest {
        text: text.clone(),
        lang: Lang::En,
        profile,
        neutral,
    };

    tracing::info!(
        clone,
        "synthesizing {:?} via Qwen3-TTS → virtual mic",
        &text
    );
    let mut stream = backend
        .synthesize_stream(request)
        .await
        .context("Qwen3-TTS synthesis failed (is the service on :8020?)")?;

    // Collect all chunks (R7 will stream them as they arrive; here we buffer).
    let mut frames = Vec::new();
    while let Some(frame) = stream.next().await {
        frames.push(frame?);
    }
    let spec = frames
        .first()
        .map(|f| f.spec)
        .context("Qwen3-TTS produced no audio")?;

    let mic = PipewireVirtualMic::spawn(spec, "live-interpreter-mic-source")
        .context("failed to start PipeWire virtual mic (is PipeWire running?)")?;
    tracing::info!(
        "virtual mic 'live-interpreter-mic-source' up: {} ch @ {} Hz — select it in your meeting app",
        spec.channels,
        spec.sample_rate
    );

    let mut bytes = 0usize;
    for frame in &frames {
        mic.submit(frame)?;
        bytes += frame.pcm.len();
    }

    // Let PipeWire drain the ring buffer (audio duration + a small tail).
    let samples = bytes / (spec.channels.max(1) as usize * 2);
    let seconds = samples as f64 / spec.sample_rate.max(1) as f64;
    tracing::info!("playing ~{seconds:.1}s of audio into the virtual mic");
    tokio::time::sleep(Duration::from_secs_f64(seconds + 1.0)).await;
    Ok(())
}

/// Placeholder profile for the neutral path (ignored when `neutral = true`).
fn neutral_profile() -> VoiceProfile {
    VoiceProfile {
        id: Uuid::nil(),
        name: "neutral".into(),
        owner: "demo".into(),
        consent_confirmed: false,
        samples: Vec::new(),
        embedding_path: None,
        default_lang: Lang::En,
        quality_score: 0.0,
        created_at: chrono::Utc::now(),
    }
}

/// Build a consented profile from `LI_VOICE_REF` (+ `LI_VOICE_REF_TEXT`) for the
/// `--clone` path. Requires the reference WAV to exist.
fn load_profile() -> Result<VoiceProfile> {
    let path = std::env::var("LI_VOICE_REF")
        .unwrap_or_else(|_| "data/voice/reference.wav".into())
        .into();
    let transcript = std::env::var("LI_VOICE_REF_TEXT").ok();
    Ok(VoiceProfile {
        id: Uuid::new_v4(),
        name: "personal".into(),
        owner: "self".into(),
        consent_confirmed: true,
        samples: vec![VoiceSample {
            path,
            transcript,
            lang: Lang::Es,
            duration_ms: 0,
            sample_rate: 24_000,
        }],
        embedding_path: None,
        default_lang: Lang::Es,
        quality_score: 1.0,
        created_at: chrono::Utc::now(),
    })
}
