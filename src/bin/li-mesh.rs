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

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use live_interpreter::asr::AsrEngine;
use live_interpreter::capture::{self, CaptureConfig};
use live_interpreter::config::Config;
use live_interpreter::mesh::{
    AudioChunk, GpuTelemetry, LiveInterpreterMesh, MeshCommand, MeshConfig, MeshRole,
    NoopGpuTelemetry, NvmlGpuTelemetry, RejectingAudioProcessor, VoiceReference,
};
use live_interpreter::mesh_pipeline::{PipelineMeshProcessor, f32_to_pcm_s16le};
use live_interpreter::pipeline::Transcriber;
use live_interpreter::quality::QualityTier;
use live_interpreter::translate::Translator;
use live_interpreter::types::{AudioSpec, Direction, Lane, Lang};
use live_interpreter::virtual_mic::{AudioOutput, PipewireVirtualMic};
use live_interpreter::voice::{
    AudioFrame, HttpQwenBackend, VoiceIdentity, VoiceProfile, VoiceRoute, VoiceSample,
    VoiceSynthesisBackend, VoiceSynthesisRequest, VoiceUsagePolicy, route_for_lane,
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
        "bench" => run_bench().await,
        other => {
            bail!("unknown LI_ROLE '{other}' (expected 'provider', 'consumer', or 'bench')")
        }
    }
}

/// GPU provider: run the real pipeline for mesh audio tasks.
async fn run_provider() -> Result<()> {
    let mut config = Config::from_env()?;
    let tier = apply_quality_tier(&mut config);
    tracing::info!(
        "quality tier = {tier:?}, whisper model = {}",
        config.whisper_model.display()
    );
    let asr = AsrEngine::load(&config).context("Whisper model not found (set LI_WHISPER_MODEL)")?;
    let translator = Translator::from_env(config.ollama_url.clone(), config.ollama_model.clone())?;
    let voice = HttpQwenBackend::from_env();
    let (profile, identity) = voice_identity_from_env();
    let uploads = config.data_dir.join("uploads");
    let voice_dir = config.data_dir.join("voice");

    let processor = PipelineMeshProcessor::new(
        Arc::new(asr),
        Arc::new(translator),
        Arc::new(voice),
        profile,
        identity,
        uploads,
        voice_dir,
    );
    let mesh = LiveInterpreterMesh::new(
        MeshConfig {
            local_role: MeshRole::GpuProvider,
            auth_token: mesh_token(),
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
    let token = mesh_token();
    let mic =
        PipewireVirtualMic::spawn(AudioSpec::mono_s16le(24_000), "live-interpreter-mic-source")
            .context("failed to start PipeWire virtual mic")?;

    let mesh = LiveInterpreterMesh::new(
        MeshConfig {
            auth_token: token.clone(),
            ..MeshConfig::default() // local_role = Consumer
        },
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
    let voice_ref = consumer_voice_reference();
    tracing::info!(
        "mesh consumer listening ({sample_rate} Hz, {} ch) → {direction:?}; output on \
         'live-interpreter-mic-source'. Needs a provider on the LAN.",
        cap.channels
    );

    while let Some(utterance) = utterances.recv().await {
        let seq = sequence.fetch_add(1, Ordering::Relaxed);
        let chunk = AudioChunk {
            session_id,
            sequence: seq,
            sample_rate_hz: sample_rate,
            direction,
            samples: utterance,
            // Ship the timbre once per session (first chunk); the provider caches it.
            voice_ref: if seq == 0 { voice_ref.clone() } else { None },
            auth_token: token.clone(),
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

/// The consumer's own timbre from `LI_VOICE_REF`, shipped to the provider for
/// cross-node cloning. Providing the file implies consent (it's your own voice).
/// Bench (R11.1): measure the local pipeline latency stage by stage, so the
/// CPU/GPU latency floor is a number you can track instead of a vibe.
///
/// Input (env, one required):
///   * `LI_BENCH_WAV=<path.wav>` — full pipeline (ASR + translate + TTS).
///   * `LI_BENCH_TEXT="…"`        — translate + TTS only (skips ASR).
///
/// `LI_DIRECTION` flips direction; `LI_BENCH_ITERS` sets the timed repeat count
/// (default 3). One untimed warmup pass absorbs model-load / first-connection
/// cost so iteration 1 isn't skewed. Voice route follows `LI_VOICE_REF` exactly
/// like the provider (clone vs neutral).
async fn run_bench() -> Result<()> {
    let mut config = Config::from_env()?;
    let tier = apply_quality_tier(&mut config);
    let direction = direction_from_env();
    let iters: usize = std::env::var("LI_BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3)
        .max(1);

    let wav_path = std::env::var("LI_BENCH_WAV").ok().map(PathBuf::from);
    let fixed_text = std::env::var("LI_BENCH_TEXT").ok();
    if wav_path.is_none() && fixed_text.is_none() {
        bail!(
            "bench needs LI_BENCH_WAV=<wav> (full pipeline) or LI_BENCH_TEXT=<text> (translate+TTS only)"
        );
    }
    if let Some(p) = &wav_path
        && !p.exists()
    {
        bail!("LI_BENCH_WAV not found: {}", p.display());
    }

    // Input audio seconds → real-time factor (RTF = processing / spoken time).
    let input_secs = match &wav_path {
        Some(p) => {
            let (samples, sample_rate) = capture::read_wav_f32(p)?;
            samples.len() as f64 / sample_rate.max(1) as f64
        }
        None => 0.0,
    };

    let asr = match &wav_path {
        Some(_) => Some(
            AsrEngine::load(&config).context("Whisper model not found (set LI_WHISPER_MODEL)")?,
        ),
        None => None,
    };
    let translator = Translator::from_env(config.ollama_url.clone(), config.ollama_model.clone())?;
    let voice = HttpQwenBackend::from_env();
    let (profile, identity) = voice_identity_from_env();
    let route = route_for_lane(identity, Lane::Local);
    let target_lang = match direction {
        Direction::EsToEn => Lang::En,
        Direction::EnToEs => Lang::Es,
    };
    let telemetry = NvmlGpuTelemetry;

    let input_label = wav_path
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| format!("text:{:?}", fixed_text.as_deref().unwrap_or("")));
    println!(
        "bench — tier={tier:?} direction={direction:?} route={route:?} iters={iters} input={input_label}"
    );
    if wav_path.is_some() {
        println!("       whisper model = {}", config.whisper_model.display());
    }

    // Warmup: pay model-load / first-connection cost once, untimed.
    let warm = bench_once(
        asr.as_ref(),
        &translator,
        &voice,
        &profile,
        route,
        direction,
        wav_path.as_deref(),
        fixed_text.as_deref(),
        target_lang,
    )
    .await
    .context("warmup pass failed (is Ollama on :11434 and Qwen3-TTS on :8020?)")?;
    println!(
        "warmup ok — \"{}\" → \"{}\"",
        warm.transcript.trim(),
        warm.translation.trim()
    );

    let vram_before = telemetry.read().await.ok();
    let mut ttfa_samples = Vec::with_capacity(iters);
    let mut total_samples = Vec::with_capacity(iters);
    for i in 0..iters {
        let r = bench_once(
            asr.as_ref(),
            &translator,
            &voice,
            &profile,
            route,
            direction,
            wav_path.as_deref(),
            fixed_text.as_deref(),
            target_lang,
        )
        .await?;
        let rtf = if input_secs > 0.0 {
            format!("{:.2}", r.total.as_secs_f64() / input_secs)
        } else {
            "n/a".into()
        };
        println!(
            "#{:<2} asr={:>7} translate={:>7} tts_first={:>7} tts_full={:>7} total={:>7} ttfa={:>7} rtf={}",
            i + 1,
            r.asr.map(fmt_ms).unwrap_or_else(|| "—".into()),
            fmt_ms(r.translate),
            fmt_ms(r.tts_first),
            fmt_ms(r.tts_full),
            fmt_ms(r.total),
            fmt_ms(r.ttfa),
            rtf,
        );
        ttfa_samples.push(r.ttfa);
        total_samples.push(r.total);
    }
    let vram_after = telemetry.read().await.ok();

    ttfa_samples.sort_unstable();
    total_samples.sort_unstable();
    println!(
        "— median ttfa={} total={}  (ttfa = time-to-first-audio: the conversational latency)",
        fmt_ms(ttfa_samples[ttfa_samples.len() / 2]),
        fmt_ms(total_samples[total_samples.len() / 2]),
    );
    if let (Some(a), Some(b)) = (&vram_before, &vram_after) {
        println!(
            "— vram free {}→{} MB of {} total",
            a.free_vram_mb, b.free_vram_mb, b.total_vram_mb
        );
    }
    if input_secs > 0.0 {
        println!("— input audio {input_secs:.2}s (rtf < 1.0 = faster than real-time)");
    }
    Ok(())
}

/// Per-stage timings for one bench iteration. `ttfa` (time-to-first-audio) is the
/// headline conversational metric: ASR + translate + first TTS frame.
struct StageTimings {
    asr: Option<Duration>,
    translate: Duration,
    tts_first: Duration,
    tts_full: Duration,
    total: Duration,
    ttfa: Duration,
    transcript: String,
    translation: String,
}

/// Run one utterance through the real components, timing each stage. Uses only
/// the public backend API (no internal pipeline timing hooks), so it mirrors the
/// provider's exact path: `transcribe` → clean `translate` → `synthesize_stream`.
#[allow(clippy::too_many_arguments)]
async fn bench_once(
    asr: Option<&AsrEngine>,
    translator: &Translator,
    voice: &HttpQwenBackend,
    profile: &VoiceProfile,
    route: VoiceRoute,
    direction: Direction,
    wav: Option<&Path>,
    fixed_text: Option<&str>,
    target_lang: Lang,
) -> Result<StageTimings> {
    let overall = Instant::now();

    let (transcript, asr_dur) = match (asr, wav) {
        (Some(asr), Some(wav)) => {
            let t = Instant::now();
            let text = asr.transcribe(wav, direction.source_lang()).await?;
            (text, Some(t.elapsed()))
        }
        _ => (fixed_text.unwrap_or_default().to_string(), None),
    };

    let t = Instant::now();
    let translation = translator.translate(&transcript, &direction).await?;
    let translate = t.elapsed();

    let (tts_first, tts_full) = if route == VoiceRoute::Off {
        (Duration::ZERO, Duration::ZERO)
    } else {
        let request = VoiceSynthesisRequest {
            text: translation.clone(),
            lang: target_lang,
            profile: profile.clone(),
            neutral: route == VoiceRoute::Neutral,
        };
        let t = Instant::now();
        let mut stream = voice.synthesize_stream(request).await?;
        let first = stream.next().await;
        let tts_first = t.elapsed();
        if let Some(frame) = first {
            frame?;
        }
        while let Some(frame) = stream.next().await {
            frame?;
        }
        (tts_first, t.elapsed())
    };

    let total = overall.elapsed();
    let ttfa = asr_dur.unwrap_or_default() + translate + tts_first;
    Ok(StageTimings {
        asr: asr_dur,
        translate,
        tts_first,
        tts_full,
        total,
        ttfa,
        transcript,
        translation,
    })
}

fn fmt_ms(d: Duration) -> String {
    format!("{}ms", d.as_millis())
}

/// R11.3: when `LI_WHISPER_MODEL` is unset, fall back to the quality tier's model
/// — but only if that file exists, so an explicit/working setup is never broken.
/// Returns the effective tier (for logging).
fn apply_quality_tier(config: &mut Config) -> QualityTier {
    let tier = QualityTier::from_env();
    if std::env::var("LI_WHISPER_MODEL").is_err() {
        let candidate = PathBuf::from(tier.whisper_model());
        if candidate.exists() {
            config.whisper_model = candidate;
        }
    }
    tier
}

fn consumer_voice_reference() -> Option<VoiceReference> {
    let path = PathBuf::from(std::env::var("LI_VOICE_REF").ok()?);
    if !path.exists() {
        tracing::info!("no LI_VOICE_REF → consumer ships no timbre (provider renders neutral)");
        return None;
    }
    // R11.2: privacy gate. With LI_VOICE_ALLOW_REMOTE=0 the timbre never leaves
    // this node — the provider then renders neutral.
    if !VoiceUsagePolicy::from_env().may_ship_reference(true) {
        tracing::info!(
            "voice usage policy blocks remote synthesis → consumer keeps its timbre local"
        );
        return None;
    }
    match capture::read_wav_f32(&path) {
        Ok((samples, sample_rate_hz)) => {
            tracing::info!("voice reference loaded → consumer ships its timbre to the provider");
            Some(VoiceReference {
                sample_rate_hz,
                samples,
                transcript: std::env::var("LI_VOICE_REF_TEXT").ok(),
                consent_confirmed: true,
            })
        }
        Err(error) => {
            tracing::warn!("failed to read LI_VOICE_REF: {error:#}");
            None
        }
    }
}

/// Shared mesh secret from `LI_MESH_TOKEN`; `None` (unset/empty) = open mesh.
fn mesh_token() -> Option<String> {
    std::env::var("LI_MESH_TOKEN")
        .ok()
        .filter(|token| !token.is_empty())
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
