//! Voice Identity Runtime: domain model + pluggable synthesis backend.
//!
//! The product differentiator is **vocal presence** — translated speech rendered
//! in the user's own timbre. `VoiceProfile` is therefore a first-class domain
//! entity (not a detail inside TTS), and synthesis goes through the
//! [`VoiceSynthesisBackend`] trait so the concrete engine (the current Qwen3-TTS
//! cloning service, a future Candle port, a neutral Kokoro voice, or a mock) is
//! swappable without touching the pipeline.
//!
//! Ethics: a profile may only render a user's timbre after explicit consent
//! (`consent_confirmed`). Framing is "personalized voice profile", never
//! "exact clone".

use crate::types::{AudioFormat, AudioSpec, Lane, Lang};
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use base64::{Engine, engine::general_purpose::STANDARD};
use chrono::{DateTime, Utc};
use futures_util::{Stream, stream};
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use std::path::PathBuf;
use std::pin::Pin;
use uuid::Uuid;

/// One recorded reference take that backs a voice identity.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct VoiceSample {
    pub path: PathBuf,
    pub transcript: Option<String>,
    pub lang: Lang,
    pub duration_ms: u32,
    pub sample_rate: u32,
}

/// A user-owned vocal identity. Consent-gated: rendering this timbre is refused
/// unless `consent_confirmed` is true and at least one sample exists.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VoiceProfile {
    pub id: Uuid,
    pub name: String,
    pub owner: String,
    pub consent_confirmed: bool,
    pub samples: Vec<VoiceSample>,
    pub embedding_path: Option<PathBuf>,
    pub default_lang: Lang,
    pub quality_score: f32,
    pub created_at: DateTime<Utc>,
}

impl VoiceProfile {
    /// True when the profile may be used to render *this user's* timbre.
    pub fn is_usable(&self) -> bool {
        self.consent_confirmed && !self.samples.is_empty()
    }
}

/// The "Voice Identity" UI selection (see design §5.4 / §7).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VoiceIdentity {
    /// Render my outgoing voice with my own profile (clone). Requires consent.
    MyProfile,
    /// Render everything with a neutral voice (Kokoro), no timbre.
    Neutral,
    /// Don't synthesize; only show the translated text.
    Off,
}

/// Which renderer a given lane uses. `Clone` = my timbre (Local/outgoing);
/// `Neutral` = Kokoro (the Remote/monitor lane — hearing the source translation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoiceRoute {
    Clone,
    Neutral,
    Off,
}

/// Per-lane backend routing policy. With `MyProfile`, the outgoing (Local) lane
/// is rendered in the user's timbre and the incoming (Remote) monitor lane uses
/// the neutral Kokoro voice — so you hear *what was said*, not a fake of the peer.
pub fn route_for_lane(identity: VoiceIdentity, lane: Lane) -> VoiceRoute {
    match identity {
        VoiceIdentity::Off => VoiceRoute::Off,
        VoiceIdentity::Neutral => VoiceRoute::Neutral,
        VoiceIdentity::MyProfile => match lane {
            Lane::Local => VoiceRoute::Clone,
            Lane::Remote => VoiceRoute::Neutral,
        },
    }
}

/// R11.2 — privacy policy governing how a voice profile may be used, especially
/// over the mesh. Default = the shipped behavior (cross-node cloning allowed,
/// consent required, output watermarked). Flip `allow_remote_synthesis` off to
/// keep the timbre on this box (remote then renders neutral).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct VoiceUsagePolicy {
    /// Hard gate: never synthesize a user's timbre without confirmed consent.
    pub require_consent: bool,
    /// May the voice reference travel to a remote provider for cross-node
    /// cloning? `false` = the reference never leaves this node.
    pub allow_remote_synthesis: bool,
    /// May the raw profile/embedding be exported off the device?
    pub allow_profile_export: bool,
    /// Stamp synthesized audio with provenance metadata ([`GeneratedAudioMeta`]).
    pub watermark: bool,
}

impl Default for VoiceUsagePolicy {
    fn default() -> Self {
        Self {
            require_consent: true,
            allow_remote_synthesis: true,
            allow_profile_export: false,
            watermark: true,
        }
    }
}

impl VoiceUsagePolicy {
    /// Read overrides from env (`LI_VOICE_REQUIRE_CONSENT`, `LI_VOICE_ALLOW_REMOTE`,
    /// `LI_VOICE_ALLOW_EXPORT`, `LI_VOICE_WATERMARK`). Unset = default; `0`/`false`/
    /// `no`/`off` disables, any other value enables.
    pub fn from_env() -> Self {
        let d = Self::default();
        Self {
            require_consent: env_flag("LI_VOICE_REQUIRE_CONSENT", d.require_consent),
            allow_remote_synthesis: env_flag("LI_VOICE_ALLOW_REMOTE", d.allow_remote_synthesis),
            allow_profile_export: env_flag("LI_VOICE_ALLOW_EXPORT", d.allow_profile_export),
            watermark: env_flag("LI_VOICE_WATERMARK", d.watermark),
        }
    }

    /// Whether the consumer may attach its voice reference to a mesh chunk. Pure:
    /// remote synthesis must be allowed and (if required) consent confirmed.
    pub fn may_ship_reference(&self, consent_confirmed: bool) -> bool {
        self.allow_remote_synthesis && (!self.require_consent || consent_confirmed)
    }
}

/// Parse a boolean env flag: `0`/`false`/`no`/`off`/empty = false, any other set
/// value = true, unset = `default`.
fn env_flag(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off" | ""
        ),
        Err(_) => default,
    }
}

/// R11.4 — provenance stamped on synthesized audio: traceable *within the system*
/// that an utterance was machine-generated and from which profile. Not audible;
/// it serves auditability and the consent ethic. The clock is injected so the
/// builder stays pure/testable.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GeneratedAudioMeta {
    /// Component or node that produced the audio (e.g. a peer id, "li-mesh").
    pub generated_by: String,
    /// The cloned profile's id; `None` for the neutral voice (no timbre).
    pub voice_profile_id: Option<Uuid>,
    /// Always true for our output (we never emit "real" audio).
    pub synthetic: bool,
    pub timestamp_ms: u64,
}

impl GeneratedAudioMeta {
    /// Build provenance for a rendered frame. `profile_id` is only retained when
    /// the route actually clones a timbre; the neutral voice carries `None`.
    pub fn new(
        generated_by: impl Into<String>,
        route: VoiceRoute,
        profile_id: Option<Uuid>,
        timestamp_ms: u64,
    ) -> Self {
        Self {
            generated_by: generated_by.into(),
            voice_profile_id: if route == VoiceRoute::Clone {
                profile_id
            } else {
                None
            },
            synthetic: true,
            timestamp_ms,
        }
    }
}

/// What the pipeline asks a backend to render.
#[derive(Clone, Debug)]
pub struct VoiceSynthesisRequest {
    pub text: String,
    pub lang: Lang,
    pub profile: VoiceProfile,
    /// Render with a neutral voice, ignoring the profile timbre (no consent needed).
    pub neutral: bool,
}

/// A chunk of synthesized audio for low-latency streaming. `spec` makes the
/// buffer self-describing for the resample/PipeWire stages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioFrame {
    pub spec: AudioSpec,
    pub pcm: Vec<u8>,
}

/// Streamed synthesis output. One frame per chunk → time-to-first-chunk latency.
pub type AudioFrameStream = Pin<Box<dyn Stream<Item = Result<AudioFrame>> + Send>>;

/// Pluggable synthesis backend.
///
/// Planned implementations: `HttpQwenBackend` (clone via the current libtorch
/// service), `CandleKokoroBackend` (neutral, on-device), `CandleQwen3Backend`
/// (clone, on-device — future port), and [`MockVoiceBackend`] for tests.
#[async_trait]
pub trait VoiceSynthesisBackend: Send + Sync {
    async fn synthesize_stream(&self, req: VoiceSynthesisRequest) -> Result<AudioFrameStream>;
}

/// Deterministic test backend: emits one short PCM chunk and enforces the
/// consent gate so the policy itself is unit-testable without a model/GPU.
pub struct MockVoiceBackend {
    pub sample_rate: u32,
}

impl Default for MockVoiceBackend {
    fn default() -> Self {
        Self {
            sample_rate: 24_000,
        }
    }
}

#[async_trait]
impl VoiceSynthesisBackend for MockVoiceBackend {
    async fn synthesize_stream(&self, req: VoiceSynthesisRequest) -> Result<AudioFrameStream> {
        if !req.neutral && !req.profile.is_usable() {
            bail!("perfil vocal sin consentimiento confirmado o sin muestras");
        }
        // Placeholder PCM: 1 s16le sample per character so length is observable.
        let frame = AudioFrame {
            spec: AudioSpec::mono_s16le(self.sample_rate),
            pcm: vec![0u8; req.text.len().max(1) * 2],
        };
        Ok(Box::pin(stream::once(async move { Ok(frame) })))
    }
}

/// Decode a 16-bit PCM WAV blob into a self-describing [`AudioFrame`]. Pure
/// (no I/O), so the WAV→PCM conversion is unit-testable without a TTS service.
pub fn wav_to_audio_frame(wav: &[u8]) -> Result<AudioFrame> {
    let reader = hound::WavReader::new(Cursor::new(wav)).context("invalid WAV from TTS backend")?;
    let spec = reader.spec();
    let mut pcm = Vec::with_capacity(wav.len());
    for sample in reader.into_samples::<i16>() {
        let value = sample.context("failed reading WAV sample")?;
        pcm.extend_from_slice(&value.to_le_bytes());
    }
    Ok(AudioFrame {
        spec: AudioSpec {
            sample_rate: spec.sample_rate,
            channels: spec.channels as u8,
            format: AudioFormat::PcmS16Le,
        },
        pcm,
    })
}

/// Cloning backend over the existing Qwen3-TTS HTTP service (port 8020). One
/// external dependency behind the trait until a Candle port lands; renders the
/// user's timbre when a consented `VoiceProfile` is supplied.
pub struct HttpQwenBackend {
    client: reqwest::Client,
    base_url: String,
    model: String,
    voice: String,
}

impl HttpQwenBackend {
    pub fn new(base_url: String, model: String, voice: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url,
            model,
            voice,
        }
    }

    /// Construct from `LI_QWEN_TTS_URL`/`_MODEL`/`_VOICE` (server defaults).
    pub fn from_env() -> Self {
        Self::new(
            std::env::var("LI_QWEN_TTS_URL").unwrap_or_else(|_| "http://127.0.0.1:8020".into()),
            std::env::var("LI_QWEN_TTS_MODEL")
                .unwrap_or_else(|_| "Qwen/Qwen3-TTS-12Hz-0.6B-Base".into()),
            std::env::var("LI_QWEN_TTS_VOICE").unwrap_or_else(|_| "alloy".into()),
        )
    }

    /// Base64 voice reference + its transcript, taken from the first profile
    /// sample when cloning (not neutral).
    async fn voice_reference(
        req: &VoiceSynthesisRequest,
    ) -> Result<(Option<String>, Option<String>)> {
        if req.neutral {
            return Ok((None, None));
        }
        let Some(sample) = req.profile.samples.first() else {
            return Ok((None, None));
        };
        let bytes = tokio::fs::read(&sample.path)
            .await
            .with_context(|| format!("failed to read voice reference {}", sample.path.display()))?;
        Ok((Some(STANDARD.encode(bytes)), sample.transcript.clone()))
    }
}

#[async_trait]
impl VoiceSynthesisBackend for HttpQwenBackend {
    async fn synthesize_stream(&self, req: VoiceSynthesisRequest) -> Result<AudioFrameStream> {
        if !req.neutral && !req.profile.is_usable() {
            bail!("perfil vocal sin consentimiento confirmado o sin muestras");
        }
        let (sample_b64, sample_text) = Self::voice_reference(&req).await?;
        let request = crate::tts::QwenTtsRequest::for_language(
            &req.text,
            req.lang.tts_language(),
            &self.model,
            &self.voice,
            sample_b64.as_deref(),
            sample_text.as_deref(),
        );
        let url = format!("{}/v1/audio/speech", self.base_url.trim_end_matches('/'));
        let response = self
            .client
            .post(url)
            .json(&request)
            .send()
            .await
            .context("failed to call Qwen3-TTS endpoint")?;
        if !response.status().is_success() {
            bail!("Qwen3-TTS endpoint returned {}", response.status());
        }
        let bytes = response.bytes().await.context("failed to read TTS audio")?;
        let frame = wav_to_audio_frame(&bytes)?;
        Ok(Box::pin(stream::once(async move { Ok(frame) })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;

    fn sample() -> VoiceSample {
        VoiceSample {
            path: PathBuf::from("data/voice/reference.wav"),
            transcript: Some("En un lugar de la Mancha".into()),
            lang: Lang::Es,
            duration_ms: 4_000,
            sample_rate: 24_000,
        }
    }

    fn profile(consent: bool, samples: Vec<VoiceSample>) -> VoiceProfile {
        VoiceProfile {
            id: Uuid::nil(),
            name: "Ramón".into(),
            owner: "ramongranda@gmail.com".into(),
            consent_confirmed: consent,
            samples,
            embedding_path: None,
            default_lang: Lang::Es,
            quality_score: 0.9,
            created_at: DateTime::<Utc>::from_timestamp(0, 0).unwrap(),
        }
    }

    fn request(profile: VoiceProfile, neutral: bool) -> VoiceSynthesisRequest {
        VoiceSynthesisRequest {
            text: "hola".into(),
            lang: Lang::En,
            profile,
            neutral,
        }
    }

    #[test]
    fn is_usable_requires_consent_and_samples() {
        assert!(profile(true, vec![sample()]).is_usable());
        assert!(!profile(false, vec![sample()]).is_usable()); // no consent
        assert!(!profile(true, vec![]).is_usable()); // no samples
    }

    #[test]
    fn default_policy_matches_shipped_behavior() {
        let p = VoiceUsagePolicy::default();
        assert!(p.require_consent);
        assert!(p.allow_remote_synthesis); // cross-node cloning stays on by default
        assert!(!p.allow_profile_export);
        assert!(p.watermark);
    }

    #[test]
    fn may_ship_reference_gates_on_remote_and_consent() {
        let open = VoiceUsagePolicy::default();
        assert!(open.may_ship_reference(true));
        assert!(!open.may_ship_reference(false)); // consent required

        let local_only = VoiceUsagePolicy {
            allow_remote_synthesis: false,
            ..VoiceUsagePolicy::default()
        };
        assert!(!local_only.may_ship_reference(true)); // timbre never leaves the box

        let no_consent_gate = VoiceUsagePolicy {
            require_consent: false,
            ..VoiceUsagePolicy::default()
        };
        assert!(no_consent_gate.may_ship_reference(false)); // consent not enforced
    }

    #[test]
    fn generated_meta_keeps_profile_only_when_cloning() {
        let id = Uuid::from_u128(7);
        let clone = GeneratedAudioMeta::new("li-mesh", VoiceRoute::Clone, Some(id), 1_234);
        assert_eq!(clone.voice_profile_id, Some(id));
        assert!(clone.synthetic);
        assert_eq!(clone.timestamp_ms, 1_234);

        let neutral = GeneratedAudioMeta::new("li-mesh", VoiceRoute::Neutral, Some(id), 1_234);
        assert_eq!(neutral.voice_profile_id, None); // neutral voice carries no timbre id
        assert!(neutral.synthetic);
    }

    #[tokio::test]
    async fn mock_refuses_profile_without_consent() {
        let backend = MockVoiceBackend::default();
        let result = backend
            .synthesize_stream(request(profile(false, vec![sample()]), false))
            .await;
        assert!(result.is_err(), "consent gate must block synthesis");
    }

    #[tokio::test]
    async fn mock_renders_with_consent() {
        let backend = MockVoiceBackend::default();
        let mut frames = backend
            .synthesize_stream(request(profile(true, vec![sample()]), false))
            .await
            .expect("synthesis allowed with consent");
        let first = frames.next().await.expect("one frame").expect("ok frame");
        assert_eq!(first.spec.sample_rate, 24_000);
        assert_eq!(first.spec.channels, 1);
        assert_eq!(first.pcm.len(), "hola".len() * 2);
    }

    #[tokio::test]
    async fn mock_renders_neutral_without_consent() {
        let backend = MockVoiceBackend::default();
        // neutral = true bypasses the profile timbre and its consent requirement.
        let result = backend
            .synthesize_stream(request(profile(false, vec![]), true))
            .await;
        assert!(result.is_ok(), "neutral rendering needs no profile consent");
    }

    #[test]
    fn route_my_profile_clones_local_and_neutral_remote() {
        // Outgoing voice = my timbre; incoming monitor = neutral Kokoro.
        assert_eq!(
            route_for_lane(VoiceIdentity::MyProfile, Lane::Local),
            VoiceRoute::Clone
        );
        assert_eq!(
            route_for_lane(VoiceIdentity::MyProfile, Lane::Remote),
            VoiceRoute::Neutral
        );
    }

    #[test]
    fn route_neutral_and_off_apply_to_all_lanes() {
        for lane in [Lane::Local, Lane::Remote] {
            assert_eq!(
                route_for_lane(VoiceIdentity::Neutral, lane),
                VoiceRoute::Neutral
            );
            assert_eq!(route_for_lane(VoiceIdentity::Off, lane), VoiceRoute::Off);
        }
    }

    #[test]
    fn wav_to_audio_frame_decodes_spec_and_pcm() {
        let mut buf = Vec::new();
        {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 24_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::new(Cursor::new(&mut buf), spec).unwrap();
            for sample in [0i16, 100, -100, 200] {
                writer.write_sample(sample).unwrap();
            }
            writer.finalize().unwrap();
        }
        let frame = wav_to_audio_frame(&buf).expect("decode wav");
        assert_eq!(frame.spec.sample_rate, 24_000);
        assert_eq!(frame.spec.channels, 1);
        assert_eq!(frame.spec.format, AudioFormat::PcmS16Le);
        assert_eq!(frame.pcm.len(), 4 * 2); // 4 i16 samples → 8 bytes
    }

    #[test]
    fn voice_profile_bincode_roundtrip() {
        let original = profile(true, vec![sample()]);
        let bytes = bincode::serialize(&original).expect("ser");
        let decoded: VoiceProfile = bincode::deserialize(&bytes).expect("de");
        assert_eq!(original, decoded);
    }
}
