//! Bridge between the libp2p mesh (`mesh.rs`) and the interpretation pipeline
//! (`pipeline.rs`).
//!
//! A GPU **provider** node receives an [`AudioChunk`](crate::mesh::AudioChunk)
//! (raw `f32` samples) from a **consumer**, runs the full
//! transcribe → translate → synthesize pipeline, and returns an
//! [`AudioTaskResult`](crate::mesh::AudioTaskResult) carrying the translated
//! voice as `f32` PCM. The consumer plays that into its virtual mic — so the
//! *translated voice travels to another node* and back (R10).
//!
//! Kept separate from both `mesh.rs` (pure libp2p) and `pipeline.rs` (pure
//! orchestration) for SRP. The PCM conversions and event→result projection are
//! pure and unit-tested; only `process` does I/O.
//!
//! Cross-node cloning: when a chunk carries a [`VoiceReference`] the provider
//! builds a consent-gated profile and caches it per `session_id`, so the
//! consumer's *own timbre* travels over the wire and renders the translation —
//! later (reference-less) chunks reuse the cache. Without a reference it falls
//! back to the provider's configured profile/identity (neutral by default).

use crate::mesh::{AudioChunk, AudioTaskResult, MeshAudioProcessor, VoiceReference};
use crate::pipeline::{TextTranslator, Transcriber, interpret_utterance};
use crate::types::{Lane, Lang, PipelineEvent};
use crate::virtual_mic::MockAudioOutput;
use crate::voice::{VoiceIdentity, VoiceProfile, VoiceSample, VoiceSynthesisBackend};
use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

/// Decode interleaved s16le bytes into normalized `f32` samples.
pub fn pcm_s16le_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32_768.0)
        .collect()
}

/// Encode normalized `f32` samples into interleaved s16le bytes.
pub fn f32_to_pcm_s16le(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// The transcript/translation/audio distilled from a pipeline run.
#[derive(Clone, Debug, PartialEq)]
pub struct InterpretedAudio {
    pub transcription: String,
    pub translation: String,
    pub tts_sample_rate_hz: u32,
    pub tts_pcm: Vec<u8>,
}

/// Project the ordered [`PipelineEvent`]s of one utterance into the fields the
/// mesh result carries. Pure: the last `Transcript`/`Translation` wins and all
/// `AudioFrame` PCM is concatenated; `fallback_rate` is used when no audio was
/// produced (e.g. `VoiceIdentity::Off`).
pub fn collect_interpreted(events: &[PipelineEvent], fallback_rate: u32) -> InterpretedAudio {
    let mut transcription = String::new();
    let mut translation = String::new();
    let mut tts_sample_rate_hz = fallback_rate;
    let mut tts_pcm = Vec::new();
    for event in events {
        match event {
            PipelineEvent::Transcript { text, .. } => transcription = text.clone(),
            PipelineEvent::Translation { text, .. } => translation = text.clone(),
            PipelineEvent::AudioFrame { spec, pcm, .. } => {
                tts_sample_rate_hz = spec.sample_rate;
                tts_pcm.extend_from_slice(pcm);
            }
            _ => {}
        }
    }
    InterpretedAudio {
        transcription,
        translation,
        tts_sample_rate_hz,
        tts_pcm,
    }
}

/// Build a consent-gated profile (and its routing identity) from a voice
/// reference shipped over the mesh, writing the reference WAV to `wav_path` so
/// the synthesis backend can read it. A reference without consent renders neutral.
pub fn profile_from_reference(
    reference: &VoiceReference,
    wav_path: &Path,
    session_id: Uuid,
) -> Result<(VoiceProfile, VoiceIdentity)> {
    crate::capture::write_wav_16le(wav_path, &reference.samples, reference.sample_rate_hz)?;
    let profile = VoiceProfile {
        id: session_id,
        name: "mesh-peer".into(),
        owner: "mesh-peer".into(),
        consent_confirmed: reference.consent_confirmed,
        samples: vec![VoiceSample {
            path: wav_path.to_path_buf(),
            transcript: reference.transcript.clone(),
            lang: Lang::Es,
            duration_ms: 0,
            sample_rate: reference.sample_rate_hz,
        }],
        embedding_path: None,
        default_lang: Lang::Es,
        quality_score: 1.0,
        created_at: chrono::Utc::now(),
    };
    let identity = if profile.is_usable() {
        VoiceIdentity::MyProfile
    } else {
        VoiceIdentity::Neutral
    };
    Ok((profile, identity))
}

/// Runs the real interpretation pipeline for mesh audio tasks (provider side).
///
/// When a chunk carries a [`VoiceReference`] the provider builds and caches the
/// consumer's profile per `session_id`, so the translation is rendered in the
/// consumer's timbre on later (reference-less) chunks too. Without any reference
/// it falls back to the provider's own configured profile/identity.
pub struct PipelineMeshProcessor {
    asr: Arc<dyn Transcriber>,
    translator: Arc<dyn TextTranslator>,
    voice: Arc<dyn VoiceSynthesisBackend>,
    fallback_profile: VoiceProfile,
    fallback_identity: VoiceIdentity,
    uploads_dir: PathBuf,
    voice_dir: PathBuf,
    session_profiles: Mutex<HashMap<Uuid, (VoiceProfile, VoiceIdentity)>>,
}

impl PipelineMeshProcessor {
    pub fn new(
        asr: Arc<dyn Transcriber>,
        translator: Arc<dyn TextTranslator>,
        voice: Arc<dyn VoiceSynthesisBackend>,
        fallback_profile: VoiceProfile,
        fallback_identity: VoiceIdentity,
        uploads_dir: PathBuf,
        voice_dir: PathBuf,
    ) -> Self {
        Self {
            asr,
            translator,
            voice,
            fallback_profile,
            fallback_identity,
            uploads_dir,
            voice_dir,
            session_profiles: Mutex::new(HashMap::new()),
        }
    }

    /// Resolve which (profile, identity) renders this chunk: a fresh reference
    /// updates the per-session cache; otherwise use the cached or fallback one.
    fn resolve_voice(&self, chunk: &AudioChunk) -> Result<(VoiceProfile, VoiceIdentity)> {
        if let Some(reference) = &chunk.voice_ref {
            std::fs::create_dir_all(&self.voice_dir).ok();
            let path = self
                .voice_dir
                .join(format!("mesh-ref-{}.wav", chunk.session_id));
            let resolved = profile_from_reference(reference, &path, chunk.session_id)?;
            self.session_profiles
                .lock()
                .unwrap()
                .insert(chunk.session_id, resolved.clone());
            return Ok(resolved);
        }
        if let Some(cached) = self
            .session_profiles
            .lock()
            .unwrap()
            .get(&chunk.session_id)
            .cloned()
        {
            return Ok(cached);
        }
        Ok((self.fallback_profile.clone(), self.fallback_identity))
    }
}

#[async_trait]
impl MeshAudioProcessor for PipelineMeshProcessor {
    async fn process(&self, chunk: AudioChunk) -> Result<AudioTaskResult> {
        let (profile, identity) = self.resolve_voice(&chunk)?;

        std::fs::create_dir_all(&self.uploads_dir).ok();
        let wav = self
            .uploads_dir
            .join(format!("mesh-{}-{}.wav", chunk.session_id, chunk.sequence));
        crate::capture::write_wav_16le(&wav, &chunk.samples, chunk.sample_rate_hz)?;

        // The provider produces audio for the remote consumer; the synthesized
        // PCM is read back from the events, so a no-op sink is fine here.
        let sink = MockAudioOutput::default();
        let result = interpret_utterance(
            &*self.asr,
            &*self.translator,
            &*self.voice,
            &sink,
            &profile,
            &wav,
            chunk.direction,
            identity,
            Lane::Local,
            Uuid::new_v4(),
            0,
        )
        .await;
        let _ = std::fs::remove_file(&wav);
        let events = result?;

        let parts = collect_interpreted(&events, chunk.sample_rate_hz);
        Ok(AudioTaskResult {
            session_id: chunk.session_id,
            sequence: chunk.sequence,
            transcription: parts.transcription,
            translation: parts.translation,
            tts_sample_rate_hz: parts.tts_sample_rate_hz,
            tts_output: pcm_s16le_to_f32(&parts.tts_pcm),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AudioSpec, Direction, Lang};
    use crate::voice::{MockVoiceBackend, VoiceSample};
    use chrono::{DateTime, Utc};
    use std::path::Path;

    #[test]
    fn pcm_f32_roundtrip_is_near_lossless() {
        let original = [0.0f32, 0.5, -0.5, 1.0, -1.0];
        let bytes = f32_to_pcm_s16le(&original);
        assert_eq!(bytes.len(), original.len() * 2);
        let decoded = pcm_s16le_to_f32(&bytes);
        for (a, b) in original.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn collect_interpreted_takes_text_and_concats_audio() {
        let id = Uuid::nil();
        let spec = AudioSpec::mono_s16le(24_000);
        let events = vec![
            PipelineEvent::Processing {
                id,
                lane: Lane::Local,
            },
            PipelineEvent::Transcript {
                id,
                lane: Lane::Local,
                lang: Lang::Es,
                text: "hola".into(),
            },
            PipelineEvent::Translation {
                id,
                lane: Lane::Local,
                lang: Lang::En,
                text: "hi".into(),
            },
            PipelineEvent::AudioFrame {
                id,
                lane: Lane::Local,
                spec,
                pcm: vec![1, 2],
            },
            PipelineEvent::AudioFrame {
                id,
                lane: Lane::Local,
                spec,
                pcm: vec![3, 4],
            },
            PipelineEvent::Done {
                id,
                lane: Lane::Local,
                latency_ms: 0,
            },
        ];
        let got = collect_interpreted(&events, 16_000);
        assert_eq!(got.transcription, "hola");
        assert_eq!(got.translation, "hi");
        assert_eq!(got.tts_sample_rate_hz, 24_000);
        assert_eq!(got.tts_pcm, vec![1, 2, 3, 4]);
    }

    #[test]
    fn collect_interpreted_uses_fallback_rate_without_audio() {
        let got = collect_interpreted(&[], 16_000);
        assert_eq!(got.tts_sample_rate_hz, 16_000);
        assert!(got.tts_pcm.is_empty());
    }

    struct MockAsr(&'static str);
    #[async_trait]
    impl Transcriber for MockAsr {
        async fn transcribe(&self, _wav: &Path, _lang: &str) -> Result<String> {
            Ok(self.0.to_string())
        }
    }

    struct MockTranslator(&'static str);
    #[async_trait]
    impl TextTranslator for MockTranslator {
        async fn translate(&self, _text: &str, _dir: &Direction) -> Result<String> {
            Ok(self.0.to_string())
        }
    }

    fn neutral_profile() -> VoiceProfile {
        VoiceProfile {
            id: Uuid::nil(),
            name: "neutral".into(),
            owner: "node".into(),
            consent_confirmed: false,
            samples: Vec::<VoiceSample>::new(),
            embedding_path: None,
            default_lang: Lang::En,
            quality_score: 0.0,
            created_at: DateTime::<Utc>::from_timestamp(0, 0).unwrap(),
        }
    }

    fn processor() -> PipelineMeshProcessor {
        PipelineMeshProcessor::new(
            Arc::new(MockAsr("hola mundo")),
            Arc::new(MockTranslator("hello world")),
            Arc::new(MockVoiceBackend::default()),
            neutral_profile(),
            VoiceIdentity::Neutral,
            std::env::temp_dir().join("li-mesh-test-uploads"),
            std::env::temp_dir().join("li-mesh-test-voice"),
        )
    }

    fn chunk(session_id: Uuid, sequence: u64, voice_ref: Option<VoiceReference>) -> AudioChunk {
        AudioChunk {
            session_id,
            sequence,
            sample_rate_hz: 16_000,
            direction: Direction::EsToEn,
            samples: vec![0.0; 1600],
            voice_ref,
            auth_token: None,
        }
    }

    #[tokio::test]
    async fn processor_runs_pipeline_and_returns_translated_audio() {
        let result = processor()
            .process(chunk(Uuid::new_v4(), 7, None))
            .await
            .expect("process ok");
        assert_eq!(result.sequence, 7);
        assert_eq!(result.transcription, "hola mundo");
        assert_eq!(result.translation, "hello world");
        assert_eq!(result.tts_sample_rate_hz, 24_000);
        // MockVoiceBackend emits 1 s16le sample per character of "hello world".
        assert!(!result.tts_output.is_empty());
    }

    #[test]
    fn profile_from_reference_clones_with_consent_neutral_without() {
        let path = std::env::temp_dir().join("li-ref-consent.wav");
        let consented = VoiceReference {
            sample_rate_hz: 24_000,
            samples: vec![0.0; 100],
            transcript: Some("hola".into()),
            consent_confirmed: true,
        };
        let (profile, identity) =
            profile_from_reference(&consented, &path, Uuid::nil()).expect("build");
        assert!(profile.is_usable());
        assert_eq!(identity, VoiceIdentity::MyProfile);

        let unconsented = VoiceReference {
            consent_confirmed: false,
            ..consented
        };
        let (_, identity) =
            profile_from_reference(&unconsented, &path, Uuid::nil()).expect("build");
        assert_eq!(identity, VoiceIdentity::Neutral);
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn reference_is_cached_per_session_for_later_chunks() {
        let processor = processor();
        let session = Uuid::new_v4();
        let reference = VoiceReference {
            sample_rate_hz: 24_000,
            samples: vec![0.1; 200],
            transcript: Some("hola".into()),
            consent_confirmed: true,
        };
        // First chunk carries the reference → session cached as MyProfile/clone.
        processor
            .process(chunk(session, 0, Some(reference)))
            .await
            .expect("seq0 ok");
        let (_, identity) = processor.resolve_voice(&chunk(session, 1, None)).unwrap();
        assert_eq!(
            identity,
            VoiceIdentity::MyProfile,
            "later chunks reuse the cached clone"
        );

        // A different session with no reference falls back to neutral.
        let (_, identity) = processor
            .resolve_voice(&chunk(Uuid::new_v4(), 0, None))
            .unwrap();
        assert_eq!(identity, VoiceIdentity::Neutral);
    }
}
