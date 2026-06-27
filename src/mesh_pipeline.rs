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
//! Limitation: the provider renders with *its own* configured `VoiceProfile` /
//! `VoiceIdentity` (the consumer's reference timbre is not yet transmitted over
//! the wire), so cross-node cloning defaults to the neutral voice unless the
//! provider is configured with a profile. Carrying the profile in `AudioChunk`
//! is a follow-up.

use crate::mesh::{AudioChunk, AudioTaskResult, MeshAudioProcessor};
use crate::pipeline::{TextTranslator, Transcriber, interpret_utterance};
use crate::types::{Lane, PipelineEvent};
use crate::virtual_mic::MockAudioOutput;
use crate::voice::{VoiceIdentity, VoiceProfile, VoiceSynthesisBackend};
use anyhow::Result;
use async_trait::async_trait;
use std::path::PathBuf;
use std::sync::Arc;
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

/// Runs the real interpretation pipeline for mesh audio tasks (provider side).
pub struct PipelineMeshProcessor {
    asr: Arc<dyn Transcriber>,
    translator: Arc<dyn TextTranslator>,
    voice: Arc<dyn VoiceSynthesisBackend>,
    profile: VoiceProfile,
    identity: VoiceIdentity,
    uploads_dir: PathBuf,
}

impl PipelineMeshProcessor {
    pub fn new(
        asr: Arc<dyn Transcriber>,
        translator: Arc<dyn TextTranslator>,
        voice: Arc<dyn VoiceSynthesisBackend>,
        profile: VoiceProfile,
        identity: VoiceIdentity,
        uploads_dir: PathBuf,
    ) -> Self {
        Self {
            asr,
            translator,
            voice,
            profile,
            identity,
            uploads_dir,
        }
    }
}

#[async_trait]
impl MeshAudioProcessor for PipelineMeshProcessor {
    async fn process(&self, chunk: AudioChunk) -> Result<AudioTaskResult> {
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
            &self.profile,
            &wav,
            chunk.direction,
            self.identity,
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

    #[tokio::test]
    async fn processor_runs_pipeline_and_returns_translated_audio() {
        let processor = PipelineMeshProcessor::new(
            Arc::new(MockAsr("hola mundo")),
            Arc::new(MockTranslator("hello world")),
            Arc::new(MockVoiceBackend::default()),
            neutral_profile(),
            VoiceIdentity::Neutral,
            std::env::temp_dir().join("li-mesh-test-uploads"),
        );
        let chunk = AudioChunk {
            session_id: Uuid::new_v4(),
            sequence: 7,
            sample_rate_hz: 16_000,
            direction: Direction::EsToEn,
            samples: vec![0.0; 1600],
        };
        let result = processor.process(chunk).await.expect("process ok");
        assert_eq!(result.sequence, 7);
        assert_eq!(result.transcription, "hola mundo");
        assert_eq!(result.translation, "hello world");
        assert_eq!(result.tts_sample_rate_hz, 24_000);
        // MockVoiceBackend emits 1 s16le sample per character of "hello world".
        assert!(!result.tts_output.is_empty());
    }
}
