//! Interpretation pipeline core: one utterance → transcript → translation →
//! synthesized (optionally cloned) voice → virtual mic, emitting `PipelineEvent`s.
//!
//! Trait-based so the orchestration is unit-testable with mocks (no GPU, no
//! network, no audio device). The real engines (`AsrEngine`, `Translator`)
//! implement the traits; `VoiceSynthesisBackend` and `AudioOutput` already exist.

use crate::types::{Direction, Lane, Lang, PipelineEvent};
use crate::virtual_mic::AudioOutput;
use crate::voice::{
    VoiceIdentity, VoiceProfile, VoiceRoute, VoiceSynthesisBackend, route_for_lane,
};
use anyhow::Result;
use async_trait::async_trait;
use futures_util::StreamExt;
use std::path::Path;
use uuid::Uuid;

/// Speech → text. Implemented by the real Whisper engine and by mocks.
#[async_trait]
pub trait Transcriber: Send + Sync {
    async fn transcribe(&self, wav: &Path, source_lang: &str) -> Result<String>;
}

/// Text → translated text. Implemented by the real Ollama translator and mocks.
#[async_trait]
pub trait TextTranslator: Send + Sync {
    async fn translate(&self, text: &str, direction: &Direction) -> Result<String>;
}

#[async_trait]
impl Transcriber for crate::asr::AsrEngine {
    async fn transcribe(&self, wav: &Path, source_lang: &str) -> Result<String> {
        let segments = self.transcribe_file(wav, source_lang).await?;
        Ok(segments
            .iter()
            .map(|segment| segment.text.trim())
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_string())
    }
}

#[async_trait]
impl TextTranslator for crate::translate::Translator {
    async fn translate(&self, text: &str, direction: &Direction) -> Result<String> {
        crate::translate::Translator::translate(self, text, direction).await
    }
}

/// Source language for a direction (the language being spoken).
pub fn source_lang(direction: &Direction) -> Lang {
    match direction {
        Direction::EsToEn => Lang::Es,
        Direction::EnToEs => Lang::En,
    }
}

/// Target language for a direction (the language being synthesized).
pub fn target_lang(direction: &Direction) -> Lang {
    match direction {
        Direction::EsToEn => Lang::En,
        Direction::EnToEs => Lang::Es,
    }
}

/// Run one utterance through the full pipeline, pushing synthesized audio to the
/// virtual mic and returning the ordered `PipelineEvent`s (for broadcast/UI).
///
/// `latency_ms` is supplied by the caller (the pure core takes no clock) so the
/// function stays deterministic and testable.
#[allow(clippy::too_many_arguments)]
pub async fn interpret_utterance(
    asr: &dyn Transcriber,
    translator: &dyn TextTranslator,
    voice: &dyn VoiceSynthesisBackend,
    mic: &dyn AudioOutput,
    profile: &VoiceProfile,
    wav: &Path,
    direction: Direction,
    identity: VoiceIdentity,
    lane: Lane,
    id: Uuid,
    latency_ms: u64,
) -> Result<Vec<PipelineEvent>> {
    let mut events = vec![PipelineEvent::Processing { id, lane }];

    let transcript = asr.transcribe(wav, direction.source_lang()).await?;
    events.push(PipelineEvent::Transcript {
        id,
        lane,
        lang: source_lang(&direction),
        text: transcript.clone(),
    });

    let translation = translator.translate(&transcript, &direction).await?;
    events.push(PipelineEvent::Translation {
        id,
        lane,
        lang: target_lang(&direction),
        text: translation.clone(),
    });

    let route = route_for_lane(identity, lane);
    render_chunks(
        voice,
        mic,
        route,
        profile,
        target_lang(&direction),
        &[translation],
        id,
        lane,
        &mut events,
    )
    .await?;

    events.push(PipelineEvent::Done {
        id,
        lane,
        latency_ms,
    });
    Ok(events)
}

/// Chunked variant: split the translation into clauses and synthesize each
/// independently, so the virtual mic starts playing the first clause while the
/// next is still being synthesized (lower time-to-first-audio). Same trait core.
#[allow(clippy::too_many_arguments)]
pub async fn interpret_utterance_chunked(
    asr: &dyn Transcriber,
    translator: &dyn TextTranslator,
    voice: &dyn VoiceSynthesisBackend,
    mic: &dyn AudioOutput,
    profile: &VoiceProfile,
    wav: &Path,
    direction: Direction,
    identity: VoiceIdentity,
    lane: Lane,
    id: Uuid,
    latency_ms: u64,
) -> Result<Vec<PipelineEvent>> {
    let mut events = vec![PipelineEvent::Processing { id, lane }];

    let transcript = asr.transcribe(wav, direction.source_lang()).await?;
    events.push(PipelineEvent::Transcript {
        id,
        lane,
        lang: source_lang(&direction),
        text: transcript.clone(),
    });

    let translation = translator.translate(&transcript, &direction).await?;
    events.push(PipelineEvent::Translation {
        id,
        lane,
        lang: target_lang(&direction),
        text: translation.clone(),
    });

    let route = route_for_lane(identity, lane);
    let clauses = split_clauses(&translation);
    render_chunks(
        voice,
        mic,
        route,
        profile,
        target_lang(&direction),
        &clauses,
        id,
        lane,
        &mut events,
    )
    .await?;

    events.push(PipelineEvent::Done {
        id,
        lane,
        latency_ms,
    });
    Ok(events)
}

/// Synthesize each text chunk in order and push its audio to the mic as it is
/// produced. With one chunk this is the single-shot path; with several it
/// overlaps playback with synthesis of later chunks.
#[allow(clippy::too_many_arguments)]
async fn render_chunks(
    voice: &dyn VoiceSynthesisBackend,
    mic: &dyn AudioOutput,
    route: VoiceRoute,
    profile: &VoiceProfile,
    lang: Lang,
    chunks: &[String],
    id: Uuid,
    lane: Lane,
    events: &mut Vec<PipelineEvent>,
) -> Result<()> {
    if route == VoiceRoute::Off {
        return Ok(());
    }
    for chunk in chunks {
        if chunk.trim().is_empty() {
            continue;
        }
        let request = crate::voice::VoiceSynthesisRequest {
            text: chunk.clone(),
            lang,
            profile: profile.clone(),
            neutral: route == VoiceRoute::Neutral,
        };
        let mut stream = voice.synthesize_stream(request).await?;
        while let Some(frame) = stream.next().await {
            let frame = frame?;
            mic.submit(&frame)?;
            events.push(PipelineEvent::AudioFrame {
                id,
                lane,
                spec: frame.spec,
                pcm: frame.pcm,
            });
        }
    }
    Ok(())
}

/// Split text into clause-sized chunks for low-latency synthesis: the first
/// chunk is spoken while later chunks are still being synthesized, so the metric
/// that matters (time-to-first-audio) tracks the *first* chunk, not the whole
/// utterance.
///
/// The TTS service can't stream and its latency scales with text length (≈linear
/// after a ~1s fixed per-call overhead — measured via the R11.1 bench), so the
/// split balances two pressures: chunks too short waste the fixed overhead and
/// sound choppy; chunks too long delay the first audio. It breaks on
///
/// * **sentence terminators** (`.!?;…`) — always;
/// * **soft separators** (`,:—`) — only once the chunk is long enough to be worth
///   a separate call (`MIN_SOFT_CHARS`), so "Hello, ..." doesn't split off "Hello,";
/// * **length** — a run-on with no usable punctuation is force-broken at a word
///   boundary near `MAX_CHARS`, bounding first-audio regardless of punctuation.
///
/// A runt trailing chunk is merged back into the previous one. Pure; no-punctuation
/// short text yields a single chunk (preserving the original behavior).
pub fn split_clauses(text: &str) -> Vec<String> {
    // Per-call TTS overhead is ~1s, so don't split below MIN_SOFT_CHARS; above
    // MAX_CHARS the first chunk takes too long, so force a word-boundary break.
    const MIN_SOFT_CHARS: usize = 24;
    const MAX_CHARS: usize = 72;
    // The FIRST clause alone gates time-to-first-audio (the conversational
    // latency), so split it at the first soft separator (e.g. an opening
    // "Hello,") even when short — one extra ~1s TTS call buys the first audio
    // sooner. Later clauses keep MIN_SOFT_CHARS since they synthesize while the
    // first one is already playing. Max stays uniform to avoid unnatural
    // mid-sentence breaks on the opening.
    const FIRST_SOFT_CHARS: usize = 6;

    let mut clauses = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        let len = current.trim().chars().count();
        let soft_min = if clauses.is_empty() {
            FIRST_SOFT_CHARS
        } else {
            MIN_SOFT_CHARS
        };
        let hard = matches!(ch, '.' | '!' | '?' | ';' | '…');
        let soft = matches!(ch, ',' | ':' | '—') && len >= soft_min;
        let too_long = len >= MAX_CHARS && ch.is_whitespace();
        if hard || soft || too_long {
            let trimmed = current.trim();
            if !trimmed.is_empty() {
                clauses.push(trimmed.to_string());
            }
            current.clear();
        }
    }
    let tail = current.trim().to_string();
    if !tail.is_empty() {
        // Merge a runt tail into the previous chunk rather than emit a tiny call.
        match clauses.last_mut() {
            Some(last) if tail.chars().count() < MIN_SOFT_CHARS => {
                last.push(' ');
                last.push_str(&tail);
            }
            _ => clauses.push(tail),
        }
    }
    if clauses.is_empty() {
        let whole = text.trim();
        if !whole.is_empty() {
            clauses.push(whole.to_string());
        }
    }
    clauses
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::virtual_mic::MockAudioOutput;
    use crate::voice::{MockVoiceBackend, VoiceSample};
    use chrono::{DateTime, Utc};
    use std::path::PathBuf;

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

    fn consented_profile() -> VoiceProfile {
        VoiceProfile {
            id: Uuid::nil(),
            name: "Ramón".into(),
            owner: "self".into(),
            consent_confirmed: true,
            samples: vec![VoiceSample {
                path: PathBuf::from("ref.wav"),
                transcript: Some("hola".into()),
                lang: Lang::Es,
                duration_ms: 1000,
                sample_rate: 24_000,
            }],
            embedding_path: None,
            default_lang: Lang::Es,
            quality_score: 1.0,
            created_at: DateTime::<Utc>::from_timestamp(0, 0).unwrap(),
        }
    }

    #[tokio::test]
    async fn neutral_lane_emits_full_event_sequence_and_feeds_mic() {
        let asr = MockAsr("hola mundo");
        let translator = MockTranslator("hello world");
        let voice = MockVoiceBackend::default();
        let mic = MockAudioOutput::default();
        let profile = consented_profile();

        let events = interpret_utterance(
            &asr,
            &translator,
            &voice,
            &mic,
            &profile,
            Path::new("utterance.wav"),
            Direction::EsToEn,
            VoiceIdentity::Neutral,
            Lane::Local,
            Uuid::nil(),
            120,
        )
        .await
        .expect("pipeline ok");

        // Processing → Transcript(es) → Translation(en) → AudioFrame → Done.
        assert!(matches!(events[0], PipelineEvent::Processing { .. }));
        assert!(matches!(
            &events[1],
            PipelineEvent::Transcript { lang: Lang::Es, text, .. } if text == "hola mundo"
        ));
        assert!(matches!(
            &events[2],
            PipelineEvent::Translation { lang: Lang::En, text, .. } if text == "hello world"
        ));
        assert!(matches!(events[3], PipelineEvent::AudioFrame { .. }));
        assert!(matches!(
            events[4],
            PipelineEvent::Done {
                latency_ms: 120,
                ..
            }
        ));
        assert_eq!(mic.frame_count(), 1);
        assert!(mic.total_pcm_bytes() > 0);
    }

    #[test]
    fn split_clauses_breaks_on_terminators_and_keeps_them() {
        assert_eq!(
            split_clauses("Hello world. How are you? Fine!"),
            vec!["Hello world.", "How are you?", "Fine!"]
        );
        // No terminator → single chunk.
        assert_eq!(split_clauses("just a phrase"), vec!["just a phrase"]);
        // Whitespace/empty → no empty chunks.
        assert!(split_clauses("   ").is_empty());
    }

    #[test]
    fn split_clauses_breaks_on_long_commas_but_not_short_ones() {
        // Comma after a long-enough run → a separate chunk (lower first-audio).
        assert_eq!(
            split_clauses("quiero probar el sistema de hoy, y medir la latencia del audio."),
            vec![
                "quiero probar el sistema de hoy,",
                "y medir la latencia del audio."
            ]
        );
        // Early short comma → not worth its own TTS call, stays in one chunk.
        assert_eq!(
            split_clauses("Hola, bienvenido."),
            vec!["Hola, bienvenido."]
        );
    }

    #[test]
    fn split_clauses_first_clause_splits_on_early_comma_for_low_ttfa() {
        // The opening "Hello," (>=6 chars) becomes its own chunk so the first
        // audio lands ASAP; later clauses keep the larger MIN_SOFT_CHARS sizing.
        assert_eq!(
            split_clauses("Hello, good morning to all. Let us begin."),
            vec!["Hello,", "good morning to all.", "Let us begin."]
        );
    }

    #[test]
    fn split_clauses_force_breaks_run_on_without_punctuation() {
        // A long run-on with no usable punctuation must still chunk, so the first
        // audio isn't gated on synthesizing the whole thing.
        let run_on = "uno dos tres cuatro cinco seis siete ocho nueve diez once doce \
                      trece catorce quince dieciseis diecisiete dieciocho";
        let chunks = split_clauses(run_on);
        assert!(chunks.len() >= 2, "run-on should force-break: {chunks:?}");
        assert!(
            chunks[0].chars().count() <= 80,
            "first chunk bounds first-audio: {:?}",
            chunks[0]
        );
    }

    #[test]
    fn split_clauses_merges_runt_tail() {
        // A tiny trailing fragment folds into the previous chunk (no 1s call for "ok").
        assert_eq!(
            split_clauses("This is a reasonably long first sentence here. ok"),
            vec!["This is a reasonably long first sentence here. ok"]
        );
    }

    #[tokio::test]
    async fn chunked_emits_one_audioframe_per_clause() {
        let mic = MockAudioOutput::default();
        let events = interpret_utterance_chunked(
            &MockAsr("hola. mundo."),
            &MockTranslator("Hello. World."),
            &MockVoiceBackend::default(),
            &mic,
            &consented_profile(),
            Path::new("u.wav"),
            Direction::EsToEn,
            VoiceIdentity::Neutral,
            Lane::Local,
            Uuid::nil(),
            0,
        )
        .await
        .expect("ok");

        // Two clauses → two synthesis calls → two frames to the mic.
        let frames = events
            .iter()
            .filter(|e| matches!(e, PipelineEvent::AudioFrame { .. }))
            .count();
        assert_eq!(frames, 2);
        assert_eq!(mic.frame_count(), 2);
    }

    #[tokio::test]
    async fn off_identity_skips_synthesis_and_mic() {
        let mic = MockAudioOutput::default();
        let events = interpret_utterance(
            &MockAsr("hola"),
            &MockTranslator("hi"),
            &MockVoiceBackend::default(),
            &mic,
            &consented_profile(),
            Path::new("u.wav"),
            Direction::EsToEn,
            VoiceIdentity::Off,
            Lane::Local,
            Uuid::nil(),
            0,
        )
        .await
        .expect("ok");

        assert!(
            !events
                .iter()
                .any(|e| matches!(e, PipelineEvent::AudioFrame { .. }))
        );
        assert_eq!(mic.frame_count(), 0, "Off must not touch the mic");
    }

    #[tokio::test]
    async fn clone_lane_requires_consent() {
        // MyProfile + Local → Clone (neutral=false); an unconsented profile must error.
        let mut profile = consented_profile();
        profile.consent_confirmed = false;
        let result = interpret_utterance(
            &MockAsr("hola"),
            &MockTranslator("hi"),
            &MockVoiceBackend::default(),
            &MockAudioOutput::default(),
            &profile,
            Path::new("u.wav"),
            Direction::EsToEn,
            VoiceIdentity::MyProfile,
            Lane::Local,
            Uuid::nil(),
            0,
        )
        .await;
        assert!(result.is_err(), "cloning without consent must fail");
    }
}
