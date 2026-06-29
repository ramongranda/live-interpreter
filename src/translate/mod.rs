//! Translation backends.
//!
//! The server only needs `translate(text, direction) -> String`. We keep that surface concrete
//! (an enum, not `dyn`) so `AppState` stays `Clone` and we avoid an `async-trait` dependency in a
//! codebase that otherwise uses native `async fn` in traits. Backends:
//!
//! - [`HttpTranslator`]: the Ollama HTTP client (default).
//! - [`CandleTranslator`] (feature `candle-translate`): in-process quantized Qwen2 GGUF — no
//!   external Ollama process, frees its VRAM. Selected with `LI_TRANSLATE_BACKEND=candle`.
//!
//! `translate_stream` yields the full translation as one chunk (non-streaming) on both backends,
//! used by `benches/translate_latency.rs`.

use crate::types::Direction;
use anyhow::Result;
use futures_util::Stream;
use serde::{Deserialize, Serialize};
use std::{collections::VecDeque, pin::Pin, time::Duration};

mod http;
pub use http::HttpTranslator;

#[cfg(feature = "candle-translate")]
mod candle;
#[cfg(feature = "candle-translate")]
pub use candle::CandleTranslator;

/// Boxed token stream. First `.next().await` marks time-to-first-token.
pub type TokenStream = Pin<Box<dyn Stream<Item = Result<String>> + Send>>;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TranslationTurn {
    pub original: String,
    pub translated: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TranslationBufferConfig {
    pub max_interactions: usize,
    pub silence_reset_ms: u64,
    pub max_chars_per_text: usize,
}

impl Default for TranslationBufferConfig {
    fn default() -> Self {
        Self {
            max_interactions: 8,
            silence_reset_ms: 12_000,
            max_chars_per_text: 280,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TranslationBuffer {
    config: TranslationBufferConfig,
    turns: VecDeque<TranslationTurn>,
}

impl TranslationBuffer {
    pub fn new(config: TranslationBufferConfig) -> Self {
        Self {
            config,
            turns: VecDeque::with_capacity(config.max_interactions),
        }
    }

    pub fn push(&mut self, original: impl Into<String>, translated: impl Into<String>) {
        if self.config.max_interactions == 0 {
            return;
        }

        while self.turns.len() >= self.config.max_interactions {
            self.turns.pop_front();
        }

        self.turns.push_back(TranslationTurn {
            original: clamp_chars(original.into(), self.config.max_chars_per_text),
            translated: clamp_chars(translated.into(), self.config.max_chars_per_text),
        });
    }

    pub fn observe_silence(&mut self, silence: Duration) {
        if silence.as_millis() >= u128::from(self.config.silence_reset_ms) {
            self.clear();
        }
    }

    pub fn clear(&mut self) {
        self.turns.clear();
        self.turns.shrink_to(self.config.max_interactions);
    }

    pub fn len(&self) -> usize {
        self.turns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    pub fn system_prompt(&self, direction: &Direction) -> String {
        let mut prompt = format!(
            "You are Live Interpreter, a low-latency meeting interpreter. Translate the current utterance to {}. Preserve speaker intent, terminology, names, numbers, and the thread of the conversation. Return only the translated sentence.\n\nRecent context:",
            direction.target_lang_name()
        );

        if self.turns.is_empty() {
            prompt.push_str("\n- none");
            return prompt;
        }

        for (index, turn) in self.turns.iter().enumerate() {
            prompt.push_str(&format!(
                "\n{}. Original: {}\n   Translation: {}",
                index + 1,
                turn.original,
                turn.translated
            ));
        }

        prompt
    }
}

/// Translation backend. Ollama HTTP by default; an in-process Candle Qwen2
/// backend when built with `candle-translate` and `LI_TRANSLATE_BACKEND=candle`.
#[derive(Clone)]
pub enum Translator {
    Http(HttpTranslator),
    #[cfg(feature = "candle-translate")]
    Candle(CandleTranslator),
}

impl Translator {
    /// Pick the backend. `LI_TRANSLATE_BACKEND` (`candle`/`ollama`) wins when
    /// set; otherwise default to the in-process Candle backend **iff** this was
    /// built with `candle-translate` and the GGUF model is present — so a GPU
    /// build with the model downloaded uses Candle automatically (faster, frees
    /// Ollama's VRAM), while every other configuration falls back to Ollama.
    /// `candle` requires the feature; the Ollama args are ignored by that backend.
    pub fn from_env(ollama_url: String, ollama_model: String) -> Result<Self> {
        let requested = std::env::var("LI_TRANSLATE_BACKEND").ok();
        let backend = resolve_backend(
            requested.as_deref(),
            cfg!(feature = "candle-translate"),
            candle_model_present(),
        );
        match backend {
            #[cfg(feature = "candle-translate")]
            "candle" => Ok(Self::Candle(CandleTranslator::from_env()?)),
            #[cfg(not(feature = "candle-translate"))]
            "candle" => anyhow::bail!(
                "LI_TRANSLATE_BACKEND=candle requires building with --features candle-translate"
            ),
            _ => Ok(Self::Http(HttpTranslator::new(ollama_url, ollama_model))),
        }
    }

    pub async fn translate(&self, text: &str, direction: &Direction) -> Result<String> {
        match self {
            Self::Http(t) => t.translate(text, direction).await,
            #[cfg(feature = "candle-translate")]
            Self::Candle(t) => t.translate(text, direction).await,
        }
    }

    pub async fn observe_silence(&self, silence: Duration) {
        match self {
            Self::Http(t) => t.observe_silence(silence).await,
            #[cfg(feature = "candle-translate")]
            Self::Candle(t) => t.observe_silence(silence).await,
        }
    }

    pub async fn translate_stream(&self, text: &str, direction: &Direction) -> Result<TokenStream> {
        match self {
            Self::Http(t) => t.translate_stream(text, direction).await,
            #[cfg(feature = "candle-translate")]
            Self::Candle(t) => t.translate_stream(text, direction).await,
        }
    }
}

/// Default GGUF path for the Candle backend (override with `LI_CANDLE_TRANSLATE_GGUF`).
/// 3B is the quality/VRAM sweet spot: it beat 1.5B on an idiom A/B and still
/// coexists with Qwen-TTS (~10GB) + Whisper on a 16GB card once Candle replaces
/// Ollama. Use the 1.5B/0.5B GGUFs via the env override for tighter VRAM.
pub(crate) const DEFAULT_CANDLE_GGUF: &str =
    "data/models/translate/qwen2.5-3b-instruct-q4_k_m.gguf";

/// Pure backend choice: an explicit `LI_TRANSLATE_BACKEND` wins; otherwise
/// prefer Candle only when both the feature is compiled in and the model is
/// present, else Ollama.
fn resolve_backend(
    requested: Option<&str>,
    candle_feature: bool,
    model_present: bool,
) -> &'static str {
    match requested {
        Some("candle") => "candle",
        Some("ollama") => "ollama",
        _ if candle_feature && model_present => "candle",
        _ => "ollama",
    }
}

fn candle_model_present() -> bool {
    let path = std::env::var("LI_CANDLE_TRANSLATE_GGUF")
        .unwrap_or_else(|_| DEFAULT_CANDLE_GGUF.to_string());
    std::path::Path::new(&path).exists()
}

/// Shared instruction prompt. Kept identical to the original `translate.rs` so server output does
/// not change when switching backends.
pub(crate) fn prompt_for(text: &str, direction: &Direction) -> String {
    format!(
        "Translate the following text to {}. Return only the translation, no notes, no markdown, no explanations.\n\n{}",
        direction.target_lang_name(),
        text
    )
}

fn clamp_chars(value: String, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }

    let mut out = String::with_capacity(value.len().min(max_chars));
    for ch in value.chars().take(max_chars) {
        out.push(ch);
    }
    out
}

/// Strip `<think>...</think>` reasoning blocks emitted by reasoning models.
pub(crate) fn strip_think(value: &str) -> String {
    let mut output = String::new();
    let mut rest = value;

    while let Some(start) = rest.find("<think>") {
        output.push_str(&rest[..start]);
        if let Some(end) = rest[start..].find("</think>") {
            rest = &rest[start + end + "</think>".len()..];
        } else {
            return output;
        }
    }

    output.push_str(rest);
    output
}

#[cfg(test)]
mod tests {
    use super::{
        TranslationBuffer, TranslationBufferConfig, TranslationTurn, clamp_chars, resolve_backend,
        strip_think,
    };
    use crate::types::Direction;
    use std::time::Duration;

    #[test]
    fn resolve_backend_respects_explicit_request() {
        assert_eq!(resolve_backend(Some("candle"), false, false), "candle");
        assert_eq!(resolve_backend(Some("ollama"), true, true), "ollama");
    }

    #[test]
    fn resolve_backend_auto_prefers_candle_only_with_feature_and_model() {
        assert_eq!(resolve_backend(None, true, true), "candle");
        assert_eq!(resolve_backend(None, true, false), "ollama");
        assert_eq!(resolve_backend(None, false, true), "ollama");
        assert_eq!(resolve_backend(None, false, false), "ollama");
    }

    #[test]
    fn resolve_backend_unknown_value_falls_back_to_auto() {
        assert_eq!(resolve_backend(Some("bogus"), true, true), "candle");
        assert_eq!(resolve_backend(Some("bogus"), false, false), "ollama");
    }

    #[test]
    fn strips_complete_think_blocks() {
        let value = "<think>internal reasoning</think>Texto limpio";
        assert_eq!(strip_think(value).trim(), "Texto limpio");
    }

    #[test]
    fn keeps_text_around_think_blocks() {
        let value = "A <think>hidden</think> B";
        assert_eq!(strip_think(value), "A  B");
    }

    #[test]
    fn drops_unclosed_think_block() {
        let value = "visible <think>never closed";
        assert_eq!(strip_think(value), "visible ");
    }

    #[test]
    fn translation_buffer_generates_system_prompt_with_recent_turns() {
        let mut buffer = TranslationBuffer::new(TranslationBufferConfig {
            max_interactions: 3,
            silence_reset_ms: 12_000,
            max_chars_per_text: 120,
        });

        buffer.push("hablamos del presupuesto", "we are discussing the budget");
        buffer.push("aplaza la demo", "postpone the demo");

        assert_eq!(
            buffer.system_prompt(&Direction::EsToEn),
            "You are Live Interpreter, a low-latency meeting interpreter. Translate the current utterance to English. Preserve speaker intent, terminology, names, numbers, and the thread of the conversation. Return only the translated sentence.\n\nRecent context:\n1. Original: hablamos del presupuesto\n   Translation: we are discussing the budget\n2. Original: aplaza la demo\n   Translation: postpone the demo"
        );
    }

    #[test]
    fn translation_buffer_uses_empty_context_marker() {
        let buffer = TranslationBuffer::new(TranslationBufferConfig::default());

        assert_eq!(
            buffer.system_prompt(&Direction::EnToEs),
            "You are Live Interpreter, a low-latency meeting interpreter. Translate the current utterance to Spanish. Preserve speaker intent, terminology, names, numbers, and the thread of the conversation. Return only the translated sentence.\n\nRecent context:\n- none"
        );
    }

    #[test]
    fn translation_buffer_keeps_only_the_last_n_turns() {
        let mut buffer = TranslationBuffer::new(TranslationBufferConfig {
            max_interactions: 2,
            silence_reset_ms: 12_000,
            max_chars_per_text: 120,
        });

        buffer.push("uno", "one");
        buffer.push("dos", "two");
        buffer.push("tres", "three");

        assert_eq!(buffer.len(), 2);
        assert_eq!(
            buffer.turns.into_iter().collect::<Vec<_>>(),
            vec![
                TranslationTurn {
                    original: "dos".into(),
                    translated: "two".into(),
                },
                TranslationTurn {
                    original: "tres".into(),
                    translated: "three".into(),
                },
            ]
        );
    }

    #[test]
    fn translation_buffer_clears_after_long_silence() {
        let mut buffer = TranslationBuffer::new(TranslationBufferConfig {
            max_interactions: 4,
            silence_reset_ms: 3_000,
            max_chars_per_text: 120,
        });

        buffer.push("seguimos", "we continue");
        buffer.observe_silence(Duration::from_millis(2_999));
        assert_eq!(buffer.len(), 1);

        buffer.observe_silence(Duration::from_millis(3_000));
        assert!(buffer.is_empty());
    }

    #[test]
    fn translation_buffer_clamps_text_to_prevent_unbounded_memory_growth() {
        let mut buffer = TranslationBuffer::new(TranslationBufferConfig {
            max_interactions: 2,
            silence_reset_ms: 12_000,
            max_chars_per_text: 4,
        });

        buffer.push("abcdef", "uvwxyz");

        let turn = buffer.turns.front().unwrap();
        assert_eq!(turn.original, "abcd");
        assert_eq!(turn.translated, "uvwx");
        assert_eq!(clamp_chars("abcdef".into(), 2), "ab");
    }

    #[test]
    fn translation_buffer_rejects_unknown_serde_fields() {
        let json = r#"{
            "config": {
                "max_interactions": 2,
                "silence_reset_ms": 12000,
                "max_chars_per_text": 120
            },
            "turns": [],
            "unexpected": true
        }"#;

        assert!(serde_json::from_str::<TranslationBuffer>(json).is_err());
    }
}
