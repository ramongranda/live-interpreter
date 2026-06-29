//! In-process quantized-LLM translation backend (Candle + Qwen2 GGUF).
//!
//! Drop-in alternative to the Ollama HTTP backend: loads a quantized Qwen2
//! instruct model once and runs greedy decoding in-process, so no external
//! Ollama process holds VRAM. Keeps the same `TranslationBuffer` rolling
//! context and the `prompt_for`/`strip_think` contract as the HTTP path, so
//! output stays consistent when switching backends. Gated behind the
//! `candle-translate` feature (heavy to compile); GPU via the `cuda` feature.

use super::{TokenStream, TranslationBuffer, TranslationBufferConfig, prompt_for, strip_think};
use crate::types::Direction;
use anyhow::{Context, Result, anyhow};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_qwen2::ModelWeights;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokenizers::Tokenizer;

/// Fixed seed: translation is greedy (temperature 0) so this only affects
/// tie-breaks; a constant keeps runs reproducible, matching Ollama's temp 0.
const SEED: u64 = 299_792_458;

/// The heavy, non-`Sync`-by-default state, guarded by a mutex. `forward` needs
/// `&mut self` (KV cache) and is CPU/GPU-bound, so calls run under
/// `spawn_blocking` while holding this lock.
struct Inner {
    model: ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
    stop_tokens: Vec<u32>,
    max_new_tokens: usize,
}

/// Quantized-LLM translator. `Clone` (shares the loaded model behind an `Arc`)
/// so `Translator`/`AppState` stay `Clone`.
#[derive(Clone)]
pub struct CandleTranslator {
    inner: Arc<Mutex<Inner>>,
    buffer: Arc<tokio::sync::Mutex<TranslationBuffer>>,
}

impl CandleTranslator {
    /// Load the GGUF model + tokenizer from local paths (defaults under
    /// `data/models/translate/`, override with `LI_CANDLE_TRANSLATE_GGUF` /
    /// `_TOKENIZER`). Fetch them with `scripts/download-translate-model.sh`.
    pub fn from_env() -> Result<Self> {
        let gguf_path = std::env::var("LI_CANDLE_TRANSLATE_GGUF")
            .unwrap_or_else(|_| super::DEFAULT_CANDLE_GGUF.to_string());
        let tokenizer_path = std::env::var("LI_CANDLE_TRANSLATE_TOKENIZER")
            .unwrap_or_else(|_| "data/models/translate/tokenizer.json".into());
        let max_new_tokens = std::env::var("LI_CANDLE_TRANSLATE_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256);

        let device = select_device();
        tracing::info!("candle translate: loading {gguf_path} on {device:?}");

        let mut file = std::fs::File::open(&gguf_path)
            .with_context(|| format!("failed to open GGUF model {gguf_path}"))?;
        let content = gguf_file::Content::read(&mut file)
            .with_context(|| format!("failed to read GGUF content from {gguf_path}"))?;
        let model = ModelWeights::from_gguf(content, &mut file, &device)
            .context("failed to build Qwen2 model from GGUF")?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("failed to load tokenizer {tokenizer_path}: {e}"))?;

        let stop_tokens = ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|t| tokenizer.token_to_id(t))
            .collect::<Vec<_>>();
        if stop_tokens.is_empty() {
            tracing::warn!("candle translate: no stop token found; relying on max_new_tokens");
        }

        Ok(Self {
            inner: Arc::new(Mutex::new(Inner {
                model,
                tokenizer,
                device,
                stop_tokens,
                max_new_tokens,
            })),
            buffer: Arc::new(tokio::sync::Mutex::new(TranslationBuffer::new(
                TranslationBufferConfig::default(),
            ))),
        })
    }

    pub async fn translate(&self, text: &str, direction: &Direction) -> Result<String> {
        let system = self.buffer.lock().await.system_prompt(direction);
        let user = prompt_for(text, direction);
        let inner = Arc::clone(&self.inner);
        let raw = tokio::task::spawn_blocking(move || -> Result<String> {
            let mut guard = inner
                .lock()
                .map_err(|_| anyhow!("candle translator mutex poisoned"))?;
            guard.generate(&system, &user)
        })
        .await
        .context("candle translate task panicked")??;

        let translated = strip_think(&raw).trim().to_string();
        self.buffer
            .lock()
            .await
            .push(text.to_string(), translated.clone());
        Ok(translated)
    }

    /// Non-streaming: yields the full translation as a single chunk, matching
    /// the HTTP backend so the bench/chunked path behaves identically.
    pub async fn translate_stream(&self, text: &str, direction: &Direction) -> Result<TokenStream> {
        let out = self.translate(text, direction).await?;
        Ok(Box::pin(futures_util::stream::once(async move { Ok(out) })))
    }

    pub async fn observe_silence(&self, silence: Duration) {
        self.buffer.lock().await.observe_silence(silence);
    }
}

impl Inner {
    fn generate(&mut self, system: &str, user: &str) -> Result<String> {
        // Independent utterance: drop any cached attention state from the last one.
        self.model.clear_kv_cache();
        let prompt = format!(
            "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        );
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!("tokenizer encode failed: {e}"))?
            .get_ids()
            .to_vec();
        if tokens.is_empty() {
            return Ok(String::new());
        }

        let mut logits_processor = LogitsProcessor::new(SEED, None, None);
        let mut generated: Vec<u32> = Vec::new();

        // Prompt pass: feed the whole prompt at position 0, sample first output token.
        let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0)?;
        let mut next = logits_processor.sample(&logits.squeeze(0)?.to_dtype(DType::F32)?)?;

        // Single-token decode; `index_pos` is the KV-cache position of `next`.
        for index_pos in (tokens.len()..).take(self.max_new_tokens) {
            if self.stop_tokens.contains(&next) {
                break;
            }
            generated.push(next);
            let input = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, index_pos)?;
            next = logits_processor.sample(&logits.squeeze(0)?.to_dtype(DType::F32)?)?;
        }

        self.tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow!("tokenizer decode failed: {e}"))
    }
}

/// CUDA when built with `--features cuda` and a device is present (honors
/// `LI_CANDLE_DEVICE=cpu` to force CPU), else CPU.
fn select_device() -> Device {
    if std::env::var("LI_CANDLE_DEVICE").as_deref() == Ok("cpu") {
        return Device::Cpu;
    }
    Device::cuda_if_available(0).unwrap_or(Device::Cpu)
}
