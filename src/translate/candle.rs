//! Option B backend: in-process quantized Qwen2 (GGUF) inference via `candle`.
//!
//! Removes the Ollama process from the path. Loads a quantized Qwen2.5-Instruct GGUF once, then
//! runs greedy decoding directly on the selected device (CUDA / Metal / CPU). The model's KV cache
//! is mutated per step and the weights are not `Sync`, so a single `Engine` is serialised behind a
//! `tokio::sync::Mutex`; inference runs on a blocking thread via `spawn_blocking` so it never
//! stalls the async runtime.
//!
//! API verified against candle `quantized_qwen2`: `forward(&mut self, x, index_pos) -> [batch,
//! vocab]` (caller squeezes the batch dim), `index_pos == 0` resets the cache for a fresh prompt,
//! and `clear_kv_cache()` is called defensively between independent translations.

use super::{TokenStream, prompt_for, strip_think};
use crate::types::Direction;
use anyhow::{Context, Result, bail};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_qwen2::ModelWeights;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct CandleTranslator {
    inner: Arc<Mutex<Engine>>,
}

struct Engine {
    model: ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
    eos: u32,
    max_tokens: usize,
}

impl CandleTranslator {
    /// Load a quantized Qwen2 GGUF and its tokenizer. `max_tokens` caps generated tokens.
    pub fn load(gguf_path: &str, tokenizer_path: &str, max_tokens: usize) -> Result<Self> {
        let device = select_device()?;
        tracing::info!("candle translate backend on device {device:?}");

        let mut file = std::fs::File::open(gguf_path)
            .with_context(|| format!("failed to open GGUF model at {gguf_path}"))?;
        let content = gguf_file::Content::read(&mut file)
            .with_context(|| format!("failed to parse GGUF at {gguf_path}"))?;
        let model = ModelWeights::from_gguf(content, &mut file, &device)
            .context("failed to load Qwen2 quantized weights")?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("failed to load tokenizer at {tokenizer_path}"))?;
        let eos = tokenizer
            .get_vocab(true)
            .get("<|im_end|>")
            .copied()
            .context("tokenizer is missing the <|im_end|> token")?;

        Ok(Self {
            inner: Arc::new(Mutex::new(Engine {
                model,
                tokenizer,
                device,
                eos,
                max_tokens,
            })),
        })
    }

    pub async fn translate(&self, text: &str, direction: &Direction) -> Result<String> {
        let prompt = chat_prompt(text, direction);
        let engine = self.inner.clone();
        let raw = tokio::task::spawn_blocking(move || -> Result<String> {
            let mut guard = engine.blocking_lock();
            guard.generate(&prompt, None)
        })
        .await
        .context("candle inference task panicked")??;
        Ok(strip_think(&raw).trim().to_string())
    }

    /// Streams one item per decoded token. First `.next().await` ≈ prompt prefill cost
    /// (time-to-first-token).
    pub async fn translate_stream(&self, text: &str, direction: &Direction) -> Result<TokenStream> {
        let prompt = chat_prompt(text, direction);
        let engine = self.inner.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String>>(32);

        tokio::task::spawn_blocking(move || {
            let mut guard = engine.blocking_lock();
            let sink = tx.clone();
            let on_token = move |piece: &str| {
                let _ = sink.blocking_send(Ok(piece.to_string()));
            };
            if let Err(error) = guard.generate(&prompt, Some(&on_token as &(dyn Fn(&str) + Send))) {
                let _ = tx.blocking_send(Err(error));
            }
        });

        Ok(Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }
}

impl Engine {
    /// Greedy decode. `on_token`, when present, receives each newly decoded text piece.
    fn generate(
        &mut self,
        prompt: &str,
        on_token: Option<&(dyn Fn(&str) + Send)>,
    ) -> Result<String> {
        // Independent translation => start from a clean cache. `index_pos == 0` below also resets,
        // this is belt-and-suspenders.
        self.model.clear_kv_cache();

        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?;
        let prompt_tokens = encoding.get_ids().to_vec();
        if prompt_tokens.is_empty() {
            bail!("prompt encoded to zero tokens");
        }

        let mut logits_processor = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        // Borrows `self.tokenizer` immutably; disjoint from the `self.model` mutable borrows below.
        let mut tos = TokenOutputStream::new(&self.tokenizer);
        let mut output = String::new();

        // Prefill the whole prompt at position 0, sample the first generated token.
        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0)?.squeeze(0)?;
        let mut next = logits_processor.sample(&logits)?;

        if next != self.eos {
            if let Some(piece) = tos.next_token(next)? {
                if let Some(callback) = on_token {
                    callback(&piece);
                }
                output.push_str(&piece);
            }
        }

        for index in 0..self.max_tokens {
            if next == self.eos {
                break;
            }
            // Feed the just-produced token at its sequence position; the cache supplies the rest.
            let input = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
            let logits = self
                .model
                .forward(&input, prompt_tokens.len() + index)?
                .squeeze(0)?;
            next = logits_processor.sample(&logits)?;
            if next == self.eos {
                break;
            }
            if let Some(piece) = tos.next_token(next)? {
                if let Some(callback) = on_token {
                    callback(&piece);
                }
                output.push_str(&piece);
            }
        }

        if let Some(piece) = tos.decode_rest()? {
            if let Some(callback) = on_token {
                callback(&piece);
            }
            output.push_str(&piece);
        }

        Ok(output)
    }
}

/// Wrap the translation instruction in the Qwen2 chat template.
fn chat_prompt(text: &str, direction: &Direction) -> String {
    format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt_for(text, direction)
    )
}

fn select_device() -> Result<Device> {
    #[cfg(feature = "candle-cuda")]
    {
        return Device::cuda_if_available(0).context("failed to init CUDA device");
    }
    #[cfg(all(feature = "candle-metal", not(feature = "candle-cuda")))]
    {
        return Device::new_metal(0).context("failed to init Metal device");
    }
    #[cfg(not(any(feature = "candle-cuda", feature = "candle-metal")))]
    {
        Ok(Device::Cpu)
    }
}

/// Incremental detokenizer: only flushes complete text so partial multi-byte tokens are buffered.
/// Mirrors candle-examples' `TokenOutputStream`, holding a borrow of the tokenizer rather than
/// owning it.
struct TokenOutputStream<'a> {
    tokenizer: &'a Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl<'a> TokenOutputStream<'a> {
    fn new(tokenizer: &'a Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(anyhow::Error::msg)
    }

    fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            self.decode(&self.tokens[self.prev_index..self.current_index])?
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().is_some_and(char::is_alphanumeric) {
            let new_text = text[prev_text.len()..].to_string();
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(new_text))
        } else {
            Ok(None)
        }
    }

    fn decode_rest(&self) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            self.decode(&self.tokens[self.prev_index..self.current_index])?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            Ok(Some(text[prev_text.len()..].to_string()))
        } else {
            Ok(None)
        }
    }
}
