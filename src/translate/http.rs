//! Option A backend: Ollama over HTTP.
//!
//! Behaviour is identical to the original `translate.rs` (single non-streaming `/api/generate`
//! call). The one addition is `keep_alive`, which pins the model in VRAM so repeated requests do
//! not pay a cold model-reload on first token — the real latency lever for the HTTP path. The
//! reqwest `Client` already pools connections, keeps HTTP keep-alive on, and sets `tcp_nodelay`
//! by default, so no further connection tuning is required.

use super::{TokenStream, TranslationBuffer, TranslationBufferConfig, prompt_for, strip_think};
use crate::types::Direction;
use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct HttpTranslator {
    client: Client,
    base_url: String,
    model: String,
    keep_alive: String,
    buffer: Arc<Mutex<TranslationBuffer>>,
}

impl HttpTranslator {
    pub fn new(base_url: String, model: String) -> Self {
        Self {
            client: Client::new(),
            base_url,
            model,
            keep_alive: std::env::var("LI_OLLAMA_KEEP_ALIVE").unwrap_or_else(|_| "30m".into()),
            buffer: Arc::new(Mutex::new(TranslationBuffer::new(
                TranslationBufferConfig::default(),
            ))),
        }
    }

    pub async fn translate(&self, text: &str, direction: &Direction) -> Result<String> {
        let system = self.buffer.lock().await.system_prompt(direction);
        let request = OllamaGenerateRequest {
            model: self.model.clone(),
            prompt: prompt_for(text, direction),
            system,
            stream: false,
            keep_alive: self.keep_alive.clone(),
            options: OllamaOptions {
                temperature: 0.0,
                num_ctx: 4096,
            },
        };

        let url = format!("{}/api/generate", self.base_url.trim_end_matches('/'));
        let response = self
            .client
            .post(url)
            .json(&request)
            .send()
            .await
            .context("failed to call Ollama")?;

        if !response.status().is_success() {
            bail!("Ollama returned {}", response.status());
        }

        let body: OllamaGenerateResponse = response.json().await.context("invalid Ollama JSON")?;
        let translated = strip_think(&body.response).trim().to_string();
        self.buffer
            .lock()
            .await
            .push(text.to_string(), translated.clone());
        Ok(translated)
    }

    /// The HTTP path is non-streaming, so this yields the full translation as a single chunk.
    /// First-token latency therefore equals total latency for this backend (honest for a bench).
    pub async fn translate_stream(&self, text: &str, direction: &Direction) -> Result<TokenStream> {
        let out = self.translate(text, direction).await?;
        Ok(Box::pin(futures_util::stream::once(async move { Ok(out) })))
    }

    pub async fn observe_silence(&self, silence: Duration) {
        self.buffer.lock().await.observe_silence(silence);
    }
}

#[derive(Debug, Serialize)]
struct OllamaGenerateRequest {
    model: String,
    prompt: String,
    system: String,
    stream: bool,
    keep_alive: String,
    options: OllamaOptions,
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
    temperature: f32,
    num_ctx: usize,
}

#[derive(Debug, Deserialize)]
struct OllamaGenerateResponse {
    response: String,
}
