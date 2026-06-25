use crate::types::Direction;
use anyhow::{Context, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct Translator {
    client: Client,
    base_url: String,
    model: String,
}

impl Translator {
    pub fn new(base_url: String, model: String) -> Self {
        Self {
            client: Client::new(),
            base_url,
            model,
        }
    }

    pub async fn translate(&self, text: &str, direction: &Direction) -> anyhow::Result<String> {
        let prompt = format!(
            "Translate the following text to {}. Return only the translation, no notes, no markdown, no explanations.\n\n{}",
            direction.target_lang_name(),
            text
        );
        let request = OllamaGenerateRequest {
            model: self.model.clone(),
            prompt,
            stream: false,
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
        Ok(strip_think(&body.response).trim().to_string())
    }
}

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
    use super::strip_think;

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
}

#[derive(Debug, Serialize)]
struct OllamaGenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
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
