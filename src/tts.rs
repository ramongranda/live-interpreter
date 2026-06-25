use crate::{config::Config, types::Direction};
use anyhow::{Context, bail};
use base64::{Engine, engine::general_purpose::STANDARD};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};
use uuid::Uuid;

#[derive(Clone)]
pub struct TtsEngine {
    client: Client,
    base_url: String,
    model: String,
    voice: String,
    voice_ref: Option<PathBuf>,
    voice_ref_text: Option<String>,
    out_dir: PathBuf,
}

impl TtsEngine {
    pub async fn new(config: &Config) -> anyhow::Result<Self> {
        let out_dir = config.data_dir.join("tts");
        tokio::fs::create_dir_all(&out_dir).await?;
        Ok(Self {
            client: Client::new(),
            base_url: config.qwen_tts_url.clone(),
            model: config.qwen_tts_model.clone(),
            voice: config.qwen_tts_voice.clone(),
            voice_ref: config.voice_ref.clone(),
            voice_ref_text: config.voice_ref_text.clone(),
            out_dir,
        })
    }

    pub async fn synthesize(
        &self,
        id: Uuid,
        text: &str,
        direction: &Direction,
    ) -> anyhow::Result<PathBuf> {
        let audio_sample = self
            .voice_ref
            .as_ref()
            .map(|path| {
                fs::read(path)
                    .with_context(|| format!("failed to read voice reference {}", path.display()))
                    .map(|bytes| STANDARD.encode(bytes))
            })
            .transpose()?;
        let request = QwenTtsRequest::new(
            text,
            direction,
            &self.model,
            &self.voice,
            audio_sample.as_deref(),
            self.voice_ref_text.as_deref(),
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

        let output = self.out_dir.join(format!("{id}.wav"));
        let bytes = response.bytes().await.context("failed to read TTS audio")?;
        tokio::fs::write(&output, bytes).await?;
        Ok(output)
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct QwenTtsRequest<'a> {
    text: &'a str,
    model: &'a str,
    language: &'a str,
    input: &'a str,
    voice: &'a str,
    audio_sample: Option<&'a str>,
    audio_sample_text: Option<&'a str>,
    response_format: &'a str,
    format: &'a str,
    stream: bool,
}

impl<'a> QwenTtsRequest<'a> {
    pub(crate) fn new(
        text: &'a str,
        direction: &'a Direction,
        model: &'a str,
        voice: &'a str,
        audio_sample: Option<&'a str>,
        audio_sample_text: Option<&'a str>,
    ) -> Self {
        Self {
            text,
            model,
            language: direction.target_tts_language(),
            input: text,
            voice,
            audio_sample,
            audio_sample_text,
            response_format: "wav",
            format: "wav",
            stream: false,
        }
    }
}

#[derive(Debug, Deserialize)]
struct _QwenTtsResponseDocOnly {}

#[cfg(test)]
mod tests {
    use super::QwenTtsRequest;
    use crate::types::Direction;

    #[test]
    fn tts_request_uses_target_language() {
        let request = QwenTtsRequest::new(
            "hello",
            &Direction::EsToEn,
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "alloy",
            Some("base64-audio"),
            Some("reference text"),
        );
        let json = serde_json::to_value(request).unwrap();
        assert_eq!(json["text"], "hello");
        assert_eq!(json["input"], "hello");
        assert_eq!(json["model"], "Qwen/Qwen3-TTS-12Hz-0.6B-Base");
        assert_eq!(json["language"], "english");
        assert_eq!(json["voice"], "alloy");
        assert_eq!(json["audio_sample"], "base64-audio");
        assert_eq!(json["audio_sample_text"], "reference text");
        assert_eq!(json["response_format"], "wav");
        assert_eq!(json["format"], "wav");
        assert_eq!(json["stream"], false);
    }
}
