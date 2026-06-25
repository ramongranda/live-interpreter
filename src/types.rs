use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    EsToEn,
    EnToEs,
}

impl Direction {
    pub fn source_lang(&self) -> &'static str {
        match self {
            Direction::EsToEn => "es",
            Direction::EnToEs => "en",
        }
    }

    pub fn target_lang_name(&self) -> &'static str {
        match self {
            Direction::EsToEn => "English",
            Direction::EnToEs => "Spanish",
        }
    }

    pub fn target_tts_language(&self) -> &'static str {
        match self {
            Direction::EsToEn => "english",
            Direction::EnToEs => "spanish",
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct TextInterpretRequest {
    pub text: String,
    pub direction: Direction,
    #[serde(default)]
    pub synthesize: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct InterpretResponse {
    pub id: Uuid,
    pub created_at: DateTime<Utc>,
    pub direction: Direction,
    pub transcript: String,
    pub translation: String,
    pub audio_path: Option<PathBuf>,
    pub audio_url: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct HealthResponse {
    pub ok: bool,
    pub whisper_model_exists: bool,
    pub ollama_url: String,
    pub ollama_model: String,
    pub qwen_tts_url: String,
    pub qwen_tts_model: String,
    pub qwen_tts_voice: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Segment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}
