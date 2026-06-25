use std::{env, net::SocketAddr, path::PathBuf};

#[derive(Clone, Debug)]
pub struct Config {
    pub bind: SocketAddr,
    pub data_dir: PathBuf,
    pub whisper_model: PathBuf,
    pub whisper_threads: i32,
    pub ollama_url: String,
    pub ollama_model: String,
    pub qwen_tts_url: String,
    pub qwen_tts_model: String,
    pub qwen_tts_voice: String,
    pub voice_ref: Option<PathBuf>,
    pub voice_ref_text: Option<String>,
    pub ffmpeg_bin: String,
    pub auth_token: Option<String>,
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        let bind = env::var("OVT_BIND")
            .unwrap_or_else(|_| "127.0.0.1:8787".to_string())
            .parse()?;
        let data_dir = PathBuf::from(env::var("OVT_DATA_DIR").unwrap_or_else(|_| "data".into()));
        let whisper_model = PathBuf::from(
            env::var("OVT_WHISPER_MODEL")
                .unwrap_or_else(|_| "data/models/ggml-large-v3-turbo.bin".into()),
        );
        let whisper_threads = env::var("OVT_WHISPER_THREADS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(8);
        let ollama_url =
            env::var("OVT_OLLAMA_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".into());
        let ollama_model =
            env::var("OVT_OLLAMA_MODEL").unwrap_or_else(|_| "translator:latest".into());
        let qwen_tts_url =
            env::var("OVT_QWEN_TTS_URL").unwrap_or_else(|_| "http://127.0.0.1:8020".into());
        let qwen_tts_model = env::var("OVT_QWEN_TTS_MODEL")
            .unwrap_or_else(|_| "Qwen/Qwen3-TTS-12Hz-0.6B-Base".into());
        let qwen_tts_voice = env::var("OVT_QWEN_TTS_VOICE").unwrap_or_else(|_| "alloy".into());
        let voice_ref = env::var("OVT_VOICE_REF").ok().map(PathBuf::from);
        let voice_ref_text = env::var("OVT_VOICE_REF_TEXT").ok();
        let ffmpeg_bin = env::var("OVT_FFMPEG_BIN").unwrap_or_else(|_| {
            if PathBuf::from("/snap/bin/ffmpeg").exists() {
                "/snap/bin/ffmpeg".into()
            } else {
                "ffmpeg".into()
            }
        });
        let auth_token = env::var("OVT_AUTH_TOKEN")
            .ok()
            .filter(|value| !value.trim().is_empty());

        Ok(Self {
            bind,
            data_dir,
            whisper_model,
            whisper_threads,
            ollama_url,
            ollama_model,
            qwen_tts_url,
            qwen_tts_model,
            qwen_tts_voice,
            voice_ref,
            voice_ref_text,
            ffmpeg_bin,
            auth_token,
        })
    }
}
