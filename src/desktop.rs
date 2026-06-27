use base64::{Engine, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    future::Future,
    path::{Path, PathBuf},
};
use tokio::sync::mpsc;

#[derive(Clone, Debug)]
pub struct DesktopConfig {
    pub root: PathBuf,
    pub stack_bind: String,
    pub server_url: String,
    pub qwen_url: String,
    pub client_bind: String,
    pub client_url: String,
    pub play_target: String,
    pub min_server_vram_mb: u64,
}

impl DesktopConfig {
    pub fn from_root(root: PathBuf) -> Self {
        let client_bind =
            std::env::var("LI_CLIENT_BIND").unwrap_or_else(|_| "127.0.0.1:8790".into());
        Self {
            root,
            stack_bind: std::env::var("LI_STACK_BIND").unwrap_or_else(|_| "0.0.0.0:8787".into()),
            server_url: std::env::var("LI_SERVER_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8787".into()),
            qwen_url: std::env::var("LI_QWEN_TTS_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8020".into()),
            client_url: format!("http://{client_bind}"),
            client_bind,
            play_target: std::env::var("LI_CLIENT_PLAY_TARGET")
                .unwrap_or_else(|_| "live-interpreter-mic-sink".into()),
            min_server_vram_mb: std::env::var("LI_MIN_SERVER_VRAM_MB")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(8_000),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ActionResult {
    pub ok: bool,
    pub output: String,
}

#[derive(Debug, Serialize)]
pub struct VoiceProfile {
    pub configured: bool,
    pub audio_path: String,
    pub text_path: String,
    pub reference_text: String,
}

#[derive(Debug, Deserialize)]
pub struct SaveVoiceProfileRequest {
    pub audio_base64: String,
    pub reference_text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamingBackpressure {
    pub raw_audio_frames: usize,
    pub transcript_fragments: usize,
    pub translated_texts: usize,
    pub generated_audio_buffers: usize,
}

impl Default for StreamingBackpressure {
    fn default() -> Self {
        Self {
            raw_audio_frames: 8,
            transcript_fragments: 8,
            translated_texts: 4,
            generated_audio_buffers: 4,
        }
    }
}

#[derive(Debug)]
pub struct StreamingChannels {
    pub raw_audio_tx: mpsc::Sender<RawAudioFrame>,
    pub raw_audio_rx: mpsc::Receiver<RawAudioFrame>,
    pub transcript_tx: mpsc::Sender<TranscriptFragment>,
    pub transcript_rx: mpsc::Receiver<TranscriptFragment>,
    pub translated_tx: mpsc::Sender<TranslatedText>,
    pub translated_rx: mpsc::Receiver<TranslatedText>,
    pub generated_audio_tx: mpsc::Sender<GeneratedAudio>,
    pub generated_audio_rx: mpsc::Receiver<GeneratedAudio>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RawAudioFrame {
    pub sequence: u64,
    pub sample_rate_hz: u32,
    pub samples: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscriptFragment {
    pub sequence: u64,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranslatedText {
    pub sequence: u64,
    pub source_text: String,
    pub translated_text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneratedAudio {
    pub sequence: u64,
    pub sample_rate_hz: u32,
    pub pcm_s16le: Vec<i16>,
}

pub trait AsrEngine: Send + Sync + 'static {
    fn transcribe(
        &self,
        frame: RawAudioFrame,
    ) -> impl Future<Output = anyhow::Result<Vec<TranscriptFragment>>> + Send;
}

pub trait TranslationEngine: Send + Sync + 'static {
    fn translate(
        &self,
        fragment: TranscriptFragment,
    ) -> impl Future<Output = anyhow::Result<TranslatedText>> + Send;
}

pub trait TtsEngine: Send + Sync + 'static {
    fn synthesize(
        &self,
        text: TranslatedText,
    ) -> impl Future<Output = anyhow::Result<GeneratedAudio>> + Send;
}

pub trait AudioSink: Send + Sync + 'static {
    fn play(&self, audio: GeneratedAudio) -> impl Future<Output = anyhow::Result<()>> + Send;
}

pub fn streaming_channels(config: StreamingBackpressure) -> StreamingChannels {
    let (raw_audio_tx, raw_audio_rx) = mpsc::channel(config.raw_audio_frames);
    let (transcript_tx, transcript_rx) = mpsc::channel(config.transcript_fragments);
    let (translated_tx, translated_rx) = mpsc::channel(config.translated_texts);
    let (generated_audio_tx, generated_audio_rx) = mpsc::channel(config.generated_audio_buffers);

    StreamingChannels {
        raw_audio_tx,
        raw_audio_rx,
        transcript_tx,
        transcript_rx,
        translated_tx,
        translated_rx,
        generated_audio_tx,
        generated_audio_rx,
    }
}

pub async fn run_asr_actor<E>(
    engine: E,
    mut raw_audio_rx: mpsc::Receiver<RawAudioFrame>,
    transcript_tx: mpsc::Sender<TranscriptFragment>,
) -> anyhow::Result<()>
where
    E: AsrEngine,
{
    while let Some(frame) = raw_audio_rx.recv().await {
        for fragment in engine.transcribe(frame).await? {
            if transcript_tx.send(fragment).await.is_err() {
                break;
            }
        }
    }
    Ok(())
}

pub async fn run_translation_actor<E>(
    engine: E,
    mut transcript_rx: mpsc::Receiver<TranscriptFragment>,
    translated_tx: mpsc::Sender<TranslatedText>,
) -> anyhow::Result<()>
where
    E: TranslationEngine,
{
    while let Some(fragment) = transcript_rx.recv().await {
        let translated = engine.translate(fragment).await?;
        if translated_tx.send(translated).await.is_err() {
            break;
        }
    }
    Ok(())
}

pub async fn run_tts_actor<E>(
    engine: E,
    mut translated_rx: mpsc::Receiver<TranslatedText>,
    generated_audio_tx: mpsc::Sender<GeneratedAudio>,
) -> anyhow::Result<()>
where
    E: TtsEngine,
{
    while let Some(text) = translated_rx.recv().await {
        let audio = engine.synthesize(text).await?;
        if generated_audio_tx.send(audio).await.is_err() {
            break;
        }
    }
    Ok(())
}

pub async fn run_audio_sink_actor<S>(
    sink: S,
    mut generated_audio_rx: mpsc::Receiver<GeneratedAudio>,
) -> anyhow::Result<()>
where
    S: AudioSink,
{
    while let Some(audio) = generated_audio_rx.recv().await {
        sink.play(audio).await?;
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuPreflight {
    pub ready: bool,
    pub message: String,
}

pub fn pid_alive(path: &Path) -> bool {
    let Ok(pid) = std::fs::read_to_string(path) else {
        return false;
    };
    let Ok(pid) = pid.trim().parse::<u32>() else {
        return false;
    };
    PathBuf::from(format!("/proc/{pid}")).exists()
}

pub fn voice_dir(config: &DesktopConfig) -> PathBuf {
    config.root.join("data/voice")
}

pub fn voice_audio_path(config: &DesktopConfig) -> PathBuf {
    voice_dir(config).join("reference.wav")
}

pub fn voice_text_path(config: &DesktopConfig) -> PathBuf {
    voice_dir(config).join("reference.txt")
}

pub fn voice_profile(config: &DesktopConfig) -> VoiceProfile {
    let audio_path = voice_audio_path(config);
    let text_path = voice_text_path(config);
    let reference_text = fs::read_to_string(&text_path)
        .ok()
        .map(|text| text.trim().to_string())
        .unwrap_or_default();
    VoiceProfile {
        configured: audio_path.exists() && !reference_text.is_empty(),
        audio_path: audio_path.display().to_string(),
        text_path: text_path.display().to_string(),
        reference_text,
    }
}

pub fn save_voice_profile(
    config: &DesktopConfig,
    request: SaveVoiceProfileRequest,
) -> ActionResult {
    let reference_text = request.reference_text.trim();
    if reference_text.is_empty() {
        return ActionResult {
            ok: false,
            output: "Escribe la transcripcion exacta de la muestra de voz.".into(),
        };
    }

    let audio_bytes = match STANDARD.decode(request.audio_base64.trim()) {
        Ok(bytes) => bytes,
        Err(error) => {
            return ActionResult {
                ok: false,
                output: format!("Audio base64 invalido: {error}"),
            };
        }
    };

    if audio_bytes.len() < 1_000 {
        return ActionResult {
            ok: false,
            output: "La muestra de voz es demasiado corta o esta vacia.".into(),
        };
    }

    let dir = voice_dir(config);
    if let Err(error) = fs::create_dir_all(&dir) {
        return ActionResult {
            ok: false,
            output: error.to_string(),
        };
    }

    let audio_path = voice_audio_path(config);
    let text_path = voice_text_path(config);
    if let Err(error) = fs::write(&audio_path, audio_bytes) {
        return ActionResult {
            ok: false,
            output: error.to_string(),
        };
    }
    if let Err(error) = fs::write(&text_path, reference_text) {
        return ActionResult {
            ok: false,
            output: error.to_string(),
        };
    }

    ActionResult {
        ok: true,
        output: format!(
            "Voz guardada. Reinicia el servidor GPU para usar {}.",
            audio_path.display()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn voice_profile_save_and_read_roundtrip() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("live-interpreter-test-{unique}"));
        let config = DesktopConfig::from_root(root.clone());
        let result = save_voice_profile(
            &config,
            SaveVoiceProfileRequest {
                audio_base64: STANDARD.encode(vec![1u8; 2048]),
                reference_text: "hola mundo".into(),
            },
        );
        assert!(result.ok);
        let profile = voice_profile(&config);
        assert!(profile.configured);
        assert_eq!(profile.reference_text, "hola mundo");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn streaming_channels_are_bounded() {
        let channels = streaming_channels(StreamingBackpressure {
            raw_audio_frames: 1,
            transcript_fragments: 1,
            translated_texts: 1,
            generated_audio_buffers: 1,
        });

        channels
            .raw_audio_tx
            .try_send(RawAudioFrame {
                sequence: 1,
                sample_rate_hz: 16_000,
                samples: vec![0.0; 160],
            })
            .unwrap();

        assert!(
            channels
                .raw_audio_tx
                .try_send(RawAudioFrame {
                    sequence: 2,
                    sample_rate_hz: 16_000,
                    samples: vec![0.0; 160],
                })
                .is_err()
        );
    }

    #[tokio::test]
    async fn actors_move_audio_through_the_streaming_pipeline() {
        let channels = streaming_channels(StreamingBackpressure {
            raw_audio_frames: 2,
            transcript_fragments: 2,
            translated_texts: 2,
            generated_audio_buffers: 2,
        });
        let played = Arc::new(AtomicUsize::new(0));

        let asr = tokio::spawn(run_asr_actor(
            MockAsr,
            channels.raw_audio_rx,
            channels.transcript_tx,
        ));
        let translator = tokio::spawn(run_translation_actor(
            MockTranslator,
            channels.transcript_rx,
            channels.translated_tx,
        ));
        let tts = tokio::spawn(run_tts_actor(
            MockTts,
            channels.translated_rx,
            channels.generated_audio_tx,
        ));
        let sink = tokio::spawn(run_audio_sink_actor(
            CountingSink {
                played: played.clone(),
            },
            channels.generated_audio_rx,
        ));

        channels
            .raw_audio_tx
            .send(RawAudioFrame {
                sequence: 7,
                sample_rate_hz: 16_000,
                samples: vec![0.25; 320],
            })
            .await
            .unwrap();
        drop(channels.raw_audio_tx);

        asr.await.unwrap().unwrap();
        translator.await.unwrap().unwrap();
        tts.await.unwrap().unwrap();
        sink.await.unwrap().unwrap();

        assert_eq!(played.load(Ordering::SeqCst), 1);
    }

    struct MockAsr;

    impl AsrEngine for MockAsr {
        async fn transcribe(
            &self,
            frame: RawAudioFrame,
        ) -> anyhow::Result<Vec<TranscriptFragment>> {
            Ok(vec![TranscriptFragment {
                sequence: frame.sequence,
                text: "hola".into(),
            }])
        }
    }

    struct MockTranslator;

    impl TranslationEngine for MockTranslator {
        async fn translate(&self, fragment: TranscriptFragment) -> anyhow::Result<TranslatedText> {
            Ok(TranslatedText {
                sequence: fragment.sequence,
                source_text: fragment.text,
                translated_text: "hello".into(),
            })
        }
    }

    struct MockTts;

    impl TtsEngine for MockTts {
        async fn synthesize(&self, text: TranslatedText) -> anyhow::Result<GeneratedAudio> {
            Ok(GeneratedAudio {
                sequence: text.sequence,
                sample_rate_hz: 24_000,
                pcm_s16le: vec![0; 480],
            })
        }
    }

    struct CountingSink {
        played: Arc<AtomicUsize>,
    }

    impl AudioSink for CountingSink {
        async fn play(&self, audio: GeneratedAudio) -> anyhow::Result<()> {
            assert_eq!(audio.sequence, 7);
            assert_eq!(audio.sample_rate_hz, 24_000);
            assert_eq!(audio.pcm_s16le.len(), 480);
            self.played.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }
}
