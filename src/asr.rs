use crate::{config::Config, types::Segment};
use anyhow::{Context, bail};
use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
};
use tokio::process::Command;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

#[derive(Clone)]
pub struct AsrEngine {
    ctx: Arc<WhisperContext>,
    threads: i32,
    ffmpeg_bin: String,
    data_dir: PathBuf,
}

impl AsrEngine {
    pub fn load(config: &Config) -> anyhow::Result<Self> {
        if !config.whisper_model.exists() {
            bail!(
                "Whisper model not found at {}. Download a whisper.cpp ggml model or set OVT_WHISPER_MODEL.",
                config.whisper_model.display()
            );
        }

        let ctx = WhisperContext::new_with_params(
            &config.whisper_model,
            WhisperContextParameters::default(),
        )
        .context("failed to load whisper model")?;

        Ok(Self {
            ctx: Arc::new(ctx),
            threads: config.whisper_threads,
            ffmpeg_bin: config.ffmpeg_bin.clone(),
            data_dir: config.data_dir.clone(),
        })
    }

    pub async fn convert_to_wav16(&self, input: &Path) -> anyhow::Result<PathBuf> {
        let wav_dir = self.data_dir.join("work");
        tokio::fs::create_dir_all(&wav_dir).await?;
        let output = wav_dir.join(format!(
            "{}.16k.wav",
            input
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("audio")
        ));

        let status = Command::new(&self.ffmpeg_bin)
            .arg("-y")
            .arg("-hide_banner")
            .arg("-loglevel")
            .arg("error")
            .arg("-i")
            .arg(input)
            .arg("-ac")
            .arg("1")
            .arg("-ar")
            .arg("16000")
            .arg("-f")
            .arg("wav")
            .arg(&output)
            .stdout(Stdio::null())
            .status()
            .await
            .context("failed to execute ffmpeg")?;

        if !status.success() {
            bail!("ffmpeg failed while converting {}", input.display());
        }

        Ok(output)
    }

    pub async fn transcribe_file(
        &self,
        input: &Path,
        language: &str,
    ) -> anyhow::Result<Vec<Segment>> {
        let ctx = self.ctx.clone();
        let threads = self.threads;
        let language = language.to_string();

        if input.extension() == Some(OsStr::new("wav")) {
            let audio = load_wav_mono_16k(input)?;
            return tokio::task::spawn_blocking(move || {
                transcribe_audio(ctx, audio, threads, &language)
            })
            .await
            .context("ASR task panicked")?;
        }

        let wav = self.convert_to_wav16(input).await?;
        tokio::task::spawn_blocking(move || {
            let audio = load_wav_mono_16k(&wav)?;
            transcribe_audio(ctx, audio, threads, &language)
        })
        .await
        .context("ASR task panicked")?
    }
}

fn transcribe_audio(
    ctx: Arc<WhisperContext>,
    audio: Vec<f32>,
    threads: i32,
    language: &str,
) -> anyhow::Result<Vec<Segment>> {
    let mut state = ctx
        .create_state()
        .context("failed to create whisper state")?;
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(threads);
    params.set_language(Some(language));
    params.set_translate(false);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    state
        .full(params, &audio)
        .context("whisper inference failed")?;

    let mut segments = Vec::new();
    for segment in state.as_iter() {
        segments.push(Segment {
            start_ms: segment.start_timestamp() * 10,
            end_ms: segment.end_timestamp() * 10,
            text: segment.to_string().trim().to_string(),
        });
    }

    Ok(segments)
}

fn load_wav_mono_16k(wav_path: &Path) -> anyhow::Result<Vec<f32>> {
    let reader = hound::WavReader::open(wav_path).context("failed to open wav")?;
    let spec = reader.spec();
    if spec.bits_per_sample != 16 || spec.sample_format != hound::SampleFormat::Int {
        bail!("only 16-bit PCM wav is supported without ffmpeg");
    }

    let samples: Vec<i16> = reader
        .into_samples::<i16>()
        .collect::<Result<Vec<_>, _>>()
        .context("failed to read wav samples")?;
    let mut interleaved = vec![0.0f32; samples.len()];
    whisper_rs::convert_integer_to_float_audio(&samples, &mut interleaved)
        .context("failed to convert PCM samples")?;

    let mono = if spec.channels == 1 {
        interleaved
    } else {
        let channels = spec.channels as usize;
        interleaved
            .chunks(channels)
            .map(|frame| frame.iter().copied().sum::<f32>() / frame.len() as f32)
            .collect()
    };

    Ok(resample_linear(&mono, spec.sample_rate, 16_000))
}

fn resample_linear(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if input.is_empty() || from_rate == to_rate {
        return input.to_vec();
    }

    let out_len = ((input.len() as u64 * to_rate as u64) / from_rate as u64).max(1) as usize;
    let ratio = from_rate as f64 / to_rate as f64;
    let mut output = Vec::with_capacity(out_len);

    for idx in 0..out_len {
        let src = idx as f64 * ratio;
        let left = src.floor() as usize;
        let right = (left + 1).min(input.len() - 1);
        let frac = (src - left as f64) as f32;
        output.push(input[left] * (1.0 - frac) + input[right] * frac);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::resample_linear;

    #[test]
    fn resample_linear_downsamples_length() {
        let input = vec![0.0; 24_000];
        let output = resample_linear(&input, 24_000, 16_000);
        assert_eq!(output.len(), 16_000);
    }

    #[test]
    fn resample_linear_keeps_same_rate() {
        let input = vec![0.0, 0.5, 1.0];
        assert_eq!(resample_linear(&input, 16_000, 16_000), input);
    }
}
