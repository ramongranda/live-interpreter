//! Shared microphone capture + energy-VAD utterance segmentation.
//!
//! Both the local interpreter (`li-interpret`) and the mesh consumer
//! (`li-mesh` consumer role) need the same thing: open the default input
//! device, downmix to mono `f32`, and emit *complete utterances* once a short
//! silence follows speech. This module owns that so the binaries stay thin.
//!
//! `cpal::Stream` is `!Send`, so [`start_capture`] returns it for the caller to
//! keep alive on the main (block_on) task; the VAD runs on a dedicated OS thread
//! and forwards finished utterances over an async channel.

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use std::sync::mpsc as std_mpsc;
use tokio::sync::mpsc::UnboundedReceiver;

/// Energy-VAD tuning. Defaults match the original `li-interpret` constants.
#[derive(Clone, Copy, Debug)]
pub struct CaptureConfig {
    pub vad_threshold: f32,
    pub silence_ms: u64,
    pub min_voice_ms: u64,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            vad_threshold: 0.012,
            silence_ms: 700,
            min_voice_ms: 300,
        }
    }
}

/// A live capture: hold `stream` to keep the device open, drain `utterances`.
pub struct CaptureHandle {
    /// Keep alive — dropping it stops the device.
    pub stream: cpal::Stream,
    pub sample_rate: u32,
    pub channels: usize,
    pub utterances: UnboundedReceiver<Vec<f32>>,
}

/// Open the default input device and start VAD-segmented capture.
pub fn start_capture(config: CaptureConfig) -> Result<CaptureHandle> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("no default input device")?;
    let supported = device.default_input_config()?;
    let sample_format = supported.sample_format();
    let stream_config: StreamConfig = supported.into();
    let sample_rate = stream_config.sample_rate.0;
    let channels = stream_config.channels as usize;

    let (raw_tx, raw_rx) = std_mpsc::channel::<Vec<f32>>();
    let stream = build_input_stream(&device, &stream_config, sample_format, channels, raw_tx)?;
    stream.play()?;

    let (utt_tx, utterances) = tokio::sync::mpsc::unbounded_channel::<Vec<f32>>();
    std::thread::spawn(move || segment_utterances(raw_rx, sample_rate, config, utt_tx));

    Ok(CaptureHandle {
        stream,
        sample_rate,
        channels,
        utterances,
    })
}

/// Write mono `f32` samples as a 16-bit PCM WAV (the format Whisper ingests).
pub fn write_wav_16le(path: &std::path::Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &sample in samples {
        let clamped = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(clamped)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Energy-based VAD: accumulate while RMS exceeds threshold; flush the utterance
/// after `silence_ms` of quiet, discarding anything shorter than `min_voice_ms`.
fn segment_utterances(
    raw_rx: std_mpsc::Receiver<Vec<f32>>,
    sample_rate: u32,
    config: CaptureConfig,
    utt_tx: tokio::sync::mpsc::UnboundedSender<Vec<f32>>,
) {
    let silence_samples = (sample_rate as u64 * config.silence_ms / 1000) as usize;
    let min_voice_samples = (sample_rate as u64 * config.min_voice_ms / 1000) as usize;
    let mut utterance: Vec<f32> = Vec::new();
    let mut voiced = 0usize;
    let mut silence = 0usize;

    while let Ok(frame) = raw_rx.recv() {
        let rms = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len().max(1) as f32).sqrt();
        let speaking = rms >= config.vad_threshold;
        if speaking {
            voiced += frame.len();
            silence = 0;
            utterance.extend_from_slice(&frame);
        } else if !utterance.is_empty() {
            silence += frame.len();
            utterance.extend_from_slice(&frame);
            if silence >= silence_samples {
                if voiced >= min_voice_samples {
                    let _ = utt_tx.send(std::mem::take(&mut utterance));
                } else {
                    utterance.clear();
                }
                voiced = 0;
                silence = 0;
            }
        }
    }
}

fn build_input_stream(
    device: &cpal::Device,
    config: &StreamConfig,
    format: SampleFormat,
    channels: usize,
    tx: std_mpsc::Sender<Vec<f32>>,
) -> Result<cpal::Stream> {
    let err = |e| tracing::error!("cpal stream error: {e}");
    let stream = match format {
        SampleFormat::F32 => device.build_input_stream(
            config,
            move |data: &[f32], _| {
                let _ = tx.send(mono(data, channels, |s| s));
            },
            err,
            None,
        )?,
        SampleFormat::I16 => device.build_input_stream(
            config,
            move |data: &[i16], _| {
                let _ = tx.send(mono(data, channels, |s| s as f32 / i16::MAX as f32));
            },
            err,
            None,
        )?,
        SampleFormat::U16 => device.build_input_stream(
            config,
            move |data: &[u16], _| {
                let _ = tx.send(mono(data, channels, |s| {
                    (s as f32 - u16::MAX as f32 / 2.0) / (u16::MAX as f32 / 2.0)
                }));
            },
            err,
            None,
        )?,
        other => anyhow::bail!("unsupported sample format {other:?}"),
    };
    Ok(stream)
}

/// Downmix interleaved frames to mono by averaging channels.
fn mono<T: Copy>(data: &[T], channels: usize, conv: impl Fn(T) -> f32) -> Vec<f32> {
    if channels <= 1 {
        return data.iter().map(|&s| conv(s)).collect();
    }
    data.chunks(channels)
        .map(|frame| frame.iter().map(|&s| conv(s)).sum::<f32>() / channels as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_averages_multichannel_frames() {
        // Stereo: [L,R, L,R] → average per frame.
        let stereo = [1.0f32, 3.0, 2.0, 4.0];
        assert_eq!(mono(&stereo, 2, |s| s), vec![2.0, 3.0]);
    }

    #[test]
    fn mono_passes_through_single_channel() {
        let m = [0.1f32, -0.2, 0.3];
        assert_eq!(mono(&m, 1, |s| s), vec![0.1, -0.2, 0.3]);
    }

    #[test]
    fn write_wav_roundtrips_samples() {
        let dir = std::env::temp_dir();
        let path = dir.join("li-capture-test.wav");
        write_wav_16le(&path, &[0.0, 0.5, -0.5, 1.0], 16_000).unwrap();
        let reader = hound::WavReader::open(&path).unwrap();
        assert_eq!(reader.spec().sample_rate, 16_000);
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.len(), 4);
        let _ = std::fs::remove_file(&path);
    }
}
