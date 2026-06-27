//! Virtual microphone: the runtime's audio output to a PipeWire `Audio/Source`
//! node that any meeting/call app can select as its mic.
//!
//! Candle-native design: the runtime *produces* the audio in-process, so the
//! virtual mic is a single PipeWire **source** stream we push PCM into — simpler
//! than the old `pw-loopback` sink+source pair. The real PipeWire backend lives
//! behind the `native-audio` feature (needs `libpipewire-0.3` + `libclang`); the
//! default build ships only the trait + a testable mock.

use crate::voice::AudioFrame;
use anyhow::Result;
use std::sync::Mutex;

/// Sink the pipeline writes synthesized audio to. The concrete implementation
/// routes it to the OS virtual microphone.
pub trait AudioOutput: Send + Sync {
    /// Queue one PCM frame for playout to the virtual mic source.
    fn submit(&self, frame: &AudioFrame) -> Result<()>;
}

/// Test double: records every submitted frame so pipeline wiring is verifiable
/// without a running PipeWire daemon.
#[derive(Default)]
pub struct MockAudioOutput {
    pub frames: Mutex<Vec<AudioFrame>>,
}

impl MockAudioOutput {
    pub fn total_pcm_bytes(&self) -> usize {
        self.frames
            .lock()
            .unwrap()
            .iter()
            .map(|f| f.pcm.len())
            .sum()
    }

    pub fn frame_count(&self) -> usize {
        self.frames.lock().unwrap().len()
    }
}

impl AudioOutput for MockAudioOutput {
    fn submit(&self, frame: &AudioFrame) -> Result<()> {
        self.frames.lock().unwrap().push(frame.clone());
        Ok(())
    }
}

#[cfg(feature = "native-audio")]
pub use native::PipewireVirtualMic;

#[cfg(feature = "native-audio")]
mod native {
    use super::AudioOutput;
    use crate::types::AudioSpec;
    use crate::voice::AudioFrame;
    use anyhow::{Context, Result, anyhow};
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};
    use std::thread::JoinHandle;

    /// Default node name exposed to meeting apps (`<name>` → selectable source).
    pub const DEFAULT_NODE_NAME: &str = "live-interpreter-mic-source";

    enum Control {
        Quit,
    }

    /// Live PipeWire source node, fed PCM from any thread via `submit`. Runs its
    /// own main loop on a dedicated OS thread (the PipeWire loop is `!Send`).
    pub struct PipewireVirtualMic {
        ring: Arc<Mutex<VecDeque<u8>>>,
        control: pipewire::channel::Sender<Control>,
        thread: Option<JoinHandle<()>>,
        spec: AudioSpec,
    }

    impl PipewireVirtualMic {
        pub fn spawn(spec: AudioSpec, node_name: &str) -> Result<Self> {
            let ring: Arc<Mutex<VecDeque<u8>>> = Arc::new(Mutex::new(VecDeque::new()));
            let (control_tx, control_rx) = pipewire::channel::channel::<Control>();
            let node_name = node_name.to_string();
            let ring_loop = Arc::clone(&ring);

            // Hand the loop a one-shot result channel so spawn() can surface
            // setup errors instead of silently dying in the thread.
            let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();

            let thread = std::thread::Builder::new()
                .name("pw-virtual-mic".into())
                .spawn(
                    move || match run_loop(spec, &node_name, ring_loop, control_rx, &ready_tx) {
                        Ok(()) => {}
                        Err(error) => {
                            let _ = ready_tx.send(Err(format!("{error:#}")));
                        }
                    },
                )
                .context("failed to spawn PipeWire mic thread")?;

            match ready_rx.recv() {
                Ok(Ok(())) => Ok(Self {
                    ring,
                    control: control_tx,
                    thread: Some(thread),
                    spec,
                }),
                Ok(Err(message)) => Err(anyhow!("PipeWire virtual mic setup failed: {message}")),
                Err(_) => Err(anyhow!(
                    "PipeWire mic thread exited before signalling readiness"
                )),
            }
        }

        /// Spawn with the standard `live-interpreter-mic-source` node name.
        pub fn spawn_default(spec: AudioSpec) -> Result<Self> {
            Self::spawn(spec, DEFAULT_NODE_NAME)
        }

        pub fn spec(&self) -> AudioSpec {
            self.spec
        }
    }

    impl AudioOutput for PipewireVirtualMic {
        fn submit(&self, frame: &AudioFrame) -> Result<()> {
            // Resampling/format negotiation is a later step; for now we trust the
            // backend to emit at the mic's configured spec.
            self.ring.lock().unwrap().extend(frame.pcm.iter().copied());
            Ok(())
        }
    }

    impl Drop for PipewireVirtualMic {
        fn drop(&mut self) {
            let _ = self.control.send(Control::Quit);
            if let Some(handle) = self.thread.take() {
                let _ = handle.join();
            }
        }
    }

    fn run_loop(
        spec: AudioSpec,
        node_name: &str,
        ring: Arc<Mutex<VecDeque<u8>>>,
        control_rx: pipewire::channel::Receiver<Control>,
        ready_tx: &std::sync::mpsc::Sender<Result<(), String>>,
    ) -> Result<()> {
        use pipewire::{properties::properties, spa, stream::Stream, stream::StreamFlags};

        pipewire::init();
        let mainloop =
            pipewire::main_loop::MainLoop::new(None).context("pw MainLoop::new failed")?;
        let context =
            pipewire::context::Context::new(&mainloop).context("pw Context::new failed")?;
        let core = context
            .connect(None)
            .context("pw Context::connect failed")?;

        let stream = Stream::new(
            &core,
            "live-interpreter-mic",
            properties! {
                *pipewire::keys::MEDIA_TYPE => "Audio",
                *pipewire::keys::MEDIA_CATEGORY => "Playback",
                *pipewire::keys::MEDIA_ROLE => "Communication",
                *pipewire::keys::MEDIA_CLASS => "Audio/Source",
                *pipewire::keys::NODE_NAME => node_name,
                *pipewire::keys::NODE_DESCRIPTION => node_name,
            },
        )
        .context("pw Stream::new failed")?;

        let channels = spec.channels.max(1) as usize;
        let stride = channels * 2; // s16le
        let ring_cb = Arc::clone(&ring);

        let _listener = stream
            .add_local_listener_with_user_data(())
            .process(move |stream, _| {
                let Some(mut buffer) = stream.dequeue_buffer() else {
                    return;
                };
                let datas = buffer.datas_mut();
                let Some(data) = datas.first_mut() else {
                    return;
                };
                let Some(dst) = data.data() else {
                    return;
                };
                let mut queue = ring_cb.lock().unwrap();
                let available = queue.len().min(dst.len());
                for slot in dst.iter_mut().take(available) {
                    *slot = queue.pop_front().unwrap_or(0);
                }
                // Underrun → fill the rest with silence to avoid glitches.
                for slot in dst.iter_mut().skip(available) {
                    *slot = 0;
                }
                let written = dst.len();
                let chunk = data.chunk_mut();
                *chunk.offset_mut() = 0;
                *chunk.stride_mut() = stride as i32;
                *chunk.size_mut() = written as u32;
            })
            .register()
            .context("pw stream listener register failed")?;

        // Build the EnumFormat param pod (S16LE, configured rate/channels).
        let mut audio_info = spa::param::audio::AudioInfoRaw::new();
        audio_info.set_format(spa::param::audio::AudioFormat::S16LE);
        audio_info.set_rate(spec.sample_rate);
        audio_info.set_channels(channels as u32);
        let values: Vec<u8> = spa::pod::serialize::PodSerializer::serialize(
            std::io::Cursor::new(Vec::new()),
            &spa::pod::Value::Object(spa::pod::Object {
                type_: spa::sys::SPA_TYPE_OBJECT_Format,
                id: spa::sys::SPA_PARAM_EnumFormat,
                properties: audio_info.into(),
            }),
        )
        .map_err(|error| anyhow!("pod serialize failed: {error:?}"))?
        .0
        .into_inner();
        let mut params = [spa::pod::Pod::from_bytes(&values)
            .ok_or_else(|| anyhow!("invalid audio format pod"))?];

        stream
            .connect(
                spa::utils::Direction::Output,
                None,
                StreamFlags::AUTOCONNECT | StreamFlags::MAP_BUFFERS | StreamFlags::RT_PROCESS,
                &mut params,
            )
            .context("pw stream connect failed")?;

        // Wire the quit channel into the loop and signal readiness.
        let main_quit = mainloop.clone();
        let _control = control_rx.attach(mainloop.loop_(), move |control| match control {
            Control::Quit => main_quit.quit(),
        });

        let _ = ready_tx.send(Ok(()));
        mainloop.run();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AudioSpec;

    fn frame(bytes: usize) -> AudioFrame {
        AudioFrame {
            spec: AudioSpec::mono_s16le(24_000),
            pcm: vec![7u8; bytes],
        }
    }

    #[test]
    fn mock_output_collects_frames() {
        let out = MockAudioOutput::default();
        out.submit(&frame(8)).unwrap();
        out.submit(&frame(4)).unwrap();
        assert_eq!(out.frame_count(), 2);
        assert_eq!(out.total_pcm_bytes(), 12);
    }

    #[test]
    fn mock_output_is_object_safe_trait() {
        let out: Box<dyn AudioOutput> = Box::new(MockAudioOutput::default());
        out.submit(&frame(2)).unwrap();
    }
}
