# Live Interpreter

Live Interpreter is a native local-first voice interpretation app for calls, meetings, streaming tools, and any desktop software that can select an audio input device.

The product goal is simple:

```text
You speak in one language.
Live Interpreter transcribes, translates, synthesizes speech, and exposes the translated voice as a virtual microphone.
Any meeting app receives only the translated voice.
```

It is not tied to a specific meeting platform. The app is designed to work with any application that can use `live-interpreter-mic-source` as its microphone.

## What The App Does

- Runs a native Rust/Tauri desktop control app.
- Starts and stops the GPU server stack from the UI.
- Blocks server mode when the NVIDIA GPU preflight does not meet the minimum VRAM requirement.
- Runs client and server on the same PC, or lets a lightweight client connect to a GPU server on another machine.
- Captures local microphone audio.
- Performs local speech-to-text with Whisper.
- Translates text through a local Ollama model with a bounded sliding context window.
- Synthesizes translated speech through a Qwen3-TTS-compatible service.
- Sends the generated voice to a PipeWire virtual microphone.
- Lets the user record a personal reference voice inside the native app.
- Stores the voice reference locally and injects it into the TTS stack on server start.
- Provides a typed actor/channel design for low-latency audio streaming and backpressure.
- Provides a first P2P mesh module for LAN GPU discovery, health gossip, and audio task failover.

## Product Shape

The app has two runtime roles:

- **Server GPU**: owns ASR, translation, TTS, virtual microphone output, and GPU-heavy processes.
- **Client**: captures microphone input and talks to the server. It can run on the same PC as the server or on another computer.

The native UI exposes both roles:

- Runtime controls for starting and stopping server/client processes.
- GPU preflight and health status.
- Qwen TTS, server, client, and microphone bridge status.
- Mesh controls for starting the LAN P2P runtime as a GPU provider or consumer.
- A voice studio where the user reads a fixed reference text, records audio, listens back, and saves the voice profile.
- A process view for GPU activity and VRAM usage.
- A console area for command output and operational errors.

## Architecture

The codebase is intentionally split by responsibility:

- `src/main.rs`: HTTP/WebSocket server entrypoint.
- `src/routes.rs`: API and WebSocket routes.
- `src/asr.rs`: Whisper speech-to-text integration.
- `src/translate/`: translation backends, prompts, streaming output, and sliding context.
- `src/tts.rs`: Qwen3-TTS-compatible HTTP client.
- `src/config.rs`: runtime configuration from `LI_*` environment variables.
- `src/desktop.rs`: desktop domain services, process orchestration, GPU preflight, voice profile persistence, streaming actor contracts, and tests.
- `src/mesh.rs`: libp2p LAN mesh, mDNS discovery, Gossipsub health, request-response audio task transport, provider ranking, and failover.
- `src/bin/live-interpreter-desktop.rs`: native Tauri adapter and native microphone recording commands.
- `src/bin/live-interpreter-control.rs`: browser control panel adapter.
- `src/bin/live-interpreter-client.rs`: meeting/client runtime.
- `src/bin/live-interpreter-ws-smoke.rs`: WebSocket smoke-test tool.
- `desktop/index.html`: embedded Tauri UI.
- `static/index.html`: browser control UI.
- `scripts/`: local setup, build, start, stop, install, and audio bridge scripts.
- `docs/`: operational contracts and technical notes.

The intended direction is TDD and clean boundaries:

- Keep core behavior in testable Rust modules.
- Keep UI adapters thin.
- Keep process orchestration in `src/desktop.rs`.
- Keep platform scripts explicit and small.
- Avoid compatibility shims for old product names because this app is not in production yet.

## Runtime Flow

```text
Microphone input
  -> Live Interpreter client
  -> Live Interpreter server
  -> Whisper ASR
  -> Ollama translation
  -> Qwen3-TTS speech synthesis
  -> PipeWire virtual microphone
  -> Any meeting/call/streaming app
```

## Streaming Architecture

The streaming design in `src/desktop.rs` is actor-oriented and built around bounded Tokio channels:

```text
RawAudioFrame f32 samples
  -> mpsc bounded channel
  -> AsrEngine actor
  -> TranscriptFragment
  -> mpsc bounded channel
  -> TranslationEngine actor
  -> TranslatedText
  -> mpsc bounded channel
  -> TtsEngine actor
  -> GeneratedAudio PCM buffer
  -> mpsc bounded channel
  -> AudioSink / PipeWire bridge
```

Backpressure is explicit. Every stage sends into a bounded `tokio::sync::mpsc` channel, so a saturated GPU, slow translator, or delayed TTS stage naturally slows upstream producers instead of growing memory without limit.

Default channel capacities are defined by `StreamingBackpressure`:

- raw audio frames: `8`
- transcript fragments: `8`
- translated texts: `4`
- generated audio buffers: `4`

The heavy AI modules are abstracted behind traits so the orchestration can be tested without loading models:

- `AsrEngine`
- `TranslationEngine`
- `TtsEngine`
- `AudioSink`

The actor tests verify bounded-channel behavior and end-to-end movement through ASR, translation, TTS, and sink mocks.

## Local P2P Mesh

`src/mesh.rs` introduces the first mesh network layer for Live Interpreter. Each app instance can run the same node code and advertise its current role:

- `GpuProvider`: a node with NVIDIA GPU capacity available to process audio.
- `Consumer`: a lightweight node that captures audio and delegates processing to the best provider in the LAN mesh.

The mesh uses `libp2p` with:

- `mDNS` for zero-config LAN peer discovery.
- `Gossipsub` topic `live-interpreter/disponibilidad-gpu` for recurrent provider health.
- `request-response` with a custom `bincode` codec for direct binary `AudioChunk` requests and `AudioTaskResult` responses.
- TCP + Noise + Yamux transport through the Tokio runtime.

Provider health is modeled as:

```rust
MeshHealth {
    peer_id,
    role,
    free_vram_mb,
    total_vram_mb,
    active_sessions,
    unix_ms,
}
```

GPU telemetry is abstracted through `GpuTelemetry`. The production implementation `NvmlGpuTelemetry` reads NVIDIA VRAM and active compute process count via NVML. Tests use `NoopGpuTelemetry`.

Audio processing is abstracted through `MeshAudioProcessor`, so libp2p stays separated from model code. The native desktop provider now wires incoming mesh `AudioChunk` requests into the real ASR -> translation -> TTS pipeline:

```text
AudioChunk
  -> temporary WAV in data/mesh
  -> Whisper ASR
  -> Translator with sliding context
  -> Qwen3-TTS
  -> AudioTaskResult
```

The native desktop app can now start and stop the mesh runtime directly:

- `Arrancar consumidor`: joins the LAN mesh and searches for GPU providers.
- `Arrancar proveedor GPU`: joins the LAN mesh and publishes this node as a GPU provider.
- `Capturar por mesh`: captures local microphone phrases, sends them to the selected GPU provider as `AudioChunk`, writes the returned audio to `data/mesh/results`, and plays it through `pw-play`.
- `Parar captura`: stops the consumer microphone capture loop.
- `Parar mesh`: shuts down the mesh command loop.

Consumer mode does not load local ASR/TTS models. Provider mode initializes the local model stack and fails early if the GPU node is not ready to process audio. Mesh capture supports `ES -> EN` and `EN -> ES` from the native UI.

Failover is local and deterministic:

- stale providers are pruned after `provider_stale_after`,
- providers are ranked by free VRAM first and active sessions second,
- failed request-response calls are retried on the next best provider,
- if all providers fail, the caller receives a typed error through the oneshot reply channel.

## Intelligent Translation

The translation module keeps a bounded sliding window of recent translation turns:

```rust
TranslationTurn {
    original,
    translated,
}
```

`TranslationBuffer` stores only the last N interactions and clamps text length per turn. This prevents unbounded memory growth during long meetings.

For each HTTP translation request, `HttpTranslator` builds a dynamic Ollama `system` prompt containing the recent context, while the current utterance remains the user prompt. The model receives meeting continuity without mixing the current sentence into the historical buffer before the translation succeeds.

The buffer has three memory controls:

- `max_interactions`: maximum number of turns kept.
- `max_chars_per_text`: maximum retained text length per original/translation.
- `silence_reset_ms`: long-silence threshold that clears context.

`Translator::observe_silence(Duration)` is the public hook for ASR/client code to reset context after a long pause. The HTTP backend stores the buffer behind `Arc<tokio::sync::Mutex<_>>`, so cloned app state shares the same context safely.

Serde is strict for translation context types through `#[serde(deny_unknown_fields)]`.

When a voice profile exists, server startup sets:

- `LI_VOICE_REF=data/voice/reference.wav`
- `LI_VOICE_REF_TEXT=<saved reference transcript>`
- Qwen3-TTS model defaults suitable for reference-voice synthesis.

## Native Desktop App

Build and run the native app:

```bash
cargo run --features desktop-native --bin live-interpreter-desktop
```

Build the release binary:

```bash
cargo build --release --features desktop-native --bin live-interpreter-desktop
```

Run the release binary:

```bash
target/release/live-interpreter-desktop
```

Install the desktop launcher and user integration:

```bash
./scripts/install-live-interpreter-desktop.sh
```

Open the installed app:

```bash
./scripts/open-live-interpreter-desktop.sh
```

## Local Stack

Build GPU binaries:

```bash
./scripts/build-gpu.sh
```

Download the ASR model:

```bash
./scripts/download-whisper-model.sh large-v3-turbo
```

Install Qwen3-TTS runtime:

```bash
./scripts/install-qwen3-tts-rs.sh
```

Start the local server stack:

```bash
./scripts/start-local-stack.sh
```

Stop the local stack:

```bash
./scripts/stop-local-stack.sh
```

The default Qwen runtime is expected in:

```text
vendor/qwen3_tts_rs
```

CPU fallback can be selected explicitly:

```bash
LI_QWEN_INSTALL_DIR=$PWD/vendor/qwen3_tts_rs_cpu ./scripts/start-local-stack.sh
```

## Browser Control Panel

The native app is the primary UI, but the browser control panel remains useful for debugging.

Start it:

```bash
./scripts/start-live-interpreter-control.sh
```

Open:

```text
http://127.0.0.1:8798
```

## Client And Server On Different Machines

Run the GPU server on the machine with the NVIDIA GPU:

```bash
LI_BIND=0.0.0.0:8787 ./scripts/start-local-stack.sh
```

Run the client on the computer where the meeting app is open:

```bash
cargo build --release --bin live-interpreter-client
LI_SERVER_URL=http://SERVER_IP:8787 ./scripts/start-meeting-client.sh
```

Open the client controls:

```text
http://127.0.0.1:8790
```

## Voice Profile

The native app includes a voice studio.

Default reference text:

```text
En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivia un hidalgo de los de lanza en astillero, adarga antigua, rocin flaco y galgo corredor.
```

Flow:

1. Open the native app.
2. Go to `Mi voz`.
3. Press `Grabar`.
4. Read the reference text naturally.
5. Press `Parar`.
6. Listen to the captured sample.
7. Press `Guardar voz`.
8. Restart the GPU server from the app.

Saved files:

```text
data/voice/reference.wav
data/voice/reference.txt
```

## Virtual Microphone

The app creates a PipeWire virtual microphone source:

```text
live-interpreter-mic-source
```

Select that device as the microphone in the target app.

## Smoke Test

Build and run the WebSocket smoke-test tool with an existing WAV:

```bash
cargo run --release --bin live-interpreter-ws-smoke -- /tmp/live-interpreter-gpu-tts.wav
```

## Verification

Recommended checks before considering a change ready:

```bash
cargo fmt
cargo test
cargo build --release --features desktop-native --bin live-interpreter-desktop
cargo build --release --bin live-interpreter --bin live-interpreter-control --bin live-interpreter-client --bin live-interpreter-ws-smoke
git diff --check
```

Current local baseline:

```text
cargo test: 33 passed
desktop release build: OK
CLI release build: OK
git diff --check: OK
```

## Configuration

- `LI_BIND`: server bind address. Default: `127.0.0.1:8787`.
- `LI_DATA_DIR`: data directory. Default: `data`.
- `LI_WHISPER_MODEL`: whisper.cpp ggml model path.
- `LI_WHISPER_THREADS`: ASR thread count. Default: `8`.
- `LI_OLLAMA_URL`: Ollama URL. Default: `http://127.0.0.1:11434`.
- `LI_OLLAMA_MODEL`: translation model. Default: `translator:latest`.
- `LI_OLLAMA_KEEP_ALIVE`: Ollama model residency hint. Default: `30m`.
- `LI_TRANSLATE_BACKEND`: translation backend, `http` by default. `candle` requires a feature build.
- `LI_CANDLE_GGUF`: Candle GGUF model path when using `LI_TRANSLATE_BACKEND=candle`.
- `LI_CANDLE_TOKENIZER`: Candle tokenizer path when using `LI_TRANSLATE_BACKEND=candle`.
- `LI_CANDLE_MAX_TOKENS`: maximum generated tokens for Candle backend.
- `LI_QWEN_TTS_URL`: Qwen3-TTS-compatible server. Default: `http://127.0.0.1:8020`.
- `LI_QWEN_TTS_MODEL`: TTS model name. Default: `Qwen/Qwen3-TTS-12Hz-0.6B-Base`.
- `LI_QWEN_TTS_VOICE`: TTS voice name. Default: `alloy`.
- `LI_MIN_SERVER_VRAM_MB`: minimum NVIDIA VRAM required to enable server mode. Default: `8000`.
- `LI_VOICE_REF`: optional reference voice WAV.
- `LI_VOICE_REF_TEXT`: transcript of `LI_VOICE_REF`.
- `LI_FFMPEG_BIN`: ffmpeg binary. Default: `/snap/bin/ffmpeg` when present, otherwise `ffmpeg`.
- `LI_AUTH_TOKEN`: optional bearer/query token for LAN clients.

## Documentation

- `docs/native-desktop.md`: native app boundary and commands.
- `docs/control-panel.md`: browser control panel.
- `docs/meeting-client.md`: client runtime.
- `docs/qwen3-tts-contract.md`: TTS HTTP contract.
- `docs/rust-native-inference.md`: native inference notes.
