# Olares Voice Translator

Local real-time voice translator for Microsoft Teams.

Primary goal:

```text
You speak Spanish -> Teams receives generated English voice only.
Teams English audio -> you see private Spanish captions.
```

Core path:

- Rust server with `axum`.
- Local ASR with `whisper-rs` / `whisper.cpp`.
- Local translation with Ollama.
- Local cloned/streaming TTS through a Qwen3-TTS-compatible HTTP endpoint.
- PipeWire virtual microphone for Teams.

## Quick commands

```bash
cd /home/rgranda/workspaces/olares-voice-translator
./scripts/build-gpu.sh
```

Download the ASR model:

```bash
./scripts/download-whisper-model.sh large-v3-turbo
./scripts/install-qwen3-tts-rs.sh
./scripts/start-local-stack.sh
```

The default stack uses the CUDA Qwen3-TTS runtime in `vendor/qwen3_tts_rs`. Use the CPU fallback only if CUDA libraries are unavailable:

```bash
OVT_QWEN_INSTALL_DIR=$PWD/vendor/qwen3_tts_rs_cpu ./scripts/start-local-stack.sh
```

See:

- `docs/teams-realtime.md`
- `docs/meeting-client.md`
- `docs/qwen3-tts-contract.md`
- `docs/rust-native-inference.md`

## Remote meeting client

Run the GPU server on the NVIDIA machine:

```bash
OVT_BIND=0.0.0.0:8787 ./scripts/start-local-stack.sh
```

Run the client on the computer where Teams is open:

```bash
cargo build --release --bin ovt-meeting-client
OVT_SERVER_URL=http://SERVER_IP:8787 ./scripts/start-meeting-client.sh
```

Open `http://127.0.0.1:8790` for transcription, translation, mute controls,
and direction switching.

## Environment

- `OVT_BIND`: server bind address, default `127.0.0.1:8787`
- `OVT_DATA_DIR`: data directory, default `data`
- `OVT_WHISPER_MODEL`: whisper.cpp ggml model path
- `OVT_WHISPER_THREADS`: ASR threads, default `8`
- `OVT_OLLAMA_URL`: Ollama URL, default `http://127.0.0.1:11434`
- `OVT_OLLAMA_MODEL`: translation model, default `translator:latest`
- `OVT_QWEN_TTS_URL`: Qwen3-TTS-compatible server, default `http://127.0.0.1:8020`
- `OVT_QWEN_TTS_MODEL`: TTS model name, default `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
- `OVT_QWEN_TTS_VOICE`: TTS voice name, default `alloy`
- `OVT_VOICE_REF`: optional reference voice wav for your own cloned voice
- `OVT_VOICE_REF_TEXT`: transcript of `OVT_VOICE_REF`, required for higher-quality voice cloning
- `OVT_FFMPEG_BIN`: ffmpeg binary, default `/snap/bin/ffmpeg` when present, otherwise `ffmpeg`
