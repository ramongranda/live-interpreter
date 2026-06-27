# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Live Interpreter is a local-first voice interpreter: you speak one language, it transcribes (Whisper) → translates (Ollama) → synthesizes (Qwen3-TTS) → plays the translated voice into a PipeWire virtual microphone (`live-interpreter-mic-source`) that any meeting/call/streaming app can select. Only `es_to_en` and `en_to_es` directions exist (`src/types.rs` `Direction`).

## Build, test, run

```bash
# GPU server (default binary). whisper-rs compiles whisper.cpp → needs a C/C++ toolchain.
cargo build --release --bin live-interpreter
cargo build --release --features cuda --bin live-interpreter   # Whisper on GPU

# Native Tauri desktop app (needs webkit2gtk system libs). Feature gates the binary + build.rs.
cargo run --features desktop-native --bin live-interpreter-desktop

# Tests. Domain logic lives behind the lib; this runs fast and needs no GPU/model:
cargo test --lib
cargo test                       # also compiles the binaries (server links whisper.cpp)
cargo test strip_think           # single test by name
cargo test --lib best_gpu_selects_largest_vram_card

cargo fmt
git diff --check
```

Stated pre-merge verification (README): `cargo fmt` + `cargo test` + `cargo build --release --features desktop-native --bin live-interpreter-desktop` + `cargo build --release --bin live-interpreter --bin live-interpreter-control --bin live-interpreter-client --bin live-interpreter-ws-smoke`.

Edition is `2024`. Default features are empty (no `cuda`, no `desktop-native`).

## Runtime stack (operational, not `cargo`)

The server is one process in a stack of external services started by `scripts/start-local-stack.sh`:

- `live-interpreter` server — HTTP/WS, port **8787** (`src/main.rs`).
- Qwen3-TTS service — port **8020**, from `vendor/qwen3_tts_rs` (installed via `scripts/install-qwen3-tts-rs.sh`).
- Ollama — port **11434**, model `translator:latest`.
- PipeWire virtual mic — `scripts/create-virtual-mic.sh`.
- `live-interpreter-client` — port **8790**; `live-interpreter-control` browser panel — port **8798**.

Whisper ggml model is **not** in the repo: `scripts/download-whisper-model.sh large-v3-turbo`. Server refuses to start without it (`src/asr.rs` `AsrEngine::load`).

## Architecture

**Two roles, one shared domain core.** The product splits into a **server** (owns ASR/translate/TTS/GPU) and a **client** (captures mic, talks to server) — they can run on the same box or different machines.

- `src/lib.rs` exposes `desktop`, `translate`, and `types`. `src/desktop.rs` is the main shared domain core; `translate`/`types` live in the lib so the binaries **and** `benches/` can reach them. The server-only modules (`asr`, `tts`, `routes`, `config`) stay private to the `live-interpreter` binary crate — this deliberately keeps whisper.cpp out of the lighter `live-interpreter-control` / `-desktop` binaries, which link the lib. Binary modules reference shared types via `live_interpreter::types` / `live_interpreter::translate`, not `crate::`.
- `src/desktop.rs` is the domain/application core (process orchestration, GPU preflight, status projection, voice-profile persistence) **plus its own unit tests**. It has no UI and no I/O framework — it shells out and reads PID files.
- The two control front-ends are **thin adapters over `desktop.rs`**, sharing `DesktopConfig`, `collect_status`, `start_server`/`stop_server`, `start_client`/`stop_client`:
  - `src/bin/live-interpreter-control.rs` — HTTP/Axum browser panel (serves `static/index.html`).
  - `src/bin/live-interpreter-desktop.rs` — Tauri IPC adapter (serves `desktop/index.html`), additionally owns native `cpal` voice recording.
- `src/bin/live-interpreter-client.rs` — standalone client runtime: `cpal` capture + VAD → WebSocket to the server's `/v1/stream/meeting` → plays returned audio to `live-interpreter-mic-sink`. It does **not** use the lib; it re-declares its own `Direction`/`StreamStart`/`StreamEvent`.

**Server request pipeline** (`src/routes.rs` → `process_audio_path`): Whisper transcribe → Ollama translate → Qwen3-TTS synthesize → persist JSON to `data/transcripts/`. Same path serves `POST /v1/interpret/file`, `POST /v1/interpret/text`, and the `GET /v1/stream/meeting` WebSocket. The WS protocol is the `StreamEvent` enum in `src/types.rs` (`ready`/`listening`/`processing`/`transcript_final`/`translation_final`/`audio_start` then a binary WAV frame/`done`/`error`).

**Things that bite:**
- **GPU gate.** `start_server` (both adapters) runs `gpu_preflight` and refuses to start if total VRAM `< LI_MIN_SERVER_VRAM_MB` (default 8000). Logic + tests are in `desktop.rs` (`parse_gpu_preflight`, `best_gpu`); it parses `nvidia-smi` CSV.
- **Voice profile injection.** Saving a voice writes `data/voice/reference.{wav,txt}`; on next `start_server` these are passed as `LI_VOICE_REF`/`LI_VOICE_REF_TEXT` env to the stack, and `tts.rs` base64-encodes the WAV into each TTS request. Requires a server restart to take effect.
- **Process lifecycle** is PID files under `data/logs/*.pid`; "running" = `/proc/<pid>` exists (`pid_alive`). "Healthy" = HTTP `/health` poll. These are independent signals in `AppStatus`.
- **Spanish user-facing strings live in the domain layer** (`role_hint`, `save_voice_profile` messages, `gpu_preflight`). Intentional — keep new user-facing status text in Spanish to match.
- **Translation is a pluggable backend.** `src/translate/` is an enum `Translator { Http, Candle }` chosen at startup by `LI_TRANSLATE_BACKEND` (default `http`). `http.rs` = the original Ollama client (plus `keep_alive` to pin the model in VRAM); `candle.rs` (feature `translate-candle`) = in-process quantized Qwen2 GGUF inference, no Ollama process. Both strip `<think>…</think>` (reasoning models) via `strip_think`. First-token latency bench: `benches/translate_latency.rs`. GPU candle build: `--features candle-cuda`; fetch the model with `scripts/download-translate-model.sh`.
- The repo recently renamed from "OVT" → "Live Interpreter"; per README, do **not** add compatibility shims for old names (pre-production).

## Configuration

Server reads `LI_*` env in `src/config.rs::from_env`; adapters/client read additional `LI_*` in `desktop.rs`/the client bin. Key ones: `LI_BIND` (server addr), `LI_WHISPER_MODEL`, `LI_OLLAMA_MODEL` (default `translator:latest`), `LI_QWEN_TTS_URL`/`_MODEL`/`_VOICE`, `LI_MIN_SERVER_VRAM_MB`, `LI_AUTH_TOKEN` (optional bearer/`?token=` for LAN). Client VAD tuning (`LI_CLIENT_VAD_THRESHOLD`, `LI_CLIENT_SILENCE_MS`, …) is documented in `docs/meeting-client.md`. Full list in README "Configuration".

## Docs

`docs/native-desktop.md` (adapter boundary), `docs/control-panel.md`, `docs/meeting-client.md`, `docs/qwen3-tts-contract.md` (note: the actual request `tts.rs` sends has more fields than that doc — trust `QwenTtsRequest`), `docs/rust-native-inference.md`.
