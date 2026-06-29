# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Live Interpreter is a local-first voice interpreter: you speak one language, it transcribes (Whisper) → translates (Ollama) → synthesizes (Qwen3-TTS) → plays the translated voice into a PipeWire virtual microphone (`live-interpreter-mic-source`) that any meeting/call/streaming app can select. Only `es_to_en` and `en_to_es` directions exist (`src/types.rs` `Direction`).

## Build, test, run

```bash
# Server (default binary). whisper-rs compiles whisper.cpp → needs a C/C++ toolchain + cmake.
cargo build --release --bin live-interpreter
cargo build --release --features cuda --bin live-interpreter      # Whisper on GPU (~20× faster ASR)

# Native-audio binaries (in-process PipeWire virtual mic): li-interpret, li-mesh, li-voice-demo.
cargo build --release --features native-audio
cargo build --release --features cuda,native-audio                # or ./scripts/build-gpu.sh

# Tests. Domain logic lives behind the lib; this runs fast and needs no GPU/model:
cargo test --lib
cargo test                       # also compiles the binaries (server links whisper.cpp)
cargo test strip_think           # single test by name

cargo fmt --all
git diff --check
```

Pre-merge verification mirrors CI (`.github/workflows/ci.yml`): `cargo fmt --all --check` + `cargo clippy --all-targets --features native-audio -- -D warnings` + `cargo test --lib` + `cargo build --release --features native-audio`. System build deps: `build-essential cmake pkg-config clang libclang-dev libasound2-dev libpipewire-0.3-dev libspa-0.2-dev`.

Edition is `2024`. Default features are empty (no `cuda`, no `native-audio`).

## Binaries

| Binary                      | Source                                 | Feature        | Notes                                                    |
| --------------------------- | -------------------------------------- | -------------- | -------------------------------------------------------- |
| `live-interpreter`          | `src/main.rs`                          | —              | HTTP/WS server, `:8787`.                                 |
| `li-control`                | `src/bin/li-control.rs`                | —              | FSM control panel, `:8799`.                              |
| `li-interpret`              | `src/bin/li-interpret.rs`              | `native-audio` | Full local loop → virtual mic.                           |
| `li-mesh`                   | `src/bin/li-mesh.rs`                   | `native-audio` | LAN mesh node (`LI_ROLE`).                               |
| `li-voice-demo`             | `src/bin/li-voice-demo.rs`             | `native-audio` | Text → cloned voice → virtual mic.                       |
| `live-interpreter-client`   | `src/bin/live-interpreter-client.rs`   | —              | Meeting client engine, `:8790` (spawned by the runtime). |
| `live-interpreter-ws-smoke` | `src/bin/live-interpreter-ws-smoke.rs` | —              | WebSocket smoke-test tool.                               |

## Runtime stack (operational, not `cargo`)

The server is one process alongside external services:

- `live-interpreter` server — HTTP/WS, port **8787** (`src/main.rs`).
- Qwen3-TTS service — port **8020**, from `vendor/qwen3_tts_rs` (`scripts/install-qwen3-tts-rs.sh`, then `scripts/start-qwen3-tts.sh`).
- Ollama — port **11434**, model `translator:latest`.
- PipeWire virtual mic `live-interpreter-mic-source` — created **in-process** by the `native-audio` runtime (`src/virtual_mic.rs`).
- `li-control` browser panel — port **8799** (serves `static/fsm-ui.html`).
- `live-interpreter-client` meeting client — port **8790**.

Whisper ggml model is **not** in the repo: `scripts/download-whisper-model.sh large-v3-turbo`. Server refuses to start without it (`src/asr.rs` `AsrEngine::load`).

## Architecture

**Server + clients over a shared lib.** `src/lib.rs` exposes the whole domain: `asr`, `capture`, `config`, `desktop`, `events`, `mesh`, `mesh_pipeline`, `pipeline`, `quality`, `runtime`, `translate`, `tts`, `types`, `virtual_mic`, `voice`, `vram`. Binaries reference shared types via `live_interpreter::…`, not `crate::`.

- `src/main.rs` → `src/routes.rs` (`process_audio_path`): Whisper transcribe → Ollama translate → Qwen3-TTS synthesize → persist JSON to `data/transcripts/`. Same path serves `POST /v1/interpret/file`, `POST /v1/interpret/text`, and the `GET /v1/stream/meeting` WebSocket. The WS protocol is the `StreamEvent` enum in `src/types.rs` (`ready`/`listening`/`processing`/`transcript_final`/`translation_final`/`audio_start` then a binary WAV frame/`done`/`error`).
- `src/runtime.rs` — `LiveRuntime`: the FSM process orchestrator behind the control panel. `ManagedService` (`QwenTts`/`Server`/`Mic`/`Client`) with `pgrep`-pattern lifecycle (no `nvidia-smi`, no Bash). `li-control` is a thin Axum adapter over it (`/api/status`, role start/stop).
- `src/desktop.rs` — domain core kept after the R8 purge: `DesktopConfig`, voice-profile persistence, the streaming actor contracts + traits, `GpuPreflight`, `pid_alive`. Plus its own unit tests.
- `src/mesh.rs` + `src/mesh_pipeline.rs` — libp2p LAN mesh: mDNS discovery, Gossipsub health, request-response audio transport, latency-aware provider ranking, clause-by-clause streamed delivery (ack-gated ordering).
- `src/bin/live-interpreter-client.rs` — standalone meeting client: `cpal` capture + VAD → WebSocket → plays returned audio to `live-interpreter-mic-sink`. Does **not** use the lib; re-declares its own `Direction`/`StreamStart`/`StreamEvent`.

**Things that bite:**

- **GPU gate.** Server mode is refused when free VRAM `< LI_MIN_SERVER_VRAM_MB` (default 8000). VRAM telemetry is NVML (`src/vram.rs`, `nvml-wrapper`), not `nvidia-smi`; the threshold check is `GpuPreflight` in `desktop.rs`. **Multi-GPU:** `nvml_snapshot` selects the device with the most *free* VRAM (or `LI_GPU_INDEX` to force one), so a second card with headroom passes the gate even when the primary is full. The runtime then pins the server to that card via `CUDA_VISIBLE_DEVICES` (+ `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so the CUDA index matches NVML's). `VramSnapshot.device_index` carries the choice.
- **Voice profile injection.** Saving a voice writes `data/voice/reference.{wav,txt}`; on next server start these are passed as `LI_VOICE_REF`/`LI_VOICE_REF_TEXT`, and `tts.rs` base64-encodes the WAV into each TTS request. Requires a server restart to take effect.
- **Process lifecycle** is PID files under `data/logs/*.pid` plus `pgrep` patterns; "healthy" = HTTP `/health` poll. Independent signals.
- **Spanish user-facing strings live in the domain/UI layer** (voice-profile messages, control-panel labels). Intentional — keep new user-facing status text in Spanish to match.
- **Translation backends behind `Translator` (enum, not `dyn` — keeps `AppState: Clone`).** `src/translate/` = `Translator::Http(HttpTranslator)` over `http.rs` (Ollama; `keep_alive` pins the model in VRAM) **or** `Translator::Candle(CandleTranslator)` over `candle.rs` (in-process quantized Qwen2 GGUF — no external Ollama process, frees its VRAM). `from_env` auto-selects Candle when built with `candle-translate` **and** the GGUF model is present (else Ollama); `LI_TRANSLATE_BACKEND` overrides (`resolve_backend`, pure+tested). The Candle path needs `--features candle-translate` (heavy compile; GPU via the `cuda` feature, weak-dep activated). Measured (GPU, es→en): Candle ~0.1s vs Ollama ~0.2s, quality ≥ Ollama; default model Qwen2.5-1.5B q4 (0.5B via `LI_CANDLE_TRANSLATE_GGUF` for low VRAM). Both share the `prompt_for` + `TranslationBuffer` context and strip `<think>…</think>` via `strip_think`. Model/tokenizer for Candle: `scripts/download-translate-model.sh` → `data/models/translate/` (`LI_CANDLE_TRANSLATE_GGUF`/`_TOKENIZER`). First-token latency bench: `benches/translate_latency.rs`.
- The repo was renamed from "OVT" → "Live Interpreter"; do **not** add compatibility shims for old names (pre-production).

## Configuration

Server reads `LI_*` env in `src/config.rs::from_env`; the runtime/client read additional `LI_*`. Key ones: `LI_BIND` (server addr), `LI_WHISPER_MODEL`, `LI_OLLAMA_URL`/`LI_OLLAMA_MODEL` (default `translator:latest`), `LI_TRANSLATE_BACKEND` (`ollama`/`candle`) + `LI_CANDLE_TRANSLATE_GGUF`/`_TOKENIZER`/`_MAX_TOKENS`/`LI_CANDLE_DEVICE`, `LI_QWEN_TTS_URL`/`_MODEL`/`_VOICE`, `LI_MIN_SERVER_VRAM_MB`, `LI_GPU_INDEX` (force a GPU; default = most-free), `LI_CONTROL_BIND` (control panel, default `127.0.0.1:8799`), `LI_ROLE` (mesh: `provider`/`consumer`/`bench`), `LI_MESH_TOKEN`, `LI_AUTH_TOKEN` (optional bearer/`?token=` for LAN). Client VAD tuning (`LI_CLIENT_VAD_THRESHOLD`, `LI_CLIENT_SILENCE_MS`, …) is in `docs/meeting-client.md`. Full list in README "Configuration" and `src/config.rs`.

## Docs

`docs/meeting-client.md` (client runtime + VAD tuning), `docs/qwen3-tts-contract.md` (note: the actual request `tts.rs` sends has more fields than that doc — trust `QwenTtsRequest`), `docs/rust-native-inference.md`, `docs/architecture-refactor-fsm.md` (architecture + roadmap).
