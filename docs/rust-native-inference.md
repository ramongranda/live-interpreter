# Rust-native inference roadmap

The current working stack is intentionally pragmatic:

```text
Rust axum server
  -> whisper-rs / whisper.cpp for ASR
  -> Ollama translator model for text translation
  -> qwen3_tts_rs API server for TTS
  -> PipeWire virtual microphone for meeting and voice apps
```

The target architecture is more native:

```text
single Rust control plane
  -> ASR in-process where practical
  -> translation model through Rust-native inference
  -> TTS through Rust-native inference or a local Rust sidecar
  -> PipeWire routing
```

## Candidates

### Candle

Use Candle when we want direct model ownership inside the Rust binary.

Best fit:

- replacing Ollama for the translation model;
- embedding smaller translation models directly;
- controlling memory, batching, and token streaming;
- avoiding Python/vLLM runtime dependencies.

Risk:

- each model family needs working Candle support;
- GPU acceleration must be validated on this host, not assumed.

### LlamaEdge / llama-core

Use this when we want an Ollama-like local server with lower operational weight.

Best fit:

- OpenAI-compatible local LLM endpoint;
- swapping out Ollama without rewriting the Rust app;
- keeping model serving as a separate process.

Risk:

- WasmEdge GPU plugin support and model compatibility need local proof.

### Kalosm

Use this if the project grows beyond live translation into an agent/runtime layer.

Best fit:

- typed structured outputs;
- local RAG and tools;
- Rust-first AI application code.

Risk:

- may be more framework than needed for the live audio path.

## Decision Rule

Do not replace working runtime pieces just for purity. Replace them when one of these is true:

- latency improves measurably;
- memory use drops measurably;
- deployment becomes simpler;
- the external process is unstable;
- the model quality is at least equal.

## Next Experiment — DONE (Candle backend shipped, behind `candle-translate`)

First target for Rust-native replacement:

```text
Ollama translator:latest -> Rust-native translation backend
```

**Implemented** as `Translator::Candle(CandleTranslator)` (`src/translate/candle.rs`):
in-process quantized **Qwen2 GGUF** via Candle (`candle_transformers::models::quantized_qwen2`),
greedy decode, same `prompt_for` + `TranslationBuffer` context + `strip_think` as the HTTP path.
Selected with `LI_TRANSLATE_BACKEND=candle`; built with `--features candle-translate`
(GPU via `cuda`, weak-dep activated — needs nvcc). Model: `scripts/download-translate-model.sh`
(default Qwen2.5-1.5B-Instruct q4_k_m). Decision: a **dedicated NMT** (Marian/Opus-MT) is more
efficient still but es→en needs offline weight/tokenizer conversion + a custom config — deferred;
the quantized LLM reuses the existing prompt and ships now.

Acceptance criteria:

- ✅ same `/v1/interpret/text` contract (drop-in behind the `Translator` enum);
- ⏳ no regression in Spanish-English translation quality — **needs A/B vs Ollama on real utterances**;
- ⏳ lower median latency than Ollama — only true on GPU (`cuda` feature); CPU is slower. Translate
  is not the pipeline bottleneck (TTS is), so the real win is freeing Ollama's ~2GB VRAM + going
  process-less;
- ✅ clean `cargo test` (default + `--features candle-translate`);
- ⏳ smoke test through `/v1/interpret/text` and `/v1/interpret/file`.

TTS should stay on `qwen3_tts_rs` for now because it is already Rust and OpenAI-compatible. The immediate TTS improvement is not a rewrite; it is fixing CUDA runtime availability so the installed CUDA binary can start instead of falling back to CPU.
