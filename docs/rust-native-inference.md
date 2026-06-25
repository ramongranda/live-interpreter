# Rust-native inference roadmap

The current working stack is intentionally pragmatic:

```text
Rust axum server
  -> whisper-rs / whisper.cpp for ASR
  -> Ollama translator model for text translation
  -> qwen3_tts_rs API server for TTS
  -> PipeWire virtual microphone for Teams
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

## Next Experiment

First target for Rust-native replacement:

```text
Ollama translator:latest -> Rust-native translation backend
```

Acceptance criteria:

- same `/v1/interpret/text` contract;
- no regression in Spanish-English translation quality;
- lower median latency than current Ollama path;
- clean `cargo test`;
- smoke test through `/v1/interpret/text` and `/v1/interpret/file`.

TTS should stay on `qwen3_tts_rs` for now because it is already Rust and OpenAI-compatible. The immediate TTS improvement is not a rewrite; it is fixing CUDA runtime availability so the installed CUDA binary can start instead of falling back to CPU.
