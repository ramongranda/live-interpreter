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

Acceptance criteria — **all met** (measured live, es→en, Qwen2.5-1.5B-Instruct q4_k_m):

- ✅ same `/v1/interpret/text` contract (drop-in behind the `Translator` enum);
- ✅ no regression in quality — Candle ≥ Ollama on an A/B of 3 utterances; on an idiomatic one
  (*"me la jugué con ese cliente… salió redondo"*) Ollama's `translator:latest` **failed** (left it
  in Spanish), Candle translated it correctly;
- ✅ lower latency on GPU (`--features candle-translate,cuda`): **~0.10–0.13s vs Ollama ~0.19–0.21s
  (~2× faster)**. CPU is **7–11s** — do not run CPU in the hot path. Translate is not the pipeline
  bottleneck (TTS is), so the headline win is freeing Ollama's ~2.3GB VRAM (Candle q4 uses ~1.1GB;
  stop Ollama → net ~1.2GB freed) and removing an external process;
- ✅ clean `cargo test` (default + `--features candle-translate`);
- ✅ smoke test through `/v1/interpret/text` (A/B vs Ollama on both servers).

Verdict: on GPU the Candle backend wins on latency, quality, and VRAM — promote it as the default
once a GPU build is the norm; keep Ollama for CPU-only hosts.

## Promotion + model choice + Marian decision

**Default backend (shipped).** `Translator::from_env` now auto-selects Candle when the binary was
built with `candle-translate` **and** the GGUF model is present (`resolve_backend`, pure + tested);
`LI_TRANSLATE_BACKEND` still overrides. So a GPU build with the model downloaded uses Candle with no
config; everything else falls back to Ollama. No footgun on non-candle builds.

**Model size A/B (es→en, GPU, q4_k_m).** Qwen2.5 **0.5B** vs **1.5B**:

| | 0.5B | 1.5B |
|---|---|---|
| latency | ~0.13s | ~0.10s |
| VRAM | ~0.5GB | ~1.1GB |
| quality | good on plain sentences; idiom *"me la jugué"* → "I played with" (wrong) | slightly better nuance; idiom → "I lost it" (closer) |

Latency is effectively tied (both <150ms, both beat Ollama). **Default stays 1.5B** for the quality
edge; 0.5B is the low-VRAM option via `LI_CANDLE_TRANSLATE_GGUF`.

**Dedicated NMT (Marian/Opus-MT): not pursued — YAGNI, evidence-backed.** A dedicated es↔en NMT would
save ~0.8GB more VRAM, but: (1) translate is already ~0.1s — negligible next to TTS ~2.5s, the real
bottleneck, so there is no latency to win; (2) the instruct LLM already matches/beats Ollama and
handles conversational context (`TranslationBuffer`), where a dedicated NMT tends to be more literal;
(3) es→en is not a Candle preset, needing offline pytorch→safetensors + tokenizer conversion and a
custom config. Cost is real, payoff is a VRAM saving that does not move the pipeline. Revisit only if
a future host is VRAM-starved enough that ~0.8GB matters.

TTS should stay on `qwen3_tts_rs` for now because it is already Rust and OpenAI-compatible. The immediate TTS improvement is not a rewrite; it is fixing CUDA runtime availability so the installed CUDA binary can start instead of falling back to CPU.
