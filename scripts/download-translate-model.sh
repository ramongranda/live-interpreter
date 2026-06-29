#!/usr/bin/env bash
set -euo pipefail

# Fetch the GGUF model + tokenizer for the in-process Candle translation backend
# (LI_TRANSLATE_BACKEND=candle). Default: Qwen2.5-1.5B-Instruct q4_k_m.
# Override the model with env vars before running.

# Default 3B (quality sweet spot; coexists with Qwen-TTS+Whisper on 16GB once
# Candle replaces Ollama). Set the env vars below for 1.5B/0.5B on tighter VRAM.
OUT_DIR="${LI_TRANSLATE_DIR:-data/models/translate}"
GGUF_REPO="${LI_TRANSLATE_GGUF_REPO:-Qwen/Qwen2.5-3B-Instruct-GGUF}"
GGUF_FILE="${LI_TRANSLATE_GGUF_FILE:-qwen2.5-3b-instruct-q4_k_m.gguf}"
TOKENIZER_REPO="${LI_TRANSLATE_TOKENIZER_REPO:-Qwen/Qwen2.5-3B-Instruct}"

mkdir -p "${OUT_DIR}"

fetch() {
  local url="$1" out="$2"
  if [ -s "${out}" ]; then
    echo "Already present: ${out}"
    return 0
  fi
  echo "Downloading ${url}"
  curl -L --fail --progress-bar "${url}" -o "${out}.part"
  mv "${out}.part" "${out}"
  echo "Ready: ${out}"
}

fetch "https://huggingface.co/${GGUF_REPO}/resolve/main/${GGUF_FILE}" "${OUT_DIR}/${GGUF_FILE}"
fetch "https://huggingface.co/${TOKENIZER_REPO}/resolve/main/tokenizer.json" "${OUT_DIR}/tokenizer.json"

cat <<EOF

Translate model ready in ${OUT_DIR}.
Run the server with the Candle backend:

  cargo build --release --features candle-translate            # CPU
  cargo build --release --features candle-translate,cuda       # GPU (needs nvcc)
  LI_TRANSLATE_BACKEND=candle \\
  LI_CANDLE_TRANSLATE_GGUF=${OUT_DIR}/${GGUF_FILE} \\
  LI_CANDLE_TRANSLATE_TOKENIZER=${OUT_DIR}/tokenizer.json \\
    ./target/release/live-interpreter
EOF
