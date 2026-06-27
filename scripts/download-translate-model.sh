#!/usr/bin/env bash
# Download the quantized Qwen2 translation model + tokenizer for the candle backend.
# Targets the defaults in src/translate/mod.rs (LI_CANDLE_GGUF / LI_CANDLE_TOKENIZER).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/data/models"
mkdir -p "${DEST}"

GGUF_FILE="${LI_CANDLE_GGUF_FILE:-qwen2.5-0.5b-instruct-q4_k_m.gguf}"
GGUF_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/${GGUF_FILE}"
TOKENIZER_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json"

echo "Downloading ${GGUF_FILE}"
curl -L --fail --progress-bar -o "${DEST}/${GGUF_FILE}" "${GGUF_URL}"

echo "Downloading tokenizer.json"
curl -L --fail --progress-bar -o "${DEST}/qwen2-tokenizer.json" "${TOKENIZER_URL}"

echo
echo "Saved to ${DEST}:"
echo "  ${DEST}/${GGUF_FILE}"
echo "  ${DEST}/qwen2-tokenizer.json"
echo
echo "Run the server with the candle backend:"
echo "  LI_TRANSLATE_BACKEND=candle cargo run --release --features candle-cuda --bin live-interpreter"
