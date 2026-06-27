#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-large-v3-turbo}"
OUT_DIR="${LI_MODEL_DIR:-data/models}"
mkdir -p "${OUT_DIR}"

case "${MODEL}" in
  tiny|base|small|medium|large-v3|large-v3-turbo)
    ;;
  *)
    echo "Usage: $0 [tiny|base|small|medium|large-v3|large-v3-turbo]" >&2
    exit 2
    ;;
esac

file="ggml-${MODEL}.bin"
url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/${file}"
out="${OUT_DIR}/${file}"

if [ -s "${out}" ]; then
  echo "Model already exists: ${out}"
  exit 0
fi

echo "Downloading ${url}"
curl -L --fail --progress-bar "${url}" -o "${out}.part"
mv "${out}.part" "${out}"
echo "Model ready: ${out}"
