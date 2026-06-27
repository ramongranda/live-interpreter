#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="${LI_QWEN_INSTALL_DIR:-${ROOT}/vendor/qwen3_tts_rs}"
ASSET_NAME="${LI_QWEN_ASSET:-qwen3-tts-linux-x86_64-cuda}"
MODEL_SIZE="${LI_QWEN_MODEL_SIZE:-0.6B}"

case "${MODEL_SIZE}" in
  0.6B)
    CUSTOM_VOICE_MODEL="Qwen3-TTS-12Hz-0.6B-CustomVoice"
    ;;
  1.7B)
    CUSTOM_VOICE_MODEL="Qwen3-TTS-12Hz-1.7B-CustomVoice"
    ;;
  *)
    echo "LI_QWEN_MODEL_SIZE must be 0.6B or 1.7B" >&2
    exit 2
    ;;
esac

BASE_MODEL="Qwen3-TTS-12Hz-0.6B-Base"
REPO="second-state/qwen3_tts_rs"

need() {
  command -v "$1" >/dev/null || {
    echo "Missing required command: $1" >&2
    exit 2
  }
}

download_model() {
  local model="$1"
  local model_dir="${INSTALL_DIR}/models/${model}"

  if [ -d "${model_dir}" ] && [ -n "$(find "${model_dir}" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]; then
    echo "Model already present: ${model}"
  else
    mkdir -p "${model_dir}"
    echo "Downloading model: ${model}"
    local api_url="https://huggingface.co/api/models/Qwen/${model}"
    local hf_url="https://huggingface.co/Qwen/${model}/resolve/main"
    local files
    files="$(curl -fsSL "${api_url}" | jq -r '.siblings[].rfilename')"

    while IFS= read -r file; do
      case "${file}" in
        ""|.gitattributes|README.md) continue ;;
      esac
      echo "  ${file}"
      mkdir -p "${model_dir}/$(dirname "${file}")"
      curl -fL --retry 3 --retry-delay 2 -o "${model_dir}/${file}" "${hf_url}/${file}"
    done <<< "${files}"
  fi

  if [ -f "${INSTALL_DIR}/tokenizers/${model}/tokenizer.json" ]; then
    cp "${INSTALL_DIR}/tokenizers/${model}/tokenizer.json" "${model_dir}/tokenizer.json"
  fi
}

need curl
need jq
need unzip

mkdir -p "${INSTALL_DIR}"

if [ ! -x "${INSTALL_DIR}/api_server" ]; then
  tmp="$(mktemp -d)"
  trap 'rm -rf "${tmp}"' EXIT
  zip_name="${ASSET_NAME}.zip"
  url="https://github.com/${REPO}/releases/latest/download/${zip_name}"
  echo "Downloading ${url}"
  curl -fL --retry 3 --retry-delay 2 -o "${tmp}/${zip_name}" "${url}"
  unzip -q "${tmp}/${zip_name}" -d "${tmp}"
  cp -r "${tmp}/${ASSET_NAME}/"* "${INSTALL_DIR}/"
  chmod +x "${INSTALL_DIR}/tts" "${INSTALL_DIR}/voice_clone" "${INSTALL_DIR}/api_server"
fi

download_model "${CUSTOM_VOICE_MODEL}"
download_model "${BASE_MODEL}"

mkdir -p "${INSTALL_DIR}/reference_audio"
for file in trump.wav trump.txt elon_musk.wav elon_musk.txt; do
  if [ ! -f "${INSTALL_DIR}/reference_audio/${file}" ]; then
    curl -fsSL \
      "https://raw.githubusercontent.com/${REPO}/main/reference_audio/${file}" \
      -o "${INSTALL_DIR}/reference_audio/${file}"
  fi
done

cat <<EOF
Qwen3-TTS ready:
  ${INSTALL_DIR}

Start API:
  ${INSTALL_DIR}/api_server ${INSTALL_DIR}/models/${CUSTOM_VOICE_MODEL} --host 127.0.0.1 --port 8020
EOF
