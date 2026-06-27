#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="${LI_QWEN_INSTALL_DIR:-${ROOT}/vendor/qwen3_tts_rs}"
MODEL="${LI_QWEN_MODEL:-Qwen3-TTS-12Hz-0.6B-CustomVoice}"
HOST="${LI_QWEN_HOST:-127.0.0.1}"
PORT="${LI_QWEN_PORT:-8020}"
DEVICE="${LI_QWEN_DEVICE:-cuda:0}"
API_SERVER="${LI_QWEN_API_SERVER:-${INSTALL_DIR}/api_server_gpu_torch212}"

source "${ROOT}/scripts/cuda-env.sh"

exec "${API_SERVER}" "${INSTALL_DIR}/models/${MODEL}" --device "${DEVICE}" --host "${HOST}" --port "${PORT}"
