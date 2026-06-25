#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="${OVT_QWEN_INSTALL_DIR:-${ROOT}/vendor/qwen3_tts_rs}"
MODEL="${OVT_QWEN_MODEL:-Qwen3-TTS-12Hz-0.6B-CustomVoice}"
HOST="${OVT_QWEN_HOST:-127.0.0.1}"
PORT="${OVT_QWEN_PORT:-8020}"
DEVICE="${OVT_QWEN_DEVICE:-cuda:0}"
API_SERVER="${OVT_QWEN_API_SERVER:-${INSTALL_DIR}/api_server_gpu_torch212}"

source "${ROOT}/scripts/cuda-env.sh"

exec "${API_SERVER}" "${INSTALL_DIR}/models/${MODEL}" --device "${DEVICE}" --host "${HOST}" --port "${PORT}"
