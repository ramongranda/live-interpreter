#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${OVT_LOG_DIR:-${ROOT}/data/logs}"
mkdir -p "${LOG_DIR}"

start_bg() {
  local name="$1"
  shift
  if pgrep -af "$*" >/dev/null; then
    echo "${name} already running"
    return 0
  fi
  echo "Starting ${name}"
  setsid "$@" >"${LOG_DIR}/${name}.log" 2>&1 &
  echo "$!" >"${LOG_DIR}/${name}.pid"
}

cd "${ROOT}"

QWEN_INSTALL_DIR="${OVT_QWEN_INSTALL_DIR:-${ROOT}/vendor/qwen3_tts_rs}"
source "${ROOT}/scripts/cuda-env.sh"
QWEN_API_SERVER="${OVT_QWEN_API_SERVER:-${QWEN_INSTALL_DIR}/api_server_gpu_torch212}"
start_bg qwen3-tts \
  "${QWEN_API_SERVER}" \
  "${QWEN_INSTALL_DIR}/models/Qwen3-TTS-12Hz-0.6B-CustomVoice" \
  --device "${OVT_QWEN_DEVICE:-cuda:0}" \
  --host 127.0.0.1 \
  --port 8020

start_bg olares-voice-translator \
  "${ROOT}/target/release/olares-voice-translator"

start_bg ovt-teams-mic \
  "${ROOT}/scripts/create-virtual-teams-mic.sh"

echo
echo "Logs: ${LOG_DIR}"
echo "Health:"
echo "  curl http://127.0.0.1:8020/health"
echo "  curl http://127.0.0.1:8787/health"
