#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LI_LOG_DIR:-${ROOT}/data/logs}"

for name in qwen3-tts live-interpreter live-interpreter-mic; do
  pid_file="${LOG_DIR}/${name}.pid"
  if [ -f "${pid_file}" ]; then
    pid="$(cat "${pid_file}")"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "Stopping ${name} (${pid})"
      kill "${pid}" || true
    fi
    rm -f "${pid_file}"
  fi
done
