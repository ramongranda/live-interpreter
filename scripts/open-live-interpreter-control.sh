#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_URL="${LI_APP_URL:-http://127.0.0.1:8798}"
APP_BIND="${LI_APP_BIND:-127.0.0.1:8798}"

cd "${ROOT}"

if command -v systemctl >/dev/null 2>&1 && timeout 3s systemctl --user list-unit-files live-interpreter-control.service >/dev/null 2>&1; then
  timeout 5s systemctl --user start live-interpreter-control.service >/dev/null 2>&1 || true
fi

if ! curl -fsS "${APP_URL}/api/status" >/dev/null 2>&1; then
  mkdir -p "${ROOT}/data/logs"
  setsid env LI_APP_BIND="${APP_BIND}" "${ROOT}/target/release/live-interpreter-control" \
    >"${ROOT}/data/logs/live-interpreter-control.log" 2>&1 &
  echo "$!" >"${ROOT}/data/logs/live-interpreter-control.pid"
else
  rm -f "${ROOT}/data/logs/live-interpreter-control.pid"
fi

for _ in $(seq 1 30); do
  if curl -fsS "${APP_URL}/api/status" >/dev/null 2>&1; then
    break
  fi
  sleep 0.2
done

if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "${APP_URL}" >/dev/null 2>&1 &
else
  printf '%s\n' "${APP_URL}"
fi
