#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_URL="${LI_SERVER_URL:-http://127.0.0.1:8787}"
CLIENT_BIND="${LI_CLIENT_BIND:-127.0.0.1:8790}"
PLAY_TARGET="${LI_CLIENT_PLAY_TARGET:-live-interpreter-mic-sink}"

cd "${ROOT}"

LI_SERVER_URL="${SERVER_URL}" \
LI_CLIENT_BIND="${CLIENT_BIND}" \
LI_CLIENT_PLAY_TARGET="${PLAY_TARGET}" \
  "${ROOT}/target/release/live-interpreter-client"
