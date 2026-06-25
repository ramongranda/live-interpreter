#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_URL="${OVT_SERVER_URL:-http://127.0.0.1:8787}"
CLIENT_BIND="${OVT_CLIENT_BIND:-127.0.0.1:8790}"
PLAY_TARGET="${OVT_CLIENT_PLAY_TARGET:-ovt-teams-mic-sink}"

cd "${ROOT}"

OVT_SERVER_URL="${SERVER_URL}" \
OVT_CLIENT_BIND="${CLIENT_BIND}" \
OVT_CLIENT_PLAY_TARGET="${PLAY_TARGET}" \
  "${ROOT}/target/release/ovt-meeting-client"
