#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT}/target/release/live-interpreter-desktop"

cd "${ROOT}"

if [ ! -x "${BIN}" ]; then
  cargo build --release --features desktop-native --bin live-interpreter-desktop
fi

exec "${BIN}"
