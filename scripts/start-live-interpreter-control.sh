#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_BIND="${LI_APP_BIND:-127.0.0.1:8798}"

cd "${ROOT}"

LI_APP_BIND="${APP_BIND}" "${ROOT}/target/release/live-interpreter-control"
