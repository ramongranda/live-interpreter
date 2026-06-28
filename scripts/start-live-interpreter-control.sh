#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTROL_BIND="${LI_CONTROL_BIND:-127.0.0.1:8799}"

cd "${ROOT}"

LI_CONTROL_BIND="${CONTROL_BIND}" "${ROOT}/target/release/li-control"
