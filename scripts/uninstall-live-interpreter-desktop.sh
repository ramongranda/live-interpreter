#!/usr/bin/env bash
set -euo pipefail

SYSTEMD_DIR="${HOME}/.config/systemd/user"
DESKTOP_DIR="${HOME}/.local/share/applications"

if command -v systemctl >/dev/null 2>&1; then
  timeout 10s systemctl --user disable --now live-interpreter-control.service >/dev/null 2>&1 || true
fi
rm -f "${SYSTEMD_DIR}/live-interpreter-control.service"
rm -f "${DESKTOP_DIR}/live-interpreter-control.desktop"
if command -v systemctl >/dev/null 2>&1; then
  timeout 10s systemctl --user daemon-reload >/dev/null 2>&1 || true
fi

echo "Removed Live Interpreter desktop launcher and user service."
