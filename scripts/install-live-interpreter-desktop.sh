#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_BIND="${LI_APP_BIND:-127.0.0.1:8798}"
APP_URL="${LI_APP_URL:-http://127.0.0.1:8798}"
CONFIG_DIR="${HOME}/.config/live-interpreter"
SYSTEMD_DIR="${HOME}/.config/systemd/user"
DESKTOP_DIR="${HOME}/.local/share/applications"
ENV_FILE="${CONFIG_DIR}/live-interpreter-control.env"
SERVICE_FILE="${SYSTEMD_DIR}/live-interpreter-control.service"
DESKTOP_FILE="${DESKTOP_DIR}/live-interpreter-control.desktop"

if [ ! -x "${ROOT}/target/release/live-interpreter-control" ]; then
  echo "Building live-interpreter-control release binary"
  cargo build --release --bin live-interpreter-control
fi

if [ ! -x "${ROOT}/target/release/live-interpreter-desktop" ]; then
  echo "Building live-interpreter-desktop release binary"
  cargo build --release --features desktop-native --bin live-interpreter-desktop
fi

mkdir -p "${CONFIG_DIR}" "${SYSTEMD_DIR}" "${DESKTOP_DIR}"

if [ ! -f "${ENV_FILE}" ]; then
  cat >"${ENV_FILE}" <<EOF
LI_APP_BIND=${APP_BIND}
LI_APP_URL=${APP_URL}
# Optional LAN token. Use the same value as LI_CLIENT_AUTH_TOKEN on clients.
# LI_AUTH_TOKEN=
EOF
fi

cat >"${SERVICE_FILE}" <<EOF
[Unit]
Description=Live Interpreter control app
After=network-online.target

[Service]
Type=simple
WorkingDirectory=${ROOT}
EnvironmentFile=-${ENV_FILE}
ExecStart=${ROOT}/target/release/live-interpreter-control
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
EOF

cat >"${DESKTOP_FILE}" <<EOF
[Desktop Entry]
Type=Application
Name=Live Interpreter
Comment=Local voice translator for calls and meetings
Exec=${ROOT}/scripts/open-live-interpreter-desktop.sh
Terminal=false
Categories=AudioVideo;Network;
StartupNotify=true
EOF

chmod +x "${ROOT}/scripts/open-live-interpreter-control.sh" "${ROOT}/scripts/open-live-interpreter-desktop.sh" "${DESKTOP_FILE}"
SYSTEMD_OK=0
if command -v systemctl >/dev/null 2>&1; then
  if timeout 10s systemctl --user daemon-reload && \
     timeout 10s systemctl --user enable --now live-interpreter-control.service; then
    SYSTEMD_OK=1
  else
    echo "systemd user service was installed but did not respond in time."
    echo "The desktop launcher will use direct process startup as fallback."
  fi
fi

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "${DESKTOP_DIR}" >/dev/null 2>&1 || true
fi

echo "Installed Live Interpreter desktop launcher:"
echo "  ${DESKTOP_FILE}"
if [ "${SYSTEMD_OK}" = "1" ]; then
  echo "Service:"
  echo "  systemctl --user status live-interpreter-control"
else
  echo "Service file:"
  echo "  ${SERVICE_FILE}"
fi
echo "Open:"
echo "  ${ROOT}/scripts/open-live-interpreter-desktop.sh"
