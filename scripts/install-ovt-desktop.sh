#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_BIND="${OVT_APP_BIND:-127.0.0.1:8798}"
APP_URL="${OVT_APP_URL:-http://127.0.0.1:8798}"
CONFIG_DIR="${HOME}/.config/ovt"
SYSTEMD_DIR="${HOME}/.config/systemd/user"
DESKTOP_DIR="${HOME}/.local/share/applications"
ENV_FILE="${CONFIG_DIR}/ovt-app.env"
SERVICE_FILE="${SYSTEMD_DIR}/ovt-app.service"
DESKTOP_FILE="${DESKTOP_DIR}/ovt-app.desktop"

if [ ! -x "${ROOT}/target/release/ovt-app" ]; then
  echo "Building ovt-app release binary"
  cargo build --release --bin ovt-app
fi

mkdir -p "${CONFIG_DIR}" "${SYSTEMD_DIR}" "${DESKTOP_DIR}"

if [ ! -f "${ENV_FILE}" ]; then
  cat >"${ENV_FILE}" <<EOF
OVT_APP_BIND=${APP_BIND}
OVT_APP_URL=${APP_URL}
# Optional LAN token. Use the same value as OVT_CLIENT_AUTH_TOKEN on clients.
# OVT_AUTH_TOKEN=
EOF
fi

cat >"${SERVICE_FILE}" <<EOF
[Unit]
Description=Olares Voice Translator control app
After=network-online.target

[Service]
Type=simple
WorkingDirectory=${ROOT}
EnvironmentFile=-${ENV_FILE}
ExecStart=${ROOT}/target/release/ovt-app
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
EOF

cat >"${DESKTOP_FILE}" <<EOF
[Desktop Entry]
Type=Application
Name=Olares Voice Translator
Comment=Local Teams voice translator
Exec=${ROOT}/scripts/open-ovt-app.sh
Terminal=false
Categories=AudioVideo;Network;
StartupNotify=true
EOF

chmod +x "${ROOT}/scripts/open-ovt-app.sh" "${DESKTOP_FILE}"
SYSTEMD_OK=0
if command -v systemctl >/dev/null 2>&1; then
  if timeout 10s systemctl --user daemon-reload && \
     timeout 10s systemctl --user enable --now ovt-app.service; then
    SYSTEMD_OK=1
  else
    echo "systemd user service was installed but did not respond in time."
    echo "The desktop launcher will use direct process startup as fallback."
  fi
fi

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "${DESKTOP_DIR}" >/dev/null 2>&1 || true
fi

echo "Installed OVT desktop launcher:"
echo "  ${DESKTOP_FILE}"
if [ "${SYSTEMD_OK}" = "1" ]; then
  echo "Service:"
  echo "  systemctl --user status ovt-app"
else
  echo "Service file:"
  echo "  ${SERVICE_FILE}"
fi
echo "Open:"
echo "  ${ROOT}/scripts/open-ovt-app.sh"
