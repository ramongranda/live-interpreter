#!/usr/bin/env bash
set -euo pipefail

NAME="${OVT_VIRTUAL_NAME:-ovt-teams-mic}"
LATENCY="${OVT_VIRTUAL_LATENCY:-30ms}"

if pgrep -af "pw-loopback.*${NAME}" >/dev/null; then
  echo "Virtual Teams mic already running: ${NAME}"
  exit 0
fi

echo "Creating virtual Teams microphone with PipeWire: ${NAME}"
echo "In Teams, select microphone/source: ${NAME}-source"
echo "Send generated audio to sink: ${NAME}-sink"

exec pw-loopback \
  --name "${NAME}" \
  --latency "${LATENCY}" \
  --capture-props="media.class=Audio/Sink node.name=${NAME}-sink node.description=${NAME}-sink audio.position=[FL]" \
  --playback-props="media.class=Audio/Source node.name=${NAME}-source node.description=${NAME}-source audio.position=[FL]" \
  --channels 1 \
  -m "[ FL ]"
