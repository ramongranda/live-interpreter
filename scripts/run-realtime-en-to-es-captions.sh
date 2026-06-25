#!/usr/bin/env bash
set -euo pipefail

SERVER="${OVT_SERVER:-http://127.0.0.1:8787}"
SOURCE="${OVT_TEAMS_OUTPUT_SOURCE:-@DEFAULT_AUDIO_SOURCE@}"
CHUNK_SECONDS="${OVT_CHUNK_SECONDS:-3}"
WORK_DIR="${OVT_WORK_DIR:-data/live-captions}"

mkdir -p "${WORK_DIR}"

echo "Teams audio capture source: ${SOURCE}"
echo "Chunk seconds: ${CHUNK_SECONDS}"
echo "Server: ${SERVER}"

while true; do
  id="$(date +%s%3N)"
  wav="${WORK_DIR}/${id}.wav"
  json="${WORK_DIR}/${id}.json"

  timeout "${CHUNK_SECONDS}" \
    pw-record \
      --target "${SOURCE}" \
      --rate 16000 \
      --channels 1 \
      --format s16 \
      "${wav}" >/dev/null 2>&1 || true

  if [ ! -s "${wav}" ]; then
    rm -f "${wav}"
    continue
  fi

  curl -fsS \
    -F "direction=en_to_es" \
    -F "synthesize=false" \
    -F "audio=@${wav}" \
    "${SERVER}/v1/interpret/file" > "${json}" || {
      echo "caption translation failed for ${wav}" >&2
      continue
    }

  jq -r '"ES: " + (.translation // "")' "${json}"
done
