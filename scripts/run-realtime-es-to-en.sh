#!/usr/bin/env bash
set -euo pipefail

SERVER="${OVT_SERVER:-http://127.0.0.1:8787}"
SOURCE="${OVT_INPUT_SOURCE:-@DEFAULT_AUDIO_SOURCE@}"
VIRTUAL_SINK="${OVT_VIRTUAL_SINK:-ovt-teams-mic-sink}"
CHUNK_SECONDS="${OVT_CHUNK_SECONDS:-3}"
WORK_DIR="${OVT_WORK_DIR:-data/live}"

mkdir -p "${WORK_DIR}"

echo "Input source: ${SOURCE}"
echo "Virtual sink for Teams mic: ${VIRTUAL_SINK}"
echo "Chunk seconds: ${CHUNK_SECONDS}"
echo "Server: ${SERVER}"
echo
echo "Teams must use microphone/source: ovt-teams-mic-source"
echo "Keep your physical microphone muted/disabled in Teams."

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
    -F "direction=es_to_en" \
    -F "synthesize=true" \
    -F "audio=@${wav}" \
    "${SERVER}/v1/interpret/file" > "${json}" || {
      echo "interpretation failed for ${wav}" >&2
      continue
    }

  audio_path="$(jq -r '.audio_path // empty' "${json}")"
  translation="$(jq -r '.translation // empty' "${json}")"
  [ -n "${translation}" ] && echo "EN: ${translation}"

  if [ -n "${audio_path}" ] && [ -f "${audio_path}" ]; then
    pw-play --target "${VIRTUAL_SINK}" "${audio_path}" || true
  else
    echo "No generated audio returned. Check OVT_QWEN_TTS_URL and Qwen3-TTS server." >&2
  fi
done
