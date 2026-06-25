# Teams real-time objective

Primary target:

```text
Spanish physical mic
  -> local ASR
  -> local translation
  -> Qwen3-TTS cloned voice in English
  -> PipeWire virtual microphone
  -> Microsoft Teams
```

Teams must be configured to use `ovt-teams-mic-source` as microphone. The physical microphone must not be selected in Teams.

The reverse private path is:

```text
Teams remote English audio
  -> PipeWire capture source
  -> local ASR
  -> local translation
  -> private Spanish captions
```

## Start

Terminal 1:

```bash
cd /home/rgranda/workspaces/olares-voice-translator
./scripts/download-whisper-model.sh large-v3-turbo
./scripts/install-qwen3-tts-rs.sh
./scripts/start-local-stack.sh
```

Manual equivalent:

```bash
cd /home/rgranda/workspaces/olares-voice-translator
OVT_WHISPER_MODEL=data/models/ggml-large-v3-turbo.bin \
OVT_OLLAMA_MODEL=translator:latest \
OVT_QWEN_TTS_URL=http://127.0.0.1:8020 \
cargo run --release
```

Terminal 2:

```bash
cd /home/rgranda/workspaces/olares-voice-translator
bash scripts/create-virtual-teams-mic.sh
```

Terminal 3:

```bash
cd /home/rgranda/workspaces/olares-voice-translator
bash scripts/list-audio.sh
OVT_INPUT_SOURCE=@DEFAULT_AUDIO_SOURCE@ \
OVT_VIRTUAL_SINK=ovt-teams-mic-sink \
bash scripts/run-realtime-es-to-en.sh
```

## Teams setup

1. Open Teams device settings.
2. Select microphone `ovt-teams-mic-source`.
3. Do not select the physical microphone in Teams.
4. Keep the local translator terminal running.

## Notes

- This is the low-latency personal route. It does not use Azure Bot Service or Graph media bots.
- The official Teams media-bot route is for corporate integration and requires Graph permissions, an application-hosted media bot, public HTTPS, and meeting admission.
- A future `teams-bridge` crate can expose Bot Framework activities, but it is intentionally not on the critical path for personal live interpretation.
- `OVT_CHUNK_SECONDS=2` lowers latency but can reduce ASR quality.
- `OVT_CHUNK_SECONDS=4` improves quality but feels less simultaneous.
