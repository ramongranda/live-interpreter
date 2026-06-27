# Qwen3-TTS HTTP contract

The Rust server expects a local Qwen3-TTS service at `LI_QWEN_TTS_URL`.

Request:

```http
POST /v1/audio/speech
content-type: application/json
```

```json
{
  "text": "Translated English text",
  "language": "en",
  "voice_ref": "/absolute/path/to/your_voice.wav",
  "format": "wav",
  "stream": false
}
```

Response:

```text
audio/wav bytes
```

The service can be backed by Qwen3-TTS directly, Voicebox remote inference, or any compatible wrapper, but the primary runtime contract stays this simple endpoint.
