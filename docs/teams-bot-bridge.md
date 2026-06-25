# Teams Bot bridge

This is not the primary path for live personal interpretation.

Use this only when the requirement becomes a corporate Teams app/bot that joins meetings through Microsoft Graph and appears as a participant.

## Why it is secondary

For the current goal, the fastest path is:

```text
local microphone -> local Rust pipeline -> PipeWire virtual microphone -> Teams
```

The Teams Bot Framework path adds:

- public HTTPS endpoint,
- Azure app registration,
- Bot Framework auth,
- Graph calling permissions,
- meeting admission,
- real-time media bot infrastructure,
- SRTP/RTP media handling.

That is useful for organization-wide deployment, but it is not the lowest-latency personal translator.

## Future crate split

If this project grows into a corporate Teams app, split the code into:

```text
crates/ia-pipeline
  ASR, translation, TTS, transcript persistence

crates/audio-router
  PipeWire virtual microphone and local capture

crates/teams-bridge
  Bot Framework Activity structs, auth, Graph callbacks, adaptive cards
```

The current single-crate structure keeps the personal real-time route simple while leaving this split straightforward.
