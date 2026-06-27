# Live Interpreter Native Desktop

`live-interpreter-desktop` is the native Tauri shell for Live Interpreter.

Architecture boundary:

- `src/desktop.rs`: domain/application services, process orchestration, GPU preflight, status projection, tests.
- `src/bin/live-interpreter-control.rs`: HTTP browser control panel adapter.
- `src/bin/live-interpreter-desktop.rs`: native Tauri adapter with IPC commands.
- `desktop/index.html`: local WebView UI assets.

Server mode is guarded by `LI_MIN_SERVER_VRAM_MB` and cannot be started through either adapter when the GPU preflight fails.

Run:

```bash
cargo run --features desktop-native --bin live-interpreter-desktop
```

Test first:

```bash
cargo test --lib
cargo build --features desktop-native --bin live-interpreter-desktop
```
