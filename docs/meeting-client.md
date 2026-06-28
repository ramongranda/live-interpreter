# Live Interpreter Client

Cliente Rust para usar el servidor GPU desde otro ordenador. El binario es
`live-interpreter-client`: captura el micro local, lo envia al servidor por
WebSocket (`/v1/stream/meeting`) y reproduce la voz traducida hacia el micro
virtual.

## Servidor GPU

En la maquina con NVIDIA:

```bash
cd /home/rgranda/workspaces/live-interpreter
LI_BIND=0.0.0.0:8787 \
LI_WHISPER_MODEL=data/models/ggml-large-v3-turbo.bin \
  ./target/release/live-interpreter
hostname -I
```

## Cliente donde corre la app de llamadas

Compilar/copiar el binario y arrancar:

```bash
cd /home/rgranda/workspaces/live-interpreter
cargo build --release --bin live-interpreter-client
LI_SERVER_URL=http://IP_DEL_SERVIDOR:8787 ./scripts/start-meeting-client.sh
```

Abrir:

```text
http://127.0.0.1:8790
```

Controles disponibles:

- Pausa.
- Mutear entrada.
- Mutear salida.
- Cambiar direccion `es_to_en` / `en_to_es`.

## Salida hacia la app de llamadas

El micro virtual `live-interpreter-mic-source` lo crea el runtime nativo (los
binarios compilados con la feature `native-audio`). El cliente reproduce por
`pw-play` hacia `LI_CLIENT_PLAY_TARGET` (por defecto `live-interpreter-mic-sink`).

En tu app de llamada, reunion o streaming, seleccionar como microfono:

```text
live-interpreter-mic-source
```

El cliente reproduce por defecto hacia:

```text
live-interpreter-mic-sink
```

Variables utiles:

```bash
LI_CLIENT_BIND=127.0.0.1:8790
LI_CLIENT_DIRECTION=es_to_en
LI_CLIENT_VAD_THRESHOLD=0.012
LI_CLIENT_SILENCE_MS=800
LI_CLIENT_MIN_VOICE_MS=280
LI_CLIENT_MAX_UTTERANCE_MS=8500
LI_CLIENT_PRE_ROLL_MS=240
LI_CLIENT_PLAY_CMD=pw-play
LI_CLIENT_PLAY_TARGET=live-interpreter-mic-sink
LI_CLIENT_AUTH_TOKEN=opcional
```

En Windows/macOS el flujo es el mismo, pero la salida debe apuntarse a un cable
virtual local como VB-Cable o BlackHole cuando tengamos el empaquetado de esos
clientes.
