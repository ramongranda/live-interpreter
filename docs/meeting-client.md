# OVT Meeting Client

Cliente Rust para usar el servidor GPU desde otro ordenador.

## Servidor GPU

En la maquina con NVIDIA:

```bash
cd /home/rgranda/workspaces/olares-voice-translator
OVT_BIND=0.0.0.0:8787 ./scripts/start-local-stack.sh
hostname -I
```

## Cliente donde corre Teams

Compilar/copiar el binario y arrancar:

```bash
cd /home/rgranda/workspaces/olares-voice-translator
cargo build --release --bin ovt-meeting-client
OVT_SERVER_URL=http://IP_DEL_SERVIDOR:8787 ./scripts/start-meeting-client.sh
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

## Salida hacia Teams

En Linux, crear antes el micro virtual:

```bash
./scripts/create-virtual-teams-mic.sh
```

En Teams seleccionar como microfono:

```text
ovt-teams-mic-source
```

El cliente reproduce por defecto hacia:

```text
ovt-teams-mic-sink
```

Variables utiles:

```bash
OVT_CLIENT_BIND=127.0.0.1:8790
OVT_CLIENT_DIRECTION=es_to_en
OVT_CLIENT_CHUNK_MS=2500
OVT_CLIENT_PLAY_CMD=pw-play
OVT_CLIENT_PLAY_TARGET=ovt-teams-mic-sink
```

En Windows/macOS el flujo es el mismo, pero la salida debe apuntarse a un cable
virtual local como VB-Cable o BlackHole cuando tengamos el empaquetado de esos
clientes.
