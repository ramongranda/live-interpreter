# OVT App

`ovt-app` es la entrada principal. Funciona como una app tipo AudioRelay:

- modo **Servidor GPU**: arranca o para Whisper, Qwen TTS y el micro virtual;
- modo **Cliente Teams**: arranca o para el cliente de reunion;
- muestra salud, procesos de GPU y VRAM;
- permite liberar la GPU al parar el servidor.

## Arranque

```bash
cd /home/rgranda/workspaces/olares-voice-translator
./scripts/build-gpu.sh
./scripts/start-ovt-app.sh
```

Abrir:

```text
http://127.0.0.1:8798
```

## Uso en la maquina GPU

Pulsa **Arrancar servidor**.

La app ejecuta:

```bash
OVT_BIND=0.0.0.0:8787 ./scripts/start-local-stack.sh
```

Pulsa **Parar servidor** para liberar VRAM.

## Uso en el ordenador con Teams

Arranca la misma app en el ordenador cliente y configura:

```bash
OVT_SERVER_URL=http://IP_DEL_SERVIDOR:8787 ./scripts/start-ovt-app.sh
```

Pulsa **Arrancar cliente** y abre **Abrir controles**.

## Seguridad LAN

Para exigir token:

```bash
OVT_AUTH_TOKEN=pon-un-token-largo ./scripts/start-ovt-app.sh
```

En el cliente usa el mismo token:

```bash
OVT_SERVER_URL=http://IP_DEL_SERVIDOR:8787 \
OVT_CLIENT_AUTH_TOKEN=pon-un-token-largo \
./scripts/start-ovt-app.sh
```

Si `OVT_AUTH_TOKEN` no esta definido, el servidor acepta peticiones sin token.
