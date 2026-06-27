# Live Interpreter Control Panel

`live-interpreter-control` es la entrada principal. Funciona como una app tipo AudioRelay:

- modo **Servidor GPU**: arranca o para Whisper, Qwen TTS y el micro virtual;
- modo **Cliente de llamadas**: arranca o para el cliente de reunion;
- muestra salud, procesos de GPU y VRAM;
- permite liberar la GPU al parar el servidor.

## Arranque

```bash
cd /home/rgranda/workspaces/live-interpreter
./scripts/build-gpu.sh
./scripts/start-live-interpreter-control.sh
```

Abrir:

```text
http://127.0.0.1:8798
```

## Instalacion como app de escritorio

```bash
cd /home/rgranda/workspaces/live-interpreter
./scripts/install-live-interpreter-desktop.sh
```

Esto instala:

- `~/.config/systemd/user/live-interpreter-control.service`
- `~/.local/share/applications/live-interpreter-control.desktop`

Abrir desde terminal:

```bash
./scripts/open-live-interpreter-control.sh
```

Parar solo la consola:

```bash
systemctl --user stop live-interpreter-control
```

Desinstalar:

```bash
./scripts/uninstall-live-interpreter-desktop.sh
```

## Uso en la maquina GPU

Pulsa **Arrancar servidor**.

La app ejecuta:

```bash
LI_BIND=0.0.0.0:8787 ./scripts/start-local-stack.sh
```

Pulsa **Parar servidor** para liberar VRAM.

## Uso en el ordenador con la app de llamadas

Arranca la misma app en el ordenador cliente y configura:

```bash
LI_SERVER_URL=http://IP_DEL_SERVIDOR:8787 ./scripts/start-live-interpreter-control.sh
```

Pulsa **Arrancar cliente** y abre **Abrir controles**.

## Seguridad LAN

Para exigir token:

```bash
LI_AUTH_TOKEN=pon-un-token-largo ./scripts/start-live-interpreter-control.sh
```

En el cliente usa el mismo token:

```bash
LI_SERVER_URL=http://IP_DEL_SERVIDOR:8787 \
LI_CLIENT_AUTH_TOKEN=pon-un-token-largo \
./scripts/start-live-interpreter-control.sh
```

Si `LI_AUTH_TOKEN` no esta definido, el servidor acepta peticiones sin token.
