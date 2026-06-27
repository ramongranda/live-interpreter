# Live Interpreter — Refactor Unificado: FSM + Supervisor Tokio + UI Reactiva (TDD/SRP)

> Documento de arquitectura vivo. Reemplaza al `architecture_guidelines.md` que inspiró el trabajo.
> Referencia durante las 5 fases de implementación.

## Contexto y objetivo

`Live Interpreter` arrastra una dualidad técnica: estado en 16 flags `bool/String` (`AppStatus`), telemetría GPU por subproceso `nvidia-smi`, ciclo de vida por scripts Bash (`start-local-stack.sh`) + archivos `.pid`, y un protocolo WS asimétrico (`StreamEvent` JSON + frame WAV binario). El objetivo es unificar bajo **Clean Code + TDD estricto (Red→Green→Refactor) + SRP**, gestionando **todo el ciclo de vida en memoria con Tokio** (purgando `nvidia-smi`, Bash y `.pid`), con un contrato de eventos binario simétrico (`PipelineEvent`/Serde+Bincode) y una UI Tauri que mapea 1:1 con `NodeState`.

> **SIN RETROCOMPATIBILIDAD.** El producto es pre-producción (README: "no añadir shims para nombres antiguos"). Se permite romper firmas, borrar enums (`StreamEvent`), colapsar binarios y reescribir el protocolo WS sin capas de compatibilidad. No se conservan los `.sh` ni los `.pid`.

> **DIRECCIÓN CANDLE-NATIVE (decidido — ver §7).** Objetivo: runtime de voz local 100% Rust, todo in-process (tasks Tokio + `CancellationToken`, sin `.pid`/Bash). ASR→Candle Whisper, mic→`pipewire-rs`, TTS tras el trait `VoiceSynthesisBackend` (clon con Qwen3-TTS actual hoy → Candle-Qwen cuando exista el port). §7 supersede las partes de proceso/TTS de §2–§5.

**Ya resuelto (reutilizar, no rehacer):** `src/vram.rs` ya hace NVML-first (`nvml_snapshot`, `vram_snapshot`, `evaluate_preflight`, `gpu_preflight_realtime`, `VramSnapshot`, `GpuProcessMem`); `nvml-wrapper=0.12` y `bincode=1.3` ya son dependencias; `src/mesh.rs` ya tiene libp2p + `MeshRole` + un **trait público `GpuTelemetry`** (colisión de nombre → el struct nuevo se llama `GpuStatus`). Falta añadir `tokio-util` (CancellationToken).

---

## 1. ARQUITECTURA DE DATOS Y ESTADO UNIFICADO

Todo vive en `src/types.rs` (única fuente de los contratos, derivando `Serialize + Deserialize`; los frames de audio usan `bincode`).

### 1.1 Máquina de estados finita — `NodeState`

```rust
/// FSM derivada; nunca se almacena como verdad — se computa cada tick desde
/// (liveness de hijos Tokio + health + preflight VRAM). Reemplaza a role_hint().
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "state", content = "detail", rename_all = "snake_case")]
pub enum NodeState {
    Idle,                          // GPU capaz, nada arrancado
    Preflight,                     // probando/insuficiente VRAM antes de arrancar
    Initializing(Vec<InitStep>),   // server vivo, /health aún no OK; pasos con progreso real
    ActiveServer,                  // pipeline local sirviendo (Modo Proveedor)
    ActiveClient,                  // capturando local y delegando a la malla (Modo Consumidor)
    Error(String),                 // gate bloqueado o un hijo propio murió inesperadamente
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InitStep { pub label: String, pub status: InitStatus, pub elapsed_ms: u64 }

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum InitStatus { Pending, Running(u8 /*pct*/), Ok, Failed(String) }
```

Regla de derivación pura (testeable sin GPU):

```rust
pub fn derive_node_state(
    live: &Liveness, server_healthy: bool, gpu: &GpuStatus,
    init: Option<Vec<InitStep>>, last_err: Option<&str>,
) -> NodeState {
    if let Some(e) = last_err           { return NodeState::Error(e.into()); }
    if live.client                      { return NodeState::ActiveClient; }       // cliente gana
    if live.server {
        return if server_healthy { NodeState::ActiveServer }
               else { NodeState::Initializing(init.unwrap_or_default()) };
    }
    if !gpu.is_capable                  { return NodeState::Preflight; }
    NodeState::Idle
}
```

### 1.2 Contrato de eventos simétrico — `PipelineEvent` (Serde + Bincode)

El protocolo WS pasa de "texto JSON + frame WAV aparte" a **un único frame `bincode` por evento**. Simetría total: cada evento lleva un `Lane` para que **tanto Cliente como Servidor** rendericen las **dos columnas en paralelo** (entrada local: transcripción ES + traducción EN que sale clonada; salida remota: transcripción EN + traducción ES que se escucha).

```rust
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Lane { Local, Remote }   // Local = mi voz (saliente) · Remote = par entrante

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PipelineEvent {
    Ready,
    State(NodeState),                                   // empuje de transiciones FSM
    Telemetry(AppStatus),                              // empuje periódico (VRAM/latencia)
    Listening   { lane: Lane },
    Processing  { id: Uuid, lane: Lane },
    Transcript  { id: Uuid, lane: Lane, lang: Lang, text: String },   // texto en lengua origen
    Translation { id: Uuid, lane: Lane, lang: Lang, text: String },   // texto en lengua destino
    AudioFrame  { id: Uuid, lane: Lane, sample_rate: u32,
                  #[serde(with = "serde_bytes")] pcm: Vec<u8> },       // s16le mono, va en bincode
    Done        { id: Uuid, lane: Lane, latency_ms: u64 },
    Error       { message: String },
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Lang { Es, En }
```

Serialización: `bincode::serialize(&event)` → `Message::Binary` en el WS y en la malla libp2p (request-response ya usa bincode). El mismo enum sirve JSON (panel de control HTTP) y binario (streaming) sin tipos paralelos. **Se elimina `StreamEvent` y `StreamStart`** (los sustituyen `PipelineEvent` + `SessionStart { direction, synthesize }`).

### 1.3 Telemetría y estado de app — `AppStatus` + `GpuStatus`

```rust
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GpuStatus {                 // telemetría NVML en memoria (vía vram.rs / nvml-wrapper)
    pub is_capable: bool,              // free_mb >= min_server_vram_mb
    pub model_name: String,            // VramSnapshot.device_name (ej. "NVIDIA RTX 3080")
    pub vram_free_mb: u64,
    pub vram_total_mb: u64,
    pub utilization_pct: u8,           // NVML utilization.gpu (nuevo en vram.rs)
    pub gate_message: String,          // evaluate_preflight().message (es-ES)
    pub processes: Vec<GpuProcessMem>, // pid/name/used_mb reales
    pub source: String,                // "nvml" | "nvidia-smi" (fallback)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppStatus {
    pub current_state: NodeState,
    pub gpu: GpuStatus,
    pub voice_configured: bool,
    pub active_connections: usize,     // sesiones de malla activas (mesh::GpuTelemetrySnapshot)
    pub services: ServiceHealth,       // dots server/qwen/mic (health independiente)
    pub pipeline_delay_ms: u64,        // última latencia (PipelineEvent::Done) para el sidebar
    pub last_error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServiceDot { pub running: bool, pub healthy: bool }
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServiceHealth { pub server: ServiceDot, pub qwen: ServiceDot, pub mic: ServiceDot }
```

`GpuProcessMem` se reexporta desde `vram.rs`. `Liveness { server, client, qwen, mic: bool }` lo provee el supervisor en memoria (sin `/proc`, sin `.pid`).

---

## 2. MAPEO DE RESPONSABILIDADES POR MÓDULO (SRP)

| Archivo | Se QUEDA | Se ELIMINA / MUEVE |
|---|---|---|
| **`src/types.rs`** | `Direction`, `Segment`, `InterpretResponse`, `TextInterpretRequest`. **Nuevo hogar** de `NodeState`, `PipelineEvent`, `Lane`, `Lang`, `InitStep/InitStatus`, `GpuStatus`, `AppStatus`, `ServiceHealth/Dot`, `SessionStart`. | `StreamEvent`, `StreamStart` (→ `PipelineEvent`/`SessionStart`). `HealthResponse` se pliega en `AppStatus`/`ServiceHealth`. |
| **`src/desktop.rs`** | `DesktopConfig`, perfil de voz (`voice_*`, `save_voice_profile`), núcleo de actores streaming + traits, `derive_node_state`, `build_gpu_status`, `collect_status`. | `gpu_preflight`, `gpu_status`, `best_gpu`, `parse_gpu_preflight`, `GpuInfo`, `GpuProcess`, `role_hint`, `run_script`, `pid_alive`, las 4 fns sueltas `start/stop_*`, `AppStatus` viejo. **Lógica de proceso → `ServiceSupervisor` (FASE 2).** |
| **`src/supervisor.rs`** *(nuevo, en lib)* | `ServiceSupervisor` (orquestador Tokio en memoria): handles `Child`, `CancellationToken`, `liveness()` vía `try_wait`, `cuda_env()` (porta `cuda-env.sh`). Spawnea qwen/mic como hijos propios; sin `setsid`, sin `.pid`, sin Bash. | — |
| **`src/main.rs`** | Punto de entrada **unificado** (FASE 4): `match LI_ROLE { Server → run_server(), Client → run_mesh_client() }`. | Topología actual de 2 binarios separados server/client (se colapsan; sin retrocompat). |
| **`src/routes.rs`** | Endpoints HTTP `interpret_text/file`, `download_audio`, `require_auth`, `persist_response`, `process_audio_path`. WS `/v1/stream/meeting` reescrito a **`PipelineEvent` binario simétrico** vía `tokio::sync::broadcast`. `/v1/init` nuevo (progreso). | `send_event` (JSON), el split `AudioStart`+WAV, `handle_meeting_stream` asimétrico. |
| **`src/asr.rs`** | `transcribe_file`, conversión WAV, resample. | Acoplamiento directo: se expone tras el trait `Transcriber`. |
| **`src/translate/http.rs`** | Cliente Ollama, `translate`, `translate_stream`, `keep_alive`. | Se expone tras el trait `TranslationBackend` (el enum `Translator` ya abstrae). |
| **`src/tts.rs`** | `synthesize`, `QwenTtsRequest` (base64 ref voz). | Se expone tras el trait `Synthesizer`. |

### 2.1 Traits puros para inyección de Mocks (TDD)

`desktop.rs` ya prueba un pipeline con mocks `MockAsr/MockTranslator/MockTts/CountingSink`. Se **formalizan como traits de producción** (object-safe vía `async-trait`, ya es dependencia):

```rust
#[async_trait] pub trait Transcriber:  Send + Sync { async fn transcribe(&self, wav:&Path, lang:Lang) -> Result<Vec<Segment>>; }
#[async_trait] pub trait TranslationBackend: Send + Sync { async fn translate(&self, text:&str, dir:&Direction) -> Result<String>; }
#[async_trait] pub trait Synthesizer:  Send + Sync { async fn synthesize(&self, id:Uuid, text:&str, dir:&Direction) -> Result<Vec<u8>/*pcm*/>; }
#[async_trait] pub trait AudioSink:    Send + Sync { async fn play(&self, frame: GeneratedAudio) -> Result<()>; }
pub trait VramProbe:    Send + Sync { fn snapshot(&self) -> Result<VramSnapshot>; }   // mock sin GPU
#[async_trait] pub trait ProcessSpawner: Send + Sync { async fn spawn(&self, spec: ChildSpec) -> Result<ManagedChild>; } // mock sin OS
```

`AppState` (routes) y `ServiceSupervisor` se parametrizan por estos traits (`Arc<dyn ...>`), de modo que cada fase escribe primero el test con el mock y luego el adaptador real (`AsrEngine: Transcriber`, `vram::NvmlProbe: VramProbe`, `TokioSpawner: ProcessSpawner`).

---

## 3. PLAN DE IMPLEMENTACIÓN INCREMENTAL EN FASES (TDD: Rojo → Verde → Refactor)

Cada fase: **(R)** escribir el test que falla → **(V)** mínimo código para pasarlo → **(F)** refactor. Tras cada fase: `cargo fmt && cargo test --lib` en verde.

### FASE 1 — Tipos y estado unificado (`src/types.rs`)

- **(R)** Tests: `pipeline_event_bincode_roundtrip` (serializa/deserializa cada variante, incl. `AudioFrame` con PCM); `app_status_json_shape` (claves `current_state/gpu/services/...`); `node_state_serde_tagged`; `init_step_status_variants`.
- **(V)** Definir `NodeState`, `PipelineEvent`, `Lane`, `Lang`, `InitStep/Status`, `GpuStatus`, `AppStatus`, `ServiceHealth/Dot`, `SessionStart`. Borrar `StreamEvent`/`StreamStart`.
- **(F)** Mover `derive_node_state`/`build_gpu_status` (puros) a `types.rs` o `desktop.rs`; añadir sus tests por-arm (`idle/preflight/initializing/active_server/client_wins/error`).
- Diagrama (productor→serde→consumidor):

  ```text
  pipeline ──PipelineEvent──► bincode::serialize ──Vec<u8>──► WS/mesh ──► bincode::deserialize ──► UI bucket(lane)
  ```

### FASE 2 — Preflight NVML + orquestador de ciclo de vida (`src/supervisor.rs`, `src/desktop.rs`)

- **(R)** Tests puros (sin GPU/OS) con mocks: `preflight_blocks_when_free_below_min` (`VramProbe` mock); `derive_node_state_*`; `cuda_env_respects_overrides`; `liveness_via_try_wait_no_pidfiles` (usa `ProcessSpawner` mock que devuelve un `Child` falso); `stop_cancels_token_and_kills_owned`; `adopted_service_is_not_killed`.
- **(V)** `ServiceSupervisor` con `children: Mutex<HashMap<&str, ManagedChild>>`, `shutdown: CancellationToken`. Métodos `start_server/stop_server/start_client/stop_client/liveness`. Spawnea con `tokio::process::Command` (qwen: binario + `cuda_env()`; mic: `pw-loopback` argv directo; server/client: el binario unificado con `LI_ROLE`). Sin `setsid`/`.pid`. Stop = SIGTERM (`kill <pid>`) → `timeout(grace, child.wait())` → `child.kill()`. Adoptados (detectados por `pgrep`) se rastrean pero **no se matan**.
- **(F)** Migrar `collect_status(sup, http)` a leer `sup.liveness()` + `vram_snapshot()` + health; borrar `nvidia-smi`/`role_hint`/`run_script` de `desktop.rs`. Añadir `utilization_pct` a `vram.rs`.
- Diagrama (cancelación cooperativa):

  ```text
  start_server: preflight(VramProbe) ─OK─► spawn qwen ─► spawn server(LI_ROLE=server) ─► spawn mic
  stop_server : shutdown.cancel() ─► tareas internas (telemetry/mesh) salen vía select!{cancelled()}
              └► stop_managed(mic→server→qwen): SIGTERM ► wait(grace) ► SIGKILL
  ```

### FASE 3 — Servidor de eventos simétricos (`src/routes.rs`)

- **(R)** Tests: `broadcast_fans_out_to_both_panels` (2 suscriptores reciben el mismo `PipelineEvent`); `lane_tagging_local_vs_remote`; `binary_frame_is_bincode_pipeline_event`; pipeline con mocks emite la secuencia `Processing→Transcript(Local)→Translation(Local)→AudioFrame(Local)→Done`.
- **(V)** Reemplazar `handle_meeting_stream`: un `tokio::sync::broadcast::Sender<PipelineEvent>` por sesión; el bucle ASR→Translate→TTS publica eventos con `lane`; cada socket suscriptor reenvía `bincode`. El servidor publica **ambos lanes** (Local desde su mic/entrada, Remote desde el par) para simetría. Endpoint `GET /v1/init` devuelve `Vec<InitStep>`; el server **bindea HTTP antes** de cargar modelos (carga en task de fondo; `/health` 200 sólo al terminar).
- **(F)** Extraer `run_pipeline(channels, traits)` reutilizable por server y mesh; canales MPSC acotados (reusar `StreamingChannels`/`StreamingBackpressure`).
- Diagrama (MPSC + broadcast, simetría de paneles):

  ```text
  mic_local ─mpsc─► [ASR]─mpsc►[Translate]─mpsc►[TTS]─mpsc►[Sink+pub Local]
  peer_audio ─mpsc─►[ASR]─mpsc►[Translate]─mpsc►[TTS]─mpsc►[Sink+pub Remote]
                                   broadcast::Sender<PipelineEvent>
                                     ├─► socket Cliente  (ve Local+Remote)
                                     └─► socket Servidor (ve Local+Remote)  ⇒ paneles idénticos
  ```

### FASE 4 — Punto de entrada unificado (`src/main.rs`)

- **(R)** Tests: `role_from_env_parses_server_and_client`; `unknown_role_errors`; `run_server_wires_real_traits` (smoke con mocks, sin red).
- **(V)** `main()` lee `LI_ROLE` (`server`|`client`, default `server`) y ramifica: `run_server()` (carga `AsrEngine`/`Translator`/`TtsEngine` reales tras los traits, sirve `app(state)`); `run_mesh_client()` (captura cpal + VAD → malla libp2p, reusa `run_pipeline`). El supervisor (FASE 2) spawnea este binario con el rol correcto — **cero Bash**. Colapsar `live-interpreter-client` dentro de `main.rs` (sin retrocompat).
- **(F)** `cuda-env.sh`/`start-local-stack.sh`/`stop-local-stack.sh`/`create-virtual-mic.sh` quedan obsoletos (borrar del flujo; opcional dejarlos como debug manual). Actualizar README/CLAUDE.md.

### FASE 5 — Adaptador de interfaz Tauri (FSM reactiva + Estudio de Voz)

- **(R)** Tests: `app_status_serializes_for_ipc`; `should_capture_blocks_during_playback` (guard half-duplex puro); `voice_profile_save_and_read_roundtrip` (se conserva).
- **(V)** Tauri commands delegan en `Arc<ServiceSupervisor>`; `spawn_vram_telemetry` hace `select!` sobre `shutdown_token().cancelled()` y emite `gpu-telemetry`/`PipelineEvent::Telemetry`. Frontend reescrito: una vista por `current_state` (ver §4). Estudio de Voz reusa `start/stop_voice_recording`+`save_voice_profile` (muestra "El Quijote", 30s, medidor de ganancia). **Guard de feedback half-duplex** (`Arc<AtomicBool> playback_active` + hangover 300ms) en cliente y captura de malla, con `should_capture(half_duplex, playback_active, paused, muted)` puro y testeado.
- **(F)** Al cerrar ventana: `shutdown.cancel()` + `stop_server/stop_client` (teardown estricto, sin detach).

---

## 4. DISEÑO VISUAL FINAL COMPLETO

**Sistema de diseño.** Tema oscuro (`#0E1116` fondo, `#161B22` tarjetas, borde `#2D333B`), acento teal `#2DD4BF` (estado OK/activo), verde `#3FB950` (banner "SYSTEM READY"/Save), ámbar `#D29922` (Initializing/Preflight), rojo `#F85149` (Error/Stop/Cancel). Tipografía Inter; íconos line. Ventana 1024×600, barra de título con engranaje (⚙ Voice Studio) a la derecha. La UI es **reflejo puro de `AppStatus`**; nunca inspecciona procesos.

**Enrutado Estado → Pantalla (1:1 con `NodeState`):**

| `NodeState` | Pantalla | Componentes clave |
|---|---|---|
| `Idle` / `Preflight` | **Configuración** | Banner `SYSTEM READY`; 2 tarjetas rol: **Server** (muestra `gpu.model_name` + `vram_free_mb/total`; botón *Start Server* deshabilitado si `!gpu.is_capable`, tooltip=`gate_message`) y **Client** ("Connect to Mesh", "Looking for servers…"). Fila *Voice Profile Settings →*. Engranaje → Voice Studio. |
| `Initializing(steps)` | **Carga** | Spinner circular + lista de `InitStep`: `label [status, elapsed_ms]` → `✓ Verificando VRAM (NVML)… [OK, 0.4s]`, `✓ Carga Whisper (large-v3) GPU… [OK, 2.1s]`, `Sincronización Malla/Clon… [Pendiente]`. Botón rojo **Cancelar operación** (`shutdown.cancel()`). |
| `ActiveServer` | **Consola — Servidor** | Vista de 3 columnas (abajo). Sidebar derecho etiqueta `MODO: SERVIDOR GPU`. |
| `ActiveClient` | **Consola — Cliente** | Misma vista; badge `LIVE-INTERPRETER-MIC-SOURCE ACTIVO` (de `services.mic.running`). |
| `Error(msg)` | **Banner Error** | Banda roja con `msg` semántico sobre la pantalla de Configuración; permite reintentar. |

**Consola activa (simétrica, idéntica en Cliente y Servidor)** — alimentada por `PipelineEvent`:

- **Columna [INPUT: TU VOZ]** = `lane == Local`: burbuja `Transcript` (ES, lengua origen) encima de `Translation` (EN, lo que sale clonado), con ícono 🔊.
- **Columna [OUTPUT: SU VOZ]** = `lane == Remote`: `Transcript` (EN) encima de `Translation` (ES, lo que escuchas).
- **Toggle de dirección ⇄** (izquierda) conmuta es↔en (`Direction`).
- **Sidebar de telemetría** (derecha, vertical): `GPU 65%` (`gpu.utilization_pct`), `VRAM 4.2/16 GB` (`gpu.vram_free/total`), `Delay 120ms` (`pipeline_delay_ms`), botón rojo **Stop Interpretation**.

**Modal Voice Studio** (sobre Configuración, vía engranaje): título *Configure Your Voice Profile*; texto de referencia precargado (El Quijote: *"En un lugar de la Mancha…"*); **Start Recording** con forma de onda + medidor de ganancia en vivo; `Reference: reference.wav`; barra *Recording Progress (0:00 / 0:30)* (tope 30s); **Play Back** + **Save Profile**; confirmación *"Voice Profile Saved locally as data/voice/reference.wav"*. Reusa `save_voice_profile` (escribe `data/voice/reference.{wav,txt}`). ⚠️ Reconciliar formato: el mockup rotula `44.1kHz 16-bit mono` pero `encode_wav_24k` escribe **24kHz** — decidir en impl (subir sample rate o ajustar rótulo).

---

## 5. Identidad Vocal (Voice Identity) — ciudadano de primera clase

El diferenciador del producto **no** es traducir: es **presencia vocal** — "hablo español → el otro me oye en inglés **con un timbre parecido al mío**". Por eso `VoiceProfile` sube del detalle interno de TTS a **entidad de dominio de primer nivel**. Encuadre legal/ético: se vende como **perfil vocal personalizado / voz sintética basada en tu timbre**, nunca como "clon exacto".

### 5.1 Modelo enriquecido con consentimiento

```rust
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VoiceProfile {
    pub id: Uuid,
    pub owner: String,                 // quién es la voz (auto-declarado)
    pub created_at: DateTime<Utc>,
    pub consent_confirmed: bool,       // gate duro: sin esto NO se sintetiza con el perfil
    pub reference_audio_path: PathBuf, // data/voice/reference.wav
    pub reference_text: String,        // transcripción exacta de la muestra
    pub sample_rate: u32,              // 24000 hoy (reconciliar rótulo UI 44.1k)
}
```

- Persistencia: sidecar `data/voice/profile.json` junto a `reference.{wav,txt}`. `save_voice_profile` **exige `consent_confirmed == true`** (validación nueva, mensaje es-ES) antes de escribir.
- Solo se permiten perfiles **propios o con permiso explícito** del titular de la voz. El gate de consentimiento es de dominio, no de UI.

### 5.2 `VoiceRenderer` trait — backend de síntesis pluggable

Desacoplar el pipeline del backend concreto: el `Synthesizer` de §2.1 se generaliza a `VoiceRenderer`, que recibe el **perfil** (no solo el texto). Así se intercambia Qwen3-TTS por XTTS / OpenVoice / Piper / Coqui / ElevenLabs sin tocar el pipeline.

```rust
#[async_trait]
pub trait VoiceRenderer: Send + Sync {
    async fn synthesize(&self, text: &str, lang: Lang, profile: &VoiceProfile) -> Result<AudioFrame>;
    /// Síntesis por chunks para baja latencia (ver 5.3); default: una sola llamada.
    async fn synthesize_stream(&self, text: &str, lang: Lang, profile: &VoiceProfile)
        -> Result<AudioChunkStream> { /* default = wrap synthesize */ }
}
// Implementaciones: QwenVoiceRenderer (actual), NeutralVoiceRenderer (voz neutra, sin perfil), …
```

### 5.3 Baja latencia: cascada parcial

El enemigo en conversación es esperar a la frase completa. Pipeline objetivo (sobre `translate_stream` ya existente en `translate/http.rs` + TTS por chunks):

```text
ASR parcial ─► Traducción parcial (token stream) ─► VoiceRenderer.synthesize_stream ─► AudioFrame chunks
```

Cada chunk emite un `PipelineEvent::AudioFrame { lane, ... }` en cuanto está, sin barrera de fin de frase. Métrica en el sidebar (`pipeline_delay_ms`) = time-to-first-audio-chunk.

### 5.4 UI — selector de identidad vocal (feature central)

En Configuración / Voice Studio:

```text
Voice Identity
[•] Usar mi perfil vocal      (requiere consent_confirmed)
[ ] Usar voz neutra
[ ] No sintetizar (solo mostrar traducción)
```

En modo activo, por lane: **Local** → idioma destino con *tu* perfil vocal; **Remote** → idioma destino con voz neutra o el perfil del par. El selector se traduce a qué `VoiceRenderer` instancia el pipeline y al flag `synthesize` de `SessionStart`.

> Reposicionamiento: el sistema no es "un traductor", es un **intérprete de voz con identidad vocal controlada por el usuario** (con consentimiento).

## 6. Verificación end-to-end

- `cargo fmt` · `git diff --check`.
- `cargo test` (units puras: FSM, `PipelineEvent` bincode roundtrip, supervisor liveness/cancel con mocks, half-duplex guard; + compilan los binarios).
- Builds pre-merge (README): `cargo build --release --features desktop-native --bin live-interpreter-desktop` y `cargo build --release --bin live-interpreter --bin live-interpreter-control --bin live-interpreter-ws-smoke` (el `-client` queda colapsado en `main.rs`).
- Manual: panel → `/api/status` con la forma nueva; *Start Server* → gate VRAM bloquea con mensaje correcto cuando hay poca VRAM → `Initializing` muestra pasos vía `/v1/init` → `ActiveServer`; `pactl list short sources | grep live-interpreter-mic-source` presente; cerrar panel → hijos propios (qwen/server/mic) caen (`pgrep` vacío), un servicio adoptado sobrevive. Half-duplex: reproducir TTS al `-sink` **no** re-dispara el VAD (sin utterance fantasma durante/tras la reproducción). Dos clientes a una sesión ven **paneles idénticos** (simetría `broadcast`).

## Archivos

`src/types.rs` (contratos unificados), `src/supervisor.rs` (nuevo), `src/desktop.rs` (FSM/telemetría, purga nvidia-smi/pid/bash), `src/routes.rs` (broadcast `PipelineEvent` + `/v1/init`), `src/main.rs` (entrada unificada `LI_ROLE`), `src/asr.rs`/`src/tts.rs`/`src/translate/http.rs` (traits para mocks), `src/bin/live-interpreter-desktop.rs` + `desktop/index.html` + `src/bin/live-interpreter-control.rs` (UI FSM), `Cargo.toml` (`tokio-util`).

---

## 7. REVISIÓN — Dirección Candle-native (decidido)

Reposicionamiento del producto: **Voice Identity Runtime** — *un runtime de voz local escrito en Rust*, no "una app que llama a modelos". Captura, traduce, sintetiza y enruta la voz **on-device, sin nube, sin Python como arquitectura**. Esta sección supersede las partes de proceso/TTS/ASR de §2–§5.

### 7.1 Decisiones

1. **TTS clon detrás del trait `VoiceSynthesisBackend`.** Se adopta el trait ya. El Qwen3-TTS actual (servicio libtorch en :8020) queda como **un backend de clonación enchufable** (`HttpQwenBackend`), no como arquitectura. `CandleKokoroBackend` (voz neutra, on-device) y `MockVoiceBackend` (test) completan la jerarquía. Migración a `CandleQwen3Backend` cuando exista el port (spike R8). ⚠️ Honestidad: Qwen3-TTS clon **no existe** en Candle hoy; Kokoro **no clona** (voz fija).
2. **ASR = Candle Whisper.** `CandleWhisperBackend: Transcriber` sustituye a `whisper-rs`/whisper.cpp → ASR 100% Rust, se elimina el toolchain C++ y la feature `cuda` ligada a whisper-rs.
3. **Mic = `pipewire-rs` nativo in-process.** Un módulo `VirtualMic` crea el sink/source en-proceso; se eliminan `pw-loopback` y `create-virtual-mic.sh`.

### 7.2 Implicación arquitectónica (gran win)

Con ASR/translate/mic in-process, **desaparecen 2 de los 3 procesos externos**. Sólo queda **1 hijo externo** (servicio Qwen3-TTS de clon) hasta que exista Candle-Qwen. El `LiveRuntime` (ex-`ServiceSupervisor`, módulo `src/runtime.rs`) se **reorienta**: de spawnear procesos OS a **supervisar tasks Tokio** con `CancellationToken` y `JoinHandle` (la `Liveness` de tasks via handle, no `/proc`/`pgrep`). Esto cumple **genuinamente** el ideal del `architecture_guidelines` original (sin `.pid`, sin Bash) que antes era inviable por los procesos externos. **Los tipos FSM (§1) y la base ya verde no cambian**; el supervisor muta su capa de proceso.

### 7.3 Trait de síntesis (streaming, baja latencia)

```rust
#[async_trait]
pub trait VoiceSynthesisBackend: Send + Sync {
    async fn synthesize_stream(
        &self,
        req: VoiceSynthesisRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<AudioFrame>> + Send>>>;
}
// Impls: HttpQwenBackend (clon, actual) · CandleKokoroBackend (neutro, on-device) · MockVoiceBackend (test)
```

`VoiceSynthesisRequest { text, lang, profile: VoiceProfile, neutral: bool }`. El pipeline chunked (ASR parcial → translate parcial → `synthesize_stream`) emite `PipelineEvent::AudioFrame` por chunk; `pipeline_delay_ms` = time-to-first-chunk.

**Enrutado por-lane (`route_for_lane`).** El backend se elige por lane según el selector `VoiceIdentity`:

- `MyProfile`: **Local** → `Clone` (tu timbre, `HttpQwenBackend` + perfil) · **Remote** → `Neutral` (**Kokoro**, monitor: oír la traducción de lo que dijo el otro, sin fingir su voz).
- `Neutral`: Kokoro en ambos lanes. · `Off`: no sintetizar (solo texto).

Es decir, **Kokoro se usa cuando quieres escuchar la traducción del origen** (lane Remote/monitor); tu clon solo sale en el lane Local. Política pura y testeada en `voice::route_for_lane`.

### 7.4 Voice Identity Runtime (modelo de dominio)

`VoiceProfile` rico (no "un wav suelto"): `{ id, name, owner, consent_confirmed, samples: Vec<VoiceSample>, embedding_path, default_lang, quality_score, created_at }`; `VoiceSample { path, transcript, lang, duration_ms, sample_rate }`. Gate de consentimiento duro antes de sintetizar con perfil. La `VoiceProfile` de status de `desktop.rs` se pliega aquí en el cutover.

### 7.5 Stack actualizado (deps)

- **+** `candle-core`/`candle-nn`/`candle-transformers`, `tokenizers`, `hf-hub` (ASR Whisper; futura TTS Candle).
- **+** `pipewire-rs`, `rubato` (resample), `symphonia` (decode), `webrtc-vad` (o VAD propio).
- **−** `whisper-rs` (tras swap ASR), `pw-loopback`/`create-virtual-mic.sh`, scripts Bash de stack.
- (ya presentes: `tokio`, `tauri`, `serde`, `bincode`, `tracing`, `uuid`, `anyhow`/`thiserror`, `cpal`, `hound`.)

### 7.6 Roadmap Candle-native (reordenado por feedback; `VirtualMic` antes)

Primer hito demo brutal: **texto fijo → CandleKokoro → micrófono virtual PipeWire** (demuestra el runtime Rust completo antes de meter ASR/traducción). Por eso `VirtualMic` sube.

- **R0 (hecho, verde):** tipos FSM + `LiveRuntime` + telemetría NVML.
- **R1 (hecho, verde):** `VoiceSynthesisBackend` + `VoiceProfile`/`VoiceSample` + `AudioSpec` + `EventEnvelope` + `MockVoiceBackend` + `route_for_lane`.
- **R2 (hecho, verde):** `VirtualMic` con `pipewire-rs` — trait `AudioOutput` + `MockAudioOutput` (default) + `PipewireVirtualMic` (stream `Audio/Source` `live-interpreter-mic-source`, main loop en hilo propio, PCM por ring buffer) tras la feature **`native-audio`** (dep `pipewire` opcional; necesita `libpipewire-0.3-dev`+`libclang`). Verificación en vivo = manual (`pw-cli`).
- **R4 (hecho, verde):** `HttpQwenBackend` con `VoiceProfile` (clon, funciona hoy; `wav_to_audio_frame` puro).
- **DEMO e2e (verificado en vivo):** bin `li-voice-demo` (`--features native-audio`): `texto → HttpQwenBackend (Qwen3-TTS :8020) → PipewireVirtualMic`. Probado contra el sistema real: sintetiza, crea el nodo `Audio/Source` `live-interpreter-mic-source` (visible en `pw-cli`), reproduce, exit 0. La tesis Candle-native (Rust produce voz e inyecta en mic virtual, sin Bash/pw-play) **funciona hoy con el clon**.
- **R6-core (hecho, verde):** `pipeline::interpret_utterance` + traits `Transcriber`/`TextTranslator` (impls reales sobre `AsrEngine`/`Translator`) + tests con mocks (`Processing→Transcript→Translation→AudioFrame→Done`, consent gate en lane Clone, `Off` no toca mic).
- **R6-bin (VERIFICADO EN VIVO end-to-end):** bin `li-interpret` (`--features native-audio`): cpal capture + VAD energético → `interpret_utterance` (Whisper + Ollama + `HttpQwenBackend` clon + `PipewireVirtualMic`). Probado contra el sistema real: dije "Gracias" al mic → `· ES: Gracias.` → `· EN: Thanks.` → `→ 1 audio chunk(s) to virtual mic`. **El traductor de voz en tiempo real con clon funciona, 100% Rust + 1 servicio Qwen, cero Bash.** ⚠️ Latencia ~23s = Whisper en **CPU** (build sin `cuda`); fix: `--features cuda,native-audio` (Whisper GPU) → R5/R7 la bajan más.
- **Latencia (medido, en vivo):** `LI_WHISPER_MODEL` controla el trade-off CPU. **large-v3-turbo ≈ 23.6s/utterance → base ≈ 6.0s (4×)**, cero deps. Recomendado para tiempo real: `data/models/ggml-base.bin`. Piso CPU = modelo + round-trip Ollama+Qwen. ⚠️ **GPU bloqueado**: `nvcc` ausente → ni `whisper-rs/cuda` ni `candle-cuda` (candle-kernels compila `.cu` con nvcc) construyen; GPU-whisper requiere instalar CUDA Toolkit. RTX 5060 Ti con solo ~3.2GB libres (Qwen+Ollama ocupan el resto).
- **R5 (port nativo, pendiente):** `CandleWhisperBackend: Transcriber` (drop whisper.cpp). Valor = cleanup 100% Rust; **no** baja latencia en CPU vs un modelo pequeño (su win de latencia necesita GPU → nvcc). Integración grande (hf-hub + mel + decode).
- **R3:** `CandleKokoroBackend` → `AudioFrame` → `VirtualMic` (voz neutra on-device; **research**: Kokoro no es first-party en Candle).
- **R7 (hecho, verde):** chunked synthesis — `pipeline::interpret_utterance_chunked` + `split_clauses` (puro) + `render_chunks`: parte la traducción en cláusulas y sintetiza+inyecta cada una en orden, así el mic reproduce la cláusula 1 mientras se sintetiza la 2 (baja time-to-first-audio en multi-frase). Cableado en `li-interpret`. ⚠️ El piso de latencia sigue siendo **ASR CPU**; chunked no lo baja (gating en Whisper). Bajar el piso = GPU (nvcc) o modelo más pequeño.
- **R8-UI (VERIFICADO EN VIVO):** panel reactivo `li-control` (Axum) + `static/fsm-ui.html`. `LiveRuntime::app_status()` (+`assemble_app_status` puro, testeado) proyecta liveness in-memory + health + **NVML** (`build_gpu_status`) por la FSM → `/api/status` devuelve el `AppStatus`/`NodeState` nuevo; la UI renderiza una pantalla por `current_state` (Idle/Preflight/Initializing/Active) con los design tokens del mockup. Probado: con 3260MB libres < gate 8000MB → `current_state:"preflight"`, `gpu.source:"nvml"` (cero `nvidia-smi`), `health.asr:"ready"`. Tarjeta Server deshabilitada con `gate_message`, como el mockup.
- **R8-WS (VERIFICADO EN VIVO):** consola activa con burbujas en vivo. `src/events.rs` `EventHub` (broadcast `EventEnvelope`, fan-out testeado) + `li-interpret` ahora sirve su propia UI+WS (un proceso = captura+pipeline+UI+WS): `/api/status`→`active_client` (NVML real), `/ws` reenvía cada `PipelineEvent` como JSON, `fsm-ui.html` abre el WS y pinta `Transcript`/`Translation` por lane + telemetría (GPU%/VRAM/Delay) en vivo. Probado: status active_client + UI + WS arriba.
- **R8-purga (HECHO, verde):** borrados los 2 adaptadores viejos (`live-interpreter-control`/`-desktop` + `desktop/index.html`) y el código muerto de `desktop.rs` (`nvidia-smi` `gpu_preflight`/`gpu_status`/`best_gpu`/`parse_gpu_preflight`, `GpuInfo`/`GpuProcess`, `role_hint`, `run_script`/Bash, `collect_status`, `AppStatus` viejo, las 4 fns `start/stop_*` sueltas + sus tests) y los 3 scripts de orquestación (`start-local-stack.sh`/`stop-local-stack.sh`/`create-virtual-mic.sh`). `desktop.rs` queda solo con `DesktopConfig`, perfil de voz, actores streaming + traits, `GpuPreflight`, `pid_alive`. Build verde, lib 63 tests. **Cero `nvidia-smi`/Bash en todo el árbol activo.**
- **R8-tail:** ✅ quitados `tauri`/`tauri-build`/`desktop-native` + `build.rs` (árbol de deps más ligero, build verde). ⏳ pendiente (legacy, rendimiento decreciente): `main.rs` `LI_ROLE` (colapsar `live-interpreter-client` en el server) y reescribir el WS provider de `routes.rs` (`process_audio_path` + `StreamEvent`) a `PipelineEvent`/`EventHub`. No es el flujo primario — `li-interpret`+su consola WS ya usan `PipelineEvent`.
- **R9:** spike `CandleQwen3Backend` (research; cuando el resto esté sólido).
- **R10:** mesh (la voz traducida viaja a otro nodo).

### 7.7 Refinamientos aplicados (feedback) — ya en código y verde

1. **`ServiceSupervisor` → `LiveRuntime`** (módulo `src/runtime.rs`): supervisa el nodo completo, no solo servicios.
2. **`ServiceHealth`/`ServiceDot` → `RuntimeHealth`** con `ComponentHealth { state, detail }` y `ComponentState { Stopped, Starting, Ready, Degraded, Failed }` por componente (`asr`, `translator`, `voice_renderer`, `audio_input`, `audio_output`, `virtual_mic`, `mesh`). Encaja con tasks internas, no "servicios".
3. **`EventEnvelope { version, session_id, seq, timestamp_ms, event }`** (+ `PROTOCOL_VERSION`): versión, orden, sesión y debugging para WS/mesh.
4. **`AudioSpec { sample_rate, channels, format }` + `AudioFormat { PcmS16Le, PcmF32Le }`**; `PipelineEvent::AudioFrame { id, lane, spec, pcm }` y `voice::AudioFrame { spec, pcm }`. Evita líos al mezclar Candle/PipeWire/resample/chunks.
5. **`TelemetrySnapshot { gpu, active_connections, pipeline_delay_ms }`** separado de `AppStatus`: `PipelineEvent::Telemetry(TelemetrySnapshot)` (ligero, audio-rate); `AppStatus` (pesado) queda para `/api/status` / IPC Tauri. `AppStatus::telemetry()` proyecta uno del otro.
6. **`VirtualMic` (`pipewire-rs`) sube en el roadmap** (R2), para el hito demo "Rust mete voz por micrófono virtual".

> Frontera estable: **Candle-native como dirección, `VoiceSynthesisBackend` como frontera** — permite construir muy-Rust sin quedar atrapado si Qwen3-en-Candle tarda.

> Pitch: *"A local-first Rust voice interpreter that performs real-time speech translation and synthetic voice-identity rendering fully on-device."*
