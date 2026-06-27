//! `LiveRuntime`: in-process Tokio node runtime + lifecycle supervisor.
//!
//! Supervises the whole node, not just external services. Replaces the Bash
//! stack orchestration (`start-local-stack.sh`) and the `data/logs/*.pid` files.
//! As the Candle-native migration lands, the managed *processes* shrink to the
//! one remaining external child (the Qwen3-TTS cloning service) while ASR /
//! translate / voice-render / audio / mesh become in-process Tokio tasks shut
//! down cooperatively through the runtime `CancellationToken`. Today it still
//! owns its managed children as `tokio::process::Child` handles and tracks
//! liveness in memory via `/proc` on their pids.
//!
//! Honest scope notes:
//!   * Owned children die with the supervisor (no `setsid`). This is the
//!     intended strict-teardown behavior: closing the app stops the stack.
//!   * Externally-running instances are *adopted* (tracked for status) but are
//!     **not** killed on stop — we only kill what we spawned.
//!   * Ollama is not managed here (assumed already running as a system daemon).

use crate::desktop::{ActionResult, DesktopConfig, voice_profile};
use std::collections::HashMap;
use std::path::Path;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::types::Liveness;

pub const QWEN: &str = "qwen3-tts";
pub const SERVER: &str = "live-interpreter";
pub const MIC: &str = "live-interpreter-mic";
pub const CLIENT: &str = "live-interpreter-client";

/// Grace period for SIGTERM before escalating to SIGKILL.
const STOP_GRACE: Duration = Duration::from_secs(5);

/// One supervised OS process: its display name and the `pgrep -f` pattern used
/// for adopt-vs-spawn idempotency (mirrors `start-local-stack.sh`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManagedService {
    QwenTts,
    Server,
    Mic,
    Client,
}

impl ManagedService {
    pub fn name(self) -> &'static str {
        match self {
            ManagedService::QwenTts => QWEN,
            ManagedService::Server => SERVER,
            ManagedService::Mic => MIC,
            ManagedService::Client => CLIENT,
        }
    }

    /// Regex for `pgrep -f`. Word-boundary anchors avoid `live-interpreter`
    /// matching `live-interpreter-client`/`-control`/`-desktop`.
    pub fn pgrep_pattern(self) -> &'static str {
        match self {
            ManagedService::QwenTts => "api_server_gpu_torch212",
            ManagedService::Server => "target/release/live-interpreter( |$)",
            ManagedService::Mic => "pw-loopback.*live-interpreter-mic",
            ManagedService::Client => "target/release/live-interpreter-client",
        }
    }
}

/// A tracked child. `child = None` means an *adopted* external process: we hold
/// only its pid and must not kill it on stop.
struct ManagedChild {
    child: Option<Child>,
    pid: u32,
    adopted: bool,
}

/// In-process node runtime. Held behind `Arc` and shared by both control
/// adapters; owns the lifecycle of the node's components/processes.
pub struct LiveRuntime {
    config: DesktopConfig,
    children: Mutex<HashMap<&'static str, ManagedChild>>,
    shutdown: CancellationToken,
}

impl LiveRuntime {
    pub fn new(config: DesktopConfig) -> Arc<Self> {
        Arc::new(Self {
            config,
            children: Mutex::new(HashMap::new()),
            shutdown: CancellationToken::new(),
        })
    }

    pub fn config(&self) -> &DesktopConfig {
        &self.config
    }

    /// In-process tasks (vram telemetry sampler, mesh runtime, capture) clone
    /// this and `select!` on `cancelled()` for cooperative shutdown.
    pub fn shutdown_token(&self) -> CancellationToken {
        self.shutdown.clone()
    }

    /// Snapshot of liveness for all four services in a single lock pass.
    pub async fn liveness(&self) -> Liveness {
        let map = self.children.lock().await;
        let alive = |name: &str| map.get(name).map(|mc| pid_alive(mc.pid)).unwrap_or(false);
        Liveness {
            server: alive(SERVER),
            client: alive(CLIENT),
            qwen: alive(QWEN),
            mic: alive(MIC),
        }
    }

    /// Real-time VRAM gate + spawn of qwen3-tts, the GPU server, and the mic.
    pub async fn start_server(&self) -> ActionResult {
        let gpu = crate::vram::gpu_preflight_realtime(self.config.min_server_vram_mb).await;
        if !gpu.ready {
            return ActionResult {
                ok: false,
                output: gpu.message,
            };
        }

        let mut log = String::new();
        for build in [
            self.qwen_command(),
            self.server_command(),
            self.mic_command(),
        ] {
            let (service, command) = build;
            match self.spawn_child(service, command).await {
                Ok(message) => log.push_str(&format!("{}\n", message)),
                Err(error) => {
                    return ActionResult {
                        ok: false,
                        output: format!("{log}{}: {error}", service.name()),
                    };
                }
            }
        }
        ActionResult {
            ok: true,
            output: log.trim_end().to_string(),
        }
    }

    /// Stop the stack in reverse spawn order (mic → server → qwen) so the mic
    /// sink is not yanked from under the server.
    pub async fn stop_server(&self) -> ActionResult {
        for service in [
            ManagedService::Mic,
            ManagedService::Server,
            ManagedService::QwenTts,
        ] {
            self.stop_child(service.name(), STOP_GRACE).await;
        }
        ActionResult {
            ok: true,
            output: "Servidor parado".into(),
        }
    }

    pub async fn start_client(&self) -> ActionResult {
        let (service, command) = self.client_command();
        match self.spawn_child(service, command).await {
            Ok(message) => ActionResult {
                ok: true,
                output: message,
            },
            Err(error) => ActionResult {
                ok: false,
                output: error,
            },
        }
    }

    pub async fn stop_client(&self) -> ActionResult {
        let stopped = self.stop_child(CLIENT, STOP_GRACE).await;
        ActionResult {
            ok: true,
            output: if stopped {
                "Cliente parado".into()
            } else {
                "Cliente no estaba arrancado".into()
            },
        }
    }

    // ---- command builders ------------------------------------------------

    fn qwen_command(&self) -> (ManagedService, Command) {
        let root = &self.config.root;
        let install = std::env::var("LI_QWEN_INSTALL_DIR")
            .map(Into::into)
            .unwrap_or_else(|_| root.join("vendor/qwen3_tts_rs"));
        let api = std::env::var("LI_QWEN_API_SERVER")
            .map(Into::into)
            .unwrap_or_else(|_| install.join("api_server_gpu_torch212"));
        let model = install.join("models/Qwen3-TTS-12Hz-0.6B-CustomVoice");
        let device = std::env::var("LI_QWEN_DEVICE").unwrap_or_else(|_| "cuda:0".into());

        let mut cmd = Command::new(api);
        cmd.current_dir(root)
            .arg(model)
            .args(["--device", &device, "--host", "127.0.0.1", "--port", "8020"])
            .envs(cuda_env(&self.config));
        for (key, value) in self.voice_ref_envs() {
            cmd.env(key, value);
        }
        (ManagedService::QwenTts, cmd)
    }

    fn server_command(&self) -> (ManagedService, Command) {
        let mut cmd = Command::new(self.config.root.join("target/release/live-interpreter"));
        cmd.current_dir(&self.config.root)
            .env("LI_ROLE", "server")
            .env("LI_BIND", &self.config.stack_bind);
        for (key, value) in self.voice_ref_envs() {
            cmd.env(key, value);
        }
        (ManagedService::Server, cmd)
    }

    fn mic_command(&self) -> (ManagedService, Command) {
        let name =
            std::env::var("LI_VIRTUAL_NAME").unwrap_or_else(|_| "live-interpreter-mic".into());
        let latency = std::env::var("LI_VIRTUAL_LATENCY").unwrap_or_else(|_| "30ms".into());
        let mut cmd = Command::new("pw-loopback");
        cmd.args(["--name", &name, "--latency", &latency])
            .arg(format!(
                "--capture-props=media.class=Audio/Sink node.name={name}-sink node.description={name}-sink audio.position=[FL]"
            ))
            .arg(format!(
                "--playback-props=media.class=Audio/Source node.name={name}-source node.description={name}-source audio.position=[FL]"
            ))
            .args(["--channels", "1", "-m", "[ FL ]"]);
        (ManagedService::Mic, cmd)
    }

    fn client_command(&self) -> (ManagedService, Command) {
        let mut cmd = Command::new(
            self.config
                .root
                .join("target/release/live-interpreter-client"),
        );
        cmd.current_dir(&self.config.root)
            .env("LI_ROLE", "client")
            .env("LI_SERVER_URL", &self.config.server_url)
            .env("LI_CLIENT_BIND", &self.config.client_bind)
            .env("LI_CLIENT_PLAY_TARGET", &self.config.play_target);
        (ManagedService::Client, cmd)
    }

    /// Voice-profile env injection (same contract as the old `start_server`).
    fn voice_ref_envs(&self) -> Vec<(String, String)> {
        let profile = voice_profile(&self.config);
        if !profile.configured {
            return Vec::new();
        }
        vec![
            ("LI_QWEN_MODEL".into(), "Qwen3-TTS-12Hz-0.6B-Base".into()),
            (
                "LI_QWEN_TTS_MODEL".into(),
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base".into(),
            ),
            ("LI_VOICE_REF".into(), profile.audio_path),
            ("LI_VOICE_REF_TEXT".into(), profile.reference_text),
        ]
    }

    // ---- spawn / stop primitives ----------------------------------------

    /// Idempotent spawn: no-op if we already own a live handle; adopt an
    /// externally-running instance (never killed on stop); else spawn fresh.
    async fn spawn_child(
        &self,
        service: ManagedService,
        mut command: Command,
    ) -> Result<String, String> {
        let name = service.name();
        let mut map = self.children.lock().await;

        if let Some(existing) = map.get(name) {
            if pid_alive(existing.pid) {
                return Ok(format!("{name} ya esta arrancado"));
            }
        }
        if let Some(pid) = pgrep(service.pgrep_pattern()).await {
            map.insert(
                name,
                ManagedChild {
                    child: None,
                    pid,
                    adopted: true,
                },
            );
            return Ok(format!(
                "{name} adoptado (pid {pid}); no se matara al parar"
            ));
        }

        let (out, err) = open_logs(&self.config, name).map_err(|e| e.to_string())?;
        let child = command
            .stdout(Stdio::from(out))
            .stderr(Stdio::from(err))
            .spawn()
            .map_err(|e| e.to_string())?;
        let pid = child.id().unwrap_or_default();
        map.insert(
            name,
            ManagedChild {
                child: Some(child),
                pid,
                adopted: false,
            },
        );
        Ok(format!("{name} arrancado (pid {pid})"))
    }

    /// SIGTERM → wait(grace) → SIGKILL for owned children. Adopted children are
    /// dropped from tracking but never killed. Returns whether anything was tracked.
    async fn stop_child(&self, name: &'static str, grace: Duration) -> bool {
        let Some(mut entry) = self.children.lock().await.remove(name) else {
            return false;
        };
        if entry.adopted {
            return true;
        }
        // Graceful SIGTERM by pid.
        let _ = Command::new("kill")
            .arg(entry.pid.to_string())
            .status()
            .await;
        if let Some(mut child) = entry.child.take() {
            match tokio::time::timeout(grace, child.wait()).await {
                Ok(_) => {}
                Err(_) => {
                    let _ = child.kill().await;
                    let _ = child.wait().await;
                }
            }
        }
        true
    }
}

/// Liveness primitive: `/proc/<pid>` existence (Linux).
fn pid_alive(pid: u32) -> bool {
    pid != 0 && Path::new(&format!("/proc/{pid}")).exists()
}

/// First pid matching the `pgrep -f` pattern, excluding our own process.
async fn pgrep(pattern: &str) -> Option<u32> {
    let output = Command::new("pgrep")
        .arg("-f")
        .arg(pattern)
        .output()
        .await
        .ok()?;
    let me = std::process::id();
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.trim().parse::<u32>().ok())
        .find(|&pid| pid != me)
}

/// Open truncate-create stdout + a cloned handle for stderr under `data/logs`.
fn open_logs(
    config: &DesktopConfig,
    name: &str,
) -> std::io::Result<(std::fs::File, std::fs::File)> {
    let dir = config.root.join("data/logs");
    std::fs::create_dir_all(&dir)?;
    let out = std::fs::File::create(dir.join(format!("{name}.log")))?;
    let err = out.try_clone()?;
    Ok((out, err))
}

/// Pure join used by `cuda_env`: order matches `cuda-env.sh`, empty segments dropped.
fn compose_ld_library_path(
    pytorch_lib: &str,
    qwen_install: &Path,
    cuda_lib: &str,
    extra: &str,
    existing: &str,
) -> String {
    [
        pytorch_lib.to_string(),
        qwen_install.join("libtorch/lib").display().to_string(),
        cuda_lib.to_string(),
        extra.to_string(),
        existing.to_string(),
    ]
    .into_iter()
    .filter(|segment| !segment.is_empty())
    .collect::<Vec<_>>()
    .join(":")
}

/// Port of `scripts/cuda-env.sh`: builds the `LD_LIBRARY_PATH` qwen3-tts needs
/// to load libtorch/CUDA. Honors every `LI_*` override; only walks the venv
/// when `LI_CUDA_LIB_PATH` is not provided.
pub fn cuda_env(config: &DesktopConfig) -> Vec<(String, String)> {
    let install = std::env::var("LI_QWEN_INSTALL_DIR")
        .map(Into::into)
        .unwrap_or_else(|_| config.root.join("vendor/qwen3_tts_rs"));
    let venv = std::env::var("LI_CUDA_WHEEL_VENV")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| config.root.join(".venv-cuda-libs"));

    let pytorch_lib = std::env::var("LI_PYTORCH_LIB_PATH").unwrap_or_else(|_| {
        let candidate = venv.join("lib/python3.12/site-packages/torch/lib");
        if candidate.is_dir() {
            candidate.display().to_string()
        } else {
            String::new()
        }
    });
    let cuda_lib =
        std::env::var("LI_CUDA_LIB_PATH").unwrap_or_else(|_| find_nvidia_libs(&venv).join(":"));
    let extra = std::env::var("LI_EXTRA_CUDA_LIB_PATH").unwrap_or_else(|_| {
        "/home/rgranda/.local/ollama-v0.30.6/lib/ollama/cuda_v12:\
         /home/rgranda/.cache/uv/archive-v0/7fYrxrEsT4mtow-nv-N7X/triton/backends/nvidia/lib/cupti"
            .into()
    });
    let existing = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();

    let joined = compose_ld_library_path(&pytorch_lib, &install, &cuda_lib, &extra, &existing);
    vec![("LD_LIBRARY_PATH".into(), joined)]
}

/// Recursively collect `*/nvidia/*/lib` directories under the CUDA wheel venv.
fn find_nvidia_libs(venv: &Path) -> Vec<String> {
    fn walk(dir: &Path, out: &mut Vec<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path.ends_with("lib")
                    && path
                        .components()
                        .any(|c| c.as_os_str().to_string_lossy() == "nvidia")
                {
                    out.push(path.display().to_string());
                }
                walk(&path, out);
            }
        }
    }
    let mut out = Vec::new();
    if venv.is_dir() {
        walk(venv, &mut out);
    }
    out.sort();
    out
}

use crate::types::{
    AppStatus, ComponentHealth, ComponentState, GpuStatus, RuntimeHealth, derive_node_state,
};

/// Map a (running, healthy) pair to a component lifecycle state.
fn component(running: bool, healthy: bool) -> ComponentHealth {
    let state = if healthy {
        ComponentState::Ready
    } else if running {
        ComponentState::Starting
    } else {
        ComponentState::Stopped
    };
    ComponentHealth {
        state,
        detail: None,
    }
}

/// Pure assembly of the reactive `AppStatus` from gathered signals. Keeps the
/// FSM derivation + per-component health mapping unit-testable without I/O.
#[allow(clippy::too_many_arguments)]
pub fn assemble_app_status(
    live: &Liveness,
    server_healthy: bool,
    qwen_healthy: bool,
    gpu: GpuStatus,
    voice_configured: bool,
    active_connections: usize,
    pipeline_delay_ms: u64,
    last_err: Option<String>,
) -> AppStatus {
    let current_state = derive_node_state(live, server_healthy, &gpu, None, last_err.as_deref());
    let health = RuntimeHealth {
        asr: component(live.server, server_healthy),
        translator: component(live.server, server_healthy),
        voice_renderer: component(live.qwen, qwen_healthy),
        audio_input: component(live.client, live.client),
        audio_output: component(live.client, live.client),
        virtual_mic: component(live.mic, live.mic),
        mesh: ComponentHealth::default(),
    };
    AppStatus {
        current_state,
        gpu,
        voice_configured,
        active_connections,
        health,
        pipeline_delay_ms,
        last_error: last_err,
    }
}

async fn http_healthy(http: &reqwest::Client, url: &str) -> bool {
    http.get(url)
        .send()
        .await
        .map(|response| response.status().is_success())
        .unwrap_or(false)
}

impl LiveRuntime {
    /// Build the reactive `AppStatus` the UI renders: liveness (in-memory) +
    /// health polls + real-time GPU telemetry, projected through the FSM.
    pub async fn app_status(&self, http: &reqwest::Client) -> AppStatus {
        let config = &self.config;
        let live = self.liveness().await;
        let server_healthy = http_healthy(http, &format!("{}/health", config.server_url)).await;
        let qwen_healthy = http_healthy(http, &format!("{}/health", config.qwen_url)).await;
        let snapshot = crate::vram::vram_snapshot().await.ok();
        let gpu = crate::vram::build_gpu_status(snapshot.as_ref(), config.min_server_vram_mb);
        let voice_configured = crate::desktop::voice_profile(config).configured;
        assemble_app_status(
            &live,
            server_healthy,
            qwen_healthy,
            gpu,
            voice_configured,
            live.client as usize,
            0,
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NodeState;

    fn config(root: std::path::PathBuf) -> DesktopConfig {
        DesktopConfig::from_root(root)
    }

    fn gpu(capable: bool) -> crate::types::GpuStatus {
        crate::types::GpuStatus {
            is_capable: capable,
            model_name: "RTX 5060 Ti".into(),
            vram_free_mb: 12_000,
            vram_total_mb: 16_000,
            utilization_pct: 30,
            gate_message: "ok".into(),
            processes: Vec::new(),
            source: "nvml".into(),
        }
    }

    #[test]
    fn assemble_app_status_active_server_maps_component_health() {
        let live = Liveness {
            server: true,
            qwen: true,
            mic: true,
            client: false,
        };
        let status = assemble_app_status(&live, true, true, gpu(true), true, 0, 120, None);
        assert_eq!(status.current_state, NodeState::ActiveServer);
        assert_eq!(status.health.asr.state, ComponentState::Ready);
        assert_eq!(status.health.voice_renderer.state, ComponentState::Ready);
        assert_eq!(status.health.virtual_mic.state, ComponentState::Ready);
        assert_eq!(status.health.audio_input.state, ComponentState::Stopped);
        assert_eq!(status.health.mesh.state, ComponentState::Stopped);
        assert_eq!(status.pipeline_delay_ms, 120);
    }

    #[test]
    fn assemble_app_status_server_up_unhealthy_is_initializing_and_starting() {
        let live = Liveness {
            server: true,
            ..Default::default()
        };
        let status = assemble_app_status(&live, false, false, gpu(true), false, 0, 0, None);
        assert!(matches!(status.current_state, NodeState::Initializing(_)));
        assert_eq!(status.health.asr.state, ComponentState::Starting);
    }

    #[test]
    fn managed_service_names_and_patterns() {
        assert_eq!(ManagedService::QwenTts.name(), "qwen3-tts");
        assert_eq!(ManagedService::Server.name(), "live-interpreter");
        assert_eq!(ManagedService::Mic.name(), "live-interpreter-mic");
        assert_eq!(ManagedService::Client.name(), "live-interpreter-client");
        // Server pattern must not match the client/control binaries.
        assert_eq!(
            ManagedService::Server.pgrep_pattern(),
            "target/release/live-interpreter( |$)"
        );
        assert!(ManagedService::Mic.pgrep_pattern().contains("pw-loopback"));
    }

    #[test]
    fn pid_alive_true_for_self_false_for_bogus() {
        assert!(pid_alive(std::process::id()));
        assert!(!pid_alive(0));
        assert!(!pid_alive(u32::MAX));
    }

    #[test]
    fn compose_ld_library_path_orders_and_filters_empties() {
        let qwen = Path::new("/q");
        let joined = compose_ld_library_path("/p", qwen, "/c", "/e", "");
        assert_eq!(joined, "/p:/q/libtorch/lib:/c:/e");
        // Empty segments (e.g. no torch lib found, no prior LD_LIBRARY_PATH) drop out.
        let sparse = compose_ld_library_path("", qwen, "", "/e", "/prev");
        assert_eq!(sparse, "/q/libtorch/lib:/e:/prev");
    }

    #[tokio::test]
    async fn liveness_reports_owned_alive_and_adopted_dead() {
        let sup = LiveRuntime::new(config(std::env::temp_dir().join("li-sup-test")));
        {
            let mut map = sup.children.lock().await;
            map.insert(
                SERVER,
                ManagedChild {
                    child: None,
                    pid: std::process::id(),
                    adopted: true,
                },
            );
            map.insert(
                QWEN,
                ManagedChild {
                    child: None,
                    pid: u32::MAX,
                    adopted: true,
                },
            );
        }
        let live = sup.liveness().await;
        assert!(live.server, "self pid must be alive");
        assert!(!live.qwen, "bogus pid must be dead");
        assert!(!live.mic, "untracked service is not running");
    }

    #[tokio::test]
    async fn stop_child_drops_adopted_without_killing() {
        let sup = LiveRuntime::new(config(std::env::temp_dir().join("li-sup-test2")));
        {
            let mut map = sup.children.lock().await;
            map.insert(
                SERVER,
                ManagedChild {
                    child: None,
                    pid: std::process::id(), // our own pid; must survive
                    adopted: true,
                },
            );
        }
        let tracked = sup.stop_child(SERVER, Duration::from_millis(50)).await;
        assert!(tracked, "adopted entry was tracked");
        assert!(sup.liveness().await.server == false, "no longer tracked");
        assert!(pid_alive(std::process::id()), "adopted process not killed");
    }
}
