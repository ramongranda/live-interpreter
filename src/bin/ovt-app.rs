use anyhow::Context;
use axum::{
    Json, Router,
    extract::State,
    response::Html,
    routing::{get, post},
};
use serde::Serialize;
use std::{
    env,
    net::SocketAddr,
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::Duration,
};
use tokio::{process::Command, sync::Mutex};
use tracing_subscriber::{EnvFilter, fmt};

#[derive(Clone)]
struct App {
    config: AppConfig,
    lock: Arc<Mutex<()>>,
    http: reqwest::Client,
}

#[derive(Clone, Debug)]
struct AppConfig {
    bind: SocketAddr,
    root: PathBuf,
    stack_bind: String,
    server_url: String,
    qwen_url: String,
    client_bind: String,
    client_url: String,
    play_target: String,
}

impl AppConfig {
    fn from_env() -> anyhow::Result<Self> {
        let root = env::current_dir().context("failed to read current directory")?;
        let bind = env::var("OVT_APP_BIND")
            .unwrap_or_else(|_| "127.0.0.1:8798".into())
            .parse()?;
        let stack_bind = env::var("OVT_STACK_BIND").unwrap_or_else(|_| "0.0.0.0:8787".into());
        let server_url =
            env::var("OVT_SERVER_URL").unwrap_or_else(|_| "http://127.0.0.1:8787".into());
        let qwen_url =
            env::var("OVT_QWEN_TTS_URL").unwrap_or_else(|_| "http://127.0.0.1:8020".into());
        let client_bind = env::var("OVT_CLIENT_BIND").unwrap_or_else(|_| "127.0.0.1:8790".into());
        let client_url = format!("http://{client_bind}");
        let play_target =
            env::var("OVT_CLIENT_PLAY_TARGET").unwrap_or_else(|_| "ovt-teams-mic-sink".into());

        Ok(Self {
            bind,
            root,
            stack_bind,
            server_url,
            qwen_url,
            client_bind,
            client_url,
            play_target,
        })
    }
}

#[derive(Debug, Serialize)]
struct AppStatus {
    server_running: bool,
    qwen_running: bool,
    client_running: bool,
    mic_bridge_running: bool,
    server_health: bool,
    qwen_health: bool,
    client_health: bool,
    gpu_summary: String,
    gpu_processes: Vec<GpuProcess>,
    server_url: String,
    client_url: String,
    role_hint: String,
    last_error: Option<String>,
}

#[derive(Debug, Serialize)]
struct GpuProcess {
    pid: String,
    name: String,
    memory: String,
}

#[derive(Debug, Serialize)]
struct ActionResult {
    ok: bool,
    output: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let config = AppConfig::from_env()?;
    let app = App {
        config,
        lock: Arc::new(Mutex::new(())),
        http: reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()?,
    };

    let router = Router::new()
        .route("/", get(index))
        .route("/api/status", get(api_status))
        .route("/api/server/start", post(api_server_start))
        .route("/api/server/stop", post(api_server_stop))
        .route("/api/client/start", post(api_client_start))
        .route("/api/client/stop", post(api_client_stop))
        .with_state(app.clone());

    tracing::info!("OVT app: http://{}", app.config.bind);
    let listener = tokio::net::TcpListener::bind(app.config.bind).await?;
    axum::serve(listener, router).await?;
    Ok(())
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn api_status(State(app): State<App>) -> Json<AppStatus> {
    Json(status(app).await)
}

async fn api_server_start(State(app): State<App>) -> Json<ActionResult> {
    let _guard = app.lock.lock().await;
    Json(
        run_script(
            &app,
            "start-local-stack.sh",
            &[("OVT_BIND", &app.config.stack_bind)],
        )
        .await,
    )
}

async fn api_server_stop(State(app): State<App>) -> Json<ActionResult> {
    let _guard = app.lock.lock().await;
    Json(run_script(&app, "stop-local-stack.sh", &[]).await)
}

async fn api_client_start(State(app): State<App>) -> Json<ActionResult> {
    let _guard = app.lock.lock().await;
    if pid_alive(&app.config.root.join("data/logs/ovt-app-client.pid")) {
        return Json(ActionResult {
            ok: true,
            output: "Cliente ya esta arrancado".into(),
        });
    }

    let bin = app.config.root.join("target/release/ovt-meeting-client");
    let log = app.config.root.join("data/logs/ovt-app-client.log");
    let pid = app.config.root.join("data/logs/ovt-app-client.pid");
    if let Err(error) = tokio::fs::create_dir_all(app.config.root.join("data/logs")).await {
        return Json(ActionResult {
            ok: false,
            output: error.to_string(),
        });
    }

    let log_file = match std::fs::File::create(&log) {
        Ok(file) => file,
        Err(error) => {
            return Json(ActionResult {
                ok: false,
                output: error.to_string(),
            });
        }
    };
    let log_file_err = match log_file.try_clone() {
        Ok(file) => file,
        Err(error) => {
            return Json(ActionResult {
                ok: false,
                output: error.to_string(),
            });
        }
    };

    match Command::new(bin)
        .current_dir(&app.config.root)
        .env("OVT_SERVER_URL", &app.config.server_url)
        .env("OVT_CLIENT_BIND", &app.config.client_bind)
        .env("OVT_CLIENT_PLAY_TARGET", &app.config.play_target)
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err))
        .spawn()
    {
        Ok(child) => {
            let child_pid = child.id().unwrap_or_default();
            let _ = tokio::fs::write(pid, child_pid.to_string()).await;
            Json(ActionResult {
                ok: true,
                output: format!("Cliente arrancado en {}", app.config.client_url),
            })
        }
        Err(error) => Json(ActionResult {
            ok: false,
            output: error.to_string(),
        }),
    }
}

async fn api_client_stop(State(app): State<App>) -> Json<ActionResult> {
    let _guard = app.lock.lock().await;
    let pid_path = app.config.root.join("data/logs/ovt-app-client.pid");
    let Ok(pid) = tokio::fs::read_to_string(&pid_path).await else {
        return Json(ActionResult {
            ok: true,
            output: "Cliente no estaba arrancado".into(),
        });
    };

    let output = Command::new("kill")
        .arg(pid.trim())
        .output()
        .await
        .map(|output| {
            let mut text = String::from_utf8_lossy(&output.stdout).to_string();
            text.push_str(&String::from_utf8_lossy(&output.stderr));
            ActionResult {
                ok: output.status.success(),
                output: if text.trim().is_empty() {
                    "Cliente parado".into()
                } else {
                    text
                },
            }
        })
        .unwrap_or_else(|error| ActionResult {
            ok: false,
            output: error.to_string(),
        });
    let _ = tokio::fs::remove_file(pid_path).await;
    Json(output)
}

async fn status(app: App) -> AppStatus {
    let server_health = health(&app.http, &format!("{}/health", app.config.server_url)).await;
    let qwen_health = health(&app.http, &format!("{}/health", app.config.qwen_url)).await;
    let client_health = health(&app.http, &format!("{}/api/state", app.config.client_url)).await;
    let (gpu_processes, gpu_summary, gpu_error) = gpu_status()
        .await
        .unwrap_or_else(|error| (Vec::new(), String::new(), Some(error.to_string())));
    let server_running = pid_alive(
        &app.config
            .root
            .join("data/logs/olares-voice-translator.pid"),
    );
    let client_running = pid_alive(&app.config.root.join("data/logs/ovt-app-client.pid"));

    let role_hint = if server_running && !client_running {
        "Modo servidor GPU".into()
    } else if client_running && !server_running {
        "Modo cliente Teams".into()
    } else if server_running && client_running {
        "Servidor y cliente activos".into()
    } else {
        "Selecciona servidor o cliente".into()
    };

    AppStatus {
        server_running,
        qwen_running: pid_alive(&app.config.root.join("data/logs/qwen3-tts.pid")),
        client_running,
        mic_bridge_running: pid_alive(&app.config.root.join("data/logs/ovt-teams-mic.pid")),
        server_health,
        qwen_health,
        client_health,
        gpu_summary,
        gpu_processes,
        server_url: app.config.server_url,
        client_url: app.config.client_url,
        role_hint,
        last_error: gpu_error,
    }
}

async fn health(http: &reqwest::Client, url: &str) -> bool {
    http.get(url)
        .send()
        .await
        .map(|response| response.status().is_success())
        .unwrap_or(false)
}

fn pid_alive(path: &Path) -> bool {
    let Ok(pid) = std::fs::read_to_string(path) else {
        return false;
    };
    let Ok(pid) = pid.trim().parse::<u32>() else {
        return false;
    };
    PathBuf::from(format!("/proc/{pid}")).exists()
}

async fn run_script(app: &App, script: &str, envs: &[(&str, &str)]) -> ActionResult {
    let mut command = Command::new("bash");
    command
        .arg(app.config.root.join("scripts").join(script))
        .current_dir(&app.config.root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (key, value) in envs {
        command.env(key, value);
    }

    match command.output().await {
        Ok(output) => {
            let mut text = String::new();
            text.push_str(&String::from_utf8_lossy(&output.stdout));
            text.push_str(&String::from_utf8_lossy(&output.stderr));
            ActionResult {
                ok: output.status.success(),
                output: text,
            }
        }
        Err(error) => ActionResult {
            ok: false,
            output: error.to_string(),
        },
    }
}

async fn gpu_status() -> anyhow::Result<(Vec<GpuProcess>, String, Option<String>)> {
    let summary = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader",
        ])
        .output()
        .await
        .context("failed to run nvidia-smi summary")?;
    let processes = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ])
        .output()
        .await
        .context("failed to run nvidia-smi processes")?;

    let summary_text = String::from_utf8_lossy(&summary.stdout).trim().to_string();
    let processes_text = String::from_utf8_lossy(&processes.stdout);
    let gpu_processes = processes_text
        .lines()
        .filter_map(|line| {
            let mut parts = line.splitn(3, ',').map(str::trim);
            Some(GpuProcess {
                pid: parts.next()?.into(),
                name: parts.next()?.into(),
                memory: parts.next()?.into(),
            })
        })
        .collect();

    Ok((gpu_processes, summary_text, None))
}

const INDEX_HTML: &str = r##"<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OVT</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; background:#101214; color:#eef1f4; }
    body { margin:0; }
    main { max-width:1120px; margin:0 auto; padding:24px; }
    header { display:flex; justify-content:space-between; gap:16px; align-items:center; margin-bottom:20px; }
    h1 { font-size:26px; margin:0; }
    .hint { color:#aab3bd; font-size:15px; }
    .modes { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:16px; }
    .mode { background:#171b1f; border:1px solid #2d343b; border-radius:8px; padding:18px; }
    .mode h2 { margin:0 0 8px; font-size:18px; }
    .mode p { color:#aab3bd; min-height:44px; }
    button, a.button { display:inline-block; border:1px solid #3a444e; background:#242b32; color:#f4f7fa; padding:10px 14px; border-radius:6px; cursor:pointer; text-decoration:none; margin-right:8px; }
    .start { background:#1f4635; border-color:#2e7d57; }
    .stop { background:#63302d; border-color:#994a42; }
    .grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-bottom:16px; }
    .card { background:#171b1f; border:1px solid #2d343b; border-radius:8px; padding:14px; }
    .card span { display:block; color:#aab3bd; font-size:12px; margin-bottom:5px; }
    .card strong { font-size:18px; overflow-wrap:anywhere; }
    .ok { color:#69d391; }
    .bad { color:#ff8f87; }
    pre { white-space:pre-wrap; background:#171b1f; border:1px solid #2d343b; border-radius:8px; padding:14px; min-height:100px; }
    table { width:100%; border-collapse:collapse; background:#171b1f; border:1px solid #2d343b; border-radius:8px; overflow:hidden; }
    th, td { text-align:left; padding:10px; border-bottom:1px solid #2d343b; }
    th { color:#aab3bd; font-weight:500; }
    @media (max-width:900px) { .modes, .grid { grid-template-columns:1fr; } header { flex-direction:column; align-items:flex-start; } }
  </style>
</head>
<body>
<main>
  <header>
    <div><h1>OVT</h1><div class="hint" id="role"></div></div>
    <div class="hint">Una app: servidor GPU o cliente Teams</div>
  </header>
  <div class="modes">
    <section class="mode">
      <h2>Servidor GPU</h2>
      <p>Carga Whisper, Qwen TTS y el puente de microfono. Paralo para liberar VRAM.</p>
      <button class="start" onclick="act('/api/server/start')">Arrancar servidor</button>
      <button class="stop" onclick="act('/api/server/stop')">Parar servidor</button>
    </section>
    <section class="mode">
      <h2>Cliente Teams</h2>
      <p>Captura tu micro, envia frases al servidor y saca la voz traducida al micro virtual.</p>
      <button class="start" onclick="act('/api/client/start')">Arrancar cliente</button>
      <button class="stop" onclick="act('/api/client/stop')">Parar cliente</button>
      <a class="button" id="clientLink" href="#" target="_blank">Abrir controles</a>
    </section>
  </div>
  <div class="grid">
    <div class="card"><span>Servidor</span><strong id="server"></strong></div>
    <div class="card"><span>Qwen</span><strong id="qwen"></strong></div>
    <div class="card"><span>Cliente</span><strong id="client"></strong></div>
    <div class="card"><span>Mic bridge</span><strong id="mic"></strong></div>
    <div class="card"><span>Health servidor</span><strong id="serverHealth"></strong></div>
    <div class="card"><span>Health Qwen</span><strong id="qwenHealth"></strong></div>
    <div class="card"><span>Health cliente</span><strong id="clientHealth"></strong></div>
    <div class="card"><span>GPU</span><strong id="gpu"></strong></div>
  </div>
  <table>
    <thead><tr><th>PID</th><th>Proceso</th><th>VRAM</th></tr></thead>
    <tbody id="processes"></tbody>
  </table>
  <h2>Salida</h2>
  <pre id="output"></pre>
</main>
<script>
function mark(value) { return value ? '<span class="ok">ON</span>' : '<span class="bad">OFF</span>'; }
async function act(url) {
  document.getElementById('output').textContent = 'Ejecutando...';
  const r = await fetch(url, {method:'POST'}).then(r => r.json());
  document.getElementById('output').textContent = r.output || (r.ok ? 'OK' : 'Error');
  await tick();
}
async function tick() {
  const s = await fetch('/api/status').then(r => r.json());
  document.getElementById('role').textContent = s.role_hint;
  document.getElementById('server').innerHTML = mark(s.server_running);
  document.getElementById('qwen').innerHTML = mark(s.qwen_running);
  document.getElementById('client').innerHTML = mark(s.client_running);
  document.getElementById('mic').innerHTML = mark(s.mic_bridge_running);
  document.getElementById('serverHealth').innerHTML = mark(s.server_health);
  document.getElementById('qwenHealth').innerHTML = mark(s.qwen_health);
  document.getElementById('clientHealth').innerHTML = mark(s.client_health);
  document.getElementById('gpu').textContent = s.gpu_summary || 'sin datos';
  document.getElementById('clientLink').href = s.client_url;
  document.getElementById('processes').innerHTML = s.gpu_processes.map(p =>
    `<tr><td>${p.pid}</td><td>${p.name}</td><td>${p.memory}</td></tr>`).join('');
  if (s.last_error) document.getElementById('output').textContent = s.last_error;
}
setInterval(tick, 2000); tick();
</script>
</body>
</html>
"##;
