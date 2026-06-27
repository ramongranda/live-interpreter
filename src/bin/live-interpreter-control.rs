use axum::{
    Json, Router,
    extract::State,
    response::Html,
    routing::{get, post},
};
use live_interpreter::desktop::{
    ActionResult, AppStatus, DesktopConfig, collect_status, start_client, start_server,
    stop_client, stop_server,
};
use std::{env, net::SocketAddr, sync::Arc, time::Duration};
use tokio::sync::Mutex;
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
    desktop: DesktopConfig,
}

impl AppConfig {
    fn from_env() -> anyhow::Result<Self> {
        let root = env::current_dir()?;
        let bind = env::var("LI_APP_BIND")
            .unwrap_or_else(|_| "127.0.0.1:8798".into())
            .parse()?;
        Ok(Self {
            bind,
            desktop: DesktopConfig::from_root(root),
        })
    }
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

    tracing::info!("Live Interpreter control panel: http://{}", app.config.bind);
    let listener = tokio::net::TcpListener::bind(app.config.bind).await?;
    axum::serve(listener, router).await?;
    Ok(())
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn api_status(State(app): State<App>) -> Json<AppStatus> {
    Json(collect_status(&app.config.desktop, &app.http).await)
}

async fn api_server_start(State(app): State<App>) -> Json<ActionResult> {
    let _guard = app.lock.lock().await;
    Json(start_server(&app.config.desktop).await)
}

async fn api_server_stop(State(app): State<App>) -> Json<ActionResult> {
    let _guard = app.lock.lock().await;
    Json(stop_server(&app.config.desktop).await)
}

async fn api_client_start(State(app): State<App>) -> Json<ActionResult> {
    let _guard = app.lock.lock().await;
    Json(start_client(&app.config.desktop).await)
}

async fn api_client_stop(State(app): State<App>) -> Json<ActionResult> {
    let _guard = app.lock.lock().await;
    Json(stop_client(&app.config.desktop).await)
}

const INDEX_HTML: &str = r##"<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Live Interpreter Control</title>
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
    button:disabled { opacity:.45; cursor:not-allowed; }
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
    <div><h1>Live Interpreter Control</h1><div class="hint" id="role"></div></div>
    <div class="hint">Una app: servidor GPU o cliente de llamadas</div>
  </header>
  <div class="modes">
    <section class="mode">
      <h2>Servidor GPU</h2>
      <p>Carga Whisper, Qwen TTS y el puente de microfono. Paralo para liberar VRAM.</p>
      <button class="start" id="serverStart" onclick="act('/api/server/start')">Arrancar servidor</button>
      <button class="stop" onclick="act('/api/server/stop')">Parar servidor</button>
    </section>
    <section class="mode">
      <h2>Cliente de llamadas</h2>
      <p>Captura tu micro, envia frases al servidor y saca la voz traducida al micro virtual para cualquier app.</p>
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
    <div class="card"><span>Preflight servidor</span><strong id="gpuGate"></strong></div>
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
  document.getElementById('gpuGate').textContent = s.gpu_gate;
  document.getElementById('serverStart').disabled = !s.gpu_ready;
  document.getElementById('serverStart').title = s.gpu_gate;
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
