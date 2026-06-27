//! FSM control panel (R8): a thin Axum adapter over `LiveRuntime`. Serves the
//! reactive UI and exposes `/api/status` (the `AppStatus` the UI renders 1:1
//! with `NodeState`) plus role start/stop. No `nvidia-smi`, no `.pid`, no Bash —
//! all status/lifecycle comes from the in-process runtime.
//!
//! ```bash
//! cargo run --bin li-control   # http://127.0.0.1:8799
//! ```

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    response::Html,
    routing::{get, post},
};
use live_interpreter::desktop::{ActionResult, DesktopConfig};
use live_interpreter::runtime::LiveRuntime;
use live_interpreter::types::AppStatus;
use tokio::net::TcpListener;

#[derive(Clone)]
struct App {
    runtime: Arc<LiveRuntime>,
    http: reqwest::Client,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let root = std::env::current_dir()?;
    let runtime = LiveRuntime::new(DesktopConfig::from_root(root));
    let app = App {
        runtime,
        http: reqwest::Client::new(),
    };

    let router = Router::new()
        .route("/", get(index))
        .route("/api/status", get(status))
        .route("/api/server/start", post(server_start))
        .route("/api/server/stop", post(server_stop))
        .route("/api/client/start", post(client_start))
        .route("/api/client/stop", post(client_stop))
        .with_state(app);

    let bind = std::env::var("LI_CONTROL_BIND").unwrap_or_else(|_| "127.0.0.1:8799".into());
    let listener = TcpListener::bind(&bind).await?;
    tracing::info!("control panel on http://{bind}");
    axum::serve(listener, router).await?;
    Ok(())
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../../static/fsm-ui.html"))
}

async fn status(State(app): State<App>) -> Json<AppStatus> {
    Json(app.runtime.app_status(&app.http).await)
}

async fn server_start(State(app): State<App>) -> Json<ActionResult> {
    Json(app.runtime.start_server().await)
}
async fn server_stop(State(app): State<App>) -> Json<ActionResult> {
    Json(app.runtime.stop_server().await)
}
async fn client_start(State(app): State<App>) -> Json<ActionResult> {
    Json(app.runtime.start_client().await)
}
async fn client_stop(State(app): State<App>) -> Json<ActionResult> {
    Json(app.runtime.stop_client().await)
}
