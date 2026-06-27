mod routes;

use crate::routes::{AppState, app};
use live_interpreter::{asr::AsrEngine, config::Config, translate::Translator, tts::TtsEngine};
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{EnvFilter, fmt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let config = Config::from_env()?;
    tokio::fs::create_dir_all(&config.data_dir).await?;

    let asr = Arc::new(AsrEngine::load(&config)?);
    let translator = Translator::from_env(config.ollama_url.clone(), config.ollama_model.clone())?;
    let tts = TtsEngine::new(&config).await?;

    let state = AppState {
        config: config.clone(),
        asr,
        translator,
        tts,
    };

    let listener = TcpListener::bind(config.bind).await?;
    tracing::info!("listening on http://{}", config.bind);
    axum::serve(listener, app(state).layer(TraceLayer::new_for_http())).await?;
    Ok(())
}
