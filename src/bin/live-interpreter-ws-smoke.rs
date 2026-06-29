use anyhow::{Context, bail};
use futures_util::{SinkExt, StreamExt};
use live_interpreter::types::{Direction, EventEnvelope, PipelineEvent, SessionStart};
use std::{env, path::PathBuf};
use tokio_tungstenite::{connect_async, tungstenite::Message};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let server_url = env::var("LI_SERVER_URL").unwrap_or_else(|_| "http://127.0.0.1:8787".into());
    let audio_path = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/live-interpreter-gpu-tts.wav"));
    let direction = match env::var("LI_CLIENT_DIRECTION")
        .unwrap_or_else(|_| "en_to_es".into())
        .as_str()
    {
        "es_to_en" => Direction::EsToEn,
        _ => Direction::EnToEs,
    };
    let token = env::var("LI_AUTH_TOKEN").ok();
    let url = websocket_url(&server_url, token.as_deref());
    let audio = tokio::fs::read(&audio_path)
        .await
        .with_context(|| format!("failed to read {}", audio_path.display()))?;

    let (mut socket, _) = connect_async(&url)
        .await
        .with_context(|| format!("failed to connect {url}"))?;
    socket
        .send(Message::Text(
            serde_json::to_string(&SessionStart {
                direction,
                synthesize: true,
            })?
            .into(),
        ))
        .await?;
    socket.send(Message::Binary(audio.into())).await?;

    let mut saw_done = false;
    while let Some(message) = socket.next().await {
        match message? {
            Message::Binary(frame) => {
                let envelope: EventEnvelope =
                    bincode::deserialize(&frame).context("failed to decode pipeline event")?;
                match &envelope.event {
                    PipelineEvent::AudioFrame { pcm, spec, .. } => {
                        println!(
                            "audio_frame pcm_bytes={} rate={}",
                            pcm.len(),
                            spec.sample_rate
                        );
                    }
                    PipelineEvent::Done { latency_ms, .. } => {
                        println!("done latency_ms={latency_ms}");
                        saw_done = true;
                        break;
                    }
                    PipelineEvent::Error { message } => bail!("{message}"),
                    other => println!("{other:?}"),
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    if !saw_done {
        bail!("websocket stream closed before done event");
    }
    Ok(())
}

fn websocket_url(server_url: &str, auth_token: Option<&str>) -> String {
    let base = server_url.trim_end_matches('/');
    let mut url = if let Some(rest) = base.strip_prefix("https://") {
        format!("wss://{rest}/v1/stream/meeting")
    } else if let Some(rest) = base.strip_prefix("http://") {
        format!("ws://{rest}/v1/stream/meeting")
    } else {
        format!("ws://{base}/v1/stream/meeting")
    };
    if let Some(token) = auth_token {
        url.push_str("?token=");
        url.push_str(token);
    }
    url
}
