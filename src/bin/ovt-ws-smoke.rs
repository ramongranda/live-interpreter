use anyhow::{Context, bail};
use futures_util::{SinkExt, StreamExt};
use serde::Serialize;
use std::{env, path::PathBuf};
use tokio_tungstenite::{connect_async, tungstenite::Message};

#[derive(Serialize)]
struct StreamStart<'a> {
    direction: &'a str,
    synthesize: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let server_url = env::var("OVT_SERVER_URL").unwrap_or_else(|_| "http://127.0.0.1:8787".into());
    let audio_path = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/ovt-gpu-tts.wav"));
    let direction = env::var("OVT_CLIENT_DIRECTION").unwrap_or_else(|_| "en_to_es".into());
    let token = env::var("OVT_AUTH_TOKEN").ok();
    let url = websocket_url(&server_url, token.as_deref());
    let audio = tokio::fs::read(&audio_path)
        .await
        .with_context(|| format!("failed to read {}", audio_path.display()))?;

    let (mut socket, _) = connect_async(&url)
        .await
        .with_context(|| format!("failed to connect {url}"))?;
    socket
        .send(Message::Text(
            serde_json::to_string(&StreamStart {
                direction: &direction,
                synthesize: true,
            })?
            .into(),
        ))
        .await?;
    socket.send(Message::Binary(audio.into())).await?;

    let mut saw_done = false;
    while let Some(message) = socket.next().await {
        match message? {
            Message::Text(text) => {
                println!("{text}");
                if text.contains(r#""event":"done""#) {
                    saw_done = true;
                    break;
                }
                if text.contains(r#""event":"error""#) {
                    bail!("{text}");
                }
            }
            Message::Binary(audio) => {
                println!("binary_audio_bytes={}", audio.len());
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
