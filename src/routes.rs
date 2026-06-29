use anyhow::Context;
use axum::{
    Json, Router,
    body::Body,
    extract::{
        DefaultBodyLimit, Multipart, Path, Query, State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    http::{HeaderMap, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use chrono::Utc;
use futures_util::{SinkExt, StreamExt};
use live_interpreter::events::EventHub;
use live_interpreter::translate::Translator;
use live_interpreter::types::{
    Direction, HealthResponse, InterpretResponse, Lane, PipelineEvent, SessionStart,
    TextInterpretRequest,
};
use live_interpreter::{asr::AsrEngine, config::Config, tts::TtsEngine};
use std::{collections::HashMap, path::Path as FsPath, sync::Arc, time::Instant};
use thiserror::Error;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use uuid::Uuid;

#[derive(Clone)]
pub struct AppState {
    pub config: Config,
    pub asr: Arc<AsrEngine>,
    pub translator: Translator,
    pub tts: TtsEngine,
}

pub fn app(state: AppState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/v1/audio/{filename}", get(download_audio))
        .route("/v1/interpret/text", post(interpret_text))
        .route("/v1/interpret/file", post(interpret_file))
        .route("/v1/stream/meeting", get(meeting_stream))
        .layer(DefaultBodyLimit::max(200 * 1024 * 1024))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        ok: true,
        whisper_model_exists: state.config.whisper_model.exists(),
        ollama_url: state.config.ollama_url,
        ollama_model: state.config.ollama_model,
        qwen_tts_url: state.config.qwen_tts_url,
        qwen_tts_model: state.config.qwen_tts_model,
        qwen_tts_voice: state.config.qwen_tts_voice,
    })
}

async fn interpret_text(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<TextInterpretRequest>,
) -> Result<Json<InterpretResponse>, AppError> {
    require_auth(&state, &headers, None)?;
    let id = Uuid::new_v4();
    let translation = state
        .translator
        .translate(&request.text, &request.direction)
        .await?;
    let audio_path = if request.synthesize {
        Some(
            state
                .tts
                .synthesize(id, &translation, &request.direction)
                .await?,
        )
    } else {
        None
    };

    Ok(Json(InterpretResponse {
        id,
        created_at: Utc::now(),
        direction: request.direction,
        transcript: request.text,
        translation,
        audio_url: audio_path.as_deref().and_then(audio_url),
        audio_path,
    }))
}

async fn interpret_file(
    State(state): State<AppState>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Result<Json<InterpretResponse>, AppError> {
    require_auth(&state, &headers, None)?;
    let id = Uuid::new_v4();
    let mut direction = Direction::EsToEn;
    let mut synthesize = false;
    let mut audio_path = None;

    while let Some(field) = multipart.next_field().await.context("invalid multipart")? {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "direction" => {
                let value = field.text().await.context("invalid direction field")?;
                direction = match value.as_str() {
                    "es_to_en" => Direction::EsToEn,
                    "en_to_es" => Direction::EnToEs,
                    _ => return Err(AppError::BadRequest("invalid direction".into())),
                };
            }
            "synthesize" => {
                let value = field.text().await.context("invalid synthesize field")?;
                synthesize = matches!(value.as_str(), "1" | "true" | "yes");
            }
            "audio" => {
                let filename = field
                    .file_name()
                    .map(safe_filename)
                    .unwrap_or_else(|| format!("{id}.wav"));
                let bytes = field.bytes().await.context("failed to read audio field")?;
                let uploads = state.config.data_dir.join("uploads");
                tokio::fs::create_dir_all(&uploads)
                    .await
                    .context("failed to create upload directory")?;
                let path = uploads.join(format!("{id}-{filename}"));
                tokio::fs::write(&path, bytes)
                    .await
                    .context("failed to write uploaded audio")?;
                audio_path = Some(path);
            }
            _ => {}
        }
    }

    let audio_path =
        audio_path.ok_or_else(|| AppError::BadRequest("missing audio field".into()))?;
    let response =
        process_audio_path(state.clone(), id, &audio_path, direction, synthesize).await?;
    persist_response(&state.config.data_dir, &response).await?;
    Ok(Json(response))
}

async fn meeting_stream(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<HashMap<String, String>>,
    upgrade: WebSocketUpgrade,
) -> impl IntoResponse {
    if let Err(error) = require_auth(&state, &headers, query.get("token").map(String::as_str)) {
        return error.into_response();
    }
    upgrade.on_upgrade(move |socket| handle_meeting_stream(socket, state))
}

async fn handle_meeting_stream(socket: WebSocket, state: AppState) {
    let (mut sink, mut receiver) = socket.split();

    // Symmetric `PipelineEvent` contract: every event is fanned out by the
    // session `EventHub` and forwarded to the socket as one bincode frame
    // (the `AudioFrame` carries its PCM inline — no separate binary frame).
    let hub = EventHub::new(Uuid::new_v4(), 64);
    let mut rx = hub.subscribe();
    let forward = tokio::spawn(async move {
        while let Ok(envelope) = rx.recv().await {
            let Ok(bytes) = bincode::serialize(&envelope) else {
                break;
            };
            if sink.send(Message::Binary(bytes.into())).await.is_err() {
                break;
            }
        }
    });

    hub.publish(PipelineEvent::Ready, now_ms());

    let mut session = SessionStart {
        direction: Direction::EsToEn,
        synthesize: true,
    };

    while let Some(message) = receiver.next().await {
        let message = match message {
            Ok(message) => message,
            Err(error) => {
                hub.publish(
                    PipelineEvent::Error {
                        message: error.to_string(),
                    },
                    now_ms(),
                );
                break;
            }
        };

        match message {
            Message::Text(text) => match serde_json::from_str::<SessionStart>(&text) {
                Ok(value) => {
                    session = value;
                    hub.publish(PipelineEvent::Listening { lane: Lane::Local }, now_ms());
                }
                Err(error) => {
                    hub.publish(
                        PipelineEvent::Error {
                            message: format!("invalid session start message: {error}"),
                        },
                        now_ms(),
                    );
                }
            },
            Message::Binary(bytes) => {
                let id = Uuid::new_v4();
                let started = Instant::now();
                hub.publish(
                    PipelineEvent::Processing {
                        id,
                        lane: Lane::Local,
                    },
                    now_ms(),
                );

                match process_stream_audio(state.clone(), id, &bytes, &session).await {
                    Ok(response) => {
                        hub.publish(
                            PipelineEvent::Transcript {
                                id,
                                lane: Lane::Local,
                                lang: session.direction.source(),
                                text: response.transcript.clone(),
                            },
                            now_ms(),
                        );
                        hub.publish(
                            PipelineEvent::Translation {
                                id,
                                lane: Lane::Local,
                                lang: session.direction.target(),
                                text: response.translation.clone(),
                            },
                            now_ms(),
                        );
                        if let Some(audio_path) = response.audio_path.as_ref() {
                            match tokio::fs::read(audio_path).await {
                                Ok(wav) => {
                                    match live_interpreter::voice::wav_to_audio_frame(&wav) {
                                        Ok(frame) => hub.publish(
                                            PipelineEvent::AudioFrame {
                                                id,
                                                lane: Lane::Local,
                                                spec: frame.spec,
                                                pcm: frame.pcm,
                                            },
                                            now_ms(),
                                        ),
                                        Err(error) => hub.publish(
                                            PipelineEvent::Error {
                                                message: error.to_string(),
                                            },
                                            now_ms(),
                                        ),
                                    }
                                }
                                Err(error) => hub.publish(
                                    PipelineEvent::Error {
                                        message: error.to_string(),
                                    },
                                    now_ms(),
                                ),
                            };
                        }
                        let latency_ms = started.elapsed().as_millis() as u64;
                        hub.publish(
                            PipelineEvent::Done {
                                id,
                                lane: Lane::Local,
                                latency_ms,
                            },
                            now_ms(),
                        );
                    }
                    Err(error) => {
                        hub.publish(
                            PipelineEvent::Error {
                                message: error.to_string(),
                            },
                            now_ms(),
                        );
                    }
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    // Dropping the hub closes the broadcast channel and ends the forward task.
    drop(hub);
    let _ = forward.await;
}

/// Wall-clock milliseconds since the Unix epoch, for stamping event envelopes.
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

async fn process_stream_audio(
    state: AppState,
    id: Uuid,
    audio: &[u8],
    start: &SessionStart,
) -> anyhow::Result<InterpretResponse> {
    let uploads = state.config.data_dir.join("uploads");
    tokio::fs::create_dir_all(&uploads)
        .await
        .context("failed to create upload directory")?;
    let audio_path = uploads.join(format!("{id}-stream.wav"));
    tokio::fs::write(&audio_path, audio)
        .await
        .context("failed to write stream audio")?;

    let response = process_audio_path(
        state.clone(),
        id,
        &audio_path,
        start.direction,
        start.synthesize,
    )
    .await?;
    persist_response(&state.config.data_dir, &response).await?;
    Ok(response)
}

async fn process_audio_path(
    state: AppState,
    id: Uuid,
    audio_path: &FsPath,
    direction: Direction,
    synthesize: bool,
) -> anyhow::Result<InterpretResponse> {
    let segments = state
        .asr
        .transcribe_file(audio_path, direction.source_lang())
        .await?;
    let transcript = segments
        .iter()
        .map(|segment| segment.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let translation = state.translator.translate(&transcript, &direction).await?;
    let generated_audio = if synthesize {
        Some(state.tts.synthesize(id, &translation, &direction).await?)
    } else {
        None
    };

    Ok(InterpretResponse {
        id,
        created_at: Utc::now(),
        direction,
        transcript,
        translation,
        audio_url: generated_audio.as_deref().and_then(audio_url),
        audio_path: generated_audio,
    })
}

async fn download_audio(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<HashMap<String, String>>,
    Path(filename): Path<String>,
) -> Result<Response, AppError> {
    require_auth(&state, &headers, query.get("token").map(String::as_str))?;
    let filename = safe_filename(&filename);
    if !filename.ends_with(".wav") {
        return Err(AppError::BadRequest(
            "only wav audio can be downloaded".into(),
        ));
    }

    let path = state.config.data_dir.join("tts").join(filename);
    let bytes = tokio::fs::read(path)
        .await
        .context("failed to read generated audio")?;

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "audio/wav")
        .body(Body::from(bytes))
        .context("failed to build audio response")
        .map_err(AppError::Internal)
}

fn require_auth(
    state: &AppState,
    headers: &HeaderMap,
    query_token: Option<&str>,
) -> Result<(), AppError> {
    let Some(expected) = state.config.auth_token.as_deref() else {
        return Ok(());
    };

    let header_token = headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));

    if header_token == Some(expected) || query_token == Some(expected) {
        Ok(())
    } else {
        Err(AppError::Unauthorized)
    }
}

async fn persist_response(data_dir: &FsPath, response: &InterpretResponse) -> anyhow::Result<()> {
    let dir = data_dir.join("transcripts");
    tokio::fs::create_dir_all(&dir).await?;
    let path = dir.join(format!("{}.json", response.id));
    let body = serde_json::to_vec_pretty(response)?;
    tokio::fs::write(path, body).await?;
    Ok(())
}

pub(crate) fn safe_filename(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn audio_url(path: &FsPath) -> Option<String> {
    let filename = path.file_name()?.to_str()?;
    Some(format!("/v1/audio/{}", safe_filename(filename)))
}

#[cfg(test)]
mod tests {
    use super::safe_filename;

    #[test]
    fn safe_filename_keeps_simple_ascii_names() {
        assert_eq!(
            safe_filename("meeting-audio_01.wav"),
            "meeting-audio_01.wav"
        );
    }

    #[test]
    fn safe_filename_replaces_path_and_shell_chars() {
        assert_eq!(
            safe_filename("../../bad name;rm.wav"),
            ".._.._bad_name_rm.wav"
        );
    }
}

#[derive(Debug, Error)]
pub enum AppError {
    #[error("unauthorized")]
    Unauthorized,
    #[error("{0}")]
    BadRequest(String),
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = match self {
            AppError::Unauthorized => StatusCode::UNAUTHORIZED,
            AppError::BadRequest(_) => StatusCode::BAD_REQUEST,
            AppError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        let body = Json(serde_json::json!({ "error": self.to_string() }));
        (status, body).into_response()
    }
}
