use crate::vram::GpuProcessMem;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    EsToEn,
    EnToEs,
}

impl Direction {
    pub fn source_lang(&self) -> &'static str {
        match self {
            Direction::EsToEn => "es",
            Direction::EnToEs => "en",
        }
    }

    pub fn target_lang_name(&self) -> &'static str {
        match self {
            Direction::EsToEn => "English",
            Direction::EnToEs => "Spanish",
        }
    }

    pub fn target_tts_language(&self) -> &'static str {
        match self {
            Direction::EsToEn => "english",
            Direction::EnToEs => "spanish",
        }
    }

    /// Source language as a `Lang` tag (for pipeline events).
    pub fn source(&self) -> Lang {
        match self {
            Direction::EsToEn => Lang::Es,
            Direction::EnToEs => Lang::En,
        }
    }

    /// Target language as a `Lang` tag (for pipeline events).
    pub fn target(&self) -> Lang {
        match self {
            Direction::EsToEn => Lang::En,
            Direction::EnToEs => Lang::Es,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct TextInterpretRequest {
    pub text: String,
    pub direction: Direction,
    #[serde(default)]
    pub synthesize: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct InterpretResponse {
    pub id: Uuid,
    pub created_at: DateTime<Utc>,
    pub direction: Direction,
    pub transcript: String,
    pub translation: String,
    pub audio_path: Option<PathBuf>,
    pub audio_url: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct HealthResponse {
    pub ok: bool,
    pub whisper_model_exists: bool,
    pub ollama_url: String,
    pub ollama_model: String,
    pub qwen_tts_url: String,
    pub qwen_tts_model: String,
    pub qwen_tts_voice: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Segment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}

fn default_stream_synthesize() -> bool {
    true
}

// ---------------------------------------------------------------------------
// Unified state & event contracts (FSM + symmetric PipelineEvent).
//
// IMPORTANT (bincode): all enums below are **externally tagged** (no
// `#[serde(tag = ...)]`). bincode is not self-describing and cannot decode
// internally/adjacently-tagged enum representations, and `PipelineEvent`,
// `NodeState`, `InitStatus`, etc. travel over bincode (binary WS / libp2p
// request-response). External tagging keeps them bincode-roundtrippable while
// still serializing cleanly to JSON for the HTTP control panel.
// ---------------------------------------------------------------------------

/// Source/target language tag carried by pipeline events.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Lang {
    Es,
    En,
}

impl Lang {
    /// Qwen3-TTS `language` field value for synthesizing in this language.
    pub fn tts_language(self) -> &'static str {
        match self {
            Lang::Es => "spanish",
            Lang::En => "english",
        }
    }
}

/// Which side of a bidirectional call a fragment belongs to. Lets both peers
/// render the two console columns in parallel from one event stream.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Lane {
    /// This node's own voice (outgoing): transcript in source lang + clone translation.
    Local,
    /// The remote peer (incoming): their transcript + the translation we hear.
    Remote,
}

/// Sample format of a raw audio buffer.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AudioFormat {
    PcmS16Le,
    PcmF32Le,
}

/// Describes a raw audio buffer so Candle/PipeWire/resample stages agree on
/// layout without guessing. Travels with every `AudioFrame`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AudioSpec {
    pub sample_rate: u32,
    pub channels: u8,
    pub format: AudioFormat,
}

impl AudioSpec {
    /// The pipeline default: 24 kHz mono s16le (Qwen3-TTS output rate).
    pub fn mono_s16le(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            channels: 1,
            format: AudioFormat::PcmS16Le,
        }
    }
}

/// Status of a single model-loading step shown on the Initializing screen.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InitStatus {
    Pending,
    Running(u8),
    Ok,
    Failed(String),
}

/// One row of the Initializing progress list (VRAM / Whisper / Ollama …).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct InitStep {
    pub label: String,
    pub status: InitStatus,
    pub elapsed_ms: u64,
}

/// Derived node lifecycle state. Never stored as truth — computed each tick by
/// `derive_node_state` from liveness + health + GPU preflight. Replaces the old
/// `role_hint` string and drives the UI 1:1.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum NodeState {
    Idle,
    Preflight,
    Initializing(Vec<InitStep>),
    ActiveServer,
    ActiveClient,
    Error(String),
}

/// In-memory liveness of the supervised children (no `/proc`, no `.pid`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Liveness {
    pub server: bool,
    pub client: bool,
    pub qwen: bool,
    pub mic: bool,
}

/// Structured GPU telemetry (NVML in-memory via `vram.rs`). Replaces the old
/// `gpu_summary` String + `gpu_ready` bool + `gpu_gate` String triplet.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GpuStatus {
    pub is_capable: bool,
    pub model_name: String,
    pub vram_free_mb: u64,
    pub vram_total_mb: u64,
    pub utilization_pct: u8,
    pub gate_message: String,
    pub processes: Vec<GpuProcessMem>,
    pub source: String,
}

/// Lifecycle of one in-process runtime component (Candle-native: these are
/// Tokio tasks, not external services).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ComponentState {
    #[default]
    Stopped,
    Starting,
    Ready,
    Degraded,
    Failed,
}

/// State of one component plus an optional human-readable detail.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ComponentHealth {
    pub state: ComponentState,
    pub detail: Option<String>,
}

impl ComponentHealth {
    pub fn ready() -> Self {
        Self {
            state: ComponentState::Ready,
            detail: None,
        }
    }
}

/// Health of every in-process runtime component, for the status sidebar.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct RuntimeHealth {
    pub asr: ComponentHealth,
    pub translator: ComponentHealth,
    pub voice_renderer: ComponentHealth,
    pub audio_input: ComponentHealth,
    pub audio_output: ComponentHealth,
    pub virtual_mic: ComponentHealth,
    pub mesh: ComponentHealth,
}

/// Compact telemetry pushed over streaming (`PipelineEvent::Telemetry`). Kept
/// separate from `AppStatus` so the heavier status payload stays on the
/// HTTP/IPC path and does not bloat audio-rate event frames.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TelemetrySnapshot {
    pub gpu: GpuStatus,
    pub active_connections: usize,
    pub pipeline_delay_ms: u64,
}

/// Single reactive snapshot the UI renders over HTTP `/api/status` / Tauri IPC.
/// Pure reflection — the UI never inspects processes itself.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AppStatus {
    pub current_state: NodeState,
    pub gpu: GpuStatus,
    pub voice_configured: bool,
    pub active_connections: usize,
    pub health: RuntimeHealth,
    pub pipeline_delay_ms: u64,
    pub last_error: Option<String>,
}

impl AppStatus {
    /// Project the compact streaming telemetry out of the full status.
    pub fn telemetry(&self) -> TelemetrySnapshot {
        TelemetrySnapshot {
            gpu: self.gpu.clone(),
            active_connections: self.active_connections,
            pipeline_delay_ms: self.pipeline_delay_ms,
        }
    }
}

/// Client→server handshake that opens a streaming session (replaces `StreamStart`).
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct SessionStart {
    pub direction: Direction,
    #[serde(default = "default_stream_synthesize")]
    pub synthesize: bool,
}

/// Unified, symmetric streaming contract. One bincode frame per event over the
/// WS / mesh; also JSON-serializable for the control panel.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PipelineEvent {
    Ready,
    State(NodeState),
    Telemetry(TelemetrySnapshot),
    Listening {
        lane: Lane,
    },
    Processing {
        id: Uuid,
        lane: Lane,
    },
    Transcript {
        id: Uuid,
        lane: Lane,
        lang: Lang,
        text: String,
    },
    Translation {
        id: Uuid,
        lane: Lane,
        lang: Lang,
        text: String,
    },
    AudioFrame {
        id: Uuid,
        lane: Lane,
        spec: AudioSpec,
        pcm: Vec<u8>,
    },
    Done {
        id: Uuid,
        lane: Lane,
        latency_ms: u64,
    },
    Error {
        message: String,
    },
}

/// Current wire-protocol version stamped into every [`EventEnvelope`].
pub const PROTOCOL_VERSION: u16 = 1;

/// Versioned, ordered, session-scoped wrapper around a `PipelineEvent`. Gives
/// the WS/mesh transport sequencing, session correlation, and a version field
/// for forward debugging. `timestamp_ms` is stamped by the producer (not in
/// pure code — passed in) so the type stays deterministic for tests.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EventEnvelope {
    pub version: u16,
    pub session_id: Uuid,
    pub seq: u64,
    pub timestamp_ms: u64,
    pub event: PipelineEvent,
}

impl EventEnvelope {
    pub fn new(session_id: Uuid, seq: u64, timestamp_ms: u64, event: PipelineEvent) -> Self {
        Self {
            version: PROTOCOL_VERSION,
            session_id,
            seq,
            timestamp_ms,
            event,
        }
    }
}

/// Pure FSM derivation. Precedence: explicit error → client mode → server
/// (healthy = active, else initializing) → GPU-incapable preflight → idle.
pub fn derive_node_state(
    live: &Liveness,
    server_healthy: bool,
    gpu: &GpuStatus,
    init: Option<Vec<InitStep>>,
    last_err: Option<&str>,
) -> NodeState {
    if let Some(error) = last_err {
        return NodeState::Error(error.to_string());
    }
    if live.client {
        return NodeState::ActiveClient;
    }
    if live.server {
        return if server_healthy {
            NodeState::ActiveServer
        } else {
            NodeState::Initializing(init.unwrap_or_default())
        };
    }
    if !gpu.is_capable {
        return NodeState::Preflight;
    }
    NodeState::Idle
}

#[cfg(test)]
mod unified_state_tests {
    use super::*;

    fn capable_gpu(is_capable: bool) -> GpuStatus {
        GpuStatus {
            is_capable,
            model_name: "NVIDIA RTX 4090".into(),
            vram_free_mb: 12_300,
            vram_total_mb: 16_384,
            utilization_pct: 12,
            gate_message: "GPU lista".into(),
            processes: Vec::new(),
            source: "nvml".into(),
        }
    }

    #[test]
    fn derive_idle_when_capable_and_nothing_running() {
        let state = derive_node_state(&Liveness::default(), false, &capable_gpu(true), None, None);
        assert_eq!(state, NodeState::Idle);
    }

    #[test]
    fn derive_preflight_when_not_capable() {
        let state = derive_node_state(&Liveness::default(), false, &capable_gpu(false), None, None);
        assert_eq!(state, NodeState::Preflight);
    }

    #[test]
    fn derive_initializing_when_server_up_unhealthy() {
        let live = Liveness {
            server: true,
            ..Default::default()
        };
        let steps = vec![InitStep {
            label: "Carga Whisper".into(),
            status: InitStatus::Running(45),
            elapsed_ms: 2_100,
        }];
        let state = derive_node_state(&live, false, &capable_gpu(true), Some(steps.clone()), None);
        assert_eq!(state, NodeState::Initializing(steps));
    }

    #[test]
    fn derive_active_server_when_healthy() {
        let live = Liveness {
            server: true,
            ..Default::default()
        };
        let state = derive_node_state(&live, true, &capable_gpu(true), None, None);
        assert_eq!(state, NodeState::ActiveServer);
    }

    #[test]
    fn derive_client_wins_over_server() {
        let live = Liveness {
            server: true,
            client: true,
            ..Default::default()
        };
        let state = derive_node_state(&live, true, &capable_gpu(true), None, None);
        assert_eq!(state, NodeState::ActiveClient);
    }

    #[test]
    fn derive_error_takes_precedence() {
        let live = Liveness {
            client: true,
            ..Default::default()
        };
        let state = derive_node_state(&live, true, &capable_gpu(true), None, Some("VRAM baja"));
        assert_eq!(state, NodeState::Error("VRAM baja".into()));
    }

    #[test]
    fn pipeline_event_bincode_roundtrip() {
        let id = Uuid::nil();
        let events = vec![
            PipelineEvent::Ready,
            PipelineEvent::State(NodeState::Initializing(vec![InitStep {
                label: "VRAM".into(),
                status: InitStatus::Ok,
                elapsed_ms: 400,
            }])),
            PipelineEvent::Listening { lane: Lane::Local },
            PipelineEvent::Transcript {
                id,
                lane: Lane::Local,
                lang: Lang::Es,
                text: "Hola".into(),
            },
            PipelineEvent::AudioFrame {
                id,
                lane: Lane::Remote,
                spec: AudioSpec::mono_s16le(24_000),
                pcm: vec![1, 2, 3, 4, 5],
            },
            PipelineEvent::Done {
                id,
                lane: Lane::Local,
                latency_ms: 180,
            },
            PipelineEvent::Error {
                message: "boom".into(),
            },
        ];
        for event in events {
            let bytes = bincode::serialize(&event).expect("bincode serialize");
            let decoded: PipelineEvent = bincode::deserialize(&bytes).expect("bincode deserialize");
            assert_eq!(event, decoded);
        }
    }

    fn sample_status() -> AppStatus {
        AppStatus {
            current_state: NodeState::ActiveServer,
            gpu: capable_gpu(true),
            voice_configured: true,
            active_connections: 2,
            health: RuntimeHealth::default(),
            pipeline_delay_ms: 120,
            last_error: None,
        }
    }

    #[test]
    fn telemetry_snapshot_bincode_roundtrip() {
        let snapshot = sample_status().telemetry();
        let event = PipelineEvent::Telemetry(snapshot.clone());
        let bytes = bincode::serialize(&event).expect("bincode serialize");
        let decoded: PipelineEvent = bincode::deserialize(&bytes).expect("bincode deserialize");
        assert_eq!(PipelineEvent::Telemetry(snapshot), decoded);
    }

    #[test]
    fn event_envelope_bincode_roundtrip() {
        let envelope = EventEnvelope::new(Uuid::nil(), 7, 1_700_000_000_000, PipelineEvent::Ready);
        assert_eq!(envelope.version, PROTOCOL_VERSION);
        let bytes = bincode::serialize(&envelope).expect("ser");
        let decoded: EventEnvelope = bincode::deserialize(&bytes).expect("de");
        assert_eq!(envelope, decoded);
    }

    #[test]
    fn app_status_json_shape_has_expected_keys() {
        let status = sample_status();
        let value = serde_json::to_value(&status).expect("json");
        for key in [
            "current_state",
            "gpu",
            "voice_configured",
            "active_connections",
            "health",
            "pipeline_delay_ms",
            "last_error",
        ] {
            assert!(value.get(key).is_some(), "missing key {key}");
        }
        // Per-component runtime health is present and nested.
        assert!(value["health"]["asr"]["state"].is_string());
    }

    #[test]
    fn runtime_health_defaults_to_stopped() {
        let health = RuntimeHealth::default();
        assert_eq!(health.asr.state, ComponentState::Stopped);
        assert_eq!(health.mesh.state, ComponentState::Stopped);
    }

    #[test]
    fn init_status_variants_roundtrip_bincode() {
        for status in [
            InitStatus::Pending,
            InitStatus::Running(45),
            InitStatus::Ok,
            InitStatus::Failed("driver".into()),
        ] {
            let bytes = bincode::serialize(&status).expect("ser");
            let decoded: InitStatus = bincode::deserialize(&bytes).expect("de");
            assert_eq!(status, decoded);
        }
    }
}
