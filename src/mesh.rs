use crate::types::Direction;
use crate::voice::GeneratedAudioMeta;
use anyhow::{Context, bail};
use async_trait::async_trait;
use futures_util::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, StreamExt};
use libp2p::{
    Multiaddr, PeerId, SwarmBuilder, gossipsub, mdns, noise,
    request_response::{self, ProtocolSupport},
    swarm::{NetworkBehaviour, SwarmEvent},
    tcp, yamux,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    io,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    select,
    sync::{mpsc, oneshot},
    time,
};
use uuid::Uuid;

const GPU_AVAILABILITY_TOPIC: &str = "live-interpreter/disponibilidad-gpu";
const AUDIO_PROTOCOL: &str = "/live-interpreter/audio-task/1";
const MAX_P2P_FRAME_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct MeshConfig {
    pub listen_addr: Multiaddr,
    pub health_interval: Duration,
    pub provider_stale_after: Duration,
    pub local_role: MeshRole,
    /// Shared secret gating the mesh. `None` = open mesh (LAN). When set, peers
    /// only trust providers and accept audio tasks carrying the same token.
    pub auth_token: Option<String>,
}

impl Default for MeshConfig {
    fn default() -> Self {
        Self {
            listen_addr: "/ip4/0.0.0.0/tcp/0"
                .parse()
                .expect("default mesh listen address is valid"),
            health_interval: Duration::from_secs(3),
            provider_stale_after: Duration::from_secs(12),
            local_role: MeshRole::Consumer,
            auth_token: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MeshRole {
    GpuProvider,
    Consumer,
}

/// The consumer's reference voice, shipped so a provider can render the
/// translation in the consumer's own timbre. Consent-gated. Sent once per
/// session (with the first chunk) and cached by the provider; `None` on later
/// chunks and whenever the consumer opts out of cloning.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VoiceReference {
    pub sample_rate_hz: u32,
    pub samples: Vec<f32>,
    pub transcript: Option<String>,
    pub consent_confirmed: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AudioChunk {
    pub session_id: Uuid,
    pub sequence: u64,
    pub sample_rate_hz: u32,
    pub direction: Direction,
    pub samples: Vec<f32>,
    /// Consumer timbre for cross-node cloning; see [`VoiceReference`].
    pub voice_ref: Option<VoiceReference>,
    /// Shared mesh secret; the provider drops the task if it doesn't match.
    pub auth_token: Option<String>,
}

/// One synthesized clause of a streamed interpretation. The provider emits these
/// as each clause finishes (clause 0 first), so the consumer plays clause 1 while
/// the provider is still synthesizing clause 2 — the distributed analog of the
/// local chunked pipeline (R7), lowering time-to-first-audio over the mesh.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AudioSegment {
    pub session_id: Uuid,
    pub sequence: u64,
    /// 0-based clause index within the utterance.
    pub clause_index: u32,
    /// True on the final clause of this utterance.
    pub last: bool,
    /// Full utterance transcription, carried on clause 0 (empty on later clauses).
    pub transcription: String,
    /// This clause's translated text.
    pub translation: String,
    pub tts_sample_rate_hz: u32,
    pub tts_output: Vec<f32>,
    /// R11.4 — provenance of the synthesized audio; `None` when no audio.
    pub meta: Option<GeneratedAudioMeta>,
    /// Shared mesh secret, so the consumer can authorize an inbound delivery.
    pub auth_token: Option<String>,
}

/// Audio-protocol request. A consumer **submits** an utterance; the provider
/// **delivers** each synthesized clause back as a separate request, so audio
/// streams clause-by-clause instead of waiting for the whole utterance.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub enum AudioRequest {
    /// Consumer → provider: interpret this utterance.
    Submit(AudioChunk),
    /// Provider → consumer: one finished clause of the interpretation.
    Deliver(AudioSegment),
}

/// Audio-protocol response (acks only; the payload travels as `Deliver` requests).
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AudioResponse {
    /// Submit accepted; clauses will follow as `Deliver` requests.
    Accepted,
    /// A delivered clause was received.
    DeliverAck,
    /// Unauthorized, or this node isn't a provider.
    Rejected,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MeshHealth {
    pub peer_id: String,
    pub role: MeshRole,
    pub free_vram_mb: u64,
    pub total_vram_mb: u64,
    pub active_sessions: u32,
    pub unix_ms: u64,
    /// Shared mesh secret; consumers ignore providers whose token mismatches.
    pub token: Option<String>,
}

/// Effective latency assigned to a provider we haven't measured yet: low enough
/// to be probed over a known-slow node, high enough not to evict a known-fast one.
const LATENCY_UNKNOWN_RANK_MS: u64 = 3_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderScore {
    pub peer_id: PeerId,
    pub free_vram_mb: u64,
    pub active_sessions: u32,
    pub last_seen: Instant,
    /// EWMA of observed round-trip latency (ms); 0 = not yet measured.
    pub avg_latency_ms: u64,
}

impl ProviderScore {
    /// Ranking for `max_by_key`: prefer lowest (effective) latency, then most
    /// free VRAM, then fewest active sessions. Real-time routing favours the
    /// consistently faster node once it has been measured.
    fn rank_key(&self) -> (std::cmp::Reverse<u64>, u64, std::cmp::Reverse<u32>) {
        let effective_latency = if self.avg_latency_ms == 0 {
            LATENCY_UNKNOWN_RANK_MS
        } else {
            self.avg_latency_ms
        };
        (
            std::cmp::Reverse(effective_latency),
            self.free_vram_mb,
            std::cmp::Reverse(self.active_sessions),
        )
    }
}

#[derive(Debug)]
pub enum MeshCommand {
    SubmitAudio {
        chunk: AudioChunk,
        /// Sink the consumer reads: the provider's clauses arrive here in order as
        /// they finish. Closes after the `last` clause (success) or when routing
        /// gives up with no provider (failure → channel closes with no `last`).
        segments: mpsc::Sender<AudioSegment>,
    },
    SetRole(MeshRole),
    Shutdown,
}

#[async_trait]
pub trait GpuTelemetry: Send + Sync + 'static {
    async fn read(&self) -> anyhow::Result<GpuTelemetrySnapshot>;
}

#[async_trait]
pub trait MeshAudioProcessor: Send + Sync + 'static {
    /// Interpret `chunk`, emitting one [`AudioSegment`] per clause through
    /// `segments` as each finishes (clause 0 first, `last` on the final one), so
    /// the consumer can play clause 1 while clause 2 is still being synthesized.
    /// Returning `Ok(())` ends the stream.
    async fn process_stream(
        &self,
        chunk: AudioChunk,
        segments: mpsc::Sender<AudioSegment>,
    ) -> anyhow::Result<()>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuTelemetrySnapshot {
    pub free_vram_mb: u64,
    pub total_vram_mb: u64,
    pub active_sessions: u32,
}

pub struct NoopGpuTelemetry;

#[async_trait]
impl GpuTelemetry for NoopGpuTelemetry {
    async fn read(&self) -> anyhow::Result<GpuTelemetrySnapshot> {
        Ok(GpuTelemetrySnapshot {
            free_vram_mb: 0,
            total_vram_mb: 0,
            active_sessions: 0,
        })
    }
}

pub struct NvmlGpuTelemetry;

#[async_trait]
impl GpuTelemetry for NvmlGpuTelemetry {
    async fn read(&self) -> anyhow::Result<GpuTelemetrySnapshot> {
        tokio::task::spawn_blocking(read_nvml_snapshot)
            .await
            .context("NVML telemetry task panicked")?
    }
}

fn read_nvml_snapshot() -> anyhow::Result<GpuTelemetrySnapshot> {
    // Single NVML source of truth: reuse crate::vram::nvml_snapshot (device 0) instead of a second
    // probe. active_sessions = processes holding VRAM on the device.
    let snapshot = crate::vram::nvml_snapshot()?;
    Ok(GpuTelemetrySnapshot {
        free_vram_mb: snapshot.free_mb,
        total_vram_mb: snapshot.total_mb,
        active_sessions: snapshot.processes.len() as u32,
    })
}

pub struct RejectingAudioProcessor;

#[async_trait]
impl MeshAudioProcessor for RejectingAudioProcessor {
    async fn process_stream(
        &self,
        _chunk: AudioChunk,
        _segments: mpsc::Sender<AudioSegment>,
    ) -> anyhow::Result<()> {
        bail!("local node is not configured as a GPU audio processor")
    }
}

pub struct NotReadyAudioProcessor;

#[async_trait]
impl MeshAudioProcessor for NotReadyAudioProcessor {
    async fn process_stream(
        &self,
        chunk: AudioChunk,
        segments: mpsc::Sender<AudioSegment>,
    ) -> anyhow::Result<()> {
        let _ = segments
            .send(AudioSegment {
                session_id: chunk.session_id,
                sequence: chunk.sequence,
                clause_index: 0,
                last: true,
                transcription: String::new(),
                translation: "mesh GPU provider is running but audio pipeline is not wired yet"
                    .into(),
                tts_sample_rate_hz: chunk.sample_rate_hz,
                tts_output: Vec::new(),
                meta: None,
                auth_token: None,
            })
            .await;
        Ok(())
    }
}

#[derive(NetworkBehaviour)]
struct MeshBehaviour {
    gossipsub: gossipsub::Behaviour,
    mdns: mdns::tokio::Behaviour,
    audio: request_response::Behaviour<AudioCodec>,
}

#[derive(Clone, Debug)]
struct AudioProtocol;

impl AsRef<str> for AudioProtocol {
    fn as_ref(&self) -> &str {
        AUDIO_PROTOCOL
    }
}

#[derive(Clone, Default)]
struct AudioCodec;

#[async_trait]
impl request_response::Codec for AudioCodec {
    type Protocol = AudioProtocol;
    type Request = AudioRequest;
    type Response = AudioResponse;

    async fn read_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Request>
    where
        T: AsyncRead + Unpin + Send,
    {
        read_bincode(io).await
    }

    async fn read_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        read_bincode(io).await
    }

    async fn write_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        req: Self::Request,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        write_bincode(io, &req).await
    }

    async fn write_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        res: Self::Response,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        write_bincode(io, &res).await
    }
}

async fn read_bincode<T, V>(io: &mut T) -> io::Result<V>
where
    T: AsyncRead + Unpin,
    V: for<'de> Deserialize<'de>,
{
    let mut len_bytes = [0u8; 4];
    io.read_exact(&mut len_bytes).await?;
    let len = u32::from_be_bytes(len_bytes) as usize;
    if len > MAX_P2P_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "mesh frame exceeds maximum size",
        ));
    }
    let mut bytes = vec![0; len];
    io.read_exact(&mut bytes).await?;
    bincode::deserialize(&bytes).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

async fn write_bincode<T, V>(io: &mut T, value: &V) -> io::Result<()>
where
    T: AsyncWrite + Unpin,
    V: Serialize,
{
    let bytes = bincode::serialize(value)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if bytes.len() > MAX_P2P_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "mesh frame exceeds maximum size",
        ));
    }
    io.write_all(&(bytes.len() as u32).to_be_bytes()).await?;
    io.write_all(&bytes).await?;
    io.close().await
}

pub struct LiveInterpreterMesh<T, P> {
    config: MeshConfig,
    telemetry: T,
    processor: Arc<P>,
    providers: HashMap<PeerId, ProviderScore>,
    /// Outstanding `Submit` requests awaiting an `Accepted` ack (for failover).
    pending: HashMap<request_response::OutboundRequestId, PendingSubmit>,
    /// Consumer side: per-(session, sequence) sink where inbound `Deliver`
    /// clauses are forwarded so the caller plays them in order.
    sessions: HashMap<(Uuid, u64), mpsc::Sender<AudioSegment>>,
    /// Provider side: outstanding `Deliver` requests awaiting their `DeliverAck`,
    /// so the synthesis task can serialize delivery (in-order clauses).
    delivery_acks: HashMap<request_response::OutboundRequestId, oneshot::Sender<()>>,
}

struct PendingSubmit {
    chunk: AudioChunk,
    attempted: Vec<PeerId>,
    sent_at: Instant,
}

/// Internal self-injected command: a provider's synthesis task asks the run loop
/// (the only holder of the swarm) to deliver a finished clause to a consumer.
/// `acked` fires once the consumer acks, so the task delivers clauses **in order**
/// (request-response gives no cross-request ordering on its own).
enum Internal {
    Deliver {
        peer: PeerId,
        segment: AudioSegment,
        acked: oneshot::Sender<()>,
    },
}

impl<T, P> LiveInterpreterMesh<T, P>
where
    T: GpuTelemetry,
    P: MeshAudioProcessor,
{
    pub fn new(config: MeshConfig, telemetry: T, processor: P) -> Self {
        Self {
            config,
            telemetry,
            processor: Arc::new(processor),
            providers: HashMap::new(),
            pending: HashMap::new(),
            sessions: HashMap::new(),
            delivery_acks: HashMap::new(),
        }
    }

    pub fn command_channel() -> (mpsc::Sender<MeshCommand>, mpsc::Receiver<MeshCommand>) {
        mpsc::channel(64)
    }

    pub async fn run(mut self, mut commands: mpsc::Receiver<MeshCommand>) -> anyhow::Result<()> {
        let mut swarm = build_swarm()?;
        let topic = gossipsub::IdentTopic::new(GPU_AVAILABILITY_TOPIC);
        swarm
            .behaviour_mut()
            .gossipsub
            .subscribe(&topic)
            .context("failed to subscribe to GPU availability topic")?;
        swarm.listen_on(self.config.listen_addr.clone())?;

        let mut health_tick = time::interval(self.config.health_interval);
        health_tick.set_missed_tick_behavior(time::MissedTickBehavior::Delay);

        // Provider synthesis tasks run off the loop and inject `Deliver`s here, the
        // only place that owns the swarm.
        let (internal_tx, mut internal_rx) = mpsc::channel::<Internal>(256);

        loop {
            select! {
                _ = health_tick.tick() => {
                    self.prune_stale_providers();
                    let health = self.local_health(*swarm.local_peer_id()).await?;
                    let payload = serde_json::to_vec(&health)?;
                    let _ = swarm.behaviour_mut().gossipsub.publish(topic.clone(), payload);
                }
                Some(command) = commands.recv() => {
                    if self.handle_command(command, &mut swarm).await? {
                        return Ok(());
                    }
                }
                Some(internal) = internal_rx.recv() => {
                    match internal {
                        Internal::Deliver { peer, segment, acked } => {
                            let request_id = swarm
                                .behaviour_mut()
                                .audio
                                .send_request(&peer, AudioRequest::Deliver(segment));
                            self.delivery_acks.insert(request_id, acked);
                        }
                    }
                }
                event = swarm.select_next_some() => {
                    self.handle_swarm_event(event, &mut swarm, &internal_tx).await?;
                }
            }
        }
    }

    async fn handle_command(
        &mut self,
        command: MeshCommand,
        swarm: &mut libp2p::Swarm<MeshBehaviour>,
    ) -> anyhow::Result<bool> {
        match command {
            MeshCommand::SubmitAudio { chunk, segments } => {
                self.submit_audio(chunk, segments, swarm);
                Ok(false)
            }
            MeshCommand::SetRole(role) => {
                self.config.local_role = role;
                Ok(false)
            }
            MeshCommand::Shutdown => Ok(true),
        }
    }

    fn submit_audio(
        &mut self,
        chunk: AudioChunk,
        segments: mpsc::Sender<AudioSegment>,
        swarm: &mut libp2p::Swarm<MeshBehaviour>,
    ) {
        // Register the sink first so inbound `Deliver` clauses have somewhere to go.
        let key = (chunk.session_id, chunk.sequence);
        self.sessions.insert(key, segments);

        let Some(peer_id) = self.best_provider(&[]) else {
            // No provider → drop the sink so the caller's recv() ends (failure).
            self.sessions.remove(&key);
            return;
        };

        let request_id = swarm
            .behaviour_mut()
            .audio
            .send_request(&peer_id, AudioRequest::Submit(chunk.clone()));
        self.pending.insert(
            request_id,
            PendingSubmit {
                chunk,
                attempted: vec![peer_id],
                sent_at: Instant::now(),
            },
        );
    }

    async fn handle_swarm_event(
        &mut self,
        event: SwarmEvent<MeshBehaviourEvent>,
        swarm: &mut libp2p::Swarm<MeshBehaviour>,
        internal_tx: &mpsc::Sender<Internal>,
    ) -> anyhow::Result<()> {
        match event {
            SwarmEvent::Behaviour(MeshBehaviourEvent::Mdns(mdns::Event::Discovered(peers))) => {
                for (peer_id, addr) in peers {
                    swarm.behaviour_mut().gossipsub.add_explicit_peer(&peer_id);
                    swarm.add_peer_address(peer_id, addr);
                }
            }
            SwarmEvent::Behaviour(MeshBehaviourEvent::Mdns(mdns::Event::Expired(peers))) => {
                for (peer_id, _addr) in peers {
                    swarm
                        .behaviour_mut()
                        .gossipsub
                        .remove_explicit_peer(&peer_id);
                    self.providers.remove(&peer_id);
                }
            }
            SwarmEvent::Behaviour(MeshBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                message,
                ..
            })) => {
                if let Ok(health) = serde_json::from_slice::<MeshHealth>(&message.data) {
                    self.observe_health(health);
                }
            }
            SwarmEvent::Behaviour(MeshBehaviourEvent::Audio(
                request_response::Event::Message { message, peer, .. },
            )) => match message {
                request_response::Message::Request {
                    request, channel, ..
                } => {
                    let response = self.handle_audio_request(request, peer, internal_tx);
                    let _ = swarm.behaviour_mut().audio.send_response(channel, response);
                }
                request_response::Message::Response {
                    request_id,
                    response,
                } => self.handle_audio_response(request_id, response, swarm),
            },
            SwarmEvent::Behaviour(MeshBehaviourEvent::Audio(
                request_response::Event::OutboundFailure {
                    request_id, error, ..
                },
            )) => {
                // A failed `Deliver`: drop its ack so the synthesis task stops
                // waiting and abandons the rest of the stream.
                if self.delivery_acks.remove(&request_id).is_some() {
                    tracing::warn!("mesh clause delivery failed: {error}");
                } else {
                    // Otherwise it was a `Submit` → fail over to the next provider.
                    self.retry_or_fail(request_id, anyhow::anyhow!(error.to_string()), swarm);
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Handle an inbound audio request and return the ack to send back.
    ///
    /// * `Submit` (provider role, authorized): ack `Accepted`, then spawn the
    ///   streaming pipeline; each finished clause is injected back via `internal_tx`
    ///   as a `Deliver` to the submitting `peer`.
    /// * `Deliver` (consumer side): forward the clause to its session sink, ack
    ///   `DeliverAck`. Unknown session is still acked (late/duplicate clause).
    fn handle_audio_request(
        &mut self,
        request: AudioRequest,
        peer: PeerId,
        internal_tx: &mpsc::Sender<Internal>,
    ) -> AudioResponse {
        match request {
            AudioRequest::Submit(chunk) => {
                let authorized = token_matches(&self.config.auth_token, &chunk.auth_token);
                if self.config.local_role != MeshRole::GpuProvider || !authorized {
                    return AudioResponse::Rejected;
                }
                let processor = self.processor.clone();
                let internal = internal_tx.clone();
                let token = self.config.auth_token.clone();
                tokio::spawn(async move {
                    let (seg_tx, mut seg_rx) = mpsc::channel::<AudioSegment>(64);
                    let synth =
                        tokio::spawn(async move { processor.process_stream(chunk, seg_tx).await });
                    // Deliver clauses one at a time, waiting for each ack before the
                    // next, so they arrive at the consumer in order. Synthesis still
                    // runs ahead (buffered in seg_rx), preserving the latency win.
                    while let Some(mut segment) = seg_rx.recv().await {
                        segment.auth_token = token.clone();
                        let (ack_tx, ack_rx) = oneshot::channel();
                        if internal
                            .send(Internal::Deliver {
                                peer,
                                segment,
                                acked: ack_tx,
                            })
                            .await
                            .is_err()
                        {
                            break; // mesh loop gone
                        }
                        if ack_rx.await.is_err() {
                            break; // delivery failed (consumer gone) → stop
                        }
                    }
                    if let Ok(Err(error)) = synth.await {
                        tracing::warn!("mesh process_stream failed: {error:#}");
                    }
                });
                AudioResponse::Accepted
            }
            AudioRequest::Deliver(segment) => {
                if !token_matches(&self.config.auth_token, &segment.auth_token) {
                    return AudioResponse::Rejected;
                }
                let key = (segment.session_id, segment.sequence);
                let last = segment.last;
                if let Some(sink) = self.sessions.get(&key) {
                    // try_send: never block the run loop on a slow consumer.
                    let _ = sink.try_send(segment);
                }
                if last {
                    self.sessions.remove(&key);
                }
                AudioResponse::DeliverAck
            }
        }
    }

    /// Handle an inbound audio response (an ack to one of our requests).
    fn handle_audio_response(
        &mut self,
        request_id: request_response::OutboundRequestId,
        response: AudioResponse,
        swarm: &mut libp2p::Swarm<MeshBehaviour>,
    ) {
        match response {
            // Our `Submit` was accepted: record routing latency; clauses now
            // stream in as `Deliver` requests handled above.
            AudioResponse::Accepted => {
                if let Some(pending) = self.pending.remove(&request_id)
                    && let Some(peer) = pending.attempted.last()
                {
                    self.record_latency(*peer, pending.sent_at.elapsed().as_millis() as u64);
                }
            }
            // A provider rejected our `Submit` → fail over to the next provider.
            AudioResponse::Rejected => {
                self.retry_or_fail(
                    request_id,
                    anyhow::anyhow!("provider rejected the audio task"),
                    swarm,
                );
            }
            // Consumer acked one of our `Deliver`s: release the next clause.
            AudioResponse::DeliverAck => {
                if let Some(ack) = self.delivery_acks.remove(&request_id) {
                    let _ = ack.send(());
                }
            }
        }
    }

    async fn local_health(&self, peer_id: PeerId) -> anyhow::Result<MeshHealth> {
        let snapshot = if self.config.local_role == MeshRole::GpuProvider {
            self.telemetry.read().await?
        } else {
            GpuTelemetrySnapshot {
                free_vram_mb: 0,
                total_vram_mb: 0,
                active_sessions: 0,
            }
        };

        Ok(MeshHealth {
            peer_id: peer_id.to_string(),
            role: self.config.local_role,
            free_vram_mb: snapshot.free_vram_mb,
            total_vram_mb: snapshot.total_vram_mb,
            active_sessions: snapshot.active_sessions,
            unix_ms: current_unix_ms(),
            token: self.config.auth_token.clone(),
        })
    }

    fn observe_health(&mut self, health: MeshHealth) {
        if !token_matches(&self.config.auth_token, &health.token) {
            return; // provider not on our mesh (token mismatch)
        }
        let Ok(peer_id) = health.peer_id.parse::<PeerId>() else {
            return;
        };

        if health.role != MeshRole::GpuProvider {
            self.providers.remove(&peer_id);
            return;
        }

        // Preserve any latency we've already measured for this peer across the
        // periodic health refresh.
        let avg_latency_ms = self
            .providers
            .get(&peer_id)
            .map(|provider| provider.avg_latency_ms)
            .unwrap_or(0);
        self.providers.insert(
            peer_id,
            ProviderScore {
                peer_id,
                free_vram_mb: health.free_vram_mb,
                active_sessions: health.active_sessions,
                last_seen: Instant::now(),
                avg_latency_ms,
            },
        );
    }

    /// Fold an observed round-trip latency into a provider's EWMA (3:1 toward
    /// history), so routing tracks sustained speed rather than one-off spikes.
    fn record_latency(&mut self, peer_id: PeerId, latency_ms: u64) {
        if let Some(provider) = self.providers.get_mut(&peer_id) {
            provider.avg_latency_ms = if provider.avg_latency_ms == 0 {
                latency_ms
            } else {
                (provider.avg_latency_ms * 3 + latency_ms) / 4
            };
        }
    }

    fn retry_or_fail(
        &mut self,
        request_id: request_response::OutboundRequestId,
        error: anyhow::Error,
        swarm: &mut libp2p::Swarm<MeshBehaviour>,
    ) {
        let Some(mut pending) = self.pending.remove(&request_id) else {
            return; // not a Submit we're tracking (e.g. a Deliver failure) → ignore
        };

        if let Some(next_peer) = self.best_provider(&pending.attempted) {
            pending.attempted.push(next_peer);
            pending.sent_at = Instant::now();
            let next_request_id = swarm
                .behaviour_mut()
                .audio
                .send_request(&next_peer, AudioRequest::Submit(pending.chunk.clone()));
            self.pending.insert(next_request_id, pending);
        } else {
            // No provider left: drop the sink so the consumer's recv() ends.
            self.sessions
                .remove(&(pending.chunk.session_id, pending.chunk.sequence));
            tracing::warn!("{:#}", error.context("all GPU providers failed"));
        }
    }

    fn prune_stale_providers(&mut self) {
        let stale_after = self.config.provider_stale_after;
        self.providers
            .retain(|_, provider| provider.last_seen.elapsed() <= stale_after);
    }

    fn best_provider(&self, excluded: &[PeerId]) -> Option<PeerId> {
        let selected = select_best_provider(self.providers.values(), excluded)?;
        // R11.5: surface the whole-utterance routing decision for explainability.
        tracing::info!("{}", explain_route(selected));
        Some(selected.peer_id)
    }
}

fn build_swarm() -> anyhow::Result<libp2p::Swarm<MeshBehaviour>> {
    let swarm = SwarmBuilder::with_new_identity()
        .with_tokio()
        .with_tcp(
            tcp::Config::default(),
            noise::Config::new,
            yamux::Config::default,
        )?
        .with_behaviour(|key| {
            let peer_id = PeerId::from(key.public());
            let gossipsub_config = gossipsub::ConfigBuilder::default()
                .heartbeat_interval(Duration::from_secs(1))
                .validation_mode(gossipsub::ValidationMode::Permissive)
                .build()
                .context("invalid gossipsub config")?;
            let gossipsub = gossipsub::Behaviour::new(
                gossipsub::MessageAuthenticity::Signed(key.clone()),
                gossipsub_config,
            )?;
            let mdns = mdns::tokio::Behaviour::new(mdns::Config::default(), peer_id)?;
            let audio = request_response::Behaviour::with_codec(
                AudioCodec,
                [(AudioProtocol, ProtocolSupport::Full)],
                request_response::Config::default().with_request_timeout(Duration::from_secs(10)),
            );
            Ok(MeshBehaviour {
                gossipsub,
                mdns,
                audio,
            })
        })?
        .build();

    Ok(swarm)
}

fn select_best_provider<'a>(
    providers: impl IntoIterator<Item = &'a ProviderScore>,
    excluded: &[PeerId],
) -> Option<&'a ProviderScore> {
    providers
        .into_iter()
        .filter(|provider| !excluded.contains(&provider.peer_id))
        .max_by_key(|provider| provider.rank_key())
}

/// R11.5 — human-readable explanation of why a provider won the routing for an
/// utterance. **Whole-utterance** routing (not per-stage, per the R11 decision):
/// the entire ASR→translate→TTS chain runs on the chosen node.
///
/// Example: `utterance → …Qm3FbA9: 18ms RTT, 4096 MB free, 1 active session(s)`.
pub fn explain_route(score: &ProviderScore) -> String {
    let latency = if score.avg_latency_ms == 0 {
        "latency unknown (probing)".to_string()
    } else {
        format!("{}ms RTT", score.avg_latency_ms)
    };
    format!(
        "utterance → {}: {}, {} MB free, {} active session(s)",
        short_peer(&score.peer_id),
        latency,
        score.free_vram_mb,
        score.active_sessions,
    )
}

/// Last 8 chars of a peer id, prefixed with `…`, for readable logs/UI.
fn short_peer(peer_id: &PeerId) -> String {
    let full = peer_id.to_string();
    format!("…{}", &full[full.len().saturating_sub(8)..])
}

/// Mesh access control: `None` expected = open mesh (accept anyone); otherwise
/// the presented token must match exactly.
fn token_matches(expected: &Option<String>, presented: &Option<String>) -> bool {
    match expected {
        None => true,
        Some(secret) => presented.as_deref() == Some(secret.as_str()),
    }
}

fn current_unix_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::identity;

    fn peer() -> PeerId {
        identity::Keypair::generate_ed25519().public().to_peer_id()
    }

    fn score(
        peer_id: PeerId,
        free_vram_mb: u64,
        active_sessions: u32,
        avg_latency_ms: u64,
    ) -> ProviderScore {
        ProviderScore {
            peer_id,
            free_vram_mb,
            active_sessions,
            last_seen: Instant::now(),
            avg_latency_ms,
        }
    }

    #[test]
    fn explain_route_reports_latency_and_resources() {
        let measured = explain_route(&score(peer(), 4_096, 1, 18));
        assert!(measured.starts_with("utterance → "));
        assert!(measured.contains("18ms RTT"));
        assert!(measured.contains("4096 MB free"));
        assert!(measured.contains("1 active session(s)"));

        let unknown = explain_route(&score(peer(), 8_000, 0, 0));
        assert!(unknown.contains("latency unknown"));
    }

    #[test]
    fn selects_provider_with_most_free_vram_then_fewer_sessions() {
        // All latencies equal (unknown) → falls through to VRAM then sessions.
        let (peer_a, peer_b, peer_c) = (peer(), peer(), peer());
        let providers = vec![
            score(peer_a, 16_000, 2, 0),
            score(peer_b, 24_000, 4, 0),
            score(peer_c, 24_000, 1, 0),
        ];

        let selected = select_best_provider(&providers, &[]).unwrap();

        assert_eq!(selected.peer_id, peer_c);
    }

    #[test]
    fn excludes_failed_provider_on_failover_selection() {
        let (peer_a, peer_b) = (peer(), peer());
        let providers = vec![score(peer_a, 32_000, 0, 0), score(peer_b, 16_000, 0, 0)];

        let selected = select_best_provider(&providers, &[peer_a]).unwrap();

        assert_eq!(selected.peer_id, peer_b);
    }

    #[test]
    fn prefers_lower_latency_provider_over_more_vram() {
        // Real-time routing: a measured-fast node beats a slower one with more VRAM.
        let (fast, slow) = (peer(), peer());
        let providers = vec![score(slow, 32_000, 0, 5_000), score(fast, 8_000, 0, 800)];
        assert_eq!(select_best_provider(&providers, &[]).unwrap().peer_id, fast);
    }

    #[test]
    fn unknown_latency_is_probed_over_known_slow_but_loses_to_known_fast() {
        let (unknown, slow, fast) = (peer(), peer(), peer());
        // Unknown (effective 3000ms) beats a known-5000ms node...
        let vs_slow = vec![score(unknown, 8_000, 0, 0), score(slow, 8_000, 0, 5_000)];
        assert_eq!(
            select_best_provider(&vs_slow, &[]).unwrap().peer_id,
            unknown
        );
        // ...but loses to a known-1000ms node.
        let vs_fast = vec![score(unknown, 8_000, 0, 0), score(fast, 8_000, 0, 1_000)];
        assert_eq!(select_best_provider(&vs_fast, &[]).unwrap().peer_id, fast);
    }

    #[test]
    fn record_latency_seeds_then_smooths_ewma() {
        let mut mesh = LiveInterpreterMesh::new(
            MeshConfig::default(),
            NoopGpuTelemetry,
            RejectingAudioProcessor,
        );
        let p = peer();
        mesh.providers.insert(p, score(p, 8_000, 0, 0));
        mesh.record_latency(p, 1_000); // first sample seeds directly
        assert_eq!(mesh.providers[&p].avg_latency_ms, 1_000);
        mesh.record_latency(p, 5_000); // EWMA: (1000*3 + 5000)/4 = 2000
        assert_eq!(mesh.providers[&p].avg_latency_ms, 2_000);
    }

    #[test]
    fn token_matches_open_and_secret_meshes() {
        assert!(token_matches(&None, &None));
        assert!(token_matches(&None, &Some("anything".into()))); // open mesh
        assert!(token_matches(&Some("s".into()), &Some("s".into())));
        assert!(!token_matches(&Some("s".into()), &Some("other".into())));
        assert!(!token_matches(&Some("s".into()), &None));
    }

    #[test]
    fn observe_health_ignores_provider_with_wrong_token() {
        let mut mesh = LiveInterpreterMesh::new(
            MeshConfig {
                auth_token: Some("secret".into()),
                ..MeshConfig::default()
            },
            NoopGpuTelemetry,
            RejectingAudioProcessor,
        );
        // Wrong token → not added.
        mesh.observe_health(health(peer(), MeshRole::GpuProvider, Some("nope".into())));
        assert!(mesh.providers.is_empty());
        // Matching token → added.
        mesh.observe_health(health(peer(), MeshRole::GpuProvider, Some("secret".into())));
        assert_eq!(mesh.providers.len(), 1);
    }

    #[test]
    fn observe_health_preserves_measured_latency_across_refresh() {
        let mut mesh = LiveInterpreterMesh::new(
            MeshConfig::default(),
            NoopGpuTelemetry,
            RejectingAudioProcessor,
        );
        let p = peer();
        mesh.observe_health(health(p, MeshRole::GpuProvider, None));
        mesh.record_latency(p, 900);
        // A fresh health tick must not wipe the measured latency.
        mesh.observe_health(health(p, MeshRole::GpuProvider, None));
        assert_eq!(mesh.providers[&p].avg_latency_ms, 900);
    }

    fn health(peer: PeerId, role: MeshRole, token: Option<String>) -> MeshHealth {
        MeshHealth {
            peer_id: peer.to_string(),
            role,
            free_vram_mb: 8_000,
            total_vram_mb: 16_000,
            active_sessions: 0,
            unix_ms: 1,
            token,
        }
    }

    #[test]
    fn bincode_roundtrip_keeps_audio_chunk_shape() {
        let chunk = AudioChunk {
            session_id: Uuid::new_v4(),
            sequence: 42,
            sample_rate_hz: 16_000,
            direction: Direction::EsToEn,
            samples: vec![0.0, 0.5, -0.25],
            voice_ref: None,
            auth_token: None,
        };

        let bytes = bincode::serialize(&chunk).unwrap();
        let decoded: AudioChunk = bincode::deserialize(&bytes).unwrap();

        assert_eq!(decoded, chunk);
    }

    #[test]
    fn bincode_roundtrip_keeps_voice_reference() {
        let chunk = AudioChunk {
            session_id: Uuid::new_v4(),
            sequence: 0,
            sample_rate_hz: 24_000,
            direction: Direction::EsToEn,
            samples: vec![0.1, 0.2],
            voice_ref: Some(VoiceReference {
                sample_rate_hz: 24_000,
                samples: vec![0.3, -0.3, 0.6],
                transcript: Some("En un lugar de la Mancha".into()),
                consent_confirmed: true,
            }),
            auth_token: Some("secret".into()),
        };

        let bytes = bincode::serialize(&chunk).unwrap();
        let decoded: AudioChunk = bincode::deserialize(&bytes).unwrap();

        assert_eq!(decoded, chunk);
    }

    #[test]
    fn bincode_roundtrip_keeps_audio_request_variants() {
        let segment = AudioSegment {
            session_id: Uuid::new_v4(),
            sequence: 3,
            clause_index: 1,
            last: true,
            transcription: "hola mundo".into(),
            translation: "world".into(),
            tts_sample_rate_hz: 24_000,
            tts_output: vec![0.1, -0.2, 0.3],
            meta: None,
            auth_token: Some("secret".into()),
        };
        // Provider → consumer: a delivered clause.
        let deliver = AudioRequest::Deliver(segment.clone());
        let decoded: AudioRequest =
            bincode::deserialize(&bincode::serialize(&deliver).unwrap()).unwrap();
        assert_eq!(decoded, deliver);

        // Consumer → provider: a submit.
        let submit = AudioRequest::Submit(AudioChunk {
            session_id: segment.session_id,
            sequence: 0,
            sample_rate_hz: 16_000,
            direction: Direction::EsToEn,
            samples: vec![0.0; 4],
            voice_ref: None,
            auth_token: None,
        });
        let decoded: AudioRequest =
            bincode::deserialize(&bincode::serialize(&submit).unwrap()).unwrap();
        assert_eq!(decoded, submit);
    }

    #[test]
    fn bincode_roundtrip_keeps_audio_response_acks() {
        for response in [
            AudioResponse::Accepted,
            AudioResponse::DeliverAck,
            AudioResponse::Rejected,
        ] {
            let decoded: AudioResponse =
                bincode::deserialize(&bincode::serialize(&response).unwrap()).unwrap();
            assert_eq!(decoded, response);
        }
    }

    #[test]
    fn consumer_health_removes_provider() {
        let p = peer();
        let mut mesh = LiveInterpreterMesh::new(
            MeshConfig::default(),
            NoopGpuTelemetry,
            RejectingAudioProcessor,
        );
        mesh.observe_health(health(p, MeshRole::GpuProvider, None));
        assert_eq!(mesh.providers.len(), 1);

        // The same peer re-announcing as a consumer drops it from the table.
        mesh.observe_health(health(p, MeshRole::Consumer, None));
        assert!(mesh.providers.is_empty());
    }
}
