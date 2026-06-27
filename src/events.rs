//! Session event hub: fan-out of `EventEnvelope`s to every connected UI/peer.
//!
//! A single `tokio::sync::broadcast` channel per session. The pipeline publishes
//! `PipelineEvent`s (stamped into ordered, versioned envelopes); WS subscribers
//! (control panel, remote mesh peers) each receive the same stream — the basis
//! for the symmetric console (both lanes visible to every subscriber).

use crate::types::{EventEnvelope, PipelineEvent};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::broadcast;
use uuid::Uuid;

/// Broadcasts ordered session events to all subscribers.
pub struct EventHub {
    tx: broadcast::Sender<EventEnvelope>,
    session_id: Uuid,
    seq: AtomicU64,
}

impl EventHub {
    pub fn new(session_id: Uuid, capacity: usize) -> Self {
        let (tx, _rx) = broadcast::channel(capacity.max(1));
        Self {
            tx,
            session_id,
            seq: AtomicU64::new(0),
        }
    }

    /// Stamp an event into an ordered envelope and fan it out. Returns the
    /// envelope (never errors on zero subscribers).
    pub fn publish(&self, event: PipelineEvent, timestamp_ms: u64) -> EventEnvelope {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let envelope = EventEnvelope::new(self.session_id, seq, timestamp_ms, event);
        let _ = self.tx.send(envelope.clone());
        envelope
    }

    pub fn subscribe(&self) -> broadcast::Receiver<EventEnvelope> {
        self.tx.subscribe()
    }

    pub fn subscriber_count(&self) -> usize {
        self.tx.receiver_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Lane;

    #[tokio::test]
    async fn fans_out_same_ordered_stream_to_all_subscribers() {
        let hub = EventHub::new(Uuid::nil(), 16);
        let mut a = hub.subscribe();
        let mut b = hub.subscribe();
        assert_eq!(hub.subscriber_count(), 2);

        hub.publish(PipelineEvent::Listening { lane: Lane::Local }, 1);
        hub.publish(
            PipelineEvent::Done {
                id: Uuid::nil(),
                lane: Lane::Local,
                latency_ms: 5,
            },
            2,
        );

        for rx in [&mut a, &mut b] {
            let first = rx.recv().await.unwrap();
            let second = rx.recv().await.unwrap();
            assert_eq!(first.seq, 0);
            assert_eq!(second.seq, 1);
            assert_eq!(first.version, crate::types::PROTOCOL_VERSION);
            assert!(matches!(first.event, PipelineEvent::Listening { .. }));
        }
    }

    #[tokio::test]
    async fn publish_without_subscribers_is_ok() {
        let hub = EventHub::new(Uuid::nil(), 4);
        let envelope = hub.publish(PipelineEvent::Ready, 0);
        assert_eq!(envelope.seq, 0);
        assert_eq!(hub.subscriber_count(), 0);
    }
}
