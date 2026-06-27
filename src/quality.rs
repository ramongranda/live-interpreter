//! R11.3 — Quality tiers: degrade gracefully instead of failing.
//!
//! Real-time voice has a hard latency floor (ASR on CPU runs ~23s/utterance with
//! the largest Whisper model — see the R11.1 bench). Rather than error when
//! resources are tight, the runtime steps *down* a quality ladder: a smaller
//! Whisper model under VRAM pressure, the neutral voice when the clone is
//! unavailable, local execution when the mesh is down. Pure policy — no I/O — so
//! the decisions are unit-testable without a GPU or a network.

use serde::{Deserialize, Serialize};

/// How much fidelity the runtime trades for latency / robustness.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum QualityTier {
    /// Lowest latency: smallest model, accept lower fidelity (conversation).
    Realtime,
    /// Balance latency and fidelity.
    #[default]
    Balanced,
    /// Highest fidelity: largest model, latency be damned (transcription).
    Quality,
}

impl QualityTier {
    /// Parse `LI_QUALITY_TIER` (`realtime`/`balanced`/`quality`); unknown/unset =
    /// [`QualityTier::Balanced`].
    pub fn from_env() -> Self {
        match std::env::var("LI_QUALITY_TIER")
            .ok()
            .as_deref()
            .map(str::trim)
        {
            Some("realtime") => QualityTier::Realtime,
            Some("quality") => QualityTier::Quality,
            _ => QualityTier::Balanced,
        }
    }

    /// Default Whisper ggml model path for this tier. Smaller = faster on CPU,
    /// the dominant latency lever. Only used as a fallback when `LI_WHISPER_MODEL`
    /// is unset *and* the file exists, so it never breaks a working startup.
    pub fn whisper_model(self) -> &'static str {
        match self {
            QualityTier::Realtime => "data/models/ggml-base.bin",
            QualityTier::Balanced => "data/models/ggml-small.bin",
            QualityTier::Quality => "data/models/ggml-large-v3-turbo.bin",
        }
    }
}

/// Live conditions the degradation policy reacts to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeConditions {
    pub free_vram_mb: u64,
    pub min_vram_mb: u64,
    pub clone_available: bool,
    pub mesh_reachable: bool,
}

/// The runtime's chosen degradations given the conditions and the requested tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Degradation {
    /// Effective tier after VRAM pressure (may step below the requested one).
    pub tier: QualityTier,
    /// Render with the neutral voice because the clone is unavailable.
    pub force_neutral: bool,
    /// Run the pipeline locally because the mesh is unreachable.
    pub force_local: bool,
}

/// Pure degradation policy: never error, step down instead. Under VRAM pressure
/// (free below the gate) drop to the smallest [`QualityTier::Realtime`] model; an
/// unavailable clone forces neutral; an unreachable mesh forces local.
pub fn degrade(requested: QualityTier, conditions: RuntimeConditions) -> Degradation {
    let tier = if conditions.free_vram_mb < conditions.min_vram_mb {
        QualityTier::Realtime
    } else {
        requested
    };
    Degradation {
        tier,
        force_neutral: !conditions.clone_available,
        force_local: !conditions.mesh_reachable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn healthy() -> RuntimeConditions {
        RuntimeConditions {
            free_vram_mb: 16_000,
            min_vram_mb: 8_000,
            clone_available: true,
            mesh_reachable: true,
        }
    }

    #[test]
    fn whisper_model_shrinks_with_tier() {
        assert!(QualityTier::Realtime.whisper_model().contains("base"));
        assert!(QualityTier::Balanced.whisper_model().contains("small"));
        assert!(QualityTier::Quality.whisper_model().contains("large"));
    }

    #[test]
    fn degrade_keeps_requested_tier_when_healthy() {
        let d = degrade(QualityTier::Quality, healthy());
        assert_eq!(d.tier, QualityTier::Quality);
        assert!(!d.force_neutral);
        assert!(!d.force_local);
    }

    #[test]
    fn vram_pressure_drops_to_realtime() {
        let starved = RuntimeConditions {
            free_vram_mb: 3_200,
            ..healthy()
        };
        assert_eq!(
            degrade(QualityTier::Quality, starved).tier,
            QualityTier::Realtime
        );
        assert_eq!(
            degrade(QualityTier::Balanced, starved).tier,
            QualityTier::Realtime
        );
    }

    #[test]
    fn missing_clone_forces_neutral() {
        let no_clone = RuntimeConditions {
            clone_available: false,
            ..healthy()
        };
        assert!(degrade(QualityTier::Balanced, no_clone).force_neutral);
    }

    #[test]
    fn unreachable_mesh_forces_local() {
        let no_mesh = RuntimeConditions {
            mesh_reachable: false,
            ..healthy()
        };
        assert!(degrade(QualityTier::Balanced, no_mesh).force_local);
    }

    #[test]
    fn default_tier_is_balanced() {
        assert_eq!(QualityTier::default(), QualityTier::Balanced);
    }
}
