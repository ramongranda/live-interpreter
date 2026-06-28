//! NVML-based VRAM orchestration for the GPU server.
//!
//! Placed in its own module (not folded into `desktop.rs`) because that file is under concurrent
//! edits; everything here reuses the stable `desktop::{DesktopConfig, GpuPreflight}` types.
//!
//! Honest scope notes for maintainers:
//!   * We CANNOT mutate a loaded model's quantization in place. Ollama/candle quant is fixed per
//!     model file/tag. [`VramOrchestrator::plan`] therefore *selects a tier* from a configured
//!     ladder; the choice is applied at (re)launch (its `model_ref` is the Ollama tag / GGUF path),
//!     not by flipping FP16->INT4 on a live model.
//!   * We CANNOT disable the NVIDIA driver's UVM/sysmem fallback for a third-party process (Ollama)
//!     from our address space on Linux. The anti-swap guarantee is enforced by *refusing to launch*:
//!     if the smallest tier set still does not fit physical VRAM we reject with the exact missing
//!     megabytes instead of letting the driver silently spill to system RAM.

use crate::desktop::{DesktopConfig, GpuPreflight};
use anyhow::{Context, Result, anyhow, bail};
use nvml_wrapper::Nvml;
use nvml_wrapper::enums::device::UsedGpuMemory;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tokio::process::Command;

/// Per-process GPU memory, in MiB.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GpuProcessMem {
    pub pid: u32,
    pub name: String,
    pub used_mb: u64,
}

/// A point-in-time view of device VRAM. `source` records which probe produced it.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct VramSnapshot {
    pub device_name: String,
    pub total_mb: u64,
    pub free_mb: u64,
    pub used_mb: u64,
    pub utilization_pct: u8,
    pub source: &'static str,
    pub processes: Vec<GpuProcessMem>,
}

/// Real-time VRAM via NVML bindings (libnvidia-ml loaded at runtime). Blocking; call off-runtime.
pub fn nvml_snapshot() -> Result<VramSnapshot> {
    let nvml =
        Nvml::init().map_err(|e| anyhow!("NVML init failed (libnvidia-ml not loadable): {e}"))?;
    let count = nvml
        .device_count()
        .map_err(|e| anyhow!("NVML device_count failed: {e}"))?;
    if count == 0 {
        bail!("NVML reports zero NVIDIA devices");
    }
    let device = nvml
        .device_by_index(0)
        .map_err(|e| anyhow!("NVML device_by_index(0) failed: {e}"))?;
    let mem = device
        .memory_info()
        .map_err(|e| anyhow!("NVML memory_info failed: {e}"))?;
    let device_name = device.name().unwrap_or_else(|_| "NVIDIA GPU".to_string());

    let compute = device.running_compute_processes().unwrap_or_default();
    let graphics = device.running_graphics_processes().unwrap_or_default();
    let mut by_pid: HashMap<u32, GpuProcessMem> = HashMap::new();
    for proc in compute.into_iter().chain(graphics) {
        let used_mb = match proc.used_gpu_memory {
            UsedGpuMemory::Used(bytes) => bytes / (1024 * 1024),
            UsedGpuMemory::Unavailable => 0,
        };
        let entry = by_pid.entry(proc.pid).or_insert_with(|| GpuProcessMem {
            pid: proc.pid,
            name: nvml
                .sys_process_name(proc.pid, 128)
                .unwrap_or_else(|_| format!("pid:{}", proc.pid)),
            used_mb: 0,
        });
        // A pid can appear in both compute and graphics lists; keep the larger reading.
        entry.used_mb = entry.used_mb.max(used_mb);
    }
    let mut processes: Vec<GpuProcessMem> = by_pid.into_values().collect();
    processes.sort_by_key(|p| std::cmp::Reverse(p.used_mb));

    let utilization_pct = device
        .utilization_rates()
        .map(|u| u.gpu.min(100) as u8)
        .unwrap_or(0);

    Ok(VramSnapshot {
        device_name,
        total_mb: mem.total / (1024 * 1024),
        free_mb: mem.free / (1024 * 1024),
        used_mb: mem.used / (1024 * 1024),
        utilization_pct,
        source: "nvml",
        processes,
    })
}

/// Fallback probe when NVML is unavailable (driver/library missing).
async fn nvidia_smi_snapshot() -> Result<VramSnapshot> {
    let summary = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .await
        .context("failed to run nvidia-smi")?;
    if !summary.status.success() {
        bail!("nvidia-smi exited with {}", summary.status);
    }
    let summary_text = String::from_utf8_lossy(&summary.stdout);
    let mut fields = summary_text
        .lines()
        .next()
        .unwrap_or_default()
        .split(',')
        .map(str::trim);
    let device_name = fields.next().unwrap_or("NVIDIA GPU").to_string();
    let total_mb = fields.next().and_then(|v| v.parse().ok()).unwrap_or(0);
    let free_mb = fields.next().and_then(|v| v.parse().ok()).unwrap_or(0);
    let used_mb = fields.next().and_then(|v| v.parse().ok()).unwrap_or(0);
    let utilization_pct = fields
        .next()
        .and_then(|v| v.parse::<u32>().ok())
        .map(|p| p.min(100) as u8)
        .unwrap_or(0);

    let apps = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .await
        .context("failed to run nvidia-smi compute-apps")?;
    let apps_text = String::from_utf8_lossy(&apps.stdout);
    let mut processes: Vec<GpuProcessMem> = apps_text
        .lines()
        .filter_map(|line| {
            let mut parts = line.split(',').map(str::trim);
            Some(GpuProcessMem {
                pid: parts.next()?.parse().ok()?,
                name: parts.next()?.to_string(),
                used_mb: parts.next()?.parse().ok()?,
            })
        })
        .collect();
    processes.sort_by_key(|p| std::cmp::Reverse(p.used_mb));

    Ok(VramSnapshot {
        device_name,
        total_mb,
        free_mb,
        used_mb,
        utilization_pct,
        source: "nvidia-smi",
        processes,
    })
}

/// NVML primary, nvidia-smi fallback.
pub async fn vram_snapshot() -> Result<VramSnapshot> {
    match tokio::task::spawn_blocking(nvml_snapshot).await {
        Ok(Ok(snapshot)) => Ok(snapshot),
        Ok(Err(error)) => {
            tracing::warn!("NVML VRAM probe failed, falling back to nvidia-smi: {error:#}");
            nvidia_smi_snapshot()
                .await
                .context("NVML and nvidia-smi VRAM probes both failed")
        }
        Err(join_error) => {
            tracing::warn!("NVML probe task panicked: {join_error}");
            nvidia_smi_snapshot()
                .await
                .context("NVML and nvidia-smi VRAM probes both failed")
        }
    }
}

/// Decide whether the server may launch given *free* VRAM right now (pure; tested).
pub fn evaluate_preflight(snapshot: &VramSnapshot, min_vram_mb: u64) -> GpuPreflight {
    if snapshot.free_mb >= min_vram_mb {
        GpuPreflight {
            ready: true,
            message: format!(
                "GPU lista: {} MiB libres de {} MiB en {} (minimo {} MiB)",
                snapshot.free_mb, snapshot.total_mb, snapshot.device_name, min_vram_mb
            ),
        }
    } else {
        GpuPreflight {
            ready: false,
            message: retention_message(snapshot, min_vram_mb),
        }
    }
}

/// Semantic error naming which processes are holding the VRAM.
fn retention_message(snapshot: &VramSnapshot, min_vram_mb: u64) -> String {
    let mut holders = snapshot.processes.clone();
    holders.sort_by_key(|p| std::cmp::Reverse(p.used_mb));
    let listed = holders
        .iter()
        .take(3)
        .map(|proc| format!("{} (pid {}) {} MiB", proc.name, proc.pid, proc.used_mb))
        .collect::<Vec<_>>()
        .join(", ");
    let deficit = min_vram_mb.saturating_sub(snapshot.free_mb);
    if listed.is_empty() {
        format!(
            "Servidor GPU bloqueado: {} MiB libres < {} MiB requeridos (faltan {} MiB). \
             No se detectan procesos reteniendo VRAM.",
            snapshot.free_mb, min_vram_mb, deficit
        )
    } else {
        format!(
            "Servidor GPU bloqueado: {} MiB libres < {} MiB requeridos (faltan {} MiB). \
             Reteniendo VRAM: {}.",
            snapshot.free_mb, min_vram_mb, deficit, listed
        )
    }
}

/// Real-time free-VRAM preflight. Call this from `desktop::start_server` instead of the
/// static/total-VRAM check:
/// `let gpu = crate::vram::gpu_preflight_realtime(config.min_server_vram_mb).await;`
pub async fn gpu_preflight_realtime(min_vram_mb: u64) -> GpuPreflight {
    match vram_snapshot().await {
        Ok(snapshot) => evaluate_preflight(&snapshot, min_vram_mb),
        Err(error) => GpuPreflight {
            ready: false,
            message: format!("Servidor GPU bloqueado: no se pudo leer la VRAM ({error:#})"),
        },
    }
}

/// Pure mapper: project a VRAM snapshot into the UI-facing `GpuStatus`. `None`
/// (probe failed) yields a not-capable status with a Spanish gate message.
pub fn build_gpu_status(
    snapshot: Option<&VramSnapshot>,
    min_vram_mb: u64,
) -> crate::types::GpuStatus {
    use crate::types::GpuStatus;
    match snapshot {
        Some(snap) => {
            let preflight = evaluate_preflight(snap, min_vram_mb);
            GpuStatus {
                is_capable: preflight.ready,
                model_name: snap.device_name.clone(),
                vram_free_mb: snap.free_mb,
                vram_total_mb: snap.total_mb,
                utilization_pct: snap.utilization_pct,
                gate_message: preflight.message,
                processes: snap.processes.clone(),
                source: snap.source.to_string(),
            }
        }
        None => GpuStatus {
            is_capable: false,
            model_name: "GPU desconocida".into(),
            vram_free_mb: 0,
            vram_total_mb: 0,
            utilization_pct: 0,
            gate_message: "Servidor GPU bloqueado: no se pudo leer la VRAM".into(),
            processes: Vec::new(),
            source: "unavailable".into(),
        },
    }
}

/// PIDs of the processes this app launched (read from the same pid files as `desktop::pid_alive`).
pub fn our_pids(config: &DesktopConfig) -> HashSet<u32> {
    [
        "live-interpreter.pid",
        "qwen3-tts.pid",
        "live-interpreter-client.pid",
        "live-interpreter-mic.pid",
    ]
    .iter()
    .filter_map(|file| std::fs::read_to_string(config.root.join("data/logs").join(file)).ok())
    .filter_map(|raw| raw.trim().parse::<u32>().ok())
    .collect()
}

/// Telemetry payload pushed to the Tauri UI: total/free device VRAM plus *our* app's exact usage.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct VramTelemetry {
    pub device_name: String,
    pub total_mb: u64,
    pub free_mb: u64,
    pub used_mb: u64,
    pub our_used_mb: u64,
    pub our_processes: Vec<GpuProcessMem>,
    pub source: String,
}

/// Project a snapshot onto this app's own processes (pure; tested).
pub fn app_vram_telemetry(config: &DesktopConfig, snapshot: &VramSnapshot) -> VramTelemetry {
    let pids = our_pids(config);
    let our_processes: Vec<GpuProcessMem> = snapshot
        .processes
        .iter()
        .filter(|proc| pids.contains(&proc.pid))
        .cloned()
        .collect();
    let our_used_mb = our_processes.iter().map(|proc| proc.used_mb).sum();
    VramTelemetry {
        device_name: snapshot.device_name.clone(),
        total_mb: snapshot.total_mb,
        free_mb: snapshot.free_mb,
        used_mb: snapshot.used_mb,
        our_used_mb,
        our_processes,
        source: snapshot.source.to_string(),
    }
}

/// One quantization tier of a model: its VRAM cost and the model reference to launch with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelTier {
    pub label: String,
    pub vram_mb: u64,
    pub model_ref: String,
}

/// A model's quantization ladder, ordered highest-quality first (e.g. fp16 -> q8 -> q4/int4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelLadder {
    pub name: String,
    pub tiers: Vec<ModelTier>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct SelectedModel {
    pub model: String,
    pub tier: String,
    pub vram_mb: u64,
    pub model_ref: String,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub enum VramDecision {
    /// All models fit at their best tier.
    Fit,
    /// Had to drop one or more models to a lower quant to stay under budget.
    Downgraded,
    /// Even the lowest tiers oversubscribe physical VRAM; refuse to launch (anti-swap).
    Reject {
        needed_mb: u64,
        available_mb: u64,
        missing_mb: u64,
    },
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct VramPlan {
    pub selected: Vec<SelectedModel>,
    pub total_mb: u64,
    pub budget_mb: u64,
    pub alert: bool,
    pub decision: VramDecision,
}

/// Strict VRAM budgeter. Caps usage at `budget_fraction` of total (anti-swap headroom) and never
/// exceeds what is physically free, downgrading quantization tiers to fit, or rejecting with the
/// exact missing megabytes when even the smallest configuration cannot fit.
pub struct VramOrchestrator {
    pub budget_fraction: f64,
    pub reserved_mb: u64,
}

impl Default for VramOrchestrator {
    fn default() -> Self {
        Self {
            budget_fraction: 0.90,
            reserved_mb: 512,
        }
    }
}

impl VramOrchestrator {
    pub fn plan(&self, total_mb: u64, free_mb: u64, ladders: &[ModelLadder]) -> Result<VramPlan> {
        for ladder in ladders {
            if ladder.tiers.is_empty() {
                bail!("model ladder '{}' has no quantization tiers", ladder.name);
            }
        }

        // Anti-swap budget: 90% of total VRAM, and never more than what is physically free now.
        let strict_cap =
            ((total_mb as f64 * self.budget_fraction) as u64).saturating_sub(self.reserved_mb);
        let available = strict_cap.min(free_mb);

        let sum_of = |idx: &[usize]| -> u64 {
            ladders
                .iter()
                .zip(idx)
                .map(|(l, &i)| l.tiers[i].vram_mb)
                .sum()
        };
        let mut idx: Vec<usize> = vec![0; ladders.len()];
        let mut total = sum_of(&idx);
        let mut downgraded = false;

        // Greedily downgrade the largest current contributor until it fits or all are at the floor.
        while total > available {
            let candidate = (0..ladders.len())
                .filter(|&pos| idx[pos] + 1 < ladders[pos].tiers.len())
                .max_by_key(|&pos| ladders[pos].tiers[idx[pos]].vram_mb);
            match candidate {
                Some(pos) => {
                    idx[pos] += 1;
                    downgraded = true;
                    total = sum_of(&idx);
                }
                None => break,
            }
        }

        let selected = ladders
            .iter()
            .zip(&idx)
            .map(|(ladder, &i)| {
                let tier = &ladder.tiers[i];
                SelectedModel {
                    model: ladder.name.clone(),
                    tier: tier.label.clone(),
                    vram_mb: tier.vram_mb,
                    model_ref: tier.model_ref.clone(),
                }
            })
            .collect();

        if total > available {
            return Ok(VramPlan {
                selected,
                total_mb: total,
                budget_mb: available,
                alert: true,
                decision: VramDecision::Reject {
                    needed_mb: total,
                    available_mb: available,
                    missing_mb: total - available,
                },
            });
        }

        Ok(VramPlan {
            selected,
            total_mb: total,
            budget_mb: available,
            alert: downgraded,
            decision: if downgraded {
                VramDecision::Downgraded
            } else {
                VramDecision::Fit
            },
        })
    }
}

/// Default model ladders for this deployment. VRAM costs are **estimates** — tune to your hardware,
/// and ensure each `model_ref` is a real Ollama tag / GGUF path / whisper model present on disk.
pub fn default_ladders() -> Vec<ModelLadder> {
    vec![
        ModelLadder {
            name: "llm".into(),
            tiers: vec![
                ModelTier {
                    label: "fp16".into(),
                    vram_mb: 8000,
                    model_ref: "translator:fp16".into(),
                },
                ModelTier {
                    label: "q8".into(),
                    vram_mb: 5000,
                    model_ref: "translator:q8_0".into(),
                },
                ModelTier {
                    label: "q4".into(),
                    vram_mb: 2500,
                    model_ref: "translator:q4_K_M".into(),
                },
            ],
        },
        ModelLadder {
            name: "asr".into(),
            tiers: vec![
                ModelTier {
                    label: "large-v3-turbo".into(),
                    vram_mb: 1600,
                    model_ref: "data/models/ggml-large-v3-turbo.bin".into(),
                },
                ModelTier {
                    label: "small".into(),
                    vram_mb: 600,
                    model_ref: "data/models/ggml-small.bin".into(),
                },
            ],
        },
        ModelLadder {
            name: "tts".into(),
            tiers: vec![ModelTier {
                label: "0.6b".into(),
                vram_mb: 1200,
                model_ref: "Qwen/Qwen3-TTS-12Hz-0.6B-Base".into(),
            }],
        },
    ]
}

/// Env var the selected tier's `model_ref` is injected into at server launch (read by `config.rs`).
fn launch_env_key(model: &str) -> Option<&'static str> {
    match model {
        "llm" => Some("LI_OLLAMA_MODEL"),
        "asr" => Some("LI_WHISPER_MODEL"),
        "tts" => Some("LI_QWEN_TTS_MODEL"),
        _ => None,
    }
}

/// Run the budgeter against a live snapshot and produce the env overrides to launch the stack with.
/// Errors (refuses launch) when even the smallest tier set oversubscribes physical VRAM — closing
/// the budget->launch loop without letting the driver swap to system RAM.
pub fn select_launch_env(
    snapshot: &VramSnapshot,
    ladders: &[ModelLadder],
) -> Result<(VramPlan, Vec<(String, String)>)> {
    let plan = VramOrchestrator::default().plan(snapshot.total_mb, snapshot.free_mb, ladders)?;
    if let VramDecision::Reject {
        needed_mb,
        available_mb,
        missing_mb,
    } = plan.decision
    {
        bail!(
            "Servidor GPU bloqueado: los modelos necesitan {needed_mb} MiB pero solo caben \
             {available_mb} MiB en VRAM fisica (faltan {missing_mb} MiB). Evitando swap a RAM."
        );
    }
    let envs = plan
        .selected
        .iter()
        .filter_map(|sel| {
            launch_env_key(&sel.model).map(|key| (key.to_string(), sel.model_ref.clone()))
        })
        .collect();
    Ok((plan, envs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(total_mb: u64, free_mb: u64) -> VramSnapshot {
        VramSnapshot {
            device_name: "GPU".into(),
            total_mb,
            free_mb,
            used_mb: total_mb - free_mb,
            utilization_pct: 0,
            source: "nvml",
            processes: vec![],
        }
    }

    #[test]
    fn select_launch_env_injects_refs_and_rejects_oversubscription() {
        let (plan, envs) = select_launch_env(&snap(16000, 16000), &default_ladders()).unwrap();
        assert_eq!(plan.decision, VramDecision::Fit);
        assert!(envs.iter().any(|(k, _)| k == "LI_OLLAMA_MODEL"));
        assert!(envs.iter().any(|(k, _)| k == "LI_WHISPER_MODEL"));
        // 2 GiB card cannot hold even the lowest tiers => refuse launch.
        assert!(select_launch_env(&snap(2000, 2000), &default_ladders()).is_err());
    }

    fn ladders() -> Vec<ModelLadder> {
        vec![
            ModelLadder {
                name: "llm".into(),
                tiers: vec![
                    ModelTier {
                        label: "fp16".into(),
                        vram_mb: 8000,
                        model_ref: "qwen:7b-fp16".into(),
                    },
                    ModelTier {
                        label: "q8".into(),
                        vram_mb: 5000,
                        model_ref: "qwen:7b-q8_0".into(),
                    },
                    ModelTier {
                        label: "q4".into(),
                        vram_mb: 2500,
                        model_ref: "qwen:7b-q4_K_M".into(),
                    },
                ],
            },
            ModelLadder {
                name: "asr".into(),
                tiers: vec![ModelTier {
                    label: "large-v3-turbo".into(),
                    vram_mb: 1500,
                    model_ref: "ggml-large-v3-turbo".into(),
                }],
            },
            ModelLadder {
                name: "tts".into(),
                tiers: vec![ModelTier {
                    label: "fp16".into(),
                    vram_mb: 1000,
                    model_ref: "qwen3-tts".into(),
                }],
            },
        ]
    }

    #[test]
    fn best_tiers_fit_under_budget() {
        let plan = VramOrchestrator::default()
            .plan(16000, 16000, &ladders())
            .unwrap();
        assert_eq!(plan.decision, VramDecision::Fit);
        assert!(!plan.alert);
        assert_eq!(plan.selected[0].tier, "fp16");
        assert_eq!(plan.total_mb, 10500);
    }

    #[test]
    fn busy_gpu_forces_quantization_downgrade() {
        // Only 9000 MiB free => fp16 set (10500) does not fit; orchestrator drops the LLM tier.
        let plan = VramOrchestrator::default()
            .plan(16000, 9000, &ladders())
            .unwrap();
        assert_eq!(plan.decision, VramDecision::Downgraded);
        assert!(plan.alert);
        assert!(plan.total_mb <= plan.budget_mb);
        assert_ne!(plan.selected[0].tier, "fp16");
    }

    #[test]
    fn rejects_when_even_lowest_tiers_oversubscribe_physical_vram() {
        // 4000 MiB card: lowest set = 2500 + 1500 + 1000 = 5000 > budget => reject with missing MB.
        let plan = VramOrchestrator::default()
            .plan(4000, 4000, &ladders())
            .unwrap();
        match plan.decision {
            VramDecision::Reject {
                needed_mb,
                available_mb,
                missing_mb,
            } => {
                assert_eq!(needed_mb, 5000);
                assert_eq!(missing_mb, needed_mb - available_mb);
                assert!(missing_mb > 0);
            }
            other => panic!("expected Reject, got {other:?}"),
        }
        assert_eq!(plan.selected[0].tier, "q4");
    }

    #[test]
    fn empty_ladder_is_an_error() {
        let bad = vec![ModelLadder {
            name: "llm".into(),
            tiers: vec![],
        }];
        assert!(
            VramOrchestrator::default()
                .plan(16000, 16000, &bad)
                .is_err()
        );
    }

    #[test]
    fn preflight_blocks_with_named_holder_when_free_below_min() {
        let snapshot = VramSnapshot {
            device_name: "RTX 5060 Ti".into(),
            total_mb: 16000,
            free_mb: 3000,
            used_mb: 13000,
            utilization_pct: 0,
            source: "nvml",
            processes: vec![
                GpuProcessMem {
                    pid: 1234,
                    name: "ollama".into(),
                    used_mb: 9000,
                },
                GpuProcessMem {
                    pid: 5678,
                    name: "python".into(),
                    used_mb: 4000,
                },
            ],
        };
        let preflight = evaluate_preflight(&snapshot, 8000);
        assert!(!preflight.ready);
        assert!(preflight.message.contains("ollama (pid 1234) 9000 MiB"));
        assert!(preflight.message.contains("faltan 5000 MiB"));
    }

    #[test]
    fn preflight_passes_when_free_meets_min() {
        let snapshot = VramSnapshot {
            device_name: "RTX 5060 Ti".into(),
            total_mb: 16000,
            free_mb: 12000,
            used_mb: 4000,
            utilization_pct: 0,
            source: "nvml",
            processes: vec![],
        };
        assert!(evaluate_preflight(&snapshot, 8000).ready);
    }

    #[test]
    fn telemetry_sums_only_our_processes() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("li-vram-test-{unique}"));
        let logs = root.join("data/logs");
        std::fs::create_dir_all(&logs).unwrap();
        std::fs::write(logs.join("live-interpreter.pid"), "1234").unwrap();
        std::fs::write(logs.join("qwen3-tts.pid"), "5678").unwrap();
        let config = DesktopConfig::from_root(root.clone());

        let snapshot = VramSnapshot {
            device_name: "GPU".into(),
            total_mb: 16000,
            free_mb: 4000,
            used_mb: 12000,
            utilization_pct: 0,
            source: "nvml",
            processes: vec![
                GpuProcessMem {
                    pid: 1234,
                    name: "live-interpreter".into(),
                    used_mb: 3000,
                },
                GpuProcessMem {
                    pid: 5678,
                    name: "qwen".into(),
                    used_mb: 5000,
                },
                GpuProcessMem {
                    pid: 9999,
                    name: "someone-else".into(),
                    used_mb: 4000,
                },
            ],
        };
        let telemetry = app_vram_telemetry(&config, &snapshot);
        assert_eq!(telemetry.our_used_mb, 8000);
        assert_eq!(telemetry.our_processes.len(), 2);
        let _ = std::fs::remove_dir_all(root);
    }
}
