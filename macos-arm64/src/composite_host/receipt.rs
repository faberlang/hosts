//! A9/A10 execution receipts: the observable device facts of one descriptor
//! execution (module hash, program-graph hash, launches, transfers,
//! readbacks, syncs, releases) plus the declared logical resource graph and
//! the inter-kernel data-flow edges (R2 consume — never re-derived).

use std::collections::BTreeMap;

use host_coordinator::DeviceBackend;
use serde::{Deserialize, Serialize};

use crate::device_descriptor::{
    DeviceBufferLifetime, DeviceBufferRole, DeviceDataType, DeviceProgramLifetime,
};
use crate::kernel::library_runtime::FusedLibraryDispatchReceipt;

/// A9-style execution receipt: every observable device fact of one
/// descriptor execution (allocations, launches, transfers, syncs, readbacks,
/// module hash, selected hardware) plus the program's lifetime regime and the
/// lifetime-classified buffer sets (S2-4: which buffers are allocated once,
/// which recycled, which read-then-released).
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceExecutionReceipt {
    /// Selected backend.
    pub backend: DeviceBackend,
    /// Selected-hardware name from the admission probe.
    pub device_name: String,
    /// FNV-1a provenance hash of the loaded module image.
    pub module_hash: u64,
    /// Command-buffer submissions this execution. Metal batches every kernel
    /// encode into one submit per step (W8-U1); CUDA submits once per kernel.
    /// `launch_ids` / `launch_entries` still name every encoded kernel.
    pub launches: usize,
    /// Descriptor launch identities dispatched, in exact descriptor order.
    pub launch_ids: Vec<u32>,
    /// Kernel entries dispatched, in exact descriptor launch order.
    pub launch_entries: Vec<String>,
    /// Producer facts for derived fused-library launches. K/V bindings are
    /// carried explicitly so a census cannot masquerade as execution.
    pub fused_library_dispatches: Vec<FusedLibraryDispatchReceipt>,
    /// Host→device copy-ins performed for input slots.
    pub copy_ins: usize,
    /// Declared output readbacks (buffer id → f32 values).
    pub outputs: BTreeMap<u32, Vec<f32>>,
    /// Program-level buffer ids allocated during the run (A9 allocations).
    pub allocated_buffers: Vec<u32>,
    /// Version-keyed buffer allocations carried by the descriptor.
    pub allocated_buffer_versions: Vec<(u32, u32)>,
    /// Temporary PerStep/ObservationPoint handles allocated by this step's
    /// pool checkout. PerProgram allocations are not included.
    pub pool_allocations: usize,
    /// Temporary PerStep/ObservationPoint handles checked out from the
    /// session-scoped pool for this step. PerProgram residency is not a pool
    /// claim.
    pub pool_reuses: usize,
    /// Temporary handles returned to the session-scoped pool at this step's
    /// boundary. A return is not a device free and is therefore not included
    /// in [`DeviceExecutionReceipt::releases`].
    pub pool_returns: usize,
    /// Program execution-lifetime regime (S2-4).
    pub program_lifetime: DeviceProgramLifetime,
    /// `PerProgram` buffer ids: allocated once per session, released at program
    /// end (persist across executions).
    pub per_program_buffers: Vec<u32>,
    /// Version-keyed `PerProgram` allocations.
    pub per_program_buffer_versions: Vec<(u32, u32)>,
    /// `PerStep` buffer ids: recycled at each step boundary.
    pub per_step_buffers: Vec<u32>,
    /// Version-keyed `PerStep` allocations.
    pub per_step_buffer_versions: Vec<(u32, u32)>,
    /// `ObservationPoint` buffer ids: read back and released per execution
    /// (read-then-release).
    pub observation_buffers: Vec<u32>,
    /// Version-keyed `ObservationPoint` allocations.
    pub observation_buffer_versions: Vec<(u32, u32)>,
    /// Declared logical resource graph (A10): every buffer identity +
    /// content version, in first-reference order.
    pub resource_graph: Vec<ReceiptBuffer>,
    /// Declared inter-kernel data-flow edges (A10), producer → consumer
    /// launch ids, in first-reference order. Derived from the launch
    /// sequence with the schema's rule: a buffer's producing launch is its
    /// first Output/InOut reference; consuming launches are later
    /// Input/InOut references — equal to `BufferRegistry::data_flow_pairs`
    /// for constructor-valid programs.
    pub data_flow_edges: Vec<DataFlowEdge>,
    /// Observed real synchronization operations this execution (R9 / W8-U1).
    /// Metal waits once at the step-boundary commit (`wait_until_completed`);
    /// CUDA waits per kernel plus the additive step-boundary `cuCtxSynchronize`.
    pub syncs: usize,
    /// Observed transfers this execution (host→device copy-ins plus
    /// device→host readbacks).
    pub transfers: usize,
    /// Device→host readbacks actually performed (the declared observation
    /// points — observation-only readback, F6).
    pub readbacks: usize,
    /// Observed device buffer frees this execution. Temporary pool returns are
    /// reported separately by `pool_returns`; they are not frees.
    pub releases: usize,
    /// The completion boundary this execution guarantees (R9): the explicit
    /// step-boundary sync after the last launch, at which every declared
    /// observation is valid. Stated exactly — never beyond the explicit
    /// synchronization the host actually performed.
    pub completion_boundary: CompletionBoundary,
    /// SHA-256 receipt of the carried program-graph facts (roots + launches +
    /// dependency edges + buffer semantic identities + observation points +
    /// backend entry-name bytes, under the distinct host-graph domain tag —
    /// OQ1), computed by the host from the descriptor it consumed — the
    /// run/session identity this run executed, distinct from the module
    /// provenance hash. Backend-entry-inclusive (S5A-U3); not a
    /// semantic-identity claim — the A10 complete-program SHA is.
    pub program_graph_hash: String,
    /// Host→device copy-in wall (µs) inside this execution. Packed-weight
    /// upload on the SingleRun spawn path lives here.
    pub copy_in_us: u64,
    /// Kernel encode + step-boundary submit + blocking wait (µs). This is
    /// the true GPU step time (W8-U1: one submit/wait after every encode).
    pub gpu_encode_submit_wait_us: u64,
    /// Observation readback wall (µs) after the step-boundary wait.
    pub readback_us: u64,
    /// Per-encoder GPU timestamps in launch order (µs). Empty when the
    /// step did not sample timestamps. Length matches `launch_entries`
    /// when present.
    pub launch_gpu_us: Vec<u64>,
    /// Per-encoder GPU start times in launch order (µs, relative to the
    /// first encoder start). Empty when unsampled. Length matches
    /// `launch_gpu_us` when present.
    pub launch_gpu_start_us: Vec<u64>,
}

/// The completion boundary of one execution (R9).
///
/// The host states the boundary exactly: completion is guaranteed at the
/// explicit step-boundary synchronization, after the last launch of the
/// ordered sequence. Every declared observation (result) is valid at or
/// after this boundary; the host never claims more than the explicit
/// synchronization it performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionBoundary {
    /// Completion is guaranteed at the explicit step-boundary sync after
    /// launch `after_launch`.
    StepSync { after_launch: u32 },
}

impl CompletionBoundary {
    /// The stable diagnostic statement of the boundary.
    #[must_use]
    pub fn spelling(self) -> String {
        match self {
            Self::StepSync { after_launch } => format!(
                "completion guaranteed at the explicit step-boundary sync after launch {after_launch}"
            ),
        }
    }
}

/// The declared end-of-run value readback of a `RepeatingStep` run (U8/U9
/// repair): the FINAL values of the declared end-of-run set (final forward,
/// final gradients, final trainable params), read back **once** after the
/// step loop at the declared completion boundary (the last step's
/// step-boundary sync). The receipt states the observed values and the
/// transfers the end-of-run boundary performed — readbacks only, zero
/// copy-in (the params are once-init'd at session creation and never
/// re-copied).
#[derive(Debug, Clone, PartialEq)]
pub struct EndOfRunReadback {
    /// The observed end-of-run values, keyed by buffer id (the faber route
    /// maps ids back to the forward / gradient / param names).
    pub values: BTreeMap<u32, Vec<f32>>,
    /// Device→host readbacks performed (exactly the declared end-of-run set
    /// size).
    pub readbacks: usize,
    /// Transfers at the end-of-run boundary (readbacks only — zero
    /// copy-in).
    pub transfers: usize,
}

/// One declared buffer of the program's logical resource graph (A10): the
/// identity facts (id, name, role, lifetime) plus the content version the
/// session executes. Mirrors the schema's `RegistryBuffer` identity.
#[derive(Debug, Clone, PartialEq)]
pub struct ReceiptBuffer {
    /// Program-level buffer identity key.
    pub id: u32,
    /// Logical name (diagnostics; target-neutral).
    pub name: String,
    /// Program-level role.
    pub role: DeviceBufferRole,
    /// Lifetime class (S2-4).
    pub lifetime: DeviceBufferLifetime,
    /// Element type of this content version.
    pub element_ty: DeviceDataType,
    /// Element count of this content version.
    pub element_count: u64,
    /// Content version executed (the codec carries one version, 1).
    pub version: u32,
}

/// One declared inter-kernel data-flow edge (A10): a buffer content version
/// produced by launch `producer` and consumed by launch `consumer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataFlowEdge {
    /// Buffer whose content flows.
    pub buffer_id: u32,
    /// Content version that flows.
    pub version: u32,
    /// Producing launch id (1-based).
    pub producer: u32,
    /// Consuming launch id (1-based).
    pub consumer: u32,
}

/// F1 wall-decomposition evidence for one timing or duration value.
///
/// `NotMeasured` is a real schema value, not a zero sentinel. A timestamp
/// that the host cannot observe therefore stays visibly absent instead of
/// being defaulted to a plausible number. `Derived` is reserved for values
/// computed from carried timestamps or other receipt facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status")]
pub enum KvCacheMeasurement {
    /// Directly observed on the host or device timeline.
    #[serde(rename = "measured")]
    Measured { value_us: u64 },
    /// Computed from other receipt facts, never presented as a measurement.
    #[serde(rename = "derived")]
    Derived { value_us: u64 },
    /// No observation exists for this value.
    #[serde(rename = "not_measured")]
    NotMeasured,
}

impl KvCacheMeasurement {
    /// Construct a directly measured microsecond value.
    #[must_use]
    pub const fn measured(value_us: u64) -> Self {
        Self::Measured { value_us }
    }

    /// Construct a value derived from other receipt facts.
    #[must_use]
    pub const fn derived(value_us: u64) -> Self {
        Self::Derived { value_us }
    }

    /// Mark a value as absent without manufacturing a numeric default.
    #[must_use]
    pub const fn not_measured() -> Self {
        Self::NotMeasured
    }
}

/// Start/end timeline facts and the associated F1 duration fact for one
/// lifecycle segment. Each timestamp is independently explicit, so a missing
/// start or end never becomes `0` and cannot be mistaken for an observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCacheTimingSpan {
    /// Monotonic start timestamp in microseconds, or [`KvCacheMeasurement::NotMeasured`].
    pub start_us: KvCacheMeasurement,
    /// Monotonic end timestamp in microseconds, or [`KvCacheMeasurement::NotMeasured`].
    pub end_us: KvCacheMeasurement,
    /// Duration classified as measured, derived, or not measured.
    pub duration_us: KvCacheMeasurement,
}

impl KvCacheTimingSpan {
    /// An explicit all-missing span for a phase that was not observed.
    #[must_use]
    pub const fn not_measured() -> Self {
        Self {
            start_us: KvCacheMeasurement::NotMeasured,
            end_us: KvCacheMeasurement::NotMeasured,
            duration_us: KvCacheMeasurement::NotMeasured,
        }
    }
}

/// Setup or steady-state timing split. The four spans are deliberately kept
/// separate because `gpu_body`, encode, submit, and wait are different F1 wall
/// terms and must not be collapsed into the legacy `kernel_us` proxy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCachePhaseTiming {
    /// GPU/device body timeline for this phase.
    pub gpu_body: KvCacheTimingSpan,
    /// Host encoding timeline for this phase.
    pub encode: KvCacheTimingSpan,
    /// Command-buffer or driver submit timeline for this phase.
    pub submit: KvCacheTimingSpan,
    /// Blocking device wait timeline for this phase.
    pub wait: KvCacheTimingSpan,
}

impl KvCachePhaseTiming {
    /// An explicit all-missing phase for setup or steady state.
    #[must_use]
    pub const fn not_measured() -> Self {
        let missing = KvCacheTimingSpan::not_measured();
        Self {
            gpu_body: missing,
            encode: missing,
            submit: missing,
            wait: missing,
        }
    }
}

/// Lifecycle facts that explain whether a claimed steady-state route stayed
/// resident. Counts and byte volumes are carried separately from timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCacheLifecycleReceipt {
    /// Modules loaded again after the resident route was prepared.
    pub module_reloads: u64,
    /// Persistent/per-program allocations made again after preparation.
    pub persistent_reallocations: u64,
    /// Weight uploads performed by the route.
    pub weight_uploads: u64,
    /// Bytes copied from an already-resident old KV prefix.
    pub old_prefix_copy_bytes: u64,
    /// Bytes written when the full KV cache was cleared.
    pub full_cache_clear_bytes: u64,
}

impl KvCacheLifecycleReceipt {
    /// Zero lifecycle events and byte volumes for a newly prepared route.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            module_reloads: 0,
            persistent_reallocations: 0,
            weight_uploads: 0,
            old_prefix_copy_bytes: 0,
            full_cache_clear_bytes: 0,
        }
    }
}

/// F4H1 host timing/lifecycle schema for a KV-cache run.
///
/// Setup and steady state each retain independent body, encode, submit, and
/// wait timestamps. Slack is classified with the same measured/derived/
/// not-measured vocabulary as F1. F4H1 only defines the receipt; F4H2 later
/// populates it from `session.rs` without changing this shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCacheTimingReceipt {
    /// One-time setup-phase timing, including any admission or preparation
    /// work that the later steady-state route must not hide.
    pub setup_phase: KvCachePhaseTiming,
    /// One steady-state step's timing split.
    pub steady_state: KvCachePhaseTiming,
    /// Explicitly unattributed wall between the named F1 terms.
    pub slack_us: KvCacheMeasurement,
    /// Reload, reallocation, upload, prefix-copy, and full-clear facts.
    pub lifecycle: KvCacheLifecycleReceipt,
}

impl KvCacheTimingReceipt {
    /// An explicit schema value with every timing fact absent and lifecycle
    /// counters set to their measured zero-event value. This is not a
    /// `Default` implementation: callers must opt into the missing receipt.
    #[must_use]
    pub const fn not_measured() -> Self {
        Self {
            setup_phase: KvCachePhaseTiming::not_measured(),
            steady_state: KvCachePhaseTiming::not_measured(),
            slack_us: KvCacheMeasurement::NotMeasured,
            lifecycle: KvCacheLifecycleReceipt::zero(),
        }
    }
}
