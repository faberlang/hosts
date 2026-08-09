//! A9/A10 execution receipts: the observable device facts of one descriptor
//! execution (module hash, program-graph hash, launches, transfers,
//! readbacks, syncs, releases) plus the declared logical resource graph and
//! the inter-kernel data-flow edges (R2 consume — never re-derived).

use std::collections::BTreeMap;

use faber::device::DeviceBackend;

use crate::device_descriptor::{
    DeviceBufferLifetime, DeviceBufferRole, DeviceDataType, DeviceProgramLifetime,
};

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
    /// Launches dispatched (each synchronizes internally).
    pub launches: usize,
    /// Descriptor launch identities dispatched, in exact descriptor order.
    pub launch_ids: Vec<u32>,
    /// Kernel entries dispatched, in exact descriptor launch order.
    pub launch_entries: Vec<String>,
    /// Host→device copy-ins performed for input slots.
    pub copy_ins: usize,
    /// Declared output readbacks (buffer id → f32 values).
    pub outputs: BTreeMap<u32, Vec<f32>>,
    /// Program-level buffer ids allocated during the run (A9 allocations).
    pub allocated_buffers: Vec<u32>,
    /// Version-keyed buffer allocations carried by the descriptor.
    pub allocated_buffer_versions: Vec<(u32, u32)>,
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
    /// Observed real synchronization operations this execution (R9): one per
    /// launch (a launch synchronizes internally) plus the explicit
    /// step-boundary barrier, counted only where the backend's `sync()`
    /// actually performs a device synchronization (a no-op step sync is not
    /// an actual synchronization event and is not counted).
    pub syncs: usize,
    /// Observed transfers this execution (host→device copy-ins plus
    /// device→host readbacks).
    pub transfers: usize,
    /// Device→host readbacks actually performed (the declared observation
    /// points — observation-only readback, F6).
    pub readbacks: usize,
    /// Observed buffer releases this execution (read-then-release plus the
    /// step-boundary `PerStep` recycle).
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
