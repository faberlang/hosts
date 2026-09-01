//! Composite product host: stdio + kernel effects + device sessions (S1-4).
//!
//! The frozen host-ownership contract (architecture record §5, N1.5) gives
//! **hosts** the native Metal/CUDA sessions and the **composite host** that
//! composes stdio + kernel effects + device sessions, and gives **faber** the
//! host factory that applies one host-construction policy across the
//! FHIR/FMIR/`fmir-bin`/image-runner routes. This module is that composite
//! host and the policy it is built by.
//!
//! # The one host-construction policy
//!
//! Every product route constructs its host through the **same** decision,
//! [`resolve_device_selection`], and then either runs CPU-only or carries a
//! device session. There is exactly one policy; the route only supplies its
//! selection request and whether it carries a device program:
//!
//! | Route | Device program? | Construction |
//! | --- | --- | --- |
//! | FHIR (source) | never (source package; no device section) | explicit backend request is **rejected** (`E_NO_DEVICE_PROGRAM`) — the unsupported route is refused, never silently CPU; `auto` → CPU-only host unchanged |
//! | FMIR (source-built image) | yes, when the package carries one | composite host with the resolved backend |
//! | `fmir-bin` (binary image) | yes, when the package carries one | composite host with the resolved backend |
//! | image-runner (`run_fmir_image_bytes_with_stdio`) | yes, when the package carries one | composite host with the resolved backend |
//!
//! Resolution (N1.1/N1.4):
//! - `auto` + no device program → CPU-only route, unchanged;
//! - `auto` + device program → exactly one admitted backend is selected; zero
//!   or more than one fails closed (`E_BACKEND_UNAVAILABLE`) with the
//!   candidates named and the explicit flag required;
//! - explicit `metal`/`cuda` + no device program → `E_NO_DEVICE_PROGRAM`
//!   ("package has no device program");
//! - explicit backend not admitted on the machine → `E_BACKEND_UNAVAILABLE`
//!   **before any launch**; an explicit GPU request never silently falls back.
//!
//! The faber host factory (S1-5, a separate routed patch) calls
//! [`CompositeHost::new`] with the route's selection; this module owns the
//! host-side component and the policy decision itself.
//!
//! # Uniform virtual-partition admission (MD3H-H3)
//!
//! Every device-carrying product host admits through one
//! [`VirtualDevicePartition`] and executes a bound plan. Construction is
//! discover → admit [`VirtualDevicePartition::implicit_local`] (N=1) → bind
//! [`BoundPlanKind::ImplicitLocal`] → execute the bound session. There is no
//! partition-free product construction: the one-runtime field is the M=1
//! member of a [`DeviceRuntimeSet`]. [`DeviceSelection`] remains backend
//! kind only (`Auto` / `Metal` / `Cuda`); ranks never enter it. N=1 stays
//! coordinator-free — empty communication graph, no extra copies, no
//! `ExecutionTransaction`.
//!
//! # A8: device execution is not provider routing
//!
//! The composite host holds the frame/kernel-effects host ([`HostKernel`])
//! and the device component ([`CompositeDeviceState`]) as **separate fields**.
//! Kernel effects (aleator/tempus/consolum/solum/processus/http + host echo) route
//! through the kernel; device sessions are never exposed as provider routes.
//! [`CompositeHost::execute_descriptor`] drives the device session directly
//! and reports an A9-style receipt (selected hardware, module hash, launches,
//! transfers, readbacks, allocations).

pub mod inference_state;
pub mod invocation_binding;
mod invocation_program;
mod paired_session;
mod receipt;
mod residency;
mod session;

pub use inference_state::{
    CandidateRows, CursorFacts, E_INVALID_ARGS, E_KV_OVERFLOW, E_KV_PHASE, E_KV_POISONED,
    E_KV_RELEASED, E_KV_STALE, FailureOutcome, FailureStage, InferenceSessionState, InvocationMode,
    InvocationTransaction, PlannedInvocation, ResetReceipt, SequencePhase, SessionError,
    SessionInspection, VerificationCommit,
};
pub use paired_session::PairedProgramSession;
pub use receipt::{
    CompletionBoundary, DataFlowEdge, DeviceExecutionReceipt, EndOfRunReadback,
    KvCacheLifecycleReceipt, KvCacheMeasurement, KvCachePhaseTiming, KvCacheTimingReceipt,
    KvCacheTimingSpan, ReceiptBuffer,
};
pub use session::{
    DeviceByteBuffer, KvCacheBindingSession, PreparedResidentSession, PreparedSessionCounters,
    PreparedSessionReceipt, ProgramSession,
};

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

pub use host_coordinator::DeviceBackend;
pub use host_coordinator::bound_plan::{BoundDistributedPlan, BoundPlanKind};

use host_coordinator::bound_plan::{
    AdmittedLogicalPlan, LogicalPartitionId, PartitionBinding, bind,
};
use host_coordinator::device_identity::{DeviceHealthGeneration, DeviceOrdinal, PhysicalDeviceId};
use host_coordinator::device_set::DeviceSet;
use host_coordinator::discovery::{
    ComputeCapability, DeviceCapabilities, DeviceDiscoveryEntry, DeviceDiscoverySnapshot,
    DeviceHealth, DeviceMemory, DtypeSurface, ProbeProvenance,
};
use host_coordinator::partition::{
    FixtureIdentityClass, PartitionBudgetLedger, SafePhysicalLimit, TransportClass,
    VirtualDevicePartition, VirtualDevicePartitionId,
};

use crate::device_descriptor::{DeviceDescriptor, errors as descriptor_errors, sha256_hex};
use crate::device_host::DeviceRuntime;
use crate::device_runtime_set::DeviceRuntimeSet;

use self::invocation_binding::RopeConfig;
use crate::Frame;
use crate::kernel::{HostError, HostKernel, HostResult};
use crate::manifest::CapabilityManifest;

/// Logical-plan hash of the host-synthesized N=1 implicit-local plan.
/// Physical ids never enter this domain (MD-A1/A15/A17).
#[must_use]
pub fn implicit_local_n1_logical_hash() -> String {
    format!("sha256:{}", sha256_hex(b"md3h-implicit-local-n1"))
}

/// Opaque logical partition identity of the N=1 implicit-local plan.
pub const IMPLICIT_LOCAL_PARTITION: &str = "implicit-local";

/// Product request for host backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceSelection {
    /// Resolve against the admitted native backends.
    Auto,
    /// Select Metal explicitly.
    Metal,
    /// Select CUDA explicitly.
    Cuda,
}

impl DeviceSelection {
    /// Stable command and diagnostic spelling.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }

    /// The physical backend named by an explicit selection.
    #[must_use]
    pub const fn backend(self) -> Option<DeviceBackend> {
        match self {
            Self::Auto => None,
            Self::Metal => Some(DeviceBackend::Metal),
            Self::Cuda => Some(DeviceBackend::Cuda),
        }
    }
}

/// One deliberate host-construction request (see module docs for the policy).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompositeHostConfig {
    /// The backend selection request (CLI `--backend` or manifest default).
    pub selection: DeviceSelection,
    /// Whether the route's package carries a device program. `false` for FHIR
    /// source routes and for payload-less `auto` runs.
    pub requires_device: bool,
}

impl CompositeHostConfig {
    /// CPU-only construction (no device program on this route).
    #[must_use]
    pub fn cpu() -> Self {
        Self {
            selection: DeviceSelection::Auto,
            requires_device: false,
        }
    }

    /// Construction with a device selection request.
    #[must_use]
    pub fn device(selection: DeviceSelection) -> Self {
        Self {
            selection,
            requires_device: true,
        }
    }
}

/// The device component of the composite host.
pub enum CompositeDeviceState {
    /// No device session (CPU-only route).
    CpuOnly,
    /// An admitted device session: the M=1 runtime-set member bound by the
    /// implicit-local plan, plus its selected-hardware name (A9 receipts).
    Device {
        /// Physical sessions by identity. Product N=1 is always M=1.
        set: DeviceRuntimeSet,
        /// Discover → admit → bind result. Product N=1 is always
        /// [`BoundPlanKind::ImplicitLocal`].
        bound_plan: BoundDistributedPlan,
        /// Human-readable selected-hardware name from the admission probe.
        device_name: String,
    },
}

/// Probe the machine for admitted native backends (discovery receipts).
#[must_use]
pub fn admitted_backends() -> Vec<DeviceBackend> {
    let mut admitted = Vec::new();
    if crate::metal_host::probe_metal_environment().admitted {
        admitted.push(DeviceBackend::Metal);
    }
    if crate::cuda_host::probe_cuda_environment().admitted {
        admitted.push(DeviceBackend::Cuda);
    }
    admitted
}

/// **The one host-construction decision** (N1.1 auto rule + N1.4 table).
///
/// Pure over the injected `admitted` list so every branch is testable without
/// hardware. Returns `None` for the CPU-only route and `Some(backend)` when a
/// device session must be constructed; every failure is a structured
/// diagnostic and never a CPU fallback.
///
/// # Errors
/// - `E_BACKEND_UNAVAILABLE` — `auto` cannot resolve (zero or more than one
///   admitted backend) or an explicit backend is not admitted;
/// - `E_NO_DEVICE_PROGRAM` — an explicit backend was requested on a route
///   whose package carries no device program.
pub fn resolve_device_selection(
    selection: DeviceSelection,
    requires_device: bool,
    admitted: &[DeviceBackend],
) -> HostResult<Option<DeviceBackend>> {
    match selection {
        DeviceSelection::Auto if !requires_device => Ok(None),
        DeviceSelection::Auto => match admitted {
            [] => Err(descriptor_errors::backend_unavailable(
                "device backend `auto` could not resolve: no native backend is admitted on this machine",
            )),
            [only] => Ok(Some(*only)),
            _ => Err(descriptor_errors::backend_unavailable(format!(
                "device backend `auto` could not resolve: multiple backends are admitted ({}) on this machine; pass an explicit --backend",
                admitted
                    .iter()
                    .map(|backend| backend.spelling())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))),
        },
        explicit => {
            let Some(backend) = explicit.backend() else {
                return Err(descriptor_errors::no_device_program(
                    "invalid device selection",
                ));
            };
            if !requires_device {
                return Err(descriptor_errors::no_device_program(format!(
                    "package has no device program; cannot construct a host for backend `{}`",
                    backend.spelling()
                )));
            }
            if admitted.contains(&backend) {
                Ok(Some(backend))
            } else {
                Err(descriptor_errors::backend_unavailable(format!(
                    "requested backend `{}` is not admitted on this machine; an explicit GPU request never silently falls back",
                    backend.spelling()
                )))
            }
        }
    }
}

/// Composite host: stdio + kernel effects (via [`HostKernel`]) composed with
/// an optional device session (A8).
pub struct CompositeHost {
    kernel: HostKernel,
    device: CompositeDeviceState,
}

impl CompositeHost {
    /// Construct the composite host under the one host-construction policy:
    /// resolve the selection against the live admission probes, then discover
    /// → admit `implicit_local` → bind [`BoundPlanKind::ImplicitLocal`] over
    /// an M=1 [`DeviceRuntimeSet`], or run CPU-only.
    ///
    /// # Errors
    /// - `E_BACKEND_UNAVAILABLE` — the resolved backend cannot be opened or
    ///   does not enumerate exactly one physical device;
    /// - `E_NO_DEVICE_PROGRAM` — explicit backend on a payload-less route.
    pub fn new(config: CompositeHostConfig) -> HostResult<Self> {
        let admitted = admitted_backends();
        let resolved =
            resolve_device_selection(config.selection, config.requires_device, &admitted)?;
        let device = match resolved {
            None => CompositeDeviceState::CpuOnly,
            Some(backend) => {
                let snapshot = discover_backend(backend)?;
                let identity = select_singleton_device(&snapshot, backend)?;
                let set = DeviceRuntimeSet::open_live([identity.clone()])?;
                admit_implicit_local(
                    set,
                    snapshot,
                    identity,
                    FixtureIdentityClass::Virtual,
                    backend_device_name(backend),
                )?
            }
        };
        Ok(Self {
            kernel: HostKernel::new(),
            device,
        })
    }

    /// Inject a device session (sequencing tests only; the driver fakes
    /// bypass live probes). Still admits `implicit_local` against a synthetic
    /// snapshot — partition-free product construction is deleted. Product
    /// construction always goes through [`CompositeHost::new`].
    pub fn with_device(runtime: DeviceRuntime, device_name: impl Into<String>) -> HostResult<Self> {
        let device_name = device_name.into();
        let identity = injected_physical_id(runtime.backend(), &device_name);
        let snapshot = synthetic_snapshot(identity.clone());
        let set = DeviceRuntimeSet::from_members([(identity.clone(), runtime)])?;
        Ok(Self {
            kernel: HostKernel::new(),
            device: admit_implicit_local(
                set,
                snapshot,
                identity,
                FixtureIdentityClass::Synthetic,
                device_name,
            )?,
        })
    }

    /// The kernel-effects host (stdio + provider routing).
    #[must_use]
    pub fn kernel(&self) -> &HostKernel {
        &self.kernel
    }

    /// The kernel-effects host (mutable).
    #[must_use]
    pub fn kernel_mut(&mut self) -> &mut HostKernel {
        &mut self.kernel
    }

    /// The live device session, when the host carries one — the M=1 member
    /// bound by the implicit-local plan.
    #[must_use]
    pub fn device(&self) -> Option<&DeviceRuntime> {
        let CompositeDeviceState::Device {
            set, bound_plan, ..
        } = &self.device
        else {
            return None;
        };
        set.get(bound_implicit_device(bound_plan)?)
    }

    /// The live device session (mutable).
    #[must_use]
    pub fn device_mut(&mut self) -> Option<&mut DeviceRuntime> {
        let id = match &self.device {
            CompositeDeviceState::Device { bound_plan, .. } => {
                bound_implicit_device(bound_plan).cloned()
            }
            CompositeDeviceState::CpuOnly => None,
        }?;
        match &mut self.device {
            CompositeDeviceState::Device { set, .. } => set.get_mut(&id),
            CompositeDeviceState::CpuOnly => None,
        }
    }

    /// The bound plan this device host admitted. `None` on the CPU-only route.
    #[must_use]
    pub fn bound_plan(&self) -> Option<&BoundDistributedPlan> {
        match &self.device {
            CompositeDeviceState::Device { bound_plan, .. } => Some(bound_plan),
            CompositeDeviceState::CpuOnly => None,
        }
    }

    /// The runtime set whose M=1 member is the product session. `None` on
    /// the CPU-only route.
    #[must_use]
    pub fn runtime_set(&self) -> Option<&DeviceRuntimeSet> {
        match &self.device {
            CompositeDeviceState::Device { set, .. } => Some(set),
            CompositeDeviceState::CpuOnly => None,
        }
    }

    /// Fail unless this host admitted exactly one `implicit_local` partition
    /// and bound [`BoundPlanKind::ImplicitLocal`]. A run that bypassed
    /// partition admission cannot execute.
    pub fn require_implicit_local(&self) -> HostResult<&BoundDistributedPlan> {
        let plan = self.bound_plan().ok_or_else(|| {
            descriptor_errors::no_device_program(
                "composite host has no bound plan; partition-free device construction is deleted",
            )
        })?;
        if !plan.is_degenerate() {
            return Err(HostError::invalid_args(
                "N=1 product execution requires BoundPlanKind::ImplicitLocal",
            ));
        }
        match plan.kind() {
            BoundPlanKind::ImplicitLocal {
                virtual_partition: Some(partition),
                ..
            } if partition.is_active() => Ok(plan),
            BoundPlanKind::ImplicitLocal {
                virtual_partition: None,
                ..
            } => Err(HostError::invalid_args(
                "implicit-local bind is missing the admitted VirtualDevicePartition",
            )),
            BoundPlanKind::ImplicitLocal { .. } => Err(HostError::invalid_args(
                "implicit-local VirtualDevicePartition is not active",
            )),
            BoundPlanKind::Distributed { .. } => Err(HostError::invalid_args(
                "N=1 product execution requires BoundPlanKind::ImplicitLocal",
            )),
        }
    }

    /// Whether the composite host carries an admitted device session.
    #[must_use]
    pub fn is_device_active(&self) -> bool {
        matches!(self.device, CompositeDeviceState::Device { .. })
    }

    /// Route a frame through stdio + kernel effects (provider routing never
    /// sees the device component — A8).
    #[must_use]
    pub fn route(&self, request: &Frame) -> Frame {
        self.kernel.route(request)
    }

    /// Discovery receipt: the capability manifest of the kernel-effects host,
    /// with `host` derived from the admitted device backend.
    #[must_use]
    pub fn manifest(&self) -> CapabilityManifest {
        let host = match self.device() {
            Some(runtime) => match runtime.backend() {
                DeviceBackend::Cuda => CapabilityManifest::HOST_CUDA_LINUX,
                DeviceBackend::Metal => CapabilityManifest::HOST_MACOS_ARM64,
            },
            None => CapabilityManifest::HOST_MACOS_ARM64,
        };
        self.kernel.manifest_for(host)
    }

    /// Create a program-scoped session for one device program (S2-1).
    ///
    /// The session owns the module (loaded once) and every `PerProgram` buffer
    /// (allocated once at creation, persisting across executions); `PerStep`
    /// and `ObservationPoint` buffers are allocated per execution and recycled
    /// / read-then-released at each step boundary (S2-4). It survives
    /// repeated executions on the same session without reloading or
    /// re-allocating `PerProgram` buffers. Call
    /// [`ProgramSession::teardown`] to release every handle in order.
    ///
    /// A `RepeatingStep` session (S5-U6, the training-loop surface) runs
    /// through [`ProgramSession::init_params`] (once-init HostProvided
    /// params) + [`ProgramSession::execute_step`]; a `SingleRun` session
    /// runs through [`ProgramSession::execute`].
    ///
    /// # Errors
    /// - `E_NO_DEVICE_PROGRAM` — no device session on this host;
    /// - `E_DEVICE_DESCRIPTOR` — wrong-backend or structurally bad descriptor;
    /// - `E_DEVICE_ABI_MISMATCH` / `E_DEVICE_DTYPE_MISMATCH` /
    ///   `E_DEVICE_SHAPE_MISMATCH` — typed descriptor conflicts;
    /// - session-level failures (module load, allocation) bubble through.
    pub fn create_program_session(
        &mut self,
        descriptor: &DeviceDescriptor,
    ) -> HostResult<ProgramSession<'_>> {
        self.require_implicit_local()?;
        let device_name = self.device_name().to_owned();
        let runtime = self.device_mut().ok_or_else(|| {
            descriptor_errors::no_device_program(
                "composite host has no device session; a device descriptor cannot execute",
            )
        })?;
        // No authoritative attention axes on this generic single-program
        // path: a `_QkvProjection` entry fails closed at plan recognition
        // rather than degrading to the Q-only carrier (FQ-1).
        ProgramSession::new(runtime, descriptor, device_name, None)
    }

    /// Prepare paired prefill and scalar-decode programs over one runtime and
    /// one semantic PerProgram model/cache owner. The returned executor
    /// selects its program explicitly from each v2 invocation mode.
    pub fn prepare_paired_session(
        &mut self,
        prefill: &DeviceDescriptor,
        decode: &DeviceDescriptor,
        prompt_tokens: Vec<u32>,
        rope: RopeConfig,
        weights: &BTreeMap<u32, Vec<f32>>,
        byte_weights: &BTreeMap<u32, DeviceByteBuffer>,
        model_identity: impl Into<String>,
        session_identity: impl Into<String>,
    ) -> HostResult<PairedProgramSession<'_>> {
        self.require_implicit_local()?;
        let device_name = self.device_name().to_owned();
        let runtime = self.device_mut().ok_or_else(|| {
            descriptor_errors::no_device_program(
                "composite host has no device session; paired programs cannot execute",
            )
        })?;
        PairedProgramSession::prepare(
            runtime,
            prefill,
            decode,
            prompt_tokens,
            rope,
            weights,
            byte_weights,
            model_identity.into(),
            session_identity.into(),
            device_name,
        )
    }

    /// Execute a typed device descriptor through the device session.
    ///
    /// Single-run convenience over the program session (S2-1): creates a
    /// session, executes the ordered launch sequence once, and tears down
    /// releasing every handle. Fail-before-launch semantics are unchanged
    /// from S1-4.
    ///
    /// # Errors
    /// - `E_NO_DEVICE_PROGRAM` — no device session on this host;
    /// - `E_DEVICE_DESCRIPTOR` — wrong-backend or structurally bad descriptor;
    /// - `E_DEVICE_ABI_MISMATCH` / `E_DEVICE_DTYPE_MISMATCH` /
    ///   `E_DEVICE_SHAPE_MISMATCH` / `E_DEVICE_ENTRY_MISMATCH` — typed
    ///   descriptor/entry/shape conflicts (see [`DeviceDescriptor::validate`]);
    /// - session-level failures bubble through unchanged.
    ///
    /// Error-path teardown (S2-3): a failed execution releases every handle
    /// inside the session before the error escapes; the session is closed and
    /// is dropped without a second teardown.
    pub fn execute_descriptor(
        &mut self,
        descriptor: &DeviceDescriptor,
        inputs: &BTreeMap<u32, Vec<f32>>,
    ) -> HostResult<DeviceExecutionReceipt> {
        let mut session = self.create_program_session(descriptor)?;
        match session.execute(inputs) {
            Ok(receipt) => {
                session.teardown()?;
                Ok(receipt)
            }
            Err(error) => {
                // The session's error path already released every handle
                // (release-on-error, S2-3); tearing down again would double
                // release. The closed session is dropped as-is.
                Err(error)
            }
        }
    }

    /// Prepare a resident session for one admitted model (E03-U1): validate
    /// the descriptor admits the prepared-session shape (a `RepeatingStep`
    /// program with once-init `HostProvided` `PerProgram` weights; a model may
    /// also declare device-resident `ZeroFill` state), create the underlying
    /// [`ProgramSession`] (module loaded once, `PerProgram` buffers allocated
    /// once), and once-init the weights so they stay device-resident. The returned
    /// [`PreparedResidentSession`] reuses the resident weights across
    /// repeated decode executions and prompt-scoped resets without reload or
    /// `PerProgram` re-allocation, and reports prepare/reuse/reset/release
    /// counts.
    ///
    /// # Errors
    /// - `E_NO_DEVICE_PROGRAM` — no device session on this host;
    /// - `E_DEVICE_DESCRIPTOR` — wrong-backend, structurally invalid, or not
    ///   a prepared-session shape;
    /// - session-level failures (module load, allocation, once-init) bubble
    ///   through.
    pub fn prepare_resident_session(
        &mut self,
        descriptor: &DeviceDescriptor,
        weights: &BTreeMap<u32, Vec<f32>>,
    ) -> HostResult<PreparedResidentSession<'_>> {
        self.require_implicit_local()?;
        let device_name = self.device_name().to_owned();
        let runtime = self.device_mut().ok_or_else(|| {
            descriptor_errors::no_device_program(
                "composite host has no device session; a device descriptor cannot be prepared",
            )
        })?;
        PreparedResidentSession::prepare(runtime, descriptor, weights, device_name)
    }

    fn device_name(&self) -> &str {
        match &self.device {
            CompositeDeviceState::CpuOnly => "none",
            CompositeDeviceState::Device { device_name, .. } => device_name,
        }
    }
}

/// Selected-hardware name for A9 receipts from the admission probe.
fn backend_device_name(backend: DeviceBackend) -> String {
    match backend {
        DeviceBackend::Metal => crate::metal_host::probe_metal_environment()
            .mtl_device
            .unwrap_or_else(|| "metal".to_owned()),
        DeviceBackend::Cuda => crate::cuda_host::probe_cuda_environment()
            .nvidia_smi
            .unwrap_or_else(|| "cuda".to_owned()),
    }
}

fn discover_backend(backend: DeviceBackend) -> HostResult<DeviceDiscoverySnapshot> {
    let probe_utc_nanos = probe_utc_nanos();
    match backend {
        DeviceBackend::Metal => crate::metal_host::discover_metal_snapshot(probe_utc_nanos),
        DeviceBackend::Cuda => crate::cuda_host::discover_cuda_snapshot(probe_utc_nanos),
    }
}

fn probe_utc_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

fn select_singleton_device(
    snapshot: &DeviceDiscoverySnapshot,
    backend: DeviceBackend,
) -> HostResult<PhysicalDeviceId> {
    let matching: Vec<&DeviceDiscoveryEntry> = snapshot
        .devices()
        .values()
        .filter(|entry| entry.backend() == backend)
        .collect();
    match matching.as_slice() {
        [only] => Ok(only.identity.clone()),
        [] => Err(descriptor_errors::backend_unavailable(format!(
            "requested backend `{}` enumerated no physical devices",
            backend.spelling()
        ))),
        _ => Err(descriptor_errors::backend_unavailable(format!(
            "requested backend `{}` enumerated {} physical devices; DeviceSelection names backend kind only and cannot choose among ranks",
            backend.spelling(),
            matching.len()
        ))),
    }
}

fn injected_physical_id(backend: DeviceBackend, device_name: &str) -> PhysicalDeviceId {
    match backend {
        DeviceBackend::Metal => PhysicalDeviceId::metal(device_name),
        DeviceBackend::Cuda => PhysicalDeviceId::cuda(device_name, None),
    }
}

fn synthetic_snapshot(identity: PhysicalDeviceId) -> DeviceDiscoverySnapshot {
    DeviceDiscoverySnapshot::from_enumerated(
        0,
        [DeviceDiscoveryEntry {
            ordinal: DeviceOrdinal::new(0),
            identity,
            device_model: Some("synthetic".to_owned()),
            capabilities: DeviceCapabilities {
                compute_capability: ComputeCapability { major: 0, minor: 0 },
                sm_count: 0,
                dtype_surface: DtypeSurface::empty(),
                max_threads_per_workgroup: 1024,
                workgroup_shared_memory_min_bytes: 32_768,
                workgroup_shared_memory_max_bytes: 32_768,
                collective_width: 32,
                unified_memory: true,
            },
            memory: DeviceMemory {
                tool_report_total_mib: None,
                api_total_bytes: 0,
            },
            health: DeviceHealth::Healthy,
            health_generation: DeviceHealthGeneration::initial(),
            probe_provenance: ProbeProvenance {
                probe: "md3h-h3 injected".to_owned(),
                tool_versions: "synthetic".to_owned(),
            },
        }],
    )
}

fn n1_ledger() -> PartitionBudgetLedger {
    // N=1 invents no transfer/collective/scratch budget for uniformity.
    PartitionBudgetLedger {
        weight_bytes: 0,
        kv_cache_bytes: 0,
        activation_scratch_bytes: 0,
        module_storage_bytes: 0,
        allocator_overhead_bytes: 0,
        transfer_staging_bytes: 0,
        concurrent_state_bytes: 0,
    }
}

fn admit_implicit_local(
    set: DeviceRuntimeSet,
    snapshot: DeviceDiscoverySnapshot,
    identity: PhysicalDeviceId,
    fixture: FixtureIdentityClass,
    device_name: String,
) -> HostResult<CompositeDeviceState> {
    if set.len() != 1 || !set.contains(&identity) {
        return Err(HostError::invalid_args(
            "N=1 product admission requires the bound PhysicalDeviceId to be the M=1 runtime-set member",
        ));
    }
    let partition = VirtualDevicePartition::implicit_local(
        VirtualDevicePartitionId::new(1),
        identity.clone(),
        n1_ledger(),
        SafePhysicalLimit::new(u64::MAX),
    )
    .map_err(|error| {
        HostError::invalid_args(format!("implicit_local admission rejected: {error:?}"))
    })?;
    let logical = LogicalPartitionId::new(IMPLICIT_LOCAL_PARTITION);
    let admitted =
        AdmittedLogicalPlan::admit(implicit_local_n1_logical_hash(), [logical.clone()], [])
            .map_err(|error| {
                HostError::invalid_args(format!("N=1 logical plan admission rejected: {error:?}"))
            })?;
    let mut bindings = BTreeMap::new();
    bindings.insert(
        logical,
        PartitionBinding::with_virtual_partition(identity.clone(), partition),
    );
    let plan = bind(
        &admitted,
        bindings,
        DeviceSet::from_members([identity]),
        &snapshot,
        DeviceHealthGeneration::initial(),
        fixture,
        TransportClass::None,
    )
    .map_err(|error| HostError::invalid_args(format!("N=1 bind rejected: {error:?}")))?;
    Ok(CompositeDeviceState::Device {
        set,
        bound_plan: plan,
        device_name,
    })
}

fn bound_implicit_device(plan: &BoundDistributedPlan) -> Option<&PhysicalDeviceId> {
    match plan.kind() {
        BoundPlanKind::ImplicitLocal { device, .. } => Some(device),
        BoundPlanKind::Distributed { .. } => None,
    }
}
