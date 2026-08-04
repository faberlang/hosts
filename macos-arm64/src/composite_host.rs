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
//! # A8: device execution is not provider routing
//!
//! The composite host holds the frame/kernel-effects host ([`HostKernel`])
//! and the device component ([`CompositeDeviceState`]) as **separate fields**.
//! Kernel effects (aleator/tempus/consolum/solum/processus + host echo) route
//! through the kernel; device sessions are never exposed as provider routes.
//! [`CompositeHost::execute_descriptor`] drives the device session directly
//! and reports an A9-style receipt (selected hardware, module hash, launches,
//! transfers, readbacks, allocations).

use std::collections::BTreeMap;

use faber::device::{DeviceBackend, DeviceHandle, DeviceSelection};

use crate::device_descriptor::{
    errors as descriptor_errors, fnv1a64, DeviceBufferInitialization, DeviceBufferLifetime,
    DeviceBufferRole, DeviceDataType, DeviceDescriptor, DeviceProgramLifetime, E_DEVICE_DESCRIPTOR,
};
use crate::device_host::{DeviceRuntime, DeviceSession};
use crate::device_registry::DriverCounters;
use crate::kernel::{HostError, HostKernel, HostResult};
use crate::manifest::CapabilityManifest;
use crate::Frame;

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
    /// A live device session plus its selected-hardware name (A9 receipts).
    Device {
        /// The selected native session.
        runtime: DeviceRuntime,
        /// Human-readable selected-hardware name from the admission probe.
        device_name: String,
    },
}

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
    /// PerProgram buffer ids: allocated once per session, released at program
    /// end (persist across executions).
    pub per_program_buffers: Vec<u32>,
    /// Version-keyed PerProgram allocations.
    pub per_program_buffer_versions: Vec<(u32, u32)>,
    /// PerStep buffer ids: recycled at each step boundary.
    pub per_step_buffers: Vec<u32>,
    /// Version-keyed PerStep allocations.
    pub per_step_buffer_versions: Vec<(u32, u32)>,
    /// ObservationPoint buffer ids: read back and released per execution
    /// (read-then-release).
    pub observation_buffers: Vec<u32>,
    /// Version-keyed ObservationPoint allocations.
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
    /// step-boundary barrier that makes the completion boundary valid.
    pub syncs: usize,
    /// Observed transfers this execution (host→device copy-ins plus
    /// device→host readbacks).
    pub transfers: usize,
    /// Device→host readbacks actually performed (the declared observation
    /// points — observation-only readback, F6).
    pub readbacks: usize,
    /// Observed buffer releases this execution (read-then-release plus the
    /// step-boundary PerStep recycle).
    pub releases: usize,
    /// The completion boundary this execution guarantees (R9): the explicit
    /// step-boundary sync after the last launch, at which every declared
    /// observation is valid. Stated exactly — never beyond the explicit
    /// synchronization the host actually performed.
    pub completion_boundary: CompletionBoundary,
    /// FNV-1a hash of the carried semantic graph (roots + launches +
    /// dependency edges + buffer semantic identities + observation points),
    /// computed by the host from the descriptor it consumed — the graph
    /// identity this execution ran, distinct from the module provenance
    /// hash.
    pub semantic_graph_hash: u64,
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

type BufferKey = (u32, u32);

/// Whether an execution copies host inputs into the declared input slots.
///
/// `SingleRun` executions copy per call (the one-shot-with-repeat surface);
/// `RepeatingStep` executions copy nothing — the HostProvided params were
/// once-init'd at session creation and stay device-resident (S5-U6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CopyMode {
    /// Copy declared host inputs per execution (SingleRun).
    PerStep,
    /// No copy-in: params are already device-resident from the once-init at
    /// session creation (RepeatingStep).
    OnceInit,
}

/// One descriptor launch record retained by a program session.
struct SessionLaunch {
    id: u32,
    kernel_index: usize,
}

/// One kernel's launch plan stored by the program session. Cloned from the
/// descriptor at session creation; the launch order, entry, buffer bindings,
/// and grid/block shape are fixed for the program's lifetime.
struct SessionKernel {
    /// Target-neutral logical entry name.
    entry: String,
    /// Typed buffer slots in launch order.
    slots: Vec<SessionSlot>,
    /// 3D dispatch grid.
    grid: [u32; 3],
    /// 3D block (threadgroup) shape.
    block: [u32; 3],
}

/// One buffer slot of a session kernel.
struct SessionSlot {
    /// Program-level buffer identity.
    buffer_id: u32,
    /// Content version selecting the session buffer allocation.
    version: u32,
    /// Logical name for diagnostics.
    buffer_name: String,
    /// Slot role at this kernel.
    role: DeviceBufferRole,
}

/// One declared observation point cloned from the descriptor at session
/// creation (F6): the result rows the session reads back and releases.
/// The producing-launch and observation-order facts were validated by
/// [`DeviceDescriptor::validate`] before the session was created; the
/// session needs only the observed buffer identity + content version.
#[derive(Debug, Clone, Copy)]
struct SessionResult {
    /// Observed buffer identity.
    buffer_id: u32,
    /// Observed content version.
    version: u32,
}

/// Per-version metadata captured at session creation for input validation,
/// lifetime-distinct allocation/release (S2-4), and the A10 declared
/// resource graph (S2-8).
struct SessionBufferMeta {
    /// Logical name of the first reference (resource-graph fact).
    name: String,
    /// Program role of the first reference (resource-graph fact).
    role: DeviceBufferRole,
    /// Element type declared by the descriptor (resource-graph fact).
    element_ty: DeviceDataType,
    /// Element count declared by the descriptor (input size check).
    element_count: u64,
    /// Byte length this buffer's storage needs on the device.
    byte_length: u64,
    /// Lifetime class that drives the session's allocation/release policy.
    lifetime: DeviceBufferLifetime,
    /// Independent initialization axis (F5): how this buffer's storage is
    /// brought to its first defined state. The session honors `ZeroFill` by
    /// zeroing the buffer at allocation; `HostProvided` buffers receive the
    /// declared input at launch; `KernelInitialized` buffers are written by
    /// a kernel before any read. Never re-derived from role or lifetime.
    initialization: DeviceBufferInitialization,
}

/// A program-scoped device session that outlives individual launches (S2-1).
///
/// Created from one [`DeviceDescriptor`], a `ProgramSession` owns:
/// - the **module** (loaded once at creation; reused by every execution);
/// - the **PerProgram buffers** (allocated once at creation; persist across
///   executions; released at program end);
/// - the **PerStep buffers** (allocated per execution, recycled at the step
///   boundary — released at the end of each execution and re-allocated for
///   the next);
/// - the **ObservationPoint buffers** (allocated per execution, read back at
///   the declared observation point, then released — read-then-release).
///
/// [`ProgramSession::execute`] runs the ordered launch sequence once (one
/// step), synchronizes at the step boundary, reads back the declared
/// observation buffers, and releases the per-step + observation buffers.
/// It can be called repeatedly on the same session without reloading the
/// module or re-allocating PerProgram buffers. [`ProgramSession::teardown`]
/// performs the ordered release (remaining buffers then module).
///
/// The S1-4 [`CompositeHost::execute_descriptor`] surface is a single-run
/// convenience over this session (create → execute → teardown).
///
/// # Lifetime-distinct release (S2-4; the schema-debt closer)
///
/// The session consumes the descriptor's typed [`DeviceBufferLifetime`]
/// facts — it never derives a lifetime from slot role alone (that would be
/// coincidence, council 3):
/// - **PerProgram** — allocated once at session creation, released at program
///   end (persists across steps);
/// - **PerStep** — live within one step, recycled at the step boundary
///   (released between executions, re-allocated for the next);
/// - **ObservationPoint** — read back at the declared observation point, then
///   released (read-then-release); it is the only class the session reads
///   back, so a readback request for any other class fails closed (no
///   undeclared readback).
///
/// The [`DeviceProgramLifetime`] regime is carried and consumed as the
/// declared program fact: `SingleRun` is the one-shot-with-repeat surface
/// (each [`ProgramSession::execute`] call copies its declared host inputs
/// and re-runs the whole program). `RepeatingStep` is the training-loop
/// surface (S5-U6): `HostProvided` params are copied into their PerProgram
/// buffers exactly once at session creation via [`ProgramSession::init_params`]
/// and never re-copied; each [`ProgramSession::execute_step`] allocates the
/// step's PerStep and ObservationPoint buffers, runs the ordered launches
/// with no copy-in, synchronizes at the step boundary, reads back the
/// declared observation (the per-step loss trace), and recycles the
/// per-step buffers. The two regimes never mix: `execute` refuses a
/// `RepeatingStep` session and `execute_step` refuses a `SingleRun` one
/// (params once-init + never re-copied is the RepeatingStep contract).
///
/// # Module cache policy (S2-2)
///
/// The session owns the module: it is loaded exactly once per program
/// (keyed by its FNV-1a provenance hash, [`ProgramSession::module_hash`]),
/// reused by every [`ProgramSession::execute`] call, and released at
/// [`ProgramSession::teardown`]. There is no global or LRU cache and no
/// cross-process persistence — a second session re-loads the same image
/// independently. The testable bar is "repeated execution does not leak",
/// proven with the fake drivers' lifecycle counters (module load = 1, module
/// release = 1, nothing persists past teardown) — NOT "module persists
/// across steps".
///
/// # Error-path teardown (S2-3; the absorbed S1-4 audit finding P2-1)
///
/// Teardown is designed into the session on every path, not bolted on: a
/// failed **creation** releases the module and every partially allocated
/// buffer before the error escapes, and a failed **execution** runs the
/// ordered release (buffers then module) before the error escapes and closes
/// the session. A failed execution at any stage (module load, allocation,
/// copy-in, launch, sync, readback) therefore leaves
/// `live_handle_count() == 0` and no handle survives a failed execution.
pub struct ProgramSession<'host> {
    runtime: &'host mut DeviceRuntime,
    backend: DeviceBackend,
    device_name: String,
    module_handle: DeviceHandle,
    module_hash: u64,
    /// Program execution-lifetime regime (S2-4).
    program_lifetime: DeviceProgramLifetime,
    /// Currently-live device buffers: (buffer_id, version) → device handle.
    /// PerProgram buffers are live from creation until teardown; PerStep and
    /// ObservationPoint buffers are live only within one execution.
    buffers: BTreeMap<BufferKey, DeviceHandle>,
    /// Per-version declared element count / byte length / lifetime class.
    buffer_meta: BTreeMap<BufferKey, SessionBufferMeta>,
    /// Kernel declarations cloned from the descriptor.
    kernels: Vec<SessionKernel>,
    /// The ordered launch plan cloned from the descriptor.
    launches: Vec<SessionLaunch>,
    /// Carried inter-kernel data-flow edges (A10/R2): the wire's
    /// producer/consumer facts per buffer version, consumed by
    /// [`ProgramSession::declared_resource_graph`] — never re-derived from
    /// launch order.
    data_flow: Vec<DataFlowEdge>,
    /// Declared observation points (F6): the result rows projected from the
    /// descriptor's observation facts; the only buffers this session reads
    /// back.
    results: Vec<SessionResult>,
    /// FNV-1a hash of the carried semantic graph the session executes.
    semantic_graph_hash: u64,
    /// Whether a `RepeatingStep` session's HostProvided params have been
    /// once-init'd via [`ProgramSession::init_params`] (S5-U6). Steps
    /// refuse until the once-init has run; a second once-init is refused
    /// ("copied in exactly once").
    params_initialized: bool,
    /// True after an error-path release (S2-3): every handle has been
    /// released and the session cannot execute again.
    closed: bool,
}

impl<'host> ProgramSession<'host> {
    /// Create a program session: validate the descriptor, load the module
    /// once, and allocate every distinct **PerProgram** buffer once (PerStep
    /// and ObservationPoint buffers are allocated per execution — S2-4).
    ///
    /// # Errors
    /// - `E_DEVICE_DESCRIPTOR` — the descriptor targets a different backend
    ///   than the runtime session, or is structurally invalid;
    /// - `E_DEVICE_ABI_MISMATCH` / `E_DEVICE_DTYPE_MISMATCH` /
    ///   `E_DEVICE_SHAPE_MISMATCH` — typed descriptor conflicts (see
    ///   [`DeviceDescriptor::validate`]);
    /// - session-level failures (module load, allocation) bubble through.
    pub fn new(
        runtime: &'host mut DeviceRuntime,
        descriptor: &DeviceDescriptor,
        device_name: String,
    ) -> HostResult<Self> {
        if runtime.backend() != descriptor.backend {
            return Err(HostError {
                code: E_DEVICE_DESCRIPTOR.to_owned(),
                message: format!(
                    "device descriptor targets backend `{}` but the composite host's device session is `{}`",
                    descriptor.backend.spelling(),
                    runtime.backend().spelling()
                ),
                retryable: false,
            });
        }
        descriptor.validate()?;

        // Load the module once (session-scoped ownership). The module-cache
        // policy (S2-2): loaded once per program keyed by its provenance
        // hash, reused by every execution, released at teardown — repeated
        // execution does not leak, and there is no cross-session or
        // cross-process cache.
        let module_handle = runtime.load_module(&descriptor.module_image)?;
        let module_hash = fnv1a64(&descriptor.module_image);

        // Allocate every distinct PerProgram buffer once (S2-4): they persist
        // for the program's lifetime. PerStep and ObservationPoint buffers
        // are not allocated here — they are allocated at each execution's
        // start and released at its step boundary / after readback. A
        // failure at any PerProgram allocation runs the error-path teardown
        // first (S2-3 release-on-error): the module and every already-
        // allocated buffer are released before the error escapes, so a
        // failed creation leaves `live_handle_count() == 0`.
        let mut buffers: BTreeMap<BufferKey, DeviceHandle> = BTreeMap::new();
        let mut buffer_meta: BTreeMap<BufferKey, SessionBufferMeta> = BTreeMap::new();
        let mut kernels: Vec<SessionKernel> = Vec::with_capacity(descriptor.kernels.len());

        let result = (|| {
            for kernel in &descriptor.kernels {
                let mut slots = Vec::with_capacity(kernel.buffers.len());
                for slot in &kernel.buffers {
                    let key = (slot.buffer_id, slot.version);
                    buffer_meta.entry(key).or_insert(SessionBufferMeta {
                        name: slot.buffer_name.clone(),
                        role: slot.role,
                        element_ty: slot.element_ty,
                        element_count: slot.element_count,
                        byte_length: slot.byte_length(),
                        lifetime: slot.lifetime,
                        initialization: slot.initialization,
                    });
                    if !buffers.contains_key(&key)
                        && slot.lifetime == DeviceBufferLifetime::PerProgram
                    {
                        let handle = runtime.alloc_bytes(slot.byte_length() as usize)?;
                        // G4 (F5): honor the carried initialization axis —
                        // ZeroFill persistent state (accumulation buffers,
                        // optimizer state) is zeroed EXACTLY ONCE at
                        // allocation so repeated executions accumulate onto
                        // a defined initial state.
                        if slot.initialization == DeviceBufferInitialization::ZeroFill {
                            runtime
                                .copy_in_f32(&handle, &vec![0.0; slot.element_count as usize])?;
                        }
                        buffers.insert(key, handle);
                    }
                    slots.push(SessionSlot {
                        buffer_id: slot.buffer_id,
                        version: slot.version,
                        buffer_name: slot.buffer_name.clone(),
                        role: slot.role,
                    });
                }
                kernels.push(SessionKernel {
                    entry: kernel.entry.clone(),
                    slots,
                    grid: kernel.grid,
                    block: kernel.block,
                });
            }
            Ok(())
        })();

        if let Err(error) = result {
            // Error-path teardown at creation (S2-3): release every buffer
            // allocated so far, then the module, before the error escapes.
            // Release failures are secondary to the creation failure, but
            // every release is still attempted.
            for handle in buffers.values() {
                drop(runtime.release(handle));
            }
            drop(runtime.release(&module_handle));
            return Err(error);
        }

        let launches = descriptor
            .launches
            .iter()
            .map(|launch| SessionLaunch {
                id: launch.id,
                kernel_index: launch.kernel_index as usize,
            })
            .collect();

        Ok(Self {
            runtime,
            backend: descriptor.backend,
            device_name,
            module_handle,
            module_hash,
            program_lifetime: descriptor.program_lifetime,
            buffers,
            buffer_meta,
            kernels,
            launches,
            data_flow: descriptor
                .data_flow
                .iter()
                .map(|edge| DataFlowEdge {
                    buffer_id: edge.buffer_id,
                    version: edge.version,
                    producer: edge.producer,
                    consumer: edge.consumer,
                })
                .collect(),
            // Declared observation points (F6): the session reads back exactly
            // the descriptor's result rows — never a caller-selected subset
            // and never an undeclared buffer.
            results: descriptor
                .results
                .iter()
                .map(|result| SessionResult {
                    buffer_id: result.buffer_id,
                    version: result.version,
                })
                .collect(),
            semantic_graph_hash: descriptor.semantic_graph_hash(),
            params_initialized: false,
            closed: false,
        })
    }

    /// Execute the ordered launch sequence once (one step) on the SingleRun
    /// surface: copies the declared host inputs into their input slots for
    /// this execution, then reuses the session's module and PerProgram
    /// buffers — does not reload the module or re-allocate PerProgram
    /// buffers. PerStep buffers are allocated for the step and recycled at
    /// the step boundary; the declared observation point buffers
    /// ([`SessionResult`]s projected from the descriptor's observation facts
    /// — F6) are allocated, read back at their producing launch's
    /// completion boundary, and released (read-then-release — the only
    /// readback the session performs, S2-4). Synchronizes at the step
    /// boundary before reading back.
    ///
    /// **Error-path teardown is designed into this method (S2-3):** a failure
    /// at any stage runs the ordered release (buffers then module) before the
    /// error escapes and closes the session, so a failed execution leaves
    /// `live_handle_count() == 0` and no handle survives. A closed session
    /// refuses further [`ProgramSession::execute`] calls.
    ///
    /// A `RepeatingStep` session refuses `execute` (S5-U6): its HostProvided
    /// params are once-init'd at session creation and never re-copied, so
    /// per-execution input copy-in is the SingleRun surface. Use
    /// [`ProgramSession::init_params`] + [`ProgramSession::execute_step`].
    ///
    /// # Errors
    /// - `E_DEVICE_ENTRY_MISMATCH` — a kernel entry is unknown to the module;
    /// - `E_DEVICE_SHAPE_MISMATCH` — a declared input is missing or its size
    ///   contradicts the declared element count;
    /// - `E_INVALID_ARGS` — a declared observation buffer id was not
    ///   allocated by the session, or the id names a buffer whose lifetime is
    ///   not ObservationPoint (an undeclared readback fails closed, S2-4);
    /// - session-level failures (copy-in, launch, sync, readback) bubble
    ///   through unchanged.
    pub fn execute(
        &mut self,
        inputs: &BTreeMap<u32, Vec<f32>>,
    ) -> HostResult<DeviceExecutionReceipt> {
        if self.closed {
            return Err(HostError::internal(
                "program session is closed after a failed execution; create a new session",
            ));
        }
        if self.program_lifetime == DeviceProgramLifetime::RepeatingStep {
            return Err(HostError::internal(
                "a RepeatingStep session executes through init_params + execute_step: HostProvided params are copied in exactly once at session creation and never re-copied on later steps; per-execution input copy-in is the SingleRun surface",
            ));
        }
        let result = self.execute_inner(inputs, CopyMode::PerStep);
        if result.is_err() {
            // Release-on-error on all paths (S2-3): the ordered release runs
            // before the error escapes, then the session is closed so no
            // stale handle can be used again. Release failures on top of the
            // stage failure are secondary — every release is still attempted.
            drop(self.release_all_handles());
            self.closed = true;
        }
        result
    }

    /// Once-init the declared `HostProvided` training params of a
    /// `RepeatingStep` session (S5-U6): each HostProvided PerProgram buffer
    /// receives its declared values exactly once, at session creation, and
    /// is never re-copied on later steps. The only buffers this copies are
    /// PerProgram + HostProvided; every such declared buffer must be present
    /// with its declared element count. A buffer id carrying more than one
    /// content version cannot be once-init'd from one value vector and fails
    /// closed.
    ///
    /// **Error-path teardown (S2-3):** a failed once-init runs the ordered
    /// release (buffers then module) before the error escapes and closes the
    /// session, so a failed once-init leaves `live_handle_count() == 0`.
    ///
    /// # Errors
    /// - `E_INTERNAL` — the session is not `RepeatingStep`, the params were
    ///   already once-init'd, or a param id carries multiple content
    ///   versions;
    /// - `E_DEVICE_SHAPE_MISMATCH` — a declared param is missing or its size
    ///   contradicts the declared element count;
    /// - session-level copy failures bubble through.
    pub fn init_params(&mut self, params: &BTreeMap<u32, Vec<f32>>) -> HostResult<()> {
        if self.program_lifetime != DeviceProgramLifetime::RepeatingStep {
            return Err(HostError::internal(
                "once-init params are a RepeatingStep contract: a SingleRun session copies its declared host inputs per execution",
            ));
        }
        if self.params_initialized {
            return Err(HostError::internal(
                "once-init params were already copied; a RepeatingStep session copies its HostProvided params exactly once at session creation",
            ));
        }
        let result = self.init_params_inner(params);
        match result {
            Ok(()) => {
                self.params_initialized = true;
                Ok(())
            }
            Err(error) => {
                // Release-on-error on all paths (S2-3).
                drop(self.release_all_handles());
                self.closed = true;
                Err(error)
            }
        }
    }

    /// The once-init body of [`ProgramSession::init_params`]: the copy loop
    /// over the declared HostProvided PerProgram params, without the
    /// error-path release, which the caller owns.
    fn init_params_inner(&mut self, params: &BTreeMap<u32, Vec<f32>>) -> HostResult<()> {
        // The declared param set: every distinct buffer id whose storage is
        // PerProgram and HostProvided (the F5 axis is carried, never
        // re-derived from role). A second content version of the same id
        // cannot be once-init'd from one value vector (a shape change is a
        // new version), so it fails closed.
        let mut param_ids: Vec<u32> = Vec::new();
        for ((id, _), meta) in &self.buffer_meta {
            if meta.lifetime == DeviceBufferLifetime::PerProgram
                && meta.initialization == DeviceBufferInitialization::HostProvided
            {
                if param_ids.contains(id) {
                    return Err(HostError::internal(format!(
                        "RepeatingStep param buffer `{}` (id {id}) carries multiple content versions; once-init requires a single param version",
                        meta.name
                    )));
                }
                param_ids.push(*id);
            }
        }
        for id in param_ids {
            let key = self
                .buffer_meta
                .keys()
                .find(|(buffer_id, _)| *buffer_id == id)
                .copied()
                .ok_or_else(|| HostError::internal("RepeatingStep param metadata disappeared"))?;
            let meta = &self.buffer_meta[&key];
            let values = params.get(&id).ok_or_else(|| {
                descriptor_errors::shape_mismatch(format!(
                    "RepeatingStep param `{}` (id {id}) is not provided at once-init; every HostProvided PerProgram buffer must receive its declared values exactly once",
                    meta.name
                ))
            })?;
            let expected = meta.element_count;
            if u64::try_from(values.len()).ok() != Some(expected) {
                return Err(descriptor_errors::shape_mismatch(format!(
                    "RepeatingStep param `{}` (id {id}) has {} f32 elements but its declared storage holds {expected}",
                    meta.name,
                    values.len()
                )));
            }
            let handle = self
                .buffers
                .get(&key)
                .copied()
                .ok_or_else(|| HostError::internal("RepeatingStep param buffer disappeared"))?;
            self.runtime.copy_in_f32(&handle, values)?;
        }
        Ok(())
    }

    /// Execute one training step on a `RepeatingStep` session whose params
    /// were once-init'd (S5-U6): allocate the step's PerStep +
    /// ObservationPoint buffers, run the ordered launch sequence with **no
    /// copy-in** (the HostProvided params are already device-resident from
    /// the once-init), synchronize at the step boundary, read back the
    /// declared observation (the per-step loss trace), and recycle the
    /// per-step buffers. The receipt counts per-step syncs, transfers
    /// (readbacks only — copy_ins is 0), readbacks, and releases.
    ///
    /// **Error-path teardown is designed into this method (S2-3):** a failure
    /// at any stage runs the ordered release before the error escapes and
    /// closes the session, so a failed step leaves
    /// `live_handle_count() == 0`.
    ///
    /// # Errors
    /// - `E_INTERNAL` — the session is not `RepeatingStep`, or the params
    ///   were not once-init'd;
    /// - session-level failures (launch, sync, readback) bubble through
    ///   unchanged.
    pub fn execute_step(&mut self) -> HostResult<DeviceExecutionReceipt> {
        if self.closed {
            return Err(HostError::internal(
                "program session is closed after a failed execution; create a new session",
            ));
        }
        if self.program_lifetime != DeviceProgramLifetime::RepeatingStep {
            return Err(HostError::internal(
                "execute_step is a RepeatingStep surface: a SingleRun session runs the whole program per execute call",
            ));
        }
        if !self.params_initialized {
            return Err(HostError::internal(
                "RepeatingStep params were not once-init'd; call init_params before execute_step",
            ));
        }
        let result = self.execute_inner(&BTreeMap::new(), CopyMode::OnceInit);
        if result.is_err() {
            // Release-on-error on all paths (S2-3).
            drop(self.release_all_handles());
            self.closed = true;
        }
        result
    }

    /// The executable body shared by [`ProgramSession::execute`] and
    /// [`ProgramSession::execute_step`]: the ordered launch sequence
    /// (step-buffer allocation → copy-in (SingleRun only) → launch →
    /// step-boundary sync → observation readback + release → per-step
    /// release) without the error-path release, which the caller owns.
    fn execute_inner(
        &mut self,
        inputs: &BTreeMap<u32, Vec<f32>>,
        mode: CopyMode,
    ) -> HostResult<DeviceExecutionReceipt> {
        let mut launch_count = 0usize;
        let mut launch_ids = Vec::with_capacity(self.launches.len());
        let mut launch_entries = Vec::with_capacity(self.launches.len());
        let mut copy_ins = 0usize;

        // Allocate this step's PerStep + ObservationPoint buffers (S2-4).
        // PerProgram buffers were allocated once at session creation and
        // stay live. A failure here runs the error-path teardown (S2-3).
        self.allocate_step_buffers()?;

        for launch in &self.launches {
            let kernel = self
                .kernels
                .get(launch.kernel_index)
                .ok_or_else(|| HostError::internal("session launch references missing kernel"))?;
            // Resolve buffer handles for this kernel's launch (PerProgram
            // live from creation; PerStep/ObservationPoint just allocated).
            let mut launch_buffers: Vec<DeviceHandle> = Vec::with_capacity(kernel.slots.len());
            for slot in &kernel.slots {
                let key = (slot.buffer_id, slot.version);
                let handle = self.buffers.get(&key).copied().ok_or_else(|| {
                    HostError::internal("session buffer disappeared during launch")
                })?;
                launch_buffers.push(handle);
            }

            // Copy-in declared inputs for this kernel — SingleRun only
            // (PerStep mode). A RepeatingStep step (OnceInit mode) copies
            // nothing: the HostProvided params were once-init'd at session
            // creation and stay device-resident (S5-U6).
            if mode == CopyMode::PerStep {
                for slot in &kernel.slots {
                    if slot.role == DeviceBufferRole::Input {
                        let values = inputs.get(&slot.buffer_id).ok_or_else(|| {
                            descriptor_errors::shape_mismatch(format!(
                                "descriptor kernel `{}` declares input buffer `{}` (id {}) but no host input was provided",
                                kernel.entry, slot.buffer_name, slot.buffer_id
                            ))
                        })?;
                        let expected = self
                            .buffer_meta
                            .get(&(slot.buffer_id, slot.version))
                            .map(|meta| meta.element_count)
                            .unwrap_or(0);
                        if u64::try_from(values.len()).ok() != Some(expected) {
                            return Err(descriptor_errors::shape_mismatch(format!(
                                "input for buffer `{}` (id {}) has {} f32 elements but kernel `{}` declares {}",
                                slot.buffer_name,
                                slot.buffer_id,
                                values.len(),
                                kernel.entry,
                                expected
                            )));
                        }
                        let handle = self
                            .buffers
                            .get(&(slot.buffer_id, slot.version))
                            .copied()
                            .ok_or_else(|| {
                                HostError::internal("session input buffer disappeared")
                            })?;
                        self.runtime.copy_in_f32(&handle, values)?;
                        copy_ins += 1;
                    }
                }
            }

            self.runtime.launch_kernel(
                &self.module_handle,
                &kernel.entry,
                &launch_buffers,
                kernel.grid,
                kernel.block,
            )?;
            launch_count += 1;
            launch_ids.push(launch.id);
            launch_entries.push(kernel.entry.clone());
        }

        // Step-boundary synchronization: every launch in this step has
        // completed before any readback. The launches also sync internally;
        // this barrier makes the step boundary explicit and observable. The
        // completion boundary is this exact barrier after the last launch
        // (R9): the receipt counts real synchronization operations and names
        // where completion is guaranteed.
        self.runtime.sync()?;

        // Observation-only readback (F6): read back exactly the DECLARED
        // observation points — the result rows projected from the
        // descriptor's observation facts at session creation. A buffer with
        // any other lifetime class is an undeclared readback and fails
        // closed. Each observation is read-then-released (S2-4).
        let mut release_count = 0usize;
        let mut readback_count = 0usize;
        let mut readbacks: BTreeMap<u32, Vec<f32>> = BTreeMap::new();
        let observed: Vec<SessionResult> = self.results.clone();
        for result in &observed {
            let key = (result.buffer_id, result.version);
            let meta = self.buffer_meta.get(&key).ok_or_else(|| {
                HostError::invalid_args(format!(
                    "declared observation buffer id {} was not allocated by the session",
                    result.buffer_id
                ))
            })?;
            if meta.lifetime != DeviceBufferLifetime::ObservationPoint {
                return Err(HostError::invalid_args(format!(
                    "declared observation buffer id {} has lifetime `{}`; only observation-point buffers are read back (no undeclared readback)",
                    result.buffer_id,
                    meta.lifetime.spelling()
                )));
            }
            let handle = self
                .buffers
                .get(&key)
                .copied()
                .ok_or_else(|| HostError::internal("session observation buffer disappeared"))?;
            let values = self.runtime.readback_f32(&handle)?;
            readbacks.insert(result.buffer_id, values);
            readback_count += 1;
            self.release_buffer(key)?;
            release_count += 1;
        }

        // Step-boundary recycle (S2-4): PerStep buffers are released at the
        // step boundary and re-allocated for the next execution.
        let per_step_ids: Vec<BufferKey> = self
            .buffers
            .iter()
            .filter(|(key, _)| {
                self.buffer_meta
                    .get(key)
                    .is_some_and(|meta| meta.lifetime == DeviceBufferLifetime::PerStep)
            })
            .map(|(key, _)| *key)
            .collect();
        for key in per_step_ids {
            self.release_buffer(key)?;
            release_count += 1;
        }

        // Declared logical resource graph + data-flow edges (A10) from the
        // session's declared facts (the descriptor projected onto the
        // program).
        let (resource_graph, data_flow_edges) = self.declared_resource_graph();

        Ok(DeviceExecutionReceipt {
            backend: self.backend,
            device_name: self.device_name.clone(),
            module_hash: self.module_hash,
            launches: launch_count,
            launch_ids,
            launch_entries,
            copy_ins,
            outputs: readbacks,
            allocated_buffers: self.allocated_buffers(),
            allocated_buffer_versions: self.allocated_buffer_versions(),
            program_lifetime: self.program_lifetime,
            per_program_buffers: self.buffers_by_lifetime(DeviceBufferLifetime::PerProgram),
            per_program_buffer_versions: self
                .buffer_versions_by_lifetime(DeviceBufferLifetime::PerProgram),
            per_step_buffers: self.buffers_by_lifetime(DeviceBufferLifetime::PerStep),
            per_step_buffer_versions: self
                .buffer_versions_by_lifetime(DeviceBufferLifetime::PerStep),
            observation_buffers: self.buffers_by_lifetime(DeviceBufferLifetime::ObservationPoint),
            observation_buffer_versions: self
                .buffer_versions_by_lifetime(DeviceBufferLifetime::ObservationPoint),
            resource_graph,
            data_flow_edges,
            // R9: real synchronization operations — one per launch (each
            // launch synchronizes internally) plus the explicit step-boundary
            // barrier. The completion boundary is that barrier after the last
            // dispatched launch.
            syncs: launch_count + 1,
            transfers: copy_ins + readback_count,
            readbacks: readback_count,
            releases: release_count,
            completion_boundary: CompletionBoundary::StepSync {
                after_launch: self.launches.last().map(|launch| launch.id).unwrap_or(0),
            },
            semantic_graph_hash: self.semantic_graph_hash,
        })
    }

    /// Allocate this step's PerStep and ObservationPoint buffers (S2-4);
    /// PerProgram buffers are already live from session creation. Buffer ids
    /// already live are left untouched (a PerProgram buffer, or a step buffer
    /// left live by an interrupted path that has not yet run the error path,
    /// is never double-allocated).
    fn allocate_step_buffers(&mut self) -> HostResult<()> {
        let to_allocate: Vec<BufferKey> = self
            .buffer_meta
            .iter()
            .filter(|(key, meta)| {
                meta.lifetime != DeviceBufferLifetime::PerProgram && !self.buffers.contains_key(key)
            })
            .map(|(key, _)| *key)
            .collect();
        for key in to_allocate {
            let meta = self
                .buffer_meta
                .get(&key)
                .ok_or_else(|| HostError::internal("session buffer metadata disappeared"))?;
            let handle = self.runtime.alloc_bytes(meta.byte_length as usize)?;
            // G4 (F5): honor the carried initialization axis at every
            // allocation — a ZeroFill step buffer (per-step accumulation
            // state) is zeroed when it comes live.
            if meta.initialization == DeviceBufferInitialization::ZeroFill {
                self.runtime
                    .copy_in_f32(&handle, &vec![0.0; meta.element_count as usize])?;
            }
            self.buffers.insert(key, handle);
        }
        Ok(())
    }

    /// Release one live buffer by key (no-op when the key is not live). Used by
    /// the read-then-release and step-boundary paths.
    fn release_buffer(&mut self, key: BufferKey) -> HostResult<()> {
        if let Some(handle) = self.buffers.remove(&key) {
            self.runtime.release(&handle)?;
        }
        Ok(())
    }

    /// The program's buffer ids classified by lifetime class (S2-4 receipt).
    fn buffers_by_lifetime(&self, lifetime: DeviceBufferLifetime) -> Vec<u32> {
        let mut ids = Vec::new();
        self.buffer_meta
            .iter()
            .filter(|(_, meta)| meta.lifetime == lifetime)
            .for_each(|((id, _), _)| {
                if ids.last() != Some(id) {
                    ids.push(*id);
                }
            });
        ids
    }

    /// The program's version-keyed buffer metadata classified by lifetime.
    fn buffer_versions_by_lifetime(&self, lifetime: DeviceBufferLifetime) -> Vec<BufferKey> {
        self.buffer_meta
            .iter()
            .filter(|(_, meta)| meta.lifetime == lifetime)
            .map(|(key, _)| *key)
            .collect()
    }

    /// The declared logical resource graph (A10): every buffer identity +
    /// carried content version plus the carried inter-kernel data-flow
    /// edges.
    ///
    /// Graph order is buffer-id ascending, which equals first-reference
    /// order for constructor-built programs (buffer ids are minted in
    /// reference order). The content versions and the producer/consumer
    /// edges are the WIRE'S carried facts (R2): the session consumes the
    /// descriptor's per-buffer version and data-flow list verbatim — it
    /// never hardcodes `version: 1` and never re-derives an edge from a
    /// first-writer launch-order coincidence rule. This is the declared
    /// (payload-derived) graph — the session never observes the intermediate
    /// (S2-4/S2-5: no undeclared readback).
    fn declared_resource_graph(&self) -> (Vec<ReceiptBuffer>, Vec<DataFlowEdge>) {
        let graph = self
            .buffer_meta
            .iter()
            .map(|((id, version), meta)| ReceiptBuffer {
                id: *id,
                name: meta.name.clone(),
                role: meta.role,
                lifetime: meta.lifetime,
                element_ty: meta.element_ty,
                element_count: meta.element_count,
                version: *version,
            })
            .collect();
        (graph, self.data_flow.clone())
    }

    /// Ordered teardown: release every buffer, then the module. After this
    /// the session is consumed and the device session's handle count returns
    /// to baseline (every allocated handle is released). Every release is
    /// attempted even if one fails, so a partial release failure does not
    /// leave later handles alive.
    ///
    /// A session already closed by an error-path release (S2-3) has nothing
    /// left to release and returns `Ok`.
    ///
    /// # Errors
    /// The first session-level release failure bubbles through after every
    /// release has been attempted.
    pub fn teardown(mut self) -> HostResult<()> {
        if self.closed {
            // The error path already released every handle.
            return Ok(());
        }
        self.release_all_handles()
    }

    /// Ordered teardown shared by the success (`teardown`) and error
    /// (release-on-error in [`ProgramSession::execute`] and `new`) paths:
    /// release every buffer, then the module, attempting every release even
    /// if one fails. Returns the first release failure, if any.
    fn release_all_handles(&mut self) -> HostResult<()> {
        let mut first_error: Option<HostError> = None;
        let buffers: Vec<DeviceHandle> = self.buffers.values().copied().collect();
        for handle in buffers {
            if let Err(error) = self.runtime.release(&handle) {
                first_error.get_or_insert(error);
            }
        }
        if let Err(error) = self.runtime.release(&self.module_handle) {
            first_error.get_or_insert(error);
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// The program-level buffer ids this session manages (A9 receipt): every
    /// distinct buffer id the descriptor declares, classified by lifetime.
    /// PerProgram ids are live for the program's lifetime; PerStep and
    /// ObservationPoint ids are live only within one execution (S2-4).
    #[must_use]
    pub fn allocated_buffers(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        for (id, _) in self.buffer_meta.keys() {
            if ids.last() != Some(id) {
                ids.push(*id);
            }
        }
        ids
    }

    /// The program's version-keyed buffer allocations.
    #[must_use]
    pub fn allocated_buffer_versions(&self) -> Vec<(u32, u32)> {
        self.buffer_meta.keys().copied().collect()
    }

    /// Number of live device handles the session currently holds (module +
    /// currently-live buffers). Used by lifecycle tests to prove no reload /
    /// no PerProgram realloc between executions and full release at teardown.
    /// PerStep and ObservationPoint buffers are released at the step boundary
    /// / after readback (S2-4), so between executions the session holds the
    /// module + PerProgram buffers only. A session closed by an error-path
    /// release (S2-3) holds no live handles and reports 0.
    #[must_use]
    pub fn session_handle_count(&self) -> usize {
        if self.closed {
            0
        } else {
            self.buffers.len() + 1 // buffers + module
        }
    }

    /// Driver-level lifecycle counters (S2-2 module-cache leak bar). The
    /// fake drivers track cumulative module loads/releases and buffer
    /// allocs/releases so session tests prove the cache policy at the driver
    /// boundary: one load per program, one release at teardown, nothing
    /// persists past teardown.
    #[must_use]
    pub fn driver_counters(&self) -> DriverCounters {
        self.runtime.driver_counters()
    }

    /// The backend this session speaks for.
    #[must_use]
    pub fn backend(&self) -> DeviceBackend {
        self.backend
    }

    /// The FNV-1a provenance hash of the loaded module.
    #[must_use]
    pub fn module_hash(&self) -> u64 {
        self.module_hash
    }

    /// The FNV-1a hash of the carried semantic graph this session executes
    /// (roots + launches + dependency edges + buffer semantic identities +
    /// observation points). The graph identity the session consumed —
    /// distinct from [`ProgramSession::module_hash`], which only names the
    /// backend blob.
    #[must_use]
    pub fn semantic_graph_hash(&self) -> u64 {
        self.semantic_graph_hash
    }
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
    /// resolve the selection against the live admission probes, then open the
    /// device session (fail-closed) or run CPU-only.
    ///
    /// # Errors
    /// - `E_BACKEND_UNAVAILABLE` — the resolved backend cannot be opened;
    /// - `E_NO_DEVICE_PROGRAM` — explicit backend on a payload-less route.
    pub fn new(config: CompositeHostConfig) -> HostResult<Self> {
        let admitted = admitted_backends();
        let resolved =
            resolve_device_selection(config.selection, config.requires_device, &admitted)?;
        let device = match resolved {
            None => CompositeDeviceState::CpuOnly,
            Some(backend) => {
                let runtime = DeviceRuntime::open(backend)?;
                CompositeDeviceState::Device {
                    runtime,
                    device_name: backend_device_name(backend),
                }
            }
        };
        Ok(Self {
            kernel: HostKernel::new(),
            device,
        })
    }

    /// Inject a device session directly (sequencing tests only; the driver
    /// fakes bypass the admission probes). Not a product construction path —
    /// product construction always goes through [`CompositeHost::new`].
    pub fn with_device(runtime: DeviceRuntime, device_name: impl Into<String>) -> HostResult<Self> {
        Ok(Self {
            kernel: HostKernel::new(),
            device: CompositeDeviceState::Device {
                runtime,
                device_name: device_name.into(),
            },
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

    /// The live device session, when the host carries one.
    #[must_use]
    pub fn device(&self) -> Option<&DeviceRuntime> {
        match &self.device {
            CompositeDeviceState::CpuOnly => None,
            CompositeDeviceState::Device { runtime, .. } => Some(runtime),
        }
    }

    /// The live device session (mutable).
    #[must_use]
    pub fn device_mut(&mut self) -> Option<&mut DeviceRuntime> {
        match &mut self.device {
            CompositeDeviceState::CpuOnly => None,
            CompositeDeviceState::Device { runtime, .. } => Some(runtime),
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

    /// Discovery receipt: the capability manifest of the kernel-effects host.
    #[must_use]
    pub fn manifest(&self) -> CapabilityManifest {
        self.kernel.manifest()
    }

    /// Create a program-scoped session for one device program (S2-1).
    ///
    /// The session owns the module (loaded once) and every PerProgram buffer
    /// (allocated once at creation, persisting across executions); PerStep
    /// and ObservationPoint buffers are allocated per execution and recycled
    /// / read-then-released at each step boundary (S2-4). It survives
    /// repeated executions on the same session without reloading or
    /// re-allocating PerProgram buffers. Call
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
        let device_name = self.device_name().to_owned();
        let runtime = self.device_mut().ok_or_else(|| {
            descriptor_errors::no_device_program(
                "composite host has no device session; a device descriptor cannot execute",
            )
        })?;
        ProgramSession::new(runtime, descriptor, device_name)
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
