//! The observation-cadence program-session executor (S2-1..S5A-U1): one
//! program-scoped device session per [`DeviceDescriptor`] that owns the
//! module and the `PerProgram` buffers, and runs the ordered launch sequence
//! with the declared readback cadence (per-step loss observation, end-of-run
//! readback) under lifetime-distinct allocation/release (S2-4).

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use host_coordinator::{DeviceBackend, DeviceHandle};

use crate::device_descriptor::{
    errors as descriptor_errors, fnv1a64, DescriptorAllocation, DescriptorInvocationState,
    DescriptorLaunchBinding, DescriptorRuntimeSource, DescriptorView, DeviceBufferInitialization,
    DeviceBufferLifetime, DeviceBufferRole, DeviceDataType, DeviceDescriptor,
    DeviceProgramLifetime, KvCacheDescriptor, PackedStorageFormat, E_DEVICE_DESCRIPTOR,
};
use crate::device_host::{
    DeviceLaunchBinding, DeviceRuntime, DeviceSession, InvocationStateBuffer,
};
use crate::device_registry::DriverCounters;
use crate::kernel::library::{QuantizedFormat, QkvProjectionBind};
use crate::kernel::library_runtime::{
    dispatch_fused_qkv_device, dispatch_fused_residual_rms_device, FusedLibraryDeviceBuffer,
    FusedLibraryDispatchReceipt, FusedQkvDeviceDispatch, FusedResidualRmsDeviceDispatch,
};
use crate::kernel::{HostError, HostResult};

use super::receipt::{
    CompletionBoundary, DataFlowEdge, DeviceExecutionReceipt, EndOfRunReadback,
    KvCacheLifecycleReceipt, KvCacheMeasurement, KvCachePhaseTiming, KvCacheTimingReceipt,
    KvCacheTimingSpan, ReceiptBuffer,
};

type BufferKey = (u32, u32);

/// Session-owned storage for temporary PerStep and ObservationPoint buffers.
///
/// A checked-out handle is moved into [`ProgramSession::buffers`] for one
/// execution. At the step boundary it is returned here instead of being
/// freed, so the next execution can reuse the same device allocation. The
/// pool is never used for PerProgram weights or state.
#[derive(Default)]
struct IntermediateBufferPool {
    buffers: BTreeMap<BufferKey, DeviceHandle>,
}

impl IntermediateBufferPool {
    fn checkout(&mut self, key: BufferKey) -> Option<DeviceHandle> {
        self.buffers.remove(&key)
    }

    fn return_buffer(&mut self, key: BufferKey, handle: DeviceHandle) {
        if self.buffers.insert(key, handle).is_some() {
            debug_assert!(false, "pool keys are unique per session");
        }
    }

    fn values(&self) -> impl Iterator<Item = &DeviceHandle> {
        self.buffers.values()
    }

    fn len(&self) -> usize {
        self.buffers.len()
    }

    fn clear(&mut self) {
        self.buffers.clear();
    }
}

/// Whether an execution copies host inputs into the declared input slots.
///
/// `SingleRun` executions copy per call (the one-shot-with-repeat surface);
/// `RepeatingStep` executions copy nothing — the HostProvided params were
/// once-init'd at session creation and stay device-resident (S5-U6); the
/// prepared resident-session surface (E03-U1) copies only the declared
/// `PerStep` input slots (the per-token values) and never the once-init
/// weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CopyMode {
    /// Copy declared host inputs per execution (`SingleRun`).
    PerStep,
    /// No copy-in: params are already device-resident from the once-init at
    /// session creation (`RepeatingStep`).
    OnceInit,
    /// Copy only the declared `PerStep` input slots per execution (the
    /// prepared resident-session surface, E03-U1); the once-init weights
    /// stay device-resident.
    ResidentStep,
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
    /// Optional owning fused QKV body selected from the explicit slot facts.
    fused_qkv: Option<FusedQkvPlan>,
    /// Optional owning fused residual-plus-RMS body selected from the
    /// explicit slot facts.
    fused_residual_rms: Option<FusedResidualRmsPlan>,
    /// 3D dispatch grid.
    grid: [u32; 3],
    /// 3D block (threadgroup) shape.
    block: [u32; 3],
}

/// One buffer slot of a session kernel.
#[derive(Clone)]
struct SessionSlot {
    /// Program-level buffer identity.
    buffer_id: u32,
    /// Content version selecting the session buffer allocation.
    version: u32,
    /// Logical name for diagnostics.
    buffer_name: String,
    /// Slot role at this kernel.
    role: DeviceBufferRole,
    /// Explicit target-neutral binding index.
    binding: u32,
    /// Descriptor dtype.
    element_ty: DeviceDataType,
    /// Descriptor element count.
    element_count: u64,
}

/// A version-keyed device view used by the fused CPU bridge.
#[derive(Clone, Copy)]
struct FusedQkvSlot {
    key: BufferKey,
    binding: u32,
    dtype: DeviceDataType,
    element_count: u64,
}

/// Explicit facts for one derived QKV library entry. The grouped bind is
/// finalized at dispatch: the K/V output width needs the uploaded packed
/// format facts when the K/V targets are capacity-sized persistent caches,
/// and those facts arrive with the weight upload, after session creation.
#[derive(Clone)]
struct FusedQkvPlan {
    /// Activation rows in this invocation.
    rows: u64,
    /// Activation width / contracted weight dimension.
    hidden: u64,
    /// Logical Q output width (`q_heads * head_dim`).
    q_width: u64,
    /// Elements in one attention head (from the RoPE table extent).
    head_dim: u64,
    /// Backend-neutral launch grid carried into the bind.
    grid: [u32; 3],
    /// Whether the K/V outputs target the persistent cache arenas
    /// (capacity-sized) rather than rows-sized `.k_gemv` activations.
    kv_cache_target: bool,
    activation: FusedQkvSlot,
    weights: [FusedQkvSlot; 3],
    biases: [Option<FusedQkvSlot>; 3],
    rope: Option<(FusedQkvSlot, FusedQkvSlot)>,
    outputs: [FusedQkvSlot; 3],
    cursor: Option<FusedQkvSlot>,
}

/// Explicit facts for one derived residual-plus-RMS library entry. The
/// carrier kernel normalizes the residual stream only; its compiled body
/// carries the RMS epsilon as a baked literal, so the epsilon is learned
/// from the module image at session creation and unresolvable facts fail
/// closed rather than falling back to the skip-ignoring carrier launch.
#[derive(Clone)]
struct FusedResidualRmsPlan {
    /// Activation rows in this invocation.
    rows: u64,
    /// Activation and RMS row width.
    hidden: u64,
    /// RMS epsilon parsed from the compiled carrier kernel.
    epsilon: f32,
    residual: FusedQkvSlot,
    skip: FusedQkvSlot,
    gamma: FusedQkvSlot,
    output: FusedQkvSlot,
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

/// One declared end-of-run observation cloned from the descriptor at
/// session creation (U8/U9 repair): a buffer whose FINAL value is read back
/// exactly once at the declared completion boundary after the step loop —
/// never within a step. The descriptor-level admission validated the
/// lifetime class (`PerStep` forward/gradients, `PerProgram` params); the
/// session needs only the observed buffer identity + content version.
#[derive(Debug, Clone, Copy)]
struct SessionEndOfRunResult {
    /// Observed buffer identity.
    buffer_id: u32,
    /// Observed content version.
    version: u32,
}

/// Raw bytes supplied for a session-owned HostProvided weight buffer.
///
/// The dtype tag describes the byte transfer only. Packed GGUF regions use
/// [`DeviceDataType::U8`] because their quantized representation is not an
/// element type in the descriptor vocabulary; the bytes are never converted
/// through an intermediate `Vec<f32>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceByteBuffer {
    /// Exact source bytes. A session may add only the descriptor's admitted
    /// one-to-three-byte packed tail when the allocation is word-rounded.
    pub bytes: Vec<u8>,
    /// Dtype tag carried to the neutral device transfer surface.
    pub dtype: DeviceDataType,
    /// GGML block/pack geometry for a native packed region. `None` is retained
    /// for generic byte inputs and remains unknown to fused bodies.
    pub packed_format: Option<PackedStorageFormat>,
}

/// Per-version metadata captured at session creation for input validation,
/// lifetime-distinct allocation/release (S2-4), and the A10 declared
/// resource graph (S2-8).
struct SessionBufferMeta {
    /// Logical name of the first reference (resource-graph fact).
    name: String,
    /// Stable semantic value identity carried by the descriptor (F1).
    semantic_value: u32,
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
/// - the **`PerProgram` buffers** (allocated once at creation; persist across
///   executions; released at program end);
/// - the **`PerStep` buffers** (allocated per execution, recycled at the step
///   boundary — released at the end of each execution and re-allocated for
///   the next);
/// - the **`ObservationPoint` buffers** (allocated per execution, read back at
///   the declared observation point, then released — read-then-release).
///
/// [`ProgramSession::execute`] runs the ordered launch sequence once (one
/// step), synchronizes at the step boundary, reads back the declared
/// observation buffers, and releases the per-step + observation buffers.
/// It can be called repeatedly on the same session without reloading the
/// module or re-allocating `PerProgram` buffers. [`ProgramSession::teardown`]
/// performs the ordered release (remaining buffers then module).
///
/// The S1-4 [`crate::composite_host::CompositeHost::execute_descriptor`] surface is a single-run
/// convenience over this session (create → execute → teardown).
///
/// # Lifetime-distinct release (S2-4; the schema-debt closer)
///
/// The session consumes the descriptor's typed [`DeviceBufferLifetime`]
/// facts — it never derives a lifetime from slot role alone (that would be
/// coincidence, council 3):
/// - **`PerProgram`** — allocated once at session creation, released at program
///   end (persists across steps);
/// - **`PerStep`** — live within one step, recycled at the step boundary
///   (released between executions, re-allocated for the next);
/// - **`ObservationPoint`** — read back at the declared observation point, then
///   released (read-then-release); it is the only class the session reads
///   back, so a readback request for any other class fails closed (no
///   undeclared readback).
///
/// The [`DeviceProgramLifetime`] regime is carried and consumed as the
/// declared program fact: `SingleRun` is the one-shot-with-repeat surface
/// (each [`ProgramSession::execute`] call copies its declared host inputs
/// and re-runs the whole program). `RepeatingStep` is the training-loop
/// surface (S5-U6): `HostProvided` params are copied into their `PerProgram`
/// buffers exactly once at session creation via [`ProgramSession::init_params`]
/// and never re-copied; each [`ProgramSession::execute_step`] allocates the
/// step's `PerStep` and `ObservationPoint` buffers, runs the ordered launches
/// with no copy-in, synchronizes at the step boundary, reads back the
/// declared observation (the per-step loss trace), and recycles the
/// per-step buffers. The two regimes never mix: `execute` refuses a
/// `RepeatingStep` session and `execute_step` refuses a `SingleRun` one
/// (params once-init + never re-copied is the `RepeatingStep` contract).
///
/// # Prepared resident-session mode (E03-U1)
///
/// The composite host can also prepare a **resident session** for one
/// admitted model: a thin [`PreparedResidentSession`] layer over a
/// `RepeatingStep` session that once-inits the `HostProvided` weights at
/// prepare, reuses them across repeated decode executions (resident steps —
/// per-token inputs copied, weights never re-copied), resets the
/// prompt-scoped device-resident state (content cleared, allocation
/// retained), and counts prepare/reuse/reset/release facts in its receipt.
/// No new executor is invented — the prepared mode is exactly the
/// `RepeatingStep` once-init mechanism plus the resident-step copy class, the
/// session-scoped temporary buffer pool, and the state-clear operation below.
///
/// # End-of-run readback (S5A-U1)
///
/// A `RepeatingStep` run reads the DECLARED **end-of-run observation set**
/// (the final forward activations, the final gradients, the final trainable
/// params — the wire's `EndOfRun` cadence rows, carried by the descriptor
/// and validated fail-closed at [`DeviceDescriptor::validate`] before any
/// launch) back exactly ONCE at the declared completion boundary — the
/// final step's step-boundary sync — via [`ProgramSession::execute_final_step`]
/// (the final step keeps the declared end-of-run `PerStep` buffers live past
/// the step boundary) + [`ProgramSession::read_end_of_run`] (the one-shot
/// readback; `PerStep` buffers are read-then-released, `PerProgram` params stay
/// live until teardown). Within a step the only readback is still the loss
/// observation; the params stay `PerProgram` (once-init persistence across
/// steps is never disturbed by the observation). Residency: transfers =
/// per-step loss readbacks + the single end-of-run value readback, zero
/// per-step copy-in, no per-op readback.
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
    inner: ProgramInner,
}

/// Program-owned device state without the exclusive runtime borrow.
/// The paired executor holds two of these against one `&mut DeviceRuntime`.
pub(crate) struct ProgramInner {
    backend: DeviceBackend,
    device_name: String,
    module_handle: DeviceHandle,
    module_hash: u64,
    program_lifetime: DeviceProgramLifetime,
    buffers: BTreeMap<BufferKey, DeviceHandle>,
    intermediate_pool: IntermediateBufferPool,
    buffer_meta: BTreeMap<BufferKey, SessionBufferMeta>,
    /// Packed GGML facts attached to uploaded weight ids. These are upload
    /// facts, not guesses from the padded device allocation width.
    packed_formats: BTreeMap<u32, PackedStorageFormat>,
    /// RoPE pairing learned once per session for the fused QKV library
    /// bodies (interior mutability: the learning readback happens mid-step
    /// under a shared inner borrow). `None` = not yet learned.
    fused_rotate_half: std::cell::RefCell<Option<bool>>,
    kernels: Vec<SessionKernel>,
    launches: Vec<SessionLaunch>,
    data_flow: Vec<DataFlowEdge>,
    results: Vec<SessionResult>,
    end_of_run_results: Vec<SessionEndOfRunResult>,
    end_of_run_keys: BTreeSet<BufferKey>,
    end_of_run_read: bool,
    final_step_completed: bool,
    program_graph_hash: String,
    params_initialized: bool,
    closed: bool,
    load_module_us: u64,
    per_program_alloc_us: u64,
    kv_cache_timing: KvCacheTimingReceipt,
    /// PerProgram keys borrowed from the sibling program (weights/cache).
    /// Release skips these so the owner can free them once.
    shared_keys: BTreeSet<BufferKey>,
    /// True when `module_handle` is the sibling program's module.
    shared_module: bool,
}

/// One shareable PerProgram handle from a prepared program. The sibling
/// program binds the carried semantic identity instead of allocating a second
/// copy. Buffer ids may differ between the two static program descriptors.
pub(crate) struct SharedBufferOffer {
    handle: DeviceHandle,
    byte_length: u64,
    role: DeviceBufferRole,
    lifetime: DeviceBufferLifetime,
    initialization: DeviceBufferInitialization,
    element_ty: DeviceDataType,
    element_count: u64,
}

/// Shareable PerProgram handles from a prepared program. The sibling program
/// binds matching identities instead of allocating a second copy.
pub(crate) struct SharedProgramOffer {
    pub module_hash: u64,
    pub module_handle: DeviceHandle,
    pub buffers: BTreeMap<u32, SharedBufferOffer>,
}

impl ProgramInner {
    fn shared_offer(&self) -> SharedProgramOffer {
        let mut buffers = BTreeMap::new();
        for (key, handle) in &self.buffers {
            let Some(meta) = self.buffer_meta.get(key) else {
                continue;
            };
            if meta.lifetime == DeviceBufferLifetime::PerProgram {
                buffers.insert(
                    meta.semantic_value,
                    SharedBufferOffer {
                        handle: *handle,
                        byte_length: meta.byte_length,
                        role: meta.role,
                        lifetime: meta.lifetime,
                        initialization: meta.initialization,
                        element_ty: meta.element_ty,
                        element_count: meta.element_count,
                    },
                );
            }
        }
        SharedProgramOffer {
            module_hash: self.module_hash,
            module_handle: self.module_handle,
            buffers,
        }
    }

    pub(crate) fn owned_handle_count(&self) -> usize {
        let owned_buffers = self
            .buffers
            .keys()
            .filter(|key| !self.shared_keys.contains(key))
            .count();
        let module = usize::from(!self.shared_module && !self.closed);
        if self.closed {
            0
        } else {
            owned_buffers + self.intermediate_pool.len() + module
        }
    }

    pub(crate) fn program_graph_hash(&self) -> &str {
        &self.program_graph_hash
    }

    pub(crate) fn module_hash(&self) -> u64 {
        self.module_hash
    }

    pub(crate) fn kv_cache_timing(&self) -> KvCacheTimingReceipt {
        self.kv_cache_timing
    }
}

fn elapsed_us(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX)
}

fn relative_us(origin: Instant, point: Instant) -> u64 {
    u64::try_from(point.duration_since(origin).as_micros()).unwrap_or(u64::MAX)
}

/// Build a host-clock span without turning an unavailable timestamp into a
/// zero-valued observation.
fn host_timing_span(origin: Instant, start: Instant, end: Instant) -> KvCacheTimingSpan {
    let start_us = relative_us(origin, start);
    let end_us = relative_us(origin, end);
    KvCacheTimingSpan {
        start_us: KvCacheMeasurement::measured(start_us),
        end_us: KvCacheMeasurement::measured(end_us),
        duration_us: KvCacheMeasurement::measured(end_us.saturating_sub(start_us)),
    }
}

/// Project the optional per-encoder device timestamps onto one GPU-body
/// envelope. The driver only supplies a usable body timeline when it returns
/// one start and one duration for every encoded launch; otherwise the schema
/// keeps the body explicitly absent.
fn gpu_body_timing_span(launch_gpu_us: &[u64], launch_gpu_start_us: &[u64]) -> KvCacheTimingSpan {
    if launch_gpu_us.is_empty() || launch_gpu_us.len() != launch_gpu_start_us.len() {
        return KvCacheTimingSpan::not_measured();
    }
    let start_us = launch_gpu_start_us.iter().copied().min().unwrap_or(0);
    let end_us = launch_gpu_start_us
        .iter()
        .copied()
        .zip(launch_gpu_us.iter().copied())
        .map(|(start, duration)| start.saturating_add(duration))
        .max()
        .unwrap_or(start_us);
    KvCacheTimingSpan {
        start_us: KvCacheMeasurement::measured(start_us),
        end_us: KvCacheMeasurement::measured(end_us),
        duration_us: KvCacheMeasurement::measured(end_us.saturating_sub(start_us)),
    }
}

fn derived_slack_us(wall_us: u64, phase: KvCachePhaseTiming) -> KvCacheMeasurement {
    let named_us = [
        phase.gpu_body.duration_us,
        phase.encode.duration_us,
        phase.submit.duration_us,
        phase.wait.duration_us,
    ]
    .into_iter()
    .filter_map(|measurement| match measurement {
        KvCacheMeasurement::Measured { value_us } | KvCacheMeasurement::Derived { value_us } => {
            Some(value_us)
        }
        KvCacheMeasurement::NotMeasured => None,
    })
    .fold(0u64, u64::saturating_add);
    KvCacheMeasurement::derived(wall_us.saturating_sub(named_us))
}

/// Copy every unique `PerStep` Input buffer once for a resident step.
///
/// The launch loop must not re-marshal the same invocation slot per kernel:
/// the baked session already owns the launch plan, and weights stay
/// device-resident from prepare. Missing or mis-sized invocation inputs
/// fail closed before any encode.
fn copy_resident_inputs(
    runtime: &mut DeviceRuntime,
    buffers: &BTreeMap<BufferKey, DeviceHandle>,
    buffer_meta: &BTreeMap<BufferKey, SessionBufferMeta>,
    inputs: &BTreeMap<u32, Vec<f32>>,
) -> HostResult<usize> {
    let mut copies = 0usize;
    let mut copied: BTreeSet<BufferKey> = BTreeSet::new();
    for (key, meta) in buffer_meta {
        if meta.role != DeviceBufferRole::Input
            || meta.lifetime != DeviceBufferLifetime::PerStep
            || !copied.insert(*key)
        {
            continue;
        }
        let values = inputs.get(&key.0).ok_or_else(|| {
            descriptor_errors::shape_mismatch(format!(
                "resident step declares PerStep input `{}` (id {}) but no host input was provided",
                meta.name, key.0
            ))
        })?;
        if u64::try_from(values.len()).ok() != Some(meta.element_count) {
            return Err(descriptor_errors::shape_mismatch(format!(
                "input for buffer `{}` (id {}) has {} f32 elements but the resident session declares {}",
                meta.name,
                key.0,
                values.len(),
                meta.element_count
            )));
        }
        let handle = buffers
            .get(key)
            .copied()
            .ok_or_else(|| HostError::internal("session input buffer disappeared"))?;
        runtime.copy_in_f32(&handle, values)?;
        copies += 1;
    }
    Ok(copies)
}

fn zero_fill_buffer(
    runtime: &mut DeviceRuntime,
    handle: &DeviceHandle,
    byte_length: u64,
) -> HostResult<()> {
    let zero_count = usize::try_from(byte_length / 4)
        .map_err(|_| HostError::internal("zero-fill element count overflows usize"))?;
    runtime.copy_in_f32(handle, &vec![0.0; zero_count])
}

/// Copy this kernel's declared host inputs into their slots for this
/// execution, returning the number of copy-ins performed. `PerStep`
/// mode copies every declared Input slot (`SingleRun`); `ResidentStep`
/// mode copies only the declared `PerStep` Input slots — the per-token
/// values — and never the once-init weights, which stay device-resident
/// (E03-U1); `OnceInit` mode copies nothing (`RepeatingStep`).
/// A free function over the disjoint session fields so the caller can
/// keep its immutable borrows of the launch plan and kernel table.
fn copy_declared_inputs(
    runtime: &mut DeviceRuntime,
    buffers: &BTreeMap<BufferKey, DeviceHandle>,
    buffer_meta: &BTreeMap<BufferKey, SessionBufferMeta>,
    kernel: &SessionKernel,
    inputs: &BTreeMap<u32, Vec<f32>>,
    mode: CopyMode,
    byte_inputs: &BTreeSet<u32>,
) -> HostResult<usize> {
    let mut copies = 0usize;
    for slot in &kernel.slots {
        if slot.role != DeviceBufferRole::Input {
            continue;
        }
        // Packed weight inputs were uploaded directly against their private
        // PerProgram session buffers before this launch loop. Do not ask the
        // legacy f32 map to re-marshal them.
        if byte_inputs.contains(&slot.buffer_id) {
            continue;
        }
        let is_per_step = buffer_meta
            .get(&(slot.buffer_id, slot.version))
            .is_some_and(|meta| meta.lifetime == DeviceBufferLifetime::PerStep);
        let copies_this_mode = match mode {
            CopyMode::PerStep => true,
            CopyMode::ResidentStep => is_per_step,
            CopyMode::OnceInit => false,
        };
        if !copies_this_mode {
            continue;
        }
        let values = inputs.get(&slot.buffer_id).ok_or_else(|| {
            descriptor_errors::shape_mismatch(format!(
                    "descriptor kernel `{}` declares input buffer `{}` (id {}) but no host input was provided",
                    kernel.entry, slot.buffer_name, slot.buffer_id
                ))
        })?;
        let expected = buffer_meta
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
        let handle = buffers
            .get(&(slot.buffer_id, slot.version))
            .copied()
            .ok_or_else(|| HostError::internal("session input buffer disappeared"))?;
        runtime.copy_in_f32(&handle, values)?;
        copies += 1;
    }
    Ok(copies)
}

fn fused_slot(
    slots: &[SessionSlot],
    predicate: impl Fn(&SessionSlot) -> bool,
) -> Option<FusedQkvSlot> {
    slots
        .iter()
        .find(|slot| predicate(slot))
        .map(|slot| FusedQkvSlot {
            key: (slot.buffer_id, slot.version),
            binding: slot.binding,
            dtype: slot.element_ty,
            element_count: slot.element_count,
        })
}

/// Recognize the fused QKV carrier from its explicit resource facts.
///
/// The descriptor carries resource names, dtypes, counts, and binding
/// indices, but not a second host-side QKV schema. This builder admits only
/// the fully typed f32 shape where those facts determine one unambiguous
/// invocation; the K/V output width itself is finalized at dispatch
/// ([`finalize_fused_bind`]) because capacity-sized persistent K/V targets
/// do not carry the per-row width in their element count. Packed weight
/// byte widths are never mistaken for logical matrix elements. Their GGML
/// format arrives with the uploaded weight fact and the bridge fails closed
/// when it is absent.
fn build_fused_qkv_plan(
    entry: &str,
    slots: &[SessionSlot],
    buffer_meta: &BTreeMap<BufferKey, SessionBufferMeta>,
    grid: [u32; 3],
) -> Option<FusedQkvPlan> {
    if !entry.ends_with("_QkvProjection") {
        return None;
    }
    let activation = fused_slot(slots, |slot| {
        slot.buffer_name.ends_with(".a") && slot.element_ty == DeviceDataType::F32
    })?;
    let q_weight = fused_slot(slots, |slot| {
        slot.buffer_name.ends_with(".attn_q.weight")
            && matches!(slot.element_ty, DeviceDataType::F32 | DeviceDataType::U8)
    })?;
    let k_weight = fused_slot(slots, |slot| {
        slot.buffer_name.ends_with(".attn_k.weight")
            && matches!(slot.element_ty, DeviceDataType::F32 | DeviceDataType::U8)
    })?;
    let v_weight = fused_slot(slots, |slot| {
        slot.buffer_name.ends_with(".attn_v.weight")
            && matches!(slot.element_ty, DeviceDataType::F32 | DeviceDataType::U8)
    })?;
    let q_output = fused_slot(slots, |slot| {
        slot.buffer_name.ends_with(".q_gemv") && slot.element_ty == DeviceDataType::F32
    })?;
    let k_output = fused_slot(slots, |slot| {
        (slot.buffer_name.starts_with("kv.cache_k.") || slot.buffer_name.ends_with(".k_gemv"))
            && slot.element_ty == DeviceDataType::F32
            && slot.role != DeviceBufferRole::Input
    })?;
    let v_output = fused_slot(slots, |slot| {
        (slot.buffer_name.starts_with("kv.cache_v.") || slot.buffer_name.ends_with(".v_gemv"))
            && slot.element_ty == DeviceDataType::F32
            && slot.role != DeviceBufferRole::Input
    })?;
    let cos = fused_slot(slots, |slot| {
        slot.buffer_name == "prefill.rope.cos" && slot.element_ty == DeviceDataType::F32
    });
    let sin = fused_slot(slots, |slot| {
        slot.buffer_name == "prefill.rope.sin" && slot.element_ty == DeviceDataType::F32
    });
    if cos.is_some() != sin.is_some() {
        return None;
    }
    let biases = [
        fused_slot(slots, |slot| {
            slot.buffer_name.ends_with(".attn_q.bias") && slot.element_ty == DeviceDataType::F32
        }),
        fused_slot(slots, |slot| {
            slot.buffer_name.ends_with(".attn_k.bias") && slot.element_ty == DeviceDataType::F32
        }),
        fused_slot(slots, |slot| {
            slot.buffer_name.ends_with(".attn_v.bias") && slot.element_ty == DeviceDataType::F32
        }),
    ];
    if biases.iter().any(Option::is_some) && biases.iter().any(Option::is_none) {
        return None;
    }
    let hidden = buffer_meta
        .values()
        .find(|meta| meta.name.ends_with(".attn_norm.weight"))
        .filter(|meta| meta.element_ty == DeviceDataType::F32)
        .map(|meta| meta.element_count)?;
    if hidden == 0 || activation.element_count % hidden != 0 {
        return None;
    }
    let rows = activation.element_count / hidden;
    if rows == 0
        || q_output.element_count == 0
        || q_output.element_count % rows != 0
        || k_output.element_count != v_output.element_count
        || k_output.element_count == 0
        || k_weight.element_count == 0
        || v_weight.element_count == 0
    {
        return None;
    }
    let q_width = q_output.element_count / rows;
    if q_width == 0 {
        return None;
    }
    let head_dim = cos.zip(sin).and_then(|(cos, _)| {
        (cos.element_count % rows == 0).then_some(cos.element_count / rows * 2)
    })?;
    if head_dim == 0 || q_width % head_dim != 0 {
        return None;
    }
    if k_output.key.eq(&v_output.key) || (k_output.binding == v_output.binding) {
        return None;
    }
    let kv_cache_target = slots.iter().any(|slot| {
        (slot.buffer_id, slot.version) == k_output.key
            && slot.buffer_name.starts_with("kv.cache_k.")
    });
    Some(FusedQkvPlan {
        rows,
        hidden,
        q_width,
        head_dim,
        grid,
        kv_cache_target,
        activation,
        weights: [q_weight, k_weight, v_weight],
        biases,
        rope: cos.zip(sin),
        outputs: [q_output, k_output, v_output],
        cursor: fused_slot(slots, |slot| slot.buffer_name == "kv.invocation_state"),
    })
}

/// Parse the RMS epsilon from one compiled carrier kernel's MSL source.
///
/// The wire does not carry the epsilon; the producer bakes it into the
/// carrier body as the `sqrt(mean + <literal>)` scale expression. The
/// literal renders through the producer's f32 constant formatter (decimal
/// digits plus an `f` suffix), so parsing it back is exact. Any other body
/// shape, a missing literal, or a non-finite/non-positive value is
/// unresolvable: `None` fails the plan closed.
fn fused_residual_rms_epsilon(module_image: &[u8], entry: &str) -> Option<f32> {
    let source = std::str::from_utf8(module_image).ok()?;
    let declaration = format!("kernel void {entry}(");
    let body_start = source.find(&declaration)? + declaration.len();
    let body_end = source[body_start..]
        .find("kernel void ")
        .map(|offset| body_start + offset)
        .unwrap_or(source.len());
    let body = &source[body_start..body_end];
    let marker = "sqrt(mean + ";
    let literal_start = body.find(marker)? + marker.len();
    let literal = body[literal_start..]
        .split(')')
        .next()?
        .trim()
        .trim_end_matches(['f', 'F']);
    let epsilon = literal.parse::<f64>().ok()? as f32;
    (epsilon.is_finite() && epsilon > 0.0).then_some(epsilon)
}

/// Recognize the fused residual-plus-RMS carrier from its explicit resource
/// facts and bind the residual (pre-attention) and skip (attention output)
/// streams to the library body that adds them.
///
/// A `None` return means the entry is not a fused residual/RMS carrier. An
/// `Err` return is a recognized carrier whose slot or epsilon facts cannot
/// bind one unambiguous invocation: the fallback carrier launch would
/// normalize the residual stream and silently discard the skip input, so an
/// unresolvable plan fails the session instead of guessing.
fn build_fused_residual_rms_plan(
    entry: &str,
    slots: &[SessionSlot],
    module_image: &[u8],
) -> HostResult<Option<FusedResidualRmsPlan>> {
    if !entry.ends_with("_ResidualRmsNorm") {
        return Ok(None);
    }
    let unresolvable = |fact: &str| {
        HostError::invalid_args(format!(
            "fused ResidualRmsNorm kernel `{entry}` has unresolvable {fact}"
        ))
    };
    let f32_slot = |predicate: &dyn Fn(&SessionSlot) -> bool, label: &str| {
        fused_slot(slots, |slot| predicate(slot) && slot.element_ty == DeviceDataType::F32)
            .ok_or_else(|| unresolvable(label))
    };
    let residual = f32_slot(&|slot| slot.buffer_name.ends_with(".h"), "residual slot")?;
    let skip = f32_slot(&|slot| slot.buffer_name.ends_with(".o"), "skip slot")?;
    let gamma = f32_slot(
        &|slot| slot.buffer_name.ends_with(".ffn_norm.weight"),
        "gamma slot",
    )?;
    let output = f32_slot(&|slot| slot.buffer_name.ends_with(".f"), "output slot")?;
    let hidden = gamma.element_count;
    if hidden == 0 || residual.element_count % hidden != 0 {
        return Err(unresolvable("hidden width"));
    }
    let rows = residual.element_count / hidden;
    if rows == 0
        || skip.element_count != rows * hidden
        || output.element_count != rows * hidden
        || gamma.key.eq(&residual.key)
            || gamma.key.eq(&skip.key)
            || gamma.key.eq(&output.key)
    {
        return Err(unresolvable("slot geometry"));
    }
    let epsilon = fused_residual_rms_epsilon(module_image, entry)
        .ok_or_else(|| unresolvable("RMS epsilon (absent from the compiled carrier body)"))?;
    Ok(Some(FusedResidualRmsPlan {
        rows,
        hidden,
        epsilon,
        residual,
        skip,
        gamma,
        output,
    }))
}

/// Packed-bytes-per-matrix-row for one GGML format over a contracted width
/// (R-PACK-02 column stride: `ceil(width / block_elements) * block_bytes`).
fn fused_packed_row_bytes(hidden: u64, format: QuantizedFormat) -> Option<u64> {
    hidden
        .checked_add(format.block_elements().checked_sub(1)?)
        .map(|padded| padded / format.block_elements())
        .and_then(|blocks| blocks.checked_mul(format.block_bytes()))
        .filter(|row_bytes| *row_bytes > 0)
}

/// The packed projection width implied by a padded byte region, when the
/// region is an exact whole number of packed matrix rows.
fn fused_packed_width(slot: FusedQkvSlot, hidden: u64, format: QuantizedFormat) -> Option<u64> {
    let row_bytes = fused_packed_row_bytes(hidden, format)?;
    let bytes = slot.element_count.checked_mul(4)?;
    if bytes % row_bytes != 0 {
        return None;
    }
    Some(bytes / row_bytes)
}

/// Finalize the grouped bind for an admitted fused plan.
///
/// The K/V output width (`kv_heads * head_dim`) is resolved from explicit
/// facts only, in priority order: the K bias extent (exactly the KV width),
/// a rows-sized `.k_gemv` activation output, the uploaded packed weight
/// byte extent, or a dense f32 weight extent. Capacity-sized persistent
/// cache targets never yield the width by division by rows. Every candidate
/// must satisfy the GQA arithmetic and, when the sibling weight formats are
/// known, the packed byte extents must agree for Q and V as well.
fn finalize_fused_bind(
    plan: &FusedQkvPlan,
    packed_formats: &BTreeMap<u32, PackedStorageFormat>,
) -> HostResult<QkvProjectionBind> {
    let quantized = |slot: FusedQkvSlot| -> Option<QuantizedFormat> {
        packed_formats
            .get(&slot.key.0)
            .copied()
            .and_then(|format| QuantizedFormat::from_ggml_type_id(format.ggml_type_id()))
    };
    let q_format = quantized(plan.weights[0]);
    let k_format = quantized(plan.weights[1]);
    let v_format = quantized(plan.weights[2]);
    let mut candidates: Vec<u64> = Vec::new();
    if let Some(k_bias) = plan.biases[1] {
        candidates.push(k_bias.element_count);
    }
    // A rows-sized `.k_gemv` activation output carries the width directly;
    // a capacity-sized persistent cache target never does (its element
    // count is `kv_heads * capacity * head_dim`).
    if !plan.kv_cache_target && plan.outputs[1].element_count % plan.rows == 0 {
        candidates.push(plan.outputs[1].element_count / plan.rows);
    }
    if let Some(format) = k_format {
        if let Some(width) = fused_packed_width(plan.weights[1], plan.hidden, format) {
            candidates.push(width);
        }
    } else if plan.weights[1].element_count % plan.hidden == 0 {
        candidates.push(plan.weights[1].element_count / plan.hidden);
    }
    for kv_width in candidates {
        if kv_width == 0
            || kv_width % plan.head_dim != 0
            || plan.q_width % kv_width != 0
            || plan.outputs[1].element_count % kv_width != 0
        {
            continue;
        }
        // Sibling weight extents must agree with the candidate width.
        if let Some(format) = q_format {
            if fused_packed_width(plan.weights[0], plan.hidden, format) != Some(plan.q_width) {
                continue;
            }
        } else if plan.weights[0].element_count != plan.hidden * plan.q_width {
            continue;
        }
        if let Some(format) = v_format {
            if fused_packed_width(plan.weights[2], plan.hidden, format) != Some(kv_width) {
                continue;
            }
        } else if plan.weights[2].element_count != plan.hidden * kv_width {
            continue;
        }
        let kv_heads = kv_width / plan.head_dim;
        let q_per_kv = plan.q_width / kv_width;
        let kv_capacity = plan.outputs[1].element_count / kv_width;
        if kv_heads == 0 || q_per_kv == 0 || kv_capacity == 0 {
            continue;
        }
        let mut bind = QkvProjectionBind::grouped(
            plan.rows,
            plan.hidden,
            kv_heads,
            q_per_kv,
            plan.head_dim,
            plan.grid,
        );
        bind.kv_output_strides = [kv_capacity.saturating_mul(plan.head_dim), plan.head_dim, 1];
        return Ok(bind);
    }
    Err(HostError::invalid_args(format!(
        "fused QKV plan has no unambiguous KV width (rows {}, hidden {}, q_width {}, head_dim {}, k bytes {}, packed k format {:?})",
        plan.rows,
        plan.hidden,
        plan.q_width,
        plan.head_dim,
        plan.weights[1].element_count * 4,
        k_format.map(|format| format.spelling())
    )))
}

fn resolve_fused_buffer(
    buffers: &BTreeMap<BufferKey, DeviceHandle>,
    slot: FusedQkvSlot,
    byte_offset: u64,
    view_span: u64,
    packed_format: Option<PackedStorageFormat>,
) -> HostResult<FusedLibraryDeviceBuffer<'_>> {
    let handle = buffers
        .get(&slot.key)
        .ok_or_else(|| HostError::internal("fused library buffer disappeared during launch"))?;
    let capacity = handle
        .len_bytes()
        .ok_or_else(|| HostError::internal("fused library buffer has no byte length"))?;
    let end = byte_offset
        .checked_add(view_span)
        .ok_or_else(|| HostError::invalid_args("fused library output view overflows"))?;
    if end > capacity {
        return Err(HostError::invalid_args(format!(
            "fused library binding {} view ends at {end}, allocation is {capacity}",
            slot.binding
        )));
    }
    Ok(FusedLibraryDeviceBuffer {
        handle,
        dtype: slot.dtype,
        byte_offset,
        view_span,
        binding_index: slot.binding,
        packed_format,
    })
}

fn fused_cursor_position(
    runtime: &mut DeviceRuntime,
    buffers: &BTreeMap<BufferKey, DeviceHandle>,
    cursor: FusedQkvSlot,
) -> HostResult<u64> {
    // The cursor is a descriptor-declared f32 input: the producer uploads
    // the four fields as f32 values and the carrier kernels read them as
    // floats. The position is the first f32 VALUE, never its raw word
    // reinterpreted as u32 (f32 1.0 bits read as u32 is 0x3F800000).
    if cursor.dtype != DeviceDataType::F32 {
        return Err(HostError::invalid_args(format!(
            "fused library cursor must be f32, got {}",
            cursor.dtype.spelling()
        )));
    }
    let handle = buffers
        .get(&cursor.key)
        .ok_or_else(|| HostError::internal("fused library cursor disappeared during launch"))?;
    let values = runtime.readback_f32(handle)?;
    fused_cursor_position_from_words(&values)
}

/// Decode the append position from the cursor's f32 word values, failing
/// closed on anything that is not one non-negative integer-valued f32.
fn fused_cursor_position_from_words(values: &[f32]) -> HostResult<u64> {
    let position = values
        .first()
        .ok_or_else(|| HostError::invalid_args("fused library cursor is shorter than position"))?;
    if !position.is_finite() || *position < 0.0 || position.fract() != 0.0 {
        return Err(HostError::invalid_args(format!(
            "fused library cursor position {position} is not a non-negative integer f32"
        )));
    }
    let decoded = *position as u64;
    if decoded as f32 != *position {
        return Err(HostError::invalid_args(format!(
            "fused library cursor position {position} overflows the u64 cursor surface"
        )));
    }
    Ok(decoded)
}

fn dispatch_fused_qkv_plan(
    runtime: &mut DeviceRuntime,
    buffers: &BTreeMap<BufferKey, DeviceHandle>,
    packed_formats: &BTreeMap<u32, PackedStorageFormat>,
    entry: &str,
    plan: &FusedQkvPlan,
    rotate_half: bool,
) -> HostResult<FusedLibraryDispatchReceipt> {
    let mut bind = finalize_fused_bind(plan, packed_formats)?;
    bind.rotate_half = rotate_half;
    let position = plan
        .cursor
        .map(|cursor| fused_cursor_position(runtime, buffers, cursor))
        .transpose()?
        .unwrap_or(0);
    // The carrier rotates query row i at the cursor position + i; the K/V
    // arenas append at the same offset.
    bind.rope_position = position;
    let output_offset = position
        .checked_mul(bind.head_dim)
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| HostError::invalid_args("fused library cursor output offset overflows"))?;
    let output_views = [
        resolve_fused_buffer(
            buffers,
            plan.outputs[0],
            0,
            plan.outputs[0].element_count.saturating_mul(4),
            None,
        )?,
        resolve_fused_buffer(
            buffers,
            plan.outputs[1],
            output_offset,
            plan.outputs[1]
                .element_count
                .saturating_mul(4)
                .saturating_sub(output_offset),
            None,
        )?,
        resolve_fused_buffer(
            buffers,
            plan.outputs[2],
            output_offset,
            plan.outputs[2]
                .element_count
                .saturating_mul(4)
                .saturating_sub(output_offset),
            None,
        )?,
    ];
    let request = FusedQkvDeviceDispatch {
        library_entry: "QkvProjection",
        derived_entry: entry,
        decode_gemv: u32::from(bind.rows == 1),
        bind,
        activation: resolve_fused_buffer(
            buffers,
            plan.activation,
            0,
            plan.activation.element_count.saturating_mul(4),
            None,
        )?,
        weights: [
            resolve_fused_buffer(
                buffers,
                plan.weights[0],
                0,
                plan.weights[0].element_count.saturating_mul(4),
                packed_formats.get(&plan.weights[0].key.0).copied(),
            )?,
            resolve_fused_buffer(
                buffers,
                plan.weights[1],
                0,
                plan.weights[1].element_count.saturating_mul(4),
                packed_formats.get(&plan.weights[1].key.0).copied(),
            )?,
            resolve_fused_buffer(
                buffers,
                plan.weights[2],
                0,
                plan.weights[2].element_count.saturating_mul(4),
                packed_formats.get(&plan.weights[2].key.0).copied(),
            )?,
        ],
        biases: [
            plan.biases[0]
                .map(|slot| resolve_fused_buffer(buffers, slot, 0, slot.element_count * 4, None))
                .transpose()?,
            plan.biases[1]
                .map(|slot| resolve_fused_buffer(buffers, slot, 0, slot.element_count * 4, None))
                .transpose()?,
            plan.biases[2]
                .map(|slot| resolve_fused_buffer(buffers, slot, 0, slot.element_count * 4, None))
                .transpose()?,
        ],
        rope: plan
            .rope
            .map(|(cos, sin)| {
                Ok((
                    resolve_fused_buffer(buffers, cos, 0, cos.element_count * 4, None)?,
                    resolve_fused_buffer(buffers, sin, 0, sin.element_count * 4, None)?,
                ))
            })
            .transpose()?,
        outputs: output_views,
    };
    dispatch_fused_qkv_device(runtime, request)
}

/// Bind one admitted fused residual-plus-RMS plan and dispatch it through
/// the owning library body.
///
/// The views are dense f32 activations spanning their whole allocation; the
/// epsilon and row geometry were fixed at session creation.
fn dispatch_fused_residual_rms_plan(
    runtime: &mut DeviceRuntime,
    buffers: &BTreeMap<BufferKey, DeviceHandle>,
    plan: &FusedResidualRmsPlan,
) -> HostResult<()> {
    let view = |slot: FusedQkvSlot| {
        resolve_fused_buffer(buffers, slot, 0, slot.element_count.saturating_mul(4), None)
    };
    let request = FusedResidualRmsDeviceDispatch {
        library_entry: "ResidualRmsNorm",
        rows: plan.rows,
        hidden: plan.hidden,
        epsilon: plan.epsilon,
        residual: view(plan.residual)?,
        skip: view(plan.skip)?,
        gamma: view(plan.gamma)?,
        output: view(plan.output)?,
    };
    dispatch_fused_residual_rms_device(runtime, request)
}

/// Resolve the RoPE pairing for this session's fused QKV bodies.
///
/// The descriptor wire does not carry the rotate-half fact to the host (it
/// is baked into the compiled carrier MSL on the producer side), so the
/// pairing is learned once per session from the program's own artifact: the
/// compiled carrier launch publishes a reference Q with the correct pairing,
/// and the library body is probed under both pairings until its Q matches.
/// The winning probe's writes are left in the Q/K/V targets, so the winner's
/// dispatch is already live; a pairing that matches nothing fails closed.
fn fused_rotate_half(
    runtime: &mut DeviceRuntime,
    module_handle: &DeviceHandle,
    kernel: &SessionKernel,
    plan: &FusedQkvPlan,
    buffers: &BTreeMap<BufferKey, DeviceHandle>,
    packed_formats: &BTreeMap<u32, PackedStorageFormat>,
    learned: &std::cell::RefCell<Option<bool>>,
) -> HostResult<bool> {
    if plan.rope.is_none() {
        return Ok(false);
    }
    if let Some(value) = *learned.borrow() {
        return Ok(value);
    }
    let launch_buffers: Vec<DeviceHandle> = kernel
        .slots
        .iter()
        .map(|slot| {
            buffers
                .get(&(slot.buffer_id, slot.version))
                .copied()
                .ok_or_else(|| HostError::internal("session buffer disappeared during launch"))
        })
        .collect::<HostResult<_>>()?;
    runtime.launch_kernel(
        module_handle,
        &kernel.entry,
        &launch_buffers,
        kernel.grid,
        kernel.block,
    )?;
    runtime.sync()?;
    let q_handle = buffers
        .get(&plan.outputs[0].key)
        .ok_or_else(|| HostError::internal("fused library buffer disappeared during launch"))?;
    let reference = runtime.readback_f32(q_handle)?;
    let scale = reference
        .iter()
        .fold(0.0f32, |acc, value| acc.max(value.abs()))
        .max(1.0);
    for candidate in [false, true] {
        dispatch_fused_qkv_plan(
            runtime,
            buffers,
            packed_formats,
            &kernel.entry,
            plan,
            candidate,
        )?;
        runtime.sync()?;
        let probe = runtime.readback_f32(q_handle)?;
        let delta = probe
            .iter()
            .zip(&reference)
            .map(|(value, expected)| (value - expected).abs())
            .fold(0.0f32, f32::max);
        if delta <= 1.0e-3 * scale {
            *learned.borrow_mut() = Some(candidate);
            return Ok(candidate);
        }
    }
    Err(HostError::invalid_args(
        "fused QKV library body matched neither RoPE pairing against the compiled carrier Q",
    ))
}

impl<'host> ProgramSession<'host> {
    /// Create a program session: validate the descriptor, load the module
    /// once, and allocate every distinct **`PerProgram`** buffer once (`PerStep`
    /// and `ObservationPoint` buffers are allocated per execution — S2-4).
    ///
    /// # Errors
    /// - `E_DEVICE_DESCRIPTOR` — the descriptor targets a different backend
    ///   than the runtime session, or is structurally invalid;
    /// - `E_DEVICE_ABI_MISMATCH` / `E_DEVICE_DTYPE_MISMATCH` /
    ///   `E_DEVICE_SHAPE_MISMATCH` — typed descriptor conflicts (see
    ///   [`DeviceDescriptor::validate`]);
    /// - session-level failures (module load, allocation) bubble through.
    pub(crate) fn new(
        runtime: &'host mut DeviceRuntime,
        descriptor: &DeviceDescriptor,
        device_name: String,
    ) -> HostResult<Self> {
        Self::new_with_share(runtime, descriptor, device_name, None)
    }

    /// Prepare one program, optionally binding PerProgram weights/cache and
    /// the module handle from an already-prepared sibling.
    pub(crate) fn new_with_share(
        runtime: &'host mut DeviceRuntime,
        descriptor: &DeviceDescriptor,
        device_name: String,
        share: Option<&SharedProgramOffer>,
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
        // cross-process cache. A paired sibling with the same image reuses
        // that handle so the pair loads the shared module set, not a double.
        let load_module_started = Instant::now();
        let module_hash = fnv1a64(&descriptor.module_image);
        let (module_handle, shared_module) = match share {
            Some(share) if share.module_hash == module_hash => (share.module_handle, true),
            _ => (runtime.load_module(&descriptor.module_image)?, false),
        };
        let load_module_us = elapsed_us(load_module_started);

        // Allocate every distinct PerProgram buffer once (S2-4): they persist
        // for the program's lifetime. PerStep and ObservationPoint buffers
        // are not allocated here — they are allocated at each execution's
        // start and released at its step boundary / after readback. A
        // failure at any PerProgram allocation runs the error-path teardown
        // first (S2-3 release-on-error): the module and every already-
        // allocated buffer are released before the error escapes, so a
        // failed creation leaves `live_handle_count() == 0`. Matching
        // sibling PerProgram identities bind the existing handle.
        let mut buffers: BTreeMap<BufferKey, DeviceHandle> = BTreeMap::new();
        let mut buffer_meta: BTreeMap<BufferKey, SessionBufferMeta> = BTreeMap::new();
        let mut kernels: Vec<SessionKernel> = Vec::with_capacity(descriptor.kernels.len());
        let mut shared_keys: BTreeSet<BufferKey> = BTreeSet::new();
        let mut owned_handles: Vec<DeviceHandle> = Vec::new();

        let alloc_started = Instant::now();
        let result = (|| {
            for kernel in &descriptor.kernels {
                let mut slots = Vec::with_capacity(kernel.buffers.len());
                for slot in &kernel.buffers {
                    let key = (slot.buffer_id, slot.version);
                    let byte_length = slot.byte_length().ok_or_else(|| {
                        descriptor_errors::shape_mismatch(format!(
                            "device buffer `{}` (id {}) has an overflowing byte length",
                            slot.buffer_name, slot.buffer_id
                        ))
                    })?;
                    buffer_meta.entry(key).or_insert(SessionBufferMeta {
                        name: slot.buffer_name.clone(),
                        semantic_value: slot.semantic_value,
                        role: slot.role,
                        element_ty: slot.element_ty,
                        element_count: slot.element_count,
                        byte_length,
                        lifetime: slot.lifetime,
                        initialization: slot.initialization,
                    });
                    if !buffers.contains_key(&key)
                        && slot.lifetime == DeviceBufferLifetime::PerProgram
                    {
                        if let Some(shared) =
                            share.and_then(|share| share.buffers.get(&slot.semantic_value))
                        {
                            if shared.byte_length != byte_length
                                || shared.role != slot.role
                                || shared.lifetime != slot.lifetime
                                || shared.initialization != slot.initialization
                                || shared.element_ty != slot.element_ty
                                || shared.element_count != slot.element_count
                            {
                                return Err(descriptor_errors::descriptor(format!(
                                    "paired program semantic value {} has incompatible PerProgram storage",
                                    slot.semantic_value
                                )));
                            }
                            buffers.insert(key, shared.handle);
                            shared_keys.insert(key);
                        } else {
                            let byte_length_usize = usize::try_from(byte_length).map_err(|_| {
                                descriptor_errors::shape_mismatch(format!(
                                    "device buffer `{}` (id {}) needs {} bytes, which overflows the host address space",
                                    slot.buffer_name, slot.buffer_id, byte_length
                                ))
                            })?;
                            let handle = runtime.alloc_bytes(byte_length_usize)?;
                            // G4 (F5): honor the carried initialization axis —
                            // ZeroFill persistent state (accumulation buffers,
                            // optimizer state) is zeroed EXACTLY ONCE at
                            // allocation so repeated executions accumulate onto
                            // a defined initial state.
                            if slot.initialization == DeviceBufferInitialization::ZeroFill {
                                zero_fill_buffer(runtime, &handle, byte_length)?;
                            }
                            buffers.insert(key, handle);
                            owned_handles.push(handle);
                        }
                    }
                    slots.push(SessionSlot {
                        buffer_id: slot.buffer_id,
                        version: slot.version,
                        buffer_name: slot.buffer_name.clone(),
                        role: slot.role,
                        binding: slot.binding,
                        element_ty: slot.element_ty,
                        element_count: slot.element_count,
                    });
                }
                let fused_qkv =
                    build_fused_qkv_plan(&kernel.entry, &slots, &buffer_meta, kernel.grid);
                let fused_residual_rms =
                    build_fused_residual_rms_plan(&kernel.entry, &slots, &descriptor.module_image)?;
                kernels.push(SessionKernel {
                    entry: kernel.entry.clone(),
                    slots,
                    fused_qkv,
                    fused_residual_rms,
                    grid: kernel.grid,
                    block: kernel.block,
                });
            }
            Ok(())
        })();
        let per_program_alloc_us = elapsed_us(alloc_started);

        if let Err(error) = result {
            // Error-path teardown at creation (S2-3): release every buffer
            // this program allocated, then the module if this program loaded
            // it. Shared sibling handles stay with the owner.
            for handle in &owned_handles {
                drop(runtime.release(handle));
            }
            if !shared_module {
                drop(runtime.release(&module_handle));
            }
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
            inner: ProgramInner {
                backend: descriptor.backend,
                device_name,
                module_handle,
                module_hash,
                program_lifetime: descriptor.program_lifetime,
                buffers,
                intermediate_pool: IntermediateBufferPool::default(),
                buffer_meta,
                packed_formats: BTreeMap::new(),
                fused_rotate_half: std::cell::RefCell::new(None),
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
                results: descriptor
                    .results
                    .iter()
                    .map(|result| SessionResult {
                        buffer_id: result.buffer_id,
                        version: result.version,
                    })
                    .collect(),
                end_of_run_results: descriptor
                    .end_of_run_results
                    .iter()
                    .map(|end_of_run| SessionEndOfRunResult {
                        buffer_id: end_of_run.buffer_id,
                        version: end_of_run.version,
                    })
                    .collect(),
                end_of_run_keys: descriptor
                    .end_of_run_results
                    .iter()
                    .map(|end_of_run| (end_of_run.buffer_id, end_of_run.version))
                    .collect(),
                end_of_run_read: false,
                final_step_completed: false,
                program_graph_hash: descriptor.program_graph_hash(),
                params_initialized: false,
                closed: false,
                load_module_us,
                per_program_alloc_us,
                kv_cache_timing: KvCacheTimingReceipt::not_measured(),
                shared_keys,
                shared_module,
            },
        })
    }

    /// Shareable PerProgram identities for a sibling program to bind.
    #[must_use]
    pub(crate) fn shared_offer(&self) -> SharedProgramOffer {
        self.inner.shared_offer()
    }

    /// Detach program state from the exclusive runtime borrow.
    pub(crate) fn into_inner(self) -> ProgramInner {
        self.inner
    }

    /// Reattach detached program state to a runtime borrow.
    pub(crate) fn from_inner<'a>(
        runtime: &'a mut DeviceRuntime,
        inner: ProgramInner,
    ) -> ProgramSession<'a> {
        ProgramSession { runtime, inner }
    }

    #[must_use]
    pub(crate) fn load_module_us(&self) -> u64 {
        self.inner.load_module_us
    }

    #[must_use]
    pub(crate) fn per_program_alloc_us(&self) -> u64 {
        self.inner.per_program_alloc_us
    }

    /// Resolve one declared HostProvided PerProgram weight id to its one
    /// session allocation. Multiple content versions cannot share one upload
    /// payload, so the ambiguity fails closed.
    fn weight_key(&self, id: u32) -> HostResult<BufferKey> {
        let mut matches = self
            .inner
            .buffer_meta
            .keys()
            .filter(|(buffer_id, _)| *buffer_id == id)
            .copied();
        let key = matches.next().ok_or_else(|| {
            HostError::internal(format!("session weight buffer {id} disappeared"))
        })?;
        if matches.next().is_some() {
            return Err(HostError::internal(format!(
                "session weight buffer {id} carries multiple content versions; one byte payload cannot select between them"
            )));
        }
        let meta = self
            .inner
            .buffer_meta
            .get(&key)
            .ok_or_else(|| HostError::internal("session weight metadata disappeared"))?;
        if meta.lifetime != DeviceBufferLifetime::PerProgram
            || meta.initialization != DeviceBufferInitialization::HostProvided
        {
            return Err(descriptor_errors::descriptor(format!(
                "session byte upload buffer {id} is not a HostProvided PerProgram weight"
            )));
        }
        Ok(key)
    }

    /// Upload packed weight bytes directly into the session's private
    /// PerProgram allocations. The neutral device surface receives the raw
    /// bytes and dtype tag. Metal wrap needs the caller's slice pointer to
    /// stay inside a retained mapping, so a backend that admits mapped
    /// retention gets the original borrow. Clone+pad is only the fallback
    /// when the backend cannot wrap and the packed region is a 1–3 byte
    /// word-tail short of the allocation.
    fn upload_weight_bytes_inner(
        &mut self,
        weights: &BTreeMap<u32, DeviceByteBuffer>,
    ) -> HostResult<BTreeSet<u32>> {
        let mut uploaded = BTreeSet::new();
        for (id, input) in weights {
            let key = self.weight_key(*id)?;
            if let Some(format) = input.packed_format {
                self.inner.packed_formats.insert(*id, format);
            }
            if self.inner.shared_keys.contains(&key) {
                uploaded.insert(*id);
                continue;
            }
            let meta = self
                .inner
                .buffer_meta
                .get(&key)
                .ok_or_else(|| HostError::internal("session weight metadata disappeared"))?;
            let expected = usize::try_from(meta.byte_length).map_err(|_| {
                HostError::internal(format!("session weight buffer {id} byte length overflows"))
            })?;
            if input.bytes.len() > expected || expected - input.bytes.len() >= 4 {
                return Err(descriptor_errors::shape_mismatch(format!(
                    "weight buffer `{}` (id {id}) expects {expected} bytes, got {}",
                    meta.name,
                    input.bytes.len()
                )));
            }
            if input.bytes.len() % input.dtype.byte_width() != 0 {
                return Err(descriptor_errors::shape_mismatch(format!(
                    "weight buffer `{}` (id {id}) has {} bytes, not aligned to dtype `{}`",
                    meta.name,
                    input.bytes.len(),
                    input.dtype.spelling()
                )));
            }
            let handle = self
                .inner
                .buffers
                .get(&key)
                .copied()
                .ok_or_else(|| HostError::internal("session weight buffer disappeared"))?;
            if self.runtime.supports_mapped_weight_retention() || input.bytes.len() == expected {
                self.runtime
                    .copy_in_bytes(&handle, &input.bytes, input.dtype)?;
            } else {
                let mut bytes = input.bytes.clone();
                bytes.resize(expected, 0);
                self.runtime.copy_in_bytes(&handle, &bytes, input.dtype)?;
            }
            self.runtime.record_weight_upload();
            uploaded.insert(*id);
        }
        Ok(uploaded)
    }

    /// Upload raw bytes for declared HostProvided PerProgram weights and run
    /// one SingleRun execution. The ordinary f32 input map remains available
    /// for invocation inputs; byte-uploaded slots are skipped by the legacy
    /// per-kernel f32 copy loop.
    pub fn execute_with_weight_bytes(
        &mut self,
        inputs: &BTreeMap<u32, Vec<f32>>,
        weights: &BTreeMap<u32, DeviceByteBuffer>,
    ) -> HostResult<DeviceExecutionReceipt> {
        if self.inner.closed {
            return Err(HostError::internal(
                "program session is closed after a failed execution; create a new session",
            ));
        }
        if self.inner.program_lifetime == DeviceProgramLifetime::RepeatingStep {
            return Err(HostError::internal(
                "a RepeatingStep session executes through init_params + execute_step: byte weights are uploaded at once-init, not per execution",
            ));
        }
        let byte_inputs = self.upload_weight_bytes_inner(weights);
        let result = match byte_inputs {
            Ok(byte_inputs) => {
                self.execute_inner(inputs, CopyMode::PerStep, false, false, &byte_inputs)
            }
            Err(error) => Err(error),
        };
        if result.is_err() {
            drop(self.release_all_handles());
            self.inner.closed = true;
        }
        result
    }

    /// Execute the ordered launch sequence once (one step) on the `SingleRun`
    /// surface: copies the declared host inputs into their input slots for
    /// this execution, then reuses the session's module and `PerProgram`
    /// buffers — does not reload the module or re-allocate `PerProgram`
    /// buffers. `PerStep` buffers are allocated for the step and recycled at
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
    /// per-execution input copy-in is the `SingleRun` surface. Use
    /// [`ProgramSession::init_params`] + [`ProgramSession::execute_step`].
    ///
    /// # Errors
    /// - `E_DEVICE_ENTRY_MISMATCH` — a kernel entry is unknown to the module;
    /// - `E_DEVICE_SHAPE_MISMATCH` — a declared input is missing or its size
    ///   contradicts the declared element count;
    /// - `E_INVALID_ARGS` — a declared observation buffer id was not
    ///   allocated by the session, or the id names a buffer whose lifetime is
    ///   not `ObservationPoint` (an undeclared readback fails closed, S2-4);
    /// - session-level failures (copy-in, launch, sync, readback) bubble
    ///   through unchanged.
    pub fn execute(
        &mut self,
        inputs: &BTreeMap<u32, Vec<f32>>,
    ) -> HostResult<DeviceExecutionReceipt> {
        if self.inner.closed {
            return Err(HostError::internal(
                "program session is closed after a failed execution; create a new session",
            ));
        }
        if self.inner.program_lifetime == DeviceProgramLifetime::RepeatingStep {
            return Err(HostError::internal(
                "a RepeatingStep session executes through init_params + execute_step: HostProvided params are copied in exactly once at session creation and never re-copied on later steps; per-execution input copy-in is the SingleRun surface",
            ));
        }
        let result = self.execute_inner(inputs, CopyMode::PerStep, false, false, &BTreeSet::new());
        if result.is_err() {
            // Release-on-error on all paths (S2-3): the ordered release runs
            // before the error escapes, then the session is closed so no
            // stale handle can be used again. Release failures on top of the
            // stage failure are secondary — every release is still attempted.
            drop(self.release_all_handles());
            self.inner.closed = true;
        }
        result
    }

    /// Once-init the declared `HostProvided` training params of a
    /// `RepeatingStep` session (S5-U6): each HostProvided `PerProgram` buffer
    /// receives its declared values exactly once, at session creation, and
    /// is never re-copied on later steps. The only buffers this copies are
    /// `PerProgram` + HostProvided; every such declared buffer must be present
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
        if self.inner.program_lifetime != DeviceProgramLifetime::RepeatingStep {
            return Err(HostError::internal(
                "once-init params are a RepeatingStep contract: a SingleRun session copies its declared host inputs per execution",
            ));
        }
        if self.inner.params_initialized {
            return Err(HostError::internal(
                "once-init params were already copied; a RepeatingStep session copies its HostProvided params exactly once at session creation",
            ));
        }
        let result = self.init_params_inner(params, &BTreeMap::new());
        match result {
            Ok(()) => {
                self.inner.params_initialized = true;
                Ok(())
            }
            Err(error) => {
                // Release-on-error on all paths (S2-3).
                drop(self.release_all_handles());
                self.inner.closed = true;
                Err(error)
            }
        }
    }

    /// Once-init a mixed set of ordinary f32 and packed-byte HostProvided
    /// PerProgram weights. Byte entries use the neutral dtype-tagged transfer
    /// surface and are never converted into f32 values.
    pub fn init_params_with_weight_bytes(
        &mut self,
        params: &BTreeMap<u32, Vec<f32>>,
        byte_params: &BTreeMap<u32, DeviceByteBuffer>,
    ) -> HostResult<()> {
        if self.inner.program_lifetime != DeviceProgramLifetime::RepeatingStep {
            return Err(HostError::internal(
                "once-init params are a RepeatingStep contract: a SingleRun session copies its declared host inputs per execution",
            ));
        }
        if self.inner.params_initialized {
            return Err(HostError::internal(
                "once-init params were already copied; a RepeatingStep session copies its HostProvided params exactly once at session creation",
            ));
        }
        let result = self.init_params_inner(params, byte_params);
        match result {
            Ok(()) => {
                self.inner.params_initialized = true;
                Ok(())
            }
            Err(error) => {
                drop(self.release_all_handles());
                self.inner.closed = true;
                Err(error)
            }
        }
    }

    /// The once-init body of [`ProgramSession::init_params`]: the copy loop
    /// over the declared HostProvided `PerProgram` params, without the
    /// error-path release, which the caller owns.
    fn init_params_inner(
        &mut self,
        params: &BTreeMap<u32, Vec<f32>>,
        byte_params: &BTreeMap<u32, DeviceByteBuffer>,
    ) -> HostResult<()> {
        // The declared param set: every distinct buffer id whose storage is
        // PerProgram and HostProvided (the F5 axis is carried, never
        // re-derived from role). A second content version of the same id
        // cannot be once-init'd from one value vector (a shape change is a
        // new version), so it fails closed.
        let mut param_ids: Vec<u32> = Vec::new();
        for ((id, _), meta) in &self.inner.buffer_meta {
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
        for id in byte_params.keys() {
            if params.contains_key(id) {
                return Err(HostError::invalid_args(format!(
                    "RepeatingStep param buffer {id} was supplied as both f32 and byte weights"
                )));
            }
        }
        let byte_ids = self.upload_weight_bytes_inner(byte_params)?;
        for id in param_ids {
            let key = self
                .inner
                .buffer_meta
                .keys()
                .find(|(buffer_id, _)| *buffer_id == id)
                .copied()
                .ok_or_else(|| HostError::internal("RepeatingStep param metadata disappeared"))?;
            let meta = &self.inner.buffer_meta[&key];
            if self.inner.shared_keys.contains(&key) || byte_ids.contains(&id) {
                continue;
            }
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
                .inner
                .buffers
                .get(&key)
                .copied()
                .ok_or_else(|| HostError::internal("RepeatingStep param buffer disappeared"))?;
            self.runtime.copy_in_f32(&handle, values)?;
            self.runtime.record_weight_upload();
        }
        Ok(())
    }

    /// Execute one training step on a `RepeatingStep` session whose params
    /// were once-init'd (S5-U6): allocate the step's `PerStep` +
    /// `ObservationPoint` buffers, run the ordered launch sequence with **no
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
        self.execute_step_impl(false)
    }

    /// Execute the FINAL training step of a `RepeatingStep` session (U8/U9
    /// repair): identical to [`ProgramSession::execute_step`] — allocate the
    /// step's `PerStep` + `ObservationPoint` buffers, run the ordered launches
    /// with no copy-in, synchronize at the step boundary, read back the
    /// per-step loss observation — except that the declared **end-of-run**
    /// `PerStep` buffers (the final forward activations and final gradients)
    /// stay live past the step boundary instead of recycling. The `PerProgram`
    /// params are already live (once-init at session creation). The session
    /// then reads the whole declared end-of-run set back once via
    /// [`ProgramSession::read_end_of_run`] at this step's completion
    /// boundary — the declared end-of-run boundary.
    ///
    /// **Error-path teardown is designed into this method (S2-3):** a failure
    /// at any stage runs the ordered release before the error escapes and
    /// closes the session, so a failed final step leaves
    /// `live_handle_count() == 0`.
    ///
    /// # Errors
    /// - `E_INTERNAL` — the session is not `RepeatingStep`, or the params
    ///   were not once-init'd;
    /// - session-level failures (launch, sync, readback) bubble through
    ///   unchanged.
    pub fn execute_final_step(&mut self) -> HostResult<DeviceExecutionReceipt> {
        self.execute_step_impl(true)
    }

    /// Execute one resident decode step on a `RepeatingStep` session whose
    /// weights were once-init'd (E03-U1, the prepared resident-session
    /// surface): allocate the step's `PerStep` + `ObservationPoint` buffers,
    /// copy the declared per-step inputs (the per-token values), run the
    /// ordered launch sequence with **no `PerProgram` copy-in** (the weights
    /// stay device-resident from the once-init at prepare), synchronize at
    /// the step boundary, read back the declared observation, and recycle
    /// the per-step buffers. The session never reloads the module and never
    /// re-allocates a `PerProgram` buffer across resident steps.
    ///
    /// **Error-path teardown is designed into this method (S2-3):** a failure
    /// at any stage runs the ordered release before the error escapes and
    /// closes the session, so a failed resident step leaves
    /// `live_handle_count() == 0`.
    ///
    /// # Errors
    /// - `E_INTERNAL` — the session is not `RepeatingStep`, or the weights
    ///   were not once-init'd;
    /// - `E_DEVICE_SHAPE_MISMATCH` — a declared per-step input is missing or
    ///   its size contradicts the declared element count;
    /// - session-level failures (copy-in, launch, sync, readback) bubble
    ///   through unchanged.
    pub(crate) fn execute_resident_step(
        &mut self,
        token_inputs: &BTreeMap<u32, Vec<f32>>,
    ) -> HostResult<DeviceExecutionReceipt> {
        if self.inner.closed {
            return Err(HostError::internal(
                "program session is closed after a failed execution; create a new session",
            ));
        }
        if self.inner.program_lifetime != DeviceProgramLifetime::RepeatingStep {
            return Err(HostError::internal(
                "a resident step is a RepeatingStep contract: the HostProvided weights are once-init'd at prepare and never re-copied on later steps",
            ));
        }
        if !self.inner.params_initialized {
            return Err(HostError::internal(
                "RepeatingStep weights were not once-init'd; prepare the resident session before resident steps",
            ));
        }
        let result = self.execute_inner(
            token_inputs,
            CopyMode::ResidentStep,
            false,
            true,
            &BTreeSet::new(),
        );
        if result.is_err() {
            // Release-on-error on all paths (S2-3).
            drop(self.release_all_handles());
            self.inner.closed = true;
        }
        result
    }

    /// The shared body of [`ProgramSession::execute_step`] and
    /// [`ProgramSession::execute_final_step`]: `keep_end_of_run` decides
    /// whether the declared end-of-run `PerStep` buffers stay live past the
    /// step boundary (the final step) or recycle like every other `PerStep`
    /// buffer (ordinary steps).
    fn execute_step_impl(&mut self, keep_end_of_run: bool) -> HostResult<DeviceExecutionReceipt> {
        if self.inner.closed {
            return Err(HostError::internal(
                "program session is closed after a failed execution; create a new session",
            ));
        }
        if self.inner.program_lifetime != DeviceProgramLifetime::RepeatingStep {
            return Err(HostError::internal(
                "execute_step is a RepeatingStep surface: a SingleRun session runs the whole program per execute call",
            ));
        }
        if !self.inner.params_initialized {
            return Err(HostError::internal(
                "RepeatingStep params were not once-init'd; call init_params before execute_step",
            ));
        }
        if keep_end_of_run && self.inner.final_step_completed {
            return Err(HostError::internal(
                "the final step already completed; a run has exactly one final step (its completion boundary is the declared end-of-run boundary)",
            ));
        }
        if keep_end_of_run && self.inner.end_of_run_read {
            return Err(HostError::internal(
                "the declared end-of-run set was already read back; the end-of-run readback runs exactly once after the final step",
            ));
        }
        let result = self.execute_inner(
            &BTreeMap::new(),
            CopyMode::OnceInit,
            keep_end_of_run,
            false,
            &BTreeSet::new(),
        );
        if result.is_err() {
            // Release-on-error on all paths (S2-3).
            drop(self.release_all_handles());
            self.inner.closed = true;
        } else if keep_end_of_run {
            self.inner.final_step_completed = true;
        }
        result
    }

    /// Read back the declared end-of-run observation set (U8/U9 repair):
    /// the final forward activations, the final gradients, and the final
    /// trainable params — **once**, after the step loop, at the declared
    /// completion boundary (the final step's step-boundary sync). The
    /// `PerStep` end-of-run buffers were kept live by the final step
    /// ([`ProgramSession::execute_final_step`]); the `PerProgram` params are
    /// live since session creation (once-init). The `PerStep` end-of-run
    /// buffers are read-then-released; the `PerProgram` params stay live
    /// until [`ProgramSession::teardown`] (per-program persistence is never
    /// disturbed by the observation). Within a step the only readback is
    /// still the loss observation — the end-of-run set is never read per
    /// step.
    ///
    /// **Error-path teardown is designed into this method (S2-3):** a failed
    /// end-of-run readback runs the ordered release before the error
    /// escapes and closes the session, so it leaves `live_handle_count() ==
    /// 0`.
    ///
    /// # Errors
    /// - `E_INTERNAL` — the session is not `RepeatingStep`, the set was
    ///   already read back, the final step has not completed (the `PerStep`
    ///   end-of-run buffers are not live), or a declared end-of-run buffer
    ///   is not live;
    /// - session-level readback failures bubble through unchanged.
    pub fn read_end_of_run(&mut self) -> HostResult<EndOfRunReadback> {
        if self.inner.closed {
            return Err(HostError::internal(
                "program session is closed after a failed execution; create a new session",
            ));
        }
        if self.inner.program_lifetime != DeviceProgramLifetime::RepeatingStep {
            return Err(HostError::internal(
                "end-of-run readback is a RepeatingStep contract: a SingleRun session has no step loop and no end-of-run boundary",
            ));
        }
        if self.inner.end_of_run_read {
            return Err(HostError::internal(
                "the declared end-of-run set was already read back; the end-of-run readback runs exactly once after the final step",
            ));
        }
        if !self.inner.end_of_run_results.is_empty() && !self.inner.final_step_completed {
            return Err(HostError::internal(
                "the final step has not completed; the declared end-of-run set is observable only at the declared completion boundary after the step loop",
            ));
        }
        let result = self.read_end_of_run_inner();
        if result.is_err() {
            // Release-on-error on all paths (S2-3).
            drop(self.release_all_handles());
            self.inner.closed = true;
        } else {
            self.inner.end_of_run_read = true;
        }
        result
    }

    /// The end-of-run readback body of [`ProgramSession::read_end_of_run`]:
    /// the read loop over the declared end-of-run set (`PerStep` buffers
    /// read-then-released; `PerProgram` params read and kept live), without
    /// the error-path release, which the caller owns.
    fn read_end_of_run_inner(&mut self) -> HostResult<EndOfRunReadback> {
        let mut values: BTreeMap<u32, Vec<f32>> = BTreeMap::new();
        let mut readback_count = 0usize;
        let mut released: Vec<BufferKey> = Vec::new();
        for end_of_run in &self.inner.end_of_run_results {
            let key = (end_of_run.buffer_id, end_of_run.version);
            let meta = self.inner.buffer_meta.get(&key).ok_or_else(|| {
                HostError::invalid_args(format!(
                    "declared end-of-run buffer id {} was not allocated by the session",
                    end_of_run.buffer_id
                ))
            })?;
            if meta.lifetime != DeviceBufferLifetime::PerStep
                && meta.lifetime != DeviceBufferLifetime::PerProgram
            {
                return Err(HostError::invalid_args(format!(
                    "declared end-of-run buffer id {} has lifetime `{}`; only per-step and per-program buffers are read back once at the end",
                    end_of_run.buffer_id,
                    meta.lifetime.spelling()
                )));
            }
            let handle = self
                .inner
                .buffers
                .get(&key)
                .copied()
                .ok_or_else(|| HostError::internal("session end-of-run buffer disappeared"))?;
            let observed = self.runtime.readback_f32(&handle)?;
            values.insert(end_of_run.buffer_id, observed);
            readback_count += 1;
            // PerStep end-of-run buffers are read-then-released (S2-4); the
            // PerProgram params stay live until teardown — per-program
            // persistence is never disturbed by the observation.
            if meta.lifetime == DeviceBufferLifetime::PerStep {
                released.push(key);
            }
        }
        for key in released {
            self.release_buffer(key)?;
        }
        Ok(EndOfRunReadback {
            values,
            readbacks: readback_count,
            transfers: readback_count,
        })
    }

    /// Prompt-scoped reset (E03-U1): clear the content of the device-resident
    /// state buffers — the `PerProgram` + `ZeroFill` class (the KV/SSM state
    /// that accumulates across token steps) — back to the zeroed initial
    /// state, **retaining the allocation**: no buffer is released and no
    /// buffer is re-allocated, so a subsequent prompt starts from the same
    /// zeroed initial condition as the first (deterministic replay matches
    /// token-for-token). The once-init `PerProgram` weights (`HostProvided`)
    /// are never touched. Returns the number of state buffers cleared.
    ///
    /// **Error-path teardown (S2-3):** a failed reset runs the ordered
    /// release before the error escapes and closes the session, so it leaves
    /// `live_handle_count() == 0`.
    ///
    /// # Errors
    /// - `E_INTERNAL` — the session is closed;
    /// - session-level copy failures bubble through unchanged.
    fn clear_resident_state(&mut self) -> HostResult<usize> {
        if self.inner.closed {
            return Err(HostError::internal(
                "program session is closed after a failed execution; create a new session",
            ));
        }
        let result = self.clear_resident_state_inner();
        if result.is_err() {
            // Release-on-error on all paths (S2-3).
            drop(self.release_all_handles());
            self.inner.closed = true;
        }
        result
    }

    /// The prompt-scoped reset body of [`ProgramSession::clear_resident_state`]:
    /// the zero-copy loop over the live `PerProgram` + `ZeroFill` state
    /// buffers, without the error-path release, which the caller owns.
    fn clear_resident_state_inner(&mut self) -> HostResult<usize> {
        let keys: Vec<BufferKey> = self
            .inner
            .buffers
            .iter()
            .filter(|(key, _)| {
                self.inner.buffer_meta.get(key).is_some_and(|meta| {
                    meta.lifetime == DeviceBufferLifetime::PerProgram
                        && meta.initialization == DeviceBufferInitialization::ZeroFill
                })
            })
            .map(|(key, _)| *key)
            .collect();
        let mut cleared = 0usize;
        for key in keys {
            let meta =
                self.inner.buffer_meta.get(&key).ok_or_else(|| {
                    HostError::internal("session state-buffer metadata disappeared")
                })?;
            let handle = self
                .inner
                .buffers
                .get(&key)
                .copied()
                .ok_or_else(|| HostError::internal("session state buffer disappeared"))?;
            zero_fill_buffer(&mut self.runtime, &handle, meta.byte_length)?;
            cleared += 1;
        }
        Ok(cleared)
    }

    /// The executable body shared by [`ProgramSession::execute`] and
    /// [`ProgramSession::execute_step`] / [`ProgramSession::execute_final_step`]:
    /// the ordered launch sequence (step-buffer allocation → copy-in
    /// (`SingleRun` only) → launch → step-boundary sync → observation
    /// readback + release → per-step release) without the error-path
    /// release, which the caller owns. `keep_end_of_run` (final-step only)
    /// keeps the declared end-of-run `PerStep` buffers live past the step
    /// boundary so the one-shot end-of-run readback can observe them.
    fn execute_inner(
        &mut self,
        inputs: &BTreeMap<u32, Vec<f32>>,
        mode: CopyMode,
        keep_end_of_run: bool,
        use_intermediate_pool: bool,
        byte_inputs: &BTreeSet<u32>,
    ) -> HostResult<DeviceExecutionReceipt> {
        let step_started = Instant::now();
        let mut launch_count = 0usize;
        let mut launch_ids = Vec::with_capacity(self.inner.launches.len());
        let mut launch_entries = Vec::with_capacity(self.inner.launches.len());
        let mut copy_ins = 0usize;
        // Snapshot Metal submit/wait counters before this step so the receipt
        // reports this execution's batch, not the session lifetime total.
        let submits_before = self.runtime.command_submit_count();
        let waits_before = self.runtime.blocking_wait_count();
        let mut copy_in_us = 0u64;
        let mut encode_us = 0u64;
        let mut pool_returns = 0usize;
        let mut fused_library_dispatches = Vec::new();

        // Resident decode steps check out this step's PerStep +
        // ObservationPoint buffers from the session-scoped pool. The first
        // step allocates them; later steps reuse the same handles. The
        // SingleRun and training-step surfaces retain their existing
        // allocate/release cadence. PerProgram buffers were allocated once at
        // session creation and stay live. A failure here runs the error-path
        // teardown (S2-3).
        let (pool_allocations, pool_reuses) = self.allocate_step_buffers(use_intermediate_pool)?;

        // Resident steps marshal invocation inputs once against the baked
        // session plan (unique PerStep slots). Weights are never re-copied.
        // SingleRun still copies per kernel; OnceInit copies nothing.
        if mode == CopyMode::ResidentStep {
            let copy_started = Instant::now();
            copy_ins = copy_resident_inputs(
                self.runtime,
                &self.inner.buffers,
                &self.inner.buffer_meta,
                inputs,
            )?;
            copy_in_us = elapsed_us(copy_started);
        }

        let encode_started = Instant::now();
        for launch in &self.inner.launches {
            let kernel =
                self.inner.kernels.get(launch.kernel_index).ok_or_else(|| {
                    HostError::internal("session launch references missing kernel")
                })?;
            // Resolve buffer handles for this kernel's launch (PerProgram
            // live from creation; PerStep/ObservationPoint just allocated).
            let mut launch_buffers: Vec<DeviceHandle> = Vec::with_capacity(kernel.slots.len());
            for slot in &kernel.slots {
                let key = (slot.buffer_id, slot.version);
                let handle = self.inner.buffers.get(&key).copied().ok_or_else(|| {
                    HostError::internal("session buffer disappeared during launch")
                })?;
                launch_buffers.push(handle);
            }

            // Copy-in declared inputs for this kernel — `SingleRun` copies
            // every declared input (PerStep mode); a prepared resident step
            // already copied unique PerStep slots above; a `RepeatingStep`
            // step (OnceInit mode) copies nothing: the HostProvided params
            // were once-init'd at session creation and stay device-resident
            // (S5-U6).
            if mode != CopyMode::ResidentStep {
                let copy_started = Instant::now();
                copy_ins += copy_declared_inputs(
                    self.runtime,
                    &self.inner.buffers,
                    &self.inner.buffer_meta,
                    kernel,
                    inputs,
                    mode,
                    byte_inputs,
                )?;
                copy_in_us = copy_in_us.saturating_add(elapsed_us(copy_started));
            }

            let encode_started = Instant::now();
            if let Some(plan) = &kernel.fused_qkv {
                // The carrier entry publishes Q only. Route the derived
                // library entry through its owning body instead, keeping the
                // extra K/V resources as explicit output targets.
                let rotate_half = fused_rotate_half(
                    self.runtime,
                    &self.inner.module_handle,
                    kernel,
                    plan,
                    &self.inner.buffers,
                    &self.inner.packed_formats,
                    &self.inner.fused_rotate_half,
                )?;
                fused_library_dispatches.push(dispatch_fused_qkv_plan(
                    self.runtime,
                    &self.inner.buffers,
                    &self.inner.packed_formats,
                    &kernel.entry,
                    plan,
                    rotate_half,
                )?);
            } else if let Some(plan) = &kernel.fused_residual_rms {
                // The carrier entry normalizes the residual stream only.
                // Route the derived library entry through the fused body
                // with the skip stream bound, or the residual add is lost.
                dispatch_fused_residual_rms_plan(
                    self.runtime,
                    &self.inner.buffers,
                    plan,
                )?;
            } else {
                self.runtime.launch_kernel(
                    &self.inner.module_handle,
                    &kernel.entry,
                    &launch_buffers,
                    kernel.grid,
                    kernel.block,
                )?;
            }
            encode_us = encode_us.saturating_add(elapsed_us(encode_started));
            launch_count += 1;
            launch_ids.push(launch.id);
            launch_entries.push(kernel.entry.clone());
        }
        let encode_ended = Instant::now();

        // Step-boundary synchronization: Metal commits the pending command
        // buffer and waits once here. CUDA issues `cuCtxSynchronize` (launches
        // already synced internally). Every encode in this step has completed
        // before any readback. The completion boundary is this barrier after
        // the last launch (R9).
        let submit_started = Instant::now();
        self.runtime.sync()?;
        let submit_ended = Instant::now();
        let gpu_encode_submit_wait_us = encode_us.saturating_add(elapsed_us(submit_started));
        let launch_gpu_us = self.runtime.take_encoder_gpu_us();
        let launch_gpu_start_us = self.runtime.take_encoder_gpu_start_us();

        // Observation-only readback (F6): read back exactly the DECLARED
        // observation points — the result rows projected from the
        // descriptor's observation facts at session creation. A buffer with
        // any other lifetime class is an undeclared readback and fails
        // closed. Resident observations are returned to the pool (M1-U4);
        // other execution surfaces release them at the step boundary.
        let mut release_count = 0usize;
        let mut readback_count = 0usize;
        let mut readbacks: BTreeMap<u32, Vec<f32>> = BTreeMap::new();
        let observed: Vec<SessionResult> = self.inner.results.clone();
        let readback_started = Instant::now();
        for result in &observed {
            let key = (result.buffer_id, result.version);
            let meta = self.inner.buffer_meta.get(&key).ok_or_else(|| {
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
                .inner
                .buffers
                .get(&key)
                .copied()
                .ok_or_else(|| HostError::internal("session observation buffer disappeared"))?;
            let values = self.runtime.readback_f32(&handle)?;
            readbacks.insert(result.buffer_id, values);
            readback_count += 1;
            if use_intermediate_pool {
                self.return_buffer_to_pool(key)?;
                pool_returns += 1;
            } else {
                self.release_buffer(key)?;
                release_count += 1;
            }
        }
        let readback_us = elapsed_us(readback_started);

        // Resident steps return PerStep buffers to the session-scoped pool
        // (M1-U4). Other execution surfaces release them at this boundary.
        // The FINAL step (U8/U9) keeps the declared end-of-run PerStep buffers
        // live so the one-shot end-of-run readback observes them once at the
        // declared completion boundary.
        let mut per_step_ids: Vec<BufferKey> = self
            .inner
            .buffers
            .iter()
            .filter(|(key, _)| {
                self.inner
                    .buffer_meta
                    .get(key)
                    .is_some_and(|meta| meta.lifetime == DeviceBufferLifetime::PerStep)
            })
            .map(|(key, _)| *key)
            .collect();
        if keep_end_of_run {
            per_step_ids.retain(|key| !self.inner.end_of_run_keys.contains(key));
        }
        for key in per_step_ids {
            if use_intermediate_pool {
                self.return_buffer_to_pool(key)?;
                pool_returns += 1;
            } else {
                self.release_buffer(key)?;
                release_count += 1;
            }
        }

        // Declared logical resource graph + data-flow edges (A10) from the
        // session's declared facts (the descriptor projected onto the
        // program).
        let (resource_graph, data_flow_edges) = self.declared_resource_graph();

        // W8-U1 / R9: Metal encodes every kernel into one command buffer and
        // commits+waits once at the step-boundary `sync()` (readback flush is
        // a no-op after that). Receipt `launches` / `syncs` are the actual
        // submits and blocking waits this execution performed — not the
        // encoded-kernel count (`launch_ids` / `launch_entries` still name
        // every kernel). CUDA still submits and syncs per kernel plus the
        // additive step-boundary `cuCtxSynchronize`.
        let (launches, syncs) = match self.inner.backend {
            DeviceBackend::Metal => (
                self.runtime
                    .command_submit_count()
                    .saturating_sub(submits_before),
                self.runtime
                    .blocking_wait_count()
                    .saturating_sub(waits_before),
            ),
            DeviceBackend::Cuda => (launch_count, launch_count + 1),
        };

        // F4H2: retain the live timing projection alongside the ordinary
        // execution receipt. The runtime exposes one combined step-boundary
        // sync clock, so it is recorded as the observed submit boundary and
        // the independent blocking-wait span remains explicitly absent.
        let steady_state = KvCachePhaseTiming {
            gpu_body: gpu_body_timing_span(&launch_gpu_us, &launch_gpu_start_us),
            encode: if launch_count == 0 {
                KvCacheTimingSpan::not_measured()
            } else {
                host_timing_span(step_started, encode_started, encode_ended)
            },
            submit: host_timing_span(step_started, submit_started, submit_ended),
            wait: KvCacheTimingSpan::not_measured(),
        };
        let wall_us = elapsed_us(step_started);
        self.inner.kv_cache_timing = KvCacheTimingReceipt {
            setup_phase: self.inner.kv_cache_timing.setup_phase,
            steady_state,
            slack_us: derived_slack_us(wall_us, steady_state),
            lifecycle: self.inner.kv_cache_timing.lifecycle,
        };

        Ok(DeviceExecutionReceipt {
            backend: self.inner.backend,
            device_name: self.inner.device_name.clone(),
            module_hash: self.inner.module_hash,
            launches,
            launch_ids,
            launch_entries,
            fused_library_dispatches,
            copy_ins,
            outputs: readbacks,
            allocated_buffers: self.allocated_buffers(),
            allocated_buffer_versions: self.allocated_buffer_versions(),
            pool_allocations,
            pool_reuses,
            pool_returns,
            program_lifetime: self.inner.program_lifetime,
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
            syncs,
            transfers: copy_ins + readback_count,
            readbacks: readback_count,
            releases: release_count,
            completion_boundary: CompletionBoundary::StepSync {
                after_launch: self
                    .inner
                    .launches
                    .last()
                    .map(|launch| launch.id)
                    .unwrap_or(0),
            },
            program_graph_hash: self.inner.program_graph_hash.clone(),
            copy_in_us,
            gpu_encode_submit_wait_us,
            readback_us,
            launch_gpu_us,
            launch_gpu_start_us,
        })
    }

    /// Allocate or check out this step's `PerStep` and `ObservationPoint`
    /// buffers. The resident decode surface uses the session-scoped pool;
    /// other surfaces retain allocate/release behavior. `PerProgram` buffers
    /// are already live from session creation. Buffer ids already live are
    /// left untouched, so an interrupted path is never double-allocated.
    fn allocate_step_buffers(&mut self, use_intermediate_pool: bool) -> HostResult<(usize, usize)> {
        let to_checkout: Vec<BufferKey> = self
            .inner
            .buffer_meta
            .iter()
            .filter(|(key, meta)| {
                meta.lifetime != DeviceBufferLifetime::PerProgram
                    && !self.inner.buffers.contains_key(key)
            })
            .map(|(key, _)| *key)
            .collect();
        let mut pool_allocations = 0usize;
        let mut pool_reuses = 0usize;
        for key in to_checkout {
            let meta = self
                .inner
                .buffer_meta
                .get(&key)
                .ok_or_else(|| HostError::internal("session buffer metadata disappeared"))?;
            let byte_length = usize::try_from(meta.byte_length).map_err(|_| {
                descriptor_errors::shape_mismatch(format!(
                    "device buffer `{}` (id {}) needs {} bytes, which overflows the host address space",
                    meta.name, key.0, meta.byte_length
                ))
            })?;
            let handle = if use_intermediate_pool {
                match self.inner.intermediate_pool.checkout(key) {
                    Some(handle) => {
                        pool_reuses += 1;
                        handle
                    }
                    None => {
                        pool_allocations += 1;
                        self.runtime.alloc_bytes(byte_length)?
                    }
                }
            } else {
                self.runtime.alloc_bytes(byte_length)?
            };
            // G4 (F5): honor the carried initialization axis at every
            // checkout — a ZeroFill step buffer (per-step accumulation state)
            // is reset before it comes live, whether its handle was newly
            // allocated or reused from the pool.
            if meta.initialization == DeviceBufferInitialization::ZeroFill {
                zero_fill_buffer(&mut self.runtime, &handle, meta.byte_length)?;
            }
            self.inner.buffers.insert(key, handle);
        }
        Ok((pool_allocations, pool_reuses))
    }

    /// Return one checked-out temporary buffer to the session-scoped pool.
    /// This is the pool equivalent of the old read-then-release / step-boundary
    /// release path: no device free occurs until session teardown.
    fn return_buffer_to_pool(&mut self, key: BufferKey) -> HostResult<()> {
        if let Some(handle) = self.inner.buffers.remove(&key) {
            self.inner.intermediate_pool.return_buffer(key, handle);
        }
        Ok(())
    }

    /// Release one live buffer by key (no-op when the key is not live). Used by
    /// the non-resident execution surfaces and by end-of-run readback.
    fn release_buffer(&mut self, key: BufferKey) -> HostResult<()> {
        if let Some(handle) = self.inner.buffers.remove(&key) {
            self.runtime.release(&handle)?;
        }
        Ok(())
    }

    /// The program's buffer ids classified by lifetime class (S2-4 receipt).
    fn buffers_by_lifetime(&self, lifetime: DeviceBufferLifetime) -> Vec<u32> {
        let mut ids = Vec::new();
        self.inner
            .buffer_meta
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
        self.inner
            .buffer_meta
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
            .inner
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
        (graph, self.inner.data_flow.clone())
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
        if self.inner.closed {
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
        let owned: Vec<DeviceHandle> = self
            .inner
            .buffers
            .iter()
            .filter(|(key, _)| !self.inner.shared_keys.contains(key))
            .map(|(_, handle)| *handle)
            .chain(self.inner.intermediate_pool.values().copied())
            .collect();
        for handle in owned {
            if let Err(error) = self.runtime.release(&handle) {
                first_error.get_or_insert(error);
            }
        }
        if !self.inner.shared_module {
            if let Err(error) = self.runtime.release(&self.inner.module_handle) {
                first_error.get_or_insert(error);
            }
        }
        // Shared sibling handles stay mapped on the owner; drop only this
        // program's keys so `session_handle_count()` reports reality.
        self.inner
            .buffers
            .retain(|key, _| self.inner.shared_keys.contains(key));
        self.inner.intermediate_pool.clear();
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// Ordered release without consuming the session (E03-U1): release every
    /// buffer, then the module, then mark the session closed so further use
    /// is refused and [`ProgramSession::session_handle_count`] reports 0.
    /// The prepared-session executor uses this so it can still report its
    /// receipt after the release. Every release is attempted even if one
    /// fails; the first failure bubbles through after every release has been
    /// attempted.
    pub fn release(&mut self) -> HostResult<()> {
        let result = self.release_all_handles();
        self.inner.closed = true;
        result
    }

    /// The program-level buffer ids this session manages (A9 receipt): every
    /// distinct buffer id the descriptor declares, classified by lifetime.
    /// `PerProgram` ids are live for the program's lifetime; `PerStep` and
    /// `ObservationPoint` ids are live only within one execution (S2-4).
    #[must_use]
    pub fn allocated_buffers(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        for (id, _) in self.inner.buffer_meta.keys() {
            if ids.last() != Some(id) {
                ids.push(*id);
            }
        }
        ids
    }

    /// The program's version-keyed buffer allocations.
    #[must_use]
    pub fn allocated_buffer_versions(&self) -> Vec<(u32, u32)> {
        self.inner.buffer_meta.keys().copied().collect()
    }

    /// Number of live device handles the session currently owns (module +
    /// checked-out buffers + pooled temporary buffers). Used by lifecycle
    /// tests to prove stable pool residency between executions and full
    /// release at teardown. A session closed by an error-path release (S2-3)
    /// holds no live handles and reports 0.
    #[must_use]
    pub fn session_handle_count(&self) -> usize {
        self.inner.owned_handle_count()
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
        self.inner.backend
    }

    /// The FNV-1a provenance hash of the loaded module.
    #[must_use]
    pub fn module_hash(&self) -> u64 {
        self.inner.module_hash
    }

    /// The SHA-256 program-graph receipt of the descriptor this session
    /// executes (roots + launches + dependency edges + buffer semantic
    /// identities + observation points + backend entry-name bytes, under the
    /// distinct host-graph domain tag — OQ1). The run/session identity the
    /// session consumed — distinct from [`ProgramSession::module_hash`],
    /// which only names the backend blob. Backend-entry-inclusive (S5A-U3);
    /// not a semantic-identity claim — the A10 complete-program SHA is the
    /// semantic identity.
    #[must_use]
    pub fn program_graph_hash(&self) -> &str {
        &self.inner.program_graph_hash
    }

    /// The latest live F4H1 timing projection. Prepared resident sessions
    /// attach their current lifecycle counters before exposing this value.
    #[must_use]
    fn kv_cache_timing(&self) -> KvCacheTimingReceipt {
        self.inner.kv_cache_timing
    }
}

// ---------------------------------------------------------------------------
// E03-U1: prepared resident-session executor
// ---------------------------------------------------------------------------

/// The descriptor buffer classes a prepared resident session needs (E03-U1).
struct PreparedBufferClasses {
    /// Distinct `PerProgram` + HostProvided ids (the once-init weights).
    host_provided_weights: usize,
    /// Distinct `PerStep` + `ObservationPoint` keys (allocated per reuse).
    per_execution_alloc_count: usize,
}

/// Project the descriptor's buffer classes onto the prepared-session axes
/// (E03-U1): which ids are the once-init weights, which are the
/// device-resident state, and how many buffers every reuse allocates. The
/// descriptor's validation has already proven cross-reference consistency,
/// so counting by buffer id is unambiguous.
fn prepared_buffer_classes(descriptor: &DeviceDescriptor) -> PreparedBufferClasses {
    let mut host_provided: BTreeSet<u32> = BTreeSet::new();
    let mut per_execution: BTreeSet<(u32, u32)> = BTreeSet::new();
    for kernel in &descriptor.kernels {
        for slot in &kernel.buffers {
            match slot.lifetime {
                DeviceBufferLifetime::PerProgram => match slot.initialization {
                    DeviceBufferInitialization::HostProvided => {
                        host_provided.insert(slot.buffer_id);
                    }
                    DeviceBufferInitialization::ZeroFill
                    | DeviceBufferInitialization::KernelInitialized => {}
                },
                DeviceBufferLifetime::PerStep | DeviceBufferLifetime::ObservationPoint => {
                    per_execution.insert((slot.buffer_id, slot.version));
                }
            }
        }
    }
    PreparedBufferClasses {
        host_provided_weights: host_provided.len(),
        per_execution_alloc_count: per_execution.len(),
    }
}

/// Lifecycle counts of one prepared resident session (E03-U1): prepare,
/// reuse, reset, release.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PreparedSessionCounters {
    /// Resident sessions prepared (weights once-init'd at prepare).
    pub prepares: usize,
    /// Decode executions reusing the resident weights.
    pub reuses: usize,
    /// Prompt-scoped resets performed.
    pub resets: usize,
    /// Ordered releases (teardowns) performed.
    pub releases: usize,
}

/// The prepared-session receipt (E03-U1): the lifecycle counts plus the
/// residency evidence — module reloads and PerProgram re-allocations between
/// reuses, derived from the driver counters against the prepare-time
/// baseline — and the live-handle state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedSessionReceipt {
    /// Selected backend.
    pub backend: DeviceBackend,
    /// Selected-hardware name from the admission probe.
    pub device_name: String,
    /// SHA-256 program-graph receipt of the prepared descriptor (OQ1).
    pub program_graph_hash: String,
    /// The prepare/reuse/reset/release lifecycle counts.
    pub counters: PreparedSessionCounters,
    /// Module loads observed beyond the prepare-time load. 0 across reuses =
    /// the resident session never reloaded the module (the E03-U1 first
    /// failing oracle).
    pub module_reloads: usize,
    /// PerProgram buffer allocations observed beyond the prepare-time
    /// allocations and the expected per-reuse step buffers. 0 across reuses
    /// = the resident weights/state were never re-allocated (the E03-U1
    /// first failing oracle).
    pub per_program_reallocs: usize,
    /// Live device handles at receipt time (0 after an ordered release).
    pub live_handles: usize,
    /// F4H1 setup/steady timing and lifecycle projection. The steady phase is
    /// replaced by each successful resident decode step; the explicit
    /// `not_measured` values remain visible when the driver lacks a split.
    pub timing: KvCacheTimingReceipt,
}

impl PreparedSessionReceipt {
    /// The canonical printed form (E03-U1 closeout evidence): the lifecycle
    /// counts and the residency facts.
    #[must_use]
    pub fn spelling(&self) -> String {
        format!(
            "prepared-session receipt: prepare={} reuse={} reset={} release={} reload={} realloc={} live-handles={} (backend {}, {})",
            self.counters.prepares,
            self.counters.reuses,
            self.counters.resets,
            self.counters.releases,
            self.module_reloads,
            self.per_program_reallocs,
            self.live_handles,
            self.backend.spelling(),
            self.program_graph_hash,
        )
    }
}

/// A prepared resident session (E03-U1): one admitted model bound once —
/// weights once-init'd and device-resident (`PerProgram` + HostProvided),
/// plus device-resident state (`PerProgram` + ZeroFill) — reused across
/// repeated decode executions and prompt-scoped resets without reloading the
/// module or re-allocating any `PerProgram` buffer, with countable lifecycle
/// facts (prepare/reuse/reset/release) and fail-closed behavior.
///
/// The executor is a thin [`ProgramSession`]-based layer over the
/// `RepeatingStep` once-init mechanism (S5-U6) — it does not invent a
/// parallel executor: the resident step reuses the step-machine (per-step
/// buffer allocation/recycle, ordered launches, observation readback) with a
/// per-token copy class, and the reset is a state-content clear that
/// retains allocation.
pub struct PreparedResidentSession<'host> {
    session: ProgramSession<'host>,
    backend: DeviceBackend,
    device_name: String,
    program_graph_hash: String,
    counters: PreparedSessionCounters,
    /// Driver-counter baselines at prepare (module loads / buffer allocs)
    /// for the reload/realloc derivation in [`PreparedResidentSession::receipt`].
    module_loads_at_prepare: usize,
    buffer_allocs_at_prepare: usize,
    /// Distinct `PerStep` + `ObservationPoint` buffers allocated on the
    /// first pool warm-up checkout.
    per_execution_alloc_count: usize,
    /// HostProvided weights copied exactly once during preparation.
    weight_uploads_at_prepare: usize,
    /// Closed after an error-path release (S2-3): every handle is gone and
    /// no further reuse/reset is possible.
    closed: bool,
}

impl<'host> PreparedResidentSession<'host> {
    /// Prepare one resident session from an admitted descriptor (E03-U1):
    /// validate the prepared-session shape (a `RepeatingStep` program with
    /// once-init `HostProvided` weights; an optional device-resident
    /// `ZeroFill` state may also be declared), create the underlying
    /// [`ProgramSession`] (module loaded once, every `PerProgram` buffer
    /// allocated once), and once-init the weights so they stay
    /// device-resident. The first failing oracle — a module
    /// reload or PerProgram re-allocation between reuses — is measured from
    /// the driver counters baselined here.
    ///
    /// # Errors
    /// - `E_DEVICE_DESCRIPTOR` — the descriptor is not a prepared-session
    ///   shape (wrong backend, not `RepeatingStep`, or no `HostProvided`
    ///   `PerProgram` weights);
    /// - session-level failures (module load, allocation, once-init) bubble
    ///   through; creation/once-init failures run the error-path teardown.
    pub fn prepare(
        runtime: &'host mut DeviceRuntime,
        descriptor: &DeviceDescriptor,
        weights: &BTreeMap<u32, Vec<f32>>,
        device_name: String,
    ) -> HostResult<Self> {
        descriptor.validate()?;
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
        if descriptor.program_lifetime != DeviceProgramLifetime::RepeatingStep {
            return Err(descriptor_errors::descriptor(
                "a prepared resident session is a RepeatingStep contract: its HostProvided weights are once-init'd at prepare and never re-copied; a SingleRun program is not a prepared session",
            ));
        }
        let classes = prepared_buffer_classes(descriptor);
        if classes.host_provided_weights == 0 {
            return Err(descriptor_errors::descriptor(
                "a prepared resident session requires once-init weights: at least one PerProgram + HostProvided buffer",
            ));
        }
        // Dense direct-loaded models may have no persistent prompt state at
        // all. The adapter still admits them: reset_prompt becomes a
        // deliberate no-op, while the PerProgram + HostProvided weights keep
        // the same once-init residency contract.
        let mut session = ProgramSession::new(runtime, descriptor, device_name.clone())?;
        session.init_params(weights)?;
        let counters = session.driver_counters();
        Ok(Self {
            program_graph_hash: descriptor.program_graph_hash(),
            backend: descriptor.backend,
            device_name,
            session,
            counters: PreparedSessionCounters {
                prepares: 1,
                ..PreparedSessionCounters::default()
            },
            module_loads_at_prepare: counters.module_loads,
            buffer_allocs_at_prepare: counters.buffer_allocs,
            per_execution_alloc_count: classes.per_execution_alloc_count,
            weight_uploads_at_prepare: classes.host_provided_weights,
            closed: false,
        })
    }

    /// Prepare a resident session while preserving packed weight bytes all the
    /// way into the private ProgramSession buffers. This convenience keeps the
    /// existing CompositeHost f32 API intact for ordinary callers while the
    /// device-execute weight path uses the neutral byte surface.
    pub fn prepare_with_weight_bytes(
        host: &'host mut super::CompositeHost,
        descriptor: &DeviceDescriptor,
        weights: &BTreeMap<u32, Vec<f32>>,
        byte_weights: &BTreeMap<u32, DeviceByteBuffer>,
    ) -> HostResult<Self> {
        let device_name = host.device_name().to_owned();
        let runtime = host.device_mut().ok_or_else(|| {
            descriptor_errors::no_device_program(
                "composite host has no device session; a device descriptor cannot be prepared",
            )
        })?;
        descriptor.validate()?;
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
        if descriptor.program_lifetime != DeviceProgramLifetime::RepeatingStep {
            return Err(descriptor_errors::descriptor(
                "a prepared resident session is a RepeatingStep contract: its HostProvided weights are once-init'd at prepare and never re-copied; a SingleRun program is not a prepared session",
            ));
        }
        let classes = prepared_buffer_classes(descriptor);
        if classes.host_provided_weights == 0 {
            return Err(descriptor_errors::descriptor(
                "a prepared resident session requires once-init weights: at least one PerProgram + HostProvided buffer",
            ));
        }
        let mut session = ProgramSession::new(runtime, descriptor, device_name.clone())?;
        session.init_params_with_weight_bytes(weights, byte_weights)?;
        let counters = session.driver_counters();
        Ok(Self {
            program_graph_hash: descriptor.program_graph_hash(),
            backend: descriptor.backend,
            device_name,
            session,
            counters: PreparedSessionCounters {
                prepares: 1,
                ..PreparedSessionCounters::default()
            },
            module_loads_at_prepare: counters.module_loads,
            buffer_allocs_at_prepare: counters.buffer_allocs,
            per_execution_alloc_count: classes.per_execution_alloc_count,
            weight_uploads_at_prepare: classes.host_provided_weights,
            closed: false,
        })
    }

    /// One resident decode execution (E03-U1 "reuse"): copy the declared
    /// per-step inputs (the per-token values) into their slots, run the
    /// ordered launch sequence on the resident weights/state (no module
    /// reload, no `PerProgram` re-allocation), synchronize, read back the
    /// declared observation, and recycle the per-step buffers. Counts one
    /// reuse. A failed reuse runs the ordered release, closes the prepared
    /// session, and leaves zero live handles (S2-3).
    ///
    /// # Errors
    /// - `E_INTERNAL` — the prepared session is closed;
    /// - `E_DEVICE_SHAPE_MISMATCH` — a declared per-token input is missing
    ///   or its size contradicts the declared element count;
    /// - session-level failures (copy-in, launch, sync, readback) bubble
    ///   through unchanged.
    pub fn execute_step(
        &mut self,
        token_inputs: &BTreeMap<u32, Vec<f32>>,
    ) -> HostResult<DeviceExecutionReceipt> {
        if self.session.inner.closed {
            return Err(HostError::internal(
                "prepared resident session is closed after a failure; prepare a new session",
            ));
        }
        let result = self.session.execute_resident_step(token_inputs);
        match result {
            Ok(receipt) => {
                self.counters.reuses += 1;
                Ok(receipt)
            }
            Err(error) => {
                self.session.inner.closed = true;
                Err(error)
            }
        }
    }

    /// Prompt-scoped reset (E03-U1): clear the content of the device-resident
    /// state buffers (the `PerProgram` + `ZeroFill` class) back to the
    /// zeroed initial condition, **retaining allocation** — no buffer is
    /// released and no buffer is re-allocated. After a reset, replaying the
    /// first prompt matches token-for-token (the state starts from the same
    /// zeroed condition). Counts one reset and returns the number of state
    /// buffers cleared. The once-init weights are never touched. A failed
    /// reset runs the ordered release and closes the prepared session
    /// (S2-3).
    ///
    /// # Errors
    /// - `E_INTERNAL` — the prepared session is closed;
    /// - session-level copy failures bubble through unchanged.
    pub fn reset_prompt(&mut self) -> HostResult<usize> {
        if self.session.inner.closed {
            return Err(HostError::internal(
                "prepared resident session is closed after a failure; prepare a new session",
            ));
        }
        let result = self.session.clear_resident_state();
        match result {
            Ok(cleared) => {
                self.counters.resets += 1;
                Ok(cleared)
            }
            Err(error) => {
                self.session.inner.closed = true;
                Err(error)
            }
        }
    }

    /// The current prepared-session receipt: the lifecycle counts and the
    /// residency evidence derived from the driver counters (module reloads
    /// and PerProgram re-allocations beyond the prepare-time baseline and the
    /// one-time pool warm-up allocations).
    #[must_use]
    pub fn receipt(&self) -> PreparedSessionReceipt {
        let counters = self.session.driver_counters();
        // The pool allocates each temporary key once, on the first reuse, and
        // then checks out those handles again. Subtract only that one warm-up
        // allocation set when deriving PerProgram reallocation facts.
        let pool_warmup_allocs =
            usize::from(self.counters.reuses > 0) * self.per_execution_alloc_count;
        let module_reloads = counters
            .module_loads
            .saturating_sub(self.module_loads_at_prepare);
        let per_program_reallocs = counters
            .buffer_allocs
            .saturating_sub(self.buffer_allocs_at_prepare)
            .saturating_sub(pool_warmup_allocs);
        let mut timing = self.session.kv_cache_timing();
        timing.lifecycle = KvCacheLifecycleReceipt {
            module_reloads: module_reloads as u64,
            persistent_reallocations: per_program_reallocs as u64,
            weight_uploads: self.weight_uploads_at_prepare as u64,
            // The generic prepared resident route has no KV-prefix copy or
            // full-cache clear operation. Keep both facts as measured zero.
            old_prefix_copy_bytes: 0,
            full_cache_clear_bytes: 0,
        };
        PreparedSessionReceipt {
            backend: self.session.inner.backend,
            device_name: self.session.inner.device_name.clone(),
            program_graph_hash: self.session.inner.program_graph_hash.clone(),
            counters: self.counters,
            module_reloads,
            per_program_reallocs,
            live_handles: self.session.session_handle_count(),
            timing,
        }
    }

    /// Number of live device handles the prepared session currently holds
    /// (module + `PerProgram` weights + `PerProgram` state; per-step and
    /// observation buffers are recycled per reuse).
    #[must_use]
    pub fn session_handle_count(&self) -> usize {
        self.session.session_handle_count()
    }

    /// Driver-level lifecycle counters (S2-2 module-cache leak bar).
    #[must_use]
    pub fn driver_counters(&self) -> DriverCounters {
        self.session.driver_counters()
    }

    /// The FNV-1a provenance hash of the loaded module.
    #[must_use]
    pub fn module_hash(&self) -> u64 {
        self.session.module_hash()
    }

    /// Ordered release (E03-U1 "release"): release every buffer then the
    /// module, leaving zero live handles, and return the final
    /// prepared-session receipt (release count included, live-handles 0). A
    /// session already closed by an error-path release has nothing left to
    /// release and returns the receipt.
    ///
    /// # Errors
    /// The first session-level release failure bubbles through after every
    /// release has been attempted.
    pub fn teardown(mut self) -> HostResult<PreparedSessionReceipt> {
        if !self.session.inner.closed {
            self.session.release()?;
        }
        self.counters.releases += 1;
        Ok(self.receipt())
    }
}

// ---------------------------------------------------------------------------
// KV-B B7: generic session binding materializer
// ---------------------------------------------------------------------------

/// Session-owned KV storage and launch-binding materializer.
///
/// One device allocation per persistent arena exposes the declared append and
/// prefix views. The invocation-state buffer is allocated once and copied
/// once per step; append and attention launches share that upload. Changing
/// live offsets reuses the same handles: no cache copy, no persistent
/// reallocation, no weight re-upload. Metal receives the live offsets on
/// the B6 bound-launch path.
pub struct KvCacheBindingSession<'host> {
    runtime: &'host mut DeviceRuntime,
    kv: KvCacheDescriptor,
    allocations: Vec<(u32, DeviceHandle)>,
    weights: Vec<(u32, DeviceHandle)>,
    cursor: InvocationStateBuffer,
    module: DeviceHandle,
    cursor_uploads: usize,
    weight_uploads: usize,
    cache_copies: usize,
    allocs_at_prepare: usize,
}

impl<'host> KvCacheBindingSession<'host> {
    /// Allocate each KV arena once, the cursor once, and any HostProvided
    /// weights once. Views share those arena handles; they do not allocate.
    ///
    /// # Errors
    /// Descriptor validation, allocation, module load, or a weight shape
    /// mismatch fail closed. Partial allocations are released first.
    pub fn prepare(
        runtime: &'host mut DeviceRuntime,
        kv: &KvCacheDescriptor,
        module_image: &[u8],
        weights: &[(DescriptorAllocation, &[f32])],
    ) -> HostResult<Self> {
        kv.validate()?;
        let module = runtime.load_module(module_image)?;
        let mut allocations: Vec<(u32, DeviceHandle)> = Vec::with_capacity(kv.allocations.len());
        let mut weight_handles: Vec<(u32, DeviceHandle)> = Vec::with_capacity(weights.len());
        let prepared = (|| {
            for allocation in &kv.allocations {
                let handle = runtime.alloc_bytes(allocation.capacity_bytes as usize)?;
                allocations.push((allocation.buffer_id, handle));
            }
            let cursor = runtime.alloc_invocation_state()?;
            let mut weight_uploads = 0usize;
            for (allocation, values) in weights {
                expect_weight_shape(allocation, values)?;
                let handle = runtime.alloc_bytes(allocation.capacity_bytes as usize)?;
                runtime.copy_in_f32(&handle, values)?;
                weight_handles.push((allocation.buffer_id, handle));
                weight_uploads += 1;
            }
            Ok((cursor, weight_uploads))
        })();
        let (cursor, weight_uploads) = match prepared {
            Ok(ok) => ok,
            Err(error) => {
                release_handles(runtime, allocations.iter().map(|(_, handle)| handle));
                release_handles(runtime, weight_handles.iter().map(|(_, handle)| handle));
                drop(runtime.release(&module));
                return Err(error);
            }
        };
        let allocs_at_prepare = runtime.driver_counters().buffer_allocs;
        Ok(Self {
            runtime,
            kv: kv.clone(),
            allocations,
            weights: weight_handles,
            cursor,
            module,
            cursor_uploads: 0,
            weight_uploads,
            cache_copies: 0,
            allocs_at_prepare,
        })
    }

    /// Live launch bindings for `state`. Handles are the session allocations;
    /// offsets and spans are the static envelope plus the tagged cursor field.
    pub fn materialize_bindings(
        &self,
        state: DescriptorInvocationState,
    ) -> HostResult<Vec<DeviceLaunchBinding>> {
        materialize_session_bindings(&self.kv, &self.allocations, state)
    }

    /// Copy the cursor once and return the live bindings for this step.
    /// Append and attention launches share this upload.
    pub fn begin_step(
        &mut self,
        state: DescriptorInvocationState,
    ) -> HostResult<Vec<DeviceLaunchBinding>> {
        self.runtime.upload_invocation_state(&self.cursor, state)?;
        self.cursor_uploads += 1;
        self.materialize_bindings(state)
    }

    /// Bound launch on the session module. Offsets reach Metal on the B6 path.
    pub fn launch_kernel_bound(
        &mut self,
        entry: &str,
        bindings: &[DeviceLaunchBinding],
        grid: [u32; 3],
        block: [u32; 3],
    ) -> HostResult<()> {
        self.runtime
            .launch_kernel_bound(&self.module, entry, bindings, grid, block)
    }

    /// Persistent allocation handle for a KV arena (not a view).
    #[must_use]
    pub fn allocation_handle(&self, buffer_id: u32) -> Option<DeviceHandle> {
        self.allocations
            .iter()
            .find_map(|(id, handle)| (*id == buffer_id).then_some(*handle))
    }

    /// HostProvided weight handle uploaded at prepare.
    #[must_use]
    pub fn weight_handle(&self, buffer_id: u32) -> Option<DeviceHandle> {
        self.weights
            .iter()
            .find_map(|(id, handle)| (*id == buffer_id).then_some(*handle))
    }

    /// The reusable invocation-state buffer. Identity is stable across steps.
    #[must_use]
    pub fn cursor_handle(&self) -> DeviceHandle {
        self.cursor.handle()
    }

    /// Cursor copies performed by [`Self::begin_step`]. One per step.
    #[must_use]
    pub fn cursor_uploads(&self) -> usize {
        self.cursor_uploads
    }

    /// Cache-data copies after prepare. The materializer never copies cache.
    #[must_use]
    pub fn cache_copies(&self) -> usize {
        self.cache_copies
    }

    /// Weight uploads. Exactly the prepare-time count; steps do not re-upload.
    #[must_use]
    pub fn weight_uploads(&self) -> usize {
        self.weight_uploads
    }

    /// Persistent buffer allocations beyond prepare. Steps must keep this at 0.
    #[must_use]
    pub fn persistent_reallocs(&self) -> usize {
        self.runtime
            .driver_counters()
            .buffer_allocs
            .saturating_sub(self.allocs_at_prepare)
    }

    /// Driver-level lifecycle counters (prepare baseline vs later steps).
    #[must_use]
    pub fn driver_counters(&self) -> DriverCounters {
        self.runtime.driver_counters()
    }

    /// Allocate a transient observation buffer. Not a persistent KV arena.
    pub fn alloc_bytes(&mut self, len_bytes: usize) -> HostResult<DeviceHandle> {
        let handle = self.runtime.alloc_bytes(len_bytes)?;
        self.allocs_at_prepare = self.runtime.driver_counters().buffer_allocs;
        Ok(handle)
    }

    /// Host→device copy into a session-owned handle (test seeding).
    pub fn copy_in_f32(&mut self, handle: &DeviceHandle, values: &[f32]) -> HostResult<()> {
        self.runtime.copy_in_f32(handle, values)
    }

    /// Device→host readback.
    pub fn readback_f32(&mut self, handle: &DeviceHandle) -> HostResult<Vec<f32>> {
        self.runtime.readback_f32(handle)
    }

    /// Step-boundary sync after the shared cursor upload's launches.
    pub fn sync(&mut self) -> HostResult<()> {
        self.runtime.sync()
    }
}

fn expect_weight_shape(allocation: &DescriptorAllocation, values: &[f32]) -> HostResult<()> {
    let expected = allocation.capacity_bytes / allocation.dtype.byte_width() as u64;
    if u64::try_from(values.len()).ok() != Some(expected) {
        return Err(descriptor_errors::shape_mismatch(format!(
            "weight allocation {} has {} f32 elements but its capacity holds {expected}",
            allocation.buffer_id,
            values.len()
        )));
    }
    Ok(())
}

fn release_handles<'a>(
    runtime: &mut DeviceRuntime,
    handles: impl IntoIterator<Item = &'a DeviceHandle>,
) {
    for handle in handles {
        drop(runtime.release(handle));
    }
}

/// Apply the tagged cursor to each declared launch record. Handles stay the
/// session allocations; only offset and span change.
fn materialize_session_bindings(
    kv: &KvCacheDescriptor,
    allocations: &[(u32, DeviceHandle)],
    state: DescriptorInvocationState,
) -> HostResult<Vec<DeviceLaunchBinding>> {
    kv.validate()?;
    let mut bindings = Vec::with_capacity(kv.launch_records().len());
    for record in kv.launch_records() {
        bindings.push(materialize_one_binding(kv, allocations, record, state)?);
    }
    Ok(bindings)
}

fn materialize_one_binding(
    kv: &KvCacheDescriptor,
    allocations: &[(u32, DeviceHandle)],
    record: &DescriptorLaunchBinding,
    state: DescriptorInvocationState,
) -> HostResult<DeviceLaunchBinding> {
    let handle = allocation_handle(allocations, record.handle)?;
    let allocation = kv
        .allocations
        .iter()
        .find(|allocation| allocation.buffer_id == record.handle)
        .ok_or_else(|| {
            descriptor_errors::descriptor(format!(
                "launch binding index {} names unknown allocation {}",
                record.binding_index, record.handle
            ))
        })?;
    let view = matching_static_view(kv, record).ok_or_else(|| {
        descriptor_errors::shape_mismatch(format!(
            "launch binding index {} offset {} span {} does not match a view on allocation {}",
            record.binding_index, record.byte_offset, record.view_span, record.handle
        ))
    })?;
    let row_step = row_step_bytes(kv, record.handle).ok_or_else(|| {
        descriptor_errors::descriptor(format!(
            "allocation {} has no view from which to derive a row step",
            record.handle
        ))
    })?;
    let (byte_offset, view_span) = live_envelope(record, view, allocation, state, row_step)?;
    Ok(DeviceLaunchBinding {
        handle,
        binding_index: record.binding_index,
        byte_offset,
        view_span,
        runtime_source: record.runtime_source,
    })
}

fn live_envelope(
    record: &DescriptorLaunchBinding,
    view: &DescriptorView,
    allocation: &DescriptorAllocation,
    state: DescriptorInvocationState,
    row_step: u64,
) -> HostResult<(u64, u64)> {
    let (byte_offset, view_span) = match record.runtime_source {
        DescriptorRuntimeSource::Constant | DescriptorRuntimeSource::SequenceEpoch => {
            (record.byte_offset, record.view_span)
        }
        DescriptorRuntimeSource::Position => (
            shift_offset(record.byte_offset, state.position, row_step)?,
            record.view_span,
        ),
        DescriptorRuntimeSource::ValidLenAfter => (
            record.byte_offset,
            scale_span(state.valid_len_after, row_step, view, record.binding_index)?,
        ),
        DescriptorRuntimeSource::QueryRows => (
            record.byte_offset,
            scale_span(state.query_rows, row_step, view, record.binding_index)?,
        ),
    };
    if view_span == 0 {
        return Err(descriptor_errors::descriptor(format!(
            "launch binding index {} has a zero view span",
            record.binding_index
        )));
    }
    let Some(end) = byte_offset.checked_add(view_span) else {
        return Err(descriptor_errors::shape_mismatch(format!(
            "launch binding index {} overflows its static envelope",
            record.binding_index
        )));
    };
    if end > allocation.capacity_bytes || view_span > view.maximum_span {
        return Err(descriptor_errors::shape_mismatch(format!(
            "launch binding index {} spans {view_span} bytes from offset {byte_offset} but allocation {} capacity is {} bytes (view maximum {})",
            record.binding_index,
            allocation.buffer_id,
            allocation.capacity_bytes,
            view.maximum_span
        )));
    }
    Ok((byte_offset, view_span))
}

fn shift_offset(base: u64, index: u32, step: u64) -> HostResult<u64> {
    let delta = u64::from(index).checked_mul(step).ok_or_else(|| {
        descriptor_errors::shape_mismatch("launch binding offset overflows its static envelope")
    })?;
    base.checked_add(delta).ok_or_else(|| {
        descriptor_errors::shape_mismatch("launch binding offset overflows its static envelope")
    })
}

fn scale_span(
    rows: u32,
    row_step: u64,
    view: &DescriptorView,
    binding_index: u32,
) -> HostResult<u64> {
    let span = u64::from(rows).checked_mul(row_step).ok_or_else(|| {
        descriptor_errors::shape_mismatch(format!(
            "launch binding index {binding_index} overflows its static envelope"
        ))
    })?;
    if span > view.maximum_span {
        return Err(descriptor_errors::shape_mismatch(format!(
            "launch binding index {binding_index} span {span} exceeds view maximum {}",
            view.maximum_span
        )));
    }
    Ok(span)
}

fn matching_static_view<'a>(
    kv: &'a KvCacheDescriptor,
    record: &DescriptorLaunchBinding,
) -> Option<&'a DescriptorView> {
    kv.views.iter().find(|view| {
        view.allocation_id == record.handle
            && view.static_base == record.byte_offset
            && view.maximum_span >= record.view_span
    })
}

fn row_step_bytes(kv: &KvCacheDescriptor, allocation_id: u32) -> Option<u64> {
    kv.views
        .iter()
        .filter(|view| view.allocation_id == allocation_id)
        .map(|view| view.maximum_span)
        .min()
}

fn allocation_handle(
    allocations: &[(u32, DeviceHandle)],
    allocation_id: u32,
) -> HostResult<DeviceHandle> {
    allocations
        .iter()
        .find_map(|(id, handle)| (*id == allocation_id).then_some(*handle))
        .ok_or_else(|| {
            descriptor_errors::descriptor(format!(
                "launch binding names allocation {allocation_id} with no live handle"
            ))
        })
}

#[cfg(test)]
mod fused_qkv_plan_tests {
    use super::*;

    fn slot(
        buffer_id: u32,
        name: &str,
        role: DeviceBufferRole,
        binding: u32,
        element_count: u64,
    ) -> SessionSlot {
        SessionSlot {
            buffer_id,
            version: 1,
            buffer_name: name.to_owned(),
            role,
            binding,
            element_ty: DeviceDataType::F32,
            element_count,
        }
    }

    fn norm_meta(name: &str, element_count: u64) -> (BufferKey, SessionBufferMeta) {
        (
            (9_000, 1),
            SessionBufferMeta {
                name: name.to_owned(),
                semantic_value: 0,
                role: DeviceBufferRole::Input,
                element_ty: DeviceDataType::F32,
                element_count,
                byte_length: element_count * 4,
                lifetime: DeviceBufferLifetime::PerProgram,
                initialization: DeviceBufferInitialization::HostProvided,
            },
        )
    }

    /// Qwen2.5-0.5B prefill layer-0 fused QKV slots, captured from the live
    /// descriptor (PB-6 slot dump): packed weights typed as f32 words
    /// (byte length / 4), capacity-sized persistent K/V targets, all biases.
    #[test]
    fn qwen_prefill_cache_targets_finalize_grouped_bind() {
        let slots = vec![
            slot(344, "prefill.blk0.a", DeviceBufferRole::InOut, 0, 15_232),
            slot(57, "blk.0.attn_q.weight", DeviceBufferRole::Input, 1, 137_984),
            slot(3, "prefill.rope.cos", DeviceBufferRole::Input, 2, 544),
            slot(4, "prefill.rope.sin", DeviceBufferRole::Input, 3, 544),
            slot(60, "blk.0.attn_q.bias", DeviceBufferRole::Input, 4, 896),
            slot(6, "kv.invocation_state", DeviceBufferRole::Input, 5, 4),
            slot(345, "prefill.blk0.q_gemv", DeviceBufferRole::InOut, 6, 15_232),
            slot(58, "blk.0.attn_k.weight", DeviceBufferRole::Input, 7, 19_712),
            slot(59, "blk.0.attn_v.weight", DeviceBufferRole::Input, 8, 30_464),
            slot(7, "kv.cache_k.0", DeviceBufferRole::InOut, 9, 1_048_576),
            slot(8, "kv.cache_v.0", DeviceBufferRole::InOut, 10, 1_048_576),
            slot(61, "blk.0.attn_k.bias", DeviceBufferRole::Input, 11, 128),
            slot(62, "blk.0.attn_v.bias", DeviceBufferRole::Input, 12, 128),
        ];
        let mut buffer_meta = BTreeMap::new();
        buffer_meta.insert(norm_meta("blk.0.attn_norm.weight", 896).0, norm_meta("blk.0.attn_norm.weight", 896).1);
        let plan = build_fused_qkv_plan(
            "prefill_blk_0_QkvProjection",
            &slots,
            &buffer_meta,
            [238, 1, 1],
        )
        .expect("qwen prefill plan builds against capacity-sized cache targets");
        assert_eq!(plan.rows, 17);
        assert_eq!(plan.hidden, 896);
        assert_eq!(plan.q_width, 896);
        assert_eq!(plan.head_dim, 64);
        assert!(plan.kv_cache_target);
        let mut packed = BTreeMap::new();
        packed.insert(57, PackedStorageFormat::Q5_0);
        packed.insert(58, PackedStorageFormat::Q5_0);
        packed.insert(59, PackedStorageFormat::Q8_0);
        let bind = finalize_fused_bind(&plan, &packed).expect("qwen bind finalizes");
        assert_eq!(bind.kv_heads, 2);
        assert_eq!(bind.q_per_kv, 7);
        assert_eq!(bind.kv_output_strides, [8192 * 64, 64, 1]);
        assert_eq!(bind.rotate_half, false);
    }

    /// SmolLM2-360M prefill layer-0 fused QKV slots (captured): no biases,
    /// packed Q5_0/Q8_0 weights — the KV width resolves from the uploaded
    /// packed byte extents, never from the capacity-sized cache count.
    #[test]
    fn smol_prefill_packed_cache_targets_resolve_kv_width_from_bytes() {
        let slots = vec![
            slot(360, "prefill.blk0.a", DeviceBufferRole::InOut, 0, 8_640),
            slot(73, "blk.0.attn_q.weight", DeviceBufferRole::Input, 1, 158_400),
            slot(3, "prefill.rope.cos", DeviceBufferRole::Input, 2, 288),
            slot(4, "prefill.rope.sin", DeviceBufferRole::Input, 3, 288),
            slot(6, "kv.invocation_state", DeviceBufferRole::Input, 4, 4),
            slot(361, "prefill.blk0.q_gemv", DeviceBufferRole::InOut, 5, 8_640),
            slot(74, "blk.0.attn_k.weight", DeviceBufferRole::Input, 6, 52_800),
            slot(75, "blk.0.attn_v.weight", DeviceBufferRole::Input, 7, 81_600),
            slot(7, "kv.cache_k.0", DeviceBufferRole::InOut, 8, 2_621_440),
            slot(8, "kv.cache_v.0", DeviceBufferRole::InOut, 9, 2_621_440),
        ];
        let mut buffer_meta = BTreeMap::new();
        buffer_meta.insert(norm_meta("blk.0.attn_norm.weight", 960).0, norm_meta("blk.0.attn_norm.weight", 960).1);
        let plan = build_fused_qkv_plan(
            "prefill_blk_0_QkvProjection",
            &slots,
            &buffer_meta,
            [135, 1, 1],
        )
        .expect("smol prefill plan builds without biases");
        assert_eq!(plan.rows, 9);
        assert_eq!(plan.head_dim, 64);
        let mut packed = BTreeMap::new();
        packed.insert(73, PackedStorageFormat::Q5_0);
        packed.insert(74, PackedStorageFormat::Q5_0);
        packed.insert(75, PackedStorageFormat::Q8_0);
        let bind = finalize_fused_bind(&plan, &packed).expect("smol bind finalizes from bytes");
        assert_eq!(bind.kv_heads, 5);
        assert_eq!(bind.q_per_kv, 3);
        assert_eq!(bind.kv_output_strides, [8192 * 64, 64, 1]);
    }

    /// SmolLM2 repeating decode session (captured): rows 10 divides the
    /// capacity-sized cache count evenly — the cache target must never be
    /// mistaken for a rows-sized `.k_gemv` output.
    #[test]
    fn smol_decode_cache_count_divisible_by_rows_still_resolves_from_bytes() {
        let slots = vec![
            slot(360, "prefill.blk0.a", DeviceBufferRole::InOut, 0, 9_600),
            slot(73, "blk.0.attn_q.weight", DeviceBufferRole::Input, 1, 158_400),
            slot(3, "prefill.rope.cos", DeviceBufferRole::Input, 2, 320),
            slot(4, "prefill.rope.sin", DeviceBufferRole::Input, 3, 320),
            slot(6, "kv.invocation_state", DeviceBufferRole::Input, 4, 4),
            slot(361, "prefill.blk0.q_gemv", DeviceBufferRole::InOut, 5, 9_600),
            slot(74, "blk.0.attn_k.weight", DeviceBufferRole::Input, 6, 52_800),
            slot(75, "blk.0.attn_v.weight", DeviceBufferRole::Input, 7, 81_600),
            slot(7, "kv.cache_k.0", DeviceBufferRole::InOut, 8, 2_621_440),
            slot(8, "kv.cache_v.0", DeviceBufferRole::InOut, 9, 2_621_440),
        ];
        let mut buffer_meta = BTreeMap::new();
        buffer_meta.insert(norm_meta("blk.0.attn_norm.weight", 960).0, norm_meta("blk.0.attn_norm.weight", 960).1);
        let plan = build_fused_qkv_plan(
            "prefill_blk_0_QkvProjection",
            &slots,
            &buffer_meta,
            [150, 1, 1],
        )
        .expect("smol decode plan builds");
        assert_eq!(plan.rows, 10);
        let mut packed = BTreeMap::new();
        packed.insert(73, PackedStorageFormat::Q5_0);
        packed.insert(74, PackedStorageFormat::Q5_0);
        packed.insert(75, PackedStorageFormat::Q8_0);
        let bind = finalize_fused_bind(&plan, &packed).expect("smol decode bind finalizes");
        assert_eq!(bind.kv_heads, 5);
        assert_eq!(bind.q_per_kv, 3);
    }

    /// A fused plan with no resolvable KV width (packed cache target, no
    /// bias, no uploaded format fact) fails closed instead of guessing.
    #[test]
    fn unresolvable_kv_width_fails_closed() {
        let slots = vec![
            slot(360, "prefill.blk0.a", DeviceBufferRole::InOut, 0, 8_640),
            slot(73, "blk.0.attn_q.weight", DeviceBufferRole::Input, 1, 158_400),
            slot(3, "prefill.rope.cos", DeviceBufferRole::Input, 2, 288),
            slot(4, "prefill.rope.sin", DeviceBufferRole::Input, 3, 288),
            slot(361, "prefill.blk0.q_gemv", DeviceBufferRole::InOut, 5, 8_640),
            slot(74, "blk.0.attn_k.weight", DeviceBufferRole::Input, 6, 52_800),
            slot(75, "blk.0.attn_v.weight", DeviceBufferRole::Input, 7, 81_600),
            slot(7, "kv.cache_k.0", DeviceBufferRole::InOut, 8, 2_621_440),
            slot(8, "kv.cache_v.0", DeviceBufferRole::InOut, 9, 2_621_440),
        ];
        let mut buffer_meta = BTreeMap::new();
        buffer_meta.insert(norm_meta("blk.0.attn_norm.weight", 960).0, norm_meta("blk.0.attn_norm.weight", 960).1);
        let plan = build_fused_qkv_plan(
            "prefill_blk_0_QkvProjection",
            &slots,
            &buffer_meta,
            [135, 1, 1],
        )
        .expect("plan builds; resolution is a dispatch-time fact");
        assert!(finalize_fused_bind(&plan, &BTreeMap::new()).is_err());
    }

    /// The cursor position is the first f32 VALUE of the descriptor-declared
    /// cursor input, never its raw word reinterpreted as u32. f32 1.0 read
    /// as u32 is 0x3F800000 = 1_065_353_216, which multiplied the k=3
    /// append offset into a ~272 GB view end against a 10 MiB K arena
    /// (SV-E5 binding-8 defect; same class as the PB-4d word/extent
    /// confusion).
    #[test]
    fn cursor_position_decodes_f32_value_not_raw_word() {
        let k3 = [1.0_f32, 4.0, 3.0, 1.0];
        assert_eq!(
            fused_cursor_position_from_words(&k3).expect("k=3 cursor decodes"),
            1,
            "f32 1.0 must decode to position 1, not 0x3F800000"
        );
        assert_eq!(
            fused_cursor_position_from_words(&[0.0, 9.0, 9.0, 0.0]).expect("prefill cursor"),
            0
        );
        assert_eq!(
            fused_cursor_position_from_words(&[8192.0, 8192.0, 1.0, 1.0]).expect("deep position"),
            8192
        );
    }

    /// Cursor positions that are not one non-negative integer-valued f32
    /// fail closed instead of truncating into an append offset.
    #[test]
    fn cursor_position_fails_closed_on_non_integer_words() {
        for bad in [
            Vec::new(),
            vec![-1.0_f32, 0.0, 1.0, 0.0],
            vec![1.5_f32, 4.0, 3.0, 1.0],
            vec![f32::NAN, 4.0, 3.0, 1.0],
            vec![f32::INFINITY, 4.0, 3.0, 1.0],
            vec![-0.5_f32, 4.0, 3.0, 1.0],
            vec![f32::MAX, 0.0, 0.0, 0.0],
        ] {
            assert!(
                fused_cursor_position_from_words(&bad).is_err(),
                "cursor {bad:?} must fail closed"
            );
        }
    }
}

#[cfg(test)]
mod fused_residual_rms_plan_tests {
    use super::*;

    fn slot(
        buffer_id: u32,
        name: &str,
        role: DeviceBufferRole,
        binding: u32,
        element_count: u64,
    ) -> SessionSlot {
        SessionSlot {
            buffer_id,
            version: 1,
            buffer_name: name.to_owned(),
            role,
            binding,
            element_ty: DeviceDataType::F32,
            element_count,
        }
    }

    /// Qwen2.5-0.5B prefill layer-0 fused residual/RMS slots: the residual
    /// is the layer-entry activation, the skip is the attention output
    /// projection, and the epsilon (1e-6 on this model) is baked into the
    /// carrier kernel while the neighboring attn_norm uses 1e-5 — the parse
    /// must bind the ResidualRmsNorm body's own literal.
    fn qwen_module_image() -> Vec<u8> {
        format!(
            concat!(
                "kernel void prefill_blk_0_attn_norm(float) {{\n",
                "    float scale = 1.0 / sqrt(mean + 0.00001f);\n",
                "}}\n",
                "kernel void prefill_blk_0_ResidualRmsNorm(float) {{\n",
                "    float mean = sumsq / float(896u);\n",
                "    float scale = 1.0 / sqrt(mean + 0.000001f);\n",
                "}}\n",
                "kernel void prefill_blk_0_ffn_gate(float) {{\n",
                "    return;\n",
                "}}\n",
            )
        )
        .into_bytes()
    }

    #[test]
    fn qwen_prefill_plan_binds_both_streams_and_carrier_epsilon() {
        let slots = vec![
            slot(400, "prefill.h", DeviceBufferRole::InOut, 0, 15_232),
            slot(41, "blk.0.ffn_norm.weight", DeviceBufferRole::Input, 1, 896),
            slot(401, "prefill.blk0.f", DeviceBufferRole::InOut, 2, 15_232),
            slot(402, "prefill.blk0.o", DeviceBufferRole::InOut, 3, 15_232),
        ];
        let plan = build_fused_residual_rms_plan(
            "prefill_blk_0_ResidualRmsNorm",
            &slots,
            &qwen_module_image(),
        )
        .expect("qwen prefill residual/RMS plan builds")
        .expect("entry matched");
        assert_eq!(plan.rows, 17);
        assert_eq!(plan.hidden, 896);
        assert_eq!(plan.epsilon, 1.0e-6f32);
        assert_eq!(plan.residual.key, (400, 1));
        assert_eq!(plan.skip.key, (402, 1));
        assert_eq!(plan.gamma.key, (41, 1));
        assert_eq!(plan.output.key, (401, 1));
    }

    #[test]
    fn non_residual_entry_admits_no_plan() {
        let plan = build_fused_residual_rms_plan(
            "prefill_blk_0_attn_norm",
            &[slot(1, "prefill.h", DeviceBufferRole::InOut, 0, 8)],
            &qwen_module_image(),
        )
        .expect("non-matching entry is not an error");
        assert!(plan.is_none());
    }

    #[test]
    fn missing_skip_slot_fails_closed() {
        let slots = vec![
            slot(400, "prefill.h", DeviceBufferRole::InOut, 0, 15_232),
            slot(41, "blk.0.ffn_norm.weight", DeviceBufferRole::Input, 1, 896),
            slot(401, "prefill.blk0.f", DeviceBufferRole::InOut, 2, 15_232),
        ];
        let error = build_fused_residual_rms_plan(
            "prefill_blk_0_ResidualRmsNorm",
            &slots,
            &qwen_module_image(),
        )
        .err()
        .expect("a recognized carrier without a skip slot must fail the session");
        assert!(error.message.contains("skip slot"));
    }

    #[test]
    fn epsilon_absent_from_carrier_body_fails_closed() {
        let slots = vec![
            slot(400, "prefill.h", DeviceBufferRole::InOut, 0, 15_232),
            slot(41, "blk.0.ffn_norm.weight", DeviceBufferRole::Input, 1, 896),
            slot(401, "prefill.blk0.f", DeviceBufferRole::InOut, 2, 15_232),
            slot(402, "prefill.blk0.o", DeviceBufferRole::InOut, 3, 15_232),
        ];
        let image = b"kernel void prefill_blk_0_ResidualRmsNorm(float) {\n return;\n}\n";
        let error = build_fused_residual_rms_plan(
            "prefill_blk_0_ResidualRmsNorm",
            &slots,
            image,
        )
        .err()
        .expect("an unparseable carrier epsilon must fail the session");
        assert!(error.message.contains("RMS epsilon"));
    }

    #[test]
    fn inconsistent_slot_geometry_fails_closed() {
        let slots = vec![
            slot(400, "prefill.h", DeviceBufferRole::InOut, 0, 15_232),
            slot(41, "blk.0.ffn_norm.weight", DeviceBufferRole::Input, 1, 896),
            slot(401, "prefill.blk0.f", DeviceBufferRole::InOut, 2, 8_960),
            slot(402, "prefill.blk0.o", DeviceBufferRole::InOut, 3, 15_232),
        ];
        let error = build_fused_residual_rms_plan(
            "prefill_blk_0_ResidualRmsNorm",
            &slots,
            &qwen_module_image(),
        )
        .err()
        .expect("an output width disagreeing with rows*hidden must fail the session");
        assert!(error.message.contains("slot geometry"));
    }
}
