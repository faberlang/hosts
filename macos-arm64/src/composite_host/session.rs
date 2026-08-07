//! The observation-cadence program-session executor (S2-1..S5A-U1): one
//! program-scoped device session per [`DeviceDescriptor`] that owns the
//! module and the `PerProgram` buffers, and runs the ordered launch sequence
//! with the declared readback cadence (per-step loss observation, end-of-run
//! readback) under lifetime-distinct allocation/release (S2-4).

use std::collections::{BTreeMap, BTreeSet};

use faber::device::{DeviceBackend, DeviceHandle};

use crate::device_descriptor::{
    errors as descriptor_errors, fnv1a64, DeviceBufferInitialization, DeviceBufferLifetime,
    DeviceBufferRole, DeviceDataType, DeviceDescriptor, DeviceProgramLifetime, E_DEVICE_DESCRIPTOR,
};
use crate::device_host::{DeviceRuntime, DeviceSession};
use crate::device_registry::DriverCounters;
use crate::kernel::{HostError, HostResult};

use super::receipt::{
    CompletionBoundary, DataFlowEdge, DeviceExecutionReceipt, EndOfRunReadback, ReceiptBuffer,
};

type BufferKey = (u32, u32);

/// Whether an execution copies host inputs into the declared input slots.
///
/// `SingleRun` executions copy per call (the one-shot-with-repeat surface);
/// `RepeatingStep` executions copy nothing — the HostProvided params were
/// once-init'd at session creation and stay device-resident (S5-U6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CopyMode {
    /// Copy declared host inputs per execution (`SingleRun`).
    PerStep,
    /// No copy-in: params are already device-resident from the once-init at
    /// session creation (`RepeatingStep`).
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
    backend: DeviceBackend,
    device_name: String,
    module_handle: DeviceHandle,
    module_hash: u64,
    /// Program execution-lifetime regime (S2-4).
    program_lifetime: DeviceProgramLifetime,
    /// Currently-live device buffers: (buffer_id, version) → device handle.
    /// `PerProgram` buffers are live from creation until teardown; `PerStep` and
    /// `ObservationPoint` buffers are live only within one execution.
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
    /// back **within a step**.
    results: Vec<SessionResult>,
    /// Declared end-of-run observations (S5A-U1): the result set the
    /// session reads back ONCE at the declared completion boundary after the
    /// step loop — never within a step. Carried by the descriptor from the
    /// wire's `EndOfRun` cadence rows (validated before any launch); may
    /// name `PerStep` buffers (final forward activations, final gradients)
    /// and `PerProgram` buffers (final trainable params, read only at the
    /// end).
    end_of_run_results: Vec<SessionEndOfRunResult>,
    /// The version-keyed keys of [`ProgramSession::end_of_run_results`]
    /// (the `PerStep` subset of the set stays live past the final step's
    /// boundary so the one-shot readback can observe it).
    end_of_run_keys: BTreeSet<BufferKey>,
    /// Whether the declared end-of-run set has been read back (read exactly
    /// once; a second readback fails closed).
    end_of_run_read: bool,
    /// Whether the FINAL step (`execute_final_step`) has completed, so the
    /// declared end-of-run set is live and observable at the declared
    /// completion boundary.
    final_step_completed: bool,
    /// FNV-1a hash of the carried execution-descriptor facts the session
    /// executes (backend-entry-inclusive; S5A-U3 — not a semantic-identity
    /// claim).
    execution_descriptor_hash: u64,
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
            // Declared end-of-run observations (S5A-U1): the wire's `EndOfRun`
            // cadence set, carried by the descriptor and cloned at session
            // creation (validated fail-closed at [`DeviceDescriptor::validate`]
            // before any launch). The session reads the set back exactly once
            // at the declared completion boundary after the final step; the
            // final step keeps the PerStep subset live past the boundary.
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
            execution_descriptor_hash: descriptor.execution_descriptor_hash(),
            params_initialized: false,
            closed: false,
        })
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
        let result = self.execute_inner(inputs, CopyMode::PerStep, false);
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
    /// over the declared HostProvided `PerProgram` params, without the
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

    /// The shared body of [`ProgramSession::execute_step`] and
    /// [`ProgramSession::execute_final_step`]: `keep_end_of_run` decides
    /// whether the declared end-of-run `PerStep` buffers stay live past the
    /// step boundary (the final step) or recycle like every other `PerStep`
    /// buffer (ordinary steps).
    fn execute_step_impl(&mut self, keep_end_of_run: bool) -> HostResult<DeviceExecutionReceipt> {
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
        if keep_end_of_run && self.final_step_completed {
            return Err(HostError::internal(
                "the final step already completed; a run has exactly one final step (its completion boundary is the declared end-of-run boundary)",
            ));
        }
        if keep_end_of_run && self.end_of_run_read {
            return Err(HostError::internal(
                "the declared end-of-run set was already read back; the end-of-run readback runs exactly once after the final step",
            ));
        }
        let result = self.execute_inner(&BTreeMap::new(), CopyMode::OnceInit, keep_end_of_run);
        if result.is_err() {
            // Release-on-error on all paths (S2-3).
            drop(self.release_all_handles());
            self.closed = true;
        } else if keep_end_of_run {
            self.final_step_completed = true;
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
        if self.closed {
            return Err(HostError::internal(
                "program session is closed after a failed execution; create a new session",
            ));
        }
        if self.program_lifetime != DeviceProgramLifetime::RepeatingStep {
            return Err(HostError::internal(
                "end-of-run readback is a RepeatingStep contract: a SingleRun session has no step loop and no end-of-run boundary",
            ));
        }
        if self.end_of_run_read {
            return Err(HostError::internal(
                "the declared end-of-run set was already read back; the end-of-run readback runs exactly once after the final step",
            ));
        }
        if !self.end_of_run_results.is_empty() && !self.final_step_completed {
            return Err(HostError::internal(
                "the final step has not completed; the declared end-of-run set is observable only at the declared completion boundary after the step loop",
            ));
        }
        let result = self.read_end_of_run_inner();
        if result.is_err() {
            // Release-on-error on all paths (S2-3).
            drop(self.release_all_handles());
            self.closed = true;
        } else {
            self.end_of_run_read = true;
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
        for end_of_run in &self.end_of_run_results {
            let key = (end_of_run.buffer_id, end_of_run.version);
            let meta = self.buffer_meta.get(&key).ok_or_else(|| {
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
        // step boundary and re-allocated for the next execution. The FINAL
        // step (U8/U9) keeps the declared end-of-run PerStep buffers (the
        // final forward activations, the final gradients) live so the
        // one-shot end-of-run readback observes them once at the declared
        // completion boundary; every other step recycles them exactly as
        // before.
        let mut per_step_ids: Vec<BufferKey> = self
            .buffers
            .iter()
            .filter(|(key, _)| {
                self.buffer_meta
                    .get(key)
                    .is_some_and(|meta| meta.lifetime == DeviceBufferLifetime::PerStep)
            })
            .map(|(key, _)| *key)
            .collect();
        if keep_end_of_run {
            per_step_ids.retain(|key| !self.end_of_run_keys.contains(key));
        }
        for key in per_step_ids {
            self.release_buffer(key)?;
            release_count += 1;
        }

        // Declared logical resource graph + data-flow edges (A10) from the
        // session's declared facts (the descriptor projected onto the
        // program).
        let (resource_graph, data_flow_edges) = self.declared_resource_graph();

        // R9 real-synchronization accounting (P1-4): one synchronization per
        // launch — the session launch contract synchronizes internally (Metal
        // waits per launch; CUDA syncs per launch) — plus the explicit
        // step-boundary barrier, which is an actual device synchronization
        // only where the backend's `sync()` performs one. On Metal, `sync()`
        // is a no-op because every launch already called `wait_until_completed`,
        // so the step barrier is NOT an actual synchronization event and is
        // not counted; on CUDA, `sync()` performs `cuCtxSynchronize`, so it
        // is additive. The completion boundary is that barrier after the last
        // dispatched launch.
        let step_boundary_syncs: usize = match self.backend {
            DeviceBackend::Metal => 0,
            DeviceBackend::Cuda => 1,
        };

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
            syncs: launch_count + step_boundary_syncs,
            transfers: copy_ins + readback_count,
            readbacks: readback_count,
            releases: release_count,
            completion_boundary: CompletionBoundary::StepSync {
                after_launch: self.launches.last().map(|launch| launch.id).unwrap_or(0),
            },
            execution_descriptor_hash: self.execution_descriptor_hash,
        })
    }

    /// Allocate this step's `PerStep` and `ObservationPoint` buffers (S2-4);
    /// `PerProgram` buffers are already live from session creation. Buffer ids
    /// already live are left untouched (a `PerProgram` buffer, or a step buffer
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
    /// `PerProgram` ids are live for the program's lifetime; `PerStep` and
    /// `ObservationPoint` ids are live only within one execution (S2-4).
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
    /// no `PerProgram` realloc between executions and full release at teardown.
    /// `PerStep` and `ObservationPoint` buffers are released at the step boundary
    /// / after readback (S2-4), so between executions the session holds the
    /// module + `PerProgram` buffers only. A session closed by an error-path
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

    /// The FNV-1a execution-descriptor hash of the descriptor this session
    /// executes (roots + launches + dependency edges + buffer semantic
    /// identities + observation points + backend entry-name bytes). The
    /// execution identity the session consumed — distinct from
    /// [`ProgramSession::module_hash`], which only names the backend blob.
    /// Backend-entry-inclusive (S5A-U3); not a semantic-identity claim —
    /// the A10 complete-program SHA is the semantic identity.
    #[must_use]
    pub fn execution_descriptor_hash(&self) -> u64 {
        self.execution_descriptor_hash
    }
}
