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
    errors as descriptor_errors, fnv1a64, DeviceBufferLifetime, DeviceBufferRole, DeviceDataType,
    DeviceDescriptor, DeviceProgramLifetime, E_DEVICE_DESCRIPTOR,
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
    /// Host→device copy-ins performed for input slots.
    pub copy_ins: usize,
    /// Declared output readbacks (buffer id → f32 values).
    pub outputs: BTreeMap<u32, Vec<f32>>,
    /// Program-level buffer ids allocated during the run (A9 allocations).
    pub allocated_buffers: Vec<u32>,
    /// Program execution-lifetime regime (S2-4).
    pub program_lifetime: DeviceProgramLifetime,
    /// PerProgram buffer ids: allocated once per session, released at program
    /// end (persist across executions).
    pub per_program_buffers: Vec<u32>,
    /// PerStep buffer ids: recycled at each step boundary.
    pub per_step_buffers: Vec<u32>,
    /// ObservationPoint buffer ids: read back and released per execution
    /// (read-then-release).
    pub observation_buffers: Vec<u32>,
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
    /// Observed step-boundary synchronizations this execution.
    pub syncs: usize,
    /// Observed transfers this execution (host→device copy-ins plus
    /// device→host readbacks).
    pub transfers: usize,
    /// Observed buffer releases this execution (read-then-release plus the
    /// step-boundary PerStep recycle).
    pub releases: usize,
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
    /// Logical name for diagnostics.
    buffer_name: String,
    /// Slot role at this kernel.
    role: DeviceBufferRole,
}

/// Per-buffer metadata captured at session creation for input validation,
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
    /// Content version carried by the wire (resource-graph fact; R2 — the
    /// host consumes the version, never hardcodes `1`).
    version: u32,
    /// Byte length this buffer's storage needs on the device.
    byte_length: u64,
    /// Lifetime class that drives the session's allocation/release policy.
    lifetime: DeviceBufferLifetime,
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
/// declared program fact: `SingleRun` is the Stage 2 fixture regime, where
/// repeated `execute()` calls are a one-shot-with-repeat surface for the
/// leak proof (each execution re-runs the whole program; per-step recycling
/// between executions is not yet meaningful). `RepeatingStep` — where
/// per-step buffers recycle as a training-iteration pool — is recorded and
/// reported but its Stage 5 training-loop semantics are out of Stage 2 scope.
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
    /// Currently-live device buffers: buffer_id → device handle. PerProgram
    /// buffers are live from creation until teardown; PerStep and
    /// ObservationPoint buffers are live only within one execution.
    buffers: BTreeMap<u32, DeviceHandle>,
    /// Per-buffer declared element count / byte length / lifetime class.
    buffer_meta: BTreeMap<u32, SessionBufferMeta>,
    /// The ordered launch plan cloned from the descriptor.
    kernels: Vec<SessionKernel>,
    /// Carried inter-kernel data-flow edges (A10/R2): the wire's
    /// producer/consumer facts per buffer version, consumed by
    /// [`ProgramSession::declared_resource_graph`] — never re-derived from
    /// launch order.
    data_flow: Vec<DataFlowEdge>,
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
        let mut buffers: BTreeMap<u32, DeviceHandle> = BTreeMap::new();
        let mut buffer_meta: BTreeMap<u32, SessionBufferMeta> = BTreeMap::new();
        let mut kernels: Vec<SessionKernel> = Vec::with_capacity(descriptor.kernels.len());

        let result = (|| {
            for kernel in &descriptor.kernels {
                let mut slots = Vec::with_capacity(kernel.buffers.len());
                for slot in &kernel.buffers {
                    buffer_meta
                        .entry(slot.buffer_id)
                        .or_insert(SessionBufferMeta {
                            name: slot.buffer_name.clone(),
                            role: slot.role,
                            element_ty: slot.element_ty,
                            element_count: slot.element_count,
                            version: slot.version,
                            byte_length: slot.byte_length(),
                            lifetime: slot.lifetime,
                        });
                    if !buffers.contains_key(&slot.buffer_id)
                        && slot.lifetime == DeviceBufferLifetime::PerProgram
                    {
                        let handle = runtime.alloc_bytes(slot.byte_length() as usize)?;
                        buffers.insert(slot.buffer_id, handle);
                    }
                    slots.push(SessionSlot {
                        buffer_id: slot.buffer_id,
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
            closed: false,
        })
    }

    /// Execute the ordered launch sequence once (one step). Reuses the
    /// session's module and PerProgram buffers — does not reload the module
    /// or re-allocate PerProgram buffers. PerStep buffers are allocated for
    /// the step and recycled at the step boundary; ObservationPoint buffers
    /// are allocated, read back at the declared observation point, and
    /// released (read-then-release — the only readback the session performs,
    /// S2-4). Synchronizes at the step boundary before reading back.
    ///
    /// **Error-path teardown is designed into this method (S2-3):** a failure
    /// at any stage runs the ordered release (buffers then module) before the
    /// error escapes and closes the session, so a failed execution leaves
    /// `live_handle_count() == 0` and no handle survives. A closed session
    /// refuses further [`ProgramSession::execute`] calls.
    ///
    /// # Errors
    /// - `E_DEVICE_ENTRY_MISMATCH` — a kernel entry is unknown to the module;
    /// - `E_DEVICE_SHAPE_MISMATCH` — a declared input is missing or its size
    ///   contradicts the declared element count;
    /// - `E_INVALID_ARGS` — a declared output id was not allocated by the
    ///   session, or the id names a buffer whose lifetime is not
    ///   ObservationPoint (an undeclared readback fails closed, S2-4);
    /// - session-level failures (copy-in, launch, sync, readback) bubble
    ///   through unchanged.
    pub fn execute(
        &mut self,
        inputs: &BTreeMap<u32, Vec<f32>>,
        outputs: &[u32],
    ) -> HostResult<DeviceExecutionReceipt> {
        if self.closed {
            return Err(HostError::internal(
                "program session is closed after a failed execution; create a new session",
            ));
        }
        let result = self.execute_inner(inputs, outputs);
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

    /// The executable body of [`ProgramSession::execute`]: the ordered launch
    /// sequence (step-buffer allocation → copy-in → launch → step-boundary
    /// sync → observation readback + release → per-step release) without the
    /// error-path release, which the caller owns.
    fn execute_inner(
        &mut self,
        inputs: &BTreeMap<u32, Vec<f32>>,
        outputs: &[u32],
    ) -> HostResult<DeviceExecutionReceipt> {
        let mut launch_count = 0usize;
        let mut copy_ins = 0usize;

        // Allocate this step's PerStep + ObservationPoint buffers (S2-4).
        // PerProgram buffers were allocated once at session creation and
        // stay live. A failure here runs the error-path teardown (S2-3).
        self.allocate_step_buffers()?;

        for kernel in &self.kernels {
            // Resolve buffer handles for this kernel's launch (PerProgram
            // live from creation; PerStep/ObservationPoint just allocated).
            let mut launch_buffers: Vec<DeviceHandle> = Vec::with_capacity(kernel.slots.len());
            for slot in &kernel.slots {
                let handle = self
                    .buffers
                    .get(&slot.buffer_id)
                    .copied()
                    .ok_or_else(|| HostError::internal("session buffer disappeared during launch"))?;
                launch_buffers.push(handle);
            }

            // Copy-in declared inputs for this kernel.
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
                        .get(&slot.buffer_id)
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
                        .get(&slot.buffer_id)
                        .copied()
                        .ok_or_else(|| HostError::internal("session input buffer disappeared"))?;
                    self.runtime.copy_in_f32(&handle, values)?;
                    copy_ins += 1;
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
        }

        // Step-boundary synchronization: every launch in this step has
        // completed before any readback. The launches also sync internally;
        // this barrier makes the step boundary explicit and observable.
        self.runtime.sync()?;

        // ObservationPoint read-then-release (S2-4): read back the declared
        // observation buffers, then release each immediately. Only
        // ObservationPoint buffers are readable — a readback request for any
        // other lifetime class is an undeclared readback and fails closed.
        let mut release_count = 0usize;
        let mut readbacks: BTreeMap<u32, Vec<f32>> = BTreeMap::new();
        for output_id in outputs {
            if readbacks.contains_key(output_id) {
                continue;
            }
            let meta = self.buffer_meta.get(output_id).ok_or_else(|| {
                HostError::invalid_args(format!(
                    "declared output buffer id {output_id} was not allocated by the session"
                ))
            })?;
            if meta.lifetime != DeviceBufferLifetime::ObservationPoint {
                return Err(HostError::invalid_args(format!(
                    "declared output buffer id {output_id} has lifetime `{}`; only observation-point buffers are read back (no undeclared readback)",
                    meta.lifetime.spelling()
                )));
            }
            let handle = self
                .buffers
                .get(output_id)
                .copied()
                .ok_or_else(|| HostError::internal("session observation buffer disappeared"))?;
            let values = self.runtime.readback_f32(&handle)?;
            readbacks.insert(*output_id, values);
            self.release_buffer(*output_id)?;
            release_count += 1;
        }

        // Step-boundary recycle (S2-4): PerStep buffers are released at the
        // step boundary and re-allocated for the next execution.
        let per_step_ids: Vec<u32> = self
            .buffers
            .iter()
            .filter(|(id, _)| {
                self.buffer_meta
                    .get(id)
                    .is_some_and(|meta| meta.lifetime == DeviceBufferLifetime::PerStep)
            })
            .map(|(id, _)| *id)
            .collect();
        for id in per_step_ids {
            self.release_buffer(id)?;
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
            copy_ins,
            outputs: readbacks,
            allocated_buffers: self.allocated_buffers(),
            program_lifetime: self.program_lifetime,
            per_program_buffers: self.buffers_by_lifetime(DeviceBufferLifetime::PerProgram),
            per_step_buffers: self.buffers_by_lifetime(DeviceBufferLifetime::PerStep),
            observation_buffers: self.buffers_by_lifetime(DeviceBufferLifetime::ObservationPoint),
            resource_graph,
            data_flow_edges,
            syncs: 1,
            transfers: copy_ins + outputs.len(),
            releases: release_count,
        })
    }

    /// Allocate this step's PerStep and ObservationPoint buffers (S2-4);
    /// PerProgram buffers are already live from session creation. Buffer ids
    /// already live are left untouched (a PerProgram buffer, or a step buffer
    /// left live by an interrupted path that has not yet run the error path,
    /// is never double-allocated).
    fn allocate_step_buffers(&mut self) -> HostResult<()> {
        let to_allocate: Vec<u32> = self
            .buffer_meta
            .iter()
            .filter(|(id, meta)| {
                meta.lifetime != DeviceBufferLifetime::PerProgram
                    && !self.buffers.contains_key(id)
            })
            .map(|(id, _)| *id)
            .collect();
        for id in to_allocate {
            let meta = self
                .buffer_meta
                .get(&id)
                .ok_or_else(|| HostError::internal("session buffer metadata disappeared"))?;
            let handle = self.runtime.alloc_bytes(meta.byte_length as usize)?;
            self.buffers.insert(id, handle);
        }
        Ok(())
    }

    /// Release one live buffer by id (no-op when the id is not live). Used by
    /// the read-then-release and step-boundary paths.
    fn release_buffer(&mut self, id: u32) -> HostResult<()> {
        if let Some(handle) = self.buffers.remove(&id) {
            self.runtime.release(&handle)?;
        }
        Ok(())
    }

    /// The program's buffer ids classified by lifetime class (S2-4 receipt).
    fn buffers_by_lifetime(&self, lifetime: DeviceBufferLifetime) -> Vec<u32> {
        self.buffer_meta
            .iter()
            .filter(|(_, meta)| meta.lifetime == lifetime)
            .map(|(id, _)| *id)
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
            .map(|(id, meta)| ReceiptBuffer {
                id: *id,
                name: meta.name.clone(),
                role: meta.role,
                lifetime: meta.lifetime,
                element_ty: meta.element_ty,
                element_count: meta.element_count,
                version: meta.version,
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
    /// repeated [`ProgramSession::execute`] calls on the same session without
    /// reloading or re-allocating PerProgram buffers. Call
    /// [`ProgramSession::teardown`] to release every handle in order.
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
        outputs: &[u32],
    ) -> HostResult<DeviceExecutionReceipt> {
        let mut session = self.create_program_session(descriptor)?;
        match session.execute(inputs, outputs) {
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
