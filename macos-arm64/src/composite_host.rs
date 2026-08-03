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
    errors as descriptor_errors, fnv1a64, DeviceBufferRole, DeviceDescriptor, E_DEVICE_DESCRIPTOR,
};
use crate::device_host::{DeviceRuntime, DeviceSession};
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
/// module hash, selected hardware).
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

/// Per-buffer metadata captured at session creation for input validation.
struct SessionBufferMeta {
    /// Element count declared by the descriptor (input size check).
    element_count: u64,
}

/// A program-scoped device session that outlives individual launches (S2-1).
///
/// Created from one [`DeviceDescriptor`], a `ProgramSession` owns:
/// - the **module** (loaded once at creation; reused by every execution);
/// - the **PerProgram buffers** (allocated once at creation; reused by every
///   execution — in S2-1 every buffer is PerProgram; S2-4 adds PerStep and
///   ObservationPoint lifetimes).
///
/// [`ProgramSession::execute`] runs the ordered launch sequence once (one
/// step), synchronizes at the step boundary, and reads back declared outputs.
/// It can be called repeatedly on the same session without reloading the
/// module or re-allocating buffers. [`ProgramSession::teardown`] performs
/// the ordered release (buffers then module).
///
/// The S1-4 [`CompositeHost::execute_descriptor`] surface is a single-run
/// convenience over this session (create → execute → teardown).
pub struct ProgramSession<'host> {
    runtime: &'host mut DeviceRuntime,
    backend: DeviceBackend,
    device_name: String,
    module_handle: DeviceHandle,
    module_hash: u64,
    /// PerProgram buffers: buffer_id → device handle. Allocated once at
    /// creation, released at teardown.
    buffers: BTreeMap<u32, DeviceHandle>,
    /// Per-buffer declared element count (input validation in execute).
    buffer_meta: BTreeMap<u32, SessionBufferMeta>,
    /// The ordered launch plan cloned from the descriptor.
    kernels: Vec<SessionKernel>,
}

impl<'host> ProgramSession<'host> {
    /// Create a program session: validate the descriptor, load the module
    /// once, and allocate every distinct buffer once.
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

        // Load the module once (session-scoped ownership; S2-2 formalizes
        // the cache policy around this single load).
        let module_handle = runtime.load_module(&descriptor.module_image)?;
        let module_hash = fnv1a64(&descriptor.module_image);

        // Allocate every distinct buffer once. In S2-1 every buffer is
        // PerProgram; S2-4 will split PerStep / ObservationPoint.
        let mut buffers: BTreeMap<u32, DeviceHandle> = BTreeMap::new();
        let mut buffer_meta: BTreeMap<u32, SessionBufferMeta> = BTreeMap::new();
        let mut kernels: Vec<SessionKernel> = Vec::with_capacity(descriptor.kernels.len());

        for kernel in &descriptor.kernels {
            let mut slots = Vec::with_capacity(kernel.buffers.len());
            for slot in &kernel.buffers {
                buffer_meta
                    .entry(slot.buffer_id)
                    .or_insert(SessionBufferMeta {
                        element_count: slot.element_count,
                    });
                if !buffers.contains_key(&slot.buffer_id) {
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

        Ok(Self {
            runtime,
            backend: descriptor.backend,
            device_name,
            module_handle,
            module_hash,
            buffers,
            buffer_meta,
            kernels,
        })
    }

    /// Execute the ordered launch sequence once (one step). Reuses the
    /// session's module and PerProgram buffers — does not reload the module
    /// or re-allocate buffers. Synchronizes at the step boundary before
    /// reading back declared outputs.
    ///
    /// # Errors
    /// - `E_DEVICE_ENTRY_MISMATCH` — a kernel entry is unknown to the module;
    /// - `E_DEVICE_SHAPE_MISMATCH` — a declared input is missing or its size
    ///   contradicts the declared element count;
    /// - `E_INVALID_ARGS` — a declared output id was not allocated by the
    ///   session;
    /// - session-level failures (copy-in, launch, sync, readback) bubble
    ///   through unchanged.
    pub fn execute(
        &mut self,
        inputs: &BTreeMap<u32, Vec<f32>>,
        outputs: &[u32],
    ) -> HostResult<DeviceExecutionReceipt> {
        let mut launch_count = 0usize;
        let mut copy_ins = 0usize;

        for kernel in &self.kernels {
            // Resolve buffer handles for this kernel's launch (all
            // pre-allocated at session creation).
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

        // Readback declared outputs.
        let mut readbacks: BTreeMap<u32, Vec<f32>> = BTreeMap::new();
        for output_id in outputs {
            let handle = self.buffers.get(output_id).copied().ok_or_else(|| {
                HostError::invalid_args(format!(
                    "declared output buffer id {output_id} was not allocated by the session"
                ))
            })?;
            readbacks.insert(*output_id, self.runtime.readback_f32(&handle)?);
        }

        Ok(DeviceExecutionReceipt {
            backend: self.backend,
            device_name: self.device_name.clone(),
            module_hash: self.module_hash,
            launches: launch_count,
            copy_ins,
            outputs: readbacks,
            allocated_buffers: self.buffers.keys().copied().collect(),
        })
    }

    /// Ordered teardown: release every buffer, then the module. After this
    /// the session is consumed and the device session's handle count returns
    /// to baseline (every allocated handle is released).
    ///
    /// # Errors
    /// Session-level release failures bubble through. On error, remaining
    /// handles may not be released; the error-path guard is S2-3.
    pub fn teardown(self) -> HostResult<()> {
        let ProgramSession {
            runtime,
            buffers,
            module_handle,
            ..
        } = self;
        for handle in buffers.values() {
            runtime.release(handle)?;
        }
        runtime.release(&module_handle)?;
        Ok(())
    }

    /// The program-level buffer ids this session allocated (A9 receipt).
    #[must_use]
    pub fn allocated_buffers(&self) -> Vec<u32> {
        self.buffers.keys().copied().collect()
    }

    /// Number of live device handles the session currently holds (module +
    /// buffers). Used by lifecycle tests to prove no reload/realloc between
    /// executions and full release at teardown.
    #[must_use]
    pub fn session_handle_count(&self) -> usize {
        self.buffers.len() + 1 // buffers + module
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
    /// (allocated once); it survives repeated [`ProgramSession::execute`]
    /// calls on the same session without reloading or re-allocating. Call
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
    pub fn execute_descriptor(
        &mut self,
        descriptor: &DeviceDescriptor,
        inputs: &BTreeMap<u32, Vec<f32>>,
        outputs: &[u32],
    ) -> HostResult<DeviceExecutionReceipt> {
        let mut session = self.create_program_session(descriptor)?;
        let receipt = session.execute(inputs, outputs)?;
        session.teardown()?;
        Ok(receipt)
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
