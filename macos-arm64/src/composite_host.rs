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


mod receipt;
mod session;

pub use receipt::{
    CompletionBoundary, DataFlowEdge, DeviceExecutionReceipt, EndOfRunReadback, ReceiptBuffer,
};
pub use session::ProgramSession;

use std::collections::BTreeMap;

use faber::device::{DeviceBackend, DeviceSelection};

use crate::device_descriptor::{errors as descriptor_errors, DeviceDescriptor};
use crate::device_host::DeviceRuntime;
use crate::kernel::{HostKernel, HostResult};
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
