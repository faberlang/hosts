//! Metal product-host lifecycle scaffolding for MIR v1 (lane M).
//!
//! Path A (this checkout): proof-environment admission and fail-closed
//! discovery, mirroring [`cuda_host`] naming 1:1. M1 delivered the injectable
//! driver seam plus a sequencing fake; M2 adds `SystemMetalDriver`, the real
//! gfx-rs `metal` binding that compiles MSL at runtime and launches on the
//! local Apple GPU. M4 closes the C5 API parity gap: generalized
//! `launch_kernel` (trait + session), session-level `sync()`, and session-side
//! `try_open`. No product run is claimed without a loadable device stack.

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use faber::Valor;
#[cfg(target_os = "macos")]
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};
use serde::{Deserialize, Serialize};

use crate::device_registry::{DriverCounters, FakeFailureStage, HandleRegistry};
use crate::kernel::frame_data;
use crate::kernel::{HostError, HostResult};

/// Stable host error code for missing Metal driver/device/toolchain.
pub const E_METAL_UNAVAILABLE: &str = "E_METAL_UNAVAILABLE";
/// Stable host error for unsupported Metal host operations / product claims.
pub const E_METAL_UNSUPPORTED: &str = "E_METAL_UNSUPPORTED";
/// Stale or unknown opaque handle.
pub const E_METAL_INVALID_HANDLE: &str = "E_METAL_INVALID_HANDLE";
/// Driver-level failure after admission.
pub const E_METAL_DRIVER: &str = "E_METAL_DRIVER";

/// Default kernel entry for the legacy `launch_elementwise_add_f32` session
/// path. Matches the emitted `add_one` entry of the U2 proof fixture
/// (input@0, output@1, extent@2).
const ELEMENTWISE_ADD_ENTRY: &[u8] = b"add_one";

/// Read-only environment admission report (never a product run claim).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetalEnvReport {
    pub admitted: bool,
    /// Present when the system default Metal device is detected. M1 has no
    /// binding, so this carries a fixed marker string; M2 fills the real
    /// device name from the binding.
    pub mtl_device: Option<String>,
    /// Metal framework paths found on this machine. Informational only —
    /// admission is driven by the device probe, not by framework presence.
    pub metal_framework_paths: Vec<String>,
    pub reason: String,
}

/// Opaque host-owned handle identity carried at the Frame control boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MetalHandleId(pub u64);

#[derive(Clone, Debug, PartialEq, Eq)]
enum MetalHandleKind {
    Module,
    Buffer { len_bytes: usize },
}

/// Injectable driver boundary (real Metal binding adapter or sequencing fake).
pub trait MetalDriver: Send {
    fn discover(&mut self) -> HostResult<MetalEnvReport>;
    fn create_context(&mut self) -> HostResult<()>;
    fn load_module(&mut self, image: &[u8]) -> HostResult<u64>;
    fn alloc(&mut self, len_bytes: usize) -> HostResult<u64>;
    fn copy_in(&mut self, token: u64, bytes: &[u8]) -> HostResult<()>;
    fn launch_elementwise_add_f32(
        &mut self,
        module: u64,
        a: u64,
        b: u64,
        out: u64,
        len: usize,
    ) -> HostResult<()>;
    /// Generalized kernel launch: resolve `entry` inside `module` and launch
    /// over the given device buffers (binding order: inputs first, output
    /// last) with the given 3D grid and 3D block shape. The session
    /// synchronizes after launching; the system driver routes the legacy
    /// elementwise-add path through this so there is exactly one
    /// encoder/commit site.
    fn launch_kernel(
        &mut self,
        module: u64,
        entry: &[u8],
        buffers: &[u64],
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
    ) -> HostResult<()>;
    fn sync(&mut self) -> HostResult<()>;
    fn copy_out(&mut self, token: u64, len_bytes: usize) -> HostResult<Vec<u8>>;
    fn free(&mut self, token: u64) -> HostResult<()>;
    /// Driver-level lifecycle counters (S2-2 module-cache leak bar). Real
    /// drivers report all-zero (their leak evidence is the S2-8 real-device
    /// gate); the fake drivers track cumulative loads/releases so tests can
    /// prove the cache policy at the driver boundary.
    fn counters(&self) -> DriverCounters {
        DriverCounters::default()
    }
}

/// Product-facing session: opaque handles + ordered lifecycle.
pub struct MetalHostSession {
    driver: Box<dyn MetalDriver>,
    handles: HandleRegistry<MetalHandleKind>,
    admitted: bool,
}

impl MetalHostSession {
    /// Open a session against the live Metal stack. Fails closed when the
    /// machine cannot admit a Metal product stack.
    pub fn try_open() -> HostResult<Self> {
        #[cfg(target_os = "macos")]
        {
            let mut driver = Box::new(SystemMetalDriver::default());
            let report = driver.discover()?;
            if !report.admitted {
                return Err(metal_unavailable(report.reason));
            }
            let mut session = Self {
                driver,
                handles: HandleRegistry::new(),
                admitted: true,
            };
            session.driver.create_context()?;
            Ok(session)
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(metal_unavailable(
                "Metal is not available on this platform (not macOS)",
            ))
        }
    }

    /// Inject a driver for unit tests (sequencing / reject paths only).
    pub fn with_driver(mut driver: Box<dyn MetalDriver>) -> HostResult<Self> {
        let report = driver.discover()?;
        let admitted = report.admitted;
        if admitted {
            driver.create_context()?;
        }
        Ok(Self {
            driver,
            handles: HandleRegistry::new(),
            admitted,
        })
    }

    pub fn is_admitted(&self) -> bool {
        self.admitted
    }

    /// Number of live opaque handles (module + buffer registrations). Used by
    /// lifecycle tests to prove teardown released every handle.
    #[must_use]
    pub fn live_handle_count(&self) -> usize {
        self.handles.len()
    }

    /// Driver-level lifecycle counters (S2-2 module-cache leak bar). The
    /// fake drivers track cumulative module loads/releases and buffer
    /// allocs/releases so session tests prove the policy at the driver
    /// boundary; the real drivers report all-zero (S2-8 real-device gate).
    #[must_use]
    pub fn driver_counters(&self) -> DriverCounters {
        self.driver.counters()
    }

    pub fn load_module(&mut self, image: &[u8]) -> HostResult<MetalHandleId> {
        self.require_admitted()?;
        if image.is_empty() {
            return Err(HostError::invalid_args("Metal module image is empty"));
        }
        let token = self.driver.load_module(image)?;
        Ok(self.insert(MetalHandleKind::Module, token))
    }

    pub fn alloc_bytes(&mut self, len_bytes: usize) -> HostResult<MetalHandleId> {
        self.require_admitted()?;
        if len_bytes == 0 {
            return Err(HostError::invalid_args(
                "Metal buffer length must be non-zero",
            ));
        }
        let token = self.driver.alloc(len_bytes)?;
        Ok(self.insert(MetalHandleKind::Buffer { len_bytes }, token))
    }

    pub fn copy_in_f32(&mut self, buffer: MetalHandleId, values: &[f32]) -> HostResult<()> {
        self.require_admitted()?;
        let (token, len_bytes) = self.buffer_token(buffer)?;
        let bytes = f32_slice_as_bytes(values);
        if bytes.len() != len_bytes {
            return Err(HostError::invalid_args(format!(
                "copy_in size mismatch: buffer {len_bytes} bytes, got {}",
                bytes.len()
            )));
        }
        self.driver.copy_in(token, bytes)
    }

    pub fn launch_elementwise_add_f32(
        &mut self,
        module: MetalHandleId,
        a: MetalHandleId,
        b: MetalHandleId,
        out: MetalHandleId,
    ) -> HostResult<()> {
        self.require_admitted()?;
        let module_token = self.module_token(module)?;
        let (a_token, a_len) = self.buffer_token(a)?;
        let (b_token, b_len) = self.buffer_token(b)?;
        let (out_token, out_len) = self.buffer_token(out)?;
        if a_len != b_len || a_len != out_len {
            return Err(HostError::invalid_args(
                "elementwise add requires equal buffer sizes",
            ));
        }
        if a_len % 4 != 0 {
            return Err(HostError::invalid_args(
                "elementwise add buffers must be multiples of 4 bytes (f32)",
            ));
        }
        let len = a_len / 4;
        self.driver
            .launch_elementwise_add_f32(module_token, a_token, b_token, out_token, len)?;
        self.driver.sync()
    }

    /// Generalized launch: resolve `entry` inside `module` and dispatch over
    /// `buffers` (inputs first, output last) with the given grid/block shape.
    /// Every buffer handle is validated and resolved to a backend token before
    /// the driver is touched; the launch synchronizes internally. This helper
    /// preserves the original 1D session surface for elementwise callers.
    pub fn launch_kernel(
        &mut self,
        module: MetalHandleId,
        entry: &str,
        buffers: &[MetalHandleId],
        grid_x: u32,
        block_x: u32,
    ) -> HostResult<()> {
        self.launch_kernel_3d(module, entry, buffers, grid_x, 1, 1, block_x, 1, 1)
    }

    /// Generalized launch with explicit 3D grid and block shape. Matmul and
    /// other collection kernels use y/z dimensions; elementwise callers can use
    /// `launch_kernel`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_kernel_3d(
        &mut self,
        module: MetalHandleId,
        entry: &str,
        buffers: &[MetalHandleId],
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
    ) -> HostResult<()> {
        self.require_admitted()?;
        let module_token = self.module_token(module)?;
        if entry.is_empty() {
            return Err(HostError::invalid_args("Metal kernel entry name is empty"));
        }
        let mut tokens = Vec::with_capacity(buffers.len());
        for id in buffers {
            let (token, _len_bytes) = self.buffer_token(*id)?;
            tokens.push(token);
        }
        self.driver.launch_kernel(
            module_token,
            entry.as_bytes(),
            &tokens,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
        )?;
        self.driver.sync()
    }

    /// Explicit device synchronization barrier. The launch paths already sync
    /// internally; this exposes the barrier for callers that need it directly.
    pub fn sync(&mut self) -> HostResult<()> {
        self.require_admitted()?;
        self.driver.sync()
    }

    pub fn readback_f32(&mut self, buffer: MetalHandleId) -> HostResult<Vec<f32>> {
        self.require_admitted()?;
        let (token, len_bytes) = self.buffer_token(buffer)?;
        let bytes = self.driver.copy_out(token, len_bytes)?;
        if bytes.len() != len_bytes || len_bytes % 4 != 0 {
            return Err(HostError::internal(
                "Metal readback returned unexpected byte length",
            ));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }

    pub fn release(&mut self, id: MetalHandleId) -> HostResult<()> {
        let Some(handle) = self.handles.remove(id.0) else {
            return Err(metal_invalid_handle(id));
        };
        self.driver.free(handle.backend_token)
    }

    /// Control-frame representation of a handle (opaque id only; no payload).
    pub fn handle_frame_data(id: MetalHandleId) -> Valor {
        frame_data::tabula([("metal_handle", Valor::Numerus(id.0 as i64))])
    }

    fn require_admitted(&self) -> HostResult<()> {
        if self.admitted {
            Ok(())
        } else {
            Err(metal_unavailable(
                "Metal host session is not admitted for product execution",
            ))
        }
    }

    fn insert(&mut self, kind: MetalHandleKind, backend_token: u64) -> MetalHandleId {
        MetalHandleId(self.handles.insert(kind, backend_token))
    }

    fn module_token(&self, id: MetalHandleId) -> HostResult<u64> {
        match self.handles.get(id.0) {
            Some(handle) if matches!(handle.kind, MetalHandleKind::Module) => {
                Ok(handle.backend_token)
            }
            Some(_) => Err(HostError::invalid_args("handle is not a Metal module")),
            None => Err(metal_invalid_handle(id)),
        }
    }

    fn buffer_token(&self, id: MetalHandleId) -> HostResult<(u64, usize)> {
        match self.handles.get(id.0) {
            Some(handle) => match &handle.kind {
                MetalHandleKind::Buffer { len_bytes } => Ok((handle.backend_token, *len_bytes)),
                _ => Err(HostError::invalid_args("handle is not a Metal buffer")),
            },
            None => Err(metal_invalid_handle(id)),
        }
    }
}

fn f32_slice_as_bytes(values: &[f32]) -> &[u8] {
    // Safe: f32 is plain bits; host owns the slice for the duration of copy_in.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn metal_unavailable(message: impl Into<String>) -> HostError {
    HostError {
        code: E_METAL_UNAVAILABLE.to_owned(),
        message: message.into(),
        retryable: false,
    }
}

fn metal_invalid_handle(id: MetalHandleId) -> HostError {
    HostError {
        code: E_METAL_INVALID_HANDLE.to_owned(),
        message: format!("unknown or released Metal handle {}", id.0),
        retryable: false,
    }
}

// System default Metal device probe. The `metal` crate (gfx-rs) manages the
// `MTLCreateSystemDefaultDevice` framework link; the probe fills the real
// device name from the binding (M2).

/// Probe this machine for a loadable Metal device stack without claiming a run.
pub fn probe_metal_environment() -> MetalEnvReport {
    let metal_framework_paths = detect_metal_framework_paths();

    #[cfg(target_os = "macos")]
    let mtl_device = Device::system_default().map(|device| device.name().to_owned());
    #[cfg(not(target_os = "macos"))]
    let mtl_device: Option<String> = None;

    let admitted = mtl_device.is_some();
    let reason = if admitted {
        "Metal default device present; SystemMetalDriver can compile MSL and launch add_one"
            .to_owned()
    } else {
        "no Metal default device detected (MTLCreateSystemDefaultDevice returned null, or this OS is not macOS); Metal product execution is not admitted".to_owned()
    };
    MetalEnvReport {
        admitted,
        mtl_device,
        metal_framework_paths,
        reason,
    }
}

/// Best-effort scan for the Metal framework bundle without walking the world.
fn detect_metal_framework_paths() -> Vec<String> {
    let mut candidates = Vec::new();
    for path in ["/System/Library/Frameworks/Metal.framework"] {
        if Path::new(path).is_dir() {
            candidates.push(path.to_owned());
        }
    }
    candidates
}

/// Sequencing-only fake driver for unit tests. Not product Metal evidence.
#[derive(Default)]
pub struct FakeMetalDriver {
    next_token: u64,
    buffers: BTreeMap<u64, Vec<u8>>,
    modules: BTreeMap<u64, Vec<u8>>,
    force_unavailable: bool,
    /// Entry names the loaded module's function table declares. Empty means
    /// the fake does not enforce entry checks (legacy sequencing behavior);
    /// non-empty means an unknown launch entry fails closed with
    /// `E_DEVICE_ENTRY_MISMATCH`, mirroring the compiled-pipeline entry check
    /// on the real lane.
    known_entries: Vec<String>,
    /// Cumulative module loads (S2-2 module-cache leak bar).
    module_loads: usize,
    /// Cumulative module releases.
    module_releases: usize,
    /// Cumulative buffer allocations.
    buffer_allocs: usize,
    /// Cumulative buffer releases.
    buffer_releases: usize,
    /// Per-stage failure injection (S2-3): stage → 1-based call number whose
    /// invocation fails with a typed `E_METAL_DRIVER` error. An absent stage
    /// never fails.
    fail_at: BTreeMap<FakeFailureStage, u32>,
    /// Running call count per stage (drives `fail_at`).
    stage_calls: BTreeMap<FakeFailureStage, u32>,
}

impl FakeMetalDriver {
    pub fn unavailable() -> Self {
        Self {
            force_unavailable: true,
            ..Self::default()
        }
    }

    /// Declare a module entry for launch-time entry validation.
    pub fn with_known_entry(mut self, entry: impl Into<String>) -> Self {
        self.known_entries.push(entry.into());
        self
    }

    /// Configure the driver to fail the `call`-th invocation of `stage` with
    /// a typed `E_METAL_DRIVER` error (S2-3 failure-injection tests). `call`
    /// is 1-based; an absent entry means the stage never fails.
    pub fn with_failure_at(mut self, stage: FakeFailureStage, call: u32) -> Self {
        self.fail_at.insert(stage, call);
        self
    }

    /// Inject a typed driver failure when the configured call number of
    /// `stage` is reached; otherwise record the call and continue. Called at
    /// the top of every stage method so an injected failure fires before any
    /// state mutation at that stage.
    fn maybe_fail(&mut self, stage: FakeFailureStage) -> HostResult<()> {
        let call = self.stage_calls.entry(stage).or_insert(0);
        *call += 1;
        if self.fail_at.get(&stage) == Some(call) {
            return Err(metal_driver(format!(
                "injected failure at {stage:?} stage (S2-3 failure-injection test)"
            )));
        }
        Ok(())
    }

    /// Simulate the elementwise-add kernel: `out[i] = a[i] + b[i]`. Shared by
    /// the legacy elementwise-add path and the generalized `launch_kernel`
    /// (the emitted kernel is the same add shape), mirroring the CUDA fake.
    fn simulate_elementwise_add(
        &mut self,
        module: u64,
        a: u64,
        b: u64,
        out: u64,
    ) -> HostResult<()> {
        if !self.modules.contains_key(&module) {
            return Err(HostError::internal("fake launch missing module"));
        }
        let a_bytes = self
            .buffers
            .get(&a)
            .ok_or_else(|| HostError::internal("fake launch missing a"))?
            .clone();
        let b_bytes = self
            .buffers
            .get(&b)
            .ok_or_else(|| HostError::internal("fake launch missing b"))?
            .clone();
        let out_buf = self
            .buffers
            .get_mut(&out)
            .ok_or_else(|| HostError::internal("fake launch missing out"))?;
        if a_bytes.len() != b_bytes.len() || a_bytes.len() != out_buf.len() {
            return Err(HostError::invalid_args("fake launch length mismatch"));
        }
        let len = a_bytes.len() / 4;
        for i in 0..len {
            let ai = f32::from_le_bytes([
                a_bytes[i * 4],
                a_bytes[i * 4 + 1],
                a_bytes[i * 4 + 2],
                a_bytes[i * 4 + 3],
            ]);
            let bi = f32::from_le_bytes([
                b_bytes[i * 4],
                b_bytes[i * 4 + 1],
                b_bytes[i * 4 + 2],
                b_bytes[i * 4 + 3],
            ]);
            let sum = (ai + bi).to_le_bytes();
            out_buf[i * 4..i * 4 + 4].copy_from_slice(&sum);
        }
        Ok(())
    }
}

impl MetalDriver for FakeMetalDriver {
    fn discover(&mut self) -> HostResult<MetalEnvReport> {
        if self.force_unavailable {
            return Err(metal_unavailable(
                "fake driver configured unavailable (sequencing test)",
            ));
        }
        Ok(MetalEnvReport {
            admitted: true,
            mtl_device: Some("fake-metal-device".to_owned()),
            metal_framework_paths: vec!["fake://Metal.framework".to_owned()],
            reason: "fake driver admitted for sequencing tests only".to_owned(),
        })
    }

    fn create_context(&mut self) -> HostResult<()> {
        Ok(())
    }

    fn load_module(&mut self, image: &[u8]) -> HostResult<u64> {
        self.maybe_fail(FakeFailureStage::ModuleLoad)?;
        self.module_loads += 1;
        let token = self.next_token;
        self.next_token += 1;
        self.modules.insert(token, image.to_vec());
        Ok(token)
    }

    fn alloc(&mut self, len_bytes: usize) -> HostResult<u64> {
        self.maybe_fail(FakeFailureStage::Alloc)?;
        self.buffer_allocs += 1;
        let token = self.next_token;
        self.next_token += 1;
        self.buffers.insert(token, vec![0; len_bytes]);
        Ok(token)
    }

    fn copy_in(&mut self, token: u64, bytes: &[u8]) -> HostResult<()> {
        self.maybe_fail(FakeFailureStage::CopyIn)?;
        let buffer = self
            .buffers
            .get_mut(&token)
            .ok_or_else(|| HostError::internal("fake copy_in missing buffer"))?;
        if buffer.len() != bytes.len() {
            return Err(HostError::invalid_args("fake copy_in size mismatch"));
        }
        buffer.copy_from_slice(bytes);
        Ok(())
    }

    fn launch_elementwise_add_f32(
        &mut self,
        module: u64,
        a: u64,
        b: u64,
        out: u64,
        _len: usize,
    ) -> HostResult<()> {
        // `_len` mirrors the session's element count; the shared simulation
        // derives it from the (session-validated, equal) buffer sizes.
        self.simulate_elementwise_add(module, a, b, out)
    }

    fn launch_kernel(
        &mut self,
        module: u64,
        entry: &[u8],
        buffers: &[u64],
        _grid_x: u32,
        _grid_y: u32,
        _grid_z: u32,
        _block_x: u32,
        _block_y: u32,
        _block_z: u32,
    ) -> HostResult<()> {
        self.maybe_fail(FakeFailureStage::Launch)?;
        // When the harness declares the module's function table, an unknown
        // entry fails closed before dispatch (mirrors the compiled-pipeline
        // entry lookup on the real lane).
        if !self.known_entries.is_empty()
            && !self
                .known_entries
                .iter()
                .any(|entry_name| entry_name.as_bytes() == entry)
        {
            return Err(HostError {
                code: crate::device_descriptor::E_DEVICE_ENTRY_MISMATCH.to_owned(),
                message: format!(
                    "module has no entry named {}",
                    String::from_utf8_lossy(entry)
                ),
                retryable: false,
            });
        }
        // The simulated elementwise-add kernel takes exactly three buffers
        // (a, b, out). Anything else fails closed in the fake just as it
        // would on device.
        if buffers.len() != 3 {
            return Err(HostError::invalid_args(
                "fake launch_kernel simulates the 3-buffer elementwise-add kernel (a, b, out)",
            ));
        }
        self.simulate_elementwise_add(module, buffers[0], buffers[1], buffers[2])
    }

    fn sync(&mut self) -> HostResult<()> {
        self.maybe_fail(FakeFailureStage::Sync)?;
        Ok(())
    }

    fn copy_out(&mut self, token: u64, len_bytes: usize) -> HostResult<Vec<u8>> {
        self.maybe_fail(FakeFailureStage::Readback)?;
        let buffer = self
            .buffers
            .get(&token)
            .ok_or_else(|| HostError::internal("fake copy_out missing buffer"))?;
        if buffer.len() != len_bytes {
            return Err(HostError::internal("fake copy_out size mismatch"));
        }
        Ok(buffer.clone())
    }

    fn free(&mut self, token: u64) -> HostResult<()> {
        if self.buffers.remove(&token).is_some() {
            self.buffer_releases += 1;
        } else if self.modules.remove(&token).is_some() {
            self.module_releases += 1;
        }
        Ok(())
    }

    fn counters(&self) -> DriverCounters {
        DriverCounters {
            module_loads: self.module_loads,
            module_releases: self.module_releases,
            buffer_allocs: self.buffer_allocs,
            buffer_releases: self.buffer_releases,
        }
    }
}

/// Live Metal driver: real gfx-rs `metal` binding (M2).
///
/// Owns the system default Metal device, a command queue, compiled compute
/// pipelines (MSL compiled at runtime via `new_library_with_source`), and
/// `StorageModeShared` buffers. Mirrors `SystemCudaDriver`'s method shape but
/// actually executes on the Apple GPU; MSL compile/pipeline/launch failures
/// map to `E_METAL_*` codes. Private like `SystemCudaDriver`: reachable only
/// through [`MetalHostSession::try_open`] so the session API surface is the
/// symmetric parity entry point.
#[cfg(target_os = "macos")]
#[derive(Default)]
struct SystemMetalDriver {
    device: Option<Device>,
    queue: Option<CommandQueue>,
    modules: BTreeMap<u64, MetalModule>,
    buffers: BTreeMap<u64, Buffer>,
    next_token: u64,
}

/// A compiled Metal compute module: a compute pipeline per declared `kernel
/// void` entry. A program module can declare several kernels (S2-5: the
/// ordered launch sequence dispatches multiple kernels from one module), so
/// each entry carries its own pipeline and a generalized launch can fail
/// closed on an unknown entry name (mirroring `cuModuleGetFunction` on the
/// CUDA lane).
#[cfg(target_os = "macos")]
struct MetalModule {
    pipelines: BTreeMap<String, ComputePipelineState>,
}

#[cfg(target_os = "macos")]
impl MetalDriver for SystemMetalDriver {
    fn discover(&mut self) -> HostResult<MetalEnvReport> {
        match Device::system_default() {
            Some(device) => {
                let name = device.name().to_owned();
                self.device = Some(device);
                Ok(MetalEnvReport {
                    admitted: true,
                    mtl_device: Some(name),
                    metal_framework_paths: detect_metal_framework_paths(),
                    reason: "system default Metal device present; SystemMetalDriver admits"
                        .to_owned(),
                })
            }
            None => Err(metal_unavailable(
                "no Metal default device (MTLCreateSystemDefaultDevice returned null)",
            )),
        }
    }

    fn create_context(&mut self) -> HostResult<()> {
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| metal_unavailable("SystemMetalDriver has no device"))?;
        self.queue = Some(device.new_command_queue());
        Ok(())
    }

    fn load_module(&mut self, image: &[u8]) -> HostResult<u64> {
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| metal_unavailable("SystemMetalDriver has no device"))?;
        let source = std::str::from_utf8(image)
            .map_err(|_| HostError::invalid_args("Metal module image is not UTF-8 MSL source"))?;
        // A module declares one `kernel void` entry per kernel (S2-5
        // multi-kernel modules); every declared entry gets a compute pipeline
        // at load time so each launch resolves its own pipeline and an
        // unknown entry fails closed with E_DEVICE_ENTRY_MISMATCH.
        let entries = msl_kernel_entry_names(source).ok_or_else(|| {
            HostError::invalid_args("Metal MSL source has no `kernel void` entry function")
        })?;
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(source, &options)
            .map_err(|message| metal_driver(format!("MSL compile failed: {message}")))?;
        let mut pipelines = BTreeMap::new();
        for entry in &entries {
            let function = library
                .get_function(entry, None)
                .map_err(|message| metal_driver(format!("Metal entry lookup failed: {message}")))?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|message| metal_driver(format!("compute pipeline failed: {message}")))?;
            pipelines.insert(entry.clone(), pipeline);
        }
        let token = self.next_token;
        self.next_token += 1;
        self.modules.insert(token, MetalModule { pipelines });
        Ok(token)
    }

    fn alloc(&mut self, len_bytes: usize) -> HostResult<u64> {
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| metal_unavailable("SystemMetalDriver has no device"))?;
        let buffer = device.new_buffer(len_bytes as u64, MTLResourceOptions::StorageModeShared);
        let token = self.next_token;
        self.next_token += 1;
        self.buffers.insert(token, buffer);
        Ok(token)
    }

    fn copy_in(&mut self, token: u64, bytes: &[u8]) -> HostResult<()> {
        let buffer = self
            .buffers
            .get(&token)
            .ok_or_else(|| metal_driver("copy_in: unknown buffer token"))?;
        if buffer.length() as usize != bytes.len() {
            return Err(HostError::invalid_args("Metal copy_in size mismatch"));
        }
        let destination = buffer.contents().cast::<u8>();
        // Safe: the buffer length is exactly `bytes.len()` (checked above), and
        // shared-memory storage is host-accessible.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), destination, bytes.len());
        }
        Ok(())
    }

    fn launch_elementwise_add_f32(
        &mut self,
        module: u64,
        a: u64,
        b: u64,
        out: u64,
        len: usize,
    ) -> HostResult<()> {
        // The emitted `add_one` kernel is unary (input + extent + output);
        // `b` is accepted for trait parity but is not bound by the kernel.
        // Validate every token fail-closed so a stale id cannot launch
        // silently, then route through the generalized launch so there is
        // exactly one encoder/commit site per backend (as CUDA does).
        for token in [a, b, out] {
            if !self.buffers.contains_key(&token) {
                return Err(metal_driver("launch: unknown buffer token"));
            }
        }
        let block_x = 256u32;
        let grid_x = len.div_ceil(block_x as usize) as u32;

        // U2 runtime-extent channel: bind the host-supplied element count so
        // the emitted kernel guards against the runtime extent, never a static
        // dispatch shape. The extent buffer rides the buffer slice at the
        // kernel's next-free binding (index 2 for add_one: input=0, output=1)
        // and is dropped after the launch completes.
        let extent = len as u32;
        let extent_buffer = {
            let device = self
                .device
                .as_ref()
                .ok_or_else(|| metal_unavailable("SystemMetalDriver has no device"))?;
            device.new_buffer_with_data(
                (&extent as *const u32).cast(),
                std::mem::size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let extent_token = self.next_token;
        self.next_token += 1;
        self.buffers.insert(extent_token, extent_buffer);

        let result = self.launch_kernel(
            module,
            ELEMENTWISE_ADD_ENTRY,
            &[a, out, extent_token],
            grid_x,
            1,
            1,
            block_x,
            1,
            1,
        );
        self.buffers.remove(&extent_token);
        result
    }

    fn launch_kernel(
        &mut self,
        module: u64,
        entry: &[u8],
        buffers: &[u64],
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
    ) -> HostResult<()> {
        // A module is compiled for exactly one kernel entry; an unknown entry
        // name fails closed with the typed E_DEVICE_ENTRY_MISMATCH (S1-4 audit
        // P2-2: the real-driver adapter reports the same stable code the fake
        // driver enforces, so entry mismatches never surface as a generic
        // driver failure).
        let module_record = self
            .modules
            .get(&module)
            .ok_or_else(|| metal_driver("launch: unknown module token"))?;
        let entry_name = String::from_utf8_lossy(entry);
        let pipeline = module_record.pipelines.get(entry_name.as_ref()).ok_or_else(|| {
            HostError {
                code: crate::device_descriptor::E_DEVICE_ENTRY_MISMATCH.to_owned(),
                message: format!(
                    "launch: module has no entry named {entry_name}"
                ),
                retryable: false,
            }
        })?;
        if grid_x == 0 || grid_y == 0 || grid_z == 0 || block_x == 0 || block_y == 0 || block_z == 0
        {
            return Err(metal_driver(
                "launch: all grid and block dimensions must be non-zero",
            ));
        }
        // Resolve every buffer token fail-closed before touching the encoder,
        // so a stale or non-buffer id cannot silently launch.
        let mut bound = Vec::with_capacity(buffers.len());
        for token in buffers {
            let buffer = self
                .buffers
                .get(token)
                .ok_or_else(|| metal_driver("launch: unknown buffer token"))?;
            bound.push(buffer);
        }
        let queue = self
            .queue
            .as_ref()
            .ok_or_else(|| metal_unavailable("SystemMetalDriver has no command queue"))?;

        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        for (index, buffer) in bound.iter().enumerate() {
            encoder.set_buffer(index as u64, Some(*buffer), 0);
        }

        // Metal caps threads per threadgroup; a block shape whose volume
        // exceeds the pipeline limit fails closed rather than being clamped,
        // because clamping would flatten the block shape and break 2D/3D
        // thread indexing (`thread_position_in_threadgroup.y` would read 0).
        // The elementwise path passes block_y=block_z=1, so its shape is
        // (block_x, 1, 1) — unchanged from before.
        let max_threads = pipeline.max_total_threads_per_threadgroup();
        let block_volume = (block_x as u128) * (block_y as u128) * (block_z as u128);
        if block_volume > max_threads as u128 {
            return Err(metal_driver(format!(
                "launch: threadgroup volume {block_volume} exceeds the pipeline limit of {max_threads} threads per threadgroup"
            )));
        }
        // Each threadgroup carries exactly one block now that the volume is
        // guaranteed to fit, so the grid needs no widening along x.
        let thread_groups_x = grid_x as u64;
        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups_x, grid_y as u64, grid_z as u64),
            MTLSize::new(block_x as u64, block_y as u64, block_z as u64),
        );
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(metal_driver("Metal command buffer did not complete"));
        }
        Ok(())
    }

    fn sync(&mut self) -> HostResult<()> {
        // Launch already calls `wait_until_completed`; shared-memory reads are
        // coherent without an additional barrier.
        Ok(())
    }

    fn copy_out(&mut self, token: u64, len_bytes: usize) -> HostResult<Vec<u8>> {
        let buffer = self
            .buffers
            .get(&token)
            .ok_or_else(|| metal_driver("copy_out: unknown buffer token"))?;
        if buffer.length() as usize != len_bytes {
            return Err(HostError::internal("Metal copy_out size mismatch"));
        }
        let mut output = vec![0u8; len_bytes];
        let source = buffer.contents().cast::<u8>();
        // Safe: the buffer length is exactly `len_bytes` (checked above), and
        // shared-memory storage is host-readable.
        unsafe {
            std::ptr::copy_nonoverlapping(source, output.as_mut_ptr(), len_bytes);
        }
        Ok(output)
    }

    fn free(&mut self, token: u64) -> HostResult<()> {
        self.buffers.remove(&token);
        self.modules.remove(&token);
        Ok(())
    }
}

/// Every `kernel void <name>` entry point in an MSL module, in source order.
/// A program module declares one entry per kernel (S2-5 multi-kernel
/// modules); a source that declares no kernels yields `None`, and a malformed
/// marker (a `kernel void` with no name) also fails closed.
fn msl_kernel_entry_names(source: &str) -> Option<Vec<String>> {
    const MARKER: &str = "kernel void";
    let mut names = Vec::new();
    let mut rest = source;
    while let Some(marker_at) = rest.find(MARKER) {
        let after = &rest[marker_at + MARKER.len()..];
        let trimmed = after.trim_start();
        let name_len = trimmed
            .bytes()
            .take_while(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
            .count();
        if name_len == 0 {
            return None;
        }
        names.push(trimmed[..name_len].to_owned());
        rest = &trimmed[name_len..];
    }
    if names.is_empty() {
        None
    } else {
        Some(names)
    }
}

fn metal_driver(message: impl Into<String>) -> HostError {
    HostError {
        code: E_METAL_DRIVER.to_owned(),
        message: message.into(),
        retryable: false,
    }
}

impl fmt::Debug for MetalHostSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalHostSession")
            .field("admitted", &self.admitted)
            .field("handles", &self.handles.len())
            .finish_non_exhaustive()
    }
}
