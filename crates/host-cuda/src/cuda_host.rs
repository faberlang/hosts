//! CUDA product-host lifecycle for MIR v1 (lane C).
//!
//! G2 wires `SystemCudaDriver` over the real CUDA Driver API via `libloading`
//! (dlopen `libcuda.so.1`, raw `fn` pointer symbols), generalized
//! `launch_kernel`, and the probe/loader-parity candidate list. A real Driver
//! API product run is not claimed without a loadable device stack; injected
//! drivers prove sequencing only.

use std::collections::BTreeMap;
use std::ffi::{c_char, c_void, CStr};
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

use faber::Valor;
use host_coordinator::device_identity::{DeviceHealthGeneration, DeviceOrdinal, PhysicalDeviceId};
use host_coordinator::discovery::{
    ComputeCapability, DeviceCapabilities, DeviceDiscoveryEntry, DeviceDiscoverySnapshot,
    DeviceHealth, DeviceMemory, DtypeSurface, ProbeProvenance,
};
use serde::{Deserialize, Serialize};

use crate::device_descriptor::DeviceDataType;
use crate::device_registry::{DriverCounters, FakeFailureStage, HandleRegistry};
use crate::kernel::frame_data;
use crate::kernel::{HostError, HostResult};

/// Stable host error code for missing CUDA driver/device/toolchain.
pub const E_CUDA_UNAVAILABLE: &str = "E_CUDA_UNAVAILABLE";
/// Reserved for future use — no current emit site; today's codes are E_CUDA_UNAVAILABLE / E_CUDA_DRIVER.
pub const E_CUDA_UNSUPPORTED: &str = "E_CUDA_UNSUPPORTED";
/// Stale or unknown opaque handle.
pub const E_CUDA_INVALID_HANDLE: &str = "E_CUDA_INVALID_HANDLE";
/// Driver-level failure after admission.
pub const E_CUDA_DRIVER: &str = "E_CUDA_DRIVER";

/// Default kernel entry for the legacy `launch_elementwise_add_f32` session
/// path. Matches the emitted `@addita` entry of the C4 proof fixture
/// (`radix/corpus/cuda/addita-proof.fab` → PTX `.entry addita`).
const ELEMENTWISE_ADD_ENTRY: &[u8] = b"addita";

/// Single source of truth for both the admission probe and the Driver API
/// loader (CTO pin from `cuda-first-product-proof` G2: probe/loader parity).
/// `libcuda.so.1` is listed first — the exact pharos path is
/// `/lib/x86_64-linux-gnu/libcuda.so.1`, which the previous probe missed.
const LIBCUDA_CANDIDATE_PATHS: &[&str] = &[
    "/lib/x86_64-linux-gnu/libcuda.so.1",
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    "/usr/lib/x86_64-linux-gnu/libcuda.so",
    "/usr/lib/libcuda.so",
    "/usr/lib/libcuda.dylib",
    "/usr/local/cuda/lib64/libcuda.so",
    "/usr/local/cuda/lib/libcuda.dylib",
];

/// Existing libcuda candidates, in probe/loader agreement order. Includes a
/// best-effort scan of common CUDA roots without walking the world.
fn libcuda_candidate_paths() -> Vec<String> {
    let mut candidates = Vec::new();
    for path in LIBCUDA_CANDIDATE_PATHS {
        if Path::new(path).exists() {
            candidates.push((*path).to_owned());
        }
    }
    for root in ["/opt/cuda", "/usr/local/cuda"] {
        let candidate = PathBuf::from(root).join("lib64/libcuda.so");
        if candidate.is_file() {
            let text = candidate.display().to_string();
            if !candidates.contains(&text) {
                candidates.push(text);
            }
        }
    }
    candidates
}

/// Read-only environment admission report (never a product run claim).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaEnvReport {
    pub admitted: bool,
    pub nvidia_smi: Option<String>,
    pub libcuda_candidates: Vec<String>,
    pub reason: String,
}

/// Opaque host-owned handle identity carried at the Frame control boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CudaHandleId(pub u64);

#[derive(Clone, Debug, PartialEq, Eq)]
enum CudaHandleKind {
    Module,
    Buffer { len_bytes: usize },
}

/// Injectable driver boundary (real Driver API adapter or sequencing fake).
pub trait CudaDriver: Send {
    fn discover(&mut self) -> HostResult<CudaEnvReport>;
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
    /// `cuLaunchKernel` call site.
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
pub struct CudaHostSession {
    driver: Box<dyn CudaDriver>,
    handles: HandleRegistry<CudaHandleKind>,
    admitted: bool,
    /// HostProvided once-init copies issued through this session.
    uploads: usize,
}

impl CudaHostSession {
    /// Open a session against the live environment. Fails closed when the
    /// machine cannot admit a CUDA product stack.
    pub fn try_open() -> HostResult<Self> {
        let mut driver = Box::new(SystemCudaDriver::default());
        let report = driver.discover()?;
        if !report.admitted {
            return Err(cuda_unavailable(report.reason));
        }
        Self::from_driver(driver, true)
    }

    /// Inject a driver for unit tests (sequencing / reject paths only).
    pub fn with_driver(mut driver: Box<dyn CudaDriver>) -> HostResult<Self> {
        let report = driver.discover()?;
        Self::from_driver(driver, report.admitted)
    }

    /// Shared session assembly from an already-discovered driver: create the
    /// backend context when the driver admits, then wrap the driver with an
    /// empty handle registry. Both the live-environment opener and the
    /// injectable test seam use this so the admission → context → session
    /// setup lives in exactly one place.
    fn from_driver(mut driver: Box<dyn CudaDriver>, admitted: bool) -> HostResult<Self> {
        if admitted {
            driver.create_context()?;
        }
        Ok(Self {
            driver,
            handles: HandleRegistry::new(),
            admitted,
            uploads: 0,
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
    /// boundary; the real drivers report those as zero (S2-8 real-device
    /// gate). HostProvided uploads are counted on the session for both.
    #[must_use]
    pub fn driver_counters(&self) -> DriverCounters {
        let mut counters = self.driver.counters();
        counters.uploads = self.uploads;
        counters
    }

    /// Record one HostProvided PerProgram weight copy through this session.
    pub fn record_weight_upload(&mut self) {
        self.uploads = self.uploads.saturating_add(1);
    }

    pub fn load_module(&mut self, image: &[u8]) -> HostResult<CudaHandleId> {
        self.require_admitted()?;
        if image.is_empty() {
            return Err(HostError::invalid_args("CUDA module image is empty"));
        }
        let token = self.driver.load_module(image)?;
        Ok(self.insert(CudaHandleKind::Module, token))
    }

    pub fn alloc_bytes(&mut self, len_bytes: usize) -> HostResult<CudaHandleId> {
        self.require_admitted()?;
        if len_bytes == 0 {
            return Err(HostError::invalid_args(
                "CUDA buffer length must be non-zero",
            ));
        }
        let token = self.driver.alloc(len_bytes)?;
        Ok(self.insert(CudaHandleKind::Buffer { len_bytes }, token))
    }

    pub fn copy_in_f32(&mut self, buffer: CudaHandleId, values: &[f32]) -> HostResult<()> {
        self.copy_in_bytes(buffer, f32_slice_as_bytes(values), DeviceDataType::F32)
    }

    /// Copy dtype-tagged bytes into a device buffer without changing their
    /// representation. Length must match the allocation and be a multiple of
    /// `dtype`'s byte width; a shorter tail is rejected, never padded.
    pub fn copy_in_bytes(
        &mut self,
        buffer: CudaHandleId,
        bytes: &[u8],
        dtype: DeviceDataType,
    ) -> HostResult<()> {
        self.require_admitted()?;
        if bytes.len() % dtype.byte_width() != 0 {
            return Err(cuda_misaligned_tail(dtype, bytes.len()));
        }
        let (token, len_bytes) = self.buffer_token(buffer)?;
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
        module: CudaHandleId,
        a: CudaHandleId,
        b: CudaHandleId,
        out: CudaHandleId,
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
        module: CudaHandleId,
        entry: &str,
        buffers: &[CudaHandleId],
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
        module: CudaHandleId,
        entry: &str,
        buffers: &[CudaHandleId],
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
            return Err(HostError::invalid_args("CUDA kernel entry name is empty"));
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

    pub fn readback_bytes(
        &mut self,
        buffer: CudaHandleId,
        dtype: DeviceDataType,
    ) -> HostResult<Vec<u8>> {
        self.require_admitted()?;
        let (token, len_bytes) = self.buffer_token(buffer)?;
        if len_bytes % dtype.byte_width() != 0 {
            return Err(cuda_misaligned_tail(dtype, len_bytes));
        }
        let bytes = self.driver.copy_out(token, len_bytes)?;
        if bytes.len() != len_bytes {
            return Err(HostError::internal(
                "CUDA readback returned unexpected byte length",
            ));
        }
        Ok(bytes)
    }

    pub fn readback_f32(&mut self, buffer: CudaHandleId) -> HostResult<Vec<f32>> {
        let bytes = self.readback_bytes(buffer, DeviceDataType::F32)?;
        if bytes.len() % 4 != 0 {
            return Err(HostError::internal(
                "CUDA readback returned unexpected f32 byte length",
            ));
        }
        Ok(f32_bytes_to_values(&bytes))
    }

    pub fn release(&mut self, id: CudaHandleId) -> HostResult<()> {
        let Some(handle) = self.handles.remove(id.0) else {
            return Err(cuda_invalid_handle(id));
        };
        self.driver.free(handle.backend_token)
    }

    /// Control-frame representation of a handle (opaque id only; no payload).
    pub fn handle_frame_data(id: CudaHandleId) -> Valor {
        frame_data::tabula([("cuda_handle", Valor::Numerus(id.0 as i64))])
    }

    fn require_admitted(&self) -> HostResult<()> {
        if self.admitted {
            Ok(())
        } else {
            Err(cuda_unavailable(
                "CUDA host session is not admitted for product execution",
            ))
        }
    }

    fn insert(&mut self, kind: CudaHandleKind, backend_token: u64) -> CudaHandleId {
        CudaHandleId(self.handles.insert(kind, backend_token))
    }

    fn module_token(&self, id: CudaHandleId) -> HostResult<u64> {
        match self.handles.get(id.0) {
            Some(handle) if matches!(handle.kind, CudaHandleKind::Module) => {
                Ok(handle.backend_token)
            }
            Some(_) => Err(HostError::invalid_args("handle is not a CUDA module")),
            None => Err(cuda_invalid_handle(id)),
        }
    }

    fn buffer_token(&self, id: CudaHandleId) -> HostResult<(u64, usize)> {
        match self.handles.get(id.0) {
            Some(handle) => match &handle.kind {
                CudaHandleKind::Buffer { len_bytes } => Ok((handle.backend_token, *len_bytes)),
                _ => Err(HostError::invalid_args("handle is not a CUDA buffer")),
            },
            None => Err(cuda_invalid_handle(id)),
        }
    }
}

fn f32_slice_as_bytes(values: &[f32]) -> &[u8] {
    // Safe: f32 is plain bits; host owns the slice for the duration of copy_in.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

/// Reinterpret f32 bytes back into values. Shared by the session readback
/// surface and the fake-driver simulation reads.
fn f32_bytes_to_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn cuda_misaligned_tail(dtype: DeviceDataType, len: usize) -> HostError {
    HostError::invalid_args(format!(
        "CUDA copy rejects a misaligned {} tail of {} bytes",
        dtype.spelling(),
        len
    ))
}

fn cuda_unavailable(message: impl Into<String>) -> HostError {
    HostError {
        code: E_CUDA_UNAVAILABLE.to_owned(),
        message: message.into(),
        retryable: false,
    }
}

fn cuda_invalid_handle(id: CudaHandleId) -> HostError {
    HostError {
        code: E_CUDA_INVALID_HANDLE.to_owned(),
        message: format!("unknown or released CUDA handle {}", id.0),
        retryable: false,
    }
}

fn cuda_driver(message: impl Into<String>) -> HostError {
    HostError {
        code: E_CUDA_DRIVER.to_owned(),
        message: message.into(),
        retryable: false,
    }
}

/// Probe this machine for a loadable CUDA product stack without claiming a run.
pub fn probe_cuda_environment() -> CudaEnvReport {
    let nvidia_smi = Command::new("nvidia-smi")
        .arg("--query-gpu=name,driver_version")
        .arg("--format=csv,noheader")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|text| !text.is_empty());

    // One candidate list shared with the Driver API loader (probe/loader
    // parity): an admitted machine must always be dlopen-able.
    let libcuda_candidates = libcuda_candidate_paths();

    let admitted = nvidia_smi.is_some() || !libcuda_candidates.is_empty();
    let reason = if admitted {
        "NVIDIA driver/device stack signals present; product launch still requires a real Driver API adapter and compiler leaf artifact".to_owned()
    } else {
        "no nvidia-smi output and no known libcuda path on this machine; CUDA product execution is not admitted".to_owned()
    };
    CudaEnvReport {
        admitted,
        nvidia_smi,
        libcuda_candidates,
        reason,
    }
}

/// One enumerated CUDA physical device. The ordinal is a locator only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaPhysicalDevice {
    /// Enumeration-order locator — never identity.
    pub ordinal: u32,
    /// nvidia-smi PCI UUID (`GPU-…` prefix included) when the tool report
    /// exists; otherwise `GPU-` plus the driver UUID.
    pub pci_uuid: String,
    /// Driver API UUID without the `GPU-` prefix, when `cuDeviceGetUuid` succeeds.
    pub driver_uuid: Option<String>,
    /// `cuDeviceGetName` model string.
    pub device_model: Option<String>,
    /// nvidia-smi `memory.total` (MiB). Distinct from the driver byte total.
    pub tool_report_total_mib: Option<u64>,
    /// `cuDeviceTotalMem` total, bytes.
    pub api_total_bytes: u64,
    /// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR`.
    pub compute_capability_major: u32,
    /// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR`.
    pub compute_capability_minor: u32,
    /// `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`.
    pub sm_count: u32,
    /// `CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK`.
    pub max_threads_per_workgroup: u32,
    /// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK`.
    pub workgroup_shared_memory_min_bytes: u32,
    /// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`.
    pub workgroup_shared_memory_max_bytes: u32,
    /// `CU_DEVICE_ATTRIBUTE_WARP_SIZE`.
    pub collective_width: u32,
    /// `CU_DEVICE_ATTRIBUTE_INTEGRATED` (1 → unified with host memory).
    pub unified_memory: bool,
    /// nvidia-smi driver version, when present.
    pub driver_version: Option<String>,
}

impl CudaPhysicalDevice {
    /// Convert this probe row into a host-coordinator discovery entry.
    #[must_use]
    pub fn to_discovery_entry(&self) -> DeviceDiscoveryEntry {
        let tool_versions = match &self.driver_version {
            Some(version) => format!("driver {version} / CUDA Driver API"),
            None => "CUDA Driver API".to_owned(),
        };
        DeviceDiscoveryEntry {
            ordinal: DeviceOrdinal::new(self.ordinal),
            identity: PhysicalDeviceId::cuda(self.pci_uuid.clone(), self.driver_uuid.clone()),
            device_model: self.device_model.clone(),
            capabilities: DeviceCapabilities {
                compute_capability: ComputeCapability {
                    major: self.compute_capability_major,
                    minor: self.compute_capability_minor,
                },
                sm_count: self.sm_count,
                dtype_surface: DtypeSurface::empty(),
                max_threads_per_workgroup: self.max_threads_per_workgroup,
                workgroup_shared_memory_min_bytes: self.workgroup_shared_memory_min_bytes,
                workgroup_shared_memory_max_bytes: self.workgroup_shared_memory_max_bytes,
                collective_width: self.collective_width,
                unified_memory: self.unified_memory,
            },
            memory: DeviceMemory {
                tool_report_total_mib: self.tool_report_total_mib,
                api_total_bytes: self.api_total_bytes,
            },
            health: DeviceHealth::Healthy,
            health_generation: DeviceHealthGeneration::initial(),
            probe_provenance: ProbeProvenance {
                probe: "cuDeviceGetCount + nvidia-smi".to_owned(),
                tool_versions,
            },
        }
    }
}

/// Enumerate every locally attached CUDA device into identity/memory facts.
///
/// Returns an empty list when the machine does not admit a CUDA stack. A
/// present-but-broken driver fails closed (`E_CUDA_UNAVAILABLE`).
pub fn enumerate_cuda_physical_devices() -> HostResult<Vec<CudaPhysicalDevice>> {
    Ok(enumerate_cuda_devices_with_handles()?
        .into_iter()
        .map(|row| row.facts)
        .collect())
}

/// Timestamped discovery snapshot of every locally attached CUDA device.
pub fn discover_cuda_snapshot(probe_utc_nanos: u64) -> HostResult<DeviceDiscoverySnapshot> {
    let devices = enumerate_cuda_physical_devices()?;
    Ok(DeviceDiscoverySnapshot::from_enumerated(
        probe_utc_nanos,
        devices.iter().map(CudaPhysicalDevice::to_discovery_entry),
    ))
}

struct EnumeratedCudaDevice {
    handle: i32,
    facts: CudaPhysicalDevice,
}

struct NvidiaSmiGpu {
    index: u32,
    uuid: String,
    memory_mib: Option<u64>,
    driver_version: Option<String>,
}

fn query_nvidia_smi_gpus() -> Vec<NvidiaSmiGpu> {
    let Some(output) = Command::new("nvidia-smi")
        .arg("--query-gpu=index,uuid,memory.total,driver_version")
        .arg("--format=csv,noheader,nounits")
        .output()
        .ok()
        .filter(|output| output.status.success())
    else {
        return Vec::new();
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.lines().filter_map(parse_nvidia_smi_gpu).collect()
}

fn parse_nvidia_smi_gpu(line: &str) -> Option<NvidiaSmiGpu> {
    let mut parts = line.split(',').map(str::trim);
    let index = parts.next()?.parse().ok()?;
    let uuid = parts.next().filter(|text| !text.is_empty())?.to_owned();
    let memory_mib = parts.next().and_then(|text| text.parse().ok());
    let driver_version = parts
        .next()
        .filter(|text| !text.is_empty())
        .map(ToOwned::to_owned);
    Some(NvidiaSmiGpu {
        index,
        uuid,
        memory_mib,
        driver_version,
    })
}

fn enumerate_cuda_devices_with_handles() -> HostResult<Vec<EnumeratedCudaDevice>> {
    let report = probe_cuda_environment();
    if !report.admitted {
        return Ok(Vec::new());
    }
    let library = load_libcuda(&report)?;
    let api = unsafe { resolve_cuda_api(&library) }?;
    let result = unsafe { (api.cu_init)(0) };
    if result != CUDA_SUCCESS {
        return Err(cuda_unavailable(format!(
            "cuInit failed with CUDA result {result} after dlopen; driver present but unusable"
        )));
    }
    let smi = query_nvidia_smi_gpus();
    enumerate_cuda_devices_with_api(&api, &smi)
}

fn enumerate_cuda_devices_with_api(
    api: &CudaDriverApi,
    smi: &[NvidiaSmiGpu],
) -> HostResult<Vec<EnumeratedCudaDevice>> {
    let mut count: i32 = 0;
    let result = unsafe { (api.cu_device_get_count)(&mut count) };
    if result != CUDA_SUCCESS {
        return Err(cuda_unavailable(format!(
            "cuDeviceGetCount failed with CUDA result {result}"
        )));
    }
    if count < 0 {
        return Err(cuda_unavailable(
            "cuDeviceGetCount returned a negative device count",
        ));
    }
    let mut devices = Vec::with_capacity(count as usize);
    for ordinal in 0..count {
        devices.push(identify_cuda_device(api, ordinal, smi)?);
    }
    Ok(devices)
}

fn identify_cuda_device(
    api: &CudaDriverApi,
    ordinal: i32,
    smi: &[NvidiaSmiGpu],
) -> HostResult<EnumeratedCudaDevice> {
    let mut handle: i32 = 0;
    let result = unsafe { (api.cu_device_get)(&mut handle, ordinal) };
    if result != CUDA_SUCCESS {
        return Err(cuda_unavailable(format!(
            "cuDeviceGet({ordinal}) failed with CUDA result {result}"
        )));
    }
    let driver_uuid = cuda_device_uuid(api, handle);
    let smi_row = smi.iter().find(|row| row.index == ordinal as u32);
    let pci_uuid = match smi_row {
        Some(row) => row.uuid.clone(),
        None => match &driver_uuid {
            Some(uuid) => format!("GPU-{uuid}"),
            None => {
                return Err(cuda_unavailable(format!(
                    "CUDA ordinal {ordinal} produced no PCI UUID and no driver UUID"
                )));
            }
        },
    };
    Ok(EnumeratedCudaDevice {
        handle,
        facts: CudaPhysicalDevice {
            ordinal: ordinal as u32,
            pci_uuid,
            driver_uuid,
            device_model: cuda_device_name(api, handle),
            tool_report_total_mib: smi_row.and_then(|row| row.memory_mib),
            api_total_bytes: cuda_device_total_mem(api, handle).unwrap_or(0),
            compute_capability_major: cuda_device_attribute(
                api,
                handle,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            )
            .unwrap_or(0),
            compute_capability_minor: cuda_device_attribute(
                api,
                handle,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            )
            .unwrap_or(0),
            sm_count: cuda_device_attribute(api, handle, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
                .unwrap_or(0),
            max_threads_per_workgroup: cuda_device_attribute(
                api,
                handle,
                CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            )
            .unwrap_or(0),
            workgroup_shared_memory_min_bytes: cuda_device_attribute(
                api,
                handle,
                CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
            )
            .unwrap_or(0),
            workgroup_shared_memory_max_bytes: cuda_device_attribute(
                api,
                handle,
                CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )
            .unwrap_or(0),
            collective_width: cuda_device_attribute(api, handle, CU_DEVICE_ATTRIBUTE_WARP_SIZE)
                .unwrap_or(0),
            unified_memory: cuda_device_attribute(api, handle, CU_DEVICE_ATTRIBUTE_INTEGRATED)
                .unwrap_or(0)
                != 0,
            driver_version: smi_row.and_then(|row| row.driver_version.clone()),
        },
    })
}

fn cuda_device_uuid(api: &CudaDriverApi, handle: i32) -> Option<String> {
    let mut uuid = CuUuid { bytes: [0u8; 16] };
    let result = unsafe { (api.cu_device_get_uuid)(&mut uuid, handle) };
    if result != CUDA_SUCCESS {
        return None;
    }
    Some(format_cuda_uuid(&uuid.bytes))
}

fn cuda_device_name(api: &CudaDriverApi, handle: i32) -> Option<String> {
    let mut name: [c_char; 256] = [0; 256];
    let result = unsafe { (api.cu_device_get_name)(name.as_mut_ptr(), name.len() as i32, handle) };
    if result != CUDA_SUCCESS {
        return None;
    }
    name[255] = 0;
    let cstr = unsafe { CStr::from_ptr(name.as_ptr()) };
    let text = cstr.to_string_lossy();
    if text.is_empty() {
        None
    } else {
        Some(text.into_owned())
    }
}

fn cuda_device_total_mem(api: &CudaDriverApi, handle: i32) -> Option<u64> {
    let mut bytes: usize = 0;
    let result = unsafe { (api.cu_device_total_mem)(&mut bytes, handle) };
    if result == CUDA_SUCCESS {
        Some(bytes as u64)
    } else {
        None
    }
}

fn cuda_device_attribute(api: &CudaDriverApi, handle: i32, attribute: i32) -> Option<u32> {
    let mut value: i32 = 0;
    let result = unsafe { (api.cu_device_get_attribute)(&mut value, attribute, handle) };
    if result == CUDA_SUCCESS && value >= 0 {
        Some(value as u32)
    } else {
        None
    }
}

fn format_cuda_uuid(bytes: &[u8; 16]) -> String {
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15]
    )
}

/// Live-environment driver over the real CUDA Driver API (G2).
///
/// Dynamically loads `libcuda.so.1` via `libloading` — the one justified
/// driver dependency — and resolves the Driver API symbols to raw `fn`
/// pointers at load time (`libloading::Symbol` is not `Send`; the raw
/// pointers keep `Box<dyn CudaDriver: Send>` honest). Every symbol stays
/// valid for the lifetime of the owned [`libloading::Library`].
///
/// Error mapping is pinned by `cuda-first-product-proof`: dlopen / symbol
/// resolution / `cuInit` failures are `E_CUDA_UNAVAILABLE` (admission is
/// skip-worthy only when the proof env vars are absent); anything after a
/// live context is `E_CUDA_DRIVER`.
#[derive(Default)]
struct SystemCudaDriver {
    /// Holds the loaded library alive for the driver's lifetime. Never read:
    /// the raw `fn` pointers in `api` stay valid only while this lives.
    _library: Option<libloading::Library>,
    /// Resolved Driver API symbols (raw `fn` pointers), present after a
    /// successful `discover`.
    api: Option<CudaDriverApi>,
    /// Module handle tokens (`CUmodule`), resolved to the module's function
    /// entry by name at launch time.
    modules: BTreeMap<u64, OpaqueHandle>,
    /// Buffer tokens mapped to `CUdeviceptr` values.
    buffers: BTreeMap<u64, u64>,
    /// Retained primary context made current before each context-dependent API
    /// call. CUDA current-context state is thread-local; reassert it so the
    /// one-shot proof is not sensitive to driver calls that disturb it.
    context: Option<OpaqueHandle>,
    /// First enumerated `CUdevice` handle from the per-ordinal discover loop.
    /// Locator only; identity lives on [`CudaPhysicalDevice`].
    enumerated_device: Option<i32>,
    next_token: u64,
}

/// Opaque Driver API handle (`CUmodule`). Raw pointers are not `Send`; this
/// wrapper asserts it deliberately: the handles are process-lifetime opaque
/// tokens we never dereference, and the driver must satisfy
/// `Box<dyn CudaDriver: Send>`.
#[derive(Clone, Copy)]
struct OpaqueHandle(*mut c_void);

// Safety: the wrapped pointer is an opaque driver-owned token (CUmodule /
// CUcontext). It is never dereferenced or freed by this struct; it is only
// handed back to the Driver API, whose lifetime the owning Library bounds.
unsafe impl Send for OpaqueHandle {}

impl CudaDriver for SystemCudaDriver {
    fn discover(&mut self) -> HostResult<CudaEnvReport> {
        let report = probe_cuda_environment();
        if !report.admitted {
            return Err(cuda_unavailable(report.reason));
        }
        // Admission signals are present — load the real binding now. dlopen,
        // symbol resolution, and cuInit all map to E_CUDA_UNAVAILABLE: a
        // present-but-broken stack must fail closed, never green-flag a run.
        let library = load_libcuda(&report)?;
        let api = unsafe { resolve_cuda_api(&library) }?;
        let result = unsafe { (api.cu_init)(0) };
        if result != CUDA_SUCCESS {
            return Err(cuda_unavailable(format!(
                "cuInit failed with CUDA result {result} after dlopen; driver present but unusable"
            )));
        }
        let smi = query_nvidia_smi_gpus();
        let enumerated = enumerate_cuda_devices_with_api(&api, &smi)?;
        let first = enumerated
            .first()
            .ok_or_else(|| cuda_unavailable("cuDeviceGetCount returned 0 devices after cuInit"))?;
        self.enumerated_device = Some(first.handle);
        self._library = Some(library);
        self.api = Some(api);
        Ok(report)
    }

    fn create_context(&mut self) -> HostResult<()> {
        let api = self.loaded_api()?;
        let device = self
            .enumerated_device
            .ok_or_else(|| cuda_driver("SystemCudaDriver has no enumerated CUDA device"))?;
        // cuDevicePrimaryCtxRetain: the modern (non-deprecated) path. cuCtxCreate
        // is deprecated in CUDA 12+ headers but functional in 13.2; the retained
        // primary context is made current for this thread, so every subsequent
        // call (module load, mem, launch, sync) targets it. The retained context
        // outlives the one-shot proof process; teardown (cuDevicePrimaryCtxRelease /
        // cuCtxDestroy) is deferred and recorded here, not silent.
        let mut context: *mut c_void = std::ptr::null_mut();
        let mut result = unsafe { (api.cu_device_primary_ctx_retain)(&mut context, device) };
        if result != CUDA_SUCCESS {
            return Err(cuda_driver(format!(
                "cuDevicePrimaryCtxRetain failed with CUDA result {result}"
            )));
        }
        result = unsafe { (api.cu_ctx_set_current)(context) };
        if result != CUDA_SUCCESS {
            return Err(cuda_driver(format!(
                "cuCtxSetCurrent failed with CUDA result {result}"
            )));
        }
        self.context = Some(OpaqueHandle(context));
        Ok(())
    }

    fn load_module(&mut self, image: &[u8]) -> HostResult<u64> {
        let api = self.current_api()?;
        // cuModuleLoadData requires a NUL-terminated image. PTX files end with
        // a newline, so append the terminator explicitly.
        let mut image_terminated = Vec::with_capacity(image.len() + 1);
        image_terminated.extend_from_slice(image);
        image_terminated.push(0);
        let mut module: *mut c_void = std::ptr::null_mut();
        let result =
            unsafe { (api.cu_module_load_data)(&mut module, image_terminated.as_ptr().cast()) };
        if result != CUDA_SUCCESS {
            return Err(cuda_driver(format!(
                "cuModuleLoadData failed with CUDA result {result} (PTX rejected by driver)"
            )));
        }
        let token = self.next_token;
        self.next_token += 1;
        self.modules.insert(token, OpaqueHandle(module));
        Ok(token)
    }

    fn alloc(&mut self, len_bytes: usize) -> HostResult<u64> {
        let api = self.current_api()?;
        let mut device_ptr: u64 = 0;
        let result = unsafe { (api.cu_mem_alloc)(&mut device_ptr, len_bytes) };
        if result != CUDA_SUCCESS {
            return Err(cuda_driver(format!(
                "cuMemAlloc failed with CUDA result {result}"
            )));
        }
        let token = self.next_token;
        self.next_token += 1;
        self.buffers.insert(token, device_ptr);
        Ok(token)
    }

    fn copy_in(&mut self, token: u64, bytes: &[u8]) -> HostResult<()> {
        let api = self.current_api()?;
        let device_ptr = self
            .buffers
            .get(&token)
            .copied()
            .ok_or_else(|| cuda_driver("copy_in: unknown buffer token"))?;
        let result =
            unsafe { (api.cu_memcpy_htod)(device_ptr, bytes.as_ptr().cast(), bytes.len()) };
        if result != CUDA_SUCCESS {
            return Err(cuda_driver(format!(
                "cuMemcpyHtoD failed with CUDA result {result}"
            )));
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
        // Route the legacy session path through the generalized launch so the
        // binding has exactly one cuLaunchKernel call site.
        let block_x = 256u32;
        let grid_x = len.div_ceil(block_x as usize) as u32;
        self.launch_kernel(
            module,
            ELEMENTWISE_ADD_ENTRY,
            &[a, b, out],
            grid_x,
            1,
            1,
            block_x,
            1,
            1,
        )
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
        let api = self.current_api()?;
        let module_handle = self
            .modules
            .get(&module)
            .copied()
            .ok_or_else(|| cuda_driver("launch: unknown module token"))?;
        // Resolve every buffer token fail-closed before touching the driver,
        // so a stale or non-buffer id cannot silently launch.
        let mut device_ptrs = Vec::with_capacity(buffers.len());
        for token in buffers {
            let device_ptr = self
                .buffers
                .get(token)
                .copied()
                .ok_or_else(|| cuda_driver("launch: unknown buffer token"))?;
            device_ptrs.push(device_ptr);
        }
        // cuModuleGetFunction needs a NUL-terminated entry name.
        let mut entry_terminated = Vec::with_capacity(entry.len() + 1);
        entry_terminated.extend_from_slice(entry);
        entry_terminated.push(0);
        let mut function: *mut c_void = std::ptr::null_mut();
        let mut result = unsafe {
            (api.cu_module_get_function)(
                &mut function,
                module_handle.0,
                entry_terminated.as_ptr().cast(),
            )
        };
        if result != CUDA_SUCCESS {
            // S1-4 audit P2-2: the real-driver adapter maps an unknown-entry
            // launch failure to the typed E_DEVICE_ENTRY_MISMATCH (the same
            // code the fake driver enforces), so host callers and the
            // composite host's fail-before-launch surface see one stable
            // spelling for entry mismatches on every driver lane.
            let code = if result == CUDA_ERROR_NOT_FOUND {
                crate::device_descriptor::E_DEVICE_ENTRY_MISMATCH.to_owned()
            } else {
                E_CUDA_DRIVER.to_owned()
            };
            return Err(HostError {
                code,
                message: format!(
                    "cuModuleGetFunction({}) failed with CUDA result {result}",
                    String::from_utf8_lossy(entry)
                ),
                retryable: false,
            });
        }
        // kernelParams is a `void**` array whose entries point at each
        // parameter value. Both the array and the `Vec<u64>` device-pointer
        // values it references must outlive the call — they do, both live in
        // this frame.
        let mut kernel_params: Vec<*mut c_void> = device_ptrs
            .iter()
            .map(|ptr| (ptr as *const u64).cast_mut().cast())
            .collect();
        result = unsafe {
            (api.cu_launch_kernel)(
                function,
                grid_x,
                grid_y,
                grid_z,
                block_x,
                block_y,
                block_z,
                0,
                std::ptr::null_mut(), // default stream
                kernel_params.as_mut_ptr(),
                std::ptr::null_mut(), // no extended launch options
            )
        };
        if result != CUDA_SUCCESS {
            return Err(cuda_driver(format!(
                "cuLaunchKernel({}) failed with CUDA result {result}",
                String::from_utf8_lossy(entry)
            )));
        }
        Ok(())
    }

    fn sync(&mut self) -> HostResult<()> {
        let api = self.current_api()?;
        let result = unsafe { (api.cu_ctx_synchronize)() };
        if result != CUDA_SUCCESS {
            return Err(cuda_driver(format!(
                "cuCtxSynchronize failed with CUDA result {result}"
            )));
        }
        Ok(())
    }

    fn copy_out(&mut self, token: u64, len_bytes: usize) -> HostResult<Vec<u8>> {
        let api = self.current_api()?;
        let device_ptr = self
            .buffers
            .get(&token)
            .copied()
            .ok_or_else(|| cuda_driver("copy_out: unknown buffer token"))?;
        let mut output = vec![0u8; len_bytes];
        let result =
            unsafe { (api.cu_memcpy_dtoh)(output.as_mut_ptr().cast(), device_ptr, len_bytes) };
        if result != CUDA_SUCCESS {
            return Err(cuda_driver(format!(
                "cuMemcpyDtoH failed with CUDA result {result}"
            )));
        }
        Ok(output)
    }

    fn free(&mut self, token: u64) -> HostResult<()> {
        if let Some(device_ptr) = self.buffers.remove(&token) {
            let api = self.current_api()?;
            let result = unsafe { (api.cu_mem_free)(device_ptr) };
            if result != CUDA_SUCCESS {
                return Err(cuda_driver(format!(
                    "cuMemFree failed with CUDA result {result}"
                )));
            }
        }
        // Module teardown (cuModuleUnload) is deferred for the one-shot proof
        // process and recorded here, not silent: the process exits right after
        // the proof and the driver reclaims the module at teardown.
        self.modules.remove(&token);
        Ok(())
    }
}

impl SystemCudaDriver {
    /// The resolved symbol table, present only after a successful `discover`.
    fn loaded_api(&self) -> HostResult<CudaDriverApi> {
        self.api
            .ok_or_else(|| cuda_unavailable("SystemCudaDriver has no loaded Driver API"))
    }

    /// Resolved symbol table after reasserting the retained primary context as
    /// current for this thread.
    fn current_api(&self) -> HostResult<CudaDriverApi> {
        let api = self.loaded_api()?;
        let context = self
            .context
            .ok_or_else(|| cuda_driver("SystemCudaDriver has no retained CUDA context"))?;
        let result = unsafe { (api.cu_ctx_set_current)(context.0) };
        if result != CUDA_SUCCESS {
            return Err(cuda_driver(format!(
                "cuCtxSetCurrent failed with CUDA result {result}"
            )));
        }
        Ok(api)
    }
}

/// `CUresult` success code (`cudaError_t`); any non-zero value is an error.
const CUDA_SUCCESS: i32 = 0;

/// `CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK`.
const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: i32 = 1;
/// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK`.
const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: i32 = 8;
/// `CU_DEVICE_ATTRIBUTE_WARP_SIZE`.
const CU_DEVICE_ATTRIBUTE_WARP_SIZE: i32 = 10;
/// `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`.
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 16;
/// `CU_DEVICE_ATTRIBUTE_INTEGRATED`.
const CU_DEVICE_ATTRIBUTE_INTEGRATED: i32 = 18;
/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR`.
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR`.
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;
/// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`.
const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: i32 = 97;

#[repr(C)]
struct CuUuid {
    bytes: [u8; 16],
}

/// `CUresult` for a symbol/entry not found (`CUDA_ERROR_NOT_FOUND`): the
/// real-driver signal for an unknown kernel entry (`cuModuleGetFunction`).
/// Value 500 is the CUDA 12+ renumbering (verified against the CUDA 13.2
/// toolkit header on pharos: `/usr/local/cuda-13.2/include/cuda.h:2972`).
const CUDA_ERROR_NOT_FOUND: i32 = 500;

/// Raw CUDA Driver API symbol table, resolved once at load time.
///
/// `libloading::Symbol` is not `Send`, so every symbol is converted to a raw
/// `fn` pointer when the library is loaded; `CudaDriverApi` is therefore
/// trivially `Send + Sync` and the driver stays boxable behind
/// `Box<dyn CudaDriver>`.
#[derive(Clone, Copy)]
struct CudaDriverApi {
    cu_init: unsafe extern "C" fn(u32) -> i32,
    cu_device_get_count: unsafe extern "C" fn(*mut i32) -> i32,
    cu_device_get: unsafe extern "C" fn(*mut i32, i32) -> i32,
    cu_device_get_uuid: unsafe extern "C" fn(*mut CuUuid, i32) -> i32,
    cu_device_get_name: unsafe extern "C" fn(*mut c_char, i32, i32) -> i32,
    cu_device_total_mem: unsafe extern "C" fn(*mut usize, i32) -> i32,
    cu_device_get_attribute: unsafe extern "C" fn(*mut i32, i32, i32) -> i32,
    cu_device_primary_ctx_retain: unsafe extern "C" fn(*mut *mut c_void, i32) -> i32,
    cu_ctx_set_current: unsafe extern "C" fn(*mut c_void) -> i32,
    cu_module_load_data: unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32,
    cu_module_get_function:
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> i32,
    cu_launch_kernel: unsafe extern "C" fn(
        *mut c_void,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> i32,
    cu_mem_alloc: unsafe extern "C" fn(*mut u64, usize) -> i32,
    cu_mem_free: unsafe extern "C" fn(u64) -> i32,
    cu_memcpy_htod: unsafe extern "C" fn(u64, *const c_void, usize) -> i32,
    cu_memcpy_dtoh: unsafe extern "C" fn(*mut c_void, u64, usize) -> i32,
    cu_ctx_synchronize: unsafe extern "C" fn() -> i32,
}

/// dlopen the first existing candidate reported by the admission probe. The
/// loader iterates exactly what the probe listed, so an admitted machine
/// cannot later fail with a misleading `E_CUDA_UNAVAILABLE`.
fn load_libcuda(report: &CudaEnvReport) -> HostResult<libloading::Library> {
    let mut last_error: Option<libloading::Error> = None;
    for candidate in &report.libcuda_candidates {
        match unsafe { libloading::Library::new(candidate) } {
            Ok(library) => return Ok(library),
            Err(error) => last_error = Some(error),
        }
    }
    Err(cuda_unavailable(format!(
        "no libcuda candidate from the admission probe could be dlopen'd: {}",
        last_error
            .map(|error| error.to_string())
            .unwrap_or_else(|| "no candidates".to_owned())
    )))
}

/// Resolve every Driver API symbol needed by the binding.
unsafe fn resolve_cuda_api(library: &libloading::Library) -> HostResult<CudaDriverApi> {
    unsafe {
        Ok(CudaDriverApi {
            cu_init: resolve_symbol(library, b"cuInit\0")?,
            cu_device_get_count: resolve_symbol(library, b"cuDeviceGetCount\0")?,
            cu_device_get: resolve_symbol(library, b"cuDeviceGet\0")?,
            cu_device_get_uuid: resolve_symbol(library, b"cuDeviceGetUuid\0")?,
            cu_device_get_name: resolve_symbol(library, b"cuDeviceGetName\0")?,
            cu_device_total_mem: resolve_symbol(library, b"cuDeviceTotalMem_v2\0")?,
            cu_device_get_attribute: resolve_symbol(library, b"cuDeviceGetAttribute\0")?,
            cu_device_primary_ctx_retain: resolve_symbol(library, b"cuDevicePrimaryCtxRetain\0")?,
            cu_ctx_set_current: resolve_symbol(library, b"cuCtxSetCurrent\0")?,
            cu_module_load_data: resolve_symbol(library, b"cuModuleLoadData\0")?,
            cu_module_get_function: resolve_symbol(library, b"cuModuleGetFunction\0")?,
            cu_launch_kernel: resolve_symbol(library, b"cuLaunchKernel\0")?,
            cu_mem_alloc: resolve_symbol(library, b"cuMemAlloc_v2\0")?,
            cu_mem_free: resolve_symbol(library, b"cuMemFree_v2\0")?,
            cu_memcpy_htod: resolve_symbol(library, b"cuMemcpyHtoD_v2\0")?,
            cu_memcpy_dtoh: resolve_symbol(library, b"cuMemcpyDtoH_v2\0")?,
            cu_ctx_synchronize: resolve_symbol(library, b"cuCtxSynchronize\0")?,
        })
    }
}

/// Resolve one symbol and copy the raw `fn` pointer out. The
/// `libloading::Symbol` wrapper (not `Send`) is dropped immediately; the
/// pointer stays valid for the lifetime of the owned `Library`.
unsafe fn resolve_symbol<T: Copy>(library: &libloading::Library, name: &[u8]) -> HostResult<T> {
    unsafe {
        library
            .get::<T>(name)
            .map(|symbol| *symbol)
            .map_err(|error| {
                cuda_unavailable(format!(
                    "libcuda symbol {} could not be resolved: {error}",
                    String::from_utf8_lossy(&name[..name.len() - 1])
                ))
            })
    }
}

/// Sequencing-only fake driver for unit tests. Not product CUDA evidence.
#[derive(Default)]
pub struct FakeCudaDriver {
    next_token: u64,
    buffers: BTreeMap<u64, Vec<u8>>,
    modules: BTreeMap<u64, Vec<u8>>,
    force_unavailable: bool,
    /// Entry names the loaded module's function table declares. Empty means
    /// the fake does not enforce entry checks (legacy sequencing behavior);
    /// non-empty means an unknown launch entry fails closed with
    /// `E_DEVICE_ENTRY_MISMATCH`, mirroring `cuModuleGetFunction` on the real
    /// lane.
    known_entries: Vec<String>,
    /// Matmul plan facts the fake simulates when configured
    /// (`with_matmul_simulation`, U-03 host adapter tests): a 3-buffer launch
    /// computes `out = a × b` for the M·K × K·N → M·N shapes. An absent
    /// configuration keeps the legacy elementwise-add simulation.
    matmul_simulation: Option<(u64, u64, u64)>,
    /// Cumulative module loads (S2-2 module-cache leak bar).
    module_loads: usize,
    /// Cumulative module releases.
    module_releases: usize,
    /// Cumulative buffer allocations.
    buffer_allocs: usize,
    /// Cumulative buffer releases.
    buffer_releases: usize,
    /// Per-stage failure injection (S2-3): stage → 1-based call number whose
    /// invocation fails with a typed `E_CUDA_DRIVER` error. An absent stage
    /// never fails.
    fail_at: BTreeMap<FakeFailureStage, u32>,
    /// Running call count per stage (drives `fail_at`).
    stage_calls: BTreeMap<FakeFailureStage, u32>,
}

impl FakeCudaDriver {
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

    /// Configure the fake to simulate the tiled-matmul kernel (U-03 host
    /// adapter tests): a 3-buffer launch computes `out = a × b` for the given
    /// M·K × K·N → M·N shapes. An absent configuration keeps the legacy
    /// elementwise-add simulation.
    pub fn with_matmul_simulation(mut self, m: u64, k: u64, n: u64) -> Self {
        self.matmul_simulation = Some((m, k, n));
        self
    }

    /// Configure the driver to fail the `call`-th invocation of `stage` with
    /// a typed `E_CUDA_DRIVER` error (S2-3 failure-injection tests). `call`
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
            return Err(cuda_driver(format!(
                "injected failure at {stage:?} stage (S2-3 failure-injection test)"
            )));
        }
        Ok(())
    }

    /// Simulate the `addita` kernel: `out[i] = a[i] + b[i]` elementwise.
    /// Shared by the legacy elementwise-add path and the generalized
    /// `launch_kernel` (the emitted kernel is the same add shape).
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

    /// Simulate the accumulation kernel (G4): `acc[i] += a[i]` — the emitted
    /// elementwise accumulation into a persistent buffer. Repeated launches
    /// accumulate onto the previous device contents (the host's ZeroFill
    /// initialization defines the first state exactly once).
    fn simulate_accumulate(&mut self, module: u64, a: u64, acc: u64) -> HostResult<()> {
        if !self.modules.contains_key(&module) {
            return Err(HostError::internal("fake accumulate missing module"));
        }
        let a_bytes = self
            .buffers
            .get(&a)
            .ok_or_else(|| HostError::internal("fake accumulate missing a"))?
            .clone();
        let acc_buf = self
            .buffers
            .get_mut(&acc)
            .ok_or_else(|| HostError::internal("fake accumulate missing acc"))?;
        if a_bytes.len() != acc_buf.len() {
            return Err(HostError::invalid_args("fake accumulate length mismatch"));
        }
        let len = a_bytes.len() / 4;
        for i in 0..len {
            let ai = f32::from_le_bytes([
                a_bytes[i * 4],
                a_bytes[i * 4 + 1],
                a_bytes[i * 4 + 2],
                a_bytes[i * 4 + 3],
            ]);
            let acc_i = f32::from_le_bytes([
                acc_buf[i * 4],
                acc_buf[i * 4 + 1],
                acc_buf[i * 4 + 2],
                acc_buf[i * 4 + 3],
            ]);
            let sum = (acc_i + ai).to_le_bytes();
            acc_buf[i * 4..i * 4 + 4].copy_from_slice(&sum);
        }
        Ok(())
    }

    /// Simulate a copy kernel: `out[i] = src[i]` — the observation kernel
    /// that reads a persistent accumulation buffer into a readback slot.
    fn simulate_copy(&mut self, module: u64, src: u64, out: u64) -> HostResult<()> {
        if !self.modules.contains_key(&module) {
            return Err(HostError::internal("fake copy missing module"));
        }
        let src_bytes = self
            .buffers
            .get(&src)
            .ok_or_else(|| HostError::internal("fake copy missing src"))?
            .clone();
        let out_buf = self
            .buffers
            .get_mut(&out)
            .ok_or_else(|| HostError::internal("fake copy missing out"))?;
        if src_bytes.len() != out_buf.len() {
            return Err(HostError::invalid_args("fake copy length mismatch"));
        }
        out_buf.copy_from_slice(&src_bytes);
        Ok(())
    }

    /// Simulate the tiled-matmul kernel for the U-03 host adapter unit tests:
    /// `out[ri * n + ci] = Σ_kk a[ri * k + kk] * b[kk * n + ci]` with the
    /// configured M·K / K·N / M·N shapes. Sequencing evidence only — never a
    /// real-device claim.
    fn simulate_matmul(
        &mut self,
        module: u64,
        a: u64,
        b: u64,
        out: u64,
        m: u64,
        k: u64,
        n: u64,
    ) -> HostResult<()> {
        if !self.modules.contains_key(&module) {
            return Err(HostError::internal("fake matmul missing module"));
        }
        let a_bytes = self
            .buffers
            .get(&a)
            .ok_or_else(|| HostError::internal("fake matmul missing a"))?
            .clone();
        let b_bytes = self
            .buffers
            .get(&b)
            .ok_or_else(|| HostError::internal("fake matmul missing b"))?
            .clone();
        let out_buf = self
            .buffers
            .get_mut(&out)
            .ok_or_else(|| HostError::internal("fake matmul missing out"))?;
        let (m, k, n) = (m as usize, k as usize, n as usize);
        let (a_len, b_len, out_len) = (m.checked_mul(k), k.checked_mul(n), m.checked_mul(n));
        let (Some(a_len), Some(b_len), Some(out_len)) = (a_len, b_len, out_len) else {
            return Err(HostError::internal("fake matmul plan dims overflow"));
        };
        if a_bytes.len() != a_len * 4 || b_bytes.len() != b_len * 4 || out_buf.len() != out_len * 4
        {
            return Err(HostError::invalid_args(
                "fake matmul buffer sizes contradict the M·K/K·N/M·N plan",
            ));
        }
        let a_values = f32_bytes_to_values(&a_bytes);
        let b_values = f32_bytes_to_values(&b_bytes);
        for ri in 0..m {
            for ci in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a_values[ri * k + kk] * b_values[kk * n + ci];
                }
                let offset = (ri * n + ci) * 4;
                out_buf[offset..offset + 4].copy_from_slice(&acc.to_le_bytes());
            }
        }
        Ok(())
    }
}

impl CudaDriver for FakeCudaDriver {
    fn discover(&mut self) -> HostResult<CudaEnvReport> {
        if self.force_unavailable {
            return Err(cuda_unavailable(
                "fake driver configured unavailable (sequencing test)",
            ));
        }
        Ok(CudaEnvReport {
            admitted: true,
            nvidia_smi: Some("fake-gpu".to_owned()),
            libcuda_candidates: vec!["fake://libcuda".to_owned()],
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
        // entry fails closed before dispatch (mirrors cuModuleGetFunction).
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
        // The emitted `addita` kernel takes exactly three buffers (a, b, out);
        // the simulated accumulation kernel takes two (a, acc) and adds a
        // into acc in place (G4). Anything else fails closed in the fake
        // just as it would on device.
        if let Some((m, k, n)) = self.matmul_simulation {
            if buffers.len() != 3 {
                return Err(HostError::invalid_args(
                    "fake launch_kernel matmul simulation requires exactly 3 buffers (a, b, out)",
                ));
            }
            return self.simulate_matmul(module, buffers[0], buffers[1], buffers[2], m, k, n);
        }
        if entry == b"accumulate" {
            if buffers.len() != 2 {
                return Err(HostError::invalid_args(
                    "fake launch_kernel 'accumulate' simulates the 2-buffer kernel (a, acc)",
                ));
            }
            return self.simulate_accumulate(module, buffers[0], buffers[1]);
        }
        if entry == b"observa" {
            if buffers.len() != 2 {
                return Err(HostError::invalid_args(
                    "fake launch_kernel 'observa' simulates the 2-buffer kernel (src, out)",
                ));
            }
            return self.simulate_copy(module, buffers[0], buffers[1]);
        }
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
            uploads: 0,
        }
    }
}

impl fmt::Debug for CudaHostSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaHostSession")
            .field("admitted", &self.admitted)
            .field("handles", &self.handles.len())
            .finish_non_exhaustive()
    }
}
