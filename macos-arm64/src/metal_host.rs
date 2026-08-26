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
use std::sync::Arc;

use faber::Valor;
use host_coordinator::device_identity::{DeviceHealthGeneration, DeviceOrdinal, PhysicalDeviceId};
use host_coordinator::discovery::{
    ComputeCapability, DeviceCapabilities, DeviceDiscoveryEntry, DeviceDiscoverySnapshot,
    DeviceHealth, DeviceMemory, DtypeSurface, ProbeProvenance,
};
#[cfg(target_os = "macos")]
use metal::{
    Buffer, CommandBuffer, CommandQueue, CompileOptions, ComputePassDescriptor,
    ComputePipelineState, CounterSampleBuffer, CounterSampleBufferDescriptor, Device,
    MTLCommandBufferStatus, MTLCounterSamplingPoint, MTLResourceOptions, MTLSize, MTLStorageMode,
    NSRange,
};
use serde::{Deserialize, Serialize};

use crate::device_descriptor::{errors, DeviceDataType};
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

/// Read-only mmap of a GGUF (or any) weight file.
///
/// The mapping is `PROT_READ` / `MAP_SHARED`. Pages stay lazy until a CPU
/// or GPU read; Metal no-copy buffers hold a clone of this handle so the
/// mapping outlives every admitted region.
#[derive(Clone, Debug)]
pub struct MappedWeightFile {
    inner: Arc<MappedWeightInner>,
}

#[derive(Debug)]
struct MappedWeightInner {
    ptr: *mut u8,
    /// Logical file length (GGUF bytes).
    file_len: usize,
    /// Kernel mapping length (`page_ceil(file_len)`). The last page past
    /// `file_len` is zero-filled.
    mapped_len: usize,
    page_size: usize,
}

unsafe impl Send for MappedWeightInner {}
unsafe impl Sync for MappedWeightInner {}

impl Drop for MappedWeightInner {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.mapped_len > 0 {
            unsafe {
                unix_mmap::munmap(self.ptr, self.mapped_len);
            }
        }
    }
}

impl MappedWeightFile {
    /// Map `path` read-only. Empty files fail closed.
    pub fn open(path: &Path) -> HostResult<Self> {
        unix_mmap::map_file(path)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.file_len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.file_len == 0
    }

    #[must_use]
    pub fn page_size(&self) -> usize {
        self.inner.page_size
    }

    #[must_use]
    pub fn mapped_len(&self) -> usize {
        self.inner.mapped_len
    }

    #[must_use]
    pub fn as_ptr(&self) -> *const u8 {
        self.inner.ptr as *const u8
    }

    /// File bytes. Does not touch pages until the caller reads them.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.inner.ptr, self.inner.file_len) }
    }

    /// True when `[ptr, ptr+len)` sits inside this mapping, including the
    /// last-page pad past `file_len`.
    #[must_use]
    pub fn contains(&self, ptr: *const u8, len: usize) -> bool {
        let start = self.inner.ptr as usize;
        let end = start.saturating_add(self.inner.mapped_len);
        let p = ptr as usize;
        let q = p.saturating_add(len);
        p >= start && q >= p && q <= end
    }
}

/// Host paging facts recorded on mmap admission (M5-U3 receipt).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct MappedWeightPaging {
    pub page_size: u64,
    pub mapped_len: u64,
    pub file_len: u64,
    /// Process resident bytes after the mmap syscall, before a GPU touch.
    pub rss_bytes: u64,
}

/// Current process resident size. Zero when the host cannot sample it.
#[must_use]
pub fn process_resident_bytes() -> u64 {
    unix_mmap::resident_bytes()
}

/// Host page size used for no-copy MTLBuffer rounding.
#[must_use]
pub fn mapped_page_size() -> usize {
    unix_mmap::page_size()
}

mod unix_mmap {
    use super::{HostError, HostResult, MappedWeightFile, MappedWeightInner};
    use std::fs::File;
    use std::path::Path;
    use std::sync::Arc;

    const PROT_READ: i32 = 1;
    const MAP_SHARED: i32 = 1;
    #[cfg(target_os = "macos")]
    const SC_PAGESIZE: i32 = 29;
    #[cfg(not(target_os = "macos"))]
    const SC_PAGESIZE: i32 = 30;

    extern "C" {
        fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
        pub(super) fn munmap(addr: *mut u8, len: usize) -> i32;
        fn sysconf(name: i32) -> i64;
        fn getrusage(who: i32, usage: *mut Rusage) -> i32;
    }

    #[repr(C)]
    struct TimeVal {
        tv_sec: i64,
        tv_usec: i32,
        _pad: i32,
    }

    #[repr(C)]
    struct Rusage {
        ru_utime: TimeVal,
        ru_stime: TimeVal,
        ru_maxrss: i64,
        _rest: [i64; 14],
    }

    pub(super) fn page_size() -> usize {
        let value = unsafe { sysconf(SC_PAGESIZE) };
        if value > 0 {
            value as usize
        } else {
            4096
        }
    }

    pub(super) fn resident_bytes() -> u64 {
        let mut usage = unsafe { std::mem::zeroed::<Rusage>() };
        if unsafe { getrusage(0, &mut usage) } != 0 {
            return 0;
        }
        let rss = usage.ru_maxrss;
        if rss <= 0 {
            return 0;
        }
        // Darwin documents ru_maxrss in bytes; Linux uses kilobytes.
        #[cfg(target_os = "macos")]
        {
            rss as u64
        }
        #[cfg(not(target_os = "macos"))]
        {
            (rss as u64).saturating_mul(1024)
        }
    }

    pub(super) fn map_file(path: &Path) -> HostResult<MappedWeightFile> {
        let file = File::open(path).map_err(|error| {
            HostError::invalid_args(format!(
                "Metal mmap failed to open {}: {error}",
                path.display()
            ))
        })?;
        let file_len = file
            .metadata()
            .map_err(|error| {
                HostError::invalid_args(format!(
                    "Metal mmap failed to stat {}: {error}",
                    path.display()
                ))
            })?
            .len();
        let file_len = usize::try_from(file_len).map_err(|_| {
            HostError::invalid_args(format!(
                "Metal mmap file {} is larger than the host address space",
                path.display()
            ))
        })?;
        if file_len == 0 {
            return Err(HostError::invalid_args(format!(
                "Metal mmap file {} is empty",
                path.display()
            )));
        }
        let page = page_size();
        let mapped_len = file_len.div_ceil(page).saturating_mul(page).max(page);
        #[cfg(unix)]
        let fd = {
            use std::os::unix::io::AsRawFd;
            file.as_raw_fd()
        };
        #[cfg(not(unix))]
        let fd: i32 = -1;
        let ptr = unsafe { mmap(std::ptr::null_mut(), file_len, PROT_READ, MAP_SHARED, fd, 0) };
        if ptr.is_null() || ptr == !0usize as *mut u8 {
            return Err(HostError::internal(format!(
                "Metal mmap of {} ({file_len} bytes) failed",
                path.display()
            )));
        }
        drop(file);
        Ok(MappedWeightFile {
            inner: Arc::new(MappedWeightInner {
                ptr,
                file_len,
                mapped_len,
                page_size: page,
            }),
        })
    }
}

/// Default kernel entry for the legacy `launch_elementwise_add_f32` session
/// path. Matches the emitted `add_one` entry of the U2 proof fixture
/// (input@0, output@1, extent@2).
const ELEMENTWISE_ADD_ENTRY: &[u8] = b"add_one";
/// Timestamp sample slots (two per encoder: start + end). 2048 covers the
/// dense 419/315 census with headroom.
const TIMESTAMP_SAMPLE_CAPACITY: u64 = 2048;

fn per_op_timing_wanted() -> bool {
    matches!(std::env::var("FABER_SPAWN_TIMING").as_deref(), Ok("1"))
        || matches!(std::env::var("FABER_PER_OP_TIMING").as_deref(), Ok("1"))
}

/// Read-only environment admission report (never a product run claim).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetalEnvReport {
    pub admitted: bool,
    /// Present when at least one Metal device is enumerated. M1 has no
    /// binding, so this carries a fixed marker string; M2 fills the real
    /// device name from the binding.
    pub mtl_device: Option<String>,
    /// Metal framework paths found on this machine. Informational only —
    /// admission is driven by the device probe, not by framework presence.
    pub metal_framework_paths: Vec<String>,
    pub reason: String,
}

/// One enumerated Metal physical device. The ordinal is a locator only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetalPhysicalDevice {
    /// Enumeration-order locator — never identity.
    pub ordinal: u32,
    /// `MTLDevice.registryID` as a decimal string.
    pub registry_id: String,
    /// Device name from the Metal binding.
    pub device_model: Option<String>,
    /// `recommendedMaxWorkingSetSize` bytes (unified memory has no separate
    /// nvidia-smi-style tool total).
    pub api_total_bytes: u64,
    /// `MTLDevice.maxThreadsPerThreadgroup` width — the device-level
    /// threadgroup ceiling (the launch-time reject stays on the pipeline).
    pub max_threads_per_workgroup: u32,
    /// `MTLDevice.maxThreadgroupMemoryLength` (Metal has no opt-in tier).
    pub workgroup_shared_memory_min_bytes: u32,
    /// Same as min: Metal exposes one threadgroup-memory ceiling.
    pub workgroup_shared_memory_max_bytes: u32,
    /// Apple GPU simdgroup width (no device-level query; 32 on Apple Silicon).
    pub collective_width: u32,
    /// `MTLDevice.hasUnifiedMemory`.
    pub unified_memory: bool,
}

impl MetalPhysicalDevice {
    /// Convert this probe row into a host-coordinator discovery entry.
    #[must_use]
    pub fn to_discovery_entry(&self) -> DeviceDiscoveryEntry {
        DeviceDiscoveryEntry {
            ordinal: DeviceOrdinal::new(self.ordinal),
            identity: PhysicalDeviceId::metal(self.registry_id.clone()),
            device_model: self.device_model.clone(),
            capabilities: DeviceCapabilities {
                compute_capability: ComputeCapability { major: 0, minor: 0 },
                sm_count: 0,
                dtype_surface: DtypeSurface::empty(),
                max_threads_per_workgroup: self.max_threads_per_workgroup,
                workgroup_shared_memory_min_bytes: self.workgroup_shared_memory_min_bytes,
                workgroup_shared_memory_max_bytes: self.workgroup_shared_memory_max_bytes,
                collective_width: self.collective_width,
                unified_memory: self.unified_memory,
            },
            memory: DeviceMemory {
                tool_report_total_mib: None,
                api_total_bytes: self.api_total_bytes,
            },
            health: DeviceHealth::Healthy,
            health_generation: DeviceHealthGeneration::initial(),
            probe_provenance: ProbeProvenance {
                probe: "MTLCopyAllDevices".to_owned(),
                tool_versions: "Metal framework".to_owned(),
            },
        }
    }
}

/// Enumerate every locally attached Metal device into identity/memory facts.
///
/// Returns an empty list when this OS is not macOS or no Metal device is
/// attached. Never a product-run claim.
pub fn enumerate_metal_physical_devices() -> HostResult<Vec<MetalPhysicalDevice>> {
    Ok(metal_devices_from_system())
}

/// Timestamped discovery snapshot of every locally attached Metal device.
pub fn discover_metal_snapshot(probe_utc_nanos: u64) -> HostResult<DeviceDiscoverySnapshot> {
    let devices = enumerate_metal_physical_devices()?;
    Ok(DeviceDiscoverySnapshot::from_enumerated(
        probe_utc_nanos,
        devices.iter().map(MetalPhysicalDevice::to_discovery_entry),
    ))
}

/// Apple GPU simdgroup width. Metal has no device-level query analogous to
/// CUDA `CU_DEVICE_ATTRIBUTE_WARP_SIZE`; simdgroups are 32 threads.
#[cfg(target_os = "macos")]
const METAL_COLLECTIVE_WIDTH: u32 = 32;

#[cfg(target_os = "macos")]
fn mtl_nsuinteger_as_u32(value: metal::NSUInteger) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

fn metal_devices_from_system() -> Vec<MetalPhysicalDevice> {
    #[cfg(not(target_os = "macos"))]
    {
        Vec::new()
    }
    #[cfg(target_os = "macos")]
    {
        Device::all()
            .into_iter()
            .enumerate()
            .map(|(index, device)| {
                let threadgroup = device.max_threads_per_threadgroup();
                let shared = mtl_nsuinteger_as_u32(device.max_threadgroup_memory_length());
                MetalPhysicalDevice {
                    ordinal: index as u32,
                    registry_id: device.registry_id().to_string(),
                    device_model: {
                        let name = device.name();
                        if name.is_empty() {
                            None
                        } else {
                            Some(name.to_owned())
                        }
                    },
                    api_total_bytes: device.recommended_max_working_set_size(),
                    max_threads_per_workgroup: mtl_nsuinteger_as_u32(threadgroup.width),
                    workgroup_shared_memory_min_bytes: shared,
                    workgroup_shared_memory_max_bytes: shared,
                    collective_width: METAL_COLLECTIVE_WIDTH,
                    unified_memory: device.has_unified_memory(),
                }
            })
            .collect()
    }
}

/// Opaque host-owned handle identity carried at the Frame control boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MetalHandleId(pub u64);

/// One Metal launch binding. Frozen B4 shape minus the typed runtime source
/// (that tag is host-level): handle, binding index, byte offset, view span.
/// `set_buffer` uses `region_offset + byte_offset`; the offset must add to
/// the mmap page remainder, never replace it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetalLaunchBinding {
    /// Live Metal buffer for this binding.
    pub handle: MetalHandleId,
    /// Declared binding index. Never dropped before launch.
    pub binding_index: u32,
    /// Byte offset into the allocation (static envelope).
    pub byte_offset: u64,
    /// View span in bytes for this binding.
    pub view_span: u64,
}

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
    /// Generalized kernel launch: resolve `entry` inside `module` and encode
    /// a dispatch over the given device buffers (binding order: inputs first,
    /// output last) with the given 3D grid and 3D block shape. Encoding is
    /// mid-step only — the driver does not commit or wait here. The session
    /// (or an explicit `sync` / readback) commits the pending command buffer
    /// once at the step boundary. The system driver routes the legacy
    /// elementwise-add path through this so there is exactly one encode site.
    /// Offset-zero caller path: each buffer binds at its mmap page remainder
    /// (0 for ordinary allocs).
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
        let zeros = vec![0u64; buffers.len()];
        self.launch_kernel_bound(
            module, entry, buffers, &zeros, &zeros, grid_x, grid_y, grid_z, block_x, block_y,
            block_z,
        )
    }
    /// Bound launch: `byte_offsets` are B4 launch-binding offsets, added to
    /// each buffer's mmap page remainder at `set_buffer`. `view_spans` bound
    /// the composed range; a zero span skips the extra span check (legacy).
    #[allow(clippy::too_many_arguments)]
    fn launch_kernel_bound(
        &mut self,
        module: u64,
        entry: &[u8],
        buffers: &[u64],
        byte_offsets: &[u64],
        view_spans: &[u64],
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
    ) -> HostResult<()>;
    /// Commit the pending step command buffer (if any) and wait once.
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
    /// Command buffers actually submitted. Encode-only `launch_kernel`
    /// calls do not increment this; `sync` / readback flush increment it
    /// once per committed buffer.
    fn command_submit_count(&self) -> usize {
        0
    }
    /// Blocking waits actually performed (`wait_until_completed` or the fake
    /// equivalent). Mid-step encodes do not wait.
    fn blocking_wait_count(&self) -> usize {
        0
    }
    /// Per-encoder GPU timestamps from the last committed step, in µs.
    /// Empty when the step did not sample timestamps.
    fn take_encoder_gpu_us(&mut self) -> Vec<u64> {
        Vec::new()
    }
    /// Per-encoder GPU start times from the last committed step, in µs
    /// relative to the first encoder start. Empty when unsampled.
    fn take_encoder_gpu_start_us(&mut self) -> Vec<u64> {
        Vec::new()
    }
    /// Keep a read-only weight mapping alive for no-copy MTLBuffer admission.
    /// Fake drivers retain the mapping so `copy_in` can take the wrap branch
    /// (they still memcpy the page into simulated storage).
    fn retain_mapped_file(&mut self, _file: MappedWeightFile) {}
    /// Times `copy_in` admitted a slice by wrapping a retained mmap.
    fn mmap_wrap_count(&self) -> usize {
        0
    }
}

/// Product-facing session: opaque handles + ordered lifecycle.
pub struct MetalHostSession {
    driver: Box<dyn MetalDriver>,
    handles: HandleRegistry<MetalHandleKind>,
    admitted: bool,
    /// HostProvided once-init copies issued through this session.
    uploads: usize,
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
                uploads: 0,
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

    /// Command buffers submitted since the session opened (W8-U1).
    #[must_use]
    pub fn command_submit_count(&self) -> usize {
        self.driver.command_submit_count()
    }

    /// Blocking waits since the session opened (W8-U1).
    #[must_use]
    pub fn blocking_wait_count(&self) -> usize {
        self.driver.blocking_wait_count()
    }

    /// Per-encoder GPU timestamps from the last committed step (µs).
    pub fn take_encoder_gpu_us(&mut self) -> Vec<u64> {
        self.driver.take_encoder_gpu_us()
    }

    /// Per-encoder GPU start times from the last committed step (µs).
    pub fn take_encoder_gpu_start_us(&mut self) -> Vec<u64> {
        self.driver.take_encoder_gpu_start_us()
    }

    /// Times this session admitted a `copy_in` by wrapping a retained mmap.
    #[must_use]
    pub fn mmap_wrap_count(&self) -> usize {
        self.driver.mmap_wrap_count()
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
        self.copy_in_bytes(buffer, f32_slice_as_bytes(values), DeviceDataType::F32)
    }

    /// Keep a GGUF mmap alive for the session so byte-region `copy_in` can
    /// wrap the mapped pages instead of uploading them.
    pub fn retain_mapped_file(&mut self, file: MappedWeightFile) -> HostResult<()> {
        self.require_admitted()?;
        self.driver.retain_mapped_file(file);
        Ok(())
    }

    /// Admit a native byte region into a Metal buffer.
    ///
    /// Bytes that sit inside a retained mmap become a no-copy MTLBuffer
    /// (page-rounded wrap; bind offset is the intra-page remainder). Other
    /// bytes copy as-is. A 1–3 byte tail may be zero-padded so the buffer's
    /// f32-word width matches; a shorter logical-F32 expansion is not
    /// admitted.
    pub fn copy_in_bytes(
        &mut self,
        buffer: MetalHandleId,
        bytes: &[u8],
        _dtype: DeviceDataType,
    ) -> HostResult<()> {
        self.require_admitted()?;
        let (token, len_bytes) = self.buffer_token(buffer)?;
        if bytes.len() == len_bytes {
            return self.driver.copy_in(token, bytes);
        }
        if bytes.len() > len_bytes || len_bytes - bytes.len() >= 4 {
            return Err(HostError::invalid_args(format!(
                "packed-region size mismatch: buffer {len_bytes} bytes, got {}",
                bytes.len()
            )));
        }
        // 1–3 byte short: mmap-backed slices wrap without padding so the
        // pointer stays inside the mapping. Owned slices still pad.
        match self.driver.copy_in(token, bytes) {
            Ok(()) => Ok(()),
            Err(_) => {
                let mut padded = vec![0u8; len_bytes];
                padded[..bytes.len()].copy_from_slice(bytes);
                self.driver.copy_in(token, &padded)
            }
        }
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

    /// Generalized launch: resolve `entry` inside `module` and encode a
    /// dispatch over `buffers` (inputs first, output last) with the given
    /// grid/block shape. Every buffer handle is validated and resolved to a
    /// backend token before the driver is touched. Encoding does not commit
    /// or wait — `sync` (or the next readback) commits the pending command
    /// buffer once. This helper preserves the original 1D session surface
    /// for elementwise callers.
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
    /// `launch_kernel`. Encodes only; call [`Self::sync`] to commit+wait.
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
        )
    }

    /// Launch with explicit B4 bindings. Binding indices select dispatch
    /// slots and are not dropped. Each `byte_offset` is added to the
    /// buffer's mmap page remainder at `set_buffer`; it does not replace it.
    /// Encoding does not copy cache data or allocate a per-kernel temp.
    pub fn launch_kernel_bound(
        &mut self,
        module: MetalHandleId,
        entry: &str,
        bindings: &[MetalLaunchBinding],
        grid: [u32; 3],
        block: [u32; 3],
    ) -> HostResult<()> {
        self.require_admitted()?;
        let module_token = self.module_token(module)?;
        if entry.is_empty() {
            return Err(HostError::invalid_args("Metal kernel entry name is empty"));
        }
        let ordered = ordered_launch_bindings(bindings)?;
        let mut tokens = Vec::with_capacity(ordered.len());
        let mut offsets = Vec::with_capacity(ordered.len());
        let mut spans = Vec::with_capacity(ordered.len());
        for binding in &ordered {
            let (token, len_bytes) = self.buffer_token(binding.handle)?;
            if binding.view_span == 0 {
                return Err(errors::descriptor(format!(
                    "launch binding index {} has a zero view span",
                    binding.binding_index
                )));
            }
            let Some(end) = binding.byte_offset.checked_add(binding.view_span) else {
                return Err(errors::shape_mismatch(format!(
                    "launch binding index {} overflows its static envelope",
                    binding.binding_index
                )));
            };
            if end > len_bytes as u64 {
                return Err(errors::shape_mismatch(format!(
                    "launch binding index {} spans {} bytes from offset {} but the allocation is {len_bytes} bytes",
                    binding.binding_index, binding.view_span, binding.byte_offset
                )));
            }
            tokens.push(token);
            offsets.push(binding.byte_offset);
            spans.push(binding.view_span);
        }
        self.driver.launch_kernel_bound(
            module_token,
            entry.as_bytes(),
            &tokens,
            &offsets,
            &spans,
            grid[0],
            grid[1],
            grid[2],
            block[0],
            block[1],
            block[2],
        )
    }

    /// Commit the pending step command buffer and wait once. A no-op when
    /// nothing is pending (already flushed, or no encodes this step).
    pub fn sync(&mut self) -> HostResult<()> {
        self.require_admitted()?;
        self.driver.sync()
    }

    pub fn readback_bytes(
        &mut self,
        buffer: MetalHandleId,
        _dtype: DeviceDataType,
    ) -> HostResult<Vec<u8>> {
        self.require_admitted()?;
        let (token, len_bytes) = self.buffer_token(buffer)?;
        let bytes = self.driver.copy_out(token, len_bytes)?;
        if bytes.len() != len_bytes {
            return Err(HostError::internal(
                "Metal readback returned unexpected byte length",
            ));
        }
        Ok(bytes)
    }

    pub fn readback_f32(&mut self, buffer: MetalHandleId) -> HostResult<Vec<f32>> {
        let bytes = self.readback_bytes(buffer, DeviceDataType::F32)?;
        if bytes.len() % 4 != 0 {
            return Err(HostError::internal(
                "Metal readback returned unexpected f32 byte length",
            ));
        }
        Ok((0..bytes.len() / 4)
            .map(|index| read_f32_at(&bytes, index))
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

/// Read the f32 stored little-endian at element `index` of a byte-laid-out
/// buffer (one f32 per 4 bytes, matching the emitted kernels' buffer layout).
fn read_f32_at(bytes: &[u8], index: usize) -> f32 {
    let offset = index * 4;
    f32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

/// Write an f32 little-endian at element `index` of a byte-laid-out buffer.
fn write_f32_at(bytes: &mut [u8], index: usize, value: f32) {
    let offset = index * 4;
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
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

/// Typed fail-closed error for a launch naming an entry the loaded module
/// does not declare. Shared by the real driver's per-entry pipeline map and
/// the fake driver's declared-function-table check so both lanes report the
/// same stable code and message (mirrors `cuModuleGetFunction` on the CUDA
/// lane).
fn metal_entry_mismatch(entry_name: &str) -> HostError {
    HostError {
        code: crate::device_descriptor::E_DEVICE_ENTRY_MISMATCH.to_owned(),
        message: format!("launch: module has no entry named {entry_name}"),
        retryable: false,
    }
}

/// Fail closed unless a fake per-entry dispatch carries exactly `count`
/// buffers. The simulated kernels have fixed binding orders (mirroring the
/// real driver's fixed binding order); a mismatched arity is a harness bug,
/// not a device failure.
fn expect_fake_arity(buffers: &[u64], count: usize, what: &str) -> HostResult<()> {
    if buffers.len() != count {
        return Err(HostError::invalid_args(format!(
            "fake launch_kernel {what}"
        )));
    }
    Ok(())
}

/// Bindings placed at their declared indices. Gaps and duplicates fail
/// closed so an index cannot be validated and then dropped.
fn ordered_launch_bindings(bindings: &[MetalLaunchBinding]) -> HostResult<Vec<MetalLaunchBinding>> {
    let mut slots: Vec<Option<MetalLaunchBinding>> = vec![None; bindings.len()];
    for binding in bindings {
        let index = binding.binding_index as usize;
        if index >= slots.len() {
            return Err(errors::abi_mismatch(format!(
                "launch binding index {} is outside 0..{}",
                binding.binding_index,
                bindings.len()
            )));
        }
        if slots[index].is_some() {
            return Err(errors::abi_mismatch(format!(
                "launch binding index {} is declared twice",
                binding.binding_index
            )));
        }
        slots[index] = Some(*binding);
    }
    slots
        .into_iter()
        .enumerate()
        .map(|(index, binding)| {
            binding.ok_or_else(|| {
                errors::abi_mismatch(format!(
                    "launch binding index {index} was validated then dropped before launch"
                ))
            })
        })
        .collect()
}

/// Compose the mmap page remainder with a B4 launch-binding offset.
/// The binding offset adds to the remainder; it must not replace it.
fn compose_bind_offset(
    region_offset: u64,
    binding_offset: u64,
    view_span: u64,
    buffer_len: u64,
) -> HostResult<u64> {
    let composed = region_offset.checked_add(binding_offset).ok_or_else(|| {
        errors::shape_mismatch(
            "launch binding offset overflows when added to the mmap page remainder",
        )
    })?;
    let end = if view_span == 0 {
        composed
    } else {
        composed.checked_add(view_span).ok_or_else(|| {
            errors::shape_mismatch("launch binding span overflows its static envelope")
        })?
    };
    if composed > buffer_len || end > buffer_len {
        return Err(errors::shape_mismatch(format!(
            "launch binding composed offset {composed} span {view_span} exceeds buffer length {buffer_len} (region remainder {region_offset})"
        )));
    }
    Ok(composed)
}

// Metal device probe. Enumeration uses `MTLCopyAllDevices` (`Device::all`);
// the probe fills the real device name from the first enumerated device (M2).

/// Probe this machine for a loadable Metal device stack without claiming a run.
pub fn probe_metal_environment() -> MetalEnvReport {
    let metal_framework_paths = detect_metal_framework_paths();

    let enumerated = metal_devices_from_system();
    let mtl_device = enumerated
        .first()
        .and_then(|device| device.device_model.clone());

    let admitted = !enumerated.is_empty();
    let reason = if admitted {
        format!(
            "Metal enumerated {} device(s); SystemMetalDriver can compile MSL and launch add_one",
            enumerated.len()
        )
    } else {
        "no Metal device enumerated (MTLCopyAllDevices was empty, or this OS is not macOS); Metal product execution is not admitted".to_owned()
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

/// Sequencing-only fake storage: owned bytes plus the mmap page remainder
/// used at bind time (0 for ordinary allocs).
struct FakeBufferSlot {
    bytes: Vec<u8>,
    region_offset: u64,
    /// Original allocation length. mmap wrap replaces `bytes` with the
    /// page-rounded mapping; kernels still see this many bytes from the
    /// remainder, matching the session handle's `len_bytes`.
    logical_len: usize,
}

/// Sequencing-only fake driver for unit tests. Not product Metal evidence.
#[derive(Default)]
pub struct FakeMetalDriver {
    next_token: u64,
    buffers: BTreeMap<u64, FakeBufferSlot>,
    maps: Vec<MappedWeightFile>,
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
    /// Encodes waiting for the next `sync` / readback flush.
    pending_encodes: usize,
    /// Command buffers submitted (one per flush of a non-empty encode batch).
    command_submits: usize,
    /// Blocking waits performed (one per flush of a non-empty encode batch).
    blocking_waits: usize,
    /// Times `copy_in` took the retained-mmap wrap branch.
    mmap_wraps: usize,
    /// Optional simulated fused QKV carrier launch (RoPE-pairing probe
    /// tests): mirrors the compiled carrier's observable decode contract.
    qkv_carrier: Option<FakeQkvCarrierSim>,
}

/// Simulated fused QKV carrier (`*_QkvProjection`) for the hosts-side
/// RoPE-pairing probe tests. This mirrors the compiled carrier's observable
/// contract — NOT its numerics: the fail-early cursor guard runs first (the
/// `DECODE_STATUS_INVALID_CURSOR` status word lands in Q[0] and NO Q is
/// written on rejection), and an admitted decode publishes a dense-f32 Q
/// through the library body under the baked rotate-half pairing. Binding
/// indices address the launch's buffer list in slot order.
#[derive(Debug, Clone)]
pub struct FakeQkvCarrierSim {
    /// Entry name this simulation serves.
    pub entry: String,
    /// Launch-buffer index of the activation input.
    pub activation: usize,
    /// Launch-buffer index of the dense column-major Q weight.
    pub q_weight: usize,
    /// Launch-buffer indices of the dense K/V weights (unused for Q).
    pub k_weight: usize,
    pub v_weight: usize,
    /// Launch-buffer index of the optional Q/K/V bias (all-or-none).
    pub q_bias: Option<usize>,
    /// Launch-buffer indices of the RoPE cosine/sine tables.
    pub cos: usize,
    pub sin: usize,
    /// Launch-buffer index of the four-field invocation-state cursor.
    pub cursor: usize,
    /// Launch-buffer index of the Q output.
    pub q_out: usize,
    /// The carrier's baked query-row constant — what its guard compares the
    /// cursor's `query_rows` against (a value other than the cursor's
    /// exercises the guard-trip / stale-reference path).
    pub baked_query_rows: u32,
    /// Bind geometry (rows, hidden, kv_heads, q_per_kv, head_dim).
    pub rows: u64,
    pub hidden: u64,
    pub kv_heads: u64,
    pub q_per_kv: u64,
    pub head_dim: u64,
    /// The baked rotate-half pairing the probe is meant to discover.
    pub rotate_half: bool,
    /// Constant added to every published Q element (forces a both-candidates
    /// mismatch to exercise the diagnostic error message).
    pub skew: f32,
}

impl FakeMetalDriver {
    /// Emitted GEA2 entries and their declared device-buffer arities. The
    /// table mirrors the 15-entry bundle ABI: transpose and causal_softmax
    /// consume two buffers, score_gemm consumes four, and the remaining
    /// entries consume three — except the three variant rows: the gathered
    /// o-projection is its own entry `gemm_qo_gathered` (the block's launch
    /// 57, radix `6a0e3780a`) binding 17 (15 context reads + o_weight +
    /// o_projection); the rope launches, whose per-instance window
    /// repair (radix `5f96ed340`)
    /// declares the packed output plus every per-head window as a write:
    /// `rope_q` binds 18 (input + table + packed output + 15 `q_head_<h>`)
    /// and `rope_k` binds 8 (input + table + packed output + 5 `k_head_<g>`);
    /// and the windowed v-projection `gemm_kv_windows` (GEA2-U5g) binds 8
    /// (input + weight + the packed [8,320] output + 5 `v_head_<kv>`).
    /// Each entry admits exactly its declared binding shapes, never a
    /// blanket range; any other arity fails closed. These launches are
    /// structural only; they do not pretend to simulate the kernel numerics.
    const GEA2_ENTRY_ARITIES: &[(&str, &[usize])] = &[
        ("rmsnorm", &[3]),
        ("gemm_qo", &[3]),
        ("gemm_qo_gathered", &[17]),
        ("gemm_kv", &[3]),
        ("gemm_kv_windows", &[8]),
        ("gemm_gate_up", &[3]),
        ("gemm_down", &[3]),
        ("rope_q", &[18]),
        ("rope_k", &[8]),
        ("transpose", &[2]),
        ("score_gemm", &[4]),
        ("causal_softmax", &[2]),
        ("context_gemm", &[3]),
        ("swiglu", &[3]),
        ("residual_add", &[3]),
    ];

    fn gea2_entry_arities(entry: &[u8]) -> Option<&'static [usize]> {
        Self::GEA2_ENTRY_ARITIES
            .iter()
            .find_map(|(name, arities)| (name.as_bytes() == entry).then_some(*arities))
    }

    /// GEA3's plan carries the full entry identity and its ordered bindings.
    /// These arities are structural fake-driver admission only; the fake does
    /// not read values or simulate any GEA3 kernel body.
    const GEA3_ENTRY_ARITIES: &[(&str, &[usize])] = &[
        ("embedding_gather", &[4]),
        ("prefill_embedding_gather", &[4]),
        ("decode_rmsnorm", &[5]),
        ("decode_gemv_qo", &[5]),
        ("decode_gemv_kv", &[5]),
        ("decode_gemv_gate_up", &[5]),
        ("decode_gemv_down", &[5]),
        ("decode_rope_q", &[4]),
        ("decode_rope_k", &[4]),
        // GEA3-U6 num-1: the mutators launch with the expanded-plan ABI
        // (history, slot/block, row, chain extra, output) — 5 bindings.
        // GEA3-U6 num-2: the mutators emit through the launch-variant
        // seam under the canonical read ABI — the inert chain read is gone.
        ("kv_append_k", &[4]),
        ("kv_append_v", &[4]),
        // GEA3-U6 num-3: the key transposes ride the launch-variant seam
        // (their strided per-head read windows must reach the emitted
        // body), which drops the inert chain read — 2 canonical bindings.
        ("decode_key_transpose", &[2]),
        ("decode_score_gemm", &[4]),
        ("decode_masked_softmax", &[4]),
        ("decode_context_gemm", &[3]),
        ("decode_swiglu", &[4]),
        ("decode_residual_add", &[4]),
        ("prefill_rmsnorm", &[5]),
        ("prefill_gemm_qo", &[5]),
        // GEA3-U6 num-10: the chunked o-projection member rides the
        // launch-variant seam (the inert chain read is dropped) — 4
        // bindings: chunked concat read, weights, inert weight extra,
        // output.
        ("prefill_gemm_o", &[4]),
        ("prefill_gemm_kv", &[5]),
        ("prefill_gemm_gate_up", &[5]),
        ("prefill_gemm_down", &[5]),
        ("prefill_rope_q", &[4]),
        ("prefill_rope_k", &[4]),
        // GEA3-U6 num-3: the key transposes ride the launch-variant seam
        // (2 canonical bindings; see decode_key_transpose above).
        ("prefill_key_transpose", &[2]),
        ("prefill_score_gemm", &[4]),
        // GEA3-U6 num-9: +1 binding — the additive causal_mask operand.
        ("prefill_causal_softmax", &[4]),
        ("prefill_context_gemm", &[3]),
        ("prefill_swiglu", &[4]),
        ("prefill_residual_add", &[4]),
        ("prefill_kv_write_k", &[4]),
        ("prefill_kv_write_v", &[4]),
        ("head_rmsnorm", &[5]),
        ("lm_head_gemv", &[5]),
        ("prefill_head_rmsnorm", &[5]),
        ("prefill_lm_head_gemv", &[5]),
    ];

    fn gea3_entry_arities(entry: &[u8]) -> Option<&'static [usize]> {
        Self::GEA3_ENTRY_ARITIES
            .iter()
            .find_map(|(name, arities)| (name.as_bytes() == entry).then_some(*arities))
    }

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

    /// Install a simulated fused QKV carrier launch (RoPE-pairing probe
    /// tests). The entry must also be declared via
    /// [`FakeMetalDriver::with_known_entry`].
    pub fn with_qkv_carrier_sim(mut self, sim: FakeQkvCarrierSim) -> Self {
        self.qkv_carrier = Some(sim);
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

    fn slot(&self, token: u64, what: &str) -> HostResult<&FakeBufferSlot> {
        self.buffers
            .get(&token)
            .ok_or_else(|| HostError::internal(format!("fake {what} missing buffer")))
    }

    fn view_from(&self, token: u64, binding_offset: u64, what: &str) -> HostResult<&[u8]> {
        let slot = self.slot(token, what)?;
        let start = compose_bind_offset(
            slot.region_offset,
            binding_offset,
            0,
            slot.bytes.len() as u64,
        )? as usize;
        let end = (slot.region_offset as usize)
            .saturating_add(slot.logical_len)
            .min(slot.bytes.len())
            .max(start);
        Ok(&slot.bytes[start..end])
    }

    /// Simulate the elementwise-add kernel: `out[i] = a[i] + b[i]`. Shared by
    /// the legacy elementwise-add path and the generalized `launch_kernel`
    /// (the emitted kernel is the same add shape), mirroring the CUDA fake.
    fn simulate_elementwise_add(
        &mut self,
        module: u64,
        a: u64,
        a_off: u64,
        b: u64,
        b_off: u64,
        out: u64,
        out_off: u64,
    ) -> HostResult<()> {
        if !self.modules.contains_key(&module) {
            return Err(HostError::internal("fake launch missing module"));
        }
        let a_bytes = self.view_from(a, a_off, "launch a")?.to_vec();
        let b_bytes = self.view_from(b, b_off, "launch b")?.to_vec();
        let out_start = {
            let slot = self.slot(out, "launch out")?;
            compose_bind_offset(slot.region_offset, out_off, 0, slot.bytes.len() as u64)? as usize
        };
        let out_buf = self
            .buffers
            .get_mut(&out)
            .ok_or_else(|| HostError::internal("fake launch missing out"))?;
        let out_view = &mut out_buf.bytes[out_start..];
        if a_bytes.len() != b_bytes.len() || a_bytes.len() != out_view.len() {
            return Err(HostError::invalid_args("fake launch length mismatch"));
        }
        let len = a_bytes.len() / 4;
        for i in 0..len {
            let ai = read_f32_at(&a_bytes, i);
            let bi = read_f32_at(&b_bytes, i);
            write_f32_at(out_view, i, ai + bi);
        }
        Ok(())
    }

    /// Simulate the accumulation kernel (G4): `acc[i] += a[i]` — the
    /// emitted elementwise accumulation into a persistent buffer. Repeated
    /// launches accumulate onto the previous device contents (the host's
    /// ZeroFill initialization defines the first state exactly once).
    fn simulate_accumulate(
        &mut self,
        module: u64,
        a: u64,
        a_off: u64,
        acc: u64,
        acc_off: u64,
    ) -> HostResult<()> {
        if !self.modules.contains_key(&module) {
            return Err(HostError::internal("fake launch missing module"));
        }
        let a_bytes = self.view_from(a, a_off, "accumulate a")?.to_vec();
        let acc_start = {
            let slot = self.slot(acc, "accumulate acc")?;
            compose_bind_offset(slot.region_offset, acc_off, 0, slot.bytes.len() as u64)? as usize
        };
        let acc_buf = self
            .buffers
            .get_mut(&acc)
            .ok_or_else(|| HostError::internal("fake accumulate missing acc"))?;
        let acc_view = &mut acc_buf.bytes[acc_start..];
        if a_bytes.len() != acc_view.len() {
            return Err(HostError::invalid_args("fake accumulate length mismatch"));
        }
        let len = a_bytes.len() / 4;
        for i in 0..len {
            let ai = read_f32_at(&a_bytes, i);
            let acc_i = read_f32_at(acc_view, i);
            write_f32_at(acc_view, i, acc_i + ai);
        }
        Ok(())
    }

    /// Simulate a copy kernel: `out[i] = src[i]` — the observation kernel
    /// that reads a persistent allocation (or a bound view of it) into a
    /// readback slot. Dest remaining storage is the write width so a smaller
    /// dest can sample a row of a larger source at a nonzero offset.
    fn simulate_copy(
        &mut self,
        module: u64,
        src: u64,
        src_off: u64,
        out: u64,
        out_off: u64,
    ) -> HostResult<()> {
        if !self.modules.contains_key(&module) {
            return Err(HostError::internal("fake copy missing module"));
        }
        let src_bytes = self.view_from(src, src_off, "copy src")?.to_vec();
        let out_start = {
            let slot = self.slot(out, "copy out")?;
            compose_bind_offset(slot.region_offset, out_off, 0, slot.bytes.len() as u64)? as usize
        };
        let out_buf = self
            .buffers
            .get_mut(&out)
            .ok_or_else(|| HostError::internal("fake copy missing out"))?;
        let out_view = &mut out_buf.bytes[out_start..];
        if src_bytes.len() < out_view.len() {
            return Err(HostError::invalid_args("fake copy length mismatch"));
        }
        out_view.copy_from_slice(&src_bytes[..out_view.len()]);
        Ok(())
    }

    /// Read one launch binding's whole slot as f32 values.
    fn read_slot_f32(&self, token: u64, what: &str) -> HostResult<Vec<f32>> {
        let bytes = self.view_from(token, 0, what)?;
        if bytes.len() % 4 != 0 {
            return Err(HostError::invalid_args(format!(
                "fake {what} is not f32 aligned"
            )));
        }
        Ok((0..bytes.len() / 4)
            .map(|index| read_f32_at(bytes, index))
            .collect())
    }

    /// Overwrite one launch binding's whole slot with f32 values.
    fn write_slot_f32(&mut self, token: u64, values: &[f32], what: &str) -> HostResult<()> {
        let start = {
            let slot = self.slot(token, what)?;
            compose_bind_offset(slot.region_offset, 0, 0, slot.bytes.len() as u64)? as usize
        };
        let end = start
            .checked_add(values.len() * 4)
            .ok_or_else(|| HostError::invalid_args("fake write overflows"))?;
        let buffer = self
            .buffers
            .get_mut(&token)
            .ok_or_else(|| HostError::internal(format!("fake {what} missing buffer")))?;
        if end > buffer.bytes.len() {
            return Err(HostError::invalid_args(format!(
                "fake {what} write of {} f32 values exceeds the slot",
                values.len()
            )));
        }
        for (index, value) in values.iter().enumerate() {
            write_f32_at(&mut buffer.bytes[start..], index, *value);
        }
        Ok(())
    }

    /// Simulate the fused QKV carrier launch: the observable decode contract
    /// of the compiled `*_QkvProjection` entry. The fail-early cursor guard
    /// runs first — on rejection the `DECODE_STATUS_INVALID_CURSOR` status
    /// word lands in Q[0] and NO Q is written (the stale-reference path the
    /// pairing probe must detect); an admitted decode publishes a dense-f32
    /// Q through the library body under the baked rotate-half pairing.
    fn simulate_qkv_carrier(
        &mut self,
        module: u64,
        buffers: &[u64],
        sim: &FakeQkvCarrierSim,
    ) -> HostResult<()> {
        use crate::kernel::library::{qkv_projection, QkvProjectionBind, QkvProjectionWeight};

        if !self.modules.contains_key(&module) {
            return Err(HostError::internal("fake launch missing module"));
        }
        let binding = |index: usize, what: &str| {
            buffers.get(index).copied().ok_or_else(|| {
                HostError::invalid_args(format!(
                    "fake QKV carrier launch has no binding {index} for {what}"
                ))
            })
        };
        let cursor = self.read_slot_f32(binding(sim.cursor, "carrier cursor")?, "cursor")?;
        let position = cursor.first().copied().unwrap_or(f32::NAN);
        let valid_len_after = cursor.get(1).copied().unwrap_or(f32::NAN);
        let query_rows = cursor.get(2).copied().unwrap_or(f32::NAN);
        if query_rows != sim.baked_query_rows as f32 || valid_len_after != position + query_rows {
            // Status word only: the rejected decode writes no Q.
            let q_out = binding(sim.q_out, "carrier Q output")?;
            let mut status = self.read_slot_f32(q_out, "carrier Q output")?;
            status[0] = f32::from_bits(1u32);
            return self.write_slot_f32(q_out, &status, "carrier Q output");
        }
        let activation =
            self.read_slot_f32(binding(sim.activation, "carrier activation")?, "activation")?;
        let q_weight =
            self.read_slot_f32(binding(sim.q_weight, "carrier Q weight")?, "Q weight")?;
        let k_weight =
            self.read_slot_f32(binding(sim.k_weight, "carrier K weight")?, "K weight")?;
        let v_weight =
            self.read_slot_f32(binding(sim.v_weight, "carrier V weight")?, "V weight")?;
        let bias = match sim.q_bias {
            Some(index) => Some(self.read_slot_f32(binding(index, "carrier bias")?, "bias")?),
            None => None,
        };
        let cos = self.read_slot_f32(binding(sim.cos, "carrier cos table")?, "cos table")?;
        let sin = self.read_slot_f32(binding(sim.sin, "carrier sin table")?, "sin table")?;
        let mut bind = QkvProjectionBind::grouped(
            sim.rows,
            sim.hidden,
            sim.kv_heads,
            sim.q_per_kv,
            sim.head_dim,
            [1, 1, 1],
        );
        bind.rotate_half = sim.rotate_half;
        bind.rope_position = position as u64;
        let q_width = sim
            .kv_heads
            .saturating_mul(sim.q_per_kv)
            .saturating_mul(sim.head_dim) as usize;
        let kv_span = sim
            .kv_heads
            .saturating_mul(sim.rows)
            .saturating_mul(sim.head_dim) as usize;
        let mut q_values = vec![0.0f32; q_width];
        let mut k_values = vec![0.0f32; kv_span];
        let mut v_values = vec![0.0f32; kv_span];
        let bias_ref = bias.as_deref();
        qkv_projection(
            &bind,
            &activation,
            QkvProjectionWeight::Dense(&q_weight),
            QkvProjectionWeight::Dense(&k_weight),
            QkvProjectionWeight::Dense(&v_weight),
            bias_ref,
            bias_ref,
            bias_ref,
            Some(&cos),
            Some(&sin),
            &mut q_values,
            &mut k_values,
            &mut v_values,
        )
        .map_err(|error| {
            HostError::invalid_args(format!("fake QKV carrier simulation failed: {error}"))
        })?;
        if sim.skew != 0.0 {
            for value in &mut q_values {
                *value += sim.skew;
            }
        }
        self.write_slot_f32(
            binding(sim.q_out, "carrier Q output")?,
            &q_values,
            "carrier Q output",
        )
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
        self.buffers.insert(
            token,
            FakeBufferSlot {
                bytes: vec![0; len_bytes],
                region_offset: 0,
                logical_len: len_bytes,
            },
        );
        Ok(token)
    }

    fn copy_in(&mut self, token: u64, bytes: &[u8]) -> HostResult<()> {
        self.maybe_fail(FakeFailureStage::CopyIn)?;
        if !self.buffers.contains_key(&token) {
            return Err(HostError::internal("fake copy_in missing buffer"));
        }
        if let Some(wrap) = mapped_wrap_for(&self.maps, bytes) {
            let page = unsafe { std::slice::from_raw_parts(wrap.ptr.cast::<u8>(), wrap.page_len) };
            let slot = self
                .buffers
                .get_mut(&token)
                .ok_or_else(|| HostError::internal("fake copy_in missing buffer"))?;
            slot.bytes = page.to_vec();
            slot.region_offset = wrap.offset;
            self.mmap_wraps += 1;
            return Ok(());
        }
        let slot = self
            .buffers
            .get_mut(&token)
            .ok_or_else(|| HostError::internal("fake copy_in missing buffer"))?;
        if slot.region_offset != 0 {
            return Err(HostError::invalid_args(
                "Metal mmap-backed region is read-only",
            ));
        }
        if slot.bytes.len() != bytes.len() {
            return Err(HostError::invalid_args("fake copy_in size mismatch"));
        }
        slot.bytes.copy_from_slice(bytes);
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
        // Encode-only: the CPU fake applies the write now (later kernels in
        // the same step must observe it) and defers the submit/wait count
        // until `sync` / readback flush.
        self.simulate_elementwise_add(module, a, 0, b, 0, out, 0)?;
        self.pending_encodes += 1;
        Ok(())
    }

    fn launch_kernel_bound(
        &mut self,
        module: u64,
        entry: &[u8],
        buffers: &[u64],
        byte_offsets: &[u64],
        view_spans: &[u64],
        _grid_x: u32,
        _grid_y: u32,
        _grid_z: u32,
        _block_x: u32,
        _block_y: u32,
        _block_z: u32,
    ) -> HostResult<()> {
        self.maybe_fail(FakeFailureStage::Launch)?;
        if buffers.len() != byte_offsets.len() || buffers.len() != view_spans.len() {
            return Err(HostError::invalid_args(
                "fake launch_kernel_bound offset/span arity mismatch",
            ));
        }
        for (token, (binding_offset, view_span)) in buffers
            .iter()
            .zip(byte_offsets.iter().zip(view_spans.iter()))
        {
            let slot = self.slot(*token, "launch")?;
            compose_bind_offset(
                slot.region_offset,
                *binding_offset,
                *view_span,
                slot.bytes.len() as u64,
            )?;
        }
        let offset_at = |index: usize| byte_offsets.get(index).copied().unwrap_or(0);
        // When the harness declares the module's function table, an unknown
        // entry fails closed before dispatch (mirrors the compiled-pipeline
        // entry lookup on the real lane).
        if !self.known_entries.is_empty()
            && !self
                .known_entries
                .iter()
                .any(|entry_name| entry_name.as_bytes() == entry)
        {
            return Err(metal_entry_mismatch(&String::from_utf8_lossy(entry)));
        }
        // The simulated fused QKV carrier serves its declared entry with the
        // compiled carrier's observable decode contract (cursor guard +
        // dense Q publish) for the RoPE-pairing probe tests.
        let carrier = match &self.qkv_carrier {
            Some(sim) if sim.entry.as_bytes() == entry => Some(sim.clone()),
            _ => None,
        };
        if let Some(sim) = carrier {
            self.simulate_qkv_carrier(module, buffers, &sim)?;
            self.pending_encodes += 1;
            return Ok(());
        }
        // Declared GEA2 entries use their emitted ABI arity and a structural,
        // encode-only simulation. The binding order is already established by
        // the session's ordered launch bindings; no kernel values are read or
        // written here. Keep this behind the declared-function-table path so
        // the legacy unnamed-module 3-buffer fake remains unchanged.
        if !self.known_entries.is_empty() {
            if let Some(arities) = Self::gea3_entry_arities(entry) {
                if !arities.contains(&buffers.len()) {
                    return Err(HostError::invalid_args(
                        "fake launch_kernel declared GEA3 entry".to_owned(),
                    ));
                }
                self.pending_encodes += 1;
                return Ok(());
            }
            if let Some(arities) = Self::gea2_entry_arities(entry) {
                if !arities.contains(&buffers.len()) {
                    return Err(HostError::invalid_args(
                        "fake launch_kernel declared GEA2 entry".to_owned(),
                    ));
                }
                self.pending_encodes += 1;
                return Ok(());
            }
        }
        // The simulated elementwise-add kernel takes exactly three buffers
        // (a, b, out); the simulated accumulate kernel takes two (a, acc)
        // and adds a into acc in place. Anything else fails closed in the
        // fake just as it would on device.
        if entry == b"accumulate" {
            expect_fake_arity(
                buffers,
                2,
                "'accumulate' simulates the 2-buffer kernel (a, acc)",
            )?;
            self.simulate_accumulate(module, buffers[0], offset_at(0), buffers[1], offset_at(1))?;
            self.pending_encodes += 1;
            return Ok(());
        }
        if entry == b"observa" {
            expect_fake_arity(
                buffers,
                2,
                "'observa' simulates the 2-buffer kernel (src, out)",
            )?;
            self.simulate_copy(module, buffers[0], offset_at(0), buffers[1], offset_at(1))?;
            self.pending_encodes += 1;
            return Ok(());
        }
        expect_fake_arity(
            buffers,
            3,
            "simulates the 3-buffer elementwise-add kernel (a, b, out)",
        )?;
        self.simulate_elementwise_add(
            module,
            buffers[0],
            offset_at(0),
            buffers[1],
            offset_at(1),
            buffers[2],
            offset_at(2),
        )?;
        self.pending_encodes += 1;
        Ok(())
    }

    fn sync(&mut self) -> HostResult<()> {
        self.maybe_fail(FakeFailureStage::Sync)?;
        self.flush_pending();
        Ok(())
    }

    fn copy_out(&mut self, token: u64, len_bytes: usize) -> HostResult<Vec<u8>> {
        // Readback coalesces behind the same flush as `sync`: one submit +
        // one wait if anything is still pending, then the host copy.
        self.flush_pending();
        self.maybe_fail(FakeFailureStage::Readback)?;
        let slot = self
            .buffers
            .get(&token)
            .ok_or_else(|| HostError::internal("fake copy_out missing buffer"))?;
        let offset = slot.region_offset as usize;
        match offset.checked_add(len_bytes) {
            Some(end) if end <= slot.bytes.len() => Ok(slot.bytes[offset..end].to_vec()),
            _ => Err(HostError::internal("fake copy_out size mismatch")),
        }
    }

    fn retain_mapped_file(&mut self, file: MappedWeightFile) {
        self.maps.push(file);
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

    fn command_submit_count(&self) -> usize {
        self.command_submits
    }

    fn blocking_wait_count(&self) -> usize {
        self.blocking_waits
    }

    fn mmap_wrap_count(&self) -> usize {
        self.mmap_wraps
    }
}

impl FakeMetalDriver {
    /// Count one command-buffer submit + blocking wait if any encodes are
    /// still pending. Idempotent when the step is already flushed.
    fn flush_pending(&mut self) {
        if self.pending_encodes == 0 {
            return;
        }
        self.command_submits += 1;
        self.blocking_waits += 1;
        self.pending_encodes = 0;
    }
}

/// Live Metal driver: real gfx-rs `metal` binding (M2).
///
/// Owns the first enumerated Metal device, a command queue, compiled compute
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
    buffers: BTreeMap<u64, MetalBufferSlot>,
    /// Read-only GGUF mappings that no-copy buffers reference.
    maps: Vec<MappedWeightFile>,
    next_token: u64,
    /// Open command buffer for the current step. Created on the first
    /// encode; committed and waited at `sync` / readback flush.
    pending: Option<CommandBuffer>,
    /// Temporary buffers (runtime-extent slots) kept alive until the pending
    /// command buffer commits.
    deferred_free: Vec<u64>,
    command_submits: usize,
    blocking_waits: usize,
    /// Timestamp counter sample buffer for per-encoder GPU times.
    timestamp_buffer: Option<CounterSampleBuffer>,
    /// Encoders sampled on the pending command buffer.
    encoder_sample_count: usize,
    /// Last committed step's per-encoder GPU times (µs).
    last_encoder_gpu_us: Vec<u64>,
    /// Last committed step's per-encoder GPU start times (µs, relative to
    /// the first encoder start). Empty when unsampled.
    last_encoder_gpu_start_us: Vec<u64>,
    /// Timestamp counters were probed and are unavailable on this device.
    timestamp_unavailable: bool,
    /// Times `copy_in` took the retained-mmap wrap branch.
    mmap_wraps: usize,
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

/// One Metal buffer plus the bind offset of the admitted region.
///
/// Ordinary allocs use offset 0. mmap wraps page-round the mapping and
/// store the intra-page remainder here so `set_buffer` still names the
/// tensor start. KV-B6 launch-binding offsets add on top of this remainder;
/// they must not replace it.
#[cfg(target_os = "macos")]
struct MetalBufferSlot {
    buffer: Buffer,
    region_offset: u64,
}

#[cfg(target_os = "macos")]
impl MetalDriver for SystemMetalDriver {
    fn discover(&mut self) -> HostResult<MetalEnvReport> {
        let mut devices = Device::all();
        if devices.is_empty() {
            return Err(metal_unavailable(
                "no Metal device enumerated (MTLCopyAllDevices returned empty)",
            ));
        }
        let name = devices[0].name().to_owned();
        let count = devices.len();
        self.device = Some(devices.remove(0));
        Ok(MetalEnvReport {
            admitted: true,
            mtl_device: Some(name),
            metal_framework_paths: detect_metal_framework_paths(),
            reason: format!("Metal enumerated {count} device(s); SystemMetalDriver admits"),
        })
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
        self.buffers.insert(
            token,
            MetalBufferSlot {
                buffer,
                region_offset: 0,
            },
        );
        Ok(token)
    }

    fn copy_in(&mut self, token: u64, bytes: &[u8]) -> HostResult<()> {
        if !self.buffers.contains_key(&token) {
            return Err(metal_driver("copy_in: unknown buffer token"));
        }
        if let Some(wrap) = self.mmap_wrap_for(bytes) {
            let device = self
                .device
                .as_ref()
                .ok_or_else(|| metal_unavailable("SystemMetalDriver has no device"))?;
            let buffer = device.new_buffer_with_bytes_no_copy(
                wrap.ptr,
                wrap.page_len as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            self.buffers.insert(
                token,
                MetalBufferSlot {
                    buffer,
                    region_offset: wrap.offset,
                },
            );
            self.mmap_wraps += 1;
            return Ok(());
        }
        let slot = self
            .buffers
            .get(&token)
            .ok_or_else(|| metal_driver("copy_in: unknown buffer token"))?;
        if slot.region_offset != 0 {
            return Err(HostError::invalid_args(
                "Metal mmap-backed region is read-only",
            ));
        }
        if slot.buffer.length() as usize != bytes.len() {
            return Err(HostError::invalid_args("Metal copy_in size mismatch"));
        }
        let destination = slot.buffer.contents().cast::<u8>();
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
        self.buffers.insert(
            extent_token,
            MetalBufferSlot {
                buffer: extent_buffer,
                region_offset: 0,
            },
        );

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
        // Keep the extent buffer until the pending command buffer commits so
        // the encoder's bind stays live across encode-only mid-step.
        if result.is_ok() {
            self.deferred_free.push(extent_token);
        } else {
            self.buffers.remove(&extent_token);
        }
        result
    }

    fn launch_kernel_bound(
        &mut self,
        module: u64,
        entry: &[u8],
        buffers: &[u64],
        byte_offsets: &[u64],
        view_spans: &[u64],
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
    ) -> HostResult<()> {
        // Validate everything before touching the encoder so a stale module,
        // unknown entry, or illegal grid cannot leave a half-recorded buffer.
        let entry_name = String::from_utf8_lossy(entry);
        let pipeline = self
            .modules
            .get(&module)
            .ok_or_else(|| metal_driver("launch: unknown module token"))?
            .pipelines
            .get(entry_name.as_ref())
            .ok_or_else(|| metal_entry_mismatch(&entry_name))?
            .clone();
        if grid_x == 0 || grid_y == 0 || grid_z == 0 || block_x == 0 || block_y == 0 || block_z == 0
        {
            return Err(metal_driver(
                "launch: all grid and block dimensions must be non-zero",
            ));
        }
        if buffers.len() != byte_offsets.len() || buffers.len() != view_spans.len() {
            return Err(metal_driver(
                "launch: binding offset/span arity does not match the buffer slice",
            ));
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
        // Retain buffer handles so the encoder bind does not borrow `self`.
        // Composed offset = mmap page remainder + B4 launch-binding offset.
        let mut bound = Vec::with_capacity(buffers.len());
        let mut offsets = Vec::with_capacity(buffers.len());
        for (token, (binding_offset, view_span)) in buffers
            .iter()
            .zip(byte_offsets.iter().zip(view_spans.iter()))
        {
            let slot = self
                .buffers
                .get(token)
                .ok_or_else(|| metal_driver("launch: unknown buffer token"))?;
            let composed = compose_bind_offset(
                slot.region_offset,
                *binding_offset,
                *view_span,
                slot.buffer.length(),
            )?;
            bound.push(slot.buffer.clone());
            offsets.push(composed);
        }
        if self.pending.is_none() {
            let queue = self
                .queue
                .as_ref()
                .ok_or_else(|| metal_unavailable("SystemMetalDriver has no command queue"))?;
            self.pending = Some(queue.new_command_buffer().to_owned());
            self.encoder_sample_count = 0;
        }
        let command_buffer = self
            .pending
            .as_ref()
            .ok_or_else(|| metal_driver("launch: pending command buffer missing after ensure"))?
            .to_owned();
        let sampled = per_op_timing_wanted()
            && self.ensure_timestamp_buffer()
            && (self.encoder_sample_count as u64)
                .saturating_mul(2)
                .saturating_add(1)
                < TIMESTAMP_SAMPLE_CAPACITY;
        let encoder = if sampled {
            let start = (self.encoder_sample_count as u64).saturating_mul(2);
            let end = start.saturating_add(1);
            let descriptor = ComputePassDescriptor::new();
            let sample_buffer = self
                .timestamp_buffer
                .as_ref()
                .ok_or_else(|| metal_driver("launch: timestamp sample buffer missing"))?;
            let attachment = descriptor.sample_buffer_attachments().object_at(0);
            if let Some(attachment) = attachment {
                attachment.set_sample_buffer(sample_buffer);
                attachment.set_start_of_encoder_sample_index(start);
                attachment.set_end_of_encoder_sample_index(end);
                self.encoder_sample_count += 1;
                command_buffer.compute_command_encoder_with_descriptor(descriptor)
            } else {
                command_buffer.new_compute_command_encoder()
            }
        } else {
            command_buffer.new_compute_command_encoder()
        };
        encoder.set_compute_pipeline_state(&pipeline);
        for (index, (buffer, offset)) in bound.iter().zip(offsets).enumerate() {
            // `offset` is mmap page remainder + launch-binding offset.
            encoder.set_buffer(index as u64, Some(buffer), offset);
        }
        // Each threadgroup carries exactly one block now that the volume is
        // guaranteed to fit, so the grid needs no widening along x.
        encoder.dispatch_thread_groups(
            MTLSize::new(grid_x as u64, grid_y as u64, grid_z as u64),
            MTLSize::new(block_x as u64, block_y as u64, block_z as u64),
        );
        encoder.end_encoding();
        Ok(())
    }

    fn sync(&mut self) -> HostResult<()> {
        self.commit_pending()
    }

    fn copy_out(&mut self, token: u64, len_bytes: usize) -> HostResult<Vec<u8>> {
        // Coalesce readback behind the same step-boundary flush: if the
        // caller skipped `sync`, the pending buffer commits here once.
        self.commit_pending()?;
        let slot = self
            .buffers
            .get(&token)
            .ok_or_else(|| metal_driver("copy_out: unknown buffer token"))?;
        let offset = slot.region_offset as usize;
        let buffer_len = slot.buffer.length() as usize;
        match offset.checked_add(len_bytes) {
            Some(end) if end <= buffer_len => {}
            _ => return Err(HostError::internal("Metal copy_out size mismatch")),
        }
        let mut output = vec![0u8; len_bytes];
        let source = slot.buffer.contents().cast::<u8>();
        // Safe: `offset + len_bytes` fits the buffer (checked above), and
        // shared-memory storage is host-readable after the flush wait.
        unsafe {
            std::ptr::copy_nonoverlapping(source.add(offset), output.as_mut_ptr(), len_bytes);
        }
        Ok(output)
    }

    fn retain_mapped_file(&mut self, file: MappedWeightFile) {
        self.maps.push(file);
    }

    fn free(&mut self, token: u64) -> HostResult<()> {
        self.buffers.remove(&token);
        self.modules.remove(&token);
        Ok(())
    }

    fn command_submit_count(&self) -> usize {
        self.command_submits
    }

    fn blocking_wait_count(&self) -> usize {
        self.blocking_waits
    }

    fn mmap_wrap_count(&self) -> usize {
        self.mmap_wraps
    }

    fn take_encoder_gpu_us(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.last_encoder_gpu_us)
    }

    fn take_encoder_gpu_start_us(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.last_encoder_gpu_start_us)
    }
}

struct MmapWrap {
    ptr: *const std::ffi::c_void,
    page_len: usize,
    offset: u64,
}

/// Page-round a slice that sits inside a retained mapping so it can wrap
/// as `newBufferWithBytesNoCopy` (pointer and length page-aligned).
fn mapped_wrap_for(maps: &[MappedWeightFile], bytes: &[u8]) -> Option<MmapWrap> {
    if bytes.is_empty() {
        return None;
    }
    let ptr = bytes.as_ptr();
    for mapped in maps {
        if !mapped.contains(ptr, bytes.len()) {
            continue;
        }
        let page = mapped.page_size();
        if page == 0 || !page.is_power_of_two() {
            return None;
        }
        let addr = ptr as usize;
        let page_addr = addr & !(page - 1);
        let map_start = mapped.as_ptr() as usize;
        let map_end = map_start.saturating_add(mapped.mapped_len());
        if page_addr < map_start {
            return None;
        }
        let offset = addr - page_addr;
        // Cover the packed f32-word width (1–3 byte pad) so copy_out of
        // the handle length still sits inside the no-copy buffer.
        let region_len = bytes.len().div_ceil(4).saturating_mul(4).max(bytes.len());
        let need = offset.saturating_add(region_len);
        let page_len = need.div_ceil(page).saturating_mul(page);
        if page_addr.saturating_add(page_len) > map_end {
            return None;
        }
        return Some(MmapWrap {
            ptr: page_addr as *const std::ffi::c_void,
            page_len,
            offset: offset as u64,
        });
    }
    None
}

#[cfg(target_os = "macos")]
impl SystemMetalDriver {
    fn mmap_wrap_for(&self, bytes: &[u8]) -> Option<MmapWrap> {
        mapped_wrap_for(&self.maps, bytes)
    }

    fn ensure_timestamp_buffer(&mut self) -> bool {
        if self.timestamp_buffer.is_some() {
            return true;
        }
        if self.timestamp_unavailable {
            return false;
        }
        let Some(device) = self.device.as_ref() else {
            self.timestamp_unavailable = true;
            return false;
        };
        if !device.supports_counter_sampling(MTLCounterSamplingPoint::AtStageBoundary) {
            self.timestamp_unavailable = true;
            return false;
        }
        let Some(counter_set) = device
            .counter_sets()
            .into_iter()
            .find(|set| set.name() == "timestamp")
        else {
            self.timestamp_unavailable = true;
            return false;
        };
        let descriptor = CounterSampleBufferDescriptor::new();
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        descriptor.set_sample_count(TIMESTAMP_SAMPLE_CAPACITY);
        descriptor.set_counter_set(&counter_set);
        match device.new_counter_sample_buffer_with_descriptor(&descriptor) {
            Ok(buffer) => {
                self.timestamp_buffer = Some(buffer);
                true
            }
            Err(_) => {
                self.timestamp_unavailable = true;
                false
            }
        }
    }

    /// Commit the pending step command buffer and block until it completes.
    /// No-op when the step has nothing pending (already flushed).
    fn commit_pending(&mut self) -> HostResult<()> {
        let Some(command_buffer) = self.pending.take() else {
            return Ok(());
        };
        let encoder_count = self.encoder_sample_count;
        self.encoder_sample_count = 0;
        self.last_encoder_gpu_us.clear();
        self.last_encoder_gpu_start_us.clear();
        let dest = if encoder_count > 0 {
            match (self.device.as_ref(), self.timestamp_buffer.as_ref()) {
                (Some(device), Some(sample_buffer)) => {
                    let sample_count = (encoder_count as u64).saturating_mul(2);
                    let dest = device.new_buffer(
                        sample_count.saturating_mul(8),
                        MTLResourceOptions::StorageModeShared,
                    );
                    let blit = command_buffer.new_blit_command_encoder();
                    blit.resolve_counters(sample_buffer, NSRange::new(0, sample_count), &dest, 0);
                    blit.end_encoding();
                    Some(dest)
                }
                _ => None,
            }
        } else {
            None
        };
        let mut cpu_start = 0u64;
        let mut gpu_start = 0u64;
        let mut cpu_end = 0u64;
        let mut gpu_end = 0u64;
        if dest.is_some() {
            if let Some(device) = self.device.as_ref() {
                device.sample_timestamps(&mut cpu_start, &mut gpu_start);
            }
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();
        self.command_submits += 1;
        self.blocking_waits += 1;
        for token in self.deferred_free.drain(..) {
            self.buffers.remove(&token);
        }
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(metal_driver("Metal command buffer did not complete"));
        }
        if dest.is_some() {
            if let Some(device) = self.device.as_ref() {
                device.sample_timestamps(&mut cpu_end, &mut gpu_end);
            }
        }
        if let Some(dest) = dest {
            let timeline = convert_encoder_gpu_timeline(
                &dest,
                encoder_count,
                cpu_start,
                cpu_end,
                gpu_start,
                gpu_end,
            );
            self.last_encoder_gpu_us = timeline.duration_us;
            self.last_encoder_gpu_start_us = timeline.start_us;
        }
        Ok(())
    }
}

/// GPU timeline for one sampled command buffer: per-encoder duration and
/// start time relative to the first encoder start.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct EncoderGpuTimeline {
    duration_us: Vec<u64>,
    start_us: Vec<u64>,
}

fn ticks_to_us(ticks: f64, gpu_span: u64, cpu_span: u64) -> u64 {
    if gpu_span == 0 || cpu_span == 0 {
        return 0;
    }
    let nanoseconds = ticks / (gpu_span as f64) * (cpu_span as f64);
    let micros = nanoseconds / 1000.0;
    if micros.is_finite() && micros > 0.0 {
        micros.round() as u64
    } else {
        0
    }
}

/// Convert interleaved start/end GPU ticks into durations and start times.
///
/// `samples` is `[start0, end0, start1, end1, …]`. Start times are relative
/// to `samples[0]` so inter-encoder gaps are `start[i+1] - (start[i] + duration[i])`.
fn encoder_gpu_timeline_from_samples(
    samples: &[u64],
    encoder_count: usize,
    cpu_start: u64,
    cpu_end: u64,
    gpu_start: u64,
    gpu_end: u64,
) -> EncoderGpuTimeline {
    let cpu_span = cpu_end.saturating_sub(cpu_start);
    let gpu_span = gpu_end.saturating_sub(gpu_start);
    let sample_count = encoder_count.saturating_mul(2);
    if encoder_count == 0 || cpu_span == 0 || gpu_span == 0 || samples.len() < sample_count {
        return EncoderGpuTimeline::default();
    }
    let origin = samples[0] as f64;
    let mut duration_us = Vec::with_capacity(encoder_count);
    let mut start_us = Vec::with_capacity(encoder_count);
    for index in 0..encoder_count {
        let begin = samples[index * 2] as f64;
        let end = samples[index * 2 + 1] as f64;
        start_us.push(ticks_to_us(begin - origin, gpu_span, cpu_span));
        duration_us.push(ticks_to_us(end - begin, gpu_span, cpu_span));
    }
    EncoderGpuTimeline {
        duration_us,
        start_us,
    }
}

#[cfg(target_os = "macos")]
fn convert_encoder_gpu_timeline(
    dest: &Buffer,
    encoder_count: usize,
    cpu_start: u64,
    cpu_end: u64,
    gpu_start: u64,
    gpu_end: u64,
) -> EncoderGpuTimeline {
    let sample_count = encoder_count.saturating_mul(2);
    if sample_count == 0 {
        return EncoderGpuTimeline::default();
    }
    let samples =
        unsafe { std::slice::from_raw_parts(dest.contents().cast::<u64>(), sample_count) };
    encoder_gpu_timeline_from_samples(
        samples,
        encoder_count,
        cpu_start,
        cpu_end,
        gpu_start,
        gpu_end,
    )
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

#[cfg(test)]
#[path = "metal_host_test.rs"]
mod encoder_gpu_timeline_tests;
