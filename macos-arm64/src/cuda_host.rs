//! CUDA product-host lifecycle for MIR v1 (lane C).
//!
//! G2 wires `SystemCudaDriver` over the real CUDA Driver API via `libloading`
//! (dlopen `libcuda.so.1`, raw `fn` pointer symbols), generalized
//! `launch_kernel`, and the probe/loader-parity candidate list. A real Driver
//! API product run is not claimed without a loadable device stack; injected
//! drivers prove sequencing only.

use std::collections::BTreeMap;
use std::ffi::{c_char, c_void};
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

use faber::Valor;
use serde::{Deserialize, Serialize};

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

struct CudaHandle {
    kind: CudaHandleKind,
    /// Backend token; fake drivers use synthetic ids. Never tensor payload.
    backend_token: u64,
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
}

/// Product-facing session: opaque handles + ordered lifecycle.
pub struct CudaHostSession {
    driver: Box<dyn CudaDriver>,
    handles: BTreeMap<u64, CudaHandle>,
    next_id: u64,
    admitted: bool,
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
        let mut session = Self {
            driver,
            handles: BTreeMap::new(),
            next_id: 1,
            admitted: true,
        };
        session.driver.create_context()?;
        Ok(session)
    }

    /// Inject a driver for unit tests (sequencing / reject paths only).
    pub fn with_driver(mut driver: Box<dyn CudaDriver>) -> HostResult<Self> {
        let report = driver.discover()?;
        let admitted = report.admitted;
        if admitted {
            driver.create_context()?;
        }
        Ok(Self {
            driver,
            handles: BTreeMap::new(),
            next_id: 1,
            admitted,
        })
    }

    pub fn is_admitted(&self) -> bool {
        self.admitted
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

    pub fn readback_f32(&mut self, buffer: CudaHandleId) -> HostResult<Vec<f32>> {
        self.require_admitted()?;
        let (token, len_bytes) = self.buffer_token(buffer)?;
        let bytes = self.driver.copy_out(token, len_bytes)?;
        if bytes.len() != len_bytes || len_bytes % 4 != 0 {
            return Err(HostError::internal(
                "CUDA readback returned unexpected byte length",
            ));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }

    pub fn release(&mut self, id: CudaHandleId) -> HostResult<()> {
        let Some(handle) = self.handles.remove(&id.0) else {
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
        let id = self.next_id;
        self.next_id += 1;
        self.handles.insert(
            id,
            CudaHandle {
                kind,
                backend_token,
            },
        );
        CudaHandleId(id)
    }

    fn module_token(&self, id: CudaHandleId) -> HostResult<u64> {
        match self.handles.get(&id.0) {
            Some(CudaHandle {
                kind: CudaHandleKind::Module,
                backend_token,
            }) => Ok(*backend_token),
            Some(_) => Err(HostError::invalid_args("handle is not a CUDA module")),
            None => Err(cuda_invalid_handle(id)),
        }
    }

    fn buffer_token(&self, id: CudaHandleId) -> HostResult<(u64, usize)> {
        match self.handles.get(&id.0) {
            Some(CudaHandle {
                kind: CudaHandleKind::Buffer { len_bytes },
                backend_token,
            }) => Ok((*backend_token, *len_bytes)),
            Some(_) => Err(HostError::invalid_args("handle is not a CUDA buffer")),
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
        self._library = Some(library);
        self.api = Some(api);
        Ok(report)
    }

    fn create_context(&mut self) -> HostResult<()> {
        let api = self.loaded_api()?;
        let mut device: i32 = 0;
        let mut result = unsafe { (api.cu_device_get)(&mut device, 0) };
        if result != CUDA_SUCCESS {
            return Err(cuda_driver(format!(
                "cuDeviceGet failed with CUDA result {result}"
            )));
        }
        // cuDevicePrimaryCtxRetain: the modern (non-deprecated) path. cuCtxCreate
        // is deprecated in CUDA 12+ headers but functional in 13.2; the retained
        // primary context is made current for this thread, so every subsequent
        // call (module load, mem, launch, sync) targets it. The retained context
        // outlives the one-shot proof process; teardown (cuDevicePrimaryCtxRelease /
        // cuCtxDestroy) is deferred and recorded here, not silent.
        let mut context: *mut c_void = std::ptr::null_mut();
        result = unsafe { (api.cu_device_primary_ctx_retain)(&mut context, device) };
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
            return Err(cuda_driver(format!(
                "cuModuleGetFunction({}) failed with CUDA result {result}",
                String::from_utf8_lossy(entry)
            )));
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

/// Raw CUDA Driver API symbol table, resolved once at load time.
///
/// `libloading::Symbol` is not `Send`, so every symbol is converted to a raw
/// `fn` pointer when the library is loaded; `CudaDriverApi` is therefore
/// trivially `Send + Sync` and the driver stays boxable behind
/// `Box<dyn CudaDriver>`.
#[derive(Clone, Copy)]
struct CudaDriverApi {
    cu_init: unsafe extern "C" fn(u32) -> i32,
    cu_device_get: unsafe extern "C" fn(*mut i32, i32) -> i32,
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
            cu_device_get: resolve_symbol(library, b"cuDeviceGet\0")?,
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
}

impl FakeCudaDriver {
    pub fn unavailable() -> Self {
        Self {
            force_unavailable: true,
            ..Self::default()
        }
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
        let token = self.next_token;
        self.next_token += 1;
        self.modules.insert(token, image.to_vec());
        Ok(token)
    }

    fn alloc(&mut self, len_bytes: usize) -> HostResult<u64> {
        let token = self.next_token;
        self.next_token += 1;
        self.buffers.insert(token, vec![0; len_bytes]);
        Ok(token)
    }

    fn copy_in(&mut self, token: u64, bytes: &[u8]) -> HostResult<()> {
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
        _entry: &[u8],
        buffers: &[u64],
        _grid_x: u32,
        _grid_y: u32,
        _grid_z: u32,
        _block_x: u32,
        _block_y: u32,
        _block_z: u32,
    ) -> HostResult<()> {
        // The emitted `addita` kernel takes exactly three buffers (a, b, out).
        // Anything else fails closed in the fake just as it would on device.
        if buffers.len() != 3 {
            return Err(HostError::invalid_args(
                "fake launch_kernel simulates the 3-buffer elementwise-add kernel (a, b, out)",
            ));
        }
        self.simulate_elementwise_add(module, buffers[0], buffers[1], buffers[2])
    }

    fn sync(&mut self) -> HostResult<()> {
        Ok(())
    }

    fn copy_out(&mut self, token: u64, len_bytes: usize) -> HostResult<Vec<u8>> {
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
        self.buffers.remove(&token);
        self.modules.remove(&token);
        Ok(())
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
