//! CUDA product-host lifecycle scaffolding for MIR v1 Track C2.
//!
//! Path A (this checkout): proof-environment admission and fail-closed
//! discovery. A real Driver API product run is not claimed without a loadable
//! device stack. Injected drivers may prove sequencing only.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

use faber::Valor;
use serde::{Deserialize, Serialize};

use crate::kernel::frame_data;
use crate::kernel::{HostError, HostResult};

/// Stable host error code for missing CUDA driver/device/toolchain.
pub const E_CUDA_UNAVAILABLE: &str = "E_CUDA_UNAVAILABLE";
/// Stable host error for unsupported CUDA host operations / product claims.
pub const E_CUDA_UNSUPPORTED: &str = "E_CUDA_UNSUPPORTED";
/// Stale or unknown opaque handle.
pub const E_CUDA_INVALID_HANDLE: &str = "E_CUDA_INVALID_HANDLE";
/// Driver-level failure after admission.
pub const E_CUDA_DRIVER: &str = "E_CUDA_DRIVER";

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

fn cuda_unsupported(message: impl Into<String>) -> HostError {
    HostError {
        code: E_CUDA_UNSUPPORTED.to_owned(),
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

    let mut libcuda_candidates = Vec::new();
    for path in [
        "/usr/lib/libcuda.so",
        "/usr/lib/x86_64-linux-gnu/libcuda.so",
        "/usr/lib/libcuda.dylib",
        "/usr/local/cuda/lib64/libcuda.so",
        "/usr/local/cuda/lib/libcuda.dylib",
    ] {
        if Path::new(path).exists() {
            libcuda_candidates.push(path.to_owned());
        }
    }
    // Best-effort scan of common Homebrew/CUDA roots without walking the world.
    for root in ["/opt/cuda", "/usr/local/cuda"] {
        let candidate = PathBuf::from(root).join("lib64/libcuda.so");
        if candidate.is_file() {
            let text = candidate.display().to_string();
            if !libcuda_candidates.contains(&text) {
                libcuda_candidates.push(text);
            }
        }
    }

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

/// Live-environment driver: admits only when probe finds stack signals; still
/// refuses product kernel launch until a real Driver API binding is wired.
#[derive(Default)]
struct SystemCudaDriver {
    report: Option<CudaEnvReport>,
}

impl CudaDriver for SystemCudaDriver {
    fn discover(&mut self) -> HostResult<CudaEnvReport> {
        let report = probe_cuda_environment();
        self.report = Some(report.clone());
        if report.admitted {
            Ok(report)
        } else {
            Err(cuda_unavailable(report.reason))
        }
    }

    fn create_context(&mut self) -> HostResult<()> {
        // Real cuCtxCreate would go here after libcuda load. Without a binding,
        // admission alone must not claim product execution.
        Err(cuda_unsupported(
            "system CUDA driver adapter is admission-only on this host; Driver API product path not wired",
        ))
    }

    fn load_module(&mut self, _image: &[u8]) -> HostResult<u64> {
        Err(cuda_unsupported("system CUDA load_module not wired"))
    }

    fn alloc(&mut self, _len_bytes: usize) -> HostResult<u64> {
        Err(cuda_unsupported("system CUDA alloc not wired"))
    }

    fn copy_in(&mut self, _token: u64, _bytes: &[u8]) -> HostResult<()> {
        Err(cuda_unsupported("system CUDA copy_in not wired"))
    }

    fn launch_elementwise_add_f32(
        &mut self,
        _module: u64,
        _a: u64,
        _b: u64,
        _out: u64,
        _len: usize,
    ) -> HostResult<()> {
        Err(cuda_unsupported("system CUDA launch not wired"))
    }

    fn sync(&mut self) -> HostResult<()> {
        Err(cuda_unsupported("system CUDA sync not wired"))
    }

    fn copy_out(&mut self, _token: u64, _len_bytes: usize) -> HostResult<Vec<u8>> {
        Err(cuda_unsupported("system CUDA copy_out not wired"))
    }

    fn free(&mut self, _token: u64) -> HostResult<()> {
        Err(cuda_unsupported("system CUDA free not wired"))
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
        len: usize,
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
        if a_bytes.len() != len * 4 || b_bytes.len() != len * 4 || out_buf.len() != len * 4 {
            return Err(HostError::invalid_args("fake launch length mismatch"));
        }
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
