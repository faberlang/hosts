//! Metal product-host lifecycle scaffolding for MIR v1 (lane M).
//!
//! Path A (this checkout): proof-environment admission and fail-closed
//! discovery, mirroring [`cuda_host`] naming 1:1. M1 is a skeleton: an
//! injectable driver seam plus a sequencing fake. The real system binding
//! (`SystemMetalDriver`, gfx-rs `metal`) is M2; no product run is claimed
//! without a loadable device stack.

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use faber::Valor;
use serde::{Deserialize, Serialize};

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

struct MetalHandle {
    kind: MetalHandleKind,
    /// Backend token; fake drivers use synthetic ids. Never tensor payload.
    backend_token: u64,
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
    fn sync(&mut self) -> HostResult<()>;
    fn copy_out(&mut self, token: u64, len_bytes: usize) -> HostResult<Vec<u8>>;
    fn free(&mut self, token: u64) -> HostResult<()>;
}

/// Product-facing session: opaque handles + ordered lifecycle.
///
/// M1 is injectable-driver only: `SystemMetalDriver` and `try_open` land with
/// the M2 binding, so a session is always opened against an injected driver.
pub struct MetalHostSession {
    driver: Box<dyn MetalDriver>,
    handles: BTreeMap<u64, MetalHandle>,
    next_id: u64,
    admitted: bool,
}

impl MetalHostSession {
    /// Inject a driver for unit tests (sequencing / reject paths only).
    pub fn with_driver(mut driver: Box<dyn MetalDriver>) -> HostResult<Self> {
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
        let Some(handle) = self.handles.remove(&id.0) else {
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
        let id = self.next_id;
        self.next_id += 1;
        self.handles.insert(
            id,
            MetalHandle {
                kind,
                backend_token,
            },
        );
        MetalHandleId(id)
    }

    fn module_token(&self, id: MetalHandleId) -> HostResult<u64> {
        match self.handles.get(&id.0) {
            Some(MetalHandle {
                kind: MetalHandleKind::Module,
                backend_token,
            }) => Ok(*backend_token),
            Some(_) => Err(HostError::invalid_args("handle is not a Metal module")),
            None => Err(metal_invalid_handle(id)),
        }
    }

    fn buffer_token(&self, id: MetalHandleId) -> HostResult<(u64, usize)> {
        match self.handles.get(&id.0) {
            Some(MetalHandle {
                kind: MetalHandleKind::Buffer { len_bytes },
                backend_token,
            }) => Ok((*backend_token, *len_bytes)),
            Some(_) => Err(HostError::invalid_args("handle is not a Metal buffer")),
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

// System default Metal device probe (null check only; M1 has no binding).
//
// `MTLCreateSystemDefaultDevice` is the single Metal framework touchpoint at
// M1 — the admission check is a device null probe, not a binding. The function
// returns a retained reference to the process-wide system default device; M1
// deliberately keeps that singleton alive (the framework caches it for the
// process lifetime, and M2's binding manages its own reference).
#[cfg(target_os = "macos")]
#[link(name = "Metal", kind = "framework")]
extern "C" {
    fn MTLCreateSystemDefaultDevice() -> *mut std::ffi::c_void;
}

/// Probe this machine for a loadable Metal device stack without claiming a run.
pub fn probe_metal_environment() -> MetalEnvReport {
    let metal_framework_paths = detect_metal_framework_paths();

    #[cfg(target_os = "macos")]
    let device_detected = {
        let device = unsafe { MTLCreateSystemDefaultDevice() };
        !device.is_null()
    };
    #[cfg(not(target_os = "macos"))]
    let device_detected = false;

    let admitted = device_detected;
    let mtl_device = if device_detected {
        Some("system default Metal device".to_owned())
    } else {
        None
    };
    let reason = if admitted {
        "Metal default device present; product launch still requires the M2 SystemMetalDriver binding and a compiled MSL kernel artifact".to_owned()
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
}

impl FakeMetalDriver {
    pub fn unavailable() -> Self {
        Self {
            force_unavailable: true,
            ..Self::default()
        }
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

impl fmt::Debug for MetalHostSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalHostSession")
            .field("admitted", &self.admitted)
            .field("handles", &self.handles.len())
            .finish_non_exhaustive()
    }
}
