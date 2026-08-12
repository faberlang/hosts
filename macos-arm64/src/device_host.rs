//! Uniform device-session surface for the composite host (campaign S1-4).
//!
//! The native sessions (`CudaHostSession`, `MetalHostSession`) are
//! backend-specific. This module adapts them behind one backend-neutral
//! [`DeviceSession`] trait and one [`DeviceRuntime`] enum so the composite
//! host composes a single device component (A8: device execution lives in
//! host components, not provider routing). Handles across this surface are
//! faber-runtime [`DeviceHandle`] carriers — opaque ids, never payload bytes
//! — and every handle carries its backend, so a handle from one backend
//! session fails closed when passed to the other.

use host_coordinator::{DeviceBackend, DeviceHandle, DeviceHandleKind};

use crate::cuda_host::{CudaHandleId, CudaHostSession};
use crate::device_registry::DriverCounters;
use crate::kernel::{HostError, HostResult};
use crate::metal_host::{MetalHandleId, MetalHostSession};

/// Stable host error code for a device handle that does not belong to the
/// runtime's backend session (cross-backend misuse or an unparsable handle).
pub const E_DEVICE_INVALID_HANDLE: &str = "E_DEVICE_INVALID_HANDLE";

/// One of the two native device sessions, behind a uniform surface.
#[allow(clippy::large_enum_variant)]
pub enum DeviceRuntime {
    /// Apple Metal session (macOS).
    Metal(MetalHostSession),
    /// NVIDIA CUDA session (Driver API).
    Cuda(CudaHostSession),
}

impl DeviceRuntime {
    /// Open a session against the live environment for the given backend.
    /// Fails closed (backend-unavailable) when the machine cannot admit it.
    pub fn open(backend: DeviceBackend) -> HostResult<Self> {
        match backend {
            DeviceBackend::Metal => MetalHostSession::try_open().map(Self::Metal),
            DeviceBackend::Cuda => CudaHostSession::try_open().map(Self::Cuda),
        }
    }

    /// The backend this runtime speaks for.
    #[must_use]
    pub fn backend(&self) -> DeviceBackend {
        match self {
            Self::Metal(_) => DeviceBackend::Metal,
            Self::Cuda(_) => DeviceBackend::Cuda,
        }
    }

    /// Number of live opaque handles in the owning session (teardown/leak
    /// checks; the A9 lifecycle gate).
    #[must_use]
    pub fn live_handle_count(&self) -> usize {
        match self {
            Self::Metal(session) => session.live_handle_count(),
            Self::Cuda(session) => session.live_handle_count(),
        }
    }

    /// Driver-level lifecycle counters (S2-2 module-cache leak bar). The
    /// fake drivers track cumulative module loads/releases and buffer
    /// allocs/releases so session tests prove the policy at the driver
    /// boundary; the real drivers report all-zero (S2-8 real-device gate).
    #[must_use]
    pub fn driver_counters(&self) -> DriverCounters {
        match self {
            Self::Metal(session) => session.driver_counters(),
            Self::Cuda(session) => session.driver_counters(),
        }
    }
}

/// Backend-neutral lifecycle surface shared by every native device session.
///
/// Every operation resolves the caller's opaque [`DeviceHandle`] against the
/// owning session's registry before the driver is touched, and every failure
/// is a structured [`HostError`] — never a panic and never a silent fallback.
pub trait DeviceSession {
    /// The backend this session speaks for.
    fn backend(&self) -> DeviceBackend;
    /// Whether the session was admitted for product execution.
    fn is_admitted(&self) -> bool;
    /// Load a compiled module image (MSL source or PTX).
    fn load_module(&mut self, image: &[u8]) -> HostResult<DeviceHandle>;
    /// Allocate a device buffer of the given byte length.
    fn alloc_bytes(&mut self, len_bytes: usize) -> HostResult<DeviceHandle>;
    /// Copy f32 values into a device buffer (exact size match required).
    fn copy_in_f32(&mut self, buffer: &DeviceHandle, values: &[f32]) -> HostResult<()>;
    /// Launch a named kernel entry over device buffers with a 3D grid/block
    /// shape; the launch synchronizes internally.
    fn launch_kernel(
        &mut self,
        module: &DeviceHandle,
        entry: &str,
        buffers: &[DeviceHandle],
        grid: [u32; 3],
        block: [u32; 3],
    ) -> HostResult<()>;
    /// Explicit device synchronization barrier.
    fn sync(&mut self) -> HostResult<()>;
    /// Read a device buffer back as f32 values.
    fn readback_f32(&mut self, buffer: &DeviceHandle) -> HostResult<Vec<f32>>;
    /// Release a handle and its underlying device object.
    fn release(&mut self, handle: &DeviceHandle) -> HostResult<()>;
}

impl DeviceSession for DeviceRuntime {
    fn backend(&self) -> DeviceBackend {
        DeviceRuntime::backend(self)
    }

    fn is_admitted(&self) -> bool {
        match self {
            Self::Metal(session) => session.is_admitted(),
            Self::Cuda(session) => session.is_admitted(),
        }
    }

    fn load_module(&mut self, image: &[u8]) -> HostResult<DeviceHandle> {
        match self {
            Self::Metal(session) => {
                let id = session.load_module(image)?;
                Ok(DeviceHandle {
                    backend: DeviceBackend::Metal,
                    kind: DeviceHandleKind::Module,
                    id: id.0,
                })
            }
            Self::Cuda(session) => {
                let id = session.load_module(image)?;
                Ok(DeviceHandle {
                    backend: DeviceBackend::Cuda,
                    kind: DeviceHandleKind::Module,
                    id: id.0,
                })
            }
        }
    }

    fn alloc_bytes(&mut self, len_bytes: usize) -> HostResult<DeviceHandle> {
        match self {
            Self::Metal(session) => {
                let id = session.alloc_bytes(len_bytes)?;
                Ok(DeviceHandle {
                    backend: DeviceBackend::Metal,
                    kind: DeviceHandleKind::Buffer {
                        len_bytes: len_bytes as u64,
                    },
                    id: id.0,
                })
            }
            Self::Cuda(session) => {
                let id = session.alloc_bytes(len_bytes)?;
                Ok(DeviceHandle {
                    backend: DeviceBackend::Cuda,
                    kind: DeviceHandleKind::Buffer {
                        len_bytes: len_bytes as u64,
                    },
                    id: id.0,
                })
            }
        }
    }

    fn copy_in_f32(&mut self, buffer: &DeviceHandle, values: &[f32]) -> HostResult<()> {
        match self {
            Self::Metal(session) => session.copy_in_f32(metal_handle(buffer)?, values),
            Self::Cuda(session) => session.copy_in_f32(cuda_handle(buffer)?, values),
        }
    }

    fn launch_kernel(
        &mut self,
        module: &DeviceHandle,
        entry: &str,
        buffers: &[DeviceHandle],
        grid: [u32; 3],
        block: [u32; 3],
    ) -> HostResult<()> {
        match self {
            Self::Metal(session) => {
                let module_id = metal_handle(module)?;
                let buffer_ids: HostResult<Vec<MetalHandleId>> =
                    buffers.iter().map(metal_handle).collect();
                session.launch_kernel_3d(
                    module_id,
                    entry,
                    &buffer_ids?,
                    grid[0],
                    grid[1],
                    grid[2],
                    block[0],
                    block[1],
                    block[2],
                )
            }
            Self::Cuda(session) => {
                let module_id = cuda_handle(module)?;
                let buffer_ids: HostResult<Vec<CudaHandleId>> =
                    buffers.iter().map(cuda_handle).collect();
                session.launch_kernel_3d(
                    module_id,
                    entry,
                    &buffer_ids?,
                    grid[0],
                    grid[1],
                    grid[2],
                    block[0],
                    block[1],
                    block[2],
                )
            }
        }
    }

    fn sync(&mut self) -> HostResult<()> {
        match self {
            Self::Metal(session) => session.sync(),
            Self::Cuda(session) => session.sync(),
        }
    }

    fn readback_f32(&mut self, buffer: &DeviceHandle) -> HostResult<Vec<f32>> {
        match self {
            Self::Metal(session) => session.readback_f32(metal_handle(buffer)?),
            Self::Cuda(session) => session.readback_f32(cuda_handle(buffer)?),
        }
    }

    fn release(&mut self, handle: &DeviceHandle) -> HostResult<()> {
        match self {
            Self::Metal(session) => session.release(metal_handle(handle)?),
            Self::Cuda(session) => session.release(cuda_handle(handle)?),
        }
    }
}

fn metal_handle(handle: &DeviceHandle) -> HostResult<MetalHandleId> {
    if handle.backend != DeviceBackend::Metal {
        return Err(device_invalid_handle(handle));
    }
    Ok(MetalHandleId(handle.id))
}

fn cuda_handle(handle: &DeviceHandle) -> HostResult<CudaHandleId> {
    if handle.backend != DeviceBackend::Cuda {
        return Err(device_invalid_handle(handle));
    }
    Ok(CudaHandleId(handle.id))
}

fn device_invalid_handle(handle: &DeviceHandle) -> HostError {
    HostError {
        code: E_DEVICE_INVALID_HANDLE.to_owned(),
        message: format!(
            "device handle {} from backend {} cannot be used on a session for a different backend",
            handle.id,
            handle.backend.spelling()
        ),
        retryable: false,
    }
}
