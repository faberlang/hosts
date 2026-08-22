//! Backend-neutral lifecycle surface shared by native device sessions.
//!
//! Concrete product crates implement this trait for their runtime enum. The
//! trait keeps the common typed byte surface above any one backend product.

use host_coordinator::{DeviceBackend, DeviceHandle};

use crate::device_descriptor::DeviceDataType;
use crate::kernel::{HostError, HostResult};

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
    /// Copy dtype-tagged bytes into a device buffer without changing their
    /// representation.
    fn copy_in_bytes(
        &mut self,
        buffer: &DeviceHandle,
        bytes: &[u8],
        dtype: DeviceDataType,
    ) -> HostResult<()>;
    /// Copy f32 values into a device buffer (exact size match required).
    /// This is a compatibility wrapper over [`Self::copy_in_bytes`].
    fn copy_in_f32(&mut self, buffer: &DeviceHandle, values: &[f32]) -> HostResult<()> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        self.copy_in_bytes(buffer, &bytes, DeviceDataType::F32)
    }
    /// Whether this backend keeps mapped weight storage alive for the session.
    fn supports_mapped_weight_retention(&self) -> bool;
    /// Launch a named kernel entry over device buffers with a 3D grid/block
    /// shape. Metal encodes into the step command buffer (commit+wait at
    /// `sync`); CUDA still synchronizes internally.
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
    /// Read a device buffer back as dtype-tagged bytes without changing their
    /// representation.
    fn readback_bytes(
        &mut self,
        buffer: &DeviceHandle,
        dtype: DeviceDataType,
    ) -> HostResult<Vec<u8>>;
    /// Read a device buffer back as f32 values.
    /// This is a compatibility wrapper over [`Self::readback_bytes`].
    fn readback_f32(&mut self, buffer: &DeviceHandle) -> HostResult<Vec<f32>> {
        let bytes = self.readback_bytes(buffer, DeviceDataType::F32)?;
        if bytes.len() % 4 != 0 {
            return Err(HostError::internal(
                "f32 readback returned an unexpected byte length",
            ));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }
    /// Release a handle and its underlying device object.
    fn release(&mut self, handle: &DeviceHandle) -> HostResult<()>;
}
