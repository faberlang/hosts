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
//!
//! KV-B B4 launch bindings carry handle, binding index, byte offset, view
//! span, and runtime source. Offsets and spans are checked against B3
//! allocation/view facts before dispatch. The invocation-state buffer is
//! one 16-byte store, uploaded each step and never reallocated. CUDA
//! rejects dynamic bindings explicitly (`E_CUDA_UNSUPPORTED`) instead of
//! binding offset zero. [`DeviceSession::launch_kernel`] remains the
//! whole-handle offset-zero wrapper.

use host_coordinator::{DeviceBackend, DeviceHandle, DeviceHandleKind};

use crate::cuda_host::{CudaHandleId, CudaHostSession, E_CUDA_UNSUPPORTED};
use crate::device_descriptor::errors;
use crate::device_descriptor::{
    DescriptorInvocationState, DescriptorLaunchBinding, DescriptorRuntimeSource, DescriptorView,
    DeviceDataType, KvCacheDescriptor,
};
use crate::device_registry::DriverCounters;
use crate::kernel::{HostError, HostResult};
use crate::metal_host::{MetalHandleId, MetalHostSession, MetalLaunchBinding};

/// Stable host error code for a device handle that does not belong to the
/// runtime's backend session (cross-backend misuse or an unparsable handle).
pub const E_DEVICE_INVALID_HANDLE: &str = "E_DEVICE_INVALID_HANDLE";

/// One launch binding at the device-session surface.
///
/// Frozen shape: handle, binding index, byte offset, view span, runtime
/// source. Declared binding indices stay on the record through dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceLaunchBinding {
    /// Live device buffer for this binding.
    pub handle: DeviceHandle,
    /// Declared binding index. Never dropped before launch.
    pub binding_index: u32,
    /// Byte offset into the allocation (static envelope).
    pub byte_offset: u64,
    /// View span in bytes for this binding.
    pub view_span: u64,
    /// Constant facts or a typed invocation-state field.
    pub runtime_source: DescriptorRuntimeSource,
}

impl DeviceLaunchBinding {
    /// Legacy whole-handle binding: offset zero, span = handle length,
    /// constant source. Binding index is the caller's slice position.
    pub fn whole_handle(handle: DeviceHandle, binding_index: u32) -> HostResult<Self> {
        let Some(view_span) = handle.len_bytes() else {
            return Err(device_invalid_handle(&handle));
        };
        Ok(Self {
            handle,
            binding_index,
            byte_offset: 0,
            view_span,
            runtime_source: DescriptorRuntimeSource::Constant,
        })
    }

    /// CUDA cannot honor runtime sources or nonzero offsets in this campaign.
    #[must_use]
    pub fn is_cuda_dynamic(&self) -> bool {
        self.byte_offset != 0 || !matches!(self.runtime_source, DescriptorRuntimeSource::Constant)
    }
}

/// Small typed invocation-state store: four little-endian `u32` fields.
/// Allocated once; each step overwrites the same 16 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvocationStateBuffer {
    handle: DeviceHandle,
}

impl InvocationStateBuffer {
    /// Byte length of [`DescriptorInvocationState`] on the device.
    pub const BYTE_LENGTH: usize = 16;

    /// The persistent device handle. Identity is stable across uploads.
    #[must_use]
    pub fn handle(self) -> DeviceHandle {
        self.handle
    }

    /// Packed little-endian encoding of the four typed cursor fields.
    #[must_use]
    pub fn encoded_bytes(state: DescriptorInvocationState) -> [u8; Self::BYTE_LENGTH] {
        let mut bytes = [0u8; Self::BYTE_LENGTH];
        bytes[0..4].copy_from_slice(&state.position.to_le_bytes());
        bytes[4..8].copy_from_slice(&state.valid_len_after.to_le_bytes());
        bytes[8..12].copy_from_slice(&state.query_rows.to_le_bytes());
        bytes[12..16].copy_from_slice(&state.sequence_epoch.to_le_bytes());
        bytes
    }
}

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

    /// Command buffers submitted on this runtime (W8-U1). CUDA reports 0 —
    /// its receipt still counts per-kernel submits.
    #[must_use]
    pub fn command_submit_count(&self) -> usize {
        match self {
            Self::Metal(session) => session.command_submit_count(),
            Self::Cuda(_) => 0,
        }
    }

    /// Blocking waits on this runtime (W8-U1). CUDA reports 0.
    #[must_use]
    pub fn blocking_wait_count(&self) -> usize {
        match self {
            Self::Metal(session) => session.blocking_wait_count(),
            Self::Cuda(_) => 0,
        }
    }

    /// Per-encoder GPU timestamps from the last committed Metal step (µs).
    pub fn take_encoder_gpu_us(&mut self) -> Vec<u64> {
        match self {
            Self::Metal(session) => session.take_encoder_gpu_us(),
            Self::Cuda(_) => Vec::new(),
        }
    }

    /// Per-encoder GPU start times from the last committed Metal step (µs).
    pub fn take_encoder_gpu_start_us(&mut self) -> Vec<u64> {
        match self {
            Self::Metal(session) => session.take_encoder_gpu_start_us(),
            Self::Cuda(_) => Vec::new(),
        }
    }

    /// Allocate the reusable 16-byte invocation-state buffer. Callers upload
    /// into this handle each step; they must not allocate a new one per step.
    pub fn alloc_invocation_state(&mut self) -> HostResult<InvocationStateBuffer> {
        let handle = self.alloc_bytes(InvocationStateBuffer::BYTE_LENGTH)?;
        Ok(InvocationStateBuffer { handle })
    }

    /// Overwrite the existing invocation-state buffer. Never reallocates.
    pub fn upload_invocation_state(
        &mut self,
        buffer: &InvocationStateBuffer,
        state: DescriptorInvocationState,
    ) -> HostResult<()> {
        if buffer.handle.backend != self.backend() {
            return Err(device_invalid_handle(&buffer.handle));
        }
        let expected = InvocationStateBuffer::BYTE_LENGTH as u64;
        match buffer.handle.kind {
            DeviceHandleKind::Buffer { len_bytes } if len_bytes == expected => {}
            _ => {
                return Err(errors::descriptor(format!(
                    "invocation-state buffer must be {expected} bytes"
                )));
            }
        }
        self.copy_in_bytes(
            &buffer.handle,
            &InvocationStateBuffer::encoded_bytes(state),
            DeviceDataType::U8,
        )
    }

    /// Launch with explicit bindings. Offsets and spans are checked against
    /// each handle's allocation size before the backend is touched. CUDA
    /// dynamic bindings fail with [`E_CUDA_UNSUPPORTED`]. Binding indices
    /// select dispatch slots; they are not dropped. Metal carries each
    /// binding's offset and span through to [`MetalHostSession::launch_kernel_bound`].
    pub fn launch_kernel_bound(
        &mut self,
        module: &DeviceHandle,
        entry: &str,
        bindings: &[DeviceLaunchBinding],
        grid: [u32; 3],
        block: [u32; 3],
    ) -> HostResult<()> {
        validate_dispatch_bindings(self.backend(), bindings)?;
        match self {
            Self::Metal(session) => {
                let module_id = metal_handle(module)?;
                let metal_bindings: Vec<MetalLaunchBinding> = bindings
                    .iter()
                    .map(metal_from_device_binding)
                    .collect::<HostResult<_>>()?;
                session.launch_kernel_bound(module_id, entry, &metal_bindings, grid, block)
            }
            Self::Cuda(_) => {
                let handles = handles_in_binding_order(bindings)?;
                dispatch_kernel(self, module, entry, &handles, grid, block)
            }
        }
    }

    /// Resolve B3 launch records, validate offsets/spans against views, then
    /// dispatch. CUDA dynamic descriptors reject before a backend bind.
    pub fn launch_kv_kernel(
        &mut self,
        module: &DeviceHandle,
        entry: &str,
        kv: &KvCacheDescriptor,
        allocations: &[(u32, DeviceHandle)],
        grid: [u32; 3],
        block: [u32; 3],
    ) -> HostResult<()> {
        let bindings = resolve_launch_bindings(kv, allocations)?;
        self.launch_kernel_bound(module, entry, &bindings, grid, block)
    }
}

/// Resolve KV descriptor launch records onto live allocation handles.
/// Binding order and indices are the descriptor's launch records.
pub fn resolve_launch_bindings(
    kv: &KvCacheDescriptor,
    allocations: &[(u32, DeviceHandle)],
) -> HostResult<Vec<DeviceLaunchBinding>> {
    kv.validate()?;
    validate_allocation_map(kv, allocations)?;
    let mut resolved = Vec::with_capacity(kv.launch_records().len());
    for record in kv.launch_records() {
        resolved.push(resolve_one_binding(kv, allocations, record)?);
    }
    validate_launch_bindings(kv, &resolved, allocations)?;
    Ok(resolved)
}

/// Validate resolved launch bindings against B3 allocation/view facts.
/// Binding order is the caller's order (never sorted or dropped).
pub fn validate_launch_bindings(
    kv: &KvCacheDescriptor,
    bindings: &[DeviceLaunchBinding],
    allocations: &[(u32, DeviceHandle)],
) -> HostResult<()> {
    kv.validate()?;
    validate_allocation_map(kv, allocations)?;
    for binding in bindings {
        let allocation_id = allocation_id_for_handle(allocations, &binding.handle)?;
        let Some(allocation) = kv
            .allocations
            .iter()
            .find(|allocation| allocation.buffer_id == allocation_id)
        else {
            return Err(errors::descriptor(format!(
                "launch binding index {} names unknown allocation {allocation_id}",
                binding.binding_index
            )));
        };
        let Some(end) = binding.byte_offset.checked_add(binding.view_span) else {
            return Err(errors::shape_mismatch(format!(
                "launch binding index {} overflows its static envelope",
                binding.binding_index
            )));
        };
        if end > allocation.capacity_bytes {
            return Err(errors::shape_mismatch(format!(
                "launch binding index {} spans {} bytes from offset {} but allocation {allocation_id} capacity is {} bytes",
                binding.binding_index,
                binding.view_span,
                binding.byte_offset,
                allocation.capacity_bytes
            )));
        }
        if matching_view(kv, allocation_id, binding).is_none() {
            return Err(errors::shape_mismatch(format!(
                "launch binding index {} offset {} span {} does not match a view on allocation {allocation_id}",
                binding.binding_index, binding.byte_offset, binding.view_span
            )));
        }
    }
    Ok(())
}

/// Handles placed at their declared binding indices. Gaps and duplicates
/// fail closed so an index cannot be validated and then dropped.
pub fn handles_in_binding_order(bindings: &[DeviceLaunchBinding]) -> HostResult<Vec<DeviceHandle>> {
    let mut slots: Vec<Option<DeviceHandle>> = vec![None; bindings.len()];
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
        slots[index] = Some(binding.handle);
    }
    slots
        .into_iter()
        .enumerate()
        .map(|(index, handle)| {
            handle.ok_or_else(|| {
                errors::abi_mismatch(format!(
                    "launch binding index {index} was validated then dropped before launch"
                ))
            })
        })
        .collect()
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

    fn supports_mapped_weight_retention(&self) -> bool {
        match self {
            Self::Metal(_) => true,
            Self::Cuda(_) => false,
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

    fn copy_in_bytes(
        &mut self,
        buffer: &DeviceHandle,
        bytes: &[u8],
        dtype: DeviceDataType,
    ) -> HostResult<()> {
        match self {
            Self::Metal(session) => session.copy_in_bytes(metal_handle(buffer)?, bytes, dtype),
            // Transitional DSB-1 behavior: CUDA still routes bytes through
            // its f32 surface. DSB-2 replaces this reinterpretation with the
            // raw-byte driver path.
            Self::Cuda(session) => {
                let _ = dtype;
                if bytes.len() % 4 != 0 {
                    return Err(HostError::invalid_args(format!(
                        "CUDA invocation-state copy requires a 4-byte multiple, got {} bytes",
                        bytes.len()
                    )));
                }
                let values: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                session.copy_in_f32(cuda_handle(buffer)?, &values)
            }
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
        let bindings = whole_handle_bindings(buffers)?;
        self.launch_kernel_bound(module, entry, &bindings, grid, block)
    }

    fn sync(&mut self) -> HostResult<()> {
        match self {
            Self::Metal(session) => session.sync(),
            Self::Cuda(session) => session.sync(),
        }
    }

    fn readback_bytes(
        &mut self,
        buffer: &DeviceHandle,
        dtype: DeviceDataType,
    ) -> HostResult<Vec<u8>> {
        match self {
            Self::Metal(session) => session.readback_bytes(metal_handle(buffer)?, dtype),
            // Transitional DSB-1 behavior: preserve the existing CUDA f32
            // readback until DSB-2 can use the driver's raw-byte result.
            Self::Cuda(session) => {
                let _ = dtype;
                let values = session.readback_f32(cuda_handle(buffer)?)?;
                Ok(values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect())
            }
        }
    }

    fn release(&mut self, handle: &DeviceHandle) -> HostResult<()> {
        match self {
            Self::Metal(session) => session.release(metal_handle(handle)?),
            Self::Cuda(session) => session.release(cuda_handle(handle)?),
        }
    }
}

fn dispatch_kernel(
    runtime: &mut DeviceRuntime,
    module: &DeviceHandle,
    entry: &str,
    buffers: &[DeviceHandle],
    grid: [u32; 3],
    block: [u32; 3],
) -> HostResult<()> {
    match runtime {
        DeviceRuntime::Metal(session) => {
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
        DeviceRuntime::Cuda(session) => {
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

fn whole_handle_bindings(buffers: &[DeviceHandle]) -> HostResult<Vec<DeviceLaunchBinding>> {
    buffers
        .iter()
        .enumerate()
        .map(|(index, handle)| {
            let binding_index = u32::try_from(index).map_err(|_| {
                errors::descriptor("launch buffer count exceeds a u32 binding index")
            })?;
            DeviceLaunchBinding::whole_handle(*handle, binding_index)
        })
        .collect()
}

fn validate_dispatch_bindings(
    backend: DeviceBackend,
    bindings: &[DeviceLaunchBinding],
) -> HostResult<()> {
    for binding in bindings {
        if binding.handle.backend != backend {
            return Err(device_invalid_handle(&binding.handle));
        }
        if binding.view_span == 0 {
            return Err(errors::descriptor(format!(
                "launch binding index {} has a zero view span",
                binding.binding_index
            )));
        }
        let Some(capacity) = binding.handle.len_bytes() else {
            return Err(device_invalid_handle(&binding.handle));
        };
        let Some(end) = binding.byte_offset.checked_add(binding.view_span) else {
            return Err(errors::shape_mismatch(format!(
                "launch binding index {} overflows its static envelope",
                binding.binding_index
            )));
        };
        if end > capacity {
            return Err(errors::shape_mismatch(format!(
                "launch binding index {} spans {} bytes from offset {} but the allocation is {capacity} bytes",
                binding.binding_index, binding.view_span, binding.byte_offset
            )));
        }
        if backend == DeviceBackend::Cuda && binding.is_cuda_dynamic() {
            return Err(cuda_dynamic_unsupported(binding));
        }
    }
    Ok(())
}

fn validate_allocation_map(
    kv: &KvCacheDescriptor,
    allocations: &[(u32, DeviceHandle)],
) -> HostResult<()> {
    let mut seen: Vec<u32> = Vec::with_capacity(allocations.len());
    for (allocation_id, handle) in allocations {
        if seen.contains(allocation_id) {
            return Err(errors::descriptor(format!(
                "allocation map repeats identity {allocation_id}"
            )));
        }
        seen.push(*allocation_id);
        let Some(allocation) = kv
            .allocations
            .iter()
            .find(|allocation| allocation.buffer_id == *allocation_id)
        else {
            return Err(errors::descriptor(format!(
                "allocation map names unknown allocation {allocation_id}"
            )));
        };
        let Some(len_bytes) = handle.len_bytes() else {
            return Err(device_invalid_handle(handle));
        };
        if len_bytes != allocation.capacity_bytes {
            return Err(errors::shape_mismatch(format!(
                "allocation {allocation_id} handle is {len_bytes} bytes but descriptor capacity is {} bytes",
                allocation.capacity_bytes
            )));
        }
    }
    Ok(())
}

fn resolve_one_binding(
    kv: &KvCacheDescriptor,
    allocations: &[(u32, DeviceHandle)],
    record: &DescriptorLaunchBinding,
) -> HostResult<DeviceLaunchBinding> {
    let handle = handle_for_allocation(allocations, record.handle)?;
    let binding = DeviceLaunchBinding {
        handle,
        binding_index: record.binding_index,
        byte_offset: record.byte_offset,
        view_span: record.view_span,
        runtime_source: record.runtime_source,
    };
    if matching_view(kv, record.handle, &binding).is_none() {
        return Err(errors::shape_mismatch(format!(
            "launch binding index {} offset {} span {} does not match a view on allocation {}",
            record.binding_index, record.byte_offset, record.view_span, record.handle
        )));
    }
    Ok(binding)
}

fn matching_view<'a>(
    kv: &'a KvCacheDescriptor,
    allocation_id: u32,
    binding: &DeviceLaunchBinding,
) -> Option<&'a DescriptorView> {
    kv.views.iter().find(|view| {
        view.allocation_id == allocation_id
            && view.static_base == binding.byte_offset
            && view.maximum_span >= binding.view_span
    })
}

fn handle_for_allocation(
    allocations: &[(u32, DeviceHandle)],
    allocation_id: u32,
) -> HostResult<DeviceHandle> {
    allocations
        .iter()
        .find_map(|(id, handle)| (*id == allocation_id).then_some(*handle))
        .ok_or_else(|| {
            errors::descriptor(format!(
                "launch binding names allocation {allocation_id} with no live handle"
            ))
        })
}

fn allocation_id_for_handle(
    allocations: &[(u32, DeviceHandle)],
    handle: &DeviceHandle,
) -> HostResult<u32> {
    let mut found = None;
    for (allocation_id, mapped) in allocations {
        if mapped == handle {
            if found.is_some() {
                return Err(errors::descriptor(format!(
                    "device handle {} is mapped to more than one allocation",
                    handle.id
                )));
            }
            found = Some(*allocation_id);
        }
    }
    found.ok_or_else(|| {
        errors::descriptor(format!(
            "device handle {} is not in the allocation map",
            handle.id
        ))
    })
}

fn cuda_dynamic_unsupported(binding: &DeviceLaunchBinding) -> HostError {
    HostError {
        code: E_CUDA_UNSUPPORTED.to_owned(),
        message: format!(
            "CUDA rejects KV-dynamic launch binding index {} (offset {}, source {}); this campaign does not bind it at offset zero",
            binding.binding_index,
            binding.byte_offset,
            runtime_source_spelling(binding.runtime_source)
        ),
        retryable: false,
    }
}

fn runtime_source_spelling(source: DescriptorRuntimeSource) -> &'static str {
    match source {
        DescriptorRuntimeSource::Constant => "constant",
        DescriptorRuntimeSource::Position => "position",
        DescriptorRuntimeSource::ValidLenAfter => "valid_len_after",
        DescriptorRuntimeSource::QueryRows => "query_rows",
        DescriptorRuntimeSource::SequenceEpoch => "sequence_epoch",
    }
}

fn metal_from_device_binding(binding: &DeviceLaunchBinding) -> HostResult<MetalLaunchBinding> {
    Ok(MetalLaunchBinding {
        handle: metal_handle(&binding.handle)?,
        binding_index: binding.binding_index,
        byte_offset: binding.byte_offset,
        view_span: binding.view_span,
    })
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
