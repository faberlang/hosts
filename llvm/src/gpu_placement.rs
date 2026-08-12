//! GPU placement ABI symbols for the LLVM host runtime.
//!
//! Implements [`__faber_gpu_v1_copy_in`], [`__faber_gpu_v1_readback`], and
//! [`__faber_gpu_v1_sync`] per the placement execution contract at
//! `radix/docs/design/placement-execution-contract.md`.
//!
//! ## Buffer model
//!
//! Simplified buffer model — bare logical ID, no generation counters, no
//! create-before-retire. Aligns with G-SPINE-07 when the unified host model
//! lands. Single-kernel proof does not exercise replacement, concurrent
//! retire, or multi-buffer identity.

use std::collections::HashMap;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Mutex;

use radix_host_abi::{
    FaberRtStatusV1, STATUS_INVALID_ARGUMENT, STATUS_IO_ERROR, STATUS_OK, STATUS_PANIC,
};

/// Device buffer map keyed by logical buffer ID.
///
/// Initialized on first use via `Mutex<Option<...>>`. The `Mutex` protects
/// the map from concurrent access; the `Option` defers allocation until the
/// first placement operation.
static DEVICE_BUFFERS: Mutex<Option<HashMap<u64, Vec<u8>>>> = Mutex::new(None);

fn ffi_status(operation: impl FnOnce() -> FaberRtStatusV1) -> FaberRtStatusV1 {
    panic::catch_unwind(AssertUnwindSafe(operation)).unwrap_or(STATUS_PANIC)
}

fn with_buffers<F>(f: F) -> Result<FaberRtStatusV1, FaberRtStatusV1>
where
    F: FnOnce(&mut HashMap<u64, Vec<u8>>) -> FaberRtStatusV1,
{
    let mut guard = DEVICE_BUFFERS.lock().map_err(|_| STATUS_PANIC)?;
    if guard.is_none() {
        *guard = Some(HashMap::new());
    }
    Ok(f(guard.as_mut().expect("guard initialized above")))
}

/// Copy host data into a device buffer identified by `logical_id`.
///
/// Allocates a `Vec<u8>` of `data_len` bytes and copies the host data into it.
/// An existing buffer at `logical_id` is overwritten (simplified model).
///
/// # Safety
///
/// `data_ptr` must be readable for `data_len` bytes, or null when `data_len`
/// is zero. `dtype` is reserved for future dtype validation.
#[no_mangle]
pub unsafe extern "C" fn __faber_gpu_v1_copy_in(
    logical_id: u64,
    data_ptr: *const u8,
    data_len: u64,
    _dtype: u32,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        if data_len > 0 && data_ptr.is_null() {
            return STATUS_INVALID_ARGUMENT;
        }
        let data_len = data_len as usize;
        let buffer = if data_len == 0 {
            Vec::new()
        } else {
            // SAFETY: caller guarantees `data_ptr` is readable for `data_len` bytes
            // (checked above for the null-pointer + positive-length case).
            unsafe { std::slice::from_raw_parts(data_ptr, data_len) }.to_vec()
        };
        match with_buffers(|buffers| {
            buffers.insert(logical_id, buffer);
            STATUS_OK
        }) {
            Ok(status) => status,
            Err(status) => status,
        }
    })
}

/// Copy a device buffer back to host memory.
///
/// Looks up the buffer by `logical_id`, copies its contents into `dest_ptr`,
/// and writes the actual byte length to `*actual_len`.
///
/// # Safety
///
/// `dest_ptr` must be writable for `dest_capacity` bytes. `actual_len` must
/// be a writable `u64` output slot. Both must be non-null.
#[no_mangle]
pub unsafe extern "C" fn __faber_gpu_v1_readback(
    logical_id: u64,
    dest_ptr: *mut u8,
    dest_capacity: u64,
    actual_len: *mut u64,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        if dest_ptr.is_null() || actual_len.is_null() {
            return STATUS_INVALID_ARGUMENT;
        }
        match with_buffers(|buffers| match buffers.get(&logical_id) {
            Some(buffer) => {
                let len = buffer.len() as u64;
                if len > dest_capacity {
                    return STATUS_IO_ERROR;
                }
                // SAFETY: `dest_ptr` is writable for `dest_capacity >= len` bytes
                // (checked above), and `buffer` is a valid source.
                unsafe {
                    std::ptr::copy_nonoverlapping(buffer.as_ptr(), dest_ptr, buffer.len());
                    *actual_len = len;
                }
                STATUS_OK
            }
            None => STATUS_INVALID_ARGUMENT,
        }) {
            Ok(status) => status,
            Err(status) => status,
        }
    })
}

/// Device-side synchronization barrier.
///
/// CPU-side LLVM execution is sequential; sync is a no-op because the device
/// queue is the CPU itself. This is honest device execution: the contract is
/// honored, the implementation reflects the hardware.
///
/// # Safety
///
/// `_logical_id` is reserved for future per-buffer barrier semantics.
/// This implementation ignores it.
#[no_mangle]
pub unsafe extern "C" fn __faber_gpu_v1_sync(_logical_id: u64) -> FaberRtStatusV1 {
    STATUS_OK
}
