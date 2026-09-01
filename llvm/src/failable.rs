//! Shared failable error-carrier row (`__faber_rt_v1_fallible_error`).
//!
//! Status-first `(STATUS_FALLIBLE, payload)` for the `ReturnError` / `cape err`
//! channel. The LLVM emitter inlines typed failable aggregates; this host row
//! is the named ABI symbol for the same pair so rustc/link no longer miss
//! `radix_host_abi::FaberRt*` carriers on the dense staged-carrier path.

use crate::abi::{FaberRtContextV1, FaberRtPtrResultV1, STATUS_INVALID_ARGUMENT, STATUS_PANIC};
use std::ffi::c_void;
use std::panic::{self, AssertUnwindSafe};

fn ffi_ptr(operation: impl FnOnce() -> FaberRtPtrResultV1) -> FaberRtPtrResultV1 {
    panic::catch_unwind(AssertUnwindSafe(operation))
        .unwrap_or(FaberRtPtrResultV1::failure(STATUS_PANIC))
}

/// Pair a typed error payload with the failable error-channel status.
///
/// # Safety
///
/// `context` must be null or a live runtime context. `error` is an opaque
/// payload handle; it is never dereferenced.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_fallible_error(
    context: *mut FaberRtContextV1,
    error: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        if context.is_null() {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        FaberRtPtrResultV1::fallible(error)
    })
}
