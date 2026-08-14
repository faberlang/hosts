//! Crate-local C ABI carriers for the LLVM host runtime.
//!
//! Authority model (faber-target-runtime inventory §3.4): the stable C ABI is
//! compiler-owned — `radix-host-abi` owns the symbol names, status code
//! values, value kinds, and the LLVM layout spellings. The **runtime side**
//! owns the struct layouts: each runtime crate carries its own `#[repr(C)]`
//! copies matching those spellings (the faber/runtime/rust contract package
//! does the same for `FaberRtStatusV1`), so the two sides cannot drift.
//!
//! [`FaberRtContextV1`] is the opaque process-lifetime runtime context: only
//! pointers cross the ABI, so the struct is a zero-sized opaque carrier. The
//! historical definitions lived in `faber-runtime/src/host_abi.rs` (migration
//! carrier).

use core::ffi::c_void;

/// Opaque process-lifetime runtime context. Only pointers cross the ABI.
#[repr(C)]
pub struct FaberRtContextV1 {
    _private: [u8; 0],
    _alignment: [*mut c_void; 0],
}

/// Stable C ABI status carrier (mirror of `radix-host-abi` `FaberRtStatusV1`
/// status codes, which radix owns as code values): `code` is the status
/// discriminator.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaberRtStatusV1 {
    pub code: i32,
}

impl FaberRtStatusV1 {
    #[must_use]
    pub const fn is_ok(self) -> bool {
        self.code == STATUS_OK.code
    }
}

/// Status 0 — the happy path (success payload follows).
pub const STATUS_OK: FaberRtStatusV1 = FaberRtStatusV1 { code: 0 };

/// Status 1 — a caller-supplied argument is invalid.
pub const STATUS_INVALID_ARGUMENT: FaberRtStatusV1 = FaberRtStatusV1 { code: 1 };

/// Status 2 — an I/O operation failed.
pub const STATUS_IO_ERROR: FaberRtStatusV1 = FaberRtStatusV1 { code: 2 };

/// Status 3 — the host recovered from a panic.
pub const STATUS_PANIC: FaberRtStatusV1 = FaberRtStatusV1 { code: 3 };

/// Status 4 — the operation is not supported on this host.
pub const STATUS_UNSUPPORTED: FaberRtStatusV1 = FaberRtStatusV1 { code: 4 };

/// Byte-slice carrier (`%FaberRtSliceV1 = type { ptr, i64 }`). `data` points
/// to `len` readable bytes; a null `data` is allowed only when `len` is zero.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaberRtSliceV1 {
    pub data: *const u8,
    pub len: u64,
}

impl FaberRtSliceV1 {
    /// Build a slice over static bytes (test helpers and literal fixtures).
    #[must_use]
    pub const fn from_static(bytes: &'static [u8]) -> Self {
        Self {
            data: bytes.as_ptr(),
            len: bytes.len() as u64,
        }
    }
}

/// Pointer-result carrier (`%FaberRtPtrResultV1 = type { i32, ptr }`): status
/// plus an opaque handle (`c_void` pointer into a runtime arena or a static
/// literal descriptor).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaberRtPtrResultV1 {
    pub status: FaberRtStatusV1,
    pub value: *mut c_void,
}

impl FaberRtPtrResultV1 {
    /// Build the happy-path result: `STATUS_OK` plus the payload handle.
    #[must_use]
    pub const fn success(value: *mut c_void) -> Self {
        Self {
            status: STATUS_OK,
            value,
        }
    }

    /// Build the failure result: `status` plus a null handle.
    #[must_use]
    pub const fn failure(status: FaberRtStatusV1) -> Self {
        Self {
            status,
            value: core::ptr::null_mut(),
        }
    }
}

/// Exit carrier crossing the program-entry boundary as one register
/// (`%FaberRtExitV1 = type i64` carrying `process_code | (status.code << 32)`).
/// `process_code` occupies the low 32 bits, `status.code` the high 32 bits.
///
/// Only the non-test `main` entry constructs this; the test profile links the
/// runtime without the program entry.
#[repr(C)]
#[cfg_attr(test, allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaberRtExitV1 {
    pub process_code: i32,
    pub status: FaberRtStatusV1,
}
