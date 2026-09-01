//! Mutable opaque octeti operations and text conversion.

use super::RuntimeContext;
use super::array::RuntimeValue;
use super::format::{store_text, text_value};
use super::option::store_option;
use super::valor_aggregate::{find_octeti, store_octeti};
use crate::abi::FaberRtContextV1;
use crate::abi::{
    FaberRtPtrResultV1, FaberRtSliceV1, FaberRtStatusV1, STATUS_INVALID_ARGUMENT, STATUS_OK,
    STATUS_PANIC,
};
use radix_host_abi::VALUE_KIND_U8;
use std::ffi::{CStr, c_char, c_void};
use std::panic::{self, AssertUnwindSafe};

fn ffi_ptr(operation: impl FnOnce() -> FaberRtPtrResultV1) -> FaberRtPtrResultV1 {
    panic::catch_unwind(AssertUnwindSafe(operation))
        .unwrap_or(FaberRtPtrResultV1::failure(STATUS_PANIC))
}

fn ffi_status(operation: impl FnOnce() -> FaberRtStatusV1) -> FaberRtStatusV1 {
    panic::catch_unwind(AssertUnwindSafe(operation)).unwrap_or(STATUS_PANIC)
}

fn runtime(context: *mut FaberRtContextV1) -> Option<&'static mut RuntimeContext> {
    (!context.is_null()).then(|| unsafe { &mut *context.cast::<RuntimeContext>() })
}

/// The byte payload for an `octeti` handle.
///
/// `octeti` is type-identical to `lista<numerus<u8>>` (`octeti/unify.fab`:
/// "representation-identical and bidirectionally assignable"), and the
/// cross-assignment crosses the arena as a plain handle — the constructed
/// `lista<numerus<u8>>` handle is stored unchanged into the `octeti` slot.
/// The octeti ABI therefore resolves the handle through EITHER the octeti
/// list or a `VALUE_KIND_U8` array list.
fn octeti_bytes<'a>(runtime: &'a RuntimeContext, handle: *mut c_void) -> Option<Vec<u8>> {
    if let Some(bytes) = find_octeti(runtime, handle) {
        return Some(bytes.clone());
    }
    let array = super::array::find_array(runtime, handle)?;
    if array.kind != VALUE_KIND_U8 {
        return None;
    }
    let mut out = Vec::with_capacity(array.values.len());
    for value in &array.values {
        match value {
            super::array::RuntimeValue::U8(byte) => out.push(byte),
            _ => return None,
        }
    }
    Some(out)
}

/// Append one byte to an `octeti` handle (octeti arena or U8 array arena).
fn octeti_push(runtime: &mut RuntimeContext, handle: *mut c_void, byte: u8) -> bool {
    if let Some(bytes) = runtime
        .octeti
        .iter_mut()
        .find(|bytes| std::ptr::eq(bytes.as_ref(), handle.cast()))
    {
        bytes.push(byte);
        return true;
    }
    let Some(array) = super::array::find_array_mut(runtime, handle) else {
        return false;
    };
    if array.kind != VALUE_KIND_U8 {
        return false;
    }
    array.values.push(super::array::RuntimeValue::U8(byte));
    true
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_octeti_append(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
    value: u8,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = runtime(context) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if !octeti_push(runtime, handle, value) {
            return STATUS_INVALID_ARGUMENT;
        }
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_octeti_get(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
    index: i64,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let value = usize::try_from(index)
            .ok()
            .and_then(|index| octeti_bytes(runtime, handle)?.get(index).copied());
        store_option(runtime, VALUE_KIND_U8, value.map(RuntimeValue::U8))
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_octeti_length(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
    output: *mut i64,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = runtime(context) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(length) =
            octeti_bytes(runtime, handle).and_then(|bytes| i64::try_from(bytes.len()).ok())
        else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(output) = (unsafe { output.as_mut() }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        *output = length;
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_octeti_from_text(
    context: *mut FaberRtContextV1,
    value: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let (Some(runtime), Some(value)) = (runtime(context), text_value(value)) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_octeti(runtime, value.into_bytes())
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_octeti_from_ascii(
    context: *mut FaberRtContextV1,
    value: *const c_char,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        if value.is_null() {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let bytes = unsafe { CStr::from_ptr(value) }.to_bytes();
        if !bytes.is_ascii() {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        store_octeti(runtime, bytes.to_vec())
    })
}

fn decode(context: *mut FaberRtContextV1, handle: *mut c_void, ascii: bool) -> FaberRtPtrResultV1 {
    let Some(runtime) = runtime(context) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    let Some(bytes) = find_octeti(runtime, handle) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    if (ascii && !bytes.is_ascii()) || std::str::from_utf8(bytes).is_err() {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    }
    if ascii {
        let mut owned = bytes.clone();
        owned.push(0);
        let owned = owned.into_boxed_slice();
        let pointer = owned.as_ptr().cast_mut().cast();
        runtime.ascii.push(super::StableBox::from_box(owned));
        FaberRtPtrResultV1::success(pointer)
    } else {
        store_text(context, String::from_utf8_lossy(bytes).into_owned())
    }
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_octeti_get_text(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| decode(context, handle, false))
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_octeti_get_ascii(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| decode(context, handle, true))
}
