//! First-order Unicode text queries and transformations for LLVM-host handles.

use super::RuntimeContext;
use super::array::{RuntimeValue, store_array};
use super::format::{store_text, text_value};
use crate::abi::FaberRtContextV1;
use crate::abi::{
    FaberRtPtrResultV1, FaberRtSliceV1, FaberRtStatusV1, STATUS_INVALID_ARGUMENT, STATUS_OK,
    STATUS_PANIC,
};
use radix_host_abi::{
    VALUE_KIND_F32, VALUE_KIND_F64, VALUE_KIND_I8, VALUE_KIND_I16, VALUE_KIND_I32, VALUE_KIND_I64,
    VALUE_KIND_PTR, VALUE_KIND_U8, VALUE_KIND_U16, VALUE_KIND_U32, VALUE_KIND_U64,
};
use std::ffi::{CStr, c_char, c_void};
use std::panic::{self, AssertUnwindSafe};

fn ffi_ptr_result(operation: impl FnOnce() -> FaberRtPtrResultV1) -> FaberRtPtrResultV1 {
    panic::catch_unwind(AssertUnwindSafe(operation))
        .unwrap_or(FaberRtPtrResultV1::failure(STATUS_PANIC))
}

fn ffi_status(operation: impl FnOnce() -> FaberRtStatusV1) -> FaberRtStatusV1 {
    panic::catch_unwind(AssertUnwindSafe(operation)).unwrap_or(STATUS_PANIC)
}

fn query(
    context: *mut FaberRtContextV1,
    out: *mut u8,
    operation: impl FnOnce() -> Option<bool>,
) -> FaberRtStatusV1 {
    panic::catch_unwind(AssertUnwindSafe(|| {
        if context.is_null() || out.is_null() {
            return STATUS_INVALID_ARGUMENT;
        }
        let Some(value) = operation() else {
            return STATUS_INVALID_ARGUMENT;
        };
        unsafe { *out = u8::from(value) };
        STATUS_OK
    }))
    .unwrap_or(STATUS_PANIC)
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_is_empty(
    context: *mut FaberRtContextV1,
    text: *const FaberRtSliceV1,
    out: *mut u8,
) -> FaberRtStatusV1 {
    query(context, out, || Some(text_value(text)?.is_empty()))
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_concat(
    context: *mut FaberRtContextV1,
    first: *const FaberRtSliceV1,
    second: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    let Some(first) = text_value(first) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    let Some(second) = text_value(second) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    ffi_ptr_result(|| store_text(context, format!("{first}{second}")))
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_contains(
    context: *mut FaberRtContextV1,
    text: *const FaberRtSliceV1,
    needle: *const FaberRtSliceV1,
    out: *mut u8,
) -> FaberRtStatusV1 {
    query(context, out, || {
        Some(text_value(text)?.contains(&text_value(needle)?))
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_starts_with(
    context: *mut FaberRtContextV1,
    text: *const FaberRtSliceV1,
    prefix: *const FaberRtSliceV1,
    out: *mut u8,
) -> FaberRtStatusV1 {
    query(context, out, || {
        Some(text_value(text)?.starts_with(&text_value(prefix)?))
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_ends_with(
    context: *mut FaberRtContextV1,
    text: *const FaberRtSliceV1,
    suffix: *const FaberRtSliceV1,
    out: *mut u8,
) -> FaberRtStatusV1 {
    query(context, out, || {
        Some(text_value(text)?.ends_with(&text_value(suffix)?))
    })
}

fn transform(
    context: *mut FaberRtContextV1,
    text: *const FaberRtSliceV1,
    operation: impl FnOnce(String) -> String,
) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| {
        let Some(text) = text_value(text) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_text(context, operation(text))
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_uppercase(
    context: *mut FaberRtContextV1,
    text: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    transform(context, text, |text| text.to_uppercase())
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_lowercase(
    context: *mut FaberRtContextV1,
    text: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    transform(context, text, |text| text.to_lowercase())
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_trim(
    context: *mut FaberRtContextV1,
    text: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    transform(context, text, |text| text.trim().to_owned())
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_slice(
    context: *mut FaberRtContextV1,
    text: *const FaberRtSliceV1,
    start: i64,
    end: i64,
) -> FaberRtPtrResultV1 {
    if start < 0 || end < 0 {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    }
    transform(context, text, |text| {
        let start = usize::try_from(start).unwrap_or(usize::MAX);
        let end = usize::try_from(end).unwrap_or(usize::MAX);
        text.chars()
            .skip(start)
            .take(end.saturating_sub(start))
            .collect()
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_replace(
    context: *mut FaberRtContextV1,
    text: *const FaberRtSliceV1,
    old: *const FaberRtSliceV1,
    new: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    let Some(old) = text_value(old) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    let Some(new) = text_value(new) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    transform(context, text, |text| text.replace(&old, &new))
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_split(
    context: *mut FaberRtContextV1,
    text: *const FaberRtSliceV1,
    separator: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| {
        let Some(text) = text_value(text) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(separator) = text_value(separator) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        if context.is_null() {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let mut values = Vec::new();
        for part in text.split(&separator) {
            let result = store_text(context, part.to_owned());
            if result.status != STATUS_OK {
                return result;
            }
            values.push(RuntimeValue::Ptr(result.value));
        }
        let runtime = unsafe { &mut *context.cast::<RuntimeContext>() };
        store_array(runtime, VALUE_KIND_PTR, values)
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_parse_integer(
    context: *mut FaberRtContextV1,
    text: *const FaberRtSliceV1,
    radix: u32,
    kind: u32,
    out: *mut c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        if context.is_null() || out.is_null() || !matches!(radix, 2 | 8 | 10 | 16) {
            return STATUS_INVALID_ARGUMENT;
        }
        let Some(text) = text_value(text) else {
            return STATUS_INVALID_ARGUMENT;
        };
        macro_rules! parse {
            ($ty:ty) => {
                match <$ty>::from_str_radix(&text, radix) {
                    Ok(value) => unsafe { out.cast::<$ty>().write(value) },
                    Err(_) => return STATUS_INVALID_ARGUMENT,
                }
            };
        }
        match kind {
            VALUE_KIND_I8 => parse!(i8),
            VALUE_KIND_I16 => parse!(i16),
            VALUE_KIND_I32 => parse!(i32),
            VALUE_KIND_I64 => parse!(i64),
            VALUE_KIND_U8 => parse!(u8),
            VALUE_KIND_U16 => parse!(u16),
            VALUE_KIND_U32 => parse!(u32),
            VALUE_KIND_U64 => parse!(u64),
            _ => return STATUS_INVALID_ARGUMENT,
        }
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_parse_float(
    context: *mut FaberRtContextV1,
    text: *const FaberRtSliceV1,
    kind: u32,
    out: *mut c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        if context.is_null() || out.is_null() {
            return STATUS_INVALID_ARGUMENT;
        }
        let Some(text) = text_value(text) else {
            return STATUS_INVALID_ARGUMENT;
        };
        match kind {
            VALUE_KIND_F32 => match text.parse::<f32>() {
                Ok(value) => unsafe { out.cast::<f32>().write(value) },
                Err(_) => return STATUS_INVALID_ARGUMENT,
            },
            VALUE_KIND_F64 => match text.parse::<f64>() {
                Ok(value) => unsafe { out.cast::<f64>().write(value) },
                Err(_) => return STATUS_INVALID_ARGUMENT,
            },
            _ => return STATUS_INVALID_ARGUMENT,
        }
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_truthy(
    context: *mut FaberRtContextV1,
    text: *const FaberRtSliceV1,
    out: *mut u8,
) -> FaberRtStatusV1 {
    query(context, out, || Some(!text_value(text)?.is_empty()))
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_ascii_truthy(
    context: *mut FaberRtContextV1,
    ascii: *const c_char,
    out: *mut u8,
) -> FaberRtStatusV1 {
    query(context, out, || {
        if ascii.is_null() {
            return None;
        }
        Some(!unsafe { CStr::from_ptr(ascii) }.to_bytes().is_empty())
    })
}

/// `textus ≡ textus` value equality (`i1`), comparing the UTF-8 payloads of
/// two text handles. Both literal descriptors (`{ ptr, i64 }` globals) and
/// arena `RuntimeText` handles share the leading slice layout, so
/// [`text_value`] resolves either representation.
///
/// L15: the emitter previously lowered `Eq`/`NotEq` on `textus` to raw LLVM
/// pointer equality, which is only valid for interned literals — a runtime
/// text handle (from `↦ textus`, valor extraction, instans rendering, …) is a
/// fresh arena allocation, so `adfirma runtimeText ≡ "literal"` false-failed,
/// latched `STATUS_PANIC`, and the L9 exit-struct fix surfaced a nonzero exit
/// code. Value equality restores Rust-oracle parity.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_eq(
    lhs: *const FaberRtSliceV1,
    rhs: *const FaberRtSliceV1,
) -> u8 {
    u8::from(match (text_value(lhs), text_value(rhs)) {
        (Some(lhs), Some(rhs)) => lhs == rhs,
        _ => false,
    })
}

/// `textus ≠ textus` value inequality (`i1`).
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_ne(
    lhs: *const FaberRtSliceV1,
    rhs: *const FaberRtSliceV1,
) -> u8 {
    u8::from(match (text_value(lhs), text_value(rhs)) {
        (Some(lhs), Some(rhs)) => lhs != rhs,
        _ => false,
    })
}

/// `ascii ≡ ascii` value equality (`i1`) over null-terminated ASCII byte
/// payloads (same rationale as [`__faber_rt_v1_text_eq`]).
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_ascii_eq(lhs: *const c_char, rhs: *const c_char) -> u8 {
    u8::from(match (ascii_bytes(lhs), ascii_bytes(rhs)) {
        (Some(lhs), Some(rhs)) => lhs == rhs,
        _ => false,
    })
}

/// `ascii ≠ ascii` value inequality (`i1`).
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_ascii_ne(lhs: *const c_char, rhs: *const c_char) -> u8 {
    u8::from(match (ascii_bytes(lhs), ascii_bytes(rhs)) {
        (Some(lhs), Some(rhs)) => lhs != rhs,
        _ => false,
    })
}

/// Read a null-terminated ASCII byte payload (fail-closed on a null pointer).
fn ascii_bytes<'a>(value: *const c_char) -> Option<&'a [u8]> {
    if value.is_null() {
        return None;
    }
    // SAFETY: the v1 ABI contract requires `ascii` handles to be live
    // null-terminated byte payloads for the process lifetime.
    Some(unsafe { CStr::from_ptr(value) }.to_bytes())
}
