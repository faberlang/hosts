//! Scalar conversion into runtime-owned opaque `valor` handles.

use super::format::{store_text, text_value};
use super::RuntimeContext;
use crate::abi::{
    FaberRtPtrResultV1, FaberRtSliceV1, FaberRtStatusV1, STATUS_INVALID_ARGUMENT, STATUS_OK,
    STATUS_PANIC,
};
use crate::abi::FaberRtContextV1;
use faber::{FromValor, Valor};
use std::ffi::{c_char, c_void, CStr};
use std::panic::{self, AssertUnwindSafe};

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_valor_i64(
    context: *mut FaberRtContextV1,
    value: i64,
) -> FaberRtPtrResultV1 {
    store_valor(context, Valor::Numerus(value))
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_valor_f64(
    context: *mut FaberRtContextV1,
    value: f64,
) -> FaberRtPtrResultV1 {
    store_valor(context, Valor::Fractus(value))
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_valor_i1(
    context: *mut FaberRtContextV1,
    value: u8,
) -> FaberRtPtrResultV1 {
    store_valor(context, Valor::Bivalens(value != 0))
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_valor_text(
    context: *mut FaberRtContextV1,
    value: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    let Some(value) = text_value(value) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    store_valor(context, Valor::Textus(value))
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_valor_ascii(
    context: *mut FaberRtContextV1,
    value: *const c_char,
) -> FaberRtPtrResultV1 {
    if value.is_null() {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    }
    let bytes = unsafe { CStr::from_ptr(value) }.to_bytes();
    if !bytes.is_ascii() {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    }
    store_valor(
        context,
        Valor::Textus(String::from_utf8_lossy(bytes).into_owned()),
    )
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_valor_nihil(
    context: *mut FaberRtContextV1,
) -> FaberRtPtrResultV1 {
    store_valor(context, Valor::Nihil)
}

pub(super) fn with_valor<T>(
    context: *mut FaberRtContextV1,
    value: *const Valor,
    operation: impl FnOnce(&Valor) -> Option<T>,
) -> Option<T> {
    if context.is_null() || value.is_null() {
        return None;
    }
    let runtime = unsafe { &*context.cast::<RuntimeContext>() };
    runtime
        .valors
        .iter()
        .find(|candidate| std::ptr::eq(candidate.as_ref(), value))
        .and_then(|candidate| operation(candidate))
}

fn extract<T: FromValor>(
    context: *mut FaberRtContextV1,
    value: *const Valor,
    out: *mut T,
) -> FaberRtStatusV1 {
    panic::catch_unwind(AssertUnwindSafe(|| {
        if out.is_null() {
            return STATUS_INVALID_ARGUMENT;
        }
        let Some(extracted) = with_valor(context, value, T::from_valor) else {
            return STATUS_INVALID_ARGUMENT;
        };
        unsafe { *out = extracted };
        STATUS_OK
    }))
    .unwrap_or(STATUS_PANIC)
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_valor_get_i64(
    context: *mut FaberRtContextV1,
    value: *const Valor,
    out: *mut i64,
) -> FaberRtStatusV1 {
    extract(context, value, out)
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_valor_get_f64(
    context: *mut FaberRtContextV1,
    value: *const Valor,
    out: *mut f64,
) -> FaberRtStatusV1 {
    extract(context, value, out)
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_valor_get_i1(
    context: *mut FaberRtContextV1,
    value: *const Valor,
    out: *mut u8,
) -> FaberRtStatusV1 {
    panic::catch_unwind(AssertUnwindSafe(|| {
        if out.is_null() {
            return STATUS_INVALID_ARGUMENT;
        }
        let Some(extracted) = with_valor(context, value, bool::from_valor) else {
            return STATUS_INVALID_ARGUMENT;
        };
        unsafe { *out = u8::from(extracted) };
        STATUS_OK
    }))
    .unwrap_or(STATUS_PANIC)
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_valor_get_text(
    context: *mut FaberRtContextV1,
    value: *const Valor,
) -> FaberRtPtrResultV1 {
    let Some(extracted) = with_valor(context, value, String::from_valor) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    store_text(context, extracted)
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_valor_get_ascii(
    context: *mut FaberRtContextV1,
    value: *const Valor,
) -> FaberRtPtrResultV1 {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        let Some(extracted) = with_valor(context, value, String::from_valor) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        if !extracted.is_ascii() || extracted.as_bytes().contains(&0) {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let runtime = unsafe { &mut *context.cast::<RuntimeContext>() };
        let mut bytes = extracted.into_bytes();
        bytes.push(0);
        let bytes = super::StableBox::from_box(bytes.into_boxed_slice());
        let pointer = bytes.as_ref().as_ptr().cast_mut().cast::<c_void>();
        runtime.ascii.push(bytes);
        FaberRtPtrResultV1::success(pointer)
    }));
    result.unwrap_or(FaberRtPtrResultV1::failure(STATUS_PANIC))
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_valor_get_nihil(
    context: *mut FaberRtContextV1,
    value: *const Valor,
) -> FaberRtStatusV1 {
    panic::catch_unwind(AssertUnwindSafe(|| {
        if with_valor(context, value, |value| Some(value.is_nihil())).unwrap_or(false) {
            STATUS_OK
        } else {
            STATUS_INVALID_ARGUMENT
        }
    }))
    .unwrap_or(STATUS_PANIC)
}

pub(super) fn store_valor(context: *mut FaberRtContextV1, value: Valor) -> FaberRtPtrResultV1 {
    panic::catch_unwind(AssertUnwindSafe(|| {
        if context.is_null() {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let runtime = unsafe { &mut *context.cast::<RuntimeContext>() };
        let value = super::StableBox::new(value);
        let handle = value.handle();
        runtime.valors.push(value);
        FaberRtPtrResultV1::success(handle)
    }))
    .unwrap_or(FaberRtPtrResultV1::failure(STATUS_PANIC))
}

/// Look up an arena-owned `valor` handle by pointer equality.
pub(super) fn find_valor(runtime: &RuntimeContext, handle: *mut c_void) -> Option<&Valor> {
    runtime
        .valors
        .iter()
        .find(|valor| std::ptr::eq(valor.as_ref(), handle.cast()))
        .map(super::StableBox::as_ref)
}

/// Re-tag one opaque runtime handle as another opaque carrier (`↦` between
/// carrier types that share the same runtime representation).
///
/// The LLVM emitter routes opaque `ptr → ptr` runtime conversions through this
/// symbol. The host represents json values and `valor` values with the same
/// [`Valor`] storage, and union boxes as opaque handle pointers, so the
/// conversion is a pointer-preserving re-interpretation: the returned handle
/// is the input handle. A null or dangling handle fails closed.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_convert_runtime_1_ptr_to_ptr(
    context: *mut FaberRtContextV1,
    value: *mut c_void,
) -> FaberRtPtrResultV1 {
    panic::catch_unwind(AssertUnwindSafe(|| {
        if context.is_null() || value.is_null() {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        FaberRtPtrResultV1::success(value)
    }))
    .unwrap_or(FaberRtPtrResultV1::failure(STATUS_PANIC))
}
