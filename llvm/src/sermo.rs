//! Sermo materialization host bindings (P8 — promotion packet
//! `sermo-runtime-surface`).
//!
//! The shared sermo ABI rows bind the faber-runtime `frame` sermo surface
//! (`sermo_open` / `sermo_set_opener` / `sermo_materialize_*`,
//! meus/tuus handles, recv/drain). `SermoOpen` returns an opaque
//! arena-owned stream handle (`frame::Sermo`); the materialization rows drain
//! the inbound frame stream to a carrier (`↦ textus`, `↦ valor`); the `_or`
//! row is the `⇥`-style recovery — a scalar materialization that substitutes
//! the fallback on a missing or wrong-typed payload instead of aborting (P6
//! per-carrier `_or` precedent). The route's host dispatch starts on first
//! consumption (builtin `runtime:*` routes and the registered host dispatch).

use super::RuntimeContext;
use super::convert::store_valor;
use super::format::{store_text, text_value};
use crate::abi::FaberRtContextV1;
use crate::abi::{
    FaberRtPtrResultV1, FaberRtSliceV1, FaberRtStatusV1, STATUS_INVALID_ARGUMENT, STATUS_PANIC,
};
use faber::Valor;
use faber::frame;
use std::ffi::c_void;
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

fn store_sermo(runtime: &mut RuntimeContext, sermo: frame::Sermo) -> FaberRtPtrResultV1 {
    let boxed = super::StableBox::new(sermo);
    let handle = boxed.handle();
    runtime.sermos.push(boxed);
    FaberRtPtrResultV1::success(handle)
}

fn find_sermo_mut(runtime: &mut RuntimeContext, handle: *mut c_void) -> Option<&mut frame::Sermo> {
    runtime
        .sermos
        .iter_mut()
        .find(|value| std::ptr::eq(value.as_ref(), handle.cast()))
        .map(super::StableBox::as_mut)
}

/// `ad 'route'(payload)` — open a sermo stream (SermoOpen).
///
/// Creates the `frame::Sermo` for `route`, installs `payload` as the opener
/// (the first request frame's data), and returns an opaque arena-owned stream
/// handle. The route's host dispatch starts on first consumption.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_sermo_open(
    context: *mut FaberRtContextV1,
    route: *const FaberRtSliceV1,
    payload: *const Valor,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let (Some(runtime), Some(route)) = (runtime(context), text_value(route)) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(payload) = (unsafe { payload.as_ref() }).cloned() else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let mut sermo = frame::sermo_open(&route);
        frame::sermo_set_opener(&mut sermo, payload);
        store_sermo(runtime, sermo)
    })
}

/// `sermo_set_opener(sermo, data)` — replace the first request frame's payload
/// before the stream is consumed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_sermo_set_opener(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
    data: *const Valor,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let (Some(runtime), Some(data)) = (runtime(context), unsafe { data.as_ref() }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(sermo) = find_sermo_mut(runtime, handle) else {
            return STATUS_INVALID_ARGUMENT;
        };
        frame::sermo_set_opener(sermo, data.clone());
        crate::abi::STATUS_OK
    })
}

/// `sermo ↦ textus` — materialize the inbound frame stream to a textus
/// carrier.
///
/// # Panics
///
/// Panics (caught by the ffi boundary → `STATUS_PANIC`) if the stream
/// produces a terminal error frame or a non-textus content frame.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_sermo_materialize_text(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(sermo) = find_sermo_mut(runtime, handle) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_text(context, frame::sermo_materialize_textus(sermo))
    })
}

/// `sermo ↦ valor` — materialize the inbound frame stream to a valor carrier.
///
/// # Panics
///
/// Panics (caught by the ffi boundary → `STATUS_PANIC`) if the stream
/// produces a terminal error frame.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_sermo_materialize_valor(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(sermo) = find_sermo_mut(runtime, handle) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_valor(context, frame::sermo_materialize_valor(sermo))
    })
}

/// `sermo ↦ i64 ⇥ fallback` — scalar materialization with `_or` recovery.
///
/// Extracts an `i64` from the stream; on a missing or wrong-typed payload the
/// `fallback` substitutes instead of aborting (P6 recovery precedent). The
/// result crosses as an arena-owned numeric box.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_sermo_materialize_i64_or(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
    fallback: i64,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(sermo) = find_sermo_mut(runtime, handle) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let value = frame::try_sermo_materialize_scalar::<i64>(sermo).unwrap_or(fallback);
        let boxed = super::StableBox::new(value);
        let handle = boxed.handle();
        runtime.numeric_boxes.push(boxed);
        FaberRtPtrResultV1::success(handle)
    })
}
