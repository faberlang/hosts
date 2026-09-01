//! Precision-aware instans conversion through arena-owned handles.

use super::RuntimeContext;
use super::convert::with_valor;
use super::format::{store_text, text_value};
use crate::abi::FaberRtContextV1;
use crate::abi::{FaberRtPtrResultV1, FaberRtSliceV1, STATUS_INVALID_ARGUMENT, STATUS_PANIC};
use faber::{Instans, InstansPraecisio, Valor};
use radix_host_abi::{
    FaberRtInstansPrecisionV1, INSTANS_PRECISION_MICROS, INSTANS_PRECISION_MILLIS,
    INSTANS_PRECISION_NANOS, INSTANS_PRECISION_SECONDS,
};
use std::ffi::c_void;
use std::panic::{self, AssertUnwindSafe};

fn ffi(operation: impl FnOnce() -> FaberRtPtrResultV1) -> FaberRtPtrResultV1 {
    panic::catch_unwind(AssertUnwindSafe(operation))
        .unwrap_or(FaberRtPtrResultV1::failure(STATUS_PANIC))
}

fn runtime(context: *mut FaberRtContextV1) -> Option<&'static mut RuntimeContext> {
    (!context.is_null()).then(|| unsafe { &mut *context.cast::<RuntimeContext>() })
}

fn precision(value: FaberRtInstansPrecisionV1) -> Option<InstansPraecisio> {
    match value {
        INSTANS_PRECISION_SECONDS => Some(InstansPraecisio::Secunda),
        INSTANS_PRECISION_MILLIS => Some(InstansPraecisio::Millisecunda),
        INSTANS_PRECISION_MICROS => Some(InstansPraecisio::Microsecunda),
        INSTANS_PRECISION_NANOS => Some(InstansPraecisio::Nanosecunda),
        _ => None,
    }
}

fn store(runtime: &mut RuntimeContext, value: Instans) -> FaberRtPtrResultV1 {
    let value = super::StableBox::new(value);
    let handle = value.handle();
    runtime.instants.push(value);
    FaberRtPtrResultV1::success(handle)
}

pub(super) fn find(runtime: &RuntimeContext, handle: *mut c_void) -> Option<Instans> {
    runtime
        .instants
        .iter()
        .find(|value| std::ptr::eq(value.as_ref(), handle.cast()))
        .map(|value| **value)
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_instans_from_text(
    context: *mut FaberRtContextV1,
    value: *const FaberRtSliceV1,
    requested: FaberRtInstansPrecisionV1,
) -> FaberRtPtrResultV1 {
    ffi(|| {
        let (Some(runtime), Some(text), Some(requested)) =
            (runtime(context), text_value(value), precision(requested))
        else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(value) = Instans::try_from_valor(&Valor::Textus(text), requested) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store(runtime, value)
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_instans_from_valor(
    context: *mut FaberRtContextV1,
    valor: *const Valor,
    requested: FaberRtInstansPrecisionV1,
) -> FaberRtPtrResultV1 {
    ffi(|| {
        let Some(requested) = precision(requested) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(value) = with_valor(context, valor, |valor| {
            Instans::try_from_valor(valor, requested)
        }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store(runtime, value)
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_instans_retag(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
    requested: FaberRtInstansPrecisionV1,
) -> FaberRtPtrResultV1 {
    ffi(|| {
        let (Some(runtime), Some(requested)) = (runtime(context), precision(requested)) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(value) = find(runtime, handle) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store(runtime, value.ad_praecisionem(requested))
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_instans_get_text(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(value) = find(runtime, handle) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_text(context, value.to_rfc3339())
    })
}

/// Render a raw `instans` value in the Rust oracle's Debug shape
/// (`Instans { nanos: …, praecisio: … }`) as a text handle.
///
/// A `nota`/`mone`/`scribe`/`vide` of a raw `instans` value renders with
/// `faber::Instans`'s `Debug` in the Rust lane (L19 `conversio/fallibilis`:
/// `nota inlineRecovery(good), tutum(good), tutumDirect(good)` prints
/// `Instans { nanos: …, praecisio: … }`), NOT the RFC3339 wire text that the
/// `↦ textus` conversion produces. The diagnostic path uses this symbol;
/// `instans_get_text` stays for the explicit text conversion.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_instans_display(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(value) = find(runtime, handle) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_text(context, format!("{value:?}"))
    })
}

/// Ordered comparison of two opaque `instans` handles.
///
/// The v1 ABI contract requires both handles to be live `instans` handles
/// created by this runtime. The comparison reads the two [`Instans`] values
/// directly through the handle pointers (no context argument crosses this
/// ABI); the result is `1` when the predicate holds.
fn compare_instans(
    lhs: *mut c_void,
    rhs: *mut c_void,
    predicate: impl Fn(&Instans, &Instans) -> bool,
) -> u8 {
    let (Some(lhs), Some(rhs)) = (
        // SAFETY: the v1 ABI contract requires both handles to be live
        // `instans` handles; reading them through the handle pointer is the
        // only way to compare without a context argument.
        unsafe { (lhs as *const Instans).as_ref() },
        unsafe { (rhs as *const Instans).as_ref() },
    ) else {
        return 0;
    };
    u8::from(predicate(lhs, rhs))
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_compare_lt_2_ptr_ptr_to_i1(
    lhs: *mut c_void,
    rhs: *mut c_void,
) -> u8 {
    compare_instans(lhs, rhs, |lhs, rhs| lhs < rhs)
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_compare_gt_2_ptr_ptr_to_i1(
    lhs: *mut c_void,
    rhs: *mut c_void,
) -> u8 {
    compare_instans(lhs, rhs, |lhs, rhs| lhs > rhs)
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_compare_lte_2_ptr_ptr_to_i1(
    lhs: *mut c_void,
    rhs: *mut c_void,
) -> u8 {
    compare_instans(lhs, rhs, |lhs, rhs| lhs <= rhs)
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_compare_gte_2_ptr_ptr_to_i1(
    lhs: *mut c_void,
    rhs: *mut c_void,
) -> u8 {
    compare_instans(lhs, rhs, |lhs, rhs| lhs >= rhs)
}

/// `instans ≡ instans` value equality (`i1`).
///
/// L15: `Eq`/`NotEq` on `instans` handles previously lowered to raw LLVM
/// pointer equality, which compares arena allocation identity rather than the
/// datetime value — two equal instants parsed at different times are distinct
/// handles, so `adfirma a ≡ b` false-failed, latched `STATUS_PANIC`, and the
/// L9 exit-struct fix surfaced a nonzero exit code. Value equality mirrors the
/// ordering family above and restores Rust-oracle parity.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_compare_eq_2_ptr_ptr_to_i1(
    lhs: *mut c_void,
    rhs: *mut c_void,
) -> u8 {
    compare_instans(lhs, rhs, |lhs, rhs| lhs == rhs)
}

/// `instans ≠ instans` value inequality (`i1`).
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_compare_ne_2_ptr_ptr_to_i1(
    lhs: *mut c_void,
    rhs: *mut c_void,
) -> u8 {
    compare_instans(lhs, rhs, |lhs, rhs| lhs != rhs)
}

/// Current wall-clock instant as an `instans<ns>` handle (`norma:time.nunc`).
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_tempus_nunc(
    context: *mut FaberRtContextV1,
) -> FaberRtPtrResultV1 {
    ffi(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        // SAFETY: nanoseconds since epoch fit in i64 for the foreseeable future.
        #[allow(clippy::cast_possible_truncation)]
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos() as i64);
        store(
            runtime,
            Instans::from_nanos(nanos, InstansPraecisio::Nanosecunda),
        )
    })
}
