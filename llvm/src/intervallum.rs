//! Arena-owned `intervallum<numerus>` carrier for the LLVM host ABI (Stage 4AC).
//!
//! v1 owns i64 bounds plus inclusivity. Algebra that may be empty returns an
//! option handle of a pointer payload (`VALUE_KIND_PTR`) so `vel`/coalesce and
//! `est nihil` share one encoding.

use super::array::{store_array, RuntimeValue};
use super::option::store_option;
use super::tensor::store_tensor;
use super::RuntimeContext;
use radix_host_abi::{FaberRtPtrResultV1, FaberRtStatusV1, STATUS_INVALID_ARGUMENT, STATUS_OK,
    STATUS_PANIC, VALUE_KIND_I64, VALUE_KIND_PTR,
};
use crate::abi::FaberRtContextV1;
use faber::{Intervallum, IntervallumKind};
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

fn store_interval(runtime: &mut RuntimeContext, value: Intervallum<i64>) -> FaberRtPtrResultV1 {
    let boxed = super::StableBox::new(value);
    let handle = boxed.handle();
    runtime.intervals.push(boxed);
    FaberRtPtrResultV1::success(handle)
}

fn find_interval(runtime: &RuntimeContext, handle: *mut c_void) -> Option<&Intervallum<i64>> {
    runtime
        .intervals
        .iter()
        .find(|value| std::ptr::eq(value.as_ref(), handle.cast()))
        .map(super::StableBox::as_ref)
}

fn kind_from_flag(inclusive: i32) -> Option<IntervallumKind> {
    match inclusive {
        0 => Some(IntervallumKind::Exclusive),
        1 => Some(IntervallumKind::Inclusive),
        _ => None,
    }
}

fn store_optional_interval(
    runtime: &mut RuntimeContext,
    value: Option<Intervallum<i64>>,
) -> FaberRtPtrResultV1 {
    match value {
        Some(interval) => {
            let stored = store_interval(runtime, interval);
            if stored.status != STATUS_OK {
                return stored;
            }
            store_option(
                runtime,
                VALUE_KIND_PTR,
                Some(RuntimeValue::Ptr(stored.value)),
            )
        }
        None => store_option(runtime, VALUE_KIND_PTR, None),
    }
}

/// Construct `intervallum<numerus>`: `inclusive` is 0 (‥) or 1 (…).
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_interval_new(
    context: *mut FaberRtContextV1,
    initium: i64,
    finis: i64,
    inclusive: i32,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let (Some(runtime), Some(kind)) = (runtime(context), kind_from_flag(inclusive)) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let interval = match kind {
            IntervallumKind::Exclusive => Intervallum::exclusive(initium, finis),
            IntervallumKind::Inclusive => Intervallum::inclusive(initium, finis),
        };
        store_interval(runtime, interval)
    })
}

/// Interval intersection; empty results are option-none of ptr payload.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_interval_intersect(
    context: *mut FaberRtContextV1,
    left: *mut c_void,
    right: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let (Some(left), Some(right)) =
            (find_interval(runtime, left), find_interval(runtime, right))
        else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_optional_interval(runtime, left.inter(*right))
    })
}

/// Interval union when overlap or adjacent; empty results are option-none.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_interval_union(
    context: *mut FaberRtContextV1,
    left: *mut c_void,
    right: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let (Some(left), Some(right)) =
            (find_interval(runtime, left), find_interval(runtime, right))
        else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_optional_interval(runtime, left.union(*right))
    })
}

/// Discrete span count (`longitudo`) for numerus intervals.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_interval_length(
    context: *mut FaberRtContextV1,
    interval: *mut c_void,
    output: *mut i64,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = runtime(context) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(interval) = find_interval(runtime, interval) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if output.is_null() {
            return STATUS_INVALID_ARGUMENT;
        }
        unsafe { output.write(interval.longitudo()) };
        STATUS_OK
    })
}

/// Point containment (`continet` / `intra`).
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_interval_contains(
    context: *mut FaberRtContextV1,
    interval: *mut c_void,
    value: i64,
    output: *mut u8,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = runtime(context) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(interval) = find_interval(runtime, interval) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if output.is_null() {
            return STATUS_INVALID_ARGUMENT;
        }
        unsafe { output.write(u8::from(interval.continet(&value))) };
        STATUS_OK
    })
}

/// Clamp a numerus value into an interval (refinement-target conversio).
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_interval_clamp_i64(
    context: *mut FaberRtContextV1,
    value: i64,
    interval: *mut c_void,
    output: *mut i64,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = runtime(context) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(interval) = find_interval(runtime, interval) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if output.is_null() {
            return STATUS_INVALID_ARGUMENT;
        }
        unsafe { output.write(interval.coercere(value)) };
        STATUS_OK
    })
}

/// Range-to-range clamp: each bound of `source` coerced into `target`.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_interval_clamp(
    context: *mut FaberRtContextV1,
    source: *mut c_void,
    target: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let (Some(source), Some(target)) = (
            find_interval(runtime, source),
            find_interval(runtime, target),
        ) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_interval(runtime, source.coercere_intervallum(target))
    })
}

/// Materialize interval values into a lista of numerus.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_interval_materialize_array(
    context: *mut FaberRtContextV1,
    interval: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(interval) = find_interval(runtime, interval) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let values = interval
            .ad_lista()
            .into_iter()
            .map(RuntimeValue::I64)
            .collect();
        store_array(runtime, VALUE_KIND_I64, values)
    })
}

/// Materialize interval values into a 1-d numerus tensor.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_interval_materialize_tensor(
    context: *mut FaberRtContextV1,
    interval: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(interval) = find_interval(runtime, interval) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let values = interval.ad_lista();
        let len = i64::try_from(values.len()).unwrap_or(0);
        let data = values.into_iter().map(RuntimeValue::I64).collect();
        store_tensor(runtime, VALUE_KIND_I64, vec![len], data)
    })
}
