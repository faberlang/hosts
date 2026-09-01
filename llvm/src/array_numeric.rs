//! Typed ordering and arithmetic over arena-owned LLVM arrays.

use super::RuntimeContext;
use super::array::{RuntimeArray, find_array, find_array_mut, write_value};
use crate::abi::FaberRtContextV1;
use crate::abi::{FaberRtStatusV1, STATUS_INVALID_ARGUMENT, STATUS_OK, STATUS_PANIC};
use radix_host_abi::FaberRtValueKindV1;
use std::ffi::c_void;
use std::panic::{self, AssertUnwindSafe};

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_array_sort(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(array) = find_array_mut(runtime, array) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if !sort_array(array) {
            return STATUS_INVALID_ARGUMENT;
        }
        STATUS_OK
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_array_sum(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
    kind: FaberRtValueKindV1,
    output: *mut c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(array) = find_array(runtime, array) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(value) = (array.kind == kind).then(|| sum_array(array)).flatten() else {
            return STATUS_INVALID_ARGUMENT;
        };
        if !(unsafe { write_value(value, output) }) {
            return STATUS_INVALID_ARGUMENT;
        }
        STATUS_OK
    })
}

fn sort_array(array: &mut RuntimeArray) -> bool {
    array.values.sort()
}

fn sum_array(array: &RuntimeArray) -> Option<super::array::RuntimeValue> {
    array.values.wrapping_sum()
}

unsafe fn runtime_mut<'a>(context: *mut FaberRtContextV1) -> Option<&'a mut RuntimeContext> {
    (!context.is_null()).then(|| unsafe { &mut *context.cast::<RuntimeContext>() })
}

fn ffi_status(operation: impl FnOnce() -> FaberRtStatusV1) -> FaberRtStatusV1 {
    panic::catch_unwind(AssertUnwindSafe(operation)).unwrap_or(STATUS_PANIC)
}
