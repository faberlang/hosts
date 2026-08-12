//! Typed ordering and arithmetic over arena-owned LLVM arrays.

use super::array::{write_value, RuntimeArray, RuntimeValue};
use super::RuntimeContext;
use radix_host_abi::{FaberRtStatusV1, FaberRtValueKindV1, STATUS_INVALID_ARGUMENT, STATUS_OK,
    STATUS_PANIC, VALUE_KIND_F32, VALUE_KIND_F64, VALUE_KIND_I16, VALUE_KIND_I32, VALUE_KIND_I64,
    VALUE_KIND_I8, VALUE_KIND_U16, VALUE_KIND_U32, VALUE_KIND_U64, VALUE_KIND_U8,
};
use crate::abi::FaberRtContextV1;
use std::cmp::Ordering;
use std::ffi::c_void;
use std::panic::{self, AssertUnwindSafe};

#[no_mangle]
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

#[no_mangle]
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
    match array.kind {
        VALUE_KIND_I8 => array.values.sort_by(compare_i8),
        VALUE_KIND_I16 => array.values.sort_by(compare_i16),
        VALUE_KIND_I32 => array.values.sort_by(compare_i32),
        VALUE_KIND_I64 => array.values.sort_by(compare_i64),
        VALUE_KIND_U8 => array.values.sort_by(compare_u8),
        VALUE_KIND_U16 => array.values.sort_by(compare_u16),
        VALUE_KIND_U32 => array.values.sort_by(compare_u32),
        VALUE_KIND_U64 => array.values.sort_by(compare_u64),
        VALUE_KIND_F32 => array.values.sort_by(compare_f32),
        VALUE_KIND_F64 => array.values.sort_by(compare_f64),
        _ => return false,
    }
    true
}

fn sum_array(array: &RuntimeArray) -> Option<RuntimeValue> {
    macro_rules! wrapping_sum {
        ($variant:ident, $zero:expr) => {{
            let mut total = $zero;
            for value in &array.values {
                let RuntimeValue::$variant(value) = value else {
                    return None;
                };
                total = total.wrapping_add(*value);
            }
            RuntimeValue::$variant(total)
        }};
    }
    Some(match array.kind {
        VALUE_KIND_I8 => wrapping_sum!(I8, 0_i8),
        VALUE_KIND_I16 => wrapping_sum!(I16, 0_i16),
        VALUE_KIND_I32 => wrapping_sum!(I32, 0_i32),
        VALUE_KIND_I64 => wrapping_sum!(I64, 0_i64),
        VALUE_KIND_U8 => wrapping_sum!(U8, 0_u8),
        VALUE_KIND_U16 => wrapping_sum!(U16, 0_u16),
        VALUE_KIND_U32 => wrapping_sum!(U32, 0_u32),
        VALUE_KIND_U64 => wrapping_sum!(U64, 0_u64),
        VALUE_KIND_F32 => {
            RuntimeValue::F32(array.values.iter().try_fold(0.0_f32, |total, value| {
                let RuntimeValue::F32(value) = value else {
                    return None;
                };
                Some(total + value)
            })?)
        }
        VALUE_KIND_F64 => {
            RuntimeValue::F64(array.values.iter().try_fold(0.0_f64, |total, value| {
                let RuntimeValue::F64(value) = value else {
                    return None;
                };
                Some(total + value)
            })?)
        }
        _ => return None,
    })
}

macro_rules! integer_comparator {
    ($name:ident, $variant:ident) => {
        fn $name(left: &RuntimeValue, right: &RuntimeValue) -> Ordering {
            match (left, right) {
                (RuntimeValue::$variant(left), RuntimeValue::$variant(right)) => left.cmp(right),
                _ => Ordering::Equal,
            }
        }
    };
}

integer_comparator!(compare_i8, I8);
integer_comparator!(compare_i16, I16);
integer_comparator!(compare_i32, I32);
integer_comparator!(compare_i64, I64);
integer_comparator!(compare_u8, U8);
integer_comparator!(compare_u16, U16);
integer_comparator!(compare_u32, U32);
integer_comparator!(compare_u64, U64);

fn compare_f32(left: &RuntimeValue, right: &RuntimeValue) -> Ordering {
    match (left, right) {
        (RuntimeValue::F32(left), RuntimeValue::F32(right)) => left.total_cmp(right),
        _ => Ordering::Equal,
    }
}

fn compare_f64(left: &RuntimeValue, right: &RuntimeValue) -> Ordering {
    match (left, right) {
        (RuntimeValue::F64(left), RuntimeValue::F64(right)) => left.total_cmp(right),
        _ => Ordering::Equal,
    }
}

unsafe fn runtime_mut<'a>(context: *mut FaberRtContextV1) -> Option<&'a mut RuntimeContext> {
    (!context.is_null()).then(|| unsafe { &mut *context.cast::<RuntimeContext>() })
}

fn find_array(runtime: &RuntimeContext, handle: *mut c_void) -> Option<&RuntimeArray> {
    runtime
        .arrays
        .iter()
        .find(|array| std::ptr::eq(array.as_ref(), handle.cast_const().cast::<RuntimeArray>()))
        .map(super::StableBox::as_ref)
}

fn find_array_mut(runtime: &mut RuntimeContext, handle: *mut c_void) -> Option<&mut RuntimeArray> {
    runtime
        .arrays
        .iter_mut()
        .find(|array| std::ptr::eq(array.as_ref(), handle.cast_const().cast::<RuntimeArray>()))
        .map(super::StableBox::as_mut)
}

fn ffi_status(operation: impl FnOnce() -> FaberRtStatusV1) -> FaberRtStatusV1 {
    panic::catch_unwind(AssertUnwindSafe(operation)).unwrap_or(STATUS_PANIC)
}
