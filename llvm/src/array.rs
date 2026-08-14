//! Arena-owned typed arrays for the LLVM host ABI.

use super::option::store_option;
use super::RuntimeContext;
use crate::abi::{
    FaberRtPtrResultV1, FaberRtStatusV1, STATUS_INVALID_ARGUMENT, STATUS_OK, STATUS_PANIC,
};
use radix_host_abi::{
    FaberRtArrayOptionModeV1, FaberRtArrayRangeModeV1, FaberRtValueKindV1, ARRAY_OPTION_FIRST, ARRAY_OPTION_INDEX, ARRAY_OPTION_LAST,
    ARRAY_OPTION_REMOVE_FIRST, ARRAY_OPTION_REMOVE_LAST, ARRAY_RANGE_DROP_FIRST, ARRAY_RANGE_SLICE,
    ARRAY_RANGE_TAKE, ARRAY_RANGE_TAKE_LAST,
    VALUE_KIND_ASCII, VALUE_KIND_F16, VALUE_KIND_F32, VALUE_KIND_F64, VALUE_KIND_I1,
    VALUE_KIND_I16, VALUE_KIND_I32, VALUE_KIND_I64, VALUE_KIND_I8, VALUE_KIND_INSTANS,
    VALUE_KIND_OPTION_I64, VALUE_KIND_PTR, VALUE_KIND_TEXT, VALUE_KIND_U16, VALUE_KIND_U32,
    VALUE_KIND_U64, VALUE_KIND_U8, VALUE_KIND_VALOR,
};
use crate::abi::FaberRtContextV1;
use std::ffi::c_void;
use std::panic::{self, AssertUnwindSafe};

#[derive(Clone, Copy, PartialEq)]
pub(super) enum RuntimeValue {
    I1(u8),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F16(u16),
    F32(f32),
    F64(f64),
    Ptr(*mut c_void),
}

pub(super) struct RuntimeArray {
    pub(super) kind: FaberRtValueKindV1,
    pub(super) values: Vec<RuntimeValue>,
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_array_new(
    context: *mut FaberRtContextV1,
    kind: FaberRtValueKindV1,
) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        if !valid_kind(kind) {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        store_array(runtime, kind, Vec::new())
    })
}

pub(super) fn store_array(
    runtime: &mut RuntimeContext,
    kind: FaberRtValueKindV1,
    values: Vec<RuntimeValue>,
) -> FaberRtPtrResultV1 {
    let array = super::StableBox::new(RuntimeArray { kind, values });
    let handle = array.handle();
    runtime.arrays.push(array);
    FaberRtPtrResultV1::success(handle)
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_array_push(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
    kind: FaberRtValueKindV1,
    value: *const c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(array) = find_array_mut(runtime, array) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(value) = (unsafe { read_value(kind, value) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if array.kind != kind {
            return STATUS_INVALID_ARGUMENT;
        }
        array.values.push(value);
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_array_extend(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
    source: *mut c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(source_index) = find_array_index(runtime, source) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let source_kind = runtime.arrays[source_index].kind;
        let source_values = runtime.arrays[source_index].values.clone();
        let Some(array) = find_array_mut(runtime, array) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if array.kind != source_kind {
            return STATUS_INVALID_ARGUMENT;
        }
        array.values.extend(source_values);
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_array_length(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
    output: *mut i64,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(array) = find_array(runtime, array) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Ok(length) = i64::try_from(array.values.len()) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if !(unsafe { write_typed(output.cast(), length) }) {
            return STATUS_INVALID_ARGUMENT;
        }
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_array_get(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
    index: i64,
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
        let Ok(index) = usize::try_from(index) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(value) = array.values.get(index).copied() else {
            return STATUS_INVALID_ARGUMENT;
        };
        if array.kind != kind || !(unsafe { write_value(value, output) }) {
            return STATUS_INVALID_ARGUMENT;
        }
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_array_set(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
    index: i64,
    kind: FaberRtValueKindV1,
    value: *const c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(value) = (unsafe { read_value(kind, value) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(array) = find_array_mut(runtime, array) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Ok(index) = usize::try_from(index) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(slot) = array.values.get_mut(index) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if array.kind != kind {
            return STATUS_INVALID_ARGUMENT;
        }
        *slot = value;
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_array_clone(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(source_index) = find_array_index(runtime, array) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let kind = runtime.arrays[source_index].kind;
        let values = runtime.arrays[source_index].values.clone();
        store_array(runtime, kind, values)
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_array_contains(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
    kind: FaberRtValueKindV1,
    value: *const c_void,
    output: *mut u8,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(value) = (unsafe { read_value(kind, value) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(array) = find_array(runtime, array) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if array.kind != kind
            || !(unsafe { write_typed(output.cast(), u8::from(array.values.contains(&value))) })
        {
            return STATUS_INVALID_ARGUMENT;
        }
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_array_is_empty(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
    output: *mut u8,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(array) = find_array(runtime, array) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if !(unsafe { write_typed(output.cast(), u8::from(array.values.is_empty())) }) {
            return STATUS_INVALID_ARGUMENT;
        }
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_array_reverse(
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
        array.values.reverse();
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_array_range(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
    mode: FaberRtArrayRangeModeV1,
    first: i64,
    second: i64,
) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(source_index) = find_array_index(runtime, array) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let source = &runtime.arrays[source_index];
        let Some((start, end)) = range_bounds(mode, first, second, source.values.len()) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let kind = source.kind;
        let values = source.values[start..end].to_vec();
        store_array(runtime, kind, values)
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_array_option(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
    mode: FaberRtArrayOptionModeV1,
    index: i64,
) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(array_index) = find_array_index(runtime, array) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let array = &mut runtime.arrays[array_index];
        let kind = array.kind;
        let value = match mode {
            ARRAY_OPTION_INDEX => usize::try_from(index)
                .ok()
                .and_then(|index| array.values.get(index).copied()),
            ARRAY_OPTION_FIRST => array.values.first().copied(),
            ARRAY_OPTION_LAST => array.values.last().copied(),
            ARRAY_OPTION_REMOVE_FIRST => (!array.values.is_empty()).then(|| array.values.remove(0)),
            ARRAY_OPTION_REMOVE_LAST => array.values.pop(),
            _ => return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT),
        };
        store_option(runtime, kind, value)
    })
}

fn range_bounds(
    mode: FaberRtArrayRangeModeV1,
    first: i64,
    second: i64,
    len: usize,
) -> Option<(usize, usize)> {
    let clamp = |value: i64| usize::try_from(value).ok().map(|value| value.min(len));
    Some(match mode {
        ARRAY_RANGE_SLICE => {
            let end = clamp(second)?;
            let start = clamp(first)?.min(end);
            (start, end)
        }
        ARRAY_RANGE_TAKE => (0, clamp(first)?),
        ARRAY_RANGE_TAKE_LAST => (len.saturating_sub(clamp(first)?), len),
        ARRAY_RANGE_DROP_FIRST => (clamp(first)?, len),
        _ => return None,
    })
}

pub(super) fn valid_kind(kind: FaberRtValueKindV1) -> bool {
    matches!(
        kind,
        VALUE_KIND_I1
            | VALUE_KIND_I8
            | VALUE_KIND_I16
            | VALUE_KIND_I32
            | VALUE_KIND_I64
            | VALUE_KIND_U8
            | VALUE_KIND_U16
            | VALUE_KIND_U32
            | VALUE_KIND_U64
            | VALUE_KIND_F16
            | VALUE_KIND_F32
            | VALUE_KIND_F64
            | VALUE_KIND_PTR
            | VALUE_KIND_TEXT
            | VALUE_KIND_VALOR
            | VALUE_KIND_OPTION_I64
            | VALUE_KIND_INSTANS
            | VALUE_KIND_ASCII
    )
}

pub(super) unsafe fn runtime_mut<'a>(
    context: *mut FaberRtContextV1,
) -> Option<&'a mut RuntimeContext> {
    (!context.is_null()).then(|| unsafe { &mut *context.cast::<RuntimeContext>() })
}

pub(super) fn find_array(runtime: &RuntimeContext, handle: *mut c_void) -> Option<&RuntimeArray> {
    runtime
        .arrays
        .iter()
        .find(|array| std::ptr::eq(array.as_ref(), handle.cast_const().cast::<RuntimeArray>()))
        .map(super::StableBox::as_ref)
}

pub(super) fn find_array_mut(
    runtime: &mut RuntimeContext,
    handle: *mut c_void,
) -> Option<&mut RuntimeArray> {
    runtime
        .arrays
        .iter_mut()
        .find(|array| std::ptr::eq(array.as_ref(), handle.cast_const().cast::<RuntimeArray>()))
        .map(super::StableBox::as_mut)
}

fn find_array_index(runtime: &RuntimeContext, handle: *mut c_void) -> Option<usize> {
    runtime
        .arrays
        .iter()
        .position(|array| std::ptr::eq(array.as_ref(), handle.cast_const().cast::<RuntimeArray>()))
}

pub(super) unsafe fn read_value(
    kind: FaberRtValueKindV1,
    value: *const c_void,
) -> Option<RuntimeValue> {
    Some(match kind {
        VALUE_KIND_I1 => RuntimeValue::I1(unsafe { read_typed(value) }?),
        VALUE_KIND_I8 => RuntimeValue::I8(unsafe { read_typed(value) }?),
        VALUE_KIND_I16 => RuntimeValue::I16(unsafe { read_typed(value) }?),
        VALUE_KIND_I32 => RuntimeValue::I32(unsafe { read_typed(value) }?),
        VALUE_KIND_I64 => RuntimeValue::I64(unsafe { read_typed(value) }?),
        VALUE_KIND_U8 => RuntimeValue::U8(unsafe { read_typed(value) }?),
        VALUE_KIND_U16 => RuntimeValue::U16(unsafe { read_typed(value) }?),
        VALUE_KIND_U32 => RuntimeValue::U32(unsafe { read_typed(value) }?),
        VALUE_KIND_U64 => RuntimeValue::U64(unsafe { read_typed(value) }?),
        VALUE_KIND_F16 => RuntimeValue::F16(unsafe { read_typed(value) }?),
        VALUE_KIND_F32 => RuntimeValue::F32(unsafe { read_typed(value) }?),
        VALUE_KIND_F64 => RuntimeValue::F64(unsafe { read_typed(value) }?),
        VALUE_KIND_PTR
        | VALUE_KIND_TEXT
        | VALUE_KIND_VALOR
        | VALUE_KIND_OPTION_I64
        | VALUE_KIND_INSTANS
        | VALUE_KIND_ASCII => RuntimeValue::Ptr(unsafe { read_typed(value) }?),
        _ => return None,
    })
}

#[allow(clippy::similar_names)]
pub(super) unsafe fn write_value(value: RuntimeValue, output: *mut c_void) -> bool {
    match value {
        RuntimeValue::I1(value) => unsafe { write_typed(output, value) },
        RuntimeValue::I8(value) => unsafe { write_typed(output, value) },
        RuntimeValue::I16(value) => unsafe { write_typed(output, value) },
        RuntimeValue::I32(value) => unsafe { write_typed(output, value) },
        RuntimeValue::I64(value) => unsafe { write_typed(output, value) },
        RuntimeValue::U8(value) => unsafe { write_typed(output, value) },
        RuntimeValue::U16(value) => unsafe { write_typed(output, value) },
        RuntimeValue::U32(value) => unsafe { write_typed(output, value) },
        RuntimeValue::U64(value) => unsafe { write_typed(output, value) },
        RuntimeValue::F16(value) => unsafe { write_typed(output, value) },
        RuntimeValue::F32(value) => unsafe { write_typed(output, value) },
        RuntimeValue::F64(value) => unsafe { write_typed(output, value) },
        RuntimeValue::Ptr(value) => unsafe { write_typed(output, value) },
    }
}

#[allow(clippy::unnecessary_wraps)]
unsafe fn read_typed<T: Copy>(value: *const c_void) -> Option<T> {
    let value = value.cast::<T>();
    (!value.is_null() && value.is_aligned()).then(|| unsafe { value.read() })
}

unsafe fn write_typed<T>(output: *mut c_void, value: T) -> bool {
    let output = output.cast::<T>();
    if output.is_null() || !output.is_aligned() {
        return false;
    }
    unsafe { output.write(value) };
    true
}

fn ffi_status(operation: impl FnOnce() -> FaberRtStatusV1) -> FaberRtStatusV1 {
    panic::catch_unwind(AssertUnwindSafe(operation)).unwrap_or(STATUS_PANIC)
}

fn ffi_ptr_result(operation: impl FnOnce() -> FaberRtPtrResultV1) -> FaberRtPtrResultV1 {
    panic::catch_unwind(AssertUnwindSafe(operation))
        .unwrap_or(FaberRtPtrResultV1::failure(STATUS_PANIC))
}
