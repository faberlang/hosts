//! Arena-owned typed maps and sets for the LLVM host ABI.

use super::array::{find_array, read_value, store_array, valid_kind, write_value, RuntimeValue};
use super::format::text_value;
use super::option::store_option;
use super::RuntimeContext;
use crate::abi::FaberRtContextV1;
use crate::abi::{
    FaberRtPtrResultV1, FaberRtSliceV1, FaberRtStatusV1, STATUS_INVALID_ARGUMENT, STATUS_OK,
    STATUS_PANIC,
};
use radix_host_abi::{FaberRtValueKindV1, VALUE_KIND_I64, VALUE_KIND_TEXT};
use std::ffi::c_void;
use std::panic::{self, AssertUnwindSafe};

pub(super) struct RuntimeMap {
    pub(super) key_kind: FaberRtValueKindV1,
    pub(super) value_kind: FaberRtValueKindV1,
    pub(super) entries: Vec<(RuntimeValue, RuntimeValue)>,
}

pub(super) struct RuntimeSet {
    pub(super) kind: FaberRtValueKindV1,
    pub(super) values: Vec<RuntimeValue>,
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_map_new(
    context: *mut FaberRtContextV1,
    key_kind: FaberRtValueKindV1,
    value_kind: FaberRtValueKindV1,
) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        if !valid_kind(key_kind) || !valid_kind(value_kind) {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        store_map(runtime, key_kind, value_kind, Vec::new())
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_map_put(
    context: *mut FaberRtContextV1,
    map: *mut c_void,
    key_kind: FaberRtValueKindV1,
    key: *const c_void,
    value_kind: FaberRtValueKindV1,
    value: *const c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(key) = (unsafe { read_value(key_kind, key) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(value) = (unsafe { read_value(value_kind, value) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(map) = find_map_mut(runtime, map) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if map.key_kind != key_kind || map.value_kind != value_kind {
            return STATUS_INVALID_ARGUMENT;
        }
        if let Some((_, existing)) = map
            .entries
            .iter_mut()
            .find(|(candidate, _)| values_equal(key_kind, *candidate, key))
        {
            *existing = value;
        } else {
            map.entries.push((key, value));
        }
        STATUS_OK
    })
}

/// Set one entry of a `tabula<textus, numerus>` map through a direct handle.
///
/// The generic aggregate index-assignment path (`aggregate[text_key] ← i64`)
/// emits `set_index(aggregate, key, value)` with a text key and an i64 value
/// when the destination aggregate is not a recognized collection carrier. The
/// aggregate handle must be a live map created by this runtime with text keys
/// and i64 values; unrecognized shapes fail closed by leaving the map
/// unchanged.
///
/// # Safety
///
/// `aggregate` must be a live map handle created by this runtime; `key` must
/// be a readable text descriptor.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_aggregate_set_index_ptr_i64(
    aggregate: *mut c_void,
    key: *const FaberRtSliceV1,
    value: i64,
) {
    let _ = panic::catch_unwind(AssertUnwindSafe(|| {
        let Some(_key_text) = text_value(key) else {
            return;
        };
        // SAFETY: per the v1 ABI contract, `aggregate` is a live map handle
        // created by this runtime; the handle is a stable box pointer.
        let Some(map) = (unsafe { (aggregate as *mut RuntimeMap).as_mut() }) else {
            return;
        };
        if map.key_kind != VALUE_KIND_TEXT || map.value_kind != VALUE_KIND_I64 {
            return;
        }
        let key = RuntimeValue::Ptr(key.cast::<c_void>().cast_mut());
        if let Some((_, existing)) = map
            .entries
            .iter_mut()
            .find(|(candidate, _)| values_equal(VALUE_KIND_TEXT, *candidate, key))
        {
            *existing = RuntimeValue::I64(value);
        } else {
            map.entries.push((key, RuntimeValue::I64(value)));
        }
    }));
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_map_option(
    context: *mut FaberRtContextV1,
    map: *mut c_void,
    key_kind: FaberRtValueKindV1,
    key: *const c_void,
    value_kind: FaberRtValueKindV1,
) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(key) = (unsafe { read_value(key_kind, key) }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let value = {
            let Some(map) = find_map(runtime, map) else {
                return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
            };
            if map.key_kind != key_kind || map.value_kind != value_kind {
                return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
            }
            map.entries
                .iter()
                .find(|(candidate, _)| values_equal(key_kind, *candidate, key))
                .map(|(_, value)| *value)
        };
        store_option(runtime, value_kind, value)
    })
}

/// Non-optional tabula index: write the value or fail closed when missing.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_map_get(
    context: *mut FaberRtContextV1,
    map: *mut c_void,
    key_kind: FaberRtValueKindV1,
    key: *const c_void,
    value_kind: FaberRtValueKindV1,
    output: *mut c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(key) = (unsafe { read_value(key_kind, key) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(map) = find_map(runtime, map) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if map.key_kind != key_kind || map.value_kind != value_kind || output.is_null() {
            return STATUS_INVALID_ARGUMENT;
        }
        let Some((_, value)) = map
            .entries
            .iter()
            .find(|(candidate, _)| values_equal(key_kind, *candidate, key))
        else {
            return STATUS_INVALID_ARGUMENT;
        };
        if !(unsafe { write_value(*value, output) }) {
            return STATUS_INVALID_ARGUMENT;
        }
        STATUS_OK
    })
}

macro_rules! map_query {
    ($name:ident, $predicate:expr) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            context: *mut FaberRtContextV1,
            map: *mut c_void,
            key_kind: FaberRtValueKindV1,
            key: *const c_void,
            output: *mut u8,
        ) -> FaberRtStatusV1 {
            ffi_status(|| {
                let Some(runtime) = (unsafe { runtime_mut(context) }) else {
                    return STATUS_INVALID_ARGUMENT;
                };
                let Some(key) = (unsafe { read_value(key_kind, key) }) else {
                    return STATUS_INVALID_ARGUMENT;
                };
                let Some(map) = find_map_mut(runtime, map) else {
                    return STATUS_INVALID_ARGUMENT;
                };
                if map.key_kind != key_kind || output.is_null() {
                    return STATUS_INVALID_ARGUMENT;
                }
                let value = ($predicate)(map, key);
                unsafe { output.write(u8::from(value)) };
                STATUS_OK
            })
        }
    };
}

map_query!(__faber_rt_v1_map_contains, |map: &mut RuntimeMap, key| map
    .entries
    .iter()
    .any(|(candidate, _)| values_equal(map.key_kind, *candidate, key)));
map_query!(__faber_rt_v1_map_delete, |map: &mut RuntimeMap, key| {
    let before = map.entries.len();
    map.entries
        .retain(|(candidate, _)| !values_equal(map.key_kind, *candidate, key));
    map.entries.len() != before
});

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_map_length(
    context: *mut FaberRtContextV1,
    map: *mut c_void,
    output: *mut i64,
) -> FaberRtStatusV1 {
    map_size_query(context, map, output, |map| {
        i64::try_from(map.entries.len()).ok()
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_map_is_empty(
    context: *mut FaberRtContextV1,
    map: *mut c_void,
    output: *mut u8,
) -> FaberRtStatusV1 {
    map_size_query(context, map, output, |map| {
        Some(u8::from(map.entries.is_empty()))
    })
}

fn map_size_query<T: Copy>(
    context: *mut FaberRtContextV1,
    map: *mut c_void,
    output: *mut T,
    operation: impl FnOnce(&RuntimeMap) -> Option<T>,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(map) = find_map(runtime, map) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(value) = operation(map) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if output.is_null() {
            return STATUS_INVALID_ARGUMENT;
        }
        unsafe { output.write(value) };
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_map_keys(
    context: *mut FaberRtContextV1,
    map: *mut c_void,
) -> FaberRtPtrResultV1 {
    map_values(context, map, true)
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_map_values(
    context: *mut FaberRtContextV1,
    map: *mut c_void,
) -> FaberRtPtrResultV1 {
    map_values(context, map, false)
}

fn map_values(context: *mut FaberRtContextV1, map: *mut c_void, keys: bool) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let (kind, values) = {
            let Some(map) = find_map(runtime, map) else {
                return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
            };
            if keys {
                (
                    map.key_kind,
                    map.entries.iter().map(|(key, _)| *key).collect(),
                )
            } else {
                (
                    map.value_kind,
                    map.entries.iter().map(|(_, value)| *value).collect(),
                )
            }
        };
        // L19: the snapshot array must be stored with the element kind the
        // array consumers pass (`array_get`/`array_set` reject a kind
        // mismatch). The emitter canonicalizes every pointer-carried element
        // (textus/ascii/valor/instans/octeti/regex …) to VALUE_KIND_PTR, so a
        // `tabula<textus, T>` key snapshot stored with the raw map key kind
        // (VALUE_KIND_TEXT) made `itera de <tabula>` fail every element read
        // (STATUS_INVALID_ARGUMENT) — no keys printed and a nonzero exit code
        // latched. The map entries themselves are stored as `RuntimeValue::Ptr`
        // for these kinds, so the PTR snapshot reads them identically.
        let kind = canonical_array_element_kind(kind);
        store_array(runtime, kind, values)
    })
}

/// Canonical element kind used by array consumers for a map key/value kind.
///
/// Pointer-carried kinds (textus, ascii, valor, instans, octeti, regex …)
/// cross the array ABI as `VALUE_KIND_PTR` handles; numeric kinds pass
/// through unchanged.
fn canonical_array_element_kind(
    kind: radix_host_abi::FaberRtValueKindV1,
) -> radix_host_abi::FaberRtValueKindV1 {
    match kind {
        radix_host_abi::VALUE_KIND_TEXT
        | radix_host_abi::VALUE_KIND_ASCII
        | radix_host_abi::VALUE_KIND_VALOR
        | radix_host_abi::VALUE_KIND_OPTION_I64
        | radix_host_abi::VALUE_KIND_INSTANS => radix_host_abi::VALUE_KIND_PTR,
        other => other,
    }
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_set_new(
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
        store_set(runtime, kind, Vec::new())
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_set_add(
    context: *mut FaberRtContextV1,
    set: *mut c_void,
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
        let Some(set) = find_set_mut(runtime, set) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if set.kind != kind {
            return STATUS_INVALID_ARGUMENT;
        }
        if !contains(set.kind, &set.values, value) {
            set.values.push(value);
        }
        STATUS_OK
    })
}

/// `lista ↦ copia` — dedupe array elements into a set of the same value kind.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_set_from_array(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(array) = find_array(runtime, array) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let kind = array.kind;
        let mut values = Vec::new();
        for value in &array.values {
            if !contains(kind, &values, value) {
                values.push(value);
            }
        }
        store_set(runtime, kind, values)
    })
}

/// `copia ↦ lista` — materialize set members into an array (unordered).
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_array_from_set(
    context: *mut FaberRtContextV1,
    set: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(set) = find_set(runtime, set) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_array(runtime, set.kind, set.values.clone())
    })
}

macro_rules! set_query {
    ($name:ident, $remove:expr) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            context: *mut FaberRtContextV1,
            set: *mut c_void,
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
                let Some(set) = find_set_mut(runtime, set) else {
                    return STATUS_INVALID_ARGUMENT;
                };
                if set.kind != kind || output.is_null() {
                    return STATUS_INVALID_ARGUMENT;
                }
                let present = contains(set.kind, &set.values, value);
                if $remove {
                    set.values
                        .retain(|candidate| !values_equal(set.kind, *candidate, value))
                }
                unsafe { output.write(u8::from(present)) };
                STATUS_OK
            })
        }
    };
}

set_query!(__faber_rt_v1_set_contains, false);
set_query!(__faber_rt_v1_set_delete, true);

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_set_length(
    context: *mut FaberRtContextV1,
    set: *mut c_void,
    output: *mut i64,
) -> FaberRtStatusV1 {
    set_size_query(context, set, output, |set| {
        i64::try_from(set.values.len()).ok()
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_set_is_empty(
    context: *mut FaberRtContextV1,
    set: *mut c_void,
    output: *mut u8,
) -> FaberRtStatusV1 {
    set_size_query(context, set, output, |set| {
        Some(u8::from(set.values.is_empty()))
    })
}

fn set_size_query<T: Copy>(
    context: *mut FaberRtContextV1,
    set: *mut c_void,
    output: *mut T,
    operation: impl FnOnce(&RuntimeSet) -> Option<T>,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(set) = find_set(runtime, set) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(value) = operation(set) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if output.is_null() {
            return STATUS_INVALID_ARGUMENT;
        }
        unsafe { output.write(value) };
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_set_union(
    context: *mut FaberRtContextV1,
    left: *mut c_void,
    right: *mut c_void,
) -> FaberRtPtrResultV1 {
    set_algebra(context, left, right, |kind, left, right| {
        left.iter()
            .chain(right)
            .copied()
            .fold(Vec::new(), |values, value| push_unique(kind, values, value))
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_set_intersection(
    context: *mut FaberRtContextV1,
    left: *mut c_void,
    right: *mut c_void,
) -> FaberRtPtrResultV1 {
    set_algebra(context, left, right, |kind, left, right| {
        left.iter()
            .filter(|value| contains(kind, right, **value))
            .copied()
            .collect()
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_set_difference(
    context: *mut FaberRtContextV1,
    left: *mut c_void,
    right: *mut c_void,
) -> FaberRtPtrResultV1 {
    set_algebra(context, left, right, |kind, left, right| {
        left.iter()
            .filter(|value| !contains(kind, right, **value))
            .copied()
            .collect()
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_set_symmetric_difference(
    context: *mut FaberRtContextV1,
    left: *mut c_void,
    right: *mut c_void,
) -> FaberRtPtrResultV1 {
    set_algebra(context, left, right, |kind, left, right| {
        left.iter()
            .filter(|value| !contains(kind, right, **value))
            .chain(right.iter().filter(|value| !contains(kind, left, **value)))
            .copied()
            .collect()
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_set_is_subset(
    context: *mut FaberRtContextV1,
    left: *mut c_void,
    right: *mut c_void,
    output: *mut u8,
) -> FaberRtStatusV1 {
    set_relation(context, left, right, output, |kind, left, right| {
        left.iter().all(|value| contains(kind, right, *value))
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_set_is_superset(
    context: *mut FaberRtContextV1,
    left: *mut c_void,
    right: *mut c_void,
    output: *mut u8,
) -> FaberRtStatusV1 {
    set_relation(context, left, right, output, |kind, left, right| {
        right.iter().all(|value| contains(kind, left, *value))
    })
}

fn set_algebra(
    context: *mut FaberRtContextV1,
    left: *mut c_void,
    right: *mut c_void,
    operation: impl FnOnce(FaberRtValueKindV1, &[RuntimeValue], &[RuntimeValue]) -> Vec<RuntimeValue>,
) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(left) = find_set(runtime, left) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(right) = find_set(runtime, right) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        if left.kind != right.kind {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let kind = left.kind;
        let values = operation(kind, &left.values, &right.values);
        store_set(runtime, kind, values)
    })
}

fn set_relation(
    context: *mut FaberRtContextV1,
    left: *mut c_void,
    right: *mut c_void,
    output: *mut u8,
    relation: impl FnOnce(FaberRtValueKindV1, &[RuntimeValue], &[RuntimeValue]) -> bool,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(left) = find_set(runtime, left) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(right) = find_set(runtime, right) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if left.kind != right.kind || output.is_null() {
            return STATUS_INVALID_ARGUMENT;
        }
        unsafe { output.write(u8::from(relation(left.kind, &left.values, &right.values))) };
        STATUS_OK
    })
}

fn push_unique(
    kind: FaberRtValueKindV1,
    mut values: Vec<RuntimeValue>,
    value: RuntimeValue,
) -> Vec<RuntimeValue> {
    if !contains(kind, &values, value) {
        values.push(value);
    }
    values
}

fn contains(kind: FaberRtValueKindV1, values: &[RuntimeValue], value: RuntimeValue) -> bool {
    values
        .iter()
        .any(|candidate| values_equal(kind, *candidate, value))
}

fn values_equal(kind: FaberRtValueKindV1, left: RuntimeValue, right: RuntimeValue) -> bool {
    if kind != VALUE_KIND_TEXT {
        return left == right;
    }
    let (RuntimeValue::Ptr(left), RuntimeValue::Ptr(right)) = (left, right) else {
        return false;
    };
    text_bytes(left)
        .zip(text_bytes(right))
        .is_some_and(|(left, right)| left == right)
}

fn text_bytes<'a>(text: *mut c_void) -> Option<&'a [u8]> {
    let text = unsafe { text.cast::<FaberRtSliceV1>().as_ref() }?;
    let len = usize::try_from(text.len).ok()?;
    if len > 0 && text.data.is_null() {
        return None;
    }
    Some(if len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(text.data, len) }
    })
}

pub(super) fn store_map(
    runtime: &mut RuntimeContext,
    key_kind: FaberRtValueKindV1,
    value_kind: FaberRtValueKindV1,
    entries: Vec<(RuntimeValue, RuntimeValue)>,
) -> FaberRtPtrResultV1 {
    let map = super::StableBox::new(RuntimeMap {
        key_kind,
        value_kind,
        entries,
    });
    let handle = map.handle();
    runtime.maps.push(map);
    FaberRtPtrResultV1::success(handle)
}

pub(super) fn store_set(
    runtime: &mut RuntimeContext,
    kind: FaberRtValueKindV1,
    values: Vec<RuntimeValue>,
) -> FaberRtPtrResultV1 {
    let set = super::StableBox::new(RuntimeSet { kind, values });
    let handle = set.handle();
    runtime.sets.push(set);
    FaberRtPtrResultV1::success(handle)
}

pub(super) fn find_map(runtime: &RuntimeContext, handle: *mut c_void) -> Option<&RuntimeMap> {
    runtime
        .maps
        .iter()
        .find(|map| std::ptr::eq(map.as_ref(), handle.cast_const().cast()))
        .map(super::StableBox::as_ref)
}
fn find_map_mut(runtime: &mut RuntimeContext, handle: *mut c_void) -> Option<&mut RuntimeMap> {
    runtime
        .maps
        .iter_mut()
        .find(|map| std::ptr::eq(map.as_ref(), handle.cast_const().cast()))
        .map(super::StableBox::as_mut)
}
pub(super) fn find_set(runtime: &RuntimeContext, handle: *mut c_void) -> Option<&RuntimeSet> {
    runtime
        .sets
        .iter()
        .find(|set| std::ptr::eq(set.as_ref(), handle.cast_const().cast()))
        .map(super::StableBox::as_ref)
}
fn find_set_mut(runtime: &mut RuntimeContext, handle: *mut c_void) -> Option<&mut RuntimeSet> {
    runtime
        .sets
        .iter_mut()
        .find(|set| std::ptr::eq(set.as_ref(), handle.cast_const().cast()))
        .map(super::StableBox::as_mut)
}
unsafe fn runtime_mut<'a>(context: *mut FaberRtContextV1) -> Option<&'a mut RuntimeContext> {
    (!context.is_null()).then(|| unsafe { &mut *context.cast::<RuntimeContext>() })
}
fn ffi_status(operation: impl FnOnce() -> FaberRtStatusV1) -> FaberRtStatusV1 {
    panic::catch_unwind(AssertUnwindSafe(operation)).unwrap_or(STATUS_PANIC)
}
fn ffi_ptr_result(operation: impl FnOnce() -> FaberRtPtrResultV1) -> FaberRtPtrResultV1 {
    panic::catch_unwind(AssertUnwindSafe(operation))
        .unwrap_or(FaberRtPtrResultV1::failure(STATUS_PANIC))
}
