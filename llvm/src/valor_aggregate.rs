//! Recursive opaque collection conversion for the LLVM host Valor ABI.

use super::array::{find_array, store_array, RuntimeArray, RuntimeValue};
use super::collection_map::{find_map, find_set, store_map, RuntimeMap};
use super::convert::{store_valor, with_valor};
use super::format::{find_text, store_text, text_value};
use super::tensor::find_tensor;
use super::RuntimeContext;
use crate::abi::FaberRtContextV1;
use crate::abi::{FaberRtPtrResultV1, FaberRtSliceV1, STATUS_INVALID_ARGUMENT, STATUS_PANIC};
use faber::{FromValor, Valor};
use radix_host_abi::{
    FaberRtValueKindV1, VALUE_KIND_F32, VALUE_KIND_F64, VALUE_KIND_I1, VALUE_KIND_I16,
    VALUE_KIND_I32, VALUE_KIND_I64, VALUE_KIND_I8, VALUE_KIND_PTR, VALUE_KIND_TEXT, VALUE_KIND_U16,
    VALUE_KIND_U32, VALUE_KIND_U64, VALUE_KIND_U8, VALUE_KIND_VALOR,
};
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::panic::{self, AssertUnwindSafe};

fn ffi_ptr(operation: impl FnOnce() -> FaberRtPtrResultV1) -> FaberRtPtrResultV1 {
    panic::catch_unwind(AssertUnwindSafe(operation))
        .unwrap_or(FaberRtPtrResultV1::failure(STATUS_PANIC))
}

fn context_mut(context: *mut FaberRtContextV1) -> Option<&'static mut RuntimeContext> {
    (!context.is_null()).then(|| unsafe { &mut *context.cast::<RuntimeContext>() })
}

fn context_ptr(runtime: &mut RuntimeContext) -> *mut FaberRtContextV1 {
    std::ptr::from_mut(runtime).cast()
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_octeti_new(
    context: *mut FaberRtContextV1,
    value: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let (Some(runtime), Some(value)) = (context_mut(context), unsafe { value.as_ref() }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        if value.data.is_null() && value.len != 0 {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let Ok(length) = usize::try_from(value.len) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let bytes = if length == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(value.data, length) }.to_vec()
        };
        store_octeti(runtime, bytes)
    })
}

pub(super) fn store_octeti(runtime: &mut RuntimeContext, bytes: Vec<u8>) -> FaberRtPtrResultV1 {
    let bytes = super::StableBox::new(bytes);
    let handle = bytes.handle();
    runtime.octeti.push(bytes);
    FaberRtPtrResultV1::success(handle)
}

pub(super) fn find_octeti(runtime: &RuntimeContext, handle: *mut c_void) -> Option<&Vec<u8>> {
    runtime
        .octeti
        .iter()
        .find(|bytes| std::ptr::eq(bytes.as_ref(), handle.cast()))
        .map(super::StableBox::as_ref)
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_valor_octeti(
    context: *mut FaberRtContextV1,
    octeti: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = context_mut(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(bytes) = find_octeti(runtime, octeti).cloned() else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_valor(context, Valor::Octeti(bytes))
    })
}

/// `tensor ↦ valor` — box flat tensor elements as a Valor lista (kernel bridge).
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_valor_tensor(
    context: *mut FaberRtContextV1,
    tensor: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = context_mut(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(tensor) = find_tensor(runtime, tensor) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let mut values = Vec::with_capacity(tensor.data.len());
        for value in &tensor.data {
            let Some(item) = runtime_value_to_valor(runtime, tensor.kind, value) else {
                return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
            };
            values.push(item);
        }
        store_valor(context, Valor::Lista(values))
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_valor_array(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = context_mut(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(array) = find_array(runtime, array) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(values) = array
            .values
            .iter()
            .map(|value| runtime_value_to_valor(runtime, array.kind, value))
            .collect::<Option<Vec<_>>>()
        else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_valor(context, Valor::Lista(values))
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_valor_map(
    context: *mut FaberRtContextV1,
    map: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = context_mut(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(map) = find_map(runtime, map) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(entries) = map
            .entries
            .iter()
            .copied()
            .map(|(key, value)| {
                let Valor::Textus(key) = runtime_value_to_valor(runtime, map.key_kind, key)? else {
                    return None;
                };
                Some((key, runtime_value_to_valor(runtime, map.value_kind, value)?))
            })
            .collect::<Option<BTreeMap<_, _>>>()
        else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_valor(context, Valor::Tabula(entries))
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_valor_get_octeti(
    context: *mut FaberRtContextV1,
    valor: *const Valor,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(bytes) = with_valor(context, valor, |value| match value {
            Valor::Octeti(bytes) => Some(bytes.clone()),
            _ => None,
        }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(runtime) = context_mut(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_octeti(runtime, bytes)
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_valor_get_array(
    context: *mut FaberRtContextV1,
    valor: *const Valor,
    kind: FaberRtValueKindV1,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(values) = with_valor(context, valor, |value| match value {
            Valor::Lista(values) => Some(values.clone()),
            _ => None,
        }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(runtime) = context_mut(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(values) = values
            .iter()
            .map(|value| valor_to_runtime_value(runtime, value, kind))
            .collect::<Option<Vec<_>>>()
        else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_array(runtime, kind, values)
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_valor_get_map(
    context: *mut FaberRtContextV1,
    valor: *const Valor,
    key_kind: FaberRtValueKindV1,
    value_kind: FaberRtValueKindV1,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        if key_kind != VALUE_KIND_TEXT {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let Some(entries) = with_valor(context, valor, |value| match value {
            Valor::Tabula(entries) => Some(entries.clone()),
            _ => None,
        }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(runtime) = context_mut(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let mut converted = Vec::with_capacity(entries.len());
        for (key, value) in entries {
            let key = store_text(context_ptr(runtime), key);
            let Some(value) = valor_to_runtime_value(runtime, &value, value_kind) else {
                return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
            };
            if !key.status.is_ok() {
                return key;
            }
            converted.push((RuntimeValue::Ptr(key.value), value));
        }
        store_map(runtime, key_kind, value_kind, converted)
    })
}

pub(super) fn runtime_value_to_valor(
    runtime: &RuntimeContext,
    kind: FaberRtValueKindV1,
    value: RuntimeValue,
) -> Option<Valor> {
    Some(match (kind, value) {
        (VALUE_KIND_I1, RuntimeValue::I1(value)) => Valor::Bivalens(value != 0),
        (VALUE_KIND_I8, RuntimeValue::I8(value)) => Valor::Numerus(value.into()),
        (VALUE_KIND_I16, RuntimeValue::I16(value)) => Valor::Numerus(value.into()),
        (VALUE_KIND_I32, RuntimeValue::I32(value)) => Valor::Numerus(value.into()),
        (VALUE_KIND_I64, RuntimeValue::I64(value)) => Valor::Numerus(value),
        (VALUE_KIND_U8, RuntimeValue::U8(value)) => Valor::Numerus(value.into()),
        (VALUE_KIND_U16, RuntimeValue::U16(value)) => Valor::Numerus(value.into()),
        (VALUE_KIND_U32, RuntimeValue::U32(value)) => Valor::Numerus(value.into()),
        (VALUE_KIND_U64, RuntimeValue::U64(value)) => Valor::Numerus(value.try_into().ok()?),
        (VALUE_KIND_F32, RuntimeValue::F32(value)) => Valor::Fractus(value.into()),
        (VALUE_KIND_F64, RuntimeValue::F64(value)) => Valor::Fractus(value),
        (VALUE_KIND_TEXT, RuntimeValue::Ptr(value)) => Valor::Textus(text_value(value.cast())?),
        // L27: a `ptr`-kind element may carry a textus payload — the array
        // element kind for textus is `ptr` (`runtime_value_abi`), so `↦ valor`
        // of a `lista<textus>` stored PTR handles (arena text or literal
        // slice descriptor) and `valor_array` failed with
        // `STATUS_INVALID_ARGUMENT`, latching a nonzero process exit for
        // stdout-correct programs. Resolve arena handles by pointer equality
        // FIRST (memory-safety: never dereference an unknown handle), recurse
        // into aggregates, and only then fall back to literal slice
        // descriptors (`text_value`). Set handles have no `Valor` variant, so
        // they fail closed here instead of reaching the descriptor fallback.
        (VALUE_KIND_PTR, RuntimeValue::Ptr(value)) => {
            if let Some(text) = find_text(runtime, value) {
                Valor::Textus(text.value.clone())
            } else if let Some(bytes) = find_octeti(runtime, value) {
                Valor::Octeti(bytes.clone())
            } else if let Some(array) = find_array(runtime, value) {
                array_to_valor(runtime, array)?
            } else if let Some(map) = find_map(runtime, value) {
                map_to_valor(runtime, map)?
            } else if find_set(runtime, value).is_some() {
                return None;
            } else {
                Valor::Textus(text_value(value.cast())?)
            }
        }
        (VALUE_KIND_VALOR, RuntimeValue::Ptr(value)) => with_valor(
            std::ptr::from_ref(runtime).cast_mut().cast(),
            value.cast(),
            |value| Some(value.clone()),
        )?,
        _ => return None,
    })
}

/// Convert a versioned array handle's elements into a `Valor::Lista`.
///
/// Element handles are resolved through [`runtime_value_to_valor`], so nested
/// aggregates (textus, octeti, further listas, tabulae) decode recursively.
fn array_to_valor(runtime: &RuntimeContext, array: &RuntimeArray) -> Option<Valor> {
    let values = array
        .values
        .iter()
        .map(|value| runtime_value_to_valor(runtime, array.kind, value))
        .collect::<Option<Vec<_>>>()?;
    Some(Valor::Lista(values))
}

/// Convert a versioned map handle's entries into a `Valor::Tabula`.
///
/// Non-text keys fail closed (the `Valor::Tabula` key contract is `String`),
/// matching the diagnostic renderer's key resolution.
fn map_to_valor(runtime: &RuntimeContext, map: &RuntimeMap) -> Option<Valor> {
    let entries = map
        .entries
        .iter()
        .copied()
        .map(|(key, value)| {
            let Valor::Textus(key) = runtime_value_to_valor(runtime, map.key_kind, key)? else {
                return None;
            };
            Some((key, runtime_value_to_valor(runtime, map.value_kind, value)?))
        })
        .collect::<Option<BTreeMap<_, _>>>()?;
    Some(Valor::Tabula(entries))
}

pub(super) fn valor_to_runtime_value(
    runtime: &mut RuntimeContext,
    valor: &Valor,
    kind: FaberRtValueKindV1,
) -> Option<RuntimeValue> {
    Some(match kind {
        VALUE_KIND_I1 => RuntimeValue::I1(u8::from(bool::from_valor(valor)?)),
        VALUE_KIND_I8 => RuntimeValue::I8(i64::from_valor(valor)?.try_into().ok()?),
        VALUE_KIND_I16 => RuntimeValue::I16(i64::from_valor(valor)?.try_into().ok()?),
        VALUE_KIND_I32 => RuntimeValue::I32(i64::from_valor(valor)?.try_into().ok()?),
        VALUE_KIND_I64 => RuntimeValue::I64(i64::from_valor(valor)?),
        VALUE_KIND_U8 => RuntimeValue::U8(i64::from_valor(valor)?.try_into().ok()?),
        VALUE_KIND_U16 => RuntimeValue::U16(i64::from_valor(valor)?.try_into().ok()?),
        VALUE_KIND_U32 => RuntimeValue::U32(i64::from_valor(valor)?.try_into().ok()?),
        VALUE_KIND_U64 => RuntimeValue::U64(i64::from_valor(valor)?.try_into().ok()?),
        VALUE_KIND_F32 => RuntimeValue::F32({
            // SAFETY: f64 → f32 truncation is the element-width contract
            // for the f32 valor lattice. Values come from Valor::from_valor.
            #[allow(clippy::cast_possible_truncation)]
            let value = f64::from_valor(valor)? as f32;
            value
        }),
        VALUE_KIND_F64 => RuntimeValue::F64(f64::from_valor(valor)?),
        VALUE_KIND_TEXT => {
            let result = store_text(context_ptr(runtime), String::from_valor(valor)?);
            result
                .status
                .is_ok()
                .then_some(RuntimeValue::Ptr(result.value))?
        }
        VALUE_KIND_VALOR => {
            let result = store_valor(context_ptr(runtime), valor.clone());
            result
                .status
                .is_ok()
                .then_some(RuntimeValue::Ptr(result.value))?
        }
        _ => return None,
    })
}
