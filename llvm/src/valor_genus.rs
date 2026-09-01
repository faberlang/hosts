//! Atomic named-field conversion between physical LLVM genus values and Valor maps.

use super::RuntimeContext;
use super::array::{RuntimeValue, read_value, write_value};
use super::convert::store_valor;
use super::format::text_value;
use super::valor_aggregate::{runtime_value_to_valor, valor_to_runtime_value};
use crate::abi::FaberRtContextV1;
use crate::abi::{
    FaberRtPtrResultV1, FaberRtSliceV1, FaberRtStatusV1, STATUS_INVALID_ARGUMENT, STATUS_OK,
    STATUS_PANIC,
};
use faber::{FromValor, Instans, Valor};
use radix_host_abi::{
    FaberRtValueKindV1, VALUE_KIND_ASCII, VALUE_KIND_INSTANS, VALUE_KIND_OPTION_I64,
};
use std::collections::BTreeMap;
use std::ffi::{CStr, c_void};
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

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_valor_genus(
    context: *mut FaberRtContextV1,
    count: u64,
    names: *const *const FaberRtSliceV1,
    kinds: *const FaberRtValueKindV1,
    values: *const *const c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some((names, kinds, values)) = (unsafe { field_inputs(count, names, kinds, values) })
        else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let mut fields = BTreeMap::new();
        for ((name, kind), value) in names.iter().zip(kinds).zip(values) {
            let Some(name) = text_value(*name) else {
                return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
            };
            let Some(value) = (unsafe { genus_value_to_valor(runtime, *kind, *value) }) else {
                return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
            };
            fields.insert(name, value);
        }
        store_valor(context, Valor::Tabula(fields))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_valor_get_genus(
    context: *mut FaberRtContextV1,
    valor: *const Valor,
    count: u64,
    names: *const *const FaberRtSliceV1,
    kinds: *const FaberRtValueKindV1,
    defaultable: *const u8,
    outputs: *const *mut c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = runtime(context) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(valor) = runtime
            .valors
            .iter()
            .find(|candidate| std::ptr::eq(candidate.as_ref(), valor))
        else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Valor::Tabula(fields) = valor.as_ref() else {
            return STATUS_INVALID_ARGUMENT;
        };
        let fields = fields.clone();
        let Ok(count) = usize::try_from(count) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if names.is_null() || kinds.is_null() || defaultable.is_null() || outputs.is_null() {
            return STATUS_INVALID_ARGUMENT;
        }
        let names = unsafe { std::slice::from_raw_parts(names, count) };
        let kinds = unsafe { std::slice::from_raw_parts(kinds, count) };
        let defaultable = unsafe { std::slice::from_raw_parts(defaultable, count) };
        let outputs = unsafe { std::slice::from_raw_parts(outputs, count) };
        let mut converted = Vec::with_capacity(count);
        for (((name, kind), defaultable), output) in
            names.iter().zip(kinds).zip(defaultable).zip(outputs)
        {
            if output.is_null() {
                return STATUS_INVALID_ARGUMENT;
            }
            let Some(name) = text_value(*name) else {
                return STATUS_INVALID_ARGUMENT;
            };
            let Some(field) = fields.get(&name) else {
                if *defaultable != 0 {
                    converted.push(None);
                    continue;
                }
                return STATUS_INVALID_ARGUMENT;
            };
            let Some(value) = genus_valor_to_value(runtime, field, *kind) else {
                return STATUS_INVALID_ARGUMENT;
            };
            converted.push(Some((value, *output)));
        }
        for entry in converted.into_iter().flatten() {
            if !(unsafe { write_value(entry.0, entry.1) }) {
                return STATUS_INVALID_ARGUMENT;
            }
        }
        STATUS_OK
    })
}

unsafe fn field_inputs<'a>(
    count: u64,
    names: *const *const FaberRtSliceV1,
    kinds: *const FaberRtValueKindV1,
    values: *const *const c_void,
) -> Option<(
    &'a [*const FaberRtSliceV1],
    &'a [FaberRtValueKindV1],
    &'a [*const c_void],
)> {
    let count = usize::try_from(count).ok()?;
    if names.is_null() || kinds.is_null() || values.is_null() {
        return None;
    }
    Some((
        unsafe { std::slice::from_raw_parts(names, count) },
        unsafe { std::slice::from_raw_parts(kinds, count) },
        unsafe { std::slice::from_raw_parts(values, count) },
    ))
}

unsafe fn genus_value_to_valor(
    runtime: &RuntimeContext,
    kind: FaberRtValueKindV1,
    value: *const c_void,
) -> Option<Valor> {
    match kind {
        VALUE_KIND_ASCII => {
            let handle: *mut c_void = unsafe { value.cast::<*mut c_void>().read() };
            if handle.is_null() {
                return None;
            }
            let bytes = unsafe { CStr::from_ptr(handle.cast()) }.to_bytes();
            bytes
                .is_ascii()
                .then(|| Valor::Textus(String::from_utf8_lossy(bytes).into_owned()))
        }
        VALUE_KIND_OPTION_I64 => {
            let handle: *mut c_void = unsafe { value.cast::<*mut c_void>().read() };
            if handle.is_null() {
                Some(Valor::Nihil)
            } else {
                Some(Valor::Numerus(unsafe { *handle.cast::<i64>() }))
            }
        }
        VALUE_KIND_INSTANS => {
            let handle: *mut c_void = unsafe { value.cast::<*mut c_void>().read() };
            let instant = runtime
                .instants
                .iter()
                .find(|candidate| std::ptr::eq(candidate.as_ref(), handle.cast()))?;
            Some(Valor::Instans(instant.to_rfc3339()))
        }
        _ => runtime_value_to_valor(runtime, kind, unsafe { read_value(kind, value) }?),
    }
}

fn genus_valor_to_value(
    runtime: &mut RuntimeContext,
    valor: &Valor,
    kind: FaberRtValueKindV1,
) -> Option<RuntimeValue> {
    match kind {
        VALUE_KIND_ASCII => {
            let extracted = String::from_valor(valor)?;
            if !extracted.is_ascii() || extracted.as_bytes().contains(&0) {
                return None;
            }
            let mut bytes = extracted.into_bytes();
            bytes.push(0);
            let bytes = super::StableBox::from_box(bytes.into_boxed_slice());
            let handle = bytes.as_ref().as_ptr().cast_mut().cast();
            runtime.ascii.push(bytes);
            Some(RuntimeValue::Ptr(handle))
        }
        VALUE_KIND_OPTION_I64 => {
            if valor == &Valor::Nihil {
                Some(RuntimeValue::Ptr(std::ptr::null_mut()))
            } else {
                let value = super::StableBox::new(i64::from_valor(valor)?);
                let handle = value.handle();
                runtime.numeric_boxes.push(value);
                Some(RuntimeValue::Ptr(handle))
            }
        }
        VALUE_KIND_INSTANS => {
            let value = super::StableBox::new(Instans::from_valor(valor)?);
            let handle = value.handle();
            runtime.instants.push(value);
            Some(RuntimeValue::Ptr(handle))
        }
        _ => valor_to_runtime_value(runtime, valor, kind),
    }
}
