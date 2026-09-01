//! Versioned `norma:*` provider carriers for the LLVM host.
//!
//! The LLVM emitter routes named providers (`norma:time.nunc`,
//! `norma:toml.parse`, `norma:value.get`, `norma:json.{stringify,parse,try_parse}`)
//! to these versioned v1 symbols. Each implementation either provides the
//! provider semantics or fails closed with a stable unsupported status that
//! the emitted program latches honestly (Stage 5 owns the semantic surface).

use super::RuntimeContext;
use super::convert::{store_valor, with_valor};
use super::format::{store_text, text_value};
use crate::abi::FaberRtContextV1;
use crate::abi::{
    FaberRtPtrResultV1, FaberRtSliceV1, STATUS_INVALID_ARGUMENT, STATUS_PANIC, STATUS_UNSUPPORTED,
};
use faber::{Json, Valor};
use std::panic::{self, AssertUnwindSafe};

fn ffi(operation: impl FnOnce() -> FaberRtPtrResultV1) -> FaberRtPtrResultV1 {
    panic::catch_unwind(AssertUnwindSafe(operation))
        .unwrap_or(FaberRtPtrResultV1::failure(STATUS_PANIC))
}

fn runtime(context: *mut FaberRtContextV1) -> Option<&'static mut RuntimeContext> {
    (!context.is_null()).then(|| unsafe { &mut *context.cast::<RuntimeContext>() })
}

/// `norma:toml.solve` — parse a TOML document into a `valor`.
///
/// TOML parsing is not yet implemented on the LLVM host; the provider fails
/// closed with a stable unsupported status.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_toml_solve(
    context: *mut FaberRtContextV1,
    _text: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    ffi(|| {
        if context.is_null() {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        FaberRtPtrResultV1::failure(STATUS_UNSUPPORTED)
    })
}

/// `norma:valor.cape` — read one field from a tabula `valor` by key.
///
/// # Safety
///
/// `context` must be live. `value` must be a `valor` handle created by this
/// runtime. `key` follows the slice validity contract of
/// [`__faber_rt_v1_write_nota_text`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_valor_cape(
    context: *mut FaberRtContextV1,
    value: *const Valor,
    key: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    ffi(|| {
        if context.is_null() {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let Some(key) = text_value(key) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(field) = with_valor(context, value, |value| match value {
            Valor::Tabula(fields) => fields.get(&key).cloned(),
            _ => None,
        }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_valor(context, field)
    })
}

/// `norma:json.pange` — serialize a `valor` to its JSON wire text.
///
/// # Safety
///
/// `context` must be live. `value` must be a `valor` handle created by this
/// runtime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_json_pange(
    context: *mut FaberRtContextV1,
    value: *const Valor,
) -> FaberRtPtrResultV1 {
    ffi(|| {
        if context.is_null() {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let Some(json) = with_valor(context, value, |value| Json::try_from(value.clone()).ok())
        else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_text(context, json.to_wire())
    })
}

/// `norma:json.solve` — parse JSON wire text into a `valor`.
///
/// # Safety
///
/// `context` must be live. `wire` follows the slice validity contract of
/// [`__faber_rt_v1_write_nota_text`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_json_solve(
    context: *mut FaberRtContextV1,
    wire: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    ffi(|| {
        if context.is_null() {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let Some(wire) = text_value(wire) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Ok(json) = Json::parse(&wire) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_valor(context, json.into_valor())
    })
}

/// `norma:json.tempta` — tentatively parse JSON wire text into a `json ∪
/// nihil` union.
///
/// The union is an opaque box whose payload is the text form the emitted
/// program renders (the emitted `nota` passes the payload straight to the
/// text diagnostic). On success the box holds the JSON wire; on failure it
/// holds the `nihil` spelling.
///
/// # Safety
///
/// `context` must be live. `wire` follows the slice validity contract of
/// [`__faber_rt_v1_write_nota_text`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_json_tempta(
    context: *mut FaberRtContextV1,
    wire: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    ffi(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let payload = match text_value(wire) {
            Some(wire) => match Json::parse(&wire) {
                Ok(json) => json.to_wire(),
                Err(_) => "nihil".to_owned(),
            },
            None => "nihil".to_owned(),
        };
        let text = store_text(context, payload).value;
        let text = super::StableBox::new(text);
        let handle = text.handle();
        runtime.union_boxes.push(text);
        FaberRtPtrResultV1::success(handle)
    })
}
