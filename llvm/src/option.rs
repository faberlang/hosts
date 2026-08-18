//! Arena-owned scalar and opaque-handle options for the LLVM host ABI.

use super::array::{read_value, valid_kind, write_value, RuntimeValue};
use super::format::{find_text, text_value};
use super::{opaque_value_text, unsupported_opaque_diagnostic, write_diagnostic, RuntimeContext};
use crate::abi::FaberRtContextV1;
use crate::abi::{
    FaberRtPtrResultV1, FaberRtStatusV1, STATUS_INVALID_ARGUMENT, STATUS_OK, STATUS_PANIC,
};
use faber::{display_bivalens, display_fractus};
use radix_host_abi::{
    FaberRtValueKindV1, VALUE_KIND_F16, VALUE_KIND_F32, VALUE_KIND_F64, VALUE_KIND_I1,
    VALUE_KIND_I16, VALUE_KIND_I32, VALUE_KIND_I64, VALUE_KIND_I8, VALUE_KIND_PTR, VALUE_KIND_TEXT,
    VALUE_KIND_U16, VALUE_KIND_U32, VALUE_KIND_U64, VALUE_KIND_U8,
};
use std::ffi::c_void;
use std::panic::{self, AssertUnwindSafe};

pub(super) struct RuntimeOption {
    pub(super) kind: FaberRtValueKindV1,
    pub(super) value: Option<RuntimeValue>,
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_option_none(
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
        store_option(runtime, kind, None)
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_option_some(
    context: *mut FaberRtContextV1,
    kind: FaberRtValueKindV1,
    value: *const c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(value) = (unsafe { read_value(kind, value) }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_option(runtime, kind, Some(value))
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_option_is_present(
    context: *mut FaberRtContextV1,
    option: *mut c_void,
    kind: FaberRtValueKindV1,
    output: *mut c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(option) = find_option(runtime, option) else {
            // Raw null-encoded option (the chain/literal result IS the value;
            // nil is the null pointer). Present iff the pointer is non-null,
            // regardless of payload kind — the pointer bits either carry the
            // scalar payload or ARE the payload handle.
            if !valid_kind(kind) {
                return STATUS_INVALID_ARGUMENT;
            }
            if !(unsafe { write_u8(output, u8::from(!option.is_null())) }) {
                return STATUS_INVALID_ARGUMENT;
            }
            return STATUS_OK;
        };
        if option.kind != kind || !(unsafe { write_u8(output, u8::from(option.value.is_some())) }) {
            return STATUS_INVALID_ARGUMENT;
        }
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_option_get(
    context: *mut FaberRtContextV1,
    option: *mut c_void,
    kind: FaberRtValueKindV1,
    output: *mut c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(option) = find_option(runtime, option) else {
            // Raw null-encoded option (chain/literal result): unwrap of a null
            // (nil) option fails closed; a non-null pointer IS the payload,
            // decoded from the pointer bits per value-kind.
            if option.is_null() {
                return STATUS_INVALID_ARGUMENT;
            }
            let Some(value) = raw_option_value(option, kind) else {
                return STATUS_INVALID_ARGUMENT;
            };
            if !(unsafe { write_value(value, output) }) {
                return STATUS_INVALID_ARGUMENT;
            }
            return STATUS_OK;
        };
        let Some(value) = option.value else {
            return STATUS_INVALID_ARGUMENT;
        };
        if option.kind != kind || !(unsafe { write_value(value, output) }) {
            return STATUS_INVALID_ARGUMENT;
        }
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_option_get_or(
    context: *mut FaberRtContextV1,
    option: *mut c_void,
    kind: FaberRtValueKindV1,
    fallback: *const c_void,
    output: *mut c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(option) = find_option(runtime, option) else {
            // Raw null-encoded option: nil is the null pointer, so a null
            // option coalesces to the fallback and a non-null option IS the
            // payload, decoded from the pointer bits per value-kind. (The
            // chain/coalesce path emits this encoding directly.)
            if !valid_kind(kind) {
                return STATUS_INVALID_ARGUMENT;
            }
            let value = if option.is_null() {
                let Some(value) = (unsafe { read_value(kind, fallback) }) else {
                    return STATUS_INVALID_ARGUMENT;
                };
                value
            } else {
                let Some(value) = raw_option_value(option, kind) else {
                    return STATUS_INVALID_ARGUMENT;
                };
                value
            };
            if !(unsafe { write_value(value, output) }) {
                return STATUS_INVALID_ARGUMENT;
            }
            return STATUS_OK;
        };
        if option.kind != kind {
            return STATUS_INVALID_ARGUMENT;
        }
        let value = if let Some(value) = option.value {
            value
        } else {
            let Some(value) = (unsafe { read_value(kind, fallback) }) else {
                return STATUS_INVALID_ARGUMENT;
            };
            value
        };
        if !(unsafe { write_value(value, output) }) {
            return STATUS_INVALID_ARGUMENT;
        }
        STATUS_OK
    })
}

/// Unwrap a local opaque option whose payload is a boxed aggregate value.
///
/// The generic LLVM option path represents `T ∪ nihil` values with a
/// non-arena payload as a pointer to the boxed payload; unwrapping is the
/// pointer-preserving box passthrough (the emitted program dereferences the
/// box itself).
///
/// # Safety
///
/// `option` must be a live box pointer produced by the option construction
/// path for the same payload type.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_option_unwrap_ptr(option: *mut c_void) -> *mut c_void {
    option
}

/// Report a `nota` of an option (`T ∪ nihil`) value with the Rust oracle's
/// `display_option` semantics: the payload itself for a present value, or
/// `nihil` for the null handle.
///
/// The handle is either an arena option box (from the option construction
/// ABI), or a raw null-encoded option where a non-null pointer IS the payload
/// (inline optional chains and `nihil` literals). The kind selects how the
/// payload renders: scalar payloads are encoded in the pointer bits, `ptr`
/// payloads carry the actual payload handle (text, array, octeti).
///
/// # Safety
///
/// `context` must be null or a live runtime context. `option` is only used
/// for pointer-equality arena lookups and bit-pattern decoding; it is never
/// dereferenced unless it is a known payload handle.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_nota_option(
    context: *mut FaberRtContextV1,
    option: *mut c_void,
    kind: FaberRtValueKindV1,
) -> FaberRtStatusV1 {
    diagnostic_option(context, option, kind, false)
}

/// Report a `mone` of an option (`T ∪ nihil`) value on the stderr stream
/// (see [`__faber_rt_v1_diagnostic_nota_option`] for the carrier contract).
///
/// # Safety
///
/// `context` must be null or a live runtime context. `option` is only used
/// for pointer-equality arena lookups and bit-pattern decoding.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_mone_option(
    context: *mut FaberRtContextV1,
    option: *mut c_void,
    kind: FaberRtValueKindV1,
) -> FaberRtStatusV1 {
    diagnostic_option(context, option, kind, true)
}

/// Report a `scribe` of an option (`T ∪ nihil`) value on the stdout stream
/// (see [`__faber_rt_v1_diagnostic_nota_option`] for the carrier contract).
///
/// # Safety
///
/// `context` must be null or a live runtime context. `option` is only used
/// for pointer-equality arena lookups and bit-pattern decoding.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_scribe_option(
    context: *mut FaberRtContextV1,
    option: *mut c_void,
    kind: FaberRtValueKindV1,
) -> FaberRtStatusV1 {
    diagnostic_option(context, option, kind, false)
}

/// Report a `vide` of an option (`T ∪ nihil`) value on the stdout stream
/// (see [`__faber_rt_v1_diagnostic_nota_option`] for the carrier contract).
///
/// # Safety
///
/// `context` must be null or a live runtime context. `option` is only used
/// for pointer-equality arena lookups and bit-pattern decoding.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_vide_option(
    context: *mut FaberRtContextV1,
    option: *mut c_void,
    kind: FaberRtValueKindV1,
) -> FaberRtStatusV1 {
    diagnostic_option(context, option, kind, false)
}

/// Shared option diagnostic carrier: render the payload with the Rust
/// oracle's `display_option` semantics on the stream's channel, or `nihil`
/// for the null handle. Opaque payloads stay arena-only (fail-closed).
fn diagnostic_option(
    context: *mut FaberRtContextV1,
    option: *mut c_void,
    kind: FaberRtValueKindV1,
    stderr: bool,
) -> FaberRtStatusV1 {
    if context.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }
    let runtime = unsafe { &*context.cast::<RuntimeContext>() };
    if option.is_null() {
        return write_diagnostic(context, stderr, "nihil");
    }
    if let Some(boxed) = find_option(runtime, option) {
        let Some(value) = &boxed.value else {
            return write_diagnostic(context, stderr, "nihil");
        };
        let Some(text) = render_option_payload(runtime, boxed.kind, value) else {
            return unsupported_opaque_diagnostic(context);
        };
        return write_diagnostic(context, stderr, text);
    }
    // Raw null-encoded option: the pointer is the payload (bits for scalar
    // payloads, the payload handle for `ptr` payloads).
    let Some(text) = render_raw_option_payload(runtime, option, kind) else {
        return unsupported_opaque_diagnostic(context);
    };
    write_diagnostic(context, stderr, text)
}

/// Render an arena-boxed option payload (the `Some` case of
/// `display_option`).
fn render_option_payload(
    runtime: &RuntimeContext,
    kind: FaberRtValueKindV1,
    value: &RuntimeValue,
) -> Option<String> {
    let text = match (kind, value) {
        (VALUE_KIND_I1, RuntimeValue::I1(value)) => display_bivalens(*value != 0).to_owned(),
        (VALUE_KIND_I8, RuntimeValue::I8(value)) => format!("{value}"),
        (VALUE_KIND_I16, RuntimeValue::I16(value)) => format!("{value}"),
        (VALUE_KIND_I32, RuntimeValue::I32(value)) => format!("{value}"),
        (VALUE_KIND_I64, RuntimeValue::I64(value)) => format!("{value}"),
        (VALUE_KIND_U8, RuntimeValue::U8(value)) => format!("{value}"),
        (VALUE_KIND_U16, RuntimeValue::U16(value)) => format!("{value}"),
        (VALUE_KIND_U32, RuntimeValue::U32(value)) => format!("{value}"),
        (VALUE_KIND_U64, RuntimeValue::U64(value)) => format!("{value}"),
        (VALUE_KIND_F32, RuntimeValue::F32(value)) => display_fractus(*value),
        (VALUE_KIND_F64, RuntimeValue::F64(value)) => display_fractus(*value),
        (VALUE_KIND_TEXT, RuntimeValue::Ptr(value)) => render_text_payload(runtime, *value)?,
        (VALUE_KIND_PTR, RuntimeValue::Ptr(value)) => {
            // A `ptr`-kind option may carry a textus payload: the array-option
            // carrier (`primus`/`ultimus`/`accipe` on a `lista<textus>`) stores
            // the element option under the array's `ptr` element kind, while the
            // diagnostic path passes `VALUE_KIND_TEXT` — the renderer must
            // resolve arena text handles / slice descriptors first (same
            // resolution order as array element rendering), then fall back to
            // opaque aggregate rendering.
            find_text(runtime, *value)
                .map(|text| text.value.clone())
                .or_else(|| text_value(value.cast()))
                .or_else(|| opaque_value_text(runtime, *value))?
        }
        _ => return None,
    };
    Some(text)
}

/// Render a raw null-encoded option payload from the pointer bits.
fn render_raw_option_payload(
    runtime: &RuntimeContext,
    option: *mut c_void,
    kind: FaberRtValueKindV1,
) -> Option<String> {
    let bits = option as usize as u64;
    let text = match kind {
        VALUE_KIND_I64 => format!("{}", bits as i64),
        VALUE_KIND_I1 => display_bivalens(bits != 0).to_owned(),
        VALUE_KIND_F32 => display_fractus(f32::from_bits(bits as u32)),
        VALUE_KIND_F64 => display_fractus(f64::from_bits(bits)),
        // Text payloads may be arena text handles or compiler-owned literal
        // slice descriptors (the `nota_text` pattern); the pointer is only
        // dereferenced through those two known layouts.
        VALUE_KIND_TEXT => render_text_payload(runtime, option)?,
        // Opaque payloads (array/octeti) stay arena-only lookups; unknown
        // handles fail closed instead of being dereferenced.
        VALUE_KIND_PTR => opaque_value_text(runtime, option)?,
        _ => return None,
    };
    Some(text)
}

/// Render a text payload handle: an arena text handle or a literal-global
/// `FaberRtSliceV1` descriptor.
fn render_text_payload(runtime: &RuntimeContext, handle: *mut c_void) -> Option<String> {
    if let Some(text) = find_text(runtime, handle) {
        return Some(text.value.clone());
    }
    text_value(handle.cast())
}

pub(super) fn store_option(
    runtime: &mut RuntimeContext,
    kind: FaberRtValueKindV1,
    value: Option<RuntimeValue>,
) -> FaberRtPtrResultV1 {
    let option = super::StableBox::new(RuntimeOption { kind, value });
    let handle = option.handle();
    runtime.options.push(option);
    FaberRtPtrResultV1::success(handle)
}

fn find_option(runtime: &RuntimeContext, handle: *mut c_void) -> Option<&RuntimeOption> {
    runtime
        .options
        .iter()
        .find(|option| std::ptr::eq(option.as_ref(), handle.cast_const().cast::<RuntimeOption>()))
        .map(super::StableBox::as_ref)
}

/// Decode a raw null-encoded option payload from the pointer bits (the L9
/// option-carrier pattern). Scalar payloads are encoded in the pointer bits;
/// text/opaque payloads ARE the payload handle. Returns `None` for unknown
/// kinds (fail-closed; opaque handles stay arena-only).
fn raw_option_value(option: *mut c_void, kind: FaberRtValueKindV1) -> Option<RuntimeValue> {
    let bits = option as usize as u64;
    Some(match kind {
        VALUE_KIND_I1 => RuntimeValue::I1(u8::from(bits != 0)),
        VALUE_KIND_I8 => RuntimeValue::I8(bits as i8),
        VALUE_KIND_I16 => RuntimeValue::I16(bits as i16),
        VALUE_KIND_I32 => RuntimeValue::I32(bits as i32),
        VALUE_KIND_I64 => RuntimeValue::I64(bits as i64),
        VALUE_KIND_U8 => RuntimeValue::U8(bits as u8),
        VALUE_KIND_U16 => RuntimeValue::U16(bits as u16),
        VALUE_KIND_U32 => RuntimeValue::U32(bits as u32),
        VALUE_KIND_U64 => RuntimeValue::U64(bits as u64),
        VALUE_KIND_F16 => RuntimeValue::F16(bits as u16),
        VALUE_KIND_F32 => RuntimeValue::F32(f32::from_bits(bits as u32)),
        VALUE_KIND_F64 => RuntimeValue::F64(f64::from_bits(bits)),
        // Text payloads may be arena text handles or literal-global slice
        // descriptors; opaque payloads carry the payload handle itself.
        VALUE_KIND_TEXT | VALUE_KIND_PTR => RuntimeValue::Ptr(option),
        _ => return None,
    })
}

unsafe fn runtime_mut<'a>(context: *mut FaberRtContextV1) -> Option<&'a mut RuntimeContext> {
    (!context.is_null()).then(|| unsafe { &mut *context.cast::<RuntimeContext>() })
}

unsafe fn write_u8(output: *mut c_void, value: u8) -> bool {
    let output = output.cast::<u8>();
    if output.is_null() {
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
