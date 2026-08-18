//! Scalar template formatting and runtime-owned LLVM text handles.

use super::RuntimeContext;
use crate::abi::FaberRtContextV1;
use crate::abi::{
    FaberRtPtrResultV1, FaberRtSliceV1, FaberRtStatusV1, STATUS_INVALID_ARGUMENT, STATUS_OK,
    STATUS_PANIC, STATUS_UNSUPPORTED,
};
use faber::{display_bivalens, display_fractus};
use std::ffi::c_void;
use std::panic::{self, AssertUnwindSafe};

pub(super) fn ffi_ptr_result(operation: impl FnOnce() -> FaberRtPtrResultV1) -> FaberRtPtrResultV1 {
    panic::catch_unwind(AssertUnwindSafe(operation))
        .unwrap_or(FaberRtPtrResultV1::failure(STATUS_PANIC))
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_format_i1(
    context: *mut FaberRtContextV1,
    template: FaberRtSliceV1,
    value: u8,
) -> FaberRtPtrResultV1 {
    format_scalar_values(
        context,
        template,
        &[display_bivalens(value != 0).to_owned()],
    )
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_format_i64(
    context: *mut FaberRtContextV1,
    template: FaberRtSliceV1,
    value: i64,
) -> FaberRtPtrResultV1 {
    format_scalar_values(context, template, &[value.to_string()])
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_format_i64_i64(
    context: *mut FaberRtContextV1,
    template: FaberRtSliceV1,
    first: i64,
    second: i64,
) -> FaberRtPtrResultV1 {
    format_scalar_values(context, template, &[first.to_string(), second.to_string()])
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_format_i64_i64_i64(
    context: *mut FaberRtContextV1,
    template: FaberRtSliceV1,
    first: i64,
    second: i64,
    third: i64,
) -> FaberRtPtrResultV1 {
    format_scalar_values(
        context,
        template,
        &[first.to_string(), second.to_string(), third.to_string()],
    )
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_format_f64(
    context: *mut FaberRtContextV1,
    template: FaberRtSliceV1,
    value: f64,
) -> FaberRtPtrResultV1 {
    // Scalar float display parity with the Rust oracle: integral floats keep
    // the `.0` decimal marker (display_fractus), matching `__faber_rt_v1_text_f64`.
    format_scalar_values(context, template, &[display_fractus(value)])
}

/// L28 (ab91f49f, W16): render a template with one f32 scalar value.
///
/// `display_fractus` keeps the f32 precision (`0.1f32` renders `0.1`, NOT the
/// widened `0.10000000149011612` an f64 carrier would produce). This is the
/// f32 display ABI the grouped multi-arg nota path needs so `fractus<f32>`
/// scribe args join like the HIR-Rust lane.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_format_f32(
    context: *mut FaberRtContextV1,
    template: FaberRtSliceV1,
    value: f32,
) -> FaberRtPtrResultV1 {
    format_scalar_values(context, template, &[display_fractus(value)])
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_format_text(
    context: *mut FaberRtContextV1,
    template: FaberRtSliceV1,
    text: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    format_text_values(context, template, &[text])
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_format_text_text(
    context: *mut FaberRtContextV1,
    template: FaberRtSliceV1,
    first: *const FaberRtSliceV1,
    second: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    format_text_values(context, template, &[first, second])
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_format_text_i64(
    context: *mut FaberRtContextV1,
    template: FaberRtSliceV1,
    text: *const FaberRtSliceV1,
    value: i64,
) -> FaberRtPtrResultV1 {
    let Some(text) = text_value(text) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    format_scalar_values(context, template, &[text, value.to_string()])
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_format_i64_text(
    context: *mut FaberRtContextV1,
    template: FaberRtSliceV1,
    value: i64,
    text: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    let Some(text) = text_value(text) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    format_scalar_values(context, template, &[value.to_string(), text])
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_format_text_text_text(
    context: *mut FaberRtContextV1,
    template: FaberRtSliceV1,
    first: *const FaberRtSliceV1,
    second: *const FaberRtSliceV1,
    third: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    format_text_values(context, template, &[first, second, third])
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_format_text_i64_i1(
    context: *mut FaberRtContextV1,
    template: FaberRtSliceV1,
    text: *const FaberRtSliceV1,
    integer: i64,
    boolean: u8,
) -> FaberRtPtrResultV1 {
    let Some(text) = text_value(text) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    format_scalar_values(
        context,
        template,
        &[
            text,
            integer.to_string(),
            display_bivalens(boolean != 0).to_owned(),
        ],
    )
}

/// Render a template with one opaque collection handle (`lista` / `octeti`).
///
/// The opaque handle is displayed in its Rust-oracle Debug shape (`[1, 2, 3]`
/// for numeric lists, `["prima", "secunda"]` for text lists, `[112, 114, …]`
/// for octeti). Unrecognized handles fail closed with `STATUS_UNSUPPORTED`.
///
/// # Safety
///
/// `context` must be live. `template` follows the slice validity contract of
/// [`__faber_rt_v1_write_nota_text`]; `value` is used only for pointer-equality
/// arena lookups and is never dereferenced directly.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_format_1_ptr_to_ptr(
    context: *mut FaberRtContextV1,
    template: FaberRtSliceV1,
    value: *mut std::ffi::c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| {
        if context.is_null() {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let runtime = unsafe { &*context.cast::<RuntimeContext>() };
        let Some(rendered) = super::opaque_value_text(runtime, value) else {
            return FaberRtPtrResultV1::failure(STATUS_UNSUPPORTED);
        };
        format_scalar_values(context, template, &[rendered])
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_length(
    context: *mut FaberRtContextV1,
    text: *const FaberRtSliceV1,
    out_length: *mut i64,
) -> FaberRtStatusV1 {
    if context.is_null() || out_length.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }
    let Some(value) = text_value(text) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Ok(length) = i64::try_from(value.chars().count()) else {
        return STATUS_INVALID_ARGUMENT;
    };
    *out_length = length;
    STATUS_OK
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_i64(
    context: *mut FaberRtContextV1,
    value: i64,
) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| store_text(context, value.to_string()))
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_f64(
    context: *mut FaberRtContextV1,
    value: f64,
) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| store_text(context, display_fractus(value)))
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_text_i1(
    context: *mut FaberRtContextV1,
    value: u8,
) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| {
        store_text(
            context,
            if value != 0 { "true" } else { "false" }.to_owned(),
        )
    })
}

fn format_scalar_values(
    context: *mut FaberRtContextV1,
    template: FaberRtSliceV1,
    args: &[String],
) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| {
        if context.is_null() || (template.len > 0 && template.data.is_null()) {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let Ok(len) = usize::try_from(template.len) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let bytes = if len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(template.data, len) }
        };
        let Ok(template) = std::str::from_utf8(bytes) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_text(context, render_template(template, args))
    })
}

fn format_text_values(
    context: *mut FaberRtContextV1,
    template: FaberRtSliceV1,
    values: &[*const FaberRtSliceV1],
) -> FaberRtPtrResultV1 {
    let Some(values) = values
        .iter()
        .map(|value| text_value(*value))
        .collect::<Option<Vec<_>>>()
    else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    format_scalar_values(context, template, &values)
}

pub(super) fn text_value(text: *const FaberRtSliceV1) -> Option<String> {
    if text.is_null() {
        return None;
    }
    let text = unsafe { &*text };
    let len = usize::try_from(text.len).ok()?;
    if len > 0 && text.data.is_null() {
        return None;
    }
    let bytes = if len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(text.data, len) }
    };
    std::str::from_utf8(bytes).ok().map(str::to_owned)
}

/// Resolve an arena text handle to its [`RuntimeText`] (pointer-equality
/// lookup; never dereferences an unknown handle).
pub(super) fn find_text(runtime: &RuntimeContext, handle: *mut c_void) -> Option<&RuntimeText> {
    runtime
        .texts
        .iter()
        .find(|text| std::ptr::eq(text.as_ref(), handle.cast_const().cast::<RuntimeText>()))
        .map(super::StableBox::as_ref)
}

pub(super) fn store_text(context: *mut FaberRtContextV1, value: String) -> FaberRtPtrResultV1 {
    if context.is_null() {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    }
    let runtime = unsafe { &mut *context.cast::<RuntimeContext>() };
    let slice = FaberRtSliceV1 {
        data: value.as_ptr(),
        len: value.len() as u64,
    };
    let text = super::StableBox::new(RuntimeText { slice, value });
    let handle = text.handle();
    runtime.texts.push(text);
    FaberRtPtrResultV1::success(handle)
}

/// Store one arena-owned text value and return its opaque handle.
///
/// # Safety
///
/// `context` must be non-null and a live runtime context.
pub(super) unsafe fn store_text_owned(
    context: *mut FaberRtContextV1,
    value: String,
) -> *mut c_void {
    let runtime = unsafe { &mut *context.cast::<RuntimeContext>() };
    let slice = FaberRtSliceV1 {
        data: value.as_ptr(),
        len: value.len() as u64,
    };
    let text = super::StableBox::new(RuntimeText { slice, value });
    let handle = text.handle();
    runtime.texts.push(text);
    handle
}

fn render_template(template: &str, args: &[String]) -> String {
    let mut output = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();
    let mut next_arg = 0usize;
    while let Some(ch) = chars.next() {
        if ch != '§' {
            output.push(ch);
            continue;
        }
        let mut index = String::new();
        while chars.peek().is_some_and(char::is_ascii_digit) {
            if let Some(digit) = chars.next() {
                index.push(digit);
            }
        }
        let arg_index = if index.is_empty() {
            let current = next_arg;
            next_arg += 1;
            current
        } else {
            match index.parse::<usize>() {
                Ok(index) => index,
                Err(_) => usize::MAX,
            }
        };
        if let Some(value) = args.get(arg_index) {
            output.push_str(value);
        } else {
            output.push('§');
            output.push_str(&index);
        }
    }
    output
}
#[repr(C)]
pub(super) struct RuntimeText {
    pub(super) slice: FaberRtSliceV1,
    pub(super) value: String,
}
