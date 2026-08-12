//! Arena-owned regex pattern carriers for the LLVM host ABI (Stage 4Z).
//!
//! v1 constructs a pattern carrier only — no engine compile/validation.

use super::format::{store_text, text_value};
use super::RuntimeContext;
use radix_host_abi::{FaberRtPtrResultV1, FaberRtSliceV1, STATUS_INVALID_ARGUMENT, STATUS_PANIC,
};
use crate::abi::FaberRtContextV1;
use faber::Regex;
use std::ffi::{c_char, c_void, CStr};
use std::panic::{self, AssertUnwindSafe};

fn ffi_ptr(operation: impl FnOnce() -> FaberRtPtrResultV1) -> FaberRtPtrResultV1 {
    panic::catch_unwind(AssertUnwindSafe(operation))
        .unwrap_or(FaberRtPtrResultV1::failure(STATUS_PANIC))
}

fn runtime(context: *mut FaberRtContextV1) -> Option<&'static mut RuntimeContext> {
    (!context.is_null()).then(|| unsafe { &mut *context.cast::<RuntimeContext>() })
}

fn store_regex(runtime: &mut RuntimeContext, value: Regex) -> FaberRtPtrResultV1 {
    let boxed = super::StableBox::new(value);
    let handle = boxed.handle();
    runtime.regexes.push(boxed);
    FaberRtPtrResultV1::success(handle)
}

fn find_regex(runtime: &RuntimeContext, handle: *mut c_void) -> Option<&Regex> {
    runtime
        .regexes
        .iter()
        .find(|value| std::ptr::eq(value.as_ref(), handle.cast()))
        .map(super::StableBox::as_ref)
}

/// `textus ↦ regex` — preserve pattern text without engine validation.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_regex_from_text(
    context: *mut FaberRtContextV1,
    value: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let (Some(runtime), Some(text)) = (runtime(context), text_value(value)) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_regex(runtime, Regex::new(&text))
    })
}

/// `ascii ↦ regex`.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_regex_from_ascii(
    context: *mut FaberRtContextV1,
    value: *const c_char,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        if value.is_null() {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let bytes = unsafe { CStr::from_ptr(value) }.to_bytes();
        if !bytes.is_ascii() {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let text = String::from_utf8_lossy(bytes).into_owned();
        store_regex(runtime, Regex::new(&text))
    })
}

/// Pattern text extraction for diagnostics / display.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_regex_get_text(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(regex) = find_regex(runtime, handle) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_text(context, regex.pattern().to_owned())
    })
}

/// Static regex literal descriptor: `{ ptr pattern, ptr flags }` where
/// `pattern` is a NUL-terminated C string and `flags` is a NUL-terminated C
/// string or null. Matches the emitter's `REGEX_DESCRIPTOR_LAYOUT`.
#[repr(C)]
pub struct RegexLiteralDescriptorV1 {
    pub(crate) pattern: *const c_char,
    pub(crate) flags: *const c_char,
}

/// Construct a regex carrier from a static regex literal descriptor.
///
/// The literal path (`"…" ↦ regex` on a literal) preserves the pattern text
/// without engine validation, mirroring [`__faber_rt_v1_regex_from_text`]; the
/// descriptor carries the pattern bytes (and optional flags text) emitted as
/// static globals by the compiler.
///
/// # Safety
///
/// `context` must be live. `descriptor` must be a readable
/// [`RegexLiteralDescriptorV1`] whose `pattern` is a valid NUL-terminated C
/// string.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_regex_literal_1_ptr_to_ptr(
    context: *mut FaberRtContextV1,
    descriptor: *const RegexLiteralDescriptorV1,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let (Some(runtime), Some(descriptor)) = (
            runtime(context),
            (!descriptor.is_null()).then(|| unsafe { &*descriptor }),
        ) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        if descriptor.pattern.is_null() {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let bytes = unsafe { CStr::from_ptr(descriptor.pattern) }.to_bytes();
        let text = String::from_utf8_lossy(bytes).into_owned();
        store_regex(runtime, Regex::new(&text))
    })
}
