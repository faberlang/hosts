//! Arena-owned regex carriers for the LLVM host ABI (Stage 4Z).
//!
//! Construction validates and compiles through `faber::Regex::new` (RD-9): a
//! pattern outside the dialect is the host's typed failure
//! (`STATUS_INVALID_ARGUMENT`) whose value is the arena text handle of the
//! rejection payload (a stable construct id, `": "`, then detail), so a
//! `cape` arm can read the id the way the runner and Rust targets do.

use super::RuntimeContext;
use super::array::{RuntimeValue, find_array, store_array};
use super::format::{store_text, text_value};
use crate::abi::FaberRtContextV1;
use crate::abi::{
    FaberRtPtrResultV1, FaberRtSliceV1, FaberRtStatusV1, STATUS_INVALID_ARGUMENT, STATUS_OK,
    STATUS_PANIC,
};
use faber::{Match, Regex, regex as ops};
use radix_host_abi::VALUE_KIND_PTR;
use std::cell::Cell;
use std::ffi::{CStr, c_char, c_void};
use std::panic::{self, AssertUnwindSafe};

fn ffi_ptr(operation: impl FnOnce() -> FaberRtPtrResultV1) -> FaberRtPtrResultV1 {
    panic::catch_unwind(AssertUnwindSafe(operation))
        .unwrap_or(FaberRtPtrResultV1::failure(STATUS_PANIC))
}

fn runtime(context: *mut FaberRtContextV1) -> Option<&'static mut RuntimeContext> {
    (!context.is_null()).then(|| unsafe { &mut *context.cast::<RuntimeContext>() })
}

fn store_regex(
    context: *mut FaberRtContextV1,
    runtime: &mut RuntimeContext,
    pattern: &str,
) -> FaberRtPtrResultV1 {
    let value = match Regex::new(pattern) {
        Ok(value) => value,
        Err(error) => {
            let payload = store_text(context, error.to_string());
            return FaberRtPtrResultV1 {
                status: STATUS_INVALID_ARGUMENT,
                value: payload.value,
            };
        }
    };
    let boxed = super::StableBox::new(value);
    let handle = boxed.handle();
    runtime
        .regex_by_handle
        .insert(handle as usize, runtime.regexes.len());
    runtime.regexes.push(boxed);
    FaberRtPtrResultV1::success(handle)
}

fn find_regex(runtime: &RuntimeContext, handle: *mut c_void) -> Option<&Regex> {
    let index = *runtime.regex_by_handle.get(&(handle as usize))?;
    runtime.regexes.get(index).map(super::StableBox::as_ref)
}

fn store_match(runtime: &mut RuntimeContext, value: Match) -> *mut c_void {
    let boxed = super::StableBox::new(value);
    let handle = boxed.handle();
    runtime
        .match_by_handle
        .insert(handle as usize, runtime.matches.len());
    runtime.matches.push(boxed);
    handle
}

fn find_match(runtime: &RuntimeContext, handle: *mut c_void) -> Option<&Match> {
    let index = *runtime.match_by_handle.get(&(handle as usize))?;
    runtime.matches.get(index).map(super::StableBox::as_ref)
}

/// The text behind a `textus` carrier (`ptr` to `{ ptr, i64 }`), borrowed for
/// the length of the call: the operations read the haystack without copying it.
fn text_str<'a>(text: *const FaberRtSliceV1) -> Option<&'a str> {
    if text.is_null() {
        return None;
    }
    let text = unsafe { &*text };
    let len = usize::try_from(text.len).ok()?;
    if len == 0 {
        return Some("");
    }
    if text.data.is_null() {
        return None;
    }
    std::str::from_utf8(unsafe { std::slice::from_raw_parts(text.data, len) }).ok()
}

fn ffi_status(operation: impl FnOnce() -> FaberRtStatusV1) -> FaberRtStatusV1 {
    panic::catch_unwind(AssertUnwindSafe(operation)).unwrap_or(STATUS_PANIC)
}

/// `none`: success with a null handle (a miss, an absent group, an unknown
/// name).
fn none() -> FaberRtPtrResultV1 {
    FaberRtPtrResultV1::success(std::ptr::null_mut())
}

fn optional_text(context: *mut FaberRtContextV1, value: Option<String>) -> FaberRtPtrResultV1 {
    match value {
        Some(value) => store_text(context, value),
        None => none(),
    }
}

/// An array of arena text handles (`lista<textus>`): the carrier `split` and
/// the closure form of `replace` use.
fn text_array(
    context: *mut FaberRtContextV1,
    runtime: &mut RuntimeContext,
    texts: Vec<String>,
) -> FaberRtPtrResultV1 {
    let mut values = Vec::with_capacity(texts.len());
    for text in texts {
        let result = store_text(context, text);
        if result.status != STATUS_OK {
            return result;
        }
        values.push(RuntimeValue::Ptr(result.value));
    }
    store_array(runtime, VALUE_KIND_PTR, values)
}

/// The strings of a `lista<textus>` handle; `None` when the handle is not a
/// pointer array or an element is not text.
fn array_texts(runtime: &RuntimeContext, handle: *mut c_void) -> Option<Vec<String>> {
    let array = find_array(runtime, handle)?;
    if array.kind != VALUE_KIND_PTR {
        return None;
    }
    array
        .values
        .iter()
        .map(|value| match value {
            RuntimeValue::Ptr(text) => text_str(text.cast_const().cast()).map(str::to_owned),
            _ => None,
        })
        .collect()
}

/// Run one handle-returning regex operation: resolve the context and regex,
/// then hand both to `operation`.
fn with_regex(
    context: *mut FaberRtContextV1,
    regex: *mut c_void,
    operation: impl FnOnce(&mut RuntimeContext, &Regex) -> FaberRtPtrResultV1,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        // The regex lives in a stable box owned by the context; cloning the
        // handle's value would copy the compiled program, so detach the borrow
        // from `runtime` the way the arena handles are detached.
        let Some(regex) = find_regex(runtime, regex).map(std::ptr::from_ref) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        operation(runtime, unsafe { &*regex })
    })
}

/// `textus ↦ regex` — validate and compile the pattern.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_regex_from_text(
    context: *mut FaberRtContextV1,
    value: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let (Some(runtime), Some(text)) = (runtime(context), text_value(value)) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_regex(context, runtime, &text)
    })
}

/// `ascii ↦ regex`.
#[unsafe(no_mangle)]
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
        store_regex(context, runtime, &text)
    })
}

/// Pattern text extraction for diagnostics / display.
#[unsafe(no_mangle)]
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

/// `regex.matches(text)`: writes 1 or 0 through `out`.
///
/// # Safety
///
/// `context` must be live, `regex` a handle from this runtime, `text` a
/// readable `textus` carrier and `out` a writable byte.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_regex_matches(
    context: *mut FaberRtContextV1,
    regex: *mut c_void,
    text: *const FaberRtSliceV1,
    out: *mut u8,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let (Some(runtime), Some(text), false) = (runtime(context), text_str(text), out.is_null())
        else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(regex) = find_regex(runtime, regex) else {
            return STATUS_INVALID_ARGUMENT;
        };
        unsafe { out.write(u8::from(ops::matches(regex, text))) };
        STATUS_OK
    })
}

/// `regex.find(text)`: the leftmost match handle, or null for `none`.
///
/// # Safety
///
/// As [`__faber_rt_v1_regex_matches`], without `out`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_regex_find(
    context: *mut FaberRtContextV1,
    regex: *mut c_void,
    text: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    let Some(text) = text_str(text) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    with_regex(context, regex, |runtime, regex| {
        match ops::find(regex, text) {
            Some(found) => FaberRtPtrResultV1::success(store_match(runtime, found)),
            None => none(),
        }
    })
}

/// `regex.find_all(text)`: a `lista` of match handles, in order.
///
/// # Safety
///
/// As [`__faber_rt_v1_regex_find`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_regex_find_all(
    context: *mut FaberRtContextV1,
    regex: *mut c_void,
    text: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    let Some(text) = text_str(text) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    with_regex(context, regex, |runtime, regex| {
        let values = ops::find_all(regex, text)
            .into_iter()
            .map(|found| RuntimeValue::Ptr(store_match(runtime, found)))
            .collect();
        store_array(runtime, VALUE_KIND_PTR, values)
    })
}

/// `regex.split(text)`: a `lista<textus>` of the pieces between the matches.
///
/// # Safety
///
/// As [`__faber_rt_v1_regex_find`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_regex_split(
    context: *mut FaberRtContextV1,
    regex: *mut c_void,
    text: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    let Some(text) = text_str(text) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    with_regex(context, regex, |runtime, regex| {
        text_array(context, runtime, ops::split(regex, text))
    })
}

/// `regex.replace(text, replacement)`: every match replaced by the literal
/// `replacement`.
///
/// # Safety
///
/// As [`__faber_rt_v1_regex_find`]; `replacement` is a readable `textus`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_regex_replace(
    context: *mut FaberRtContextV1,
    regex: *mut c_void,
    text: *const FaberRtSliceV1,
    replacement: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    let (Some(text), Some(replacement)) = (text_str(text), text_str(replacement)) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    with_regex(context, regex, |_, regex| {
        store_text(context, ops::replace(regex, text, replacement))
    })
}

/// The replacement step of the closure form of `replace`: the i-th match of
/// the iteration is replaced by the i-th text of `replacements`. The lowering
/// produces one text per match of the same iteration; a count that disagrees
/// is an invalid argument, never a partial result.
///
/// # Safety
///
/// As [`__faber_rt_v1_regex_find`]; `replacements` is a `lista<textus>`
/// handle from this runtime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_regex_replace_each(
    context: *mut FaberRtContextV1,
    regex: *mut c_void,
    text: *const FaberRtSliceV1,
    replacements: *mut c_void,
) -> FaberRtPtrResultV1 {
    let Some(text) = text_str(text) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    with_regex(context, regex, |runtime, regex| {
        let Some(replacements) = array_texts(runtime, replacements) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let taken = Cell::new(0usize);
        let out = ops::replace_with(regex, text, |_| {
            let index = taken.get();
            taken.set(index + 1);
            replacements.get(index).map_or("", String::as_str)
        });
        if taken.get() != replacements.len() {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        store_text(context, out)
    })
}

/// `textus.escape()`: a pattern text matching `text` literally.
///
/// # Safety
///
/// `context` must be live and `text` a readable `textus` carrier.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_regex_escape(
    context: *mut FaberRtContextV1,
    text: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| match text_str(text) {
        Some(text) => store_text(context, ops::escape(text)),
        None => FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT),
    })
}

fn with_match(
    context: *mut FaberRtContextV1,
    found: *mut c_void,
    operation: impl FnOnce(&Match) -> FaberRtPtrResultV1,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(found) = find_match(runtime, found) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        operation(found)
    })
}

/// `match.text()`.
///
/// # Safety
///
/// `context` must be live and `found` a match handle from this runtime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_regex_match_text(
    context: *mut FaberRtContextV1,
    found: *mut c_void,
) -> FaberRtPtrResultV1 {
    with_match(context, found, |found| {
        store_text(context, ops::match_text(found))
    })
}

fn match_offset(
    context: *mut FaberRtContextV1,
    found: *mut c_void,
    out: *mut i64,
    offset: fn(&Match) -> i64,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let (Some(runtime), false) = (runtime(context), out.is_null()) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(found) = find_match(runtime, found) else {
            return STATUS_INVALID_ARGUMENT;
        };
        unsafe { out.write(offset(found)) };
        STATUS_OK
    })
}

/// `match.start()`: the code-point offset where the match starts.
///
/// # Safety
///
/// As [`__faber_rt_v1_regex_match_text`]; `out` is a writable `i64`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_regex_match_start(
    context: *mut FaberRtContextV1,
    found: *mut c_void,
    out: *mut i64,
) -> FaberRtStatusV1 {
    match_offset(context, found, out, ops::match_start)
}

/// `match.end()`: the code-point offset where the match ends (exclusive).
///
/// # Safety
///
/// As [`__faber_rt_v1_regex_match_start`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_regex_match_end(
    context: *mut FaberRtContextV1,
    found: *mut c_void,
    out: *mut i64,
) -> FaberRtStatusV1 {
    match_offset(context, found, out, ops::match_end)
}

/// `match.group(index)`: the group's text handle, null for `none`.
///
/// # Safety
///
/// As [`__faber_rt_v1_regex_match_text`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_regex_match_group(
    context: *mut FaberRtContextV1,
    found: *mut c_void,
    index: i64,
) -> FaberRtPtrResultV1 {
    with_match(context, found, |found| {
        optional_text(context, ops::match_group(found, index))
    })
}

/// `match.named(name)`: the named group's text handle, null for `none`.
///
/// # Safety
///
/// As [`__faber_rt_v1_regex_match_text`]; `name` is a readable `textus`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_regex_match_named(
    context: *mut FaberRtContextV1,
    found: *mut c_void,
    name: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    let Some(name) = text_str(name) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    with_match(context, found, |found| {
        optional_text(context, ops::match_named(found, name))
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
/// The literal path validates and compiles the pattern like
/// [`__faber_rt_v1_regex_from_text`]; the
/// descriptor carries the pattern bytes (and optional flags text) emitted as
/// static globals by the compiler.
///
/// # Safety
///
/// `context` must be live. `descriptor` must be a readable
/// [`RegexLiteralDescriptorV1`] whose `pattern` is a valid NUL-terminated C
/// string.
#[unsafe(no_mangle)]
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
        store_regex(context, runtime, &text)
    })
}

#[cfg(test)]
#[path = "regex_rt_test.rs"]
mod tests;
