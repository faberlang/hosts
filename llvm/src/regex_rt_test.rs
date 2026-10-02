//! Regex conversion rows validate the pattern at construction (RD-9).

use super::*;
use crate::{__faber_rt_v1_init, __faber_rt_v1_shutdown, STATUS_OK};

fn context() -> *mut FaberRtContextV1 {
    let mut context = std::ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, std::ptr::null(), &raw mut context) },
        STATUS_OK
    );
    context
}

#[test]
fn from_text_rejects_a_pattern_outside_the_dialect() {
    let context = context();
    for (pattern, id) in [
        (&b"a(?=b)"[..], "lookahead"),
        (b"(", "syntax"),
        (b"(a)\\1", "backreference"),
        (b"(?g)a", "unsupported_flag"),
    ] {
        let slice = FaberRtSliceV1 {
            data: pattern.as_ptr(),
            len: pattern.len() as u64,
        };
        let result = unsafe { __faber_rt_v1_regex_from_text(context, &raw const slice) };
        assert_eq!(result.status, STATUS_INVALID_ARGUMENT);
        let payload = unsafe { &*result.value.cast::<FaberRtSliceV1>() };
        let text = unsafe { std::slice::from_raw_parts(payload.data, payload.len as usize) };
        let text = std::str::from_utf8(text).expect("payload is utf-8");
        assert!(text.starts_with(id), "{text}");
    }
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn from_ascii_and_literal_reject_an_invalid_pattern() {
    let context = context();
    let result = unsafe { __faber_rt_v1_regex_from_ascii(context, c"a(?=b)".as_ptr()) };
    assert_eq!(result.status, STATUS_INVALID_ARGUMENT);
    let descriptor = RegexLiteralDescriptorV1 {
        pattern: c"(".as_ptr(),
        flags: std::ptr::null(),
    };
    let result =
        unsafe { __faber_rt_v1_regex_literal_1_ptr_to_ptr(context, &raw const descriptor) };
    assert_eq!(result.status, STATUS_INVALID_ARGUMENT);
    assert!(!result.value.is_null());
    unsafe { __faber_rt_v1_shutdown(context) };
}
