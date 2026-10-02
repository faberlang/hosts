//! Regex conversion rows validate the pattern at construction (RD-9); the
//! operation rows (RX-10d) run on the compiled carrier.

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

fn text(bytes: &'static [u8]) -> FaberRtSliceV1 {
    FaberRtSliceV1::from_static(bytes)
}

fn regex(context: *mut FaberRtContextV1, pattern: &'static [u8]) -> *mut c_void {
    let slice = text(pattern);
    let result = unsafe { __faber_rt_v1_regex_from_text(context, &raw const slice) };
    assert_eq!(result.status, STATUS_OK);
    result.value
}

fn read(handle: *mut c_void) -> String {
    assert!(!handle.is_null());
    let slice = unsafe { &*handle.cast::<FaberRtSliceV1>() };
    let bytes = unsafe { std::slice::from_raw_parts(slice.data, slice.len as usize) };
    String::from_utf8(bytes.to_vec()).expect("utf-8")
}

fn runtime_of<'a>(context: *mut FaberRtContextV1) -> &'a mut RuntimeContext {
    runtime(context).expect("live context")
}

fn texts(context: *mut FaberRtContextV1, array: *mut c_void) -> Vec<String> {
    let runtime = runtime_of(context);
    array_texts(runtime, array).expect("a text array")
}

#[test]
fn matches_reports_a_hit_and_a_miss() {
    let context = context();
    let digits = regex(context, b"[0-9]+");
    for (haystack, expected) in [(&b"ab 42"[..], 1), (b"none", 0)] {
        let haystack = FaberRtSliceV1 {
            data: haystack.as_ptr(),
            len: haystack.len() as u64,
        };
        let mut out = 7u8;
        let status = unsafe {
            __faber_rt_v1_regex_matches(context, digits, &raw const haystack, &raw mut out)
        };
        assert_eq!(status, STATUS_OK);
        assert_eq!(out, expected);
    }
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn find_reads_code_point_offsets_groups_and_names() {
    let context = context();
    let pair = regex(context, b"(?P<key>[a-z]+)=([0-9]+)?");
    let haystack = text("\u{1F600}\u{e9} ab=".as_bytes());
    let found = unsafe { __faber_rt_v1_regex_find(context, pair, &raw const haystack) };
    assert_eq!(found.status, STATUS_OK);
    assert_eq!(
        read(unsafe { __faber_rt_v1_regex_match_text(context, found.value) }.value),
        "ab="
    );
    let (mut start, mut end) = (0i64, 0i64);
    unsafe {
        assert_eq!(
            __faber_rt_v1_regex_match_start(context, found.value, &raw mut start),
            STATUS_OK
        );
        assert_eq!(
            __faber_rt_v1_regex_match_end(context, found.value, &raw mut end),
            STATUS_OK
        );
    }
    assert_eq!((start, end), (3, 6));
    let group = |index| unsafe { __faber_rt_v1_regex_match_group(context, found.value, index) };
    assert_eq!(read(group(0).value), "ab=");
    assert_eq!(read(group(1).value), "ab");
    // a group that did not take part, and an index past the last group
    assert!(group(2).value.is_null());
    assert_eq!(group(2).status, STATUS_OK);
    assert!(group(9).value.is_null());
    assert!(group(-1).value.is_null());
    let key = text(b"key");
    let named = unsafe { __faber_rt_v1_regex_match_named(context, found.value, &raw const key) };
    assert_eq!(read(named.value), "ab");
    let nope = text(b"nope");
    let unknown = unsafe { __faber_rt_v1_regex_match_named(context, found.value, &raw const nope) };
    assert!(unknown.value.is_null());
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn find_misses_with_a_null_handle_and_ok_status() {
    let context = context();
    let digits = regex(context, b"[0-9]+");
    let haystack = text(b"none here");
    let found = unsafe { __faber_rt_v1_regex_find(context, digits, &raw const haystack) };
    assert_eq!(found.status, STATUS_OK);
    assert!(found.value.is_null());
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn find_all_skips_an_empty_match_that_abuts_the_previous_one() {
    let context = context();
    let stars = regex(context, b"a*");
    let haystack = text(b"baaac");
    let all = unsafe { __faber_rt_v1_regex_find_all(context, stars, &raw const haystack) };
    assert_eq!(all.status, STATUS_OK);
    let runtime = runtime_of(context);
    let array = find_array(runtime, all.value).expect("array handle");
    let mut spans = Vec::new();
    for value in array.values.iter() {
        let RuntimeValue::Ptr(handle) = value else {
            panic!("match handles are pointers");
        };
        let (mut start, mut end) = (0i64, 0i64);
        unsafe {
            __faber_rt_v1_regex_match_start(context, handle, &raw mut start);
            __faber_rt_v1_regex_match_end(context, handle, &raw mut end);
        }
        spans.push((start, end));
    }
    assert_eq!(spans, [(0, 0), (1, 4), (5, 5)]);
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn split_replace_and_replace_each_run_one_iteration() {
    let context = context();
    let comma = regex(context, b",\\s*");
    let haystack = text(b",a, b,,c,");
    let pieces = unsafe { __faber_rt_v1_regex_split(context, comma, &raw const haystack) };
    assert_eq!(texts(context, pieces.value), ["", "a", "b", "", "c", ""]);

    let a = regex(context, b"a");
    let banana = text(b"banana");
    let replacement = text(b"$1");
    let replaced = unsafe {
        __faber_rt_v1_regex_replace(context, a, &raw const banana, &raw const replacement)
    };
    assert_eq!(read(replaced.value), "b$1n$1n$1");

    let found = unsafe { __faber_rt_v1_regex_find_all(context, a, &raw const banana) };
    assert_eq!(found.status, STATUS_OK);
    let runtime = runtime_of(context);
    let three = text_array(
        context,
        runtime,
        vec!["x".to_owned(), "yy".to_owned(), "".to_owned()],
    );
    let each =
        unsafe { __faber_rt_v1_regex_replace_each(context, a, &raw const banana, three.value) };
    assert_eq!(each.status, STATUS_OK);
    assert_eq!(read(each.value), "bxnyyn");
    // a count that disagrees with the iteration is an invalid argument
    let runtime = runtime_of(context);
    let two = text_array(context, runtime, vec!["x".to_owned(), "y".to_owned()]);
    let short =
        unsafe { __faber_rt_v1_regex_replace_each(context, a, &raw const banana, two.value) };
    assert_eq!(short.status, STATUS_INVALID_ARGUMENT);
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn escape_backslashes_the_metacharacters_only() {
    let context = context();
    let input = text("a.b*(c) \u{e9}".as_bytes());
    let escaped = unsafe { __faber_rt_v1_regex_escape(context, &raw const input) };
    assert_eq!(read(escaped.value), "a\\.b\\*\\(c\\) \u{e9}");
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn operations_reject_an_unknown_handle() {
    let context = context();
    let haystack = text(b"x");
    let bogus = 8usize as *mut c_void;
    let found = unsafe { __faber_rt_v1_regex_find(context, bogus, &raw const haystack) };
    assert_eq!(found.status, STATUS_INVALID_ARGUMENT);
    let mut start = 0i64;
    let status = unsafe { __faber_rt_v1_regex_match_start(context, bogus, &raw mut start) };
    assert_eq!(status, STATUS_INVALID_ARGUMENT);
    unsafe { __faber_rt_v1_shutdown(context) };
}
