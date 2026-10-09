//! Module display tokens: the words a program prints for a `bivalens`, the
//! null value and a tuple head. The print locale is the code locale, so a
//! non-Latin module hands its words to the host once at start-up through
//! `__faber_rt_v1_set_display_tokens`. A host whose entry was never called
//! prints the Latin words, exactly as before the entry existed.

use super::RuntimeContext;
use crate::abi::{FaberRtContextV1, FaberRtSliceV1, FaberRtStatusV1};
use crate::abi::{STATUS_INVALID_ARGUMENT, STATUS_OK, STATUS_PANIC};
use crate::format::text_value;
use faber::display::DisplayTokens;
use std::panic::{self, AssertUnwindSafe};

/// The tokens the context prints through (Latin until the entry is called).
pub(super) fn of(runtime: &RuntimeContext) -> DisplayTokens {
    runtime.display_tokens.unwrap_or(DisplayTokens::LATIN)
}

/// [`of`] for an ABI context pointer; a null context prints Latin.
pub(super) fn of_context(context: *mut FaberRtContextV1) -> DisplayTokens {
    if context.is_null() {
        return DisplayTokens::LATIN;
    }
    of(unsafe { &*context.cast::<RuntimeContext>() })
}

/// Store the module's four display words in the context.
///
/// Each argument is a text descriptor (`FaberRtSliceV1`, the text-literal
/// global layout). The words live for the rest of the process: `DisplayTokens`
/// borrows `'static` strings, so they are leaked once per distinct set, and
/// the entry runs once at program start.
///
/// # Safety
///
/// `context` must be live. Each word pointer must be a valid text descriptor.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_set_display_tokens(
    context: *mut FaberRtContextV1,
    true_word: *const FaberRtSliceV1,
    false_word: *const FaberRtSliceV1,
    none_word: *const FaberRtSliceV1,
    tuple_word: *const FaberRtSliceV1,
) -> FaberRtStatusV1 {
    panic::catch_unwind(AssertUnwindSafe(|| {
        if context.is_null() {
            return STATUS_INVALID_ARGUMENT;
        }
        let words = [true_word, false_word, none_word, tuple_word].map(text_value);
        let [Some(true_), Some(false_), Some(none), Some(tuple)] = words else {
            return STATUS_INVALID_ARGUMENT;
        };
        let runtime = unsafe { &mut *context.cast::<RuntimeContext>() };
        let leak = |word: String| -> &'static str { Box::leak(word.into_boxed_str()) };
        let current = of(runtime);
        if current.true_ == true_
            && current.false_ == false_
            && current.none == none
            && current.tuple == tuple
        {
            return STATUS_OK;
        }
        runtime.display_tokens = Some(DisplayTokens {
            true_: leak(true_),
            false_: leak(false_),
            none: leak(none),
            tuple: leak(tuple),
        });
        STATUS_OK
    }))
    .unwrap_or(STATUS_PANIC)
}
