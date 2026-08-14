//! Filesystem boundaries used by LLVM-host `norma:solum` providers.

use super::array::{store_array, RuntimeValue};
use super::format::{store_text, text_value};
use super::valor_aggregate::store_octeti;
use super::RuntimeContext;
use crate::abi::{
    FaberRtPtrResultV1, FaberRtSliceV1, FaberRtStatusV1, STATUS_INVALID_ARGUMENT, STATUS_IO_ERROR,
    STATUS_OK,
};
use radix_host_abi::VALUE_KIND_PTR;
use crate::abi::FaberRtContextV1;
use std::io::{self, BufRead};

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_solum_read_text(
    context: *mut FaberRtContextV1,
    path: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    let Some(path) = text_value(path) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    match std::fs::read_to_string(path) {
        Ok(text) => store_text(context, text),
        Err(_) => FaberRtPtrResultV1::failure(STATUS_IO_ERROR),
    }
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_solum_read_lines(
    context: *mut FaberRtContextV1,
    path: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    let Some(path) = text_value(path) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    let Ok(content) = std::fs::read_to_string(path) else {
        return FaberRtPtrResultV1::failure(STATUS_IO_ERROR);
    };
    if context.is_null() {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    }
    let runtime = unsafe { &mut *context.cast::<RuntimeContext>() };
    let mut values = Vec::new();
    for line in content.lines() {
        let result = store_text(context, line.to_owned());
        if result.status != STATUS_OK {
            return result;
        }
        values.push(RuntimeValue::Ptr(result.value));
    }
    store_array(runtime, VALUE_KIND_PTR, values)
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_solum_read_bytes(
    context: *mut FaberRtContextV1,
    path: *const FaberRtSliceV1,
) -> FaberRtPtrResultV1 {
    let Some(path) = text_value(path) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    if context.is_null() {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    }
    let runtime = unsafe { &mut *context.cast::<RuntimeContext>() };
    match std::fs::read(path) {
        Ok(bytes) => store_octeti(runtime, bytes),
        Err(_) => FaberRtPtrResultV1::failure(STATUS_IO_ERROR),
    }
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_solum_write_text(
    context: *mut FaberRtContextV1,
    path: *const FaberRtSliceV1,
    text: *const FaberRtSliceV1,
) -> FaberRtStatusV1 {
    if context.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }
    let (Some(path), Some(text)) = (text_value(path), text_value(text)) else {
        return STATUS_INVALID_ARGUMENT;
    };
    match std::fs::write(path, text) {
        Ok(()) => STATUS_OK,
        Err(_) => STATUS_IO_ERROR,
    }
}

/// `lege` — read one line from the active input stream (stdin).
///
/// Reads one line of stdin; the trailing newline is stripped. End of input
/// returns a null handle (`nihil`), matching `T ∪ nihil` semantics. The text
/// is stored in the arena so diagnostics can render it.
///
/// # Safety
///
/// `context` must be live.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_read_line_0_to_ptr(
    context: *mut FaberRtContextV1,
) -> FaberRtPtrResultV1 {
    if context.is_null() {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    }
    let mut line = String::new();
    match io::stdin().lock().read_line(&mut line) {
        Ok(0) => FaberRtPtrResultV1::success(std::ptr::null_mut()),
        Ok(_) => {
            let trimmed = line.trim_end_matches(['\n', '\r']).to_owned();
            store_text(context, trimmed)
        }
        Err(_) => FaberRtPtrResultV1::failure(STATUS_IO_ERROR),
    }
}
