mod abi;
mod array;
mod array_numeric;
mod cli;
mod collection_map;
mod convert;
mod failable;
mod format;
mod gpu_placement;
mod gradient;
mod instans;
mod intervallum;
mod octeti;
mod option;
mod provider;
mod regex_rt;
mod sermo;
mod solum;
mod sparsa;
mod tensor;
mod text;
mod valor_aggregate;
mod valor_genus;

// LLVM-host side of the FaberRt C ABI. radix-host-abi owns symbol names and
// status *codes*; this crate owns the repr(C) layouts (see `crate::abi`).
pub use crate::abi::{
    FaberRtContextV1, FaberRtExitV1, FaberRtPtrResultV1, FaberRtSliceV1, FaberRtStatusV1,
    STATUS_FALLIBLE, STATUS_INVALID_ARGUMENT, STATUS_IO_ERROR, STATUS_OK, STATUS_PANIC,
    STATUS_UNSUPPORTED,
};
pub use crate::failable::__faber_rt_v1_fallible_error;

use array::RuntimeArray;
#[cfg(test)]
use array::{
    __faber_rt_v1_array_clone, __faber_rt_v1_array_contains, __faber_rt_v1_array_extend,
    __faber_rt_v1_array_get, __faber_rt_v1_array_is_empty, __faber_rt_v1_array_length,
    __faber_rt_v1_array_new, __faber_rt_v1_array_option, __faber_rt_v1_array_push,
    __faber_rt_v1_array_range, __faber_rt_v1_array_reverse, __faber_rt_v1_array_set,
};
#[cfg(test)]
use array_numeric::{__faber_rt_v1_array_sort, __faber_rt_v1_array_sum};
#[cfg(test)]
use collection_map::{
    __faber_rt_v1_aggregate_set_index_ptr_i64, __faber_rt_v1_array_from_set,
    __faber_rt_v1_map_contains, __faber_rt_v1_map_delete, __faber_rt_v1_map_get,
    __faber_rt_v1_map_is_empty, __faber_rt_v1_map_keys, __faber_rt_v1_map_length,
    __faber_rt_v1_map_new, __faber_rt_v1_map_option, __faber_rt_v1_map_put,
    __faber_rt_v1_map_values, __faber_rt_v1_set_add, __faber_rt_v1_set_contains,
    __faber_rt_v1_set_delete, __faber_rt_v1_set_difference, __faber_rt_v1_set_from_array,
    __faber_rt_v1_set_intersection, __faber_rt_v1_set_is_empty, __faber_rt_v1_set_is_subset,
    __faber_rt_v1_set_is_superset, __faber_rt_v1_set_length, __faber_rt_v1_set_new,
    __faber_rt_v1_set_symmetric_difference, __faber_rt_v1_set_union,
};
use collection_map::{RuntimeMap, RuntimeSet};
#[cfg(test)]
use convert::{
    __faber_rt_v1_convert_runtime_1_ptr_to_ptr, __faber_rt_v1_valor_ascii, __faber_rt_v1_valor_f64,
    __faber_rt_v1_valor_get_ascii, __faber_rt_v1_valor_get_f64, __faber_rt_v1_valor_get_i1,
    __faber_rt_v1_valor_get_i64, __faber_rt_v1_valor_get_nihil, __faber_rt_v1_valor_get_text,
    __faber_rt_v1_valor_i1, __faber_rt_v1_valor_i64, __faber_rt_v1_valor_nihil,
    __faber_rt_v1_valor_text,
};
use faber::{display_bivalens, display_fractus, Valor};
#[cfg(test)]
use format::{
    __faber_rt_v1_format_1_ptr_to_ptr, __faber_rt_v1_format_f32, __faber_rt_v1_format_f64,
    __faber_rt_v1_format_i1, __faber_rt_v1_format_i64, __faber_rt_v1_format_i64_i64,
    __faber_rt_v1_format_i64_i64_i64, __faber_rt_v1_format_text, __faber_rt_v1_format_text_i64,
    __faber_rt_v1_format_text_i64_i1, __faber_rt_v1_format_text_text, __faber_rt_v1_text_f64,
    __faber_rt_v1_text_i1, __faber_rt_v1_text_i64, __faber_rt_v1_text_length,
};
use format::{text_value, RuntimeText};
#[cfg(test)]
use gpu_placement::{__faber_gpu_v1_copy_in, __faber_gpu_v1_readback, __faber_gpu_v1_sync};
#[cfg(test)]
use gradient::{
    __faber_rt_v1_gradient_accumulate, __faber_rt_v1_gradient_create, __faber_rt_v1_gradient_read,
    __faber_rt_v1_gradient_zero,
};
#[cfg(test)]
use instans::{
    __faber_rt_v1_compare_gt_2_ptr_ptr_to_i1, __faber_rt_v1_compare_gte_2_ptr_ptr_to_i1,
    __faber_rt_v1_compare_lt_2_ptr_ptr_to_i1, __faber_rt_v1_compare_lte_2_ptr_ptr_to_i1,
    __faber_rt_v1_instans_from_text, __faber_rt_v1_instans_from_valor,
    __faber_rt_v1_instans_get_text, __faber_rt_v1_instans_retag, __faber_rt_v1_tempus_nunc,
};
#[cfg(test)]
use intervallum::{
    __faber_rt_v1_interval_clamp, __faber_rt_v1_interval_clamp_i64,
    __faber_rt_v1_interval_contains, __faber_rt_v1_interval_intersect,
    __faber_rt_v1_interval_length, __faber_rt_v1_interval_materialize_array,
    __faber_rt_v1_interval_materialize_tensor, __faber_rt_v1_interval_new,
    __faber_rt_v1_interval_union,
};
#[cfg(test)]
use octeti::{
    __faber_rt_v1_octeti_append, __faber_rt_v1_octeti_from_ascii, __faber_rt_v1_octeti_from_text,
    __faber_rt_v1_octeti_get, __faber_rt_v1_octeti_get_ascii, __faber_rt_v1_octeti_get_text,
    __faber_rt_v1_octeti_length,
};
use option::RuntimeOption;
#[cfg(test)]
use option::{
    __faber_rt_v1_diagnostic_mone_option, __faber_rt_v1_diagnostic_nota_option,
    __faber_rt_v1_diagnostic_scribe_option, __faber_rt_v1_diagnostic_vide_option,
    __faber_rt_v1_option_get, __faber_rt_v1_option_get_or, __faber_rt_v1_option_is_present,
    __faber_rt_v1_option_none, __faber_rt_v1_option_some, __faber_rt_v1_option_unwrap_ptr,
};
#[cfg(test)]
use provider::{
    __faber_rt_v1_json_pange, __faber_rt_v1_json_solve, __faber_rt_v1_json_tempta,
    __faber_rt_v1_toml_solve, __faber_rt_v1_valor_cape,
};
#[cfg(test)]
use radix_host_abi::{
    FaberRtValueKindV1, ARRAY_OPTION_FIRST, ARRAY_OPTION_INDEX, ARRAY_OPTION_LAST,
    ARRAY_OPTION_REMOVE_FIRST, ARRAY_OPTION_REMOVE_LAST, ARRAY_RANGE_DROP_FIRST, ARRAY_RANGE_SLICE,
    ARRAY_RANGE_TAKE, ARRAY_RANGE_TAKE_LAST, INSTANS_PRECISION_MICROS, INSTANS_PRECISION_MILLIS,
    INSTANS_PRECISION_SECONDS, VALUE_KIND_ASCII, VALUE_KIND_F16, VALUE_KIND_F32, VALUE_KIND_F64,
    VALUE_KIND_I1, VALUE_KIND_I16, VALUE_KIND_I32, VALUE_KIND_I64, VALUE_KIND_I8, VALUE_KIND_PTR,
    VALUE_KIND_TEXT, VALUE_KIND_U16, VALUE_KIND_U32, VALUE_KIND_U64, VALUE_KIND_U8,
};
#[cfg(test)]
use regex_rt::{
    __faber_rt_v1_regex_from_ascii, __faber_rt_v1_regex_from_text, __faber_rt_v1_regex_get_text,
    __faber_rt_v1_regex_literal_1_ptr_to_ptr,
};
#[cfg(test)]
use sermo::{
    __faber_rt_v1_sermo_materialize_i64_or, __faber_rt_v1_sermo_materialize_text,
    __faber_rt_v1_sermo_materialize_valor, __faber_rt_v1_sermo_open,
    __faber_rt_v1_sermo_set_opener,
};
#[cfg(test)]
use solum::{
    __faber_rt_v1_read_line_0_to_ptr, __faber_rt_v1_solum_read_bytes,
    __faber_rt_v1_solum_read_lines,
};
use sparsa::RuntimeSparse;
#[cfg(test)]
use sparsa::{
    __faber_rt_v1_sparse_densify, __faber_rt_v1_sparse_from_tensor, __faber_rt_v1_sparse_get,
    __faber_rt_v1_sparse_new, __faber_rt_v1_sparse_nonzero, __faber_rt_v1_sparse_set,
};
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void};
use std::fmt::Display;
use std::io::{self, Write};
use std::ops::{Deref, DerefMut};
use std::panic::{self, AssertUnwindSafe};
use std::pin::Pin;
use std::ptr;
use tensor::RuntimeTensor;
#[cfg(test)]
use tensor::{
    __faber_rt_v1_tensor_add, __faber_rt_v1_tensor_convert, __faber_rt_v1_tensor_create,
    __faber_rt_v1_tensor_fill, __faber_rt_v1_tensor_flatten, __faber_rt_v1_tensor_from_flat,
    __faber_rt_v1_tensor_get, __faber_rt_v1_tensor_materialize, __faber_rt_v1_tensor_matmul,
    __faber_rt_v1_tensor_mean, __faber_rt_v1_tensor_mul, __faber_rt_v1_tensor_new,
    __faber_rt_v1_tensor_rank, __faber_rt_v1_tensor_reshape, __faber_rt_v1_tensor_set,
    __faber_rt_v1_tensor_shape, __faber_rt_v1_tensor_slice, __faber_rt_v1_tensor_sub,
    __faber_rt_v1_tensor_sum,
};
#[cfg(test)]
use text::{
    __faber_rt_v1_ascii_truthy, __faber_rt_v1_text_concat, __faber_rt_v1_text_contains,
    __faber_rt_v1_text_ends_with, __faber_rt_v1_text_is_empty, __faber_rt_v1_text_lowercase,
    __faber_rt_v1_text_parse_float, __faber_rt_v1_text_parse_integer, __faber_rt_v1_text_replace,
    __faber_rt_v1_text_slice, __faber_rt_v1_text_split, __faber_rt_v1_text_starts_with,
    __faber_rt_v1_text_trim, __faber_rt_v1_text_truthy, __faber_rt_v1_text_uppercase,
};
#[cfg(test)]
use valor_aggregate::{
    __faber_rt_v1_octeti_new, __faber_rt_v1_valor_array, __faber_rt_v1_valor_get_array,
    __faber_rt_v1_valor_get_map, __faber_rt_v1_valor_get_octeti, __faber_rt_v1_valor_map,
    __faber_rt_v1_valor_octeti,
};
#[cfg(test)]
use valor_genus::{__faber_rt_v1_valor_genus, __faber_rt_v1_valor_get_genus};

/// Owns a pinned heap allocation whose address is exported as an opaque ABI handle.
///
/// The host returns pointers into these allocations, so the allocation must not move
/// when the owning context's vectors grow.
struct StableBox<T: ?Sized> {
    value: Pin<Box<T>>,
}

impl<T> StableBox<T> {
    fn new(value: T) -> Self {
        Self {
            value: Box::pin(value),
        }
    }
}

impl<T: ?Sized> StableBox<T> {
    fn from_box(value: Box<T>) -> Self {
        Self {
            value: Pin::from(value),
        }
    }

    fn as_ref(&self) -> &T {
        self.value.as_ref().get_ref()
    }

    fn handle(&self) -> *mut std::ffi::c_void {
        std::ptr::from_ref(self.as_ref()).cast_mut().cast()
    }
}

impl<T: ?Sized + Unpin> StableBox<T> {
    fn as_mut(&mut self) -> &mut T {
        self.value.as_mut().get_mut()
    }
}

impl<T: ?Sized> Deref for StableBox<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<T: ?Sized + Unpin> DerefMut for StableBox<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut()
    }
}

struct RuntimeContext {
    /// Process argumenta captured at `__faber_rt_v1_init`, excluding the host
    /// argv[0] program path (Faber argumenta semantics: `std::env::args()`
    /// parity).
    arguments: Vec<Vec<u8>>,
    /// Typed CLI value table produced by `__faber_rt_v1_cli_parse` (S8.2).
    cli_table: Option<StableBox<cli::RuntimeCliTable>>,
    texts: Vec<StableBox<RuntimeText>>,
    valors: Vec<StableBox<Valor>>,
    ascii: Vec<StableBox<[u8]>>,
    octeti: Vec<StableBox<Vec<u8>>>,
    numeric_boxes: Vec<StableBox<i64>>,
    instants: Vec<StableBox<faber::Instans>>,
    arrays: Vec<StableBox<RuntimeArray>>,
    array_by_handle: HashMap<usize, usize>,
    options: Vec<StableBox<RuntimeOption>>,
    maps: Vec<StableBox<RuntimeMap>>,
    sets: Vec<StableBox<RuntimeSet>>,
    tensors: Vec<StableBox<RuntimeTensor>>,
    tensor_by_handle: HashMap<usize, usize>,
    sparses: Vec<StableBox<RuntimeSparse>>,
    gradients: Vec<StableBox<gradient::GradientStorage>>,
    gradient_views: Vec<StableBox<gradient::GradientViewV1>>,
    regexes: Vec<StableBox<faber::Regex>>,
    intervals: Vec<StableBox<faber::Intervallum<i64>>>,
    union_boxes: Vec<StableBox<*mut std::ffi::c_void>>,
    sermos: Vec<StableBox<faber::frame::Sermo>>,
}

/// Initialize one process-lifetime LLVM host context.
///
/// # Safety
///
/// `out_context` must be writable. When `argc` is positive, `argv` must point
/// to `argc` valid C strings. A successful context must be shut down exactly
/// once with [`__faber_rt_v1_shutdown`].
#[allow(clippy::similar_names)]
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_init(
    argc: c_int,
    argv: *const *const c_char,
    out_context: *mut *mut FaberRtContextV1,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        if out_context.is_null() || argc < 0 || (argc > 0 && argv.is_null()) {
            return STATUS_INVALID_ARGUMENT;
        }
        // SAFETY: `argc` is checked non-negative by the guard above.
        let argc = usize::try_from(argc).unwrap_or(0);
        // Faber argumenta semantics: argv excludes the host argv[0] program
        // path (the Rust oracle's `std::env::args()` excludes it too), so the
        // captured context holds exactly the program arguments.
        let mut arguments = Vec::with_capacity(argc.saturating_sub(1));
        for index in 1..argc {
            let value = *argv.add(index);
            if value.is_null() {
                return STATUS_INVALID_ARGUMENT;
            }
            arguments.push(std::ffi::CStr::from_ptr(value).to_bytes().to_vec());
        }
        let context = Box::new(RuntimeContext {
            arguments,
            cli_table: None,
            texts: Vec::new(),
            valors: Vec::new(),
            ascii: Vec::new(),
            octeti: Vec::new(),
            numeric_boxes: Vec::new(),
            instants: Vec::new(),
            arrays: Vec::new(),
            array_by_handle: HashMap::new(),
            options: Vec::new(),
            maps: Vec::new(),
            sets: Vec::new(),
            tensors: Vec::new(),
            tensor_by_handle: HashMap::new(),
            sparses: Vec::new(),
            gradients: Vec::new(),
            gradient_views: Vec::new(),
            regexes: Vec::new(),
            intervals: Vec::new(),
            union_boxes: Vec::new(),
            sermos: Vec::new(),
        });
        *out_context = Box::into_raw(context).cast();
        STATUS_OK
    })
}

/// Release a context returned by [`__faber_rt_v1_init`].
///
/// # Safety
///
/// `context` must be null or a live context returned by this runtime and not
/// previously shut down.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_shutdown(context: *mut FaberRtContextV1) {
    if context.is_null() {
        return;
    }
    drop(panic::catch_unwind(AssertUnwindSafe(|| {
        drop(Box::from_raw(context.cast::<RuntimeContext>()));
        drop(io::stdout().flush());
        drop(io::stderr().flush());
    })));
}

/// Return the process argumenta captured at [`__faber_rt_v1_init`] as an
/// arena-owned `lista<textus>` handle.
///
/// Faber argumenta semantics: the list excludes the host argv[0] program path
/// (the Rust oracle's `std::env::args()` excludes it too), so the returned
/// elements are exactly the program arguments the Rust lane observes.
///
/// # Safety
///
/// `context` must be null or a live runtime context.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_arguments(
    context: *mut FaberRtContextV1,
) -> crate::abi::FaberRtPtrResultV1 {
    format::ffi_ptr_result(|| {
        let Some(runtime) = (unsafe { array::runtime_mut(context) }) else {
            return crate::abi::FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let mut values = Vec::with_capacity(runtime.arguments.len());
        for argument in &runtime.arguments {
            let text =
                format::store_text_owned(context, String::from_utf8_lossy(argument).into_owned());
            values.push(array::RuntimeValue::Ptr(text));
        }
        array::store_array(runtime, radix_host_abi::VALUE_KIND_PTR, values)
    })
}

/// Write one `nota` text payload followed by its canonical newline.
///
/// # Safety
///
/// `context` must be live. `text.data` must be readable for `text.len` bytes,
/// except that a null pointer is allowed when the length is zero.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_write_nota_text(
    context: *mut FaberRtContextV1,
    text: FaberRtSliceV1,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        if context.is_null() || (text.len > 0 && text.data.is_null()) {
            return STATUS_INVALID_ARGUMENT;
        }
        let Ok(len) = usize::try_from(text.len) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let bytes = if len == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(text.data, len)
        };
        let mut stdout = io::stdout().lock();
        match stdout
            .write_all(bytes)
            .and_then(|()| stdout.write_all(b"\n"))
            .and_then(|()| stdout.flush())
        {
            Ok(()) => STATUS_OK,
            Err(_) => STATUS_IO_ERROR,
        }
    })
}

/// Evaluate one assertion without allowing a panic to cross the C ABI.
///
/// # Safety
///
/// `context` must be live.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_assert(
    context: *mut FaberRtContextV1,
    condition: u8,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        if context.is_null() {
            STATUS_INVALID_ARGUMENT
        } else if condition == 0 {
            STATUS_PANIC
        } else {
            STATUS_OK
        }
    })
}

/// Evaluate one assertion and report its literal message on failure.
///
/// # Safety
///
/// `context` must be live. `message` follows the slice validity contract of
/// [`__faber_rt_v1_write_nota_text`].
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_assert_message(
    context: *mut FaberRtContextV1,
    condition: u8,
    message: FaberRtSliceV1,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        if context.is_null() || (message.len > 0 && message.data.is_null()) {
            return STATUS_INVALID_ARGUMENT;
        }
        if condition != 0 {
            return STATUS_OK;
        }
        let Ok(len) = usize::try_from(message.len) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let bytes = if len == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(message.data, len)
        };
        let mut stderr = io::stderr().lock();
        match stderr
            .write_all(bytes)
            .and_then(|()| stderr.write_all(b"\n"))
            .and_then(|()| stderr.flush())
        {
            Ok(()) => STATUS_PANIC,
            Err(_) => STATUS_IO_ERROR,
        }
    })
}

fn write_diagnostic(
    context: *mut FaberRtContextV1,
    stderr: bool,
    value: impl Display,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        if context.is_null() {
            return STATUS_INVALID_ARGUMENT;
        }
        let result = if stderr {
            let mut output = io::stderr().lock();
            writeln!(output, "{value}").and_then(|()| output.flush())
        } else {
            let mut output = io::stdout().lock();
            writeln!(output, "{value}").and_then(|()| output.flush())
        };
        match result {
            Ok(()) => STATUS_OK,
            Err(_) => STATUS_IO_ERROR,
        }
    })
}

fn write_text_diagnostic(
    context: *mut FaberRtContextV1,
    stderr: bool,
    value: *const FaberRtSliceV1,
) -> FaberRtStatusV1 {
    if value.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }
    let Some(value) = text_value(value) else {
        return STATUS_INVALID_ARGUMENT;
    };
    write_diagnostic(context, stderr, value)
}

fn write_ascii_diagnostic(
    context: *mut FaberRtContextV1,
    stderr: bool,
    value: *const u8,
) -> FaberRtStatusV1 {
    if value.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }
    let Ok(value) = unsafe { std::ffi::CStr::from_ptr(value.cast()) }.to_str() else {
        return STATUS_INVALID_ARGUMENT;
    };
    write_diagnostic(context, stderr, value)
}

fn unsupported_opaque_diagnostic(context: *mut FaberRtContextV1) -> FaberRtStatusV1 {
    if context.is_null() {
        STATUS_INVALID_ARGUMENT
    } else {
        STATUS_UNSUPPORTED
    }
}

/// Render an opaque handle the LLVM host can display: an arena-owned `lista`
/// handle (numeric or text elements), an `octeti` byte payload, a `valor`, a
/// `tabula`, or a `copia`. Each renders in the Rust oracle's Debug shape
/// (`[1, 2, 3]` / `["prima", "secunda"]` / `[112, 114, …]` /
/// `Json(Tabula({...}))` / `{1, 2, 3}`). Returns `None` for unrecognized or
/// unsupported handles (fail-closed).
pub(crate) fn opaque_value_text(runtime: &RuntimeContext, handle: *mut c_void) -> Option<String> {
    if let Some(array) = array::find_array(runtime, handle) {
        let mut rendered = Vec::with_capacity(array.values.len());
        for element in &array.values {
            rendered.push(opaque_element_text(runtime, array.kind, element)?);
        }
        return Some(format!("[{}]", rendered.join(", ")));
    }
    if let Some(bytes) = valor_aggregate::find_octeti(runtime, handle) {
        return Some(format!("{bytes:?}"));
    }
    if let Some(instans) = instans::find(runtime, handle) {
        // L28 (ab91f49f): a raw `instans` handle in a grouped multi-arg nota
        // renders its Rust-oracle Debug shape — the same carrier
        // `__faber_rt_v1_instans_display` uses for the per-argument path.
        return Some(format!("{instans:?}"));
    }
    if let Some(valor) = convert::find_valor(runtime, handle) {
        return Some(match valor {
            // `bytes ↦ valor` boxes the payload; the Rust oracle renders the
            // equivalent `Valor::Lista` of numeri as the byte-list Debug shape,
            // so an octeti payload renders `[222, 173]` rather than
            // `display_valor`'s `<n bytes>` placeholder.
            Valor::Octeti(bytes) => format!("{bytes:?}"),
            other => faber::display_valor(other),
        });
    }
    if let Some(map) = collection_map::find_map(runtime, handle) {
        // JSON-literal `tabula` values render in the Rust oracle's derived
        // `Json(Valor::Tabula({...}))` Debug shape. Non-text keys fail closed.
        let mut entries = std::collections::BTreeMap::new();
        for (key, value) in &map.entries {
            let Some(Valor::Textus(key)) =
                valor_aggregate::runtime_value_to_valor(runtime, map.key_kind, *key)
            else {
                return None;
            };
            let Some(value) =
                valor_aggregate::runtime_value_to_valor(runtime, map.value_kind, *value)
            else {
                return None;
            };
            entries.insert(key, value);
        }
        let valor = Valor::Tabula(entries);
        return Some(format!("Json({valor:?})"));
    }
    if let Some(set) = collection_map::find_set(runtime, handle) {
        // `copia` values render `{1, 2, 3}` in stored order (the Rust oracle's
        // `HashSet` Debug order is per-instance nondeterministic, so no
        // byte-exact order is guaranteed).
        let mut rendered = Vec::with_capacity(set.values.len());
        for element in &set.values {
            rendered.push(opaque_element_text(runtime, set.kind, *element)?);
        }
        return Some(format!("{{{}}}", rendered.join(", ")));
    }
    None
}

/// Render one `lista`/`copia` element in the Rust oracle's Debug shape.
fn opaque_element_text(
    runtime: &RuntimeContext,
    kind: radix_host_abi::FaberRtValueKindV1,
    element: array::RuntimeValue,
) -> Option<String> {
    Some(match (kind, element) {
        (radix_host_abi::VALUE_KIND_PTR, array::RuntimeValue::Ptr(element_handle)) => {
            // Text payload (arena handle or static text-literal descriptor)
            // quotes it (`"prima"`), matching `Vec<String>` Debug shape.
            let payload = format::find_text(runtime, element_handle)
                .map(|text| text.value.clone())
                .or_else(|| format::text_value(element_handle.cast()));
            format!("{:?}", payload?)
        }
        (radix_host_abi::VALUE_KIND_I1, array::RuntimeValue::I1(value)) => {
            format!("{:?}", value != 0)
        }
        (radix_host_abi::VALUE_KIND_I8, array::RuntimeValue::I8(value)) => {
            format!("{value}")
        }
        (radix_host_abi::VALUE_KIND_I16, array::RuntimeValue::I16(value)) => {
            format!("{value}")
        }
        (radix_host_abi::VALUE_KIND_I32, array::RuntimeValue::I32(value)) => {
            format!("{value}")
        }
        (radix_host_abi::VALUE_KIND_I64, array::RuntimeValue::I64(value)) => {
            format!("{value}")
        }
        (radix_host_abi::VALUE_KIND_U8, array::RuntimeValue::U8(value)) => {
            format!("{value}")
        }
        (radix_host_abi::VALUE_KIND_U16, array::RuntimeValue::U16(value)) => {
            format!("{value}")
        }
        (radix_host_abi::VALUE_KIND_U32, array::RuntimeValue::U32(value)) => {
            format!("{value}")
        }
        (radix_host_abi::VALUE_KIND_U64, array::RuntimeValue::U64(value)) => {
            format!("{value}")
        }
        (radix_host_abi::VALUE_KIND_F32, array::RuntimeValue::F32(value)) => display_fractus(value),
        (radix_host_abi::VALUE_KIND_F64, array::RuntimeValue::F64(value)) => display_fractus(value),
        _ => return None,
    })
}

/// Render an opaque `nota`/`mone` handle the LLVM host can display (see
/// [`opaque_value_text`]). Returns `None` for unrecognized or unsupported
/// handles (fail-closed).
fn opaque_diagnostic_text(runtime: &RuntimeContext, handle: *mut c_void) -> Option<String> {
    opaque_value_text(runtime, handle)
}

/// Render an opaque `nota`/`mone` value (see [`opaque_diagnostic_text`]).
///
/// # Safety
///
/// `context` must be null or a live runtime context. `value` is only used for
/// pointer-equality arena lookups; it is never dereferenced directly.
fn render_opaque_diagnostic(
    context: *mut FaberRtContextV1,
    stderr: bool,
    value: *const u8,
) -> FaberRtStatusV1 {
    if context.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }
    let runtime = unsafe { &*context.cast::<RuntimeContext>() };
    let handle = value.cast_mut().cast::<c_void>();
    let Some(text) = opaque_diagnostic_text(runtime, handle) else {
        return unsupported_opaque_diagnostic(context);
    };
    write_diagnostic(context, stderr, text)
}

/// Report an opaque `nota` value (`lista<textus>` / `octeti` when displayable).
///
/// # Safety
///
/// `context` must be null or a live runtime context. `value` is only used for
/// pointer-equality arena lookups.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_nota_ptr(
    context: *mut FaberRtContextV1,
    value: *const u8,
) -> FaberRtStatusV1 {
    render_opaque_diagnostic(context, false, value)
}

/// Report a text `nota` value.
///
/// # Safety
///
/// `context` must be null or a live runtime context. `value` must point to a
/// readable [`FaberRtSliceV1`], with readable data for its length.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_nota_text(
    context: *mut FaberRtContextV1,
    value: *const FaberRtSliceV1,
) -> FaberRtStatusV1 {
    write_text_diagnostic(context, false, value)
}

/// Report an ASCII `nota` value.
///
/// # Safety
///
/// `context` must be null or a live runtime context. `value` must point to a
/// valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_nota_ascii(
    context: *mut FaberRtContextV1,
    value: *const u8,
) -> FaberRtStatusV1 {
    write_ascii_diagnostic(context, false, value)
}

/// Report an integer `nota` value.
///
/// # Safety
///
/// `context` must be null or a live runtime context.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_nota_i64(
    context: *mut FaberRtContextV1,
    value: i64,
) -> FaberRtStatusV1 {
    write_diagnostic(context, false, value)
}

/// Report an unsigned 64-bit `nota` value.
///
/// # Safety
///
/// `context` must be null or a live runtime context.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_nota_u64(
    context: *mut FaberRtContextV1,
    value: u64,
) -> FaberRtStatusV1 {
    write_diagnostic(context, false, value)
}

/// Report a boolean `nota` value.
///
/// # Safety
///
/// `context` must be null or a live runtime context.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_nota_i1(
    context: *mut FaberRtContextV1,
    value: u8,
) -> FaberRtStatusV1 {
    write_diagnostic(context, false, display_bivalens(value != 0))
}

/// Report an f32 `nota` value.
///
/// # Safety
///
/// `context` must be null or a live runtime context.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_nota_f32(
    context: *mut FaberRtContextV1,
    value: f32,
) -> FaberRtStatusV1 {
    write_diagnostic(context, false, display_fractus(value))
}

/// Report an f64 `nota` value.
///
/// # Safety
///
/// `context` must be null or a live runtime context.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_nota_f64(
    context: *mut FaberRtContextV1,
    value: f64,
) -> FaberRtStatusV1 {
    write_diagnostic(context, false, display_fractus(value))
}

/// Report an i8 `nota` value.
///
/// # Safety
///
/// `context` must be null or a live runtime context.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_nota_i8(
    context: *mut FaberRtContextV1,
    value: i8,
) -> FaberRtStatusV1 {
    write_diagnostic(context, false, value)
}

/// Report an i32 `nota` value.
///
/// # Safety
///
/// `context` must be null or a live runtime context.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_nota_i32(
    context: *mut FaberRtContextV1,
    value: i32,
) -> FaberRtStatusV1 {
    write_diagnostic(context, false, value)
}

/// Report an unsupported opaque `mone` value.
///
/// # Safety
///
/// `context` must be null or a live runtime context. `_value` is ignored.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_mone_ptr(
    context: *mut FaberRtContextV1,
    value: *const u8,
) -> FaberRtStatusV1 {
    render_opaque_diagnostic(context, true, value)
}

/// Report a text `mone` value.
///
/// # Safety
///
/// `context` must be null or a live runtime context. `value` must point to a
/// readable [`FaberRtSliceV1`], with readable data for its length.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_mone_text(
    context: *mut FaberRtContextV1,
    value: *const FaberRtSliceV1,
) -> FaberRtStatusV1 {
    write_text_diagnostic(context, true, value)
}

/// Report an ASCII `mone` value.
///
/// # Safety
///
/// `context` must be null or a live runtime context. `value` must point to a
/// valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_mone_ascii(
    context: *mut FaberRtContextV1,
    value: *const u8,
) -> FaberRtStatusV1 {
    write_ascii_diagnostic(context, true, value)
}

/// Report an integer `mone` value.
///
/// # Safety
///
/// `context` must be null or a live runtime context.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_mone_i64(
    context: *mut FaberRtContextV1,
    value: i64,
) -> FaberRtStatusV1 {
    write_diagnostic(context, true, value)
}

/// Report an unsupported opaque `vide` value.
///
/// # Safety
///
/// `context` must be null or a live runtime context. `_value` is ignored.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_vide_ptr(
    context: *mut FaberRtContextV1,
    _value: *const u8,
) -> FaberRtStatusV1 {
    unsupported_opaque_diagnostic(context)
}

/// Report a text `vide` value.
///
/// # Safety
///
/// `context` must be null or a live runtime context. `value` must point to a
/// readable [`FaberRtSliceV1`], with readable data for its length.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_vide_text(
    context: *mut FaberRtContextV1,
    value: *const FaberRtSliceV1,
) -> FaberRtStatusV1 {
    write_text_diagnostic(context, false, value)
}

/// Report an ASCII `vide` value.
///
/// # Safety
///
/// `context` must be null or a live runtime context. `value` must point to a
/// valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_vide_ascii(
    context: *mut FaberRtContextV1,
    value: *const u8,
) -> FaberRtStatusV1 {
    write_ascii_diagnostic(context, false, value)
}

/// Report an integer `vide` value.
///
/// # Safety
///
/// `context` must be null or a live runtime context.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_diagnostic_vide_i64(
    context: *mut FaberRtContextV1,
    value: i64,
) -> FaberRtStatusV1 {
    write_diagnostic(context, false, value)
}

/// Emit a fatal diagnostic and abort without unwinding across the C boundary.
///
/// # Safety
///
/// The context and message slice follow the same validity requirements as
/// [`__faber_rt_v1_write_nota_text`]. This function never returns.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_fatal(
    context: *mut FaberRtContextV1,
    message: FaberRtSliceV1,
) -> ! {
    if !context.is_null() && (message.len == 0 || !message.data.is_null()) {
        if let Ok(len) = usize::try_from(message.len) {
            let bytes = if len == 0 {
                &[]
            } else {
                std::slice::from_raw_parts(message.data, len)
            };
            drop(io::stderr().write_all(bytes));
            drop(io::stderr().write_all(b"\n"));
            drop(io::stderr().flush());
        }
    }
    std::process::abort()
}

/// Abort for a message whose opaque runtime representation has no byte-length
/// contract at this ABI boundary.
///
/// # Safety
///
/// `context` must be live. `message` is intentionally never dereferenced.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_fatal_opaque(
    context: *mut FaberRtContextV1,
    _message: *const u8,
) -> ! {
    if !context.is_null() {
        drop(io::stderr().write_all(b"fatal error\n"));
        drop(io::stderr().flush());
    }
    std::process::abort()
}

fn ffi_status(operation: impl FnOnce() -> FaberRtStatusV1) -> FaberRtStatusV1 {
    panic::catch_unwind(AssertUnwindSafe(operation)).unwrap_or(STATUS_PANIC)
}

/// Panic on checked-arithmetic overflow with the Rust oracle's exact message
/// (`numerus overflow`, from the generated `checked_add(…).expect("numerus
/// overflow")`). The debug-built Rust lane panics on `numerus` overflow, so
/// the LLVM host must too (L19 `operatores/numerus-overflow.fab`); wrapping
/// stays on the explicit `modulus<W>` modular-word type. Never returns.
///
/// # Safety
///
/// `context` must be null or a live runtime context.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_numerus_overflow(context: *mut FaberRtContextV1) -> ! {
    if !context.is_null() {
        drop(io::stderr().write_all(b"numerus overflow\n"));
        drop(io::stderr().flush());
    }
    std::process::abort()
}

#[cfg(not(test))]
extern "C" {
    fn __faber_program_entry_v1(context: *mut FaberRtContextV1) -> FaberRtExitV1;
}

#[cfg(not(test))]
#[no_mangle]
#[allow(clippy::similar_names)]
/// C process entry owned by the LLVM host runtime.
///
/// # Safety
///
/// The platform launcher must provide the normal C `argc`/`argv` contract.
pub unsafe extern "C" fn main(argc: c_int, argv: *const *const c_char) -> c_int {
    let mut context = ptr::null_mut();
    let init = __faber_rt_v1_init(argc, argv, &raw mut context);
    if !init.is_ok() {
        return init.code;
    }
    let outcome = panic::catch_unwind(AssertUnwindSafe(|| __faber_program_entry_v1(context)))
        .unwrap_or(FaberRtExitV1 {
            process_code: STATUS_PANIC.code,
            status: STATUS_PANIC,
        });
    __faber_rt_v1_shutdown(context);
    if outcome.status.is_ok() {
        outcome.process_code
    } else {
        outcome.status.code
    }
}

#[cfg(test)]
#[path = "lib_test.rs"]
mod tests;
