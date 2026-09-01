//! W13 collection/scalar display arenas for the portable product host.
//!
//! Mirrors the LLVM host's opaque-value model (faber-runtime `array.rs` /
//! `collection_map.rs` / `option.rs`): arena-owned `lista`/`copia`/`tabula`
//! and option values, each tagged with the shared `VALUE_KIND_*` element
//! identity, render in the Rust-oracle Debug shapes when a `nota`/`vide`
//! diagnostic resolves the handle. The wasm row signatures carry scalar
//! values widened to i64 (there are no pointer carriers in wasm); the host
//! interprets each value per its declared kind.
//!
//! W14 tensor arena: dense tensor values (element kind + row-major shape +
//! flat values) stored arena-style like the collections/maps/options, so the
//! wasm tensor rows (`tensor_new/create/from_flat/get/set/…`) keep the same
//! opaque-handle contract the collection rows use.

use radix_host_abi::{
    VALUE_KIND_ASCII, VALUE_KIND_F32, VALUE_KIND_F64, VALUE_KIND_I1, VALUE_KIND_I8, VALUE_KIND_I16,
    VALUE_KIND_I32, VALUE_KIND_I64, VALUE_KIND_PTR, VALUE_KIND_TEXT, VALUE_KIND_U8, VALUE_KIND_U16,
    VALUE_KIND_U32, VALUE_KIND_U64, VALUE_KIND_VALOR,
};

/// Kind-tagged element/option payload value (mirrors the LLVM host
/// `RuntimeValue`; wasm i32 handles for text/aggregate elements).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum RuntimeValue {
    I1(bool),
    I32(i32),
    I64(i64),
    F64(f64),
    /// Opaque handle (text/aggregate/valor element).
    Handle(i32),
}

/// One dense tensor: element kind + row-major shape dims + flat element
/// values (W14). The shape mirrors the LLVM host's `Tensor` metadata; the
/// element kind tags the flat values exactly like a collection's.
#[derive(Debug)]
pub(crate) struct TensorValue {
    pub(crate) kind: u32,
    pub(crate) shape: Vec<i64>,
    pub(crate) values: Vec<RuntimeValue>,
}

/// One collection (`lista` or `copia`): element kind + stored values.
#[derive(Debug)]
pub(crate) struct CollectionValue {
    pub(crate) set: bool,
    pub(crate) kind: u32,
    pub(crate) values: Vec<RuntimeValue>,
}

/// One map (`tabula`): key/value kinds + entries in stored order.
#[derive(Debug)]
pub(crate) struct MapValue {
    pub(crate) key_kind: u32,
    pub(crate) value_kind: u32,
    pub(crate) entries: Vec<(RuntimeValue, RuntimeValue)>,
}

/// One option: payload kind + present payload.
#[derive(Debug)]
pub(crate) struct OptionValue {
    pub(crate) kind: u32,
    pub(crate) payload: Option<RuntimeValue>,
}

/// Interpret an i64 value per its declared kind (the wasm rows carry scalar
/// values widened to i64; wasm has no pointer carriers).
pub(crate) fn value_from_i64(kind: u32, value: i64) -> Option<RuntimeValue> {
    Some(match kind {
        VALUE_KIND_I1 => RuntimeValue::I1(value != 0),
        VALUE_KIND_I8 | VALUE_KIND_I16 | VALUE_KIND_I32 | VALUE_KIND_U8 | VALUE_KIND_U16
        | VALUE_KIND_U32 => RuntimeValue::I32(value as i32),
        VALUE_KIND_I64 | VALUE_KIND_U64 => RuntimeValue::I64(value),
        VALUE_KIND_F32 | VALUE_KIND_F64 => RuntimeValue::F64(f64::from_bits(value as u64)),
        VALUE_KIND_TEXT | VALUE_KIND_ASCII | VALUE_KIND_PTR | VALUE_KIND_VALOR => {
            RuntimeValue::Handle(value as i32)
        }
        _ => return None,
    })
}

/// Encode a runtime value back onto the i64 value carrier.
pub(crate) fn value_to_i64(value: RuntimeValue) -> i64 {
    match value {
        RuntimeValue::I1(value) => i64::from(value),
        RuntimeValue::I32(value) => i64::from(value),
        RuntimeValue::I64(value) => value,
        RuntimeValue::F64(value) => value.to_bits() as i64,
        RuntimeValue::Handle(handle) => i64::from(handle),
    }
}

/// The values are equal under the element kind (identity for handles, scalar
/// equality otherwise).
pub(crate) fn runtime_value_eq(left: RuntimeValue, right: RuntimeValue) -> bool {
    match (left, right) {
        (RuntimeValue::I1(a), RuntimeValue::I1(b)) => a == b,
        (RuntimeValue::I32(a), RuntimeValue::I32(b)) => a == b,
        (RuntimeValue::I64(a), RuntimeValue::I64(b)) => a == b,
        (RuntimeValue::F64(a), RuntimeValue::F64(b)) => a == b,
        (RuntimeValue::Handle(a), RuntimeValue::Handle(b)) => a == b,
        // Mixed numeric comparison is not part of the v1 value surface.
        _ => false,
    }
}

/// Rust-oracle float Debug shape: integral floats keep the `.0` marker.
pub(crate) fn display_fractus(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.1}")
    } else {
        value.to_string()
    }
}

// ---------------------------------------------------------------------------
// W14 tensor shape arithmetic (mirrors the radix-runtime-contract tensor
// shape math — row-major flat offsets and element counts).
// ---------------------------------------------------------------------------

/// Total element count of a shape, or `None` on overflow/negative dims.
pub(crate) fn tensor_shape_element_count(shape: &[i64]) -> Option<usize> {
    shape.iter().try_fold(1_usize, |acc, dim| {
        let dim = usize::try_from(*dim).ok()?;
        acc.checked_mul(dim)
    })
}

/// Row-major flat offset for `index` into a tensor of `shape`, or `None` when
/// the ranks disagree, a dimension/index is negative, or an index is out of
/// bounds (mirrors `radix-runtime-contract::tensor::tensor_flat_offset`).
pub(crate) fn tensor_flat_offset(shape: &[i64], index: &[i64]) -> Option<usize> {
    if shape.len() != index.len() {
        return None;
    }
    let mut offset = 0_usize;
    let mut stride = 1_usize;
    for (dim, idx) in shape.iter().zip(index.iter()).rev() {
        let dim = usize::try_from(*dim).ok()?;
        let idx = usize::try_from(*idx).ok()?;
        if idx >= dim {
            return None;
        }
        offset = offset.checked_add(idx.checked_mul(stride)?)?;
        stride = stride.checked_mul(dim)?;
    }
    Some(offset)
}

// ---------------------------------------------------------------------------
// U6-B cursor-stream yield channel
// ---------------------------------------------------------------------------

/// One in-flight cursor-stream materialization (U6-B, P5 host ABI). The host
/// pushes a yield buffer when it starts materializing a generator (invoking
/// the `__faber_rt_v1_cursor_stream` row), the bound cede (yield) imports
/// append to the active buffer, and popping the buffer yields the materialized
/// `lista<T>`. Reference semantics: the MIR stepper's `eval_cursor_stream`
/// (run the generator to completion, collect its `cede` yields in program
/// order, discard the generator's own return value).
#[derive(Debug, Default)]
pub(crate) struct CursorYieldBuffer {
    /// The recorded yields in program order.
    pub(crate) values: Vec<RuntimeValue>,
    /// The element kind once the first yield fixes it. An empty
    /// materialization has no observable element kind (an empty `lista<T>`
    /// renders `[]` and exposes no values under any kind), so `None` until the
    /// first yield.
    pub(crate) kind: Option<u32>,
}

impl CursorYieldBuffer {
    /// Record one `cede` yield. The first yield fixes the element kind; a
    /// later yield of a different kind fails closed (the v1 collection model
    /// keeps one kind per `lista<T>`, so a heterogeneous yield sequence is a
    /// typed runtime failure — never a silent coercion).
    pub(crate) fn push(&mut self, kind: u32, value: RuntimeValue) -> Result<(), String> {
        if let Some(existing) = self.kind {
            if existing != kind {
                return Err(format!(
                    "cursor-stream yield kind mismatch: kind {existing} then kind {kind}"
                ));
            }
        } else {
            self.kind = Some(kind);
        }
        self.values.push(value);
        Ok(())
    }
}
