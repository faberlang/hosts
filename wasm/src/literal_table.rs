//! W12 literal-table initialization for the portable product host.
//!
//! The radix Wasm emitter declares a literal table in module linear memory:
//! a sequence of `(i32 kind, i32 offset, i32 length)` rows at a deterministic
//! table base, published through two exported globals
//! (`__faber_rt_v1_literal_table_ptr` / `__faber_rt_v1_literal_table_count`).
//!
//! Row kinds: `0` = text/ascii UTF-8 payload, `1` = octeti raw byte payload,
//! `2` = regex pattern (UTF-8), `3` = regex flags (UTF-8). A regex literal
//! with flags occupies its pattern row (kind `2`) followed immediately by its
//! flags row (kind `3`); the program references the pattern row index.
//!
//! Generated initialization (the host lifecycle step between instantiation
//! and entry invocation) reads the declared table and interns each row's
//! payload into the typed arena for its kind (text arena / octeti arena /
//! regex arena). Raw row indices are exactly the arena handles the program
//! references; the flags row of a regex literal keeps a continuation row so
//! later rows keep their raw indices. The product runner never receives an
//! external handle table, never parses WAT, and never scrapes an interner map
//! to reconstruct literals (W11/W12).

use crate::imports::HostState;
use crate::outcome::RunOutcome;
use wasmtime::{Instance, Memory, Store, Val};

/// Exported globals of the declared W11/W12 literal-table contract (must
/// match the `radix-mir-wasm` emitter's `literal_table.rs`).
pub(crate) const LITERAL_TABLE_PTR_GLOBAL: &str = "__faber_rt_v1_literal_table_ptr";
pub(crate) const LITERAL_TABLE_COUNT_GLOBAL: &str = "__faber_rt_v1_literal_table_count";

/// Declared row kinds (must match the emitter's `literal_table.rs`).
pub(crate) const ROW_KIND_TEXT: u32 = 0;
pub(crate) const ROW_KIND_OCTETI: u32 = 1;
pub(crate) const ROW_KIND_REGEX_PATTERN: u32 = 2;
pub(crate) const ROW_KIND_REGEX_FLAGS: u32 = 3;

/// Bytes per table row: `(kind: u32le, offset: u32le, length: u32le)`.
const ROW_BYTES: usize = 12;

/// One raw declared table row: kind plus payload bytes.
#[derive(Debug, Clone)]
pub(crate) struct RawRow {
    pub(crate) kind: u32,
    pub(crate) payload: Vec<u8>,
}

/// Run W12 generated initialization: read the declared literal table from
/// linear memory and intern each literal into the typed arena for its kind.
///
/// A module that declares neither table global has no literal table (no-op).
/// A module that declares one without the other, or whose table/payloads
/// extend past linear memory, or whose rows declare an unknown kind, fails
/// initialization with a typed [`RunOutcome::InitializationFailed`] — entry
/// never runs. (UTF-8 validation of text/regex payloads happens during
/// interning, still before entry.)
pub(crate) fn initialize_literal_table(
    instance: &Instance,
    store: &mut Store<HostState>,
) -> Result<(), RunOutcome> {
    let ptr_global = instance.get_global(&mut *store, LITERAL_TABLE_PTR_GLOBAL);
    let count_global = instance.get_global(&mut *store, LITERAL_TABLE_COUNT_GLOBAL);
    if ptr_global.is_none() && count_global.is_none() {
        return Ok(());
    }
    let ptr = global_i32(ptr_global.as_ref(), LITERAL_TABLE_PTR_GLOBAL, store)?;
    let count = global_i32(count_global.as_ref(), LITERAL_TABLE_COUNT_GLOBAL, store)?;
    if ptr < 0 || count < 0 {
        return Err(init_failed(
            "literal table pointer/count must be non-negative",
        ));
    }
    let memory = instance
        .get_memory(&mut *store, "memory")
        .ok_or_else(|| init_failed("literal table declared but the module exports no `memory`"))?;
    let rows = read_literal_rows(&memory, store, ptr, count)?;
    store.data_mut().intern_literal_table(&rows)?;
    Ok(())
}

fn global_i32(
    global: Option<&wasmtime::Global>,
    name: &str,
    store: &mut Store<HostState>,
) -> Result<i32, RunOutcome> {
    let Some(global) = global else {
        return Err(init_failed(format!(
            "literal table declares only one of `{LITERAL_TABLE_PTR_GLOBAL}` / \
             `{LITERAL_TABLE_COUNT_GLOBAL}` (missing `{name}`)"
        )));
    };
    match global.get(store) {
        Val::I32(value) => Ok(value),
        other => Err(init_failed(format!(
            "`{name}` is not an i32 global: {other:?}"
        ))),
    }
}

fn read_literal_rows(
    memory: &Memory,
    store: &Store<HostState>,
    ptr: i32,
    count: i32,
) -> Result<Vec<RawRow>, RunOutcome> {
    let data = memory.data(store);
    let ptr = usize::try_from(ptr).expect("non-negative pointer");
    let count = usize::try_from(count).expect("non-negative count");
    let table_bytes = count
        .checked_mul(ROW_BYTES)
        .ok_or_else(|| init_failed("literal table byte size overflows host address space"))?;
    let table_end = ptr
        .checked_add(table_bytes)
        .ok_or_else(|| init_failed("literal table extends past host address space"))?;
    if table_end > data.len() {
        return Err(init_failed(
            "declared literal table extends past linear memory",
        ));
    }
    let mut rows = Vec::with_capacity(count);
    for index in 0..count {
        let row_ptr = ptr + index * ROW_BYTES;
        let kind = read_u32_le(data, row_ptr)?;
        let payload = read_payload(data, row_ptr + 4)?.to_vec();
        rows.push(RawRow { kind, payload });
    }
    Ok(rows)
}

fn read_payload<'a>(data: &'a [u8], offset: usize) -> Result<&'a [u8], RunOutcome> {
    let start = usize::try_from(read_i32_le(data, offset)?).expect("non-negative row offset");
    let length = usize::try_from(read_i32_le(data, offset + 4)?).expect("non-negative row length");
    let end = start
        .checked_add(length)
        .ok_or_else(|| init_failed("literal payload extends past host address space"))?;
    if end > data.len() {
        return Err(init_failed(
            "declared literal payload extends past linear memory",
        ));
    }
    Ok(&data[start..end])
}

fn read_i32_le(data: &[u8], offset: usize) -> Result<i32, RunOutcome> {
    let bytes = data
        .get(offset..offset + 4)
        .ok_or_else(|| init_failed("literal table row extends past linear memory"))?;
    Ok(i32::from_le_bytes(
        bytes.try_into().expect("four-byte table row slice"),
    ))
}

fn read_u32_le(data: &[u8], offset: usize) -> Result<u32, RunOutcome> {
    let bytes = data
        .get(offset..offset + 4)
        .ok_or_else(|| init_failed("literal table row extends past linear memory"))?;
    Ok(u32::from_le_bytes(
        bytes.try_into().expect("four-byte table row slice"),
    ))
}

fn init_failed(message: impl Into<String>) -> RunOutcome {
    RunOutcome::InitializationFailed {
        message: format!("literal-table initialization failed: {}", message.into()),
    }
}
