//! W11 literal-table initialization for the portable product host.
//!
//! The radix Wasm emitter declares a literal table in module linear memory:
//! a sequence of `(i32 offset, i32 length)` rows at a deterministic table
//! base, published through two exported globals
//! (`__faber_rt_v1_literal_table_ptr` / `__faber_rt_v1_literal_table_count`).
//!
//! Generated initialization (the host lifecycle step between instantiation
//! and entry invocation) reads the declared table, interns each row's payload
//! into the host text arena, and the table row indices are exactly the arena
//! handles the program references. The product runner never receives an
//! external handle table, never parses WAT, and never scrapes an interner
//! map to reconstruct literals (W11).

use crate::imports::HostState;
use crate::outcome::RunOutcome;
use wasmtime::{Instance, Memory, Store, Val};

/// Exported globals of the declared W11 literal-table contract (must match
/// the `radix-mir-wasm` emitter's `literal_table.rs`).
pub(crate) const LITERAL_TABLE_PTR_GLOBAL: &str = "__faber_rt_v1_literal_table_ptr";
pub(crate) const LITERAL_TABLE_COUNT_GLOBAL: &str = "__faber_rt_v1_literal_table_count";

/// Run W11 generated initialization: read the declared literal table from
/// linear memory and intern each literal into the host arena.
///
/// A module that declares neither table global has no literal table (no-op).
/// A module that declares one without the other, or whose table/payloads
/// extend past linear memory, or whose payloads are not valid UTF-8, fails
/// initialization with a typed [`RunOutcome::InitializationFailed`] — entry
/// never runs.
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
    let count = global_i32(
        count_global.as_ref(),
        LITERAL_TABLE_COUNT_GLOBAL,
        store,
    )?;
    if ptr < 0 || count < 0 {
        return Err(init_failed(
            "literal table pointer/count must be non-negative",
        ));
    }
    let memory = instance
        .get_memory(&mut *store, "memory")
        .ok_or_else(|| {
            init_failed("literal table declared but the module exports no `memory`")
        })?;
    let rows = read_literal_rows(&memory, store, ptr, count)?;
    store.data_mut().intern_literal_table(&rows);
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
) -> Result<Vec<String>, RunOutcome> {
    let data = memory.data(store);
    let ptr = usize::try_from(ptr).expect("non-negative pointer");
    let count = usize::try_from(count).expect("non-negative count");
    let table_bytes = count.checked_mul(8).ok_or_else(|| {
        init_failed("literal table byte size overflows host address space")
    })?;
    let table_end = ptr
        .checked_add(table_bytes)
        .ok_or_else(|| init_failed("literal table extends past host address space"))?;
    if table_end > data.len() {
        return Err(init_failed(
            "declared literal table extends past linear memory",
        ));
    }
    let mut rows = Vec::with_capacity(count);
    for row in 0..count {
        let base = ptr + row * 8;
        let offset = read_i32_le(data, base)?;
        let length = read_i32_le(data, base + 4)?;
        if offset < 0 || length < 0 {
            return Err(init_failed("literal table row offsets must be non-negative"));
        }
        let start = usize::try_from(offset).expect("non-negative row offset");
        let end = start
            .checked_add(usize::try_from(length).expect("non-negative row length"))
            .ok_or_else(|| init_failed("literal payload extends past host address space"))?;
        if end > data.len() {
            return Err(init_failed(
                "declared literal payload extends past linear memory",
            ));
        }
        let text = std::str::from_utf8(&data[start..end])
            .map_err(|_| init_failed("literal payload is not valid UTF-8"))?
            .to_owned();
        rows.push(text);
    }
    Ok(rows)
}

fn read_i32_le(data: &[u8], offset: usize) -> Result<i32, RunOutcome> {
    let bytes = data
        .get(offset..offset + 4)
        .ok_or_else(|| init_failed("literal table row extends past linear memory"))?;
    Ok(i32::from_le_bytes(
        bytes.try_into().expect("four-byte table row slice"),
    ))
}

fn init_failed(message: impl Into<String>) -> RunOutcome {
    RunOutcome::InitializationFailed {
        message: format!("literal-table initialization failed: {}", message.into()),
    }
}
