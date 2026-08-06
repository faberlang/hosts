//! Closed `faber_rt_v1` import registry for the portable product host.
//!
//! Only module `faber_rt_v1` is admitted. Every admitted field is bound with
//! its canonical signature; legacy modules and unbound fields reject during
//! preflight with a typed [`RunOutcome::ImportRejected`]. A known admitted
//! symbol whose behavior is not implemented in this stage produces a typed
//! runtime failure when invoked — never a plausible default (architecture.md:
//! "must not return a plausible default").

use crate::outcome::RunOutcome;
use std::collections::HashMap;
use wasmtime::{Linker, Module};

/// Import module for the closed CPU host ABI v1 surface.
pub const WASM_IMPORT_MODULE_V1: &str = "faber_rt_v1";

/// Admitted v1 field surface for the Stage 2 product host: the scalar and
/// pointer-carrier diagnostic family the predecessor v1 host proved. The
/// registry grows by semantic family in later stages (architecture.md import
/// registry; W4B adds the solum/consolum text surface, W12 the text/format
/// family and the regex conversion rows).
pub(crate) const V1_DIAGNOSTIC_FIELDS: &[&str] = &[
    "__faber_rt_v1_diagnostic_nota_ptr",
    "__faber_rt_v1_diagnostic_mone_ptr",
    "__faber_rt_v1_diagnostic_vide_ptr",
    "__faber_rt_v1_diagnostic_nota_i64",
    "__faber_rt_v1_diagnostic_mone_i64",
    "__faber_rt_v1_diagnostic_vide_i64",
    "__faber_rt_v1_diagnostic_nota_i32",
    "__faber_rt_v1_diagnostic_nota_i8",
    "__faber_rt_v1_diagnostic_nota_i1",
    "__faber_rt_v1_diagnostic_nota_f64",
    "__faber_rt_v1_diagnostic_nota_f32",
];

/// W4B provider text surface: the closed-set v1 rows the radix Wasm emitter
/// now emits for `solum` (lege/leget/carpe/carpiet/ha uri/hauriet read verbs
/// -> `read_text`/`read_lines`/`read_bytes` by result carrier, scribe/scribet
/// write -> `write_text`) and `consolum` (scribe/scribet close-overlap onto
/// `__faber_rt_v1_diagnostic_nota_text`, mone onto
/// `__faber_rt_v1_diagnostic_mone_text`). Every row carries text-handle i32
/// operands; the product host never accepts an externally reconstructed
/// handle table. W11 literal initialization made consolum `scribe` (nota)
/// renderable through the host text arena; solum reads/writes remain typed
/// unsupported for the W15 deny-by-default filesystem reason and mone streams
/// to the host-captured stderr (W12).
pub(crate) const V1_PROVIDER_TEXT_FIELDS: &[&str] = &[
    "__faber_rt_v1_solum_read_text",
    "__faber_rt_v1_solum_read_lines",
    "__faber_rt_v1_solum_read_bytes",
    "__faber_rt_v1_solum_write_text",
    "__faber_rt_v1_diagnostic_nota_text",
    "__faber_rt_v1_diagnostic_mone_text",
];

/// WE6 json surface: the closed-set v1 rows the radix Wasm emitter now emits
/// for `json` (pange -> `__faber_rt_v1_json_pange`, solve ->
/// `__faber_rt_v1_json_solve`, tempta -> `__faber_rt_v1_json_tempta`). The
/// emitter rows carry `(param i32) (result i32)` handle carriers (pange: a
/// Json/Valor value handle in, a text handle out; solve/tempta: a text wire
/// handle in, a json value handle out). No host json implementation exists in
/// this stage (the W13 json host impl is a later stage), so invoking one is a
/// typed unsupported outcome — never a silent no-op or a plausible default
/// (W4B pattern).
pub(crate) const V1_JSON_FIELDS: &[&str] = &[
    "__faber_rt_v1_json_pange",
    "__faber_rt_v1_json_solve",
    "__faber_rt_v1_json_tempta",
];

/// W12 text-format surface: the fixed v1 format-arity rows the radix Wasm
/// emitter emits for `§`-template application (`"nomen: §"(nomen)` etc.).
/// Every row carries the template as a leading text-handle operand and
/// returns a new text handle; scalar args use their native carriers and the
/// bivalens carrier is the wasm i32 (formatted `verum`/`falsum`).
pub(crate) const V1_FORMAT_FIELDS: &[&str] = &[
    "__faber_rt_v1_format_i1",
    "__faber_rt_v1_format_i64",
    "__faber_rt_v1_format_i64_i64",
    "__faber_rt_v1_format_i64_i64_i64",
    "__faber_rt_v1_format_f64",
    "__faber_rt_v1_format_text",
    "__faber_rt_v1_format_text_text",
    "__faber_rt_v1_format_text_i64",
    "__faber_rt_v1_format_i64_text",
    "__faber_rt_v1_format_text_text_text",
    "__faber_rt_v1_format_text_i64_i1",
    "__faber_rt_v1_format_1_ptr_to_ptr",
];

/// W12 text arena surface: concat/equality and the first-order Unicode query
/// and transformation rows the radix Wasm emitter emits (text collection ops
/// and the `text_concat`/`text_eq`/`text_ne` rows). `text_split` is bound
/// with a typed unsupported outcome (no lista arena in this stage); every
/// other row is implemented against the host text arena.
pub(crate) const V1_TEXT_FIELDS: &[&str] = &[
    "__faber_rt_v1_text_concat",
    "__faber_rt_v1_text_eq",
    "__faber_rt_v1_text_ne",
    "__faber_rt_v1_text_length",
    "__faber_rt_v1_text_is_empty",
    "__faber_rt_v1_text_contains",
    "__faber_rt_v1_text_starts_with",
    "__faber_rt_v1_text_ends_with",
    "__faber_rt_v1_text_uppercase",
    "__faber_rt_v1_text_lowercase",
    "__faber_rt_v1_text_trim",
    "__faber_rt_v1_text_slice",
    "__faber_rt_v1_text_split",
    "__faber_rt_v1_text_replace",
];

/// W12 regex conversion surface: the closed-set v1 rows the radix Wasm
/// emitter emits for `textus ↦ regex` / `ascii ↦ regex` (constant-folding
/// emits regex literals through the literal table instead). Both rows carry
/// one text handle in and one regex aggregate handle out.
pub(crate) const V1_REGEX_FIELDS: &[&str] = &[
    "__faber_rt_v1_regex_from_text",
    "__faber_rt_v1_regex_from_ascii",
];

/// True when `field` is admitted by the closed v1 registry.
fn is_admitted_field(field: &str) -> bool {
    V1_DIAGNOSTIC_FIELDS.contains(&field)
        || V1_PROVIDER_TEXT_FIELDS.contains(&field)
        || V1_JSON_FIELDS.contains(&field)
        || V1_FORMAT_FIELDS.contains(&field)
        || V1_TEXT_FIELDS.contains(&field)
        || V1_REGEX_FIELDS.contains(&field)
}

/// Kind of one interned literal-table row (W12).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InternedRowKind {
    Text,
    Octeti,
    Regex,
}

/// One interned literal-table row: its kind and its arena entry.
#[derive(Debug)]
pub(crate) struct InternedRow {
    pub(crate) kind: InternedRowKind,
    /// Arena index into the kind's arena (text_arena / octeti_arena /
    /// regex_arena).
    pub(crate) arena: u32,
}

/// A regex value held by the host regex arena.
#[derive(Debug, Clone)]
pub(crate) struct RegexValue {
    pub(crate) pattern: String,
    /// Declared flags text (emitter kind-3 row). Retained for the value's
    /// identity; this stage renders regex values as their pattern only
    /// (matching the shared oracle).
    #[allow(dead_code)]
    pub(crate) flags: Option<String>,
}

/// A host-allocated dynamic value (format results, conversion results). The
/// handle space starts after the last declared literal-table row, so dynamic
/// handles never collide with row indices.
#[derive(Debug)]
enum DynamicValue {
    Text(String),
    Regex(RegexValue),
}

/// Per-run host state: captured stdout/stderr, capture bound, the typed
/// unsupported-symbol record for admitted-but-unfinished behavior, the W12
/// typed arenas of interned literals, and the dynamic-handle space.
#[derive(Debug)]
pub(crate) struct HostState {
    pub(crate) stdout: String,
    pub(crate) stderr: String,
    pub(crate) max_stdout_bytes: usize,
    pub(crate) unsupported: Option<String>,
    /// Interned literal-table rows by row index (the program's literal
    /// operands are row indices).
    rows: Vec<InternedRow>,
    /// Text payloads: interned text rows plus dynamic texts (format results).
    text_arena: Vec<String>,
    /// Octeti byte payloads: interned octeti rows plus dynamic octetis.
    octeti_arena: Vec<Vec<u8>>,
    /// Regex values: interned regex rows plus dynamic regexes.
    regex_arena: Vec<RegexValue>,
    /// Dynamic-handle allocator: starts after the last declared row.
    next_dynamic: i32,
    /// Host-allocated dynamic values by handle.
    dynamic: HashMap<i32, DynamicValue>,
}

impl HostState {
    pub(crate) fn new(max_stdout_bytes: usize) -> Self {
        Self {
            stdout: String::new(),
            stderr: String::new(),
            max_stdout_bytes,
            unsupported: None,
            rows: Vec::new(),
            text_arena: Vec::new(),
            octeti_arena: Vec::new(),
            regex_arena: Vec::new(),
            next_dynamic: 0,
            dynamic: HashMap::new(),
        }
    }

    /// Intern the declared literal-table rows into the typed arena for each
    /// row's kind and record the per-row arena handles the program
    /// references. Raw row indices are the handles; a regex literal's flags
    /// row keeps a continuation row so later rows keep their raw indices.
    /// Dynamic handles then start after the last raw row.
    ///
    /// Text rows dedup by content (identical payloads share one arena entry)
    /// while each table row keeps its own handle slot. The emitter already
    /// deduplicates rows, so row handles normally equal their row index; the
    /// indirection keeps the contract honest if a future emitter emits
    /// duplicate payloads. Octeti and regex rows intern one arena entry per
    /// declared row. Text and regex payloads must be valid UTF-8 (typed
    /// initialization failure otherwise — entry never runs).
    pub(crate) fn intern_literal_table(
        &mut self,
        rows: &[crate::literal_table::RawRow],
    ) -> Result<(), RunOutcome> {
        let mut content_to_handle = HashMap::<String, i32>::default();
        let mut index = 0usize;
        while index < rows.len() {
            let raw = &rows[index];
            match raw.kind {
                crate::literal_table::ROW_KIND_TEXT => {
                    let payload = std::str::from_utf8(&raw.payload)
                        .map_err(|_| {
                            RunOutcome::InitializationFailed {
                                message: format!(
                                    "literal-table initialization failed: text row {index} \
                                     payload is not valid UTF-8"
                                ),
                            }
                        })?
                        .to_owned();
                    let handle = if let Some(handle) = content_to_handle.get(&payload) {
                        *handle
                    } else {
                        let handle =
                            i32::try_from(self.text_arena.len()).expect("text arena handle fits i32");
                        self.text_arena.push(payload.clone());
                        content_to_handle.insert(payload, handle);
                        handle
                    };
                    self.rows.push(InternedRow {
                        kind: InternedRowKind::Text,
                        arena: u32::try_from(handle).expect("arena handle fits u32"),
                    });
                    index += 1;
                }
                crate::literal_table::ROW_KIND_OCTETI => {
                    let handle = i32::try_from(self.octeti_arena.len())
                        .expect("octeti arena handle fits i32");
                    self.octeti_arena.push(raw.payload.clone());
                    self.rows.push(InternedRow {
                        kind: InternedRowKind::Octeti,
                        arena: u32::try_from(handle).expect("arena handle fits u32"),
                    });
                    index += 1;
                }
                crate::literal_table::ROW_KIND_REGEX_PATTERN => {
                    let pattern = std::str::from_utf8(&raw.payload)
                        .map_err(|_| RunOutcome::InitializationFailed {
                            message: format!(
                                "literal-table initialization failed: regex row {index} pattern \
                                 is not valid UTF-8"
                            ),
                        })?
                        .to_owned();
                    let flags = if index + 1 < rows.len()
                        && rows[index + 1].kind == crate::literal_table::ROW_KIND_REGEX_FLAGS
                    {
                        let flags_payload = &rows[index + 1].payload;
                        Some(
                            std::str::from_utf8(flags_payload)
                                .map_err(|_| RunOutcome::InitializationFailed {
                                    message: format!(
                                        "literal-table initialization failed: regex flags row \
                                         {} is not valid UTF-8",
                                        index + 1
                                    ),
                                })?
                                .to_owned(),
                        )
                    } else {
                        None
                    };
                    let handle = i32::try_from(self.regex_arena.len())
                        .expect("regex arena handle fits i32");
                    self.regex_arena.push(RegexValue {
                        pattern: pattern.clone(),
                        flags: flags.clone(),
                    });
                    self.rows.push(InternedRow {
                        kind: InternedRowKind::Regex,
                        arena: u32::try_from(handle).expect("arena handle fits u32"),
                    });
                    index += 1;
                    if flags.is_some() {
                        // The flags row occupies a raw row index but has no
                        // program reference; a continuation row keeps later
                        // rows at their raw indices.
                        self.rows.push(InternedRow {
                            kind: InternedRowKind::Regex,
                            arena: u32::try_from(handle).expect("arena handle fits u32"),
                        });
                        index += 1;
                    }
                }
                other => {
                    return Err(RunOutcome::InitializationFailed {
                        message: format!(
                            "literal-table initialization failed: row {index} declares unknown \
                             kind {other}"
                        ),
                    });
                }
            }
        }
        self.next_dynamic =
            i32::try_from(self.rows.len()).expect("literal table row count fits i32");
        Ok(())
    }

    /// Allocate one dynamic value and return its handle (never collides with
    /// declared row indices).
    fn alloc_dynamic(&mut self, value: DynamicValue) -> i32 {
        let handle = self.next_dynamic;
        self.next_dynamic += 1;
        self.dynamic.insert(handle, value);
        handle
    }

    /// Allocate one dynamic text (format/conversion result) and return its
    /// handle.
    pub(crate) fn alloc_text(&mut self, text: String) -> i32 {
        self.alloc_dynamic(DynamicValue::Text(text))
    }

    /// Allocate one dynamic regex (conversion result) and return its handle.
    pub(crate) fn alloc_regex(&mut self, pattern: String, flags: Option<String>) -> i32 {
        self.alloc_dynamic(DynamicValue::Regex(RegexValue { pattern, flags }))
    }

    /// Resolve a text handle: an interned text row through the row map, else
    /// a dynamic text handle.
    pub(crate) fn resolve_text(&self, handle: i32) -> Option<&str> {
        let index = usize::try_from(handle).ok()?;
        if index < self.rows.len() {
            let row = &self.rows[index];
            if !matches!(row.kind, InternedRowKind::Text) {
                return None;
            }
            let arena = usize::try_from(row.arena).ok()?;
            return self.text_arena.get(arena).map(String::as_str);
        }
        if let Some(DynamicValue::Text(text)) = self.dynamic.get(&handle) {
            return Some(text.as_str());
        }
        None
    }

    /// Resolve an octeti handle: an interned octeti row.
    pub(crate) fn resolve_octeti(&self, handle: i32) -> Option<&[u8]> {
        let index = usize::try_from(handle).ok()?;
        if index < self.rows.len() {
            let row = &self.rows[index];
            if !matches!(row.kind, InternedRowKind::Octeti) {
                return None;
            }
            let arena = usize::try_from(row.arena).ok()?;
            return self.octeti_arena.get(arena).map(Vec::as_slice);
        }
        None
    }

    /// Resolve a regex handle: an interned regex row, else a dynamic regex.
    pub(crate) fn resolve_regex(&self, handle: i32) -> Option<&RegexValue> {
        let index = usize::try_from(handle).ok()?;
        if index < self.rows.len() {
            let row = &self.rows[index];
            if !matches!(row.kind, InternedRowKind::Regex) {
                return None;
            }
            let arena = usize::try_from(row.arena).ok()?;
            return self.regex_arena.get(arena);
        }
        if let Some(DynamicValue::Regex(regex)) = self.dynamic.get(&handle) {
            return Some(regex);
        }
        None
    }

    /// Resolve a diagnostic handle to its rendered line: text rows and
    /// dynamic texts render as-is, regex handles render their pattern, and
    /// octeti handles render the byte-list Debug shape (mirroring the LLVM
    /// host's opaque display).
    pub(crate) fn resolve_diagnostic(&self, handle: i32) -> Option<String> {
        if let Some(text) = self.resolve_text(handle) {
            return Some(text.to_owned());
        }
        if let Some(regex) = self.resolve_regex(handle) {
            return Some(regex.pattern.clone());
        }
        if let Some(bytes) = self.resolve_octeti(handle) {
            return Some(format!("{bytes:?}"));
        }
        None
    }

    /// Append one diagnostic line (terminated by `\n`) to stdout, bounded by
    /// the run configuration's stdout cap.
    pub(crate) fn write_line(&mut self, text: &str) {
        write_capped(&mut self.stdout, text, self.max_stdout_bytes);
    }

    /// Append one diagnostic line (terminated by `\n`) to the captured
    /// stderr, bounded by the same run configuration cap (W12 stderr capture).
    pub(crate) fn write_stderr_line(&mut self, text: &str) {
        write_capped(&mut self.stderr, text, self.max_stdout_bytes);
    }
}

fn write_capped(out: &mut String, text: &str, max_bytes: usize) {
    let mut line = String::with_capacity(text.len() + 1);
    line.push_str(text);
    line.push('\n');
    let remaining = max_bytes.saturating_sub(out.len());
    if remaining == 0 {
        return;
    }
    let take = line.len().min(remaining);
    out.push_str(&line[..take]);
}

/// Preflight the module's import surface. Every import must live on module
/// `faber_rt_v1` and use an admitted field; otherwise the run rejects before
/// linking with a typed [`RunOutcome::ImportRejected`].
pub(crate) fn preflight_imports(module: &Module) -> Result<(), RunOutcome> {
    for import in module.imports() {
        let module_name = import.module();
        let field = import.name();
        if module_name != WASM_IMPORT_MODULE_V1 {
            return Err(RunOutcome::ImportRejected {
                module: module_name.to_owned(),
                field: field.to_owned(),
                message: format!(
                    "product host accepts only `{WASM_IMPORT_MODULE_V1}` imports; \
                     module `{module_name}` is not the closed v1 surface"
                ),
            });
        }
        if !is_admitted_field(field) {
            return Err(RunOutcome::ImportRejected {
                module: module_name.to_owned(),
                field: field.to_owned(),
                message: format!(
                    "unknown `{WASM_IMPORT_MODULE_V1}` field `{field}` \
                     (not admitted by the v1 product registry)"
                ),
            });
        }
    }
    Ok(())
}

/// Bind every admitted v1 import on the linker. A declared signature that
/// conflicts with the admitted binding fails at bind/instantiate time and
/// surfaces as [`RunOutcome::LinkFailed`].
pub(crate) fn link_v1_imports(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    bind_scalar_i64(linker, "__faber_rt_v1_diagnostic_nota_i64")?;
    bind_scalar_i64(linker, "__faber_rt_v1_diagnostic_mone_i64")?;
    bind_scalar_i64(linker, "__faber_rt_v1_diagnostic_vide_i64")?;
    bind_scalar_i32(linker, "__faber_rt_v1_diagnostic_nota_i32")?;
    bind_scalar_i32(linker, "__faber_rt_v1_diagnostic_nota_i8")?;
    bind_scalar_i32(linker, "__faber_rt_v1_diagnostic_nota_i1")?;
    bind_scalar_f64(linker, "__faber_rt_v1_diagnostic_nota_f64")?;
    bind_scalar_f64(linker, "__faber_rt_v1_diagnostic_nota_f32")?;
    // W11/W12: `nota`/`vide` text diagnostics resolve the interned literal
    // (the operand is a literal-table row) and write it to the captured
    // stdout line — the wasm outcome unblocker. The pointer carriers also
    // render regex (pattern) and octeti (byte list) handles. `mone` streams
    // to the host-captured stderr (W12) — never a silent redirect to stdout.
    bind_stdout_diagnostic(linker, "__faber_rt_v1_diagnostic_nota_ptr", "nota/stdout")?;
    bind_stderr_diagnostic(linker, "__faber_rt_v1_diagnostic_mone_ptr")?;
    bind_stdout_diagnostic(linker, "__faber_rt_v1_diagnostic_vide_ptr", "vide/stdout")?;
    // W4B provider text surface: bound with the exact signatures the radix
    // Wasm emitter emits. Solum reads/writes stay typed unsupported (no fs
    // capability in RunConfig per W15 deny-by-default); consolum
    // scribe/nota_text close-overlap the nota stdout renderer above, and
    // mone_text streams to captured stderr (W12).
    bind_solum_read_handle(linker, "__faber_rt_v1_solum_read_text")?;
    bind_solum_read_handle(linker, "__faber_rt_v1_solum_read_lines")?;
    bind_solum_read_handle(linker, "__faber_rt_v1_solum_read_bytes")?;
    bind_solum_write_text(linker)?;
    bind_stdout_text(linker, "__faber_rt_v1_diagnostic_nota_text", "nota/stdout (scribe)")?;
    bind_stderr_text(linker, "__faber_rt_v1_diagnostic_mone_text")?;
    // WE6 json surface: bound with the exact `(param i32) (result i32)`
    // handle signatures the radix Wasm emitter emits. No json host
    // implementation exists in this stage, so invoking one is a typed
    // unsupported outcome (W13) — never a silent no-op.
    bind_json_handle(linker, "__faber_rt_v1_json_pange", "pange")?;
    bind_json_handle(linker, "__faber_rt_v1_json_solve", "solve")?;
    bind_json_handle(linker, "__faber_rt_v1_json_tempta", "tempta")?;
    // W12 text-format surface.
    bind_format_i1(linker)?;
    bind_format_i64(linker)?;
    bind_format_i64_i64(linker)?;
    bind_format_i64_i64_i64(linker)?;
    bind_format_f64(linker)?;
    bind_format_text(linker)?;
    bind_format_text_text(linker)?;
    bind_format_text_i64(linker)?;
    bind_format_i64_text(linker)?;
    bind_format_text_text_text(linker)?;
    bind_format_text_i64_i1(linker)?;
    bind_format_1_ptr_to_ptr(linker)?;
    // W12 text arena surface.
    bind_text_concat(linker)?;
    bind_text_eq_ne(linker, "__faber_rt_v1_text_eq", true)?;
    bind_text_eq_ne(linker, "__faber_rt_v1_text_ne", false)?;
    bind_text_length(linker)?;
    bind_text_predicate1(linker, "__faber_rt_v1_text_is_empty", TextPredicate1::IsEmpty)?;
    bind_text_predicate2(linker, "__faber_rt_v1_text_contains", TextPredicate2::Contains)?;
    bind_text_predicate2(linker, "__faber_rt_v1_text_starts_with", TextPredicate2::StartsWith)?;
    bind_text_predicate2(linker, "__faber_rt_v1_text_ends_with", TextPredicate2::EndsWith)?;
    bind_text_transform(linker, "__faber_rt_v1_text_uppercase", TextTransform::Uppercase)?;
    bind_text_transform(linker, "__faber_rt_v1_text_lowercase", TextTransform::Lowercase)?;
    bind_text_transform(linker, "__faber_rt_v1_text_trim", TextTransform::Trim)?;
    bind_text_slice(linker)?;
    bind_text_split(linker)?;
    bind_text_replace(linker)?;
    // W12 regex conversion rows.
    bind_regex_from_text(linker, "__faber_rt_v1_regex_from_text")?;
    bind_regex_from_text(linker, "__faber_rt_v1_regex_from_ascii")?;
    Ok(())
}

fn bind_scalar_i64(linker: &mut Linker<HostState>, field: &str) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        field,
        move |mut caller: wasmtime::Caller<'_, HostState>, value: i64| {
            caller.data_mut().write_line(&value.to_string());
            Ok(())
        },
    )?;
    Ok(())
}

fn bind_scalar_i32(linker: &mut Linker<HostState>, field: &str) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        field,
        move |mut caller: wasmtime::Caller<'_, HostState>, value: i32| {
            caller.data_mut().write_line(&value.to_string());
            Ok(())
        },
    )?;
    Ok(())
}

fn bind_scalar_f64(linker: &mut Linker<HostState>, field: &str) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        field,
        move |mut caller: wasmtime::Caller<'_, HostState>, value: f64| {
            caller.data_mut().write_line(&format_float(value));
            Ok(())
        },
    )?;
    Ok(())
}

/// Stdout-stream opaque diagnostics (`nota_ptr`/`vide_ptr`): W11/W12
/// materialize the interned literal — the operand is a literal-table row
/// resolved through the typed host arenas — into the captured stdout line.
/// Text rows render as-is, regex rows render their pattern, and octeti rows
/// render the byte-list Debug shape. An unresolvable handle is a typed
/// runtime failure; the product runner never accepts an externally
/// reconstructed handle table.
fn bind_stdout_diagnostic(
    linker: &mut Linker<HostState>,
    field: &'static str,
    stream: &'static str,
) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        field,
        move |mut caller: wasmtime::Caller<'_, HostState>, handle: i32| -> Result<(), wasmtime::Error> {
            let rendered = caller.data().resolve_diagnostic(handle);
            match rendered {
                Some(rendered) => {
                    caller.data_mut().write_line(&rendered);
                    Ok(())
                }
                None => Err(typed_unsupported(
                    &mut caller,
                    format!(
                        "`{field}` handle {handle}: unknown handle ({stream} oracle stream needs \
                         an interned literal or dynamic text/regex/octeti value; W11/W12 literal \
                         initialization); the product runner accepts no external handle table"
                    ),
                )),
            }
        },
    )?;
    Ok(())
}

/// Stdout-stream text diagnostics (`nota_text` for consolum `scribe`): W11
/// materializes the interned literal — the operand is a literal-table row
/// resolved through the host text arena — into the captured stdout line. An
/// unresolvable handle is a typed runtime failure.
fn bind_stdout_text(
    linker: &mut Linker<HostState>,
    field: &'static str,
    stream: &'static str,
) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        field,
        move |mut caller: wasmtime::Caller<'_, HostState>, handle: i32| -> Result<(), wasmtime::Error> {
            let text = caller.data().resolve_text(handle).map(str::to_owned);
            match text {
                Some(text) => {
                    caller.data_mut().write_line(&text);
                    Ok(())
                }
                None => Err(typed_unsupported(
                    &mut caller,
                    format!(
                        "`{field}` handle {handle}: unknown text handle ({stream} oracle stream \
                         needs an interned literal; W11/W12 literal initialization); the product \
                         runner accepts no external handle table"
                    ),
                )),
            }
        },
    )?;
    Ok(())
}

/// Mone opaque diagnostics (`mone_ptr`) stream to stderr, which the W12
/// product host captures into `RunOutcome::Success::stderr`. Resolves the
/// same typed arenas as the stdout diagnostics.
fn bind_stderr_diagnostic(
    linker: &mut Linker<HostState>,
    field: &'static str,
) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        field,
        move |mut caller: wasmtime::Caller<'_, HostState>, handle: i32| -> Result<(), wasmtime::Error> {
            let rendered = caller.data().resolve_diagnostic(handle);
            match rendered {
                Some(rendered) => {
                    caller.data_mut().write_stderr_line(&rendered);
                    Ok(())
                }
                None => Err(typed_unsupported(
                    &mut caller,
                    format!(
                        "`{field}` handle {handle}: unknown handle (mone/stderr oracle stream needs \
                         an interned literal or dynamic text/regex/octeti value); the product \
                         runner accepts no external handle table"
                    ),
                )),
            }
        },
    )?;
    Ok(())
}

/// Mone text (`mone_text`) streams to stderr, which the W12 product host
/// captures into `RunOutcome::Success::stderr` — never a silent redirect to
/// stdout.
fn bind_stderr_text(
    linker: &mut Linker<HostState>,
    field: &'static str,
) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        field,
        move |mut caller: wasmtime::Caller<'_, HostState>, handle: i32| -> Result<(), wasmtime::Error> {
            let text = caller.data().resolve_text(handle).map(str::to_owned);
            match text {
                Some(text) => {
                    caller.data_mut().write_stderr_line(&text);
                    Ok(())
                }
                None => Err(typed_unsupported(
                    &mut caller,
                    format!(
                        "`{field}` handle {handle}: unknown text handle (mone/stderr oracle \
                         stream needs an interned literal); the product runner accepts no \
                         external handle table"
                    ),
                )),
            }
        },
    )?;
    Ok(())
}

/// Solum read carriers (`read_text`/`read_lines`/`read_bytes`) are
/// admitted-but-unimplemented in this stage. The path argument is an opaque
/// text handle the runner cannot materialize without linear-memory literal
/// data, and resolving it would require a filesystem capability the product
/// host `RunConfig` does not carry (W15 deny-by-default). Invoking one is a
/// typed runtime failure (W13) — never a synthesized result handle.
fn bind_solum_read_handle(
    linker: &mut Linker<HostState>,
    field: &'static str,
) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        field,
        move |mut caller: wasmtime::Caller<'_, HostState>, handle: i32| -> Result<i32, wasmtime::Error> {
            Err(typed_unsupported(
                &mut caller,
                format!(
                    "`{field}` handle {handle}: solum read requires a filesystem capability \
                     (W15 deny-by-default; the product host RunConfig carries no fs adapter) and \
                     text-handle materialization (W11/W12 literal initialization); \
                     declared-but-unimplemented -> typed unsupported"
                ),
            ))
        },
    )?;
    Ok(())
}

/// Solum `write_text` (scribe/scribet) is admitted-but-unimplemented for the
/// same capability reasons as the read carriers: no fs adapter in the product
/// host `RunConfig`, and the path/content arguments are opaque text handles.
fn bind_solum_write_text(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_solum_write_text",
        move |mut caller: wasmtime::Caller<'_, HostState>, path: i32, text: i32| -> Result<(), wasmtime::Error> {
            Err(typed_unsupported(
                &mut caller,
                format!(
                    "`__faber_rt_v1_solum_write_text` path {path} text {text}: solum write requires \
                     a filesystem capability (W15 deny-by-default; the product host RunConfig carries \
                     no fs adapter) and text-handle materialization (W11/W12 literal initialization); \
                     declared-but-unimplemented -> typed unsupported"
                ),
            ))
        },
    )?;
    Ok(())
}

/// The json v1 rows (`pange`/`solve`/`tempta`) are admitted-but-unimplemented
/// in this stage: no json host implementation exists (a later stage lands the
/// W13 json host impl), and the emitted operands are opaque handles the runner
/// cannot materialize without linear-memory literal data (W11/W12 literal
/// initialization). Invoking one is a typed runtime failure (W13) — never a
/// synthesized result handle.
fn bind_json_handle(
    linker: &mut Linker<HostState>,
    field: &'static str,
    verb: &'static str,
) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        field,
        move |mut caller: wasmtime::Caller<'_, HostState>, handle: i32| -> Result<i32, wasmtime::Error> {
            Err(typed_unsupported(
                &mut caller,
                format!(
                    "`{field}` handle {handle}: json {verb} requires a json host implementation \
                     (admitted symbol, unfinished behavior; typed unsupported until the json host \
                     impl lands) and handle materialization (W11/W12 literal initialization); \
                     declared-but-unimplemented -> typed unsupported"
                ),
            ))
        },
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// W12 text-format surface
// ---------------------------------------------------------------------------

/// Record a typed unsupported outcome and return the runtime failure error.
fn typed_unsupported(
    caller: &mut wasmtime::Caller<'_, HostState>,
    message: impl Into<String>,
) -> wasmtime::Error {
    caller.data_mut().unsupported = Some(message.into());
    wasmtime::Error::msg("unsupported v1 operation (typed runtime failure)")
}

/// Render a `§`-template with formatted args (mirrors the LLVM host's
/// `render_template`): bare `§` consumes the next positional arg, `§N` the
/// numbered arg; a missing arg keeps the literal `§N` text.
pub(crate) fn render_template(template: &str, args: &[String]) -> String {
    let mut output = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();
    let mut next_arg = 0usize;
    while let Some(ch) = chars.next() {
        if ch != '§' {
            output.push(ch);
            continue;
        }
        let mut index = String::new();
        while chars.peek().is_some_and(char::is_ascii_digit) {
            if let Some(digit) = chars.next() {
                index.push(digit);
            }
        }
        let arg_index = if index.is_empty() {
            let current = next_arg;
            next_arg += 1;
            current
        } else {
            index.parse::<usize>().unwrap_or(usize::MAX)
        };
        if let Some(value) = args.get(arg_index) {
            output.push_str(value);
        } else {
            output.push('§');
            output.push_str(&index);
        }
    }
    output
}

/// Shared format tail: resolve the template text, render the args, and return
/// a new dynamic text handle.
fn format_result(
    caller: &mut wasmtime::Caller<'_, HostState>,
    template: i32,
    args: Vec<String>,
) -> Result<i32, wasmtime::Error> {
    let Some(template_text) = caller.data().resolve_text(template).map(str::to_owned) else {
        return Err(typed_unsupported(
            caller,
            format!("format template handle {template}: unknown text handle"),
        ));
    };
    Ok(caller.data_mut().alloc_text(render_template(&template_text, &args)))
}

fn bind_format_i1(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_format_i1",
        move |mut caller: wasmtime::Caller<'_, HostState>, template: i32, value: i32| -> Result<i32, wasmtime::Error> {
            format_result(&mut caller, template, vec![display_bivalens(value).to_owned()])
        },
    )?;
    Ok(())
}

fn bind_format_i64(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_format_i64",
        move |mut caller: wasmtime::Caller<'_, HostState>, template: i32, value: i64| -> Result<i32, wasmtime::Error> {
            format_result(&mut caller, template, vec![value.to_string()])
        },
    )?;
    Ok(())
}

fn bind_format_i64_i64(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_format_i64_i64",
        move |mut caller: wasmtime::Caller<'_, HostState>, template: i32, first: i64, second: i64| -> Result<i32, wasmtime::Error> {
            format_result(
                &mut caller,
                template,
                vec![first.to_string(), second.to_string()],
            )
        },
    )?;
    Ok(())
}

fn bind_format_i64_i64_i64(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_format_i64_i64_i64",
        move |mut caller: wasmtime::Caller<'_, HostState>, template: i32, first: i64, second: i64, third: i64| -> Result<i32, wasmtime::Error> {
            format_result(
                &mut caller,
                template,
                vec![first.to_string(), second.to_string(), third.to_string()],
            )
        },
    )?;
    Ok(())
}

fn bind_format_f64(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_format_f64",
        move |mut caller: wasmtime::Caller<'_, HostState>, template: i32, value: f64| -> Result<i32, wasmtime::Error> {
            // Scalar float display parity with the Rust oracle: integral
            // floats keep the `.0` decimal marker (`display_fractus`).
            format_result(&mut caller, template, vec![display_fractus(value)])
        },
    )?;
    Ok(())
}

fn bind_format_text(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_format_text",
        move |mut caller: wasmtime::Caller<'_, HostState>, template: i32, text: i32| -> Result<i32, wasmtime::Error> {
            let Some(value) = caller.data().resolve_text(text).map(str::to_owned) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("format text arg handle {text}: unknown text handle"),
                ));
            };
            format_result(&mut caller, template, vec![value])
        },
    )?;
    Ok(())
}

fn bind_format_text_text(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_format_text_text",
        move |mut caller: wasmtime::Caller<'_, HostState>, template: i32, first: i32, second: i32| -> Result<i32, wasmtime::Error> {
            let Some(first) = caller.data().resolve_text(first).map(str::to_owned) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("format text arg handle {first}: unknown text handle"),
                ));
            };
            let Some(second) = caller.data().resolve_text(second).map(str::to_owned) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("format text arg handle {second}: unknown text handle"),
                ));
            };
            format_result(&mut caller, template, vec![first, second])
        },
    )?;
    Ok(())
}

fn bind_format_text_i64(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_format_text_i64",
        move |mut caller: wasmtime::Caller<'_, HostState>, template: i32, text: i32, value: i64| -> Result<i32, wasmtime::Error> {
            let Some(text) = caller.data().resolve_text(text).map(str::to_owned) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("format text arg handle {text}: unknown text handle"),
                ));
            };
            format_result(&mut caller, template, vec![text, value.to_string()])
        },
    )?;
    Ok(())
}

fn bind_format_i64_text(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_format_i64_text",
        move |mut caller: wasmtime::Caller<'_, HostState>, template: i32, value: i64, text: i32| -> Result<i32, wasmtime::Error> {
            let Some(text) = caller.data().resolve_text(text).map(str::to_owned) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("format text arg handle {text}: unknown text handle"),
                ));
            };
            format_result(&mut caller, template, vec![value.to_string(), text])
        },
    )?;
    Ok(())
}

fn bind_format_text_text_text(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_format_text_text_text",
        move |mut caller: wasmtime::Caller<'_, HostState>, template: i32, first: i32, second: i32, third: i32| -> Result<i32, wasmtime::Error> {
            let mut args = Vec::with_capacity(3);
            for (index, handle) in [first, second, third].into_iter().enumerate() {
                let Some(value) = caller.data().resolve_text(handle).map(str::to_owned) else {
                    return Err(typed_unsupported(
                        &mut caller,
                        format!("format text arg {index} handle {handle}: unknown text handle"),
                    ));
                };
                args.push(value);
            }
            format_result(&mut caller, template, args)
        },
    )?;
    Ok(())
}

fn bind_format_text_i64_i1(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_format_text_i64_i1",
        move |mut caller: wasmtime::Caller<'_, HostState>, template: i32, text: i32, integer: i64, boolean: i32| -> Result<i32, wasmtime::Error> {
            let Some(text) = caller.data().resolve_text(text).map(str::to_owned) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("format text arg handle {text}: unknown text handle"),
                ));
            };
            format_result(
                &mut caller,
                template,
                vec![text, integer.to_string(), display_bivalens(boolean).to_owned()],
            )
        },
    )?;
    Ok(())
}

/// Render a template with one opaque aggregate handle (regex/octeti in this
/// stage; lista/tabula/copia arenas land with their families). The opaque
/// handle is displayed like the LLVM host's opaque rendering: regex renders
/// its pattern, octeti renders the byte-list Debug shape. Unrecognized
/// handles fail closed with a typed unsupported outcome.
fn bind_format_1_ptr_to_ptr(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_format_1_ptr_to_ptr",
        move |mut caller: wasmtime::Caller<'_, HostState>, template: i32, value: i32| -> Result<i32, wasmtime::Error> {
            let Some(rendered) = caller.data().resolve_diagnostic(value) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("format opaque arg handle {value}: unrecognized aggregate handle"),
                ));
            };
            format_result(&mut caller, template, vec![rendered])
        },
    )?;
    Ok(())
}

fn display_fractus(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.1}")
    } else {
        value.to_string()
    }
}

fn display_bivalens(value: i32) -> &'static str {
    if value != 0 {
        "verum"
    } else {
        "falsum"
    }
}

// ---------------------------------------------------------------------------
// W12 text arena surface
// ---------------------------------------------------------------------------

fn bind_text_concat(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_text_concat",
        move |mut caller: wasmtime::Caller<'_, HostState>, first: i32, second: i32| -> Result<i32, wasmtime::Error> {
            let first = caller.data().resolve_text(first).map(str::to_owned);
            let second = caller.data().resolve_text(second).map(str::to_owned);
            let (Some(first), Some(second)) = (first, second) else {
                return Err(typed_unsupported(
                    &mut caller,
                    "text_concat received an unknown text handle",
                ));
            };
            Ok(caller.data_mut().alloc_text(format!("{first}{second}")))
        },
    )?;
    Ok(())
}

fn bind_text_eq_ne(
    linker: &mut Linker<HostState>,
    field: &'static str,
    eq: bool,
) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        field,
        move |mut caller: wasmtime::Caller<'_, HostState>, first: i32, second: i32| -> Result<i32, wasmtime::Error> {
            let first = caller.data().resolve_text(first).map(str::to_owned);
            let second = caller.data().resolve_text(second).map(str::to_owned);
            let (Some(first), Some(second)) = (first, second) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("`{field}` received an unknown text handle"),
                ));
            };
            let equal = first == second;
            Ok(i32::from(if eq { equal } else { !equal }))
        },
    )?;
    Ok(())
}

fn bind_text_length(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_text_length",
        move |mut caller: wasmtime::Caller<'_, HostState>, text: i32| -> Result<i64, wasmtime::Error> {
            let Some(text) = caller.data().resolve_text(text) else {
                return Err(typed_unsupported(
                    &mut caller,
                    "text_length received an unknown text handle",
                ));
            };
            Ok(i64::try_from(text.chars().count()).expect("text length fits i64"))
        },
    )?;
    Ok(())
}

#[derive(Clone, Copy)]
enum TextPredicate1 {
    IsEmpty,
}

#[derive(Clone, Copy)]
enum TextPredicate2 {
    Contains,
    StartsWith,
    EndsWith,
}

fn bind_text_predicate1(
    linker: &mut Linker<HostState>,
    field: &'static str,
    predicate: TextPredicate1,
) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        field,
        move |mut caller: wasmtime::Caller<'_, HostState>, text: i32| -> Result<i32, wasmtime::Error> {
            let Some(text) = caller.data().resolve_text(text) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("`{field}` received an unknown text handle"),
                ));
            };
            Ok(i32::from(match predicate {
                TextPredicate1::IsEmpty => text.is_empty(),
            }))
        },
    )?;
    Ok(())
}

fn bind_text_predicate2(
    linker: &mut Linker<HostState>,
    field: &'static str,
    predicate: TextPredicate2,
) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        field,
        move |mut caller: wasmtime::Caller<'_, HostState>, text: i32, other: i32| -> Result<i32, wasmtime::Error> {
            let text = caller.data().resolve_text(text).map(str::to_owned);
            let other = caller.data().resolve_text(other).map(str::to_owned);
            let (Some(text), Some(other)) = (text, other) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("`{field}` received an unknown text handle"),
                ));
            };
            let matched = match predicate {
                TextPredicate2::Contains => text.contains(&other),
                TextPredicate2::StartsWith => text.starts_with(&other),
                TextPredicate2::EndsWith => text.ends_with(&other),
            };
            Ok(i32::from(matched))
        },
    )?;
    Ok(())
}

#[derive(Clone, Copy)]
enum TextTransform {
    Uppercase,
    Lowercase,
    Trim,
}

fn bind_text_transform(
    linker: &mut Linker<HostState>,
    field: &'static str,
    transform: TextTransform,
) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        field,
        move |mut caller: wasmtime::Caller<'_, HostState>, text: i32| -> Result<i32, wasmtime::Error> {
            let Some(text) = caller.data().resolve_text(text).map(str::to_owned) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("`{field}` received an unknown text handle"),
                ));
            };
            let transformed = match transform {
                TextTransform::Uppercase => text.to_uppercase(),
                TextTransform::Lowercase => text.to_lowercase(),
                TextTransform::Trim => text.trim().to_owned(),
            };
            Ok(caller.data_mut().alloc_text(transformed))
        },
    )?;
    Ok(())
}

fn bind_text_slice(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_text_slice",
        move |mut caller: wasmtime::Caller<'_, HostState>, text: i32, start: i64, end: i64| -> Result<i32, wasmtime::Error> {
            let Some(text) = caller.data().resolve_text(text).map(str::to_owned) else {
                return Err(typed_unsupported(
                    &mut caller,
                    "text_slice received an unknown text handle",
                ));
            };
            let start = usize::try_from(start).unwrap_or(usize::MAX);
            let end = usize::try_from(end).unwrap_or(usize::MAX);
            let sliced: String = text
                .chars()
                .skip(start)
                .take(end.saturating_sub(start))
                .collect();
            Ok(caller.data_mut().alloc_text(sliced))
        },
    )?;
    Ok(())
}

/// `text_split` returns a lista of textus — the lista arena is not part of
/// this stage's host, so invoking the row is a typed unsupported outcome
/// (never a synthesized aggregate handle).
fn bind_text_split(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_text_split",
        move |mut caller: wasmtime::Caller<'_, HostState>, _text: i32, _separator: i32| -> Result<i32, wasmtime::Error> {
            Err(typed_unsupported(
                &mut caller,
                "`__faber_rt_v1_text_split` requires a lista arena (not in this stage's host); \
                 declared-but-unimplemented -> typed unsupported",
            ))
        },
    )?;
    Ok(())
}

fn bind_text_replace(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_text_replace",
        move |mut caller: wasmtime::Caller<'_, HostState>, text: i32, old: i32, new: i32| -> Result<i32, wasmtime::Error> {
            let text = caller.data().resolve_text(text).map(str::to_owned);
            let old = caller.data().resolve_text(old).map(str::to_owned);
            let new = caller.data().resolve_text(new).map(str::to_owned);
            let (Some(text), Some(old), Some(new)) = (text, old, new) else {
                return Err(typed_unsupported(
                    &mut caller,
                    "text_replace received an unknown text handle",
                ));
            };
            Ok(caller.data_mut().alloc_text(text.replace(&old, &new)))
        },
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// W12 regex conversion rows
// ---------------------------------------------------------------------------

/// `regex_from_text`/`regex_from_ascii` construct a regex carrier from one
/// text handle (the `textus ↦ regex` / `ascii ↦ regex` conversio the emitter
/// does not constant-fold). The returned handle resolves through the regex
/// arena when a later op renders it.
fn bind_regex_from_text(
    linker: &mut Linker<HostState>,
    field: &'static str,
) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        field,
        move |mut caller: wasmtime::Caller<'_, HostState>, text: i32| -> Result<i32, wasmtime::Error> {
            let Some(pattern) = caller.data().resolve_text(text).map(str::to_owned) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("`{field}` received an unknown text handle"),
                ));
            };
            Ok(caller.data_mut().alloc_regex(pattern, None))
        },
    )?;
    Ok(())
}

fn format_float(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{}", value as i64)
    } else {
        value.to_string()
    }
}
