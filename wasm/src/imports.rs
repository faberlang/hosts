//! Closed `faber_rt_v1` import registry for the portable product host.
//!
//! Only module `faber_rt_v1` is admitted. Every admitted field is bound with
//! its canonical signature; legacy modules and unbound fields reject during
//! preflight with a typed [`RunOutcome::ImportRejected`]. A known admitted
//! symbol whose behavior is not implemented in this stage produces a typed
//! runtime failure when invoked — never a plausible default (architecture.md:
//! "must not return a plausible default").

use crate::collections::{
    display_fractus, runtime_value_eq, tensor_flat_offset, tensor_shape_element_count,
    value_from_i64, value_to_i64, CollectionValue, CursorYieldBuffer, MapValue, OptionValue,
    RuntimeValue, TensorValue,
};
use crate::outcome::RunOutcome;
use radix_host_abi::{
    SYMBOL_CURSOR_STREAM, VALUE_KIND_ASCII, VALUE_KIND_F32, VALUE_KIND_F64, VALUE_KIND_I1,
    VALUE_KIND_I16, VALUE_KIND_I32, VALUE_KIND_I64, VALUE_KIND_I8, VALUE_KIND_PTR, VALUE_KIND_TEXT,
    VALUE_KIND_U16, VALUE_KIND_U32, VALUE_KIND_U64, VALUE_KIND_U8, VALUE_KIND_VALOR,
};
use std::collections::{BTreeMap, HashMap, HashSet};
use wasmtime::{FuncType, Instance, Linker, Module, Store, Val, ValType};

/// Import module for the closed CPU host ABI v1 surface.
pub const WASM_IMPORT_MODULE_V1: &str = "faber_rt_v1";

/// U6-E — package-namespace import module for external identities. The radix
/// Wasm emitter lands every external identity import on this module — both
/// library identities (`norma:*`, dependency packages) and, in package-aware
/// emit mode (`emit_wasm_text_probe_package_aware`, U6-C), same-package
/// cross-module identities (`importa:auxilium:saluta`) — under the canonical
/// identity-based field name (`external_product_<product>_module_…_func_<name>`,
/// the LLVM-lane naming). The product host admits this module ONLY in package
/// run mode ([`crate::host::WasmRtV1Host::run_package`]) and resolves each
/// field against the sibling modules' canonical external-symbol exports.
/// Single-module runs keep the closed v1 preflight, so `faber_external`
/// rejects there — a package import is never a host symbol (W13 registry
/// cleanliness).
pub(crate) const FABER_EXTERNAL_IMPORT_MODULE: &str = "faber_external";

/// The emitter's canonical external-symbol prefix
/// (`radix-mir-wasm/src/import_names.rs` `FABER_EXTERNAL_IMPORT_PREFIX`): a
/// `faber_external` import with field `F` resolves against a sibling module
/// export named `__faber_{F}`. The canonical symbol never carries the
/// `__faber_rt_v1_` prefix, so the closed host-symbol registry never sees a
/// package import.
pub(crate) const FABER_EXTERNAL_IMPORT_PREFIX: &str = "__faber_";

/// Legacy import module for generator cede (yield) rows (U6-B admitted
/// exception). The radix Wasm emitter declares generator yields on this
/// module; the product host admits exactly the closed cede field grammar
/// below and binds it to the cursor-stream yield channel.
pub(crate) const LEGACY_CEDE_MODULE: &str = "faber_runtime";

/// U6-B — the cede (yield) channel fields the radix Wasm emitter declares on
/// the legacy `faber_runtime` module (one import per carrier pair,
/// `cede_1_{arg}_to_{result}` over the i32/i64/f64 wasm carriers). Recorded
/// unit decision: moving these rows onto the closed `faber_rt_v1` surface is
/// a radix-mir-wasm emitter change (out of U6-B's scope), so the yield
/// channel is bound as an **explicit admitted exception** — exactly this
/// closed set, nothing else. Any other legacy-module import still rejects
/// during preflight; the closed v1 registry for all standard rows is
/// unchanged.
pub(crate) const LEGACY_CEDE_ROWS: &[(&str, ValType, ValType)] = &[
    ("cede_1_i32_to_i32", ValType::I32, ValType::I32),
    ("cede_1_i32_to_i64", ValType::I32, ValType::I64),
    ("cede_1_i32_to_f64", ValType::I32, ValType::F64),
    ("cede_1_i64_to_i32", ValType::I64, ValType::I32),
    ("cede_1_i64_to_i64", ValType::I64, ValType::I64),
    ("cede_1_i64_to_f64", ValType::I64, ValType::F64),
    ("cede_1_f64_to_i32", ValType::F64, ValType::I32),
    ("cede_1_f64_to_i64", ValType::F64, ValType::I64),
    ("cede_1_f64_to_f64", ValType::F64, ValType::F64),
];

/// True for the closed cede (yield) field grammar (U6-B exception).
fn is_cede_yield_field(field: &str) -> bool {
    LEGACY_CEDE_ROWS.iter().any(|(name, _, _)| *name == field)
}

/// The module's callable-table export the P5 cursor-stream host resolves the
/// generator function-id reference against (the U6-A mechanism: the emitter
/// emits and exports `faber_callables` whenever a cursor stream is
/// materialized).
pub(crate) const FABER_CALLABLE_TABLE: &str = "faber_callables";

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

/// W13 collection surface: the closed-set v1 rows the radix Wasm emitter now
/// emits for the collection/scalar display family. Every value/key argument
/// crosses the normalized `i64` value carrier (wasm has no pointer carriers);
/// the host interprets each value per the collection's declared `VALUE_KIND`.
/// `array_push` returns the collection handle so literal construction can
/// chain `(local.set $t (call push ...))`.
pub(crate) const V1_COLLECTION_FIELDS: &[&str] = &[
    "__faber_rt_v1_array_new",
    "__faber_rt_v1_array_push",
    "__faber_rt_v1_array_extend",
    "__faber_rt_v1_array_length",
    "__faber_rt_v1_array_get",
    "__faber_rt_v1_array_option",
    "__faber_rt_v1_array_contains",
    "__faber_rt_v1_array_is_empty",
    "__faber_rt_v1_array_reverse",
    "__faber_rt_v1_array_sort",
    "__faber_rt_v1_array_sum",
    "__faber_rt_v1_map_new",
    "__faber_rt_v1_map_put",
    "__faber_rt_v1_map_delete",
    "__faber_rt_v1_map_keys",
    "__faber_rt_v1_map_values",
    "__faber_rt_v1_set_new",
    "__faber_rt_v1_set_from_array",
    "__faber_rt_v1_array_from_set",
    "__faber_rt_v1_set_union",
    "__faber_rt_v1_set_intersection",
    "__faber_rt_v1_set_difference",
    "__faber_rt_v1_set_symmetric_difference",
    "__faber_rt_v1_set_is_subset",
    "__faber_rt_v1_set_is_superset",
];

/// W13 option surface: the closed-set v1 rows for the null-encoded/arena
/// option model. `none`/`some` carry the payload kind; `get`/`get_or` cross
/// the payload widened to `i64`.
pub(crate) const V1_OPTION_FIELDS: &[&str] = &[
    "__faber_rt_v1_option_none",
    "__faber_rt_v1_option_some",
    "__faber_rt_v1_option_get",
    "__faber_rt_v1_option_get_or",
    "__faber_rt_v1_option_is_present",
];

/// W13 scalar conversion surface: the closed-set v1 rows for scalar/format
/// display and text↦scalar conversion the corpus fixtures route through.
pub(crate) const V1_SCALAR_FIELDS: &[&str] = &[
    "__faber_rt_v1_assert",
    "__faber_rt_v1_assert_message",
    "__faber_rt_v1_text_i64",
    "__faber_rt_v1_text_f64",
    "__faber_rt_v1_text_i1",
    "__faber_rt_v1_text_truthy",
    "__faber_rt_v1_ascii_truthy",
    "__faber_rt_v1_text_parse_integer",
    "__faber_rt_v1_text_parse_integer_or",
    "__faber_rt_v1_text_parse_float",
    "__faber_rt_v1_text_parse_float_or",
    "__faber_rt_v1_read_line_0_to_ptr",
];

/// W14 tensor surface: the closed-set v1 rows the radix Wasm emitter now
/// emits for the tensor display family. Construction (`tensor_new` vacua /
/// `tensor_create` fill+shape / `tensor_from_flat` flat+shape) produces a
/// dense tensor handle; reads (`tensor_rank`/`tensor_shape`/`tensor_get`)
/// and writes (`tensor_set`/`tensor_fill`) match the LLVM lane's tensor
/// carrier semantics. `tensor_convert` carries the source/target element
/// kinds as trailing i32 consts (one host binding per kind pair). The
/// element/value carriers are i32 handles for tensors and shapes, f64 for
/// f32/f64 element values, and i64 for the row-major index vectors.
pub(crate) const V1_TENSOR_FIELDS: &[&str] = &[
    "__faber_rt_v1_tensor_new",
    "__faber_rt_v1_tensor_create",
    "__faber_rt_v1_tensor_from_flat",
    "__faber_rt_v1_tensor_rank",
    "__faber_rt_v1_tensor_shape",
    "__faber_rt_v1_tensor_reshape",
    "__faber_rt_v1_tensor_get",
    "__faber_rt_v1_tensor_set",
    "__faber_rt_v1_tensor_fill",
    "__faber_rt_v1_tensor_flatten",
    "__faber_rt_v1_tensor_materialize",
    "__faber_rt_v1_tensor_slice",
    "__faber_rt_v1_tensor_add",
    "__faber_rt_v1_tensor_sub",
    "__faber_rt_v1_tensor_mul",
    "__faber_rt_v1_tensor_matmul",
    "__faber_rt_v1_tensor_sum",
    "__faber_rt_v1_tensor_mean",
    "__faber_rt_v1_tensor_convert",
];

/// U6-B cursor-stream surface: the closed-set v1 row the radix Wasm emitter
/// emits for `@ cursor` / `fiunt` / `fient` materialization. One fixed symbol
/// whose signature carries the generator function-id reference (i32 — the
/// generator's entry in the exported callable table) plus the generator's
/// argument carriers and returns the materialized `lista<T>` aggregate
/// handle; the host invokes the referenced generator to completion and
/// collects its `cede` yields (P5 contract).
pub(crate) const V1_CURSOR_STREAM_FIELDS: &[&str] = &["__faber_rt_v1_cursor_stream"];

/// True when `field` is admitted by the closed v1 registry.
fn is_admitted_field(field: &str) -> bool {
    V1_DIAGNOSTIC_FIELDS.contains(&field)
        || V1_PROVIDER_TEXT_FIELDS.contains(&field)
        || V1_JSON_FIELDS.contains(&field)
        || V1_FORMAT_FIELDS.contains(&field)
        || V1_TEXT_FIELDS.contains(&field)
        || V1_REGEX_FIELDS.contains(&field)
        || V1_COLLECTION_FIELDS.contains(&field)
        || V1_OPTION_FIELDS.contains(&field)
        || V1_SCALAR_FIELDS.contains(&field)
        || V1_TENSOR_FIELDS.contains(&field)
        || V1_CURSOR_STREAM_FIELDS.contains(&field)
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

/// A host-allocated dynamic value (format results, conversion results, and
/// the W13 collection/scalar display arenas). The handle space starts after
/// the last declared literal-table row, so dynamic handles never collide with
/// row indices.
#[derive(Debug)]
enum DynamicValue {
    Text(String),
    Regex(RegexValue),
    Collection {
        index: usize,
    },
    Map {
        index: usize,
    },
    Option {
        index: usize,
    },
    /// W14 — dense tensor arena entry.
    Tensor {
        index: usize,
    },
}

/// Per-run host state: captured stdout/stderr, capture bound, the typed
/// unsupported-symbol record for admitted-but-unfinished behavior, the W12
/// typed arenas of interned literals, the W13 collection/map/option arenas,
/// and the dynamic-handle space.
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
    /// W13 — collection arena (`lista`/`copia` entries).
    collections: Vec<CollectionValue>,
    /// W13 — map arena (`tabula` entries).
    maps: Vec<MapValue>,
    /// W13 — option arena (payload entries).
    options: Vec<OptionValue>,
    /// W14 — tensor arena (dense element-kind + shape + flat values).
    tensors: Vec<TensorValue>,
    /// U6-B — cursor-stream yield-buffer stack. The host pushes one buffer
    /// per active materialization; the bound cede (yield) rows append to the
    /// active buffer, and popping it yields the materialized `lista<T>`.
    pub(crate) cursor_yields: Vec<CursorYieldBuffer>,
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
            collections: Vec::new(),
            maps: Vec::new(),
            options: Vec::new(),
            tensors: Vec::new(),
            cursor_yields: Vec::new(),
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
                        .map_err(|_| RunOutcome::InitializationFailed {
                            message: format!(
                                "literal-table initialization failed: text row {index} \
                                     payload is not valid UTF-8"
                            ),
                        })?
                        .to_owned();
                    let handle = if let Some(handle) = content_to_handle.get(&payload) {
                        *handle
                    } else {
                        let handle = i32::try_from(self.text_arena.len())
                            .expect("text arena handle fits i32");
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
                    let handle =
                        i32::try_from(self.regex_arena.len()).expect("regex arena handle fits i32");
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

    /// Resolve an octeti literal row mutably (an `octeti` unifies with
    /// `lista<u8>`: `appende`/`longitudo`/`accipe` run on the interned byte
    /// payload — W13).
    pub(crate) fn resolve_octeti_mut(&mut self, handle: i32) -> Option<&mut Vec<u8>> {
        let index = usize::try_from(handle).ok()?;
        if index < self.rows.len() {
            let row = &self.rows[index];
            if !matches!(row.kind, InternedRowKind::Octeti) {
                return None;
            }
            let arena = usize::try_from(row.arena).ok()?;
            return self.octeti_arena.get_mut(arena);
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

    // -----------------------------------------------------------------------
    // W13 collection/scalar display arenas
    // -----------------------------------------------------------------------

    /// Allocate one collection (`lista`/`copia`) and return its handle.
    pub(crate) fn alloc_collection(
        &mut self,
        set: bool,
        kind: u32,
        values: Vec<RuntimeValue>,
    ) -> i32 {
        let index = self.collections.len();
        self.collections.push(CollectionValue { set, kind, values });
        self.alloc_dynamic(DynamicValue::Collection { index })
    }

    /// Allocate one map (`tabula`) and return its handle.
    pub(crate) fn alloc_map(
        &mut self,
        key_kind: u32,
        value_kind: u32,
        entries: Vec<(RuntimeValue, RuntimeValue)>,
    ) -> i32 {
        let index = self.maps.len();
        self.maps.push(MapValue {
            key_kind,
            value_kind,
            entries,
        });
        self.alloc_dynamic(DynamicValue::Map { index })
    }

    /// Allocate one option and return its handle.
    pub(crate) fn alloc_option(&mut self, kind: u32, payload: Option<RuntimeValue>) -> i32 {
        let index = self.options.len();
        self.options.push(OptionValue { kind, payload });
        self.alloc_dynamic(DynamicValue::Option { index })
    }

    /// Encode one index-read option result. i64/f64 payloads cannot fit a
    /// null-encoded i32 handle, so they wrap in the option arena (nota renders
    /// the payload, `option_get_or` unwraps it); i32-carrying payloads
    /// null-encode — the payload value IS the handle (0 = absent), matching
    /// the emitter's inline select coalesce.
    pub(crate) fn option_result(&mut self, kind: u32, payload: Option<RuntimeValue>) -> i32 {
        let scalar_kind = matches!(
            kind,
            VALUE_KIND_I64 | VALUE_KIND_U64 | VALUE_KIND_F32 | VALUE_KIND_F64
        );
        if scalar_kind {
            self.alloc_option(kind, payload)
        } else {
            match payload {
                Some(payload) => value_to_i64(payload) as i32,
                None => 0,
            }
        }
    }

    /// Resolve a collection (`lista`/`copia`) handle.
    pub(crate) fn find_collection(&self, handle: i32) -> Option<&CollectionValue> {
        match self.dynamic.get(&handle) {
            Some(DynamicValue::Collection { index }) => self.collections.get(*index),
            _ => None,
        }
    }

    /// Resolve a collection handle mutably.
    pub(crate) fn find_collection_mut(&mut self, handle: i32) -> Option<&mut CollectionValue> {
        match self.dynamic.get(&handle) {
            Some(DynamicValue::Collection { index }) => self.collections.get_mut(*index),
            _ => None,
        }
    }

    /// Resolve a map (`tabula`) handle.
    pub(crate) fn find_map(&self, handle: i32) -> Option<&MapValue> {
        match self.dynamic.get(&handle) {
            Some(DynamicValue::Map { index }) => self.maps.get(*index),
            _ => None,
        }
    }

    /// Resolve a map handle mutably.
    pub(crate) fn find_map_mut(&mut self, handle: i32) -> Option<&mut MapValue> {
        match self.dynamic.get(&handle) {
            Some(DynamicValue::Map { index }) => self.maps.get_mut(*index),
            _ => None,
        }
    }

    /// Resolve an option handle.
    pub(crate) fn find_option(&self, handle: i32) -> Option<&OptionValue> {
        match self.dynamic.get(&handle) {
            Some(DynamicValue::Option { index }) => self.options.get(*index),
            _ => None,
        }
    }

    /// W14 — allocate one dense tensor and return its handle.
    pub(crate) fn alloc_tensor(&mut self, tensor: TensorValue) -> i32 {
        let index = self.tensors.len();
        self.tensors.push(tensor);
        self.alloc_dynamic(DynamicValue::Tensor { index })
    }

    /// W14 — resolve a tensor handle.
    pub(crate) fn find_tensor(&self, handle: i32) -> Option<&TensorValue> {
        match self.dynamic.get(&handle) {
            Some(DynamicValue::Tensor { index }) => self.tensors.get(*index),
            _ => None,
        }
    }

    /// W14 — resolve a tensor handle mutably.
    pub(crate) fn find_tensor_mut(&mut self, handle: i32) -> Option<&mut TensorValue> {
        match self.dynamic.get(&handle) {
            Some(DynamicValue::Tensor { index }) => self.tensors.get_mut(*index),
            _ => None,
        }
    }

    /// Render one collection element in the Rust-oracle Debug shape. Text
    /// elements quote (`"prima"`, matching `Vec<String>` Debug), bivalens
    /// elements render `true`/`false`, and nested aggregate handles resolve
    /// recursively.
    fn render_element(&self, kind: u32, value: RuntimeValue) -> Option<String> {
        Some(match (kind, value) {
            (VALUE_KIND_I1, RuntimeValue::I1(value)) => {
                format!("{value}")
            }
            (_, RuntimeValue::I1(value)) => format!("{value}"),
            (_, RuntimeValue::I32(value)) => format!("{value}"),
            (_, RuntimeValue::I64(value)) => format!("{value}"),
            (_, RuntimeValue::F64(value)) => display_fractus(value),
            (VALUE_KIND_TEXT | VALUE_KIND_ASCII, RuntimeValue::Handle(handle)) => {
                format!("{:?}", self.resolve_text(handle)?)
            }
            (VALUE_KIND_PTR | VALUE_KIND_VALOR, RuntimeValue::Handle(handle)) => {
                self.render_handle_display(handle)?
            }
            _ => return None,
        })
    }

    /// Render a `[a, b, c]` / `{a, b, c}` collection in the Rust-oracle Debug
    /// shape (stored order for sets, matching the L10 LLVM host).
    fn render_collection(&self, collection: &CollectionValue) -> Option<String> {
        let mut rendered = Vec::with_capacity(collection.values.len());
        for element in &collection.values {
            rendered.push(self.render_element(collection.kind, *element)?);
        }
        let body = rendered.join(", ");
        if collection.set {
            Some(format!("{{{body}}}"))
        } else {
            Some(format!("[{body}]"))
        }
    }

    /// Render a `tabula` handle in the Rust-oracle derived
    /// `Json(Tabula({...}))` Debug shape (keys sorted like the LLVM host's
    /// BTreeMap; non-text keys fail closed).
    fn render_map(&self, map: &MapValue) -> Option<String> {
        let mut entries = BTreeMap::new();
        for (key, value) in &map.entries {
            let RuntimeValue::Handle(key_handle) = key else {
                return None;
            };
            let key = self.resolve_text(*key_handle)?;
            let value = self.render_valor_value(map.value_kind, *value)?;
            entries.insert(key.to_owned(), value);
        }
        Some(format!("Json({})", render_valor_tabula(&entries)))
    }

    /// Render a value in the `Valor` Debug shape (`Numerus(10)` /
    /// `Textus("x")` / `Fractus(1.0)` / `Bivalens(true)` / `Nihil` /
    /// nested `Tabula({...})` / `Lista([...])`).
    fn render_valor_value(&self, kind: u32, value: RuntimeValue) -> Option<String> {
        Some(match (kind, value) {
            (VALUE_KIND_I1, RuntimeValue::I1(value)) => format!("Bivalens({value})"),
            (VALUE_KIND_I32 | VALUE_KIND_I64, RuntimeValue::I32(value)) => {
                format!("Numerus({value})")
            }
            (VALUE_KIND_I64, RuntimeValue::I64(value)) => format!("Numerus({value})"),
            (VALUE_KIND_F64, RuntimeValue::F64(value)) => format!("Fractus({value:?})"),
            (VALUE_KIND_TEXT | VALUE_KIND_ASCII, RuntimeValue::Handle(handle)) => {
                format!("Textus({:?})", self.resolve_text(handle)?)
            }
            (VALUE_KIND_PTR | VALUE_KIND_VALOR, RuntimeValue::Handle(handle)) => {
                self.render_handle_valor(handle)?
            }
            _ => return None,
        })
    }

    /// Render a handle as a nested `Valor` Debug value.
    fn render_handle_valor(&self, handle: i32) -> Option<String> {
        if let Some(text) = self.resolve_text(handle) {
            return Some(format!("Textus({text:?})"));
        }
        if let Some(bytes) = self.resolve_octeti(handle) {
            return Some(format!("Octeti({bytes:?})"));
        }
        if let Some(map) = self.find_map(handle) {
            let mut entries = BTreeMap::new();
            for (key, value) in &map.entries {
                let RuntimeValue::Handle(key_handle) = key else {
                    return None;
                };
                let key = self.resolve_text(*key_handle)?;
                let value = self.render_valor_value(map.value_kind, *value)?;
                entries.insert(key.to_owned(), value);
            }
            return Some(render_valor_tabula(&entries));
        }
        if let Some(collection) = self.find_collection(handle) {
            let mut items = Vec::with_capacity(collection.values.len());
            for element in &collection.values {
                items.push(self.render_valor_value(collection.kind, *element)?);
            }
            return Some(format!("Lista({items:?})"));
        }
        if let Some(option) = self.find_option(handle) {
            return match option.payload {
                Some(payload) => self.render_valor_value(option.kind, payload),
                None => Some("Nihil".to_owned()),
            };
        }
        if handle == 0 {
            return Some("Nihil".to_owned());
        }
        None
    }

    /// Render one opaque handle in the shared oracle display shape (the L10
    /// opaque display contract): text rows render as-is, regex handles render
    /// their pattern, octeti handles render the byte-list Debug shape, and
    /// the W13 collection/scalar arenas render in the Rust-oracle Debug
    /// shapes (`[1, 2, 3]` / `["prima", "secunda"]` / `{1, 2}` /
    /// `Json(Tabula({...}))` / payload-or-`nihil`) — mirroring the LLVM
    /// host's opaque display. Shared by the diagnostic streams
    /// (nota/vide/mone), the format opaque-arg row, and nested
    /// collection/option element display.
    fn render_handle_display(&self, handle: i32) -> Option<String> {
        if let Some(text) = self.resolve_text(handle) {
            return Some(text.to_owned());
        }
        if let Some(regex) = self.resolve_regex(handle) {
            return Some(regex.pattern.clone());
        }
        if let Some(bytes) = self.resolve_octeti(handle) {
            return Some(format!("{bytes:?}"));
        }
        if let Some(collection) = self.find_collection(handle) {
            return self.render_collection(collection);
        }
        if let Some(map) = self.find_map(handle) {
            return self.render_map(map);
        }
        if let Some(option) = self.find_option(handle) {
            return match option.payload {
                Some(payload) => self.render_option_payload(option.kind, payload),
                None => Some("nihil".to_owned()),
            };
        }
        if handle == 0 {
            return Some("nihil".to_owned());
        }
        None
    }

    /// Render an option payload in the nota/option Debug shape: text payloads
    /// render plain, numeric payloads render decimal, nested handles render
    /// recursively (the L10 opaque display contract).
    fn render_option_payload(&self, kind: u32, payload: RuntimeValue) -> Option<String> {
        Some(match (kind, payload) {
            (VALUE_KIND_I1, RuntimeValue::I1(value)) => {
                if value {
                    "verum".to_owned()
                } else {
                    "falsum".to_owned()
                }
            }
            (VALUE_KIND_I32, RuntimeValue::I32(value)) => format!("{value}"),
            (VALUE_KIND_I64, RuntimeValue::I64(value)) => format!("{value}"),
            (VALUE_KIND_F64, RuntimeValue::F64(value)) => display_fractus(value),
            (VALUE_KIND_TEXT | VALUE_KIND_ASCII, RuntimeValue::Handle(handle)) => {
                self.resolve_text(handle)?.to_owned()
            }
            (VALUE_KIND_PTR | VALUE_KIND_VALOR, RuntimeValue::Handle(handle)) => {
                self.render_handle_display(handle)?
            }
            _ => return None,
        })
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

/// Render the `Valor::Tabula` Debug body: `Tabula({})` /
/// `Tabula({"x": Numerus(10), "y": Numerus(20)})` in sorted key order.
fn render_valor_tabula(entries: &BTreeMap<String, String>) -> String {
    let body = entries
        .iter()
        .map(|(key, value)| format!("{key:?}: {value}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("Tabula({{{body}}})")
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
    preflight_closed_surface(module, false)
}

/// U6-E — preflight one package module's closed import surface. Same rule
/// set as the single-module [`preflight_imports`] (the closed `faber_rt_v1`
/// registry plus the U6-B legacy cede exception); when `admit_external` is
/// set, the package-namespace `faber_external` module is deferred to the
/// package resolver ([`preflight_package_imports`]) instead of rejecting.
fn preflight_closed_surface(module: &Module, admit_external: bool) -> Result<(), RunOutcome> {
    for import in module.imports() {
        let module_name = import.module();
        let field = import.name();
        // U6-B recorded decision: the generator cede (yield) channel is an
        // explicit admitted exception on the legacy `faber_runtime` module —
        // exactly the closed field grammar the radix Wasm emitter declares,
        // bound to the cursor-stream yield channel. Every other legacy-module
        // import still rejects below.
        if module_name == LEGACY_CEDE_MODULE && is_cede_yield_field(field) {
            continue;
        }
        if module_name != WASM_IMPORT_MODULE_V1 {
            if admit_external && module_name == FABER_EXTERNAL_IMPORT_MODULE {
                continue;
            }
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

/// Canonical external symbol for one `faber_external` import field:
/// `__faber_` + field (mirrors `radix-mir-wasm` `external_function_import_symbol`).
fn external_canonical_symbol(field: &str) -> String {
    format!("{FABER_EXTERNAL_IMPORT_PREFIX}{field}")
}

/// U6-E — preflight a package run's whole import surface (entry + siblings).
///
/// Every module keeps the closed `faber_rt_v1` preflight (the U6-B cede
/// exception included). The package-namespace `faber_external` module is the
/// one new admission: each of its imports must resolve to the canonical
/// external-symbol export (`__faber_{field}`) of another module in the
/// package set — the entry (index 0) is never a provider (it instantiates
/// last), so the entry's imports resolve against the siblings and a sibling's
/// against the other siblings. An unresolvable external import rejects before
/// any linking with a typed [`RunOutcome::ImportRejected`] naming the module
/// and field — the `wasm_external.rs` `MissingImport` bucket (typed, never a
/// silent default). Single-module runs never enter this path; `faber_external`
/// keeps rejecting there ([`preflight_imports`]).
pub(crate) fn preflight_package_imports(
    entry: &Module,
    siblings: &[Module],
) -> Result<(), RunOutcome> {
    let mut modules = Vec::with_capacity(siblings.len() + 1);
    modules.push(entry);
    modules.extend(siblings.iter());

    // Canonical external-symbol export surface per module.
    let export_sets: Vec<HashSet<String>> = modules
        .iter()
        .map(|module| {
            module
                .exports()
                .filter(|export| export.name().starts_with(FABER_EXTERNAL_IMPORT_PREFIX))
                .map(|export| export.name().to_owned())
                .collect()
        })
        .collect();

    for (index, module) in modules.iter().enumerate() {
        preflight_closed_surface(module, true)?;
        for import in module.imports() {
            if import.module() != FABER_EXTERNAL_IMPORT_MODULE {
                continue;
            }
            let field = import.name();
            let canonical = external_canonical_symbol(field);
            let resolvable = export_sets
                .iter()
                .enumerate()
                .any(|(other, set)| other != index && other != 0 && set.contains(&canonical));
            if !resolvable {
                return Err(RunOutcome::ImportRejected {
                    module: FABER_EXTERNAL_IMPORT_MODULE.to_owned(),
                    field: field.to_owned(),
                    message: format!(
                        "package run cannot resolve `{FABER_EXTERNAL_IMPORT_MODULE}::{field}`: \
                         no other module in the package exports the canonical symbol \
                         `{canonical}` (U6-C package-aware surface); typed missing import"
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Bind every admitted v1 import on the linker. A declared signature that
/// conflicts with the admitted binding fails at bind/instantiate time and
/// surfaces as [`RunOutcome::LinkFailed`]. The module is inspected for the
/// declared cursor-stream signature (the one v1 field whose signature varies
/// with the generator's argument carriers).
pub(crate) fn link_v1_imports(
    linker: &mut Linker<HostState>,
    module: &Module,
) -> Result<(), wasmtime::Error> {
    bind_scalar_i64(linker, "__faber_rt_v1_diagnostic_nota_i64")?;
    bind_scalar_i64(linker, "__faber_rt_v1_diagnostic_mone_i64")?;
    bind_scalar_i64(linker, "__faber_rt_v1_diagnostic_vide_i64")?;
    bind_scalar_i32(linker, "__faber_rt_v1_diagnostic_nota_i32")?;
    bind_scalar_i32(linker, "__faber_rt_v1_diagnostic_nota_i8")?;
    // W13 — bivalens diagnostics render `verum`/`falsum` (never the integer
    // shape); the emitter routes bivalens args to the closed-set `i1` row.
    bind_scalar_i1(linker, "__faber_rt_v1_diagnostic_nota_i1")?;
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
    bind_stdout_text(
        linker,
        "__faber_rt_v1_diagnostic_nota_text",
        "nota/stdout (scribe)",
    )?;
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
    bind_text_predicate1(
        linker,
        "__faber_rt_v1_text_is_empty",
        TextPredicate1::IsEmpty,
    )?;
    bind_text_predicate2(
        linker,
        "__faber_rt_v1_text_contains",
        TextPredicate2::Contains,
    )?;
    bind_text_predicate2(
        linker,
        "__faber_rt_v1_text_starts_with",
        TextPredicate2::StartsWith,
    )?;
    bind_text_predicate2(
        linker,
        "__faber_rt_v1_text_ends_with",
        TextPredicate2::EndsWith,
    )?;
    bind_text_transform(
        linker,
        "__faber_rt_v1_text_uppercase",
        TextTransform::Uppercase,
    )?;
    bind_text_transform(
        linker,
        "__faber_rt_v1_text_lowercase",
        TextTransform::Lowercase,
    )?;
    bind_text_transform(linker, "__faber_rt_v1_text_trim", TextTransform::Trim)?;
    bind_text_slice(linker)?;
    bind_text_split(linker)?;
    bind_text_replace(linker)?;
    // W12 regex conversion rows.
    bind_regex_from_text(linker, "__faber_rt_v1_regex_from_text")?;
    bind_regex_from_text(linker, "__faber_rt_v1_regex_from_ascii")?;
    // W13 scalar display rows.
    bind_assert(linker, "__faber_rt_v1_assert")?;
    bind_assert_message(linker)?;
    bind_text_i64(linker)?;
    bind_text_f64(linker)?;
    bind_text_i1(linker)?;
    bind_text_truthy(linker, "__faber_rt_v1_text_truthy")?;
    bind_text_truthy(linker, "__faber_rt_v1_ascii_truthy")?;
    bind_text_parse_integer(linker)?;
    bind_text_parse_integer_or(linker)?;
    bind_text_parse_float(linker)?;
    bind_text_parse_float_or(linker)?;
    bind_read_line(linker)?;
    // W13 collection display rows.
    bind_array_new(linker)?;
    bind_array_push(linker)?;
    bind_array_extend(linker)?;
    bind_array_length(linker)?;
    bind_array_get(linker)?;
    bind_array_option(linker)?;
    bind_array_contains(linker)?;
    bind_array_is_empty(linker)?;
    bind_array_reverse(linker)?;
    bind_array_sort(linker)?;
    bind_array_sum(linker)?;
    bind_map_new(linker)?;
    bind_map_put(linker)?;
    bind_map_delete(linker)?;
    bind_map_keys(linker)?;
    bind_map_values(linker)?;
    bind_set_new(linker)?;
    bind_set_from_array(linker)?;
    bind_array_from_set(linker)?;
    bind_set_union(linker)?;
    bind_set_intersection(linker)?;
    bind_set_difference(linker)?;
    bind_set_symmetric_difference(linker)?;
    bind_set_is_subset(linker)?;
    bind_set_is_superset(linker)?;
    // W13 option rows.
    bind_option_none(linker)?;
    bind_option_some(linker)?;
    bind_option_get(linker)?;
    bind_option_get_or(linker)?;
    bind_option_is_present(linker)?;
    // W14 tensor display rows.
    bind_tensor_new(linker)?;
    bind_tensor_create(linker)?;
    bind_tensor_from_flat(linker)?;
    bind_tensor_rank(linker)?;
    bind_tensor_shape(linker)?;
    bind_tensor_reshape(linker)?;
    bind_tensor_get(linker)?;
    bind_tensor_set(linker)?;
    bind_tensor_fill(linker)?;
    bind_tensor_flatten(linker)?;
    bind_tensor_materialize(linker)?;
    bind_tensor_slice(linker)?;
    bind_tensor_add_sub_mul(linker, "__faber_rt_v1_tensor_add", TensorBinaryOp::Add)?;
    bind_tensor_add_sub_mul(linker, "__faber_rt_v1_tensor_sub", TensorBinaryOp::Sub)?;
    bind_tensor_add_sub_mul(linker, "__faber_rt_v1_tensor_mul", TensorBinaryOp::Mul)?;
    bind_tensor_matmul(linker)?;
    bind_tensor_sum(linker)?;
    bind_tensor_mean(linker)?;
    bind_tensor_convert(linker)?;
    // U6-B cursor-stream materialization + the cede (yield) channel.
    bind_cursor_stream(linker, module)?;
    bind_cede_fields(linker)?;
    Ok(())
}

/// U6-E — bind one module's `faber_external` imports against the
/// already-instantiated provider instances. Each import field `F` resolves
/// to the provider export named `__faber_{F}` (the canonical external
/// symbol). A provider that is not yet instantiated — a package provided out
/// of dependency order — or a declared signature that conflicts with the
/// provider's export fails at link/instantiate time and surfaces as
/// [`RunOutcome::LinkFailed`]; the package preflight gate
/// ([`preflight_package_imports`]) has already proved the symbol exists
/// somewhere in the package set.
pub(crate) fn bind_external_imports(
    linker: &mut Linker<HostState>,
    store: &mut Store<HostState>,
    module: &Module,
    providers: &[Instance],
) -> Result<(), wasmtime::Error> {
    for import in module.imports() {
        if import.module() != FABER_EXTERNAL_IMPORT_MODULE {
            continue;
        }
        let field = import.name();
        let canonical = external_canonical_symbol(field);
        let Some(func) = providers
            .iter()
            .find_map(|instance| instance.get_func(&mut *store, &canonical))
        else {
            return Err(wasmtime::Error::msg(format!(
                "package external import `{FABER_EXTERNAL_IMPORT_MODULE}::{field}` has no \
                 instantiated provider for `{canonical}` (dependency order or package assembly)"
            )));
        };
        linker.define(&mut *store, FABER_EXTERNAL_IMPORT_MODULE, field, func)?;
    }
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
            // The oracle Debug shape keeps the `.0` marker for integral
            // floats (`nota 9.0` → `9.0`), matching the collection/option
            // display renderer (`display_fractus`) and the LLVM lane.
            caller.data_mut().write_line(&display_fractus(value));
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
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32|
              -> Result<(), wasmtime::Error> {
            let rendered = caller.data().render_handle_display(handle);
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
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32|
              -> Result<(), wasmtime::Error> {
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
            let rendered = caller.data().render_handle_display(handle);
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
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32|
              -> Result<(), wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32|
              -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32|
              -> Result<i32, wasmtime::Error> {
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
    Ok(caller
        .data_mut()
        .alloc_text(render_template(&template_text, &args)))
}

fn bind_format_i1(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_format_i1",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              template: i32,
              value: i32|
              -> Result<i32, wasmtime::Error> {
            format_result(
                &mut caller,
                template,
                vec![display_bivalens(value).to_owned()],
            )
        },
    )?;
    Ok(())
}

fn bind_format_i64(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_format_i64",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              template: i32,
              value: i64|
              -> Result<i32, wasmtime::Error> {
            format_result(&mut caller, template, vec![value.to_string()])
        },
    )?;
    Ok(())
}

fn bind_format_i64_i64(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_format_i64_i64",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              template: i32,
              first: i64,
              second: i64|
              -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>,
              template: i32,
              first: i64,
              second: i64,
              third: i64|
              -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>,
              template: i32,
              value: f64|
              -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>,
              template: i32,
              text: i32|
              -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>,
              template: i32,
              first: i32,
              second: i32|
              -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>,
              template: i32,
              text: i32,
              value: i64|
              -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>,
              template: i32,
              value: i64,
              text: i32|
              -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>,
              template: i32,
              first: i32,
              second: i32,
              third: i32|
              -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>,
              template: i32,
              text: i32,
              integer: i64,
              boolean: i32|
              -> Result<i32, wasmtime::Error> {
            let Some(text) = caller.data().resolve_text(text).map(str::to_owned) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("format text arg handle {text}: unknown text handle"),
                ));
            };
            format_result(
                &mut caller,
                template,
                vec![
                    text,
                    integer.to_string(),
                    display_bivalens(boolean).to_owned(),
                ],
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
        move |mut caller: wasmtime::Caller<'_, HostState>,
              template: i32,
              value: i32|
              -> Result<i32, wasmtime::Error> {
            let Some(rendered) = caller.data().render_handle_display(value) else {
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
        move |mut caller: wasmtime::Caller<'_, HostState>,
              first: i32,
              second: i32|
              -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>,
              first: i32,
              second: i32|
              -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>,
              text: i32|
              -> Result<i64, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>,
              text: i32|
              -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>,
              text: i32,
              other: i32|
              -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>,
              text: i32|
              -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>,
              text: i32,
              start: i64,
              end: i64|
              -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>,
              _text: i32,
              _separator: i32|
              -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>,
              text: i32,
              old: i32,
              new: i32|
              -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>,
              text: i32|
              -> Result<i32, wasmtime::Error> {
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

// ---------------------------------------------------------------------------
// W13 collection/scalar display rows
// ---------------------------------------------------------------------------

/// `array_new (param i32) (result i32)`: kind → empty `lista`.
fn bind_array_new(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_array_new",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              kind: i32|
              -> Result<i32, wasmtime::Error> {
            Ok(caller
                .data_mut()
                .alloc_collection(false, kind as u32, Vec::new()))
        },
    )?;
    Ok(())
}

/// `array_push (param i32 i64) (result i32)`: append one element (interpreted
/// per the collection's declared kind) and return the collection handle.
fn bind_array_push(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_array_push",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32,
              value: i64|
              -> Result<i32, wasmtime::Error> {
            let collection = caller.data().find_collection(handle).map(|c| c.kind);
            if let Some(kind) = collection {
                let Some(value) = value_from_i64(kind, value) else {
                    return Err(typed_unsupported(
                        &mut caller,
                        format!("array_push handle {handle}: unknown element kind {kind}"),
                    ));
                };
                let Some(collection) = caller.data_mut().find_collection_mut(handle) else {
                    return Err(typed_unsupported(
                        &mut caller,
                        format!("array_push handle {handle}: unknown collection handle"),
                    ));
                };
                collection.values.push(value);
                return Ok(handle);
            }
            // An `octeti` unifies with `lista<u8>`: `appende` mutates the
            // interned byte payload.
            if let Some(bytes) = caller.data_mut().resolve_octeti_mut(handle) {
                bytes.push(value as u8);
                return Ok(handle);
            }
            Err(typed_unsupported(
                &mut caller,
                format!("array_push handle {handle}: unknown collection handle"),
            ))
        },
    )?;
    Ok(())
}

/// `array_extend (param i32 i32) (result i32)`: splice one collection into
/// another (`[sparge src, ...]`) and return the destination handle.
fn bind_array_extend(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_array_extend",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              destination: i32,
              source: i32|
              -> Result<i32, wasmtime::Error> {
            let state = caller.data();
            let dest_kind = state.find_collection(destination).map(|c| c.kind);
            let source_values = state.find_collection(source).map(|c| c.values.clone());
            let (Some(dest_kind), Some(source_values)) = (dest_kind, source_values) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("array_extend {destination} <- {source}: unknown collection handle"),
                ));
            };
            let Some(collection) = caller.data_mut().find_collection_mut(destination) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("array_extend {destination}: unknown collection handle"),
                ));
            };
            if collection.kind != dest_kind {
                return Err(typed_unsupported(
                    &mut caller,
                    "array_extend element-kind mismatch",
                ));
            }
            collection.values.extend(source_values);
            Ok(destination)
        },
    )?;
    Ok(())
}

/// `array_length (param i32) (result i64)`.
fn bind_array_length(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_array_length",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32|
              -> Result<i64, wasmtime::Error> {
            if let Some(collection) = caller.data().find_collection(handle) {
                return Ok(i64::try_from(collection.values.len()).expect("length fits i64"));
            }
            if let Some(map) = caller.data().find_map(handle) {
                return Ok(i64::try_from(map.entries.len()).expect("length fits i64"));
            }
            // An `octeti` unifies with `lista<u8>`: length reads the payload.
            if let Some(bytes) = caller.data().resolve_octeti(handle) {
                return Ok(i64::try_from(bytes.len()).expect("length fits i64"));
            }
            Err(typed_unsupported(
                &mut caller,
                format!("array_length handle {handle}: unknown collection handle"),
            ))
        },
    )?;
    Ok(())
}

/// `array_get (param i32 i64) (result i64)`: read one element as an i64
/// carrier. Out-of-range reads return `0` (the null-encoded option handle for
/// `?[` chains; direct out-of-range subscripts do not panic in this stage).
fn bind_array_get(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_array_get",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32,
              index: i64|
              -> Result<i64, wasmtime::Error> {
            let state = caller.data();
            if let Some(collection) = state.find_collection(handle) {
                let Ok(index) = usize::try_from(index) else {
                    return Ok(0);
                };
                return Ok(match collection.values.get(index) {
                    Some(value) => value_to_i64(*value),
                    None => 0,
                });
            }
            // `octeti` unifies with `lista<u8>`: reads return the byte.
            if let Some(bytes) = state.resolve_octeti(handle) {
                let Ok(index) = usize::try_from(index) else {
                    return Ok(0);
                };
                return Ok(bytes.get(index).copied().map(i64::from).unwrap_or(0));
            }
            Err(typed_unsupported(
                &mut caller,
                format!("array_get handle {handle}: unknown collection handle"),
            ))
        },
    )?;
    Ok(())
}

/// `array_option (param i32 i64) (result i32)`: index/key lookup returning an
/// option result. The emitter normalizes `first`/`last`/`remove_first`/
/// `remove_last`/`index` onto this row and discriminates the op by a negative
/// key sentinel (`-1`..`-4`; indices and text-key handles are never
/// negative). `remove_first`/`remove_last` also mutate the collection.
fn bind_array_option(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_array_option",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32,
              key: i64|
              -> Result<i32, wasmtime::Error> {
            // Resolve `(kind, payload)` under a scoped borrow, then encode the
            // option result mutably.
            let kind_payload = {
                let state = caller.data();
                if let Some(collection) = state.find_collection(handle) {
                    let kind = collection.kind;
                    let index = match key {
                        -1 => 0,
                        -2 => collection.values.len().saturating_sub(1),
                        _ => usize::try_from(key).unwrap_or(usize::MAX),
                    };
                    let remove = matches!(key, -3 | -4);
                    let last = key == -4 || key == -2;
                    let payload = if remove {
                        None
                    } else {
                        collection.values.get(index).copied()
                    };
                    Some((kind, payload, remove, last))
                } else if let Some(map) = state.find_map(handle) {
                    let key_handle = key as i32;
                    let payload = map
                        .entries
                        .iter()
                        .find(|(k, _)| *k == RuntimeValue::Handle(key_handle))
                        .map(|(_, v)| *v);
                    Some((map.value_kind, payload, false, false))
                } else if let Some(bytes) = state.resolve_octeti(handle) {
                    let Ok(index) = usize::try_from(key) else {
                        return Ok(0);
                    };
                    let payload = bytes
                        .get(index)
                        .copied()
                        .map(|b| RuntimeValue::I32(b as i32));
                    Some((VALUE_KIND_U8, payload, false, false))
                } else {
                    None
                }
            };
            let Some((kind, payload, remove, last)) = kind_payload else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("array_option handle {handle}: unknown collection handle"),
                ));
            };
            let payload = if remove {
                if last {
                    caller
                        .data_mut()
                        .find_collection_mut(handle)
                        .and_then(|c| c.values.pop())
                } else {
                    caller.data_mut().find_collection_mut(handle).and_then(|c| {
                        if c.values.is_empty() {
                            None
                        } else {
                            Some(c.values.remove(0))
                        }
                    })
                }
            } else {
                payload
            };
            Ok(caller.data_mut().option_result(kind, payload))
        },
    )?;
    Ok(())
}

/// `array_contains (param i32 i64) (result i32)`: element/value present?
fn bind_array_contains(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_array_contains",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32,
              value: i64|
              -> Result<i32, wasmtime::Error> {
            let state = caller.data();
            if let Some(collection) = state.find_collection(handle) {
                let Some(value) = value_from_i64(collection.kind, value) else {
                    return Err(typed_unsupported(
                        &mut caller,
                        format!("array_contains handle {handle}: unknown element kind"),
                    ));
                };
                return Ok(i32::from(
                    collection
                        .values
                        .iter()
                        .any(|v| runtime_value_eq(*v, value)),
                ));
            }
            if let Some(map) = state.find_map(handle) {
                let key_handle = value as i32;
                return Ok(i32::from(
                    map.entries
                        .iter()
                        .any(|(k, _)| *k == RuntimeValue::Handle(key_handle)),
                ));
            }
            Err(typed_unsupported(
                &mut caller,
                format!("array_contains handle {handle}: unknown collection handle"),
            ))
        },
    )?;
    Ok(())
}

/// `array_is_empty (param i32) (result i32)`.
fn bind_array_is_empty(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_array_is_empty",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32|
              -> Result<i32, wasmtime::Error> {
            let state = caller.data();
            if let Some(collection) = state.find_collection(handle) {
                return Ok(i32::from(collection.values.is_empty()));
            }
            if let Some(map) = state.find_map(handle) {
                return Ok(i32::from(map.entries.is_empty()));
            }
            Err(typed_unsupported(
                &mut caller,
                format!("array_is_empty handle {handle}: unknown collection handle"),
            ))
        },
    )?;
    Ok(())
}

/// `array_reverse (param i32)`: in-place reversal.
fn bind_array_reverse(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_array_reverse",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32|
              -> Result<(), wasmtime::Error> {
            let Some(collection) = caller.data_mut().find_collection_mut(handle) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("array_reverse handle {handle}: unknown collection handle"),
                ));
            };
            collection.values.reverse();
            Ok(())
        },
    )?;
    Ok(())
}

/// `array_sort (param i32) (result i32)`: sort the collection (in place) and
/// return its handle.
fn bind_array_sort(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_array_sort",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32|
              -> Result<i32, wasmtime::Error> {
            if caller.data().find_collection(handle).is_none() {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("array_sort handle {handle}: unknown collection handle"),
                ));
            }
            let Some(collection) = caller.data_mut().find_collection_mut(handle) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("array_sort handle {handle}: unknown collection handle"),
                ));
            };
            collection.values.sort_by_key(|value| value_to_i64(*value));
            Ok(handle)
        },
    )?;
    Ok(())
}

/// `array_sum (param i32) (result i64)`: sum of the collection's elements
/// (i64 interpretation; the f64-sum shape stays a separate cluster).
fn bind_array_sum(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_array_sum",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32|
              -> Result<i64, wasmtime::Error> {
            let state = caller.data();
            let Some(collection) = state.find_collection(handle) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("array_sum handle {handle}: unknown collection handle"),
                ));
            };
            let mut sum = 0i64;
            for value in &collection.values {
                sum = sum.wrapping_add(value_to_i64(*value));
            }
            Ok(sum)
        },
    )?;
    Ok(())
}

/// `map_new (param i32 i32) (result i32)`: (key kind, value kind) → empty
/// `tabula`.
fn bind_map_new(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_map_new",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              key_kind: i32,
              value_kind: i32|
              -> Result<i32, wasmtime::Error> {
            Ok(caller
                .data_mut()
                .alloc_map(key_kind as u32, value_kind as u32, Vec::new()))
        },
    )?;
    Ok(())
}

/// `map_put (param i32 i32 i64)`: (map, key-handle, value-as-i64). The value
/// is interpreted per the map's declared value kind.
fn bind_map_put(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_map_put",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32,
              key: i32,
              value: i64|
              -> Result<(), wasmtime::Error> {
            let state = caller.data();
            let (value_kind, key_kind) = match state.find_map(handle) {
                Some(map) => (map.value_kind, map.key_kind),
                None => {
                    return Err(typed_unsupported(
                        &mut caller,
                        format!("map_put handle {handle}: unknown map handle"),
                    ));
                }
            };
            let Some(value) = value_from_i64(value_kind, value) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("map_put handle {handle}: unknown value kind {value_kind}"),
                ));
            };
            let Some(map) = caller.data_mut().find_map_mut(handle) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("map_put handle {handle}: unknown map handle"),
                ));
            };
            let key_value =
                value_from_i64(key_kind, i64::from(key)).unwrap_or(RuntimeValue::Handle(key));
            // Insert or replace by key identity.
            if let Some(entry) = map
                .entries
                .iter_mut()
                .find(|(k, _)| runtime_value_eq(*k, key_value))
            {
                entry.1 = value;
            } else {
                map.entries.push((key_value, value));
            }
            Ok(())
        },
    )?;
    Ok(())
}

/// `map_delete (param i32 i64) (result i32)`: remove the key (as i64 carrier;
/// text keys widen their handle) and return whether it was present. A `copia`
/// handle is a set: delete removes the element value.
fn bind_map_delete(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_map_delete",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32,
              key: i64|
              -> Result<i32, wasmtime::Error> {
            let state = caller.data();
            if let Some(map) = state.find_map(handle) {
                let key_kind = map.key_kind;
                let key_value =
                    value_from_i64(key_kind, key).unwrap_or(RuntimeValue::Handle(key as i32));
                let Some(map) = caller.data_mut().find_map_mut(handle) else {
                    return Err(typed_unsupported(
                        &mut caller,
                        format!("map_delete handle {handle}: unknown map handle"),
                    ));
                };
                let before = map.entries.len();
                map.entries
                    .retain(|(k, _)| !runtime_value_eq(*k, key_value));
                return Ok(i32::from(map.entries.len() != before));
            }
            // `copia.dele(x)` deletes the element value (sets route the delete
            // op through the closed map_delete row).
            if let Some(collection) = state.find_collection(handle) {
                let kind = collection.kind;
                let Some(value) = value_from_i64(kind, key) else {
                    return Err(typed_unsupported(
                        &mut caller,
                        format!("map_delete handle {handle}: unknown element kind {kind}"),
                    ));
                };
                let Some(collection) = caller.data_mut().find_collection_mut(handle) else {
                    return Err(typed_unsupported(
                        &mut caller,
                        format!("map_delete handle {handle}: unknown collection handle"),
                    ));
                };
                let before = collection.values.len();
                collection
                    .values
                    .retain(|existing| !runtime_value_eq(*existing, value));
                return Ok(i32::from(collection.values.len() != before));
            }
            Err(typed_unsupported(
                &mut caller,
                format!("map_delete handle {handle}: unknown map handle"),
            ))
        },
    )?;
    Ok(())
}

/// `map_keys (param i32) (result i32)`: a `lista<textus>` of the map's keys.
fn bind_map_keys(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_map_keys",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32|
              -> Result<i32, wasmtime::Error> {
            let state = caller.data();
            let Some(map) = state.find_map(handle) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("map_keys handle {handle}: unknown map handle"),
                ));
            };
            let mut keys = Vec::with_capacity(map.entries.len());
            for (key, _) in &map.entries {
                keys.push(*key);
            }
            Ok(caller
                .data_mut()
                .alloc_collection(false, VALUE_KIND_TEXT, keys))
        },
    )?;
    Ok(())
}

/// `map_values (param i32) (result i32)`: a `lista` of the map's values.
fn bind_map_values(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_map_values",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32|
              -> Result<i32, wasmtime::Error> {
            let state = caller.data();
            let Some(map) = state.find_map(handle) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("map_values handle {handle}: unknown map handle"),
                ));
            };
            let kind = map.value_kind;
            let mut values = Vec::with_capacity(map.entries.len());
            for (_, value) in &map.entries {
                values.push(*value);
            }
            Ok(caller.data_mut().alloc_collection(false, kind, values))
        },
    )?;
    Ok(())
}

/// `set_new (param i32) (result i32)`: kind → empty `copia`.
fn bind_set_new(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_set_new",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              kind: i32|
              -> Result<i32, wasmtime::Error> {
            Ok(caller
                .data_mut()
                .alloc_collection(true, kind as u32, Vec::new()))
        },
    )?;
    Ok(())
}

/// `set_from_array (param i32) (result i32)`: build a `copia` from a `lista`
/// (dedup, first-seen order).
fn bind_set_from_array(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_set_from_array",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32|
              -> Result<i32, wasmtime::Error> {
            let state = caller.data();
            let Some(collection) = state.find_collection(handle) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("set_from_array handle {handle}: unknown collection handle"),
                ));
            };
            let kind = collection.kind;
            let mut unique = Vec::new();
            for value in &collection.values {
                if !unique
                    .iter()
                    .any(|existing| runtime_value_eq(*existing, *value))
                {
                    unique.push(*value);
                }
            }
            Ok(caller.data_mut().alloc_collection(true, kind, unique))
        },
    )?;
    Ok(())
}

/// `array_from_set (param i32) (result i32)`: a `lista` copy of a `copia`.
fn bind_array_from_set(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_array_from_set",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32|
              -> Result<i32, wasmtime::Error> {
            let state = caller.data();
            let Some(collection) = state.find_collection(handle) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("array_from_set handle {handle}: unknown collection handle"),
                ));
            };
            let kind = collection.kind;
            let values = collection.values.clone();
            Ok(caller.data_mut().alloc_collection(false, kind, values))
        },
    )?;
    Ok(())
}

/// One set algebra op shared by the `set_union`/`intersection`/`difference`/
/// `symmetric_difference`/`is_subset`/`is_superset` rows. Returns a result
/// handle or a bivalens.
fn set_algebra<F>(
    caller: &mut wasmtime::Caller<'_, HostState>,
    left: i32,
    right: i32,
    reduce: F,
) -> Result<i32, wasmtime::Error>
where
    F: Fn(&CollectionValue, &CollectionValue) -> Result<Vec<RuntimeValue>, ()>,
{
    let state = caller.data();
    let (Some(left), Some(right)) = (state.find_collection(left), state.find_collection(right))
    else {
        return Err(typed_unsupported(
            caller,
            "set algebra received an unknown collection handle",
        ));
    };
    let kind = left.kind;
    let values = reduce(left, right)
        .map_err(|()| typed_unsupported(caller, "set algebra element-kind mismatch"))?;
    Ok(caller.data_mut().alloc_collection(true, kind, values))
}

/// `set_union (param i32 i32) (result i32)`.
fn bind_set_union(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_set_union",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              left: i32,
              right: i32|
              -> Result<i32, wasmtime::Error> {
            set_algebra(&mut caller, left, right, |l, r| {
                let mut values = l.values.clone();
                for value in &r.values {
                    if !values.iter().any(|v| runtime_value_eq(*v, *value)) {
                        values.push(*value);
                    }
                }
                Ok(values)
            })
        },
    )?;
    Ok(())
}

/// `set_intersection (param i32 i32) (result i32)`.
fn bind_set_intersection(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_set_intersection",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              left: i32,
              right: i32|
              -> Result<i32, wasmtime::Error> {
            set_algebra(&mut caller, left, right, |l, r| {
                Ok(l.values
                    .iter()
                    .copied()
                    .filter(|value| r.values.iter().any(|v| runtime_value_eq(*v, *value)))
                    .collect())
            })
        },
    )?;
    Ok(())
}

/// `set_difference (param i32 i32) (result i32)`.
fn bind_set_difference(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_set_difference",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              left: i32,
              right: i32|
              -> Result<i32, wasmtime::Error> {
            set_algebra(&mut caller, left, right, |l, r| {
                Ok(l.values
                    .iter()
                    .copied()
                    .filter(|value| !r.values.iter().any(|v| runtime_value_eq(*v, *value)))
                    .collect())
            })
        },
    )?;
    Ok(())
}

/// `set_symmetric_difference (param i32 i32) (result i32)`.
fn bind_set_symmetric_difference(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_set_symmetric_difference",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              left: i32,
              right: i32|
              -> Result<i32, wasmtime::Error> {
            set_algebra(&mut caller, left, right, |l, r| {
                let mut values = Vec::new();
                for value in &l.values {
                    if !r.values.iter().any(|v| runtime_value_eq(*v, *value)) {
                        values.push(*value);
                    }
                }
                for value in &r.values {
                    if !l.values.iter().any(|v| runtime_value_eq(*v, *value)) {
                        values.push(*value);
                    }
                }
                Ok(values)
            })
        },
    )?;
    Ok(())
}

/// `set_is_subset (param i32 i32) (result i32)`.
fn bind_set_is_subset(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_set_is_subset",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              left: i32,
              right: i32|
              -> Result<i32, wasmtime::Error> {
            let state = caller.data();
            let (Some(left), Some(right)) =
                (state.find_collection(left), state.find_collection(right))
            else {
                return Err(typed_unsupported(
                    &mut caller,
                    "set_is_subset received an unknown collection handle",
                ));
            };
            Ok(i32::from(left.values.iter().all(|value| {
                right.values.iter().any(|v| runtime_value_eq(*v, *value))
            })))
        },
    )?;
    Ok(())
}

/// `set_is_superset (param i32 i32) (result i32)`.
fn bind_set_is_superset(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_set_is_superset",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              left: i32,
              right: i32|
              -> Result<i32, wasmtime::Error> {
            let state = caller.data();
            let (Some(left), Some(right)) =
                (state.find_collection(left), state.find_collection(right))
            else {
                return Err(typed_unsupported(
                    &mut caller,
                    "set_is_superset received an unknown collection handle",
                ));
            };
            Ok(i32::from(right.values.iter().all(|value| {
                left.values.iter().any(|v| runtime_value_eq(*v, *value))
            })))
        },
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// W13 option rows
// ---------------------------------------------------------------------------

/// `option_none (param i32) (result i32)`: kind → absent option handle.
fn bind_option_none(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_option_none",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              kind: i32|
              -> Result<i32, wasmtime::Error> {
            Ok(caller.data_mut().alloc_option(kind as u32, None))
        },
    )?;
    Ok(())
}

/// `option_some (param i64 i32) (result i32)`: (payload-as-i64, kind) →
/// present option handle.
fn bind_option_some(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_option_some",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              value: i64,
              kind: i32|
              -> Result<i32, wasmtime::Error> {
            let Some(payload) = value_from_i64(kind as u32, value) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("option_some received an unknown payload kind {kind}"),
                ));
            };
            Ok(caller.data_mut().alloc_option(kind as u32, Some(payload)))
        },
    )?;
    Ok(())
}

/// `option_get (param i32) (result i64)`: payload as i64 (0 for absent).
fn bind_option_get(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_option_get",
        move |caller: wasmtime::Caller<'_, HostState>,
              handle: i32|
              -> Result<i64, wasmtime::Error> {
            let state = caller.data();
            match state.find_option(handle) {
                Some(option) => match option.payload {
                    Some(payload) => Ok(value_to_i64(payload)),
                    None => Ok(0),
                },
                None => Ok(i64::from(handle)),
            }
        },
    )?;
    Ok(())
}

/// `option_get_or (param i32 i64) (result i64)`: present → payload, absent →
/// fallback (both as i64 carriers).
fn bind_option_get_or(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_option_get_or",
        move |caller: wasmtime::Caller<'_, HostState>,
              handle: i32,
              fallback: i64|
              -> Result<i64, wasmtime::Error> {
            let state = caller.data();
            match state.find_option(handle) {
                Some(option) => match option.payload {
                    Some(payload) => Ok(value_to_i64(payload)),
                    None => Ok(fallback),
                },
                None => Ok(fallback),
            }
        },
    )?;
    Ok(())
}

/// `option_is_present (param i32) (result i32)`: arena option present, or a
/// raw null-encoded handle being non-zero.
fn bind_option_is_present(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_option_is_present",
        move |caller: wasmtime::Caller<'_, HostState>,
              handle: i32|
              -> Result<i32, wasmtime::Error> {
            let state = caller.data();
            Ok(match state.find_option(handle) {
                Some(option) => i32::from(option.payload.is_some()),
                None => i32::from(handle != 0),
            })
        },
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// W13 scalar display + conversion rows
// ---------------------------------------------------------------------------

/// `nota_i1 (param i32)`: bivalens diagnostics render `verum`/`falsum`.
fn bind_scalar_i1(
    linker: &mut Linker<HostState>,
    field: &'static str,
) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        field,
        move |mut caller: wasmtime::Caller<'_, HostState>, value: i32| {
            caller
                .data_mut()
                .write_line(if value != 0 { "verum" } else { "falsum" });
            Ok(())
        },
    )?;
    Ok(())
}

/// `assert (param i32)` / `assert_message (param i32 i32)`: an `adfirma` that
/// fails aborts the run with a typed runtime failure (never a silent no-op).
fn bind_assert(linker: &mut Linker<HostState>, field: &'static str) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        field,
        move |mut caller: wasmtime::Caller<'_, HostState>,
              condition: i32|
              -> Result<(), wasmtime::Error> {
            if condition == 0 {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("`{field}` assertion failed"),
                ));
            }
            Ok(())
        },
    )?;
    Ok(())
}

/// `assert_message (param i32 i32)`: assert with a message handle.
fn bind_assert_message(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_assert_message",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              condition: i32,
              message: i32|
              -> Result<(), wasmtime::Error> {
            if condition == 0 {
                let message = caller
                    .data()
                    .resolve_text(message)
                    .map(str::to_owned)
                    .unwrap_or_else(|| "<unknown>".to_owned());
                return Err(typed_unsupported(
                    &mut caller,
                    format!("assertion failed: {message}"),
                ));
            }
            Ok(())
        },
    )?;
    Ok(())
}

/// `text_i64 (param i64) (result i32)`: scalar → text conversion.
fn bind_text_i64(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_text_i64",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              value: i64|
              -> Result<i32, wasmtime::Error> {
            Ok(caller.data_mut().alloc_text(value.to_string()))
        },
    )?;
    Ok(())
}

/// `text_f64 (param f64) (result i32)`.
fn bind_text_f64(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_text_f64",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              value: f64|
              -> Result<i32, wasmtime::Error> {
            Ok(caller.data_mut().alloc_text(display_fractus(value)))
        },
    )?;
    Ok(())
}

/// `text_i1 (param i32) (result i32)`.
fn bind_text_i1(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_text_i1",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              value: i32|
              -> Result<i32, wasmtime::Error> {
            Ok(caller.data_mut().alloc_text(if value != 0 {
                "verum".to_owned()
            } else {
                "falsum".to_owned()
            }))
        },
    )?;
    Ok(())
}

/// `text_truthy (param i32) (result i32)` / `ascii_truthy (param i32)
/// (result i32)`: text/ascii → bivalens carrier (non-empty text is verum).
fn bind_text_truthy(
    linker: &mut Linker<HostState>,
    field: &'static str,
) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        field,
        move |caller: wasmtime::Caller<'_, HostState>, text: i32| -> Result<i32, wasmtime::Error> {
            let truthy = caller
                .data()
                .resolve_text(text)
                .map(|text| !text.is_empty())
                .unwrap_or(false);
            Ok(i32::from(truthy))
        },
    )?;
    Ok(())
}

/// `text_parse_integer (param i32) (result i64)`: `textus ↦ numerus` (0 on
/// parse failure, mirroring the shared oracle's fallback).
fn bind_text_parse_integer(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_text_parse_integer",
        move |caller: wasmtime::Caller<'_, HostState>, text: i32| -> Result<i64, wasmtime::Error> {
            let parsed = caller
                .data()
                .resolve_text(text)
                .and_then(|text| text.trim().parse::<i64>().ok())
                .unwrap_or(0);
            Ok(parsed)
        },
    )?;
    Ok(())
}

/// `text_parse_integer_or (param i32 i64) (result i64)`: parse with a
/// fallback value.
fn bind_text_parse_integer_or(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_text_parse_integer_or",
        move |caller: wasmtime::Caller<'_, HostState>,
              text: i32,
              fallback: i64|
              -> Result<i64, wasmtime::Error> {
            let parsed = caller
                .data()
                .resolve_text(text)
                .and_then(|text| text.trim().parse::<i64>().ok())
                .unwrap_or(fallback);
            Ok(parsed)
        },
    )?;
    Ok(())
}

/// `text_parse_float (param i32) (result f64)`.
fn bind_text_parse_float(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_text_parse_float",
        move |caller: wasmtime::Caller<'_, HostState>, text: i32| -> Result<f64, wasmtime::Error> {
            let parsed = caller
                .data()
                .resolve_text(text)
                .and_then(|text| text.trim().parse::<f64>().ok())
                .unwrap_or(0.0);
            Ok(parsed)
        },
    )?;
    Ok(())
}

/// `text_parse_float_or (param i32 f64) (result f64)`.
fn bind_text_parse_float_or(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_text_parse_float_or",
        move |caller: wasmtime::Caller<'_, HostState>,
              text: i32,
              fallback: f64|
              -> Result<f64, wasmtime::Error> {
            let parsed = caller
                .data()
                .resolve_text(text)
                .and_then(|text| text.trim().parse::<f64>().ok())
                .unwrap_or(fallback);
            Ok(parsed)
        },
    )?;
    Ok(())
}

/// `read_line_0_to_ptr (result i32)`: `lege` reads one stdin line as
/// `option<textus>`. The product host `RunConfig` carries no stdin adapter, so
/// the honest outcome is an absent option (typed; never a synthesized text).
fn bind_read_line(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_read_line_0_to_ptr",
        move |mut caller: wasmtime::Caller<'_, HostState>| -> Result<i32, wasmtime::Error> {
            let kind = VALUE_KIND_TEXT as i32;
            Ok(caller.data_mut().alloc_option(kind as u32, None))
        },
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// W14 tensor display rows
// ---------------------------------------------------------------------------

/// Resolve a tensor element value carrier helper: element kinds whose scalar
/// carrier is f64 (`fractus<f32>`/`fractus<f64>`) cross the f64 row carrier;
/// numerus kinds cross i64; handle kinds cross i32 handles. Mirrors the
/// emitter's `collection_element_kind`-driven carriers.
fn tensor_value_from_carrier(
    tensor: &TensorValue,
    value: RuntimeValue,
) -> Result<RuntimeValue, String> {
    match (tensor.kind, value) {
        (VALUE_KIND_F32 | VALUE_KIND_F64, RuntimeValue::F64(value)) => Ok(RuntimeValue::F64(value)),
        (VALUE_KIND_I64 | VALUE_KIND_U64, RuntimeValue::I64(value)) => Ok(RuntimeValue::I64(value)),
        (
            VALUE_KIND_I8 | VALUE_KIND_I16 | VALUE_KIND_I32 | VALUE_KIND_U8 | VALUE_KIND_U16
            | VALUE_KIND_U32,
            RuntimeValue::I64(value),
        ) => Ok(RuntimeValue::I32(value as i32)),
        (VALUE_KIND_TEXT | VALUE_KIND_ASCII, RuntimeValue::Handle(handle)) => {
            Ok(RuntimeValue::Handle(handle))
        }
        _ => Err(format!(
            "tensor element kind {} with value {value:?}",
            tensor.kind
        )),
    }
}

/// Read one index vector from a `lista` handle (the index dims the emitter
/// passes as an i64 collection). Returns `None` on unknown/unsupported shape.
fn read_index_vector(state: &HostState, handle: i32) -> Option<Vec<i64>> {
    let collection = state.find_collection(handle)?;
    collection
        .values
        .iter()
        .map(|value| match value {
            RuntimeValue::I64(value) => Some(*value),
            RuntimeValue::I32(value) => Some(i64::from(*value)),
            _ => None,
        })
        .collect()
}

/// `tensor_new (param i32) (result i32)`: element kind → empty rank-0 tensor.
fn bind_tensor_new(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_tensor_new",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              kind: i32|
              -> Result<i32, wasmtime::Error> {
            Ok(caller.data_mut().alloc_tensor(TensorValue {
                kind: kind as u32,
                shape: Vec::new(),
                values: Vec::new(),
            }))
        },
    )?;
    Ok(())
}

/// `tensor_create (param i32 f64 i32) (result i32)`: (seed receiver, fill
/// value, shape lista) → dense tensor with the given shape filled with the
/// value (the seed carries the element kind).
fn bind_tensor_create(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_tensor_create",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              seed: i32,
              fill: f64,
              shape: i32|
              -> Result<i32, wasmtime::Error> {
            let (kind, shape) = (|| {
                let state = caller.data();
                let tensor = state
                    .find_tensor(seed)
                    .ok_or_else(|| format!("tensor_create seed {seed}: unknown tensor handle"))?;
                let shape = read_index_vector(state, shape)
                    .ok_or_else(|| "tensor_create shape must be a lista of dims".to_owned())?;
                Ok::<_, String>((tensor.kind, shape))
            })()
            .map_err(|message| typed_unsupported(&mut caller, message))?;
            let count = tensor_shape_element_count(&shape).ok_or_else(|| {
                typed_unsupported(&mut caller, "tensor_create shape element count overflow")
            })?;
            let value = tensor_value_from_carrier(
                &TensorValue {
                    kind,
                    shape: shape.clone(),
                    values: Vec::new(),
                },
                RuntimeValue::F64(fill),
            )
            .map_err(|message| typed_unsupported(&mut caller, message))?;
            Ok(caller.data_mut().alloc_tensor(TensorValue {
                kind,
                shape,
                values: vec![value; count],
            }))
        },
    )?;
    Ok(())
}

/// `tensor_from_flat (param i32 i32 i32) (result i32)`: (seed receiver, flat
/// lista, shape lista) → dense tensor whose values are the flat lista's
/// elements (the seed carries the element kind).
fn bind_tensor_from_flat(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_tensor_from_flat",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              seed: i32,
              flat: i32,
              shape: i32|
              -> Result<i32, wasmtime::Error> {
            let (kind, shape, values) = (|| {
                let state = caller.data();
                let tensor = state.find_tensor(seed).ok_or_else(|| {
                    format!("tensor_from_flat seed {seed}: unknown tensor handle")
                })?;
                let shape = read_index_vector(state, shape)
                    .ok_or_else(|| "tensor_from_flat shape must be a lista of dims".to_owned())?;
                let collection = state
                    .find_collection(flat)
                    .ok_or_else(|| format!("tensor_from_flat flat {flat}: unknown lista handle"))?;
                Ok::<_, String>((tensor.kind, shape, collection.values.clone()))
            })()
            .map_err(|message| typed_unsupported(&mut caller, message))?;
            let expected = tensor_shape_element_count(&shape).ok_or_else(|| {
                typed_unsupported(&mut caller, "tensor_from_flat shape element count overflow")
            })?;
            if values.len() != expected {
                return Err(typed_unsupported(
                    &mut caller,
                    format!(
                        "tensor_from_flat flat length {} != shape element count {expected}",
                        values.len()
                    ),
                ));
            }
            Ok(caller.data_mut().alloc_tensor(TensorValue {
                kind,
                shape,
                values,
            }))
        },
    )?;
    Ok(())
}

/// `tensor_rank (param i32) (result i64)`: receiver → rank (the number of
/// shape dimensions; tensor `longitudo`).
fn bind_tensor_rank(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_tensor_rank",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32|
              -> Result<i64, wasmtime::Error> {
            let state = caller.data();
            let Some(tensor) = state.find_tensor(handle) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("tensor_rank handle {handle}: unknown tensor handle"),
                ));
            };
            Ok(i64::try_from(tensor.shape.len()).expect("rank fits i64"))
        },
    )?;
    Ok(())
}

/// `tensor_shape (param i32) (result i32)`: receiver → shape dims as a
/// `lista` handle (renders `[2, 3]` through the collection display).
fn bind_tensor_shape(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_tensor_shape",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32|
              -> Result<i32, wasmtime::Error> {
            let state = caller.data();
            let Some(tensor) = state.find_tensor(handle) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("tensor_shape handle {handle}: unknown tensor handle"),
                ));
            };
            let dims = tensor
                .shape
                .iter()
                .map(|dim| RuntimeValue::I64(*dim))
                .collect::<Vec<_>>();
            Ok(caller
                .data_mut()
                .alloc_collection(false, VALUE_KIND_I64, dims))
        },
    )?;
    Ok(())
}

/// `tensor_reshape (param i32 i32) (result i32)`: receiver + new shape lista
/// → tensor with the same flat values and the new shape (element count must
/// match).
fn bind_tensor_reshape(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_tensor_reshape",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32,
              shape: i32|
              -> Result<i32, wasmtime::Error> {
            let (kind, values, new_shape) = (|| {
                let state = caller.data();
                let tensor = state.find_tensor(handle).ok_or_else(|| {
                    format!("tensor_reshape handle {handle}: unknown tensor handle")
                })?;
                let new_shape = read_index_vector(state, shape)
                    .ok_or_else(|| "tensor_reshape shape must be a lista of dims".to_owned())?;
                Ok::<_, String>((tensor.kind, tensor.values.clone(), new_shape))
            })()
            .map_err(|message| typed_unsupported(&mut caller, message))?;
            let expected = tensor_shape_element_count(&new_shape).ok_or_else(|| {
                typed_unsupported(&mut caller, "tensor_reshape shape element count overflow")
            })?;
            if values.len() != expected {
                return Err(typed_unsupported(
                    &mut caller,
                    "tensor_reshape element count mismatch",
                ));
            }
            Ok(caller.data_mut().alloc_tensor(TensorValue {
                kind,
                shape: new_shape,
                values,
            }))
        },
    )?;
    Ok(())
}

/// `tensor_get (param i32 i32) (result i32)`: receiver + index lista → option
/// result (present payload or absent, matching the emitter's coalesce paths).
fn bind_tensor_get(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_tensor_get",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32,
              index: i32|
              -> Result<i32, wasmtime::Error> {
            let (kind, shape, values) = {
                let state = caller.data();
                let Some(tensor) = state.find_tensor(handle) else {
                    return Err(typed_unsupported(
                        &mut caller,
                        format!("tensor_get handle {handle}: unknown tensor handle"),
                    ));
                };
                (tensor.kind, tensor.shape.clone(), tensor.values.clone())
            };
            let Some(index) = read_index_vector(caller.data(), index) else {
                return Err(typed_unsupported(
                    &mut caller,
                    "tensor_get index must be a lista of dims",
                ));
            };
            let payload =
                tensor_flat_offset(&shape, &index).and_then(|offset| values.get(offset).copied());
            let result = caller.data_mut().option_result(kind, payload);
            Ok(result)
        },
    )?;
    Ok(())
}

/// `tensor_set (param i32 i32 f64)`: receiver + index lista + value — writes
/// the element in place (no result).
fn bind_tensor_set(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_tensor_set",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32,
              index: i32,
              value: f64|
              -> Result<(), wasmtime::Error> {
            let (kind, shape) = (|| {
                let state = caller.data();
                let tensor = state
                    .find_tensor(handle)
                    .ok_or_else(|| format!("tensor_set handle {handle}: unknown tensor handle"))?;
                Ok::<_, String>((tensor.kind, tensor.shape.clone()))
            })()
            .map_err(|message| typed_unsupported(&mut caller, message))?;
            let index = read_index_vector(caller.data(), index).ok_or_else(|| {
                typed_unsupported(&mut caller, "tensor_set index must be a lista of dims")
            })?;
            let Some(offset) = tensor_flat_offset(&shape, &index) else {
                // Out-of-bounds set mirrors the oracle's no-op (the Rust
                // runtime latches an error; the display rows read the
                // original value).
                return Ok(());
            };
            let converted = tensor_value_from_carrier(
                &TensorValue {
                    kind,
                    shape,
                    values: Vec::new(),
                },
                RuntimeValue::F64(value),
            )
            .map_err(|message| typed_unsupported(&mut caller, message))?;
            let Some(tensor) = caller.data_mut().find_tensor_mut(handle) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("tensor_set handle {handle}: unknown tensor handle"),
                ));
            };
            if let Some(slot) = tensor.values.get_mut(offset) {
                *slot = converted;
            }
            Ok(())
        },
    )?;
    Ok(())
}

/// `tensor_fill (param i32 f64) (result i32)`: receiver + fill value — fills
/// every element and returns the receiver.
fn bind_tensor_fill(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_tensor_fill",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32,
              value: f64|
              -> Result<i32, wasmtime::Error> {
            let (kind, shape, count) = (|| {
                let state = caller.data();
                let tensor = state
                    .find_tensor(handle)
                    .ok_or_else(|| format!("tensor_fill handle {handle}: unknown tensor handle"))?;
                let count = tensor_shape_element_count(&tensor.shape)
                    .ok_or_else(|| "tensor_fill element count overflow".to_owned())?;
                Ok::<_, String>((tensor.kind, tensor.shape.clone(), count))
            })()
            .map_err(|message| typed_unsupported(&mut caller, message))?;
            let converted = tensor_value_from_carrier(
                &TensorValue {
                    kind,
                    shape,
                    values: Vec::new(),
                },
                RuntimeValue::F64(value),
            )
            .map_err(|message| typed_unsupported(&mut caller, message))?;
            let Some(tensor) = caller.data_mut().find_tensor_mut(handle) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("tensor_fill handle {handle}: unknown tensor handle"),
                ));
            };
            tensor.values = vec![converted; count];
            Ok(handle)
        },
    )?;
    Ok(())
}

/// `tensor_flatten (param i32) (result i32)`: receiver → the flat element
/// `lista` handle (renders `[1.0, 4.0, …]` through the collection display).
fn bind_tensor_flatten(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_tensor_flatten",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32|
              -> Result<i32, wasmtime::Error> {
            let (kind, values) = {
                let state = caller.data();
                let Some(tensor) = state.find_tensor(handle) else {
                    return Err(typed_unsupported(
                        &mut caller,
                        format!("tensor_flatten handle {handle}: unknown tensor handle"),
                    ));
                };
                (tensor.kind, tensor.values.clone())
            };
            Ok(caller.data_mut().alloc_collection(false, kind, values))
        },
    )?;
    Ok(())
}

/// `tensor_materialize (param i32) (result i32)`: receiver → an owned copy
/// (the arena tensors are already owned; the row returns the copy).
fn bind_tensor_materialize(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_tensor_materialize",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32|
              -> Result<i32, wasmtime::Error> {
            let state = caller.data();
            let Some(tensor) = state.find_tensor(handle) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("tensor_materialize handle {handle}: unknown tensor handle"),
                ));
            };
            let copy = TensorValue {
                kind: tensor.kind,
                shape: tensor.shape.clone(),
                values: tensor.values.clone(),
            };
            Ok(caller.data_mut().alloc_tensor(copy))
        },
    )?;
    Ok(())
}

/// `tensor_slice (param i32 i64 i64) (result i32)`: receiver + start/end row
/// bounds → the axis-0 slice (`sectio`). Mirrors the LLVM lane's
/// `tensor_slice` row: a contiguous row slice over the first dimension.
fn bind_tensor_slice(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_tensor_slice",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32,
              start: i64,
              end: i64|
              -> Result<i32, wasmtime::Error> {
            let (kind, shape, values) = {
                let state = caller.data();
                let Some(tensor) = state.find_tensor(handle) else {
                    return Err(typed_unsupported(
                        &mut caller,
                        format!("tensor_slice handle {handle}: unknown tensor handle"),
                    ));
                };
                (tensor.kind, tensor.shape.clone(), tensor.values.clone())
            };
            let Some((&first, rest)) = shape.split_first() else {
                return Err(typed_unsupported(
                    &mut caller,
                    "tensor_slice requires rank >= 1",
                ));
            };
            if start < 0 || end < start || end > first {
                return Err(typed_unsupported(
                    &mut caller,
                    "tensor_slice bounds out of range",
                ));
            }
            let row_stride = tensor_shape_element_count(rest).unwrap_or(1);
            let row_len = usize::try_from(row_stride).expect("row stride fits usize");
            let take = usize::try_from(end - start).expect("slice width fits usize");
            let mut sliced = Vec::with_capacity(take * row_len);
            for row in start..end {
                let base = usize::try_from(row).expect("row fits usize") * row_len;
                sliced.extend_from_slice(&values[base..base + row_len]);
            }
            let mut new_shape = Vec::with_capacity(shape.len());
            new_shape.push(end - start);
            new_shape.extend(rest.iter().copied());
            Ok(caller.data_mut().alloc_tensor(TensorValue {
                kind,
                shape: new_shape,
                values: sliced,
            }))
        },
    )?;
    Ok(())
}

/// One elementwise tensor binary op (`addita`/`subtrahe`/`multiplica`).
#[derive(Clone, Copy)]
enum TensorBinaryOp {
    Add,
    Sub,
    Mul,
}

fn tensor_binary_values(
    op: TensorBinaryOp,
    left: &[RuntimeValue],
    right: &[RuntimeValue],
) -> Result<Vec<RuntimeValue>, String> {
    if left.len() != right.len() {
        return Err("tensor binary operand element count mismatch".to_owned());
    }
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| match (op, a, b) {
            (TensorBinaryOp::Add, RuntimeValue::F64(a), RuntimeValue::F64(b)) => {
                Ok(RuntimeValue::F64(a + b))
            }
            (TensorBinaryOp::Sub, RuntimeValue::F64(a), RuntimeValue::F64(b)) => {
                Ok(RuntimeValue::F64(a - b))
            }
            (TensorBinaryOp::Mul, RuntimeValue::F64(a), RuntimeValue::F64(b)) => {
                Ok(RuntimeValue::F64(a * b))
            }
            (TensorBinaryOp::Add, RuntimeValue::I64(a), RuntimeValue::I64(b)) => {
                Ok(RuntimeValue::I64(a.wrapping_add(*b)))
            }
            (TensorBinaryOp::Sub, RuntimeValue::I64(a), RuntimeValue::I64(b)) => {
                Ok(RuntimeValue::I64(a.wrapping_sub(*b)))
            }
            (TensorBinaryOp::Mul, RuntimeValue::I64(a), RuntimeValue::I64(b)) => {
                Ok(RuntimeValue::I64(a.wrapping_mul(*b)))
            }
            _ => Err(format!("tensor binary op on {a:?} / {b:?}")),
        })
        .collect()
}

/// `tensor_add`/`tensor_sub`/`tensor_mul (param i32 i32) (result i32)`:
/// elementwise arithmetic on same-shaped tensors → new tensor.
fn bind_tensor_add_sub_mul(
    linker: &mut Linker<HostState>,
    field: &'static str,
    op: TensorBinaryOp,
) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        field,
        move |mut caller: wasmtime::Caller<'_, HostState>,
              left: i32,
              right: i32|
              -> Result<i32, wasmtime::Error> {
            let (kind, shape, values) = {
                let state = caller.data();
                let Some(tensor) = state.find_tensor(left) else {
                    return Err(typed_unsupported(
                        &mut caller,
                        format!("{field} left {left}: unknown tensor handle"),
                    ));
                };
                (tensor.kind, tensor.shape.clone(), tensor.values.clone())
            };
            let right_values = {
                let state = caller.data();
                let Some(tensor) = state.find_tensor(right) else {
                    return Err(typed_unsupported(
                        &mut caller,
                        format!("{field} right {right}: unknown tensor handle"),
                    ));
                };
                tensor.values.clone()
            };
            let values = tensor_binary_values(op, &values, &right_values)
                .map_err(|message| typed_unsupported(&mut caller, message))?;
            Ok(caller.data_mut().alloc_tensor(TensorValue {
                kind,
                shape,
                values,
            }))
        },
    )?;
    Ok(())
}

/// `tensor_matmul (param i32 i32) (result i32)`: rank-2 matrix multiply
/// (`matmul`). V1 is rank-2 only; the inner dimension must unify.
fn bind_tensor_matmul(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_tensor_matmul",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              left: i32,
              right: i32|
              -> Result<i32, wasmtime::Error> {
            let (kind, shape, values) = {
                let state = caller.data();
                let Some(tensor) = state.find_tensor(left) else {
                    return Err(typed_unsupported(
                        &mut caller,
                        format!("tensor_matmul left {left}: unknown tensor handle"),
                    ));
                };
                (tensor.kind, tensor.shape.clone(), tensor.values.clone())
            };
            let (right_shape, right_values) = {
                let state = caller.data();
                let Some(tensor) = state.find_tensor(right) else {
                    return Err(typed_unsupported(
                        &mut caller,
                        format!("tensor_matmul right {right}: unknown tensor handle"),
                    ));
                };
                (tensor.shape.clone(), tensor.values.clone())
            };
            let [m, k] = shape.as_slice() else {
                return Err(typed_unsupported(
                    &mut caller,
                    "tensor_matmul requires rank-2 receiver",
                ));
            };
            let [k2, n] = right_shape.as_slice() else {
                return Err(typed_unsupported(
                    &mut caller,
                    "tensor_matmul requires rank-2 argument",
                ));
            };
            if k != k2 {
                return Err(typed_unsupported(
                    &mut caller,
                    "tensor_matmul inner dimension mismatch",
                ));
            }
            let (m, k, n) = (
                usize::try_from(*m).map_err(|_| typed_unsupported(&mut caller, "matmul dim"))?,
                usize::try_from(*k).map_err(|_| typed_unsupported(&mut caller, "matmul dim"))?,
                usize::try_from(*n).map_err(|_| typed_unsupported(&mut caller, "matmul dim"))?,
            );
            let mut out = Vec::with_capacity(m * n);
            for row in 0..m {
                for col in 0..n {
                    let mut acc = RuntimeValue::F64(0.0);
                    for inner in 0..k {
                        let a = values[row * k + inner];
                        let b = right_values[inner * n + col];
                        match (a, b) {
                            (RuntimeValue::F64(a), RuntimeValue::F64(b)) => {
                                acc = RuntimeValue::F64(
                                    match acc {
                                        RuntimeValue::F64(current) => current,
                                        _ => 0.0,
                                    } + a * b,
                                );
                            }
                            (RuntimeValue::I64(a), RuntimeValue::I64(b)) => {
                                acc = RuntimeValue::I64(
                                    match acc {
                                        RuntimeValue::I64(current) => current,
                                        _ => 0,
                                    }
                                    .wrapping_add(a.wrapping_mul(b)),
                                );
                            }
                            _ => {
                                return Err(typed_unsupported(
                                    &mut caller,
                                    "tensor_matmul mixed element carriers",
                                ));
                            }
                        }
                    }
                    out.push(acc);
                }
            }
            Ok(caller.data_mut().alloc_tensor(TensorValue {
                kind,
                shape: vec![m as i64, n as i64],
                values: out,
            }))
        },
    )?;
    Ok(())
}

/// `tensor_sum (param i32) (result f64)`: full-reduction sum of all elements.
fn bind_tensor_sum(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_tensor_sum",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32|
              -> Result<f64, wasmtime::Error> {
            let state = caller.data();
            let Some(tensor) = state.find_tensor(handle) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("tensor_sum handle {handle}: unknown tensor handle"),
                ));
            };
            let sum = tensor
                .values
                .iter()
                .try_fold(0.0_f64, |acc, value| match value {
                    RuntimeValue::F64(value) => Some(acc + value),
                    RuntimeValue::I64(value) => Some(acc + *value as f64),
                    RuntimeValue::I32(value) => Some(acc + f64::from(*value)),
                    _ => None,
                });
            sum.ok_or_else(|| {
                typed_unsupported(&mut caller, "tensor_sum element kind not summable to f64")
            })
        },
    )?;
    Ok(())
}

/// `tensor_mean (param i32) (result f64)`: full-reduction mean of all
/// elements.
fn bind_tensor_mean(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_tensor_mean",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32|
              -> Result<f64, wasmtime::Error> {
            let (sum, count) = (|| {
                let state = caller.data();
                let tensor = state
                    .find_tensor(handle)
                    .ok_or_else(|| format!("tensor_mean handle {handle}: unknown tensor handle"))?;
                if tensor.values.is_empty() {
                    return Err("tensor_mean requires at least one element".to_owned());
                }
                let sum = tensor
                    .values
                    .iter()
                    .try_fold(0.0_f64, |acc, value| match value {
                        RuntimeValue::F64(value) => Some(acc + value),
                        RuntimeValue::I64(value) => Some(acc + *value as f64),
                        RuntimeValue::I32(value) => Some(acc + f64::from(*value)),
                        _ => None,
                    });
                let sum =
                    sum.ok_or_else(|| "tensor_mean element kind not summable to f64".to_owned())?;
                Ok((sum, tensor.values.len()))
            })()
            .map_err(|message| typed_unsupported(&mut caller, message))?;
            Ok(sum / count as f64)
        },
    )?;
    Ok(())
}

/// Convert one tensor element from the source kind to the target kind
/// (`tensor_convert` element-width conversions: numerus→fractus etc.).
fn convert_tensor_value(from: u32, to: u32, value: RuntimeValue) -> Option<RuntimeValue> {
    let scalar = match (from, value) {
        (VALUE_KIND_I64 | VALUE_KIND_U64, RuntimeValue::I64(value)) => value as f64,
        (
            VALUE_KIND_I32 | VALUE_KIND_I8 | VALUE_KIND_I16 | VALUE_KIND_U8 | VALUE_KIND_U16
            | VALUE_KIND_U32,
            RuntimeValue::I32(value),
        ) => f64::from(value),
        (VALUE_KIND_F32 | VALUE_KIND_F64, RuntimeValue::F64(value)) => value,
        _ => return None,
    };
    Some(match to {
        VALUE_KIND_F32 | VALUE_KIND_F64 => RuntimeValue::F64(scalar),
        VALUE_KIND_I64 | VALUE_KIND_U64 => RuntimeValue::I64(scalar as i64),
        VALUE_KIND_I32 | VALUE_KIND_I8 | VALUE_KIND_I16 | VALUE_KIND_U8 | VALUE_KIND_U16
        | VALUE_KIND_U32 => RuntimeValue::I32(scalar as i32),
        _ => return None,
    })
}

/// `tensor_convert (param i32 i32 i32) (result i32)`: source + from-kind +
/// to-kind → a tensor with the same shape and the element values converted
/// (`tensor ↦ tensor` element-width conversio).
fn bind_tensor_convert(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_tensor_convert",
        move |mut caller: wasmtime::Caller<'_, HostState>,
              handle: i32,
              from: i32,
              to: i32|
              -> Result<i32, wasmtime::Error> {
            let (shape, values) = {
                let state = caller.data();
                let Some(tensor) = state.find_tensor(handle) else {
                    return Err(typed_unsupported(
                        &mut caller,
                        format!("tensor_convert handle {handle}: unknown tensor handle"),
                    ));
                };
                (tensor.shape.clone(), tensor.values.clone())
            };
            let converted = values
                .iter()
                .map(|value| {
                    convert_tensor_value(from as u32, to as u32, *value).ok_or_else(|| {
                        typed_unsupported(
                            &mut caller,
                            format!("tensor_convert {from}→{to} on {value:?}",),
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(caller.data_mut().alloc_tensor(TensorValue {
                kind: to as u32,
                shape,
                values: converted,
            }))
        },
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// U6-B cursor-stream materialization + cede (yield) channel
// ---------------------------------------------------------------------------

/// The P5 cursor-stream materialization row (`__faber_rt_v1_cursor_stream`).
/// The host invokes the referenced generator (function-id i32 into the
/// module's exported callable table, then the generator's argument carriers)
/// to completion and collects its `cede` yields into a `lista<T>` aggregate
/// handle (reference semantics: the MIR stepper's `eval_cursor_stream`; the
/// generator's own return value is discarded). The binding's signature is
/// derived from the module's declared import (one field, one signature per
/// module — the emitter dedups cursor-stream shapes by argument carriers).
fn bind_cursor_stream(
    linker: &mut Linker<HostState>,
    module: &Module,
) -> Result<(), wasmtime::Error> {
    let mut declared = Vec::<FuncType>::new();
    for import in module.imports() {
        if import.module() != WASM_IMPORT_MODULE_V1 || import.name() != SYMBOL_CURSOR_STREAM {
            continue;
        }
        match import.ty() {
            wasmtime::ExternType::Func(ty) => declared.push(ty),
            other => {
                return Err(wasmtime::Error::msg(format!(
                    "`{SYMBOL_CURSOR_STREAM}` is declared as a non-function import: {other:?}"
                )));
            }
        }
    }
    if declared.is_empty() {
        return Ok(());
    }
    let signature = declared
        .pop()
        .expect("declared cursor-stream imports are non-empty");
    let same_shape = |left: &FuncType, right: &FuncType| {
        left.params()
            .map(|ty| ty.to_string())
            .eq(right.params().map(|ty| ty.to_string()))
            && left
                .results()
                .map(|ty| ty.to_string())
                .eq(right.results().map(|ty| ty.to_string()))
    };
    if declared.iter().any(|other| !same_shape(other, &signature)) {
        return Err(wasmtime::Error::msg(format!(
            "`{SYMBOL_CURSOR_STREAM}` imports declare conflicting signatures under the one \
             v1 field; a module must declare exactly one cursor-stream shape"
        )));
    }
    linker.func_new(
        WASM_IMPORT_MODULE_V1,
        SYMBOL_CURSOR_STREAM,
        signature,
        move |mut caller: wasmtime::Caller<'_, HostState>,
              params: &[Val],
              results: &mut [Val]|
              -> wasmtime::Result<()> {
            let Some(Val::I32(function_id)) = params.first().copied() else {
                return Err(typed_unsupported(
                    &mut caller,
                    "cursor-stream: first parameter must be the generator function-id (i32)",
                ));
            };
            let index = u64::try_from(function_id).map_err(|_| {
                typed_unsupported(
                    &mut caller,
                    format!("cursor-stream: generator function-id {function_id} is negative"),
                )
            })?;
            // Resolve the referenced generator through the module's exported
            // callable table (U6-A mechanism: the emitter exports
            // `faber_callables` whenever a cursor stream is materialized).
            let generator = {
                let Some(export) = caller.get_export(FABER_CALLABLE_TABLE) else {
                    return Err(typed_unsupported(
                        &mut caller,
                        format!(
                            "cursor-stream: module does not export the callable table \
                             `{FABER_CALLABLE_TABLE}`"
                        ),
                    ));
                };
                let Some(table) = export.into_table() else {
                    return Err(typed_unsupported(
                        &mut caller,
                        format!("cursor-stream: `{FABER_CALLABLE_TABLE}` is not a table export"),
                    ));
                };
                match table.get(&mut caller, index) {
                    Some(wasmtime::Ref::Func(Some(func))) => func,
                    Some(wasmtime::Ref::Func(None)) => {
                        return Err(typed_unsupported(
                            &mut caller,
                            format!(
                                "cursor-stream: callable-table entry {function_id} is null \
                                 (the generator is not tabled)"
                            ),
                        ));
                    }
                    None => {
                        return Err(typed_unsupported(
                            &mut caller,
                            format!(
                                "cursor-stream: callable-table entry {function_id} is out of \
                                 bounds"
                            ),
                        ));
                    }
                    Some(other) => {
                        return Err(typed_unsupported(
                            &mut caller,
                            format!(
                                "cursor-stream: callable-table entry {function_id} is not a \
                                 function: {other:?}"
                            ),
                        ));
                    }
                }
            };
            // Materialize: run the generator to completion over a fresh yield
            // buffer. The generator's own return value is discarded, so a
            // result buffer is allocated per the generator's declared results.
            caller.data_mut().cursor_yields.push(CursorYieldBuffer::default());
            let mut discarded = Vec::new();
            for result_ty in generator.ty(&caller).results() {
                let Some(value) = default_wasm_value(&result_ty) else {
                    caller.data_mut().cursor_yields.pop();
                    return Err(typed_unsupported(
                        &mut caller,
                        format!(
                            "cursor-stream: generator result type {result_ty} cannot be \
                             discarded on the v1 cursor surface"
                        ),
                    ));
                };
                discarded.push(value);
            }
            let generator_result = generator.call(&mut caller, &params[1..], &mut discarded);
            // Pop the buffer even on generator failure so stacked
            // materializations stay balanced.
            let buffer = caller.data_mut().cursor_yields.pop().unwrap_or_default();
            generator_result.map_err(|error| {
                wasmtime::Error::msg(format!(
                    "cursor-stream: generator (callable-table entry {function_id}) failed: {error:#}"
                ))
            })?;
            // The materialized `lista<T>`: one collection, element kind fixed
            // by the first yield (an empty materialization has no observable
            // element kind and renders `[]` under the neutral numerus kind).
            let kind = buffer.kind.unwrap_or(VALUE_KIND_I64);
            let handle = caller.data_mut().alloc_collection(false, kind, buffer.values);
            if let Some(result) = results.first_mut() {
                *result = Val::I32(handle);
            }
            Ok(())
        },
    )?;
    Ok(())
}

/// A neutral wasm value for one result carrier (used to discard a
/// generator's own return value during cursor-stream materialization).
fn default_wasm_value(ty: &ValType) -> Option<Val> {
    Some(match ty {
        ValType::I32 => Val::I32(0),
        ValType::I64 => Val::I64(0),
        ValType::F32 => Val::F32(0),
        ValType::F64 => Val::F64(0),
        ValType::V128 => Val::V128(0u128.into()),
        ValType::Ref(_) => return None,
    })
}

/// The `VALUE_KIND_*` element kind for one cede yield carrier. i64 crosses
/// scalar (numerus) yields, f64 crosses fractus; the i32 carrier is a handle
/// carrier with no declared element kind on the v1 cursor surface, so it
/// fails closed (never a guessed kind).
fn cede_carrier_kind(ty: ValType) -> Option<u32> {
    match ty {
        ValType::I64 => Some(VALUE_KIND_I64),
        ValType::F64 => Some(VALUE_KIND_F64),
        _ => None,
    }
}

/// Bind the closed cede (yield) channel fields on the legacy
/// `faber_runtime` module (U6-B admitted exception). Every field is
/// signature-exact per the emitter's carrier grammar. Inside an active
/// cursor-stream materialization the yielded value appends to the active
/// yield buffer; outside one the row is identity (the stepper's non-generator
/// `cede` passthrough).
fn bind_cede_fields(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    for (field, arg, result) in LEGACY_CEDE_ROWS {
        let arg_ty = arg.clone();
        let result_ty = result.clone();
        let arg_kind = cede_carrier_kind(arg_ty.clone());
        linker.func_new(
            LEGACY_CEDE_MODULE,
            field,
            FuncType::new(linker.engine(), [arg_ty.clone()], [result_ty.clone()]),
            move |mut caller: wasmtime::Caller<'_, HostState>,
                  params: &[Val],
                  results: &mut [Val]|
                  -> wasmtime::Result<()> {
                // Handle-carrier (i32) yields: fail typed when a
                // materialization is active (no declared element kind on the
                // v1 surface); outside one the row is identity.
                let Some(kind) = arg_kind else {
                    if caller.data().cursor_yields.last().is_some() {
                        return Err(typed_unsupported(
                            &mut caller,
                            format!(
                                "`{field}`: a handle-carrier (i32) cede yield has no declared \
                                 element kind on the v1 cursor surface; the yield channel is \
                                 bound for i64/f64 carriers (U6-B recorded decision)"
                            ),
                        ));
                    }
                    if let (Some(dest), Some(src)) = (results.first_mut(), params.first()) {
                        *dest = *src;
                    }
                    return Ok(());
                };
                let value = match (&arg_ty, params.first()) {
                    (ValType::I64, Some(Val::I64(value))) => RuntimeValue::I64(*value),
                    (ValType::F64, Some(Val::F64(bits))) => {
                        RuntimeValue::F64(f64::from_bits(*bits))
                    }
                    _ => {
                        return Err(wasmtime::Error::msg(format!(
                            "`{field}`: unexpected cede yield carrier"
                        )));
                    }
                };
                let push_error = {
                    let state = caller.data_mut();
                    match state.cursor_yields.last_mut() {
                        Some(active) => active.push(kind, value),
                        None => Ok(()),
                    }
                };
                if let Err(message) = push_error {
                    return Err(typed_unsupported(&mut caller, message));
                }
                // Identity passthrough: the emitter drops the result in
                // statement position, and outside a materialization `cede`
                // is identity.
                if let (Some(dest), Some(src)) = (results.first_mut(), params.first()) {
                    *dest = *src;
                }
                Ok(())
            },
        )?;
    }
    Ok(())
}
