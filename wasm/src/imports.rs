//! Closed `faber_rt_v1` import registry for the portable product host.
//!
//! Only module `faber_rt_v1` is admitted. Every admitted field is bound with
//! its canonical signature; legacy modules and unbound fields reject during
//! preflight with a typed [`RunOutcome::ImportRejected`]. A known admitted
//! symbol whose behavior is not implemented in this stage produces a typed
//! runtime failure when invoked — never a plausible default (architecture.md:
//! "must not return a plausible default").

use crate::collections::{
    display_fractus, runtime_value_eq, value_from_i64, value_to_i64, CollectionValue, MapValue,
    OptionValue, RuntimeValue,
};
use crate::outcome::RunOutcome;
use radix_host_abi::{
    VALUE_KIND_ASCII, VALUE_KIND_F32, VALUE_KIND_F64, VALUE_KIND_I1, VALUE_KIND_I32,
    VALUE_KIND_I64, VALUE_KIND_PTR, VALUE_KIND_TEXT, VALUE_KIND_U64, VALUE_KIND_U8, VALUE_KIND_VALOR,
};
use std::collections::{BTreeMap, HashMap};
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
    "__faber_rt_v1_diagnostic_nota_i1",
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

    /// Resolve a diagnostic handle to its rendered line: text rows and
    /// dynamic texts render as-is, regex handles render their pattern,
    /// octeti handles render the byte-list Debug shape, and the W13
    /// collection/scalar arenas render in the Rust-oracle Debug shapes
    /// (`[1, 2, 3]` / `["prima", "secunda"]` / `{1, 2}` /
    /// `Json(Tabula({...}))` / payload-or-`nihil`) — mirroring the LLVM
    /// host's opaque display.
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
        // Null-encoded option: handle 0 is `nihil`.
        if handle == 0 {
            return Some("nihil".to_owned());
        }
        None
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

    /// Render an opaque handle for collection-element/option display: text
    /// renders plain, collections/maps/options render recursively.
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

// ---------------------------------------------------------------------------
// W13 collection/scalar display rows
// ---------------------------------------------------------------------------

/// `array_new (param i32) (result i32)`: kind → empty `lista`.
fn bind_array_new(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_array_new",
        move |mut caller: wasmtime::Caller<'_, HostState>, kind: i32| -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>, handle: i32, value: i64| -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>, destination: i32, source: i32| -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>, handle: i32| -> Result<i64, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>, handle: i32, index: i64| -> Result<i64, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>, handle: i32, key: i64| -> Result<i32, wasmtime::Error> {
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
                    let payload =
                        bytes.get(index).copied().map(|b| RuntimeValue::I32(b as i32));
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
        move |mut caller: wasmtime::Caller<'_, HostState>, handle: i32, value: i64| -> Result<i32, wasmtime::Error> {
            let state = caller.data();
            if let Some(collection) = state.find_collection(handle) {
                let Some(value) = value_from_i64(collection.kind, value) else {
                    return Err(typed_unsupported(
                        &mut caller,
                        format!("array_contains handle {handle}: unknown element kind"),
                    ));
                };
                return Ok(i32::from(
                    collection.values.iter().any(|v| runtime_value_eq(*v, value)),
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
        move |mut caller: wasmtime::Caller<'_, HostState>, handle: i32| -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>, handle: i32| -> Result<(), wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>, handle: i32| -> Result<i32, wasmtime::Error> {
            let kind = caller.data().find_collection(handle).map(|c| c.kind);
            let Some(kind) = kind else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("array_sort handle {handle}: unknown collection handle"),
                ));
            };
            let Some(collection) = caller.data_mut().find_collection_mut(handle) else {
                return Err(typed_unsupported(
                    &mut caller,
                    format!("array_sort handle {handle}: unknown collection handle"),
                ));
            };
            collection.values.sort_by_key(|value| value_to_i64(*value));
            let _ = kind;
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
        move |mut caller: wasmtime::Caller<'_, HostState>, handle: i32| -> Result<i64, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>, key_kind: i32, value_kind: i32| -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>, handle: i32, key: i32, value: i64| -> Result<(), wasmtime::Error> {
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
            let key_value = value_from_i64(key_kind, i64::from(key)).unwrap_or(RuntimeValue::Handle(key));
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
        move |mut caller: wasmtime::Caller<'_, HostState>, handle: i32, key: i64| -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>, handle: i32| -> Result<i32, wasmtime::Error> {
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
            Ok(caller.data_mut().alloc_collection(false, VALUE_KIND_TEXT, keys))
        },
    )?;
    Ok(())
}

/// `map_values (param i32) (result i32)`: a `lista` of the map's values.
fn bind_map_values(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_map_values",
        move |mut caller: wasmtime::Caller<'_, HostState>, handle: i32| -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>, kind: i32| -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>, handle: i32| -> Result<i32, wasmtime::Error> {
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
                if !unique.iter().any(|existing| runtime_value_eq(*existing, *value)) {
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
        move |mut caller: wasmtime::Caller<'_, HostState>, handle: i32| -> Result<i32, wasmtime::Error> {
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
    let values = reduce(left, right).map_err(|()| {
        typed_unsupported(caller, "set algebra element-kind mismatch")
    })?;
    Ok(caller.data_mut().alloc_collection(true, kind, values))
}

/// `set_union (param i32 i32) (result i32)`.
fn bind_set_union(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_set_union",
        move |mut caller: wasmtime::Caller<'_, HostState>, left: i32, right: i32| -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>, left: i32, right: i32| -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>, left: i32, right: i32| -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>, left: i32, right: i32| -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>, left: i32, right: i32| -> Result<i32, wasmtime::Error> {
            let state = caller.data();
            let (Some(left), Some(right)) = (state.find_collection(left), state.find_collection(right))
            else {
                return Err(typed_unsupported(
                    &mut caller,
                    "set_is_subset received an unknown collection handle",
                ));
            };
            Ok(i32::from(
                left.values
                    .iter()
                    .all(|value| right.values.iter().any(|v| runtime_value_eq(*v, *value))),
            ))
        },
    )?;
    Ok(())
}

/// `set_is_superset (param i32 i32) (result i32)`.
fn bind_set_is_superset(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        "__faber_rt_v1_set_is_superset",
        move |mut caller: wasmtime::Caller<'_, HostState>, left: i32, right: i32| -> Result<i32, wasmtime::Error> {
            let state = caller.data();
            let (Some(left), Some(right)) = (state.find_collection(left), state.find_collection(right))
            else {
                return Err(typed_unsupported(
                    &mut caller,
                    "set_is_superset received an unknown collection handle",
                ));
            };
            Ok(i32::from(
                right
                    .values
                    .iter()
                    .all(|value| left.values.iter().any(|v| runtime_value_eq(*v, *value))),
            ))
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
        move |mut caller: wasmtime::Caller<'_, HostState>, kind: i32| -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>, value: i64, kind: i32| -> Result<i32, wasmtime::Error> {
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
        move |caller: wasmtime::Caller<'_, HostState>, handle: i32| -> Result<i64, wasmtime::Error> {
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
        move |caller: wasmtime::Caller<'_, HostState>, handle: i32, fallback: i64| -> Result<i64, wasmtime::Error> {
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
        move |caller: wasmtime::Caller<'_, HostState>, handle: i32| -> Result<i32, wasmtime::Error> {
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
fn bind_scalar_i1(linker: &mut Linker<HostState>, field: &'static str) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        field,
        move |mut caller: wasmtime::Caller<'_, HostState>, value: i32| {
            caller.data_mut().write_line(if value != 0 {
                "verum"
            } else {
                "falsum"
            });
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
        move |mut caller: wasmtime::Caller<'_, HostState>, condition: i32| -> Result<(), wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>, condition: i32, message: i32| -> Result<(), wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>, value: i64| -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>, value: f64| -> Result<i32, wasmtime::Error> {
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
        move |mut caller: wasmtime::Caller<'_, HostState>, value: i32| -> Result<i32, wasmtime::Error> {
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
fn bind_text_truthy(linker: &mut Linker<HostState>, field: &'static str) -> Result<(), wasmtime::Error> {
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
        move |caller: wasmtime::Caller<'_, HostState>, text: i32, fallback: i64| -> Result<i64, wasmtime::Error> {
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
        move |caller: wasmtime::Caller<'_, HostState>, text: i32, fallback: f64| -> Result<f64, wasmtime::Error> {
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
