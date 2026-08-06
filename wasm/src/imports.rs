//! Closed `faber_rt_v1` import registry for the portable product host.
//!
//! Only module `faber_rt_v1` is admitted. Every admitted field is bound with
//! its canonical signature; legacy modules and unbound fields reject during
//! preflight with a typed [`RunOutcome::ImportRejected`]. A known admitted
//! symbol whose behavior is not implemented in this stage produces a typed
//! runtime failure when invoked — never a plausible default (architecture.md:
//! "must not return a plausible default").

use wasmtime::{Linker, Module};

use crate::outcome::RunOutcome;

/// Import module for the closed CPU host ABI v1 surface.
pub const WASM_IMPORT_MODULE_V1: &str = "faber_rt_v1";

/// Admitted v1 field surface for the Stage 2 product host: the scalar and
/// pointer-carrier diagnostic family the predecessor v1 host proved. The
/// registry grows by semantic family in later stages (architecture.md import
/// registry; W4B adds the solum/consolum text surface).
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
/// unsupported for the W15 deny-by-default filesystem reason and mone stays
/// unsupported because this stage's host captures stdout only (W13:
/// declared-but-unimplemented -> typed unsupported, never a silent no-op or a
/// plausible default).
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

/// True when `field` is admitted by the closed v1 registry.
fn is_admitted_field(field: &str) -> bool {
    V1_DIAGNOSTIC_FIELDS.contains(&field)
        || V1_PROVIDER_TEXT_FIELDS.contains(&field)
        || V1_JSON_FIELDS.contains(&field)
}

/// Per-run host state: captured stdout, capture bound, the typed
/// unsupported-symbol record for admitted-but-unfinished behavior, and the
/// W11 text arena of interned literals.
#[derive(Debug)]
pub(crate) struct HostState {
    pub(crate) stdout: String,
    pub(crate) max_stdout_bytes: usize,
    pub(crate) unsupported: Option<String>,
    /// Interned literal payloads (W11). One entry per distinct literal; the
    /// interning dedup map guarantees content-unique entries.
    text_arena: Vec<String>,
    /// Table row -> arena handle. The program's literal operands are table
    /// row indices; this map resolves them to interned arena entries. Empty
    /// when the module declares no literal table.
    row_handles: Vec<i32>,
}

impl HostState {
    pub(crate) fn new(max_stdout_bytes: usize) -> Self {
        Self {
            stdout: String::new(),
            max_stdout_bytes,
            unsupported: None,
            text_arena: Vec::new(),
            row_handles: Vec::new(),
        }
    }

    /// Intern the declared literal-table rows into the text arena and record
    /// the per-row arena handles the program references.
    ///
    /// Interning dedups by content (identical payloads share one arena entry)
    /// while each table row keeps its own handle slot. The emitter already
    /// deduplicates rows by unescaped payload, so row handles normally equal
    /// their row index; the indirection keeps the contract honest if a future
    /// emitter emits duplicate payloads.
    pub(crate) fn intern_literal_table(&mut self, rows: &[String]) {
        let mut content_to_handle = std::collections::HashMap::<&str, i32>::default();
        for row in rows {
            let handle = if let Some(handle) = content_to_handle.get(row.as_str()) {
                *handle
            } else {
                let handle =
                    i32::try_from(self.text_arena.len()).expect("text arena handle fits i32");
                self.text_arena.push(row.clone());
                content_to_handle.insert(row.as_str(), handle);
                handle
            };
            self.row_handles.push(handle);
        }
    }

    /// Resolve a program text-handle operand to its interned literal.
    pub(crate) fn resolve_text(&self, handle: i32) -> Option<&str> {
        let row = usize::try_from(handle).ok()?;
        let arena_handle = *self.row_handles.get(row)?;
        let index = usize::try_from(arena_handle).ok()?;
        self.text_arena.get(index).map(String::as_str)
    }

    /// Append one diagnostic line (terminated by `\n`), bounded by the run
    /// configuration's stdout cap.
    pub(crate) fn write_line(&mut self, text: &str) {
        let mut line = String::with_capacity(text.len() + 1);
        line.push_str(text);
        line.push('\n');
        let remaining = self.max_stdout_bytes.saturating_sub(self.stdout.len());
        if remaining == 0 {
            return;
        }
        let take = line.len().min(remaining);
        self.stdout.push_str(&line[..take]);
    }
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
    // W11: `nota`/`vide` text diagnostics resolve the interned literal (the
    // operand is a literal-table row) and write it to the captured stdout
    // line — the wasm outcome unblocker. `mone` streams to stderr, which this
    // stage's product host does not capture, so it stays a typed unsupported
    // outcome (never a silent redirect to stdout).
    bind_stdout_text(linker, "__faber_rt_v1_diagnostic_nota_ptr", "nota/stdout")?;
    bind_mone_text(linker, "__faber_rt_v1_diagnostic_mone_ptr")?;
    bind_stdout_text(linker, "__faber_rt_v1_diagnostic_vide_ptr", "vide/stdout")?;
    // W4B provider text surface: bound with the exact signatures the radix
    // Wasm emitter emits. Solum reads/writes stay typed unsupported (no fs
    // capability in RunConfig per W15 deny-by-default); consolum
    // scribe/nota_text close-overlap the nota stdout renderer above, and
    // mone_text stays a typed unsupported stderr stream (W13 — never a
    // silent no-op).
    bind_solum_read_handle(linker, "__faber_rt_v1_solum_read_text")?;
    bind_solum_read_handle(linker, "__faber_rt_v1_solum_read_lines")?;
    bind_solum_read_handle(linker, "__faber_rt_v1_solum_read_bytes")?;
    bind_solum_write_text(linker)?;
    bind_stdout_text(linker, "__faber_rt_v1_diagnostic_nota_text", "nota/stdout (scribe)")?;
    bind_mone_text(linker, "__faber_rt_v1_diagnostic_mone_text")?;
    // WE6 json surface: bound with the exact `(param i32) (result i32)`
    // handle signatures the radix Wasm emitter emits. No json host
    // implementation exists in this stage, so invoking one is a typed
    // unsupported outcome (W13) — never a silent no-op.
    bind_json_handle(linker, "__faber_rt_v1_json_pange", "pange")?;
    bind_json_handle(linker, "__faber_rt_v1_json_solve", "solve")?;
    bind_json_handle(linker, "__faber_rt_v1_json_tempta", "tempta")?;
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

/// Stdout-stream text diagnostics (`nota`, `vide`, consolum `scribe`): W11
/// materializes the interned literal — the operand is a literal-table row
/// resolved through the host text arena — into the captured stdout line. An
/// unresolvable handle is a typed runtime failure; the product runner never
/// accepts an externally reconstructed handle table.
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
                None => {
                    caller.data_mut().unsupported = Some(format!(
                        "`{field}` handle {handle}: unknown text handle ({stream} oracle stream \
                         needs an interned literal; Stage 4 literal initialization); the product \
                         runner accepts no external handle table"
                    ));
                    Err(wasmtime::Error::msg(
                        "unsupported v1 text materialization (unknown literal-table handle)",
                    ))
                }
            }
        },
    )?;
    Ok(())
}

/// Mone text (`mone_ptr`/`mone_text`) streams to stderr, which this stage's
/// product host does not capture (stdout only). Invoking one is a typed
/// runtime failure naming the oracle stream — never a silent redirect to
/// stdout and never a plausible default (W13).
fn bind_mone_text(linker: &mut Linker<HostState>, field: &'static str) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        field,
        move |mut caller: wasmtime::Caller<'_, HostState>, handle: i32| -> Result<(), wasmtime::Error> {
            caller.data_mut().unsupported = Some(format!(
                "`{field}` handle {handle}: mone/stderr oracle stream is not captured by this \
                 stage's product host (stdout capture only); typed unsupported until stderr \
                 capture lands — never a silent redirect to stdout"
            ));
            Err(wasmtime::Error::msg(
                "unsupported v1 mone text (admitted symbol, stderr not captured)",
            ))
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
            caller.data_mut().unsupported = Some(format!(
                "`{field}` handle {handle}: solum read requires a filesystem capability \
                 (W15 deny-by-default; the product host RunConfig carries no fs adapter) and \
                 text-handle materialization (Stage 4 literal initialization); \
                 declared-but-unimplemented -> typed unsupported"
            ));
            Err(wasmtime::Error::msg(
                "unsupported v1 solum read (admitted symbol, unfinished behavior)",
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
            caller.data_mut().unsupported = Some(format!(
                "`__faber_rt_v1_solum_write_text` path {path} text {text}: solum write requires \
                 a filesystem capability (W15 deny-by-default; the product host RunConfig carries \
                 no fs adapter) and text-handle materialization (Stage 4 literal initialization); \
                 declared-but-unimplemented -> typed unsupported"
            ));
            Err(wasmtime::Error::msg(
                "unsupported v1 solum write (admitted symbol, unfinished behavior)",
            ))
        },
    )?;
    Ok(())
}

/// The json v1 rows (`pange`/`solve`/`tempta`) are admitted-but-unimplemented
/// in this stage: no json host implementation exists (a later stage lands the
/// W13 json host impl), and the emitted operands are opaque handles the runner
/// cannot materialize without linear-memory literal data (Stage 4 literal
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
            caller.data_mut().unsupported = Some(format!(
                "`{field}` handle {handle}: json {verb} requires a json host implementation \
                 (admitted symbol, unfinished behavior; typed unsupported until the json host \
                 impl lands) and handle materialization (Stage 4 literal initialization); \
                 declared-but-unimplemented -> typed unsupported"
            ));
            Err(wasmtime::Error::msg(
                "unsupported v1 json (admitted symbol, unfinished behavior)",
            ))
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
