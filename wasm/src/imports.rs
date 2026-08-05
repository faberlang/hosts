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
/// handle table, and text materialization requires linear-memory literal data
/// (Stage 4 literal initialization). Invoking one is a typed unsupported
/// outcome (W13: declared-but-unimplemented -> typed unsupported, never a
/// silent no-op or a plausible default).
pub(crate) const V1_PROVIDER_TEXT_FIELDS: &[&str] = &[
    "__faber_rt_v1_solum_read_text",
    "__faber_rt_v1_solum_read_lines",
    "__faber_rt_v1_solum_read_bytes",
    "__faber_rt_v1_solum_write_text",
    "__faber_rt_v1_diagnostic_nota_text",
    "__faber_rt_v1_diagnostic_mone_text",
];

/// True when `field` is admitted by the closed v1 registry.
fn is_admitted_field(field: &str) -> bool {
    V1_DIAGNOSTIC_FIELDS.contains(&field) || V1_PROVIDER_TEXT_FIELDS.contains(&field)
}

/// Per-run host state: captured stdout, capture bound, and the typed
/// unsupported-symbol record for admitted-but-unfinished behavior.
#[derive(Debug)]
pub(crate) struct HostState {
    pub(crate) stdout: String,
    pub(crate) max_stdout_bytes: usize,
    pub(crate) unsupported: Option<String>,
}

impl HostState {
    pub(crate) fn new(max_stdout_bytes: usize) -> Self {
        Self {
            stdout: String::new(),
            max_stdout_bytes,
            unsupported: None,
        }
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
    bind_text_handle(linker, "__faber_rt_v1_diagnostic_nota_ptr")?;
    bind_text_handle(linker, "__faber_rt_v1_diagnostic_mone_ptr")?;
    bind_text_handle(linker, "__faber_rt_v1_diagnostic_vide_ptr")?;
    // W4B provider text surface: bound with the exact signatures the radix
    // Wasm emitter emits. No host implementation exists in this stage (no fs
    // capability in RunConfig per W15 deny-by-default; text handles cannot be
    // materialized without linear-memory literal data), so invoking one is a
    // typed unsupported outcome — never a silent no-op (W13).
    bind_solum_read_handle(linker, "__faber_rt_v1_solum_read_text")?;
    bind_solum_read_handle(linker, "__faber_rt_v1_solum_read_lines")?;
    bind_solum_read_handle(linker, "__faber_rt_v1_solum_read_bytes")?;
    bind_solum_write_text(linker)?;
    bind_consolum_diagnostic_text(
        linker,
        "__faber_rt_v1_diagnostic_nota_text",
        "scribe -> nota/stdout",
    )?;
    bind_consolum_diagnostic_text(
        linker,
        "__faber_rt_v1_diagnostic_mone_text",
        "mone -> stderr",
    )?;
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

/// Text/aggregate handle carriers are admitted symbols whose behavior is not
/// implemented until Stage 4 literal initialization puts text in module
/// memory. Invoking one is a typed runtime failure — the product runner never
/// accepts an externally reconstructed handle table.
fn bind_text_handle(linker: &mut Linker<HostState>, field: &'static str) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        field,
        move |mut caller: wasmtime::Caller<'_, HostState>, handle: i32| -> Result<(), wasmtime::Error> {
            caller.data_mut().unsupported = Some(format!(
                "`{field}` handle {handle}: text materialization requires linear-memory \
                 literal data (Stage 4 literal initialization); the product runner accepts \
                 no external handle table"
            ));
            Err(wasmtime::Error::msg(
                "unsupported v1 text materialization (admitted symbol, unfinished behavior)",
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

/// Consolum scribe/mone close-overlap the v1 diagnostic text rows. The
/// emitted operand is an opaque text handle and the runner never accepts an
/// external handle table, so the oracle stream semantics (`stream` — scribe
/// -> nota/stdout, mone -> stderr) cannot be realized until linear-memory
/// literal initialization lands. Invoking one is a typed runtime failure.
fn bind_consolum_diagnostic_text(
    linker: &mut Linker<HostState>,
    field: &'static str,
    stream: &'static str,
) -> Result<(), wasmtime::Error> {
    linker.func_wrap(
        WASM_IMPORT_MODULE_V1,
        field,
        move |mut caller: wasmtime::Caller<'_, HostState>, handle: i32| -> Result<(), wasmtime::Error> {
            caller.data_mut().unsupported = Some(format!(
                "`{field}` handle {handle}: consolum diagnostic text materialization requires \
                 linear-memory literal data (Stage 4 literal initialization); the product runner \
                 accepts no external handle table ({stream} oracle stream parity lands with it); \
                 declared-but-unimplemented -> typed unsupported"
            ));
            Err(wasmtime::Error::msg(
                "unsupported v1 consolum diagnostic text (admitted symbol, unfinished behavior)",
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
