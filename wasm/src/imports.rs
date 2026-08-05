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

/// Admitted v1 field surface for the Stage 2 product host. This is the
/// diagnostic family the predecessor v1 host proved; the registry grows by
/// semantic family in later stages (architecture.md import registry).
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
        if !V1_DIAGNOSTIC_FIELDS.contains(&field) {
            return Err(RunOutcome::ImportRejected {
                module: module_name.to_owned(),
                field: field.to_owned(),
                message: format!(
                    "unknown `{WASM_IMPORT_MODULE_V1}` field `{field}` \
                     (not admitted by the Stage 2 product registry)"
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

fn format_float(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{}", value as i64)
    } else {
        value.to_string()
    }
}
