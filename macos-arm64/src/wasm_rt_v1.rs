//! Plain-module Wasm host for the shared CPU ABI v1 import surface.
//!
//! This is the product proof path for `wasm-host-parity` Track B2. It is
//! intentionally separate from the legacy `capability-call` WasmHost proof.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use wasmtime::{Config, Engine, Linker, Module, Store};

use crate::wasm::WasmHostError;

/// Import module for the closed CPU host ABI v1 surface.
pub const WASM_IMPORT_MODULE_V1: &str = "faber_rt_v1";

type HostResult<T> = Result<T, WasmHostError>;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WasmRtV1RunResult {
    pub stdout: String,
    pub success: bool,
}

struct RtV1State {
    text_handles: BTreeMap<i32, String>,
    stdout: String,
}

/// Host that loads plain core Wasm and resolves only `faber_rt_v1` imports.
pub struct WasmRtV1Host {
    engine: Engine,
}

impl WasmRtV1Host {
    pub fn new() -> HostResult<Self> {
        let config = Config::new();
        Ok(Self {
            engine: Engine::new(&config)?,
        })
    }

    pub fn run_file(
        &self,
        path: impl AsRef<Path>,
        entry: &str,
        text_handles: BTreeMap<i32, String>,
    ) -> HostResult<WasmRtV1RunResult> {
        let bytes = fs::read(path)?;
        self.run_module(&bytes, entry, text_handles)
    }

    /// Run a WAT or Wasm binary module. `text_handles` maps interner/symbol
    /// handles (as emitted `i32.const` text handles) to UTF-8 text for
    /// `*_diagnostic_nota_ptr` and similar ptr carriers.
    pub fn run_module(
        &self,
        module_bytes: &[u8],
        entry: &str,
        text_handles: BTreeMap<i32, String>,
    ) -> HostResult<WasmRtV1RunResult> {
        let module = Module::new(&self.engine, module_bytes)?;
        let mut store = Store::new(
            &self.engine,
            RtV1State {
                text_handles,
                stdout: String::new(),
            },
        );
        let mut linker: Linker<RtV1State> = Linker::new(&self.engine);
        link_declared_v1_imports(&mut linker, &module)?;
        let instance = linker
            .instantiate(&mut store, &module)
            .map_err(|error| host_error(format!("instantiate failed: {error:#}")))?;
        let func = instance
            .get_func(&mut store, entry)
            .ok_or_else(|| host_error(format!("module export not found: {entry}")))?;
        func.call(&mut store, &[], &mut [])
            .map_err(|error| host_error(format!("export `{entry}` trapped: {error:#}")))?;
        Ok(WasmRtV1RunResult {
            stdout: store.into_data().stdout,
            success: true,
        })
    }
}

fn host_error(message: impl Into<String>) -> WasmHostError {
    WasmHostError::new(message)
}

fn link_declared_v1_imports(linker: &mut Linker<RtV1State>, module: &Module) -> HostResult<()> {
    for import in module.imports() {
        let module_name = import.module();
        let field = import.name();
        if module_name != WASM_IMPORT_MODULE_V1 {
            return Err(host_error(format!(
                "unsupported import module `{module_name}` (product host accepts only `{WASM_IMPORT_MODULE_V1}`)"
            )));
        }
        if !field.starts_with("__faber_rt_v1_") {
            return Err(host_error(format!(
                "unsupported import field `{field}` (expected complete __faber_rt_v1_* symbol)"
            )));
        }
        match field {
            "__faber_rt_v1_diagnostic_nota_ptr"
            | "__faber_rt_v1_diagnostic_mone_ptr"
            | "__faber_rt_v1_diagnostic_vide_ptr" => {
                let field_owned = field.to_owned();
                linker
                    .func_wrap(
                        WASM_IMPORT_MODULE_V1,
                        field,
                        move |mut caller: wasmtime::Caller<'_, RtV1State>, handle: i32| {
                            nota_ptr(&mut caller, handle)?;
                            Ok(())
                        },
                    )
                    .map_err(|error| {
                        host_error(format!("link `{field_owned}` failed: {error:#}"))
                    })?;
            }
            "__faber_rt_v1_diagnostic_nota_i64"
            | "__faber_rt_v1_diagnostic_mone_i64"
            | "__faber_rt_v1_diagnostic_vide_i64" => {
                let field_owned = field.to_owned();
                linker
                    .func_wrap(
                        WASM_IMPORT_MODULE_V1,
                        field,
                        move |mut caller: wasmtime::Caller<'_, RtV1State>, value: i64| {
                            write_line(&mut caller, &value.to_string());
                            Ok(())
                        },
                    )
                    .map_err(|error| {
                        host_error(format!("link `{field_owned}` failed: {error:#}"))
                    })?;
            }
            "__faber_rt_v1_diagnostic_nota_i32"
            | "__faber_rt_v1_diagnostic_nota_i8"
            | "__faber_rt_v1_diagnostic_nota_i1" => {
                let field_owned = field.to_owned();
                linker
                    .func_wrap(
                        WASM_IMPORT_MODULE_V1,
                        field,
                        move |mut caller: wasmtime::Caller<'_, RtV1State>, value: i32| {
                            write_line(&mut caller, &value.to_string());
                            Ok(())
                        },
                    )
                    .map_err(|error| {
                        host_error(format!("link `{field_owned}` failed: {error:#}"))
                    })?;
            }
            "__faber_rt_v1_diagnostic_nota_f64" | "__faber_rt_v1_diagnostic_nota_f32" => {
                let field_owned = field.to_owned();
                linker
                    .func_wrap(
                        WASM_IMPORT_MODULE_V1,
                        field,
                        move |mut caller: wasmtime::Caller<'_, RtV1State>, value: f64| {
                            write_line(&mut caller, &format_float(value));
                            Ok(())
                        },
                    )
                    .map_err(|error| {
                        host_error(format!("link `{field_owned}` failed: {error:#}"))
                    })?;
            }
            other => {
                return Err(host_error(format!(
                    "unsupported v1 host import `{other}` (not bound in first product proof host)"
                )));
            }
        }
    }
    Ok(())
}

fn nota_ptr(
    caller: &mut wasmtime::Caller<'_, RtV1State>,
    handle: i32,
) -> Result<(), wasmtime::Error> {
    let text = caller
        .data()
        .text_handles
        .get(&handle)
        .cloned()
        .ok_or_else(|| {
            wasmtime::Error::msg(format!(
                "unresolved text handle {handle} for diagnostic_nota_ptr (host string table missing entry)"
            ))
        })?;
    write_line(caller, &text);
    Ok(())
}

fn write_line(caller: &mut wasmtime::Caller<'_, RtV1State>, text: &str) {
    let stdout = &mut caller.data_mut().stdout;
    stdout.push_str(text);
    stdout.push('\n');
}

fn format_float(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{}", value as i64)
    } else {
        value.to_string()
    }
}
