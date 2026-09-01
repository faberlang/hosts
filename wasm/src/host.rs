//! Portable plain-module Wasm host resolving the closed `faber_rt_v1` import
//! surface.
//!
//! The runner consumes only Wasm bytes plus an explicit [`RunConfig`]; it
//! never receives source, an interner, WAT, or an opaque-handle table. Every
//! run returns a typed [`RunOutcome`].

use wasmtime::{Config, Engine, Instance, Linker, Module, Store};

use crate::config::RunConfig;
use crate::error::HostError;
use crate::imports::{
    HostState, bind_external_imports, link_v1_imports, preflight_imports, preflight_package_imports,
};
use crate::outcome::RunOutcome;

/// Portable v1 product host. One engine is shared across runs; per-run state
/// lives in the store.
pub struct WasmRtV1Host {
    engine: Engine,
}

impl WasmRtV1Host {
    /// Create a host with a fresh engine.
    ///
    /// # Errors
    ///
    /// Returns [`HostError`] if the Wasmtime engine cannot be created.
    pub fn new() -> Result<Self, HostError> {
        let config = Config::new();
        Ok(Self {
            engine: Engine::new(&config)?,
        })
    }

    /// Run a core-Wasm module with explicit configuration.
    ///
    /// The run lifecycle mirrors architecture.md: validate, preflight the
    /// import surface, link and instantiate, look up the entry, invoke it,
    /// and return a typed outcome. The captured stdout is available on
    /// success; a trap still classifies the outcome honestly.
    #[must_use]
    pub fn run(&self, module_bytes: &[u8], config: &RunConfig) -> RunOutcome {
        let module = match Module::new(&self.engine, module_bytes) {
            Ok(module) => module,
            Err(error) => {
                return RunOutcome::ValidationFailed {
                    message: format!("module validation failed: {error:#}"),
                };
            }
        };

        if let Err(outcome) = preflight_imports(&module) {
            return outcome;
        }

        let mut store = Store::new(&self.engine, HostState::new(config.max_stdout_bytes));
        let mut linker = Linker::new(&self.engine);
        if let Err(error) = link_v1_imports(&mut linker, &module) {
            return RunOutcome::LinkFailed {
                message: format!("link failed: {error:#}"),
            };
        }
        let instance = match linker.instantiate(&mut store, &module) {
            Ok(instance) => instance,
            Err(error) => {
                return RunOutcome::LinkFailed {
                    message: format!("instantiate failed: {error:#}"),
                };
            }
        };

        // W11 generated initialization: read the declared literal table from
        // linear memory and intern each literal into the host arena. A
        // malformed or missing declaration fails initialization — entry never
        // runs (architecture lifecycle: host calls generated init exactly
        // once, before invoking the entry).
        if let Err(outcome) = crate::literal_table::initialize_literal_table(&instance, &mut store)
        {
            return outcome;
        }

        let entry = config.entry.clone();
        let Some(func) = instance.get_func(&mut store, &entry) else {
            return RunOutcome::EntryMissing { entry };
        };

        match func.call(&mut store, &[], &mut []) {
            Ok(()) => {
                let state = store.into_data();
                RunOutcome::Success {
                    stdout: state.stdout,
                    stderr: state.stderr,
                }
            }
            Err(error) => {
                let state = store.into_data();
                if let Some(message) = state.unsupported {
                    return RunOutcome::RuntimeFailure { message };
                }
                RunOutcome::EntryTrapped {
                    entry: entry.clone(),
                    message: format!("export `{entry}` trapped: {error:#}"),
                }
            }
        }
    }

    /// Run a package (U6-E): instantiate a set of modules together and
    /// resolve the package-namespace external imports — the `faber_external`
    /// surface the radix package-aware emitter
    /// (`emit_wasm_text_probe_package_aware`) produces for same-package
    /// cross-module identities (`importa:auxilium:saluta`) — against the
    /// sibling modules' canonical external-symbol exports
    /// (`__faber_external_product_…_module_…_func_…`).
    ///
    /// `entry` is the package root unit, whose `config.entry` export is
    /// invoked after instantiation. `siblings` are the helper units in
    /// dependency-first order; each sibling's own `faber_external` imports
    /// resolve against the siblings instantiated before it, and the entry's
    /// against the whole sibling set. Every module keeps the closed
    /// `faber_rt_v1` preflight (the U6-B cede exception included);
    /// `faber_external` is admitted only on this path — single-module
    /// [`Self::run`] behavior is unchanged, so a `faber_external` import
    /// still rejects there (a package import is never a host symbol).
    ///
    /// Typed buckets preserved on this path: `MissingImport` →
    /// [`RunOutcome::ImportRejected`] (preflight: an external import no
    /// sibling exports), `NoEntryExport` → [`RunOutcome::EntryMissing`],
    /// `EntryTrap` → [`RunOutcome::EntryTrapped`], and `LinkFailed` →
    /// [`RunOutcome::LinkFailed`] (a provider not yet instantiated or a
    /// declared-signature conflict).
    #[must_use]
    pub fn run_package(&self, entry: &[u8], siblings: &[&[u8]], config: &RunConfig) -> RunOutcome {
        let entry_module = match Module::new(&self.engine, entry) {
            Ok(module) => module,
            Err(error) => {
                return RunOutcome::ValidationFailed {
                    message: format!("entry module validation failed: {error:#}"),
                };
            }
        };
        let mut parsed_siblings = Vec::with_capacity(siblings.len());
        for (index, bytes) in siblings.iter().enumerate() {
            match Module::new(&self.engine, *bytes) {
                Ok(module) => parsed_siblings.push(module),
                Err(error) => {
                    return RunOutcome::ValidationFailed {
                        message: format!("sibling module {index} validation failed: {error:#}"),
                    };
                }
            }
        }

        if let Err(outcome) = preflight_package_imports(&entry_module, &parsed_siblings) {
            return outcome;
        }

        let mut store = Store::new(&self.engine, HostState::new(config.max_stdout_bytes));
        // One linker for the whole package. The closed v1 surface is re-bound
        // per module (the cursor-stream binding is per-module), so shadowing
        // is enabled on this fresh per-run linker; each sibling instance's
        // canonical external exports are then defined as `faber_external`
        // fields as they come online.
        let mut linker = Linker::new(&self.engine);
        linker.allow_shadowing(true);
        let mut providers: Vec<Instance> = Vec::with_capacity(parsed_siblings.len());
        for module in &parsed_siblings {
            if let Err(error) = link_v1_imports(&mut linker, module) {
                return RunOutcome::LinkFailed {
                    message: format!("link failed: {error:#}"),
                };
            }
            if let Err(error) = bind_external_imports(&mut linker, &mut store, module, &providers) {
                return RunOutcome::LinkFailed {
                    message: format!("link failed: {error:#}"),
                };
            }
            let instance = match linker.instantiate(&mut store, module) {
                Ok(instance) => instance,
                Err(error) => {
                    return RunOutcome::LinkFailed {
                        message: format!("instantiate failed: {error:#}"),
                    };
                }
            };
            // W11 generated initialization for the sibling's own literal
            // table: a helper that renders its literals needs them interned
            // before the entry can call it.
            if let Err(outcome) =
                crate::literal_table::initialize_literal_table(&instance, &mut store)
            {
                return outcome;
            }
            providers.push(instance);
        }
        // The entry instantiates last, its external imports resolved against
        // the sibling instances.
        if let Err(error) = link_v1_imports(&mut linker, &entry_module) {
            return RunOutcome::LinkFailed {
                message: format!("link failed: {error:#}"),
            };
        }
        if let Err(error) =
            bind_external_imports(&mut linker, &mut store, &entry_module, &providers)
        {
            return RunOutcome::LinkFailed {
                message: format!("link failed: {error:#}"),
            };
        }
        let instance = match linker.instantiate(&mut store, &entry_module) {
            Ok(instance) => instance,
            Err(error) => {
                return RunOutcome::LinkFailed {
                    message: format!("instantiate failed: {error:#}"),
                };
            }
        };
        if let Err(outcome) = crate::literal_table::initialize_literal_table(&instance, &mut store)
        {
            return outcome;
        }

        let entry_name = config.entry.clone();
        let Some(func) = instance.get_func(&mut store, &entry_name) else {
            return RunOutcome::EntryMissing { entry: entry_name };
        };

        match func.call(&mut store, &[], &mut []) {
            Ok(()) => {
                let state = store.into_data();
                RunOutcome::Success {
                    stdout: state.stdout,
                    stderr: state.stderr,
                }
            }
            Err(error) => {
                let state = store.into_data();
                if let Some(message) = state.unsupported {
                    return RunOutcome::RuntimeFailure { message };
                }
                RunOutcome::EntryTrapped {
                    entry: entry_name.clone(),
                    message: format!("export `{entry_name}` trapped: {error:#}"),
                }
            }
        }
    }
}
