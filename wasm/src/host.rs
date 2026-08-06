//! Portable plain-module Wasm host resolving the closed `faber_rt_v1` import
//! surface.
//!
//! The runner consumes only Wasm bytes plus an explicit [`RunConfig`]; it
//! never receives source, an interner, WAT, or an opaque-handle table. Every
//! run returns a typed [`RunOutcome`].

use wasmtime::{Config, Engine, Linker, Module, Store};

use crate::config::RunConfig;
use crate::error::HostError;
use crate::imports::{link_v1_imports, preflight_imports, HostState};
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
        if let Err(error) = link_v1_imports(&mut linker) {
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
}
