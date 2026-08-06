//! Portable core-Wasm product host for the closed `faber_rt_v1` surface.
//!
//! This crate is the single v1 product-host owner for the wasm-host-parity
//! campaign (Stage 2). It runs plain core-Wasm modules against the closed
//! `faber_rt_v1` import surface and returns typed outcomes. The runner
//! consumes only Wasm bytes plus an explicit [`RunConfig`] — never source,
//! an interner, WAT, or an externally reconstructed opaque-handle table.
//!
//! Exempla's product-runner adapter calls [`WasmRtV1Host::run`] directly and
//! maps results to ledger outcomes; later Faber packaging reuses the same API.

mod collections;
pub mod config;
pub mod error;
pub mod host;
pub mod imports;
mod literal_table;
pub mod outcome;

pub use config::RunConfig;
pub use error::HostError;
pub use host::WasmRtV1Host;
pub use imports::WASM_IMPORT_MODULE_V1;
pub use outcome::{OutcomeCategory, RunOutcome};
