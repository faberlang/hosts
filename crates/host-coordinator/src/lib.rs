//! Multi-device coordinator surface for Faber hosts (HOSTS-COORD).
//!
//! Moved from `faber-runtime/src/` (faber-target-runtime inventory §3.2 /
//! DDPP0 row 7): device identity, device-set topology, discovery, bound
//! plans, backend capability results, virtual partitions, priority policy,
//! typed transport, execution transactions, and the physical device-handle
//! split (`DeviceHandle`/`DeviceHandleKind`).
//!
//! Ownership boundary: these are **device-lifecycle facts** — never generated
//! language values, never runtime-contract types. The faber/runtime/rust
//! contract package stays free of device/session behavior (C2 isolation).
//! The build/selection surface (`DeviceSelection`, `from_spelling`, backend
//! selection metadata) is RADIX-ARTIFACT+FABER-BUILD and stays with the
//! compiler/product; this crate carries the physical `backend` discriminator
//! only ([`crate::backend::DeviceBackend`]).

pub mod backend;
pub mod bound_plan;
pub mod capability;
pub mod device_handle;
pub mod device_identity;
pub mod device_set;
pub mod discovery;
pub mod execution_transaction;
pub mod partition;
pub mod policy;
pub mod transport;

pub use backend::DeviceBackend;
pub use device_handle::{DeviceHandle, DeviceHandleKind};

// `capability` has no in-module test decl in the source layout; its test
// module is wired at the crate root (as in faber-runtime/src/lib.rs). All
// other modules self-declare their `#[path = "..."] mod tests;`.
#[cfg(test)]
#[path = "capability_test.rs"]
mod capability_test;
