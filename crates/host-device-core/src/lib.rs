//! Backend-neutral device primitives shared by native host products.
//!
//! This crate owns the descriptor vocabulary, opaque-handle lifecycle
//! bookkeeping, host error/frame-data carriers, and the common device-session
//! byte surface. Product crates retain their concrete runtime and driver
//! implementations.

pub mod device_descriptor;
pub mod device_registry;
pub mod device_session;
pub mod kernel;

pub use device_session::DeviceSession;
pub use kernel::{HostError, HostResult};
