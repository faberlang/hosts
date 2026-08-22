//! Backend-neutral host errors and frame-data helpers.

pub mod frame_data;

mod error;

pub use error::{HostError, HostResult};
