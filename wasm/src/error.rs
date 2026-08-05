//! Infra-level host errors.
//!
//! These cover host construction (engine setup), which is the one fallible
//! boundary outside a per-run [`crate::outcome::RunOutcome`]. Every run
//! failure is a typed outcome instead.

use std::fmt;

/// Host construction or engine-level error.
#[derive(Debug)]
pub struct HostError {
    message: String,
}

impl HostError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(f)
    }
}

impl std::error::Error for HostError {}

impl From<wasmtime::Error> for HostError {
    fn from(error: wasmtime::Error) -> Self {
        Self::new(format!("{error:#}"))
    }
}
