//! Explicit run configuration for one program execution.

/// Configuration for one Wasm program execution.
///
/// The runner accepts Wasm bytes plus this configuration. It never accepts
/// source text, an interner, WAT, or an opaque-handle table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunConfig {
    /// Name of the exported entry function invoked after instantiation.
    pub entry: String,
    /// Upper bound on captured stdout bytes. Capture stops at this bound so a
    /// misbehaving module cannot grow host memory without limit.
    pub max_stdout_bytes: usize,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            entry: "incipit".to_owned(),
            max_stdout_bytes: 1 << 20,
        }
    }
}
