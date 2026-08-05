//! Typed run outcomes for the portable product host.
//!
//! The campaign contract (architecture.md lifecycle) requires stable
//! categories that preserve the validation / import / link / initialization /
//! entry / trap / runtime-failure / policy-denial / exit distinctions. Every
//! run returns exactly one [`RunOutcome`]; the runner never fails with a
//! generic error at the run boundary.

/// Stable outcome category for one run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutcomeCategory {
    /// Entry completed; captured stdout is available.
    Success,
    /// Module bytes failed Wasm validation.
    ValidationFailed,
    /// Import surface rejected during preflight: legacy module, unknown
    /// field, or a declared field the registry does not admit.
    ImportRejected,
    /// Linking or instantiation failed after import admission (for example a
    /// declared signature incompatible with the admitted binding).
    LinkFailed,
    /// Generated initialization failed (reserved until an initialization
    /// surface exists in the emit profile).
    InitializationFailed,
    /// The configured entry export is missing.
    EntryMissing,
    /// The entry export trapped during invocation.
    EntryTrapped,
    /// Host-side runtime failure (an admitted-but-unfinished symbol was
    /// invoked, policy denied a capability, or an internal error occurred).
    RuntimeFailure,
    /// Capability policy denied the run (reserved until the capability
    /// layer lands in a later stage).
    PolicyDenied,
    /// The program exited with an explicit status (reserved until the exit
    /// surface exists in the emit profile).
    Exited,
}

/// Full typed outcome of one run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunOutcome {
    /// Entry completed; `stdout` holds the captured program output.
    Success {
        stdout: String,
    },
    /// Module bytes failed Wasm validation.
    ValidationFailed {
        message: String,
    },
    /// Import surface rejected during preflight.
    ImportRejected {
        module: String,
        field: String,
        message: String,
    },
    /// Linking or instantiation failed after import admission.
    LinkFailed {
        message: String,
    },
    /// Generated initialization failed.
    InitializationFailed {
        message: String,
    },
    /// The configured entry export is missing.
    EntryMissing {
        entry: String,
    },
    /// The entry export trapped during invocation.
    EntryTrapped {
        entry: String,
        message: String,
    },
    /// Host-side runtime failure.
    RuntimeFailure {
        message: String,
    },
    /// Capability policy denied the run.
    PolicyDenied {
        message: String,
    },
    /// The program exited with an explicit status.
    Exited {
        code: i32,
        stdout: String,
    },
}

impl RunOutcome {
    /// Stable category for this outcome.
    #[must_use]
    pub fn category(&self) -> OutcomeCategory {
        match self {
            Self::Success { .. } => OutcomeCategory::Success,
            Self::ValidationFailed { .. } => OutcomeCategory::ValidationFailed,
            Self::ImportRejected { .. } => OutcomeCategory::ImportRejected,
            Self::LinkFailed { .. } => OutcomeCategory::LinkFailed,
            Self::InitializationFailed { .. } => OutcomeCategory::InitializationFailed,
            Self::EntryMissing { .. } => OutcomeCategory::EntryMissing,
            Self::EntryTrapped { .. } => OutcomeCategory::EntryTrapped,
            Self::RuntimeFailure { .. } => OutcomeCategory::RuntimeFailure,
            Self::PolicyDenied { .. } => OutcomeCategory::PolicyDenied,
            Self::Exited { .. } => OutcomeCategory::Exited,
        }
    }

    /// Captured stdout when the outcome carries captured output, else empty.
    #[must_use]
    pub fn stdout(&self) -> &str {
        match self {
            Self::Success { stdout } | Self::Exited { stdout, .. } => stdout,
            Self::ValidationFailed { .. }
            | Self::ImportRejected { .. }
            | Self::LinkFailed { .. }
            | Self::InitializationFailed { .. }
            | Self::EntryMissing { .. }
            | Self::EntryTrapped { .. }
            | Self::RuntimeFailure { .. }
            | Self::PolicyDenied { .. } => "",
        }
    }
}
