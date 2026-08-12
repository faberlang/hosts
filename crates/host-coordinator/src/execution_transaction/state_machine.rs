//! The deterministic transaction state machine (`New → Prepared → Executing →
//! Committed | Aborted`, with `Failed → Aborted`) and its identity vocabulary:
//! the transaction id, the abstract publication ordinal, and the recorded
//! failure. Section split out of `execution_transaction.rs` (polish).

use crate::execution_transaction::backend::BackendError;

/// Machine-local opaque identity of one transaction.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TransactionId(String);

impl TransactionId {
    /// Build a transaction id from its stable string.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The stable identity string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TransactionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The **abstract publication ordinal** — the transaction-scoped publication
/// counter a committed transaction records in its receipt. This is the
/// "abstract execution generation ordinal" of the MD3 spec: a transaction-
/// scoped publication ordinal, **never the semantic `ValueGeneration`**
/// (naming contract §3). Minted by the coordinator that owns the publication
/// counter; `commit` records it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PublicationOrdinal(u64);

impl PublicationOrdinal {
    /// Build a publication ordinal from a publication-counter value.
    #[must_use]
    pub const fn new(ordinal: u64) -> Self {
        Self(ordinal)
    }

    /// The raw ordinal value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for PublicationOrdinal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "pub:{}", self.0)
    }
}

/// The deterministic state machine: `New → Prepared → Executing →
/// Committed | Aborted`, with `Failed → Aborted`. Retry is disabled — there
/// is no re-execution path once the machine leaves `Prepared`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransactionState {
    /// Constructed; nothing reserved.
    New,
    /// `prepare` succeeded: the reservation is recorded and held.
    Prepared,
    /// `execute` is running the accepted plan (or finished dispatching it,
    /// awaiting the boundary).
    Executing,
    /// `commit` published the staged write-set atomically.
    Committed,
    /// An operation or the publication failed; `abort` completes teardown.
    Failed(TransactionFailure),
    /// `abort` completed teardown; no publication happened.
    Aborted(TransactionFailure),
}

/// The recorded failure of the transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransactionFailure {
    /// A backend operation or reservation failed.
    Backend(BackendError),
    /// The atomic publication failed after the boundary was reached.
    PublishFailed {
        /// What failed, as reported by the backend.
        detail: String,
    },
    /// The transaction was cancelled.
    Cancelled {
        /// Why it was cancelled.
        reason: String,
    },
}
