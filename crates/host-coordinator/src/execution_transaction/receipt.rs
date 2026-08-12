//! The transaction receipt types: the commit/abort decision, the executed-
//! operation records, the publication/teardown/timing summaries, and the
//! assembled `TransactionReceipt`. Section split out of
//! `execution_transaction.rs` (polish).

use crate::bound_plan::LogicalPartitionId;
use crate::execution_transaction::mirror::{
    OperationEvent, OutputRef, StagedWrite, TransactionOperation,
};
use crate::execution_transaction::reservation::ReservationRecord;
use crate::execution_transaction::state_machine::{
    PublicationOrdinal, TransactionFailure, TransactionId,
};
use crate::partition::PartitionReceipt;
use crate::transport::TransportReceipt;
use std::collections::{BTreeMap, BTreeSet};

/// The commit/abort decision of the transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransactionDecision {
    /// The staged write-set was published atomically.
    Committed {
        /// The abstract publication ordinal (transaction-scoped).
        publication_ordinal: PublicationOrdinal,
    },
    /// The transaction was aborted; nothing was published.
    Aborted {
        /// The recorded failure (the cancel reason or the originating
        /// failure).
        failure: TransactionFailure,
    },
}

/// One executed operation of the snapshot, with its exact byte contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutedOperationRecord {
    /// The operation as dispatched.
    pub operation: TransactionOperation,
    /// The operation's exact byte contract (bytes moved across the mesh).
    pub byte_count: u64,
}

/// The atomic-publication summary recorded in a commit receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PublishSummary {
    /// Total bytes staged before publication.
    pub staged_bytes: u64,
    /// Total bytes published.
    pub published_bytes: u64,
    /// Always true when present — publication is all-or-nothing.
    pub atomic: bool,
}

/// Teardown facts: which partitions were released and which were retired.
/// `partial_publication` is an invariant — always false.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TeardownFacts {
    /// Partitions whose reservations were released (commit, or abort without
    /// staged state).
    pub released_partitions: BTreeSet<LogicalPartitionId>,
    /// Partitions whose staged state was retired on abort/failure.
    pub retired_partitions: BTreeSet<LogicalPartitionId>,
    /// Invariant: a partial publication never happens.
    pub partial_publication: bool,
}

/// Wall-clock phase timings of the transaction (runtime evidence; not
/// canonical).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TransactionTimings {
    /// `prepare` elapsed, nanoseconds.
    pub prepare_nanos: u64,
    /// `execute` elapsed, nanoseconds.
    pub execute_nanos: u64,
    /// `commit`/`abort` elapsed, nanoseconds.
    pub finalize_nanos: u64,
}

/// The base `TransactionReceipt` (exit-gate bullet 6; S4): transaction id,
/// both plan hashes, the device/virtual identities from the bound plan, the
/// per-partition reservation summary, the declared staged write-set, the
/// executed operations with exact bytes, the synchronization events, the
/// commit/abort decision + reason, the publication summary, teardown facts,
/// phase timings, and the S4 selected-transport section (the actual selected
/// transports: copy path/staging/events/timeout/bytes/timing + budget
/// accounting — the folded [`TransportReceipt`], CTO sanity-check amendment;
/// `None` when no transport adapter recorded transfers for this transaction).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransactionReceipt {
    /// The transaction identity.
    pub transaction_id: TransactionId,
    /// The admitted logical plan hash (from the bound plan; never re-derived).
    pub logical_distributed_plan_hash: String,
    /// The bound-plan hash (from the bound plan).
    pub bound_distributed_plan_hash: String,
    /// Device/virtual identities from the bound plan (`hardware_isolation_claimed=false`).
    pub plan_receipt: PartitionReceipt,
    /// Per-partition reservation summary (the prepare receipt).
    pub reservation_summary: BTreeMap<LogicalPartitionId, ReservationRecord>,
    /// The declared staged write-set (reserved at prepare).
    pub declared_write_set: BTreeMap<OutputRef, StagedWrite>,
    /// The executed operations in plan order, with exact bytes.
    pub executed_operations: Vec<ExecutedOperationRecord>,
    /// The synchronization events completed when the transaction finalized.
    pub synchronization_events: BTreeSet<OperationEvent>,
    /// The commit/abort decision + reason.
    pub decision: TransactionDecision,
    /// The atomic-publication summary (`None` when nothing was published).
    pub publish_summary: Option<PublishSummary>,
    /// Teardown facts.
    pub teardown: TeardownFacts,
    /// Phase timings.
    pub timings: TransactionTimings,
    /// The S4 selected-transport section folded from the transport adapter
    /// used during the transaction (path/staging/events/timeout/bytes/timing
    /// + budget accounting at the measured rates). `None` when no transport
    /// adapter recorded transfers for this transaction.
    pub selected_transports: Option<TransportReceipt>,
}
