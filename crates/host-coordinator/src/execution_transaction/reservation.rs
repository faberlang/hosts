//! Reservation derivation (S3): the per-partition byte reservation `prepare`
//! derives from the accepted plan and charges against the admitted ledger
//! class budgets (class 6 transfer staging / class 3 activation scratch).
//! Section split out of `execution_transaction.rs` (polish).

use crate::bound_plan::LogicalPartitionId;
use crate::execution_transaction::mirror::{OutputRef, StagedWrite, TransactionOperation};
use std::collections::{BTreeMap, BTreeSet};

/// Declared per-operation event-object byte cost, charged to class 3
/// (`activation_scratch_bytes`). Each operation reserves one event object per
/// involved partition.
pub const EVENT_OBJECT_BYTES: u64 = 64;

/// Declared per-partition transaction-scratch byte cost, charged to class 3
/// (`activation_scratch_bytes`). Every participating partition reserves this
/// coordinator scratch for the transaction's lifetime.
pub const TRANSACTION_SCRATCH_BYTES_PER_PARTITION: u64 = 4096;

/// Which admitted ledger class a reservation field is charged against
/// (md0-closeout §3.2 item 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetClass {
    /// Ledger class 6 — `PartitionBudgetLedger::transfer_staging_bytes`.
    TransferStaging,
    /// Ledger class 3 — `PartitionBudgetLedger::activation_scratch_bytes`.
    ActivationScratch,
}

impl BudgetClass {
    /// The ledger class number (6 or 3).
    #[must_use]
    pub const fn ledger_class(self) -> u64 {
        match self {
            Self::TransferStaging => 6,
            Self::ActivationScratch => 3,
        }
    }
}

/// The per-partition reservation `prepare` derives from the accepted plan
/// (S3). Staging charges class 6 (`transfer_staging_bytes`); output buffers,
/// events, and transaction scratch charge class 3 (`activation_scratch_bytes`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReservationRecord {
    /// Class 6: typed transfer staging buffers (in-flight copies at full
    /// size).
    transfer_staging_bytes: u64,
    /// Class 3: staged output buffer bytes.
    output_buffer_bytes: u64,
    /// Class 3: event objects.
    event_bytes: u64,
    /// Class 3: transaction scratch.
    transaction_scratch_bytes: u64,
}

impl ReservationRecord {
    /// Build a reservation record from its four charged fields.
    #[must_use]
    pub const fn new(
        transfer_staging_bytes: u64,
        output_buffer_bytes: u64,
        event_bytes: u64,
        transaction_scratch_bytes: u64,
    ) -> Self {
        Self {
            transfer_staging_bytes,
            output_buffer_bytes,
            event_bytes,
            transaction_scratch_bytes,
        }
    }

    /// The class 6 staging reservation.
    #[must_use]
    pub const fn transfer_staging_bytes(&self) -> u64 {
        self.transfer_staging_bytes
    }

    /// The class 3 output-buffer reservation.
    #[must_use]
    pub const fn output_buffer_bytes(&self) -> u64 {
        self.output_buffer_bytes
    }

    /// The class 3 event reservation.
    #[must_use]
    pub const fn event_bytes(&self) -> u64 {
        self.event_bytes
    }

    /// The class 3 transaction-scratch reservation.
    #[must_use]
    pub const fn transaction_scratch_bytes(&self) -> u64 {
        self.transaction_scratch_bytes
    }

    /// The class 6 charge (transfer staging).
    #[must_use]
    pub const fn class_six_bytes(&self) -> u64 {
        self.transfer_staging_bytes
    }

    /// The class 3 charge (output buffers + events + transaction scratch).
    #[must_use]
    pub const fn class_three_bytes(&self) -> u64 {
        self.output_buffer_bytes + self.event_bytes + self.transaction_scratch_bytes
    }
}

/// Deterministic reservation derivation over the accepted plan (S3).
///
/// Accounting rules (charged to the partition that holds the resource):
///
/// - A transfer reserves full `byte_count` staging at **both** endpoints
///   (the host-staged move stages both halves, class 6) and the destination's
///   staged output write (class 3).
/// - A broadcast reserves `byte_count` staging at **every** participant (the
///   value leaves the source and enters each destination through host
///   staging, class 6) and `byte_count` output at every non-source
///   participant (class 3).
/// - A launch reserves its declared `output_bytes` at its partition (class 3).
/// - Every operation reserves [`EVENT_OBJECT_BYTES`] per involved partition
///   and every participating partition reserves
///   [`TRANSACTION_SCRATCH_BYTES_PER_PARTITION`] (class 3).
pub(super) fn derive_reservation(
    operations: &[TransactionOperation],
    referenced: &BTreeSet<LogicalPartitionId>,
) -> BTreeMap<LogicalPartitionId, ReservationRecord> {
    let mut staging: BTreeMap<LogicalPartitionId, u64> = BTreeMap::new();
    let mut outputs: BTreeMap<LogicalPartitionId, u64> = BTreeMap::new();
    let mut events: BTreeMap<LogicalPartitionId, u64> = BTreeMap::new();
    for operation in operations {
        for partition in operation.partitions() {
            *events.entry(partition).or_insert(0) += EVENT_OBJECT_BYTES;
        }
        match operation {
            TransactionOperation::Launch {
                partition,
                output_bytes,
                ..
            } => {
                *outputs.entry(partition.clone()).or_insert(0) += *output_bytes;
            }
            TransactionOperation::Transfer(transfer) => {
                let count = transfer.byte_count();
                *staging.entry(transfer.source().clone()).or_insert(0) += count;
                *staging.entry(transfer.destination().clone()).or_insert(0) += count;
                *outputs.entry(transfer.destination().clone()).or_insert(0) += count;
            }
            TransactionOperation::CollectiveBroadcast(broadcast) => {
                let count = broadcast.byte_count();
                for participant in broadcast.participants() {
                    *staging.entry(participant.clone()).or_insert(0) += count;
                    if participant != broadcast.source() {
                        *outputs.entry(participant.clone()).or_insert(0) += count;
                    }
                }
            }
            TransactionOperation::Barrier { .. } => {}
        }
    }
    referenced
        .iter()
        .map(|partition| {
            let record = ReservationRecord::new(
                staging.get(partition).copied().unwrap_or(0),
                outputs.get(partition).copied().unwrap_or(0),
                events.get(partition).copied().unwrap_or(0),
                TRANSACTION_SCRATCH_BYTES_PER_PARTITION,
            );
            (partition.clone(), record)
        })
        .collect()
}

/// The declared write-set of the accepted plan — the set of writes the
/// transaction will publish, derived deterministically at prepare (one
/// `OutputRef` per producing operation).
pub(super) fn derive_declared_write_set(
    operations: &[TransactionOperation],
) -> BTreeMap<OutputRef, StagedWrite> {
    let mut writes = BTreeMap::new();
    for operation in operations {
        for write in operation.staged_writes() {
            writes.insert(write.output_ref().clone(), write);
        }
    }
    writes
}
