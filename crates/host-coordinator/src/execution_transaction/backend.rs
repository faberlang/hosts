//! The device-execution abstraction the transaction drives (MD3-X1): the
//! `DeviceExecutionBackend` trait, its error vocabulary, and the minimal
//! happy-path fake over the T2 §5 fixture. Section split out of
//! `execution_transaction.rs` (polish).

use crate::bound_plan::LogicalPartitionId;
use crate::execution_transaction::mirror::{
    OperationEvent, OutputRef, StagedWrite, TransactionOperation,
};
use crate::execution_transaction::reservation::ReservationRecord;
use crate::execution_transaction::state_machine::TransactionFailure;
use std::collections::{BTreeMap, BTreeSet};

/// A backend failure. The fault classes mirror the MD3-F1 suite vocabulary
/// (cancel / timeout / transfer or kernel error / device loss / allocation);
/// the coordinator aborts on any backend failure with no partial publication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    /// A reserve/allocate request could not be satisfied (physical pressure).
    Allocation {
        /// The affected partition.
        partition: LogicalPartitionId,
        /// What failed, as reported by the runtime.
        detail: String,
    },
    /// An operation (transfer / kernel / barrier) failed.
    Operation {
        /// The affected partition.
        partition: LogicalPartitionId,
        /// What failed, as reported by the runtime.
        detail: String,
    },
    /// The bound physical device failed or was removed (MD-A13).
    DeviceLoss {
        /// The lost partition's bound device, via its partition.
        partition: LogicalPartitionId,
        /// What happened, as reported by the runtime.
        detail: String,
    },
    /// The operation was cancelled before completion.
    Cancelled {
        /// The affected partition.
        partition: LogicalPartitionId,
        /// Why it was cancelled.
        detail: String,
    },
    /// The operation timed out.
    Timeout {
        /// The affected partition.
        partition: LogicalPartitionId,
        /// The timeout fact, as reported by the runtime.
        detail: String,
    },
}

impl BackendError {
    /// An allocation failure for one partition.
    #[must_use]
    pub fn allocation(partition: LogicalPartitionId, detail: impl Into<String>) -> Self {
        Self::Allocation {
            partition,
            detail: detail.into(),
        }
    }

    /// An operation failure (transfer / kernel / barrier) for one partition.
    #[must_use]
    pub fn operation(partition: LogicalPartitionId, detail: impl Into<String>) -> Self {
        Self::Operation {
            partition,
            detail: detail.into(),
        }
    }

    /// A device-loss failure (MD-A13) for one partition.
    #[must_use]
    pub fn device_loss(partition: LogicalPartitionId, detail: impl Into<String>) -> Self {
        Self::DeviceLoss {
            partition,
            detail: detail.into(),
        }
    }

    /// A cancellation for one partition.
    #[must_use]
    pub fn cancelled(partition: LogicalPartitionId, detail: impl Into<String>) -> Self {
        Self::Cancelled {
            partition,
            detail: detail.into(),
        }
    }

    /// A timeout for one partition.
    #[must_use]
    pub fn timeout(partition: LogicalPartitionId, detail: impl Into<String>) -> Self {
        Self::Timeout {
            partition,
            detail: detail.into(),
        }
    }

    /// The partition the failure names.
    #[must_use]
    pub fn partition(&self) -> &LogicalPartitionId {
        match self {
            Self::Allocation { partition, .. }
            | Self::Operation { partition, .. }
            | Self::DeviceLoss { partition, .. }
            | Self::Cancelled { partition, .. }
            | Self::Timeout { partition, .. } => partition,
        }
    }
}

/// The device-execution abstraction the transaction drives (MD3-X1; the real
/// implementation over MD1-H1's `DeviceRuntimeSet` is MD3-S1, QUEUED).
///
/// The transaction dispatches each snapshot operation via
/// [`DeviceExecutionBackend::run_operation`], stages the declared writes via
/// [`DeviceExecutionBackend::stage_write`], gates publication on the declared
/// boundary through [`DeviceExecutionBackend::event_completed`], publishes
/// the staged write-set atomically via [`DeviceExecutionBackend::publish`],
/// and tears down via [`DeviceExecutionBackend::release`] /
/// [`DeviceExecutionBackend::retire`].
pub trait DeviceExecutionBackend {
    /// Reserve the declared transaction resources for one partition.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the runtime cannot hold the reservation.
    fn reserve(
        &mut self,
        partition: &LogicalPartitionId,
        reservation: &ReservationRecord,
    ) -> Result<(), BackendError>;

    /// Run one operation of the accepted plan.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the runtime cannot run the operation.
    fn run_operation(&mut self, operation: &TransactionOperation) -> Result<(), BackendError>;

    /// Whether a previously dispatched operation's event has completed
    /// (asynchronous join).
    fn event_completed(&self, event: &OperationEvent) -> bool;

    /// Stage one declared write into the staged write-set. Staging the same
    /// write twice is an error (the write-set is declared once).
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the write is already staged or the
    /// runtime cannot stage it.
    fn stage_write(&mut self, write: &StagedWrite) -> Result<(), BackendError>;

    /// Publish every staged write atomically — all or nothing.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when atomic publication fails; nothing is
    /// published.
    fn publish(&mut self) -> Result<(), BackendError>;

    /// The total bytes currently staged.
    fn staged_bytes(&self) -> u64;

    /// The total bytes published by the last publish.
    fn published_bytes(&self) -> u64;

    /// Release the reservation held for one partition (commit path).
    fn release(&mut self, partition: &LogicalPartitionId);

    /// Retire a partition's staged state after a failure (no publication).
    fn retire(&mut self, partition: &LogicalPartitionId, failure: &TransactionFailure);
}

/// Minimal happy-path `DeviceExecutionBackend` driving the T2 §5 virtual
/// fixture. It records reservations, executes operations (completing their
/// events synchronously, or on demand when auto-complete is off), stages and
/// publishes the write-set atomically, and tracks byte accounting. MD3-F1's
/// fault-injecting backend wraps this fake; MD3-S1 implements the real
/// backend over `DeviceRuntimeSet`.
#[derive(Debug, Clone)]
pub struct FakeExecutionBackend {
    reservations: BTreeMap<LogicalPartitionId, ReservationRecord>,
    staged_writes: BTreeMap<OutputRef, StagedWrite>,
    published_writes: BTreeMap<OutputRef, StagedWrite>,
    staged_total_bytes: u64,
    published_total_bytes: u64,
    published_once: bool,
    executed_operations: Vec<TransactionOperation>,
    dispatched_events: BTreeSet<OperationEvent>,
    completed_events: BTreeSet<OperationEvent>,
    released_partitions: BTreeSet<LogicalPartitionId>,
    retired_partitions: BTreeSet<LogicalPartitionId>,
    auto_complete: bool,
}

impl FakeExecutionBackend {
    /// A fresh happy-path fake; events complete synchronously.
    #[must_use]
    pub fn new() -> Self {
        Self {
            reservations: BTreeMap::new(),
            staged_writes: BTreeMap::new(),
            published_writes: BTreeMap::new(),
            staged_total_bytes: 0,
            published_total_bytes: 0,
            published_once: false,
            executed_operations: Vec::new(),
            dispatched_events: BTreeSet::new(),
            completed_events: BTreeSet::new(),
            released_partitions: BTreeSet::new(),
            retired_partitions: BTreeSet::new(),
            auto_complete: true,
        }
    }

    /// When on, every dispatched operation completes its events
    /// synchronously. When off, events stay pending until
    /// [`Self::complete_event`] — the asynchronous-join simulation the
    /// boundary gate test drives.
    pub fn set_auto_complete(&mut self, on: bool) {
        self.auto_complete = on;
    }

    /// Mark a dispatched event completed (asynchronous join). Lenient: an
    /// unknown event is recorded as completed anyway.
    pub fn complete_event(&mut self, event: OperationEvent) {
        self.dispatched_events.insert(event.clone());
        self.completed_events.insert(event);
    }

    /// The held reservations.
    #[must_use]
    pub fn reservations(&self) -> &BTreeMap<LogicalPartitionId, ReservationRecord> {
        &self.reservations
    }

    /// The staged write-set.
    #[must_use]
    pub fn staged_writes(&self) -> &BTreeMap<OutputRef, StagedWrite> {
        &self.staged_writes
    }

    /// The published write-set (populated by `publish`).
    #[must_use]
    pub fn published_writes(&self) -> &BTreeMap<OutputRef, StagedWrite> {
        &self.published_writes
    }

    /// The operations dispatched, in dispatch order.
    #[must_use]
    pub fn executed_operations(&self) -> &[TransactionOperation] {
        &self.executed_operations
    }

    /// The completed events.
    #[must_use]
    pub fn completed_events(&self) -> &BTreeSet<OperationEvent> {
        &self.completed_events
    }

    /// The dispatched-but-not-yet-completed events.
    #[must_use]
    pub fn pending_events(&self) -> BTreeSet<OperationEvent> {
        self.dispatched_events
            .difference(&self.completed_events)
            .cloned()
            .collect()
    }

    /// The partitions whose reservations were released.
    #[must_use]
    pub fn released_partitions(&self) -> &BTreeSet<LogicalPartitionId> {
        &self.released_partitions
    }

    /// The partitions whose staged state was retired.
    #[must_use]
    pub fn retired_partitions(&self) -> &BTreeSet<LogicalPartitionId> {
        &self.retired_partitions
    }
}

impl Default for FakeExecutionBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceExecutionBackend for FakeExecutionBackend {
    fn reserve(
        &mut self,
        partition: &LogicalPartitionId,
        reservation: &ReservationRecord,
    ) -> Result<(), BackendError> {
        if self.reservations.contains_key(partition) {
            return Err(BackendError::Allocation {
                partition: partition.clone(),
                detail: format!("partition {partition} already holds a reservation"),
            });
        }
        self.reservations.insert(partition.clone(), *reservation);
        Ok(())
    }

    fn run_operation(&mut self, operation: &TransactionOperation) -> Result<(), BackendError> {
        let events = operation.completed_events();
        self.executed_operations.push(operation.clone());
        for event in events {
            self.dispatched_events.insert(event.clone());
            if self.auto_complete {
                self.completed_events.insert(event);
            }
        }
        Ok(())
    }

    fn event_completed(&self, event: &OperationEvent) -> bool {
        self.completed_events.contains(event)
    }

    fn stage_write(&mut self, write: &StagedWrite) -> Result<(), BackendError> {
        if self.staged_writes.contains_key(write.output_ref()) {
            return Err(BackendError::Allocation {
                partition: write.partition().clone(),
                detail: format!("output {} staged twice", write.output_ref()),
            });
        }
        self.staged_total_bytes = self.staged_total_bytes.saturating_add(write.byte_count());
        self.staged_writes
            .insert(write.output_ref().clone(), write.clone());
        Ok(())
    }

    fn publish(&mut self) -> Result<(), BackendError> {
        if self.published_once {
            return Err(BackendError::Operation {
                partition: LogicalPartitionId::new("unknown"),
                detail: "publish called twice — publication is one-shot".to_owned(),
            });
        }
        // Atomic all-or-nothing: promote the whole staged write-set.
        self.published_writes = self.staged_writes.clone();
        self.published_total_bytes = self.staged_total_bytes;
        self.published_once = true;
        Ok(())
    }

    fn staged_bytes(&self) -> u64 {
        self.staged_total_bytes
    }

    fn published_bytes(&self) -> u64 {
        self.published_total_bytes
    }

    fn release(&mut self, partition: &LogicalPartitionId) {
        self.reservations.remove(partition);
        self.released_partitions.insert(partition.clone());
    }

    fn retire(&mut self, partition: &LogicalPartitionId, _failure: &TransactionFailure) {
        let retired: u64 = self
            .staged_writes
            .values()
            .filter(|write| write.partition() == partition)
            .map(super::mirror::StagedWrite::byte_count)
            .sum();
        self.staged_total_bytes = self.staged_total_bytes.saturating_sub(retired);
        self.staged_writes
            .retain(|_, write| write.partition() != partition);
        self.reservations.remove(partition);
        self.retired_partitions.insert(partition.clone());
    }
}
