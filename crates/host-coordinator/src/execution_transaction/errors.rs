//! The typed error vocabulary of construction, prepare, execute, commit, and
//! abort. Section split out of `execution_transaction.rs` (polish).

use crate::bound_plan::LogicalPartitionId;
use crate::execution_transaction::backend::BackendError;
use crate::execution_transaction::mirror::{BarrierRef, BoundaryRef, LaunchRef, OperationRef};
use crate::execution_transaction::reservation::BudgetClass;
use crate::execution_transaction::state_machine::TransactionState;
use std::collections::BTreeSet;

/// Why a transaction could not be constructed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstructError {
    /// The bound plan is the MD-A15 single-partition degenerate — one-device
    /// execution stays coordinator-free (no `ExecutionTransaction`).
    DegeneratePlan,
    /// The snapshot is empty — a distributed plan declares at least one
    /// operation.
    EmptySnapshot,
}

/// Why `prepare` rejected the accepted plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrepareError {
    /// The transaction is not in the `New` state.
    InvalidState {
        /// The current state.
        state: TransactionState,
    },
    /// An operation references a partition that is not declared by the bound
    /// plan (topology authority).
    UnknownPartition {
        /// The undeclared partition.
        partition: LogicalPartitionId,
        /// The index of the offending operation in the snapshot.
        operation_index: usize,
    },
    /// The snapshot contains two operations with the same stable identity.
    DuplicateOperation {
        /// The duplicated operation reference.
        operation_ref: OperationRef,
    },
    /// The declared boundary names a barrier the snapshot does not contain.
    UndeclaredBoundaryBarrier {
        /// The undeclared barrier.
        barrier: BarrierRef,
    },
    /// The declared boundary names a launch the snapshot does not contain.
    UndeclaredBoundaryLaunch {
        /// The undeclared launch.
        launch: LaunchRef,
    },
    /// A referenced partition's binding carries no admitted virtual partition,
    /// so there is no admitted budget the reservation can be checked against.
    MissingAdmittedBudget {
        /// The partition with no admitted budget.
        partition: LogicalPartitionId,
    },
    /// The derived reservation exceeds the partition's admitted budget for
    /// one ledger class (S3) — the focused over-commit diagnostic
    /// (MD2-C1 residual 2 closure).
    ReservationExceedsBudget {
        /// The over-committed partition.
        partition: LogicalPartitionId,
        /// The ledger class that was exceeded.
        class: BudgetClass,
        /// The derived reservation for that class.
        declared_bytes: u64,
        /// The admitted budget for that class.
        admitted_bytes: u64,
    },
    /// The backend could not hold the reservation.
    Backend(BackendError),
}

/// Why `execute` failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecuteError {
    /// The transaction is not in the `Prepared`/`Executing` state the call
    /// requires. Retry is disabled: after `Failed`/`Aborted`/`Committed`
    /// there is no re-execution path.
    InvalidState {
        /// The current state.
        state: TransactionState,
    },
    /// The operation is not part of the prepare snapshot (the accepted
    /// plan) — `execute` must never silently grow the accepted plan (S3).
    OperationOutsideSnapshot {
        /// The rejected operation reference.
        operation_ref: OperationRef,
    },
    /// The backend failed while running or staging an operation. The
    /// transaction recorded the failure and moved to `Failed`; no partial
    /// publication is possible.
    Backend(BackendError),
}

/// Why `commit` did not publish.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommitError {
    /// The transaction is not in the `Executing` state.
    InvalidState {
        /// The current state.
        state: TransactionState,
    },
    /// The declared `TransactionCommitBoundary` was not reached — some
    /// boundary barriers/launches have not completed. Transient: nothing was
    /// published, the transaction stays `Executing`, and `commit` may be
    /// called again once the events join the boundary.
    BoundaryNotReached {
        /// The boundary references not yet completed.
        missing: BTreeSet<BoundaryRef>,
    },
    /// The atomic publication failed. Nothing was published; the transaction
    /// moved to `Failed` and must be aborted.
    PublishFailed(BackendError),
}

/// Why `abort` could not run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbortError {
    /// The transaction already committed — a committed transaction is
    /// terminal.
    AlreadyCommitted,
}
