//! Generic `ExecutionTransaction` state machine + staged write-set + atomic
//! publication (gpu-inference-multi-device, MD3-X1 — the serial gate).
//!
//! [`ExecutionTransaction`] coordinates one **abstract** execution of a
//! [`BoundDistributedPlan`](crate::bound_plan::BoundDistributedPlan) (MD2-B1)
//! over a [`DeviceExecutionBackend`]:
//!
//! - [`ExecutionTransaction::prepare`] reserves typed transfer staging,
//!   output buffers, events, and transaction scratch against the **admitted**
//!   [`PartitionBudgetLedger`](crate::partition::PartitionBudgetLedger) class
//!   budgets (CTO `6badaa01` S3): staging against class 6
//!   `transfer_staging_bytes`; outputs/events/scratch against class 3
//!   `activation_scratch_bytes`. A reservation that exceeds the bound
//!   partition's admitted budget fails **before** execute, and the reservation
//!   is recorded (the prepare receipt).
//! - [`ExecutionTransaction::execute`] runs the accepted plan in order
//!   against the backend, reusing the reservation and **never silently
//!   growing the accepted plan** (the prepare snapshot *is* the accepted
//!   plan; an operation outside it fails — S3).
//! - [`ExecutionTransaction::commit`] publishes the staged write-set
//!   **atomically** only after every required device reaches the declared
//!   [`TransactionCommitBoundary`], releases the reservations, and records the
//!   **abstract publication ordinal** (a transaction-scoped publication
//!   counter — never the semantic `ValueGeneration`, naming contract §3) in
//!   the commit receipt.
//! - [`ExecutionTransaction::abort`] releases or retires every affected
//!   resource with **no partial publication** (MD-A13).
//!
//! ## Mirror vocabulary
//!
//! faber-runtime cannot import radix-mir (FC18), so the transaction consumes
//! the accepted plan as a **dependency-free mirror** — the
//! [`DeclaredPlacementConstraint`](crate::bound_plan::DeclaredPlacementConstraint)
//! pattern (FC8): [`TransactionOperation`] mirrors
//! `ExecutionOperation::{Launch,Transfer,Collective,Barrier}` and
//! [`TransactionCommitBoundary`] mirrors `ExecutionCommitBoundary` (barriers +
//! launches). Mirrors serialize to **stable canonical bytes** — the
//! `push_str`/`push_u64` discipline from `device_identity.rs`/`bound_plan.rs`
//! (faber-runtime has no serde, FC6).
//!
//! ## Generic (S5)
//!
//! The transaction introduces **no inference or training vocabulary** (the
//! inference binding is MD3I's surface, CTO `6badaa01` S5). It is not a
//! universal inference/training facade (MD-A16). **Retry is disabled**: the
//! state machine (`New → Prepared → Executing → Committed | Aborted`, with
//! `Failed → Aborted`) has no re-execution path.
//!
//! ## Constraint-tampering authority (MD2-C1 residual 2)
//!
//! `prepare` validates the accepted plan's declared placement against the
//! **bound device set/topology and the actual admitted partition budgets** —
//! the frozen `PartitionBudgetLedger` of the virtual partitions attached to
//! the bound plan's bindings. A tampered-but-internally-consistent declared
//! plan that over-commits the bound resources derives a reservation that
//! exceeds the admitted class budgets and fails at `prepare`, before any
//! execution.
//!
//! ## Structure
//!
//! The module splits along its section boundaries into clean submodules:
//! `mirror` (vocabulary), `reservation` (S3 derivation), `backend`
//! (abstraction), `state_machine`, `errors`, `receipt`, and `transaction`
//! (the coordinator). Everything is re-exported here so the public surface is
//! unchanged.

mod backend;
mod errors;
mod mirror;
mod receipt;
mod reservation;
mod state_machine;
mod transaction;

pub use backend::{BackendError, DeviceExecutionBackend, FakeExecutionBackend};
pub use errors::{AbortError, CommitError, ConstructError, ExecuteError, PrepareError};
pub use mirror::{
    BarrierRef, BoundaryRef, CollectiveBroadcastMirror, CollectiveRef, LaunchRef, MirroredDtype,
    MirroredStorageLayout, OperationEvent, OperationRef, OutputRef, StagedWrite,
    TransactionCommitBoundary, TransactionOperation, TransferDirectionMirror,
    TransferOperationMirror, TransferRef, TransportPathMirror,
};
pub use receipt::{
    ExecutedOperationRecord, PublishSummary, TeardownFacts, TransactionDecision,
    TransactionReceipt, TransactionTimings,
};
pub use reservation::{
    BudgetClass, ReservationRecord, EVENT_OBJECT_BYTES, TRANSACTION_SCRATCH_BYTES_PER_PARTITION,
};
pub use state_machine::{PublicationOrdinal, TransactionFailure, TransactionId, TransactionState};
pub use transaction::ExecutionTransaction;

#[cfg(test)]
#[path = "execution_transaction_test.rs"]
mod tests;
