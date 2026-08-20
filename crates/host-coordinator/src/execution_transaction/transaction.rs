//! The generic `ExecutionTransaction` coordinator (MD3-X1): prepare/execute/
//! commit/abort over the mirror accepted plan and the device-execution
//! backend. Section split out of `execution_transaction.rs` (polish).

use crate::bound_plan::{BoundDistributedPlan, LogicalPartitionId};
use crate::execution_transaction::backend::DeviceExecutionBackend;
use crate::execution_transaction::errors::{
    AbortError, CommitError, ConstructError, ExecuteError, PrepareError,
};
use crate::execution_transaction::mirror::{
    BoundaryRef, OperationEvent, OperationRef, OutputRef, StagedWrite, TransactionCommitBoundary,
    TransactionOperation,
};
use crate::execution_transaction::receipt::{
    ExecutedOperationRecord, PublishSummary, TeardownFacts, TransactionDecision,
    TransactionReceipt, TransactionTimings,
};
use crate::execution_transaction::reservation::{
    derive_declared_write_set, derive_reservation, BudgetClass, ReservationRecord,
};
use crate::execution_transaction::state_machine::{
    PublicationOrdinal, TransactionFailure, TransactionId, TransactionState,
};
use crate::partition::PartitionBudgetLedger;
use crate::transport::TransportReceipt;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

/// The generic `ExecutionTransaction` coordinator (MD3-X1).
///
/// Consumes the accepted plan as a dependency-free mirror over a
/// [`BoundDistributedPlan`]. Lifecycle: [`new`](Self::new) → [`prepare`](
/// Self::prepare) → [`execute`](Self::execute) → [`commit`](Self::commit) |
/// [`abort`](Self::abort).
#[derive(Debug, Clone)]
pub struct ExecutionTransaction {
    id: TransactionId,
    bound_plan: BoundDistributedPlan,
    operations: Vec<TransactionOperation>,
    commit_boundary: TransactionCommitBoundary,
    operation_keys: BTreeMap<OperationRef, usize>,
    state: TransactionState,
    reservation: Option<BTreeMap<LogicalPartitionId, ReservationRecord>>,
    declared_write_set: BTreeMap<OutputRef, StagedWrite>,
    staged_write_set: BTreeMap<OutputRef, StagedWrite>,
    executed_operations: Vec<ExecutedOperationRecord>,
    completed_events: BTreeSet<OperationEvent>,
    failure: Option<TransactionFailure>,
    decision: Option<TransactionDecision>,
    publish_summary: Option<PublishSummary>,
    teardown: TeardownFacts,
    timings: TransactionTimings,
    transport_receipt: Option<TransportReceipt>,
    receipt: Option<TransactionReceipt>,
}

impl ExecutionTransaction {
    /// Construct a transaction over an accepted (already bound) distributed
    /// plan. The MD-A15 single-partition degenerate and an empty snapshot are
    /// rejected at construction. The snapshot is fixed here — it is the
    /// accepted plan and never grows.
    #[must_use]
    pub fn new(
        id: TransactionId,
        bound_plan: BoundDistributedPlan,
        operations: Vec<TransactionOperation>,
        commit_boundary: TransactionCommitBoundary,
    ) -> Result<Self, ConstructError> {
        if bound_plan.is_degenerate() {
            return Err(ConstructError::DegeneratePlan);
        }
        if operations.is_empty() {
            return Err(ConstructError::EmptySnapshot);
        }
        Ok(Self {
            id,
            bound_plan,
            operations,
            commit_boundary,
            operation_keys: BTreeMap::new(),
            state: TransactionState::New,
            reservation: None,
            declared_write_set: BTreeMap::new(),
            staged_write_set: BTreeMap::new(),
            executed_operations: Vec::new(),
            completed_events: BTreeSet::new(),
            failure: None,
            decision: None,
            publish_summary: None,
            teardown: TeardownFacts::default(),
            timings: TransactionTimings::default(),
            transport_receipt: None,
            receipt: None,
        })
    }

    /// The transaction identity.
    #[must_use]
    pub fn id(&self) -> &TransactionId {
        &self.id
    }

    /// The bound plan this transaction executes.
    #[must_use]
    pub fn bound_plan(&self) -> &BoundDistributedPlan {
        &self.bound_plan
    }

    /// The current state-machine state.
    #[must_use]
    pub fn state(&self) -> &TransactionState {
        &self.state
    }

    /// The accepted plan (the prepare snapshot), in plan order.
    #[must_use]
    pub fn operations(&self) -> &[TransactionOperation] {
        &self.operations
    }

    /// The declared commit boundary.
    #[must_use]
    pub fn commit_boundary(&self) -> &TransactionCommitBoundary {
        &self.commit_boundary
    }

    /// The per-partition reservation recorded at prepare (the prepare
    /// receipt's reservation summary); `None` before prepare.
    #[must_use]
    pub fn reservation(&self) -> Option<&BTreeMap<LogicalPartitionId, ReservationRecord>> {
        self.reservation.as_ref()
    }

    /// The declared staged write-set (reserved at prepare); empty before
    /// prepare.
    #[must_use]
    pub fn declared_write_set(&self) -> &BTreeMap<OutputRef, StagedWrite> {
        &self.declared_write_set
    }

    /// The executed operations in plan order (populated by execute).
    #[must_use]
    pub fn executed_operations(&self) -> &[ExecutedOperationRecord] {
        &self.executed_operations
    }

    /// The completed synchronization events.
    #[must_use]
    pub fn completed_events(&self) -> &BTreeSet<OperationEvent> {
        &self.completed_events
    }

    /// The recorded failure, when the transaction failed or was aborted.
    #[must_use]
    pub fn failure(&self) -> Option<&TransactionFailure> {
        self.failure.as_ref()
    }

    /// The final receipt, present after commit or abort.
    #[must_use]
    pub fn receipt(&self) -> Option<&TransactionReceipt> {
        self.receipt.as_ref()
    }

    /// The S4 selected-transport section recorded for this transaction, if
    /// the coordinator handed the transport adapter's `transport_receipt()`
    /// over ([`with_transport_receipt`](Self::with_transport_receipt)).
    #[must_use]
    pub fn selected_transports(&self) -> Option<&TransportReceipt> {
        self.transport_receipt.as_ref()
    }

    /// Record the S4 selected-transport section from the transport adapter
    /// used during this transaction. The coordinator folds the adapter's
    /// `transport_receipt()` over after execution (the actual selected
    /// transports: copy path/staging/events/timeout/bytes/timing + budget
    /// accounting at the measured rates); the commit/abort receipt carries it
    /// verbatim. Additive — a transaction that never touched a transport
    /// adapter records `None`. The portable logical plan is never touched
    /// (S4: the mirror carries only the admissibility `path_label`).
    pub fn with_transport_receipt(&mut self, receipt: TransportReceipt) -> &mut Self {
        self.transport_receipt = Some(receipt);
        self
    }

    /// Reserve the transaction's resources and validate the accepted plan
    /// (`New → Prepared`).
    ///
    /// Validation order (deterministic):
    ///
    /// 1. **Topology authority** — every operation's partitions must be
    ///    declared by the bound plan.
    /// 2. **Snapshot integrity** — no duplicate operation identities.
    /// 3. **Boundary declaration** — every boundary barrier/launch must be
    ///    declared by the snapshot.
    /// 4. **Admitted budgets** — every referenced partition must carry an
    ///    admitted virtual partition (the actual admitted ledger, frozen at
    ///    admission).
    /// 5. **Reservation (S3)** — the derived reservation is charged against
    ///    the admitted class budgets; an over-commit (a tampered-but-
    ///    internally-consistent plan) fails here, before execute (MD2-C1
    ///    residual 2 closure).
    ///
    /// On success the reservation is held on the backend and recorded (the
    /// prepare receipt). On any failure nothing is reserved and the
    /// transaction stays `New`.
    pub fn prepare(
        &mut self,
        backend: &mut dyn DeviceExecutionBackend,
    ) -> Result<(), PrepareError> {
        if self.state != TransactionState::New {
            return Err(PrepareError::InvalidState {
                state: self.state.clone(),
            });
        }
        let start = Instant::now();
        let bindings = self
            .bound_plan
            .bindings()
            .expect("construction rejected the degenerate plan");
        let declared_partitions: BTreeSet<LogicalPartitionId> = bindings.keys().cloned().collect();

        // 1. Topology authority.
        for (index, operation) in self.operations.iter().enumerate() {
            for partition in operation.partitions() {
                if !declared_partitions.contains(&partition) {
                    return Err(PrepareError::UnknownPartition {
                        partition,
                        operation_index: index,
                    });
                }
            }
        }

        // 2. Snapshot integrity: unique operation identities.
        let mut operation_keys = BTreeMap::new();
        for (index, operation) in self.operations.iter().enumerate() {
            let key = operation.operation_ref();
            if operation_keys.insert(key.clone(), index).is_some() {
                return Err(PrepareError::DuplicateOperation { operation_ref: key });
            }
        }

        // 3. Boundary declaration.
        for barrier in self.commit_boundary.barriers() {
            let declared = self.operations.iter().any(|operation| {
                matches!(operation,
                        TransactionOperation::Barrier { barrier_ref, .. }
                            if barrier_ref == barrier)
            });
            if !declared {
                return Err(PrepareError::UndeclaredBoundaryBarrier {
                    barrier: barrier.clone(),
                });
            }
        }
        for launch in self.commit_boundary.launches() {
            let declared = self.operations.iter().any(|operation| {
                matches!(operation,
                    TransactionOperation::Launch { launch_ref, .. }
                        if launch_ref == launch)
            });
            if !declared {
                return Err(PrepareError::UndeclaredBoundaryLaunch {
                    launch: launch.clone(),
                });
            }
        }

        // 4. Admitted budgets: every referenced partition must carry the
        //    actual admitted ledger (frozen at admission).
        let referenced: BTreeSet<LogicalPartitionId> = self
            .operations
            .iter()
            .flat_map(super::mirror::TransactionOperation::partitions)
            .collect();
        for partition in &referenced {
            let binding = bindings
                .get(partition)
                .expect("step 1 validated the partition is declared");
            if binding.virtual_partition().is_none() {
                return Err(PrepareError::MissingAdmittedBudget {
                    partition: partition.clone(),
                });
            }
        }

        // 5. Reservation derivation + budget check (S3).
        let reservation = derive_reservation(&self.operations, &referenced);
        for (partition, record) in &reservation {
            let binding = bindings
                .get(partition)
                .expect("step 4 validated the admitted budget");
            let ledger: &PartitionBudgetLedger = binding
                .virtual_partition()
                .expect("step 4 validated the admitted budget")
                .ledger();
            let class_six = record.class_six_bytes();
            if class_six > ledger.transfer_staging_bytes {
                return Err(PrepareError::ReservationExceedsBudget {
                    partition: partition.clone(),
                    class: BudgetClass::TransferStaging,
                    declared_bytes: class_six,
                    admitted_bytes: ledger.transfer_staging_bytes,
                });
            }
            let class_three = record.class_three_bytes();
            if class_three > ledger.activation_scratch_bytes {
                return Err(PrepareError::ReservationExceedsBudget {
                    partition: partition.clone(),
                    class: BudgetClass::ActivationScratch,
                    declared_bytes: class_three,
                    admitted_bytes: ledger.activation_scratch_bytes,
                });
            }
        }

        // Hold the reservation on the backend.
        for (partition, record) in &reservation {
            if let Err(error) = backend.reserve(partition, record) {
                return Err(PrepareError::Backend(error));
            }
        }

        self.operation_keys = operation_keys;
        self.reservation = Some(reservation);
        self.declared_write_set = derive_declared_write_set(&self.operations);
        self.timings.prepare_nanos = start.elapsed().as_nanos() as u64;
        self.state = TransactionState::Prepared;
        Ok(())
    }

    /// Run the accepted plan in order (`Prepared → Executing`).
    ///
    /// The plan's declared order is the dependency-ready order; the backend
    /// gates actual readiness. Each operation dispatches via
    /// [`execute_operation`](Self::execute_operation), so an operation
    /// outside the prepare snapshot can never run. On a backend failure the
    /// transaction moves to `Failed` (the recorded failure); no partial
    /// publication is possible and `abort` completes teardown. Retry is
    /// disabled — `execute` cannot run again.
    pub fn execute(
        &mut self,
        backend: &mut dyn DeviceExecutionBackend,
    ) -> Result<(), ExecuteError> {
        if self.state != TransactionState::Prepared {
            return Err(ExecuteError::InvalidState {
                state: self.state.clone(),
            });
        }
        let start = Instant::now();
        self.state = TransactionState::Executing;
        let snapshot = self.operations.clone();
        for operation in &snapshot {
            if let Err(error) = self.execute_operation(backend, operation) {
                self.timings.execute_nanos = start.elapsed().as_nanos() as u64;
                return Err(error);
            }
        }
        self.timings.execute_nanos = start.elapsed().as_nanos() as u64;
        Ok(())
    }

    /// Run one operation of the accepted plan against the backend.
    ///
    /// The operation must be **exactly** an operation of the prepare snapshot
    /// (same identity *and* same declared facts) — an operation outside the
    /// snapshot fails with [`ExecuteError::OperationOutsideSnapshot`]
    /// (no silent growth, S3). On success the operation's staged writes are
    /// staged and its completed events are recorded.
    pub fn execute_operation(
        &mut self,
        backend: &mut dyn DeviceExecutionBackend,
        operation: &TransactionOperation,
    ) -> Result<(), ExecuteError> {
        if self.state != TransactionState::Executing {
            return Err(ExecuteError::InvalidState {
                state: self.state.clone(),
            });
        }
        let key = operation.operation_ref();
        let index = match self.operation_keys.get(&key) {
            Some(index) => *index,
            None => {
                return Err(ExecuteError::OperationOutsideSnapshot { operation_ref: key });
            }
        };
        if &self.operations[index] != operation {
            return Err(ExecuteError::OperationOutsideSnapshot { operation_ref: key });
        }

        if let Err(error) = backend.run_operation(operation) {
            self.fail(TransactionFailure::Backend(error.clone()));
            return Err(ExecuteError::Backend(error));
        }
        for write in operation.staged_writes() {
            if let Err(error) = backend.stage_write(&write) {
                self.fail(TransactionFailure::Backend(error.clone()));
                return Err(ExecuteError::Backend(error));
            }
            self.staged_write_set
                .insert(write.output_ref().clone(), write);
        }
        self.executed_operations.push(ExecutedOperationRecord {
            operation: operation.clone(),
            byte_count: operation.byte_count(),
        });
        for event in operation.completed_events() {
            if backend.event_completed(&event) {
                self.completed_events.insert(event);
            }
        }
        Ok(())
    }

    /// Publish the staged write-set **atomically** after the declared
    /// [`TransactionCommitBoundary`] is reached (`Executing → Committed`).
    ///
    /// Every boundary barrier/launch must have completed (the backend
    /// confirms the events). A missing boundary is transient: nothing was
    /// published and the transaction stays `Executing` so the events can join
    /// and `commit` can be called again. After the boundary is reached the
    /// whole staged write-set publishes atomically; on success every
    /// reservation is released and the receipt records the abstract
    /// publication ordinal. A publication failure moves the transaction to
    /// `Failed` (nothing published) — `abort` then completes teardown.
    pub fn commit(
        &mut self,
        backend: &mut dyn DeviceExecutionBackend,
        ordinal: PublicationOrdinal,
    ) -> Result<TransactionReceipt, CommitError> {
        if self.state != TransactionState::Executing {
            return Err(CommitError::InvalidState {
                state: self.state.clone(),
            });
        }
        let start = Instant::now();

        // Re-check every snapshot event against the backend (async joins).
        self.refresh_completed_events(backend);

        let missing = self.missing_boundary_refs();
        if !missing.is_empty() {
            return Err(CommitError::BoundaryNotReached { missing });
        }

        if let Err(error) = backend.publish() {
            self.fail(TransactionFailure::PublishFailed {
                detail: format!("{error:?}"),
            });
            return Err(CommitError::PublishFailed(error));
        }

        // Release every held reservation (commit path).
        let reserved: BTreeSet<LogicalPartitionId> = self
            .reservation
            .as_ref()
            .map(|reservation| reservation.keys().cloned().collect())
            .unwrap_or_default();
        for partition in &reserved {
            backend.release(partition);
        }
        self.teardown.released_partitions.extend(reserved);

        self.publish_summary = Some(PublishSummary {
            staged_bytes: backend.staged_bytes(),
            published_bytes: backend.published_bytes(),
            atomic: true,
        });
        self.decision = Some(TransactionDecision::Committed {
            publication_ordinal: ordinal,
        });
        self.state = TransactionState::Committed;
        self.timings.finalize_nanos = start.elapsed().as_nanos() as u64;
        let receipt = self.build_receipt();
        self.receipt = Some(receipt.clone());
        Ok(receipt)
    }

    /// Release or retire every affected resource with no partial publication
    /// (`New | Prepared | Executing | Failed → Aborted`).
    ///
    /// Partitions holding staged writes are **retired** (staged state
    /// invalidated, reservation released); partitions only holding a
    /// reservation are **released**. The receipt records the abort decision
    /// with the reason. Idempotent: aborting an already-aborted transaction
    /// returns its receipt. A committed transaction cannot be aborted.
    pub fn abort(
        &mut self,
        backend: &mut dyn DeviceExecutionBackend,
        reason: impl Into<String>,
    ) -> Result<TransactionReceipt, AbortError> {
        if matches!(self.state, TransactionState::Aborted(_)) {
            return Ok(self
                .receipt
                .clone()
                .expect("an aborted transaction always has its receipt"));
        }
        if self.state == TransactionState::Committed {
            return Err(AbortError::AlreadyCommitted);
        }
        let start = Instant::now();

        let staged_partitions: BTreeSet<LogicalPartitionId> = self
            .staged_write_set
            .values()
            .map(|write| write.partition().clone())
            .collect();
        let reserved: BTreeSet<LogicalPartitionId> = self
            .reservation
            .as_ref()
            .map(|reservation| reservation.keys().cloned().collect())
            .unwrap_or_default();

        let failure = match &self.state {
            TransactionState::Failed(failure) => failure.clone(),
            _ => TransactionFailure::Cancelled {
                reason: reason.into(),
            },
        };

        // Retire staged state, release the remaining reservations.
        for partition in &staged_partitions {
            backend.retire(partition, &failure);
        }
        for partition in reserved.difference(&staged_partitions) {
            backend.release(partition);
        }
        self.teardown
            .retired_partitions
            .extend(staged_partitions.iter().cloned());
        self.teardown
            .released_partitions
            .extend(reserved.difference(&staged_partitions).cloned());

        self.failure = Some(failure.clone());
        self.state = TransactionState::Aborted(failure.clone());
        self.decision = Some(TransactionDecision::Aborted { failure });
        self.timings.finalize_nanos = start.elapsed().as_nanos() as u64;
        let receipt = self.build_receipt();
        self.receipt = Some(receipt.clone());
        Ok(receipt)
    }

    /// Record a failure and move the state machine to `Failed` (first failure
    /// wins).
    fn fail(&mut self, failure: TransactionFailure) {
        if matches!(
            self.state,
            TransactionState::New | TransactionState::Prepared | TransactionState::Executing
        ) {
            self.failure = Some(failure.clone());
            self.state = TransactionState::Failed(failure);
        }
    }

    /// Re-query the backend for every snapshot event and record the completed
    /// ones (asynchronous joins).
    fn refresh_completed_events(&mut self, backend: &dyn DeviceExecutionBackend) {
        for operation in &self.operations {
            for event in operation.completed_events() {
                if backend.event_completed(&event) {
                    self.completed_events.insert(event);
                }
            }
        }
    }

    /// The boundary references whose events have not completed. Every
    /// boundary barrier must be completed by every participant partition;
    /// every boundary launch must be completed by its partition.
    fn missing_boundary_refs(&self) -> BTreeSet<BoundaryRef> {
        let mut missing = BTreeSet::new();
        for barrier in self.commit_boundary.barriers() {
            let operation = self.operations.iter().find(|operation| {
                matches!(operation,
                    TransactionOperation::Barrier { barrier_ref, .. }
                        if barrier_ref == barrier)
            });
            let Some(operation) = operation else {
                // Prepare validated boundary declaration; defensive only.
                missing.insert(BoundaryRef::Barrier(barrier.clone()));
                continue;
            };
            for partition in operation.partitions() {
                let event = OperationEvent::BarrierCompleted {
                    partition: partition.clone(),
                    barrier_ref: barrier.clone(),
                };
                if !self.completed_events.contains(&event) {
                    missing.insert(BoundaryRef::Barrier(barrier.clone()));
                }
            }
        }
        for launch in self.commit_boundary.launches() {
            let operation = self.operations.iter().find(|operation| {
                matches!(operation,
                    TransactionOperation::Launch { launch_ref, .. }
                        if launch_ref == launch)
            });
            let Some(operation) = operation else {
                // Prepare validated boundary declaration; defensive only.
                missing.insert(BoundaryRef::Launch(launch.clone()));
                continue;
            };
            for partition in operation.partitions() {
                let event = OperationEvent::LaunchCompleted {
                    partition: partition.clone(),
                    launch_ref: launch.clone(),
                };
                if !self.completed_events.contains(&event) {
                    missing.insert(BoundaryRef::Launch(launch.clone()));
                }
            }
        }
        missing
    }

    /// Assemble the final receipt from the recorded transaction state. Called
    /// at commit and abort only.
    fn build_receipt(&self) -> TransactionReceipt {
        TransactionReceipt {
            transaction_id: self.id.clone(),
            logical_distributed_plan_hash: self
                .bound_plan
                .logical_distributed_plan_hash()
                .to_owned(),
            bound_distributed_plan_hash: self.bound_plan.bound_distributed_plan_hash().to_owned(),
            plan_receipt: self.bound_plan.receipt(),
            reservation_summary: self.reservation.clone().unwrap_or_default(),
            declared_write_set: self.declared_write_set.clone(),
            executed_operations: self.executed_operations.clone(),
            synchronization_events: self.completed_events.clone(),
            decision: self
                .decision
                .clone()
                .expect("the receipt is built only at commit or abort"),
            publish_summary: self.publish_summary,
            teardown: self.teardown.clone(),
            timings: self.timings,
            selected_transports: self.transport_receipt.clone(),
        }
    }
}
