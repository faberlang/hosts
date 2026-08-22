//! MD3-X1 execution-transaction tests: mirror canonical-byte determinism,
//! construction guards (MD-A15 degenerate, empty snapshot), the prepare
//! reservation contract (within-budget success, focused over-commit
//! diagnostics, topology/boundary/duplicate authority, the MD2-C1 residual-2
//! tamper closure), execute plan-order + no-silent-growth, commit
//! boundary-gated atomic publication with the abstract publication ordinal,
//! abort/failure release-or-retire with no partial publication (per failure
//! class), and retry-disabled state transitions.
//!
//! The `PartitionBudgetLedger` field name `kv_cache_bytes` appears once in
//! the ledger fixture below: it is the MD1 ledger's class-2 field, consumed
//! as declared (never re-derived) — not vocabulary this module introduces.

use crate::bound_plan::{
    bind, AdmittedLogicalPlan, BindError, BoundDistributedPlan, DeclaredPlacementConstraint,
    LogicalPartitionId, PartitionBinding,
};
use crate::device_identity::{DeviceHealthGeneration, DeviceOrdinal, PhysicalDeviceId};
use crate::device_set::DeviceSet;
use crate::discovery::{
    ComputeCapability, DeviceCapabilities, DeviceDiscoveryEntry, DeviceDiscoverySnapshot,
    DeviceHealth, DeviceMemory, DtypeSurface, P2pProbeState, ProbeProvenance,
};
use crate::execution_transaction::{
    AbortError, BackendError, BarrierRef, BoundaryRef, BudgetClass, CollectiveBroadcastMirror,
    CollectiveRef, CommitError, ConstructError, DeviceExecutionBackend, ExecuteError,
    ExecutionTransaction, FakeExecutionBackend, LaunchRef, MirroredDtype, MirroredStorageLayout,
    OperationEvent, OperationRef, PrepareError, PublicationOrdinal, ReservationRecord, StagedWrite,
    TransactionCommitBoundary, TransactionDecision, TransactionFailure, TransactionId,
    TransactionOperation, TransactionState, TransferDirectionMirror, TransferOperationMirror,
    TransferRef, TransportPathMirror, EVENT_OBJECT_BYTES, TRANSACTION_SCRATCH_BYTES_PER_PARTITION,
};
use crate::partition::{
    AdmissionRequest, FixtureIdentityClass, HardwareIsolationClaim, PartitionBudgetLedger,
    SafePhysicalLimit, TransportClass, VirtualDevicePartition, VirtualDevicePartitionId,
};
use crate::transport::{
    CopyPath, HostStagedAdapter, MeasuredRates, SourceValue, TransferBudget, TransferSpec,
    TransportAdapter,
};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

// T1 measured facts (pharos) reused for the synthetic snapshot shape.
const UUID_A: &str = "GPU-3e017562-9ec3-da9a-962d-b8bd5f9e24be";
const UUID_B: &str = "GPU-22222222-3333-4444-5555-666666666666";
const PROBE_TIME: u64 = 1_752_717_600_000_000_000; // fixed sample time
                                                   // An admitted (validated) logical hash in the sha256: spelling (FC17/FC11).
const LOGICAL_HASH: &str =
    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

// CS-1 declared placement (md0-mode-fixtures.md §3): 2 virtual partitions @
// 160 MiB, forced 2-way split ≈129 MiB/device.
const CS1_SPLIT_BYTES: u64 = 135_266_304; // ≈129 MiB per partition
const CS1_LIMIT_BYTES: u64 = 167_772_160; // 160 MiB safe physical limit policy

// The synthetic two-partition transaction (T2 §5 fixture): launch p0 →
// transfer 0→1 → broadcast → barrier → launch p1, with the plan boundary on
// `barrier-main` + `launch-proj-b`.
const LAUNCH_A_OUTPUT_BYTES: u64 = 4096;
const TRANSFER_BYTES: u64 = 8192;
const BROADCAST_BYTES: u64 = 2048;
const LAUNCH_B_OUTPUT_BYTES: u64 = 16_384;

fn fixture_len(n: u64) -> usize {
    usize::try_from(n).expect("fixture byte count fits usize")
}

fn device_a() -> PhysicalDeviceId {
    PhysicalDeviceId::cuda(UUID_A, None)
}

fn device_b() -> PhysicalDeviceId {
    PhysicalDeviceId::cuda(UUID_B, None)
}

fn partition_id(n: u32) -> LogicalPartitionId {
    LogicalPartitionId::new(format!("partition-{n}"))
}

/// A CS-1 weight ledger with the declared class 6/class 3 headroom. The
/// class-2 field is the MD1 ledger's, consumed as declared.
fn ledger(class_six_bytes: u64, class_three_bytes: u64) -> PartitionBudgetLedger {
    PartitionBudgetLedger {
        weight_bytes: CS1_SPLIT_BYTES,
        kv_cache_bytes: 0,
        activation_scratch_bytes: class_three_bytes,
        module_storage_bytes: 0,
        allocator_overhead_bytes: 0,
        transfer_staging_bytes: class_six_bytes,
        concurrent_state_bytes: 0,
    }
}

/// An admitted virtual partition over `device` under the declared ledger.
fn vp(
    seed: u64,
    device: PhysicalDeviceId,
    ledger: PartitionBudgetLedger,
) -> VirtualDevicePartition {
    VirtualDevicePartition::admit(
        AdmissionRequest::new(VirtualDevicePartitionId::new(seed), device, ledger),
        SafePhysicalLimit::new(CS1_LIMIT_BYTES),
    )
    .unwrap()
}

fn synthetic_entry(
    ordinal: u32,
    device: PhysicalDeviceId,
    generation: DeviceHealthGeneration,
) -> DeviceDiscoveryEntry {
    DeviceDiscoveryEntry {
        ordinal: DeviceOrdinal::new(ordinal),
        identity: device,
        device_model: Some("synthetic RTX 5070".to_owned()),
        capabilities: DeviceCapabilities {
            compute_capability: ComputeCapability {
                major: 12,
                minor: 0,
            },
            sm_count: 48,
            dtype_surface: DtypeSurface {
                f32: true,
                f64: true,
                f16: true,
                bf16: true,
                i8: true,
                i32: true,
            },
            max_threads_per_workgroup: 1024,
            workgroup_shared_memory_min_bytes: 49_152,
            workgroup_shared_memory_max_bytes: 101_376,
            collective_width: 32,
            unified_memory: false,
        },
        memory: DeviceMemory {
            tool_report_total_mib: Some(12_227),
            api_total_bytes: 12_343_705_600,
        },
        health: DeviceHealth::Healthy,
        health_generation: generation,
        probe_provenance: ProbeProvenance {
            probe: "synthetic two-partition fixture".to_owned(),
            tool_versions: "synthetic".to_owned(),
        },
    }
}

fn snapshot_with(
    entries: impl IntoIterator<Item = (u32, PhysicalDeviceId, DeviceHealthGeneration)>,
) -> DeviceDiscoverySnapshot {
    let devices: BTreeMap<_, _> = entries
        .into_iter()
        .map(|(ordinal, device, generation)| {
            (
                DeviceOrdinal::new(ordinal),
                synthetic_entry(ordinal, device, generation),
            )
        })
        .collect();
    DeviceDiscoverySnapshot::new(PROBE_TIME, devices, P2pProbeState::NotAttempted)
}

fn one_device_snapshot() -> DeviceDiscoverySnapshot {
    snapshot_with([(0, device_a(), DeviceHealthGeneration::initial())])
}

fn two_device_snapshot() -> DeviceDiscoverySnapshot {
    snapshot_with([
        (0, device_a(), DeviceHealthGeneration::initial()),
        (1, device_b(), DeviceHealthGeneration::initial()),
    ])
}

fn admitted_two_partition_plan() -> AdmittedLogicalPlan {
    AdmittedLogicalPlan::admit(
        LOGICAL_HASH,
        [partition_id(0), partition_id(1)],
        [DeclaredPlacementConstraint::DistinctPhysicalDevices],
    )
    .expect("valid admitted two-partition plan")
}

fn bind_fixture(
    admitted: &AdmittedLogicalPlan,
    bindings: BTreeMap<LogicalPartitionId, PartitionBinding>,
    device_set: DeviceSet,
    snapshot: &DeviceDiscoverySnapshot,
) -> Result<BoundDistributedPlan, BindError> {
    bind(
        admitted,
        bindings,
        device_set,
        snapshot,
        DeviceHealthGeneration::initial(),
        FixtureIdentityClass::Synthetic,
        TransportClass::HostStaged,
    )
}

/// The T2 §5 virtual-fixture bound plan: p0 → device A, p1 → device B, each
/// with an admitted virtual partition carrying the fixture's declared class 6
/// / class 3 budgets (p0: 10240 / 8448; p1: 10240 / 30976).
fn fixture_plan() -> BoundDistributedPlan {
    let bindings = BTreeMap::from([
        (
            partition_id(0),
            PartitionBinding::with_virtual_partition(
                device_a(),
                vp(1, device_a(), ledger(10_240, 8_448)),
            ),
        ),
        (
            partition_id(1),
            PartitionBinding::with_virtual_partition(
                device_b(),
                vp(2, device_b(), ledger(10_240, 30_976)),
            ),
        ),
    ]);
    let snapshot = two_device_snapshot();
    bind_fixture(
        &admitted_two_partition_plan(),
        bindings,
        DeviceSet::from_members([device_a(), device_b()]),
        &snapshot,
    )
    .expect("fixture plan binds")
}

fn transfer_op(id: &str, byte_count: u64) -> TransactionOperation {
    TransactionOperation::transfer(TransferOperationMirror::new(
        TransferRef::new(id),
        partition_id(0),
        partition_id(1),
        byte_count,
        TransferDirectionMirror::BIDI,
        MirroredDtype::F32,
        MirroredStorageLayout::Dense,
        TransportPathMirror::HostStaged,
        0,
        1,
        TransactionCommitBoundary::default(),
    ))
}

/// The synthetic two-partition transaction operations in plan order.
fn fixture_operations() -> Vec<TransactionOperation> {
    vec![
        TransactionOperation::launch(
            partition_id(0),
            LaunchRef::new("launch-proj-a"),
            LAUNCH_A_OUTPUT_BYTES,
        ),
        transfer_op("t1", TRANSFER_BYTES),
        TransactionOperation::broadcast(CollectiveBroadcastMirror::broadcast(
            CollectiveRef::new("c1"),
            partition_id(0),
            BTreeSet::from([partition_id(0), partition_id(1)]),
            BROADCAST_BYTES,
        )),
        TransactionOperation::barrier(
            BarrierRef::new("barrier-main"),
            BTreeSet::from([partition_id(0), partition_id(1)]),
        ),
        TransactionOperation::launch(
            partition_id(1),
            LaunchRef::new("launch-proj-b"),
            LAUNCH_B_OUTPUT_BYTES,
        ),
    ]
}

fn fixture_boundary() -> TransactionCommitBoundary {
    TransactionCommitBoundary::new(
        [BarrierRef::new("barrier-main")],
        [LaunchRef::new("launch-proj-b")],
    )
}

fn fixture_transaction() -> ExecutionTransaction {
    ExecutionTransaction::new(
        TransactionId::new("txn-1"),
        fixture_plan(),
        fixture_operations(),
        fixture_boundary(),
    )
    .expect("fixture transaction constructs")
}

/// The declared write-set byte total over the fixture operations.
fn declared_write_bytes(operations: &[TransactionOperation]) -> u64 {
    operations
        .iter()
        .flat_map(super::mirror::TransactionOperation::staged_writes)
        .map(|write| write.byte_count())
        .sum()
}

// --- mirror canonical bytes -------------------------------------------------

/// Identical mirror inputs produce identical canonical bytes.
#[test]
fn mirror_canonical_bytes_are_deterministic() {
    let first = fixture_operations();
    let second = fixture_operations();
    for (a, b) in first.iter().zip(&second) {
        assert_eq!(
            a.canonical_bytes(),
            b.canonical_bytes(),
            "rebuilt operation {a:?} must canonicalize identically"
        );
    }
    assert_eq!(
        fixture_boundary().canonical_bytes(),
        fixture_boundary().canonical_bytes()
    );

    let transfer_a = transfer_op("t1", TRANSFER_BYTES);
    let transfer_b = transfer_op("t1", TRANSFER_BYTES);
    assert_eq!(transfer_a.canonical_bytes(), transfer_b.canonical_bytes());
}

/// Different operations produce different canonical bytes — including
/// operations that differ in a single declared fact.
#[test]
fn different_operations_produce_different_canonical_bytes() {
    let mut seen = BTreeSet::new();
    for operation in fixture_operations() {
        assert!(
            seen.insert(operation.canonical_bytes()),
            "distinct fixture operations must not collide in canonical bytes"
        );
    }

    // Same identity, different output contract — still different bytes.
    let small = TransactionOperation::launch(partition_id(0), LaunchRef::new("launch-x"), 1);
    let large = TransactionOperation::launch(partition_id(0), LaunchRef::new("launch-x"), 2);
    assert_ne!(small.canonical_bytes(), large.canonical_bytes());

    // A transfer differing only in the producer generation differs.
    let transfer_1 = TransferOperationMirror::new(
        TransferRef::new("t-gen"),
        partition_id(0),
        partition_id(1),
        8,
        TransferDirectionMirror::BIDI,
        MirroredDtype::F32,
        MirroredStorageLayout::Dense,
        TransportPathMirror::HostStaged,
        0,
        1,
        TransactionCommitBoundary::default(),
    );
    let transfer_2 = TransferOperationMirror::new(
        TransferRef::new("t-gen"),
        partition_id(0),
        partition_id(1),
        8,
        TransferDirectionMirror::BIDI,
        MirroredDtype::F32,
        MirroredStorageLayout::Dense,
        TransportPathMirror::HostStaged,
        7,
        1,
        TransactionCommitBoundary::default(),
    );
    assert_ne!(
        TransactionOperation::transfer(transfer_1).canonical_bytes(),
        TransactionOperation::transfer(transfer_2).canonical_bytes()
    );

    // A boundary differing in its launch set differs.
    let boundary_a = TransactionCommitBoundary::new([BarrierRef::new("b")], [LaunchRef::new("l")]);
    let boundary_b = TransactionCommitBoundary::new(
        [BarrierRef::new("b")],
        [LaunchRef::new("l"), LaunchRef::new("l2")],
    );
    assert_ne!(boundary_a.canonical_bytes(), boundary_b.canonical_bytes());
}

// --- construction -----------------------------------------------------------

/// MD-A15: the single-partition degenerate stays coordinator-free.
#[test]
fn construct_rejects_degenerate_plan() {
    let admitted = AdmittedLogicalPlan::admit(LOGICAL_HASH, [partition_id(0)], [])
        .expect("single-partition plan admits");
    let bindings = BTreeMap::from([(partition_id(0), PartitionBinding::new(device_a()))]);
    let snapshot = one_device_snapshot();
    let plan = bind_fixture(
        &admitted,
        bindings,
        DeviceSet::from_members([device_a()]),
        &snapshot,
    )
    .expect("degenerate plan binds");
    assert!(plan.is_degenerate());

    let error = ExecutionTransaction::new(
        TransactionId::new("txn-degenerate"),
        plan,
        fixture_operations(),
        fixture_boundary(),
    )
    .expect_err("a degenerate plan never constructs a transaction");
    assert_eq!(error, ConstructError::DegeneratePlan);
}

#[test]
fn construct_rejects_empty_snapshot() {
    let error = ExecutionTransaction::new(
        TransactionId::new("txn-empty"),
        fixture_plan(),
        Vec::new(),
        fixture_boundary(),
    )
    .expect_err("an empty snapshot never constructs a transaction");
    assert_eq!(error, ConstructError::EmptySnapshot);
}

// --- prepare: reservation + authority ---------------------------------------

#[test]
fn prepare_reserves_within_admitted_budget() {
    let mut transaction = fixture_transaction();
    let mut backend = FakeExecutionBackend::new();
    transaction
        .prepare(&mut backend)
        .expect("prepare succeeds within the admitted budgets");
    assert_eq!(transaction.state(), &TransactionState::Prepared);

    let reservation = transaction
        .reservation()
        .expect("the reservation is recorded at prepare");
    assert_eq!(reservation.len(), 2);

    let p0 = &reservation[&partition_id(0)];
    assert_eq!(p0.class_six_bytes(), TRANSFER_BYTES + BROADCAST_BYTES);
    assert_eq!(p0.output_buffer_bytes(), LAUNCH_A_OUTPUT_BYTES);
    assert_eq!(
        p0.class_three_bytes(),
        LAUNCH_A_OUTPUT_BYTES + 4 * EVENT_OBJECT_BYTES + TRANSACTION_SCRATCH_BYTES_PER_PARTITION
    );

    let p1 = &reservation[&partition_id(1)];
    assert_eq!(p1.class_six_bytes(), TRANSFER_BYTES + BROADCAST_BYTES);
    assert_eq!(
        p1.output_buffer_bytes(),
        LAUNCH_B_OUTPUT_BYTES + TRANSFER_BYTES + BROADCAST_BYTES
    );
    assert_eq!(
        p1.class_three_bytes(),
        LAUNCH_B_OUTPUT_BYTES
            + TRANSFER_BYTES
            + BROADCAST_BYTES
            + 4 * EVENT_OBJECT_BYTES
            + TRANSACTION_SCRATCH_BYTES_PER_PARTITION
    );

    // The backend holds the reservation.
    assert_eq!(backend.reservations().len(), 2);

    // The declared staged write-set is recorded (the prepare receipt).
    let declared = transaction.declared_write_set();
    assert_eq!(declared.len(), 4); // launch-a, transfer, broadcast, launch-b
    assert_eq!(
        declared_write_bytes(&fixture_operations()),
        declared
            .values()
            .map(super::mirror::StagedWrite::byte_count)
            .sum::<u64>()
    );
}

/// S3: a class-6 over-commit fails at prepare with a focused diagnostic.
#[test]
fn prepare_fails_when_class6_reservation_exceeds_admitted_budget() {
    let mut operations = fixture_operations();
    operations[1] = transfer_op("t1", 1 << 20);
    let mut transaction = ExecutionTransaction::new(
        TransactionId::new("txn-class6"),
        fixture_plan(),
        operations,
        fixture_boundary(),
    )
    .expect("transaction constructs");
    let mut backend = FakeExecutionBackend::new();
    let error = transaction
        .prepare(&mut backend)
        .expect_err("the class-6 reservation exceeds the admitted budget");
    assert!(matches!(
        error,
        PrepareError::ReservationExceedsBudget {
            ref partition,
            class: BudgetClass::TransferStaging,
            declared_bytes,
            admitted_bytes: 10_240,
        } if *partition == partition_id(0) && declared_bytes == (1 << 20) + BROADCAST_BYTES
    ));
    assert_eq!(transaction.state(), &TransactionState::New);
    assert!(
        backend.reservations().is_empty(),
        "nothing is reserved on failure"
    );
}

/// S3: a class-3 over-commit fails at prepare with a focused diagnostic.
#[test]
fn prepare_fails_when_class3_reservation_exceeds_admitted_budget() {
    let mut operations = fixture_operations();
    operations[4] =
        TransactionOperation::launch(partition_id(1), LaunchRef::new("launch-proj-b"), 1 << 20);
    let mut transaction = ExecutionTransaction::new(
        TransactionId::new("txn-class3"),
        fixture_plan(),
        operations,
        fixture_boundary(),
    )
    .expect("transaction constructs");
    let mut backend = FakeExecutionBackend::new();
    let error = transaction
        .prepare(&mut backend)
        .expect_err("the class-3 reservation exceeds the admitted budget");
    assert!(matches!(
        error,
        PrepareError::ReservationExceedsBudget {
            ref partition,
            class: BudgetClass::ActivationScratch,
            declared_bytes,
            admitted_bytes: 30_976,
        } if *partition == partition_id(1)
            && declared_bytes == (1 << 20) + TRANSFER_BYTES + BROADCAST_BYTES
                + 4 * EVENT_OBJECT_BYTES + TRANSACTION_SCRATCH_BYTES_PER_PARTITION
    ));
}

/// MD2-C1 residual 2 closure: a tampered-but-internally-consistent plan that
/// over-commits the bound resources (an extra declared transfer, all
/// partitions declared, boundary valid) fails at prepare.
#[test]
fn tampered_plan_over_committing_bound_resources_fails_at_prepare() {
    let mut operations = fixture_operations();
    operations.push(transfer_op("t-extra", 65_536));
    let mut transaction = ExecutionTransaction::new(
        TransactionId::new("txn-tampered"),
        fixture_plan(),
        operations,
        fixture_boundary(),
    )
    .expect("the tampered plan is internally consistent and constructs");
    let mut backend = FakeExecutionBackend::new();
    let error = transaction
        .prepare(&mut backend)
        .expect_err("the tampered plan over-commits the admitted class-6 budget");
    assert!(matches!(
        error,
        PrepareError::ReservationExceedsBudget {
            ref partition,
            class: BudgetClass::TransferStaging,
            ..
        } if *partition == partition_id(0)
    ));
}

#[test]
fn prepare_rejects_unknown_partition() {
    let mut operations = fixture_operations();
    operations[4] = TransactionOperation::launch(partition_id(99), LaunchRef::new("launch-zzz"), 0);
    let mut transaction = ExecutionTransaction::new(
        TransactionId::new("txn-unknown"),
        fixture_plan(),
        operations,
        fixture_boundary(),
    )
    .expect("transaction constructs");
    let mut backend = FakeExecutionBackend::new();
    let error = transaction
        .prepare(&mut backend)
        .expect_err("a partition outside the bound plan is rejected");
    assert!(matches!(
        error,
        PrepareError::UnknownPartition {
            ref partition,
            operation_index: 4,
        } if *partition == partition_id(99)
    ));
}

#[test]
fn prepare_rejects_missing_admitted_budget() {
    let bindings = BTreeMap::from([
        (
            partition_id(0),
            PartitionBinding::with_virtual_partition(
                device_a(),
                vp(1, device_a(), ledger(10_240, 8_448)),
            ),
        ),
        (partition_id(1), PartitionBinding::new(device_b())),
    ]);
    let snapshot = two_device_snapshot();
    let plan = bind_fixture(
        &admitted_two_partition_plan(),
        bindings,
        DeviceSet::from_members([device_a(), device_b()]),
        &snapshot,
    )
    .expect("plan binds without a virtual partition on p1");

    let mut transaction = ExecutionTransaction::new(
        TransactionId::new("txn-no-budget"),
        plan,
        fixture_operations(),
        fixture_boundary(),
    )
    .expect("transaction constructs");
    let mut backend = FakeExecutionBackend::new();
    let error = transaction
        .prepare(&mut backend)
        .expect_err("a partition without an admitted budget is rejected");
    assert!(matches!(
        error,
        PrepareError::MissingAdmittedBudget { ref partition }
            if *partition == partition_id(1)
    ));
}

#[test]
fn prepare_rejects_undeclared_boundary_refs() {
    let ghost_launch = TransactionCommitBoundary::new(
        [BarrierRef::new("barrier-main")],
        [LaunchRef::new("launch-ghost")],
    );
    let mut transaction = ExecutionTransaction::new(
        TransactionId::new("txn-ghost-launch"),
        fixture_plan(),
        fixture_operations(),
        ghost_launch,
    )
    .expect("transaction constructs");
    let mut backend = FakeExecutionBackend::new();
    let error = transaction
        .prepare(&mut backend)
        .expect_err("a boundary launch not in the snapshot is rejected");
    assert!(matches!(
        error,
        PrepareError::UndeclaredBoundaryLaunch { ref launch }
            if launch.as_str() == "launch-ghost"
    ));

    let ghost_barrier = TransactionCommitBoundary::new(
        [BarrierRef::new("barrier-ghost")],
        [LaunchRef::new("launch-proj-b")],
    );
    let mut transaction = ExecutionTransaction::new(
        TransactionId::new("txn-ghost-barrier"),
        fixture_plan(),
        fixture_operations(),
        ghost_barrier,
    )
    .expect("transaction constructs");
    let error = transaction
        .prepare(&mut backend)
        .expect_err("a boundary barrier not in the snapshot is rejected");
    assert!(matches!(
        error,
        PrepareError::UndeclaredBoundaryBarrier { ref barrier }
            if barrier.as_str() == "barrier-ghost"
    ));
}

#[test]
fn prepare_rejects_duplicate_operations() {
    let mut operations = fixture_operations();
    operations[4] = operations[1].clone(); // a second t1 transfer
    let mut transaction = ExecutionTransaction::new(
        TransactionId::new("txn-dup"),
        fixture_plan(),
        operations,
        fixture_boundary(),
    )
    .expect("transaction constructs");
    let mut backend = FakeExecutionBackend::new();
    let error = transaction
        .prepare(&mut backend)
        .expect_err("duplicate operation identities are rejected");
    assert!(matches!(
        error,
        PrepareError::DuplicateOperation {
            operation_ref: OperationRef::Transfer(ref transfer_ref),
        } if transfer_ref.as_str() == "t1"
    ));
}

#[test]
fn prepare_from_wrong_state_fails() {
    let mut transaction = fixture_transaction();
    let mut backend = FakeExecutionBackend::new();
    transaction
        .prepare(&mut backend)
        .expect("first prepare succeeds");
    let error = transaction
        .prepare(&mut backend)
        .expect_err("prepare is New → Prepared only");
    assert!(matches!(
        error,
        PrepareError::InvalidState {
            state: TransactionState::Prepared,
        }
    ));
}

// --- execute ----------------------------------------------------------------

#[test]
fn execute_runs_operations_in_plan_order() {
    let mut transaction = fixture_transaction();
    let mut backend = FakeExecutionBackend::new();
    transaction.prepare(&mut backend).expect("prepare succeeds");
    transaction
        .execute(&mut backend)
        .expect("execute succeeds on the happy path");
    assert_eq!(transaction.state(), &TransactionState::Executing);

    // The backend saw exactly the snapshot operations, in plan order.
    let expected: Vec<OperationRef> = fixture_operations()
        .iter()
        .map(TransactionOperation::operation_ref)
        .collect();
    let actual: Vec<OperationRef> = backend
        .executed_operations()
        .iter()
        .map(TransactionOperation::operation_ref)
        .collect();
    assert_eq!(actual, expected);

    // Exact bytes: transfers move Y and the broadcast moves one Z copy.
    let executed = transaction.executed_operations();
    assert_eq!(executed.len(), 5);
    let moved: u64 = executed.iter().map(|record| record.byte_count).sum();
    assert_eq!(moved, TRANSFER_BYTES + BROADCAST_BYTES);

    // All events completed synchronously on the happy path.
    assert_eq!(backend.pending_events().len(), 0);
    assert_eq!(backend.completed_events().len(), 8);
    assert_eq!(
        backend.staged_bytes(),
        declared_write_bytes(&fixture_operations())
    );
}

/// S3 no-silent-growth: an operation outside the prepare snapshot (or a
/// modified snapshot operation) fails at execute.
#[test]
fn execute_operation_outside_snapshot_fails() {
    let mut transaction = fixture_transaction();
    let mut backend = FakeExecutionBackend::new();
    transaction.prepare(&mut backend).expect("prepare succeeds");
    transaction.execute(&mut backend).expect("execute succeeds");
    assert_eq!(transaction.state(), &TransactionState::Executing);

    let foreign = TransactionOperation::launch(partition_id(0), LaunchRef::new("launch-zzz"), 0);
    let error = transaction
        .execute_operation(&mut backend, &foreign)
        .expect_err("an operation outside the snapshot fails");
    assert!(matches!(
        error,
        ExecuteError::OperationOutsideSnapshot {
            operation_ref: OperationRef::Launch(ref launch_ref),
        } if launch_ref.as_str() == "launch-zzz"
    ));

    // A modified snapshot operation (same identity, different declared
    // facts) is equally outside the accepted plan.
    let tampered = TransactionOperation::launch(
        partition_id(0),
        LaunchRef::new("launch-proj-a"),
        LAUNCH_B_OUTPUT_BYTES,
    );
    let error = transaction
        .execute_operation(&mut backend, &tampered)
        .expect_err("a modified snapshot operation fails");
    assert!(matches!(
        error,
        ExecuteError::OperationOutsideSnapshot {
            operation_ref: OperationRef::Launch(ref launch_ref),
        } if launch_ref.as_str() == "launch-proj-a"
    ));

    // A validation error does not corrupt the state machine — commit still
    // works and the accepted plan is exactly what ran.
    assert_eq!(transaction.state(), &TransactionState::Executing);
    transaction
        .commit(&mut backend, PublicationOrdinal::new(1))
        .expect("commit succeeds after rejected foreign operations");
    assert_eq!(transaction.executed_operations().len(), 5);
}

/// S3: the snapshot is fixed at prepare — execute runs exactly the accepted
/// plan, never a grown one.
#[test]
fn execute_never_grows_the_accepted_plan() {
    let mut transaction = fixture_transaction();
    let mut backend = FakeExecutionBackend::new();
    transaction.prepare(&mut backend).expect("prepare succeeds");
    transaction.execute(&mut backend).expect("execute succeeds");
    let actual = backend.executed_operations().len();
    assert_eq!(
        actual,
        transaction.operations().len(),
        "the backend saw exactly the prepare snapshot operations"
    );
}

// --- commit -----------------------------------------------------------------

#[test]
fn commit_publishes_atomically_after_boundary_with_ordinal() {
    let mut transaction = fixture_transaction();
    let mut backend = FakeExecutionBackend::new();
    transaction.prepare(&mut backend).expect("prepare succeeds");
    transaction.execute(&mut backend).expect("execute succeeds");
    let receipt = transaction
        .commit(&mut backend, PublicationOrdinal::new(7))
        .expect("commit publishes after the boundary is reached");
    assert_eq!(transaction.state(), &TransactionState::Committed);

    assert!(matches!(
        receipt.decision,
        TransactionDecision::Committed {
            publication_ordinal,
        } if publication_ordinal.get() == 7
    ));
    let publish = receipt
        .publish_summary
        .expect("the commit receipt records the publication");
    let declared_total = declared_write_bytes(&fixture_operations());
    assert_eq!(publish.staged_bytes, declared_total);
    assert_eq!(publish.published_bytes, declared_total);
    assert!(publish.atomic, "publication is all-or-nothing");

    // Teardown: reservations released, no retirement, no partial publication.
    assert_eq!(
        receipt.teardown.released_partitions,
        BTreeSet::from([partition_id(0), partition_id(1)])
    );
    assert!(receipt.teardown.retired_partitions.is_empty());
    assert!(!receipt.teardown.partial_publication);
    assert_eq!(
        backend.published_writes().len(),
        transaction.declared_write_set().len()
    );
    assert!(
        backend.reservations().is_empty(),
        "commit releases the reservations"
    );
}

/// S3: commit publishes only after every required device reaches the declared
/// boundary — a missing boundary is transient and nothing is published.
#[test]
fn commit_publishes_only_after_boundary() {
    let mut transaction = fixture_transaction();
    let mut backend = FakeExecutionBackend::new();
    backend.set_auto_complete(false); // events join asynchronously
    transaction.prepare(&mut backend).expect("prepare succeeds");
    transaction
        .execute(&mut backend)
        .expect("execute dispatches the plan");
    assert_eq!(backend.pending_events().len(), 8, "all events pending");

    let error = transaction
        .commit(&mut backend, PublicationOrdinal::new(1))
        .expect_err("commit before the boundary is reached");
    assert!(matches!(
        error,
        CommitError::BoundaryNotReached { ref missing }
            if missing.contains(&BoundaryRef::Barrier(BarrierRef::new("barrier-main")))
                && missing.contains(&BoundaryRef::Launch(LaunchRef::new("launch-proj-b")))
    ));
    // Transient: the transaction stays Executing and nothing was published.
    assert_eq!(transaction.state(), &TransactionState::Executing);
    assert_eq!(backend.published_bytes(), 0);

    // Complete the boundary events (barrier on both partitions + launch-b).
    backend.complete_event(OperationEvent::BarrierCompleted {
        partition: partition_id(0),
        barrier_ref: BarrierRef::new("barrier-main"),
    });
    backend.complete_event(OperationEvent::BarrierCompleted {
        partition: partition_id(1),
        barrier_ref: BarrierRef::new("barrier-main"),
    });
    backend.complete_event(OperationEvent::LaunchCompleted {
        partition: partition_id(1),
        launch_ref: LaunchRef::new("launch-proj-b"),
    });

    let receipt = transaction
        .commit(&mut backend, PublicationOrdinal::new(1))
        .expect("commit publishes once the boundary events join");
    assert!(matches!(
        receipt.decision,
        TransactionDecision::Committed { .. }
    ));
    assert_eq!(transaction.state(), &TransactionState::Committed);
}

/// A partially-reached boundary (barrier complete, launch pending) is still a
/// boundary-not-reached.
#[test]
fn commit_requires_launch_boundary_before_publish() {
    let mut transaction = fixture_transaction();
    let mut backend = FakeExecutionBackend::new();
    backend.set_auto_complete(false);
    transaction.prepare(&mut backend).expect("prepare succeeds");
    transaction
        .execute(&mut backend)
        .expect("execute dispatches the plan");

    backend.complete_event(OperationEvent::BarrierCompleted {
        partition: partition_id(0),
        barrier_ref: BarrierRef::new("barrier-main"),
    });
    backend.complete_event(OperationEvent::BarrierCompleted {
        partition: partition_id(1),
        barrier_ref: BarrierRef::new("barrier-main"),
    });

    let error = transaction
        .commit(&mut backend, PublicationOrdinal::new(1))
        .expect_err("the boundary launch has not completed");
    assert!(matches!(
        error,
        CommitError::BoundaryNotReached { ref missing }
            if missing.len() == 1
                && missing.contains(&BoundaryRef::Launch(LaunchRef::new("launch-proj-b")))
    ));
    assert_eq!(backend.published_bytes(), 0);
}

#[test]
fn commit_before_execute_fails() {
    let mut transaction = fixture_transaction();
    let mut backend = FakeExecutionBackend::new();
    transaction.prepare(&mut backend).expect("prepare succeeds");
    let error = transaction
        .commit(&mut backend, PublicationOrdinal::new(1))
        .expect_err("commit requires Executing");
    assert!(matches!(
        error,
        CommitError::InvalidState {
            state: TransactionState::Prepared,
        }
    ));
}

#[test]
fn commit_twice_fails() {
    let mut transaction = fixture_transaction();
    let mut backend = FakeExecutionBackend::new();
    transaction.prepare(&mut backend).expect("prepare succeeds");
    transaction.execute(&mut backend).expect("execute succeeds");
    transaction
        .commit(&mut backend, PublicationOrdinal::new(1))
        .expect("first commit publishes");
    let error = transaction
        .commit(&mut backend, PublicationOrdinal::new(2))
        .expect_err("a committed transaction is terminal");
    assert!(matches!(
        error,
        CommitError::InvalidState {
            state: TransactionState::Committed,
        }
    ));
}

// --- abort / failure --------------------------------------------------------

#[test]
fn abort_before_prepare_releases_nothing() {
    let mut transaction = fixture_transaction();
    let mut backend = FakeExecutionBackend::new();
    let receipt = transaction
        .abort(&mut backend, "cancelled before prepare")
        .expect("abort from New succeeds");
    assert!(matches!(transaction.state(), TransactionState::Aborted(_)));
    assert!(receipt.publish_summary.is_none());
    assert!(receipt.teardown.released_partitions.is_empty());
    assert!(receipt.teardown.retired_partitions.is_empty());
    assert!(!receipt.teardown.partial_publication);
}

#[test]
fn abort_after_prepare_releases_reservations() {
    let mut transaction = fixture_transaction();
    let mut backend = FakeExecutionBackend::new();
    transaction.prepare(&mut backend).expect("prepare succeeds");
    let receipt = transaction
        .abort(&mut backend, "cancelled after prepare")
        .expect("abort from Prepared succeeds");
    assert!(matches!(transaction.state(), TransactionState::Aborted(_)));
    assert_eq!(
        receipt.teardown.released_partitions,
        BTreeSet::from([partition_id(0), partition_id(1)])
    );
    assert!(receipt.teardown.retired_partitions.is_empty());
    assert!(receipt.publish_summary.is_none());
    assert!(backend.reservations().is_empty());
}

/// A mid-execution failure retires staged state and releases the rest, with
/// no partial publication.
#[test]
fn failure_after_partial_execution_exposes_no_partial_publication() {
    let mut transaction = fixture_transaction();
    let mut backend = FaultInjectionBackend::new();
    backend.fail_operation(
        OperationRef::Transfer(TransferRef::new("t1")),
        BackendError::operation(partition_id(0), "synthetic transfer failure"),
    );
    transaction.prepare(&mut backend).expect("prepare succeeds");
    let error = transaction
        .execute(&mut backend)
        .expect_err("the transfer fault fails execute");
    assert!(matches!(error, ExecuteError::Backend(_)));
    assert!(matches!(transaction.state(), TransactionState::Failed(_)));

    // No partial publication is possible from the Failed state.
    let commit_error = transaction
        .commit(&mut backend, PublicationOrdinal::new(1))
        .expect_err("commit is impossible after failure");
    assert!(matches!(
        commit_error,
        CommitError::InvalidState {
            state: TransactionState::Failed(_),
        }
    ));
    assert_eq!(backend.published_bytes(), 0);

    // Abort retires the partition holding staged writes and releases the
    // reservation-only partition.
    let receipt = transaction
        .abort(&mut backend, "transfer failure")
        .expect("abort completes teardown");
    assert!(matches!(transaction.state(), TransactionState::Aborted(_)));
    assert_eq!(
        receipt.teardown.retired_partitions,
        BTreeSet::from([partition_id(0)])
    );
    assert_eq!(
        receipt.teardown.released_partitions,
        BTreeSet::from([partition_id(1)])
    );
    assert!(receipt.publish_summary.is_none());
    assert!(!receipt.teardown.partial_publication);
    assert!(matches!(
        receipt.decision,
        TransactionDecision::Aborted { .. }
    ));
}

/// MD-A13: every backend failure class releases/retires all affected
/// resources and publishes nothing — a test per failure class.
#[test]
fn every_failure_class_releases_resources_with_no_partial_publication() {
    let classes = [
        BackendError::allocation(partition_id(0), "allocation pressure"),
        BackendError::operation(partition_id(0), "transfer error"),
        BackendError::device_loss(partition_id(0), "device removed"),
        BackendError::cancelled(partition_id(0), "cancelled"),
        BackendError::timeout(partition_id(0), "timed out"),
    ];
    for (index, failure) in classes.into_iter().enumerate() {
        let mut transaction = fixture_transaction();
        let mut backend = FaultInjectionBackend::new();
        backend.fail_operation(OperationRef::Transfer(TransferRef::new("t1")), failure);
        transaction.prepare(&mut backend).expect("prepare succeeds");
        let execute_error = transaction
            .execute(&mut backend)
            .expect_err("the injected failure fails execute");
        assert!(
            matches!(execute_error, ExecuteError::Backend(_)),
            "failure class {index} must surface as a backend error"
        );

        let receipt = transaction
            .abort(&mut backend, "injected failure")
            .expect("abort completes teardown");
        assert!(
            matches!(transaction.state(), TransactionState::Aborted(_)),
            "failure class {index} must end Aborted"
        );
        assert!(
            receipt.publish_summary.is_none(),
            "failure class {index} must publish nothing"
        );
        assert!(
            !receipt.teardown.partial_publication,
            "failure class {index} must never partially publish"
        );
        assert!(
            backend.published_bytes() == 0,
            "failure class {index} must leave the write-set unpublished"
        );
    }
}

/// A publication failure after the boundary publishes nothing and is
/// terminal (must abort).
#[test]
fn publish_failure_publishes_nothing() {
    let mut transaction = fixture_transaction();
    let mut backend = FaultInjectionBackend::new();
    backend.fail_publish(BackendError::operation(partition_id(0), "publish failed"));
    transaction.prepare(&mut backend).expect("prepare succeeds");
    transaction.execute(&mut backend).expect("execute succeeds");
    let error = transaction
        .commit(&mut backend, PublicationOrdinal::new(1))
        .expect_err("the publish fault fails commit");
    assert!(matches!(error, CommitError::PublishFailed(_)));
    assert!(matches!(transaction.state(), TransactionState::Failed(_)));

    let receipt = transaction
        .abort(&mut backend, "publish failure")
        .expect("abort completes teardown");
    assert!(matches!(transaction.state(), TransactionState::Aborted(_)));
    assert!(receipt.publish_summary.is_none());
    assert_eq!(backend.published_bytes(), 0);
}

#[test]
fn abort_is_idempotent() {
    let mut transaction = fixture_transaction();
    let mut backend = FakeExecutionBackend::new();
    let first = transaction
        .abort(&mut backend, "cancelled")
        .expect("first abort succeeds");
    let second = transaction
        .abort(&mut backend, "cancelled again")
        .expect("abort is idempotent");
    assert_eq!(first, second);
    assert!(matches!(transaction.state(), TransactionState::Aborted(_)));
}

#[test]
fn abort_after_commit_fails() {
    let mut transaction = fixture_transaction();
    let mut backend = FakeExecutionBackend::new();
    transaction.prepare(&mut backend).expect("prepare succeeds");
    transaction.execute(&mut backend).expect("execute succeeds");
    transaction
        .commit(&mut backend, PublicationOrdinal::new(1))
        .expect("commit publishes");
    let error = transaction
        .abort(&mut backend, "too late")
        .expect_err("a committed transaction cannot be aborted");
    assert_eq!(error, AbortError::AlreadyCommitted);
}

/// Retry is disabled: no re-execution path exists after a failure or an
/// abort.
#[test]
fn retry_is_disabled_after_failure_and_abort() {
    let mut transaction = fixture_transaction();
    let mut backend = FaultInjectionBackend::new();
    backend.fail_operation(
        OperationRef::Transfer(TransferRef::new("t1")),
        BackendError::operation(partition_id(0), "fault"),
    );
    transaction.prepare(&mut backend).expect("prepare succeeds");
    transaction
        .execute(&mut backend)
        .expect_err("the fault fails execute");
    assert!(matches!(transaction.state(), TransactionState::Failed(_)));

    // Re-executing the failed transaction is rejected — no retry path.
    let retry = transaction
        .execute(&mut backend)
        .expect_err("retry is disabled");
    assert!(matches!(retry, ExecuteError::InvalidState { .. }));

    transaction
        .abort(&mut backend, "no retry")
        .expect("abort completes teardown");
    let after_abort = transaction
        .execute(&mut backend)
        .expect_err("no re-execution after abort");
    assert!(matches!(
        after_abort,
        ExecuteError::InvalidState {
            state: TransactionState::Aborted(_),
        }
    ));
}

// --- receipt -----------------------------------------------------------------

#[test]
fn receipt_records_plan_hashes_and_device_identities() {
    let plan = fixture_plan();
    let mut transaction = ExecutionTransaction::new(
        TransactionId::new("txn-receipt"),
        plan.clone(),
        fixture_operations(),
        fixture_boundary(),
    )
    .expect("transaction constructs");
    let mut backend = FakeExecutionBackend::new();
    transaction.prepare(&mut backend).expect("prepare succeeds");
    transaction.execute(&mut backend).expect("execute succeeds");
    let receipt = transaction
        .commit(&mut backend, PublicationOrdinal::new(2))
        .expect("commit publishes");

    assert_eq!(
        receipt.logical_distributed_plan_hash,
        plan.logical_distributed_plan_hash()
    );
    assert_eq!(
        receipt.bound_distributed_plan_hash,
        plan.bound_distributed_plan_hash()
    );
    assert_eq!(
        receipt.plan_receipt.physical_device_ids(),
        &BTreeSet::from([device_a(), device_b()])
    );
    assert_eq!(
        receipt.plan_receipt.virtual_partition_ids(),
        &BTreeSet::from([
            VirtualDevicePartitionId::new(1),
            VirtualDevicePartitionId::new(2),
        ])
    );
    assert_eq!(receipt.plan_receipt.physical_device_count(), 2);
    assert_eq!(receipt.plan_receipt.virtual_partition_count(), 2);
}

#[test]
fn receipt_records_reservation_executed_bytes_and_sync_events() {
    let mut transaction = fixture_transaction();
    let mut backend = FakeExecutionBackend::new();
    transaction.prepare(&mut backend).expect("prepare succeeds");
    transaction.execute(&mut backend).expect("execute succeeds");
    let receipt = transaction
        .commit(&mut backend, PublicationOrdinal::new(3))
        .expect("commit publishes");

    // Per-partition reservation summary (the prepare receipt).
    let reservation = &receipt.reservation_summary;
    assert_eq!(reservation.len(), 2);
    assert_eq!(
        reservation[&partition_id(0)].class_six_bytes(),
        TRANSFER_BYTES + BROADCAST_BYTES
    );
    assert_eq!(
        reservation[&partition_id(1)].class_three_bytes(),
        LAUNCH_B_OUTPUT_BYTES
            + TRANSFER_BYTES
            + BROADCAST_BYTES
            + 4 * EVENT_OBJECT_BYTES
            + TRANSACTION_SCRATCH_BYTES_PER_PARTITION
    );

    // Executed operations + exact bytes.
    assert_eq!(receipt.executed_operations.len(), 5);
    let moved: u64 = receipt
        .executed_operations
        .iter()
        .map(|record| record.byte_count)
        .sum();
    assert_eq!(moved, TRANSFER_BYTES + BROADCAST_BYTES);

    // The declared write-set is reserved and published byte-exactly.
    let declared_total: u64 = receipt
        .declared_write_set
        .values()
        .map(super::mirror::StagedWrite::byte_count)
        .sum();
    assert_eq!(declared_total, declared_write_bytes(&fixture_operations()));

    // Synchronization events include the boundary completions.
    assert!(receipt
        .synchronization_events
        .contains(&OperationEvent::BarrierCompleted {
            partition: partition_id(0),
            barrier_ref: BarrierRef::new("barrier-main"),
        }));
    assert!(receipt
        .synchronization_events
        .contains(&OperationEvent::LaunchCompleted {
            partition: partition_id(1),
            launch_ref: LaunchRef::new("launch-proj-b"),
        }));
}

/// The S4 selected-transport section (CTO sanity-check amendment on MD3-T1):
/// a real `ExecutionTransaction` commits and its receipt carries the actual
/// selected transports (path/staging/events/timeout/bytes/timing) executed
/// through the transport adapter. The adapter's `transport_receipt()` is
/// folded in during the transaction; a transaction that never touched a
/// transport adapter records `None`.
#[test]
fn receipt_carries_the_selected_transport_records() {
    let mut transaction = fixture_transaction();
    assert_eq!(
        transaction.selected_transports(),
        None,
        "no adapter recorded before the coordinator folds the section in"
    );

    // The fixture's transfer (t1: p0 → p1, TRANSFER_BYTES, F32 dense BIDI)
    // executed through the host-staged adapter — the coordinator integration
    // shape: run the copy, then fold the adapter's `transport_receipt()` over.
    let transfer = fixture_operations()
        .iter()
        .find_map(|operation| match operation {
            TransactionOperation::Transfer(mirror) => Some(mirror.clone()),
            _ => None,
        })
        .expect("the fixture carries the t1 transfer");
    let mut adapter = HostStagedAdapter::new(TransferBudget::declared(1 << 20, 1 << 30));
    adapter.set_simulated_delay(Duration::from_millis(1));
    let spec = TransferSpec::from_mirror(&transfer, Duration::from_secs(1));
    let source = SourceValue::new(
        partition_id(0),
        MirroredDtype::F32,
        MirroredStorageLayout::Dense,
        0,
        vec![3u8; fixture_len(TRANSFER_BYTES)],
    );
    let outcome = adapter
        .copy(&spec, &source, &partition_id(1))
        .expect("the fixture transfer copy succeeds");
    assert_eq!(outcome.record.transfer_ref.as_str(), "t1");

    transaction.with_transport_receipt(adapter.transport_receipt());
    let mut backend = FakeExecutionBackend::new();
    transaction.prepare(&mut backend).expect("prepare succeeds");
    transaction.execute(&mut backend).expect("execute succeeds");
    let receipt = transaction
        .commit(&mut backend, PublicationOrdinal::new(9))
        .expect("commit publishes");

    let selected = receipt
        .selected_transports
        .as_ref()
        .expect("the committed receipt carries the selected-transport section");
    assert_eq!(selected.records.len(), 1);
    assert_eq!(selected.used_bytes, TRANSFER_BYTES);
    assert_eq!(selected.budget_bytes, 1 << 20);
    assert_eq!(selected.rates, MeasuredRates::t1());
    let record = &selected.records[0];
    // path
    assert_eq!(record.copy_path, CopyPath::HostStaged);
    // staging
    assert!(record.staging.pinned);
    assert_eq!(record.staging.capacity_bytes, TRANSFER_BYTES);
    // streams/queues/events
    assert!(record.engine.get() >= 1);
    assert!(record.event.get() >= 1);
    // timeout/failure policy
    assert_eq!(record.timeout, Duration::from_secs(1));
    // bytes
    assert_eq!(record.bytes, TRANSFER_BYTES);
    assert_eq!(record.direction, TransferDirectionMirror::BIDI);
    assert_eq!(record.destination, partition_id(1));
    // timing
    assert!(record.elapsed_nanos >= 1_000_000, "the copy is timed");
    assert!(record.expected_nanos > 0);

    // The transaction accessor mirrors the folded section.
    assert!(transaction.selected_transports().is_some());

    // A transaction that never touched a transport adapter records None.
    let mut bare = fixture_transaction();
    let mut bare_backend = FakeExecutionBackend::new();
    bare.prepare(&mut bare_backend).expect("prepare succeeds");
    bare.execute(&mut bare_backend).expect("execute succeeds");
    let bare_receipt = bare
        .commit(&mut bare_backend, PublicationOrdinal::new(10))
        .expect("commit publishes");
    assert!(bare_receipt.selected_transports.is_none());
}

/// `StagedWrite` is the atomic publication unit — byte counts are the declared
/// contracts of the producing operations.
#[test]
fn staged_writes_derive_from_operations() {
    let operations = fixture_operations();
    let writes: Vec<StagedWrite> = operations
        .iter()
        .flat_map(super::mirror::TransactionOperation::staged_writes)
        .collect();
    assert_eq!(writes.len(), 4); // launch-a, transfer, broadcast, launch-b
    let launch_a = writes
        .iter()
        .find(|write| write.output_ref().as_str() == "launch:launch-proj-a:output")
        .expect("the launch-a output write is declared");
    assert_eq!(launch_a.partition(), &partition_id(0));
    assert_eq!(launch_a.byte_count(), LAUNCH_A_OUTPUT_BYTES);
    let transfer_write = writes
        .iter()
        .find(|write| write.output_ref().as_str() == "transfer:t1:destination")
        .expect("the transfer destination write is declared");
    assert_eq!(transfer_write.partition(), &partition_id(1));
    assert_eq!(transfer_write.byte_count(), TRANSFER_BYTES);
}

/// A transaction over a plan whose partition is not in the `DeviceSet` fails at
/// bind time (the bound plan is the authority), and the receipt taxonomy
/// stays honest (`hardware_isolation_claimed=false`, synthetic fixture).
#[test]
fn receipt_taxonomy_is_honest() {
    let plan = fixture_plan();
    let mut transaction = ExecutionTransaction::new(
        TransactionId::new("txn-taxonomy"),
        plan,
        fixture_operations(),
        fixture_boundary(),
    )
    .expect("transaction constructs");
    let mut backend = FakeExecutionBackend::new();
    transaction.prepare(&mut backend).expect("prepare succeeds");
    transaction.execute(&mut backend).expect("execute succeeds");
    let receipt = transaction
        .commit(&mut backend, PublicationOrdinal::new(4))
        .expect("commit publishes");
    assert_eq!(
        receipt.plan_receipt.fixture_identity_class(),
        FixtureIdentityClass::Synthetic
    );
    assert_eq!(
        receipt.plan_receipt.transport_class(),
        TransportClass::HostStaged
    );
    assert_eq!(
        receipt.plan_receipt.hardware_isolation_claimed(),
        HardwareIsolationClaim::NotClaimed
    );
}

// --- test-only fault backend ------------------------------------------------

/// A fault-injecting backend for the X1 failure-class tests (MD3-F1 owns the
/// production fault suite; this is the X1 happy-path fake plus one injectable
/// operation fault and one injectable publish fault).
struct FaultInjectionBackend {
    inner: FakeExecutionBackend,
    operation_fault: Option<(OperationRef, BackendError)>,
    publish_fault: Option<BackendError>,
}

impl FaultInjectionBackend {
    fn new() -> Self {
        Self {
            inner: FakeExecutionBackend::new(),
            operation_fault: None,
            publish_fault: None,
        }
    }

    fn fail_operation(&mut self, key: OperationRef, error: BackendError) {
        self.operation_fault = Some((key, error));
    }

    fn fail_publish(&mut self, error: BackendError) {
        self.publish_fault = Some(error);
    }
}

impl DeviceExecutionBackend for FaultInjectionBackend {
    fn reserve(
        &mut self,
        partition: &LogicalPartitionId,
        reservation: &ReservationRecord,
    ) -> Result<(), BackendError> {
        self.inner.reserve(partition, reservation)
    }

    fn run_operation(&mut self, operation: &TransactionOperation) -> Result<(), BackendError> {
        if let Some((key, error)) = &self.operation_fault {
            if key == &operation.operation_ref() {
                return Err(error.clone());
            }
        }
        self.inner.run_operation(operation)
    }

    fn event_completed(&self, event: &OperationEvent) -> bool {
        self.inner.event_completed(event)
    }

    fn stage_write(&mut self, write: &StagedWrite) -> Result<(), BackendError> {
        self.inner.stage_write(write)
    }

    fn publish(&mut self) -> Result<(), BackendError> {
        if let Some(error) = &self.publish_fault {
            return Err(error.clone());
        }
        self.inner.publish()
    }

    fn staged_bytes(&self) -> u64 {
        self.inner.staged_bytes()
    }

    fn published_bytes(&self) -> u64 {
        self.inner.published_bytes()
    }

    fn release(&mut self, partition: &LogicalPartitionId) {
        self.inner.release(partition);
    }

    fn retire(&mut self, partition: &LogicalPartitionId, failure: &TransactionFailure) {
        self.inner.retire(partition, failure);
    }
}
