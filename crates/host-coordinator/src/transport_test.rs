//! MD3-T1 transport tests: typed/ranged rejections (dtype, layout, bounds,
//! generation, owner, destination — all fail **before copy**), the
//! host-staged adapter's labeled + timed byte-exact copy with budget
//! discipline at the T1 measured rates, the S4 selected-transport receipt
//! records (path/staging/events/timeout/bytes/timing), the timeout/failure
//! policy surfacing transfer errors the coordinator aborts on, the explicit
//! NOT-ATTEMPTED peer path with the per-directed-pair admission check, and
//! the structural mirror/admissibility separation (the portable logical plan
//! never carries the selected transport).
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
    BackendError, DeviceExecutionBackend, ExecuteError, ExecutionTransaction, FakeExecutionBackend,
    MirroredDtype, MirroredStorageLayout, OperationEvent, ReservationRecord, StagedWrite,
    TransactionCommitBoundary, TransactionFailure, TransactionId, TransactionOperation,
    TransactionState, TransferDirectionMirror, TransferOperationMirror, TransferRef,
    TransportPathMirror,
};
use crate::partition::{
    AdmissionRequest, FixtureIdentityClass, PartitionBudgetLedger, SafePhysicalLimit,
    TransportClass, VirtualDevicePartition, VirtualDevicePartitionId,
};
use crate::transport::{
    expected_copy_time_nanos, select_copy_path, validate_before_copy, ByteRange, CopyPath,
    DirectedPair, HostStagedAdapter, MeasuredRates, PairAdmissionError, PeerAdapter,
    PeerPairMeasurement, PeerPairRegistry, SourceValue, StagingPool, TransferBudget, TransferError,
    TransferRejection, TransferSpec, TransportAdapter, TransportReceipt,
};
use std::collections::BTreeMap;
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

// The typed/ranged transfer facts of the base fixture.
const GENERATION: u64 = 7;
const SOURCE_TOTAL_BYTES: u64 = 4096;
const TRANSFER_RANGE_BYTES: u64 = 1024;
// The transfer-only transaction (coordinator chain tests).
const TRANSFER_ONLY_BYTES: u64 = 1024;

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

/// A portable logical transfer mirror (the admissibility label only).
fn transfer_mirror(id: &str, byte_count: u64) -> TransferOperationMirror {
    TransferOperationMirror::new(
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
    )
}

fn transfer_op(id: &str, byte_count: u64) -> TransactionOperation {
    TransactionOperation::transfer(transfer_mirror(id, byte_count))
}

// --- typed/ranged transfer builders -----------------------------------------

/// The base declared transfer facts: owner p0, destination p1, range
/// `0..1024` of the source, F32 dense, generation [`GENERATION`], BIDI.
fn base_spec(timeout: Duration) -> TransferSpec {
    TransferSpec::new(
        TransferRef::new("t-base"),
        partition_id(0),
        partition_id(1),
        ByteRange::new(0, TRANSFER_RANGE_BYTES),
        TransferDirectionMirror::BIDI,
        MirroredDtype::F32,
        MirroredStorageLayout::Dense,
        GENERATION,
        timeout,
    )
}

/// The base actual source content: owner p0, F32 dense, generation
/// [`GENERATION`], 4096 bytes.
fn base_source() -> SourceValue {
    SourceValue::new(
        partition_id(0),
        MirroredDtype::F32,
        MirroredStorageLayout::Dense,
        GENERATION,
        vec![3u8; SOURCE_TOTAL_BYTES as usize],
    )
}

/// A rejection must leave the adapter untouched: no recorded copy, no budget
/// charge, no staging allocation — the fail-before-copy contract.
fn assert_adapter_untouched(adapter: &HostStagedAdapter) {
    assert!(
        adapter.selected_transfer_records().is_empty(),
        "a rejected transfer must never be recorded"
    );
    assert_eq!(
        adapter.used_bytes(),
        0,
        "a rejected transfer must charge no budget bytes"
    );
    assert_eq!(
        adapter.used_time_nanos(),
        0,
        "a rejected transfer must charge no budget time"
    );
    assert_eq!(
        adapter.staging_pool().allocations(),
        0,
        "a rejected transfer must allocate no staging (fail before copy)"
    );
}

// --- typed/ranged rejections (fail before copy) ------------------------------

#[test]
fn dtype_mismatch_rejects_before_copy() {
    let mut adapter = HostStagedAdapter::new(TransferBudget::declared(1 << 20, 1 << 30));
    let spec = base_spec(Duration::from_secs(1));
    let source = SourceValue::new(
        partition_id(0),
        MirroredDtype::I32,
        MirroredStorageLayout::Dense,
        GENERATION,
        vec![3u8; SOURCE_TOTAL_BYTES as usize],
    );
    let error = adapter
        .copy(&spec, &source, &partition_id(1))
        .expect_err("a dtype mismatch must reject before copy");
    match error {
        TransferError::Rejected(rejection) => {
            assert!(matches!(
                rejection,
                TransferRejection::DtypeMismatch {
                    ref transfer,
                    declared: MirroredDtype::F32,
                    actual: MirroredDtype::I32,
                } if transfer.as_str() == "t-base"
            ));
            assert_eq!(rejection.class(), "dtype mismatch");
            assert!(
                rejection.to_string().contains("dtype mismatch"),
                "the diagnostic must name the violated class + failing fact"
            );
        }
        other => panic!("expected Rejected, got {other:?}"),
    }
    assert_adapter_untouched(&adapter);
}

#[test]
fn layout_mismatch_rejects_before_copy() {
    let mut adapter = HostStagedAdapter::new(TransferBudget::declared(1 << 20, 1 << 30));
    let spec = base_spec(Duration::from_secs(1));
    let source = SourceValue::new(
        partition_id(0),
        MirroredDtype::F32,
        MirroredStorageLayout::BlockPacked,
        GENERATION,
        vec![3u8; SOURCE_TOTAL_BYTES as usize],
    );
    let error = adapter
        .copy(&spec, &source, &partition_id(1))
        .expect_err("a layout mismatch must reject before copy");
    assert!(matches!(
        error,
        TransferError::Rejected(TransferRejection::LayoutMismatch {
            declared: MirroredStorageLayout::Dense,
            actual: MirroredStorageLayout::BlockPacked,
            ..
        })
    ));
    assert_adapter_untouched(&adapter);
}

#[test]
fn out_of_bounds_range_rejects_before_copy() {
    let mut adapter = HostStagedAdapter::new(TransferBudget::declared(1 << 20, 1 << 30));
    let spec = TransferSpec::new(
        TransferRef::new("t-base"),
        partition_id(0),
        partition_id(1),
        ByteRange::new(SOURCE_TOTAL_BYTES - 512, TRANSFER_RANGE_BYTES),
        TransferDirectionMirror::BIDI,
        MirroredDtype::F32,
        MirroredStorageLayout::Dense,
        GENERATION,
        Duration::from_secs(1),
    );
    let source = base_source();
    let error = adapter
        .copy(&spec, &source, &partition_id(1))
        .expect_err("an out-of-bounds range must reject before copy");
    match error {
        TransferError::Rejected(rejection) => {
            assert!(matches!(
                rejection,
                TransferRejection::RangeOutOfBounds {
                    ref transfer,
                    range: ByteRange { offset: 3584, length: 1024 },
                    source_bytes: SOURCE_TOTAL_BYTES,
                } if transfer.as_str() == "t-base"
            ));
            assert_eq!(rejection.class(), "out-of-bounds range");
            assert!(rejection.to_string().contains("out-of-bounds range"));
        }
        other => panic!("expected Rejected, got {other:?}"),
    }
    assert_adapter_untouched(&adapter);
}

#[test]
fn generation_mismatch_rejects_before_copy() {
    let mut adapter = HostStagedAdapter::new(TransferBudget::declared(1 << 20, 1 << 30));
    let spec = base_spec(Duration::from_secs(1));
    let source = SourceValue::new(
        partition_id(0),
        MirroredDtype::F32,
        MirroredStorageLayout::Dense,
        GENERATION + 1,
        vec![3u8; SOURCE_TOTAL_BYTES as usize],
    );
    let error = adapter
        .copy(&spec, &source, &partition_id(1))
        .expect_err("a generation mismatch must reject before copy");
    match error {
        TransferError::Rejected(rejection) => {
            assert!(matches!(
                rejection,
                TransferRejection::GenerationMismatch {
                    ref transfer,
                    declared: GENERATION,
                    actual,
                } if transfer.as_str() == "t-base" && actual == GENERATION + 1
            ));
            assert_eq!(rejection.class(), "generation mismatch");
        }
        other => panic!("expected Rejected, got {other:?}"),
    }
    assert_adapter_untouched(&adapter);
}

#[test]
fn owner_mismatch_rejects_before_copy() {
    let mut adapter = HostStagedAdapter::new(TransferBudget::declared(1 << 20, 1 << 30));
    let spec = base_spec(Duration::from_secs(1));
    let source = SourceValue::new(
        partition_id(1),
        MirroredDtype::F32,
        MirroredStorageLayout::Dense,
        GENERATION,
        vec![3u8; SOURCE_TOTAL_BYTES as usize],
    );
    let error = adapter
        .copy(&spec, &source, &partition_id(1))
        .expect_err("an owner mismatch must reject before copy");
    match error {
        TransferError::Rejected(rejection) => {
            assert!(matches!(
                rejection,
                TransferRejection::OwnerMismatch {
                    ref transfer,
                    declared: _,
                    actual: _,
                } if transfer.as_str() == "t-base"
            ));
            assert_eq!(rejection.class(), "owner mismatch");
            assert!(rejection.to_string().contains("owner mismatch"));
        }
        other => panic!("expected Rejected, got {other:?}"),
    }
    assert_adapter_untouched(&adapter);
}

#[test]
fn destination_mismatch_rejects_before_copy() {
    let mut adapter = HostStagedAdapter::new(TransferBudget::declared(1 << 20, 1 << 30));
    let spec = base_spec(Duration::from_secs(1));
    let source = base_source();
    let error = adapter
        .copy(&spec, &source, &partition_id(0))
        .expect_err("a destination mismatch must reject before copy");
    match error {
        TransferError::Rejected(rejection) => {
            assert!(matches!(
                rejection,
                TransferRejection::DestinationMismatch {
                    ref transfer,
                    declared: _,
                    actual: _,
                } if transfer.as_str() == "t-base"
            ));
            assert_eq!(rejection.class(), "destination mismatch");
            assert!(rejection.to_string().contains("destination mismatch"));
        }
        other => panic!("expected Rejected, got {other:?}"),
    }
    assert_adapter_untouched(&adapter);
}

// --- the host-staged adapter: labeled + timed, byte-exact --------------------

#[test]
fn host_staged_copy_is_labeled_timed_and_byte_exact() {
    let mut adapter = HostStagedAdapter::new(TransferBudget::declared(1 << 20, 1 << 30));
    adapter.set_simulated_delay(Duration::from_millis(1));
    let spec = base_spec(Duration::from_secs(1));
    let source = base_source();

    let outcome = adapter
        .copy(&spec, &source, &partition_id(1))
        .expect("a valid typed/ranged copy succeeds");
    let expected_bytes: Vec<u8> = source.bytes()[0..TRANSFER_RANGE_BYTES as usize].to_vec();
    assert_eq!(
        outcome.destination_bytes, expected_bytes,
        "the copy must be byte-exact"
    );

    let record = &outcome.record;
    assert_eq!(
        record.copy_path,
        CopyPath::HostStaged,
        "labeled host-staged"
    );
    assert_eq!(record.staging.capacity_bytes, TRANSFER_RANGE_BYTES);
    assert!(record.staging.pinned, "host staging is pinned host memory");
    assert_eq!(record.bytes, TRANSFER_RANGE_BYTES, "exact byte accounting");
    assert!(record.elapsed_nanos >= 1_000_000, "the copy is timed");
    assert_eq!(
        record.expected_nanos,
        expected_copy_time_nanos(
            TRANSFER_RANGE_BYTES,
            TransferDirectionMirror::BIDI,
            MeasuredRates::t1()
        )
    );
    assert_eq!(record.timeout, Duration::from_secs(1));
    assert_eq!(record.destination, partition_id(1));
    assert!(record.engine.get() >= 1, "the copy ran on an engine/stream");
    assert!(
        record.event.get() >= 1,
        "the copy recorded a completion event"
    );

    assert_eq!(adapter.used_bytes(), TRANSFER_RANGE_BYTES);
    assert_eq!(adapter.used_time_nanos(), record.expected_nanos);
    assert_eq!(adapter.selected_transfer_records().len(), 1);
    assert_eq!(
        adapter.staging_pool().active_count(),
        0,
        "in-flight staging is released at copy end"
    );

    // A second copy accumulates exactly.
    let second = adapter
        .copy(&spec, &source, &partition_id(1))
        .expect("the second copy succeeds");
    assert_eq!(adapter.used_bytes(), 2 * TRANSFER_RANGE_BYTES);
    assert_eq!(adapter.selected_transfer_records().len(), 2);
    assert!(
        second.record.engine > record.engine,
        "streams advance per copy"
    );
    assert!(
        second.record.event > record.event,
        "events advance per copy"
    );
}

#[test]
fn transport_receipt_records_path_staging_events_timeout_bytes_timing() {
    let mut adapter = HostStagedAdapter::new(TransferBudget::declared(1 << 20, 1 << 30));
    adapter.set_simulated_delay(Duration::from_millis(1));
    let spec = base_spec(Duration::from_secs(1));
    let source = base_source();
    adapter
        .copy(&spec, &source, &partition_id(1))
        .expect("first copy succeeds");
    adapter
        .copy(&spec, &source, &partition_id(1))
        .expect("second copy succeeds");

    let receipt: TransportReceipt = adapter.transport_receipt();
    assert_eq!(receipt.records.len(), 2);
    assert_eq!(receipt.used_bytes, 2 * TRANSFER_RANGE_BYTES);
    assert_eq!(receipt.budget_bytes, 1 << 20);
    assert_eq!(receipt.budget_time_nanos, 1 << 30);
    assert_eq!(receipt.rates, MeasuredRates::t1());
    for record in &receipt.records {
        // path
        assert_eq!(record.copy_path, CopyPath::HostStaged);
        // staging
        assert!(record.staging.pinned);
        assert_eq!(record.staging.capacity_bytes, TRANSFER_RANGE_BYTES);
        // streams/queues/events
        assert!(record.engine.get() >= 1);
        assert!(record.event.get() >= 1);
        // timeout/failure policy
        assert_eq!(record.timeout, Duration::from_secs(1));
        // bytes
        assert_eq!(record.bytes, TRANSFER_RANGE_BYTES);
        // timing
        assert!(record.elapsed_nanos >= 1_000_000);
        assert!(record.expected_nanos > 0);
    }
}

#[test]
fn copy_exceeding_declared_budget_is_rejected() {
    let mut adapter =
        HostStagedAdapter::new(TransferBudget::declared(TRANSFER_RANGE_BYTES, 1 << 30));
    let spec = base_spec(Duration::from_secs(1));
    let source = base_source();
    adapter
        .copy(&spec, &source, &partition_id(1))
        .expect("one copy fits the declared budget");
    let error = adapter
        .copy(&spec, &source, &partition_id(1))
        .expect_err("the second copy exceeds the declared transfer budget");
    assert!(matches!(
        error,
        TransferError::BudgetExceeded {
            transfer: _,
            budget_bytes: TRANSFER_RANGE_BYTES,
            used_bytes: TRANSFER_RANGE_BYTES,
            needed_bytes: TRANSFER_RANGE_BYTES,
        }
    ));
    assert_eq!(adapter.selected_transfer_records().len(), 1);
}

#[test]
fn fixture_measured_rate_is_recorded_and_rebudgeted() {
    let mut adapter = HostStagedAdapter::new(TransferBudget::declared(1 << 20, 1 << 30));
    let rates_before = adapter.rates();
    // A materially different fixture-measured H2D rate: 1 MiB in 100 ms is
    // ≈10 MB/s — an order of magnitude slower than the T1 constant, so the
    // recorded fixture-measured rate re-budgets future accounting.
    adapter.record_fixture_measurement(TransferDirectionMirror::H2D, 1_000_000, 100_000_000);
    assert_eq!(adapter.rates().h2d_bytes_per_sec, 10_000_000);
    assert_ne!(adapter.rates(), rates_before);
    assert_eq!(adapter.rate_observations().len(), 1);
    assert_eq!(
        adapter.rate_observations()[0].measured_bytes_per_sec,
        10_000_000
    );
    // Re-budgeted: the recorded rate now drives future budget-time accounting.
    let h2d_spec = TransferSpec::new(
        TransferRef::new("t-h2d"),
        partition_id(0),
        partition_id(1),
        ByteRange::new(0, TRANSFER_RANGE_BYTES),
        TransferDirectionMirror::H2D,
        MirroredDtype::F32,
        MirroredStorageLayout::Dense,
        GENERATION,
        Duration::from_secs(1),
    );
    adapter
        .copy(&h2d_spec, &base_source(), &partition_id(1))
        .expect("the re-budgeted copy succeeds");
    let record = &adapter.selected_transfer_records()[0];
    assert_eq!(
        record.expected_nanos,
        expected_copy_time_nanos(
            TRANSFER_RANGE_BYTES,
            TransferDirectionMirror::H2D,
            adapter.rates()
        ),
        "budget time is accounted at the recorded fixture-measured rate"
    );
}

// --- timeout/failure policy (S4) ---------------------------------------------

#[test]
fn copy_exceeding_declared_timeout_surfaces_transfer_error() {
    let mut adapter = HostStagedAdapter::new(TransferBudget::declared(1 << 20, 1 << 30));
    adapter.set_simulated_delay(Duration::from_millis(20));
    let spec = base_spec(Duration::from_millis(1));
    let source = base_source();
    let error = adapter
        .copy(&spec, &source, &partition_id(1))
        .expect_err("a copy past its declared deadline must time out");
    match error {
        TransferError::Timeout {
            transfer,
            declared_timeout,
            elapsed_nanos,
        } => {
            assert_eq!(transfer.as_str(), "t-base");
            assert_eq!(declared_timeout, Duration::from_millis(1));
            assert!(elapsed_nanos >= 20_000_000);
        }
        other => panic!("expected Timeout, got {other:?}"),
    }
    // A timed-out copy records nothing and leaks no staging.
    assert!(adapter.selected_transfer_records().is_empty());
    assert_eq!(adapter.staging_pool().active_count(), 0);
    assert_eq!(adapter.used_bytes(), 0);
}

#[test]
fn failed_copy_surfaces_transfer_error() {
    let mut adapter = HostStagedAdapter::new(TransferBudget::declared(1 << 20, 1 << 30));
    adapter.set_simulated_failure(Some("synthetic DMA failure".to_owned()));
    let spec = base_spec(Duration::from_secs(1));
    let source = base_source();
    let error = adapter
        .copy(&spec, &source, &partition_id(1))
        .expect_err("an injected mid-copy failure must surface a transfer error");
    assert!(matches!(
        error,
        TransferError::Failed { transfer, detail } if transfer.as_str() == "t-base"
            && detail == "synthetic DMA failure"
    ));
    assert!(adapter.selected_transfer_records().is_empty());
    assert_eq!(adapter.staging_pool().active_count(), 0);
}

#[test]
fn transfer_error_maps_into_coordinator_backend_error_vocabulary() {
    let timeout = TransferError::Timeout {
        transfer: TransferRef::new("t-x"),
        declared_timeout: Duration::from_millis(5),
        elapsed_nanos: 21_000_000,
    };
    assert!(matches!(
        timeout.into_backend_error(partition_id(0)),
        BackendError::Timeout { partition, .. } if partition == partition_id(0)
    ));
    let failed = TransferError::Failed {
        transfer: TransferRef::new("t-x"),
        detail: "boom".to_owned(),
    };
    assert!(matches!(
        failed.into_backend_error(partition_id(1)),
        BackendError::Operation { partition, .. } if partition == partition_id(1)
    ));
    let rejected = TransferError::Rejected(TransferRejection::DtypeMismatch {
        transfer: TransferRef::new("t-x"),
        declared: MirroredDtype::F32,
        actual: MirroredDtype::I32,
    });
    assert!(matches!(
        rejected.into_backend_error(partition_id(1)),
        BackendError::Operation { .. }
    ));
}

// --- the coordinator aborts on a transfer error ------------------------------

/// A `DeviceExecutionBackend` that performs transfer operations through the
/// host-staged adapter — the coordinator integration shape (the real wiring
/// lands at MD3-S1; the fake backend surface is X1's).
struct AdapterBackend {
    inner: FakeExecutionBackend,
    adapter: HostStagedAdapter,
    source_bytes: Vec<u8>,
    timeout: Duration,
}

impl AdapterBackend {
    fn new(source_bytes: Vec<u8>, timeout: Duration, budget: TransferBudget) -> Self {
        Self {
            inner: FakeExecutionBackend::new(),
            adapter: HostStagedAdapter::new(budget),
            source_bytes,
            timeout,
        }
    }
}

impl DeviceExecutionBackend for AdapterBackend {
    fn reserve(
        &mut self,
        partition: &LogicalPartitionId,
        reservation: &ReservationRecord,
    ) -> Result<(), BackendError> {
        self.inner.reserve(partition, reservation)
    }

    fn run_operation(&mut self, operation: &TransactionOperation) -> Result<(), BackendError> {
        if let TransactionOperation::Transfer(mirror) = operation {
            let spec = TransferSpec::from_mirror(mirror, self.timeout);
            let source = SourceValue::new(
                mirror.source().clone(),
                mirror.element_dtype(),
                mirror.layout(),
                mirror.producer_generation(),
                self.source_bytes.clone(),
            );
            if let Err(error) = self.adapter.copy(&spec, &source, mirror.destination()) {
                let backend_error = error.into_backend_error(mirror.destination().clone());
                return Err(backend_error);
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

/// A transaction whose snapshot is one transfer (p0 → p1, 1024 bytes BIDI) —
/// it fits the fixture's declared class 6/class 3 budgets at prepare.
fn transfer_only_transaction(byte_count: u64) -> ExecutionTransaction {
    ExecutionTransaction::new(
        TransactionId::new("txn-t1"),
        fixture_plan(),
        vec![transfer_op("t-transfer", byte_count)],
        TransactionCommitBoundary::default(),
    )
    .expect("transfer-only transaction constructs")
}

#[test]
fn timed_out_copy_surfaces_transfer_error_the_coordinator_aborts_on() {
    let mut transaction = transfer_only_transaction(TRANSFER_ONLY_BYTES);
    let mut backend = AdapterBackend::new(
        vec![7u8; TRANSFER_ONLY_BYTES as usize],
        Duration::from_millis(5),
        TransferBudget::declared(1 << 20, 1 << 30),
    );
    backend
        .adapter
        .set_simulated_delay(Duration::from_millis(50));
    transaction.prepare(&mut backend).expect("prepare succeeds");
    let error = transaction
        .execute(&mut backend)
        .expect_err("the timed-out copy fails execute");
    assert!(matches!(
        error,
        ExecuteError::Backend(BackendError::Timeout { .. })
    ));
    assert!(matches!(transaction.state(), TransactionState::Failed(_)));

    // Q8 fail-closed: abort retires/releases everything, publishes nothing.
    let receipt = transaction
        .abort(&mut backend, "timed-out transfer")
        .expect("abort completes teardown");
    assert!(matches!(transaction.state(), TransactionState::Aborted(_)));
    assert!(receipt.publish_summary.is_none());
    assert!(!receipt.teardown.partial_publication);
    assert_eq!(backend.inner.published_bytes(), 0);
}

#[test]
fn failed_copy_surfaces_transfer_error_the_coordinator_aborts_on() {
    let mut transaction = transfer_only_transaction(TRANSFER_ONLY_BYTES);
    let mut backend = AdapterBackend::new(
        vec![7u8; TRANSFER_ONLY_BYTES as usize],
        Duration::from_secs(1),
        TransferBudget::declared(1 << 20, 1 << 30),
    );
    backend
        .adapter
        .set_simulated_failure(Some("synthetic DMA failure".to_owned()));
    transaction.prepare(&mut backend).expect("prepare succeeds");
    let error = transaction
        .execute(&mut backend)
        .expect_err("the failed copy fails execute");
    assert!(matches!(
        error,
        ExecuteError::Backend(BackendError::Operation { .. })
    ));
    assert!(matches!(transaction.state(), TransactionState::Failed(_)));

    let receipt = transaction
        .abort(&mut backend, "failed transfer")
        .expect("abort completes teardown");
    assert!(matches!(transaction.state(), TransactionState::Aborted(_)));
    assert!(receipt.publish_summary.is_none());
    assert!(!receipt.teardown.partial_publication);
    assert_eq!(backend.inner.published_bytes(), 0);
}

// --- the explicit peer path (NOT ATTEMPTED, per-pair flip rule) ---------------

#[test]
fn peer_transfer_on_unmeasured_pair_is_rejected() {
    let mut peer = PeerAdapter::new();
    let spec = base_spec(Duration::from_secs(1));
    let source = base_source();
    let error = peer
        .copy(&spec, &source, &partition_id(1))
        .expect_err("a peer transfer on an unmeasured pair is rejected (NOT ATTEMPTED)");
    match error {
        TransferError::PeerNotAdmitted { pair } => {
            assert_eq!(
                pair,
                DirectedPair::new(partition_id(0), partition_id(1)),
                "the rejection names the exact directed pair"
            );
            assert!(peer.selected_transfer_records().is_empty());
        }
        other => panic!("expected PeerNotAdmitted, got {other:?}"),
    }
}

#[test]
fn peer_admission_requires_a_real_measurement() {
    let mut peer = PeerAdapter::new();
    let pair = DirectedPair::new(partition_id(0), partition_id(1));
    let error = peer
        .admit_pair(PeerPairMeasurement::new(
            pair.clone(),
            "probe",
            10_000_000_000,
            500,
            "",
        ))
        .expect_err("a measurement without an evidence reference is not admitted (T1 §7)");
    assert!(matches!(error, PairAdmissionError::MissingEvidence { .. }));
    assert!(!peer.pair_admitted(&pair));

    let self_pair = DirectedPair::new(partition_id(0), partition_id(0));
    let error = peer
        .admit_pair(PeerPairMeasurement::new(
            self_pair.clone(),
            "probe",
            10_000_000_000,
            500,
            "ev:sha256:deadbeef",
        ))
        .expect_err("a self pair is not a P2P row (T1 §2.1)");
    assert!(matches!(error, PairAdmissionError::SelfPair { .. }));
    assert!(!peer.pair_admitted(&self_pair));
}

#[test]
fn admitted_pair_flip_is_per_directed_pair_never_global() {
    let mut registry = PeerPairRegistry::new();
    let ab = DirectedPair::new(partition_id(0), partition_id(1));
    let ba = DirectedPair::new(partition_id(1), partition_id(0));
    let ac = DirectedPair::new(partition_id(0), partition_id(2));
    assert_eq!(select_copy_path(&ab, &registry), CopyPath::HostStaged);

    registry
        .admit(PeerPairMeasurement::new(
            ab.clone(),
            "synthetic probe re-run (mechanism test)",
            25_000_000_000,
            500,
            "ev:sha256:deadbeef",
        ))
        .expect("a measured pair admits");

    assert_eq!(
        select_copy_path(&ab, &registry),
        CopyPath::Peer,
        "the measured pair flips"
    );
    assert_eq!(
        select_copy_path(&ba, &registry),
        CopyPath::HostStaged,
        "the reverse pair stays host-staged"
    );
    assert_eq!(
        select_copy_path(&ac, &registry),
        CopyPath::HostStaged,
        "an unrelated pair stays host-staged"
    );
    assert_eq!(registry.admitted_pairs().len(), 1, "never a global switch");

    // Even an admitted pair is NOT ATTEMPTED by the peer adapter: the flip is
    // recorded, the execution row stays NOT ATTEMPTED (lane queue §5b) —
    // never a fabricated peer pass.
    let mut peer = PeerAdapter::new();
    peer.admit_pair(PeerPairMeasurement::new(
        ab.clone(),
        "synthetic probe re-run (mechanism test)",
        25_000_000_000,
        500,
        "ev:sha256:deadbeef",
    ))
    .expect("admits");
    let spec = base_spec(Duration::from_secs(1));
    let source = base_source();
    let error = peer
        .copy(&spec, &source, &partition_id(1))
        .expect_err("an admitted pair is still NOT ATTEMPTED");
    assert!(matches!(
        error,
        TransferError::PeerNotAttempted { pair, .. } if pair == ab
    ));
}

// --- the mirror/admissibility separation (S4) --------------------------------

#[test]
fn logical_plan_never_carries_the_selected_transport() {
    // The portable logical plan's transport surface is exactly the
    // admissibility label. The exhaustive match is the structural claim:
    // `TransportPathMirror` v1 = {host-staged} — a selected peer path cannot
    // be expressed in the logical plan at all.
    let mirror = transfer_mirror("t-structural", TRANSFER_RANGE_BYTES);
    let path_label = mirror.path_label();
    let spelling = match path_label {
        TransportPathMirror::HostStaged => path_label.spelling(),
    };
    assert_eq!(spelling, "host-staged");

    // The logical plan's canonical bytes are selection-independent: a runtime
    // copy (which mints staging/engine/event/timing facts) never changes
    // them — the selected transport never absorbs into the portable plan.
    let before = mirror.canonical_bytes();
    let mut adapter = HostStagedAdapter::new(TransferBudget::declared(1 << 20, 1 << 30));
    let spec = TransferSpec::from_mirror(&mirror, Duration::from_secs(1));
    let source = SourceValue::new(
        partition_id(0),
        MirroredDtype::F32,
        MirroredStorageLayout::Dense,
        0,
        vec![7u8; TRANSFER_RANGE_BYTES as usize],
    );
    adapter
        .copy(&spec, &source, &partition_id(1))
        .expect("host-staged copy succeeds");
    assert_eq!(
        mirror.canonical_bytes(),
        before,
        "the portable logical plan must not absorb selected-transport facts"
    );
    assert_eq!(
        adapter.selected_transfer_records()[0].copy_path,
        CopyPath::HostStaged,
        "the selected transport records only to the runtime receipt section"
    );

    // The runtime declared-facts carrier (the spec) also carries only the
    // admissibility label — no selected-path surface.
    assert_eq!(spec.path_label(), TransportPathMirror::HostStaged);
    // `validate_before_copy` consumes declared + actual facts only; no
    // selected-transport input exists in the fail-before-copy surface.
    assert!(validate_before_copy(&spec, &source, &partition_id(1)).is_ok());
}

// --- the S4 section and staging bookkeeping ----------------------------------

#[test]
fn staging_pool_accounts_allocations_and_releases() {
    let mut pool = StagingPool::new();
    assert_eq!(pool.active_count(), 0);
    assert_eq!(pool.allocations(), 0);
    let buffer = pool.allocate(1024);
    assert!(buffer.pinned);
    assert_eq!(buffer.capacity_bytes, 1024);
    assert_eq!(pool.active_count(), 1);
    assert_eq!(pool.allocations(), 1);
    pool.release(buffer.id.clone());
    assert_eq!(pool.active_count(), 0);
    assert_eq!(pool.allocations(), 1, "allocations count is monotonic");
}
