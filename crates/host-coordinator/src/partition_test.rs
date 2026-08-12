//! MD1-V1 partition contract tests: identity-class distinctness, lifecycle,
//! the complete eight-class budget ledger, the three-class failure taxonomy,
//! the MD-A15 one-device degenerate, software-admission wording, and the
//! receipt taxonomy.

use crate::device_identity::PhysicalDeviceId;
use crate::partition::{
    AdmissionError, AdmissionRequest, FixtureIdentityClass, HardwareIsolationClaim,
    PartitionBudgetLedger, PartitionFailure, PartitionReceipt, PartitionState, SafePhysicalLimit,
    TransportClass, VirtualDevicePartition, VirtualDevicePartitionId,
};
use std::collections::BTreeSet;

const UUID_A: &str = "GPU-3e017562-9ec3-da9a-962d-b8bd5f9e24be";

fn cuda_device() -> PhysicalDeviceId {
    PhysicalDeviceId::cuda(UUID_A, None)
}

/// A weight-only ledger (all other classes zero) for lifecycle tests.
fn ledger(weight_bytes: u64) -> PartitionBudgetLedger {
    PartitionBudgetLedger {
        weight_bytes,
        kv_cache_bytes: 0,
        activation_scratch_bytes: 0,
        module_storage_bytes: 0,
        allocator_overhead_bytes: 0,
        transfer_staging_bytes: 0,
        concurrent_state_bytes: 0,
    }
}

/// A partition binds exactly one `PhysicalDeviceId`, and the
/// `VirtualDevicePartitionId` identity class is distinct — a virtual
/// partition never receives a physical id.
#[test]
fn partition_binds_exactly_one_physical_device_and_id_class_is_distinct() {
    let device = cuda_device();
    let id = VirtualDevicePartitionId::new(1);

    let partition = VirtualDevicePartition::admit(
        AdmissionRequest::new(id, device.clone(), ledger(100)),
        SafePhysicalLimit::new(1000),
    )
    .unwrap();

    // Exactly one physical device is bound: `bound_device` is a single value,
    // and the partition's public surface exposes exactly that one device.
    assert_eq!(partition.bound_device(), &device);
    assert_eq!(*partition.bound_device(), device);

    // The partition id is the virtual identity class; it is never the bound
    // physical id. The two are distinct types with no conversion — a
    // `VirtualDevicePartitionId` can never be passed as a
    // `PhysicalDeviceId` (type-level separation) and no equality exists
    // between them.
    assert_eq!(partition.id(), id);
    assert_eq!(partition.id().get(), 1);

    // The display domains are disjoint, pinning the separation at the
    // string level too.
    assert_eq!(partition.id().to_string(), "vp:1");
    assert!(partition.bound_device().to_string().starts_with("cuda:"));
}

/// Declared requirements exceeding the partition policy limit are rejected
/// at admission as `budget_exceeded` — deterministic fail-closed, before any
/// allocation.
#[test]
fn declared_requirements_over_policy_limit_rejected_as_budget_exceeded() {
    let err = VirtualDevicePartition::admit(
        AdmissionRequest::new(VirtualDevicePartitionId::new(1), cuda_device(), ledger(100)),
        SafePhysicalLimit::new(99),
    )
    .unwrap_err();
    assert_eq!(
        err,
        AdmissionError::BudgetExceeded {
            declared_total_bytes: Some(100),
            policy_limit_bytes: 99,
        }
    );

    // Exactly at the limit admits (boundary).
    let request =
        AdmissionRequest::new(VirtualDevicePartitionId::new(2), cuda_device(), ledger(100));
    assert!(VirtualDevicePartition::admit(request, SafePhysicalLimit::new(100)).is_ok());
}

/// Admission is gated by the safe physical limit *policy*, never by total
/// reported memory — the two are never conflated (nvidia-smi MiB totals and
/// driver-API byte totals are device facts, not the admission limit).
#[test]
fn admission_is_policy_gated_never_total_reported_memory() {
    // The declared budget is tiny next to the pharos device's reported
    // totals, yet a small policy ceiling still rejects: the gate is the
    // policy, not the memory report.
    let err = VirtualDevicePartition::admit(
        AdmissionRequest::new(
            VirtualDevicePartitionId::new(1),
            cuda_device(),
            ledger(1_000_000),
        ),
        SafePhysicalLimit::new(500_000),
    )
    .unwrap_err();
    assert!(matches!(
        err,
        AdmissionError::BudgetExceeded {
            declared_total_bytes: Some(1_000_000),
            policy_limit_bytes: 500_000,
        }
    ));

    // The partition API never accepts a device memory report at all — the
    // only limit input is the policy type.
    let request = AdmissionRequest::new(
        VirtualDevicePartitionId::new(2),
        cuda_device(),
        ledger(10_000),
    );
    assert!(VirtualDevicePartition::admit(request, SafePhysicalLimit::new(20_000)).is_ok());
}

/// A post-admission allocation failure maps to `allocation_failure` — a
/// `PartitionFailure`, never an `AdmissionError` (admission already passed).
#[test]
fn post_admission_allocation_failure_maps_to_allocation_failure() {
    let request =
        AdmissionRequest::new(VirtualDevicePartitionId::new(1), cuda_device(), ledger(100));
    let mut partition =
        VirtualDevicePartition::admit(request, SafePhysicalLimit::new(1000)).unwrap();
    assert!(partition.is_active());

    partition.record_allocation_failure("cuMemAlloc failed under physical pressure");
    assert_eq!(
        partition.state(),
        &PartitionState::Failed(PartitionFailure::AllocationFailure {
            detail: "cuMemAlloc failed under physical pressure".to_owned(),
        })
    );
    assert!(matches!(
        partition.failure(),
        Some(PartitionFailure::AllocationFailure { .. })
    ));
}

/// A removed/failed bound device maps to `device_loss` (MD-A13).
#[test]
fn removed_bound_device_maps_to_device_loss() {
    let mut partition = VirtualDevicePartition::admit(
        AdmissionRequest::new(VirtualDevicePartitionId::new(1), cuda_device(), ledger(100)),
        SafePhysicalLimit::new(1000),
    )
    .unwrap();

    partition.report_device_loss("device removed; health epoch advanced");
    assert_eq!(
        partition.state(),
        &PartitionState::Failed(PartitionFailure::DeviceLoss {
            detail: "device removed; health epoch advanced".to_owned(),
        })
    );
    assert!(matches!(
        partition.failure(),
        Some(PartitionFailure::DeviceLoss { .. })
    ));
}

/// The three failure classes are distinct and never conflated:
/// `budget_exceeded` is the sole admission-time class, while
/// `allocation_failure` and `device_loss` are the sole post-admission
/// classes, and the first recorded failure wins.
#[test]
fn three_failure_classes_are_distinct_and_never_conflated() {
    // BudgetExceeded is admission-only: the AdmissionError type is the only
    // admission outcome, and PartitionFailure (the post-admission type) has
    // no BudgetExceeded variant.
    let err = VirtualDevicePartition::admit(
        AdmissionRequest::new(
            VirtualDevicePartitionId::new(1),
            cuda_device(),
            ledger(10_000),
        ),
        SafePhysicalLimit::new(100),
    )
    .unwrap_err();
    assert!(matches!(err, AdmissionError::BudgetExceeded { .. }));

    // A ledger whose classes overflow u64 is still fail-closed as
    // BudgetExceeded (declared total unknown, deterministic rejection).
    let overflowing = PartitionBudgetLedger {
        weight_bytes: u64::MAX,
        kv_cache_bytes: 2,
        ..ledger(0)
    };
    let err = VirtualDevicePartition::admit(
        AdmissionRequest::new(VirtualDevicePartitionId::new(2), cuda_device(), overflowing),
        SafePhysicalLimit::new(u64::MAX),
    )
    .unwrap_err();
    assert!(matches!(
        err,
        AdmissionError::BudgetExceeded {
            declared_total_bytes: None,
            ..
        }
    ));

    // Post-admission: allocation_failure first, then device_loss — the first
    // failure wins, so an allocation failure is never relabeled as a device
    // loss.
    let request =
        AdmissionRequest::new(VirtualDevicePartitionId::new(3), cuda_device(), ledger(10));
    let mut p = VirtualDevicePartition::admit(request, SafePhysicalLimit::new(100)).unwrap();
    p.record_allocation_failure("oom");
    p.report_device_loss("lost later");
    assert!(matches!(
        p.failure(),
        Some(PartitionFailure::AllocationFailure { .. })
    ));

    // And a device loss is never downgraded to an allocation failure.
    let request =
        AdmissionRequest::new(VirtualDevicePartitionId::new(4), cuda_device(), ledger(10));
    let mut q = VirtualDevicePartition::admit(request, SafePhysicalLimit::new(100)).unwrap();
    q.report_device_loss("lost");
    q.record_allocation_failure("oom after loss");
    assert!(matches!(
        q.failure(),
        Some(PartitionFailure::DeviceLoss { .. })
    ));
}

/// MD-A15 degenerate: one-device execution derives an implicit/local
/// partition for admission/GI5 accounting with no distributed wrapper — the
/// type surface carries no distributed plan, transfer graph, or execution
/// transaction, and the constructor accepts none.
#[test]
fn one_device_degenerate_builds_implicit_local_partition_without_distributed_wrapper() {
    let device = cuda_device();
    let mut partition = VirtualDevicePartition::implicit_local(
        VirtualDevicePartitionId::new(1),
        device.clone(),
        ledger(42),
        SafePhysicalLimit::new(1000),
    )
    .unwrap();

    assert!(partition.is_active());
    assert_eq!(partition.bound_device(), &device);
    assert_eq!(partition.ledger().weight_bytes, 42);
    assert_eq!(partition.admitted_total_bytes(), Some(42));

    // Teardown completes the lifecycle; it is idempotent.
    partition.teardown();
    assert_eq!(partition.state(), &PartitionState::TornDown);
    partition.teardown();
    assert_eq!(partition.state(), &PartitionState::TornDown);
}

/// The ledger covers all eight byte classes — seven concrete fields plus the
/// safe physical limit policy as class 8.
#[test]
fn ledger_covers_all_eight_byte_classes() {
    let full = PartitionBudgetLedger {
        weight_bytes: 258_000_000,           // (1) weights incl. repack/duplication
        kv_cache_bytes: 4_000_000,           // (2) KV per KvCacheLayout (consumed, GI4-owned)
        activation_scratch_bytes: 2_000_000, // (3) peak activations + scratch
        module_storage_bytes: 1_000_000,     // (4) module/kernel/descriptor storage
        allocator_overhead_bytes: 500_000,   // (5) granularity/alignment/headroom
        transfer_staging_bytes: 3_000_000,   // (6) transfer/staging + in-flight
        concurrent_state_bytes: 100_000,     // (7) concurrent requests/models
    };
    assert_eq!(
        full.total_bytes(),
        Some(258_000_000 + 4_000_000 + 2_000_000 + 1_000_000 + 500_000 + 3_000_000 + 100_000)
    );

    // Class 8 is the safe physical limit policy — a distinct, policy-declared
    // ceiling, never a device memory report.
    let limit = SafePhysicalLimit::new(300_000_000);
    assert_eq!(limit.get(), 300_000_000);

    // Admission uses the full ledger against the policy limit.
    let request = AdmissionRequest::new(VirtualDevicePartitionId::new(1), cuda_device(), full);
    assert!(VirtualDevicePartition::admit(request, limit).is_ok());
}

/// The receipt taxonomy fields serialize deterministically, every field
/// participates, and counts derive from the id sets.
#[test]
fn receipt_taxonomy_fields_serialize_deterministically() {
    let devices = BTreeSet::from([cuda_device()]);
    let ids = BTreeSet::from([
        VirtualDevicePartitionId::new(7),
        VirtualDevicePartitionId::new(3),
    ]);
    let receipt = PartitionReceipt::new(
        devices.clone(),
        ids.clone(),
        FixtureIdentityClass::Virtual,
        TransportClass::HostStaged,
    );

    // Every taxonomy field is present.
    assert_eq!(receipt.physical_device_count(), 1);
    assert_eq!(receipt.virtual_partition_count(), 2);
    assert_eq!(receipt.physical_device_ids(), &devices);
    assert_eq!(receipt.virtual_partition_ids(), &ids);
    assert_eq!(
        receipt.fixture_identity_class(),
        FixtureIdentityClass::Virtual
    );
    assert_eq!(receipt.transport_class(), TransportClass::HostStaged);
    assert_eq!(
        receipt.hardware_isolation_claimed(),
        HardwareIsolationClaim::NotClaimed
    );

    // Deterministic serialization: identical receipts → identical bytes.
    let bytes = receipt.canonical_bytes();
    let again = PartitionReceipt::new(
        devices.clone(),
        ids.clone(),
        FixtureIdentityClass::Virtual,
        TransportClass::HostStaged,
    );
    assert_eq!(bytes, again.canonical_bytes());

    // Field sensitivity: changing any taxonomy field changes the bytes.
    assert_ne!(
        bytes,
        PartitionReceipt::new(
            BTreeSet::new(),
            ids.clone(),
            FixtureIdentityClass::Virtual,
            TransportClass::HostStaged,
        )
        .canonical_bytes(),
        "physical_device_ids must participate"
    );
    assert_ne!(
        bytes,
        PartitionReceipt::new(
            devices.clone(),
            BTreeSet::new(),
            FixtureIdentityClass::Virtual,
            TransportClass::HostStaged,
        )
        .canonical_bytes(),
        "virtual_partition_ids must participate"
    );
    assert_ne!(
        bytes,
        PartitionReceipt::new(
            devices.clone(),
            ids.clone(),
            FixtureIdentityClass::Physical,
            TransportClass::HostStaged,
        )
        .canonical_bytes(),
        "fixture_identity_class must participate"
    );
    assert_ne!(
        bytes,
        PartitionReceipt::new(
            devices.clone(),
            ids.clone(),
            FixtureIdentityClass::Virtual,
            TransportClass::DirectedPeerNotAttempted,
        )
        .canonical_bytes(),
        "transport_class must participate"
    );

    // Physical and virtual identity classes stay distinct in the receipt:
    // the serialized virtual ids live in their own field, separate from the
    // physical id bytes.
    assert_ne!(
        PartitionReceipt::new(
            devices.clone(),
            ids.clone(),
            FixtureIdentityClass::Virtual,
            TransportClass::None,
        )
        .canonical_bytes(),
        PartitionReceipt::new(
            BTreeSet::from([cuda_device()]),
            BTreeSet::from([VirtualDevicePartitionId::new(3)]),
            FixtureIdentityClass::Virtual,
            TransportClass::None,
        )
        .canonical_bytes(),
    );
}

/// Software admission wording is enforced in the type surface:
/// `hardware_isolation_claimed=false` is the only representable value and
/// serializes as false.
#[test]
fn hardware_isolation_claimed_is_always_false_in_the_type_surface() {
    // HardwareIsolationClaim has exactly one value — NotClaimed. There is no
    // way to construct a "claimed" claim.
    let claim = HardwareIsolationClaim::NotClaimed;
    let claimed = match claim {
        HardwareIsolationClaim::NotClaimed => false,
    };
    assert!(!claimed);

    let receipt = PartitionReceipt::new(
        BTreeSet::from([cuda_device()]),
        BTreeSet::from([VirtualDevicePartitionId::new(1)]),
        FixtureIdentityClass::Virtual,
        TransportClass::None,
    );
    assert_eq!(
        receipt.hardware_isolation_claimed(),
        HardwareIsolationClaim::NotClaimed
    );

    // The serialized claim byte is false (canonical encoding writes the
    // isolation claim last).
    let bytes = receipt.canonical_bytes();
    assert_eq!(bytes.last(), Some(&0u8));
}
