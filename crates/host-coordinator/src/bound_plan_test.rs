//! MD2-B1 bound-plan tests: the bind contract (admit-marker enforcement,
//! health-epoch device rejection, topology mismatch, constraint violations),
//! the T2 §5 synthetic two-partition fixture determinism, the MD-A15
//! single-partition degenerate, the bound-plan hash domain, and the receipt
//! taxonomy (`hardware_isolation_claimed=false`).

use crate::backend::DeviceBackend;
use crate::bound_plan::{
    bind, AdmitError, AdmittedLogicalPlan, BindError, BoundDistributedPlan, BoundPlanKind,
    DeclaredPlacementConstraint, LogicalPartitionId, PartitionBinding,
};
use crate::device_identity::{DeviceHealthGeneration, DeviceOrdinal, PhysicalDeviceId};
use crate::device_set::{DeviceSet, MembershipError};
use crate::discovery::{
    ComputeCapability, DeviceCapabilities, DeviceDiscoveryEntry, DeviceDiscoverySnapshot,
    DeviceHealth, DeviceMemory, DtypeSurface, P2pProbeState, ProbeProvenance,
};
use crate::partition::{
    AdmissionRequest, FixtureIdentityClass, HardwareIsolationClaim, PartitionBudgetLedger,
    SafePhysicalLimit, TransportClass, VirtualDevicePartition, VirtualDevicePartitionId,
};
use std::collections::{BTreeMap, BTreeSet};

// T1 measured facts (pharos) reused for the synthetic snapshot shape.
const UUID_A: &str = "GPU-3e017562-9ec3-da9a-962d-b8bd5f9e24be";
const UUID_B: &str = "GPU-22222222-3333-4444-5555-666666666666";
const UUID_C: &str = "GPU-88888888-9999-aaaa-bbbb-cccccccccccc";
const PROBE_TIME: u64 = 1_752_717_600_000_000_000; // fixed sample time
                                                   // An admitted (validated) logical hash in the sha256: spelling (FC17/FC11).
const LOGICAL_HASH: &str =
    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

// CS-1 declared placement (md0-mode-fixtures.md §3): 2 virtual partitions @
// 160 MiB, forced 2-way split ≈129 MiB/device.
const CS1_SPLIT_BYTES: u64 = 135_266_304; // ≈129 MiB per partition
const CS1_LIMIT_BYTES: u64 = 167_772_160; // 160 MiB safe physical limit policy

fn device_a() -> PhysicalDeviceId {
    PhysicalDeviceId::cuda(UUID_A, None)
}

fn device_b() -> PhysicalDeviceId {
    PhysicalDeviceId::cuda(UUID_B, None)
}

fn device_c() -> PhysicalDeviceId {
    PhysicalDeviceId::cuda(UUID_C, None)
}

fn partition_id(n: u32) -> LogicalPartitionId {
    LogicalPartitionId::new(format!("partition-{n}"))
}

/// A weight-only ledger (all other classes zero).
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

/// An admitted virtual partition over `device` under the CS-1 shape.
fn vp(seed: u64, device: PhysicalDeviceId) -> VirtualDevicePartition {
    VirtualDevicePartition::admit(
        AdmissionRequest::new(
            VirtualDevicePartitionId::new(seed),
            device,
            ledger(CS1_SPLIT_BYTES),
        ),
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
    snapshot_at(PROBE_TIME, entries)
}

fn snapshot_at(
    probe_utc_nanos: u64,
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
    DeviceDiscoverySnapshot::new(probe_utc_nanos, devices, P2pProbeState::NotAttempted)
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

fn three_device_snapshot() -> DeviceDiscoverySnapshot {
    snapshot_with([
        (0, device_a(), DeviceHealthGeneration::initial()),
        (1, device_b(), DeviceHealthGeneration::initial()),
        (2, device_c(), DeviceHealthGeneration::initial()),
    ])
}

/// A two-partition binding map from two (device, optional partition) pairs.
fn two_partition_bindings(
    first: (PhysicalDeviceId, Option<VirtualDevicePartition>),
    second: (PhysicalDeviceId, Option<VirtualDevicePartition>),
) -> BTreeMap<LogicalPartitionId, PartitionBinding> {
    let mut bindings = BTreeMap::new();
    bindings.insert(partition_id(0), binding(first));
    bindings.insert(partition_id(1), binding(second));
    bindings
}

fn binding(
    (device, partition): (PhysicalDeviceId, Option<VirtualDevicePartition>),
) -> PartitionBinding {
    match partition {
        Some(p) => PartitionBinding::with_virtual_partition(device, p),
        None => PartitionBinding::new(device),
    }
}

/// The done-when admitted plan: a validated logical hash + 2 declared
/// partitions, colocated on the single physical device (T2 §5).
fn admitted_two_partition_plan() -> AdmittedLogicalPlan {
    AdmittedLogicalPlan::admit(
        LOGICAL_HASH,
        [partition_id(0), partition_id(1)],
        [DeclaredPlacementConstraint::Colocated {
            partitions: BTreeSet::from([partition_id(0), partition_id(1)]),
        }],
    )
    .expect("valid admitted two-partition plan")
}

/// The bind call shape used by most tests (synthetic fixture, host-staged
/// transport, current health epoch).
fn bind_plan(
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

/// A stale or unadmitted logical hash is rejected at the admission path —
/// the bind API consumes only an [`AdmittedLogicalPlan`] marker, so a
/// `BoundDistributedPlan` can never be built from one.
#[test]
fn admit_rejects_stale_or_unadmitted_logical_hash() {
    for bad in [
        String::new(),
        "sha256:".to_owned(),
        "sha256:ABC".to_owned(),
        format!("sha256:{}", "A".repeat(64)), // uppercase hex — not the contract digest
        format!("sha256:{}", "a".repeat(63)), // wrong length
        format!("sha256:{}", "z".repeat(64)), // non-hex
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n".to_owned(),
        "md5:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
        "fnv64:0000000000000000".to_owned(),
    ] {
        let err = AdmittedLogicalPlan::admit(bad, [partition_id(0)], []).unwrap_err();
        assert!(
            matches!(err, AdmitError::UnadmittedLogicalHash { .. }),
            "expected UnadmittedLogicalHash"
        );
    }

    // The valid sha256: + 64-lowercase-hex spelling admits.
    let admitted = AdmittedLogicalPlan::admit(LOGICAL_HASH, [partition_id(0)], []);
    assert!(admitted.is_ok());
}

/// The admission path rejects an empty declared partition set.
#[test]
fn admit_rejects_empty_declared_partition_set() {
    let err = AdmittedLogicalPlan::admit(LOGICAL_HASH, [], []).unwrap_err();
    assert_eq!(err, AdmitError::NoDeclaredPartitions);
}

/// A declared constraint referencing an undeclared partition is rejected at
/// admission.
#[test]
fn admit_rejects_constraint_referencing_undeclared_partition() {
    let err = AdmittedLogicalPlan::admit(
        LOGICAL_HASH,
        [partition_id(0)],
        [DeclaredPlacementConstraint::Colocated {
            partitions: BTreeSet::from([partition_id(5)]),
        }],
    )
    .unwrap_err();
    assert_eq!(
        err,
        AdmitError::ConstraintReferencesUnknownPartition {
            partition: partition_id(5)
        }
    );
}

/// The done-when fixture (T2 §5 smaller proof boundary): a logical hash +
/// 2-partition binding to a synthetic 2-partition virtual `DeviceSet`
/// produces a deterministic `BoundDistributedPlan` + `bound_distributed_plan_hash`.
#[test]
fn synthetic_two_partition_bind_is_deterministic() {
    let snapshot = one_device_snapshot();
    let admitted = admitted_two_partition_plan();

    let bindings = two_partition_bindings(
        (device_a(), Some(vp(1, device_a()))),
        (device_a(), Some(vp(2, device_a()))),
    );
    let device_set = DeviceSet::from_members([device_a()]);

    let plan = bind_plan(&admitted, bindings, device_set, &snapshot)
        .expect("synthetic two-partition fixture binds");

    // Identical inputs → identical plan + identical bound hash.
    let again = bind_plan(
        &admitted,
        two_partition_bindings(
            (device_a(), Some(vp(1, device_a()))),
            (device_a(), Some(vp(2, device_a()))),
        ),
        DeviceSet::from_members([device_a()]),
        &snapshot,
    )
    .unwrap();
    assert_eq!(plan, again);
    assert_eq!(
        plan.bound_distributed_plan_hash(),
        again.bound_distributed_plan_hash()
    );

    // Bound hash spelling: sha256: + 64 lowercase hex (FC17/FC11).
    let hash = plan.bound_distributed_plan_hash();
    assert!(hash.starts_with("sha256:"));
    assert_eq!(hash.len(), "sha256:".len() + 64);
    assert!(
        hash["sha256:".len()..]
            .chars()
            .all(|c| c.is_ascii_hexdigit()),
        "bound hash digest must be lowercase hex"
    );

    // The admitted logical hash is preserved verbatim (opaque, FC18).
    assert_eq!(plan.logical_distributed_plan_hash(), LOGICAL_HASH);

    // Distributed kind with both bindings; not the MD-A15 degenerate.
    assert!(matches!(plan.kind(), BoundPlanKind::Distributed { .. }));
    assert!(!plan.is_degenerate());
    assert_eq!(
        plan.bindings()
            .expect("distributed plan has bindings")
            .len(),
        2
    );

    // Bound device set + content-addressed snapshot id recorded.
    assert_eq!(plan.device_set(), &DeviceSet::from_members([device_a()]));
    assert_eq!(plan.snapshot_id(), snapshot.id());

    // Receipt: 2 virtual partitions over 1 physical device, isolation not
    // claimed, host-staged transport.
    let receipt = plan.receipt();
    assert_eq!(receipt.physical_device_count(), 1);
    assert_eq!(receipt.physical_device_ids(), &BTreeSet::from([device_a()]));
    assert_eq!(receipt.virtual_partition_count(), 2);
    assert_eq!(
        receipt.virtual_partition_ids(),
        &BTreeSet::from([
            VirtualDevicePartitionId::new(1),
            VirtualDevicePartitionId::new(2)
        ])
    );
    assert_eq!(
        receipt.fixture_identity_class(),
        FixtureIdentityClass::Synthetic
    );
    assert_eq!(receipt.transport_class(), TransportClass::HostStaged);
}

/// The two-physical-device shape: distinct-device constraint + two virtual
/// partitions, one per device, serializes with both identity classes kept in
/// their own receipt fields — virtual partitions never receive a
/// `PhysicalDeviceId`.
#[test]
fn two_partitions_on_two_devices_with_distinct_constraint() {
    let snapshot = two_device_snapshot();
    let admitted = AdmittedLogicalPlan::admit(
        LOGICAL_HASH,
        [partition_id(0), partition_id(1)],
        [DeclaredPlacementConstraint::DistinctPhysicalDevices],
    )
    .unwrap();

    let plan = bind_plan(
        &admitted,
        two_partition_bindings(
            (device_a(), Some(vp(1, device_a()))),
            (device_b(), Some(vp(2, device_b()))),
        ),
        DeviceSet::from_members([device_a(), device_b()]),
        &snapshot,
    )
    .unwrap();

    assert!(matches!(plan.kind(), BoundPlanKind::Distributed { .. }));
    let receipt = plan.receipt();
    assert_eq!(receipt.physical_device_count(), 2);
    assert_eq!(receipt.virtual_partition_count(), 2);
    assert_eq!(
        receipt.physical_device_ids(),
        &BTreeSet::from([device_a(), device_b()])
    );
    assert_eq!(
        receipt.virtual_partition_ids(),
        &BTreeSet::from([
            VirtualDevicePartitionId::new(1),
            VirtualDevicePartitionId::new(2)
        ])
    );
}

/// The receipt taxonomy serializes with `hardware_isolation_claimed=false` —
/// software admission partitions claim no hardware isolation (CTO
/// correction #5; md0-closeout §3.2 #4).
#[test]
fn receipt_serializes_with_hardware_isolation_not_claimed() {
    let snapshot = one_device_snapshot();
    let plan = bind_plan(
        &admitted_two_partition_plan(),
        two_partition_bindings(
            (device_a(), Some(vp(1, device_a()))),
            (device_a(), Some(vp(2, device_a()))),
        ),
        DeviceSet::from_members([device_a()]),
        &snapshot,
    )
    .unwrap();

    let receipt = plan.receipt();
    assert_eq!(
        receipt.hardware_isolation_claimed(),
        HardwareIsolationClaim::NotClaimed
    );

    // The canonical serialization writes the isolation claim last and false.
    let bytes = receipt.canonical_bytes();
    assert_eq!(bytes.last(), Some(&0u8));
}

/// An unknown or **replaced** `PhysicalDeviceId` is rejected at the health
/// epoch: replacement yields a *new* id that no snapshot entry carries until
/// the next probe (naming contract §1).
#[test]
fn bind_rejects_unknown_or_replaced_device_at_health_epoch() {
    let snapshot = one_device_snapshot();
    let admitted = admitted_two_partition_plan();

    // device_b is not recorded in the snapshot (unknown device; a replaced
    // device at the same ordinal would likewise be a new id absent here).
    let err = bind_plan(
        &admitted,
        two_partition_bindings((device_a(), None), (device_b(), None)),
        DeviceSet::from_members([device_a(), device_b()]),
        &snapshot,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        BindError::Membership(MembershipError::UnknownDevice(id)) if id == device_b()
    ));
}

/// A snapshot recorded under a stale health generation can never be the
/// frozen basis of a bind (MD1-Q3).
#[test]
fn bind_rejects_stale_snapshot() {
    let snapshot = snapshot_with([(0, device_a(), DeviceHealthGeneration::initial())]);
    let admitted = admitted_two_partition_plan();
    let bindings = two_partition_bindings((device_a(), None), (device_a(), None));

    let err = bind(
        &admitted,
        bindings,
        DeviceSet::from_members([device_a()]),
        &snapshot,
        DeviceHealthGeneration::initial().advance(), // epoch advanced
        FixtureIdentityClass::Synthetic,
        TransportClass::HostStaged,
    )
    .unwrap_err();
    assert!(matches!(err, BindError::StaleSnapshot { .. }));
}

/// A device set that does not match the plan's declared partition
/// set/bindings is a topology mismatch: extra unbound members, bindings
/// outside the set, missing bindings, and undeclared binding keys all reject.
#[test]
fn bind_rejects_topology_mismatch() {
    let snapshot = three_device_snapshot();
    let admitted = admitted_two_partition_plan();

    // (a) The device set carries a device no binding references.
    let err = bind_plan(
        &admitted,
        two_partition_bindings((device_a(), None), (device_b(), None)),
        DeviceSet::from_members([device_a(), device_b(), device_c()]),
        &snapshot,
    )
    .unwrap_err();
    assert!(
        matches!(err, BindError::TopologyMismatch { .. }),
        "extra unbound device must reject as topology mismatch"
    );

    // (b) A binding references a device outside the set.
    let err = bind_plan(
        &admitted,
        two_partition_bindings((device_a(), None), (device_b(), None)),
        DeviceSet::from_members([device_a()]),
        &snapshot,
    )
    .unwrap_err();
    assert!(
        matches!(err, BindError::TopologyMismatch { .. }),
        "binding outside the device set must reject"
    );

    // (c) A declared partition is missing from the bindings.
    let mut partial = BTreeMap::new();
    partial.insert(partition_id(0), PartitionBinding::new(device_a()));
    let err = bind_plan(
        &admitted,
        partial,
        DeviceSet::from_members([device_a()]),
        &snapshot,
    )
    .unwrap_err();
    assert!(
        matches!(err, BindError::TopologyMismatch { .. }),
        "missing binding must reject"
    );

    // (d) An undeclared partition is bound.
    let mut extra = two_partition_bindings((device_a(), None), (device_b(), None));
    extra.insert(partition_id(9), PartitionBinding::new(device_a()));
    let err = bind_plan(
        &admitted,
        extra,
        DeviceSet::from_members([device_a(), device_b()]),
        &snapshot,
    )
    .unwrap_err();
    assert!(
        matches!(err, BindError::TopologyMismatch { .. }),
        "undeclared binding key must reject"
    );
}

/// A binding that would violate a declared `PlacementConstraint` is rejected
/// — distinct-devices, colocation, and backend constraints each name their
/// violated class.
#[test]
fn bind_rejects_declared_constraint_violation() {
    let snapshot = two_device_snapshot();

    // DistinctPhysicalDevices declared, both partitions colocated → violation.
    let admitted = AdmittedLogicalPlan::admit(
        LOGICAL_HASH,
        [partition_id(0), partition_id(1)],
        [DeclaredPlacementConstraint::DistinctPhysicalDevices],
    )
    .unwrap();
    let err = bind_plan(
        &admitted,
        two_partition_bindings((device_a(), None), (device_a(), None)),
        DeviceSet::from_members([device_a()]),
        &snapshot,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        BindError::ConstraintViolation {
            constraint: DeclaredPlacementConstraint::DistinctPhysicalDevices,
            ..
        }
    ));

    // Colocated declared, partitions split across two devices → violation.
    let admitted = AdmittedLogicalPlan::admit(
        LOGICAL_HASH,
        [partition_id(0), partition_id(1)],
        [DeclaredPlacementConstraint::Colocated {
            partitions: BTreeSet::from([partition_id(0), partition_id(1)]),
        }],
    )
    .unwrap();
    let err = bind_plan(
        &admitted,
        two_partition_bindings((device_a(), None), (device_b(), None)),
        DeviceSet::from_members([device_a(), device_b()]),
        &snapshot,
    )
    .unwrap_err();
    assert!(matches!(err, BindError::ConstraintViolation { .. }));

    // RequiredBackend violated: the partition binds a CUDA device while the
    // declared backend is Metal.
    let admitted = AdmittedLogicalPlan::admit(
        LOGICAL_HASH,
        [partition_id(0)],
        [DeclaredPlacementConstraint::RequiredBackend {
            partitions: BTreeSet::from([partition_id(0)]),
            backend: DeviceBackend::Metal,
        }],
    )
    .unwrap();
    let mut bindings = BTreeMap::new();
    bindings.insert(partition_id(0), PartitionBinding::new(device_a()));
    let err = bind_plan(
        &admitted,
        bindings,
        DeviceSet::from_members([device_a()]),
        &snapshot,
    )
    .unwrap_err();
    assert!(matches!(err, BindError::ConstraintViolation { .. }));
}

/// MD-A15 degenerate: a single-partition logical plan binds to the
/// implicit/local partition with **no** distributed wrapper, transfer graph,
/// or `ExecutionTransaction` — and the shape checks still hold.
#[test]
fn single_partition_binds_to_implicit_local_without_distributed_wrapper() {
    let snapshot = one_device_snapshot();
    let admitted = AdmittedLogicalPlan::admit(LOGICAL_HASH, [partition_id(0)], []).unwrap();

    let implicit = VirtualDevicePartition::implicit_local(
        VirtualDevicePartitionId::new(1),
        device_a(),
        ledger(CS1_SPLIT_BYTES),
        SafePhysicalLimit::new(CS1_LIMIT_BYTES),
    )
    .unwrap();

    let mut bindings = BTreeMap::new();
    bindings.insert(
        partition_id(0),
        PartitionBinding::with_virtual_partition(device_a(), implicit.clone()),
    );

    // Single-device execution exercises no transport: TransportClass::None.
    let plan = bind(
        &admitted,
        bindings,
        DeviceSet::from_members([device_a()]),
        &snapshot,
        DeviceHealthGeneration::initial(),
        FixtureIdentityClass::Synthetic,
        TransportClass::None,
    )
    .expect("single-partition degenerate binds");

    // No distributed wrapper: the plan is the implicit/local shape and
    // exposes no per-partition binding map, transfer graph, or execution
    // transaction.
    assert!(plan.is_degenerate());
    assert!(plan.bindings().is_none());
    match plan.kind() {
        BoundPlanKind::ImplicitLocal {
            device,
            virtual_partition,
        } => {
            assert_eq!(device, &device_a());
            assert_eq!(virtual_partition.as_ref(), Some(&implicit));
        }
        BoundPlanKind::Distributed { .. } => {
            panic!("single-partition plan must never produce a distributed wrapper")
        }
    }

    // Receipt: one physical device, one (implicit local) virtual partition,
    // no transport exercised.
    let receipt = plan.receipt();
    assert_eq!(receipt.physical_device_count(), 1);
    assert_eq!(receipt.virtual_partition_count(), 1);
    assert_eq!(receipt.transport_class(), TransportClass::None);
    assert_eq!(
        receipt.hardware_isolation_claimed(),
        HardwareIsolationClaim::NotClaimed
    );

    // A multi-partition binding shape is still a topology mismatch for a
    // single-partition plan.
    let err = bind_plan(
        &admitted,
        two_partition_bindings((device_a(), None), (device_a(), None)),
        DeviceSet::from_members([device_a()]),
        &snapshot,
    )
    .unwrap_err();
    assert!(matches!(err, BindError::TopologyMismatch { .. }));
}

/// A binding carrying a virtual partition that binds a different physical
/// device, or that is not active, is rejected as an inconsistent binding.
#[test]
fn bind_rejects_inconsistent_or_inactive_virtual_partition() {
    let snapshot = two_device_snapshot();
    let admitted =
        AdmittedLogicalPlan::admit(LOGICAL_HASH, [partition_id(0), partition_id(1)], []).unwrap();

    // The attached partition binds device_b while the binding names device_a.
    let mut mismatched = BTreeMap::new();
    mismatched.insert(
        partition_id(0),
        PartitionBinding::with_virtual_partition(device_a(), vp(1, device_b())),
    );
    mismatched.insert(partition_id(1), PartitionBinding::new(device_b()));
    let err = bind_plan(
        &admitted,
        mismatched,
        DeviceSet::from_members([device_a(), device_b()]),
        &snapshot,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        BindError::InvalidPartitionBinding { partition, .. } if partition == partition_id(0)
    ));

    // A torn-down partition is not active and cannot be bound.
    let mut torn = vp(2, device_a());
    torn.teardown();
    let mut bindings = BTreeMap::new();
    bindings.insert(
        partition_id(0),
        PartitionBinding::with_virtual_partition(device_a(), torn),
    );
    bindings.insert(partition_id(1), PartitionBinding::new(device_b()));
    let err = bind_plan(
        &admitted,
        bindings,
        DeviceSet::from_members([device_a(), device_b()]),
        &snapshot,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        BindError::InvalidPartitionBinding { partition, .. } if partition == partition_id(0)
    ));
}

/// `bound_distributed_plan_hash` is sensitive to every binding fact:
/// physical ids enter THIS hash domain (never A10), the logical hash
/// participates, and the snapshot id participates.
#[test]
fn bound_hash_is_sensitive_to_binding_facts() {
    let snapshot = two_device_snapshot();
    let admitted =
        AdmittedLogicalPlan::admit(LOGICAL_HASH, [partition_id(0), partition_id(1)], []).unwrap();

    let baseline = bind_plan(
        &admitted,
        two_partition_bindings((device_a(), None), (device_b(), None)),
        DeviceSet::from_members([device_a(), device_b()]),
        &snapshot,
    )
    .unwrap();
    let baseline_hash = baseline.bound_distributed_plan_hash().to_owned();

    // Same logical plan, different physical binding → different bound hash
    // (physical ids enter the bound-plan hash domain, never A10).
    let swapped = bind_plan(
        &admitted,
        two_partition_bindings((device_b(), None), (device_a(), None)),
        DeviceSet::from_members([device_a(), device_b()]),
        &snapshot,
    )
    .unwrap();
    assert_ne!(swapped.bound_distributed_plan_hash(), baseline_hash);

    // Different logical hash → different bound hash.
    let other = AdmittedLogicalPlan::admit(
        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        [partition_id(0), partition_id(1)],
        [],
    )
    .unwrap();
    let other_plan = bind_plan(
        &other,
        two_partition_bindings((device_a(), None), (device_b(), None)),
        DeviceSet::from_members([device_a(), device_b()]),
        &snapshot,
    )
    .unwrap();
    assert_ne!(other_plan.bound_distributed_plan_hash(), baseline_hash);

    // Different snapshot (fresh probe time) → different bound hash.
    let fresh_snapshot = snapshot_at(
        PROBE_TIME + 1,
        [
            (0, device_a(), DeviceHealthGeneration::initial()),
            (1, device_b(), DeviceHealthGeneration::initial()),
        ],
    );
    let fresh_plan = bind_plan(
        &admitted,
        two_partition_bindings((device_a(), None), (device_b(), None)),
        DeviceSet::from_members([device_a(), device_b()]),
        &fresh_snapshot,
    )
    .unwrap();
    assert_ne!(fresh_plan.bound_distributed_plan_hash(), baseline_hash);
}
