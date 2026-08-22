//! MD3H-H4: FMIR distributed wire → transaction mirror, F1 fixtures, CLI bind.
//!
//! Consumes MD3H-F1 postcard artifacts (`be69f5ace`) as built bytes. The
//! 8:1-promoted-as-8-physical rejection is the red-first row (now green).
//! MD3J-B1 splits the declared bind count: `--bind-count 2` binds 4+4
//! across a synthetic 2-physical snapshot (red-first — was `Unsupported`).
//! Declared `--bind-count` maps onto that policy and prepares.

use std::collections::BTreeSet;

use faber_host_macos_arm64::device_execute::prepare_distributed_image;
use faber_host_macos_arm64::distributed_translate::{
    bind_policy_for_declared_count, bind_translated, bind_translated_with_constraints,
    translate_device_section_bytes, BindPolicy, TranslateError,
};
use host_coordinator::bound_plan::{BindError, DeclaredPlacementConstraint, LogicalPartitionId};
use host_coordinator::device_identity::{DeviceHealthGeneration, DeviceOrdinal, PhysicalDeviceId};
use host_coordinator::discovery::{
    ComputeCapability, DeviceCapabilities, DeviceDiscoveryEntry, DeviceDiscoverySnapshot,
    DeviceHealth, DeviceMemory, DtypeSurface, ProbeProvenance,
};
use host_coordinator::execution_transaction::{
    ExecutionTransaction, FakeExecutionBackend, TransactionId, TransactionState,
};
use host_coordinator::partition::FixtureIdentityClass;

// MD3H-F1 built artifacts (`be69f5ace`), consumed as postcard bytes.
const EIGHT_RANK: &[u8] = include_bytes!("fixtures/md3h/eight-rank.postcard");
const ONE_PARTITION: &[u8] = include_bytes!("fixtures/md3h/one-partition.postcard");

const PROBE_TIME: u64 = 1_752_717_600_000_000_000;
const SNAPSHOT_UUID: &str = "GPU-3e017562-9ec3-da9a-962d-b8bd5f9e24be";

fn snapshot_device() -> PhysicalDeviceId {
    PhysicalDeviceId::cuda(SNAPSHOT_UUID, None)
}

fn snapshot_entry(ordinal: u32, identity: PhysicalDeviceId, model: &str) -> DeviceDiscoveryEntry {
    DeviceDiscoveryEntry {
        ordinal: DeviceOrdinal::new(ordinal),
        identity,
        device_model: Some(model.to_owned()),
        capabilities: DeviceCapabilities {
            compute_capability: ComputeCapability {
                major: 12,
                minor: 0,
            },
            sm_count: 48,
            dtype_surface: DtypeSurface::empty(),
            max_threads_per_workgroup: 1024,
            workgroup_shared_memory_min_bytes: 32_768,
            workgroup_shared_memory_max_bytes: 32_768,
            collective_width: 32,
            unified_memory: false,
        },
        memory: DeviceMemory {
            tool_report_total_mib: Some(12_227),
            api_total_bytes: 12_343_705_600,
        },
        health: DeviceHealth::Healthy,
        health_generation: DeviceHealthGeneration::initial(),
        probe_provenance: ProbeProvenance {
            probe: "md3h-h4 fixture".to_owned(),
            tool_versions: "synthetic".to_owned(),
        },
    }
}

fn one_physical_snapshot() -> DeviceDiscoverySnapshot {
    DeviceDiscoverySnapshot::from_enumerated(
        PROBE_TIME,
        [snapshot_entry(0, snapshot_device(), "synthetic 8:1 fixture")],
    )
}

// MD3J-B1 synthetic multi-physical fixtures — two distinct same-SKU CUDA
// identities, plus a third for the over-membership rejection row.
const SECOND_SNAPSHOT_UUID: &str = "GPU-1a7f4c2b-8d5e-4f6a-b3c9-2e6d5a4b7c81";
const THIRD_SNAPSHOT_UUID: &str = "GPU-9c3e6b2d-4f1a-8e7c-5b3d-2a9f8c7e6d54";

fn two_physical_snapshot() -> DeviceDiscoverySnapshot {
    DeviceDiscoverySnapshot::from_enumerated(
        PROBE_TIME,
        [
            snapshot_entry(0, snapshot_device(), "synthetic 8:2 fixture device 0"),
            snapshot_entry(
                1,
                PhysicalDeviceId::cuda(SECOND_SNAPSHOT_UUID, None),
                "synthetic 8:2 fixture device 1",
            ),
        ],
    )
}

fn three_physical_snapshot() -> DeviceDiscoverySnapshot {
    DeviceDiscoverySnapshot::from_enumerated(
        PROBE_TIME,
        [
            snapshot_entry(0, snapshot_device(), "synthetic 8:2 fixture device 0"),
            snapshot_entry(
                1,
                PhysicalDeviceId::cuda(SECOND_SNAPSHOT_UUID, None),
                "synthetic 8:2 fixture device 1",
            ),
            snapshot_entry(
                2,
                PhysicalDeviceId::cuda(THIRD_SNAPSHOT_UUID, None),
                "synthetic 8:2 fixture device 2",
            ),
        ],
    )
}

#[test]
fn eight_rank_f1_image_translates_eight_partitions() {
    let plan = translate_device_section_bytes(EIGHT_RANK).expect("F1 8-rank postcard translates");
    assert_eq!(plan.partitions().len(), 8);
    assert!(plan.logical_distributed_plan_hash().starts_with("sha256:"));
    assert!(!plan.operations().is_empty());
    assert!(!plan.commit_boundary().is_empty());
}

#[test]
fn one_partition_f1_image_translates() {
    let plan =
        translate_device_section_bytes(ONE_PARTITION).expect("F1 1-partition postcard translates");
    assert_eq!(plan.partitions().len(), 1);
    assert!(plan.operations().iter().all(|operation| {
        operation
            .partitions()
            .iter()
            .all(|partition| partition.as_str() == "partition-0")
    }));
}

#[test]
fn translated_mirror_bytes_are_deterministic() {
    let first = translate_device_section_bytes(EIGHT_RANK).expect("first");
    let second = translate_device_section_bytes(EIGHT_RANK).expect("second");
    assert_eq!(first.canonical_bytes(), second.canonical_bytes());
    assert_eq!(first.operations().len(), second.operations().len());
    for (a, b) in first.operations().iter().zip(second.operations()) {
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
    }
    assert_eq!(
        first.commit_boundary().canonical_bytes(),
        second.commit_boundary().canonical_bytes()
    );
}

#[test]
fn eight_rank_colocated_bind_prepares_on_one_physical_snapshot() {
    let translated =
        translate_device_section_bytes(EIGHT_RANK).expect("F1 8-rank postcard translates");
    let snapshot = one_physical_snapshot();
    let bound = bind_translated(&translated, &snapshot, BindPolicy::ColocateOnSnapshot)
        .expect("8 virtual partitions bind onto 1 physical");
    assert_eq!(bound.device_set().len(), 1);
    assert_eq!(
        bound.fixture_identity_class(),
        FixtureIdentityClass::Virtual
    );
    let bindings = bound
        .bindings()
        .expect("8-rank bind is distributed, not degenerate");
    assert_eq!(bindings.len(), 8);
    let devices: BTreeSet<_> = bindings
        .values()
        .map(|binding| binding.device().clone())
        .collect();
    assert_eq!(devices.len(), 1);
    assert_eq!(
        devices.iter().next().expect("one device"),
        &snapshot_device()
    );

    let mut backend = FakeExecutionBackend::new();
    let mut transaction = ExecutionTransaction::new(
        TransactionId::new("md3h-h4-8on1"),
        bound,
        translated.operations().to_vec(),
        translated.commit_boundary().clone(),
    )
    .expect("8-rank transaction constructs");
    transaction
        .prepare(&mut backend)
        .expect("8:1 prepare reserves against admitted virtual partitions");
    assert_eq!(transaction.state(), &TransactionState::Prepared);
}

/// Claiming 8 physical devices for the F1 8-rank image against a 1-physical
/// snapshot rejects TopologyMismatch (the red row from the first H4 commit,
/// now green).
#[test]
fn eight_rank_promoted_as_eight_physical_rejects_topology_mismatch() {
    let translated =
        translate_device_section_bytes(EIGHT_RANK).expect("F1 8-rank postcard translates");
    let snapshot = one_physical_snapshot();
    let error = bind_translated(&translated, &snapshot, BindPolicy::OnePhysicalPerPartition)
        .expect_err("promoting 8:1 as 8 physical on a 1-physical snapshot must reject");
    assert!(
        matches!(error, BindError::TopologyMismatch { .. }),
        "8:1-promoted-as-8-physical must be TopologyMismatch-class, got {error:?}"
    );
}

#[test]
fn declared_bind_count_one_colocates() {
    assert_eq!(
        bind_policy_for_declared_count(8, 1).expect("bind-count 1"),
        BindPolicy::ColocateOnSnapshot
    );
    assert_eq!(
        bind_policy_for_declared_count(1, 1).expect("1:1"),
        BindPolicy::ColocateOnSnapshot
    );
}

#[test]
fn declared_bind_count_matching_partitions_is_one_physical_each() {
    assert_eq!(
        bind_policy_for_declared_count(8, 8).expect("bind-count 8"),
        BindPolicy::OnePhysicalPerPartition
    );
}

/// MD3J-B1 (was the red-first row): `--bind-count 2` for 8 partitions
/// derives the split policy.
#[test]
fn declared_bind_count_two_is_split_policy() {
    assert_eq!(
        bind_policy_for_declared_count(8, 2).expect("8:2"),
        BindPolicy::SplitAcrossMembership { bind_count: 2 }
    );
}

/// Any divisor of the partition count is a legal split shape (8:4).
#[test]
fn declared_bind_count_four_is_split_policy() {
    assert_eq!(
        bind_policy_for_declared_count(8, 4).expect("8:4"),
        BindPolicy::SplitAcrossMembership { bind_count: 4 }
    );
}

/// Non-divisor bind counts stay fail-closed unsupported (8:3 — no legal
/// contiguous split of 8 partitions into 3 physicals).
#[test]
fn declared_bind_count_three_stays_unsupported() {
    let error = bind_policy_for_declared_count(8, 3).expect_err("8:3 is not a legal split");
    assert!(
        matches!(error, TranslateError::Unsupported(_)),
        "non-divisor bind counts must stay unsupported, got {error:?}"
    );
}

/// MD3J-B1 red-first row: `--bind-count 2` on a synthetic 2-physical
/// snapshot prepares 8:2 (4+4), not the current `Unsupported` rejection.
#[test]
fn eight_rank_bind_count_two_prepares_four_and_four_on_two_physical_snapshot() {
    let snapshot = two_physical_snapshot();
    let receipt = prepare_distributed_image(EIGHT_RANK, &snapshot, 2)
        .expect("F1 8-rank image prepares 8:2 (4+4) on a 2-physical snapshot");
    assert_eq!(receipt.physical_device_count, 2);
    assert_eq!(receipt.physical_device_ids.len(), 2);
    assert_eq!(receipt.virtual_partition_count, 8);
    assert_eq!(receipt.bind_shape, "8:2");
    assert_eq!(receipt.fixture_identity_class, "virtual");
    assert!(!receipt.hardware_isolation_claimed);
    assert_eq!(receipt.transaction_state, "prepared");
    assert!(receipt.logical_distributed_plan_hash.starts_with("sha256:"));
    assert!(receipt.bound_distributed_plan_hash.starts_with("sha256:"));
}

/// MD3J-B1 red-first row: `--bind-count 2` on a 1-physical snapshot must
/// reject `TopologyMismatch` (currently `Unsupported`).
#[test]
fn eight_rank_bind_count_two_on_one_physical_rejects_topology_mismatch() {
    let snapshot = one_physical_snapshot();
    let error = prepare_distributed_image(EIGHT_RANK, &snapshot, 2)
        .expect_err("bind-count 2 on a 1-physical snapshot must reject");
    assert!(
        error.message.contains("TopologyMismatch"),
        "1-physical bind-count 2 must be TopologyMismatch-class, got {}",
        error.message
    );
}

/// MD3J-B1 red-first row: `--bind-count 2` on a 3-physical snapshot must
/// reject `TopologyMismatch` (currently `Unsupported`).
#[test]
fn eight_rank_bind_count_two_on_three_physical_rejects_topology_mismatch() {
    let snapshot = three_physical_snapshot();
    let error = prepare_distributed_image(EIGHT_RANK, &snapshot, 2)
        .expect_err("bind-count 2 on a 3-physical snapshot must reject");
    assert!(
        error.message.contains("TopologyMismatch"),
        "3-physical bind-count 2 must be TopologyMismatch-class, got {}",
        error.message
    );
}

/// 8:2 split bind across the synthetic 2-physical membership: contiguous
/// halves in plan order (partitions 0–3 → first physical, 4–7 → second),
/// deterministically — repeat binds produce identical plans and hashes.
#[test]
fn eight_rank_split_two_binds_contiguous_halves_deterministically() {
    let translated = translate_device_section_bytes(EIGHT_RANK).expect("8-rank translates");
    let plan_order: Vec<String> = translated
        .partitions()
        .iter()
        .map(|partition| partition.as_str().to_owned())
        .collect();
    let expected_order: Vec<String> = (0..8).map(|id| format!("partition-{id}")).collect();
    assert_eq!(plan_order, expected_order, "plan order is the split's rank order");

    let snapshot = two_physical_snapshot();
    let policy = BindPolicy::SplitAcrossMembership { bind_count: 2 };
    let first = bind_translated(&translated, &snapshot, policy).expect("8:2 split binds");
    let second = bind_translated(&translated, &snapshot, policy).expect("repeat split binds");
    assert_eq!(first, second, "repeat binds are identical");
    assert_eq!(
        first.bound_distributed_plan_hash(),
        second.bound_distributed_plan_hash()
    );
    assert_eq!(first.canonical_bytes(), second.canonical_bytes());

    assert_eq!(first.device_set().len(), 2);
    let bindings = first.bindings().expect("8:2 is a distributed bind");
    assert_eq!(bindings.len(), 8);
    let first_half_device = snapshot_device();
    let second_half_device = PhysicalDeviceId::cuda(SECOND_SNAPSHOT_UUID, None);
    for (index, partition) in translated.partitions().iter().enumerate() {
        let binding = bindings
            .get(partition)
            .expect("every declared partition is bound");
        let expected = if index < 4 {
            &first_half_device
        } else {
            &second_half_device
        };
        assert_eq!(
            binding.device(),
            expected,
            "partition {partition} binds its contiguous plan-order half"
        );
    }
}

/// Split bind on a 1-physical snapshot rejects `TopologyMismatch`.
#[test]
fn eight_rank_split_two_on_one_physical_rejects_topology_mismatch() {
    let translated = translate_device_section_bytes(EIGHT_RANK).expect("8-rank translates");
    let error = bind_translated(
        &translated,
        &one_physical_snapshot(),
        BindPolicy::SplitAcrossMembership { bind_count: 2 },
    )
    .expect_err("8:2 on 1 physical must reject");
    assert!(
        matches!(error, BindError::TopologyMismatch { .. }),
        "8:2 on 1 physical must be TopologyMismatch-class, got {error:?}"
    );
}

/// Split bind on a 3-physical snapshot rejects `TopologyMismatch`.
#[test]
fn eight_rank_split_two_on_three_physical_rejects_topology_mismatch() {
    let translated = translate_device_section_bytes(EIGHT_RANK).expect("8-rank translates");
    let error = bind_translated(
        &translated,
        &three_physical_snapshot(),
        BindPolicy::SplitAcrossMembership { bind_count: 2 },
    )
    .expect_err("8:2 on 3 physicals must reject");
    assert!(
        matches!(error, BindError::TopologyMismatch { .. }),
        "8:2 on 3 physicals must be TopologyMismatch-class, got {error:?}"
    );
}

/// A declared `Colocated` placement constraint spanning the split boundary
/// (partitions on both physicals) rejects `ConstraintViolation` — declared
/// placement constraints enforce per binding under the split policy.
#[test]
fn split_two_colocated_constraint_crossing_split_rejects() {
    let translated = translate_device_section_bytes(EIGHT_RANK).expect("8-rank translates");
    let snapshot = two_physical_snapshot();
    let constraint = DeclaredPlacementConstraint::Colocated {
        partitions: BTreeSet::from([
            LogicalPartitionId::new("partition-0"),
            LogicalPartitionId::new("partition-4"),
        ]),
    };
    let error = bind_translated_with_constraints(
        &translated,
        &snapshot,
        BindPolicy::SplitAcrossMembership { bind_count: 2 },
        &[constraint],
    )
    .expect_err("colocating partitions across the split must reject");
    assert!(
        matches!(error, BindError::ConstraintViolation { .. }),
        "split-crossing colocation must be ConstraintViolation-class, got {error:?}"
    );
}

/// A `Colocated` constraint contained within one split half admits.
#[test]
fn split_two_colocated_constraint_within_one_half_admits() {
    let translated = translate_device_section_bytes(EIGHT_RANK).expect("8-rank translates");
    let snapshot = two_physical_snapshot();
    let constraint = DeclaredPlacementConstraint::Colocated {
        partitions: BTreeSet::from([
            LogicalPartitionId::new("partition-0"),
            LogicalPartitionId::new("partition-3"),
        ]),
    };
    bind_translated_with_constraints(
        &translated,
        &snapshot,
        BindPolicy::SplitAcrossMembership { bind_count: 2 },
        &[constraint],
    )
    .expect("colocation within one half is consistent with the split");
}

/// Virtual and physical identity classes never merge in the 8:2 binding
/// table: bindings name exactly the snapshot's two physical ids, the
/// attached virtual partitions carry `vp:` ids from the machine-local
/// partition namespace, and no id string is shared across the classes.
#[test]
fn split_two_binding_table_keeps_identity_classes_distinct() {
    let translated = translate_device_section_bytes(EIGHT_RANK).expect("8-rank translates");
    let snapshot = two_physical_snapshot();
    let bound = bind_translated(
        &translated,
        &snapshot,
        BindPolicy::SplitAcrossMembership { bind_count: 2 },
    )
    .expect("8:2 split binds");
    let bindings = bound.bindings().expect("distributed bind");
    assert_eq!(bound.device_set().len(), 2);

    let physical_ids: BTreeSet<String> = bindings
        .values()
        .map(|binding| binding.device().to_string())
        .collect();
    assert_eq!(physical_ids.len(), 2);
    assert!(
        physical_ids.iter().all(|id| id.starts_with("cuda:GPU-")),
        "bindings name physical ids, got {physical_ids:?}"
    );

    let virtual_ids: BTreeSet<String> = bindings
        .values()
        .filter_map(|binding| binding.virtual_partition().map(|vp| vp.id().to_string()))
        .collect();
    assert_eq!(virtual_ids.len(), 8);
    assert!(
        virtual_ids.iter().all(|id| id.starts_with("vp:")),
        "virtual partitions carry vp: ids, got {virtual_ids:?}"
    );

    assert!(
        physical_ids.is_disjoint(&virtual_ids),
        "virtual and physical identity namespaces must never merge"
    );
    assert_eq!(bound.fixture_identity_class(), FixtureIdentityClass::Virtual);
}

#[test]
fn eight_rank_bind_count_one_prepares_on_one_physical_cuda_snapshot() {
    let snapshot = one_physical_snapshot();
    let receipt = prepare_distributed_image(EIGHT_RANK, &snapshot, 1)
        .expect("F1 8-rank image prepares 8:1 on a 1-physical CUDA snapshot");
    assert_eq!(receipt.physical_device_count, 1);
    assert_eq!(receipt.virtual_partition_count, 8);
    assert_eq!(receipt.fixture_identity_class, "virtual");
    assert_eq!(receipt.hardware_isolation_claimed, false);
    assert_eq!(receipt.bind_shape, "8:1");
    assert!(receipt.communication_graph_edge_count > 0);
    assert_eq!(receipt.transaction_state, "prepared");
    assert!(receipt.logical_distributed_plan_hash.starts_with("sha256:"));
    assert!(receipt.bound_distributed_plan_hash.starts_with("sha256:"));
    assert_eq!(receipt.physical_device_ids.len(), 1);
    assert_eq!(receipt.virtual_partition_ids.len(), 8);
}

#[test]
fn eight_rank_bind_count_eight_rejects_topology_mismatch() {
    let snapshot = one_physical_snapshot();
    let error = prepare_distributed_image(EIGHT_RANK, &snapshot, 8)
        .expect_err("bind-count 8 on a 1-physical snapshot must reject");
    assert!(
        error.message.contains("TopologyMismatch"),
        "CLI bind-count 8 must stay TopologyMismatch-class, got {}",
        error.message
    );
}
