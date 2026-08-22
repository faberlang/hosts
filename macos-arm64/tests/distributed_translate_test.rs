//! MD3H-H4 part 1: FMIR distributed wire → transaction mirror, F1 fixtures.
//!
//! Consumes MD3H-F1 postcard artifacts (`be69f5ace`) as built bytes. The
//! 8:1-promoted-as-8-physical rejection is the red-first row.

use std::collections::BTreeSet;

use faber_host_macos_arm64::distributed_translate::{
    bind_translated, translate_device_section_bytes, BindPolicy,
};
use host_coordinator::bound_plan::BindError;
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

fn one_physical_snapshot() -> DeviceDiscoverySnapshot {
    DeviceDiscoverySnapshot::from_enumerated(
        PROBE_TIME,
        [DeviceDiscoveryEntry {
            ordinal: DeviceOrdinal::new(0),
            identity: snapshot_device(),
            device_model: Some("synthetic 8:1 fixture".to_owned()),
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
        }],
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
