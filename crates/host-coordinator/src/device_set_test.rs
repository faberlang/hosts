//! MD1-S1 device-set tests: healthy-epoch membership validation, explicit
//! selection vs the snapshot, the directed-link topology gate (admitted /
//! NOT-ATTEMPTED / rejected — never assumed), ordinal-free set equality, and
//! the legal size-1 set on the pharos snapshot.

use crate::backend::DeviceBackend;
use crate::device_identity::{DeviceHealthGeneration, DeviceOrdinal, PhysicalDeviceId};
use crate::device_set::{
    DeviceLink, DeviceLinkState, DeviceSet, DeviceSetConstraints, DeviceSetSelection,
    DeviceTopologySnapshot, LinkFacts, LinkGateError, LinkPathClass, MembershipError,
    SelectionError,
};
use crate::discovery::{
    ComputeCapability, DeviceCapabilities, DeviceDiscoveryEntry, DeviceDiscoverySnapshot,
    DeviceHealth, DeviceMemory, DtypeSurface, P2pProbeState, ProbeProvenance,
};
use std::collections::BTreeMap;
use std::collections::BTreeSet;

// T1 measured facts on pharos (md0-topology-evidence.md §2/§3).
const PCI_UUID: &str = "GPU-3e017562-9ec3-da9a-962d-b8bd5f9e24be";
const DRIVER_UUID: &str = "3e017562-9ec3-da9a-962d-b8bd5f9e24be";
const T1_DRIVER_BYTES: u64 = 12_343_705_600; // cuDeviceTotalMem
const PROBE_TIME: u64 = 1_752_717_600_000_000_000; // fixed sample time

const UUID_B: &str = "GPU-11111111-2222-3333-4444-555555555555";
const UUID_C: &str = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";
const UNKNOWN_UUID: &str = "GPU-99999999-8888-7777-6666-555555555555";

fn t1_entry() -> DeviceDiscoveryEntry {
    DeviceDiscoveryEntry {
        ordinal: DeviceOrdinal::new(0),
        identity: PhysicalDeviceId::cuda(PCI_UUID, Some(DRIVER_UUID.to_owned())),
        device_model: Some("NVIDIA GeForce RTX 5070".to_owned()),
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
            api_total_bytes: T1_DRIVER_BYTES,
        },
        health: DeviceHealth::Healthy,
        health_generation: DeviceHealthGeneration::initial(),
        probe_provenance: ProbeProvenance {
            probe: "device_enum + nvidia-smi".to_owned(),
            tool_versions: "driver 595.71.05 / CUDA 13.2".to_owned(),
        },
    }
}

fn t1_snapshot() -> DeviceDiscoverySnapshot {
    let mut devices = BTreeMap::new();
    devices.insert(DeviceOrdinal::new(0), t1_entry());
    DeviceDiscoverySnapshot::new(PROBE_TIME, devices, P2pProbeState::NotAttempted)
}

/// A minimal current-epoch healthy entry for synthetic multi-device
/// snapshots.
fn synthetic_entry(ordinal: u32, identity: PhysicalDeviceId) -> DeviceDiscoveryEntry {
    DeviceDiscoveryEntry {
        ordinal: DeviceOrdinal::new(ordinal),
        identity,
        device_model: None,
        capabilities: DeviceCapabilities {
            compute_capability: ComputeCapability {
                major: 12,
                minor: 0,
            },
            sm_count: 48,
            dtype_surface: DtypeSurface::empty(),
        },
        memory: DeviceMemory {
            tool_report_total_mib: None,
            api_total_bytes: 0,
        },
        health: DeviceHealth::Healthy,
        health_generation: DeviceHealthGeneration::initial(),
        probe_provenance: ProbeProvenance {
            probe: "synthetic fixture".to_owned(),
            tool_versions: "test".to_owned(),
        },
    }
}

/// A `DeviceSet` admits only current-epoch members and rejects an
/// unknown/replaced id (exit gate).
#[test]
fn set_admits_only_current_epoch_members_and_rejects_unknown_replaced() {
    let snap = t1_snapshot();
    let current = DeviceHealthGeneration::initial();

    // Healthy current-epoch member: admitted.
    let pharos = PhysicalDeviceId::cuda(PCI_UUID, Some(DRIVER_UUID.to_owned()));
    let ok = DeviceSet::from_members([pharos.clone()]);
    assert!(ok.validate(&snap, current).is_ok());

    // Unknown id: not recorded anywhere in the snapshot.
    let unknown = PhysicalDeviceId::cuda(UNKNOWN_UUID, None);
    let bad = DeviceSet::from_members([pharos.clone(), unknown.clone()]);
    assert_eq!(
        bad.validate(&snap, current),
        Err(MembershipError::UnknownDevice(unknown.clone()))
    );

    // Replaced id: same ordinal, different identity facts — a *new* id that
    // no snapshot entry carries until the next probe (naming contract §1
    // replacement detection).
    let replaced = PhysicalDeviceId::cuda(UUID_C, None);
    let replaced_set = DeviceSet::from_members([pharos.clone(), replaced.clone()]);
    assert_eq!(
        replaced_set.validate(&snap, current),
        Err(MembershipError::UnknownDevice(replaced.clone()))
    );
}

/// A member recorded under a stale health generation is rejected (MD1-Q3
/// default: the epoch advances on any admission-gating change).
#[test]
fn stale_epoch_member_is_rejected() {
    let snap = t1_snapshot(); // recorded under epoch 1
    let current = DeviceHealthGeneration::initial().advance(); // observed change → epoch 2
    let pharos = PhysicalDeviceId::cuda(PCI_UUID, Some(DRIVER_UUID.to_owned()));
    let set = DeviceSet::from_members([pharos.clone()]);

    assert_eq!(
        set.validate(&snap, current),
        Err(MembershipError::StaleEpoch {
            id: pharos.clone(),
            recorded: DeviceHealthGeneration::initial(),
            current,
        })
    );
}

/// An explicit `DeviceSetSelection` validates membership against the snapshot
/// (exit gate).
#[test]
fn explicit_selection_validates_membership_against_the_snapshot() {
    let snap = t1_snapshot();
    let current = DeviceHealthGeneration::initial();
    let pharos = PhysicalDeviceId::cuda(PCI_UUID, Some(DRIVER_UUID.to_owned()));

    let set = DeviceSetSelection::explicit([pharos.clone()])
        .resolve(&snap, current)
        .expect("the pharos id is a current-epoch member");
    assert!(set.contains(&pharos));
    assert!(set.is_singleton());

    // The same selection against a later (advanced) epoch is stale-rejected.
    let err = DeviceSetSelection::explicit([pharos.clone()])
        .resolve(&snap, current.advance())
        .unwrap_err();
    assert!(matches!(
        err,
        SelectionError::Membership(MembershipError::StaleEpoch { .. })
    ));

    // One unknown id rejects the whole selection.
    let unknown = PhysicalDeviceId::cuda(UNKNOWN_UUID, None);
    let err = DeviceSetSelection::explicit([pharos.clone(), unknown.clone()])
        .resolve(&snap, current)
        .unwrap_err();
    assert_eq!(
        err,
        SelectionError::Membership(MembershipError::UnknownDevice(unknown.clone()))
    );
}

/// Constraint-based selection resolves membership shape (backend, exclusions,
/// count bounds) against the snapshot.
#[test]
fn constraint_selection_resolves_membership_shape() {
    let cuda_a = PhysicalDeviceId::cuda(UUID_B, None);
    let cuda_b = PhysicalDeviceId::cuda(UUID_C, None);
    let metal = PhysicalDeviceId::metal("device-registry-1");
    let mut devices = BTreeMap::new();
    devices.insert(DeviceOrdinal::new(0), synthetic_entry(0, cuda_a.clone()));
    devices.insert(DeviceOrdinal::new(1), synthetic_entry(1, cuda_b.clone()));
    devices.insert(DeviceOrdinal::new(2), synthetic_entry(2, metal.clone()));
    let snap = DeviceDiscoverySnapshot::new(PROBE_TIME, devices, P2pProbeState::NotAttempted);
    let current = DeviceHealthGeneration::initial();

    // Backend constraint selects only CUDA members.
    let mut constraints = DeviceSetConstraints::default();
    constraints.backend = Some(DeviceBackend::Cuda);
    let set = DeviceSetSelection::constraints(constraints)
        .resolve(&snap, current)
        .expect("CUDA constraint is satisfiable");
    assert_eq!(set.len(), 2);
    assert!(set.contains(&cuda_a));
    assert!(set.contains(&cuda_b));
    assert!(!set.contains(&metal));

    // Excluding one member plus a declared minimum fails honestly.
    let mut constraints = DeviceSetConstraints::default();
    constraints.backend = Some(DeviceBackend::Cuda);
    constraints.exclude = BTreeSet::from([cuda_b.clone()]);
    constraints.min_count = 2;
    let err = DeviceSetSelection::constraints(constraints)
        .resolve(&snap, current)
        .unwrap_err();
    assert_eq!(err, SelectionError::BelowMinimum { min: 2, actual: 1 });

    // A declared maximum is enforced.
    let mut constraints = DeviceSetConstraints::default();
    constraints.max_count = Some(1);
    let err = DeviceSetSelection::constraints(constraints)
        .resolve(&snap, current)
        .unwrap_err();
    assert_eq!(err, SelectionError::AboveMaximum { max: 1, actual: 3 });
}

/// A *degraded* member stays selectable but is flagged for the gate layer
/// (C2 §2: "may stay in the `DeviceSet` but is ineligible for any placement
/// whose gate it cannot satisfy").
#[test]
fn degraded_member_stays_selectable_but_is_flagged_for_gates() {
    let degraded_id = PhysicalDeviceId::cuda(UUID_B, None);
    let mut degraded = synthetic_entry(0, degraded_id.clone());
    degraded.health = DeviceHealth::Degraded("reduced usable memory".to_owned());
    let mut devices = BTreeMap::new();
    devices.insert(DeviceOrdinal::new(0), degraded);
    let snap = DeviceDiscoverySnapshot::new(PROBE_TIME, devices, P2pProbeState::NotAttempted);
    let current = DeviceHealthGeneration::initial();

    let set = DeviceSetSelection::explicit([degraded_id.clone()])
        .resolve(&snap, current)
        .expect("a degraded member at the current epoch is selectable");
    assert!(set.contains(&degraded_id));

    // The degraded fact is exposed so gates can evaluate it (MD1-P1
    // consumes the per-device facts; it is not assumed away here).
    let topo = DeviceTopologySnapshot::new(snap, []);
    let member = topo
        .member(&degraded_id)
        .expect("member present in the topology");
    assert_eq!(
        member.health,
        DeviceHealth::Degraded("reduced usable memory".to_owned())
    );
}

/// A directed `DeviceLink` carries admitted / NOT-ATTEMPTED / rejected state
/// with path class + measured facts; the topology gate rejects any request
/// that would traverse a non-admitted link (exit gate).
#[test]
fn topology_gate_rejects_any_request_traversing_a_non_admitted_link() {
    let a = PhysicalDeviceId::cuda(UUID_B, None);
    let b = PhysicalDeviceId::cuda(UUID_C, None);
    let c = PhysicalDeviceId::cuda("GPU-cccccccc-2222-3333-4444-555555555555", None);
    let mut devices = BTreeMap::new();
    devices.insert(DeviceOrdinal::new(0), synthetic_entry(0, a.clone()));
    devices.insert(DeviceOrdinal::new(1), synthetic_entry(1, b.clone()));
    devices.insert(DeviceOrdinal::new(2), synthetic_entry(2, c.clone()));
    let snap = DeviceDiscoverySnapshot::new(PROBE_TIME, devices, P2pProbeState::Attempted);

    let topo = DeviceTopologySnapshot::new(
        snap,
        [
            DeviceLink::admitted(
                a.clone(),
                b.clone(),
                LinkPathClass::Peer,
                LinkFacts {
                    bandwidth_bytes_per_sec: 200_000_000_000,
                    latency_nanos: 1_500,
                },
            ),
            DeviceLink::not_attempted(b.clone(), a.clone()),
            DeviceLink::rejected(b.clone(), c.clone(), "peer access check failed"),
        ],
    );

    // The explicitly admitted link passes.
    assert_eq!(topo.traversal_allowed(&a, &b), Ok(()));

    // An explicit NOT-ATTEMPTED row is rejected.
    assert_eq!(
        topo.traversal_allowed(&b, &a),
        Err(LinkGateError::NotAttempted {
            from: b.clone(),
            to: a.clone()
        })
    );

    // An explicitly rejected row is rejected with the recorded reason.
    assert_eq!(
        topo.traversal_allowed(&b, &c),
        Err(LinkGateError::Rejected {
            from: b.clone(),
            to: c.clone(),
            reason: "peer access check failed".to_owned()
        })
    );

    // An absent row is never assumed (unmeasured pairs are not links).
    assert_eq!(
        topo.traversal_allowed(&a, &c),
        Err(LinkGateError::NoLinkRecorded {
            from: a.clone(),
            to: c.clone()
        })
    );

    // An endpoint outside the topology has no links at all.
    let stranger = PhysicalDeviceId::cuda(UNKNOWN_UUID, None);
    assert_eq!(
        topo.traversal_allowed(&a, &stranger),
        Err(LinkGateError::UnknownEndpoint {
            endpoint: stranger.clone()
        })
    );

    // A self-move is a local copy, not a link traversal.
    assert_eq!(topo.traversal_allowed(&a, &a), Ok(()));
}

/// A directed link carries the admitted-class + measured facts, and the three
/// states are distinct (never conflated).
#[test]
fn directed_link_carries_admitted_class_and_measured_facts() {
    let a = PhysicalDeviceId::cuda(UUID_B, None);
    let b = PhysicalDeviceId::cuda(UUID_C, None);
    let link = DeviceLink::admitted(
        a.clone(),
        b.clone(),
        LinkPathClass::HostStaged,
        LinkFacts {
            bandwidth_bytes_per_sec: 10_000_000_000,
            latency_nanos: 1_300,
        },
    );
    assert_eq!(link.from(), &a);
    assert_eq!(link.to(), &b);
    match link.state() {
        DeviceLinkState::Admitted { path_class, facts } => {
            assert_eq!(*path_class, LinkPathClass::HostStaged);
            assert_eq!(facts.bandwidth_bytes_per_sec, 10_000_000_000);
            assert_eq!(facts.latency_nanos, 1_300);
        }
        other => panic!("expected admitted, got {other:?}"),
    }

    assert_ne!(
        DeviceLink::admitted(
            a.clone(),
            b.clone(),
            LinkPathClass::Peer,
            LinkFacts {
                bandwidth_bytes_per_sec: 1,
                latency_nanos: 1
            }
        ),
        DeviceLink::not_attempted(a.clone(), b.clone())
    );
    assert_ne!(
        DeviceLink::not_attempted(a.clone(), b.clone()),
        DeviceLink::rejected(a.clone(), b.clone(), "no")
    );
}

/// Ordinal never participates in set equality (ordinal-rename locator-only
/// proof, exit gate).
#[test]
fn ordinal_never_participates_in_set_equality() {
    let pharos = PhysicalDeviceId::cuda(PCI_UUID, Some(DRIVER_UUID.to_owned()));
    let current = DeviceHealthGeneration::initial();

    // The same device observed at ordinal 0 and ordinal 7 (renamed locator).
    let mut at_0 = BTreeMap::new();
    at_0.insert(DeviceOrdinal::new(0), t1_entry());
    let snap_0 = DeviceDiscoverySnapshot::new(PROBE_TIME, at_0, P2pProbeState::NotAttempted);

    let mut renamed = t1_entry();
    renamed.ordinal = DeviceOrdinal::new(7);
    let mut at_7 = BTreeMap::new();
    at_7.insert(DeviceOrdinal::new(7), renamed);
    let snap_7 = DeviceDiscoverySnapshot::new(PROBE_TIME + 1, at_7, P2pProbeState::NotAttempted);

    let set_0 = DeviceSetSelection::explicit([pharos.clone()])
        .resolve(&snap_0, current)
        .expect("member at ordinal 0");
    let set_7 = DeviceSetSelection::explicit([pharos.clone()])
        .resolve(&snap_7, current)
        .expect("member at ordinal 7");
    assert_eq!(set_0, set_7);
    assert_eq!(set_0.members(), set_7.members());

    // A different device is a different set even at the same ordinal.
    let other = PhysicalDeviceId::cuda(UNKNOWN_UUID, None);
    let mut at_0_other = BTreeMap::new();
    let mut other_entry = t1_entry();
    other_entry.identity = other.clone();
    at_0_other.insert(DeviceOrdinal::new(0), other_entry);
    let snap_other =
        DeviceDiscoverySnapshot::new(PROBE_TIME, at_0_other, P2pProbeState::NotAttempted);
    let set_other = DeviceSetSelection::explicit([other.clone()])
        .resolve(&snap_other, current)
        .expect("member in the other snapshot");
    assert_ne!(set_0, set_other);
}

/// A size-1 `DeviceSet` on the pharos snapshot is legal and validates (T1
/// consequence 1; exit gate).
#[test]
fn size_one_set_on_the_pharos_snapshot_validates() {
    let snap = t1_snapshot();
    let current = DeviceHealthGeneration::initial();
    let pharos = PhysicalDeviceId::cuda(PCI_UUID, Some(DRIVER_UUID.to_owned()));

    let set = DeviceSetSelection::explicit([pharos.clone()])
        .resolve(&snap, current)
        .expect("the single pharos device is selectable");
    assert!(set.is_singleton());
    assert!(set.contains(&pharos));

    // The topology over the single-device host: per-device facts intact, no
    // directed link rows, and no cross-device traversal is possible —
    // two-physical-identity / P2P rows stay NOT ATTEMPTED (T1 §3).
    let topo = DeviceTopologySnapshot::new(snap, []);
    assert_eq!(topo.links().count(), 0);
    let member = topo.member(&pharos).expect("pharos device in the topology");
    assert_eq!(member.memory.api_total_bytes, T1_DRIVER_BYTES);
    assert_eq!(member.health, DeviceHealth::Healthy);
    assert_eq!(member.health_generation, current);
    assert_eq!(topo.traversal_allowed(&pharos, &pharos), Ok(()));
}

/// A link never joins a device to itself (T1 §3: self is not a P2P row) —
/// fail fast, like the discovery snapshot's assertion style.
#[test]
#[should_panic(expected = "directed pair")]
fn a_link_never_joins_a_device_to_itself() {
    let a = PhysicalDeviceId::cuda(UUID_B, None);
    let _ = DeviceLink::admitted(
        a.clone(),
        a,
        LinkPathClass::Peer,
        LinkFacts {
            bandwidth_bytes_per_sec: 1,
            latency_nanos: 1,
        },
    );
}

/// A topology refuses links whose endpoints are not devices in the discovery
/// sample.
#[test]
#[should_panic(expected = "link endpoint")]
fn topology_rejects_links_to_unknown_devices() {
    let snap = t1_snapshot();
    let pharos = PhysicalDeviceId::cuda(PCI_UUID, Some(DRIVER_UUID.to_owned()));
    let stranger = PhysicalDeviceId::cuda(UNKNOWN_UUID, None);
    let _ = DeviceTopologySnapshot::new(
        snap,
        [DeviceLink::admitted(
            pharos,
            stranger,
            LinkPathClass::Peer,
            LinkFacts {
                bandwidth_bytes_per_sec: 1,
                latency_nanos: 1,
            },
        )],
    );
}
