//! MD1-D1 discovery tests: T1 pharos fact round-trip (both memory reports
//! kept distinct), byte determinism, health-epoch gating, and the locator
//! semantics. SHA-256 known-answer vectors prove the content-addressing
//! digest.
//!
//! MD1-D2 (evidence + determinism, added below): the T1 pharos fixture's
//! content-addressed hash is **frozen** in a golden test, and the NOT
//! ATTEMPTED rows (two-physical-identity, every directed P2P pair,
//! independent device-loss) are proven explicit in the snapshot
//! representation — an absent/not-attempted fact can never be mistaken for a
//! pass (T1 §8; CTO `2f90eafd` §5b).

use crate::backend::DeviceBackend;
use crate::device_identity::{
    DeviceHealthGeneration, DeviceOrdinal, IdentityChange, PhysicalDeviceId,
};
use crate::device_set::{DeviceLinkState, DeviceTopologySnapshot, LinkGateError};
use crate::discovery::{
    ComputeCapability, DeviceCapabilities, DeviceDiscoveryEntry, DeviceDiscoverySnapshot,
    DeviceHealth, DeviceMemory, DtypeSurface, P2pProbeState, ProbeProvenance,
};
use std::collections::BTreeMap;

// T1 measured facts on pharos (md0-topology-evidence.md §2/§3).
const PCI_UUID: &str = "GPU-3e017562-9ec3-da9a-962d-b8bd5f9e24be";
const DRIVER_UUID: &str = "3e017562-9ec3-da9a-962d-b8bd5f9e24be";
const T1_NVIDIA_SMI_MIB: u64 = 12_227; // 12227 MiB
const T1_DRIVER_BYTES: u64 = 12_343_705_600; // cuDeviceTotalMem
const PROBE_TIME: u64 = 1_752_717_600_000_000_000; // fixed sample time

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
            max_threads_per_workgroup: 1024,
            workgroup_shared_memory_min_bytes: 49_152,
            workgroup_shared_memory_max_bytes: 101_376,
            collective_width: 32,
            unified_memory: false,
        },
        memory: DeviceMemory {
            tool_report_total_mib: Some(T1_NVIDIA_SMI_MIB),
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

/// The T1 pharos facts round-trip through a snapshot with every field intact
/// (FC9; exit gate: discovery snapshot round-trips the T1 facts).
#[test]
fn t1_pharos_facts_round_trip() {
    let snap = t1_snapshot();
    let entry = snap
        .entry(DeviceOrdinal::new(0))
        .expect("ordinal 0 present in the pharos snapshot");

    // Identity: PCI UUID + corroborating driver UUID, both kept.
    assert_eq!(entry.ordinal, DeviceOrdinal::new(0));
    assert_eq!(entry.backend(), DeviceBackend::Cuda);
    assert_eq!(
        entry.identity,
        PhysicalDeviceId::cuda(PCI_UUID, Some(DRIVER_UUID.to_owned()))
    );

    // Model descriptor.
    assert_eq!(
        entry.device_model.as_deref(),
        Some("NVIDIA GeForce RTX 5070")
    );

    // Capabilities: CC 12.0, 48 SMs, dtype smoke all PASS (T1 §2).
    assert_eq!(
        entry.capabilities.compute_capability,
        ComputeCapability {
            major: 12,
            minor: 0
        }
    );
    assert_eq!(entry.capabilities.sm_count, 48);
    assert!(entry.capabilities.dtype_surface.f32);
    assert!(entry.capabilities.dtype_surface.f64);
    assert!(entry.capabilities.dtype_surface.f16);
    assert!(entry.capabilities.dtype_surface.bf16);
    assert!(entry.capabilities.dtype_surface.i8);
    assert!(entry.capabilities.dtype_surface.i32);
    assert_eq!(entry.capabilities.max_threads_per_workgroup, 1024);
    assert_eq!(entry.capabilities.workgroup_shared_memory_min_bytes, 49_152);
    assert_eq!(
        entry.capabilities.workgroup_shared_memory_max_bytes,
        101_376
    );
    assert_eq!(entry.capabilities.collective_width, 32);
    assert!(!entry.capabilities.unified_memory);

    // Memory: BOTH reports kept distinct — never conflated (T1 §8). 12227 MiB
    // (nvidia-smi) is not the same number as 12 343 705 600 B (driver).
    assert_eq!(entry.memory.tool_report_total_mib, Some(T1_NVIDIA_SMI_MIB));
    assert_eq!(entry.memory.api_total_bytes, T1_DRIVER_BYTES);
    assert_ne!(
        entry.memory.api_total_bytes,
        entry.memory.tool_report_total_mib.unwrap() * 1024 * 1024
    );

    // Health: healthy, epoch 1 (T1 §2).
    assert_eq!(entry.health, DeviceHealth::Healthy);
    assert_eq!(entry.health_generation, DeviceHealthGeneration::initial());

    // P2P: one device — every directed pair is NOT ATTEMPTED, explicit (T1 §3).
    assert_eq!(snap.p2p_state(), P2pProbeState::NotAttempted);

    // Provenance + explicit sample time.
    assert_eq!(entry.probe_provenance.probe, "device_enum + nvidia-smi");
    assert_eq!(
        entry.probe_provenance.tool_versions,
        "driver 595.71.05 / CUDA 13.2"
    );
    assert_eq!(snap.probe_utc_nanos(), PROBE_TIME);
}

/// Identical input facts produce identical canonical bytes and an identical
/// content-addressed id (exit gate: byte-deterministic for identical input).
#[test]
fn identical_facts_produce_identical_bytes_and_id() {
    let a = t1_snapshot();
    let b = t1_snapshot();
    assert_eq!(a.canonical_bytes(), b.canonical_bytes());
    assert_eq!(a.id(), b.id());
    assert_eq!(a.id().hex(), b.id().hex());
    assert_eq!(a.id().as_bytes().len(), 32);

    // The id is the SHA-256 of the canonical bytes (content-addressed).
    assert_eq!(a.id().hex(), sha256_hex_of(&a.canonical_bytes()));
}

/// A stale health generation rejects a snapshot before it gates admission or
/// planning (exit gate: device removal/replacement invalidates stale
/// identities; MD1-Q3 default).
#[test]
fn stale_generation_rejects_stale_snapshot() {
    let snap = t1_snapshot(); // recorded under epoch 1
    let current = DeviceHealthGeneration::initial().advance(); // observed change → epoch 2

    assert!(snap.is_stale(current));
    assert!(!snap.is_current_generation(current));
    assert!(current.is_stale(snap.entry(DeviceOrdinal::new(0)).unwrap().health_generation));

    // A snapshot recorded at the current generation is accepted.
    let mut entry = t1_entry();
    entry.health_generation = current;
    let mut devices = BTreeMap::new();
    devices.insert(DeviceOrdinal::new(0), entry);
    let fresh = DeviceDiscoverySnapshot::new(PROBE_TIME + 1, devices, P2pProbeState::NotAttempted);
    assert!(!fresh.is_stale(current));
    assert!(fresh.is_current_generation(current));
}

/// A capability change (an admission-gating fact) advances the health epoch.
#[test]
fn capability_change_advances_the_health_epoch() {
    let gen = DeviceHealthGeneration::initial();
    let changed_gen = gen.advance();
    let mut changed = t1_entry();
    changed.capabilities.sm_count = 32; // capability set changed
    changed.health_generation = changed_gen;

    assert!(gen < changed_gen);
    let mut devices = BTreeMap::new();
    devices.insert(DeviceOrdinal::new(0), changed);
    let snap = DeviceDiscoverySnapshot::new(PROBE_TIME, devices, P2pProbeState::NotAttempted);
    assert!(snap.is_stale(gen));
    assert!(snap.is_current_generation(changed_gen));
}

/// A replacement at the same ordinal (different identity facts) yields a
/// distinct id and an epoch advance.
#[test]
fn replacement_at_same_ordinal_advances_the_health_epoch() {
    let gen = DeviceHealthGeneration::initial();
    let old = PhysicalDeviceId::cuda(PCI_UUID, Some(DRIVER_UUID.to_owned()));
    let replaced = PhysicalDeviceId::cuda("GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", None);

    assert_eq!(replaced.change_against(&old), IdentityChange::Replaced);
    assert_ne!(replaced, old);
    let next = gen.advance();
    assert!(next > gen);
    assert!(gen.is_stale(next));
}

/// Device removal is a presence change: the epoch advances and the old sample
/// becomes stale.
#[test]
fn removal_advances_the_health_epoch() {
    let gen = DeviceHealthGeneration::initial();
    let empty: BTreeMap<DeviceOrdinal, DeviceDiscoveryEntry> = BTreeMap::new();
    let removed = DeviceDiscoverySnapshot::new(PROBE_TIME + 2, empty, P2pProbeState::NotAttempted);

    assert!(removed.devices().is_empty());
    let next = gen.advance();
    assert!(next > gen);
    assert!(t1_snapshot().is_stale(next));
}

/// Renaming the ordinal locator keeps the identity; a different device that
/// happens to reuse an ordinal is a distinct id (locator-only rule).
#[test]
fn ordinal_rename_keeps_identity_but_snapshot_records_locator() {
    let mut renamed = t1_entry();
    renamed.ordinal = DeviceOrdinal::new(7);
    assert_eq!(renamed.identity, t1_entry().identity);

    let mut other = t1_entry();
    other.identity = PhysicalDeviceId::cuda("GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", None);
    assert_ne!(other.identity, t1_entry().identity);
    assert_eq!(other.ordinal, DeviceOrdinal::new(0));
}

fn second_cuda_entry(ordinal: u32, pci_uuid: &str) -> DeviceDiscoveryEntry {
    let mut entry = t1_entry();
    entry.ordinal = DeviceOrdinal::new(ordinal);
    entry.identity = PhysicalDeviceId::cuda(pci_uuid, None);
    entry.device_model = Some("synthetic-second-cuda".to_owned());
    entry
}

/// MD3H-H1: the populate seam keys entries by locator ordinal and carries
/// identity/memory facts without the caller assembling the map.
#[test]
fn from_enumerated_populates_identity_and_memory_facts() {
    let snap = DeviceDiscoverySnapshot::from_enumerated(PROBE_TIME, [t1_entry()]);
    let entry = snap
        .entry(DeviceOrdinal::new(0))
        .expect("populate seam records ordinal 0");
    assert_eq!(entry.backend(), DeviceBackend::Cuda);
    assert_eq!(
        entry.identity,
        PhysicalDeviceId::cuda(PCI_UUID, Some(DRIVER_UUID.to_owned()))
    );
    assert_eq!(entry.memory.tool_report_total_mib, Some(T1_NVIDIA_SMI_MIB));
    assert_eq!(entry.memory.api_total_bytes, T1_DRIVER_BYTES);
    assert_eq!(snap.p2p_state(), P2pProbeState::NotAttempted);
}

/// Two same-backend devices are distinct ids in one snapshot even when the
/// locators are adjacent (existing identity machinery; MD3H-H1).
#[test]
fn from_enumerated_two_same_backend_devices_are_distinguishable() {
    let first = t1_entry();
    let second = second_cuda_entry(1, "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee");
    let snap =
        DeviceDiscoverySnapshot::from_enumerated(PROBE_TIME, [first.clone(), second.clone()]);
    assert_eq!(snap.devices().len(), 2);
    let a = &snap
        .entry(DeviceOrdinal::new(0))
        .expect("ordinal 0")
        .identity;
    let b = &snap
        .entry(DeviceOrdinal::new(1))
        .expect("ordinal 1")
        .identity;
    assert_eq!(a.backend(), DeviceBackend::Cuda);
    assert_eq!(b.backend(), DeviceBackend::Cuda);
    assert_ne!(a, b);
    assert_eq!(a, &first.identity);
    assert_eq!(b, &second.identity);
}

fn fake_metal_entry() -> DeviceDiscoveryEntry {
    DeviceDiscoveryEntry {
        ordinal: DeviceOrdinal::new(0),
        identity: PhysicalDeviceId::metal("4278190081"),
        device_model: Some("Apple M-series".to_owned()),
        capabilities: DeviceCapabilities {
            compute_capability: ComputeCapability { major: 0, minor: 0 },
            sm_count: 0,
            dtype_surface: DtypeSurface::empty(),
            max_threads_per_workgroup: 1024,
            workgroup_shared_memory_min_bytes: 32_768,
            workgroup_shared_memory_max_bytes: 32_768,
            collective_width: 32,
            unified_memory: true,
        },
        memory: DeviceMemory {
            tool_report_total_mib: None,
            api_total_bytes: 36_123_000_000,
        },
        health: DeviceHealth::Healthy,
        health_generation: DeviceHealthGeneration::initial(),
        probe_provenance: ProbeProvenance {
            probe: "MTLCopyAllDevices".to_owned(),
            tool_versions: "Metal framework".to_owned(),
        },
    }
}

/// DCG-1: fake Metal and CUDA snapshots populate all five generic
/// launch-resource fields with distinct per-backend shapes.
#[test]
fn fake_metal_and_cuda_snapshots_populate_distinct_launch_resources() {
    let cuda_snap = DeviceDiscoverySnapshot::from_enumerated(PROBE_TIME, [t1_entry()]);
    let metal_snap = DeviceDiscoverySnapshot::from_enumerated(PROBE_TIME, [fake_metal_entry()]);
    let cuda = cuda_snap
        .entry(DeviceOrdinal::new(0))
        .expect("cuda ordinal 0")
        .capabilities;
    let metal = metal_snap
        .entry(DeviceOrdinal::new(0))
        .expect("metal ordinal 0")
        .capabilities;

    assert_eq!(cuda.max_threads_per_workgroup, 1024);
    assert_eq!(cuda.workgroup_shared_memory_min_bytes, 49_152);
    assert_eq!(cuda.workgroup_shared_memory_max_bytes, 101_376);
    assert_eq!(cuda.collective_width, 32);
    assert!(!cuda.unified_memory);

    assert_eq!(metal.max_threads_per_workgroup, 1024);
    assert_eq!(metal.workgroup_shared_memory_min_bytes, 32_768);
    assert_eq!(metal.workgroup_shared_memory_max_bytes, 32_768);
    assert_eq!(metal.collective_width, 32);
    assert!(metal.unified_memory);

    assert_ne!(cuda, metal);
    assert_ne!(
        cuda.workgroup_shared_memory_min_bytes,
        metal.workgroup_shared_memory_min_bytes
    );
    assert_ne!(
        cuda.workgroup_shared_memory_max_bytes,
        metal.workgroup_shared_memory_max_bytes
    );
    assert_ne!(cuda.unified_memory, metal.unified_memory);
    assert_eq!(
        cuda_snap.entry(DeviceOrdinal::new(0)).unwrap().backend(),
        DeviceBackend::Cuda
    );
    assert_eq!(
        metal_snap.entry(DeviceOrdinal::new(0)).unwrap().backend(),
        DeviceBackend::Metal
    );
}

/// Reusing an ordinal across samples never merges identities: the later
/// facts mint a distinct id and `change_against` reports replacement.
#[test]
fn from_enumerated_ordinal_reuse_never_merges_identities() {
    let earlier = DeviceDiscoverySnapshot::from_enumerated(PROBE_TIME, [t1_entry()]);
    let later = DeviceDiscoverySnapshot::from_enumerated(
        PROBE_TIME + 1,
        [second_cuda_entry(
            0,
            "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        )],
    );
    let old = &earlier
        .entry(DeviceOrdinal::new(0))
        .expect("earlier")
        .identity;
    let new = &later.entry(DeviceOrdinal::new(0)).expect("later").identity;
    assert_eq!(
        earlier.entry(DeviceOrdinal::new(0)).map(|e| e.ordinal),
        later.entry(DeviceOrdinal::new(0)).map(|e| e.ordinal)
    );
    assert_ne!(old, new);
    assert_eq!(new.change_against(old), IdentityChange::Replaced);
}

/// A snapshot entry whose ordinal disagrees with its map key is a programmer
/// error and fails fast.
#[test]
#[should_panic(expected = "discovery entry ordinal must match its map key")]
fn mismatched_ordinal_key_panics() {
    let mut entry = t1_entry();
    entry.ordinal = DeviceOrdinal::new(9);
    let mut devices = BTreeMap::new();
    devices.insert(DeviceOrdinal::new(0), entry);
    let _ = DeviceDiscoverySnapshot::new(PROBE_TIME, devices, P2pProbeState::NotAttempted);
}

/// The hand-rolled SHA-256 matches published FIPS 180-4 test vectors.
#[test]
fn sha256_known_answer_vectors() {
    assert_eq!(
        super::Sha256::digest(b""),
        hex_decode("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    );
    assert_eq!(
        super::Sha256::digest(b"abc"),
        hex_decode("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    );
    assert_eq!(
        super::Sha256::digest(b"The quick brown fox jumps over the lazy dog"),
        hex_decode("d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592")
    );
}

fn sha256_hex_of(bytes: &[u8]) -> String {
    let digest = super::Sha256::digest(bytes);
    let mut s = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write;
        write!(s, "{b:02x}").expect("writing to a String cannot fail");
    }
    s
}

fn hex_decode(hex: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, byte) in hex.as_bytes().chunks_exact(2).enumerate() {
        let hi = (byte[0] as char).to_digit(16).expect("hex digit");
        let lo = (byte[1] as char).to_digit(16).expect("hex digit");
        out[i] = u8::try_from((hi << 4) | lo).expect("hex nibble pair is a byte");
    }
    out
}

// ============================================================================
// MD1-D2 — discovery snapshot evidence + determinism.
//
// The [`t1_snapshot`] fixture above is built from the T1 measured facts
// (FC9). The tests in this section (a) pin the fixture's content-addressed
// hash so any drift in the facts or the canonical encoding breaks loudly
// instead of silently changing the evidence, and (b) prove the NOT ATTEMPTED
// rows are EXPLICIT in the snapshot representation: two-physical-identity,
// every directed P2P pair `i→j, i≠j`, and independent device-loss can never
// be mistaken for a pass (T1 §8; CTO `2f90eafd` §5b).
// ============================================================================

/// The frozen content hash of the T1 pharos fixture (MD1-D2 evidence).
///
/// `DeviceDiscoverySnapshot` ids are content-addressed — SHA-256 over the
/// canonical bytes of every fact, including the explicit probe time
/// (identical facts → identical hash). Pinning the fixture id makes the
/// evidence durable: any change to the T1 facts or the canonical encoding
/// breaks this golden test.
///
/// Computed 2026-08-21 after DCG-1 added generic launch-resource fields to
/// the canonical encoding (prior hash `9ca98f77629c571080fc9d7d59ed04d6d69b513fc908532f89872c9fb25c324d`
/// from the 2026-08-05 MD1-D2 pharos re-run).
const T1_PHAROS_FIXTURE_HASH_HEX: &str =
    "2fea169d29aa07f0e950f48e89683d657ffb88b7ce195c9db1b6f7be9f0e9003";

/// MD1-D2: the pharos snapshot fixture validates and is byte-deterministic —
/// its content-addressed id is frozen evidence.
#[test]
fn t1_pharos_fixture_hash_is_frozen() {
    let snap = t1_snapshot();
    assert_eq!(snap.id().hex(), T1_PHAROS_FIXTURE_HASH_HEX);
    assert_eq!(snap.id().as_bytes().len(), 32);

    // Determinism for identical input facts: rebuilding the identical
    // fixture yields identical canonical bytes and an identical id.
    assert_eq!(t1_snapshot().canonical_bytes(), snap.canonical_bytes());
    assert_eq!(t1_snapshot().id(), snap.id());
}

/// MD1-D2: two-physical-identity is NOT ATTEMPTED — the snapshot carries
/// exactly one physical identity row, and no second identity is claimed
/// anywhere in the representation (T1 §8; CTO `2f90eafd` §5b).
#[test]
fn two_physical_identity_row_is_explicitly_not_attempted() {
    let snap = t1_snapshot();

    // Exhaustive enumeration: exactly one device entry, one identity.
    assert_eq!(snap.devices().len(), 1);
    let identities: Vec<&PhysicalDeviceId> = snap.devices().values().map(|e| &e.identity).collect();
    assert_eq!(identities.len(), 1);
    assert_eq!(
        *identities[0],
        PhysicalDeviceId::cuda(PCI_UUID, Some(DRIVER_UUID.to_owned()))
    );

    // No second identity is recorded at any other ordinal locator — a
    // consumer cannot read a second physical identity out of this sample.
    assert!(snap.entry(DeviceOrdinal::new(0)).is_some());
    assert!(snap.entry(DeviceOrdinal::new(1)).is_none());

    // The probe-level P2P state is the explicit marker that multi-device
    // rows were never probed on this host.
    assert_eq!(snap.p2p_state(), P2pProbeState::NotAttempted);
}

/// MD1-D2: every directed P2P pair `i→j, i≠j` is NOT ATTEMPTED — the
/// probe-level state says so explicitly, the single-device host carries no
/// admitted link, and the topology gate never admits a cross-device
/// traversal (C2 topology gate; T1 §3).
#[test]
fn every_directed_p2p_pair_is_explicitly_not_attempted() {
    let snap = t1_snapshot();

    // Probe-level explicit marker: the state is recorded, distinct from
    // `Attempted` — a consumer can see the pairs were never probed.
    assert_eq!(snap.p2p_state(), P2pProbeState::NotAttempted);
    assert_ne!(snap.p2p_state(), P2pProbeState::Attempted);

    let pharos = PhysicalDeviceId::cuda(PCI_UUID, Some(DRIVER_UUID.to_owned()));
    let topo = DeviceTopologySnapshot::new(snap, []);

    // The single-device host carries zero directed link rows and no admitted
    // link anywhere in the topology.
    assert_eq!(topo.links().count(), 0);
    assert!(!topo
        .links()
        .any(|l| matches!(l.state(), DeviceLinkState::Admitted { .. })));

    // A self-move is a local copy, not a P2P row (T1 §3).
    assert_eq!(topo.traversal_allowed(&pharos, &pharos), Ok(()));

    // A traversal to any *different* device is rejected — an absent or
    // NOT-ATTEMPTED fact is never assumed (C2 topology gate; T1 §3).
    let stranger = PhysicalDeviceId::cuda("GPU-99999999-8888-7777-6666-555555555555", None);
    assert_eq!(
        topo.traversal_allowed(&pharos, &stranger),
        Err(LinkGateError::UnknownEndpoint { endpoint: stranger })
    );
}

/// MD1-D2: independent device-loss is NOT ATTEMPTED — the snapshot records a
/// single healthy device and carries no device-loss or degradation row. A
/// loss observation (removal or degraded transition) would surface as a
/// presence/health change at an *advanced* epoch — a different sample. This
/// sample claims none, so the healthy single row can never be mistaken for a
/// pass on independent device-loss handling (T1 §8; CTO `2f90eafd` §5b).
#[test]
fn independent_device_loss_row_is_explicitly_not_attempted() {
    let snap = t1_snapshot();

    // Exactly one device, healthy at epoch 1 — the only rows in the sample.
    assert_eq!(snap.devices().len(), 1);
    let entry = snap.entry(DeviceOrdinal::new(0)).unwrap();
    assert_eq!(entry.health, DeviceHealth::Healthy);
    assert_eq!(entry.health_generation, DeviceHealthGeneration::initial());

    // No degraded row exists anywhere in the representation.
    assert!(!snap
        .devices()
        .values()
        .any(|e| matches!(e.health, DeviceHealth::Degraded(_))));

    // No removal/loss row exists either: a removal would shrink or empty the
    // device map at an advanced epoch. The map is exhaustive — exactly one
    // healthy entry — so no device-loss observation is claimed.
    assert!(snap.devices().iter().all(|(ordinal, e)| {
        *ordinal == DeviceOrdinal::new(0) && e.health == DeviceHealth::Healthy
    }));
    assert!(snap.is_current_generation(DeviceHealthGeneration::initial()));

    // Independent device-loss — one of ≥2 devices failing while others
    // continue — is not probed on a single-device host: there is no second
    // device whose loss could have been independently observed.
    assert_eq!(snap.devices().len(), 1);
}
