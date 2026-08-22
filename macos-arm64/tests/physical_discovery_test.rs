//! MD3H-H1: per-ordinal CUDA/Metal physical discovery into host-coordinator
//! snapshot facts. Ordinals are locators, never identity.

use faber_host_macos_arm64::{
    discover_cuda_snapshot, discover_metal_snapshot, enumerate_cuda_physical_devices,
    enumerate_metal_physical_devices, probe_cuda_environment, probe_metal_environment,
    CudaPhysicalDevice, MetalPhysicalDevice,
};
use host_coordinator::device_identity::{DeviceOrdinal, IdentityChange, PhysicalDeviceId};
use host_coordinator::discovery::DeviceDiscoverySnapshot;
use host_coordinator::DeviceBackend;

const PROBE_TIME: u64 = 1_752_717_600_000_000_000;

fn synthetic_cuda(ordinal: u32, pci_uuid: &str, driver_uuid: Option<&str>) -> CudaPhysicalDevice {
    CudaPhysicalDevice {
        ordinal,
        pci_uuid: pci_uuid.to_owned(),
        driver_uuid: driver_uuid.map(ToOwned::to_owned),
        device_model: Some(format!("synthetic-cuda-{ordinal}")),
        tool_report_total_mib: Some(12_227),
        api_total_bytes: 12_343_705_600,
        compute_capability_major: 12,
        compute_capability_minor: 0,
        sm_count: 48,
        max_threads_per_workgroup: 1024,
        workgroup_shared_memory_min_bytes: 49_152,
        workgroup_shared_memory_max_bytes: 101_376,
        collective_width: 32,
        unified_memory: false,
        driver_version: Some("595.71.05".to_owned()),
    }
}

fn synthetic_metal(ordinal: u32, registry_id: &str) -> MetalPhysicalDevice {
    MetalPhysicalDevice {
        ordinal,
        registry_id: registry_id.to_owned(),
        device_model: Some(format!("synthetic-metal-{ordinal}")),
        api_total_bytes: 36_123_000_000,
        max_threads_per_workgroup: 1024,
        workgroup_shared_memory_min_bytes: 32_768,
        workgroup_shared_memory_max_bytes: 32_768,
        collective_width: 32,
        unified_memory: true,
    }
}

#[test]
fn two_same_backend_cuda_devices_are_distinguishable() {
    let a = synthetic_cuda(
        0,
        "GPU-3e017562-9ec3-da9a-962d-b8bd5f9e24be",
        Some("3e017562-9ec3-da9a-962d-b8bd5f9e24be"),
    );
    let b = synthetic_cuda(1, "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", None);
    let snap = DeviceDiscoverySnapshot::from_enumerated(
        PROBE_TIME,
        [a.to_discovery_entry(), b.to_discovery_entry()],
    );
    assert_eq!(snap.devices().len(), 2);
    let id_a = &snap
        .entry(DeviceOrdinal::new(0))
        .expect("ordinal 0")
        .identity;
    let id_b = &snap
        .entry(DeviceOrdinal::new(1))
        .expect("ordinal 1")
        .identity;
    assert_eq!(id_a.backend(), DeviceBackend::Cuda);
    assert_eq!(id_b.backend(), DeviceBackend::Cuda);
    assert_ne!(id_a, id_b);
    assert_eq!(
        id_a,
        &PhysicalDeviceId::cuda(
            "GPU-3e017562-9ec3-da9a-962d-b8bd5f9e24be",
            Some("3e017562-9ec3-da9a-962d-b8bd5f9e24be".to_owned())
        )
    );
}

#[test]
fn two_same_backend_metal_devices_are_distinguishable() {
    let a = synthetic_metal(0, "4278190081");
    let b = synthetic_metal(1, "4278190082");
    let snap = DeviceDiscoverySnapshot::from_enumerated(
        PROBE_TIME,
        [a.to_discovery_entry(), b.to_discovery_entry()],
    );
    let id_a = &snap
        .entry(DeviceOrdinal::new(0))
        .expect("ordinal 0")
        .identity;
    let id_b = &snap
        .entry(DeviceOrdinal::new(1))
        .expect("ordinal 1")
        .identity;
    assert_eq!(id_a.backend(), DeviceBackend::Metal);
    assert_eq!(id_b.backend(), DeviceBackend::Metal);
    assert_ne!(id_a, id_b);
}

#[test]
fn ordinal_reuse_never_merges_cuda_or_metal_identities() {
    let first = DeviceDiscoverySnapshot::from_enumerated(
        PROBE_TIME,
        [
            synthetic_cuda(0, "GPU-3e017562-9ec3-da9a-962d-b8bd5f9e24be", None)
                .to_discovery_entry(),
        ],
    );
    let reused = DeviceDiscoverySnapshot::from_enumerated(
        PROBE_TIME + 1,
        [
            synthetic_cuda(0, "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", None)
                .to_discovery_entry(),
        ],
    );
    let old = &first.entry(DeviceOrdinal::new(0)).expect("first").identity;
    let new = &reused
        .entry(DeviceOrdinal::new(0))
        .expect("reused")
        .identity;
    assert_ne!(old, new);
    assert_eq!(new.change_against(old), IdentityChange::Replaced);

    let metal_first = DeviceDiscoverySnapshot::from_enumerated(
        PROBE_TIME,
        [synthetic_metal(0, "111").to_discovery_entry()],
    );
    let metal_reused = DeviceDiscoverySnapshot::from_enumerated(
        PROBE_TIME + 1,
        [synthetic_metal(0, "222").to_discovery_entry()],
    );
    let old_m = &metal_first
        .entry(DeviceOrdinal::new(0))
        .expect("metal first")
        .identity;
    let new_m = &metal_reused
        .entry(DeviceOrdinal::new(0))
        .expect("metal reused")
        .identity;
    assert_ne!(old_m, new_m);
    assert_eq!(new_m.change_against(old_m), IdentityChange::Replaced);
}

#[test]
fn enumerates_local_metal_devices_into_discovery_snapshot() {
    let devices = enumerate_metal_physical_devices().expect("metal enumeration is fail-closed");
    let snap = discover_metal_snapshot(PROBE_TIME).expect("metal snapshot");
    assert_eq!(snap.devices().len(), devices.len());

    if probe_metal_environment().admitted {
        assert_eq!(
            devices.len(),
            1,
            "burgus (and admitted Metal hosts) enumerate exactly one Metal device"
        );
        let entry = snap
            .entry(DeviceOrdinal::new(0))
            .expect("Metal ordinal 0 present");
        assert_eq!(entry.backend(), DeviceBackend::Metal);
        assert_eq!(
            entry.identity,
            PhysicalDeviceId::metal(&devices[0].registry_id)
        );
        assert!(!devices[0].registry_id.is_empty());
        assert!(entry.memory.tool_report_total_mib.is_none());
        assert!(entry.memory.api_total_bytes > 0);
        assert_eq!(entry.ordinal, DeviceOrdinal::new(0));
        eprintln!(
            "MD3H-H1 metal receipt: count={} registry_id={} model={:?} api_total_bytes={} snapshot={}",
            devices.len(),
            devices[0].registry_id,
            devices[0].device_model,
            entry.memory.api_total_bytes,
            snap.id().hex()
        );
    } else {
        assert!(devices.is_empty());
        assert!(snap.devices().is_empty());
        eprintln!("MD3H-H1 metal receipt: count=0 (not admitted)");
    }
}

#[test]
fn enumerates_local_cuda_devices_into_discovery_snapshot() {
    let devices = enumerate_cuda_physical_devices().expect("cuda enumeration is fail-closed");
    let snap = discover_cuda_snapshot(PROBE_TIME).expect("cuda snapshot");
    assert_eq!(snap.devices().len(), devices.len());

    if probe_cuda_environment().admitted {
        assert_eq!(
            devices.len(),
            1,
            "pharos (and admitted CUDA hosts in this campaign) enumerate exactly one CUDA device"
        );
        let entry = snap
            .entry(DeviceOrdinal::new(0))
            .expect("CUDA ordinal 0 present");
        assert_eq!(entry.backend(), DeviceBackend::Cuda);
        assert_eq!(
            entry.identity,
            PhysicalDeviceId::cuda(&devices[0].pci_uuid, devices[0].driver_uuid.clone())
        );
        assert!(devices[0].pci_uuid.starts_with("GPU-"));
        assert!(entry.memory.api_total_bytes > 0);
        assert_eq!(entry.ordinal, DeviceOrdinal::new(0));
        eprintln!(
            "MD3H-H1 cuda receipt: count={} pci_uuid={} driver_uuid={:?} model={:?} api_total_bytes={} tool_mib={:?} snapshot={}",
            devices.len(),
            devices[0].pci_uuid,
            devices[0].driver_uuid,
            devices[0].device_model,
            entry.memory.api_total_bytes,
            entry.memory.tool_report_total_mib,
            snap.id().hex()
        );
    } else {
        assert!(devices.is_empty());
        assert!(snap.devices().is_empty());
        eprintln!("MD3H-H1 cuda receipt: count=0 (not admitted)");
    }
}
