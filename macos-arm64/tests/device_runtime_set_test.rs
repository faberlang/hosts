//! MD3H-H2: DeviceRuntimeSet composition (M=1 live, M>1 by fake sessions).

use std::collections::BTreeMap;

use faber_host_macos_arm64::device_host::DeviceRuntime;
use faber_host_macos_arm64::{
    CudaHostSession, DeviceRuntimeSet, FakeCudaDriver, FakeMetalDriver, MetalHostSession,
    enumerate_cuda_physical_devices, enumerate_metal_physical_devices, probe_cuda_environment,
    probe_metal_environment,
};
use host_coordinator::DeviceBackend;
use host_coordinator::device_identity::PhysicalDeviceId;

fn fake_metal(id: PhysicalDeviceId) -> (PhysicalDeviceId, DeviceRuntime) {
    let session =
        MetalHostSession::with_driver(Box::new(FakeMetalDriver::default())).expect("fake metal");
    (id, DeviceRuntime::Metal(session))
}

fn fake_cuda(id: PhysicalDeviceId) -> (PhysicalDeviceId, DeviceRuntime) {
    let session =
        CudaHostSession::with_driver(Box::new(FakeCudaDriver::default())).expect("fake cuda");
    (id, DeviceRuntime::Cuda(session))
}

#[test]
fn empty_set_fails_closed() {
    let error = DeviceRuntimeSet::from_members(BTreeMap::new()).expect_err("empty");
    assert_eq!(error.code, "E_INVALID_ARGS");
}

#[test]
fn mixed_metal_cuda_set_fails_closed() {
    let metal = fake_metal(PhysicalDeviceId::metal("1"));
    let cuda = fake_cuda(PhysicalDeviceId::cuda("GPU-aaaa", None));
    let error = DeviceRuntimeSet::from_members([metal, cuda]).expect_err("mixed");
    assert_eq!(error.code, "E_INVALID_ARGS");
    assert!(error.message.contains("one backend"));
}

#[test]
fn m1_composition_holds_one_session_per_id() {
    let id = PhysicalDeviceId::metal("4278190081");
    let set = DeviceRuntimeSet::from_members([fake_metal(id.clone())]).expect("M=1");
    assert_eq!(set.len(), 1);
    assert_eq!(set.backend(), DeviceBackend::Metal);
    assert!(set.contains(&id));
    assert_eq!(
        set.get(&id).map(DeviceRuntime::backend),
        Some(DeviceBackend::Metal)
    );
}

#[test]
fn m_gt_1_shape_is_one_session_per_physical_id() {
    let a = PhysicalDeviceId::metal("4278190081");
    let b = PhysicalDeviceId::metal("4278190082");
    let set = DeviceRuntimeSet::from_members([fake_metal(a.clone()), fake_metal(b.clone())])
        .expect("M>1 composition");
    assert_eq!(set.len(), 2);
    assert_eq!(set.backend(), DeviceBackend::Metal);
    assert!(set.contains(&a));
    assert!(set.contains(&b));
    let ids: Vec<_> = set.ids().cloned().collect();
    assert_eq!(ids, vec![a, b]);
}

#[test]
fn two_cuda_fakes_compose_without_sharing_a_session() {
    let a = PhysicalDeviceId::cuda("GPU-aaa", None);
    let b = PhysicalDeviceId::cuda("GPU-bbb", None);
    let set = DeviceRuntimeSet::from_members([fake_cuda(a.clone()), fake_cuda(b.clone())])
        .expect("M>1 cuda composition");
    assert_eq!(set.len(), 2);
    assert_eq!(set.backend(), DeviceBackend::Cuda);
}

#[test]
fn live_open_is_m1_and_rejects_two_distinct_ids() {
    let a = PhysicalDeviceId::metal("1");
    let b = PhysicalDeviceId::metal("2");
    let error = DeviceRuntimeSet::open_live([a, b]).expect_err("M>1 live");
    assert_eq!(error.code, "E_INVALID_ARGS");
    assert!(error.message.contains("M=1"));
}

#[test]
fn live_open_matches_enumerated_metal_or_fails_closed() {
    let devices = enumerate_metal_physical_devices().expect("metal enum");
    if probe_metal_environment().admitted {
        assert_eq!(devices.len(), 1);
        let id = PhysicalDeviceId::metal(&devices[0].registry_id);
        let set = DeviceRuntimeSet::open_live([id.clone()]).expect("live Metal M=1");
        assert_eq!(set.len(), 1);
        assert_eq!(set.backend(), DeviceBackend::Metal);
        assert!(set.contains(&id));
        eprintln!(
            "MD3H-H2 metal runtime-set receipt: M={} registry_id={}",
            set.len(),
            devices[0].registry_id
        );
    } else {
        let error = DeviceRuntimeSet::open_live([PhysicalDeviceId::metal("missing")])
            .expect_err("not admitted");
        assert_eq!(error.code, "E_INVALID_ARGS");
        eprintln!("MD3H-H2 metal runtime-set receipt: PENDING (Metal not admitted)");
    }
}

#[test]
fn live_open_matches_enumerated_cuda_or_fails_closed() {
    let devices = enumerate_cuda_physical_devices().expect("cuda enum");
    if probe_cuda_environment().admitted {
        assert_eq!(devices.len(), 1);
        let id = PhysicalDeviceId::cuda(&devices[0].pci_uuid, devices[0].driver_uuid.clone());
        let set = DeviceRuntimeSet::open_live([id.clone()]).expect("live CUDA M=1");
        assert_eq!(set.len(), 1);
        assert_eq!(set.backend(), DeviceBackend::Cuda);
        assert!(set.contains(&id));
        eprintln!(
            "MD3H-H2 cuda runtime-set receipt: M={} pci_uuid={}",
            set.len(),
            devices[0].pci_uuid
        );
    } else {
        let error = DeviceRuntimeSet::open_live([PhysicalDeviceId::cuda("GPU-missing", None)])
            .expect_err("not admitted");
        assert_eq!(error.code, "E_INVALID_ARGS");
        eprintln!("MD3H-H2 cuda runtime-set receipt: PENDING (CUDA not admitted)");
    }
}
