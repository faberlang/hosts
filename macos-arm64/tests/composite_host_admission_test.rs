//! MD3H-H3: every product device run admits through one virtual partition.
//!
//! Structural: DeviceSelection stays backend-kind-only; a device-carrying
//! CompositeHost constructs exactly one implicit_local admission; N=1 binds
//! BoundPlanKind::ImplicitLocal with an empty communication graph; a CPU-only
//! host (no admission) cannot execute.

use std::collections::BTreeMap;

use faber_host_macos_arm64::composite_host::{
    implicit_local_n1_logical_hash, resolve_device_selection, BoundPlanKind, CompositeHost,
    CompositeHostConfig, DeviceSelection,
};
use faber_host_macos_arm64::device_descriptor::{
    DescriptorBuffer, DescriptorBufferVersion, DescriptorKernel, DescriptorLaunch,
    DescriptorResult, DeviceBufferInitialization, DeviceBufferLifetime, DeviceBufferRole,
    DeviceDataType, DeviceDescriptor, DeviceProgramLifetime, E_NO_DEVICE_PROGRAM,
};
use faber_host_macos_arm64::device_host::DeviceRuntime;
use faber_host_macos_arm64::{CudaHostSession, FakeCudaDriver, FakeMetalDriver, MetalHostSession};
use host_coordinator::partition::{FixtureIdentityClass, HardwareIsolationClaim, TransportClass};
use host_coordinator::DeviceBackend;

const MODULE_IMAGE: &[u8] = b"// fake compiler-owned module image";

#[test]
fn device_selection_is_backend_kind_only() {
    // Exhaustive match: a rank/ordinal/device-id variant is a compile error.
    match DeviceSelection::Auto {
        DeviceSelection::Auto | DeviceSelection::Metal | DeviceSelection::Cuda => {}
    }
    assert_eq!(DeviceSelection::Auto.spelling(), "auto");
    assert_eq!(DeviceSelection::Metal.spelling(), "metal");
    assert_eq!(DeviceSelection::Cuda.spelling(), "cuda");
    assert_eq!(DeviceSelection::Auto.backend(), None);
    assert_eq!(DeviceSelection::Metal.backend(), Some(DeviceBackend::Metal));
    assert_eq!(DeviceSelection::Cuda.backend(), Some(DeviceBackend::Cuda));
    assert_eq!(
        resolve_device_selection(DeviceSelection::Metal, true, &[DeviceBackend::Metal])
            .expect("explicit metal"),
        Some(DeviceBackend::Metal)
    );
}

#[test]
fn cpu_only_host_has_no_admission_and_cannot_execute() {
    let mut host = CompositeHost::new(CompositeHostConfig::cpu()).expect("cpu");
    assert!(host.bound_plan().is_none());
    assert!(host.runtime_set().is_none());
    let error = host
        .require_implicit_local()
        .expect_err("cpu-only has no partition admission");
    assert_eq!(error.code, E_NO_DEVICE_PROGRAM);
    match host.create_program_session(&elementwise_add(DeviceBackend::Metal)) {
        Ok(_) => panic!("cpu-only host must refuse a bypassing program session"),
        Err(error) => assert_eq!(error.code, E_NO_DEVICE_PROGRAM),
    }
}

#[test]
fn metal_product_session_admits_exactly_one_implicit_local_partition() {
    let host = metal_host();
    assert_one_n1_admission(&host, DeviceBackend::Metal, FixtureIdentityClass::Synthetic);
}

#[test]
fn cuda_product_session_admits_exactly_one_implicit_local_partition() {
    let host = cuda_host();
    assert_one_n1_admission(&host, DeviceBackend::Cuda, FixtureIdentityClass::Synthetic);
}

#[test]
fn n1_parity_keeps_numeric_session_and_leak_bars_on_both_backends() {
    let inputs = BTreeMap::from([(1, vec![1.0, 2.0]), (2, vec![3.0, 4.0])]);
    for (backend, mut host) in [
        (DeviceBackend::Metal, metal_host()),
        (DeviceBackend::Cuda, cuda_host()),
    ] {
        let descriptor = elementwise_add(backend);
        let mut session = host
            .create_program_session(&descriptor)
            .expect("N=1 session admission");
        let first = session.execute(&inputs).expect("first numeric execution");
        let second = session.execute(&inputs).expect("second numeric execution");

        for receipt in [first, second] {
            assert_eq!(receipt.outputs.get(&3), Some(&vec![4.0, 6.0]));
            assert_eq!(receipt.launches, 1);
            assert_eq!(receipt.copy_ins, 2);
            assert_eq!(receipt.readbacks, 1);
            assert_eq!(receipt.transfers, 3);
        }
        session.teardown().expect("N=1 session teardown");
        assert_eq!(host.device().expect("device").live_handle_count(), 0);
    }
}

#[test]
fn n1_bound_plan_has_zero_communication_ops() {
    let host = metal_host();
    let plan = host.bound_plan().expect("bound plan");
    assert!(plan.is_degenerate());
    assert!(plan.bindings().is_none());
    assert_eq!(plan.transport_class(), TransportClass::None);
    assert_eq!(plan.device_set().len(), 1);
    let set = host.runtime_set().expect("runtime set");
    assert_eq!(set.len(), 1);
    assert_eq!(set.backend(), DeviceBackend::Metal);
    assert_eq!(
        host.device().map(DeviceRuntime::backend),
        Some(DeviceBackend::Metal)
    );
}

#[test]
fn admitted_session_executes_without_inventing_copies() {
    let mut host = metal_host();
    let mut inputs = BTreeMap::new();
    inputs.insert(1, vec![1.0, 2.0]);
    inputs.insert(2, vec![3.0, 4.0]);
    let receipt = host
        .execute_descriptor(&elementwise_add(DeviceBackend::Metal), &inputs)
        .expect("N=1 execute");
    assert_eq!(receipt.launches, 1);
    assert_eq!(receipt.copy_ins, 2);
    assert_eq!(receipt.outputs.get(&3), Some(&vec![4.0, 6.0]));
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

fn assert_one_n1_admission(
    host: &CompositeHost,
    backend: DeviceBackend,
    fixture: FixtureIdentityClass,
) {
    let plan = host
        .require_implicit_local()
        .expect("device host must admit");
    assert_eq!(
        plan.logical_distributed_plan_hash(),
        implicit_local_n1_logical_hash()
    );
    match plan.kind() {
        BoundPlanKind::ImplicitLocal {
            virtual_partition: Some(partition),
            ..
        } => {
            assert!(partition.is_active());
            assert_eq!(partition.id().get(), 1);
            assert_eq!(partition.bound_device().backend(), backend);
        }
        BoundPlanKind::ImplicitLocal {
            virtual_partition: None,
            ..
        } => panic!("implicit-local bind must carry the admitted partition"),
        BoundPlanKind::Distributed { .. } => {
            panic!("N=1 must not produce a distributed wrapper")
        }
    }
    let receipt = plan.receipt();
    assert_eq!(receipt.physical_device_count(), 1);
    assert_eq!(receipt.virtual_partition_count(), 1);
    assert_eq!(receipt.fixture_identity_class(), fixture);
    assert_eq!(receipt.transport_class(), TransportClass::None);
    assert_eq!(
        receipt.hardware_isolation_claimed(),
        HardwareIsolationClaim::NotClaimed
    );
    assert!(plan.bindings().is_none());
}

fn metal_host() -> CompositeHost {
    let runtime = DeviceRuntime::Metal(
        MetalHostSession::with_driver(Box::new(
            FakeMetalDriver::default().with_known_entry("add_one"),
        ))
        .expect("fake metal"),
    );
    CompositeHost::with_device(runtime, "fake-metal-device").expect("admitted metal host")
}

fn cuda_host() -> CompositeHost {
    let runtime = DeviceRuntime::Cuda(
        CudaHostSession::with_driver(Box::new(
            FakeCudaDriver::default().with_known_entry("add_one"),
        ))
        .expect("fake cuda"),
    );
    CompositeHost::with_device(runtime, "fake-cuda-device").expect("admitted cuda host")
}

fn elementwise_add(backend: DeviceBackend) -> DeviceDescriptor {
    let kernels = vec![DescriptorKernel {
        entry: "add_one".to_owned(),
        buffers: vec![
            slot(1, "a", DeviceBufferRole::Input, 0, 2),
            slot(2, "b", DeviceBufferRole::Input, 1, 2),
            slot(3, "out", DeviceBufferRole::Output, 2, 2),
        ],
        grid: [1, 1, 1],
        block: [2, 1, 1],
    }];
    let launches = vec![DescriptorLaunch {
        id: 1,
        kernel_index: 0,
    }];
    DeviceDescriptor {
        backend,
        module_image: MODULE_IMAGE.to_vec(),
        buffer_versions: buffer_versions(&kernels),
        kernels,
        launches,
        program_lifetime: DeviceProgramLifetime::SingleRun,
        data_flow: Vec::new(),
        roots: vec![1],
        results: vec![DescriptorResult {
            buffer_id: 3,
            version: 1,
            produced_by: 1,
            at_launch: 1,
        }],
        end_of_run_results: Vec::new(),
    }
}

fn slot(id: u32, name: &str, role: DeviceBufferRole, binding: u32, count: u64) -> DescriptorBuffer {
    DescriptorBuffer {
        buffer_id: id,
        buffer_name: name.to_owned(),
        semantic_value: id,
        role,
        lifetime: match role {
            DeviceBufferRole::Input => DeviceBufferLifetime::PerProgram,
            DeviceBufferRole::Output => DeviceBufferLifetime::ObservationPoint,
            DeviceBufferRole::InOut => DeviceBufferLifetime::PerStep,
        },
        initialization: match role {
            DeviceBufferRole::Input => DeviceBufferInitialization::HostProvided,
            DeviceBufferRole::InOut => DeviceBufferInitialization::ZeroFill,
            DeviceBufferRole::Output => DeviceBufferInitialization::KernelInitialized,
        },
        binding,
        element_ty: DeviceDataType::F32,
        element_count: count,
        version: 1,
    }
}

fn buffer_versions(kernels: &[DescriptorKernel]) -> Vec<DescriptorBufferVersion> {
    let mut versions = Vec::new();
    for kernel in kernels {
        for item in &kernel.buffers {
            if versions.iter().any(|version: &DescriptorBufferVersion| {
                version.buffer_id == item.buffer_id && version.version == item.version
            }) {
                continue;
            }
            versions.push(DescriptorBufferVersion {
                buffer_id: item.buffer_id,
                version: item.version,
                element_ty: item.element_ty,
                element_count: item.element_count,
            });
        }
    }
    versions
}
