//! MD3H-H2: real DeviceExecutionBackend over DeviceRuntimeSet.
//!
//! Composition tests drive fake sessions (M=1 and M>1). Live tests run an
//! ExecutionTransaction prepare/execute/commit/abort against a real Metal
//! session (burgus) and a real CUDA session (pharos); a machine that does
//! not admit the backend records PENDING.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use faber_host_macos_arm64::device_host::DeviceRuntime;
use faber_host_macos_arm64::{
    enumerate_cuda_physical_devices, enumerate_metal_physical_devices, probe_cuda_environment,
    probe_metal_environment, CudaHostSession, DeviceRuntimeBackend, DeviceRuntimeSet,
    FakeCudaDriver, FakeMetalDriver, LaunchProgram, MetalHostSession,
};
use host_coordinator::bound_plan::{
    bind, AdmittedLogicalPlan, BoundDistributedPlan, LogicalPartitionId, PartitionBinding,
};
use host_coordinator::device_identity::{DeviceHealthGeneration, DeviceOrdinal, PhysicalDeviceId};
use host_coordinator::device_set::DeviceSet;
use host_coordinator::discovery::{
    ComputeCapability, DeviceCapabilities, DeviceDiscoveryEntry, DeviceDiscoverySnapshot,
    DeviceHealth, DeviceMemory, DtypeSurface, ProbeProvenance,
};
use host_coordinator::execution_transaction::{
    BarrierRef, CollectiveBroadcastMirror, CollectiveRef, DeviceExecutionBackend,
    ExecutionTransaction, LaunchRef, MirroredDtype, MirroredStorageLayout, PublicationOrdinal,
    TransactionCommitBoundary, TransactionDecision, TransactionId, TransactionOperation,
    TransactionState, TransferDirectionMirror, TransferOperationMirror, TransferRef,
    TransportPathMirror,
};
use host_coordinator::partition::{
    AdmissionRequest, FixtureIdentityClass, PartitionBudgetLedger, SafePhysicalLimit,
    TransportClass, VirtualDevicePartition, VirtualDevicePartitionId,
};
use host_coordinator::transport::{CopyPath, TransferBudget};
use host_coordinator::DeviceBackend;

const PROBE_TIME: u64 = 1_752_717_600_000_000_000;
const LOGICAL_HASH: &str =
    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const CS1_SPLIT_BYTES: u64 = 135_266_304;
const CS1_LIMIT_BYTES: u64 = 167_772_160;
const LAUNCH_A_OUTPUT_BYTES: u64 = 4096;
const TRANSFER_BYTES: u64 = 8192;
const BROADCAST_BYTES: u64 = 2048;
const LAUNCH_B_OUTPUT_BYTES: u64 = 16_384;

const OBSERVA_MSL: &str = r#"#include <metal_stdlib>
using namespace metal;

kernel void observa(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  dst[id] = src[id];
}
"#;

const OBSERVA_PTX: &str = r#"
.version 4.1
.target sm_52
.address_size 64

.visible .entry observa(
	.param .u64 observa_param_0,
	.param .u64 observa_param_1
)
{
	.reg .b32 	%r<5>;
	.reg .f32 	%f<2>;
	.reg .b64 	%rd<6>;

	mov.u32 	%r1, %tid.x;
	mov.u32 	%r2, %ntid.x;
	mov.u32 	%r3, %ctaid.x;
	mad.lo.s32 	%r4, %r2, %r3, %r1;
	ld.param.u64 	%rd1, [observa_param_0];
	ld.param.u64 	%rd2, [observa_param_1];
	mul.wide.u32 	%rd3, %r4, 4;
	add.s64 	%rd4, %rd1, %rd3;
	add.s64 	%rd5, %rd2, %rd3;
	ld.global.f32 	%f1, [%rd4];
	st.global.f32 	[%rd5], %f1;
	ret;
}
"#;

fn partition_id(n: u32) -> LogicalPartitionId {
    LogicalPartitionId::new(format!("partition-{n}"))
}

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

fn vp(
    seed: u64,
    device: PhysicalDeviceId,
    budget: PartitionBudgetLedger,
) -> VirtualDevicePartition {
    VirtualDevicePartition::admit(
        AdmissionRequest::new(VirtualDevicePartitionId::new(seed), device, budget),
        SafePhysicalLimit::new(CS1_LIMIT_BYTES),
    )
    .unwrap()
}

fn entry(ordinal: u32, device: PhysicalDeviceId) -> DeviceDiscoveryEntry {
    DeviceDiscoveryEntry {
        ordinal: DeviceOrdinal::new(ordinal),
        identity: device,
        device_model: Some("synthetic".to_owned()),
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
            api_total_bytes: 8 * 1024 * 1024 * 1024,
        },
        health: DeviceHealth::Healthy,
        health_generation: DeviceHealthGeneration::initial(),
        probe_provenance: ProbeProvenance {
            probe: "md3h-h2 fixture".to_owned(),
            tool_versions: "synthetic".to_owned(),
        },
    }
}

fn plan_on(devices: &[PhysicalDeviceId]) -> BoundDistributedPlan {
    assert!(!devices.is_empty());
    let p0_device = devices[0].clone();
    let p1_device = devices
        .get(1)
        .cloned()
        .unwrap_or_else(|| devices[0].clone());
    let snapshot = DeviceDiscoverySnapshot::from_enumerated(
        PROBE_TIME,
        devices
            .iter()
            .enumerate()
            .map(|(index, device)| entry(index as u32, device.clone())),
    );
    let admitted = AdmittedLogicalPlan::admit(LOGICAL_HASH, [partition_id(0), partition_id(1)], [])
        .expect("two-partition plan admits");
    let bindings = BTreeMap::from([
        (
            partition_id(0),
            PartitionBinding::with_virtual_partition(
                p0_device.clone(),
                vp(1, p0_device.clone(), ledger(10_240, 8_448)),
            ),
        ),
        (
            partition_id(1),
            PartitionBinding::with_virtual_partition(
                p1_device.clone(),
                vp(2, p1_device.clone(), ledger(10_240, 30_976)),
            ),
        ),
    ]);
    let unique: BTreeSet<_> = devices.iter().cloned().collect();
    bind(
        &admitted,
        bindings,
        DeviceSet::from_members(unique),
        &snapshot,
        DeviceHealthGeneration::initial(),
        FixtureIdentityClass::Virtual,
        TransportClass::HostStaged,
    )
    .expect("plan binds")
}

fn transfer_op() -> TransactionOperation {
    TransactionOperation::transfer(TransferOperationMirror::new(
        TransferRef::new("t1"),
        partition_id(0),
        partition_id(1),
        TRANSFER_BYTES,
        TransferDirectionMirror::BIDI,
        MirroredDtype::F32,
        MirroredStorageLayout::Dense,
        TransportPathMirror::HostStaged,
        0,
        1,
        TransactionCommitBoundary::default(),
    ))
}

fn fixture_operations() -> Vec<TransactionOperation> {
    vec![
        TransactionOperation::launch(
            partition_id(0),
            LaunchRef::new("launch-proj-a"),
            LAUNCH_A_OUTPUT_BYTES,
        ),
        transfer_op(),
        TransactionOperation::broadcast(CollectiveBroadcastMirror::broadcast(
            CollectiveRef::new("c1"),
            partition_id(0),
            BTreeSet::from([partition_id(0), partition_id(1)]),
            BROADCAST_BYTES,
        )),
        TransactionOperation::barrier(
            BarrierRef::new("barrier-main"),
            BTreeSet::from([partition_id(0), partition_id(1)]),
        ),
        TransactionOperation::launch(
            partition_id(1),
            LaunchRef::new("launch-proj-b"),
            LAUNCH_B_OUTPUT_BYTES,
        ),
    ]
}

fn fixture_boundary() -> TransactionCommitBoundary {
    TransactionCommitBoundary::new(
        [BarrierRef::new("barrier-main")],
        [LaunchRef::new("launch-proj-b")],
    )
}

fn declared_write_bytes() -> u64 {
    fixture_operations()
        .iter()
        .flat_map(TransactionOperation::staged_writes)
        .map(|write| write.byte_count())
        .sum()
}

fn fake_program() -> LaunchProgram {
    LaunchProgram::new(b"// fake observa image".to_vec(), "observa")
}

fn live_metal_program() -> LaunchProgram {
    LaunchProgram::new(OBSERVA_MSL.as_bytes().to_vec(), "observa")
}

fn live_cuda_program() -> LaunchProgram {
    LaunchProgram::new(OBSERVA_PTX.as_bytes().to_vec(), "observa")
}

fn partition_map(plan: &BoundDistributedPlan) -> BTreeMap<LogicalPartitionId, PhysicalDeviceId> {
    plan.bindings()
        .expect("distributed")
        .iter()
        .map(|(partition, binding)| (partition.clone(), binding.device().clone()))
        .collect()
}

fn backend_over(
    set: DeviceRuntimeSet,
    plan: &BoundDistributedPlan,
    launch: LaunchProgram,
) -> DeviceRuntimeBackend {
    DeviceRuntimeBackend::new(
        set,
        partition_map(plan),
        launch,
        TransferBudget::declared(1 << 20, 10_000_000_000),
        Duration::from_secs(30),
    )
    .expect("backend constructs")
}

fn fake_metal_runtime() -> DeviceRuntime {
    DeviceRuntime::Metal(
        MetalHostSession::with_driver(Box::new(FakeMetalDriver::default())).expect("fake metal"),
    )
}

fn fake_cuda_runtime() -> DeviceRuntime {
    DeviceRuntime::Cuda(
        CudaHostSession::with_driver(Box::new(FakeCudaDriver::default())).expect("fake cuda"),
    )
}

fn run_commit(
    plan: BoundDistributedPlan,
    set: DeviceRuntimeSet,
    launch: LaunchProgram,
) -> (u64, u64, usize) {
    let mut backend = backend_over(set, &plan, launch);
    let mut transaction = ExecutionTransaction::new(
        TransactionId::new("txn-h2"),
        plan,
        fixture_operations(),
        fixture_boundary(),
    )
    .expect("constructs");
    transaction.prepare(&mut backend).expect("prepare");
    transaction.execute(&mut backend).expect("execute");
    let receipt = transaction
        .commit(&mut backend, PublicationOrdinal::new(1))
        .expect("commit");
    assert_eq!(transaction.state(), &TransactionState::Committed);
    assert!(matches!(
        receipt.decision,
        TransactionDecision::Committed { .. }
    ));
    let staged = backend.staged_bytes();
    let published = backend.published_bytes();
    assert_eq!(staged, declared_write_bytes());
    assert_eq!(published, staged);
    let host_staged = backend
        .transport_receipt()
        .records
        .iter()
        .filter(|record| record.copy_path == CopyPath::HostStaged)
        .count();
    assert!(
        host_staged >= 2,
        "transfer + broadcast must be labeled host-staged"
    );
    for record in &backend.transport_receipt().records {
        assert_eq!(record.copy_path, CopyPath::HostStaged);
        assert!(record.elapsed_nanos > 0 || record.bytes > 0);
    }
    (staged, published, host_staged)
}

fn run_abort(plan: BoundDistributedPlan, set: DeviceRuntimeSet, launch: LaunchProgram) {
    let mut backend = backend_over(set, &plan, launch);
    let mut transaction = ExecutionTransaction::new(
        TransactionId::new("txn-h2-abort"),
        plan,
        fixture_operations(),
        fixture_boundary(),
    )
    .expect("constructs");
    transaction.prepare(&mut backend).expect("prepare");
    transaction.execute(&mut backend).expect("execute");
    assert!(backend.staged_bytes() > 0);
    assert_eq!(backend.published_bytes(), 0);
    let receipt = transaction
        .abort(&mut backend, "operator abort")
        .expect("abort");
    assert!(matches!(
        receipt.decision,
        TransactionDecision::Aborted { .. }
    ));
    assert_eq!(backend.published_bytes(), 0);
    assert_eq!(backend.published_write_set().len(), 0);
}

#[test]
fn m1_fake_metal_transaction_commits_with_byte_exact_accounting() {
    let id = PhysicalDeviceId::metal("4278190081");
    let plan = plan_on(&[id.clone()]);
    let set = DeviceRuntimeSet::from_members([(id, fake_metal_runtime())]).expect("M=1");
    assert_eq!(set.len(), 1);
    let (staged, published, _) = run_commit(plan, set, fake_program());
    assert_eq!(staged, published);
}

#[test]
fn m_gt_1_fake_metal_composition_commits() {
    let a = PhysicalDeviceId::metal("4278190081");
    let b = PhysicalDeviceId::metal("4278190082");
    let plan = plan_on(&[a.clone(), b.clone()]);
    let set =
        DeviceRuntimeSet::from_members([(a, fake_metal_runtime()), (b, fake_metal_runtime())])
            .expect("M>1");
    assert_eq!(set.len(), 2);
    let (staged, published, host_staged) = run_commit(plan, set, fake_program());
    assert_eq!(staged, published);
    assert!(host_staged >= 2);
}

#[test]
fn m_gt_1_fake_cuda_composition_commits() {
    let a = PhysicalDeviceId::cuda("GPU-aaa", None);
    let b = PhysicalDeviceId::cuda("GPU-bbb", None);
    let plan = plan_on(&[a.clone(), b.clone()]);
    let set = DeviceRuntimeSet::from_members([(a, fake_cuda_runtime()), (b, fake_cuda_runtime())])
        .expect("M>1 cuda");
    let (staged, published, _) = run_commit(plan, set, fake_program());
    assert_eq!(staged, published);
}

#[test]
fn abort_retires_without_publication() {
    let id = PhysicalDeviceId::metal("4278190081");
    let plan = plan_on(&[id.clone()]);
    let set = DeviceRuntimeSet::from_members([(id, fake_metal_runtime())]).expect("M=1");
    run_abort(plan, set, fake_program());
}

#[test]
fn live_metal_transaction_or_pending() {
    if !probe_metal_environment().admitted {
        eprintln!("MD3H-H2 metal transaction receipt: PENDING (Metal not admitted)");
        return;
    }
    let devices = enumerate_metal_physical_devices().expect("enum");
    assert_eq!(devices.len(), 1);
    let id = PhysicalDeviceId::metal(&devices[0].registry_id);
    let plan = plan_on(&[id.clone()]);
    let set = DeviceRuntimeSet::open_live([id]).expect("live metal set");
    assert_eq!(set.backend(), DeviceBackend::Metal);
    let (staged, published, host_staged) = run_commit(plan, set, live_metal_program());
    eprintln!(
        "MD3H-H2 metal transaction receipt: staged={staged} published={published} host_staged={host_staged} registry_id={}",
        devices[0].registry_id
    );
}

#[test]
fn live_metal_abort_or_pending() {
    if !probe_metal_environment().admitted {
        eprintln!("MD3H-H2 metal abort receipt: PENDING (Metal not admitted)");
        return;
    }
    let devices = enumerate_metal_physical_devices().expect("enum");
    let id = PhysicalDeviceId::metal(&devices[0].registry_id);
    let plan = plan_on(&[id.clone()]);
    let set = DeviceRuntimeSet::open_live([id]).expect("live metal set");
    run_abort(plan, set, live_metal_program());
    eprintln!("MD3H-H2 metal abort receipt: published=0");
}

#[test]
fn live_cuda_transaction_or_pending() {
    if !probe_cuda_environment().admitted {
        eprintln!("MD3H-H2 cuda transaction receipt: PENDING (CUDA not admitted)");
        return;
    }
    let devices = enumerate_cuda_physical_devices().expect("enum");
    assert_eq!(devices.len(), 1);
    let id = PhysicalDeviceId::cuda(&devices[0].pci_uuid, devices[0].driver_uuid.clone());
    let plan = plan_on(&[id.clone()]);
    let set = DeviceRuntimeSet::open_live([id]).expect("live cuda set");
    assert_eq!(set.backend(), DeviceBackend::Cuda);
    let (staged, published, host_staged) = run_commit(plan, set, live_cuda_program());
    eprintln!(
        "MD3H-H2 cuda transaction receipt: staged={staged} published={published} host_staged={host_staged} pci_uuid={}",
        devices[0].pci_uuid
    );
}

#[test]
fn live_cuda_abort_or_pending() {
    if !probe_cuda_environment().admitted {
        eprintln!("MD3H-H2 cuda abort receipt: PENDING (CUDA not admitted)");
        return;
    }
    let devices = enumerate_cuda_physical_devices().expect("enum");
    let id = PhysicalDeviceId::cuda(&devices[0].pci_uuid, devices[0].driver_uuid.clone());
    let plan = plan_on(&[id.clone()]);
    let set = DeviceRuntimeSet::open_live([id]).expect("live cuda set");
    run_abort(plan, set, live_cuda_program());
    eprintln!("MD3H-H2 cuda abort receipt: published=0");
}
