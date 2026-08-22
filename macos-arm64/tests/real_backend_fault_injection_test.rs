//! MD3H-X1: fault injection through the real H2 `DeviceExecutionBackend`.
//!
//! The wrapper below only implements the H2 trait and delegates every
//! non-injected call to `DeviceRuntimeBackend`. The transaction therefore
//! drives the real Metal/CUDA runtime for all operations before the injected
//! failure. A failed transaction must retire staged work without replacing the
//! last committed publication.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use faber_host_macos_arm64::{
    enumerate_cuda_physical_devices, enumerate_metal_physical_devices, probe_cuda_environment,
    probe_metal_environment, DeviceRuntimeBackend, DeviceRuntimeSet, LaunchProgram,
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
    BackendError, BarrierRef, CollectiveBroadcastMirror, CollectiveRef, DeviceExecutionBackend,
    ExecuteError, ExecutionTransaction, LaunchRef, MirroredDtype, MirroredStorageLayout,
    OperationEvent, OutputRef, ReservationRecord, StagedWrite, TransactionCommitBoundary,
    TransactionDecision, TransactionFailure, TransactionId, TransactionOperation,
    TransactionReceipt, TransactionState, TransferDirectionMirror, TransferOperationMirror,
    TransferRef, TransportPathMirror,
};
use host_coordinator::partition::{
    AdmissionRequest, FixtureIdentityClass, PartitionBudgetLedger, SafePhysicalLimit,
    TransportClass, VirtualDevicePartition, VirtualDevicePartitionId,
};
use host_coordinator::transport::TransferBudget;
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
    .reg .b32 %r<5>;
    .reg .f32 %f<2>;
    .reg .b64 %rd<6>;

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %ctaid.x;
    mad.lo.s32 %r4, %r2, %r3, %r1;
    ld.param.u64 %rd1, [observa_param_0];
    ld.param.u64 %rd2, [observa_param_1];
    mul.wide.u32 %rd3, %r4, 4;
    add.s64 %rd4, %rd1, %rd3;
    add.s64 %rd5, %rd2, %rd3;
    ld.global.f32 %f1, [%rd4];
    st.global.f32 [%rd5], %f1;
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

fn virtual_partition(
    seed: u64,
    device: PhysicalDeviceId,
    budget: PartitionBudgetLedger,
) -> VirtualDevicePartition {
    VirtualDevicePartition::admit(
        AdmissionRequest::new(VirtualDevicePartitionId::new(seed), device, budget),
        SafePhysicalLimit::new(CS1_LIMIT_BYTES),
    )
    .expect("fixture partition admits")
}

fn discovery_entry(ordinal: u32, device: PhysicalDeviceId) -> DeviceDiscoveryEntry {
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
            probe: "md3h-x1 real fault fixture".to_owned(),
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
            .map(|(index, device)| discovery_entry(index as u32, device.clone())),
    );
    let admitted = AdmittedLogicalPlan::admit(LOGICAL_HASH, [partition_id(0), partition_id(1)], [])
        .expect("two-partition plan admits");
    let bindings = BTreeMap::from([
        (
            partition_id(0),
            PartitionBinding::with_virtual_partition(
                p0_device.clone(),
                virtual_partition(1, p0_device.clone(), ledger(10_240, 8_448)),
            ),
        ),
        (
            partition_id(1),
            PartitionBinding::with_virtual_partition(
                p1_device.clone(),
                virtual_partition(2, p1_device.clone(), ledger(10_240, 30_976)),
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

fn fixture_operations() -> Vec<TransactionOperation> {
    vec![
        TransactionOperation::launch(
            partition_id(0),
            LaunchRef::new("launch-proj-a"),
            LAUNCH_A_OUTPUT_BYTES,
        ),
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
        )),
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

fn partition_map(plan: &BoundDistributedPlan) -> BTreeMap<LogicalPartitionId, PhysicalDeviceId> {
    plan.bindings()
        .expect("distributed plan")
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
    .expect("real backend constructs")
}

fn program_for(backend: DeviceBackend) -> LaunchProgram {
    match backend {
        DeviceBackend::Metal => LaunchProgram::new(OBSERVA_MSL.as_bytes().to_vec(), "observa"),
        DeviceBackend::Cuda => LaunchProgram::new(OBSERVA_PTX.as_bytes().to_vec(), "observa"),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FaultKind {
    Cancelled,
    Timeout,
    Kernel,
    Transfer,
    DeviceLoss,
}

impl FaultKind {
    fn label(self) -> &'static str {
        match self {
            Self::Cancelled => "cancelled",
            Self::Timeout => "timeout",
            Self::Kernel => "kernel",
            Self::Transfer => "transfer",
            Self::DeviceLoss => "device-loss",
        }
    }

    fn matches(self, operation: &TransactionOperation) -> bool {
        match self {
            Self::Transfer => matches!(operation, TransactionOperation::Transfer(_)),
            Self::Cancelled | Self::Timeout | Self::Kernel | Self::DeviceLoss => {
                matches!(
                    operation,
                    TransactionOperation::Launch {
                        partition,
                        launch_ref,
                        ..
                    } if *partition == partition_id(1) && launch_ref.as_str() == "launch-proj-b"
                )
            }
        }
    }

    fn error(self) -> BackendError {
        let partition = partition_id(1);
        match self {
            Self::Cancelled => {
                BackendError::cancelled(partition, "injected cancellation at launch:launch-proj-b")
            }
            Self::Timeout => {
                BackendError::timeout(partition, "injected timeout at launch:launch-proj-b")
            }
            Self::Kernel => BackendError::operation(
                partition,
                "injected kernel failure at launch:launch-proj-b",
            ),
            Self::Transfer => {
                BackendError::operation(partition, "injected transfer failure at transfer:t1")
            }
            Self::DeviceLoss => {
                BackendError::device_loss(partition, "injected device loss at launch:launch-proj-b")
            }
        }
    }
}

/// Test-only fault injection at the H2 trait boundary. It deliberately does
/// not fork or modify `DeviceRuntimeBackend`; all other calls reach the real
/// backend and therefore the real device session.
struct FaultInjectingBackend {
    inner: DeviceRuntimeBackend,
    kind: FaultKind,
    injected: bool,
}

impl FaultInjectingBackend {
    fn new(inner: DeviceRuntimeBackend, kind: FaultKind) -> Self {
        Self {
            inner,
            kind,
            injected: false,
        }
    }

    fn inner(&self) -> &DeviceRuntimeBackend {
        &self.inner
    }
}

impl DeviceExecutionBackend for FaultInjectingBackend {
    fn reserve(
        &mut self,
        partition: &LogicalPartitionId,
        reservation: &ReservationRecord,
    ) -> Result<(), BackendError> {
        self.inner.reserve(partition, reservation)
    }

    fn run_operation(&mut self, operation: &TransactionOperation) -> Result<(), BackendError> {
        if !self.injected && self.kind.matches(operation) {
            self.injected = true;
            return Err(self.kind.error());
        }
        self.inner.run_operation(operation)
    }

    fn event_completed(&self, event: &OperationEvent) -> bool {
        self.inner.event_completed(event)
    }

    fn stage_write(&mut self, write: &StagedWrite) -> Result<(), BackendError> {
        self.inner.stage_write(write)
    }

    fn publish(&mut self) -> Result<(), BackendError> {
        self.inner.publish()
    }

    fn staged_bytes(&self) -> u64 {
        self.inner.staged_bytes()
    }

    fn published_bytes(&self) -> u64 {
        self.inner.published_bytes()
    }

    fn release(&mut self, partition: &LogicalPartitionId) {
        self.inner.release(partition)
    }

    fn retire(&mut self, partition: &LogicalPartitionId, failure: &TransactionFailure) {
        self.inner.retire(partition, failure)
    }
}

fn live_metal() -> Option<(DeviceRuntimeSet, PhysicalDeviceId)> {
    if !probe_metal_environment().admitted {
        eprintln!("MD3H-X1 metal fault suite: PENDING (Metal not admitted)");
        return None;
    }
    let devices = enumerate_metal_physical_devices().expect("Metal enumeration");
    if devices.is_empty() {
        eprintln!("MD3H-X1 metal fault suite: PENDING (Metal enumeration empty)");
        return None;
    }
    assert_eq!(devices.len(), 1, "H2 live Metal is M=1");
    let id = PhysicalDeviceId::metal(&devices[0].registry_id);
    let set = DeviceRuntimeSet::open_live([id.clone()]).expect("live Metal runtime set");
    Some((set, id))
}

fn live_cuda() -> Option<(DeviceRuntimeSet, PhysicalDeviceId)> {
    if !probe_cuda_environment().admitted {
        eprintln!("MD3H-X1 CUDA fault suite: PENDING (CUDA not admitted)");
        return None;
    }
    let devices = match enumerate_cuda_physical_devices() {
        Ok(devices) if !devices.is_empty() => devices,
        Ok(_) => {
            eprintln!("MD3H-X1 CUDA fault suite: PENDING (CUDA enumeration empty)");
            return None;
        }
        Err(error) => {
            eprintln!("MD3H-X1 CUDA fault suite: PENDING (CUDA enumeration failed: {error})");
            return None;
        }
    };
    assert_eq!(devices.len(), 1, "H2 live CUDA is M=1");
    let id = PhysicalDeviceId::cuda(&devices[0].pci_uuid, devices[0].driver_uuid.clone());
    let set = match DeviceRuntimeSet::open_live([id.clone()]) {
        Ok(set) => set,
        Err(error) => {
            eprintln!("MD3H-X1 CUDA fault suite: PENDING (CUDA open failed: {error})");
            return None;
        }
    };
    Some((set, id))
}

fn run_fault(kind: FaultKind, set: DeviceRuntimeSet, device: PhysicalDeviceId) {
    let plan = plan_on(std::slice::from_ref(&device));
    let backend_kind = set.backend();
    let mut inner = backend_over(set, &plan, program_for(backend_kind));

    // Seed the publication store with the prior committed state. The failed
    // transaction must leave this state untouched while retiring its staging.
    let prior = StagedWrite::new(partition_id(0), OutputRef::new("prior:committed"), 32);
    inner.stage_write(&prior).expect("seed prior staged state");
    inner.publish().expect("seed prior committed state");
    let prior_published = inner.published_write_set().clone();
    let prior_bytes = inner.published_bytes();

    let mut backend = FaultInjectingBackend::new(inner, kind);
    let mut transaction = ExecutionTransaction::new(
        TransactionId::new(format!("txn-md3h-x1-{}", kind.label())),
        plan,
        fixture_operations(),
        fixture_boundary(),
    )
    .expect("constructs");
    transaction.prepare(&mut backend).expect("prepare succeeds");

    let execute_error = transaction
        .execute(&mut backend)
        .expect_err("injected fault fails execute");
    let expected_error = kind.error();
    assert_eq!(execute_error, ExecuteError::Backend(expected_error.clone()));
    assert!(matches!(transaction.state(), TransactionState::Failed(_)));
    assert_eq!(
        transaction.failure(),
        Some(&TransactionFailure::Backend(expected_error.clone()))
    );

    let receipt = transaction
        .abort(&mut backend, format!("{} fault", kind.label()))
        .expect("abort completes teardown");
    assert_eq!(
        receipt.decision,
        TransactionDecision::Aborted {
            failure: TransactionFailure::Backend(expected_error),
        }
    );
    assert!(matches!(transaction.state(), TransactionState::Aborted(_)));
    assert!(receipt.publish_summary.is_none());
    assert!(!receipt.teardown.partial_publication);

    let expected_retired = if kind == FaultKind::Transfer {
        BTreeSet::from([partition_id(0)])
    } else {
        BTreeSet::from([partition_id(0), partition_id(1)])
    };
    let expected_released = if kind == FaultKind::Transfer {
        BTreeSet::from([partition_id(1)])
    } else {
        BTreeSet::new()
    };
    assert_eq!(receipt.teardown.retired_partitions, expected_retired);
    assert_eq!(receipt.teardown.released_partitions, expected_released);

    // No new bytes were promoted, and the prior committed state remains the
    // only authoritative publication after the failed transaction.
    assert_eq!(backend.inner().staged_bytes(), 0);
    assert_eq!(backend.inner().published_bytes(), prior_bytes);
    assert_eq!(backend.inner().published_write_set(), &prior_published);

    let expected_executed = if kind == FaultKind::Transfer { 1 } else { 4 };
    assert_eq!(transaction.executed_operations().len(), expected_executed);
    eprintln!(
        "MD3H-X1 {} fault receipt: failure_partition={} retired={:?} released={:?} published_bytes={}",
        kind.label(),
        expected_error_partition(&receipt),
        receipt.teardown.retired_partitions,
        receipt.teardown.released_partitions,
        backend.inner().published_bytes()
    );
}

fn expected_error_partition(receipt: &TransactionReceipt) -> &LogicalPartitionId {
    match &receipt.decision {
        TransactionDecision::Aborted {
            failure:
                TransactionFailure::Backend(
                    BackendError::Allocation { partition, .. }
                    | BackendError::Operation { partition, .. }
                    | BackendError::DeviceLoss { partition, .. }
                    | BackendError::Cancelled { partition, .. }
                    | BackendError::Timeout { partition, .. },
                ),
        } => partition,
        _ => panic!("fault receipt did not carry a backend failure"),
    }
}

#[test]
fn live_metal_cancel_fault_has_no_partial_publication() {
    let Some((set, device)) = live_metal() else {
        return;
    };
    run_fault(FaultKind::Cancelled, set, device);
}

#[test]
fn live_metal_timeout_fault_has_no_partial_publication() {
    let Some((set, device)) = live_metal() else {
        return;
    };
    run_fault(FaultKind::Timeout, set, device);
}

#[test]
fn live_metal_kernel_fault_has_no_partial_publication() {
    let Some((set, device)) = live_metal() else {
        return;
    };
    run_fault(FaultKind::Kernel, set, device);
}

#[test]
fn live_metal_transfer_fault_has_no_partial_publication() {
    let Some((set, device)) = live_metal() else {
        return;
    };
    run_fault(FaultKind::Transfer, set, device);
}

#[test]
fn live_metal_device_loss_fault_has_no_partial_publication() {
    let Some((set, device)) = live_metal() else {
        return;
    };
    run_fault(FaultKind::DeviceLoss, set, device);
}

#[test]
fn live_cuda_faults_are_pending_when_cuda_is_unreachable() {
    for kind in [
        FaultKind::Cancelled,
        FaultKind::Timeout,
        FaultKind::Kernel,
        FaultKind::Transfer,
        FaultKind::DeviceLoss,
    ] {
        let Some((set, device)) = live_cuda() else {
            return;
        };
        run_fault(kind, set, device);
    }
}
