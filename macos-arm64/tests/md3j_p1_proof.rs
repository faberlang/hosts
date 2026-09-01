//! MD3J-P1: on-pod two-physical same-SKU execution receipt.
//!
//! This is intentionally ignored on ordinary hosts. The accepted run is on a
//! two-GPU RunPod pod: the F1 image is bound 8:2, a real CUDA runtime set is
//! composed from the two discovered physical ids, and the translated
//! transaction crosses the commit boundary with host-staged transfers. The
//! same image is then prepared as 8:1 for hash comparability, followed by the
//! fail-closed over-budget row.

use std::collections::BTreeMap;
use std::time::Duration;

use faber_host_macos_arm64::device_execute::prepare_distributed_image;
use faber_host_macos_arm64::device_host::DeviceRuntime;
use faber_host_macos_arm64::distributed_translate::{
    bind_policy_for_declared_count, bind_translated, oq2_default_headroom_policy_bytes,
    translate_device_section_bytes,
};
use faber_host_macos_arm64::{
    CudaHostSession, DeviceRuntimeBackend, DeviceRuntimeSet, LaunchProgram, discover_cuda_snapshot,
    enumerate_cuda_physical_devices, probe_cuda_environment,
};
use host_coordinator::bound_plan::{BoundDistributedPlan, LogicalPartitionId};
use host_coordinator::device_identity::PhysicalDeviceId;
use host_coordinator::execution_transaction::{
    DeviceExecutionBackend, ExecutionTransaction, PublicationOrdinal, TransactionId,
    TransactionState,
};
use host_coordinator::transport::{CopyPath, TransferBudget};

const EIGHT_RANK: &[u8] = include_bytes!("fixtures/md3j/eight-rank.postcard");
const OVER_BUDGET_EIGHT_RANK: &[u8] =
    include_bytes!("fixtures/md3j/eight-rank-over-budget.postcard");
const REALCARD_OVER_BUDGET_EIGHT_RANK: &[u8] =
    include_bytes!("fixtures/md3j/eight-rank-over-budget-realcard.postcard");
const OVER_BUDGET_DECLARED_TOTAL_BYTES: u64 = 40 * 1024 * 1024 * 1024;
const REALCARD_OVER_BUDGET_DECLARED_TOTAL_BYTES: u64 = 79 * 1024 * 1024 * 1024;
const PROBE_TIME: u64 = 1_752_717_600_000_000_000;

// The transaction backend only needs a tiny elementwise kernel for the one
// non-empty output launch in the postcard. The graph and all transfer bytes
// still come from the frozen F1 image.
const COPY_PTX: &str = r#"
.version 4.1
.target sm_52
.address_size 64

.visible .entry md3j_copy(
    .param .u64 md3j_copy_src,
    .param .u64 md3j_copy_dst
)
{
    .reg .b32 %r<5>;
    .reg .b64 %rd<6>;

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %ctaid.x;
    mad.lo.s32 %r4, %r2, %r3, %r1;
    ld.param.u64 %rd1, [md3j_copy_src];
    ld.param.u64 %rd2, [md3j_copy_dst];
    mul.wide.u32 %rd3, %r4, 4;
    add.s64 %rd4, %rd1, %rd3;
    add.s64 %rd5, %rd2, %rd3;
    ld.global.u32 %r1, [%rd4];
    st.global.u32 [%rd5], %r1;
    ret;
}
"#;

fn physical_ids(
    snapshot: &host_coordinator::discovery::DeviceDiscoverySnapshot,
) -> Vec<PhysicalDeviceId> {
    snapshot
        .devices()
        .values()
        .map(|entry| entry.identity.clone())
        .collect()
}

fn partition_devices(
    plan: &BoundDistributedPlan,
) -> BTreeMap<LogicalPartitionId, PhysicalDeviceId> {
    plan.bindings()
        .expect("8-rank bind is distributed")
        .iter()
        .map(|(partition, binding)| (partition.clone(), binding.device().clone()))
        .collect()
}

fn live_cuda_runtime_set(ids: &[PhysicalDeviceId]) -> DeviceRuntimeSet {
    assert_eq!(ids.len(), 2, "P1 requires exactly two physical CUDA ids");
    // Each session is opened through the product's live CUDA admission path;
    // DeviceRuntimeSet owns the one-session-per-physical composition and the
    // transaction backend routes each bound partition through its member.
    let first = CudaHostSession::try_open().expect("open first live CUDA session");
    let second = CudaHostSession::try_open().expect("open second live CUDA session");
    DeviceRuntimeSet::from_members([
        (ids[0].clone(), DeviceRuntime::Cuda(first)),
        (ids[1].clone(), DeviceRuntime::Cuda(second)),
    ])
    .expect("compose M=2 CUDA DeviceRuntimeSet")
}

#[test]
#[ignore = "requires the operator-authorized two-GPU RunPod pod"]
fn runpod_cuda_eight_on_two_receipt_is_honest() {
    let environment = probe_cuda_environment();
    assert!(
        environment.admitted,
        "P1 requires an admitted CUDA environment: {}",
        environment.reason
    );

    let devices = enumerate_cuda_physical_devices().expect("enumerate pod CUDA devices");
    assert_eq!(
        devices.len(),
        2,
        "P1 is the exactly-two-physical-device 8:2 row"
    );
    assert_eq!(
        devices[0].device_model, devices[1].device_model,
        "P1 requires two same-SKU devices"
    );
    let snapshot = discover_cuda_snapshot(PROBE_TIME).expect("pod CUDA discovery");
    let ids = physical_ids(&snapshot);
    assert_eq!(ids.len(), 2);
    assert_ne!(
        ids[0], ids[1],
        "same-SKU devices must have distinct identities"
    );

    let receipt_82 = prepare_distributed_image(EIGHT_RANK, &snapshot, 2)
        .expect("F1 eight-rank image prepares as 8:2");
    assert_eq!(receipt_82.physical_device_count, 2);
    assert_eq!(receipt_82.physical_device_ids.len(), 2);
    assert_ne!(
        receipt_82.physical_device_ids[0], receipt_82.physical_device_ids[1],
        "8:2 receipt must carry two distinct physical ids"
    );
    assert_eq!(receipt_82.virtual_partition_count, 8);
    assert_eq!(receipt_82.virtual_partition_ids.len(), 8);
    assert_eq!(receipt_82.fixture_identity_class, "virtual");
    assert_eq!(receipt_82.transport_class, "host_staged");
    assert!(!receipt_82.hardware_isolation_claimed);
    assert_eq!(receipt_82.bind_shape, "8:2");
    assert!(receipt_82.communication_graph_edge_count > 0);
    assert_eq!(receipt_82.snapshot_id, snapshot.id().hex());
    assert!(
        receipt_82
            .logical_distributed_plan_hash
            .starts_with("sha256:")
    );
    assert!(
        receipt_82
            .bound_distributed_plan_hash
            .starts_with("sha256:")
    );
    assert_eq!(receipt_82.transaction_state, "prepared");

    let translated = translate_device_section_bytes(EIGHT_RANK)
        .expect("F1 image translates into the transaction mirror");
    let policy =
        bind_policy_for_declared_count(translated.partitions().len(), 2).expect("8:2 bind policy");
    let bound = bind_translated(&translated, &snapshot, policy).expect("8:2 bind");
    let mapping = partition_devices(&bound);
    assert_eq!(mapping.len(), 8);
    assert_eq!(mapping.values().filter(|id| *id == &ids[0]).count(), 4);
    assert_eq!(mapping.values().filter(|id| *id == &ids[1]).count(), 4);

    let runtime_set = live_cuda_runtime_set(&ids);
    assert_eq!(runtime_set.len(), 2, "DeviceRuntimeSet must be M=2");
    let mut backend = DeviceRuntimeBackend::new(
        runtime_set,
        mapping,
        LaunchProgram::new(COPY_PTX.as_bytes().to_vec(), "md3j_copy"),
        TransferBudget::declared(1 << 20, 10_000_000_000),
        Duration::from_secs(60),
    )
    .expect("construct M=2 real CUDA execution backend");
    let mut transaction = ExecutionTransaction::new(
        TransactionId::new("md3j-p1-runpod-8-on-2"),
        bound,
        translated.operations().to_vec(),
        translated.commit_boundary().clone(),
    )
    .expect("construct execution transaction");
    transaction
        .prepare(&mut backend)
        .expect("M=2 transaction prepare");
    transaction
        .execute(&mut backend)
        .expect("M=2 transaction execute");
    transaction
        .commit(&mut backend, PublicationOrdinal::new(1))
        .expect("M=2 transaction commit");
    assert_eq!(transaction.state(), &TransactionState::Committed);
    assert!(
        backend.staged_bytes() > 0,
        "transaction must stage output bytes"
    );
    assert_eq!(
        backend.published_bytes(),
        backend.staged_bytes(),
        "commit must publish all staged bytes"
    );
    let transport = backend.transport_receipt();
    assert!(
        !transport.records.is_empty(),
        "cross-physical transfers must be recorded"
    );
    assert!(transport.records.iter().all(|record| {
        record.copy_path == CopyPath::HostStaged && (record.elapsed_nanos > 0 || record.bytes > 0)
    }));

    let one_snapshot = host_coordinator::discovery::DeviceDiscoverySnapshot::from_enumerated(
        PROBE_TIME,
        snapshot.devices().values().next().cloned().into_iter(),
    );
    let receipt_81 = prepare_distributed_image(EIGHT_RANK, &one_snapshot, 1)
        .expect("same F1 image prepares as the 8:1 comparison row");
    assert_eq!(
        receipt_81.logical_distributed_plan_hash, receipt_82.logical_distributed_plan_hash,
        "logical hash must not vary with bind count"
    );
    assert_ne!(
        receipt_81.bound_distributed_plan_hash, receipt_82.bound_distributed_plan_hash,
        "bound hash must capture the physical bind divergence"
    );
    assert_eq!(receipt_81.bind_shape, "8:1");

    // The original F1 over-budget variant is intentionally only over the
    // small synthetic B2 snapshot. Its 40 GiB ledger declaration admits on a
    // real 80 GiB A100, so pin the correct big-card behavior here.
    let real_card_admits_40gib = prepare_distributed_image(OVER_BUDGET_EIGHT_RANK, &snapshot, 2)
        .expect("40 GiB ledger declaration admits on the real A100 policy shape");
    assert_eq!(real_card_admits_40gib.bind_shape, "8:2");
    assert_eq!(
        OVER_BUDGET_DECLARED_TOTAL_BYTES,
        40 * 1024 * 1024 * 1024,
        "the retained synthetic over-budget fixture must remain the 40 GiB case"
    );

    // The real-card variant uses a 79 GiB ledger declaration. For the
    // observed 80 GiB A100 shape, floor(80 GiB × 0.9) is 77,309,411,328
    // bytes, so this variant must reject before any runtime is opened.
    let realcard_over_budget =
        prepare_distributed_image(REALCARD_OVER_BUDGET_EIGHT_RANK, &snapshot, 2)
            .expect_err("real-card over-budget variant must reject before execution");
    assert_eq!(realcard_over_budget.code, "E_INVALID_ARGS");
    let policy_limit_bytes = snapshot
        .devices()
        .values()
        .next()
        .expect("real A100 snapshot has a first device")
        .memory
        .api_total_bytes;
    let policy_limit_bytes = oq2_default_headroom_policy_bytes(policy_limit_bytes);
    assert!(
        policy_limit_bytes < REALCARD_OVER_BUDGET_DECLARED_TOTAL_BYTES,
        "real-card fixture declaration must exceed the observed OQ-2 policy limit"
    );
    assert!(
        realcard_over_budget.message.contains("BudgetExceeded")
            && realcard_over_budget.message.contains(&format!(
                "declared_total_bytes: Some({REALCARD_OVER_BUDGET_DECLARED_TOTAL_BYTES})"
            ))
            && realcard_over_budget
                .message
                .contains(&format!("policy_limit_bytes: {policy_limit_bytes}")),
        "real-card BudgetExceeded row must retain both byte facts: {}",
        realcard_over_budget.message
    );

    eprintln!(
        "MD3J-P1 8:2 frozen receipt: physical_device_count={} physical_device_ids={:?} virtual_partition_count={} virtual_partition_ids={:?} fixture_identity_class={} transport_class={} hardware_isolation_claimed={} bind_shape={} communication_graph_edge_count={} snapshot_id={} logical_distributed_plan_hash={} bound_distributed_plan_hash={} transaction_state={} staged_bytes={} published_bytes={} host_staged_records={}",
        receipt_82.physical_device_count,
        receipt_82.physical_device_ids,
        receipt_82.virtual_partition_count,
        receipt_82.virtual_partition_ids,
        receipt_82.fixture_identity_class,
        receipt_82.transport_class,
        receipt_82.hardware_isolation_claimed,
        receipt_82.bind_shape,
        receipt_82.communication_graph_edge_count,
        receipt_82.snapshot_id,
        receipt_82.logical_distributed_plan_hash,
        receipt_82.bound_distributed_plan_hash,
        receipt_82.transaction_state,
        backend.staged_bytes(),
        backend.published_bytes(),
        transport.records.len(),
    );
    for (rank, (partition, physical)) in
        snapshot_mapping_rows(&receipt_82, &ids, &translated, &snapshot)
    {
        eprintln!(
            "MD3J-P1 rank->partition->physical: rank={} partition={} physical={}",
            rank, partition, physical
        );
    }
    eprintln!(
        "MD3J-P1 8:1 comparability: bind_shape=8:1 logical_distributed_plan_hash={} bound_distributed_plan_hash={} logical_hash_identical=true bound_hash_different=true",
        receipt_81.logical_distributed_plan_hash, receipt_81.bound_distributed_plan_hash
    );
    eprintln!(
        "MD3J-P1 40GiB-on-80GB admission row: class=admitted bind_shape={} transaction_state={}",
        real_card_admits_40gib.bind_shape, real_card_admits_40gib.transaction_state
    );
    eprintln!(
        "MD3J-P1 BudgetExceeded row: class=BudgetExceeded declared_total_bytes={} policy_limit_bytes={} message={}",
        REALCARD_OVER_BUDGET_DECLARED_TOTAL_BYTES, policy_limit_bytes, realcard_over_budget.message
    );
    eprintln!(
        "MD3J-P1 honest exclusions: No physical-capacity or speedup claim. AllocationFailure under real physical pressure is NOT tested. 8:8 stays deferred behind the RunPod 8x same-SKU gate; this rung does not close it and emits no 8:8 row beyond NOT ATTEMPTED. P2P/peer admission is NOT ATTEMPTED; cross-physical transfers are host-staged."
    );
}

fn snapshot_mapping_rows(
    _receipt: &faber_host_macos_arm64::device_execute::DistributedPrepareReceipt,
    _ids: &[PhysicalDeviceId],
    translated: &faber_host_macos_arm64::distributed_translate::TranslatedDistributedPlan,
    snapshot: &host_coordinator::discovery::DeviceDiscoverySnapshot,
) -> Vec<(usize, (String, String))> {
    let policy = bind_policy_for_declared_count(translated.partitions().len(), 2)
        .expect("8:2 mapping policy");
    let bound = bind_translated(translated, snapshot, policy).expect("8:2 mapping bind");
    bound
        .bindings()
        .expect("8:2 mapping is distributed")
        .iter()
        .enumerate()
        .map(|(rank, (partition, binding))| {
            (
                rank,
                (partition.as_str().to_owned(), binding.device().to_string()),
            )
        })
        .collect()
}
