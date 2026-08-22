//! MD3H-P2: live pharos CUDA mechanics receipt.
//!
//! This proof consumes the committed MD3H-F1 eight-rank postcard through the
//! host distributed seam. It admits eight virtual partitions on the one
//! locally discovered CUDA device, then checks the separate rejection path for
//! a bind that claims one physical device per rank. It does not attempt the
//! eight-physical-device hardware row.

use faber_host_macos_arm64::device_execute::prepare_distributed_image;
use faber_host_macos_arm64::{discover_cuda_snapshot, probe_cuda_environment};

const EIGHT_RANK: &[u8] = include_bytes!("fixtures/md3h/eight-rank.postcard");
const PROBE_TIME: u64 = 1_752_717_600_000_000_000;

/// The P2 oracle on pharos: one physical CUDA device, eight virtual
/// partitions, and an explicitly software-only receipt.
#[test]
#[ignore = "requires the admitted pharos CUDA machine"]
fn pharos_cuda_eight_rank_mechanics_receipt_is_honest() {
    let environment = probe_cuda_environment();
    assert!(
        environment.admitted,
        "P2 requires an admitted CUDA environment: {}",
        environment.reason
    );

    let snapshot = discover_cuda_snapshot(PROBE_TIME).expect("pharos CUDA discovery");
    assert_eq!(
        snapshot.devices().len(),
        1,
        "P2 is the one-physical-device 8:1 mechanics row"
    );

    let receipt = prepare_distributed_image(EIGHT_RANK, &snapshot, 1)
        .expect("F1 eight-rank image prepares through the host distributed seam");
    assert_eq!(receipt.physical_device_count, 1);
    assert_eq!(receipt.virtual_partition_count, 8);
    assert_eq!(receipt.fixture_identity_class, "virtual");
    assert_eq!(receipt.hardware_isolation_claimed, false);
    assert_eq!(receipt.bind_shape, "8:1");
    assert_eq!(receipt.transport_class, "host_staged");
    assert!(receipt.communication_graph_edge_count > 0);
    assert_eq!(receipt.snapshot_id, snapshot.id().hex());
    assert!(receipt.logical_distributed_plan_hash.starts_with("sha256:"));
    assert!(receipt.bound_distributed_plan_hash.starts_with("sha256:"));
    assert_eq!(receipt.physical_device_ids.len(), 1);
    assert_eq!(receipt.virtual_partition_ids.len(), 8);
    assert_eq!(receipt.transaction_state, "prepared");

    let promoted = prepare_distributed_image(EIGHT_RANK, &snapshot, 8)
        .expect_err("a promoted eight-physical bind must reject on one physical device");
    assert!(
        promoted.message.contains("TopologyMismatch"),
        "promoted bind must remain TopologyMismatch-class: {}",
        promoted.message
    );

    eprintln!(
        "MD3H-P2 pharos CUDA receipt: physical_device_count={} physical_device_ids={:?} virtual_partition_count={} virtual_partition_ids={:?} fixture_identity_class={} transport_class={} hardware_isolation_claimed={} bind_shape={} communication_graph_edge_count={} snapshot_id={} logical_distributed_plan_hash={} bound_distributed_plan_hash={} transaction_state={}",
        receipt.physical_device_count,
        receipt.physical_device_ids,
        receipt.virtual_partition_count,
        receipt.virtual_partition_ids,
        receipt.fixture_identity_class,
        receipt.transport_class,
        receipt.hardware_isolation_claimed,
        receipt.bind_shape,
        receipt.communication_graph_edge_count,
        receipt.snapshot_id,
        receipt.logical_distributed_plan_hash,
        receipt.bound_distributed_plan_hash,
        receipt.transaction_state,
    );
    eprintln!(
        "MD3H-P2 promoted-as-eight-physical rejection: class=TopologyMismatch message={}",
        promoted.message
    );
}
