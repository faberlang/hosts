//! Companion tests for [`super::device_execute`] (repo companion-test convention).

use super::*;
use crate::composite_host::{
    CompletionBoundary, KvCacheLifecycleReceipt, KvCacheMeasurement, KvCachePhaseTiming,
    KvCacheTimingReceipt, KvCacheTimingSpan,
};
use std::collections::BTreeMap;

fn measured_span(duration_us: u64) -> KvCacheTimingSpan {
    KvCacheTimingSpan {
        start_us: KvCacheMeasurement::measured(100),
        end_us: KvCacheMeasurement::measured(100 + duration_us),
        duration_us: KvCacheMeasurement::measured(duration_us),
    }
}

fn sample_receipt() -> DeviceExecutionReceipt {
    DeviceExecutionReceipt {
        backend: DeviceBackend::Metal,
        device_name: "test-device".to_owned(),
        module_hash: 1,
        launches: 1,
        launch_ids: vec![1],
        launch_entries: vec!["kernel".to_owned()],
        fused_library_dispatches: Vec::new(),
        copy_ins: 0,
        outputs: BTreeMap::new(),
        allocated_buffers: Vec::new(),
        allocated_buffer_versions: Vec::new(),
        pool_allocations: 0,
        pool_reuses: 0,
        pool_returns: 0,
        program_lifetime: DeviceProgramLifetime::SingleRun,
        per_program_buffers: Vec::new(),
        per_program_buffer_versions: Vec::new(),
        per_step_buffers: Vec::new(),
        per_step_buffer_versions: Vec::new(),
        observation_buffers: Vec::new(),
        observation_buffer_versions: Vec::new(),
        resource_graph: Vec::new(),
        data_flow_edges: Vec::new(),
        syncs: 1,
        transfers: 0,
        readbacks: 0,
        releases: 0,
        completion_boundary: CompletionBoundary::StepSync { after_launch: 1 },
        program_graph_hash: "graph".to_owned(),
        copy_in_us: 0,
        gpu_encode_submit_wait_us: 0,
        readback_us: 0,
        launch_gpu_us: Vec::new(),
        launch_gpu_start_us: Vec::new(),
    }
}

#[test]
fn gguf_weight_ingress_carries_q8_block_geometry() {
    let mut gguf = b"GGUF".to_vec();
    gguf.extend_from_slice(&3u32.to_le_bytes());
    gguf.extend_from_slice(&1u64.to_le_bytes());
    gguf.extend_from_slice(&0u64.to_le_bytes());
    gguf.extend_from_slice(&1u64.to_le_bytes());
    gguf.extend_from_slice(b"q");
    gguf.extend_from_slice(&1u32.to_le_bytes());
    gguf.extend_from_slice(&32u64.to_le_bytes());
    gguf.extend_from_slice(&8u32.to_le_bytes());
    gguf.extend_from_slice(&0u64.to_le_bytes());
    gguf.resize(64, 0);
    gguf.extend(std::iter::repeat_n(0u8, 34));
    let map = BTreeMap::from([(
        7,
        WeightFileRange {
            offset: 64,
            len: 34,
            elems: 9,
        },
    )]);
    let inputs = inputs_from_gguf(&gguf, &map).expect("admit GGUF bytes");
    assert_eq!(
        inputs.byte_map()[&7].packed_format,
        Some(PackedStorageFormat::Q8_0)
    );
}

fn gguf_header(n_kv: u64, n_tensors: u64) -> Vec<u8> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&n_tensors.to_le_bytes());
    bytes.extend_from_slice(&n_kv.to_le_bytes());
    bytes
}

fn push_scalar_kv(bytes: &mut Vec<u8>) {
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
}

fn push_tensor_info(bytes: &mut Vec<u8>) {
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&32u64.to_le_bytes());
    bytes.extend_from_slice(&8u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
}

/// One metadata entry whose value nests `levels` arrays; the deepest
/// array sits at nesting depth `levels - 1`.
fn push_nested_array_kv(bytes: &mut Vec<u8>, levels: usize) {
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&9u32.to_le_bytes());
    for level in 0..levels {
        let leaf = level + 1 == levels;
        let elem: u32 = if leaf { 4 } else { 9 };
        let count: u64 = if leaf { 0 } else { 1 };
        bytes.extend_from_slice(&elem.to_le_bytes());
        bytes.extend_from_slice(&count.to_le_bytes());
    }
}

fn align32(len: usize) -> usize {
    len.div_ceil(32) * 32
}

#[test]
fn gguf_metadata_array_nesting_is_depth_bounded() {
    let mut bounded = gguf_header(1, 0);
    push_nested_array_kv(&mut bounded, 65);
    bounded.resize(align32(bounded.len()), 0);
    gguf_region_table(&bounded, &BTreeMap::new()).expect("depth 64 metadata parses");

    let mut too_deep = gguf_header(1, 0);
    push_nested_array_kv(&mut too_deep, 66);
    too_deep.resize(align32(too_deep.len()), 0);
    assert!(
        gguf_region_table(&too_deep, &BTreeMap::new()).is_err(),
        "metadata array nesting past depth 64 must be rejected, not recursed"
    );
}

#[test]
fn gguf_walker_rejects_counts_above_the_bounded_ceiling() {
    // Every declared entry is present on the wire, so an unbounded walker
    // would parse the header and admit it; rejection proves the ceiling
    // fires before the count loops iterate or the fact vector allocates.
    let mut kv_over = gguf_header(4097, 0);
    for _ in 0..4097 {
        push_scalar_kv(&mut kv_over);
    }
    kv_over.resize(align32(kv_over.len()), 0);
    assert!(
        gguf_region_table(&kv_over, &BTreeMap::new()).is_err(),
        "declared metadata count over 4096 must be rejected"
    );

    let mut tensors_over = gguf_header(0, 4097);
    for _ in 0..4097 {
        push_tensor_info(&mut tensors_over);
    }
    let data_start = align32(tensors_over.len());
    tensors_over.resize(data_start + 34, 0);
    assert!(
        gguf_region_table(&tensors_over, &BTreeMap::new()).is_err(),
        "declared tensor count over 4096 must be rejected"
    );
    let map = BTreeMap::from([(
        7u32,
        WeightFileRange {
            offset: data_start as u64,
            len: 34,
            elems: 9,
        },
    )]);
    assert!(
        inputs_from_gguf(&tensors_over, &map).is_err(),
        "declared tensor count over 4096 must be rejected before tensor facts are allocated"
    );
}

#[test]
fn gguf_walker_admits_a_bounded_header() {
    let mut bytes = gguf_header(2, 1);
    push_scalar_kv(&mut bytes);
    push_scalar_kv(&mut bytes);
    push_tensor_info(&mut bytes);
    let data_start = align32(bytes.len());
    bytes.resize(data_start + 34, 0);
    let table = gguf_region_table(&bytes, &BTreeMap::new()).expect("bounded header parses");
    assert_eq!(table.data_start, data_start as u64);
    let map = BTreeMap::from([(
        7u32,
        WeightFileRange {
            offset: data_start as u64,
            len: 34,
            elems: 9,
        },
    )]);
    inputs_from_gguf(&bytes, &map).expect("bounded tensor facts admitted");
}

#[test]
fn v2_receipt_carries_measured_phase_and_host_product_timing() {
    let timing = KvCacheTimingReceipt {
        setup_phase: KvCachePhaseTiming::not_measured(),
        steady_state: KvCachePhaseTiming {
            gpu_body: measured_span(7),
            encode: measured_span(11),
            submit: measured_span(13),
            wait: KvCacheTimingSpan::not_measured(),
        },
        slack_us: KvCacheMeasurement::derived(2),
        lifecycle: KvCacheLifecycleReceipt::zero(),
    };
    let wire = project_v2_invocation_receipt(&sample_receipt(), timing, 1, 20, 5);
    let encoded: serde_json::Value =
        serde_json::from_slice(&receipt_to_json(&wire).expect("encode receipt"))
            .expect("parse receipt");

    assert_eq!(wire.encode_us, 11);
    assert_eq!(wire.submit_us, 13);
    assert_eq!(wire.host_product_work_us, 5);
    assert_eq!(wire.wait_us, 0);
    assert_eq!(encoded["encode_us"], 11);
    assert_eq!(encoded["submit_us"], 13);
    assert_eq!(encoded["host_product_work_us"], 5);
}
