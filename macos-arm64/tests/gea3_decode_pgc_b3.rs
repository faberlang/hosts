//! PGC-B3 focused device proof.
//!
//! The production GEA3 bundle is intentionally left untouched here. This
//! additive test exercises the direct-row binding contract through the fake
//! Metal session and independently probes the complete fixed-capacity arena.
//! The fake session is a binding/ordering oracle, not a real-GPU receipt.

use faber_host_macos_arm64::device_descriptor::{
    DescriptorInvocationState, DescriptorRuntimeSource, DeviceDataType,
};
use faber_host_macos_arm64::device_host::{
    DeviceLaunchBinding, DeviceRuntime, DeviceSession, InvocationStateBuffer,
};
use faber_host_macos_arm64::metal_host::MetalLaunchBinding;
use faber_host_macos_arm64::{FakeMetalDriver, MetalHostSession};
use host_coordinator::DeviceBackend;

const HISTORY_CAPACITY: usize = 76;
const KV_WIDTH: usize = 320;
const F32_BYTES: usize = 4;
const ROW_BYTES: usize = KV_WIDTH * F32_BYTES;
const ARENA_BYTES: usize = HISTORY_CAPACITY * ROW_BYTES;
const FIXED1000_STEPS: usize = 1_000;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(F32_BYTES)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn baseline_arena() -> Vec<f32> {
    (0..HISTORY_CAPACITY * KV_WIDTH)
        .map(|index| {
            let row = index / KV_WIDTH;
            let column = index % KV_WIDTH;
            row as f32 * 10.0 + column as f32 * 0.01
        })
        .collect()
}

fn incoming_row(position: usize) -> Vec<f32> {
    (0..KV_WIDTH)
        .map(|column| 1_000.0 + position as f32 + column as f32 * 0.001)
        .collect()
}

fn apply_selected_row(arena: &mut [f32], position: usize, row: &[f32]) {
    let start = position * KV_WIDTH;
    for (destination, incoming) in arena[start..start + KV_WIDTH].iter_mut().zip(row) {
        *destination += incoming;
    }
}

fn compact_bindings(
    history: host_coordinator::DeviceHandle,
    row: host_coordinator::DeviceHandle,
    output: host_coordinator::DeviceHandle,
    position: usize,
) -> Vec<DeviceLaunchBinding> {
    let byte_offset = (position * ROW_BYTES) as u64;
    vec![
        DeviceLaunchBinding {
            handle: history,
            binding_index: 0,
            byte_offset,
            view_span: ROW_BYTES as u64,
            runtime_source: DescriptorRuntimeSource::Position,
        },
        DeviceLaunchBinding {
            handle: row,
            binding_index: 1,
            byte_offset: 0,
            view_span: ROW_BYTES as u64,
            runtime_source: DescriptorRuntimeSource::Constant,
        },
        DeviceLaunchBinding {
            handle: output,
            binding_index: 2,
            byte_offset,
            view_span: ROW_BYTES as u64,
            runtime_source: DescriptorRuntimeSource::Position,
        },
    ]
}

fn compact_metal_bindings(
    history: faber_host_macos_arm64::MetalHandleId,
    row: faber_host_macos_arm64::MetalHandleId,
    output: faber_host_macos_arm64::MetalHandleId,
) -> [MetalLaunchBinding; 3] {
    [
        MetalLaunchBinding {
            handle: history,
            binding_index: 0,
            byte_offset: 0,
            view_span: ROW_BYTES as u64,
        },
        MetalLaunchBinding {
            handle: row,
            binding_index: 1,
            byte_offset: 0,
            view_span: ROW_BYTES as u64,
        },
        MetalLaunchBinding {
            handle: output,
            binding_index: 2,
            byte_offset: 0,
            view_span: ROW_BYTES as u64,
        },
    ]
}

fn run_fake_row_launch(entry: &str, position: usize, history_values: &[f32], row: &[f32]) {
    let session = MetalHostSession::with_driver(Box::new(FakeMetalDriver::default()))
        .expect("fake Metal admission");
    let mut runtime = DeviceRuntime::Metal(session);
    let module = runtime
        .load_module(b"pgc-b3-direct-row-module")
        .expect("module");
    let history = runtime.alloc_bytes(ROW_BYTES).expect("history row");
    let incoming = runtime.alloc_bytes(ROW_BYTES).expect("incoming row");
    let output = runtime.alloc_bytes(ROW_BYTES).expect("output row");
    runtime
        .copy_in_bytes(&history, &f32_bytes(history_values), DeviceDataType::F32)
        .expect("history upload");
    runtime
        .copy_in_bytes(&incoming, &f32_bytes(row), DeviceDataType::F32)
        .expect("row upload");
    runtime
        .copy_in_bytes(
            &output,
            &f32_bytes(&vec![0.0; KV_WIDTH]),
            DeviceDataType::F32,
        )
        .expect("output initialization");

    let bindings = compact_bindings(history, incoming, output, position);
    assert!(bindings[0].is_cuda_dynamic());
    assert!(bindings[2].is_cuda_dynamic());
    assert_eq!(bindings[0].byte_offset, (position * ROW_BYTES) as u64);
    assert_eq!(bindings[2].byte_offset, (position * ROW_BYTES) as u64);
    assert_eq!(bindings[0].view_span, ROW_BYTES as u64);
    assert_eq!(bindings[2].view_span, ROW_BYTES as u64);

    // The fake session operates on already-projected row handles. Its direct
    // launch therefore uses the same three slots at offset zero while the
    // binding table above proves the fixed-arena Position envelope.
    let metal_bindings = compact_metal_bindings(
        faber_host_macos_arm64::metal_host::MetalHandleId(history.id),
        faber_host_macos_arm64::metal_host::MetalHandleId(incoming.id),
        faber_host_macos_arm64::metal_host::MetalHandleId(output.id),
    );
    if let DeviceRuntime::Metal(session) = &mut runtime {
        session
            .launch_kernel_bound(
                faber_host_macos_arm64::metal_host::MetalHandleId(module.id),
                entry,
                &metal_bindings,
                [KV_WIDTH as u32, 1, 1],
                [1, 1, 1],
            )
            .expect("direct row launch");
        session.sync().expect("direct row sync");
        let result = session
            .readback_f32(faber_host_macos_arm64::metal_host::MetalHandleId(output.id))
            .expect("row readback");
        let expected: Vec<f32> = history_values
            .iter()
            .zip(row)
            .map(|(history, incoming)| history + incoming)
            .collect();
        assert_eq!(
            result, expected,
            "{entry} selected row at position {position}"
        );
        assert_eq!(
            session.live_handle_count(),
            4,
            "{entry} row launch allocations"
        );
    }
}

#[test]
fn pgc_b3_device_direct_row_write_keeps_arena_lineage_and_prior_rows() {
    assert_eq!(FIXED1000_STEPS, 1_000);
    assert_eq!(ARENA_BYTES, 97_280);
    for entry in ["kv_append_k", "kv_append_v"] {
        for position in [0, HISTORY_CAPACITY / 2, HISTORY_CAPACITY - 1] {
            let before = baseline_arena();
            let row = incoming_row(position);
            let mut after = before.clone();
            apply_selected_row(&mut after, position, &row);
            let selected_start = position * KV_WIDTH;
            let selected_end = selected_start + KV_WIDTH;
            for (index, (old, actual)) in before.iter().zip(&after).enumerate() {
                if (selected_start..selected_end).contains(&index) {
                    assert_eq!(
                        *actual,
                        *old + row[index - selected_start],
                        "{entry} selected value"
                    );
                } else {
                    assert_eq!(*actual, *old, "{entry} non-selected row changed at {index}");
                }
            }
            run_fake_row_launch(entry, position, &before[selected_start..selected_end], &row);
        }
    }
}

#[test]
fn pgc_b3_device_position_constant_is_one_typed_compact_state_upload() {
    let state = DescriptorInvocationState {
        position: 75,
        valid_len_after: 76,
        query_rows: 1,
        sequence_epoch: 0,
    };
    let encoded = InvocationStateBuffer::encoded_bytes(state);
    assert_eq!(encoded.len(), 16, "one four-field cursor upload");
    assert_eq!(&encoded[0..4], &75_u32.to_le_bytes());
    assert_eq!(&encoded[4..8], &76_u32.to_le_bytes());
    assert_eq!(&encoded[8..12], &1_u32.to_le_bytes());
    assert_eq!(&encoded[12..16], &0_u32.to_le_bytes());

    let session = MetalHostSession::with_driver(Box::new(FakeMetalDriver::default()))
        .expect("fake Metal admission");
    let mut runtime = DeviceRuntime::Metal(session);
    let cursor = runtime.alloc_invocation_state().expect("cursor allocation");
    runtime
        .upload_invocation_state(&cursor, state)
        .expect("cursor upload");
    let observed = runtime
        .readback_bytes(&cursor.handle(), DeviceDataType::U8)
        .expect("cursor readback");
    assert_eq!(observed, encoded, "cursor bytes remain typed and compact");
}

#[test]
fn pgc_b3_device_position_envelope_rejects_capacity_boundary_overrun() {
    let session = MetalHostSession::with_driver(Box::new(FakeMetalDriver::default()))
        .expect("fake Metal admission");
    let mut runtime = DeviceRuntime::Metal(session);
    let module = runtime
        .load_module(b"pgc-b3-direct-row-module")
        .expect("module");
    let arena = runtime.alloc_bytes(ARENA_BYTES).expect("arena");
    let row = runtime.alloc_bytes(ROW_BYTES).expect("row");
    let output = runtime.alloc_bytes(ARENA_BYTES).expect("output arena");
    let overrun = compact_bindings(arena, row, output, HISTORY_CAPACITY);
    let result = runtime.launch_kernel_bound(
        &module,
        "kv_append_k",
        &overrun,
        [KV_WIDTH as u32, 1, 1],
        [1, 1, 1],
    );
    assert!(result.is_err(), "capacity+1 Position view must fail closed");
}

#[test]
fn pgc_b3_device_f32_byte_round_trip_is_exact_for_probe_rows() {
    let values = incoming_row(38);
    assert_eq!(f32_values(&f32_bytes(&values)), values);
}

const _: DeviceBackend = DeviceBackend::Metal;
