//! U-03 host launch adapter tests (fake driver sequencing only — never a real
//! device; real-device execution is the runpod-gated U-05 step).

use std::collections::BTreeMap;

use faber_host_macos_arm64::cuda_host::FakeCudaDriver;
use faber_host_macos_arm64::cuda_launch_adapter::{
    launch_descriptor, parse_descriptor, AdapterBufferRole, NumericOracle, NvvmElementType,
    NVVM_DESCRIPTOR_SCHEMA_VERSION,
};
use faber_host_macos_arm64::device_descriptor::{
    E_DEVICE_ABI_MISMATCH, E_DEVICE_DESCRIPTOR, E_DEVICE_DTYPE_MISMATCH, E_DEVICE_ENTRY_MISMATCH,
    E_DEVICE_SHAPE_MISMATCH,
};
use faber_host_macos_arm64::CudaHostSession;

/// The emitted `rung-0-matmul` v2 descriptor shape
/// (`examples/gpu-workload/rung-0-matmul.fab` → `[2,3] × [3,2] → [2,2]`):
/// schema v2, `tiled_matmul` plan `M=2,K=3,N=2`, block 8×8, grid 1×1.
const RUNG0_MATMUL_DESCRIPTOR: &str = r#"{ "schema_version": 2, "target": "llvm-nvvm",
  "kernels": [ { "entry": "rung0_matmul_kernel", "element_type": "f32",
                 "element_byte_width": 4, "element_count": 6,
                 "element_counts": [6, 6, 4],
                 "input_buffers": 2, "output_buffers": 1,
                 "accumulation_buffers": 0,
                 "buffers": [
                   { "role": "input", "binding": 0, "element_count": 6, "shape": [2, 3] },
                   { "role": "extra-input", "binding": 1, "element_count": 6, "shape": [3, 2] },
                   { "role": "output", "binding": 2, "element_count": 4, "shape": [2, 2] } ],
                 "launch": { "workgroup": { "x": 8, "y": 8, "z": 1 },
                              "dispatch": { "x": 1, "y": 1, "z": 1 } },
                 "plan": { "kind": "tiled_matmul", "m": 2, "k": 3, "n": 2,
                           "tile": 8, "workgroup_x": 8, "workgroup_y": 8 } } ] }"#;

/// The `rung-0-matmul` host inputs and the independent oracle row
/// (`[58.0, 64.0, 139.0, 154.0]`, matching `rung-0-matmul.ref.json`).
const RUNG0_A: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
const RUNG0_B: [f32; 6] = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
const RUNG0_ORACLE: [f32; 4] = [58.0, 64.0, 139.0, 154.0];

fn rung0_inputs() -> BTreeMap<u32, Vec<f32>> {
    let mut inputs = BTreeMap::new();
    inputs.insert(0, RUNG0_A.to_vec());
    inputs.insert(1, RUNG0_B.to_vec());
    inputs
}

fn rung0_oracle() -> NumericOracle {
    NumericOracle::new(RUNG0_ORACLE.to_vec(), 0.00001, 0.00001)
}

fn fake_session() -> CudaHostSession {
    CudaHostSession::with_driver(Box::new(
        FakeCudaDriver::default().with_matmul_simulation(2, 3, 2),
    ))
    .expect("fake admit")
}

#[test]
fn parse_accepts_rung0_matmul_descriptor() {
    let plan = parse_descriptor(RUNG0_MATMUL_DESCRIPTOR.as_bytes())
        .expect("valid rung-0 descriptor parses");
    assert_eq!(plan.entry, "rung0_matmul_kernel");
    assert_eq!(plan.element_ty, NvvmElementType::F32);
    assert_eq!(plan.grid, [1, 1, 1]);
    assert_eq!(plan.block, [8, 8, 1]);
    assert_eq!(plan.plan_kind.as_deref(), Some("tiled_matmul"));
    assert_eq!(plan.buffers.len(), 3);
    assert_eq!(plan.buffers[0].role, AdapterBufferRole::Input);
    assert_eq!(plan.buffers[0].binding, 0);
    assert_eq!(plan.buffers[0].element_count, 6);
    assert_eq!(plan.buffers[0].shape, vec![2, 3]);
    assert_eq!(plan.buffers[1].role, AdapterBufferRole::ExtraInput);
    assert_eq!(plan.buffers[1].element_count, 6);
    assert_eq!(plan.buffers[1].shape, vec![3, 2]);
    assert_eq!(plan.buffers[2].role, AdapterBufferRole::Output);
    assert_eq!(plan.buffers[2].element_count, 4);
    assert_eq!(plan.buffers[2].shape, vec![2, 2]);
}

#[test]
fn parse_rejects_wrong_schema_version() {
    let descriptor = br#"{"schema_version": 1, "target": "llvm-nvvm",
      "kernels": [ { "entry": "addita", "element_type": "f32", "element_byte_width": 4,
                     "element_count": 256, "element_counts": [256, 256, 256],
                     "input_buffers": 2, "output_buffers": 1, "accumulation_buffers": 0,
                     "buffers": [], "launch": { "workgroup": { "x": 1, "y": 1, "z": 1 },
                     "dispatch": { "x": 256, "y": 1, "z": 1 } } } ] }"#;
    let err =
        parse_descriptor(descriptor).expect_err("v1 sidecar must fail closed on schema version");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
    assert!(err.message.contains("schema_version 1"), "{}", err.message);
    assert!(err.message.contains("2"), "{}", err.message);
}

#[test]
fn parse_rejects_unknown_target() {
    let descriptor = br#"{"schema_version": 2, "target": "llvm-cuda",
      "kernels": [ { "entry": "k", "element_type": "f32", "element_byte_width": 4,
                     "element_count": 1, "element_counts": [1, 1], "input_buffers": 1,
                     "output_buffers": 1, "accumulation_buffers": 0,
                     "buffers": [ { "role": "input", "binding": 0, "element_count": 1,
                     "shape": [1] }, { "role": "output", "binding": 1, "element_count": 1,
                     "shape": [1] } ],
                     "launch": { "workgroup": { "x": 1, "y": 1, "z": 1 },
                     "dispatch": { "x": 1, "y": 1, "z": 1 } } } ] }"#;
    let err = parse_descriptor(descriptor).expect_err("unknown target must fail closed");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
    assert!(err.message.contains("llvm-cuda"), "{}", err.message);
}

#[test]
fn parse_rejects_empty_kernel_list() {
    let descriptor = br#"{"schema_version": 2, "target": "llvm-nvvm", "kernels": []}"#;
    let err = parse_descriptor(descriptor).expect_err("empty kernel list must fail closed");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
}

#[test]
fn parse_rejects_non_nvvm_dtype() {
    let descriptor = br#"{"schema_version": 2, "target": "llvm-nvvm",
      "kernels": [ { "entry": "k", "element_type": "f64", "element_byte_width": 8,
                     "element_count": 1, "element_counts": [1, 1], "input_buffers": 1,
                     "output_buffers": 1, "accumulation_buffers": 0,
                     "buffers": [ { "role": "input", "binding": 0, "element_count": 1,
                     "shape": [1] }, { "role": "output", "binding": 1, "element_count": 1,
                     "shape": [1] } ],
                     "launch": { "workgroup": { "x": 1, "y": 1, "z": 1 },
                     "dispatch": { "x": 1, "y": 1, "z": 1 } } } ] }"#;
    let err = parse_descriptor(descriptor).expect_err("f64 outside the NVVM family");
    assert_eq!(err.code, E_DEVICE_DTYPE_MISMATCH);
    assert!(err.message.contains("f64"), "{}", err.message);
}

#[test]
fn parse_rejects_byte_width_dtype_conflict() {
    let descriptor = br#"{"schema_version": 2, "target": "llvm-nvvm",
      "kernels": [ { "entry": "k", "element_type": "f32", "element_byte_width": 8,
                     "element_count": 1, "element_counts": [1, 1], "input_buffers": 1,
                     "output_buffers": 1, "accumulation_buffers": 0,
                     "buffers": [ { "role": "input", "binding": 0, "element_count": 1,
                     "shape": [1] }, { "role": "output", "binding": 1, "element_count": 1,
                     "shape": [1] } ],
                     "launch": { "workgroup": { "x": 1, "y": 1, "z": 1 },
                     "dispatch": { "x": 1, "y": 1, "z": 1 } } } ] }"#;
    let err = parse_descriptor(descriptor).expect_err("byte width must match the dtype");
    assert_eq!(err.code, E_DEVICE_DTYPE_MISMATCH);
}

#[test]
fn parse_rejects_count_shape_contradiction() {
    let descriptor = br#"{"schema_version": 2, "target": "llvm-nvvm",
      "kernels": [ { "entry": "k", "element_type": "f32", "element_byte_width": 4,
                     "element_count": 6, "element_counts": [6, 4], "input_buffers": 1,
                     "output_buffers": 1, "accumulation_buffers": 0,
                     "buffers": [ { "role": "input", "binding": 0, "element_count": 7,
                     "shape": [2, 3] }, { "role": "output", "binding": 1, "element_count": 4,
                     "shape": [2, 2] } ],
                     "launch": { "workgroup": { "x": 2, "y": 2, "z": 1 },
                     "dispatch": { "x": 1, "y": 1, "z": 1 } } } ] }"#;
    let err = parse_descriptor(descriptor).expect_err("element_count must equal the shape product");
    assert_eq!(err.code, E_DEVICE_SHAPE_MISMATCH);
}

#[test]
fn parse_rejects_element_counts_mismatch() {
    let descriptor = br#"{"schema_version": 2, "target": "llvm-nvvm",
      "kernels": [ { "entry": "k", "element_type": "f32", "element_byte_width": 4,
                     "element_count": 6, "element_counts": [6, 6, 4], "input_buffers": 1,
                     "output_buffers": 1, "accumulation_buffers": 0,
                     "buffers": [ { "role": "input", "binding": 0, "element_count": 6,
                     "shape": [2, 3] }, { "role": "output", "binding": 1, "element_count": 4,
                     "shape": [2, 2] } ],
                     "launch": { "workgroup": { "x": 2, "y": 2, "z": 1 },
                     "dispatch": { "x": 1, "y": 1, "z": 1 } } } ] }"#;
    let err = parse_descriptor(descriptor)
        .expect_err("element_counts must cover exactly the storage buffers");
    assert_eq!(err.code, E_DEVICE_SHAPE_MISMATCH);
}

#[test]
fn parse_rejects_role_counts_conflict() {
    let descriptor = br#"{"schema_version": 2, "target": "llvm-nvvm",
      "kernels": [ { "entry": "k", "element_type": "f32", "element_byte_width": 4,
                     "element_count": 6, "element_counts": [6, 4], "input_buffers": 3,
                     "output_buffers": 1, "accumulation_buffers": 0,
                     "buffers": [ { "role": "input", "binding": 0, "element_count": 6,
                     "shape": [2, 3] }, { "role": "output", "binding": 1, "element_count": 4,
                     "shape": [2, 2] } ],
                     "launch": { "workgroup": { "x": 2, "y": 2, "z": 1 },
                     "dispatch": { "x": 1, "y": 1, "z": 1 } } } ] }"#;
    let err = parse_descriptor(descriptor)
        .expect_err("declared buffer counts must match the buffer roles");
    assert_eq!(err.code, E_DEVICE_ABI_MISMATCH);
}

#[test]
fn parse_rejects_duplicate_bindings() {
    let descriptor = br#"{"schema_version": 2, "target": "llvm-nvvm",
      "kernels": [ { "entry": "k", "element_type": "f32", "element_byte_width": 4,
                     "element_count": 6, "element_counts": [6, 4], "input_buffers": 1,
                     "output_buffers": 1, "accumulation_buffers": 0,
                     "buffers": [ { "role": "input", "binding": 0, "element_count": 6,
                     "shape": [2, 3] }, { "role": "output", "binding": 0, "element_count": 4,
                     "shape": [2, 2] } ],
                     "launch": { "workgroup": { "x": 2, "y": 2, "z": 1 },
                     "dispatch": { "x": 1, "y": 1, "z": 1 } } } ] }"#;
    let err = parse_descriptor(descriptor).expect_err("duplicate binding must fail closed");
    assert_eq!(err.code, E_DEVICE_ABI_MISMATCH);
}

#[test]
fn parse_rejects_zero_launch_axis() {
    let descriptor = br#"{"schema_version": 2, "target": "llvm-nvvm",
      "kernels": [ { "entry": "k", "element_type": "f32", "element_byte_width": 4,
                     "element_count": 1, "element_counts": [1, 1], "input_buffers": 1,
                     "output_buffers": 1, "accumulation_buffers": 0,
                     "buffers": [ { "role": "input", "binding": 0, "element_count": 1,
                     "shape": [1] }, { "role": "output", "binding": 1, "element_count": 1,
                     "shape": [1] } ],
                     "launch": { "workgroup": { "x": 1, "y": 1, "z": 1 },
                     "dispatch": { "x": 0, "y": 1, "z": 1 } } } ] }"#;
    let err = parse_descriptor(descriptor).expect_err("zero dispatch axis must fail closed");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
    assert!(err.message.contains("zero"), "{}", err.message);
}

#[test]
fn parse_rejects_out_of_range_axis_without_saturation() {
    // 2^32 does not fit u32: the adapter must reject, never saturate to
    // u32::MAX.
    let descriptor = br#"{"schema_version": 2, "target": "llvm-nvvm",
      "kernels": [ { "entry": "k", "element_type": "f32", "element_byte_width": 4,
                     "element_count": 1, "element_counts": [1, 1], "input_buffers": 1,
                     "output_buffers": 1, "accumulation_buffers": 0,
                     "buffers": [ { "role": "input", "binding": 0, "element_count": 1,
                     "shape": [1] }, { "role": "output", "binding": 1, "element_count": 1,
                     "shape": [1] } ],
                     "launch": { "workgroup": { "x": 4294967296, "y": 1, "z": 1 },
                     "dispatch": { "x": 1, "y": 1, "z": 1 } } } ] }"#;
    let err = parse_descriptor(descriptor)
        .expect_err("an out-of-range axis must fail closed, not saturate to u32::MAX");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
    assert!(err.message.contains("u32"), "{}", err.message);
    assert!(err.message.contains("not saturated"), "{}", err.message);
}

#[test]
fn parse_rejects_matmul_plan_shape_contradiction() {
    // The buffer's own count is internally consistent (shape [5], count 5),
    // but the `tiled_matmul` plan expects M·K = 6 at position 0 — the plan
    // cross-check must fail closed.
    let descriptor = br#"{"schema_version": 2, "target": "llvm-nvvm",
      "kernels": [ { "entry": "k", "element_type": "f32", "element_byte_width": 4,
                     "element_count": 5, "element_counts": [5, 6, 4], "input_buffers": 2,
                     "output_buffers": 1, "accumulation_buffers": 0,
                     "buffers": [ { "role": "input", "binding": 0, "element_count": 5,
                     "shape": [5] }, { "role": "extra-input", "binding": 1,
                     "element_count": 6, "shape": [3, 2] }, { "role": "output",
                     "binding": 2, "element_count": 4, "shape": [2, 2] } ],
                     "launch": { "workgroup": { "x": 8, "y": 8, "z": 1 },
                     "dispatch": { "x": 1, "y": 1, "z": 1 } },
                     "plan": { "kind": "tiled_matmul", "m": 2, "k": 3, "n": 2,
                     "tile": 8, "workgroup_x": 8, "workgroup_y": 8 } } ] }"#;
    let err = parse_descriptor(descriptor)
        .expect_err("the tiled_matmul plan must match the M·K buffer count");
    assert_eq!(err.code, E_DEVICE_SHAPE_MISMATCH);
    assert!(err.message.contains("expects 6"), "{}", err.message);
}

#[test]
fn parse_rejects_matmul_plan_launch_authority_conflict() {
    let descriptor = br#"{"schema_version": 2, "target": "llvm-nvvm",
      "kernels": [ { "entry": "k", "element_type": "f32", "element_byte_width": 4,
                     "element_count": 6, "element_counts": [6, 6, 4], "input_buffers": 2,
                     "output_buffers": 1, "accumulation_buffers": 0,
                     "buffers": [ { "role": "input", "binding": 0, "element_count": 6,
                     "shape": [2, 3] }, { "role": "extra-input", "binding": 1,
                     "element_count": 6, "shape": [3, 2] }, { "role": "output",
                     "binding": 2, "element_count": 4, "shape": [2, 2] } ],
                     "launch": { "workgroup": { "x": 8, "y": 8, "z": 1 },
                     "dispatch": { "x": 1, "y": 1, "z": 1 } },
                     "plan": { "kind": "tiled_matmul", "m": 2, "k": 3, "n": 2,
                     "tile": 8, "workgroup_x": 16, "workgroup_y": 8 } } ] }"#;
    let err = parse_descriptor(descriptor)
        .expect_err("plan workgroup_x must agree with the launch workgroup");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
    assert!(
        err.message.contains("single launch authority"),
        "{}",
        err.message
    );
}

/// F1.1 tree-reduction sidecar: stored length 8192, 256-wide workgroups,
/// 32 partials. `dispatch.x` is the caller-supplied grid axis.
fn tree_reduction_f11_descriptor(dispatch_x: u64) -> String {
    format!(
        r#"{{"schema_version": 2, "target": "llvm-nvvm",
      "kernels": [ {{ "entry": "reduce_sum", "element_type": "f32", "element_byte_width": 4,
                     "element_count": 8192, "element_counts": [8192, 32], "input_buffers": 1,
                     "output_buffers": 1, "accumulation_buffers": 0,
                     "buffers": [ {{ "role": "input", "binding": 0, "element_count": 8192,
                     "shape": [8192] }}, {{ "role": "output", "binding": 1, "element_count": 32,
                     "shape": [32] }} ],
                     "launch": {{ "workgroup": {{ "x": 256, "y": 1, "z": 1 }},
                     "dispatch": {{ "x": {dispatch_x}, "y": 1, "z": 1 }} }},
                     "plan": {{ "kind": "tree_reduction", "op": "sum", "length": 8192,
                     "partials": 32, "workgroup_x": 256 }} }} ] }}"#
    )
}

#[test]
fn parse_rejects_reduction_f11_grid_authority_conflict() {
    // F1.1: dispatch.x copies the stored length (8192) instead of partials
    // (32). The adapter must reject the contradiction by named
    // E_DEVICE_DESCRIPTOR — not launch 8192 blocks.
    let descriptor = tree_reduction_f11_descriptor(8192);
    let err = parse_descriptor(descriptor.as_bytes())
        .expect_err("F1.1 dispatch.x=8192 vs partials=32 must fail closed");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
    assert!(
        err.message.contains("single launch authority"),
        "{}",
        err.message
    );
    assert!(err.message.contains("8192"), "{}", err.message);
    assert!(err.message.contains("32"), "{}", err.message);
}

#[test]
fn parse_accepts_reduction_plan_with_partials_grid() {
    // Valid plan: grid.x == partials == 32, the launch the adapter must
    // dispatch (32 blocks × 256 threads, not 8192).
    let plan = parse_descriptor(tree_reduction_f11_descriptor(32).as_bytes())
        .expect("grid.x == partials is a valid tree_reduction launch");
    assert_eq!(plan.entry, "reduce_sum");
    assert_eq!(plan.plan_kind.as_deref(), Some("tree_reduction"));
    assert_eq!(plan.grid, [32, 1, 1], "valid plan launches 32 blocks");
    assert_eq!(plan.block, [256, 1, 1]);
}

#[test]
fn parse_rejects_matmul_plan_grid_authority_conflict() {
    // Rung-0 M=2,N=2,tile=8 → expected grid (ceil(2/8), ceil(2/8)) = (1,1).
    // A 2×2 dispatch contradicts the plan-derived tile grid.
    let descriptor = br#"{"schema_version": 2, "target": "llvm-nvvm",
      "kernels": [ { "entry": "k", "element_type": "f32", "element_byte_width": 4,
                     "element_count": 6, "element_counts": [6, 6, 4], "input_buffers": 2,
                     "output_buffers": 1, "accumulation_buffers": 0,
                     "buffers": [ { "role": "input", "binding": 0, "element_count": 6,
                     "shape": [2, 3] }, { "role": "extra-input", "binding": 1,
                     "element_count": 6, "shape": [3, 2] }, { "role": "output",
                     "binding": 2, "element_count": 4, "shape": [2, 2] } ],
                     "launch": { "workgroup": { "x": 8, "y": 8, "z": 1 },
                     "dispatch": { "x": 2, "y": 2, "z": 1 } },
                     "plan": { "kind": "tiled_matmul", "m": 2, "k": 3, "n": 2,
                     "tile": 8, "workgroup_x": 8, "workgroup_y": 8 } } ] }"#;
    let err = parse_descriptor(descriptor)
        .expect_err("tiled_matmul grid must match ceil(n/tile)×ceil(m/tile)");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
    assert!(
        err.message.contains("single launch authority"),
        "{}",
        err.message
    );
}

#[test]
fn parse_rejects_unknown_plan_kind() {
    let descriptor = br#"{"schema_version": 2, "target": "llvm-nvvm",
      "kernels": [ { "entry": "k", "element_type": "f32", "element_byte_width": 4,
                     "element_count": 6, "element_counts": [6, 6, 4], "input_buffers": 2,
                     "output_buffers": 1, "accumulation_buffers": 0,
                     "buffers": [ { "role": "input", "binding": 0, "element_count": 6,
                     "shape": [2, 3] }, { "role": "extra-input", "binding": 1,
                     "element_count": 6, "shape": [3, 2] }, { "role": "output",
                     "binding": 2, "element_count": 4, "shape": [2, 2] } ],
                     "launch": { "workgroup": { "x": 8, "y": 8, "z": 1 },
                     "dispatch": { "x": 1, "y": 1, "z": 1 } },
                     "plan": { "kind": "frobnicate", "m": 2, "k": 3, "n": 2 } } ] }"#;
    let err = parse_descriptor(descriptor).expect_err("unknown plan kind must fail closed");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
    assert!(err.message.contains("frobnicate"), "{}", err.message);
}

#[test]
fn execute_launches_matmul_and_matches_oracle() {
    let mut session = fake_session();
    let receipt = launch_descriptor(
        &mut session,
        RUNG0_MATMUL_DESCRIPTOR.as_bytes(),
        b"// fake compiler-owned PTX bytes",
        &rung0_inputs(),
        Some(&rung0_oracle()),
    )
    .expect("adapter launch");
    assert_eq!(receipt.entry, "rung0_matmul_kernel");
    assert_eq!(receipt.launches, 1, "single launch authority");
    assert_eq!(
        receipt.allocated_buffers, 3,
        "buffers sized from the descriptor"
    );
    assert_eq!(receipt.copy_ins, 2, "two host input buffers copied in");
    assert_eq!(receipt.zero_fills, 0, "rung-0 has no accumulation buffers");
    assert_eq!(receipt.readbacks, 1, "one output buffer read back");
    assert_eq!(
        receipt.releases, 4,
        "3 buffers + module released after the launch"
    );
    let output = receipt
        .outputs
        .get(&2)
        .expect("output binding 2 is read back");
    assert_eq!(output, &RUNG0_ORACLE.to_vec());
    let oracle = receipt.oracle.expect("oracle check recorded");
    assert!(oracle.matched);
    assert_eq!(oracle.max_abs_delta, 0.0);

    // Leak-free bar (S2-2 posture): nothing persists after the launch.
    assert_eq!(session.live_handle_count(), 0);
    let counters = session.driver_counters();
    assert_eq!(counters.module_loads, 1);
    assert_eq!(counters.module_releases, 1);
    assert_eq!(counters.buffer_allocs, 3);
    assert_eq!(counters.buffer_releases, 3);
}

#[test]
fn execute_rejects_missing_input() {
    let mut session = fake_session();
    let mut inputs = BTreeMap::new();
    inputs.insert(0, RUNG0_A.to_vec());
    let err = launch_descriptor(
        &mut session,
        RUNG0_MATMUL_DESCRIPTOR.as_bytes(),
        b"// fake PTX",
        &inputs,
        None,
    )
    .expect_err("a missing input buffer must fail closed");
    assert_eq!(err.code, E_DEVICE_SHAPE_MISMATCH);
    assert!(err.message.contains("binding 1"), "{}", err.message);
    // Error-path teardown: the failed launch leaks nothing.
    assert_eq!(session.live_handle_count(), 0);
}

#[test]
fn execute_rejects_wrong_input_length() {
    let mut session = fake_session();
    let mut inputs = rung0_inputs();
    inputs.insert(0, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let err = launch_descriptor(
        &mut session,
        RUNG0_MATMUL_DESCRIPTOR.as_bytes(),
        b"// fake PTX",
        &inputs,
        None,
    )
    .expect_err("a wrong-sized input must fail closed");
    assert_eq!(err.code, E_DEVICE_SHAPE_MISMATCH);
    assert_eq!(session.live_handle_count(), 0);
}

#[test]
fn execute_propagates_entry_mismatch() {
    let mut session = CudaHostSession::with_driver(Box::new(
        FakeCudaDriver::default()
            .with_known_entry("addita")
            .with_matmul_simulation(2, 3, 2),
    ))
    .expect("fake admit");
    let err = launch_descriptor(
        &mut session,
        RUNG0_MATMUL_DESCRIPTOR.as_bytes(),
        b"// fake PTX",
        &rung0_inputs(),
        None,
    )
    .expect_err("a module without the declared entry must fail closed");
    assert_eq!(err.code, E_DEVICE_ENTRY_MISMATCH);
    assert_eq!(session.live_handle_count(), 0);
}

#[test]
fn execute_rejects_empty_ptx() {
    let mut session = fake_session();
    let err = launch_descriptor(
        &mut session,
        RUNG0_MATMUL_DESCRIPTOR.as_bytes(),
        b"",
        &rung0_inputs(),
        None,
    )
    .expect_err("empty PTX must fail closed");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
    assert_eq!(session.live_handle_count(), 0);
}

#[test]
fn execute_rejects_non_f32_transfer_route() {
    // `i32` is inside the NVVM scalar family, so the descriptor parses; the
    // session's transfer surface is f32-only, so the launch fails closed
    // before any driver work.
    let descriptor = br#"{"schema_version": 2, "target": "llvm-nvvm",
      "kernels": [ { "entry": "int_kernel", "element_type": "i32", "element_byte_width": 4,
                     "element_count": 4, "element_counts": [4, 4], "input_buffers": 1,
                     "output_buffers": 1, "accumulation_buffers": 0,
                     "buffers": [ { "role": "input", "binding": 0, "element_count": 4,
                     "shape": [4] }, { "role": "output", "binding": 1, "element_count": 4,
                     "shape": [4] } ],
                     "launch": { "workgroup": { "x": 4, "y": 1, "z": 1 },
                     "dispatch": { "x": 1, "y": 1, "z": 1 } } } ] }"#;
    let plan = parse_descriptor(descriptor).expect("i32 is inside the NVVM family");
    assert_eq!(plan.element_ty, NvvmElementType::I32);
    let mut session = fake_session();
    let err = launch_descriptor(
        &mut session,
        descriptor,
        b"// fake PTX",
        &BTreeMap::new(),
        None,
    )
    .expect_err("non-f32 transfers must fail closed");
    assert_eq!(err.code, E_DEVICE_DTYPE_MISMATCH);
    assert!(err.message.contains("i32"), "{}", err.message);
    assert_eq!(session.live_handle_count(), 0);
}

#[test]
fn schema_constant_is_v2() {
    assert_eq!(NVVM_DESCRIPTOR_SCHEMA_VERSION, 2);
}
