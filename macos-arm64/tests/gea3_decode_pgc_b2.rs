//! PGC-B2 additive device proof: the T=1 one-row GEMV dispatch shape.
//!
//! The shared GEA3 decode test is untouched.  This proof mirror-parses the
//! exported decode program plan and pins the one-row launch law at the
//! host boundary: every named T=1 entry (the four projection GEMVs, the
//! score/context attention GEMVs, `lm_head_gemv`, and the decode embedding
//! gather) carries workgroup `(w, 1, 1)` — `w` a tile-multiple lane count
//! (the B2-RETUNE width knob, 8..=64) — over the `ceil(N / w)` grid,
//! never the pre-change 8-row `(8, 8, 1)` tile — while every
//! multi-row launch (the KV appends) keeps the full shared-tile
//! workgroup.  The row-work census counts dispatched versus useful row
//! lanes across one full decode step: the pre-change shape dispatched 8×
//! the useful row work (the card's standing 3,814,195,200 vs 476,528,640
//! FMA proxy).  A fake-Metal structural launch then proves the host
//! session admits and encodes the one-row threadgroup shape.

use std::fs;
use std::path::{Path, PathBuf};

use faber_host_macos_arm64::metal_host::MetalLaunchBinding;
use faber_host_macos_arm64::{FakeMetalDriver, MetalHostSession};
use serde_json::Value;

const PLAN_MEMBER: &str = "gea3-program-plan.json";

/// The named T=1 decode entries and their output widths `N` (the tile-grid
/// extent `ceil(N/8)` each one dispatches).
const ONE_ROW_ENTRIES: &[(&str, u64)] = &[
    ("decode_gemv_qo", 960),
    ("decode_gemv_kv", 320),
    ("decode_gemv_gate_up", 2560),
    ("decode_gemv_down", 960),
    ("decode_score_gemm", 76),
    ("decode_context_gemm", 64),
    ("lm_head_gemv", 49_152),
    ("embedding_gather", 960),
];

/// Multi-row launches that must keep the full `(8, 8, 1)` shared-tile
/// workgroup (the standing KV-append capacity geometry).
const MULTI_ROW_ENTRIES: &[&str] = &["kv_append_k", "kv_append_v"];

fn gea3_artifact_dir() -> PathBuf {
    let root = std::env::var_os("GEA3_ARTIFACT_DIR")
        .map(PathBuf::from)
        .expect("GEA3_ARTIFACT_DIR must identify the exported GEA3 bundle");
    assert!(
        root.join(PLAN_MEMBER).is_file(),
        "missing GEA3 plan member in {}",
        root.display()
    );
    root
}

fn decode_kernels(plan: &Value) -> Vec<&Value> {
    plan["programs"]["decode_step"]["kernels"]
        .as_array()
        .expect("decode_step kernel list")
        .iter()
        .collect()
}

fn workgroup(kernel: &Value) -> (u64, u64, u64) {
    let launch = &kernel["launch"]["workgroup"];
    (
        launch["x"].as_u64().expect("workgroup x"),
        launch["y"].as_u64().expect("workgroup y"),
        launch["z"].as_u64().expect("workgroup z"),
    )
}

fn workgroup_count(kernel: &Value) -> (u64, u64, u64) {
    let grid = &kernel["launch"]["workgroup_count"];
    (
        grid["x"].as_u64().expect("grid x"),
        grid["y"].as_u64().expect("grid y"),
        grid["z"].as_u64().expect("grid z"),
    )
}

/// One decode step's row-work census over the T=1 launches: dispatched row
/// lanes versus the one useful row lane each launch has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RowWorkCensus {
    launches: u64,
    useful_row_lanes: u64,
    dispatched_row_lanes: u64,
}

fn row_work_census(kernels: &[&Value]) -> RowWorkCensus {
    let mut census = RowWorkCensus {
        launches: 0,
        useful_row_lanes: 0,
        dispatched_row_lanes: 0,
    };
    for kernel in kernels {
        let entry = kernel["entry"].as_str().expect("entry name");
        let m = kernel["plan"]["TiledMatMul"]["m"].as_u64();
        if m != Some(1) {
            continue;
        }
        let (_, workgroup_y, _) = workgroup(kernel);
        let (grid_x, grid_y, _) = workgroup_count(kernel);
        assert_eq!(grid_y, 1, "{entry} tile grid is single-row");
        census.launches += 1;
        census.useful_row_lanes += grid_x;
        census.dispatched_row_lanes += grid_x * workgroup_y;
    }
    census
}

#[test]
fn gea3_pgc_b2_decode_t1_entries_dispatch_the_one_row_workgroup() {
    let plan: Value = serde_json::from_slice(
        &fs::read(gea3_artifact_dir().join(PLAN_MEMBER)).expect("read GEA3 program plan"),
    )
    .expect("parse GEA3 program plan");
    let kernels = decode_kernels(&plan);

    for (name, n) in ONE_ROW_ENTRIES {
        let mut seen = false;
        for kernel in &kernels {
            if kernel["entry"].as_str() != Some(name) {
                continue;
            }
            seen = true;
            let (workgroup_x, workgroup_y, workgroup_z) = workgroup(kernel);
            assert_eq!(
                (workgroup_y, workgroup_z),
                (1, 1),
                "{name} must dispatch the one-row workgroup (w, 1, 1)"
            );
            assert!(
                workgroup_x.is_multiple_of(8) && (8..=64).contains(&workgroup_x),
                "{name} one-row width must be a tile-multiple lane count in [8, 64], got {workgroup_x}"
            );
            assert_eq!(
                workgroup_count(kernel),
                (n.div_ceil(workgroup_x), 1, 1),
                "{name} grid is ceil(N / w) over the one-row width"
            );
            assert_eq!(
                kernel["plan"]["TiledMatMul"]["workgroup_y"].as_u64(),
                Some(1),
                "{name} plan carries the one-row workgroup fact"
            );
            assert_eq!(
                kernel["plan"]["TiledMatMul"]["workgroup_x"].as_u64(),
                Some(workgroup_x),
                "{name} plan and launch carry the same one-row width"
            );
        }
        assert!(seen, "{name} appears in the exported decode step");
    }
    for name in MULTI_ROW_ENTRIES {
        let kernel = kernels
            .iter()
            .find(|kernel| kernel["entry"].as_str() == Some(name))
            .expect("multi-row control entry");
        assert_eq!(
            workgroup(kernel),
            (8, 8, 1),
            "{name} keeps the full shared-tile workgroup"
        );
    }
}

#[test]
fn gea3_pgc_b2_decode_row_work_census_counts_useful_versus_dispatched() {
    let plan: Value = serde_json::from_slice(
        &fs::read(gea3_artifact_dir().join(PLAN_MEMBER)).expect("read GEA3 program plan"),
    )
    .expect("parse GEA3 program plan");
    let kernels = decode_kernels(&plan);
    let census = row_work_census(&kernels);
    assert!(census.launches > 0, "the decode step has T=1 launches");
    assert_eq!(
        census.dispatched_row_lanes,
        census.useful_row_lanes,
        "every dispatched row lane is useful after PGC-B2"
    );
    // The pre-change census (workgroup y = 8 on the same launches): the
    // removed row work is exactly seven eighths.
    let pre_change = census.useful_row_lanes * 8;
    assert_eq!(
        pre_change - census.dispatched_row_lanes,
        census.useful_row_lanes * 7,
        "the 8-row shape dispatched 8x the useful row work"
    );
}

/// The host session admits and encodes the one-row threadgroup shape: a
/// structural fake-Metal launch of one projection GEMV with grid
/// `ceil(N/8)` and block `(8, 1, 1)` succeeds and reads back the kernel
/// body's written bytes unchanged (the fake driver is encode-structural;
/// the readback proves the launch and binding path, not kernel math).
#[test]
fn gea3_pgc_b2_host_session_encodes_the_one_row_threadgroup() {
    let mut runtime = MetalHostSession::with_driver(Box::new(
        FakeMetalDriver::default().with_known_entry("decode_gemv_qo"),
    ))
    .expect("fake Metal admission");
    let input = runtime
        .alloc_bytes(960 * 4)
        .expect("input allocation");
    let weights = runtime
        .alloc_bytes(960 * 960 * 4)
        .expect("weights allocation");
    let output = runtime
        .alloc_bytes(960 * 4)
        .expect("output allocation");
    let module = runtime.load_module(b"pgc-b2-one-row").expect("module");
    let binding = |handle, index, span| MetalLaunchBinding {
        handle,
        binding_index: index,
        byte_offset: 0,
        view_span: span,
    };
    let extra = runtime.alloc_bytes(4).expect("plan extra allocation");
    runtime
        .launch_kernel_bound(
            module,
            "decode_gemv_qo",
            &[
                binding(input, 0, 960 * 4),
                binding(weights, 1, 960 * 960 * 4),
                binding(output, 2, 960 * 4),
                // The launch-variant ABI's inert residency uploads (the
                // plan's weight-id bindings the kernel body never reads).
                binding(extra, 3, 4),
                binding(extra, 4, 4),
            ],
            [120, 1, 1],
            [8, 1, 1],
        )
        .expect("one-row threadgroup launch encodes");
    let observed = runtime.readback_f32(output).expect("output readback");
    assert_eq!(observed.len(), 960);
}
