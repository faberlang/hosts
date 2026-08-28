//! PGC-B1 device-dispatch proof for the fixed-1000 decode statue.
//!
//! The export test writes two compile-time member bundles: an early 64-row
//! history bucket and a late 1088-row bucket.  This host-side proof executes
//! both plans through the sequencing device boundary for all 1,000 fixed
//! steps.  It deliberately does not fake tensor values or certify numerical
//! parity; the physical parity receipt remains the source of the 1000/1000
//! output certificate.

use faber_host_macos_arm64::{FakeMetalDriver, MetalHostSession};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};

const ENTRIES: [&str; 4] = [
    "decode_key_transpose",
    "decode_score_gemm",
    "decode_masked_softmax",
    "decode_context_gemm",
];
const FIXED1000_STEPS: usize = 1_000;
const FIXED1000_CAPACITY: u64 = 1_100;
const PREFILL_ROWS: u64 = 36;
const EARLY_EXTENT: u64 = 64;
const LATE_EXTENT: u64 = 1_088;

fn artifact_root() -> PathBuf {
    std::env::var_os("GEA3_PGC_B1_ARTIFACT_DIR")
        .map(PathBuf::from)
        .expect("GEA3_PGC_B1_ARTIFACT_DIR must name the exported PGC-B1 bundles")
}

fn plan(root: &Path) -> Value {
    serde_json::from_slice(
        &fs::read(root.join("gea3-program-plan.json")).expect("read PGC-B1 plan"),
    )
    .expect("parse PGC-B1 plan")
}

fn manifest(root: &Path) -> Value {
    serde_json::from_slice(
        &fs::read(root.join("gea3-artifact-bundle-manifest.json")).expect("read PGC-B1 manifest"),
    )
    .expect("parse PGC-B1 manifest")
}

fn module_image(root: &Path, plan: &Value) -> Vec<u8> {
    plan["module_members"]
        .as_array()
        .expect("module member list")
        .iter()
        .map(|member| {
            let name = member.as_str().expect("module member name");
            fs::read(root.join(name)).unwrap_or_else(|error| panic!("read {name}: {error}"))
        })
        .fold(Vec::new(), |mut image, bytes| {
            image.extend(bytes);
            image
        })
}

fn kernel<'a>(plan: &'a Value, entry: &str) -> &'a Value {
    plan["programs"]["decode_step"]["kernels"]
        .as_array()
        .expect("decode kernels")
        .iter()
        .find(|kernel| kernel["entry"] == entry)
        .unwrap_or_else(|| panic!("missing decode kernel {entry}"))
}

fn extent_for(entry: &str, plan: &Value) -> u64 {
    let kernel = kernel(plan, entry);
    match entry {
        "decode_key_transpose" => kernel["plan"]["Transpose"]["m"]
            .as_u64()
            .expect("transpose work extent"),
        "decode_score_gemm" => kernel["plan"]["TiledMatMul"]["n"]
            .as_u64()
            .expect("score work extent"),
        "decode_masked_softmax" => kernel["plan"]["CausalMaskedSoftmax"]["cols"]
            .as_u64()
            .expect("softmax work extent"),
        "decode_context_gemm" => kernel["plan"]["TiledMatMul"]["k"]
            .as_u64()
            .expect("context work extent"),
        _ => unreachable!("entry list is fixed"),
    }
}

fn assert_extent_geometry(plan: &Value, manifest: &Value, expected: u64) {
    assert!(expected < FIXED1000_CAPACITY);
    assert!(expected >= PREFILL_ROWS + 1);
    assert_eq!(plan["kv_geometry"]["capacity"], FIXED1000_CAPACITY);
    assert_eq!(
        plan["kv_geometry"]["declared_history_length"],
        FIXED1000_CAPACITY
    );
    assert_eq!(
        plan["programs"]["decode_step"]["declared_history_length"],
        FIXED1000_CAPACITY
    );
    assert_eq!(manifest["statue"]["n_predict"], json!(FIXED1000_STEPS));
    assert_eq!(manifest["statue"]["work_extent"], expected);

    for entry in ENTRIES {
        assert_eq!(extent_for(entry, plan), expected, "{entry} work extent");
        let launch = kernel(plan, entry)["launch"].clone();
        assert!(launch["workgroup_count"]["x"].as_u64().unwrap_or(0) > 0);
        assert!(launch["workgroup_count"]["y"].as_u64().unwrap_or(0) > 0);
        assert_eq!(
            launch["dispatch_size"], launch["workgroup_count"],
            "{entry} carries one dispatch geometry"
        );
    }

    // The context and transpose windows address only the bucket while their
    // producer is still the 1,100-row KV allocation.
    for entry in ["decode_key_transpose", "decode_context_gemm"] {
        let resources = kernel(plan, entry)["resources"]
            .as_array()
            .expect("attention resources");
        assert!(
            resources
                .iter()
                .any(|resource| { resource["version"]["sub_window"]["row_count"] == expected }),
            "{entry} carries the declared work window"
        );
    }
}

fn execute_bucket(root: &Path, expected_extent: u64) -> usize {
    let plan = plan(root);
    let manifest = manifest(root);
    assert_extent_geometry(&plan, &manifest, expected_extent);
    let image = module_image(root, &plan);
    assert!(!image.is_empty(), "PGC-B1 Metal module image is empty");

    let mut driver = FakeMetalDriver::default();
    for entry in ENTRIES {
        driver = driver.with_known_entry(entry);
    }
    let mut session = MetalHostSession::with_driver(Box::new(driver)).expect("fake device session");
    let module = session.load_module(&image).expect("load PGC-B1 module");
    let placeholder = session.alloc_bytes(4).expect("allocate fake binding");
    let mut launches = 0;
    let result = (|| {
        for _step in 0..FIXED1000_STEPS {
            for entry in ENTRIES {
                let kernel = kernel(&plan, entry);
                let resources = kernel["resources"].as_array().expect("kernel resources");
                let grid = [
                    kernel["launch"]["workgroup_count"]["x"]
                        .as_u64()
                        .expect("grid x") as usize,
                    kernel["launch"]["workgroup_count"]["y"]
                        .as_u64()
                        .expect("grid y") as usize,
                    kernel["launch"]["workgroup_count"]["z"]
                        .as_u64()
                        .expect("grid z") as usize,
                ];
                let block = [
                    kernel["launch"]["workgroup"]["x"]
                        .as_u64()
                        .expect("block x") as usize,
                    kernel["launch"]["workgroup"]["y"]
                        .as_u64()
                        .expect("block y") as usize,
                    kernel["launch"]["workgroup"]["z"]
                        .as_u64()
                        .expect("block z") as usize,
                ];
                let bindings = vec![placeholder; resources.len()];
                session
                    .launch_kernel_3d(
                        module,
                        entry,
                        &bindings,
                        grid[0] as u32,
                        grid[1] as u32,
                        grid[2] as u32,
                        block[0] as u32,
                        block[1] as u32,
                        block[2] as u32,
                    )
                    .map_err(|error| error.message.clone())?;
                launches += 1;
            }
            session.sync().map_err(|error| error.message.clone())?;
        }
        Ok::<_, String>(launches)
    })();
    let _ = session.release(placeholder);
    let _ = session.release(module);
    result.unwrap_or_else(|error| panic!("PGC-B1 {expected_extent}-row dispatch failed: {error}"))
}

#[test]
fn gea3_decode_pgc_b1_dispatches_early_and_late_work_buckets() {
    assert!(LATE_EXTENT >= PREFILL_ROWS + FIXED1000_STEPS as u64);
    let root = artifact_root();
    let early = execute_bucket(&root.join("early"), EARLY_EXTENT);
    let late = execute_bucket(&root.join("late"), LATE_EXTENT);
    assert_eq!(early, ENTRIES.len() * FIXED1000_STEPS);
    assert_eq!(late, ENTRIES.len() * FIXED1000_STEPS);
}

/// Physical parity receipt join.  The exporter/device-dispatch proof above is
/// structural, while this ignored gate consumes the operator's real Metal
/// receipt and comparator stderr to certify the fixed-output pair as 1000/1000
/// without manufacturing a numerical result in a fake driver.
#[test]
#[ignore = "physical Metal gate; requires PGC-B1 buckets and certified arm evidence"]
fn gea3_decode_pgc_b1_certified_fixed1000_output() {
    let faber_receipt_path = PathBuf::from(
        std::env::var_os("GEA3_PGC_B1_CERTIFIED_RECEIPT")
            .expect("GEA3_PGC_B1_CERTIFIED_RECEIPT must name the Faber receipt"),
    );
    let comparator_stderr_path = PathBuf::from(
        std::env::var_os("GEA3_PGC_B1_COMPARATOR_STDERR")
            .expect("GEA3_PGC_B1_COMPARATOR_STDERR must name comparator stderr"),
    );
    let receipt: Value = serde_json::from_slice(
        &fs::read(&faber_receipt_path).expect("read certified PGC-B1 Faber receipt"),
    )
    .expect("parse certified PGC-B1 Faber receipt");
    assert_eq!(receipt["status"], "green");
    assert_eq!(receipt["statue"]["n_predict"], json!(FIXED1000_STEPS));
    assert_eq!(receipt["statue"]["l_max"], json!(FIXED1000_CAPACITY));
    assert_eq!(
        receipt["execution"]["step_count"]["value"],
        json!(FIXED1000_STEPS)
    );
    assert_eq!(
        receipt["steps"].as_array().map(Vec::len),
        Some(FIXED1000_STEPS + 1),
        "Faber receipt must include prefill plus all fixed decode steps"
    );

    let comparator_stderr_bytes =
        fs::read(&comparator_stderr_path).expect("read comparator stderr");
    let comparator_stderr = String::from_utf8_lossy(&comparator_stderr_bytes);
    assert!(
        comparator_stderr.lines().any(|line| {
            line.contains("eval time")
                && line
                    .split_whitespace()
                    .any(|word| word == "40")
                && line.contains("tokens")
        }),
        "comparator stderr lacks its stable eval-time line; the pinned comparator stops at 40 tokens (greedy EOS) in every recorded family capture"
    );
}
