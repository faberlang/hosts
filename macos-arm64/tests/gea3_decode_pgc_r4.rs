//! PGC-R4 focused device proof: simdgroup-matrix prefill GEMM recipes.
//!
//! The four live prefill projection entries (`prefill_gemm_qo`,
//! `prefill_gemm_kv`, `prefill_gemm_gate_up`, `prefill_gemm_down`, plus
//! the shared-body `prefill_gemm_o`) keep their launch graph exactly —
//! workgroup (8,8,1), tile-grid dispatch, the five-binding ABI — while
//! the Metal body swaps the scalar cooperative-tile inner product for
//! the simdgroup fast path (two simdgroups split K by parity; full bands
//! load directly from device memory with zero K-loop barriers; the M=36
//! tail band stages double-buffered float4 tiles at ONE barrier per K
//! slice with zero-fill guards).
//!
//! The default tests are the binding/geometry and FMA-census oracle over
//! the fake Metal session. The ignored physical gate compiles the
//! exported entry sources on the real device, proves the M=36/N-multiple
//! tail edges read zero-fill without changing valid outputs, and checks
//! the outputs against the CPU reference AND the frozen old-recipe
//! per-family tolerance (`evidence/PGC-R4/frozen-tolerance.json`) — the
//! two-class class-B oracle: simdgroup accumulate contracts the
//! multiply-add, so tolerance (never widened) is the pass law, with the
//! old recipe's own vs-CPU deviation as the reference bound. No wall
//! claim lives here (condition-B rider).

use faber_host_macos_arm64::device_descriptor::DeviceDataType;
use faber_host_macos_arm64::metal_host::{MetalLaunchBinding, MetalHandleId};
use faber_host_macos_arm64::device_host::{DeviceRuntime, DeviceSession};
use faber_host_macos_arm64::{FakeMetalDriver, MetalHostSession};
use host_coordinator::DeviceBackend;

const PREFILL_ROWS: usize = 36;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn f32_from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(F32_BYTES)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}
const F32_BYTES: usize = 4;
const TILE: usize = 8;

fn frozen_abs_delta(
    observed: &[f32],
    reference: &[f32],
    bound: f32,
) -> Result<f32, String> {
    if !bound.is_finite() || bound < 0.0 {
        return Err(format!("frozen bound is not finite and non-negative: {bound}"));
    }
    if observed.len() != reference.len() {
        return Err(format!(
            "comparison length mismatch: observed {} values, reference {} values",
            observed.len(),
            reference.len()
        ));
    }
    let mut max_abs = 0.0_f32;
    for (&observed, &reference) in observed.iter().zip(reference) {
        if !observed.is_finite() || !reference.is_finite() {
            return Err(format!(
                "comparison contains non-finite values: observed {observed}, reference {reference}"
            ));
        }
        let abs = (observed - reference).abs();
        if !abs.is_finite() {
            return Err("comparison delta is non-finite".to_string());
        }
        max_abs = max_abs.max(abs);
    }
    if max_abs > bound {
        return Err(format!(
            "max_abs {max_abs} exceeds frozen bound {bound}"
        ));
    }
    Ok(max_abs)
}

fn assert_frozen_abs_bound(
    entry: &str,
    comparison: &str,
    observed: &[f32],
    reference: &[f32],
    bound: f32,
) {
    frozen_abs_delta(observed, reference, bound)
        .unwrap_or_else(|error| panic!("{entry} {comparison}: {error}"));
}

#[test]
fn frozen_old_recipe_bound_rejects_drift() {
    let new_output = [1.0_f32, -2.0];
    let old_output = [1.0_f32, -2.001];
    let error = frozen_abs_delta(&new_output, &old_output, 0.0005)
        .expect_err("old-recipe drift must exceed the frozen bound");
    assert!(error.contains("exceeds frozen bound"));
}

/// The prefill projection family: (entry, K, N) at M=36 rows.
const FAMILY: &[(&str, usize, usize)] = &[
    ("prefill_gemm_qo", 960, 960),
    ("prefill_gemm_kv", 960, 320),
    ("prefill_gemm_gate_up", 960, 2560),
    ("prefill_gemm_down", 2560, 960),
];

/// The frozen per-family tolerance record emitted by the PGC-R4 device
/// A/B capture: `{"entry": {"max_abs_vs_cpu": .., "max_rel_vs_cpu": ..,
/// "old_max_abs_vs_cpu": .., "max_abs_vs_old": ..}}`. The physical gate
/// asserts the observed deltas stay at or under the frozen values —
/// never widened after observation.
#[derive(Debug, serde::Deserialize)]
struct FrozenTolerance {
    max_abs_vs_cpu: f32,
    max_rel_vs_cpu: f32,
    old_max_abs_vs_cpu: f32,
    max_abs_vs_old: f32,
}

fn fake_metal(entries: &[&str]) -> MetalHostSession {
    let mut driver = FakeMetalDriver::default();
    for entry in entries {
        driver = driver.with_known_entry(*entry);
    }
    MetalHostSession::with_driver(Box::new(driver)).expect("fake Metal")
}

/// The launch contract is UNCHANGED by the recipe swap: the same
/// five-binding ABI (input, weights, two inert plan extras, output) at
/// the same tile-grid geometry — one (8,8,1) workgroup per ceil-divided
/// output tile. This is the B2-class dispatch-shape fact the card pins
/// separately from the numeric contract.
#[test]
fn pgc_r4_prefill_gemm_launch_geometry_is_unchanged() {
    let entries: Vec<&str> = FAMILY.iter().map(|(entry, _, _)| *entry).collect();
    let session = fake_metal(&entries);
    let mut runtime = DeviceRuntime::Metal(session);
    for &(entry, k, n) in FAMILY {
        let input = runtime
            .alloc_bytes(PREFILL_ROWS * k * F32_BYTES)
            .expect("input");
        let weights = runtime.alloc_bytes(k * n * F32_BYTES).expect("weights");
        let extra0 = runtime.alloc_bytes(F32_BYTES).expect("plan extra 0");
        let extra1 = runtime.alloc_bytes(F32_BYTES).expect("plan extra 1");
        let output = runtime
            .alloc_bytes(PREFILL_ROWS * n * F32_BYTES)
            .expect("output");
        let module = runtime.load_module(b"pgc-r4-prefill-gemm").expect("module");
        let handles = [input, weights, extra0, extra1, output];
        let spans = [
            (PREFILL_ROWS * k) as u64,
            (k * n) as u64,
            1_u64,
            1_u64,
            (PREFILL_ROWS * n) as u64,
        ];
        let bindings: Vec<MetalLaunchBinding> = handles
            .iter()
            .zip(spans)
            .enumerate()
            .map(|(index, (handle, span))| MetalLaunchBinding {
                handle: MetalHandleId(handle.id),
                binding_index: index as u32,
                byte_offset: 0,
                view_span: span * F32_BYTES as u64,
            })
            .collect();
        let DeviceRuntime::Metal(session) = &mut runtime else {
            unreachable!("fake Metal runtime");
        };
        session
            .launch_kernel_bound(
                MetalHandleId(module.id),
                entry,
                &bindings,
                [(n.div_ceil(TILE)) as u32, (PREFILL_ROWS.div_ceil(TILE)) as u32, 1],
                [TILE as u32, TILE as u32, 1],
            )
            .unwrap_or_else(|error| panic!("{entry} launch: {error}"));
        session.sync().expect("launch sync");
    }
}

/// The useful-vs-dispatched FMA census (condition-B primary evidence,
/// frozen numbers also recorded under `evidence/PGC-R4/census.md`,
/// corrected per CTO-B finding 2, verdict `a694cd2c`): "dispatched FMA"
/// counts every matrix slot the 8×8 MMA executes — the simdgroup recipe
/// dispatches the tile-PADDED extent `ceil(M/8)·8 × N × K` per launch,
/// the same 40-row-padded slot count the old scalar body ran. The
/// partial band's zero-filled rows are executed slots, not skipped
/// slots: useful FMAs (`M × N × K`) and zero-filled/padded slots
/// (dispatched − useful) are separate census columns and the
/// padding-removal criterion is NOT met. This test pins the arithmetic
/// the census table records; the per-family launch counts ride the
/// exported plan.
#[test]
fn pgc_r4_fma_census_dispatched_vs_zero_filled_pinned() {
    // Launch counts per admitted entry (exported plan, unchanged from the
    // R3 family); the census totals multiply the per-launch class by
    // these counts.
    let launches: &[(&str, u64)] = &[
        ("prefill_gemm_qo", 32),
        ("prefill_gemm_kv", 64),
        ("prefill_gemm_gate_up", 64),
        ("prefill_gemm_down", 32),
    ];
    // The four-entry family's 40-row padding class: 4 padded rows over
    // every (N × K) extent.
    assert_eq!(PREFILL_ROWS.div_ceil(TILE) * TILE - PREFILL_ROWS, 4);
    let mut zero_filled_total: u64 = 0;
    for &(entry, launches) in launches {
        let (k, n) = FAMILY
            .iter()
            .find(|(name, _, _)| *name == entry)
            .map(|&(_, k, n)| (k, n))
            .expect("admitted entry in FAMILY");
        let useful = (PREFILL_ROWS * n * k) as u64;
        let dispatched = (PREFILL_ROWS.div_ceil(TILE) * TILE * n * k) as u64;
        let zero_filled = dispatched - useful;
        // Corrected census law: dispatched counts every slot the 8×8 MMA
        // executes, so useful FMAs do NOT equal dispatched FMAs — the
        // partial band's zero-filled rows still execute as slots.
        assert!(useful < dispatched, "{entry} carries a padded slot class");
        assert_eq!(
            zero_filled,
            (4 * n * k) as u64,
            "{entry} zero-filled/padded slots per launch"
        );
        zero_filled_total += zero_filled * launches;
    }
    assert_eq!(
        zero_filled_total,
        (4 * (960 * 960 * 32 + 960 * 320 * 64 + 960 * 2560 * 64 + 2560 * 960 * 32)) as u64,
        "the family's zero-filled/padded slot total across launches — still executed (~1.14B)"
    );
}

/// Physical gate: compile the exported entry sources on the real device
/// and prove the recipe's numeric contract. Requires:
/// - `GEA3_R4_METAL_SOURCE_DIR`: the exported artifacts dir (contains
///   `prefill_gemm_<name>.metal` emitted from this branch);
/// - `GEA3_R4_OLD_METAL_SOURCE_DIR`: the frozen old R3 recipe artifacts dir;
/// - `GEA3_R4_FROZEN_TOLERANCE`: `frozen-tolerance.json` from
///   `evidence/PGC-R4/`.
///
/// Proves: M=36 tail band zero-fill (rows past 35 never leak garbage —
/// the CPU reference covers exactly the 36 valid rows), the N-multiple
/// edges (960/320/2560 full bands on the zero-barrier direct path), and
/// the class-B tolerances versus both the CPU reference and the old recipe
/// at or under the frozen bounds. The old recipe is executed rather than
/// inferred from the CPU reference, so the new-vs-old bound is fail-closed.
#[test]
#[ignore = "physical Metal gate; requires new/old entry sources and the frozen tolerance record"]
fn pgc_r4_physical_simdgroup_recipe_on_device() {
    let Some(source_dir) = std::env::var_os("GEA3_R4_METAL_SOURCE_DIR") else {
        panic!("GEA3_R4_METAL_SOURCE_DIR must identify the exported .metal dir");
    };
    let Some(old_source_dir) = std::env::var_os("GEA3_R4_OLD_METAL_SOURCE_DIR") else {
        panic!("GEA3_R4_OLD_METAL_SOURCE_DIR must identify the frozen old-recipe .metal dir");
    };
    let Some(tolerance_path) = std::env::var_os("GEA3_R4_FROZEN_TOLERANCE") else {
        panic!("GEA3_R4_FROZEN_TOLERANCE must identify frozen-tolerance.json");
    };
    let record: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&tolerance_path).expect("read frozen tolerance record"),
    )
    .expect("parse frozen tolerance record");
    let frozen_families: std::collections::BTreeMap<String, FrozenTolerance> =
        serde_json::from_value(record.get("families").cloned().expect("families map"))
            .expect("parse frozen per-family bounds");
    let session = MetalHostSession::try_open().expect("physical Metal session");
    let mut runtime = DeviceRuntime::Metal(session);
    for &(entry, k, n) in FAMILY {
        let source_path = std::path::Path::new(&source_dir).join(format!("{entry}.metal"));
        let module_bytes = std::fs::read(&source_path)
            .unwrap_or_else(|error| panic!("read {entry}.metal: {error}"));
        let source = String::from_utf8_lossy(&module_bytes);
        assert!(
            source.contains("simdgroup_multiply_accumulate"),
            "{entry} must carry the simdgroup recipe"
        );
        assert!(
            !source.contains("acc += shared_a["),
            "{entry} must not carry the scalar inner product"
        );

        let old_source_path =
            std::path::Path::new(&old_source_dir).join(format!("{entry}.metal"));
        let old_module_bytes = std::fs::read(&old_source_path)
            .unwrap_or_else(|error| panic!("read frozen old {entry}.metal: {error}"));
        let old_source = String::from_utf8_lossy(&old_module_bytes);
        assert!(
            old_source.contains("acc += shared_a["),
            "{entry} old recipe must carry the scalar inner product"
        );
        assert!(
            !old_source.contains("simdgroup_multiply_accumulate"),
            "{entry} old recipe must not carry the simdgroup recipe"
        );

        let m = PREFILL_ROWS;
        let mut next = 0x2545_F491_4F6C_DD1D_u64;
        let mut value = || {
            next = next.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((next >> 33) as i32) as f32 / (1u64 << 31) as f32
        };
        let input: Vec<f32> = (0..m * k).map(|_| value()).collect();
        let weights: Vec<f32> = (0..n * k).map(|_| value()).collect();
        let input_handle = runtime.alloc_bytes(input.len() * F32_BYTES).expect("input");
        let weights_handle = runtime.alloc_bytes(weights.len() * F32_BYTES).expect("weights");
        let extra0 = runtime.alloc_bytes(F32_BYTES).expect("plan extra 0");
        let extra1 = runtime.alloc_bytes(F32_BYTES).expect("plan extra 1");
        let output_handle = runtime.alloc_bytes(m * n * F32_BYTES).expect("output");
        let old_output_handle = runtime
            .alloc_bytes(m * n * F32_BYTES)
            .expect("old output");
        runtime
            .copy_in_bytes(&input_handle, &f32_bytes(&input), DeviceDataType::F32)
            .expect("input upload");
        runtime
            .copy_in_bytes(
                &weights_handle,
                &f32_bytes(&weights),
                DeviceDataType::F32,
            )
            .expect("weights upload");
        runtime
            .copy_in_bytes(&extra0, &[0u8; 4], DeviceDataType::F32)
            .expect("plan extra 0 upload");
        runtime
            .copy_in_bytes(&extra1, &[0u8; 4], DeviceDataType::F32)
            .expect("plan extra 1 upload");
        for output in [&output_handle, &old_output_handle] {
            runtime
                .copy_in_bytes(
                    output,
                    &vec![0u8; m * n * F32_BYTES],
                    DeviceDataType::F32,
                )
                .expect("output init");
        }
        let old_module = runtime
            .load_module(&old_module_bytes)
            .expect("compile frozen old entry module");
        let module = runtime.load_module(&module_bytes).expect("compile entry module");
        let spans = [
            (m * k) as u64,
            (n * k) as u64,
            1_u64,
            1_u64,
            (m * n) as u64,
        ];
        let old_handles = [
            input_handle,
            weights_handle,
            extra0,
            extra1,
            old_output_handle,
        ];
        let old_bindings: Vec<MetalLaunchBinding> = old_handles
            .iter()
            .zip(spans)
            .enumerate()
            .map(|(index, (handle, span))| MetalLaunchBinding {
                handle: MetalHandleId(handle.id),
                binding_index: index as u32,
                byte_offset: 0,
                view_span: span * F32_BYTES as u64,
            })
            .collect();
        let handles = [input_handle, weights_handle, extra0, extra1, output_handle];
        let bindings: Vec<MetalLaunchBinding> = handles
            .iter()
            .zip(spans)
            .enumerate()
            .map(|(index, (handle, span))| MetalLaunchBinding {
                handle: MetalHandleId(handle.id),
                binding_index: index as u32,
                byte_offset: 0,
                view_span: span * F32_BYTES as u64,
            })
            .collect();
        let DeviceRuntime::Metal(session) = &mut runtime else {
            unreachable!("physical Metal runtime");
        };
        session
            .launch_kernel_bound(
                MetalHandleId(old_module.id),
                entry,
                &old_bindings,
                [(n.div_ceil(TILE)) as u32, (m.div_ceil(TILE)) as u32, 1],
                [TILE as u32, TILE as u32, 1],
            )
            .expect("frozen old device launch");
        session
            .launch_kernel_bound(
                MetalHandleId(module.id),
                entry,
                &bindings,
                [(n.div_ceil(TILE)) as u32, (m.div_ceil(TILE)) as u32, 1],
                [TILE as u32, TILE as u32, 1],
            )
            .expect("device launch");
        session.sync().expect("device sync");
        let old_observed = runtime
            .readback_bytes(&old_output_handle, DeviceDataType::F32)
            .expect("old output readback");
        let old_observed = f32_from_bytes(&old_observed);
        let observed = runtime
            .readback_bytes(&output_handle, DeviceDataType::F32)
            .expect("output readback");
        let observed = f32_from_bytes(&observed);
        assert_eq!(observed.len(), m * n, "the readback covers exactly M×N");
        assert_eq!(
            old_observed.len(),
            m * n,
            "the old-recipe readback covers exactly M×N"
        );
        // CPU reference over the 36 valid rows: proves the M tail never
        // leaks garbage into a valid row and every valid element is the
        // declared inner product within the frozen tolerance.
        let mut reference = vec![0.0_f32; m * n];
        for row in 0..m {
            for col in 0..n {
                for i in 0..k {
                    reference[row * n + col] += input[row * k + i] * weights[col * k + i];
                }
            }
        }
        let mut max_rel = 0.0_f32;
        for (&reference, &got) in reference.iter().zip(&observed) {
            let rel = (reference - got).abs() / reference.abs().max(1e-3);
            assert!(rel.is_finite(), "{entry} max_rel comparison is non-finite");
            max_rel = max_rel.max(rel);
        }
        let frozen = frozen_families
            .get(entry)
            .unwrap_or_else(|| panic!("frozen tolerance for {entry}"));
        assert_frozen_abs_bound(
            entry,
            "new recipe vs CPU",
            &observed,
            &reference,
            frozen.max_abs_vs_cpu,
        );
        assert!(
            max_rel <= frozen.max_rel_vs_cpu,
            "{entry} max_rel {max_rel} widened past frozen {}",
            frozen.max_rel_vs_cpu
        );
        assert_frozen_abs_bound(
            entry,
            "old recipe vs CPU",
            &old_observed,
            &reference,
            frozen.old_max_abs_vs_cpu,
        );
        assert_frozen_abs_bound(
            entry,
            "new recipe vs old recipe",
            &observed,
            &old_observed,
            frozen.max_abs_vs_old,
        );
    }
}

const _: DeviceBackend = DeviceBackend::Metal;
