//! Companion tests for [`super::session`] (repo companion-test convention).

use super::*;
use crate::metal_host::{FakeMetalDriver, FakeQkvCarrierSim, MetalHostSession};

const ENTRY: &str = "decode_blk_0_QkvProjection";

/// Dense-f32 M=1 decode geometry: head_dim 8, hidden 16, kv_heads 2,
/// q_per_kv 3 (q_width 48), K/V arenas of 8 rows (kv_width 16, 128
/// elements), capacity-sized RoPE tables of 8 rows (32 elements).
fn carrier_sim(rotate_half: bool, baked_query_rows: u32, skew: f32) -> FakeQkvCarrierSim {
    FakeQkvCarrierSim {
        entry: ENTRY.to_owned(),
        activation: 0,
        q_weight: 1,
        k_weight: 2,
        v_weight: 3,
        q_bias: None,
        cos: 4,
        sin: 5,
        cursor: 6,
        q_out: 7,
        baked_query_rows,
        rows: 1,
        hidden: 16,
        kv_heads: 2,
        q_per_kv: 3,
        head_dim: 8,
        rotate_half,
        skew,
    }
}

struct ProbeFixture {
    runtime: DeviceRuntime,
    module: DeviceHandle,
    kernel: SessionKernel,
    plan: FusedQkvPlan,
    buffers: BTreeMap<BufferKey, DeviceHandle>,
}

fn probe_fixture(sim: FakeQkvCarrierSim, cursor: [f32; 4]) -> ProbeFixture {
    let mut runtime = DeviceRuntime::Metal(
        MetalHostSession::with_driver(Box::new(
            FakeMetalDriver::default()
                .with_known_entry(ENTRY)
                .with_qkv_carrier_sim(sim),
        ))
        .expect("fake metal admit"),
    );
    let slot =
        |buffer_id: u32, name: &str, role: DeviceBufferRole, binding: u32, element_count: u64| {
            SessionSlot {
                buffer_id,
                version: 1,
                buffer_name: name.to_owned(),
                role,
                binding,
                element_ty: DeviceDataType::F32,
                element_count,
            }
        };
    let slots = vec![
        slot(360, "decode.blk0.a", DeviceBufferRole::InOut, 0, 16),
        slot(73, "blk.0.attn_q.weight", DeviceBufferRole::Input, 1, 768),
        slot(74, "blk.0.attn_k.weight", DeviceBufferRole::Input, 2, 256),
        slot(75, "blk.0.attn_v.weight", DeviceBufferRole::Input, 3, 256),
        slot(3, "prefill.rope.cos", DeviceBufferRole::Input, 4, 32),
        slot(4, "prefill.rope.sin", DeviceBufferRole::Input, 5, 32),
        slot(6, "kv.invocation_state", DeviceBufferRole::Input, 6, 4),
        slot(361, "decode.blk0.q_gemv", DeviceBufferRole::InOut, 7, 48),
        slot(7, "kv.cache_k.0", DeviceBufferRole::InOut, 8, 128),
        slot(8, "kv.cache_v.0", DeviceBufferRole::InOut, 9, 128),
    ];
    let mut buffer_meta = BTreeMap::new();
    buffer_meta.insert(
        (9_000, 1),
        SessionBufferMeta {
            name: "blk.0.attn_norm.weight".to_owned(),
            semantic_value: 0,
            role: DeviceBufferRole::Input,
            element_ty: DeviceDataType::F32,
            element_count: 16,
            byte_length: 64,
            lifetime: DeviceBufferLifetime::PerProgram,
            initialization: DeviceBufferInitialization::HostProvided,
        },
    );
    let plan = build_fused_qkv_plan(
        ENTRY,
        &slots,
        &buffer_meta,
        Some(FusedAttentionAxes { head_dim: 8 }),
        [1, 1, 1],
    )
    .expect("the M=1 decode plan builds from the frozen slot facts")
    .expect("entry matched");
    assert_eq!(plan.rows, 1);
    let module = runtime
        .load_module(b"qkv-probe-module")
        .expect("module loads");
    let mut buffers = BTreeMap::new();
    let seed = |runtime: &mut DeviceRuntime,
                buffers: &mut BTreeMap<BufferKey, DeviceHandle>,
                id: u32,
                values: &[f32]| {
        let handle = runtime
            .alloc_bytes(values.len() * 4)
            .expect("probe buffer allocates");
        runtime
            .copy_in_f32(&handle, values)
            .expect("probe input copies");
        buffers.insert((id, 1), handle);
    };
    seed(
        &mut runtime,
        &mut buffers,
        360,
        &(0..16).map(|i| i as f32 * 0.25 - 1.75).collect::<Vec<_>>(),
    );
    seed(
        &mut runtime,
        &mut buffers,
        73,
        &(0..768)
            .map(|i| (i % 13) as f32 * 0.07 - 0.42)
            .collect::<Vec<_>>(),
    );
    seed(
        &mut runtime,
        &mut buffers,
        74,
        &(0..256)
            .map(|i| (i % 11) as f32 * 0.06 - 0.3)
            .collect::<Vec<_>>(),
    );
    seed(
        &mut runtime,
        &mut buffers,
        75,
        &(0..256)
            .map(|i| (i % 9) as f32 * 0.05 - 0.2)
            .collect::<Vec<_>>(),
    );
    seed(
        &mut runtime,
        &mut buffers,
        3,
        &(0..32).map(|i| (i % 7) as f32 * 0.11).collect::<Vec<_>>(),
    );
    seed(
        &mut runtime,
        &mut buffers,
        4,
        &(0..32)
            .map(|i| 0.02 + (i % 5) as f32 * 0.09)
            .collect::<Vec<_>>(),
    );
    seed(&mut runtime, &mut buffers, 6, &cursor);
    seed(&mut runtime, &mut buffers, 361, &[0.0f32; 48]);
    seed(&mut runtime, &mut buffers, 7, &vec![0.0f32; 128]);
    seed(&mut runtime, &mut buffers, 8, &vec![0.0f32; 128]);
    let kernel = SessionKernel {
        entry: ENTRY.to_owned(),
        slots,
        fused_qkv: None,
        fused_residual_rms: None,
        grid: [1, 1, 1],
        block: [32, 1, 1],
    };
    ProbeFixture {
        runtime,
        module,
        kernel,
        plan,
        buffers,
    }
}

/// Real M=1 decode facts (position 5, valid_len_after 6, query_rows 1)
/// pass the carrier's guard: the carrier publishes Q, the reference is
/// non-stale, and the probe learns the carrier's baked rotate-half
/// pairing (the Qwen RED case).
#[test]
fn probe_learns_carrier_pairing_under_real_m1_facts() {
    let mut fixture = probe_fixture(carrier_sim(true, 1, 0.0), [5.0, 6.0, 1.0, 0.0]);
    let learned = std::cell::RefCell::new(None);
    let pairing = fused_rotate_half(
        &mut fixture.runtime,
        &fixture.module,
        &fixture.kernel,
        &fixture.plan,
        &fixture.buffers,
        &BTreeMap::new(),
        &learned,
    )
    .expect("the probe resolves the carrier's baked rotate-half pairing");
    assert!(pairing, "the carrier baked rotate_half=true");
    assert_eq!(*learned.borrow(), Some(true));
}

/// The consecutive-pair carrier (SmolLM2 class) is learned the same way.
#[test]
fn probe_learns_pair_rotate_under_real_m1_facts() {
    let mut fixture = probe_fixture(carrier_sim(false, 1, 0.0), [5.0, 6.0, 1.0, 0.0]);
    let learned = std::cell::RefCell::new(None);
    let pairing = fused_rotate_half(
        &mut fixture.runtime,
        &fixture.module,
        &fixture.kernel,
        &fixture.plan,
        &fixture.buffers,
        &BTreeMap::new(),
        &learned,
    )
    .expect("the probe resolves the carrier's baked pair-rotate pairing");
    assert!(!pairing, "the carrier baked rotate_half=false");
}

/// The M1-NUM zeroed-cursor case: stale cursor facts fail closed BEFORE
/// the carrier launch, naming the unavailable fact — never a blind
/// launch that compares the Q buffer's stale contents.
#[test]
fn zeroed_cursor_fails_closed_naming_the_fact() {
    let mut fixture = probe_fixture(carrier_sim(true, 1, 0.0), [0.0, 0.0, 0.0, 0.0]);
    let learned = std::cell::RefCell::new(None);
    let error = fused_rotate_half(
        &mut fixture.runtime,
        &fixture.module,
        &fixture.kernel,
        &fixture.plan,
        &fixture.buffers,
        &BTreeMap::new(),
        &learned,
    )
    .expect_err("a zeroed cursor must fail closed before the carrier launch");
    assert!(
        error.message.contains("query_rows"),
        "the error names the unavailable fact: {}",
        error.message
    );
    assert!(
        error.message.contains("stale reference"),
        "the error names the stale-reference consequence: {}",
        error.message
    );
}

/// Kimi's surviving mechanism, reproduced: the host-side facts validate
/// against the plan, but the compiled carrier's BAKED query-row constant
/// disagrees — the carrier's fail-early guard rejects the decode and
/// writes no Q. The poison-fill assertion catches the stale reference
/// instead of comparing it under both pairings.
#[test]
fn guard_trip_despite_valid_host_facts_is_detected_as_stale() {
    let mut fixture = probe_fixture(carrier_sim(true, 2, 0.0), [5.0, 6.0, 1.0, 0.0]);
    let learned = std::cell::RefCell::new(None);
    let error = fused_rotate_half(
        &mut fixture.runtime,
        &fixture.module,
        &fixture.kernel,
        &fixture.plan,
        &fixture.buffers,
        &BTreeMap::new(),
        &learned,
    )
    .expect_err("a carrier that publishes no Q must fail the probe closed");
    assert!(
        error.message.contains("published no Q"),
        "the error states the carrier wrote nothing: {}",
        error.message
    );
    assert!(
        error.message.contains("47/48"),
        "the error reports the surviving poison count: {}",
        error.message
    );
    assert!(
        error.message.contains("STALE"),
        "the error names the stale reference: {}",
        error.message
    );
}

/// Repair law 4: a both-candidates mismatch reports per-candidate
/// deltas and the probe inputs instead of "matched neither pairing".
#[test]
fn no_match_error_reports_candidate_deltas_and_inputs() {
    let mut fixture = probe_fixture(carrier_sim(true, 1, 1.0), [5.0, 6.0, 1.0, 0.0]);
    let learned = std::cell::RefCell::new(None);
    let error = fused_rotate_half(
        &mut fixture.runtime,
        &fixture.module,
        &fixture.kernel,
        &fixture.plan,
        &fixture.buffers,
        &BTreeMap::new(),
        &learned,
    )
    .expect_err("a skewed carrier Q matches neither candidate");
    for fragment in [
        "rotate_half=false max_delta=",
        "rotate_half=true max_delta=",
        "rows=1",
        "head_dim=8",
        "position 5",
        "tolerance=",
        "stale",
    ] {
        assert!(
            error.message.contains(fragment),
            "diagnostic missing `{fragment}`: {}",
            error.message
        );
    }
}
