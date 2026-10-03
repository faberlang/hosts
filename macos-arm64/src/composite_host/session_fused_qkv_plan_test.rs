//! Companion tests for [`super::session`] (repo companion-test convention).

use super::*;

fn slot(
    buffer_id: u32,
    name: &str,
    role: DeviceBufferRole,
    binding: u32,
    element_count: u64,
) -> SessionSlot {
    SessionSlot {
        buffer_id,
        version: 1,
        buffer_name: name.to_owned(),
        role,
        binding,
        element_ty: DeviceDataType::F32,
        element_count,
    }
}

fn norm_meta(name: &str, element_count: u64) -> (BufferKey, SessionBufferMeta) {
    (
        (9_000, 1),
        SessionBufferMeta {
            name: name.to_owned(),
            semantic_value: 0,
            role: DeviceBufferRole::Input,
            element_ty: DeviceDataType::F32,
            element_count,
            byte_length: element_count * 4,
            lifetime: DeviceBufferLifetime::PerProgram,
            initialization: DeviceBufferInitialization::HostProvided,
        },
    )
}

/// The validated attention axes both model families carry (head_dim 64
/// — the producer's own RoPE row width).
fn axes64() -> Option<FusedAttentionAxes> {
    Some(FusedAttentionAxes { head_dim: 64 })
}

/// Qwen2.5-0.5B prefill layer-0 fused QKV slots, captured from the live
/// descriptor (PB-6 slot dump): packed weights typed as f32 words
/// (byte length / 4), capacity-sized persistent K/V targets, all biases.
#[test]
fn qwen_prefill_cache_targets_finalize_grouped_bind() {
    let slots = vec![
        slot(344, "prefill.blk0.a", DeviceBufferRole::InOut, 0, 15_232),
        slot(
            57,
            "blk.0.attn_q.weight",
            DeviceBufferRole::Input,
            1,
            137_984,
        ),
        slot(3, "prefill.rope.cos", DeviceBufferRole::Input, 2, 544),
        slot(4, "prefill.rope.sin", DeviceBufferRole::Input, 3, 544),
        slot(60, "blk.0.attn_q.bias", DeviceBufferRole::Input, 4, 896),
        slot(6, "kv.invocation_state", DeviceBufferRole::Input, 5, 4),
        slot(
            345,
            "prefill.blk0.q_gemv",
            DeviceBufferRole::InOut,
            6,
            15_232,
        ),
        slot(
            58,
            "blk.0.attn_k.weight",
            DeviceBufferRole::Input,
            7,
            19_712,
        ),
        slot(
            59,
            "blk.0.attn_v.weight",
            DeviceBufferRole::Input,
            8,
            30_464,
        ),
        slot(7, "kv.cache_k.0", DeviceBufferRole::InOut, 9, 1_048_576),
        slot(8, "kv.cache_v.0", DeviceBufferRole::InOut, 10, 1_048_576),
        slot(61, "blk.0.attn_k.bias", DeviceBufferRole::Input, 11, 128),
        slot(62, "blk.0.attn_v.bias", DeviceBufferRole::Input, 12, 128),
    ];
    let mut buffer_meta = BTreeMap::new();
    buffer_meta.insert(
        norm_meta("blk.0.attn_norm.weight", 896).0,
        norm_meta("blk.0.attn_norm.weight", 896).1,
    );
    let plan = build_fused_qkv_plan(
        "prefill_blk_0_QkvProjection",
        &slots,
        &buffer_meta,
        axes64(),
        [238, 1, 1],
    )
    .expect("qwen prefill plan builds against capacity-sized cache targets")
    .expect("entry matched");
    assert_eq!(plan.rows, 17);
    assert_eq!(plan.hidden, 896);
    assert_eq!(plan.q_width, 896);
    assert_eq!(plan.head_dim, 64);
    assert!(plan.kv_cache_target);
    let mut packed = BTreeMap::new();
    packed.insert(57, PackedStorageFormat::Q5_0);
    packed.insert(58, PackedStorageFormat::Q5_0);
    packed.insert(59, PackedStorageFormat::Q8_0);
    let bind = finalize_fused_bind(&plan, &packed).expect("qwen bind finalizes");
    assert_eq!(bind.kv_heads, 2);
    assert_eq!(bind.q_per_kv, 7);
    assert_eq!(bind.kv_output_strides, [8192 * 64, 64, 1]);
    assert!(!bind.rotate_half);
}

/// SmolLM2-360M prefill layer-0 fused QKV slots (captured): no biases,
/// packed Q5_0/Q8_0 weights — the KV width resolves from the uploaded
/// packed byte extents, never from the capacity-sized cache count.
#[test]
fn smol_prefill_packed_cache_targets_resolve_kv_width_from_bytes() {
    let slots = vec![
        slot(360, "prefill.blk0.a", DeviceBufferRole::InOut, 0, 8_640),
        slot(
            73,
            "blk.0.attn_q.weight",
            DeviceBufferRole::Input,
            1,
            158_400,
        ),
        slot(3, "prefill.rope.cos", DeviceBufferRole::Input, 2, 288),
        slot(4, "prefill.rope.sin", DeviceBufferRole::Input, 3, 288),
        slot(6, "kv.invocation_state", DeviceBufferRole::Input, 4, 4),
        slot(
            361,
            "prefill.blk0.q_gemv",
            DeviceBufferRole::InOut,
            5,
            8_640,
        ),
        slot(
            74,
            "blk.0.attn_k.weight",
            DeviceBufferRole::Input,
            6,
            52_800,
        ),
        slot(
            75,
            "blk.0.attn_v.weight",
            DeviceBufferRole::Input,
            7,
            81_600,
        ),
        slot(7, "kv.cache_k.0", DeviceBufferRole::InOut, 8, 2_621_440),
        slot(8, "kv.cache_v.0", DeviceBufferRole::InOut, 9, 2_621_440),
    ];
    let mut buffer_meta = BTreeMap::new();
    buffer_meta.insert(
        norm_meta("blk.0.attn_norm.weight", 960).0,
        norm_meta("blk.0.attn_norm.weight", 960).1,
    );
    let plan = build_fused_qkv_plan(
        "prefill_blk_0_QkvProjection",
        &slots,
        &buffer_meta,
        axes64(),
        [135, 1, 1],
    )
    .expect("smol prefill plan builds without biases")
    .expect("entry matched");
    assert_eq!(plan.rows, 9);
    assert_eq!(plan.head_dim, 64);
    let mut packed = BTreeMap::new();
    packed.insert(73, PackedStorageFormat::Q5_0);
    packed.insert(74, PackedStorageFormat::Q5_0);
    packed.insert(75, PackedStorageFormat::Q8_0);
    let bind = finalize_fused_bind(&plan, &packed).expect("smol bind finalizes from bytes");
    assert_eq!(bind.kv_heads, 5);
    assert_eq!(bind.q_per_kv, 3);
    assert_eq!(bind.kv_output_strides, [8192 * 64, 64, 1]);
}

/// SmolLM2 repeating decode session (captured): rows 10 divides the
/// capacity-sized cache count evenly — the cache target must never be
/// mistaken for a rows-sized `.k_gemv` output.
#[test]
fn smol_decode_cache_count_divisible_by_rows_still_resolves_from_bytes() {
    let slots = vec![
        slot(360, "prefill.blk0.a", DeviceBufferRole::InOut, 0, 9_600),
        slot(
            73,
            "blk.0.attn_q.weight",
            DeviceBufferRole::Input,
            1,
            158_400,
        ),
        slot(3, "prefill.rope.cos", DeviceBufferRole::Input, 2, 320),
        slot(4, "prefill.rope.sin", DeviceBufferRole::Input, 3, 320),
        slot(6, "kv.invocation_state", DeviceBufferRole::Input, 4, 4),
        slot(
            361,
            "prefill.blk0.q_gemv",
            DeviceBufferRole::InOut,
            5,
            9_600,
        ),
        slot(
            74,
            "blk.0.attn_k.weight",
            DeviceBufferRole::Input,
            6,
            52_800,
        ),
        slot(
            75,
            "blk.0.attn_v.weight",
            DeviceBufferRole::Input,
            7,
            81_600,
        ),
        slot(7, "kv.cache_k.0", DeviceBufferRole::InOut, 8, 2_621_440),
        slot(8, "kv.cache_v.0", DeviceBufferRole::InOut, 9, 2_621_440),
    ];
    let mut buffer_meta = BTreeMap::new();
    buffer_meta.insert(
        norm_meta("blk.0.attn_norm.weight", 960).0,
        norm_meta("blk.0.attn_norm.weight", 960).1,
    );
    let plan = build_fused_qkv_plan(
        "prefill_blk_0_QkvProjection",
        &slots,
        &buffer_meta,
        axes64(),
        [150, 1, 1],
    )
    .expect("smol decode plan builds")
    .expect("entry matched");
    assert_eq!(plan.rows, 10);
    let mut packed = BTreeMap::new();
    packed.insert(73, PackedStorageFormat::Q5_0);
    packed.insert(74, PackedStorageFormat::Q5_0);
    packed.insert(75, PackedStorageFormat::Q8_0);
    let bind = finalize_fused_bind(&plan, &packed).expect("smol decode bind finalizes");
    assert_eq!(bind.kv_heads, 5);
    assert_eq!(bind.q_per_kv, 3);
}

/// A fused plan with no resolvable KV width (packed cache target, no
/// bias, no uploaded format fact) fails closed instead of guessing.
#[test]
fn unresolvable_kv_width_fails_closed() {
    let slots = vec![
        slot(360, "prefill.blk0.a", DeviceBufferRole::InOut, 0, 8_640),
        slot(
            73,
            "blk.0.attn_q.weight",
            DeviceBufferRole::Input,
            1,
            158_400,
        ),
        slot(3, "prefill.rope.cos", DeviceBufferRole::Input, 2, 288),
        slot(4, "prefill.rope.sin", DeviceBufferRole::Input, 3, 288),
        slot(
            361,
            "prefill.blk0.q_gemv",
            DeviceBufferRole::InOut,
            5,
            8_640,
        ),
        slot(
            74,
            "blk.0.attn_k.weight",
            DeviceBufferRole::Input,
            6,
            52_800,
        ),
        slot(
            75,
            "blk.0.attn_v.weight",
            DeviceBufferRole::Input,
            7,
            81_600,
        ),
        slot(7, "kv.cache_k.0", DeviceBufferRole::InOut, 8, 2_621_440),
        slot(8, "kv.cache_v.0", DeviceBufferRole::InOut, 9, 2_621_440),
    ];
    let mut buffer_meta = BTreeMap::new();
    buffer_meta.insert(
        norm_meta("blk.0.attn_norm.weight", 960).0,
        norm_meta("blk.0.attn_norm.weight", 960).1,
    );
    let plan = build_fused_qkv_plan(
        "prefill_blk_0_QkvProjection",
        &slots,
        &buffer_meta,
        axes64(),
        [135, 1, 1],
    )
    .expect("plan builds; resolution is a dispatch-time fact")
    .expect("entry matched");
    assert!(finalize_fused_bind(&plan, &BTreeMap::new()).is_err());
}

/// FQ-1 (pb4 findings P1): the actual frozen M=1 decode descriptor
/// shape — one activation row, capacity-sized RoPE tables
/// (8192 * 64/2 = 262144 elements), persistent K/V cache targets — for
/// BOTH model Q widths (SmolLM2 960, Qwen2.5 896). Recognition must
/// resolve rows == 1 and head_dim == 64. Dividing the RoPE table's
/// total element count by activation rows invents 524288, fails
/// divisibility, and silently degrades to the Q-only carrier, which
/// never appends the current K/V row.
#[test]
fn m1_decode_capacity_rope_tables_resolve_rows_one_head_dim_64() {
    let smol_slots = vec![
        slot(360, "prefill.blk0.a", DeviceBufferRole::InOut, 0, 960),
        slot(
            73,
            "blk.0.attn_q.weight",
            DeviceBufferRole::Input,
            1,
            158_400,
        ),
        slot(3, "prefill.rope.cos", DeviceBufferRole::Input, 2, 262_144),
        slot(4, "prefill.rope.sin", DeviceBufferRole::Input, 3, 262_144),
        slot(6, "kv.invocation_state", DeviceBufferRole::Input, 4, 4),
        slot(361, "prefill.blk0.q_gemv", DeviceBufferRole::InOut, 5, 960),
        slot(
            74,
            "blk.0.attn_k.weight",
            DeviceBufferRole::Input,
            6,
            52_800,
        ),
        slot(
            75,
            "blk.0.attn_v.weight",
            DeviceBufferRole::Input,
            7,
            81_600,
        ),
        slot(7, "kv.cache_k.0", DeviceBufferRole::InOut, 8, 2_621_440),
        slot(8, "kv.cache_v.0", DeviceBufferRole::InOut, 9, 2_621_440),
    ];
    let mut smol_meta = BTreeMap::new();
    let smol_norm = norm_meta("blk.0.attn_norm.weight", 960);
    smol_meta.insert(smol_norm.0, smol_norm.1);
    let smol = build_fused_qkv_plan(
        "prefill_blk_0_QkvProjection",
        &smol_slots,
        &smol_meta,
        axes64(),
        [150, 1, 1],
    )
    .expect("SmolLM2 M=1 decode plan must resolve, never fall back to the Q-only carrier")
    .expect("entry matched");
    assert_eq!(smol.rows, 1, "one activation row is the M=1 decode row");
    assert_eq!(smol.hidden, 960);
    assert_eq!(smol.q_width, 960);
    assert_eq!(
        smol.head_dim, 64,
        "head_dim comes from the model, not the capacity table"
    );
    assert!(smol.kv_cache_target);
    let mut smol_packed = BTreeMap::new();
    smol_packed.insert(73, PackedStorageFormat::Q5_0);
    smol_packed.insert(74, PackedStorageFormat::Q5_0);
    smol_packed.insert(75, PackedStorageFormat::Q8_0);
    let smol_bind = finalize_fused_bind(&smol, &smol_packed).expect("SmolLM2 M=1 bind finalizes");
    assert_eq!(smol_bind.kv_heads, 5);
    assert_eq!(smol_bind.q_per_kv, 3);
    assert_eq!(smol_bind.kv_output_strides, [8192 * 64, 64, 1]);

    let qwen_slots = vec![
        slot(344, "prefill.blk0.a", DeviceBufferRole::InOut, 0, 896),
        slot(
            57,
            "blk.0.attn_q.weight",
            DeviceBufferRole::Input,
            1,
            137_984,
        ),
        slot(3, "prefill.rope.cos", DeviceBufferRole::Input, 2, 262_144),
        slot(4, "prefill.rope.sin", DeviceBufferRole::Input, 3, 262_144),
        slot(60, "blk.0.attn_q.bias", DeviceBufferRole::Input, 4, 896),
        slot(6, "kv.invocation_state", DeviceBufferRole::Input, 5, 4),
        slot(345, "prefill.blk0.q_gemv", DeviceBufferRole::InOut, 6, 896),
        slot(
            58,
            "blk.0.attn_k.weight",
            DeviceBufferRole::Input,
            7,
            19_712,
        ),
        slot(
            59,
            "blk.0.attn_v.weight",
            DeviceBufferRole::Input,
            8,
            30_464,
        ),
        slot(7, "kv.cache_k.0", DeviceBufferRole::InOut, 9, 1_048_576),
        slot(8, "kv.cache_v.0", DeviceBufferRole::InOut, 10, 1_048_576),
        slot(61, "blk.0.attn_k.bias", DeviceBufferRole::Input, 11, 128),
        slot(62, "blk.0.attn_v.bias", DeviceBufferRole::Input, 12, 128),
    ];
    let mut qwen_meta = BTreeMap::new();
    let qwen_norm = norm_meta("blk.0.attn_norm.weight", 896);
    qwen_meta.insert(qwen_norm.0, qwen_norm.1);
    let qwen = build_fused_qkv_plan(
        "prefill_blk_0_QkvProjection",
        &qwen_slots,
        &qwen_meta,
        axes64(),
        [238, 1, 1],
    )
    .expect("Qwen2.5 M=1 decode plan must resolve, never fall back to the Q-only carrier")
    .expect("entry matched");
    assert_eq!(qwen.rows, 1);
    assert_eq!(qwen.hidden, 896);
    assert_eq!(qwen.q_width, 896);
    assert_eq!(qwen.head_dim, 64);
    assert!(qwen.kv_cache_target);
    let mut qwen_packed = BTreeMap::new();
    qwen_packed.insert(57, PackedStorageFormat::Q5_0);
    qwen_packed.insert(58, PackedStorageFormat::Q5_0);
    qwen_packed.insert(59, PackedStorageFormat::Q8_0);
    let qwen_bind = finalize_fused_bind(&qwen, &qwen_packed).expect("Qwen2.5 M=1 bind finalizes");
    assert_eq!(qwen_bind.kv_heads, 2);
    assert_eq!(qwen_bind.q_per_kv, 7);
    assert_eq!(qwen_bind.kv_output_strides, [8192 * 64, 64, 1]);
}

/// A `_QkvProjection` entry with no authoritative attention axes is a
/// recognition ERROR, never a silent Q-only carrier fallback (FQ-1
/// fail-closed).
#[test]
fn qkv_entry_without_axes_fails_closed() {
    let slots = vec![
        slot(360, "prefill.blk0.a", DeviceBufferRole::InOut, 0, 960),
        slot(
            73,
            "blk.0.attn_q.weight",
            DeviceBufferRole::Input,
            1,
            158_400,
        ),
        slot(3, "prefill.rope.cos", DeviceBufferRole::Input, 2, 262_144),
        slot(4, "prefill.rope.sin", DeviceBufferRole::Input, 3, 262_144),
        slot(361, "prefill.blk0.q_gemv", DeviceBufferRole::InOut, 5, 960),
        slot(
            74,
            "blk.0.attn_k.weight",
            DeviceBufferRole::Input,
            6,
            52_800,
        ),
        slot(
            75,
            "blk.0.attn_v.weight",
            DeviceBufferRole::Input,
            7,
            81_600,
        ),
        slot(7, "kv.cache_k.0", DeviceBufferRole::InOut, 8, 2_621_440),
        slot(8, "kv.cache_v.0", DeviceBufferRole::InOut, 9, 2_621_440),
    ];
    let mut buffer_meta = BTreeMap::new();
    let norm = norm_meta("blk.0.attn_norm.weight", 960);
    buffer_meta.insert(norm.0, norm.1);
    let error = build_fused_qkv_plan(
        "prefill_blk_0_QkvProjection",
        &slots,
        &buffer_meta,
        None,
        [150, 1, 1],
    )
    .err()
    .expect("a QkvProjection entry without carried axes must fail the session");
    assert!(
        error.message.contains("attention axes"),
        "error names the missing fact: {}",
        error.message
    );
}

/// A carried head_dim that contradicts the descriptor facts (q_width is
/// not a whole multiple of it) is a recognition error, not a guess.
#[test]
fn qkv_entry_with_contradicting_axes_fails_closed() {
    let slots = vec![
        slot(360, "prefill.blk0.a", DeviceBufferRole::InOut, 0, 960),
        slot(
            73,
            "blk.0.attn_q.weight",
            DeviceBufferRole::Input,
            1,
            158_400,
        ),
        slot(361, "prefill.blk0.q_gemv", DeviceBufferRole::InOut, 5, 960),
        slot(
            74,
            "blk.0.attn_k.weight",
            DeviceBufferRole::Input,
            6,
            52_800,
        ),
        slot(
            75,
            "blk.0.attn_v.weight",
            DeviceBufferRole::Input,
            7,
            81_600,
        ),
        slot(7, "kv.cache_k.0", DeviceBufferRole::InOut, 8, 2_621_440),
        slot(8, "kv.cache_v.0", DeviceBufferRole::InOut, 9, 2_621_440),
    ];
    let mut buffer_meta = BTreeMap::new();
    let norm = norm_meta("blk.0.attn_norm.weight", 960);
    buffer_meta.insert(norm.0, norm.1);
    let error = build_fused_qkv_plan(
        "prefill_blk_0_QkvProjection",
        &slots,
        &buffer_meta,
        Some(FusedAttentionAxes { head_dim: 128 }),
        [150, 1, 1],
    )
    .err()
    .expect("a head_dim that does not divide q_width must fail the session");
    assert!(
        error.message.contains("q width grouping"),
        "error names the contradicted fact: {}",
        error.message
    );
}

/// The cursor position is the first f32 VALUE of the descriptor-declared
/// cursor input, never its raw word reinterpreted as u32. f32 1.0 read
/// as u32 is 0x3F800000 = 1_065_353_216, which multiplied the k=3
/// append offset into a ~272 GB view end against a 10 MiB K arena
/// (SV-E5 binding-8 defect; same class as the PB-4d word/extent
/// confusion).
#[test]
fn cursor_position_decodes_f32_value_not_raw_word() {
    let k3 = [1.0_f32, 4.0, 3.0, 1.0];
    assert_eq!(
        fused_cursor_position_from_words(&k3).expect("k=3 cursor decodes"),
        1,
        "f32 1.0 must decode to position 1, not 0x3F800000"
    );
    assert_eq!(
        fused_cursor_position_from_words(&[0.0, 9.0, 9.0, 0.0]).expect("prefill cursor"),
        0
    );
    assert_eq!(
        fused_cursor_position_from_words(&[8192.0, 8192.0, 1.0, 1.0]).expect("deep position"),
        8192
    );
}

/// Cursor positions that are not one non-negative integer-valued f32
/// fail closed instead of truncating into an append offset.
#[test]
fn cursor_position_fails_closed_on_non_integer_words() {
    for bad in [
        Vec::new(),
        vec![-1.0_f32, 0.0, 1.0, 0.0],
        vec![1.5_f32, 4.0, 3.0, 1.0],
        vec![f32::NAN, 4.0, 3.0, 1.0],
        vec![f32::INFINITY, 4.0, 3.0, 1.0],
        vec![-0.5_f32, 4.0, 3.0, 1.0],
        vec![f32::MAX, 0.0, 0.0, 0.0],
    ] {
        assert!(
            fused_cursor_position_from_words(&bad).is_err(),
            "cursor {bad:?} must fail closed"
        );
    }
}
