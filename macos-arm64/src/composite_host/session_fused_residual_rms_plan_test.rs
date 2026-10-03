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

/// Qwen2.5-0.5B prefill layer-0 fused residual/RMS slots: the residual
/// is the layer-entry activation, the skip is the attention output
/// projection, and the epsilon (1e-6 on this model) is baked into the
/// carrier kernel while the neighboring attn_norm uses 1e-5 — the parse
/// must bind the ResidualRmsNorm body's own literal.
fn qwen_module_image() -> Vec<u8> {
    concat!(
        "kernel void prefill_blk_0_attn_norm(float) {\n",
        "    float scale = 1.0 / sqrt(mean + 0.00001f);\n",
        "}\n",
        "kernel void prefill_blk_0_ResidualRmsNorm(float) {\n",
        "    float mean = sumsq / float(896u);\n",
        "    float scale = 1.0 / sqrt(mean + 0.000001f);\n",
        "}\n",
        "kernel void prefill_blk_0_ffn_gate(float) {\n",
        "    return;\n",
        "}\n",
    )
    .to_string()
    .into_bytes()
}

#[test]
fn qwen_prefill_plan_binds_both_streams_and_carrier_epsilon() {
    let slots = vec![
        slot(400, "prefill.h", DeviceBufferRole::InOut, 0, 15_232),
        slot(41, "blk.0.ffn_norm.weight", DeviceBufferRole::Input, 1, 896),
        slot(401, "prefill.blk0.f", DeviceBufferRole::InOut, 2, 15_232),
        slot(402, "prefill.blk0.o", DeviceBufferRole::InOut, 3, 15_232),
    ];
    let plan = build_fused_residual_rms_plan(
        "prefill_blk_0_ResidualRmsNorm",
        &slots,
        &qwen_module_image(),
    )
    .expect("qwen prefill residual/RMS plan builds")
    .expect("entry matched");
    assert_eq!(plan.rows, 17);
    assert_eq!(plan.hidden, 896);
    assert_eq!(plan.epsilon, 1.0e-6f32);
    assert_eq!(plan.residual.key, (400, 1));
    assert_eq!(plan.skip.key, (402, 1));
    assert_eq!(plan.gamma.key, (41, 1));
    assert_eq!(plan.output.key, (401, 1));
}

#[test]
fn non_residual_entry_admits_no_plan() {
    let plan = build_fused_residual_rms_plan(
        "prefill_blk_0_attn_norm",
        &[slot(1, "prefill.h", DeviceBufferRole::InOut, 0, 8)],
        &qwen_module_image(),
    )
    .expect("non-matching entry is not an error");
    assert!(plan.is_none());
}

#[test]
fn missing_skip_slot_fails_closed() {
    let slots = vec![
        slot(400, "prefill.h", DeviceBufferRole::InOut, 0, 15_232),
        slot(41, "blk.0.ffn_norm.weight", DeviceBufferRole::Input, 1, 896),
        slot(401, "prefill.blk0.f", DeviceBufferRole::InOut, 2, 15_232),
    ];
    let error = build_fused_residual_rms_plan(
        "prefill_blk_0_ResidualRmsNorm",
        &slots,
        &qwen_module_image(),
    )
    .err()
    .expect("a recognized carrier without a skip slot must fail the session");
    assert!(error.message.contains("skip slot"));
}

#[test]
fn epsilon_absent_from_carrier_body_fails_closed() {
    let slots = vec![
        slot(400, "prefill.h", DeviceBufferRole::InOut, 0, 15_232),
        slot(41, "blk.0.ffn_norm.weight", DeviceBufferRole::Input, 1, 896),
        slot(401, "prefill.blk0.f", DeviceBufferRole::InOut, 2, 15_232),
        slot(402, "prefill.blk0.o", DeviceBufferRole::InOut, 3, 15_232),
    ];
    let image = b"kernel void prefill_blk_0_ResidualRmsNorm(float) {\n return;\n}\n";
    let error = build_fused_residual_rms_plan("prefill_blk_0_ResidualRmsNorm", &slots, image)
        .err()
        .expect("an unparseable carrier epsilon must fail the session");
    assert!(error.message.contains("RMS epsilon"));
}

#[test]
fn inconsistent_slot_geometry_fails_closed() {
    let slots = vec![
        slot(400, "prefill.h", DeviceBufferRole::InOut, 0, 15_232),
        slot(41, "blk.0.ffn_norm.weight", DeviceBufferRole::Input, 1, 896),
        slot(401, "prefill.blk0.f", DeviceBufferRole::InOut, 2, 8_960),
        slot(402, "prefill.blk0.o", DeviceBufferRole::InOut, 3, 15_232),
    ];
    let error = build_fused_residual_rms_plan(
        "prefill_blk_0_ResidualRmsNorm",
        &slots,
        &qwen_module_image(),
    )
    .err()
    .expect("an output width disagreeing with rows*hidden must fail the session");
    assert!(error.message.contains("slot geometry"));
}
