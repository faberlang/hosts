//! PPE-P2: pure cursor-to-binding projection tests.

use faber_host_macos_arm64::composite_host::invocation_binding::{
    project_invocation_bindings, RopeConfig, KV_PREFIX_IDS, PROMPT_TOKENS, Q_PREFIX_IDS, ROPE_COS,
    ROPE_SIN,
};
use faber_host_macos_arm64::device_descriptor::{
    DescriptorBuffer, DescriptorBufferVersion, DescriptorKernel, DeviceBufferInitialization,
    DeviceBufferLifetime, DeviceBufferRole, DeviceDataType, DeviceDescriptor,
    DeviceProgramLifetime,
};
use faber_host_macos_arm64::device_execute::{
    DeviceExecuteInvocation, DeviceExecuteInvocationMode,
};
use host_coordinator::DeviceBackend;

const ROPE: RopeConfig = RopeConfig {
    head_dim: 8,
    theta: 10_000.0,
};

fn input(id: u32, name: &str, element_count: u64) -> DescriptorBuffer {
    DescriptorBuffer {
        buffer_id: id,
        buffer_name: name.to_owned(),
        semantic_value: id,
        role: DeviceBufferRole::Input,
        lifetime: DeviceBufferLifetime::PerStep,
        initialization: DeviceBufferInitialization::HostProvided,
        binding: id,
        element_ty: DeviceDataType::F32,
        element_count,
        version: 1,
    }
}

fn descriptor(prompt_count: u64, rope_count: u64, prefix_count: Option<u64>) -> DeviceDescriptor {
    let mut buffers = vec![
        input(1, PROMPT_TOKENS, prompt_count),
        input(2, ROPE_COS, rope_count),
        input(3, ROPE_SIN, rope_count),
        // A model weight is a declared input but not an invocation binding.
        input(9, "token_embd.weight", 32),
    ];
    if let Some(prefix_count) = prefix_count {
        buffers.push(input(4, Q_PREFIX_IDS, prefix_count));
        buffers.push(input(5, KV_PREFIX_IDS, prefix_count));
    }
    DeviceDescriptor {
        backend: DeviceBackend::Metal,
        module_image: b"module".to_vec(),
        kernels: vec![DescriptorKernel {
            entry: "dense".to_owned(),
            buffers,
            grid: [1, 1, 1],
            block: [1, 1, 1],
        }],
        launches: vec![],
        buffer_versions: vec![DescriptorBufferVersion {
            buffer_id: 1,
            version: 1,
            element_ty: DeviceDataType::F32,
            element_count: prompt_count,
        }],
        program_lifetime: DeviceProgramLifetime::RepeatingStep,
        data_flow: vec![],
        roots: vec![],
        results: vec![],
        end_of_run_results: vec![],
    }
}

fn decode_invocation(
    token: Option<u32>,
    position: u32,
    prefix_before: u32,
    valid_len_after: u32,
) -> DeviceExecuteInvocation {
    DeviceExecuteInvocation {
        mode: DeviceExecuteInvocationMode::ScalarDecode,
        token,
        position,
        sequence_epoch: 7,
        prefix_before,
        valid_len_after,
        query_start: position,
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn decode_projects_only_declared_per_invocation_buffers() {
    let descriptor = descriptor(1, 4, Some(4));
    let invocation = decode_invocation(Some(42), 3, 3, 4);
    let projected =
        project_invocation_bindings(&descriptor, &invocation, &[100, 101, 102, 103], ROPE)
            .expect("decode projection");

    assert_eq!(
        projected.keys().copied().collect::<Vec<_>>(),
        vec![1, 2, 3, 4, 5]
    );
    assert_eq!(bits(projected.get(&1).expect("token")), vec![42]);
    assert_eq!(
        bits(projected.get(&4).expect("q prefix ids")),
        vec![0, 1, 2, 3]
    );
    assert_eq!(
        bits(projected.get(&5).expect("kv prefix ids")),
        vec![0, 1, 2, 3]
    );
    assert_eq!(
        projected.get(&2).expect("cos").len(),
        ROPE.head_dim as usize / 2
    );
    assert_eq!(
        projected.get(&3).expect("sin").len(),
        ROPE.head_dim as usize / 2
    );
    assert!(
        !projected.contains_key(&9),
        "resident model weights are not per-step inputs"
    );
    assert_eq!(projected.get(&1).expect("decode token").len(), 1);
}

#[test]
fn decode_prefix_ids_follow_declared_gather_width_not_sequence_len() {
    // Dense M=1 gather tables are the logical Q/K/V span (960), not valid_len.
    let descriptor = descriptor(1, 4, Some(960));
    let invocation = decode_invocation(Some(42), 9, 9, 10);
    let projected = project_invocation_bindings(&descriptor, &invocation, &[1, 2], ROPE)
        .expect("gather-width decode projection");
    let q = bits(projected.get(&4).expect("q prefix ids"));
    let kv = bits(projected.get(&5).expect("kv prefix ids"));
    assert_eq!(q.len(), 960);
    assert_eq!(kv.len(), 960);
    assert_eq!(q[0], 0);
    assert_eq!(q[959], 959);
    assert_eq!(q, kv);
}

#[test]
fn decode_rope_uses_absolute_position_not_step_count() {
    let descriptor = descriptor(1, 4, Some(18));
    let invocation = decode_invocation(Some(42), 17, 17, 18);
    let projected = project_invocation_bindings(&descriptor, &invocation, &[1, 2, 3], ROPE)
        .expect("position-gap decode projection");
    let cos = projected.get(&2).expect("cos row");
    let sin = projected.get(&3).expect("sin row");

    for (pair, (&actual_cos, &actual_sin)) in cos.iter().zip(sin).enumerate() {
        let angle = 17.0_f64
            * ROPE
                .theta
                .powf(-(2.0 * pair as f64) / f64::from(ROPE.head_dim));
        assert_eq!(actual_cos.to_bits(), (angle.cos() as f32).to_bits());
        assert_eq!(actual_sin.to_bits(), (angle.sin() as f32).to_bits());
    }
    assert_eq!(cos.len(), 4, "one RoPE row, not the full prompt table");
}

#[test]
fn prefill_projects_prompt_and_full_prompt_rope_once() {
    let descriptor = descriptor(3, 12, None);
    let invocation = DeviceExecuteInvocation {
        mode: DeviceExecuteInvocationMode::Prefill,
        token: None,
        position: 0,
        sequence_epoch: 7,
        prefix_before: 0,
        valid_len_after: 3,
        query_start: 0,
    };
    let projected = project_invocation_bindings(&descriptor, &invocation, &[10, 11, 12], ROPE)
        .expect("prefill projection");

    assert_eq!(projected.keys().copied().collect::<Vec<_>>(), vec![1, 2, 3]);
    assert_eq!(bits(projected.get(&1).expect("prompt")), vec![10, 11, 12]);
    assert_eq!(projected.get(&2).expect("cos").len(), 12);
    assert_eq!(projected.get(&3).expect("sin").len(), 12);
}

#[test]
fn malformed_cursor_facts_fail_before_projection() {
    let descriptor = descriptor(1, 4, Some(4));

    let overflow = decode_invocation(Some(9), u32::MAX, u32::MAX, 0);
    let error = project_invocation_bindings(&descriptor, &overflow, &[], ROPE)
        .expect_err("overflow must fail before any binding is built");
    assert_eq!(error.code, "E_INVALID_ARGS");

    let missing_token = decode_invocation(None, 3, 3, 4);
    let error = project_invocation_bindings(&descriptor, &missing_token, &[], ROPE)
        .expect_err("scalar decode must have one token row");
    assert_eq!(error.code, "E_INVALID_ARGS");
    assert!(
        error.message.contains("token row"),
        "missing scalar token must fail closed before projection, got {}",
        error.message
    );

    let mismatched_after = decode_invocation(Some(9), 3, 3, 9);
    let error = project_invocation_bindings(&descriptor, &mismatched_after, &[], ROPE)
        .expect_err("valid length must match prefix plus query rows");
    assert_eq!(error.code, "E_INVALID_ARGS");
}
