//! M8-U2b: SSM family dispatch + hybrid probes through the plan path (Metal).
//!
//! The R-PACK-04 probe goldens (SSM conv/scan prefill + decode arms, the PS5
//! hybrid-layer SSM+GQA mix, and the PS6 reset/replay determinism rows) run
//! green through the compiled-plan dispatch seams, and one synthetic hybrid
//! SSM layer executes `ssm_conv1d` + `ssm_scan` as the real minted Metal
//! module through the prepared resident schedule — prefill and decode arms —
//! with the census rows recorded (two launches, one submission, one declared
//! observation readback per step).

use std::collections::BTreeMap;

use faber_host_macos_arm64::composite_host::{
    CompositeHost, DeviceByteBuffer, PreparedResidentSession,
};
use faber_host_macos_arm64::device_descriptor::{
    DescriptorBuffer, DescriptorBufferVersion, DescriptorDataFlow, DescriptorKernel,
    DescriptorLaunch, DescriptorResult, DeviceBufferInitialization, DeviceBufferLifetime,
    DeviceBufferRole, DeviceDataType, DeviceDescriptor, DeviceProgramLifetime,
};
use faber_host_macos_arm64::device_host::DeviceRuntime;
use faber_host_macos_arm64::kernel::library::{dispatch, CausalAttentionBind, LibraryKernel};
use faber_host_macos_arm64::kernel::ssm::{
    ssm_family_dispatch, ssm_family_msl, SsmFamilyDispatch, SsmFamilyMslFacts,
};
use faber_host_macos_arm64::kernel::ssm_conv1d::SsmConv1dBind;
use faber_host_macos_arm64::kernel::ssm_scan::SsmScanBind;
use faber_host_macos_arm64::MetalHostSession;
use host_coordinator::DeviceBackend;

// ---------------------------------------------------------------------------
// PS5-class hybrid-layer fixture (qwen35moe MODEL-01 shape family).
// ---------------------------------------------------------------------------

const PREFILL_TOKENS: usize = 3;
const HEAD_COUNT: usize = 16;
const KV_HEAD_COUNT: usize = 2;
const Q_PER_KV: usize = HEAD_COUNT / KV_HEAD_COUNT;
const HEAD_DIM: usize = 256;
const SSM_STATE_SIZE: usize = 128;
const SSM_CONV_KERNEL: usize = 4;

fn synthetic_ssm_input(length: usize) -> Vec<f32> {
    (0..length * SSM_STATE_SIZE)
        .map(|index| {
            let token = index / SSM_STATE_SIZE;
            let channel = index % SSM_STATE_SIZE;
            0.01 + token as f32 * 0.002 + channel as f32 * 0.0001
        })
        .collect()
}

fn cpu_conv1d(input: &[f32], length: usize, channels: usize, kernel: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for time in 0..length {
        for channel in 0..channels {
            let mut value = 0.0;
            for offset in 0..kernel.len().min(time + 1) {
                value += input[(time - offset) * channels + channel] * kernel[offset];
            }
            output[time * channels + channel] = value;
        }
    }
    output
}

fn cpu_scan(input: &[f32], length: usize, state_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for state in 0..state_dim {
        let mut carry = 0.0;
        for time in 0..length {
            carry += input[time * state_dim + state];
            output[time * state_dim + state] = carry;
        }
    }
    output
}

fn fill_attention_q(sequence: usize, value_offset: f32) -> Vec<f32> {
    let row = Q_PER_KV * sequence * HEAD_DIM;
    (0..KV_HEAD_COUNT * row)
        .map(|index| {
            let group = index / row;
            let within_group = index % row;
            let head = within_group / (sequence * HEAD_DIM);
            let within_head = within_group % (sequence * HEAD_DIM);
            let token = within_head / HEAD_DIM;
            let dimension = within_head % HEAD_DIM;
            value_offset
                + group as f32 * 0.003
                + head as f32 * 0.0002
                + token as f32 * 0.001
                + dimension as f32 * 0.00001
        })
        .collect()
}

fn fill_attention_kv(sequence: usize, value_offset: f32) -> Vec<f32> {
    (0..KV_HEAD_COUNT * sequence * HEAD_DIM)
        .map(|index| {
            let group = index / (sequence * HEAD_DIM);
            let within_group = index % (sequence * HEAD_DIM);
            let token = within_group / HEAD_DIM;
            let dimension = within_group % HEAD_DIM;
            value_offset
                + group as f32 * 0.003
                + token as f32 * 0.001
                + dimension as f32 * 0.00001
        })
        .collect()
}

fn cpu_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_block: usize,
    query_seq: usize,
) -> Vec<f32> {
    let q_row = Q_PER_KV * query_seq * HEAD_DIM;
    let q_head = query_seq * HEAD_DIM;
    let k_row = seq_block * HEAD_DIM;
    let query_start = seq_block - query_seq;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut output = vec![0.0; KV_HEAD_COUNT * q_row];

    for group in 0..KV_HEAD_COUNT {
        for q_head_index in 0..Q_PER_KV {
            for query in 0..query_seq {
                let query_position = query_start + query;
                let visible = query_position + 1;
                let q_base = group * q_row + q_head_index * q_head + query * HEAD_DIM;
                let k_base = group * k_row;
                let mut scores = Vec::with_capacity(visible);
                for token in 0..visible {
                    let mut dot = 0.0;
                    for dimension in 0..HEAD_DIM {
                        dot += q[q_base + dimension] * k[k_base + token * HEAD_DIM + dimension];
                    }
                    scores.push(dot * scale);
                }
                let row_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights: Vec<f32> = scores
                    .iter()
                    .map(|score| (*score - row_max).exp())
                    .collect();
                let row_sum: f32 = weights.iter().sum();
                let output_base = q_base;
                for dimension in 0..HEAD_DIM {
                    let value = weights
                        .iter()
                        .enumerate()
                        .map(|(token, weight)| {
                            *weight * v[k_base + token * HEAD_DIM + dimension]
                        })
                        .sum::<f32>()
                        / row_sum;
                    output[output_base + dimension] = value;
                }
            }
        }
    }
    output
}

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    let max_abs_delta = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs_delta <= 2.0e-5,
        "{label} max_abs_delta={max_abs_delta}"
    );
    assert!(
        actual.iter().all(|value| value.is_finite()),
        "{label} contains a non-finite value"
    );
}

fn byte_identity(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// One PS5-class hybrid layer driven through the plan dispatch seams: the
/// SSM conv/scan families through [`ssm_family_dispatch`] and the GQA mix
/// through the landed attention library dispatch.  Returns the per-layer
/// row bytes (conv, state, decode state, attention prefill/decode) for the
/// reset/replay determinism proof.
fn run_hybrid_probe_layer() -> Vec<Vec<u8>> {
    let conv_kernel = [0.25f32, -0.1, 0.05, 0.02];

    // SSM prefill arm: conv then scan through the plan seams.
    let prefill_input = synthetic_ssm_input(PREFILL_TOKENS);
    let expected_conv_prefill =
        cpu_conv1d(&prefill_input, PREFILL_TOKENS, SSM_STATE_SIZE, &conv_kernel);
    let conv_prefill_bind = SsmConv1dBind::channels_last(
        PREFILL_TOKENS as u64,
        SSM_STATE_SIZE as u64,
        SSM_CONV_KERNEL as u64,
        [SSM_STATE_SIZE as u32, PREFILL_TOKENS as u32, 1],
    );
    let mut conv_prefill = vec![0.0; expected_conv_prefill.len()];
    ssm_family_dispatch(SsmFamilyDispatch::SsmConv1d {
        library_entry: Some("SsmConv1d"),
        bind: &conv_prefill_bind,
        input: &prefill_input,
        kernel: &conv_kernel,
        output: &mut conv_prefill,
    })
    .expect("plan-path SSM prefill convolution");
    assert_close(&conv_prefill, &expected_conv_prefill, "SSM prefill convolution");

    let expected_state_prefill = cpu_scan(&expected_conv_prefill, PREFILL_TOKENS, SSM_STATE_SIZE);
    let scan_prefill_bind = SsmScanBind::prefill(
        PREFILL_TOKENS as u64,
        SSM_STATE_SIZE as u64,
        [SSM_STATE_SIZE as u32, PREFILL_TOKENS as u32, 1],
    );
    let mut state_prefill = vec![0.0; expected_state_prefill.len()];
    ssm_family_dispatch(SsmFamilyDispatch::SsmScan {
        library_entry: Some("SsmScan"),
        bind: &scan_prefill_bind,
        input: &conv_prefill,
        output: &mut state_prefill,
    })
    .expect("plan-path SSM prefill scan");
    assert_close(&state_prefill, &expected_state_prefill, "SSM prefill state");

    // SSM decode arm: length-one conv + carry-in state update through the
    // plan seams (the decode regime row).
    let decode_input = synthetic_ssm_input(1);
    let expected_conv_decode = cpu_conv1d(&decode_input, 1, SSM_STATE_SIZE, &conv_kernel);
    let conv_decode_bind = SsmConv1dBind::channels_last(1, SSM_STATE_SIZE as u64, SSM_CONV_KERNEL as u64, [SSM_STATE_SIZE as u32, 1, 1]);
    let mut conv_decode = vec![0.0; expected_conv_decode.len()];
    ssm_family_dispatch(SsmFamilyDispatch::SsmConv1d {
        library_entry: Some("SsmConv1d"),
        bind: &conv_decode_bind,
        input: &decode_input,
        kernel: &conv_kernel,
        output: &mut conv_decode,
    })
    .expect("plan-path SSM decode convolution");
    assert_close(&conv_decode, &expected_conv_decode, "SSM decode convolution");

    let previous_state = &state_prefill[(PREFILL_TOKENS - 1) * SSM_STATE_SIZE..];
    let decode_state_input: Vec<f32> = previous_state
        .iter()
        .zip(&conv_decode)
        .map(|(previous, update)| previous + update)
        .collect();
    let scan_decode_bind = SsmScanBind::decode(SSM_STATE_SIZE as u64, [SSM_STATE_SIZE as u32, 1, 1]);
    let mut state_decode = vec![0.0; SSM_STATE_SIZE];
    ssm_family_dispatch(SsmFamilyDispatch::SsmScan {
        library_entry: Some("SsmScan"),
        bind: &scan_decode_bind,
        input: &decode_state_input,
        output: &mut state_decode,
    })
    .expect("plan-path SSM decode scan");
    assert_close(&state_decode, &decode_state_input, "SSM decode state");

    // GQA mix arm (the PS5 hybrid attention family) through the landed
    // library dispatch, both regimes.
    let prefill_q = fill_attention_q(PREFILL_TOKENS, 0.02);
    let prefill_k = fill_attention_kv(PREFILL_TOKENS, 0.01);
    let prefill_v = fill_attention_kv(PREFILL_TOKENS, 0.04);
    let prefill_bind = CausalAttentionBind::grouped(
        HEAD_DIM as u64,
        PREFILL_TOKENS as u64,
        Q_PER_KV as u64,
        KV_HEAD_COUNT as u64,
        PREFILL_TOKENS as u64,
        [Q_PER_KV as u32, KV_HEAD_COUNT as u32, PREFILL_TOKENS as u32],
    );
    let mut attention_prefill = vec![0.0; prefill_q.len()];
    dispatch(
        LibraryKernel::CausalAttention,
        &prefill_bind,
        &prefill_q,
        &prefill_k,
        &prefill_v,
        &mut attention_prefill,
    )
    .expect("landed attention prefill body");
    assert_close(
        &attention_prefill,
        &cpu_attention(&prefill_q, &prefill_k, &prefill_v, PREFILL_TOKENS, PREFILL_TOKENS),
        "KV prefill output",
    );

    let decode_q = fill_attention_q(1, 0.025);
    // Decode KV extends the prefill rows by one token per group.
    let mut decode_k = prefill_k.clone();
    let mut decode_v = prefill_v.clone();
    for group in 0..KV_HEAD_COUNT {
        for dimension in 0..HEAD_DIM {
            let token = PREFILL_TOKENS;
            decode_k.push(0.01 + group as f32 * 0.003 + token as f32 * 0.001 + dimension as f32 * 0.00001);
            decode_v.push(0.04 + group as f32 * 0.003 + token as f32 * 0.001 + dimension as f32 * 0.00001);
        }
    }
    let decode_bind = CausalAttentionBind::grouped(
        HEAD_DIM as u64,
        (PREFILL_TOKENS + 1) as u64,
        Q_PER_KV as u64,
        KV_HEAD_COUNT as u64,
        1,
        [Q_PER_KV as u32, KV_HEAD_COUNT as u32, 1],
    );
    let mut attention_decode = vec![0.0; decode_q.len()];
    dispatch(
        LibraryKernel::CausalAttention,
        &decode_bind,
        &decode_q,
        &decode_k,
        &decode_v,
        &mut attention_decode,
    )
    .expect("landed attention decode body");
    assert_close(
        &attention_decode,
        &cpu_attention(&decode_q, &decode_k, &decode_v, PREFILL_TOKENS + 1, 1),
        "KV decode output",
    );

    // Hybrid per-layer state/output rows (recorded for the U2b row b).
    let final_state = &state_prefill[(PREFILL_TOKENS - 1) * SSM_STATE_SIZE..];
    println!(
        "m8-u2b hybrid rows: conv_prefill[last][0..4]={:?} state_prefill[last][0..4]={:?} decode_state[0..4]={:?} attn_prefill[0..2]={:?} attn_decode[0..2]={:?}",
        &conv_prefill[conv_prefill.len() - 4..],
        &final_state[..4],
        &state_decode[..4],
        &attention_prefill[..2],
        &attention_decode[..2],
    );

    vec![
        byte_identity(&conv_prefill),
        byte_identity(&state_prefill),
        byte_identity(&state_decode),
        byte_identity(&attention_prefill),
        byte_identity(&attention_decode),
    ]
}

#[test]
fn ps5_hybrid_probe_goldens_green_through_plan_dispatch() {
    assert_eq!(HEAD_COUNT, KV_HEAD_COUNT * Q_PER_KV);
    let rows = run_hybrid_probe_layer();
    assert_eq!(rows.len(), 5, "per-layer row set");
    assert!(
        rows.iter().all(|row| !row.is_empty()),
        "every per-layer row must produce bytes"
    );
}

#[test]
fn ps6_reset_replay_deterministic_through_plan_dispatch() {
    // Two full passes over the hybrid layer (reset between) are
    // byte-identical on every recorded row.
    let first_pass = run_hybrid_probe_layer();
    let second_pass = run_hybrid_probe_layer();
    for (index, (first, second)) in first_pass.iter().zip(&second_pass).enumerate() {
        assert_eq!(
            first, second,
            "reset/replay row {index} must be byte-identical"
        );
    }
}

#[test]
fn ssm_plan_dispatch_fails_closed_rows() {
    // Wrong library entry fails closed before any buffer access.
    let scan_bind = SsmScanBind::decode(2, [2, 1, 1]);
    let input = [1.0f32, 2.0];
    let mut output = [7.0f32; 2];
    let wrong_entry = ssm_family_dispatch(SsmFamilyDispatch::SsmScan {
        library_entry: Some("CausalAttention"),
        bind: &scan_bind,
        input: &input,
        output: &mut output,
    })
    .expect_err("wrong library entry must fail closed");
    assert!(matches!(
        wrong_entry,
        faber_host_macos_arm64::kernel::library::KernelBodyError::InvalidBind(message)
            if message.contains("disagrees with library_entry")
    ));
    assert_eq!(output, [7.0; 2], "failed selection must not write");

    // Unservable state layout sentinel fails closed.
    let mut bad_conv = SsmConv1dBind::channels_last(2, 2, 2, [4, 1, 1]);
    bad_conv.layout = faber_host_macos_arm64::kernel::ssm_conv1d::SsmConv1dLayout::Unsupported;
    let unservable = ssm_family_dispatch(SsmFamilyDispatch::SsmConv1d {
        library_entry: Some("SsmConv1d"),
        bind: &bad_conv,
        input: &[],
        kernel: &[],
        output: &mut [],
    })
    .expect_err("unsupported layout must fail closed");
    assert!(matches!(
        unservable,
        faber_host_macos_arm64::kernel::library::KernelBodyError::InvalidBind(message)
            if message.contains("not servable")
    ));

    // Regime mismatch fails closed: a length-one launch labeled prefill.
    let bad_regime = SsmScanBind::prefill(1, 4, [4, 1, 1]);
    let regime = ssm_family_dispatch(SsmFamilyDispatch::SsmScan {
        library_entry: Some("SsmScan"),
        bind: &bad_regime,
        input: &[0.0; 4],
        output: &mut [0.0; 4],
    })
    .expect_err("length-one prefill regime must fail closed");
    assert!(matches!(
        regime,
        faber_host_macos_arm64::kernel::library::KernelBodyError::InvalidBind(message)
            if message.contains("prefill regime requires a sequence length greater than one")
    ));
}

// ---------------------------------------------------------------------------
// One synthetic SSM layer plan-path run on the resident schedule (real
// Metal): prefill and decode arms through the minted family module.
// ---------------------------------------------------------------------------

fn slot(
    id: u32,
    name: &str,
    binding: u32,
    role: DeviceBufferRole,
    lifetime: DeviceBufferLifetime,
    initialization: DeviceBufferInitialization,
    dtype: DeviceDataType,
    count: u64,
) -> DescriptorBuffer {
    DescriptorBuffer {
        buffer_id: id,
        buffer_name: name.to_owned(),
        semantic_value: id,
        role,
        lifetime,
        initialization,
        binding,
        element_ty: dtype,
        element_count: count,
        version: 1,
    }
}

fn buffer_versions_for(kernels: &[DescriptorKernel]) -> Vec<DescriptorBufferVersion> {
    let mut versions = Vec::new();
    for kernel in kernels {
        for slot in &kernel.buffers {
            if versions.iter().any(|version: &DescriptorBufferVersion| {
                version.buffer_id == slot.buffer_id && version.version == slot.version
            }) {
                continue;
            }
            versions.push(DescriptorBufferVersion {
                buffer_id: slot.buffer_id,
                version: slot.version,
                element_ty: slot.element_ty,
                element_count: slot.element_count,
            });
        }
    }
    versions
}

fn ssm_layer_descriptor(
    backend: DeviceBackend,
    module: &[u8],
    length: u64,
    state_dim: u64,
    kernel_width: u64,
) -> DeviceDescriptor {
    let state_span = length * state_dim;
    let conv_kernel = DescriptorKernel {
        entry: "ssm_conv1d".to_owned(),
        buffers: vec![
            slot(
                3,
                "state_input",
                0,
                DeviceBufferRole::Input,
                DeviceBufferLifetime::PerStep,
                DeviceBufferInitialization::ZeroFill,
                DeviceDataType::F32,
                state_span,
            ),
            slot(
                1,
                "conv_kernel",
                1,
                DeviceBufferRole::Input,
                DeviceBufferLifetime::PerProgram,
                DeviceBufferInitialization::HostProvided,
                DeviceDataType::F32,
                kernel_width,
            ),
            slot(
                4,
                "conv_state",
                2,
                DeviceBufferRole::InOut,
                DeviceBufferLifetime::PerStep,
                DeviceBufferInitialization::KernelInitialized,
                DeviceDataType::F32,
                state_span,
            ),
        ],
        grid: [state_span as u32, 1, 1],
        block: [64, 1, 1],
    };
    let scan_kernel = DescriptorKernel {
        entry: "ssm_scan".to_owned(),
        buffers: vec![
            slot(
                4,
                "conv_state",
                0,
                DeviceBufferRole::InOut,
                DeviceBufferLifetime::PerStep,
                DeviceBufferInitialization::KernelInitialized,
                DeviceDataType::F32,
                state_span,
            ),
            slot(
                5,
                "scan_state",
                1,
                DeviceBufferRole::Output,
                DeviceBufferLifetime::ObservationPoint,
                DeviceBufferInitialization::KernelInitialized,
                DeviceDataType::F32,
                state_span,
            ),
        ],
        grid: [state_dim as u32, 1, 1],
        block: [64, 1, 1],
    };
    let mut descriptor = DeviceDescriptor {
        backend,
        module_image: module.to_vec(),
        kernels: vec![conv_kernel, scan_kernel],
        launches: vec![
            DescriptorLaunch {
                id: 1,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 2,
                kernel_index: 1,
            },
        ],
        buffer_versions: Vec::new(),
        program_lifetime: DeviceProgramLifetime::RepeatingStep,
        // The conv produces the convolved state; the scan consumes it on the
        // device — the state family stays device-resident inside the step.
        data_flow: vec![DescriptorDataFlow {
            buffer_id: 4,
            version: 1,
            producer: 1,
            consumer: 2,
        }],
        roots: vec![1],
        results: vec![DescriptorResult {
            buffer_id: 5,
            version: 1,
            produced_by: 2,
            at_launch: 2,
        }],
        end_of_run_results: Vec::new(),
    };
    descriptor.buffer_versions = buffer_versions_for(&descriptor.kernels);
    descriptor
}

/// CPU parity oracle for one SSM layer through the plan dispatch seams.
fn ssm_layer_cpu_oracle(length: u64, state_dim: u64, kernel_width: u64) -> Vec<f32> {
    let kernel: Vec<f32> = (0..kernel_width).map(|tap| 0.25 - tap as f32 * 0.05).collect();
    let input: Vec<f32> = (0..length * state_dim)
        .map(|index| {
            let token = index / state_dim;
            let channel = index % state_dim;
            0.01 + token as f32 * 0.002 + channel as f32 * 0.0001
        })
        .collect();
    let bind = if length == 1 {
        SsmConv1dBind::channels_last(1, state_dim, kernel_width, [state_dim as u32, 1, 1])
    } else {
        SsmConv1dBind::channels_last(
            length,
            state_dim,
            kernel_width,
            [state_dim as u32, length as u32, 1],
        )
    };
    let mut convolved = vec![0.0; input.len()];
    ssm_family_dispatch(SsmFamilyDispatch::SsmConv1d {
        library_entry: Some("SsmConv1d"),
        bind: &bind,
        input: &input,
        kernel: &kernel,
        output: &mut convolved,
    })
    .expect("CPU conv parity");
    // The Metal scan is the additive recurrence seeded from zero carry over
    // the convolved rows; the length-one decode arm is the same recurrence
    // over its single row, so the regime-matched bind reproduces it.
    let scan_bind = if length == 1 {
        SsmScanBind::decode(state_dim, [state_dim as u32, 1, 1])
    } else {
        SsmScanBind::prefill(length, state_dim, [state_dim as u32, length as u32, 1])
    };
    let mut state = vec![0.0; convolved.len()];
    ssm_family_dispatch(SsmFamilyDispatch::SsmScan {
        library_entry: Some("SsmScan"),
        bind: &scan_bind,
        input: &convolved,
        output: &mut state,
    })
    .expect("CPU scan parity");
    state
}

#[test]
fn synthetic_ssm_layer_prefill_and_decode_plan_path_run_on_metal() {
    // Environment-gated: only runs where a real Metal device exists. The
    // fake-driver lanes prove sequencing; this proof is the device numeric
    // golden for the minted SSM family bodies.
    let Ok(session) = MetalHostSession::try_open() else {
        return;
    };
    let mut host = CompositeHost::with_device(DeviceRuntime::Metal(session), "metal")
        .expect("real Metal composite host");

    let conv_kernel: Vec<f32> = (0..SSM_CONV_KERNEL as u64)
        .map(|tap| 0.25 - tap as f32 * 0.05)
        .collect();
    let kernel_bytes = conv_kernel
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<u8>>();

    for (arm, length) in [("prefill", 3u64), ("decode", 1u64)] {
        let state_dim = SSM_STATE_SIZE as u64;
        let facts = SsmFamilyMslFacts {
            length,
            state_dim,
            kernel_width: SSM_CONV_KERNEL as u64,
        };
        let module = ssm_family_msl(&facts).expect("mint SSM family MSL");
        let descriptor = ssm_layer_descriptor(
            DeviceBackend::Metal,
            module.as_bytes(),
            length,
            state_dim,
            SSM_CONV_KERNEL as u64,
        );
        let byte_weights = BTreeMap::from([(
            1,
            DeviceByteBuffer {
                bytes: kernel_bytes.clone(),
                dtype: DeviceDataType::F32,
                packed_format: None,
            },
        )]);
        let mut prepared = PreparedResidentSession::prepare_with_weight_bytes(
            &mut host,
            &descriptor,
            &BTreeMap::new(),
            &byte_weights,
        )
        .unwrap_or_else(|error| panic!("{arm} prepare: {error}"));
        // One HostProvided conv kernel copied exactly once at prepare; the
        // upload counter is host-cumulative, so the decode arm observes the
        // prefill upload plus its own.
        assert_eq!(
            prepared.driver_counters().uploads,
            if arm == "prefill" { 1 } else { 2 },
            "{arm} HostProvided conv kernel once-init"
        );

        let input: Vec<f32> = (0..length * state_dim)
            .map(|index| {
                let token = index / state_dim;
                let channel = index % state_dim;
                0.01 + token as f32 * 0.002 + channel as f32 * 0.0001
            })
            .collect();
        let expected = ssm_layer_cpu_oracle(length, state_dim, SSM_CONV_KERNEL as u64);

        // Two steps: identical output (reset/replay determinism through the
        // plan path) and the census rows (two launches, one submission, one
        // declared observation readback per step).
        let inputs = BTreeMap::from([(3u32, input)]);
        let mut first: Option<Vec<f32>> = None;
        for step in 0..2 {
            let receipt = prepared
                .execute_step(&inputs)
                .unwrap_or_else(|error| panic!("{arm} resident step {step}: {error}"));
            let observed = receipt
                .outputs
                .get(&5)
                .cloned()
                .unwrap_or_else(|| panic!("{arm} step {step} output observation"));
            assert_eq!(
                observed.len(),
                (length * state_dim) as usize,
                "{arm} state span"
            );
            let mut max_dev = 0.0f32;
            let mut max_abs = 0.0f32;
            for (index, (&got, &reference)) in observed.iter().zip(&expected).enumerate() {
                assert!(
                    got.is_finite(),
                    "{arm} step {step} non-finite at {index}: {got:?}"
                );
                max_dev = max_dev.max((got - reference).abs());
                max_abs = max_abs.max(got.abs());
            }
            assert!(
                max_dev <= max_abs * 1e-5 + 1e-7,
                "{arm} step {step} deviation {max_dev}/{max_abs} exceeds band"
            );
            // Census rows: launch_ids [1, 2] in order, one submission, and
            // the only readback is the declared scan-state observation.
            assert_eq!(receipt.launch_ids, vec![1, 2], "{arm} step {step} launches");
            assert_eq!(
                receipt.launch_entries,
                vec!["ssm_conv1d".to_owned(), "ssm_scan".to_owned()],
                "{arm} step {step} launch entries"
            );
            assert_eq!(receipt.launches, 1, "{arm} one submission per step");
            assert_eq!(
                receipt.readbacks, 1,
                "{arm} step {step} reads back exactly the state observation"
            );
            assert_eq!(receipt.observation_buffers, vec![5]);
            assert!(
                !receipt.observation_buffers.contains(&4),
                "the convolved state must never be read back per step"
            );
            if let Some(prior) = &first {
                assert_eq!(
                    prior, &observed,
                    "{arm} steps must be byte-identical (reset/replay)"
                );
            } else {
                first = Some(observed);
            }
            println!(
                "m8-u2b plan-path {arm} step {step}: submissions={} launches={:?} readbacks={} max-dev={max_dev}",
                receipt.launches, receipt.launch_ids, receipt.readbacks
            );
            // M8-U2c StageTiming evidence rows (probe-class; no llama bar).
            println!(
                "m8-u2c stage-timing {arm} step {step}: copy_in_us={} gpu_encode_submit_wait_us={} readback_us={} launch_gpu_us={:?}",
                receipt.copy_in_us,
                receipt.gpu_encode_submit_wait_us,
                receipt.readback_us,
                receipt.launch_gpu_us
            );
        }

        prepared.teardown().expect("teardown");
        assert_eq!(
            host.device().expect("device").live_handle_count(),
            0,
            "{arm} teardown leaves zero live handles"
        );
    }
}
