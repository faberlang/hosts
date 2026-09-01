//! Focused EXEC02-PS5 per-layer hybrid prefill/decode probe.
//!
//! The fixture uses the MODEL-01 qwen35moe shape family: a hybrid block with
//! the 2-KV-group / 16-query-head GQA attention mix and the 128-channel,
//! four-tap SSM state family.  It is deliberately synthetic until MODEL-03
//! lands; this test proves that the two state families remain separate while
//! both prefill and one-token decode use their named regimes.

use faber_host_macos_arm64::kernel::library::{dispatch, CausalAttentionBind, LibraryKernel};
use faber_host_macos_arm64::kernel::ssm_conv1d::{
    dispatch_ssm_conv1d, SsmConv1dBind, SsmConv1dKernel,
};
use faber_host_macos_arm64::kernel::ssm_scan::{dispatch_ssm_scan, SsmScanBind, SsmScanKernel};

const MODEL_BLOCK: usize = 0;
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
            value_offset + group as f32 * 0.003 + token as f32 * 0.001 + dimension as f32 * 0.00001
        })
        .collect()
}

fn extend_attention_kv(prefill: &[f32], value_offset: f32) -> Vec<f32> {
    let mut output = vec![0.0; KV_HEAD_COUNT * (PREFILL_TOKENS + 1) * HEAD_DIM];
    let prefill_group_stride = PREFILL_TOKENS * HEAD_DIM;
    let decode_group_stride = (PREFILL_TOKENS + 1) * HEAD_DIM;
    for group in 0..KV_HEAD_COUNT {
        let prefill_start = group * prefill_group_stride;
        let decode_start = group * decode_group_stride;
        output[decode_start..decode_start + prefill_group_stride]
            .copy_from_slice(&prefill[prefill_start..prefill_start + prefill_group_stride]);
        let token = PREFILL_TOKENS;
        for dimension in 0..HEAD_DIM {
            output[decode_start + token * HEAD_DIM + dimension] = value_offset
                + group as f32 * 0.003
                + token as f32 * 0.001
                + dimension as f32 * 0.00001;
        }
    }
    output
}

fn assert_attention_kv_prefix(prefill: &[f32], decode: &[f32], label: &str) {
    let prefill_group_stride = PREFILL_TOKENS * HEAD_DIM;
    let decode_group_stride = (PREFILL_TOKENS + 1) * HEAD_DIM;
    for group in 0..KV_HEAD_COUNT {
        let prefill_start = group * prefill_group_stride;
        let decode_start = group * decode_group_stride;
        assert_eq!(
            &decode[decode_start..decode_start + prefill_group_stride],
            &prefill[prefill_start..prefill_start + prefill_group_stride],
            "{label} group {group}"
        );
    }
}

fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], seq_block: usize, query_seq: usize) -> Vec<f32> {
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
                        .map(|(token, weight)| *weight * v[k_base + token * HEAD_DIM + dimension])
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

#[test]
fn qwen35moe_hybrid_layer_prefill_and_decode_keep_state_families_separate() {
    assert_eq!(
        MODEL_BLOCK % 4,
        0,
        "fixture must be a hybrid MODEL-01 block"
    );
    assert_eq!(HEAD_COUNT, KV_HEAD_COUNT * Q_PER_KV);

    let conv_kernel = [0.25f32, -0.1, 0.05, 0.02];
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
    dispatch_ssm_conv1d(
        SsmConv1dKernel::Causal,
        &conv_prefill_bind,
        &prefill_input,
        &conv_kernel,
        &mut conv_prefill,
    )
    .expect("PS1 SSM prefill body");
    assert_close(
        &conv_prefill,
        &expected_conv_prefill,
        "SSM prefill convolution",
    );

    let expected_state_prefill = cpu_scan(&expected_conv_prefill, PREFILL_TOKENS, SSM_STATE_SIZE);
    let scan_prefill_bind = SsmScanBind::prefill(
        PREFILL_TOKENS as u64,
        SSM_STATE_SIZE as u64,
        [SSM_STATE_SIZE as u32, PREFILL_TOKENS as u32, 1],
    );
    let mut state_prefill = vec![0.0; expected_state_prefill.len()];
    dispatch_ssm_scan(
        SsmScanKernel::Additive,
        &scan_prefill_bind,
        &conv_prefill,
        &mut state_prefill,
    )
    .expect("PS2 SSM prefill body");
    assert_close(&state_prefill, &expected_state_prefill, "SSM prefill state");

    let decode_input = synthetic_ssm_input(1);
    let expected_conv_decode = cpu_conv1d(&decode_input, 1, SSM_STATE_SIZE, &conv_kernel);
    let conv_decode_bind = SsmConv1dBind::channels_last(
        1,
        SSM_STATE_SIZE as u64,
        SSM_CONV_KERNEL as u64,
        [SSM_STATE_SIZE as u32, 1, 1],
    );
    let mut conv_decode = vec![0.0; expected_conv_decode.len()];
    dispatch_ssm_conv1d(
        SsmConv1dKernel::Causal,
        &conv_decode_bind,
        &decode_input,
        &conv_kernel,
        &mut conv_decode,
    )
    .expect("PS1 SSM decode body");
    assert_close(
        &conv_decode,
        &expected_conv_decode,
        "SSM decode convolution",
    );

    let previous_state = &state_prefill[(PREFILL_TOKENS - 1) * SSM_STATE_SIZE..];
    let decode_state_input: Vec<f32> = previous_state
        .iter()
        .zip(&conv_decode)
        .map(|(previous, update)| previous + update)
        .collect();
    let scan_decode_bind =
        SsmScanBind::decode(SSM_STATE_SIZE as u64, [SSM_STATE_SIZE as u32, 1, 1]);
    let mut state_decode = vec![0.0; SSM_STATE_SIZE];
    dispatch_ssm_scan(
        SsmScanKernel::Additive,
        &scan_decode_bind,
        &decode_state_input,
        &mut state_decode,
    )
    .expect("PS2 SSM decode body");
    assert_close(&state_decode, &decode_state_input, "SSM decode state");

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
        &cpu_attention(
            &prefill_q,
            &prefill_k,
            &prefill_v,
            PREFILL_TOKENS,
            PREFILL_TOKENS,
        ),
        "KV prefill output",
    );

    let decode_q = fill_attention_q(1, 0.025);
    let decode_k = extend_attention_kv(&prefill_k, 0.01);
    let decode_v = extend_attention_kv(&prefill_v, 0.04);
    assert_attention_kv_prefix(
        &prefill_k,
        &decode_k,
        "decode KV state starts from prefill rows",
    );
    assert_attention_kv_prefix(
        &prefill_v,
        &decode_v,
        "decode KV values start from prefill rows",
    );
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

    println!(
        "EXEC02-PS5 layer=blk.{MODEL_BLOCK} state=SSM prefill_rows={PREFILL_TOKENS} decode_rows=1 state_size={SSM_STATE_SIZE}; state=KV prefill_rows={PREFILL_TOKENS} decode_rows=1 kv_heads={KV_HEAD_COUNT} q_heads={HEAD_COUNT} q_per_kv={Q_PER_KV} head_dim={HEAD_DIM}; regimes=prefill,decode"
    );
}
