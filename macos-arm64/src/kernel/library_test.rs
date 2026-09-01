use super::*;

#[test]
fn strided_residual_matches_contiguous_logical_values() {
    let contiguous = BindDescriptor::row_major(vec![2, 3], [1, 1, 1]);
    let strided = BindDescriptor::strided(vec![2, 3], vec![4, 1], [1, 1, 1]);
    let left = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let right = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let mut expected = [0.0; 6];
    residual(&contiguous, &left, &right, &mut expected).expect("contiguous residual");
    let left_strided = [1.0, 2.0, 3.0, 99.0, 4.0, 5.0, 6.0];
    let right_strided = [6.0, 5.0, 4.0, 88.0, 3.0, 2.0, 1.0];
    let mut actual = [0.0; 7];
    residual(&strided, &left_strided, &right_strided, &mut actual).expect("strided residual");
    assert_eq!(&actual[..3], &expected[..3]);
    assert_eq!(&actual[4..7], &expected[3..6]);
}

#[test]
fn all_library_bodies_consume_bind_facts() {
    let bind = BindDescriptor::row_major(vec![2, 4], [1, 1, 1]);
    let input = [0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0];
    let gamma = [1.0, 0.9, 1.1, 0.8];
    let mut output = [0.0; 8];
    rms(&bind, &input, &gamma, &mut output, 1e-5).expect("rms");
    residual(&bind, &input, &output, &mut [0.0; 8]).expect("residual");
    swiglu(&bind, &input, &output, &mut [0.0; 8]).expect("swiglu");
    softmax(&bind, &input, &mut output).expect("softmax");
    let rope_bind = BindDescriptor {
        dims: vec![2, 4],
        strides: vec![4, 1],
        layout: BindLayout::RopeConsecutivePair { rotated_width: 4 },
        grid: [1, 1, 1],
    };
    rope(&rope_bind, &input, &[1.0, 1.0], &[0.0, 0.0], &mut output).expect("rope");
}

#[test]
fn q6_k_lane_outside_block_fails_closed() {
    assert_eq!(
        q6_k_lane_quant(4, 0, 0, 0, 0),
        Err(KernelBodyError::ShapeMismatch(
            "Q6_K lane is bounded by 256-element blocks"
        ))
    );
    assert!(q6_k_lane_quant(3, 0x11, 0x11, 0, 0).is_ok());
}

#[test]
fn invalid_bind_fails_before_buffer_access() {
    let bind = BindDescriptor::strided(vec![2, 3], vec![0, 1], [1, 1, 1]);
    let mut output = [0.0; 6];
    assert!(matches!(
        residual(&bind, &[], &[], &mut output),
        Err(KernelBodyError::InvalidBind(_))
    ));
}

#[derive(Debug, serde::Deserialize)]
struct DenseGolden {
    inputs: Vec<DenseGoldenTensor>,
    expected_output: DenseGoldenTensor,
}

#[derive(Debug, serde::Deserialize)]
struct DenseGoldenTensor {
    name: String,
    elements: usize,
    f32_le_hex: String,
}

fn dense_golden_values(tensor: &DenseGoldenTensor) -> Vec<f32> {
    let hex: String = tensor
        .f32_le_hex
        .chars()
        .filter(|character| !character.is_ascii_whitespace())
        .collect();
    assert_eq!(hex.len(), tensor.elements * 8, "{} hex width", tensor.name);
    hex.as_bytes()
        .chunks_exact(8)
        .map(|chunk| {
            let bytes = chunk
                .chunks_exact(2)
                .map(|pair| {
                    u8::from_str_radix(std::str::from_utf8(pair).expect("ASCII hex"), 16)
                        .expect("f32 hex")
                })
                .collect::<Vec<_>>();
            f32::from_bits(u32::from_le_bytes(bytes.try_into().expect("four bytes")))
        })
        .collect()
}

#[test]
fn qkv_projection_matches_dense_golden_class() {
    const GOLDEN: &str = include_str!("../../tests/fixtures/gi2-2-op-goldens/dense.json");
    let golden: DenseGolden = serde_json::from_str(GOLDEN).expect("dense golden");
    let weight = golden
        .inputs
        .iter()
        .find(|tensor| tensor.name == "weight")
        .map(dense_golden_values)
        .expect("dense weight");
    let input = golden
        .inputs
        .iter()
        .find(|tensor| tensor.name == "input")
        .map(dense_golden_values)
        .expect("dense input");
    let expected = dense_golden_values(&golden.expected_output);
    let bind = QkvProjectionBind::grouped(1, 960, 1, 1, 64, [64, 1, 1]);
    let mut q = vec![0.0; 64];
    let mut k = vec![0.0; 64];
    let mut v = vec![0.0; 64];
    qkv_projection(
        &bind,
        &input,
        QkvProjectionWeight::Dense(&weight),
        QkvProjectionWeight::Dense(&weight),
        QkvProjectionWeight::Dense(&weight),
        None,
        None,
        None,
        None,
        None,
        &mut q,
        &mut k,
        &mut v,
    )
    .expect("dense QKV body");
    assert_eq!(q.len(), expected.len());
    let max_abs_delta = q
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs_delta <= 1.0e-6,
        "dense QKV golden mismatch: max_abs_delta={max_abs_delta}"
    );
}

#[test]
fn qkv_projection_applies_qwen_bias_and_gqa_rope_in_one_body() {
    let mut bind = QkvProjectionBind::grouped(1, 28, 2, 7, 2, [28, 1, 1]);
    bind.rotate_half = true;
    let input = vec![1.0f32; 28];
    let q_weight = vec![1.0f32; 28 * 28];
    let k_weight = vec![2.0f32; 28 * 4];
    let v_weight = vec![3.0f32; 28 * 4];
    let q_bias = vec![0.5f32; 28];
    let k_bias = vec![1.0f32; 4];
    let v_bias = vec![2.0f32; 4];
    let cos = [1.0f32, 1.0];
    let sin = [0.0f32, 0.0];
    let mut q = vec![0.0; 28];
    let mut k = vec![0.0; 4];
    let mut v = vec![0.0; 4];
    let selected = select_qkv_projection(Some("QkvProjection"), 1, QkvProjectionLayout::Grouped)
        .expect("QKV selector")
        .expect("QKV body selected");
    dispatch_qkv_projection(
        selected,
        &bind,
        &input,
        [
            QkvProjectionWeight::Dense(&q_weight),
            QkvProjectionWeight::Dense(&k_weight),
            QkvProjectionWeight::Dense(&v_weight),
        ],
        [Some(&q_bias), Some(&k_bias), Some(&v_bias)],
        Some((&cos, &sin)),
        [&mut q, &mut k, &mut v],
    )
    .expect("Qwen QKV body");
    assert!(q.iter().all(|value| (*value - 28.5).abs() <= 1.0e-6));
    assert!(k.iter().all(|value| (*value - 57.0).abs() <= 1.0e-6));
    assert!(v.iter().all(|value| (*value - 86.0).abs() <= 1.0e-6));
    assert!(matches!(
        select_qkv_projection(Some("QkvProjection"), 0, QkvProjectionLayout::Unsupported),
        Err(KernelBodyError::InvalidBind(message)) if message.contains("not servable")
    ));
}

#[test]
fn qkv_projection_rotates_query_rows_at_cursor_position() {
    // rows=2, one head, head_dim=2 (one RoPE pair per row). The identity
    // weight publishes the activation rows untouched, so Q carries exactly
    // the rotation of [1, 2] and [3, 4]; V stays unrotated.
    let mut bind = QkvProjectionBind::grouped(2, 2, 1, 1, 2, [2, 1, 1]);
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let weight = vec![1.0f32, 0.0, 0.0, 1.0];
    // Table rows 0/1/2 rotate by 0, quarter, and half turn respectively.
    let cos = [1.0f32, 0.0, -1.0];
    let sin = [0.0f32, 1.0, 0.0];
    let mut run = |rope_position: u64| {
        bind.rope_position = rope_position;
        let mut q = vec![0.0f32; 4];
        let mut k = vec![0.0f32; 4];
        let mut v = vec![0.0f32; 4];
        qkv_projection(
            &bind,
            &input,
            QkvProjectionWeight::Dense(&weight),
            QkvProjectionWeight::Dense(&weight),
            QkvProjectionWeight::Dense(&weight),
            None,
            None,
            None,
            Some(&cos),
            Some(&sin),
            &mut q,
            &mut k,
            &mut v,
        )
        .expect("positioned QKV body");
        (q, v)
    };
    // Default position 0: row i rotates at table row i.
    let (q, v) = run(0);
    assert!(q
        .iter()
        .zip([1.0f32, 2.0, -4.0, 3.0])
        .all(|(a, e)| (a - e).abs() <= 1.0e-6));
    assert!(v.iter().zip(&input).all(|(a, e)| (a - e).abs() <= 1.0e-6));
    // Cursor position 1: row i rotates at table row 1 + i.
    let (q, v) = run(1);
    assert!(q
        .iter()
        .zip([-2.0f32, 1.0, -3.0, -4.0])
        .all(|(a, e)| (a - e).abs() <= 1.0e-6));
    assert!(v.iter().zip(&input).all(|(a, e)| (a - e).abs() <= 1.0e-6));
}

#[test]
fn qkv_projection_fails_closed_when_rope_table_under_covers_cursor_span() {
    let mut bind = QkvProjectionBind::grouped(2, 2, 1, 1, 2, [2, 1, 1]);
    bind.rope_position = 1;
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let weight = vec![1.0f32, 0.0, 0.0, 1.0];
    // Two table rows cover positions 0..1; the cursor span needs 0..=2.
    let cos = [1.0f32, 0.0];
    let sin = [0.0f32, 1.0];
    let mut q = vec![0.0f32; 4];
    let mut k = vec![0.0f32; 4];
    let mut v = vec![0.0f32; 4];
    let error = qkv_projection(
        &bind,
        &input,
        QkvProjectionWeight::Dense(&weight),
        QkvProjectionWeight::Dense(&weight),
        QkvProjectionWeight::Dense(&weight),
        None,
        None,
        None,
        Some(&cos),
        Some(&sin),
        &mut q,
        &mut k,
        &mut v,
    )
    .expect_err("under-covering table must fail closed");
    assert!(
        matches!(error, KernelBodyError::BufferTooShort { ref buffer, required: 3, actual: 2 } if *buffer == "QKV RoPE table"),
        "unexpected error: {error:?}"
    );
    assert!(q.iter().all(|value| *value == 0.0));
}

#[test]
fn residual_rms_norm_preserves_residual_then_rms_order() {
    let bind = BindDescriptor::row_major(vec![2, 4], [1, 1, 1]);
    let residual_values = [0.25f32, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0];
    let skip = [0.5f32, 0.25, -0.25, 0.75, -0.5, 0.5, -0.75, 1.0];
    let gamma = [1.0f32, 0.9, 1.1, 0.8];
    let mut summed = [0.0f32; 8];
    let mut expected = [0.0f32; 8];
    residual(&bind, &residual_values, &skip, &mut summed).expect("residual baseline");
    rms(&bind, &summed, &gamma, &mut expected, 1e-5).expect("RMS baseline");

    let selected = select_residual_rms_norm(Some("ResidualRmsNorm"), BindLayout::RowMajor)
        .expect("ResidualRmsNorm selector")
        .expect("ResidualRmsNorm body selected");
    let mut actual = [0.0f32; 8];
    dispatch_residual_rms_norm(
        selected,
        &bind,
        &residual_values,
        &skip,
        &gamma,
        &mut actual,
        1e-5,
    )
    .expect("ResidualRmsNorm body");
    assert_eq!(actual, expected, "fused arithmetic order changed");
    assert!(matches!(
        select_residual_rms_norm(Some("ResidualRmsNorm"), BindLayout::Flat),
        Err(KernelBodyError::InvalidBind(message)) if message.contains("not servable")
    ));
}

#[derive(Debug, Clone, Copy)]
struct GroupedExpertAnalog {
    name: &'static str,
    rows: usize,
    columns: usize,
}

const GROUPED_EXPERT_ANALOGS: [GroupedExpertAnalog; 2] = [
    GroupedExpertAnalog {
        name: "smollm2-360m-gqa",
        rows: 3,
        columns: 5,
    },
    GroupedExpertAnalog {
        name: "qwen2.5-0.5b-gqa",
        rows: 7,
        columns: 2,
    },
];

const GROUPED_EXPERT_K: usize = 32;
const GROUPED_EXPERT_COUNT: usize = 2;

fn grouped_q8_0_block(value: i8) -> Vec<u8> {
    let mut block = vec![0x00, 0x3c]; // f16 scale = 1.0
    block.extend(std::iter::repeat_n(value as u8, GROUPED_EXPERT_K));
    block
}

fn grouped_expert_activation(fixture: GroupedExpertAnalog) -> Vec<f32> {
    (0..fixture.rows * GROUPED_EXPERT_K)
        .map(|index| {
            let row = index / GROUPED_EXPERT_K;
            let element = index % GROUPED_EXPERT_K;
            0.5 + row as f32 * 0.25 + element as f32 * 0.01
        })
        .collect()
}

fn grouped_expert_packed_weights(fixture: GroupedExpertAnalog, tied: bool) -> Vec<u8> {
    (0..GROUPED_EXPERT_COUNT)
        .flat_map(|expert| {
            (0..fixture.columns).flat_map(move |column| {
                let value = if tied {
                    column + 1
                } else {
                    expert * 2 + column + 1
                };
                grouped_q8_0_block(value as i8)
            })
        })
        .collect()
}

fn grouped_expert_reference(fixture: GroupedExpertAnalog, tied: bool) -> (Vec<Vec<f32>>, Vec<f32>) {
    let activation = grouped_expert_activation(fixture);
    let mut intermediates =
        vec![vec![0.0f32; fixture.rows * fixture.columns]; GROUPED_EXPERT_COUNT];
    for (expert, rows) in intermediates.iter_mut().enumerate() {
        for row in 0..fixture.rows {
            for column in 0..fixture.columns {
                let value = if tied {
                    column + 1
                } else {
                    expert * 2 + column + 1
                };
                let weight = value as f32;
                rows[row * fixture.columns + column] = (0..GROUPED_EXPERT_K)
                    .map(|element| activation[row * GROUPED_EXPERT_K + element] * weight)
                    .sum();
            }
        }
    }
    let mut accumulated = vec![0.0f32; fixture.rows * fixture.columns];
    for rows in &intermediates {
        for (output, value) in accumulated.iter_mut().zip(rows) {
            *output += *value;
        }
    }
    (intermediates, accumulated)
}

fn grouped_expert_bind(fixture: GroupedExpertAnalog, experts: usize) -> GroupedExpertGemmBind {
    GroupedExpertGemmBind::contiguous(
        fixture.rows as u64,
        GROUPED_EXPERT_K as u64,
        fixture.columns as u64,
        experts as u64,
        QuantizedFormat::Q8_0,
        [fixture.columns as u32, fixture.rows as u32, 1],
    )
}

#[test]
fn grouped_expert_gemm_matches_gqa_shape_analogs_and_accumulates_segments() {
    for fixture in GROUPED_EXPERT_ANALOGS {
        let bind = grouped_expert_bind(fixture, GROUPED_EXPERT_COUNT);
        let shapes = [[GROUPED_EXPERT_K as u64, fixture.columns as u64]; GROUPED_EXPERT_COUNT];
        let activation = grouped_expert_activation(fixture);
        let packed = grouped_expert_packed_weights(fixture, false);
        let (expected_intermediates, expected) = grouped_expert_reference(fixture, false);
        let mut actual = vec![0.0f32; fixture.rows * fixture.columns];
        dispatch_grouped_expert_gemm(
            GroupedExpertGemmKernel::Packed,
            &bind,
            &shapes,
            &activation,
            &packed,
            &mut actual,
        )
        .unwrap_or_else(|error| panic!("{} grouped body: {error}", fixture.name));
        assert_eq!(actual, expected, "{} accumulated output", fixture.name);

        // Run each segment as a one-expert dispatch.  This checks the packed
        // weights and each expert intermediate independently of accumulation.
        let one_expert_bind = grouped_expert_bind(fixture, 1);
        let expert_stride = bind.packed_expert_stride_bytes as usize;
        for expert in 0..GROUPED_EXPERT_COUNT {
            let start = expert * expert_stride;
            let end = start + expert_stride;
            let mut intermediate = vec![0.0f32; fixture.rows * fixture.columns];
            dispatch_grouped_expert_gemm(
                GroupedExpertGemmKernel::Packed,
                &one_expert_bind,
                &[[GROUPED_EXPERT_K as u64, fixture.columns as u64]],
                &activation,
                &packed[start..end],
                &mut intermediate,
            )
            .unwrap_or_else(|error| panic!("{} expert {expert}: {error}", fixture.name));
            assert_eq!(
                intermediate, expected_intermediates[expert],
                "{} expert {expert} intermediate",
                fixture.name
            );
        }
    }
}

#[test]
fn grouped_expert_gemm_tied_contributions_are_byte_deterministic() {
    let fixture = GROUPED_EXPERT_ANALOGS[0];
    let bind = grouped_expert_bind(fixture, GROUPED_EXPERT_COUNT);
    let shapes = [[GROUPED_EXPERT_K as u64, fixture.columns as u64]; GROUPED_EXPERT_COUNT];
    let activation = grouped_expert_activation(fixture);
    let packed = grouped_expert_packed_weights(fixture, true);
    let (_, expected) = grouped_expert_reference(fixture, true);
    let mut first = vec![0.0f32; fixture.rows * fixture.columns];
    let mut second = vec![0.0f32; fixture.rows * fixture.columns];
    for output in [&mut first, &mut second] {
        dispatch_grouped_expert_gemm(
            GroupedExpertGemmKernel::Packed,
            &bind,
            &shapes,
            &activation,
            &packed,
            output,
        )
        .expect("tied grouped expert body");
    }
    assert_eq!(first, expected, "tied expert accumulation oracle");
    assert_eq!(first, second, "expert traversal must be deterministic");
}

#[test]
fn grouped_expert_gemm_rejects_mixed_shape_batches_before_buffer_access() {
    let fixture = GROUPED_EXPERT_ANALOGS[0];
    let bind = grouped_expert_bind(fixture, GROUPED_EXPERT_COUNT);
    let mixed_shapes = [[GROUPED_EXPERT_K as u64, fixture.columns as u64], [31, 1]];
    let mut output = [f32::NAN; 15];
    let error = dispatch_grouped_expert_gemm(
        GroupedExpertGemmKernel::Packed,
        &bind,
        &mixed_shapes,
        &[],
        &[],
        &mut output,
    )
    .expect_err("mixed-shape expert batch must fail closed");
    assert!(matches!(
        error,
        KernelBodyError::ShapeMismatch(message) if message.contains("mixed-shape")
    ));
    assert!(output.iter().all(|value| value.is_nan()));
}
