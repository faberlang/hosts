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
