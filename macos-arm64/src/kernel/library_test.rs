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
