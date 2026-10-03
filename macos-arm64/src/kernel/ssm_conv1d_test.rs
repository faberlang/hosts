//! Companion tests for [`super::ssm_conv1d`] (repo companion-test convention).

use super::*;

#[test]
fn causal_conv_matches_per_channel_reference_rows() {
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let weights = [0.5f32, -1.0, 2.0];
    let bind = SsmConv1dBind::channels_last(4, 2, 3, [8, 1, 1]);
    let mut output = [0.0f32; 8];

    dispatch_ssm_conv1d(
        SsmConv1dKernel::Causal,
        &bind,
        &input,
        &weights,
        &mut output,
    )
    .expect("causal SSM convolution");

    assert_eq!(output, [0.5, 1.0, 0.5, 0.0, 1.5, 3.0, 4.5, 6.0]);
}

#[test]
fn unservable_state_layout_fails_closed_before_buffer_access() {
    let mut bind = SsmConv1dBind::channels_last(2, 2, 2, [4, 1, 1]);
    bind.layout = SsmConv1dLayout::Unsupported;
    let mut output = [91.0f32; 4];

    let error = ssm_conv1d(&bind, &[], &[], &mut output)
        .expect_err("unsupported state layout must fail closed");
    assert!(matches!(
        error,
        KernelBodyError::InvalidBind(message) if message.contains("not servable")
    ));
    assert_eq!(output, [91.0; 4]);
}
