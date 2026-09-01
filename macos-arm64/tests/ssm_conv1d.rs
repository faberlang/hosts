//! Focused EXEC02-PS1 SsmConv1d Metal-library probe.

use faber_host_macos_arm64::kernel::library::KernelBodyError;
use faber_host_macos_arm64::kernel::ssm_conv1d::{
    SsmConv1dBind, SsmConv1dKernel, SsmConv1dLayout, dispatch_ssm_conv1d,
};

#[test]
fn ssm_conv1d_dispatch_matches_synthetic_state_channel_rows() {
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
    .expect("SsmConv1d Metal body");

    assert_eq!(output, [0.5, 1.0, 0.5, 0.0, 1.5, 3.0, 4.5, 6.0]);
    // One selected body dispatch covers all [time, channel] outputs.
    assert_eq!(bind.grid, [8, 1, 1]);
}

#[test]
fn ssm_conv1d_rejects_unservable_state_layout() {
    let mut bind = SsmConv1dBind::channels_last(2, 2, 2, [4, 1, 1]);
    bind.layout = SsmConv1dLayout::Unsupported;
    let mut output = [17.0f32; 4];

    let error = dispatch_ssm_conv1d(SsmConv1dKernel::Causal, &bind, &[], &[], &mut output)
        .expect_err("unsupported state layout must fail closed");
    assert!(matches!(
        error,
        KernelBodyError::InvalidBind(message) if message.contains("not servable")
    ));
    assert_eq!(output, [17.0; 4]);
}
