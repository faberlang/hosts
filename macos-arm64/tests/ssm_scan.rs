//! Focused EXEC02-PS2 SsmScan Metal-library probe.
//!
//! The fixture is synthetic by design: the admitted SsmScan body has no
//! coefficient or prior-state buffers, so it proves the additive recurrence
//! `state[t, d] = state[t - 1, d] + input[t, d]` in prefill and its length-one
//! decode/state-update specialization.

use faber_host_macos_arm64::kernel::library::KernelBodyError;
use faber_host_macos_arm64::kernel::ssm_scan::{
    dispatch_ssm_scan, SsmScanBind, SsmScanKernel, SsmScanLayout, SsmScanRegime,
};

fn cpu_additive_scan(input: &[f32], length: usize, state_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for state in 0..state_dim {
        let mut carry = 0.0;
        for time in 0..length {
            let index = time * state_dim + state;
            carry += input[index];
            output[index] = carry;
        }
    }
    output
}

#[test]
fn ssm_scan_prefill_and_decode_match_cpu_reference() {
    let length = 4usize;
    let state_dim = 3usize;
    let input = [
        1.0, 2.0, 3.0, // t0
        0.5, -1.0, 2.0, // t1
        2.0, 4.0, -0.5, // t2
        -1.0, 0.25, 1.5, // t3
    ];
    let expected = cpu_additive_scan(&input, length, state_dim);
    let bind = SsmScanBind::prefill(length as u64, state_dim as u64, [12, 1, 1]);
    let mut actual = vec![0.0f32; input.len()];

    dispatch_ssm_scan(SsmScanKernel::Additive, &bind, &input, &mut actual)
        .expect("SsmScan Metal prefill body");

    assert_eq!(actual, expected, "prefill state transitions");
    assert_eq!(bind.grid, [12, 1, 1]);

    let decode_input = [2.5f32, -0.75];
    let decode_bind = SsmScanBind::decode(decode_input.len() as u64, [2, 1, 1]);
    let decode_expected = cpu_additive_scan(&decode_input, 1, decode_input.len());
    let mut decode_actual = vec![0.0f32; decode_input.len()];

    dispatch_ssm_scan(
        SsmScanKernel::Additive,
        &decode_bind,
        &decode_input,
        &mut decode_actual,
    )
    .expect("SsmScan Metal decode body");

    assert_eq!(decode_actual, decode_expected, "decode state update");
}

#[test]
fn ssm_scan_rejects_regime_mislabel_before_buffer_access() {
    let mut bind = SsmScanBind::prefill(4, 2, [8, 1, 1]);
    bind.regime = SsmScanRegime::Decode;
    let mut output = [f32::NAN; 8];

    let error = dispatch_ssm_scan(SsmScanKernel::Additive, &bind, &[], &mut output)
        .expect_err("length-four decode label must fail closed");
    assert!(matches!(
        error,
        KernelBodyError::InvalidBind(message) if message.contains("decode regime")
    ));
    assert!(output.iter().all(|value| value.is_nan()));
}

#[test]
fn ssm_scan_rejects_unservable_state_layout() {
    let mut bind = SsmScanBind::decode(2, [2, 1, 1]);
    bind.layout = SsmScanLayout::Unsupported;
    let mut output = [17.0f32; 2];

    let error = dispatch_ssm_scan(SsmScanKernel::Additive, &bind, &[], &mut output)
        .expect_err("unsupported SSM state layout must fail closed");
    assert!(matches!(
        error,
        KernelBodyError::InvalidBind(message) if message.contains("not servable")
    ));
    assert_eq!(output, [17.0; 2]);
}
