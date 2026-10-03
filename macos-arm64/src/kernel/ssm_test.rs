//! Companion tests for [`super::ssm`] (repo companion-test convention).

use super::*;

#[test]
fn ssm_family_msl_fails_closed_on_zero_dimension() {
    let error = ssm_family_msl(&SsmFamilyMslFacts {
        length: 0,
        state_dim: 8,
        kernel_width: 4,
    })
    .expect_err("zero length must fail closed");
    assert!(matches!(
        error,
        KernelBodyError::InvalidBind(message) if message.contains("zero dimension")
    ));
}

#[test]
fn ssm_family_dispatch_rejects_a_mismatched_library_entry() {
    let bind = SsmScanBind::decode(2, [2, 1, 1]);
    let input = [1.0f32, 2.0];
    let mut output = [0.0f32; 2];
    let error = ssm_family_dispatch(SsmFamilyDispatch::SsmScan {
        library_entry: Some("SsmConv1d"),
        bind: &bind,
        input: &input,
        output: &mut output,
    })
    .expect_err("wrong entry must fail closed");
    assert!(matches!(
        error,
        KernelBodyError::InvalidBind(message) if message.contains("disagrees with library_entry")
    ));
    assert_eq!(output, [0.0; 2], "failed selection must not write");
}
