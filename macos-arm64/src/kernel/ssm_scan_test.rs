//! Companion tests for [`super::ssm_scan`] (repo companion-test convention).

use super::*;

#[test]
fn additive_prefill_scan_matches_state_rows() {
    let input = [
        1.0f32, 2.0, 3.0, // t0
        0.5, -1.0, 2.0, // t1
        2.0, 4.0, -0.5, // t2
        -1.0, 0.25, 1.5, // t3
    ];
    let bind = SsmScanBind::prefill(4, 3, [12, 1, 1]);
    let mut output = [0.0f32; 12];

    dispatch_ssm_scan(SsmScanKernel::Additive, &bind, &input, &mut output)
        .expect("prefill SSM scan");

    assert_eq!(
        output,
        [
            1.0, 2.0, 3.0, // t0
            1.5, 1.0, 5.0, // t1
            3.5, 5.0, 4.5, // t2
            2.5, 5.25, 6.0, // t3
        ]
    );
}

#[test]
fn decode_arm_is_the_length_one_state_update() {
    let input = [2.5f32, -0.75];
    let bind = SsmScanBind::decode(2, [2, 1, 1]);
    let mut output = [0.0f32; 2];

    dispatch_ssm_scan(SsmScanKernel::Additive, &bind, &input, &mut output)
        .expect("decode SSM scan");

    assert_eq!(output, input);
}
