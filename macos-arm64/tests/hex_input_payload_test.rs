//! Fail-closed hex decoding for dtype-tagged device-execute inputs.

use faber_host_macos_arm64::device_execute::inputs_from_json;

fn tagged(bytes: &str) -> String {
    format!(r#"{{"1":{{"dtype":"f16","bytes":"{bytes}"}}}}"#)
}

#[test]
fn tagged_hex_accepts_mixed_case_and_optional_prefix() {
    let decoded = inputs_from_json(tagged("0xABcd").as_bytes()).expect("mixed case with prefix");
    assert_eq!(decoded.byte_map()[&1].bytes, vec![0xAB, 0xCD]);

    let decoded = inputs_from_json(tagged("ef01").as_bytes()).expect("lowercase without prefix");
    assert_eq!(decoded.byte_map()[&1].bytes, vec![0xEF, 0x01]);
}

#[test]
fn tagged_hex_rejects_odd_length_and_non_digits() {
    let odd = inputs_from_json(tagged("0x0").as_bytes()).expect_err("odd length");
    assert_eq!(odd.code, "E_INVALID_ARGS");
    assert!(
        odd.message.contains("not whole bytes"),
        "odd-length error must name the defect, got {}",
        odd.message
    );

    let bad = inputs_from_json(tagged("0x0g").as_bytes()).expect_err("non-hex digit");
    assert_eq!(bad.code, "E_INVALID_ARGS");
    assert!(
        bad.message.contains("not hex"),
        "non-hex error must name the defect, got {}",
        bad.message
    );
}
