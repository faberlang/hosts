//! Numeric coverage for the M6-U1 sequence-one quantized GEMV variants.
//!
//! These synthetic blocks exercise the same packed byte layouts and f32
//! dequant formulas as the R-PACK-02 Metal bodies.  The bind carries every
//! shape and stride fact; no model-specific projection dimensions are hidden
//! in the test or library body.

use faber_host_macos_arm64::kernel::library::{
    self, dispatch_gemv, GemvKernel, KernelBodyError, QuantizedFormat, QuantizedGemvBind,
};

fn q4_k_ones_block() -> Vec<u8> {
    let mut block = vec![0u8; 144];
    block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
    for byte in &mut block[4..8] {
        *byte = 1;
    }
    for byte in &mut block[12..16] {
        *byte = 1;
    }
    block[16..].fill(0x11);
    block
}

fn q5_0_negative_fifteen_block() -> Vec<u8> {
    let mut block = vec![0u8; 22];
    block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
    block[6..].fill(0x11);
    block
}

fn q5_k_ones_block() -> Vec<u8> {
    let mut block = vec![0u8; 176];
    block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
    block[4..8].fill(1);
    block[12..16].fill(1);
    block[48..].fill(0x11);
    block
}

fn q6_k_negative_thirty_one_block() -> Vec<u8> {
    let mut block = vec![0u8; 210];
    block[192..208].fill(1);
    block[208..210].copy_from_slice(&0x3c00u16.to_le_bytes());
    block[..128].fill(0x11);
    block
}

fn q8_0_ramp_block() -> Vec<u8> {
    let mut block = vec![0u8; 34];
    block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
    for (index, byte) in block[2..].iter_mut().enumerate() {
        *byte = u8::try_from(index).expect("Q8_0 test ramp fits in u8");
    }
    block
}

fn repeated_columns(block: &[u8], columns: usize) -> Vec<u8> {
    block.repeat(columns)
}

#[test]
fn each_rpack02_format_dequantizes_inside_gemv() {
    let cases = [
        (QuantizedFormat::Q4K, 256, q4_k_ones_block(), 256.0f32),
        (
            QuantizedFormat::Q5_0,
            32,
            q5_0_negative_fifteen_block(),
            -480.0,
        ),
        (QuantizedFormat::Q5K, 256, q5_k_ones_block(), 256.0),
        (
            QuantizedFormat::Q6K,
            256,
            q6_k_negative_thirty_one_block(),
            -7936.0,
        ),
        (QuantizedFormat::Q8_0, 32, q8_0_ramp_block(), 496.0),
    ];

    for (format, k, block, expected) in cases {
        let bind = QuantizedGemvBind::decode(k, 2, format, [2, 1, 1]);
        let packed = repeated_columns(&block, 2);
        let activation = vec![1.0f32; k as usize];
        let mut output = vec![0.0f32; 2];
        dispatch_gemv(
            GemvKernel::Quantized,
            &bind,
            &activation,
            &packed,
            &mut output,
        )
        .unwrap_or_else(|error| panic!("{} GEMV failed: {error}", format.spelling()));
        assert_eq!(output, [expected, expected], "{} output", format.spelling());
    }
}

#[test]
fn gemv_bind_strides_parameterize_activation_output_and_weight_columns() {
    let block = q5_0_negative_fifteen_block();
    let bind = QuantizedGemvBind::strided(32, 2, 2, 3, 24, QuantizedFormat::Q5_0, [2, 1, 1]);
    let mut activation = vec![0.0f32; 63];
    for index in 0..32 {
        activation[index * 2] = 1.0;
    }
    let mut packed = vec![0xa5u8; 48];
    packed[..22].copy_from_slice(&block);
    packed[24..46].copy_from_slice(&block);
    let mut output = vec![99.0f32; 5];

    library::quantized_gemv(&bind, &activation, &packed, &mut output)
        .expect("strided quantized GEMV");
    assert_eq!(output[0], -480.0);
    assert_eq!(output[3], -480.0);
    assert_eq!(output[1], 99.0);
    assert_eq!(output[2], 99.0);
    assert_eq!(output[4], 99.0);
}

#[test]
fn gemv_rejects_unaligned_k_and_truncated_blocks_before_access() {
    let unaligned = QuantizedGemvBind::decode(33, 1, QuantizedFormat::Q8_0, [1, 1, 1]);
    assert!(matches!(
        library::quantized_gemv(&unaligned, &[], &[], &mut []),
        Err(KernelBodyError::InvalidBind(message)) if message.contains("block aligned")
    ));

    let bind = QuantizedGemvBind::decode(32, 1, QuantizedFormat::Q5_0, [1, 1, 1]);
    let error = library::quantized_gemv(&bind, &[1.0; 32], &[0u8; 21], &mut [0.0])
        .expect_err("truncated packed block must fail closed");
    assert!(matches!(
        error,
        KernelBodyError::BufferTooShort {
            buffer: "packed_weight",
            required: 22,
            actual: 21
        }
    ));
}
