//! PB-8 GPU Q4_K llama.cpp-exactness pins — the three-leg proof, durable.
//!
//! Audit `3c2a4769` (P2): the PB-8 claim that the GPU Q4_K path is
//! llama.cpp-exact three ways (MSL-semantics sim vs GPU trace; independent
//! dequant vs llama-quantize F32; uploaded device bytes vs GGUF region) lived
//! only in mutable `/tmp` artifacts.  This test re-lands that evidence from
//! committed fixtures captured from the real model and the real GPU trace;
//! provenance, commands, and content hashes are frozen in
//! `fixtures/q4-k-exactness/q4_k_exactness.receipt.md`.

use faber_host_macos_arm64::kernel::library::{
    GemvKernel, QuantizedFormat, QuantizedGemvBind, dispatch_gemv,
};

const GGUF_COLUMNS: &[u8] = include_bytes!("fixtures/q4-k-exactness/q4k-cols62-69.gguf.bin");
const DEVICE_COLUMNS: &[u8] = include_bytes!("fixtures/q4-k-exactness/q4k-cols62-69.device.bin");
const HH_ROW0: &[u8] = include_bytes!("fixtures/q4-k-exactness/hh-row0.f32");
const GPU_ROW0: &[u8] = include_bytes!("fixtures/q4-k-exactness/gpu-down-row0-cols62-69.f32");
const LLAMA_F32: &[u8] = include_bytes!("fixtures/q4-k-exactness/llama-f32-cols62-69.f32");
const RECEIPT: &str = include_str!("fixtures/q4-k-exactness/q4_k_exactness.receipt.md");

/// Qwen2.5-0.5B `blk.2.ffn_down`: K = 4864 = 19 superblocks × 256.
const K: u64 = 4864;
/// Pinned columns 62–69 of the GEMV output (= GGUF rows 62–69).
const FIRST_COLUMN: usize = 62;
const COLUMNS: usize = 8;
/// 19 superblocks × 144 bytes per packed column.
const COLUMN_BYTES: usize = 19 * 144;

fn f32_le(bytes: &[u8], index: usize) -> f32 {
    let chunk = bytes
        .get(index * 4..index * 4 + 4)
        .unwrap_or_else(|| panic!("fixture f32 index {index} out of range"));
    f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
}

/// Leg 1 — MSL-semantics simulation reproduces the GPU trace values.
///
/// The hosts `gemv_q4_k` body is the committed mirror of the emitted MSL
/// chunk body (`radix … quantized_matmul.rs::q4_k_chunk_statements`): the
/// same per-element dequant formula and simdgroup-lane reduction shape.
/// Over the committed GGUF columns and the committed trace input row, its
/// output must match the committed GPU-observed values within the
/// accumulation-order band (observed max deviation at capture: 6.1e-4 at
/// column 62; the band allows lane-order drift without hiding a real
/// dequant divergence, which is orders of magnitude larger — the PB-5b
/// oracle bug showed up as ~947.0 absolute).
#[test]
fn msl_semantics_gemv_reproduces_gpu_trace_values() {
    let activation: Vec<f32> = (0..K as usize)
        .map(|index| f32_le(HH_ROW0, index))
        .collect();
    let bind = QuantizedGemvBind::decode(K, COLUMNS as u64, QuantizedFormat::Q4K, [8, 1, 1]);
    let mut output = vec![f32::NAN; COLUMNS];
    dispatch_gemv(
        GemvKernel::Quantized,
        &bind,
        &activation,
        GGUF_COLUMNS,
        &mut output,
    )
    .expect("Q4_K GEMV over pinned trace fixtures");

    for (offset, computed) in output.iter().enumerate() {
        let gpu = f32_le(GPU_ROW0, offset);
        let deviation = (computed - gpu).abs();
        assert!(
            deviation <= 2.0e-3,
            "column {}: MSL-semantics GEMV {computed} vs GPU trace {gpu} (Δ {deviation})",
            FIRST_COLUMN + offset,
        );
    }
}

/// Leg 2 — independent dequant is bit-exact against llama-quantize F32.
///
/// A unit activation at element `i` makes the GEMV output exactly the
/// dequantized weight at that element, so f32 bit equality against the
/// committed llama-quantize F32 reference is the 0.0-diff claim pinned
/// element-for-element (8 columns × 4864 elements).
#[test]
fn independent_dequant_is_bit_exact_against_llama_quantize_f32() {
    for block in 0..19usize {
        // Column c's superblock `block` sits at c·2736 + block·144; the
        // strided bind reads all 8 columns from one block-relative slice.
        let bind = QuantizedGemvBind::strided(
            256,
            COLUMNS as u64,
            1,
            1,
            COLUMN_BYTES as u64,
            QuantizedFormat::Q4K,
            [8, 1, 1],
        );
        let packed = &GGUF_COLUMNS[block * 144..];
        for element in 0..256usize {
            let mut activation = vec![0.0f32; 256];
            activation[element] = 1.0;
            let mut output = vec![f32::NAN; COLUMNS];
            dispatch_gemv(
                GemvKernel::Quantized,
                &bind,
                &activation,
                packed,
                &mut output,
            )
            .expect("Q4_K dequant probe GEMV");
            for column in 0..COLUMNS {
                let reference = f32_le(LLAMA_F32, column * 4864 + block * 256 + element);
                assert_eq!(
                    output[column].to_bits(),
                    reference.to_bits(),
                    "column {} element {}: dequant bits vs llama-quantize F32",
                    FIRST_COLUMN + column,
                    block * 256 + element,
                );
            }
        }
    }
}

/// Leg 3 — the uploaded device bytes are byte-identical to the GGUF region.
///
/// The device capture (instrumented-host uploaded weight buffer) and the
/// GGUF-region capture are committed as separate fixtures; equality here
/// pins the identity the GPU actually consumed the model's packed bytes.
/// The receipt freezes the full-tensor check (2451456 bytes, both sha256
/// `92ecbb72…`).
#[test]
fn uploaded_device_bytes_match_gguf_region() {
    assert_eq!(GGUF_COLUMNS.len(), COLUMNS * COLUMN_BYTES);
    assert_eq!(DEVICE_COLUMNS.len(), GGUF_COLUMNS.len());
    assert_eq!(
        GGUF_COLUMNS, DEVICE_COLUMNS,
        "device-captured packed bytes must equal the GGUF region bytes"
    );
    assert!(RECEIPT.contains("92ecbb7216a294133f4a41a34891a3b276aeac6ccfb8c924355c39f66b18b4c1"));
}
