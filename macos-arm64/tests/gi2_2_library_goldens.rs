//! Numeric GI2-2 oracle coverage for the M2 library-v0 bodies.
//!
//! The fixtures are byte-for-byte copies of the read-only
//! `faber-prefill-oracle/testdata/gi2-2-op-goldens` tree.  These tests call the
//! parameterized host bodies directly and compare their f32 values, never
//! emitted MSL text.  Dense and attention fixtures remain in the copied tree
//! for their later M3/M4 owners; M2 covers the bodies that M2-U1 parameterized.

use faber_host_macos_arm64::device_descriptor::sha256_hex;
use faber_host_macos_arm64::kernel::library::{self, BindDescriptor, BindLayout};
use serde::Deserialize;

const RMS_NORM_GOLDEN: &str = include_str!("fixtures/gi2-2-op-goldens/rms_norm.json");
const RESIDUAL_GOLDEN: &str = include_str!("fixtures/gi2-2-op-goldens/residual.json");
const ROPE_GOLDEN: &str = include_str!("fixtures/gi2-2-op-goldens/rope.json");
const SWIGLU_GOLDEN: &str = include_str!("fixtures/gi2-2-op-goldens/swiglu.json");

// The GI2-2 producer's f32 comparison band is 1e-6.  A fixture may override
// it with a row-specific `max_abs_delta`; zero remains the exact-match path.
const GI2_2_F32_TOLERANCE: f32 = 1e-6;

#[derive(Debug, Deserialize)]
struct Golden {
    op: String,
    #[serde(default)]
    max_abs_delta: Option<f32>,
    inputs: Vec<Tensor>,
    expected_output: Tensor,
}

#[derive(Debug, Deserialize)]
struct Tensor {
    name: String,
    elements: usize,
    f32_le_hex: String,
    #[serde(default)]
    sha256: Option<String>,
}

fn golden(source: &str) -> Golden {
    serde_json::from_str(source).expect("valid gi2-2 golden fixture")
}

fn tensor_values(tensor: &Tensor) -> Vec<f32> {
    let hex: String = tensor
        .f32_le_hex
        .chars()
        .filter(|character| !character.is_ascii_whitespace())
        .collect();
    assert_eq!(
        hex.len(),
        tensor.elements * 8,
        "{} element count",
        tensor.name
    );
    assert!(hex.is_ascii(), "{} contains non-ASCII hex", tensor.name);
    let values = hex
        .as_bytes()
        .chunks_exact(8)
        .map(|chunk| {
            let bytes = chunk
                .chunks_exact(2)
                .map(|pair| {
                    u8::from_str_radix(std::str::from_utf8(pair).expect("ASCII hex"), 16)
                        .expect("f32 little-endian hex")
                })
                .collect::<Vec<_>>();
            let bytes: [u8; 4] = bytes.try_into().expect("four f32 bytes");
            f32::from_bits(u32::from_le_bytes(bytes))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        values.len(),
        tensor.elements,
        "{} decoded elements",
        tensor.name
    );
    if let Some(expected_hash) = tensor.sha256.as_deref() {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(
            sha256_hex(&bytes),
            expected_hash,
            "{} f32 payload hash",
            tensor.name
        );
    }
    values
}

fn input<'a>(golden: &'a Golden, name: &str) -> &'a Tensor {
    golden
        .inputs
        .iter()
        .find(|tensor| tensor.name == name)
        .unwrap_or_else(|| panic!("{} input missing from {} golden", name, golden.op))
}

fn assert_numeric(golden: &Golden, actual: &[f32]) {
    let expected = tensor_values(&golden.expected_output);
    assert_eq!(actual.len(), expected.len(), "{} output length", golden.op);
    let mut max_abs_delta = 0.0f32;
    for (index, (&observed, &reference)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            observed.is_finite(),
            "{} produced non-finite output at index {index}: {observed:?}",
            golden.op
        );
        assert!(
            reference.is_finite(),
            "{} fixture has non-finite output at index {index}: {reference:?}",
            golden.op
        );
        max_abs_delta = max_abs_delta.max((observed - reference).abs());
    }
    let allowed = golden.max_abs_delta.unwrap_or(GI2_2_F32_TOLERANCE);
    assert!(
        max_abs_delta <= allowed,
        "{} numeric mismatch: max_abs_delta={max_abs_delta:?}, allowed={allowed:?}",
        golden.op
    );
}

#[test]
fn rms_norm_golden_matches_parameterized_body() {
    let golden = golden(RMS_NORM_GOLDEN);
    let x = tensor_values(input(&golden, "x"));
    let weight = tensor_values(input(&golden, "weight"));
    let bind = BindDescriptor::row_major(vec![1, 960], [1, 1, 1]);
    let mut actual = vec![0.0; x.len()];

    library::rms(&bind, &x, &weight, &mut actual, 1e-5).expect("parameterized RMS body");
    assert_numeric(&golden, &actual);
}

#[test]
fn residual_golden_matches_parameterized_body() {
    let golden = golden(RESIDUAL_GOLDEN);
    let left = tensor_values(input(&golden, "a"));
    let right = tensor_values(input(&golden, "b"));
    let bind = BindDescriptor::row_major(vec![1, 960], [1, 1, 1]);
    let mut actual = vec![0.0; left.len()];

    library::residual(&bind, &left, &right, &mut actual).expect("parameterized residual body");
    assert_numeric(&golden, &actual);
}

#[test]
fn rope_golden_matches_parameterized_body() {
    let golden = golden(ROPE_GOLDEN);
    let input = tensor_values(input(&golden, "head"));
    let dim = 64usize;
    let position = 8.0f64;
    let freq_base = 100_000.0f64;
    let mut cos = Vec::with_capacity(dim / 2);
    let mut sin = Vec::with_capacity(dim / 2);
    for pair in 0..dim / 2 {
        let theta = position * freq_base.powf(-2.0 * pair as f64 / dim as f64);
        cos.push(theta.cos() as f32);
        sin.push(theta.sin() as f32);
    }
    let bind = BindDescriptor {
        dims: vec![1, dim as u64],
        strides: vec![dim as u64, 1],
        layout: BindLayout::RopeConsecutivePair {
            rotated_width: dim as u64,
        },
        grid: [1, 1, 1],
    };
    let mut actual = vec![0.0; input.len()];

    library::rope(&bind, &input, &cos, &sin, &mut actual).expect("parameterized RoPE body");
    assert_numeric(&golden, &actual);
}

#[test]
fn swiglu_golden_matches_parameterized_body() {
    let golden = golden(SWIGLU_GOLDEN);
    let gate = tensor_values(input(&golden, "gate"));
    let up = tensor_values(input(&golden, "up"));
    let bind = BindDescriptor::row_major(vec![1, 2560], [1, 1, 1]);
    let mut actual = vec![0.0; gate.len()];

    library::swiglu(&bind, &gate, &up, &mut actual).expect("parameterized SwiGLU body");
    assert_numeric(&golden, &actual);
}
