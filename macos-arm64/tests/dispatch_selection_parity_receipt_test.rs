//! M4-U2b: dispatch-selection parity receipts for the decode projection seam.
//!
//! The batched body remains the numeric baseline.  The decode GEMV body is
//! allowed to differ only within the KV-A cross-path logits ruling, and the
//! receipt keeps the last-row comparison explicit for each accepted selector.
//! This remains a focused host test: it exercises the production Metal-library
//! runtime bridge and does not change dispatch policy.

use faber_host_macos_arm64::kernel::library::{
    dispatch_gemv, residual, rms, select_decode_gemv, select_residual_rms_norm, BindDescriptor,
    BindLayout, GemvKernel, KernelBodyError, QkvProjectionBind, QkvProjectionLayout,
    QkvProjectionWeight, QuantizedFormat, QuantizedGemvBind,
};
use faber_host_macos_arm64::kernel::library_runtime::{
    dispatch_metal_library, library_family_msl, LibraryFamilyMslFacts, MetalLibraryDispatch,
};
use faber_host_macos_arm64::MetalHostSession;

const K: usize = 32;
const KV_A_CAPACITY_ROWS: usize = 4;
const PREFIX_BEFORE: usize = KV_A_CAPACITY_ROWS - 1;
const QUERY_ROWS: usize = 1;

// KV-A's authorized cross-path logits envelope.
const KV_A_MAX_ULP: u32 = 1024;
const KV_A_MAX_RELATIVE_ERROR: f32 = 1.0e-4;

#[derive(Debug, Clone, Copy)]
struct Rung {
    name: &'static str,
    q_per_kv: usize,
    kv_heads: usize,
}

const RUNGS: [Rung; 2] = [
    Rung {
        name: "smollm2-360m",
        q_per_kv: 3,
        kv_heads: 5,
    },
    Rung {
        name: "qwen2.5-0.5b",
        q_per_kv: 7,
        kv_heads: 2,
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelectedBody {
    BatchedGemm,
    DecodeGemv,
}

impl SelectedBody {
    fn spelling(self) -> &'static str {
        match self {
            Self::BatchedGemm => "batched_gemm",
            Self::DecodeGemv => "decode_gemv",
        }
    }
}

#[derive(Debug, Clone)]
struct ProjectionFixture {
    rows: usize,
    columns: usize,
    activation: Vec<f32>,
    packed_weight: Vec<u8>,
}

impl ProjectionFixture {
    fn for_rung(rung: Rung) -> Self {
        let columns = rung.q_per_kv * rung.kv_heads;
        let activation = (0..KV_A_CAPACITY_ROWS * K)
            .map(|index| {
                let row = index / K;
                let element = index % K;
                0.125 + row as f32 * 0.03125 + element as f32 * 0.001
            })
            .collect();
        let packed_weight = (0..columns)
            .flat_map(|column| q8_0_block(column).into_iter())
            .collect();
        Self {
            rows: KV_A_CAPACITY_ROWS,
            columns,
            activation,
            packed_weight,
        }
    }
}

#[derive(Debug, Clone)]
struct QkvProjectionFixture {
    rows: usize,
    hidden: usize,
    q_columns: usize,
    kv_columns: usize,
    activation: Vec<f32>,
    weights: [Vec<u8>; 3],
}

impl QkvProjectionFixture {
    fn for_rung(rung: Rung) -> Self {
        let head_dim = 32;
        let q_columns = rung.q_per_kv * rung.kv_heads * head_dim;
        let kv_columns = rung.kv_heads * head_dim;
        let hidden = q_columns;
        let weights = [
            q8_0_columns(q_columns, hidden, 1),
            q8_0_columns(kv_columns, hidden, 7),
            q8_0_columns(kv_columns, hidden, 13),
        ];
        let activation = (0..KV_A_CAPACITY_ROWS * hidden)
            .map(|index| {
                let row = index / hidden;
                let element = index % hidden;
                0.0625 + row as f32 * 0.015625 + element as f32 * 0.00025
            })
            .collect();
        Self {
            rows: KV_A_CAPACITY_ROWS,
            hidden,
            q_columns,
            kv_columns,
            activation,
            weights,
        }
    }

    fn bind(&self, rung: Rung) -> QkvProjectionBind {
        QkvProjectionBind::grouped(
            self.rows as u64,
            self.hidden as u64,
            rung.kv_heads as u64,
            rung.q_per_kv as u64,
            32,
            [self.q_columns as u32, 1, 1],
        )
    }
}

fn q8_0_columns(columns: usize, hidden: usize, salt: i32) -> Vec<u8> {
    let blocks = hidden / K;
    (0..columns)
        .flat_map(|column| {
            (0..blocks).flat_map(move |block| {
                let mut bytes = vec![0u8; 34];
                bytes[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
                for element in 0..K {
                    let value =
                        ((column as i32 * 5 + block as i32 * 11 + element as i32 * 3 + salt) % 31)
                            - 15;
                    bytes[2 + element] = (value as i8) as u8;
                }
                bytes
            })
        })
        .collect()
}

fn q8_0_column_value(packed: &[u8], column: usize, hidden: usize, element: usize) -> f32 {
    let blocks = hidden / K;
    let offset = (column * blocks + element / K) * 34;
    assert_eq!(
        u16::from_le_bytes([packed[offset], packed[offset + 1]]),
        0x3c00
    );
    i8::from_ne_bytes([packed[offset + 2 + element % K]]) as f32
}

fn qkv_batched_baseline(fixture: &QkvProjectionFixture, weights: &[Vec<u8>; 3]) -> [Vec<f32>; 3] {
    [
        (0..fixture.rows * fixture.q_columns)
            .map(|index| {
                let row = index / fixture.q_columns;
                let column = index % fixture.q_columns;
                (0..fixture.hidden)
                    .map(|element| {
                        fixture.activation[row * fixture.hidden + element]
                            * q8_0_column_value(&weights[0], column, fixture.hidden, element)
                    })
                    .sum()
            })
            .collect(),
        (0..fixture.rows * fixture.kv_columns)
            .map(|index| {
                let row = index / fixture.kv_columns;
                let column = index % fixture.kv_columns;
                (0..fixture.hidden)
                    .map(|element| {
                        fixture.activation[row * fixture.hidden + element]
                            * q8_0_column_value(&weights[1], column, fixture.hidden, element)
                    })
                    .sum()
            })
            .collect(),
        (0..fixture.rows * fixture.kv_columns)
            .map(|index| {
                let row = index / fixture.kv_columns;
                let column = index % fixture.kv_columns;
                (0..fixture.hidden)
                    .map(|element| {
                        fixture.activation[row * fixture.hidden + element]
                            * q8_0_column_value(&weights[2], column, fixture.hidden, element)
                    })
                    .sum()
            })
            .collect(),
    ]
}

fn qkv_batched_last_rows(fixture: &QkvProjectionFixture, outputs: &[Vec<f32>; 3]) -> [Vec<f32>; 3] {
    let row = fixture.rows - 1;
    [
        outputs[0][row * fixture.q_columns..(row + 1) * fixture.q_columns].to_vec(),
        outputs[1][row * fixture.kv_columns..(row + 1) * fixture.kv_columns].to_vec(),
        outputs[2][row * fixture.kv_columns..(row + 1) * fixture.kv_columns].to_vec(),
    ]
}

fn qkv_logical_last_rows(
    rung: Rung,
    fixture: &QkvProjectionFixture,
    outputs: &[Vec<f32>; 3],
) -> [Vec<f32>; 3] {
    let row = fixture.rows - 1;
    let head_dim = 32;
    let mut q = Vec::with_capacity(fixture.q_columns);
    for group in 0..rung.kv_heads {
        for query_head in 0..rung.q_per_kv {
            for dimension in 0..head_dim {
                let offset = (group * rung.q_per_kv + query_head) * fixture.rows * head_dim
                    + row * head_dim
                    + dimension;
                q.push(outputs[0][offset]);
            }
        }
    }
    let kv = |values: &Vec<f32>| {
        let mut row_values = Vec::with_capacity(fixture.kv_columns);
        for group in 0..rung.kv_heads {
            for dimension in 0..head_dim {
                row_values
                    .push(values[group * fixture.rows * head_dim + row * head_dim + dimension]);
            }
        }
        row_values
    };
    [q, kv(&outputs[1]), kv(&outputs[2])]
}

/// One reusable, printed parity record for one selected body.
#[derive(Debug)]
struct ParityReceipt {
    rung: &'static str,
    library_entry: Option<&'static str>,
    decode_gemv: u32,
    body: SelectedBody,
    last_row: Vec<f32>,
    baseline_last_row: Vec<f32>,
    residuals: Vec<f32>,
    max_ulp: u32,
    max_relative_error: f32,
}

#[derive(Debug, Clone, Copy)]
struct KvaRowBounds {
    capacity_rows: usize,
    prefix_before: usize,
    query_rows: usize,
}

impl KvaRowBounds {
    fn last_row(self) -> usize {
        self.prefix_before + self.query_rows - 1
    }

    fn assert_valid(self, fixture: &ProjectionFixture) {
        let valid_len_after = self
            .prefix_before
            .checked_add(self.query_rows)
            .expect("KV-A valid length must not overflow");
        assert!(
            valid_len_after <= self.capacity_rows,
            "KV-A valid length {valid_len_after} exceeds capacity {}",
            self.capacity_rows
        );
        assert!(
            self.last_row() < fixture.rows,
            "last row {} exceeds fixture rows {}",
            self.last_row(),
            fixture.rows
        );
        let row_end = (self.last_row() + 1)
            .checked_mul(K)
            .expect("KV-A activation row span must not overflow");
        assert!(
            row_end <= fixture.activation.len(),
            "last-row activation end {row_end} exceeds activation length {}",
            fixture.activation.len()
        );
    }
}

fn q8_0_block(column: usize) -> Vec<u8> {
    let mut block = vec![0u8; 34];
    // f16(1.0), so the baseline and the host GEMV body consume the same
    // packed Q8_0 values without introducing a second scale oracle.
    block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
    for element in 0..K {
        let q = ((column as i32 * 5 + element as i32 * 3) % 31) - 15;
        block[2 + element] = (q as i8) as u8;
    }
    block
}

fn q8_0_weight(packed_weight: &[u8], column: usize, element: usize) -> f32 {
    let block = &packed_weight[column * 34..(column + 1) * 34];
    assert_eq!(
        u16::from_le_bytes([block[0], block[1]]),
        0x3c00,
        "fixture scale drifted from f16(1.0)"
    );
    i8::from_ne_bytes([block[2 + element]]) as f32
}

/// Independent batched reference: decode each column into f32 weights first,
/// then evaluate every row.  The selected batched body below uses the opposite
/// loop nesting and reads packed bytes at use, keeping this an actual oracle.
fn batched_baseline(fixture: &ProjectionFixture) -> Vec<f32> {
    let weights = (0..fixture.columns)
        .map(|column| {
            (0..K)
                .map(|element| q8_0_weight(&fixture.packed_weight, column, element))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut output = vec![0.0f32; fixture.rows * fixture.columns];
    for row in 0..fixture.rows {
        let activation = &fixture.activation[row * K..(row + 1) * K];
        for (column, weights) in weights.iter().enumerate() {
            output[row * fixture.columns + column] = activation
                .iter()
                .zip(weights)
                .map(|(input, weight)| input * weight)
                .sum();
        }
    }
    output
}

/// The selected GEMM-side body.  It is intentionally separate from the
/// baseline so a selector cannot pass by comparing a path with itself.
fn run_batched_gemm(fixture: &ProjectionFixture) -> Vec<f32> {
    let mut output = vec![0.0f32; fixture.rows * fixture.columns];
    for column in 0..fixture.columns {
        for row in 0..fixture.rows {
            let mut sum = 0.0f32;
            for element in 0..K {
                sum += fixture.activation[row * K + element]
                    * q8_0_weight(&fixture.packed_weight, column, element);
            }
            output[row * fixture.columns + column] = sum;
        }
    }
    output
}

fn run_decode_gemv(fixture: &ProjectionFixture, last_row: usize) -> Vec<f32> {
    let bind = QuantizedGemvBind::decode(
        K as u64,
        fixture.columns as u64,
        QuantizedFormat::Q8_0,
        [fixture.columns as u32, 1, 1],
    );
    let activation = &fixture.activation[last_row * K..(last_row + 1) * K];
    let mut output = vec![f32::NAN; fixture.columns];
    dispatch_gemv(
        GemvKernel::Quantized,
        &bind,
        activation,
        &fixture.packed_weight,
        &mut output,
    )
    .expect("selected decode GEMV must stay inside its bind");
    output
}

fn selected_body(
    library_entry: Option<&'static str>,
    decode_gemv: u32,
) -> Result<SelectedBody, KernelBodyError> {
    Ok(match select_decode_gemv(library_entry, decode_gemv)? {
        Some(_) => SelectedBody::DecodeGemv,
        None => SelectedBody::BatchedGemm,
    })
}

fn parity_receipt(
    rung: Rung,
    fixture: &ProjectionFixture,
    library_entry: Option<&'static str>,
    decode_gemv: u32,
    bounds: KvaRowBounds,
) -> ParityReceipt {
    bounds.assert_valid(fixture);
    let baseline = batched_baseline(fixture);
    let baseline_last_row = baseline
        [bounds.last_row() * fixture.columns..(bounds.last_row() + 1) * fixture.columns]
        .to_vec();
    let body = selected_body(library_entry, decode_gemv).expect("selection must be admitted");
    let last_row = match body {
        SelectedBody::BatchedGemm => {
            let output = run_batched_gemm(fixture);
            assert_eq!(
                output.len(),
                fixture.rows * fixture.columns,
                "batched GEMM output must cover only the admitted fixture rows"
            );
            output[bounds.last_row() * fixture.columns..(bounds.last_row() + 1) * fixture.columns]
                .to_vec()
        }
        SelectedBody::DecodeGemv => run_decode_gemv(fixture, bounds.last_row()),
    };
    assert_eq!(last_row.len(), baseline_last_row.len());

    let residuals = last_row
        .iter()
        .zip(&baseline_last_row)
        .map(|(actual, expected)| (actual - expected).abs())
        .collect::<Vec<_>>();
    let mut max_ulp = 0;
    let mut max_relative_error = 0.0f32;
    for (actual, expected) in last_row.iter().zip(&baseline_last_row) {
        assert!(
            actual.is_finite(),
            "selected body produced non-finite output"
        );
        assert!(expected.is_finite(), "baseline produced non-finite output");
        max_ulp = max_ulp.max(ulp_distance(*actual, *expected));
        let denominator = expected.abs().max(f32::MIN_POSITIVE);
        max_relative_error = max_relative_error.max((actual - expected).abs() / denominator);
    }
    assert_eq!(argmax(&last_row), argmax(&baseline_last_row));
    assert!(
        max_ulp <= KV_A_MAX_ULP,
        "{} {} uniform={} exceeded KV-A ULP bound: {} > {}",
        rung.name,
        body.spelling(),
        decode_gemv,
        max_ulp,
        KV_A_MAX_ULP
    );
    assert!(
        max_relative_error <= KV_A_MAX_RELATIVE_ERROR,
        "{} {} uniform={} exceeded KV-A relative bound: {} > {}",
        rung.name,
        body.spelling(),
        decode_gemv,
        max_relative_error,
        KV_A_MAX_RELATIVE_ERROR
    );

    let receipt = ParityReceipt {
        rung: rung.name,
        library_entry,
        decode_gemv,
        body,
        last_row,
        baseline_last_row,
        residuals,
        max_ulp,
        max_relative_error,
    };
    eprintln!(
        "M4-U2b parity receipt: rung={} entry={:?} decode_gemv={} body={} last_row={:?} baseline={:?} residuals={:?} max_ulp={} max_relative_error={:.3e} bounds=(ulp<={},relative<={:.3e})",
        receipt.rung,
        receipt.library_entry,
        receipt.decode_gemv,
        receipt.body.spelling(),
        receipt.last_row,
        receipt.baseline_last_row,
        receipt.residuals,
        receipt.max_ulp,
        receipt.max_relative_error,
        KV_A_MAX_ULP,
        KV_A_MAX_RELATIVE_ERROR,
    );
    receipt
}

fn ulp_distance(left: f32, right: f32) -> u32 {
    fn ordered_bits(value: f32) -> u32 {
        let bits = value.to_bits();
        if bits & 0x8000_0000 != 0 {
            !bits
        } else {
            bits | 0x8000_0000
        }
    }
    ordered_bits(left).abs_diff(ordered_bits(right))
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .expect("projection must have at least one output column")
}

#[test]
fn dispatch_selection_parity_receipt_covers_both_rungs_and_paths() {
    for rung in RUNGS {
        let fixture = ProjectionFixture::for_rung(rung);
        let bounds = KvaRowBounds {
            capacity_rows: KV_A_CAPACITY_ROWS,
            prefix_before: PREFIX_BEFORE,
            query_rows: QUERY_ROWS,
        };
        let mut saw_gemm = false;
        let mut saw_gemv = false;

        // These are every currently live selector spelling that can flip
        // between the batched and decode bodies on the uniform.
        for entry in [
            None,
            Some("QkvProjection"),
            Some("OutputProjection"),
            Some("SwiGlu"),
        ] {
            for uniform in [0, 1] {
                let receipt = parity_receipt(rung, &fixture, entry, uniform, bounds);
                saw_gemm |= receipt.body == SelectedBody::BatchedGemm;
                saw_gemv |= receipt.body == SelectedBody::DecodeGemv;
            }
        }

        // The generic executor spellings are live too, but each is pinned to
        // one matching uniform; the flipped fact is intentionally tested below
        // as fail-closed drift rather than silently changing policy.
        let gemm = parity_receipt(rung, &fixture, Some("quantized_gemm"), 0, bounds);
        let gemv = parity_receipt(rung, &fixture, Some("quantized_gemv"), 1, bounds);
        saw_gemm |= gemm.body == SelectedBody::BatchedGemm;
        saw_gemv |= gemv.body == SelectedBody::DecodeGemv;

        assert!(saw_gemm, "{} did not exercise batched GEMM", rung.name);
        assert!(saw_gemv, "{} did not exercise decode GEMV", rung.name);
    }
}

#[test]
fn flipping_decode_uniform_swaps_bodies_without_leaving_kva_bounds() {
    for rung in RUNGS {
        let fixture = ProjectionFixture::for_rung(rung);
        let bounds = KvaRowBounds {
            capacity_rows: KV_A_CAPACITY_ROWS,
            prefix_before: PREFIX_BEFORE,
            query_rows: QUERY_ROWS,
        };
        for entry in [
            None,
            Some("QkvProjection"),
            Some("OutputProjection"),
            Some("SwiGlu"),
        ] {
            let prefill = parity_receipt(rung, &fixture, entry, 0, bounds);
            let decode = parity_receipt(rung, &fixture, entry, 1, bounds);
            assert_eq!(prefill.body, SelectedBody::BatchedGemm);
            assert_eq!(decode.body, SelectedBody::DecodeGemv);
            assert_ne!(prefill.body, decode.body);
            assert_eq!(prefill.last_row.len(), fixture.columns);
            assert_eq!(decode.last_row.len(), fixture.columns);
        }
    }
}

#[test]
fn qkv_single_body_selection_parity_receipt_covers_both_rungs_and_uniforms() {
    for rung in RUNGS {
        let fixture = QkvProjectionFixture::for_rung(rung);
        let bind = fixture.bind(rung);
        let baseline = qkv_batched_baseline(&fixture, &fixture.weights);
        let baseline_last = qkv_batched_last_rows(&fixture, &baseline);
        for decode_gemv in [0, 1] {
            let mut outputs = [
                vec![f32::NAN; fixture.q_columns * fixture.rows],
                vec![f32::NAN; fixture.kv_columns * fixture.rows],
                vec![f32::NAN; fixture.kv_columns * fixture.rows],
            ];
            let weights = [
                QkvProjectionWeight::Quantized {
                    bind: QuantizedGemvBind::decode(
                        fixture.hidden as u64,
                        fixture.q_columns as u64,
                        QuantizedFormat::Q8_0,
                        [fixture.q_columns as u32, 1, 1],
                    ),
                    packed: &fixture.weights[0],
                },
                QkvProjectionWeight::Quantized {
                    bind: QuantizedGemvBind::decode(
                        fixture.hidden as u64,
                        fixture.kv_columns as u64,
                        QuantizedFormat::Q8_0,
                        [fixture.kv_columns as u32, 1, 1],
                    ),
                    packed: &fixture.weights[1],
                },
                QkvProjectionWeight::Quantized {
                    bind: QuantizedGemvBind::decode(
                        fixture.hidden as u64,
                        fixture.kv_columns as u64,
                        QuantizedFormat::Q8_0,
                        [fixture.kv_columns as u32, 1, 1],
                    ),
                    packed: &fixture.weights[2],
                },
            ];
            let (q_output, rest) = outputs.split_at_mut(1);
            let (k_output, v_output) = rest.split_at_mut(1);
            dispatch_metal_library(MetalLibraryDispatch::QkvProjection {
                library_entry: Some("QkvProjection"),
                decode_gemv,
                layout: QkvProjectionLayout::Grouped,
                bind: &bind,
                activation: &fixture.activation,
                weights,
                biases: [None, None, None],
                rope: None,
                outputs: [&mut q_output[0], &mut k_output[0], &mut v_output[0]],
            })
            .expect("selected QKV body");
            let actual_last = qkv_logical_last_rows(rung, &fixture, &outputs);
            for (actual, expected) in actual_last.iter().zip(&baseline_last) {
                assert_eq!(actual.len(), expected.len());
                let max_ulp = actual
                    .iter()
                    .zip(expected)
                    .map(|(actual, expected)| ulp_distance(*actual, *expected))
                    .max()
                    .unwrap_or(0);
                let max_relative_error = actual
                    .iter()
                    .zip(expected)
                    .map(|(actual, expected)| {
                        (actual - expected).abs() / expected.abs().max(f32::MIN_POSITIVE)
                    })
                    .fold(0.0f32, f32::max);
                assert!(actual.iter().all(|value| value.is_finite()));
                assert!(expected.iter().all(|value| value.is_finite()));
                assert_eq!(argmax(actual), argmax(expected));
                assert!(
                    max_ulp <= KV_A_MAX_ULP,
                    "rung={} decode_gemv={} max_ulp={} actual={:?} expected={:?}",
                    rung.name,
                    decode_gemv,
                    max_ulp,
                    actual,
                    expected
                );
                assert!(
                    max_relative_error <= KV_A_MAX_RELATIVE_ERROR,
                    "rung={} decode_gemv={} max_relative_error={} actual={:?} expected={:?}",
                    rung.name,
                    decode_gemv,
                    max_relative_error,
                    actual,
                    expected
                );
                eprintln!(
                    "M4-U2b parity receipt: runtime=metal-library rung={} entry=Some(QkvProjection) decode_gemv={} max_ulp={} max_relative_error={:.3e} rows={} qkv=single-body",
                    rung.name,
                    decode_gemv,
                    max_ulp,
                    max_relative_error,
                    fixture.rows,
                );
            }
        }
    }
    assert!(select_qkv_projection_for_test_drift().is_err());
}

fn select_qkv_projection_for_test_drift() -> Result<(), KernelBodyError> {
    dispatch_metal_library(MetalLibraryDispatch::QkvProjection {
        library_entry: Some("QkvProjection"),
        decode_gemv: 2,
        layout: QkvProjectionLayout::Grouped,
        bind: &QkvProjectionBind::grouped(1, 32, 1, 1, 32, [1, 1, 1]),
        activation: &[],
        weights: [
            QkvProjectionWeight::Dense(&[]),
            QkvProjectionWeight::Dense(&[]),
            QkvProjectionWeight::Dense(&[]),
        ],
        biases: [None, None, None],
        rope: None,
        outputs: [&mut [], &mut [], &mut []],
    })
}

#[test]
fn dispatch_selection_fails_closed_on_uniform_or_vocabulary_drift() {
    for entry in [
        Some("quantized_gemv"),
        Some("quantized_gemm"),
        Some("QkvProjection"),
        Some("OutputProjection"),
        Some("SwiGlu"),
        None,
    ] {
        assert!(
            select_decode_gemv(entry, 2).is_err(),
            "uniform drift must fail closed for {entry:?}"
        );
    }
    assert!(
        select_decode_gemv(Some("quantized_gemv"), 0).is_err(),
        "GEMV entry with prefill uniform must fail closed"
    );
    assert!(
        select_decode_gemv(Some("quantized_gemm"), 1).is_err(),
        "GEMM entry with decode uniform must fail closed"
    );
    assert!(
        select_decode_gemv(Some("unowned_future_projection"), 1).is_err(),
        "unowned vocabulary must fail closed"
    );
}

#[derive(Debug, Clone)]
struct ResidualRmsFixture {
    rows: usize,
    width: usize,
    stride: usize,
    residual: Vec<f32>,
    skip: Vec<f32>,
    gamma: Vec<f32>,
}

impl ResidualRmsFixture {
    fn for_rung(rung: Rung, strided: bool) -> Self {
        let rows = KV_A_CAPACITY_ROWS;
        let width = rung.q_per_kv * rung.kv_heads * 4;
        let stride = if strided { width + 3 } else { width };
        let mut residual = vec![0.0; rows * stride];
        let mut skip = vec![0.0; rows * stride];
        for row in 0..rows {
            for column in 0..width {
                let offset = row * stride + column;
                residual[offset] = 0.125 + row as f32 * 0.03125 + column as f32 * 0.001;
                skip[offset] = -0.25 + row as f32 * 0.017 + column as f32 * 0.0005;
            }
        }
        let gamma = (0..width)
            .map(|column| 0.75 + (column % 7) as f32 * 0.025)
            .collect();
        Self {
            rows,
            width,
            stride,
            residual,
            skip,
            gamma,
        }
    }

    fn bind(&self, strided: bool) -> BindDescriptor {
        if strided {
            BindDescriptor::strided(
                vec![self.rows as u64, self.width as u64],
                vec![self.stride as u64, 1],
                [self.width as u32, 1, 1],
            )
        } else {
            BindDescriptor::row_major(
                vec![self.rows as u64, self.width as u64],
                [self.width as u32, 1, 1],
            )
        }
    }
}

fn residual_rms_logical_row(values: &[f32], row: usize, width: usize, stride: usize) -> Vec<f32> {
    values[row * stride..row * stride + width].to_vec()
}

#[test]
fn residual_rms_norm_selection_parity_receipt_covers_both_rungs_and_bind_paths() {
    for rung in RUNGS {
        for strided in [false, true] {
            let fixture = ResidualRmsFixture::for_rung(rung, strided);
            let bind = fixture.bind(strided);
            let mut composed_residual = vec![f32::NAN; fixture.rows * fixture.stride];
            residual(
                &bind,
                &fixture.residual,
                &fixture.skip,
                &mut composed_residual,
            )
            .expect("composed residual baseline");
            let mut baseline = vec![f32::NAN; fixture.rows * fixture.stride];
            rms(
                &bind,
                &composed_residual,
                &fixture.gamma,
                &mut baseline,
                1e-5,
            )
            .expect("composed RMS baseline");

            let mut actual = vec![f32::NAN; fixture.rows * fixture.stride];
            dispatch_metal_library(MetalLibraryDispatch::ResidualRmsNorm {
                library_entry: Some("ResidualRmsNorm"),
                layout: bind.layout,
                bind: &bind,
                residual: &fixture.residual,
                skip: &fixture.skip,
                gamma: &fixture.gamma,
                output: &mut actual,
                epsilon: 1e-5,
            })
            .expect("runtime ResidualRmsNorm body");

            let last_row = PREFIX_BEFORE;
            let actual_last =
                residual_rms_logical_row(&actual, last_row, fixture.width, fixture.stride);
            let baseline_last =
                residual_rms_logical_row(&baseline, last_row, fixture.width, fixture.stride);
            let residuals = actual_last
                .iter()
                .zip(&baseline_last)
                .map(|(actual, expected)| (actual - expected).abs())
                .collect::<Vec<_>>();
            let max_ulp = actual_last
                .iter()
                .zip(&baseline_last)
                .map(|(actual, expected)| ulp_distance(*actual, *expected))
                .max()
                .unwrap_or(0);
            assert_eq!(argmax(&actual_last), argmax(&baseline_last));
            assert_eq!(max_ulp, 0, "ResidualRmsNorm arithmetic order changed");
            assert!(actual_last.iter().all(|value| value.is_finite()));
            assert!(baseline_last.iter().all(|value| value.is_finite()));
            eprintln!(
                "M4-U2b parity receipt: runtime=metal-library rung={} entry=Some(ResidualRmsNorm) layout={} last_row={:?} baseline={:?} residuals={:?} max_ulp={} rms=1 launch/layer",
                rung.name,
                if strided { "strided" } else { "row_major" },
                actual_last,
                baseline_last,
                residuals,
                max_ulp,
            );
        }
    }
    assert!(select_residual_rms_norm(Some("ResidualRmsNorm"), BindLayout::Flat).is_err());
    assert!(select_residual_rms_norm(Some("rms"), BindLayout::RowMajor).is_err());
}

#[test]
fn fused_library_runtime_reaches_real_metal_entries() {
    // The host package builds on non-macOS for CUDA lanes.  A missing Metal
    // device is an environmental skip, not a green claim about device work.
    let Ok(mut session) = MetalHostSession::try_open() else {
        return;
    };

    let facts = LibraryFamilyMslFacts {
        rows: 2,
        hidden: 8,
        kv_heads: 1,
        q_per_kv: 2,
        head_dim: 4,
        epsilon: 1.0e-5,
    };
    let module_image = library_family_msl(&facts).expect("mint fused library Metal module");
    let module = session
        .load_module(module_image.as_bytes())
        .expect("load fused library Metal module");

    let activation: Vec<f32> = (0..(facts.rows * facts.hidden) as usize)
        .map(|index| 0.125 + index as f32 * 0.007)
        .collect();
    let q_width = facts.q_width() as usize;
    let kv_width = facts.kv_width() as usize;
    let q_weight: Vec<f32> = (0..q_width * facts.hidden as usize)
        .map(|index| 0.01 + (index % 13) as f32 * 0.002)
        .collect();
    let k_weight: Vec<f32> = (0..kv_width * facts.hidden as usize)
        .map(|index| 0.02 + (index % 11) as f32 * 0.003)
        .collect();
    let v_weight: Vec<f32> = (0..kv_width * facts.hidden as usize)
        .map(|index| 0.03 + (index % 7) as f32 * 0.004)
        .collect();
    let q_bind = QkvProjectionBind::grouped(
        facts.rows,
        facts.hidden,
        facts.kv_heads,
        facts.q_per_kv,
        facts.head_dim,
        [q_width as u32 * facts.rows as u32, 1, 1],
    );
    let mut expected_q = vec![f32::NAN; facts.rows as usize * q_width];
    let mut expected_k = vec![f32::NAN; facts.rows as usize * kv_width];
    let mut expected_v = vec![f32::NAN; facts.rows as usize * kv_width];
    dispatch_metal_library(MetalLibraryDispatch::QkvProjection {
        library_entry: Some("QkvProjection"),
        decode_gemv: 0,
        layout: QkvProjectionLayout::Grouped,
        bind: &q_bind,
        activation: &activation,
        weights: [
            QkvProjectionWeight::Dense(&q_weight),
            QkvProjectionWeight::Dense(&k_weight),
            QkvProjectionWeight::Dense(&v_weight),
        ],
        biases: [None, None, None],
        rope: None,
        outputs: [&mut expected_q, &mut expected_k, &mut expected_v],
    })
    .expect("CPU runtime bridge QKV parity");

    let activation_buffer = session
        .alloc_bytes(activation.len() * std::mem::size_of::<f32>())
        .expect("allocate activation");
    let q_weight_buffer = session
        .alloc_bytes(q_weight.len() * std::mem::size_of::<f32>())
        .expect("allocate Q weight");
    let k_weight_buffer = session
        .alloc_bytes(k_weight.len() * std::mem::size_of::<f32>())
        .expect("allocate K weight");
    let v_weight_buffer = session
        .alloc_bytes(v_weight.len() * std::mem::size_of::<f32>())
        .expect("allocate V weight");
    let q_output_buffer = session
        .alloc_bytes(expected_q.len() * std::mem::size_of::<f32>())
        .expect("allocate Q output");
    let k_output_buffer = session
        .alloc_bytes(expected_k.len() * std::mem::size_of::<f32>())
        .expect("allocate K output");
    let v_output_buffer = session
        .alloc_bytes(expected_v.len() * std::mem::size_of::<f32>())
        .expect("allocate V output");
    session
        .copy_in_f32(activation_buffer, &activation)
        .expect("upload activation");
    session
        .copy_in_f32(q_weight_buffer, &q_weight)
        .expect("upload Q weight");
    session
        .copy_in_f32(k_weight_buffer, &k_weight)
        .expect("upload K weight");
    session
        .copy_in_f32(v_weight_buffer, &v_weight)
        .expect("upload V weight");
    session
        .launch_kernel_3d(
            module,
            "QkvProjection",
            &[
                activation_buffer,
                q_weight_buffer,
                k_weight_buffer,
                v_weight_buffer,
                q_output_buffer,
                k_output_buffer,
                v_output_buffer,
            ],
            q_width as u32 * facts.rows as u32,
            1,
            1,
            1,
            1,
            1,
        )
        .expect("encode QkvProjection runtime entry");
    session.sync().expect("submit QkvProjection runtime entry");
    let actual_q = session
        .readback_f32(q_output_buffer)
        .expect("read Q output");
    let actual_k = session
        .readback_f32(k_output_buffer)
        .expect("read K output");
    let actual_v = session
        .readback_f32(v_output_buffer)
        .expect("read V output");
    assert_close("QkvProjection Q", &actual_q, &expected_q);
    assert_close("QkvProjection K", &actual_k, &expected_k);
    assert_close("QkvProjection V", &actual_v, &expected_v);

    let residual: Vec<f32> = activation.iter().map(|value| value + 0.5).collect();
    let skip: Vec<f32> = activation.iter().map(|value| value * 0.25).collect();
    let gamma: Vec<f32> = (0..facts.hidden as usize)
        .map(|index| 0.75 + index as f32 * 0.01)
        .collect();
    let rms_bind =
        BindDescriptor::row_major(vec![facts.rows, facts.hidden], [facts.hidden as u32, 1, 1]);
    let mut expected_rms = vec![f32::NAN; residual.len()];
    dispatch_metal_library(MetalLibraryDispatch::ResidualRmsNorm {
        library_entry: Some("ResidualRmsNorm"),
        layout: BindLayout::RowMajor,
        bind: &rms_bind,
        residual: &residual,
        skip: &skip,
        gamma: &gamma,
        output: &mut expected_rms,
        epsilon: facts.epsilon,
    })
    .expect("CPU runtime bridge ResidualRmsNorm parity");
    let residual_buffer = session
        .alloc_bytes(residual.len() * std::mem::size_of::<f32>())
        .expect("allocate residual");
    let skip_buffer = session
        .alloc_bytes(skip.len() * std::mem::size_of::<f32>())
        .expect("allocate skip");
    let gamma_buffer = session
        .alloc_bytes(gamma.len() * std::mem::size_of::<f32>())
        .expect("allocate gamma");
    let rms_output_buffer = session
        .alloc_bytes(expected_rms.len() * std::mem::size_of::<f32>())
        .expect("allocate RMS output");
    session
        .copy_in_f32(residual_buffer, &residual)
        .expect("upload residual");
    session
        .copy_in_f32(skip_buffer, &skip)
        .expect("upload skip");
    session
        .copy_in_f32(gamma_buffer, &gamma)
        .expect("upload gamma");
    session
        .launch_kernel_3d(
            module,
            "ResidualRmsNorm",
            &[
                residual_buffer,
                skip_buffer,
                gamma_buffer,
                rms_output_buffer,
            ],
            facts.rows as u32 * facts.hidden as u32,
            1,
            1,
            1,
            1,
            1,
        )
        .expect("encode ResidualRmsNorm runtime entry");
    session
        .sync()
        .expect("submit ResidualRmsNorm runtime entry");
    let actual_rms = session
        .readback_f32(rms_output_buffer)
        .expect("read RMS output");
    assert_close("ResidualRmsNorm", &actual_rms, &expected_rms);
    eprintln!(
        "PB-4a runtime parity receipt: entries=[QkvProjection,ResidualRmsNorm] rows={} hidden={} q_width={} kv_width={} real_metal=true",
        facts.rows, facts.hidden, q_width, kv_width
    );

    for handle in [
        activation_buffer,
        q_weight_buffer,
        k_weight_buffer,
        v_weight_buffer,
        q_output_buffer,
        k_output_buffer,
        v_output_buffer,
        residual_buffer,
        skip_buffer,
        gamma_buffer,
        rms_output_buffer,
        module,
    ] {
        session
            .release(handle)
            .expect("release fused runtime handle");
    }
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    let max_abs = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs <= 2.0e-5,
        "{label} runtime mismatch: max_abs={max_abs} actual={actual:?} expected={expected:?}"
    );
}
