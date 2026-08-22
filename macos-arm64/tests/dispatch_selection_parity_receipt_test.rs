//! M4-U2b: dispatch-selection parity receipts for the decode projection seam.
//!
//! The batched body remains the numeric baseline.  The decode GEMV body is
//! allowed to differ only within the KV-A cross-path logits ruling, and the
//! receipt keeps the last-row comparison explicit for each accepted selector.
//! This is deliberately a focused host test: it does not add a kernel or
//! change dispatch policy.

use faber_host_macos_arm64::kernel::library::{
    dispatch_gemv, select_decode_gemv, GemvKernel, KernelBodyError, QuantizedFormat,
    QuantizedGemvBind,
};

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
