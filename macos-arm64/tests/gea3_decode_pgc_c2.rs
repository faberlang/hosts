//! PGC-C2: terminal-row-only prefill logits at the host boundary.
//!
//! The fake Metal driver is intentionally structural for GEA3 entries.  The
//! first proof therefore uses its `observa` copy primitive to exercise the
//! exact non-zero row view and readback bytes, then the second proof admits the
//! same view in the five-slot `lm_head_gemv` ABI.  No CPU LM-head replacement
//! is used and no shared GEA3 fixture is modified.

use faber_host_macos_arm64::metal_host::MetalLaunchBinding;
use faber_host_macos_arm64::{FakeMetalDriver, MetalHostSession};

const PREFILL_ROWS: usize = 36;
const TERMINAL_ROW: usize = PREFILL_ROWS - 1;
const HIDDEN: usize = 960;
const VOCAB: usize = 49_152;
const F32_BYTES: usize = std::mem::size_of::<f32>();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TerminalRowBinding {
    row_index: usize,
    row_width: usize,
}

impl TerminalRowBinding {
    fn logits() -> Self {
        Self {
            row_index: TERMINAL_ROW,
            row_width: VOCAB,
        }
    }

    fn byte_offset(self) -> u64 {
        (self.row_index * self.row_width * F32_BYTES) as u64
    }

    fn byte_span(self) -> u64 {
        (self.row_width * F32_BYTES) as u64
    }
}

fn fake_metal() -> MetalHostSession {
    MetalHostSession::with_driver(Box::new(FakeMetalDriver::default()))
        .expect("fake Metal admission")
}

fn fake_metal_with_lm_head() -> MetalHostSession {
    let driver = FakeMetalDriver::default().with_known_entry("lm_head_gemv");
    MetalHostSession::with_driver(Box::new(driver)).expect("fake Metal admission")
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .expect("non-empty logits")
}

fn binding(
    handle: faber_host_macos_arm64::MetalHandleId,
    index: u32,
    byte_offset: u64,
    view_span: u64,
) -> MetalLaunchBinding {
    MetalLaunchBinding {
        handle,
        binding_index: index,
        byte_offset,
        view_span,
    }
}

#[test]
fn terminal_row_view_reads_back_one_logits_row_and_same_next_token() {
    let view = TerminalRowBinding::logits();
    assert_eq!(view.row_index, 35);
    assert_eq!(view.byte_offset(), 35 * VOCAB as u64 * F32_BYTES as u64);
    assert_eq!(view.byte_span(), VOCAB as u64 * F32_BYTES as u64);

    // This is the old full-row observation fixture: each prompt row has its
    // own candidate so a row-34 or row-0 binding cannot accidentally pass.
    let mut full_rows = vec![0.0_f32; PREFILL_ROWS * VOCAB];
    for row in 0..PREFILL_ROWS {
        full_rows[row * VOCAB + row] = 10_000.0 + row as f32;
    }
    full_rows[(PREFILL_ROWS - 2) * VOCAB + 17] = 50_000.0;
    let terminal_start = TERMINAL_ROW * VOCAB;
    full_rows[terminal_start + 42_424] = 100_000.0;
    let expected = full_rows[terminal_start..terminal_start + VOCAB].to_vec();
    let expected_token = argmax(&expected);
    assert_eq!(expected_token, 42_424);

    let mut runtime = fake_metal();
    let source = runtime
        .alloc_bytes(full_rows.len() * F32_BYTES)
        .expect("full prefill logits allocation");
    let output = runtime
        .alloc_bytes(view.row_width * F32_BYTES)
        .expect("one-row logits output allocation");
    let module = runtime.load_module(b"pgc-c2-observation").expect("module");
    runtime
        .copy_in_f32(source, &full_rows)
        .expect("fixture upload");

    // `observa` is the fake driver's byte-copy observation primitive.  Its
    // source binding is the exact terminal row; the destination is one vocab
    // row, so a full [36,V] readback cannot fit this launch.
    runtime
        .launch_kernel_bound(
            module,
            "observa",
            &[
                binding(source, 0, view.byte_offset(), view.byte_span()),
                binding(output, 1, 0, view.byte_span()),
            ],
            [VOCAB as u32, 1, 1],
            [1, 1, 1],
        )
        .expect("terminal-row observation launch");
    let observed = runtime
        .readback_f32(output)
        .expect("one-row logits readback");

    assert_eq!(observed.len(), VOCAB);
    assert_eq!(
        observed, expected,
        "new one-row observation must match row 35"
    );
    assert_eq!(argmax(&observed), expected_token);
    assert_ne!(observed, full_rows[..VOCAB], "row 0 must not be observed");
    assert_eq!(runtime.command_submit_count(), 1);
    assert_eq!(runtime.blocking_wait_count(), 1);

    runtime.release(module).expect("release observation module");
    runtime.release(output).expect("release logits output");
    runtime.release(source).expect("release full-row fixture");
    assert_eq!(runtime.live_handle_count(), 0);
}

#[test]
fn lm_head_gemv_terminal_binding_is_one_row_and_readback_is_one_vocab_row() {
    let view = TerminalRowBinding {
        row_index: TERMINAL_ROW,
        row_width: HIDDEN,
    };
    assert_eq!(view.row_index, 35);
    assert_eq!(view.byte_offset(), 35 * HIDDEN as u64 * F32_BYTES as u64);
    assert_eq!(view.byte_span(), HIDDEN as u64 * F32_BYTES as u64);

    let mut runtime = fake_metal_with_lm_head();
    let activation = runtime
        .alloc_bytes(PREFILL_ROWS * HIDDEN * F32_BYTES)
        .expect("prefill activation allocation");
    let embeddings = runtime
        .alloc_bytes(VOCAB * HIDDEN * F32_BYTES)
        .expect("tied embedding allocation");
    let plan_extra = runtime.alloc_bytes(F32_BYTES).expect("plan extra");
    let head_norm = runtime
        .alloc_bytes(HIDDEN * F32_BYTES)
        .expect("head norm allocation");
    let output = runtime
        .alloc_bytes(VOCAB * F32_BYTES)
        .expect("one-row logits allocation");
    let module = runtime.load_module(b"pgc-c2-lm-head").expect("module");

    // The five slots mirror the existing expanded GEA3 ABI:
    // activation, tied weight, plan extra, prior head output, output.
    runtime
        .launch_kernel_bound(
            module,
            "lm_head_gemv",
            &[
                binding(activation, 0, view.byte_offset(), view.byte_span()),
                binding(
                    embeddings,
                    1,
                    0,
                    VOCAB as u64 * HIDDEN as u64 * F32_BYTES as u64,
                ),
                binding(plan_extra, 2, 0, F32_BYTES as u64),
                binding(head_norm, 3, 0, HIDDEN as u64 * F32_BYTES as u64),
                binding(output, 4, 0, VOCAB as u64 * F32_BYTES as u64),
            ],
            [VOCAB as u32 / 8, 1, 1],
            [8, 8, 1],
        )
        .expect("decode-shaped lm_head terminal-row launch");

    // The fake GEA3 entry is encode-only, so its physical assertion is the
    // admitted output allocation and readback shape, not fabricated logits.
    let logits = runtime
        .readback_f32(output)
        .expect("lm-head logits readback");
    assert_eq!(logits.len(), VOCAB);
    assert!(logits.iter().all(|value| *value == 0.0));
    assert_eq!(runtime.command_submit_count(), 1);
    assert_eq!(runtime.blocking_wait_count(), 1);

    runtime.release(module).expect("release lm-head module");
    runtime.release(output).expect("release logits output");
    runtime
        .release(head_norm)
        .expect("release prior head output");
    runtime.release(plan_extra).expect("release plan extra");
    runtime
        .release(embeddings)
        .expect("release tied embeddings");
    runtime
        .release(activation)
        .expect("release prefill activation");
    assert_eq!(runtime.live_handle_count(), 0);
}
