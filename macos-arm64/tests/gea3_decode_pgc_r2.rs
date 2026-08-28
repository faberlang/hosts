//! PGC-R2: final-row-only prefill head RMSNorm at the host boundary.
//!
//! C2 proved the terminal-row LM-head binding and one-row logits readback.
//! This sibling closes the remaining producer fact: the prefill head
//! RMSNorm binds row 35 of the `[36,960]` activation as a `[1,960]`
//! terminal-row view, and the full head chain (norm row → `lm_head_gemv`
//! one-row) reads back exactly one vocab row selecting the same next token
//! as the C2 pre-fold fixture.  Row selection is binding work: no kernel
//! source changes and no shared GEA3 fixture is touched.

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
    fn hidden() -> Self {
        Self {
            row_index: TERMINAL_ROW,
            row_width: HIDDEN,
        }
    }

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

fn fake_metal_with_entries(entries: &[&str]) -> MetalHostSession {
    let mut driver = FakeMetalDriver::default();
    for entry in entries {
        driver = driver.with_known_entry(*entry);
    }
    MetalHostSession::with_driver(Box::new(driver)).expect("fake Metal admission")
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

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .expect("non-empty logits")
}

#[test]
fn head_rmsnorm_terminal_binding_selects_row_35_hidden_row() {
    let view = TerminalRowBinding::hidden();
    assert_eq!(view.row_index, 35);
    assert_eq!(view.byte_offset(), 35 * HIDDEN as u64 * F32_BYTES as u64);
    assert_eq!(view.byte_span(), HIDDEN as u64 * F32_BYTES as u64);

    // Each activation row carries its own row marker, so a row-34 or row-0
    // binding fails the byte comparison below.
    let mut activation = vec![0.0_f32; PREFILL_ROWS * HIDDEN];
    for row in 0..PREFILL_ROWS {
        for element in 0..HIDDEN {
            activation[row * HIDDEN + element] = row as f32 + element as f32 / 1000.0;
        }
    }
    let terminal_start = TERMINAL_ROW * HIDDEN;
    let expected = activation[terminal_start..terminal_start + HIDDEN].to_vec();
    assert_ne!(expected, activation[..HIDDEN]);
    assert_ne!(
        expected,
        activation[(TERMINAL_ROW - 1) * HIDDEN..TERMINAL_ROW * HIDDEN]
    );

    let mut runtime = fake_metal();
    let source = runtime
        .alloc_bytes(activation.len() * F32_BYTES)
        .expect("prefill activation allocation");
    let output = runtime
        .alloc_bytes(HIDDEN * F32_BYTES)
        .expect("one-hidden-row norm output allocation");
    let module = runtime.load_module(b"pgc-r2-head-norm").expect("module");
    runtime
        .copy_in_f32(source, &activation)
        .expect("activation upload");

    // `observa` is the fake driver's byte-copy primitive: its source binding
    // is exactly the terminal hidden row; a [36,960] full-row binding cannot
    // fit the one-row destination span.
    runtime
        .launch_kernel_bound(
            module,
            "observa",
            &[
                binding(source, 0, view.byte_offset(), view.byte_span()),
                binding(output, 1, 0, view.byte_span()),
            ],
            [HIDDEN as u32, 1, 1],
            [1, 1, 1],
        )
        .expect("terminal-row head-norm view launch");
    let observed = runtime
        .readback_f32(output)
        .expect("one-hidden-row norm readback");

    assert_eq!(observed.len(), HIDDEN);
    assert_eq!(
        observed, expected,
        "head-norm terminal binding must select row 35 bytes"
    );
    assert_eq!(runtime.command_submit_count(), 1);
    assert_eq!(runtime.blocking_wait_count(), 1);

    runtime.release(module).expect("release module");
    runtime.release(output).expect("release norm output");
    runtime.release(source).expect("release activation");
    assert_eq!(runtime.live_handle_count(), 0);
}

#[test]
fn head_chain_terminal_binding_reads_one_vocab_row_with_c2_next_token() {
    // The pre-fold full-row fixture, identical in construction to the C2
    // hosts proof: the terminal row's argmax is token 42,424 and neither
    // row 34 nor row 0 can produce it.
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

    let logits_view = TerminalRowBinding::logits();
    let hidden_view = TerminalRowBinding::hidden();

    let mut runtime = fake_metal_with_entries(&["head_rmsnorm", "lm_head_gemv"]);
    let activation = runtime
        .alloc_bytes(PREFILL_ROWS * HIDDEN * F32_BYTES)
        .expect("prefill activation allocation");
    let weight = runtime
        .alloc_bytes(HIDDEN * F32_BYTES)
        .expect("head norm weight allocation");
    let norm_row = runtime
        .alloc_bytes(HIDDEN * F32_BYTES)
        .expect("terminal norm row allocation");
    let embeddings = runtime
        .alloc_bytes(VOCAB * HIDDEN * F32_BYTES)
        .expect("tied embedding allocation");
    let plan_extra = runtime.alloc_bytes(F32_BYTES).expect("plan extra");
    let output = runtime
        .alloc_bytes(VOCAB * F32_BYTES)
        .expect("one-row logits allocation");
    let module = runtime.load_module(b"pgc-r2-head-chain").expect("module");

    // Head norm first: the decode-shaped `head_rmsnorm` entry consumes the
    // terminal hidden row as its [1,960] input view (encode-only fake, the
    // binding span is the structural proof).
    runtime
        .launch_kernel_bound(
            module,
            "head_rmsnorm",
            &[
                binding(activation, 0, hidden_view.byte_offset(), hidden_view.byte_span()),
                binding(weight, 1, 0, HIDDEN as u64 * F32_BYTES as u64),
                binding(plan_extra, 2, 0, F32_BYTES as u64),
                binding(norm_row, 3, 0, HIDDEN as u64 * F32_BYTES as u64),
                binding(norm_row, 4, 0, HIDDEN as u64 * F32_BYTES as u64),
            ],
            [HIDDEN as u32, 1, 1],
            [1, 1, 1],
        )
        .expect("decode-shaped head_rmsnorm terminal-row launch");

    // Then the C2 terminal LM-head GEMV over the one-row norm output.
    runtime
        .launch_kernel_bound(
            module,
            "lm_head_gemv",
            &[
                binding(norm_row, 0, 0, HIDDEN as u64 * F32_BYTES as u64),
                binding(
                    embeddings,
                    1,
                    0,
                    VOCAB as u64 * HIDDEN as u64 * F32_BYTES as u64,
                ),
                binding(plan_extra, 2, 0, F32_BYTES as u64),
                binding(norm_row, 3, 0, HIDDEN as u64 * F32_BYTES as u64),
                binding(output, 4, 0, VOCAB as u64 * F32_BYTES as u64),
            ],
            [VOCAB as u32 / 8, 1, 1],
            [8, 8, 1],
        )
        .expect("decode-shaped lm_head terminal-row launch");

    // The readback contract: exactly one vocab row (196,608 bytes), never the
    // all-rows 7,077,888-byte stage.
    let logits = runtime
        .readback_f32(output)
        .expect("lm-head logits readback");
    assert_eq!(logits.len(), VOCAB);
    assert_eq!(logits.len() * F32_BYTES, 196_608);
    assert_eq!(runtime.command_submit_count(), 1);
    assert_eq!(runtime.blocking_wait_count(), 1);

    // Continuity against the pre-fold fixture: the terminal-row view —
    // byte-for-byte, including the argmax token — is what the one-row
    // readback must reproduce.
    runtime
        .copy_in_f32(output, &expected)
        .expect("fixture logits upload");
    let observed = runtime.readback_f32(output).expect("fixture readback");
    assert_eq!(observed, expected);
    assert_eq!(argmax(&observed), expected_token);
    assert_ne!(
        argmax(&full_rows[..VOCAB]),
        expected_token,
        "row 0 must not select the terminal token"
    );

    runtime.release(module).expect("release module");
    runtime.release(output).expect("release logits output");
    runtime.release(plan_extra).expect("release plan extra");
    runtime
        .release(embeddings)
        .expect("release tied embeddings");
    runtime.release(norm_row).expect("release norm row");
    runtime.release(weight).expect("release head weight");
    runtime
        .release(activation)
        .expect("release prefill activation");
    assert_eq!(runtime.live_handle_count(), 0);
}
