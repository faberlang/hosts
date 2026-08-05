//! Portable product runner proofs and typed reject cases.
//!
//! The success proofs use real compiler artifacts emitted by the radix Wasm
//! target from Stage 1 ledger fixtures (`sic/sic.fab`, `per/per.fab`) and
//! checked into `fixtures/`. Both modules import only scalar
//! `__faber_rt_v1_diagnostic_*` functions, so their execution needs no
//! externally reconstructed opaque-handle table. Reject cases assert the
//! typed validation/import/link/entry/trap/runtime distinctions.

use faber_host_wasm::{OutcomeCategory, RunConfig, RunOutcome, WasmRtV1Host, WASM_IMPORT_MODULE_V1};

const SIC_WASM: &[u8] = include_bytes!("fixtures/sic.wasm");
const PER_WASM: &[u8] = include_bytes!("fixtures/per.wasm");

fn host() -> WasmRtV1Host {
    WasmRtV1Host::new().expect("host init")
}

/// Parse synthetic reject-module WAT to Wasm bytes so the runner receives
/// only Wasm bytes (never WAT) as its input contract requires.
fn wat_bytes(wat: &str) -> Vec<u8> {
    wat::parse_str(wat).expect("synthetic module must parse")
}

// ---------------------------------------------------------------------------
// Proofs: real compiler artifacts through the portable runner
// ---------------------------------------------------------------------------

/// `sic/sic.fab` (Stage 1 ledger row): si/sic/secus max over 3 and 9, one
/// `nota_i64`. Rust oracle outcome is `9`.
#[test]
fn sic_compiler_artifact_matches_rust_outcome() {
    let outcome = host().run(SIC_WASM, &RunConfig::default());
    assert_eq!(
        outcome,
        RunOutcome::Success {
            stdout: "9\n".to_owned()
        },
        "sic must run without any opaque-handle table"
    );
    assert_eq!(outcome.category(), OutcomeCategory::Success);
}

/// `per/per.fab` (Stage 1 ledger row): itera over 0..8 step 2, four
/// `nota_i64` events. Rust oracle outcome is `0\n2\n4\n6`.
#[test]
fn per_compiler_artifact_matches_rust_outcome() {
    let outcome = host().run(PER_WASM, &RunConfig::default());
    assert_eq!(
        outcome,
        RunOutcome::Success {
            stdout: "0\n2\n4\n6\n".to_owned()
        },
        "per must run without any opaque-handle table"
    );
    assert_eq!(outcome.category(), OutcomeCategory::Success);
}

// ---------------------------------------------------------------------------
// Reject proof: typed outcome categories
// ---------------------------------------------------------------------------

/// A legacy import module (`faber_diag`) rejects during preflight with a
/// typed import outcome naming the offending module.
#[test]
fn legacy_import_module_rejects_with_typed_outcome() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_diag" "nota_i64" (func $legacy (param i64)))
  (func (export "incipit") (call $legacy (i64.const 1)))
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert!(
        matches!(
            &outcome,
            RunOutcome::ImportRejected { module, field, .. }
                if module == "faber_diag" && field == "nota_i64"
        ),
        "legacy module must reject with ImportRejected, got: {outcome:?}"
    );
    assert_eq!(outcome.category(), OutcomeCategory::ImportRejected);
}

/// A known v1 symbol outside the admitted Stage 2 registry rejects during
/// preflight with a typed import outcome naming the field.
#[test]
fn unknown_v1_field_rejects_with_typed_outcome() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_array_length" (func $len (param i32) (result i64)))
  (func (export "incipit") (drop (call $len (i32.const 0))))
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert!(
        matches!(
            &outcome,
            RunOutcome::ImportRejected { module, field, .. }
                if module == WASM_IMPORT_MODULE_V1 && field == "__faber_rt_v1_array_length"
        ),
        "unbound v1 field must reject with ImportRejected, got: {outcome:?}"
    );
    assert_eq!(outcome.category(), OutcomeCategory::ImportRejected);
}

/// An admitted field declared with a conflicting signature fails at link time
/// with a typed link outcome (distinct from the preflight import rejection).
#[test]
fn signature_mismatch_fails_link_with_typed_outcome() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_i64" (func $nota (param i32)))
  (func (export "incipit") (call $nota (i32.const 1)))
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome.category(),
        OutcomeCategory::LinkFailed,
        "declared signature conflicting with the admitted binding must be LinkFailed, got: {outcome:?}"
    );
}

/// A module without the configured entry export rejects with a typed
/// entry-missing outcome.
#[test]
fn missing_entry_rejects_with_typed_outcome() {
    let bytes = wat_bytes(r#"
(module
  (func (export "other") (return))
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert!(
        matches!(&outcome, RunOutcome::EntryMissing { entry } if entry == "incipit"),
        "missing entry must reject with EntryMissing, got: {outcome:?}"
    );
    assert_eq!(outcome.category(), OutcomeCategory::EntryMissing);
}

/// Invalid bytes fail Wasm validation with a typed validation outcome.
#[test]
fn invalid_bytes_fail_validation() {
    let outcome = host().run(b"not a wasm module", &RunConfig::default());
    assert_eq!(
        outcome.category(),
        OutcomeCategory::ValidationFailed,
        "invalid bytes must be a ValidationFailed outcome, got: {outcome:?}"
    );
}

/// A module whose entry traps produces a typed entry-trap outcome.
#[test]
fn entry_trap_is_typed() {
    let bytes = wat_bytes(r#"
(module
  (func (export "incipit") (unreachable))
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome.category(),
        OutcomeCategory::EntryTrapped,
        "trapped entry must be EntryTrapped, got: {outcome:?}"
    );
}

/// An admitted-but-unfinished text-handle symbol produces a typed runtime
/// failure when invoked (never an external-handle lookup or a plausible
/// default).
#[test]
fn text_handle_call_produces_typed_runtime_failure() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_ptr" (func $nota_ptr (param i32)))
  (func (export "incipit") (call $nota_ptr (i32.const 18)))
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome.category(),
        OutcomeCategory::RuntimeFailure,
        "admitted-but-unfinished text materialization must be RuntimeFailure, got: {outcome:?}"
    );
    if let RunOutcome::RuntimeFailure { message } = &outcome {
        assert!(
            message.contains("Stage 4"),
            "message must name the Stage 4 literal-initialization boundary: {message}"
        );
    }
}
