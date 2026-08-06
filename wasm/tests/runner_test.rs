//! Portable product runner proofs and typed reject cases.
//!
//! The success proofs use real compiler artifacts emitted by the radix Wasm
//! target from Stage 1 ledger fixtures (`sic/sic.fab`, `per/per.fab`,
//! `salve-munde/salve-munde.fab`) and checked into `fixtures/`. The scalar
//! fixtures (`sic`/`per`) need no opaque-handle table; `salve-munde` proves
//! the W11 literal-table contract (a text literal lives in module linear
//! memory, the host interns it at generated initialization, and the program's
//! literal reference is the arena handle). Reject cases assert the typed
//! validation/import/link/initialization/entry/trap/runtime distinctions.

use faber_host_wasm::{OutcomeCategory, RunConfig, RunOutcome, WasmRtV1Host, WASM_IMPORT_MODULE_V1};

const SIC_WASM: &[u8] = include_bytes!("fixtures/sic.wasm");
const PER_WASM: &[u8] = include_bytes!("fixtures/per.wasm");
const SALVE_MUNDE_WASM: &[u8] = include_bytes!("fixtures/salve-munde.wasm");

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

// ---------------------------------------------------------------------------
// W4B provider text surface: solum + consolum rows the emitter now emits
// ---------------------------------------------------------------------------

/// All six W4B provider symbols are admitted by preflight and bound at link
/// with the exact signatures the radix Wasm emitter emits. A module that
/// declares them but never invokes them runs to success — proof that the
/// closed-set surface is accepted, not rejected as unknown (W13).
#[test]
fn w4b_provider_surface_accepted_by_preflight_and_link() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_solum_read_text" (func $read_text (param i32) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_solum_read_lines" (func $read_lines (param i32) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_solum_read_bytes" (func $read_bytes (param i32) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_solum_write_text" (func $write_text (param i32 i32)))
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_text" (func $nota_text (param i32)))
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_mone_text" (func $mone_text (param i32)))
  (func (export "incipit") (nop))
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome,
        RunOutcome::Success { stdout: String::new() },
        "declared-only W4B provider surface must pass preflight and link, got: {outcome:?}"
    );
}

/// Solum fixture: `lege<textus>` (read_text) is admitted but has no host
/// implementation in this stage (no filesystem capability; W15
/// deny-by-default), so invoking it is a typed unsupported outcome naming the
/// symbol — never a silent no-op or a synthesized result handle.
#[test]
fn w4b_solum_fixture_read_produces_typed_unsupported() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_solum_read_text" (func $read_text (param i32) (result i32)))
  (func (export "incipit") (drop (call $read_text (i32.const 3))))
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome.category(),
        OutcomeCategory::RuntimeFailure,
        "solum read_text without a filesystem capability must be RuntimeFailure, got: {outcome:?}"
    );
    if let RunOutcome::RuntimeFailure { message } = &outcome {
        assert!(
            message.contains("__faber_rt_v1_solum_read_text"),
            "message must name the solum symbol: {message}"
        );
        assert!(
            message.contains("typed unsupported"),
            "message must declare the typed-unsupported outcome: {message}"
        );
    }
}

/// The full solum fixture surface (read_text/read_lines/read_bytes/write_text,
/// the exact emitter signatures) links and then produces one typed unsupported
/// outcome on first invocation.
#[test]
fn w4b_solum_fixture_full_surface_links_then_typed_unsupported() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_solum_read_text" (func $read_text (param i32) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_solum_read_lines" (func $read_lines (param i32) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_solum_read_bytes" (func $read_bytes (param i32) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_solum_write_text" (func $write_text (param i32 i32)))
  (func (export "incipit")
    (call $write_text (i32.const 3) (i32.const 4))
    (drop (call $read_text (i32.const 3)))
  )
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome.category(),
        OutcomeCategory::RuntimeFailure,
        "solum invoke without a filesystem capability must be RuntimeFailure, got: {outcome:?}"
    );
}

/// Consolum fixture: `scribe` (nota_text) and `mone` (mone_text) close-overlap
/// the v1 diagnostic text rows. The operand is an opaque text handle the
/// runner cannot materialize, so invoking either is a typed unsupported
/// outcome naming the symbol and its oracle stream.
#[test]
fn w4b_consolum_fixture_produces_typed_unsupported() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_text" (func $nota_text (param i32)))
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_mone_text" (func $mone_text (param i32)))
  (func (export "incipit")
    (call $nota_text (i32.const 5))
    (call $mone_text (i32.const 6))
  )
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome.category(),
        OutcomeCategory::RuntimeFailure,
        "consolum diagnostic text without literal materialization must be RuntimeFailure, got: {outcome:?}"
    );
    if let RunOutcome::RuntimeFailure { message } = &outcome {
        assert!(
            message.contains("__faber_rt_v1_diagnostic_nota_text"),
            "message must name the consolum symbol: {message}"
        );
        assert!(
            message.contains("nota/stdout"),
            "message must record the oracle stream semantics: {message}"
        );
    }
}

/// A declared signature that conflicts with an admitted W4B binding fails at
/// link time with a typed link outcome (the binding is signature-checked, not
/// permissive).
#[test]
fn w4b_provider_signature_mismatch_fails_link() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_solum_read_text" (func $read_text (param i32)))
  (func (export "incipit") (call $read_text (i32.const 3)))
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome.category(),
        OutcomeCategory::LinkFailed,
        "declared signature conflicting with the admitted solum binding must be LinkFailed, got: {outcome:?}"
    );
}

// ---------------------------------------------------------------------------
// WE6 json surface: the three emitter-route json v1 symbols
// ---------------------------------------------------------------------------

/// All three WE6 json symbols are admitted by preflight and bound at link
/// with the exact signatures the radix Wasm emitter emits (`(param i32)
/// (result i32)` handle carriers). A module that declares them but never
/// invokes them runs to success — proof that the closed-set surface is
/// accepted, not rejected as unknown (W13).
#[test]
fn we6_json_provider_surface_accepted_by_preflight_and_link() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_json_pange" (func $pange (param i32) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_json_solve" (func $solve (param i32) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_json_tempta" (func $tempta (param i32) (result i32)))
  (func (export "incipit") (nop))
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome,
        RunOutcome::Success { stdout: String::new() },
        "declared-only WE6 json surface must pass preflight and link, got: {outcome:?}"
    );
}

/// Json fixture: `solve` (parse wire text -> json value) is admitted but has
/// no host json implementation in this stage (W13 typed-unsupported until the
/// json host impl lands), so invoking it is a typed unsupported outcome
/// naming the symbol — never a silent no-op or a synthesized result handle.
#[test]
fn we6_json_fixture_produces_typed_unsupported() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_json_solve" (func $solve (param i32) (result i32)))
  (func (export "incipit") (drop (call $solve (i32.const 3))))
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome.category(),
        OutcomeCategory::RuntimeFailure,
        "json solve without a host json implementation must be RuntimeFailure, got: {outcome:?}"
    );
    if let RunOutcome::RuntimeFailure { message } = &outcome {
        assert!(
            message.contains("__faber_rt_v1_json_solve"),
            "message must name the json symbol: {message}"
        );
        assert!(
            message.contains("typed unsupported"),
            "message must declare the typed-unsupported outcome: {message}"
        );
    }
}

/// A declared signature that conflicts with an admitted WE6 json binding
/// fails at link time with a typed link outcome (the binding is
/// signature-checked, not permissive).
#[test]
fn we6_json_signature_mismatch_fails_link() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_json_pange" (func $pange (param i64) (result i32)))
  (func (export "incipit") (drop (call $pange (i64.const 3))))
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome.category(),
        OutcomeCategory::LinkFailed,
        "declared signature conflicting with the admitted json binding must be LinkFailed, got: {outcome:?}"
    );
}

// ---------------------------------------------------------------------------
// W11 literal table: text literals in linear memory, interned at init
// ---------------------------------------------------------------------------

/// `salve-munde/salve-munde.fab` (canonical hello-world): module-scope
/// `nota "Salve, Munde!"` — a text literal the radix Wasm emitter puts in
/// module linear memory under the W11 literal-table contract. The product
/// host reads the declared table at generated initialization, interns the
/// literal into its text arena, and renders the nota line from the interned
/// text. The Rust oracle outcome is `Salve, Munde!\n`.
#[test]
fn salve_munde_renders_literal_through_the_host_text_arena() {
    let outcome = host().run(SALVE_MUNDE_WASM, &RunConfig::default());
    assert_eq!(
        outcome,
        RunOutcome::Success {
            stdout: "Salve, Munde!\n".to_owned()
        },
        "salve-munde must render its literal through the host arena, got: {outcome:?}"
    );
    assert_eq!(outcome.category(), OutcomeCategory::Success);
}

/// A synthetic module declaring the W11 literal table exactly as the emitter
/// generates it (payload data + table rows + exported globals): the host
/// interns the literal at init and the nota_text call renders it. This is the
/// emitter's contract shape, not a host-side reconstruction of an interner
/// table from WAT.
#[test]
fn declared_literal_table_interning_renders_text() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_text" (func $nota_text (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "Salve, Munde!")
  (data (i32.const 13) "\00\00\00\00\0D\00\00\00")
  (global (export "__faber_rt_v1_literal_table_ptr") i32 (i32.const 13))
  (global (export "__faber_rt_v1_literal_table_count") i32 (i32.const 1))
  (func (export "incipit")
    (call $nota_text (i32.const 0))
  )
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome,
        RunOutcome::Success {
            stdout: "Salve, Munde!\n".to_owned()
        },
        "the interned literal must render through the nota text row, got: {outcome:?}"
    );
}

/// A module declaring only one of the two literal-table globals fails
/// generated initialization with a typed outcome — entry never runs.
#[test]
fn partial_literal_table_declaration_fails_initialization() {
    let bytes = wat_bytes(r#"
(module
  (global (export "__faber_rt_v1_literal_table_ptr") i32 (i32.const 0))
  (func (export "incipit") (return))
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome.category(),
        OutcomeCategory::InitializationFailed,
        "a half-declared literal table must be InitializationFailed, got: {outcome:?}"
    );
    if let RunOutcome::InitializationFailed { message } = &outcome {
        assert!(
            message.contains("__faber_rt_v1_literal_table_count"),
            "message must name the missing table global: {message}"
        );
    }
}

/// A literal table whose rows extend past linear memory fails generated
/// initialization with a typed outcome — the host never reads past the
/// module's memory and never synthesizes a handle table.
#[test]
fn literal_table_out_of_bounds_fails_initialization() {
    let bytes = wat_bytes(r#"
(module
  (memory (export "memory") 1)
  (global (export "__faber_rt_v1_literal_table_ptr") i32 (i32.const 65532))
  (global (export "__faber_rt_v1_literal_table_count") i32 (i32.const 1))
  (func (export "incipit") (return))
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome.category(),
        OutcomeCategory::InitializationFailed,
        "an out-of-bounds literal table must be InitializationFailed, got: {outcome:?}"
    );
}

/// A literal-table handle with no interned row still produces a typed runtime
/// failure (never a plausible default), even when the module does declare a
/// table.
#[test]
fn uninterned_handle_produces_typed_runtime_failure() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_text" (func $nota_text (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "Salve, Munde!")
  (data (i32.const 13) "\00\00\00\00\0D\00\00\00")
  (global (export "__faber_rt_v1_literal_table_ptr") i32 (i32.const 13))
  (global (export "__faber_rt_v1_literal_table_count") i32 (i32.const 1))
  (func (export "incipit")
    (call $nota_text (i32.const 1))
  )
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome.category(),
        OutcomeCategory::RuntimeFailure,
        "a handle outside the interned table must be RuntimeFailure, got: {outcome:?}"
    );
}
