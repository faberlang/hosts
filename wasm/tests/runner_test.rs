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
            stdout: "9\n".to_owned(),
            stderr: String::new(),
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
            stdout: "0\n2\n4\n6\n".to_owned(),
            stderr: String::new(),
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

/// A known v1 symbol outside the admitted Stage 2/5 registry rejects during
/// preflight with a typed import outcome naming the field.
#[test]
fn unknown_v1_field_rejects_with_typed_outcome() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_tensor_shape" (func $len (param i32) (result i32)))
  (func (export "incipit") (drop (call $len (i32.const 0))))
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert!(
        matches!(
            &outcome,
            RunOutcome::ImportRejected { module, field, .. }
                if module == WASM_IMPORT_MODULE_V1 && field == "__faber_rt_v1_tensor_shape"
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
            message.contains("W11/W12"),
            "message must name the W11/W12 literal-initialization boundary: {message}"
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
        RunOutcome::Success {
            stdout: String::new(),
            stderr: String::new(),
        },
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
        RunOutcome::Success {
            stdout: String::new(),
            stderr: String::new(),
        },
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
            stdout: "Salve, Munde!\n".to_owned(),
            stderr: String::new(),
        },
        "salve-munde must render its literal through the host arena, got: {outcome:?}"
    );
    assert_eq!(outcome.category(), OutcomeCategory::Success);
}

/// A synthetic module declaring the W12 literal table exactly as the emitter
/// generates it (payload data + `(kind, offset, length)` rows + exported
/// globals): the host interns the literal at init and the nota_text call
/// renders it. This is the emitter's contract shape, not a host-side
/// reconstruction of an interner table from WAT.
#[test]
fn declared_literal_table_interning_renders_text() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_text" (func $nota_text (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "Salve, Munde!")
  (data (i32.const 13) "\00\00\00\00\00\00\00\00\0D\00\00\00")
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
            stdout: "Salve, Munde!\n".to_owned(),
            stderr: String::new(),
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
  (data (i32.const 13) "\00\00\00\00\00\00\00\00\0D\00\00\00")
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

// ---------------------------------------------------------------------------
// W12 kind-tagged literal table: octeti + regex rows
// ---------------------------------------------------------------------------

/// A synthetic module declaring octeti rows (kind 1) exactly as the emitter
/// generates them: the host interns the byte payloads at init, and the
/// `nota_ptr` pointer diagnostic renders an octeti handle in the byte-list
/// Debug shape (`[104, 105]`), mirroring the LLVM host's opaque display.
#[test]
fn octeti_rows_intern_and_render_byte_list_through_nota() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_ptr" (func $nota_ptr (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "hi")
  (data (i32.const 2) "\DE\AD\BE\EF")
  (data (i32.const 6) "\01\00\00\00\00\00\00\00\02\00\00\00")
  (data (i32.const 18) "\01\00\00\00\02\00\00\00\04\00\00\00")
  (global (export "__faber_rt_v1_literal_table_ptr") i32 (i32.const 6))
  (global (export "__faber_rt_v1_literal_table_count") i32 (i32.const 2))
  (func (export "incipit")
    (call $nota_ptr (i32.const 0))
    (call $nota_ptr (i32.const 1))
  )
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome,
        RunOutcome::Success {
            stdout: "[104, 105]\n[222, 173, 190, 239]\n".to_owned(),
            stderr: String::new(),
        },
        "interned octeti rows must render through the pointer diagnostic, got: {outcome:?}"
    );
}

/// A synthetic module declaring a regex pattern row (kind 2) plus its flags
/// row (kind 3): the host pairs them into one regex value at init, and
/// `nota_ptr` renders the pattern text — the shared oracle's regex display.
#[test]
fn regex_rows_pair_flags_and_render_pattern_through_nota() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_ptr" (func $nota_ptr (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "\5Cd+")
  (data (i32.const 3) "i")
  (data (i32.const 4) "\02\00\00\00\00\00\00\00\03\00\00\00")
  (data (i32.const 16) "\03\00\00\00\03\00\00\00\01\00\00\00")
  (global (export "__faber_rt_v1_literal_table_ptr") i32 (i32.const 4))
  (global (export "__faber_rt_v1_literal_table_count") i32 (i32.const 2))
  (func (export "incipit")
    (call $nota_ptr (i32.const 0))
  )
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome,
        RunOutcome::Success {
            stdout: "\\d+\n".to_owned(),
            stderr: String::new(),
        },
        "the interned regex pattern must render through the pointer diagnostic, got: {outcome:?}"
    );
}

/// A regex literal without a flags row (pattern row only) also resolves.
#[test]
fn regex_row_without_flags_render_pattern_through_nota() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_ptr" (func $nota_ptr (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "(?i)\5Cw+")
  (data (i32.const 7) "\02\00\00\00\00\00\00\00\07\00\00\00")
  (global (export "__faber_rt_v1_literal_table_ptr") i32 (i32.const 7))
  (global (export "__faber_rt_v1_literal_table_count") i32 (i32.const 1))
  (func (export "incipit")
    (call $nota_ptr (i32.const 0))
  )
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome,
        RunOutcome::Success {
            stdout: "(?i)\\w+\n".to_owned(),
            stderr: String::new(),
        },
        "a flagless regex row must render its pattern, got: {outcome:?}"
    );
}

/// A literal table declaring an unknown row kind fails generated
/// initialization with a typed outcome — entry never runs.
#[test]
fn unknown_literal_row_kind_fails_initialization() {
    let bytes = wat_bytes(r#"
(module
  (memory (export "memory") 1)
  (data (i32.const 0) "x")
  (data (i32.const 1) "\63\00\00\00\00\00\00\00\01\00\00\00")
  (global (export "__faber_rt_v1_literal_table_ptr") i32 (i32.const 1))
  (global (export "__faber_rt_v1_literal_table_count") i32 (i32.const 1))
  (func (export "incipit") (return))
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome.category(),
        OutcomeCategory::InitializationFailed,
        "an unknown literal row kind must be InitializationFailed, got: {outcome:?}"
    );
}

/// A flags-carrying regex followed by another literal keeps later rows at
/// their raw indices (the flags row is a continuation row, not a handle).
#[test]
fn regex_flags_row_keeps_later_rows_at_raw_indices() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_ptr" (func $nota_ptr (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "\5Cd+")
  (data (i32.const 3) "i")
  (data (i32.const 4) "post")
  (data (i32.const 8) "\02\00\00\00\00\00\00\00\03\00\00\00")
  (data (i32.const 20) "\03\00\00\00\03\00\00\00\01\00\00\00")
  (data (i32.const 32) "\00\00\00\00\04\00\00\00\04\00\00\00")
  (global (export "__faber_rt_v1_literal_table_ptr") i32 (i32.const 8))
  (global (export "__faber_rt_v1_literal_table_count") i32 (i32.const 3))
  (func (export "incipit")
    (call $nota_ptr (i32.const 0))
    (call $nota_ptr (i32.const 2))
  )
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome,
        RunOutcome::Success {
            stdout: "\\d+\npost\n".to_owned(),
            stderr: String::new(),
        },
        "the flags continuation row must keep the following text row at raw index 2, got: {outcome:?}"
    );
}

// ---------------------------------------------------------------------------
// W12 text-format surface: `§`-template application
// ---------------------------------------------------------------------------

/// A synthetic `format_text` call: the template and its text arg are table
/// rows; the returned dynamic text handle feeds a later `nota_ptr`.
#[test]
fn format_text_renders_template_into_a_new_text_handle() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_format_text" (func $format_text (param i32 i32) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_ptr" (func $nota_ptr (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "Mundus")
  (data (i32.const 6) "Salve, \C2\A7!")
  (data (i32.const 16) "\00\00\00\00\00\00\00\00\06\00\00\00")
  (data (i32.const 28) "\00\00\00\00\06\00\00\00\0A\00\00\00")
  (global (export "__faber_rt_v1_literal_table_ptr") i32 (i32.const 16))
  (global (export "__faber_rt_v1_literal_table_count") i32 (i32.const 2))
  (func (export "incipit")
    (drop (call $format_text (i32.const 1) (i32.const 0)))
    (call $nota_ptr (i32.const 2))
  )
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome,
        RunOutcome::Success {
            stdout: "Salve, Mundus!\n".to_owned(),
            stderr: String::new(),
        },
        "format_text must substitute the § template arg into a new text handle, got: {outcome:?}"
    );
}

/// `format_i64` substitutes the decimal rendering of a scalar arg, and a
/// numbered `§1` placeholder selects the second arg (mirrors the shared
/// oracle template policy).
#[test]
fn format_i64_renders_scalar_and_numbered_placeholders() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_format_i64_i64" (func $format_i64_i64 (param i32 i64 i64) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_format_i64" (func $format_i64 (param i32 i64) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_ptr" (func $nota_ptr (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "aetas: \C2\A7")
  (data (i32.const 9) "coordinata: \C2\A7 \C2\A7")
  (data (i32.const 26) "\00\00\00\00\00\00\00\00\09\00\00\00")
  (data (i32.const 38) "\00\00\00\00\09\00\00\00\11\00\00\00")
  (global (export "__faber_rt_v1_literal_table_ptr") i32 (i32.const 26))
  (global (export "__faber_rt_v1_literal_table_count") i32 (i32.const 2))
  (func (export "incipit")
    (drop (call $format_i64 (i32.const 0) (i64.const 30)))
    (call $nota_ptr (i32.const 2))
    (drop (call $format_i64_i64 (i32.const 1) (i64.const 10) (i64.const 20)))
    (call $nota_ptr (i32.const 3))
  )
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome,
        RunOutcome::Success {
            stdout: "aetas: 30\ncoordinata: 10 20\n".to_owned(),
            stderr: String::new(),
        },
        "scalar template args must render as the shared oracle formats them, got: {outcome:?}"
    );
}

// ---------------------------------------------------------------------------
// W12 text arena surface: concat, equality, queries, transforms
// ---------------------------------------------------------------------------

/// `text_concat` produces a new text handle and `text_eq`/`text_ne` compare
/// interned literals directly on the arena.
#[test]
fn text_concat_and_eq_operate_on_the_host_text_arena() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_text_concat" (func $concat (param i32 i32) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_text_eq" (func $eq (param i32 i32) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_text_ne" (func $ne (param i32 i32) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_ptr" (func $nota_ptr (param i32)))
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_i32" (func $nota_i32 (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "Sa")
  (data (i32.const 2) "lve")
  (data (i32.const 5) "\00\00\00\00\00\00\00\00\02\00\00\00\00\00\00\00\02\00\00\00\03\00\00\00")
  (global (export "__faber_rt_v1_literal_table_ptr") i32 (i32.const 5))
  (global (export "__faber_rt_v1_literal_table_count") i32 (i32.const 2))
  (func (export "incipit")
    (drop (call $concat (i32.const 0) (i32.const 1)))
    (call $nota_ptr (i32.const 2))
    (call $nota_i32 (call $eq (i32.const 2) (i32.const 2)))
    (call $nota_i32 (call $ne (i32.const 0) (i32.const 1)))
  )
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome,
        RunOutcome::Success {
            stdout: "Salve\n1\n1\n".to_owned(),
            stderr: String::new(),
        },
        "concat and text equality must resolve through the host text arena, got: {outcome:?}"
    );
}

/// The first-order text arena ops (`length`, `contains`, `uppercase`,
/// `slice`) mirror the LLVM host semantics against the interned literals.
#[test]
fn text_query_and_transform_rows_operate_on_the_host_text_arena() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_text_length" (func $len (param i32) (result i64)))
  (import "faber_rt_v1" "__faber_rt_v1_text_contains" (func $contains (param i32 i32) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_text_uppercase" (func $up (param i32) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_text_slice" (func $slice (param i32 i64 i64) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_i64" (func $nota_i64 (param i64)))
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_i32" (func $nota_i32 (param i32)))
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_ptr" (func $nota_ptr (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) " Ave Roma ")
  (data (i32.const 10) "Roma")
  (data (i32.const 14) "\00\00\00\00\00\00\00\00\0A\00\00\00\00\00\00\00\0A\00\00\00\04\00\00\00")
  (global (export "__faber_rt_v1_literal_table_ptr") i32 (i32.const 14))
  (global (export "__faber_rt_v1_literal_table_count") i32 (i32.const 2))
  (func (export "incipit")
    (call $nota_i64 (call $len (i32.const 0)))
    (call $nota_ptr (call $up (i32.const 0)))
    (call $nota_i32 (call $contains (i32.const 0) (i32.const 1)))
    (call $nota_ptr (call $slice (i32.const 0) (i64.const 1) (i64.const 4)))
  )
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome,
        RunOutcome::Success {
            stdout: "10\n AVE ROMA \n1\nAve\n".to_owned(),
            stderr: String::new(),
        },
        "text query/transform rows must mirror the LLVM host semantics, got: {outcome:?}"
    );
}

// ---------------------------------------------------------------------------
// W12 stderr capture + regex conversion rows
// ---------------------------------------------------------------------------

/// `mone` (mone_ptr) streams to stderr: the W12 product host captures it into
/// `Success::stderr` — never a silent redirect to stdout.
#[test]
fn mone_streams_to_captured_stderr() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_mone_ptr" (func $mone_ptr (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "cave")
  (data (i32.const 4) "\00\00\00\00\00\00\00\00\04\00\00\00")
  (global (export "__faber_rt_v1_literal_table_ptr") i32 (i32.const 4))
  (global (export "__faber_rt_v1_literal_table_count") i32 (i32.const 1))
  (func (export "incipit")
    (call $mone_ptr (i32.const 0))
  )
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome,
        RunOutcome::Success {
            stdout: String::new(),
            stderr: "cave\n".to_owned(),
        },
        "mone must stream to the captured stderr, got: {outcome:?}"
    );
    assert_eq!(outcome.category(), OutcomeCategory::Success);
}

/// `regex_from_text` (the `textus ↦ regex` conversion the emitter does not
/// constant-fold) returns a regex handle the pointer diagnostic renders as
/// the pattern text.
#[test]
fn regex_from_text_converts_a_text_handle_to_a_renderable_regex() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_regex_from_text" (func $from_text (param i32) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_ptr" (func $nota_ptr (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "/home/acme/.*")
  (data (i32.const 13) "\00\00\00\00\00\00\00\00\0D\00\00\00")
  (global (export "__faber_rt_v1_literal_table_ptr") i32 (i32.const 13))
  (global (export "__faber_rt_v1_literal_table_count") i32 (i32.const 1))
  (func (export "incipit")
    (drop (call $from_text (i32.const 0)))
    (call $nota_ptr (i32.const 1))
  )
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome,
        RunOutcome::Success {
            stdout: "/home/acme/.*\n".to_owned(),
            stderr: String::new(),
        },
        "regex_from_text must produce a regex the pointer diagnostic renders, got: {outcome:?}"
    );
}

// ---------------------------------------------------------------------------
// W13 collection/scalar display rows
// ---------------------------------------------------------------------------

/// An array literal constructed through `array_new(kind)` + `array_push`
/// renders in the Rust-oracle `[1, 2, 3]` Debug shape when a pointer
/// diagnostic resolves the handle.
#[test]
fn array_literal_renders_bracket_debug_shape() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_array_new" (func $array_new (param i32) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_array_push" (func $array_push (param i32 i64) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_ptr" (func $nota_ptr (param i32)))
  (func (export "incipit")
    (local $t i32)
    (local.set $t (call $array_new (i32.const 4)))
    (local.set $t (call $array_push (local.get $t) (i64.const 1)))
    (local.set $t (call $array_push (local.get $t) (i64.const 2)))
    (local.set $t (call $array_push (local.get $t) (i64.const 3)))
    (call $nota_ptr (local.get $t))
  )
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome,
        RunOutcome::Success {
            stdout: "[1, 2, 3]\n".to_owned(),
            stderr: String::new(),
        },
        "array handles must render in the lista Debug shape, got: {outcome:?}"
    );
}

/// A text-element array renders the quoted `["prima", "secunda"]` shape and
/// `array_get` reads elements back as i64 carriers.
#[test]
fn text_array_renders_quoted_elements_and_array_get_reads_elements() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_array_new" (func $array_new (param i32) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_array_push" (func $array_push (param i32 i64) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_array_get" (func $array_get (param i32 i64) (result i64)))
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_ptr" (func $nota_ptr (param i32)))
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_i64" (func $nota_i64 (param i64)))
  (memory (export "memory") 1)
  (data (i32.const 0) "prima")
  (data (i32.const 5) "secunda")
  (data (i32.const 12) "\00\00\00\00\00\00\00\00\05\00\00\00\00\00\00\00\05\00\00\00\07\00\00\00")
  (global (export "__faber_rt_v1_literal_table_ptr") i32 (i32.const 12))
  (global (export "__faber_rt_v1_literal_table_count") i32 (i32.const 2))
  (func (export "incipit")
    (local $t i32)
    (local.set $t (call $array_new (i32.const 14)))
    (local.set $t (call $array_push (local.get $t) (i64.const 0)))
    (local.set $t (call $array_push (local.get $t) (i64.const 1)))
    (call $nota_ptr (local.get $t))
    (call $nota_i64 (call $array_get (local.get $t) (i64.const 1)))
  )
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome,
        RunOutcome::Success {
            stdout: "[\"prima\", \"secunda\"]\n1\n".to_owned(),
            stderr: String::new(),
        },
        "text arrays must quote elements and array_get must read them back, got: {outcome:?}"
    );
}

/// A map literal constructed through `map_new(kinds)` + `map_put` renders in
/// the Rust-oracle derived `Json(Tabula({...}))` Debug shape.
#[test]
fn map_literal_renders_json_tabula_debug_shape() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_map_new" (func $map_new (param i32 i32) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_map_put" (func $map_put (param i32 i32 i64)))
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_ptr" (func $nota_ptr (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "alpha")
  (data (i32.const 5) "beta")
  (data (i32.const 9) "\00\00\00\00\00\00\00\00\05\00\00\00\00\00\00\00\05\00\00\00\04\00\00\00")
  (global (export "__faber_rt_v1_literal_table_ptr") i32 (i32.const 9))
  (global (export "__faber_rt_v1_literal_table_count") i32 (i32.const 2))
  (func (export "incipit")
    (local $t i32)
    (local.set $t (call $map_new (i32.const 14) (i32.const 4)))
    (call $map_put (local.get $t) (i32.const 1) (i64.const 20))
    (call $map_put (local.get $t) (i32.const 0) (i64.const 10))
    (call $nota_ptr (local.get $t))
  )
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome,
        RunOutcome::Success {
            stdout: "Json(Tabula({\"alpha\": Numerus(10), \"beta\": Numerus(20)}))\n".to_owned(),
            stderr: String::new(),
        },
        "map handles must render in the Json(Tabula(...)) Debug shape, got: {outcome:?}"
    );
}

/// A `copia` (`set_new` + `array_push`) renders the `{1, 2, 3}` shape and
/// `array_contains`/`array_length` read back set facts.
#[test]
fn set_renders_brace_shape_and_reads_back_facts() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_set_new" (func $set_new (param i32) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_array_push" (func $array_push (param i32 i64) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_array_contains" (func $contains (param i32 i64) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_array_length" (func $length (param i32) (result i64)))
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_ptr" (func $nota_ptr (param i32)))
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_i32" (func $nota_i32 (param i32)))
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_i64" (func $nota_i64 (param i64)))
  (func (export "incipit")
    (local $t i32)
    (local.set $t (call $set_new (i32.const 4)))
    (local.set $t (call $array_push (local.get $t) (i64.const 1)))
    (local.set $t (call $array_push (local.get $t) (i64.const 2)))
    (local.set $t (call $array_push (local.get $t) (i64.const 3)))
    (call $nota_ptr (local.get $t))
    (call $nota_i32 (call $contains (local.get $t) (i64.const 2)))
    (call $nota_i64 (call $length (local.get $t)))
  )
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome,
        RunOutcome::Success {
            stdout: "{1, 2, 3}\n1\n3\n".to_owned(),
            stderr: String::new(),
        },
        "set handles must render braces and read back contains/length, got: {outcome:?}"
    );
}

/// A map index lookup (`array_option`) returns an option handle that `nota`
/// renders as the payload (present) or `nihil` (absent); `option_get_or`
/// unwraps with a fallback.
#[test]
fn map_index_returns_option_rendered_as_payload_or_nihil() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_map_new" (func $map_new (param i32 i32) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_map_put" (func $map_put (param i32 i32 i64)))
  (import "faber_rt_v1" "__faber_rt_v1_array_option" (func $array_option (param i32 i64) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_option_get_or" (func $get_or (param i32 i64) (result i64)))
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_ptr" (func $nota_ptr (param i32)))
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_i64" (func $nota_i64 (param i64)))
  (memory (export "memory") 1)
  (data (i32.const 0) "aelia")
  (data (i32.const 5) "balbus")
  (data (i32.const 11) "\00\00\00\00\00\00\00\00\05\00\00\00\00\00\00\00\05\00\00\00\06\00\00\00")
  (global (export "__faber_rt_v1_literal_table_ptr") i32 (i32.const 11))
  (global (export "__faber_rt_v1_literal_table_count") i32 (i32.const 2))
  (func (export "incipit")
    (local $map i32)
    (local $opt i32)
    (local.set $map (call $map_new (i32.const 14) (i32.const 4)))
    (call $map_put (local.get $map) (i32.const 0) (i64.const 95))
    (local.set $opt (call $array_option (local.get $map) (i64.const 0)))
    (call $nota_ptr (local.get $opt))
    (local.set $opt (call $array_option (local.get $map) (i64.const 1)))
    (call $nota_ptr (local.get $opt))
    (local.set $opt (call $array_option (local.get $map) (i64.const 1)))
    (call $nota_i64 (call $get_or (local.get $opt) (i64.const 0)))
  )
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome,
        RunOutcome::Success {
            stdout: "95\nnihil\n0\n".to_owned(),
            stderr: String::new(),
        },
        "map index options must render payload or nihil and get_or must fall back, got: {outcome:?}"
    );
}

/// `nota_i1` renders bivalens diagnostics as `verum`/`falsum` (the scalar
/// display half of the cluster).
#[test]
fn bivalens_diagnostics_render_verum_falsum() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_i1" (func $nota_i1 (param i32)))
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_i32" (func $nota_i32 (param i32)))
  (func (export "incipit")
    (call $nota_i1 (i32.const 1))
    (call $nota_i1 (i32.const 0))
    (call $nota_i32 (i32.const 1))
  )
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome,
        RunOutcome::Success {
            stdout: "verum\nfalsum\n1\n".to_owned(),
            stderr: String::new(),
        },
        "bivalens diagnostics must render verum/falsum while i32 stays integer, got: {outcome:?}"
    );
}

/// `map_keys`/`map_values` project a map into `lista` handles that render in
/// the `["aelia", "balbus"]` / `[95, 87]` shapes.
#[test]
fn map_keys_and_values_project_lista_handles() {
    let bytes = wat_bytes(r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_map_new" (func $map_new (param i32 i32) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_map_put" (func $map_put (param i32 i32 i64)))
  (import "faber_rt_v1" "__faber_rt_v1_map_keys" (func $map_keys (param i32) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_map_values" (func $map_values (param i32) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_ptr" (func $nota_ptr (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "aelia")
  (data (i32.const 5) "balbus")
  (data (i32.const 11) "\00\00\00\00\00\00\00\00\05\00\00\00\00\00\00\00\05\00\00\00\06\00\00\00")
  (global (export "__faber_rt_v1_literal_table_ptr") i32 (i32.const 11))
  (global (export "__faber_rt_v1_literal_table_count") i32 (i32.const 2))
  (func (export "incipit")
    (local $map i32)
    (local.set $map (call $map_new (i32.const 14) (i32.const 4)))
    (call $map_put (local.get $map) (i32.const 0) (i64.const 95))
    (call $map_put (local.get $map) (i32.const 1) (i64.const 87))
    (call $nota_ptr (call $map_keys (local.get $map)))
    (call $nota_ptr (call $map_values (local.get $map)))
  )
)
"#);
    let outcome = host().run(&bytes, &RunConfig::default());
    assert_eq!(
        outcome,
        RunOutcome::Success {
            stdout: "[\"aelia\", \"balbus\"]\n[95, 87]\n".to_owned(),
            stderr: String::new(),
        },
        "map_keys/map_values must project renderable lista handles, got: {outcome:?}"
    );
}
