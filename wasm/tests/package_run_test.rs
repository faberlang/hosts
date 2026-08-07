//! U6-E — package run mode: cross-module resolution of the `faber_external`
//! surface.
//!
//! The radix package-aware emitter (`emit_wasm_text_probe_package_aware`,
//! U6-C) emits a same-package cross-module identity
//! (`importa:auxilium:saluta`) as an import on the `faber_external` module
//! under the canonical identity-based field name
//! (`external_product_importa_module_auxilium_func_saluta`); the sibling
//! unit's module defines the function under the canonical symbol
//! (`__faber_external_product_importa_module_auxilium_func_saluta`). These
//! proofs run synthetic WAT through [`WasmRtV1Host::run_package`] to verify
//! the product host instantiates the module set together, resolves those
//! imports against the sibling exports, and keeps every bucket typed:
//! `MissingImport` → `ImportRejected` (preflight), `NoEntryExport` →
//! `EntryMissing`, `EntryTrap` → `EntryTrapped`, `LinkFailed` → `LinkFailed`.
//! Single-module runs keep the closed `faber_rt_v1` preflight —
//! `faber_external` rejects there.

use faber_host_wasm::{OutcomeCategory, RunConfig, RunOutcome, WasmRtV1Host};

fn host() -> WasmRtV1Host {
    WasmRtV1Host::new().expect("host init")
}

/// Parse synthetic WAT to Wasm bytes so the runner receives only Wasm bytes
/// (never WAT) as its input contract requires.
fn wat_bytes(wat: &str) -> Vec<u8> {
    wat::parse_str(wat).expect("synthetic module must parse")
}

/// The import field the entry declares for `importa:auxilium:saluta` on the
/// `faber_external` module (the U6-C emitted field name).
const SALUTA_FIELD: &str = "external_product_importa_module_auxilium_func_saluta";
/// The canonical external symbol the sibling defines (`__faber_` + field).
const SALUTA_SYMBOL: &str = "__faber_external_product_importa_module_auxilium_func_saluta";

/// Helper module (library mode): defines `saluta` under the canonical
/// external symbol, `(param i64) (result i64)`, returning input + 1.
fn saluta_helper_wat() -> &'static str {
    r#"
(module
  (func (export "__faber_external_product_importa_module_auxilium_func_saluta")
        (param i64) (result i64)
    (i64.add (local.get 0) (i64.const 1)))
)
"#
}

/// Entry module: imports `saluta` on the `faber_external` module, calls it
/// with 41, and prints the result through the closed v1 `nota_i64` row.
fn saluta_entry_wat() -> &'static str {
    r#"
(module
  (import "faber_external" "external_product_importa_module_auxilium_func_saluta"
    (func $saluta (param i64) (result i64)))
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_i64" (func $nota (param i64)))
  (func (export "incipit")
    (local $t i64)
    (local.set $t (call $saluta (i64.const 41)))
    (call $nota (local.get $t)))
)
"#
}

// ---------------------------------------------------------------------------
// Package resolution proofs
// ---------------------------------------------------------------------------

/// U6-E done_when core: a two-module package (entry + helper, as U6-D emits
/// them) instantiates through the product host with the helper import
/// resolved to the sibling export. The helper BODY runs — 41 → 42 — so the
/// captured stdout proves the sibling function executed (a silent default or
/// a stubbed import would not print), and the entry's closed v1 surface
/// still binds as usual.
#[test]
fn two_module_package_resolves_helper_import_and_runs_to_success() {
    let entry = wat_bytes(saluta_entry_wat());
    let helper = wat_bytes(saluta_helper_wat());
    let outcome = host().run_package(&entry, &[&helper], &RunConfig::default());
    assert_eq!(
        outcome,
        RunOutcome::Success {
            stdout: "42\n".to_owned(),
            stderr: String::new(),
        },
        "two-module package must resolve the faber_external import to the sibling export, \
         got: {outcome:?}"
    );
    assert_eq!(outcome.category(), OutcomeCategory::Success);
}

/// U6-E done_when: a missing helper import yields the typed `MissingImport`
/// bucket — `RunOutcome::ImportRejected` naming `faber_external` and the
/// canonical field — never a silent default.
#[test]
fn missing_helper_import_is_typed_missing_import() {
    let entry = wat_bytes(saluta_entry_wat());
    // The sibling exports no canonical symbol, so `saluta` cannot resolve
    // anywhere in the package.
    let empty_helper = wat_bytes(r#"(module (func (export "helper_noop")))"#);
    let outcome = host().run_package(&entry, &[&empty_helper], &RunConfig::default());
    assert!(
        matches!(
            &outcome,
            RunOutcome::ImportRejected { module, field, .. }
                if module == "faber_external" && field == SALUTA_FIELD
        ),
        "unresolvable package external import must reject with ImportRejected naming \
         `faber_external::{SALUTA_FIELD}`, got: {outcome:?}"
    );
    if let RunOutcome::ImportRejected { message, .. } = &outcome {
        assert!(
            message.contains(SALUTA_SYMBOL),
            "message must name the canonical symbol: {message}"
        );
    }
    assert_eq!(outcome.category(), OutcomeCategory::ImportRejected);
}

/// U6-E non_goal: the closed `faber_rt_v1` preflight is unchanged for
/// single-module runs — a `faber_external` import still rejects through
/// [`WasmRtV1Host::run`] (a package import is never a host symbol).
#[test]
fn single_module_faber_external_import_still_rejects() {
    let entry = wat_bytes(saluta_entry_wat());
    let outcome = host().run(&entry, &RunConfig::default());
    assert!(
        matches!(
            &outcome,
            RunOutcome::ImportRejected { module, field, .. }
                if module == "faber_external" && field == SALUTA_FIELD
        ),
        "single-module run must keep rejecting `faber_external`, got: {outcome:?}"
    );
    assert_eq!(outcome.category(), OutcomeCategory::ImportRejected);
}

/// A three-module package: the outer helper itself imports a deeper helper's
/// canonical symbol, proving resolution is not entry-only — each module's
/// `faber_external` imports resolve against the siblings instantiated before
/// it (dependency-first order).
#[test]
fn sibling_chain_resolves_across_sibling_instances() {
    let inner = wat_bytes(
        r#"
(module
  (func (export "__faber_external_product_importa_module_b_func_plus_two")
        (param i64) (result i64)
    (i64.add (local.get 0) (i64.const 2)))
)
"#,
    );
    let outer = wat_bytes(
        r#"
(module
  (import "faber_external" "external_product_importa_module_b_func_plus_two"
    (func $plus_two (param i64) (result i64)))
  (func (export "__faber_external_product_importa_module_auxilium_func_saluta")
        (param i64) (result i64)
    (i64.add (call $plus_two (local.get 0)) (i64.const 1)))
)
"#,
    );
    let entry = wat_bytes(saluta_entry_wat());
    // Dependency-first order: the inner helper before the outer helper.
    let outcome = host().run_package(&entry, &[&inner, &outer], &RunConfig::default());
    assert_eq!(
        outcome,
        RunOutcome::Success {
            stdout: "44\n".to_owned(),
            stderr: String::new(),
        },
        "41 → +2 → +1 = 44 proves chained cross-module resolution, got: {outcome:?}"
    );
}

// ---------------------------------------------------------------------------
// Typed bucket preservation on the package path
// ---------------------------------------------------------------------------

/// A declared signature that conflicts with the sibling export passes the
/// symbol-presence preflight and fails at instantiate as `LinkFailed` —
/// distinct from the preflight `MissingImport` bucket.
#[test]
fn signature_mismatch_between_import_and_sibling_export_is_link_failed() {
    let entry = wat_bytes(saluta_entry_wat()); // imports (param i64) (result i64)
    let wrong_helper = wat_bytes(
        r#"
(module
  (func (export "__faber_external_product_importa_module_auxilium_func_saluta")
        (param i32) (result i32)
    (i32.const 7))
)
"#,
    );
    let outcome = host().run_package(&entry, &[&wrong_helper], &RunConfig::default());
    assert_eq!(
        outcome.category(),
        OutcomeCategory::LinkFailed,
        "signature conflict must be LinkFailed, got: {outcome:?}"
    );
}

/// `NoEntryExport` bucket: the package links (the helper import resolves) but
/// the entry module lacks the configured entry export → typed `EntryMissing`.
#[test]
fn package_missing_entry_is_entry_missing() {
    let helper = wat_bytes(saluta_helper_wat());
    let no_incipit_entry = wat_bytes(
        r#"
(module
  (import "faber_external" "external_product_importa_module_auxilium_func_saluta"
    (func $saluta (param i64) (result i64)))
  (func (export "other") (drop (call $saluta (i64.const 1))))
)
"#,
    );
    let outcome = host().run_package(&no_incipit_entry, &[&helper], &RunConfig::default());
    assert!(
        matches!(&outcome, RunOutcome::EntryMissing { entry } if entry == "incipit"),
        "package run with a missing entry export must be EntryMissing, got: {outcome:?}"
    );
    assert_eq!(outcome.category(), OutcomeCategory::EntryMissing);
}

/// `EntryTrap` bucket: the package instantiates (the import resolves) and the
/// entry traps → typed `EntryTrapped`.
#[test]
fn package_entry_trap_is_entry_trapped() {
    let helper = wat_bytes(saluta_helper_wat());
    let trapping_entry = wat_bytes(
        r#"
(module
  (import "faber_external" "external_product_importa_module_auxilium_func_saluta"
    (func $saluta (param i64) (result i64)))
  (func (export "incipit") (unreachable))
)
"#,
    );
    let outcome = host().run_package(&trapping_entry, &[&helper], &RunConfig::default());
    assert_eq!(
        outcome.category(),
        OutcomeCategory::EntryTrapped,
        "trapped package entry must be EntryTrapped, got: {outcome:?}"
    );
}

// ---------------------------------------------------------------------------
// Closed-surface invariants on the package path
// ---------------------------------------------------------------------------

/// A sibling module's closed import surface is preflighted too: a sibling
/// importing a legacy module rejects the whole package with the typed import
/// outcome (no special casing for helper modules).
#[test]
fn sibling_import_surface_is_preflighted() {
    let entry = wat_bytes(saluta_entry_wat());
    let bad_sibling = wat_bytes(
        r#"
(module
  (import "faber_diag" "nota_i64" (func $legacy (param i64)))
  (func (export "__faber_external_product_importa_module_auxilium_func_saluta")
        (param i64) (result i64)
    (i64.const 1))
)
"#,
    );
    let outcome = host().run_package(&entry, &[&bad_sibling], &RunConfig::default());
    assert!(
        matches!(
            &outcome,
            RunOutcome::ImportRejected { module, field, .. }
                if module == "faber_diag" && field == "nota_i64"
        ),
        "sibling legacy-module import must reject the package, got: {outcome:?}"
    );
    assert_eq!(outcome.category(), OutcomeCategory::ImportRejected);
}

/// A package run with zero siblings behaves like a single-module run: a plain
/// v1 module (closed surface only) instantiates and runs to success.
#[test]
fn lone_entry_package_run_matches_single_module_behavior() {
    let entry = wat_bytes(
        r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_i64" (func $nota (param i64)))
  (func (export "incipit") (call $nota (i64.const 9)))
)
"#,
    );
    let outcome = host().run_package(&entry, &[], &RunConfig::default());
    assert_eq!(
        outcome,
        RunOutcome::Success {
            stdout: "9\n".to_owned(),
            stderr: String::new(),
        },
        "lone-entry package run must match single-module behavior, got: {outcome:?}"
    );
}

/// Invalid bytes in either position fail validation with the typed outcome.
#[test]
fn invalid_package_bytes_fail_validation() {
    let helper = wat_bytes(saluta_helper_wat());
    let outcome = host().run_package(b"not a wasm module", &[&helper], &RunConfig::default());
    assert_eq!(
        outcome.category(),
        OutcomeCategory::ValidationFailed,
        "invalid entry bytes must be ValidationFailed, got: {outcome:?}"
    );
    let entry = wat_bytes(saluta_entry_wat());
    let outcome = host().run_package(&entry, &[b"not a wasm module"], &RunConfig::default());
    assert_eq!(
        outcome.category(),
        OutcomeCategory::ValidationFailed,
        "invalid sibling bytes must be ValidationFailed, got: {outcome:?}"
    );
}
