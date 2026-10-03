#![allow(clippy::absurd_extreme_comparisons)]

use hygiene_ratchet::{Budgets, ScanConfig, assert_budgets};
use std::path::Path;

fn config() -> ScanConfig {
    ScanConfig {
        source_roots: vec![Path::new(env!("CARGO_MANIFEST_DIR")).join("src")],
        exclude_path_suffixes: Vec::new(),
        subtract_self_expect: false,
        check_companion_convention: false,
    }
}

// Ratchet ruling (2026-10-03): 83 is the measured baseline when the workspace
// gate was introduced. This budget may only decrease; never raise it silently.
const BUDGETS: Budgets = Budgets {
    unwrap: 0,
    expect: 83,
    panic: 0,
    unreachable: 0,
    todo: 0,
    unimplemented: 0,
    let_underscore: 0,
    // Ratchet ruling (2026-10-03): 8 is the measured baseline of inline
    // #[cfg(test)] mod bodies under src/. This budget may only decrease;
    // never raise it silently.
    // Correction to 7fa245d's subject: eight module bodies moved out of six
    // production files. The scanner's observed inline-module count went 8 -> 0;
    // its observed count of production files containing #[test] went 6 -> 0.
    // Neither ceiling changed: inline modules stayed at 8, and #[test] stayed
    // at 0. The claimed "budget 6 -> 0" was a count/ceiling mix-up, not a
    // ratchet reduction. This correction leaves the ceilings unchanged rather
    // than changing an assertion to make the historical claim appear true.
    inline_test_modules: 8,
    test_attr_in_production: 0,
};

#[test]
fn production_hygiene_budgets() {
    let files = hygiene_ratchet::collect_production_files(&config());
    let counts = hygiene_ratchet::count_budgets(&files, config().subtract_self_expect);
    assert_budgets(counts, BUDGETS);
}
