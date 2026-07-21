//! Product proof: plain Wasm + faber_rt_v1 imports vs Rust-observable output.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use faber_host_macos_arm64::{WasmRtV1Host, WASM_IMPORT_MODULE_V1};

fn radix_root() -> PathBuf {
    // hosts/macos-arm64 → sibling radix/
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../radix")
}

fn faberlang_root() -> PathBuf {
    // hosts/macos-arm64 → faberlang/
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn emit_wasm_text(source_path: &std::path::Path) -> String {
    let radix_manifest = radix_root().join("crates/radix/Cargo.toml");
    let output = Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "--manifest-path",
            radix_manifest.to_str().expect("utf8 path"),
            "--bin",
            "radix",
            "--",
            "emit",
            "-t",
            "wasm-text",
            source_path.to_str().expect("utf8 path"),
        ])
        .current_dir(radix_root())
        .output()
        .expect("spawn radix emit");
    assert!(
        output.status.success(),
        "radix emit failed: status={:?} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("wat utf8")
}

fn extract_nota_ptr_handle(wat: &str) -> i32 {
    // First product fixture emits a single nota_ptr call with one i32.const handle.
    let handles = extract_i32_consts(wat);
    assert_eq!(
        handles.len(),
        1,
        "expected exactly one i32.const text handle in salve-munde wat:\n{wat}"
    );
    handles[0]
}

/// Collect every `i32.const N` in emission order (text handles for the current
/// emit profile; numeric work uses `i64.const`).
fn extract_i32_consts(wat: &str) -> Vec<i32> {
    let marker = "i32.const ";
    let mut out = Vec::new();
    let mut rest = wat;
    while let Some(idx) = rest.find(marker) {
        rest = &rest[idx + marker.len()..];
        let digits: String = rest
            .chars()
            .take_while(|ch| ch.is_ascii_digit() || *ch == '-')
            .collect();
        if digits.is_empty() {
            continue;
        }
        let value: i32 = digits
            .parse()
            .unwrap_or_else(|_| panic!("parse i32.const from `{digits}` in wat"));
        out.push(value);
        rest = &rest[digits.len()..];
    }
    out
}

/// First `i32.const N` appearing after `anchor` in the WAT.
fn extract_i32_const_after(wat: &str, anchor: &str) -> Option<i32> {
    let idx = wat.find(anchor)?;
    let window = &wat[idx..];
    let marker = "i32.const ";
    let cidx = window.find(marker)?;
    let rest = &window[cidx + marker.len()..];
    let digits: String = rest
        .chars()
        .take_while(|ch| ch.is_ascii_digit() || *ch == '-')
        .collect();
    digits.parse().ok()
}

fn assert_closed_set_v1_only(wat: &str) {
    assert!(
        !wat.contains("faber_diag"),
        "closed-set must not emit legacy faber_diag dialect:\n{wat}"
    );
    assert!(
        !wat.contains("__faber_diag_"),
        "closed-set must not emit legacy __faber_diag_ symbols:\n{wat}"
    );
    assert!(
        !wat.contains(r#"(import "faber_aggregate""#),
        "second product fixture must stay on faber_rt_v1 only (no aggregate):\n{wat}"
    );
    assert!(
        !wat.contains(r#"(import "faber_runtime""#),
        "second product fixture must stay on faber_rt_v1 only (no faber_runtime):\n{wat}"
    );
    assert!(
        !wat.contains(r#"(import "faber_text""#),
        "second product fixture must stay on faber_rt_v1 only (no faber_text):\n{wat}"
    );
}

#[test]
fn salve_munde_wasm_rt_v1_matches_expected_stdout() {
    let source = faberlang_root().join("examples/corpus/incipit/salve-munde.fab");
    let expected = std::fs::read_to_string(
        faberlang_root().join("examples/corpus/incipit/salve-munde.expected"),
    )
    .expect("expected file");
    let wat = emit_wasm_text(&source);

    assert!(
        wat.contains(&format!(
            r#"(import "{WASM_IMPORT_MODULE_V1}" "__faber_rt_v1_diagnostic_nota_ptr""#
        )),
        "fixture must import closed-set v1 nota_ptr:\n{wat}"
    );
    assert_closed_set_v1_only(&wat);

    let handle = extract_nota_ptr_handle(&wat);
    let mut texts = BTreeMap::new();
    // Module-scope `nota "Salve, Munde!"` lowers to a text handle + nota_ptr.
    // The host product boundary receives the handle table for string materialization
    // until linear-memory string data is part of the emit profile.
    texts.insert(handle, "Salve, Munde!".to_owned());

    let host = WasmRtV1Host::new().expect("host init");
    let result = host
        .run_module(wat.as_bytes(), "incipit", texts)
        .expect("wasm rt v1 run");
    assert!(result.success);
    assert_eq!(result.stdout, expected);
}

/// Second B2 fixture: multi-function corpus program with text handles **and**
/// scalar `nota_i64` (second closed-set import family beyond salve-munde's ptr).
#[test]
fn functio_wasm_rt_v1_matches_expected_stdout() {
    let source = faberlang_root().join("examples/corpus/functio/functio.fab");
    let expected =
        std::fs::read_to_string(faberlang_root().join("examples/corpus/functio/functio.expected"))
            .expect("expected file");
    let wat = emit_wasm_text(&source);

    assert!(
        wat.contains(&format!(
            r#"(import "{WASM_IMPORT_MODULE_V1}" "__faber_rt_v1_diagnostic_nota_ptr""#
        )),
        "fixture must import closed-set v1 nota_ptr:\n{wat}"
    );
    assert!(
        wat.contains(&format!(
            r#"(import "{WASM_IMPORT_MODULE_V1}" "__faber_rt_v1_diagnostic_nota_i64""#
        )),
        "second fixture must import closed-set v1 nota_i64 (scalar family):\n{wat}"
    );
    assert_closed_set_v1_only(&wat);

    // Map handles from call-site / defining-body patterns (module declaration
    // order of i32.const is not the same as runtime argument order).
    let saluta_handle = extract_i32_const_after(&wat, "func $saluta").expect("saluta text handle");
    let dic_arg_handle =
        extract_i32_const_after(&wat, "call $dic").expect("dic argument text handle");
    let nomen_handle =
        extract_i32_const_after(&wat, "func $nomen").expect("nomen return text handle");
    let mut texts = BTreeMap::new();
    texts.insert(saluta_handle, "Salve, Mundus!".to_owned());
    texts.insert(dic_arg_handle, "Bonum diem!".to_owned());
    texts.insert(nomen_handle, "Marcus Aurelius".to_owned());

    let host = WasmRtV1Host::new().expect("host init");
    let result = host
        .run_module(wat.as_bytes(), "incipit", texts)
        .expect("wasm rt v1 functio run");
    assert!(result.success);
    // Corpus .expected has no trailing newline; host diagnostics always end lines.
    assert_eq!(result.stdout, format!("{expected}\n"));
}

#[test]
fn wasm_rt_v1_rejects_legacy_import_module() {
    let wat = r#"
(module
  (import "faber_diag" "nota_i64" (func $legacy (param i64)))
  (func (export "incipit")
    (call $legacy (i64.const 1))
  )
)
"#;
    let host = WasmRtV1Host::new().expect("host init");
    let error = host
        .run_module(wat.as_bytes(), "incipit", BTreeMap::new())
        .expect_err("legacy module must reject");
    let message = error.to_string();
    assert!(
        message.contains("unsupported import module") || message.contains("faber_diag"),
        "unexpected error: {message}"
    );
}

#[test]
fn wasm_rt_v1_rejects_unbound_v1_import() {
    let wat = r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_array_length" (func $len (param i32) (result i64)))
  (func (export "incipit")
    (drop (call $len (i32.const 0)))
  )
)
"#;
    let host = WasmRtV1Host::new().expect("host init");
    let error = host
        .run_module(wat.as_bytes(), "incipit", BTreeMap::new())
        .expect_err("unbound v1 import must reject");
    let message = error.to_string();
    assert!(
        message.contains("unsupported v1 host import") || message.contains("array_length"),
        "unexpected error: {message}"
    );
}
