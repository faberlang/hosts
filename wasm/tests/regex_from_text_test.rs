//! `regex_from_text` validates the pattern at the conversion (RD-9): a valid
//! pattern yields a renderable handle, a rejected one is the host's typed
//! failure carrying the construct id.

use faber_host_wasm::{OutcomeCategory, RunConfig, RunOutcome, WasmRtV1Host};

fn run_regex_from_text(pattern: &str) -> RunOutcome {
    let len = pattern.len();
    let table_at = len;
    let pattern: String = pattern
        .bytes()
        .map(|byte| format!("\\{byte:02X}"))
        .collect();
    let wat = format!(
        r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_regex_from_text" (func $from_text (param i32) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_ptr" (func $nota_ptr (param i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "{pattern}")
  (data (i32.const {table_at}) "\00\00\00\00\00\00\00\00\{len:02X}\00\00\00")
  (global (export "__faber_rt_v1_literal_table_ptr") i32 (i32.const {table_at}))
  (global (export "__faber_rt_v1_literal_table_count") i32 (i32.const 1))
  (func (export "incipit")
    (call $nota_ptr (call $from_text (i32.const 0)))
  )
)
"#
    );
    let bytes = wat::parse_str(wat).expect("synthetic module must parse");
    WasmRtV1Host::new()
        .expect("host init")
        .run(&bytes, &RunConfig::default())
}

#[test]
fn valid_pattern_converts_to_a_renderable_regex() {
    let outcome = run_regex_from_text("a+b");
    assert_eq!(
        outcome,
        RunOutcome::Success {
            stdout: "a+b\n".to_owned(),
            stderr: String::new(),
        }
    );
}

#[test]
fn rejected_pattern_is_a_typed_failure_with_the_construct_id() {
    for (pattern, id) in [
        ("a(?=b)", "lookahead"),
        ("(", "syntax"),
        ("(a)\\1", "backreference"),
    ] {
        let outcome = run_regex_from_text(pattern);
        assert_eq!(
            outcome.category(),
            OutcomeCategory::RuntimeFailure,
            "{pattern}"
        );
        let RunOutcome::RuntimeFailure { message } = outcome else {
            unreachable!("category checked above");
        };
        assert!(message.contains(&format!("{id}: ")), "{pattern}: {message}");
    }
}
