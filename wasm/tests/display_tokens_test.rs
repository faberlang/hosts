//! Module display tokens (need 694b33ce): a module that calls
//! `set_display_tokens` at the start of its entry makes the host render option
//! none, an option bool payload, collection bool elements and bool format
//! arguments through the module's words. A module that never calls it keeps
//! today's output exactly.

use faber_host_wasm::{RunConfig, RunOutcome, WasmRtV1Host};

fn run_wat(wat: &str) -> RunOutcome {
    let bytes = wat::parse_str(wat).expect("module must parse");
    WasmRtV1Host::new()
        .expect("host init")
        .run(&bytes, &RunConfig::default())
}

/// Four text rows (`true`, `false`, `none`, `tuple`) in the literal table.
const TABLE: &str = r#"
  (memory (export "memory") 1)
  (data (i32.const 0) "truefalsenonetuple")
  (data (i32.const 18) "\00\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\04\00\00\00\05\00\00\00\00\00\00\00\09\00\00\00\04\00\00\00\00\00\00\00\0d\00\00\00\05\00\00\00")
  (global (export "__faber_rt_v1_literal_table_ptr") i32 (i32.const 18))
  (global (export "__faber_rt_v1_literal_table_count") i32 (i32.const 4))
"#;

/// Print an option none, an option `true`, a `[true, false]` list and a
/// bivalens through the host; `set` is the optional start-up call.
fn module(set: &str) -> String {
    format!(
        r#"
(module
  (import "faber_rt_v1" "__faber_rt_v1_set_display_tokens" (func $set (param i32 i32 i32 i32)))
  (import "faber_rt_v1" "__faber_rt_v1_option_none" (func $none (param i32) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_option_some" (func $some (param i64 i32) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_array_new" (func $array_new (param i32) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_array_push" (func $array_push (param i32 i64) (result i32)))
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_ptr" (func $nota_ptr (param i32)))
  (import "faber_rt_v1" "__faber_rt_v1_diagnostic_nota_i1" (func $nota_i1 (param i32)))
  {TABLE}
  (func (export "incipit")
    (local $list i32)
    {set}
    (call $nota_ptr (call $none (i32.const 14)))
    (call $nota_ptr (call $some (i64.const 1) (i32.const 1)))
    (local.set $list (call $array_new (i32.const 1)))
    (local.set $list (call $array_push (local.get $list) (i64.const 1)))
    (local.set $list (call $array_push (local.get $list) (i64.const 0)))
    (call $nota_ptr (local.get $list))
    (call $nota_i1 (i32.const 1))
  )
)
"#
    )
}

#[test]
fn english_tokens_render_option_none_and_collection_bool() {
    let wat = module("(call $set (i32.const 0) (i32.const 1) (i32.const 2) (i32.const 3))");
    assert_eq!(
        run_wat(&wat),
        RunOutcome::Success {
            stdout: "none\ntrue\n[true, false]\ntrue\n".to_owned(),
            stderr: String::new(),
        }
    );
}

#[test]
fn default_tokens_stay_as_today() {
    assert_eq!(
        run_wat(&module("")),
        RunOutcome::Success {
            stdout: "nihil\nverum\n[true, false]\nverum\n".to_owned(),
            stderr: String::new(),
        }
    );
}
