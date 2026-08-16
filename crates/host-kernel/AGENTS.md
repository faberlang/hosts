# Agent Instructions

Work in this crate's checkout on `main` unless the operator assigned a
packet. Container law: [`../../../AGENTS.md`](../../../AGENTS.md).

- Keep this crate transport-neutral: no worker threads, filesystem/process
  effects, or concrete provider dependencies.
- Registration must fail closed on invalid prefixes, duplicate prefixes/routes,
  and manifest/provider mismatches.
- Run `cargo fmt --check`, `cargo test`, and `cargo clippy --all-targets -- -D warnings`
  before intentional commits.
- Do not use destructive Git cleanup commands; preserve foreign work.
