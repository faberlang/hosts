# Agent Instructions

- Keep this crate transport-neutral: no worker threads, filesystem/process
  effects, or concrete provider dependencies.
- Registration must fail closed on invalid prefixes, duplicate prefixes/routes,
  and manifest/provider mismatches.
- Run `cargo fmt --check`, `cargo test`, and `cargo clippy --all-targets -- -D warnings`
  before intentional commits.
- Do not use destructive Git cleanup commands; preserve foreign work.
