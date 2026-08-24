# Agent Instructions

Work in this crate's checkout on `main` unless the operator assigned a
packet. Container law: [`../../../AGENTS.md`](../../../AGENTS.md).

GPU architecture follows the canonical public
[GPU Execution Architecture](https://github.com/faberlang/faber/blob/main/docs/gpu-execution-architecture.md).
Despite this crate's name, ML kernel source belongs in Gradus as Faber. This
crate may provide transport-neutral host routing and contracts, but it must not
become an alternate ML kernel library or a silent CPU fallback.

- Keep this crate transport-neutral: no worker threads, filesystem/process
  effects, or concrete provider dependencies.
- Registration must fail closed on invalid prefixes, duplicate prefixes/routes,
  and manifest/provider mismatches.
- Run `cargo fmt --check`, `cargo test`, and `cargo clippy --all-targets -- -D warnings`
  before intentional commits.
- Do not use destructive Git cleanup commands; preserve foreign work.
