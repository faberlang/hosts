# Agent Instructions

Work in this crate's checkout on `main` unless the operator assigned a
packet. Container law: [`../../../AGENTS.md`](../../../AGENTS.md).

All routing follows the canonical public
[Host Execution Architecture](https://github.com/faberlang/faber/blob/main/docs/host-execution-architecture.md).
This crate routes capabilities and artifacts; it does not implement portable
library behavior. Existing library-specific paths are migration debt, not
precedent.

GPU architecture additionally follows
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
