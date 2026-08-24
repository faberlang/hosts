# Agent Instructions

Work in this crate's checkout on `main` unless the operator assigned a
packet. Container law: [`../../../AGENTS.md`](../../../AGENTS.md).

GPU architecture follows the canonical public
[GPU Execution Architecture](https://github.com/faberlang/faber/blob/main/docs/gpu-execution-architecture.md).
Gradus owns ML kernel source, Radix owns compiled artifacts and execution
facts, and Hosts owns their physical execution. Scheduling code here must not
acquire ML semantics or introduce a CPU substitute for a declared GPU path.

- This crate owns scheduling and the `faber::HostDispatch` adapter only.
- Keep provider effects in `hosts/crates`; do not add Norma route logic
  here or in `hosts/crates/host-kernel`.
- Workers must be bounded, cancellation-aware, and explicitly shut down.
- Run `cargo fmt --check`, `cargo test`, and clippy before commits.
