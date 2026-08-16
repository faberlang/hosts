# Agent Instructions

Work in this crate's checkout on `main` unless the operator assigned a
packet. Container law: [`../../../AGENTS.md`](../../../AGENTS.md).

- This crate owns scheduling and the `faber::HostDispatch` adapter only.
- Keep provider effects in `hosts/crates`; do not add Norma route logic
  here or in `hosts/crates/host-kernel`.
- Workers must be bounded, cancellation-aware, and explicitly shut down.
- Run `cargo fmt --check`, `cargo test`, and clippy before commits.
