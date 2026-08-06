# faber-host-wasm

Portable core-Wasm product host for the closed `faber_rt_v1` host surface.

The single v1 product-host owner for the wasm-host-parity campaign (Stage 2).
It runs plain core-Wasm modules against the closed `faber_rt_v1` import
registry and returns typed outcomes. The runner consumes only Wasm bytes plus
an explicit `RunConfig` — never source, an interner, WAT, or an externally
reconstructed opaque-handle table.

```rust
let host = faber_host_wasm::WasmRtV1Host::new()?;
let outcome = host.run(&wasm_bytes, &faber_host_wasm::RunConfig::default());
match outcome {
    faber_host_wasm::RunOutcome::Success { stdout } => { /* captured stdout */ }
    other => { /* typed validation/import/link/entry/trap/runtime outcome */ }
}
```

Consumers:

- exempla's product-runner adapter calls this library directly and maps
  results to Stage 1 ledger outcomes (`faber/crates/exempla/src/exempla_e2e/
  wasm_product.rs`).
- Faber packaging consumes the same API later (defer-release).

Proof fixtures under `tests/fixtures/` are real compiler artifacts emitted by
the radix Wasm target from Stage 1 ledger fixtures (`sic/sic.fab`,
`per/per.fab`); both need no opaque-handle table.

## Test

```bash
cargo test -p faber-host-wasm
```
