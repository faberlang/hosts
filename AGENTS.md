# hosts — agent notes

Sibling monorepo for host libraries and host products. Not the Radix compiler.

## Orientation

| Path | Package / product |
| --- | --- |
| `crates/host-kernel` | `host-kernel` |
| `crates/host-native` | `host-native` |
| `crates/{aleator,consolum,http,processus,solum,tempus}` | providers |
| `macos-arm64` | `faber-host-macos-arm64` |
| `webgpu-browser` | browser WebGPU product |

Path deps expect sibling `faberlang/{faber-runtime,radix}` for library/product work.

## Invariants

1. Hosts are **execution** products and libraries. Radix does not need this repo to compile.
2. Emitters only know **ABI contracts** (`radix-host-abi` inside radix).
3. One directory per host **product**. Shared kernel/providers live under `crates/`.
4. Library crates keep **explicit** path deps (no workspace inheritance) so Faber core-support can embed them alone.
5. Do not restore `host-kernel-rs` / `host-native-rs` / `host-providers-rs` as live source trees.

## Validation

```bash
cargo test --workspace
cargo test -p faber-host-macos-arm64
./scripta/webgpu-browser-proof check   # optional; needs node + sibling faber/radix/triga
```
