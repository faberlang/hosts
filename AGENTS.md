# hosts — agent notes

Sibling monorepo for host libraries and host products. Not the Radix compiler.

**Workspace work mode.** Ordinary development is **direct** in this
checkout on `main`. Worktree packets under `../worktrees/<lane>/` are
optional Tugboat isolation. Do not stand up lanes unless the operator
asked. Container law: [`../AGENTS.md`](../AGENTS.md).

## Orientation

| Path | Package / product |
| --- | --- |
| `crates/host-kernel` | `host-kernel` |
| `crates/host-native` | `host-native` |
| `crates/host-coordinator` | `host-coordinator` |
| `crates/provider-contracts` | shared provider contracts |
| `crates/{aleator,consolum,http,processus,solum,tempus}` | providers |
| `macos-arm64` | `faber-host-macos-arm64` |
| `webgpu-browser` | browser WebGPU product |

Path deps expect sibling `faberlang/{faber,radix}` for library/product work.
Public generated-Rust carriers live under `faber/runtime/rust`.

All host work follows the canonical public
[Host Execution Architecture](https://github.com/faberlang/faber/blob/main/docs/host-execution-architecture.md).
Hosts is a microkernel for physical capabilities and execution. Libraries own
portable behavior; Radix supplies complete target artifacts and execution
descriptors. A new library operation must not require a Hosts implementation.
If a future Linux, Windows, browser, or device host would have to copy the
code, it belongs in the owning library or Radix unless it is a genuinely new
physical capability.

GPU work additionally follows
[GPU Execution Architecture](https://github.com/faberlang/faber/blob/main/docs/gpu-execution-architecture.md).
Gradus owns ML semantics, logical placement and sharding intent, and all ML
kernel source in Faber. Radix compiles target artifacts and explicit execution
facts. Hosts owns physical discovery, virtual-partition admission, binding,
residency, launch, synchronization, and readback. Do not add or template an ML
kernel body, decode a packed representation as an operator, recover missing
model facts from names or resource extents, probe semantic alternatives, or
silently run a CPU implementation on a declared GPU path. A missing artifact
or descriptor fact is an upstream failure and must fail closed here.

## Invariants

1. Hosts are **execution** products and libraries. Compiler-only Radix builds do
   not need this repo; Faber product builds with host features do.
2. Emitters only know **ABI contracts** (`radix-host-abi` inside radix).
3. One directory per host **product**. Shared kernel/providers live under `crates/`.
4. Library crates keep **explicit** path deps (no workspace inheritance) so Faber core-support can embed them alone.
5. Do not restore `host-kernel-rs` / `host-native-rs` / `host-providers-rs` as live source trees.
6. Existing library-semantic bodies in this repo are migration debt, not
   extension points or precedent.

## Validation

```bash
./scripta/stage1                       # static gate: rustfmt check + execution-purity ratchet
cargo test --workspace
cargo test -p faber-host-macos-arm64
./scripta/webgpu-browser-proof check   # optional; needs node + sibling faber/radix/triga
```
