# Faber hosts

Platform host **products** and shared host **libraries** for Faber-produced artifacts.

Radix compiles Faber and emits host ABI imports (`radix-host-abi`). This repo owns:

1. **Libraries** — kernel, native adapter, capability providers  
2. **Products** — platform/browser runtimes that load artifacts and supply capabilities

## Layout

```text
hosts/
  crates/
    host-kernel/       package `host-kernel` — transport-neutral routing / manifests
    host-native/       package `host-native` — bounded native HostDispatch adapter
    aleator/ consolum/ http/ processus/ solum/ tempus/
    provider-contracts/
  macos-arm64/         product: Wasm/component host for macOS arm64
  wasm/                product: portable core-Wasm v1 host (`faber-host-wasm`)
  webgpu-browser/      product: browser WebGPU host (JS/static; not a Cargo member)
  scripta/             host-local helpers
```

Sibling path deps (not in this repo):

| Sibling | Role |
| --- | --- |
| `faber/runtime/rust/` | Generated Rust carriers (`use faber::...`) |
| `radix/` | Compiler (ABI contract only) |
| `faber/` | User CLI; embeds selected `hosts/crates/*` as core-support |

## Build

```bash
# From this repo root
cargo test --workspace
cargo test -p faber-host-macos-arm64
cargo run -p faber-host-macos-arm64 -- manifest

# Browser host (needs sibling ../radix)
./scripta/webgpu-browser-proof check
```

Library crates use **explicit** path deps so Faber's embedded core-support archive
can materialize individual crates without shipping this whole workspace.

## Former repo homes

| Old sibling | New path |
| --- | --- |
| `host-kernel-rs` | `crates/host-kernel` |
| `host-native-rs` | `crates/host-native` |
| `host-providers-rs` | `crates/{aleator,consolum,…}` |
