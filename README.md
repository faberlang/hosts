# Faber hosts

Platform and browser **host products** for Faber-produced artifacts.

Radix compiles Faber and emits host ABI imports (`radix-host-abi`). This repo
owns the runtimes that load those artifacts and supply capabilities.

## Layout

```text
hosts/
  macos-arm64/       Cargo crate `faber-host-macos-arm64` (Wasm/component host proof)
  webgpu-browser/    Browser WebGPU host product (static + JS; not a Cargo member)
  scripta/           Host-local helpers (e.g. webgpu-browser-proof)
```

Each host implementation is its own product directory. Shared libraries remain
siblings under `faberlang/`:

| Sibling | Role |
| --- | --- |
| `faber-runtime/` | Runtime types (`use faber::…`) |
| `host-kernel-rs/` | Transport-neutral provider kernel |
| `host-providers-rs/` | Capability providers |
| `host-native-rs/` | Native `HostDispatch` adapter |
| `radix/` | Compiler (ABI contract only; no host runtime) |

## Build (macOS arm64 host)

From this repo root (requires sibling checkouts for path deps):

```bash
cargo build -p faber-host-macos-arm64
cargo test -p faber-host-macos-arm64
cargo run -p faber-host-macos-arm64 -- manifest
```

## WebGPU browser host

```bash
./scripta/webgpu-browser-proof generate   # needs sibling ../radix
./scripta/webgpu-browser-proof check
./scripta/webgpu-browser-proof serve
```

## What does not live here

- Compiler host ABI table → `radix/crates/radix-host-abi`
- Norma source → `norma/`
- Exempla e2e harness → `faber/crates/exempla`
