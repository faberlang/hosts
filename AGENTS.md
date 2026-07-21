# hosts — agent notes

Sibling repo of platform/browser host products. Not the Radix compiler.

## Orientation

- **macOS Wasm host** → `macos-arm64/` (`faber-host-macos-arm64`)
- **Browser WebGPU host** → `webgpu-browser/`
- **Proof script** → `./scripta/webgpu-browser-proof`
- Path deps expect `faberlang/{faber-runtime,host-kernel-rs,host-providers-rs,radix}` as siblings

## Invariants

1. Hosts are **execution** products. Radix does not need this repo to compile.
2. Emitters only know **ABI contracts** (`radix-host-abi` inside radix).
3. One directory per host product. Do not fold hosts into radix or faber CLI.
4. Shared kernel/providers stay in their own sibling repos; depend via path (or later published crates).
5. Clean break: no radix workspace member for hosts.

## Validation

```bash
# From hosts/
cargo test -p faber-host-macos-arm64
./scripta/webgpu-browser-proof check   # optional; needs node + sibling radix
```

## Out of scope here

Compiler changes, Norma package source, exempla corpus harness, cista packaging.
