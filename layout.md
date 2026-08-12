# Hosts Multi-Language Native Package Layout

Layout authority for native host packages under this repository. The hosts
repo is a multi-language monorepo: Rust library crates and host products today,
with TypeScript, Go, and Swift host-package layouts frozen here for the
target-runtime packages (faber-target-runtime inventory §3.2 / §7
destinations).

Frozen by **S3-U1** of the `faber-target-runtime` campaign. Each row is one
destination with one owner; the old `faber-runtime` repo is a migration
carrier only (deleted at campaign closeout).

## Layout

```text
hosts/
  crates/              Rust library crates (the Rust native-package layer)
    host-kernel/       host-kernel       — transport-neutral provider kernel
    host-native/       host-native       — bounded HostDispatch adapter over the kernel
    host-coordinator/  host-coordinator  — multi-device coordinator surface (S3-U4)
    aleator/           aleator           — random provider
    consolum/          consolum          — console I/O provider
    http/              http              — HTTP provider + client effects
    processus/         processus         — process provider
    solum/             solum             — filesystem provider
    tempus/            tempus            — time provider
    provider-contracts/                  — composed provider lifecycle contracts
  macos-arm64/         faber-host-macos-arm64 — macOS arm64 host product (Rust)
  llvm/                faber-host-llvm   — LLVM host runtime crate (Rust) (S3-U3)
  wasm/                faber-host-wasm   — core-Wasm host product (Rust)
  typescript/          TypeScript host package layout (skeleton)
  go/                  Go host package layout (skeleton)
  swift/               Swift host package layout (skeleton)
  webgpu-browser/      browser WebGPU host product (TypeScript)
```

## Rules

1. **One directory per host product.** Shared kernel/providers/coordinator
   live under `crates/`; product hosts are top-level directories
   (`macos-arm64/`, `wasm/`, `llvm/`, `webgpu-browser/`).
2. **Rust library crates keep explicit path deps** (no workspace
   inheritance) so Faber's embedded core-support archive can path-depend on
   individual crates without shipping this workspace root or product hosts.
3. **Non-Cargo native packages** (TypeScript, Go, Swift) keep their native
   package manifests (`package.json`, `go.mod`, `Package.swift`) inside their
   layout directory. They are not Cargo workspace members.
4. **Hosts are execution products and libraries.** Emitters only know ABI
   contracts (`radix-host-abi` inside radix); no private Radix source enters
   hosts (C2 isolation).
5. **No concrete host effect, device session, ML semantics, or private Radix
   source enters `faber/runtime/{target}`** — the runtime packages stay
   dependency-free of hosts, and hosts stays dependency-free of the runtime
   packages' internals (they may depend on the `faber/runtime/rust/`
   contract only).

## Destination authority

| Native package | Layout home | Owns |
| --- | --- | --- |
| Rust host libraries | `crates/` | kernel, providers, coordinator, dispatch |
| Rust host products | `macos-arm64/`, `wasm/`, `llvm/` | native/macOS host, core-Wasm host, LLVM host runtime |
| TypeScript host packages | `typescript/` (skeleton) | generated-TS host surface (Stage 4 consumers) |
| Go host packages | `go/` (skeleton) | generated-Go host surface (Stage 5 consumers) |
| Swift host packages | `swift/` (skeleton) | generated-Swift host surface (Stage 6 consumers) |
| Browser product | `webgpu-browser/` | WebGPU browser runtime behavior |
