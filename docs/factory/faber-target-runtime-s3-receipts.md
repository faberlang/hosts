# faber-target-runtime Stage 3 — Hosts Provider Move Receipts

**Campaign**: `faber-target-runtime` (gol_8dc80e638c8b8037) — Stage 3
(re-target hosts dependencies)
**Lane**: factory/hand-5 (hosts)
**Authority**: `radix/docs/factory/faber-target-runtime/stage0/inventory.md`
§3.2 (intra-module splits, never dual authority) + DDPP0 runtime inventory
§1 destination rows (HOSTS-PROVIDERS, HOSTS-COORD, HOSTS-LLVM)
**Operating law**: memo `a97edf13` (deferred verification — no per-unit green;
breakage is expected and recorded)

Receipts below record where each former `faber-runtime` provider surface now
**has its forward home** in hosts. The old `faber-runtime` copies remain as
migration carriers until S8C deletion; the destination listed here is the
single forward owner.

## S3-U3 receipts (concrete providers)

### 1. `frame` split — builtin route arms → hosts providers

Per inventory §3.2: `frame`'s primary destination is `faber/runtime/rust/`
(GENRUST-CONTRACT; done in S1-U2/U3); the intra-module split
`builtin_route_frames`/`dispatch_builtin_route` implementation arms
(tempus/solum/processus/consolum/aleator) land with **HOSTS-PROVIDERS**.

**Status: forward authority confirmed — hosts provider crates.** The five
provider crates (`crates/tempus`, `crates/solum`, `crates/processus`,
`crates/consolum`, `crates/aleator`) plus `crates/http` are the concrete
implementations; the `crates/host-native` crate installs them behind
`faber::install_host_dispatch` (HostDispatch over the GENRUST-CONTRACT).

Coverage verification (S3-U3): every `builtin_route_key` route in the
faber-runtime carrier (83 routes) is covered by a provider manifest +
dispatch, with exactly three exceptions, all deliberate and recorded:

| Route | Status | Receipt |
| --- | --- | --- |
| `processus:exi` | intentionally unmanifested | no protocol-visible terminal response for host exit yet (coverage packet `provider-manifest-dispatch-coverage.md`); rejected with `E_NO_ROUTE` |
| `tempus:expectet` | legacy alias, unmanifested | omitted from manifest by policy (sync/async pair surfaces only) |
| `runtime:echo` | carrier-only route name | hosts kernel builtin is `host:echo` (`macos-arm64/tests/host_kernel_test.rs`); name reconciliation is a Stage 8B item, not a second echo implementation |

The faber/runtime/rust `frame` module fails closed on uninstalled dispatch
(S1-U3 `start_host_dispatch`); the builtin fallback lives with the providers
in hosts — no dual authority in the runtime contract.

### 2. `host_abi` split — LLVM host side → hosts/llvm

Per inventory §3.2/§3.4: `host_abi`'s contract side (symbols/layouts) is
radix-owned (`radix-host-abi`); the **LLVM host side** of the symbols moves
to **HOSTS-LLVM**.

**Status: moved (S3-U3).** `faber-host-llvm` crate now lives at `hosts/llvm/`
(workspace member; produces `libfaber_host_llvm.a`). Wiring:
- `faber::host_abi::*` imports re-pointed to `radix_host_abi` (the ABI
  authority).
- Opaque `FaberRtContextV1` carried crate-locally (`llvm/src/abi.rs`) pending
  radix-host-abi adoption (radix scope; residual).
- Cargo deps: `faber` → `faber/runtime/rust` contract, plus `radix-host-abi`.

Archive identity + SHA-256 content receipt and the faber build re-source are
Stage 7 (`S7-U1`/`S7-U2`); this receipt closes the crate move.

### 3. `http` → hosts provider (S1-U3 landing verified)

Per inventory §3.2: `http` client effects → **HOSTS-PROVIDERS** (one provider
authority, no runtime fallback duplication).

**Status: landed (S1-U3) + wiring verified (S3-U3).** `crates/http` carries
the `http:*` provider (manifest: `listen`/`accept`/`respond`/`stop`) and the
concrete HTTP client effects (`src/client.rs`, `pub mod client`). Nothing
missing; no additional move required.

## S3-U4 receipts (device/session moves)

*See `crates/host-coordinator` (appended by S3-U4).*
