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

Per inventory §3.2 + DDPP0 row 7 (HOSTS-COORD), the device-lifecycle surface
moves to hosts as the **`host-coordinator`** crate
(`crates/host-coordinator`), in DDPP-ordered sequence. The faber-runtime
copies remain migration carriers until S8C; the coordinator is the single
forward owner.

| Former faber-runtime module | Destination (this repo) | Receipt |
| --- | --- | --- |
| `device_identity` | `crates/host-coordinator/src/device_identity.rs` | moves with the physical device identity surface |
| `device_set` | `crates/host-coordinator/src/device_set.rs` | moves with the device topology surface |
| `discovery` | `crates/host-coordinator/src/discovery.rs` | moves with the discovery surface |
| `bound_plan` | `crates/host-coordinator/src/bound_plan.rs` | moves with the multi-device coordinator surface |
| `capability` | `crates/host-coordinator/src/capability.rs` | backend-capability result types are device-lifecycle facts |
| `execution_transaction` | `crates/host-coordinator/src/execution_transaction.rs` + `execution_transaction/{backend,errors,mirror,receipt,reservation,state_machine,transaction}.rs` | all 7 submodule files move together (MD3-X1) |
| `partition` | `crates/host-coordinator/src/partition.rs` | moves with the partition/coordinator surface |
| `policy` | `crates/host-coordinator/src/policy.rs` | moves with the multi-device policy surface |
| `transport` | `crates/host-coordinator/src/transport.rs` | moves with the transport/coordinator surface |
| `device` (split) | `crates/host-coordinator/src/device_handle.rs` | `DeviceHandle`/`DeviceHandleKind` (physical-handle carriers) land with HOSTS-COORD; the selection/build metadata half (`DeviceSelection`, `from_spelling`) stays RADIX-ARTIFACT+FABER-BUILD |

**Split + wiring decisions (recorded, never dual authority):**

1. **`DeviceBackend` physical discriminator** — carried by `host-coordinator`
   (`src/backend.rs`): identity, topology, discovery, bound-plan, transport,
   and handle types all require the physical backend fact. The
   *selection/build metadata* surface (`DeviceSelection`, selection spelling)
   remains RADIX-ARTIFACT+FABER-BUILD; `macos-arm64` keeps importing it from
   `faber::device` (expected breakage until S8A re-points it to the Radix
   artifact-contract selection surface / Faber build config — the faber CLI
   resolves selection and passes the backend). The `metal`/`cuda` spellings
   are stable ABI facts; S8B reconciles if the Radix side defines its own
   enum.
2. **`host-coordinator` deps** — consumes the GENRUST-CONTRACT (`faber` =
   `faber/runtime/rust`) for `Valor`/`Json`/frame carriers only; no device
   session behavior enters the runtime contract (C2).
3. **Cross-destination re-points (expected breakage, recorded):**
   `bound_plan`/`capability` reference `faber::model_format` (GRADUS-PML —
   `PinnedDtype`, `sha256`) and `faber::repack_plan` (RADIX-LEAF —
   `RepackDescriptor`/`RepackSelection`/`RowIdentity`); those destinations
   land in the campaign's gradus/radix moves (S8A) and the imports re-point
   then.
4. **`macos-arm64` consumers** — `device_descriptor.rs`, `device_host.rs`,
   `composite_host{,.rs,/,receipt.rs}` and the program-graph-hash /
   composite-host tests import `DeviceBackend`/`DeviceHandle`/
   `DeviceHandleKind` from `host-coordinator` (new path dep). The faber
   `DeviceSelection` import lines remain as residuals (see row 1).
5. **Test files moved with their modules** (`*_test.rs`), wired by the same
   `#[path = "..."]` declarations; `capability_test` is crate-root-wired (as
   in faber-runtime).
