# DELIVERY: device-session-byte-surface — dtype-tagged bytes both directions; packed weights reach every backend

**Status**: lowered 2026-08-21 — ready for Mind to file Hands
**Goal:** [`goal.md`](goal.md) (goal-check verdict: **READY** — record below)
**Campaign:** `emission-lane-parity` ELP-07 / finding F3.2 (radix [`CAMPAIGN.md`](../../../../radix/docs/factory/emission-lane-parity/CAMPAIGN.md))
**Repos:** `hosts` only (all units); no radix writes
**Cross-campaign seam:** D5 (vivi `5da8539d`, KV control-protocol v2) owns `macos-arm64/src/device_execute.rs` + `tests/device_execute_cli_test.rs` until it lands. **DSB-4 writes those files and is scheduled strictly after D5.**

---

## 0. Goal check (gate for this lowering)

**Verdict: READY.** Evaluator: planner. Consumer: delivery.

| Goal claim | Live check | Result |
| --- | --- | --- |
| Trait exposes only `copy_in_f32`/`readback_f32` | `device_host.rs:380, 395` | live, exact |
| CUDA path requires 4-byte multiples, reinterprets as f32 | free fn `copy_in_bytes`, `device_host.rs:698-719` | live, exact |
| Packed bytes + mmap retention are Metal-only methods | `metal_host.rs:531-535` (`retain_mapped_file`), `:542-566` (`copy_in_packed_bytes`) | live, exact |
| GGUF mmap retained only when runtime is Metal | **citation drifted**: goal says `device_execute.rs:919-921`; the block now sits at `device_execute.rs:1342-1352` (`retain_mapped_weights`, `if let Some(DeviceRuntime::Metal(session))`) | live in substance; D5's in-flight control-protocol work inserted ~430 lines above it |
| `DeviceDataType` cannot name f16/bf16 | `device_descriptor.rs:61` — F32/F64/I32/I64/U8 only; `DtypeSurface` (`host-coordinator/src/discovery.rs:121+`) says devices execute f16/bf16 | live, exact |

Architecture direction is settled in the goal (neutral surface, mmap as declared
capability, dtype vocabulary, one placement contract); open questions carry
defaults. No blocking gaps. The citation drift is recorded, not a defect of the
goal's substance.

---

## 1. Interpreted Unit

F3.2: the backend-neutral `DeviceSession` transfer surface is f32-only. Metal
carries the real byte path as private extras (`copy_in_packed_bytes`,
`retain_mapped_file`); the shared free-fn byte entry reinterprets bytes as f32
words and rejects non-4-multiple lengths on CUDA; `device-execute` retains the
GGUF mmap only for Metal, so a discrete CUDA GPU has no way to receive
Q8_0/Q4_K packed weight regions. The dtype vocabulary cannot even name f16/bf16.

## 2. Normalized Spec

The `DeviceSession` trait carries **dtype-tagged byte buffers in both
directions** as the neutral surface: `copy_in_bytes(handle, &[u8],
DeviceDataType)` and `readback_bytes(handle, DeviceDataType) -> Vec<u8>`, with
the f32 methods as delegating wrappers (goal OQ1 default: least caller churn).
Every admitted backend implements the byte methods on raw bytes — no transfer
path reinterprets bytes as f32. Mmap retention stays Metal-only but is declared
through a narrow capability accessor the executor consults, making the
`device-execute` weight-upload path backend-neutral. `DeviceDataType` gains
F16/BF16 wired to the single placement-dtype contract (radix
`placement-debt-audit` F2 owns the discriminant; hosts coordinates, does not
duplicate).

Delivery-level non-goals (inherited from goal): no in-kernel dequant (EXEC-02);
no pinned-host memory; no HTTP/remote transport; no ELP-09 capability-model
generalization; no radix writes.

## 3. Repo-Aware Baseline

| Surface | Today | Note |
| --- | --- | --- |
| `DeviceSession` trait | `device_host.rs:369-397` — `copy_in_f32`/`readback_f32` only | the ABI seam every backend implements |
| Free fn `copy_in_bytes` | `device_host.rs:698-719` — Metal→`copy_in_packed_bytes`; CUDA→4-byte check + f32 reinterpret via `copy_in_f32` | sole caller: invocation-state upload, `device_host.rs:216` |
| Metal byte extras | `metal_host.rs:531-535` (`retain_mapped_file`), `:542-566` (`copy_in_packed_bytes`, 1–3-byte tail admission), `readback_f32` over `copy_out` at `:720-732` | real byte machinery already exists |
| CUDA transfer | `cuda_host.rs:216` (`copy_in_f32` → `driver.copy_in(token, bytes)`), `:320` (`readback_f32`); cuMemcpyHtoD/DtoH symbols resolved `:864-865` | driver seam is already raw bytes — only the session layer reinterprets |
| Weight upload | `device_execute.rs`: `WeightInputs` `:1100-1126` (f32-typed map contract `:1117`), weight-map regions pass through `packed_bytes_as_native_region` (`:1080`), mmap retention Metal-only (`retain_mapped_weights` `:1342-1352`, called `:590`, `:798`) | `device_mut()` direct access is an established pattern here |
| Dtype vocabulary | `device_descriptor.rs:61-107` — F32/F64/I32/I64/U8 | no F16/BF16 slots |
| Test seams | `FakeCudaDriver` (`cuda_host.rs`, `tests/cuda_host_test.rs`), `tests/metal_host_test.rs`, `tests/device_execute_cli_test.rs` | fake-driver CUDA proof is live harness |
| Foreign lanes | D5 (`5da8539d`, live): `device_execute.rs` + `device_execute_cli_test.rs` (currently `M` in tree — its WIP, untouched). F4H2 (per D5 task text): `composite_host/session.rs` + `prepared_session_test.rs` | DSB-4 waits on D5; no unit writes session.rs |

## 4. Stage Graph — Hand units

```text
DSB-1 (trait bytes + Metal) ──> DSB-2 (CUDA raw bytes) ──┐
DSB-3 (dtype F16/BF16, parallel) ────────────────────────┼──> DSB-4 (device-execute weight path)
D5 (5da8539d) landed ────────────────────────────────────┘   [HARD seam: see DSB-4]
```

### DSB-1 — Neutral dtype-tagged byte transfer on `DeviceSession`; Metal implements, CUDA transitional

| Field | Value |
| --- | --- |
| outcome | Trait (`device_host.rs:369`) gains `copy_in_bytes(&mut self, buffer, bytes: &[u8], dtype: DeviceDataType)`, `readback_bytes(&mut self, buffer, dtype) -> HostResult<Vec<u8>>`, and `supports_mapped_weight_retention(&self) -> bool`. `copy_in_f32`/`readback_f32` remain trait methods delegating to the byte methods with `F32` (composite_host call sites unchanged). Metal implements the byte methods over the existing `copy_in_packed_bytes`/`copy_out` machinery; the 1–3-byte tail admission stays as declared Metal behavior. Free fn `copy_in_bytes` (`:698-719`) is deleted; its caller (`:216`) uses the trait method. The `DeviceRuntime::Cuda` arm keeps today's observable behavior behind the new method, commented transitional — DSB-2 removes it. |
| write_scope | `hosts/macos-arm64/src/device_host.rs`, `hosts/macos-arm64/src/metal_host.rs`, `hosts/macos-arm64/tests/metal_host_test.rs` |
| done_when | Trait compiles with the three new methods; all existing f32 callers unchanged and green; Metal byte round-trip test uploads/reads back arbitrary-length payloads (incl. a 1–3-byte tail) byte-identically; free fn gone; capability accessor true for Metal, false for CUDA |
| depends_on | none |
| sanity | `cargo test -p faber-host-macos-arm64 --test metal_host_test` |
| non_goals | CUDA raw-byte semantics (DSB-2); F16/BF16 (DSB-3); ELP-09 capability model; any `device_execute.rs` change (D5 lane) |
| risk | medium — the trait is the ABI seam every backend and ELP-09 generalize on; wrapper direction bakes in the caller contract |
| integrable | yes |

### DSB-2 — CUDA raw-byte H2D/D2H; misaligned tails fail explicit, not silent

| Field | Value |
| --- | --- |
| outcome | `CudaHostSession` implements the byte methods over the existing raw-byte driver seam (`driver.copy_in`/`copy_out`; symbols at `cuda_host.rs:864-865`) — zero f32 reinterpretation in the CUDA path. The `DeviceRuntime::Cuda` dispatch arm drops the 4-byte-multiple f32 conversion. Non-4-multiple tails are rejected with a structured `HostError` naming dtype and length (goal OQ2 default: CUDA rejects misaligned tails rather than padding). |
| write_scope | `hosts/macos-arm64/src/cuda_host.rs`, `hosts/macos-arm64/src/device_host.rs` (Cuda dispatch arm only), `hosts/macos-arm64/tests/cuda_host_test.rs` |
| done_when | FakeCudaDriver test round-trips a 34-byte payload byte-identically (first-failing oracle, now green); misaligned-tail case returns the structured error; no CUDA path converts device bytes to f32 for transfer |
| depends_on | DSB-1 |
| sanity | `cargo test -p faber-host-macos-arm64 --test cuda_host_test` |
| non_goals | pinned-host memory (`cuHostRegister` — recorded follow-up); real-device proof (rung harness / CAP-02 records it) |
| risk | low |
| integrable | yes |

### DSB-3 — `DeviceDataType` F16/BF16 vocabulary; one placement dtype contract

| Field | Value |
| --- | --- |
| outcome | `DeviceDataType` (`device_descriptor.rs:61`) gains `F16`/`BF16` with spelling/`from_spelling`/`byte_width == 2`; the mapping to the placement-ABI dtype discriminant is documented at the enum, naming radix `placement-debt-audit` F2 as the single contract owner (coordinate read-only; no radix writes). |
| write_scope | `hosts/macos-arm64/src/device_descriptor.rs` (+ its existing test module/file) |
| done_when | Variants round-trip spelling/parse; byte widths correct; doc comment states the discriminant mapping and the F2 coordination |
| depends_on | none (parallel-safe — files disjoint from DSB-1/DSB-2) |
| sanity | targeted descriptor tests under `cargo test -p faber-host-macos-arm64` |
| non_goals | radix-host-abi edits; capability surface (ELP-09); wire-format changes |
| risk | low |
| integrable | yes |

### DSB-4 — Backend-neutral weight upload in device-execute (AFTER D5)

| Field | Value |
| --- | --- |
| outcome | `--weights/--weight-map` regions upload as dtype-tagged bytes through the neutral surface on **every** admitted backend; `retain_mapped_weights` (`device_execute.rs:1342-1352`) consults `supports_mapped_weight_retention()` instead of `if let Some(DeviceRuntime::Metal)`; the weight-map path stops producing f32-reinterpreted vectors (`packed_bytes_as_native_region` leaves the weight path at `:1080`); a CUDA-backed run receives Q8_0/Q4_K-shaped packed regions. Preferred mechanism: post-prepare direct upload from device-execute via the neutral trait methods (the `device_mut()` precedent is `retain_mapped_weights` itself), leaving the f32-typed `WeightInputs::map()` consumer contract untouched. |
| write_scope | `hosts/macos-arm64/src/device_execute.rs`, `hosts/macos-arm64/tests/device_execute_cli_test.rs` |
| depends_on | **DSB-1, DSB-2, and D5 (vivi `5da8539d`) landed** — HARD cross-campaign seam: D5 owns both files in this write scope right now (live WIP in the tree). Mind confirms D5's closeout receipt on `5da8539d` before spawning DSB-4; the Hand rebase-tolerates if hosts moved meanwhile. |
| boundary rule | If the Hand concludes the `WeightInputs::map()` contract (`device_execute.rs:1117`) must change shape in `composite_host/session.rs`: **STOP and report to Mind** — that file is F4H2's lane (named in D5's task). Never expand write scope silently. |
| done_when | A CLI-level test admits a packed (non-4-multiple) weight region on a non-Metal path (fake-CUDA seam where the harness allows); Metal mmap-retention behavior unchanged; grep-clean of `DeviceRuntime::Metal` special-cases in the weight upload path; existing weight/mmap CLI tests green |
| sanity | `cargo test -p faber-host-macos-arm64 --test device_execute_cli_test` |
| non_goals | inputs-hex f32 wire (open question 1); in-kernel dequant (EXEC-02); ELP-09 capability generalization |
| risk | medium — lands in D5's wake on a hot file; direct-upload vs contract-change is the named fork |
| integrable | yes (each commit compiles and preserves Metal behavior) |

## 5. Implementation Work (Mind pointers)

Each Hand task is a pointer: goal path + unit id + write_scope + done_when from
§4. Suggested spawn order: DSB-1 and DSB-3 in parallel immediately; DSB-2 on
DSB-1's receipt; DSB-4 only after DSB-2's receipt **and** D5's closeout on
`5da8539d`. Mind owns the goal ledger update (`goal.md` §Ledger) as units land.

## 6. Checkpoints And Gates

**Batching / split decision:** four Hands, no merge gate — every unit is
independently integrable; DSB-4 is the only sequential tail (two internal deps
plus the D5 seam). DSB-1→DSB-2 share the trait surface (serial); DSB-3 is
parallel throughout; DSB-4 is last.

**Lane-owned gates (named once, never copied onto child Hands):**

| Lane | Owns |
| --- | --- |
| lint | `cargo fmt -p faber-host-macos-arm64` per touched crate |
| test | `cargo test --workspace` (cwd `hosts`); cross-crate suites that consume the trait |
| merge | landing order (DSB-4 last); `./scripta/cuda-tier-f-proof` on pharos (RTX 5070 Ti) stays green — the f32 tier must not regress while the byte tier lands; recorded, not gated, per goal |

**Release posture:** `defer-release` — the trait additions are public API of
`faber-host-macos-arm64`, but the repo has no per-change release automation;
the surface lands with campaign milestone M2 (neutrality).

## 7. Validation

**First-failing oracle (write red before any green):** in
`hosts/macos-arm64/tests/cuda_host_test.rs`, a `FakeCudaDriver` test uploads a
34-byte payload (one Q8_0-block-shaped row, `len % 4 == 2`) through the device
byte surface and asserts `readback_bytes` returns the same 34 bytes. Today this
fails at `device_host.rs:705-710` — `CUDA invocation-state copy requires a
4-byte multiple, got 34 bytes` — proving F3.2 live. The delivery is not done
while an equivalent test cannot pass on every admitted backend.

**Closeout command:**

```bash
cd hosts && cargo test --workspace
```

**Delivery done-when:** all four units' `done_when` hold; the first-failing
oracle is green on Metal and fake-CUDA; the closeout command is green; the
pharos f32-tier proof is recorded green; Mind has updated the goal ledger.

## 8. Companion Skill Plan

- `$delivery` — this artifact (complete).
- Mind → Hand prep per §5; audit after landing per campaign ELP-07 row acceptance (M2 gate).
- `$campaign` — ELP-09 (`device-capability-generalization`) consumes this surface next; its dependency on ELP-07 is the dtype unit (DSB-3) plus the neutral surface (DSB-1/2).

## 9. Open Questions

1. **Inputs-hex path**: `--inputs` hex strings still reinterpret bytes as f32 words (`device_execute.rs:1220`, `packed_bytes_as_native_region`). Outside this delivery's weight-path contract; folding it in changes the shared inputs wire contract on F4H2's lane. Default: follow-up after F4H2 closes. **Flag for Mind:** the goal invariant reads "no transfer path silently reinterprets bytes as f32" — after DSB-1/2 the `DeviceSession` surface no longer reinterprets, but the CLI wire still pre-converts. Decide whether ELP-07 acceptance requires the follow-up inside this goal or as a named successor.
2. **Wrapper direction** (goal OQ1) — default taken: f32 stays a trait method delegating to bytes.
3. **Tail admission** (goal OQ2) — default taken: Metal admits 1–3-byte tails (declared); CUDA rejects misaligned tails explicitly.
4. **Endian** (goal OQ3) — default taken: native order on both admitted backends; assert, don't convert.
5. **DSB-4 mechanism fork** — direct post-prepare upload (preferred, device-execute-only) vs `WeightInputs` contract change (escalates to Mind; F4H2 seam).
