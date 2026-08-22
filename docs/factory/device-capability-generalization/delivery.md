# DELIVERY: device-capability-generalization — capability model describes and gates every backend

**Status**: lowered 2026-08-21 — ready for Mind to file Hands (after ELP-08 spine start; see §5)
**Goal:** [`goal.md`](goal.md) (goal-check verdict: **READY** — record below)
**Campaign:** `emission-lane-parity` ELP-09 / finding F3.3 (radix [`CAMPAIGN.md`](../../../../radix/docs/factory/emission-lane-parity/CAMPAIGN.md))
**Repos:** `hosts` only (all units); no radix writes
**Live-seat fences:** pedantic seat (open task a3a0a0a0) owns clippy debt in `crates/host-coordinator` (46+63) + `crates/solum`; H2 (md3h W1) owns `macos-arm64/src/lib.rs` (M) + new runtime-set/transaction files. See §3 Foreign lanes.

---

## 0. Goal check (gate for this lowering)

**Verdict: READY.** Evaluator: planner. Consumer: delivery.

| Goal claim | Live check | Result |
| --- | --- | --- |
| `DeviceCapabilities { compute_capability, sm_count, dtype_surface }` — first two CUDA-only | `host-coordinator/src/discovery.rs:103-110` | live, exact |
| Policy compatibility gate checks exactly those CUDA facts; cannot evaluate Metal | `policy.rs:60-68` (`GateClass::Compatibility` doc: "compute capability, SM count, dtype surface"); `evaluate` at `:467` | live, exact |
| No generic launch-resource fields anywhere; Metal's enforced threadgroup ceiling private to `metal_host.rs` | **citation drifted**: the limit now sits at `metal_host.rs:1846-1856` (`pipeline.max_total_threads_per_threadgroup()`, launch-time reject), not `:1693-1706`; still private and invisible to planning. `partition.rs` has only memory-shaped `SafePhysicalLimit`; `bound_plan.rs` rejects admission/topology/staleness only — no launch-resource limits | live in substance |
| `DeviceDataType` (F32/F64/I32/I64/U8) cannot name f16/bf16 | **stale**: F16 landed with the archived byte-surface goal (DSB-3, hosts `563ea2b`); `device_descriptor.rs:58-78` documents the discriminant table incl. F16; BF16 documented slotless pending radix `placement-debt-audit` F2 | unit 3 re-scoped to the consistency ratchet (below) |
| DtypeSurface says devices execute f16/bf16 | `discovery.rs:121-137` (`DtypeSurface` incl. `f16`/`bf16` flags) | live, exact |

Baseline: `cargo test -p host-coordinator --lib` = 131 passed, 0 failed
(2026-08-21, this checkout) — units below turn new tests red-first against a
green baseline.

## 1. Interpreted theme

F3.3: the capability model is too CUDA-shaped to gate Metal (policy consumes
CUDA-only identity facts) and too thin to plan CUDA launches (no
threadgroup/shared-memory/warp/unified-memory facts anywhere in the model).
The bias runs both directions.

## 2. Normalized spec

`DeviceCapabilities` gains first-class generic launch-resource fields —
`max_threads_per_workgroup`, `workgroup_shared_memory_bytes` (minimum
guaranteed **and** max opt-in as two fields, goal OQ2 default),
`collective_width` (warp/simdgroup), `unified_memory: bool` — populated at
discovery from live per-backend queries (Metal: `MTLDevice` limits including
the threadgroup ceiling currently private at `metal_host.rs:1846-1856`; CUDA:
the `cuDeviceGetAttribute` set — max threads per block, shared mem per block
opt-in, warp size, integrated flag). `compute_capability`/`sm_count` remain
CUDA identity facts consumed only where arch matters (ELP-04 PTX negotiation).
The policy compatibility gate consumes generic fields so both backends
evaluate. `DtypeSurface`↔`DeviceDataType` consistency becomes a pinned test
(F16 named; BF16 slotless as the documented radix-F2 exception).
`bound_plan` consumes the limits fail-closed: a plan exceeding a device's
threadgroup/shared limits rejects with a named typed error rather than
launching.

Delivery-level non-goals (inherited): no new backends; no scheduler/
partitioning redesign (fail-closed checks only); CUDA timing-counter zeros
recorded as a known gap for device-executor.

## 3. Repo-aware baseline

| Surface | Today | Note |
| --- | --- | --- |
| Capability facts | `discovery.rs:103-110` (`DeviceCapabilities`), `:121+` (`DtypeSurface`); tests `discovery_test.rs` (module-inline `_test.rs` convention) | DCG-1 surface |
| Policy gate | `policy.rs` `GateClass` `:60-68`, `evaluate` `:467`, tests `policy_test.rs` | DCG-2 surface |
| Metal population source | `metal_host.rs:1846-1856` private launch-time threadgroup reject; `MTLDevice` limits otherwise unused by the model | DCG-1 surface (file is H2-clean) |
| CUDA population source | `cuda_host.rs:754-762` `cuda_device_attribute` helper already wraps `cuDeviceGetAttribute` (used for identity facts at `:702-716`) | DCG-1 surface; rides CLH-2's move if ELP-08 lands first (see §5) |
| Dtype vocabulary | `device_descriptor.rs:58-107` — F16 present, BF16 slotless documented (radix F2 dependency) | DCG-3 surface |
| Plan bind | `bound_plan.rs` — typed rejects for admission/stale/topology only; no launch-resource limits; tests `bound_plan_test.rs` | DCG-4 surface |
| Foreign lanes | **Pedantic seat (a3a0a0a0, open)**: clippy debt `host-coordinator` 46+63 + `solum`. **H2 (live)**: `macos-arm64/src/lib.rs` (M) + `device_runtime_set.rs`/`transaction_backend.rs` (+ tests). | DCG-1/2/4 write `host-coordinator` — semantic edits only, no clippy churn (boundary rule); no DCG unit writes H2 files or `lib.rs` |

## 4. Stage Graph — Hand units

```text
DCG-1 (fields + population) ──> DCG-2 (policy gate) ──> (goal closeout)
   │
   └──────────────────────────> DCG-4 (bound_plan fail-closed)
DCG-3 (dtype consistency ratchet; parallel from t=0)
```

### DCG-1 — Generic launch-resource fields + per-backend population

| Field | Value |
| --- | --- |
| outcome | `DeviceCapabilities` (`discovery.rs:103`) gains `max_threads_per_workgroup: u32`, `workgroup_shared_memory_min_bytes: u32`, `workgroup_shared_memory_max_bytes: u32`, `collective_width: u32`, `unified_memory: bool` (goal OQ1 default: neutral workgroup-family spellings already used across radix MIR). Population per backend from live queries: Metal populates from `MTLDevice` (incl. the threadgroup ceiling now private at `metal_host.rs:1846-1856`, surfaced without moving the launch-time reject); CUDA populates via the existing `cuda_device_attribute` seam (max threads per block, shared-mem opt-in min/max, warp size = `CU_DEVICE_ATTRIBUTE_WARP_SIZE`, integrated flag). Fake snapshots in tests carry both backends' shapes. |
| write_scope | `hosts/crates/host-coordinator/src/discovery.rs` (+ `discovery_test.rs`), `hosts/macos-arm64/src/metal_host.rs` (population only), `hosts/macos-arm64/src/cuda_host.rs` (population only — if ELP-08 CLH-2 landed first, the moved path `hosts/crates/host-cuda/src/cuda_host.rs`) |
| done_when | New `discovery_test` cases: fake Metal and fake CUDA snapshots populate all five fields with distinct values (first-failing oracle: does not compile today — fields absent); existing capability tests green; no caller of `DeviceCapabilities` construction left unpopulated (compile-enforced) |
| depends_on | none |
| sanity | `cargo test -p host-coordinator --lib` |
| non_goals | policy-gate semantics (DCG-2); bound_plan checks (DCG-4); new backends; moving the Metal launch-time reject |
| boundary rule | `host-coordinator` is the pedantic seat's clippy-debt target: semantic edits only — no drive-by clippy churn; Mind sequences the pedantic burn-down after this lands or holds it to `solum`. |
| risk | medium — struct growth compile-ripples through every construction site; field set is the ELP-04 planning contract |
| integrable | yes |

### DCG-2 — Policy compatibility gate on generic fields (Metal evaluable)

| Field | Value |
| --- | --- |
| outcome | The `Compatibility` gate (`policy.rs`) consumes the generic launch-resource fields (goal proposal 2) so a Metal-shaped snapshot evaluates cleanly instead of being unevaluable; `compute_capability`/`sm_count` demote to CUDA identity facts consumed only where arch matters. A Metal-device gate test proves a Metal snapshot passes/fails on generic facts (e.g. threadgroup ceiling) — today no such evaluation is possible. |
| write_scope | `hosts/crates/host-coordinator/src/policy.rs` (+ `policy_test.rs`) |
| done_when | New `policy_test`: a Metal-shaped snapshot is admitted by `evaluate`, and a snapshot whose plan demands exceed its `max_threads_per_workgroup` is rejected with the compatibility reject reason (first-failing oracle: fails today — gate reads CUDA-only facts); all existing policy tests green |
| depends_on | DCG-1 |
| sanity | `cargo test -p host-coordinator --lib policy` |
| non_goals | ranking/scoring (gates reject, never rank); topology/memory gates (unchanged) |
| boundary rule | Same pedantic-seat rule as DCG-1. |
| risk | low |
| integrable | yes |

### DCG-3 — DtypeSurface↔DeviceDataType consistency ratchet

| Field | Value |
| --- | --- |
| outcome | A pinned consistency test stating the rule: every dtype `DtypeSurface` claims executable must be nameable by `DeviceDataType` on the transfer surface, with BF16 as the single documented slotless exception (radix `placement-debt-audit` F2 owns the discriminant — coordinate read-only, no radix writes). Re-scoped from the goal's "F16/BF16 slots" — F16 already landed (DSB-3, `563ea2b`); this unit pins the invariant so the model can never again claim an unnameable dtype. Honest oracle note: this lands **green** (ratchet), not red-first — the defect it guards against is regression. |
| write_scope | one new/existing matching `_test.rs` under `hosts/macos-arm64/tests/` (test-only; no production file changes) |
| done_when | Test enumerates `DtypeSurface` flags ↔ `DeviceDataType` spellings; F16 asserted named; BF16 asserted slotless **with the F2 dependency named in the assertion message**; runs in the macos-arm64 suite |
| depends_on | none (parallel-safe; disjoint from DCG-1/2/4 files) |
| sanity | the new test under `cargo test -p faber-host-macos-arm64` |
| non_goals | adding a BF16 slot (radix F2 dependency — not hosts' to unblock); dtype semantics changes |
| risk | low |
| integrable | yes |

### DCG-4 — bound_plan fail-closed launch-resource limits

| Field | Value |
| --- | --- |
| outcome | `bound_plan` consumes DCG-1's generic limits fail-closed: binding a plan whose declared threadgroup volume or shared-memory demand exceeds the target device's `max_threads_per_workgroup` / shared-memory fields rejects with a named typed reject (the file's existing reject vocabulary), before any launch path can see it. Metal's private launch-time reject (`metal_host.rs:1846-1856`) stays as defense-in-depth. |
| write_scope | `hosts/crates/host-coordinator/src/bound_plan.rs` (+ `bound_plan_test.rs`) |
| done_when | New `bound_plan_test`: an oversized-threadgroup plan against a fake device snapshot rejects with the named error (first-failing oracle: fails today — binds clean); an in-limits plan binds unchanged; existing bind-contract tests green |
| depends_on | DCG-1 |
| sanity | `cargo test -p host-coordinator --lib bound_plan` |
| non_goals | scheduler/partition redesign; per-kernel rewrites of the plan; metal_host launch reject removal |
| boundary rule | Same pedantic-seat rule as DCG-1. |
| risk | low |
| integrable | yes |

## 5. Implementation Work (Mind pointers)

Each Hand task is a pointer: goal path + unit id + write_scope + done_when
from §4. Spawn order: DCG-1 and DCG-3 in parallel immediately; DCG-2 and
DCG-4 on DCG-1's receipt (parallel with each other — disjoint files).
Sequencing note vs ELP-08: DCG-1 touches `cuda_host.rs` for population — if
CLH-2 (crate move) is in flight, Mind spawns DCG-1 against the post-move
path or holds the CUDA population half one turn; Metal half and the
host-coordinator fields are never blocked.

## 6. Checkpoints And Gates

**Batching:** four Hands, no merge gate — every unit independently
integrable. DCG-1→{DCG-2, DCG-4} is the only spine; DCG-3 parallel
throughout.

**Lane-owned gates (named once, never copied onto child Hands):**

| Lane | Owns |
| --- | --- |
| lint | `cargo fmt` per touched crate; the pedantic clippy-debt task (a3a0a0a0) on `host-coordinator`/`solum` is sequenced by Mind around these semantic units, never folded into them |
| test | `cargo test -p host-coordinator -p faber-host-macos-arm64` per spine landing; `cargo test --workspace` (cwd `hosts`) at goal closeout |

## 7. Open questions for Mind

1. **DCG-1 CUDA population vs CLH-2 move** — spawn-order ruling (§5) when
   both goals run concurrently.
2. **Shared-memory opt-in tiers** modeled as min/max pair per goal OQ2
   default — flag if ELP-04's planner needs a different shape before DCG-1
   lands (post-landing widening is additive, not breaking).
3. CUDA timing-counter zeros stay recorded as a device-executor gap — no
   unit here; confirm no auditor expectation otherwise.
