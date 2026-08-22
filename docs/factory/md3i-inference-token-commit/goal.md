# GOAL: md3i-inference-token-commit — bind execution transactions to inference token/sequence commit

**Status**: planned — lowered 2026-08-22 ([`md3i-delivery.md`](../../../../radix/docs/factory/gpu-inference-multi-device/md3i-delivery.md)); ready for unit tasking (entry gate met: MD3H archived `7014785`, Gradus PML5 decode/KV semantics structural-tier delivered, hosts D1–D7 session/KV facts live)
**Created**: 2026-08-22
**Campaign:** `gpu-inference-multi-device` (radix: [`docs/factory/gpu-inference-multi-device/CAMPAIGN.md`](../../../../radix/docs/factory/gpu-inference-multi-device/CAMPAIGN.md))
**Source:** CAMPAIGN.md §MD3I + the 2026-08-21 amendment (MD3I follows MD3H); lowered spec [`md3i-delivery.md`](../../../../radix/docs/factory/gpu-inference-multi-device/md3i-delivery.md); frozen session facts [`gi4-contract.md`](../../../../radix/docs/factory/gpu-inference-gguf/gi4-contract.md) §1–§6; ownership amendment [`gi4-ownership-amendment.md`](../../../../radix/docs/factory/gpu-inference-gguf/gi4-ownership-amendment.md)
**Repos:** `hosts` (primary: `macos-arm64` product host — the commit binding is a consumers-side layer; `host-coordinator` is **read-only**), `radix` (fixture surface only)
**Related:** archived [`md3h-virtual-partition-host`](../../archived/md3h-virtual-partition-host/goal.md) (uniform virtual-partition path + real backend + distributed seam) · archived [`kv-cache-model-session`](../../../../radix/docs/archived/kv-cache-model-session/goal.md) (D1/D4 logical sequence machine MD3I binds) · device-executor M8 / gpu-production-readiness (seam adjacency — see MD3I-WIN)

---

## Invariant

Token id, sequence position, KV generations, and visible output advance
**together**, and **only** after the mesh-wide `ExecutionTransaction` commit.
Failure or cancellation before commit leaves the last committed
token/position authoritative; retry is disabled unless replay from the last
committed generation is proven deterministic. N=1 is the one-rank case of the
same commit rule. The generic transaction stays generic: training callers
consume MD3H without importing token/sequence semantics.

## Problem

The generic `ExecutionTransaction` (MD3/MD3H) publishes staged writes
atomically but carries no inference vocabulary — its own module doc names the
inference binding as MD3I's surface. The live D1/D4 `InferenceSessionState`
already implements the transactional token-mutation rule at the **logical
sequence tier** (commit/poison, never retryable), and the KV-D dense full-model
run executes prefill + changing-token decode on the single-device v2 seam —
but neither path is bound to the generic transaction publication. The
distributed seam stops at `prepare` (FakeExecutionBackend). No `TokenCommit`
surface exists anywhere. Without the binding, N=1 and N=8 token/sequence
advance would stay uncoordinated and MD4A+ capacity/KV/MoE stages would build
commit semantics on sand.

## Proposal

Bind the inference commit onto the accepted transaction, per the lowered
delivery spec (unit details, write scopes, done-when proofs, and open
questions live there — this goal is the hosts-side tracking and ledger
authority):

1. **C1** `TokenCommit` binding vocabulary: token id + sequence position + KV
   generation + visible output advance together only on a mesh-wide commit;
   N=1 is the one-rank case; partial advance refuses; `host-coordinator`
   gains no inference vocabulary (training-caller gate).
2. **F1** (radix) token/sequence commit wire fixtures over the FMIR session
   surface; malformed variants fail closed.
3. **S1** logits/sampling ownership: token selected only from complete
   assembled per-rank logits; greedy argmax default; Gradus sampling consumed
   read-only.
4. **F2** abort/retry-fail binding: transaction abort/failure → D1/D4 fail
   (pre-dispatch unchanged / post-dispatch poison); retry disabled;
   deterministic-replay NOT ATTEMPTED row.
5. **C2** mesh-wide commit wiring + `device-execute` seam extension
   (invoke + `TokenCommit` receipt); N=1 same wiring, one rank.
6. **X1** token-commit fault suite: cancel/timeout/device-loss before commit
   leave no visible output past the last committed token/position.
7. **P1** (co-tracked with radix) N=1 parity on existing single-device
   fixtures.
8. **P2** 8:1 token-commit mechanics proof with honest receipts; 8:8
   NOT ATTEMPTED (RunPod hardware gate).
9. **C3** (radix-owned) closeout: exit-gate evidence table + CAMPAIGN
   status + this goal's ledger (the `md3i-closeout.md` doc lives in the
   radix campaign dir).

### Non-goals

- Editing `host-coordinator` (transaction mechanics stay MD3/MD3H; MD-A16).
- Editing Gradus / Norma / Examples / Inferentia (semantics consumed).
- Serving/HTTP/batching (Inferentia); KV paging (MD4B); MoE placement
  (MD4C); capacity sharding (MD4A); physical 8:8 / 8:2 rows.
- Deterministic-replay implementation beyond the named-gate record.
- Weakening any pinned oracle, tolerance, or NOT ATTEMPTED row.

## Units (lowered; see md3i-delivery.md §P3 for full unit tables)

| Unit | Scope | Depends on |
| --- | --- | --- |
| MD3I-C1 | `TokenCommit` binding vocabulary (new `composite_host/token_commit.rs`) | — |
| MD3I-F1 (radix) | token/sequence commit wire fixtures (fmir test surface) | — |
| MD3I-S1 | logits/sampling ownership binding (new `composite_host/logits_ownership.rs`) | MD3I-C1 |
| MD3I-F2 | abort/retry-fail binding (new `composite_host/abort_binding.rs`) | MD3I-C1 |
| MD3I-C2 | mesh-wide commit wiring + seam extension (`device_execute.rs` hot zone) | MD3I-S1, MD3I-F2, MD3I-F1 + **MD3I-WIN** window |
| MD3I-X1 | token-commit fault suite (tests, disjoint from C2 files) | MD3I-C2, MD3I-F2 |
| MD3I-P1 | N=1 parity on existing single-device fixtures (co-tracked with radix) | MD3I-C2 |
| MD3I-P2 | 8:1 token-commit mechanics proof + honest receipts + NOT ATTEMPTED 8:8 | MD3I-C2, MD3I-F1, MD3I-X1 |
| MD3I-C3 | closeout: exit-gate table, CAMPAIGN status, this ledger (radix-owned closeout doc) | all units |

Pre-dependencies on radix units are named by ID (`MD3I-F1`); routing/ordering
is the Mind's. The spine is C1 → S1 ‖ F2 → C2 → X1 ‖ P1 → P2 → C3.
C2 is a **named mega-row** with one writer on the `macos-arm64` hot files —
it dispatches alone behind the MD3I-WIN gate, never in parallel. X1 (fault
suite) and P1 (N=1 parity) run **after** C2, in parallel with each other on
disjoint files; P2 follows X1; C3 closes out all nine units (this ledger's
C3 row is ticked by the radix closeout).

## Validation

- Per unit: `cargo test -p faber-host-macos-arm64` (+ `-p host-coordinator`
  unchanged-green where the seam touches the transaction); `git diff --check`.
- P1/P2 machine evidence: burgus (Metal) + pharos (CUDA) runs, receipts in
  the unit reports / `md3i-8on1-evidence.md`.
- Closeout: `cargo test --workspace` in hosts; radix
  `./scripta/check-factory-goal-status --json --fail-on error` (0 findings).

## Ledger

| Unit | Status | Seat | Receipt | Notes |
| --- | --- | --- | --- | --- |
| MD3I-C1 | planned | — | — | — |
| MD3I-F1 | planned | — | — | — |
| MD3I-S1 | planned | — | — | — |
| MD3I-F2 | planned | — | — | — |
| MD3I-C2 | planned | — | — | — |
| MD3I-X1 | planned | — | — | — |
| MD3I-P1 | planned | — | — | — |
| MD3I-P2 | planned | — | — | — |
| MD3I-C3 | planned | — | — | closeout edits this ledger + CAMPAIGN status; radix `md3i-closeout.md` is the primary closeout doc owner |

## Open questions

Owned by the delivery spec (`md3i-delivery.md` OQ-1..OQ-5): seam shape for
invoke/commit; deterministic-replay proof gate; logits/sampling ownership for
the first mesh; N=1 commit route; token loop home.

---

*Planning artifact only. No product code was written. Hands implement from
the delivery spec; Mind files units and owns the MD3I-WIN window.*
