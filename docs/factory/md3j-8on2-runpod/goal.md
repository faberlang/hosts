# GOAL: md3j-8on2-runpod — 8-on-2 RunPod hardware rung (M=2 multi-aspect proof)

**Status**: planned — lowered 2026-08-22 ([`md3j-delivery.md`](../../../../radix/docs/factory/gpu-inference-multi-device/md3j-delivery.md)); ready for unit tasking (entry gate met: MD3H archived `7014785`; operator directive `6ac4d2a8` authorizes RunPod 2× same-SKU spend). Runs **parallel to MD3I** — does not displace it.
**Created**: 2026-08-22
**Campaign:** `gpu-inference-multi-device` (radix: [`docs/factory/gpu-inference-multi-device/CAMPAIGN.md`](../../../../radix/docs/factory/gpu-inference-multi-device/CAMPAIGN.md))
**Source:** operator directive mail `6ac4d2a8` (2026-08-22 — 8-on-2 rung is next in the multi-device test sequence, after MD3H 8:1 PASS, before any 8:8 attempt); lowered spec [`md3j-delivery.md`](../../../../radix/docs/factory/gpu-inference-multi-device/md3j-delivery.md); 8:1 receipts [`md3h-8on1-evidence.md`](../../../../radix/docs/factory/gpu-inference-multi-device/md3h-8on1-evidence.md)
**Repos:** `hosts` (primary: `macos-arm64` distributed bind/budget seams + proof surface; `host-coordinator` **read-only**), `radix` (fmir fixture surface only); evidence receipts under `~/work/ianzepp/trials/runpod-gpu-verification/`
**Related:** archived [`md3h-virtual-partition-host`](../../archived/md3h-virtual-partition-host/goal.md) (the 8:1 rung this diverges from) · [`md3i-inference-token-commit`](../md3i-inference-token-commit/goal.md) (parallel 8:1 path; MD3J does not displace) · need `702a7c5c` (MD3J-WIN/MD3I-WIN gates must enumerate concrete handles — reconciled)

---

## Invariant

One 8-rank plan binds 8 `VirtualDevicePartition`s 4+4 across exactly 2
same-SKU physical devices on one machine, the process running **on** that
machine; the MULTI aspect is proven — M=2 `DeviceRuntimeSet`, deterministic
rank→partition→physical bind divergence, host-staged cross-device transfer,
barrier/commit coordination through `ExecutionTransaction`, and
`BudgetExceeded` fail-closed on declared budgets — and nothing else is
claimed: no capacity, no speedup, no `AllocationFailure` coverage, no 8:8.

## Problem

8:2 is an owned orphan: named in the campaign bind tables, ruled NOT
ATTEMPTED by md3h-delivery OQ-4, and parked outside MD3I by its FC12. The
operator directive (`6ac4d2a8`) makes it the next hardware rung and
authorizes the RunPod pod. Verified gaps close it: the bind seam rejects
`--bind-count 2` for 8 partitions outright (`Unsupported`,
`distributed_translate.rs:99-128`), and the admission path sets
`SafePhysicalLimit` equal to the declared budget, so the
`AdmissionError::BudgetExceeded` fail-closed behavior the taxonomy promises
is unreachable from the declared-budget path and over-budget rejects report
as a generic binding failure (`distributed_translate.rs:478-488`). M=2
`DeviceRuntimeSet` exists structurally (MD3H-H2) with no live execution
receipt.

## Proposal

Per the lowered delivery spec (unit details, write scopes, done-when
proofs, and the frozen receipt field set live there — this goal is the
hosts-side tracking and ledger authority):

1. **F1** (radix) 8-rank fixture with explicit 8 GiB `PartitionBudgetBytes`
   per partition (small software budget regardless of card size) + one
   over-budget variant; wire-legal both.
2. **B1** split bind policy: 8 partitions bind 4+4 across exactly the
   snapshot's 2-physical membership, deterministic, placement constraints
   enforced, `TopologyMismatch` on count mismatch; red-first.
3. **B2** policy-derived `SafePhysicalLimit` (named headroom policy, never
   the raw total, never the declared budget) with the `BudgetExceeded`
   admission class surfaced as its own bind error; red-first.
4. **P1** the pod proof run, process on the pod (operator-held RunPod
   access via mail — gate MD3J-RP): 8:2 receipts per the frozen field set,
   the 8:1 comparability row (same image, identical logical plan hash,
   different bound plan hash), and the on-pod `BudgetExceeded` rejection.
5. **C1** closeout: evidence doc, campaign status, this ledger.

### Non-goals

- Capacity or speedup claims (not MD4A); `AllocationFailure` under real
  pressure; 8:8 (stays deferred behind the RunPod 8× same-SKU gate).
- MD3I surfaces; P2P/peer admission; cross-host or remote-device mapping;
  mixed SKU; mixed vendor.
- `host-coordinator` edits (read-only) — the partition taxonomy already
  suffices.
- Weakening any pinned oracle, tolerance, or NOT ATTEMPTED row.

## Units (lowered; see md3j-delivery.md §P3 for full unit tables)

| Unit | Scope | Depends on |
| --- | --- | --- |
| MD3J-F1 (radix) | 8 GiB budget-declared 8-rank fixture + over-budget variant | — |
| MD3J-B1 | split bind policy 8:2 (`distributed_translate.rs`) | — |
| MD3J-B2 | policy-derived safe limit + `BudgetExceeded` class (same file) | MD3J-B1 |
| MD3J-P1 | RunPod 2× same-SKU pod proof + honest receipts + comparability row | MD3J-F1, MD3J-B1, MD3J-B2 + **MD3J-RP** |
| MD3J-C1 | closeout: evidence doc, campaign status, this ledger | all units |

Spine: F1 ‖ (B1 → B2) → P1 → C1. **MD3J-WIN** applies only if a unit must
touch the `macos-arm64` hot files (`device_execute.rs`,
`composite_host.rs`, `host_spawn.rs`, `cuda_host.rs`, `metal_host.rs`) —
default plan touches none — and the gate must enumerate the concrete
in-flight handles claiming those files (MD3I-C2, KV-B6, KV-D5, E6-D,
device-executor M8, `cuda-host-linux-home`), never a process-only
"no sibling mid-flight" check (need `702a7c5c`).

## Validation

- Per unit: `cargo test -p faber-host-macos-arm64` (focused test targets per
  the spec); `git diff --check`.
- P1 machine evidence: the RunPod pod run itself (receipts under
  `~/work/ianzepp/trials/runpod-gpu-verification/`; lane default `dc-a100`).
- Closeout: `cargo test --workspace` in hosts; radix
  `./scripta/check-factory-goal-status --json --fail-on error` (0 findings).

## Ledger

| Unit | Status | Seat | Receipt | Notes |
| --- | --- | --- | --- | --- |
| MD3J-F1 | landed | hand | radix 5bf4ce6d8 + hosts c192dd4 | 8GiB fixtures + over-budget variant; graph bytes preserved; 186 fmir tests green |
| MD3J-B1 | landed | hand | hosts 6614514 (red) + 8593996 (green) | split policy 8:2; 21/21; deterministic 4+4; fail-closed both directions |
| MD3J-B2 | landed | hand | hosts 5c5d756 (red) + 216e1c5 (green) | oq2_default_headroom_policy floor(api_total x 0.9); BindError::BudgetExceeded class + byte facts; 25/25 + 137+1; host-coordinator touch = one ADDITIVE BindError variant (disclosed) |
| MD3J-P1 | landed | hand | radix 29798174d + hosts 00581504a + evidence 071641c (P1b clean_pass; attempts 1-2 partial: wrong image, then honest fixture-scale gap) | ALL frozen rows verified on-pod: 8:2 bind 4+4, 8:1 comparability, 40GiB admits, 79GiB BudgetExceeded w/ byte facts; ~$9 total spend |
| MD3J-C1 | planned | — | — | closeout edits this ledger + CAMPAIGN MD3J status |

## Open questions

Owned by the delivery spec (`md3j-delivery.md` OQ-1..OQ-5): pod build route;
safe-limit headroom constant; split rule; P2P posture (default host-staged,
peer NOT ATTEMPTED); lane choice (default `dc-a100`).

---

*Planning artifact only. No product code was written. Hands implement from
the delivery spec; Mind files units and owns the MD3J-RP and MD3J-WIN
gates. Honest exclusions (operator directive `6ac4d2a8`) travel verbatim in
every receipt: no capacity or speedup claim; `AllocationFailure` under real
pressure NOT tested; 8:8 stays deferred.*
