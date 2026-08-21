# GOAL: md3h-virtual-partition-host — every device run admits through a virtual partition

**Status**: active — W0 dispatched 2026-08-22 (R1 radix-mir `7f7ed3c6`, F1 fmir fixtures `2af2dc3a`, H1 discovery `d22be982`); spine H1→H2→H3→C1→P1→C2; MD3H-WIN hot-file window gate (Mind) before H3
**Created**: 2026-08-21
**Campaign:** `gpu-inference-multi-device` (radix: [`docs/factory/gpu-inference-multi-device/CAMPAIGN.md`](../../../../radix/docs/factory/gpu-inference-multi-device/CAMPAIGN.md))
**Source:** CAMPAIGN.md §MD3H + the 2026-08-21 operator amendment (uniform N=1; eight-rank bind); lowered spec [`md3h-delivery.md`](../../../../radix/docs/factory/gpu-inference-multi-device/md3h-delivery.md)
**Repos:** `hosts` (primary: `crates/host-coordinator` additive seams, `macos-arm64` product host)
**Related:** hosts [`cuda-host-linux-home`](../cuda-host-linux-home/goal.md) (cuda_host.rs adjacency) · hosts [`device-capability-generalization`](../device-capability-generalization/goal.md) (discovery/capability fields) · radix `kv-cache-decode` KV-B6/KV-D5 + `device-executor` M5-U3/U4 (hot-file serialization — see MD3H-WIN)

---

## Invariant

Every device execution in the product host — N=1 and N=8 alike — admits
through a `VirtualDevicePartition` bound to exactly one `PhysicalDeviceId`
and executes a bound plan; one-device execution is the same architecture
with an empty communication graph, never a parallel partition-free path.

## Problem

The campaign's verified ground truth (2026-08-21): `CompositeHost` opens one
`DeviceRuntime` (`composite_host.rs:153,251`); CUDA hardcodes ordinal 0
(`cuda_host.rs:568`); `ProgramSession` bypasses virtual partitions;
`host-coordinator` has the accepted library (`partition.rs:326`
`implicit_local`, `bound_plan.rs:263` `BoundPlanKind`, the
`ExecutionTransaction` + `DeviceExecutionBackend` trait) with only
`FakeExecutionBackend` (`execution_transaction/backend.rs:118-158`) and no
`DeviceRuntimeSet`. Without the uniform host path, N=1 and N=8 drift and
eight-rank work never executes for real.

## Proposal

Wire the real product host onto the accepted library, per the lowered
delivery spec (unit details, write scopes, done-when proofs, and the frozen
receipt field set live there — this goal is the hosts-side tracking and
ledger authority):

1. **H1** multi-ordinal physical discovery (CUDA count/identity per ordinal,
   Metal enumeration) into `host-coordinator` discovery facts.
2. **H2** `DeviceRuntimeSet` (M physical sessions by composition) + the real
   `DeviceExecutionBackend` over them — absorbs the queued MD1-H1/MD3-S1
   residuals.
3. **H3** `CompositeHost` + `device-execute` uniform admission: discover →
   admit (`implicit_local` for N=1) → bind → execute; partition-free product
   construction deleted (clean break); `DeviceSelection` stays backend kind.
4. **H4** distributed FMIR ingestion + translation into the transaction
   mirror (canonical bytes; `host-coordinator` stays serde-free and
   radix-mir-free); minimal `device-execute` arg-surface extension.
5. **X1** real-backend fault suite: cancel/timeout/kernel/transfer/device-loss
   retire with no partial publication.
6. **P2** 8:1 mechanics proof with honest receipts (`physical_device_count=1`,
   `virtual_partition_count=8`, `hardware_isolation_claimed=false`); the
   promote-8:1-to-8-physical bind rejects; 8:8 recorded NOT ATTEMPTED.
7. **P1** (co-tracked with radix MD3H-P1) N=1 parity on existing fixtures.

### Non-goals

- MD3I token/sequence/KV commit semantics; the Gradus rank API.
- Cross-host/remote GPUs (RunPod as local `PhysicalDeviceId`s), Metal+CUDA
  one job, cross-vendor meshes, peer/P2P admission, collective libraries.
- 8:2 proof rows (types legal; NOT ATTEMPTED at closeout).
- `host-coordinator` semantic changes — additive seams only; existing
  library tests stay green unchanged.

## Units (lowered; see md3h-delivery.md §P3 for full unit tables)

| Unit | Scope | Depends on | Hand evidence |
| --- | --- | --- | --- |
| MD3H-H1 | Multi-ordinal discovery into physical-identity facts; ordinal-0 hardcode removed | — | none |
| MD3H-H2 | `DeviceRuntimeSet` + real `DeviceExecutionBackend` (absorbs MD1-H1/MD3-S1) | MD3H-H1 | none |
| MD3H-H3 | CompositeHost + device-execute uniform admission; partition-free path deleted | MD3H-H2 + **MD3H-WIN** window | none |
| MD3H-H4 | Distributed-section ingestion + mirror translation + CLI arg surface | MD3H-F1 (radix), MD3H-H2 | none |
| MD3H-X1 | Real-backend fault injection suite (no partial commit) | MD3H-H2 | none |
| MD3H-P2 | 8:1 mechanics proof + honest receipts + NOT ATTEMPTED 8:8 row | MD3H-H3, MD3H-H4, MD3H-F1 | none |
| MD3H-P1 | N=1 parity on existing single-device fixtures (co-tracked with radix) | MD3H-C1 (radix) | none |

Pre-dependencies on radix units are named by ID (`MD3H-F1`, `MD3H-C1`,
`MD3H-R1`); routing/ordering is the Mind's. The spine is
H1 → H2 → H3 (one writer on the `macos-arm64` hot files); H4/X1 run parallel
to H3 on disjoint files once H2 lands.

## Validation

- Per unit: `cargo test -p faber-host-macos-arm64` (+ `-p host-coordinator`
  where touched); `git diff --check`.
- H1/P1/P2 machine evidence: burgus (Metal) + pharos (CUDA) runs, receipts
  in the unit reports / `md3h-8on1-evidence.md`.
- Closeout: `cargo test --workspace` in hosts; radix
  `./scripta/check-factory-goal-status --json --fail-on error` (0 findings).

## Ledger

| Unit | Status | Seat | Receipt | Notes |
| --- | --- | --- | --- | --- |
| MD3H-H1 | pending | — | — | discovery |
| MD3H-H2 | pending | — | — | runtime set + real backend |
| MD3H-H3 | pending | — | — | uniform admission (MD3H-WIN gated) |
| MD3H-H4 | pending | — | — | ingestion + translation |
| MD3H-X1 | pending | — | — | fault suite |
| MD3H-P2 | pending | — | — | 8:1 proof |
| MD3H-P1 | pending | — | — | parity (with radix) |

## Open questions

Owned by the delivery spec (`md3h-delivery.md` OQ-1..OQ-5); the two that
touch this goal directly: translation dependency route (OQ-2 — default
hosts-side in `macos-arm64`, `host-coordinator` stays dependency-free) and
the spawn-seam extension shape (OQ-5 — default minimal `device-execute`
args). Decided at the named unit audits; never silently.
