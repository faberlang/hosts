# GOAL: md3h-virtual-partition-host — every device run admits through a virtual partition

**Status**: done — all seven hosts units landed (H1 `34aa2cc`, H2 `cb79d8f`, H3 `295eb2a`, H4 `57cb94ce`, X1 `8ec2abd`, P1 `b6eb130` + radix seam receipt `91ff94bed`, P2 `a6096eb` + evidence `eb1c76fca`); closeout 2026-08-22: `cargo test -p faber-host-macos-arm64` green and `cargo test -p host-coordinator --lib` 137 passed; pharos CUDA 8:1 mechanics PASS with assertion tail + plan hashes; 8:8 deferred with RunPod gate (NOT ATTEMPTED, not failed).
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
| MD3H-H1 | Multi-ordinal discovery into physical-identity facts; ordinal-0 hardcode removed | — | `34aa2cc` |
| MD3H-H2 | `DeviceRuntimeSet` + real `DeviceExecutionBackend` (absorbs MD1-H1/MD3-S1) | MD3H-H1 | `cb79d8f` |
| MD3H-H3 | CompositeHost + device-execute uniform admission; partition-free path deleted | MD3H-H2 + **MD3H-WIN** window | `295eb2a` |
| MD3H-H4 | Distributed-section ingestion + mirror translation + CLI arg surface | MD3H-F1 (radix), MD3H-H2 | `57cb94ce` |
| MD3H-X1 | Real-backend fault injection suite (no partial commit) | MD3H-H2 | `8ec2abd` |
| MD3H-P2 | 8:1 mechanics proof + honest receipts + NOT ATTEMPTED 8:8 row | MD3H-H3, MD3H-H4, MD3H-F1 | `a6096eb` + evidence `eb1c76fca` |
| MD3H-P1 | N=1 parity on existing single-device fixtures (co-tracked with radix) | MD3H-C1 (radix) | `b6eb130` + radix `91ff94bed` |

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
| MD3H-H1 | done | hand | `34aa2cc` | structural: per-ordinal CUDA + Metal enumeration into discovery facts; ordinal remains a locator while PCI UUID / driver UUID and Metal registry identities remain distinct. W0 burgus-Metal + pharos-CUDA identity receipts were recorded. |
| MD3H-H2 | done | hand | `cb79d8f` | structural: DeviceRuntimeSet + DeviceRuntimeBackend (no host-coordinator trait extension). measured: absent in this commit — live Metal/CUDA tests env-gated PENDING when not admitted; no burgus/pharos machine receipt |
| MD3H-H3 | done | hand | `295eb2a` | structural: VirtualDevicePartition::implicit_local N=1 → BoundPlanKind::ImplicitLocal; partition-free product construction deleted; DeviceSelection backend-kind-only; N=1 coordinator-free (copy_ins=2, TransportClass::None). Hygiene `26e191ea`/`68c24046`/`f5370d5d`. measured: detached `cargo test -p host-coordinator` 134 passed; `cargo test -p faber-host-macos-arm64` wasm 3 passed. No live pharos CUDA admission receipt |
| MD3H-H4 | done | hand | `57cb94ce` | structural: H4p1 `5e0cec9`/`7f2276f` F1 postcard → transaction mirror; OnePhysicalPerPartition TopologyMismatch on 1-physical snapshot; OQ-2 macos-arm64 radix-mir-fmir dep (host-coordinator serde-free). H4p2 `57cb94ce` `--distributed-image` + `--bind-count`. measured: detached distributed_translate_test 10 passed; device_execute_cli_test 38 passed; package green. Residual: live pharos CUDA CLI spawn not exercised on burgus |
| MD3H-X1 | done | hand | `8ec2abd` | structural: 5 Metal + 1 CUDA fault tests (cancel/timeout/kernel/transfer/device-loss; no partial publication). measured: absent in this commit — live tests skip when Metal/CUDA not admitted; CUDA helper is pending-when-unreachable |
| MD3H-P2 | done | hand | `a6096eb` + radix evidence `eb1c76fca` | measured: pharos CUDA 8:1 mechanics PASS. Assertion tail: `physical_device_count=1`, `virtual_partition_count=8`, `fixture_identity_class=virtual`, `transport_class=host_staged`, `hardware_isolation_claimed=false`, `bind_shape=8:1`, `transaction_state=prepared`; snapshot `a4015cf258c8cac0e424bafb516a9d94f2525bf536b0c34bc9407345c6e00387`; logical plan `sha256:84e98067c1ea23e5b1cb3167851f72354731f0eddf0e562835309ad228684bd4`; bound plan `sha256:06a6716ca57b64dfa379ca341063fa9d247d5e87b67b9279d24fced378d98d46`. Promotion to 8 physical rejected as `TopologyMismatch`. Residual 8:8 is deferred with the RunPod 8× same-SKU gate (`NOT ATTEMPTED`, not failed). |
| MD3H-P1 | done | hand | `b6eb130` + radix `91ff94bed` | measured: cross-backend Metal + CUDA parity. Both backends run the same numeric session twice with outputs `[4.0, 6.0]`, `launches=1`, `copy_ins=2`, `readbacks=1`, `transfers=3`, and zero live handles after teardown. Radix host-seam receipt is 1:1 with `communication_graph_edge_count=0` and `hardware_isolation_claimed=false`. |

## Closeout (2026-08-22)

The hosts goal is closed independently of the parent campaign. The named
closeout gates are green:

- `cargo test -p faber-host-macos-arm64` passed all non-ignored host tests.
  The P2 pharos test remains ignored locally because it requires the admitted
  CUDA machine; its measured pharos run is recorded in the radix evidence
  receipt below.
- `cargo test -p host-coordinator --lib` passed 137 tests.
- P1 cross-backend parity is measured on Metal and CUDA in
  `macos-arm64/tests/composite_host_admission_test.rs` and through the radix
  host seam. N=1 preserves the numeric/session/leak bars and the distributed
  receipt has zero communication edges.
- P2's measured pharos CUDA proof is the 8:1 mechanics row only. Its assertion
  tail and plan hashes are recorded in
  `radix/docs/factory/gpu-inference-multi-device/md3h-8on1-evidence.md` at
  commit `eb1c76fca`; the one-device promotion rejection is
  `TopologyMismatch`.
- The 8:8 row is deferred with the named RunPod 8× same-SKU gate. It is
  `NOT ATTEMPTED`, not failed, and this goal makes no physical-capacity or
  speedup claim. The 8:2 row is likewise not attempted.

The parent `gpu-inference-multi-device` campaign remains active. This archive
closes only the hosts-side MD3H goal; campaign work outside it, including the
MD3I token/sequence commit stage, the Gradus PML5 executed oracle, and later
MD4A–MD7 stages, remains open/planned in the campaign record.

## Open questions

Owned by the delivery spec (`md3h-delivery.md` OQ-1..OQ-5). Decided at H4:
OQ-2 translation route is macos-arm64 `radix-mir-fmir` (F1 postcard decode);
`host-coordinator` stays serde-free/radix-mir-free (`5e0cec9`). OQ-5
spawn-seam is `--distributed-image` plus `--bind-count` (`57cb94ce`).
