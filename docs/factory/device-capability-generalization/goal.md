# GOAL: device-capability-generalization — the capability model describes and gates every backend

**Status**: planned — pre-implementation; drafted 2026-08-21 from campaign evidence F3.3; dtype unit rides device-session-byte-surface
**Created**: 2026-08-21
**Campaign:** `emission-lane-parity` (radix: [`docs/factory/emission-lane-parity/CAMPAIGN.md`](../../../../radix/docs/factory/emission-lane-parity/CAMPAIGN.md))
**Source:** operator architecture-audit session 2026-08-21 (campaign evidence F3.3)
**Repos:** `hosts` (primary: `crates/host-coordinator/src/{discovery,policy}.rs`, `macos-arm64/src/{metal_host,cuda_host,device_descriptor}.rs`)
**Related:** hosts [`device-session-byte-surface`](../device-session-byte-surface/goal.md) (F16/BF16 transfer slots) · hosts [`cuda-host-linux-home`](../cuda-host-linux-home/goal.md) · radix [`device-route-backend-parity`](../../../../radix/docs/factory/device-route-backend-parity/goal.md) (consumes arch facts from here)

---

## Invariant

`host-coordinator`'s device capability model can describe and policy-gate
**every** admitted backend: launch-resource limits (max threads per
block/threadgroup, shared/threadgroup-memory bytes, collective/warp width,
unified vs discrete memory) are first-class generic fields populated per
backend from live device queries, and CUDA-only identity facts
(compute capability, SM count) remain what they are — identity, not the
whole model.

## Problem

Campaign evidence F3.3:

| Gap | Evidence |
| --- | --- |
| `DeviceCapabilities { compute_capability, sm_count, dtype_surface }` — the first two are CUDA-only concepts with no Metal meaning | `hosts/crates/host-coordinator/src/discovery.rs:103-109` |
| The policy compatibility gate checks exactly those CUDA facts, so it cannot evaluate a Metal device at all | `policy.rs:65-68` |
| No field anywhere for what a launcher actually needs per backend: max threads per block/threadgroup, shared-memory bytes, warp/simd width, unified-memory flag; Metal's one enforced limit (`max_total_threads_per_threadgroup`) lives privately in `metal_host.rs` and is invisible to planning | `metal_host.rs:1693-1706` |
| The host transfer dtype vocabulary (`DeviceDataType`: F32/F64/I32/I64/U8) cannot name f16/bf16 that `DtypeSurface` says devices execute | `device_descriptor.rs:63-74` vs `discovery.rs:121-137` |

The bias here runs in both directions: a Metal device cannot be policy
evaluated, and a CUDA device's launch resources cannot be planned against —
the model is too CUDA-shaped to gate Metal and too thin to launch CUDA.

## Proposal

1. **Generic launch-resource fields** on `DeviceCapabilities`:
   `max_threads_per_workgroup`, `workgroup_shared_memory_bytes`,
   `collective_width` (warp/simdgroup), `unified_memory: bool` — populated
   per backend from live queries (Metal: `MTLDevice` limits incl. the
   threadgroup ceiling currently private in `metal_host.rs`; CUDA:
   `cuDeviceGetAttribute` set: max threads per block, shared mem per block
   opt-in, warp size, integrated flag).
2. **Policy gate on generic fields**: compatibility checks consume the
   generic fields so both backends evaluate; `compute_capability`/`sm_count`
   stay as CUDA identity facts consumed only where arch matters (e.g. radix
   PTX negotiation via ELP-04).
3. **Dtype vocabulary closure**: `DeviceDataType` F16/BF16 slots land with
   the byte surface (sibling goal unit 3) and `DtypeSurface`↔transfer
   surface consistency gets a test — the model can no longer claim an
   executable dtype the transfer surface cannot name.
4. **Planning wiring**: `bound_plan`/partition consume the limits
   fail-closed (a plan exceeding a device's threadgroup/shared limits
   rejects with a named error rather than launching).

### Non-goals

- No new backends (AMD/MI300X stays the trials probe it is today).
- No scheduler/partitioning redesign — only fail-closed limit checks.
- Observability asymmetry (CUDA timing counters reporting zeros) is recorded
  here as a known gap and left to device-executor's measurement work.

## Units (lowering sketch — refine via `$delivery`)

| Unit | Scope | Depends on | Hand evidence |
| --- | --- | --- | --- |
| 1 | Generic fields + per-backend population (Metal from MTLDevice incl. the private threadgroup limit; CUDA from cuDeviceGetAttribute) + tests | — | none |
| 2 | Policy gate on generic fields (both backends evaluable) + a Metal-device gate test | 1 | none |
| 3 | DtypeSurface↔DeviceDataType consistency test (F16/BF16 slots with the byte-surface goal) | byte-surface unit 3 | none |
| 4 | bound_plan fail-closed limit checks + tests | 1 | none |

## Validation

- `cargo test -p host-coordinator -p faber-host-macos-arm64` in hosts.
- `cargo test --workspace` at closeout.

## Ledger

| Unit | Status | Seat | Receipt | Notes |
| --- | --- | --- | --- | --- |
| 1 | pending | — | — | fields |
| 2 | pending | — | — | policy gate |
| 3 | pending | — | — | dtype consistency |
| 4 | pending | — | — | plan checks |

## Open questions

1. **Field naming**: `workgroup`-spelled (Metal) vs `block`-spelled (CUDA)
   vs neutral (`max_threads_per_dispatch_group`)? *Default: neutral
   `workgroup`-family spellings already used across radix MIR — one
   vocabulary, not two.*
2. **CUDA shared-memory opt-in tiers** (static/dynamic carveout): model the
   minimum or the max opt-in? *Default: minimum guaranteed + max opt-in as
   two fields — planners need both.*
3. **Populate at discovery or at session open?** *Default: discovery, from
   the same live queries the identity facts use.*
