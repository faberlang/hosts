# GOAL: device-session-byte-surface — dtype-tagged bytes both directions; packed weights reach every backend

**Status**: planned — pre-implementation; drafted 2026-08-21 from campaign evidence F3.2; not lowered
**Created**: 2026-08-21
**Campaign:** `emission-lane-parity` (radix: [`docs/factory/emission-lane-parity/CAMPAIGN.md`](../../../../radix/docs/factory/emission-lane-parity/CAMPAIGN.md))
**Source:** operator architecture-audit session 2026-08-21 (campaign evidence F3.2, parent-verified)
**Repos:** `hosts` (primary: `macos-arm64/src/device_host.rs`, `metal_host.rs`, `cuda_host.rs`, `device_execute.rs`)
**Related:** hosts [`cuda-host-linux-home`](../cuda-host-linux-home/goal.md) · hosts [`device-capability-generalization`](../device-capability-generalization/goal.md) (dtype unit rides this surface) · radix [`placement-debt-audit`](../../../../radix/docs/factory/placement-debt-audit/goal.md) F2 dtype contract (coordinate, no duplicate) · radix [`cuda-rung-device-parity`](../../../../radix/docs/factory/cuda-rung-device-parity/goal.md)

---

## Invariant

The backend-neutral `DeviceSession` surface carries dtype-tagged byte
buffers in both directions: packed (non-f32) weights upload and read back on
every admitted backend, zero-copy/mmap retention is a declared per-backend
capability rather than a type asymmetry, and no transfer path silently
reinterprets bytes as f32.

## Problem

Campaign evidence F3.2 (parent-verified). The neutral surface is f32-only
and Metal carries the byte path as private extras:

| Gap | Evidence |
| --- | --- |
| Shared trait exposes only `copy_in_f32` / `readback_f32` | `hosts/macos-arm64/src/device_host.rs:380, 395` |
| Packed bytes and no-copy mmap are Metal-only methods (`copy_in_packed_bytes` admits 1–3-byte tails; `retain_mapped_file` binds mmap-backed MTLBuffers) | `metal_host.rs:542-566`, `:531-535` |
| The CUDA byte path requires 4-byte multiples and reinterprets bytes as f32 | `device_host.rs:705-717` |
| `device-execute --weights/--weight-map` retains the GGUF mmap only when the runtime is Metal (`if let Some(DeviceRuntime::Metal(..))`) | `device_execute.rs:919-921` |

Net: a discrete CUDA GPU — the RunPod A100/H100 class this workspace keeps
accounts for — has **no way to receive Q8_0/Q4_K packed weight regions**,
and the host transfer dtype vocabulary (`DeviceDataType`:
f32/f64/i32/i64/u8) cannot even name f16/bf16 that `DtypeSurface`
(`host-coordinator/src/discovery.rs:121-137`) says devices execute.

## Proposal

1. **Neutral byte surface**: `DeviceSession` gains
   `copy_in_bytes(handle, &[u8], DeviceDataType)` and
   `readback_bytes(handle, DeviceDataType) -> Vec<u8>`; the f32 methods
   become convenience wrappers. CUDA implements via `cuMemcpyHtoD`/DtoH on
   raw bytes (no alignment reinterpretation); Metal's packed path moves onto
   the neutral method with its 1–3-byte tail admission kept as a declared
   Metal capability.
2. **Mmap as capability**: `retain_mapped_file` stays Metal-only but is
   declared through a per-backend capability flag the executor consults,
   so `device_execute`'s weight upload path becomes backend-neutral: every
   backend gets bytes; Metal may additionally retain the mapping.
3. **Dtype vocabulary**: `DeviceDataType` gains `F16`/`BF16` slots wired to
   the placement-ABI dtype discriminant (coordinate with
   `placement-debt-audit` F2 — one dtype contract, not two).
4. **Proof**: fake-driver unit tests for the CUDA byte path; a real-device
   packed upload proof rides the rung harness (radix ELP-06 pod) or CAP-02
   — recorded, not gated, here.

### Non-goals

- No in-kernel dequant (EXEC-02 packed kernels own that; this goal only
  moves bytes).
- No pinned-host memory (`cuHostRegister`) — recorded as a follow-up perf
  lever for device-executor.
- No HTTP/remote transfer surface (RunPod transport stays the trials
  harness's job).

## Units (lowering sketch — refine via `$delivery`)

| Unit | Scope | Depends on | Hand evidence |
| --- | --- | --- | --- |
| 1 | Trait: `copy_in_bytes`/`readback_bytes` + f32 wrappers + Metal re-expression (packed-tail capability flag) | — | none |
| 2 | CUDA byte H2D/D2H (no 4-byte reinterpretation) + FakeCudaDriver tests incl. odd-length packed rows | 1 | none |
| 3 | `DeviceDataType` F16/BF16 + placement-dtype discriminant coordination (one contract) | 1 | none |
| 4 | Backend-neutral `device_execute` weight upload (mmap retention behind the capability flag) + tests | 1, 2 | none |

## Validation

- `cargo test --workspace` in hosts.
- `./scripta/cuda-tier-f-proof` on pharos (RTX 5070 Ti) stays green — the
  f32 tier must not regress while the byte tier lands.

## Ledger

| Unit | Status | Seat | Receipt | Notes |
| --- | --- | --- | --- | --- |
| 1 | pending | — | — | trait + metal |
| 2 | pending | — | — | cuda bytes |
| 3 | pending | — | — | dtype vocab |
| 4 | pending | — | — | neutral upload |

## Open questions

1. **Wrapper direction**: keep `copy_in_f32` as a trait method or move it to
   an extension helper? *Default: trait method delegating to bytes — least
   caller churn.*
2. **Tail admission**: neutral surface allows odd tails universally, or
   Metal-only via capability? *Default: capability — CUDA rejects misaligned
   tails explicitly rather than silently padding.*
3. **Byte-swap/endian**: any lane needing non-native endian? *Default: no —
   device byte order matches host on both admitted backends; assert, don't
   convert.*
