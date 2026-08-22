# GOAL: cuda-host-linux-home — the CUDA host runtime is a peer product surface, not a macOS tenant

**Status**: active — lowered 2026-08-21 ([`delivery.md`](delivery.md) CLH-1..5; goal-check READY; priority — CLH-2 unlocks ELP-04; awaiting Mind dispatch)
**Created**: 2026-08-21
**Campaign:** `emission-lane-parity` (radix: [`docs/factory/emission-lane-parity/CAMPAIGN.md`](../../../../radix/docs/factory/emission-lane-parity/CAMPAIGN.md))
**Source:** operator architecture-audit session 2026-08-21 (campaign evidence F3.1, F3.4)
**Repos:** `hosts` (primary: `macos-arm64/src/{cuda_host,cuda_launch_adapter}.rs`, `manifest.rs`, `Cargo.toml`)
**Related:** hosts [`device-session-byte-surface`](../device-session-byte-surface/goal.md) · hosts [`device-capability-generalization`](../device-capability-generalization/goal.md) · radix [`device-route-backend-parity`](../../../../radix/docs/factory/device-route-backend-parity/goal.md) (its spawn table resolves to this home) · [`docs/factory/audits/metal-m4-api-parity-closure-9116571.md`](../audits/metal-m4-api-parity-closure-9116571.md) (the parity this goal must preserve)

---

## Invariant

The CUDA host runtime builds and runs on Linux without the macOS product:
it lives in its own hosts crate that `faber-host-macos-arm64` consumes as a
dependency, the exported capability manifest names a per-backend host, and
the launch path reuses one loaded PTX module per session instead of one per
launch — with the M4 driver-parity law (identical Metal/CUDA method sets)
preserved throughout.

## Problem

Campaign evidence F3.1, F3.4:

| Gap | Evidence |
| --- | --- |
| All CUDA host code (`cuda_host.rs`, `cuda_launch_adapter.rs`, proofs, tests) lives inside `faber-host-macos-arm64`, whose own identity is the macOS product; the Metal binding is `cfg(target_os = "macos")` so the crate compiles on Linux "for CUDA proof runs" — CUDA is structurally a tenant | `hosts/macos-arm64/Cargo.toml:2, 24-30` |
| The exported `CapabilityManifest.host` is hardcoded `"macos-arm64"` — a Linux CUDA execution would report itself as a macOS host | `manifest.rs:19` |
| The adapter loads the PTX module once **per launch** and teardown defers unload; every launch is a full-device sync on the null stream — multi-kernel chains pay N module loads + N context syncs | `cuda_launch_adapter.rs:21-22`; `cuda_host.rs:299-311, 705-723, 754-760` |
| Real-driver leak counters untracked (module unload deferred) | `device_registry.rs:91-99` |

There is no Linux/CUDA host product for the radix product layer to spawn
(radix `host_spawn.rs:754-770` builds `faber-host-macos-arm64`
unconditionally — resolved by sibling goal device-route-backend-parity).

## Proposal

1. **Extract** the CUDA driver binding, session, launch adapter, and their
   fakes into a new hosts crate (Q1: `crates/host-cuda`, mirroring the
   provider-crate pattern, with `macos-arm64` depending on it; pure move +
   re-export, no behavior change). The kernel library's CUDA halves move or
   parameterize likewise.
2. **Per-backend manifest host**: `CapabilityManifest.host` derives from the
   admitted backend(s) rather than the string `"macos-arm64"`; composite
   admission tests cover a Linux/CUDA-shaped manifest.
3. **Module cache**: load the PTX module once per session (keyed by module
   bytes/hash), retire at teardown with the deferred-unload debt closed and
   leak counters tracked; per-launch reload disappears.
4. **Linux build proof**: extend `scripta/cuda-tier-f-proof` (or a sibling
   script) with a Linux build/run leg on pharos — crate builds and the
   driver lane works without any macOS target present.
5. **Stream model stays null-stream** (Q3): pipelining/streams are
   device-executor perf territory; this goal records, not redesigns.

### Non-goals

- No stream/event pipelining (device-executor owns it).
- No remote/HTTP execution surface (RunPod stays the trials harness).
- No KV/resident-session CUDA enablement — that is
  gpu-production-readiness EXEC-03; this goal changes structure only.

## Units (lowering sketch — refine via `$delivery`)

| Unit | Scope | Depends on | Hand evidence |
| --- | --- | --- | --- |
| 1 | Crate extraction (pure move + re-export; macos-arm64 green unchanged) | — | none |
| 2 | Per-backend manifest host + admission tests (incl. Linux/CUDA shape) | 1 | none |
| 3 | Module cache + teardown/leak-counter closure + adapter tests | 1 | none |
| 4 | Linux build/run proof leg in scripta (pharos) | 1, 3 | none |

Lowered to [`delivery.md`](delivery.md) as CLH-1..5: the extraction unit splits into CLH-1 (neutral `host-device-core` crate — forced by the trait/descriptor/registry types both backends share) + CLH-2 (`host-cuda` crate); unit 3 re-scoped (production `ProgramSession` already loads once per session and hash-shares modules; remaining debt is the real-driver deferred `cuModuleUnload`, zero real `DriverCounters`, and the adapter's per-launch reload — delivery §0).

## Validation

- `cargo test --workspace` in hosts (macos-arm64 suite unchanged-green).
- `./scripta/cuda-tier-f-proof` on pharos green through the new crate path.
- Linux leg: `cargo check`/`test` for the new crate on pharos (Linux).
- M4 parity law re-asserted: the driver method sets stay identical.

## Ledger

| Unit | Status | Seat | Receipt | Notes |
| --- | --- | --- | --- | --- |
| CLH-1 | pending | — | — | host-device-core extraction |
| CLH-2 | done | prior seat (ended) | 2804c0d | host-cuda extraction landed; cargo check/test -p host-cuda and cargo test --workspace green; aliases clean (reconciled 2026-08-22) |
| CLH-3 | pending | — | — | manifest host |
| CLH-4 | pending | — | — | module lifecycle debt |
| CLH-5 | pending | — | — | linux proof |

## Open questions

1. **Crate shape**: `crates/host-cuda` (library consumed by both products)
   vs `hosts/cuda-host` (standalone product dir)? *Default:
   `crates/host-cuda` — a Linux product binary wrapper can come later with
   the radix spawn table; the library is the unblocking artifact.*
2. **What moves**: whole adapter+driver+session, or driver only with session
   staying composite? *Default: whole CUDA surface moves; composite host
   keeps its trait wiring via the dependency.*
3. **Streams**: keep null-stream with full sync per launch boundary even
   after the module cache? *Default: yes, record as device-executor input —
   correctness first, this goal is structural.*
