# DELIVERY: cuda-host-linux-home — CUDA host runtime as a peer product surface

**Status**: lowered 2026-08-21 — ready for Mind to file Hands (priority: unblocks ELP-04)
**Goal:** [`goal.md`](goal.md) (goal-check verdict: **READY** — record below)
**Campaign:** `emission-lane-parity` ELP-08 / findings F3.1, F3.4 (radix [`CAMPAIGN.md`](../../../../radix/docs/factory/emission-lane-parity/CAMPAIGN.md))
**Repos:** `hosts` only (all units); no radix writes
**Live-seat fences:** H2 (md3h W1) owns `macos-arm64/src/lib.rs` (M in tree) + new `device_runtime_set.rs` / `transaction_backend.rs` (+ tests); pedantic seat owns clippy debt in `crates/host-coordinator` + `crates/solum`. See §3 Foreign lanes.

---

## 0. Goal check (gate for this lowering)

**Verdict: READY.** Evaluator: planner. Consumer: delivery.

| Goal claim | Live check | Result |
| --- | --- | --- |
| All CUDA host code lives inside `faber-host-macos-arm64`; Metal binding `cfg(target_os = "macos")` so the crate compiles on Linux "for CUDA proof runs" | `macos-arm64/Cargo.toml:29-31` (cfg-gated metal dep), CUDA surface = `src/cuda_host.rs` (1677 ln), `src/cuda_launch_adapter.rs` (1105 ln), tests `cuda_host_test.rs` / `cuda_launch_adapter_test.rs` / `cuda_host_proof.rs` | live, exact |
| `CapabilityManifest.host` hardcoded `"macos-arm64"` | `manifest.rs:19` (`from_parts`), sole production caller `kernel/host.rs:95` | live, exact |
| Adapter loads the PTX module once **per launch**; multi-kernel chains pay N loads | **partially stale**: the per-launch load is real but lives only on the public `launch_descriptor` entry (`cuda_launch_adapter.rs:946`, doc `:18` "loaded once per launch") which has **no production callers** (test-only + `lib.rs:31` re-export). The production path `ProgramSession` (`composite_host/session.rs:622-628`) already loads once per session **and** hash-shares sibling modules; teardown releases in order (`:725`) | live in substance; unit re-scoped (CLH-4) |
| Teardown defers unload; real-driver leak counters untracked | `cuda_host.rs:1100` "Module teardown (cuModuleUnload) is deferred for the one-shot proof"; `device_registry.rs` `DriverCounters` doc: real drivers report the four lifecycle counters as zero | live, exact |
| "The kernel library's CUDA halves move or parameterize likewise" | `macos-arm64/src/kernel/**` has **zero** cuda/nvvm/ptx references (grep clean; kernel library is the Metal-side quantized GEMV + frame machinery) | **stale — no CUDA halves exist**; extraction surface is cuda_host + adapter + tests only (goal OQ2 default "whole CUDA surface moves" confirmed) |

Dependency fork the goal's sketch did not surface (settles OQ1/OQ2 default,
not new architecture): `cuda_host.rs`/`cuda_launch_adapter.rs` consume
backend-neutral shared types that Metal also consumes — `device_descriptor`
(`DeviceDataType` + `E_DEVICE_*` errors), `device_registry`
(`HandleRegistry`/`DriverCounters`/`FakeFailureStage`, used by `metal_host.rs:32`),
and the device-kernel `HostError`/`HostResult` + `frame_data`
(`kernel/error.rs`, `kernel/frame_data.rs`). The `DeviceSession` trait
(`device_host.rs:392`) is implemented by both sessions. A Linux-buildable
`host-cuda` cannot depend on the macOS product, so the neutral slice must sit
**above** both: one neutral crate first (CLH-1), then the CUDA crate (CLH-2).
This is the minimal correct DAG forced by the goal's own invariant, not an
invented layer.

## 1. Interpreted theme

F3.1/F3.4: CUDA is structurally a tenant of the macOS product crate. The
manifest lies about the host on Linux, the real driver leaks its module
(cuModuleUnload deferred, counters zero), and there is no Linux/CUDA host
crate for the radix product layer to spawn (ELP-04's spawn table resolves
here).

## 2. Normalized spec

A backend-neutral `crates/host-device-core` owns the device session trait,
descriptor dtype/error vocabulary, handle registry + driver counters, and the
device-kernel error/frame-data types. `crates/host-cuda` owns the CUDA driver
binding, session, and launch adapter (+ their tests and the env-gated proof),
depends only on the core crate, and builds on Linux with no macOS target
present. `faber-host-macos-arm64` depends on both and keeps every existing
`crate::` path compiling via re-export aliases (zero caller edits; H2's
in-flight `device_runtime_set.rs` keeps its `crate::cuda_host::` imports
unmodified). The manifest host derives from the admitted backend(s), not the
string `"macos-arm64"`. The real driver closes its module lifecycle
(cuModuleUnload at teardown, lifecycle counters tracked), and the adapter's
per-launch reload retires behind a session-keyed module cache mirroring
`session.rs:624-626`. Null-stream launch boundaries stay (goal OQ3 default —
recorded as device-executor input, not redesigned here). M4 driver-parity law
(identical Metal/CUDA method sets) preserved throughout.

Delivery-level non-goals (inherited): no stream/event pipelining; no
remote/HTTP execution; no KV/resident-session CUDA enablement (EXEC-03); no
behavior change on any green path.

## 3. Repo-aware baseline

| Surface | Today | Note |
| --- | --- | --- |
| CUDA surface | `macos-arm64/src/cuda_host.rs`, `cuda_launch_adapter.rs`; tests `cuda_host_test.rs`, `cuda_launch_adapter_test.rs`, `cuda_host_proof.rs` (env-gated) | the move set |
| Neutral deps of the move set | `device_descriptor.rs` (1826 ln), `device_registry.rs` (138 ln), `kernel/error.rs` (69 ln), `kernel/frame_data.rs` (138 ln), `DeviceSession` trait inside `device_host.rs` (828 ln, trait at `:392`, `DeviceRuntime` enum at `:108`) | shared with Metal; must land above both products |
| Manifest | `manifest.rs` (52 ln) hardcoded host; caller `kernel/host.rs:95` | CLH-3 surface |
| Module lifecycle | production: load-once + hash-share + ordered teardown (`composite_host/session.rs:622-628, 725`); adapter `launch_descriptor`: per-launch load (`:946`); real driver: cuModuleUnload deferred (`cuda_host.rs:1100`), `DriverCounters` real = zeros (`device_registry.rs`) | CLH-4 surface |
| Proof script | `scripta/cuda-tier-f-proof` (exit codes 0/1/2/3; anti-false-green contract; runs on pharos) | CLH-5 extends with a Linux crate leg |
| Workspace shape | root `Cargo.toml` members list; library crates use explicit path deps (AGENTS.md invariant 4); `crates/` is the shared-library home (invariant 3) | `crates/host-device-core` + `crates/host-cuda` fit standing law |
| Foreign lanes | **H2 (live)**: `lib.rs` (M), `device_runtime_set.rs`, `transaction_backend.rs`, `device_runtime_set_test.rs`, `transaction_backend_test.rs` (??). **Pedantic seat (open task a3a0a0a0)**: clippy debt `crates/host-coordinator` (46+63) + `crates/solum`. | no unit writes those files; `lib.rs` edits are alias-only (see CLH-1/2 boundary rules) |

## 4. Stage Graph — Hand units

```text
CLH-1 (host-device-core) ──> CLH-2 (host-cuda) ──> CLH-5 (Linux proof leg)
        └──────────────────────────> CLH-4 (module lifecycle debt) ──┘
CLH-3 (manifest host; parallel from t=0)
```

### CLH-1 — Neutral device-core crate `crates/host-device-core` (pure move + path aliases)

| Field | Value |
| --- | --- |
| outcome | New workspace crate `crates/host-device-core` owning, as a pure move: `device_descriptor.rs`, `device_registry.rs`, `kernel/error.rs`, `kernel/frame_data.rs`, and the `DeviceSession` trait (+ its dtype-tagged byte methods) extracted from `device_host.rs` — the `DeviceRuntime` enum and its concrete dispatch arms stay in `macos-arm64`. `macos-arm64` depends on the core crate and preserves every existing path via re-export aliases (`pub use host_device_core::device_descriptor;` etc., and `kernel/mod.rs` re-exporting `HostError`/`HostResult`/frame_data), so **no caller file changes**. Workspace green, zero behavior change. |
| write_scope | `hosts/Cargo.toml` (members + one path dep), new `hosts/crates/host-device-core/**`, `hosts/macos-arm64/Cargo.toml` (path dep), `hosts/macos-arm64/src/lib.rs` (alias re-exports **only**), `hosts/macos-arm64/src/kernel/mod.rs` (re-exports), `hosts/macos-arm64/src/device_host.rs` (trait extraction residue only) |
| done_when | `cargo check -p host-device-core` green (first-failing oracle: fails today — crate absent); `cargo test --workspace` in hosts green with zero test deletions; `git diff` shows no edits to any file other than the move set + aliases; `crate::device_descriptor::`/`crate::device_registry::`/`crate::kernel::{HostError, frame_data}` paths still resolve for every existing caller |
| depends_on | none |
| sanity | `cargo test -p host-device-core && cargo test -p faber-host-macos-arm64 --lib` |
| non_goals | moving any CUDA or Metal implementation file (CLH-2); manifest changes (CLH-3); behavior changes of any kind; touching `composite_host/` beyond import-path compatibility |
| boundary rule | `macos-arm64/src/lib.rs` is H2-live (M in tree): add alias lines only; **never** edit H2's `device_runtime_set` / `transaction_backend` declarations. If alias hunks collide with H2's uncommitted lines, STOP and report to Mind for sequencing — do not rebase foreign work. |
| risk | medium — trait extraction from an 828-ln file while H2 holds `lib.rs`; path-alias strategy keeps the blast radius at compile-time |
| integrable | yes |

### CLH-2 — CUDA crate `crates/host-cuda` (pure move + re-export aliases)

| Field | Value |
| --- | --- |
| outcome | New workspace crate `crates/host-cuda` owning, as a pure move: `cuda_host.rs`, `cuda_launch_adapter.rs`, and tests `cuda_host_test.rs`, `cuda_launch_adapter_test.rs`, `cuda_host_proof.rs`. Depends on `host-device-core` (and `libloading` only). `macos-arm64` depends on it; `pub use host_cuda as cuda_host; pub use host_cuda as cuda_launch_adapter;` module aliases keep `crate::cuda_host::…` callers — including H2's `device_runtime_set.rs` — compiling unmodified. Workspace green, zero behavior change; M4 driver-parity method sets untouched. |
| write_scope | `hosts/Cargo.toml` (members + path dep), new `hosts/crates/host-cuda/**` (moved sources + tests), `hosts/macos-arm64/Cargo.toml` (path dep), `hosts/macos-arm64/src/lib.rs` (alias re-exports only) |
| done_when | `cargo check -p host-cuda` green (first-failing oracle: fails today — crate absent); `cargo test -p host-cuda` green (moved suites, zero deletions); `cargo test --workspace` green; `grep -rn 'mod cuda_host' macos-arm64/src/` clean while `crate::cuda_host::` paths still resolve |
| depends_on | CLH-1 |
| sanity | `cargo test -p host-cuda` |
| non_goals | manifest host (CLH-3); module-lifecycle behavior (CLH-4); kernel-library moves (nothing CUDA lives there — verified); stream model changes |
| boundary rule | Same `lib.rs` alias-only rule as CLH-1; H2's files import `crate::cuda_host` and must keep compiling **without edits**. |
| risk | low — pure move over a settled core seam |
| integrable | yes |

### CLH-3 — Manifest host derives from admitted backend(s)

| Field | Value |
| --- | --- |
| outcome | `CapabilityManifest.host` stops being the literal `"macos-arm64"`: `from_parts` (`manifest.rs:19`) takes the host identity from the admitted backend surface at its call site (`kernel/host.rs:95`), naming a per-backend host (macOS product stays `"macos-arm64"`; a CUDA-admitted host reports a CUDA-shaped host string). Composite admission tests cover a Linux/CUDA-shaped manifest. |
| write_scope | `hosts/macos-arm64/src/manifest.rs`, `hosts/macos-arm64/src/kernel/host.rs` (call site), matching `*_test.rs` / `tests/` manifest tests |
| done_when | New test: a CUDA-admitted manifest's `host` is not `"macos-arm64"` (first-failing oracle: fails today — hardcoded); existing macOS manifest tests keep their identity; spelling of the CUDA host string documented at the struct |
| depends_on | none (parallel-safe: files disjoint from CLH-1/2 move set except `kernel/host.rs`, which CLH-1 does not move) |
| sanity | targeted manifest tests under `cargo test -p faber-host-macos-arm64` |
| non_goals | manifest_version bumps; provider/syscall manifest changes; ELP-04 spawn-table work (radix) |
| risk | low |
| integrable | yes |

### CLH-4 — Real-driver module lifecycle closure + adapter session cache

| Field | Value |
| --- | --- |
| outcome | (a) `SystemCudaDriver` closes the deferred-unload debt: teardown calls `cuModuleUnload` (today deferred at `cuda_host.rs:1100`) and real-driver `DriverCounters` track the four lifecycle counters instead of reporting zeros (fake-driver counters already prove the policy). (b) The `launch_descriptor` adapter path stops paying one module load per launch: `CudaHostSession` carries a module cache keyed by PTX bytes/hash (mirroring `composite_host/session.rs:624-626` share logic), released at session teardown. Production `ProgramSession` behavior unchanged. |
| write_scope | `hosts/crates/host-cuda/src/cuda_host.rs`, `hosts/crates/host-cuda/src/cuda_launch_adapter.rs` (post-CLH-2 locations), matching moved test files |
| done_when | Fake-driver oracle (first-failing today): two `launch_descriptor` calls with identical PTX in one session report `module_loads == 1` and one release at teardown (today `module_loads == 2`); a code-level assertion/grep that the real driver release path calls the unload symbol; real-driver counters wired (fake-suite proof; on-device evidence rides the existing S2-8-style real-device gate, not this unit) |
| depends_on | CLH-2 |
| sanity | `cargo test -p host-cuda` |
| non_goals | stream/event pipelining (OQ3 recorded default: null-stream stays); ProgramSession share-policy changes; pharos on-device proof (CLH-5 lane) |
| risk | medium — touches the real driver binding; fake-driver suite is the harness |
| integrable | yes |

### CLH-5 — Linux build/run proof leg (pharos)

| Field | Value |
| --- | --- |
| outcome | `scripta/cuda-tier-f-proof` (or a sibling `scripta/cuda-host-linux-proof`) gains a Linux leg that, on pharos: `cargo check`/`cargo test -p host-cuda` with no macOS target present, and the existing tier-F pipeline runs green **through the new crate path** (proof binary build step repointed at `host-cuda` where applicable). The script's exit-code and anti-false-green contract is preserved verbatim. |
| write_scope | `hosts/scripta/*` (the one script), no Rust sources |
| done_when | Linux leg present and green on pharos (first-failing oracle: today the script has no such leg / no `host-cuda` crate to build); exit-code contract unchanged; macOS leg behavior unchanged |
| depends_on | CLH-2 (CLH-4 for the full run leg) |
| sanity | `bash -n scripta/cuda-tier-f-proof` + a dry-run of the leg on pharos per its own gating |
| non_goals | CI workflow changes; RunPod/trials surfaces; on-device perf claims |
| risk | low — script-only, environment-gated |
| integrable | yes |

## 5. Implementation Work (Mind pointers)

Each Hand task is a pointer: goal path + unit id + write_scope + done_when
from §4. **Priority order (ELP-08 unlocks ELP-04):** CLH-1 immediately;
CLH-3 in parallel from t=0 (disjoint files); CLH-2 on CLH-1's receipt —
**CLH-2 landing is the ELP-04 unlock point** (the spawn-table target crate
exists then; Mind may lower ELP-04 without waiting for CLH-4/5); CLH-4 on
CLH-2's receipt; CLH-5 last. Mind owns the goal ledger update as units land.

## 6. Checkpoints And Gates

**Batching:** five Hands, no merge gate — every unit independently integrable
and green at its own commit. CLH-1→CLH-2→{CLH-4, CLH-5} is the only serial
spine; CLH-3 is parallel throughout.

**Lane-owned gates (named once, never copied onto child Hands):**

| Lane | Owns |
| --- | --- |
| lint | `cargo fmt` per touched crate; pedantic clippy debt burn-down on `host-coordinator`/`solum` stays on the pedantic seat's task (a3a0a0a0), sequenced by Mind **after** DCG semantic units land if it collides — never folded into CLH units |
| test | `cargo test --workspace` (cwd `hosts`) at goal closeout |
| merge | workspace-green check per landed unit; pharos leg evidence attached at closeout |

## 7. Open questions for Mind

1. **CLH-3 host spelling** for a CUDA-admitted host (e.g. `cuda-linux` vs
   driver-derived string) — Hand default: a documented per-backend constant;
   flag if ELP-04's spawn table needs a specific spelling.
2. **CLH-4 real-device counter evidence** rides the standing real-device gate
   (S2-8-style) rather than this goal's fake-driver proof — confirm Mind
   accepts fake-suite + code-assertion as this unit's done-when.
3. **CLH-1↔H2 collision** on `lib.rs` alias lines: if H2 is still uncommitted
   when CLH-1 spawns, Mind sequences (H2 first, or explicit rebase ruling).
