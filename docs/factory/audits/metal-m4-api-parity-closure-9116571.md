# Audit: Metal M4 — API parity closure review (9116571)

```yaml
kind: audit
audit_version: 1
auditor: auditor-2
assignment: 87a7dd9f
repository: hosts (faberlang/hosts)
base: d9db9d2d5a71013114ddaf6f801caf4c02cd8609  # cuda-host C5 SystemCudaDriver binding (pre-M4 hosts tip)
head: 9116571309896ba6bc1b1eea05df151567431c6a  # metal-host M4 C5 API parity closure (== hosts main tip / HEAD)
scope:
  review_mode: implementation (landed single commit, real-device verification)
  requirements:
    - "1. MetalDriver trait now has 10 methods matching CudaDriver 1:1 (including launch_kernel)"
    - "2. MetalHostSession has launch_kernel + public sync()"
    - "3. try_open moved to session (matching CUDA); SystemMetalDriver now private (matching CUDA)"
    - "4. FakeMetalDriver mirrors the CUDA fake"
    - "5. Real-device add_one still passes via the routed legacy path"
    - "6. metal-spike and metal-faber-spike both pass"
    - "7. No touching cuda_host.rs"
  changed:   # files touched by commit 9116571 only
    - macos-arm64/src/metal_host.rs            (+278/-72, trait+session+system+fake)
    - macos-arm64/tests/metal_host_test.rs     (+119/-3, try_open / generalized launch / guards / real-device scale_two)
  reviewed:
    - macos-arm64/src/metal_host.rs (full diff + trait block + impl MetalDriver for SystemMetalDriver + FakeMetalDriver + MetalHostSession)
    - macos-arm64/tests/metal_host_test.rs (full diff)
    - macos-arm64/src/cuda_host.rs (parity reference: CudaDriver trait 96-126, CudaHostSession 136-267, FakeCudaDriver 800-935, SystemCudaDriver 415)
    - macos-arm64/src/lib.rs (re-export surface unchanged; SystemMetalDriver correctly NOT re-exported)
    - docs/factory/audits/metal-m3-api-parity-c5-vs-m2.md (M3 findings PAR-1/2/3 being closed)
  excluded:
    - macos-arm64/src/cuda_host.rs changes in the frozen range — those belong to d2c3660
      (cuda-host(C5): relabel E_CUDA_UNSUPPORTED as reserved), a separate C5 commit, NOT the
      M4 review target. Commit 9116571 itself touches only metal_host.rs + metal_host_test.rs
      (verified via `git show --name-only 9116571`).
    - Cargo.lock/Cargo.toml — unchanged by 9116571 (metal dep landed in M2/652b07b).
risk: medium - real GPU execution path (unsafe FFI + Metal command encoding), but confined to a
  macOS-only host crate not in the faber release payload; no CI. Real-device proof is the
  strongest safe evidence available and was run.
verdict: clean_pass
verdict_basis: >
  Frozen single-commit range (d9db9d2 → 9116571, HEAD==head, clean tree). Every assigned changed
  surface accounted for. All 7 task items SATISFIED with live evidence. The M3 audit findings
  PAR-1 (trait launch_kernel), PAR-2 (session launch_kernel + sync + try_open), and PAR-3
  (SystemMetalDriver visibility) are demonstrably closed by direct source comparison against the
  CUDA parity reference. Required validation passed on the real Apple M5 Max device: metal-spike
  and metal-faber-spike both exit 0 with the expected add_one [0..15]→[1..16] mapping; metal_host_test
  9/9 passes including the real-device routed-legacy add_one proof and a real-device generalized
  session launch (scale_two over a buffer slice); cuda_host_test 4/4 baseline parity green; clippy
  --all-targets clean (only pre-existing transitive block v0.1.6 future-incompat warning); cargo
  check --workspace clean. No P0/P1/P2 findings. No blind spots that affect the verdict.
findings: []
req_status:
  - req: "1. MetalDriver trait 10 methods matching CudaDriver 1:1 (incl launch_kernel)"
    status: SATISFIED
    evidence: >
      metal_host.rs:70-100 declares exactly 10 trait methods in CUDA order: discover,
      create_context, load_module, alloc, copy_in, launch_elementwise_add_f32, launch_kernel
      (NEW, 89-96), sync, copy_out, free. cuda_host.rs:96-126 declares the identical 10-method
      set with identical signature shapes (only the EnvReport return type differs by backend, as
      expected). launch_kernel signature matches byte-for-byte: (module: u64, entry: &[u8],
      buffers: &[u64], grid_x: u32, block_x: u32) -> HostResult<()>. Closes M3 PAR-1.
  - req: "2. MetalHostSession has launch_kernel + public sync()"
    status: SATISFIED
    evidence: >
      metal_host.rs:209-237 `pub fn launch_kernel(&mut self, module, entry: &str, buffers:
      &[MetalHandleId], grid_x, block_x)` — validates require_admitted, module_token, empty entry,
      resolves every buffer handle to a backend token, calls driver.launch_kernel + driver.sync.
      metal_host.rs:240-243 `pub fn sync(&mut self)` — require_admitted + driver.sync. Both mirror
      CudaHostSession::launch_kernel (cuda_host.rs:239-260) and CudaHostSession::sync (264-267)
      structurally. Closes M3 PAR-2(a)(b).
  - req: "3. try_open moved to session (matching CUDA); SystemMetalDriver now private"
    status: SATISFIED
    evidence: >
      metal_host.rs:113-127 `pub fn MetalHostSession::try_open()` constructs SystemMetalDriver
      internally, discovers, fail-closes on !admitted, creates context — structurally identical
      to CudaHostSession::try_open (cuda_host.rs:139-153). SystemMetalDriver is now `struct
      SystemMetalDriver` (metal_host.rs:554, no pub), matching SystemCudaDriver (cuda_host.rs:415,
      no pub). The old SystemMetalDriver::try_open() is gone. lib.rs (22-24) re-exports
      MetalHostSession but NOT SystemMetalDriver, symmetric with CUDA. Closes M3 PAR-2(c) and PAR-3.
  - req: "4. FakeMetalDriver mirrors the CUDA fake"
    status: SATISFIED
    evidence: >
      FakeMetalDriver::launch_kernel (metal_host.rs:498-511) checks `buffers.len() != 3` then
      calls simulate_elementwise_add(module, buffers[0..3]) — byte-for-byte mirror of
      FakeCudaDriver::launch_kernel (cuda_host.rs:900-916). The shared helper
      FakeMetalDriver::simulate_elementwise_add (metal_host.rs:393-443) is extracted identically
      to FakeCudaDriver::simulate_elementwise_add (cuda_host.rs:800+). The legacy
      launch_elementwise_add_f32 now delegates to simulate_elementwise_add (matches CUDA fake).
      Tested by fake_driver_sequences_generalized_launch_kernel (PASS) and
      session_fails_closed_on_guard_checks (PASS).
  - req: "5. Real-device add_one still passes via the routed legacy path"
    status: SATISFIED
    evidence: >
      system_driver_compiles_msl_launches_add_one_and_reads_back PASS (0.283s) on Apple M5 Max:
      loads ADD_ONE_MSL (`kernel void add_one`, bindings input@0, output@1, extent@2), calls
      session.launch_elementwise_add_f32(module, a, b, out), which now routes through
      SystemMetalDriver::launch_kernel with buffers [a, out, extent_token] bound at indices 0,1,2
      — matching the kernel's [[buffer(0/1/2)]] attributes. Readback equals [1.0..16.0]. The
      extent buffer is inserted before launch and removed after (no leak; result captured).
      OBS-1 kernel-shape difference (unary add_one vs binary addita) is preserved and accepted.
  - req: "6. metal-spike and metal-faber-spike both pass"
    status: SATISFIED
    evidence: >
      `./scripta/metal-spike` (from radix/) exit 0: "metal spike OK: Apple M5 Max add_one
      [0.0..15.0] -> [1.0..16.0]". `./scripta/metal-faber-spike` exit 0: "metal faber spike OK:
      Apple M5 Max add_one [0.0..15.0] -> [1.0..16.0]". These are the Swift/Radix environmental
      baselines corroborating the real Metal device and the U2 extent contract independently of
      the Rust host crate.
  - req: "7. No touching cuda_host.rs"
    status: SATISFIED
    evidence: >
      `git show --name-only --format="" 9116571` lists exactly two files: macos-arm64/src/metal_host.rs
      and macos-arm64/tests/metal_host_test.rs. cuda_host.rs IS modified within the frozen range
      (d9db9d2 → 9116571) but by a DIFFERENT commit, d2c3660 (cuda-host(C5): relabel E_CUDA_UNSUPPORTED
      as reserved), not by the M4 review target.
validation:
  - command: git rev-parse d9db9d2 && git rev-parse 9116571 && git rev-parse HEAD && git status --short
    result: pass
    note: base=d9db9d2..., head=9116571..., HEAD==head; clean tree, no foreign dirt on the metal surface
  - command: git show --name-only --format="" 9116571
    result: pass
    note: exactly 2 files (metal_host.rs, metal_host_test.rs); cuda_host.rs NOT in commit 9116571
  - command: ./scripta/metal-spike  (from radix/)
    result: pass
    note: exit 0; "metal spike OK: Apple M5 Max add_one [0.0..15.0] -> [1.0..16.0]"
  - command: ./scripta/metal-faber-spike  (from radix/)
    result: pass
    note: exit 0; "metal faber spike OK: Apple M5 Max add_one [0.0..15.0] -> [1.0..16.0]"
  - command: cargo nextest run -p faber-host-macos-arm64 --test metal_host_test
    result: pass
    note: 9/9 incl. system_driver_compiles_msl_launches_add_one_and_reads_back (routed legacy,
      0.283s), system_session_launch_kernel_dispatches_over_buffer_slice (generalized, 0.283s),
      try_open_opens_live_session_or_fails_closed (0.302s), fake_driver_sequences_generalized_launch_kernel,
      session_fails_closed_on_guard_checks (generalized guards)
  - command: cargo nextest run -p faber-host-macos-arm64 --test cuda_host_test --test metal_host_test
    result: pass
    note: 13/13; CUDA baseline 4/4 green (parity reference unchanged)
  - command: cargo clippy -p faber-host-macos-arm64 --tests --all-targets
    result: pass
    note: clean; only pre-existing transitive block v0.1.6 future-incompat warning (metal dep)
  - command: cargo check --workspace
    result: pass
    note: full hosts workspace typechecks; no broken callers from SystemMetalDriver visibility narrowing
adversarial_review_notes:
  - >
    Dispatch math (launch_kernel threadgroup clamping): threads_per_threadgroup = min(block_x,
    pipeline.max_total_threads_per_threadgroup()).max(1); thread_groups = (grid_x*block_x)
    .div_ceil(threads_per_threadgroup). In both regimes (max_threads >= block_x → exact grid_x
    dispatch; max_threads < block_x → ceil widens group count) total dispatched threads >=
    requested grid_x*block_x. Never under-dispatches. Legacy path sets block_x=256, grid_x=
    len.div_ceil(256); the add_one kernel guards `if (id >= extent) return;` so over-dispatch is
    safe. Verified on real device (test PASS). No regression vs pre-M4 dispatch.
  - >
    Extent buffer lifecycle (launch_elementwise_add_f32): extent_token inserted before
    launch_kernel, removed unconditionally after (result captured then returned). No leak on
    success or failure path.
  - >
    Fail-closed paths: launch_kernel validates module token, entry name vs stored module entry,
    grid_x/block_x non-zero, every buffer token resolved before touching the encoder. Session
    launch_kernel adds require_admitted, module_token, empty-entry, per-buffer handle validation.
    Tested by session_fails_closed_on_guard_checks (empty entry, non-buffer handle, stale handle
    all return E_INVALID_ARGS / E_METAL_INVALID_HANDLE). Honest guards, not mocked away.
  - >
    Entry-name binding contract: ELEMENTWISE_ADD_ENTRY=b"add_one" matches the emitted
    `kernel void add_one` (ADD_ONE_MSL, buffer(0)=input, buffer(1)=output, buffer(2)=extent).
    Legacy routing binds [a, out, extent_token] at indices 0,1,2 → matches kernel attributes.
    scale_two generalized launch binds [a, out] at 0,1 → matches its buffer(0/1) attributes.
    Both PASS on real device.
blind_spots: []
not_claimed:
  - Global repository correctness
  - CUDA C5 runtime correctness (separate audit 7c479bb2)
  - Performance or portability of either lane
  - Implementation of follow-on binary-add Metal kernel (OBS-1, accepted M2 scope; re-evaluate when emitted)
```

## Summary

Metal M4 API-parity closure review of single commit 9116571 on hosts main. **Verdict: clean_pass.**

All 7 task items are SATISFIED with live real-device evidence. The M3 audit findings PAR-1
(MetalDriver missing `launch_kernel`), PAR-2 (MetalHostSession missing `launch_kernel` /
`sync()` / `try_open()`), and PAR-3 (`SystemMetalDriver` visibility asymmetry) are demonstrably
closed by direct source comparison against the CUDA parity reference. `MetalDriver` and
`CudaDriver` now declare the identical 10-method set; `MetalHostSession` and `CudaHostSession`
expose the same public lifecycle surface; both system drivers are private with session-side
`try_open`. The legacy `launch_elementwise_add_f32` now routes through the single
`launch_kernel` encoder/commit site (one launch site per backend, as CUDA does), preserving the
accepted OBS-1 unary-add_one kernel shape and the U2 runtime-extent channel.

Real-device proof on Apple M5 Max: `metal-spike` and `metal-faber-spike` both exit 0 with the
expected add_one mapping; `metal_host_test` 9/9 passes including the routed-legacy real-device
add_one proof and a real-device generalized session launch (`scale_two` over a buffer slice);
CUDA baseline 4/4 green; clippy and full workspace check clean. Dispatch math, extent-buffer
lifecycle, and fail-closed guard paths reviewed adversarially with no defect found. No material
finding; no blind spots affecting the verdict.
