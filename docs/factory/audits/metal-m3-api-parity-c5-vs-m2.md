# Audit: Metal M3 — API parity review (cuda_host.rs post-C5 vs metal_host.rs post-M2)

```yaml
kind: audit
audit_version: 1
auditor: auditor-2
assignment: cb6e10f3
repository: hosts (faberlang/hosts)
base: 652b07b0fcd7f4dc43be3a5ece7b6385a3fa7fc3  # metal_host.rs post-M2 (unchanged at HEAD)
head: d9db9d2d5a71013114ddaf6f801caf4c02cd8609  # cuda_host.rs post-C5 (== hosts main tip / HEAD)
scope:
  review_mode: api-parity (interface comparison of two compute-host surfaces)
  requirements:
    - "1. MetalDriver trait methods match CudaDriver trait methods (incl. new launch_kernel from C5)"
    - "2. Session lifecycle methods match (MetalHostSession vs CudaHostSession)"
    - "3. Error codes mirror (E_METAL_* vs E_CUDA_*)"
    - "4. Handle model matches (MetalHandleId vs CudaHandleId)"
    - "5. Probe structure parallels"
    - "6. Fake driver seams match (FakeMetalDriver vs FakeCudaDriver)"
    - "7. Any divergence that would make the two compute-host surfaces inconsistent"
  compared:
    - macos-arm64/src/cuda_host.rs @ d9db9d2 (947 lines, post-C5)
    - macos-arm64/src/metal_host.rs @ 652b07b (678 lines, post-M2; git diff d9db9d2 empty → unchanged at HEAD)
    - macos-arm64/src/lib.rs (re-export surface)
  excluded:
    - Runtime correctness of either real driver (CUDA C5 audited separately under 7c479bb2;
      Metal M2 audited under dd665576 / auditor-1 clean_pass). This unit is interface parity only.
risk: low - static interface comparison of two frozen snapshots; no product mutation
verdict: residual
verdict_basis: >
  Parity gap is real, material, and precisely characterized. The Metal lane has NOT
  absorbed CUDA C5's generalized launch_kernel (trait + session), the session-level
  sync() barrier, or the session-level try_open() constructor; SystemCudaDriver is
  private while SystemMetalDriver is pub (a consequence of the try_open placement
  divergence). These make the two compute-host surfaces inconsistent (item 7 = YES).
  Non-blocking for Metal's current scope: M2's stated purpose (add_one real-device
  proof) is fully met — 10/10 host tests pass including the live Metal binding. The
  gap is a known, tracked residual: the C5 commit explicitly filed a cross-lane need
  ("MetalDriver mirrors CudaDriver 1:1"). No UNEXPECTED divergence was found: error
  codes (item 3), handle model (item 4), probe structure (item 5), and fake seams
  (item 6) all mirror correctly modulo the launch_kernel family. residual because
  there is a material parity finding requiring explicit disposition (route closure to
  a Metal follow-up step, e.g. M4); not block_ship because nothing currently depends
  on the missing Metal launch_kernel and M2's proof stands; not clean_pass because the
  two surfaces are demonstrably not 1:1 and item 7's answer is yes.
findings:
  - id: PAR-1
    severity: P2
    confidence: confirmed
    category: interfaces
    where: macos-arm64/src/metal_host.rs MetalDriver trait (lines 63-80) vs cuda_host.rs CudaDriver (lines 96-127)
    expected: MetalDriver mirrors CudaDriver method-for-method per the M1 1:1 mirroring contract and the C5 cross-lane need.
    actual: >
      MetalDriver declares 9 methods; CudaDriver declares 10. Metal is MISSING the
      generalized launch_kernel(&mut self, module: u64, entry: &[u8], buffers: &[u64],
      grid_x: u32, block_x: u32) -> HostResult<()> added to CudaDriver in C5
      (cuda_host.rs:115-122). The other 9 (discover, create_context, load_module,
      alloc, copy_in, launch_elementwise_add_f32, sync, copy_out, free) match 1:1 in
      name, order, and signature shape (only the EnvReport return type differs by
      backend, as expected).
    impact: >
      Backend-agnostic code parameterized over the driver/session trait cannot issue a
      generalized kernel launch on the Metal lane. The two compute-host surfaces are
      not interchangeable above the elementwise-add milestone.
    evidence: >
      grep "launch_kernel" across metal_host.rs + metal_host_test.rs → (none). CudaDriver
      trait method list (cuda_host.rs:97-125) has 10 entries; MetalDriver (metal_host.rs
      after line 63) has 9. C5 commit d9db9d2 message: "CudaDriver::launch_kernel(...) +
      CudaHostSession::launch_kernel ... cross-lane need filed to Metal lane (MetalDriver
      mirrors CudaDriver 1:1)".
    reproduction: static — compare trait blocks; no runtime needed.
    fix_direction: >
      Add MetalDriver::launch_kernel(module, entry, buffers, grid_x, block_x) +
      MetalHostSession::launch_kernel(module, entry: &str, buffers: &[MetalHandleId],
      grid_x, block_x) + FakeMetalDriver/SystemMetalDriver impls. Route the existing
      launch_elementwise_add_f32 through it (as CUDA now does) so there is one launch
      site per backend. Route via a Metal follow-up step (M4).
    suggested_owner: hand on Metal lane (route by Mind)
    done_when: >
      MetalDriver and CudaDriver declare the same method set; a backend-agnostic caller
      can launch a named entry over a buffer slice on either lane with identical session
      ergonomics.

  - id: PAR-2
    severity: P2
    confidence: confirmed
    category: interfaces
    where: macos-arm64/src/metal_host.rs MetalHostSession impl (lines 91-247) vs cuda_host.rs CudaHostSession (lines 131-342)
    expected: MetalHostSession exposes the same public lifecycle methods as CudaHostSession.
    actual: >
      MetalHostSession is MISSING three public methods that CudaHostSession gained/has:
      (a) pub fn launch_kernel(...) (cuda_host.rs:239) — see PAR-1;
      (b) pub fn sync(&mut self) -> HostResult<()> (cuda_host.rs:265) — explicit device
          sync barrier exposed at the session. Metal's session calls self.driver.sync()
          only internally inside launch_elementwise_add_f32 (metal_host.rs:171) but does
          NOT expose a public session sync();
      (c) pub fn try_open() -> HostResult<Self> (cuda_host.rs:139) — CudaHostSession
          constructs SystemCudaDriver internally and returns a session. MetalHostSession
          has NO try_open; the equivalent lives on SystemMetalDriver::try_open()
          (metal_host.rs:458) which returns MetalHostSession. Functionally present but
          structurally misplaced for parity.
    impact: >
      A caller targeting the session API surface finds try_open on different types
      (CudaHostSession vs SystemMetalDriver) and cannot sync explicitly on the Metal
      session. Inconsistent entry-point ergonomics across the two lanes.
    evidence: >
      "pub fn " in impl CudaHostSession = 12 entries (incl. try_open, launch_kernel,
      sync); impl MetalHostSession = 9 entries (no try_open, no launch_kernel, no sync).
      try_open grep: cuda_host.rs:139 (in impl CudaHostSession); metal_host.rs:458 (in
      impl SystemMetalDriver).
    reproduction: static — compare impl block pub fn lists.
    fix_direction: >
      (a) covered by PAR-1; (b) add MetalHostSession::sync() delegating to driver.sync()
      with require_admitted(); (c) either add MetalHostSession::try_open() that wraps
      SystemMetalDriver (matching CUDA's shape) or document the divergent placement as
      intentional if the Metal lane prefers the driver-side entry. Mind decides the
      target shape; CUDA's session-side try_open is the parity reference.
    suggested_owner: hand on Metal lane (route by Mind)
    done_when: >
      MetalHostSession and CudaHostSession expose the same public method set, or the
      try_open placement divergence is documented as an intentional, accepted asymmetry.

  - id: PAR-3
    severity: P2
    confidence: confirmed
    category: interfaces
    where: macos-arm64/src/metal_host.rs:448 (pub struct SystemMetalDriver) vs cuda_host.rs:415 (struct SystemCudaDriver, private)
    expected: Symmetric visibility of the real system driver type across lanes.
    actual: >
      SystemCudaDriver is a private struct (cuda_host.rs:415, no pub) — only reachable
      via CudaHostSession::try_open(). SystemMetalDriver is pub (metal_host.rs:448) and
      carries the lane's try_open; it is NOT re-exported from lib.rs but is accessible
      via the module path crate::metal_host::SystemMetalDriver.
    impact: >
      Asymmetric public type surface. Consequence of the PAR-2(c) try_open placement:
      Metal must publish the driver type to make its try_open callable, CUDA does not.
      Resolving PAR-2(c) resolves this.
    evidence: grep "struct SystemCudaDriver" → no pub; "struct SystemMetalDriver" → pub.
    reproduction: static.
    fix_direction: folds into PAR-2(c); make try_open placement symmetric and the system
      driver visibility follows.
    suggested_owner: hand on Metal lane (route by Mind)
    done_when: both system drivers have matching visibility given a symmetric try_open.

  - id: OBS-1
    severity: none (informational; already dispositioned in M2 audit dd665576 OBS-1)
    confidence: confirmed
    category: behavior
    where: launch_elementwise_add_f32 kernel shape — CUDA runs binary addita (a+b), Metal runs unary add_one (a+1)
    expected: trait method name launch_elementwise_add_f32 shared for parity.
    actual: >
      The two lanes run DIFFERENT kernels under the same trait method name: CUDA's
      addita is binary (out=a+b, all three buffers bound via launch_kernel); Metal's
      add_one is unary (input@0, output@1, extent@2; b validated fail-closed but not
      bound to the encoder). This is the documented M2 scope (add_one proof) and was
      consciously accepted in the M2 audit (OBS-1) — it becomes relevant when a true
      binary add kernel is emitted for Metal.
    impact: none for current milestones; recorded for continuity when PAR-1 lands a
      generalized Metal launch.
    evidence: metal_host.rs:566-572 (b validated, not bound); M2 audit OBS-1.
    reproduction: static.
    fix_direction: none now; re-evaluate when Metal gains a binary kernel.
    suggested_owner: n/a (accepted)
    done_when: n/a (informational).

parity_matrix:
  item_1_trait_methods: DIVERGE — Metal missing launch_kernel (PAR-1); other 9 match 1:1
  item_2_session_lifecycle: DIVERGE — Metal missing launch_kernel, sync(), try_open() (PAR-2)
  item_3_error_codes: MATCH — E_{METAL,CUDA}_{UNAVAILABLE,UNSUPPORTED,INVALID_HANDLE,DRIVER} mirror 1:1; helpers symmetric; neither re-exports E_*_DRIVER from lib.rs (symmetric). Tested by metal_host_test::error_codes_mirror_cuda_family (PASS).
  item_4_handle_model: MATCH — *HandleId(pub u64) identical derives; *HandleKind (Module | Buffer{len_bytes}); *Handle{kind, backend_token}. 1:1.
  item_5_probe_structure: PARALLEL — *EnvReport{admitted, device_signal: Option<String>, candidate_paths: Vec<String>, reason}; probe functions parallel (device signal + candidate scan + admitted derivation). Field names differ by backend (appropriate).
  item_6_fake_seams: MATCH modulo launch_kernel — same fields/ctor/method set except CUDA fake has launch_kernel + extracted simulate_elementwise_add helper; Metal fake has neither.
  item_7_inconsistency: YES — PAR-1/PAR-2/PAR-3 make the surfaces inconsistent; OBS-1 is an accepted kernel-shape difference.
validation:
  - command: git rev-parse 652b07b && git rev-parse d9db9d2 && git rev-parse HEAD
    result: pass
    note: base=652b07b0..., head/d9db9d2...; HEAD==d9db9d2; CUDA target == HEAD, Metal target == M2 (unchanged at HEAD)
  - command: git diff d9db9d2 -- macos-arm64/src/metal_host.rs
    result: pass
    note: empty → metal_host.rs identical at M2 and at HEAD (post-C5); confirms post-M2 snapshot reviewed
  - command: git status --short
    result: pass
    note: clean tree; no foreign dirt on the compared surface
  - command: grep -rn "launch_kernel" macos-arm64/src/metal_host.rs macos-arm64/tests/metal_host_test.rs
    result: pass
    note: no matches → Metal has zero launch_kernel surface (trait/session/fake/system/test)
  - command: cargo check -p faber-host-macos-arm64 --tests
    result: pass
    note: typechecks; only future-incompat warning on transitive block v0.1.6 (metal dep), not our code
  - command: cargo nextest run -p faber-host-macos-arm64 --test cuda_host_test --test metal_host_test
    result: pass
    note: 10/10 passed (0.198s). CUDA sequencing 4/4 (probe + fake; real proof env-gated, no CUDA hw).
      Metal 6/6 incl. system_driver_compiles_msl_launches_add_one_and_reads_back (live Apple GPU,
      0.198s) and error_codes_mirror_cuda_family. Baseline green on both lanes.
blind_spots:
  - >
    Runtime parity of a generalized launch is untested on Metal because the API does
    not exist there yet (PAR-1). This is the gap itself, not an evidence shortage in
    the review: the review is static interface parity, and the runtime M2 proof
    (add_one) was already audited clean_pass by auditor-1.
  - >
    The cross-lane need C5 claims to have filed ("MetalDriver mirrors CudaDriver 1:1")
    was not located in the project mailspace by keyword search; its absence from search
    does not contradict the gap — the gap is confirmed by direct source comparison
    either way. Mind may reconcile the need's handle if it wants the follow-up tracked
    there specifically.
not_claimed:
  - Global repository correctness
  - CUDA C5 runtime correctness (separate audit 7c479bb2)
  - Metal M2 runtime correctness (separate audit dd665576, clean_pass)
  - Performance or portability of either lane
  - Implementation of the missing Metal launch_kernel (review-only unit; no implement)
```

## Summary

Metal M3 API-parity review against the CUDA post-C5 surface. **Verdict: residual.**

The Metal lane has not absorbed CUDA C5's generalized `launch_kernel` (trait + session), the session-level `sync()` barrier, or the session-level `try_open()` constructor (PAR-1, PAR-2). `SystemMetalDriver` is `pub` while `SystemCudaDriver` is private, a consequence of the `try_open` placement divergence (PAR-3). Everything else mirrors 1:1: error codes (item 3), handle model (item 4), probe structure (item 5), fake seams (item 6). The kernel-shape difference under `launch_elementwise_add_f32` (binary `addita` vs unary `add_one`) is the accepted M2 scope, already dispositioned (OBS-1).

Non-blocking: M2's stated purpose (add_one real-device proof) is fully met — 10/10 host tests pass including the live Metal binding on this Apple GPU. The gap is a known, tracked residual; the C5 commit filed the cross-lane need. Disposition: route `launch_kernel` + `sync()` + `try_open()` parity closure to a Metal follow-up step (M4).
