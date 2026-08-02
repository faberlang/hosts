# Audit: Metal M2 SystemMetalDriver (hosts 778c0f1 → 652b07b)

```yaml
kind: audit
audit_version: 1
auditor: auditor-1
assignment: dd665576
repository: hosts (faberlang/hosts)
base: 778c0f1e76f8f431538900b9bab707912d11e80e
head: 652b07b0fcd7f4dc43be3a5ece7b6385a3fa7fc3  # hosts main tip
scope:
  requirements:
    - SystemMetalDriver implements MetalDriver trait fully (all 9 methods)
    - Runtime MSL compile via new_library_with_source (no offline metallib)
    - Launch add_one over 16 elements: dispatch_thread_groups + wait_until_completed
    - Readback [1.0..16.0] — real device execution, not faked
    - Runtime-extent binding populated (binding 2, u32 count)
    - Cargo.toml change minimal (metal crate dep only)
    - No touching cuda_host.rs or lib.rs re-exports
    - Env-gated test: skips without Metal device, hard-fails with broken stack
    - metal-spike and metal-faber-spike both pass
  changed:
    - Cargo.lock                                       (+100, metal transitive deps)
    - macos-arm64/Cargo.toml                           (+2, metal = "0.33")
    - macos-arm64/src/metal_host.rs                    (+274/-28, SystemMetalDriver)
    - macos-arm64/tests/metal_host_test.rs             (+48, real-binding test)
  reviewed:
    - macos-arm64/src/metal_host.rs (full diff + trait + session wrappers)
    - macos-arm64/tests/metal_host_test.rs (full diff)
    - macos-arm64/Cargo.toml (diff)
    - Cargo.lock (diff scope — metal transitive deps only)
    - radix/scripta/metal-spike, radix/scripta/metal-faber-spike (live run)
    - faber/core-support-manifest.txt (metal-dep blast radius)
  excluded:
    - webgpu-browser/public/generated/{kernel.wgsl,reflection.json} — foreign WIP
      (dirty in working tree), outside the assigned macos-arm64/metal surface;
      does not affect range reconstruction.
risk: medium - real GPU execution path (unsafe FFI + Metal command encoding), but
  confined to a macOS-only host crate not in the faber release payload; no CI.
verdict: clean_pass
verdict_basis: >
  All 9 requirements evidenced with live device execution. The new nextest test
  system_driver_compiles_msl_launches_add_one_and_reads_back compiles MSL at
  runtime via new_library_with_source, launches add_one over 16 elements
  (dispatch_thread_groups + wait_until_completed), and reads back exactly
  [1.0..16.0] on an Apple M5 Max in 0.192s — proving real device execution, not
  a fake. Trait is implemented method-for-method (9/9). Runtime-extent channel
  populates binding 2 with a u32 count. Cargo.toml change is the single metal
  crate dep; Cargo.lock additions are exactly its transitive deps. cuda_host.rs
  and lib.rs untouched; cuda baseline 4/4 green. Env-gating is correct (skip on
  no device; .expect hard-fail on broken stack). Both Swift spikes pass. Unsafe
  blocks are bounded and size-checked. No P0/P1/P2. The ungated metal dep is
  confined to a macOS-only crate absent from core-support-manifest, so it has no
  cross-platform release blast radius (recorded as blind spot, not a finding).
findings: []
observations:
  - id: OBS-1
    severity: none (informational, not a finding)
    confidence: confirmed
    category: interfaces
    where: macos-arm64/src/metal_host.rs launch_elementwise_add_f32 (b arg)
    expected: trait method launch_elementwise_add_f32(module,a,b,out,len)
    actual: >
      The emitted add_one kernel is unary (input@0, output@1, extent@2). The b
      token is validated fail-closed (stale id cannot launch silently) but is
      not bound to the encoder, matching the documented M2 design.
    impact: none. Documented in-source; matches trait parity with cuda_host and
      the current milestone scope (add_one only).
    evidence: driver validates contains_key(&b) then only set_buffer(0,1,2).
    disposition_for_mind: conscious acceptance; no action. Becomes relevant when
      a true binary add kernel is emitted (later milestone).
  - id: OBS-2
    severity: none (informational, not a finding)
    confidence: confirmed
    category: integration / ops
    where: macos-arm64/Cargo.toml (metal = "0.33", ungated by target cfg)
    expected: dependency confined to the macOS-only host crate
    actual: >
      metal = "0.33" is not wrapped in [target.'cfg(target_os="macos")'.dependencies].
      SystemMetalDriver + its impl block are also not cfg-gated (only
      probe_metal_environment's device probe is).
    impact: none currently. faber/core-support-manifest.txt does NOT embed
      macos-arm64 (it lists only hosts/crates/*), so the faber release build
      never pulls the metal dep. hosts has no .github/workflows CI. Local macOS
      build/clippy/test all green. The dep cannot reach a non-macOS build.
    evidence: grep shows no core-support crate depends on macos-arm64; manifest
      enumerated; ls .github/workflows empty.
    disposition_for_mind: >
      No action required for M2. If hosts later gains non-macOS CI or
      macos-arm64 is admitted to a cross-platform release surface, gate the dep
      and the SystemMetalDriver block behind cfg(target_os="macos") at that time.
requirements_evidence:
  - req: "1. SystemMetalDriver implements MetalDriver trait fully (all 9 methods)"
    status: SATISFIED
    evidence: >
      MetalDriver trait (metal_host.rs:63-82) declares exactly 9 methods:
      discover, create_context, load_module, alloc, copy_in,
      launch_elementwise_add_f32, sync, copy_out, free. SystemMetalDriver
      implements all 9 (impl block at metal_host.rs:440-654 region). Compiler
      accepts the impl; 6/6 tests pass.
  - req: "2. Runtime MSL compile via new_library_with_source (no offline metallib)"
    status: SATISFIED
    evidence: >
      load_module parses UTF-8 MSL source, calls device.new_library_with_source
      then get_function then new_compute_pipeline_state_with_function. No
      .metallib file path or offline artifact anywhere. The test passes MSL text
      (ADD_ONE_MSL constant) as image bytes.
  - req: "3. Launch add_one over 16 elements: dispatch_thread_groups + wait_until_completed"
    status: SATISFIED
    evidence: >
      launch_elementwise_add_f32 builds a command buffer + compute encoder,
      set_buffer(0,1,2), dispatch_thread_groups(MTLSize thread_groups x
      threads_per_threadgroup), end_encoding, commit(), wait_until_completed(),
      and asserts status == Completed. Test allocates 16 f32s and launches; live
      run returned in 0.192s.
  - req: "4. Readback [1.0..16.0] — real device execution, not faked"
    status: SATISFIED
    evidence: >
      Test input (0..16).map(i as f32), expected (1..=16).map(i as f32);
      assert_eq!(values, expected) PASSED on Apple M5 Max. The Rust
      SystemMetalDriver (not the Swift spike) executed: copy_out reads
      StorageModeShared buffer contents via copy_nonoverlapping. Both Swift
      spikes independently corroborate: "add_one [0.0..15.0] -> [1.0..16.0]".
  - req: "5. Runtime-extent binding populated (binding 2, u32 count)"
    status: SATISFIED
    evidence: >
      extent = len as u32; device.new_buffer_with_data(&extent, size_of::<u32>(),
      StorageModeShared); encoder.set_buffer(2, Some(&extent_buffer), 0). The
      emitted kernel guards `if (id >= extent_2[0]) return;`. metal-faber-spike
      mirrors the same binding-2 contract and passed.
  - req: "6. Cargo.toml change minimal (metal crate dep only)"
    status: SATISFIED
    evidence: >
      macos-arm64/Cargo.toml diff is exactly two lines: comment + `metal = "0.33"`.
      Cargo.lock diff adds only metal's transitive deps (block, core-foundation,
      core-foundation-sys, core-graphics-types, foreign-types(+macros/shared),
      malloc_buf, objc, paste) — no unrelated version bumps.
  - req: "7. No touching cuda_host.rs or lib.rs re-exports"
    status: SATISFIED
    evidence: >
      git diff --stat 778c0f1..652b07b lists exactly 4 files (Cargo.lock,
      macos-arm64/Cargo.toml, macos-arm64/src/metal_host.rs,
      macos-arm64/tests/metal_host_test.rs). cuda_host.rs and lib.rs absent.
      cuda_host_test baseline 4/4 green (parity preserved).
  - req: "8. Env-gated test: skips without Metal device, hard-fails with broken stack"
    status: SATISFIED
    evidence: >
      Test opens via `let Ok(mut session) = SystemMetalDriver::try_open() else {
      return; };` — try_open -> with_driver -> discover returns Err when
      Device::system_default() is None, so the test returns early (skip) on a
      machine without Metal. With a device present, any broken-stack failure
      (MSL compile, pipeline, launch, readback) hits .expect() and panics (hard
      fail). No silent pass path.
  - req: "9. metal-spike and metal-faber-spike both pass"
    status: SATISFIED
    evidence: >
      metal-spike: "metal spike OK: Apple M5 Max add_one [0.0..15.0] ->
      [1.0..16.0]" (exit 0). metal-faber-spike: "metal faber spike OK: Apple M5
      Max add_one [0.0..15.0] -> [1.0..16.0]" (exit 0). These are Swift
      environmental baselines (local Metal runtime + Radix metal-text emission);
      they corroborate the device is real and the U2 extent contract holds.
validation:
  - command: git rev-parse 778c0f1 && git rev-parse 652b07b && git rev-parse HEAD
    result: pass
    note: base=778c0f1e76f8..., head=652b07b0fcd7..., HEAD==head; range frozen
  - command: git diff --stat 778c0f1..652b07b
    result: pass
    note: 4 files, +396/-28; no cuda_host.rs/lib.rs in range
  - command: ./scripta/metal-spike  (from radix/)
    result: pass
    note: exit 0; "metal spike OK: Apple M5 Max add_one [0.0..15.0] -> [1.0..16.0]"
  - command: ./scripta/metal-faber-spike  (from radix/)
    result: pass
    note: exit 0, 7.81s; "metal faber spike OK: Apple M5 Max ..."
  - command: cargo nextest run -p faber-host-macos-arm64 --test metal_host_test
    result: pass
    note: 6/6 passed incl. system_driver_compiles_msl_launches_add_one_and_reads_back (0.192s)
  - command: cargo nextest run -p faber-host-macos-arm64 --test cuda_host_test
    result: pass
    note: baseline parity 4/4 passed
  - command: cargo clippy -p faber-host-macos-arm64 --all-targets
    result: pass
    note: zero warnings/errors on our surface; block v0.1.6 future-incompat note is a metal transitive dep, not our code
blind_spots:
  - >
    OBS-2: the metal = "0.33" dep and SystemMetalDriver impl are not cfg-gated to
    target_os="macos". Assessed as non-blocking: macos-arm64 is absent from
    core-support-manifest (not in the faber release payload) and hosts has no CI,
    so the dep cannot reach a non-macOS build today. If either condition changes,
    gate the dep + impl behind cfg(target_os="macos"). Verdict unaffected.
  - >
    No cross-lane consumer of SystemMetalDriver was audited beyond the
    macos-arm64 crate; this is a new driver with no external caller yet, so
    integration is proven against the MetalDriver trait contract and the local
    device only.
not_claimed:
  - Global repository correctness
  - Metal performance characteristics
  - Correctness of a true binary elementwise-add kernel (add_one is unary; M2 scope)
  - Non-macOS build portability (no such build exists in the release surface)
```
