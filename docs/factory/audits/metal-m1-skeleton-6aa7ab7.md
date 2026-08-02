# Audit: Metal M1 skeleton (hosts 15802c1 → 6aa7ab7)

```yaml
kind: audit
audit_version: 1
auditor: auditor-2
assignment: 628a4ea4
repository: hosts (faberlang/hosts)
base: 15802c15c25c8a38f3ad794dbaad2f03ed52b025
head: 6aa7ab7dd3a87ed6967dd4cbc7ad1b6e6cb13fb7  # hosts main tip, clean tree
scope:
  requirements:
    - MetalDriver trait method-for-method identical to CudaDriver (zero deviation)
    - MetalHostSession + MetalHandleId mirror cuda_host.rs naming and lifecycle
    - probe_metal_environment() structured report, fail-closed when absent
    - E_METAL_* codes mirror E_CUDA_*
    - FakeMetalDriver in-memory test seam works
    - metal_host_test.rs: 5/5 fail-closed tests pass
    - lib.rs re-export correct, no Cargo.toml breakage
    - No real Metal binding (M2, deferred)
  changed:
    - macos-arm64/src/metal_host.rs   (+460, new)
    - macos-arm64/tests/metal_host_test.rs (+108, new)
    - macos-arm64/src/lib.rs          (+5, additive mod + pub use)
  reviewed:
    - macos-arm64/src/metal_host.rs (full)
    - macos-arm64/src/cuda_host.rs (full, reference parity)
    - macos-arm64/tests/metal_host_test.rs (full)
    - macos-arm64/src/lib.rs (diff)
    - macos-arm64/Cargo.toml (diff + dep surface)
    - crates/host-kernel/src/lib.rs (HostError::invalid_args -> E_INVALID_ARGS)
  excluded: []
risk: low - additive skeleton mirroring an established cuda_host pattern; no system driver, no product execution path, single gated admission probe
verdict: clean_pass
verdict_basis: >
  All 8 requirements evidenced; no P0/P1/P2. Trait parity is method-for-method
  identical (9 methods, : Send). Session/handle lifecycle mirrors cuda 1:1 with
  one documented intentional omission (try_open, required by the M2 deferral).
  Build + clippy clean; 5/5 metal tests pass; cuda baseline (4 tests) green;
  Cargo.toml unchanged; no Metal product binding crate present. The single FFI
  touchpoint is an admission probe analogous to cuda's nvidia-smi, not the M2
  SystemMetalDriver binding; recorded as a blind spot, not a finding.
findings: []
observations:
  - id: OBS-1
    severity: none (informational, not a finding)
    confidence: confirmed
    category: interfaces / integration
    where: macos-arm64/src/metal_host.rs:274-276, 284-288
    expected: admission probe detects device presence; mirrors cuda's nvidia-smi admission probe in role
    actual: >
      cuda_host uses a subprocess (nvidia-smi) + filesystem path checks for
      admission; metal_host uses a direct FFI link (#[link(name="Metal", kind="framework")]
      extern "C" { MTLCreateSystemDefaultDevice }) for the device null probe.
      This is a real Metal framework link, stronger coupling than a subprocess.
    impact: none blocking. The call is cfg-guarded to target_os="macos" (non-macOS
      fails closed with device_detected=false), performs a null check only (no
      pointer dereference), and never implements alloc/launch/copy. The retained
      system-default device singleton is the Apple-recommended process-lifetime
      pattern. A filesystem-only check would be strictly weaker (every macOS has
      Metal.framework, so presence != device capability). Documented at
      metal_host.rs:6-8 and 266-276 as the single M1 touchpoint.
    evidence: grep confirms MTLCreateSystemDefaultDevice is the ONLY Metal symbol;
      no metal/gfx-rs/objc2 crate dep exists in hosts. Test #1 passes on Apple
      Silicon (admitted path) exercising the real probe.
    reproduction: cargo test -p faber-host-macos-arm64 --test metal_host_test
    disposition_for_mind: conscious acceptance only; no action required. This is
      the documented, intentional admission mechanism, not a leaked M2 binding.
requirements_evidence:
  - req: "1. MetalDriver trait method-for-method identical to CudaDriver"
    status: SATISFIED
    evidence: >
      Both traits: Send + 9 methods with identical arg/return shape: discover,
      create_context, load_module(image)->u64, alloc(len_bytes)->u64,
      copy_in(token,bytes), launch_elementwise_add_f32(module,a,b,out,len),
      sync, copy_out(token,len_bytes)->Vec<u8>, free(token). Only the discover
      report type name differs (CudaEnvReport vs MetalEnvReport), the structural
      equivalent. cuda_host.rs:54-72 vs metal_host.rs:59-77.
  - req: "2. MetalHostSession + MetalHandleId mirror cuda_host.rs naming/lifecycle"
    status: SATISFIED (documented intentional omission)
    evidence: >
      All mirrored 1:1: with_driver, is_admitted, load_module, alloc_bytes,
      copy_in_f32, launch_elementwise_add_f32, readback_f32, release,
      handle_frame_data (key "metal_handle" vs "cuda_handle"), private
      require_admitted/insert/module_token/buffer_token. MetalHandleId(pub u64)
      mirrors CudaHandleId; MetalHandleKind (Module | Buffer{len_bytes}) mirrors
      CudaHandleKind; Debug impl mirrors. OMISSION: try_open() absent — REQUIRED
      by req #8 (SystemMetalDriver is M2); documented at metal_host.rs:6-8, 80-81.
      Logic bodies byte-equivalent modulo Cuda<->Metal name substitutions.
  - req: "3. probe_metal_environment() structured report, fail-closed when absent"
    status: SATISFIED
    evidence: >
      Returns MetalEnvReport{admitted, mtl_device: Option<String>,
      metal_framework_paths: Vec<String>, reason}. macOS+device => admitted=true,
      mtl_device=Some("system default Metal device"). Non-macOS / null device =>
      admitted=false with explicit fail-closed reason. cfg(not(target_os="macos"))
      forces device_detected=false. Test #1 passes on this Apple Silicon host.
  - req: "4. E_METAL_* codes mirror E_CUDA_*"
    status: SATISFIED
    evidence: >
      E_METAL_UNAVAILABLE/E_METAL_UNSUPPORTED/E_METAL_INVALID_HANDLE/E_METAL_DRIVER
      mirror the 4 E_CUDA_* codes. Test #2 asserts all 4 string values. lib.rs
      re-export set mirrors cuda exactly (3 of 4 re-exported; _DRIVER reachable
      via module path on both sides — parity preserved).
  - req: "5. FakeMetalDriver in-memory test seam works"
    status: SATISFIED
    evidence: >
      FakeMetalDriver implements MetalDriver with in-memory buffers/modules;
      computes f32 elementwise add. Test #3 runs full lifecycle and asserts
      readback == [4.0, 6.0] (oracle 1+3, 2+4). Test #4 verifies unavailable()
      reject path returns E_METAL_UNAVAILABLE.
  - req: "6. metal_host_test.rs: 5/5 fail-closed tests pass"
    status: SATISFIED
    evidence: >
      5 tests, 5 passed/0 failed: error_codes_mirror_cuda_family,
      fake_unavailable_driver_rejects_session_open,
      session_fails_closed_on_guard_checks (5 negative assertions on E_INVALID_ARGS
      + stale handle E_METAL_INVALID_HANDLE),
      fake_driver_sequences_elementwise_add_without_product_label,
      probe_reports_structured_admission_without_claiming_product_run.
  - req: "7. lib.rs re-export correct, no Cargo.toml breakage"
    status: SATISFIED
    evidence: >
      lib.rs diff is purely additive (pub mod metal_host + pub use block mirroring
      cuda's). Cargo.toml has zero diff in 15802c1..6aa7ab7. No metal/gfx-rs/objc2
      dependency anywhere in hosts. cargo build + clippy --all-targets clean.
  - req: "8. No real Metal binding (M2, deferred)"
    status: SATISFIED
    evidence: >
      No SystemMetalDriver type exists. No metal/gfx-rs/objc2 crate dependency.
      The only Metal framework symbol is MTLCreateSystemDefaultDevice, used solely
      as a device-existence admission probe (see OBS-1) — not a product binding.
      Full Driver-API surface (alloc/launch/copy/context) has no system impl; only
      FakeMetalDriver (sequencing) is present. Consistent with M2 deferral.
validation:
  - command: git rev-parse 6aa7ab7 && git rev-parse 15802c1
    result: pass
    note: both SHAs resolve; range frozen; head == main tip; tree clean
  - command: git diff --stat 15802c1 6aa7ab7
    result: pass
    note: 3 files, +573/-0; no Cargo.toml in diff
  - command: cargo build -p faber-host-macos-arm64
    result: pass
    note: clean compile, 0.09s cached
  - command: cargo test -p faber-host-macos-arm64 --test metal_host_test
    result: pass
    note: 5 passed, 0 failed, 0 ignored
  - command: cargo test -p faber-host-macos-arm64 --test cuda_host_test
    result: pass
    note: baseline parity 4 passed, 0 failed
  - command: cargo clippy -p faber-host-macos-arm64 --all-targets
    result: pass
    note: zero warnings, zero errors
  - command: grep -rn "metal" --include=Cargo.toml . (filtered)
    result: pass
    note: no metal crate dependency anywhere in hosts
blind_spots:
  - >
    OBS-1 (the MTLCreateSystemDefaultDevice FFI admission probe) is a real Metal
    framework link. I assess it as an intentional admission probe analogous to
    cuda's nvidia-smi, NOT the deferred M2 SystemMetalDriver product binding, and
    therefore not a finding. Mind should consciously accept this design choice;
    if a stricter "no Metal framework link at M1" policy is intended, that is a
    scope decision for Mind, not a code defect. Verdict unaffected.
  - >
    No cross-lane consumer of the MetalHostSession surface was audited beyond the
    macos-arm64 crate; this is a new additive module with no external callers yet,
    so integration parity is proven against cuda_host.rs (the named reference) only.
not_claimed:
  - Global repository correctness
  - M2 SystemMetalDriver correctness (does not exist; out of scope)
  - Production Metal execution (no binding; intentionally unclaimed)
  - Performance characteristics
```
