//! Environment-gated CUDA Driver API proof (G3 artifact; the G2 wiring that
//! carries it).
//!
//! # How to compile and run the CUDA proof
//!
//! The proof runs end-to-end on a machine with an NVIDIA GPU and the CUDA
//! Driver API (e.g. pharos: RTX 5070, sm_120, driver 595.71.05,
//! `libcuda.so.1` at `/lib/x86_64-linux-gnu/libcuda.so.1`). It requires three
//! artifacts: the PTX file (compiler-emitted LLVM IR lowered through an
//! NVPTX backend), the kernel descriptor JSON sidecar, and this test binary
//! (compiled from the `faber-host-macos-arm64` crate).
//!
//! ## Prerequisites
//!
//! 1. **Rust toolchain** on the GPU machine (rustup + stable).
//! 2. **LLVM with NVPTX target** — either on the GPU machine or on a build
//!    machine that can emit PTX. Check with `clang --print-targets | grep
//!    nvptx64` or `llc --version | grep nvptx64`.
//! 3. **The faberlang repos** — `radix/` and `hosts/` checked out as siblings
//!    (the standard faberlang container layout).
//! 4. **cfg-gate**: the `metal` crate dependency in
//!    `hosts/macos-arm64/Cargo.toml` must be target-gated to macOS so the
//!    crate compiles on Linux:
//!    ```toml
//!    [target.'cfg(target_os = "macos")'.dependencies]
//!    metal = "0.33"
//!    ```
//!    If `metal_host.rs` has unconditional `use metal::*`, those must be
//!    cfg-gated too.
//!
//! ## Option A — Split build (recommended for single-GPU setups)
//!
//! Emit and lower on a machine with LLVM, run the proof on the GPU machine:
//!
//! ```sh
//! # Step 1: Emit .ll + descriptor (build machine with Rust + radix)
//! cd /path/to/faberlang/radix
//! cargo build -p radix --bin radix
//! ./target/debug/radix emit -t llvm-text \
//!   --cuda-descriptor /tmp/addita.json \
//!   corpus/cuda/addita-proof.fab > /tmp/addita.ll
//!
//! # Step 2: Lower to PTX (build machine with LLVM NVPTX backend)
//! clang --target=nvptx64-nvidia-cuda -S -O1 --cuda-feature=+ptx87 \
//!   -o /tmp/addita.ptx /tmp/addita.ll
//! # Or: llc -mtriple=nvptx64-nvidia-cuda -mattr=+ptx87 -o /tmp/addita.ptx /tmp/addita.ll
//!
//! # Step 3: Copy artifacts to the GPU machine
//! scp /tmp/addita.ptx /tmp/addita.json pharos:/tmp/
//!
//! # Step 4: Build and run the proof (GPU machine)
//! cd /path/to/faberlang/hosts
//! cargo test --manifest-path macos-arm64/Cargo.toml --test cuda_host_proof -- --nocapture
//! ```
//!
//! ## Option B — Full script (GPU machine has everything)
//!
//! Run the entire pipeline in one shot:
//!
//! ```sh
//! cd /path/to/faberlang/hosts
//! ./scripta/cuda-tier-f-proof
//! ```
//!
//! The script builds radix, emits, lowers, builds the host proof, and runs it.
//! Exit codes: 0 = PASS, 1 = FAIL, 2 = G3 not attempted (no NVPTX backend),
//! 3 = config error.
//!
//! ## Anti-false-green contract
//!
//! Requires both `CUDA_PROOF_PTX` and `CUDA_PROOF_DESCRIPTOR`. When either is
//! absent the test prints SKIP and exits clean — that is the only skip-worthy
//! state (anti-false-green: a present-but-broken CUDA stack must never look
//! green). When both are set, every failure is a loud FAIL, including a
//! `try_open` failure (dlopen/`cuInit` → `E_CUDA_UNAVAILABLE`; later driver
//! failures → `E_CUDA_DRIVER`).
//!
//! The descriptor is the C4 sidecar (schema_version 1, target `llvm-nvvm`):
//! a single `addita` kernel, f32, 4-byte, N elements, 2 input buffers /
//! 1 output buffer. Grid is `ceil(N / 256)`, block is 256.

use faber_host_macos_arm64::CudaHostSession;
use serde::Deserialize;

/// Sentinel bit pattern: every output byte is 0xFE. Prefilled into the output
/// buffer so a no-write or wrong-buffer bug is a hard mismatch, not a false
/// green.
const SENTINEL_BITS: u32 = 0xFEFE_FEFE;
const BLOCK_X: u32 = 256;
/// Pinned tolerance: `|actual − expected| ≤ 1e-6 * max(1, |expected|)`. Exact
/// IEEE equality is expected for a single correctly-rounded f32 add; this is a
/// backstop, not a relaxation.
const TOLERANCE: f32 = 1e-6;

#[derive(Deserialize)]
struct ProofDescriptor {
    schema_version: u32,
    target: String,
    kernels: Vec<ProofKernel>,
}

#[derive(Deserialize)]
struct ProofKernel {
    entry: String,
    element_type: String,
    element_byte_width: u32,
    element_count: u64,
    input_buffers: usize,
    output_buffers: usize,
}

#[test]
fn cuda_driver_api_proof() {
    let ptx_path = match std::env::var("CUDA_PROOF_PTX") {
        Ok(path) => path,
        Err(_) => {
            println!("SKIP: CUDA_PROOF_PTX not set — CUDA proof not requested");
            return;
        }
    };
    let descriptor_path = match std::env::var("CUDA_PROOF_DESCRIPTOR") {
        Ok(path) => path,
        Err(_) => {
            println!("SKIP: CUDA_PROOF_DESCRIPTOR not set — CUDA proof not requested");
            return;
        }
    };

    let ptx = std::fs::read(&ptx_path)
        .unwrap_or_else(|error| panic!("FAIL: CUDA_PROOF_PTX = {ptx_path} unreadable: {error}"));
    let descriptor_json = std::fs::read_to_string(&descriptor_path).unwrap_or_else(|error| {
        panic!("FAIL: CUDA_PROOF_DESCRIPTOR = {descriptor_path} unreadable: {error}")
    });
    let descriptor: ProofDescriptor = serde_json::from_str(&descriptor_json)
        .unwrap_or_else(|error| panic!("FAIL: proof descriptor JSON invalid: {error}"));

    assert_eq!(
        descriptor.schema_version, 1,
        "FAIL: descriptor schema_version {}",
        descriptor.schema_version
    );
    assert_eq!(
        descriptor.target, "llvm-nvvm",
        "FAIL: descriptor target {}",
        descriptor.target
    );
    assert_eq!(
        descriptor.kernels.len(),
        1,
        "FAIL: proof expects exactly one kernel, got {}",
        descriptor.kernels.len()
    );
    let kernel = &descriptor.kernels[0];
    assert_eq!(
        kernel.element_type, "f32",
        "FAIL: proof fixture element type {}",
        kernel.element_type
    );
    assert_eq!(
        kernel.element_byte_width, 4,
        "FAIL: proof fixture element byte width {}",
        kernel.element_byte_width
    );
    assert_eq!(
        kernel.input_buffers, 2,
        "FAIL: proof fixture input_buffers {}",
        kernel.input_buffers
    );
    assert_eq!(
        kernel.output_buffers, 1,
        "FAIL: proof fixture output_buffers {}",
        kernel.output_buffers
    );

    let n = kernel.element_count as usize;
    assert!(n > 0, "FAIL: proof element_count must be positive");
    let bytes = n * 4;

    // Env vars set ⇒ try_open failure is a loud FAIL, never a silent skip.
    let mut session = CudaHostSession::try_open().unwrap_or_else(|error| {
        panic!(
            "FAIL: CUDA proof requested but try_open failed: code={} message={}",
            error.code, error.message
        )
    });

    let module = session
        .load_module(&ptx)
        .unwrap_or_else(|error| panic!("FAIL: load_module: {}", error.message));
    let a = session
        .alloc_bytes(bytes)
        .unwrap_or_else(|error| panic!("FAIL: alloc a: {}", error.message));
    let b = session
        .alloc_bytes(bytes)
        .unwrap_or_else(|error| panic!("FAIL: alloc b: {}", error.message));
    let out = session
        .alloc_bytes(bytes)
        .unwrap_or_else(|error| panic!("FAIL: alloc out: {}", error.message));

    // Deterministic inputs (pinned by the goal): a[i] = i*3 + 1, b[i] = i*7.
    let input_a: Vec<f32> = (0..n).map(|i| (i * 3 + 1) as f32).collect();
    let input_b: Vec<f32> = (0..n).map(|i| (i * 7) as f32).collect();
    session
        .copy_in_f32(a, &input_a)
        .unwrap_or_else(|error| panic!("FAIL: copy a: {}", error.message));
    session
        .copy_in_f32(b, &input_b)
        .unwrap_or_else(|error| panic!("FAIL: copy b: {}", error.message));

    // Sentinel discipline: prefill the output destination with 0xFE bytes and
    // require the kernel to overwrite them.
    let sentinel = f32::from_bits(SENTINEL_BITS);
    let prefill = vec![sentinel; n];
    session
        .copy_in_f32(out, &prefill)
        .unwrap_or_else(|error| panic!("FAIL: sentinel prefill: {}", error.message));

    let grid_x = n.div_ceil(BLOCK_X as usize) as u32;
    session
        .launch_kernel(module, &kernel.entry, &[a, b, out], grid_x, BLOCK_X)
        .unwrap_or_else(|error| panic!("FAIL: launch_kernel: {}", error.message));

    let values = session
        .readback_f32(out)
        .unwrap_or_else(|error| panic!("FAIL: readback: {}", error.message));

    // The 0xFE sentinel must be gone — a no-write kernel cannot look green.
    assert!(
        !values.iter().any(|value| value.to_bits() == SENTINEL_BITS),
        "FAIL: output destination not overwritten (0xFE sentinel still present)"
    );

    // Rust reference: out[i] = a[i] + b[i].
    let expected: Vec<f32> = (0..n)
        .map(|i| ((i * 3 + 1) as f32) + ((i * 7) as f32))
        .collect();
    assert_eq!(
        values.len(),
        expected.len(),
        "FAIL: result length {} != {}",
        values.len(),
        expected.len()
    );
    for (i, (actual, expected)) in values.iter().zip(&expected).enumerate() {
        let tolerance = TOLERANCE * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "FAIL: element {i}: |{actual} − {expected}| > {tolerance}"
        );
    }
    println!(
        "PASS: CUDA proof — entry {} over {n} f32 elements (grid_x={grid_x}, block_x={BLOCK_X}) matched the Rust reference",
        kernel.entry
    );
}
