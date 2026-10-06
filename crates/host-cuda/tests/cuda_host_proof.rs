//! Environment-gated CUDA Driver API proof (G3 artifact; the G2 wiring that
//! carries it).
//!
//! # How to compile and run the CUDA proof
//!
//! The proof runs end-to-end on a machine with an NVIDIA GPU and the CUDA
//! Driver API (e.g. pharos: RTX 5070, `sm_120`, driver 595.71.05,
//! `libcuda.so.1` at `/lib/x86_64-linux-gnu/libcuda.so.1`). It requires three
//! artifacts: the PTX file (compiler-emitted LLVM IR lowered through an
//! NVPTX backend), the kernel descriptor JSON sidecar, and this test binary
//! (compiled from the `host-cuda` crate).
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
//! ## Pipeline (GPU machine has everything)
//!
//! ```sh
//! cd /path/to/faberlang/radix
//! ./target/debug/radix emit -t llvm-text \
//!   --cuda-descriptor /tmp/cuda-g6/<name>.descriptor.json \
//!   corpus/cuda/<name>.fab > /tmp/cuda-g6/<name>.ll
//! llc -march=nvptx64 -mcpu=sm_80 -mattr=+ptx87 \
//!   /tmp/cuda-g6/<name>.ll -o /tmp/cuda-g6/<name>.ptx
//! cd /path/to/faberlang/hosts
//! CUDA_PROOF_PTX=/tmp/cuda-g6/<name>.ptx \
//! CUDA_PROOF_DESCRIPTOR=/tmp/cuda-g6/<name>.descriptor.json \
//! cargo test -p host-cuda --test cuda_host_proof -- --nocapture
//! ```
//!
//! `./scripta/cuda-tier-f-proof` (radix repo) drives the whole pipeline for
//! the `addita` proof row.
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
//! ## Descriptor-driven multi-kernel proof
//!
//! The harness iterates EVERY kernel the descriptor carries (the corpus
//! proof files may hold several `@ kernel` functions — the glyph elementwise
//! proof carries rank-1 and rank-2 hadamard). Each kernel is launched with
//! its own descriptor launch geometry (recipes are load-bearing: the tiled
//! matmul needs its `(8, 8, 1)` workgroup over the `(2, 2, 1)` tile grid)
//! and checked against a per-element host oracle.
//!
//! The oracle comes from the descriptor's own typed facts: a recipe kernel's
//! `plan` (`tiled_matmul` / `tree_reduction` / `transpose`) fully determines
//! the reference computation, so no kernel-entry keying is needed there.
//! Elementwise kernels carry no plan fact (the descriptor records none), so
//! the fixture table below pins the corpus semantics per entry — the same
//! way the original single-kernel proof pinned `out[i] = a[i] + b[i]` for
//! `addita`. This is test-fixture oracle pinning only: the compiler and the
//! host never key device behavior on entry names. An entry outside the
//! table fails closed (`FAIL: no pinned oracle`), never silently passes.

use host_cuda::{
    CudaHandleId, CudaHostSession, NVVM_DESCRIPTOR_SCHEMA_VERSION, NVVM_DESCRIPTOR_TARGET,
};
use serde::Deserialize;

/// Sentinel bit pattern: every output byte is 0xFE. Prefilled into the output
/// buffer so a no-write or wrong-buffer bug is a hard mismatch, not a false
/// green.
const SENTINEL_BITS: u32 = 0xFEFE_FEFE;
/// Pinned tolerance: `|actual − expected| ≤ TOLERANCE * max(1, |expected|)`.
/// Every oracle input is a small exact integer, so exact-integer results
/// (add, mul, matmul MACs, chunk sums, transpose copies) must match to the
/// backstop only; the composed silu polynomial exp is accurate far inside it.
const TOLERANCE: f32 = 1e-5;

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
    element_counts: Vec<u64>,
    input_buffers: usize,
    output_buffers: usize,
    accumulation_buffers: usize,
    buffers: Vec<ProofBuffer>,
    launch: ProofLaunch,
    plan: Option<ProofPlan>,
}

#[derive(Deserialize)]
struct ProofPlan {
    kind: String,
    m: Option<u64>,
    k: Option<u64>,
    n: Option<u64>,
    workgroup_x: Option<u32>,
    op: Option<String>,
    length: Option<u64>,
    partials: Option<u64>,
}

#[derive(Deserialize)]
struct ProofBuffer {
    role: String,
    binding: u32,
    element_count: u64,
    shape: Vec<u64>,
}

#[derive(Deserialize)]
struct ProofLaunch {
    workgroup: ProofAxis,
    dispatch: ProofAxis,
}

#[derive(Deserialize)]
struct ProofAxis {
    x: u64,
    y: u64,
    z: u64,
}

/// One pinned f32 input sequence: `(i * multiplier + addend) % bound + 1`.
/// The bound keeps every product the oracles form inside f32's exact-integer
/// range, so correctly-rounded device arithmetic matches the host reference
/// exactly and `TOLERANCE` stays a backstop, not a relaxation.
fn bounded_f32_sequence(len: usize, multiplier: usize, addend: usize, bound: usize) -> Vec<f32> {
    let values: Vec<usize> = (0..len)
        .map(|index| (index * multiplier + addend) % bound + 1)
        .collect();
    values
        .into_iter()
        .map(|value| {
            #[allow(
                clippy::cast_precision_loss,
                reason = "the bound keeps every value a small exact integer"
            )]
            {
                value as f32
            }
        })
        .collect()
}

/// The pinned addita law (the original G3 goal input): `a[i] = i*3 + 1`,
/// `b[i] = i*7`. Exact for the proof sizes (the sequence assert below
/// guards the f32 exact-integer range).
fn pinned_sequence(len: usize, multiplier: usize, addend: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let integer = index
                .checked_mul(multiplier)
                .and_then(|value| value.checked_add(addend))
                .expect("FAIL: proof input sequence overflows host usize");
            assert!(
                integer <= 1 << f32::MANTISSA_DIGITS,
                "FAIL: proof input sequence exceeds f32's exact integer range"
            );
            #[allow(
                clippy::cast_precision_loss,
                reason = "the preceding bound proves this integer is exactly representable as f32"
            )]
            {
                integer as f32
            }
        })
        .collect()
}

/// The host oracle for one kernel: deterministic input fills per buffer and
/// the expected output per output buffer.
struct KernelOracle {
    inputs: Vec<Vec<f32>>,
    expected_outputs: Vec<Vec<f32>>,
    label: &'static str,
}

/// The oracle for a recipe kernel, derived from the descriptor's own plan
/// facts. Missing plan facts fail closed.
fn plan_oracle(kernel: &ProofKernel) -> Result<KernelOracle, String> {
    let plan = kernel
        .plan
        .as_ref()
        .ok_or_else(|| "recipe kernel carries no plan fact".to_owned())?;
    let input_counts: Vec<u64> = kernel
        .buffers
        .iter()
        .filter(|buffer| buffer.role == "input" || buffer.role == "extra-input")
        .map(|buffer| buffer.element_count)
        .collect();
    match plan.kind.as_str() {
        "tiled_matmul" => {
            let m = plan.m.ok_or("tiled_matmul plan fact m missing")?;
            let k = plan.k.ok_or("tiled_matmul plan fact k missing")?;
            let n = plan.n.ok_or("tiled_matmul plan fact n missing")?;
            if input_counts.len() != 2
                || input_counts[0] != m * k
                || input_counts[1] != k * n
                || kernel.output_buffers != 1
                || kernel.buffers.last().map(|b| b.element_count) != Some(m * n)
            {
                return Err("tiled_matmul buffer counts contradict the plan M·K/K·N/M·N".to_owned());
            }
            let a = pinned_sequence(input_counts[0] as usize, 3, 1);
            let b = pinned_sequence(input_counts[1] as usize, 7, 0);
            let mut expected = vec![0.0f32; (m * n) as usize];
            for i in 0..m as usize {
                for j in 0..n as usize {
                    let mut acc = 0.0f32;
                    for kk in 0..k as usize {
                        acc += a[i * k as usize + kk] * b[kk * n as usize + j];
                    }
                    expected[i * n as usize + j] = acc;
                }
            }
            Ok(KernelOracle {
                inputs: vec![a, b],
                expected_outputs: vec![expected],
                label: "tiled_matmul",
            })
        }
        "tree_reduction" => {
            let length = plan
                .length
                .ok_or("tree_reduction plan fact length missing")?;
            let partials = plan
                .partials
                .ok_or("tree_reduction plan fact partials missing")?;
            if input_counts.len() != 1
                || input_counts[0] != length
                || kernel.output_buffers != 1
                || kernel.buffers.last().map(|b| b.element_count) != Some(partials)
                || partials == 0
                || length < partials
            {
                return Err(
                    "tree_reduction buffer counts contradict the plan length/partials".to_owned(),
                );
            }
            let a = pinned_sequence(length as usize, 3, 1);
            // The body's grid-stride law: workgroup w reduces its contiguous
            // leading chunk `[w·chunk, (w+1)·chunk)` (chunk = length /
            // partials, workgroup_x lanes striding by workgroup_x · partials).
            let chunk = (length / partials) as usize;
            let expected = (0..partials as usize)
                .map(|w| a[w * chunk..(w + 1) * chunk].iter().sum::<f32>())
                .collect();
            Ok(KernelOracle {
                inputs: vec![a],
                expected_outputs: vec![expected],
                label: "tree_reduction",
            })
        }
        "transpose" => {
            let m = plan.m.ok_or("transpose plan fact m missing")?;
            let n = plan.n.ok_or("transpose plan fact n missing")?;
            if input_counts.len() != 1
                || input_counts[0] != m * n
                || kernel.output_buffers != 1
                || kernel.buffers.last().map(|b| b.element_count) != Some(m * n)
            {
                return Err("transpose buffer counts contradict the plan M·N".to_owned());
            }
            let input = pinned_sequence((m * n) as usize, 3, 1);
            let mut expected = vec![0.0f32; (m * n) as usize];
            for i in 0..m as usize {
                for j in 0..n as usize {
                    expected[j * m as usize + i] = input[i * n as usize + j];
                }
            }
            Ok(KernelOracle {
                inputs: vec![input],
                expected_outputs: vec![expected],
                label: "transpose",
            })
        }
        other => Err(format!("descriptor plan kind {other} has no proof oracle")),
    }
}

/// The pinned corpus oracle for a plan-less (elementwise) kernel. Keyed by
/// entry — see the module doc: fixture pinning only, never device behavior.
fn elementwise_oracle(kernel: &ProofKernel) -> Result<KernelOracle, String> {
    let input_counts: Vec<u64> = kernel
        .buffers
        .iter()
        .filter(|buffer| buffer.role == "input" || buffer.role == "extra-input")
        .map(|buffer| buffer.element_count)
        .collect();
    let equal_counts = kernel.output_buffers == 1
        && !input_counts.is_empty()
        && input_counts.iter().all(|count| {
            *count == input_counts[0]
                && Some(*count) == kernel.buffers.last().map(|b| b.element_count)
        });
    match kernel.entry.as_str() {
        // addita-proof: the original pinned G3 law `out[i] = a[i] + b[i]`
        // over `a[i] = i*3+1`, `b[i] = i*7`.
        "addita" if equal_counts && input_counts.len() == 2 => {
            let n = input_counts[0] as usize;
            let a = pinned_sequence(n, 3, 1);
            let b = pinned_sequence(n, 7, 0);
            let expected = a.iter().zip(&b).map(|(x, y)| x + y).collect();
            Ok(KernelOracle {
                inputs: vec![a, b],
                expected_outputs: vec![expected],
                label: "elementwise add (pinned addita law)",
            })
        }
        // glyph-elementwise-proof: rank-1 and rank-2 hadamard products.
        // The bounded law keeps products inside f32's exact-integer range.
        "hadamard" | "hadamard_rank2" if equal_counts && input_counts.len() == 2 => {
            let n = input_counts[0] as usize;
            let a = bounded_f32_sequence(n, 3, 1, 16);
            let b = bounded_f32_sequence(n, 7, 0, 16);
            let expected = a.iter().zip(&b).map(|(x, y)| x * y).collect();
            Ok(KernelOracle {
                inputs: vec![a, b],
                expected_outputs: vec![expected],
                label: "elementwise mul (hadamard)",
            })
        }
        // silu-proof: the composed activation x/(1+exp(−x)).
        "silu" if equal_counts && input_counts.len() == 1 => {
            let n = input_counts[0] as usize;
            let x = pinned_sequence(n, 3, 1);
            let expected = x
                .iter()
                .map(|value| value / (1.0 + (-value).exp()))
                .collect();
            Ok(KernelOracle {
                inputs: vec![x],
                expected_outputs: vec![expected],
                label: "elementwise silu (composed)",
            })
        }
        _ => Err(format!(
            "no pinned oracle for kernel entry `{}` ({} inputs, {} output, plan {:?}) — \
             extend the elementwise fixture table",
            kernel.entry,
            input_counts.len(),
            kernel.output_buffers,
            kernel.plan.as_ref().map(|plan| plan.kind.as_str()),
        )),
    }
}

fn assert_common_kernel_facts(kernel: &ProofKernel) {
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
        kernel.accumulation_buffers, 0,
        "FAIL: proof fixture accumulation_buffers {}",
        kernel.accumulation_buffers
    );
    let expected_buffer_count =
        kernel.input_buffers + kernel.output_buffers + kernel.accumulation_buffers;
    assert_eq!(
        kernel.buffers.len(),
        expected_buffer_count,
        "FAIL: kernel {} buffers.len() {} != input+output+accumulation {}",
        kernel.entry,
        kernel.buffers.len(),
        expected_buffer_count
    );
    assert_eq!(
        kernel.element_counts.len(),
        kernel.buffers.len(),
        "FAIL: kernel {} element_counts.len() {} != buffers.len() {}",
        kernel.entry,
        kernel.element_counts.len(),
        kernel.buffers.len()
    );
    // The launch passes device buffers positionally in binding order; the
    // descriptor must carry the identity binding map.
    for (index, buffer) in kernel.buffers.iter().enumerate() {
        assert_eq!(
            buffer.binding, index as u32,
            "FAIL: kernel {} buffer position {index} carries binding {} (identity order required)",
            kernel.entry, buffer.binding
        );
        assert_eq!(
            kernel.element_counts[index], buffer.element_count,
            "FAIL: kernel {} element_counts[{index}] {} != buffer binding {} count {}",
            kernel.entry, kernel.element_counts[index], buffer.binding, buffer.element_count
        );
        let shape_product = buffer.shape.iter().product::<u64>();
        assert_eq!(
            shape_product, buffer.element_count,
            "FAIL: kernel {} buffer binding {} shape {:?} product {shape_product} != element_count {}",
            kernel.entry, buffer.binding, buffer.shape, buffer.element_count
        );
        assert!(
            buffer.element_count > 0,
            "FAIL: kernel {} buffer binding {} is empty",
            kernel.entry,
            buffer.binding
        );
    }
    for (label, axis) in [
        ("workgroup", &kernel.launch.workgroup),
        ("dispatch", &kernel.launch.dispatch),
    ] {
        assert!(
            axis.x > 0 && axis.y > 0 && axis.z > 0,
            "FAIL: kernel {} launch.{label} has a zero axis ({}, {}, {})",
            kernel.entry,
            axis.x,
            axis.y,
            axis.z
        );
        assert!(
            u32::try_from(axis.x).is_ok()
                && u32::try_from(axis.y).is_ok()
                && u32::try_from(axis.z).is_ok(),
            "FAIL: kernel {} launch.{label} axis does not fit u32 ({}, {}, {})",
            kernel.entry,
            axis.x,
            axis.y,
            axis.z
        );
    }
}

fn launch_oracle(
    session: &mut CudaHostSession,
    module: CudaHandleId,
    kernel: &ProofKernel,
    oracle: &KernelOracle,
) -> Result<(), String> {
    let mut handles = Vec::with_capacity(kernel.buffers.len());
    for (index, buffer) in kernel.buffers.iter().enumerate() {
        let bytes = buffer.element_count as usize * std::mem::size_of::<f32>();
        let handle = session
            .alloc_bytes(bytes)
            .map_err(|error| format!("alloc buffer {}: {}", buffer.binding, error.message))?;
        if let Some(values) = oracle.inputs.get(index) {
            session
                .copy_in_f32(handle, values)
                .map_err(|error| format!("copy buffer {}: {}", buffer.binding, error.message))?;
        } else {
            // Output destination: sentinel prefill so a no-write or
            // wrong-buffer bug is a hard mismatch, not a false green.
            let prefill = vec![f32::from_bits(SENTINEL_BITS); buffer.element_count as usize];
            session.copy_in_f32(handle, &prefill).map_err(|error| {
                format!("sentinel prefill {}: {}", buffer.binding, error.message)
            })?;
        }
        handles.push(handle);
    }
    let launch = &kernel.launch;
    let grid_x = u32::try_from(launch.dispatch.x).map_err(|_| "dispatch.x does not fit u32")?;
    let grid_y = u32::try_from(launch.dispatch.y).map_err(|_| "dispatch.y does not fit u32")?;
    let grid_z = u32::try_from(launch.dispatch.z).map_err(|_| "dispatch.z does not fit u32")?;
    let block_x = u32::try_from(launch.workgroup.x).map_err(|_| "workgroup.x does not fit u32")?;
    let block_y = u32::try_from(launch.workgroup.y).map_err(|_| "workgroup.y does not fit u32")?;
    let block_z = u32::try_from(launch.workgroup.z).map_err(|_| "workgroup.z does not fit u32")?;
    session
        .launch_kernel_3d(
            module,
            &kernel.entry,
            &handles,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
        )
        .map_err(|error| format!("launch_kernel: {}", error.message))?;

    for (index, buffer) in kernel.buffers.iter().enumerate() {
        let is_output = buffer.role == "output" || buffer.role == "extra-output";
        let output_slot = index.wrapping_sub(oracle.inputs.len());
        if !is_output || output_slot >= oracle.expected_outputs.len() {
            continue;
        }
        let expected = &oracle.expected_outputs[output_slot];
        let values = session
            .readback_f32(handles[index])
            .map_err(|error| format!("readback buffer {}: {}", buffer.binding, error.message))?;
        assert!(
            values.iter().all(|value| value.to_bits() != SENTINEL_BITS),
            "FAIL: kernel {} output buffer {} not fully overwritten (0xFE sentinel still present)",
            kernel.entry,
            buffer.binding
        );
        assert_eq!(
            values.len(),
            expected.len(),
            "FAIL: kernel {} output length {} != expected {}",
            kernel.entry,
            values.len(),
            expected.len()
        );
        for (i, (actual, expected)) in values.iter().zip(expected).enumerate() {
            let tolerance = TOLERANCE * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "FAIL: kernel {} ({}) element {i}: |{actual} − {expected}| > {tolerance}",
                kernel.entry,
                oracle.label
            );
        }
    }
    for handle in handles {
        session
            .release(handle)
            .map_err(|error| format!("release buffer: {}", error.message))?;
    }
    Ok(())
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "the environment-gated proof is one end-to-end CUDA allocation, launch, readback, and teardown scenario"
)]
fn cuda_driver_api_proof() {
    let Ok(ptx_path) = std::env::var("CUDA_PROOF_PTX") else {
        println!("SKIP: CUDA_PROOF_PTX not set — CUDA proof not requested");
        return;
    };
    let Ok(descriptor_path) = std::env::var("CUDA_PROOF_DESCRIPTOR") else {
        println!("SKIP: CUDA_PROOF_DESCRIPTOR not set — CUDA proof not requested");
        return;
    };

    let ptx = std::fs::read(&ptx_path)
        .unwrap_or_else(|error| panic!("FAIL: CUDA_PROOF_PTX = {ptx_path} unreadable: {error}"));
    let descriptor_json = std::fs::read_to_string(&descriptor_path).unwrap_or_else(|error| {
        panic!("FAIL: CUDA_PROOF_DESCRIPTOR = {descriptor_path} unreadable: {error}")
    });
    let descriptor: ProofDescriptor = serde_json::from_str(&descriptor_json)
        .unwrap_or_else(|error| panic!("FAIL: proof descriptor JSON invalid: {error}"));

    assert_eq!(
        descriptor.schema_version, NVVM_DESCRIPTOR_SCHEMA_VERSION,
        "FAIL: descriptor schema_version {} (expected {NVVM_DESCRIPTOR_SCHEMA_VERSION})",
        descriptor.schema_version
    );
    assert_eq!(
        descriptor.target, NVVM_DESCRIPTOR_TARGET,
        "FAIL: descriptor target {}",
        descriptor.target
    );
    assert!(
        !descriptor.kernels.is_empty(),
        "FAIL: descriptor carries no kernels"
    );

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

    for kernel in &descriptor.kernels {
        assert_common_kernel_facts(kernel);
        let oracle = if kernel.plan.is_some() {
            plan_oracle(kernel)
        } else {
            elementwise_oracle(kernel)
        }
        .unwrap_or_else(|error| panic!("FAIL: kernel {} oracle: {error}", kernel.entry));
        launch_oracle(&mut session, module, kernel, &oracle)
            .unwrap_or_else(|error| panic!("FAIL: kernel {}: {error}", kernel.entry));
        println!(
            "PASS: kernel {} — {} (grid {}×{}×{}, block {}×{}×{})",
            kernel.entry,
            if kernel.plan.is_some() {
                format!(
                    "plan {:?}",
                    kernel.plan.as_ref().map(|plan| plan.kind.clone())
                )
            } else {
                "pinned elementwise oracle".to_owned()
            },
            kernel.launch.dispatch.x,
            kernel.launch.dispatch.y,
            kernel.launch.dispatch.z,
            kernel.launch.workgroup.x,
            kernel.launch.workgroup.y,
            kernel.launch.workgroup.z,
        );
    }
    println!(
        "PASS: CUDA proof — {} kernel(s) matched their host oracles",
        descriptor.kernels.len()
    );
}
