/**
 * Matmul + add multi-kernel proof for real WebGPU execution.
 *
 * Dispatches matmul [4×3] · [3×2] then elementwise add [4×2] + [4×2]
 * on-device via runKernelChain, reads back the final output, compares
 * against the known stepper reference, and sets window.faberWebGpuProof.
 *
 * Expected values (hand-computed stepper reference for tiny-linear):
 *   y = [[9.1, 12.2], [18.1, 24.2], [27.1, 36.2], [36.1, 48.2]]
 */

import { FaberKernelContractError } from "./contract/artifact-admission.js";
import {
  acquireWebGpuDevice,
  runKernelChain,
} from "./backend/webgpu-runtime.js";

// ── Matrix shape constants ────────────────────────────────────────────────
const M = 4; // x rows
const K = 3; // x cols / W rows
const N = 2; // W cols

const X_ELEM = M * K;    // 12
const W_ELEM = K * N;    // 6
const OUT_ELEM = M * N;  // 8

const F32_BYTES = 4;
const X_BYTES = X_ELEM * F32_BYTES;   // 48
const W_BYTES = W_ELEM * F32_BYTES;   // 24
const OUT_BYTES = OUT_ELEM * F32_BYTES; // 32

// Stepper reference: y = x·W + b
// x  = [[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
// W  = [[1,2],[3,4],[5,6]]
// b  = [[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2]]
const INPUT_X = new Float32Array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]);
const INPUT_W = new Float32Array([1, 2, 3, 4, 5, 6]);
const INPUT_B = new Float32Array([0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2]);
const EXPECTED = [9.1, 12.2, 18.1, 24.2, 27.1, 36.2, 36.1, 48.2];
const EPSILON = 0.001;

// Resource layout (shared buffers Map):
//   0 → x       (48 bytes, matmul input A)
//   1 → W       (24 bytes, matmul input B)
//   2 → matmul_out / c  (32 bytes, matmul output → add input)
//   3 → b       (32 bytes, add input bias)
//   4 → add_out / y     (32 bytes, add output → final readback)

// ── WGSL kernel sources ───────────────────────────────────────────────────

const MATMUL_WGSL = `\
// Naive tiled matmul [M×K] · [K×N] = [M×N] for M=4, K=3, N=2.
// One workgroup per output element: (M, N, 1) = (4, 2, 1).

@group(0) @binding(0) var<storage, read> a_in: array<f32>;
@group(0) @binding(1) var<storage, read> b_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(4, 2, 1)
fn matmul_4x3_3x2(@builtin(global_invocation_id) id: vec3<u32>) {
  let row: u32 = id.x;
  let col: u32 = id.y;
  if (row >= ${M}u || col >= ${N}u) { return; }
  var sum: f32 = 0.0;
  for (var k: u32 = 0u; k < ${K}u; k = k + 1u) {
    sum = sum + a_in[row * ${K}u + k] * b_in[k * ${N}u + col];
  }
  output[row * ${N}u + col] = sum;
}`;

const ADD_WGSL = `\
// Elementwise add [E] + [E] = [E] for E=${OUT_ELEM}.
@group(0) @binding(0) var<storage, read> a_in: array<f32>;
@group(0) @binding(1) var<storage, read> b_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(${OUT_ELEM}, 1, 1)
fn add_bias(@builtin(global_invocation_id) id: vec3<u32>) {
  let i: u32 = id.x;
  if (i >= ${OUT_ELEM}u) { return; }
  output[i] = a_in[i] + b_in[i];
}`;

// ── Reflection descriptors (minimal — only dispatch sizes needed) ────────

const MATMUL_REFLECTION = Object.freeze({
  schema_version: 1,
  target: "wgsl-text",
  kernels: [Object.freeze({
    entry_name: "matmul_4x3_3x2",
    shader_stage: "compute",
    launch: Object.freeze({
      webgpu_adapter: Object.freeze({
        dispatch_workgroups: Object.freeze({ x: M, y: N, z: 1 }),
      }),
    }),
  })],
});

const ADD_REFLECTION = Object.freeze({
  schema_version: 1,
  target: "wgsl-text",
  kernels: [Object.freeze({
    entry_name: "add_bias",
    shader_stage: "compute",
    launch: Object.freeze({
      webgpu_adapter: Object.freeze({
        dispatch_workgroups: Object.freeze({ x: OUT_ELEM, y: 1, z: 1 }),
      }),
    }),
  })],
});

// ── Entry point ───────────────────────────────────────────────────────────

window.faberWebGpuProof = Object.freeze({ ok: false, status: "starting" });

main().catch((error) => {
  const proof = proofFailure(error);
  window.faberWebGpuProof = proof;
  console.log("FABER_MATMUL_PROOF:", JSON.stringify(proof));
});

async function main() {
  const { device } = await acquireWebGpuDevice();

  // ── Create shared buffers ───────────────────────────────────────────────
  const STORAGE_DST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const STORAGE_SRC = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;

  const xBuf  = device.createBuffer({ size: X_BYTES,  usage: STORAGE_DST });
  const wBuf  = device.createBuffer({ size: W_BYTES,  usage: STORAGE_DST });
  const cBuf  = device.createBuffer({ size: OUT_BYTES, usage: STORAGE_DST | STORAGE_SRC }); // matmul output, add input, readback
  const bBuf  = device.createBuffer({ size: OUT_BYTES, usage: STORAGE_DST });
  const yBuf  = device.createBuffer({ size: OUT_BYTES, usage: STORAGE_SRC }); // add output, final readback

  const buffers = new Map();
  buffers.set(0, { buffer: xBuf });
  buffers.set(1, { buffer: wBuf });
  buffers.set(2, { buffer: cBuf });
  buffers.set(3, { buffer: bBuf });
  buffers.set(4, { buffer: yBuf });

  // ── Copy input data to device ───────────────────────────────────────────
  device.queue.writeBuffer(xBuf, 0, INPUT_X);
  device.queue.writeBuffer(wBuf, 0, INPUT_W);
  device.queue.writeBuffer(bBuf, 0, INPUT_B);

  // ── Matmul pipeline + bind group ────────────────────────────────────────
  const matmulModule   = device.createShaderModule({ code: MATMUL_WGSL });
  const matmulBgl      = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });
  const matmulPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [matmulBgl] }),
    compute: { module: matmulModule, entryPoint: "matmul_4x3_3x2" },
  });
  const matmulBg = device.createBindGroup({
    layout: matmulBgl,
    entries: [
      { binding: 0, resource: { buffer: xBuf } },
      { binding: 1, resource: { buffer: wBuf } },
      { binding: 2, resource: { buffer: cBuf } },
    ],
  });

  // ── Add pipeline + bind group ───────────────────────────────────────────
  const addModule   = device.createShaderModule({ code: ADD_WGSL });
  const addBgl      = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });
  const addPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [addBgl] }),
    compute: { module: addModule, entryPoint: "add_bias" },
  });
  const addBg = device.createBindGroup({
    layout: addBgl,
    entries: [
      { binding: 0, resource: { buffer: cBuf } },
      { binding: 1, resource: { buffer: bBuf } },
      { binding: 2, resource: { buffer: yBuf } },
    ],
  });

  // ── Build chain ─────────────────────────────────────────────────────────
  const chain = [
    {
      pipeline: matmulPipeline,
      bindGroups: [{ bindGroupIndex: 0, bindGroup: matmulBg }],
      dispatchWorkgroups: MATMUL_REFLECTION.kernels[0].launch.webgpu_adapter.dispatch_workgroups,
      outputBindings: [{ resourceIndex: 2, bufferByteLen: OUT_BYTES }],
    },
    {
      pipeline: addPipeline,
      bindGroups: [{ bindGroupIndex: 0, bindGroup: addBg }],
      dispatchWorkgroups: ADD_REFLECTION.kernels[0].launch.webgpu_adapter.dispatch_workgroups,
      outputBindings: [{ resourceIndex: 4, bufferByteLen: OUT_BYTES }],
    },
  ];

  const resources = { buffers };
  const { results } = await runKernelChain(device, resources, chain);

  // results[0] = matmul readback (c), results[1] = add readback (y)
  const yValues = results[1].values;

  // ── Compare against stepper reference ───────────────────────────────────
  const failures = [];
  for (let i = 0; i < EXPECTED.length; i++) {
    const diff = Math.abs(yValues[i] - EXPECTED[i]);
    if (diff > EPSILON) {
      const row = Math.floor(i / N);
      const col = i % N;
      failures.push(`[${row},${col}]: expected ${EXPECTED[i]}, got ${yValues[i]} (diff ${diff})`);
    }
  }

  if (failures.length > 0) {
    throw new FaberKernelContractError(
      "readback",
      "matmul+add result mismatch:\n  " + failures.join("\n  "),
      "product",
    );
  }

  window.faberWebGpuProof = Object.freeze({
    ok: true,
    status: "ready",
    kind: "ok",
    entryName: "matmul_4x3_3x2 + add_bias",
    values: yValues,
    expected: EXPECTED,
    dispatchWorkgroups: { x: M, y: N, z: 1 },
  });

  console.log("FABER_MATMUL_PROOF:", JSON.stringify(window.faberWebGpuProof));
}

function proofFailure(error) {
  const kind =
    error instanceof FaberKernelContractError
      ? error.kind
      : typeof error?.kind === "string"
        ? error.kind
        : "product";
  return Object.freeze({
    ok: false,
    status: "error",
    kind,
    path: error?.path ?? null,
    error: error?.message ?? String(error),
  });
}
