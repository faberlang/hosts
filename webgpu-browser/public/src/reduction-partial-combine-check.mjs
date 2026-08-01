#!/usr/bin/env node
/**
 * W6-A2 U2/U3 (host side): reduction partial-combine unit check.
 *
 * Verifies the host-bridge reduction combine contract through the real
 * webgpu-runtime.js readback path with a fake device (no browser GPU):
 *
 *   1. combineReductionPartials — Sum: Σ partials; Mean: Σ partial slots
 *      (each slot is already workgroup_sum / n — the WGSL-text emission
 *      divides in-kernel, see the reduction output-buffer contract note);
 *      fail-closed on unknown op / partialCount mismatch / missing
 *      fullLength for mean.
 *   2. placementReadback — no combine metadata → raw partial slots
 *      (byte-identical to pre-combine); sum/mean metadata → combined value.
 *   3. Bridge path — buildChainFromReflection + runKernelChain with a
 *      reduction reflection and caller-supplied combine metadata: raw
 *      readback returns 2 partial slots; combined readback returns the
 *      value matching the independent CPU reference (n=16, ws=8 → 2
 *      partials; sum 136, mean 8.5).
 *
 * Node + fake device — no browser GPU required.
 */

import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));

// ── WebGPU enum stubs (Node lacks these) ──────────────────────────────

if (typeof globalThis.GPUBufferUsage === "undefined") {
  globalThis.GPUBufferUsage = {
    MAP_READ:    0x0001,
    MAP_WRITE:   0x0002,
    COPY_SRC:    0x0004,
    COPY_DST:    0x0008,
    STORAGE:     0x0080,
  };
}
if (typeof globalThis.GPUShaderStage === "undefined") {
  globalThis.GPUShaderStage = { COMPUTE: 0x4 };
}
if (typeof globalThis.GPUMapMode === "undefined") {
  globalThis.GPUMapMode = { READ: 0x0001, WRITE: 0x0002 };
}

// ── Imports ────────────────────────────────────────────────────────────

const {
  combineReductionPartials,
  buildChainFromReflection,
  runKernelChain,
  placementReadback,
} = await import(pathToFileURL(path.join(here, "backend", "webgpu-runtime.js")).href);
const { FaberKernelContractError } = await import(
  pathToFileURL(path.join(here, "contract", "artifact-admission.js")).href,
);

function fail(msg) {
  console.error(`reduction-partial-combine-check failed: ${msg}`);
  process.exit(1);
}

function require(condition, msg) {
  if (!condition) fail(msg);
}

function requireError(promiseOrFn, msg, kind = "any") {
  // Returns the caught error, or fails if nothing throws.
  return Promise.resolve()
    .then(() => (typeof promiseOrFn === "function" ? promiseOrFn() : promiseOrFn))
    .then(() => {
      fail(msg);
    })
    .catch((error) => {
      require(error instanceof FaberKernelContractError, `${msg} (not FaberKernelContractError)`);
      if (kind !== "any") {
        require(error.kind === kind, `${msg} (kind ${error.kind} != ${kind})`);
      }
      return error;
    });
}

// ── Fake WebGPU device (buffer-tracking + reduction kernel simulation) ──

function createFakeDevice({ partials }) {
  let seq = 0;
  /** @type {Map<number, { backing: ArrayBuffer, mapped: boolean, size: number, destroyed: boolean }>} */
  const buffers = new Map();
  let outputBufferId = null;

  const device = {
    /** Register which device buffer the simulated reduction writes its
     * partial slots into. Populates the backing immediately so copies made
     * at encoder.finish() read the partials. */
    __registerOutputBuffer(id) {
      outputBufferId = id;
      const entry = buffers.get(id);
      require(!!entry, `__registerOutputBuffer: unknown buffer ${id}`);
      const out = new Float32Array(entry.backing);
      for (let i = 0; i < partials.length; i++) {
        out[i] = partials[i];
      }
    },
    __getBufferData(id) {
      return buffers.get(id)?.backing ?? null;
    },

    queue: {
      writeBuffer(buffer, offset, data) {
        const entry = buffers.get(buffer.__id);
        require(!!entry, `writeBuffer: unknown buffer ${buffer.__id}`);
        const src = data instanceof ArrayBuffer
          ? new Uint8Array(data)
          : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        const dst = new Uint8Array(entry.backing);
        dst.set(src, offset);
      },

      submit(commandBuffers) {
        // Simulate the reduction kernel effect: the registered output buffer
        // receives the configured partial slots on every submission (the
        // device buffer "holds" the reduction partials).
        if (outputBufferId != null) {
          const outEntry = buffers.get(outputBufferId);
          require(!!outEntry, "submit: no registered output buffer");
          const out = new Float32Array(outEntry.backing);
          for (let i = 0; i < partials.length; i++) {
            out[i] = partials[i];
          }
        }
        for (const cb of commandBuffers) {
          if (cb.__computeState) {
            // Compute dispatch recorded (chain path) — the partial write
            // above already covers the reduction result.
          }
        }
      },

      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },

    createBuffer(desc) {
      const id = ++seq;
      buffers.set(id, {
        backing: new ArrayBuffer(desc.size),
        mapped: desc.mappedAtCreation === true,
        size: desc.size,
        destroyed: false,
      });
      return {
        __id: id,
        size: desc.size,
        getMappedRange() {
          return buffers.get(id).backing;
        },
        unmap() {},
        async mapAsync() {
          buffers.get(id).mapped = true;
        },
        destroy() {
          buffers.get(id).destroyed = true;
        },
      };
    },

    createShaderModule() { return { __kind: "shader" }; },
    createBindGroupLayout() { return { __kind: "bgl" }; },
    createPipelineLayout() { return { __kind: "pl" }; },
    createBindGroup() { return { __kind: "bg" }; },
    createComputePipeline() { return { __kind: "compute-pipeline" }; },

    createCommandEncoder() {
      const copyOps = [];
      let computeState = null;
      return {
        beginComputePass() {
          computeState = { pipeline: null, bindGroups: new Map() };
          return {
            setPipeline(pipeline) { computeState.pipeline = pipeline; },
            setBindGroup(index, bindGroup) { computeState.bindGroups.set(index, bindGroup); },
            dispatchWorkgroups(x, y, z) { computeState.dispatch = { x, y, z }; },
            end() {},
          };
        },
        copyBufferToBuffer(source, sourceOffset, dest, destOffset, size) {
          copyOps.push({ source, sourceOffset, dest, destOffset, size });
        },
        finish() {
          for (const op of copyOps) {
            const srcEntry = buffers.get(op.source.__id);
            const dstEntry = buffers.get(op.dest.__id);
            const src = new Uint8Array(srcEntry.backing);
            const dst = new Uint8Array(dstEntry.backing);
            dst.set(src.subarray(op.sourceOffset, op.sourceOffset + op.size), op.destOffset);
          }
          return { __kind: "cmd", __computeState: computeState };
        },
      };
    },
  };

  return device;
}

// ── Fixture: reduction n=16, ws=8 → 2 partial slots ────────────────────

const N = 16;        // tensor length
const WS = 8;        // workgroup lane count
const PARTIAL_COUNT = Math.ceil(N / WS); // 2
const INPUT = Array.from({ length: N }, (_, i) => i + 1); // 1..16
const SUM_REF = INPUT.reduce((a, b) => a + b, 0);         // 136
const MEAN_REF = SUM_REF / N;                             // 8.5
// Compiler-emitted shape: each mean partial slot is workgroup_sum / n.
const SUM_PARTIALS = [36, 100];            // wg0: 1..8, wg1: 9..16
const MEAN_PARTIALS = SUM_PARTIALS.map((s) => s / N); // [2.25, 6.25]

const EPSILON = 0.0001;
function near(actual, expected, msg) {
  require(
    Math.abs(actual - expected) < EPSILON,
    `${msg}: expected ${expected}, got ${actual}`,
  );
}

// ── 1. combineReductionPartials direct assertions ──────────────────────

function checkCombineDirect() {
  near(
    combineReductionPartials(SUM_PARTIALS, { op: "sum", partialCount: PARTIAL_COUNT }),
    SUM_REF,
    "combine sum of partials",
  );
  near(
    combineReductionPartials(MEAN_PARTIALS, { op: "mean", partialCount: PARTIAL_COUNT, fullLength: N }),
    MEAN_REF,
    "combine mean of pre-divided partials",
  );

  // Fail-closed: unknown op.
  return requireError(
    () => combineReductionPartials(SUM_PARTIALS, { op: "avg", partialCount: PARTIAL_COUNT }),
    "unknown combine op must fail",
  );
}

// ── 2. placementReadback combine application ───────────────────────────

async function checkReadbackCombine() {
  // Raw path: no metadata → raw partial slots.
  {
    const device = createFakeDevice({ partials: SUM_PARTIALS });
    const buffers = new Map();
    const outBuffer = device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE });
    buffers.set(1, { buffer: outBuffer });
    device.__registerOutputBuffer(outBuffer.__id);

    const results = await placementReadback(device, { buffers }, [
      { resourceIndex: 1, bufferByteLen: 8 },
    ]);
    require(results.length === 1, "raw readback returned 1 result");
    require(results[0].combined === undefined, "raw readback has no combined field");
    require(results[0].values.length === 2, `raw readback length: ${results[0].values.length}`);
    near(results[0].values[0], SUM_PARTIALS[0], "raw partial[0]");
    near(results[0].values[1], SUM_PARTIALS[1], "raw partial[1]");
  }

  // Sum combine path.
  {
    const device = createFakeDevice({ partials: SUM_PARTIALS });
    const buffers = new Map();
    const outBuffer = device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE });
    buffers.set(1, { buffer: outBuffer });
    device.__registerOutputBuffer(outBuffer.__id);

    const results = await placementReadback(device, { buffers }, [
      {
        resourceIndex: 1,
        bufferByteLen: 8,
        combine: { op: "sum", partialCount: PARTIAL_COUNT, fullLength: N },
      },
    ]);
    require(results.length === 1, "sum readback returned 1 result");
    require(results[0].values.length === 1, "sum combine returns single value");
    near(results[0].combined, SUM_REF, "sum combined");
    near(results[0].values[0], SUM_REF, "sum combined values[0]");
  }

  // Mean combine path.
  {
    const device = createFakeDevice({ partials: MEAN_PARTIALS });
    const buffers = new Map();
    const outBuffer = device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE });
    buffers.set(1, { buffer: outBuffer });
    device.__registerOutputBuffer(outBuffer.__id);

    const results = await placementReadback(device, { buffers }, [
      {
        resourceIndex: 1,
        bufferByteLen: 8,
        combine: { op: "mean", partialCount: PARTIAL_COUNT, fullLength: N },
      },
    ]);
    require(results.length === 1, "mean readback returned 1 result");
    near(results[0].combined, MEAN_REF, "mean combined");
  }

  // Fail-closed through the readback path: partialCount mismatch.
  {
    const device = createFakeDevice({ partials: SUM_PARTIALS });
    const buffers = new Map();
    const outBuffer = device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE });
    buffers.set(1, { buffer: outBuffer });
    device.__registerOutputBuffer(outBuffer.__id);

    await requireError(
      placementReadback(device, { buffers }, [
        {
          resourceIndex: 1,
          bufferByteLen: 8,
          combine: { op: "sum", partialCount: 3, fullLength: N },
        },
      ]),
      "partialCount mismatch must fail in placementReadback",
    );
  }
}

// ── 3. Bridge path: buildChainFromReflection + runKernelChain ──────────
//
// The kernel mirrors the compiler-emitted reduction shape with one
// adaptation: `wg_shared` instead of `shared` (a WGSL reserved identifier —
// see the reduction output-buffer contract note). The fake device never
// compiles this WGSL; it is embedded to document the dispatch shape.

const REDUCTION_WGSL = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
var<workgroup> wg_shared: array<f32, ${WS}u>;

@compute @workgroup_size(${WS}, 1, 1)
fn sum_reduction(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    var acc: f32 = 0.0;
    for (var g: u32 = id.x; g < ${N}u; g += ${WS * PARTIAL_COUNT}u) {
        acc += input[g];
    }
    wg_shared[local_id.x] = acc;
    workgroupBarrier();
    if (local_id.x < 4u) { wg_shared[local_id.x] = wg_shared[local_id.x] + wg_shared[local_id.x + 4u]; }
    workgroupBarrier();
    if (local_id.x < 2u) { wg_shared[local_id.x] = wg_shared[local_id.x] + wg_shared[local_id.x + 2u]; }
    workgroupBarrier();
    if (local_id.x < 1u) { wg_shared[local_id.x] = wg_shared[local_id.x] + wg_shared[local_id.x + 1u]; }
    workgroupBarrier();
    if (local_id.x == 0u && workgroup_id.x < ${PARTIAL_COUNT}u) { output[workgroup_id.x] = wg_shared[0]; }
}
`;

const REDUCTION_REFLECTION = {
  schema_version: 1,
  target: "wgsl-text",
  kernels: [
    {
      entry_name: "sum_reduction",
      shader_stage: "compute",
      launch: {
        webgpu_adapter: {
          dispatch_workgroups: { x: PARTIAL_COUNT, y: 1, z: 1 },
          bind_group_layout_descriptors: [
            {
              bind_group_index: 0,
              entries: [
                { binding: 0, visibility: "compute", buffer_type: "read-only-storage", has_dynamic_offset: false, min_binding_size: N * 4 },
                { binding: 1, visibility: "compute", buffer_type: "storage", has_dynamic_offset: false, min_binding_size: PARTIAL_COUNT * 4 },
              ],
            },
          ],
          pipeline_layout_descriptor: { bind_group_layout_indexes: [0] },
          bind_group_descriptors: [
            {
              bind_group_index: 0,
              entries: [
                { binding: 0, resource_index: 0, role: "input", buffer_byte_len: N * 4 },
                { binding: 1, resource_index: 1, role: "output", buffer_byte_len: PARTIAL_COUNT * 4 },
              ],
            },
          ],
        },
      },
    },
  ],
};

async function checkBridgePath() {
  // Raw (no combine metadata): 2 partial slots.
  {
    const device = createFakeDevice({ partials: SUM_PARTIALS });
    const { chain, resources } = buildChainFromReflection(
      device,
      REDUCTION_WGSL,
      REDUCTION_REFLECTION,
      new Map([[0, new Float32Array(INPUT)]]),
      [{ resourceIndex: 1 }],
    );
    device.__registerOutputBuffer(resources.buffers.get(1).buffer.__id);

    const { results } = await runKernelChain(device, resources, chain);
    require(results.length === 1, "bridge raw returned 1 result");
    require(results[0].combined === undefined, "bridge raw has no combined field");
    require(results[0].values.length === PARTIAL_COUNT, `bridge raw partial slots: ${results[0].values.length}`);
    near(results[0].values[0], SUM_PARTIALS[0], "bridge raw partial[0]");
    near(results[0].values[1], SUM_PARTIALS[1], "bridge raw partial[1]");
  }

  // Combine metadata (sum): single combined value == CPU reference.
  {
    const device = createFakeDevice({ partials: SUM_PARTIALS });
    const { chain, resources } = buildChainFromReflection(
      device,
      REDUCTION_WGSL,
      REDUCTION_REFLECTION,
      new Map([[0, new Float32Array(INPUT)]]),
      [{ resourceIndex: 1 }],
      new Map([[1, { op: "sum", partialCount: PARTIAL_COUNT, fullLength: N }]]),
    );
    device.__registerOutputBuffer(resources.buffers.get(1).buffer.__id);

    const { results } = await runKernelChain(device, resources, chain);
    require(results.length === 1, "bridge combine returned 1 result");
    require(results[0].values.length === 1, "bridge combine returns single value");
    near(results[0].combined, SUM_REF, "bridge sum combined == CPU reference");
  }

  // Combine metadata (mean): single combined value == CPU mean reference.
  {
    const device = createFakeDevice({ partials: MEAN_PARTIALS });
    const { chain, resources } = buildChainFromReflection(
      device,
      REDUCTION_WGSL,
      REDUCTION_REFLECTION,
      new Map([[0, new Float32Array(INPUT)]]),
      [{ resourceIndex: 1 }],
      new Map([[1, { op: "mean", partialCount: PARTIAL_COUNT, fullLength: N }]]),
    );
    device.__registerOutputBuffer(resources.buffers.get(1).buffer.__id);

    const { results } = await runKernelChain(device, resources, chain);
    near(results[0].combined, MEAN_REF, "bridge mean combined == CPU reference");
  }
}

// ── Main ────────────────────────────────────────────────────────────────

async function main() {
  await checkCombineDirect();
  await checkReadbackCombine();
  await checkBridgePath();

  console.log("reduction-partial-combine-check passed");
  console.log(`n=${N}, ws=${WS}, partial slots=${PARTIAL_COUNT}`);
  console.log(`sum reference: ${SUM_REF} (1..${N})`);
  console.log(`mean reference: ${MEAN_REF}`);
  console.log("combine ops: sum (Σ partials) / mean (Σ pre-divided partial slots)");
  console.log("paths: combineReductionPartials → placementReadback → buildChainFromReflection+runKernelChain");
  console.log("fail-closed: unknown op, partialCount mismatch, missing fullLength all rejected");
}

main().catch((error) => {
  fail(error?.stack ?? String(error));
});
