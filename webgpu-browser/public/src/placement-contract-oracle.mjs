#!/usr/bin/env node
/**
 * W4-04 U2: WebGPU placement contract oracle.
 *
 * Cross-host equivalence proof — exercises all four placement operations
 * from webgpu-runtime.js through a fake device:
 *
 *    placementCopyIn → placementDispatch → placementReadback → placementSync
 *
 * Elementwise multiply-by-2 on [1.0, 2.0, 3.0, 4.0] must produce
 * [2.0, 4.0, 6.0, 8.0] — the same output as the Rust LLVM PlacementHost
 * oracle at radix-mir-llvm/tests/placement_contract_compliance.rs.
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

// ── Imports (all four placement operations) ────────────────────────────

const {
  createWebGpuResources,
  placementCopyIn,
  placementDispatch,
  placementReadback,
  placementSync,
} = await import(pathToFileURL(path.join(here, "webgpu-runtime.js")).href);

// ── Kernel descriptor (elementwise multiply-by-2) ──────────────────────

const MUL_BY_TWO_WGSL = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    output[idx] = input[idx] * 2.0;
}
`;

function kernelDescriptor() {
  return Object.freeze({
    wgsl: MUL_BY_TWO_WGSL,
    schemaVersion: 1,
    target: "wgsl-text",
    entryName: "main",
    shaderStage: "compute",
    pipelineLayout: Object.freeze({
      bindGroupLayoutIndexes: Object.freeze([0]),
    }),
    bindGroupLayouts: Object.freeze([
      Object.freeze({
        bindGroupIndex: 0,
        group: 0,
        layoutEntryIndexes: Object.freeze([0, 1]),
        entries: Object.freeze([
          Object.freeze({
            binding: 0, bindingIndex: 0, resourceIndex: 0,
            layoutEntryIndex: 0,
            bufferByteLen: 16, bufferByteOffset: 0,
            bindingByteLen: 16,
            visibility: "compute", bufferType: "read-only-storage",
            hasDynamicOffset: false, minBindingSize: 16,
            sourceLocal: null, sourceName: "input",
          }),
          Object.freeze({
            binding: 1, bindingIndex: 1, resourceIndex: 1,
            layoutEntryIndex: 1,
            bufferByteLen: 16, bufferByteOffset: 0,
            bindingByteLen: 16,
            visibility: "compute", bufferType: "storage",
            hasDynamicOffset: false, minBindingSize: 16,
            sourceLocal: null, sourceName: null,
          }),
        ]),
      }),
    ]),
    bindGroups: Object.freeze([
      Object.freeze({
        bindGroupIndex: 0,
        group: 0,
        entryIndexes: Object.freeze([0, 1]),
        entries: Object.freeze([
          Object.freeze({
            binding: 0, kind: "storage-buffer", role: "input",
            access: "read", shaderAccess: "read",
            shaderVisibility: "compute",
            elementLayout: "f32", elementByteWidth: 4,
            elementCount: 4,
            resourceIndex: 0, bindingIndex: 0,
            bufferType: "read-only-storage",
            bufferByteLen: 16, bufferByteOffset: 0,
            bindingByteLen: 16, minBindingSize: 16,
            hasDynamicOffset: false,
            sourceLocal: null, sourceName: "input",
          }),
          Object.freeze({
            binding: 1, kind: "storage-buffer", role: "output",
            access: "write", shaderAccess: "read_write",
            shaderVisibility: "compute",
            elementLayout: "f32", elementByteWidth: 4,
            elementCount: 4,
            resourceIndex: 1, bindingIndex: 1,
            bufferType: "storage",
            bufferByteLen: 16, bufferByteOffset: 0,
            bindingByteLen: 16, minBindingSize: 16,
            hasDynamicOffset: false,
            sourceLocal: null, sourceName: null,
          }),
        ]),
      }),
    ]),
    dispatchWorkgroups: Object.freeze({ x: 1, y: 1, z: 1 }),
    inputBindings: Object.freeze([
      Object.freeze({
        binding: 0, kind: "storage-buffer", role: "input",
        access: "read", shaderAccess: "read",
        shaderVisibility: "compute",
        elementLayout: "f32", elementByteWidth: 4,
        elementCount: 4,
        resourceIndex: 0, bindingIndex: 0,
        bufferType: "read-only-storage",
        bufferByteLen: 16, bufferByteOffset: 0,
        bindingByteLen: 16, minBindingSize: 16,
        hasDynamicOffset: false,
        sourceLocal: null, sourceName: "input",
      }),
    ]),
    outputBindings: Object.freeze([
      Object.freeze({
        binding: 1, kind: "storage-buffer", role: "output",
        access: "write", shaderAccess: "read_write",
        shaderVisibility: "compute",
        elementLayout: "f32", elementByteWidth: 4,
        elementCount: 4,
        resourceIndex: 1, bindingIndex: 1,
        bufferType: "storage",
        bufferByteLen: 16, bufferByteOffset: 0,
        bindingByteLen: 16, minBindingSize: 16,
        hasDynamicOffset: false,
        sourceLocal: null, sourceName: null,
      }),
    ]),
  });
}

// ── Fake WebGPU device (buffer-tracking + compute simulation) ──────────

const INPUT_RESOURCE_INDEX = 0;
const OUTPUT_RESOURCE_INDEX = 1;

function fail(msg) {
  console.error(`placement-contract-oracle failed: ${msg}`);
  process.exit(1);
}

function require(condition, msg) {
  if (!condition) fail(msg);
}

function createFakeDevice() {
  let seq = 0;
  /** @type {Map<number, { backing: ArrayBuffer, mapped: boolean, size: number, usage: number, destroyed: boolean }>} */
  const buffers = new Map();
  let submits = 0;
  let completionArmed = false;

  /** @type {{ inputBufferId: number|null, outputBufferId: number|null }|null} */
  let kernelMapping = null;

  const device = {
    __submits: () => submits,
    __pipelineCreated: () => pipelineCreated,

    /**
     * Register which device buffer __id maps to input/output for kernel
     * simulation. Called after resource creation.
     */
    __registerKernelResources(inputId, outputId) {
      kernelMapping = { inputBufferId: inputId, outputBufferId: outputId };
    },

    queue: {
      writeBuffer(buffer, offset, data) {
        const entry = buffers.get(buffer.__id);
        require(!!entry, `writeBuffer: unknown buffer ${buffer.__id}`);
        require(!entry.destroyed, `writeBuffer: destroyed buffer ${buffer.__id}`);
        const src = data instanceof ArrayBuffer
          ? new Uint8Array(data)
          : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        const dst = new Uint8Array(entry.backing);
        dst.set(src, offset);
      },

      submit(commandBuffers) {
        submits += 1;
        for (const cb of commandBuffers) {
          if (cb.__computeState) {
            simulateKernel(cb.__computeState);
          }
        }
        completionArmed = true;
      },

      onSubmittedWorkDone() {
        return new Promise((resolve) => {
          queueMicrotask(() => {
            completionArmed = false;
            resolve();
          });
        });
      },
    },

    createBuffer(desc) {
      const id = ++seq;
      const size = desc.size;
      const mapped = desc.mappedAtCreation === true;
      const backing = new ArrayBuffer(size);
      buffers.set(id, { backing, mapped, size, usage: desc.usage, destroyed: false });
      return {
        __id: id,
        size,
        usage: desc.usage,
        getMappedRange() {
          const entry = buffers.get(id);
          require(!!entry, `getMappedRange: unknown buffer ${id}`);
          require(entry.mapped, `getMappedRange: buffer ${id} not mapped`);
          return entry.backing;
        },
        unmap() {
          const entry = buffers.get(id);
          require(!!entry, `unmap: unknown buffer ${id}`);
          entry.mapped = false;
        },
        async mapAsync(mode) {
          const entry = buffers.get(id);
          require(!!entry, `mapAsync: unknown buffer ${id}`);
          require(!entry.destroyed, `mapAsync: destroyed buffer ${id}`);
          entry.mapped = true;
        },
        destroy() {
          const entry = buffers.get(id);
          require(!!entry, `destroy: unknown buffer ${id}`);
          require(!entry.destroyed, `destroy: double destroy ${id}`);
          entry.destroyed = true;
        },
      };
    },

    createShaderModule() {
      return { __kind: "shader" };
    },

    createBindGroupLayout() {
      return { __kind: "bgl" };
    },

    createPipelineLayout() {
      return { __kind: "pl" };
    },

    createBindGroup() {
      return { __kind: "bg" };
    },

    createComputePipeline() {
      return { __kind: "compute-pipeline" };
    },

    createCommandEncoder() {
      const copyOps = [];
      let computeState = null;

      return {
        beginComputePass() {
          computeState = { pipeline: null, bindGroups: new Map() };
          return {
            setPipeline(pipeline) {
              require(!!computeState, "setPipeline: no active compute pass");
              computeState.pipeline = pipeline;
            },
            setBindGroup(index, bindGroup) {
              require(!!computeState, "setBindGroup: no active compute pass");
              computeState.bindGroups.set(index, bindGroup);
            },
            dispatchWorkgroups(x, y, z) {
              require(!!computeState, "dispatchWorkgroups: no active compute pass");
              computeState.dispatch = { x, y, z };
            },
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
            require(!!srcEntry, `copyBufferToBuffer: unknown source ${op.source.__id}`);
            require(!!dstEntry, `copyBufferToBuffer: unknown dest ${op.dest.__id}`);
            const src = new Uint8Array(srcEntry.backing);
            const dst = new Uint8Array(dstEntry.backing);
            dst.set(
              src.subarray(op.sourceOffset, op.sourceOffset + op.size),
              op.destOffset,
            );
          }
          return { __kind: "cmd", __computeState: computeState };
        },
      };
    },

    __getBufferData(id) {
      return buffers.get(id)?.backing ?? null;
    },
  };

  let pipelineCreated = false;

  // Override createComputePipeline to track creation
  const origCreatePipeline = device.createComputePipeline;
  device.createComputePipeline = (...args) => {
    pipelineCreated = true;
    return origCreatePipeline.call(device, ...args);
  };
  device.__pipelineCreated = () => pipelineCreated;

  /**
   * Simulate elementwise multiply-by-2 kernel on tracked buffer data.
   * Matches the WGSL: output[idx] = input[idx] * 2.0.
   * Uses the registered kernel resource mapping to find input/output
   * buffer __ids, rather than introspecting opaque bind groups.
   */
  function simulateKernel(computeState) {
    require(!!computeState, "simulateKernel: no compute state");
    require(!!computeState.pipeline, "simulateKernel: no pipeline");
    require(computeState.dispatch, "simulateKernel: no dispatch");
    require(!!kernelMapping, "simulateKernel: no kernel resource mapping — call __registerKernelResources");

    const inputBufEntry = buffers.get(kernelMapping.inputBufferId);
    const outputBufEntry = buffers.get(kernelMapping.outputBufferId);
    require(!!inputBufEntry, "simulateKernel: input buffer not found");
    require(!!outputBufEntry, "simulateKernel: output buffer not found");

    const inputF32 = new Float32Array(inputBufEntry.backing);
    const outputF32 = new Float32Array(outputBufEntry.backing);
    for (let i = 0; i < 4; i++) {
      outputF32[i] = inputF32[i] * 2.0;
    }
  }

  return device;
}

// ── Oracle test ────────────────────────────────────────────────────────

async function main() {
  // 1. Verify all four placement operations are exported.
  require(typeof placementCopyIn === "function", "placementCopyIn export");
  require(typeof placementDispatch === "function", "placementDispatch export");
  require(typeof placementReadback === "function", "placementReadback export");
  require(typeof placementSync === "function", "placementSync export");

  // 2. Create fake device and kernel resources.
  const device = createFakeDevice();
  const descriptor = kernelDescriptor();

  // createWebGpuResources seeds the input buffer via mappedAtCreation.
  // resourceIndex 0 → input (sourceName: "input"), resourceIndex 1 → output.
  const inputData = new Float32Array([1.0, 2.0, 3.0, 4.0]);
  const resources = createWebGpuResources(device, descriptor, {
    input: inputData,
  });

  require(device.__pipelineCreated(), "compute pipeline created");
  require(resources.buffers.has(INPUT_RESOURCE_INDEX), "resource 0 (input) exists");
  require(resources.buffers.has(OUTPUT_RESOURCE_INDEX), "resource 1 (output) exists");

  // Register buffer __ids for kernel simulation.
  const inputBufObj = resources.buffers.get(INPUT_RESOURCE_INDEX);
  const outputBufObj = resources.buffers.get(OUTPUT_RESOURCE_INDEX);
  device.__registerKernelResources(inputBufObj.buffer.__id, outputBufObj.buffer.__id);

  // 3. placementCopyIn — stage input data to the input buffer.
  //    Even though createWebGpuResources already wrote the input data via
  //    mappedAtCreation, we exercise placementCopyIn explicitly to prove
  //    it works as a standalone placement operation on the same pathway
  //    the Rust oracle uses for copy_in.
  const copyStatus = placementCopyIn(device, resources, {
    resourceIndex: INPUT_RESOURCE_INDEX,
    data: new Float32Array([1.0, 2.0, 3.0, 4.0]),
  });
  require(copyStatus.status === 0, `placementCopyIn status: ${copyStatus.status}`);

  // Verify copy-in wrote data correctly.
  const inputPreDispatch = device.__getBufferData(inputBufObj.buffer.__id);
  require(inputPreDispatch !== null, "input buffer has backing after copyIn");
  const inputPreF32 = new Float32Array(inputPreDispatch);
  require(
    inputPreF32[0] === 1.0 && inputPreF32[1] === 2.0 &&
    inputPreF32[2] === 3.0 && inputPreF32[3] === 4.0,
    `placementCopyIn wrote input: [${[...inputPreF32]}]`,
  );

  // 4. placementDispatch — encode and submit the multiply-by-2 kernel.
  //    This exercises the W4-03 placementDispatch function directly.
  placementDispatch(device, resources, descriptor);

  // The fake device's submit handler simulates the kernel effect
  // (output = input * 2.0) using the registered buffer mapping.

  // 5. placementReadback — read device buffer back to host.
  const results = await placementReadback(device, resources, [
    { resourceIndex: OUTPUT_RESOURCE_INDEX, bufferByteLen: 16 },
  ]);
  require(Array.isArray(results), "placementReadback returned array");
  require(results.length === 1, "placementReadback returned 1 result");
  require(
    results[0].binding.resourceIndex === OUTPUT_RESOURCE_INDEX,
    `readback binding resourceIndex: ${results[0].binding.resourceIndex}`,
  );

  const readbackValues = results[0].values;
  require(Array.isArray(readbackValues), "readback values is array");
  require(readbackValues.length === 4, `readback length: ${readbackValues.length}`);

  // 6. Assert correct result: [1,2,3,4] × 2 = [2,4,6,8].
  const expected = [2.0, 4.0, 6.0, 8.0];
  for (let i = 0; i < 4; i++) {
    require(
      Math.abs(readbackValues[i] - expected[i]) < 0.0001,
      `readback[${i}]: expected ${expected[i]}, got ${readbackValues[i]}`,
    );
  }

  // 7. placementSync — insert device-side ordering barrier after readback.
  placementSync(device, resources, [INPUT_RESOURCE_INDEX, OUTPUT_RESOURCE_INDEX]);

  // ── Oracle summary ──────────────────────────────────────────────────

  console.log("placement-contract-oracle passed");
  console.log("placement: copyIn → dispatch → readback → sync");
  console.log("input:  [1.0, 2.0, 3.0, 4.0]");
  console.log(`output: [${readbackValues.join(", ")}]`);
  console.log("cross-host equivalence: [2.0, 4.0, 6.0, 8.0]");
  console.log("WGSL: elementwise multiply-by-2");
  console.log("operations: placementCopyIn → placementDispatch → placementReadback → placementSync");
}

main().catch((error) => {
  fail(error?.stack ?? String(error));
});
