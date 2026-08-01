#!/usr/bin/env node
/**
 * Device-limit clamp gate proof.
 *
 * Node + fake device with limits.  No browser GPU.
 * Covers:
 * - Buffer size exceeding maxBufferSize → rejected (createBuffers,
 *   normalizeComputePayload, createComputeGpuEntry, createStorageBuffers,
 *   createMappedBuffer, createGradientBuffer)
 * - Dispatch count exceeding maxComputeWorkgroupsPerDimension → rejected
 *   (placementDispatch, runKernelChain)
 * - Buffer/dispatch within limits → accepted (green path)
 */

import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

// ── WebGPU enum stubs (Node lacks these) ──────────────────────────────

if (typeof globalThis.GPUBufferUsage === "undefined") {
  globalThis.GPUBufferUsage = {
    MAP_READ:    0x0001,
    MAP_WRITE:   0x0002,
    COPY_SRC:    0x0004,
    COPY_DST:    0x0008,
    INDEX:       0x0010,
    VERTEX:      0x0020,
    UNIFORM:     0x0040,
    STORAGE:     0x0080,
    INDIRECT:    0x0100,
    QUERY_RESOLVE: 0x0200,
  };
}
if (typeof globalThis.GPUQueryType === "undefined") {
  globalThis.GPUQueryType = {};
}
if (typeof globalThis.GPUTextureUsage === "undefined") {
  globalThis.GPUTextureUsage = {
    COPY_SRC: 0x01,
    COPY_DST: 0x02,
    TEXTURE_BINDING: 0x04,
    STORAGE_BINDING: 0x08,
    RENDER_ATTACHMENT: 0x10,
  };
}
if (typeof globalThis.GPUShaderStage === "undefined") {
  globalThis.GPUShaderStage = {
    VERTEX:   0x1,
    FRAGMENT: 0x2,
    COMPUTE:  0x4,
  };
}
if (typeof globalThis.GPUMapMode === "undefined") {
  globalThis.GPUMapMode = { READ: 0x0001, WRITE: 0x0002 };
}

const { FaberKernelContractError } = await import(
  pathToFileURL(path.join(here, "contract", "artifact-admission.js")).href
);
const {
  createWebGpuResources,
  placementDispatch,
  runKernelChain,
  applyComputeResourceReplace,
  createGradientBuffer,
  createChunkGraphicsResources,
} = await import(pathToFileURL(path.join(here, "backend", "webgpu-runtime.js")).href);

// ── Test harness ──────────────────────────────────────────────────────────

function fail(message) {
  console.error(`device-limit-check failed: ${message}`);
  process.exit(1);
}

function require(condition, message) {
  if (!condition) fail(message);
}

async function expectReject(label, expectedKind, run) {
  try {
    await run();
    fail(`${label}: expected FaberKernelContractError kind=${expectedKind}`);
  } catch (error) {
    require(
      error instanceof FaberKernelContractError,
      `${label}: expected FaberKernelContractError, got ${error?.name ?? typeof error}`,
    );
    if (expectedKind !== undefined) {
      require(
        error.kind === expectedKind,
        `${label}: expected kind ${expectedKind}, got ${error.kind}`,
      );
    }
  }
}

// ── Fake WebGPU device with test limits ──────────────────────────────────

const LIMIT_MAX_BUF = 1024;       // maxBufferSize
const LIMIT_MAX_DIM = 128;        // maxComputeWorkgroupsPerDimension

let bufferSeq = 0;

function createFakeDevice() {
  let submitted = 0;
  let completionArmed = false;

  const device = {
    limits: {
      maxBufferSize: LIMIT_MAX_BUF,
      maxComputeWorkgroupsPerDimension: LIMIT_MAX_DIM,
    },

    queue: {
      writeBuffer() {},
      submit(_commands) {
        submitted += 1;
        completionArmed = true;
        return undefined;
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
      const id = ++bufferSeq;
      const size = desc.size;
      let mapped = desc.mappedAtCreation === true;
      const buffer = {
        id, size, usage: desc.usage,
        __faberDestroyed: false,
        getMappedRange() {
          require(mapped, `fake buffer ${id}: getMappedRange without map`);
          return new ArrayBuffer(size);
        },
        unmap() { mapped = false; },
        async mapAsync() { mapped = true; },
        destroy() {
          if (buffer.__faberDestroyed) return;
          buffer.__faberDestroyed = true;
        },
      };
      return buffer;
    },

    createShaderModule()      { return { __kind: "shader" }; },
    createBindGroupLayout()   { return { __kind: "bgl" }; },
    createPipelineLayout()    { return { __kind: "pl" }; },
    createBindGroup()         { return { __kind: "bg" }; },
    createComputePipeline()   { return { __kind: "cp" }; },
    createRenderPipeline()    { return { __kind: "rp" }; },
    createTexture()           { return { __kind: "tex", createView() { return {}; }, format: "bgra8unorm" }; },
    createCommandEncoder() {
      return {
        beginComputePass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups() {},
            end() {},
          };
        },
        beginRenderPass() {
          return {
            setPipeline() {},
            setVertexBuffer() {},
            setIndexBuffer() {},
            setBindGroup() {},
            drawIndexed() {},
            end() {},
          };
        },
        copyBufferToBuffer() {},
        finish() { return { __kind: "cmd" }; },
      };
    },

    __submitted: () => submitted,
  };

  return device;
}

// ── Minimal compute descriptor ───────────────────────────────────────────

function computeDescriptor() {
  return {
    wgsl: "@compute @workgroup_size(1) fn main() {}",
    schemaVersion: 1,
    target: "wgsl-text",
    entryName: "main",
    shaderStage: "compute",
    pipelineLayout: { bindGroupLayoutIndexes: [] },
    bindGroupLayouts: [],
    bindGroups: [],
    dispatchWorkgroups: { x: 1, y: 1, z: 1 },
    inputBindings: [],
    outputBindings: [],
  };
}

// ── Tests ─────────────────────────────────────────────────────────────────

async function main() {
  // ── 1. Buffer size exceeds maxBufferSize → rejected via createBuffers ──
  {
    const device = createFakeDevice();
    const desc = computeDescriptor();
    // One entry with size above limit
    desc.bindGroupLayouts = [{
      bindGroupIndex: 0, group: 0,
      layoutEntryIndexes: [0],
      entries: [{
        binding: 0, bindingIndex: 0, resourceIndex: 0,
        layoutEntryIndex: 0,
        bufferByteLen: LIMIT_MAX_BUF + 1,
        bufferByteOffset: 0, bindingByteLen: LIMIT_MAX_BUF + 1,
        minBindingSize: LIMIT_MAX_BUF + 1,
        visibility: "compute",
        bufferType: "storage",
        hasDynamicOffset: false,
        sourceLocal: null, sourceName: "data",
      }],
    }];
    desc.bindGroups = [{
      bindGroupIndex: 0, group: 0,
      entryIndexes: [0],
      entries: [{
        binding: 0, bindingIndex: 0, resourceIndex: 0,
        kind: "storage-buffer", role: "output",
        access: "write", shaderAccess: "read_write",
        shaderVisibility: "compute",
        elementLayout: "f32", elementByteWidth: 4,
        elementCount: (LIMIT_MAX_BUF + 1) / 4,
        bufferType: "storage",
        bufferByteLen: LIMIT_MAX_BUF + 1,
        bufferByteOffset: 0, bindingByteLen: LIMIT_MAX_BUF + 1,
        minBindingSize: LIMIT_MAX_BUF + 1,
        hasDynamicOffset: false,
        sourceLocal: null, sourceName: null,
      }],
    }];
    desc.outputBindings = desc.bindGroups[0].entries.filter((e) => e.role === "output");

    await expectReject("createBuffers oversized buffer", "webgpu", () => {
      createWebGpuResources(device, desc);
    });
  }

  // ── 2. Buffer size within limit → accepted via createBuffers ───────────
  {
    const device = createFakeDevice();
    const desc = computeDescriptor();
    const safeSize = 64;
    desc.bindGroupLayouts = [{
      bindGroupIndex: 0, group: 0,
      layoutEntryIndexes: [0],
      entries: [{
        binding: 0, bindingIndex: 0, resourceIndex: 0,
        layoutEntryIndex: 0,
        bufferByteLen: safeSize,
        bufferByteOffset: 0, bindingByteLen: safeSize,
        minBindingSize: safeSize,
        visibility: "compute",
        bufferType: "read-only-storage",
        hasDynamicOffset: false,
        sourceLocal: null, sourceName: "input",
      }, {
        binding: 1, bindingIndex: 1, resourceIndex: 1,
        layoutEntryIndex: 1,
        bufferByteLen: safeSize,
        bufferByteOffset: 0, bindingByteLen: safeSize,
        minBindingSize: safeSize,
        visibility: "compute",
        bufferType: "storage",
        hasDynamicOffset: false,
        sourceLocal: null, sourceName: null,
      }],
    }];
    desc.bindGroups = [{
      bindGroupIndex: 0, group: 0,
      entryIndexes: [0, 1],
      entries: [{
        binding: 0, bindingIndex: 0, resourceIndex: 0,
        kind: "storage-buffer", role: "input",
        access: "read", shaderAccess: "read",
        shaderVisibility: "compute",
        elementLayout: "f32", elementByteWidth: 4,
        elementCount: safeSize / 4,
        bufferType: "read-only-storage",
        bufferByteLen: safeSize,
        bufferByteOffset: 0, bindingByteLen: safeSize,
        minBindingSize: safeSize,
        hasDynamicOffset: false,
        sourceLocal: null, sourceName: "input",
      }, {
        binding: 1, bindingIndex: 1, resourceIndex: 1,
        kind: "storage-buffer", role: "output",
        access: "write", shaderAccess: "read_write",
        shaderVisibility: "compute",
        elementLayout: "f32", elementByteWidth: 4,
        elementCount: safeSize / 4,
        bufferType: "storage",
        bufferByteLen: safeSize,
        bufferByteOffset: 0, bindingByteLen: safeSize,
        minBindingSize: safeSize,
        hasDynamicOffset: false,
        sourceLocal: null, sourceName: null,
      }],
    }];
    desc.inputBindings = desc.bindGroups[0].entries.filter((e) => e.role === "input");
    desc.outputBindings = desc.bindGroups[0].entries.filter((e) => e.role === "output");

    const resources = createWebGpuResources(device, desc, { input: new Float32Array(safeSize / 4) });
    require(resources.buffers.size === 2, `in-limit createBuffers: expected 2 buffers, got ${resources.buffers.size}`);
  }

  // ── 3. Dispatch count exceeds maxComputeWorkgroupsPerDimension → rejected ──
  {
    const device = createFakeDevice();
    const desc = computeDescriptor();
    const resources = createWebGpuResources(device, desc);

    desc.dispatchWorkgroups = { x: LIMIT_MAX_DIM + 1, y: 1, z: 1 };
    await expectReject("placementDispatch oversized x", "webgpu", () => {
      placementDispatch(device, resources, desc);
    });

    desc.dispatchWorkgroups = { x: 1, y: LIMIT_MAX_DIM + 1, z: 1 };
    await expectReject("placementDispatch oversized y", "webgpu", () => {
      placementDispatch(device, resources, desc);
    });

    desc.dispatchWorkgroups = { x: 1, y: 1, z: LIMIT_MAX_DIM + 1 };
    await expectReject("placementDispatch oversized z", "webgpu", () => {
      placementDispatch(device, resources, desc);
    });
  }

  // ── 4. Dispatch count within limit → accepted via placementDispatch ─────
  {
    const device = createFakeDevice();
    const desc = computeDescriptor();
    const resources = createWebGpuResources(device, desc);

    desc.dispatchWorkgroups = { x: LIMIT_MAX_DIM, y: LIMIT_MAX_DIM, z: LIMIT_MAX_DIM };
    // Should not throw
    placementDispatch(device, resources, desc);
  }

  // ── 5. Dispatch count exceeds limit → rejected via runKernelChain ────────
  {
    const device = createFakeDevice();
    const desc = computeDescriptor();
    const resources = createWebGpuResources(device, desc);

    const chain = [
      {
        pipeline: device.createComputePipeline(),
        bindGroups: [],
        dispatchWorkgroups: { x: LIMIT_MAX_DIM + 1, y: 1, z: 1 },
        outputBindings: [],
      },
    ];
    await expectReject("runKernelChain oversized dispatch", "webgpu", () => {
      return runKernelChain(device, resources, chain);
    });
  }

  // ── 6. Dispatch within limit → accepted via runKernelChain ──────────────
  {
    const device = createFakeDevice();
    const desc = computeDescriptor();
    const resources = createWebGpuResources(device, desc);

    const chain = [
      {
        pipeline: device.createComputePipeline(),
        bindGroups: [],
        dispatchWorkgroups: { x: 1, y: 1, z: 1 },
        outputBindings: [],
      },
    ];
    // Should not throw
    await runKernelChain(device, resources, chain);
  }

  // ── 7. applyComputeResourceReplace oversized buffer → rejected ─────────
  {
    const device = createFakeDevice();
    const resources = createWebGpuResources(device, computeDescriptor());

    // Create an in-limit buffer first
    const created = applyComputeResourceReplace(device, resources, {
      resource_index: 0,
      generation: 0,
      buffer_descriptor: { size: 64, usage: GPUBufferUsage.STORAGE },
    });
    require(created.kind === "created", "baseline create ok");

    // Oversized replace
    await expectReject("compute replace oversized", "webgpu", () => {
      applyComputeResourceReplace(device, resources, {
        resource_index: 0,
        generation: 1,
        buffer_descriptor: { size: LIMIT_MAX_BUF + 1, usage: GPUBufferUsage.STORAGE },
      });
    });

    // Oversized create (new resource)
    await expectReject("compute create oversized", "webgpu", () => {
      applyComputeResourceReplace(device, resources, {
        resource_index: 99,
        generation: 0,
        buffer_descriptor: { size: LIMIT_MAX_BUF + 1, usage: GPUBufferUsage.STORAGE },
      });
    });
  }

  // ── 8. createGradientBuffer oversized → rejected ───────────────────────
  {
    const device = createFakeDevice();
    const oversizedElements = Math.floor(LIMIT_MAX_BUF / 4) + 1;
    await expectReject("createGradientBuffer oversized", "webgpu", () => {
      createGradientBuffer(device, oversizedElements);
    });
  }

  // ── 9. createGradientBuffer within limit → accepted ────────────────────
  {
    const device = createFakeDevice();
    // Should not throw
    const handle = createGradientBuffer(device, 4);
    require(typeof handle === "number", "gradient handle is a number");
    require(handle >= 0, "gradient handle non-negative");
  }

  // ── 10. createChunkGraphicsResources with oversized storage → rejected ──
  {
    const device = createFakeDevice();
    const oversized = LIMIT_MAX_BUF + 1;

    const gfxDesc = {
      wgsl: "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(0); } @fragment fn fs() -> @location(0) vec4f { return vec4f(0); }",
      schemaVersion: 1,
      target: "wgsl-text",
      kernels: [
        {
          entryName: "vs",
          shaderStage: "vertex",
          vertexInputs: [{
            sourceName: "pos", location: 0, format: "float32x3",
            stepMode: "vertex", offsetBytes: 0, strideBytes: 12,
          }],
          vertexBufferLayouts: [{
            bufferIndex: 0, arrayStride: 12, stepMode: "vertex",
            attributes: [{ shaderLocation: 0, format: "float32x3", offset: 0, sourceName: "pos" }],
          }],
        },
        { entryName: "fs", shaderStage: "fragment" },
      ],
      pipeline: {
        colorTargetFormats: ["bgra8unorm"],
        primitiveTopology: "triangle-list",
        vertexCount: 3,
        depthStencil: { depthWriteEnabled: false, depthCompare: "always" },
      },
      pipelineLayout: { bindGroupLayoutIndexes: [0] },
      bindGroupLayouts: [{
        bindGroupIndex: 0, group: 0,
        layoutEntryIndexes: [0],
        entries: [{
          binding: 0, bindingIndex: 0, resourceIndex: 0,
          layoutEntryIndex: 0,
          bufferByteLen: oversized,
          bufferByteOffset: 0, bindingByteLen: oversized,
          minBindingSize: oversized,
          visibility: "vertex",
          bufferType: "read-only-storage",
          hasDynamicOffset: false,
          sourceLocal: null, sourceName: "data",
        }],
      }],
      bindGroups: [{
        bindGroupIndex: 0, group: 0,
        entryIndexes: [0],
        entries: [{
          binding: 0, bindingIndex: 0, resourceIndex: 0,
          kind: "storage-buffer", role: "input",
          access: "read", shaderAccess: "read",
          shaderVisibility: "vertex",
          elementLayout: "f32", elementByteWidth: 4,
          elementCount: Math.floor(oversized / 4),
          bufferType: "read-only-storage",
          bufferByteLen: oversized,
          bufferByteOffset: 0, bindingByteLen: oversized,
          minBindingSize: oversized,
          hasDynamicOffset: false,
          sourceLocal: null, sourceName: "data",
        }],
      }],
      draw: {
        indexFormat: "uint16",
        instanceCount: 1, baseVertex: 0, firstIndex: 0, indexCount: 3,
      },
    };

    const canvasContext = {
      getCurrentTexture() {
        return {
          format: "bgra8unorm",
          createView() { return {}; },
          width: 1, height: 1,
        };
      },
      configure() {},
    };

    await expectReject("createStorageBuffers oversized (graphics)", "webgpu", () => {
      createChunkGraphicsResources(device, gfxDesc, {}, canvasContext);
    });
  }

  console.log("device-limit-check passed");
  console.log(`limits: maxBufferSize=${LIMIT_MAX_BUF}, maxComputeWorkgroupsPerDimension=${LIMIT_MAX_DIM}`);
  console.log("covered:");
  console.log("  createBuffers:  oversize reject + in-limit accept");
  console.log("  placementDispatch: oversize reject (x/y/z) + in-limit accept");
  console.log("  runKernelChain: oversize reject + in-limit accept");
  console.log("  applyComputeResourceReplace: oversize reject (create + replace)");
  console.log("  createGradientBuffer: oversize reject + in-limit accept");
  console.log("  createStorageBuffers (graphics): oversize reject");
}

main().catch((error) => {
  fail(error?.stack ?? String(error));
});
