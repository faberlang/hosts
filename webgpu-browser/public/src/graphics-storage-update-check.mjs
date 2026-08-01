#!/usr/bin/env node
/**
 * Focused host tests for updateGraphicsStorage.
 *
 * Covers four required cases (plus a fifth optional Float32Array rejection):
 *   1. Successful reflected storage update (by resourceIndex and sourceName)
 *   2. Unknown resource rejection (invalid resourceIndex, unknown sourceName)
 *   3. Non-input resource rejection (output-role entry)
 *   4. Out-of-bounds write rejection (data.byteLength > bufferByteLen)
 *   5. Non-Float32Array payload rejection (before any queue effect)
 *
 * Uses the fake-device stub pattern from placement-contract-oracle.mjs.
 * Runs against the HV-02 static indexed-render reflection fixture shape.
 * All rejections must fire before any device.queue.writeBuffer or
 * device.queue.submit call.
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
  globalThis.GPUShaderStage = { VERTEX: 0x1, FRAGMENT: 0x2, COMPUTE: 0x4 };
}
if (typeof globalThis.GPUMapMode === "undefined") {
  globalThis.GPUMapMode = { READ: 0x0001, WRITE: 0x0002 };
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

// ── Imports ──────────────────────────────────────────────────────────

const { FaberKernelContractError } = await import(
  pathToFileURL(path.join(here, "contract", "artifact-admission.js")).href
);
const { updateGraphicsStorage } = await import(
  pathToFileURL(path.join(here, "backend", "webgpu-runtime.js")).href
);

// ── Fixture constants (from HV-02 generated graphics-reflection.json) ─

const RESOURCE_INDEX_INPUT = 0;
const RESOURCE_INDEX_OUTPUT = 1;
const BUFFER_BYTE_LEN = 256; // 64 f32 elements × 4 bytes

// ── Helpers ──────────────────────────────────────────────────────────

function fail(message) {
  console.error(`graphics-storage-update-check failed: ${message}`);
  process.exit(1);
}

function require(condition, message) {
  if (!condition) fail(message);
}

// ── Fake device (tracks buffer backing and writeBuffer/submit calls) ─

function createFakeDevice() {
  let seq = 0;
  /** @type {Map<number, { backing: ArrayBuffer, mapped: boolean, size: number, usage: number, destroyed: boolean }>} */
  const _buffers = new Map();
  let _writeBufferCount = 0;
  let _submitCount = 0;

  const device = {
    __writeBufferCount: () => _writeBufferCount,
    __submitCount: () => _submitCount,
    __getBufferData(id) {
      return _buffers.get(id)?.backing ?? null;
    },

    queue: {
      writeBuffer(buffer, offset, data) {
        _writeBufferCount += 1;
        const entry = _buffers.get(buffer.__id);
        require(!!entry, `writeBuffer: unknown buffer ${buffer.__id}`);
        require(!entry.destroyed, `writeBuffer: destroyed buffer ${buffer.__id}`);
        const src = data instanceof ArrayBuffer
          ? new Uint8Array(data)
          : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        const dst = new Uint8Array(entry.backing);
        dst.set(src, offset);
      },

      submit(commandBuffers) {
        _submitCount += 1;
      },
    },

    createBuffer(desc) {
      const id = ++seq;
      const size = desc.size;
      const mapped = desc.mappedAtCreation === true;
      const backing = new ArrayBuffer(size);
      _buffers.set(id, { backing, mapped, size, usage: desc.usage, destroyed: false });
      return {
        __id: id,
        size,
        usage: desc.usage,
        getMappedRange() {
          const entry = _buffers.get(id);
          require(!!entry, `getMappedRange: unknown buffer ${id}`);
          require(entry.mapped, `getMappedRange: buffer ${id} not mapped`);
          return entry.backing;
        },
        unmap() {
          const entry = _buffers.get(id);
          require(!!entry, `unmap: unknown buffer ${id}`);
          entry.mapped = false;
        },
        async mapAsync(mode) {
          const entry = _buffers.get(id);
          require(!!entry, `mapAsync: unknown buffer ${id}`);
          require(!entry.destroyed, `mapAsync: destroyed buffer ${id}`);
          entry.mapped = true;
        },
        destroy() {
          const entry = _buffers.get(id);
          require(!!entry, `destroy: unknown buffer ${id}`);
          require(!entry.destroyed, `destroy: double destroy ${id}`);
          entry.destroyed = true;
        },
      };
    },

    createTexture() {
      return { __kind: "texture" };
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

    createRenderPipeline() {
      return { __kind: "render-pipeline" };
    },

    createCommandEncoder() {
      return {
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
        finish() {
          return { __kind: "cmd" };
        },
      };
    },
  };

  return device;
}

// ── Test fixture construction helpers ────────────────────────────────

/**
 * Build a resources object matching createGraphicsResources output shape.
 * storageBuffers is a Map<number, {buffer, generation, logicalId}>.
 */
function buildResources(device) {
  // Create one storage buffer for resourceIndex 0 (input/transform, 256 bytes)
  const buffer = device.createBuffer({
    size: BUFFER_BYTE_LEN,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  // Initialize with zero data
  new Uint8Array(buffer.getMappedRange()).fill(0);
  buffer.unmap();

  const storageBuffers = new Map();
  storageBuffers.set(RESOURCE_INDEX_INPUT, {
    buffer,
    generation: 0,
    logicalId: RESOURCE_INDEX_INPUT,
  });

  return { storageBuffers };
}

/**
 * Build a descriptor with a single bind group containing one input entry.
 * Mirrors the HV-02 graphics-reflection.json fixture shape after admission
 * through loadFaberGraphicsPipeline.
 */
function buildDescriptor(inputRole) {
  const role = inputRole != null ? inputRole : "input";
  return {
    wgsl: "",
    schemaVersion: 1,
    target: "wgsl-text",
    kernels: [],
    pipeline: {},
    pipelineLayout: {},
    bindGroupLayouts: [],
    bindGroups: [
      {
        bindGroupIndex: 0,
        group: 0,
        entryIndexes: [0],
        entries: [
          {
            binding: 0,
            bindingIndex: 0,
            resourceIndex: RESOURCE_INDEX_INPUT,
            kind: "storage-buffer",
            role,
            access: role === "input" ? "read" : "write",
            shaderAccess: role === "input" ? "read" : "read_write",
            shaderVisibility: "vertex",
            bufferType: role === "input" ? "read-only-storage" : "storage",
            elementLayout: "f32",
            elementByteWidth: 4,
            elementCount: 64,
            bufferByteLen: BUFFER_BYTE_LEN,
            bufferByteOffset: 0,
            bindingByteLen: BUFFER_BYTE_LEN,
            minBindingSize: BUFFER_BYTE_LEN,
            hasDynamicOffset: false,
            sourceLocal: null,
            sourceName: "transform",
          },
        ],
      },
    ],
    draw: {},
    inputBindings: [],
    outputBindings: [],
  };
}

// ── Test cases ───────────────────────────────────────────────────────

async function main() {
  // Verify the export exists
  require(typeof updateGraphicsStorage === "function", "updateGraphicsStorage must be a function");

  // ── Test 1: Successful storage update ───────────────────────────────

  {
    // 1a. By resourceIndex
    const device = createFakeDevice();
    const resources = buildResources(device);
    const descriptor = buildDescriptor("input");
    const data = new Float32Array(64); // 256 bytes, matches bufferByteLen

    // Fill with known values
    for (let i = 0; i < 64; i++) data[i] = i * 1.5;

    const result = updateGraphicsStorage(device, resources, descriptor, {
      resourceIndex: RESOURCE_INDEX_INPUT,
      data,
    });

    require(result.status === 0, `T1a: status must be 0, got ${result.status}`);
    require(result.resourceIndex === RESOURCE_INDEX_INPUT, `T1a: resourceIndex mismatch`);
    require(result.generation === 1, `T1a: generation must be 1, got ${result.generation}`);
    require(device.__writeBufferCount() === 1, `T1a: writeBuffer must be called once, got ${device.__writeBufferCount()}`);
    require(device.__submitCount() === 0, `T1a: submit must not be called, got ${device.__submitCount()}`);

    // Verify data was written to backing
    const storageEntry = resources.storageBuffers.get(RESOURCE_INDEX_INPUT);
    const backing = device.__getBufferData(storageEntry.buffer.__id);
    const written = new Float32Array(backing);
    require(written[0] === 0, `T1a: data[0] expected 0, got ${written[0]}`);
    // Note: writeBuffer writes to backing in the fake device, but that's the
    // buffer's backing (newly created, not yet written). The writeBuffer call
    // writes data to it. Let's verify specific values.
    // Actually, the initial fill(0) is overwritten by writeBuffer.
    require(written[10] === 15, `T1a: data[10] expected 15, got ${written[10]}`);
    require(written[63] === 94.5, `T1a: data[63] expected 94.5, got ${written[63]}`);

    // Generation must have advanced on the entry
    require(storageEntry.generation === 1, `T1a: entry generation must be 1, got ${storageEntry.generation}`);

    console.log("T1a PASS: update by resourceIndex");
  }

  {
    // 1b. By sourceName
    const device = createFakeDevice();
    const resources = buildResources(device);
    const descriptor = buildDescriptor("input");
    const data = new Float32Array(32); // 128 bytes, within bounds

    for (let i = 0; i < 32; i++) data[i] = i;

    const result = updateGraphicsStorage(device, resources, descriptor, {
      sourceName: "transform",
      data,
    });

    require(result.status === 0, `T1b: status must be 0, got ${result.status}`);
    require(result.resourceIndex === RESOURCE_INDEX_INPUT, `T1b: resourceIndex mismatch`);
    require(result.generation === 1, `T1b: generation must be 1, got ${result.generation}`);
    require(device.__writeBufferCount() === 1, `T1b: writeBuffer must be called once`);
    require(device.__submitCount() === 0, `T1b: submit must not be called`);

    const storageEntry = resources.storageBuffers.get(RESOURCE_INDEX_INPUT);
    require(storageEntry.generation === 1, `T1b: entry generation must be 1`);

    console.log("T1b PASS: update by sourceName");
  }

  // ── Test 2: Unknown resource rejection ──────────────────────────────

  {
    // 2a. Invalid resourceIndex
    const device = createFakeDevice();
    const resources = buildResources(device);
    const descriptor = buildDescriptor("input");
    const data = new Float32Array(16);

    try {
      updateGraphicsStorage(device, resources, descriptor, {
        resourceIndex: 999,
        data,
      });
      fail("T2a: expected FaberKernelContractError for unknown resourceIndex");
    } catch (error) {
      require(
        error instanceof FaberKernelContractError,
        `T2a: expected FaberKernelContractError, got ${error?.name ?? typeof error}`,
      );
      require(
        device.__writeBufferCount() === 0,
        `T2a: writeBuffer must not be called before rejection, got ${device.__writeBufferCount()}`,
      );
      require(
        device.__submitCount() === 0,
        `T2a: submit must not be called before rejection`,
      );
      console.log("T2a PASS: unknown resourceIndex rejected before queue effect");
    }
  }

  {
    // 2b. Unknown sourceName
    const device = createFakeDevice();
    const resources = buildResources(device);
    const descriptor = buildDescriptor("input");
    const data = new Float32Array(16);

    try {
      updateGraphicsStorage(device, resources, descriptor, {
        sourceName: "nonexistent",
        data,
      });
      fail("T2b: expected FaberKernelContractError for unknown sourceName");
    } catch (error) {
      require(
        error instanceof FaberKernelContractError,
        `T2b: expected FaberKernelContractError, got ${error?.name ?? typeof error}`,
      );
      require(
        device.__writeBufferCount() === 0,
        `T2b: writeBuffer must not be called before rejection`,
      );
      require(
        device.__submitCount() === 0,
        `T2b: submit must not be called before rejection`,
      );
      console.log("T2b PASS: unknown sourceName rejected before queue effect");
    }
  }

  // ── Test 3: Non-input resource rejection ────────────────────────────

  {
    const device = createFakeDevice();
    const resources = buildResources(device);
    // Clone the descriptor and add a second entry with role "output" for
    // resourceIndex 0 (same resource, now declared as output)
    const descriptor = buildDescriptor("output");

    const data = new Float32Array(16);

    try {
      updateGraphicsStorage(device, resources, descriptor, {
        resourceIndex: RESOURCE_INDEX_INPUT,
        data,
      });
      fail("T3: expected FaberKernelContractError for non-input role");
    } catch (error) {
      require(
        error instanceof FaberKernelContractError,
        `T3: expected FaberKernelContractError, got ${error?.name ?? typeof error}`,
      );
      require(
        device.__writeBufferCount() === 0,
        `T3: writeBuffer must not be called before rejection`,
      );
      require(
        device.__submitCount() === 0,
        `T3: submit must not be called before rejection`,
      );
      console.log("T3 PASS: non-input role rejected before queue effect");
    }
  }

  // ── Test 4: Out-of-bounds write rejection ───────────────────────────

  {
    const device = createFakeDevice();
    const resources = buildResources(device);
    const descriptor = buildDescriptor("input");
    // 65 f32 elements = 260 bytes > 256 bufferByteLen
    const data = new Float32Array(65);

    try {
      updateGraphicsStorage(device, resources, descriptor, {
        resourceIndex: RESOURCE_INDEX_INPUT,
        data,
      });
      fail("T4: expected FaberKernelContractError for out-of-bounds write");
    } catch (error) {
      require(
        error instanceof FaberKernelContractError,
        `T4: expected FaberKernelContractError, got ${error?.name ?? typeof error}`,
      );
      require(
        device.__writeBufferCount() === 0,
        `T4: writeBuffer must not be called before rejection`,
      );
      require(
        device.__submitCount() === 0,
        `T4: submit must not be called before rejection`,
      );
      console.log("T4 PASS: out-of-bounds write rejected before queue effect");
    }
  }

  // ── Test 5 (optional): Non-Float32Array payload rejection ───────────

  {
    const device = createFakeDevice();
    const resources = buildResources(device);
    const descriptor = buildDescriptor("input");
    const data = new Uint8Array(16); // wrong type

    try {
      updateGraphicsStorage(device, resources, descriptor, {
        resourceIndex: RESOURCE_INDEX_INPUT,
        data,
      });
      fail("T5: expected FaberKernelContractError for non-Float32Array data");
    } catch (error) {
      require(
        error instanceof FaberKernelContractError,
        `T5: expected FaberKernelContractError, got ${error?.name ?? typeof error}`,
      );
      require(
        device.__writeBufferCount() === 0,
        `T5: writeBuffer must not be called before rejection`,
      );
      require(
        device.__submitCount() === 0,
        `T5: submit must not be called before rejection`,
      );
      console.log("T5 PASS: non-Float32Array rejected before queue effect");
    }
  }

  // ── Summary ─────────────────────────────────────────────────────────

  console.log("");
  console.log("graphics-storage-update-check passed");
  console.log("cases: resourceIndex, sourceName, unknown resource, non-input role, out-of-bounds, non-Float32Array");
  console.log("queue effects: 0 submit calls across all rejection cases");
}

main().catch((error) => {
  fail(error?.stack ?? String(error));
});
