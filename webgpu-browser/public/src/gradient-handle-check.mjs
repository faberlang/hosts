#!/usr/bin/env node
/**
 * G-A-09 U3: gradient handle behavioral integration test.
 *
 * Node + fake device/queue — no browser GPU.
 * Covers:
 * - createGradientBuffer → opaque handle
 * - accumulateGradient ×2 → host-side elementwise accumulation
 * - readGradient → verify accumulated values
 * - zeroGradient → reset to zeros
 * - shape mismatch rejection
 * - unknown handle rejection
 */

import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

// Node has no WebGPU enums — install minimal constants the runtime reads.
if (typeof globalThis.GPUBufferUsage === "undefined") {
  globalThis.GPUBufferUsage = {
    MAP_READ: 0x0001,
    MAP_WRITE: 0x0002,
    COPY_SRC: 0x0004,
    COPY_DST: 0x0008,
    INDEX: 0x0010,
    VERTEX: 0x0020,
    UNIFORM: 0x0040,
    STORAGE: 0x0080,
    INDIRECT: 0x0100,
    QUERY_RESOLVE: 0x0200,
  };
}
if (typeof globalThis.GPUMapMode === "undefined") {
  globalThis.GPUMapMode = { READ: 0x0001, WRITE: 0x0002 };
}

const { FaberKernelContractError } = await import(
  pathToFileURL(path.join(here, "contract", "artifact-admission.js")).href
);
const {
  createGradientBuffer,
  accumulateGradient,
  readGradient,
  zeroGradient,
} = await import(pathToFileURL(path.join(here, "backend", "webgpu-runtime.js")).href);

function fail(message) {
  console.error(`gradient-handle-check failed: ${message}`);
  process.exit(1);
}

function require(condition, message) {
  if (!condition) {
    fail(message);
  }
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
    require(
      error.kind === expectedKind,
      `${label}: expected kind ${expectedKind}, got ${error.kind}`,
    );
  }
}

// ── Fake WebGPU device ───────────────────────────────────────────────────
// Gradient ops need: createBuffer → buffer with mapAsync/getMappedRange/unmap.
// No queue, shader, pipeline, or encoder required — accumulation is host-side.

function createFakeDevice() {
  const device = {
    createBuffer(desc) {
      const size = desc.size;
      let mapped = desc.mappedAtCreation === true;
      const backing = new ArrayBuffer(size);
      const buffer = {
        size,
        usage: desc.usage,
        getMappedRange() {
          require(mapped, "getMappedRange called on unmapped buffer");
          return backing;
        },
        unmap() {
          mapped = false;
        },
        mapAsync(_mode) {
          return new Promise((resolve) => {
            queueMicrotask(() => {
              mapped = true;
              resolve();
            });
          });
        },
        destroy() {},
      };
      return buffer;
    },
  };
  return device;
}

async function main() {
  require(typeof createGradientBuffer === "function", "export createGradientBuffer");
  require(typeof accumulateGradient === "function", "export accumulateGradient");
  require(typeof readGradient === "function", "export readGradient");
  require(typeof zeroGradient === "function", "export zeroGradient");

  // ── Step 1: Create gradient buffer for 4-element f32 tensor ──────────
  const device = createFakeDevice();
  const handle = createGradientBuffer(device, 4);
  require(typeof handle === "number" && handle >= 0, `handle must be a non-negative number, got ${handle}`);

  // ── Step 2: Accumulate first gradient ────────────────────────────────
  await accumulateGradient(device, handle, new Float32Array([1.0, 2.0, 3.0, 4.0]));

  // ── Step 3: Read back and verify ─────────────────────────────────────
  let result = await readGradient(device, handle);
  require(result instanceof Float32Array, "readGradient returns Float32Array");
  require(result.length === 4, `expected 4 elements, got ${result.length}`);
  require(
    result[0] === 1.0 && result[1] === 2.0 && result[2] === 3.0 && result[3] === 4.0,
    `Step 3 failed: expected [1,2,3,4], got [${result}]`,
  );

  // ── Step 4: Accumulate second gradient ───────────────────────────────
  await accumulateGradient(device, handle, new Float32Array([0.5, 0.5, 0.5, 0.5]));

  // ── Step 5: Read back and verify accumulated values ──────────────────
  result = await readGradient(device, handle);
  require(
    result[0] === 1.5 && result[1] === 2.5 && result[2] === 3.5 && result[3] === 4.5,
    `Step 5 failed: expected [1.5,2.5,3.5,4.5], got [${result}]`,
  );

  // ── Step 6: Zero gradient ────────────────────────────────────────────
  await zeroGradient(device, handle);

  // ── Step 7: Read back and verify zeros ───────────────────────────────
  result = await readGradient(device, handle);
  require(
    result[0] === 0.0 && result[1] === 0.0 && result[2] === 0.0 && result[3] === 0.0,
    `Step 7 failed: expected [0,0,0,0], got [${result}]`,
  );

  // ── Step 8: Shape mismatch rejection ─────────────────────────────────
  await expectReject("shape mismatch", "reflection", () =>
    accumulateGradient(device, handle, new Float32Array([1.0])),
  );

  // ── Step 9: Unknown handle rejection ─────────────────────────────────
  await expectReject("unknown handle", "reflection", () =>
    accumulateGradient(device, 999, new Float32Array([1.0])),
  );

  console.log("gradient-handle-check passed");
  console.log("covered: create, accumulate ×2, read, zero, shape-mismatch rejection, unknown-handle rejection");
}

main().catch((error) => {
  fail(error?.stack ?? String(error));
});
