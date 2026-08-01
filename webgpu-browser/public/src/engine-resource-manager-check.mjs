#!/usr/bin/env node
/**
 * Engine resource-manager gate proof (DS-S2 Phase 2, item B).
 *
 * Logical handles → GPU residency with generation checks under the
 * create-before-retire contract (extends the compute/chunk lifecycle
 * harnesses; same semantics as applyComputeResourceReplace /
 * destroyRetiredComputeResources in the backend).
 *
 * Covers:
 * - create-before-retire: a retire never destroys live buffers out from under
 *   a handle; replacement allocates the new generation before retiring the old.
 * - generation-mismatch rejection: a stale handle (unknown index, wrong
 *   generation, retired slot) is a typed FaberKernelContractError — never a
 *   silent skip.
 * - deferred destruction through queue completion.
 * - counters (created / live / retired / destroyed) for the oracle harnesses.
 */

import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

if (typeof globalThis.GPUBufferUsage === "undefined") {
  globalThis.GPUBufferUsage = {
    MAP_READ: 0x0001, MAP_WRITE: 0x0002, COPY_SRC: 0x0004, COPY_DST: 0x0008,
    INDEX: 0x0010, VERTEX: 0x0020, UNIFORM: 0x0040, STORAGE: 0x0080,
    INDIRECT: 0x0100, QUERY_RESOLVE: 0x0200,
  };
}

const { FaberKernelContractError } = await import(
  pathToFileURL(path.join(here, "contract", "artifact-admission.js")).href
);
const { MeshResourceManager } = await import(
  pathToFileURL(path.join(here, "engine", "resource-manager.js")).href
);

function fail(message) {
  console.error(`engine-resource-manager-check failed: ${message}`);
  process.exit(1);
}

function require(condition, message) {
  if (!condition) fail(message);
}

async function expectReject(label, run) {
  try {
    run();
    fail(`${label}: expected FaberKernelContractError rejection`);
  } catch (error) {
    require(
      error instanceof FaberKernelContractError,
      `${label}: expected FaberKernelContractError, got ${error?.name ?? typeof error}: ${error?.message}`,
    );
  }
}

// ── Fake WebGPU device with tracked buffer lifecycle ─────────────────────

function createFakeDevice() {
  let seq = 0;
  const buffers = new Map();
  let _submitted = 0;

  const device = {
    __destroyedCount: () => {
      let count = 0;
      for (const b of buffers.values()) if (b.destroyed) count += 1;
      return count;
    },
    __submitted: () => _submitted,
    queue: {
      writeBuffer() {},
      submit() {
        _submitted += 1;
      },
      onSubmittedWorkDone() {
        return new Promise((resolve) => queueMicrotask(resolve));
      },
    },
    createBuffer(desc) {
      const id = ++seq;
      const backing = new ArrayBuffer(desc.size);
      const mapped = desc.mappedAtCreation === true;
      buffers.set(id, { backing, mapped, destroyed: false, id });
      return {
        __id: id,
        size: desc.size,
        getMappedRange() {
          const entry = buffers.get(id);
          require(!entry.destroyed, `fake device: getMappedRange on destroyed buffer ${id}`);
          require(entry.mapped, `fake device: getMappedRange without map ${id}`);
          return entry.backing;
        },
        unmap() {
          const entry = buffers.get(id);
          entry.mapped = false;
        },
        destroy() {
          const entry = buffers.get(id);
          if (entry.destroyed) return;
          entry.destroyed = true;
        },
      };
    },
  };
  return device;
}

function meshFixture(name, vertexCount) {
  const vertices = new Float32Array(vertexCount * 9);
  for (let i = 0; i < vertices.length; i++) vertices[i] = i * 0.25;
  return { name, role: "static", vertices, indices: new Uint32Array(vertexCount * 3) };
}

async function main() {
  // ── 1. create → acquire (baseline residency) ──────────────────────────
  {
    const device = createFakeDevice();
    const manager = new MeshResourceManager();
    const handle = manager.create(device, meshFixture("terrain", 4));
    require(handle.index === 0 && handle.generation === 1, "first handle index 0 gen 1");
    const entry = manager.acquire(handle);
    require(entry.indexCount === 12, "acquire returns the live GPU entry");
    const snap = manager.snapshot();
    require(snap.created === 1 && snap.live === 1 && snap.retired === 0, "counters after create");
    console.log("T1 PASS: create → acquire (logical handle → residency)");
  }

  // ── 2. Generation-mismatch rejections (stale handles) ─────────────────
  {
    const device = createFakeDevice();
    const manager = new MeshResourceManager();
    const handle = manager.create(device, meshFixture("water", 2));

    await expectReject("acquire unknown index", () => manager.acquire({ index: 99, generation: 1 }));
    await expectReject("acquire wrong generation", () => manager.acquire({ index: 0, generation: 0 }));
    await expectReject("acquire missing handle", () => manager.acquire(null));
    console.log("T2 PASS: stale-generation acquire rejected");
  }

  // ── 3. retire validation (create-before-retire) ───────────────────────
  {
    const device = createFakeDevice();
    const manager = new MeshResourceManager();
    const handle = manager.create(device, meshFixture("a", 2));

    await expectReject("retire wrong generation", () => manager.retire({ index: 0, generation: 9 }));
    await expectReject("retire unknown index", () => manager.retire({ index: 7, generation: 1 }));

    manager.retire(handle);
    const snap = manager.snapshot();
    require(snap.live === 0 && snap.retired === 1 && snap.pending_retire_groups === 1, "retire enqueues deferred destruction");
    await expectReject("acquire after retire", () => manager.acquire(handle));
    await expectReject("double retire", () => manager.retire(handle));
    console.log("T3 PASS: retire validation — stale/unknown/double rejected");
  }

  // ── 4. replace: create-before-retire, generation bumps ────────────────
  {
    const device = createFakeDevice();
    const manager = new MeshResourceManager();
    const first = manager.create(device, meshFixture("mesh", 2));
    const second = manager.replace(device, first, meshFixture("mesh", 3));

    require(second.index === 1 && second.generation === 1, "replace allocates a fresh logical id");
    require(manager.acquire(second).indexCount === 9, "new generation is live");
    await expectReject("old generation no longer live", () => manager.acquire(first));

    const snap = manager.snapshot();
    require(snap.created === 2 && snap.live === 1 && snap.retired === 1, "replace retires the old generation");
    // create-before-retire: the old buffers must still exist (only enqueued),
    // not destroyed at replace time.
    require(
      device.__destroyedCount() === 0,
      "create-before-retire: nothing destroyed at replace time",
    );
    console.log("T4 PASS: replace — new generation live, old generation retired (not destroyed)");
  }

  // ── 5. destroyRetired after queue completion ──────────────────────────
  {
    const device = createFakeDevice();
    const manager = new MeshResourceManager();
    const first = manager.create(device, meshFixture("mesh", 2));
    manager.replace(device, first, meshFixture("mesh", 3));
    require(device.__destroyedCount() === 0, "nothing destroyed before destroyRetired");

    const result = await manager.destroyRetired(device);
    require(result.destroyed_groups === 1, "one retired group destroyed");
    require(result.destroyed_buffers === 2, "VB + IB destroyed for the retired group");
    require(device.__destroyedCount() === 2, "fake device agrees on destroyed buffers");

    const snap = manager.snapshot();
    require(snap.destroyed === 2 && snap.pending_retire_groups === 0, "counters after destroyRetired");

    // Idempotent when nothing is pending.
    const again = await manager.destroyRetired(device);
    require(again.destroyed_groups === 0, "second destroyRetired is a no-op");
    console.log("T5 PASS: deferred destruction after queue completion");
  }

  // ── 6. trackExisting (greybox renderer path) ──────────────────────────
  {
    const device = createFakeDevice();
    const manager = new MeshResourceManager();
    const entry = { name: "terrain", role: "static", vertexBuffer: device.createBuffer({ size: 64, usage: GPUBufferUsage.VERTEX }), indexBuffer: device.createBuffer({ size: 24, usage: GPUBufferUsage.INDEX }), indexCount: 6 };
    const handle = manager.trackExisting(entry);
    require(manager.acquire(handle) === entry, "trackExisting registers the existing entry");
    await expectReject("trackExisting requires buffers", () => manager.trackExisting({ name: "x", indexCount: 1 }));
    console.log("T6 PASS: trackExisting registers already-created entries (no re-upload)");
  }

  console.log("");
  console.log("engine-resource-manager-check passed");
  console.log("covered: create/acquire residency, generation-mismatch rejection, retire validation,");
  console.log("         create-before-retire replace, deferred destruction, counters, trackExisting");
}

main().catch((error) => {
  fail(error?.stack ?? String(error));
});
