#!/usr/bin/env node
/**
 * G-SPINE-07 S3: compute resource lifecycle proof.
 *
 * Node + fake device/queue — no browser GPU.
 * Covers:
 * - create / replace (create-before-retire) / remove transitions
 * - counter correctness (created, live, retired, destroyed)
 * - invalid transition rejection (no live resource, generation mismatch, stale generation)
 * - queue-completion gate (destroy only after onSubmittedWorkDone)
 * - missing API detection
 * - concurrent retire during completion
 * - onSubmittedWorkDone rejection re-queues pendingRetire (no orphan)
 * - unaffected resource identity across neighbor replace
 * - empty pending destroy no-op
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
    VERTEX: 0x1,
    FRAGMENT: 0x2,
    COMPUTE: 0x4,
  };
}
if (typeof globalThis.GPUMapMode === "undefined") {
  globalThis.GPUMapMode = { READ: 0x0001, WRITE: 0x0002 };
}

const { FaberKernelContractError } = await import(
  pathToFileURL(path.join(here, "faber-kernel.js")).href
);
const {
  createWebGpuResources,
  applyComputeResourceReplace,
  destroyRetiredComputeResources,
  computeResourceCounters,
} = await import(pathToFileURL(path.join(here, "webgpu-runtime.js")).href);

function fail(message) {
  console.error(`compute-resource-lifecycle-check failed: ${message}`);
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

// ── Minimal compute descriptor ───────────────────────────────────────────

function computeDescriptor() {
  return {
    wgsl: "@compute @workgroup_size(1) fn main() {}",
    entryName: "main",
    bindGroupLayouts: [],
    pipelineLayout: { bindGroupLayoutIndexes: [] },
    bindGroups: [],
    outputBindings: [],
  };
}

/** Build a buffer descriptor accepted by applyComputeResourceReplace. */
function storageBuf(size = 256, usage = GPUBufferUsage.STORAGE) {
  return { size, usage };
}

/** A different buffer shape for replace scenarios (proves new allocation). */
function storageBufAlt(size = 512, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC) {
  return { size, usage };
}

// ── Fake WebGPU device ───────────────────────────────────────────────────

let bufferSeq = 0;

function createFakeDevice() {
  const destroyed = [];
  const created = [];
  let submitted = 0;
  let completionArmed = false;

  const queue = {
    submit(_commands) {
      submitted += 1;
      completionArmed = true;
      return undefined;
    },
    onSubmittedWorkDone() {
      return new Promise((resolve) => {
        // Resolve asynchronously so tests observe pre-destroy counters.
        queueMicrotask(() => {
          completionArmed = false;
          resolve();
        });
      });
    },
  };

  const device = {
    queue,
    createBuffer(desc) {
      const id = ++bufferSeq;
      const size = desc.size;
      let mapped = desc.mappedAtCreation === true;
      const buffer = {
        id,
        size,
        usage: desc.usage,
        __faberDestroyed: false,
        getMappedRange() {
          require(mapped, `fake buffer ${id}: getMappedRange without map`);
          return new ArrayBuffer(size);
        },
        unmap() {
          mapped = false;
        },
        destroy() {
          require(!buffer.__faberDestroyed, `fake buffer ${id}: double destroy`);
          buffer.__faberDestroyed = true;
          destroyed.push(id);
        },
      };
      created.push(buffer);
      return buffer;
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
      return { __kind: "cp" };
    },
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
        finish() {
          return { __kind: "cmd" };
        },
      };
    },
    __created: created,
    __destroyed: destroyed,
    __submitted: () => submitted,
  };

  return device;
}

async function main() {
  require(typeof createWebGpuResources === "function", "export createWebGpuResources");
  require(typeof applyComputeResourceReplace === "function", "export applyComputeResourceReplace");
  require(typeof destroyRetiredComputeResources === "function", "export destroyRetiredComputeResources");
  require(typeof computeResourceCounters === "function", "export computeResourceCounters");

  // ── 1. Create ─────────────────────────────────────────────────────────
  {
    const device = createFakeDevice();
    const resources = createWebGpuResources(device, computeDescriptor());

    require(resources.path === "compute", "path must be compute");
    require(resources.buffers instanceof Map, "buffers is a Map");
    require(Array.isArray(resources.pendingRetire), "pendingRetire is array");
    require(resources.buffers.size === 0, "buffers empty before any create");

    const created = applyComputeResourceReplace(device, resources, {
      resource_index: 0,
      generation: 0,
      buffer_descriptor: storageBuf(256),
    });
    require(created.kind === "created", "create kind");
    require(created.resource_index === 0, "create resource_index");
    require(created.generation === 0, "create generation");

    const entry = resources.buffers.get(0);
    require(entry != null, "entry present after create");
    require(entry.logicalId === 0, "entry logicalId");
    require(entry.generation === 0, "entry generation");
    require(entry.buffer != null, "entry has buffer");
    require(Array.isArray(entry.buffers) && entry.buffers.length === 1, "entry buffers array");
    require(entry.buffers[0] === entry.buffer, "buffers[0] is the buffer");
    require(!entry.buffer.__faberDestroyed, "buffer not destroyed");

    const counters = computeResourceCounters(resources);
    require(counters.created === 1, `created: expected 1 got ${counters.created}`);
    require(counters.live === 1, `live: expected 1 got ${counters.live}`);
    require(counters.retired === 0, "retired still 0");
    require(counters.destroyed === 0, "destroyed still 0");
    require(counters.pending_retire_groups === 0, "no pending retire");
    require(counters.path === "compute", "counters.path");
  }

  // ── 2. Replace (create-before-retire) ─────────────────────────────────
  {
    const device = createFakeDevice();
    const resources = createWebGpuResources(device, computeDescriptor());

    applyComputeResourceReplace(device, resources, {
      resource_index: 3,
      generation: 0,
      buffer_descriptor: storageBuf(256),
    });
    const firstBuffer = resources.buffers.get(3).buffer;
    require(!firstBuffer.__faberDestroyed, "first buffer not destroyed");

    const replaced = applyComputeResourceReplace(device, resources, {
      resource_index: 3,
      generation: 1,
      buffer_descriptor: storageBufAlt(512),
    });
    require(replaced.kind === "replaced", "replace kind");
    require(replaced.previous_generation === 0, "previous gen 0");
    require(replaced.generation === 1, "new gen 1");
    require(replaced.resource_index === 3, "same resource_index");

    const mid = computeResourceCounters(resources);
    require(mid.created === 2, `created after replace: expected 2 got ${mid.created}`);
    require(mid.live === 1, `live after replace: expected 1 got ${mid.live}`);
    require(mid.retired === 1, `retired after replace: expected 1 got ${mid.retired}`);
    require(mid.destroyed === 0, "destroyed still 0 before queue done");
    require(mid.pending_retire_groups === 1, "one pending retire group");

    require(!firstBuffer.__faberDestroyed, "old buffer not destroyed before work done");
    require(resources.buffers.get(3).generation === 1, "live gen is 1");
    require(resources.pendingRetire.length === 1, "one group in pendingRetire");

    // Complete queue then destroy.
    await device.queue.onSubmittedWorkDone();
    const destroyResult = await destroyRetiredComputeResources(device, resources);
    require(destroyResult.destroyed_groups === 1, "destroyed one group");
    require(destroyResult.destroyed_buffers === 1, "destroyed one buffer");
    require(firstBuffer.__faberDestroyed, "old buffer destroyed after work done");

    const after = computeResourceCounters(resources);
    require(after.destroyed === 1, `destroyed counter: expected 1 got ${after.destroyed}`);
    require(after.live === 1, "live remains 1");
    require(after.retired === 1, "retired cumulative 1");
    require(after.pending_retire_groups === 0, "pending cleared");
    require(resources.pendingRetire.length === 0, "pending array empty");
  }

  // ── 3. Remove (retire without create) ─────────────────────────────────
  {
    const device = createFakeDevice();
    const resources = createWebGpuResources(device, computeDescriptor());

    applyComputeResourceReplace(device, resources, {
      resource_index: 5,
      generation: 0,
      buffer_descriptor: storageBuf(128),
    });
    const old = resources.buffers.get(5).buffer;

    const removed = applyComputeResourceReplace(device, resources, {
      resource_index: 5,
      generation: 0,
      buffer_descriptor: null,
    });
    require(removed.kind === "removed", "remove kind");
    require(removed.resource_index === 5, "remove resource_index");
    require(removed.previous_generation === 0, "remove previous gen");
    require(resources.buffers.has(5) === false, "entry removed from live map");

    const mid = computeResourceCounters(resources);
    require(mid.live === 0, "live 0 after remove");
    require(mid.retired === 1, "retired 1 after remove");
    require(mid.destroyed === 0, "not destroyed yet");
    require(!old.__faberDestroyed, "buffer survives until queue done");

    await device.queue.onSubmittedWorkDone();
    await destroyRetiredComputeResources(device, resources);
    require(old.__faberDestroyed, "removed buffer destroyed after work done");
    require(computeResourceCounters(resources).destroyed === 1, "destroyed 1");
  }

  // ── 4. Remove undefined buffer_descriptor is also a remove ────────────
  {
    const device = createFakeDevice();
    const resources = createWebGpuResources(device, computeDescriptor());

    applyComputeResourceReplace(device, resources, {
      resource_index: 0,
      generation: 0,
      buffer_descriptor: storageBuf(64),
    });
    const removed = applyComputeResourceReplace(device, resources, {
      resource_index: 0,
      generation: 0,
      buffer_descriptor: undefined,
    });
    require(removed.kind === "removed", "undefined buffer_descriptor is remove");
  }

  // ── 5. Remove rejection: no live resource ─────────────────────────────
  {
    const device = createFakeDevice();
    const resources = createWebGpuResources(device, computeDescriptor());

    await expectReject("remove missing resource", "product", () => {
      applyComputeResourceReplace(device, resources, {
        resource_index: 99,
        generation: 0,
        buffer_descriptor: null,
      });
    });
  }

  // ── 6. Remove rejection: generation mismatch ──────────────────────────
  {
    const device = createFakeDevice();
    const resources = createWebGpuResources(device, computeDescriptor());

    applyComputeResourceReplace(device, resources, {
      resource_index: 1,
      generation: 0,
      buffer_descriptor: storageBuf(256),
    });

    await expectReject("remove generation mismatch", "product", () => {
      applyComputeResourceReplace(device, resources, {
        resource_index: 1,
        generation: 99,
        buffer_descriptor: null,
      });
    });
  }

  // ── 7. Stale generation rejection ─────────────────────────────────────
  {
    const device = createFakeDevice();
    const resources = createWebGpuResources(device, computeDescriptor());

    applyComputeResourceReplace(device, resources, {
      resource_index: 2,
      generation: 5,
      buffer_descriptor: storageBuf(256),
    });

    // Same generation → stale
    await expectReject("same generation replace", "product", () => {
      applyComputeResourceReplace(device, resources, {
        resource_index: 2,
        generation: 5,
        buffer_descriptor: storageBufAlt(512),
      });
    });

    // Lower generation → stale
    await expectReject("lower generation replace", "product", () => {
      applyComputeResourceReplace(device, resources, {
        resource_index: 2,
        generation: 3,
        buffer_descriptor: storageBufAlt(512),
      });
    });
  }

  // ── 7b. Invalid resource_index rejection ──────────────────────────────
  {
    const device = createFakeDevice();
    const resources = createWebGpuResources(device, computeDescriptor());

    await expectReject("negative resource_index", "product", () => {
      applyComputeResourceReplace(device, resources, {
        resource_index: -1,
        generation: 0,
        buffer_descriptor: storageBuf(256),
      });
    });

    await expectReject("missing resource_index", "product", () => {
      applyComputeResourceReplace(device, resources, {
        generation: 0,
        buffer_descriptor: storageBuf(256),
      });
    });
  }

  // ── 7c. Invalid descriptor rejection ──────────────────────────────────
  {
    const device = createFakeDevice();
    const resources = createWebGpuResources(device, computeDescriptor());

    await expectReject("zero size", "product", () => {
      applyComputeResourceReplace(device, resources, {
        resource_index: 0,
        generation: 0,
        buffer_descriptor: { size: 0, usage: GPUBufferUsage.STORAGE },
      });
    });

    await expectReject("zero usage", "product", () => {
      applyComputeResourceReplace(device, resources, {
        resource_index: 0,
        generation: 0,
        buffer_descriptor: { size: 256, usage: 0 },
      });
    });

    await expectReject("non-object descriptor", "product", () => {
      applyComputeResourceReplace(device, resources, {
        resource_index: 0,
        generation: 0,
        buffer_descriptor: 42,
      });
    });
  }

  // ── 8. Non-compute resources path rejection ───────────────────────────
  {
    const device = createFakeDevice();

    await expectReject("non-compute path", "product", () => {
      applyComputeResourceReplace(device, { path: "graphics" }, {
        resource_index: 0,
        generation: 0,
        buffer_descriptor: storageBuf(256),
      });
    });
  }

  // ── 9. Queue-completion gate ──────────────────────────────────────────
  {
    const device = createFakeDevice();
    const resources = createWebGpuResources(device, computeDescriptor());

    applyComputeResourceReplace(device, resources, {
      resource_index: 10,
      generation: 0,
      buffer_descriptor: storageBuf(256),
    });
    applyComputeResourceReplace(device, resources, {
      resource_index: 10,
      generation: 1,
      buffer_descriptor: storageBufAlt(512),
    });

    const retiredEntry = resources.pendingRetire[0];
    require(retiredEntry != null, "pending retire group exists");
    require(retiredEntry.logicalId === 10, "pending retire logicalId");
    require(retiredEntry.buffers.length === 1, "one buffer in pending group");
    require(!retiredEntry.buffers[0].__faberDestroyed, "pending buffer not destroyed before queue");

    // No submit happened yet — but onSubmittedWorkDone is still callable.
    const destroyResult = await destroyRetiredComputeResources(device, resources);
    require(destroyResult.destroyed_groups === 1, "group destroyed");
    require(destroyResult.destroyed_buffers === 1, "buffer destroyed");
    require(retiredEntry.buffers[0].__faberDestroyed, "pending buffer destroyed after queue");
    require(computeResourceCounters(resources).destroyed === 1, "destroyed counter updated");
  }

  // ── 10. Missing API detection ─────────────────────────────────────────
  {
    const device = createFakeDevice();
    const resources = createWebGpuResources(device, computeDescriptor());

    applyComputeResourceReplace(device, resources, {
      resource_index: 0,
      generation: 0,
      buffer_descriptor: storageBuf(256),
    });
    applyComputeResourceReplace(device, resources, {
      resource_index: 0,
      generation: 1,
      buffer_descriptor: storageBufAlt(512),
    });

    const broken = {
      queue: {
        submit() {},
        // no onSubmittedWorkDone
      },
    };
    await expectReject("missing onSubmittedWorkDone", "webgpu", () =>
      destroyRetiredComputeResources(broken, resources),
    );
    require(
      resources.buffers.get(0).buffers.every((b) => !b.__faberDestroyed),
      "live buffers intact without completion API",
    );
    require(
      resources.pendingRetire[0].buffers.every((b) => !b.__faberDestroyed),
      "pending not destroyed without completion API",
    );
  }

  // ── 11. Concurrent retire during completion ───────────────────────────
  {
    const device = createFakeDevice();
    const resources = createWebGpuResources(device, computeDescriptor());

    applyComputeResourceReplace(device, resources, {
      resource_index: 20,
      generation: 0,
      buffer_descriptor: storageBuf(256),
    });
    applyComputeResourceReplace(device, resources, {
      resource_index: 21,
      generation: 0,
      buffer_descriptor: storageBuf(256),
    });

    // Retire 20 → pending group A
    applyComputeResourceReplace(device, resources, {
      resource_index: 20,
      generation: 1,
      buffer_descriptor: storageBufAlt(512),
    });
    const groupA = resources.pendingRetire[0].buffers.slice();
    require(resources.pendingRetire.length === 1, "one pending before destroy");

    // During completion wait, retire 21 → group B must survive this destroy.
    const origDone = device.queue.onSubmittedWorkDone.bind(device.queue);
    let injected = false;
    device.queue.onSubmittedWorkDone = async function () {
      if (!injected) {
        injected = true;
        applyComputeResourceReplace(device, resources, {
          resource_index: 21,
          generation: 1,
          buffer_descriptor: storageBufAlt(512),
        });
        require(resources.pendingRetire.length === 1, "only concurrent group pending during await");
      }
      return origDone();
    };

    const destroyResult = await destroyRetiredComputeResources(device, resources);

    require(destroyResult.destroyed_groups === 1, "only pre-await group destroyed");
    require(destroyResult.destroyed_buffers === 1, "only one buffer from snapshot");
    require(groupA.every((b) => b.__faberDestroyed), "snapshot group A destroyed");
    require(resources.pendingRetire.length === 1, "concurrent group B still pending");
    require(
      resources.pendingRetire[0].buffers.every((b) => !b.__faberDestroyed),
      "group B buffers not destroyed under earlier completion",
    );
    require(computeResourceCounters(resources).destroyed === 1, "destroyed counter is 1 not 2");

    // Restore normal completion; later wait covers group B.
    device.queue.onSubmittedWorkDone = origDone;
    const second = await destroyRetiredComputeResources(device, resources);
    require(second.destroyed_groups === 1 && second.destroyed_buffers === 1, "second destroy covers B");
    require(resources.pendingRetire.length === 0, "pending empty after second destroy");
    require(computeResourceCounters(resources).destroyed === 2, "both groups destroyed over two waits");
  }

  // ── 12. Unaffected resource identity ──────────────────────────────────
  {
    const device = createFakeDevice();
    const resources = createWebGpuResources(device, computeDescriptor());

    applyComputeResourceReplace(device, resources, {
      resource_index: 30,
      generation: 0,
      buffer_descriptor: storageBuf(256),
    });
    applyComputeResourceReplace(device, resources, {
      resource_index: 31,
      generation: 0,
      buffer_descriptor: storageBuf(128),
    });

    const stableEntry = resources.buffers.get(31);
    const stableBuffer = stableEntry.buffer;
    const stableGen = stableEntry.generation;
    require(stableGen === 0, "stable gen starts at 0");

    // Replace neighbor 30 — 31 must be unaffected.
    applyComputeResourceReplace(device, resources, {
      resource_index: 30,
      generation: 1,
      buffer_descriptor: storageBufAlt(512),
    });

    const after = resources.buffers.get(31);
    require(after.generation === stableGen, "unaffected generation stable");
    require(after.buffer === stableBuffer, "unaffected buffer identity stable");
    require(!after.buffer.__faberDestroyed, "unaffected buffer not destroyed");
    require(resources.buffers.get(30).generation === 1, "neighbor generation updated");

    await device.queue.onSubmittedWorkDone();
    await destroyRetiredComputeResources(device, resources);
    require(after.buffer === stableBuffer && !after.buffer.__faberDestroyed, "unaffected buffer still intact after destroy");
  }

  // ── 13. Empty pending destroy is no-op ────────────────────────────────
  {
    const device = createFakeDevice();
    const resources = createWebGpuResources(device, computeDescriptor());

    const result = await destroyRetiredComputeResources(device, resources);
    require(result.destroyed_groups === 0 && result.destroyed_buffers === 0, "empty destroy no-op");
    require(resources.pendingRetire.length === 0, "pending stays empty");
  }

  // ── 14. onSubmittedWorkDone rejection re-queues pendingRetire ─────────
  {
    const device = createFakeDevice();
    const resources = createWebGpuResources(device, computeDescriptor());

    applyComputeResourceReplace(device, resources, {
      resource_index: 40,
      generation: 0,
      buffer_descriptor: storageBuf(256),
    });
    applyComputeResourceReplace(device, resources, {
      resource_index: 40,
      generation: 1,
      buffer_descriptor: storageBufAlt(512),
    });
    require(resources.pendingRetire.length === 1, "one pending retire group before reject");

    const origDone = device.queue.onSubmittedWorkDone.bind(device.queue);
    device.queue.onSubmittedWorkDone = async function () {
      throw new Error("simulated fence rejection");
    };

    try {
      await destroyRetiredComputeResources(device, resources);
      fail("expected onSubmittedWorkDone to reject");
    } catch (_e) {
      /* expected */
    }
    require(
      resources.pendingRetire.length === 1,
      "pendingRetire re-queued after reject (not orphaned)",
    );
    require(
      resources.pendingRetire[0].buffers.every((b) => !b.__faberDestroyed),
      "pending buffers not destroyed after reject",
    );

    // Restore normal fence; a second call must still succeed.
    device.queue.onSubmittedWorkDone = origDone;
    await destroyRetiredComputeResources(device, resources);
    require(resources.pendingRetire.length === 0, "pendingRetire empty after second call");
  }

  // ── Counters: cross-scenario invariant ────────────────────────────────
  {
    const device = createFakeDevice();
    const resources = createWebGpuResources(device, computeDescriptor());

    // Create three resources.
    for (let i = 0; i < 3; i++) {
      applyComputeResourceReplace(device, resources, {
        resource_index: i,
        generation: 0,
        buffer_descriptor: storageBuf(256),
      });
    }
    let c = computeResourceCounters(resources);
    require(c.created === 3 && c.live === 3 && c.retired === 0 && c.destroyed === 0, "all created, nothing retired");

    // Replace one (generation bump).
    applyComputeResourceReplace(device, resources, {
      resource_index: 1,
      generation: 1,
      buffer_descriptor: storageBufAlt(512),
    });
    c = computeResourceCounters(resources);
    require(c.created === 4, `created 4 got ${c.created}`);
    require(c.live === 3, `live 3 got ${c.live}`);
    require(c.retired === 1, `retired 1 got ${c.retired}`);
    require(c.live === c.created - c.retired, `live (${c.live}) = created (${c.created}) - retired (${c.retired})`);
    require(c.retired >= c.destroyed, `retired (${c.retired}) >= destroyed (${c.destroyed})`);

    // Remove one.
    applyComputeResourceReplace(device, resources, {
      resource_index: 2,
      generation: 0,
      buffer_descriptor: null,
    });
    c = computeResourceCounters(resources);
    require(c.live === 2, `live 2 after remove got ${c.live}`);
    require(c.retired === 2, `retired 2 after remove got ${c.retired}`);
    require(c.live === c.created - c.retired, `live (${c.live}) = created (${c.created}) - retired (${c.retired}) after remove`);

    // Destroy retired.
    await device.queue.onSubmittedWorkDone();
    await destroyRetiredComputeResources(device, resources);
    c = computeResourceCounters(resources);
    require(c.destroyed === 2, `destroyed 2 got ${c.destroyed}`);
    require(c.live === 2, `live still 2 got ${c.live}`);
    require(c.retired >= c.destroyed, `retired (${c.retired}) >= destroyed (${c.destroyed}) final`);
    require(c.retired === 2 && c.destroyed === 2, "cumulative counters: retired equals destroyed");
  }

  // ── Counters: sequential replace + destroy across two waits ───────────
  {
    const device = createFakeDevice();
    const resources = createWebGpuResources(device, computeDescriptor());

    applyComputeResourceReplace(device, resources, {
      resource_index: 50,
      generation: 0,
      buffer_descriptor: storageBuf(256),
    });
    let c = computeResourceCounters(resources);
    require(c.created === 1 && c.live === 1, "single create");
    require(c.retired === 0, "retired 0");

    applyComputeResourceReplace(device, resources, {
      resource_index: 50,
      generation: 1,
      buffer_descriptor: storageBufAlt(512),
    });
    c = computeResourceCounters(resources);
    require(c.created === 2 && c.live === 1 && c.retired === 1, "replace increments created and retired");

    await device.queue.onSubmittedWorkDone();
    await destroyRetiredComputeResources(device, resources);
    c = computeResourceCounters(resources);
    require(c.destroyed === 1, "destroyed increments");

    applyComputeResourceReplace(device, resources, {
      resource_index: 50,
      generation: 2,
      buffer_descriptor: storageBuf(128),
    });
    c = computeResourceCounters(resources);
    require(c.created === 3 && c.live === 1 && c.retired === 2, "second replace");
    require(c.retired > c.destroyed, "retired > destroyed before second wait");

    await device.queue.onSubmittedWorkDone();
    await destroyRetiredComputeResources(device, resources);
    c = computeResourceCounters(resources);
    require(c.destroyed === 2, "destroyed 2 after second wait");
    require(c.created === 3, "created never regresses");
    require(c.live === 1, "live never regresses below created-retired");
    require(c.retired === 2, "retired never regresses");
    require(c.destroyed === 2, "destroyed never regresses");
  }

  console.log("compute-resource-lifecycle-check passed");
  console.log("path: compute");
  console.log("covered: create, replace create-before-retire, remove, counters, invalid transitions, queue completion gate, missing API detection, concurrent retire-during-await, unaffected identity, empty pending destroy");
}

main().catch((error) => {
  fail(error?.stack ?? String(error));
});
