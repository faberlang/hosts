#!/usr/bin/env node
/**
 * HV-07B: queue-safe per-chunk replace/retire + multi-draw residual close.
 *
 * Node + fake device/queue — no browser GPU.
 * Covers:
 * - create / replace / remove transitions
 * - create-before-retire ordering
 * - destroy only after onSubmittedWorkDone
 * - honest created/live/retired/destroyed counters
 * - invalid transition rejection
 * - one resource pair + one draw per non-empty chunk
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

const { FaberKernelContractError, loadFaberGraphicsPipeline } = await import(
  pathToFileURL(path.join(here, "faber-kernel.js")).href
);
const {
  createChunkGraphicsResources,
  applyChunkResourceReplace,
  destroyRetiredChunkResources,
  runChunkGraphicsFrame,
  chunkResourceCounters,
  liveChunkIds,
  chunkResourceSnapshot,
} = await import(pathToFileURL(path.join(here, "webgpu-runtime.js")).href);

function fail(message) {
  console.error(`chunk-resource-lifecycle-check failed: ${message}`);
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

// ── Minimal graphics descriptor (matches hello-voxel vertex layout) ──────

function graphicsReflection() {
  return {
    schema_version: 1,
    target: "wgsl-text",
    kernels: [
      {
        entry_name: "hello_voxel_vertex",
        shader_stage: "vertex",
        vertex_input_count: 2,
        vertex_inputs: [
          { source_name: "position", location: 0, format: "float32x3", step_mode: "vertex", offset_bytes: 0, stride_bytes: 12 },
          { source_name: "color", location: 1, format: "float32x3", step_mode: "vertex", offset_bytes: 0, stride_bytes: 12 },
        ],
        resources: [
          { group: 0, binding: 0, kind: "storage-buffer", role: "input", access: "read", element_layout: "f32", element_byte_width: 4, element_count: 64, buffer_byte_len: 256, source_local: null, source_name: "transform" },
        ],
        launch: {
          entry_name: "hello_voxel_vertex",
          shader_stage: "vertex",
          webgpu_adapter: {
            pipeline_layout_descriptor: { bind_group_layout_count: 1, bind_group_layout_indexes: [0], bind_group_layout_index_count: 1 },
            bind_group_layout_descriptor_count: 1,
            bind_group_layout_descriptor_indexes: [0],
            bind_group_layout_descriptor_index_count: 1,
            bind_group_layout_descriptors: [{
              bind_group_index: 0, group: 0,
              layout_entry_indexes: [0], layout_entry_index_count: 1, entry_count: 1,
              entries: [{ binding: 0, binding_index: 0, buffer_byte_len: 256, buffer_byte_offset: 0, binding_byte_len: 256, visibility: "vertex", buffer_type: "read-only-storage", has_dynamic_offset: false, min_binding_size: 256, resource_index: 0, layout_entry_index: 0, source_local: null, source_name: "transform" }],
            }],
            bind_group_descriptor_count: 1,
            bind_group_descriptor_indexes: [0],
            bind_group_descriptor_index_count: 1,
            bind_group_descriptors: [{
              bind_group_index: 0, group: 0,
              entry_indexes: [0], entry_index_count: 1, entry_count: 1,
              entries: [{ binding: 0, kind: "storage-buffer", role: "input", access: "read", shader_access: "read", shader_visibility: "vertex", element_layout: "f32", element_byte_width: 4, element_count: 64, resource_index: 0, binding_index: 0, buffer_type: "read-only-storage", buffer_byte_len: 256, buffer_byte_offset: 0, binding_byte_len: 256, min_binding_size: 256, has_dynamic_offset: false, source_local: null, source_name: "transform" }],
            }],
            vertex_buffer_layout_descriptor_count: 2,
            vertex_buffer_layout_descriptor_indexes: [0, 1],
            vertex_buffer_layout_descriptor_index_count: 2,
            vertex_buffer_layout_descriptors: [
              { buffer_index: 0, array_stride: 12, step_mode: "vertex", attribute_indexes: [0], attribute_index_count: 1, attribute_count: 1, attributes: [{ attribute_index: 0, shader_location: 0, format: "float32x3", offset: 0, source_name: "position" }], source_name: "position" },
              { buffer_index: 1, array_stride: 12, step_mode: "vertex", attribute_indexes: [0], attribute_index_count: 1, attribute_count: 1, attributes: [{ attribute_index: 0, shader_location: 1, format: "float32x3", offset: 0, source_name: "color" }], source_name: "color" },
            ],
            dispatch_workgroup_dimension_count: 3,
            dispatch_workgroups: { x: 1, y: 1, z: 1 },
          },
        },
      },
      {
        entry_name: "hello_voxel_fragment",
        shader_stage: "fragment",
        vertex_input_count: 0,
        vertex_inputs: [],
        resources: [],
        launch: {
          entry_name: "hello_voxel_fragment",
          shader_stage: "fragment",
          webgpu_adapter: {
            pipeline_layout_descriptor: { bind_group_layout_count: 1, bind_group_layout_indexes: [0], bind_group_layout_index_count: 1 },
            bind_group_layout_descriptor_count: 1,
            bind_group_layout_descriptor_indexes: [0],
            bind_group_layout_descriptor_index_count: 1,
            bind_group_layout_descriptors: [{
              bind_group_index: 0, group: 0,
              layout_entry_indexes: [0], layout_entry_index_count: 1, entry_count: 1,
              entries: [{ binding: 0, binding_index: 0, buffer_byte_len: 256, buffer_byte_offset: 0, binding_byte_len: 256, visibility: "fragment", buffer_type: "read-only-storage", has_dynamic_offset: false, min_binding_size: 256, resource_index: 0, layout_entry_index: 0, source_local: null, source_name: "transform" }],
            }],
            bind_group_descriptor_count: 1,
            bind_group_descriptor_indexes: [0],
            bind_group_descriptor_index_count: 1,
            bind_group_descriptors: [{
              bind_group_index: 0, group: 0,
              entry_indexes: [0], entry_index_count: 1, entry_count: 1,
              entries: [{ binding: 0, kind: "storage-buffer", role: "input", access: "read", shader_access: "read", shader_visibility: "fragment", element_layout: "f32", element_byte_width: 4, element_count: 64, resource_index: 0, binding_index: 0, buffer_type: "read-only-storage", buffer_byte_len: 256, buffer_byte_offset: 0, binding_byte_len: 256, min_binding_size: 256, has_dynamic_offset: false, source_local: null, source_name: "transform" }],
            }],
            vertex_buffer_layout_descriptor_count: 0,
            vertex_buffer_layout_descriptor_indexes: [],
            vertex_buffer_layout_descriptor_index_count: 0,
            vertex_buffer_layout_descriptors: [],
            dispatch_workgroup_dimension_count: 3,
            dispatch_workgroups: { x: 1, y: 1, z: 1 },
          },
        },
      },
    ],
    pipeline: {
      color_target_formats: ["bgra8unorm"],
      primitive_topology: "triangle-list",
      vertex_count: 6,
      depth_stencil: {
        depth_write_enabled: true,
        depth_compare: "less",
        stencil_read_mask: 4294967295,
        stencil_write_mask: 4294967295,
        stencil_front: { compare: "always", fail_op: "keep", depth_fail_op: "keep", pass_op: "keep" },
        stencil_back: { compare: "always", fail_op: "keep", depth_fail_op: "keep", pass_op: "keep" },
      },
    },
  };
}

function graphicsWgsl() {
  return "@vertex fn hello_voxel_vertex() -> @builtin(position) vec4<f32> {}\n@fragment fn hello_voxel_fragment() -> @location(0) vec4<f32> {}";
}

function drawManifest() {
  return { index_format: "uint32", instance_count: 1, base_vertex: 0, first_index: 0, index_count: 6 };
}

// ── Fake WebGPU device ───────────────────────────────────────────────────

let bufferSeq = 0;

function createFakeDevice() {
  const destroyed = [];
  const created = [];
  let workDoneResolvers = [];
  let submitted = 0;
  let completionArmed = false;

  const queue = {
    submit(commands) {
      submitted += 1;
      completionArmed = true;
      return undefined;
    },
    onSubmittedWorkDone() {
      if (!completionArmed && submitted === 0) {
        // Still resolve — WebGPU allows calling without prior submit.
      }
      return new Promise((resolve) => {
        // Resolve asynchronously so tests can observe pre-destroy counters.
        queueMicrotask(() => {
          completionArmed = false;
          resolve();
          for (const r of workDoneResolvers.splice(0)) {
            r();
          }
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
      const backing = new ArrayBuffer(size);
      const buffer = {
        id,
        size,
        usage: desc.usage,
        __faberDestroyed: false,
        getMappedRange() {
          require(mapped, `fake buffer ${id}: getMappedRange without map`);
          return backing;
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
    createRenderPipeline() {
      return { __kind: "rp" };
    },
    createTexture(desc) {
      return {
        __kind: "tex",
        width: desc.size.width,
        height: desc.size.height,
        createView() {
          return { __kind: "view" };
        },
        destroy() {},
      };
    },
    createCommandEncoder() {
      const draws = [];
      const pass = {
        setPipeline() {},
        setBindGroup() {},
        setVertexBuffer() {},
        setIndexBuffer() {},
        drawIndexed(indexCount, instanceCount, firstIndex, baseVertex) {
          draws.push({ indexCount, instanceCount, firstIndex, baseVertex });
        },
        end() {},
      };
      return {
        beginRenderPass() {
          return pass;
        },
        finish() {
          return { __kind: "cmd", draws };
        },
      };
    },
    __created: created,
    __destroyed: destroyed,
    __submitted: () => submitted,
  };

  return device;
}

function createFakeCanvasContext() {
  return {
    getCurrentTexture() {
      return {
        format: "bgra8unorm",
        width: 64,
        height: 64,
        createView() {
          return { __kind: "swap-view" };
        },
      };
    },
  };
}

/** One triangle face: 4 unique verts × float32x3 + 6 uint32 indices. */
function meshPayload(seed = 0) {
  const positions = new Float32Array([
    seed, 0, 0,
    seed + 1, 0, 0,
    seed + 1, 1, 0,
    seed, 1, 0,
  ]);
  const colors = new Float32Array([
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
    1, 1, 0,
  ]);
  const indices = new Uint32Array([0, 1, 2, 0, 2, 3]);
  return { positions, colors, indices };
}

function meshPayloadAlt(seed = 10) {
  // Different content, same topology — generation must still advance.
  const positions = new Float32Array([
    seed, 0, 1,
    seed + 1, 0, 1,
    seed + 1, 1, 1,
    seed, 1, 1,
  ]);
  const colors = new Float32Array([
    0.5, 0.5, 0.5,
    0.5, 0.5, 0.5,
    0.5, 0.5, 0.5,
    0.5, 0.5, 0.5,
  ]);
  const indices = new Uint32Array([0, 1, 2, 0, 2, 3]);
  return { positions, colors, indices };
}

async function main() {
  require(typeof createChunkGraphicsResources === "function", "export createChunkGraphicsResources");
  require(typeof applyChunkResourceReplace === "function", "export applyChunkResourceReplace");
  require(typeof destroyRetiredChunkResources === "function", "export destroyRetiredChunkResources");
  require(typeof runChunkGraphicsFrame === "function", "export runChunkGraphicsFrame");
  require(typeof chunkResourceCounters === "function", "export chunkResourceCounters");

  const descriptor = loadFaberGraphicsPipeline({
    wgsl: graphicsWgsl(),
    reflection: graphicsReflection(),
    drawManifest: drawManifest(),
  });

  const transform = new Float32Array(64);
  transform[0] = 1;
  transform[5] = 1;
  transform[10] = 1;
  transform[15] = 1;

  // ── Create two chunks: counters + multi-draw ───────────────────────────
  {
    const device = createFakeDevice();
    const context = createFakeCanvasContext();
    const resources = createChunkGraphicsResources(
      device,
      descriptor,
      { storageData: { transform } },
      context,
    );

    require(resources.path === "per-chunk-multi-draw", "path must be per-chunk-multi-draw");

    const c0 = applyChunkResourceReplace(device, resources, {
      logical_id: 0,
      generation: 0,
      payload: meshPayload(0),
    });
    require(c0.kind === "created", "chunk 0 create kind");
    require(c0.index_count === 6, "chunk 0 index_count");

    const c1 = applyChunkResourceReplace(device, resources, {
      logical_id: 1,
      generation: 0,
      payload: meshPayload(2),
    });
    require(c1.kind === "created", "chunk 1 create kind");

    const counters = chunkResourceCounters(resources);
    require(counters.created === 6, `created after two creates: expected 6 got ${counters.created}`);
    require(counters.live === 6, `live after two creates: expected 6 got ${counters.live}`);
    require(counters.retired === 0, "retired still 0");
    require(counters.destroyed === 0, "destroyed still 0");
    require(counters.live_chunks === 2, "live_chunks 2");
    require(counters.path === "per-chunk-multi-draw", "counters.path");

    const ids = liveChunkIds(resources);
    require(ids.length === 2 && ids[0] === 0 && ids[1] === 1, "live ids [0,1]");

    const snap0 = chunkResourceSnapshot(resources, 0);
    require(snap0 && snap0.generation === 0 && snap0.buffer_count === 3, "snapshot chunk 0");

    const frameState = { submittedFrameCount: 0 };
    const drawResult = runChunkGraphicsFrame(device, context, resources, descriptor, frameState, {
      recordSubmit: true,
    });
    require(drawResult.draw_count === 2, `draw_count expected 2 got ${drawResult.draw_count}`);
    require(drawResult.draws[0].logical_id === 0, "draw order chunk 0 first");
    require(drawResult.draws[1].logical_id === 1, "draw order chunk 1 second");
    require(frameState.submittedFrameCount === 1, "submittedFrameCount");
    require(frameState.submits[0].multi_draw === true, "submit records multi_draw");
    require(frameState.submits[0].path === "per-chunk-multi-draw", "submit path");
    require(frameState.submits[0].draw_count === 2, "submit draw_count");
    require(device.__submitted() === 1, "queue.submit called once");
  }

  // ── Replace: create-before-retire; destroy only after work done ────────
  {
    const device = createFakeDevice();
    const context = createFakeCanvasContext();
    const resources = createChunkGraphicsResources(
      device,
      descriptor,
      { storageData: { transform } },
      context,
    );

    applyChunkResourceReplace(device, resources, {
      logical_id: 0,
      generation: 0,
      payload: meshPayload(0),
    });
    const createdBefore = device.__created.filter((b) => !b.__faberDestroyed).length;
    require(createdBefore === 3 + 1, "3 mesh buffers + 1 storage after create"); // storage + 3 mesh
    // storage buffer is also created; track mesh via counters only.

    const firstBuffers = resources.chunks.get(0).buffers.slice();
    require(firstBuffers.length === 3, "first entry has 3 buffers");
    require(firstBuffers.every((b) => !b.__faberDestroyed), "first buffers not destroyed yet");

    const replaced = applyChunkResourceReplace(device, resources, {
      logical_id: 0,
      generation: 1,
      payload: meshPayloadAlt(0),
    });
    require(replaced.kind === "replaced", "replace kind");
    require(replaced.previous_generation === 0, "previous generation 0");
    require(replaced.generation === 1, "new generation 1");

    const mid = chunkResourceCounters(resources);
    require(mid.created === 6, `created after replace: expected 6 got ${mid.created}`);
    require(mid.live === 3, `live after replace: expected 3 got ${mid.live}`);
    require(mid.retired === 3, `retired after replace: expected 3 got ${mid.retired}`);
    require(mid.destroyed === 0, "destroyed still 0 before queue done");
    require(firstBuffers.every((b) => !b.__faberDestroyed), "old buffers not destroyed before work done");
    require(resources.chunks.get(0).generation === 1, "live generation is 1");
    require(resources.pendingRetire.length === 1, "one pending retire group");

    // Submit a frame that uses the new buffers, then complete.
    const frameState = { submittedFrameCount: 0 };
    runChunkGraphicsFrame(device, context, resources, descriptor, frameState);
    const destroyResult = await destroyRetiredChunkResources(device, resources);
    require(destroyResult.destroyed_groups === 1, "destroyed one group");
    require(destroyResult.destroyed_buffers === 3, "destroyed three buffers");
    require(firstBuffers.every((b) => b.__faberDestroyed), "old buffers destroyed after work done");

    const after = chunkResourceCounters(resources);
    require(after.destroyed === 3, `destroyed counter: expected 3 got ${after.destroyed}`);
    require(after.live === 3, "live remains 3");
    require(after.retired === 3, "retired remains 3 (cumulative)");
    require(after.pending_retire_groups === 0, "pending cleared");
    require(resources.pendingRetire.length === 0, "pending array empty");
  }

  // ── Remove empty: retire then destroy ──────────────────────────────────
  {
    const device = createFakeDevice();
    const context = createFakeCanvasContext();
    const resources = createChunkGraphicsResources(
      device,
      descriptor,
      { storageData: { transform } },
      context,
    );

    applyChunkResourceReplace(device, resources, {
      logical_id: 2,
      generation: 0,
      payload: meshPayload(5),
    });
    const old = resources.chunks.get(2).buffers.slice();

    const removed = applyChunkResourceReplace(device, resources, {
      logical_id: 2,
      generation: 0,
      payload: null,
    });
    require(removed.kind === "removed", "remove kind");
    require(liveChunkIds(resources).length === 0, "no live chunks after remove");

    const mid = chunkResourceCounters(resources);
    require(mid.live === 0, "live 0 after remove");
    require(mid.retired === 3, "retired 3 after remove");
    require(mid.destroyed === 0, "not destroyed yet");
    require(old.every((b) => !b.__faberDestroyed), "buffers survive until queue done");

    runChunkGraphicsFrame(device, context, resources, descriptor, { submittedFrameCount: 0 });
    await destroyRetiredChunkResources(device, resources);
    require(old.every((b) => b.__faberDestroyed), "removed buffers destroyed after work done");
    require(chunkResourceCounters(resources).destroyed === 3, "destroyed 3");
  }

  // ── Empty object payload is remove ─────────────────────────────────────
  {
    const device = createFakeDevice();
    const context = createFakeCanvasContext();
    const resources = createChunkGraphicsResources(
      device,
      descriptor,
      { storageData: { transform } },
      context,
    );
    applyChunkResourceReplace(device, resources, {
      logical_id: 0,
      generation: 0,
      payload: meshPayload(0),
    });
    const removed = applyChunkResourceReplace(device, resources, {
      logical_id: 0,
      generation: 0,
      payload: { empty: true },
    });
    require(removed.kind === "removed", "empty:true is remove");
  }

  // ── Invalid transitions ────────────────────────────────────────────────
  {
    const device = createFakeDevice();
    const context = createFakeCanvasContext();
    const resources = createChunkGraphicsResources(
      device,
      descriptor,
      { storageData: { transform } },
      context,
    );

    await expectReject("remove missing chunk", "product", () => {
      applyChunkResourceReplace(device, resources, {
        logical_id: 0,
        generation: 0,
        payload: null,
      });
    });

    applyChunkResourceReplace(device, resources, {
      logical_id: 0,
      generation: 0,
      payload: meshPayload(0),
    });

    await expectReject("same generation replace", "product", () => {
      applyChunkResourceReplace(device, resources, {
        logical_id: 0,
        generation: 0,
        payload: meshPayloadAlt(0),
      });
    });

    await expectReject("stale generation replace", "product", () => {
      applyChunkResourceReplace(device, resources, {
        logical_id: 0,
        generation: -1,
        payload: meshPayloadAlt(0),
      });
    });

    await expectReject("remove wrong generation", "product", () => {
      applyChunkResourceReplace(device, resources, {
        logical_id: 0,
        generation: 99,
        payload: null,
      });
    });

    await expectReject("negative logical_id", "product", () => {
      applyChunkResourceReplace(device, resources, {
        logical_id: -1,
        generation: 0,
        payload: meshPayload(0),
      });
    });

    await expectReject("partial empty payload", "product", () => {
      applyChunkResourceReplace(device, resources, {
        logical_id: 1,
        generation: 0,
        payload: {
          positions: new Float32Array([0, 0, 0]),
          colors: new Float32Array([]),
          indices: new Uint32Array([0, 1, 2]),
        },
      });
    });

    await expectReject("non-chunk resources session", "product", () => {
      applyChunkResourceReplace(device, { path: "other" }, {
        logical_id: 0,
        generation: 0,
        payload: meshPayload(0),
      });
    });
  }

  // ── Destroy without onSubmittedWorkDone rejects ────────────────────────
  {
    const device = createFakeDevice();
    const context = createFakeCanvasContext();
    const resources = createChunkGraphicsResources(
      device,
      descriptor,
      { storageData: { transform } },
      context,
    );
    applyChunkResourceReplace(device, resources, {
      logical_id: 0,
      generation: 0,
      payload: meshPayload(0),
    });
    applyChunkResourceReplace(device, resources, {
      logical_id: 0,
      generation: 1,
      payload: meshPayloadAlt(0),
    });

    const broken = {
      queue: {
        submit() {},
        // no onSubmittedWorkDone
      },
    };
    await expectReject("missing onSubmittedWorkDone", "webgpu", () =>
      destroyRetiredChunkResources(broken, resources),
    );
    require(
      resources.chunks.get(0).buffers.every((b) => !b.__faberDestroyed),
      "live buffers intact without completion API",
    );
    require(
      resources.pendingRetire[0].buffers.every((b) => !b.__faberDestroyed),
      "pending not destroyed without completion API",
    );
  }

  // ── Unaffected chunk identity stable across neighbor replace ───────────
  {
    const device = createFakeDevice();
    const context = createFakeCanvasContext();
    const resources = createChunkGraphicsResources(
      device,
      descriptor,
      { storageData: { transform } },
      context,
    );

    applyChunkResourceReplace(device, resources, {
      logical_id: 0,
      generation: 0,
      payload: meshPayload(0),
    });
    applyChunkResourceReplace(device, resources, {
      logical_id: 1,
      generation: 0,
      payload: meshPayload(2),
    });
    const stableBuffers = resources.chunks.get(1).buffers.slice();
    const stableGen = resources.chunks.get(1).generation;

    applyChunkResourceReplace(device, resources, {
      logical_id: 0,
      generation: 1,
      payload: meshPayloadAlt(0),
    });

    require(resources.chunks.get(1).generation === stableGen, "unaffected generation stable");
    require(
      resources.chunks.get(1).buffers.every((b, i) => b === stableBuffers[i]),
      "unaffected buffer identity stable",
    );
    require(chunkResourceSnapshot(resources, 1).generation === 0, "snapshot unaffected");

    runChunkGraphicsFrame(device, context, resources, descriptor, { submittedFrameCount: 0 });
    await destroyRetiredChunkResources(device, resources);
    require(
      resources.chunks.get(1).buffers.every((b) => !b.__faberDestroyed),
      "unaffected buffers not destroyed",
    );
  }

  // ── Empty pending destroy is no-op (no queue wait required) ────────────
  {
    const device = createFakeDevice();
    const context = createFakeCanvasContext();
    const resources = createChunkGraphicsResources(
      device,
      descriptor,
      { storageData: { transform } },
      context,
    );
    const result = await destroyRetiredChunkResources(device, resources);
    require(result.destroyed_groups === 0 && result.destroyed_buffers === 0, "empty destroy no-op");
  }

  // ── Concurrent retire during onSubmittedWorkDone is not destroyed ──────
  // Snapshot-before-await: groups retired while waiting stay pending.
  {
    const device = createFakeDevice();
    const context = createFakeCanvasContext();
    const resources = createChunkGraphicsResources(
      device,
      descriptor,
      { storageData: { transform } },
      context,
    );

    applyChunkResourceReplace(device, resources, {
      logical_id: 0,
      generation: 0,
      payload: meshPayload(0),
    });
    applyChunkResourceReplace(device, resources, {
      logical_id: 1,
      generation: 0,
      payload: meshPayload(2),
    });

    // Retire chunk 0 → pending group A
    applyChunkResourceReplace(device, resources, {
      logical_id: 0,
      generation: 1,
      payload: meshPayloadAlt(0),
    });
    const groupA = resources.pendingRetire[0].buffers.slice();
    require(resources.pendingRetire.length === 1, "one pending before destroy");

    // During the completion wait, retire chunk 1 → group B must survive this destroy.
    // One-shot inject so a later destroy uses the normal completion path.
    const origDone = device.queue.onSubmittedWorkDone.bind(device.queue);
    let injected = false;
    device.queue.onSubmittedWorkDone = async function () {
      if (!injected) {
        injected = true;
        applyChunkResourceReplace(device, resources, {
          logical_id: 1,
          generation: 1,
          payload: meshPayloadAlt(2),
        });
        require(resources.pendingRetire.length === 1, "only concurrent group in pending during await");
      }
      return origDone();
    };

    runChunkGraphicsFrame(device, context, resources, descriptor, { submittedFrameCount: 0 });
    const destroyResult = await destroyRetiredChunkResources(device, resources);

    require(destroyResult.destroyed_groups === 1, "only pre-await group destroyed");
    require(destroyResult.destroyed_buffers === 3, "only three buffers from snapshot");
    require(groupA.every((b) => b.__faberDestroyed), "snapshot group A destroyed");
    require(resources.pendingRetire.length === 1, "concurrent group B still pending");
    require(
      resources.pendingRetire[0].buffers.every((b) => !b.__faberDestroyed),
      "group B buffers not destroyed under earlier completion",
    );
    require(chunkResourceCounters(resources).destroyed === 3, "destroyed counter is 3 not 6");

    // Restore normal completion; later wait covers group B.
    device.queue.onSubmittedWorkDone = origDone;
    runChunkGraphicsFrame(device, context, resources, descriptor, { submittedFrameCount: 1 });
    const second = await destroyRetiredChunkResources(device, resources);
    require(second.destroyed_groups === 1 && second.destroyed_buffers === 3, "second destroy covers B");
    require(resources.pendingRetire.length === 0, "pending empty after second destroy");
    require(chunkResourceCounters(resources).destroyed === 6, "both groups destroyed over two waits");
  }

  // ── indexCount follows resources.indexFormat (not byteLength%4) ────────
  {
    const device = createFakeDevice();
    const context = createFakeCanvasContext();
    const resources = createChunkGraphicsResources(
      device,
      descriptor,
      { storageData: { transform } },
      context,
    );
    require(resources.indexFormat === "uint32", "session stores draw indexFormat");

    // 6 uint32 indices → 24 bytes (also %4===0 under the old heuristic).
    const created = applyChunkResourceReplace(device, resources, {
      logical_id: 0,
      generation: 0,
      payload: meshPayload(0),
    });
    require(created.index_count === 6, "uint32 indexCount is 6");
    require(resources.chunks.get(0).indexCount === 6, "entry indexCount is 6");
    require(resources.chunks.get(0).indexFormat === "uint32", "entry records indexFormat");
  }

  console.log("chunk-resource-lifecycle-check passed");
  console.log("path: per-chunk-multi-draw");
  console.log("covered: create, replace create-before-retire, remove, multi-draw, counters, invalid transitions, queue completion gate, unaffected identity, concurrent retire-during-await, indexFormat-driven indexCount");
}

main().catch((error) => {
  fail(error?.stack ?? String(error));
});
