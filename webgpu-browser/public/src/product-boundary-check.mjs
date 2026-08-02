#!/usr/bin/env node
/**
 * Focused product-boundary admission for webgpu-browser.
 *
 * Runs under Node without a browser GPU. Covers:
 * - artifact-fetch failure
 * - unsupported / rejected reflection
 * - unavailable WebGPU
 *
 * Does not claim browser GPU execution or exact add-one readback (those remain
 * controlled/manual browser inspection evidence).
 */

import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const publicDir = path.resolve(here, "..");
const generatedDir = path.join(publicDir, "generated");

const { FaberKernelContractError, fetchFaberKernelArtifacts, loadFaberKernel, loadFaberGraphicsPipeline } = await import(
  pathToFileURL(path.join(here, "contract", "artifact-admission.js")).href
);
const {
  acquireWebGpuDevice,
  createWebGpuResources,
  runKernel,
  createGraphicsResources,
  runGraphicsFrame,
  replaceDepthTextureOnResize,
  onDeviceLost,
  createChunkGraphicsResources,
  applyChunkResourceReplace,
  destroyRetiredChunkResources,
  runChunkGraphicsFrame,
  chunkResourceCounters,
} = await import(pathToFileURL(path.join(here, "backend", "webgpu-runtime.js")).href);

// ── Graphics test fixtures ────────────────────────────────────────────────

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
      vertex_count: 36,
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
  return { index_format: "uint32", instance_count: 1, base_vertex: 0, first_index: 0, index_count: 36 };
}

function fail(message) {
  console.error(`product-boundary-check failed: ${message}`);
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

async function main() {
  const wgsl = await readFile(path.join(generatedDir, "kernel.wgsl"), "utf8");
  const reflection = JSON.parse(await readFile(path.join(generatedDir, "reflection.json"), "utf8"));

  // Happy path for static reflection consumer (not GPU execution).
  const kernel = loadFaberKernel({ wgsl, reflection });
  require(kernel.entryName === "add_one", "kernel entry must be add_one");
  // U2: the reflection keeps the static (1,1,1) dispatch hint; the host
  // supplies the real extent through the runtime-extent binding.
  require(kernel.dispatchWorkgroups.x === 1, "dispatch x must be 1 (static shape hint)");
  const dataInputs = kernel.inputBindings.filter((entry) => entry.kind === "storage-buffer");
  const extentInputs = kernel.inputBindings.filter((entry) => entry.kind === "runtime-extent");
  require(dataInputs.length === 1, "expected one storage-buffer input binding");
  require(extentInputs.length === 1, "expected one runtime-extent input binding");
  require(kernel.inputBindings.length === 2, "expected input + runtime-extent bindings");
  require(kernel.outputBindings.length === 1, "expected one output binding");

  await expectReject("artifact-fetch failure", "artifact-fetch", () =>
    fetchFaberKernelArtifacts({
      fetchImpl: async () => ({ ok: false, status: 404 }),
    }),
  );

  await expectReject("unsupported reflection schema", "reflection", async () => {
    loadFaberKernel({
      wgsl,
      reflection: { ...reflection, schema_version: 999 },
    });
  });

  await expectReject("missing webgpu_adapter", "reflection", async () => {
    const bad = structuredClone(reflection);
    delete bad.kernels[0].launch.webgpu_adapter;
    loadFaberKernel({ wgsl, reflection: bad });
  });

  await expectReject("unavailable WebGPU", "webgpu", () =>
    acquireWebGpuDevice({ navigator: {} }),
  );

  await expectReject("no WebGPU adapter", "webgpu", () =>
    acquireWebGpuDevice({
      navigator: {
        gpu: {
          requestAdapter: async () => null,
        },
      },
    }),
  );

  // ── Graphics runtime tests ────────────────────────────────────────────

  // Verify all graphics exports are functions
  require(typeof createGraphicsResources === "function", "graphics runtime: createGraphicsResources must be a function");
  require(typeof runGraphicsFrame === "function", "graphics runtime: runGraphicsFrame must be a function");
  require(typeof replaceDepthTextureOnResize === "function", "graphics runtime: replaceDepthTextureOnResize must be a function");
  require(typeof onDeviceLost === "function", "graphics runtime: onDeviceLost must be a function");
  // HV-07B per-chunk lifecycle exports
  require(typeof createChunkGraphicsResources === "function", "graphics runtime: createChunkGraphicsResources must be a function");
  require(typeof applyChunkResourceReplace === "function", "graphics runtime: applyChunkResourceReplace must be a function");
  require(typeof destroyRetiredChunkResources === "function", "graphics runtime: destroyRetiredChunkResources must be a function");
  require(typeof runChunkGraphicsFrame === "function", "graphics runtime: runChunkGraphicsFrame must be a function");
  require(typeof chunkResourceCounters === "function", "graphics runtime: chunkResourceCounters must be a function");

  // onDeviceLost registers a loss callback on a mock device
  {
    let received = null;
    const mockDevice = {
      lost: Promise.resolve({ reason: "destroyed", message: "test device loss" }),
    };
    onDeviceLost(mockDevice, (info) => {
      received = info;
    });
    // Wait for the promise to resolve
    await new Promise((resolve) => setTimeout(resolve, 10));
    require(received !== null, "graphics runtime: onDeviceLost must invoke callback");
    require(received.kind === "device-lost", "graphics runtime: loss kind must be device-lost");
    require(received.reason === "destroyed", "graphics runtime: loss reason must be destroyed");
    require(received.message === "test device loss", "graphics runtime: loss message must match");
  }

  // Compute runtime re-verify: createWebGpuResources and runKernel still export
  require(typeof createWebGpuResources === "function", "compute runtime: createWebGpuResources must be a function");
  require(typeof runKernel === "function", "compute runtime: runKernel must be a function");

  // Compute runtime device-failure paths still work (via acquireWebGpuDevice)
  await expectReject("compute: unavailable WebGPU after graphics addition", "webgpu", () =>
    acquireWebGpuDevice({ navigator: {} }),
  );

  // ── Graphics admission tests ──────────────────────────────────────────

  const gfxWgsl = graphicsWgsl();
  const gfxRefl = graphicsReflection();
  const draw = drawManifest();

  // Happy path: valid graphics descriptor
  {
    const desc = loadFaberGraphicsPipeline({ wgsl: gfxWgsl, reflection: gfxRefl, drawManifest: draw });
    require(desc.kernels.length === 2, "graphics: expected two kernels");
    require(desc.kernels[0].shaderStage === "vertex", "graphics: first kernel must be vertex");
    require(desc.kernels[1].shaderStage === "fragment", "graphics: second kernel must be fragment");
    require(desc.kernels[0].vertexInputs.length === 2, "graphics: expected two vertex inputs");
    require(desc.pipeline.colorTargetFormats[0] === "bgra8unorm", "graphics: color target must be bgra8unorm");
    require(desc.pipeline.primitiveTopology === "triangle-list", "graphics: topology must be triangle-list");
    require(desc.pipeline.vertexCount === 36, "graphics: vertex count must be 36");
    require(desc.pipeline.depthStencil.depthWriteEnabled === true, "graphics: depth write must be enabled");
    require(desc.pipeline.depthStencil.depthCompare === "less", "graphics: depth compare must be less");
    require(desc.draw.indexFormat === "uint32", "graphics: index format must be uint32");
    require(desc.draw.instanceCount === 1, "graphics: instance count must be 1");
    require(desc.draw.baseVertex === 0, "graphics: base vertex must be 0");
    require(desc.draw.firstIndex === 0, "graphics: first index must be 0");
    require(desc.draw.indexCount === 36, "graphics: index count must be 36");
    require(desc.inputBindings.length === 1, "graphics: expected one input binding");
  }

  // Missing vertex kernel
  await expectReject("graphics: missing vertex kernel", "reflection", () => {
    const bad = structuredClone(gfxRefl);
    bad.kernels[0].shader_stage = "compute";
    loadFaberGraphicsPipeline({ wgsl: gfxWgsl, reflection: bad, drawManifest: draw });
  });

  // Missing fragment kernel
  await expectReject("graphics: missing fragment kernel", "reflection", () => {
    const bad = structuredClone(gfxRefl);
    bad.kernels[1].shader_stage = "compute";
    loadFaberGraphicsPipeline({ wgsl: gfxWgsl, reflection: bad, drawManifest: draw });
  });

  // Wrong kernel count (only one kernel)
  await expectReject("graphics: wrong kernel count", "reflection", () => {
    const bad = structuredClone(gfxRefl);
    bad.kernels = [bad.kernels[0]];
    loadFaberGraphicsPipeline({ wgsl: gfxWgsl, reflection: bad, drawManifest: draw });
  });

  // Missing pipeline block
  await expectReject("graphics: missing pipeline", "reflection", () => {
    const bad = structuredClone(gfxRefl);
    delete bad.pipeline;
    loadFaberGraphicsPipeline({ wgsl: gfxWgsl, reflection: bad, drawManifest: draw });
  });

  // Empty vertex inputs
  await expectReject("graphics: empty vertex inputs", "reflection", () => {
    const bad = structuredClone(gfxRefl);
    bad.kernels[0].vertex_inputs = [];
    bad.kernels[0].vertex_input_count = 0;
    loadFaberGraphicsPipeline({ wgsl: gfxWgsl, reflection: bad, drawManifest: draw });
  });

  // Mismatched vertex buffer layouts
  await expectReject("graphics: mismatched vertex buffer layouts", "reflection", () => {
    const bad = structuredClone(gfxRefl);
    bad.kernels[0].launch.webgpu_adapter.vertex_buffer_layout_descriptors = [];
    bad.kernels[0].launch.webgpu_adapter.vertex_buffer_layout_descriptor_count = 0;
    loadFaberGraphicsPipeline({ wgsl: gfxWgsl, reflection: bad, drawManifest: draw });
  });

  // Malformed draw manifest (bad index_format)
  await expectReject("graphics: bad index format", "reflection", () => {
    const badDraw = { ...draw, index_format: "uint8" };
    loadFaberGraphicsPipeline({ wgsl: gfxWgsl, reflection: gfxRefl, drawManifest: badDraw });
  });

  // Malformed draw manifest (base_vertex >= vertex_count)
  await expectReject("graphics: base_vertex out of range", "reflection", () => {
    const badDraw = { ...draw, base_vertex: 36 };
    loadFaberGraphicsPipeline({ wgsl: gfxWgsl, reflection: gfxRefl, drawManifest: badDraw });
  });

  // Malformed draw manifest (missing instance_count)
  await expectReject("graphics: missing instance count", "reflection", () => {
    const badDraw = { index_format: "uint32", first_index: 0, base_vertex: 0 };
    loadFaberGraphicsPipeline({ wgsl: gfxWgsl, reflection: gfxRefl, drawManifest: badDraw });
  });

  // Malformed draw manifest (index_count <= 0)
  await expectReject("graphics: bad index count (zero)", "reflection", () => {
    const badDraw = { ...draw, index_count: 0 };
    loadFaberGraphicsPipeline({ wgsl: gfxWgsl, reflection: gfxRefl, drawManifest: badDraw });
  });

  // Bad primitive topology
  await expectReject("graphics: bad primitive topology", "reflection", () => {
    const bad = structuredClone(gfxRefl);
    bad.pipeline.primitive_topology = "triangle-strip";
    loadFaberGraphicsPipeline({ wgsl: gfxWgsl, reflection: bad, drawManifest: draw });
  });

  // Bad color target format
  await expectReject("graphics: bad color target", "reflection", () => {
    const bad = structuredClone(gfxRefl);
    bad.pipeline.color_target_formats = ["rgba16float"];
    loadFaberGraphicsPipeline({ wgsl: gfxWgsl, reflection: bad, drawManifest: draw });
  });

  // Missing depth_stencil
  await expectReject("graphics: missing depth_stencil", "reflection", () => {
    const bad = structuredClone(gfxRefl);
    delete bad.pipeline.depth_stencil;
    loadFaberGraphicsPipeline({ wgsl: gfxWgsl, reflection: bad, drawManifest: draw });
  });

  // Fragment bind group divergence: fragment declares bind group not in vertex pipeline layout
  await expectReject("graphics: fragment bind group divergence (extra group)", "reflection", () => {
    const bad = structuredClone(gfxRefl);
    bad.kernels[1].launch.webgpu_adapter.bind_group_descriptor_count = 2;
    bad.kernels[1].launch.webgpu_adapter.bind_group_descriptor_indexes = [0, 1];
    bad.kernels[1].launch.webgpu_adapter.bind_group_descriptor_index_count = 2;
    bad.kernels[1].launch.webgpu_adapter.bind_group_descriptors.push({
      bind_group_index: 1, group: 1,
      entry_indexes: [0], entry_index_count: 1, entry_count: 1,
      entries: [{ binding: 0, kind: "storage-buffer", role: "input", access: "read", shader_access: "read", shader_visibility: "fragment", element_layout: "f32", element_byte_width: 4, element_count: 64, resource_index: 1, binding_index: 0, buffer_type: "read-only-storage", buffer_byte_len: 256, buffer_byte_offset: 0, binding_byte_len: 256, min_binding_size: 256, has_dynamic_offset: false, source_local: null, source_name: "extra" }],
    });
    // Also add the corresponding bind_group_layout_descriptor so parseBindGroups doesn't fail first
    bad.kernels[1].launch.webgpu_adapter.bind_group_layout_descriptor_count = 2;
    bad.kernels[1].launch.webgpu_adapter.bind_group_layout_descriptor_indexes = [0, 1];
    bad.kernels[1].launch.webgpu_adapter.bind_group_layout_descriptor_index_count = 2;
    bad.kernels[1].launch.webgpu_adapter.bind_group_layout_descriptors.push({
      bind_group_index: 1, group: 1,
      layout_entry_indexes: [0], layout_entry_index_count: 1, entry_count: 1,
      entries: [{ binding: 0, binding_index: 0, buffer_byte_len: 256, buffer_byte_offset: 0, binding_byte_len: 256, visibility: "fragment", buffer_type: "read-only-storage", has_dynamic_offset: false, min_binding_size: 256, resource_index: 1, layout_entry_index: 0, source_local: null, source_name: "extra" }],
    });
    loadFaberGraphicsPipeline({ wgsl: gfxWgsl, reflection: bad, drawManifest: draw });
  });

  // ── Real generated graphics reflection admission ──────────────────────

  // Load real generated files and validate through loadFaberGraphicsPipeline
  {
    const realGfxWgsl = await readFile(path.join(generatedDir, "graphics.wgsl"), "utf8");
    const realGfxRefl = JSON.parse(await readFile(path.join(generatedDir, "graphics-reflection.json"), "utf8"));
    const realDraw = JSON.parse(await readFile(path.join(generatedDir, "draw.json"), "utf8"));

    const desc = loadFaberGraphicsPipeline({ wgsl: realGfxWgsl, reflection: realGfxRefl, drawManifest: realDraw });
    require(desc.kernels.length === 2, "real graphics: expected two kernels");
    require(desc.kernels[0].shaderStage === "vertex", "real graphics: first kernel must be vertex");
    require(desc.kernels[0].entryName === "hello_voxel_vertex", "real graphics: vertex entry_name mismatch");
    require(desc.kernels[0].vertexInputs.length === 2, "real graphics: expected two vertex inputs");
    require(desc.kernels[1].shaderStage === "fragment", "real graphics: second kernel must be fragment");
    require(desc.kernels[1].entryName === "hello_voxel_fragment", "real graphics: fragment entry_name mismatch");
    require(desc.pipeline.colorTargetFormats[0] === "bgra8unorm", "real graphics: color target must be bgra8unorm");
    require(desc.pipeline.primitiveTopology === "triangle-list", "real graphics: topology must be triangle-list");
    require(desc.pipeline.vertexCount === 36, "real graphics: vertex count must be 36");
    require(desc.pipeline.depthStencil.depthWriteEnabled === true, "real graphics: depth write must be enabled");
    require(desc.pipeline.depthStencil.depthCompare === "less", "real graphics: depth compare must be less");
    require(desc.draw.indexFormat === "uint32", "real graphics: index format must be uint32");
    require(desc.draw.instanceCount === 1, "real graphics: instance count must be 1");
    require(desc.draw.baseVertex === 0, "real graphics: base vertex must be 0");
    require(desc.draw.firstIndex === 0, "real graphics: first index must be 0");
    require(desc.inputBindings.length === 1, "real graphics: expected one input binding");
    require(desc.bindGroupLayouts.length === 1, "real graphics: expected one bind group layout");
    require(desc.bindGroups.length === 1, "real graphics: expected one bind group");
    require(desc.kernels[0].vertexBufferLayouts.length === 2, "real graphics: expected two vertex buffer layouts");
    require(desc.kernels[0].vertexBufferLayouts[0].arrayStride === 12, "real graphics: first layout stride must be 12");
    require(desc.kernels[0].vertexBufferLayouts[0].stepMode === "vertex", "real graphics: first layout step mode must be vertex");

    console.log("real generated graphics reflection: admitted through loadFaberGraphicsPipeline");
  }

  // Compute admission unchanged (re-verify after graphics additions)
  {
    const kernel = loadFaberKernel({ wgsl, reflection });
    require(kernel.entryName === "add_one", "compute re-verify: kernel entry must be add_one");
    require(kernel.inputBindings.length === 2, "compute re-verify: expected input + runtime-extent bindings");
    require(
      kernel.inputBindings.filter((entry) => entry.kind === "storage-buffer").length === 1,
      "compute re-verify: expected one storage-buffer input binding",
    );
  }

  console.log("product-boundary-check passed");
  console.log("kinds covered: artifact-fetch, reflection, webgpu, device-lost");
  console.log("graphics admission: valid descriptor, missing kernel, wrong stage, missing pipeline, empty vertex inputs, mismatched layouts, malformed draw manifest, bad topology, bad color target, missing depth_stencil, fragment bind group divergence");
  console.log("graphics runtime: all exports verified, onDeviceLost callback, compute re-verified");
  console.log("HV-07B exports: createChunkGraphicsResources, applyChunkResourceReplace, destroyRetiredChunkResources, runChunkGraphicsFrame, chunkResourceCounters");
  console.log(
    "manual browser still required for: window.faberWebGpuProof.ok === true && value === 42",
  );
  console.log(
    "HV-07B lifecycle evidence: node public/src/chunk-resource-lifecycle-check.mjs",
  );
}

main().catch((error) => {
  fail(error?.stack ?? String(error));
});
