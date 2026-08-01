/**
 * engine.js — greybox engine facade for Triga corpus demos.
 *
 * Phase 1 move of corpus/_host/public/greybox-host.js (DS-S2 P1.2):
 *   - greybox facade (U2 triangle + U4 multi-mesh scene) preserved as-is;
 *   - `parseSceneGeometryBlob` moved to ./scene-extractor.js;
 *   - mesh upload moved to ./resource-manager.js;
 *   - `buildDescriptorFromReflection` RETIRED (deleted, not moved). Admission
 *     is the only path: the demos route through
 *     contract/artifact-admission.js (loadFaberGraphicsPipeline) once the
 *     triga-lit artifacts are regenerated through radix.
 *
 * JS owns transport + WebGPU lifecycle only. Simulation stays in Faber.
 */

import {
  createGraphicsResources,
  runGraphicsFrame,
  runGraphicsFrameWithTexture,
  mapPixelBuffers,
  updateGraphicsStorage,
  replaceDepthTextureOnResize,
} from "../backend/webgpu-runtime.js";
import {
  FaberKernelContractError,
  loadFaberGraphicsPipeline,
} from "../contract/artifact-admission.js";
import { createMeshGpuEntry } from "./resource-manager.js";

// ── Test geometry: one colored triangle (U2) ───────────────────────────────

const TRIANGLE_VERTICES = new Float32Array([
  // position (x,y,z)      // normal (nx,ny,nz)    // color (r,g,b)
  -0.5, -0.5, 0.0,          0.0, 0.0, 1.0,         1.0, 0.0, 0.0,
   0.5, -0.5, 0.0,          0.0, 0.0, 1.0,         0.0, 1.0, 0.0,
   0.0,  0.5, 0.0,          0.0, 0.0, 1.0,         0.0, 0.0, 1.0,
]);

const TRIANGLE_INDICES = new Uint32Array([0, 1, 2]);

const IDENTITY_TRANSFORM = new Float32Array([
  1, 0, 0, 0,   0, 1, 0, 0,   0, 0, 1, 0,   0, 0, 0, 1,
  1, 0, 0, 0,   0, 1, 0, 0,   0, 0, 1, 0,   0, 0, 0, 1,
]);

const IDENTITY_MODEL = new Float32Array([
  1, 0, 0, 0,   0, 1, 0, 0,   0, 0, 1, 0,   0, 0, 0, 1,
]);

/**
 * Greybox draw manifest. The scene path draws one drawIndexed per mesh with
 * each mesh's own indexCount; the manifest indexCount is only the admission
 * bounds anchor (uint32 matches the Uint32Array mesh upload path).
 * @param {number} indexCount
 */
function greyboxDrawManifest(indexCount) {
  return Object.freeze({
    index_format: "uint32",
    instance_count: 1,
    base_vertex: 0,
    first_index: 0,
    index_count: indexCount,
  });
}

// ── Admission ──────────────────────────────────────────────────────────────

/**
 * Fetch compiled WGSL and reflection, build an admitted graphics descriptor.
 *
 * Resolves against this module URL, not the document URL (the page lives at
 * /pages/index.html; artifacts are copied to the runtime /public/ dir by the
 * corpus run.sh, so from public/src/engine/ the triga-lit artifacts resolve
 * two levels up).
 *
 * NOTE — placeholder artifact state: hosts public/generated/triga-lit.* are
 * byte-identical copies of the pre-radix kernel.wgsl + reflection.json (old
 * reflection format) until the radix regeneration lands (gated on 80 Stage
 * 4–5 graphics-MIR completeness). loadFaberGraphicsPipeline requires
 * schema_version / target / launch.webgpu_adapter, so the placeholder
 * reflection is REJECTED by admission with a typed FaberKernelContractError
 * and no draw happens. That is the documented gated state, not a defect.
 *
 * @param {GPUDevice} device
 * @returns {Promise<{ descriptor: object, wgsl: string, reflection: object }>}
 */
export async function loadGreyboxPipeline(device) {
  const wgslUrl = new URL("../../triga-lit.wgsl", import.meta.url);
  const reflectionUrl = new URL("../../triga-lit-reflection.json", import.meta.url);
  const [wgslResp, reflectionResp] = await Promise.all([
    fetch(wgslUrl),
    fetch(reflectionUrl),
  ]);

  if (!wgslResp.ok) {
    throw new FaberKernelContractError(
      "fetch",
      `failed to fetch triga-lit.wgsl (${wgslUrl.href}): ${wgslResp.status}`,
      "artifact-fetch",
    );
  }
  if (!reflectionResp.ok) {
    throw new FaberKernelContractError(
      "fetch",
      `failed to fetch triga-lit-reflection.json (${reflectionUrl.href}): ${reflectionResp.status}`,
      "artifact-fetch",
    );
  }

  const wgsl = await wgslResp.text();
  const reflection = await reflectionResp.json();
  const descriptor = loadFaberGraphicsPipeline({
    wgsl,
    reflection,
    drawManifest: greyboxDrawManifest(TRIANGLE_INDICES.length),
  });

  return Object.freeze({ descriptor, wgsl, reflection });
}

/**
 * Create GPU resources for the greybox pipeline with a test triangle (U2).
 *
 * @param {GPUDevice} device
 * @param {object} descriptor
 * @param {GPUCanvasContext} canvasContext
 * @returns {object} frozen renderState
 */
export function initGreyboxRenderer(device, descriptor, canvasContext) {
  const payloads = {
    vertexBuffers: [
      { slot: 0, data: TRIANGLE_VERTICES },
    ],
    indexData: TRIANGLE_INDICES,
    storageData: {
      transform: IDENTITY_TRANSFORM,
      lighting: new Float32Array([
        -0.45, 0.75, 0.35, 0.0,
        1.0, 0.92, 0.78, 0.0,
        0.25, 0.28, 0.35, 0.0,
      ]),
    },
  };

  const resources = createGraphicsResources(device, descriptor, payloads, canvasContext);

  const frameState = {
    submittedFrameCount: 0,
  };

  return Object.freeze({
    device,
    context: canvasContext,
    descriptor,
    resources,
    frameState,
    mode: "triangle",
  });
}

/**
 * Create multi-mesh scene renderer (U4): one draw call per object.
 *
 * Meshes arrive in world space; the published transform payload carries the
 * per-object model (identity for now) plus view-projection. No spawn or pose
 * correction happens here — that is Faber's to publish.
 *
 * The scene descriptor is re-admitted through loadFaberGraphicsPipeline with
 * the first mesh's index count as the draw anchor.
 *
 * @param {GPUDevice} device
 * @param {object} pipelinePack - from loadGreyboxPipeline ({ descriptor, wgsl, reflection })
 * @param {GPUCanvasContext} canvasContext
 * @param {Array<{ name: string, role: string, vertices: Float32Array, indices: Uint32Array }>} meshes
 * @returns {object} renderState (mutable resources holder for resize)
 */
export function initGreyboxSceneRenderer(device, pipelinePack, canvasContext, meshes, lightingData) {
  if (!Array.isArray(meshes) || meshes.length === 0) {
    throw new Error("initGreyboxSceneRenderer: meshes required");
  }

  const first = meshes[0];
  const descriptor = loadFaberGraphicsPipeline({
    wgsl: pipelinePack.wgsl,
    reflection: pipelinePack.reflection,
    drawManifest: greyboxDrawManifest(first.indices.length),
  });

  const payloads = {
    vertexBuffers: [{ slot: 0, data: first.vertices }],
    indexData: first.indices,
    storageData: {
      transform: IDENTITY_TRANSFORM,
      lighting: lightingData || new Float32Array(12),
    },
  };

  let resources = createGraphicsResources(device, descriptor, payloads, canvasContext);

  const meshEntries = meshes.map((m, i) => {
    if (i === 0) {
      // Reuse buffers from createGraphicsResources for mesh 0.
      return {
        name: m.name,
        role: m.role || "static",
        vertexBuffer: resources.vertexBuffers[0].buffer,
        indexBuffer: resources.indexBuffer,
        indexCount: m.indices.length,
      };
    }
    return createMeshGpuEntry(device, m);
  });

  return {
    device,
    context: canvasContext,
    descriptor,
    get resources() {
      return resources;
    },
    set resources(next) {
      resources = next;
    },
    frameState: { submittedFrameCount: 0 },
    meshes: meshEntries,
    mode: "scene",
    objectCount: meshEntries.length,
  };
}

/**
 * Render one greybox frame (triangle path).
 */
export function renderGreyboxFrame(renderState, options = {}) {
  const { device, context, descriptor, resources, frameState } = renderState;
  runGraphicsFrame(device, context, resources, descriptor, frameState, {
    clearValue: options.clearValue ?? { r: 0.02, g: 0.06, b: 0.07, a: 1.0 },
    recordSubmit: options.recordSubmit ?? false,
  });
}

/**
 * Render multi-mesh scene: one drawIndexed per object, all sharing the
 * published model + view-proj (model is identity until a demo publishes
 * real poses).
 *
 * Per-object model via multi-pass (storage buffer rewritten between passes;
 * first pass clears, later passes load).
 *
 * @param {object} renderState - from initGreyboxSceneRenderer
 * @param {Float32Array} transform32 - model(16) + view-proj(16)
 * @param {{ clearValue?: GPUColor }} [options]
 */
export function renderGreyboxSceneFrame(renderState, transform32, options = {}) {
  const { device, context, descriptor, meshes } = renderState;
  const resources = renderState.resources;
  const clearValue = options.clearValue ?? { r: 0.45, g: 0.62, b: 0.80, a: 1.0 };

  if (!(transform32 instanceof Float32Array) || transform32.length < 32) {
    throw new Error("renderGreyboxSceneFrame: transform32 must be Float32Array(32)");
  }

  const model = transform32.subarray(0, 16);
  const viewProj = transform32.subarray(16, 32);

  const combined = new Float32Array(32);
  combined.set(model, 0);
  combined.set(viewProj, 16);

  // One canvas texture, one MSAA color view, and one depth view for the whole
  // frame; the passes differ only in load op and which mesh they draw. Each
  // pass resolves the multisampled target into the canvas texture.
  const textureView = context.getCurrentTexture().createView();
  const depthView = resources.depthTexture.createView();
  const colorView = resources.msaaTexture ? resources.msaaTexture.createView() : textureView;
  const resolveTarget = resources.msaaTexture ? textureView : undefined;

  for (let i = 0; i < meshes.length; i++) {
    const mesh = meshes[i];

    updateGraphicsStorage(device, resources, descriptor, {
      resourceIndex: 0,
      data: combined,
      sourceName: "transform",
    });

    const commandEncoder = device.createCommandEncoder();
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: colorView,
          resolveTarget,
          clearValue,
          loadOp: i === 0 ? "clear" : "load",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: depthView,
        depthClearValue: 1.0,
        depthLoadOp: i === 0 ? "clear" : "load",
        depthStoreOp: "store",
      },
    });

    renderPass.setPipeline(resources.pipeline);
    for (const group of resources.bindGroups) {
      renderPass.setBindGroup(group.bindGroupIndex, group.bindGroup);
    }
    renderPass.setVertexBuffer(0, mesh.vertexBuffer);
    renderPass.setIndexBuffer(mesh.indexBuffer, "uint32", 0);
    renderPass.drawIndexed(mesh.indexCount, 1, 0, 0, 0);
    renderPass.end();
    device.queue.submit([commandEncoder.finish()]);
  }

  renderState.frameState.submittedFrameCount =
    (renderState.frameState.submittedFrameCount ?? 0) + 1;

  return Object.freeze({
    draw_count: meshes.length,
    frame_index: renderState.frameState.submittedFrameCount,
  });
}

/**
 * Render one frame AND capture pixel samples (U2 triangle path).
 */
export function renderGreyboxFrameWithSamples(renderState, pixelSamples, options = {}) {
  const { device, context, descriptor, resources, frameState } = renderState;
  return runGraphicsFrameWithTexture(device, context, resources, descriptor, frameState, {
    clearValue: options.clearValue ?? { r: 0.02, g: 0.06, b: 0.07, a: 1.0 },
    pixelSamples,
  });
}

/**
 * Write model and view-projection data to the transform storage buffer.
 */
export function updateGreyboxTransform(renderState, modelData, viewProjData) {
  const { device, descriptor } = renderState;
  const resources = renderState.resources;
  const combined = new Float32Array(32);

  if (modelData) {
    combined.set(modelData, 0);
  } else {
    combined.set(IDENTITY_MODEL, 0);
  }

  if (viewProjData) {
    combined.set(viewProjData, 16);
  } else {
    combined.set(IDENTITY_MODEL, 16);
  }

  updateGraphicsStorage(device, resources, descriptor, {
    resourceIndex: 0,
    data: combined,
    sourceName: "transform",
  });
}

/**
 * Resize depth texture for scene or triangle render state.
 */
export function resizeGreyboxRenderer(renderState, width, height) {
  const { device } = renderState;
  const next = replaceDepthTextureOnResize(device, renderState.resources, width, height);
  if (renderState.mode === "scene") {
    renderState.resources = next;
  } else {
    // triangle path: frozen state — return new state
    return Object.freeze({
      ...renderState,
      resources: next,
    });
  }
  return renderState;
}

export { mapPixelBuffers, replaceDepthTextureOnResize };
