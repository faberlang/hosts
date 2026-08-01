/**
 * engine.js — renderer/session facade for the shared engine.
 *
 * Phase 1 move of corpus/_host/public/greybox-host.js (DS-S2 P1.2):
 *   - greybox facade (U2 triangle + U4 multi-mesh scene) preserved as-is;
 *   - `parseSceneGeometryBlob` moved to ./scene-extractor.js;
 *   - mesh upload moved to ./resource-manager.js;
 *   - the hand-rolled old-format admission helper is RETIRED (deleted, not
 *     moved). Admission is the only path: the demos route through
 *     contract/artifact-admission.js (loadFaberGraphicsPipeline) once the
 *     triga-lit artifacts are regenerated through radix.
 *
 * Phase 2 (S2 vertical slice, item A): the facade + explicit state machine
 * (`startup → ready → suspended → device-lost → recovering → failed`), the
 * session facade (`createEngineSession`), the standard-material pipeline-cache
 * identity (shared-instance oracle anchor), capability admission BEFORE any
 * draw (contract/capability-admission.js), and the oracle hooks (numeric:
 * transform sequence / draw counts; pixel: deterministic sample capture).
 * The greybox render path stays functional under the facade.
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
  onDeviceLost,
} from "../backend/webgpu-runtime.js";
import {
  FaberKernelContractError,
  loadFaberGraphicsPipeline,
} from "../contract/artifact-admission.js";
import {
  admitCapabilities,
  CapabilityAdmissionError,
} from "../contract/capability-admission.js";
import { createMeshGpuEntry, MeshResourceManager } from "./resource-manager.js";
import { extractSceneRenderItems } from "./scene-extractor.js";
import { createFrameScheduler } from "./frame-scheduler.js";

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

const TRANSFORM_BYTE_LEN = 128; // 32 f32 × 4 bytes
const MAX_TRANSFORM_SEQUENCE = 600;

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

// ══════════════════════════════════════════════════════════════════════════
// Phase 2 — explicit engine state machine + session facade (S2 slice item A)
// ══════════════════════════════════════════════════════════════════════════

/** The engine's explicit state vocabulary (T-F §2). */
export const ENGINE_STATES = Object.freeze([
  "startup",
  "ready",
  "suspended",
  "device-lost",
  "recovering",
  "failed",
]);

/**
 * Legal state transitions. `failed` is terminal (recovery out of a fatal
 * session error is a new session, never a silent resume).
 */
const ENGINE_STATE_TRANSITIONS = Object.freeze({
  startup:      ["ready", "failed"],
  ready:        ["suspended", "device-lost", "failed"],
  suspended:    ["ready", "device-lost", "failed"],
  "device-lost": ["recovering", "failed"],
  recovering:   ["ready", "failed"],
  failed:       [],
});

/** Typed error for an invalid state transition or state assertion. */
export class EngineStateError extends Error {
  /**
   * @param {string} message
   * @param {object} [detail]
   * @param {string} [detail.from]
   * @param {string} [detail.to]
   */
  constructor(message, { from = null, to = null } = {}) {
    super(message);
    this.name = "EngineStateError";
    this.from = from;
    this.to = to;
  }
}

/**
 * Explicit state machine. Pure JS (no DOM, no GPU) so the transition table is
 * directly harness-testable (engine-state-machine-check.mjs).
 */
export class EngineStateMachine {
  /** @param {{ onTransition?: (entry: {from: string, to: string}) => void }} [options] */
  constructor({ onTransition } = {}) {
    this._state = "startup";
    this._onTransition = onTransition;
    this._history = [];
  }

  get state() {
    return this._state;
  }

  /** Frozen history of applied transitions (oracle evidence). */
  get history() {
    return this._history.slice();
  }

  /**
   * Transition to `next`. Invalid transitions throw EngineStateError — the
   * machine never silently ignores a requested transition.
   * @param {string} next
   * @returns {string} the new state
   */
  transition(next) {
    if (!ENGINE_STATES.includes(next)) {
      throw new EngineStateError(
        `engine state ${JSON.stringify(next)} is not in the state vocabulary`,
        { from: this._state, to: next },
      );
    }
    const allowed = ENGINE_STATE_TRANSITIONS[this._state] ?? [];
    if (!allowed.includes(next)) {
      throw new EngineStateError(
        `engine state transition ${this._state} → ${next} is not admitted by the state machine`,
        { from: this._state, to: next },
      );
    }
    const from = this._state;
    this._state = next;
    const entry = Object.freeze({ from, to: next });
    this._history.push(entry);
    if (typeof this._onTransition === "function") {
      this._onTransition(entry);
    }
    return this._state;
  }

  /**
   * Assert the machine is in one of the given states; otherwise throw.
   * @param {...string} states
   */
  assert(...states) {
    if (!states.includes(this._state)) {
      throw new EngineStateError(
        `engine state ${this._state} not in expected states [${states.join(", ")}]`,
        { from: this._state, to: null },
      );
    }
  }

  /** Reset to startup (new session). */
  reset() {
    this._state = "startup";
    this._history = [];
  }
}

/**
 * Deterministic standard-material pipeline-cache identity (the shared-instance
 * oracle anchor). The key is a function of the ADMITTED artifact set only
 * (WGSL source + reflection spine fields + pipeline block), so two demos that
 * admit the same triga-lit artifact set resolve to exactly one identity.
 *
 * @param {object} options
 * @param {string} options.wgsl
 * @param {object} options.reflection
 * @returns {string}
 */
export function computePipelineCacheKey({ wgsl, reflection }) {
  const document = reflection && typeof reflection === "object" ? reflection : {};
  const pipeline = document.pipeline ?? {};
  const seed = [
    wgsl,
    String(document.schema_version),
    String(document.target),
    String(pipeline.primitive_topology),
    (Array.isArray(pipeline.color_target_formats) ? pipeline.color_target_formats : []).join(","),
    String(pipeline.vertex_count),
    String(pipeline.depth_stencil?.depth_compare ?? ""),
    String(pipeline.depth_stencil?.depth_write_enabled ?? ""),
  ].join("\u0000");
  return `standard-material:${fnv1a(seed)}`;
}

function fnv1a(text) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = (hash * 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}

const RENDER_STATUS_BY_STATE = Object.freeze({
  startup: "starting",
  ready: "ready",
  suspended: "suspended",
  "device-lost": "device-lost",
  recovering: "recovering",
  failed: "failed",
});

const DEVICE_STATUS_BY_STATE = Object.freeze({
  startup: "pending",
  ready: "active",
  suspended: "suspended",
  "device-lost": "lost",
  recovering: "recovering",
  failed: "failed",
});

/**
 * Create the engine session facade.
 *
 * Owns: capability admission (typed rejection BEFORE any draw), pipeline
 * admission, scene extraction → render items, resource residency, the frame
 * loop, resize, device loss (device-lost → recovering → ready, or clean
 * failed), and the oracle hooks. The greybox render path (triangle + scene)
 * remains the draw layer underneath.
 *
 * DOM specifics (facts publishing, transform/scene fact polling) are injected
 * so the facade stays node-testable.
 *
 * @param {object} options
 * @param {GPUDevice} options.device
 * @param {object} [options.adapter]
 * @param {HTMLCanvasElement} options.canvas
 * @param {GPUCanvasContext} options.context
 * @param {object} [options.capabilityRequest] - overrides for admitCapabilities
 * @param {Float32Array} [options.lightingData] - 12 f32 lighting uniform
 * @param {(name: string, value: string) => void} options.facts - setFact bridge
 * @param {() => Float32Array|null} options.readTransform - parsed transform payload
 * @param {() => string|null} [options.readTransformText] - raw transform payload text
 * @param {() => Promise<string>} options.waitForSceneBlob - resolves the
 *   data-scene-geometry blob (rejects on timeout)
 * @param {Array<{ name: string, x: number, y: number }>} [options.pixelSamples]
 * @param {{ r: number, g: number, b: number, a: number }} [options.clearValue]
 * @param {(info: { reason: string, message?: string, session: object }) => void} [options.onDeviceLoss]
 *   invoked after the session observes device loss (device-lost state already
 *   entered, loop stopped). Wire recovery here: re-acquire a device and call
 *   `session.recover({ device, adapter, context })`.
 * @returns {object} the session facade
 */
export function createEngineSession({
  device,
  adapter = null,
  canvas,
  context,
  capabilityRequest = {},
  lightingData = null,
  facts = () => {},
  readTransform = () => null,
  readTransformText = () => null,
  waitForSceneBlob = null,
  pixelSamples = null,
  clearValue = { r: 0.45, g: 0.62, b: 0.80, a: 1.0 },
  onDeviceLoss = null,
} = {}) {
  const machine = new EngineStateMachine({ onTransition: publishStateFacts });
  const renderState = { scene: null, triangle: null };
  const residency = new MeshResourceManager();
  const residencyHandles = [];

  // Session device/context are mutable so `recover` can re-drive the session
  // on a freshly acquired device after device loss.
  let currentDevice = device;
  let currentAdapter = adapter;
  let currentContext = context;

  let admitted = null;
  let pipelinePack = null;
  let pipelineCacheKey = null;
  let sceneResources = null;
  let triangleResources = null;
  let renderItems = null;
  let scheduler = null;
  let mirrorBuffer = null;
  let sceneMounted = false;
  let sceneRendered = false;
  let started = false;
  let destroyed = false;
  let sessionError = null;
  let frameCount = 0;
  let lastTransform = null;
  const transformSequence = [];
  const drawCountHistory = [];

  function publishStateFacts({ from, to }) {
    void from;
    facts("data-render-status", RENDER_STATUS_BY_STATE[to]);
    facts("data-device-status", DEVICE_STATUS_BY_STATE[to]);
  }

  function publishSessionError(err) {
    sessionError = err;
    facts("data-render-status", "failed");
    facts("data-render-gate", "blocked-session");
    facts("data-render-error", err?.message ?? String(err));
    if (err instanceof CapabilityAdmissionError) {
      facts("data-render-error-kind", "capability");
      facts("data-capability-admission", "rejected");
    } else if (err instanceof FaberKernelContractError) {
      facts("data-render-error-kind", err.kind ?? "contract");
      // Artifact-admission rejection (stale/missing artifact): the pipeline
      // never loaded — publish the pipeline status facts (deterministic
      // failure 2 vocabulary).
      if (!pipelinePack) {
        facts("data-pipeline-status", "failed");
        facts("data-pipeline-error", err?.message ?? String(err));
      }
    } else {
      facts("data-render-error-kind", "engine");
    }
  }

  function recordTransform(transform) {
    lastTransform = transform;
    if (transformSequence.length >= MAX_TRANSFORM_SEQUENCE) {
      transformSequence.shift();
    }
    transformSequence.push(new Float32Array(transform));
  }

  function destroyMirrorBuffer() {
    if (mirrorBuffer) {
      try {
        mirrorBuffer.destroy();
      } catch (_) {
        // already destroyed
      }
      mirrorBuffer = null;
    }
  }

  function stopScheduler() {
    if (scheduler) {
      try {
        scheduler.destroy();
      } catch (_) {
        // already stopped
      }
      scheduler = null;
    }
  }

  function renderTriangleFrame() {
    if (!triangleResources) return;
    renderGreyboxFrame(triangleResources, { clearValue });
  }

  function renderSceneFrame(transform) {
    if (!sceneResources) return;
    const result = renderGreyboxSceneFrame(sceneResources, transform, { clearValue });
    frameCount += 1;
    drawCountHistory.push({
      frame: result.frame_index,
      draw_count: result.draw_count,
      scene_object_count: sceneResources.objectCount,
    });
    if (drawCountHistory.length > MAX_TRANSFORM_SEQUENCE) drawCountHistory.shift();
    facts("data-draw-count", String(result.draw_count));
    facts("data-frame-index", String(result.frame_index));
    if (!sceneRendered) {
      sceneRendered = true;
      facts("data-render-status", "live-direct-webgpu");
      facts("data-render-gate", "open");
    }
  }

  function buildScheduler() {
    const storageDescriptor = Object.freeze({
      bindGroups: [
        {
          entries: [
            {
              resourceIndex: 0,
              sourceName: "transform",
              role: "input",
              bufferByteLen: TRANSFORM_BYTE_LEN,
            },
          ],
        },
      ],
    });
    const storageResources = Object.freeze({
      storageBuffers: new Map([[0, { buffer: mirrorBuffer, generation: 0 }]]),
    });

    scheduler = createFrameScheduler({
      device: currentDevice,
      canvas,
      context: currentContext,
      renderState,
      storageResources,
      storageDescriptor,
      updateStorage: (res, desc, opts) => updateGraphicsStorage(currentDevice, res, desc, opts),
      renderScene: (state, transform) => renderSceneFrame(transform),
      renderTriangle: renderTriangleFrame,
      readTransform: () => {
        const floats = readTransform();
        if (floats) recordTransform(floats);
        return floats ?? lastTransform;
      },
      onSceneFirstRender: () => {},
      onResize: (w, h) => {
        if (renderState.scene) {
          resizeGreyboxRenderer(renderState.scene, w, h);
        } else if (renderState.triangle) {
          renderState.triangle = resizeGreyboxRenderer(renderState.triangle, w, h);
        }
        facts("data-canvas-size", `${w}x${h}`);
      },
    });
  }

  function buildMirrorBuffer() {
    destroyMirrorBuffer();
    mirrorBuffer = currentDevice.createBuffer({
      size: TRANSFORM_BYTE_LEN,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  function registerDeviceLoss() {
    onDeviceLost(currentDevice, async (info) => {
      if (destroyed) return;
      handleDeviceLoss(info.reason);
      if (typeof onDeviceLoss === "function") {
        try {
          await onDeviceLoss({ reason: info.reason, message: info.message, session });
        } catch (_) {
          // recovery wiring failed — the session is already in device-lost;
          // bootstrap decides whether to fail the session.
        }
      }
    });
  }

  function registerUncapturedError() {
    if (typeof currentDevice.addEventListener !== "function") return;
    currentDevice.addEventListener("uncapturederror", (event) => {
      console.error("engine: uncaptured WebGPU error", event.error);
    });
  }

  /**
   * Admit capabilities, load the pipeline through artifact-admission, mount
   * the scene from DOM facts, and start the frame loop. Typed rejections
   * leave the session in `failed` with structured diagnostics — never a
   * silent fallback.
   */
  async function start() {
    if (started) return;
    started = true;

    try {
      // 1. Capability admission — BEFORE any draw.
      admitted = admitCapabilities({
        device: currentDevice,
        adapter: currentAdapter,
        requested: capabilityRequest,
        artifact: "triga-lit",
      });
      facts("data-capability-admission", "ok");

      // 2. Pipeline admission (artifact-admission path).
      pipelinePack = await loadGreyboxPipeline(currentDevice);
      pipelineCacheKey = computePipelineCacheKey({
        wgsl: pipelinePack.wgsl,
        reflection: pipelinePack.reflection,
      });
      facts("data-pipeline-status", "loaded");
      facts("data-pipeline-cache-key", pipelineCacheKey);

      // 3. U2 triangle fallback while the controller publishes the scene
      //    (existing greybox behavior; only when admission succeeded).
      triangleResources = initGreyboxRenderer(currentDevice, pipelinePack.descriptor, currentContext);
      renderState.triangle = triangleResources;
      if (Array.isArray(pixelSamples)) {
        try {
          const { pixelBuffers } = renderGreyboxFrameWithSamples(
            triangleResources,
            pixelSamples,
            { clearValue },
          );
          const samples = await mapPixelBuffers(pixelBuffers);
          const center = samples.find((s) => s.name === "center");
          const nonClear = center && (center.r > 10 || center.g > 10 || center.b > 10);
          facts("data-pixel-readback", nonClear ? "verified" : "clear-only");
          if (center) facts("data-pixel-center-hex", center.hex);
        } catch (err) {
          console.warn("engine: first triangle render / readback failed", err);
          facts("data-pixel-readback", "failed");
          facts("data-pixel-readback-error", err.message);
        }
      }

      // 4. Scene extraction → render items (DOM facts through the admitted
      //    descriptor; the extractor rejects cleanly if the reflection was
      //    not admitted or the facts do not conform).
      //
      // A scene-mount failure after a successful admission is NON-fatal: the
      // greybox triangle keeps rendering (existing waiting behavior) and the
      // typed rejection is published as facts. The extractor itself never
      // silently falls back — it throws; the session records and continues.
      if (typeof waitForSceneBlob === "function") {
        try {
          const sceneBlob = await waitForSceneBlob();
          const extracted = extractSceneRenderItems({
            sceneBlob,
            transformText: readTransformText(),
            descriptor: pipelinePack.descriptor,
          });
          renderItems = extracted.items;
          if (extracted.transform) recordTransform(extracted.transform);

          // 5. Mount the greybox scene renderer on the extracted meshes and
          //    register its mesh entries into resource residency.
          const meshes = extracted.meshes;
          sceneResources = initGreyboxSceneRenderer(
            currentDevice,
            pipelinePack,
            currentContext,
            meshes,
            lightingData,
          );
          for (const entry of sceneResources.meshes) {
            residencyHandles.push(residency.trackExisting(entry));
          }
          sceneMounted = true;
          renderState.scene = sceneResources;
          facts("data-scene-upload", "ok");
          facts("data-scene-object-count", String(sceneResources.objectCount));
          facts("data-render-gate", "pending-first-frame");
        } catch (err) {
          // Typed rejection published; triangle fallback stays live.
          console.warn("engine: scene mount rejected", err);
          facts("data-scene-upload", "failed");
          facts("data-scene-upload-error", err?.message ?? String(err));
          facts("data-render-gate", "blocked-geometry");
        }
      }

      // 6. Frame loop + lifecycle.
      buildMirrorBuffer();
      buildScheduler();
      registerDeviceLoss();
      registerUncapturedError();
      scheduler.start();
      machine.transition("ready");
    } catch (err) {
      try {
        machine.transition("failed");
      } catch (_) {
        // already terminal
      }
      publishSessionError(err);
      throw err;
    }
  }

  /**
   * Suspend the session (e.g. page hidden). ready → suspended.
   */
  function suspend() {
    machine.assert("ready");
    stopScheduler();
    machine.transition("suspended");
  }

  /**
   * Resume a suspended session. suspended → ready.
   */
  function resume() {
    machine.assert("suspended");
    buildMirrorBuffer();
    buildScheduler();
    scheduler.start();
    machine.transition("ready");
  }

  /**
   * Recover after device loss: device-lost → recovering → ready by
   * re-admitting, re-loading, and re-mounting on a fresh device/context.
   * Clean failure on any step (device-lost/recovering → failed).
   *
   * @param {object} [options]
   * @param {GPUDevice} [options.device]
   * @param {object} [options.adapter]
   * @param {GPUCanvasContext} [options.context]
   */
  async function recover({ device: nextDevice = currentDevice, adapter: nextAdapter = currentAdapter, context: nextContext = currentContext } = {}) {
    machine.assert("device-lost", "recovering");
    try {
      machine.transition("recovering");
      // Tear down the old device's session pieces.
      stopScheduler();
      destroyMirrorBuffer();
      // Re-drive the session on the fresh device.
      currentDevice = nextDevice;
      currentAdapter = nextAdapter;
      currentContext = nextContext;
      started = false;
      sceneMounted = false;
      sceneRendered = false;
      renderState.scene = null;
      renderState.triangle = null;
      sceneResources = null;
      triangleResources = null;
      renderItems = null;
      // The previous device's buffers died with it; clear the stale residency
      // registrations so destroy() never touches invalid handles.
      residencyHandles.length = 0;
      await start();
    } catch (err) {
      try {
        machine.transition("failed");
      } catch (_) {
        // already terminal
      }
      publishSessionError(err);
    }
  }

  /**
   * Handle an observed device loss from outside (test switch / driver
   * notification). Publishes the device-lost facts and stops the loop.
   */
  function handleDeviceLoss(reason = "unknown") {
    facts("data-device-status", "lost");
    facts("data-device-loss-reason", String(reason));
    try {
      machine.transition("device-lost");
    } catch (_) {
      // terminal
    }
    stopScheduler();
  }

  /**
   * Pixel oracle hook: render the current greybox state and capture
   * deterministic pixel samples. Typed rejection when no draw layer is
   * available (nothing drawn, no guess).
   * @param {Array<{ name: string, x: number, y: number }>} samples
   * @returns {Promise<Array<object>>} mapped RGBA samples
   */
  async function capturePixels(samples) {
    const state = renderState.scene || renderState.triangle;
    if (!state) {
      throw new FaberKernelContractError(
        "engine.capturePixels",
        "no draw layer mounted — cannot capture pixels before admission succeeds",
        "product",
      );
    }
    const { pixelBuffers } = renderGreyboxFrameWithSamples(state, samples, { clearValue });
    return mapPixelBuffers(pixelBuffers);
  }

  /**
   * Numeric oracle hooks (scaffold): the world-transform sequence and the
   * draw-count history published by the session so far.
   */
  function numericOracle() {
    return Object.freeze({
      transformSequence: transformSequence.map((t) => t.slice()),
      drawCounts: drawCountHistory.map((entry) => ({ ...entry })),
      scene_object_count: renderState.scene?.objectCount ?? (renderItems ? renderItems.length : 0),
    });
  }

  function destroy() {
    if (destroyed) return;
    destroyed = true;
    stopScheduler();
    destroyMirrorBuffer();
    if (residencyHandles.length > 0) {
      // Retire + best-effort destroy of the registered scene entries.
      for (const handle of residencyHandles) {
        try {
          residency.retire(handle);
        } catch (_) {
          // already retired
        }
      }
      residency.destroyRetired(currentDevice).catch(() => {});
    }
    renderState.scene = null;
    renderState.triangle = null;
    try {
      machine.transition("failed");
    } catch (_) {
      // terminal
    }
  }

  const session = Object.freeze({
    // state machine
    get state() {
      return machine.state;
    },
    machine,
    // admission results
    get capabilities() {
      return admitted;
    },
    get pipelineCacheKey() {
      return pipelineCacheKey;
    },
    get pipelineLoaded() {
      return pipelinePack !== null;
    },
    get sceneMounted() {
      return sceneMounted;
    },
    get renderItems() {
      return renderItems;
    },
    get sessionError() {
      return sessionError;
    },
    // session lifecycle
    start,
    suspend,
    resume,
    recover,
    handleDeviceLoss,
    resize: () => {
      if (scheduler) scheduler.resize();
    },
    destroy,
    // oracle hooks (numeric + pixel vocabulary, scaffolded for the S2 suite)
    capturePixels,
    numericOracle,
  });

  return session;
}
