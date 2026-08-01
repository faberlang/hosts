/**
 * bootstrap.js — engine session bootstrap + DOM bridge. Exports `initEngine()`,
 * the page entry for corpus demos. Phase 1 move of the session half of corpus
 * host-init.js (DS-S2 P1.2); the rAF/readback/resize machinery lives in
 * engine/frame-scheduler.js, the canvas surface in presentation/canvas.js, and
 * the facts contract in presentation/debug-overlay.js.
 *
 * Transport + WebGPU lifecycle only. Scene facts and camera stay in Faber.
 */

import {
  acquireWebGpuDevice,
  updateGraphicsStorage,
  onDeviceLost,
} from "../backend/webgpu-runtime.js";
import {
  loadGreyboxPipeline,
  initGreyboxRenderer,
  initGreyboxSceneRenderer,
  renderGreyboxFrame,
  renderGreyboxFrameWithSamples,
  renderGreyboxSceneFrame,
  mapPixelBuffers,
  resizeGreyboxRenderer,
} from "../engine/engine.js";
import { parseSceneGeometryBlob } from "../engine/scene-extractor.js";
import { createFrameScheduler } from "../engine/frame-scheduler.js";
import {
  findTrigaCanvas,
  configureWebGpuContext,
} from "../presentation/canvas.js";
import {
  setFact,
  waitForSceneGeometry,
  readTransformPayload,
} from "../presentation/debug-overlay.js";

const TRANSFORM_BYTE_LEN = 128; // 32 f32 × 4 bytes

/**
 * Initialize the WebGPU host session with the greybox scene renderer.
 * @returns {Promise<object>}
 */
export async function initEngine() {
  const { device } = await acquireWebGpuDevice();
  setFact("data-device-status", "active");

  const canvas = findTrigaCanvas();
  if (!canvas) {
    throw new Error("bootstrap: canvas not found (.triga-canvas)");
  }

  const context = configureWebGpuContext(canvas, device);
  const initialWidth = canvas.width;
  const initialHeight = canvas.height;

  // ── Load greybox pipeline (triga-lit artifacts from public/) ───────────

  let pipelinePack = null;
  let pipelineLoaded = false;

  try {
    pipelinePack = await loadGreyboxPipeline(device);
    pipelineLoaded = true;
    setFact("data-pipeline-status", "loaded");
  } catch (err) {
    console.warn("bootstrap: corpus pipeline load failed", err);
    setFact("data-pipeline-status", "failed");
    setFact("data-pipeline-error", err.message);
    setFact("data-render-status", "pipeline-load-failed");
    setFact("data-render-gate", "blocked-pipeline");
  }

  // ── U2 fallback: triangle path while waiting for scene geometry ─────────

  let triangleState = null;
  let readbackSamples = null;
  let renderStatus = "none";

  if (pipelineLoaded && pipelinePack) {
    try {
      triangleState = initGreyboxRenderer(device, pipelinePack.descriptor, context);
      const pixelSamples = [
        { name: "center", x: Math.floor(initialWidth / 2), y: Math.floor(initialHeight / 2) },
        { name: "corner", x: 10, y: 10 },
      ];
      const { pixelBuffers } = renderGreyboxFrameWithSamples(triangleState, pixelSamples);
      readbackSamples = await mapPixelBuffers(pixelBuffers);
      const center = readbackSamples.find((s) => s.name === "center");
      if (center) {
        const isNonClear = center.r > 10 || center.g > 10 || center.b > 10;
        renderStatus = isNonClear ? "verified" : "clear-only";
        setFact("data-pixel-readback", renderStatus);
        setFact("data-pixel-center-hex", center.hex);
        setFact(
          "data-pixel-center-rgba",
          `${center.r},${center.g},${center.b},${center.a}`,
        );
      }
    } catch (err) {
      console.warn("bootstrap: first triangle render / readback failed", err);
      renderStatus = "failed";
      setFact("data-pixel-readback", "failed");
      setFact("data-pixel-readback-error", err.message);
    }
  }

  // ── U4: wait for controller geometry, mount multi-mesh scene ────────────

  let sceneState = null;
  let sceneMounted = false;

  // Lighting data: warm afternoon sun from the west, high angle.
  // 48 bytes / 12 f32: sun_dir(3)+pad, sun_color(3)+pad, ambient(3)+fog_density
  // Fog density is per-demo: <canvas data-fog-density="0.003">, 0 = shader default.
  const fogDensity = Number.parseFloat(canvas.getAttribute("data-fog-density") ?? "0") || 0;
  const lightingData = new Float32Array([
    -0.45, 0.75, 0.35, 0.0,   // sun direction (normalized) + pad
    1.0, 0.92, 0.78, 0.0,     // sun color (warm)
    0.25, 0.28, 0.35, fogDensity, // ambient (cool sky fill) + fog density
  ]);

  if (pipelineLoaded && pipelinePack) {
    try {
      const blob = await waitForSceneGeometry();
      const meshes = parseSceneGeometryBlob(blob);
      sceneState = initGreyboxSceneRenderer(device, pipelinePack, context, meshes, lightingData);

      sceneMounted = true;
      setFact("data-scene-upload", "ok");
      setFact("data-scene-object-count", String(meshes.length));
      setFact("data-render-status", "scene-mounted");
      setFact("data-render-gate", "pending-first-frame");
      // Drop geometry blob from DOM after upload (one-shot mount evidence kept via counts).
      const el = document.querySelector(".triga-facts");
      if (el) el.removeAttribute("data-scene-geometry");
    } catch (err) {
      console.warn("bootstrap: corpus scene geometry mount failed", err);
      setFact("data-scene-upload", "failed");
      setFact("data-scene-upload-error", err.message);
      setFact("data-render-status", "scene-upload-failed");
      setFact("data-render-gate", "blocked-geometry");
    }
  }

  // ── Transform storage (U4 bridge / readback proof) ──────────────────────
  // WebGPU: MapRead may only combine with CopyDst (not Storage).
  // updateGraphicsStorage uses queue.writeBuffer → COPY_DST is enough.
  const transformBuffer = device.createBuffer({
    size: TRANSFORM_BYTE_LEN,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const storageResources = Object.freeze({
    storageBuffers: new Map([
      [0, { buffer: transformBuffer, generation: 0 }],
    ]),
  });

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

  // ── Frame loop and lifecycle ───────────────────────────────────────────

  const renderState = { scene: null, triangle: null };

  function destroyBuffers() {
    try {
      transformBuffer.destroy();
    } catch (_) {
      // already destroyed
    }
  }

  const scheduler = createFrameScheduler({
    device,
    canvas,
    context,
    renderState,
    storageResources,
    storageDescriptor,
    updateStorage: (res, desc, opts) => updateGraphicsStorage(device, res, desc, opts),
    renderScene: (state, transform) => renderGreyboxSceneFrame(state, transform),
    renderTriangle: (state) =>
      renderGreyboxFrame(state, {
        clearValue: { r: 0.45, g: 0.62, b: 0.80, a: 1.0 },
      }),
    readTransform: readTransformPayload,
    onResize: (w, h) => {
      if (renderState.scene) {
        resizeGreyboxRenderer(renderState.scene, w, h);
      } else if (renderState.triangle) {
        renderState.triangle = resizeGreyboxRenderer(renderState.triangle, w, h);
      }
    },
  });

  // Device loss — bounded, no uncaught errors
  onDeviceLost(device, (info) => {
    setFact("data-device-status", "lost");
    setFact("data-device-loss-reason", info.reason || "unknown");
    scheduler.destroy();
    destroyBuffers();
  });

  device.lost.then((info) => {
    setFact("data-device-status", "lost");
    scheduler.destroy();
    destroyBuffers();
    return { reason: info.reason, message: info.message };
  });

  device.addEventListener("uncapturederror", (event) => {
    console.error("bootstrap: uncaptured WebGPU error", event.error);
  });

  renderState.scene = sceneState;
  renderState.triangle = triangleState;
  scheduler.start();

  return Object.freeze({
    device,
    pipelineLoaded,
    sceneMounted,
    renderStatus,
    readbackSamples,
    greyboxRenderState: renderState.scene || renderState.triangle,
    updateGraphicsStorage: (res, desc, opts) =>
      updateGraphicsStorage(device, res, desc, opts),
    resize: () => scheduler.resize(),
    destroy: () => {
      scheduler.destroy();
      destroyBuffers();
    },
  });
}
