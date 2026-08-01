/**
 * bootstrap.js — engine session bootstrap + DOM bridge. Exports `initEngine()`,
 * the page entry for corpus demos. Phase 1 move of the session half of corpus
 * host-init.js (DS-S2 P1.2).
 *
 * Phase 2 (S2 slice, item D): the session itself now lives in the engine
 * facade (engine/engine.js `createEngineSession`) — capability admission,
 * pipeline admission, scene extraction → render items, resource residency,
 * the frame loop, resize, device loss, and the oracle hooks. This module is
 * the thin page bridge: acquire the device, find + configure the canvas,
 * inject the DOM facts/transform bridges, wire device-loss recovery and page
 * visibility (suspend/resume), and start the session. The state machine is
 * published to the DOM facts (`data-render-status` / `data-device-status`,
 * the examples/triga-drift-city precedent).
 *
 * Transport + WebGPU lifecycle only. Scene facts and camera stay in Faber.
 */

import {
  acquireWebGpuDevice,
  updateGraphicsStorage,
} from "../backend/webgpu-runtime.js";
import { createEngineSession } from "../engine/engine.js";
import {
  setFact,
  waitForSceneGeometry,
  readTransformPayload,
  readTransformPayloadText,
} from "../presentation/debug-overlay.js";
import {
  findTrigaCanvas,
  configureWebGpuContext,
} from "../presentation/canvas.js";

/**
 * Initialize the WebGPU host session through the shared engine facade.
 * @returns {Promise<object>} the session facade
 */
export async function initEngine() {
  const { device, adapter } = await acquireWebGpuDevice();
  setFact("data-device-status", "pending");

  const canvas = findTrigaCanvas();
  if (!canvas) {
    throw new Error("bootstrap: canvas not found (.triga-canvas)");
  }

  const context = configureWebGpuContext(canvas, device);
  const initialWidth = canvas.width;
  const initialHeight = canvas.height;

  // Lighting data: warm afternoon sun from the west, high angle.
  // 48 bytes / 12 f32: sun_dir(3)+pad, sun_color(3)+pad, ambient(3)+fog_density
  // Fog density is per-demo: <canvas data-fog-density="0.0032">, 0 = shader default.
  const fogDensity = Number.parseFloat(canvas.getAttribute("data-fog-density") ?? "0") || 0;
  const lightingData = new Float32Array([
    -0.45, 0.75, 0.35, 0.0,   // sun direction (normalized) + pad
    1.0, 0.92, 0.78, 0.0,     // sun color (warm)
    0.25, 0.28, 0.35, fogDensity, // ambient (cool sky fill) + fog density
  ]);

  const session = createEngineSession({
    device,
    adapter,
    canvas,
    context,
    lightingData,
    facts: setFact,
    readTransform: readTransformPayload,
    readTransformText: readTransformPayloadText,
    waitForSceneBlob: waitForSceneGeometry,
    pixelSamples: [
      { name: "center", x: Math.floor(initialWidth / 2), y: Math.floor(initialHeight / 2) },
      { name: "corner", x: 10, y: 10 },
    ],
    onDeviceLoss: async ({ reason, message }) => {
      setFact("data-device-loss-reason", String(reason || "unknown"));
      if (message) setFact("data-device-loss-message", String(message));
      try {
        // Recovery: re-acquire a fresh device, re-configure the canvas, and
        // re-drive the session (device-lost → recovering → ready).
        const fresh = await acquireWebGpuDevice();
        const freshContext = configureWebGpuContext(canvas, fresh.device);
        await session.recover({
          device: fresh.device,
          adapter: fresh.adapter,
          context: freshContext,
        });
      } catch (err) {
        console.error("bootstrap: device-loss recovery failed", err);
        // The session already reports device-lost/failed facts.
      }
    },
  });

  // Page visibility: rAF is throttled when hidden — the explicit suspended
  // state keeps the machine honest about the loop being paused.
  document.addEventListener("visibilitychange", () => {
    try {
      if (document.hidden) {
        session.suspend();
      } else {
        session.resume();
      }
    } catch (_) {
      // session not in a suspendable state (failed/terminal) — ignore
    }
  });

  await session.start();
  setFact("data-host-session", "ok");

  return Object.freeze({
    ...session,
    device,
    context,
    updateGraphicsStorage: (res, desc, opts) =>
      updateGraphicsStorage(device, res, desc, opts),
  });
}
