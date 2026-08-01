/**
 * frame-scheduler.js — rAF frame loop, transform readback phases, and resize
 * ordering. Phase 1 move of the loop machinery from corpus host-init.js
 * (DS-S2 P1.2). The session (device, canvas, pipeline load, scene mount,
 * storage mirror) lives in product/bootstrap.js; this module owns when frames
 * happen.
 *
 * Frame ordering per tick: read controller transform → render scene (or the
 * U2 triangle fallback) → mirror the transform into the MAP_READ buffer for
 * the readback proof. Resize ordering: canvas backing size → context
 * re-configure → engine texture resize.
 */

import {
  backingSize,
  resizeCanvasToBackingSize,
} from "../presentation/canvas.js";
import { setFact } from "../presentation/debug-overlay.js";

const TRANSFORM_READBACK_PHASE_WAITING = 0;
const TRANSFORM_READBACK_PHASE_SNAPSHOT = 1;
const TRANSFORM_READBACK_PHASE_VERIFIED = 2;
const TRANSFORM_READBACK_PHASE_FAILED = -1;

/**
 * Create the frame loop for an engine session.
 *
 * @param {object} options
 * @param {GPUDevice} options.device
 * @param {HTMLCanvasElement} options.canvas
 * @param {GPUCanvasContext} options.context
 * @param {{ scene: object|null, triangle: object|null }} options.renderState
 *   mutable holder of the live render states (resize swaps the triangle state)
 * @param {object} options.storageResources — MAP_READ mirror resources
 *   ({ storageBuffers: Map<number, { buffer, generation }> })
 * @param {object} options.storageDescriptor — mirror bind-group descriptor
 * @param {(res: object, desc: object, opts: object) => object} options.updateStorage
 * @param {(sceneState: object, transform: Float32Array) => void} options.renderScene
 * @param {(triangleState: object) => void} options.renderTriangle
 * @param {() => Float32Array|null} options.readTransform
 * @param {() => void} [options.onSceneFirstRender]
 * @param {(width: number, height: number) => void} options.onResize
 * @returns {{ start: () => void, resize: () => void, destroy: () => void }}
 */
export function createFrameScheduler(options) {
  const {
    device,
    canvas,
    context,
    renderState,
    storageResources,
    storageDescriptor,
    updateStorage,
    renderScene,
    renderTriangle,
    readTransform,
    onSceneFirstRender,
    onResize,
  } = options;

  let frameId = null;
  let running = true;
  let frameCount = 0;
  let readbackSnapshot = null;
  let readbackPhase = TRANSFORM_READBACK_PHASE_WAITING;
  let readbackBusy = false;
  let sceneRendered = false;
  let lastTransform = null;
  let resizeObserver = null;
  let resizeListenerBound = false;

  function stopLoop() {
    running = false;
    if (frameId !== null) {
      cancelAnimationFrame(frameId);
      frameId = null;
    }
  }

  async function doReadback() {
    if (!running || readbackBusy || readbackPhase < 0 || readbackPhase >= TRANSFORM_READBACK_PHASE_VERIFIED) {
      return;
    }
    // While this buffer is map-pending or mapped, queue.writeBuffer against it
    // is a validation error, so the frame loop must skip its mirror write.
    readbackBusy = true;
    try {
      const mirror = storageResources.storageBuffers.get(0)?.buffer;
      if (!mirror) {
        throw new Error("frame-scheduler: missing MAP_READ mirror buffer at resourceIndex 0");
      }
      await device.queue.onSubmittedWorkDone();
      await mirror.mapAsync(GPUMapMode.READ);
      const mapped = new Float32Array(mirror.getMappedRange());
      const copy = new Float32Array(mapped);
      mirror.unmap();

      if (readbackPhase === TRANSFORM_READBACK_PHASE_WAITING) {
        readbackSnapshot = copy;
        readbackPhase = TRANSFORM_READBACK_PHASE_SNAPSHOT;
      } else if (readbackPhase === TRANSFORM_READBACK_PHASE_SNAPSHOT) {
        let changed = false;
        for (let i = 0; i < 32; i++) {
          if (copy[i] !== readbackSnapshot[i]) {
            changed = true;
            break;
          }
        }
        readbackPhase = changed
          ? TRANSFORM_READBACK_PHASE_VERIFIED
          : TRANSFORM_READBACK_PHASE_SNAPSHOT;
        setFact("data-readback-proof", changed ? "verified" : "unchanged");
      }
    } catch (_) {
      readbackPhase = TRANSFORM_READBACK_PHASE_FAILED;
      setFact("data-readback-proof", "failed");
    } finally {
      readbackBusy = false;
    }
  }

  function frameLoop() {
    if (!running) return;

    try {
      const floats = readTransform();
      if (floats) lastTransform = floats;

      if (renderState.scene) {
        // U4 path: multi-draw scene with per-object model + view-proj. A payload
        // that fails to parse must not drop the scene back to the U2 triangle,
        // so hold the last good transform until a new one arrives.
        const transform = floats ?? lastTransform;
        if (transform) {
          renderScene(renderState.scene, transform);
          if (!sceneRendered) {
            sceneRendered = true;
            setFact("data-render-status", "live-direct-webgpu");
            setFact("data-render-gate", "open");
            if (typeof onSceneFirstRender === "function") onSceneFirstRender();
          }
          if (!readbackBusy) {
            // Mirror transform into MAP_READ buffer for readback proof
            updateStorage(storageResources, storageDescriptor, {
              resourceIndex: 0,
              data: transform,
            });
            frameCount++;
            if (frameCount >= 2 && readbackPhase < TRANSFORM_READBACK_PHASE_VERIFIED) {
              doReadback();
            }
          }
        }
      } else if (renderState.triangle) {
        // U2 path while the scene geometry has not mounted yet
        renderTriangle(renderState.triangle);
        if (floats && !readbackBusy) {
          updateStorage(storageResources, storageDescriptor, {
            resourceIndex: 0,
            data: floats,
          });
        }
      }
    } catch (err) {
      console.warn("frame-scheduler: frame loop error", err);
    }

    frameId = requestAnimationFrame(frameLoop);
  }

  function resize() {
    const { width: w, height: h } = backingSize(canvas);
    if (w === canvas.width && h === canvas.height) return;

    resizeCanvasToBackingSize(canvas, context, device);
    if (typeof onResize === "function") onResize(w, h);
  }

  function destroy() {
    stopLoop();
    if (resizeObserver) {
      resizeObserver.disconnect();
      resizeObserver = null;
    }
    if (resizeListenerBound) {
      window.removeEventListener("resize", resize);
      resizeListenerBound = false;
    }
  }

  return Object.freeze({
    start() {
      frameId = requestAnimationFrame(frameLoop);
      if (typeof ResizeObserver !== "undefined") {
        resizeObserver = new ResizeObserver(() => resize());
        resizeObserver.observe(canvas);
      } else {
        window.addEventListener("resize", resize);
        resizeListenerBound = true;
      }
    },
    resize,
    destroy,
  });
}
