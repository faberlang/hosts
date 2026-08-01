/**
 * canvas.js — presentation canvas surface.
 *
 * Canvas lookup, backing-size (device pixels), `webgpu` context configure,
 * and resize. Phase 1 move of the canvas layer from corpus host-init.js
 * (DS-S2 P1.2).
 */

export const CANVAS_SELECTOR = ".triga-canvas";

const CANVAS_FORMAT = "bgra8unorm";
const CANVAS_ALPHA_MODE = "opaque";

function canvasUsage() {
  return GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC;
}

/** The demo canvas element (.triga-canvas). */
export function findTrigaCanvas() {
  return document.querySelector(CANVAS_SELECTOR);
}

/**
 * Backing-store size for a canvas laid out by CSS, in device pixels.
 * @param {HTMLCanvasElement} canvas
 * @returns {{ width: number, height: number }}
 */
export function backingSize(canvas) {
  const ratio = globalThis.devicePixelRatio || 1;
  const cssWidth = canvas.clientWidth || canvas.width || 960;
  const cssHeight = canvas.clientHeight || canvas.height || 540;
  return {
    width: Math.max(1, Math.round(cssWidth * ratio)),
    height: Math.max(1, Math.round(cssHeight * ratio)),
  };
}

/**
 * Configure the canvas backing store and WebGPU context at the current
 * backing size. Returns the context.
 * @param {HTMLCanvasElement} canvas
 * @param {GPUDevice} device
 * @returns {GPUCanvasContext}
 */
export function configureWebGpuContext(canvas, device) {
  const { width, height } = backingSize(canvas);
  canvas.width = width;
  canvas.height = height;

  const context = canvas.getContext("webgpu");
  if (!context) {
    throw new Error("canvas: WebGPU canvas context unavailable");
  }

  context.configure({
    device,
    format: CANVAS_FORMAT,
    alphaMode: CANVAS_ALPHA_MODE,
    usage: canvasUsage(),
  });
  return context;
}

/**
 * Resize the canvas backing store to the current CSS layout size and
 * reconfigure the WebGPU context. Returns the new backing size.
 * @param {HTMLCanvasElement} canvas
 * @param {GPUCanvasContext} context
 * @param {GPUDevice} device
 * @returns {{ width: number, height: number }}
 */
export function resizeCanvasToBackingSize(canvas, context, device) {
  const { width, height } = backingSize(canvas);
  canvas.width = width;
  canvas.height = height;
  context.configure({
    device,
    format: CANVAS_FORMAT,
    alphaMode: CANVAS_ALPHA_MODE,
    usage: canvasUsage(),
  });
  return { width, height };
}
