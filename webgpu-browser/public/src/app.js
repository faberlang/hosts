import * as THREE from "three/webgpu";
import {
  FaberKernelContractError,
  fetchFaberKernelArtifacts,
  loadFaberKernel,
  loadFaberGraphicsPipeline,
} from "./contract/artifact-admission.js";
import {
  acquireWebGpuDevice,
  createWebGpuResources,
  runKernel,
  createGraphicsResources,
  runGraphicsFrame,
  replaceDepthTextureOnResize,
  onDeviceLost,
} from "./backend/webgpu-runtime.js";

const INPUT_VALUE = 41.0;
const EXPECTED_VALUE = 42.0;
const EPSILON = 0.000_001;

const elements = {
  scene: document.querySelector("#scene"),
  kernelName: document.querySelector("#kernel-name"),
  inputValue: document.querySelector("#input-value"),
  outputValue: document.querySelector("#output-value"),
  statusValue: document.querySelector("#status-value"),
};

window.faberWebGpuProof = Object.freeze({ ok: false, status: "starting" });
window.faberWebGpuGraphicsProof = Object.freeze({ ok: false, status: "starting" });

main().catch((error) => {
  const proof = proofFailure(error);
  setStatus("error", proof.error);
  window.faberWebGpuProof = proof;
});

async function main() {
  // Run compute proof (existing path).
  await runComputeProof();
  // Run graphics proof (new path — best-effort; failure does not block compute).
  runGraphicsProof().catch((error) => {
    window.faberWebGpuGraphicsProof = proofFailure(error);
  });
}

async function runComputeProof() {
  setStatus("pending", "Loading");

  const artifacts = await fetchFaberKernelArtifacts();
  const kernel = loadFaberKernel(artifacts);
  const { device } = await acquireWebGpuDevice();
  const resources = createWebGpuResources(device, kernel, {
    x: new Float32Array([INPUT_VALUE]),
    // U2 runtime-extent channel: the scalar-shaped add_one kernel reads its
    // OOB guard from the extent binding; the host supplies the count (1 for
    // the single-element proof).
    runtime_extent: new Uint32Array([1]),
  });
  const result = await runKernel(device, resources, kernel);
  const value = result.values[0];

  if (Math.abs(value - EXPECTED_VALUE) > EPSILON) {
    throw new FaberKernelContractError(
      "readback",
      `expected ${EXPECTED_VALUE}, got ${value}`,
      "product",
    );
  }

  elements.kernelName.textContent = kernel.entryName;
  elements.inputValue.textContent = INPUT_VALUE.toFixed(1);
  elements.outputValue.textContent = value.toFixed(1);
  setStatus("ok", "Ready");

  await renderResultScene(value);

  window.faberWebGpuProof = Object.freeze({
    ok: true,
    status: "ready",
    kind: "ok",
    entryName: kernel.entryName,
    value,
    expected: EXPECTED_VALUE,
    dispatchWorkgroups: kernel.dispatchWorkgroups,
  });
}

// ── Graphics proof ────────────────────────────────────────────────────────

async function runGraphicsProof() {
  window.faberWebGpuGraphicsProof = Object.freeze({ ok: false, status: "starting" });

  // Fetch graphics artifacts.
  const [wgslResponse, reflectionResponse, positionsResponse, colorsResponse, indicesResponse, transformResponse, drawResponse] = await Promise.all([
    fetch("./generated/graphics.wgsl"),
    fetch("./generated/graphics-reflection.json"),
    fetch("./generated/graphics-vertex-positions.bin"),
    fetch("./generated/graphics-vertex-colors.bin"),
    fetch("./generated/graphics-indices.bin"),
    fetch("./generated/graphics-transform.bin"),
    fetch("./generated/draw.json"),
  ]);

  for (const [label, response] of [["wgsl", wgslResponse], ["reflection", reflectionResponse], ["positions", positionsResponse], ["colors", colorsResponse], ["indices", indicesResponse], ["transform", transformResponse], ["draw", drawResponse]]) {
    if (!response.ok) {
      throw new FaberKernelContractError(label, `failed to fetch graphics ${label}`, "artifact-fetch");
    }
  }

  const wgsl = await wgslResponse.text();
  const reflection = await reflectionResponse.json();
  const drawManifest = await drawResponse.json();

  const descriptor = loadFaberGraphicsPipeline({ wgsl, reflection, drawManifest });

  const { device } = await acquireWebGpuDevice();
  onDeviceLost(device, (info) => {
    window.faberWebGpuGraphicsProof = Object.freeze({
      ok: false,
      status: "error",
      kind: info.kind,
      reason: info.reason,
      message: info.message,
    });
  });

  // Get or create a canvas with a WebGPU context.
  let canvas = document.querySelector("#gpu-canvas");
  if (!canvas) {
    canvas = document.createElement("canvas");
    canvas.id = "gpu-canvas";
    canvas.style.display = "none";
    document.body.append(canvas);
  }

  const context = canvas.getContext("webgpu");
  if (!context) {
    throw new FaberKernelContractError("canvas", "WebGPU canvas context is unavailable", "webgpu");
  }

  context.configure({
    device,
    format: "bgra8unorm",
    alphaMode: "opaque",
  });

  // Set canvas to a small size (proof-only; not visible).
  canvas.width = 256;
  canvas.height = 256;

  // Create payloads from fetched binary data.
  const positionsBuffer = await positionsResponse.arrayBuffer();
  const colorsBuffer = await colorsResponse.arrayBuffer();
  const indicesBuffer = await indicesResponse.arrayBuffer();
  const transformBuffer = await transformResponse.arrayBuffer();

  const payloads = {
    vertexBuffers: [
      { slot: 0, data: positionsBuffer },
      { slot: 1, data: colorsBuffer },
    ],
    indexData: new Uint32Array(indicesBuffer),
    storageData: {
      transform: new Float32Array(transformBuffer),
    },
  };

  let resources = createGraphicsResources(device, descriptor, payloads, context);

  const frameState = { submittedFrameCount: 0 };

  // Submit one indexed render pass.
  runGraphicsFrame(device, context, resources, descriptor, frameState);

  window.faberWebGpuGraphicsProof = Object.freeze({
    ok: true,
    status: "ready",
    kind: "ok",
    submittedFrameCount: frameState.submittedFrameCount,
    vertexCount: descriptor.pipeline.vertexCount,
    indexFormat: descriptor.draw.indexFormat,
    colorTarget: descriptor.pipeline.colorTargetFormats[0],
  });
}

/** Map product failures into a stable console-inspectable shape. */
function proofFailure(error) {
  const kind =
    error instanceof FaberKernelContractError
      ? error.kind
      : typeof error?.kind === "string"
        ? error.kind
        : "product";
  return Object.freeze({
    ok: false,
    status: "error",
    kind,
    path: error?.path ?? null,
    error: error?.message ?? String(error),
  });
}

async function renderResultScene(value) {
  const renderer = new THREE.WebGPURenderer({ antialias: true });
  await renderer.init();
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(elements.scene.clientWidth, elements.scene.clientHeight);
  elements.scene.append(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf5f7f2);

  const camera = new THREE.PerspectiveCamera(42, sceneAspect(), 0.1, 100);
  camera.position.set(0, 1.2, 5);

  const geometry = new THREE.BoxGeometry(1.2, resultHeight(value), 1.2);
  const material = new THREE.MeshStandardMaterial({ color: 0x2f6f5e, roughness: 0.45, metalness: 0.08 });
  const mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);

  const fill = new THREE.DirectionalLight(0xffffff, 2.6);
  fill.position.set(3, 5, 4);
  scene.add(fill);
  scene.add(new THREE.AmbientLight(0xbfd7d0, 1.4));

  window.addEventListener("resize", () => {
    renderer.setSize(elements.scene.clientWidth, elements.scene.clientHeight);
    camera.aspect = sceneAspect();
    camera.updateProjectionMatrix();
  });

  renderer.setAnimationLoop(() => {
    mesh.rotation.x += 0.004;
    mesh.rotation.y += 0.007;
    renderer.render(scene, camera);
  });
}

function resultHeight(value) {
  return Math.max(0.4, Math.min(3.2, value / 14));
}

function sceneAspect() {
  return Math.max(0.1, elements.scene.clientWidth / Math.max(1, elements.scene.clientHeight));
}

function setStatus(state, label) {
  elements.statusValue.dataset.state = state;
  elements.statusValue.textContent = label;
}
