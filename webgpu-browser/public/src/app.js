import * as THREE from "three/webgpu";
import {
  FaberKernelContractError,
  fetchFaberKernelArtifacts,
  loadFaberKernel,
} from "./faber-kernel.js";
import { acquireWebGpuDevice, createWebGpuResources, runKernel } from "./webgpu-runtime.js";

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

main().catch((error) => {
  const proof = proofFailure(error);
  setStatus("error", proof.error);
  window.faberWebGpuProof = proof;
});

async function main() {
  setStatus("pending", "Loading");

  const artifacts = await fetchFaberKernelArtifacts();
  const kernel = loadFaberKernel(artifacts);
  const { device } = await acquireWebGpuDevice();
  const resources = createWebGpuResources(device, kernel, {
    x: new Float32Array([INPUT_VALUE]),
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
