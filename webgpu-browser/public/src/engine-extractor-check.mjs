#!/usr/bin/env node
/**
 * Engine scene-extractor gate proof (DS-S2 Phase 2, item B).
 *
 * Extractor-equals-traversal on a frozen fixture scene: render items produced
 * from a `data-scene-geometry` blob + an admitted graphics descriptor must
 * equal a hand-written reference traversal (names, index counts, per-item
 * draw manifests, admitted vertex layout) — the numeric oracle vocabulary.
 *
 * Covers:
 * - extractor-equals-traversal (frozen fixture vs reference).
 * - parseTransformPayload (valid 32-float payload; malformed → null).
 * - Typed rejection paths (NO host guessing):
 *   - null / unadmitted descriptor → FaberKernelContractError;
 *   - geometry blob absent / malformed → typed error;
 *   - published vertex stride not conforming to the admitted layout → typed.
 */

import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

const { FaberKernelContractError } = await import(
  pathToFileURL(path.join(here, "contract", "artifact-admission.js")).href
);
const {
  parseSceneGeometryBlob,
  parseTransformPayload,
  buildRenderItems,
  extractSceneRenderItems,
} = await import(pathToFileURL(path.join(here, "engine", "scene-extractor.js")).href);

function fail(message) {
  console.error(`engine-extractor-check failed: ${message}`);
  process.exit(1);
}

function require(condition, message) {
  if (!condition) fail(message);
}

function deepEqual(a, b, label) {
  const sa = JSON.stringify(a);
  const sb = JSON.stringify(b);
  require(sa === sb, `${label}: expected ${sb}, got ${sa}`);
}

async function expectReject(label, run) {
  try {
    run();
    fail(`${label}: expected FaberKernelContractError rejection`);
  } catch (error) {
    require(
      error instanceof FaberKernelContractError,
      `${label}: expected FaberKernelContractError, got ${error?.name ?? typeof error}: ${error?.message}`,
    );
  }
}

// ── Frozen fixture scene (representative of the corpus terrain demo: two
//    stride-36 meshes — a heightfield grid + a water quad) ─────────────────

function vertexPayload(seed) {
  const floats = [];
  for (let v = 0; v < 4; v++) {
    // pos3 + normal3 + color3 (stride 36)
    floats.push(v, seed, v * 0.5);
    floats.push(0, 1, 0);
    floats.push(0.2 + v * 0.1, 0.4, 0.6);
  }
  return floats.join(" ");
}

const TERRAIN_INDICES = "0 2 1 1 2 3";
const WATER_INDICES = "0 2 1 0 3 2";

const FROZEN_SCENE_BLOB =
  `terrain;static;4;6;${vertexPayload(0)};${TERRAIN_INDICES}` +
  `|water;static;4;6;${vertexPayload(1)};${WATER_INDICES}`;

/**
 * Admitted graphics descriptor in the post-admission shape (the lit shader:
 * pos3+normal3+color3, stride 36, uint32 indices).
 */
function admittedDescriptor(indexCount = 6) {
  return {
    wgsl: "",
    schemaVersion: 1,
    target: "wgsl-text",
    kernels: [
      {
        entryName: "greybox_vertex",
        shaderStage: "vertex",
        vertexInputs: [],
        vertexBufferLayouts: [
          {
            bufferIndex: 0,
            arrayStride: 36,
            stepMode: "vertex",
            attributes: [
              { shaderLocation: 0, format: "float32x3", offset: 0, sourceName: "position" },
              { shaderLocation: 1, format: "float32x3", offset: 12, sourceName: "normal" },
              { shaderLocation: 2, format: "float32x3", offset: 24, sourceName: "color" },
            ],
          },
        ],
      },
      { entryName: "greybox_fragment", shaderStage: "fragment" },
    ],
    pipeline: {
      colorTargetFormats: ["bgra8unorm"],
      primitiveTopology: "triangle-list",
      vertexCount: 36,
      depthStencil: { depthWriteEnabled: true, depthCompare: "less-equal" },
    },
    pipelineLayout: { bindGroupLayoutIndexes: [0] },
    bindGroupLayouts: [],
    bindGroups: [],
    draw: { indexFormat: "uint32", instanceCount: 1, baseVertex: 0, firstIndex: 0, indexCount },
    inputBindings: [],
    outputBindings: [],
  };
}

/** The hand-written reference traversal the extractor must equal. */
function referenceTraversal() {
  return [
    {
      name: "terrain",
      role: "static",
      vertexCount: 4,
      indexCount: 6,
      indexFormat: "uint32",
      draw: { firstIndex: 0, indexCount: 6, instanceCount: 1, baseVertex: 0 },
      vertexLayout: {
        arrayStride: 36,
        stepMode: "vertex",
        attributes: [
          { shaderLocation: 0, format: "float32x3", offset: 0, sourceName: "position" },
          { shaderLocation: 1, format: "float32x3", offset: 12, sourceName: "normal" },
          { shaderLocation: 2, format: "float32x3", offset: 24, sourceName: "color" },
        ],
      },
    },
    {
      name: "water",
      role: "static",
      vertexCount: 4,
      indexCount: 6,
      indexFormat: "uint32",
      draw: { firstIndex: 0, indexCount: 6, instanceCount: 1, baseVertex: 0 },
      vertexLayout: {
        arrayStride: 36,
        stepMode: "vertex",
        attributes: [
          { shaderLocation: 0, format: "float32x3", offset: 0, sourceName: "position" },
          { shaderLocation: 1, format: "float32x3", offset: 12, sourceName: "normal" },
          { shaderLocation: 2, format: "float32x3", offset: 24, sourceName: "color" },
        ],
      },
    },
  ];
}

const VALID_TRANSFORM_TEXT =
  "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1";

async function main() {
  // ── 1. parseSceneGeometryBlob round-trip ──────────────────────────────
  {
    const meshes = parseSceneGeometryBlob(FROZEN_SCENE_BLOB);
    require(meshes.length === 2, `fixture has 2 meshes, got ${meshes.length}`);
    require(meshes[0].name === "terrain" && meshes[1].name === "water", "fixture mesh names");
    require(meshes[0].vertices.length === 36, "terrain vertices = 4 × 9 floats");
    require(meshes[1].indices.length === 6, "water index count 6");
    console.log("T1 PASS: parseSceneGeometryBlob parses the frozen fixture");
  }

  // ── 2. Extractor-equals-traversal ─────────────────────────────────────
  {
    const extracted = extractSceneRenderItems({
      sceneBlob: FROZEN_SCENE_BLOB,
      transformText: VALID_TRANSFORM_TEXT,
      descriptor: admittedDescriptor(),
    });
    deepEqual(extracted.items, referenceTraversal(), "render items equal reference traversal");
    require(
      extracted.transform instanceof Float32Array && extracted.transform.length === 32,
      "extractor parses the 32-float transform payload",
    );
    require(extracted.meshes.length === 2, "extractor returns parsed meshes for the renderer");
    console.log("T2 PASS: extractor-equals-traversal on the frozen fixture scene");
  }

  // ── 3. parseTransformPayload edge cases ───────────────────────────────
  {
    const floats = parseTransformPayload(VALID_TRANSFORM_TEXT);
    require(floats instanceof Float32Array && floats.length === 32, "valid 32-float payload parsed");
    require(parseTransformPayload(null) === null, "null payload → null");
    require(parseTransformPayload("") === null, "empty payload → null");
    require(parseTransformPayload("1 2 3") === null, "short payload → null");
    require(parseTransformPayload(`${VALID_TRANSFORM_TEXT} 1`) === null, "33 floats → null");
    console.log("T3 PASS: parseTransformPayload edge cases");
  }

  // ── 4. Typed rejection paths (no host guessing) ───────────────────────
  {
    await expectReject("null descriptor (unadmitted artifact)", () =>
      buildRenderItems({ meshes: parseSceneGeometryBlob(FROZEN_SCENE_BLOB), descriptor: null }),
    );
    await expectReject("missing scene blob", () =>
      extractSceneRenderItems({ sceneBlob: null, transformText: null, descriptor: admittedDescriptor() }),
    );
    await expectReject("malformed blob", () =>
      extractSceneRenderItems({ sceneBlob: "terrain;static;4;6;1 2 3;0 1 2", transformText: null, descriptor: admittedDescriptor() }),
    );
    console.log("T4 PASS: unadmitted descriptor / missing / malformed scene facts rejected cleanly");
  }

  // ── 5. Conformity: published stride must equal the admitted layout ────
  {
    // A blob whose per-vertex float count does not match the admitted stride
    // 36 (e.g. pos3+normal3 = 24 bytes) must be a typed rejection, not a guess.
    const wrongStrideBlob = `box;static;4;6;0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0;0 1 2 2 1 3`;
    await expectReject("published stride 24 ≠ admitted stride 36", () =>
      extractSceneRenderItems({ sceneBlob: wrongStrideBlob, transformText: null, descriptor: admittedDescriptor() }),
    );
    console.log("T5 PASS: non-conforming published vertex layout rejected (no host guessing)");
  }

  console.log("");
  console.log("engine-extractor-check passed");
  console.log("covered: extractor-equals-traversal, transform payload parsing, typed rejections,");
  console.log("         stride-conformity enforcement against the admitted layout");
}

main().catch((error) => {
  fail(error?.stack ?? String(error));
});
