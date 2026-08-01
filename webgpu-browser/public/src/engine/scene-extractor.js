/**
 * scene-extractor.js — scene extraction for the engine.
 *
 * Phase 1 (DS-S2 P1.2): carries `parseSceneGeometryBlob` over verbatim from
 * corpus/_host/public/greybox-host.js.
 *
 * Phase 2 (S2 vertical slice, item B): the extractor consumes the reflection
 * + DOM scene facts (`data-scene-geometry`, `data-transform-payload`) and
 * produces **render items** — with NO host guessing. Vertex layout and draw
 * facts come from the admitted graphics descriptor (artifact-admission); the
 * extractor validates the published scene facts against that descriptor and
 * rejects with a typed error when they do not conform. Where the (P1.3
 * placeholder) reflection is rejected by admission, the extractor refuses to
 * build items at all: a clean typed rejection, never a silent fallback.
 *
 * JS owns transport + WebGPU lifecycle only. Simulation stays in Faber.
 */

import { FaberKernelContractError } from "../contract/artifact-admission.js";

/**
 * Parse `data-scene-geometry` published once by the Faber controller.
 *
 * Format (pipe-separated objects):
 *   name;role;vertexCount;indexCount;v0 v1 ...;i0 i1 ...|name2;...
 *
 * role is "static".
 *
 * @param {string} blob
 * @returns {Array<{ name: string, role: string, vertices: Float32Array, indices: Uint32Array }>}
 */
export function parseSceneGeometryBlob(blob) {
  if (!blob || typeof blob !== "string") {
    throw new FaberKernelContractError(
      "scene-extractor",
      "empty scene geometry blob — the controller has not published data-scene-geometry",
      "product",
    );
  }
  const objects = [];
  const parts = blob.split("|").filter((p) => p.length > 0);
  for (const part of parts) {
    const fields = part.split(";");
    if (fields.length !== 6) {
      throw new FaberKernelContractError(
        "scene-extractor",
        `bad geometry object fields (${fields.length}): ${part.slice(0, 40)}`,
        "product",
      );
    }
    const name = fields[0];
    const role = fields[1];
    const vertexCount = Number(fields[2]);
    const indexCount = Number(fields[3]);
    const verts = fields[4].trim().split(/\s+/).map(Number);
    const idxs = fields[5].trim().split(/\s+/).map(Number);
    if (verts.length !== vertexCount * 9) {
      throw new FaberKernelContractError(
        `scene-extractor.${name}`,
        `vertex float count ${verts.length} != ${vertexCount * 9} (expected 9 floats per vertex: pos3+normal3+color3)`,
        "product",
      );
    }
    if (idxs.length !== indexCount) {
      throw new FaberKernelContractError(
        `scene-extractor.${name}`,
        `index count ${idxs.length} != ${indexCount}`,
        "product",
      );
    }
    objects.push({
      name,
      role,
      vertexCount,
      vertices: new Float32Array(verts),
      indices: new Uint32Array(idxs),
    });
  }
  if (objects.length === 0) {
    throw new FaberKernelContractError(
      "scene-extractor",
      "no scene objects in geometry blob",
      "product",
    );
  }
  return objects;
}

/**
 * Parse transform payload text ("f0 f1 ... f31") into Float32Array(32):
 * model(16) + view-projection(16), column-major.
 *
 * @param {string|null} text
 * @returns {Float32Array|null}
 */
export function parseTransformPayload(text) {
  if (!text) return null;
  const parts = text.trim().split(/\s+/);
  if (parts.length !== 32) return null;
  const floats = new Float32Array(32);
  for (let i = 0; i < 32; i++) {
    floats[i] = Number(parts[i]);
    if (!Number.isFinite(floats[i])) return null;
  }
  return floats;
}

/**
 * Build render items from parsed scene meshes and the ADMITTED graphics
 * descriptor. The vertex layout (stride, attribute profile) and the draw
 * bounds come from the descriptor — i.e. from reflection via
 * artifact-admission — never from hardcoded host knowledge.
 *
 * Each item is a deterministic structure (frozen) that the numeric oracles
 * compare against reference traversals (extractor-equals-traversal).
 *
 * @param {object} options
 * @param {Array<{ name: string, role: string, vertices: Float32Array, indices: Uint32Array }>} options.meshes
 * @param {object} options.descriptor - admitted graphics descriptor
 *   (from loadFaberGraphicsPipeline). REQUIRED — a null descriptor means the
 *   artifact was not admitted, and the extractor refuses to guess.
 * @returns {Array<object>} frozen render items
 */
export function buildRenderItems({ meshes, descriptor }) {
  if (!descriptor || typeof descriptor !== "object") {
    throw new FaberKernelContractError(
      "scene-extractor",
      "buildRenderItems requires an admitted graphics descriptor; the pipeline " +
        "artifact was not admitted (stale or missing reflection) — refusing host guessing",
      "product",
    );
  }
  if (!Array.isArray(meshes) || meshes.length === 0) {
    throw new FaberKernelContractError(
      "scene-extractor",
      "buildRenderItems requires at least one scene mesh",
      "product",
    );
  }

  const vertexLayout = descriptor.kernels?.[0]?.vertexBufferLayouts?.[0];
  if (!vertexLayout || typeof vertexLayout.arrayStride !== "number" || vertexLayout.arrayStride <= 0) {
    throw new FaberKernelContractError(
      "scene-extractor",
      "admitted descriptor carries no vertex buffer layout (arrayStride) — cannot derive render items",
      "product",
    );
  }

  const draw = descriptor.draw;
  if (!draw || typeof draw !== "object") {
    throw new FaberKernelContractError(
      "scene-extractor",
      "admitted descriptor carries no draw manifest — cannot derive render items",
      "product",
    );
  }

  return Object.freeze(
    meshes.map((mesh) => {
      // Conformity check: the published per-vertex byte stride must equal the
      // admitted vertex layout stride (reflection-derived, not guessed).
      // vertexCount is a declared fact in the blob (field[2]); 9 floats per
      // vertex is the pos3+normal3+color3 blob contract the controller emits.
      const floatsPerVertex = mesh.vertexCount > 0 ? mesh.vertices.length / mesh.vertexCount : 0;
      const strideBytes = floatsPerVertex * 4;
      if (strideBytes !== vertexLayout.arrayStride) {
        throw new FaberKernelContractError(
          `scene-extractor.${mesh.name}`,
          `published vertex stride ${strideBytes}B does not match admitted vertex layout ` +
            `arrayStride ${vertexLayout.arrayStride}B`,
          "product",
        );
      }

      return Object.freeze({
        name: mesh.name,
        role: mesh.role || "static",
        vertexCount: mesh.vertexCount,
        indexCount: mesh.indices.length,
        indexFormat: draw.indexFormat,
        draw: Object.freeze({
          firstIndex: draw.firstIndex,
          indexCount: mesh.indices.length,
          instanceCount: draw.instanceCount,
          baseVertex: draw.baseVertex,
        }),
        vertexLayout: Object.freeze({
          arrayStride: vertexLayout.arrayStride,
          stepMode: vertexLayout.stepMode,
          attributes: Object.freeze(
            vertexLayout.attributes.map((attr) =>
              Object.freeze({
                shaderLocation: attr.shaderLocation,
                format: attr.format,
                offset: attr.offset,
                sourceName: attr.sourceName,
              }),
            ),
          ),
        }),
      });
    }),
  );
}

/**
 * Extract render items from the DOM scene facts through the admitted
 * descriptor: parse `data-scene-geometry` + `data-transform-payload` and
 * derive the deterministic render-item set.
 *
 * Typed rejection paths (never silent fallback):
 *   - descriptor null / not admitted  → FaberKernelContractError (product)
 *   - geometry blob missing/malformed → FaberKernelContractError (product)
 *   - vertex data does not conform to the admitted layout → typed error
 *
 * The transform payload is a per-frame render fact, not structure: a missing
 * payload yields `transform: null` (the frame loop holds the last published
 * transform; identity is the documented initial value).
 *
 * @param {object} options
 * @param {string} options.sceneBlob - data-scene-geometry text
 * @param {string|null} options.transformText - data-transform-payload text
 * @param {object} options.descriptor - admitted graphics descriptor
 * @returns {{ items: Array<object>, transform: Float32Array|null, meshes: Array<object> }}
 */
export function extractSceneRenderItems({ sceneBlob, transformText, descriptor }) {
  const meshes = parseSceneGeometryBlob(sceneBlob);
  const items = buildRenderItems({ meshes, descriptor });
  return Object.freeze({
    items,
    transform: parseTransformPayload(transformText),
    meshes,
  });
}
