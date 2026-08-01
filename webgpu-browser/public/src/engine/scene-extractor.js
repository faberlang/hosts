/**
 * scene-extractor.js — scene geometry extraction for the engine.
 *
 * Phase 1 (DS-S2 P1.2): carries `parseSceneGeometryBlob` over verbatim from
 * corpus/_host/public/greybox-host.js. Phase 2 (S2 vertical slice) adds the
 * Triga-scene-store → render-items path on top of this module.
 */

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
    throw new Error("scene-extractor: empty scene geometry blob");
  }
  const objects = [];
  const parts = blob.split("|").filter((p) => p.length > 0);
  for (const part of parts) {
    const fields = part.split(";");
    if (fields.length !== 6) {
      throw new Error(`scene-extractor: bad geometry object fields (${fields.length}): ${part.slice(0, 40)}`);
    }
    const name = fields[0];
    const role = fields[1];
    const vertexCount = Number(fields[2]);
    const indexCount = Number(fields[3]);
    const verts = fields[4].trim().split(/\s+/).map(Number);
    const idxs = fields[5].trim().split(/\s+/).map(Number);
    if (verts.length !== vertexCount * 9) {
      throw new Error(
        `scene-extractor: ${name} vertex float count ${verts.length} != ${vertexCount * 9} (expected 9 floats per vertex: pos3+normal3+color3)`,
      );
    }
    if (idxs.length !== indexCount) {
      throw new Error(
        `scene-extractor: ${name} index count ${idxs.length} != ${indexCount}`,
      );
    }
    objects.push({
      name,
      role,
      vertices: new Float32Array(verts),
      indices: new Uint32Array(idxs),
    });
  }
  if (objects.length === 0) {
    throw new Error("scene-extractor: no scene objects in geometry blob");
  }
  return objects;
}
