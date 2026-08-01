/**
 * resource-manager.js — GPU resource residency for engine render items.
 *
 * Phase 1 (DS-S2 P1.2): carries the mesh upload helpers over verbatim from
 * corpus/_host/public/greybox-host.js (VB/IB mapped-at-creation upload).
 * Phase 2 (S2 vertical slice) extends this with generation checks and the
 * lifecycle half of the backend lane split.
 */

function createMappedGpuBuffer(device, data, usage) {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(
    data instanceof Uint8Array
      ? data
      : new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
  );
  buffer.unmap();
  return buffer;
}

/**
 * Build GPU mesh resources for one object (interleaved pos+color VB + IB).
 * @param {GPUDevice} device
 * @param {{ name: string, role: string, vertices: Float32Array, indices: Uint32Array }} mesh
 */
export function createMeshGpuEntry(device, mesh) {
  const vertexBuffer = createMappedGpuBuffer(
    device,
    mesh.vertices,
    GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  );
  const indexBuffer = createMappedGpuBuffer(
    device,
    mesh.indices,
    GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  );
  return {
    name: mesh.name,
    role: mesh.role || "static",
    vertexBuffer,
    indexBuffer,
    indexCount: mesh.indices.length,
  };
}
