/**
 * resource-manager.js — GPU resource residency for engine render items.
 *
 * Phase 1 (DS-S2 P1.2): carries the mesh upload helpers over verbatim from
 * corpus/_host/public/greybox-host.js (VB/IB mapped-at-creation upload).
 *
 * Phase 2 (S2 vertical slice, item B): adds the residency layer — logical
 * handles → GPU residency with generation checks under the create-before-
 * retire contract, modeled on the compute/chunk lifecycle in
 * backend/webgpu-runtime.js (applyComputeResourceReplace / enqueueRetire /
 * destroyRetired): every transition is validated, a stale generation is a
 * typed rejection, and a retire only ever destroys buffers that were first
 * created.
 */

import { FaberKernelContractError } from "../contract/artifact-admission.js";

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

/**
 * Mesh residency manager: logical handles → GPU residency with generation
 * checks. The contract mirrors the backend compute lifecycle:
 *
 *   - `create` allocates a NEW generation (create-before-retire — a retire
 *     never destroys live buffers out from under a handle);
 *   - `acquire` / `retire` / `replace` validate the handle's generation
 *     against the live slot — a stale handle is a typed rejection
 *     (generation-mismatch), never a silent skip;
 *   - `trackExisting` registers an already-created GPU entry (the greybox
 *     renderer path) into the residency layer without re-uploading.
 */
export class MeshResourceManager {
  constructor() {
    /** @type {Map<number, { logicalId: number, generation: number, entry: object, retired: boolean }>} */
    this._slots = new Map();
    this._nextLogicalId = 0;
    /** @type {Array<{ logicalId: number, generation: number, buffers: Array<object> }>} */
    this._pendingRetire = [];
    this._counters = { created: 0, live: 0, retired: 0, destroyed: 0 };
  }

  /**
   * Upload a mesh into residency under a fresh logical handle + generation.
   * @param {GPUDevice} device
   * @param {object} mesh - { name, role, vertices, indices }
   * @returns {{ index: number, generation: number }} logical handle
   */
  create(device, mesh) {
    if (!device?.createBuffer) {
      throw new FaberKernelContractError(
        "resource-manager.create",
        "device is required for mesh residency create",
        "product",
      );
    }
    const logicalId = this._nextLogicalId;
    this._nextLogicalId += 1;
    const generation = 1;
    const entry = createMeshGpuEntry(device, mesh);
    this._slots.set(logicalId, { logicalId, generation, entry, retired: false });
    this._counters.created += 1;
    this._counters.live += 1;
    return Object.freeze({ index: logicalId, generation });
  }

  /**
   * Register an already-created GPU entry (greybox renderer path) into the
   * residency layer. Buffers are tracked, not re-uploaded.
   * @param {object} entry - { name, role, vertexBuffer, indexBuffer, indexCount }
   * @returns {{ index: number, generation: number }} logical handle
   */
  trackExisting(entry) {
    if (!entry?.vertexBuffer || !entry?.indexBuffer) {
      throw new FaberKernelContractError(
        "resource-manager.trackExisting",
        "tracked entry must carry vertexBuffer and indexBuffer",
        "product",
      );
    }
    const logicalId = this._nextLogicalId;
    this._nextLogicalId += 1;
    this._slots.set(logicalId, {
      logicalId,
      generation: 1,
      entry,
      retired: false,
    });
    this._counters.created += 1;
    this._counters.live += 1;
    return Object.freeze({ index: logicalId, generation: 1 });
  }

  /**
   * Resolve a logical handle to its live GPU entry. Generation mismatch or a
   * retired slot is a typed rejection.
   * @param {{ index: number, generation: number }} handle
   * @returns {object} the live entry
   */
  acquire(handle) {
    const slot = this._requireLiveSlot(handle, "acquire");
    return slot.entry;
  }

  /**
   * Retire a handle: validates create-before-retire + generation, enqueues
   * the buffers for deferred destruction. The buffers stay valid until
   * `destroyRetired` runs after queue completion.
   * @param {{ index: number, generation: number }} handle
   */
  retire(handle) {
    const slot = this._requireLiveSlot(handle, "retire");
    slot.retired = true;
    this._pendingRetire.push({
      logicalId: slot.logicalId,
      generation: slot.generation,
      buffers: [slot.entry.vertexBuffer, slot.entry.indexBuffer],
    });
    this._counters.live -= 1;
    this._counters.retired += 1;
  }

  /**
   * Replace a live mesh under create-before-retire: allocate the next
   * generation first, then retire the previous generation's buffers.
   * @param {GPUDevice} device
   * @param {{ index: number, generation: number }} handle
   * @param {object} mesh
   * @returns {{ index: number, generation: number }} the new logical handle
   */
  replace(device, handle, mesh) {
    const slot = this._requireLiveSlot(handle, "replace");
    const next = this.create(device, mesh);
    this.retire(Object.freeze({ index: slot.logicalId, generation: slot.generation }));
    return next;
  }

  /**
   * Destroy retired buffers after queue completion (mirrors
   * destroyRetiredComputeResources: snapshot before awaiting, re-queue on
   * fence rejection).
   * @param {GPUDevice} device
   * @returns {Promise<{ destroyed_groups: number, destroyed_buffers: number }>}
   */
  async destroyRetired(device) {
    if (this._pendingRetire.length === 0) {
      return Object.freeze({ destroyed_groups: 0, destroyed_buffers: 0 });
    }
    const done = device?.queue?.onSubmittedWorkDone;
    if (typeof done !== "function") {
      throw new FaberKernelContractError(
        "resource-manager.destroyRetired",
        "queue completion is required before destroying retired mesh buffers",
        "webgpu",
      );
    }
    const groups = this._pendingRetire.splice(0, this._pendingRetire.length);
    try {
      await done.call(device.queue);
    } catch (error) {
      this._pendingRetire.unshift(...groups);
      throw error;
    }
    let destroyedBuffers = 0;
    for (const group of groups) {
      for (const buffer of group.buffers) {
        if (buffer && typeof buffer.destroy === "function" && !buffer.__faberDestroyed) {
          buffer.destroy();
          buffer.__faberDestroyed = true;
          destroyedBuffers += 1;
          this._counters.destroyed += 1;
        }
      }
    }
    return Object.freeze({
      destroyed_groups: groups.length,
      destroyed_buffers: destroyedBuffers,
    });
  }

  /** Residency counters for oracles and the lifecycle harness. */
  snapshot() {
    return Object.freeze({
      created: this._counters.created,
      live: this._counters.live,
      retired: this._counters.retired,
      destroyed: this._counters.destroyed,
      pending_retire_groups: this._pendingRetire.length,
    });
  }

  _requireLiveSlot(handle, op) {
    if (!handle || typeof handle !== "object") {
      throw new FaberKernelContractError(
        `resource-manager.${op}`,
        "a logical handle { index, generation } is required",
        "product",
      );
    }
    const slot = this._slots.get(handle.index);
    if (!slot) {
      throw new FaberKernelContractError(
        `resource-manager.${op}`,
        `unknown logical handle index ${handle.index} — create before use`,
        "product",
      );
    }
    if (slot.generation !== handle.generation) {
      throw new FaberKernelContractError(
        `resource-manager.${op}.generation`,
        `handle generation ${handle.generation} does not match live generation ` +
          `${slot.generation} at index ${handle.index} — stale handle rejected`,
        "product",
      );
    }
    if (slot.retired) {
      throw new FaberKernelContractError(
        `resource-manager.${op}`,
        `logical handle ${handle.index}@${handle.generation} is retired`,
        "product",
      );
    }
    return slot;
  }
}
