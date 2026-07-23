import { FaberKernelContractError } from "./faber-kernel.js";

const BUFFER_USAGE = {
  input: () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  output: () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  readback: () => GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  vertex: () => GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  index: () => GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
};

const EXPECTED_CANVAS_FORMAT = "bgra8unorm";
const DEPTH_FORMAT = "depth24plus";

function shaderStageFor(visibility) {
  switch (visibility) {
    case "compute":
      return GPUShaderStage.COMPUTE;
    case "vertex":
      return GPUShaderStage.VERTEX;
    case "fragment":
      return GPUShaderStage.FRAGMENT;
    default:
      throw new FaberKernelContractError(
        "visibility",
        `unknown shader visibility: ${visibility}`,
      );
  }
}

export async function acquireWebGpuDevice({
  navigator: nav = globalThis.navigator,
} = {}) {
  if (!nav?.gpu) {
    throw new FaberKernelContractError(
      "navigator.gpu",
      "WebGPU is unavailable in this browser",
      "webgpu",
    );
  }

  const adapter = await nav.gpu.requestAdapter();
  if (!adapter) {
    throw new FaberKernelContractError(
      "navigator.gpu.requestAdapter",
      "no WebGPU adapter available",
      "webgpu",
    );
  }

  const device = await adapter.requestDevice();
  return { adapter, device };
}

export function createWebGpuResources(device, descriptor, initialInputs = {}) {
  const buffers = createBuffers(device, descriptor, initialInputs);
  const bindGroupLayouts = createBindGroupLayouts(device, descriptor);
  const pipelineLayout = createPipelineLayout(device, descriptor, bindGroupLayouts);
  const shaderModule = device.createShaderModule({ code: descriptor.wgsl });
  const pipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: {
      module: shaderModule,
      entryPoint: descriptor.entryName,
    },
  });
  const bindGroups = createBindGroups(device, descriptor, bindGroupLayouts, buffers);

  return Object.freeze({
    buffers,
    bindGroupLayouts,
    pipelineLayout,
    shaderModule,
    pipeline,
    bindGroups,
    /** @type {Map<number, object>} */
    pendingRetire: [],
    counters: {
      created: 0,
      live: 0,
      retired: 0,
      destroyed: 0,
    },
    path: COMPUTE_RESOURCE_PATH,
  });
}

/**
 * Dispatch a compute kernel and read back outputs. For backward compatibility
 * the single-output path returns { values, outputBinding }; multiple outputs
 * return { results, outputBindings }.
 */
export async function runKernel(device, resources, descriptor) {
  // Validate all output buffers exist (removes single-output constraint)
  for (const binding of descriptor.outputBindings) {
    if (!resources.buffers.has(binding.resourceIndex)) {
      throw new FaberKernelContractError(
        "resources.buffers",
        `missing output resource ${binding.resourceIndex}`,
      );
    }
  }

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(resources.pipeline);
  for (const group of resources.bindGroups) {
    pass.setBindGroup(group.bindGroupIndex, group.bindGroup);
  }
  pass.dispatchWorkgroups(
    descriptor.dispatchWorkgroups.x,
    descriptor.dispatchWorkgroups.y,
    descriptor.dispatchWorkgroups.z,
  );
  pass.end();
  device.queue.submit([encoder.finish()]);

  const results = await placementReadback(device, resources, descriptor.outputBindings);

  // Backward compatibility: single-output returns { values, outputBinding }
  if (results.length === 1) {
    return Object.freeze({ values: results[0].values, outputBinding: results[0].binding });
  }
  return Object.freeze({ results, outputBindings: descriptor.outputBindings });
}

// ── Placement operations (D-SPINE-02 S3) ──────────────────────────────────

/**
 * Write host data to a named device buffer using device.queue.writeBuffer.
 * Separable from kernel dispatch — call before dispatch to stage input data.
 *
 * @param {GPUDevice} device
 * @param {object} resources - must have resources.buffers Map<number, ComputeResourceEntry>
 * @param {{ resourceIndex: number, data: ArrayBuffer|TypedArray }} descriptor
 * @returns {{ status: number }}
 */
export function placementCopyIn(device, resources, { resourceIndex, data }) {
  const entry = resources.buffers.get(resourceIndex);
  if (!entry) {
    throw new FaberKernelContractError(
      "placementCopyIn",
      `missing resource ${resourceIndex}`,
    );
  }
  if (!(data instanceof ArrayBuffer) && !ArrayBuffer.isView(data)) {
    throw new FaberKernelContractError(
      "placementCopyIn",
      "data must be an ArrayBuffer or typed array",
    );
  }
  device.queue.writeBuffer(entry.buffer, 0, data);
  return Object.freeze({ status: 0 });
}

/**
 * Read back device buffer contents to the host. Accepts a list of output
 * bindings — not limited to a single output.
 *
 * @param {GPUDevice} device
 * @param {object} resources - must have resources.buffers Map<number, ComputeResourceEntry>
 * @param {Array<{ resourceIndex: number, bufferByteLen: number }>} outputBindings
 * @returns {Promise<Array<{ binding: object, values: number[] }>>}
 */
export async function placementReadback(device, resources, outputBindings) {
  if (!Array.isArray(outputBindings)) {
    throw new FaberKernelContractError(
      "placementReadback",
      "outputBindings must be an array",
    );
  }

  const encoder = device.createCommandEncoder();
  const transfers = [];

  for (const binding of outputBindings) {
    const entry = resources.buffers.get(binding.resourceIndex);
    if (!entry) {
      throw new FaberKernelContractError(
        "placementReadback",
        `missing resource ${binding.resourceIndex}`,
      );
    }
    const readbackBuffer = device.createBuffer({
      size: binding.bufferByteLen,
      usage: BUFFER_USAGE.readback(),
    });
    encoder.copyBufferToBuffer(entry.buffer, 0, readbackBuffer, 0, binding.bufferByteLen);
    transfers.push({ binding, buffer: readbackBuffer });
  }

  device.queue.submit([encoder.finish()]);

  const results = [];
  for (const { binding, buffer } of transfers) {
    await buffer.mapAsync(GPUMapMode.READ);
    const copy = buffer.getMappedRange().slice(0);
    buffer.unmap();
    buffer.destroy();
    results.push({
      binding,
      values: Array.from(new Float32Array(copy)),
    });
  }

  return Object.freeze(results);
}

/**
 * Insert a device-side ordering barrier for the named buffer IDs. Does not
 * block the host — sync is a queue-level ordering assertion, not a host-
 * visible fence.
 *
 * @param {GPUDevice} device
 * @param {object} resources - must have resources.buffers Map<number, ComputeResourceEntry>
 * @param {number[]} bufferIds - resource indices to order
 */
export function placementSync(device, resources, bufferIds) {
  for (const bufferId of bufferIds) {
    if (!resources.buffers.has(bufferId)) {
      throw new FaberKernelContractError(
        "placementSync",
        `unknown buffer ${bufferId}`,
      );
    }
  }
  // Submit an empty encoder to create an ordering point on the device queue.
  // WebGPU submission order defines execution order — subsequent submissions
  // are ordered after this empty submission.
  const encoder = device.createCommandEncoder();
  device.queue.submit([encoder.finish()]);
}

function createBuffers(device, descriptor, initialInputs) {
  const buffers = new Map();

  for (const group of descriptor.bindGroups) {
    for (const entry of group.entries) {
      if (buffers.has(entry.resourceIndex)) {
        continue;
      }

      const buffer = device.createBuffer({
        size: entry.bufferByteLen,
        usage: bufferUsageForRole(entry.role),
        mappedAtCreation: entry.role === "input",
      });

      if (entry.role === "input") {
        writeInitialInput(buffer, entry, initialInputs);
      }

      buffers.set(entry.resourceIndex, {
        buffer,
        generation: 0,
        logicalId: entry.resourceIndex,
      });
    }
  }

  return buffers;
}

function writeInitialInput(buffer, entry, initialInputs) {
  const inputName = entry.sourceName ?? `resource_${entry.resourceIndex}`;
  const value = initialInputs[inputName];
  if (!(value instanceof Float32Array)) {
    throw new FaberKernelContractError(`initialInputs.${inputName}`, "expected Float32Array");
  }
  if (value.byteLength !== entry.bufferByteLen) {
    throw new FaberKernelContractError(
      `initialInputs.${inputName}`,
      `expected ${entry.bufferByteLen} bytes, got ${value.byteLength}`,
    );
  }

  const target = new Uint8Array(buffer.getMappedRange());
  target.set(new Uint8Array(value.buffer, value.byteOffset, value.byteLength));
  buffer.unmap();
}

function createBindGroupLayouts(device, descriptor) {
  const layouts = new Map();

  for (const layout of descriptor.bindGroupLayouts) {
    const entries = layout.entries.map((entry) => ({
      binding: entry.binding,
      visibility: shaderStageFor(entry.visibility),
      buffer: {
        type: entry.bufferType,
        hasDynamicOffset: false,
        minBindingSize: entry.minBindingSize,
      },
    }));
    layouts.set(layout.bindGroupIndex, device.createBindGroupLayout({ entries }));
  }

  return layouts;
}

function createPipelineLayout(device, descriptor, bindGroupLayouts) {
  const orderedLayouts = descriptor.pipelineLayout.bindGroupLayoutIndexes.map((index) => {
    const layout = bindGroupLayouts.get(index);
    if (!layout) {
      throw new FaberKernelContractError("pipelineLayout.bindGroupLayoutIndexes", `missing layout ${index}`);
    }
    return layout;
  });

  return device.createPipelineLayout({ bindGroupLayouts: orderedLayouts });
}

function createBindGroups(device, descriptor, bindGroupLayouts, buffers) {
  return descriptor.bindGroups.map((group) => {
    const layout = bindGroupLayouts.get(group.bindGroupIndex);
    if (!layout) {
      throw new FaberKernelContractError("bindGroupLayouts", `missing layout ${group.bindGroupIndex}`);
    }

    const entries = group.entries.map((entry) => {
      const computeEntry = buffers.get(entry.resourceIndex);
      if (!computeEntry) {
        throw new FaberKernelContractError("buffers", `missing resource ${entry.resourceIndex}`);
      }

      return {
        binding: entry.binding,
        resource: {
          buffer: computeEntry.buffer,
          offset: entry.bufferByteOffset,
          size: entry.bindingByteLen,
        },
      };
    });

    return Object.freeze({
      bindGroupIndex: group.bindGroupIndex,
      bindGroup: device.createBindGroup({ layout, entries }),
    });
  });
}

function bufferUsageForRole(role) {
  const usage = BUFFER_USAGE[role];
  if (!usage) {
    throw new FaberKernelContractError("binding.role", `unsupported role ${role}`);
  }
  return usage();
}

function expectSingleOutputBinding(descriptor) {
  if (descriptor.outputBindings.length !== 1) {
    throw new FaberKernelContractError(
      "descriptor.outputBindings",
      `expected one output binding, got ${descriptor.outputBindings.length}`,
    );
  }
  return descriptor.outputBindings[0];
}

// ── Compute resource lifecycle ────────────────────────────────────────────

/**
 * @typedef {object} ComputeResourceEntry
 * @property {GPUBuffer} buffer - the backing GPU buffer
 * @property {number} generation - monotonic generation counter
 * @property {number} logicalId - stable resource identity (resourceIndex)
 */

/**
 * Snapshot of honest buffer counters for a compute session.
 */
export function computeResourceCounters(resources) {
  expectComputeResources(resources);
  return Object.freeze({
    created: resources.counters.created,
    live: resources.counters.live,
    retired: resources.counters.retired,
    destroyed: resources.counters.destroyed,
    pending_retire_groups: resources.pendingRetire.length,
    path: resources.path,
  });
}

function expectComputeResources(resources) {
  if (!resources || resources.path !== COMPUTE_RESOURCE_PATH) {
    throw new FaberKernelContractError(
      "resources",
      "expected compute resource session (compute path)",
      "product",
    );
  }
  if (!(resources.buffers instanceof Map) || !Array.isArray(resources.pendingRetire) || !resources.counters) {
    throw new FaberKernelContractError(
      "resources",
      "compute resource session is missing map/counters",
      "product",
    );
  }
}

/**
 * Create one compute GPU buffer entry. Increments created and live counters.
 *
 * @param {GPUDevice} device
 * @param {object} resources - compute resource session
 * @param {number} logicalId
 * @param {number} generation
 * @param {{ size: number, usage: number, mappedAtCreation?: boolean }} bufferDescriptor
 * @returns {{ logicalId: number, generation: number, buffer: GPUBuffer, buffers: GPUBuffer[] }}
 */
function createComputeGpuEntry(device, resources, logicalId, generation, bufferDescriptor) {
  const buffer = device.createBuffer(bufferDescriptor);
  resources.counters.created += 1;
  resources.counters.live += 1;
  return {
    logicalId,
    generation,
    buffer,
    buffers: [buffer],
  };
}

/**
 * Enqueue a compute entry for deferred destruction after queue completion.
 * Decrements live and increments retired counters.
 *
 * @param {object} resources - compute resource session
 * @param {{ logicalId: number, generation: number, buffers: GPUBuffer[] }} entry
 */
function enqueueComputeRetire(resources, entry) {
  resources.pendingRetire.push({
    logicalId: entry.logicalId,
    generation: entry.generation,
    buffers: entry.buffers.slice(),
  });
  resources.counters.live -= 1;
  resources.counters.retired += 1;
}

/**
 * Apply one compute resource transition under the create-before-retire contract.
 *
 * transition = {
 *   resource_index: number,
 *   generation: number,
 *   buffer_descriptor: null | undefined | { size, usage, mappedAtCreation? }
 * }
 *
 * Empty buffer_descriptor removes the live resource (retire previous after
 * queue completion).  Non-empty creates or replaces (create-before-retire).
 * Invalid transitions throw FaberKernelContractError kind=product.
 */
export function applyComputeResourceReplace(device, resources, transition) {
  expectComputeResources(resources);
  if (!device?.createBuffer) {
    throw new FaberKernelContractError("device", "device is required for compute replace", "product");
  }

  const resourceIndex = expectNonNegativeInt(transition?.resource_index, "transition.resource_index");
  const generation = expectNonNegativeInt(transition?.generation, "transition.generation");
  const empty = isEmptyComputePayload(transition?.buffer_descriptor);
  const current = resources.buffers.get(resourceIndex) ?? null;

  if (empty) {
    if (!current) {
      throw new FaberKernelContractError(
        "transition",
        `cannot remove resource ${resourceIndex}: no live resource`,
        "product",
      );
    }
    if (generation !== current.generation) {
      throw new FaberKernelContractError(
        "transition.generation",
        `remove requires generation ${current.generation}, got ${generation}`,
        "product",
      );
    }
    enqueueComputeRetire(resources, current);
    resources.buffers.delete(resourceIndex);
    return Object.freeze({
      kind: "removed",
      resource_index: resourceIndex,
      generation,
      previous_generation: current.generation,
    });
  }

  const descriptor = normalizeComputePayload(transition.buffer_descriptor, resourceIndex);

  if (!current) {
    // create
    const entry = createComputeGpuEntry(device, resources, resourceIndex, generation, descriptor);
    resources.buffers.set(resourceIndex, entry);
    return Object.freeze({
      kind: "created",
      resource_index: resourceIndex,
      generation,
    });
  }

  if (generation <= current.generation) {
    throw new FaberKernelContractError(
      "transition.generation",
      `replace requires generation > ${current.generation}, got ${generation}`,
      "product",
    );
  }

  // create-before-retire: allocate new, then retire old
  const next = createComputeGpuEntry(device, resources, resourceIndex, generation, descriptor);
  enqueueComputeRetire(resources, current);
  resources.buffers.set(resourceIndex, next);
  return Object.freeze({
    kind: "replaced",
    resource_index: resourceIndex,
    generation,
    previous_generation: current.generation,
  });
}

/**
 * After work that referenced retired buffers has been submitted, wait for
 * queue completion and destroy pending retired buffers.
 *
 * Snapshot/splice pendingRetire *before* awaiting onSubmittedWorkDone.
 * Destroy only that snapshot after completion.  Groups retired during the
 * wait stay in pendingRetire for a later completion that covers them.
 */
export async function destroyRetiredComputeResources(device, resources) {
  expectComputeResources(resources);
  if (resources.pendingRetire.length === 0) {
    return Object.freeze({ destroyed_groups: 0, destroyed_buffers: 0 });
  }

  const done = device?.queue?.onSubmittedWorkDone;
  if (typeof done !== "function") {
    throw new FaberKernelContractError(
      "queue.onSubmittedWorkDone",
      "queue completion is required before destroying retired compute buffers",
      "webgpu",
    );
  }

  // Take ownership of currently pending groups before waiting.  Concurrent
  // retires during the await must not be destroyed under this completion.
  const groups = resources.pendingRetire.splice(0, resources.pendingRetire.length);

  await done.call(device.queue);

  let destroyedBuffers = 0;
  for (const group of groups) {
    for (const buffer of group.buffers) {
      if (buffer && typeof buffer.destroy === "function" && !buffer.__faberDestroyed) {
        buffer.destroy();
        buffer.__faberDestroyed = true;
        destroyedBuffers += 1;
        resources.counters.destroyed += 1;
      }
    }
  }

  return Object.freeze({
    destroyed_groups: groups.length,
    destroyed_buffers: destroyedBuffers,
  });
}

// ── Compute lifecycle helpers ─────────────────────────────────────────────

function isEmptyComputePayload(descriptor) {
  return descriptor == null;
}

function normalizeComputePayload(descriptor, resourceIndex) {
  if (!descriptor || typeof descriptor !== "object") {
    throw new FaberKernelContractError(
      "transition.buffer_descriptor",
      `resource ${resourceIndex}: buffer descriptor is required`,
      "product",
    );
  }
  if (typeof descriptor.size !== "number" || descriptor.size <= 0) {
    throw new FaberKernelContractError(
      "transition.buffer_descriptor.size",
      `resource ${resourceIndex}: size must be a positive number`,
      "product",
    );
  }
  if (typeof descriptor.usage !== "number" || descriptor.usage <= 0) {
    throw new FaberKernelContractError(
      "transition.buffer_descriptor.usage",
      `resource ${resourceIndex}: usage must be a positive number`,
      "product",
    );
  }
  return {
    size: descriptor.size,
    usage: descriptor.usage,
    mappedAtCreation: descriptor.mappedAtCreation ?? false,
  };
}

// ── Graphics WebGPU effects ───────────────────────────────────────────────

/**
 * Create WebGPU resources for a graphics pipeline from an admitted graphics
 * descriptor and payload data. Reuses shared buffer and bind-group primitives
 * where ownership is identical to the compute path.
 *
 * Payload shape:
 *   { vertexBuffers: [{ slot, data: ArrayBuffer }],
 *     indexData: Uint16Array | Uint32Array,
 *     storageData: { [sourceName]: Float32Array } }
 */
export function createGraphicsResources(device, descriptor, payloads, canvasContext) {
  const currentTexture = canvasContext.getCurrentTexture();
  if (currentTexture.format !== EXPECTED_CANVAS_FORMAT) {
    throw new FaberKernelContractError(
      "canvasContext",
      `expected ${EXPECTED_CANVAS_FORMAT} canvas format, got ${currentTexture.format}`,
      "webgpu",
    );
  }

  const shaderModule = device.createShaderModule({ code: descriptor.wgsl });

  const storageBuffers = createStorageBuffers(device, descriptor, payloads.storageData ?? {});
  const bindGroupLayouts = createGraphicsBindGroupLayouts(device, descriptor);
  const pipelineLayout = createPipelineLayout(device, descriptor, bindGroupLayouts);
  const bindGroups = createBindGroups(device, descriptor, bindGroupLayouts, storageBuffers);

  const vertexBuffers = createVertexBuffers(device, descriptor, payloads.vertexBuffers ?? []);
  const { indexBuffer, indexCount } = createIndexBuffer(device, descriptor, payloads.indexData);

  const depthTexture = createDepthTexture(device, currentTexture.width, currentTexture.height);

  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: descriptor.kernels[0].entryName,
      buffers: descriptor.kernels[0].vertexBufferLayouts.map((layout) => ({
        arrayStride: layout.arrayStride,
        stepMode: layout.stepMode,
        attributes: layout.attributes.map((attr) => ({
          shaderLocation: attr.shaderLocation,
          format: attr.format,
          offset: attr.offset,
        })),
      })),
    },
    fragment: {
      module: shaderModule,
      entryPoint: descriptor.kernels[1].entryName,
      targets: descriptor.pipeline.colorTargetFormats.map((fmt) => ({
        format: fmt,
      })),
    },
    primitive: {
      topology: descriptor.pipeline.primitiveTopology,
      cullMode: "none",
    },
    depthStencil: {
      depthWriteEnabled: descriptor.pipeline.depthStencil.depthWriteEnabled,
      depthCompare: descriptor.pipeline.depthStencil.depthCompare,
      format: DEPTH_FORMAT,
    },
  });

  return Object.freeze({
    storageBuffers,
    vertexBuffers,
    indexBuffer,
    indexCount,
    bindGroupLayouts,
    pipelineLayout,
    shaderModule,
    pipeline,
    bindGroups,
    depthTexture,
  });
}

/**
 * Encode and submit one indexed render pass. Increments submittedFrameCount
 * on the frameState object.
 *
 * options.clearValue — optional GPUColor clear (default black).
 * options.recordSubmit — when true, append drawIndexed observation to frameState.submits.
 */
export function runGraphicsFrame(device, context, resources, descriptor, frameState, options = {}) {
  const textureView = context.getCurrentTexture().createView();
  const clearValue = options.clearValue ?? { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };

  const commandEncoder = device.createCommandEncoder();
  const renderPass = commandEncoder.beginRenderPass({
    colorAttachments: [
      {
        view: textureView,
        clearValue,
        loadOp: "clear",
        storeOp: "store",
      },
    ],
    depthStencilAttachment: {
      view: resources.depthTexture.createView(),
      depthClearValue: 1.0,
      depthLoadOp: "clear",
      depthStoreOp: "store",
    },
  });

  renderPass.setPipeline(resources.pipeline);

  for (const vb of resources.vertexBuffers) {
    renderPass.setVertexBuffer(vb.slot, vb.buffer);
  }

  renderPass.setIndexBuffer(
    resources.indexBuffer,
    descriptor.draw.indexFormat,
    0,
  );

  for (const group of resources.bindGroups) {
    renderPass.setBindGroup(group.bindGroupIndex, group.bindGroup);
  }

  const firstIndex = descriptor.draw.firstIndex;
  const indexCount = descriptor.draw.indexCount;
  const instanceCount = descriptor.draw.instanceCount;
  const baseVertex = descriptor.draw.baseVertex;
  if (firstIndex + indexCount > resources.indexCount) {
    throw new FaberKernelContractError(
      "drawManifest",
      `first_index ${firstIndex} + index_count ${indexCount} exceeds buffer index count ${resources.indexCount}`,
    );
  }

  renderPass.drawIndexed(
    indexCount,
    instanceCount,
    firstIndex,
    baseVertex,
    0,
  );

  renderPass.end();
  device.queue.submit([commandEncoder.finish()]);

  frameState.submittedFrameCount = (frameState.submittedFrameCount ?? 0) + 1;
  if (options.recordSubmit) {
    if (!Array.isArray(frameState.submits)) {
      frameState.submits = [];
    }
    frameState.submits.push({
      method: "drawIndexed",
      drawIndexed: true,
      index_count: indexCount,
      instance_count: instanceCount,
      first_index: firstIndex,
      base_vertex: baseVertex,
      depth_attachment: true,
      depth_test_enabled: descriptor.pipeline.depthStencil.depthWriteEnabled
        || descriptor.pipeline.depthStencil.depthCompare !== "always",
      depth_write_enabled: descriptor.pipeline.depthStencil.depthWriteEnabled,
      depth_compare: descriptor.pipeline.depthStencil.depthCompare,
      clear_value: clearValue,
      frame_index: frameState.submittedFrameCount,
    });
  }
}

/**
 * Read RGBA8 pixels from a canvas texture that was just drawn.
 * Must use the same GPUTexture instance as the render pass — calling
 * context.getCurrentTexture() again yields a new (empty) swapchain image.
 * Requires COPY_SRC usage on the canvas configuration.
 */
export async function readTexturePixelsRgba(device, texture, samples) {
  const bytesPerRow = 256; // WebGPU copy bytesPerRow alignment
  const results = [];
  for (const sample of samples) {
    const buffer = device.createBuffer({
      size: bytesPerRow,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer(
      { texture, origin: { x: sample.x, y: sample.y, z: 0 } },
      { buffer, bytesPerRow },
      { width: 1, height: 1, depthOrArrayLayers: 1 },
    );
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const bgra = new Uint8Array(buffer.getMappedRange().slice(0, 4));
    buffer.unmap();
    buffer.destroy();
    results.push({
      name: sample.name,
      x: sample.x,
      y: sample.y,
      r: bgra[2],
      g: bgra[1],
      b: bgra[0],
      a: bgra[3],
      hex: `#${[bgra[2], bgra[1], bgra[0]].map((v) => v.toString(16).padStart(2, "0")).join("")}`,
    });
  }
  return results;
}

/**
 * Encode + submit one indexed pass. When options.pixelSamples is provided,
 * copies those pixels in the same command encoder (before swapchain expiry)
 * and returns { texture, pixelBuffers } for later mapAsync readback.
 */
export function runGraphicsFrameWithTexture(device, context, resources, descriptor, frameState, options = {}) {
  const texture = context.getCurrentTexture();
  const textureView = texture.createView();
  const clearValue = options.clearValue ?? { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
  const pixelSamples = options.pixelSamples ?? null;

  const commandEncoder = device.createCommandEncoder();
  const renderPass = commandEncoder.beginRenderPass({
    colorAttachments: [
      {
        view: textureView,
        clearValue,
        loadOp: "clear",
        storeOp: "store",
      },
    ],
    depthStencilAttachment: {
      view: resources.depthTexture.createView(),
      depthClearValue: 1.0,
      depthLoadOp: "clear",
      depthStoreOp: "store",
    },
  });

  renderPass.setPipeline(resources.pipeline);
  for (const vb of resources.vertexBuffers) {
    renderPass.setVertexBuffer(vb.slot, vb.buffer);
  }
  renderPass.setIndexBuffer(resources.indexBuffer, descriptor.draw.indexFormat, 0);
  for (const group of resources.bindGroups) {
    renderPass.setBindGroup(group.bindGroupIndex, group.bindGroup);
  }

  const firstIndex = descriptor.draw.firstIndex;
  const indexCount = descriptor.draw.indexCount;
  const instanceCount = descriptor.draw.instanceCount;
  const baseVertex = descriptor.draw.baseVertex;
  if (firstIndex + indexCount > resources.indexCount) {
    throw new FaberKernelContractError(
      "drawManifest",
      `first_index ${firstIndex} + index_count ${indexCount} exceeds buffer index count ${resources.indexCount}`,
    );
  }

  renderPass.drawIndexed(indexCount, instanceCount, firstIndex, baseVertex, 0);
  renderPass.end();

  // Copy pixels in the same encoder so the swapchain texture is still current.
  const pixelBuffers = [];
  const bytesPerRow = 256;
  if (Array.isArray(pixelSamples)) {
    for (const sample of pixelSamples) {
      const buffer = device.createBuffer({
        size: bytesPerRow,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      commandEncoder.copyTextureToBuffer(
        { texture, origin: { x: sample.x, y: sample.y, z: 0 } },
        { buffer, bytesPerRow },
        { width: 1, height: 1, depthOrArrayLayers: 1 },
      );
      pixelBuffers.push({ sample, buffer });
    }
  }

  device.queue.submit([commandEncoder.finish()]);

  frameState.submittedFrameCount = (frameState.submittedFrameCount ?? 0) + 1;
  if (options.recordSubmit) {
    if (!Array.isArray(frameState.submits)) frameState.submits = [];
    frameState.submits.push({
      method: "drawIndexed",
      drawIndexed: true,
      index_count: indexCount,
      instance_count: instanceCount,
      first_index: firstIndex,
      base_vertex: baseVertex,
      depth_attachment: true,
      depth_test_enabled: descriptor.pipeline.depthStencil.depthWriteEnabled
        || descriptor.pipeline.depthStencil.depthCompare !== "always",
      depth_write_enabled: descriptor.pipeline.depthStencil.depthWriteEnabled,
      depth_compare: descriptor.pipeline.depthStencil.depthCompare,
      clear_value: clearValue,
      frame_index: frameState.submittedFrameCount,
    });
  }

  return { texture, pixelBuffers };
}

/** Map pixel buffers produced by runGraphicsFrameWithTexture into RGBA samples. */
export async function mapPixelBuffers(pixelBuffers) {
  const results = [];
  for (const { sample, buffer } of pixelBuffers) {
    await buffer.mapAsync(GPUMapMode.READ);
    const bgra = new Uint8Array(buffer.getMappedRange().slice(0, 4));
    buffer.unmap();
    buffer.destroy();
    results.push({
      name: sample.name,
      x: sample.x,
      y: sample.y,
      r: bgra[2],
      g: bgra[1],
      b: bgra[0],
      a: bgra[3],
      hex: `#${[bgra[2], bgra[1], bgra[0]].map((v) => v.toString(16).padStart(2, "0")).join("")}`,
    });
  }
  return results;
}

/**
 * Replace the depth texture after a physical canvas resize. Destroys the old
 * texture and returns a new resources object with the updated depth texture.
 */
export function replaceDepthTextureOnResize(device, resources, width, height) {
  if (resources.depthTexture) {
    resources.depthTexture.destroy();
  }
  const depthTexture = createDepthTexture(device, width, height);
  return Object.freeze({
    ...resources,
    depthTexture,
  });
}

/**
 * Register a callback for device loss. The callback receives a structured
 * loss info object with kind, reason, and message.
 */
export function onDeviceLost(device, callback) {
  device.lost.then((info) => {
    callback(
      Object.freeze({
        kind: "device-lost",
        reason: info.reason,
        message: info.message,
      }),
    );
  });
}

// ── Graphics resource helpers ─────────────────────────────────────────────

function createGraphicsBindGroupLayouts(device, descriptor) {
  const layouts = new Map();

  for (const layout of descriptor.bindGroupLayouts) {
    const entries = layout.entries.map((entry) => ({
      binding: entry.binding,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
      buffer: {
        type: entry.bufferType,
        hasDynamicOffset: false,
        minBindingSize: entry.minBindingSize,
      },
    }));
    layouts.set(layout.bindGroupIndex, device.createBindGroupLayout({ entries }));
  }

  return layouts;
}

function createVertexBuffers(device, descriptor, vertexPayloads) {
  const vertexKernel = descriptor.kernels[0];
  const buffers = [];

  // Indexed draws address unique vertices. pipeline.vertexCount is the draw
  // element total (e.g. 36), not the unique buffer length (e.g. 8 corners).
  // Require at least one full vertex and stride alignment; index bounds are
  // checked when encoding the draw.
  for (const payload of vertexPayloads) {
    const layout = vertexKernel.vertexBufferLayouts.find(
      (vbl) => vbl.bufferIndex === payload.slot,
    );
    if (!layout) {
      throw new FaberKernelContractError(
        "payloads.vertexBuffers",
        `no vertex buffer layout for slot ${payload.slot}`,
      );
    }

    const data = payload.data instanceof ArrayBuffer
      ? new Uint8Array(payload.data)
      : new Uint8Array(payload.data.buffer, payload.data.byteOffset, payload.data.byteLength);

    if (layout.arrayStride <= 0 || data.byteLength < layout.arrayStride) {
      throw new FaberKernelContractError(
        "payloads.vertexBuffers",
        `expected at least one vertex (${layout.arrayStride} bytes) for slot ${payload.slot}, got ${data.byteLength}`,
      );
    }
    if (data.byteLength % layout.arrayStride !== 0) {
      throw new FaberKernelContractError(
        "payloads.vertexBuffers",
        `slot ${payload.slot} byte length ${data.byteLength} is not a multiple of stride ${layout.arrayStride}`,
      );
    }

    const buffer = device.createBuffer({
      size: data.byteLength,
      usage: BUFFER_USAGE.vertex(),
      mappedAtCreation: true,
    });
    new Uint8Array(buffer.getMappedRange()).set(data);
    buffer.unmap();

    buffers.push(Object.freeze({ slot: payload.slot, buffer }));
  }

  return Object.freeze(buffers);
}

function createIndexBuffer(device, descriptor, indexData) {
  if (!indexData) {
    throw new FaberKernelContractError(
      "payloads.indexData",
      "index data is required for indexed draw",
    );
  }

  const data = indexData instanceof ArrayBuffer
    ? new Uint8Array(indexData)
    : new Uint8Array(indexData.buffer, indexData.byteOffset, indexData.byteLength);

  const indexByteSize = descriptor.draw.indexFormat === "uint16" ? 2 : 4;
  const indexCount = Math.floor(data.byteLength / indexByteSize);

  if (indexCount === 0) {
    throw new FaberKernelContractError(
      "payloads.indexData",
      `index data too short for ${descriptor.draw.indexFormat} format`,
    );
  }

  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: BUFFER_USAGE.index(),
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(data);
  buffer.unmap();

  return Object.freeze({ indexBuffer: buffer, indexCount });
}

function createDepthTexture(device, width, height) {
  return device.createTexture({
    size: { width, height },
    format: DEPTH_FORMAT,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
}

function createStorageBuffers(device, descriptor, storageData) {
  const buffers = new Map();

  for (const group of descriptor.bindGroups) {
    for (const entry of group.entries) {
      if (buffers.has(entry.resourceIndex)) {
        continue;
      }

      const buffer = device.createBuffer({
        size: entry.bufferByteLen,
        usage: bufferUsageForRole(entry.role),
        mappedAtCreation: entry.role === "input",
      });

      if (entry.role === "input") {
        writeGraphicsStorageInput(buffer, entry, storageData);
      }

      buffers.set(entry.resourceIndex, buffer);
    }
  }

  return buffers;
}

function writeGraphicsStorageInput(buffer, entry, storageData) {
  const inputName = entry.sourceName ?? `resource_${entry.resourceIndex}`;
  const value = storageData[inputName];
  if (!(value instanceof Float32Array)) {
    throw new FaberKernelContractError(
      `payloads.storageData.${inputName}`,
      "expected Float32Array",
    );
  }
  if (value.byteLength > entry.bufferByteLen) {
    throw new FaberKernelContractError(
      `payloads.storageData.${inputName}`,
      `expected at most ${entry.bufferByteLen} bytes, got ${value.byteLength}`,
    );
  }

  const target = new Uint8Array(buffer.getMappedRange());
  target.set(new Uint8Array(value.buffer, value.byteOffset, value.byteLength));
  buffer.unmap();
}

// ── Per-chunk resource lifecycle (HV-07B) ─────────────────────────────────
//
// Frozen replace payload contract (HV-07A ↔ HV-07B):
//   logical_id  = stable chunk index
//   generation  = advances on full mesh payload change (not face_count-only)
//   payload     = { positions, colors, indices } OR empty (remove)
//
// Create-before-retire. Destroy only after queue.onSubmittedWorkDone (or
// equivalent completion promise). Frame-count dispose is forbidden.
// Admitted path is per-chunk multi-draw — not concatenated-single-buffer.

const COMPUTE_RESOURCE_PATH = "compute";
const CHUNK_BUFFERS_PER_PAIR = 3; // position VB + color VB + index buffer
const CHUNK_RESOURCE_PATH = "per-chunk-multi-draw";

/**
 * Create graphics resources for the admitted per-chunk multi-draw path.
 * Shared pipeline / storage / depth are owned once; mesh buffers live in the
 * chunk map and are replaced independently via applyChunkResourceReplace.
 *
 * basePayloads: { storageData } only — no world-level vertex/index upload.
 */
export function createChunkGraphicsResources(device, descriptor, basePayloads, canvasContext) {
  const currentTexture = canvasContext.getCurrentTexture();
  if (currentTexture.format !== EXPECTED_CANVAS_FORMAT) {
    throw new FaberKernelContractError(
      "canvasContext",
      `expected ${EXPECTED_CANVAS_FORMAT} canvas format, got ${currentTexture.format}`,
      "webgpu",
    );
  }

  const shaderModule = device.createShaderModule({ code: descriptor.wgsl });
  const storageBuffers = createStorageBuffers(device, descriptor, basePayloads?.storageData ?? {});
  const bindGroupLayouts = createGraphicsBindGroupLayouts(device, descriptor);
  const pipelineLayout = createPipelineLayout(device, descriptor, bindGroupLayouts);
  const bindGroups = createBindGroups(device, descriptor, bindGroupLayouts, storageBuffers);
  const depthTexture = createDepthTexture(device, currentTexture.width, currentTexture.height);

  const pipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: descriptor.kernels[0].entryName,
      buffers: descriptor.kernels[0].vertexBufferLayouts.map((layout) => ({
        arrayStride: layout.arrayStride,
        stepMode: layout.stepMode,
        attributes: layout.attributes.map((attr) => ({
          shaderLocation: attr.shaderLocation,
          format: attr.format,
          offset: attr.offset,
        })),
      })),
    },
    fragment: {
      module: shaderModule,
      entryPoint: descriptor.kernels[1].entryName,
      targets: descriptor.pipeline.colorTargetFormats.map((fmt) => ({
        format: fmt,
      })),
    },
    primitive: {
      topology: descriptor.pipeline.primitiveTopology,
      cullMode: "none",
    },
    depthStencil: {
      depthWriteEnabled: descriptor.pipeline.depthStencil.depthWriteEnabled,
      depthCompare: descriptor.pipeline.depthStencil.depthCompare,
      format: DEPTH_FORMAT,
    },
  });

  return {
    storageBuffers,
    bindGroupLayouts,
    pipelineLayout,
    shaderModule,
    pipeline,
    bindGroups,
    depthTexture,
    /** Draw index format for per-chunk indexCount (not byteLength heuristics). */
    indexFormat: descriptor.draw.indexFormat,
    /** @type {Map<number, object>} */
    chunks: new Map(),
    /** @type {Array<object>} */
    pendingRetire: [],
    counters: {
      created: 0,
      live: 0,
      retired: 0,
      destroyed: 0,
    },
    path: CHUNK_RESOURCE_PATH,
  };
}

/**
 * Snapshot of honest buffer counters for a chunk graphics session.
 * created/live/retired/destroyed count individual GPU buffers (3 per pair).
 */
export function chunkResourceCounters(resources) {
  expectChunkResources(resources);
  return Object.freeze({
    created: resources.counters.created,
    live: resources.counters.live,
    retired: resources.counters.retired,
    destroyed: resources.counters.destroyed,
    live_chunks: resources.chunks.size,
    pending_retire_groups: resources.pendingRetire.length,
    path: resources.path,
  });
}

/** Live chunk ids in ascending order. */
export function liveChunkIds(resources) {
  expectChunkResources(resources);
  return Object.freeze([...resources.chunks.keys()].sort((a, b) => a - b));
}

/** Generation + index_count for one live chunk, or null if absent. */
export function chunkResourceSnapshot(resources, logicalId) {
  expectChunkResources(resources);
  const entry = resources.chunks.get(logicalId);
  if (!entry) {
    return null;
  }
  return Object.freeze({
    logical_id: entry.logicalId,
    generation: entry.generation,
    index_count: entry.indexCount,
    buffer_count: CHUNK_BUFFERS_PER_PAIR,
  });
}

/**
 * Apply one chunk resource transition under the frozen payload contract.
 *
 * transition = {
 *   logical_id: number,
 *   generation: number,
 *   payload: null | undefined | { positions, colors, indices }
 * }
 *
 * Empty payload removes the live resource (retire previous after submit).
 * Non-empty payload creates or replaces (create-before-retire).
 * Invalid transitions throw FaberKernelContractError kind=product.
 */
export function applyChunkResourceReplace(device, resources, transition) {
  expectChunkResources(resources);
  if (!device?.createBuffer) {
    throw new FaberKernelContractError("device", "device is required for chunk replace", "product");
  }

  const logicalId = expectNonNegativeInt(transition?.logical_id, "transition.logical_id");
  const generation = expectNonNegativeInt(transition?.generation, "transition.generation");
  const empty = isEmptyChunkPayload(transition?.payload);
  const current = resources.chunks.get(logicalId) ?? null;

  if (empty) {
    if (!current) {
      throw new FaberKernelContractError(
        "transition",
        `cannot remove logical_id ${logicalId}: no live resource`,
        "product",
      );
    }
    if (generation !== current.generation) {
      throw new FaberKernelContractError(
        "transition.generation",
        `remove requires generation ${current.generation}, got ${generation}`,
        "product",
      );
    }
    enqueueRetire(resources, current);
    resources.chunks.delete(logicalId);
    return Object.freeze({
      kind: "removed",
      logical_id: logicalId,
      generation,
      previous_generation: current.generation,
    });
  }

  const mesh = normalizeChunkPayload(transition.payload, logicalId);

  if (!current) {
    // create
    const entry = createChunkGpuEntry(device, resources, logicalId, generation, mesh);
    resources.chunks.set(logicalId, entry);
    return Object.freeze({
      kind: "created",
      logical_id: logicalId,
      generation,
      index_count: entry.indexCount,
    });
  }

  if (generation <= current.generation) {
    throw new FaberKernelContractError(
      "transition.generation",
      `replace requires generation > ${current.generation}, got ${generation}`,
      "product",
    );
  }

  // create-before-retire: allocate new, then retire old
  const next = createChunkGpuEntry(device, resources, logicalId, generation, mesh);
  enqueueRetire(resources, current);
  resources.chunks.set(logicalId, next);
  return Object.freeze({
    kind: "replaced",
    logical_id: logicalId,
    generation,
    previous_generation: current.generation,
    index_count: next.indexCount,
  });
}

/**
 * After work that referenced retired buffers has been submitted, wait for
 * queue completion and destroy pending retired buffers.
 * Must not be called as a frame-count guess — only with real completion.
 *
 * Snapshot/splice pendingRetire *before* awaiting onSubmittedWorkDone.
 * Destroy only that snapshot after completion. Groups retired during the
 * wait stay in pendingRetire for a later completion that covers them.
 */
export async function destroyRetiredChunkResources(device, resources) {
  expectChunkResources(resources);
  if (resources.pendingRetire.length === 0) {
    return Object.freeze({ destroyed_groups: 0, destroyed_buffers: 0 });
  }

  const done = device?.queue?.onSubmittedWorkDone;
  if (typeof done !== "function") {
    throw new FaberKernelContractError(
      "queue.onSubmittedWorkDone",
      "queue completion is required before destroying retired chunk buffers",
      "webgpu",
    );
  }

  // Take ownership of currently pending groups before waiting. Concurrent
  // retires during the await must not be destroyed under this completion.
  const groups = resources.pendingRetire.splice(0, resources.pendingRetire.length);

  await done.call(device.queue);

  let destroyedBuffers = 0;
  for (const group of groups) {
    for (const buffer of group.buffers) {
      if (buffer && typeof buffer.destroy === "function" && !buffer.__faberDestroyed) {
        buffer.destroy();
        buffer.__faberDestroyed = true;
        destroyedBuffers += 1;
        resources.counters.destroyed += 1;
      }
    }
  }

  return Object.freeze({
    destroyed_groups: groups.length,
    destroyed_buffers: destroyedBuffers,
  });
}

/**
 * Encode + submit one multi-draw render pass: one drawIndexed per live chunk.
 * Closes the concatenated-single-buffer residual for the admitted path.
 *
 * options.clearValue — optional GPUColor clear (default black).
 * options.recordSubmit — when true, append multi-draw observation to frameState.submits.
 */
export function runChunkGraphicsFrame(device, context, resources, descriptor, frameState, options = {}) {
  expectChunkResources(resources);
  const textureView = context.getCurrentTexture().createView();
  const clearValue = options.clearValue ?? { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };

  const commandEncoder = device.createCommandEncoder();
  const renderPass = commandEncoder.beginRenderPass({
    colorAttachments: [
      {
        view: textureView,
        clearValue,
        loadOp: "clear",
        storeOp: "store",
      },
    ],
    depthStencilAttachment: {
      view: resources.depthTexture.createView(),
      depthClearValue: 1.0,
      depthLoadOp: "clear",
      depthStoreOp: "store",
    },
  });

  renderPass.setPipeline(resources.pipeline);
  for (const group of resources.bindGroups) {
    renderPass.setBindGroup(group.bindGroupIndex, group.bindGroup);
  }

  const instanceCount = descriptor.draw.instanceCount;
  const baseVertex = descriptor.draw.baseVertex;
  const draws = [];
  const ordered = [...resources.chunks.values()].sort((a, b) => a.logicalId - b.logicalId);

  for (const entry of ordered) {
    for (const vb of entry.vertexBuffers) {
      renderPass.setVertexBuffer(vb.slot, vb.buffer);
    }
    renderPass.setIndexBuffer(entry.indexBuffer, descriptor.draw.indexFormat, 0);
    renderPass.drawIndexed(entry.indexCount, instanceCount, 0, baseVertex, 0);
    draws.push({
      logical_id: entry.logicalId,
      generation: entry.generation,
      index_count: entry.indexCount,
    });
  }

  renderPass.end();
  device.queue.submit([commandEncoder.finish()]);

  frameState.submittedFrameCount = (frameState.submittedFrameCount ?? 0) + 1;
  if (options.recordSubmit) {
    if (!Array.isArray(frameState.submits)) {
      frameState.submits = [];
    }
    frameState.submits.push({
      method: "drawIndexed",
      multi_draw: true,
      path: CHUNK_RESOURCE_PATH,
      draw_count: draws.length,
      draws,
      instance_count: instanceCount,
      base_vertex: baseVertex,
      depth_attachment: true,
      depth_test_enabled: descriptor.pipeline.depthStencil.depthWriteEnabled
        || descriptor.pipeline.depthStencil.depthCompare !== "always",
      depth_write_enabled: descriptor.pipeline.depthStencil.depthWriteEnabled,
      depth_compare: descriptor.pipeline.depthStencil.depthCompare,
      clear_value: clearValue,
      frame_index: frameState.submittedFrameCount,
    });
  }

  return Object.freeze({ draw_count: draws.length, draws });
}

// ── Chunk lifecycle helpers ───────────────────────────────────────────────

function expectChunkResources(resources) {
  if (!resources || resources.path !== CHUNK_RESOURCE_PATH) {
    throw new FaberKernelContractError(
      "resources",
      "expected createChunkGraphicsResources session (per-chunk-multi-draw path)",
      "product",
    );
  }
  if (!(resources.chunks instanceof Map) || !Array.isArray(resources.pendingRetire) || !resources.counters) {
    throw new FaberKernelContractError(
      "resources",
      "chunk resource session is missing map/counters",
      "product",
    );
  }
}

function expectNonNegativeInt(value, path) {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
    throw new FaberKernelContractError(path, `expected non-negative integer, got ${value}`, "product");
  }
  return value;
}

function isEmptyChunkPayload(payload) {
  if (payload == null) {
    return true;
  }
  if (payload.empty === true) {
    return true;
  }
  const positions = payload.positions;
  const colors = payload.colors;
  const indices = payload.indices;
  const posLen = byteLengthOf(positions);
  const colLen = byteLengthOf(colors);
  const idxLen = byteLengthOf(indices);
  if (posLen === 0 && colLen === 0 && idxLen === 0) {
    return true;
  }
  return false;
}

function byteLengthOf(data) {
  if (data == null) {
    return 0;
  }
  if (data instanceof ArrayBuffer) {
    return data.byteLength;
  }
  if (ArrayBuffer.isView(data)) {
    return data.byteLength;
  }
  return -1;
}

function asUint8View(data, path) {
  if (data instanceof ArrayBuffer) {
    return new Uint8Array(data);
  }
  if (ArrayBuffer.isView(data)) {
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  }
  throw new FaberKernelContractError(path, "expected ArrayBuffer or typed array", "product");
}

function normalizeChunkPayload(payload, logicalId) {
  if (!payload || typeof payload !== "object") {
    throw new FaberKernelContractError(
      "transition.payload",
      `logical_id ${logicalId}: non-empty payload required`,
      "product",
    );
  }
  if (payload.positions == null || payload.colors == null || payload.indices == null) {
    throw new FaberKernelContractError(
      "transition.payload",
      `logical_id ${logicalId}: positions, colors, and indices are required`,
      "product",
    );
  }

  const positions = asUint8View(payload.positions, "transition.payload.positions");
  const colors = asUint8View(payload.colors, "transition.payload.colors");
  const indices = asUint8View(payload.indices, "transition.payload.indices");

  if (positions.byteLength === 0 || colors.byteLength === 0 || indices.byteLength === 0) {
    throw new FaberKernelContractError(
      "transition.payload",
      `logical_id ${logicalId}: partial empty payload is invalid (use fully empty to remove)`,
      "product",
    );
  }
  if (positions.byteLength % 12 !== 0) {
    throw new FaberKernelContractError(
      "transition.payload.positions",
      `byte length ${positions.byteLength} is not a multiple of float32x3 stride 12`,
      "product",
    );
  }
  if (colors.byteLength % 12 !== 0) {
    throw new FaberKernelContractError(
      "transition.payload.colors",
      `byte length ${colors.byteLength} is not a multiple of float32x3 stride 12`,
      "product",
    );
  }
  if (positions.byteLength !== colors.byteLength) {
    throw new FaberKernelContractError(
      "transition.payload",
      `positions (${positions.byteLength}) and colors (${colors.byteLength}) byte lengths must match`,
      "product",
    );
  }
  if (indices.byteLength % 4 !== 0 && indices.byteLength % 2 !== 0) {
    throw new FaberKernelContractError(
      "transition.payload.indices",
      `index byte length ${indices.byteLength} is not a multiple of 2 or 4`,
      "product",
    );
  }

  return Object.freeze({ positions, colors, indices });
}

function createMappedBuffer(device, data, usage) {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

function createChunkGpuEntry(device, resources, logicalId, generation, mesh) {
  const positionBuffer = createMappedBuffer(device, mesh.positions, BUFFER_USAGE.vertex());
  const colorBuffer = createMappedBuffer(device, mesh.colors, BUFFER_USAGE.vertex());
  const indexBuffer = createMappedBuffer(device, mesh.indices, BUFFER_USAGE.index());

  // Match createIndexBuffer: indexCount from descriptor draw indexFormat.
  const indexFormat = resources.indexFormat;
  const indexByteSize = indexFormat === "uint16" ? 2 : 4;
  if (mesh.indices.byteLength % indexByteSize !== 0) {
    positionBuffer.destroy();
    colorBuffer.destroy();
    indexBuffer.destroy();
    throw new FaberKernelContractError(
      "transition.payload.indices",
      `index byte length ${mesh.indices.byteLength} is not a multiple of ${indexByteSize} for ${indexFormat}`,
      "product",
    );
  }
  const indexCount = Math.floor(mesh.indices.byteLength / indexByteSize);
  if (indexCount === 0) {
    positionBuffer.destroy();
    colorBuffer.destroy();
    indexBuffer.destroy();
    throw new FaberKernelContractError(
      "transition.payload.indices",
      `index data too short for ${indexFormat} format`,
      "product",
    );
  }

  resources.counters.created += CHUNK_BUFFERS_PER_PAIR;
  resources.counters.live += CHUNK_BUFFERS_PER_PAIR;

  return {
    logicalId,
    generation,
    vertexBuffers: Object.freeze([
      Object.freeze({ slot: 0, buffer: positionBuffer }),
      Object.freeze({ slot: 1, buffer: colorBuffer }),
    ]),
    indexBuffer,
    indexCount,
    indexFormat,
    buffers: [positionBuffer, colorBuffer, indexBuffer],
  };
}

function enqueueRetire(resources, entry) {
  resources.pendingRetire.push({
    logicalId: entry.logicalId,
    generation: entry.generation,
    buffers: entry.buffers.slice(),
  });
  resources.counters.live -= CHUNK_BUFFERS_PER_PAIR;
  resources.counters.retired += CHUNK_BUFFERS_PER_PAIR;
}
