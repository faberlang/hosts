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
  });
}

export async function runKernel(device, resources, descriptor) {
  const outputBinding = expectSingleOutputBinding(descriptor);
  const outputBuffer = resources.buffers.get(outputBinding.resourceIndex);
  if (!outputBuffer) {
    throw new FaberKernelContractError("resources.buffers", `missing output resource ${outputBinding.resourceIndex}`);
  }

  const readbackBuffer = device.createBuffer({
    size: outputBinding.bufferByteLen,
    usage: BUFFER_USAGE.readback(),
  });

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
  encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputBinding.bufferByteLen);

  device.queue.submit([encoder.finish()]);
  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const copy = readbackBuffer.getMappedRange().slice(0);
  readbackBuffer.unmap();

  const values = Array.from(new Float32Array(copy));
  return Object.freeze({ values, outputBinding });
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

      buffers.set(entry.resourceIndex, buffer);
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
      const buffer = buffers.get(entry.resourceIndex);
      if (!buffer) {
        throw new FaberKernelContractError("buffers", `missing resource ${entry.resourceIndex}`);
      }

      return {
        binding: entry.binding,
        resource: {
          buffer,
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
