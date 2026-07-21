import { FaberKernelContractError } from "./faber-kernel.js";

const BUFFER_USAGE = {
  input: () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  output: () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  readback: () => GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
};

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
      visibility: GPUShaderStage.COMPUTE,
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
