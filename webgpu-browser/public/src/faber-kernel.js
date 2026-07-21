/** @typedef {"artifact-fetch" | "reflection" | "webgpu" | "product"} FaberKernelErrorKind */

export class FaberKernelContractError extends Error {
  /**
   * @param {string} path
   * @param {string} message
   * @param {FaberKernelErrorKind} [kind]
   */
  constructor(path, message, kind = "reflection") {
    super(`${path}: ${message}`);
    this.name = "FaberKernelContractError";
    this.path = path;
    this.kind = kind;
  }
}

export async function fetchFaberKernelArtifacts({
  wgslUrl = "./generated/kernel.wgsl",
  reflectionUrl = "./generated/reflection.json",
  fetchImpl = globalThis.fetch,
} = {}) {
  if (typeof fetchImpl !== "function") {
    throw new FaberKernelContractError("fetch", "browser fetch is unavailable", "artifact-fetch");
  }

  const [wgslResponse, reflectionResponse] = await Promise.all([
    fetchImpl(wgslUrl),
    fetchImpl(reflectionUrl),
  ]);

  if (!wgslResponse.ok) {
    throw new FaberKernelContractError("wgsl", `failed to fetch ${wgslUrl}`, "artifact-fetch");
  }
  if (!reflectionResponse.ok) {
    throw new FaberKernelContractError(
      "reflection",
      `failed to fetch ${reflectionUrl}`,
      "artifact-fetch",
    );
  }

  return {
    wgsl: await wgslResponse.text(),
    reflection: await reflectionResponse.json(),
  };
}

export function loadFaberKernel({ wgsl, reflection }) {
  const document = parseReflection(reflection);
  expectValue(document.schema_version, 1, "reflection.schema_version");
  expectValue(document.target, "wgsl-text", "reflection.target");

  const kernels = expectArray(document.kernels, "reflection.kernels");
  if (kernels.length !== 1) {
    throw new FaberKernelContractError("reflection.kernels", `expected one kernel, got ${kernels.length}`);
  }

  const kernel = expectObject(kernels[0], "reflection.kernels[0]");
  const launch = expectObject(kernel.launch, "reflection.kernels[0].launch");
  const adapter = expectObject(launch.webgpu_adapter, "reflection.kernels[0].launch.webgpu_adapter");

  expectValue(kernel.shader_stage, "compute", "reflection.kernels[0].shader_stage");
  expectValue(launch.shader_stage, "compute", "reflection.kernels[0].launch.shader_stage");
  const entryName = expectString(kernel.entry_name, "reflection.kernels[0].entry_name");
  expectValue(launch.entry_name, entryName, "reflection.kernels[0].launch.entry_name");

  const pipelineLayout = parsePipelineLayout(adapter.pipeline_layout_descriptor);
  const bindGroupLayouts = parseBindGroupLayouts(adapter);
  const bindGroups = parseBindGroups(adapter);
  const dispatchWorkgroups = parseDispatchWorkgroups(adapter);

  validateDescriptorIndexes(adapter, bindGroupLayouts, bindGroups, pipelineLayout);
  validateLayoutAndGroupEntries(bindGroupLayouts, bindGroups);

  return Object.freeze({
    wgsl: expectString(wgsl, "wgsl"),
    schemaVersion: document.schema_version,
    target: document.target,
    entryName,
    shaderStage: kernel.shader_stage,
    pipelineLayout,
    bindGroupLayouts,
    bindGroups,
    dispatchWorkgroups,
    inputBindings: bindGroups.flatMap((group) => group.entries.filter((entry) => entry.role === "input")),
    outputBindings: bindGroups.flatMap((group) => group.entries.filter((entry) => entry.role === "output")),
  });
}

function parseReflection(reflection) {
  if (typeof reflection === "string") {
    try {
      return JSON.parse(reflection);
    } catch (error) {
      throw new FaberKernelContractError("reflection", `invalid JSON: ${error.message}`);
    }
  }
  return expectObject(reflection, "reflection");
}

function parsePipelineLayout(rawLayout) {
  const layout = expectObject(rawLayout, "webgpu_adapter.pipeline_layout_descriptor");
  const indexes = expectArray(
    layout.bind_group_layout_indexes,
    "webgpu_adapter.pipeline_layout_descriptor.bind_group_layout_indexes",
  );
  expectCount(
    layout.bind_group_layout_count,
    indexes.length,
    "webgpu_adapter.pipeline_layout_descriptor.bind_group_layout_count",
  );
  expectCount(
    layout.bind_group_layout_index_count,
    indexes.length,
    "webgpu_adapter.pipeline_layout_descriptor.bind_group_layout_index_count",
  );

  return Object.freeze({
    bindGroupLayoutIndexes: indexes.map((index, position) =>
      expectNonNegativeInteger(
        index,
        `webgpu_adapter.pipeline_layout_descriptor.bind_group_layout_indexes[${position}]`,
      ),
    ),
  });
}

function parseBindGroupLayouts(adapter) {
  const layouts = expectArray(adapter.bind_group_layout_descriptors, "webgpu_adapter.bind_group_layout_descriptors");
  expectCount(
    adapter.bind_group_layout_descriptor_count,
    layouts.length,
    "webgpu_adapter.bind_group_layout_descriptor_count",
  );

  return layouts.map((layout, layoutPosition) => {
    const path = `webgpu_adapter.bind_group_layout_descriptors[${layoutPosition}]`;
    const object = expectObject(layout, path);
    const entries = expectArray(object.entries, `${path}.entries`).map((entry, entryPosition) =>
      parseLayoutEntry(entry, `${path}.entries[${entryPosition}]`),
    );

    expectCount(object.entry_count, entries.length, `${path}.entry_count`);
    expectCount(object.layout_entry_index_count, entries.length, `${path}.layout_entry_index_count`);
    expectIndexList(
      object.layout_entry_indexes,
      entries.map((entry) => entry.layoutEntryIndex),
      `${path}.layout_entry_indexes`,
    );

    return Object.freeze({
      bindGroupIndex: expectNonNegativeInteger(object.bind_group_index, `${path}.bind_group_index`),
      group: expectNonNegativeInteger(object.group, `${path}.group`),
      layoutEntryIndexes: entries.map((entry) => entry.layoutEntryIndex),
      entries,
    });
  });
}

function parseLayoutEntry(entry, path) {
  const object = expectObject(entry, path);
  expectValue(object.visibility, "compute", `${path}.visibility`);
  expectValue(object.has_dynamic_offset, false, `${path}.has_dynamic_offset`);
  expectOneOf(object.buffer_type, ["read-only-storage", "storage"], `${path}.buffer_type`);

  return Object.freeze({
    binding: expectNonNegativeInteger(object.binding, `${path}.binding`),
    bindingIndex: expectNonNegativeInteger(object.binding_index, `${path}.binding_index`),
    resourceIndex: expectNonNegativeInteger(object.resource_index, `${path}.resource_index`),
    layoutEntryIndex: expectNonNegativeInteger(object.layout_entry_index, `${path}.layout_entry_index`),
    bufferByteLen: expectPositiveInteger(object.buffer_byte_len, `${path}.buffer_byte_len`),
    bufferByteOffset: expectNonNegativeInteger(object.buffer_byte_offset, `${path}.buffer_byte_offset`),
    bindingByteLen: expectPositiveInteger(object.binding_byte_len, `${path}.binding_byte_len`),
    minBindingSize: expectPositiveInteger(object.min_binding_size, `${path}.min_binding_size`),
    bufferType: object.buffer_type,
    sourceLocal: nullableInteger(object.source_local, `${path}.source_local`),
    sourceName: nullableString(object.source_name, `${path}.source_name`),
  });
}

function parseBindGroups(adapter) {
  const groups = expectArray(adapter.bind_group_descriptors, "webgpu_adapter.bind_group_descriptors");
  expectCount(adapter.bind_group_descriptor_count, groups.length, "webgpu_adapter.bind_group_descriptor_count");

  return groups.map((group, groupPosition) => {
    const path = `webgpu_adapter.bind_group_descriptors[${groupPosition}]`;
    const object = expectObject(group, path);
    const entries = expectArray(object.entries, `${path}.entries`).map((entry, entryPosition) =>
      parseBindGroupEntry(entry, `${path}.entries[${entryPosition}]`),
    );

    expectCount(object.entry_count, entries.length, `${path}.entry_count`);
    expectCount(object.entry_index_count, entries.length, `${path}.entry_index_count`);
    expectIndexList(object.entry_indexes, entries.map((entry) => entry.bindingIndex), `${path}.entry_indexes`);

    return Object.freeze({
      bindGroupIndex: expectNonNegativeInteger(object.bind_group_index, `${path}.bind_group_index`),
      group: expectNonNegativeInteger(object.group, `${path}.group`),
      entryIndexes: entries.map((entry) => entry.bindingIndex),
      entries,
    });
  });
}

function parseBindGroupEntry(entry, path) {
  const object = expectObject(entry, path);
  expectValue(object.kind, "storage-buffer", `${path}.kind`);
  expectOneOf(object.role, ["input", "output"], `${path}.role`);
  expectOneOf(object.access, ["read", "write"], `${path}.access`);
  expectOneOf(object.shader_access, ["read", "read_write"], `${path}.shader_access`);
  expectValue(object.shader_visibility, "compute", `${path}.shader_visibility`);
  expectOneOf(object.buffer_type, ["read-only-storage", "storage"], `${path}.buffer_type`);
  expectValue(object.element_layout, "f32", `${path}.element_layout`);
  expectValue(object.element_byte_width, 4, `${path}.element_byte_width`);
  expectValue(object.has_dynamic_offset, false, `${path}.has_dynamic_offset`);
  validateRolePolicy(object, path);

  return Object.freeze({
    binding: expectNonNegativeInteger(object.binding, `${path}.binding`),
    bindingIndex: expectNonNegativeInteger(object.binding_index, `${path}.binding_index`),
    resourceIndex: expectNonNegativeInteger(object.resource_index, `${path}.resource_index`),
    kind: object.kind,
    role: object.role,
    access: object.access,
    shaderAccess: object.shader_access,
    shaderVisibility: object.shader_visibility,
    bufferType: object.buffer_type,
    elementLayout: object.element_layout,
    elementByteWidth: object.element_byte_width,
    elementCount: expectPositiveInteger(object.element_count, `${path}.element_count`),
    bufferByteLen: expectPositiveInteger(object.buffer_byte_len, `${path}.buffer_byte_len`),
    bufferByteOffset: expectNonNegativeInteger(object.buffer_byte_offset, `${path}.buffer_byte_offset`),
    bindingByteLen: expectPositiveInteger(object.binding_byte_len, `${path}.binding_byte_len`),
    minBindingSize: expectPositiveInteger(object.min_binding_size, `${path}.min_binding_size`),
    sourceLocal: nullableInteger(object.source_local, `${path}.source_local`),
    sourceName: nullableString(object.source_name, `${path}.source_name`),
  });
}

function parseDispatchWorkgroups(adapter) {
  expectValue(adapter.dispatch_workgroup_dimension_count, 3, "webgpu_adapter.dispatch_workgroup_dimension_count");
  const dispatch = expectObject(adapter.dispatch_workgroups, "webgpu_adapter.dispatch_workgroups");
  return Object.freeze({
    x: expectPositiveInteger(dispatch.x, "webgpu_adapter.dispatch_workgroups.x"),
    y: expectPositiveInteger(dispatch.y, "webgpu_adapter.dispatch_workgroups.y"),
    z: expectPositiveInteger(dispatch.z, "webgpu_adapter.dispatch_workgroups.z"),
  });
}

function validateDescriptorIndexes(adapter, layouts, groups, pipelineLayout) {
  expectCount(
    adapter.bind_group_layout_descriptor_index_count,
    layouts.length,
    "webgpu_adapter.bind_group_layout_descriptor_index_count",
  );
  expectIndexList(
    adapter.bind_group_layout_descriptor_indexes,
    layouts.map((layout) => layout.bindGroupIndex),
    "webgpu_adapter.bind_group_layout_descriptor_indexes",
  );
  expectCount(
    adapter.bind_group_descriptor_index_count,
    groups.length,
    "webgpu_adapter.bind_group_descriptor_index_count",
  );
  expectIndexList(
    adapter.bind_group_descriptor_indexes,
    groups.map((group) => group.bindGroupIndex),
    "webgpu_adapter.bind_group_descriptor_indexes",
  );
  expectIndexList(
    pipelineLayout.bindGroupLayoutIndexes,
    layouts.map((layout) => layout.bindGroupIndex),
    "webgpu_adapter.pipeline_layout_descriptor.bind_group_layout_indexes",
  );
}

function validateLayoutAndGroupEntries(layouts, groups) {
  if (layouts.length !== groups.length) {
    throw new FaberKernelContractError(
      "webgpu_adapter",
      "layout descriptor count must match bind group descriptor count",
    );
  }

  for (const layout of layouts) {
    const group = groups.find((candidate) => candidate.bindGroupIndex === layout.bindGroupIndex);
    if (!group) {
      throw new FaberKernelContractError(
        "webgpu_adapter.bind_group_descriptors",
        `missing bind group descriptor for layout ${layout.bindGroupIndex}`,
      );
    }
    if (group.group !== layout.group) {
      throw new FaberKernelContractError(
        `webgpu_adapter.bind_group_descriptors[${group.bindGroupIndex}].group`,
        `expected group ${layout.group}, got ${group.group}`,
      );
    }
    for (const layoutEntry of layout.entries) {
      const bindingEntry = group.entries.find((entry) => entry.bindingIndex === layoutEntry.bindingIndex);
      if (!bindingEntry) {
        throw new FaberKernelContractError(
          `webgpu_adapter.bind_group_descriptors[${group.bindGroupIndex}].entries`,
          `missing binding entry ${layoutEntry.bindingIndex}`,
        );
      }
      validateEntryPair(layoutEntry, bindingEntry);
    }
  }
}

function validateEntryPair(layoutEntry, bindingEntry) {
  const path = `webgpu_adapter.bind_group_descriptors[*].entries[${bindingEntry.bindingIndex}]`;
  expectValue(bindingEntry.binding, layoutEntry.binding, `${path}.binding`);
  expectValue(bindingEntry.resourceIndex, layoutEntry.resourceIndex, `${path}.resource_index`);
  expectValue(bindingEntry.bufferType, layoutEntry.bufferType, `${path}.buffer_type`);
  expectValue(bindingEntry.bufferByteLen, layoutEntry.bufferByteLen, `${path}.buffer_byte_len`);
  expectValue(bindingEntry.bufferByteOffset, layoutEntry.bufferByteOffset, `${path}.buffer_byte_offset`);
  expectValue(bindingEntry.bindingByteLen, layoutEntry.bindingByteLen, `${path}.binding_byte_len`);
  expectValue(bindingEntry.minBindingSize, layoutEntry.minBindingSize, `${path}.min_binding_size`);
}

function validateRolePolicy(object, path) {
  if (object.role === "input") {
    expectValue(object.access, "read", `${path}.access`);
    expectValue(object.shader_access, "read", `${path}.shader_access`);
    expectValue(object.buffer_type, "read-only-storage", `${path}.buffer_type`);
    return;
  }

  expectValue(object.role, "output", `${path}.role`);
  expectValue(object.access, "write", `${path}.access`);
  expectValue(object.shader_access, "read_write", `${path}.shader_access`);
  expectValue(object.buffer_type, "storage", `${path}.buffer_type`);
}

function expectObject(value, path) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new FaberKernelContractError(path, "expected object");
  }
  return value;
}

function expectArray(value, path) {
  if (!Array.isArray(value)) {
    throw new FaberKernelContractError(path, "expected array");
  }
  return value;
}

function expectString(value, path) {
  if (typeof value !== "string") {
    throw new FaberKernelContractError(path, "expected string");
  }
  return value;
}

function nullableString(value, path) {
  if (value == null) {
    return null;
  }
  return expectString(value, path);
}

function nullableInteger(value, path) {
  if (value == null) {
    return null;
  }
  return expectNonNegativeInteger(value, path);
}

function expectPositiveInteger(value, path) {
  const integer = expectNonNegativeInteger(value, path);
  if (integer === 0) {
    throw new FaberKernelContractError(path, "expected positive integer");
  }
  return integer;
}

function expectNonNegativeInteger(value, path) {
  if (!Number.isInteger(value) || value < 0) {
    throw new FaberKernelContractError(path, "expected non-negative integer");
  }
  return value;
}

function expectValue(value, expected, path) {
  if (value !== expected) {
    throw new FaberKernelContractError(path, `expected ${JSON.stringify(expected)}, got ${JSON.stringify(value)}`);
  }
}

function expectOneOf(value, accepted, path) {
  if (!accepted.includes(value)) {
    throw new FaberKernelContractError(path, `expected one of ${accepted.join(", ")}, got ${JSON.stringify(value)}`);
  }
}

function expectCount(value, expected, path) {
  expectValue(expectNonNegativeInteger(value, path), expected, path);
}

function expectIndexList(value, expected, path) {
  const actual = expectArray(value, path).map((index, position) =>
    expectNonNegativeInteger(index, `${path}[${position}]`),
  );
  expectCount(actual.length, expected.length, `${path}.length`);
  for (let index = 0; index < expected.length; index += 1) {
    expectValue(actual[index], expected[index], `${path}[${index}]`);
  }
}
