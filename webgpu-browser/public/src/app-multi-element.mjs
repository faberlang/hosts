/**
 * Multi-element WGSL execution proof (U3-b).
 *
 * Dispatches the emitted post-U2 `add_one` kernel over N elements through
 * the real WebGPU host. The kernel's OOB guard reads the runtime-extent
 * binding (`extent_2[0]`), so this proof is precisely the host-side half of
 * the U2 channel: the host supplies the element count via a u32 extent
 * binding, dispatches by buffer length, and asserts all N elements read back
 * correctly.
 *
 * The reflection's static 4-byte buffer sizes (single-element hint) are
 * overridden with N-element sizes; the reflection's static (1,1,1)
 * `dispatch_workgroups` hint is overridden with { x: N }. Both overrides are
 * the documented host contract (U2 done_when (e)): the static shape is a
 * hint, the host supplies the real extent.
 *
 * Expected: input [0.0..15.0] → output [1.0..16.0].
 */

import {
  FaberKernelContractError,
  fetchFaberKernelArtifacts,
  loadFaberKernel,
} from "./contract/artifact-admission.js";
import {
  acquireWebGpuDevice,
  createWebGpuResources,
  runKernel,
} from "./backend/webgpu-runtime.js";

const N = 16;
const F32_BYTES = 4;
const EPSILON = 0.000_001;

const INPUT = Array.from({ length: N }, (_, i) => i); // 0..15
const EXPECTED = INPUT.map((value) => value + 1);     // 1..16

window.faberWebGpuMultiElementProof = Object.freeze({ ok: false, status: "starting" });

main().catch((error) => {
  const proof = proofFailure(error);
  window.faberWebGpuMultiElementProof = proof;
  console.log("FABER_MULTI_ELEMENT_PROOF:", JSON.stringify(proof));
});

async function main() {
  const artifacts = await fetchFaberKernelArtifacts();
  const kernel = loadFaberKernel(artifacts);

  // The admission must have carried the post-U2 runtime-extent binding.
  const extentEntry = kernel.bindGroups[0].entries.find(
    (entry) => entry.kind === "runtime-extent",
  );
  if (!extentEntry) {
    throw new FaberKernelContractError(
      "reflection",
      "runtime-extent binding missing from reflection (pre-U2 kernel?)",
      "reflection",
    );
  }

  // Size the storage buffers for N elements and dispatch by buffer length.
  const descriptor = sizeForMultiElement(kernel, N);

  const { device } = await acquireWebGpuDevice();
  const resources = createWebGpuResources(device, descriptor, {
    x: new Float32Array(INPUT),
    runtime_extent: new Uint32Array([N]),
  });
  const result = await runKernel(device, resources, descriptor);
  const values = result.values;

  if (values.length !== N) {
    throw new FaberKernelContractError(
      "readback",
      `expected ${N} output values, got ${values.length}`,
      "product",
    );
  }

  const failures = [];
  for (let i = 0; i < N; i++) {
    if (Math.abs(values[i] - EXPECTED[i]) > EPSILON) {
      failures.push(`[${i}]: expected ${EXPECTED[i]}, got ${values[i]}`);
    }
  }
  if (failures.length > 0) {
    throw new FaberKernelContractError(
      "readback",
      "multi-element mismatch:\n  " + failures.join("\n  "),
      "product",
    );
  }

  window.faberWebGpuMultiElementProof = Object.freeze({
    ok: true,
    status: "ready",
    kind: "ok",
    entryName: kernel.entryName,
    n: N,
    values,
    expected: EXPECTED,
    dispatchWorkgroups: descriptor.dispatchWorkgroups,
    extentValue: N,
  });
  console.log("FABER_MULTI_ELEMENT_PROOF:", JSON.stringify(window.faberWebGpuMultiElementProof));
}

/**
 * Clone the admitted kernel descriptor with multi-element sizes. Storage
 * buffers grow to N*f32 bytes; the runtime-extent binding stays a single
 * u32 (4 bytes) carrying the host-supplied count. The dispatch hint becomes
 * { x: N } so execution covers all N elements.
 */
function sizeForMultiElement(kernel, n) {
  const bytes = n * F32_BYTES;
  const resizeEntry = (entry) =>
    entry.kind === "runtime-extent"
      ? { ...entry }
      : { ...entry, bufferByteLen: bytes, bindingByteLen: bytes, minBindingSize: bytes };

  return Object.freeze({
    ...kernel,
    bindGroups: kernel.bindGroups.map((group) =>
      Object.freeze({
        ...group,
        entries: group.entries.map((entry) => Object.freeze(resizeEntry(entry))),
      }),
    ),
    outputBindings: kernel.outputBindings.map((binding) =>
      Object.freeze({ ...binding, bufferByteLen: bytes }),
    ),
    dispatchWorkgroups: Object.freeze({ x: n, y: 1, z: 1 }),
  });
}

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
