#!/usr/bin/env node
/**
 * Focused product-boundary admission for webgpu-browser.
 *
 * Runs under Node without a browser GPU. Covers:
 * - artifact-fetch failure
 * - unsupported / rejected reflection
 * - unavailable WebGPU
 *
 * Does not claim browser GPU execution or exact add-one readback (those remain
 * controlled/manual browser inspection evidence).
 */

import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const publicDir = path.resolve(here, "..");
const generatedDir = path.join(publicDir, "generated");

const { FaberKernelContractError, fetchFaberKernelArtifacts, loadFaberKernel } = await import(
  pathToFileURL(path.join(here, "faber-kernel.js")).href
);
const { acquireWebGpuDevice } = await import(pathToFileURL(path.join(here, "webgpu-runtime.js")).href);

function fail(message) {
  console.error(`product-boundary-check failed: ${message}`);
  process.exit(1);
}

function require(condition, message) {
  if (!condition) {
    fail(message);
  }
}

async function expectReject(label, expectedKind, run) {
  try {
    await run();
    fail(`${label}: expected FaberKernelContractError kind=${expectedKind}`);
  } catch (error) {
    require(
      error instanceof FaberKernelContractError,
      `${label}: expected FaberKernelContractError, got ${error?.name ?? typeof error}`,
    );
    require(
      error.kind === expectedKind,
      `${label}: expected kind ${expectedKind}, got ${error.kind}`,
    );
  }
}

async function main() {
  const wgsl = await readFile(path.join(generatedDir, "kernel.wgsl"), "utf8");
  const reflection = JSON.parse(await readFile(path.join(generatedDir, "reflection.json"), "utf8"));

  // Happy path for static reflection consumer (not GPU execution).
  const kernel = loadFaberKernel({ wgsl, reflection });
  require(kernel.entryName === "add_one", "kernel entry must be add_one");
  require(kernel.dispatchWorkgroups.x === 1, "dispatch x must be 1");
  require(kernel.inputBindings.length === 1, "expected one input binding");
  require(kernel.outputBindings.length === 1, "expected one output binding");

  await expectReject("artifact-fetch failure", "artifact-fetch", () =>
    fetchFaberKernelArtifacts({
      fetchImpl: async () => ({ ok: false, status: 404 }),
    }),
  );

  await expectReject("unsupported reflection schema", "reflection", async () => {
    loadFaberKernel({
      wgsl,
      reflection: { ...reflection, schema_version: 999 },
    });
  });

  await expectReject("missing webgpu_adapter", "reflection", async () => {
    const bad = structuredClone(reflection);
    delete bad.kernels[0].launch.webgpu_adapter;
    loadFaberKernel({ wgsl, reflection: bad });
  });

  await expectReject("unavailable WebGPU", "webgpu", () =>
    acquireWebGpuDevice({ navigator: {} }),
  );

  await expectReject("no WebGPU adapter", "webgpu", () =>
    acquireWebGpuDevice({
      navigator: {
        gpu: {
          requestAdapter: async () => null,
        },
      },
    }),
  );

  console.log("product-boundary-check passed");
  console.log("kinds covered: artifact-fetch, reflection, webgpu");
  console.log(
    "manual browser still required for: window.faberWebGpuProof.ok === true && value === 42",
  );
}

main().catch((error) => {
  fail(error?.stack ?? String(error));
});
