#!/usr/bin/env node
/**
 * Engine admission gate proof (DS-S2 Phase 2, items A).
 *
 * Covers the two fail-closed admission seams:
 *
 * 1. artifact-admission: a stale/missing artifact (reflection missing
 *    `schema_version` / `target`, or the pre-radix placeholder format) is
 *    rejected with a typed FaberKernelContractError — no draw, no fallback.
 *    The live placeholder `public/generated/triga-lit-reflection.json` is
 *    asserted to reject: that is the documented P1.3-gated state.
 * 2. capability-admission: an unsupported capability (MSAA 8, non-admitted
 *    color format, >1 directional light, PBR material, device-limit breach)
 *    is a typed CapabilityAdmissionError naming layer/pass/capability.
 *
 * Also greps the hosts tree: the retired hand-rolled admission helper (the
 * old `buildDescriptor…Reflection` function, assembled in this file to keep
 * the grep honest) must be ABSENT (gate 5).
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const srcRoot = here;

const { FaberKernelContractError, loadFaberGraphicsPipeline } = await import(
  pathToFileURL(path.join(here, "contract", "artifact-admission.js")).href
);
const {
  CapabilityAdmissionError,
  admitCapabilities,
  S2_SLICE_CAPABILITIES,
} = await import(
  pathToFileURL(path.join(here, "contract", "capability-admission.js")).href
);

function fail(message) {
  console.error(`engine-admission-check failed: ${message}`);
  process.exit(1);
}

function require(condition, message) {
  if (!condition) fail(message);
}

async function expectReject(label, expectedCtor, run) {
  try {
    await run();
    fail(`${label}: expected ${expectedCtor.name} rejection`);
  } catch (error) {
    require(
      error instanceof expectedCtor,
      `${label}: expected ${expectedCtor.name}, got ${error?.name ?? typeof error}: ${error?.message}`,
    );
  }
}

function drawManifest(indexCount = 3) {
  return Object.freeze({
    index_format: "uint32",
    instance_count: 1,
    base_vertex: 0,
    first_index: 0,
    index_count: indexCount,
  });
}

async function main() {
  // ── 1. Stale placeholder reflection (the live generated artifact) ─────
  {
    const reflectionPath = path.join(srcRoot, "..", "generated", "triga-lit-reflection.json");
    if (fs.existsSync(reflectionPath)) {
      const reflection = JSON.parse(fs.readFileSync(reflectionPath, "utf8"));
      require(
        reflection.schema_version === undefined || reflection.target === undefined,
        "placeholder fixture must lack schema_version/target (pre-radix old format)",
      );
      await expectReject("loadFaberGraphicsPipeline rejects stale placeholder reflection", FaberKernelContractError, () =>
        loadFaberGraphicsPipeline({ wgsl: "@vertex fn vs(){} @fragment fn fs(){}", reflection, drawManifest: drawManifest() }),
      );
      console.log("T1 PASS: live placeholder triga-lit-reflection.json rejected by admission (typed, no draw)");
    } else {
      console.log("T1 NOTE: generated/triga-lit-reflection.json not found — skipping live-fixture assertion");
    }
  }

  // ── 2. Inline stale fixtures: missing schema_version / target ─────────
  {
    const staleMissingVersion = {
      target: "wgsl-text",
      kernels: [
        { shader_stage: "vertex", launch: {} },
        { shader_stage: "fragment", launch: {} },
      ],
      pipeline: {},
    };
    await expectReject("reflection missing schema_version", FaberKernelContractError, () =>
      loadFaberGraphicsPipeline({ wgsl: "", reflection: staleMissingVersion, drawManifest: drawManifest() }),
    );

    const staleMissingTarget = {
      schema_version: 1,
      kernels: [],
      pipeline: {},
    };
    await expectReject("reflection missing target", FaberKernelContractError, () =>
      loadFaberGraphicsPipeline({ wgsl: "", reflection: staleMissingTarget, drawManifest: drawManifest() }),
    );

    await expectReject("missing reflection entirely", FaberKernelContractError, () =>
      loadFaberGraphicsPipeline({ wgsl: "", reflection: null, drawManifest: drawManifest() }),
    );
    console.log("T2 PASS: stale/missing artifact rejected (missing schema_version/target/reflection)");
  }

  // ── 3. Capability admission — S2 slice defaults admitted ──────────────
  {
    const admitted = admitCapabilities({ requested: {} });
    require(admitted.sampleCount === S2_SLICE_CAPABILITIES.sampleCount, "default MSAA is the S2 slice fact");
    require(admitted.depthFormat === "depth24plus", "default depth format is depth24plus");
    require(admitted.colorFormat === "bgra8unorm", "default color format is bgra8unorm");
    require(admitted.lightCount === 1, "default is one directional light");
    console.log("T3 PASS: S2 slice capability defaults admitted");
  }

  // ── 4. Unsupported capability → typed rejection BEFORE draw ───────────
  {
    await expectReject("MSAA 8 (deterministic failure 1)", CapabilityAdmissionError, () =>
      admitCapabilities({ requested: { sampleCount: 8 } }),
    );
    try {
      admitCapabilities({ requested: { sampleCount: 8 } });
      fail("MSAA 8 must reject");
    } catch (error) {
      require(error.layer === "target", "MSAA rejection names layer target");
      require(error.capability === "msaa.sampleCount", "MSAA rejection names capability");
      require(error.pass === "opaque-standard", "MSAA rejection names pass");
    }

    await expectReject("non-admitted color format", CapabilityAdmissionError, () =>
      admitCapabilities({ requested: { colorFormat: "rgba16float" } }),
    );
    await expectReject("non-admitted depth format", CapabilityAdmissionError, () =>
      admitCapabilities({ requested: { depthFormat: "depth32float" } }),
    );
    await expectReject("two directional lights", CapabilityAdmissionError, () =>
      admitCapabilities({ requested: { lightCount: 2 } }),
    );
    await expectReject("PBR material request", CapabilityAdmissionError, () =>
      admitCapabilities({ requested: { standardMaterial: "pbr-metallic-roughness" } }),
    );

    // Device-limit cross-check (device-limit harness pattern).
    const fakeDevice = { limits: { maxBufferSize: 64 } };
    await expectReject("transform buffer exceeds maxBufferSize", CapabilityAdmissionError, () =>
      admitCapabilities({ device: fakeDevice, requested: {} }),
    );
    const okDevice = { limits: { maxBufferSize: 4096 } };
    admitCapabilities({ device: okDevice, requested: {} }); // no throw
    console.log("T4 PASS: unsupported capabilities rejected before draw with layer/pass/capability");
  }

  // ── 5. retired hand-rolled admission helper absent from the hosts tree ──
  {
    // Needle is assembled from parts so this harness's own source does not
    // contain the retired identifier (the tree-wide grep must stay clean).
    const retiredHelper = `buildDescriptor${"From"}Reflection`;
    const offenders = [];
    function walk(dir) {
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          walk(full);
        } else if (entry.name.endsWith(".js")) {
          const text = fs.readFileSync(full, "utf8");
          if (text.includes(retiredHelper)) {
            offenders.push(full);
          }
        }
      }
    }
    walk(srcRoot);
    require(
      offenders.length === 0,
      `the retired hand-rolled admission helper must be absent from the tree; found in: ${offenders.join(", ")}`,
    );
    console.log("T5 PASS: retired hand-rolled admission helper absent from hosts tree");
  }

  console.log("");
  console.log("engine-admission-check passed");
  console.log("covered: stale/missing artifact rejection, capability defaults, unsupported-capability");
  console.log("         typed rejection (layer/pass/capability), device-limit cross-check, retired");
  console.log("         hand-rolled admission grep");
}

main().catch((error) => {
  fail(error?.stack ?? String(error));
});
