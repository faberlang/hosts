#!/usr/bin/env node
/**
 * Engine shared-instance oracle scaffold (DS-S2 Phase 2, item E — SCAFFOLD
 * ONLY, does NOT assert the two-demo S2 gate).
 *
 * The S2 shared-instance oracle asserts that Demo A and Demo B use the SAME
 * shader-module / pipeline-cache identity for the shared standard material.
 * Demo A (80 Stage 5 Hello Triga) is gated on 80 Stage 5, so this harness
 * scaffolds the static half of the oracle:
 *
 *   - the standard-material pipeline-cache key is ONE deterministic identity
 *     derived from the admitted artifact set (computePipelineCacheKey);
 *   - both corpus demos resolve the engine from the same module set
 *     (run.sh HOST_DIR → hosts/webgpu-browser; page import →
 *     public/src/product/bootstrap.js).
 *
 * The cross-demo runtime equality assertion is recorded as gated, not run.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

const { computePipelineCacheKey } = await import(
  pathToFileURL(path.join(here, "engine", "engine.js")).href
);

const CORPUS_ROOT = path.resolve(here, "..", "..", "..", "..", "triga", "corpus");
const DEMOS = ["webgl-geometry-terrain", "webgl-geometries"];

function fail(message) {
  console.error(`engine-shared-instance-check failed: ${message}`);
  process.exit(1);
}

function require(condition, message) {
  if (!condition) fail(message);
}

const WAVEFRONT_REFLECTION = {
  schema_version: 1,
  target: "wgsl-text",
  pipeline: {
    primitive_topology: "triangle-list",
    color_target_formats: ["bgra8unorm"],
    vertex_count: 36,
    depth_stencil: { depth_compare: "less-equal", depth_write_enabled: true },
  },
};

async function main() {
  // ── 1. Deterministic pipeline-cache identity ──────────────────────────
  {
    const keyA = computePipelineCacheKey({ wgsl: "WGSL-A", reflection: WAVEFRONT_REFLECTION });
    const keyB = computePipelineCacheKey({ wgsl: "WGSL-A", reflection: WAVEFRONT_REFLECTION });
    require(keyA === keyB, "same artifact set → one deterministic identity");
    require(/^standard-material:[0-9a-f]{8}$/.test(keyA), `key format is standard-material:<hash>, got ${keyA}`);

    const keyC = computePipelineCacheKey({ wgsl: "WGSL-B", reflection: WAVEFRONT_REFLECTION });
    require(keyA !== keyC, "different wgsl → different identity");
    console.log(`T1 PASS: standard-material pipeline-cache identity deterministic: ${keyA}`);
  }

  // ── 2. Both demos resolve the engine from the same module set ─────────
  {
    for (const demo of DEMOS) {
      const runSh = path.join(CORPUS_ROOT, demo, "tests", "run.sh");
      const page = path.join(CORPUS_ROOT, demo, "pages", "index.html");
      if (!fs.existsSync(runSh) || !fs.existsSync(page)) {
        console.log(`T2 NOTE: corpus demo ${demo} not present (${runSh}) — module-set assertion skipped`);
        continue;
      }
      const runShText = fs.readFileSync(runSh, "utf8");
      require(
        runShText.includes('HOST_DIR="$(cd "$WORKSPACE/hosts/webgpu-browser" && pwd)"'),
        `${demo}: run.sh must copy the engine from hosts/webgpu-browser`,
      );
      const pageText = fs.readFileSync(page, "utf8");
      require(
        pageText.includes("public/src/product/bootstrap.js"),
        `${demo}: page must import the shared bootstrap facade`,
      );
      require(
        pageText.includes("initEngine"),
        `${demo}: page must consume initEngine from the shared facade`,
      );
      console.log(`T2 PASS: ${demo} resolves the engine from the shared hosts module set`);
    }
  }

  // ── 3. Gated: cross-demo runtime identity (Demo A = 80 Stage 5) ───────
  {
    console.log("T3 GATED: cross-demo runtime identity (both demos report one");
    console.log("          shader-module/pipeline-cache identity) requires Demo A = 80 Stage 5.");
    console.log("          Scaffolded here; the S2 gate is NOT asserted by this run.");
  }

  console.log("");
  console.log("engine-shared-instance-check passed (scaffold)");
  console.log("covered: deterministic pipeline-cache identity, shared module-set resolution,");
  console.log("         gated two-demo runtime identity explicitly not asserted");
}

main().catch((error) => {
  fail(error?.stack ?? String(error));
});
