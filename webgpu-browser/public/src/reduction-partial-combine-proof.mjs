#!/usr/bin/env node

/**
 * W6-A2 U3 — Headless Chrome GPU Reduction Partial-Combine Proof
 *
 * Dispatches a real reduction through the host bridge on the WebGPU device
 * path (headless Chrome, SwiftShader): a two-kernel chain descriptor
 * (Sum + Mean reductions, n=16, ws=8 → exactly 2 partial slots per output)
 * is dispatched via dispatchChainFromDescriptor, read back, and combined via
 * the W6-A2 host combine (D-A2 default A — caller-driven combine metadata).
 *
 * The proof asserts:
 *   (a) the raw readback returns 2 partial slots per output (multi-workgroup
 *       execution actually happened — not a single-workgroup disguise);
 *   (b) the caller-supplied combine yields the value matching the independent
 *       CPU reference (sum 136, mean 8.5 for input 1..16);
 *   (c) the combined assertion would FAIL against the raw partials (combine
 *       is active, not identity).
 *
 * Contract: radix/docs/factory/mir-wgsl/reduction-output-contract.md
 * (partial-slot law, combine ops, fail-closed rule). The mean partial slots
 * are pre-divided by n in-kernel (compiler emission), so the host mean
 * combine is the sum of the slots.
 *
 * Usage:
 *   node hosts/webgpu-browser/public/src/reduction-partial-combine-proof.mjs
 *
 * Dependencies:
 *   - Node.js >= 18
 *   - puppeteer (npm)
 *   - Chrome for Testing >= 120 with WebGPU via --enable-unsafe-swiftshader
 *
 * Exit codes (skip discipline):
 *   0 — proof passed (real device dispatch, combine verified)
 *   1 — proof failed (result mismatch or runtime error)
 *   2 — environment error (missing arguments/dependencies, Chrome/WebGPU
 *        unavailable) — recorded as a condition, NOT a pass
 */

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

// puppeteer is imported dynamically (not statically): a missing package is a
// recorded environment condition (exit 2) per the skip-discipline contract
// in the header, not an uncaught module-evaluation error (exit 1).
let puppeteer = null;
try {
  puppeteer = (await import("puppeteer")).default;
} catch {
  console.error("Reduction Partial-Combine Proof SKIPPED (environment): puppeteer package unavailable");
  console.error("Recorded condition: cannot import 'puppeteer' — run `npm install puppeteer` (or `npx puppeteer browsers install chrome` for a Chrome-only fix).");
  console.error("This is NOT a pass; no combine claim is asserted from this run.");
  process.exit(2);
}

// ── Constants ─────────────────────────────────────────────────────────────

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC_DIR = path.resolve(__dirname, "..");
const PORT = 8790;
const BASE_URL = `http://127.0.0.1:${PORT}`;
const TIMEOUT_MS = 60_000;
const DEFAULT_TOLERANCE = 0.0001;

// ── Fixture: reduction n=16, ws=8 → ceil(16/8)=2 partial slots ───────────

const N = 16;         // tensor length
const WS = 8;         // workgroup lane count
const PARTIAL_COUNT = Math.ceil(N / WS); // 2
const INPUT = Array.from({ length: N }, (_, i) => i + 1); // 1..16

// Independent CPU reference (computed here in the harness, embedded into the
// page; never recomputed by the code path under test).
const SUM_REF = INPUT.reduce((a, b) => a + b, 0); // 136
const MEAN_REF = SUM_REF / N;                     // 8.5

// ── Argument parsing (self-contained proof; no required args) ────────────

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = { tolerance: DEFAULT_TOLERANCE };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case "--tolerance":
        opts.tolerance = Number(args[++i]);
        break;
      case "--help":
      case "-h":
        printUsage();
        process.exit(0);
      default:
        console.error(`unknown option: ${args[i]}`);
        printUsage();
        process.exit(2);
    }
  }

  if (!Number.isFinite(opts.tolerance) || opts.tolerance < 0) {
    console.error("error: --tolerance must be a non-negative finite number");
    process.exit(2);
  }
  return opts;
}

function printUsage() {
  console.log(`
Usage: node reduction-partial-combine-proof.mjs [options]

Dispatches a Sum + Mean reduction (n=${N}, ws=${WS} → ${PARTIAL_COUNT} partial
slots) through the real WebGPU bridge, combines via caller-driven combine
metadata, and asserts the result matches the independent CPU reference
(sum ${SUM_REF}, mean ${MEAN_REF}).

Options:
  --tolerance <float>      f32 comparison tolerance (default ${DEFAULT_TOLERANCE})
  --help, -h               Show this help
`);
}

// ── Chain descriptor (compiler-emitted shape, G-SPINE-10) ────────────────
//
// Kernel 0 (sum_reduction) and kernel 1 (mean_reduction) each use their own
// storage buffers: bindings 0/1 (input/output) and 2/3 (input/output).
// workgroup_size == (2,1,1) — the partial count, i.e. the number of
// workgroups to dispatch (per-kernel thread size 8 lives in the WGSL source).

// Reduction kernels follow the compiler-emitted shape (grid-stride loop,
// workgroup tree reduction, per-workgroup partial write) with one adaptation:
// the shared-memory variable is named `wg_shared`, NOT `shared`. The WGSL
// spec reserves the identifier `shared` (Chrome rejects it at shader-module
// creation); the compiler currently emits `var<workgroup> shared` — a
// device-compilation blocker recorded in the reduction output-buffer contract
// note. The kernel semantics are otherwise identical to the emission.

const SUM_KERNEL_SOURCE = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
var<workgroup> wg_shared: array<f32, ${WS}u>;

@compute @workgroup_size(${WS}, 1, 1)
fn sum_reduction(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    var acc: f32 = 0.0;
    for (var g: u32 = id.x; g < ${N}u; g += ${WS * PARTIAL_COUNT}u) {
        acc += input[g];
    }
    wg_shared[local_id.x] = acc;
    workgroupBarrier();
    if (local_id.x < 4u) { wg_shared[local_id.x] = wg_shared[local_id.x] + wg_shared[local_id.x + 4u]; }
    workgroupBarrier();
    if (local_id.x < 2u) { wg_shared[local_id.x] = wg_shared[local_id.x] + wg_shared[local_id.x + 2u]; }
    workgroupBarrier();
    if (local_id.x < 1u) { wg_shared[local_id.x] = wg_shared[local_id.x] + wg_shared[local_id.x + 1u]; }
    workgroupBarrier();
    if (local_id.x == 0u && workgroup_id.x < ${PARTIAL_COUNT}u) { output[workgroup_id.x] = wg_shared[0]; }
}
`;

// Mean kernel matches the compiler emission shape: the in-kernel division
// `wg_shared[0] = wg_shared[0] / f32(n)` happens before the partial write, so
// each partial slot carries workgroup_sum / n.
const MEAN_KERNEL_SOURCE = `
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
var<workgroup> wg_shared: array<f32, ${WS}u>;

@compute @workgroup_size(${WS}, 1, 1)
fn mean_reduction(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    var acc: f32 = 0.0;
    for (var g: u32 = id.x; g < ${N}u; g += ${WS * PARTIAL_COUNT}u) {
        acc += input[g];
    }
    wg_shared[local_id.x] = acc;
    workgroupBarrier();
    if (local_id.x < 4u) { wg_shared[local_id.x] = wg_shared[local_id.x] + wg_shared[local_id.x + 4u]; }
    workgroupBarrier();
    if (local_id.x < 2u) { wg_shared[local_id.x] = wg_shared[local_id.x] + wg_shared[local_id.x + 2u]; }
    workgroupBarrier();
    if (local_id.x < 1u) { wg_shared[local_id.x] = wg_shared[local_id.x] + wg_shared[local_id.x + 1u]; }
    workgroupBarrier();
    if (local_id.x == 0u) { wg_shared[0] = wg_shared[0] / f32(${N}u); }
    if (local_id.x == 0u && workgroup_id.x < ${PARTIAL_COUNT}u) { output[workgroup_id.x] = wg_shared[0]; }
}
`;

const CHAIN_DESCRIPTOR = {
  chain: [
    {
      entry_point: "sum_reduction",
      source: SUM_KERNEL_SOURCE,
      storage_buffers: [
        { binding: 0, group: 0, size: N * 4 },
        { binding: 1, group: 0, size: PARTIAL_COUNT * 4 },
      ],
      workgroup_size: [PARTIAL_COUNT, 1, 1],
      bind_group_layout: [
        { group: 0, binding: 0, buffer_index: 0 },
        { group: 0, binding: 1, buffer_index: 1 },
      ],
      output_bindings: [1],
    },
    {
      entry_point: "mean_reduction",
      source: MEAN_KERNEL_SOURCE,
      storage_buffers: [
        { binding: 2, group: 0, size: N * 4 },
        { binding: 3, group: 0, size: PARTIAL_COUNT * 4 },
      ],
      workgroup_size: [PARTIAL_COUNT, 1, 1],
      bind_group_layout: [
        { group: 0, binding: 2, buffer_index: 0 },
        { group: 0, binding: 3, buffer_index: 1 },
      ],
      output_bindings: [1],
    },
  ],
  buffer_identities: [],
};

// Combine metadata keyed by storage-buffer @binding (resource index).
const COMBINE_METADATA = {
  1: { op: "sum", partialCount: PARTIAL_COUNT, fullLength: N },
  3: { op: "mean", partialCount: PARTIAL_COUNT, fullLength: N },
};

// ── Server ────────────────────────────────────────────────────────────────

function mimeType(ext) {
  switch (ext) {
    case ".html": return "text/html; charset=utf-8";
    case ".mjs":
    case ".js":   return "application/javascript; charset=utf-8";
    case ".json": return "application/json";
    case ".wgsl": return "text/plain; charset=utf-8";
    default:      return "application/octet-stream";
  }
}

function startServer(tolerance) {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      const norm = req.url === "/" ? "/index.html" : req.url;

      if (req.url === "/reduction-combine-proof.html") {
        serveProofPage(res, tolerance);
        return;
      }

      const filePath = path.join(PUBLIC_DIR, norm);
      if (!filePath.startsWith(PUBLIC_DIR)) {
        res.writeHead(404);
        res.end("Not found");
        return;
      }
      try {
        if (fs.statSync(filePath).isFile()) {
          const ext = path.extname(filePath);
          const data = fs.readFileSync(filePath);
          res.writeHead(200, { "Content-Type": mimeType(ext) });
          res.end(data);
        } else {
          res.writeHead(404);
          res.end("Not found");
        }
      } catch {
        res.writeHead(404);
        res.end("Not found");
      }
    });

    server.listen(PORT, "127.0.0.1", () => {
      resolve({ server, url: BASE_URL });
    });
    server.on("error", reject);
  });
}

// ── Proof page generation ─────────────────────────────────────────────────

function serveProofPage(res, tolerance) {
  const descriptorJson = JSON.stringify(CHAIN_DESCRIPTOR);
  const resourceSpecsJson = JSON.stringify([
    { key: 0, data: INPUT },
    { key: 1, bytes: PARTIAL_COUNT * 4 },
    { key: 2, data: INPUT },
    { key: 3, bytes: PARTIAL_COUNT * 4 },
  ]);
  const combineJson = JSON.stringify(COMBINE_METADATA);
  const expectedJson = JSON.stringify({ sum: SUM_REF, mean: MEAN_REF });
  const toleranceJson = JSON.stringify(tolerance);

  const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reduction Partial-Combine Proof</title>
</head>
<body>
<h1>Reduction Partial-Combine Proof</h1>
<pre id="proof-output">pending</pre>
<script type="module">
import {
  acquireWebGpuDevice,
  dispatchChainFromDescriptor,
} from "./src/webgpu-runtime.js";
import { FaberKernelContractError } from "./src/faber-kernel.js";

// ── Embedded fixture (n=16, ws=8 → 2 partial slots) ──────────────────
const CHAIN_DESCRIPTOR = ${descriptorJson};
const RESOURCE_SPECS = ${resourceSpecsJson};
const COMBINE_METADATA = ${combineJson};
const EXPECTED = ${expectedJson};
const TOLERANCE = ${toleranceJson};

window.faberReductionProof = { ok: false, status: "starting" };

main().catch((error) => {
  const proof = proofFailure(error);
  window.faberReductionProof = proof;
  console.log("FABER_REDUCTION_PROOF:", JSON.stringify(proof));
});

function near(actual, expected, msg) {
  if (Math.abs(actual - expected) > TOLERANCE) {
    throw new FaberKernelContractError(
      "readback",
      \`$\{msg}: expected $\{expected}, got $\{actual}\`,
      "product",
    );
  }
}

async function buildResources(device) {
  // Keying follows the descriptor's storage-buffer @binding namespace
  // (dispatchChainFromDescriptor resolves resources by bufDecl.binding).
  // Output buffers (bytes) are allocated empty; inputs (data) are written.
  const buffers = new Map();
  const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  for (const spec of RESOURCE_SPECS) {
    const size = spec.bytes ?? spec.data.length * 4;
    const buffer = device.createBuffer({ size, usage: STORAGE });
    if (spec.data) {
      device.queue.writeBuffer(buffer, 0, new Float32Array(spec.data));
    }
    buffers.set(spec.key, { buffer });
  }
  return { buffers };
}

async function main() {
  const { device } = await acquireWebGpuDevice();
  const resources = await buildResources(device);

  // ── Dispatch WITH caller-driven combine metadata ───────────────────
  const { results } = await dispatchChainFromDescriptor(
    device,
    resources,
    CHAIN_DESCRIPTOR,
    COMBINE_METADATA,
  );
  const sumCombined = results[0].combined;
  const meanCombined = results[1].combined;
  const sumCombinedValues = results[0].values;
  const meanCombinedValues = results[1].values;

  // ── Dispatch WITHOUT metadata: raw partial slots ────────────────────
  const { results: rawResults } = await dispatchChainFromDescriptor(
    device,
    resources,
    CHAIN_DESCRIPTOR,
  );
  const rawSum = rawResults[0].values;
  const rawMean = rawResults[1].values;

  // ── Assertions ─────────────────────────────────────────────────────
  // (b) combined values match the independent CPU reference.
  near(sumCombined, EXPECTED.sum, "combined sum");
  near(meanCombined, EXPECTED.mean, "combined mean");
  // Combined readback collapses to a single value.
  if (sumCombinedValues.length !== 1) {
    throw new FaberKernelContractError(
      "readback",
      \`combined sum values length $\{sumCombinedValues.length} != 1\`,
      "product",
    );
  }
  if (meanCombinedValues.length !== 1) {
    throw new FaberKernelContractError(
      "readback",
      \`combined mean values length $\{meanCombinedValues.length} != 1\`,
      "product",
    );
  }
  // (a) raw readback returns 2 partial slots — multi-workgroup execution
  //     actually happened (n=16, ws=8 → ceil(16/8)=2 workgroups).
  if (rawSum.length !== 2) {
    throw new FaberKernelContractError(
      "readback",
      \`raw sum partial slots $\{rawSum.length} != 2 (multi-workgroup execution not observed)\`,
      "product",
    );
  }
  if (rawMean.length !== 2) {
    throw new FaberKernelContractError(
      "readback",
      \`raw mean partial slots $\{rawMean.length} != 2 (multi-workgroup execution not observed)\`,
      "product",
    );
  }
  // (c) the combined assertion would FAIL against the raw partials —
  //     proving combine is active, not identity.
  for (const raw of rawSum) {
    if (Math.abs(raw - EXPECTED.sum) <= TOLERANCE) {
      throw new FaberKernelContractError(
        "readback",
        \`raw sum partial $\{raw} already equals the reference — combine did not transform\`,
        "product",
      );
    }
  }
  for (const raw of rawMean) {
    if (Math.abs(raw - EXPECTED.mean) <= TOLERANCE) {
      throw new FaberKernelContractError(
        "readback",
        \`raw mean partial $\{raw} already equals the reference — combine did not transform\`,
        "product",
      );
    }
  }
  // The 2 partial slots must sum to the reference (multi-workgroup partials).
  const rawSumTotal = rawSum[0] + rawSum[1];
  near(rawSumTotal, EXPECTED.sum, "raw partial sum total");

  window.faberReductionProof = {
    ok: true,
    status: "ready",
    kind: "ok",
    n: ${N},
    ws: ${WS},
    partialCount: 2,
    combined: { sum: sumCombined, mean: meanCombined },
    rawPartials: { sum: rawSum, mean: rawMean },
    expected: EXPECTED,
    tolerance: TOLERANCE,
  };

  console.log("FABER_REDUCTION_PROOF:", JSON.stringify(window.faberReductionProof));
}

function proofFailure(error) {
  const kind =
    error instanceof FaberKernelContractError
      ? error.kind
      : typeof error?.kind === "string"
        ? error.kind
        : "product";
  return {
    ok: false,
    status: "error",
    kind,
    path: error?.path ?? null,
    error: error?.message ?? String(error),
  };
}
</script>
</body>
</html>`;

  const buf = Buffer.from(html, "utf-8");
  res.writeHead(200, {
    "Content-Type": "text/html; charset=utf-8",
    "Content-Length": buf.length,
  });
  res.end(buf);
}

// ── Proof capture ────────────────────────────────────────────────────────

async function captureProof(page) {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      resolve(null);
    }, TIMEOUT_MS);

    page.on("console", (msg) => {
      const text = msg.text();
      const prefix = "FABER_REDUCTION_PROOF:";
      if (text.startsWith(prefix)) {
        clearTimeout(timeout);
        try {
          resolve(JSON.parse(text.slice(prefix.length)));
        } catch {
          resolve(null);
        }
      }
    });

    page.on("pageerror", (err) => {
      clearTimeout(timeout);
      reject(err);
    });
  });
}

// ── Main ──────────────────────────────────────────────────────────────────

async function main() {
  const opts = parseArgs();
  console.log("Reduction Partial-Combine Proof: starting");
  console.log(`Fixture: n=${N}, ws=${WS} → ${PARTIAL_COUNT} partial slots`);
  console.log(`References: sum=${SUM_REF}, mean=${MEAN_REF} (input 1..${N})`);

  // 1. Verify dependencies (Chrome with WebGPU via SwiftShader).
  const chromePath = await puppeteer.executablePath();
  if (!chromePath || !fs.existsSync(String(chromePath))) {
    console.error("Reduction Partial-Combine Proof SKIPPED (environment): Chrome not found at", chromePath);
    console.error("Recorded condition: puppeteer Chrome unavailable — run `npx puppeteer browsers install chrome`.");
    console.error("This is NOT a pass; no combine claim is asserted from this run.");
    process.exit(2);
  }
  console.log("Chrome:", chromePath);

  // 2. Start HTTP server.
  console.log("Starting HTTP server on", BASE_URL, "...");
  const { server } = await startServer(opts.tolerance);
  console.log("Server ready");

  // 3. Launch headless Chrome with SwiftShader WebGPU.
  console.log("Launching headless Chrome ...");
  const browser = await puppeteer.launch({
    executablePath: chromePath,
    headless: "new",
    args: [
      "--headless=new",
      "--no-sandbox",
      "--enable-unsafe-swiftshader",
      "--enable-webgpu",
      "--enable-features=Vulkan,UseSkiaRenderer,WebGPU",
      "--window-size=640,480",
    ],
  });

  let exitCode = 2;
  try {
    const page = await browser.newPage();
    page.on("console", () => {});

    // 4. Capture proof.
    console.log("Navigating to reduction proof page ...");
    const proofPromise = captureProof(page);
    await page.goto(`${BASE_URL}/reduction-combine-proof.html`, {
      waitUntil: "networkidle0",
      timeout: TIMEOUT_MS,
    });

    // 5. Wait for proof result.
    const proof = await proofPromise;

    // 6. Assert and report.
    if (!proof) {
      console.error("Reduction Partial-Combine Proof FAILED: timeout waiting for proof result");
      exitCode = 1;
    } else if (proof.ok !== true) {
      if (proof.kind === "webgpu") {
        console.error("Reduction Partial-Combine Proof SKIPPED (environment): WebGPU unavailable in headless Chrome");
        console.error("Recorded condition:", proof.error);
        console.error("This is NOT a pass; no combine claim is asserted from this run.");
        exitCode = 2;
      } else {
        console.error("Reduction Partial-Combine Proof FAILED:", proof.error || proof.kind);
        console.error("Proof:", JSON.stringify(proof, null, 2));
        exitCode = 1;
      }
    } else {
      console.log("Reduction Partial-Combine Proof PASSED");
      console.log("  n:", proof.n, "ws:", proof.ws, "partial slots:", proof.partialCount);
      console.log("  combined sum:", proof.combined.sum, "(expected", proof.expected.sum + ")");
      console.log("  combined mean:", proof.combined.mean, "(expected", proof.expected.mean + ")");
      console.log("  raw partials (sum):", JSON.stringify(proof.rawPartials.sum), "— 2 slots, multi-workgroup execution observed");
      console.log("  raw partials (mean):", JSON.stringify(proof.rawPartials.mean));
      console.log("  combine active: raw partials != combined result");
      exitCode = 0;
    }
  } catch (err) {
    console.error("Reduction Partial-Combine Proof FAILED with exception:", err.message);
    exitCode = 1;
  } finally {
    await browser.close();
    server.close();
  }

  console.log(`Reduction Partial-Combine Proof: exiting with code ${exitCode}`);
  process.exit(exitCode);
}

main().catch((err) => {
  console.error("Reduction Partial-Combine Proof FATAL:", err.message);
  process.exit(2);
});
