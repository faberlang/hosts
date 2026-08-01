#!/usr/bin/env node

/**
 * Compiler Chain Bridge — G-P-12 S2 proof script.
 *
 * Reads WGSL + reflection JSON from compiler output, constructs
 * runKernelChain descriptors via buildChainFromReflection, dispatches
 * via runKernelChain in headless Chrome, and reports pass/fail.
 *
 * Usage:
 *   node hosts/webgpu-browser/public/src/compiler-chain-bridge.mjs \
 *     --wgsl ./kernel.wgsl \
 *     --reflection ./reflection.json \
 *     --input '{"0":[1,1,1,2,2,2,3,3,3,4,4,4],"1":[1,2,3,4,5,6],"3":[0.1,0.2,0.1,0.2,0.1,0.2,0.1,0.2]}' \
 *     --output '[{"resourceIndex":4}]' \
 *     --combine '{"4":{"op":"sum","partialCount":2,"fullLength":16}}' \
 *     --expected '[9.1,12.2,18.1,24.2,27.1,36.2,36.1,48.2]'
 *
 * `--combine` is optional reduction combine metadata (W6-A2 D-A2-A): a JSON
 * object mapping output resource index → { op: "sum"|"mean", partialCount,
 * fullLength }. Attached to the bridge's output bindings and applied by
 * placementReadback; absent metadata means raw readback.
 *
 * Dependencies:
 *   - Node.js >= 18
 *   - puppeteer (npm)
 *   - A browser cache with Chrome for Testing >= 120
 *
 * Exit codes:
 *   0 — proof passed
 *   1 — proof failed
 *   2 — environment error
 */

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

// ── Constants ─────────────────────────────────────────────────────────────

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC_DIR = path.resolve(__dirname, "..");
const PORT = 8788;
const BASE_URL = `http://127.0.0.1:${PORT}`;
const TIMEOUT_MS = 60_000;

// ── Argument parsing ──────────────────────────────────────────────────────

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = {};

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case "--wgsl":
        opts.wgslPath = args[++i];
        break;
      case "--reflection":
        opts.reflectionPath = args[++i];
        break;
      case "--input":
        opts.inputJson = args[++i];
        break;
      case "--output":
        opts.outputJson = args[++i];
        break;
      case "--combine":
        opts.combineJson = args[++i];
        break;
      case "--expected":
        opts.expectedJson = args[++i];
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

  if (!opts.reflectionPath) {
    console.error("error: --reflection <path> is required");
    printUsage();
    process.exit(2);
  }

  return opts;
}

function printUsage() {
  console.log(`
Usage: node compiler-chain-bridge.mjs [options]

Options:
  --wgsl <path>             Path to WGSL source file (required for
                            multi-file compiler output)
  --reflection <path>       Path to reflection JSON file (required)
  --input <json>            Input data as JSON object mapping resource
                            index to value array, e.g.
                            '{"0":[1,2,3],"1":[4,5,6]}'
  --output <json>           Output bindings as JSON array, e.g.
                            '[{"resourceIndex":4}]'
  --combine <json>          Optional reduction combine metadata per output
                            resource index (D-A2, W6-A2), e.g.
                            '{"4":{"op":"sum","partialCount":2,"fullLength":16}}'.
                            Applied by placementReadback; absent → raw readback.
  --expected <json>         Expected output values as JSON array, e.g.
                            '[9.1,12.2,18.1,24.2]'
  --help, -h                Show this help
`);
}

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

function startServer() {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      const norm = req.url === "/" ? "/index.html" : req.url;

      // Serve the dynamic proof page at /compiler-proof.html
      if (req.url === "/compiler-proof.html") {
        serveProofPage(res);
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
//
// The proof page is a self-contained HTML document that imports
// webgpu-runtime.js and runs the compiler bridge test inline. The
// input data and reflection are embedded as a script tag.

let _wgslSource = "";
let _reflection = null;
let _inputData = null;
let _outputBindings = null;
let _combineMetadata = null;
let _expectedValues = null;

function setProofData(wgsl, reflection, input, output, combine, expected) {
  _wgslSource = wgsl;
  _reflection = reflection;
  _inputData = input;
  _outputBindings = output;
  _combineMetadata = combine;
  _expectedValues = expected;
}

function serveProofPage(res) {
  const inputDataJson = _inputData
    ? JSON.stringify([..._inputData.entries()])
    : "null";
  const outputBindingsJson = _outputBindings
    ? JSON.stringify(_outputBindings)
    : "null";
  const combineJson = _combineMetadata
    ? JSON.stringify(Object.fromEntries(_combineMetadata.entries()))
    : "null";
  const expectedJson = _expectedValues
    ? JSON.stringify(_expectedValues)
    : "null";
  const reflectionJson = _reflection
    ? JSON.stringify(_reflection)
    : "null";
  const wgslJson = JSON.stringify(_wgslSource);

  const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Compiler Chain Bridge Proof</title>
</head>
<body>
<h1>Compiler Chain Bridge Proof</h1>
<pre id="proof-output">pending</pre>
<script type="module">
import {
  acquireWebGpuDevice,
  buildChainFromReflection,
  runKernelChain,
} from "./src/backend/webgpu-runtime.js";
import { FaberKernelContractError } from "./src/contract/artifact-admission.js";

// ── Embedded compiler output ──────────────────────────────────────────
const WGSL_SOURCE = ${wgslJson};
const REFLECTION = ${reflectionJson};
const INPUT_DATA = ${inputDataJson};
const OUTPUT_BINDINGS = ${outputBindingsJson};
const COMBINE_METADATA = ${combineJson};
const EXPECTED_VALUES = ${expectedJson};

window.faberCompilerProof = { ok: false, status: "starting" };

main().catch((error) => {
  const proof = proofFailure(error);
  window.faberCompilerProof = proof;
  console.log("FABER_COMPILER_PROOF:", JSON.stringify(proof));
});

async function main() {
  const { device } = await acquireWebGpuDevice();

  // Reconstruct input Map from serialized entries
  const inputMap = INPUT_DATA
    ? new Map(INPUT_DATA.map(([k, v]) => [k, new Float32Array(v)]))
    : new Map();

  // Automatically detect output bindings from reflection if not provided
  const outputBindings = OUTPUT_BINDINGS || autoDetectOutputs(REFLECTION);

  const { chain, resources } = buildChainFromReflection(
    device,
    WGSL_SOURCE,
    REFLECTION,
    inputMap,
    outputBindings,
    COMBINE_METADATA,
  );

  const { results } = await runKernelChain(device, resources, chain);

  // Collect all output values
  const allValues = [];
  for (const result of results) {
    for (const v of result.values) {
      allValues.push(v);
    }
  }

  // Compare against expected values if provided
  let ok = true;
  let failures = [];
  if (EXPECTED_VALUES) {
    const EPSILON = 0.001;
    for (let i = 0; i < EXPECTED_VALUES.length; i++) {
      const diff = Math.abs(allValues[i] - EXPECTED_VALUES[i]);
      if (diff > EPSILON) {
        ok = false;
        failures.push(
          \`[$\{i}]: expected \${EXPECTED_VALUES[i]}, got \${allValues[i]} (diff \${diff})\`,
        );
      }
    }
  }

  if (!ok) {
    throw new FaberKernelContractError(
      "readback",
      "compiler bridge result mismatch:\\n  " + failures.join("\\n  "),
      "product",
    );
  }

  window.faberCompilerProof = {
    ok: true,
    status: "ready",
    kind: "ok",
    kernelCount: REFLECTION.kernels.length,
    values: allValues,
    expected: EXPECTED_VALUES,
  };

  console.log("FABER_COMPILER_PROOF:", JSON.stringify(window.faberCompilerProof));
}

function autoDetectOutputs(reflection) {
  const outputs = [];
  const seen = new Set();
  if (reflection && Array.isArray(reflection.kernels)) {
    for (const kernel of reflection.kernels) {
      const adapter = kernel.launch?.webgpu_adapter;
      if (!adapter) continue;
      for (const bgd of adapter.bind_group_descriptors || []) {
        for (const entry of bgd.entries || []) {
          if (entry.role === "output" && !seen.has(entry.resource_index)) {
            seen.add(entry.resource_index);
            outputs.push({ resourceIndex: entry.resource_index });
          }
        }
      }
    }
  }
  return outputs;
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
      const prefix = "FABER_COMPILER_PROOF:";
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
  console.log("Compiler Chain Bridge Proof: starting");

  // 1. Read compiler output files
  let wgslSource = "";
  if (opts.wgslPath) {
    wgslSource = fs.readFileSync(path.resolve(opts.wgslPath), "utf-8");
    console.log("WGSL:", opts.wgslPath, `(${wgslSource.length} chars)`);
  }

  const reflectionPath = path.resolve(opts.reflectionPath);
  const reflectionRaw = fs.readFileSync(reflectionPath, "utf-8");
  let reflection;
  try {
    reflection = JSON.parse(reflectionRaw);
  } catch (e) {
    console.error("error: invalid reflection JSON:", e.message);
    process.exit(2);
  }
  console.log("Reflection:", opts.reflectionPath,
    `(kernels: ${reflection.kernels?.length ?? 0})`);

  // 2. Parse input data
  let inputData = null;
  if (opts.inputJson) {
    try {
      const parsed = JSON.parse(opts.inputJson);
      inputData = new Map();
      for (const [key, values] of Object.entries(parsed)) {
        inputData.set(Number(key), new Float32Array(values));
      }
    } catch (e) {
      console.error("error: invalid input JSON:", e.message);
      process.exit(2);
    }
  }

  // 3. Parse output bindings
  let outputBindings = null;
  if (opts.outputJson) {
    try {
      outputBindings = JSON.parse(opts.outputJson);
    } catch (e) {
      console.error("error: invalid output JSON:", e.message);
      process.exit(2);
    }
  }

  // 4. Parse expected values
  let expectedValues = null;
  if (opts.expectedJson) {
    try {
      expectedValues = JSON.parse(opts.expectedJson);
    } catch (e) {
      console.error("error: invalid expected JSON:", e.message);
      process.exit(2);
    }
  }

  // 4b. Parse reduction combine metadata (W6-A2 D-A2-A): optional map of
  //     resource index → { op, partialCount, fullLength }.
  let combineMetadata = null;
  if (opts.combineJson) {
    try {
      const parsed = JSON.parse(opts.combineJson);
      if (typeof parsed !== "object" || Array.isArray(parsed)) {
        console.error("error: --combine must be a JSON object keyed by resource index");
        process.exit(2);
      }
      combineMetadata = new Map();
      for (const [key, value] of Object.entries(parsed)) {
        combineMetadata.set(Number(key), value);
      }
    } catch (e) {
      console.error("error: invalid combine JSON:", e.message);
      process.exit(2);
    }
  }

  // 5. Inject proof data into the server's state
  setProofData(wgslSource, reflection, inputData, outputBindings, combineMetadata, expectedValues);

  // 6. Verify dependencies
  const chromePath = await puppeteer.executablePath();
  if (!chromePath || !fs.existsSync(String(chromePath))) {
    console.error("Compiler Chain Bridge FAILED: Chrome not found at", chromePath);
    console.error("Run: npx puppeteer browsers install chrome");
    process.exit(2);
  }
  console.log("Chrome:", chromePath);

  // 7. Start HTTP server
  console.log("Starting HTTP server on", BASE_URL, "...");
  const { server } = await startServer();
  console.log("Server ready");

  // 8. Launch headless Chrome with SwiftShader WebGPU
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
    // Suppress noisy console (we only care about FABER_COMPILER_PROOF)
    page.on("console", () => {});

    // 9. Capture proof
    console.log("Navigating to compiler proof page ...");
    const proofPromise = captureProof(page);
    await page.goto(`${BASE_URL}/compiler-proof.html`, {
      waitUntil: "networkidle0",
      timeout: TIMEOUT_MS,
    });

    // 10. Wait for proof result
    const proof = await proofPromise;

    // 11. Assert and report
    if (!proof) {
      console.error("Compiler Chain Bridge FAILED: timeout waiting for proof result");
      exitCode = 1;
    } else if (proof.ok !== true) {
      console.error("Compiler Chain Bridge FAILED:", proof.error || proof.kind);
      console.error("Proof:", JSON.stringify(proof, null, 2));
      exitCode = 1;
    } else {
      console.log("Compiler Chain Bridge PASSED");
      console.log("  ok:", proof.ok);
      console.log("  status:", proof.status);
      console.log("  kernels:", proof.kernelCount);
      console.log("  values:", JSON.stringify(proof.values));
      if (proof.expected) {
        console.log("  expected matches:", JSON.stringify(proof.expected));
      }
      exitCode = 0;
    }
  } catch (err) {
    console.error("Compiler Chain Bridge FAILED with exception:", err.message);
    exitCode = 1;
  } finally {
    await browser.close();
    server.close();
  }

  console.log(`Compiler Chain Bridge: exiting with code ${exitCode}`);
  process.exit(exitCode);
}

main().catch((err) => {
  console.error("Compiler Chain Bridge FATAL:", err.message);
  process.exit(2);
});
