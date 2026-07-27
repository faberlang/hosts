# Faber WebGPU Browser Host

This directory is the browser-first WebGPU **product** boundary for
Faber-emitted WGSL and MIR GPU reflection metadata.

## Boundary

Faber owns compilation and reflection. This host owns browser runtime behavior:
adapter and device acquisition, WebGPU resource creation, command encoding,
dispatch, queue submission, result readback, and the minimal three.js surface
used to inspect the proof.

Do not put browser WebGPU runtime behavior into MIR emitters or into
`macos-arm64`. The macOS host remains an architecture reference for host
ownership boundaries, not a browser runtime dependency.

## Browser Product Boundary

| Concern | Authority |
| --- | --- |
| Product entrypoint | `public/index.html` |
| Product manifest | `public/faber-webgpu-product.json` |
| Generated inputs | `public/generated/{kernel.wgsl,reflection.json,graphics.wgsl,graphics-reflection.json,graphics-*.bin,draw.json}` |
| Launch / serve | `./scripta/webgpu-browser-proof serve` (this hosts repo root) |
| Static admission | `./scripta/webgpu-browser-proof check` |
| Focused non-GPU boundary | `node public/src/product-boundary-check.mjs` |
| Focused graphics storage update | `node public/src/graphics-storage-update-check.mjs` |
| Success state | `window.faberWebGpuProof.ok === true` and `.value === 42`; graphics state reports submitted frames through `window.faberWebGpuGraphicsProof` |

The host consumes generated WGSL plus `launch.webgpu_adapter` reflection,
dispatches the focused compute kernel through browser WebGPU APIs, reads back
the result, and optionally reflects it in a minimal three.js WebGPU scene.
Three.js is presentation chrome only; it never supplies binding or launch facts.

Native GPU providers, Triga graphics parity, and a general three.js integration
layer remain out of scope.

## Proof Fixture

`fixtures/add-one.fab` is a focused compute fixture:

```fab
@ nucleum
functio add_one(fractus<f32> x) → fractus<f32> {
    redde x + 1.0
}
```

The fixture is intentionally scalar so the first browser host can prove the
descriptor contract with one input storage buffer and one output storage buffer.

## Artifact Contract

Browser runtime code consumes the checked-in generated artifacts from:

```text
webgpu-browser/public/generated/kernel.wgsl
webgpu-browser/public/generated/reflection.json
webgpu-browser/public/generated/graphics.wgsl
webgpu-browser/public/generated/graphics-reflection.json
webgpu-browser/public/generated/graphics-vertex-positions.bin
webgpu-browser/public/generated/graphics-vertex-colors.bin
webgpu-browser/public/generated/graphics-indices.bin
webgpu-browser/public/generated/graphics-transform.bin
webgpu-browser/public/generated/draw.json
```

Regenerate them from this hosts repo root after fixture, graphics fixture, or compiler reflection changes
(requires sibling `../radix` and `../triga`):

```bash
./scripta/webgpu-browser-proof generate
```

The command uses the sibling radix workspace:

```bash
cargo run --manifest-path ../radix/Cargo.toml -p radix --bin radix -- emit -t wgsl-text
cargo run --manifest-path ../radix/Cargo.toml -p radix --bin radix -- emit --reflection -t wgsl-text
```

`reflection.json` is the source of binding, layout, resource, and dispatch
truth. Host code must not parse WGSL to discover bindings.

## Local Serve Command

After regenerating artifacts, serve the proof directory with:

```bash
./scripta/webgpu-browser-proof serve
```

The default URL is:

```text
http://127.0.0.1:8787/
```

The `serve` command is present in this scaffold phase so later browser runtime
phases can use a stable local URL without adding global package tooling.

### Static and focused non-GPU admission

```bash
./scripta/webgpu-browser-proof check
```

`check` does three things and **does not claim browser GPU execution**:

1. Regenerates compute and graphics WGSL/reflection into a temp dir and compares
   checked-in `public/generated/` text artifacts for freshness.
2. Validates the reflection/static-page/product manifest contract, including
   graphics pipeline and checked-in binary payload shape.
3. Runs `node public/src/product-boundary-check.mjs` for focused product-boundary
   rejects (artifact-fetch, unsupported reflection, unavailable WebGPU).

Additional focused Node checks cover runtime-only contracts without compiler
regeneration or browser GPU execution:

```bash
node public/src/graphics-storage-update-check.mjs
node public/src/chunk-resource-lifecycle-check.mjs
node public/src/compute-resource-lifecycle-check.mjs
node public/src/device-limit-check.mjs
node public/src/gradient-handle-check.mjs
node public/src/placement-execution-v1-check.mjs
node public/src/placement-contract-oracle.mjs
```

## Failure Outcomes

Failures set `window.faberWebGpuProof` to a clear error shape:

```js
{
  ok: false,
  status: "error",
  kind: "artifact-fetch" | "reflection" | "webgpu" | "product",
  path: string | null,
  error: string
}
```

| `kind` | When |
| --- | --- |
| `artifact-fetch` | Missing/failed fetch of `kernel.wgsl` or `reflection.json` |
| `reflection` | Unsupported or incomplete `launch.webgpu_adapter` descriptors |
| `webgpu` | `navigator.gpu` missing, no adapter, or device acquisition fails |
| `product` | Other product-path failures (for example unexpected readback) |

`FaberKernelContractError` carries the same `kind` for console and harness use.

## Browser Host API

The reflection consumer starts at:

```js
import { loadFaberKernel } from "./src/faber-kernel.js";
```

`loadFaberKernel({ wgsl, reflection })` returns a checked kernel descriptor for
the current proof. It consumes `launch.webgpu_adapter` as the binding and layout
source of truth and throws `FaberKernelContractError` for unsupported descriptor
variants.

## Controlled Manual Browser Inspection

Static and Node product-boundary checks are **non-GPU evidence**. Exact add-one
readback (`42.0`) still requires a WebGPU-capable browser:

```bash
./scripta/webgpu-browser-proof serve
```

Open `http://127.0.0.1:8787/` (Chrome/Edge with WebGPU, or equivalent). The page
dispatches `add_one` with input `41.0`, reads back the output buffer, and
reflects the expected `42.0` result (three.js visual is optional chrome).

Console admission:

```js
window.faberWebGpuProof.ok === true
window.faberWebGpuProof.value === 42
window.faberWebGpuProof.kind === "ok"
window.faberWebGpuGraphicsProof.submittedFrameCount > 0
```

On unsupported environments, expect `ok === false` with a specific `kind` from
the table above — not a silent hang and not three.js deciding bindings.
