/**
 * capability-admission.js — fail-closed capability admission for the engine
 * facade (DS-S2 Phase 2, item A).
 *
 * The engine admits the device against a *requested capability set* BEFORE
 * any draw: an unsupported capability is a typed `CapabilityAdmissionError`
 * naming layer / pass / capability (the engine-owned structured-diagnostics
 * vocabulary; T-F §2 deterministic failure 1).
 *
 * The Triga-declared required/optional capability set does not exist yet
 * (reflection boundary gap G9 — Triga declares, Radix reflects, host admits).
 * Until that lands, admission runs against the frozen S2 slice facts:
 *
 *   - one engine-owned opaque standard material (lit shader; NOT PBR — S3);
 *   - one directional light;
 *   - MSAA 4× (WebGPU render-pass sample counts are 1 or 4);
 *   - depth24plus;
 *   - bgra8unorm canvas color format.
 *
 * These are the same facts the backend hard-codes today
 * (backend/webgpu-runtime.js: EXPECTED_CANVAS_FORMAT, DEPTH_FORMAT,
 * GRAPHICS_SAMPLE_COUNT). This module makes them an explicit, typed,
 * pre-draw admission seam instead of draw-time validation errors.
 */

/** @typedef {"material" | "lighting" | "target" | "backend" | "device"} CapabilityLayer */

/**
 * Typed rejection for an unsupported requested capability. Structured
 * diagnostics: layer / pass / capability (plus optional artifact) — the
 * engine-owned production-outcomes taxonomy (T-F §2, capstone item 10).
 */
export class CapabilityAdmissionError extends Error {
  /**
   * @param {object} fields
   * @param {CapabilityLayer} fields.layer
   * @param {string} fields.pass
   * @param {string} fields.capability
   * @param {string} fields.message
   * @param {string} [fields.artifact]
   */
  constructor({ layer, pass, capability, message, artifact = null }) {
    super(
      `capability-admission ${layer}/${pass}/${capability}: ${message}` +
        (artifact ? ` (artifact: ${artifact})` : ""),
    );
    this.name = "CapabilityAdmissionError";
    this.layer = layer;
    this.pass = pass;
    this.capability = capability;
    this.artifact = artifact;
  }
}

/** The S2 slice's single engine-owned opaque standard material family. */
export const STANDARD_MATERIAL_FAMILY = "standard-opaque-lit";

/** WebGPU render-pass multisample counts that the S2 slice admits. */
export const S2_SUPPORTED_SAMPLE_COUNTS = Object.freeze([1, 4]);

/** Canvas / color-target formats the S2 slice admits. */
export const S2_ADMITTED_COLOR_FORMATS = Object.freeze(["bgra8unorm"]);

/** Depth formats the S2 slice admits. */
export const S2_ADMITTED_DEPTH_FORMATS = Object.freeze(["depth24plus"]);

/** The S2 slice capability set (the admission default). */
export const S2_SLICE_CAPABILITIES = Object.freeze({
  standardMaterial: STANDARD_MATERIAL_FAMILY,
  lightCount: 1, // one directional light
  sampleCount: 4, // MSAA 4×
  depthFormat: "depth24plus",
  colorFormat: "bgra8unorm",
  // Per-frame transform storage (32 f32 = 128 B) plus the lighting uniform
  // (12 f32 = 48 B). Used for the device-limit cross-check only when the
  // caller provides a buffer size.
  transformBufferBytes: 128,
});

/**
 * Admit the device against the requested capability set.
 *
 * `requested` may override any subset of the S2 slice facts; the defaults are
 * the frozen slice facts above. Any unsupported value throws a typed
 * `CapabilityAdmissionError` BEFORE any draw — never a silent fallback.
 *
 * @param {object} [options]
 * @param {object} [options.device] - optional GPUDevice (for device-limit checks)
 * @param {object} [options.adapter] - optional GPUAdapter (reserved for future
 *   adapter-limit checks; the S2 slice facts are guaranteed-level)
 * @param {object} [options.requested] - requested capability overrides
 * @param {object} [options.artifact] - artifact label for diagnostics
 * @returns {object} frozen admitted capability set
 */
export function admitCapabilities({ device, adapter, requested = {}, artifact = null } = {}) {
  void adapter; // S2 slice facts are guaranteed-level; adapter checks land with G9.
  const req = { ...S2_SLICE_CAPABILITIES, ...requested };

  const checks = [
    {
      layer: "target",
      pass: "opaque-standard",
      capability: "msaa.sampleCount",
      ok: S2_SUPPORTED_SAMPLE_COUNTS.includes(req.sampleCount),
      message:
        `MSAA sample count ${req.sampleCount} is not supported by the S2 slice ` +
        `(supported: ${S2_SUPPORTED_SAMPLE_COUNTS.join(", ")})`,
    },
    {
      layer: "target",
      pass: "opaque-standard",
      capability: "color-format",
      ok: S2_ADMITTED_COLOR_FORMATS.includes(req.colorFormat),
      message:
        `color format ${JSON.stringify(req.colorFormat)} is not admitted ` +
        `(admitted: ${S2_ADMITTED_COLOR_FORMATS.join(", ")})`,
    },
    {
      layer: "target",
      pass: "opaque-standard",
      capability: "depth-format",
      ok: S2_ADMITTED_DEPTH_FORMATS.includes(req.depthFormat),
      message:
        `depth format ${JSON.stringify(req.depthFormat)} is not admitted ` +
        `(admitted: ${S2_ADMITTED_DEPTH_FORMATS.join(", ")})`,
    },
    {
      layer: "lighting",
      pass: "opaque-standard",
      capability: "lights.directional",
      ok: req.lightCount <= 1,
      message:
        `directional light count ${req.lightCount} exceeds the S2 slice limit ` +
        "(one directional light; multi-light is deferred beyond S2)",
    },
    {
      layer: "material",
      pass: "opaque-standard",
      capability: "material.standard",
      ok: req.standardMaterial === STANDARD_MATERIAL_FAMILY,
      message:
        `material family ${JSON.stringify(req.standardMaterial)} is not the S2 ` +
        "standard material (single engine-owned opaque lit material; PBR is deferred to S3)",
    },
  ];

  for (const check of checks) {
    if (!check.ok) {
      throw new CapabilityAdmissionError({
        layer: check.layer,
        pass: check.pass,
        capability: check.capability,
        message: check.message,
        artifact,
      });
    }
  }

  // Device-limit cross-check (device-limit harness pattern): a requested
  // transform/storage buffer size must fit maxBufferSize when the limit is
  // exposed. Passes silently when device.limits is absent (fake-device
  // compatibility, same as validateBufferSize in the backend).
  const maxSize = device?.limits?.maxBufferSize;
  if (maxSize !== undefined && req.transformBufferBytes > maxSize) {
    throw new CapabilityAdmissionError({
      layer: "backend",
      pass: "opaque-standard",
      capability: "device-limit.maxBufferSize",
      message:
        `transform storage ${req.transformBufferBytes}B exceeds device limit ` +
        `maxBufferSize ${maxSize}`,
      artifact,
    });
  }

  return Object.freeze({
    standardMaterial: req.standardMaterial,
    lightCount: req.lightCount,
    sampleCount: req.sampleCount,
    depthFormat: req.depthFormat,
    colorFormat: req.colorFormat,
  });
}
