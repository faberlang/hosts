/**
 * debug-overlay.js — .triga-facts attribute publishing + DOM polling for the
 * Faber controller contract. Phase 1 move of the facts layer from corpus
 * host-init.js (DS-S2 P1.2). The DOM facts contract is unchanged:
 * `data-scene-geometry`, `data-transform-payload`, `.triga-canvas`,
 * `.triga-facts` keep working for the demos' Faber controllers.
 *
 * Phase 2 (S2 slice): the transform-payload *parser* moves to
 * engine/scene-extractor.js (the extractor owns render-fact parsing — no host
 * guessing). This module keeps the DOM bridge (attribute read/publish) and
 * re-exports the parser for callers that read the payload text directly.
 */

import { parseTransformPayload } from "../engine/scene-extractor.js";

export const FACTS_SELECTOR = ".triga-facts";
export const GEOMETRY_ATTR = "data-scene-geometry";
export const TRANSFORM_ATTR = "data-transform-payload";

const GEOMETRY_WAIT_MS = 8000;
const GEOMETRY_POLL_MS = 50;

function factsEl() {
  return document.querySelector(FACTS_SELECTOR);
}

/** Publish one data-* fact to the .triga-facts element. */
export function setFact(name, value) {
  const el = factsEl();
  if (el) el.setAttribute(name, value);
}

/**
 * Wait until the Faber controller publishes the one-shot scene geometry blob.
 * @param {{ timeoutMs?: number, pollMs?: number }} [options]
 * @returns {Promise<string>}
 */
export function waitForSceneGeometry({ timeoutMs = GEOMETRY_WAIT_MS, pollMs = GEOMETRY_POLL_MS } = {}) {
  return new Promise((resolve, reject) => {
    const start = performance.now();
    function tick() {
      const el = factsEl();
      const blob = el?.getAttribute(GEOMETRY_ATTR);
      if (blob && blob.length > 0) {
        resolve(blob);
        return;
      }
      if (performance.now() - start > timeoutMs) {
        reject(new Error(`debug-overlay: timed out waiting for ${GEOMETRY_ATTR}`));
        return;
      }
      setTimeout(tick, pollMs);
    }
    tick();
  });
}

// Re-export the transform-payload parser (owned by engine/scene-extractor).
export { parseTransformPayload };

/**
 * Read + parse the current `data-transform-payload` from the facts element.
 * @returns {Float32Array|null}
 */
export function readTransformPayload() {
  const el = factsEl();
  return parseTransformPayload(el?.getAttribute(TRANSFORM_ATTR) ?? null);
}

/**
 * Read the RAW `data-transform-payload` text from the facts element (the
 * scene-extractor validates the 32-float shape from the raw text).
 * @returns {string|null}
 */
export function readTransformPayloadText() {
  return factsEl()?.getAttribute(TRANSFORM_ATTR) ?? null;
}
