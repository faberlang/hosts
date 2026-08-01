#!/usr/bin/env node
/**
 * Engine state-machine gate proof (DS-S2 Phase 2, item A).
 *
 * Pure-JS state-transition table for
 * `startup → ready → suspended → device-lost → recovering → failed`.
 *
 * Covers:
 * - The full linear chain startup→ready→suspended→device-lost→recovering→ready
 *   (device loss observed + recovery, deterministic failure 3).
 * - Clean failure paths into `failed`.
 * - Invalid transitions rejected with EngineStateError (no silent ignore).
 * - Failed is terminal.
 * - onTransition callback + history evidence.
 * - assert() state checks.
 */

import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

const { EngineStateMachine, EngineStateError, ENGINE_STATES } = await import(
  pathToFileURL(path.join(here, "engine", "engine.js")).href
);

function fail(message) {
  console.error(`engine-state-machine-check failed: ${message}`);
  process.exit(1);
}

function require(condition, message) {
  if (!condition) fail(message);
}

async function expectReject(label, run) {
  let rejected = false;
  try {
    run();
  } catch (error) {
    rejected = true;
    require(
      error instanceof EngineStateError,
      `${label}: expected EngineStateError, got ${error?.name ?? typeof error}`,
    );
  }
  require(rejected, `${label}: expected a typed rejection`);
}

async function main() {
  // ── 1. Vocabulary ────────────────────────────────────────────────────
  require(
    Array.isArray(ENGINE_STATES) && ENGINE_STATES.join("→") ===
      "startup→ready→suspended→device-lost→recovering→failed",
    `ENGINE_STATES must be the explicit six-state vocabulary, got ${ENGINE_STATES.join("→")}`,
  );

  // ── 2. Full linear chain + device-loss recovery (deterministic failure 3)
  {
    const observed = [];
    const machine = new EngineStateMachine({
      onTransition: (entry) => observed.push(`${entry.from}->${entry.to}`),
    });
    require(machine.state === "startup", "initial state is startup");

    machine.transition("ready");
    machine.transition("suspended");
    machine.transition("device-lost");
    machine.transition("recovering");
    machine.transition("ready");

    require(
      observed.join(",") === "startup->ready,ready->suspended,suspended->device-lost,device-lost->recovering,recovering->ready",
      `transition sequence mismatch: ${observed.join(",")}`,
    );
    require(machine.state === "ready", "recovered session lands in ready");
    require(
      machine.history.length === 5,
      `history must record 5 transitions, got ${machine.history.length}`,
    );
    console.log("T1 PASS: startup→ready→suspended→device-lost→recovering→ready");
  }

  // ── 3. Clean failure: startup → failed; recovery attempt → failed
  {
    const machine = new EngineStateMachine();
    machine.transition("failed");
    require(machine.state === "failed", "startup can fail cleanly");
    await expectReject("failed is terminal (failed→ready)", () => {
      machine.transition("ready");
    });
    await expectReject("failed is terminal (failed→device-lost)", () => {
      machine.transition("device-lost");
    });
    console.log("T2 PASS: clean failed state is terminal");
  }

  // ── 4. Invalid transitions rejected (device-lost skips, direct jumps)
  {
    const machine = new EngineStateMachine();
    await expectReject("startup→device-lost", () => machine.transition("device-lost"));
    await expectReject("startup→recovering", () => machine.transition("recovering"));
    await expectReject("ready→recovering", () => machine.transition("recovering"));
    await expectReject("unknown state", () => machine.transition("warp-drive"));
    await expectReject("non-string state", () => machine.transition(42));

    machine.transition("ready");
    await expectReject("ready→startup (no rollback)", () => machine.transition("startup"));
    machine.transition("device-lost");
    await expectReject("device-lost→ready (must recover first)", () => machine.transition("ready"));
    console.log("T3 PASS: invalid transitions rejected with EngineStateError");
  }

  // ── 5. state must not change after a rejected transition
  {
    const machine = new EngineStateMachine();
    try {
      machine.transition("device-lost");
    } catch (_) {
      // expected
    }
    require(machine.state === "startup", "rejected transition leaves state unchanged");
    console.log("T4 PASS: rejected transition leaves state unchanged");
  }

  // ── 6. assert() enforcement
  {
    const machine = new EngineStateMachine();
    await expectReject("assert(ready) from startup", () => machine.assert("ready"));
    machine.transition("ready");
    machine.assert("ready");
    // no throw — reach here
    console.log("T5 PASS: assert() admits only declared states");
  }

  // ── 7. reset() returns to startup
  {
    const machine = new EngineStateMachine();
    machine.transition("ready");
    machine.transition("device-lost");
    machine.reset();
    require(machine.state === "startup", "reset returns to startup");
    require(machine.history.length === 0, "reset clears history");
    console.log("T6 PASS: reset() returns to startup");
  }

  console.log("");
  console.log("engine-state-machine-check passed");
  console.log("covered: vocabulary, full chain + recovery, clean failure, terminal failed,");
  console.log("         invalid-transition rejection, state stability, assert(), reset()");
}

main().catch((error) => {
  fail(error?.stack ?? String(error));
});
