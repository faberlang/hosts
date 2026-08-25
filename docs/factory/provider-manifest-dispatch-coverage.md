# Provider Manifest / Dispatch Coverage Packet

**Status:** coverage packet recorded; solum read-route manifest mismatch fixed
**Date:** 2026-07-14
**Scope:** `aleator`, `consolum`, `processus`, `solum`, and `tempus` provider
crates in this repository. This packet does not claim public runnable support,
contract export, install support, or cross-host parity.

## Rule

Each provider manifest is canonical for this workspace and must agree with the
provider dispatch table in both directions. A route is coverage-ready here only
when it is present in `src/manifest.json`, handled by the local `Provider`
implementation, and covered by existing unit/integration evidence or a named
validation command.

This repository can claim local provider coverage evidence. Public product
claims still require downstream package/export/run evidence outside this repo.

## Coverage Matrix

| Provider | Manifest | Manifest routes | Dispatch coverage | Existing test evidence | Local state |
| --- | --- | ---: | --- | --- | --- |
| `aleator` | `crates/aleator/src/manifest.json` | 5 | All manifest routes handled by `Aleator::dispatch`: `fractum`, `sortire`, `octetos`, `uuid`, `semina`. `aleator:octetos` rejects requests above the provider-local 1 MiB byte allocation cap before allocating. | `manifest_registers_all_canonical_routes`; seeded integer/bytes reply-shape test; `octetos` zero/negative/over-limit bounds test. | Covered locally. |
| `consolum` | `crates/consolum/src/manifest.json` | 16 | All manifest routes handled by `Consolum::dispatch`; sync/async pairs share local handlers where appropriate. `consolum:hauri`/`hauriet` reject requests above the provider-local 1 MiB stdin buffer cap before allocating. | Manifest canonical-route/legacy-alias test; terminal predicate shape; opener decoding; stdin read zero/negative/over-limit bounds test; Unix cancellation tests. | Covered locally. |
| `processus` | `crates/processus/src/manifest.json` | 9 | All manifest routes handled by `Processus::dispatch`: shell, capture, detached spawn, env read, cwd, pid, args. `processus:exi` is intentionally unmanifested and rejected until host exit has a protocol-visible terminal response. `processus:scribe` is intentionally unmanifested and rejected until process-wide environment mutation has a safe serialization policy. | Manifest route count and `exi`/`scribe` exclusion; capture shape; shell stdout; cancellation and descendant termination tests. | Covered locally. |
| `solum` | `crates/solum/src/manifest.json` | 45 | All manifest routes handled by `Solum::dispatch`; `solum:lege` is text-only, while `solum:carpe` carries list-of-text line reads and `solum:hauri`/`solum:hauriet` carry byte reads. `solum:partem` and `solum:inveni` reject bounded ranges above the provider-local 1 MiB allocation cap before allocating. Path helpers and filesystem operations are included. | Manifest canonical-route/legacy-alias test; mode/symlink; range/read/find zero/negative/over-limit bounds; kernel-level read-route contract split; delete/touch error propagation. | Covered locally. |
| `tempus` | `crates/tempus/src/manifest.json` | 4 | All manifest routes handled by `Tempus::dispatch`: `nunc`, `monotonicum`, `activum`, `dormiet`. | Manifest legacy-alias test; sleep/cancellation; clock scalar shape; invalid duration test. | Covered locally. |

Read-only comparison on 2026-07-14 found no manifest route missing from the
matching Rust dispatch strings for any of the five providers.

Follow-up fix on 2026-07-14 reconciled the `solum:lege` result contract with
kernel validation: `solum:lege` now rejects non-text materialization targets
instead of returning list or byte frames behind a `textus` manifest result.
List and byte read behavior remains available through the manifest-matched
`solum:carpe` and `solum:hauri` routes.

## Unsupported / Deferred Route Families

These are intentionally not support claims from this repository:

- Legacy aliases are not manifest routes: `consolum:fundet`, `solum:fundet`,
  `solum:leget`, and `tempus:expectet` remain omitted by tests.
- `processus:scribe` is not a manifest route: native provider dispatch can run
  concurrently, and request-scoped host calls must not mutate process-global
  environment state until a process-wide exclusion policy exists.
- Norma source routes that still defer through `mori` are outside this provider
  coverage packet until their manifests, dispatch, and run evidence land.
- No provider manifest exists here for deferred families such as `arca`,
  `caelum`, `crypta`, `http`, `nuncius`, `pressura`, `thesaurus`, `codex`,
  `toml`, `yaml`, or deferred `text` wire/mechanical routes.
- `tempus` does not claim recurring timer cursor support; this workspace only
  manifests one-shot `tempus:dormiet` plus scalar clock routes.
- This packet does not export provider manifests into public contracts and does
  not prove package bootstrap or released compiler integration.

## Validation

Required local validation for this packet:

```bash
cargo fmt --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
git diff --check
```

If a future pass finds a manifest/dispatch mismatch, add a narrow regression
test in the affected crate before updating this packet. If a future pass
promotes public support, add contract-export and public run evidence outside
this repository.

## Public Claim Status

Allowed:

- local provider manifest/dispatch coverage evidence for the five crates above;
- route-family candidate evidence for downstream claim-gate work;
- local Rust unit test coverage descriptions.

Blocked:

- public runnable Norma support;
- public provider support matrix or exported `providers.json` truth;
- install/package examples requiring host dispatch;
- cross-host parity or production backend behavior;
- support claims for non-manifested or deferred Norma route families.
