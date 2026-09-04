# DELIVERY: HEP-2 exact Host semantic-debt ratchet

**Status**: done — exact ratchet and red-first proof committed in Hosts `be8ef1e`
**Campaign authority:** ../../../../radix/docs/factory/host-execution-purity/CAMPAIGN.md
**Census authority:** ../../../../radix/docs/factory/host-execution-purity/host-semantic-census.md
**Binding decisions:** ../../../../radix/docs/factory/host-execution-purity/architecture-rulings.md

## 1. Interpreted unit

Install a temporary, exact static ratchet before any migration removes the
frozen debt. It must reject new Hosts target-source authoring, operation/model
recognition, semantic CPU/browser fallback, production numerical fake, and
library implementation dependency. It must report existing named debt without
blessing it, and removal of a named entry must lower the reported debt count.

This phase does not migrate semantics, change an artifact wire format, delete
existing debt, run device work, or add an OS host.

## 2. Normalized spec

The checker has one source of temporary authority: a versioned data ledger of
exact path/symbol debt rows derived from the HEP-1 census. It scans only
tracked production sources and Cargo manifests. It does not scan tests,
fixtures, generated code, vendor code, or the ratchet's own fixtures.

The checker must:

1. reject an unallowlisted target-source literal in a production source;
2. reject a new recognized semantic vocabulary, descriptor reconstruction, CPU
   fallback, or semantic fake above the frozen path/symbol ceiling;
3. reject a new dependency from Hosts to Gradus, Norma, Triga, Tela, or another
   library implementation repository;
4. print the exact current debt rows and retired-row identities, so deletion
   reduces the count;
5. fail closed on a malformed ledger, unknown ledger class, missing required
   path, or count above the frozen ceiling; and
6. expose a fixture mode that proves an unallowlisted source literal fails.

The checker may permit only existing exact ledger rows. It may not permit a
directory, wildcard operation family, unbounded comment exemption, or future
symbol.

## 3. Repo-aware baseline

| Item | Frozen fact |
| --- | --- |
| Hosts source baseline | e8fee5b2da972cceb956633cbf13a47349bea46e, clean before HEP-1 |
| Census | 118 production-source modules; HEP-1 commit d0a4d8ad6 |
| Rulings | Gradus owns inference/GGUF semantics; Radix owns DeviceProgram wire authority; Triga owns scene semantics; Hosts owns physical execution only |
| Existing static convention | executable scripts under hosts/scripta; no workspace-wide CI/static-gate dispatcher exists |
| Test runner | an executable scripta proof is the smallest stable hook; no Cargo package needs the ratchet implementation |

The ratchet must not depend on private Radix source. It consumes only local
source plus its checked-in exact ledger.

## 4. Stage graph

| Unit | Inputs | Output | Depends on | Done when |
| --- | --- | --- | --- | --- |
| H2-A ledger | HEP-1 census and rulings | docs/factory/host-execution-purity/debt-ledger.toml | committed HEP-1 | every temporary exception is path/symbol/class/count specific |
| H2-B checker | H2-A | scripta/check-host-execution-purity | H2-A | current tree passes, malformed/new violations fail closed |
| H2-C proof hook | H2-A, H2-B | scripta/test-host-execution-purity and fixture roots | H2-B | red fixture fails with an unallowlisted diagnostic; retired fixture lowers reported count |
| H2-D checkpoint | H2-A through H2-C | status/docs/commit | H2-C | scoped checks pass; review finds no broad exemption or test-only production authority |

## 5. Implementation work

| Unit | Write scope | Out of scope | Thin done-when |
| --- | --- | --- | --- |
| H2-A | docs/factory/host-execution-purity/debt-ledger.toml | source migrations; broad path exemptions | exact, parseable debt data records every checker exception |
| H2-B | scripta/check-host-execution-purity | Cargo dependency; target execution | checker parses ledger, reports debt, and rejects the four no-growth categories |
| H2-C | scripta/test-host-execution-purity and scripta/fixtures/host-execution-purity | device/browser proof | test hook proves red target-source failure and retired-debt count reduction |
| H2-D | HEP docs only if phase succeeds | later HEP artifacts | record HEP-2 status/receipt and next dependency |

**Batching / split decision:** one coherent static-gate phase. The ledger,
checker, and red fixture are inseparable acceptance evidence. No parallel edit
owners are used.

## 6. Checkpoints and gates

### Hand sanity

- The ledger has no directory-wide exception.
- A changed path/symbol can only lower, never raise, a ceiling.
- The checker names the violated category and source location.
- Fixtures are excluded from production enumeration.

### Phase verification

1. scripta/test-host-execution-purity;
2. scripta/check-host-execution-purity;
3. cargo test -p faber-host-macos-arm64 --test hygiene only if the phase adds
   or changes the existing hygiene hook; otherwise not applicable; and
4. static factory-status audit after Radix status updates.

### Checkpoint

HEP-2 completes only when the red fixture fails for the intended reason, the
frozen current debt passes, and a fixture with an absent named debt symbol
reports a lower count. The ratchet is temporary and becomes smaller whenever
HEP-4 through HEP-9 delete authority.

Release decision: not applicable. This is an internal static gate with no
component version or product release.

## 7. Validation

The phase deliberately proves static policy, not device execution. It must not
claim GPU execution, numerical parity, portability completion, or removal of
any census row.

## 8. Companion skill plan

- factory red-green incremental mode for the negative fixture;
- correctness review of matcher/allowlist boundaries;
- clean-break only in future deletion waves, not HEP-2;
- polish over the new checker and proof scripts before checkpoint.

## 9. Open questions

None. The exact ownership and permitted physical boundary are settled by the
binding architecture rulings.
