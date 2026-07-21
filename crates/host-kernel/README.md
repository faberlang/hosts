# host-kernel

Transport-neutral prefix routing and manifest validation for Faber host
providers. The kernel owns request/reply contracts and registration checks; it
does not perform operating-system work or schedule workers.

## Manifest Admission Contract

Provider coverage can rely on host-kernel admission being fail-closed:

- only manifest version `1` is accepted;
- provider identity must be non-empty and globally unique per kernel;
- prefixes must be lowercase ASCII route-family names that start with a letter
  and then contain only lowercase letters, digits, `_`, or `-`; prefixes must
  be unique both within a manifest and across registered providers;
- manifests must export at least one call;
- every route must be `prefix:verb`, the prefix must be one of the manifest's
  prefixes, the verb must be non-empty, and routes must be unique;
- every call must declare one accepted opener contract: `vacuum`,
  `sponte<numerus>`, `textus`, `numerus`, `octeti`, `lista<textus>`,
  `lista<numerus>`, `lista<valor>`, or `valor`;
- every call must declare one accepted result contract: `vacuum`, `textus`,
  `numerus`, `fractus`, `bivalens`, `octeti`, `instans<ns>`, `lista<textus>`,
  `valor`, `bytes`, `lista-valor`, or `bulk-valor`;
- unknown manifest fields are rejected during JSON parsing;
- `dispatch` only admits registered, manifest-exported routes, and
  `supports_route` is the admission predicate for callers that need a cheap
  preflight check;
- `dispatch` validates provider replies against the manifest's declared result
  contract before returning them to the caller.

This crate remains provider-neutral: it validates and routes contracts, but it
does not know concrete provider manifests or perform provider effects.
