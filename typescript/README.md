# TypeScript host package layout

Skeleton directory for TypeScript native host packages (faber-target-runtime
S3-U1). Layout authority: `../layout.md`. The generated-TypeScript runtime
package lives in `faber/runtime/typescript/` (faber repo, Stage 4); host-side
TypeScript packages that implement or extend that surface live here.

Package manifests are native (`package.json`) and are NOT Cargo workspace
members. One directory per host package.
