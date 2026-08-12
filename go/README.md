# Go host package layout

Skeleton directory for Go native host packages (faber-target-runtime S3-U1).
Layout authority: `../layout.md`. The generated-Go runtime module lives in
`faber/runtime/go/` (faber repo, Stage 5); host-side Go packages that
implement or extend that surface live here.

Package manifests are native (`go.mod`) and are NOT Cargo workspace members.
One directory per host package.
