# Swift host package layout

Skeleton directory for Swift native host packages (faber-target-runtime
S3-U1). Layout authority: `../layout.md`. The generated-Swift runtime package
lives in `faber/runtime/swift/` (faber repo, Stage 6); host-side Swift
packages that implement or extend that surface live here.

Package manifests are native (`Package.swift`) and are NOT Cargo workspace
members. One directory per host package.
