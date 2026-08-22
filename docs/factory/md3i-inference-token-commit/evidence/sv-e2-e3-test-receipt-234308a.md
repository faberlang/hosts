# Receipt — hosts SV tests at frozen tip 234308a

- **Goal**: close auditor 65bdef24 residual P2 evidence gap (warm-boot audit 2b43f700): SV-E2 (a297a7f) and SV-E3 (234308a) landed with test sources but no executed receipt.
- **Command**:

  ```text
  cargo test -p faber-host-macos-arm64 --test speculative_verification_session_test --test inference_invocation_program_test
  ```

  Note: the task named `-p macos-arm64`, which matches no package ID; the crate's package name is `faber-host-macos-arm64`.
- **Frozen revision**: hosts 234308ac6125636debcbcf08ddc4e93b2f461481 (clean checkout, no worktree needed)
- **cwd**: `/Users/ianzepp/work/faberlang/hosts`
- **Date**: 2026-08-22
- **Exit status**: 0
- **Output (summary)**:

  ```text
  Running tests/inference_invocation_program_test.rs
  test result: ok. 14 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

  Running tests/speculative_verification_session_test.rs
  test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
  ```

- **Result**: green — 30 tests passed, 0 failed across both named integration-test targets at the frozen tip.
