# M2-U2 GI2-2 numeric library-golden receipt

The copied fixtures under `fixtures/gi2-2-op-goldens/` are byte-for-byte
copies of the read-only Radix `faber-prefill-oracle` tree.  The four M2 rows
that map to the M2-U1 parameterized bodies are compared as f32 values by
`gi2_2_library_goldens.rs`; emitted MSL text is not part of the oracle.  A
fixture row's `max_abs_delta` is honored when present.  Otherwise the GI2-2
f32 band is `1e-6`; a value outside the band fails the test.

## Unit proof

```text
cargo test -p faber-host-macos-arm64 --test gi2_2_library_goldens -- --nocapture
running 4 tests

test rope_golden_matches_parameterized_body ... ok
test rms_norm_golden_matches_parameterized_body ... ok
test residual_golden_matches_parameterized_body ... ok
test swiglu_golden_matches_parameterized_body ... ok

test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

## Required regression proof

```text
cargo test -p faber-host-macos-arm64 --test composite_host_test --test prepared_session_test
composite_host_test: 95 passed; 0 failed
prepared_session_test: 9 passed; 0 failed
```

## Frozen-rung baseline

The following rows are copied verbatim from the committed frozen-rung
`dense-full-model-goldens/manifest.json` receipt.  They record the unchanged
per-rung kernel counts and the prompt-end `30/304` top-1 pins (the first
continuation decode rows remain 198/279).

| model | regime | kernels | StageTiming `(kernel, transfer, host round trip, total)` (µs) | top-1 | max delta | receipt_present | all_finite |
| --- | --- | ---: | --- | ---: | ---: | :---: | :---: |
| SmolLM2-360M | prefill | 5219 | `(85283, 3026, 17325, 105634)` | 30 | `0.022698163986206055` vs GI2-3 | true | true |
| SmolLM2-360M | decode | 5219 | `(83692, 103403, 736336, 923431)` | 198 | `0.0` | true | true |
| Qwen2.5-0.5B | prefill | 3723 | `(72909, 5757, 13320, 91986)` | 304 | `0.0` | true | true |
| Qwen2.5-0.5B | decode | 3723 | `(76437, 160809, 829074, 1066320)` | 279 | `0.0` | true | true |

The frozen-rung run legitimately stops at the existing GI2-3 envelope
residual.  The committed residual is quoted verbatim:

> **GI2-3 envelope question — ROUTED TO OPERATOR.** The SmolLM2 prefill
> comparison records the exact max delta `0.022698163986206055` versus
> `Q2_ENVELOPE=0.0065`. This is the pre-existing oracle provenance versus
> envelope-width question, not a goal blocker: the three new rows pass at
> `0.0`, and the capture is deterministic.
