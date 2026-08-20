# M4-U1 `CausalAttention` fused-library receipt

The host-only M4-U1 unit adds one plan-path library selection.  It does not
alter the legacy device descriptor decomposition or select a fused kernel;
M4-U2 owns that selection.  The numeric body receives all shape and layout
facts through `CausalAttentionBind`.

## Numeric unit proof

```text
cargo test -p faber-host-macos-arm64 --test gi2_2_library_goldens -- --nocapture
running 6 tests

... 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

The six tests include the GI2-2 `CausalAttention` golden, the q-head
independence perturbation proof, and the four M2 library goldens.  The
attention golden uses the pinned sequence-major KV layout through explicit
bind strides.  Any mismatch fails the test with a nonzero exit.

## Required regression proof

```text
cargo test -p faber-host-macos-arm64 --test composite_host_test --test prepared_session_test -- --nocapture
composite_host_test: 95 passed; 0 failed
prepared_session_test: 9 passed; 0 failed
```

The prepared-session receipt printed during this run was:

```text
prepared-session receipt: prepare=1 reuse=6 reset=1 release=1 reload=0 realloc=0 live-handles=0 (backend metal, sha256:7f6e34b47192abbc8c616ebf41954217eb6ddcfeb1bf6d2c0b42c2e7715a0f09)
```

## Frozen two-rung receipts

Kernel selection remains unchanged until M4-U2.  The per-rung baseline is
recorded as-is from the committed M2-U2 receipt:

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
