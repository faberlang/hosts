# GEA3-U6 physical numerical rerun #3 receipt

This is new evidence for the linkage-repaired route.  The prior GEA3
physical and numerical receipts are unchanged.

## Identity

| fact | value |
| --- | --- |
| machine | `burgus.local` |
| device | Apple M5 Max, Metal, ordinal 0, registry `4294968623` |
| Gradus | `2e9a2c5de1e3e27b443942920e5eb416f673d2a6` |
| Radix | `97af2854c35900c53ef7ec60a54e39564f5a1b25` (`fix(gea3): enforce producer-consumer buffer linkage`) |
| Hosts | `b7c1db308f3aaf52043f48b173ac5f91e755fa1c` |
| model | SmolLM2-360M-Instruct derived F32 GGUF |
| model SHA-256 | `4d10b02ea1b189cb9637b39ba1543c61f69a8766099076880888f4443754e128` |
| tensor residency | 290 weights, 1,447,284,480 bytes; 64 KV arenas, 6,225,920 bytes |

The fresh Radix bundle was regenerated from the linkage repair into a
temporary artifact directory.  Its launch-4 and launch-5 producer/consumer
identity is buffer `13` on both sides.  The bundle export test passed (`1
passed`).

## Staged diagnostic pre-check

The real-Metal staged diagnostic passed its test gate and advanced through
launch 5 with non-zero data.  It stopped at the first numerical bad entry at
launch 6, rather than incorrectly treating launch 5 as a linkage failure.

| launch | entry | input | output | result |
| ---: | --- | --- | --- | --- |
| 1 | `embedding_gather` | buffer 2: 1/49,152 non-zero | buffer 1: 960/960 non-zero | pass |
| 2 | `decode_rmsnorm` | buffer 1: non-zero | buffer 7: 960/960 non-zero | pass |
| 3 | `decode_gemv_qo` | buffer 7: non-zero | buffer 10: 960/960 non-zero | pass |
| 4 | `decode_gemv_kv` | buffer 10: non-zero | buffer 13: 320/320 non-zero | pass |
| 5 | `decode_gemv_kv` | buffer 13: 320/320 non-zero | buffer 16: 320/320 non-zero | pass |
| 6 | `decode_rope_q` | buffer 16: 320/320 non-zero | buffer 19: exact zero, 0/960 non-zero | **first bad** |

The staged receipt is `gea3-staged-composition-diagnostic-v1`; its SHA-256
is `33e566a0bda81c8fe418d5dec3c60f36aa495a1e74b9a1b048ffd97e30bb00f7`.
The launch-6 input and expected upstream buffer agree (`buffer 16`), so this
pre-check does not report a producer-consumer identity mismatch.

## Full physical route

The full physical receipt test passed structurally: prefill plus eight decode
steps, 2,051 launches per program, 290 weight uploads, 0 CPU substitutes, 0
CPU bridges, logits-only readback, and explicit synchronization.  Structural
status is not promoted to numerical success.

The new raw physical receipt SHA-256 is
`27fb039d88f58c60a49c483ab15c7645ba6f86fb9f997c547e9f831087b07a18`.

| leg | result | evidence |
| --- | --- | --- |
| physical prefill | exact zero artifact | 36 x 49,152 F32 values; SHA-256 `468a4a459772da4e498bc9635d3a9c9b12584490e1edcc1d914ee7df6448d8b2` |
| physical decode | exact zero artifact | all 8 rows; each SHA-256 `3381de4ca9f3a477f25989dfc8b744e7916046b7aa369f61a9a2f7dc0963ec9e` |
| finite/readback shape | pass | prefill 1,769,472 elements; decode 49,152 elements per row |
| numerical route | **FAIL** | zero artifacts do not satisfy the frozen element-wise policy |

The exact-zero claim is a digest classification using the landed
`is_zero_logits_artifact` helper.  The raw logits are not embedded in the
physical receipt, so the record does not pretend that hashes alone are raw
row proof.

## Oracle and comparison

The independent scalar F32 oracle ran against the frozen GGUF in 98.68 s.
The pinned `llama-cli-exact` identity gate passed, and the oracle matched its
frozen greedy sequence 8/8:

```text
[504, 31469, 6740, 335, 2591, 314, 5509, 38921]
```

The landed element-wise comparator was run against the digest-resolved exact
zero rows.  The frozen policy was unchanged:

```text
abs(error) <= 5e-4 OR
(rel(error) <= 2e-5 AND ulp(error) <= 1024)
```

| comparison | result | exact measured result |
| --- | --- | --- |
| prefill logits | **FAIL** | 0/36 rows pass; max abs `33.63347` at row 33/index 9531; max ULP `1,107,724,460` |
| decode logits comparable to oracle | **FAIL** | 0/7 rows pass; max abs `26.447237` at row 2/index 335; max ULP `1,104,385,009` |
| eighth physical decode row | UNMEASURED | the frozen oracle exposes 7 decode rows after the prefill argmax; no row was invented |
| physical matched tokens | **FAIL** | observed `[0,0,0,0,0,0,0,0,0]`; matched `0/8`; first divergence position 0, expected `504`, observed `0`; expected length 8, observed length 9 |
| oracle vs pinned llama-cli | PASS | 8/8 exact tokens; no divergence |

The maximum-error and digest rows were recomputed with the same landed
`compare_logits_rows` and `is_zero_logits_artifact` machinery.  The helper
suite passed 8/8.  The physical failure is therefore a new measured
zero-logits result after the linkage repair, not the old buffer-13-to-17
failure.

## Throughput

| metric | rerun #3 | frozen llama-bench baseline | delta |
| --- | ---: | ---: | ---: |
| prefill | 330.581548 t/s (108,899 us) | 203.426623 t/s | +127.154925 t/s (+62.506531%) |
| decode | 25.682018 t/s (311,502 us) | 126.842262 t/s | -101.160244 t/s (-79.752791%) |

The new route's eight decode GPU-body measurements were
`[26244, 13045, 12927, 12926, 12997, 12982, 13042, 13034]` us.  The first
step is materially slower than the remaining steps; these are reported as
measured and are not normalized away.

**Verdict: numerical closeout remains FAIL.** The linkage seam is physically
closed through launch 5, but `decode_rope_q` launch 6 is the first observed
zero-producing entry in the staged route, and the full logits route remains
an exact-zero artifact.
