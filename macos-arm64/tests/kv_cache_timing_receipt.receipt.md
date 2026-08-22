# KV-F F7 receipt addendum — host product work

This hosts-side addendum records the 2026-08-22 `(a)` amendment ruling for
protocol-v2 receipts.

## Ruling

Option **(a)** is accepted for the host-measurable term. The v2 receipt carries
`host_product_work_us` from a direct monotonic host-clock measurement around
the selected invocation's cursor/binding projection and transaction commit.
It is a named measurement, not a residual manufactured by subtracting kernel,
transfer, or fused-sync time from invocation wall time.

The fused Metal step-boundary sync remains one observed `submit` clock. The
fused term without an independent clock, `wait`, stays `not_measured` with
that fusion reason because commit and blocking wait share one runtime seam.
The fused-sync separation is deferred to its owning unit. `unattributed` also stays
`not_measured`; this unit does not invent a residual to force the F7 seven-term
close.

## Receipt surface

| F7 term | v2 host receipt | evidence |
| --- | --- | --- |
| `host_product_work` | `host_product_work_us` | direct host-clock measurement |
| `submit` | `submit_us` | F4H2 step-boundary clock |
| `wait` | `not_measured` | fused with Metal submit; independent seam deferred |
| `unattributed` | `not_measured` | no residual subtraction |

The focused `device_execute` unit test proves JSON projection of the new field
and preserves the explicit zero compatibility value for the fused wait wire
field. The zero wire value is not promoted to a measured wait claim.
