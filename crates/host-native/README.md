# host-native

The bounded worker and `faber::HostDispatch` adapter for public Faber host
providers. Provider registration is supplied by generated package bootstrap;
this crate does not own Norma implementations.

## Native Adapter Lifecycle Contract

`NativeHost` is the transport-neutral adapter between `faber::HostDispatch` and
`host-kernel` routing. Its contract is intentionally narrow:

- `start` performs a cheap admission check, then enqueues work without waiting
  for provider execution.
- Unknown or non-exported routes are rejected before enqueue using the
  kernel's route admission predicate.
- The worker queue is bounded; saturation returns `host_queue_saturated`
  immediately instead of growing unbounded work.
- Shutdown closes the queue, rejects new starts, cancels active jobs, and gives
  queued jobs a terminal error when workers drain them.
- Provider cancellation is observed before content is sent and while the kernel
  dispatch context is active.
- Provider panics are caught and converted into terminal error frames.
- Cloning keeps the adapter alive; dropping the last public handle begins
  shutdown without joining active worker threads. Explicit `shutdown` joins
  workers.

Non-goals: this crate does not own concrete provider effects, provider support
matrices, package bootstrap, public runnable host-effect claims, or cross-host
parity guarantees.
