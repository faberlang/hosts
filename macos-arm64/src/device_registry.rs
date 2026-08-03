//! Shared opaque-handle registry for native device sessions (S1-4).
//!
//! Both native sessions (`CudaHostSession`, `MetalHostSession`) manage the
//! same lifecycle: a driver allocates a backend token, the session owns a
//! session-local opaque id, and every later call resolves the id back to the
//! token — never re-reading payload bytes from the caller. That registry was
//! proof-local (a private `BTreeMap` in each session). Productizing it as one
//! shared component removes the duplicated bookkeeping and keeps the
//! invariant central: a registry entry is an **id → (kind, backend token)**
//! pair and can never carry tensor payload.
//!
//! Session-side errors (stale id vs wrong-kind id vs invalid args) stay
//! session-specific; the registry returns `Option` and the session maps the
//! missing/kind cases onto its `E_*_INVALID_HANDLE` / `E_INVALID_ARGS` codes.

use std::collections::BTreeMap;

/// One registered opaque handle: the session-local id maps to this record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegisteredHandle<K> {
    /// What kind of device object the id names (module vs buffer).
    pub kind: K,
    /// Backend-owned token. Fake drivers use synthetic ids; the real driver
    /// adapters use driver handles (`CUmodule`, `CUdeviceptr`, Metal
    /// buffer/command state). Never tensor payload.
    pub backend_token: u64,
}

/// Opaque-handle registry: session-local id → [`RegisteredHandle`].
///
/// Ids are allocated monotonically from 1, so a session-owned handle id is
/// opaque to callers and unambiguous within the owning session.
#[derive(Default)]
pub struct HandleRegistry<K> {
    handles: BTreeMap<u64, RegisteredHandle<K>>,
    next_id: u64,
}

impl<K> HandleRegistry<K> {
    /// A new, empty registry. Ids start at 1 (0 is never handed out).
    #[must_use]
    pub fn new() -> Self {
        Self {
            handles: BTreeMap::new(),
            next_id: 1,
        }
    }

    /// Allocate a session-local id for a driver-owned token.
    #[must_use]
    pub fn insert(&mut self, kind: K, backend_token: u64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.handles.insert(
            id,
            RegisteredHandle {
                kind,
                backend_token,
            },
        );
        id
    }

    /// Resolve a live id to its registration; `None` for stale/unknown ids.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&RegisteredHandle<K>> {
        self.handles.get(&id)
    }

    /// Remove a registration, returning it for teardown; `None` for a stale
    /// id. This is the only way a registration leaves the registry, so a
    /// released id can never launch or be read back again.
    #[must_use]
    pub fn remove(&mut self, id: u64) -> Option<RegisteredHandle<K>> {
        self.handles.remove(&id)
    }

    /// Number of live registrations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.handles.len()
    }

    /// Whether the registry holds no live registrations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }
}

/// Driver-level lifecycle counters for the module-cache leak-free bar (S2-2).
///
/// The fake drivers increment these counters so a test can prove the cache
/// policy at the driver boundary: one module load per program session, one
/// release at teardown, buffers allocated once and released once, and
/// nothing persists past teardown. The real drivers do not track counters —
/// their leak evidence is the S2-8 real-device gate — so the default is
/// all-zero; this observability is for the fakes only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DriverCounters {
    /// Cumulative module loads (each program session loads its module once).
    pub module_loads: usize,
    /// Cumulative module releases (session teardown releases the module).
    pub module_releases: usize,
    /// Cumulative buffer allocations.
    pub buffer_allocs: usize,
    /// Cumulative buffer releases.
    pub buffer_releases: usize,
}
