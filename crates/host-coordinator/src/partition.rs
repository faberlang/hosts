//! Virtual device partition contract (gpu-inference-multi-device, MD1-V1).
//!
//! A [`VirtualDevicePartition`] is a **runtime/admission abstraction only**
//! (CTO `2f90eafd` §C1 default; md0-closeout §3.2): created by an admission
//! request, it binds **exactly one** [`PhysicalDeviceId`] — the physical
//! device whose budget it draws from — enforces deterministic admission
//! accounting, and is torn down. It is **never portable package content**
//! and never enters the A10 canonical bytes (naming contract §2):
//! partition facts live in bound-plan/runtime identity and receipts only.
//!
//! **Identity classes are distinct.** [`VirtualDevicePartitionId`] is a
//! separate machine-local identity class from [`PhysicalDeviceId`] — a
//! virtual partition never *receives* a physical id. The partition *binds* a
//! physical device as a value; the two classes have no conversion and no
//! equality between them.
//!
//! **Lifecycle:** [`AdmissionRequest`] → [`VirtualDevicePartition::admit`]
//! (or [`VirtualDevicePartition::implicit_local`], the MD-A15 degenerate)
//! → enforced accounting → [`VirtualDevicePartition::teardown`].
//!
//! **Failure taxonomy — three classes, never conflated** (CTO correction
//! #2; md0-closeout §3.2):
//!
//! - [`AdmissionError::BudgetExceeded`] — admission-time only: the declared
//!   requirements exceed the partition's safe physical limit. Deterministic
//!   fail-closed; no allocation has happened.
//! - [`PartitionFailure::AllocationFailure`] — a runtime allocation failed
//!   after admission (physical pressure). Post-admission only.
//! - [`PartitionFailure::DeviceLoss`] — the bound physical device failed or
//!   was removed (MD-A13). Post-admission only.
//!
//! **MD-A15 degenerate:** one-device execution derives an implicit/local
//! partition for admission/GI5 accounting with **no** distributed plan,
//! transfer graph, or execution-transaction coordination, and a single-device
//! FMIR package has no distributed section. This module defines no
//! distributed wrapper types and the degenerate constructor accepts none.
//!
//! **Budget ledger:** [`PartitionBudgetLedger`] covers all eight byte
//! classes of the admission equation, each as a concrete field or an
//! explicit bound. Class 8 — the safe physical limit
//! ([`SafePhysicalLimit`]) — is a **policy fact distinct from total
//! reported memory**: nvidia-smi MiB totals and driver-API byte totals are
//! device facts, never the admission limit, and on Metal unified memory is
//! OS-managed. "Partition budget = physical size" is forbidden wording.
//!
//! **Software admission:** partitions are **software admission partitions**
//! with `hardware_isolation_claimed=false`
//! ([`HardwareIsolationClaim`]); no hardware reservation or isolation is
//! claimed (CTO correction #5; md0-closeout §3.2 items 4–5).
//!
//! **Receipts:** [`PartitionReceipt`] carries the receipt taxonomy —
//! `physical_device_count`, `physical_device_ids`, `virtual_partition_count`,
//! `virtual_partition_ids`, `fixture_identity_class`, `transport_class`, and
//! `hardware_isolation_claimed=false` — and serializes deterministically
//! (canonical bytes; never A10 bytes).

use crate::device_identity::{push_bool, push_u64, PhysicalDeviceId};
use std::collections::BTreeSet;

/// Machine-local identity of one virtual partition.
///
/// A **distinct identity class from [`PhysicalDeviceId`]**: a virtual
/// partition never receives a physical id, and the two types have no
/// conversion and no equality between them. The runtime mints these ids
/// (machine-local, e.g. a monotonic counter) at admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VirtualDevicePartitionId(u64);

impl VirtualDevicePartitionId {
    /// Build a partition id from a machine-local seed.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The raw machine-local id value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for VirtualDevicePartitionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "vp:{}", self.0)
    }
}

/// The admission ledger: the complete accounting equation for one partition.
///
/// Every one of the eight byte classes is present as a field or an explicit
/// bound (md0-closeout §3.2 item 2; CTO `2f90eafd` correction #2):
///
/// 1. admitted weight bytes, including any repack/duplication;
/// 2. KV bytes **per `KvCacheLayout`** — consumed, not re-derived (GI4 owns
///    the layout: slots, context, layers/heads, dtype, reserve policy; this
///    partition never recomputes it — it is an explicit bound supplied by
///    the caller at admission);
/// 3. peak activations plus operation scratch/workspace;
/// 4. module/kernel/descriptor storage where material (explicit bound: `0`
///    when the backend holds modules host-side or loads them lazily);
/// 5. allocator granularity/alignment/fragmentation/headroom — the computed
///    overhead contribution for this admission;
/// 6. transfer/staging buffers plus in-flight copies, at **full size** in the
///    budget; pinned host allocations do **not** consume the device budget
///    (T1 §2.3);
/// 7. concurrent requests/models plus pinned/in-flight state;
/// 8. a **safe physical limit policy distinct from total reported memory** —
///    carried as [`SafePhysicalLimit`] beside the ledger.
///
/// The ledger is **frozen at admission**: enforced accounting means the
/// admitted totals are immutable, and any post-admission growth is a runtime
/// allocation outcome ([`PartitionFailure::AllocationFailure`]), never a
/// silent ledger rewrite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PartitionBudgetLedger {
    /// (1) Admitted weight bytes, including any repack/duplication.
    pub weight_bytes: u64,
    /// (2) KV bytes per `KvCacheLayout`. **Consumed, not re-derived** — GI4
    /// owns the layout (slots, context, layers/heads, dtype, reserve
    /// policy); the partition never recomputes it.
    pub kv_cache_bytes: u64,
    /// (3) Peak activations plus operation scratch/workspace bytes.
    pub activation_scratch_bytes: u64,
    /// (4) Module/kernel/descriptor storage where material; explicit bound
    /// `0` when the backend holds modules host-side or loads lazily.
    pub module_storage_bytes: u64,
    /// (5) Allocator granularity/alignment/fragmentation/headroom overhead
    /// contribution for this admission, as computed by the caller's allocator
    /// policy.
    pub allocator_overhead_bytes: u64,
    /// (6) Transfer/staging buffers plus in-flight copies, at full size.
    /// Pinned host allocations do not consume the device budget (T1 §2.3).
    pub transfer_staging_bytes: u64,
    /// (7) Concurrent requests/models plus pinned/in-flight state bytes.
    pub concurrent_state_bytes: u64,
}

impl PartitionBudgetLedger {
    /// The checked sum of the seven byte classes (classes 1–7). `None` on
    /// overflow — admission treats overflow as `budget_exceeded`
    /// (deterministic fail-closed; no allocation happens).
    #[must_use]
    pub fn total_bytes(&self) -> Option<u64> {
        let mut total = 0u64;
        total = total.checked_add(self.weight_bytes)?;
        total = total.checked_add(self.kv_cache_bytes)?;
        total = total.checked_add(self.activation_scratch_bytes)?;
        total = total.checked_add(self.module_storage_bytes)?;
        total = total.checked_add(self.allocator_overhead_bytes)?;
        total = total.checked_add(self.transfer_staging_bytes)?;
        total = total.checked_add(self.concurrent_state_bytes)?;
        Some(total)
    }
}

/// A policy-declared admission ceiling for one partition — **a policy fact,
/// never a hardware report**.
///
/// Distinct from total reported memory: the nvidia-smi MiB total and the
/// driver-API byte total are device facts (T1 §8), never the admission
/// limit. On Metal, unified memory is OS-managed — "partition budget =
/// physical size" is forbidden wording. The host admission layer derives the
/// limit from its memory policy (tenant reservation, OS-managed pressure,
/// fragmentation headroom); it is not derived from a memory *total* alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SafePhysicalLimit(u64);

impl SafePhysicalLimit {
    /// Build a safe physical limit from a policy-declared byte ceiling.
    #[must_use]
    pub const fn new(policy_limit_bytes: u64) -> Self {
        Self(policy_limit_bytes)
    }

    /// The policy-declared byte ceiling.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// An admission request: declared requirements plus the bound physical device.
///
/// The request carries **exactly one** [`PhysicalDeviceId`] — the physical
/// device whose budget the partition draws from.
#[derive(Debug, Clone)]
pub struct AdmissionRequest {
    partition_id: VirtualDevicePartitionId,
    bound_device: PhysicalDeviceId,
    declared: PartitionBudgetLedger,
}

impl AdmissionRequest {
    /// Build an admission request for one partition over exactly one physical
    /// device.
    #[must_use]
    pub const fn new(
        partition_id: VirtualDevicePartitionId,
        bound_device: PhysicalDeviceId,
        declared: PartitionBudgetLedger,
    ) -> Self {
        Self {
            partition_id,
            bound_device,
            declared,
        }
    }

    /// The requested partition id.
    #[must_use]
    pub const fn partition_id(&self) -> VirtualDevicePartitionId {
        self.partition_id
    }

    /// The single bound physical device.
    #[must_use]
    pub fn bound_device(&self) -> &PhysicalDeviceId {
        &self.bound_device
    }

    /// The declared requirements ledger.
    #[must_use]
    pub fn declared(&self) -> &PartitionBudgetLedger {
        &self.declared
    }
}

/// Admission-time failure — the **only** error class admission can return.
///
/// Post-admission failures are [`PartitionFailure`]; the two types are never
/// conflated, so a `budget_exceeded` outcome can never be produced after a
/// partition exists and an `allocation_failure`/`device_loss` can never be an
/// admission outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionError {
    /// The declared requirements exceed the partition's safe physical limit.
    /// Deterministic fail-closed: no allocation has happened.
    BudgetExceeded {
        /// The declared total; `None` when the declared classes overflowed
        /// `u64` (still fail-closed).
        declared_total_bytes: Option<u64>,
        /// The policy-declared ceiling that was exceeded.
        policy_limit_bytes: u64,
    },
}

/// Post-admission failure of a bound partition. Never an admission outcome
/// (admission rejects with [`AdmissionError::BudgetExceeded`] instead).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartitionFailure {
    /// A runtime allocation failed after admission (physical pressure).
    AllocationFailure {
        /// What failed, as reported by the runtime.
        detail: String,
    },
    /// The bound physical device failed or was removed (MD-A13).
    DeviceLoss {
        /// What happened to the device, as reported by the runtime.
        detail: String,
    },
}

/// Lifecycle state of a bound partition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartitionState {
    /// Admitted and bound; deterministic accounting is enforced. The ledger
    /// is frozen; further growth is a runtime allocation outcome.
    Active,
    /// A post-admission failure ended the partition ([`PartitionFailure`]).
    Failed(PartitionFailure),
    /// Teardown completed; the partition no longer draws a budget.
    TornDown,
}

/// A virtual device partition — a runtime/admission abstraction that binds
/// **exactly one** physical device and enforces a declared admission budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualDevicePartition {
    id: VirtualDevicePartitionId,
    bound_device: PhysicalDeviceId,
    ledger: PartitionBudgetLedger,
    safe_limit: SafePhysicalLimit,
    state: PartitionState,
}

impl VirtualDevicePartition {
    /// Admit a partition. Deterministic fail-closed: when the declared
    /// requirements exceed the partition's safe physical limit (or overflow),
    /// admission is rejected as [`AdmissionError::BudgetExceeded`] before any
    /// allocation.
    #[must_use]
    pub fn admit(
        request: AdmissionRequest,
        safe_limit: SafePhysicalLimit,
    ) -> Result<Self, AdmissionError> {
        let limit = safe_limit.get();
        match request.declared.total_bytes() {
            Some(total) if total <= limit => Ok(Self {
                id: request.partition_id,
                bound_device: request.bound_device,
                ledger: request.declared,
                safe_limit,
                state: PartitionState::Active,
            }),
            Some(total) => Err(AdmissionError::BudgetExceeded {
                declared_total_bytes: Some(total),
                policy_limit_bytes: limit,
            }),
            None => Err(AdmissionError::BudgetExceeded {
                declared_total_bytes: None,
                policy_limit_bytes: limit,
            }),
        }
    }

    /// MD-A15 degenerate: derive an implicit/local partition for one-device
    /// admission/GI5 accounting.
    ///
    /// A single-device package has no distributed section, and one-device
    /// execution performs **no** `DistributedExecutionPlan`, transfer-graph,
    /// or `ExecutionTransaction` coordination. This constructor accepts no
    /// distributed input and this module defines no distributed wrapper
    /// types — the partition is purely local runtime state.
    #[must_use]
    pub fn implicit_local(
        id: VirtualDevicePartitionId,
        device: PhysicalDeviceId,
        ledger: PartitionBudgetLedger,
        safe_limit: SafePhysicalLimit,
    ) -> Result<Self, AdmissionError> {
        Self::admit(AdmissionRequest::new(id, device, ledger), safe_limit)
    }

    /// The partition's machine-local id (virtual identity class).
    #[must_use]
    pub const fn id(&self) -> VirtualDevicePartitionId {
        self.id
    }

    /// The exactly-one bound physical device whose budget this partition
    /// draws from.
    #[must_use]
    pub fn bound_device(&self) -> &PhysicalDeviceId {
        &self.bound_device
    }

    /// The frozen admitted ledger (enforced accounting).
    #[must_use]
    pub fn ledger(&self) -> &PartitionBudgetLedger {
        &self.ledger
    }

    /// The safe physical limit under which this partition was admitted.
    #[must_use]
    pub const fn safe_physical_limit(&self) -> SafePhysicalLimit {
        self.safe_limit
    }

    /// The current lifecycle state.
    #[must_use]
    pub fn state(&self) -> &PartitionState {
        &self.state
    }

    /// True when the partition is admitted and enforcing its budget.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.state == PartitionState::Active
    }

    /// The post-admission failure, when the partition has one.
    #[must_use]
    pub fn failure(&self) -> Option<&PartitionFailure> {
        match &self.state {
            PartitionState::Failed(failure) => Some(failure),
            PartitionState::Active | PartitionState::TornDown => None,
        }
    }

    /// The admitted total of the seven byte classes.
    #[must_use]
    pub fn admitted_total_bytes(&self) -> Option<u64> {
        self.ledger.total_bytes()
    }

    /// Record a post-admission failure. **First failure wins**: a later
    /// record never downgrades or overwrites an earlier one, so a
    /// `DeviceLoss` is never relabeled as an `AllocationFailure` (or vice
    /// versa) — the taxonomy classes stay distinct.
    pub fn record_failure(&mut self, failure: PartitionFailure) {
        if matches!(self.state, PartitionState::Active) {
            self.state = PartitionState::Failed(failure);
        }
    }

    /// Record a post-admission allocation failure (physical pressure) —
    /// [`PartitionFailure::AllocationFailure`].
    pub fn record_allocation_failure(&mut self, detail: impl Into<String>) {
        self.record_failure(PartitionFailure::AllocationFailure {
            detail: detail.into(),
        });
    }

    /// Report that the bound physical device failed or was removed (MD-A13)
    /// — [`PartitionFailure::DeviceLoss`].
    pub fn report_device_loss(&mut self, detail: impl Into<String>) {
        self.record_failure(PartitionFailure::DeviceLoss {
            detail: detail.into(),
        });
    }

    /// Tear down the partition: it stops drawing a budget. Idempotent.
    pub fn teardown(&mut self) {
        self.state = PartitionState::TornDown;
    }
}

/// `hardware_isolation_claimed=false` — enforced in the type surface.
///
/// Partitions are **software admission partitions**: they account and gate
/// admission deterministically, but claim **no hardware reservation or
/// isolation**. This type has exactly one value, so a receipt can never
/// carry a true isolation claim (CTO correction #5; md0-closeout §3.2 items
/// 4–5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HardwareIsolationClaim {
    /// No hardware isolation is claimed.
    NotClaimed,
}

/// What kind of fixture produced a receipt (md0-closeout §3.2 item 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FixtureIdentityClass {
    /// Real physical device(s) on the acceptance host.
    Physical,
    /// Virtual partitions over physical device(s) — hardware isolation NOT
    /// claimed.
    Virtual,
    /// A synthetic/derived fixture (snapshot-driven; no live device).
    Synthetic,
}

impl FixtureIdentityClass {
    /// Deterministic serialization tag.
    #[must_use]
    pub const fn tag(self) -> u64 {
        match self {
            FixtureIdentityClass::Physical => 0,
            FixtureIdentityClass::Virtual => 1,
            FixtureIdentityClass::Synthetic => 2,
        }
    }
}

/// Transport evidence class for a receipt. A NOT-ATTEMPTED row is never
/// mistaken for a pass (T1 §8; md0-closeout §3.2 item 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransportClass {
    /// No transport exercised (single device).
    None,
    /// Host-staged copies (T1 measured host staging on the acceptance host).
    HostStaged,
    /// Directed peer/interconnect pairs — NOT ATTEMPTED until a real
    /// same-host ≥2-device topology is admitted.
    DirectedPeerNotAttempted,
}

impl TransportClass {
    /// Deterministic serialization tag.
    #[must_use]
    pub const fn tag(self) -> u64 {
        match self {
            TransportClass::None => 0,
            TransportClass::HostStaged => 1,
            TransportClass::DirectedPeerNotAttempted => 2,
        }
    }
}

/// A partition receipt — the receipt taxonomy (md0-closeout §3.2 item 4).
///
/// Fields: `physical_device_count`, `physical_device_ids`,
/// `virtual_partition_count`, `virtual_partition_ids`,
/// `fixture_identity_class`, `transport_class`, and
/// `hardware_isolation_claimed=false`. Counts derive from the id sets, so a
/// count can never disagree with its ids. Serializes deterministically
/// (canonical bytes) for runtime identity and bound-plan receipts — never
/// A10 canonical package bytes (naming contract §2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionReceipt {
    physical_device_count: u64,
    physical_device_ids: BTreeSet<PhysicalDeviceId>,
    virtual_partition_count: u64,
    virtual_partition_ids: BTreeSet<VirtualDevicePartitionId>,
    fixture_identity_class: FixtureIdentityClass,
    transport_class: TransportClass,
    hardware_isolation_claimed: HardwareIsolationClaim,
}

impl PartitionReceipt {
    /// Build a receipt. Counts are derived from the id sets (single source
    /// of truth — they cannot disagree).
    #[must_use]
    pub fn new(
        physical_device_ids: BTreeSet<PhysicalDeviceId>,
        virtual_partition_ids: BTreeSet<VirtualDevicePartitionId>,
        fixture_identity_class: FixtureIdentityClass,
        transport_class: TransportClass,
    ) -> Self {
        let physical_device_count = physical_device_ids.len() as u64;
        let virtual_partition_count = virtual_partition_ids.len() as u64;
        Self {
            physical_device_count,
            physical_device_ids,
            virtual_partition_count,
            virtual_partition_ids,
            fixture_identity_class,
            transport_class,
            hardware_isolation_claimed: HardwareIsolationClaim::NotClaimed,
        }
    }

    /// The number of distinct physical devices in this receipt.
    #[must_use]
    pub const fn physical_device_count(&self) -> u64 {
        self.physical_device_count
    }

    /// The distinct physical device ids in this receipt.
    #[must_use]
    pub fn physical_device_ids(&self) -> &BTreeSet<PhysicalDeviceId> {
        &self.physical_device_ids
    }

    /// The number of virtual partitions in this receipt.
    #[must_use]
    pub const fn virtual_partition_count(&self) -> u64 {
        self.virtual_partition_count
    }

    /// The virtual partition ids in this receipt.
    #[must_use]
    pub fn virtual_partition_ids(&self) -> &BTreeSet<VirtualDevicePartitionId> {
        &self.virtual_partition_ids
    }

    /// The fixture identity class of this receipt.
    #[must_use]
    pub const fn fixture_identity_class(&self) -> FixtureIdentityClass {
        self.fixture_identity_class
    }

    /// The transport evidence class of this receipt.
    #[must_use]
    pub const fn transport_class(&self) -> TransportClass {
        self.transport_class
    }

    /// `hardware_isolation_claimed` — always [`HardwareIsolationClaim::NotClaimed`]
    /// (software admission partitions; no hardware isolation claimed).
    #[must_use]
    pub const fn hardware_isolation_claimed(&self) -> HardwareIsolationClaim {
        self.hardware_isolation_claimed
    }

    /// Deterministic canonical bytes of every receipt field. Identical
    /// receipts produce identical bytes; changing any field changes the
    /// bytes. These are runtime/bound-plan receipt bytes — never A10
    /// canonical package bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_u64(&mut out, self.physical_device_count);
        push_u64(&mut out, self.physical_device_ids.len() as u64);
        for id in &self.physical_device_ids {
            out.extend_from_slice(&id.canonical_bytes());
        }
        push_u64(&mut out, self.virtual_partition_count);
        push_u64(&mut out, self.virtual_partition_ids.len() as u64);
        for id in &self.virtual_partition_ids {
            push_u64(&mut out, id.get());
        }
        push_u64(&mut out, self.fixture_identity_class.tag());
        push_u64(&mut out, self.transport_class.tag());
        // hardware_isolation_claimed=false — the single-variant type makes a
        // true claim unrepresentable; the exhaustive match keeps the
        // serialization honest if a variant is ever added.
        let claimed = match self.hardware_isolation_claimed {
            HardwareIsolationClaim::NotClaimed => false,
        };
        push_bool(&mut out, claimed);
        out
    }
}

#[cfg(test)]
#[path = "partition_test.rs"]
mod tests;
