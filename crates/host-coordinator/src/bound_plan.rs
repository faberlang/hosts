//! `BoundDistributedPlan` — the topology-bound plan (gpu-inference-multi-device,
//! MD2-B1; CTO `06ac0f04`; naming contract §3).
//!
//! Binds an **admitted** logical `DistributedExecutionPlan` — referenced only
//! by its opaque `logical_distributed_plan_hash` (faber-runtime cannot import
//! radix-mir, FC18) — to MD1's [`PhysicalDeviceId`]/[`DeviceSet`]/
//! [`DeviceDiscoverySnapshotId`]/[`VirtualDevicePartition`] types.
//!
//! Frozen contracts:
//!
//! - **Three hash domains** (md0-naming-contract.md §2): the A10 semantic
//!   `device_identity_hash` is unchanged and physical ids/ordinals/topology
//!   never enter it; `logical_distributed_plan_hash` and
//!   `bound_distributed_plan_hash` are **additional identities, never
//!   replacements**. Physical ids **do** enter the bound-plan hash domain.
//! - **Hash spelling** (FC17/FC11): every plan hash is `sha256:` + 64
//!   lowercase hex over its class's canonical bytes.
//! - **Bind only after admission** (MD-A6): [`bind`] consumes an
//!   [`AdmittedLogicalPlan`] marker produced by the admission path
//!   ([`AdmittedLogicalPlan::admit`]); the logical plan is never
//!   structurally re-derived here.
//! - **Bind contract**: rejects a stale/unadmitted logical hash, an
//!   unknown/replaced [`PhysicalDeviceId`] (MD1 health-epoch rule), a device
//!   set that does not match the plan's declared partition set/bindings
//!   (topology mismatch), any binding that would violate a declared
//!   [`DeclaredPlacementConstraint`], and a declared [`LaunchResourceDemand`]
//!   that exceeds the bound device's generic launch-resource limits.
//! - **MD-A15 degenerate**: a single-partition logical plan binds to the
//!   implicit/local partition ([`BoundPlanKind::ImplicitLocal`]) with **no**
//!   distributed wrapper, transfer graph, or `ExecutionTransaction` —
//!   one-device execution stays coordinator-free.
//! - **Receipt taxonomy** (md0-closeout.md §3.2 #4): the bound plan derives a
//!   [`PartitionReceipt`] — `physical_device_count`, `physical_device_ids`,
//!   `virtual_partition_count`, `virtual_partition_ids`,
//!   `fixture_identity_class`, `transport_class`,
//!   `hardware_isolation_claimed=false`. Virtual partitions never receive a
//!   [`PhysicalDeviceId`].
//!
//! Structural runtime binding (kernels/transfers to runtimes) is MD3 +
//! MD1-H1.

use crate::backend::DeviceBackend;
use crate::device_identity::{push_str, push_u64, DeviceHealthGeneration, PhysicalDeviceId};
use crate::device_set::{DeviceSet, MembershipError};
use crate::discovery::{DeviceCapabilities, DeviceDiscoverySnapshot, DeviceDiscoverySnapshotId};
use crate::partition::{
    FixtureIdentityClass, PartitionReceipt, TransportClass, VirtualDevicePartition,
};
use std::collections::{BTreeMap, BTreeSet};

/// Machine-local opaque identity of one logical partition of the admitted
/// plan (naming contract §3).
///
/// The logical partition identity class lives in radix-mir (MD2-D1,
/// `DeviceProgramPartition`); faber-runtime cannot import it (FC18), so the
/// identity travels as this opaque stable string. It is **not** a
/// [`VirtualDevicePartitionId`](crate::partition::VirtualDevicePartitionId)
/// (that class is minted by admission at runtime) and never a
/// [`PhysicalDeviceId`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LogicalPartitionId(String);

impl LogicalPartitionId {
    /// Build an opaque partition identity from its stable string.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The stable identity string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for LogicalPartitionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A dependency-free mirror of a declared logical `PlacementConstraint`
/// (naming contract §3; CTO `06ac0f04` §2).
///
/// The logical placement vocabulary is owned by radix-mir's distributed
/// module (MD2-D1); faber-runtime cannot import it (FC18), so [`bind`]
/// receives the **declared** facts in this mirror. Only the checks a
/// topology bind can evaluate are carried — the placement *planner* is MD5's
/// scope and never lives here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeclaredPlacementConstraint {
    /// Every declared partition binds to a distinct physical device.
    DistinctPhysicalDevices,
    /// The named partitions must all bind to the same physical device.
    Colocated {
        /// The colocated partition set.
        partitions: BTreeSet<LogicalPartitionId>,
    },
    /// The named partitions must bind to a device of the declared backend.
    RequiredBackend {
        /// The constrained partition set.
        partitions: BTreeSet<LogicalPartitionId>,
        /// The backend each named partition must bind to.
        backend: DeviceBackend,
    },
}

/// Declared launch-resource demand of an admitted logical plan (DCG-4).
///
/// Checked at [`bind`] against each bound device's generic
/// [`DeviceCapabilities`] fields. `None` on a field leaves that check
/// undeclared. A declared demand that exceeds the device ceiling, or a
/// zero/missing device fact, rejects fail-closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LaunchResourceDemand {
    /// Declared workgroup thread volume. Rejects when it exceeds the device
    /// `max_threads_per_workgroup` ceiling.
    pub threads_per_workgroup: Option<u32>,
    /// Declared workgroup shared-memory demand, bytes. Rejects when it
    /// exceeds the device `workgroup_shared_memory_max_bytes` opt-in
    /// ceiling.
    pub workgroup_shared_memory_bytes: Option<u32>,
}

/// Why the runtime admission path rejected a logical-plan reference
/// ([`AdmittedLogicalPlan::admit`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmitError {
    /// The hash is not in the `sha256:` + 64-lowercase-hex spelling
    /// (FC17/FC11) — it cannot be a validated
    /// `logical_distributed_plan_hash`, so it is stale or unadmitted and is
    /// never accepted by the admission path.
    UnadmittedLogicalHash {
        /// The rejected value.
        given: String,
    },
    /// The declared partition set is empty — a logical plan declares at
    /// least one partition.
    NoDeclaredPartitions,
    /// A declared placement constraint references a partition that was not
    /// declared.
    ConstraintReferencesUnknownPartition {
        /// The undeclared partition reference.
        partition: LogicalPartitionId,
    },
}

/// The admitted-hash marker (MD-A6).
///
/// Produced **only** by the runtime admission path — `admit` — which the
/// host calls after the logical-plan validation/admission (the MD2-W1
/// equivalence gate) has produced the `logical_distributed_plan_hash` and
/// the plan's declared partition facts. The logical plan is never
/// structurally re-derived here. [`bind`] consumes a marker, so a
/// `BoundDistributedPlan` can never be built from a stale or unadmitted
/// hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmittedLogicalPlan {
    logical_distributed_plan_hash: String,
    declared_partitions: BTreeSet<LogicalPartitionId>,
    declared_constraints: Vec<DeclaredPlacementConstraint>,
    launch_resource_demand: LaunchResourceDemand,
}

impl AdmittedLogicalPlan {
    /// Admit a validated logical-plan reference.
    ///
    /// - The hash must be in the `sha256:` + 64-lowercase-hex spelling
    ///   (FC17/FC11) — anything else is rejected as stale/unadmitted.
    /// - At least one partition must be declared.
    /// - Every declared constraint must reference only declared partitions.
    ///
    /// # Errors
    ///
    /// Returns [`AdmitError`] when the hash spelling is invalid, no
    /// partitions are declared, or a constraint names an unknown partition.
    #[must_use]
    pub fn admit(
        logical_distributed_plan_hash: impl Into<String>,
        declared_partitions: impl IntoIterator<Item = LogicalPartitionId>,
        declared_constraints: impl IntoIterator<Item = DeclaredPlacementConstraint>,
    ) -> Result<Self, AdmitError> {
        let hash = logical_distributed_plan_hash.into();
        if !is_sha256_hex(&hash) {
            return Err(AdmitError::UnadmittedLogicalHash { given: hash });
        }
        let partitions: BTreeSet<LogicalPartitionId> = declared_partitions.into_iter().collect();
        if partitions.is_empty() {
            return Err(AdmitError::NoDeclaredPartitions);
        }
        let constraints: Vec<DeclaredPlacementConstraint> =
            declared_constraints.into_iter().collect();
        for constraint in &constraints {
            if let Some(partition) = constraint_unknown_partition(constraint, &partitions) {
                return Err(AdmitError::ConstraintReferencesUnknownPartition { partition });
            }
        }
        Ok(Self {
            logical_distributed_plan_hash: hash,
            declared_partitions: partitions,
            declared_constraints: constraints,
            launch_resource_demand: LaunchResourceDemand::default(),
        })
    }

    /// Attach a declared launch-resource demand checked at [`bind`].
    ///
    /// Absence (the default from [`Self::admit`]) leaves the limits
    /// undeclared. Existing topology-only admits bind unchanged.
    #[must_use]
    pub fn with_launch_resource_demand(mut self, demand: LaunchResourceDemand) -> Self {
        self.launch_resource_demand = demand;
        self
    }

    /// The declared launch-resource demand. All fields are `None` when undeclared.
    #[must_use]
    pub const fn launch_resource_demand(&self) -> LaunchResourceDemand {
        self.launch_resource_demand
    }

    /// The admitted opaque logical-plan hash.
    #[must_use]
    pub fn logical_distributed_plan_hash(&self) -> &str {
        &self.logical_distributed_plan_hash
    }

    /// The declared partition identities, in stable order.
    #[must_use]
    pub fn declared_partitions(&self) -> &BTreeSet<LogicalPartitionId> {
        &self.declared_partitions
    }

    /// The declared placement constraints, in declaration order.
    #[must_use]
    pub fn declared_constraints(&self) -> &[DeclaredPlacementConstraint] {
        &self.declared_constraints
    }

    /// The declared partition count.
    #[must_use]
    pub fn declared_partition_count(&self) -> usize {
        self.declared_partitions.len()
    }

    /// True when the logical plan declares exactly one partition — the
    /// MD-A15 degenerate shape.
    #[must_use]
    pub fn is_single_partition(&self) -> bool {
        self.declared_partitions.len() == 1
    }
}

/// One partition binding: a logical partition bound to exactly one
/// [`PhysicalDeviceId`], optionally carrying the already-admitted
/// [`VirtualDevicePartition`] that draws its budget from that device
/// (MD1-V1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionBinding {
    device: PhysicalDeviceId,
    virtual_partition: Option<VirtualDevicePartition>,
}

impl PartitionBinding {
    /// A device-only binding (no attached virtual partition).
    #[must_use]
    pub fn new(device: PhysicalDeviceId) -> Self {
        Self {
            device,
            virtual_partition: None,
        }
    }

    /// A binding carrying an already-admitted virtual partition. The
    /// partition must bind the same physical device and be active — [`bind`]
    /// rejects an inconsistent or inactive attachment.
    #[must_use]
    pub fn with_virtual_partition(
        device: PhysicalDeviceId,
        partition: VirtualDevicePartition,
    ) -> Self {
        Self {
            device,
            virtual_partition: Some(partition),
        }
    }

    /// The bound physical device.
    #[must_use]
    pub fn device(&self) -> &PhysicalDeviceId {
        &self.device
    }

    /// The attached virtual partition, when one was bound.
    #[must_use]
    pub fn virtual_partition(&self) -> Option<&VirtualDevicePartition> {
        self.virtual_partition.as_ref()
    }
}

/// The topology-bound plan shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoundPlanKind {
    /// **MD-A15 degenerate**: one logical partition binds to the
    /// implicit/local partition on one local physical device. There is **no**
    /// distributed wrapper, no transfer graph, and no `ExecutionTransaction`
    /// — one-device execution stays coordinator-free.
    ImplicitLocal {
        /// The single bound physical device.
        device: PhysicalDeviceId,
        /// The implicit/local partition, when admission created one (MD1-V1
        /// `implicit_local` accounting).
        virtual_partition: Option<VirtualDevicePartition>,
    },
    /// A multi-partition topology-bound plan with explicit per-partition
    /// bindings.
    Distributed {
        /// Logical partition → binding, in stable partition-identity order.
        bindings: BTreeMap<LogicalPartitionId, PartitionBinding>,
    },
}

/// Why [`bind`] rejected a topology-bound plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindError {
    /// The discovery snapshot carries a stale health generation — it can
    /// never be the frozen basis of a bind (MD1-Q3; naming contract §1).
    StaleSnapshot {
        /// The rejected snapshot.
        snapshot_id: DeviceDiscoverySnapshotId,
        /// The generation current at bind time.
        current: DeviceHealthGeneration,
    },
    /// A bound device is not a current-epoch member of the snapshot — an
    /// unknown device or a **replaced** device (same ordinal, different
    /// identity facts; naming contract §1). Replacement yields a *new* id
    /// that no snapshot entry carries until the next probe.
    Membership(MembershipError),
    /// The device set does not match the plan's declared partition
    /// set/bindings (topology mismatch): the binding keys must be **exactly**
    /// the declared partitions, and the device set must be **exactly** the
    /// distinct physical devices the bindings reference.
    TopologyMismatch {
        /// The failing fact.
        detail: String,
    },
    /// A binding carries a virtual partition inconsistent with the binding —
    /// it binds a different physical device, or is not active.
    InvalidPartitionBinding {
        /// The offending logical partition.
        partition: LogicalPartitionId,
        /// The failing fact.
        detail: String,
    },
    /// A binding's declared requirements exceed the policy-derived safe
    /// physical limit — admission-time only, deterministic fail-closed,
    /// before any allocation (the `AdmissionError::BudgetExceeded`
    /// taxonomy surfaced at the bind seam; MD3J-B2).
    BudgetExceeded {
        /// The offending logical partition.
        partition: LogicalPartitionId,
        /// The declared total; `None` when the declared classes overflowed
        /// `u64` (still fail-closed).
        declared_total_bytes: Option<u64>,
        /// The policy-declared ceiling that was exceeded — the named
        /// headroom policy, never the raw memory total, never the declared
        /// budget.
        policy_limit_bytes: u64,
    },
    /// A declared [`DeclaredPlacementConstraint`] is violated by the
    /// bindings.
    ConstraintViolation {
        /// The violated constraint.
        constraint: DeclaredPlacementConstraint,
        /// The failing fact.
        detail: String,
    },
    /// A declared [`LaunchResourceDemand`] exceeds the bound device's
    /// generic launch-resource limit, or the device fact is unevaluable
    /// (zero sentinel / missing snapshot entry).
    LaunchResourceLimit {
        /// The bound device whose limit failed.
        device: PhysicalDeviceId,
        /// The failing fact.
        detail: String,
    },
}

/// The topology-bound plan (naming contract §3).
///
/// Holds the admitted logical plan's **opaque**
/// `logical_distributed_plan_hash`, the per-partition bindings (or the
/// MD-A15 implicit/local degenerate), the bound [`DeviceSet`] +
/// [`DeviceDiscoverySnapshotId`], and `bound_distributed_plan_hash` —
/// `sha256:` over the bound-plan canonical bytes (logical hash + binding
/// facts). Physical ids enter **this** hash domain, never A10.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundDistributedPlan {
    logical_distributed_plan_hash: String,
    kind: BoundPlanKind,
    device_set: DeviceSet,
    snapshot_id: DeviceDiscoverySnapshotId,
    fixture_identity_class: FixtureIdentityClass,
    transport_class: TransportClass,
    #[allow(clippy::struct_field_names)] // contract identity, not a struct-name prefix
    bound_distributed_plan_hash: String,
}

/// Bind an admitted logical plan to a physical topology.
///
/// `admitted` is the admitted-hash marker produced by the admission path
/// (MD-A6) — a `BoundDistributedPlan` cannot be built without one, and the
/// logical plan is never structurally re-derived here. `bindings` maps every
/// declared logical partition to exactly one physical device (optionally
/// carrying an already-admitted [`VirtualDevicePartition`]); `device_set` is
/// the selected membership the plan binds to; `snapshot` + `current_generation`
/// freeze the health epoch the bind is validated against; the two receipt
/// classes are declared by the caller (md0-closeout §3.2 #4).
///
/// Rejections, in deterministic order:
///
/// 1. **Health epoch** — a stale snapshot, or an unknown/replaced bound
///    device (MD1 health-epoch rule), rejects before anything else.
/// 2. **Topology shape** — the binding keys must be exactly the declared
///    partitions, and the device set must be exactly the distinct physical
///    devices the bindings reference (topology mismatch).
/// 3. **Binding consistency** — an attached virtual partition must bind the
///    same physical device and be active.
/// 4. **Declared placement constraints** — the first violated constraint in
///    declaration order rejects.
/// 5. **Launch-resource limits** — a declared threadgroup or shared-memory
///    demand that exceeds the bound device's generic capability fields
///    rejects fail-closed (DCG-4).
///
/// A single-partition admitted plan binds to the implicit/local partition
/// ([`BoundPlanKind::ImplicitLocal`]) — no distributed wrapper (MD-A15).
///
/// # Errors
///
/// Returns [`BindError`] when the snapshot is stale, membership fails, the
/// binding topology does not match, a virtual partition is inconsistent, a
/// declared placement constraint is violated, or a declared launch-resource
/// demand exceeds the bound device's generic limits.
///
/// # Panics
///
/// Panics if a single-partition admitted plan has no binding after the
/// topology-shape check (a programmer invariant).
#[must_use]
pub fn bind(
    admitted: &AdmittedLogicalPlan,
    bindings: BTreeMap<LogicalPartitionId, PartitionBinding>,
    device_set: DeviceSet,
    snapshot: &DeviceDiscoverySnapshot,
    current_generation: DeviceHealthGeneration,
    fixture_identity_class: FixtureIdentityClass,
    transport_class: TransportClass,
) -> Result<BoundDistributedPlan, BindError> {
    // 1. Health epoch (MD1-Q3): a stale snapshot or a stale/unknown/replaced
    //    member never gates a bind.
    if snapshot.is_stale(current_generation) {
        return Err(BindError::StaleSnapshot {
            snapshot_id: snapshot.id(),
            current: current_generation,
        });
    }
    device_set
        .validate(snapshot, current_generation)
        .map_err(BindError::Membership)?;

    // 2. Topology shape: the binding keys must be exactly the declared
    //    partitions.
    let declared_keys: BTreeSet<&LogicalPartitionId> =
        admitted.declared_partitions().iter().collect();
    let bound_keys: BTreeSet<&LogicalPartitionId> = bindings.keys().collect();
    if declared_keys != bound_keys {
        return Err(BindError::TopologyMismatch {
            detail: format!(
                "binding set does not match the declared partition set: declared {} partition(s), {} binding(s)",
                admitted.declared_partition_count(),
                bindings.len()
            ),
        });
    }

    // 2b. Topology shape: the bound device set must be exactly the distinct
    //     physical devices the bindings reference.
    let bound_devices: BTreeSet<PhysicalDeviceId> =
        bindings.values().map(|b| b.device().clone()).collect();
    if &bound_devices != device_set.members() {
        return Err(BindError::TopologyMismatch {
            detail: format!(
                "the device set does not match the bound device set: set has {} member(s), bindings reference {}",
                device_set.len(),
                bound_devices.len()
            ),
        });
    }

    // 3. Binding consistency: an attached virtual partition must bind the
    //    same physical device and be active.
    for (partition, binding) in &bindings {
        if let Some(partition_instance) = binding.virtual_partition() {
            if partition_instance.bound_device() != binding.device() {
                return Err(BindError::InvalidPartitionBinding {
                    partition: partition.clone(),
                    detail: format!(
                        "attached partition {} binds {}, but the binding names {}",
                        partition_instance.id(),
                        partition_instance.bound_device(),
                        binding.device()
                    ),
                });
            }
            if !partition_instance.is_active() {
                return Err(BindError::InvalidPartitionBinding {
                    partition: partition.clone(),
                    detail: format!(
                        "attached partition {} is not active",
                        partition_instance.id()
                    ),
                });
            }
        }
    }

    // 4. Declared placement constraints, in declaration order.
    for constraint in admitted.declared_constraints() {
        if let Err(detail) = check_constraint(constraint, &bindings) {
            return Err(BindError::ConstraintViolation {
                constraint: constraint.clone(),
                detail,
            });
        }
    }

    // 5. Declared launch-resource limits against each bound device.
    check_launch_resource_limits(admitted.launch_resource_demand(), &bound_devices, snapshot)?;

    let snapshot_id = snapshot.id();
    let kind = if admitted.is_single_partition() {
        // MD-A15 degenerate: the one declared partition binds the
        // implicit/local partition — no distributed wrapper.
        let (_, binding) = bindings
            .into_iter()
            .next()
            .expect("a single-partition admitted plan has exactly one binding");
        BoundPlanKind::ImplicitLocal {
            device: binding.device().clone(),
            virtual_partition: binding.virtual_partition().cloned(),
        }
    } else {
        BoundPlanKind::Distributed { bindings }
    };

    let mut plan = BoundDistributedPlan {
        logical_distributed_plan_hash: admitted.logical_distributed_plan_hash().to_owned(),
        kind,
        device_set,
        snapshot_id,
        fixture_identity_class,
        transport_class,
        bound_distributed_plan_hash: String::new(),
    };
    let bytes = plan.canonical_bytes();
    plan.bound_distributed_plan_hash =
        format!("sha256:{}", hex_lower(&crate::repack_plan::sha256(&bytes)));
    Ok(plan)
}

impl BoundDistributedPlan {
    /// The admitted logical plan's opaque hash (never re-derived here).
    #[must_use]
    pub fn logical_distributed_plan_hash(&self) -> &str {
        &self.logical_distributed_plan_hash
    }

    /// The bound plan shape — implicit/local (MD-A15) or distributed.
    #[must_use]
    pub fn kind(&self) -> &BoundPlanKind {
        &self.kind
    }

    /// True for the MD-A15 single-partition degenerate — no distributed
    /// wrapper.
    #[must_use]
    pub fn is_degenerate(&self) -> bool {
        matches!(self.kind, BoundPlanKind::ImplicitLocal { .. })
    }

    /// The bound device set (selected membership).
    #[must_use]
    pub fn device_set(&self) -> &DeviceSet {
        &self.device_set
    }

    /// The discovery snapshot the bind was frozen at (content-addressed).
    #[must_use]
    pub fn snapshot_id(&self) -> DeviceDiscoverySnapshotId {
        self.snapshot_id
    }

    /// `bound_distributed_plan_hash` — `sha256:` over the bound-plan
    /// canonical bytes (logical hash + binding facts; physical ids enter
    /// THIS hash domain, never A10).
    #[must_use]
    pub fn bound_distributed_plan_hash(&self) -> &str {
        &self.bound_distributed_plan_hash
    }

    /// The per-partition bindings of a distributed plan; `None` for the
    /// MD-A15 degenerate (which has no distributed wrapper).
    #[must_use]
    pub fn bindings(&self) -> Option<&BTreeMap<LogicalPartitionId, PartitionBinding>> {
        match &self.kind {
            BoundPlanKind::Distributed { bindings } => Some(bindings),
            BoundPlanKind::ImplicitLocal { .. } => None,
        }
    }

    /// The declared fixture identity class.
    #[must_use]
    pub const fn fixture_identity_class(&self) -> FixtureIdentityClass {
        self.fixture_identity_class
    }

    /// The declared transport evidence class.
    #[must_use]
    pub const fn transport_class(&self) -> TransportClass {
        self.transport_class
    }

    /// The receipt taxonomy (md0-closeout §3.2 #4): `physical_device_count`,
    /// `physical_device_ids`, `virtual_partition_count`,
    /// `virtual_partition_ids`, `fixture_identity_class`, `transport_class`,
    /// and `hardware_isolation_claimed=false` — virtual partitions never
    /// receive a [`PhysicalDeviceId`]. Counts derive from the id sets.
    #[must_use]
    pub fn receipt(&self) -> PartitionReceipt {
        match &self.kind {
            BoundPlanKind::ImplicitLocal {
                device,
                virtual_partition,
            } => {
                let physical = BTreeSet::from([device.clone()]);
                let virtual_ids: BTreeSet<_> = virtual_partition
                    .iter()
                    .map(super::partition::VirtualDevicePartition::id)
                    .collect();
                PartitionReceipt::new(
                    physical,
                    virtual_ids,
                    self.fixture_identity_class,
                    self.transport_class,
                )
            }
            BoundPlanKind::Distributed { bindings } => {
                let physical: BTreeSet<_> = bindings.values().map(|b| b.device().clone()).collect();
                let virtual_ids: BTreeSet<_> = bindings
                    .values()
                    .filter_map(|b| {
                        b.virtual_partition()
                            .map(super::partition::VirtualDevicePartition::id)
                    })
                    .collect();
                PartitionReceipt::new(
                    physical,
                    virtual_ids,
                    self.fixture_identity_class,
                    self.transport_class,
                )
            }
        }
    }

    /// Deterministic canonical bytes of the bound plan: the logical hash,
    /// the binding facts (partition identity → physical device + optional
    /// virtual partition id), the bound device set, the snapshot id, and the
    /// receipt classes. Identical inputs produce identical bytes; any binding
    /// fact change changes the bytes — and therefore the
    /// `bound_distributed_plan_hash`.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_str(&mut out, &self.logical_distributed_plan_hash);
        match &self.kind {
            BoundPlanKind::ImplicitLocal {
                device,
                virtual_partition,
            } => {
                out.push(0u8); // tag: implicit/local (MD-A15 degenerate)
                out.extend_from_slice(&device.canonical_bytes());
                match virtual_partition {
                    Some(p) => {
                        out.push(1u8);
                        push_u64(&mut out, p.id().get());
                    }
                    None => out.push(0u8),
                }
            }
            BoundPlanKind::Distributed { bindings } => {
                out.push(1u8); // tag: distributed
                push_u64(&mut out, bindings.len() as u64);
                for (partition, binding) in bindings {
                    push_str(&mut out, partition.as_str());
                    out.extend_from_slice(&binding.device().canonical_bytes());
                    match binding.virtual_partition() {
                        Some(p) => {
                            out.push(1u8);
                            push_u64(&mut out, p.id().get());
                        }
                        None => out.push(0u8),
                    }
                }
            }
        }
        push_u64(&mut out, self.device_set.len() as u64);
        for id in self.device_set.members() {
            out.extend_from_slice(&id.canonical_bytes());
        }
        out.extend_from_slice(self.snapshot_id.as_bytes());
        push_u64(&mut out, self.fixture_identity_class.tag());
        push_u64(&mut out, self.transport_class.tag());
        out
    }
}

/// Whether a hash string is the `sha256:` + 64-lowercase-hex spelling
/// (FC17/FC11). Uppercase hex is rejected — the contract digest is lowercase.
#[must_use]
fn is_sha256_hex(hash: &str) -> bool {
    let Some(digest) = hash.strip_prefix("sha256:") else {
        return false;
    };
    digest.len() == 64 && digest.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f'))
}

/// The first undeclared partition a constraint references, if any.
#[must_use]
fn constraint_unknown_partition(
    constraint: &DeclaredPlacementConstraint,
    declared: &BTreeSet<LogicalPartitionId>,
) -> Option<LogicalPartitionId> {
    match constraint {
        DeclaredPlacementConstraint::DistinctPhysicalDevices => None,
        DeclaredPlacementConstraint::Colocated { partitions }
        | DeclaredPlacementConstraint::RequiredBackend { partitions, .. } => {
            partitions.iter().find(|p| !declared.contains(*p)).cloned()
        }
    }
}

/// Fail-closed launch-resource checks against every bound device (DCG-4).
///
/// Undeclared fields (`None`) skip that check. A zero device ceiling or a
/// missing snapshot entry is unevaluable, never a fake limit of zero.
fn check_launch_resource_limits(
    demand: LaunchResourceDemand,
    bound_devices: &BTreeSet<PhysicalDeviceId>,
    snapshot: &DeviceDiscoverySnapshot,
) -> Result<(), BindError> {
    if demand.threads_per_workgroup.is_none() && demand.workgroup_shared_memory_bytes.is_none() {
        return Ok(());
    }
    for device in bound_devices {
        let Some(caps) = snapshot_caps(snapshot, device) else {
            return Err(launch_limit_error(
                device,
                format!("no capability facts recorded for {device} in the snapshot"),
            ));
        };
        if let Some(threads) = demand.threads_per_workgroup {
            let limit =
                evaluable_launch_limit(caps.max_threads_per_workgroup, "max_threads_per_workgroup")
                    .map_err(|detail| launch_limit_error(device, detail))?;
            if threads > limit {
                return Err(launch_limit_error(
                    device,
                    format!(
                        "threads per workgroup {threads} exceeds device max_threads_per_workgroup {limit}"
                    ),
                ));
            }
        }
        if let Some(bytes) = demand.workgroup_shared_memory_bytes {
            let limit = evaluable_launch_limit(
                caps.workgroup_shared_memory_max_bytes,
                "workgroup_shared_memory_max_bytes",
            )
            .map_err(|detail| launch_limit_error(device, detail))?;
            if bytes > limit {
                return Err(launch_limit_error(
                    device,
                    format!(
                        "workgroup shared memory {bytes} bytes exceeds device max {limit} bytes"
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn snapshot_caps<'a>(
    snapshot: &'a DeviceDiscoverySnapshot,
    device: &PhysicalDeviceId,
) -> Option<&'a DeviceCapabilities> {
    snapshot
        .devices()
        .values()
        .find(|entry| &entry.identity == device)
        .map(|entry| &entry.capabilities)
}

fn evaluable_launch_limit(value: u32, field: &'static str) -> Result<u32, String> {
    if value == 0 {
        Err(format!("{field} is unevaluable (device reports 0)"))
    } else {
        Ok(value)
    }
}

fn launch_limit_error(device: &PhysicalDeviceId, detail: String) -> BindError {
    BindError::LaunchResourceLimit {
        device: device.clone(),
        detail,
    }
}

/// Evaluate one declared placement constraint against the bindings.
///
/// Every `pid` in `constraint` is a declared partition, and [`bind`] has
/// already enforced that the binding keys are exactly the declared
/// partitions — so the lookup cannot miss.
fn check_constraint(
    constraint: &DeclaredPlacementConstraint,
    bindings: &BTreeMap<LogicalPartitionId, PartitionBinding>,
) -> Result<(), String> {
    match constraint {
        DeclaredPlacementConstraint::DistinctPhysicalDevices => {
            let distinct: BTreeSet<&PhysicalDeviceId> =
                bindings.values().map(PartitionBinding::device).collect();
            if distinct.len() != bindings.len() {
                return Err(format!(
                    "{} partition(s) must each bind a distinct physical device; {} distinct device(s) referenced",
                    bindings.len(),
                    distinct.len()
                ));
            }
        }
        DeclaredPlacementConstraint::Colocated { partitions } => {
            let mut devices = BTreeSet::new();
            for partition in partitions {
                let binding = bindings
                    .get(partition)
                    .expect("admit validated constraint partitions are declared; bind enforced binding keys == declared partitions");
                devices.insert(binding.device());
            }
            if devices.len() > 1 {
                return Err(format!(
                    "colocated partitions must bind exactly one physical device; found {}",
                    devices.len()
                ));
            }
        }
        DeclaredPlacementConstraint::RequiredBackend {
            partitions,
            backend,
        } => {
            for partition in partitions {
                let binding = bindings
                    .get(partition)
                    .expect("admit validated constraint partitions are declared; bind enforced binding keys == declared partitions");
                if binding.device().backend() != *backend {
                    return Err(format!(
                        "partition {partition} bound to a {} device; declared backend is {backend}",
                        binding.device().backend()
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Lowercase hex of a byte slice.
#[must_use]
fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        write!(out, "{byte:02x}").expect("writing to a String cannot fail");
    }
    out
}

#[cfg(test)]
#[path = "bound_plan_test.rs"]
mod tests;
