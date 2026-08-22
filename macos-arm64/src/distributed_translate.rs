//! FMIR distributed-section → host-coordinator transaction-mirror translation
//! (MD3H-H4).
//!
//! Ingests an admitted FMIR device-section postcard (MD3H-F1 built artifacts)
//! and produces [`TransactionOperation`] / [`TransactionCommitBoundary`]
//! mirrors with deterministic canonical bytes. Bind policy maps the translated
//! virtual partitions onto a discovery snapshot. The `device-execute` CLI
//! carries the image plus a declared bind count (OQ-5).
//!
//! ## OQ-2 — translation dependency route
//!
//! **Decision:** `macos-arm64` depends on `radix-mir-fmir` (the FMIR format
//! crate) and decodes F1 postcard artifacts with that schema. A hand-rolled
//! decoder would duplicate the positional postcard layout of
//! `FmirDeviceSection` and drift from F1.
//!
//! **Rationale:** this binary already owns the wire/descriptor seam;
//! `host-coordinator` stays serde-free and radix-mir-free (FC10). The format
//! crate is the image codec, not a runner.

use std::collections::BTreeMap;
use std::fmt;

use host_coordinator::bound_plan::{
    bind, AdmittedLogicalPlan, BindError, BoundDistributedPlan, DeclaredPlacementConstraint,
    LogicalPartitionId, PartitionBinding,
};
use host_coordinator::device_identity::{DeviceHealthGeneration, PhysicalDeviceId};
use host_coordinator::device_set::DeviceSet;
use host_coordinator::discovery::DeviceDiscoverySnapshot;
use host_coordinator::execution_transaction::{
    BarrierRef, CollectiveBroadcastMirror, CollectiveRef, LaunchRef, MirroredDtype,
    MirroredStorageLayout, TransactionCommitBoundary, TransactionOperation,
    TransferDirectionMirror, TransferOperationMirror, TransferRef, TransportPathMirror,
};
use host_coordinator::partition::{
    AdmissionRequest, FixtureIdentityClass, PartitionBudgetLedger, SafePhysicalLimit,
    TransportClass, VirtualDevicePartition, VirtualDevicePartitionId,
};
use radix_mir_fmir::schema::{
    FmirDeviceSection, WireCollectiveKind, WireExecutionCommitBoundary, WireExecutionOperation,
    WirePlacementConstraint, WireStorageLayout, WireTransferDirection,
    WIRE_DISTRIBUTED_SECTION_VERSION,
};
use radix_mir_fmir::{admit_device_section_with_compat, default_compatibility_table};

/// CS-1 per-partition budget used when the section omits a partition-budget
/// constraint (matches MD3H-F1's 160 MiB).
const DEFAULT_PARTITION_BUDGET_BYTES: u64 = 160 * 1024 * 1024;

/// Why ingest or translation failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranslateError {
    /// Postcard bytes were not a `FmirDeviceSection`.
    Decode(String),
    /// The device section carries no distributed execution section.
    MissingDistributedSection,
    /// The section version ratchet rejected the image.
    VersionMismatch {
        /// Version encoded on the section.
        actual: u32,
        /// Version this translator admits.
        expected: u32,
    },
    /// FMIR admission rejected the device section.
    Admit(String),
    /// A wire fact is outside the translator's closed vocabulary.
    Unsupported(String),
}

impl fmt::Display for TranslateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(detail) => write!(f, "distributed image decode failed: {detail}"),
            Self::MissingDistributedSection => {
                write!(f, "device section carries no distributed execution section")
            }
            Self::VersionMismatch { actual, expected } => {
                write!(f, "distributed section version {actual} is not {expected}")
            }
            Self::Admit(detail) => write!(f, "distributed section admission failed: {detail}"),
            Self::Unsupported(detail) => write!(f, "distributed section unsupported: {detail}"),
        }
    }
}

/// How virtual partitions bind onto a discovery snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindPolicy {
    /// Every logical partition colocates on the snapshot's physical membership
    /// (8 virtual → 1 physical when the snapshot has one device).
    ColocateOnSnapshot,
    /// Claim one distinct physical device per logical partition (8:8
    /// promotion). A 1-physical snapshot must reject this as topology
    /// mismatch.
    OnePhysicalPerPartition,
    /// Split the partitions in contiguous plan-order halves across exactly
    /// `bind_count` physical devices (8:2 → 4+4; OQ-3). Legal when
    /// `1 < bind_count < partition_count` and
    /// `partition_count % bind_count == 0`. The snapshot's physical
    /// membership must be exactly `bind_count` devices or the bind rejects
    /// [`BindError::TopologyMismatch`].
    SplitAcrossMembership {
        /// The number of physical devices the split claims.
        bind_count: u32,
    },
}

/// Map the CLI declared bind count onto [`BindPolicy`].
///
/// `--bind-count 1` colocates every virtual partition on the snapshot
/// (8:1). `--bind-count` equal to the partition count claims one physical
/// per partition (8:8). Any other divisor of the partition count splits
/// contiguous plan-order halves across that many physicals (8:2 → 4+4,
/// 8:4 → 2+2+2+2; OQ-3). Any other count is unsupported at this seam
/// (OQ-5 — richer bind negotiation waits for MD5).
///
/// # Errors
///
/// Returns [`TranslateError::Unsupported`] when `bind_count` is 0 or is
/// neither 1, the partition count, nor a divisor of the partition count.
pub fn bind_policy_for_declared_count(
    partition_count: usize,
    bind_count: u32,
) -> Result<BindPolicy, TranslateError> {
    if bind_count == 0 {
        return Err(TranslateError::Unsupported(
            "declared bind count 0 is not a legal bind".to_owned(),
        ));
    }
    if bind_count == 1 {
        return Ok(BindPolicy::ColocateOnSnapshot);
    }
    let bind = bind_count as usize;
    if bind == partition_count {
        return Ok(BindPolicy::OnePhysicalPerPartition);
    }
    if bind < partition_count && partition_count.is_multiple_of(bind) {
        return Ok(BindPolicy::SplitAcrossMembership { bind_count });
    }
    Err(TranslateError::Unsupported(format!(
        "declared bind count {bind_count} for {partition_count} partitions is not 1 (colocate), {partition_count} (one physical per partition), or a divisor (split); richer bind negotiation waits for MD5"
    )))
}

/// An admitted FMIR distributed section translated into the transaction
/// mirror vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranslatedDistributedPlan {
    logical_distributed_plan_hash: String,
    partitions: Vec<LogicalPartitionId>,
    operations: Vec<TransactionOperation>,
    commit_boundary: TransactionCommitBoundary,
    partition_budget_bytes: BTreeMap<LogicalPartitionId, u64>,
}

impl TranslatedDistributedPlan {
    /// The admitted `logical_distributed_plan_hash`.
    #[must_use]
    pub fn logical_distributed_plan_hash(&self) -> &str {
        &self.logical_distributed_plan_hash
    }

    /// Logical partitions in plan order.
    #[must_use]
    pub fn partitions(&self) -> &[LogicalPartitionId] {
        &self.partitions
    }

    /// Mirror operations in plan order.
    #[must_use]
    pub fn operations(&self) -> &[TransactionOperation] {
        &self.operations
    }

    /// Declared commit boundary.
    #[must_use]
    pub fn commit_boundary(&self) -> &TransactionCommitBoundary {
        &self.commit_boundary
    }

    /// Deterministic concatenation of every operation's canonical bytes plus
    /// the commit-boundary bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for operation in &self.operations {
            out.extend_from_slice(&operation.canonical_bytes());
        }
        out.extend_from_slice(&self.commit_boundary.canonical_bytes());
        out
    }

    /// Communication-graph edge count: transfers, collectives, and barriers.
    /// Launches are not edges. N=1 is zero.
    #[must_use]
    pub fn communication_graph_edge_count(&self) -> u64 {
        self.operations
            .iter()
            .filter(|operation| !matches!(operation, TransactionOperation::Launch { .. }))
            .count() as u64
    }
}

/// Decode an F1 `FmirDeviceSection` postcard, admit it, and translate the
/// distributed section into the transaction mirror.
///
/// # Errors
///
/// Returns [`TranslateError`] when the bytes do not decode, admission fails,
/// the distributed section is missing or version-mismatched, or a wire fact
/// is outside the closed vocabulary.
pub fn translate_device_section_bytes(
    bytes: &[u8],
) -> Result<TranslatedDistributedPlan, TranslateError> {
    let device: FmirDeviceSection =
        postcard::from_bytes(bytes).map_err(|error| TranslateError::Decode(error.to_string()))?;
    admit_device_section_with_compat(&device, &default_compatibility_table())
        .map_err(|error| TranslateError::Admit(error.to_string()))?;
    let section = device
        .distributed
        .as_ref()
        .ok_or(TranslateError::MissingDistributedSection)?;
    if section.v != WIRE_DISTRIBUTED_SECTION_VERSION {
        return Err(TranslateError::VersionMismatch {
            actual: section.v,
            expected: WIRE_DISTRIBUTED_SECTION_VERSION,
        });
    }
    translate_admitted_section(section)
}

fn translate_admitted_section(
    section: &radix_mir_fmir::schema::FmirDistributedExecutionSection,
) -> Result<TranslatedDistributedPlan, TranslateError> {
    let partitions: Vec<LogicalPartitionId> = section
        .logical_plan
        .partitions
        .iter()
        .map(|partition| logical_partition(partition.id))
        .collect();
    let mut buffer_bytes = BTreeMap::new();
    for fact in &section.declared_placement {
        buffer_bytes.insert(fact.buffer, fact.byte_count);
    }
    let mut launch_output_bytes = BTreeMap::new();
    for partition in &section.logical_plan.partitions {
        for result in &partition.results {
            let bytes = buffer_bytes.get(&result.buffer).copied().unwrap_or(0);
            launch_output_bytes.insert(result.produced_by, bytes);
        }
    }
    let mut operations = Vec::with_capacity(section.logical_plan.graph.operations.len());
    for operation in &section.logical_plan.graph.operations {
        operations.push(translate_operation(operation, &launch_output_bytes)?);
    }
    let mut partition_budget_bytes = BTreeMap::new();
    for partition in &partitions {
        partition_budget_bytes.insert(partition.clone(), DEFAULT_PARTITION_BUDGET_BYTES);
    }
    for constraint in &section.logical_plan.declared_placement.constraints {
        if let WirePlacementConstraint::PartitionBudgetBytes {
            partition,
            budget_bytes,
        } = constraint
        {
            partition_budget_bytes.insert(logical_partition(*partition), *budget_bytes);
        }
    }
    Ok(TranslatedDistributedPlan {
        logical_distributed_plan_hash: section.logical_distributed_plan_hash.clone(),
        partitions,
        operations,
        commit_boundary: translate_commit_boundary(&section.logical_plan.graph.commit_boundary),
        partition_budget_bytes,
    })
}

fn translate_operation(
    operation: &WireExecutionOperation,
    launch_output_bytes: &BTreeMap<u32, u64>,
) -> Result<TransactionOperation, TranslateError> {
    match operation {
        WireExecutionOperation::Launch(launch) => {
            let output_bytes = launch_output_bytes
                .get(&launch.launch)
                .copied()
                .unwrap_or(0);
            Ok(TransactionOperation::launch(
                logical_partition(launch.partition),
                launch_ref(launch.launch),
                output_bytes,
            ))
        }
        WireExecutionOperation::Transfer(transfer) => Ok(TransactionOperation::transfer(
            TransferOperationMirror::new(
                TransferRef::new(format!("transfer-{}", transfer.id)),
                logical_partition(transfer.source),
                logical_partition(transfer.destination),
                transfer.byte_count,
                translate_direction(transfer.direction),
                translate_dtype(&transfer.element_ty)?,
                translate_layout(transfer.layout),
                TransportPathMirror::HostStaged,
                u64::from(transfer.producer_generation),
                u64::from(transfer.consumer_generation),
                translate_commit_boundary(&transfer.completion_boundary),
            ),
        )),
        WireExecutionOperation::Collective(collective) => match collective.kind {
            WireCollectiveKind::Broadcast => {
                let participants = collective
                    .participants
                    .iter()
                    .copied()
                    .map(logical_partition)
                    .collect();
                Ok(TransactionOperation::broadcast(
                    CollectiveBroadcastMirror::broadcast(
                        CollectiveRef::new(format!("collective-{}", collective.id)),
                        logical_partition(collective.source),
                        participants,
                        collective.byte_count,
                    ),
                ))
            }
        },
        WireExecutionOperation::Barrier(barrier) => {
            let partitions = barrier
                .partitions
                .iter()
                .copied()
                .map(logical_partition)
                .collect();
            Ok(TransactionOperation::barrier(
                barrier_ref(barrier.id),
                partitions,
            ))
        }
    }
}

fn translate_commit_boundary(boundary: &WireExecutionCommitBoundary) -> TransactionCommitBoundary {
    TransactionCommitBoundary::new(
        boundary.barriers.iter().copied().map(barrier_ref),
        boundary.launches.iter().copied().map(launch_ref),
    )
}

fn translate_direction(direction: WireTransferDirection) -> TransferDirectionMirror {
    match direction {
        WireTransferDirection::H2D => TransferDirectionMirror::H2D,
        WireTransferDirection::D2H => TransferDirectionMirror::D2H,
        WireTransferDirection::BIDI => TransferDirectionMirror::BIDI,
    }
}

fn translate_dtype(spelling: &str) -> Result<MirroredDtype, TranslateError> {
    match spelling {
        "f32" => Ok(MirroredDtype::F32),
        "f64" => Ok(MirroredDtype::F64),
        "f16" => Ok(MirroredDtype::F16),
        "bf16" => Ok(MirroredDtype::BF16),
        "i8" => Ok(MirroredDtype::I8),
        "i32" => Ok(MirroredDtype::I32),
        other => Err(TranslateError::Unsupported(format!(
            "element type `{other}` is outside the mirrored dtype surface"
        ))),
    }
}

fn translate_layout(_layout: WireStorageLayout) -> MirroredStorageLayout {
    // Wire storage (host-owned vs device-handle) is not the tensor layout
    // the transfer mirror consumes. F1 fixtures are dense element storage.
    MirroredStorageLayout::Dense
}

fn logical_partition(id: u32) -> LogicalPartitionId {
    LogicalPartitionId::new(format!("partition-{id}"))
}

fn launch_ref(id: u32) -> LaunchRef {
    LaunchRef::new(format!("launch-{id}"))
}

fn barrier_ref(id: u32) -> BarrierRef {
    BarrierRef::new(format!("barrier-{id}"))
}

/// Bind a translated plan onto a discovery snapshot under [`BindPolicy`].
///
/// # Errors
///
/// Returns [`BindError`] when admission, membership, or topology fails.
/// [`BindPolicy::OnePhysicalPerPartition`] on a snapshot whose physical
/// membership is not exactly the partition count rejects
/// [`BindError::TopologyMismatch`]. [`BindPolicy::SplitAcrossMembership`]
/// on a snapshot whose physical membership is not exactly `bind_count`
/// devices rejects [`BindError::TopologyMismatch`].
pub fn bind_translated(
    plan: &TranslatedDistributedPlan,
    snapshot: &DeviceDiscoverySnapshot,
    policy: BindPolicy,
) -> Result<BoundDistributedPlan, BindError> {
    bind_translated_with_constraints(plan, snapshot, policy, &[])
}

/// Bind a translated plan under [`BindPolicy`] with declared placement
/// constraints enforced per binding.
///
/// The mirror constraint vocabulary is [`DeclaredPlacementConstraint`]
/// (host-coordinator, read-only). A constraint violated by the bindings —
/// e.g. a [`DeclaredPlacementConstraint::Colocated`] set spanning the split
/// boundary — rejects [`BindError::ConstraintViolation`].
pub fn bind_translated_with_constraints(
    plan: &TranslatedDistributedPlan,
    snapshot: &DeviceDiscoverySnapshot,
    policy: BindPolicy,
    declared_constraints: &[DeclaredPlacementConstraint],
) -> Result<BoundDistributedPlan, BindError> {
    match policy {
        BindPolicy::ColocateOnSnapshot => bind_colocate(plan, snapshot, declared_constraints),
        BindPolicy::OnePhysicalPerPartition => {
            bind_one_physical_per_partition(plan, snapshot, declared_constraints)
        }
        BindPolicy::SplitAcrossMembership { bind_count } => {
            bind_split_across_membership(plan, snapshot, bind_count, declared_constraints)
        }
    }
}

fn snapshot_physical_ids(snapshot: &DeviceDiscoverySnapshot) -> Vec<PhysicalDeviceId> {
    snapshot
        .devices()
        .values()
        .map(|entry| entry.identity.clone())
        .collect()
}

fn partition_budget(plan: &TranslatedDistributedPlan, partition: &LogicalPartitionId) -> u64 {
    plan.partition_budget_bytes
        .get(partition)
        .copied()
        .unwrap_or(DEFAULT_PARTITION_BUDGET_BYTES)
}

fn bind_colocate(
    plan: &TranslatedDistributedPlan,
    snapshot: &DeviceDiscoverySnapshot,
    constraints: &[DeclaredPlacementConstraint],
) -> Result<BoundDistributedPlan, BindError> {
    let device = match snapshot_physical_ids(snapshot).into_iter().next() {
        Some(device) => device,
        None => {
            return Err(BindError::TopologyMismatch {
                detail: "discovery snapshot has no physical devices".to_owned(),
            });
        }
    };
    let mut bindings = BTreeMap::new();
    for (index, partition) in plan.partitions.iter().enumerate() {
        let vp = admit_virtual(
            index as u64 + 1,
            device.clone(),
            partition_budget(plan, partition),
        )?;
        bindings.insert(
            partition.clone(),
            PartitionBinding::with_virtual_partition(device.clone(), vp),
        );
    }
    finish_bind(
        plan,
        bindings,
        DeviceSet::from_members([device]),
        snapshot,
        constraints,
    )
}

fn bind_one_physical_per_partition(
    plan: &TranslatedDistributedPlan,
    snapshot: &DeviceDiscoverySnapshot,
    constraints: &[DeclaredPlacementConstraint],
) -> Result<BoundDistributedPlan, BindError> {
    // Device set is the snapshot's real membership. Bindings claim one
    // distinct physical id per logical partition — fabricated ids fill
    // ranks the snapshot does not have. bind() then TopologyMismatch
    // when the claimed set is not exactly the snapshot membership
    // (8:1 promoted as 8 physical).
    let snapshot_devices = snapshot_physical_ids(snapshot);
    if snapshot_devices.is_empty() {
        return Err(BindError::TopologyMismatch {
            detail: "discovery snapshot has no physical devices".to_owned(),
        });
    }
    let mut bindings = BTreeMap::new();
    for (index, partition) in plan.partitions.iter().enumerate() {
        let device = match snapshot_devices.get(index) {
            Some(device) => device.clone(),
            None => PhysicalDeviceId::cuda(format!("GPU-promoted-{index}"), None),
        };
        let vp = admit_virtual(
            index as u64 + 1,
            device.clone(),
            partition_budget(plan, partition),
        )?;
        bindings.insert(
            partition.clone(),
            PartitionBinding::with_virtual_partition(device, vp),
        );
    }
    finish_bind(
        plan,
        bindings,
        DeviceSet::from_members(snapshot_devices),
        snapshot,
        constraints,
    )
}

/// Bind under [`BindPolicy::SplitAcrossMembership`]: `bind_count` physical
/// devices claim `partition_count / bind_count` contiguous partitions each,
/// in plan order (OQ-3 — partitions 1–4 → first physical, 5–8 → second).
///
/// The snapshot's physical membership must be exactly `bind_count` devices
/// or the bind rejects [`BindError::TopologyMismatch`] — fail-closed in
/// both directions (fewer and more). Declared placement constraints are
/// enforced per binding by [`bind`].
fn bind_split_across_membership(
    plan: &TranslatedDistributedPlan,
    snapshot: &DeviceDiscoverySnapshot,
    bind_count: u32,
    constraints: &[DeclaredPlacementConstraint],
) -> Result<BoundDistributedPlan, BindError> {
    let members = bind_count as usize;
    let partition_count = plan.partitions.len();
    if members == 0 || members >= partition_count || !partition_count.is_multiple_of(members) {
        // Fail-closed even when the policy was constructed directly,
        // bypassing [`bind_policy_for_declared_count`]'s divisor check.
        return Err(BindError::TopologyMismatch {
            detail: format!(
                "split bind count {bind_count} is not a legal split of {partition_count} partitions (1 < m < {partition_count}, {partition_count} % m == 0)"
            ),
        });
    }
    let snapshot_devices = snapshot_physical_ids(snapshot);
    if snapshot_devices.len() != members {
        return Err(BindError::TopologyMismatch {
            detail: format!(
                "split bind count {bind_count} requires exactly {bind_count} physical device(s); snapshot has {}",
                snapshot_devices.len()
            ),
        });
    }
    let chunk = partition_count / members;
    let mut bindings = BTreeMap::new();
    for (index, partition) in plan.partitions.iter().enumerate() {
        let device = snapshot_devices[index / chunk].clone();
        let vp = admit_virtual(
            index as u64 + 1,
            device.clone(),
            partition_budget(plan, partition),
        )?;
        bindings.insert(
            partition.clone(),
            PartitionBinding::with_virtual_partition(device, vp),
        );
    }
    finish_bind(
        plan,
        bindings,
        DeviceSet::from_members(snapshot_devices),
        snapshot,
        constraints,
    )
}

fn admit_virtual(
    seed: u64,
    device: PhysicalDeviceId,
    budget_bytes: u64,
) -> Result<VirtualDevicePartition, BindError> {
    let ledger = partition_ledger(budget_bytes);
    VirtualDevicePartition::admit(
        AdmissionRequest::new(VirtualDevicePartitionId::new(seed), device.clone(), ledger),
        SafePhysicalLimit::new(budget_bytes),
    )
    .map_err(|_| BindError::InvalidPartitionBinding {
        partition: LogicalPartitionId::new(format!("vp:{seed}")),
        detail: format!("virtual partition vp:{seed} exceeded safe physical limit {budget_bytes}"),
    })
}

fn partition_ledger(budget_bytes: u64) -> PartitionBudgetLedger {
    let quarter = budget_bytes / 4;
    PartitionBudgetLedger {
        weight_bytes: 0,
        kv_cache_bytes: 0,
        activation_scratch_bytes: quarter,
        module_storage_bytes: 0,
        allocator_overhead_bytes: 0,
        transfer_staging_bytes: quarter,
        concurrent_state_bytes: 0,
    }
}

fn finish_bind(
    plan: &TranslatedDistributedPlan,
    bindings: BTreeMap<LogicalPartitionId, PartitionBinding>,
    device_set: DeviceSet,
    snapshot: &DeviceDiscoverySnapshot,
    constraints: &[DeclaredPlacementConstraint],
) -> Result<BoundDistributedPlan, BindError> {
    let admitted = AdmittedLogicalPlan::admit(
        plan.logical_distributed_plan_hash.clone(),
        plan.partitions.iter().cloned(),
        constraints.iter().cloned(),
    )
    .map_err(|error| BindError::TopologyMismatch {
        detail: format!("translated plan failed logical admission: {error:?}"),
    })?;
    bind(
        &admitted,
        bindings,
        device_set,
        snapshot,
        DeviceHealthGeneration::initial(),
        FixtureIdentityClass::Virtual,
        TransportClass::HostStaged,
    )
}
