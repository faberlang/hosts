//! Mirror vocabulary (FC8): the dependency-free, stable-canonical mirror of
//! radix-mir's execution-graph operations and references that the transaction
//! consumes as its accepted plan (faber-runtime cannot import radix-mir,
//! FC18). Section split out of `execution_transaction.rs` (polish).

use crate::bound_plan::LogicalPartitionId;
use crate::device_identity::{push_str, push_u64};
use std::collections::BTreeSet;

/// Stable opaque reference to one semantic launch (mirror of radix-mir
/// `LaunchId`; faber-runtime cannot import radix-mir, FC18).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LaunchRef(String);

impl LaunchRef {
    /// Build a launch reference from its stable string.
    #[must_use]
    pub fn new(reference: impl Into<String>) -> Self {
        Self(reference.into())
    }

    /// The stable reference string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for LaunchRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Stable opaque identity of one transfer in the execution graph (mirror of
/// radix-mir `TransferId`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TransferRef(String);

impl TransferRef {
    /// Build a transfer reference from its stable string.
    #[must_use]
    pub fn new(reference: impl Into<String>) -> Self {
        Self(reference.into())
    }

    /// The stable reference string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TransferRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Stable opaque identity of one collective in the execution graph (mirror of
/// radix-mir `CollectiveId`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CollectiveRef(String);

impl CollectiveRef {
    /// Build a collective reference from its stable string.
    #[must_use]
    pub fn new(reference: impl Into<String>) -> Self {
        Self(reference.into())
    }

    /// The stable reference string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for CollectiveRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Stable opaque identity of one barrier in the execution graph (mirror of
/// radix-mir `BarrierId`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BarrierRef(String);

impl BarrierRef {
    /// Build a barrier reference from its stable string.
    #[must_use]
    pub fn new(reference: impl Into<String>) -> Self {
        Self(reference.into())
    }

    /// The stable reference string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for BarrierRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Stable opaque identity of one staged output buffer the transaction will
/// publish. A write is identified by the operation that produces it, so an
/// `OutputRef` is unique within a snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OutputRef(String);

impl OutputRef {
    /// Build an output reference from its stable string.
    #[must_use]
    pub fn new(reference: impl Into<String>) -> Self {
        Self(reference.into())
    }

    /// The stable reference string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for OutputRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Direction of a host-staged transfer (T1 §2.2 measurement vocabulary).
///
/// `H2D` / `D2H` are the host-boundary half-moves; `BIDI` is the combined
/// device-to-device host-staged copy (concurrent H2D ∥ D2H — the way every
/// cross-partition move on the acceptance host traverses the admitted path).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferDirectionMirror {
    /// Host → device (copy in to the destination partition).
    H2D,
    /// Device → host (copy out from the source partition).
    D2H,
    /// Both directions concurrently (the host-staged device-to-device move).
    BIDI,
}

impl TransferDirectionMirror {
    /// Deterministic serialization tag.
    #[must_use]
    pub const fn tag(self) -> u64 {
        match self {
            Self::H2D => 0,
            Self::D2H => 1,
            Self::BIDI => 2,
        }
    }

    /// Short diagnostic spelling.
    #[must_use]
    pub fn spelling(self) -> &'static str {
        match self {
            Self::H2D => "h2d",
            Self::D2H => "d2h",
            Self::BIDI => "bidi",
        }
    }
}

/// Mirror of the element type of a transferred value — a **stable canonical
/// form** (FC6/FC18). The mapping from the logical `MirType` (radix-mir,
/// opaque) into this closed set is the translator's obligation at MD3-S1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MirroredDtype {
    /// 32-bit float.
    F32,
    /// 64-bit float.
    F64,
    /// 16-bit float.
    F16,
    /// bfloat16.
    BF16,
    /// Signed 8-bit integer.
    I8,
    /// Signed 32-bit integer.
    I32,
}

impl MirroredDtype {
    /// Deterministic serialization tag.
    #[must_use]
    pub const fn tag(self) -> u64 {
        match self {
            Self::F32 => 0,
            Self::F64 => 1,
            Self::F16 => 2,
            Self::BF16 => 3,
            Self::I8 => 4,
            Self::I32 => 5,
        }
    }

    /// Short diagnostic spelling.
    #[must_use]
    pub fn spelling(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::I8 => "i8",
            Self::I32 => "i32",
        }
    }
}

/// Mirror of the storage layout of a transferred value — a stable canonical
/// form (the logical layout vocabulary is radix-mir's; this is the
/// dependency-free mirror the typed/ranged transfer checks consume).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MirroredStorageLayout {
    /// Dense row-major element storage.
    Dense,
    /// Block-packed storage (quantized block layout).
    BlockPacked,
}

impl MirroredStorageLayout {
    /// Deterministic serialization tag.
    #[must_use]
    pub const fn tag(self) -> u64 {
        match self {
            Self::Dense => 0,
            Self::BlockPacked => 1,
        }
    }

    /// Short diagnostic spelling.
    #[must_use]
    pub fn spelling(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::BlockPacked => "block-packed",
        }
    }
}

/// Transport path label of a transfer (T2 §7 — silent host staging
/// forbidden).
///
/// `host-staged` is the only admitted path on the acceptance host. The label
/// is a transport-**admissibility** constraint on the logical plan; the
/// **selected** transport records to the transaction receipt, never to the
/// portable logical plan (S4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransportPathMirror {
    /// Pinned host memory ↔ device over PCIe — the admitted path.
    HostStaged,
}

impl TransportPathMirror {
    /// Deterministic serialization tag.
    #[must_use]
    pub const fn tag(self) -> u64 {
        match self {
            Self::HostStaged => 0,
        }
    }

    /// Short diagnostic spelling.
    #[must_use]
    pub fn spelling(self) -> &'static str {
        match self {
            Self::HostStaged => "host-staged",
        }
    }
}

/// Mirror of `ExecutionCommitBoundary` (naming contract §3) — the declared
/// barrier/launch completion set that commits a value or a plan. **Never
/// called "execution generation"** (`ValueGeneration` is taken).
/// [`ExecutionTransaction::commit`](crate::execution_transaction::ExecutionTransaction::commit)
/// publishes the staged write-set only after every boundary barrier/launch
/// completed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TransactionCommitBoundary {
    barriers: BTreeSet<BarrierRef>,
    launches: BTreeSet<LaunchRef>,
}

impl TransactionCommitBoundary {
    /// Build a boundary from its barrier and launch reference sets.
    #[must_use]
    pub fn new(
        barriers: impl IntoIterator<Item = BarrierRef>,
        launches: impl IntoIterator<Item = LaunchRef>,
    ) -> Self {
        Self {
            barriers: barriers.into_iter().collect(),
            launches: launches.into_iter().collect(),
        }
    }

    /// The declared barrier references, in stable order.
    #[must_use]
    pub fn barriers(&self) -> &BTreeSet<BarrierRef> {
        &self.barriers
    }

    /// The declared launch references, in stable order.
    #[must_use]
    pub fn launches(&self) -> &BTreeSet<LaunchRef> {
        &self.launches
    }

    /// True when the boundary declares nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.barriers.is_empty() && self.launches.is_empty()
    }

    /// Deterministic canonical bytes: barrier references then launch
    /// references, each set in stable order.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_u64(&mut out, self.barriers.len() as u64);
        for barrier in &self.barriers {
            push_str(&mut out, barrier.as_str());
        }
        push_u64(&mut out, self.launches.len() as u64);
        for launch in &self.launches {
            push_str(&mut out, launch.as_str());
        }
        out
    }
}

/// Mirror of radix-mir `TransferOperation`: one typed/ranged host-staged
/// cross-partition move, identified by byte count + logical
/// dtype/layout/generation, with the declared path label and per-transfer
/// completion boundary. The typed/ranged *validation before copy* is the
/// transport adapter's obligation (MD3-T1); this mirror carries the declared
/// facts in stable canonical form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferOperationMirror {
    id: TransferRef,
    source: LogicalPartitionId,
    destination: LogicalPartitionId,
    byte_count: u64,
    direction: TransferDirectionMirror,
    element_dtype: MirroredDtype,
    layout: MirroredStorageLayout,
    path_label: TransportPathMirror,
    producer_generation: u64,
    consumer_generation: u64,
    completion_boundary: TransactionCommitBoundary,
}

impl TransferOperationMirror {
    /// Build a transfer mirror. `producer_generation` / `consumer_generation`
    /// are the content versions the producer wrote and the consumer reads —
    /// transfer-operation facts, never the semantic `ValueGeneration`.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        id: TransferRef,
        source: LogicalPartitionId,
        destination: LogicalPartitionId,
        byte_count: u64,
        direction: TransferDirectionMirror,
        element_dtype: MirroredDtype,
        layout: MirroredStorageLayout,
        path_label: TransportPathMirror,
        producer_generation: u64,
        consumer_generation: u64,
        completion_boundary: TransactionCommitBoundary,
    ) -> Self {
        Self {
            id,
            source,
            destination,
            byte_count,
            direction,
            element_dtype,
            layout,
            path_label,
            producer_generation,
            consumer_generation,
            completion_boundary,
        }
    }

    /// The stable transfer identity.
    #[must_use]
    pub fn id(&self) -> &TransferRef {
        &self.id
    }

    /// The source partition (the value's owner for a read move).
    #[must_use]
    pub fn source(&self) -> &LogicalPartitionId {
        &self.source
    }

    /// The destination partition.
    #[must_use]
    pub fn destination(&self) -> &LogicalPartitionId {
        &self.destination
    }

    /// Byte count of the transferred value.
    #[must_use]
    pub const fn byte_count(&self) -> u64 {
        self.byte_count
    }

    /// The declared direction.
    #[must_use]
    pub const fn direction(&self) -> TransferDirectionMirror {
        self.direction
    }

    /// The declared element type.
    #[must_use]
    pub const fn element_dtype(&self) -> MirroredDtype {
        self.element_dtype
    }

    /// The declared storage layout.
    #[must_use]
    pub const fn layout(&self) -> MirroredStorageLayout {
        self.layout
    }

    /// The declared transport path label (admissibility constraint, v1 =
    /// `{host-staged}`).
    #[must_use]
    pub const fn path_label(&self) -> TransportPathMirror {
        self.path_label
    }

    /// The producer content version.
    #[must_use]
    pub const fn producer_generation(&self) -> u64 {
        self.producer_generation
    }

    /// The consumer content version.
    #[must_use]
    pub const fn consumer_generation(&self) -> u64 {
        self.consumer_generation
    }

    /// The per-transfer completion boundary (mirror fact; publication gates
    /// on the plan-level boundary).
    #[must_use]
    pub fn completion_boundary(&self) -> &TransactionCommitBoundary {
        &self.completion_boundary
    }

    /// Deterministic canonical bytes of every declared field.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_str(&mut out, self.id.as_str());
        push_str(&mut out, self.source.as_str());
        push_str(&mut out, self.destination.as_str());
        push_u64(&mut out, self.byte_count);
        push_u64(&mut out, self.direction.tag());
        push_u64(&mut out, self.element_dtype.tag());
        push_u64(&mut out, self.layout.tag());
        push_u64(&mut out, self.path_label.tag());
        push_u64(&mut out, self.producer_generation);
        push_u64(&mut out, self.consumer_generation);
        out.extend_from_slice(&self.completion_boundary.canonical_bytes());
        out
    }
}

/// Mirror of a `Collective::Broadcast` (the only admitted collective, v1).
/// Broadcast is composed from labeled host-staged transfers + local kernels —
/// no collective library (md0-transport §8). The mirror carries the source,
/// the participant set (source plus every consumer), and the value's byte
/// contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectiveBroadcastMirror {
    id: CollectiveRef,
    source: LogicalPartitionId,
    participants: BTreeSet<LogicalPartitionId>,
    byte_count: u64,
}

impl CollectiveBroadcastMirror {
    /// Build a broadcast mirror. The source must be a participant; every
    /// participant is a declared partition (validated at prepare).
    #[must_use]
    pub fn broadcast(
        id: CollectiveRef,
        source: LogicalPartitionId,
        participants: BTreeSet<LogicalPartitionId>,
        byte_count: u64,
    ) -> Self {
        debug_assert!(
            participants.contains(&source),
            "the broadcast source must be a participant"
        );
        Self {
            id,
            source,
            participants,
            byte_count,
        }
    }

    /// The stable collective identity.
    #[must_use]
    pub fn id(&self) -> &CollectiveRef {
        &self.id
    }

    /// The source partition (owner / primary replica).
    #[must_use]
    pub fn source(&self) -> &LogicalPartitionId {
        &self.source
    }

    /// The participant set (source plus all consumers), in stable order.
    #[must_use]
    pub fn participants(&self) -> &BTreeSet<LogicalPartitionId> {
        &self.participants
    }

    /// Byte count of the broadcast value (per participant copy).
    #[must_use]
    pub const fn byte_count(&self) -> u64 {
        self.byte_count
    }

    /// Deterministic canonical bytes of every declared field.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_str(&mut out, self.id.as_str());
        push_str(&mut out, self.source.as_str());
        push_u64(&mut out, self.participants.len() as u64);
        for participant in &self.participants {
            push_str(&mut out, participant.as_str());
        }
        push_u64(&mut out, self.byte_count);
        out
    }
}

/// Stable reference of one operation within the prepare snapshot — the key
/// the no-silent-growth check runs on.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OperationRef {
    /// A launch, by its launch reference.
    Launch(LaunchRef),
    /// A transfer, by its transfer identity.
    Transfer(TransferRef),
    /// A broadcast collective, by its collective identity.
    Collective(CollectiveRef),
    /// A barrier, by its barrier reference.
    Barrier(BarrierRef),
}

/// A `BarrierRef` or `LaunchRef` that a [`TransactionCommitBoundary`] names.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BoundaryRef {
    /// A barrier named by the boundary.
    Barrier(BarrierRef),
    /// A launch named by the boundary.
    Launch(LaunchRef),
}

/// One staged write the transaction will publish.
///
/// A `StagedWrite` is the atomic publication unit: reserved at prepare
/// (byte-counted against the partition's admitted class 3 budget), staged at
/// its partition during execute, and published **all-or-nothing** at commit.
/// A failure or cancel before commit publishes nothing.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StagedWrite {
    partition: LogicalPartitionId,
    output_ref: OutputRef,
    byte_count: u64,
}

impl StagedWrite {
    /// Build a staged write for one partition's output buffer.
    #[must_use]
    pub const fn new(
        partition: LogicalPartitionId,
        output_ref: OutputRef,
        byte_count: u64,
    ) -> Self {
        Self {
            partition,
            output_ref,
            byte_count,
        }
    }

    /// The partition that stages this write.
    #[must_use]
    pub fn partition(&self) -> &LogicalPartitionId {
        &self.partition
    }

    /// The stable output reference.
    #[must_use]
    pub fn output_ref(&self) -> &OutputRef {
        &self.output_ref
    }

    /// The byte count of the staged output.
    #[must_use]
    pub const fn byte_count(&self) -> u64 {
        self.byte_count
    }
}

/// Mirror of one operation of the accepted plan (FC8 pattern).
///
/// `Launch` carries the declared output byte contract of the launch — the
/// write the launch stages — so `prepare` can reserve output buffers
/// (S3). `Transfer` / `CollectiveBroadcast` mirror the logical operation
/// facts in stable canonical form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransactionOperation {
    /// A launch of the semantic program on one partition.
    Launch {
        /// The partition that runs the launch.
        partition: LogicalPartitionId,
        /// The stable reference of the launched program/launch.
        launch_ref: LaunchRef,
        /// The declared byte contract of the launch's output write.
        output_bytes: u64,
    },
    /// A host-staged cross-partition transfer.
    Transfer(TransferOperationMirror),
    /// A broadcast collective (source → participants).
    CollectiveBroadcast(CollectiveBroadcastMirror),
    /// A barrier synchronizing its participant partitions.
    Barrier {
        /// The stable barrier reference.
        barrier_ref: BarrierRef,
        /// The participant partition set.
        partitions: BTreeSet<LogicalPartitionId>,
    },
}

impl TransactionOperation {
    /// A launch operation.
    #[must_use]
    pub fn launch(partition: LogicalPartitionId, launch_ref: LaunchRef, output_bytes: u64) -> Self {
        Self::Launch {
            partition,
            launch_ref,
            output_bytes,
        }
    }

    /// A transfer operation.
    #[must_use]
    pub fn transfer(transfer: TransferOperationMirror) -> Self {
        Self::Transfer(transfer)
    }

    /// A broadcast operation.
    #[must_use]
    pub fn broadcast(broadcast: CollectiveBroadcastMirror) -> Self {
        Self::CollectiveBroadcast(broadcast)
    }

    /// A barrier operation.
    #[must_use]
    pub fn barrier(barrier_ref: BarrierRef, partitions: BTreeSet<LogicalPartitionId>) -> Self {
        Self::Barrier {
            barrier_ref,
            partitions,
        }
    }

    /// The operation's stable snapshot key.
    #[must_use]
    pub fn operation_ref(&self) -> OperationRef {
        match self {
            Self::Launch { launch_ref, .. } => OperationRef::Launch(launch_ref.clone()),
            Self::Transfer(transfer) => OperationRef::Transfer(transfer.id().clone()),
            Self::CollectiveBroadcast(broadcast) => {
                OperationRef::Collective(broadcast.id().clone())
            }
            Self::Barrier { barrier_ref, .. } => OperationRef::Barrier(barrier_ref.clone()),
        }
    }

    /// The partitions this operation involves, in stable order.
    #[must_use]
    pub fn partitions(&self) -> BTreeSet<LogicalPartitionId> {
        match self {
            Self::Launch { partition, .. } => BTreeSet::from([partition.clone()]),
            Self::Transfer(transfer) => {
                BTreeSet::from([transfer.source().clone(), transfer.destination().clone()])
            }
            Self::CollectiveBroadcast(broadcast) => broadcast.participants().clone(),
            Self::Barrier { partitions, .. } => partitions.clone(),
        }
    }

    /// The exact byte contract of the operation: the bytes moved across the
    /// mesh (a transfer moves its byte count; a broadcast moves one copy per
    /// non-source participant; launches and barriers move nothing — their
    /// writes are accounted in the staged write-set).
    #[must_use]
    pub fn byte_count(&self) -> u64 {
        match self {
            Self::Launch { .. } | Self::Barrier { .. } => 0,
            Self::Transfer(transfer) => transfer.byte_count(),
            Self::CollectiveBroadcast(broadcast) => {
                broadcast.byte_count() * (broadcast.participants().len().saturating_sub(1) as u64)
            }
        }
    }

    /// The events this operation completes when it runs, one per involved
    /// partition. An operation is not complete until its declared events join
    /// the boundary or cancellation-safe reclamation reclaims them (S3).
    #[must_use]
    pub fn completed_events(&self) -> BTreeSet<OperationEvent> {
        match self {
            Self::Launch {
                partition,
                launch_ref,
                ..
            } => BTreeSet::from([OperationEvent::LaunchCompleted {
                partition: partition.clone(),
                launch_ref: launch_ref.clone(),
            }]),
            Self::Transfer(transfer) => BTreeSet::from([
                OperationEvent::TransferCompleted {
                    partition: transfer.source().clone(),
                    transfer_ref: transfer.id().clone(),
                },
                OperationEvent::TransferCompleted {
                    partition: transfer.destination().clone(),
                    transfer_ref: transfer.id().clone(),
                },
            ]),
            Self::CollectiveBroadcast(broadcast) => broadcast
                .participants()
                .iter()
                .map(|partition| OperationEvent::BroadcastCompleted {
                    partition: partition.clone(),
                    collective_ref: broadcast.id().clone(),
                })
                .collect(),
            Self::Barrier {
                barrier_ref,
                partitions,
            } => partitions
                .iter()
                .map(|partition| OperationEvent::BarrierCompleted {
                    partition: partition.clone(),
                    barrier_ref: barrier_ref.clone(),
                })
                .collect(),
        }
    }

    /// The staged writes this operation publishes, derived deterministically
    /// from the operation's declared facts (the atomic publication set is the
    /// union over the snapshot).
    #[must_use]
    pub fn staged_writes(&self) -> Vec<StagedWrite> {
        match self {
            Self::Launch {
                partition,
                launch_ref,
                output_bytes,
            } => vec![StagedWrite::new(
                partition.clone(),
                OutputRef::new(format!("launch:{launch_ref}:output")),
                *output_bytes,
            )],
            Self::Transfer(transfer) => vec![StagedWrite::new(
                transfer.destination().clone(),
                OutputRef::new(format!("transfer:{}:destination", transfer.id())),
                transfer.byte_count(),
            )],
            Self::CollectiveBroadcast(broadcast) => broadcast
                .participants()
                .iter()
                .filter(|participant| **participant != *broadcast.source())
                .map(|participant| {
                    StagedWrite::new(
                        participant.clone(),
                        OutputRef::new(format!("broadcast:{}:{participant}", broadcast.id())),
                        broadcast.byte_count(),
                    )
                })
                .collect(),
            Self::Barrier { .. } => Vec::new(),
        }
    }

    /// Deterministic canonical bytes of the operation — identical inputs
    /// produce identical bytes and different operations produce different
    /// bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            Self::Launch {
                partition,
                launch_ref,
                output_bytes,
            } => {
                out.push(0u8); // tag: Launch
                push_str(&mut out, partition.as_str());
                push_str(&mut out, launch_ref.as_str());
                push_u64(&mut out, *output_bytes);
            }
            Self::Transfer(transfer) => {
                out.push(1u8); // tag: Transfer
                out.extend_from_slice(&transfer.canonical_bytes());
            }
            Self::CollectiveBroadcast(broadcast) => {
                out.push(2u8); // tag: CollectiveBroadcast
                out.extend_from_slice(&broadcast.canonical_bytes());
            }
            Self::Barrier {
                barrier_ref,
                partitions,
            } => {
                out.push(3u8); // tag: Barrier
                push_str(&mut out, barrier_ref.as_str());
                push_u64(&mut out, partitions.len() as u64);
                for partition in partitions {
                    push_str(&mut out, partition.as_str());
                }
            }
        }
        out
    }
}

/// One synchronization event of the transaction — a partition reaching one
/// operation. Events join the declared boundary at commit.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OperationEvent {
    /// The partition completed the named launch.
    LaunchCompleted {
        /// The partition that ran the launch.
        partition: LogicalPartitionId,
        /// The completed launch reference.
        launch_ref: LaunchRef,
    },
    /// The partition completed the named transfer.
    TransferCompleted {
        /// The partition (source or destination).
        partition: LogicalPartitionId,
        /// The completed transfer identity.
        transfer_ref: TransferRef,
    },
    /// The partition completed the named broadcast.
    BroadcastCompleted {
        /// The participant partition.
        partition: LogicalPartitionId,
        /// The completed collective identity.
        collective_ref: CollectiveRef,
    },
    /// The partition reached the named barrier.
    BarrierCompleted {
        /// The participant partition.
        partition: LogicalPartitionId,
        /// The reached barrier reference.
        barrier_ref: BarrierRef,
    },
}
