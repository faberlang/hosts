//! KV-D D2/D4: shared model/sequence residency.
//!
//! Pure composition over B3 allocation/view types and the D1 sequence
//! machine. No device allocation, upload, launch, or cache clear. Handle
//! identity is the resident object (KV-L7): equal byte counts are not
//! proof of sharing. D4 adds O(1) logical reset, pre-dispatch abort, poison
//! on possible partial mutation, and release that drops every live handle.
//!
//! Parent registration is a private `mod residency` in `composite_host.rs`;
//! this unit cannot re-export it.

// Load-bearing: the #[path]-included residency tests consume the surface
// the lib build would otherwise flag.
#![allow(dead_code)]

use std::sync::atomic::{AtomicU64, Ordering};

use super::inference_state::{
    CursorFacts, FailureOutcome, InferenceSessionState, InvocationMode, InvocationTransaction,
    PlannedInvocation, ResetReceipt, SequencePhase, SessionError, SessionInspection,
    E_INVALID_ARGS,
};
use crate::device_descriptor::{
    DescriptorAllocation, DescriptorInvocationState, DescriptorView, DeviceBufferLifetime,
    KvCacheDescriptor,
};

/// Process-wide mint so two sessions with equal capacities still get
/// distinct allocation objects (KV-L7).
static NEXT_ALLOCATION_IDENTITY: AtomicU64 = AtomicU64::new(1);

fn mint_identity() -> AllocationIdentity {
    AllocationIdentity(NEXT_ALLOCATION_IDENTITY.fetch_add(1, Ordering::Relaxed))
}

fn invalid_args(message: impl Into<String>) -> SessionError {
    SessionError {
        code: E_INVALID_ARGS,
        message: message.into(),
    }
}

/// Unique identity of one resident allocation. Minted once; never derived
/// from [`DescriptorAllocation::capacity_bytes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AllocationIdentity(u64);

/// One resident allocation: a B3 descriptor plus a minted object identity.
/// Programs resolve this object, not a copy sized the same way.
#[derive(Debug, PartialEq, Eq)]
pub struct ResidentAllocation {
    identity: AllocationIdentity,
    descriptor: DescriptorAllocation,
}

impl ResidentAllocation {
    fn new(descriptor: DescriptorAllocation) -> Result<Self, SessionError> {
        expect_persistent(&descriptor, "allocation")?;
        Ok(Self {
            identity: mint_identity(),
            descriptor,
        })
    }

    #[must_use]
    pub fn identity(&self) -> AllocationIdentity {
        self.identity
    }

    #[must_use]
    pub fn descriptor(&self) -> DescriptorAllocation {
        self.descriptor
    }

    #[must_use]
    pub fn buffer_id(&self) -> u32 {
        self.descriptor.buffer_id
    }

    #[must_use]
    pub fn capacity_bytes(&self) -> u64 {
        self.descriptor.capacity_bytes
    }

    /// Same resident object, not equal capacity (KV-L7).
    #[must_use]
    pub fn is_same_object(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

fn expect_persistent(allocation: &DescriptorAllocation, what: &str) -> Result<(), SessionError> {
    if allocation.buffer_id == 0 {
        return Err(invalid_args(format!(
            "{what} uses the reserved zero buffer identity"
        )));
    }
    if allocation.capacity_bytes == 0 {
        return Err(invalid_args(format!("{what} has a zero byte capacity")));
    }
    if allocation.lifetime != DeviceBufferLifetime::PerProgram {
        return Err(invalid_args(format!(
            "{what} must have PerProgram lifetime; got {:?}",
            allocation.lifetime
        )));
    }
    Ok(())
}

fn expect_view(
    view: &DescriptorView,
    allocation: &DescriptorAllocation,
    what: &str,
) -> Result<(), SessionError> {
    if view.allocation_id != allocation.buffer_id {
        return Err(invalid_args(format!(
            "{what} allocation_id {} does not match allocation {}",
            view.allocation_id, allocation.buffer_id
        )));
    }
    if view.logical_dims.is_empty() || view.logical_dims.len() != view.strides.len() {
        return Err(invalid_args(format!(
            "{what} has rank-mismatched dims and strides"
        )));
    }
    let end = view
        .static_base
        .checked_add(view.maximum_span)
        .ok_or_else(|| invalid_args(format!("{what} overflows its static envelope")))?;
    if end > allocation.capacity_bytes {
        return Err(invalid_args(format!(
            "{what} spans {end} bytes but allocation capacity is {}",
            allocation.capacity_bytes
        )));
    }
    Ok(())
}

fn expect_unique_buffer_ids(ids: &[u32]) -> Result<(), SessionError> {
    for (index, id) in ids.iter().enumerate() {
        if ids[..index].contains(id) {
            return Err(invalid_args(format!(
                "allocation buffer identity {id} is repeated"
            )));
        }
    }
    Ok(())
}

/// Stable model identity. Shared by every invocation program.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelIdentity {
    name: String,
    epoch: u32,
}

impl ModelIdentity {
    #[must_use]
    pub fn new(name: impl Into<String>, epoch: u32) -> Self {
        Self {
            name: name.into(),
            epoch,
        }
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Prefill, scalar-decode, and verification module artifacts. Every admitted
/// invocation graph is prepared before the first invocation; no route loads or
/// compiles lazily when the mode changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedArtifacts {
    prefill: Vec<u8>,
    scalar_decode: Vec<u8>,
    verification: Vec<u8>,
}

impl PreparedArtifacts {
    #[must_use]
    pub fn prefill(&self) -> &[u8] {
        &self.prefill
    }

    #[must_use]
    pub fn scalar_decode(&self) -> &[u8] {
        &self.scalar_decode
    }

    #[must_use]
    pub fn verification(&self) -> &[u8] {
        &self.verification
    }

    #[must_use]
    pub fn for_mode(&self, mode: InvocationMode) -> &[u8] {
        match mode {
            InvocationMode::Prefill => &self.prefill,
            InvocationMode::ScalarDecode => &self.scalar_decode,
            InvocationMode::Verification => &self.verification,
        }
    }
}

/// Input facts for [`ModelResidency::prepare`].
pub struct ModelSpec {
    pub identity: ModelIdentity,
    pub prefill_artifact: Vec<u8>,
    pub decode_artifact: Vec<u8>,
    pub verification_artifact: Vec<u8>,
    pub weights: Vec<DescriptorAllocation>,
}

/// Model residency: one identity, artifacts prepared before first
/// invocation, and shared weight allocations uploaded once.
#[derive(Debug, PartialEq, Eq)]
pub struct ModelResidency {
    identity: ModelIdentity,
    artifacts: PreparedArtifacts,
    weights: Vec<ResidentAllocation>,
    weight_uploads: u32,
    artifact_prepares: u32,
}

impl ModelResidency {
    /// Prepare both program artifacts and upload weights once.
    pub fn prepare(spec: ModelSpec) -> Result<Self, SessionError> {
        if spec.identity.name.is_empty() {
            return Err(invalid_args("model identity name must not be empty"));
        }
        if spec.prefill_artifact.is_empty() {
            return Err(invalid_args(
                "prefill artifact must be prepared before first invocation",
            ));
        }
        if spec.decode_artifact.is_empty() {
            return Err(invalid_args(
                "scalar-decode artifact must be prepared before first invocation",
            ));
        }
        if spec.verification_artifact.is_empty() {
            return Err(invalid_args(
                "verification artifact must be prepared before first invocation",
            ));
        }
        if spec.weights.is_empty() {
            return Err(invalid_args(
                "model residency requires at least one weight allocation",
            ));
        }
        let mut ids = Vec::with_capacity(spec.weights.len());
        let mut weights = Vec::with_capacity(spec.weights.len());
        for weight in spec.weights {
            ids.push(weight.buffer_id);
            weights.push(ResidentAllocation::new(weight)?);
        }
        expect_unique_buffer_ids(&ids)?;
        Ok(Self {
            identity: spec.identity,
            artifacts: PreparedArtifacts {
                prefill: spec.prefill_artifact,
                scalar_decode: spec.decode_artifact,
                verification: spec.verification_artifact,
            },
            weights,
            weight_uploads: 1,
            artifact_prepares: 1,
        })
    }

    #[must_use]
    pub fn identity(&self) -> &ModelIdentity {
        &self.identity
    }

    #[must_use]
    pub fn artifacts(&self) -> &PreparedArtifacts {
        &self.artifacts
    }

    #[must_use]
    pub fn weights(&self) -> &[ResidentAllocation] {
        &self.weights
    }

    #[must_use]
    pub fn weight_uploads(&self) -> u32 {
        self.weight_uploads
    }

    #[must_use]
    pub fn artifact_prepares(&self) -> u32 {
        self.artifact_prepares
    }
}

/// Input facts for [`SequenceResidency::prepare`].
pub struct SequenceSpec {
    pub k_arena: DescriptorAllocation,
    pub v_arena: DescriptorAllocation,
    pub k_prefix: DescriptorView,
    pub k_append: DescriptorView,
    pub v_prefix: DescriptorView,
    pub v_append: DescriptorView,
    pub invocation_state: DescriptorAllocation,
    pub capacity: u32,
}

/// Sequence residency: one K arena, one V arena, one invocation-state
/// buffer, and D1 cursor/epoch ownership.
#[derive(Debug, PartialEq, Eq)]
pub struct SequenceResidency {
    k_arena: ResidentAllocation,
    v_arena: ResidentAllocation,
    k_prefix: DescriptorView,
    k_append: DescriptorView,
    v_prefix: DescriptorView,
    v_append: DescriptorView,
    invocation_state_allocation: ResidentAllocation,
    invocation_state: DescriptorInvocationState,
    state: InferenceSessionState,
}

impl SequenceResidency {
    /// Allocate one K and one V arena plus the invocation-state buffer.
    /// Prefix/append views share those arenas; they do not copy them.
    pub fn prepare(spec: SequenceSpec) -> Result<Self, SessionError> {
        expect_view(&spec.k_prefix, &spec.k_arena, "K prefix view")?;
        expect_view(&spec.k_append, &spec.k_arena, "K append view")?;
        expect_view(&spec.v_prefix, &spec.v_arena, "V prefix view")?;
        expect_view(&spec.v_append, &spec.v_arena, "V append view")?;
        expect_unique_buffer_ids(&[
            spec.k_arena.buffer_id,
            spec.v_arena.buffer_id,
            spec.invocation_state.buffer_id,
        ])?;
        let state = InferenceSessionState::new(spec.capacity)?;
        let invocation_state = DescriptorInvocationState {
            position: 0,
            valid_len_after: 0,
            query_rows: 0,
            sequence_epoch: state.sequence_epoch(),
        };
        Ok(Self {
            k_arena: ResidentAllocation::new(spec.k_arena)?,
            v_arena: ResidentAllocation::new(spec.v_arena)?,
            k_prefix: spec.k_prefix,
            k_append: spec.k_append,
            v_prefix: spec.v_prefix,
            v_append: spec.v_append,
            invocation_state_allocation: ResidentAllocation::new(spec.invocation_state)?,
            invocation_state,
            state,
        })
    }

    #[must_use]
    pub fn k_arena(&self) -> &ResidentAllocation {
        &self.k_arena
    }

    #[must_use]
    pub fn v_arena(&self) -> &ResidentAllocation {
        &self.v_arena
    }

    #[must_use]
    pub fn k_prefix(&self) -> &DescriptorView {
        &self.k_prefix
    }

    #[must_use]
    pub fn k_append(&self) -> &DescriptorView {
        &self.k_append
    }

    #[must_use]
    pub fn v_prefix(&self) -> &DescriptorView {
        &self.v_prefix
    }

    #[must_use]
    pub fn v_append(&self) -> &DescriptorView {
        &self.v_append
    }

    #[must_use]
    pub fn invocation_state_allocation(&self) -> &ResidentAllocation {
        &self.invocation_state_allocation
    }

    #[must_use]
    pub fn invocation_state(&self) -> DescriptorInvocationState {
        self.invocation_state
    }

    #[must_use]
    pub fn state(&self) -> &InferenceSessionState {
        &self.state
    }

    /// B3 storage plan for the two persistent arenas. Launch bindings are
    /// empty here: D2 owns residency, not dispatch.
    #[must_use]
    pub fn cache_plan(&self) -> KvCacheDescriptor {
        KvCacheDescriptor {
            allocations: vec![self.k_arena.descriptor, self.v_arena.descriptor],
            views: vec![
                self.k_prefix.clone(),
                self.k_append.clone(),
                self.v_prefix.clone(),
                self.v_append.clone(),
            ],
            invocation_state: self.invocation_state,
            launch_bindings: Vec::new(),
        }
    }

    fn sync_invocation_state(&mut self, facts: &CursorFacts) {
        self.invocation_state = DescriptorInvocationState {
            position: facts.write_position,
            valid_len_after: facts.valid_len_after,
            query_rows: facts.query_rows,
            sequence_epoch: self.state.sequence_epoch(),
        };
    }

    fn sync_invocation_state_reset(&mut self) {
        self.invocation_state = DescriptorInvocationState {
            position: 0,
            valid_len_after: 0,
            query_rows: 0,
            sequence_epoch: self.state.sequence_epoch(),
        };
    }

    pub fn begin_transaction(
        &self,
        mode: InvocationMode,
        query_rows: u32,
    ) -> Result<InvocationTransaction, SessionError> {
        self.state.begin_transaction(mode, query_rows)
    }

    pub fn commit(&mut self, plan: &PlannedInvocation) -> Result<CursorFacts, SessionError> {
        let facts = self.state.commit(plan)?;
        self.sync_invocation_state(&facts);
        Ok(facts)
    }

    pub fn commit_transaction(
        &mut self,
        tx: &InvocationTransaction,
    ) -> Result<CursorFacts, SessionError> {
        self.commit(tx.plan())
    }

    /// Logical reset (KV-L8): epoch advances, cursor/valid length return to
    /// zero, arenas and views stay put. No cache clear or zero-fill.
    pub fn logical_reset(&mut self) -> Result<ResetReceipt, SessionError> {
        let receipt = self.state.logical_reset()?;
        self.sync_invocation_state_reset();
        Ok(receipt)
    }

    pub fn abort_pre_dispatch(&self, tx: &InvocationTransaction) -> Result<(), SessionError> {
        self.state.abort_pre_dispatch(tx)
    }

    pub fn fail(&mut self, tx: &InvocationTransaction) -> Result<FailureOutcome, SessionError> {
        // Poison leaves the last committed cursor; pre-dispatch is a no-op.
        self.state.fail(tx)
    }

    pub fn release(&mut self) -> Result<(), SessionError> {
        self.state.release()
    }
}

/// Handles both invocation programs resolve. References are into the
/// session-owned objects, so pointer equality is structural identity.
#[derive(Debug, Clone, Copy)]
pub struct ResolvedHandles<'a> {
    pub mode: InvocationMode,
    pub model_identity: &'a ModelIdentity,
    pub artifact: &'a [u8],
    pub weights: &'a [ResidentAllocation],
    pub k_arena: &'a ResidentAllocation,
    pub v_arena: &'a ResidentAllocation,
    pub invocation_state: &'a ResidentAllocation,
}

/// Shared model and sequence residency for one model session (KV-L7).
/// Invocation programs (D3) resolve handles through [`Self::resolve`].
#[derive(Debug, PartialEq, Eq)]
pub struct SessionResidency {
    model: ModelResidency,
    sequence: SequenceResidency,
    released_allocations: usize,
    live_handles: bool,
    cache_clear_bytes: u64,
    buffer_zero_fill_bytes: u64,
    old_prefix_copy_bytes: u64,
    reset_count: u32,
}

impl SessionResidency {
    /// Compose prepared model and sequence residency. Buffer identities
    /// must be unique across weights, K, V, and invocation-state.
    pub fn compose(
        model: ModelResidency,
        sequence: SequenceResidency,
    ) -> Result<Self, SessionError> {
        let mut ids: Vec<u32> = model
            .weights
            .iter()
            .map(ResidentAllocation::buffer_id)
            .collect();
        ids.push(sequence.k_arena.buffer_id());
        ids.push(sequence.v_arena.buffer_id());
        ids.push(sequence.invocation_state_allocation.buffer_id());
        expect_unique_buffer_ids(&ids)?;
        Ok(Self {
            model,
            sequence,
            released_allocations: 0,
            live_handles: true,
            cache_clear_bytes: 0,
            buffer_zero_fill_bytes: 0,
            old_prefix_copy_bytes: 0,
            reset_count: 0,
        })
    }

    pub fn prepare(model: ModelSpec, sequence: SequenceSpec) -> Result<Self, SessionError> {
        Self::compose(
            ModelResidency::prepare(model)?,
            SequenceResidency::prepare(sequence)?,
        )
    }

    #[must_use]
    pub fn model(&self) -> &ModelResidency {
        &self.model
    }

    #[must_use]
    pub fn sequence(&self) -> &SequenceResidency {
        &self.sequence
    }

    #[must_use]
    pub fn weight_uploads(&self) -> u32 {
        self.model.weight_uploads
    }

    #[must_use]
    pub fn artifact_prepares(&self) -> u32 {
        self.model.artifact_prepares
    }

    #[must_use]
    pub fn live_allocation_count(&self) -> usize {
        if self.live_handles {
            self.model.weights.len() + 3
        } else {
            0
        }
    }

    #[must_use]
    pub fn cache_clear_bytes(&self) -> u64 {
        self.cache_clear_bytes
    }

    #[must_use]
    pub fn buffer_zero_fill_bytes(&self) -> u64 {
        self.buffer_zero_fill_bytes
    }

    #[must_use]
    pub fn old_prefix_copy_bytes(&self) -> u64 {
        self.old_prefix_copy_bytes
    }

    #[must_use]
    pub fn reset_count(&self) -> u32 {
        self.reset_count
    }

    #[must_use]
    pub fn released(&self) -> bool {
        self.sequence.state.released()
    }

    #[must_use]
    pub fn released_allocation_count(&self) -> usize {
        self.released_allocations
    }

    #[must_use]
    pub fn phase(&self) -> SequencePhase {
        self.sequence.state.phase()
    }

    #[must_use]
    pub fn valid_len(&self) -> u32 {
        self.sequence.state.valid_len()
    }

    #[must_use]
    pub fn sequence_epoch(&self) -> u32 {
        self.sequence.state.sequence_epoch()
    }

    #[must_use]
    pub fn inspect(&self) -> SessionInspection {
        self.sequence.state.inspect()
    }

    /// Resolve the shared handles for an explicit program. Mode selects the
    /// prepared artifact; it does not allocate, upload, or load.
    #[must_use]
    pub fn resolve(&self, mode: InvocationMode) -> ResolvedHandles<'_> {
        ResolvedHandles {
            mode,
            model_identity: &self.model.identity,
            artifact: self.model.artifacts.for_mode(mode),
            weights: &self.model.weights,
            k_arena: &self.sequence.k_arena,
            v_arena: &self.sequence.v_arena,
            invocation_state: &self.sequence.invocation_state_allocation,
        }
    }

    pub fn begin_invocation(
        &self,
        mode: InvocationMode,
        query_rows: u32,
    ) -> Result<PlannedInvocation, SessionError> {
        self.sequence.state.begin_invocation(mode, query_rows)
    }

    pub fn begin_transaction(
        &self,
        mode: InvocationMode,
        query_rows: u32,
    ) -> Result<InvocationTransaction, SessionError> {
        self.sequence.begin_transaction(mode, query_rows)
    }

    /// Commit through D1. The prefill→decode transition allocates nothing,
    /// uploads no weights, and releases nothing.
    pub fn commit(&mut self, plan: &PlannedInvocation) -> Result<CursorFacts, SessionError> {
        self.sequence.commit(plan)
    }

    pub fn commit_transaction(
        &mut self,
        tx: &InvocationTransaction,
    ) -> Result<CursorFacts, SessionError> {
        self.sequence.commit_transaction(tx)
    }

    /// Logical reset (KV-L8): O(1) cursor/epoch update. Allocations,
    /// capacities, weight uploads, and program artifacts stay put. No
    /// cache clear, buffer zero-fill, or upload.
    pub fn logical_reset(&mut self) -> Result<ResetReceipt, SessionError> {
        let receipt = self.sequence.logical_reset()?;
        self.reset_count = self.reset_count.saturating_add(1);
        Ok(receipt)
    }

    pub fn abort_pre_dispatch(&self, tx: &InvocationTransaction) -> Result<(), SessionError> {
        self.sequence.abort_pre_dispatch(tx)
    }

    /// Pre-dispatch: unchanged. Possible partial mutation: poison; reset
    /// and retry stay illegal because rollback is not proven.
    pub fn fail(&mut self, tx: &InvocationTransaction) -> Result<FailureOutcome, SessionError> {
        self.sequence.fail(tx)
    }

    /// Terminal release: every live handle is dropped. Inspect remains
    /// legal. No device work — the live-handle count is the drop.
    pub fn release(&mut self) -> Result<(), SessionError> {
        self.sequence.release()?;
        if self.live_handles {
            let live = self.model.weights.len() + 3;
            self.released_allocations = self.released_allocations.saturating_add(live);
            self.live_handles = false;
        }
        Ok(())
    }
}
