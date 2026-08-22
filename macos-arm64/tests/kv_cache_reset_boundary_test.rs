//! KV-E E5: reset and capacity boundary integration.
//!
//! Pure host-level proof over B3 descriptor views, the D1/D4 sequence
//! machine, and D4 residency identities. No device execution.
//!
//! Parent registration is a private `mod` in `composite_host.rs`. This unit
//! cannot edit that file, so the test crate compiles the modules directly.

mod device_descriptor {
    pub use faber_host_macos_arm64::device_descriptor::*;
}

#[path = "../src/composite_host/inference_state.rs"]
mod inference_state;

#[path = "../src/composite_host/residency.rs"]
mod residency;

use device_descriptor::{
    DescriptorAllocation, DescriptorInvocationState, DescriptorView, DeviceBufferInitialization,
    DeviceBufferLifetime, DeviceDataType,
};
use inference_state::{
    CursorFacts, FailureStage, InvocationMode, SequencePhase, SessionInspection, E_KV_OVERFLOW,
    E_KV_POISONED,
};
use residency::{
    AllocationIdentity, ModelIdentity, ModelSpec, ResidentAllocation, SequenceSpec,
    SessionResidency,
};

const LAYERS: u64 = 2;
const KV_HEADS: u64 = 2;
const HEAD_DIM: u64 = 4;
const F32_WIDTH: u64 = 4;

const K_ALLOCATION: u32 = 1;
const V_ALLOCATION: u32 = 2;
const INVOCATION_STATE: u32 = 3;
const WEIGHT_ALLOCATION: u32 = 10;

const CAPACITY: u32 = 8;

fn arena_capacity_bytes(positions: u64) -> u64 {
    LAYERS * KV_HEADS * positions * HEAD_DIM * F32_WIDTH
}

fn append_span_bytes() -> u64 {
    LAYERS * KV_HEADS * HEAD_DIM * F32_WIDTH
}

fn arena_strides(positions: u64) -> Vec<u64> {
    let dim = 1;
    let position = HEAD_DIM;
    let kv_head = positions * HEAD_DIM;
    let layer = KV_HEADS * positions * HEAD_DIM;
    vec![layer, kv_head, position, dim]
}

fn persistent_arena(buffer_id: u32, positions: u64) -> DescriptorAllocation {
    DescriptorAllocation {
        buffer_id,
        dtype: DeviceDataType::F32,
        capacity_bytes: arena_capacity_bytes(positions),
        lifetime: DeviceBufferLifetime::PerProgram,
        initialization: DeviceBufferInitialization::ZeroFill,
    }
}

fn prefix_view(allocation_id: u32, positions: u64) -> DescriptorView {
    DescriptorView {
        allocation_id,
        logical_dims: vec![LAYERS, KV_HEADS, positions, HEAD_DIM],
        strides: arena_strides(positions),
        static_base: 0,
        maximum_span: arena_capacity_bytes(positions),
    }
}

fn append_view(allocation_id: u32, positions: u64) -> DescriptorView {
    DescriptorView {
        allocation_id,
        logical_dims: vec![LAYERS, KV_HEADS, 1, HEAD_DIM],
        strides: arena_strides(positions),
        static_base: 0,
        maximum_span: append_span_bytes(),
    }
}

fn weight_allocation() -> DescriptorAllocation {
    DescriptorAllocation {
        buffer_id: WEIGHT_ALLOCATION,
        dtype: DeviceDataType::F32,
        capacity_bytes: 1024,
        lifetime: DeviceBufferLifetime::PerProgram,
        initialization: DeviceBufferInitialization::HostProvided,
    }
}

fn invocation_state_allocation() -> DescriptorAllocation {
    DescriptorAllocation {
        buffer_id: INVOCATION_STATE,
        dtype: DeviceDataType::U8,
        capacity_bytes: 16,
        lifetime: DeviceBufferLifetime::PerProgram,
        initialization: DeviceBufferInitialization::ZeroFill,
    }
}

fn model_spec() -> ModelSpec {
    ModelSpec {
        identity: ModelIdentity::new("dense-rung", 1),
        prefill_artifact: b"prefill-module".to_vec(),
        decode_artifact: b"decode-module".to_vec(),
        verification_artifact: b"verification-module".to_vec(),
        weights: vec![weight_allocation()],
    }
}

fn sequence_spec(capacity: u32) -> SequenceSpec {
    let positions = u64::from(capacity);
    SequenceSpec {
        k_arena: persistent_arena(K_ALLOCATION, positions),
        v_arena: persistent_arena(V_ALLOCATION, positions),
        k_prefix: prefix_view(K_ALLOCATION, positions),
        k_append: append_view(K_ALLOCATION, positions),
        v_prefix: prefix_view(V_ALLOCATION, positions),
        v_append: append_view(V_ALLOCATION, positions),
        invocation_state: invocation_state_allocation(),
        capacity,
    }
}

fn prepare_session(capacity: u32) -> SessionResidency {
    SessionResidency::prepare(model_spec(), sequence_spec(capacity)).expect("prepare admits")
}

/// Distinctive host-level K/V payload for one cache position.
/// Logical reset does not clear physical cells (KV-L8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CacheCell {
    epoch: u32,
    position: u32,
    token: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArenaSide {
    K,
    V,
}

/// Host-level K/V overlay. Physical occupancy survives reset; live views
/// are gated by invocation-state `valid_len_after` and `sequence_epoch`.
struct LogicalCache {
    capacity: u32,
    k: Vec<Option<CacheCell>>,
    v: Vec<Option<CacheCell>>,
}

impl LogicalCache {
    fn new(capacity: u32) -> Self {
        let slots = capacity as usize;
        Self {
            capacity,
            k: vec![None; slots],
            v: vec![None; slots],
        }
    }

    fn write_commit(&mut self, facts: CursorFacts, epoch: u32, token_base: u32) {
        for i in 0..facts.query_rows {
            let position = facts.write_position + i;
            assert!(
                position < self.capacity,
                "commit wrote past allocation capacity at position {position}"
            );
            let token = token_base + i;
            let slot = position as usize;
            self.k[slot] = Some(CacheCell {
                epoch,
                position,
                token,
            });
            self.v[slot] = Some(CacheCell {
                epoch,
                position,
                token: token.wrapping_add(10_000),
            });
        }
    }

    fn physical(&self, side: ArenaSide) -> Vec<CacheCell> {
        let arena = match side {
            ArenaSide::K => &self.k,
            ArenaSide::V => &self.v,
        };
        arena.iter().copied().flatten().collect()
    }

    fn cell(&self, side: ArenaSide, position: u32) -> Option<CacheCell> {
        let arena = match side {
            ArenaSide::K => &self.k,
            ArenaSide::V => &self.v,
        };
        arena.get(position as usize).copied().flatten()
    }
}

/// Live positions a B3 view may expose, given the D1/D4 cursor.
///
/// Prefix views (`logical_dims[2] == capacity`) read `[0, valid_len_after)`.
/// Append views (`logical_dims[2] == 1`) read the runtime write window.
fn live_positions_for_view(view: &DescriptorView, cursor: DescriptorInvocationState) -> Vec<u32> {
    assert!(
        view.logical_dims.len() >= 3,
        "KV views are [layer, kv_head, position, dim]"
    );
    let axis = view.logical_dims[2];
    if axis == 1 {
        if cursor.query_rows == 0 {
            return Vec::new();
        }
        let start = cursor.position;
        let end = start
            .saturating_add(cursor.query_rows)
            .min(cursor.valid_len_after);
        (start..end).collect()
    } else {
        let cap = u32::try_from(axis).expect("position axis fits u32");
        (0..cursor.valid_len_after.min(cap)).collect()
    }
}

fn side_for_view(view: &DescriptorView) -> ArenaSide {
    if view.allocation_id == K_ALLOCATION {
        ArenaSide::K
    } else {
        assert_eq!(view.allocation_id, V_ALLOCATION);
        ArenaSide::V
    }
}

/// Observe one B3 view through the live cursor. Stale-epoch or
/// past-`valid_len` physical cells are not returned.
fn observe_view(
    cache: &LogicalCache,
    view: &DescriptorView,
    cursor: DescriptorInvocationState,
) -> Vec<CacheCell> {
    let side = side_for_view(view);
    live_positions_for_view(view, cursor)
        .into_iter()
        .filter_map(|position| {
            let cell = cache.cell(side, position)?;
            if cell.epoch != cursor.sequence_epoch || cell.position >= cursor.valid_len_after {
                return None;
            }
            Some(cell)
        })
        .collect()
}

fn observe_all_views(
    session: &SessionResidency,
    cache: &LogicalCache,
) -> Vec<(ArenaSide, CacheCell)> {
    let plan = session.sequence().cache_plan();
    let cursor = plan.invocation_state;
    let mut observed = Vec::new();
    for view in &plan.views {
        for cell in observe_view(cache, view, cursor) {
            observed.push((side_for_view(view), cell));
        }
    }
    observed
}

fn live_tokens(session: &SessionResidency, cache: &LogicalCache, side: ArenaSide) -> Vec<u32> {
    let cursor = session.sequence().invocation_state();
    (0..cursor.valid_len_after)
        .map(|position| {
            let cell = cache
                .cell(side, position)
                .expect("live prefix must be dense");
            assert_eq!(
                cell.epoch, cursor.sequence_epoch,
                "live prefix must not include a stale-epoch row at {position}"
            );
            cell.token
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ObservableStep {
    mode: InvocationMode,
    facts: CursorFacts,
    phase: SequencePhase,
    live_k: Vec<u32>,
    live_v: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AllocationSnapshot {
    k: AllocationIdentity,
    v: AllocationIdentity,
    invocation: AllocationIdentity,
    weight: AllocationIdentity,
    k_ptr: *const ResidentAllocation,
    v_ptr: *const ResidentAllocation,
    k_capacity: u64,
    v_capacity: u64,
    live: usize,
    uploads: u32,
    prepares: u32,
}

impl AllocationSnapshot {
    fn capture(session: &SessionResidency) -> Self {
        Self {
            k: session.sequence().k_arena().identity(),
            v: session.sequence().v_arena().identity(),
            invocation: session.sequence().invocation_state_allocation().identity(),
            weight: session.model().weights()[0].identity(),
            k_ptr: session.sequence().k_arena() as *const ResidentAllocation,
            v_ptr: session.sequence().v_arena() as *const ResidentAllocation,
            k_capacity: session.sequence().k_arena().capacity_bytes(),
            v_capacity: session.sequence().v_arena().capacity_bytes(),
            live: session.live_allocation_count(),
            uploads: session.weight_uploads(),
            prepares: session.artifact_prepares(),
        }
    }

    fn assert_stable(&self, session: &SessionResidency, what: &str) {
        let now = Self::capture(session);
        assert_eq!(now.k, self.k, "{what}: K identity");
        assert_eq!(now.v, self.v, "{what}: V identity");
        assert_eq!(
            now.invocation, self.invocation,
            "{what}: invocation-state identity"
        );
        assert_eq!(now.weight, self.weight, "{what}: weight identity");
        assert_eq!(now.k_ptr, self.k_ptr, "{what}: K object");
        assert_eq!(now.v_ptr, self.v_ptr, "{what}: V object");
        assert_eq!(now.k_capacity, self.k_capacity, "{what}: K capacity");
        assert_eq!(now.v_capacity, self.v_capacity, "{what}: V capacity");
        assert_eq!(now.live, self.live, "{what}: live allocations");
        assert_eq!(now.uploads, self.uploads, "{what}: weight uploads");
        assert_eq!(now.prepares, self.prepares, "{what}: artifact prepares");
        assert_eq!(session.released_allocation_count(), 0, "{what}: no release");
        assert_eq!(session.cache_clear_bytes(), 0, "{what}: no cache clear");
        assert_eq!(session.buffer_zero_fill_bytes(), 0, "{what}: no zero-fill");
        assert_eq!(session.old_prefix_copy_bytes(), 0, "{what}: no prefix copy");
    }
}

fn assert_coordinates(facts: CursorFacts, prefix_before: u32, query_rows: u32, capacity: u32) {
    assert_eq!(facts.prefix_before, prefix_before);
    assert_eq!(facts.query_rows, query_rows);
    assert_eq!(facts.write_position, prefix_before);
    assert_eq!(facts.valid_len_after, prefix_before + query_rows);
    assert_eq!(facts.capacity, capacity);
    assert_eq!(facts.query_start, prefix_before);
}

fn commit_prefill(
    session: &mut SessionResidency,
    cache: &mut LogicalCache,
    query_rows: u32,
    token_base: u32,
) -> CursorFacts {
    let tx = session
        .begin_transaction(InvocationMode::Prefill, query_rows)
        .expect("prefill admits");
    let facts = session.commit_transaction(&tx).expect("prefill commits");
    cache.write_commit(facts, session.sequence_epoch(), token_base);
    facts
}

fn commit_decode(
    session: &mut SessionResidency,
    cache: &mut LogicalCache,
    token: u32,
) -> CursorFacts {
    let tx = session
        .begin_transaction(InvocationMode::ScalarDecode, 1)
        .expect("decode admits");
    let facts = session.commit_transaction(&tx).expect("decode commits");
    cache.write_commit(facts, session.sequence_epoch(), token);
    facts
}

fn prefill_decode_cycle(
    session: &mut SessionResidency,
    cache: &mut LogicalCache,
    final_len: u32,
    token_base: u32,
) -> (CursorFacts, CursorFacts) {
    assert!(
        final_len >= 2,
        "a full prefill+decode cycle needs at least two rows"
    );
    let prefill = commit_prefill(session, cache, final_len - 1, token_base);
    let decode = commit_decode(session, cache, token_base + (final_len - 1));
    (prefill, decode)
}

fn record_step(
    session: &SessionResidency,
    cache: &LogicalCache,
    mode: InvocationMode,
    facts: CursorFacts,
) -> ObservableStep {
    ObservableStep {
        mode,
        facts,
        phase: session.phase(),
        live_k: live_tokens(session, cache, ArenaSide::K),
        live_v: live_tokens(session, cache, ArenaSide::V),
    }
}

fn assert_machine_unchanged(before: &SessionInspection, session: &SessionResidency, what: &str) {
    let after = session.inspect();
    assert_eq!(after, *before, "{what}: inspect must be identical");
    assert_eq!(after.valid_len, before.valid_len, "{what}: valid_len");
    assert_eq!(after.phase, before.phase, "{what}: phase");
    assert_eq!(after.sequence_epoch, before.sequence_epoch, "{what}: epoch");
    assert_eq!(after.last_commit, before.last_commit, "{what}: last_commit");
    assert_eq!(
        session.sequence().invocation_state().valid_len_after,
        before.valid_len,
        "{what}: invocation-state valid_len_after tracks inspect"
    );
}

#[test]
fn capacity_minus_one_completes_prefill_and_decode() {
    let mut session = prepare_session(CAPACITY);
    let mut cache = LogicalCache::new(CAPACITY);
    let identities = AllocationSnapshot::capture(&session);
    let target = CAPACITY - 1;

    let (prefill, decode) = prefill_decode_cycle(&mut session, &mut cache, target, 1);
    assert_coordinates(prefill, 0, target - 1, CAPACITY);
    assert_coordinates(decode, target - 1, 1, CAPACITY);
    assert_eq!(session.valid_len(), target);
    assert_eq!(session.phase(), SequencePhase::Decode);
    assert_eq!(
        session
            .inspect()
            .last_commit
            .map(|facts| facts.valid_len_after),
        Some(target)
    );
    assert_eq!(
        decode
            .causal_end_exclusive(0)
            .expect("decode attends through the new row"),
        target
    );
    assert_eq!(
        live_tokens(&session, &cache, ArenaSide::K).len(),
        target as usize
    );
    identities.assert_stable(&session, "capacity-1 cycle");
}

#[test]
fn exactly_at_capacity_completes_prefill_and_decode() {
    let mut session = prepare_session(CAPACITY);
    let mut cache = LogicalCache::new(CAPACITY);
    let identities = AllocationSnapshot::capture(&session);

    let (prefill, decode) = prefill_decode_cycle(&mut session, &mut cache, CAPACITY, 20);
    assert_coordinates(prefill, 0, CAPACITY - 1, CAPACITY);
    assert_coordinates(decode, CAPACITY - 1, 1, CAPACITY);
    assert_eq!(session.valid_len(), CAPACITY);
    assert_eq!(session.phase(), SequencePhase::Decode);
    assert_eq!(
        session.sequence().invocation_state().valid_len_after,
        CAPACITY
    );
    assert_eq!(
        decode
            .causal_end_exclusive(0)
            .expect("capacity decode includes the last row"),
        CAPACITY
    );
    identities.assert_stable(&session, "capacity cycle");

    let mut filled = prepare_session(CAPACITY);
    let mut filled_cache = LogicalCache::new(CAPACITY);
    let prefill_only = commit_prefill(&mut filled, &mut filled_cache, CAPACITY, 40);
    assert_coordinates(prefill_only, 0, CAPACITY, CAPACITY);
    assert_eq!(filled.valid_len(), CAPACITY);
    assert_eq!(filled.phase(), SequencePhase::Prefill);
}

#[test]
fn capacity_plus_one_fails_atomically() {
    let over = prepare_session(CAPACITY);
    let identities = AllocationSnapshot::capture(&over);
    let before = over.inspect();
    let cursor = over.sequence().invocation_state();
    let err = over
        .begin_transaction(InvocationMode::Prefill, CAPACITY + 1)
        .expect_err("prefill capacity+1");
    assert_eq!(err.code, E_KV_OVERFLOW);
    assert_machine_unchanged(&before, &over, "prefill capacity+1");
    assert_eq!(over.sequence().invocation_state(), cursor);
    identities.assert_stable(&over, "prefill capacity+1");
    assert_eq!(over.phase(), SequencePhase::Fresh);
    assert_eq!(over.valid_len(), 0);

    let mut session = prepare_session(CAPACITY);
    let mut cache = LogicalCache::new(CAPACITY);
    let identities = AllocationSnapshot::capture(&session);
    prefill_decode_cycle(&mut session, &mut cache, CAPACITY, 1);
    let before = session.inspect();
    let cursor = session.sequence().invocation_state();
    let plan_hash = session.sequence().cache_plan().program_graph_hash();
    let err = session
        .begin_transaction(InvocationMode::ScalarDecode, 1)
        .expect_err("decode past capacity");
    assert_eq!(err.code, E_KV_OVERFLOW);
    assert_machine_unchanged(&before, &session, "decode capacity+1");
    assert_eq!(session.sequence().invocation_state(), cursor);
    assert_eq!(session.valid_len(), CAPACITY);
    assert_eq!(
        session
            .inspect()
            .last_commit
            .map(|facts| facts.valid_len_after),
        Some(CAPACITY),
        "overflow must not install a partial cursor"
    );
    identities.assert_stable(&session, "decode capacity+1");
    assert_eq!(
        session.sequence().cache_plan().program_graph_hash(),
        plan_hash,
        "overflow must not mutate the static storage plan"
    );
    assert_eq!(
        live_tokens(&session, &cache, ArenaSide::K).len(),
        CAPACITY as usize
    );
}

#[test]
fn logical_reset_hides_stale_rows_and_preserves_allocations() {
    let mut session = prepare_session(CAPACITY);
    let mut cache = LogicalCache::new(CAPACITY);
    let identities = AllocationSnapshot::capture(&session);
    let plan_hash = session.sequence().cache_plan().program_graph_hash();

    prefill_decode_cycle(&mut session, &mut cache, 5, 100);
    commit_decode(&mut session, &mut cache, 105);
    assert_eq!(session.valid_len(), 6);
    let stale_k: Vec<_> = cache.physical(ArenaSide::K);
    let stale_v: Vec<_> = cache.physical(ArenaSide::V);
    assert_eq!(stale_k.len(), 6);
    let stale_k_tokens: Vec<u32> = stale_k.iter().map(|cell| cell.token).collect();
    let stale_v_tokens: Vec<u32> = stale_v.iter().map(|cell| cell.token).collect();
    let previous_epoch = session.sequence_epoch();

    let receipt = session.logical_reset().expect("logical reset");
    assert_eq!(receipt.previous_epoch, previous_epoch);
    assert_eq!(receipt.sequence_epoch, previous_epoch + 1);
    assert_eq!(receipt.previous_valid_len, 6);
    assert_eq!(receipt.valid_len, 0);
    assert_eq!(receipt.capacity, CAPACITY);
    assert!(!receipt.cache_cleared);
    assert!(!receipt.buffers_zero_filled);
    assert_eq!(receipt.uploads, 0);

    assert_eq!(session.valid_len(), 0);
    assert_eq!(session.phase(), SequencePhase::Fresh);
    assert_eq!(session.sequence_epoch(), previous_epoch + 1);
    assert!(session.inspect().last_commit.is_none());
    let cursor = session.sequence().invocation_state();
    assert_eq!(cursor.position, 0);
    assert_eq!(cursor.valid_len_after, 0);
    assert_eq!(cursor.query_rows, 0);
    assert_eq!(cursor.sequence_epoch, session.sequence_epoch());

    let live = observe_all_views(&session, &cache);
    assert!(
        live.is_empty(),
        "post-reset views must hide every pre-reset row: {live:?}"
    );
    assert!(live_tokens(&session, &cache, ArenaSide::K).is_empty());
    assert!(live_tokens(&session, &cache, ArenaSide::V).is_empty());

    let physical_k = cache.physical(ArenaSide::K);
    let physical_v = cache.physical(ArenaSide::V);
    assert_eq!(physical_k, stale_k, "reset must not clear physical K cells");
    assert_eq!(physical_v, stale_v, "reset must not clear physical V cells");

    identities.assert_stable(&session, "after logical reset");
    assert_eq!(
        session.sequence().cache_plan().program_graph_hash(),
        plan_hash,
        "reset is a cursor/epoch update; static views and allocations stay put"
    );
    let prefix = session.sequence().k_prefix();
    assert_eq!(
        prefix.logical_dims[2],
        u64::from(CAPACITY),
        "static prefix extent remains capacity; live length is the cursor"
    );

    let replay_prefill = commit_prefill(&mut session, &mut cache, 2, 1);
    let replay_decode = commit_decode(&mut session, &mut cache, 3);
    assert_coordinates(replay_prefill, 0, 2, CAPACITY);
    assert_coordinates(replay_decode, 2, 1, CAPACITY);
    assert_eq!(session.valid_len(), 3);

    let live = observe_all_views(&session, &cache);
    assert!(
        !live.is_empty(),
        "post-replay decode must observe the new prefix"
    );
    for (side, cell) in &live {
        let stale = match side {
            ArenaSide::K => &stale_k_tokens,
            ArenaSide::V => &stale_v_tokens,
        };
        assert!(
            !stale.contains(&cell.token),
            "post-reset decode observed pre-reset {:?} token {}",
            side,
            cell.token
        );
        assert_eq!(
            cell.epoch,
            session.sequence_epoch(),
            "live view must not leak a pre-reset epoch"
        );
        assert!(
            cell.position < session.valid_len(),
            "live view must not include a row past valid_len"
        );
    }
    assert_eq!(live_tokens(&session, &cache, ArenaSide::K), vec![1, 2, 3]);
    assert_eq!(
        live_tokens(&session, &cache, ArenaSide::V),
        vec![10_001, 10_002, 10_003]
    );

    let leftover_k = cache
        .cell(ArenaSide::K, 4)
        .expect("physical occupancy beyond the new prefix");
    assert_eq!(leftover_k.epoch, previous_epoch);
    assert_eq!(leftover_k.token, 104);
    assert!(
        observe_all_views(&session, &cache)
            .iter()
            .all(|(_, cell)| cell.position != 4),
        "position 4 is physically occupied but must be unreadable through every view"
    );

    identities.assert_stable(&session, "after replay");
}

#[test]
fn replay_after_reset_matches_fresh_run_observable_sequence() {
    let tokens_prefill = 3;
    let replay_base = 7;

    let mut replay = prepare_session(CAPACITY);
    let mut replay_cache = LogicalCache::new(CAPACITY);
    prefill_decode_cycle(&mut replay, &mut replay_cache, 5, 100);
    let identities = AllocationSnapshot::capture(&replay);
    replay.logical_reset().expect("reset before replay");

    let mut replay_steps = Vec::new();
    let prefill = commit_prefill(&mut replay, &mut replay_cache, tokens_prefill, replay_base);
    replay_steps.push(record_step(
        &replay,
        &replay_cache,
        InvocationMode::Prefill,
        prefill,
    ));
    let decode = commit_decode(&mut replay, &mut replay_cache, replay_base + tokens_prefill);
    replay_steps.push(record_step(
        &replay,
        &replay_cache,
        InvocationMode::ScalarDecode,
        decode,
    ));
    identities.assert_stable(&replay, "replay allocations");

    let mut fresh = prepare_session(CAPACITY);
    let mut fresh_cache = LogicalCache::new(CAPACITY);
    let mut fresh_steps = Vec::new();
    let prefill = commit_prefill(&mut fresh, &mut fresh_cache, tokens_prefill, replay_base);
    fresh_steps.push(record_step(
        &fresh,
        &fresh_cache,
        InvocationMode::Prefill,
        prefill,
    ));
    let decode = commit_decode(&mut fresh, &mut fresh_cache, replay_base + tokens_prefill);
    fresh_steps.push(record_step(
        &fresh,
        &fresh_cache,
        InvocationMode::ScalarDecode,
        decode,
    ));

    assert_eq!(
        replay_steps, fresh_steps,
        "replay after reset must match a fresh run's coordinates, phase, and live K/V"
    );
    assert_ne!(
        replay.sequence_epoch(),
        fresh.sequence_epoch(),
        "replay is a new epoch on the same allocations, not a new session"
    );
    assert_ne!(
        replay.sequence().k_arena().identity(),
        fresh.sequence().k_arena().identity(),
        "a fresh run is a different resident object"
    );
}

#[test]
fn poisoned_sequence_rejects_reset_and_retry() {
    let mut session = prepare_session(CAPACITY);
    let mut cache = LogicalCache::new(CAPACITY);
    let identities = AllocationSnapshot::capture(&session);
    commit_prefill(&mut session, &mut cache, 4, 1);
    let epoch = session.sequence_epoch();
    let valid_len = session.valid_len();
    let cursor = session.sequence().invocation_state();
    let before = session.inspect();

    let mut tx = session
        .begin_transaction(InvocationMode::ScalarDecode, 1)
        .expect("decode admits");
    tx.record_possible_mutation(FailureStage::Dispatch);
    let outcome = session.fail(&tx).expect("unproven mutation poisons");
    assert_eq!(
        outcome,
        inference_state::FailureOutcome::Poisoned {
            epoch,
            failure_stage: FailureStage::Dispatch,
        }
    );
    assert_eq!(
        session.phase(),
        SequencePhase::Poisoned {
            epoch,
            failure_stage: FailureStage::Dispatch,
        }
    );
    assert_eq!(session.valid_len(), valid_len);
    assert_eq!(session.sequence().invocation_state(), cursor);
    assert_eq!(session.inspect().last_commit, before.last_commit);
    identities.assert_stable(&session, "poison leaves allocations");

    let poisoned = session.inspect();
    let reset_err = session.logical_reset().expect_err("reset after poison");
    assert_eq!(reset_err.code, E_KV_POISONED);
    assert_machine_unchanged(&poisoned, &session, "rejected reset");
    identities.assert_stable(&session, "rejected reset");

    let retry_err = session
        .begin_transaction(InvocationMode::ScalarDecode, 1)
        .expect_err("retry after poison");
    assert_eq!(retry_err.code, E_KV_POISONED);
    let prefill_err = session
        .begin_transaction(InvocationMode::Prefill, 1)
        .expect_err("prefill after poison");
    assert_eq!(prefill_err.code, E_KV_POISONED);
    assert_machine_unchanged(&poisoned, &session, "rejected retry");
    identities.assert_stable(&session, "rejected retry");

    assert_eq!(
        live_tokens(&session, &cache, ArenaSide::K),
        vec![1, 2, 3, 4],
        "poison must not drop or extend the last committed prefix"
    );
    let inspection = session.inspect();
    assert!(!inspection.released);
    assert_eq!(
        inspection
            .poisoned_invocation
            .as_ref()
            .map(|plan| plan.mode()),
        Some(InvocationMode::ScalarDecode)
    );
}
