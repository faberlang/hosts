//! KV-D D2: shared model/sequence residency.
//!
//! Parent registration is a private `mod residency` in `composite_host.rs`.
//! This unit cannot edit that file, so the test crate compiles the module
//! directly. `device_descriptor` is re-exported so residency.rs can keep
//! `crate::device_descriptor` in both compilations.

mod device_descriptor {
    pub use faber_host_macos_arm64::device_descriptor::*;
}

#[path = "../src/composite_host/inference_state.rs"]
mod inference_state;

#[path = "../src/composite_host/residency.rs"]
mod residency;

use device_descriptor::{
    DescriptorAllocation, DescriptorView, DeviceBufferInitialization, DeviceBufferLifetime,
    DeviceDataType,
};
use inference_state::{
    FailureOutcome, FailureStage, InvocationMode, SequencePhase, E_INVALID_ARGS, E_KV_OVERFLOW,
    E_KV_PHASE, E_KV_POISONED, E_KV_RELEASED,
};
use residency::{ModelIdentity, ModelSpec, ResidentAllocation, SequenceSpec, SessionResidency};

const LAYERS: u64 = 2;
const KV_HEADS: u64 = 2;
const HEAD_DIM: u64 = 4;
const F32_WIDTH: u64 = 4;

const K_ALLOCATION: u32 = 1;
const V_ALLOCATION: u32 = 2;
const INVOCATION_STATE: u32 = 3;
const WEIGHT_ALLOCATION: u32 = 10;

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

fn weight_allocation(buffer_id: u32) -> DescriptorAllocation {
    DescriptorAllocation {
        buffer_id,
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
        weights: vec![weight_allocation(WEIGHT_ALLOCATION)],
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

fn assert_same_object(left: &ResidentAllocation, right: &ResidentAllocation, what: &str) {
    assert!(
        left.is_same_object(right),
        "{what} must be the same resident object"
    );
    assert_eq!(left.identity(), right.identity(), "{what} identity");
    assert_eq!(left.buffer_id(), right.buffer_id(), "{what} B3 handle");
}

#[test]
fn prepare_admits_one_weight_identity_and_one_k_and_v_allocation() {
    let session = prepare_session(8);
    assert_eq!(session.model().identity().name(), "dense-rung");
    assert_eq!(session.weight_uploads(), 1);
    assert_eq!(session.artifact_prepares(), 1);
    assert_eq!(session.model().weights().len(), 1);
    assert_eq!(session.sequence().k_arena().buffer_id(), K_ALLOCATION);
    assert_eq!(session.sequence().v_arena().buffer_id(), V_ALLOCATION);
    assert_eq!(session.live_allocation_count(), 4);
    assert_eq!(session.released_allocation_count(), 0);
    assert_eq!(session.phase(), SequencePhase::Fresh);
    assert_eq!(session.valid_len(), 0);
    assert_eq!(session.sequence_epoch(), 1);

    let k = session.sequence().k_arena();
    let v = session.sequence().v_arena();
    assert_eq!(k.capacity_bytes(), v.capacity_bytes());
    assert_ne!(
        k.identity(),
        v.identity(),
        "K and V are distinct objects even at equal capacity"
    );
    assert!(!k.is_same_object(v));
}

#[test]
fn equal_byte_counts_are_not_shared_identity() {
    let first = prepare_session(8);
    let second = prepare_session(8);
    let first_k = first.sequence().k_arena();
    let second_k = second.sequence().k_arena();
    assert_eq!(first_k.capacity_bytes(), second_k.capacity_bytes());
    assert_eq!(
        first_k.descriptor().capacity_bytes,
        second_k.descriptor().capacity_bytes
    );
    assert_eq!(first_k.buffer_id(), second_k.buffer_id());
    assert_ne!(
        first_k.identity(),
        second_k.identity(),
        "KV-L7: equal byte counts and buffer ids are not shared identity"
    );
    assert!(!first_k.is_same_object(second_k));

    let first_w = &first.model().weights()[0];
    let second_w = &second.model().weights()[0];
    assert_eq!(first_w.capacity_bytes(), second_w.capacity_bytes());
    assert_ne!(first_w.identity(), second_w.identity());
    assert!(!std::ptr::eq(
        first.model().identity(),
        second.model().identity()
    ));
}

#[test]
fn prefill_and_decode_resolve_identical_allocation_objects() {
    let session = prepare_session(8);
    let prefill = session.resolve(InvocationMode::Prefill);
    let decode = session.resolve(InvocationMode::ScalarDecode);

    assert_eq!(prefill.mode, InvocationMode::Prefill);
    assert_eq!(decode.mode, InvocationMode::ScalarDecode);
    assert!(std::ptr::eq(prefill.model_identity, decode.model_identity));
    assert_eq!(prefill.model_identity.name(), "dense-rung");

    assert_same_object(prefill.k_arena, decode.k_arena, "K arena");
    assert_same_object(prefill.v_arena, decode.v_arena, "V arena");
    assert_same_object(
        prefill.invocation_state,
        decode.invocation_state,
        "invocation-state buffer",
    );
    assert!(
        std::ptr::eq(prefill.weights, decode.weights),
        "weight slice must be the same allocation vector"
    );
    assert_eq!(prefill.weights.len(), 1);
    assert_same_object(&prefill.weights[0], &decode.weights[0], "weight");

    assert_eq!(prefill.artifact, b"prefill-module");
    assert_eq!(decode.artifact, b"decode-module");
    assert_ne!(
        prefill.artifact, decode.artifact,
        "separate graphs are allowed; they still share residency"
    );
}

#[test]
fn both_program_artifacts_are_prepared_before_first_invocation() {
    let session = prepare_session(8);
    assert_eq!(session.phase(), SequencePhase::Fresh);
    assert_eq!(session.valid_len(), 0);
    assert_eq!(session.artifact_prepares(), 1);
    let prefill = session.resolve(InvocationMode::Prefill);
    let decode = session.resolve(InvocationMode::ScalarDecode);
    assert!(!prefill.artifact.is_empty());
    assert!(!decode.artifact.is_empty());
    assert_eq!(session.weight_uploads(), 1);
}

#[test]
fn prefix_and_append_views_share_one_k_and_one_v_allocation() {
    let session = prepare_session(8);
    let sequence = session.sequence();
    let k = sequence.k_arena();
    let v = sequence.v_arena();
    assert_eq!(sequence.k_prefix().allocation_id, k.buffer_id());
    assert_eq!(sequence.k_append().allocation_id, k.buffer_id());
    assert_eq!(sequence.v_prefix().allocation_id, v.buffer_id());
    assert_eq!(sequence.v_append().allocation_id, v.buffer_id());
    assert_ne!(
        sequence.k_prefix().maximum_span,
        sequence.k_append().maximum_span,
        "append and prefix extents differ over the same allocation"
    );
    assert_eq!(
        sequence.k_prefix().maximum_span,
        k.capacity_bytes(),
        "prefix span is the arena capacity; append is the row"
    );
    assert_eq!(sequence.k_append().maximum_span, append_span_bytes());

    let plan = sequence.cache_plan();
    plan.validate().expect("composed K/V plan must admit");
    assert_eq!(plan.allocations.len(), 2);
    let k_views = plan
        .views
        .iter()
        .filter(|view| view.allocation_id == K_ALLOCATION)
        .count();
    let v_views = plan
        .views
        .iter()
        .filter(|view| view.allocation_id == V_ALLOCATION)
        .count();
    assert_eq!(k_views, 2);
    assert_eq!(v_views, 2);
}

#[test]
fn prefill_to_decode_releases_nothing_and_does_not_reupload() {
    let mut session = prepare_session(8);
    let live = session.live_allocation_count();
    let uploads = session.weight_uploads();
    let prepares = session.artifact_prepares();
    let k_ptr = session.sequence().k_arena() as *const ResidentAllocation;
    let v_ptr = session.sequence().v_arena() as *const ResidentAllocation;
    let weight_ptr = &session.model().weights()[0] as *const ResidentAllocation;
    let k_id = session.sequence().k_arena().identity();
    let v_id = session.sequence().v_arena().identity();
    let weight_id = session.model().weights()[0].identity();

    let prefill = session
        .begin_invocation(InvocationMode::Prefill, 4)
        .expect("prefill admits");
    let facts = session.commit(&prefill).expect("prefill commits");
    assert_eq!(facts.valid_len_after, 4);
    assert_eq!(session.phase(), SequencePhase::Prefill);
    assert_eq!(session.valid_len(), 4);
    assert_eq!(session.live_allocation_count(), live);
    assert_eq!(session.released_allocation_count(), 0);
    assert_eq!(session.weight_uploads(), uploads);
    assert_eq!(session.artifact_prepares(), prepares);

    let decode = session
        .begin_invocation(InvocationMode::ScalarDecode, 1)
        .expect("decode admits");
    let facts = session.commit(&decode).expect("decode commits");
    assert_eq!(facts.prefix_before, 4);
    assert_eq!(facts.query_rows, 1);
    assert_eq!(session.phase(), SequencePhase::Decode);
    assert_eq!(session.valid_len(), 5);
    assert_eq!(session.sequence_epoch(), 1);

    assert_eq!(session.live_allocation_count(), live);
    assert_eq!(session.released_allocation_count(), 0);
    assert_eq!(session.weight_uploads(), 1);
    assert_eq!(session.artifact_prepares(), 1);
    assert_eq!(session.sequence().k_arena().identity(), k_id);
    assert_eq!(session.sequence().v_arena().identity(), v_id);
    assert_eq!(session.model().weights()[0].identity(), weight_id);
    assert_eq!(
        session.sequence().k_arena() as *const ResidentAllocation,
        k_ptr
    );
    assert_eq!(
        session.sequence().v_arena() as *const ResidentAllocation,
        v_ptr
    );
    assert_eq!(
        &session.model().weights()[0] as *const ResidentAllocation,
        weight_ptr
    );

    let prefill_handles = session.resolve(InvocationMode::Prefill);
    let decode_handles = session.resolve(InvocationMode::ScalarDecode);
    assert_same_object(
        prefill_handles.k_arena,
        decode_handles.k_arena,
        "K after transition",
    );
    assert_same_object(
        prefill_handles.v_arena,
        decode_handles.v_arena,
        "V after transition",
    );
    assert!(std::ptr::eq(
        prefill_handles.weights,
        decode_handles.weights
    ));
}

#[test]
fn invocation_state_tracks_d1_cursor_and_epoch() {
    let mut session = prepare_session(8);
    let before = session.sequence().invocation_state();
    assert_eq!(before.position, 0);
    assert_eq!(before.valid_len_after, 0);
    assert_eq!(before.sequence_epoch, 1);

    let plan = session
        .begin_invocation(InvocationMode::Prefill, 3)
        .expect("prefill");
    let facts = session.commit(&plan).expect("commit");
    let after = session.sequence().invocation_state();
    assert_eq!(after.position, facts.write_position);
    assert_eq!(after.valid_len_after, facts.valid_len_after);
    assert_eq!(after.query_rows, 3);
    assert_eq!(after.sequence_epoch, session.sequence_epoch());
    assert_eq!(
        session.sequence().invocation_state_allocation().buffer_id(),
        INVOCATION_STATE
    );
}

#[test]
fn resolve_does_not_load_or_allocate() {
    let session = prepare_session(8);
    let live = session.live_allocation_count();
    let uploads = session.weight_uploads();
    let prepares = session.artifact_prepares();
    let _ = session.resolve(InvocationMode::Prefill);
    let _ = session.resolve(InvocationMode::ScalarDecode);
    let _ = session.resolve(InvocationMode::ScalarDecode);
    assert_eq!(session.live_allocation_count(), live);
    assert_eq!(session.weight_uploads(), uploads);
    assert_eq!(session.artifact_prepares(), prepares);
    assert_eq!(session.released_allocation_count(), 0);
}

#[test]
fn prepare_rejects_empty_decode_artifact() {
    let mut spec = model_spec();
    spec.decode_artifact.clear();
    let err = SessionResidency::prepare(spec, sequence_spec(8)).expect_err("empty decode");
    assert_eq!(err.code, E_INVALID_ARGS);
}

#[test]
fn prepare_rejects_missing_weights() {
    let mut spec = model_spec();
    spec.weights.clear();
    let err = SessionResidency::prepare(spec, sequence_spec(8)).expect_err("no weights");
    assert_eq!(err.code, E_INVALID_ARGS);
}

#[test]
fn prepare_rejects_repeated_buffer_identity() {
    let mut spec = sequence_spec(8);
    spec.v_arena.buffer_id = K_ALLOCATION;
    spec.v_prefix.allocation_id = K_ALLOCATION;
    spec.v_append.allocation_id = K_ALLOCATION;
    let err = SessionResidency::prepare(model_spec(), spec).expect_err("K/V id collision");
    assert_eq!(err.code, E_INVALID_ARGS);
}

#[test]
fn prepare_rejects_view_on_the_wrong_allocation() {
    let mut spec = sequence_spec(8);
    spec.k_append.allocation_id = V_ALLOCATION;
    let err = SessionResidency::prepare(model_spec(), spec).expect_err("view skew");
    assert_eq!(err.code, E_INVALID_ARGS);
}

#[test]
fn one_weight_upload_covers_several_weight_buffers() {
    let mut spec = model_spec();
    spec.weights.push(weight_allocation(11));
    let session = SessionResidency::prepare(spec, sequence_spec(8)).expect("two weights");
    assert_eq!(session.model().weights().len(), 2);
    assert_eq!(session.weight_uploads(), 1);
    assert_eq!(session.live_allocation_count(), 5);
    let prefill = session.resolve(InvocationMode::Prefill);
    let decode = session.resolve(InvocationMode::ScalarDecode);
    assert!(std::ptr::eq(prefill.weights, decode.weights));
    assert_ne!(prefill.weights[0].identity(), prefill.weights[1].identity());
}

#[test]
fn logical_reset_is_o1_preserves_allocations_and_does_not_clear() {
    let mut session = prepare_session(8);
    let live = session.live_allocation_count();
    let uploads = session.weight_uploads();
    let prepares = session.artifact_prepares();
    let k_ptr = session.sequence().k_arena() as *const ResidentAllocation;
    let v_ptr = session.sequence().v_arena() as *const ResidentAllocation;
    let weight_ptr = &session.model().weights()[0] as *const ResidentAllocation;
    let inv_ptr = session.sequence().invocation_state_allocation() as *const ResidentAllocation;
    let k_id = session.sequence().k_arena().identity();
    let v_id = session.sequence().v_arena().identity();
    let k_capacity = session.sequence().k_arena().capacity_bytes();
    let v_capacity = session.sequence().v_arena().capacity_bytes();

    let prefill = session
        .begin_transaction(InvocationMode::Prefill, 4)
        .expect("prefill");
    session.commit_transaction(&prefill).expect("commit");
    session
        .commit(
            &session
                .begin_invocation(InvocationMode::ScalarDecode, 1)
                .expect("decode"),
        )
        .expect("decode commit");
    assert_eq!(session.valid_len(), 5);

    let previous_epoch = session.sequence_epoch();
    let receipt = session.logical_reset().expect("logical reset");
    assert_eq!(receipt.previous_epoch, previous_epoch);
    assert_eq!(receipt.sequence_epoch, previous_epoch + 1);
    assert_eq!(receipt.previous_valid_len, 5);
    assert_eq!(receipt.valid_len, 0);
    assert!(!receipt.cache_cleared);
    assert!(!receipt.buffers_zero_filled);
    assert_eq!(receipt.uploads, 0);

    assert_eq!(session.valid_len(), 0);
    assert_eq!(session.phase(), SequencePhase::Fresh);
    assert_eq!(session.sequence_epoch(), previous_epoch + 1);
    assert_eq!(session.reset_count(), 1);
    assert_eq!(session.cache_clear_bytes(), 0);
    assert_eq!(session.buffer_zero_fill_bytes(), 0);
    assert_eq!(session.old_prefix_copy_bytes(), 0);
    assert_eq!(session.weight_uploads(), uploads);
    assert_eq!(session.artifact_prepares(), prepares);
    assert_eq!(session.live_allocation_count(), live);
    assert_eq!(session.released_allocation_count(), 0);

    let cursor = session.sequence().invocation_state();
    assert_eq!(cursor.position, 0);
    assert_eq!(cursor.valid_len_after, 0);
    assert_eq!(cursor.query_rows, 0);
    assert_eq!(cursor.sequence_epoch, session.sequence_epoch());

    assert_eq!(session.sequence().k_arena().identity(), k_id);
    assert_eq!(session.sequence().v_arena().identity(), v_id);
    assert_eq!(session.sequence().k_arena().capacity_bytes(), k_capacity);
    assert_eq!(session.sequence().v_arena().capacity_bytes(), v_capacity);
    assert_eq!(
        session.sequence().k_arena() as *const ResidentAllocation,
        k_ptr
    );
    assert_eq!(
        session.sequence().v_arena() as *const ResidentAllocation,
        v_ptr
    );
    assert_eq!(
        &session.model().weights()[0] as *const ResidentAllocation,
        weight_ptr
    );
    assert_eq!(
        session.sequence().invocation_state_allocation() as *const ResidentAllocation,
        inv_ptr
    );
}

#[test]
fn replay_after_reset_drives_a_new_valid_prefix() {
    let mut session = prepare_session(8);
    session
        .commit(
            &session
                .begin_invocation(InvocationMode::Prefill, 5)
                .expect("first prefill"),
        )
        .expect("first commit");
    let k_ptr = session.sequence().k_arena() as *const ResidentAllocation;
    session.logical_reset().expect("reset");

    let replay = session
        .begin_transaction(InvocationMode::Prefill, 2)
        .expect("replay prefill");
    let facts = session.commit_transaction(&replay).expect("replay commit");
    assert_eq!(facts.prefix_before, 0);
    assert_eq!(facts.query_rows, 2);
    assert_eq!(facts.valid_len_after, 2);
    assert_eq!(session.valid_len(), 2);
    assert_eq!(session.phase(), SequencePhase::Prefill);
    assert_eq!(session.sequence().invocation_state().valid_len_after, 2);
    assert_eq!(session.sequence().invocation_state().position, 0);
    assert_eq!(
        session.sequence().k_arena() as *const ResidentAllocation,
        k_ptr,
        "replay must not reallocate the K arena"
    );
    assert_eq!(session.weight_uploads(), 1);
    assert_eq!(session.cache_clear_bytes(), 0);
}

#[test]
fn pre_dispatch_failure_leaves_residency_unchanged() {
    let mut session = prepare_session(4);
    session
        .commit(
            &session
                .begin_invocation(InvocationMode::Prefill, 4)
                .expect("fill"),
        )
        .expect("commit");
    let inspect = session.inspect();
    let cursor = session.sequence().invocation_state();
    let live = session.live_allocation_count();
    let uploads = session.weight_uploads();
    let k_ptr = session.sequence().k_arena() as *const ResidentAllocation;
    let k_id = session.sequence().k_arena().identity();

    let err = session
        .begin_transaction(InvocationMode::ScalarDecode, 1)
        .expect_err("overflow is pre-dispatch");
    assert_eq!(err.code, E_KV_OVERFLOW);
    assert_eq!(session.inspect(), inspect);
    assert_eq!(session.sequence().invocation_state(), cursor);
    assert_eq!(session.live_allocation_count(), live);
    assert_eq!(session.weight_uploads(), uploads);
    assert_eq!(session.released_allocation_count(), 0);
    assert_eq!(session.sequence().k_arena().identity(), k_id);
    assert_eq!(
        session.sequence().k_arena() as *const ResidentAllocation,
        k_ptr
    );

    let phase_err = session
        .begin_transaction(InvocationMode::Prefill, 1)
        .expect_err("second prefill is pre-dispatch");
    assert_eq!(phase_err.code, E_KV_PHASE);
    assert_eq!(session.inspect(), inspect);
    assert_eq!(session.sequence().invocation_state(), cursor);
}

#[test]
fn admitted_pre_dispatch_abort_leaves_cursor_and_handles() {
    let mut session = prepare_session(8);
    session
        .commit(
            &session
                .begin_invocation(InvocationMode::Prefill, 3)
                .expect("prefill"),
        )
        .expect("commit");
    let inspect = session.inspect();
    let cursor = session.sequence().invocation_state();
    let live = session.live_allocation_count();
    let k_ptr = session.sequence().k_arena() as *const ResidentAllocation;

    let tx = session
        .begin_transaction(InvocationMode::ScalarDecode, 1)
        .expect("admitted");
    assert!(tx.is_pre_dispatch());
    assert_eq!(
        session.fail(&tx).expect("pre-dispatch abort"),
        FailureOutcome::Unchanged
    );
    assert_eq!(session.inspect(), inspect);
    assert_eq!(session.sequence().invocation_state(), cursor);
    assert_eq!(session.live_allocation_count(), live);
    assert_eq!(session.released_allocation_count(), 0);
    assert_eq!(
        session.sequence().k_arena() as *const ResidentAllocation,
        k_ptr
    );
}

#[test]
fn possible_partial_mutation_poisons_and_release_drops_handles() {
    let mut session = prepare_session(8);
    session
        .commit(
            &session
                .begin_invocation(InvocationMode::Prefill, 3)
                .expect("prefill"),
        )
        .expect("commit");
    let live = session.live_allocation_count();
    let cursor = session.sequence().invocation_state();
    let epoch = session.sequence_epoch();
    let valid_len = session.valid_len();
    let k_id = session.sequence().k_arena().identity();

    let mut tx = session
        .begin_transaction(InvocationMode::ScalarDecode, 1)
        .expect("admitted");
    tx.record_possible_mutation(FailureStage::Sync);
    let outcome = session.fail(&tx).expect("poison");
    assert_eq!(
        outcome,
        FailureOutcome::Poisoned {
            epoch,
            failure_stage: FailureStage::Sync,
        }
    );
    assert_eq!(
        session.phase(),
        SequencePhase::Poisoned {
            epoch,
            failure_stage: FailureStage::Sync,
        }
    );
    assert_eq!(session.valid_len(), valid_len);
    assert_eq!(
        session.sequence().invocation_state(),
        cursor,
        "poison leaves the last committed cursor"
    );
    assert_eq!(session.live_allocation_count(), live);
    assert_eq!(session.released_allocation_count(), 0);
    assert_eq!(session.sequence().k_arena().identity(), k_id);

    let reset_err = session.logical_reset().expect_err("reset after poison");
    assert_eq!(reset_err.code, E_KV_POISONED);
    let retry_err = session
        .begin_transaction(InvocationMode::ScalarDecode, 1)
        .expect_err("retry after poison");
    assert_eq!(retry_err.code, E_KV_POISONED);
    let commit_err = session.commit(tx.plan()).expect_err("commit after poison");
    assert_eq!(commit_err.code, E_KV_POISONED);
    assert_eq!(session.live_allocation_count(), live);

    let inspection = session.inspect();
    assert!(!inspection.released);
    assert_eq!(inspection.sequence_epoch, epoch);

    session.release().expect("release after poison");
    assert!(session.released());
    assert_eq!(session.live_allocation_count(), 0);
    assert_eq!(session.released_allocation_count(), live);
    let released = session.inspect();
    assert!(released.released);
    assert_eq!(
        released.phase,
        SequencePhase::Poisoned {
            epoch,
            failure_stage: FailureStage::Sync,
        }
    );

    let begin_err = session
        .begin_invocation(InvocationMode::Prefill, 1)
        .expect_err("begin after release");
    assert_eq!(begin_err.code, E_KV_RELEASED);
    let reset_err = session.logical_reset().expect_err("reset after release");
    assert_eq!(reset_err.code, E_KV_RELEASED);
    let rerelease = session.release().expect_err("double release");
    assert_eq!(rerelease.code, E_KV_RELEASED);
    assert_eq!(session.live_allocation_count(), 0);
}
