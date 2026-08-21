//! KV-D D1: pure model-session state machine.
//!
//! Parent registration is a private `mod inference_state` in
//! `composite_host.rs`. This unit cannot edit that file, so the test crate
//! compiles the machine directly.

#[path = "../src/composite_host/inference_state.rs"]
mod inference_state;

use inference_state::{
    CursorFacts, FailureOutcome, FailureStage, InferenceSessionState, InvocationMode,
    SequencePhase, E_INVALID_ARGS, E_KV_OVERFLOW, E_KV_PHASE, E_KV_POISONED, E_KV_RELEASED,
    E_KV_STALE,
};

fn fresh(capacity: u32) -> InferenceSessionState {
    InferenceSessionState::new(capacity).expect("capacity >= 1")
}

fn commit_prefill(state: &mut InferenceSessionState, query_rows: u32) -> CursorFacts {
    let plan = state
        .begin_invocation(InvocationMode::Prefill, query_rows)
        .expect("prefill admits");
    state.commit(&plan).expect("prefill commits")
}

fn commit_decode(state: &mut InferenceSessionState) -> CursorFacts {
    let plan = state
        .begin_invocation(InvocationMode::ScalarDecode, 1)
        .expect("decode admits");
    state.commit(&plan).expect("decode commits")
}

fn assert_coordinates(facts: CursorFacts, prefix_before: u32, query_rows: u32, capacity: u32) {
    assert_eq!(facts.prefix_before, prefix_before);
    assert_eq!(facts.query_rows, query_rows);
    assert_eq!(facts.write_position, prefix_before);
    assert_eq!(facts.valid_len_after, prefix_before + query_rows);
    assert_eq!(facts.capacity, capacity);
    assert_eq!(facts.query_start, prefix_before);
}

#[test]
fn new_rejects_zero_capacity() {
    let err = InferenceSessionState::new(0).expect_err("zero capacity");
    assert_eq!(err.code, E_INVALID_ARGS);
}

#[test]
fn fresh_starts_empty_epoch_one() {
    let state = fresh(8);
    assert_eq!(state.phase(), SequencePhase::Fresh);
    assert_eq!(state.valid_len(), 0);
    assert_eq!(state.capacity(), 8);
    assert_eq!(state.sequence_epoch(), 1);
    assert!(!state.released());
}

#[test]
fn fresh_prefill_decode_phase_transitions() {
    let mut state = fresh(8);
    let prefill = commit_prefill(&mut state, 4);
    assert_coordinates(prefill, 0, 4, 8);
    assert_eq!(state.phase(), SequencePhase::Prefill);
    assert_eq!(state.valid_len(), 4);

    let decode = commit_decode(&mut state);
    assert_coordinates(decode, 4, 1, 8);
    assert_eq!(state.phase(), SequencePhase::Decode);
    assert_eq!(state.valid_len(), 5);

    let decode_again = commit_decode(&mut state);
    assert_coordinates(decode_again, 5, 1, 8);
    assert_eq!(state.phase(), SequencePhase::Decode);
    assert_eq!(state.valid_len(), 6);
}

#[test]
fn contiguous_commit_advances_valid_len_by_query_rows() {
    let mut state = fresh(16);
    assert_eq!(state.valid_len(), 0);
    commit_prefill(&mut state, 7);
    assert_eq!(state.valid_len(), 7);
    commit_decode(&mut state);
    assert_eq!(state.valid_len(), 8);
    commit_decode(&mut state);
    assert_eq!(state.valid_len(), 9);
}

#[test]
fn coordinate_facts_are_distinct_and_arithmetic_holds() {
    let mut state = fresh(32);
    let prefill = commit_prefill(&mut state, 6);
    assert_eq!(prefill.write_position, prefill.prefix_before);
    assert_eq!(
        prefill.valid_len_after,
        prefill.prefix_before + prefill.query_rows
    );
    assert_eq!(prefill.query_start, prefill.prefix_before);
    // Distinct facts: six named fields, not a catch-all seq_len.
    let names = [
        "prefix_before",
        "query_rows",
        "write_position",
        "valid_len_after",
        "capacity",
        "query_start",
    ];
    for name in names {
        assert!(
            names.iter().filter(|n| **n == name).count() == 1,
            "coordinate fact {name} must be unique"
        );
        assert_ne!(name, "seq_len");
    }

    let decode = commit_decode(&mut state);
    assert_eq!(decode.prefix_before, 6);
    assert_eq!(decode.query_rows, 1);
    assert_eq!(decode.write_position, 6);
    assert_eq!(decode.valid_len_after, 7);
    assert_eq!(decode.query_start, 6);
    assert_eq!(decode.capacity, 32);
}

#[test]
fn query_row_i_attends_prefix_plus_i_plus_one() {
    let mut state = fresh(8);
    let prefill = state
        .begin_invocation(InvocationMode::Prefill, 4)
        .expect("prefill");
    let facts = prefill.coordinates();
    assert_eq!(facts.causal_end_exclusive(0).expect("row 0"), 1);
    assert_eq!(facts.causal_end_exclusive(1).expect("row 1"), 2);
    assert_eq!(facts.causal_end_exclusive(2).expect("row 2"), 3);
    assert_eq!(facts.causal_end_exclusive(3).expect("row 3"), 4);
    let oob = facts.causal_end_exclusive(4).expect_err("row 4");
    assert_eq!(oob.code, E_INVALID_ARGS);
    state.commit(&prefill).expect("commit prefill");

    let decode = state
        .begin_invocation(InvocationMode::ScalarDecode, 1)
        .expect("decode");
    assert_eq!(
        decode.coordinates().causal_end_exclusive(0).expect("row 0"),
        5
    );
}

#[test]
fn overflow_capacity_plus_one_rejected_before_mutation() {
    let mut state = fresh(4);
    let before = state.clone();
    let err = state
        .begin_invocation(InvocationMode::Prefill, 5)
        .expect_err("capacity+1");
    assert_eq!(err.code, E_KV_OVERFLOW);
    assert_eq!(state, before);

    commit_prefill(&mut state, 4);
    assert_eq!(state.valid_len(), 4);
    let filled = state.clone();
    let err = state
        .begin_invocation(InvocationMode::ScalarDecode, 1)
        .expect_err("decode past capacity");
    assert_eq!(err.code, E_KV_OVERFLOW);
    assert_eq!(state, filled);
}

#[test]
fn overflow_partial_fill_plus_too_many_query_rows_is_fail_closed() {
    let mut state = fresh(4);
    commit_prefill(&mut state, 3);
    let before = state.clone();
    let err = state
        .begin_invocation(InvocationMode::ScalarDecode, 2)
        .expect_err("decode M=2 is invalid before overflow arithmetic");
    assert_eq!(err.code, E_INVALID_ARGS);
    assert_eq!(state, before);
}

#[test]
fn reset_zeros_valid_len_advances_epoch_preserves_capacity() {
    let mut state = fresh(8);
    let epoch = state.sequence_epoch();
    commit_prefill(&mut state, 3);
    commit_decode(&mut state);
    assert_eq!(state.valid_len(), 4);
    assert_eq!(state.capacity(), 8);

    let next = state.reset().expect("reset");
    assert_eq!(next, epoch + 1);
    assert_eq!(state.sequence_epoch(), epoch + 1);
    assert_eq!(state.valid_len(), 0);
    assert_eq!(state.capacity(), 8);
    assert_eq!(state.phase(), SequencePhase::Fresh);
    assert!(state.inspect().last_commit.is_none());

    commit_prefill(&mut state, 2);
    assert_eq!(state.valid_len(), 2);
    assert_eq!(state.sequence_epoch(), epoch + 1);
}

#[test]
fn mode_is_explicit_not_inferred_from_length() {
    let mut state = fresh(8);
    let before = state.clone();
    let err = state
        .begin_invocation(InvocationMode::ScalarDecode, 1)
        .expect_err("decode from fresh");
    assert_eq!(err.code, E_KV_PHASE);
    assert_eq!(state, before);

    commit_prefill(&mut state, 1);
    let after_prefill = state.clone();
    let err = state
        .begin_invocation(InvocationMode::Prefill, 1)
        .expect_err("second prefill");
    assert_eq!(err.code, E_KV_PHASE);
    assert_eq!(state, after_prefill);

    commit_decode(&mut state);
    let after_decode = state.clone();
    let err = state
        .begin_invocation(InvocationMode::Prefill, 2)
        .expect_err("prefill after decode");
    assert_eq!(err.code, E_KV_PHASE);
    assert_eq!(state, after_decode);
}

#[test]
fn scalar_decode_rejects_query_rows_other_than_one() {
    let mut state = fresh(8);
    commit_prefill(&mut state, 2);
    let before = state.clone();
    let err = state
        .begin_invocation(InvocationMode::ScalarDecode, 2)
        .expect_err("M=2 decode");
    assert_eq!(err.code, E_INVALID_ARGS);
    assert_eq!(state, before);
}

#[test]
fn zero_query_rows_rejected_before_mutation() {
    let state = fresh(8);
    let before = state.clone();
    let err = state
        .begin_invocation(InvocationMode::Prefill, 0)
        .expect_err("zero rows");
    assert_eq!(err.code, E_INVALID_ARGS);
    assert_eq!(state, before);
}

#[test]
fn pre_dispatch_failure_leaves_state_unchanged() {
    let mut state = fresh(4);
    commit_prefill(&mut state, 4);
    let before = state.clone();
    let err = state
        .begin_invocation(InvocationMode::ScalarDecode, 1)
        .expect_err("overflow is pre-dispatch");
    assert_eq!(err.code, E_KV_OVERFLOW);
    assert_eq!(state.phase(), SequencePhase::Prefill);
    assert_eq!(state.valid_len(), 4);
    assert_eq!(state.sequence_epoch(), before.sequence_epoch());
    assert_eq!(state, before);
}

#[test]
fn stale_plan_commit_leaves_state_unchanged() {
    let mut state = fresh(8);
    let plan = state
        .begin_invocation(InvocationMode::Prefill, 3)
        .expect("plan");
    state.reset().expect("reset advances epoch");
    let before = state.clone();
    let err = state.commit(&plan).expect_err("stale epoch");
    assert_eq!(err.code, E_KV_STALE);
    assert_eq!(state, before);
}

#[test]
fn poison_rejects_reset_and_retry_allows_inspect_and_release() {
    let mut state = fresh(8);
    let prefill = state
        .begin_invocation(InvocationMode::Prefill, 4)
        .expect("prefill");
    state.commit(&prefill).expect("commit");
    let decode = state
        .begin_invocation(InvocationMode::ScalarDecode, 1)
        .expect("decode");
    let epoch = state.sequence_epoch();
    let valid_len = state.valid_len();

    let phase = state
        .poison(&decode, FailureStage::Dispatch)
        .expect("poison");
    assert_eq!(
        phase,
        SequencePhase::Poisoned {
            epoch,
            failure_stage: FailureStage::Dispatch,
        }
    );
    assert_eq!(state.valid_len(), valid_len);
    assert_eq!(state.sequence_epoch(), epoch);

    let inspection = state.inspect();
    assert_eq!(inspection.phase, phase);
    assert_eq!(inspection.valid_len, valid_len);
    assert_eq!(inspection.sequence_epoch, epoch);
    assert_eq!(
        inspection.poisoned_invocation.as_ref().map(|p| p.mode()),
        Some(InvocationMode::ScalarDecode)
    );
    assert!(!inspection.released);

    let after_poison = state.clone();
    let reset_err = state.reset().expect_err("reset after poison");
    assert_eq!(reset_err.code, E_KV_POISONED);
    assert_eq!(state, after_poison);

    let retry_err = state
        .begin_invocation(InvocationMode::ScalarDecode, 1)
        .expect_err("retry after poison");
    assert_eq!(retry_err.code, E_KV_POISONED);
    assert_eq!(state, after_poison);

    let commit_err = state.commit(&decode).expect_err("commit after poison");
    assert_eq!(commit_err.code, E_KV_POISONED);
    assert_eq!(state, after_poison);

    state.release().expect("release after poison");
    assert!(state.released());
    assert_eq!(
        state.phase(),
        SequencePhase::Poisoned {
            epoch,
            failure_stage: FailureStage::Dispatch,
        }
    );
    let released = state.inspect();
    assert!(released.released);
    assert_eq!(released.phase, state.phase());

    let after_release = state.clone();
    let begin_err = state
        .begin_invocation(InvocationMode::Prefill, 1)
        .expect_err("begin after release");
    assert_eq!(begin_err.code, E_KV_RELEASED);
    let reset_err = state.reset().expect_err("reset after release");
    assert_eq!(reset_err.code, E_KV_RELEASED);
    let rerelease = state.release().expect_err("double release");
    assert_eq!(rerelease.code, E_KV_RELEASED);
    assert_eq!(state.inspect().valid_len, after_release.valid_len());
}

#[test]
fn poison_cursor_upload_records_failure_stage() {
    let mut state = fresh(8);
    let plan = state
        .begin_invocation(InvocationMode::Prefill, 2)
        .expect("plan");
    state
        .poison(&plan, FailureStage::CursorUpload)
        .expect("poison");
    match state.phase() {
        SequencePhase::Poisoned {
            epoch,
            failure_stage,
        } => {
            assert_eq!(epoch, 1);
            assert_eq!(failure_stage, FailureStage::CursorUpload);
        }
        other => panic!("expected poison, got {other:?}"),
    }
}

#[test]
fn release_from_fresh_allows_inspect_only() {
    let mut state = fresh(4);
    state.release().expect("release");
    assert!(state.inspect().released);
    let err = state.reset().expect_err("reset released");
    assert_eq!(err.code, E_KV_RELEASED);
}

#[test]
fn u32_capacity_max_decode_overflows_fail_closed() {
    let mut full = fresh(u32::MAX);
    commit_prefill(&mut full, u32::MAX);
    let before = full.clone();
    let err = full
        .begin_invocation(InvocationMode::ScalarDecode, 1)
        .expect_err("MAX+1");
    assert_eq!(err.code, E_KV_OVERFLOW);
    assert_eq!(full, before);
}

#[test]
fn logical_reset_is_o1_no_cache_clear_zero_fill_or_upload() {
    let mut state = fresh(8);
    commit_prefill(&mut state, 5);
    commit_decode(&mut state);
    let previous_epoch = state.sequence_epoch();
    let previous_valid_len = state.valid_len();
    let capacity = state.capacity();

    let receipt = state.logical_reset().expect("logical reset");
    assert_eq!(receipt.previous_epoch, previous_epoch);
    assert_eq!(receipt.sequence_epoch, previous_epoch + 1);
    assert_eq!(receipt.previous_valid_len, previous_valid_len);
    assert_eq!(receipt.valid_len, 0);
    assert_eq!(receipt.capacity, capacity);
    assert!(!receipt.cache_cleared);
    assert!(!receipt.buffers_zero_filled);
    assert_eq!(receipt.uploads, 0);
    assert_eq!(state.valid_len(), 0);
    assert_eq!(state.capacity(), capacity);
    assert_eq!(state.phase(), SequencePhase::Fresh);
    assert_eq!(state.sequence_epoch(), previous_epoch + 1);
}

#[test]
fn replay_after_reset_drives_a_new_valid_prefix() {
    let mut state = fresh(8);
    commit_prefill(&mut state, 4);
    commit_decode(&mut state);
    assert_eq!(state.valid_len(), 5);
    let epoch = state.logical_reset().expect("reset").sequence_epoch;

    let replay = commit_prefill(&mut state, 2);
    assert_eq!(replay.prefix_before, 0);
    assert_eq!(replay.query_rows, 2);
    assert_eq!(replay.valid_len_after, 2);
    assert_eq!(state.valid_len(), 2);
    assert_eq!(state.sequence_epoch(), epoch);
    assert_eq!(state.phase(), SequencePhase::Prefill);

    let decode = commit_decode(&mut state);
    assert_eq!(decode.prefix_before, 2);
    assert_eq!(state.valid_len(), 3);
}

#[test]
fn pre_dispatch_transaction_abort_leaves_state_unchanged() {
    let mut state = fresh(8);
    commit_prefill(&mut state, 3);
    let before = state.clone();
    let tx = state
        .begin_transaction(InvocationMode::ScalarDecode, 1)
        .expect("admitted");
    assert!(tx.is_pre_dispatch());
    assert_eq!(
        state.fail(&tx).expect("pre-dispatch abort"),
        FailureOutcome::Unchanged
    );
    assert_eq!(state, before);

    let overflow = state
        .begin_transaction(InvocationMode::Prefill, 1)
        .expect_err("prefill after commit is pre-dispatch");
    assert_eq!(overflow.code, E_KV_PHASE);
    assert_eq!(state, before);
}

#[test]
fn possible_partial_mutation_poisons_and_rejects_reset_retry() {
    let mut state = fresh(8);
    commit_prefill(&mut state, 3);
    let mut tx = state
        .begin_transaction(InvocationMode::ScalarDecode, 1)
        .expect("admitted");
    tx.record_possible_mutation(FailureStage::Dispatch);
    let epoch = state.sequence_epoch();
    let valid_len = state.valid_len();
    let before_cursor = state.inspect();

    let outcome = state.fail(&tx).expect("unproven mutation poisons");
    assert_eq!(
        outcome,
        FailureOutcome::Poisoned {
            epoch,
            failure_stage: FailureStage::Dispatch,
        }
    );
    assert_eq!(
        state.phase(),
        SequencePhase::Poisoned {
            epoch,
            failure_stage: FailureStage::Dispatch,
        }
    );
    assert_eq!(state.valid_len(), valid_len);
    assert_eq!(state.inspect().valid_len, before_cursor.valid_len);
    assert_eq!(state.inspect().last_commit, before_cursor.last_commit);

    let poisoned = state.clone();
    let reset_err = state.logical_reset().expect_err("reset after poison");
    assert_eq!(reset_err.code, E_KV_POISONED);
    let retry_err = state
        .begin_transaction(InvocationMode::ScalarDecode, 1)
        .expect_err("retry after poison");
    assert_eq!(retry_err.code, E_KV_POISONED);
    assert_eq!(state, poisoned);

    state.release().expect("release after poison");
    assert!(state.released());
    let inspect = state.inspect();
    assert!(inspect.released);
    assert_eq!(inspect.phase, state.phase());
}

#[test]
fn abort_pre_dispatch_rejected_after_possible_mutation() {
    let mut state = fresh(8);
    let mut tx = state
        .begin_transaction(InvocationMode::Prefill, 2)
        .expect("admitted");
    tx.record_possible_mutation(FailureStage::CursorUpload);
    let before = state.clone();
    let err = state.abort_pre_dispatch(&tx).expect_err("not pre-dispatch");
    assert_eq!(err.code, E_INVALID_ARGS);
    assert_eq!(state, before);
    state.fail(&tx).expect("poison path");
    match state.phase() {
        SequencePhase::Poisoned {
            failure_stage: FailureStage::CursorUpload,
            ..
        } => {}
        other => panic!("expected CursorUpload poison, got {other:?}"),
    }
}
