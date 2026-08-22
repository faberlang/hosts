//! SV-E2: verification session transaction over the KV-D machine.
//!
//! Candidate rows `[L, L+k)` are an uncommitted view over fixed-capacity
//! storage; commit advances by exactly the accepted `r ≤ k`; `r=0` abort
//! leaves committed state byte-identical; post-dispatch failure poisons
//! (D4 law, no rollback receipt).

#[path = "../src/composite_host/inference_state.rs"]
mod inference_state;

use inference_state::{
    CandidateRows, FailureOutcome, FailureStage, InferenceSessionState, InvocationMode,
    SequencePhase, VerificationCommit, E_INVALID_ARGS, E_KV_OVERFLOW, E_KV_PHASE, E_KV_POISONED,
    E_KV_STALE,
};

fn at_l(capacity: u32, prefix: u32) -> InferenceSessionState {
    let mut state = InferenceSessionState::new(capacity).expect("capacity >= 1");
    let plan = state
        .begin_invocation(InvocationMode::Prefill, prefix)
        .expect("prefill admits");
    state.commit(&plan).expect("prefill commits");
    state
}

fn begin_verification(
    state: &InferenceSessionState,
    k: u32,
) -> inference_state::InvocationTransaction {
    state
        .begin_transaction(InvocationMode::Verification, k)
        .expect("verification admits")
}

#[test]
fn verification_admits_at_nonzero_l_with_k_over_one() {
    let state = at_l(16, 5);
    let tx = begin_verification(&state, 4);
    assert_eq!(tx.mode(), InvocationMode::Verification);
    let coordinates = tx.coordinates();
    assert_eq!(coordinates.prefix_before, 5);
    assert_eq!(coordinates.query_rows, 4);
    assert_eq!(coordinates.write_position, 5);
    assert_eq!(coordinates.valid_len_after, 9);
    assert_eq!(coordinates.capacity, 16);
    assert_eq!(coordinates.query_start, 5);
    assert_eq!(tx.sequence_epoch(), state.sequence_epoch());
    assert!(tx.is_pre_dispatch());
}

#[test]
fn verification_rejects_k_one_fresh_and_zero_l() {
    let state = at_l(16, 5);
    let err = state
        .begin_invocation(InvocationMode::Verification, 1)
        .expect_err("k=1 is scalar decode");
    assert_eq!(err.code, E_INVALID_ARGS);

    let fresh = InferenceSessionState::new(16).expect("capacity >= 1");
    let err = fresh
        .begin_invocation(InvocationMode::Verification, 4)
        .expect_err("fresh sequence cannot verify");
    assert_eq!(err.code, E_KV_PHASE);

    let zero_l = InferenceSessionState::new(16).expect("capacity >= 1");
    // A committed zero-length prefill is still not a nonzero L.
    let plan = zero_l
        .begin_invocation(InvocationMode::Prefill, 0)
        .err()
        .map(|_| ());
    assert!(plan.is_some(), "zero-row prefill is malformed");
}

#[test]
fn verification_rejects_overflow_beyond_capacity() {
    let state = at_l(8, 7);
    let err = state
        .begin_invocation(InvocationMode::Verification, 2)
        .expect_err("7+2 exceeds capacity 8");
    assert_eq!(err.code, E_KV_OVERFLOW);
}

#[test]
fn candidate_rows_view_has_no_old_prefix_copy() {
    let state = at_l(16, 5);
    let tx = begin_verification(&state, 4);
    let view: CandidateRows = tx.candidate_rows();
    assert_eq!(
        view,
        CandidateRows {
            start: 5,
            rows: 4,
            capacity: 16
        }
    );
    assert_eq!(view.position(0), Ok(5));
    assert_eq!(view.position(3), Ok(8));
    assert!(view.position(4).is_err());
    // The committed prefix [0, L) is not part of the scratch view.
    assert!(view.position(4).unwrap_err().code == E_INVALID_ARGS);
    // One allocation: view start == committed length, capacity unchanged.
    assert_eq!(view.start, state.valid_len());
    assert_eq!(view.capacity, state.capacity());
}

#[test]
fn commit_full_r_equals_k() {
    let mut state = at_l(16, 5);
    let tx = begin_verification(&state, 4);
    let outcome = state.commit_verification(&tx, 4).expect("r=k commits");
    match outcome {
        VerificationCommit::Committed {
            accepted_rows,
            committed,
        } => {
            assert_eq!(accepted_rows, 4);
            assert_eq!(committed.valid_len_after, 9);
            assert_eq!(committed.query_rows, 4);
        }
        VerificationCommit::AbortedZero => panic!("r=k must commit"),
    }
    assert_eq!(state.valid_len(), 9);
    assert_eq!(state.phase(), SequencePhase::Decode);
    assert_eq!(state.sequence_epoch(), 1);
    assert_eq!(
        state.inspect().last_commit.map(|c| c.valid_len_after),
        Some(9)
    );
}

#[test]
fn commit_partial_r_below_k_advances_exactly_r() {
    let mut state = at_l(16, 5);
    let tx = begin_verification(&state, 4);
    let outcome = state.commit_verification(&tx, 2).expect("0<r<k commits");
    match outcome {
        VerificationCommit::Committed {
            accepted_rows,
            committed,
        } => {
            assert_eq!(accepted_rows, 2);
            assert_eq!(committed.valid_len_after, 7);
            assert_eq!(committed.query_rows, 2);
        }
        VerificationCommit::AbortedZero => panic!("r=2 must commit"),
    }
    assert_eq!(state.valid_len(), 7);
    assert_eq!(state.phase(), SequencePhase::Decode);
    assert_eq!(state.inspect().last_commit.map(|c| c.query_rows), Some(2));
}

#[test]
fn r_exceeding_k_rejects_without_mutation() {
    let mut state = at_l(16, 5);
    let tx = begin_verification(&state, 4);
    let err = state.commit_verification(&tx, 5).expect_err("r>k rejects");
    assert_eq!(err.code, E_INVALID_ARGS);
    assert_eq!(state.valid_len(), 5);
    assert_eq!(state.phase(), SequencePhase::Prefill);
    assert_eq!(state.inspect().last_commit.map(|c| c.query_rows), Some(5));
}

#[test]
fn r_zero_abort_is_byte_identical_and_reusable() {
    let mut state = at_l(16, 5);
    let before = state.inspect();
    let tx = begin_verification(&state, 4);
    let outcome = state
        .commit_verification(&tx, 0)
        .expect("r=0 aborts cleanly");
    assert_eq!(outcome, VerificationCommit::AbortedZero);
    assert_eq!(state.inspect(), before);
    // Reusable: a fresh admission after the r=0 abort succeeds and commits.
    let tx = begin_verification(&state, 4);
    state
        .commit_verification(&tx, 3)
        .expect("fresh admission after r=0 commits");
    assert_eq!(state.valid_len(), 8);
}

#[test]
fn pre_dispatch_abort_leaves_machine_unchanged_and_reusable() {
    let mut state = at_l(16, 5);
    let before = state.inspect();
    let mut tx = begin_verification(&state, 4);
    assert_eq!(state.fail(&tx), Ok(FailureOutcome::Unchanged));
    assert_eq!(state.inspect(), before);
    // Reusable: fresh admission after pre-dispatch abort.
    tx = begin_verification(&state, 4);
    state
        .commit_verification(&tx, 1)
        .expect("admission after abort commits");
    assert_eq!(state.valid_len(), 6);
}

#[test]
fn pre_dispatch_abort_rejects_after_possible_mutation() {
    let mut state = at_l(16, 5);
    let mut tx = begin_verification(&state, 4);
    tx.record_possible_mutation(FailureStage::Dispatch);
    let err = state
        .abort_pre_dispatch(&tx)
        .expect_err("post-dispatch must poison, not abort");
    assert_eq!(err.code, E_INVALID_ARGS);
}

#[test]
fn post_dispatch_failure_poisons_at_every_stage() {
    for stage in [
        FailureStage::CursorUpload,
        FailureStage::Dispatch,
        FailureStage::Sync,
        FailureStage::Readback,
    ] {
        let mut state = at_l(16, 5);
        let mut tx = begin_verification(&state, 4);
        tx.record_possible_mutation(stage);
        match state.fail(&tx).expect("post-dispatch failure closes") {
            FailureOutcome::Poisoned {
                epoch,
                failure_stage,
            } => {
                assert_eq!(epoch, 1);
                assert_eq!(failure_stage, stage);
            }
            FailureOutcome::Unchanged => panic!("{stage:?} must poison, not stay unchanged"),
        }
        // No rollback claim: committed length frozen, poison terminal.
        assert_eq!(state.valid_len(), 5);
        assert_eq!(
            state.phase(),
            SequencePhase::Poisoned {
                epoch: 1,
                failure_stage: stage
            }
        );
        let err = state
            .begin_invocation(InvocationMode::ScalarDecode, 1)
            .expect_err("poisoned sequence rejects new work");
        assert_eq!(err.code, E_KV_POISONED);
        let err = state
            .logical_reset()
            .expect_err("no rollback proof, no reset");
        assert_eq!(err.code, E_KV_POISONED);
    }
}

#[test]
fn stale_epoch_and_cursor_reject() {
    let mut state = at_l(16, 5);
    let tx = begin_verification(&state, 4);
    state.logical_reset().expect("reset advances epoch");
    let err = state
        .commit_verification(&tx, 2)
        .expect_err("pre-reset plan is stale");
    assert_eq!(err.code, E_KV_STALE);
    // Fresh-sequence verification still rejects (epoch aside, L=0).
    let err = state
        .begin_invocation(InvocationMode::Verification, 4)
        .expect_err("no committed prefill after reset");
    assert_eq!(err.code, E_KV_PHASE);

    // Stale cursor: commit then reuse the old transaction.
    let mut state = at_l(16, 5);
    let tx = begin_verification(&state, 4);
    state.commit_verification(&tx, 4).expect("first commit");
    let tx2 = begin_verification(&state, 2);
    let err = state
        .commit_verification(&tx, 1)
        .expect_err("old cursor is stale");
    assert_eq!(err.code, E_KV_STALE);
    state
        .commit_verification(&tx2, 2)
        .expect("live plan still commits");
}

#[test]
fn plain_commit_rejects_wholesale_k_for_verification() {
    let mut state = at_l(16, 5);
    let tx = begin_verification(&state, 4);
    let err = state
        .commit_transaction(&tx)
        .expect_err("verification must commit exactly r");
    assert_eq!(err.code, E_INVALID_ARGS);
    assert_eq!(state.valid_len(), 5);
}

#[test]
fn commit_verification_rejects_non_verification_transaction() {
    let mut state = at_l(16, 5);
    let tx = state
        .begin_transaction(InvocationMode::ScalarDecode, 1)
        .expect("decode admits");
    let err = state
        .commit_verification(&tx, 1)
        .expect_err("exact-r is verification-only");
    assert_eq!(err.code, E_INVALID_ARGS);
}

#[test]
fn reset_and_teardown_follow_kv_d() {
    let mut state = at_l(16, 5);
    let tx = begin_verification(&state, 4);
    state.commit_verification(&tx, 2).expect("partial commit");
    assert_eq!(state.valid_len(), 7);
    let receipt = state.logical_reset().expect("reset");
    assert_eq!(receipt.previous_valid_len, 7);
    assert_eq!(receipt.valid_len, 0);
    assert_eq!(receipt.previous_epoch, 1);
    assert_eq!(receipt.sequence_epoch, 2);
    assert_eq!(receipt.capacity, 16);
    assert_eq!(state.phase(), SequencePhase::Fresh);
    // After reset, verification needs a new committed prefill (not inferred).
    let err = state
        .begin_invocation(InvocationMode::Verification, 4)
        .expect_err("reset clears L");
    assert_eq!(err.code, E_KV_PHASE);

    // Teardown: release is legal after poison from a failed verification.
    let mut state = at_l(16, 5);
    let mut tx = begin_verification(&state, 4);
    tx.record_possible_mutation(FailureStage::Sync);
    state.fail(&tx).expect("poison");
    state.release().expect("release legal after poison");
    let inspection = state.inspect();
    assert!(inspection.released);
    assert!(matches!(inspection.phase, SequencePhase::Poisoned { .. }));
}

#[test]
fn sequential_transactions_chain_after_partial_commit() {
    let mut state = at_l(32, 4);
    let tx = begin_verification(&state, 4);
    state.commit_verification(&tx, 2).expect("first");
    let tx = begin_verification(&state, 4);
    state.commit_verification(&tx, 4).expect("second");
    let tx = begin_verification(&state, 4);
    state.commit_verification(&tx, 0).expect("abort");
    let tx = state
        .begin_transaction(InvocationMode::ScalarDecode, 1)
        .expect("decode admits");
    state.commit_transaction(&tx).expect("decode commits");
    assert_eq!(state.valid_len(), 11);
    assert_eq!(state.phase(), SequencePhase::Decode);
}
