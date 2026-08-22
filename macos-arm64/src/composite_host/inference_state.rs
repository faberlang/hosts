//! KV-D D1/D4: model-session sequence state machine.
//!
//! Pure logical cursor/phase ownership. No device allocation, upload, launch,
//! or cache clear. Coordinate facts stay distinct (KV-L1/KV-L2). Reset is
//! logical (KV-L8). Failure is fail-closed (KV-L9). D4 adds the invocation
//! transaction and the O(1) reset receipt; rollback is not proven, so a
//! possible partial device mutation poisons.
//!
//! Parent registration is a private `mod inference_state` in
//! `composite_host.rs`; this unit cannot re-export it.

#![allow(dead_code)]

use std::fmt;

/// Capacity+1 / `valid_len_after > capacity` is rejected before mutation.
pub const E_KV_OVERFLOW: &str = "E_KV_OVERFLOW";
/// Mode or phase is illegal for the current sequence (never inferred).
pub const E_KV_PHASE: &str = "E_KV_PHASE";
/// Poisoned sequence: only inspect and release are legal.
pub const E_KV_POISONED: &str = "E_KV_POISONED";
/// Released sequence: only inspect remains legal.
pub const E_KV_RELEASED: &str = "E_KV_RELEASED";
/// Commit/poison plan does not match the live cursor or epoch.
pub const E_KV_STALE: &str = "E_KV_STALE";
/// Malformed request (zero capacity, zero rows, decode M≠1, query-row range).
pub const E_INVALID_ARGS: &str = "E_INVALID_ARGS";

/// Fail-closed session-state error. Never retryable: a rejected operation
/// leaves the machine unchanged (KV-L9 pre-dispatch).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionError {
    pub code: &'static str,
    pub message: String,
}

impl SessionError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    fn overflow(prefix_before: u32, query_rows: u32, capacity: u32) -> Self {
        Self::new(
            E_KV_OVERFLOW,
            format!(
                "KV overflow: prefix_before={prefix_before} + query_rows={query_rows} exceeds capacity={capacity}"
            ),
        )
    }

    fn phase(message: impl Into<String>) -> Self {
        Self::new(E_KV_PHASE, message)
    }

    fn poisoned(epoch: u32, failure_stage: FailureStage) -> Self {
        Self::new(
            E_KV_POISONED,
            format!(
                "sequence epoch {epoch} is poisoned at {failure_stage:?}; only receipt inspection and release are legal"
            ),
        )
    }

    fn released() -> Self {
        Self::new(
            E_KV_RELEASED,
            "sequence is released; only receipt inspection is legal",
        )
    }

    fn stale(message: impl Into<String>) -> Self {
        Self::new(E_KV_STALE, message)
    }

    fn invalid_args(message: impl Into<String>) -> Self {
        Self::new(E_INVALID_ARGS, message)
    }
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for SessionError {}

/// Explicit invocation program. Never inferred from sequence length (KV-D).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvocationMode {
    /// Prefill M=T, `prefix_before = 0`.
    Prefill,
    /// Scalar decode M=1, `prefix_before = L`.
    ScalarDecode,
    /// Suffix verification M=k>1 at committed nonzero `prefix_before = L`
    /// (SV-E2). Candidate rows `[L, L+k)` are an uncommitted view; commit
    /// advances by exactly the accepted `r ≤ k` (never `k`).
    Verification,
}

/// Post-dispatch stage at which a possible partial device mutation occurred.
/// Pre-dispatch admission is not a poison stage: it never mutates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureStage {
    /// Invocation-state cursor upload after admission.
    CursorUpload,
    /// Device dispatch after the cursor upload.
    Dispatch,
    /// Synchronization after dispatch.
    Sync,
    /// Output readback after dispatch.
    Readback,
}

/// Sequence lifecycle. Fresh → prefill → decode; poison is terminal except
/// inspect + release. Rollback is not proven in D1/D4, so poison rejects
/// reset and retry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequencePhase {
    Fresh,
    Prefill,
    Decode,
    Poisoned {
        epoch: u32,
        failure_stage: FailureStage,
    },
}

/// Distinct cursor facts for one invocation (KV-L1). Not a catch-all `seq_len`.
///
/// Arithmetic (KV-L2): `write_position = prefix_before`,
/// `valid_len_after = prefix_before + query_rows`, query row `i` attends
/// `[0, prefix_before + i + 1)`. `query_start` is the absolute position of
/// query row 0 (contiguous append: equal in value to `prefix_before`,
/// distinct as a fact).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CursorFacts {
    pub prefix_before: u32,
    pub query_rows: u32,
    pub write_position: u32,
    pub valid_len_after: u32,
    pub capacity: u32,
    pub query_start: u32,
}

impl CursorFacts {
    /// Exclusive end of the causal window for query row `i` (KV-L2).
    pub fn causal_end_exclusive(&self, query_row: u32) -> Result<u32, SessionError> {
        if query_row >= self.query_rows {
            return Err(SessionError::invalid_args(format!(
                "query row {query_row} is outside query_rows={}",
                self.query_rows
            )));
        }
        self.prefix_before
            .checked_add(query_row)
            .and_then(|position| position.checked_add(1))
            .ok_or_else(|| {
                SessionError::overflow(self.prefix_before, self.query_rows, self.capacity)
            })
    }
}

/// Admitted invocation that has not yet committed or poisoned.
///
/// Planning is pure ([`InferenceSessionState::begin_invocation`]); mutation
/// happens only on [`InferenceSessionState::commit`] or
/// [`InferenceSessionState::poison`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedInvocation {
    mode: InvocationMode,
    coordinates: CursorFacts,
    sequence_epoch: u32,
}

/// Uncommitted candidate-row view over the fixed-capacity storage (SV-E2).
///
/// Rows `[start, start + rows)` of the one committed allocation; there is no
/// old-prefix copy on any path. The view is scratch until
/// [`InferenceSessionState::commit_verification`] admits exactly `r` rows or
/// the transaction aborts/poisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateRows {
    pub start: u32,
    pub rows: u32,
    pub capacity: u32,
}

impl CandidateRows {
    /// Absolute position of candidate row `i` (`i < rows`).
    pub fn position(&self, row: u32) -> Result<u32, SessionError> {
        if row >= self.rows {
            return Err(SessionError::invalid_args(format!(
                "candidate row {row} is outside rows={}",
                self.rows
            )));
        }
        Ok(self.start + row)
    }
}

/// Outcome of [`InferenceSessionState::commit_verification`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationCommit {
    /// `r = 0` abort: committed length/phase/epoch/cursor are byte-identical.
    AbortedZero,
    /// Committed exactly `accepted_rows ≤ k`; `valid_len` advanced by `r`.
    Committed {
        accepted_rows: u32,
        committed: CursorFacts,
    },
}

/// Open invocation transaction (D4).
///
/// Admission is pure. The machine mutates only on commit or on a
/// post-dispatch poison. `possible_mutation == None` is pre-dispatch
/// (KV-L9 unchanged). `Some(stage)` means a device write may have started;
/// D4 does not prove rollback, so [`InferenceSessionState::fail`] poisons.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvocationTransaction {
    plan: PlannedInvocation,
    possible_mutation: Option<FailureStage>,
}

impl InvocationTransaction {
    #[must_use]
    pub fn plan(&self) -> &PlannedInvocation {
        &self.plan
    }

    #[must_use]
    pub fn mode(&self) -> InvocationMode {
        self.plan.mode
    }

    #[must_use]
    pub fn coordinates(&self) -> CursorFacts {
        self.plan.coordinates
    }

    #[must_use]
    pub fn sequence_epoch(&self) -> u32 {
        self.plan.sequence_epoch
    }

    #[must_use]
    pub fn possible_mutation(&self) -> Option<FailureStage> {
        self.possible_mutation
    }

    #[must_use]
    pub fn is_pre_dispatch(&self) -> bool {
        self.possible_mutation.is_none()
    }

    /// Candidate scratch rows `[L, L+k)` over the fixed-capacity storage
    /// (SV-E2). No old-prefix copy: the view aliases committed storage past
    /// the committed prefix and is uncommitted until exact-`r` commit.
    #[must_use]
    pub fn candidate_rows(&self) -> CandidateRows {
        let coordinates = self.plan.coordinates;
        CandidateRows {
            start: coordinates.query_start,
            rows: coordinates.query_rows,
            capacity: coordinates.capacity,
        }
    }

    /// Record that device work at `stage` may have mutated the cache.
    /// Does not mutate the sequence machine.
    pub fn record_possible_mutation(&mut self, stage: FailureStage) {
        self.possible_mutation = Some(stage);
    }
}

/// O(1) logical-reset receipt (KV-L8). Capacity is preserved. Cache clear,
/// buffer zero-fill, and upload are not on this path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResetReceipt {
    pub previous_epoch: u32,
    pub sequence_epoch: u32,
    pub previous_valid_len: u32,
    pub valid_len: u32,
    pub capacity: u32,
    pub cache_cleared: bool,
    pub buffers_zero_filled: bool,
    pub uploads: u32,
}

/// Outcome of [`InferenceSessionState::fail`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureOutcome {
    /// Pre-dispatch abort: the machine was not mutated (KV-L9).
    Unchanged,
    /// Possible partial device write without proven rollback.
    Poisoned {
        epoch: u32,
        failure_stage: FailureStage,
    },
}

impl PlannedInvocation {
    #[must_use]
    pub fn mode(&self) -> InvocationMode {
        self.mode
    }

    #[must_use]
    pub fn coordinates(&self) -> CursorFacts {
        self.coordinates
    }

    #[must_use]
    pub fn sequence_epoch(&self) -> u32 {
        self.sequence_epoch
    }
}

/// Observable sequence facts. Legal in every phase, including poison and
/// after release (KV-L9 receipt inspection).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionInspection {
    pub phase: SequencePhase,
    pub capacity: u32,
    pub valid_len: u32,
    pub sequence_epoch: u32,
    pub last_commit: Option<CursorFacts>,
    pub poisoned_invocation: Option<PlannedInvocation>,
    pub released: bool,
}

/// Logical model-session sequence machine: capacity, committed valid length,
/// epoch, and phase. No device handles or residency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceSessionState {
    capacity: u32,
    valid_len: u32,
    sequence_epoch: u32,
    phase: SequencePhase,
    released: bool,
    last_commit: Option<CursorFacts>,
    poisoned_invocation: Option<PlannedInvocation>,
}

impl InferenceSessionState {
    /// Fresh sequence: `valid_len = 0`, epoch 1, fixed capacity.
    pub fn new(capacity: u32) -> Result<Self, SessionError> {
        if capacity == 0 {
            return Err(SessionError::invalid_args(
                "sequence capacity must be at least 1",
            ));
        }
        Ok(Self {
            capacity,
            valid_len: 0,
            sequence_epoch: 1,
            phase: SequencePhase::Fresh,
            released: false,
            last_commit: None,
            poisoned_invocation: None,
        })
    }

    #[must_use]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    #[must_use]
    pub fn valid_len(&self) -> u32 {
        self.valid_len
    }

    #[must_use]
    pub fn sequence_epoch(&self) -> u32 {
        self.sequence_epoch
    }

    #[must_use]
    pub fn phase(&self) -> SequencePhase {
        self.phase
    }

    #[must_use]
    pub fn released(&self) -> bool {
        self.released
    }

    /// Receipt inspection: always legal, including after poison or release.
    #[must_use]
    pub fn inspect(&self) -> SessionInspection {
        SessionInspection {
            phase: self.phase,
            capacity: self.capacity,
            valid_len: self.valid_len,
            sequence_epoch: self.sequence_epoch,
            last_commit: self.last_commit,
            poisoned_invocation: self.poisoned_invocation.clone(),
            released: self.released,
        }
    }

    /// Pre-dispatch admission. Fail-closed: the machine is not mutated.
    ///
    /// Mode is the caller's explicit program selection. Overflow
    /// (`valid_len_after > capacity`) is rejected here, before commit.
    pub fn begin_invocation(
        &self,
        mode: InvocationMode,
        query_rows: u32,
    ) -> Result<PlannedInvocation, SessionError> {
        self.ensure_mutable()?;
        if query_rows == 0 {
            return Err(SessionError::invalid_args("query_rows must be at least 1"));
        }
        match mode {
            InvocationMode::Prefill => {
                if self.phase != SequencePhase::Fresh || self.valid_len != 0 {
                    return Err(SessionError::phase(
                        "prefill requires a fresh sequence with prefix_before=0",
                    ));
                }
            }
            InvocationMode::ScalarDecode => {
                if query_rows != 1 {
                    return Err(SessionError::invalid_args(
                        "scalar decode is M=1; query_rows must be 1",
                    ));
                }
                match self.phase {
                    SequencePhase::Prefill | SequencePhase::Decode => {}
                    SequencePhase::Fresh => {
                        return Err(SessionError::phase(
                            "scalar decode requires a committed prefill; mode is not inferred from length",
                        ));
                    }
                    SequencePhase::Poisoned {
                        epoch,
                        failure_stage,
                    } => {
                        return Err(SessionError::poisoned(epoch, failure_stage));
                    }
                }
            }
            InvocationMode::Verification => {
                if query_rows < 2 {
                    return Err(SessionError::invalid_args(
                        "verification is M=k>1; use scalar decode for a single row",
                    ));
                }
                match self.phase {
                    SequencePhase::Prefill | SequencePhase::Decode => {}
                    SequencePhase::Fresh => {
                        return Err(SessionError::phase(
                            "verification requires a committed prefill at nonzero L; a fresh-prefill masquerade is rejected",
                        ));
                    }
                    SequencePhase::Poisoned {
                        epoch,
                        failure_stage,
                    } => {
                        return Err(SessionError::poisoned(epoch, failure_stage));
                    }
                }
                if self.valid_len == 0 {
                    return Err(SessionError::phase(
                        "verification requires a committed nonzero prefix_before=L",
                    ));
                }
            }
        }

        let prefix_before = self.valid_len;
        let valid_len_after = match prefix_before.checked_add(query_rows) {
            Some(after) if after <= self.capacity => after,
            Some(_) | None => {
                return Err(SessionError::overflow(
                    prefix_before,
                    query_rows,
                    self.capacity,
                ));
            }
        };
        // Contiguous append: write at the live prefix; query row 0 starts there.
        let coordinates = CursorFacts {
            prefix_before,
            query_rows,
            write_position: prefix_before,
            valid_len_after,
            capacity: self.capacity,
            query_start: prefix_before,
        };
        Ok(PlannedInvocation {
            mode,
            coordinates,
            sequence_epoch: self.sequence_epoch,
        })
    }

    /// Open a D4 transaction around pre-dispatch admission.
    pub fn begin_transaction(
        &self,
        mode: InvocationMode,
        query_rows: u32,
    ) -> Result<InvocationTransaction, SessionError> {
        Ok(InvocationTransaction {
            plan: self.begin_invocation(mode, query_rows)?,
            possible_mutation: None,
        })
    }

    /// Contiguous commit: valid length advances by exactly `query_rows`.
    ///
    /// Verification transactions must not commit `k` rows wholesale; they
    /// commit exactly `r ≤ k` via [`Self::commit_verification`].
    pub fn commit(&mut self, plan: &PlannedInvocation) -> Result<CursorFacts, SessionError> {
        self.ensure_mutable()?;
        self.ensure_plan_matches(plan)?;
        if plan.mode == InvocationMode::Verification {
            return Err(SessionError::invalid_args(
                "verification commits exactly r accepted rows, not k; use commit_verification",
            ));
        }
        let coordinates = plan.coordinates;
        self.valid_len = coordinates.valid_len_after;
        self.phase = match plan.mode {
            InvocationMode::Prefill => SequencePhase::Prefill,
            // Verification never reaches here: `commit` rejects it in favor
            // of `commit_verification`.
            InvocationMode::ScalarDecode | InvocationMode::Verification => SequencePhase::Decode,
        };
        self.last_commit = Some(coordinates);
        Ok(coordinates)
    }

    /// Commit the admitted transaction plan.
    pub fn commit_transaction(
        &mut self,
        tx: &InvocationTransaction,
    ) -> Result<CursorFacts, SessionError> {
        self.commit(&tx.plan)
    }

    /// Verification commit of exactly `accepted_rows = r ≤ k` (SV-E2).
    ///
    /// `r = 0` is an abort: committed length/phase/epoch/cursor stay
    /// byte-identical and the sequence stays reusable. `0 < r ≤ k` advances
    /// `valid_len` by exactly `r` and lands in `Decode`. Stale epoch or
    /// cursor coordinates reject before any mutation.
    pub fn commit_verification(
        &mut self,
        tx: &InvocationTransaction,
        accepted_rows: u32,
    ) -> Result<VerificationCommit, SessionError> {
        self.ensure_mutable()?;
        if tx.mode() != InvocationMode::Verification {
            return Err(SessionError::invalid_args(
                "commit_verification requires a Verification transaction",
            ));
        }
        self.ensure_plan_matches(&tx.plan)?;
        let coordinates = tx.plan.coordinates;
        if accepted_rows > coordinates.query_rows {
            return Err(SessionError::invalid_args(format!(
                "accepted_rows={accepted_rows} exceeds candidate query_rows={}",
                coordinates.query_rows
            )));
        }
        if accepted_rows == 0 {
            return Ok(VerificationCommit::AbortedZero);
        }
        let committed = CursorFacts {
            prefix_before: coordinates.prefix_before,
            query_rows: accepted_rows,
            write_position: coordinates.prefix_before,
            valid_len_after: coordinates.prefix_before + accepted_rows,
            capacity: coordinates.capacity,
            query_start: coordinates.prefix_before,
        };
        self.valid_len = committed.valid_len_after;
        self.phase = SequencePhase::Decode;
        self.last_commit = Some(committed);
        Ok(VerificationCommit::Committed {
            accepted_rows,
            committed,
        })
    }

    /// Possible partial device mutation: poison the sequence (KV-L9).
    ///
    /// D1/D4 do not prove rollback, so the only legal follow-ups are inspect
    /// and release. The committed cursor is left unchanged.
    pub fn poison(
        &mut self,
        plan: &PlannedInvocation,
        failure_stage: FailureStage,
    ) -> Result<SequencePhase, SessionError> {
        self.ensure_mutable()?;
        self.ensure_plan_matches(plan)?;
        let phase = SequencePhase::Poisoned {
            epoch: self.sequence_epoch,
            failure_stage,
        };
        self.phase = phase;
        self.poisoned_invocation = Some(plan.clone());
        Ok(phase)
    }

    /// Pre-dispatch abort (KV-L9): the machine is not mutated.
    ///
    /// Illegal after [`InvocationTransaction::record_possible_mutation`];
    /// that path must [`Self::fail`].
    pub fn abort_pre_dispatch(&self, tx: &InvocationTransaction) -> Result<(), SessionError> {
        self.ensure_mutable()?;
        self.ensure_plan_matches(&tx.plan)?;
        if let Some(stage) = tx.possible_mutation {
            return Err(SessionError::invalid_args(format!(
                "abort_pre_dispatch after possible mutation at {stage:?}; poison because rollback is not proven"
            )));
        }
        Ok(())
    }

    /// Possible partial mutation without proven rollback: poison (KV-L9).
    pub fn fail_unproven(
        &mut self,
        tx: &InvocationTransaction,
    ) -> Result<SequencePhase, SessionError> {
        let stage = tx.possible_mutation.ok_or_else(|| {
            SessionError::invalid_args(
                "fail_unproven requires a possible mutation stage; abort_pre_dispatch if still pre-dispatch",
            )
        })?;
        self.poison(&tx.plan, stage)
    }

    /// Close a failed transaction: pre-dispatch stays unchanged; a possible
    /// device write poisons because rollback is not proven.
    pub fn fail(&mut self, tx: &InvocationTransaction) -> Result<FailureOutcome, SessionError> {
        if tx.is_pre_dispatch() {
            self.abort_pre_dispatch(tx)?;
            Ok(FailureOutcome::Unchanged)
        } else {
            match self.fail_unproven(tx)? {
                SequencePhase::Poisoned {
                    epoch,
                    failure_stage,
                } => Ok(FailureOutcome::Poisoned {
                    epoch,
                    failure_stage,
                }),
                other => Err(SessionError::invalid_args(format!(
                    "expected poison after unproven mutation, got {other:?}"
                ))),
            }
        }
    }

    /// Logical prompt reset (KV-L8): valid length → 0, epoch advances,
    /// capacity preserved. No cache clear, zero-fill, or upload.
    pub fn logical_reset(&mut self) -> Result<ResetReceipt, SessionError> {
        self.ensure_mutable()?;
        let previous_epoch = self.sequence_epoch;
        let previous_valid_len = self.valid_len;
        let capacity = self.capacity;
        let next_epoch = self
            .sequence_epoch
            .checked_add(1)
            .ok_or_else(|| SessionError::invalid_args("sequence epoch overflow"))?;
        self.valid_len = 0;
        self.sequence_epoch = next_epoch;
        self.phase = SequencePhase::Fresh;
        self.last_commit = None;
        self.poisoned_invocation = None;
        Ok(ResetReceipt {
            previous_epoch,
            sequence_epoch: next_epoch,
            previous_valid_len,
            valid_len: 0,
            capacity,
            cache_cleared: false,
            buffers_zero_filled: false,
            uploads: 0,
        })
    }

    /// Logical prompt reset (KV-L8): valid length → 0, epoch advances,
    /// capacity preserved. No cache clear.
    pub fn reset(&mut self) -> Result<u32, SessionError> {
        Ok(self.logical_reset()?.sequence_epoch)
    }

    /// Terminal release. Legal after poison. Inspect remains legal.
    pub fn release(&mut self) -> Result<(), SessionError> {
        if self.released {
            return Err(SessionError::released());
        }
        self.released = true;
        Ok(())
    }

    fn ensure_mutable(&self) -> Result<(), SessionError> {
        if self.released {
            return Err(SessionError::released());
        }
        if let SequencePhase::Poisoned {
            epoch,
            failure_stage,
        } = self.phase
        {
            return Err(SessionError::poisoned(epoch, failure_stage));
        }
        Ok(())
    }

    fn ensure_plan_matches(&self, plan: &PlannedInvocation) -> Result<(), SessionError> {
        if plan.sequence_epoch != self.sequence_epoch {
            return Err(SessionError::stale(format!(
                "plan epoch {} does not match live epoch {}",
                plan.sequence_epoch, self.sequence_epoch
            )));
        }
        if plan.coordinates.prefix_before != self.valid_len {
            return Err(SessionError::stale(format!(
                "plan prefix_before {} does not match live valid_len {}",
                plan.coordinates.prefix_before, self.valid_len
            )));
        }
        if plan.coordinates.capacity != self.capacity {
            return Err(SessionError::stale(format!(
                "plan capacity {} does not match live capacity {}",
                plan.coordinates.capacity, self.capacity
            )));
        }
        Ok(())
    }
}
