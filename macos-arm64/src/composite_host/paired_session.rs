//! PPE-P1/P3/P4: one-runtime paired prefill/scalar-decode executor.
//!
//! A pair is prepared before its first invocation. The pair owns one mutable
//! device runtime and detaches the two program-owned states from the legacy
//! `ProgramSession` borrow while they are idle. A short-lived `ProgramSession`
//! reattaches each state only for the selected dispatch, so both programs use
//! the same runtime and the same semantic PerProgram owner. Reset is a D4
//! logical cursor/epoch update: K/V arenas and weights stay put, and the
//! normal path uploads zero cache bytes. Lifecycle receipt fields are derived
//! from driver-counter baselines taken at prepare, with per-program pool
//! warm-up subtracted.

use std::collections::{BTreeMap, BTreeSet};

use crate::composite_host::invocation_binding::{project_invocation_bindings, RopeConfig};
use crate::device_descriptor::{DeviceBufferLifetime, DeviceDescriptor, DeviceProgramLifetime};
use crate::device_execute::{DeviceExecuteInvocation, DeviceExecuteInvocationMode};
use crate::device_host::{DeviceRuntime, DeviceSession};
use crate::device_registry::DriverCounters;
use crate::kernel::{HostError, HostResult};

use super::inference_state::{
    FailureStage, InferenceSessionState, InvocationMode, InvocationTransaction, ResetReceipt,
    SequencePhase, SessionError, E_KV_STALE,
};
use super::session::{ProgramInner, ProgramSession};
use super::{DeviceExecutionReceipt, KvCacheTimingReceipt};

/// One prepared prefill/scalar-decode pair over one runtime and model owner.
pub struct PairedProgramSession<'host> {
    runtime: &'host mut DeviceRuntime,
    prefill: Option<ProgramInner>,
    decode: Option<ProgramInner>,
    prefill_descriptor: DeviceDescriptor,
    decode_descriptor: DeviceDescriptor,
    prompt_tokens: Vec<u32>,
    rope: RopeConfig,
    model_identity: String,
    session_identity: String,
    state: InferenceSessionState,
    reuses: usize,
    resets: usize,
    reset_cleared: usize,
    /// Driver-counter baselines at prepare for reload/realloc derivation.
    module_loads_at_prepare: usize,
    buffer_allocs_at_prepare: usize,
    prefill_per_execution_alloc_count: usize,
    decode_per_execution_alloc_count: usize,
    prefill_pool_warmed: bool,
    decode_pool_warmed: bool,
    /// Driver upload-counter baseline captured before once-init, so
    /// prepare-time HostProvided copies are included in the derived count.
    uploads_at_prepare: usize,
}

impl<'host> PairedProgramSession<'host> {
    /// Prepare both static programs and once-init their shared weights before
    /// the first invocation. Matching PerProgram resources are selected by
    /// the descriptor's carried semantic value identity, never by byte count.
    pub fn prepare(
        runtime: &'host mut DeviceRuntime,
        prefill: &DeviceDescriptor,
        decode: &DeviceDescriptor,
        prompt_tokens: Vec<u32>,
        rope: RopeConfig,
        weights: &BTreeMap<u32, Vec<f32>>,
        byte_weights: &BTreeMap<u32, super::DeviceByteBuffer>,
        model_identity: String,
        session_identity: String,
        device_name: String,
    ) -> HostResult<Self> {
        if model_identity.is_empty() || session_identity.is_empty() {
            return Err(HostError::invalid_args(
                "paired session requires non-empty model and session identities",
            ));
        }
        if prompt_tokens.is_empty() {
            return Err(HostError::invalid_args(
                "paired session requires a non-empty prefill prompt",
            ));
        }
        rope.validate_for_pair()?;
        validate_pair_descriptors(prefill, decode)?;
        let prefill_descriptor = prefill.clone();
        let decode_descriptor = decode.clone();
        let uploads_at_prepare = runtime.driver_counters().uploads;

        let mut prefill_session = ProgramSession::new(runtime, prefill, device_name.clone())?;
        if let Err(error) = prefill_session.init_params_with_weight_bytes(weights, byte_weights) {
            return Err(error);
        }
        let offer = prefill_session.shared_offer();
        let prefill = prefill_session.into_inner();

        let mut decode_session =
            match ProgramSession::new_with_share(runtime, decode, device_name, Some(&offer)) {
                Ok(session) => session,
                Err(error) => {
                    release_inner(runtime, prefill);
                    return Err(error);
                }
            };
        if let Err(error) = decode_session.init_params_with_weight_bytes(weights, byte_weights) {
            let decode = decode_session.into_inner();
            release_inner(runtime, decode);
            release_inner(runtime, prefill);
            return Err(error);
        }
        let decode = decode_session.into_inner();
        let state = InferenceSessionState::new(pair_sequence_capacity(prompt_tokens.len())?)
            .map_err(session_error)?;
        let counters = runtime.driver_counters();
        let prefill_per_execution_alloc_count = per_execution_alloc_count(&prefill_descriptor);
        let decode_per_execution_alloc_count = per_execution_alloc_count(&decode_descriptor);

        Ok(Self {
            runtime,
            prefill: Some(prefill),
            decode: Some(decode),
            prefill_descriptor,
            decode_descriptor,
            prompt_tokens,
            rope,
            model_identity,
            session_identity,
            state,
            reuses: 0,
            resets: 0,
            reset_cleared: 0,
            module_loads_at_prepare: counters.module_loads,
            buffer_allocs_at_prepare: counters.buffer_allocs,
            prefill_per_execution_alloc_count,
            decode_per_execution_alloc_count,
            prefill_pool_warmed: false,
            decode_pool_warmed: false,
            uploads_at_prepare,
        })
    }

    /// Dispatch one explicitly selected invocation through the corresponding
    /// prepared program. D4 admits the mode against the live cursor first.
    /// P2 then projects only the selected descriptor's dynamic inputs. Both
    /// of those steps are pre-dispatch (KV-L9): a rejection leaves the
    /// sequence unchanged. A device-side failure poisons because rollback
    /// is not proven.
    pub fn execute_invocation(
        &mut self,
        invocation: &DeviceExecuteInvocation,
    ) -> HostResult<DeviceExecutionReceipt> {
        let mode = invocation_mode(invocation.mode);
        let query_rows = invocation_query_rows(invocation, self.prompt_tokens.len())?;
        let mut tx = self
            .state
            .begin_transaction(mode, query_rows)
            .map_err(session_error)?;
        if let Err(error) = match_invocation_cursor(invocation, &tx) {
            drop(self.state.fail(&tx));
            return Err(error);
        }
        let descriptor = match invocation.mode {
            DeviceExecuteInvocationMode::Prefill => &self.prefill_descriptor,
            DeviceExecuteInvocationMode::ScalarDecode => &self.decode_descriptor,
        };
        let inputs = match project_invocation_bindings(
            descriptor,
            invocation,
            &self.prompt_tokens,
            self.rope,
        ) {
            Ok(inputs) => inputs,
            Err(error) => {
                drop(self.state.fail(&tx));
                return Err(error);
            }
        };
        tx.record_possible_mutation(FailureStage::Dispatch);
        let result = match invocation.mode {
            DeviceExecuteInvocationMode::Prefill => self.execute_prefill(&inputs),
            DeviceExecuteInvocationMode::ScalarDecode => self.execute_decode(&inputs),
        };
        match result {
            Ok(receipt) => match self.state.commit_transaction(&tx) {
                Ok(_) => {
                    self.reuses += 1;
                    Ok(receipt)
                }
                Err(error) => {
                    drop(self.state.fail(&tx));
                    drop(self.release_all());
                    Err(session_error(error))
                }
            },
            Err(error) => {
                drop(self.state.fail(&tx));
                drop(self.release_all());
                Err(error)
            }
        }
    }

    /// Logical reset (KV-L8): epoch advances, committed valid length returns
    /// to zero, K/V arenas and weights stay put. No zero-fill upload.
    /// Poisoned sequences reject reset with the D4 error shape because
    /// rollback is not proven.
    pub fn reset(&mut self) -> HostResult<ResetReceipt> {
        let receipt = self.state.logical_reset().map_err(session_error)?;
        self.resets = self.resets.saturating_add(1);
        self.reset_cleared = receipt.previous_valid_len as usize;
        Ok(receipt)
    }

    /// The latest F4H1 timing for the selected paired program.
    #[must_use]
    pub fn kv_cache_timing(&self, mode: DeviceExecuteInvocationMode) -> KvCacheTimingReceipt {
        let program = match mode {
            DeviceExecuteInvocationMode::Prefill => &self.prefill,
            DeviceExecuteInvocationMode::ScalarDecode => &self.decode,
        };
        program
            .as_ref()
            .map(ProgramInner::kv_cache_timing)
            .unwrap_or_else(KvCacheTimingReceipt::not_measured)
    }

    /// Number of successful prefill/decode dispatches.
    #[must_use]
    pub fn reuses(&self) -> usize {
        self.reuses
    }

    /// Successful logical resets through this pair.
    #[must_use]
    pub fn resets(&self) -> usize {
        self.resets
    }

    /// Rows logically retired by the last successful reset. Zero until a
    /// reset commits; never a zero-fill upload count.
    #[must_use]
    pub fn reset_cleared(&self) -> usize {
        self.reset_cleared
    }

    /// Live D4 sequence epoch.
    #[must_use]
    pub fn sequence_epoch(&self) -> u32 {
        self.state.sequence_epoch()
    }

    /// Live committed valid length.
    #[must_use]
    pub fn valid_len(&self) -> u32 {
        self.state.valid_len()
    }

    /// Live D4 sequence phase.
    #[must_use]
    pub fn phase(&self) -> SequencePhase {
        self.state.phase()
    }

    /// Shared module handle id plus PerProgram semantic identities. Stable
    /// across logical reset; used to prove the owner was not reallocated.
    pub fn shared_owner_identity(&mut self) -> HostResult<(u64, BTreeSet<u32>)> {
        let inner = self
            .prefill
            .take()
            .ok_or_else(|| HostError::internal("paired program session is closed"))?;
        let session = ProgramSession::from_inner(self.runtime, inner);
        let offer = session.shared_offer();
        let identity = (
            offer.module_handle.id,
            offer.buffers.keys().copied().collect(),
        );
        self.prefill = Some(session.into_inner());
        Ok(identity)
    }

    /// The explicitly carried model identity.
    #[must_use]
    pub fn model_identity(&self) -> &str {
        &self.model_identity
    }

    /// The explicitly carried sequence/session identity.
    #[must_use]
    pub fn session_identity(&self) -> &str {
        &self.session_identity
    }

    /// Number of declared physical kernels in both static programs.
    #[must_use]
    pub fn kernel_count(&self) -> usize {
        self.prefill_descriptor.kernels.len() + self.decode_descriptor.kernels.len()
    }

    /// Program graph identities in the stable control-receipt spelling.
    #[must_use]
    pub fn program_identities(&self) -> BTreeMap<String, String> {
        BTreeMap::from([
            (
                "prefill".to_owned(),
                self.prefill_descriptor.program_graph_hash(),
            ),
            (
                "scalar_decode".to_owned(),
                self.decode_descriptor.program_graph_hash(),
            ),
        ])
    }

    /// Live handles owned by the pair's runtime, including the shared model
    /// handles exactly once.
    #[must_use]
    pub fn live_handles(&self) -> usize {
        self.runtime.live_handle_count()
    }

    /// Raw driver lifecycle counters for the pair's shared runtime. P4 owns
    /// receipt derivation; P1 exposes this only for identity/lifecycle proof.
    #[must_use]
    pub fn driver_counters(&self) -> DriverCounters {
        self.runtime.driver_counters()
    }

    /// Module loads observed beyond the prepare-time baseline. 0 across
    /// reuses means the shared module was never reloaded.
    #[must_use]
    pub fn module_reloads(&self) -> usize {
        self.runtime
            .driver_counters()
            .module_loads
            .saturating_sub(self.module_loads_at_prepare)
    }

    /// Buffer allocations beyond prepare and the one-time per-program step
    /// pool warm-up. 0 across reuses means no PerProgram owner was reallocated.
    #[must_use]
    pub fn per_program_reallocs(&self) -> usize {
        self.runtime
            .driver_counters()
            .buffer_allocs
            .saturating_sub(self.buffer_allocs_at_prepare)
            .saturating_sub(self.pool_warmup_allocs())
    }

    /// HostProvided weight copies since the prepare baseline. The baseline is
    /// captured before once-init, so a clean prepare reports the copies that
    /// actually ran (shared identities copy once). Reuses must not move this.
    #[must_use]
    pub fn weight_uploads(&self) -> usize {
        self.runtime
            .driver_counters()
            .uploads
            .saturating_sub(self.uploads_at_prepare)
    }

    /// Extra module load through the pair's runtime. Derived `module_reloads`
    /// must move with the driver counter; a structural zero cannot.
    pub fn force_module_load(&mut self) -> HostResult<()> {
        let handle = self.runtime.load_module(b"p4-forced-module-reload")?;
        self.runtime.release(&handle)
    }

    /// Extra buffer allocation through the pair's runtime. Derived
    /// `per_program_reallocs` must move with the driver after pool warm-up
    /// subtraction.
    pub fn force_buffer_alloc(&mut self) -> HostResult<()> {
        let handle = self.runtime.alloc_bytes(16)?;
        self.runtime.release(&handle)
    }

    /// Extra HostProvided upload through the pair's runtime. Derived
    /// `weight_uploads` must move with the driver upload counter.
    pub fn force_weight_upload(&mut self) -> HostResult<()> {
        self.runtime.record_weight_upload();
        Ok(())
    }

    /// Release decode-owned handles first, then the shared prefill owner.
    /// Shared handles are released once by the prefill owner.
    pub fn teardown(mut self) -> HostResult<()> {
        self.release_all()
    }

    fn execute_prefill(
        &mut self,
        inputs: &BTreeMap<u32, Vec<f32>>,
    ) -> HostResult<DeviceExecutionReceipt> {
        let result = execute_inner(&mut self.runtime, &mut self.prefill, inputs);
        if result.is_ok() {
            self.prefill_pool_warmed = true;
        }
        result
    }

    fn execute_decode(
        &mut self,
        inputs: &BTreeMap<u32, Vec<f32>>,
    ) -> HostResult<DeviceExecutionReceipt> {
        let result = execute_inner(&mut self.runtime, &mut self.decode, inputs);
        if result.is_ok() {
            self.decode_pool_warmed = true;
        }
        result
    }

    fn pool_warmup_allocs(&self) -> usize {
        usize::from(self.prefill_pool_warmed) * self.prefill_per_execution_alloc_count
            + usize::from(self.decode_pool_warmed) * self.decode_per_execution_alloc_count
    }

    fn release_all(&mut self) -> HostResult<()> {
        let mut first_error = None;
        for slot in [&mut self.decode, &mut self.prefill] {
            let Some(inner) = slot.take() else {
                continue;
            };
            if let Err(error) = ProgramSession::from_inner(&mut *self.runtime, inner).teardown() {
                first_error.get_or_insert(error);
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

impl Drop for PairedProgramSession<'_> {
    fn drop(&mut self) {
        drop(self.release_all());
    }
}

fn execute_inner(
    runtime: &mut DeviceRuntime,
    slot: &mut Option<ProgramInner>,
    inputs: &BTreeMap<u32, Vec<f32>>,
) -> HostResult<DeviceExecutionReceipt> {
    let inner = slot
        .take()
        .ok_or_else(|| HostError::internal("paired program session is closed"))?;
    let mut session = ProgramSession::from_inner(runtime, inner);
    let result = session.execute_resident_step(inputs);
    *slot = Some(session.into_inner());
    result
}

fn release_inner(runtime: &mut DeviceRuntime, inner: ProgramInner) {
    drop(ProgramSession::from_inner(runtime, inner).teardown());
}

fn session_error(error: SessionError) -> HostError {
    HostError {
        code: error.code.to_string(),
        message: error.message,
        retryable: false,
    }
}

fn pair_sequence_capacity(prompt_len: usize) -> HostResult<u32> {
    let prompt = u32::try_from(prompt_len).map_err(|_| {
        HostError::invalid_args("prefill prompt is longer than the sequence coordinate space")
    })?;
    // The prepare API has no model context length (composite_host.rs is
    // outside this unit). Admit a long decode window; D4 still rejects
    // u32 overflow pre-dispatch.
    Ok(prompt.saturating_add(65_536).max(1))
}

fn invocation_mode(mode: DeviceExecuteInvocationMode) -> InvocationMode {
    match mode {
        DeviceExecuteInvocationMode::Prefill => InvocationMode::Prefill,
        DeviceExecuteInvocationMode::ScalarDecode => InvocationMode::ScalarDecode,
    }
}

fn invocation_query_rows(
    invocation: &DeviceExecuteInvocation,
    prompt_len: usize,
) -> HostResult<u32> {
    let query_rows = invocation
        .valid_len_after
        .checked_sub(invocation.prefix_before)
        .ok_or_else(|| {
            HostError::invalid_args("invocation valid_len_after is before prefix_before")
        })?;
    match invocation.mode {
        DeviceExecuteInvocationMode::Prefill => {
            let prompt_rows = u32::try_from(prompt_len).map_err(|_| {
                HostError::invalid_args(
                    "prefill prompt is longer than the sequence coordinate space",
                )
            })?;
            if query_rows != prompt_rows {
                return Err(HostError::invalid_args(
                    "prefill query_rows must equal the prepared prompt length",
                ));
            }
        }
        DeviceExecuteInvocationMode::ScalarDecode => {
            if query_rows != 1 {
                return Err(HostError::invalid_args(
                    "scalar decode is M=1; valid_len_after must be prefix_before + 1",
                ));
            }
        }
    }
    Ok(query_rows)
}

fn match_invocation_cursor(
    invocation: &DeviceExecuteInvocation,
    tx: &InvocationTransaction,
) -> HostResult<()> {
    let coords = tx.coordinates();
    if invocation.sequence_epoch != tx.sequence_epoch() {
        return Err(HostError {
            code: E_KV_STALE.to_owned(),
            message: format!(
                "invocation epoch {} does not match live epoch {}",
                invocation.sequence_epoch,
                tx.sequence_epoch()
            ),
            retryable: false,
        });
    }
    if invocation.prefix_before != coords.prefix_before
        || invocation.valid_len_after != coords.valid_len_after
    {
        return Err(HostError {
            code: E_KV_STALE.to_owned(),
            message: format!(
                "invocation prefix_before={} valid_len_after={} does not match live prefix_before={} valid_len_after={}",
                invocation.prefix_before,
                invocation.valid_len_after,
                coords.prefix_before,
                coords.valid_len_after
            ),
            retryable: false,
        });
    }
    if invocation.position != coords.write_position || invocation.query_start != coords.query_start
    {
        return Err(HostError::invalid_args(format!(
            "invocation position={} query_start={} does not match write_position={} query_start={}",
            invocation.position, invocation.query_start, coords.write_position, coords.query_start
        )));
    }
    Ok(())
}

fn validate_pair_descriptors(
    prefill: &DeviceDescriptor,
    decode: &DeviceDescriptor,
) -> HostResult<()> {
    prefill.validate()?;
    decode.validate()?;
    if prefill.backend != decode.backend {
        return Err(HostError::invalid_args(format!(
            "paired programs target different backends: {} and {}",
            prefill.backend.spelling(),
            decode.backend.spelling()
        )));
    }
    if prefill.program_lifetime != DeviceProgramLifetime::RepeatingStep
        || decode.program_lifetime != DeviceProgramLifetime::RepeatingStep
    {
        return Err(HostError::invalid_args(
            "paired programs must both use the RepeatingStep lifetime",
        ));
    }
    let prefill_resources = persistent_semantics(prefill);
    let decode_resources = persistent_semantics(decode);
    if prefill_resources != decode_resources {
        return Err(HostError::invalid_args(
            "paired programs do not declare one shared PerProgram model/cache owner",
        ));
    }
    if prefill_resources.is_empty() {
        return Err(HostError::invalid_args(
            "paired programs require at least one shared PerProgram resource",
        ));
    }
    Ok(())
}

fn persistent_semantics(descriptor: &DeviceDescriptor) -> BTreeSet<u32> {
    descriptor
        .kernels
        .iter()
        .flat_map(|kernel| &kernel.buffers)
        .filter(|slot| slot.lifetime == DeviceBufferLifetime::PerProgram)
        .map(|slot| slot.semantic_value)
        .collect()
}

fn per_execution_alloc_count(descriptor: &DeviceDescriptor) -> usize {
    let mut per_execution = BTreeSet::new();
    for kernel in &descriptor.kernels {
        for slot in &kernel.buffers {
            match slot.lifetime {
                DeviceBufferLifetime::PerStep | DeviceBufferLifetime::ObservationPoint => {
                    per_execution.insert((slot.buffer_id, slot.version));
                }
                DeviceBufferLifetime::PerProgram => {}
            }
        }
    }
    per_execution.len()
}

impl RopeConfig {
    fn validate_for_pair(self) -> HostResult<()> {
        if self.head_dim == 0 || self.head_dim % 2 != 0 {
            return Err(HostError::invalid_args(format!(
                "RoPE head_dim must be a nonzero even number; got {}",
                self.head_dim
            )));
        }
        if !self.theta.is_finite() || self.theta <= 0.0 {
            return Err(HostError::invalid_args(format!(
                "RoPE theta must be finite and positive; got {}",
                self.theta
            )));
        }
        Ok(())
    }
}
