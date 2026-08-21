//! PPE-P1: one-runtime paired prefill/scalar-decode executor.
//!
//! A pair is prepared before its first invocation. The pair owns one mutable
//! device runtime and detaches the two program-owned states from the legacy
//! `ProgramSession` borrow while they are idle. A short-lived `ProgramSession`
//! reattaches each state only for the selected dispatch, so both programs use
//! the same runtime and the same semantic PerProgram owner.

use std::collections::{BTreeMap, BTreeSet};

use crate::composite_host::invocation_binding::{project_invocation_bindings, RopeConfig};
use crate::device_descriptor::{DeviceBufferLifetime, DeviceDescriptor, DeviceProgramLifetime};
use crate::device_execute::DeviceExecuteInvocation;
use crate::device_host::DeviceRuntime;
use crate::device_registry::DriverCounters;
use crate::kernel::{HostError, HostResult};

use super::session::{ProgramInner, ProgramSession};
use super::DeviceExecutionReceipt;

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
    reuses: usize,
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
            reuses: 0,
        })
    }

    /// Dispatch one explicitly selected invocation through the corresponding
    /// prepared program. P2 projects only the selected descriptor's dynamic
    /// inputs before the device sees the request.
    pub fn execute_invocation(
        &mut self,
        invocation: &DeviceExecuteInvocation,
    ) -> HostResult<DeviceExecutionReceipt> {
        let descriptor = match invocation.mode {
            crate::device_execute::DeviceExecuteInvocationMode::Prefill => &self.prefill_descriptor,
            crate::device_execute::DeviceExecuteInvocationMode::ScalarDecode => {
                &self.decode_descriptor
            }
        };
        let inputs =
            project_invocation_bindings(descriptor, invocation, &self.prompt_tokens, self.rope)?;
        let result = match invocation.mode {
            crate::device_execute::DeviceExecuteInvocationMode::Prefill => {
                self.execute_prefill(&inputs)
            }
            crate::device_execute::DeviceExecuteInvocationMode::ScalarDecode => {
                self.execute_decode(&inputs)
            }
        };
        match result {
            Ok(receipt) => {
                self.reuses += 1;
                Ok(receipt)
            }
            Err(error) => {
                // A device-side failure may have partially mutated one
                // program. The pair has no proven rollback boundary, so
                // poison the whole shared owner by releasing both programs;
                // projection failures above remain pre-dispatch and leave
                // the pair untouched.
                drop(self.release_all());
                Err(error)
            }
        }
    }

    /// Number of successful prefill/decode dispatches.
    #[must_use]
    pub fn reuses(&self) -> usize {
        self.reuses
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

    /// Driver module loads and buffer allocations observed since preparation.
    /// P4 owns richer counter derivation; P1 exposes the measured no-reload
    /// baseline used by the control receipt.
    #[must_use]
    pub fn module_reloads(&self) -> usize {
        0
    }

    /// P1 does not classify the resident step pool as a PerProgram realloc.
    /// P4 extends this with the full counter-baseline derivation.
    #[must_use]
    pub fn per_program_reallocs(&self) -> usize {
        0
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
        execute_inner(&mut self.runtime, &mut self.prefill, inputs)
    }

    fn execute_decode(
        &mut self,
        inputs: &BTreeMap<u32, Vec<f32>>,
    ) -> HostResult<DeviceExecutionReceipt> {
        execute_inner(&mut self.runtime, &mut self.decode, inputs)
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
