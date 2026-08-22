//! KV-D D3: dual invocation programs over shared residency.
//!
//! Prefill(M=T) and ScalarDecode(M=1) are separate static graphs, both
//! prepared before the first invocation. They share one [`SessionResidency`].
//! Selection is explicit by [`InvocationMode`] and never inferred from
//! sequence length. Switching programs does not load, compile, allocate, or
//! upload. No device work — composition over D1/D2 and B5 descriptor shapes.
//!
//! Parent registration is a private `mod invocation_program` in
//! `composite_host.rs`; this unit cannot re-export it.

#![allow(dead_code)]

use super::inference_state::{
    CursorFacts, InvocationMode, PlannedInvocation, SessionError, E_INVALID_ARGS,
};
use super::residency::{ModelIdentity, ModelSpec, ResolvedHandles, SequenceSpec, SessionResidency};
use crate::device_descriptor::{
    DescriptorAllocation, DescriptorInvocationState, DescriptorLaunchBinding, DescriptorView,
    KvCacheDescriptor,
};

/// Scalar-decode query-row width. Never inferred from sequence length.
pub const SCALAR_DECODE_QUERY_ROWS: u32 = 1;

fn invalid_args(message: impl Into<String>) -> SessionError {
    SessionError {
        code: E_INVALID_ARGS,
        message: message.into(),
    }
}

/// One prepared invocation graph: declared mode, M, and the B5 launch plan.
///
/// Artifacts live on the shared residency; this record does not own a second
/// module or allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvocationGraph {
    mode: InvocationMode,
    query_rows: u32,
    launch_bindings: Vec<DescriptorLaunchBinding>,
}

impl InvocationGraph {
    #[must_use]
    pub fn mode(&self) -> InvocationMode {
        self.mode
    }

    /// Declared query-row width M. Prefill is M=T; scalar decode is M=1.
    #[must_use]
    pub fn query_rows(&self) -> u32 {
        self.query_rows
    }

    #[must_use]
    pub fn launch_bindings(&self) -> &[DescriptorLaunchBinding] {
        &self.launch_bindings
    }
}

/// Explicitly selected program. Sequence length is not an input and is not
/// consulted.
#[derive(Debug, Clone, Copy)]
pub struct SelectedProgram<'a> {
    graph: &'a InvocationGraph,
    handles: ResolvedHandles<'a>,
}

impl<'a> SelectedProgram<'a> {
    #[must_use]
    pub fn mode(&self) -> InvocationMode {
        self.graph.mode
    }

    #[must_use]
    pub fn query_rows(&self) -> u32 {
        self.graph.query_rows
    }

    #[must_use]
    pub fn artifact(&self) -> &'a [u8] {
        self.handles.artifact
    }

    #[must_use]
    pub fn launch_bindings(&self) -> &'a [DescriptorLaunchBinding] {
        &self.graph.launch_bindings
    }

    #[must_use]
    pub fn handles(&self) -> ResolvedHandles<'a> {
        self.handles
    }
}

/// Synthetic admitted descriptor built from B5 public host shapes.
///
/// The KV plan is a [`KvCacheDescriptor`]: allocations, views, default
/// invocation-state (no live cursor), and declared launch bindings.
/// Prefill carries M=T; scalar decode is always M=1. Weights and the
/// invocation-state buffer sit beside the cache plan.
pub struct AdmittedDescriptor {
    pub identity: ModelIdentity,
    pub prefill_artifact: Vec<u8>,
    pub decode_artifact: Vec<u8>,
    pub prefill_query_rows: u32,
    pub weights: Vec<DescriptorAllocation>,
    pub kv: KvCacheDescriptor,
    pub invocation_state: DescriptorAllocation,
    pub capacity: u32,
}

/// Dual invocation programs sharing one model/sequence residency (KV-L7).
#[derive(Debug, PartialEq, Eq)]
pub struct InvocationPrograms {
    residency: SessionResidency,
    prefill: InvocationGraph,
    scalar_decode: InvocationGraph,
    kv: KvCacheDescriptor,
    module_loads: u32,
    compiles: u32,
}

impl InvocationPrograms {
    /// Admit both graphs and the shared residency. Both artifacts are
    /// prepared here, before any invocation.
    pub fn admit(descriptor: AdmittedDescriptor) -> Result<Self, SessionError> {
        if descriptor.prefill_query_rows == 0 {
            return Err(invalid_args("prefill query_rows (M=T) must be at least 1"));
        }
        if descriptor.prefill_query_rows > descriptor.capacity {
            return Err(invalid_args(format!(
                "prefill M={} exceeds capacity {}",
                descriptor.prefill_query_rows, descriptor.capacity
            )));
        }
        if descriptor.kv.invocation_state != DescriptorInvocationState::default() {
            return Err(invalid_args(
                "admitted KV plan must not carry live cursor values",
            ));
        }
        admit_kv_plan(&descriptor.kv)?;
        let sequence = sequence_spec_from_kv(
            &descriptor.kv,
            descriptor.invocation_state,
            descriptor.capacity,
        )?;
        let model = ModelSpec {
            identity: descriptor.identity,
            prefill_artifact: descriptor.prefill_artifact,
            decode_artifact: descriptor.decode_artifact,
            weights: descriptor.weights,
        };
        let residency = SessionResidency::prepare(model, sequence)?;
        let launch_bindings = descriptor.kv.launch_bindings.clone();
        Ok(Self {
            residency,
            prefill: InvocationGraph {
                mode: InvocationMode::Prefill,
                query_rows: descriptor.prefill_query_rows,
                launch_bindings: launch_bindings.clone(),
            },
            scalar_decode: InvocationGraph {
                mode: InvocationMode::ScalarDecode,
                query_rows: SCALAR_DECODE_QUERY_ROWS,
                launch_bindings,
            },
            kv: descriptor.kv,
            module_loads: 2,
            compiles: 2,
        })
    }

    #[must_use]
    pub fn residency(&self) -> &SessionResidency {
        &self.residency
    }

    pub fn residency_mut(&mut self) -> &mut SessionResidency {
        &mut self.residency
    }

    #[must_use]
    pub fn prefill(&self) -> &InvocationGraph {
        &self.prefill
    }

    #[must_use]
    pub fn scalar_decode(&self) -> &InvocationGraph {
        &self.scalar_decode
    }

    /// Admitted B5 KV plan. Launch bindings keep declared indices and order.
    #[must_use]
    pub fn kv(&self) -> &KvCacheDescriptor {
        &self.kv
    }

    /// Explicit program selection. Sequence length is not consulted.
    #[must_use]
    pub fn select(&self, mode: InvocationMode) -> SelectedProgram<'_> {
        let graph = match mode {
            InvocationMode::Prefill => &self.prefill,
            InvocationMode::ScalarDecode => &self.scalar_decode,
            // SV-E2 session shape; no verification program exists until SV-E3
            // materializes it. Fail closed rather than alias the scalar loop.
            InvocationMode::Verification => {
                panic!("verification program is not materialized until SV-E3")
            }
        };
        SelectedProgram {
            graph,
            handles: self.residency.resolve(mode),
        }
    }

    #[must_use]
    pub fn resolve(&self, mode: InvocationMode) -> ResolvedHandles<'_> {
        self.residency.resolve(mode)
    }

    /// Begin the selected program at its declared M. Mode is not inferred
    /// from `valid_len`.
    pub fn begin_selected(&self, mode: InvocationMode) -> Result<PlannedInvocation, SessionError> {
        let selected = self.select(mode);
        self.residency.begin_invocation(mode, selected.query_rows())
    }

    pub fn commit(&mut self, plan: &PlannedInvocation) -> Result<CursorFacts, SessionError> {
        self.residency.commit(plan)
    }

    #[must_use]
    pub fn module_loads(&self) -> u32 {
        self.module_loads
    }

    #[must_use]
    pub fn compiles(&self) -> u32 {
        self.compiles
    }

    #[must_use]
    pub fn live_allocation_count(&self) -> usize {
        self.residency.live_allocation_count()
    }

    #[must_use]
    pub fn weight_uploads(&self) -> u32 {
        self.residency.weight_uploads()
    }

    #[must_use]
    pub fn artifact_prepares(&self) -> u32 {
        self.residency.artifact_prepares()
    }
}

fn admit_kv_plan(kv: &KvCacheDescriptor) -> Result<(), SessionError> {
    kv.validate()
        .map_err(|err| invalid_args(format!("admitted KV descriptor: {}", err.message)))
}

fn sequence_spec_from_kv(
    kv: &KvCacheDescriptor,
    invocation_state: DescriptorAllocation,
    capacity: u32,
) -> Result<SequenceSpec, SessionError> {
    if kv.allocations.len() != 2 {
        return Err(invalid_args(format!(
            "admitted KV plan must declare exactly two cache allocations; found {}",
            kv.allocations.len()
        )));
    }
    let k_arena = kv.allocations[0];
    let v_arena = kv.allocations[1];
    let (k_prefix, k_append) = split_prefix_append(&kv.views, &k_arena)?;
    let (v_prefix, v_append) = split_prefix_append(&kv.views, &v_arena)?;
    Ok(SequenceSpec {
        k_arena,
        v_arena,
        k_prefix,
        k_append,
        v_prefix,
        v_append,
        invocation_state,
        capacity,
    })
}

fn split_prefix_append(
    views: &[DescriptorView],
    allocation: &DescriptorAllocation,
) -> Result<(DescriptorView, DescriptorView), SessionError> {
    let owned: Vec<&DescriptorView> = views
        .iter()
        .filter(|view| view.allocation_id == allocation.buffer_id)
        .collect();
    if owned.len() != 2 {
        return Err(invalid_args(format!(
            "allocation {} must expose prefix and append views; found {}",
            allocation.buffer_id,
            owned.len()
        )));
    }
    if owned[0].maximum_span == owned[1].maximum_span {
        return Err(invalid_args(format!(
            "allocation {} prefix and append views have the same span",
            allocation.buffer_id
        )));
    }
    if owned[0].maximum_span > owned[1].maximum_span {
        Ok((owned[0].clone(), owned[1].clone()))
    } else {
        Ok((owned[1].clone(), owned[0].clone()))
    }
}
