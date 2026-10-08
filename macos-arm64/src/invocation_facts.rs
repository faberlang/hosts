//! Invocation facts and distributed-image prepare shared by the composite
//! host. The device command-line surface that used to live here was
//! removed with the radix child-process route.

use host_coordinator::bound_plan::BindError;
use host_coordinator::discovery::DeviceDiscoverySnapshot;
use host_coordinator::execution_transaction::{
    ExecutionTransaction, FakeExecutionBackend, TransactionId, TransactionState,
};
use host_coordinator::partition::{FixtureIdentityClass, TransportClass};
use serde::{Deserialize, Serialize};

use crate::distributed_translate::{
    TranslateError, bind_policy_for_declared_count, bind_translated, translate_device_section_bytes,
};
use crate::kernel::{HostError, HostResult};

/// Explicit v2 invocation regime. The spelling is a stable wire name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceExecuteInvocationMode {
    Prefill,
    ScalarDecode,
}

impl DeviceExecuteInvocationMode {
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::ScalarDecode => "scalar_decode",
        }
    }
}

/// Runtime facts for one v2 invocation. Cursor facts are explicit and are
/// never inferred from the token or from the current step count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceExecuteInvocation {
    pub mode: DeviceExecuteInvocationMode,
    /// Decode's one token. Prefill may omit this when its token rows are in
    /// the declared input stream.
    #[serde(default)]
    pub token: Option<u32>,
    /// Absolute cache position written by this invocation.
    pub position: u32,
    pub sequence_epoch: u32,
    pub prefix_before: u32,
    pub valid_len_after: u32,
    pub query_start: u32,
}

/// Frozen MD3H host-run receipt for a distributed-image prepare.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistributedPrepareReceipt {
    /// Distinct physical devices the bind named.
    pub physical_device_count: u64,
    /// Physical identities in canonical order.
    pub physical_device_ids: Vec<String>,
    /// Admitted virtual partitions.
    pub virtual_partition_count: u64,
    /// Virtual partition identities in canonical order.
    pub virtual_partition_ids: Vec<String>,
    /// `virtual` for any software-admission bind.
    pub fixture_identity_class: String,
    /// Transport evidence class (`host_staged` for an 8-rank graph).
    pub transport_class: String,
    /// Always false for a virtual bind.
    pub hardware_isolation_claimed: bool,
    /// `virtual:physical` shape (`8:1`, `8:8`, `1:1`).
    pub bind_shape: String,
    /// Transfer/collective/barrier count. Zero for N=1.
    pub communication_graph_edge_count: u64,
    /// Content-addressed discovery snapshot id (lowercase hex).
    pub snapshot_id: String,
    /// Admitted logical plan hash.
    pub logical_distributed_plan_hash: String,
    /// Bound-plan hash (physical ids enter this domain).
    pub bound_distributed_plan_hash: String,
    /// `ExecutionTransaction` state after prepare (`prepared`).
    pub transaction_state: String,
}

/// Translate, bind, and `ExecutionTransaction::prepare` an F1 image.
///
/// Bind policy follows [`bind_policy_for_declared_count`]: 1 colocates
/// every virtual partition on the snapshot (8:1); a bind count equal to
/// the partition count claims one physical per partition and rejects
/// `TopologyMismatch` on a 1-physical snapshot.
///
/// # Errors
///
/// Decode/admit/translation failures, an unsupported bind count, a bind
/// topology/membership failure, or a transaction prepare failure.
pub fn prepare_distributed_image(
    image: &[u8],
    snapshot: &DeviceDiscoverySnapshot,
    bind_count: u32,
) -> HostResult<DistributedPrepareReceipt> {
    let translated = translate_device_section_bytes(image).map_err(translate_to_host_error)?;
    let policy = bind_policy_for_declared_count(translated.partitions().len(), bind_count)
        .map_err(translate_to_host_error)?;
    let bound = bind_translated(&translated, snapshot, policy).map_err(bind_to_host_error)?;
    let mut backend = FakeExecutionBackend::new();
    let mut transaction = ExecutionTransaction::new(
        TransactionId::new("md3h-h4-prepare"),
        bound.clone(),
        translated.operations().to_vec(),
        translated.commit_boundary().clone(),
    )
    .map_err(|error| {
        HostError::invalid_args(format!(
            "distributed transaction construct failed: {error:?}"
        ))
    })?;
    transaction.prepare(&mut backend).map_err(|error| {
        HostError::invalid_args(format!("distributed transaction prepare failed: {error:?}"))
    })?;
    if transaction.state() != &TransactionState::Prepared {
        return Err(HostError::invalid_args(format!(
            "distributed transaction prepare left state {:?}",
            transaction.state()
        )));
    }
    Ok(distributed_prepare_receipt(
        &bound,
        translated.communication_graph_edge_count(),
    ))
}

/// Load `--distributed-image` against a live discovery snapshot and prepare.
///
/// # Errors
///
/// Missing flags, file read, backend discovery, translation, bind, or

fn translate_to_host_error(error: TranslateError) -> HostError {
    HostError::invalid_args(format!("distributed translate {error}"))
}

fn bind_to_host_error(error: BindError) -> HostError {
    match error {
        BindError::TopologyMismatch { detail } => {
            HostError::invalid_args(format!("TopologyMismatch: {detail}"))
        }
        other => HostError::invalid_args(format!("distributed bind rejected: {other:?}")),
    }
}

fn distributed_prepare_receipt(
    bound: &host_coordinator::bound_plan::BoundDistributedPlan,
    communication_graph_edge_count: u64,
) -> DistributedPrepareReceipt {
    let receipt = bound.receipt();
    let physical_device_ids: Vec<String> = receipt
        .physical_device_ids()
        .iter()
        .map(ToString::to_string)
        .collect();
    let virtual_partition_ids: Vec<String> = receipt
        .virtual_partition_ids()
        .iter()
        .map(ToString::to_string)
        .collect();
    let physical_device_count = receipt.physical_device_count();
    let virtual_partition_count = receipt.virtual_partition_count();
    DistributedPrepareReceipt {
        physical_device_count,
        physical_device_ids,
        virtual_partition_count,
        virtual_partition_ids,
        fixture_identity_class: fixture_spelling(receipt.fixture_identity_class()).to_owned(),
        transport_class: transport_spelling(receipt.transport_class()).to_owned(),
        hardware_isolation_claimed: false,
        bind_shape: format!("{virtual_partition_count}:{physical_device_count}"),
        communication_graph_edge_count,
        snapshot_id: bound.snapshot_id().to_string(),
        logical_distributed_plan_hash: bound.logical_distributed_plan_hash().to_owned(),
        bound_distributed_plan_hash: bound.bound_distributed_plan_hash().to_owned(),
        transaction_state: "prepared".to_owned(),
    }
}

fn fixture_spelling(class: FixtureIdentityClass) -> &'static str {
    match class {
        FixtureIdentityClass::Physical => "physical",
        FixtureIdentityClass::Virtual => "virtual",
        FixtureIdentityClass::Synthetic => "synthetic",
    }
}

fn transport_spelling(class: TransportClass) -> &'static str {
    match class {
        TransportClass::None => "none",
        TransportClass::HostStaged => "host_staged",
        TransportClass::DirectedPeerNotAttempted => "directed_peer_not_attempted",
    }
}

