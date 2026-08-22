//! Priority policy engine — the two-level evaluator (C2 §1 / MD-A8; MD1-P1).
//!
//! Two passes, in this order ([`md0-cost-contract.md`](radix/docs/factory/gpu-inference-multi-device/md0-cost-contract.md)
//! §1):
//!
//! 1. **Gate pass (satisfiability).** Hard constraints reject infeasible
//!    devices/plans outright. Four gate classes are evaluated:
//!    - [`GateClass::HealthEpochMembership`] — a stale snapshot, a selection
//!      that does not resolve (unknown/replaced id, stale epoch, count
//!      bounds), or a plan referencing a device outside the resolved
//!      selection rejects (MD1-Q3 default; replacement detection, naming
//!      contract §1).
//!    - [`GateClass::Compatibility`] — every operation placed on a device
//!      must be executable there (C2 §1.2 compatibility): compute
//!      capability, SM count, and dtype surface from the snapshot's
//!      capability facts.
//!    - [`GateClass::RequiredMemory`] — the declared requirements per device
//!      (including [`crate::partition`] budgets, MD1-V1) must fit that
//!      device's declared admitted budget ([`SafePhysicalLimit`] — a policy
//!      fact, never a hardware report), and the plan's declared total must
//!      fit the declared total budget (C2 §1.2 required memory).
//!    - [`GateClass::Topology`] — every transfer traverses only admitted
//!      directed links (C2 §1.2 topology/peer-access): NOT-ATTEMPTED,
//!      rejected, absent, and unmeasured links are never assumed.
//! 2. **Objective pass (ranking).** The seven C2 objectives (latency,
//!    throughput, capacity reserve, cache affinity, expert hotness, transfer
//!    budget, power) rank **only** the plans that passed every gate. No
//!    objective ever rejects, no gate ever ranks (C2 §1.1 review test).
//!
//! The evaluator is **deterministic** for one frozen
//! [`DeviceDiscoverySnapshot`] + [`DeviceSetSelection`] + declared
//! constraints/objectives: identical inputs produce the identical plan and
//! the identical explained receipt. Ties break by declared objective order,
//! then stable [`PhysicalDeviceId`] — **never ordinal** (naming contract
//! §1: the ordinal is a locator only and never participates).
//!
//! Evaluation is **pure**: no allocation, no side effects. Impossible
//! memory/capability/topology constraints fail before any allocation.
//!
//! The explained receipt ([`ExplainedReceipt`], MD-A14) names the frozen
//! snapshot id, the declared constraints/objectives, rejected devices with
//! the violated gate + exact failing fact, and selected devices with ranks +
//! objective scores, plus a determinism fingerprint.

use crate::device_identity::{
    push_bool, push_str, push_u64, DeviceHealthGeneration, PhysicalDeviceId,
};
use crate::device_set::{
    DeviceSet, DeviceSetSelection, DeviceTopologySnapshot, LinkGateError, MembershipError,
    SelectionError,
};
use crate::discovery::{
    ComputeCapability, DeviceDiscoveryEntry, DeviceDiscoverySnapshot, DtypeSurface,
};
use crate::partition::{PartitionBudgetLedger, SafePhysicalLimit};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

/// The four hard-constraint gate classes evaluated by [`evaluate`] (C2 §1.2;
/// MD-A8). Every gate **rejects**; no gate ever ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GateClass {
    /// Health-epoch membership (C2 §2 *healthy*/*degraded*; MD1-Q3 default):
    /// only current-epoch members are selectable.
    HealthEpochMembership,
    /// Compatibility — dtype/quantization support from the snapshot's
    /// capability facts (compute capability, SM count, dtype surface).
    Compatibility,
    /// Required memory per device (incl. partition budgets) + declared
    /// total (C2 §1.2 required memory).
    RequiredMemory,
    /// Topology/peer-access — admitted directed links only (C2 §1.2).
    Topology,
}

impl GateClass {
    /// Deterministic serialization tag.
    #[must_use]
    pub const fn tag(self) -> u64 {
        match self {
            Self::HealthEpochMembership => 0,
            Self::Compatibility => 1,
            Self::RequiredMemory => 2,
            Self::Topology => 3,
        }
    }
}

/// One gate violation: the violated constraint class + the exact failing
/// fact (MD-A14 observable cost — the rejection reason is part of the
/// receipt).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateViolation {
    /// The violated gate class.
    pub gate: GateClass,
    /// The device whose fact failed; `None` for plan-level violations (the
    /// declared total, a topology pair, a selection-level rejection).
    pub device: Option<PhysicalDeviceId>,
    /// The exact fact that failed (device facts, byte numbers, link state —
    /// never a paraphrase).
    pub failing_fact: String,
}

/// Why one candidate plan was rejected: its name plus every gate violation
/// recorded against it. A plan with a non-empty violation list is **never
/// ranked** (C2 §1: rejected outright).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanRejection {
    /// The candidate plan's stable name.
    pub plan: String,
    /// Every gate violation collected for this plan, in gate-evaluation
    /// order (membership, compatibility, memory, topology).
    pub violations: Vec<GateViolation>,
}

/// The seven C2 soft objectives (C2 §1.2; Q6 default). Objectives **rank
/// only plans that already pass every gate**; they never reject and never
/// change dtype, quantization, model semantics, isolation, or failure
/// policy to win (MD-A8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Objective {
    /// Latency — minimize time from request submission to result completion.
    Latency,
    /// Throughput — maximize sustained tokens per second.
    Throughput,
    /// Capacity reserve — favor plans leaving the most headroom below each
    /// device's declared budget.
    CapacityReserve,
    /// Cache affinity — favor placements keeping hot blocks resident.
    CacheAffinity,
    /// Expert hotness — favor placing live-router-selected hot experts on
    /// fast devices.
    ExpertHotness,
    /// Transfer budget — favor plans whose total transfer bytes stay within
    /// the declared budget.
    TransferBudget,
    /// Power — favor the lowest measured power draw.
    Power,
}

impl Objective {
    /// Deterministic serialization tag.
    #[must_use]
    pub const fn tag(self) -> u64 {
        match self {
            Self::Latency => 0,
            Self::Throughput => 1,
            Self::CapacityReserve => 2,
            Self::CacheAffinity => 3,
            Self::ExpertHotness => 4,
            Self::TransferBudget => 5,
            Self::Power => 6,
        }
    }

    /// The preference direction used when ranking (C2 §1.2 rows).
    #[must_use]
    pub const fn direction(self) -> ObjectiveDirection {
        match self {
            Self::Latency | Self::TransferBudget | Self::Power => ObjectiveDirection::Minimize,
            Self::Throughput
            | Self::CapacityReserve
            | Self::CacheAffinity
            | Self::ExpertHotness => ObjectiveDirection::Maximize,
        }
    }
}

/// Whether an objective prefers smaller or larger scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveDirection {
    /// Smaller is better (latency, transfer budget, power).
    Minimize,
    /// Larger is better (throughput, capacity reserve, cache affinity,
    /// expert hotness).
    Maximize,
}

/// One objective's score for one ranked plan, in declared objective order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectiveScore {
    /// Which objective produced the score.
    pub objective: Objective,
    /// The score value (raw fact or computed headroom). For `Minimize`
    /// objectives smaller is better; for `Maximize` larger is better.
    pub value: u64,
}

/// A gate-passing plan with its rank and objective scores.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankedPlan {
    /// Rank within the gate-passing set (1 = best under the declared
    /// objectives).
    pub rank: usize,
    /// The candidate plan's stable name.
    pub plan: String,
    /// The plan's devices, in stable [`PhysicalDeviceId`] order (ordinal-free).
    pub devices: Vec<PhysicalDeviceId>,
    /// Objective scores, one per declared objective in declared order.
    pub scores: Vec<ObjectiveScore>,
}

/// The explained receipt (MD-A14) of one evaluation.
///
/// Names the frozen snapshot id, the declared constraints/objectives,
/// rejected plans with the violated gate + exact failing fact, selected
/// plans with ranks + objective scores, and a determinism fingerprint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExplainedReceipt {
    /// Hex of the frozen [`DeviceDiscoverySnapshot`] id the evaluation ran
    /// against.
    pub snapshot_id_hex: String,
    /// The health generation the evaluation was frozen at.
    pub current_generation: DeviceHealthGeneration,
    /// Declared per-device admitted budgets, in stable identity order.
    pub per_device_budget: Vec<(PhysicalDeviceId, SafePhysicalLimit)>,
    /// Declared total admitted budget across the device set, when declared.
    pub total_budget: Option<SafePhysicalLimit>,
    /// Declared objectives in declared order (the ranking order).
    pub objectives: Vec<Objective>,
    /// Rejected plans, in input order, each with gate + exact failing fact.
    pub rejected: Vec<PlanRejection>,
    /// Selected (gate-passing) plans with ranks + objective scores, in rank
    /// order.
    pub ranked: Vec<RankedPlan>,
    /// Determinism fingerprint: FNV-1a-64 over the receipt's canonical
    /// bytes. Identical inputs → identical fingerprint.
    pub determinism_fingerprint: String,
}

impl ExplainedReceipt {
    /// Deterministic canonical bytes of every receipt field *except* the
    /// fingerprint itself (the fingerprint is derived *from* these bytes).
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_str(&mut out, &self.snapshot_id_hex);
        push_u64(&mut out, self.current_generation.get());
        push_u64(&mut out, self.per_device_budget.len() as u64);
        for (id, limit) in &self.per_device_budget {
            out.extend_from_slice(&id.canonical_bytes());
            push_u64(&mut out, limit.get());
        }
        push_bool(&mut out, self.total_budget.is_some());
        if let Some(total) = self.total_budget {
            push_u64(&mut out, total.get());
        }
        push_u64(&mut out, self.objectives.len() as u64);
        for objective in &self.objectives {
            push_u64(&mut out, objective.tag());
        }
        push_u64(&mut out, self.rejected.len() as u64);
        for rejection in &self.rejected {
            push_str(&mut out, &rejection.plan);
            push_u64(&mut out, rejection.violations.len() as u64);
            for violation in &rejection.violations {
                push_u64(&mut out, violation.gate.tag());
                push_bool(&mut out, violation.device.is_some());
                if let Some(device) = &violation.device {
                    out.extend_from_slice(&device.canonical_bytes());
                }
                push_str(&mut out, &violation.failing_fact);
            }
        }
        push_u64(&mut out, self.ranked.len() as u64);
        for ranked in &self.ranked {
            push_u64(&mut out, ranked.rank as u64);
            push_str(&mut out, &ranked.plan);
            push_u64(&mut out, ranked.devices.len() as u64);
            for device in &ranked.devices {
                out.extend_from_slice(&device.canonical_bytes());
            }
            push_u64(&mut out, ranked.scores.len() as u64);
            for score in &ranked.scores {
                push_u64(&mut out, score.objective.tag());
                push_u64(&mut out, score.value);
            }
        }
        out
    }
}

/// The result of one [`evaluate`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyOutcome {
    /// Rejected plans (never ranked), in input order.
    pub rejected: Vec<PlanRejection>,
    /// Gate-passing plans with ranks + objective scores, in rank order.
    pub ranked: Vec<RankedPlan>,
    /// The explained receipt (MD-A14).
    pub receipt: ExplainedReceipt,
}

/// Declared hard constraints the evaluator gates against (C2 §1.2).
#[derive(Debug, Clone)]
pub struct DeclaredConstraints {
    /// The health generation the evaluation is frozen at; snapshots and
    /// members recorded under any other generation are stale and reject.
    pub current_generation: DeviceHealthGeneration,
    /// The topology measured from the same frozen discovery sample:
    /// per-device facts plus directed [`crate::device_set::DeviceLink`]
    /// rows for the topology/peer-access gate.
    pub topology: DeviceTopologySnapshot,
    /// Declared admitted budget per device (policy facts — never hardware
    /// memory reports). A device with no declared budget is fail-closed.
    pub per_device_budget: BTreeMap<PhysicalDeviceId, SafePhysicalLimit>,
    /// Declared total admitted budget across the device set; `None` leaves
    /// the declared-total check undeclared.
    pub total_budget: Option<SafePhysicalLimit>,
}

/// Declared capability requirements one device assignment must satisfy for
/// the compatibility gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequiredCapabilities {
    /// Minimum compute capability (`major.minor`), when required.
    pub min_compute_capability: Option<ComputeCapability>,
    /// Minimum SM count, when required.
    pub min_sm_count: Option<u32>,
    /// Required dtypes: only the `true` bits are required; a device whose
    /// surface lacks any required dtype fails the compatibility gate.
    pub required_dtypes: DtypeSurface,
}

impl Default for RequiredCapabilities {
    /// No declared requirements: compatible with every capability surface
    /// (nothing is required, nothing is checked).
    fn default() -> Self {
        Self {
            min_compute_capability: None,
            min_sm_count: None,
            required_dtypes: DtypeSurface::empty(),
        }
    }
}

/// Declared soft-objective facts of one candidate plan.
///
/// Missing facts are **least-favored** for the corresponding objective
/// (never a rejection): `None` scores `0` for maximize objectives and
/// `u64::MAX` for minimize objectives. Capacity reserve is not declared —
/// it is computed from the assignments and declared budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ObjectiveFacts {
    /// Estimated end-to-end latency, nanoseconds (latency objective).
    pub latency_nanos: Option<u64>,
    /// Estimated sustained throughput, tokens per second (throughput).
    pub throughput_tokens_per_sec: Option<u64>,
    /// Expected cache hit rate, per-mille `0..=1000` (cache affinity).
    pub cache_affinity_hit_rate_milli: Option<u64>,
    /// Expert hotness placement score (expert hotness).
    pub expert_hotness_score: Option<u64>,
    /// Total transfer bytes (transfer budget).
    pub transfer_bytes: Option<u64>,
    /// Estimated power draw, milliwatts (power).
    pub estimated_power_milliwatts: Option<u64>,
}

/// One device assignment in a candidate plan: bind one device to a plan role
/// with its declared memory requirement (partition budget) and the
/// capabilities the role requires on that device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceAssignment {
    /// The bound physical device.
    pub device: PhysicalDeviceId,
    /// Declared memory requirement for this device — the partition budget
    /// ledger (MD1-V1, complete eight-class accounting).
    pub required: PartitionBudgetLedger,
    /// Capabilities this device must provide for this assignment
    /// (compatibility gate).
    pub required_capabilities: RequiredCapabilities,
}

impl DeviceAssignment {
    /// Build an assignment with the given requirement and **no** declared
    /// capability requirements (compatible with every surface).
    #[must_use]
    pub fn new(device: PhysicalDeviceId, required: PartitionBudgetLedger) -> Self {
        Self {
            device,
            required,
            required_capabilities: RequiredCapabilities::default(),
        }
    }
}

/// A directed transfer `from → to` the plan requires (topology gate).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transfer {
    /// Source device.
    pub from: PhysicalDeviceId,
    /// Destination device.
    pub to: PhysicalDeviceId,
}

impl Transfer {
    /// Build a directed transfer.
    #[must_use]
    pub fn new(from: PhysicalDeviceId, to: PhysicalDeviceId) -> Self {
        Self { from, to }
    }
}

/// A candidate plan: a declared device assignment (or set of assignments)
/// plus the transfers, declared total memory, and objective facts the
/// evaluator gates and ranks. Plans are inputs to [`evaluate`] — generating
/// placement plans is MD5's scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidatePlan {
    /// Stable name for the receipt.
    pub name: String,
    /// Per-device assignments (each binds one physical device).
    pub assignments: Vec<DeviceAssignment>,
    /// Transfers this plan requires (topology gate).
    pub transfers: Vec<Transfer>,
    /// The plan's declared total required memory across the device set
    /// (required-memory gate).
    pub declared_total_bytes: u64,
    /// Declared objective facts.
    pub facts: ObjectiveFacts,
}

impl CandidatePlan {
    /// Build a candidate plan.
    #[must_use]
    pub fn new(
        name: String,
        assignments: Vec<DeviceAssignment>,
        transfers: Vec<Transfer>,
        declared_total_bytes: u64,
        facts: ObjectiveFacts,
    ) -> Self {
        Self {
            name,
            assignments,
            transfers,
            declared_total_bytes,
            facts,
        }
    }
}

/// A policy evaluation request: one frozen discovery snapshot, one device-set
/// selection, the declared constraints, the declared objectives (in order),
/// and the candidate plans to evaluate.
#[derive(Debug, Clone, Copy)]
pub struct PolicyRequest<'a> {
    /// The frozen discovery snapshot (never timeless).
    pub discovery: &'a DeviceDiscoverySnapshot,
    /// The device-set selection request.
    pub selection: &'a DeviceSetSelection,
    /// The declared hard constraints.
    pub constraints: &'a DeclaredConstraints,
    /// Declared objectives in declared order (the ranking order).
    pub objectives: &'a [Objective],
    /// Candidate plans to gate and rank.
    pub plans: &'a [CandidatePlan],
}

/// The two-pass policy evaluator (C2 §1 / MD-A8).
///
/// Pure evaluation: no allocation, no side effects. Impossible
/// memory/capability/topology constraints fail before any allocation.
/// Deterministic for one frozen snapshot + selection + declared
/// constraints/objectives.
///
/// # Panics
///
/// Panics if `request.constraints.topology` was not measured from
/// `request.discovery`.
#[must_use]
pub fn evaluate(request: &PolicyRequest) -> PolicyOutcome {
    assert_eq!(
        request.constraints.topology.discovery(),
        request.discovery,
        "the declared topology must be measured from the frozen discovery snapshot"
    );

    let mut rejected = Vec::new();
    let mut ranked = Vec::new();

    // Pass 1, gate 1 at the selection level: a stale snapshot or a selection
    // that cannot resolve rejects every candidate plan before ranking.
    match gate_health_epoch_selection(request) {
        Ok(set) => {
            let mut passers = Vec::new();
            for (idx, plan) in request.plans.iter().enumerate() {
                let mut violations = Vec::new();
                violations.extend(gate_membership(plan, &set));
                violations.extend(gate_compatibility(plan, request.discovery));
                violations.extend(gate_memory(plan, request.constraints));
                violations.extend(gate_topology(plan, &request.constraints.topology));
                if violations.is_empty() {
                    passers.push((idx, plan));
                } else {
                    rejected.push(PlanRejection {
                        plan: plan.name.clone(),
                        violations,
                    });
                }
            }
            ranked = rank_plans(passers, request.objectives, request.constraints);
        }
        Err(violation) => {
            for plan in request.plans {
                rejected.push(PlanRejection {
                    plan: plan.name.clone(),
                    violations: vec![violation.clone()],
                });
            }
        }
    }

    let mut receipt = ExplainedReceipt {
        snapshot_id_hex: request.discovery.id().hex(),
        current_generation: request.constraints.current_generation,
        per_device_budget: request
            .constraints
            .per_device_budget
            .iter()
            .map(|(id, limit)| (id.clone(), *limit))
            .collect(),
        total_budget: request.constraints.total_budget,
        objectives: request.objectives.to_vec(),
        rejected: rejected.clone(),
        ranked: ranked.clone(),
        determinism_fingerprint: String::new(),
    };
    let bytes = receipt.canonical_bytes();
    receipt.determinism_fingerprint = hex64(fnv1a64(&bytes));

    PolicyOutcome {
        rejected,
        ranked,
        receipt,
    }
}

/// Gate 1 (selection level): a stale snapshot or an unresolvable selection
/// is rejected before it gates admission or planning.
fn gate_health_epoch_selection(request: &PolicyRequest) -> Result<DeviceSet, GateViolation> {
    let current = request.constraints.current_generation;

    // A snapshot carrying a stale generation can never be the frozen basis
    // of a plan (MD1-Q3 default; stale epoch rejects stale plans).
    if request.discovery.is_stale(current) {
        let stale_entries: Vec<String> = request
            .discovery
            .devices()
            .values()
            .filter(|entry| current.is_stale(entry.health_generation))
            .map(|entry| {
                format!(
                    "{} recorded at generation {}",
                    entry.identity, entry.health_generation
                )
            })
            .collect();
        return Err(GateViolation {
            gate: GateClass::HealthEpochMembership,
            device: None,
            failing_fact: format!(
                "snapshot {} carries stale health generation(s): {}",
                request.discovery.id(),
                stale_entries.join("; ")
            ),
        });
    }

    request
        .selection
        .resolve(request.discovery, current)
        .map_err(|err| match &err {
            SelectionError::Membership(MembershipError::UnknownDevice(id)) => GateViolation {
                gate: GateClass::HealthEpochMembership,
                device: Some(id.clone()),
                failing_fact: format!(
                    "device {id} is not recorded in the snapshot (unknown or replaced)"
                ),
            },
            SelectionError::Membership(MembershipError::StaleEpoch {
                id,
                recorded,
                current,
            }) => GateViolation {
                gate: GateClass::HealthEpochMembership,
                device: Some(id.clone()),
                failing_fact: format!(
                    "device {id} recorded at generation {recorded}, current is {current}"
                ),
            },
            SelectionError::BelowMinimum { min, actual } => GateViolation {
                gate: GateClass::HealthEpochMembership,
                device: None,
                failing_fact: format!(
                    "selection resolved {actual} member(s), below declared minimum {min}"
                ),
            },
            SelectionError::AboveMaximum { max, actual } => GateViolation {
                gate: GateClass::HealthEpochMembership,
                device: None,
                failing_fact: format!(
                    "selection resolved {actual} member(s), above declared maximum {max}"
                ),
            },
        })
}

/// Gate 1 (per plan): every assigned device must be a member of the resolved
/// selection (which already guarantees current-epoch membership).
fn gate_membership(plan: &CandidatePlan, set: &DeviceSet) -> Vec<GateViolation> {
    let mut out = Vec::new();
    for assignment in &plan.assignments {
        if !set.contains(&assignment.device) {
            out.push(GateViolation {
                gate: GateClass::HealthEpochMembership,
                device: Some(assignment.device.clone()),
                failing_fact: format!(
                    "device {} is not a current-epoch member of the resolved selection",
                    assignment.device
                ),
            });
        }
    }
    out
}

/// Gate 2: every operation placed on a device must be executable there —
/// compute capability, SM count, and dtype surface from the snapshot's
/// capability facts (C2 §1.2 compatibility).
fn gate_compatibility(
    plan: &CandidatePlan,
    discovery: &DeviceDiscoverySnapshot,
) -> Vec<GateViolation> {
    let mut out = Vec::new();
    for assignment in &plan.assignments {
        let Some(entry) = find_device(discovery, &assignment.device) else {
            out.push(GateViolation {
                gate: GateClass::Compatibility,
                device: Some(assignment.device.clone()),
                failing_fact: format!(
                    "no capability facts recorded for {} in the snapshot",
                    assignment.device
                ),
            });
            continue;
        };
        let caps = &entry.capabilities;
        let required = &assignment.required_capabilities;

        if let Some(required_cc) = required.min_compute_capability {
            let got = caps.compute_capability;
            if (got.major, got.minor) < (required_cc.major, required_cc.minor) {
                out.push(GateViolation {
                    gate: GateClass::Compatibility,
                    device: Some(assignment.device.clone()),
                    failing_fact: format!(
                        "compute capability {}.{} is below required {}.{}",
                        got.major, got.minor, required_cc.major, required_cc.minor
                    ),
                });
            }
        }

        for name in DTYPE_NAMES {
            if dtype_supported(required.required_dtypes, name)
                && !dtype_supported(caps.dtype_surface, name)
            {
                out.push(GateViolation {
                    gate: GateClass::Compatibility,
                    device: Some(assignment.device.clone()),
                    failing_fact: format!(
                        "device {} lacks required dtype {name}",
                        assignment.device
                    ),
                });
            }
        }

        if let Some(required_sms) = required.min_sm_count {
            if caps.sm_count < required_sms {
                out.push(GateViolation {
                    gate: GateClass::Compatibility,
                    device: Some(assignment.device.clone()),
                    failing_fact: format!(
                        "sm count {} is below required {}",
                        caps.sm_count, required_sms
                    ),
                });
            }
        }
    }
    out
}

/// Gate 3: the declared requirements per device (incl. partition budgets)
/// must fit that device's declared admitted budget, and the plan's declared
/// total must fit the declared total budget (C2 §1.2 required memory).
fn gate_memory(plan: &CandidatePlan, constraints: &DeclaredConstraints) -> Vec<GateViolation> {
    let mut out = Vec::new();
    for assignment in &plan.assignments {
        let Some(limit) = constraints.per_device_budget.get(&assignment.device) else {
            out.push(GateViolation {
                gate: GateClass::RequiredMemory,
                device: Some(assignment.device.clone()),
                failing_fact: format!(
                    "no declared admitted budget for device {} (fail-closed)",
                    assignment.device
                ),
            });
            continue;
        };
        match assignment.required.total_bytes() {
            Some(total) if total <= limit.get() => {}
            Some(total) => out.push(GateViolation {
                gate: GateClass::RequiredMemory,
                device: Some(assignment.device.clone()),
                failing_fact: format!(
                    "budget_exceeded: declared {total} bytes exceeds policy limit {} bytes (SafePhysicalLimit)",
                    limit.get()
                ),
            }),
            // Declared ledger overflow is still fail-closed as
            // budget_exceeded (MD1-V1: deterministic rejection).
            None => out.push(GateViolation {
                gate: GateClass::RequiredMemory,
                device: Some(assignment.device.clone()),
                failing_fact: format!(
                    "budget_exceeded: declared ledger overflow; policy limit {} bytes",
                    limit.get()
                ),
            }),
        }
    }
    if let Some(total_budget) = constraints.total_budget {
        if plan.declared_total_bytes > total_budget.get() {
            out.push(GateViolation {
                gate: GateClass::RequiredMemory,
                device: None,
                failing_fact: format!(
                    "declared total {} bytes exceeds declared total budget {} bytes",
                    plan.declared_total_bytes,
                    total_budget.get()
                ),
            });
        }
    }
    out
}

/// Gate 4: every transfer traverses only admitted directed links (C2 §1.2
/// topology/peer-access) — NOT-ATTEMPTED, rejected, absent, and unmeasured
/// links are never assumed.
fn gate_topology(plan: &CandidatePlan, topology: &DeviceTopologySnapshot) -> Vec<GateViolation> {
    let mut out = Vec::new();
    for transfer in &plan.transfers {
        if let Err(err) = topology.traversal_allowed(&transfer.from, &transfer.to) {
            let failing_fact = match &err {
                LinkGateError::UnknownEndpoint { endpoint } => format!(
                    "transfer {} → {}: endpoint {endpoint} is not a device in the topology",
                    transfer.from, transfer.to
                ),
                LinkGateError::NoLinkRecorded { from, to } => format!(
                    "transfer {from} → {to}: no directed link recorded; unmeasured pairs are never assumed"
                ),
                LinkGateError::NotAttempted { from, to } => format!(
                    "transfer {from} → {to}: directed link is NOT ATTEMPTED"
                ),
                LinkGateError::Rejected { from, to, reason } => format!(
                    "transfer {from} → {to}: directed link rejected ({reason})"
                ),
            };
            out.push(GateViolation {
                gate: GateClass::Topology,
                device: None,
                failing_fact,
            });
        }
    }
    out
}

/// Pass 2: rank only the gate-passing plans by the declared objectives in
/// order, then by stable [`PhysicalDeviceId`] — never ordinal.
fn rank_plans(
    passers: Vec<(usize, &CandidatePlan)>,
    objectives: &[Objective],
    constraints: &DeclaredConstraints,
) -> Vec<RankedPlan> {
    if passers.is_empty() {
        return Vec::new();
    }

    let mut scored: Vec<(
        (usize, &CandidatePlan),
        Vec<u64>,
        BTreeSet<PhysicalDeviceId>,
    )> = passers
        .into_iter()
        .map(|(idx, plan)| {
            let values: Vec<u64> = objectives
                .iter()
                .map(|objective| objective_value(plan, *objective, constraints))
                .collect();
            let devices: BTreeSet<PhysicalDeviceId> = plan
                .assignments
                .iter()
                .map(|assignment| assignment.device.clone())
                .collect();
            ((idx, plan), values, devices)
        })
        .collect();

    scored.sort_by(|(_, a_values, a_devices), (_, b_values, b_devices)| {
        for (i, objective) in objectives.iter().enumerate() {
            let ord = match objective.direction() {
                ObjectiveDirection::Minimize => a_values[i].cmp(&b_values[i]),
                ObjectiveDirection::Maximize => b_values[i].cmp(&a_values[i]),
            };
            if ord != Ordering::Equal {
                return ord;
            }
        }
        // Final tie-break: stable identity ordering — never ordinal. The
        // ordinal is a locator only and never participates (naming
        // contract §1).
        a_devices.cmp(b_devices)
    });

    scored
        .into_iter()
        .enumerate()
        .map(|(rank, ((_idx, plan), values, devices))| {
            let objective_scores = objectives
                .iter()
                .zip(&values)
                .map(|(objective, measured)| ObjectiveScore {
                    objective: *objective,
                    value: *measured,
                })
                .collect();
            RankedPlan {
                rank: rank + 1,
                plan: plan.name.clone(),
                devices: devices.into_iter().collect(),
                scores: objective_scores,
            }
        })
        .collect()
}

/// One objective's score for a plan. Missing declared facts are least-favored
/// — an objective never rejects a plan for lacking a fact (C2 §1.1: no
/// objective has reject power).
fn objective_value(
    plan: &CandidatePlan,
    objective: Objective,
    constraints: &DeclaredConstraints,
) -> u64 {
    match objective {
        Objective::Latency => plan.facts.latency_nanos.unwrap_or(u64::MAX),
        Objective::Throughput => plan.facts.throughput_tokens_per_sec.unwrap_or(0),
        Objective::CapacityReserve => capacity_reserve_bytes(plan, constraints),
        Objective::CacheAffinity => plan.facts.cache_affinity_hit_rate_milli.unwrap_or(0),
        Objective::ExpertHotness => plan.facts.expert_hotness_score.unwrap_or(0),
        Objective::TransferBudget => plan.facts.transfer_bytes.unwrap_or(u64::MAX),
        Objective::Power => plan.facts.estimated_power_milliwatts.unwrap_or(u64::MAX),
    }
}

/// Computed headroom: the sum over assigned devices of (declared admitted
/// budget − declared requirement). For gate-passing plans every assigned
/// device has a declared budget and fits it, so the reserve is honest
/// (C2 §1.2 capacity reserve).
fn capacity_reserve_bytes(plan: &CandidatePlan, constraints: &DeclaredConstraints) -> u64 {
    let mut reserve = 0u64;
    for assignment in &plan.assignments {
        if let Some(limit) = constraints.per_device_budget.get(&assignment.device) {
            if let Some(required) = assignment.required.total_bytes() {
                reserve = reserve.saturating_add(limit.get().saturating_sub(required));
            }
        }
    }
    reserve
}

/// The six raw arithmetic dtypes of the snapshot's dtype surface (T1 §2).
const DTYPE_NAMES: [&str; 6] = ["f32", "f64", "f16", "bf16", "i8", "i32"];

/// Whether a [`DtypeSurface`] reports support for the named dtype.
#[must_use]
fn dtype_supported(surface: DtypeSurface, name: &str) -> bool {
    match name {
        "f32" => surface.f32,
        "f64" => surface.f64,
        "f16" => surface.f16,
        "bf16" => surface.bf16,
        "i8" => surface.i8,
        "i32" => surface.i32,
        _ => false,
    }
}

fn find_device<'a>(
    discovery: &'a DeviceDiscoverySnapshot,
    id: &PhysicalDeviceId,
) -> Option<&'a DeviceDiscoveryEntry> {
    discovery
        .devices()
        .values()
        .find(|entry| &entry.identity == id)
}

/// FNV-1a 64 — dependency-free, deterministic across processes and machines.
#[must_use]
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Lowercase 16-digit hex of a 64-bit value.
#[must_use]
fn hex64(value: u64) -> String {
    format!("{value:016x}")
}

#[cfg(test)]
#[path = "policy_test.rs"]
mod tests;
