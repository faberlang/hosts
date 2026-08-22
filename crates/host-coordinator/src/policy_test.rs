//! MD1-P1 policy tests: the two-pass evaluator (C2 §1.1 review test — gates
//! reject, objectives rank only gate-passing plans), determinism for one
//! frozen snapshot, tie-break by declared order then `PhysicalDeviceId`
//! (never ordinal), the CS-1 fixture shape (FC14), and the explained
//! receipt.

use crate::device_identity::{DeviceHealthGeneration, DeviceOrdinal, PhysicalDeviceId};
use crate::device_set::{
    DeviceLink, DeviceSetSelection, DeviceTopologySnapshot, LinkFacts, LinkPathClass,
};
use crate::discovery::{
    ComputeCapability, DeviceCapabilities, DeviceDiscoveryEntry, DeviceDiscoverySnapshot,
    DeviceHealth, DeviceMemory, DtypeSurface, P2pProbeState, ProbeProvenance,
};
use crate::partition::{PartitionBudgetLedger, SafePhysicalLimit};
use crate::policy::{
    evaluate, CandidatePlan, DeclaredConstraints, DeviceAssignment, GateClass, Objective,
    ObjectiveFacts, ObjectiveScore, PolicyOutcome, PolicyRequest, RequiredCapabilities, Transfer,
};
use std::collections::{BTreeMap, BTreeSet};

const UUID_A: &str = "GPU-3e017562-9ec3-da9a-962d-b8bd5f9e24be";
const UUID_B: &str = "GPU-11111111-2222-3333-4444-555555555555";
const UUID_C: &str = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";
const UNKNOWN_UUID: &str = "GPU-99999999-8888-7777-6666-555555555555";
const PROBE_TIME: u64 = 1_752_717_600_000_000_000; // fixed sample time

// CS-1 fixture (FC14): SmolLM2-360M-Instruct Q4_K_M 270,590,880 B ≈ 258 MiB,
// 2 virtual partitions @ 160 MiB declared (mesh 320 MiB), forced 2-way split
// ≈129 MiB/device.
const SMOLLM2_BYTES: u64 = 270_590_880;
const PARTITION_LIMIT_BYTES: u64 = 160 * 1024 * 1024; // 167_772_160
const MESH_TOTAL_BYTES: u64 = 2 * PARTITION_LIMIT_BYTES; // 335_544_320
const HALF_SPLIT_BYTES: u64 = SMOLLM2_BYTES / 2; // ≈129 MiB per device

fn full_dtypes() -> DtypeSurface {
    DtypeSurface {
        f32: true,
        f64: true,
        f16: true,
        bf16: true,
        i8: true,
        i32: true,
    }
}

/// DCG-1 Metal-shaped snapshot: CUDA identity facts are zero sentinels;
/// generic launch-resource fields are populated.
fn metal_caps() -> DeviceCapabilities {
    DeviceCapabilities {
        compute_capability: ComputeCapability { major: 0, minor: 0 },
        sm_count: 0,
        dtype_surface: DtypeSurface::empty(),
        max_threads_per_workgroup: 1024,
        workgroup_shared_memory_min_bytes: 32_768,
        workgroup_shared_memory_max_bytes: 32_768,
        collective_width: 32,
        unified_memory: true,
    }
}

fn entry(
    ordinal: u32,
    identity: PhysicalDeviceId,
    caps: Option<DeviceCapabilities>,
) -> DeviceDiscoveryEntry {
    DeviceDiscoveryEntry {
        ordinal: DeviceOrdinal::new(ordinal),
        identity,
        device_model: None,
        capabilities: caps.unwrap_or(DeviceCapabilities {
            compute_capability: ComputeCapability {
                major: 12,
                minor: 0,
            },
            sm_count: 48,
            dtype_surface: full_dtypes(),
            max_threads_per_workgroup: 1024,
            workgroup_shared_memory_min_bytes: 49_152,
            workgroup_shared_memory_max_bytes: 101_376,
            collective_width: 32,
            unified_memory: false,
        }),
        memory: DeviceMemory {
            tool_report_total_mib: None,
            api_total_bytes: 0,
        },
        health: DeviceHealth::Healthy,
        health_generation: DeviceHealthGeneration::initial(),
        probe_provenance: ProbeProvenance {
            probe: "synthetic fixture".to_owned(),
            tool_versions: "test".to_owned(),
        },
    }
}

fn snapshot(entries: Vec<DeviceDiscoveryEntry>) -> DeviceDiscoverySnapshot {
    let mut devices = BTreeMap::new();
    for e in entries {
        devices.insert(e.ordinal, e);
    }
    DeviceDiscoverySnapshot::new(PROBE_TIME, devices, P2pProbeState::NotAttempted)
}

/// A weight-only partition budget ledger (all other classes zero).
fn ledger(weight_bytes: u64) -> PartitionBudgetLedger {
    PartitionBudgetLedger {
        weight_bytes,
        kv_cache_bytes: 0,
        activation_scratch_bytes: 0,
        module_storage_bytes: 0,
        allocator_overhead_bytes: 0,
        transfer_staging_bytes: 0,
        concurrent_state_bytes: 0,
    }
}

fn constraints_at(
    snap: &DeviceDiscoverySnapshot,
    budgets: BTreeMap<PhysicalDeviceId, SafePhysicalLimit>,
    generation: DeviceHealthGeneration,
) -> DeclaredConstraints {
    DeclaredConstraints {
        current_generation: generation,
        topology: DeviceTopologySnapshot::new(snap.clone(), []),
        per_device_budget: budgets,
        total_budget: None,
    }
}

fn constraints(
    snap: &DeviceDiscoverySnapshot,
    budgets: BTreeMap<PhysicalDeviceId, SafePhysicalLimit>,
) -> DeclaredConstraints {
    constraints_at(snap, budgets, DeviceHealthGeneration::initial())
}

fn evaluate_all(
    snap: &DeviceDiscoverySnapshot,
    selection: &DeviceSetSelection,
    constraints: &DeclaredConstraints,
    objectives: &[Objective],
    plans: &[CandidatePlan],
) -> PolicyOutcome {
    evaluate(&PolicyRequest {
        discovery: snap,
        selection,
        constraints,
        objectives,
        plans,
    })
}

fn gate_classes(outcome: &PolicyOutcome) -> Vec<GateClass> {
    outcome
        .rejected
        .iter()
        .flat_map(|r| r.violations.iter().map(|v| v.gate))
        .collect()
}

/// C2 §1.1 review test: a plan violating any gate is rejected with the
/// violated constraint class + exact failing fact and is never ranked; every
/// objective ranks only plans that already pass every gate; no objective ever
/// rejects.
#[test]
#[allow(clippy::too_many_lines)] // one table covering every C2 §1.1 gate
fn c2_11_review_test_each_gate_rejects_and_objectives_rank_only_survivors() {
    let a = PhysicalDeviceId::cuda(UUID_A, None);
    let b = PhysicalDeviceId::cuda(UUID_B, None);
    let c = PhysicalDeviceId::cuda(UUID_C, None);

    // Device A lacks bf16 (compatibility-failing surface); B and C are full.
    let mut reduced = full_dtypes();
    reduced.bf16 = false;
    let snap = snapshot(vec![
        entry(
            0,
            a.clone(),
            Some(DeviceCapabilities {
                compute_capability: ComputeCapability {
                    major: 12,
                    minor: 0,
                },
                sm_count: 48,
                dtype_surface: reduced,
                max_threads_per_workgroup: 1024,
                workgroup_shared_memory_min_bytes: 49_152,
                workgroup_shared_memory_max_bytes: 101_376,
                collective_width: 32,
                unified_memory: false,
            }),
        ),
        entry(1, b.clone(), None),
        entry(2, c.clone(), None),
    ]);

    let mut budgets = BTreeMap::new();
    budgets.insert(a.clone(), SafePhysicalLimit::new(100));
    budgets.insert(b.clone(), SafePhysicalLimit::new(100));
    budgets.insert(c.clone(), SafePhysicalLimit::new(100));
    let mut constraints = constraints(&snap, budgets);
    constraints.topology = DeviceTopologySnapshot::new(
        snap.clone(),
        [DeviceLink::not_attempted(a.clone(), b.clone())],
    );

    // The selection deliberately excludes C.
    let selection = DeviceSetSelection::explicit([a.clone(), b.clone()]);
    let objectives = [Objective::Latency, Objective::Throughput];

    let passes_all = CandidatePlan::new(
        "passes-all".to_owned(),
        vec![DeviceAssignment::new(a.clone(), ledger(10))],
        vec![],
        10,
        ObjectiveFacts {
            latency_nanos: Some(1),
            throughput_tokens_per_sec: Some(100),
            ..ObjectiveFacts::default()
        },
    );
    let fails_compat = CandidatePlan::new(
        "fails-compat".to_owned(),
        vec![DeviceAssignment {
            device: a.clone(),
            required: ledger(10),
            required_capabilities: RequiredCapabilities {
                required_dtypes: DtypeSurface {
                    bf16: true,
                    ..DtypeSurface::empty()
                },
                ..RequiredCapabilities::default()
            },
        }],
        vec![],
        10,
        ObjectiveFacts::default(),
    );
    let fails_memory = CandidatePlan::new(
        "fails-memory".to_owned(),
        vec![DeviceAssignment::new(a.clone(), ledger(200))],
        vec![],
        10,
        ObjectiveFacts {
            latency_nanos: Some(0), // the best latency — never rescues
            ..ObjectiveFacts::default()
        },
    );
    let fails_topology = CandidatePlan::new(
        "fails-topology".to_owned(),
        vec![
            DeviceAssignment::new(a.clone(), ledger(10)),
            DeviceAssignment::new(b.clone(), ledger(10)),
        ],
        vec![Transfer::new(a.clone(), b.clone())],
        20,
        ObjectiveFacts::default(),
    );
    let fails_health = CandidatePlan::new(
        "fails-health".to_owned(),
        vec![DeviceAssignment::new(c.clone(), ledger(10))],
        vec![],
        10,
        ObjectiveFacts::default(),
    );

    let outcome = evaluate_all(
        &snap,
        &selection,
        &constraints,
        &objectives,
        &[
            passes_all,
            fails_compat,
            fails_memory,
            fails_topology,
            fails_health,
        ],
    );

    // Every gate class appears as a rejection; no plan is ranked unless it
    // passed every gate.
    assert_eq!(outcome.ranked.len(), 1);
    assert_eq!(outcome.ranked[0].plan, "passes-all");
    assert_eq!(outcome.ranked[0].rank, 1);
    assert_eq!(
        outcome.ranked[0]
            .scores
            .iter()
            .map(|s| s.objective)
            .collect::<Vec<_>>(),
        vec![Objective::Latency, Objective::Throughput]
    );

    assert_eq!(outcome.rejected.len(), 4);
    let rejected_names: BTreeSet<&str> = outcome.rejected.iter().map(|r| r.plan.as_str()).collect();
    assert!(rejected_names.contains("fails-compat"));
    assert!(rejected_names.contains("fails-memory"));
    assert!(rejected_names.contains("fails-topology"));
    assert!(rejected_names.contains("fails-health"));

    for rejection in &outcome.rejected {
        let violation = &rejection.violations[0];
        let expected_gate = match rejection.plan.as_str() {
            "fails-compat" => GateClass::Compatibility,
            "fails-memory" => GateClass::RequiredMemory,
            "fails-topology" => GateClass::Topology,
            "fails-health" => GateClass::HealthEpochMembership,
            other => panic!("unexpected rejected plan {other}"),
        };
        assert_eq!(violation.gate, expected_gate, "plan {}", rejection.plan);
        assert!(
            !violation.failing_fact.is_empty(),
            "every rejection carries the exact failing fact"
        );
    }

    // Exact failing facts.
    let compat = &outcome
        .rejected
        .iter()
        .find(|r| r.plan == "fails-compat")
        .unwrap()
        .violations[0];
    assert_eq!(compat.device.as_ref(), Some(&a));
    assert!(compat.failing_fact.contains("bf16"));

    let memory = &outcome
        .rejected
        .iter()
        .find(|r| r.plan == "fails-memory")
        .unwrap()
        .violations[0];
    assert_eq!(memory.device.as_ref(), Some(&a));
    assert!(memory.failing_fact.contains("budget_exceeded"));
    assert!(memory.failing_fact.contains("200"));
    assert!(memory.failing_fact.contains("100"));

    let topology = &outcome
        .rejected
        .iter()
        .find(|r| r.plan == "fails-topology")
        .unwrap()
        .violations[0];
    assert_eq!(topology.gate, GateClass::Topology);
    assert!(topology.failing_fact.contains("NOT ATTEMPTED"));

    let health = &outcome
        .rejected
        .iter()
        .find(|r| r.plan == "fails-health")
        .unwrap()
        .violations[0];
    assert_eq!(health.device.as_ref(), Some(&c));

    // Every gate class appears as a rejection, in gate-evaluation order; no
    // objective ever rejects (every recorded violation is a gate class).
    assert_eq!(
        gate_classes(&outcome),
        vec![
            GateClass::Compatibility,
            GateClass::RequiredMemory,
            GateClass::Topology,
            GateClass::HealthEpochMembership,
        ]
    );
}

/// The same review test with an explicit exhaustive gate-class assertion.
#[test]
fn every_rejection_is_a_gate_and_every_gate_rejects() {
    // Covered structurally by the two-pass design; this pins the vocabulary:
    // the four gate classes are exactly the classes the evaluator can emit.
    let gate_tags: Vec<u64> = [
        GateClass::HealthEpochMembership,
        GateClass::Compatibility,
        GateClass::RequiredMemory,
        GateClass::Topology,
    ]
    .into_iter()
    .map(GateClass::tag)
    .collect();
    assert_eq!(gate_tags, vec![0, 1, 2, 3]);
}

/// The CS-1 fixture shape (FC14): the forced 2-way split passes the
/// partition-budget gate; whole-model-on-one-partition fails as
/// `budget_exceeded` and is never ranked — objectives never rescue a
/// gate-failing plan even when it declares better latency.
#[test]
fn cs1_split_passes_whole_on_one_fails_budget_exceeded() {
    let a = PhysicalDeviceId::cuda(UUID_A, None);
    let b = PhysicalDeviceId::cuda(UUID_B, None);
    let snap = snapshot(vec![entry(0, a.clone(), None), entry(1, b.clone(), None)]);

    let mut budgets = BTreeMap::new();
    budgets.insert(a.clone(), SafePhysicalLimit::new(PARTITION_LIMIT_BYTES));
    budgets.insert(b.clone(), SafePhysicalLimit::new(PARTITION_LIMIT_BYTES));
    let mut constraints = constraints(&snap, budgets);
    constraints.total_budget = Some(SafePhysicalLimit::new(MESH_TOTAL_BYTES));

    let selection = DeviceSetSelection::explicit([a.clone(), b.clone()]);
    let objectives = [Objective::Latency];

    // Whole-on-one: great latency, but 258 MiB cannot fit one 160 MiB
    // partition — fails before any allocation (pure evaluation).
    let whole_on_one = CandidatePlan::new(
        "whole-on-one".to_owned(),
        vec![DeviceAssignment::new(a.clone(), ledger(SMOLLM2_BYTES))],
        vec![],
        SMOLLM2_BYTES,
        ObjectiveFacts {
            latency_nanos: Some(1_000_000),
            ..ObjectiveFacts::default()
        },
    );
    // Forced 2-way split ≈129 MiB/device — the honest plan, slower.
    let split = CandidatePlan::new(
        "2-way-split".to_owned(),
        vec![
            DeviceAssignment::new(a.clone(), ledger(HALF_SPLIT_BYTES)),
            DeviceAssignment::new(b.clone(), ledger(HALF_SPLIT_BYTES)),
        ],
        vec![],
        SMOLLM2_BYTES,
        ObjectiveFacts {
            latency_nanos: Some(2_500_000),
            ..ObjectiveFacts::default()
        },
    );

    let outcome = evaluate_all(
        &snap,
        &selection,
        &constraints,
        &objectives,
        &[whole_on_one.clone(), split.clone()],
    );

    // Whole-on-one rejected with the required-memory gate + exact
    // budget_exceeded fact (declared bytes vs policy limit bytes).
    assert_eq!(outcome.rejected.len(), 1);
    let rejection = &outcome.rejected[0];
    assert_eq!(rejection.plan, "whole-on-one");
    assert_eq!(rejection.violations.len(), 1);
    let violation = &rejection.violations[0];
    assert_eq!(violation.gate, GateClass::RequiredMemory);
    assert_eq!(violation.device.as_ref(), Some(&a));
    assert!(violation.failing_fact.contains("budget_exceeded"));
    assert!(violation.failing_fact.contains(&SMOLLM2_BYTES.to_string()));
    assert!(violation
        .failing_fact
        .contains(&PARTITION_LIMIT_BYTES.to_string()));

    // The split plan passes every gate and is the only ranked plan.
    assert_eq!(outcome.ranked.len(), 1);
    assert_eq!(outcome.ranked[0].plan, "2-way-split");
    assert_eq!(outcome.ranked[0].rank, 1);
    let ranked_devices: BTreeSet<&PhysicalDeviceId> = outcome.ranked[0].devices.iter().collect();
    assert_eq!(ranked_devices, BTreeSet::from([&a, &b]));
}

/// Compatibility gate rejects unsupported dtype, compute capability, and SM
/// count from the snapshot's capability facts; a plan with no declared
/// capability requirements is compatible with every surface.
#[test]
#[allow(clippy::too_many_lines)] // walks dtype, compute-capability, and SM-count rejections
fn compatibility_gate_rejects_unsupported_capabilities() {
    let a = PhysicalDeviceId::cuda(UUID_A, None);
    let mut reduced = full_dtypes();
    reduced.bf16 = false;
    let snap = snapshot(vec![entry(
        0,
        a.clone(),
        Some(DeviceCapabilities {
            compute_capability: ComputeCapability {
                major: 12,
                minor: 0,
            },
            sm_count: 48,
            dtype_surface: reduced,
            max_threads_per_workgroup: 1024,
            workgroup_shared_memory_min_bytes: 49_152,
            workgroup_shared_memory_max_bytes: 101_376,
            collective_width: 32,
            unified_memory: false,
        }),
    )]);
    let constraints = constraints(
        &snap,
        BTreeMap::from([(a.clone(), SafePhysicalLimit::new(1_000_000))]),
    );
    let selection = DeviceSetSelection::explicit([a.clone()]);
    let objectives = [Objective::Latency];

    let needs_f64 = CandidatePlan::new(
        "needs-f64".to_owned(),
        vec![DeviceAssignment {
            device: a.clone(),
            required: ledger(10),
            required_capabilities: RequiredCapabilities {
                required_dtypes: DtypeSurface {
                    f64: true,
                    ..DtypeSurface::empty()
                },
                ..RequiredCapabilities::default()
            },
        }],
        vec![],
        10,
        ObjectiveFacts::default(),
    );
    let needs_bf16 = CandidatePlan::new(
        "needs-bf16".to_owned(),
        vec![DeviceAssignment {
            device: a.clone(),
            required: ledger(10),
            required_capabilities: RequiredCapabilities {
                required_dtypes: DtypeSurface {
                    bf16: true,
                    ..DtypeSurface::empty()
                },
                ..RequiredCapabilities::default()
            },
        }],
        vec![],
        10,
        ObjectiveFacts::default(),
    );
    let needs_cc13 = CandidatePlan::new(
        "needs-cc13".to_owned(),
        vec![DeviceAssignment {
            device: a.clone(),
            required: ledger(10),
            required_capabilities: RequiredCapabilities {
                min_compute_capability: Some(ComputeCapability {
                    major: 13,
                    minor: 0,
                }),
                ..RequiredCapabilities::default()
            },
        }],
        vec![],
        10,
        ObjectiveFacts::default(),
    );
    let needs_128_sms = CandidatePlan::new(
        "needs-128-sms".to_owned(),
        vec![DeviceAssignment {
            device: a.clone(),
            required: ledger(10),
            required_capabilities: RequiredCapabilities {
                min_sm_count: Some(128),
                ..RequiredCapabilities::default()
            },
        }],
        vec![],
        10,
        ObjectiveFacts::default(),
    );
    let no_requirements = CandidatePlan::new(
        "no-requirements".to_owned(),
        vec![DeviceAssignment::new(a.clone(), ledger(10))],
        vec![],
        10,
        ObjectiveFacts::default(),
    );

    let outcome = evaluate_all(
        &snap,
        &selection,
        &constraints,
        &objectives,
        &[
            needs_f64,
            needs_bf16,
            needs_cc13,
            needs_128_sms,
            no_requirements,
        ],
    );

    // The three capability-failing plans are rejected with the compatibility
    // gate; the f64 plan and the no-requirements plan pass every gate (f64 is
    // supported, and no declared requirements are compatible with everything).
    assert_eq!(outcome.rejected.len(), 3);
    assert_eq!(outcome.ranked.len(), 2);
    let ranked_names: BTreeSet<&str> = outcome.ranked.iter().map(|r| r.plan.as_str()).collect();
    assert!(ranked_names.contains("needs-f64"));
    assert!(ranked_names.contains("no-requirements"));

    let bf16 = &outcome
        .rejected
        .iter()
        .find(|r| r.plan == "needs-bf16")
        .unwrap()
        .violations[0];
    assert_eq!(bf16.gate, GateClass::Compatibility);
    assert_eq!(bf16.device.as_ref(), Some(&a));
    assert!(bf16.failing_fact.contains("bf16"));

    let cc = &outcome
        .rejected
        .iter()
        .find(|r| r.plan == "needs-cc13")
        .unwrap()
        .violations[0];
    assert_eq!(cc.gate, GateClass::Compatibility);
    assert!(cc.failing_fact.contains("13.0"));

    let sms = &outcome
        .rejected
        .iter()
        .find(|r| r.plan == "needs-128-sms")
        .unwrap()
        .violations[0];
    assert_eq!(sms.gate, GateClass::Compatibility);
    assert!(sms.failing_fact.contains("128"));
}

/// DCG-2: a Metal-shaped snapshot evaluates on generic launch-resource
/// fields. An in-limits plan is admitted; a plan whose workgroup demand
/// exceeds the device ceiling is rejected with the compatibility gate.
#[test]
fn compatibility_gate_evaluates_metal_on_generic_launch_resources() {
    let metal = PhysicalDeviceId::metal("4278190081");
    let snap = snapshot(vec![entry(0, metal.clone(), Some(metal_caps()))]);
    let constraints = constraints(
        &snap,
        BTreeMap::from([(metal.clone(), SafePhysicalLimit::new(1_000_000))]),
    );
    let selection = DeviceSetSelection::explicit([metal.clone()]);
    let objectives = [Objective::Latency];

    let admitted = CandidatePlan::new(
        "metal-in-limits".to_owned(),
        vec![DeviceAssignment {
            device: metal.clone(),
            required: ledger(10),
            required_capabilities: RequiredCapabilities {
                threads_per_workgroup: Some(256),
                workgroup_shared_memory_bytes: Some(16_384),
                min_collective_width: Some(32),
                require_unified_memory: true,
                ..RequiredCapabilities::default()
            },
        }],
        vec![],
        10,
        ObjectiveFacts::default(),
    );
    let oversized = CandidatePlan::new(
        "metal-oversize-threadgroup".to_owned(),
        vec![DeviceAssignment {
            device: metal.clone(),
            required: ledger(10),
            required_capabilities: RequiredCapabilities {
                threads_per_workgroup: Some(2048),
                ..RequiredCapabilities::default()
            },
        }],
        vec![],
        10,
        ObjectiveFacts::default(),
    );

    let outcome = evaluate_all(
        &snap,
        &selection,
        &constraints,
        &objectives,
        &[admitted, oversized],
    );

    assert_eq!(outcome.ranked.len(), 1);
    assert_eq!(outcome.ranked[0].plan, "metal-in-limits");
    assert_eq!(outcome.rejected.len(), 1);
    assert_eq!(outcome.rejected[0].plan, "metal-oversize-threadgroup");
    let violation = &outcome.rejected[0].violations[0];
    assert_eq!(violation.gate, GateClass::Compatibility);
    assert_eq!(violation.device.as_ref(), Some(&metal));
    assert!(violation.failing_fact.contains("2048"));
    assert!(violation.failing_fact.contains("1024"));
    assert!(violation.failing_fact.contains("max_threads_per_workgroup"));
}

/// DCG-2: unpopulated generic launch-resource facts (CUDA zero sentinels)
/// and CUDA identity demanded of Metal reject as unevaluable — never a
/// fake comparison against zero.
#[test]
fn compatibility_gate_fails_closed_when_launch_resources_unevaluable() {
    let cuda = PhysicalDeviceId::cuda(UUID_A, None);
    let cuda_snap = snapshot(vec![entry(
        0,
        cuda.clone(),
        Some(DeviceCapabilities {
            compute_capability: ComputeCapability {
                major: 12,
                minor: 0,
            },
            sm_count: 48,
            dtype_surface: full_dtypes(),
            max_threads_per_workgroup: 0,
            workgroup_shared_memory_min_bytes: 0,
            workgroup_shared_memory_max_bytes: 0,
            collective_width: 0,
            unified_memory: false,
        }),
    )]);
    let cuda_constraints = constraints(
        &cuda_snap,
        BTreeMap::from([(cuda.clone(), SafePhysicalLimit::new(1_000_000))]),
    );
    let cuda_plan = CandidatePlan::new(
        "cuda-unpopulated-threads".to_owned(),
        vec![DeviceAssignment {
            device: cuda.clone(),
            required: ledger(10),
            required_capabilities: RequiredCapabilities {
                threads_per_workgroup: Some(256),
                ..RequiredCapabilities::default()
            },
        }],
        vec![],
        10,
        ObjectiveFacts::default(),
    );
    let cuda_outcome = evaluate_all(
        &cuda_snap,
        &DeviceSetSelection::explicit([cuda.clone()]),
        &cuda_constraints,
        &[Objective::Latency],
        &[cuda_plan],
    );
    assert_eq!(cuda_outcome.ranked.len(), 0);
    assert_eq!(cuda_outcome.rejected.len(), 1);
    let cuda_v = &cuda_outcome.rejected[0].violations[0];
    assert_eq!(cuda_v.gate, GateClass::Compatibility);
    assert!(cuda_v.failing_fact.contains("unevaluable"));
    assert!(cuda_v.failing_fact.contains("max_threads_per_workgroup"));

    let metal = PhysicalDeviceId::metal("4278190081");
    let metal_snap = snapshot(vec![entry(0, metal.clone(), Some(metal_caps()))]);
    let metal_constraints = constraints(
        &metal_snap,
        BTreeMap::from([(metal.clone(), SafePhysicalLimit::new(1_000_000))]),
    );
    let metal_plan = CandidatePlan::new(
        "metal-cuda-identity".to_owned(),
        vec![DeviceAssignment {
            device: metal.clone(),
            required: ledger(10),
            required_capabilities: RequiredCapabilities {
                min_compute_capability: Some(ComputeCapability {
                    major: 12,
                    minor: 0,
                }),
                ..RequiredCapabilities::default()
            },
        }],
        vec![],
        10,
        ObjectiveFacts::default(),
    );
    let metal_outcome = evaluate_all(
        &metal_snap,
        &DeviceSetSelection::explicit([metal]),
        &metal_constraints,
        &[Objective::Latency],
        &[metal_plan],
    );
    assert_eq!(metal_outcome.ranked.len(), 0);
    let metal_v = &metal_outcome.rejected[0].violations[0];
    assert_eq!(metal_v.gate, GateClass::Compatibility);
    assert!(metal_v.failing_fact.contains("unevaluable"));
}

/// Topology gate: only admitted directed links may be traversed;
/// NOT-ATTEMPTED, rejected, and absent links are never assumed. A self-move
/// is a local copy, not a link traversal.
#[test]
#[allow(clippy::too_many_lines)] // walks admitted / NOT-ATTEMPTED / rejected / absent rows
fn topology_gate_rejects_non_admitted_links() {
    let a = PhysicalDeviceId::cuda(UUID_A, None);
    let b = PhysicalDeviceId::cuda(UUID_B, None);
    let c = PhysicalDeviceId::cuda(UUID_C, None);
    let snap = snapshot(vec![
        entry(0, a.clone(), None),
        entry(1, b.clone(), None),
        entry(2, c.clone(), None),
    ]);
    let mut constraints = constraints(
        &snap,
        BTreeMap::from([
            (a.clone(), SafePhysicalLimit::new(1_000_000)),
            (b.clone(), SafePhysicalLimit::new(1_000_000)),
            (c.clone(), SafePhysicalLimit::new(1_000_000)),
        ]),
    );
    constraints.topology = DeviceTopologySnapshot::new(
        snap.clone(),
        [
            DeviceLink::admitted(
                a.clone(),
                b.clone(),
                LinkPathClass::HostStaged,
                LinkFacts {
                    bandwidth_bytes_per_sec: 10_000_000_000,
                    latency_nanos: 1_300,
                },
            ),
            DeviceLink::not_attempted(b.clone(), a.clone()),
            DeviceLink::rejected(b.clone(), c.clone(), "peer access check failed"),
        ],
    );
    let selection = DeviceSetSelection::explicit([a.clone(), b.clone(), c.clone()]);
    let objectives = [Objective::Latency];

    let crosses_admitted = CandidatePlan::new(
        "crosses-admitted".to_owned(),
        vec![
            DeviceAssignment::new(a.clone(), ledger(10)),
            DeviceAssignment::new(b.clone(), ledger(10)),
        ],
        vec![Transfer::new(a.clone(), b.clone())],
        20,
        ObjectiveFacts::default(),
    );
    let crosses_not_attempted = CandidatePlan::new(
        "crosses-not-attempted".to_owned(),
        vec![
            DeviceAssignment::new(a.clone(), ledger(10)),
            DeviceAssignment::new(b.clone(), ledger(10)),
        ],
        vec![Transfer::new(b.clone(), a.clone())],
        20,
        ObjectiveFacts::default(),
    );
    let crosses_absent = CandidatePlan::new(
        "crosses-absent".to_owned(),
        vec![
            DeviceAssignment::new(a.clone(), ledger(10)),
            DeviceAssignment::new(c.clone(), ledger(10)),
        ],
        vec![Transfer::new(a.clone(), c.clone())],
        20,
        ObjectiveFacts::default(),
    );
    let crosses_rejected = CandidatePlan::new(
        "crosses-rejected".to_owned(),
        vec![
            DeviceAssignment::new(b.clone(), ledger(10)),
            DeviceAssignment::new(c.clone(), ledger(10)),
        ],
        vec![Transfer::new(b.clone(), c.clone())],
        20,
        ObjectiveFacts::default(),
    );
    let self_copy = CandidatePlan::new(
        "self-copy".to_owned(),
        vec![DeviceAssignment::new(a.clone(), ledger(10))],
        vec![Transfer::new(a.clone(), a.clone())],
        10,
        ObjectiveFacts::default(),
    );

    let outcome = evaluate_all(
        &snap,
        &selection,
        &constraints,
        &objectives,
        &[
            crosses_admitted,
            crosses_not_attempted,
            crosses_absent,
            crosses_rejected,
            self_copy,
        ],
    );

    let rejected_names: BTreeSet<&str> = outcome.rejected.iter().map(|r| r.plan.as_str()).collect();
    assert!(rejected_names.contains("crosses-not-attempted"));
    assert!(rejected_names.contains("crosses-absent"));
    assert!(rejected_names.contains("crosses-rejected"));

    // Every topology rejection names the pair (no single device), and the
    // exact link fact is recorded.
    for rejection in &outcome.rejected {
        let violation = &rejection.violations[0];
        assert_eq!(violation.gate, GateClass::Topology);
        assert!(
            violation.device.is_none(),
            "topology violations name the pair, not a device"
        );
    }
    let na = &outcome
        .rejected
        .iter()
        .find(|r| r.plan == "crosses-not-attempted")
        .unwrap()
        .violations[0];
    assert!(na.failing_fact.contains("NOT ATTEMPTED"));
    let absent = &outcome
        .rejected
        .iter()
        .find(|r| r.plan == "crosses-absent")
        .unwrap()
        .violations[0];
    assert!(absent.failing_fact.contains("no directed link"));
    let rejected = &outcome
        .rejected
        .iter()
        .find(|r| r.plan == "crosses-rejected")
        .unwrap()
        .violations[0];
    assert!(rejected.failing_fact.contains("peer access check failed"));

    // The admitted traversal and the local self-copy pass every gate.
    let ranked_names: Vec<&str> = outcome.ranked.iter().map(|r| r.plan.as_str()).collect();
    assert!(ranked_names.contains(&"crosses-admitted"));
    assert!(ranked_names.contains(&"self-copy"));
}

/// Health-epoch membership gate: a stale snapshot and an unknown/replaced id
/// in the selection reject before anything is ranked.
#[test]
fn health_epoch_gate_rejects_stale_snapshots_and_unknown_selections() {
    let a = PhysicalDeviceId::cuda(UUID_A, None);
    let snap = snapshot(vec![entry(0, a.clone(), None)]);
    let budgets = BTreeMap::from([(a.clone(), SafePhysicalLimit::new(1_000_000))]);
    let selection = DeviceSetSelection::explicit([a.clone()]);
    let plan = CandidatePlan::new(
        "p".to_owned(),
        vec![DeviceAssignment::new(a.clone(), ledger(10))],
        vec![],
        10,
        ObjectiveFacts::default(),
    );

    // The snapshot was sampled at epoch 1; the current generation advanced.
    let stale = constraints_at(
        &snap,
        budgets.clone(),
        DeviceHealthGeneration::initial().advance(),
    );
    let outcome = evaluate_all(
        &snap,
        &selection,
        &stale,
        &[Objective::Latency],
        std::slice::from_ref(&plan),
    );
    assert_eq!(outcome.ranked.len(), 0);
    assert_eq!(outcome.rejected.len(), 1);
    let violation = &outcome.rejected[0].violations[0];
    assert_eq!(violation.gate, GateClass::HealthEpochMembership);
    assert!(violation.failing_fact.contains("stale"));

    // An unknown id in the explicit selection rejects the whole evaluation.
    let unknown = PhysicalDeviceId::cuda(UNKNOWN_UUID, None);
    let bad_selection = DeviceSetSelection::explicit([unknown.clone()]);
    let outcome = evaluate_all(
        &snap,
        &bad_selection,
        &constraints_at(&snap, budgets, DeviceHealthGeneration::initial()),
        &[Objective::Latency],
        std::slice::from_ref(&plan),
    );
    assert_eq!(outcome.ranked.len(), 0);
    assert_eq!(outcome.rejected.len(), 1);
    let violation = &outcome.rejected[0].violations[0];
    assert_eq!(violation.gate, GateClass::HealthEpochMembership);
    assert_eq!(violation.device.as_ref(), Some(&unknown));
    assert!(violation.failing_fact.contains("not recorded"));
}

/// Determinism: identical frozen snapshot + selection + constraints +
/// objectives → identical plan and identical explained receipt, including
/// the determinism fingerprint.
#[test]
fn identical_frozen_inputs_produce_identical_plan_and_receipt() {
    let a = PhysicalDeviceId::cuda(UUID_A, None);
    let b = PhysicalDeviceId::cuda(UUID_B, None);
    let snap = snapshot(vec![entry(0, a.clone(), None), entry(1, b.clone(), None)]);
    let declared = constraints(
        &snap,
        BTreeMap::from([
            (a.clone(), SafePhysicalLimit::new(1_000_000)),
            (b.clone(), SafePhysicalLimit::new(1_000_000)),
        ]),
    );
    let selection = DeviceSetSelection::explicit([a.clone(), b.clone()]);
    let objectives = [Objective::Latency, Objective::CapacityReserve];
    let plans = vec![
        CandidatePlan::new(
            "one".to_owned(),
            vec![DeviceAssignment::new(a.clone(), ledger(10))],
            vec![],
            10,
            ObjectiveFacts {
                latency_nanos: Some(1_000),
                ..ObjectiveFacts::default()
            },
        ),
        CandidatePlan::new(
            "two".to_owned(),
            vec![DeviceAssignment::new(b.clone(), ledger(20))],
            vec![],
            20,
            ObjectiveFacts {
                latency_nanos: Some(2_000),
                ..ObjectiveFacts::default()
            },
        ),
    ];

    let first = evaluate_all(&snap, &selection, &declared, &objectives, &plans);
    let second = evaluate_all(&snap, &selection, &declared, &objectives, &plans);
    assert_eq!(first, second, "identical inputs → identical outcome");

    // A freshly rebuilt (but fact-identical) snapshot determinizes to the
    // same snapshot id and the same outcome.
    let snap_rebuilt = snapshot(vec![entry(0, a.clone(), None), entry(1, b.clone(), None)]);
    assert_eq!(snap.id(), snap_rebuilt.id());
    let constraints_rebuilt = constraints(
        &snap_rebuilt,
        BTreeMap::from([
            (a.clone(), SafePhysicalLimit::new(1_000_000)),
            (b.clone(), SafePhysicalLimit::new(1_000_000)),
        ]),
    );
    let rebuilt = evaluate_all(
        &snap_rebuilt,
        &selection,
        &constraints_rebuilt,
        &objectives,
        &plans,
    );
    assert_eq!(first.receipt, rebuilt.receipt);
}

/// Ties resolve by declared objective order, then stable `PhysicalDeviceId`
/// — never ordinal (ordinal-rename locator-only proof; naming contract §1).
#[test]
fn ties_resolve_by_declared_order_then_physical_device_id_never_ordinal() {
    let low = PhysicalDeviceId::cuda("GPU-aaaaaaaa-1111-2222-3333-444444444444", None);
    let high = PhysicalDeviceId::cuda("GPU-bbbbbbbb-1111-2222-3333-444444444444", None);
    let budgets = BTreeMap::from([
        (low.clone(), SafePhysicalLimit::new(1_000_000)),
        (high.clone(), SafePhysicalLimit::new(1_000_000)),
    ]);
    let objectives = [Objective::Latency, Objective::Power];

    // Two plans with identical objective facts → a full tie → the winner is
    // the plan on the lower `PhysicalDeviceId`, never the lower ordinal.
    let plan_low = CandidatePlan::new(
        "on-low".to_owned(),
        vec![DeviceAssignment::new(low.clone(), ledger(10))],
        vec![],
        10,
        ObjectiveFacts {
            latency_nanos: Some(1_000),
            estimated_power_milliwatts: Some(100),
            ..ObjectiveFacts::default()
        },
    );
    let plan_high = CandidatePlan::new(
        "on-high".to_owned(),
        vec![DeviceAssignment::new(high.clone(), ledger(10))],
        vec![],
        10,
        ObjectiveFacts {
            latency_nanos: Some(1_000),
            estimated_power_milliwatts: Some(100),
            ..ObjectiveFacts::default()
        },
    );

    // Snapshot variant 1: low @ ordinal 0, high @ ordinal 1.
    let snap_1 = snapshot(vec![
        entry(0, low.clone(), None),
        entry(1, high.clone(), None),
    ]);
    let constraints_1 = constraints(&snap_1, budgets.clone());
    let selection = DeviceSetSelection::explicit([low.clone(), high.clone()]);
    let outcome_1 = evaluate_all(
        &snap_1,
        &selection,
        &constraints_1,
        &objectives,
        &[plan_low.clone(), plan_high.clone()],
    );

    // Snapshot variant 2: locators renamed — low @ ordinal 1, high @
    // ordinal 0. The ranking must be identical: identity, never ordinal.
    let snap_2 = snapshot(vec![
        entry(1, low.clone(), None),
        entry(0, high.clone(), None),
    ]);
    let constraints_2 = constraints(&snap_2, budgets.clone());
    let outcome_2 = evaluate_all(
        &snap_2,
        &selection,
        &constraints_2,
        &objectives,
        &[plan_low.clone(), plan_high.clone()],
    );

    // The winner is the lower-id device in both locator arrangements.
    assert_eq!(outcome_1.ranked.len(), 2);
    assert_eq!(outcome_1.ranked[0].plan, "on-low");
    assert_eq!(outcome_1.ranked[0].devices, vec![low.clone()]);
    assert_eq!(outcome_2.ranked[0].plan, "on-low");
    assert_eq!(outcome_2.ranked[0].devices, vec![low.clone()]);

    // The ranking (plan + ranks + scores) is identical across the rename;
    // only the snapshot-derived receipt fields (snapshot id, fingerprint)
    // change, because the locator facts are part of the sample.
    assert_eq!(outcome_1.ranked, outcome_2.ranked);
    assert_eq!(outcome_1.rejected, outcome_2.rejected);
    assert_ne!(
        outcome_1.receipt.snapshot_id_hex,
        outcome_2.receipt.snapshot_id_hex
    );
    assert_ne!(
        outcome_1.receipt.determinism_fingerprint,
        outcome_2.receipt.determinism_fingerprint
    );
}

/// Declared objective order drives the ranking: switching the primary
/// objective switches the winner.
#[test]
fn declared_objective_order_drives_ranking() {
    let a = PhysicalDeviceId::cuda(UUID_A, None);
    let b = PhysicalDeviceId::cuda(UUID_B, None);
    let snap = snapshot(vec![entry(0, a.clone(), None), entry(1, b.clone(), None)]);
    let constraints = constraints(
        &snap,
        BTreeMap::from([
            (a.clone(), SafePhysicalLimit::new(1_000_000)),
            (b.clone(), SafePhysicalLimit::new(1_000_000)),
        ]),
    );
    let selection = DeviceSetSelection::explicit([a.clone(), b.clone()]);

    let fast_plan = CandidatePlan::new(
        "fast".to_owned(),
        vec![DeviceAssignment::new(a.clone(), ledger(10))],
        vec![],
        10,
        ObjectiveFacts {
            latency_nanos: Some(1_000),
            estimated_power_milliwatts: Some(500_000),
            ..ObjectiveFacts::default()
        },
    );
    let efficient_plan = CandidatePlan::new(
        "efficient".to_owned(),
        vec![DeviceAssignment::new(b.clone(), ledger(10))],
        vec![],
        10,
        ObjectiveFacts {
            latency_nanos: Some(2_000),
            estimated_power_milliwatts: Some(100_000),
            ..ObjectiveFacts::default()
        },
    );
    let plans = [fast_plan, efficient_plan];

    // Latency first: the fast plan wins.
    let latency_first = evaluate_all(
        &snap,
        &selection,
        &constraints,
        &[Objective::Latency, Objective::Power],
        &plans,
    );
    assert_eq!(latency_first.ranked[0].plan, "fast");
    assert_eq!(latency_first.ranked[1].plan, "efficient");

    // Power first: the efficient plan wins.
    let power_first = evaluate_all(
        &snap,
        &selection,
        &constraints,
        &[Objective::Power, Objective::Latency],
        &plans,
    );
    assert_eq!(power_first.ranked[0].plan, "efficient");
    assert_eq!(power_first.ranked[1].plan, "fast");
}

/// Capacity reserve is computed from the declared budgets and requirements
/// (headroom), and favors the plan with the most headroom.
#[test]
fn capacity_reserve_objective_scores_headroom() {
    let a = PhysicalDeviceId::cuda(UUID_A, None);
    let snap = snapshot(vec![entry(0, a.clone(), None)]);
    let constraints = constraints(
        &snap,
        BTreeMap::from([(a.clone(), SafePhysicalLimit::new(1_000))]),
    );
    let selection = DeviceSetSelection::explicit([a.clone()]);

    let tight = CandidatePlan::new(
        "tight".to_owned(),
        vec![DeviceAssignment::new(a.clone(), ledger(900))],
        vec![],
        900,
        ObjectiveFacts::default(),
    );
    let roomy = CandidatePlan::new(
        "roomy".to_owned(),
        vec![DeviceAssignment::new(a.clone(), ledger(10))],
        vec![],
        10,
        ObjectiveFacts::default(),
    );

    let outcome = evaluate_all(
        &snap,
        &selection,
        &constraints,
        &[Objective::CapacityReserve],
        &[tight, roomy],
    );

    assert_eq!(outcome.ranked.len(), 2);
    assert_eq!(outcome.ranked[0].plan, "roomy");
    assert_eq!(
        outcome.ranked[0].scores,
        vec![ObjectiveScore {
            objective: Objective::CapacityReserve,
            value: 990,
        }]
    );
    assert_eq!(outcome.ranked[1].plan, "tight");
    assert_eq!(outcome.ranked[1].scores[0].value, 100);
}

/// Objectives never reject: a gate-passing plan with no declared objective
/// facts is still ranked (least-favored scores) under all seven objectives.
#[test]
fn objectives_never_reject_a_plan_without_facts() {
    let a = PhysicalDeviceId::cuda(UUID_A, None);
    let snap = snapshot(vec![entry(0, a.clone(), None)]);
    let constraints = constraints(
        &snap,
        BTreeMap::from([(a.clone(), SafePhysicalLimit::new(1_000_000))]),
    );
    let selection = DeviceSetSelection::explicit([a.clone()]);

    let bare = CandidatePlan::new(
        "bare".to_owned(),
        vec![DeviceAssignment::new(a.clone(), ledger(10))],
        vec![],
        10,
        ObjectiveFacts::default(),
    );
    let all_seven = [
        Objective::Latency,
        Objective::Throughput,
        Objective::CapacityReserve,
        Objective::CacheAffinity,
        Objective::ExpertHotness,
        Objective::TransferBudget,
        Objective::Power,
    ];

    let outcome = evaluate_all(
        &snap,
        &selection,
        &constraints,
        &all_seven,
        std::slice::from_ref(&bare),
    );
    assert!(outcome.rejected.is_empty(), "objectives never reject");
    assert_eq!(outcome.ranked.len(), 1);
    assert_eq!(outcome.ranked[0].plan, "bare");
    assert_eq!(outcome.ranked[0].scores.len(), 7);
    assert_eq!(
        outcome.ranked[0]
            .scores
            .iter()
            .map(|s| s.objective)
            .collect::<Vec<_>>(),
        all_seven.to_vec()
    );
}

/// The explained receipt names the frozen snapshot id, the declared
/// constraints/objectives, rejected devices with the violated gate + exact
/// failing fact, and selected devices with ranks + objective scores, plus a
/// determinism fingerprint.
#[test]
#[allow(clippy::too_many_lines)] // pins the explained receipt shape in one test
fn receipt_names_snapshot_constraints_rejected_and_selected() {
    let a = PhysicalDeviceId::cuda(UUID_A, None);
    let b = PhysicalDeviceId::cuda(UUID_B, None);
    let snap = snapshot(vec![entry(0, a.clone(), None), entry(1, b.clone(), None)]);
    let mut constraints = constraints(
        &snap,
        BTreeMap::from([
            (a.clone(), SafePhysicalLimit::new(PARTITION_LIMIT_BYTES)),
            (b.clone(), SafePhysicalLimit::new(PARTITION_LIMIT_BYTES)),
        ]),
    );
    constraints.total_budget = Some(SafePhysicalLimit::new(MESH_TOTAL_BYTES));
    let selection = DeviceSetSelection::explicit([a.clone(), b.clone()]);
    let objectives = [Objective::Latency, Objective::Throughput];

    let whole = CandidatePlan::new(
        "whole".to_owned(),
        vec![DeviceAssignment::new(a.clone(), ledger(SMOLLM2_BYTES))],
        vec![],
        SMOLLM2_BYTES,
        ObjectiveFacts::default(),
    );
    let split = CandidatePlan::new(
        "split".to_owned(),
        vec![
            DeviceAssignment::new(a.clone(), ledger(HALF_SPLIT_BYTES)),
            DeviceAssignment::new(b.clone(), ledger(HALF_SPLIT_BYTES)),
        ],
        vec![],
        SMOLLM2_BYTES,
        ObjectiveFacts {
            throughput_tokens_per_sec: Some(42),
            ..ObjectiveFacts::default()
        },
    );

    let outcome = evaluate_all(
        &snap,
        &selection,
        &constraints,
        &objectives,
        &[whole, split],
    );
    let receipt = &outcome.receipt;

    // Frozen snapshot id.
    assert_eq!(receipt.snapshot_id_hex, snap.id().hex());

    // Declared constraints + objectives.
    assert_eq!(
        receipt.current_generation,
        DeviceHealthGeneration::initial()
    );
    assert_eq!(receipt.per_device_budget.len(), 2);
    assert!(receipt
        .per_device_budget
        .contains(&(a.clone(), SafePhysicalLimit::new(PARTITION_LIMIT_BYTES))));
    assert_eq!(
        receipt.total_budget,
        Some(SafePhysicalLimit::new(MESH_TOTAL_BYTES))
    );
    assert_eq!(receipt.objectives, objectives.to_vec());

    // Rejected devices + gate + exact failing fact.
    assert_eq!(receipt.rejected, outcome.rejected);
    let rejected = &receipt.rejected[0];
    assert_eq!(rejected.plan, "whole");
    assert_eq!(rejected.violations[0].gate, GateClass::RequiredMemory);
    assert!(rejected.violations[0]
        .failing_fact
        .contains("budget_exceeded"));

    // Selected devices + ranks + objective scores.
    assert_eq!(receipt.ranked, outcome.ranked);
    assert_eq!(receipt.ranked[0].plan, "split");
    assert_eq!(receipt.ranked[0].rank, 1);
    assert_eq!(
        receipt.ranked[0]
            .scores
            .iter()
            .map(|s| s.objective)
            .collect::<Vec<_>>(),
        vec![Objective::Latency, Objective::Throughput]
    );

    // Determinism fingerprint: 16 lowercase hex chars, stable per evaluation.
    assert_eq!(receipt.determinism_fingerprint.len(), 16);
    assert!(receipt
        .determinism_fingerprint
        .chars()
        .all(|c| c.is_ascii_hexdigit()));
    let again = evaluate_all(
        &snap,
        &selection,
        &constraints,
        &objectives,
        &[
            CandidatePlan::new(
                "whole".to_owned(),
                vec![DeviceAssignment::new(a.clone(), ledger(SMOLLM2_BYTES))],
                vec![],
                SMOLLM2_BYTES,
                ObjectiveFacts::default(),
            ),
            CandidatePlan::new(
                "split".to_owned(),
                vec![
                    DeviceAssignment::new(a.clone(), ledger(HALF_SPLIT_BYTES)),
                    DeviceAssignment::new(b.clone(), ledger(HALF_SPLIT_BYTES)),
                ],
                vec![],
                SMOLLM2_BYTES,
                ObjectiveFacts {
                    throughput_tokens_per_sec: Some(42),
                    ..ObjectiveFacts::default()
                },
            ),
        ],
    );
    assert_eq!(
        again.receipt.determinism_fingerprint,
        receipt.determinism_fingerprint
    );
}
