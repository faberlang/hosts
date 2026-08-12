//! S3-U5 (CG2-M07): differential identity tests for the host program-graph
//! receipt [`DeviceDescriptor::program_graph_hash`].
//!
//! The receipt is a SHA-256 digest over the domain-tagged canonical byte
//! stream of the descriptor's EXECUTION-affecting program-graph facts. These
//! tests are pure byte-stream differentials (no hardware oracle): they prove
//! field sensitivity — every execution-affecting field's inclusion flips the
//! hash — and declaration-order invariance — reordering kernel or buffer slot
//! DECLARATIONS does not change the receipt.
//!
//! The audit's omitted-field set (binding, lifetime, initialization,
//! program_lifetime) is the focus; every already-hashed field is re-flipped
//! as a control.

use host_coordinator::DeviceBackend;
use faber_host_macos_arm64::device_descriptor::{
    DescriptorBuffer, DescriptorBufferVersion, DescriptorDataFlow, DescriptorEndOfRunResult,
    DescriptorKernel, DescriptorLaunch, DescriptorResult, DeviceBufferInitialization,
    DeviceBufferLifetime, DeviceBufferRole, DeviceDataType, DeviceDescriptor,
    DeviceProgramLifetime,
};

/// The S2-4 lifetime the constructor derives from ABI facts (Input →
/// PerProgram, Output → ObservationPoint, InOut → PerStep).
fn lifetime_for_role(role: DeviceBufferRole) -> DeviceBufferLifetime {
    match role {
        DeviceBufferRole::Input => DeviceBufferLifetime::PerProgram,
        DeviceBufferRole::Output => DeviceBufferLifetime::ObservationPoint,
        DeviceBufferRole::InOut => DeviceBufferLifetime::PerStep,
    }
}

/// The F5 initialization axis the constructor projects from the role.
fn initialization_for_role(role: DeviceBufferRole) -> DeviceBufferInitialization {
    match role {
        DeviceBufferRole::Input => DeviceBufferInitialization::HostProvided,
        DeviceBufferRole::InOut => DeviceBufferInitialization::ZeroFill,
        DeviceBufferRole::Output => DeviceBufferInitialization::KernelInitialized,
    }
}

/// One typed buffer slot (F1: one distinct semantic value per buffer id).
fn slot(
    id: u32,
    name: &str,
    role: DeviceBufferRole,
    binding: u32,
    count: u64,
) -> DescriptorBuffer {
    DescriptorBuffer {
        buffer_id: id,
        buffer_name: name.to_owned(),
        semantic_value: id,
        role,
        lifetime: lifetime_for_role(role),
        initialization: initialization_for_role(role),
        binding,
        element_ty: DeviceDataType::F32,
        element_count: count,
        version: 1,
    }
}

/// The version-keyed metadata mirroring the slots (shape facts are carried by
/// the wire; the hash consumes the validated slot facts).
fn buffer_versions_for(kernels: &[DescriptorKernel]) -> Vec<DescriptorBufferVersion> {
    let mut versions = Vec::new();
    for kernel in kernels {
        for slot in &kernel.buffers {
            if versions
                .iter()
                .any(|version: &DescriptorBufferVersion| {
                    version.buffer_id == slot.buffer_id && version.version == slot.version
                })
            {
                continue;
            }
            versions.push(DescriptorBufferVersion {
                buffer_id: slot.buffer_id,
                version: slot.version,
                element_ty: slot.element_ty,
                element_count: slot.element_count,
            });
        }
    }
    versions
}

/// The differential base: two kernels with distinct entries (so kernel
/// declaration reordering is visible), an ordered two-launch schedule with a
/// carried data-flow edge, one per-step observation point, one end-of-run
/// observation, and the single-run program regime. Buffer 3 (`acc`) is the
/// InOut accumulation buffer flowing launch 1 → launch 2.
fn base_descriptor() -> DeviceDescriptor {
    let kernels = vec![
        DescriptorKernel {
            entry: "kernel_zero".to_owned(),
            buffers: vec![
                slot(1, "a", DeviceBufferRole::Input, 0, 2),
                slot(2, "b", DeviceBufferRole::Input, 1, 2),
                slot(3, "acc", DeviceBufferRole::InOut, 2, 2),
            ],
            grid: [1, 1, 1],
            block: [2, 1, 1],
        },
        DescriptorKernel {
            entry: "kernel_one".to_owned(),
            buffers: vec![
                slot(3, "acc", DeviceBufferRole::InOut, 0, 2),
                slot(4, "out", DeviceBufferRole::Output, 1, 2),
            ],
            grid: [1, 1, 1],
            block: [2, 1, 1],
        },
    ];
    let versions = buffer_versions_for(&kernels);
    DeviceDescriptor {
        backend: DeviceBackend::Metal,
        module_image: b"// fake compiler-owned module image".to_vec(),
        kernels,
        launches: vec![
            DescriptorLaunch {
                id: 1,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 2,
                kernel_index: 1,
            },
        ],
        buffer_versions: versions,
        program_lifetime: DeviceProgramLifetime::SingleRun,
        data_flow: vec![DescriptorDataFlow {
            buffer_id: 3,
            version: 1,
            producer: 1,
            consumer: 2,
        }],
        roots: vec![1, 2],
        results: vec![DescriptorResult {
            buffer_id: 4,
            version: 1,
            produced_by: 2,
            at_launch: 2,
        }],
        end_of_run_results: vec![DescriptorEndOfRunResult {
            buffer_id: 3,
            version: 1,
        }],
    }
}

/// Apply one field mutation to a fresh clone of the base descriptor and
/// assert the receipt CHANGES: the mutated field must be in the hash byte
/// stream.
fn assert_field_flips_hash(mutate: impl FnOnce(&mut DeviceDescriptor)) {
    let mut descriptor = base_descriptor();
    let baseline = descriptor.program_graph_hash();
    mutate(&mut descriptor);
    let flipped = descriptor.program_graph_hash();
    assert_ne!(
        baseline, flipped,
        "the program-graph hash must change when an execution-affecting \
         field changes (the field must be in the canonical byte stream)"
    );
}

// ---------------------------------------------------------------------------
// S3-U5 field sensitivity: the audit's omitted set (M07).
// ---------------------------------------------------------------------------

/// `.binding` is a per-slot ABI fact (unique per kernel; the same buffer id
/// may bind different indices in different kernels) — it joins the per-launch
/// inlined slot facts.
#[test]
fn identity_field_binding_changes_the_hash() {
    assert_field_flips_hash(|d| {
        d.kernels[0].buffers[0].binding = 7;
    });
}

/// `.lifetime` is a buffer identity fact driving the session's per-class
/// allocation/release policy — it joins the sorted canonical stream (and the
/// inlined slot facts).
#[test]
fn identity_field_lifetime_changes_the_hash() {
    assert_field_flips_hash(|d| {
        d.kernels[0].buffers[0].lifetime = DeviceBufferLifetime::ObservationPoint;
    });
}

/// `.initialization` is the independent F5 allocation axis — it joins the
/// sorted canonical stream (and the inlined slot facts).
#[test]
fn identity_field_initialization_changes_the_hash() {
    assert_field_flips_hash(|d| {
        d.kernels[0].buffers[0].initialization = DeviceBufferInitialization::KernelInitialized;
    });
}

/// `.program_lifetime` is the program execution regime (single-run vs
/// repeating training step) — it joins the descriptor-level stream.
#[test]
fn identity_field_program_lifetime_changes_the_hash() {
    assert_field_flips_hash(|d| {
        d.program_lifetime = DeviceProgramLifetime::RepeatingStep;
    });
}

// ---------------------------------------------------------------------------
// Control: every already-hashed execution-affecting field still flips the
// hash after the S3-U5 changes.
// ---------------------------------------------------------------------------

#[test]
fn identity_control_fields_each_change_the_hash() {
    // DescriptorBuffer: buffer_id, semantic_value, version, element_ty,
    // element_count (sorted stream + inlined facts), role (inlined facts).
    assert_field_flips_hash(|d| {
        d.kernels[0].buffers[0].buffer_id = 11;
    });
    assert_field_flips_hash(|d| {
        d.kernels[0].buffers[0].semantic_value = 11;
    });
    assert_field_flips_hash(|d| {
        d.kernels[0].buffers[0].version = 2;
    });
    assert_field_flips_hash(|d| {
        d.kernels[0].buffers[0].element_ty = DeviceDataType::F64;
    });
    assert_field_flips_hash(|d| {
        d.kernels[0].buffers[0].element_count = 3;
    });
    assert_field_flips_hash(|d| {
        d.kernels[0].buffers[0].role = DeviceBufferRole::InOut;
    });
    // DescriptorKernel: entry, grid, block (inlined per launch).
    assert_field_flips_hash(|d| {
        d.kernels[0].entry = "kernel_zero_alt".to_owned();
    });
    assert_field_flips_hash(|d| {
        d.kernels[0].grid = [2, 1, 1];
    });
    assert_field_flips_hash(|d| {
        d.kernels[0].block = [3, 1, 1];
    });
    // DescriptorLaunch: id, and kernel_index (covered via the inlined kernel
    // facts — the launch names a different kernel's facts).
    assert_field_flips_hash(|d| {
        d.launches[0].id = 11;
    });
    assert_field_flips_hash(|d| {
        d.launches[0].kernel_index = 1;
    });
    // DeviceDescriptor: roots, data_flow, results, end_of_run_results.
    assert_field_flips_hash(|d| {
        d.roots = vec![1];
    });
    assert_field_flips_hash(|d| {
        d.data_flow[0].buffer_id = 4;
    });
    assert_field_flips_hash(|d| {
        d.results[0].produced_by = 1;
    });
    assert_field_flips_hash(|d| {
        d.end_of_run_results[0].buffer_id = 4;
    });
}

// ---------------------------------------------------------------------------
// Declaration-order invariance.
// ---------------------------------------------------------------------------

/// Reordering kernel DECLARATIONS (renumbering the launch kernel_index to the
/// same kernels) must not change the receipt: kernel facts are inlined per
/// launch, so declaration position is never an execution authority.
#[test]
fn identity_order_invariance_kernel_declarations() {
    let zero = base_descriptor().kernels[0].clone();
    let one = base_descriptor().kernels[1].clone();
    let declared_zero_first = descriptor_with_kernels(zero.clone(), one.clone(), 0, 1);
    let declared_one_first = descriptor_with_kernels(one.clone(), zero.clone(), 1, 0);
    assert_eq!(
        declared_zero_first.program_graph_hash(),
        declared_one_first.program_graph_hash()
    );
}

/// Reordering buffer slot DECLARATIONS within a kernel must not change the
/// receipt: the per-launch inlined slot facts are sorted by the total order
/// (buffer_id, version, binding).
#[test]
fn identity_order_invariance_buffer_declarations() {
    let declared = base_descriptor();
    let mut reversed = base_descriptor();
    reversed.kernels[0].buffers.reverse();
    assert_eq!(
        declared.program_graph_hash(),
        reversed.program_graph_hash()
    );}

/// The launch SEQUENCE is the execution schedule and IS part of the graph
/// identity: reordering launches changes the receipt. Declaration-order
/// invariance applies to declarations (kernel/buffer slot order) — the
/// ordered schedule is hashed deliberately.
#[test]
fn identity_launch_sequence_is_part_of_the_identity() {
    assert_field_flips_hash(|d| {
        d.launches.swap(0, 1);
    });
}

/// `buffer_name` is diagnostic-only ("Logical name for diagnostics") and must
/// NOT be part of the identity — the census excludes it.
#[test]
fn identity_buffer_name_is_diagnostic_only() {
    let mut descriptor = base_descriptor();
    let baseline = descriptor.program_graph_hash();
    descriptor.kernels[0].buffers[0].buffer_name = "renamed".to_owned();
    assert_eq!(
        baseline,
        descriptor.program_graph_hash(),
        "the diagnostic buffer name must not change the program-graph receipt"
    );
}

/// Build a descriptor launching `zero` as launch 1 and `one` as launch 2
/// (the same graph, whatever the declaration order of the kernel vec).
fn descriptor_with_kernels(
    zero: DescriptorKernel,
    one: DescriptorKernel,
    zero_index: usize,
    one_index: usize,
) -> DeviceDescriptor {
    let mut descriptor = base_descriptor();
    descriptor.kernels = vec![zero, one];
    descriptor.launches = vec![
        DescriptorLaunch {
            id: 1,
            kernel_index: zero_index as u32,
        },
        DescriptorLaunch {
            id: 2,
            kernel_index: one_index as u32,
        },
    ];
    descriptor
}
