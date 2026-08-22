//! M8-U1b: MoE family dispatch through the resident schedule (Metal).
//!
//! The R-PACK-03 probe goldens (weights/intermediate/accumulate rows and
//! deterministic ties) run green through the compiled-plan dispatch seam,
//! and the placement ruling is proven device-side: one synthetic MoE-layer
//! decode step executes `router_selection` + `grouped_expert_gemm` on the
//! real Metal device with zero per-step host readback of the router /
//! grouped-dispatch buffers (the ids/weights stay device-resident; the only
//! readback is the declared output observation).

use std::collections::BTreeMap;

use faber_host_macos_arm64::composite_host::{
    CompositeHost, DeviceByteBuffer, PreparedResidentSession,
};
use faber_host_macos_arm64::device_descriptor::{
    DescriptorBuffer, DescriptorBufferVersion, DescriptorDataFlow, DescriptorKernel,
    DescriptorLaunch, DescriptorResult, DeviceBufferInitialization, DeviceBufferLifetime,
    DeviceBufferRole, DeviceDataType, DeviceDescriptor, DeviceProgramLifetime,
};
use faber_host_macos_arm64::device_host::DeviceRuntime;
use faber_host_macos_arm64::kernel::library::{
    dispatch_grouped_expert_gemm_selected, dispatch_selected, GroupedExpertGemmBind,
    GroupedExpertGemmKernel, KernelBodyError, LibraryDispatch, QuantizedFormat,
};
use faber_host_macos_arm64::kernel::moe::{
    dispatch_router_selection, moe_family_msl, MoeFamilyMslFacts, RouterSelectionBind,
    RouterSelectionKernel,
};
use faber_host_macos_arm64::MetalHostSession;
use host_coordinator::DeviceBackend;

// ---------------------------------------------------------------------------
// R-PACK-03 GQA-shape analog fixtures (the PM1 probe geometry).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct GroupedExpertAnalog {
    name: &'static str,
    rows: usize,
    columns: usize,
}

const GROUPED_EXPERT_ANALOGS: [GroupedExpertAnalog; 2] = [
    GroupedExpertAnalog {
        name: "smollm2-360m-gqa",
        rows: 3,
        columns: 5,
    },
    GroupedExpertAnalog {
        name: "qwen2.5-0.5b-gqa",
        rows: 7,
        columns: 2,
    },
];

const GROUPED_EXPERT_K: usize = 32;
const GROUPED_EXPERT_COUNT: usize = 2;

fn q8_0_block(value: i8) -> Vec<u8> {
    let mut block = vec![0x00, 0x3c]; // f16 scale = 1.0
    block.extend(std::iter::repeat_n(value as u8, GROUPED_EXPERT_K));
    block
}

fn grouped_expert_activation(fixture: GroupedExpertAnalog) -> Vec<f32> {
    (0..fixture.rows * GROUPED_EXPERT_K)
        .map(|index| {
            let row = index / GROUPED_EXPERT_K;
            let element = index % GROUPED_EXPERT_K;
            0.5 + row as f32 * 0.25 + element as f32 * 0.01
        })
        .collect()
}

fn grouped_expert_packed_weights(fixture: GroupedExpertAnalog, tied: bool) -> Vec<u8> {
    (0..GROUPED_EXPERT_COUNT)
        .flat_map(|expert| {
            (0..fixture.columns).flat_map(move |column| {
                let value = if tied {
                    column + 1
                } else {
                    expert * 2 + column + 1
                };
                q8_0_block(value as i8)
            })
        })
        .collect()
}

fn grouped_expert_reference(fixture: GroupedExpertAnalog, tied: bool) -> Vec<f32> {
    let activation = grouped_expert_activation(fixture);
    let mut accumulated = vec![0.0f32; fixture.rows * fixture.columns];
    for expert in 0..GROUPED_EXPERT_COUNT {
        for row in 0..fixture.rows {
            for column in 0..fixture.columns {
                let value = if tied {
                    column + 1
                } else {
                    expert * 2 + column + 1
                };
                let weight = value as f32;
                accumulated[row * fixture.columns + column] += (0..GROUPED_EXPERT_K)
                    .map(|element| activation[row * GROUPED_EXPERT_K + element] * weight)
                    .sum::<f32>();
            }
        }
    }
    accumulated
}

/// One expert's intermediate row: `out[row][col] = sum_k act * W_e[k][col]`.
fn grouped_expert_intermediate(fixture: GroupedExpertAnalog, expert: usize) -> Vec<f32> {
    let activation = grouped_expert_activation(fixture);
    let mut intermediate = vec![0.0f32; fixture.rows * fixture.columns];
    for row in 0..fixture.rows {
        for column in 0..fixture.columns {
            let weight = (expert * 2 + column + 1) as f32;
            intermediate[row * fixture.columns + column] = (0..GROUPED_EXPERT_K)
                .map(|element| activation[row * GROUPED_EXPERT_K + element] * weight)
                .sum();
        }
    }
    intermediate
}

/// The router-weight row: each expert's intermediate scaled before
/// accumulation.
fn grouped_expert_weighted_reference(
    fixture: GroupedExpertAnalog,
    expert_weights: &[f32],
) -> Vec<f32> {
    let activation = grouped_expert_activation(fixture);
    let mut output = vec![0.0f32; fixture.rows * fixture.columns];
    for row in 0..fixture.rows {
        for column in 0..fixture.columns {
            let mut value = 0.0f32;
            for (expert, weight) in expert_weights.iter().enumerate() {
                let column_weight = (expert * 2 + column + 1) as f32;
                let dot = (0..GROUPED_EXPERT_K)
                    .map(|element| activation[row * GROUPED_EXPERT_K + element] * column_weight)
                    .sum::<f32>();
                value += weight * dot;
            }
            output[row * fixture.columns + column] = value;
        }
    }
    output
}

fn grouped_expert_bind(fixture: GroupedExpertAnalog, experts: usize) -> GroupedExpertGemmBind {
    GroupedExpertGemmBind::contiguous(
        fixture.rows as u64,
        GROUPED_EXPERT_K as u64,
        fixture.columns as u64,
        experts as u64,
        QuantizedFormat::Q8_0,
        [fixture.columns as u32, fixture.rows as u32, 1],
    )
}

// ---------------------------------------------------------------------------
// Focused family numeric tests: probe goldens through the plan dispatch.
// ---------------------------------------------------------------------------

#[test]
fn grouped_probe_goldens_green_through_plan_dispatch() {
    for fixture in GROUPED_EXPERT_ANALOGS {
        let bind = grouped_expert_bind(fixture, GROUPED_EXPERT_COUNT);
        let activation = grouped_expert_activation(fixture);
        let packed = grouped_expert_packed_weights(fixture, false);
        let expected = grouped_expert_reference(fixture, false);

        // The ids/weights buffers are per-row: `[rows * active]` elements.
        let per_row_ids = |slot_values: &[u32]| -> Vec<u32> {
            std::iter::repeat_n(slot_values, fixture.rows)
                .flatten()
                .copied()
                .collect()
        };
        let per_row_weights = |slot_values: &[f32]| -> Vec<f32> {
            std::iter::repeat_n(slot_values, fixture.rows)
                .flatten()
                .copied()
                .collect()
        };

        // Accumulate row: both experts active with unit router weights,
        // selected through the plan dispatch seam.
        let ids = per_row_ids(&[0, 1]);
        let unit_weights = per_row_weights(&[1.0, 1.0]);
        let mut accumulated = vec![0.0f32; fixture.rows * fixture.columns];
        dispatch_selected(LibraryDispatch::GroupedExpertGemm {
            library_entry: Some("GroupedExpertGemm"),
            bind: &bind,
            activation: &activation,
            expert_ids: &ids,
            expert_weights: &unit_weights,
            packed_weights: &packed,
            output: &mut accumulated,
        })
        .unwrap_or_else(|error| panic!("{} plan-path grouped body: {error}", fixture.name));
        assert_eq!(
            accumulated, expected,
            "{} accumulated output through the plan dispatch",
            fixture.name
        );

        // Intermediate rows: each expert independently through the plan
        // dispatch proves the packed weights and one expert's row. The
        // sliced packed region is one expert wide, so its relative id is 0.
        let one_expert_bind = grouped_expert_bind(fixture, 1);
        let expert_stride = bind.packed_expert_stride_bytes as usize;
        for expert in 0..GROUPED_EXPERT_COUNT {
            let start = expert * expert_stride;
            let mut intermediate = vec![0.0f32; fixture.rows * fixture.columns];
            dispatch_selected(LibraryDispatch::GroupedExpertGemm {
                library_entry: Some("GroupedExpertGemm"),
                bind: &one_expert_bind,
                activation: &activation,
                expert_ids: &per_row_ids(&[0]),
                expert_weights: &per_row_weights(&[1.0]),
                packed_weights: &packed[start..start + expert_stride],
                output: &mut intermediate,
            })
            .unwrap_or_else(|error| panic!("{} expert {expert}: {error}", fixture.name));
            assert_eq!(
                intermediate,
                grouped_expert_intermediate(fixture, expert),
                "{} expert {expert} intermediate through the plan dispatch",
                fixture.name
            );
        }

        // Router-weight row: non-unit weights scale each expert before
        // accumulation.
        let router_weights = per_row_weights(&[0.6, 0.4]);
        let mut weighted = vec![0.0f32; fixture.rows * fixture.columns];
        dispatch_selected(LibraryDispatch::GroupedExpertGemm {
            library_entry: Some("GroupedExpertGemm"),
            bind: &bind,
            activation: &activation,
            expert_ids: &ids,
            expert_weights: &router_weights,
            packed_weights: &packed,
            output: &mut weighted,
        })
        .expect("weighted grouped body");
        assert_eq!(
            weighted,
            grouped_expert_weighted_reference(fixture, &[0.6, 0.4]),
            "{} router-weight row through the plan dispatch",
            fixture.name
        );

        // Deterministic ties: a second dispatch is byte-identical.
        let mut second = vec![0.0f32; fixture.rows * fixture.columns];
        dispatch_selected(LibraryDispatch::GroupedExpertGemm {
            library_entry: Some("GroupedExpertGemm"),
            bind: &bind,
            activation: &activation,
            expert_ids: &ids,
            expert_weights: &unit_weights,
            packed_weights: &packed,
            output: &mut second,
        })
        .expect("tied grouped body");
        assert_eq!(
            accumulated, second,
            "{} expert traversal must be deterministic",
            fixture.name
        );
    }
}

#[test]
fn grouped_plan_dispatch_fails_closed_on_bad_selection() {
    let fixture = GROUPED_EXPERT_ANALOGS[0];
    let bind = grouped_expert_bind(fixture, GROUPED_EXPERT_COUNT);
    let activation = grouped_expert_activation(fixture);
    let packed = grouped_expert_packed_weights(fixture, false);

    let mut output = vec![f32::NAN; fixture.rows * fixture.columns];
    let wrong_entry = dispatch_selected(LibraryDispatch::GroupedExpertGemm {
        library_entry: Some("QkvProjection"),
        bind: &bind,
        activation: &activation,
        expert_ids: &[0, 1, 0, 1, 0, 1],
        expert_weights: &[1.0; 6],
        packed_weights: &packed,
        output: &mut output,
    })
    .expect_err("wrong library entry must fail closed");
    assert!(matches!(
        wrong_entry,
        KernelBodyError::InvalidBind(message) if message.contains("disagrees with library_entry")
    ));

    let out_of_range = dispatch_grouped_expert_gemm_selected(
        GroupedExpertGemmKernel::Device,
        &bind,
        &activation,
        &[0, 2, 0, 2, 0, 2],
        &[1.0; 6],
        &packed,
        &mut output,
    )
    .expect_err("out-of-range expert id must fail closed");
    assert!(matches!(
        out_of_range,
        KernelBodyError::ShapeMismatch(message) if message.contains("out of the packed expert range")
    ));

    let non_finite = dispatch_grouped_expert_gemm_selected(
        GroupedExpertGemmKernel::Device,
        &bind,
        &activation,
        &[0, 1, 0, 1, 0, 1],
        &[1.0, f32::NAN, 1.0, 1.0, 1.0, 1.0],
        &packed,
        &mut output,
    )
    .expect_err("non-finite router weight must fail closed");
    assert!(matches!(
        non_finite,
        KernelBodyError::ShapeMismatch(message) if message.contains("weight is non-finite")
    ));
}

// ---------------------------------------------------------------------------
// Router selection: the device-side policy mirrors the PM3 host seam.
// ---------------------------------------------------------------------------

/// Independent implementation of the PM3 host-seam policy
/// (`radix/crates/faber/src/package/device/router_selection.rs`): descending
/// logit, lower expert id wins an equal-logit tie, softmax over the selected
/// experts after subtracting the selected-row maximum.
fn seam_reference(logits: &[f32], active: usize) -> (Vec<u32>, Vec<f32>) {
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|(left_index, left), (right_index, right)| {
        right
            .partial_cmp(left)
            .expect("finite logits")
            .then_with(|| left_index.cmp(right_index))
    });
    ranked.truncate(active);
    let max_logit = ranked[0].1;
    let mut weights: Vec<f32> = ranked
        .iter()
        .map(|(_, logit)| (*logit - max_logit).exp())
        .collect();
    let normalizer: f32 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= normalizer;
    }
    (
        ranked.iter().map(|(index, _)| *index as u32).collect(),
        weights,
    )
}

struct RouterFixture {
    bind: RouterSelectionBind,
    activation: Vec<f32>,
    packed: Vec<u8>,
}

/// A router fixture whose packed Q8_0 weights are `value` repeated across the
/// activation width for each declared expert (scale 1.0), so expert `e` gets
/// logit `values[e] * sum(activation)`.
fn router_fixture(rows: usize, experts: usize, active: usize, values: &[i8]) -> RouterFixture {
    let bind = RouterSelectionBind::packed(
        rows as u64,
        GROUPED_EXPERT_K as u64,
        experts as u64,
        active as u64,
        QuantizedFormat::Q8_0,
        [rows as u32, 1, 1],
    );
    let packed = values
        .iter()
        .flat_map(|value| q8_0_block(*value))
        .collect::<Vec<_>>();
    let activation = (0..rows * GROUPED_EXPERT_K)
        .map(|index| (index % GROUPED_EXPERT_K) as f32 * 0.01 + 1.0)
        .collect();
    RouterFixture {
        bind,
        activation,
        packed,
    }
}

#[test]
fn router_selection_through_plan_dispatch_matches_seam_policy() {
    // A tied top: experts 2 and 3 share the maximum logit. The lower id must
    // win, and softmax is evaluated only over the selected pair.
    let fixture = router_fixture(2, 4, 2, &[1, 2, 3, 3]);
    let mut ids = vec![0u32; 4];
    let mut weights = vec![0.0f32; 4];
    dispatch_selected(LibraryDispatch::RouterSelection {
        library_entry: Some("RouterSelection"),
        bind: &fixture.bind,
        activation: &fixture.activation,
        router_weight: &fixture.packed,
        expert_ids: &mut ids,
        expert_weights: &mut weights,
    })
    .expect("plan-path router selection");

    let dot = fixture.activation[..GROUPED_EXPERT_K].iter().sum::<f32>();
    let logits = [1.0 * dot, 2.0 * dot, 3.0 * dot, 3.0 * dot];
    let (expected_ids, expected_weights) = seam_reference(&logits, 2);
    let mut max_dev = 0.0f32;
    for row in 0..2 {
        let row_ids = &ids[row * 2..row * 2 + 2];
        assert_eq!(
            row_ids, &expected_ids,
            "row {row} ids must match the seam exactly"
        );
        let row_weights = &weights[row * 2..row * 2 + 2];
        assert_eq!(row_weights, &expected_weights, "row {row} weights");
        for (observed, reference) in row_weights.iter().zip(&expected_weights) {
            max_dev = max_dev.max((observed - reference).abs());
        }
    }
    // max-dev recorded against the MODEL-02-class band (6.44e-8 reference;
    // the f32 seam-parity band here is 1e-6).
    assert!(
        max_dev < 1e-6,
        "max weight deviation {max_dev} exceeds 1e-6"
    );
    println!("m8-u1b router seam parity: max-dev={max_dev} (band 1e-6)");
}

#[test]
fn router_plan_dispatch_fails_closed_on_non_finite_logit_and_bad_entry() {
    let bind = RouterSelectionBind::packed(
        1,
        GROUPED_EXPERT_K as u64,
        2,
        1,
        QuantizedFormat::Q8_0,
        [1, 1, 1],
    );
    let mut packed = q8_0_block(1);
    packed[0] = 0x00;
    packed[1] = 0x7c; // f16 scale = +infinity
    packed.extend(q8_0_block(1));
    let activation = vec![1.0f32; GROUPED_EXPERT_K];
    let mut ids = vec![0u32; 1];
    let mut weights = vec![0.0f32; 1];
    let non_finite = dispatch_selected(LibraryDispatch::RouterSelection {
        library_entry: Some("RouterSelection"),
        bind: &bind,
        activation: &activation,
        router_weight: &packed,
        expert_ids: &mut ids,
        expert_weights: &mut weights,
    })
    .expect_err("non-finite logit must fail closed");
    assert!(matches!(
        non_finite,
        KernelBodyError::NonFiniteLogit { row: 0, expert: 0 }
    ));

    let wrong_entry = dispatch_selected(LibraryDispatch::RouterSelection {
        library_entry: Some("GroupedExpertGemm"),
        bind: &bind,
        activation: &activation,
        router_weight: &packed,
        expert_ids: &mut ids,
        expert_weights: &mut weights,
    })
    .expect_err("wrong library entry must fail closed");
    assert!(matches!(
        wrong_entry,
        KernelBodyError::InvalidBind(message) if message.contains("disagrees with library_entry")
    ));
}

// ---------------------------------------------------------------------------
// One synthetic MoE-layer plan-path run on the resident schedule (real Metal).
// ---------------------------------------------------------------------------

const PLAN_PATH_ROWS: u64 = 1;
const PLAN_PATH_K: u64 = 32;
const PLAN_PATH_N: u64 = 5;
const PLAN_PATH_EXPERTS: u64 = 4;
const PLAN_PATH_ACTIVE: u64 = 2;

fn plan_path_facts() -> MoeFamilyMslFacts {
    MoeFamilyMslFacts {
        rows: PLAN_PATH_ROWS,
        k: PLAN_PATH_K,
        n: PLAN_PATH_N,
        experts: PLAN_PATH_EXPERTS,
        active: PLAN_PATH_ACTIVE,
        format: QuantizedFormat::Q8_0,
    }
}

fn plan_path_activation() -> Vec<f32> {
    (0..PLAN_PATH_K as usize)
        .map(|element| (element % 8) as f32 * 0.01 + 0.25)
        .collect()
}

fn plan_path_router_packed() -> Vec<u8> {
    // Logits = value * sum(activation). Experts 1..3 tie at the top, so the
    // lower-id tie rule must select exactly [1, 2] on the device — the
    // chosen set is observable through the output because each expert owns
    // different packed weights.
    (0..PLAN_PATH_EXPERTS)
        .flat_map(|expert| q8_0_block((expert.min(1) + 1) as i8))
        .collect()
}

fn plan_path_expert_packed() -> Vec<u8> {
    (0..PLAN_PATH_EXPERTS)
        .flat_map(|expert| {
            (0..PLAN_PATH_N).flat_map(move |column| q8_0_block((expert * 10 + column + 1) as i8))
        })
        .collect()
}

fn slot(
    id: u32,
    name: &str,
    binding: u32,
    role: DeviceBufferRole,
    lifetime: DeviceBufferLifetime,
    initialization: DeviceBufferInitialization,
    dtype: DeviceDataType,
    count: u64,
) -> DescriptorBuffer {
    DescriptorBuffer {
        buffer_id: id,
        buffer_name: name.to_owned(),
        semantic_value: id,
        role,
        lifetime,
        initialization,
        binding,
        element_ty: dtype,
        element_count: count,
        version: 1,
    }
}

fn buffer_versions_for(kernels: &[DescriptorKernel]) -> Vec<DescriptorBufferVersion> {
    let mut versions = Vec::new();
    for kernel in kernels {
        for slot in &kernel.buffers {
            if versions.iter().any(|version: &DescriptorBufferVersion| {
                version.buffer_id == slot.buffer_id && version.version == slot.version
            }) {
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

/// One synthetic MoE transformer layer through the resident schedule.
///
/// The router GEMV + top-k/softmax runs on the device and writes the
/// grouped-dispatch ids/weights buffers; the grouped expert body reads those
/// buffers from the device. Neither buffer is an observation point, so the
/// decode step's only host readback is the declared output observation.
fn plan_path_descriptor(backend: DeviceBackend, module: &[u8]) -> DeviceDescriptor {
    let router_kernel = DescriptorKernel {
        entry: "router_selection".to_owned(),
        buffers: vec![
            slot(
                3,
                "activation",
                0,
                DeviceBufferRole::Input,
                DeviceBufferLifetime::PerStep,
                DeviceBufferInitialization::ZeroFill,
                DeviceDataType::F32,
                PLAN_PATH_K,
            ),
            slot(
                1,
                "router_weight",
                1,
                DeviceBufferRole::Input,
                DeviceBufferLifetime::PerProgram,
                DeviceBufferInitialization::HostProvided,
                DeviceDataType::U8,
                (PLAN_PATH_EXPERTS * 34) as u64,
            ),
            slot(
                4,
                "expert_ids",
                2,
                DeviceBufferRole::InOut,
                DeviceBufferLifetime::PerStep,
                DeviceBufferInitialization::KernelInitialized,
                DeviceDataType::I32,
                PLAN_PATH_ACTIVE,
            ),
            slot(
                5,
                "expert_weights",
                3,
                DeviceBufferRole::InOut,
                DeviceBufferLifetime::PerStep,
                DeviceBufferInitialization::KernelInitialized,
                DeviceDataType::F32,
                PLAN_PATH_ACTIVE,
            ),
        ],
        grid: [PLAN_PATH_ROWS as u32, 1, 1],
        block: [1, 1, 1],
    };
    let grouped_kernel = DescriptorKernel {
        entry: "grouped_expert_gemm".to_owned(),
        buffers: vec![
            slot(
                3,
                "activation",
                0,
                DeviceBufferRole::Input,
                DeviceBufferLifetime::PerStep,
                DeviceBufferInitialization::ZeroFill,
                DeviceDataType::F32,
                PLAN_PATH_K,
            ),
            slot(
                4,
                "expert_ids",
                1,
                DeviceBufferRole::InOut,
                DeviceBufferLifetime::PerStep,
                DeviceBufferInitialization::KernelInitialized,
                DeviceDataType::I32,
                PLAN_PATH_ACTIVE,
            ),
            slot(
                5,
                "expert_weights",
                2,
                DeviceBufferRole::InOut,
                DeviceBufferLifetime::PerStep,
                DeviceBufferInitialization::KernelInitialized,
                DeviceDataType::F32,
                PLAN_PATH_ACTIVE,
            ),
            slot(
                2,
                "expert_weight",
                3,
                DeviceBufferRole::Input,
                DeviceBufferLifetime::PerProgram,
                DeviceBufferInitialization::HostProvided,
                DeviceDataType::U8,
                (PLAN_PATH_EXPERTS * PLAN_PATH_N * 34) as u64,
            ),
            slot(
                6,
                "output",
                4,
                DeviceBufferRole::Output,
                DeviceBufferLifetime::ObservationPoint,
                DeviceBufferInitialization::KernelInitialized,
                DeviceDataType::F32,
                PLAN_PATH_N,
            ),
        ],
        grid: [(PLAN_PATH_ROWS * PLAN_PATH_N) as u32, 1, 1],
        block: [(PLAN_PATH_ROWS * PLAN_PATH_N) as u32, 1, 1],
    };
    let mut descriptor = DeviceDescriptor {
        backend,
        module_image: module.to_vec(),
        kernels: vec![router_kernel, grouped_kernel],
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
        buffer_versions: Vec::new(),
        program_lifetime: DeviceProgramLifetime::RepeatingStep,
        // The router produces the grouped-dispatch ids/weights buffers; the
        // grouped kernel consumes them on the device.
        data_flow: vec![
            DescriptorDataFlow {
                buffer_id: 4,
                version: 1,
                producer: 1,
                consumer: 2,
            },
            DescriptorDataFlow {
                buffer_id: 5,
                version: 1,
                producer: 1,
                consumer: 2,
            },
        ],
        roots: vec![1],
        results: vec![DescriptorResult {
            buffer_id: 6,
            version: 1,
            produced_by: 2,
            at_launch: 2,
        }],
        end_of_run_results: Vec::new(),
    };
    descriptor.buffer_versions = buffer_versions_for(&descriptor.kernels);
    descriptor
}

#[test]
fn synthetic_moe_layer_plan_path_run_on_metal() {
    // Environment-gated: only runs where a real Metal device exists. The
    // fake-driver lanes prove sequencing; this proof is the device numeric
    // golden for the minted grouped-dispatch bodies.
    let Ok(session) = MetalHostSession::try_open() else {
        return;
    };
    let mut host = CompositeHost::with_device(DeviceRuntime::Metal(session), "metal")
        .expect("real Metal composite host");

    let module = moe_family_msl(&plan_path_facts()).expect("mint MoE family MSL");
    let descriptor = plan_path_descriptor(DeviceBackend::Metal, module.as_bytes());
    let byte_weights = BTreeMap::from([
        (
            1,
            DeviceByteBuffer {
                bytes: plan_path_router_packed(),
                dtype: DeviceDataType::U8,
            },
        ),
        (
            2,
            DeviceByteBuffer {
                bytes: plan_path_expert_packed(),
                dtype: DeviceDataType::U8,
            },
        ),
    ]);
    let mut prepared = PreparedResidentSession::prepare_with_weight_bytes(
        &mut host,
        &descriptor,
        &BTreeMap::new(),
        &byte_weights,
    )
    .expect("prepare the synthetic MoE-layer resident session");
    // The real driver reports module-load counters as zero (its leak evidence
    // is the S2-8 real-device gate); the host-side upload counter counts the
    // two HostProvided packed weights copied exactly once at prepare.
    assert_eq!(
        prepared.driver_counters().uploads,
        2,
        "two HostProvided packed weights once-init"
    );

    // CPU parity oracle for the minted bodies.
    let activation = plan_path_activation();
    let router_bind = RouterSelectionBind::packed(
        PLAN_PATH_ROWS,
        PLAN_PATH_K,
        PLAN_PATH_EXPERTS,
        PLAN_PATH_ACTIVE,
        QuantizedFormat::Q8_0,
        [PLAN_PATH_ROWS as u32, 1, 1],
    );
    let grouped_bind = GroupedExpertGemmBind::contiguous(
        PLAN_PATH_ROWS,
        PLAN_PATH_K,
        PLAN_PATH_N,
        PLAN_PATH_EXPERTS,
        QuantizedFormat::Q8_0,
        [(PLAN_PATH_ROWS * PLAN_PATH_N) as u32, 1, 1],
    );
    let mut ids = vec![0u32; (PLAN_PATH_ROWS * PLAN_PATH_ACTIVE) as usize];
    let mut weights = vec![0.0f32; (PLAN_PATH_ROWS * PLAN_PATH_ACTIVE) as usize];
    dispatch_router_selection(
        RouterSelectionKernel::Device,
        &router_bind,
        &activation,
        &plan_path_router_packed(),
        &mut ids,
        &mut weights,
    )
    .expect("CPU router parity");
    assert_eq!(
        ids,
        [1, 2],
        "the three-way top tie must select [1, 2] (lower id wins)"
    );
    let mut expected = vec![0.0f32; (PLAN_PATH_ROWS * PLAN_PATH_N) as usize];
    dispatch_grouped_expert_gemm_selected(
        GroupedExpertGemmKernel::Device,
        &grouped_bind,
        &activation,
        &ids,
        &weights,
        &plan_path_expert_packed(),
        &mut expected,
    )
    .expect("CPU grouped parity");

    // Two decode steps: identical output (deterministic ties through the
    // plan path) and zero per-step host readback of the router / grouped-
    // dispatch buffers (the counter row).
    let input = BTreeMap::from([(3u32, plan_path_activation())]);
    let mut first: Option<Vec<f32>> = None;
    for step in 0..2 {
        let receipt = prepared
            .execute_step(&input)
            .unwrap_or_else(|error| panic!("resident MoE decode step {step}: {error}"));
        let observed = receipt
            .outputs
            .get(&6)
            .cloned()
            .expect("output observation");
        assert_eq!(observed.len(), PLAN_PATH_N as usize, "output width");
        let mut max_dev = 0.0f32;
        let mut max_abs = 0.0f32;
        for (index, (&got, &reference)) in observed.iter().zip(&expected).enumerate() {
            assert!(
                got.is_finite(),
                "non-finite output at index {index}: {got:?}"
            );
            max_dev = max_dev.max((got - reference).abs());
            max_abs = max_abs.max(got.abs());
        }
        // The device arithmetic is f32 scalar-order-parallel: the CPU parity
        // oracle matches within a relative band (the absolute deviation is
        // dominated by the output magnitude, ~1e-7 relative here). The
        // selected expert set is exact — a different tie choice changes the
        // output far beyond this band because each expert owns distinct
        // packed weights.
        assert!(
            max_dev < 1e-3,
            "step {step} output deviates {max_dev} from the CPU parity oracle (absolute band 1e-3)"
        );
        assert!(
            max_dev <= max_abs * 1e-5 + 1e-7,
            "step {step} output relative deviation {max_dev}/{max_abs} exceeds 1e-5"
        );
        // Counter row: the only per-step readback is the declared output
        // observation. The router ids/weights buffers are device-resident.
        assert_eq!(
            receipt.readbacks, 1,
            "step {step} must read back exactly the output observation (zero router readback)"
        );
        assert_eq!(receipt.observation_buffers, vec![6]);
        assert!(
            !receipt.observation_buffers.contains(&4) && !receipt.observation_buffers.contains(&5),
            "grouped-dispatch ids/weights must never be read back per step"
        );
        if let Some(prior) = &first {
            assert_eq!(prior, &observed, "decode steps must be byte-identical");
        } else {
            first = Some(observed.clone());
        }
        println!(
            "m8-u1b plan-path step {step}: readbacks={} max-dev={max_dev} ids={ids:?} weights={weights:?}",
            receipt.readbacks
        );
        // M8-U1c StageTiming evidence rows (probe-class; no llama bar).
        println!(
            "m8-u1c stage-timing step {step}: copy_in_us={} gpu_encode_submit_wait_us={} readback_us={} launch_gpu_us={:?} launches={}",
            receipt.copy_in_us,
            receipt.gpu_encode_submit_wait_us,
            receipt.readback_us,
            receipt.launch_gpu_us,
            receipt.launches
        );
    }

    prepared.teardown().expect("teardown");
    assert_eq!(
        host.device().expect("device").live_handle_count(),
        0,
        "teardown leaves zero live handles"
    );
}
