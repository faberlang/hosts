//! Unit 4 (kernel-codegen-from-mir): physical numeric proof that the
//! compiler-generated four-child MLP (`composed_mlp__0..3`) executes on a
//! real Metal device and matches a CPU reference within a named tolerance.
//!
//! The fixture is the exact Faber-lowered `composed_mlp` body the kernel-
//! codegen units materialize: Gate matmul ∥ Up matmul roots, the SiLU×Up
//! elementwise join, the Down matmul. The device program is built through
//! the REAL generic constructor (`build_device_program`) and the MSL through
//! the REAL device-artifact emitter (`emit_metal_device_artifact`) — never
//! a hand-assembled kernel table or hand-written MSL — then compiled and
//! launched through the product Metal host session
//! (`MTLDevice.makeLibrary(source:)`, the same route module preparation
//! uses) on the enumerated physical device.
//!
//! The Metal crate's Unit-3 fixture is test-only inside `radix-mir-metal`,
//! so the body ships here byte-equivalently with the constructor's
//! naming/stage adaptations applied.

use radix_hir::DefId;
use radix_lexer::{Interner, Span};
use radix_mir::device::{MirCompanionMap, MirDeviceFunctionMetadata, MirDeviceFunctionMetadataMap};
use radix_mir::device_program::{DeviceProgramLifetime, build_device_program};
use radix_mir::{
    BufferId, BufferRole, MirBlock, MirBlockId, MirCollectionOp, MirConstant, MirDeviceRole,
    MirFunction, MirFunctionId, MirFunctionOrigin, MirFunctionVisibility, MirIntrinsic,
    MirKernelShaderStage, MirLocal, MirLocalId, MirOperand, MirParam, MirParamMode, MirPlace,
    MirProgram, MirRuntimeCall, MirStatement, MirStatementKind, MirSyntheticFunctionIdentity,
    MirTemp, MirTempId, MirTerminator, MirTerminatorKind, MirType, MirUnOp, MirValidationContext,
    MirValue, MirValueId, MirValueKind, ValidatedMir,
};
use radix_types::{IndexExpr, NumericWidth, Primitive, Type, TypeTable};

/// The exact direct-op parent shape emitted by the Faber lowering (mirrors
/// `radix-mir`'s `composed_mlp_real_lowered_function`): collection calls
/// write result temps, typed assignments forward results into stage locals,
/// and the result temps carry alias type ids.
fn composed_mlp_real_lowered_function(types: &mut TypeTable) -> MirFunction {
    let rank2 = |types: &mut TypeTable, dims: [u64; 2]| {
        let f32_id = types.sized_numeric(Primitive::Fractus, NumericWidth::F32);
        let dim0 = types.intern_index(IndexExpr::Literal(dims[0]));
        let dim1 = types.intern_index(IndexExpr::Literal(dims[1]));
        let shape = types.intern_index(IndexExpr::Tuple(vec![dim0, dim1]));
        MirType::semantic(types.intern(Type::Tensor(f32_id, shape)))
    };
    let input = rank2(types, [2, 4]);
    let gate_weights = rank2(types, [4, 3]);
    let up_weights = rank2(types, [4, 3]);
    let down_weights = rank2(types, [3, 5]);
    let gate = rank2(types, [2, 3]);
    let up = rank2(types, [2, 3]);
    let activated = rank2(types, [2, 3]);
    let fused = rank2(types, [2, 3]);
    let down = rank2(types, [2, 5]);
    let alias = |name: u32, ty: MirType, types: &mut TypeTable| {
        MirType::semantic(types.intern(Type::Alias(DefId(name), ty.semantic_id())))
    };
    let gate_result = alias(100, gate, types);
    let up_result = alias(101, up, types);
    let activated_result = alias(102, activated, types);
    let fused_result = alias(103, fused, types);
    let down_result = alias(104, down, types);
    let span = Span::default();
    let matmul = |left: MirLocalId,
                  right: MirLocalId,
                  destination: MirTempId,
                  return_ty: MirType| MirStatement {
        kind: MirStatementKind::RuntimeCall {
            destination: Some(MirPlace::temp(destination)),
            call: MirRuntimeCall {
                intrinsic: MirIntrinsic::Collection(MirCollectionOp::TensorMatMul),
                args: vec![
                    MirOperand::Place(MirPlace::local(left)),
                    MirOperand::Place(MirPlace::local(right)),
                ],
                return_ty,
            },
        },
        span,
    };
    let forward = |local: MirLocalId, temp: MirTempId, ty: MirType, id: u32| MirStatement {
        kind: MirStatementKind::Assign {
            place: MirPlace::local(local),
            value: MirValue {
                id: MirValueId(id),
                kind: MirValueKind::Operand(MirOperand::Temp(temp)),
                ty,
                span,
            },
        },
        span,
    };
    let runtime = |op: MirCollectionOp,
                   destination: MirTempId,
                   args: Vec<MirOperand>,
                   return_ty: MirType| MirStatement {
        kind: MirStatementKind::RuntimeCall {
            destination: Some(MirPlace::temp(destination)),
            call: MirRuntimeCall {
                intrinsic: MirIntrinsic::Collection(op),
                args,
                return_ty,
            },
        },
        span,
    };
    MirFunction {
        id: MirFunctionId(0),
        source: None,
        name: None,
        params: vec![
            MirParam {
                local: MirLocalId(0),
                name: None,
                ty: input,
                mode: MirParamMode::Owned,
                span,
            },
            MirParam {
                local: MirLocalId(1),
                name: None,
                ty: gate_weights,
                mode: MirParamMode::Owned,
                span,
            },
            MirParam {
                local: MirLocalId(2),
                name: None,
                ty: up_weights,
                mode: MirParamMode::Owned,
                span,
            },
            MirParam {
                local: MirLocalId(3),
                name: None,
                ty: down_weights,
                mode: MirParamMode::Owned,
                span,
            },
            MirParam {
                local: MirLocalId(4),
                name: None,
                ty: down,
                mode: MirParamMode::MutRef,
                span,
            },
            MirParam {
                local: MirLocalId(5),
                name: None,
                ty: MirType::semantic(types.sized_numeric(Primitive::Fractus, NumericWidth::F32)),
                mode: MirParamMode::Owned,
                span,
            },
        ],
        locals: vec![
            MirLocal {
                id: MirLocalId(6),
                name: None,
                ty: gate,
                mutable: true,
                span,
            },
            MirLocal {
                id: MirLocalId(7),
                name: None,
                ty: up,
                mutable: false,
                span,
            },
            MirLocal {
                id: MirLocalId(8),
                name: None,
                ty: activated,
                mutable: false,
                span,
            },
            MirLocal {
                id: MirLocalId(9),
                name: None,
                ty: fused,
                mutable: false,
                span,
            },
        ],
        temps: vec![
            MirTemp {
                id: MirTempId(0),
                ty: gate_result,
                span,
            },
            MirTemp {
                id: MirTempId(1),
                ty: up_result,
                span,
            },
            MirTemp {
                id: MirTempId(2),
                ty: activated,
                span,
            },
            MirTemp {
                id: MirTempId(3),
                ty: activated,
                span,
            },
            MirTemp {
                id: MirTempId(4),
                ty: activated,
                span,
            },
            MirTemp {
                id: MirTempId(5),
                ty: activated_result,
                span,
            },
            MirTemp {
                id: MirTempId(6),
                ty: fused_result,
                span,
            },
            MirTemp {
                id: MirTempId(7),
                ty: down_result,
                span,
            },
        ],
        blocks: vec![MirBlock {
            id: MirBlockId(0),
            statements: vec![
                matmul(MirLocalId(0), MirLocalId(1), MirTempId(0), gate_result),
                forward(MirLocalId(6), MirTempId(0), gate, 0),
                matmul(MirLocalId(0), MirLocalId(2), MirTempId(1), up_result),
                forward(MirLocalId(7), MirTempId(1), up, 1),
                runtime(
                    MirCollectionOp::TensorNeg,
                    MirTempId(2),
                    vec![MirOperand::Place(MirPlace::local(MirLocalId(6)))],
                    activated,
                ),
                MirStatement {
                    kind: MirStatementKind::Assign {
                        place: MirPlace::temp(MirTempId(3)),
                        value: MirValue {
                            id: MirValueId(2),
                            kind: MirValueKind::Unary {
                                op: MirUnOp::Exp,
                                operand: MirOperand::Temp(MirTempId(2)),
                            },
                            ty: activated,
                            span,
                        },
                    },
                    span,
                },
                runtime(
                    MirCollectionOp::TensorAdd,
                    MirTempId(4),
                    vec![
                        MirOperand::Temp(MirTempId(3)),
                        MirOperand::Constant(MirConstant::Float(1.0)),
                    ],
                    activated,
                ),
                runtime(
                    MirCollectionOp::TensorDiv,
                    MirTempId(5),
                    vec![
                        MirOperand::Place(MirPlace::local(MirLocalId(6))),
                        MirOperand::Temp(MirTempId(4)),
                    ],
                    activated_result,
                ),
                forward(MirLocalId(8), MirTempId(5), activated, 3),
                runtime(
                    MirCollectionOp::TensorMul,
                    MirTempId(6),
                    vec![
                        MirOperand::Place(MirPlace::local(MirLocalId(8))),
                        MirOperand::Place(MirPlace::local(MirLocalId(7))),
                    ],
                    fused_result,
                ),
                forward(MirLocalId(9), MirTempId(6), fused, 4),
                matmul(MirLocalId(9), MirLocalId(3), MirTempId(7), down_result),
                forward(MirLocalId(4), MirTempId(7), down, 5),
            ],
            terminator: MirTerminator {
                kind: MirTerminatorKind::Return(None),
                span,
            },
            span,
        }],
        return_ty: MirType::semantic(types.primitive(Primitive::Vacuum)),
        error_ty: None,
        is_async: false,
        is_generator: false,
        shader_stage: None,
        span,
    }
}

/// The constructor-facing fixture: the real-lowered body carries the name,
/// compute stage, plain-tensor collection results, and named locals the
/// generic constructor requires.
fn named_composed_mlp_function(types: &mut TypeTable, interner: &mut Interner) -> MirFunction {
    let mut function = composed_mlp_real_lowered_function(types);
    function.name = Some(interner.intern("composed_mlp"));
    function.shader_stage = Some(MirKernelShaderStage::Compute);
    for block in &mut function.blocks {
        for statement in &mut block.statements {
            if let MirStatementKind::RuntimeCall { call, .. } = &mut statement.kind
                && let Type::Alias(_, inner) = types.get(call.return_ty.semantic_id())
            {
                call.return_ty = MirType::semantic(*inner);
            }
        }
    }
    let local_names: [(u32, &str); 10] = [
        (0, "x"),
        (1, "gate_weight"),
        (2, "up_weight"),
        (3, "down_weight"),
        (4, "down"),
        (5, "seed"),
        (6, "gate"),
        (7, "up"),
        (8, "activated"),
        (9, "fused"),
    ];
    for (id, name) in local_names {
        let symbol = interner.intern(name);
        if let Some(param) = function
            .params
            .iter_mut()
            .find(|param| param.local == MirLocalId(id))
        {
            param.name = Some(symbol);
        }
        if let Some(local) = function
            .locals
            .iter_mut()
            .find(|local| local.id == MirLocalId(id))
        {
            local.name = Some(symbol);
        }
    }
    function
}

fn fixture_metadata(functions: &[MirFunction]) -> MirDeviceFunctionMetadataMap {
    functions
        .iter()
        .filter(|function| function.shader_stage == Some(MirKernelShaderStage::Compute))
        .map(|function| {
            (
                function.id,
                MirDeviceFunctionMetadata {
                    identity: MirSyntheticFunctionIdentity {
                        parent: MirFunctionId(0),
                        ordinal: 0,
                    },
                    role: MirDeviceRole::Kernel,
                    shader_stage: MirKernelShaderStage::Compute,
                    visibility: MirFunctionVisibility::Public,
                    origin: MirFunctionOrigin::Source,
                },
            )
        })
        .collect()
}

/// The real generic-constructor four-child MLP device program plus its MSL
/// artifact — the exact Unit-3 acceptance pipeline this unit executes.
fn four_child_mlp_artifact() -> (
    radix_mir_metal::MetalDeviceArtifact,
    radix_mir::device_program::DeviceProgram,
    Vec<radix_mir::device_semantics::DependencyEdge>,
) {
    let mut types = TypeTable::new();
    let mut interner = Interner::new();
    let function = named_composed_mlp_function(&mut types, &mut interner);
    let validated = ValidatedMir::new(
        MirProgram {
            functions: vec![function],
        },
        MirValidationContext::new(&types),
    )
    .expect("the composed MLP fixture validates");
    let (device_program, semantics, _) = build_device_program(
        &validated,
        &interner,
        &MirCompanionMap::default(),
        None,
        DeviceProgramLifetime::SingleRun,
        0,
        &fixture_metadata(&validated.program().functions),
    )
    .expect("the composed MLP body materializes through the generic path");
    let artifact =
        radix_mir_metal::emit_metal_device_artifact(&device_program, &validated, &interner)
            .expect("the four-child device program emits MSL");
    (artifact, device_program, semantics.dependencies)
}

/// Structural regression anchor (portable, no device): the generated
/// program is four ordinary child launches with the fork/join DAG the goal
/// settles — Gate ∥ Up roots, a join consuming both, Down after the join —
/// never one composed parent launch and never a Gate→Up edge.
#[test]
fn four_child_program_shape_is_generated_children() {
    let (artifact, program, dependencies) = four_child_mlp_artifact();
    let entries: Vec<&str> = program
        .kernels
        .iter()
        .map(|kernel| kernel.entry.as_str())
        .collect();
    assert_eq!(
        entries,
        [
            "composed_mlp__0",
            "composed_mlp__1",
            "composed_mlp__2",
            "composed_mlp__3",
        ],
        "the parent must lower to the four generated children"
    );
    assert_eq!(program.launches.len(), 4, "one launch per child");
    let launched: Vec<&str> = program
        .launches
        .iter()
        .map(|launch| program.kernels[launch.kernel_index].entry.as_str())
        .collect();
    assert_eq!(launched, entries, "launch enumeration covers every child");
    let mut edges: Vec<(u32, u32, u32)> = dependencies
        .iter()
        .map(|edge| (edge.producer.0, edge.consumer.0, edge.buffer.0))
        .collect();
    edges.sort_unstable();
    // Launch ids are 1-based: 1=gate matmul, 2=up matmul, 3=join, 4=down.
    // The join (3) consumes BOTH roots' buffers (3=gate, 5=up); the down
    // matmul (4) consumes the fused join result (6). No Gate→Up edge exists.
    assert_eq!(edges, vec![(1, 3, 3), (2, 3, 5), (3, 4, 6)]);
    // The emitted module declares every child entry (generated MSL, not a
    // composed parent body).
    for entry in &entries {
        let header = format!("kernel void {entry}(");
        assert!(
            artifact.source.contains(&header),
            "the emitted module must declare generated child {entry}"
        );
    }
    assert!(
        !artifact
            .source
            .contains("if (workgroup_id.x == 0u && workgroup_id.y == 0u)"),
        "no workgroup-zero serialization may survive in child MSL"
    );
}

/// Deterministic input values in (-1, 1) from a fixed-seed LCG (platform-
/// independent bit math; every environment gets the same tensors).
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // 31 high bits of the state, mapped to (-1, 1).
        let bits = (self.0 >> 33) as f64;
        ((bits / 2_147_483_648.0) * 2.0 - 1.0) as f32
    }

    fn values(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| self.next_f32()).collect()
    }
}

/// CPU reference in f64 over the exact lowered semantics: the matmul right
/// operands are `[N][K]`-indexed (`w[n * K + k]`, the emitted MSL's
/// `col * K + k` addressing) and the join is
/// `silu(gate) * up` with `silu(g) = g / (exp(-g) + 1)`.
fn mlp_reference(
    x: &[f32],
    gate_weight: &[f32],
    up_weight: &[f32],
    down_weight: &[f32],
) -> Vec<f64> {
    let silu = |g: f64| g / ((-g).exp() + 1.0);
    let matmul = |a: &[f32], m: usize, k_len: usize, w: &[f32], n: usize| -> Vec<f64> {
        let mut out = vec![0.0f64; m * n];
        for r in 0..m {
            for c in 0..n {
                let mut acc = 0.0f64;
                for (k, a_elem) in a[r * k_len..r * k_len + k_len].iter().enumerate() {
                    acc += f64::from(*a_elem) * f64::from(w[c * k_len + k]);
                }
                out[r * n + c] = acc;
            }
        }
        out
    };
    let gate = matmul(x, 2, 4, gate_weight, 3);
    let up = matmul(x, 2, 4, up_weight, 3);
    let fused: Vec<f64> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
    let fused_f32: Vec<f32> = fused.iter().map(|value| *value as f32).collect();
    matmul(&fused_f32, 2, 3, down_weight, 5)
}

/// Goal Unit 4 done-when: the generated four-child MLP plan executes on a
/// REAL Metal device (runtime `makeLibrary(source:)`, generated child
/// launches, zero CPU substitutes/bridges) and its readback is finite and
/// within a named explicit tolerance of the CPU reference. Off-macOS the
/// run reports NOT ATTEMPTED; a missing device never becomes green
/// evidence.
#[test]
#[ignore = "physical Metal gate; run with -- --ignored --nocapture"]
fn physical_four_child_mlp_numeric() {
    const ABS_TOLERANCE: f64 = 1e-4;
    const REL_TOLERANCE: f64 = 1e-3;

    let (artifact, program, _dependencies) = four_child_mlp_artifact();
    #[cfg(target_os = "macos")]
    {
        use faber_host_macos_arm64::{
            MetalHostSession, enumerate_metal_physical_devices, metal_host::MetalHandleId,
        };
        use radix_mir::device_program::DeviceResource;
        use std::collections::BTreeMap;

        let devices = enumerate_metal_physical_devices()
            .expect("enumerating local Metal devices must not fail on macOS");
        let (device_model, registry_id) = devices
            .first()
            .map(|device| {
                (
                    device.device_model.clone().unwrap_or_default(),
                    device.registry_id.clone(),
                )
            })
            .expect("physical Metal execution requires an enumerated Metal device");

        let mut session = MetalHostSession::try_open()
            .expect("physical Metal execution requires an admitted Metal product stack");
        let module = session
            .load_module(artifact.source.as_bytes())
            .expect("the generated four-child MSL must compile at runtime");

        // One device buffer per distinct program resource identity. Every
        // kernel that names a buffer must agree on its element count.
        let mut buffer_counts: BTreeMap<BufferId, u64> = BTreeMap::new();
        let mut buffer_names: BTreeMap<BufferId, String> = BTreeMap::new();
        for kernel in &program.kernels {
            for resource in &kernel.resources {
                let count = resource.version.element_count;
                if let Some(seen) = buffer_counts.insert(resource.buffer.id, count)
                    && seen != count
                {
                    panic!(
                        "buffer {} ({}) has conflicting element counts {seen} vs {count}",
                        resource.buffer.id.0, resource.buffer.name
                    );
                }
                buffer_names.insert(resource.buffer.id, resource.buffer.name.clone());
            }
        }
        let mut handles: BTreeMap<BufferId, MetalHandleId> = BTreeMap::new();
        for (id, count) in &buffer_counts {
            let handle = session
                .alloc_bytes(usize::try_from(*count).expect("element count fits usize") * 4)
                .expect("device buffer allocation must succeed");
            handles.insert(*id, handle);
        }

        // Deterministic host-provided inputs for the Input-role buffers;
        // everything else is kernel-initialized before its first read.
        let is_input_buffer = |id: &BufferId| {
            program.kernels.iter().any(|kernel| {
                kernel.resources.iter().any(|resource| {
                    resource.buffer.id == *id && resource.buffer.role == BufferRole::Input
                })
            })
        };
        let mut lcg = Lcg(0x5eed_1234);
        let mut inputs: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        for (id, name) in &buffer_names {
            if is_input_buffer(id) {
                let count = buffer_counts[id];
                let values = lcg.values(count as usize);
                session
                    .copy_in_f32(handles[id], &values)
                    .unwrap_or_else(|error| panic!("copy_in for {name} failed: {error:?}"));
                inputs.insert(name.clone(), values);
            }
        }
        let named_input = |name: &str| -> Vec<f32> {
            inputs
                .get(name)
                .unwrap_or_else(|| panic!("input buffer {name} must exist"))
                .clone()
        };
        let x = named_input("x");
        let gate_weight = named_input("gate_weight");
        let up_weight = named_input("up_weight");
        let down_weight = named_input("down_weight");

        // Execute the carried launch list: buffers bind in (group, binding)
        // order, which is exactly the emitted `[[buffer(i)]]` order.
        let mut launched_entries: Vec<String> = Vec::new();
        for launch in &program.launches {
            let kernel = &program.kernels[launch.kernel_index];
            let mut slots: Vec<&DeviceResource> = kernel.resources.iter().collect();
            slots.sort_by_key(|resource| (resource.binding.group, resource.binding.binding));
            for (index, resource) in slots.iter().enumerate() {
                assert_eq!(
                    (resource.binding.group, resource.binding.binding),
                    (0, index as u32),
                    "{}: positional Metal buffer index must equal the emitted [[buffer({index})]]",
                    kernel.entry
                );
            }
            let bound: Vec<MetalHandleId> = slots
                .iter()
                .map(|resource| handles[&resource.buffer.id])
                .collect();
            let grid = kernel.launch.workgroup_count;
            let block = kernel.launch.workgroup;
            let grid = [
                u32::try_from(grid.x).expect("grid extent fits u32"),
                u32::try_from(grid.y).expect("grid extent fits u32"),
                u32::try_from(grid.z).expect("grid extent fits u32"),
            ];
            session
                .launch_kernel_3d(
                    module,
                    &kernel.entry,
                    &bound,
                    grid[0],
                    grid[1],
                    grid[2],
                    block.x,
                    block.y,
                    block.z,
                )
                .unwrap_or_else(|error| panic!("launch {} failed: {error:?}", kernel.entry));
            launched_entries.push(kernel.entry.clone());
        }
        assert_eq!(
            launched_entries,
            [
                "composed_mlp__0",
                "composed_mlp__1",
                "composed_mlp__2",
                "composed_mlp__3",
            ],
            "every generated child launches exactly once — no composed parent"
        );

        // One declared completion boundary: all four encodes commit in one
        // command buffer.
        session.sync().expect("the step boundary must commit");
        assert_eq!(
            session.command_submit_count(),
            1,
            "the four child launches must share one command buffer"
        );
        assert!(
            session.blocking_wait_count() >= 1,
            "the step boundary must wait on the device"
        );

        // Read back the program's declared result buffer.
        assert_eq!(program.results.len(), 1, "one declared result");
        let result = &program.results[0];
        assert_eq!(result.buffer.role, BufferRole::Output);
        assert_eq!(
            result.produced_by, program.launches[3].id,
            "the result is produced by the last child launch"
        );
        let element_count = result.version.element_count as usize;
        let gpu = session
            .readback_f32(handles[&result.buffer.id])
            .expect("result readback must succeed");
        assert_eq!(gpu.len(), element_count, "readback covers the full result");
        for (index, value) in gpu.iter().enumerate() {
            assert!(
                value.is_finite(),
                "output element {index} is not finite: {value}"
            );
        }

        // Numeric oracle: the CPU reference is the comparison only — no
        // CPU substitute executed any part of the plan.
        let expected = mlp_reference(&x, &gate_weight, &up_weight, &down_weight);
        let mut max_abs_deviation = 0.0f64;
        let mut worst_index = 0usize;
        for (index, (observed, reference)) in gpu.iter().zip(&expected).enumerate() {
            let deviation = (f64::from(*observed) - reference).abs();
            if deviation > max_abs_deviation {
                max_abs_deviation = deviation;
                worst_index = index;
            }
            let bound = ABS_TOLERANCE + REL_TOLERANCE * reference.abs();
            assert!(
                deviation <= bound,
                "output element {index}: |{observed} - {reference}| = {deviation} exceeds \
                 tolerance |gpu-ref| <= {ABS_TOLERANCE} + {REL_TOLERANCE}*|ref| = {bound}"
            );
        }

        // Receipt (kernel-codegen Unit 4 done-when).
        println!("RECEIPT kernel-codegen Unit 4: physical numeric four-child MLP");
        println!("  metal_device: {device_model:?} (registry {registry_id})");
        println!("  module: runtime MTLDevice.makeLibrary(source:) over the generated artifact");
        println!("  child_entries: {}", launched_entries.join(", "));
        println!("  launches: 4 generated child launches, 0 composed parent launches");
        println!("  cpu_substitutes: 0, cpu_bridges: 0 (reference used for comparison only)");
        println!(
            "  command_buffers: 1 ({} submits, {} waits)",
            session.command_submit_count(),
            session.blocking_wait_count()
        );
        println!(
            "  output: {element_count} elements, all finite; result buffer `{}`",
            buffer_names[&result.buffer.id]
        );
        println!(
            "  tolerance: |gpu-ref| <= {ABS_TOLERANCE} + {REL_TOLERANCE}*|ref| per element: PASS \
             (max_abs_deviation={max_abs_deviation:.3e} at index {worst_index})"
        );
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (&artifact, &program);
        eprintln!(
            "NOT ATTEMPTED: physical four-child MLP execution requires a macOS Metal device; \
             this target has none"
        );
    }
}
