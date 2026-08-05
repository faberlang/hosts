//! Composite-host fake-backend sequencing + negative tests (campaign S1-4).
//!
//! These tests prove the N1.4 fail-before-launch surface on the composite
//! host with injected fake drivers: missing backend, bad descriptor, ABI /
//! entry / dtype / shape mismatch all fail with typed diagnostics before any
//! launch; the valid lifecycle (load → alloc → copy → launch → sync →
//! readback → release) sequences correctly and teardown releases every
//! handle. Real-device proofs are S1-6; the fakes prove sequencing only.

use std::collections::BTreeMap;

use faber::device::{DeviceBackend, DeviceHandle, DeviceHandleKind, DeviceSelection};
use faber::Valor;
use faber_host_macos_arm64::composite_host::{
    resolve_device_selection, CompletionBoundary, CompositeHost, CompositeHostConfig, DataFlowEdge,
};
use faber_host_macos_arm64::cuda_host::E_CUDA_DRIVER;
use faber_host_macos_arm64::device_descriptor::{
    fnv1a64, DescriptorBuffer, DescriptorBufferVersion, DescriptorDataFlow, DescriptorKernel,
    DescriptorLaunch, DescriptorResult, DeviceBufferInitialization, DeviceBufferLifetime,
    DeviceBufferRole, DeviceDataType, DeviceDescriptor, DeviceProgramLifetime,
    E_BACKEND_UNAVAILABLE, E_DEVICE_ABI_MISMATCH, E_DEVICE_DESCRIPTOR, E_DEVICE_DTYPE_MISMATCH,
    E_DEVICE_ENTRY_MISMATCH, E_DEVICE_SHAPE_MISMATCH, E_NO_DEVICE_PROGRAM,
};
use faber_host_macos_arm64::device_host::{DeviceRuntime, DeviceSession, E_DEVICE_INVALID_HANDLE};
use faber_host_macos_arm64::device_registry::FakeFailureStage;
use faber_host_macos_arm64::kernel::frame_data;
use faber_host_macos_arm64::metal_host::E_METAL_DRIVER;
use faber_host_macos_arm64::{
    CudaHostSession, FakeCudaDriver, FakeMetalDriver, Frame, MetalHostSession, Status,
};

const MODULE_IMAGE: &[u8] = b"// fake compiler-owned module image";

/// One elementwise-add kernel: `out = a + b` over `count` f32 elements.
/// Matches the simulated `addita` / `add_one` kernel shape (3 buffers).
/// The S2-4 lifetime mapping the faber constructor derives from ABI facts:
/// Input → PerProgram, Output → ObservationPoint, InOut → PerStep. Test
/// descriptors mirror that mapping so the fake-driver sequencing proves the
/// constructor-derived payload path.
fn lifetime_for_role(role: DeviceBufferRole) -> DeviceBufferLifetime {
    match role {
        DeviceBufferRole::Input => DeviceBufferLifetime::PerProgram,
        DeviceBufferRole::Output => DeviceBufferLifetime::ObservationPoint,
        DeviceBufferRole::InOut => DeviceBufferLifetime::PerStep,
    }
}

fn add_slot(
    id: u32,
    name: &str,
    role: DeviceBufferRole,
    binding: u32,
    count: u64,
) -> DescriptorBuffer {
    add_slot_version(id, name, role, binding, count, 1)
}

fn add_slot_version(
    id: u32,
    name: &str,
    role: DeviceBufferRole,
    binding: u32,
    count: u64,
    version: u32,
) -> DescriptorBuffer {
    DescriptorBuffer {
        buffer_id: id,
        buffer_name: name.to_owned(),
        // F1: one distinct semantic value per buffer identity (the wire's
        // carried value fact; the faber constructor mints value id == buffer
        // id for the first projection).
        semantic_value: id,
        role,
        lifetime: lifetime_for_role(role),
        // F5: the initialization axis is a carried fact, decided from the
        // role for hand-built descriptors (HostProvided inputs, ZeroFill
        // InOut state, KernelInitialized outputs) — the same classification
        // the faber constructor projects from the wire.
        initialization: initialization_for_role(role),
        binding,
        element_ty: DeviceDataType::F32,
        element_count: count,
        version,
    }
}

/// The constructor's role-consistent initialization classification (F5) for
/// hand-built test descriptors: inputs are uploaded, InOut state is
/// zero-filled, outputs are kernel-initialized. Mirrors the faber
/// constructor's carried initialization facts.
fn initialization_for_role(role: DeviceBufferRole) -> DeviceBufferInitialization {
    match role {
        DeviceBufferRole::Input => DeviceBufferInitialization::HostProvided,
        DeviceBufferRole::InOut => DeviceBufferInitialization::ZeroFill,
        DeviceBufferRole::Output => DeviceBufferInitialization::KernelInitialized,
    }
}

/// The constructor rule for legal execution roots (F3): every launch no
/// dependency edge consumes. Mirrors the faber materializer's root set.
fn default_roots(launches: &[DescriptorLaunch], data_flow: &[DescriptorDataFlow]) -> Vec<u32> {
    launches
        .iter()
        .filter(|launch| !data_flow.iter().any(|edge| edge.consumer == launch.id))
        .map(|launch| launch.id)
        .collect()
}

/// One declared observation point (F6): the buffer the host reads back at
/// its producing launch's completion boundary.
fn result(id: u32, produced_by: u32) -> DescriptorResult {
    DescriptorResult {
        buffer_id: id,
        version: 1,
        produced_by,
        at_launch: produced_by,
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

fn make_descriptor(
    backend: DeviceBackend,
    kernels: Vec<DescriptorKernel>,
    launches: Vec<DescriptorLaunch>,
    data_flow: Vec<DescriptorDataFlow>,
    results: Vec<DescriptorResult>,
) -> DeviceDescriptor {
    let roots = default_roots(&launches, &data_flow);
    DeviceDescriptor {
        backend,
        module_image: MODULE_IMAGE.to_vec(),
        buffer_versions: buffer_versions_for(&kernels),
        kernels,
        launches,
        program_lifetime: DeviceProgramLifetime::SingleRun,
        data_flow,
        roots,
        results,
    }
}

fn elementwise_add_descriptor(backend: DeviceBackend, entry: &str, count: u64) -> DeviceDescriptor {
    make_descriptor(
        backend,
        vec![DescriptorKernel {
            entry: entry.to_owned(),
            buffers: vec![
                add_slot(1, "a", DeviceBufferRole::Input, 0, count),
                add_slot(2, "b", DeviceBufferRole::Input, 1, count),
                add_slot(3, "out", DeviceBufferRole::Output, 2, count),
            ],
            grid: [1, 1, 1],
            block: [count as u32, 1, 1],
        }],
        vec![DescriptorLaunch {
            id: 1,
            kernel_index: 0,
        }],
        Vec::new(),
        vec![result(3, 1)],
    )
}

fn metal_composite(entry: &str) -> HostResult<CompositeHost> {
    let runtime = DeviceRuntime::Metal(
        MetalHostSession::with_driver(Box::new(FakeMetalDriver::default().with_known_entry(entry)))
            .expect("fake metal admit"),
    );
    CompositeHost::with_device(runtime, "fake-metal-device")
}

fn multi_kernel_metal_composite() -> HostResult<CompositeHost> {
    let runtime = DeviceRuntime::Metal(
        MetalHostSession::with_driver(Box::new(
            FakeMetalDriver::default()
                .with_known_entry("kernel_zero")
                .with_known_entry("kernel_one"),
        ))
        .expect("fake metal admit"),
    );
    CompositeHost::with_device(runtime, "fake-metal-device")
}

fn cuda_composite(entry: &str) -> HostResult<CompositeHost> {
    let runtime = DeviceRuntime::Cuda(
        CudaHostSession::with_driver(Box::new(FakeCudaDriver::default().with_known_entry(entry)))
            .expect("fake cuda admit"),
    );
    CompositeHost::with_device(runtime, "fake-cuda-device")
}

/// A fake-metal composite whose module declares every kernel entry of the
/// S3-B3 Mul+Mean companion program (S3-B3 tests launch all four kernels).
fn mul_mean_metal_composite() -> HostResult<CompositeHost> {
    let runtime = DeviceRuntime::Metal(
        MetalHostSession::with_driver(Box::new(
            FakeMetalDriver::default()
                .with_known_entry("loss_mul")
                .with_known_entry("loss_mean")
                .with_known_entry("loss_backward_x")
                .with_known_entry("loss_backward_w"),
        ))
        .expect("fake metal admit"),
    );
    CompositeHost::with_device(runtime, "fake-metal-device")
}

/// The CUDA lane of [`mul_mean_metal_composite`].
fn mul_mean_cuda_composite() -> HostResult<CompositeHost> {
    let runtime = DeviceRuntime::Cuda(
        CudaHostSession::with_driver(Box::new(
            FakeCudaDriver::default()
                .with_known_entry("loss_mul")
                .with_known_entry("loss_mean")
                .with_known_entry("loss_backward_x")
                .with_known_entry("loss_backward_w"),
        ))
        .expect("fake cuda admit"),
    );
    CompositeHost::with_device(runtime, "fake-cuda-device")
}

/// A fake-metal composite whose driver fails the `call`-th invocation of
/// `stage` with a typed `E_METAL_DRIVER` error (S2-3 failure injection).
fn metal_composite_failing(
    entry: &str,
    stage: FakeFailureStage,
    call: u32,
) -> HostResult<CompositeHost> {
    let runtime = DeviceRuntime::Metal(
        MetalHostSession::with_driver(Box::new(
            FakeMetalDriver::default()
                .with_known_entry(entry)
                .with_failure_at(stage, call),
        ))
        .expect("fake metal admit"),
    );
    CompositeHost::with_device(runtime, "fake-metal-device")
}

/// A fake-cuda composite whose driver fails the `call`-th invocation of
/// `stage` with a typed `E_CUDA_DRIVER` error (S2-3 failure injection).
fn cuda_composite_failing(
    entry: &str,
    stage: FakeFailureStage,
    call: u32,
) -> HostResult<CompositeHost> {
    let runtime = DeviceRuntime::Cuda(
        CudaHostSession::with_driver(Box::new(
            FakeCudaDriver::default()
                .with_known_entry(entry)
                .with_failure_at(stage, call),
        ))
        .expect("fake cuda admit"),
    );
    CompositeHost::with_device(runtime, "fake-cuda-device")
}

/// The two-kernel InOut chain descriptor (kernel 1 writes `acc`, kernel 2
/// reads it): a, b → acc → out, with c as kernel 2's second input. Shared by
/// the mid-chain failure-injection test and the S2-1 chaining tests. The
/// `backend` targets the fake driver of the composite host under test (the
/// shape is backend-neutral, so the same mid-chain coverage runs on the fake
/// Metal and fake CUDA lanes).
fn two_kernel_inout_descriptor(backend: DeviceBackend) -> DeviceDescriptor {
    make_descriptor(
        backend,
        vec![
            DescriptorKernel {
                entry: "add_one".to_owned(),
                buffers: vec![
                    add_slot(1, "a", DeviceBufferRole::Input, 0, 2),
                    add_slot(2, "b", DeviceBufferRole::Input, 1, 2),
                    add_slot(3, "acc", DeviceBufferRole::InOut, 2, 2),
                ],
                grid: [1, 1, 1],
                block: [2, 1, 1],
            },
            DescriptorKernel {
                entry: "add_one".to_owned(),
                buffers: vec![
                    add_slot(3, "acc", DeviceBufferRole::InOut, 0, 2),
                    add_slot(4, "c", DeviceBufferRole::Input, 1, 2),
                    add_slot(5, "out", DeviceBufferRole::Output, 2, 2),
                ],
                grid: [1, 1, 1],
                block: [2, 1, 1],
            },
        ],
        vec![
            DescriptorLaunch {
                id: 1,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 2,
                kernel_index: 1,
            },
        ],
        // R2: the data-flow edge is CARRIED by the wire (buffer 3 `acc`,
        // version 1: launch 1 produces, launch 2 consumes) — the host
        // consumes it; it never re-derives a first-writer edge from launch
        // order.
        vec![DescriptorDataFlow {
            buffer_id: 3,
            version: 1,
            producer: 1,
            consumer: 2,
        }],
        vec![result(5, 2)],
    )
}

/// Host inputs for [`two_kernel_inout_descriptor`]: a, b, and c.
fn two_kernel_inputs() -> BTreeMap<u32, Vec<f32>> {
    let mut inputs = BTreeMap::new();
    inputs.insert(1, vec![1.0, 2.0]);
    inputs.insert(2, vec![3.0, 4.0]);
    inputs.insert(4, vec![10.0, 10.0]);
    inputs
}

/// The S3-B3 Mul+Mean companion fixture (S3-B1 shape) as a descriptor: the
/// forward kernels (`loss_mul` elementwise mul, `loss_mean` mean reduction)
/// plus the generated backward companion, whose tuple gradient outputs
/// `grad_x`/`grad_w` are ObservationPoint and whose accumulation/partial
/// intermediates are PerStep. The buffer inventory mirrors the S3-B3
/// evidence-note classification:
///
/// | Buffer | Role → class |
/// | --- | --- |
/// | x, w (1, 2) | Input → PerProgram |
/// | product, partial, acc (3, 4, 5) | InOut → PerStep |
/// | grad_x, grad_w (6, 7) | Output → ObservationPoint |
///
/// The fake driver simulates the 3-buffer elementwise-add kernel only, so
/// the companion's `(grad_x, grad_w)` tuple is modeled as two companion
/// kernels, each producing one gradient output (the real materialized
/// companion writes both tuple elements in one kernel via the S3-A1
/// multi-output ABI). Every slot's lifetime is derived by the same
/// `lifetime_for_role` mapping the faber constructor uses (S2-4), so the
/// fake-driver sequencing proves the constructor-derived classification's
/// allocation/recycle/release policy: PerProgram allocated once, PerStep
/// recycled at the step boundary, ObservationPoint read-then-released.
fn mul_mean_companion_descriptor(backend: DeviceBackend) -> DeviceDescriptor {
    make_descriptor(
        backend,
        vec![
            DescriptorKernel {
                entry: "loss_mul".to_owned(),
                buffers: vec![
                    add_slot(1, "x", DeviceBufferRole::Input, 0, 2),
                    add_slot(2, "w", DeviceBufferRole::Input, 1, 2),
                    add_slot(3, "product", DeviceBufferRole::InOut, 2, 2),
                ],
                grid: [1, 1, 1],
                block: [2, 1, 1],
            },
            DescriptorKernel {
                entry: "loss_mean".to_owned(),
                buffers: vec![
                    add_slot(3, "product", DeviceBufferRole::InOut, 0, 2),
                    add_slot(4, "partial", DeviceBufferRole::InOut, 1, 2),
                    add_slot(5, "acc", DeviceBufferRole::InOut, 2, 2),
                ],
                grid: [1, 1, 1],
                block: [2, 1, 1],
            },
            DescriptorKernel {
                entry: "loss_backward_x".to_owned(),
                buffers: vec![
                    add_slot(5, "acc", DeviceBufferRole::InOut, 0, 2),
                    add_slot(1, "x", DeviceBufferRole::Input, 1, 2),
                    add_slot(6, "grad_x", DeviceBufferRole::Output, 2, 2),
                ],
                grid: [1, 1, 1],
                block: [2, 1, 1],
            },
            DescriptorKernel {
                entry: "loss_backward_w".to_owned(),
                buffers: vec![
                    add_slot(5, "acc", DeviceBufferRole::InOut, 0, 2),
                    add_slot(2, "w", DeviceBufferRole::Input, 1, 2),
                    add_slot(7, "grad_w", DeviceBufferRole::Output, 2, 2),
                ],
                grid: [1, 1, 1],
                block: [2, 1, 1],
            },
        ],
        (0..4)
            .map(|kernel_index| DescriptorLaunch {
                id: kernel_index + 1,
                kernel_index,
            })
            .collect(),
        Vec::new(),
        // F6: the declared observation points — grad_x (launch 3) and
        // grad_w (launch 4) are the only readbacks.
        vec![result(6, 3), result(7, 4)],
    )
}

/// Host inputs for [`mul_mean_companion_descriptor`]: x and w (PerProgram
/// inputs read by the forward `loss_mul` kernel and the companion kernels).
fn mul_mean_inputs() -> BTreeMap<u32, Vec<f32>> {
    let mut inputs = BTreeMap::new();
    inputs.insert(1, vec![1.0, 2.0]);
    inputs.insert(2, vec![3.0, 4.0]);
    inputs
}

type HostResult<T> = Result<T, faber_host_macos_arm64::HostError>;

fn add_inputs(a: Vec<f32>, b: Vec<f32>) -> BTreeMap<u32, Vec<f32>> {
    let mut inputs = BTreeMap::new();
    inputs.insert(1, a);
    inputs.insert(2, b);
    inputs
}

// ---------------------------------------------------------------------------
// Host-construction policy (the one policy across every route)
// ---------------------------------------------------------------------------

#[test]
fn auto_without_device_program_is_cpu_route() {
    let result = resolve_device_selection(DeviceSelection::Auto, false, &[DeviceBackend::Metal]);
    assert_eq!(result.expect("cpu route"), None);
}

#[test]
fn auto_with_device_program_picks_the_single_admitted_backend() {
    let result = resolve_device_selection(DeviceSelection::Auto, true, &[DeviceBackend::Metal]);
    assert_eq!(result.expect("single admitted"), Some(DeviceBackend::Metal));
}

#[test]
fn auto_with_zero_admitted_backends_fails_closed() {
    let err = resolve_device_selection(DeviceSelection::Auto, true, &[])
        .expect_err("zero admitted must fail closed");
    assert_eq!(err.code, E_BACKEND_UNAVAILABLE);
}

#[test]
fn auto_with_multiple_admitted_backends_fails_closed_and_names_candidates() {
    let err = resolve_device_selection(
        DeviceSelection::Auto,
        true,
        &[DeviceBackend::Metal, DeviceBackend::Cuda],
    )
    .expect_err("multiple admitted must fail closed");
    assert_eq!(err.code, E_BACKEND_UNAVAILABLE);
    assert!(err.message.contains("metal"));
    assert!(err.message.contains("cuda"));
    assert!(err.message.contains("--backend"));
}

#[test]
fn explicit_backend_on_payloadless_route_is_rejected() {
    let err = resolve_device_selection(DeviceSelection::Metal, false, &[DeviceBackend::Metal])
        .expect_err("explicit GPU on a payload-less package must fail closed");
    assert_eq!(err.code, E_NO_DEVICE_PROGRAM);
}

#[test]
fn explicit_unavailable_backend_never_silently_falls_back() {
    let err = resolve_device_selection(DeviceSelection::Cuda, true, &[DeviceBackend::Metal])
        .expect_err("explicit cuda with only metal admitted must fail closed");
    assert_eq!(err.code, E_BACKEND_UNAVAILABLE);
}

#[test]
fn explicit_admitted_backend_resolves() {
    let result = resolve_device_selection(DeviceSelection::Metal, true, &[DeviceBackend::Metal]);
    assert_eq!(
        result.expect("explicit admitted"),
        Some(DeviceBackend::Metal)
    );
}

// ---------------------------------------------------------------------------
// Composite composition: stdio + kernel effects still route (A8)
// ---------------------------------------------------------------------------

#[test]
fn composite_host_still_routes_kernel_effects_and_echoes() {
    let host = CompositeHost::new(CompositeHostConfig::cpu()).expect("cpu composite");
    let data = frame_data::tabula([("value", Valor::Textus("salve".into()))]);
    let request = Frame::request_with("host:echo", data);
    let response = host.route(&request);
    assert_eq!(response.status, Status::Done);
    assert_eq!(response.call, "host:echo");
    assert!(!host.manifest().builtins.is_empty());
    assert!(!host.is_device_active());
}

#[test]
fn device_carrying_composite_is_discovery_visible() {
    let host = metal_composite("add_one").expect("metal composite");
    assert!(host.is_device_active());
    assert!(!host.manifest().builtins.is_empty());
}

// ---------------------------------------------------------------------------
// Fake-backend lifecycle sequencing
// ---------------------------------------------------------------------------

#[test]
fn metal_fake_sequences_full_lifecycle_and_receipt() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let receipt = host
        .execute_descriptor(&descriptor, &add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]))
        .expect("execute");

    assert_eq!(receipt.backend, DeviceBackend::Metal);
    assert_eq!(receipt.device_name, "fake-metal-device");
    assert_eq!(receipt.module_hash, fnv1a64(MODULE_IMAGE));
    assert_eq!(receipt.launches, 1);
    assert_eq!(receipt.copy_ins, 2);
    assert_eq!(receipt.outputs.get(&3), Some(&vec![4.0, 6.0]));
    assert_eq!(receipt.allocated_buffers, vec![1, 2, 3]);

    // Ordered teardown: every handle (buffers + module) released.
    let device = host.device().expect("device present");
    assert_eq!(device.live_handle_count(), 0);
}

#[test]
fn program_session_executes_reordered_and_repeated_launches_verbatim() {
    let mut host = multi_kernel_metal_composite().expect("metal composite");
    let descriptor = make_descriptor(
        DeviceBackend::Metal,
        vec![
            DescriptorKernel {
                entry: "kernel_zero".to_owned(),
                buffers: vec![
                    add_slot(1, "a0", DeviceBufferRole::Input, 0, 2),
                    add_slot(2, "b0", DeviceBufferRole::Input, 1, 2),
                    add_slot(3, "out0", DeviceBufferRole::Output, 2, 2),
                ],
                grid: [1, 1, 1],
                block: [2, 1, 1],
            },
            DescriptorKernel {
                entry: "kernel_one".to_owned(),
                buffers: vec![
                    add_slot(4, "a1", DeviceBufferRole::Input, 0, 2),
                    add_slot(5, "b1", DeviceBufferRole::Input, 1, 2),
                    add_slot(6, "out1", DeviceBufferRole::Output, 2, 2),
                ],
                grid: [1, 1, 1],
                block: [2, 1, 1],
            },
        ],
        vec![
            DescriptorLaunch {
                id: 1,
                kernel_index: 1,
            },
            DescriptorLaunch {
                id: 2,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 3,
                kernel_index: 1,
            },
        ],
        Vec::new(),
        // F6: out1 (buffer 6) is the declared observation point, produced by
        // launch 3.
        vec![result(6, 3)],
    );
    let inputs = BTreeMap::from([
        (1, vec![1.0, 2.0]),
        (2, vec![3.0, 4.0]),
        (4, vec![5.0, 6.0]),
        (5, vec![7.0, 8.0]),
    ]);

    let receipt = host
        .execute_descriptor(&descriptor, &inputs)
        .expect("reordered repeated launches");

    assert_eq!(receipt.launches, 3);
    assert_eq!(receipt.launch_ids, vec![1, 2, 3]);
    assert_eq!(
        receipt.launch_entries,
        vec!["kernel_one", "kernel_zero", "kernel_one"]
    );
    assert_eq!(receipt.allocated_buffers, vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(receipt.outputs.get(&6), Some(&vec![12.0, 14.0]));
    assert_eq!(
        host.device().expect("device present").live_handle_count(),
        0
    );
}

#[test]
fn program_session_keeps_same_buffer_versions_separate() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = make_descriptor(
        DeviceBackend::Metal,
        vec![
            DescriptorKernel {
                entry: "add_one".to_owned(),
                buffers: vec![
                    add_slot_version(1, "a", DeviceBufferRole::Input, 0, 2, 1),
                    add_slot_version(2, "b", DeviceBufferRole::Input, 1, 2, 1),
                    add_slot_version(9, "acc", DeviceBufferRole::InOut, 2, 2, 1),
                ],
                grid: [1, 1, 1],
                block: [2, 1, 1],
            },
            DescriptorKernel {
                entry: "add_one".to_owned(),
                buffers: vec![
                    add_slot_version(9, "acc", DeviceBufferRole::InOut, 0, 4, 2),
                    add_slot_version(4, "c", DeviceBufferRole::Input, 1, 4, 1),
                    add_slot_version(5, "out", DeviceBufferRole::Output, 2, 4, 1),
                ],
                grid: [1, 1, 1],
                block: [4, 1, 1],
            },
        ],
        vec![
            DescriptorLaunch {
                id: 21,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 22,
                kernel_index: 1,
            },
            DescriptorLaunch {
                id: 23,
                kernel_index: 1,
            },
        ],
        vec![
            DescriptorDataFlow {
                buffer_id: 9,
                version: 1,
                producer: 21,
                consumer: 22,
            },
            DescriptorDataFlow {
                buffer_id: 9,
                version: 2,
                producer: 22,
                consumer: 23,
            },
        ],
        // F6: out (buffer 5, version 1) is the declared observation point,
        // produced by launch 23.
        vec![DescriptorResult {
            buffer_id: 5,
            version: 1,
            produced_by: 23,
            at_launch: 23,
        }],
    );
    let inputs = BTreeMap::from([
        (1, vec![1.0, 2.0]),
        (2, vec![3.0, 4.0]),
        (4, vec![10.0, 20.0, 30.0, 40.0]),
    ]);

    let mut session = host
        .create_program_session(&descriptor)
        .expect("versioned session");
    assert_eq!(
        session.allocated_buffer_versions(),
        vec![(1, 1), (2, 1), (4, 1), (5, 1), (9, 1), (9, 2)]
    );

    let receipt = session.execute(&inputs).expect("versioned chain execution");
    assert_eq!(receipt.launch_ids, vec![21, 22, 23]);
    assert_eq!(
        receipt.allocated_buffer_versions,
        vec![(1, 1), (2, 1), (4, 1), (5, 1), (9, 1), (9, 2)]
    );
    assert_eq!(receipt.per_step_buffer_versions, vec![(9, 1), (9, 2)]);

    let acc_versions: Vec<_> = receipt
        .resource_graph
        .iter()
        .filter(|buffer| buffer.id == 9)
        .map(|buffer| (buffer.version, buffer.element_count))
        .collect();
    assert_eq!(acc_versions, vec![(1, 2), (2, 4)]);
    assert_eq!(
        receipt.data_flow_edges,
        vec![
            DataFlowEdge {
                buffer_id: 9,
                version: 1,
                producer: 21,
                consumer: 22,
            },
            DataFlowEdge {
                buffer_id: 9,
                version: 2,
                producer: 22,
                consumer: 23,
            },
        ]
    );
    session.teardown().expect("teardown");
    assert_eq!(
        host.device().expect("device present").live_handle_count(),
        0
    );
}

#[test]
fn cuda_fake_sequences_full_lifecycle_and_receipt() {
    let mut host = cuda_composite("addita").expect("cuda composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Cuda, "addita", 3);
    let receipt = host
        .execute_descriptor(
            &descriptor,
            &add_inputs(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]),
        )
        .expect("execute");

    assert_eq!(receipt.backend, DeviceBackend::Cuda);
    assert_eq!(receipt.launches, 1);
    assert_eq!(receipt.outputs.get(&3), Some(&vec![5.0, 7.0, 9.0]));
    let device = host.device().expect("device present");
    assert_eq!(device.live_handle_count(), 0);
}

#[test]
fn inout_buffer_stays_device_resident_across_kernels() {
    // Kernel 1: acc = a + b (acc is InOut). Kernel 2: out = acc + c. The
    // acc buffer is allocated once and never copied back to the host — no
    // host roundtrip per operation (A9).
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = make_descriptor(
        DeviceBackend::Metal,
        vec![
            DescriptorKernel {
                entry: "add_one".to_owned(),
                buffers: vec![
                    add_slot(1, "a", DeviceBufferRole::Input, 0, 2),
                    add_slot(2, "b", DeviceBufferRole::Input, 1, 2),
                    add_slot(3, "acc", DeviceBufferRole::InOut, 2, 2),
                ],
                grid: [1, 1, 1],
                block: [2, 1, 1],
            },
            DescriptorKernel {
                entry: "add_one".to_owned(),
                buffers: vec![
                    add_slot(3, "acc", DeviceBufferRole::InOut, 0, 2),
                    add_slot(4, "c", DeviceBufferRole::Input, 1, 2),
                    add_slot(5, "out", DeviceBufferRole::Output, 2, 2),
                ],
                grid: [1, 1, 1],
                block: [2, 1, 1],
            },
        ],
        vec![
            DescriptorLaunch {
                id: 1,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 2,
                kernel_index: 1,
            },
        ],
        Vec::new(),
        vec![result(5, 2)],
    );
    let mut inputs = BTreeMap::new();
    inputs.insert(1, vec![1.0, 2.0]);
    inputs.insert(2, vec![3.0, 4.0]);
    inputs.insert(4, vec![10.0, 10.0]);

    let receipt = host
        .execute_descriptor(&descriptor, &inputs)
        .expect("two-kernel chain");

    assert_eq!(receipt.launches, 2);
    // Only the three Input slots were copied in; the InOut acc never was.
    assert_eq!(receipt.copy_ins, 3);
    assert_eq!(receipt.outputs.get(&5), Some(&vec![14.0, 16.0]));
    assert_eq!(receipt.allocated_buffers, vec![1, 2, 3, 4, 5]);
    let device = host.device().expect("device present");
    assert_eq!(device.live_handle_count(), 0);
}

// ---------------------------------------------------------------------------
// Negative tests: fail before launch (N1.4)
// ---------------------------------------------------------------------------

#[test]
fn same_buffer_versions_bind_by_key_across_launches() {
    let descriptor = make_descriptor(
        DeviceBackend::Metal,
        vec![
            DescriptorKernel {
                entry: "add_one".to_owned(),
                buffers: vec![add_slot(9, "acc", DeviceBufferRole::InOut, 0, 2)],
                grid: [1, 1, 1],
                block: [2, 1, 1],
            },
            DescriptorKernel {
                entry: "add_one".to_owned(),
                buffers: vec![add_slot_version(9, "acc", DeviceBufferRole::InOut, 0, 4, 2)],
                grid: [1, 1, 1],
                block: [4, 1, 1],
            },
        ],
        vec![
            DescriptorLaunch {
                id: 11,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 12,
                kernel_index: 1,
            },
        ],
        Vec::new(),
        Vec::new(),
    );

    descriptor
        .validate()
        .expect("same-buffer v1/v2 slot chain must bind both keyed versions");
    assert_eq!(
        descriptor
            .kernels
            .iter()
            .flat_map(|kernel| kernel.buffers.iter())
            .map(|slot| (slot.buffer_id, slot.version))
            .collect::<Vec<_>>(),
        vec![(9, 1), (9, 2)]
    );
    assert!(descriptor
        .buffer_versions
        .contains(&DescriptorBufferVersion {
            buffer_id: 9,
            version: 1,
            element_ty: DeviceDataType::F32,
            element_count: 2,
        }));
    assert!(descriptor
        .buffer_versions
        .contains(&DescriptorBufferVersion {
            buffer_id: 9,
            version: 2,
            element_ty: DeviceDataType::F32,
            element_count: 4,
        }));
}

#[test]
fn invalid_launch_reference_fails_before_module_or_driver_launch() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let mut descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    descriptor.launches[0].kernel_index = 1;

    let err = host
        .execute_descriptor(&descriptor, &add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]))
        .expect_err("unknown launch kernel must fail before session creation");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
    assert_eq!(
        host.device().expect("device present").live_handle_count(),
        0
    );
}

#[test]
fn impossible_version_metadata_fails_before_module_or_driver_launch() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let mut descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    descriptor.buffer_versions.push(DescriptorBufferVersion {
        buffer_id: 1,
        version: 1,
        element_ty: DeviceDataType::F32,
        element_count: 4,
    });

    let err = host
        .execute_descriptor(&descriptor, &add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]))
        .expect_err("conflicting version facts must fail before session creation");
    assert_eq!(err.code, E_DEVICE_SHAPE_MISMATCH);
    assert_eq!(
        host.device().expect("device present").live_handle_count(),
        0
    );
}

#[test]
fn descriptor_preserves_repeated_launches_and_version_chain() {
    let mut descriptor = make_descriptor(
        DeviceBackend::Metal,
        vec![DescriptorKernel {
            entry: "add_one".to_owned(),
            buffers: vec![add_slot(9, "acc", DeviceBufferRole::InOut, 0, 2)],
            grid: [1, 1, 1],
            block: [2, 1, 1],
        }],
        vec![
            DescriptorLaunch {
                id: 11,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 12,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 13,
                kernel_index: 0,
            },
        ],
        vec![
            DescriptorDataFlow {
                buffer_id: 9,
                version: 1,
                producer: 11,
                consumer: 12,
            },
            DescriptorDataFlow {
                buffer_id: 9,
                version: 2,
                producer: 12,
                consumer: 13,
            },
        ],
        Vec::new(),
    );
    descriptor.buffer_versions.push(DescriptorBufferVersion {
        buffer_id: 9,
        version: 2,
        element_ty: DeviceDataType::F32,
        element_count: 4,
    });

    descriptor
        .validate()
        .expect("repeated launches and versioned metadata are valid");
    assert_eq!(
        descriptor.launches,
        vec![
            DescriptorLaunch {
                id: 11,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 12,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 13,
                kernel_index: 0,
            },
        ]
    );
    assert!(descriptor
        .buffer_versions
        .contains(&DescriptorBufferVersion {
            buffer_id: 9,
            version: 1,
            element_ty: DeviceDataType::F32,
            element_count: 2,
        }));
    assert!(descriptor
        .buffer_versions
        .contains(&DescriptorBufferVersion {
            buffer_id: 9,
            version: 2,
            element_ty: DeviceDataType::F32,
            element_count: 4,
        }));
    assert_eq!(descriptor.data_flow.len(), 2);
    // F3: the carried graph is acyclic and fully reachable from the declared
    // root (launch 11), so the repeated-launch chain is schedulable.
    descriptor.validate().expect("acyclic graph admits");
}

#[test]
fn slot_without_keyed_version_metadata_fails_before_launch() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let mut descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    descriptor.kernels[0].buffers[0].version = 2;

    let err = host
        .execute_descriptor(&descriptor, &add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]))
        .expect_err("a slot without keyed metadata must fail before launch");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
    assert!(err.message.contains("no keyed metadata"));
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

#[test]
fn unknown_launch_kernel_reference_fails_closed() {
    let mut descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    descriptor.launches[0].kernel_index = 1;
    let err = descriptor
        .validate()
        .expect_err("unknown launch kernel reference must fail closed");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
}

#[test]
fn invalid_launch_identity_fails_closed() {
    let mut zero_id = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    zero_id.launches[0].id = 0;
    let err = zero_id
        .validate()
        .expect_err("zero launch identity must fail closed");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);

    let mut duplicate_id = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    duplicate_id.launches.push(DescriptorLaunch {
        id: 1,
        kernel_index: 0,
    });
    let err = duplicate_id
        .validate()
        .expect_err("duplicate launch identity must fail closed");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
}

#[test]
fn conflicting_version_metadata_fails_closed() {
    let mut descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    descriptor.buffer_versions.push(DescriptorBufferVersion {
        buffer_id: 1,
        version: 1,
        element_ty: DeviceDataType::F32,
        element_count: 4,
    });
    let err = descriptor
        .validate()
        .expect_err("conflicting version metadata must fail closed");
    assert_eq!(err.code, E_DEVICE_SHAPE_MISMATCH);
}

#[test]
fn cpu_only_host_rejects_descriptor_execution() {
    let mut host = CompositeHost::new(CompositeHostConfig::cpu()).expect("cpu composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let err = host
        .execute_descriptor(&descriptor, &BTreeMap::new())
        .expect_err("cpu-only host must refuse device execution");
    assert_eq!(err.code, E_NO_DEVICE_PROGRAM);
}

#[test]
fn wrong_backend_descriptor_fails_closed() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Cuda, "add_one", 2);
    let err = host
        .execute_descriptor(&descriptor, &BTreeMap::new())
        .expect_err("cuda descriptor on a metal session must fail");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
}

#[test]
fn empty_module_image_is_a_bad_descriptor() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let mut descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    descriptor.module_image.clear();
    let err = host
        .execute_descriptor(&descriptor, &BTreeMap::new())
        .expect_err("empty module image must fail before launch");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
}

#[test]
fn descriptor_with_no_kernels_fails_closed() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let mut descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    descriptor.kernels.clear();
    let err = host
        .execute_descriptor(&descriptor, &BTreeMap::new())
        .expect_err("no kernels must fail before launch");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
}

#[test]
fn empty_kernel_entry_fails_closed() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let mut descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    descriptor.kernels[0].entry.clear();
    let err = host
        .execute_descriptor(&descriptor, &BTreeMap::new())
        .expect_err("empty entry must fail before launch");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
}

#[test]
fn zero_grid_axis_fails_closed() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let mut descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    descriptor.kernels[0].grid[1] = 0;
    let err = host
        .execute_descriptor(&descriptor, &BTreeMap::new())
        .expect_err("zero grid must fail before launch");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
}

#[test]
fn duplicate_binding_fails_as_abi_mismatch() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let mut descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    descriptor.kernels[0].buffers[2].binding = 0; // collides with slot `a`
    let err = host
        .execute_descriptor(&descriptor, &BTreeMap::new())
        .expect_err("duplicate binding must fail as an ABI mismatch");
    assert_eq!(err.code, E_DEVICE_ABI_MISMATCH);
}

#[test]
fn conflicting_buffer_roles_fail_as_abi_mismatch() {
    let mut host = metal_composite("add_one").expect("metal composite");
    // The same buffer id is Input in one kernel and Output in another.
    let descriptor = make_descriptor(
        DeviceBackend::Metal,
        vec![
            DescriptorKernel {
                entry: "add_one".to_owned(),
                buffers: vec![
                    add_slot(1, "x", DeviceBufferRole::Input, 0, 2),
                    add_slot(2, "y", DeviceBufferRole::Input, 1, 2),
                    add_slot(3, "z", DeviceBufferRole::Output, 2, 2),
                ],
                grid: [1, 1, 1],
                block: [2, 1, 1],
            },
            DescriptorKernel {
                entry: "add_one".to_owned(),
                buffers: vec![
                    add_slot(1, "x", DeviceBufferRole::Output, 0, 2),
                    add_slot(3, "z", DeviceBufferRole::Input, 1, 2),
                    add_slot(4, "w", DeviceBufferRole::Output, 2, 2),
                ],
                grid: [1, 1, 1],
                block: [2, 1, 1],
            },
        ],
        vec![
            DescriptorLaunch {
                id: 1,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 2,
                kernel_index: 1,
            },
        ],
        Vec::new(),
        Vec::new(),
    );
    let err = host
        .execute_descriptor(&descriptor, &BTreeMap::new())
        .expect_err("input/output role conflict must fail as an ABI mismatch");
    assert_eq!(err.code, E_DEVICE_ABI_MISMATCH);
}

#[test]
fn conflicting_dtypes_fail_as_dtype_mismatch() {
    let mut host = metal_composite("add_one").expect("metal composite");
    // Two kernels reference buffer id 3 with the same count but different
    // element types: a dtype conflict must fail before launch.
    let mut descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let mut first = add_slot(1, "a", DeviceBufferRole::Input, 0, 2);
    first.element_ty = DeviceDataType::F32;
    descriptor.kernels[0].buffers = vec![
        first,
        add_slot(2, "b", DeviceBufferRole::Input, 1, 2),
        add_slot(3, "x", DeviceBufferRole::InOut, 2, 2),
    ];
    descriptor.kernels.push(DescriptorKernel {
        entry: "add_one".to_owned(),
        buffers: vec![
            add_slot(3, "x", DeviceBufferRole::InOut, 0, 2),
            add_slot(4, "c", DeviceBufferRole::Input, 1, 2),
            add_slot(5, "out", DeviceBufferRole::Output, 2, 2),
        ],
        grid: [1, 1, 1],
        block: [2, 1, 1],
    });
    // Relabel buffer 3 as i32 in the second kernel (same count → pure dtype
    // conflict, no shape conflict).
    descriptor.kernels[1].buffers[0].element_ty = DeviceDataType::I32;
    let err = host
        .execute_descriptor(&descriptor, &BTreeMap::new())
        .expect_err("dtype conflict must fail before launch");
    assert_eq!(err.code, E_DEVICE_DTYPE_MISMATCH);
}

#[test]
fn conflicting_shapes_fail_as_shape_mismatch() {
    let mut host = metal_composite("add_one").expect("metal composite");
    // Two kernels reference buffer id 3 with different element counts: a
    // shape change must be a new version, never in-place reinterpretation.
    let descriptor = make_descriptor(
        DeviceBackend::Metal,
        vec![
            DescriptorKernel {
                entry: "add_one".to_owned(),
                buffers: vec![
                    add_slot(1, "a", DeviceBufferRole::Input, 0, 2),
                    add_slot(2, "b", DeviceBufferRole::Input, 1, 2),
                    add_slot(3, "x", DeviceBufferRole::InOut, 2, 2),
                ],
                grid: [1, 1, 1],
                block: [2, 1, 1],
            },
            DescriptorKernel {
                entry: "add_one".to_owned(),
                buffers: vec![
                    add_slot(3, "x", DeviceBufferRole::InOut, 0, 4), // conflict
                    add_slot(4, "c", DeviceBufferRole::Input, 1, 4),
                    add_slot(5, "out", DeviceBufferRole::Output, 2, 4),
                ],
                grid: [1, 1, 1],
                block: [4, 1, 1],
            },
        ],
        vec![
            DescriptorLaunch {
                id: 1,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 2,
                kernel_index: 1,
            },
        ],
        Vec::new(),
        Vec::new(),
    );
    let err = host
        .execute_descriptor(&descriptor, &BTreeMap::new())
        .expect_err("shape conflict must fail before launch");
    assert_eq!(err.code, E_DEVICE_SHAPE_MISMATCH);
}

#[test]
fn unknown_kernel_entry_fails_before_launch() {
    // The fake declares only `add_one`; the descriptor asks for `add_two`.
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_two", 2);
    let err = host
        .execute_descriptor(&descriptor, &add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]))
        .expect_err("unknown entry must fail before launch");
    assert_eq!(err.code, E_DEVICE_ENTRY_MISMATCH);
    // The failed execution released every handle (S2-3 release-on-error).
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

#[test]
fn missing_input_fails_before_launch() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let mut inputs = BTreeMap::new();
    inputs.insert(1, vec![1.0, 2.0]); // buffer 2 missing
    let err = host
        .execute_descriptor(&descriptor, &inputs)
        .expect_err("missing declared input must fail before launch");
    assert_eq!(err.code, E_DEVICE_SHAPE_MISMATCH);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

#[test]
fn input_size_mismatch_fails_before_launch() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let err = host
        .execute_descriptor(
            &descriptor,
            &add_inputs(vec![1.0, 2.0, 3.0], vec![4.0, 5.0]),
        )
        .expect_err("input size vs declared shape must fail before launch");
    assert_eq!(err.code, E_DEVICE_SHAPE_MISMATCH);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

#[test]
fn undeclared_observation_point_fails_closed() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);

    // A result naming a buffer no kernel slot allocates is an undeclared
    // observation and fails host admission before launch.
    let mut phantom = descriptor.clone();
    phantom.results = vec![DescriptorResult {
        buffer_id: 99,
        version: 1,
        produced_by: 1,
        at_launch: 1,
    }];
    let err = host
        .create_program_session(&phantom)
        .err()
        .expect("a result for an unallocated buffer must fail admission");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
    assert!(err.message.contains("no kernel slot allocates"));

    // A writable intermediate (PerStep InOut) exposed as a result without an
    // explicit observation fact is the F6 observation mismatch: the host
    // rejects it exactly as the constructor would (constructor + host
    // admission agree).
    let descriptor = two_kernel_inout_descriptor(DeviceBackend::Metal);
    let mut intermediate_result = descriptor.clone();
    intermediate_result.results = vec![DescriptorResult {
        buffer_id: 3, // acc — a PerStep InOut intermediate
        version: 1,
        produced_by: 1,
        at_launch: 1,
    }];
    let err = host
        .create_program_session(&intermediate_result)
        .err()
        .expect("a writable intermediate as a result must fail host admission");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
    assert!(err.message.contains("observation-point"));

    // The valid descriptor reads back exactly its declared observation
    // points — nothing else is observable.
    let mut host = metal_composite("add_one").expect("metal composite");
    let receipt = host
        .execute_descriptor(&descriptor, &two_kernel_inputs())
        .expect("declared observations only");
    assert_eq!(receipt.outputs.len(), 1);
    assert_eq!(receipt.readbacks, 1);
    assert!(receipt.outputs.contains_key(&5));
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

#[test]
fn cross_backend_handle_use_fails_closed() {
    let mut runtime = DeviceRuntime::Metal(
        MetalHostSession::with_driver(Box::new(FakeMetalDriver::default()))
            .expect("fake metal admit"),
    );
    let metal_buffer = runtime.alloc_bytes(8).expect("alloc on metal");
    // The same opaque id relabeled as a CUDA handle must fail closed on the
    // Metal session — handles never cross backends.
    let cuda_relabel = DeviceHandle {
        backend: DeviceBackend::Cuda,
        kind: DeviceHandleKind::Buffer { len_bytes: 8 },
        id: metal_buffer.id,
    };
    let err = runtime
        .readback_f32(&cuda_relabel)
        .expect_err("cross-backend handle must fail closed");
    assert_eq!(err.code, E_DEVICE_INVALID_HANDLE);
    runtime.release(&metal_buffer).expect("release");
}

// ---------------------------------------------------------------------------
// S2-1: program session sequencing (create → run → run again → teardown)
// ---------------------------------------------------------------------------

/// The session loads the module once and allocates every PerProgram buffer
/// once at creation; repeated `execute` calls on the same session do NOT
/// reload the module or re-allocate PerProgram buffers; the ObservationPoint
/// output is allocated per execution, read back, and released (S2-4);
/// teardown releases every handle.
#[test]
fn program_session_executes_repeatedly_without_reload_or_realloc() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let inputs = add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]);

    // Create: module loaded once (1) + 2 PerProgram buffers (a, b) = 3
    // handles. The ObservationPoint output is NOT allocated at creation — it
    // is allocated per execution and read-then-released (S2-4).
    let mut session = host
        .create_program_session(&descriptor)
        .expect("session create");
    assert_eq!(session.session_handle_count(), 3);
    assert_eq!(session.module_hash(), fnv1a64(MODULE_IMAGE));
    assert_eq!(session.allocated_buffers(), vec![1, 2, 3]);

    // Execute 1: no new persistent handles (module + PerProgram reused; the
    // observation buffer is allocated, read back, released).
    let receipt1 = session.execute(&inputs).expect("execute 1");
    assert_eq!(receipt1.launches, 1);
    assert_eq!(receipt1.copy_ins, 2);
    assert_eq!(receipt1.outputs.get(&3), Some(&vec![4.0, 6.0]));
    assert_eq!(session.session_handle_count(), 3); // unchanged

    // Execute 2 on the SAME session: no reload, no PerProgram realloc.
    let receipt2 = session
        .execute(&add_inputs(vec![10.0, 20.0], vec![30.0, 40.0]))
        .expect("execute 2");
    assert_eq!(receipt2.outputs.get(&3), Some(&vec![40.0, 60.0]));
    assert_eq!(session.session_handle_count(), 3); // still unchanged

    // Teardown: ordered release (buffers then module); every handle gone.
    session.teardown().expect("teardown");
    let device = host.device().expect("device present");
    assert_eq!(device.live_handle_count(), 0);
}

/// The session proves repeated execution on CUDA too (backend-neutral surface).
#[test]
fn program_session_repeated_execution_cuda() {
    let mut host = cuda_composite("addita").expect("cuda composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Cuda, "addita", 3);

    let mut session = host
        .create_program_session(&descriptor)
        .expect("session create");
    assert_eq!(session.session_handle_count(), 3); // module + 2 PerProgram

    let r1 = session
        .execute(&add_inputs(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]))
        .expect("execute 1");
    assert_eq!(r1.outputs.get(&3), Some(&vec![5.0, 7.0, 9.0]));
    assert_eq!(session.session_handle_count(), 3);

    let r2 = session
        .execute(&add_inputs(vec![10.0, 20.0, 30.0], vec![1.0, 2.0, 3.0]))
        .expect("execute 2");
    assert_eq!(r2.outputs.get(&3), Some(&vec![11.0, 22.0, 33.0]));
    assert_eq!(session.session_handle_count(), 3);

    session.teardown().expect("teardown");
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

/// A two-kernel InOut chain through the session API: the intermediate buffer
/// stays device-resident across kernels within one step, and the session
/// reuses it across steps without re-allocation.
#[test]
fn program_session_two_kernel_chain_reuses_buffers_across_steps() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = make_descriptor(
        DeviceBackend::Metal,
        vec![
            DescriptorKernel {
                entry: "add_one".to_owned(),
                buffers: vec![
                    add_slot(1, "a", DeviceBufferRole::Input, 0, 2),
                    add_slot(2, "b", DeviceBufferRole::Input, 1, 2),
                    add_slot(3, "acc", DeviceBufferRole::InOut, 2, 2),
                ],
                grid: [1, 1, 1],
                block: [2, 1, 1],
            },
            DescriptorKernel {
                entry: "add_one".to_owned(),
                buffers: vec![
                    add_slot(3, "acc", DeviceBufferRole::InOut, 0, 2),
                    add_slot(4, "c", DeviceBufferRole::Input, 1, 2),
                    add_slot(5, "out", DeviceBufferRole::Output, 2, 2),
                ],
                grid: [1, 1, 1],
                block: [2, 1, 1],
            },
        ],
        vec![
            DescriptorLaunch {
                id: 1,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 2,
                kernel_index: 1,
            },
        ],
        Vec::new(),
        vec![result(5, 2)],
    );
    let mut inputs = BTreeMap::new();
    inputs.insert(1, vec![1.0, 2.0]);
    inputs.insert(2, vec![3.0, 4.0]);
    inputs.insert(4, vec![10.0, 10.0]);

    let mut session = host
        .create_program_session(&descriptor)
        .expect("session create");
    // Module + 3 PerProgram buffers (ids 1, 2, 4). The InOut intermediate
    // (id 3, PerStep) and the ObservationPoint output (id 5) are allocated
    // per execution and released at the step boundary (S2-4).
    assert_eq!(session.session_handle_count(), 4);
    assert_eq!(session.allocated_buffers(), vec![1, 2, 3, 4, 5]);

    let receipt = session.execute(&inputs).expect("execute");
    assert_eq!(receipt.launches, 2);
    assert_eq!(receipt.copy_ins, 3); // only Input slots (a, b, c)
    assert_eq!(receipt.outputs.get(&5), Some(&vec![14.0, 16.0]));
    assert_eq!(session.session_handle_count(), 4); // per-step + observation released

    // Second step: same session, same PerProgram buffers, fresh per-step /
    // observation allocations (recycled at the step boundary).
    let receipt2 = session.execute(&inputs).expect("execute 2");
    assert_eq!(receipt2.outputs.get(&5), Some(&vec![14.0, 16.0]));
    assert_eq!(session.session_handle_count(), 4);

    session.teardown().expect("teardown");
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

/// `execute_descriptor` remains a single-run convenience over the session:
/// create → execute → teardown in one call, same receipt shape as S1-4.
#[test]
fn execute_descriptor_is_single_run_convenience_over_session() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let receipt = host
        .execute_descriptor(&descriptor, &add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]))
        .expect("execute_descriptor");

    assert_eq!(receipt.backend, DeviceBackend::Metal);
    assert_eq!(receipt.module_hash, fnv1a64(MODULE_IMAGE));
    assert_eq!(receipt.launches, 1);
    assert_eq!(receipt.copy_ins, 2);
    assert_eq!(receipt.outputs.get(&3), Some(&vec![4.0, 6.0]));
    assert_eq!(receipt.allocated_buffers, vec![1, 2, 3]);
    // Single-run convenience tears down internally.
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

/// A CPU-only host cannot create a program session (same fail-closed surface
/// as execute_descriptor).
#[test]
fn cpu_only_host_rejects_program_session_creation() {
    let mut host = CompositeHost::new(CompositeHostConfig::cpu()).expect("cpu composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let err = host
        .create_program_session(&descriptor)
        .err()
        .expect("cpu-only host must refuse session creation");
    assert_eq!(err.code, E_NO_DEVICE_PROGRAM);
}

/// Session creation validates the descriptor before any allocation
/// (fail-before-launch surface preserved through the session API).
#[test]
fn program_session_creation_validates_descriptor_before_launch() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let mut descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    descriptor.kernels.clear();
    let err = host
        .create_program_session(&descriptor)
        .err()
        .expect("empty kernels must fail before session creation");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
}

// ---------------------------------------------------------------------------
// S2-2: module cache at the leak-free bar (council 2)
// ---------------------------------------------------------------------------

/// The session loads the module exactly once (module load = 1) even across N
/// repeated executions; teardown releases it exactly once (module release =
/// 1); nothing persists past teardown (loads == releases, live handles == 0).
/// This is the S2-2 leak-free bar: repeated execution does not leak — not
/// "module persists across steps". With the S2-4 lifetime policy the two
/// PerProgram input buffers are allocated once at creation while the
/// ObservationPoint output is allocated/read/released per execution, so the
/// buffer counters climb exactly one alloc + one release per execution and
/// still return to balance at teardown.
#[test]
fn module_cache_loads_once_and_releases_on_teardown() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let inputs = add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]);

    let mut session = host
        .create_program_session(&descriptor)
        .expect("session create");

    // Session creation loaded the module once and allocated the two PerProgram
    // inputs (a, b); the ObservationPoint output is allocated per execution.
    let counters = session.driver_counters();
    assert_eq!(counters.module_loads, 1);
    assert_eq!(counters.buffer_allocs, 2);
    // One live module: loaded, not yet released — the only persistence the
    // policy allows is the session keeping the module alive for the program.
    assert_eq!(counters.module_loads - counters.module_releases, 1);

    // N repeated executions: still one load, no reload; each execution
    // allocates the ObservationPoint output, reads it back, and releases it
    // (read-then-release, S2-4) — exactly one alloc + one release per run.
    for step in 0..5usize {
        let receipt = session.execute(&inputs).expect("execute");
        assert_eq!(receipt.outputs.get(&3), Some(&vec![4.0, 6.0]));
        let counters = session.driver_counters();
        assert_eq!(counters.module_loads, 1);
        assert_eq!(counters.buffer_allocs, 2 + step + 1);
        assert_eq!(counters.buffer_releases, step + 1);
    }

    // Teardown releases the module exactly once and the PerProgram buffers.
    session.teardown().expect("teardown");
    let counters = host.device().expect("device").driver_counters();
    assert_eq!(counters.module_loads, 1);
    assert_eq!(counters.module_releases, 1);
    assert_eq!(counters.buffer_allocs, 2 + 5);
    assert_eq!(counters.buffer_releases, 2 + 5);
    // Nothing persists past teardown: no live module, no live handles.
    assert_eq!(counters.module_loads, counters.module_releases);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

/// The same leak-free bar on CUDA (backend-neutral surface): one module
/// load, one release, no module persists past teardown; the ObservationPoint
/// output is read-then-released per execution.
#[test]
fn module_cache_loads_once_and_releases_on_teardown_cuda() {
    let mut host = cuda_composite("addita").expect("cuda composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Cuda, "addita", 3);
    let inputs = add_inputs(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]);

    let mut session = host
        .create_program_session(&descriptor)
        .expect("session create");
    assert_eq!(session.driver_counters().module_loads, 1);
    assert_eq!(session.driver_counters().buffer_allocs, 2);

    for step in 0..5usize {
        let receipt = session.execute(&inputs).expect("execute");
        assert_eq!(receipt.outputs.get(&3), Some(&vec![5.0, 7.0, 9.0]));
        let counters = session.driver_counters();
        assert_eq!(counters.module_loads, 1);
        assert_eq!(counters.buffer_allocs, 2 + step + 1);
        assert_eq!(counters.buffer_releases, step + 1);
    }

    session.teardown().expect("teardown");
    let counters = host.device().expect("device").driver_counters();
    assert_eq!(counters.module_loads, 1);
    assert_eq!(counters.module_releases, 1);
    assert_eq!(counters.module_loads, counters.module_releases);
    assert_eq!(counters.buffer_allocs, 2 + 5);
    assert_eq!(counters.buffer_releases, 2 + 5);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

/// A second program session re-loads the module independently: there is no
/// cross-session cache, so the same provenance-hash image is loaded again by
/// session 2 (module load = 2) and released again at its teardown (module
/// release = 2). No module persists past either teardown.
#[test]
fn second_session_reloads_module_independently() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let inputs = add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]);

    // Session 1: load once, execute, teardown releases.
    let mut session1 = host
        .create_program_session(&descriptor)
        .expect("session 1 create");
    assert_eq!(session1.module_hash(), fnv1a64(MODULE_IMAGE));
    assert_eq!(session1.driver_counters().module_loads, 1);
    session1.execute(&inputs).expect("session 1 execute");
    assert_eq!(session1.driver_counters().module_loads, 1);
    session1.teardown().expect("session 1 teardown");
    let counters = host.device().expect("device").driver_counters();
    assert_eq!(counters.module_loads, 1);
    assert_eq!(counters.module_releases, 1);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);

    // Session 2 on the same host, same image hash: re-loads independently.
    let mut session2 = host
        .create_program_session(&descriptor)
        .expect("session 2 create");
    assert_eq!(session2.module_hash(), fnv1a64(MODULE_IMAGE));
    assert_eq!(session2.driver_counters().module_loads, 2);
    session2.execute(&inputs).expect("session 2 execute");
    assert_eq!(session2.driver_counters().module_loads, 2);
    session2.teardown().expect("session 2 teardown");
    let counters = host.device().expect("device").driver_counters();
    assert_eq!(counters.module_loads, 2);
    assert_eq!(counters.module_releases, 2);
    assert_eq!(counters.module_loads, counters.module_releases);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

/// The S1-4 single-run convenience (`execute_descriptor`) also hits the
/// leak-free bar: one load, one release, nothing persists.
#[test]
fn execute_descriptor_single_run_releases_module() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    host.execute_descriptor(&descriptor, &add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]))
        .expect("execute_descriptor");
    let counters = host.device().expect("device").driver_counters();
    assert_eq!(counters.module_loads, 1);
    assert_eq!(counters.module_releases, 1);
    assert_eq!(counters.buffer_allocs, 3);
    assert_eq!(counters.buffer_releases, 3);
    assert_eq!(counters.module_loads, counters.module_releases);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

// ---------------------------------------------------------------------------
// S2-3: error-path teardown (P2-1) — a failed execution at ANY stage leaves
// live_handle_count() == 0 (release-on-error designed into the session)
// ---------------------------------------------------------------------------

// Creation-stage failures (module load, allocation). The fake driver injects
// the typed driver error; the session's creation guard releases the module
// and any partially allocated buffers before the error escapes.

#[test]
fn metal_module_load_failure_releases_every_handle() {
    let mut host = metal_composite_failing("add_one", FakeFailureStage::ModuleLoad, 1)
        .expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let err = host
        .create_program_session(&descriptor)
        .err()
        .expect("injected module-load failure must fail session creation");
    assert_eq!(err.code, E_METAL_DRIVER);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

#[test]
fn metal_allocation_failure_releases_module_and_partial_buffers() {
    // The module and the first buffer are already registered when the second
    // allocation fails; the creation guard must release both (P2-1).
    let mut host =
        metal_composite_failing("add_one", FakeFailureStage::Alloc, 2).expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let err = host
        .create_program_session(&descriptor)
        .err()
        .expect("injected allocation failure must fail session creation");
    assert_eq!(err.code, E_METAL_DRIVER);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

#[test]
fn cuda_module_load_failure_releases_every_handle() {
    let mut host =
        cuda_composite_failing("addita", FakeFailureStage::ModuleLoad, 1).expect("cuda composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Cuda, "addita", 3);
    let err = host
        .create_program_session(&descriptor)
        .err()
        .expect("injected module-load failure must fail session creation");
    assert_eq!(err.code, E_CUDA_DRIVER);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

#[test]
fn cuda_allocation_failure_releases_module_and_partial_buffers() {
    let mut host =
        cuda_composite_failing("addita", FakeFailureStage::Alloc, 2).expect("cuda composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Cuda, "addita", 3);
    let err = host
        .create_program_session(&descriptor)
        .err()
        .expect("injected allocation failure must fail session creation");
    assert_eq!(err.code, E_CUDA_DRIVER);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

// Execution-stage failures (copy-in, launch, sync, readback). Each test runs
// the full single-kernel program through `execute_descriptor` with the driver
// failing exactly one stage; the session's release-on-error must release the
// module + every buffer before the typed error escapes.

#[test]
fn metal_copy_in_failure_releases_every_handle() {
    let mut host =
        metal_composite_failing("add_one", FakeFailureStage::CopyIn, 1).expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let err = host
        .execute_descriptor(&descriptor, &add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]))
        .expect_err("injected copy-in failure must fail the execution");
    assert_eq!(err.code, E_METAL_DRIVER);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

#[test]
fn metal_launch_failure_releases_every_handle() {
    let mut host =
        metal_composite_failing("add_one", FakeFailureStage::Launch, 1).expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let err = host
        .execute_descriptor(&descriptor, &add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]))
        .expect_err("injected launch failure must fail the execution");
    assert_eq!(err.code, E_METAL_DRIVER);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

#[test]
fn metal_sync_failure_releases_every_handle() {
    // The driver syncs once inside each launch and once at the step boundary;
    // sync call 2 is the explicit step-boundary barrier in `execute`.
    let mut host =
        metal_composite_failing("add_one", FakeFailureStage::Sync, 2).expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let err = host
        .execute_descriptor(&descriptor, &add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]))
        .expect_err("injected sync failure must fail the execution");
    assert_eq!(err.code, E_METAL_DRIVER);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

#[test]
fn metal_readback_failure_releases_every_handle() {
    let mut host =
        metal_composite_failing("add_one", FakeFailureStage::Readback, 1).expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let err = host
        .execute_descriptor(&descriptor, &add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]))
        .expect_err("injected readback failure must fail the execution");
    assert_eq!(err.code, E_METAL_DRIVER);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

#[test]
fn cuda_copy_in_failure_releases_every_handle() {
    let mut host =
        cuda_composite_failing("addita", FakeFailureStage::CopyIn, 1).expect("cuda composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Cuda, "addita", 3);
    let err = host
        .execute_descriptor(
            &descriptor,
            &add_inputs(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]),
        )
        .expect_err("injected copy-in failure must fail the execution");
    assert_eq!(err.code, E_CUDA_DRIVER);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

#[test]
fn cuda_launch_failure_releases_every_handle() {
    let mut host =
        cuda_composite_failing("addita", FakeFailureStage::Launch, 1).expect("cuda composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Cuda, "addita", 3);
    let err = host
        .execute_descriptor(
            &descriptor,
            &add_inputs(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]),
        )
        .expect_err("injected launch failure must fail the execution");
    assert_eq!(err.code, E_CUDA_DRIVER);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

#[test]
fn cuda_sync_failure_releases_every_handle() {
    // Same call sequence as the Metal lane: sync call 2 is the step boundary.
    let mut host =
        cuda_composite_failing("addita", FakeFailureStage::Sync, 2).expect("cuda composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Cuda, "addita", 3);
    let err = host
        .execute_descriptor(
            &descriptor,
            &add_inputs(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]),
        )
        .expect_err("injected sync failure must fail the execution");
    assert_eq!(err.code, E_CUDA_DRIVER);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

#[test]
fn cuda_readback_failure_releases_every_handle() {
    let mut host =
        cuda_composite_failing("addita", FakeFailureStage::Readback, 1).expect("cuda composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Cuda, "addita", 3);
    let err = host
        .execute_descriptor(
            &descriptor,
            &add_inputs(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]),
        )
        .expect_err("injected readback failure must fail the execution");
    assert_eq!(err.code, E_CUDA_DRIVER);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

// Mid-chain failure: kernel 1 fully succeeds, then the second launch fails.
// This is the exact P2-1 shape — Stage 1's `execute_descriptor` leaked the
// module + every buffer on a mid-chain `?`-return. The session's
// release-on-error must release everything.

#[test]
fn mid_chain_second_launch_failure_releases_every_handle() {
    let mut host =
        metal_composite_failing("add_one", FakeFailureStage::Launch, 2).expect("metal composite");
    let descriptor = two_kernel_inout_descriptor(DeviceBackend::Metal);
    let err = host
        .execute_descriptor(&descriptor, &two_kernel_inputs())
        .expect_err("injected second-launch failure must fail the execution");
    assert_eq!(err.code, E_METAL_DRIVER);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

/// Mid-chain failure on the fake CUDA driver, mirroring the Metal lane:
/// kernel 1 fully succeeds, then the second launch fails via
/// `with_failure_at(Launch, 2)`. The typed `E_CUDA_DRIVER` error escapes and
/// release-on-error leaves zero live handles — CUDA proves the same
/// mid-chain shape, not just single-launch failure stages.
#[test]
fn cuda_mid_chain_second_launch_failure_releases_every_handle() {
    let mut host =
        cuda_composite_failing("add_one", FakeFailureStage::Launch, 2).expect("cuda composite");
    let descriptor = two_kernel_inout_descriptor(DeviceBackend::Cuda);
    let err = host
        .execute_descriptor(&descriptor, &two_kernel_inputs())
        .expect_err("injected second-launch failure must fail the execution");
    assert_eq!(err.code, E_CUDA_DRIVER);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

/// The two-kernel InOut chain executes end-to-end on the fake CUDA driver:
/// kernel 1 writes the device-resident intermediate (id 3), kernel 2 reads
/// it, and the observation output (id 5) comes back as `(a+b)+c = 14/16`.
/// This proves the mid-chain shape (two ordered launches + step-boundary
/// sync + observation readback) succeeds on CUDA, not just Metal.
#[test]
fn cuda_two_kernel_chain_executes_end_to_end() {
    let mut host = cuda_composite("add_one").expect("cuda composite");
    let descriptor = two_kernel_inout_descriptor(DeviceBackend::Cuda);
    let receipt = host
        .execute_descriptor(&descriptor, &two_kernel_inputs())
        .expect("two-kernel chain must execute");
    assert_eq!(receipt.backend, DeviceBackend::Cuda);
    assert_eq!(receipt.launches, 2);
    assert_eq!(receipt.copy_ins, 3); // a, b at kernel 1; c at kernel 2
    assert_eq!(receipt.outputs.get(&5), Some(&vec![14.0, 16.0]));
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

/// A failed execution closes the session: every handle is released, the
/// session reports 0 handles, and a closed session refuses further execution
/// instead of reusing stale handles.
#[test]
fn failed_execution_closes_session_and_blocks_reuse() {
    let mut host =
        metal_composite_failing("add_one", FakeFailureStage::CopyIn, 1).expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let inputs = add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]);

    let mut session = host
        .create_program_session(&descriptor)
        .expect("session create");
    assert_eq!(session.session_handle_count(), 3); // module + 2 PerProgram

    let err = session
        .execute(&inputs)
        .expect_err("injected copy-in failure must fail the execution");
    assert_eq!(err.code, E_METAL_DRIVER);
    // Release-on-error: no handle survives the failed execution at the
    // session level (the host-level count is checked after the session is
    // consumed, once the session's borrow of the host has ended).
    assert_eq!(session.session_handle_count(), 0);

    // A closed session cannot execute again (no stale-handle reuse).
    let again = session
        .execute(&inputs)
        .expect_err("a closed session must refuse execution");
    assert_eq!(again.code, "E_INTERNAL");

    // Teardown of an already-closed session is a safe no-op (no double
    // release).
    session.teardown().expect("teardown of closed session");
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

// ---------------------------------------------------------------------------
// S2-4: BufferLifetime host-interpretation (council 3)
// ---------------------------------------------------------------------------

/// The two-lifetime fixture from the S2-4 done-when: a PerProgram input is
/// allocated once at session creation and persists across executions (no
/// realloc), while an ObservationPoint output is allocated, read back, and
/// released on every execution (read-then-release). The fake-driver counters
/// make the lifetime-distinct allocation/release events observable.
#[test]
fn per_program_persists_while_observation_read_then_releases() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let inputs = add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]);

    let mut session = host
        .create_program_session(&descriptor)
        .expect("session create");
    // Creation allocates only the two PerProgram inputs; the observation
    // output (id 3) is not allocated until an execution.
    let counters = session.driver_counters();
    assert_eq!(counters.buffer_allocs, 2);

    // Execute 1: observation buffer allocated (allocs 3), read back, released
    // (releases 1). PerProgram inputs stay live; no realloc.
    let receipt = session.execute(&inputs).expect("execute 1");
    assert_eq!(receipt.outputs.get(&3), Some(&vec![4.0, 6.0]));
    assert_eq!(receipt.per_program_buffers, vec![1, 2]);
    assert_eq!(receipt.observation_buffers, vec![3]);
    assert!(receipt.per_step_buffers.is_empty());
    let counters = session.driver_counters();
    assert_eq!(counters.buffer_allocs, 3);
    assert_eq!(counters.buffer_releases, 1);
    assert_eq!(session.session_handle_count(), 3); // module + 2 PerProgram

    // Execute 2: the observation buffer is re-allocated, read back, released
    // again; the PerProgram inputs were never re-allocated (allocs stay 4).
    let receipt2 = session
        .execute(&add_inputs(vec![10.0, 20.0], vec![30.0, 40.0]))
        .expect("execute 2");
    assert_eq!(receipt2.outputs.get(&3), Some(&vec![40.0, 60.0]));
    let counters = session.driver_counters();
    assert_eq!(counters.buffer_allocs, 4);
    assert_eq!(counters.buffer_releases, 2);

    // Teardown releases the PerProgram inputs; balance returns to zero.
    session.teardown().expect("teardown");
    let counters = host.device().expect("device").driver_counters();
    assert_eq!(counters.buffer_allocs, 4);
    assert_eq!(counters.buffer_releases, 4);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

/// PerStep buffers are recycled at the step boundary: allocated per
/// execution, released at the end of each execution (so a second execution
/// re-allocates them), and never persist past teardown. The InOut
/// intermediate (id 3) of the two-kernel chain is the PerStep buffer.
#[test]
fn per_step_buffers_recycle_at_step_boundary() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = two_kernel_inout_descriptor(DeviceBackend::Metal);
    let inputs = two_kernel_inputs();

    let mut session = host
        .create_program_session(&descriptor)
        .expect("session create");
    // Creation allocates the three PerProgram inputs (ids 1, 2, 4).
    assert_eq!(session.driver_counters().buffer_allocs, 3);
    assert_eq!(session.session_handle_count(), 4); // module + 3 PerProgram

    // Execute 1 allocates the PerStep intermediate (id 3) and the
    // ObservationPoint output (id 5), then releases both at the step
    // boundary / after readback: allocs 3 → 5, releases 0 → 2.
    let receipt = session.execute(&inputs).expect("execute 1");
    assert_eq!(receipt.outputs.get(&5), Some(&vec![14.0, 16.0]));
    assert_eq!(receipt.per_program_buffers, vec![1, 2, 4]);
    assert_eq!(receipt.per_step_buffers, vec![3]);
    assert_eq!(receipt.observation_buffers, vec![5]);
    let counters = session.driver_counters();
    assert_eq!(counters.buffer_allocs, 5);
    assert_eq!(counters.buffer_releases, 2);

    // Execute 2: the PerStep intermediate and the observation output are
    // allocated fresh and released again — recycled at each step boundary.
    let receipt2 = session.execute(&inputs).expect("execute 2");
    assert_eq!(receipt2.outputs.get(&5), Some(&vec![14.0, 16.0]));
    let counters = session.driver_counters();
    assert_eq!(counters.buffer_allocs, 7);
    assert_eq!(counters.buffer_releases, 4);

    // Teardown releases the three PerProgram inputs; every allocation is
    // released (no leak) and the PerStep pool is gone.
    session.teardown().expect("teardown");
    let counters = host.device().expect("device").driver_counters();
    assert_eq!(counters.buffer_allocs, 7);
    assert_eq!(counters.buffer_releases, 7);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

// ---------------------------------------------------------------------------
// S3-B3: Mul+Mean companion classification — the fake-driver suite reuses
// the S2-4 lifetime policy against the S3-B1 companion fixture shape.
// PerProgram once, PerStep recycled, ObservationPoint read-then-released.
// ---------------------------------------------------------------------------

/// The S3-B3 classified-buffer policy over the Mul+Mean companion program:
/// the session allocates the PerProgram inputs (x, w) exactly once at
/// creation, recycles the PerStep intermediates (product, partial, acc) at
/// the step boundary, and read-then-releases the ObservationPoint tuple
/// gradient outputs (grad_x, grad_w). Same-shaped grad_x/grad_w stay
/// distinct buffer ids (6 vs 7) in the session's declared resource graph.
#[test]
fn mul_mean_companion_classified_buffers_follow_lifetime_policy() {
    let mut host = mul_mean_metal_composite().expect("metal composite");
    let descriptor = mul_mean_companion_descriptor(DeviceBackend::Metal);
    let inputs = mul_mean_inputs();

    // Creation: module loaded once + the two PerProgram inputs allocated
    // once. The PerStep intermediates (3, 4, 5) and the ObservationPoint
    // gradient outputs (6, 7) are NOT allocated at creation — they are
    // allocated per execution and released at the step boundary / after
    // readback (S2-4).
    let mut session = host
        .create_program_session(&descriptor)
        .expect("session create");
    assert_eq!(session.session_handle_count(), 3); // module + 2 PerProgram
    assert_eq!(session.driver_counters().module_loads, 1);
    assert_eq!(session.driver_counters().buffer_allocs, 2);
    assert_eq!(session.allocated_buffers(), vec![1, 2, 3, 4, 5, 6, 7]);

    // Execute 1: four launches (mul, mean, backward_x, backward_w);
    // PerProgram inputs copied per input slot (x, w in loss_mul, x in
    // loss_backward_x, w in loss_backward_w = 4); grad_x/grad_w read back
    // and released; the PerStep intermediates recycled at the step
    // boundary.
    let receipt = session.execute(&inputs).expect("execute 1");
    assert_eq!(receipt.launches, 4);
    assert_eq!(receipt.copy_ins, 4);
    assert_eq!(receipt.outputs.get(&6), Some(&vec![5.0, 8.0])); // grad_x
    assert_eq!(receipt.outputs.get(&7), Some(&vec![7.0, 10.0])); // grad_w
    assert_eq!(receipt.per_program_buffers, vec![1, 2]);
    assert_eq!(receipt.per_step_buffers, vec![3, 4, 5]);
    assert_eq!(receipt.observation_buffers, vec![6, 7]);
    // The session is back to module + PerProgram only: the per-step and
    // observation buffers were released (read-then-release + step recycle).
    assert_eq!(session.session_handle_count(), 3);
    let counters = session.driver_counters();
    assert_eq!(counters.buffer_allocs, 2 + 5); // + product, partial, acc, grad_x, grad_w
    assert_eq!(counters.buffer_releases, 5);
    assert_eq!(counters.module_loads, 1);

    // Execute 2 on the SAME session: no reload, no PerProgram realloc; the
    // PerStep and ObservationPoint buffers are allocated fresh and released
    // again (recycled at each step boundary, read-then-released after).
    let receipt2 = session.execute(&inputs).expect("execute 2");
    assert_eq!(receipt2.outputs.get(&6), Some(&vec![5.0, 8.0]));
    assert_eq!(receipt2.outputs.get(&7), Some(&vec![7.0, 10.0]));
    assert_eq!(session.session_handle_count(), 3);
    let counters = session.driver_counters();
    assert_eq!(counters.module_loads, 1);
    assert_eq!(counters.buffer_allocs, 2 + 10);
    assert_eq!(counters.buffer_releases, 10);

    // Teardown releases the module + PerProgram buffers; nothing persists.
    session.teardown().expect("teardown");
    let counters = host.device().expect("device").driver_counters();
    assert_eq!(counters.module_loads, 1);
    assert_eq!(counters.module_releases, 1);
    assert_eq!(counters.buffer_allocs, 2 + 10);
    assert_eq!(counters.buffer_releases, 2 + 10);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

/// The same classified-buffer policy on the CUDA fake-driver lane
/// (backend-neutral surface): PerProgram once, PerStep recycled,
/// ObservationPoint read-then-released, grad_x/grad_w distinct ids.
#[test]
fn mul_mean_companion_classified_buffers_follow_lifetime_policy_cuda() {
    let mut host = mul_mean_cuda_composite().expect("cuda composite");
    let descriptor = mul_mean_companion_descriptor(DeviceBackend::Cuda);
    let inputs = mul_mean_inputs();

    let mut session = host
        .create_program_session(&descriptor)
        .expect("session create");
    assert_eq!(session.session_handle_count(), 3); // module + 2 PerProgram
    assert_eq!(session.driver_counters().buffer_allocs, 2);

    let receipt = session.execute(&inputs).expect("execute 1");
    assert_eq!(receipt.launches, 4);
    assert_eq!(receipt.outputs.get(&6), Some(&vec![5.0, 8.0]));
    assert_eq!(receipt.outputs.get(&7), Some(&vec![7.0, 10.0]));
    assert_eq!(receipt.per_program_buffers, vec![1, 2]);
    assert_eq!(receipt.per_step_buffers, vec![3, 4, 5]);
    assert_eq!(receipt.observation_buffers, vec![6, 7]);
    assert_eq!(session.session_handle_count(), 3);

    let receipt2 = session.execute(&inputs).expect("execute 2");
    assert_eq!(receipt2.outputs.get(&6), Some(&vec![5.0, 8.0]));
    assert_eq!(receipt2.outputs.get(&7), Some(&vec![7.0, 10.0]));
    assert_eq!(session.session_handle_count(), 3);

    session.teardown().expect("teardown");
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

/// The A9 receipt reflects the lifetime policy: per-buffer lifetime class
/// sets and the program-level lifetime regime (S2-4 done-when). A
/// RepeatingStep session runs through the step-mode surface (S5-U6):
/// once-init the HostProvided params, then execute steps.
#[test]
fn receipt_reports_lifetime_classes_and_program_lifetime() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let mut descriptor = two_kernel_inout_descriptor(DeviceBackend::Metal);
    descriptor.program_lifetime = DeviceProgramLifetime::RepeatingStep;

    let mut session = host
        .create_program_session(&descriptor)
        .expect("session create");
    session
        .init_params(&two_kernel_inputs())
        .expect("once-init params");
    let receipt = session.execute_step().expect("execute step");
    assert_eq!(
        receipt.program_lifetime,
        DeviceProgramLifetime::RepeatingStep,
        "the program regime is carried on the descriptor and consumed into the receipt"
    );
    assert_eq!(receipt.per_program_buffers, vec![1, 2, 4]);
    assert_eq!(receipt.per_step_buffers, vec![3]);
    assert_eq!(receipt.observation_buffers, vec![5]);
    assert_eq!(
        receipt.allocated_buffers,
        vec![1, 2, 3, 4, 5],
        "allocated_buffers is the union of the three lifetime classes"
    );
    session.teardown().expect("teardown");
}

// ---------------------------------------------------------------------------
// S5-U6: RepeatingStep host step-mode — once-init HostProvided params,
// N-step loop with PerStep recycle, per-step observation (loss trace),
// per-step receipts, leak-free teardown.
// ---------------------------------------------------------------------------

/// The S5-U6 training-step fixture as a descriptor: two HostProvided
/// PerProgram params (w, b) once-init'd at session creation, one PerStep
/// InOut intermediate (h) recycled at each step boundary, and one
/// ObservationPoint loss output (l) read back per step. The fake driver
/// simulates both kernels as elementwise add, so the step semantics are
/// h = w + b (launch 1), l = h + w (launch 2).
///
/// | Buffer | Role → class | Init |
/// | --- | --- | --- |
/// | w, b (1, 2) | Input → PerProgram | HostProvided |
/// | h (3) | InOut → PerStep | KernelInitialized |
/// | l (4) | Output → ObservationPoint | KernelInitialized |
fn training_step_descriptor(backend: DeviceBackend) -> DeviceDescriptor {
    let mut descriptor = make_descriptor(
        backend,
        vec![
            DescriptorKernel {
                entry: "add_one".to_owned(),
                buffers: vec![
                    add_slot(1, "w", DeviceBufferRole::Input, 0, 2),
                    add_slot(2, "b", DeviceBufferRole::Input, 1, 2),
                    kernel_init_slot(3, "h", 2),
                ],
                grid: [1, 1, 1],
                block: [2, 1, 1],
            },
            DescriptorKernel {
                entry: "add_one".to_owned(),
                buffers: vec![
                    kernel_init_slot(3, "h", 0),
                    add_slot(1, "w", DeviceBufferRole::Input, 1, 2),
                    add_slot(4, "l", DeviceBufferRole::Output, 2, 2),
                ],
                grid: [1, 1, 1],
                block: [2, 1, 1],
            },
        ],
        vec![
            DescriptorLaunch {
                id: 1,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 2,
                kernel_index: 1,
            },
        ],
        // R2: the carried data-flow edge — launch 1 produces the PerStep
        // intermediate h, launch 2 consumes it.
        vec![DescriptorDataFlow {
            buffer_id: 3,
            version: 1,
            producer: 1,
            consumer: 2,
        }],
        // F6: the declared observation point — the loss l, produced by
        // launch 2, read back once per step.
        vec![result(4, 2)],
    );
    descriptor.program_lifetime = DeviceProgramLifetime::RepeatingStep;
    descriptor
}

/// A PerStep InOut slot that is fully written by a device kernel before any
/// read (the training-step intermediate `h`): InOut role → PerStep lifetime
/// (S2-4 mapping), but KernelInitialized initialization — launch 1 writes it
/// before launch 2 reads it, so the step allocates it without a zero-fill
/// copy (the once-init copy accounting stays exact: only the params copy).
fn kernel_init_slot(id: u32, name: &str, binding: u32) -> DescriptorBuffer {
    DescriptorBuffer {
        buffer_id: id,
        buffer_name: name.to_owned(),
        semantic_value: id,
        role: DeviceBufferRole::InOut,
        lifetime: DeviceBufferLifetime::PerStep,
        initialization: DeviceBufferInitialization::KernelInitialized,
        binding,
        element_ty: DeviceDataType::F32,
        element_count: 2,
        version: 1,
    }
}

/// Host values for the once-init params of [`training_step_descriptor`]:
/// w = [1, 2], b = [3, 4]. The simulated step then yields h = [4, 6] and the
/// per-step loss observation l = h + w = [5, 8] — stable across steps only
/// while the params persist on device.
fn training_step_params() -> BTreeMap<u32, Vec<f32>> {
    let mut params = BTreeMap::new();
    params.insert(1, vec![1.0, 2.0]); // w
    params.insert(2, vec![3.0, 4.0]); // b
    params
}

/// A PerProgram + HostProvided slot for any role — the real U5 training
/// shape for params (PerProgram InOut ReadWrite buffers with HostProvided
/// init at session creation, per the Stage 5 delivery architecture). The
/// step-mode once-init copies these exactly once at session creation,
/// regardless of slot role, never via the per-step input path.
fn host_provided_param_slot(
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
        lifetime: DeviceBufferLifetime::PerProgram,
        initialization: DeviceBufferInitialization::HostProvided,
        binding,
        element_ty: DeviceDataType::F32,
        element_count: count,
        version: 1,
    }
}

/// The [`training_step_descriptor`] shape with the params in their real U5
/// wire form: PerProgram InOut ReadWrite buffers with HostProvided init
/// (not Input-role slots). Step semantics are unchanged (h = w + b, then
/// l = h + w).
fn training_step_inout_param_descriptor(backend: DeviceBackend) -> DeviceDescriptor {
    let mut descriptor = make_descriptor(
        backend,
        vec![
            DescriptorKernel {
                entry: "add_one".to_owned(),
                buffers: vec![
                    host_provided_param_slot(1, "w", DeviceBufferRole::InOut, 0, 2),
                    host_provided_param_slot(2, "b", DeviceBufferRole::InOut, 1, 2),
                    kernel_init_slot(3, "h", 2),
                ],
                grid: [1, 1, 1],
                block: [2, 1, 1],
            },
            DescriptorKernel {
                entry: "add_one".to_owned(),
                buffers: vec![
                    kernel_init_slot(3, "h", 0),
                    host_provided_param_slot(1, "w", DeviceBufferRole::InOut, 1, 2),
                    add_slot(4, "l", DeviceBufferRole::Output, 2, 2),
                ],
                grid: [1, 1, 1],
                block: [2, 1, 1],
            },
        ],
        vec![
            DescriptorLaunch {
                id: 1,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 2,
                kernel_index: 1,
            },
        ],
        vec![DescriptorDataFlow {
            buffer_id: 3,
            version: 1,
            producer: 1,
            consumer: 2,
        }],
        vec![result(4, 2)],
    );
    descriptor.program_lifetime = DeviceProgramLifetime::RepeatingStep;
    descriptor
}

/// The S5-U6 done-when fake-driver test: one session runs N steps — params
/// persist and are copied in exactly once, PerStep buffers recycle per
/// step, the observation (loss) readback happens exactly once per declared
/// observation, receipts count per-step syncs/transfers/readbacks/releases,
/// and teardown returns `live_handle_count() == 0` (leak-free).
#[test]
fn repeating_step_session_runs_n_steps_once_init_recycle_observation_leak_free() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = training_step_descriptor(DeviceBackend::Metal);

    let mut session = host
        .create_program_session(&descriptor)
        .expect("session create");
    // Creation: module + the two PerProgram params allocated once; the
    // PerStep intermediate (h) and the ObservationPoint loss (l) are NOT
    // allocated until a step. No copy-in has happened yet.
    assert_eq!(session.session_handle_count(), 3); // module + w + b
    assert_eq!(session.driver_counters().buffer_allocs, 2);
    assert_eq!(session.driver_counters().buffer_releases, 0);

    // Once-init: the HostProvided params are copied in exactly once.
    session
        .init_params(&training_step_params())
        .expect("once-init params");

    // N steps on one session: PerStep recycled per step, observation read
    // back per step, receipts count per-step syncs/transfers/readbacks/
    // releases, no copy-in during steps, params persisted (stable loss).
    const STEPS: usize = 5;
    for _ in 0..STEPS {
        let receipt = session.execute_step().expect("execute step");
        assert_eq!(
            receipt.program_lifetime,
            DeviceProgramLifetime::RepeatingStep
        );
        assert_eq!(receipt.launches, 2);
        assert_eq!(receipt.launch_ids, vec![1, 2]);
        assert_eq!(
            receipt.copy_ins, 0,
            "steps never re-copy the once-init params"
        );
        assert_eq!(
            receipt.readbacks, 1,
            "the loss is observed exactly once per declared observation"
        );
        assert_eq!(
            receipt.transfers, 1,
            "readback-only transfers per step (no copy-ins)"
        );
        assert_eq!(
            receipt.syncs, 2,
            "2 launches each sync internally; the Metal step-boundary sync() is a no-op and is not counted"
        );
        assert_eq!(
            receipt.releases, 2,
            "loss read-then-release + h step-boundary recycle per step"
        );
        // The persisted params keep the loss trace stable across steps.
        assert_eq!(receipt.outputs.get(&4), Some(&vec![5.0, 8.0]));
        assert_eq!(receipt.per_program_buffers, vec![1, 2]);
        assert_eq!(receipt.per_step_buffers, vec![3]);
        assert_eq!(receipt.observation_buffers, vec![4]);
        // Between steps the session holds only module + PerProgram params.
        assert_eq!(session.session_handle_count(), 3);
    }

    // Driver-level accounting: w + b allocated once at creation; each step
    // allocates and releases h + l (2 allocs + 2 releases per step). The
    // module is loaded exactly once.
    let counters = session.driver_counters();
    assert_eq!(counters.module_loads, 1);
    assert_eq!(counters.buffer_allocs, 2 + 2 * STEPS);
    assert_eq!(counters.buffer_releases, 2 * STEPS);

    // Teardown after the loop: module + PerProgram params released; every
    // allocation balances and nothing persists (leak-free).
    session.teardown().expect("teardown");
    let counters = host.device().expect("device").driver_counters();
    assert_eq!(counters.module_loads, 1);
    assert_eq!(counters.module_releases, 1);
    assert_eq!(counters.buffer_allocs, 2 + 2 * STEPS);
    assert_eq!(counters.buffer_releases, 2 + 2 * STEPS);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

/// The same N-step step-mode loop on the fake CUDA lane (backend-neutral
/// surface): once-init, per-step observation, PerStep recycle, leak-free
/// teardown.
#[test]
fn repeating_step_n_step_loop_on_both_fake_backends() {
    for backend in [DeviceBackend::Metal, DeviceBackend::Cuda] {
        let mut host = match backend {
            DeviceBackend::Metal => metal_composite("add_one").expect("metal composite"),
            DeviceBackend::Cuda => cuda_composite("add_one").expect("cuda composite"),
        };
        let descriptor = training_step_descriptor(backend);

        let mut session = host
            .create_program_session(&descriptor)
            .expect("session create");
        session
            .init_params(&training_step_params())
            .expect("once-init params");
        for _ in 0..3 {
            let receipt = session.execute_step().expect("execute step");
            assert_eq!(receipt.outputs.get(&4), Some(&vec![5.0, 8.0]));
            assert_eq!(receipt.copy_ins, 0);
            assert_eq!(receipt.readbacks, 1);
            assert_eq!(receipt.releases, 2);
            assert_eq!(session.session_handle_count(), 3);
        }
        session.teardown().expect("teardown");
        assert_eq!(host.device().expect("device").live_handle_count(), 0);
    }
}

/// The real U5 training shape for params (PerProgram InOut ReadWrite with
/// HostProvided init — the delivery doc's SGD/param classification) works
/// through the step-mode once-init: params are copied in exactly once at
/// session creation regardless of slot role, persist across steps, and
/// steps run without re-copying.
#[test]
fn repeating_step_once_init_inout_host_provided_params() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = training_step_inout_param_descriptor(DeviceBackend::Metal);

    let mut session = host
        .create_program_session(&descriptor)
        .expect("session create");
    assert_eq!(session.session_handle_count(), 3); // module + w + b (PerProgram)
    session
        .init_params(&training_step_params())
        .expect("once-init InOut params");
    for _ in 0..3 {
        let receipt = session.execute_step().expect("execute step");
        assert_eq!(receipt.outputs.get(&4), Some(&vec![5.0, 8.0]));
        assert_eq!(receipt.copy_ins, 0);
        assert_eq!(receipt.per_program_buffers, vec![1, 2]);
        assert_eq!(receipt.readbacks, 1);
    }
    session.teardown().expect("teardown");
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

/// The once-init copy accounting is observable at the driver boundary: a
/// RepeatingStep session copies its HostProvided params exactly once (w is
/// copy call 1, b is copy call 2) and steps copy nothing. Injecting a
/// CopyIn failure at call 3 must therefore NOT fire across N steps — a
/// step that re-copied params would fail.
#[test]
fn repeating_step_steps_copy_nothing_and_params_copy_exactly_once() {
    let mut host =
        metal_composite_failing("add_one", FakeFailureStage::CopyIn, 3).expect("metal composite");
    let descriptor = training_step_descriptor(DeviceBackend::Metal);

    let mut session = host
        .create_program_session(&descriptor)
        .expect("session create");
    session
        .init_params(&training_step_params())
        .expect("once-init consumes copy calls 1 and 2 (w, b)");
    for _ in 0..3 {
        session
            .execute_step()
            .expect("a step performs no copy-in (the injected call-3 failure never fires)");
    }
    session.teardown().expect("teardown");
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

/// A failure during the once-init copy (S2-3 error-path teardown): the
/// second param copy (b, copy call 2) fails, the session releases every
/// handle and closes — `live_handle_count() == 0`, and a closed session
/// refuses further steps.
#[test]
fn repeating_step_init_failure_releases_every_handle() {
    let mut host =
        metal_composite_failing("add_one", FakeFailureStage::CopyIn, 2).expect("metal composite");
    let descriptor = training_step_descriptor(DeviceBackend::Metal);

    let mut session = host
        .create_program_session(&descriptor)
        .expect("session create");
    let err = session
        .init_params(&training_step_params())
        .expect_err("the second param copy (b) is injected to fail");
    assert_eq!(err.code, E_METAL_DRIVER);
    assert_eq!(
        session.session_handle_count(),
        0,
        "a failed once-init releases every handle (S2-3 error-path teardown)"
    );

    let err = session
        .execute_step()
        .expect_err("a session closed by a failed once-init refuses steps");
    assert_eq!(err.code, "E_INTERNAL");
    drop(session);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

/// The step-mode and SingleRun surfaces never mix (S5-U6): `execute` on a
/// RepeatingStep session, `execute_step` on a SingleRun session, a step
/// before the once-init, and a second once-init all fail closed with typed
/// diagnostics — never a silent fallback.
#[test]
fn repeating_step_surfaces_fail_closed_on_misuse() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let step_descriptor = training_step_descriptor(DeviceBackend::Metal);

    // SingleRun session: execute_step and init_params are refused (the
    // step-mode surface is a RepeatingStep contract).
    let mut single_descriptor = step_descriptor.clone();
    single_descriptor.program_lifetime = DeviceProgramLifetime::SingleRun;
    let mut session = host
        .create_program_session(&single_descriptor)
        .expect("session create");
    let err = session
        .execute_step()
        .expect_err("execute_step is a RepeatingStep surface");
    assert_eq!(err.code, "E_INTERNAL");
    assert!(err.message.contains("RepeatingStep"));
    let err = session
        .init_params(&training_step_params())
        .expect_err("init_params is a RepeatingStep surface");
    assert_eq!(err.code, "E_INTERNAL");
    assert!(err.message.contains("RepeatingStep"));
    session.teardown().expect("teardown");

    // RepeatingStep session: execute (per-execution input copy-in) is
    // refused — params are once-init'd and never re-copied.
    let mut session = host
        .create_program_session(&step_descriptor)
        .expect("session create");
    let err = session
        .execute(&training_step_params())
        .expect_err("execute is the SingleRun surface");
    assert_eq!(err.code, "E_INTERNAL");
    assert!(err.message.contains("SingleRun"));

    // A step before the once-init is refused.
    let err = session
        .execute_step()
        .expect_err("params must be once-init'd first");
    assert_eq!(err.code, "E_INTERNAL");
    assert!(err.message.contains("init_params"));

    // The once-init runs exactly once; a second copy is refused.
    session
        .init_params(&training_step_params())
        .expect("once-init");
    let err = session
        .init_params(&training_step_params())
        .expect_err("HostProvided params are copied in exactly once");
    assert_eq!(err.code, "E_INTERNAL");
    assert!(err.message.contains("exactly once"));
    session.teardown().expect("teardown");
}

/// Once-init validates every declared param: a missing param or a size
/// that contradicts the declared element count fails with
/// `E_DEVICE_SHAPE_MISMATCH`, and the failed once-init releases every
/// handle (S2-3 error-path teardown).
#[test]
fn repeating_step_init_requires_every_declared_param() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = training_step_descriptor(DeviceBackend::Metal);

    // Missing b.
    let mut missing = training_step_params();
    missing.remove(&2);
    let mut session = host
        .create_program_session(&descriptor)
        .expect("session create");
    let err = session
        .init_params(&missing)
        .expect_err("a missing declared param must fail the once-init");
    assert_eq!(err.code, E_DEVICE_SHAPE_MISMATCH);
    assert!(err.message.contains("b"));
    assert_eq!(
        session.session_handle_count(),
        0,
        "a failed once-init releases every handle (S2-3 error-path teardown)"
    );
    drop(session);

    // Wrong size for w.
    let mut bad_size = training_step_params();
    bad_size.insert(1, vec![1.0]);
    let mut session = host
        .create_program_session(&descriptor)
        .expect("session create");
    let err = session
        .init_params(&bad_size)
        .expect_err("a param size contradicting the declared element count must fail");
    assert_eq!(err.code, E_DEVICE_SHAPE_MISMATCH);
    drop(session);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

/// The descriptor-level RepeatingStep contract (S5-U6): a HostProvided
/// buffer with a non-PerProgram lifetime could never receive its values in
/// step-mode (steps copy nothing), so the descriptor fails closed at
/// validation before any launch.
#[test]
fn repeating_step_host_provided_buffer_requires_per_program_lifetime() {
    let mut invalid = training_step_descriptor(DeviceBackend::Metal);
    // Relabel the PerStep intermediate h (id 3) as HostProvided: step-mode
    // cannot once-init it (it is not PerProgram), so validation fails.
    for kernel in &mut invalid.kernels {
        for buffer in &mut kernel.buffers {
            if buffer.buffer_id == 3 {
                buffer.initialization = DeviceBufferInitialization::HostProvided;
            }
        }
    }
    let err = invalid
        .validate()
        .expect_err("a HostProvided non-PerProgram buffer must fail closed");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
    assert!(err.message.contains("RepeatingStep"));
    assert!(err.message.contains("per-program"));
}

/// ObservationPoint is the only readback: a result naming a PerProgram
/// buffer is an undeclared readback and fails closed at host admission with
/// `E_DEVICE_DESCRIPTOR` before any launch (F6 — the constructor and host
/// admission agree).
#[test]
fn non_observation_readback_fails_closed() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);

    // Buffer 1 is a PerProgram input; declaring it as a result is not a
    // declared observation point and must fail closed before any launch.
    let mut per_program_result = descriptor.clone();
    per_program_result.results = vec![DescriptorResult {
        buffer_id: 1,
        version: 1,
        produced_by: 1,
        at_launch: 1,
    }];
    let err = host
        .create_program_session(&per_program_result)
        .err()
        .expect("reading back a PerProgram buffer is an undeclared readback");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
    assert!(err.message.contains("observation-point"));
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

/// A buffer's lifetime is an identity fact: two kernels referencing the same
/// buffer id with different lifetimes is a descriptor conflict that fails
/// before launch (E_DEVICE_ABI_MISMATCH), matching the radix schema's
/// BufferIdentityConflict on lifetime.
#[test]
fn conflicting_buffer_lifetimes_fail_as_abi_mismatch() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let mut descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    // Relabel buffer id 3 (the observation output) as PerProgram in the
    // second reference — no, a single-kernel descriptor has one reference;
    // add a second kernel referencing id 3 with a different lifetime.
    descriptor.kernels.push(DescriptorKernel {
        entry: "add_one".to_owned(),
        buffers: vec![
            add_slot(3, "out", DeviceBufferRole::Output, 0, 2),
            add_slot(4, "c", DeviceBufferRole::Input, 1, 2),
            add_slot(5, "d", DeviceBufferRole::Input, 2, 2),
        ],
        grid: [1, 1, 1],
        block: [2, 1, 1],
    });
    descriptor.kernels[1].buffers[0].lifetime = DeviceBufferLifetime::PerProgram;
    let err = host
        .create_program_session(&descriptor)
        .err()
        .expect("conflicting lifetimes must fail before launch");
    assert_eq!(err.code, E_DEVICE_ABI_MISMATCH);
    assert!(err.message.contains("lifetimes"));
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

/// A failure while allocating a PerStep buffer (a new allocation point
/// introduced by S2-4) leaves live_handle_count() == 0 — the S2-3 error-path
/// teardown covers the per-step allocation stage too.
#[test]
fn per_step_allocation_failure_releases_every_handle() {
    // Creation allocates PerProgram ids 1, 2, 4 (alloc calls 1-3); the first
    // per-step allocation (id 3) is alloc call 4.
    let mut host =
        metal_composite_failing("add_one", FakeFailureStage::Alloc, 4).expect("metal composite");
    let descriptor = two_kernel_inout_descriptor(DeviceBackend::Metal);
    let err = host
        .create_program_session(&descriptor)
        .expect("session create")
        .execute(&two_kernel_inputs())
        .expect_err("injected per-step allocation failure must fail the execution");
    assert_eq!(err.code, E_METAL_DRIVER);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

/// The S2-8 A9/A10 receipt: the declared logical resource graph (every
/// buffer identity + content version, roles, lifetimes) and the observed
/// lifecycle events (launches, syncs, transfers, readbacks, releases) for
/// the two-kernel chain. The data-flow edge matches the schema's
/// `data_flow_pairs`: launch 1 writes the InOut intermediate `acc`, launch 2
/// reads it.
#[test]
fn receipt_declares_resource_graph_and_observed_events() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = two_kernel_inout_descriptor(DeviceBackend::Metal);

    let mut session = host
        .create_program_session(&descriptor)
        .expect("session create");
    let receipt = session.execute(&two_kernel_inputs()).expect("execute");

    // Declared resource graph (A10), buffer-id order: identity facts +
    // content version 1 for all five buffers.
    let graph = &receipt.resource_graph;
    assert_eq!(graph.len(), 5);
    assert_eq!(graph[0].id, 1);
    assert_eq!(graph[0].name, "a");
    assert_eq!(graph[0].role, DeviceBufferRole::Input);
    assert_eq!(graph[0].lifetime, DeviceBufferLifetime::PerProgram);
    assert_eq!(graph[2].id, 3);
    assert_eq!(graph[2].name, "acc");
    assert_eq!(graph[2].role, DeviceBufferRole::InOut);
    assert_eq!(graph[2].lifetime, DeviceBufferLifetime::PerStep);
    assert_eq!(graph[4].id, 5);
    assert_eq!(graph[4].name, "out");
    assert_eq!(graph[4].role, DeviceBufferRole::Output);
    assert_eq!(graph[4].lifetime, DeviceBufferLifetime::ObservationPoint);
    for buffer in graph {
        assert_eq!(buffer.element_count, 2);
        assert_eq!(buffer.element_ty, DeviceDataType::F32);
        assert_eq!(buffer.version, 1);
    }

    // Data-flow edges (A10): launch 1 produces the InOut intermediate, launch
    // 2 consumes it; no other inter-kernel edge.
    assert_eq!(
        receipt.data_flow_edges,
        vec![DataFlowEdge {
            buffer_id: 3,
            version: 1,
            producer: 1,
            consumer: 2,
        }]
    );

    // Observed lifecycle events (A9): 2 launches, 2 real synchronization
    // operations (one per launch's internal wait — the Metal step-boundary
    // `sync()` is a no-op because every launch already waited, so it is NOT
    // an actual synchronization event and is not counted), 3 copy-ins + 1
    // readback = 4 transfers, 1 readback, and 2 releases (read-then-release
    // of `out` + step-boundary recycle of `acc`).
    assert_eq!(receipt.launches, 2);
    assert_eq!(receipt.syncs, 2);
    assert_eq!(receipt.copy_ins, 3);
    assert_eq!(receipt.transfers, 4);
    assert_eq!(receipt.readbacks, 1);
    assert_eq!(receipt.outputs.len(), 1);
    assert_eq!(receipt.releases, 2);

    // R9: the completion boundary is the explicit step-boundary sync after
    // the last launch (2) — stated exactly, never beyond the synchronization
    // the host actually performed.
    assert_eq!(
        receipt.completion_boundary,
        CompletionBoundary::StepSync { after_launch: 2 }
    );
    assert_eq!(
        receipt.completion_boundary.spelling(),
        "completion guaranteed at the explicit step-boundary sync after launch 2"
    );

    // The receipt carries the semantic graph hash the host computed from the
    // descriptor it consumed (the graph identity of this execution).
    assert_eq!(
        receipt.semantic_graph_hash,
        descriptor.semantic_graph_hash()
    );
    assert_eq!(
        session.semantic_graph_hash(),
        descriptor.semantic_graph_hash()
    );

    session.teardown().expect("teardown");
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

/// P1-4 (A9 evidence honesty): a receipt counts only ACTUAL device
/// synchronization events. Every launch synchronizes internally on both
/// backends (Metal waits per launch; CUDA syncs per launch) — counted. The
/// explicit step-boundary `sync()` is counted only where it performs a real
/// synchronization: CUDA issues `cuCtxSynchronize` (additive), while Metal is
/// a no-op because every launch already waited (NOT counted). For the
/// identical two-launch descriptor, the Metal receipt reports 2 syncs and the
/// CUDA receipt reports 3.
#[test]
fn receipt_syncs_count_only_real_synchronizations_per_backend() {
    for (backend, composite) in [
        (DeviceBackend::Metal, metal_composite("add_one")),
        (DeviceBackend::Cuda, cuda_composite("add_one")),
    ] {
        let mut host = composite.expect("composite");
        let descriptor = two_kernel_inout_descriptor(backend);
        let mut session = host
            .create_program_session(&descriptor)
            .expect("session create");
        let receipt = session.execute(&two_kernel_inputs()).expect("execute");

        assert_eq!(receipt.launches, 2);
        let expected_syncs = match backend {
            // One real sync per launch's internal wait; the step-boundary
            // sync() is a no-op on Metal and must not be counted.
            DeviceBackend::Metal => 2,
            // One real sync per launch's cuCtxSynchronize plus the additive
            // step-boundary cuCtxSynchronize.
            DeviceBackend::Cuda => 3,
        };
        assert_eq!(receipt.syncs, expected_syncs);
        // The completion boundary stays the step-boundary barrier after the
        // last launch on both lanes.
        assert_eq!(
            receipt.completion_boundary,
            CompletionBoundary::StepSync { after_launch: 2 }
        );

        session.teardown().expect("teardown");
        assert_eq!(host.device().expect("device").live_handle_count(), 0);
    }
}

/// R2 (S3-A8): the A10 resource graph consumes the WIRE'S carried version
/// facts — a buffer carrying version 2 and a carried producer/consumer edge
/// render AS-IS, never hardcoded `version: 1` and never re-derived from a
/// first-writer launch-order coincidence rule.
#[test]
fn receipt_consumes_carried_version_facts_not_coincidence() {
    let mut host = metal_composite("add_one").expect("metal composite");
    // The two-kernel chain (the same shape the S2-8 receipt test drives):
    // `acc` is an accumulation buffer whose slots carry version 2 (the S2-5
    // `version > 1` accumulation pattern). The wire carries the edge for it.
    let mut descriptor = two_kernel_inout_descriptor(DeviceBackend::Metal);
    descriptor.kernels[0].buffers[2].version = 2;
    descriptor.kernels[1].buffers[0].version = 2;
    descriptor.buffer_versions = buffer_versions_for(&descriptor.kernels);
    descriptor.data_flow = vec![DescriptorDataFlow {
        buffer_id: 3,
        version: 2,
        producer: 1,
        consumer: 2,
    }];

    let mut session = host
        .create_program_session(&descriptor)
        .expect("session create");
    let receipt = session.execute(&two_kernel_inputs()).expect("execute");

    // The rendered graph consumes the carried version (2), not a hardcoded 1.
    let acc = receipt
        .resource_graph
        .iter()
        .find(|buffer| buffer.id == 3)
        .expect("acc in graph");
    assert_eq!(acc.version, 2, "the host must consume the carried version");

    // The rendered edges are the carried facts, not a first-writer derivation.
    assert_eq!(
        receipt.data_flow_edges,
        vec![DataFlowEdge {
            buffer_id: 3,
            version: 2,
            producer: 1,
            consumer: 2,
        }]
    );

    session.teardown().expect("teardown");
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

// ---------------------------------------------------------------------------
// Stage 3R U5: coherent host execution — graph-ordered scheduling (F3/G1),
// observation-only readback (F6), honest receipts (R9), semantic graph hash
// ---------------------------------------------------------------------------

/// The declaration-reorder red test (G1 host side): reordering the kernel
/// DECLARATIONS — the host never infers execution from declaration order —
/// does not change the executed launch sequence, the results, or the
/// semantic graph hash. The host schedules the carried launches + graph,
/// so the same launches with differently-ordered declarations are one
/// program.
#[test]
fn declaration_reorder_does_not_change_execution_or_graph_hash() {
    let mut host = multi_kernel_metal_composite().expect("metal composite");

    let kernel_zero = DescriptorKernel {
        entry: "kernel_zero".to_owned(),
        buffers: vec![
            add_slot(1, "a0", DeviceBufferRole::Input, 0, 2),
            add_slot(2, "b0", DeviceBufferRole::Input, 1, 2),
            add_slot(3, "out0", DeviceBufferRole::Output, 2, 2),
        ],
        grid: [1, 1, 1],
        block: [2, 1, 1],
    };
    let kernel_one = DescriptorKernel {
        entry: "kernel_one".to_owned(),
        buffers: vec![
            add_slot(4, "a1", DeviceBufferRole::Input, 0, 2),
            add_slot(5, "b1", DeviceBufferRole::Input, 1, 2),
            add_slot(6, "out1", DeviceBufferRole::Output, 2, 2),
        ],
        grid: [1, 1, 1],
        block: [2, 1, 1],
    };

    // The semantic program: launch order kernel_one, kernel_zero, kernel_one
    // with out1 (buffer 6) the declared observation point. Declaration order
    // is NOT part of the program — the two descriptors differ only in how
    // the kernels are declared (and the launch indices remapped to match).
    let build = |kernels: Vec<DescriptorKernel>, one: u32, zero: u32| DeviceDescriptor {
        backend: DeviceBackend::Metal,
        module_image: MODULE_IMAGE.to_vec(),
        buffer_versions: buffer_versions_for(&kernels),
        kernels,
        launches: vec![
            DescriptorLaunch {
                id: 1,
                kernel_index: one,
            },
            DescriptorLaunch {
                id: 2,
                kernel_index: zero,
            },
            DescriptorLaunch {
                id: 3,
                kernel_index: one,
            },
        ],
        program_lifetime: DeviceProgramLifetime::SingleRun,
        data_flow: Vec::new(),
        roots: vec![1, 2, 3],
        results: vec![result(6, 3)],
    };
    let declared_zero_first = build(vec![kernel_zero.clone(), kernel_one.clone()], 1, 0);
    let declared_one_first = build(vec![kernel_one.clone(), kernel_zero.clone()], 0, 1);

    let inputs = BTreeMap::from([
        (1, vec![1.0, 2.0]),
        (2, vec![3.0, 4.0]),
        (4, vec![5.0, 6.0]),
        (5, vec![7.0, 8.0]),
    ]);

    let receipt_a = host
        .execute_descriptor(&declared_zero_first, &inputs)
        .expect("declaration order A");
    let receipt_b = host
        .execute_descriptor(&declared_one_first, &inputs)
        .expect("declaration order B");

    // Execution order + results are declaration-independent.
    assert_eq!(
        receipt_a.launch_entries,
        vec!["kernel_one", "kernel_zero", "kernel_one"]
    );
    assert_eq!(receipt_b.launch_entries, receipt_a.launch_entries);
    assert_eq!(receipt_b.launch_ids, receipt_a.launch_ids);
    assert_eq!(receipt_b.outputs, receipt_a.outputs);
    assert_eq!(receipt_b.outputs.get(&6), Some(&vec![12.0, 14.0]));

    // The semantic graph hash is declaration-independent too: both
    // descriptors name the same launches, graph, semantic identities, and
    // observation points.
    assert_eq!(
        declared_zero_first.semantic_graph_hash(),
        declared_one_first.semantic_graph_hash()
    );
    assert_eq!(
        receipt_a.semantic_graph_hash,
        declared_zero_first.semantic_graph_hash()
    );
    assert_eq!(
        host.device().expect("device present").live_handle_count(),
        0
    );
}

/// F3 red test: a dependency cycle fails validation before launch — the host
/// never schedules an acyclic graph that isn't there.
#[test]
fn dependency_cycle_fails_validation_before_launch() {
    let descriptor = make_descriptor(
        DeviceBackend::Metal,
        vec![DescriptorKernel {
            entry: "add_one".to_owned(),
            buffers: vec![
                add_slot(8, "x", DeviceBufferRole::InOut, 0, 2),
                add_slot(9, "acc", DeviceBufferRole::InOut, 1, 2),
            ],
            grid: [1, 1, 1],
            block: [2, 1, 1],
        }],
        vec![
            DescriptorLaunch {
                id: 1,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 2,
                kernel_index: 0,
            },
        ],
        vec![
            DescriptorDataFlow {
                buffer_id: 9,
                version: 1,
                producer: 1,
                consumer: 2,
            },
            DescriptorDataFlow {
                buffer_id: 8,
                version: 1,
                producer: 2,
                consumer: 1,
            },
        ],
        Vec::new(),
    );
    let mut descriptor = descriptor;
    // Anchor the schedule at a root so the graph checks (not the root set)
    // decide this descriptor's fate.
    descriptor.roots = vec![1];
    let err = descriptor
        .validate()
        .expect_err("a dependency cycle must fail before launch");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
    assert!(err.message.contains("dependency graph"));
}

/// F2 red test: two launches defining the same value generation (same
/// buffer, same content version) is a duplicate definition and fails
/// validation before launch — one generation has exactly one producer.
#[test]
fn duplicate_value_generation_producer_fails_validation() {
    let descriptor = make_descriptor(
        DeviceBackend::Metal,
        vec![DescriptorKernel {
            entry: "add_one".to_owned(),
            buffers: vec![add_slot(9, "acc", DeviceBufferRole::InOut, 0, 2)],
            grid: [1, 1, 1],
            block: [2, 1, 1],
        }],
        vec![
            DescriptorLaunch {
                id: 1,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 2,
                kernel_index: 0,
            },
        ],
        vec![
            DescriptorDataFlow {
                buffer_id: 9,
                version: 1,
                producer: 1,
                consumer: 2,
            },
            DescriptorDataFlow {
                buffer_id: 9,
                version: 1,
                producer: 2,
                consumer: 1,
            },
        ],
        Vec::new(),
    );
    let mut descriptor = descriptor;
    // Anchor the schedule at a root so the graph checks (not the root set)
    // decide this descriptor's fate.
    descriptor.roots = vec![1];
    let err = descriptor
        .validate()
        .expect_err("a second producer of the same value generation must fail");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
    assert!(err.message.contains("producer"));
}

/// U7 regression: one producer feeding several consumers is a legitimate
/// fan-out. The wire carries one data-flow edge per (producer, consumer)
/// pair (DescriptorDataFlow mirrors `BufferRegistry::data_flow_pairs`), so
/// the producer repeats across edges; the check asserts producer-uniqueness,
/// not edge-uniqueness. Mirrors the MLP buffer 14 v1 shape (producer launch
/// 4, consumers 5 and 8).
#[test]
fn single_producer_multi_consumer_fan_out_passes_validation() {
    let descriptor = make_descriptor(
        DeviceBackend::Metal,
        vec![DescriptorKernel {
            entry: "add_one".to_owned(),
            buffers: vec![add_slot(
                14,
                "companion_grad",
                DeviceBufferRole::InOut,
                0,
                2,
            )],
            grid: [1, 1, 1],
            block: [2, 1, 1],
        }],
        vec![
            DescriptorLaunch {
                id: 4,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 5,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 8,
                kernel_index: 0,
            },
        ],
        vec![
            DescriptorDataFlow {
                buffer_id: 14,
                version: 1,
                producer: 4,
                consumer: 5,
            },
            DescriptorDataFlow {
                buffer_id: 14,
                version: 1,
                producer: 4,
                consumer: 8,
            },
        ],
        Vec::new(),
    );
    descriptor
        .validate()
        .expect("one producer with several consumers is a legitimate fan-out");
}

/// U7 guard: admitting repeated edges with the same producer (fan-out) must
/// not mask a later DIFFERENT producer of the same value generation — the
/// "one value generation has exactly one producer" invariant still holds.
#[test]
fn fan_out_does_not_mask_a_different_producer() {
    let descriptor = make_descriptor(
        DeviceBackend::Metal,
        vec![DescriptorKernel {
            entry: "add_one".to_owned(),
            buffers: vec![add_slot(9, "acc", DeviceBufferRole::InOut, 0, 2)],
            grid: [1, 1, 1],
            block: [2, 1, 1],
        }],
        vec![
            DescriptorLaunch {
                id: 2,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 4,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 5,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 8,
                kernel_index: 0,
            },
        ],
        vec![
            DescriptorDataFlow {
                buffer_id: 9,
                version: 1,
                producer: 4,
                consumer: 5,
            },
            DescriptorDataFlow {
                buffer_id: 9,
                version: 1,
                producer: 4,
                consumer: 8,
            },
            DescriptorDataFlow {
                buffer_id: 9,
                version: 1,
                producer: 2,
                consumer: 5,
            },
        ],
        Vec::new(),
    );
    let err = descriptor
        .validate()
        .expect_err("a different producer must still fail after admitted fan-out");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
    assert!(
        err.message
            .contains("one value generation has exactly one producer")
    );
}

/// F3 red test: a consumer scheduled before its producer is a missing
/// dependency and fails validation before launch.
#[test]
fn consumer_before_producer_fails_validation() {
    let descriptor = make_descriptor(
        DeviceBackend::Metal,
        vec![DescriptorKernel {
            entry: "add_one".to_owned(),
            buffers: vec![add_slot(9, "acc", DeviceBufferRole::InOut, 0, 2)],
            grid: [1, 1, 1],
            block: [2, 1, 1],
        }],
        vec![
            DescriptorLaunch {
                id: 1,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 2,
                kernel_index: 0,
            },
        ],
        vec![DescriptorDataFlow {
            buffer_id: 9,
            version: 1,
            producer: 2,
            consumer: 1,
        }],
        Vec::new(),
    );
    let err = descriptor
        .validate()
        .expect_err("a consumer before its producer must fail before launch");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
    assert!(err.message.contains("dependency graph"));
}

/// F3 red test: a launch the declared roots cannot reach is an incomplete
/// schedule and fails validation before launch.
#[test]
fn unreachable_launch_fails_validation() {
    let mut descriptor = make_descriptor(
        DeviceBackend::Metal,
        vec![DescriptorKernel {
            entry: "add_one".to_owned(),
            buffers: vec![add_slot(9, "acc", DeviceBufferRole::InOut, 0, 2)],
            grid: [1, 1, 1],
            block: [2, 1, 1],
        }],
        vec![
            DescriptorLaunch {
                id: 1,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 2,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 3,
                kernel_index: 0,
            },
        ],
        vec![DescriptorDataFlow {
            buffer_id: 9,
            version: 1,
            producer: 1,
            consumer: 2,
        }],
        Vec::new(),
    );
    // Declare only launch 1 as a root: launch 3 is unreachable.
    descriptor.roots = vec![1];
    let err = descriptor
        .validate()
        .expect_err("an unreachable launch must fail before launch");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
    assert!(err.message.contains("not reachable"));
}

/// A fake-metal composite whose module declares the G4 accumulation chain
/// entries (`accumulate` + `observa`).
fn accumulation_metal_composite() -> HostResult<CompositeHost> {
    let runtime = DeviceRuntime::Metal(
        MetalHostSession::with_driver(Box::new(
            FakeMetalDriver::default()
                .with_known_entry("accumulate")
                .with_known_entry("observa"),
        ))
        .expect("fake metal admit"),
    );
    CompositeHost::with_device(runtime, "fake-metal-device")
}

/// The CUDA lane of [`accumulation_metal_composite`].
fn accumulation_cuda_composite() -> HostResult<CompositeHost> {
    let runtime = DeviceRuntime::Cuda(
        CudaHostSession::with_driver(Box::new(
            FakeCudaDriver::default()
                .with_known_entry("accumulate")
                .with_known_entry("observa"),
        ))
        .expect("fake cuda admit"),
    );
    CompositeHost::with_device(runtime, "fake-cuda-device")
}

/// A PerProgram + ZeroFill accumulation slot (the constructor's G4
/// classification for in-place ReadWrite state): allocated once at session
/// creation, zero-filled once, persistent across executions — never recycled
/// at a step boundary.
fn accumulation_slot(id: u32, name: &str, binding: u32, count: u64) -> DescriptorBuffer {
    DescriptorBuffer {
        buffer_id: id,
        buffer_name: name.to_owned(),
        semantic_value: id,
        role: DeviceBufferRole::InOut,
        lifetime: DeviceBufferLifetime::PerProgram,
        initialization: DeviceBufferInitialization::ZeroFill,
        binding,
        element_ty: DeviceDataType::F32,
        element_count: count,
        version: 1,
    }
}

/// G4 production accumulation descriptor: the two-kernel accumulate chain —
/// kernel `accumulate` adds the input `a` (id 1) into the persistent
/// ZeroFill accumulation buffer `acc` (id 2, PerProgram) in place; kernel
/// `observa` copies `acc` into the observation slot `out` (id 3,
/// ObservationPoint) so the repeated-write test reads the accumulated value
/// back. The data-flow edge (acc, launch 1 -> launch 2) is the carried
/// dependency; the accumulation buffer itself persists across executions.
fn accumulation_descriptor(backend: DeviceBackend) -> DeviceDescriptor {
    make_descriptor(
        backend,
        vec![
            DescriptorKernel {
                entry: "accumulate".to_owned(),
                buffers: vec![
                    add_slot(1, "a", DeviceBufferRole::Input, 0, 4),
                    accumulation_slot(2, "acc", 1, 4),
                ],
                grid: [1, 1, 1],
                block: [4, 1, 1],
            },
            DescriptorKernel {
                entry: "observa".to_owned(),
                buffers: vec![
                    // The program-level InOut role of the shared accumulation
                    // buffer repeats here with the same PerProgram lifetime
                    // and ZeroFill initialization (device-resident, never a
                    // host input — the host copies only Input-role slots).
                    accumulation_slot(2, "acc", 0, 4),
                    add_slot(3, "out", DeviceBufferRole::Output, 1, 4),
                ],
                grid: [1, 1, 1],
                block: [4, 1, 1],
            },
        ],
        vec![
            DescriptorLaunch {
                id: 1,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 2,
                kernel_index: 1,
            },
        ],
        vec![DescriptorDataFlow {
            buffer_id: 2,
            version: 1,
            producer: 1,
            consumer: 2,
        }],
        vec![result(3, 2)],
    )
}

/// G4 (P2): the production accumulation lifecycle — a persistent ZeroFill
/// accumulation buffer is initialized EXACTLY ONCE at session creation,
/// updated by every launch (`acc[i] += a[i]`), and read back only through a
/// declared observation slot. Two executions on one session produce twice
/// the single-run value (repeated-write proof through the production host
/// path, both backends).
#[test]
fn accumulation_buffer_initialized_once_and_accumulates_across_executions() {
    for backend in [DeviceBackend::Metal, DeviceBackend::Cuda] {
        let mut host = match backend {
            DeviceBackend::Metal => accumulation_metal_composite().expect("metal composite"),
            DeviceBackend::Cuda => accumulation_cuda_composite().expect("cuda composite"),
        };
        let descriptor = accumulation_descriptor(backend);
        let mut session = host
            .create_program_session(&descriptor)
            .expect("session create");

        let mut inputs = BTreeMap::new();
        inputs.insert(1, vec![1.0, 2.0, 3.0, 4.0]);

        // Execution 1: acc = zero-fill + a = a; out = acc = a.
        let receipt_one = session.execute(&inputs).expect("first execute");
        assert_eq!(
            receipt_one.outputs.get(&3).map(Vec::as_slice),
            Some([1.0, 2.0, 3.0, 4.0].as_slice()),
            "first execution must read back a (acc initialized once, zero-filled)"
        );

        // Execution 2: acc persists (never re-initialized) -> acc = 2a;
        // out = 2a. The repeated-write proof.
        let receipt_two = session.execute(&inputs).expect("second execute");
        assert_eq!(
            receipt_two.outputs.get(&3).map(Vec::as_slice),
            Some([2.0, 4.0, 6.0, 8.0].as_slice()),
            "second execution must read back 2a (persistent accumulation buffer)"
        );

        session.teardown().expect("teardown");
        assert_eq!(host.device().expect("device").live_handle_count(), 0);
    }
}
