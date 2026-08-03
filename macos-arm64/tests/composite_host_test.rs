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
    resolve_device_selection, CompositeHost, CompositeHostConfig,
};
use faber_host_macos_arm64::cuda_host::E_CUDA_DRIVER;
use faber_host_macos_arm64::device_descriptor::{
    fnv1a64, DescriptorBuffer, DescriptorKernel, DeviceBufferRole, DeviceDataType,
    DeviceDescriptor, E_BACKEND_UNAVAILABLE, E_DEVICE_ABI_MISMATCH, E_DEVICE_DESCRIPTOR,
    E_DEVICE_DTYPE_MISMATCH, E_DEVICE_ENTRY_MISMATCH, E_DEVICE_SHAPE_MISMATCH, E_NO_DEVICE_PROGRAM,
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
fn add_slot(
    id: u32,
    name: &str,
    role: DeviceBufferRole,
    binding: u32,
    count: u64,
) -> DescriptorBuffer {
    DescriptorBuffer {
        buffer_id: id,
        buffer_name: name.to_owned(),
        role,
        binding,
        element_ty: DeviceDataType::F32,
        element_count: count,
    }
}

fn elementwise_add_descriptor(backend: DeviceBackend, entry: &str, count: u64) -> DeviceDescriptor {
    DeviceDescriptor {
        backend,
        module_image: MODULE_IMAGE.to_vec(),
        kernels: vec![DescriptorKernel {
            entry: entry.to_owned(),
            buffers: vec![
                add_slot(1, "a", DeviceBufferRole::Input, 0, count),
                add_slot(2, "b", DeviceBufferRole::Input, 1, count),
                add_slot(3, "out", DeviceBufferRole::Output, 2, count),
            ],
            grid: [1, 1, 1],
            block: [count as u32, 1, 1],
        }],
    }
}

fn metal_composite(entry: &str) -> HostResult<CompositeHost> {
    let runtime = DeviceRuntime::Metal(
        MetalHostSession::with_driver(Box::new(FakeMetalDriver::default().with_known_entry(entry)))
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

/// A fake-metal composite whose driver fails the `call`-th invocation of
/// `stage` with a typed `E_METAL_DRIVER` error (S2-3 failure injection).
fn metal_composite_failing(
    entry: &str,
    stage: FakeFailureStage,
    call: u32,
) -> HostResult<CompositeHost> {
    let runtime = DeviceRuntime::Metal(
        MetalHostSession::with_driver(
            Box::new(
                FakeMetalDriver::default()
                    .with_known_entry(entry)
                    .with_failure_at(stage, call),
            ),
        )
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
        CudaHostSession::with_driver(
            Box::new(
                FakeCudaDriver::default()
                    .with_known_entry(entry)
                    .with_failure_at(stage, call),
            ),
        )
        .expect("fake cuda admit"),
    );
    CompositeHost::with_device(runtime, "fake-cuda-device")
}

/// The two-kernel InOut chain descriptor (kernel 1 writes `acc`, kernel 2
/// reads it): a, b → acc → out, with c as kernel 2's second input. Shared by
/// the mid-chain failure-injection test and the S2-1 chaining tests.
fn two_kernel_inout_descriptor() -> DeviceDescriptor {
    DeviceDescriptor {
        backend: DeviceBackend::Metal,
        module_image: MODULE_IMAGE.to_vec(),
        kernels: vec![
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
    }
}

/// Host inputs for [`two_kernel_inout_descriptor`]: a, b, and c.
fn two_kernel_inputs() -> BTreeMap<u32, Vec<f32>> {
    let mut inputs = BTreeMap::new();
    inputs.insert(1, vec![1.0, 2.0]);
    inputs.insert(2, vec![3.0, 4.0]);
    inputs.insert(4, vec![10.0, 10.0]);
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
        .execute_descriptor(
            &descriptor,
            &add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]),
            &[3],
        )
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
fn cuda_fake_sequences_full_lifecycle_and_receipt() {
    let mut host = cuda_composite("addita").expect("cuda composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Cuda, "addita", 3);
    let receipt = host
        .execute_descriptor(
            &descriptor,
            &add_inputs(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]),
            &[3],
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
    let descriptor = DeviceDescriptor {
        backend: DeviceBackend::Metal,
        module_image: MODULE_IMAGE.to_vec(),
        kernels: vec![
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
    };
    let mut inputs = BTreeMap::new();
    inputs.insert(1, vec![1.0, 2.0]);
    inputs.insert(2, vec![3.0, 4.0]);
    inputs.insert(4, vec![10.0, 10.0]);

    let receipt = host
        .execute_descriptor(&descriptor, &inputs, &[5])
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
fn cpu_only_host_rejects_descriptor_execution() {
    let mut host = CompositeHost::new(CompositeHostConfig::cpu()).expect("cpu composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let err = host
        .execute_descriptor(&descriptor, &BTreeMap::new(), &[])
        .expect_err("cpu-only host must refuse device execution");
    assert_eq!(err.code, E_NO_DEVICE_PROGRAM);
}

#[test]
fn wrong_backend_descriptor_fails_closed() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Cuda, "add_one", 2);
    let err = host
        .execute_descriptor(&descriptor, &BTreeMap::new(), &[])
        .expect_err("cuda descriptor on a metal session must fail");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
}

#[test]
fn empty_module_image_is_a_bad_descriptor() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let mut descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    descriptor.module_image.clear();
    let err = host
        .execute_descriptor(&descriptor, &BTreeMap::new(), &[])
        .expect_err("empty module image must fail before launch");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
}

#[test]
fn descriptor_with_no_kernels_fails_closed() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let mut descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    descriptor.kernels.clear();
    let err = host
        .execute_descriptor(&descriptor, &BTreeMap::new(), &[])
        .expect_err("no kernels must fail before launch");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
}

#[test]
fn empty_kernel_entry_fails_closed() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let mut descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    descriptor.kernels[0].entry.clear();
    let err = host
        .execute_descriptor(&descriptor, &BTreeMap::new(), &[])
        .expect_err("empty entry must fail before launch");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
}

#[test]
fn zero_grid_axis_fails_closed() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let mut descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    descriptor.kernels[0].grid[1] = 0;
    let err = host
        .execute_descriptor(&descriptor, &BTreeMap::new(), &[])
        .expect_err("zero grid must fail before launch");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);
}

#[test]
fn duplicate_binding_fails_as_abi_mismatch() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let mut descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    descriptor.kernels[0].buffers[2].binding = 0; // collides with slot `a`
    let err = host
        .execute_descriptor(&descriptor, &BTreeMap::new(), &[])
        .expect_err("duplicate binding must fail as an ABI mismatch");
    assert_eq!(err.code, E_DEVICE_ABI_MISMATCH);
}

#[test]
fn conflicting_buffer_roles_fail_as_abi_mismatch() {
    let mut host = metal_composite("add_one").expect("metal composite");
    // The same buffer id is Input in one kernel and Output in another.
    let descriptor = DeviceDescriptor {
        backend: DeviceBackend::Metal,
        module_image: MODULE_IMAGE.to_vec(),
        kernels: vec![
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
    };
    let err = host
        .execute_descriptor(&descriptor, &BTreeMap::new(), &[])
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
        .execute_descriptor(&descriptor, &BTreeMap::new(), &[])
        .expect_err("dtype conflict must fail before launch");
    assert_eq!(err.code, E_DEVICE_DTYPE_MISMATCH);
}

#[test]
fn conflicting_shapes_fail_as_shape_mismatch() {
    let mut host = metal_composite("add_one").expect("metal composite");
    // Two kernels reference buffer id 3 with different element counts: a
    // shape change must be a new version, never in-place reinterpretation.
    let descriptor = DeviceDescriptor {
        backend: DeviceBackend::Metal,
        module_image: MODULE_IMAGE.to_vec(),
        kernels: vec![
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
    };
    let err = host
        .execute_descriptor(&descriptor, &BTreeMap::new(), &[])
        .expect_err("shape conflict must fail before launch");
    assert_eq!(err.code, E_DEVICE_SHAPE_MISMATCH);
}

#[test]
fn unknown_kernel_entry_fails_before_launch() {
    // The fake declares only `add_one`; the descriptor asks for `add_two`.
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_two", 2);
    let err = host
        .execute_descriptor(
            &descriptor,
            &add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]),
            &[3],
        )
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
        .execute_descriptor(&descriptor, &inputs, &[3])
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
            &[3],
        )
        .expect_err("input size vs declared shape must fail before launch");
    assert_eq!(err.code, E_DEVICE_SHAPE_MISMATCH);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

#[test]
fn declared_output_not_allocated_fails_closed() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let err = host
        .execute_descriptor(
            &descriptor,
            &add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]),
            &[99],
        )
        .expect_err("output id never allocated must fail closed");
    assert_eq!(err.code, "E_INVALID_ARGS");
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

/// The session loads the module once and allocates every buffer once at
/// creation; repeated `execute` calls on the same session do NOT reload the
/// module or re-allocate buffers; teardown releases every handle.
#[test]
fn program_session_executes_repeatedly_without_reload_or_realloc() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let inputs = add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]);

    // Create: module loaded once (1) + 3 PerProgram buffers allocated = 4 handles.
    let mut session = host
        .create_program_session(&descriptor)
        .expect("session create");
    assert_eq!(session.session_handle_count(), 4);
    assert_eq!(session.module_hash(), fnv1a64(MODULE_IMAGE));
    assert_eq!(session.allocated_buffers(), vec![1, 2, 3]);

    // Execute 1: no new handles (module + buffers reused).
    let receipt1 = session.execute(&inputs, &[3]).expect("execute 1");
    assert_eq!(receipt1.launches, 1);
    assert_eq!(receipt1.copy_ins, 2);
    assert_eq!(receipt1.outputs.get(&3), Some(&vec![4.0, 6.0]));
    assert_eq!(session.session_handle_count(), 4); // unchanged

    // Execute 2 on the SAME session: no reload, no realloc.
    let receipt2 = session
        .execute(&add_inputs(vec![10.0, 20.0], vec![30.0, 40.0]), &[3])
        .expect("execute 2");
    assert_eq!(receipt2.outputs.get(&3), Some(&vec![40.0, 60.0]));
    assert_eq!(session.session_handle_count(), 4); // still unchanged

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
    assert_eq!(session.session_handle_count(), 4); // module + 3 buffers

    let r1 = session
        .execute(&add_inputs(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]), &[3])
        .expect("execute 1");
    assert_eq!(r1.outputs.get(&3), Some(&vec![5.0, 7.0, 9.0]));
    assert_eq!(session.session_handle_count(), 4);

    let r2 = session
        .execute(&add_inputs(vec![10.0, 20.0, 30.0], vec![1.0, 2.0, 3.0]), &[3])
        .expect("execute 2");
    assert_eq!(r2.outputs.get(&3), Some(&vec![11.0, 22.0, 33.0]));
    assert_eq!(session.session_handle_count(), 4);

    session.teardown().expect("teardown");
    assert_eq!(
        host.device().expect("device").live_handle_count(),
        0
    );
}

/// A two-kernel InOut chain through the session API: the intermediate buffer
/// stays device-resident across kernels within one step, and the session
/// reuses it across steps without re-allocation.
#[test]
fn program_session_two_kernel_chain_reuses_buffers_across_steps() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = DeviceDescriptor {
        backend: DeviceBackend::Metal,
        module_image: MODULE_IMAGE.to_vec(),
        kernels: vec![
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
    };
    let mut inputs = BTreeMap::new();
    inputs.insert(1, vec![1.0, 2.0]);
    inputs.insert(2, vec![3.0, 4.0]);
    inputs.insert(4, vec![10.0, 10.0]);

    let mut session = host
        .create_program_session(&descriptor)
        .expect("session create");
    // Module + 5 distinct buffers (ids 1-5).
    assert_eq!(session.session_handle_count(), 6);
    assert_eq!(session.allocated_buffers(), vec![1, 2, 3, 4, 5]);

    let receipt = session.execute(&inputs, &[5]).expect("execute");
    assert_eq!(receipt.launches, 2);
    assert_eq!(receipt.copy_ins, 3); // only Input slots (a, b, c)
    assert_eq!(receipt.outputs.get(&5), Some(&vec![14.0, 16.0]));
    assert_eq!(session.session_handle_count(), 6); // no realloc

    // Second step: same session, same buffers.
    let receipt2 = session.execute(&inputs, &[5]).expect("execute 2");
    assert_eq!(receipt2.outputs.get(&5), Some(&vec![14.0, 16.0]));
    assert_eq!(session.session_handle_count(), 6);

    session.teardown().expect("teardown");
    assert_eq!(
        host.device().expect("device").live_handle_count(),
        0
    );
}

/// `execute_descriptor` remains a single-run convenience over the session:
/// create → execute → teardown in one call, same receipt shape as S1-4.
#[test]
fn execute_descriptor_is_single_run_convenience_over_session() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let receipt = host
        .execute_descriptor(
            &descriptor,
            &add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]),
            &[3],
        )
        .expect("execute_descriptor");

    assert_eq!(receipt.backend, DeviceBackend::Metal);
    assert_eq!(receipt.module_hash, fnv1a64(MODULE_IMAGE));
    assert_eq!(receipt.launches, 1);
    assert_eq!(receipt.copy_ins, 2);
    assert_eq!(receipt.outputs.get(&3), Some(&vec![4.0, 6.0]));
    assert_eq!(receipt.allocated_buffers, vec![1, 2, 3]);
    // Single-run convenience tears down internally.
    assert_eq!(
        host.device().expect("device").live_handle_count(),
        0
    );
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
/// "module persists across steps".
#[test]
fn module_cache_loads_once_and_releases_on_teardown() {
    let mut host = metal_composite("add_one").expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let inputs = add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]);

    let mut session = host
        .create_program_session(&descriptor)
        .expect("session create");

    // Session creation loaded the module once (plus one alloc per buffer).
    let counters = session.driver_counters();
    assert_eq!(counters.module_loads, 1);
    assert_eq!(counters.buffer_allocs, 3);
    // One live module: loaded, not yet released — the only persistence the
    // policy allows is the session keeping the module alive for the program.
    assert_eq!(counters.module_loads - counters.module_releases, 1);

    // N repeated executions: still one load, no reload, no re-alloc.
    for _ in 0..5 {
        let receipt = session.execute(&inputs, &[3]).expect("execute");
        assert_eq!(receipt.outputs.get(&3), Some(&vec![4.0, 6.0]));
        let counters = session.driver_counters();
        assert_eq!(counters.module_loads, 1);
        assert_eq!(counters.buffer_allocs, 3);
    }

    // Teardown releases the module exactly once; buffers released too.
    session.teardown().expect("teardown");
    let counters = host.device().expect("device").driver_counters();
    assert_eq!(counters.module_loads, 1);
    assert_eq!(counters.module_releases, 1);
    assert_eq!(counters.buffer_allocs, 3);
    assert_eq!(counters.buffer_releases, 3);
    // Nothing persists past teardown: no live module, no live handles.
    assert_eq!(counters.module_loads, counters.module_releases);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

/// The same leak-free bar on CUDA (backend-neutral surface): one module
/// load, one release, no module persists past teardown.
#[test]
fn module_cache_loads_once_and_releases_on_teardown_cuda() {
    let mut host = cuda_composite("addita").expect("cuda composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Cuda, "addita", 3);
    let inputs = add_inputs(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]);

    let mut session = host
        .create_program_session(&descriptor)
        .expect("session create");
    assert_eq!(session.driver_counters().module_loads, 1);

    for _ in 0..5 {
        let receipt = session.execute(&inputs, &[3]).expect("execute");
        assert_eq!(receipt.outputs.get(&3), Some(&vec![5.0, 7.0, 9.0]));
        assert_eq!(session.driver_counters().module_loads, 1);
    }

    session.teardown().expect("teardown");
    let counters = host.device().expect("device").driver_counters();
    assert_eq!(counters.module_loads, 1);
    assert_eq!(counters.module_releases, 1);
    assert_eq!(counters.module_loads, counters.module_releases);
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
    session1.execute(&inputs, &[3]).expect("session 1 execute");
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
    session2.execute(&inputs, &[3]).expect("session 2 execute");
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
    host.execute_descriptor(
        &descriptor,
        &add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]),
        &[3],
    )
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
    let mut host = metal_composite_failing("add_one", FakeFailureStage::Alloc, 2)
        .expect("metal composite");
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
    let mut host = cuda_composite_failing("addita", FakeFailureStage::ModuleLoad, 1)
        .expect("cuda composite");
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
    let mut host = cuda_composite_failing("addita", FakeFailureStage::Alloc, 2)
        .expect("cuda composite");
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
    let mut host = metal_composite_failing("add_one", FakeFailureStage::CopyIn, 1)
        .expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let err = host
        .execute_descriptor(
            &descriptor,
            &add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]),
            &[3],
        )
        .expect_err("injected copy-in failure must fail the execution");
    assert_eq!(err.code, E_METAL_DRIVER);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

#[test]
fn metal_launch_failure_releases_every_handle() {
    let mut host = metal_composite_failing("add_one", FakeFailureStage::Launch, 1)
        .expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let err = host
        .execute_descriptor(
            &descriptor,
            &add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]),
            &[3],
        )
        .expect_err("injected launch failure must fail the execution");
    assert_eq!(err.code, E_METAL_DRIVER);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

#[test]
fn metal_sync_failure_releases_every_handle() {
    // The driver syncs once inside each launch and once at the step boundary;
    // sync call 2 is the explicit step-boundary barrier in `execute`.
    let mut host = metal_composite_failing("add_one", FakeFailureStage::Sync, 2)
        .expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let err = host
        .execute_descriptor(
            &descriptor,
            &add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]),
            &[3],
        )
        .expect_err("injected sync failure must fail the execution");
    assert_eq!(err.code, E_METAL_DRIVER);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

#[test]
fn metal_readback_failure_releases_every_handle() {
    let mut host = metal_composite_failing("add_one", FakeFailureStage::Readback, 1)
        .expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let err = host
        .execute_descriptor(
            &descriptor,
            &add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]),
            &[3],
        )
        .expect_err("injected readback failure must fail the execution");
    assert_eq!(err.code, E_METAL_DRIVER);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

#[test]
fn cuda_copy_in_failure_releases_every_handle() {
    let mut host = cuda_composite_failing("addita", FakeFailureStage::CopyIn, 1)
        .expect("cuda composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Cuda, "addita", 3);
    let err = host
        .execute_descriptor(
            &descriptor,
            &add_inputs(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]),
            &[3],
        )
        .expect_err("injected copy-in failure must fail the execution");
    assert_eq!(err.code, E_CUDA_DRIVER);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

#[test]
fn cuda_launch_failure_releases_every_handle() {
    let mut host = cuda_composite_failing("addita", FakeFailureStage::Launch, 1)
        .expect("cuda composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Cuda, "addita", 3);
    let err = host
        .execute_descriptor(
            &descriptor,
            &add_inputs(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]),
            &[3],
        )
        .expect_err("injected launch failure must fail the execution");
    assert_eq!(err.code, E_CUDA_DRIVER);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

#[test]
fn cuda_sync_failure_releases_every_handle() {
    // Same call sequence as the Metal lane: sync call 2 is the step boundary.
    let mut host = cuda_composite_failing("addita", FakeFailureStage::Sync, 2)
        .expect("cuda composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Cuda, "addita", 3);
    let err = host
        .execute_descriptor(
            &descriptor,
            &add_inputs(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]),
            &[3],
        )
        .expect_err("injected sync failure must fail the execution");
    assert_eq!(err.code, E_CUDA_DRIVER);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

#[test]
fn cuda_readback_failure_releases_every_handle() {
    let mut host = cuda_composite_failing("addita", FakeFailureStage::Readback, 1)
        .expect("cuda composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Cuda, "addita", 3);
    let err = host
        .execute_descriptor(
            &descriptor,
            &add_inputs(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]),
            &[3],
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
    let mut host = metal_composite_failing("add_one", FakeFailureStage::Launch, 2)
        .expect("metal composite");
    let descriptor = two_kernel_inout_descriptor();
    let err = host
        .execute_descriptor(&descriptor, &two_kernel_inputs(), &[5])
        .expect_err("injected second-launch failure must fail the execution");
    assert_eq!(err.code, E_METAL_DRIVER);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

/// A failed execution closes the session: every handle is released, the
/// session reports 0 handles, and a closed session refuses further execution
/// instead of reusing stale handles.
#[test]
fn failed_execution_closes_session_and_blocks_reuse() {
    let mut host = metal_composite_failing("add_one", FakeFailureStage::CopyIn, 1)
        .expect("metal composite");
    let descriptor = elementwise_add_descriptor(DeviceBackend::Metal, "add_one", 2);
    let inputs = add_inputs(vec![1.0, 2.0], vec![3.0, 4.0]);

    let mut session = host
        .create_program_session(&descriptor)
        .expect("session create");
    assert_eq!(session.session_handle_count(), 4); // module + 3 buffers

    let err = session
        .execute(&inputs, &[3])
        .expect_err("injected copy-in failure must fail the execution");
    assert_eq!(err.code, E_METAL_DRIVER);
    // Release-on-error: no handle survives the failed execution at the
    // session level (the host-level count is checked after the session is
    // consumed, once the session's borrow of the host has ended).
    assert_eq!(session.session_handle_count(), 0);

    // A closed session cannot execute again (no stale-handle reuse).
    let again = session
        .execute(&inputs, &[3])
        .expect_err("a closed session must refuse execution");
    assert_eq!(again.code, "E_INTERNAL");

    // Teardown of an already-closed session is a safe no-op (no double
    // release).
    session.teardown().expect("teardown of closed session");
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}
