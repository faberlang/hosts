//! KV-B B3: allocation/view split on the host device descriptor.
//!
//! Allocation capacity and view extent are separate typed facts. Two
//! persistent K/V arenas expose append and prefix views without copying.
//! Runtime cursor values never join the graph hash. Declared binding
//! indices and order are the launch records.

use faber_host_macos_arm64::cuda_host::{E_CUDA_UNSUPPORTED, FakeCudaDriver};
use faber_host_macos_arm64::device_descriptor::{
    DescriptorAllocation, DescriptorBuffer, DescriptorBufferVersion, DescriptorInvocationState,
    DescriptorKernel, DescriptorLaunch, DescriptorLaunchBinding, DescriptorResult,
    DescriptorRuntimeSource, DescriptorView, DeviceBufferInitialization, DeviceBufferLifetime,
    DeviceBufferRole, DeviceDataType, DeviceDescriptor, DeviceProgramLifetime,
    E_DEVICE_ABI_MISMATCH, E_DEVICE_SHAPE_MISMATCH, KvCacheDescriptor,
};
use faber_host_macos_arm64::device_host::{
    DeviceLaunchBinding, DeviceRuntime, DeviceSession, InvocationStateBuffer,
    handles_in_binding_order, resolve_launch_bindings, validate_launch_bindings,
};
use faber_host_macos_arm64::{CudaHostSession, FakeMetalDriver, MetalHostSession};
use host_coordinator::DeviceBackend;

/// Layers × KV-heads × positions × dim for the persistent arenas.
const LAYERS: u64 = 2;
const KV_HEADS: u64 = 2;
const CAPACITY_POSITIONS: u64 = 8;
const HEAD_DIM: u64 = 4;
const F32_WIDTH: u64 = 4;

const K_ALLOCATION: u32 = 1;
const V_ALLOCATION: u32 = 2;

fn arena_capacity_bytes() -> u64 {
    LAYERS * KV_HEADS * CAPACITY_POSITIONS * HEAD_DIM * F32_WIDTH
}

fn append_span_bytes() -> u64 {
    LAYERS * KV_HEADS * HEAD_DIM * F32_WIDTH
}

/// Element strides for `[layer, kv_head, position, dim]`.
fn arena_strides(positions: u64) -> Vec<u64> {
    let dim = 1;
    let position = HEAD_DIM;
    let kv_head = positions * HEAD_DIM;
    let layer = KV_HEADS * positions * HEAD_DIM;
    vec![layer, kv_head, position, dim]
}

fn persistent_arena(buffer_id: u32) -> DescriptorAllocation {
    DescriptorAllocation {
        buffer_id,
        dtype: DeviceDataType::F32,
        capacity_bytes: arena_capacity_bytes(),
        lifetime: DeviceBufferLifetime::PerProgram,
        initialization: DeviceBufferInitialization::ZeroFill,
    }
}

fn prefix_view(allocation_id: u32) -> DescriptorView {
    DescriptorView {
        allocation_id,
        logical_dims: vec![LAYERS, KV_HEADS, CAPACITY_POSITIONS, HEAD_DIM],
        strides: arena_strides(CAPACITY_POSITIONS),
        static_base: 0,
        maximum_span: arena_capacity_bytes(),
    }
}

fn append_view(allocation_id: u32) -> DescriptorView {
    DescriptorView {
        allocation_id,
        logical_dims: vec![LAYERS, KV_HEADS, 1, HEAD_DIM],
        strides: arena_strides(CAPACITY_POSITIONS),
        static_base: 0,
        maximum_span: append_span_bytes(),
    }
}

/// Bindings in a deliberately non-monotonic index order so preservation is
/// visible: K-append@2, K-prefix@0, V-append@3, V-prefix@1.
fn declared_launch_bindings() -> Vec<DescriptorLaunchBinding> {
    vec![
        DescriptorLaunchBinding {
            handle: K_ALLOCATION,
            binding_index: 2,
            byte_offset: 0,
            view_span: append_span_bytes(),
            runtime_source: DescriptorRuntimeSource::Position,
        },
        DescriptorLaunchBinding {
            handle: K_ALLOCATION,
            binding_index: 0,
            byte_offset: 0,
            view_span: arena_capacity_bytes(),
            runtime_source: DescriptorRuntimeSource::ValidLenAfter,
        },
        DescriptorLaunchBinding {
            handle: V_ALLOCATION,
            binding_index: 3,
            byte_offset: 0,
            view_span: append_span_bytes(),
            runtime_source: DescriptorRuntimeSource::Position,
        },
        DescriptorLaunchBinding {
            handle: V_ALLOCATION,
            binding_index: 1,
            byte_offset: 0,
            view_span: arena_capacity_bytes(),
            runtime_source: DescriptorRuntimeSource::ValidLenAfter,
        },
    ]
}

fn base_kv() -> KvCacheDescriptor {
    KvCacheDescriptor {
        allocations: vec![
            persistent_arena(K_ALLOCATION),
            persistent_arena(V_ALLOCATION),
        ],
        views: vec![
            prefix_view(K_ALLOCATION),
            append_view(K_ALLOCATION),
            prefix_view(V_ALLOCATION),
            append_view(V_ALLOCATION),
        ],
        invocation_state: DescriptorInvocationState {
            position: 3,
            valid_len_after: 4,
            query_rows: 1,
            sequence_epoch: 7,
        },
        launch_bindings: declared_launch_bindings(),
    }
}

fn slot(id: u32, name: &str, role: DeviceBufferRole, binding: u32) -> DescriptorBuffer {
    DescriptorBuffer {
        buffer_id: id,
        buffer_name: name.to_owned(),
        semantic_value: id,
        role,
        lifetime: match role {
            DeviceBufferRole::Input => DeviceBufferLifetime::PerProgram,
            DeviceBufferRole::Output => DeviceBufferLifetime::ObservationPoint,
            DeviceBufferRole::InOut => DeviceBufferLifetime::PerStep,
        },
        initialization: match role {
            DeviceBufferRole::Input => DeviceBufferInitialization::HostProvided,
            DeviceBufferRole::InOut => DeviceBufferInitialization::ZeroFill,
            DeviceBufferRole::Output => DeviceBufferInitialization::KernelInitialized,
        },
        binding,
        element_ty: DeviceDataType::F32,
        element_count: 2,
        version: 1,
    }
}

fn base_device_descriptor() -> DeviceDescriptor {
    DeviceDescriptor {
        backend: DeviceBackend::Metal,
        module_image: b"// fake compiler-owned module image".to_vec(),
        kernels: vec![DescriptorKernel {
            entry: "kv_step".to_owned(),
            buffers: vec![
                slot(1, "q", DeviceBufferRole::Input, 0),
                slot(2, "out", DeviceBufferRole::Output, 1),
            ],
            grid: [1, 1, 1],
            block: [1, 1, 1],
        }],
        launches: vec![DescriptorLaunch {
            id: 1,
            kernel_index: 0,
        }],
        buffer_versions: vec![
            DescriptorBufferVersion {
                buffer_id: 1,
                version: 1,
                element_ty: DeviceDataType::F32,
                element_count: 2,
            },
            DescriptorBufferVersion {
                buffer_id: 2,
                version: 1,
                element_ty: DeviceDataType::F32,
                element_count: 2,
            },
        ],
        program_lifetime: DeviceProgramLifetime::SingleRun,
        data_flow: vec![],
        roots: vec![1],
        results: vec![DescriptorResult {
            buffer_id: 2,
            version: 1,
            produced_by: 1,
            at_launch: 1,
        }],
        end_of_run_results: vec![],
    }
}

#[test]
fn multiple_views_share_one_allocation_for_k_and_v_arenas() {
    let kv = base_kv();
    kv.validate().expect("shared K/V views must admit");

    let k_views: Vec<&DescriptorView> = kv
        .views
        .iter()
        .filter(|view| view.allocation_id == K_ALLOCATION)
        .collect();
    let v_views: Vec<&DescriptorView> = kv
        .views
        .iter()
        .filter(|view| view.allocation_id == V_ALLOCATION)
        .collect();
    assert_eq!(k_views.len(), 2, "K prefix and append share the K arena");
    assert_eq!(v_views.len(), 2, "V prefix and append share the V arena");
    assert_eq!(kv.allocations.len(), 2);
    assert_ne!(
        k_views[0].maximum_span, k_views[1].maximum_span,
        "append and prefix extents differ over the same allocation"
    );
}

#[test]
fn allocation_capacity_and_view_extent_are_separate_facts() {
    let kv = base_kv();
    let k = kv
        .allocations
        .iter()
        .find(|allocation| allocation.buffer_id == K_ALLOCATION)
        .expect("K arena");
    let k_append = kv
        .views
        .iter()
        .find(|view| view.allocation_id == K_ALLOCATION && view.maximum_span == append_span_bytes())
        .expect("K append view");
    assert_eq!(k.capacity_bytes, arena_capacity_bytes());
    assert_eq!(k_append.maximum_span, append_span_bytes());
    assert_ne!(
        k.capacity_bytes, k_append.maximum_span,
        "allocation capacity must not be the view extent"
    );

    let baseline = kv.program_graph_hash();
    let mut capacity_only = base_kv();
    capacity_only.allocations[0].capacity_bytes *= 2;
    capacity_only
        .validate()
        .expect("grown K arena still admits");
    assert_eq!(
        capacity_only.views[1].maximum_span, kv.views[1].maximum_span,
        "growing capacity must leave view extent unchanged"
    );
    assert_ne!(
        baseline,
        capacity_only.program_graph_hash(),
        "changing allocation capacity must change the static graph hash"
    );

    let mut extent_only = base_kv();
    extent_only.views[1].maximum_span = append_span_bytes() * 2;
    extent_only.launch_bindings[0].view_span = append_span_bytes() * 2;
    extent_only
        .validate()
        .expect("wider append still fits the arena");
    assert_ne!(
        baseline,
        extent_only.program_graph_hash(),
        "changing view extent must change the static graph hash without changing capacity"
    );
    assert_eq!(
        extent_only.allocations[0].capacity_bytes,
        kv.allocations[0].capacity_bytes
    );
}

#[test]
fn runtime_cursor_values_do_not_affect_graph_hash() {
    let kv = base_kv();
    let descriptor = base_device_descriptor();
    let kv_hash = kv.program_graph_hash();
    let combined = descriptor.program_graph_hash_with_kv(&kv);

    let mut cursor = base_kv();
    cursor.invocation_state.position = 99;
    cursor.invocation_state.valid_len_after = 100;
    cursor.invocation_state.query_rows = 8;
    cursor.invocation_state.sequence_epoch = 42;
    cursor
        .validate()
        .expect("cursor mutation is not a plan error");
    assert_eq!(
        kv_hash,
        cursor.program_graph_hash(),
        "runtime cursor values must not enter the KV graph hash"
    );
    assert_eq!(
        combined,
        descriptor.program_graph_hash_with_kv(&cursor),
        "runtime cursor values must not enter the descriptor graph hash"
    );
}

#[test]
fn binding_indices_and_order_are_preserved_through_launch_records() {
    let kv = base_kv();
    kv.validate().expect("fixture admits");
    let records = kv.launch_records();
    assert_eq!(records, declared_launch_bindings().as_slice());
    assert_eq!(
        records
            .iter()
            .map(|binding| binding.binding_index)
            .collect::<Vec<_>>(),
        vec![2, 0, 3, 1],
        "declared binding indices must not be sorted or dropped"
    );
    assert_eq!(records[0].handle, K_ALLOCATION);
    assert_eq!(records[2].handle, V_ALLOCATION);

    let mut reordered = base_kv();
    reordered.launch_bindings.swap(0, 1);
    assert_ne!(
        kv.program_graph_hash(),
        reordered.program_graph_hash(),
        "launch-record order is part of the static binding expression"
    );
    assert_eq!(
        reordered.launch_records()[0].binding_index,
        0,
        "launch_records must follow the declared vec, not a canonical sort"
    );
}

#[test]
fn static_binding_source_tags_join_the_graph_hash() {
    let baseline = base_kv().program_graph_hash();
    let mut tagged = base_kv();
    tagged.launch_bindings[0].runtime_source = DescriptorRuntimeSource::Constant;
    tagged.launch_bindings[0].binding_index = 7;
    assert_ne!(
        baseline,
        tagged.program_graph_hash(),
        "binding expressions (index and runtime-source tag) join the static hash"
    );
    tagged.invocation_state.position = 0;
    assert_ne!(
        baseline,
        tagged.program_graph_hash(),
        "a source-tag change must remain visible after cursor mutation"
    );
}

#[test]
fn view_extent_beyond_allocation_capacity_fails_before_launch() {
    let mut kv = base_kv();
    kv.views[1].maximum_span = arena_capacity_bytes() + 4;
    let err = kv
        .validate()
        .expect_err("a view larger than its allocation must fail closed");
    assert_eq!(err.code, E_DEVICE_SHAPE_MISMATCH);
}

fn fake_metal() -> DeviceRuntime {
    DeviceRuntime::Metal(
        MetalHostSession::with_driver(Box::new(FakeMetalDriver::default())).expect("fake metal"),
    )
}

fn fake_cuda() -> DeviceRuntime {
    DeviceRuntime::Cuda(
        CudaHostSession::with_driver(Box::new(FakeCudaDriver::default())).expect("fake cuda"),
    )
}

fn live_arenas(runtime: &mut DeviceRuntime) -> [(u32, host_coordinator::DeviceHandle); 2] {
    let capacity = arena_capacity_bytes() as usize;
    let k = runtime.alloc_bytes(capacity).expect("K arena");
    let v = runtime.alloc_bytes(capacity).expect("V arena");
    [(K_ALLOCATION, k), (V_ALLOCATION, v)]
}

#[test]
fn launch_bindings_carry_handle_index_offset_span_and_source() {
    let mut runtime = fake_metal();
    let map = live_arenas(&mut runtime);
    let kv = base_kv();
    let bindings = resolve_launch_bindings(&kv, &map).expect("B3 records resolve");
    assert_eq!(bindings.len(), 4);
    assert_eq!(
        bindings
            .iter()
            .map(|binding| {
                (
                    binding.handle,
                    binding.binding_index,
                    binding.byte_offset,
                    binding.view_span,
                    binding.runtime_source,
                )
            })
            .collect::<Vec<_>>(),
        vec![
            (
                map[0].1,
                2,
                0,
                append_span_bytes(),
                DescriptorRuntimeSource::Position
            ),
            (
                map[0].1,
                0,
                0,
                arena_capacity_bytes(),
                DescriptorRuntimeSource::ValidLenAfter
            ),
            (
                map[1].1,
                3,
                0,
                append_span_bytes(),
                DescriptorRuntimeSource::Position
            ),
            (
                map[1].1,
                1,
                0,
                arena_capacity_bytes(),
                DescriptorRuntimeSource::ValidLenAfter
            ),
        ]
    );
}

#[test]
fn handles_in_binding_order_keep_declared_indices() {
    let mut runtime = fake_metal();
    let map = live_arenas(&mut runtime);
    let bindings = resolve_launch_bindings(&base_kv(), &map).expect("resolve");
    let ordered = handles_in_binding_order(&bindings).expect("dense indices");
    assert_eq!(
        ordered,
        vec![map[0].1, map[1].1, map[0].1, map[1].1],
        "slot i is the binding whose declared index is i"
    );

    let mut dropped = bindings;
    dropped.pop();
    let err = handles_in_binding_order(&dropped)
        .expect_err("a missing index must not be dropped into a packed slice");
    assert_eq!(err.code, E_DEVICE_ABI_MISMATCH);
}

#[test]
fn offset_that_matches_no_view_fails_before_dispatch() {
    let mut runtime = fake_metal();
    let map = live_arenas(&mut runtime);
    let module = runtime.load_module(b"// unused").expect("module");
    let mut kv = base_kv();
    kv.launch_bindings[0].byte_offset = 4;
    kv.validate()
        .expect("offset 4 still fits allocation capacity");
    let err = resolve_launch_bindings(&kv, &map).expect_err("offset 4 is not a view static_base");
    assert_eq!(err.code, E_DEVICE_SHAPE_MISMATCH);

    let err = runtime
        .launch_kv_kernel(&module, "kv_step", &kv, &map, [1, 1, 1], [1, 1, 1])
        .expect_err("dispatch must not run after a view mismatch");
    assert_eq!(err.code, E_DEVICE_SHAPE_MISMATCH);
}

#[test]
fn span_beyond_every_view_on_the_allocation_fails_before_dispatch() {
    let mut runtime = fake_metal();
    let mut kv = base_kv();
    kv.views
        .retain(|view| view.maximum_span == append_span_bytes());
    kv.launch_bindings
        .retain(|binding| binding.view_span == append_span_bytes());
    kv.validate().expect("append-only plan admits");

    let map = live_arenas(&mut runtime);
    let mut bindings = resolve_launch_bindings(&kv, &map).expect("append views resolve");
    bindings[0].view_span = append_span_bytes() * 2;
    let err = validate_launch_bindings(&kv, &bindings, &map)
        .expect_err("wider span than every remaining view");
    assert_eq!(err.code, E_DEVICE_SHAPE_MISMATCH);
}

#[test]
fn cursor_buffer_is_allocated_once_and_reused_across_steps() {
    let mut runtime = fake_metal();
    let buffer = runtime
        .alloc_invocation_state()
        .expect("one invocation-state allocation");
    let live_after_alloc = runtime.live_handle_count();
    let handle = buffer.handle();

    let first = DescriptorInvocationState {
        position: 3,
        valid_len_after: 4,
        query_rows: 1,
        sequence_epoch: 7,
    };
    runtime
        .upload_invocation_state(&buffer, first)
        .expect("first step upload");
    assert_eq!(runtime.live_handle_count(), live_after_alloc);
    assert_eq!(buffer.handle(), handle);

    let second = DescriptorInvocationState {
        position: 4,
        valid_len_after: 5,
        query_rows: 1,
        sequence_epoch: 7,
    };
    runtime
        .upload_invocation_state(&buffer, second)
        .expect("second step overwrites the same buffer");
    assert_eq!(runtime.live_handle_count(), live_after_alloc);
    assert_eq!(buffer.handle(), handle);
    assert_eq!(
        handle.len_bytes(),
        Some(InvocationStateBuffer::BYTE_LENGTH as u64)
    );

    let words = runtime.readback_f32(&handle).expect("typed upload");
    assert_eq!(words[0].to_bits(), second.position);
    assert_eq!(words[1].to_bits(), second.valid_len_after);
    assert_eq!(words[2].to_bits(), second.query_rows);
    assert_eq!(words[3].to_bits(), second.sequence_epoch);
}

#[test]
fn dynamic_cuda_descriptor_rejects_explicitly_instead_of_offset_zero() {
    let mut runtime = fake_cuda();
    let map = live_arenas(&mut runtime);
    let bindings = resolve_launch_bindings(&base_kv(), &map).expect("resolve");
    assert!(
        bindings.iter().any(DeviceLaunchBinding::is_cuda_dynamic),
        "the KV fixture carries runtime sources"
    );
    let module = runtime
        .load_module(b"// fake compiler-owned image bytes")
        .expect("load");
    let err = runtime
        .launch_kernel_bound(&module, "addita", &bindings, [1, 1, 1], [1, 1, 1])
        .expect_err("CUDA must reject KV-dynamic bindings");
    assert_eq!(err.code, E_CUDA_UNSUPPORTED);
    assert!(
        err.message.contains("does not bind it at offset zero"),
        "rejection must not be a silent offset-zero bind: {}",
        err.message
    );

    let mut constant_offset = DeviceLaunchBinding::whole_handle(map[0].1, 0).expect("whole");
    constant_offset.byte_offset = append_span_bytes();
    constant_offset.view_span = append_span_bytes();
    let err = runtime
        .launch_kernel_bound(&module, "addita", &[constant_offset], [1, 1, 1], [1, 1, 1])
        .expect_err("CUDA must reject a nonzero constant offset");
    assert_eq!(err.code, E_CUDA_UNSUPPORTED);
}

#[test]
fn legacy_whole_handle_offset_zero_wrapper_stays_green_on_metal() {
    let mut runtime = fake_metal();
    let module = runtime
        .load_module(b"// fake compiler-owned image bytes")
        .expect("load");
    let a = runtime.alloc_bytes(8).expect("a");
    let b = runtime.alloc_bytes(8).expect("b");
    let out = runtime.alloc_bytes(8).expect("out");
    runtime.copy_in_f32(&a, &[1.0, 2.0]).expect("copy a");
    runtime.copy_in_f32(&b, &[3.0, 4.0]).expect("copy b");
    runtime
        .launch_kernel(&module, "add_one", &[a, b, out], [1, 1, 1], [8, 1, 1])
        .expect("legacy whole-handle wrapper");
    let values = runtime.readback_f32(&out).expect("readback");
    assert_eq!(values, vec![4.0, 6.0]);

    let bindings = [
        DeviceLaunchBinding::whole_handle(a, 0).expect("a"),
        DeviceLaunchBinding::whole_handle(b, 1).expect("b"),
        DeviceLaunchBinding::whole_handle(out, 2).expect("out"),
    ];
    assert!(bindings.iter().all(|binding| {
        binding.byte_offset == 0
            && binding.runtime_source == DescriptorRuntimeSource::Constant
            && !DeviceLaunchBinding::is_cuda_dynamic(binding)
    }));
}

#[test]
fn cuda_constant_offset_zero_wrapper_still_launches() {
    let mut runtime = fake_cuda();
    let module = runtime
        .load_module(b"// fake compiler-owned image bytes")
        .expect("load");
    let a = runtime.alloc_bytes(8).expect("a");
    let b = runtime.alloc_bytes(8).expect("b");
    let out = runtime.alloc_bytes(8).expect("out");
    runtime.copy_in_f32(&a, &[1.0, 2.0]).expect("copy a");
    runtime.copy_in_f32(&b, &[3.0, 4.0]).expect("copy b");
    runtime
        .launch_kernel(&module, "addita", &[a, b, out], [2, 2, 1], [8, 8, 1])
        .expect("CUDA static offset-zero wrapper");
    let values = runtime.readback_f32(&out).expect("readback");
    assert_eq!(values, vec![4.0, 6.0]);

    runtime
        .launch_kernel_bound(
            &module,
            "addita",
            &[
                DeviceLaunchBinding::whole_handle(a, 0).expect("a"),
                DeviceLaunchBinding::whole_handle(b, 1).expect("b"),
                DeviceLaunchBinding::whole_handle(out, 2).expect("out"),
            ],
            [2, 2, 1],
            [8, 8, 1],
        )
        .expect("CUDA constant offset-zero bound launch");
}

#[test]
fn dispatch_rejects_offset_past_handle_capacity_before_backend() {
    let mut runtime = fake_metal();
    let module = runtime
        .load_module(b"// fake compiler-owned image bytes")
        .expect("load");
    let a = runtime.alloc_bytes(8).expect("a");
    let b = runtime.alloc_bytes(8).expect("b");
    let out = runtime.alloc_bytes(8).expect("out");
    let mut over = DeviceLaunchBinding::whole_handle(a, 0).expect("a");
    over.byte_offset = 8;
    over.view_span = 8;
    let err = runtime
        .launch_kernel_bound(
            &module,
            "add_one",
            &[
                over,
                DeviceLaunchBinding::whole_handle(b, 1).expect("b"),
                DeviceLaunchBinding::whole_handle(out, 2).expect("out"),
            ],
            [1, 1, 1],
            [8, 1, 1],
        )
        .expect_err("offset past the allocation must fail closed");
    assert_eq!(err.code, E_DEVICE_SHAPE_MISMATCH);
}
