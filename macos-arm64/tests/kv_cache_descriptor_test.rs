//! KV-B B3: allocation/view split on the host device descriptor.
//!
//! Allocation capacity and view extent are separate typed facts. Two
//! persistent K/V arenas expose append and prefix views without copying.
//! Runtime cursor values never join the graph hash. Declared binding
//! indices and order are the launch records.

use faber_host_macos_arm64::device_descriptor::{
    DescriptorAllocation, DescriptorBuffer, DescriptorBufferVersion, DescriptorInvocationState,
    DescriptorKernel, DescriptorLaunch, DescriptorLaunchBinding, DescriptorResult,
    DescriptorRuntimeSource, DescriptorView, DeviceBufferInitialization, DeviceBufferLifetime,
    DeviceBufferRole, DeviceDataType, DeviceDescriptor, DeviceProgramLifetime, KvCacheDescriptor,
    E_DEVICE_SHAPE_MISMATCH,
};
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
