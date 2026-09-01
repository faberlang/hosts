//! KV-B B7: generic session materializer.
//!
//! One allocation per arena exposes append and prefix views. One cursor copy
//! per step is shared by append and attention launches. Changing live offsets
//! reuses stable handles: zero cache copies, zero persistent reallocations,
//! zero weight re-uploads. Offsets reach Metal on the B6 bound-launch path.

use faber_host_macos_arm64::composite_host::KvCacheBindingSession;
use faber_host_macos_arm64::device_descriptor::{
    DescriptorAllocation, DescriptorInvocationState, DescriptorLaunchBinding,
    DescriptorRuntimeSource, DescriptorView, DeviceBufferInitialization, DeviceBufferLifetime,
    DeviceDataType, E_DEVICE_SHAPE_MISMATCH, KvCacheDescriptor,
};
use faber_host_macos_arm64::device_host::{DeviceLaunchBinding, DeviceRuntime};
use faber_host_macos_arm64::{FakeMetalDriver, MetalHostSession};
use host_coordinator::DeviceHandle;

const POSITIONS: u64 = 4;
const HEAD_DIM: u64 = 4;
const F32_WIDTH: u64 = 4;
const K_ALLOCATION: u32 = 1;
const V_ALLOCATION: u32 = 2;
const WEIGHT_ALLOCATION: u32 = 10;
const WEIGHT_VALUES: [f32; 4] = [9.0, 8.0, 7.0, 6.0];

fn row_bytes() -> u64 {
    HEAD_DIM * F32_WIDTH
}

fn arena_capacity_bytes() -> u64 {
    POSITIONS * row_bytes()
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
        logical_dims: vec![POSITIONS, HEAD_DIM],
        strides: vec![HEAD_DIM, 1],
        static_base: 0,
        maximum_span: arena_capacity_bytes(),
    }
}

fn append_view(allocation_id: u32) -> DescriptorView {
    DescriptorView {
        allocation_id,
        logical_dims: vec![1, HEAD_DIM],
        strides: vec![HEAD_DIM, 1],
        static_base: 0,
        maximum_span: row_bytes(),
    }
}

fn declared_launch_bindings() -> Vec<DescriptorLaunchBinding> {
    vec![
        DescriptorLaunchBinding {
            handle: K_ALLOCATION,
            binding_index: 2,
            byte_offset: 0,
            view_span: row_bytes(),
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
            view_span: row_bytes(),
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
        invocation_state: DescriptorInvocationState::default(),
        launch_bindings: declared_launch_bindings(),
    }
}

fn weight_allocation() -> DescriptorAllocation {
    DescriptorAllocation {
        buffer_id: WEIGHT_ALLOCATION,
        dtype: DeviceDataType::F32,
        capacity_bytes: (WEIGHT_VALUES.len() as u64) * F32_WIDTH,
        lifetime: DeviceBufferLifetime::PerProgram,
        initialization: DeviceBufferInitialization::HostProvided,
    }
}

fn fake_metal() -> DeviceRuntime {
    DeviceRuntime::Metal(
        MetalHostSession::with_driver(Box::new(FakeMetalDriver::default())).expect("fake metal"),
    )
}

fn prepare_session(runtime: &mut DeviceRuntime) -> KvCacheBindingSession<'_> {
    KvCacheBindingSession::prepare(
        runtime,
        &base_kv(),
        b"// fake compiler-owned module image",
        &[(weight_allocation(), WEIGHT_VALUES.as_slice())],
    )
    .expect("prepare")
}

fn state_at(position: u32, valid_len_after: u32) -> DescriptorInvocationState {
    DescriptorInvocationState {
        position,
        valid_len_after,
        query_rows: 1,
        sequence_epoch: 1,
    }
}

fn binding_at(bindings: &[DeviceLaunchBinding], index: u32) -> DeviceLaunchBinding {
    *bindings
        .iter()
        .find(|binding| binding.binding_index == index)
        .expect("declared binding index")
}

fn observa_pair(src: DeviceLaunchBinding, dest: DeviceHandle) -> [DeviceLaunchBinding; 2] {
    let dest_span = dest.len_bytes().expect("dest buffer");
    [
        DeviceLaunchBinding {
            binding_index: 0,
            ..src
        },
        DeviceLaunchBinding {
            handle: dest,
            binding_index: 1,
            byte_offset: 0,
            view_span: dest_span,
            runtime_source: DescriptorRuntimeSource::Constant,
        },
    ]
}

fn seed_rows(session: &mut KvCacheBindingSession<'_>) {
    let k = session.allocation_handle(K_ALLOCATION).expect("K");
    let v = session.allocation_handle(V_ALLOCATION).expect("V");
    let rows: Vec<f32> = (0..16).map(|i| i as f32).collect();
    session.copy_in_f32(&k, &rows).expect("seed K");
    session.copy_in_f32(&v, &rows).expect("seed V");
}

#[test]
fn one_allocation_per_session_exposes_append_and_prefix_views() {
    let mut runtime = fake_metal();
    let session = prepare_session(&mut runtime);
    let k = session.allocation_handle(K_ALLOCATION).expect("K");
    let v = session.allocation_handle(V_ALLOCATION).expect("V");
    assert_eq!(k.len_bytes(), Some(arena_capacity_bytes()));
    assert_eq!(v.len_bytes(), Some(arena_capacity_bytes()));
    assert_ne!(k, v, "K and V are distinct arenas");

    let bindings = session.materialize_bindings(state_at(0, 1)).expect("row 0");
    let k_prefix = binding_at(&bindings, 0);
    let k_append = binding_at(&bindings, 2);
    assert_eq!(k_prefix.handle, k);
    assert_eq!(k_append.handle, k);
    assert_eq!(k_prefix.handle, k_append.handle);
    assert_eq!(k_append.view_span, row_bytes());
    assert_eq!(k_prefix.view_span, row_bytes());
    assert_ne!(
        k_append.view_span,
        arena_capacity_bytes(),
        "append extent is not allocation capacity"
    );
}

#[test]
fn one_cursor_copy_per_step_is_shared_by_append_and_attention_launches() {
    let mut runtime = fake_metal();
    let mut session = prepare_session(&mut runtime);
    seed_rows(&mut session);
    let dest_append = session
        .alloc_bytes(row_bytes() as usize)
        .expect("append dest");
    let dest_prefix = session
        .alloc_bytes(row_bytes() as usize)
        .expect("prefix dest");

    let bindings = session.begin_step(state_at(1, 2)).expect("one upload");
    assert_eq!(session.cursor_uploads(), 1);
    session
        .launch_kernel_bound(
            "observa",
            &observa_pair(binding_at(&bindings, 2), dest_append),
            [1, 1, 1],
            [4, 1, 1],
        )
        .expect("append launch");
    session
        .launch_kernel_bound(
            "observa",
            &observa_pair(binding_at(&bindings, 0), dest_prefix),
            [1, 1, 1],
            [4, 1, 1],
        )
        .expect("attention launch");
    session.sync().expect("one step barrier");
    assert_eq!(
        session.cursor_uploads(),
        1,
        "append and attention share the one cursor copy"
    );
    assert_eq!(session.cursor_handle().len_bytes(), Some(16));
}

#[test]
fn changing_offsets_between_steps_reuse_stable_handles() {
    let mut runtime = fake_metal();
    let session = prepare_session(&mut runtime);
    let k = session.allocation_handle(K_ALLOCATION).expect("K");
    let v = session.allocation_handle(V_ALLOCATION).expect("V");
    let cursor = session.cursor_handle();
    let weight = session.weight_handle(WEIGHT_ALLOCATION).expect("weight");

    let first = session
        .materialize_bindings(state_at(0, 1))
        .expect("position 0");
    let second = session
        .materialize_bindings(state_at(2, 3))
        .expect("position 2");
    let first_append = binding_at(&first, 2);
    let second_append = binding_at(&second, 2);
    assert_eq!(first_append.handle, k);
    assert_eq!(second_append.handle, k);
    assert_eq!(first_append.byte_offset, 0);
    assert_eq!(second_append.byte_offset, 2 * row_bytes());
    assert_eq!(binding_at(&first, 0).view_span, row_bytes());
    assert_eq!(binding_at(&second, 0).view_span, 3 * row_bytes());
    assert_eq!(binding_at(&first, 3).handle, v);
    assert_eq!(binding_at(&second, 3).handle, v);
    assert_eq!(session.cursor_handle(), cursor);
    assert_eq!(session.weight_handle(WEIGHT_ALLOCATION), Some(weight));
}

#[test]
fn zero_cache_copies_zero_persistent_reallocs_zero_weight_reuploads() {
    let mut runtime = fake_metal();
    let mut session = prepare_session(&mut runtime);
    seed_rows(&mut session);
    let dest = session.alloc_bytes(row_bytes() as usize).expect("dest");
    let k = session.allocation_handle(K_ALLOCATION).expect("K");
    let weight = session.weight_handle(WEIGHT_ALLOCATION).expect("weight");
    let allocs = session.driver_counters().buffer_allocs;
    assert_eq!(session.weight_uploads(), 1);

    for position in 0..3 {
        let bindings = session
            .begin_step(state_at(position, position + 1))
            .expect("step");
        session
            .launch_kernel_bound(
                "observa",
                &observa_pair(binding_at(&bindings, 2), dest),
                [1, 1, 1],
                [4, 1, 1],
            )
            .expect("append");
        session
            .launch_kernel_bound(
                "observa",
                &observa_pair(binding_at(&bindings, 0), dest),
                [1, 1, 1],
                [4, 1, 1],
            )
            .expect("attention");
        session.sync().expect("step barrier");
        assert_eq!(session.allocation_handle(K_ALLOCATION), Some(k));
        assert_eq!(session.weight_handle(WEIGHT_ALLOCATION), Some(weight));
    }

    assert_eq!(session.cursor_uploads(), 3);
    assert_eq!(session.cache_copies(), 0);
    assert_eq!(session.weight_uploads(), 1);
    assert_eq!(session.persistent_reallocs(), 0);
    assert_eq!(session.driver_counters().buffer_allocs, allocs);
    assert_eq!(
        session.readback_f32(&weight).expect("weights stay"),
        WEIGHT_VALUES
    );
}

#[test]
fn launch_binding_offsets_reach_metal() {
    let mut runtime = fake_metal();
    let mut session = prepare_session(&mut runtime);
    seed_rows(&mut session);
    let dest = session.alloc_bytes(row_bytes() as usize).expect("dest");

    let row0 = session.begin_step(state_at(0, 1)).expect("row 0 cursor");
    session
        .launch_kernel_bound(
            "observa",
            &observa_pair(binding_at(&row0, 2), dest),
            [1, 1, 1],
            [4, 1, 1],
        )
        .expect("row 0");
    session.sync().expect("sync row 0");
    assert_eq!(
        session.readback_f32(&dest).expect("row 0"),
        vec![0.0, 1.0, 2.0, 3.0]
    );

    let row2 = session.begin_step(state_at(2, 3)).expect("row 2 cursor");
    assert_eq!(binding_at(&row2, 2).byte_offset, 2 * row_bytes());
    session
        .launch_kernel_bound(
            "observa",
            &observa_pair(binding_at(&row2, 2), dest),
            [1, 1, 1],
            [4, 1, 1],
        )
        .expect("row 2");
    session.sync().expect("sync row 2");
    assert_eq!(
        session
            .readback_f32(&dest)
            .expect("row 2 must be the bound offset, not a cache copy"),
        vec![8.0, 9.0, 10.0, 11.0]
    );
}

#[test]
fn overflow_offset_and_prefix_span_fail_closed() {
    let mut runtime = fake_metal();
    let session = prepare_session(&mut runtime);
    let over_position = session
        .materialize_bindings(state_at(POSITIONS as u32, 1))
        .expect_err("position at capacity overflows the arena");
    assert_eq!(over_position.code, E_DEVICE_SHAPE_MISMATCH);

    let over_prefix = session
        .materialize_bindings(state_at(0, POSITIONS as u32 + 1))
        .expect_err("prefix longer than capacity overflows the view");
    assert_eq!(over_prefix.code, E_DEVICE_SHAPE_MISMATCH);
}
