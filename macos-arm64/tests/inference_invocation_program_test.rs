//! KV-D D3: dual invocation programs over shared residency.
//!
//! Parent registration is a private `mod invocation_program` in
//! `composite_host.rs`. This unit cannot edit that file, so the test crate
//! compiles the module directly. `device_descriptor` is re-exported so the
//! path-compiled modules can keep `crate::device_descriptor`.

mod device_descriptor {
    pub use faber_host_macos_arm64::device_descriptor::*;
}

#[path = "../src/composite_host/inference_state.rs"]
mod inference_state;

#[path = "../src/composite_host/residency.rs"]
mod residency;

#[path = "../src/composite_host/invocation_program.rs"]
mod invocation_program;

use device_descriptor::{
    DescriptorAllocation, DescriptorInvocationState, DescriptorLaunchBinding,
    DescriptorRuntimeSource, DescriptorView, DeviceBufferInitialization, DeviceBufferLifetime,
    DeviceDataType, KvCacheDescriptor,
};
use inference_state::{InvocationMode, SequencePhase, E_INVALID_ARGS, E_KV_PHASE};
use invocation_program::{AdmittedDescriptor, InvocationPrograms, SCALAR_DECODE_QUERY_ROWS};
use residency::{ModelIdentity, ResidentAllocation};

const LAYERS: u64 = 2;
const KV_HEADS: u64 = 2;
const HEAD_DIM: u64 = 4;
const F32_WIDTH: u64 = 4;

const K_ALLOCATION: u32 = 1;
const V_ALLOCATION: u32 = 2;
const INVOCATION_STATE: u32 = 3;
const WEIGHT_ALLOCATION: u32 = 10;

const PREFILL_ARTIFACT: &[u8] = b"prefill-module";
const DECODE_ARTIFACT: &[u8] = b"decode-module";

fn arena_capacity_bytes(positions: u64) -> u64 {
    LAYERS * KV_HEADS * positions * HEAD_DIM * F32_WIDTH
}

fn append_span_bytes() -> u64 {
    LAYERS * KV_HEADS * HEAD_DIM * F32_WIDTH
}

fn arena_strides(positions: u64) -> Vec<u64> {
    let dim = 1;
    let position = HEAD_DIM;
    let kv_head = positions * HEAD_DIM;
    let layer = KV_HEADS * positions * HEAD_DIM;
    vec![layer, kv_head, position, dim]
}

fn persistent_arena(buffer_id: u32, positions: u64) -> DescriptorAllocation {
    DescriptorAllocation {
        buffer_id,
        dtype: DeviceDataType::F32,
        capacity_bytes: arena_capacity_bytes(positions),
        lifetime: DeviceBufferLifetime::PerProgram,
        initialization: DeviceBufferInitialization::ZeroFill,
    }
}

fn prefix_view(allocation_id: u32, positions: u64) -> DescriptorView {
    DescriptorView {
        allocation_id,
        logical_dims: vec![LAYERS, KV_HEADS, positions, HEAD_DIM],
        strides: arena_strides(positions),
        static_base: 0,
        maximum_span: arena_capacity_bytes(positions),
    }
}

fn append_view(allocation_id: u32, positions: u64) -> DescriptorView {
    DescriptorView {
        allocation_id,
        logical_dims: vec![LAYERS, KV_HEADS, 1, HEAD_DIM],
        strides: arena_strides(positions),
        static_base: 0,
        maximum_span: append_span_bytes(),
    }
}

fn weight_allocation(buffer_id: u32) -> DescriptorAllocation {
    DescriptorAllocation {
        buffer_id,
        dtype: DeviceDataType::F32,
        capacity_bytes: 1024,
        lifetime: DeviceBufferLifetime::PerProgram,
        initialization: DeviceBufferInitialization::HostProvided,
    }
}

fn invocation_state_allocation() -> DescriptorAllocation {
    DescriptorAllocation {
        buffer_id: INVOCATION_STATE,
        dtype: DeviceDataType::U8,
        capacity_bytes: 16,
        lifetime: DeviceBufferLifetime::PerProgram,
        initialization: DeviceBufferInitialization::ZeroFill,
    }
}

/// Bindings in a deliberately non-monotonic index order so B5 preservation
/// is visible: K-append@2, K-prefix@0, V-append@3, V-prefix@1.
fn declared_launch_bindings(positions: u64) -> Vec<DescriptorLaunchBinding> {
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
            view_span: arena_capacity_bytes(positions),
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
            view_span: arena_capacity_bytes(positions),
            runtime_source: DescriptorRuntimeSource::ValidLenAfter,
        },
    ]
}

fn kv_descriptor(capacity: u32) -> KvCacheDescriptor {
    let positions = u64::from(capacity);
    KvCacheDescriptor {
        allocations: vec![
            persistent_arena(K_ALLOCATION, positions),
            persistent_arena(V_ALLOCATION, positions),
        ],
        views: vec![
            prefix_view(K_ALLOCATION, positions),
            append_view(K_ALLOCATION, positions),
            prefix_view(V_ALLOCATION, positions),
            append_view(V_ALLOCATION, positions),
        ],
        invocation_state: DescriptorInvocationState::default(),
        launch_bindings: declared_launch_bindings(positions),
    }
}

fn admitted(capacity: u32, prefill_query_rows: u32) -> AdmittedDescriptor {
    AdmittedDescriptor {
        identity: ModelIdentity::new("dense-rung", 1),
        prefill_artifact: PREFILL_ARTIFACT.to_vec(),
        decode_artifact: DECODE_ARTIFACT.to_vec(),
        prefill_query_rows,
        weights: vec![weight_allocation(WEIGHT_ALLOCATION)],
        kv: kv_descriptor(capacity),
        invocation_state: invocation_state_allocation(),
        capacity,
    }
}

fn admit_programs(capacity: u32, prefill_query_rows: u32) -> InvocationPrograms {
    InvocationPrograms::admit(admitted(capacity, prefill_query_rows)).expect("admit")
}

fn assert_same_object(left: &ResidentAllocation, right: &ResidentAllocation, what: &str) {
    assert!(
        left.is_same_object(right),
        "{what} must be the same resident object"
    );
    assert_eq!(left.identity(), right.identity(), "{what} identity");
    assert_eq!(left.buffer_id(), right.buffer_id(), "{what} B3 handle");
}

fn snapshot_lifecycle(programs: &InvocationPrograms) -> (u32, u32, usize, u32, u32) {
    (
        programs.module_loads(),
        programs.compiles(),
        programs.live_allocation_count(),
        programs.weight_uploads(),
        programs.artifact_prepares(),
    )
}

fn assert_lifecycle_unchanged(
    programs: &InvocationPrograms,
    before: (u32, u32, usize, u32, u32),
    what: &str,
) {
    assert_eq!(programs.module_loads(), before.0, "{what}: module loads");
    assert_eq!(programs.compiles(), before.1, "{what}: compiles");
    assert_eq!(
        programs.live_allocation_count(),
        before.2,
        "{what}: allocations"
    );
    assert_eq!(
        programs.weight_uploads(),
        before.3,
        "{what}: weight uploads"
    );
    assert_eq!(
        programs.artifact_prepares(),
        before.4,
        "{what}: artifact prepares"
    );
}

#[test]
fn admit_prepares_both_graphs_before_first_invocation() {
    let programs = admit_programs(8, 4);
    assert_eq!(programs.residency().phase(), SequencePhase::Fresh);
    assert_eq!(programs.residency().valid_len(), 0);
    assert_eq!(programs.artifact_prepares(), 1);
    assert_eq!(programs.module_loads(), 2);
    assert_eq!(programs.compiles(), 2);
    assert_eq!(programs.weight_uploads(), 1);

    let prefill = programs.select(InvocationMode::Prefill);
    let decode = programs.select(InvocationMode::ScalarDecode);
    assert!(!prefill.artifact().is_empty());
    assert!(!decode.artifact().is_empty());
    assert_eq!(prefill.artifact(), PREFILL_ARTIFACT);
    assert_eq!(decode.artifact(), DECODE_ARTIFACT);
    assert_eq!(
        prefill.artifact(),
        programs.residency().model().artifacts().prefill()
    );
    assert_eq!(
        decode.artifact(),
        programs.residency().model().artifacts().scalar_decode()
    );
}

#[test]
fn prefill_is_m_t_and_scalar_decode_is_m_1() {
    let programs = admit_programs(8, 4);
    assert_eq!(programs.prefill().mode(), InvocationMode::Prefill);
    assert_eq!(programs.prefill().query_rows(), 4);
    assert_eq!(
        programs.scalar_decode().mode(),
        InvocationMode::ScalarDecode
    );
    assert_eq!(
        programs.scalar_decode().query_rows(),
        SCALAR_DECODE_QUERY_ROWS
    );
    assert_eq!(SCALAR_DECODE_QUERY_ROWS, 1);

    let prefill = programs.select(InvocationMode::Prefill);
    let decode = programs.select(InvocationMode::ScalarDecode);
    assert_eq!(prefill.query_rows(), 4);
    assert_eq!(decode.query_rows(), 1);
    assert_ne!(
        prefill.query_rows(),
        decode.query_rows(),
        "Prefill(M=T) and ScalarDecode(M=1) are distinct graphs"
    );
}

#[test]
fn both_programs_share_one_residency_and_identical_handles() {
    let programs = admit_programs(8, 4);
    let prefill = programs.select(InvocationMode::Prefill);
    let decode = programs.select(InvocationMode::ScalarDecode);

    assert!(std::ptr::eq(
        prefill.handles().model_identity,
        decode.handles().model_identity
    ));
    assert_eq!(prefill.handles().model_identity.name(), "dense-rung");
    assert_same_object(
        prefill.handles().k_arena,
        decode.handles().k_arena,
        "K arena",
    );
    assert_same_object(
        prefill.handles().v_arena,
        decode.handles().v_arena,
        "V arena",
    );
    assert_same_object(
        prefill.handles().invocation_state,
        decode.handles().invocation_state,
        "invocation-state buffer",
    );
    assert!(
        std::ptr::eq(prefill.handles().weights, decode.handles().weights),
        "weight slice must be the same allocation vector"
    );
    assert_eq!(prefill.handles().weights.len(), 1);
    assert_same_object(
        &prefill.handles().weights[0],
        &decode.handles().weights[0],
        "weight",
    );

    assert_eq!(prefill.launch_bindings(), decode.launch_bindings());
    assert_eq!(
        prefill.launch_bindings(),
        programs.kv().launch_bindings.as_slice()
    );
}

#[test]
fn switching_mode_does_not_load_compile_allocate_or_upload() {
    let programs = admit_programs(8, 4);
    let before = snapshot_lifecycle(&programs);
    assert_eq!(before, (2, 2, 4, 1, 1));

    let _ = programs.select(InvocationMode::Prefill);
    let _ = programs.select(InvocationMode::ScalarDecode);
    let _ = programs.select(InvocationMode::Prefill);
    let _ = programs.resolve(InvocationMode::ScalarDecode);
    assert_lifecycle_unchanged(&programs, before, "select/resolve");
}

#[test]
fn prefill_to_decode_commit_releases_nothing_and_does_not_reload() {
    let mut programs = admit_programs(8, 4);
    let before = snapshot_lifecycle(&programs);
    let k_ptr = programs.residency().sequence().k_arena() as *const ResidentAllocation;
    let v_ptr = programs.residency().sequence().v_arena() as *const ResidentAllocation;
    let weight_ptr = &programs.residency().model().weights()[0] as *const ResidentAllocation;
    let k_id = programs.residency().sequence().k_arena().identity();
    let v_id = programs.residency().sequence().v_arena().identity();
    let weight_id = programs.residency().model().weights()[0].identity();

    let prefill = programs
        .begin_selected(InvocationMode::Prefill)
        .expect("prefill admits");
    assert_eq!(prefill.mode(), InvocationMode::Prefill);
    assert_eq!(prefill.coordinates().query_rows, 4);
    assert_eq!(prefill.coordinates().prefix_before, 0);
    let facts = programs.commit(&prefill).expect("prefill commits");
    assert_eq!(facts.valid_len_after, 4);
    assert_eq!(programs.residency().phase(), SequencePhase::Prefill);
    assert_lifecycle_unchanged(&programs, before, "after prefill");

    let decode = programs
        .begin_selected(InvocationMode::ScalarDecode)
        .expect("decode admits");
    assert_eq!(decode.mode(), InvocationMode::ScalarDecode);
    assert_eq!(decode.coordinates().query_rows, 1);
    assert_eq!(decode.coordinates().prefix_before, 4);
    programs.commit(&decode).expect("decode commits");
    assert_eq!(programs.residency().phase(), SequencePhase::Decode);
    assert_eq!(programs.residency().valid_len(), 5);
    assert_lifecycle_unchanged(&programs, before, "after decode");

    assert_eq!(programs.residency().sequence().k_arena().identity(), k_id);
    assert_eq!(programs.residency().sequence().v_arena().identity(), v_id);
    assert_eq!(
        programs.residency().model().weights()[0].identity(),
        weight_id
    );
    assert_eq!(
        programs.residency().sequence().k_arena() as *const ResidentAllocation,
        k_ptr
    );
    assert_eq!(
        programs.residency().sequence().v_arena() as *const ResidentAllocation,
        v_ptr
    );
    assert_eq!(
        &programs.residency().model().weights()[0] as *const ResidentAllocation,
        weight_ptr
    );
}

#[test]
fn program_selection_is_never_inferred_from_sequence_length() {
    let mut programs = admit_programs(8, 4);
    assert_eq!(programs.residency().valid_len(), 0);

    let decode_on_fresh = programs.select(InvocationMode::ScalarDecode);
    assert_eq!(decode_on_fresh.mode(), InvocationMode::ScalarDecode);
    assert_eq!(decode_on_fresh.query_rows(), 1);
    assert_ne!(
        decode_on_fresh.query_rows(),
        programs.residency().valid_len(),
        "decode M is not inferred from valid_len=0"
    );

    let prefill_on_fresh = programs.select(InvocationMode::Prefill);
    assert_eq!(prefill_on_fresh.query_rows(), 4);
    assert_ne!(
        prefill_on_fresh.query_rows(),
        programs.residency().valid_len(),
        "prefill M=T is not inferred from valid_len"
    );

    let plan = programs
        .begin_selected(InvocationMode::Prefill)
        .expect("explicit prefill");
    programs.commit(&plan).expect("commit");
    assert_eq!(programs.residency().valid_len(), 4);

    let still_prefill = programs.select(InvocationMode::Prefill);
    assert_eq!(still_prefill.mode(), InvocationMode::Prefill);
    assert_eq!(still_prefill.query_rows(), 4);
    assert_eq!(still_prefill.query_rows(), programs.residency().valid_len());

    let decode = programs.select(InvocationMode::ScalarDecode);
    assert_eq!(decode.mode(), InvocationMode::ScalarDecode);
    assert_eq!(decode.query_rows(), 1);
    assert_ne!(
        decode.query_rows(),
        programs.residency().valid_len(),
        "valid_len=T must not select decode"
    );

    let err = programs
        .begin_selected(InvocationMode::Prefill)
        .expect_err("committed prefill is not silently decoded");
    assert_eq!(err.code, E_KV_PHASE);
    assert_eq!(programs.residency().valid_len(), 4);
    assert_eq!(programs.residency().phase(), SequencePhase::Prefill);
}

#[test]
fn synthetic_admitted_descriptor_is_the_b5_host_field_table() {
    let descriptor = admitted(8, 4);
    assert_eq!(
        descriptor.kv.invocation_state,
        DescriptorInvocationState::default(),
        "B5: live cursor values are not carried by the plan"
    );
    assert_eq!(descriptor.kv.allocations.len(), 2);
    assert_eq!(descriptor.kv.views.len(), 4);
    assert_eq!(
        descriptor
            .kv
            .launch_bindings
            .iter()
            .map(|binding| binding.binding_index)
            .collect::<Vec<_>>(),
        vec![2, 0, 3, 1],
        "declared binding indices must not be sorted or dropped"
    );
    assert_eq!(
        descriptor.kv.launch_bindings[0].runtime_source,
        DescriptorRuntimeSource::Position
    );
    assert_eq!(
        descriptor.kv.launch_bindings[1].runtime_source,
        DescriptorRuntimeSource::ValidLenAfter
    );
    descriptor.kv.validate().expect("B5 plan admits");

    let programs = InvocationPrograms::admit(descriptor).expect("admit");
    assert_eq!(
        programs
            .kv()
            .launch_bindings
            .iter()
            .map(|binding| binding.binding_index)
            .collect::<Vec<_>>(),
        vec![2, 0, 3, 1]
    );
    assert_eq!(
        programs.kv().invocation_state,
        DescriptorInvocationState::default()
    );
    let cursor = programs.residency().sequence().invocation_state();
    assert_eq!(cursor.position, 0);
    assert_eq!(cursor.valid_len_after, 0);
    assert_eq!(cursor.query_rows, 0);
    assert_eq!(cursor.sequence_epoch, 1);
}

#[test]
fn equal_byte_counts_are_not_shared_identity() {
    let first = admit_programs(8, 4);
    let second = admit_programs(8, 4);
    let first_k = first.select(InvocationMode::Prefill).handles().k_arena;
    let second_k = second
        .select(InvocationMode::ScalarDecode)
        .handles()
        .k_arena;
    assert_eq!(first_k.capacity_bytes(), second_k.capacity_bytes());
    assert_eq!(first_k.buffer_id(), second_k.buffer_id());
    assert_ne!(
        first_k.identity(),
        second_k.identity(),
        "KV-L7: equal byte counts and buffer ids are not shared identity"
    );
    assert!(!first_k.is_same_object(second_k));
}

#[test]
fn begin_selected_uses_declared_m_not_valid_len() {
    let mut programs = admit_programs(16, 7);
    let plan = programs
        .begin_selected(InvocationMode::Prefill)
        .expect("prefill");
    assert_eq!(plan.coordinates().query_rows, 7);
    assert_eq!(
        plan.coordinates().query_rows,
        programs.prefill().query_rows()
    );
    assert_ne!(
        plan.coordinates().query_rows,
        programs.residency().valid_len()
    );
    programs.commit(&plan).expect("commit");

    let decode = programs
        .begin_selected(InvocationMode::ScalarDecode)
        .expect("decode");
    assert_eq!(decode.coordinates().query_rows, 1);
    assert_ne!(
        decode.coordinates().query_rows,
        programs.residency().valid_len()
    );
}

#[test]
fn admit_rejects_empty_decode_artifact() {
    let mut descriptor = admitted(8, 4);
    descriptor.decode_artifact.clear();
    let err = InvocationPrograms::admit(descriptor).expect_err("empty decode");
    assert_eq!(err.code, E_INVALID_ARGS);
}

#[test]
fn admit_rejects_prefill_width_exceeding_capacity() {
    let err = InvocationPrograms::admit(admitted(4, 8)).expect_err("M>T capacity");
    assert_eq!(err.code, E_INVALID_ARGS);
}

#[test]
fn admit_rejects_live_cursor_on_the_static_plan() {
    let mut descriptor = admitted(8, 4);
    descriptor.kv.invocation_state = DescriptorInvocationState {
        position: 3,
        valid_len_after: 4,
        query_rows: 1,
        sequence_epoch: 7,
    };
    let err = InvocationPrograms::admit(descriptor).expect_err("live cursor");
    assert_eq!(err.code, E_INVALID_ARGS);
}

#[test]
fn admit_rejects_kv_plan_that_is_not_two_cache_arenas() {
    let mut descriptor = admitted(8, 4);
    descriptor.kv.allocations.pop();
    descriptor
        .kv
        .views
        .retain(|view| view.allocation_id == K_ALLOCATION);
    descriptor
        .kv
        .launch_bindings
        .retain(|binding| binding.handle == K_ALLOCATION);
    let err = InvocationPrograms::admit(descriptor).expect_err("one arena");
    assert_eq!(err.code, E_INVALID_ARGS);
}
