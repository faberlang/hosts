//! PPE-P1 paired prefill/scalar-decode executor proof.
//!
//! The fake backend proves the host-side dispatch seam only. Both programs are
//! prepared before invocation, bind one semantic weight allocation, and route
//! their selected v2 mode through a real resident step.

use faber_host_macos_arm64::composite_host::invocation_binding::{
    RopeConfig, KV_PREFIX_IDS, PROMPT_TOKENS, Q_PREFIX_IDS, ROPE_COS, ROPE_SIN,
};
use faber_host_macos_arm64::composite_host::{
    CompositeHost, DeviceByteBuffer, SequencePhase, E_KV_PHASE, E_KV_POISONED,
};
use faber_host_macos_arm64::device_descriptor::{
    DescriptorBuffer, DescriptorBufferVersion, DescriptorKernel, DescriptorLaunch,
    DescriptorResult, DeviceBufferInitialization, DeviceBufferLifetime, DeviceBufferRole,
    DeviceDataType, DeviceDescriptor, DeviceProgramLifetime,
};
use faber_host_macos_arm64::device_execute::{
    DeviceExecuteInvocation, DeviceExecuteInvocationMode,
};
use faber_host_macos_arm64::device_host::DeviceRuntime;
use faber_host_macos_arm64::{FakeMetalDriver, MetalHostSession};
use host_coordinator::DeviceBackend;

const MODULE_IMAGE: &[u8] = b"// paired fake module image";

fn slot(
    id: u32,
    name: &str,
    semantic_value: u32,
    role: DeviceBufferRole,
    lifetime: DeviceBufferLifetime,
    initialization: DeviceBufferInitialization,
    element_count: u64,
    binding: u32,
) -> DescriptorBuffer {
    DescriptorBuffer {
        buffer_id: id,
        buffer_name: name.to_owned(),
        semantic_value,
        role,
        lifetime,
        initialization,
        binding,
        element_ty: DeviceDataType::F32,
        element_count,
        version: 1,
    }
}

fn buffer_versions(kernels: &[DescriptorKernel]) -> Vec<DescriptorBufferVersion> {
    let mut versions = Vec::new();
    for kernel in kernels {
        for buffer in &kernel.buffers {
            if versions.iter().any(|version: &DescriptorBufferVersion| {
                version.buffer_id == buffer.buffer_id && version.version == buffer.version
            }) {
                continue;
            }
            versions.push(DescriptorBufferVersion {
                buffer_id: buffer.buffer_id,
                version: buffer.version,
                element_ty: buffer.element_ty,
                element_count: buffer.element_count,
            });
        }
    }
    versions
}

/// The launched kernel is a normal fake `addita`/`observa` body. The second
/// declaration carries the P2 dynamic input names without adding those
/// metadata-only bindings to the fake launch arity.
fn prefill_descriptor() -> DeviceDescriptor {
    let kernels = vec![
        DescriptorKernel {
            entry: "addita".to_owned(),
            buffers: vec![
                slot(
                    1,
                    PROMPT_TOKENS,
                    1,
                    DeviceBufferRole::Input,
                    DeviceBufferLifetime::PerStep,
                    DeviceBufferInitialization::ZeroFill,
                    2,
                    0,
                ),
                slot(
                    10,
                    "model.weight",
                    10,
                    DeviceBufferRole::Input,
                    DeviceBufferLifetime::PerProgram,
                    DeviceBufferInitialization::HostProvided,
                    2,
                    1,
                ),
                slot(
                    20,
                    "prefill.output",
                    20,
                    DeviceBufferRole::Output,
                    DeviceBufferLifetime::ObservationPoint,
                    DeviceBufferInitialization::KernelInitialized,
                    2,
                    2,
                ),
            ],
            grid: [1, 1, 1],
            block: [2, 1, 1],
        },
        DescriptorKernel {
            entry: "prefill_inputs".to_owned(),
            buffers: vec![
                slot(
                    2,
                    ROPE_COS,
                    2,
                    DeviceBufferRole::Input,
                    DeviceBufferLifetime::PerStep,
                    DeviceBufferInitialization::ZeroFill,
                    8,
                    0,
                ),
                slot(
                    3,
                    ROPE_SIN,
                    3,
                    DeviceBufferRole::Input,
                    DeviceBufferLifetime::PerStep,
                    DeviceBufferInitialization::ZeroFill,
                    8,
                    1,
                ),
            ],
            grid: [1, 1, 1],
            block: [1, 1, 1],
        },
    ];
    DeviceDescriptor {
        backend: DeviceBackend::Metal,
        module_image: MODULE_IMAGE.to_vec(),
        buffer_versions: buffer_versions(&kernels),
        kernels,
        launches: vec![DescriptorLaunch {
            id: 1,
            kernel_index: 0,
        }],
        program_lifetime: DeviceProgramLifetime::RepeatingStep,
        data_flow: Vec::new(),
        roots: vec![1],
        results: vec![DescriptorResult {
            buffer_id: 20,
            version: 1,
            produced_by: 1,
            at_launch: 1,
        }],
        end_of_run_results: Vec::new(),
    }
}

fn decode_descriptor() -> DeviceDescriptor {
    let kernels = vec![
        DescriptorKernel {
            entry: "observa".to_owned(),
            buffers: vec![
                slot(
                    101,
                    PROMPT_TOKENS,
                    101,
                    DeviceBufferRole::Input,
                    DeviceBufferLifetime::PerStep,
                    DeviceBufferInitialization::ZeroFill,
                    1,
                    0,
                ),
                slot(
                    120,
                    "decode.output",
                    120,
                    DeviceBufferRole::Output,
                    DeviceBufferLifetime::ObservationPoint,
                    DeviceBufferInitialization::KernelInitialized,
                    1,
                    1,
                ),
            ],
            grid: [1, 1, 1],
            block: [1, 1, 1],
        },
        DescriptorKernel {
            entry: "decode_inputs".to_owned(),
            buffers: vec![
                // Different program buffer id, same semantic value identity
                // as prefill's model.weight: this is the sharing assertion.
                slot(
                    110,
                    "model.weight",
                    10,
                    DeviceBufferRole::Input,
                    DeviceBufferLifetime::PerProgram,
                    DeviceBufferInitialization::HostProvided,
                    2,
                    0,
                ),
                slot(
                    102,
                    ROPE_COS,
                    102,
                    DeviceBufferRole::Input,
                    DeviceBufferLifetime::PerStep,
                    DeviceBufferInitialization::ZeroFill,
                    4,
                    1,
                ),
                slot(
                    103,
                    ROPE_SIN,
                    103,
                    DeviceBufferRole::Input,
                    DeviceBufferLifetime::PerStep,
                    DeviceBufferInitialization::ZeroFill,
                    4,
                    2,
                ),
                slot(
                    104,
                    Q_PREFIX_IDS,
                    104,
                    DeviceBufferRole::Input,
                    DeviceBufferLifetime::PerStep,
                    DeviceBufferInitialization::ZeroFill,
                    3,
                    3,
                ),
                slot(
                    105,
                    KV_PREFIX_IDS,
                    105,
                    DeviceBufferRole::Input,
                    DeviceBufferLifetime::PerStep,
                    DeviceBufferInitialization::ZeroFill,
                    3,
                    4,
                ),
            ],
            grid: [1, 1, 1],
            block: [1, 1, 1],
        },
    ];
    DeviceDescriptor {
        backend: DeviceBackend::Metal,
        module_image: MODULE_IMAGE.to_vec(),
        buffer_versions: buffer_versions(&kernels),
        kernels,
        launches: vec![DescriptorLaunch {
            id: 1,
            kernel_index: 0,
        }],
        program_lifetime: DeviceProgramLifetime::RepeatingStep,
        data_flow: Vec::new(),
        roots: vec![1],
        results: vec![DescriptorResult {
            buffer_id: 120,
            version: 1,
            produced_by: 1,
            at_launch: 1,
        }],
        end_of_run_results: Vec::new(),
    }
}

fn invocation(
    mode: DeviceExecuteInvocationMode,
    token: Option<u32>,
    position: u32,
    prefix_before: u32,
    valid_len_after: u32,
) -> DeviceExecuteInvocation {
    invocation_on(mode, token, position, prefix_before, valid_len_after, 1)
}

fn invocation_on(
    mode: DeviceExecuteInvocationMode,
    token: Option<u32>,
    position: u32,
    prefix_before: u32,
    valid_len_after: u32,
    sequence_epoch: u32,
) -> DeviceExecuteInvocation {
    DeviceExecuteInvocation {
        mode,
        token,
        position,
        sequence_epoch,
        prefix_before,
        valid_len_after,
        query_start: position,
    }
}

fn cache_slots(k_id: u32, v_id: u32) -> DescriptorKernel {
    DescriptorKernel {
        entry: "cache".to_owned(),
        buffers: vec![
            slot(
                k_id,
                "cache.k",
                30,
                DeviceBufferRole::Input,
                DeviceBufferLifetime::PerProgram,
                DeviceBufferInitialization::ZeroFill,
                8,
                0,
            ),
            slot(
                v_id,
                "cache.v",
                31,
                DeviceBufferRole::Input,
                DeviceBufferLifetime::PerProgram,
                DeviceBufferInitialization::ZeroFill,
                8,
                1,
            ),
        ],
        grid: [1, 1, 1],
        block: [1, 1, 1],
    }
}

fn prefill_descriptor_with_cache() -> DeviceDescriptor {
    let mut prefill = prefill_descriptor();
    prefill.kernels.push(cache_slots(30, 31));
    prefill.buffer_versions = buffer_versions(&prefill.kernels);
    prefill
}

fn decode_descriptor_with_cache() -> DeviceDescriptor {
    let mut decode = decode_descriptor();
    decode.kernels.push(cache_slots(130, 131));
    decode.buffer_versions = buffer_versions(&decode.kernels);
    decode
}

fn paired_host() -> CompositeHost {
    let runtime = DeviceRuntime::Metal(
        MetalHostSession::with_driver(Box::new(
            FakeMetalDriver::default()
                .with_known_entry("addita")
                .with_known_entry("observa"),
        ))
        .expect("fake Metal admission"),
    );
    CompositeHost::with_device(runtime, "fake-metal-device").expect("composite host")
}

#[test]
fn paired_session_prepares_once_and_dispatches_each_mode() {
    let mut host = paired_host();
    let prefill = prefill_descriptor();
    let decode = decode_descriptor();
    let weights = std::collections::BTreeMap::from([(10, vec![10.0, 20.0])]);
    let mut pair = host
        .prepare_paired_session(
            &prefill,
            &decode,
            vec![11, 22],
            RopeConfig {
                head_dim: 8,
                theta: 10_000.0,
            },
            &weights,
            &std::collections::BTreeMap::<u32, DeviceByteBuffer>::new(),
            "model",
            "session",
        )
        .expect("prepare both programs");

    // The module and the one semantic weight owner are live before either
    // dispatch. The decode buffer id differs, so a second allocation would
    // prove that semantic sharing was lost.
    assert_eq!(pair.live_handles(), 2);
    assert_eq!(pair.driver_counters().module_loads, 1);
    assert_eq!(pair.driver_counters().buffer_allocs, 1);

    let prefill_receipt = pair
        .execute_invocation(&invocation(
            DeviceExecuteInvocationMode::Prefill,
            None,
            0,
            0,
            2,
        ))
        .expect("prefill dispatch");
    assert_eq!(prefill_receipt.launch_entries, vec!["addita"]);
    assert_eq!(prefill_receipt.launches, 1);
    assert!(prefill_receipt.outputs.contains_key(&20));
    assert_eq!(pair.reuses(), 1);

    let decode_receipt = pair
        .execute_invocation(&invocation(
            DeviceExecuteInvocationMode::ScalarDecode,
            Some(42),
            2,
            2,
            3,
        ))
        .expect("scalar decode dispatch");
    assert_eq!(decode_receipt.launch_entries, vec!["observa"]);
    assert_eq!(decode_receipt.launches, 1);
    assert!(decode_receipt.outputs.contains_key(&120));
    assert_eq!(pair.reuses(), 2);

    // The second static program reuses the already loaded module and the
    // prefill weight allocation; only the selected program's step pool grows.
    let counters = pair.driver_counters();
    assert_eq!(counters.module_loads, 1);
    assert_eq!(counters.buffer_allocs, 1 + 4 + 6);
    pair.teardown().expect("paired teardown");
    assert_eq!(host.device().expect("runtime").live_handle_count(), 0);
    let counters = host.device().expect("runtime").driver_counters();
    assert_eq!(counters.module_loads, counters.module_releases);
    assert_eq!(counters.buffer_allocs, counters.buffer_releases);
}

#[test]
fn paired_projection_failure_is_pre_dispatch_and_does_not_poison_the_pair() {
    let mut host = paired_host();
    let prefill = prefill_descriptor();
    let decode = decode_descriptor();
    let mut pair = host
        .prepare_paired_session(
            &prefill,
            &decode,
            vec![11, 22],
            RopeConfig {
                head_dim: 8,
                theta: 10_000.0,
            },
            &std::collections::BTreeMap::from([(10, vec![10.0, 20.0])]),
            &std::collections::BTreeMap::new(),
            "model",
            "session",
        )
        .expect("prepare pair");

    pair.execute_invocation(&invocation(
        DeviceExecuteInvocationMode::Prefill,
        None,
        0,
        0,
        2,
    ))
    .expect("prefill so decode is the legal next mode");
    assert_eq!(pair.valid_len(), 2);
    assert_eq!(pair.sequence_epoch(), 1);
    assert_eq!(pair.reuses(), 1);
    let handles = pair.live_handles();

    let error = pair
        .execute_invocation(&invocation(
            DeviceExecuteInvocationMode::ScalarDecode,
            Some(42),
            3,
            2,
            3,
        ))
        .expect_err("a position gap is rejected before dispatch");
    assert_eq!(error.code, "E_INVALID_ARGS");
    assert_eq!(pair.reuses(), 1);
    assert_eq!(pair.valid_len(), 2);
    assert_eq!(pair.sequence_epoch(), 1);
    assert_eq!(pair.phase(), SequencePhase::Prefill);
    assert_eq!(pair.live_handles(), handles);

    let receipt = pair
        .execute_invocation(&invocation(
            DeviceExecuteInvocationMode::ScalarDecode,
            Some(42),
            2,
            2,
            3,
        ))
        .expect("the pair remains usable after pre-dispatch rejection");
    assert_eq!(receipt.launch_entries, vec!["observa"]);
    assert_eq!(pair.valid_len(), 3);
    pair.teardown().expect("teardown");
}

#[test]
fn paired_device_failure_releases_the_shared_owner() {
    let runtime = DeviceRuntime::Metal(
        MetalHostSession::with_driver(Box::new(
            FakeMetalDriver::default()
                .with_known_entry("addita")
                .with_known_entry("observa")
                .with_failure_at(
                    faber_host_macos_arm64::device_registry::FakeFailureStage::Launch,
                    1,
                ),
        ))
        .expect("fake Metal admission"),
    );
    let mut host = CompositeHost::with_device(runtime, "fake-metal-device").expect("host");
    let mut pair = host
        .prepare_paired_session(
            &prefill_descriptor(),
            &decode_descriptor(),
            vec![11, 22],
            RopeConfig {
                head_dim: 8,
                theta: 10_000.0,
            },
            &std::collections::BTreeMap::from([(10, vec![10.0, 20.0])]),
            &std::collections::BTreeMap::new(),
            "model",
            "session",
        )
        .expect("prepare pair");

    let error = pair
        .execute_invocation(&invocation(
            DeviceExecuteInvocationMode::Prefill,
            None,
            0,
            0,
            2,
        ))
        .expect_err("launch failure");
    assert!(error.message.contains("injected failure"));
    assert_eq!(pair.live_handles(), 0);
    assert!(pair
        .execute_invocation(&invocation(
            DeviceExecuteInvocationMode::ScalarDecode,
            Some(42),
            0,
            0,
            1,
        ))
        .is_err());
    let reset_err = pair
        .reset()
        .expect_err("poisoned sequence rejects reset; rollback is not proven");
    assert_eq!(reset_err.code, E_KV_POISONED);
    pair.teardown().expect("poisoned pair teardown");
    assert_eq!(host.device().expect("runtime").live_handle_count(), 0);
}

#[test]
fn paired_reset_replays_prefill_without_cache_upload() {
    let mut host = paired_host();
    let mut pair = host
        .prepare_paired_session(
            &prefill_descriptor_with_cache(),
            &decode_descriptor_with_cache(),
            vec![11, 22],
            RopeConfig {
                head_dim: 8,
                theta: 10_000.0,
            },
            &std::collections::BTreeMap::from([(10, vec![10.0, 20.0])]),
            &std::collections::BTreeMap::new(),
            "model",
            "session",
        )
        .expect("prepare pair with shared K/V arenas");

    // Module + weight + K + V. The arenas exist before any step so reset
    // can prove it does not reallocate or zero-fill them.
    assert_eq!(pair.live_handles(), 4);
    assert_eq!(pair.driver_counters().buffer_allocs, 3);
    let identity = pair
        .shared_owner_identity()
        .expect("shared owner at prepare");
    assert_eq!(identity.1, std::collections::BTreeSet::from([10, 30, 31]));

    pair.execute_invocation(&invocation(
        DeviceExecuteInvocationMode::Prefill,
        None,
        0,
        0,
        2,
    ))
    .expect("prefill");
    pair.execute_invocation(&invocation(
        DeviceExecuteInvocationMode::ScalarDecode,
        Some(42),
        2,
        2,
        3,
    ))
    .expect("decode");
    assert_eq!(pair.valid_len(), 3);
    assert_eq!(pair.sequence_epoch(), 1);
    assert_eq!(pair.reuses(), 2);
    assert_eq!(pair.resets(), 0);
    assert_eq!(pair.reset_cleared(), 0);

    let allocs_before_reset = pair.driver_counters().buffer_allocs;
    let handles_before_reset = pair.live_handles();
    let identity_before_reset = pair
        .shared_owner_identity()
        .expect("shared owner before reset");
    let epoch_before = pair.sequence_epoch();

    let receipt = pair.reset().expect("logical reset");
    assert_eq!(receipt.previous_epoch, epoch_before);
    assert_eq!(receipt.sequence_epoch, epoch_before + 1);
    assert_eq!(receipt.previous_valid_len, 3);
    assert_eq!(receipt.valid_len, 0);
    assert!(!receipt.cache_cleared);
    assert!(!receipt.buffers_zero_filled);
    assert_eq!(receipt.uploads, 0);
    assert_eq!(pair.valid_len(), 0);
    assert_eq!(pair.sequence_epoch(), epoch_before + 1);
    assert_eq!(pair.phase(), SequencePhase::Fresh);
    assert_eq!(pair.resets(), 1);
    assert_eq!(pair.reset_cleared(), 3);
    assert_eq!(pair.driver_counters().buffer_allocs, allocs_before_reset);
    assert_eq!(pair.live_handles(), handles_before_reset);
    assert_eq!(
        pair.shared_owner_identity()
            .expect("shared owner after reset"),
        identity_before_reset
    );

    let stale = pair
        .execute_invocation(&invocation(
            DeviceExecuteInvocationMode::Prefill,
            None,
            0,
            0,
            2,
        ))
        .expect_err("stale epoch is pre-dispatch");
    assert_eq!(stale.code, "E_KV_STALE");
    assert_eq!(pair.valid_len(), 0);
    assert_eq!(pair.sequence_epoch(), epoch_before + 1);
    assert_eq!(pair.resets(), 1);

    let replay = pair
        .execute_invocation(&invocation_on(
            DeviceExecuteInvocationMode::Prefill,
            None,
            0,
            0,
            2,
            epoch_before + 1,
        ))
        .expect("fresh-sequence prefill after reset");
    assert_eq!(replay.launch_entries, vec!["addita"]);
    assert_eq!(pair.valid_len(), 2);
    assert_eq!(pair.sequence_epoch(), epoch_before + 1);
    assert_eq!(pair.phase(), SequencePhase::Prefill);
    assert_eq!(pair.driver_counters().buffer_allocs, allocs_before_reset);
    pair.teardown().expect("teardown");
}

#[test]
fn paired_pre_dispatch_failure_leaves_sequence_unchanged() {
    let mut host = paired_host();
    let mut pair = host
        .prepare_paired_session(
            &prefill_descriptor(),
            &decode_descriptor(),
            vec![11, 22],
            RopeConfig {
                head_dim: 8,
                theta: 10_000.0,
            },
            &std::collections::BTreeMap::from([(10, vec![10.0, 20.0])]),
            &std::collections::BTreeMap::new(),
            "model",
            "session",
        )
        .expect("prepare pair");

    pair.execute_invocation(&invocation(
        DeviceExecuteInvocationMode::Prefill,
        None,
        0,
        0,
        2,
    ))
    .expect("prefill");
    let epoch = pair.sequence_epoch();
    let valid_len = pair.valid_len();
    let reuses = pair.reuses();
    let handles = pair.live_handles();
    let identity = pair.shared_owner_identity().expect("identity");

    let phase_err = pair
        .execute_invocation(&invocation(
            DeviceExecuteInvocationMode::Prefill,
            None,
            0,
            0,
            2,
        ))
        .expect_err("second prefill is illegal until reset");
    assert_eq!(phase_err.code, E_KV_PHASE);
    assert_eq!(pair.sequence_epoch(), epoch);
    assert_eq!(pair.valid_len(), valid_len);
    assert_eq!(pair.reuses(), reuses);
    assert_eq!(pair.resets(), 0);
    assert_eq!(pair.live_handles(), handles);
    assert_eq!(
        pair.shared_owner_identity().expect("identity after reject"),
        identity
    );

    let decode = pair
        .execute_invocation(&invocation(
            DeviceExecuteInvocationMode::ScalarDecode,
            Some(42),
            2,
            2,
            3,
        ))
        .expect("decode remains legal after a pre-dispatch reject");
    assert_eq!(decode.launch_entries, vec!["observa"]);
    assert_eq!(pair.valid_len(), 3);
    pair.teardown().expect("teardown");
}
