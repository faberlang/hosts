//! E03-U1 prepared resident-session executor tests (fake-backend lanes).
//!
//! These tests prove the prepared resident-session done-when on the
//! composite host with injected fake drivers: (a) one session binds an
//! admitted descriptor with weights once-init and device-resident
//! (`PerProgram`, HostProvided); (b) ≥2 sequential executions record
//! reload = 0 and PerProgram reallocation = 0 with identical allocated
//! buffers; (c) a prompt-scoped reset clears state-buffer content and
//! retains allocation — deterministic replay of the first prompt after the
//! reset matches token-for-token; (d) teardown leaves zero live handles;
//! (e) the prepared-session receipt prints prepare/reuse/reset/release
//! counts; and (f) a dense direct-loaded weight-only descriptor admits
//! once-init weights without inventing a persistent state buffer. The fakes
//! prove sequencing only (real-device proofs are the EXEC-03 successor units).

use std::collections::BTreeMap;

use faber_host_macos_arm64::composite_host::{
    CompositeHost, DeviceByteBuffer, PreparedResidentSession,
};
use faber_host_macos_arm64::device_descriptor::{
    fnv1a64, DescriptorBuffer, DescriptorBufferVersion, DescriptorDataFlow, DescriptorKernel,
    DescriptorLaunch, DescriptorResult, DeviceBufferInitialization, DeviceBufferLifetime,
    DeviceBufferRole, DeviceDataType, DeviceDescriptor, DeviceProgramLifetime, E_DEVICE_DESCRIPTOR,
    E_DEVICE_SHAPE_MISMATCH,
};
use faber_host_macos_arm64::device_host::DeviceRuntime;
use faber_host_macos_arm64::device_registry::FakeFailureStage;
use faber_host_macos_arm64::metal_host::E_METAL_DRIVER;
use faber_host_macos_arm64::{CudaHostSession, FakeCudaDriver, FakeMetalDriver, MetalHostSession};
use host_coordinator::DeviceBackend;

type HostResult<T> = Result<T, faber_host_macos_arm64::HostError>;

const MODULE_IMAGE: &[u8] = b"// fake compiler-owned module image";

// ---------------------------------------------------------------------------
// The E03-U1 prepared-decode fixture
// ---------------------------------------------------------------------------

/// The prepared-decode descriptor: one resident decode step is
/// `h = t + w` (add_one), `s += h` (accumulate), `l = s` (observa), where
/// `w` are the once-init weights, `t` is the per-token input, `s` is the
/// device-resident state, and `l` is the observed logits.
///
/// | Buffer | Role → class | Init |
/// | --- | --- | --- |
/// | w (1) | Input → `PerProgram` | HostProvided (once-init weights) |
/// | t (2) | Input → `PerStep` | ZeroFill (per-token input, copied per reuse) |
/// | h (3) | InOut → `PerStep` | KernelInitialized (per-step hidden) |
/// | s (4) | InOut → `PerProgram` | ZeroFill (device-resident state) |
/// | l (5) | Output → `ObservationPoint` | KernelInitialized (logits) |
fn prepared_decode_descriptor(backend: DeviceBackend) -> DeviceDescriptor {
    let mut descriptor = DeviceDescriptor {
        backend,
        module_image: MODULE_IMAGE.to_vec(),
        kernels: vec![
            DescriptorKernel {
                entry: "add_one".to_owned(),
                buffers: vec![
                    token_slot(2, "t", 0),
                    weights_slot(1, "w", 1),
                    kernel_init_slot(3, "h", 2),
                ],
                grid: [1, 1, 1],
                block: [4, 1, 1],
            },
            DescriptorKernel {
                entry: "accumulate".to_owned(),
                buffers: vec![kernel_init_slot(3, "h", 0), accumulation_slot(4, "s", 1)],
                grid: [1, 1, 1],
                block: [4, 1, 1],
            },
            DescriptorKernel {
                entry: "observa".to_owned(),
                buffers: vec![accumulation_slot(4, "s", 0), logits_slot(5, "l", 1)],
                grid: [1, 1, 1],
                block: [4, 1, 1],
            },
        ],
        launches: vec![
            DescriptorLaunch {
                id: 1,
                kernel_index: 0,
            },
            DescriptorLaunch {
                id: 2,
                kernel_index: 1,
            },
            DescriptorLaunch {
                id: 3,
                kernel_index: 2,
            },
        ],
        buffer_versions: Vec::new(),
        program_lifetime: DeviceProgramLifetime::RepeatingStep,
        // R2: the carried data-flow edges — launch 1 produces the per-step
        // hidden h, launch 2 produces the resident state s, launch 3
        // observes it.
        data_flow: vec![
            DescriptorDataFlow {
                buffer_id: 3,
                version: 1,
                producer: 1,
                consumer: 2,
            },
            DescriptorDataFlow {
                buffer_id: 4,
                version: 1,
                producer: 2,
                consumer: 3,
            },
        ],
        roots: vec![1],
        // F6: the declared observation point — the logits l, produced by
        // launch 3, read back once per reuse.
        results: vec![DescriptorResult {
            buffer_id: 5,
            version: 1,
            produced_by: 3,
            at_launch: 3,
        }],
        end_of_run_results: Vec::new(),
    };
    descriptor.buffer_versions = buffer_versions_for(&descriptor.kernels);
    descriptor
}

/// The once-init weights slot: `PerProgram` + HostProvided (copied exactly
/// once at prepare, never re-copied on later reuses).
fn weights_slot(id: u32, name: &str, binding: u32) -> DescriptorBuffer {
    DescriptorBuffer {
        buffer_id: id,
        buffer_name: name.to_owned(),
        semantic_value: id,
        role: DeviceBufferRole::Input,
        lifetime: DeviceBufferLifetime::PerProgram,
        initialization: DeviceBufferInitialization::HostProvided,
        binding,
        element_ty: DeviceDataType::F32,
        element_count: 4,
        version: 1,
    }
}

/// The per-token input slot: `PerStep` + ZeroFill (allocated per reuse,
/// zero-filled at allocation, then overwritten with the token value).
fn token_slot(id: u32, name: &str, binding: u32) -> DescriptorBuffer {
    DescriptorBuffer {
        buffer_id: id,
        buffer_name: name.to_owned(),
        semantic_value: id,
        role: DeviceBufferRole::Input,
        lifetime: DeviceBufferLifetime::PerStep,
        initialization: DeviceBufferInitialization::ZeroFill,
        binding,
        element_ty: DeviceDataType::F32,
        element_count: 4,
        version: 1,
    }
}

/// A `PerStep` slot fully written by a device kernel before any read.
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
        element_count: 4,
        version: 1,
    }
}

/// The device-resident state slot: `PerProgram` + ZeroFill (allocated once,
/// zero-filled once at creation, accumulates across reuses, cleared by the
/// prompt-scoped reset with the allocation retained).
fn accumulation_slot(id: u32, name: &str, binding: u32) -> DescriptorBuffer {
    DescriptorBuffer {
        buffer_id: id,
        buffer_name: name.to_owned(),
        semantic_value: id,
        role: DeviceBufferRole::InOut,
        lifetime: DeviceBufferLifetime::PerProgram,
        initialization: DeviceBufferInitialization::ZeroFill,
        binding,
        element_ty: DeviceDataType::F32,
        element_count: 4,
        version: 1,
    }
}

/// The observed logits slot: `ObservationPoint` + KernelInitialized (read
/// back and released per reuse).
fn logits_slot(id: u32, name: &str, binding: u32) -> DescriptorBuffer {
    DescriptorBuffer {
        buffer_id: id,
        buffer_name: name.to_owned(),
        semantic_value: id,
        role: DeviceBufferRole::Output,
        lifetime: DeviceBufferLifetime::ObservationPoint,
        initialization: DeviceBufferInitialization::KernelInitialized,
        binding,
        element_ty: DeviceDataType::F32,
        element_count: 4,
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

/// A dense-row-shaped resident descriptor: direct-loaded GGUF weight `w` is
/// `PerProgram` + `HostProvided`, while token `t` is `PerStep`. There is no
/// prompt-state buffer, so the adapter must not require the E03 reset axis.
fn weight_only_descriptor(backend: DeviceBackend) -> DeviceDescriptor {
    let mut descriptor = prepared_decode_descriptor(backend);
    descriptor.kernels = vec![DescriptorKernel {
        entry: "add_one".to_owned(),
        buffers: vec![
            token_slot(2, "t", 0),
            weights_slot(1, "w", 1),
            logits_slot(5, "l", 2),
        ],
        grid: [1, 1, 1],
        block: [4, 1, 1],
    }];
    descriptor.launches = vec![DescriptorLaunch {
        id: 1,
        kernel_index: 0,
    }];
    descriptor.data_flow.clear();
    descriptor.roots = vec![1];
    descriptor.results = vec![DescriptorResult {
        buffer_id: 5,
        version: 1,
        produced_by: 1,
        at_launch: 1,
    }];
    descriptor.buffer_versions = buffer_versions_for(&descriptor.kernels);
    descriptor
}

/// Once-init weights for [`prepared_decode_descriptor`]: w = [10, 20, 30,
/// 40]. The simulated decode then yields the expected logits below.
fn prepared_weights() -> BTreeMap<u32, Vec<f32>> {
    BTreeMap::from([(1, vec![10.0, 20.0, 30.0, 40.0])])
}

fn token(values: [f32; 4]) -> BTreeMap<u32, Vec<f32>> {
    BTreeMap::from([(2, values.to_vec())])
}

/// Prompt A: three tokens. With w = [10, 20, 30, 40], the first pass
/// observes the logits [t+w, t1+t2+2w, t1+t2+t3+3w]:
/// `[11,20,30,40]`, `[21,41,60,80]`, `[31,61,91,120]`.
fn prompt_a() -> Vec<BTreeMap<u32, Vec<f32>>> {
    vec![
        token([1.0, 0.0, 0.0, 0.0]),
        token([0.0, 1.0, 0.0, 0.0]),
        token([0.0, 0.0, 1.0, 0.0]),
    ]
}

/// EXEC-03 prompt stream: each prompt is long enough to exercise the
/// steady-state residency window, while the values stay small enough for the
/// fake backend's exact arithmetic. The two seeds make prompt B observably
/// distinct without changing the descriptor or its compiled session.
fn prompt_stream(seed: f32) -> Vec<BTreeMap<u32, Vec<f32>>> {
    (0..256)
        .map(|step| {
            let lane = step % 4;
            let mut values = [0.0_f32; 4];
            values[lane] = seed + step as f32 * 0.001;
            token(values)
        })
        .collect()
}

fn expected_first_pass_logits() -> Vec<Vec<f32>> {
    vec![
        vec![11.0, 20.0, 30.0, 40.0],
        vec![21.0, 41.0, 60.0, 80.0],
        vec![31.0, 61.0, 91.0, 120.0],
    ]
}

fn prepared_metal_composite() -> HostResult<CompositeHost> {
    let runtime = DeviceRuntime::Metal(
        MetalHostSession::with_driver(Box::new(
            FakeMetalDriver::default()
                .with_known_entry("add_one")
                .with_known_entry("accumulate")
                .with_known_entry("observa"),
        ))
        .expect("fake metal admit"),
    );
    CompositeHost::with_device(runtime, "fake-metal-device")
}

fn prepared_cuda_composite() -> HostResult<CompositeHost> {
    let runtime = DeviceRuntime::Cuda(
        CudaHostSession::with_driver(Box::new(
            FakeCudaDriver::default()
                .with_known_entry("add_one")
                .with_known_entry("accumulate")
                .with_known_entry("observa"),
        ))
        .expect("fake cuda admit"),
    );
    CompositeHost::with_device(runtime, "fake-cuda-device")
}

/// A fake-metal composite whose driver fails the `call`-th invocation of
/// `stage` (S2-3 failure injection for the resident-step error path).
fn prepared_metal_composite_failing(
    stage: FakeFailureStage,
    call: u32,
) -> HostResult<CompositeHost> {
    let runtime = DeviceRuntime::Metal(
        MetalHostSession::with_driver(Box::new(
            FakeMetalDriver::default()
                .with_known_entry("add_one")
                .with_known_entry("accumulate")
                .with_known_entry("observa")
                .with_failure_at(stage, call),
        ))
        .expect("fake metal admit"),
    );
    CompositeHost::with_device(runtime, "fake-metal-device")
}

/// Execute prompt A on a prepared session, asserting every observed logits
/// value matches the expected token-for-token sequence. Returns the
/// observed logits for the caller's replay comparison.
fn run_prompt(
    prepared: &mut PreparedResidentSession<'_>,
    expected_logits: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let mut logits = Vec::new();
    for (index, step_token) in prompt_a().iter().enumerate() {
        let receipt = prepared
            .execute_step(step_token)
            .expect("resident decode step");
        let observed = receipt.outputs.get(&5).cloned().expect("logits observed");
        assert_eq!(&observed, &expected_logits[index], "token {index} diverges");
        logits.push(observed);
    }
    logits
}

// ---------------------------------------------------------------------------
// DSB-4b: packed bytes once-init directly into the resident session buffers.
// ---------------------------------------------------------------------------

#[test]
fn prepared_session_accepts_raw_weight_bytes_on_both_fake_backends() {
    let bytes: Vec<u8> = [10.0_f32, 20.0, 30.0, 40.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let byte_weights = BTreeMap::from([(
        1,
        DeviceByteBuffer {
            bytes,
            dtype: DeviceDataType::U8,
        },
    )]);

    for (backend, mut host) in [
        (
            DeviceBackend::Metal,
            prepared_metal_composite().expect("fake metal composite"),
        ),
        (
            DeviceBackend::Cuda,
            prepared_cuda_composite().expect("fake cuda composite"),
        ),
    ] {
        let descriptor = prepared_decode_descriptor(backend);
        let mut prepared = PreparedResidentSession::prepare_with_weight_bytes(
            &mut host,
            &descriptor,
            &BTreeMap::new(),
            &byte_weights,
        )
        .expect("raw byte prepare");
        let receipt = prepared
            .execute_step(&prompt_a()[0])
            .expect("resident step");
        assert_eq!(receipt.outputs.get(&5), Some(&vec![11.0, 20.0, 30.0, 40.0]));
        prepared.teardown().expect("teardown");
        assert_eq!(host.device().expect("device").live_handle_count(), 0);
    }
}

// ---------------------------------------------------------------------------
// M1-U2: direct-loaded dense weights are resident even without prompt state.
// ---------------------------------------------------------------------------

#[test]
fn prepared_session_resident_dense_weights_need_no_state_buffer() {
    let mut host = prepared_metal_composite().expect("metal composite");
    let descriptor = weight_only_descriptor(DeviceBackend::Metal);
    let mut prepared = host
        .prepare_resident_session(&descriptor, &prepared_weights())
        .expect("weight-only resident prepare");

    assert_eq!(
        prepared.session_handle_count(),
        2,
        "module + resident weight"
    );
    assert_eq!(prepared.driver_counters().module_loads, 1);
    for (index, token_values) in prompt_a().into_iter().take(2).enumerate() {
        let receipt = prepared
            .execute_step(&token_values)
            .expect("resident dense step");
        assert_eq!(receipt.copy_ins, 1, "only the PerStep token is copied");
        assert_eq!(receipt.pool_returns, 2);
        if index == 0 {
            assert_eq!(receipt.pool_allocations, 2);
            assert_eq!(receipt.pool_reuses, 0);
        } else {
            assert_eq!(receipt.pool_allocations, 0);
            assert_eq!(receipt.pool_reuses, 2);
        }
        assert_eq!(receipt.releases, 0);
        assert_eq!(receipt.per_program_buffers, vec![1]);
        assert_eq!(receipt.per_step_buffers, vec![2]);
        assert_eq!(receipt.observation_buffers, vec![5]);
        assert_eq!(
            receipt
                .resource_graph
                .iter()
                .find(|buffer| buffer.id == 1)
                .map(|buffer| buffer.lifetime),
            Some(DeviceBufferLifetime::PerProgram)
        );
        assert_eq!(
            receipt
                .resource_graph
                .iter()
                .find(|buffer| buffer.id == 2)
                .map(|buffer| buffer.lifetime),
            Some(DeviceBufferLifetime::PerStep)
        );
    }

    let timing = serde_json::to_value(prepared.receipt().timing).expect("timing receipt");
    assert_eq!(
        timing["steady_state"]["encode"]["duration_us"]["status"],
        "measured"
    );
    assert_eq!(
        timing["steady_state"]["submit"]["duration_us"]["status"],
        "measured"
    );
    assert_eq!(
        timing["steady_state"]["wait"]["duration_us"]["status"],
        "not_measured"
    );
    assert_eq!(timing["lifecycle"]["module_reloads"], 0);
    assert_eq!(timing["lifecycle"]["persistent_reallocations"], 0);
    assert_eq!(timing["lifecycle"]["weight_uploads"], 1);
    assert_eq!(timing["lifecycle"]["old_prefix_copy_bytes"], 0);
    assert_eq!(timing["lifecycle"]["full_cache_clear_bytes"], 0);

    assert_eq!(prepared.driver_counters().module_loads, 1);
    assert_eq!(prepared.receipt().module_reloads, 0);
    assert_eq!(prepared.receipt().per_program_reallocs, 0);
    assert_eq!(prepared.reset_prompt().expect("no-op state reset"), 0);

    let receipt = prepared.teardown().expect("teardown");
    assert_eq!(receipt.counters.reuses, 2);
    assert_eq!(receipt.module_reloads, 0);
    assert_eq!(receipt.per_program_reallocs, 0);
    assert_eq!(receipt.live_handles, 0);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

// ---------------------------------------------------------------------------
// E03-U1 done-when (a): prepare binds an admitted descriptor with weights
// once-init and device-resident (PerProgram, HostProvided)
// ---------------------------------------------------------------------------

#[test]
fn prepared_session_prepare_once_inits_resident_weights_and_state() {
    let mut host = prepared_metal_composite().expect("metal composite");
    let descriptor = prepared_decode_descriptor(DeviceBackend::Metal);

    let prepared = host
        .prepare_resident_session(&descriptor, &prepared_weights())
        .expect("prepare");

    // Creation: module + the two PerProgram buffers (weights w, state s)
    // allocated once. The per-step (t, h) and observation (l) buffers are
    // not allocated until a reuse.
    assert_eq!(prepared.session_handle_count(), 3);
    assert_eq!(prepared.module_hash(), fnv1a64(MODULE_IMAGE));
    let counters = prepared.driver_counters();
    assert_eq!(counters.module_loads, 1);
    assert_eq!(counters.buffer_allocs, 2);

    // The program-graph identity rides the prepared session too.
    assert_eq!(
        prepared.receipt().program_graph_hash,
        descriptor.program_graph_hash()
    );

    // The receipt starts at one prepare and zero reuses/resets/releases.
    let receipt = prepared.receipt();
    assert_eq!(receipt.counters.prepares, 1);
    assert_eq!(receipt.counters.reuses, 0);
    assert_eq!(receipt.counters.resets, 0);
    assert_eq!(receipt.counters.releases, 0);
    assert_eq!(receipt.module_reloads, 0);
    assert_eq!(receipt.per_program_reallocs, 0);
    assert_eq!(receipt.live_handles, 3);

    prepared.teardown().expect("teardown");
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

// ---------------------------------------------------------------------------
// E03-U1 done-when (b): ≥2 sequential executions record reload = 0 and
// PerProgram reallocation = 0 (allocated buffers identical)
// ---------------------------------------------------------------------------

#[test]
fn prepared_session_reuses_record_reload_zero_realloc_zero_identical_buffers() {
    let mut host = prepared_metal_composite().expect("metal composite");
    let descriptor = prepared_decode_descriptor(DeviceBackend::Metal);

    let mut prepared = host
        .prepare_resident_session(&descriptor, &prepared_weights())
        .expect("prepare");

    let mut executions = Vec::new();
    for (index, step_token) in prompt_a().iter().enumerate() {
        let receipt = prepared
            .execute_step(step_token)
            .expect("resident decode step");
        assert_eq!(
            receipt.copy_ins, 1,
            "only the per-token input is copied per reuse; the once-init weights are never re-copied"
        );
        assert_eq!(receipt.pool_returns, 3);
        if index == 0 {
            assert_eq!(receipt.pool_allocations, 3);
            assert_eq!(receipt.pool_reuses, 0);
        } else {
            assert_eq!(receipt.pool_allocations, 0);
            assert_eq!(receipt.pool_reuses, 3);
        }
        assert_eq!(receipt.releases, 0);
        assert_eq!(
            receipt.launches, 1,
            "W8-U1: three kernel encodes batch into one command-buffer submit"
        );
        assert_eq!(receipt.syncs, 1);
        assert_eq!(
            receipt.program_lifetime,
            DeviceProgramLifetime::RepeatingStep
        );
        executions.push(receipt);
    }

    // ≥2 sequential executions on one session: the allocated buffers are
    // identical and the PerProgram allocations never change.
    assert_eq!(executions.len(), 3);
    for pair in executions.windows(2) {
        assert_eq!(pair[0].allocated_buffers, pair[1].allocated_buffers);
        assert_eq!(
            pair[0].allocated_buffer_versions,
            pair[1].allocated_buffer_versions
        );
        assert_eq!(pair[0].per_program_buffers, pair[1].per_program_buffers);
        assert_eq!(
            pair[0].per_program_buffer_versions,
            pair[1].per_program_buffer_versions
        );
    }
    assert_eq!(executions[0].per_program_buffers, vec![1, 4]);
    assert_eq!(
        executions[0].per_program_buffer_versions,
        vec![(1, 1), (4, 1)]
    );

    // Between reuses the session holds module + PerProgram weights/state +
    // the three pooled temporary handles; the module was loaded exactly once.
    assert_eq!(prepared.session_handle_count(), 6);
    assert_eq!(prepared.driver_counters().module_loads, 1);

    // The prepared-session receipt records reload = 0 and PerProgram
    // reallocation = 0 across the three reuses.
    let receipt = prepared.receipt();
    assert_eq!(receipt.counters.reuses, 3);
    assert_eq!(receipt.module_reloads, 0);
    assert_eq!(receipt.per_program_reallocs, 0);
    assert_eq!(receipt.live_handles, 6);

    prepared.teardown().expect("teardown");
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

// ---------------------------------------------------------------------------
// E03-U1 done-when (c)+(d)+(e): prompt-scoped reset clears state content
// and retains allocation — deterministic replay matches token-for-token;
// teardown leaves zero live handles; the receipt prints the lifecycle
// counts
// ---------------------------------------------------------------------------

#[test]
fn prepared_session_reset_clears_state_and_replay_matches_token_for_token() {
    let mut host = prepared_metal_composite().expect("metal composite");
    let descriptor = prepared_decode_descriptor(DeviceBackend::Metal);

    let mut prepared = host
        .prepare_resident_session(&descriptor, &prepared_weights())
        .expect("prepare");

    // Prompt A, first pass: three tokens accumulate into the resident state.
    let expected = expected_first_pass_logits();
    let first_pass = run_prompt(&mut prepared, &expected);

    // The prompt-scoped reset clears the state-buffer content and retains
    // the allocation: the handle count and the driver counters do not move.
    let allocs_before_reset = prepared.driver_counters().buffer_allocs;
    let releases_before_reset = prepared.driver_counters().buffer_releases;
    let cleared = prepared.reset_prompt().expect("prompt-scoped reset");
    assert_eq!(
        cleared, 1,
        "exactly the device-resident state buffer (s) is cleared"
    );
    assert_eq!(prepared.session_handle_count(), 6, "allocation retained");
    let counters = prepared.driver_counters();
    assert_eq!(
        counters.buffer_allocs, allocs_before_reset,
        "no re-allocation during the reset"
    );
    assert_eq!(
        counters.buffer_releases, releases_before_reset,
        "no release during the reset"
    );

    // Replay prompt A after the reset: the state restarted from the same
    // zeroed initial condition, so the replay matches token-for-token.
    let replay = run_prompt(&mut prepared, &expected);
    assert_eq!(
        replay, first_pass,
        "deterministic replay after the reset must match the first pass token-for-token"
    );

    // The final receipt: prepare=1, reuse=6, reset=1, release=1, reload=0,
    // realloc=0, zero live handles post-teardown (E03-U1 closeout evidence).
    let final_receipt = prepared.teardown().expect("teardown");
    assert_eq!(final_receipt.counters.prepares, 1);
    assert_eq!(final_receipt.counters.reuses, 6);
    assert_eq!(final_receipt.counters.resets, 1);
    assert_eq!(final_receipt.counters.releases, 1);
    assert_eq!(final_receipt.module_reloads, 0);
    assert_eq!(final_receipt.per_program_reallocs, 0);
    assert_eq!(final_receipt.live_handles, 0);
    println!("{}", final_receipt.spelling());

    // Teardown leaves zero live handles per the device registry and every
    // driver allocation balances (leak-free bar).
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
    let counters = host.device().expect("device").driver_counters();
    assert_eq!(counters.module_loads, 1);
    assert_eq!(counters.module_releases, 1);
    assert_eq!(counters.buffer_allocs, 2 + 3);
    assert_eq!(counters.buffer_releases, 2 + 3);
}

// ---------------------------------------------------------------------------
// E03-U1 done-when (d): teardown leaves zero live handles
// ---------------------------------------------------------------------------

/// EXEC-03 oracle: two prompts of at least 256 new-token steps share one
/// admitted session, with a prompt-scoped reset between them. The fake driver
/// makes the lifecycle proof deterministic; the dense-device receipt carries
/// the 419/315 census and 30/304 prompt-end pins separately.
#[test]
fn exec03_two_256_token_prompts_reuse_one_session_and_reset_state() {
    let mut host = prepared_metal_composite().expect("metal composite");
    let descriptor = prepared_decode_descriptor(DeviceBackend::Metal);
    let mut prepared = host
        .prepare_resident_session(&descriptor, &prepared_weights())
        .expect("prepare");
    let prompt_a = prompt_stream(1.0);
    let prompt_b = prompt_stream(2.0);

    let run_prompt = |prepared: &mut PreparedResidentSession<'_>,
                      prompt: &[BTreeMap<u32, Vec<f32>>],
                      first_prompt: bool| {
        assert_eq!(prompt.len(), 256);
        let mut last_logits = None;
        let mut allocated_buffers = None;
        for (step, inputs) in prompt.iter().enumerate() {
            let receipt = prepared.execute_step(inputs).expect("resident decode step");
            assert_eq!(receipt.copy_ins, 1, "only the PerStep token is copied");
            assert_eq!(receipt.launches, 1);
            assert_eq!(receipt.syncs, 1);
            assert_eq!(receipt.releases, 0);
            if step == 0 {
                if first_prompt {
                    assert_eq!(receipt.pool_allocations, 3);
                    assert_eq!(receipt.pool_reuses, 0);
                } else {
                    assert_eq!(receipt.pool_allocations, 0);
                    assert_eq!(receipt.pool_reuses, 3);
                }
                allocated_buffers = Some(receipt.allocated_buffers.clone());
            } else {
                assert_eq!(receipt.pool_allocations, 0);
                assert_eq!(receipt.pool_reuses, 3);
                assert_eq!(
                    receipt.allocated_buffers,
                    allocated_buffers.clone().expect("warm-up allocations")
                );
            }
            last_logits = Some(receipt.outputs.get(&5).cloned().expect("logits observed"));
        }
        last_logits.expect("prompt has a final observation")
    };

    let prompt_a_last = run_prompt(&mut prepared, &prompt_a, true);
    let after_a = prepared.receipt();
    assert_eq!(after_a.counters.prepares, 1);
    assert_eq!(after_a.counters.reuses, 256);
    assert_eq!(after_a.counters.resets, 0);
    assert_eq!(after_a.module_reloads, 0);
    assert_eq!(after_a.per_program_reallocs, 0);
    assert_eq!(after_a.live_handles, 6);
    assert!(prompt_a_last.iter().any(|value| *value != 0.0));

    let allocations_before_reset = prepared.driver_counters().buffer_allocs;
    let releases_before_reset = prepared.driver_counters().buffer_releases;
    assert_eq!(prepared.reset_prompt().expect("prompt-scoped reset"), 1);
    assert_eq!(prepared.session_handle_count(), 6);
    let after_reset_counters = prepared.driver_counters();
    assert_eq!(after_reset_counters.buffer_allocs, allocations_before_reset);
    assert_eq!(after_reset_counters.buffer_releases, releases_before_reset);
    assert_eq!(prepared.receipt().counters.resets, 1);

    let prompt_b_last = run_prompt(&mut prepared, &prompt_b, false);
    assert_ne!(
        prompt_a_last, prompt_b_last,
        "prompt B must start from its own input"
    );
    let before_teardown = prepared.receipt();
    assert_eq!(before_teardown.counters.reuses, 512);
    assert_eq!(before_teardown.counters.resets, 1);
    assert_eq!(before_teardown.module_reloads, 0);
    assert_eq!(before_teardown.per_program_reallocs, 0);
    assert_eq!(before_teardown.live_handles, 6);

    let final_receipt = prepared.teardown().expect("teardown");
    assert_eq!(final_receipt.counters.prepares, 1);
    assert_eq!(final_receipt.counters.reuses, 512);
    assert_eq!(final_receipt.counters.resets, 1);
    assert_eq!(final_receipt.counters.releases, 1);
    assert_eq!(final_receipt.module_reloads, 0);
    assert_eq!(final_receipt.per_program_reallocs, 0);
    assert_eq!(final_receipt.live_handles, 0);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

#[test]
fn prepared_session_teardown_releases_every_handle() {
    let mut host = prepared_metal_composite().expect("metal composite");
    let descriptor = prepared_decode_descriptor(DeviceBackend::Metal);

    let mut prepared = host
        .prepare_resident_session(&descriptor, &prepared_weights())
        .expect("prepare");
    prepared.execute_step(&prompt_a()[0]).expect("one reuse");
    prepared.reset_prompt().expect("one reset");

    let receipt = prepared.teardown().expect("teardown");
    assert_eq!(receipt.counters.releases, 1);
    assert_eq!(receipt.live_handles, 0);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);

    // Leak-free bar: every allocation balances and the module persists
    // nowhere past the release.
    let counters = host.device().expect("device").driver_counters();
    assert_eq!(counters.module_loads, counters.module_releases);
    assert_eq!(counters.buffer_allocs, counters.buffer_releases);
}

// ---------------------------------------------------------------------------
// Prepared-session admission: only the prepared-session shape is admitted,
// fail-closed with typed diagnostics before any session creation
// ---------------------------------------------------------------------------

#[test]
fn prepared_session_rejects_non_prepared_descriptors() {
    let mut host = prepared_metal_composite().expect("metal composite");

    // Not a RepeatingStep program: the prepared session is the RepeatingStep
    // once-init contract.
    let mut single = prepared_decode_descriptor(DeviceBackend::Metal);
    single.program_lifetime = DeviceProgramLifetime::SingleRun;
    let err = host
        .prepare_resident_session(&single, &prepared_weights())
        .err()
        .expect("a SingleRun program is not a prepared session");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);

    // No HostProvided weights: a prepared session needs once-init weights.
    let mut no_weights = prepared_decode_descriptor(DeviceBackend::Metal);
    for kernel in &mut no_weights.kernels {
        for buffer in &mut kernel.buffers {
            if buffer.buffer_id == 1 {
                buffer.initialization = DeviceBufferInitialization::ZeroFill;
            }
        }
    }
    let err = host
        .prepare_resident_session(&no_weights, &prepared_weights())
        .err()
        .expect("a prepared session needs once-init weights");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);

    // No rejected prepare left any handle behind.
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

// ---------------------------------------------------------------------------
// Misuse + error paths (S2-3): wrong backend, missing per-token input,
// failed resident step — every failure leaves zero live handles and the
// closed session refuses further use
// ---------------------------------------------------------------------------

#[test]
fn prepared_session_surfaces_fail_closed_on_misuse() {
    // Wrong-backend descriptor fails prepare with E_DEVICE_DESCRIPTOR.
    let mut host = prepared_metal_composite().expect("metal composite");
    let descriptor = prepared_decode_descriptor(DeviceBackend::Cuda);
    let err = host
        .prepare_resident_session(&descriptor, &prepared_weights())
        .err()
        .expect("a cuda descriptor on a metal session must fail prepare");
    assert_eq!(err.code, E_DEVICE_DESCRIPTOR);

    // A resident step without its per-token input fails closed and the
    // session releases every handle (S2-3 error-path teardown).
    let mut prepared = host
        .prepare_resident_session(
            &prepared_decode_descriptor(DeviceBackend::Metal),
            &prepared_weights(),
        )
        .expect("prepare");
    let err = prepared
        .execute_step(&BTreeMap::new())
        .expect_err("a missing per-token input must fail the resident step");
    assert_eq!(err.code, E_DEVICE_SHAPE_MISMATCH);
    assert_eq!(prepared.session_handle_count(), 0);

    // A closed prepared session refuses further reuses and resets.
    let err = prepared
        .execute_step(&prompt_a()[0])
        .expect_err("a closed prepared session refuses reuses");
    assert_eq!(err.code, "E_INTERNAL");
    let err = prepared
        .reset_prompt()
        .expect_err("a closed prepared session refuses resets");
    assert_eq!(err.code, "E_INTERNAL");

    // Teardown of the closed session is a safe no-op that still reports the
    // receipt.
    let receipt = prepared.teardown().expect("teardown of closed session");
    assert_eq!(receipt.counters.releases, 1);
    assert_eq!(receipt.live_handles, 0);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

/// A failed resident step (S2-3): the second launch of the step is injected
/// to fail; every handle is released, the prepared session closes, and
/// zero live handles survive.
#[test]
fn failed_resident_step_releases_every_handle() {
    let mut host =
        prepared_metal_composite_failing(FakeFailureStage::Launch, 2).expect("metal composite");
    let descriptor = prepared_decode_descriptor(DeviceBackend::Metal);

    let mut prepared = host
        .prepare_resident_session(&descriptor, &prepared_weights())
        .expect("prepare");
    let err = prepared
        .execute_step(&prompt_a()[0])
        .expect_err("the injected second-launch failure must fail the resident step");
    assert_eq!(err.code, E_METAL_DRIVER);
    assert_eq!(prepared.session_handle_count(), 0);

    drop(prepared);
    assert_eq!(host.device().expect("device").live_handle_count(), 0);
}

// ---------------------------------------------------------------------------
// Both fake backends pass the same prepared-session flow where the machine
// admits them (backend-neutral surface, E03-U1 hardware/backend authority)
// ---------------------------------------------------------------------------

#[test]
fn prepared_session_flow_on_both_fake_backends() {
    for backend in [DeviceBackend::Metal, DeviceBackend::Cuda] {
        let mut host = match backend {
            DeviceBackend::Metal => prepared_metal_composite().expect("metal composite"),
            DeviceBackend::Cuda => prepared_cuda_composite().expect("cuda composite"),
        };
        let descriptor = prepared_decode_descriptor(backend);
        let mut prepared = host
            .prepare_resident_session(&descriptor, &prepared_weights())
            .expect("prepare");

        // Two sequential reuses on one session: reload = 0, realloc = 0.
        prepared.execute_step(&prompt_a()[0]).expect("reuse 1");
        prepared.execute_step(&prompt_a()[1]).expect("reuse 2");
        let receipt = prepared.receipt();
        assert_eq!(receipt.counters.reuses, 2);
        assert_eq!(receipt.module_reloads, 0);
        assert_eq!(receipt.per_program_reallocs, 0);
        assert_eq!(receipt.live_handles, 6);

        // The prompt-scoped reset retains allocation on both lanes.
        prepared.reset_prompt().expect("reset");
        assert_eq!(prepared.session_handle_count(), 6);

        prepared.teardown().expect("teardown");
        assert_eq!(host.device().expect("device").live_handle_count(), 0);
    }
}
