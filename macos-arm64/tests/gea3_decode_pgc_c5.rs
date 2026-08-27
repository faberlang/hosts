//! PGC-C5 additive device proof.
//!
//! The fixture is intentionally small and arithmetic-capable on the fake
//! Metal driver.  It makes the same distinction as the fixed-1000 prefill
//! route: immutable weight-shaped inputs are prepared once, while mutable
//! activation values are copied for each invocation.  The test records the
//! copy-in census and the device receipt without changing the shared Gea3
//! decode test or the host session implementation.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

use faber_host_macos_arm64::composite_host::CompositeHost;
use faber_host_macos_arm64::device_descriptor::{
    DescriptorBuffer, DescriptorBufferVersion, DescriptorDataFlow, DescriptorEndOfRunResult,
    DescriptorKernel, DescriptorLaunch, DescriptorResult, DeviceBufferInitialization,
    DeviceBufferLifetime, DeviceBufferRole, DeviceDataType, DeviceDescriptor,
    DeviceProgramLifetime,
};
use faber_host_macos_arm64::device_host::DeviceRuntime;
use faber_host_macos_arm64::{FakeMetalDriver, MetalHostSession};
use host_coordinator::DeviceBackend;
use serde_json::Value;

const MODULE_IMAGE: &[u8] = b"// PGC-C5 prefill residency probe";
/// Standing before-oracle value from the PGC-C5 card.  The post-change
/// physical receipt must show only the dynamic prefill staging below it.
const STANDING_PREFILL_RESTAGING_BYTES: u64 = 23_000_000;
const STANDING_PREFILL_COPY_HANDLES: u64 = 1_089;
const WEIGHT_ID: u32 = 1;
const ACTIVATION_ID: u32 = 2;
const LOGITS_ID: u32 = 3;
const ELEMENTS: u64 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CopyInCensus {
    handles: usize,
    bytes: u64,
}

fn input_census(descriptor: &DeviceDescriptor, resident: bool) -> CopyInCensus {
    let mut seen = BTreeSet::new();
    let mut census = CopyInCensus {
        handles: 0,
        bytes: 0,
    };
    for kernel in &descriptor.kernels {
        for slot in &kernel.buffers {
            let key = (slot.buffer_id, slot.version);
            if slot.role != DeviceBufferRole::Input || !seen.insert(key) {
                continue;
            }
            if resident && slot.lifetime == DeviceBufferLifetime::PerProgram {
                continue;
            }
            census.handles += 1;
            census.bytes += slot.element_count * slot.element_ty.byte_width() as u64;
        }
    }
    census
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

fn slot(
    buffer_id: u32,
    name: &str,
    role: DeviceBufferRole,
    lifetime: DeviceBufferLifetime,
    initialization: DeviceBufferInitialization,
    binding: u32,
) -> DescriptorBuffer {
    DescriptorBuffer {
        buffer_id,
        buffer_name: name.to_owned(),
        semantic_value: buffer_id,
        role,
        lifetime,
        initialization,
        binding,
        element_ty: DeviceDataType::F32,
        element_count: ELEMENTS,
        version: 1,
    }
}

/// A prefill-shaped one-launch program: `activation + weight → logits`.
///
/// The same kernel slot facts are exercised in both regimes.  Only the
/// declared program lifetime changes: `SingleRun` is the re-staging control,
/// and `RepeatingStep` is the C5 prepared-resident path.
fn prefill_descriptor(lifetime: DeviceProgramLifetime) -> DeviceDescriptor {
    let kernels = vec![DescriptorKernel {
        entry: "prefill_rmsnorm".to_owned(),
        buffers: vec![
            slot(
                ACTIVATION_ID,
                "prefill.activation",
                DeviceBufferRole::Input,
                DeviceBufferLifetime::PerStep,
                DeviceBufferInitialization::ZeroFill,
                0,
            ),
            slot(
                WEIGHT_ID,
                "blk.00.attn_norm.weight",
                DeviceBufferRole::Input,
                DeviceBufferLifetime::PerProgram,
                DeviceBufferInitialization::HostProvided,
                1,
            ),
            slot(
                LOGITS_ID,
                "prefill.logits",
                DeviceBufferRole::Output,
                DeviceBufferLifetime::ObservationPoint,
                DeviceBufferInitialization::KernelInitialized,
                2,
            ),
        ],
        grid: [1, 1, 1],
        block: [4, 1, 1],
    }];
    let mut descriptor = DeviceDescriptor {
        backend: DeviceBackend::Metal,
        module_image: MODULE_IMAGE.to_vec(),
        buffer_versions: buffer_versions_for(&kernels),
        kernels,
        launches: vec![DescriptorLaunch {
            id: 1,
            kernel_index: 0,
        }],
        program_lifetime: lifetime,
        data_flow: Vec::<DescriptorDataFlow>::new(),
        roots: vec![1],
        results: vec![DescriptorResult {
            buffer_id: LOGITS_ID,
            version: 1,
            produced_by: 1,
            at_launch: 1,
        }],
        end_of_run_results: Vec::<DescriptorEndOfRunResult>::new(),
    };
    descriptor.buffer_versions = buffer_versions_for(&descriptor.kernels);
    descriptor
}

fn fake_metal_host() -> CompositeHost {
    let runtime = DeviceRuntime::Metal(
        MetalHostSession::with_driver(Box::new(FakeMetalDriver::default()))
            .expect("fake Metal admission"),
    );
    CompositeHost::with_device(runtime, "fake-metal-device").expect("fake Metal composite")
}

fn weights() -> BTreeMap<u32, Vec<f32>> {
    BTreeMap::from([(WEIGHT_ID, vec![10.0, 20.0, 30.0, 40.0])])
}

fn inputs(activation: [f32; 4]) -> BTreeMap<u32, Vec<f32>> {
    BTreeMap::from([(ACTIVATION_ID, activation.to_vec())])
}

fn single_run_inputs(activation: [f32; 4]) -> BTreeMap<u32, Vec<f32>> {
    let mut values = inputs(activation);
    values.extend(weights());
    values
}

#[test]
fn gea3_pgc_c5_prefill_weights_are_once_resident_and_activations_stay_dynamic() {
    let repeating = prefill_descriptor(DeviceProgramLifetime::RepeatingStep);
    let single = prefill_descriptor(DeviceProgramLifetime::SingleRun);
    assert!(
        repeating.validate().is_ok(),
        "prepared descriptor must admit"
    );
    assert!(single.validate().is_ok(), "single-run control must admit");

    // The control path stages both the weight-shaped input and the mutable
    // activation.  The prepared path stages only the activation; this is the
    // same census that the fixed-1000 physical receipt reports in bytes.
    let control_census = input_census(&single, false);
    let resident_census = input_census(&repeating, true);
    assert_eq!(
        control_census,
        CopyInCensus {
            handles: 2,
            bytes: 32
        }
    );
    assert_eq!(
        resident_census,
        CopyInCensus {
            handles: 1,
            bytes: 16
        }
    );
    assert_eq!(
        control_census.bytes - resident_census.bytes,
        16,
        "the resident path removes the weight-shaped copy-in from each invocation"
    );

    // SingleRun is the red/control shape: each prefill invocation receives
    // both host inputs and produces the corresponding arithmetic result.
    let mut control_host = fake_metal_host();
    for (activation, expected) in [
        ([1.0, 0.0, 0.0, 0.0], vec![11.0, 20.0, 30.0, 40.0]),
        ([0.0, 1.0, 0.0, 0.0], vec![10.0, 21.0, 30.0, 40.0]),
    ] {
        let receipt = control_host
            .execute_descriptor(&single, &single_run_inputs(activation))
            .expect("single-run prefill control");
        assert_eq!(receipt.program_lifetime, DeviceProgramLifetime::SingleRun);
        assert_eq!(receipt.copy_ins, control_census.handles);
        assert_eq!(
            receipt.transfers, 3,
            "two copy-ins plus one logits readback"
        );
        assert_eq!(receipt.outputs.get(&LOGITS_ID), Some(&expected));
        assert_eq!(receipt.per_program_buffers, vec![WEIGHT_ID]);
        assert_eq!(receipt.observation_buffers, vec![LOGITS_ID]);
    }
    assert_eq!(
        control_host
            .device()
            .expect("control device")
            .live_handle_count(),
        0,
        "the control sessions tear down after each invocation"
    );

    // RepeatingStep is the green C5 shape: prepare uploads the immutable
    // weight once, then each resident prefill copies only its mutable
    // activation.  Outputs change with the activation, proving that the
    // resident promotion did not retain request state.
    let mut resident_host = fake_metal_host();
    let mut prepared = resident_host
        .prepare_resident_session(&repeating, &weights())
        .expect("prepare resident prefill");
    for (activation, expected) in [
        ([1.0, 0.0, 0.0, 0.0], vec![11.0, 20.0, 30.0, 40.0]),
        ([0.0, 1.0, 0.0, 0.0], vec![10.0, 21.0, 30.0, 40.0]),
    ] {
        let receipt = prepared
            .execute_step(&inputs(activation))
            .expect("resident prefill invocation");
        assert_eq!(
            receipt.program_lifetime,
            DeviceProgramLifetime::RepeatingStep
        );
        assert_eq!(receipt.copy_ins, resident_census.handles);
        assert_eq!(
            receipt.transfers, 2,
            "one activation copy-in plus one logits readback"
        );
        assert_eq!(receipt.outputs.get(&LOGITS_ID), Some(&expected));
        assert_eq!(receipt.per_program_buffers, vec![WEIGHT_ID]);
        assert_eq!(receipt.per_step_buffers, vec![ACTIVATION_ID]);
        assert_eq!(receipt.observation_buffers, vec![LOGITS_ID]);
        assert_eq!(
            receipt.releases, 0,
            "resident temporaries return to the pool"
        );
    }

    let receipt = prepared.receipt();
    assert_eq!(receipt.counters.prepares, 1);
    assert_eq!(receipt.counters.reuses, 2);
    assert_eq!(receipt.module_reloads, 0);
    assert_eq!(receipt.per_program_reallocs, 0);
    assert_eq!(receipt.timing.lifecycle.weight_uploads, 1);
    assert_eq!(receipt.timing.lifecycle.old_prefix_copy_bytes, 0);
    assert_eq!(receipt.timing.lifecycle.full_cache_clear_bytes, 0);

    let final_receipt = prepared.teardown().expect("resident prefill teardown");
    assert_eq!(final_receipt.counters.releases, 1);
    assert_eq!(final_receipt.live_handles, 0);
    assert_eq!(
        resident_host
            .device()
            .expect("resident device")
            .live_handle_count(),
        0,
        "the prepared session releases weights and pooled temporaries"
    );
}

/// Validate one captured fixed-1000 paired-parity receipt.  This remains an
/// ignored evidence test because a physical Metal receipt is an explicit
/// measurement input, not a repository fixture.
#[test]
#[ignore = "requires the fixed-1000 physical receipt and parity companion"]
fn gea3_pgc_c5_fixed1000_prefill_receipt_reports_reduced_staging() {
    let receipt_path = PathBuf::from(
        std::env::var_os("GEA3_METAL_RECEIPT")
            .expect("GEA3_METAL_RECEIPT must identify the physical receipt"),
    );
    let companion_path = PathBuf::from(
        std::env::var_os("GEA3_PARITY_TIMING_COMPANION")
            .expect("GEA3_PARITY_TIMING_COMPANION must identify the parity companion"),
    );
    let receipt: Value =
        serde_json::from_slice(&fs::read(&receipt_path).expect("read fixed-1000 physical receipt"))
            .expect("physical receipt JSON");
    let companion: Value = serde_json::from_slice(
        &fs::read(&companion_path).expect("read fixed-1000 parity companion"),
    )
    .expect("parity companion JSON");

    assert_eq!(receipt["schema"], "gea3-metal-receipt-v1");
    assert_eq!(companion["schema"], "gea3-parity-timing-companion-v1");
    let steps = receipt["steps"].as_array().expect("physical receipt steps");
    let prefill = steps
        .iter()
        .find(|step| step["mode"] == "prefill")
        .expect("prefill physical receipt step");
    assert!(prefill["timing_us"]["wall"].is_number());
    assert!(prefill["input_uploads"].is_number());

    let prefill_phases = companion["phases"]["prefill"]
        .as_array()
        .expect("prefill parity phases");
    assert_eq!(
        prefill_phases.len(),
        1,
        "exactly one paired prefill capture"
    );
    let upload = &prefill_phases[0]["host_input_upload"];
    let bytes = upload["bytes"]["value"]
        .as_u64()
        .expect("measured prefill upload bytes");
    let copies = upload["copies"]["value"]
        .as_u64()
        .expect("measured prefill upload handle count");
    assert!(bytes < STANDING_PREFILL_RESTAGING_BYTES);
    assert!(copies > 0 && copies <= STANDING_PREFILL_COPY_HANDLES);
    assert!(upload["duration_us"]["value"].is_number());
}
