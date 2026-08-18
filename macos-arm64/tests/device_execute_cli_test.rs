//! CLI + wire tests for `device-execute`.

use std::collections::BTreeMap;
use std::fs;
use std::process::Command;

use faber_host_macos_arm64::composite_host::CompositeHost;
use faber_host_macos_arm64::device_descriptor::{
    DescriptorBuffer, DescriptorBufferVersion, DescriptorKernel, DescriptorLaunch,
    DescriptorResult, DeviceBufferInitialization, DeviceBufferLifetime, DeviceBufferRole,
    DeviceDataType, DeviceDescriptor, DeviceProgramLifetime, E_DEVICE_DESCRIPTOR,
};
use faber_host_macos_arm64::device_execute::{
    descriptor_from_json, descriptor_to_json, inputs_from_json, inputs_to_json,
    parse_device_execute_args, receipt_to_json, DeviceExecuteReceipt,
};
use faber_host_macos_arm64::device_host::DeviceRuntime;
use faber_host_macos_arm64::{FakeMetalDriver, MetalHostSession};
use host_coordinator::DeviceBackend;
use serde_json::Value;

const MODULE_IMAGE: &[u8] = b"// fake compiler-owned module image";

fn add_slot(
    id: u32,
    name: &str,
    role: DeviceBufferRole,
    binding: u32,
    count: u64,
) -> DescriptorBuffer {
    let (lifetime, initialization) = match role {
        DeviceBufferRole::Input => (
            DeviceBufferLifetime::PerProgram,
            DeviceBufferInitialization::HostProvided,
        ),
        DeviceBufferRole::Output => (
            DeviceBufferLifetime::ObservationPoint,
            DeviceBufferInitialization::KernelInitialized,
        ),
        DeviceBufferRole::InOut => (
            DeviceBufferLifetime::PerStep,
            DeviceBufferInitialization::ZeroFill,
        ),
    };
    DescriptorBuffer {
        buffer_id: id,
        buffer_name: name.to_owned(),
        semantic_value: id,
        role,
        lifetime,
        initialization,
        binding,
        element_ty: DeviceDataType::F32,
        element_count: count,
        version: 1,
    }
}

fn elementwise_add_descriptor() -> DeviceDescriptor {
    let kernels = vec![DescriptorKernel {
        entry: "addita".to_owned(),
        buffers: vec![
            add_slot(1, "a", DeviceBufferRole::Input, 0, 2),
            add_slot(2, "b", DeviceBufferRole::Input, 1, 2),
            add_slot(3, "out", DeviceBufferRole::Output, 2, 2),
        ],
        grid: [1, 1, 1],
        block: [2, 1, 1],
    }];
    DeviceDescriptor {
        backend: DeviceBackend::Metal,
        module_image: MODULE_IMAGE.to_vec(),
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
            DescriptorBufferVersion {
                buffer_id: 3,
                version: 1,
                element_ty: DeviceDataType::F32,
                element_count: 2,
            },
        ],
        kernels,
        launches: vec![DescriptorLaunch {
            id: 1,
            kernel_index: 0,
        }],
        program_lifetime: DeviceProgramLifetime::SingleRun,
        data_flow: Vec::new(),
        roots: vec![1],
        results: vec![DescriptorResult {
            buffer_id: 3,
            version: 1,
            produced_by: 1,
            at_launch: 1,
        }],
        end_of_run_results: Vec::new(),
    }
}

fn add_inputs() -> BTreeMap<u32, Vec<f32>> {
    BTreeMap::from([(1, vec![1.0, 2.0]), (2, vec![3.0, 4.0])])
}

#[test]
fn parse_device_execute_args_requires_the_three_paths() {
    let err = parse_device_execute_args(&[]).expect_err("missing flags");
    assert!(err.contains("device-execute"), "{err}");
}

#[test]
fn parse_device_execute_args_rejects_unknown_flags() {
    let args = [
        "--descriptor".to_owned(),
        "d.json".to_owned(),
        "--module".to_owned(),
        "m.bin".to_owned(),
        "--inputs".to_owned(),
        "i.json".to_owned(),
        "--surprise".to_owned(),
    ];
    let err = parse_device_execute_args(&args).expect_err("unknown flag");
    assert!(err.contains("--surprise"), "{err}");
}

#[test]
fn parse_device_execute_args_accepts_explicit_backend() {
    let args = [
        "--backend".to_owned(),
        "metal".to_owned(),
        "--descriptor".to_owned(),
        "d.json".to_owned(),
        "--module".to_owned(),
        "m.bin".to_owned(),
        "--inputs".to_owned(),
        "i.json".to_owned(),
    ];
    let parsed = parse_device_execute_args(&args).expect("valid flags");
    assert_eq!(
        parsed.backend,
        Some(faber_host_macos_arm64::composite_host::DeviceSelection::Metal)
    );
}

#[test]
fn descriptor_json_round_trips_without_the_module_image() {
    let original = elementwise_add_descriptor();
    let json = descriptor_to_json(&original).expect("encode");
    let decoded = descriptor_from_json(&json, b"reloaded".to_vec()).expect("decode");
    assert_eq!(decoded.backend, DeviceBackend::Metal);
    assert_eq!(decoded.module_image, b"reloaded");
    assert_eq!(decoded.kernels, original.kernels);
    assert_eq!(decoded.launches, original.launches);
    assert_eq!(decoded.roots, original.roots);
    assert_eq!(decoded.results, original.results);
    assert_eq!(decoded.program_lifetime, DeviceProgramLifetime::SingleRun);
}

#[test]
fn inputs_json_round_trips_buffer_ids() {
    let inputs = add_inputs();
    let json = inputs_to_json(&inputs).expect("encode");
    let decoded = inputs_from_json(&json).expect("decode");
    assert_eq!(decoded, inputs);
}

#[test]
fn structurally_bad_descriptor_fails_before_host_open() {
    let mut descriptor = elementwise_add_descriptor();
    descriptor.kernels.clear();
    descriptor
        .validate()
        .expect_err("empty kernels must fail closed");
}

#[test]
fn wire_execute_on_fake_metal_returns_observation_outputs() {
    let descriptor = elementwise_add_descriptor();
    let json = descriptor_to_json(&descriptor).expect("encode");
    let decoded = descriptor_from_json(&json, MODULE_IMAGE.to_vec()).expect("decode");
    decoded.validate().expect("descriptor must validate");

    let runtime = DeviceRuntime::Metal(
        MetalHostSession::with_driver(Box::new(
            FakeMetalDriver::default().with_known_entry("addita"),
        ))
        .expect("fake metal admit"),
    );
    let mut host = CompositeHost::with_device(runtime, "fake-metal-device").expect("host");
    let receipt = host
        .execute_descriptor(&decoded, &add_inputs())
        .expect("execute");
    assert_eq!(receipt.outputs.get(&3), Some(&vec![4.0, 6.0]));
    assert_eq!(receipt.launches, 1);

    let wire = DeviceExecuteReceipt::from_host(&receipt);
    let encoded = receipt_to_json(&wire).expect("encode receipt");
    let parsed: Value = serde_json::from_slice(&encoded).expect("receipt json");
    assert_eq!(parsed["backend"], "metal");
    assert_eq!(parsed["launches"], 1);
    assert_eq!(parsed["outputs"]["3"], Value::from(vec![4.0, 6.0]));
}

#[test]
fn cli_device_execute_usage_exits_64() {
    let output = Command::new(env!("CARGO_BIN_EXE_faber-host-macos-arm64"))
        .args(["device-execute"])
        .output()
        .expect("run device-execute");
    assert_eq!(output.status.code(), Some(64));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("device-execute"), "{stderr}");
}

#[test]
fn cli_device_execute_malformed_descriptor_exits_2() {
    let dir = tempfile::tempdir().expect("tempdir");
    let descriptor = dir.path().join("descriptor.json");
    let module = dir.path().join("module.bin");
    let inputs = dir.path().join("inputs.json");
    fs::write(&descriptor, b"{not-json").expect("write descriptor");
    fs::write(&module, MODULE_IMAGE).expect("write module");
    fs::write(&inputs, b"{}").expect("write inputs");

    let output = Command::new(env!("CARGO_BIN_EXE_faber-host-macos-arm64"))
        .args([
            "device-execute",
            "--backend",
            "metal",
            "--descriptor",
            descriptor.to_str().expect("utf8"),
            "--module",
            module.to_str().expect("utf8"),
            "--inputs",
            inputs.to_str().expect("utf8"),
        ])
        .output()
        .expect("run device-execute");
    assert_eq!(output.status.code(), Some(2));
    let json: Value = serde_json::from_slice(&output.stdout).expect("error json");
    assert_eq!(json["code"], "E_INVALID_ARGS");
}

#[test]
fn cli_device_execute_invalid_descriptor_fails_closed() {
    let dir = tempfile::tempdir().expect("tempdir");
    let descriptor = dir.path().join("descriptor.json");
    let module = dir.path().join("module.bin");
    let inputs = dir.path().join("inputs.json");
    let mut bad = elementwise_add_descriptor();
    bad.kernels.clear();
    fs::write(&descriptor, descriptor_to_json(&bad).expect("encode")).expect("write descriptor");
    fs::write(&module, MODULE_IMAGE).expect("write module");
    fs::write(&inputs, b"{}").expect("write inputs");

    let output = Command::new(env!("CARGO_BIN_EXE_faber-host-macos-arm64"))
        .args([
            "device-execute",
            "--descriptor",
            descriptor.to_str().expect("utf8"),
            "--module",
            module.to_str().expect("utf8"),
            "--inputs",
            inputs.to_str().expect("utf8"),
        ])
        .output()
        .expect("run device-execute");
    assert_eq!(output.status.code(), Some(2));
    let json: Value = serde_json::from_slice(&output.stdout).expect("error json");
    assert_eq!(json["code"], E_DEVICE_DESCRIPTOR);
}
