//! GEA1-U4: map, bind, and execute the paired BF16/F32 Metal artifacts.
//!
//! The fake proof exercises only host admission, resident handles, binding
//! order, and readback bounds. It deliberately makes no numerical claim: the
//! fake driver's CPU simulation is not GPU evidence. The ignored test is the
//! physical proof and is fail-closed when Metal, the exported bundle, or any
//! frozen identity is missing.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use faber_host_macos_arm64::device_descriptor::{
    DescriptorBuffer, DescriptorBufferVersion, DescriptorKernel, DescriptorLaunch,
    DescriptorResult, DeviceBufferInitialization, DeviceBufferLifetime, DeviceBufferRole,
    DeviceDataType, DeviceDescriptor, DeviceProgramLifetime,
};
use faber_host_macos_arm64::metal_host::MetalLaunchBinding;
use faber_host_macos_arm64::{
    enumerate_metal_physical_devices, FakeMetalDriver, MetalHandleId, MetalHostSession,
};
use serde::Deserialize;
use serde_json::{json, Value};

const BF16_ENTRY: &str = "gemv_bf16_f32acc";
const F32_ENTRY: &str = "gemv_f32_f32acc";
const OUTPUT_ELEMENTS: usize = 320;
const INPUT_ELEMENTS: usize = 960;
const WEIGHT_ELEMENTS: usize = OUTPUT_ELEMENTS * INPUT_ELEMENTS;
const DISPATCH: [u32; 3] = [320, 1, 1];
// The tiled GEMV artifact uses one 8x8 threadgroup per output row.
const BLOCK: [u32; 3] = [8, 8, 1];
const MEASUREMENT_FIELDS: [&str; 20] = [
    "admission_us",
    "admission_bytes",
    "tensor_range_read_us",
    "tensor_range_read_bytes",
    "bf16_to_f32_us",
    "bf16_to_f32_bytes",
    "allocation_us",
    "upload_us",
    "upload_bytes",
    "pipeline_create_us",
    "binding_us",
    "warmups",
    "repetitions",
    "gpu_body_us",
    "launch_submit_orchestration_us",
    "sync_wait_us",
    "readback_us",
    "readback_bytes",
    "effective_bandwidth_bytes_per_s",
    "e2e_us",
];

#[derive(Debug, Deserialize)]
struct BundleManifest {
    schema: String,
    required_members: Vec<String>,
    entries: Vec<BundleEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct BundleEntry {
    entry: String,
    source: String,
    source_sha256: String,
    artifact: String,
    artifact_sha256: String,
    descriptor: String,
    descriptor_sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ArtifactDescriptor {
    entry: String,
    weight_dtype: String,
    weight_shape: [u64; 2],
    weight_range: [u64; 2],
    dispatch: [u64; 3],
    output_shape: [u64; 1],
    provenance: String,
}

#[derive(Debug, Clone)]
struct PreparedEntry {
    bundle: BundleEntry,
    descriptor: ArtifactDescriptor,
    descriptor_bytes: Vec<u8>,
    artifact_bytes: Vec<u8>,
}

#[derive(Debug)]
struct KernelRun {
    prepared: PreparedEntry,
    dtype: DeviceDataType,
    physical_dtype: String,
    weight_handle: MetalHandleId,
    input_handle: MetalHandleId,
    output_handle: MetalHandleId,
    weight_bytes: usize,
    input_bytes: usize,
    output_bytes: usize,
    output_raw: Vec<u8>,
    output_f32: Vec<f32>,
    admission_us: u64,
    allocation_us: u64,
    upload_us: u64,
    pipeline_create_us: u64,
    binding_us: u64,
    launch_submit_orchestration_us: u64,
    sync_wait_us: u64,
    gpu_body_us: u64,
    readback_us: u64,
    tensor_range_read_us: u64,
    bf16_to_f32_us: u64,
    numerical_rows: Vec<Value>,
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("faberlang workspace root")
}

fn sha256_file(path: &Path) -> String {
    let output = Command::new("shasum")
        .args(["-a", "256"])
        .arg(path)
        .output()
        .unwrap_or_else(|error| panic!("run shasum for {}: {error}", path.display()));
    assert!(
        output.status.success(),
        "shasum failed for {}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    let hash = String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_owned();
    assert_eq!(hash.len(), 64, "invalid SHA-256 for {}", path.display());
    hash
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut child = Command::new("shasum")
        .args(["-a", "256"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn shasum for bytes");
    child
        .stdin
        .take()
        .expect("shasum stdin")
        .write_all(bytes)
        .expect("write bytes to shasum");
    let output = child.wait_with_output().expect("wait for shasum");
    assert!(
        output.status.success(),
        "shasum stdin failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()
        .expect("shasum digest")
        .to_owned()
}

fn git_revision(path: &Path) -> String {
    let output = Command::new("git")
        .args(["-C"])
        .arg(path)
        .args(["rev-parse", "HEAD"])
        .output()
        .unwrap_or_else(|error| panic!("git revision {}: {error}", path.display()));
    assert!(
        output.status.success(),
        "git revision failed for {}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_owned()
}

fn read_range(path: &Path, range: [u64; 2]) -> Vec<u8> {
    assert!(range[1] > range[0], "empty range in {}", path.display());
    let length = usize::try_from(range[1] - range[0]).expect("range fits host usize");
    let mut file =
        File::open(path).unwrap_or_else(|error| panic!("open {}: {error}", path.display()));
    file.seek(SeekFrom::Start(range[0]))
        .unwrap_or_else(|error| panic!("seek {}: {error}", path.display()));
    let mut bytes = vec![0; length];
    file.read_exact(&mut bytes)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    bytes
}

fn decode_f32_le(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0, "F32 fixture is not word aligned");
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 2, 0, "BF16 values are not halfword aligned");
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
            f32::from_bits(bits << 16)
        })
        .collect()
}

/// Mirror the U5 scalar F32 GEMV oracle over the admitted logical matrix.
fn reference_gemv(weights: &[f32], input: &[f32]) -> Vec<f32> {
    assert_eq!(weights.len(), WEIGHT_ELEMENTS);
    assert_eq!(input.len(), INPUT_ELEMENTS);
    let mut output = Vec::with_capacity(OUTPUT_ELEMENTS);
    for row in weights.chunks_exact(INPUT_ELEMENTS) {
        let mut sum = 0.0_f32;
        for (&weight, &value) in row.iter().zip(input) {
            sum += weight * value;
        }
        output.push(sum);
    }
    output
}

fn measured(value: impl serde::Serialize) -> Value {
    json!({"value": value, "status": "measured"})
}

fn derived(value: impl serde::Serialize, formula: &str) -> Value {
    json!({"value": value, "status": "derived", "formula": formula})
}

fn unmeasured(reason: &str) -> Value {
    json!({"value": Value::Null, "status": "unmeasured", "reason": reason})
}

fn descriptor_dtype(spelling: &str) -> DeviceDataType {
    DeviceDataType::from_spelling(&spelling.to_ascii_lowercase())
        .unwrap_or_else(|| panic!("unknown GEA1 descriptor dtype {spelling}"))
}

fn descriptor_slot(
    buffer_id: u32,
    name: &str,
    role: DeviceBufferRole,
    lifetime: DeviceBufferLifetime,
    initialization: DeviceBufferInitialization,
    binding: u32,
    element_ty: DeviceDataType,
    element_count: u64,
) -> DescriptorBuffer {
    DescriptorBuffer {
        buffer_id,
        buffer_name: name.to_owned(),
        semantic_value: buffer_id,
        role,
        lifetime,
        initialization,
        binding,
        element_ty,
        element_count,
        version: 1,
    }
}

fn host_descriptor(dtype: DeviceDataType, entry: &str, module_image: &[u8]) -> DeviceDescriptor {
    // This is the first admission check for a selected placement type. A
    // missing BF16 slot must fail here, not be guessed from a two-byte width.
    let placement = dtype
        .placement_discriminant()
        .unwrap_or_else(|| panic!("{entry} has no placement discriminant"));
    assert_eq!(
        DeviceDataType::from_placement_discriminant(placement),
        Some(dtype)
    );

    let kernels = vec![DescriptorKernel {
        entry: entry.to_owned(),
        buffers: vec![
            descriptor_slot(
                1,
                "weights",
                DeviceBufferRole::Input,
                DeviceBufferLifetime::PerProgram,
                DeviceBufferInitialization::HostProvided,
                0,
                dtype,
                WEIGHT_ELEMENTS as u64,
            ),
            descriptor_slot(
                2,
                "input",
                DeviceBufferRole::Input,
                DeviceBufferLifetime::PerProgram,
                DeviceBufferInitialization::HostProvided,
                1,
                DeviceDataType::F32,
                INPUT_ELEMENTS as u64,
            ),
            descriptor_slot(
                3,
                "output",
                DeviceBufferRole::Output,
                DeviceBufferLifetime::ObservationPoint,
                DeviceBufferInitialization::KernelInitialized,
                2,
                DeviceDataType::F32,
                OUTPUT_ELEMENTS as u64,
            ),
        ],
        grid: DISPATCH,
        block: BLOCK,
    }];
    DeviceDescriptor {
        backend: host_coordinator::DeviceBackend::Metal,
        module_image: module_image.to_vec(),
        buffer_versions: kernels[0]
            .buffers
            .iter()
            .map(|slot| DescriptorBufferVersion {
                buffer_id: slot.buffer_id,
                version: slot.version,
                element_ty: slot.element_ty,
                element_count: slot.element_count,
            })
            .collect(),
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

fn fake_bytes(dtype: DeviceDataType, elements: usize) -> Vec<u8> {
    vec![0; elements * dtype.byte_width()]
}

fn artifact_root() -> PathBuf {
    let root = std::env::var_os("GEA1_ARTIFACT_DIR")
        .map(PathBuf::from)
        .expect("GEA1_ARTIFACT_DIR must identify the exported GEA1 bundle");
    assert!(
        root.is_dir(),
        "missing GEA1 artifact directory {}",
        root.display()
    );
    root
}

fn load_bundle(root: &Path) -> Vec<PreparedEntry> {
    let manifest_path = root.join("gea1-artifact-bundle-manifest.json");
    let manifest_bytes = fs::read(&manifest_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", manifest_path.display()));
    let manifest: BundleManifest =
        serde_json::from_slice(&manifest_bytes).expect("valid GEA1 artifact bundle manifest");
    assert_eq!(manifest.schema, "gea1-artifact-bundle-v1");
    assert!(manifest
        .required_members
        .iter()
        .any(|member| member == "gea1-artifact-bundle-manifest.json"));
    for member in &manifest.required_members {
        assert!(
            root.join(member).is_file(),
            "missing GEA1 bundle member {member}"
        );
    }
    assert_eq!(
        manifest.entries.len(),
        2,
        "GEA1 bundle must have two entries"
    );

    let mut prepared = Vec::new();
    for expected_entry in [BF16_ENTRY, F32_ENTRY] {
        let bundle = manifest
            .entries
            .iter()
            .find(|entry| entry.entry == expected_entry)
            .unwrap_or_else(|| panic!("bundle has no exact {expected_entry} identity"))
            .clone();
        let artifact_path = root.join(&bundle.artifact);
        let descriptor_path = root.join(&bundle.descriptor);
        let source_path = root.join(&bundle.source);
        let artifact_bytes = fs::read(&artifact_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", artifact_path.display()));
        let descriptor_bytes = fs::read(&descriptor_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", descriptor_path.display()));
        assert_eq!(sha256_file(&artifact_path), bundle.artifact_sha256);
        assert_eq!(sha256_file(&descriptor_path), bundle.descriptor_sha256);
        assert_eq!(sha256_file(&source_path), bundle.source_sha256);
        let descriptor: ArtifactDescriptor = serde_json::from_slice(&descriptor_bytes)
            .unwrap_or_else(|error| panic!("parse {}: {error}", descriptor_path.display()));
        assert_eq!(descriptor.entry, expected_entry);
        assert_eq!(descriptor.weight_shape, [320, 960]);
        assert_eq!(descriptor.output_shape, [320]);
        assert_eq!(descriptor.dispatch, [320, 1, 1]);
        assert_eq!(
            descriptor.provenance,
            format!("gradus/src/kernel.fab::{expected_entry}")
        );
        prepared.push(PreparedEntry {
            bundle,
            descriptor,
            descriptor_bytes,
            artifact_bytes,
        });
    }
    prepared
}

fn identity_manifest(root: &Path) -> (Value, Vec<f32>, PathBuf, PathBuf) {
    let path = root
        .join("radix/docs/factory/gpu-execution-architecture/evidence/gea1-input-manifest.json");
    let bytes = fs::read(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let identity: Value = serde_json::from_slice(&bytes).expect("valid GEA1 identity manifest");
    assert_eq!(identity["schema"], "gea1-input-manifest-v1");
    assert_eq!(identity["delivery"], "GEA1-U1");
    assert_eq!(identity["source"]["dtype"], "BF16");
    assert_eq!(identity["derived"]["dtype"], "F32");
    assert_eq!(
        identity["logical_identity"]["expanded_bf16_sha256"],
        identity["logical_identity"]["gguf_f32_sha256"]
    );

    let activation_path = root.join("gradus/fixtures/activations/gea1-gemv-input.bin");
    let activation_manifest_path =
        root.join("gradus/fixtures/activations/gea1-gemv-input.manifest.json");
    let activation = fs::read(&activation_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", activation_path.display()));
    let activation_manifest: Value = serde_json::from_slice(
        &fs::read(&activation_manifest_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", activation_manifest_path.display())),
    )
    .expect("valid activation manifest");
    assert_eq!(activation.len(), 3840);
    assert_eq!(sha256_bytes(&activation), identity["activation"]["sha256"]);
    assert_eq!(sha256_bytes(&activation), activation_manifest["sha256"]);
    assert_eq!(activation_manifest["shape"], json!([960]));
    assert_eq!(activation_manifest["dtype"], "F32");
    let activation_values = decode_f32_le(&activation);
    assert_eq!(activation_values.len(), INPUT_ELEMENTS);

    let source_path = PathBuf::from(
        identity["source"]["file"]
            .as_str()
            .expect("source identity path"),
    );
    let derived_path = PathBuf::from(
        identity["derived"]["file"]
            .as_str()
            .expect("derived identity path"),
    );
    assert!(
        source_path.is_file(),
        "missing source identity {}",
        source_path.display()
    );
    assert!(
        derived_path.is_file(),
        "missing derived identity {}",
        derived_path.display()
    );
    assert_eq!(
        sha256_file(&source_path),
        identity["source"]["sha256"].as_str().expect("source hash")
    );
    assert_eq!(
        sha256_file(&derived_path),
        identity["derived"]["sha256"]
            .as_str()
            .expect("derived hash")
    );
    (identity, activation_values, source_path, derived_path)
}

#[test]
fn gea1_descriptor_admission() {
    let bf16 = host_descriptor(DeviceDataType::BF16, BF16_ENTRY, b"fake-bf16-module");
    let f32 = host_descriptor(DeviceDataType::F32, F32_ENTRY, b"fake-f32-module");
    bf16.validate().expect("BF16 descriptor admission");
    f32.validate().expect("F32 descriptor admission");

    assert_eq!(DeviceDataType::BF16.byte_width(), 2);
    assert_eq!(DeviceDataType::F32.byte_width(), 4);
    assert_eq!(bf16.kernels[0].grid, DISPATCH);
    assert_eq!(f32.kernels[0].grid, DISPATCH);
    assert_eq!(
        bf16.kernels[0].buffers[0].element_count,
        WEIGHT_ELEMENTS as u64
    );
    assert_eq!(
        f32.kernels[0].buffers[0].element_count,
        WEIGHT_ELEMENTS as u64
    );
    for descriptor in [&bf16, &f32] {
        assert_eq!(
            descriptor.kernels[0].buffers[1].element_ty,
            DeviceDataType::F32
        );
        assert_eq!(
            descriptor.kernels[0].buffers[1].element_count,
            INPUT_ELEMENTS as u64
        );
        assert_eq!(
            descriptor.kernels[0].buffers[2].element_ty,
            DeviceDataType::F32
        );
        assert_eq!(
            descriptor.kernels[0].buffers[2].element_count,
            OUTPUT_ELEMENTS as u64
        );
    }
}

#[test]
fn gea1_fake_sequence_has_no_cpu_substitute() {
    // These are intentionally structural fake launches. No kernel library or
    // CPU oracle is called; the FakeMetalDriver's internal simulation is not
    // interpreted as numerical evidence.
    let mut session = MetalHostSession::with_driver(Box::new(FakeMetalDriver::default()))
        .expect("fake Metal admission");
    let mut launch_order = Vec::new();
    for (entry, dtype) in [
        (BF16_ENTRY, DeviceDataType::BF16),
        (F32_ENTRY, DeviceDataType::F32),
    ] {
        let descriptor = host_descriptor(dtype, entry, b"fake GEA1 module");
        descriptor.validate().expect("fake descriptor admission");
        let module = session
            .load_module(descriptor.module_image.as_slice())
            .expect("fake module load");
        // FakeMetalDriver's generic three-buffer simulation requires equal
        // spans; dimensional admission is proved separately above.
        let bytes = fake_bytes(DeviceDataType::F32, OUTPUT_ELEMENTS);
        let weight = session.alloc_bytes(bytes.len()).expect("resident weight");
        let input = session.alloc_bytes(bytes.len()).expect("resident input");
        let output = session.alloc_bytes(bytes.len()).expect("resident output");
        session
            .copy_in_bytes(weight, &bytes, dtype)
            .expect("weight upload");
        session.record_weight_upload();
        session
            .copy_in_bytes(input, &bytes, DeviceDataType::F32)
            .expect("input upload");
        session
            .copy_in_bytes(output, &bytes, DeviceDataType::F32)
            .expect("output initialization");
        let bindings = [
            MetalLaunchBinding {
                handle: weight,
                binding_index: 0,
                byte_offset: 0,
                view_span: bytes.len() as u64,
            },
            MetalLaunchBinding {
                handle: input,
                binding_index: 1,
                byte_offset: 0,
                view_span: bytes.len() as u64,
            },
            MetalLaunchBinding {
                handle: output,
                binding_index: 2,
                byte_offset: 0,
                view_span: bytes.len() as u64,
            },
        ];
        session
            .launch_kernel_bound(module, entry, &bindings, DISPATCH, BLOCK)
            .expect("fake GEA1 launch");
        session.sync().expect("fake GEA1 synchronization");
        let output_bytes = session
            .readback_bytes(output, DeviceDataType::F32)
            .expect("fake declared output readback");
        assert_eq!(
            output_bytes.len(),
            OUTPUT_ELEMENTS * DeviceDataType::F32.byte_width()
        );
        launch_order.push(entry);
    }
    assert_eq!(launch_order, [BF16_ENTRY, F32_ENTRY]);
    assert_eq!(
        session.driver_counters().uploads,
        2,
        "one weight upload per dtype"
    );
    assert_eq!(
        session.driver_counters().buffer_allocs,
        6,
        "resident allocations only"
    );
    assert_eq!(session.command_submit_count(), 2);
    assert_eq!(session.blocking_wait_count(), 2);
}

#[test]
fn gea1_fake_failure_modes_fail_closed() {
    let mut session = MetalHostSession::with_driver(Box::new(FakeMetalDriver::default()))
        .expect("fake Metal admission");
    let module = session.load_module(b"fake GEA1 module").expect("module");
    let output = session.alloc_bytes(OUTPUT_ELEMENTS * 4).expect("output");

    let nonresident = session
        .launch_kernel_bound(
            module,
            BF16_ENTRY,
            &[
                MetalLaunchBinding {
                    handle: MetalHandleId(u64::MAX),
                    binding_index: 0,
                    byte_offset: 0,
                    view_span: 4,
                },
                MetalLaunchBinding {
                    handle: output,
                    binding_index: 1,
                    byte_offset: 0,
                    view_span: (OUTPUT_ELEMENTS * 4) as u64,
                },
                MetalLaunchBinding {
                    handle: output,
                    binding_index: 2,
                    byte_offset: 0,
                    view_span: (OUTPUT_ELEMENTS * 4) as u64,
                },
            ],
            DISPATCH,
            BLOCK,
        )
        .expect_err("nonresident weight binding must fail before driver launch");
    assert_eq!(nonresident.code, "E_METAL_INVALID_HANDLE");

    let outside_output = session
        .launch_kernel_bound(
            module,
            F32_ENTRY,
            &[
                MetalLaunchBinding {
                    handle: output,
                    binding_index: 0,
                    byte_offset: 0,
                    view_span: (OUTPUT_ELEMENTS * 4) as u64,
                },
                MetalLaunchBinding {
                    handle: output,
                    binding_index: 1,
                    byte_offset: 0,
                    view_span: (OUTPUT_ELEMENTS * 4) as u64,
                },
                MetalLaunchBinding {
                    handle: output,
                    binding_index: 2,
                    byte_offset: 4,
                    view_span: (OUTPUT_ELEMENTS * 4) as u64,
                },
            ],
            DISPATCH,
            BLOCK,
        )
        .expect_err("readback/output binding outside declared range must fail");
    assert_eq!(outside_output.code, "E_DEVICE_SHAPE_MISMATCH");

    assert_eq!(
        DeviceDataType::from_placement_discriminant(11),
        Some(DeviceDataType::BF16),
        "BF16 placement must be present before fake execution"
    );
    assert_eq!(
        DeviceDataType::from_placement_discriminant(12),
        Some(DeviceDataType::F32),
        "the former F32 slot must remain F32"
    );
}

fn numerical_rows(expected: &[f32], observed: &[f32]) -> Vec<Value> {
    assert_eq!(expected.len(), OUTPUT_ELEMENTS);
    assert_eq!(
        observed, expected,
        "GEA1 GEMV output differs from the scalar oracle"
    );
    (0..8)
        .map(|index| {
            json!({
                "index": index,
                "expected_f32": expected[index],
                "observed_f32": observed[index],
                "absolute_error": (observed[index] - expected[index]).abs(),
            })
        })
        .collect()
}

fn output_receipt(run: &KernelRun, e2e_us: u64) -> Value {
    let f32_output_bytes = f32_bytes(&run.output_f32);
    let physical_output_bytes = run.output_raw.len();
    let effective_bandwidth = if run.gpu_body_us == 0 {
        unmeasured("Metal device timestamp rounded this sub-microsecond body to zero")
    } else {
        let bandwidth = run.weight_bytes as f64 * 1_000_000.0 / run.gpu_body_us as f64;
        derived(
            bandwidth,
            "declared resident weight bytes / measured gpu_body_us * 1_000_000",
        )
    };
    json!({
        "entry": run.prepared.descriptor.entry,
        "artifact": {
            "source": run.prepared.bundle.source,
            "source_sha256": run.prepared.bundle.source_sha256,
            "artifact": run.prepared.bundle.artifact,
            "artifact_sha256": run.prepared.bundle.artifact_sha256,
            "descriptor": run.prepared.bundle.descriptor,
            "descriptor_sha256": run.prepared.bundle.descriptor_sha256,
        },
        "dtype": run.physical_dtype,
        "launch": {
            "dispatch": DISPATCH,
            "block": BLOCK,
            "binding_order": [0, 1, 2],
        },
        "output_range": {
            "offset_elements": 0,
            "elements": OUTPUT_ELEMENTS,
            "logical_dtype": "F32",
            "logical_bytes": OUTPUT_ELEMENTS * 4,
            "physical_dtype": "F32",
            "physical_bytes": physical_output_bytes,
        },
        "output_hashes": {
            "physical_sha256": sha256_bytes(&run.output_raw),
            "f32_sha256": sha256_bytes(&f32_output_bytes),
        },
        "numerical_rows": run.numerical_rows,
        "measurements": {
            "admission_us": measured(run.admission_us),
            "admission_bytes": measured(run.prepared.descriptor_bytes.len()),
            "tensor_range_read_us": measured(run.tensor_range_read_us),
            "tensor_range_read_bytes": measured(run.weight_bytes),
            "bf16_to_f32_us": measured(run.bf16_to_f32_us),
            "bf16_to_f32_bytes": measured(0),
            "allocation_us": measured(run.allocation_us),
            "upload_us": measured(run.upload_us),
            "upload_bytes": measured(run.weight_bytes),
            "pipeline_create_us": measured(run.pipeline_create_us),
            "binding_us": measured(run.binding_us),
            "warmups": measured(0),
            "repetitions": measured(1),
            "gpu_body_us": measured(run.gpu_body_us),
            "launch_submit_orchestration_us": measured(run.launch_submit_orchestration_us),
            "sync_wait_us": measured(run.sync_wait_us),
            "readback_us": measured(run.readback_us),
            "readback_bytes": measured(physical_output_bytes),
            "effective_bandwidth_bytes_per_s": effective_bandwidth,
            "e2e_us": measured(e2e_us),
        },
    })
}

#[test]
#[ignore = "physical Metal gate; run only with the exact §6 command"]
fn gea1_real_metal_receipt() {
    let e2e_start = Instant::now();
    std::env::set_var("FABER_PER_OP_TIMING", "1");
    let workspace = workspace_root();
    let artifact_dir = artifact_root();
    let receipt_path = PathBuf::from(
        std::env::var_os("GEA1_METAL_RECEIPT")
            .expect("GEA1_METAL_RECEIPT must identify the receipt output"),
    );
    let prepared = load_bundle(&artifact_dir);
    let (identity, activation_values, source_path, derived_path) = identity_manifest(&workspace);
    let source_revision = identity["source"]["revision"]
        .as_str()
        .expect("source revision");
    let gradus_revision = git_revision(&workspace.join("gradus"));
    let radix_revision = git_revision(&workspace.join("radix"));
    let hosts_revision = git_revision(&workspace.join("hosts"));
    let devices = enumerate_metal_physical_devices().expect("Metal device enumeration");
    assert!(
        !devices.is_empty(),
        "Metal selected but no physical device identity exists"
    );
    let device = &devices[0];
    assert!(
        !device.registry_id.is_empty(),
        "Metal registry identity is required"
    );
    assert!(
        device.api_total_bytes > 0,
        "Metal memory capability is required"
    );

    // This is deliberately an expect, never an ignored/skip branch when
    // Metal is selected by the physical gate.
    let mut session =
        MetalHostSession::try_open().expect("Metal selected but session admission failed");
    let mut runs = Vec::new();
    for prepared_entry in prepared {
        let descriptor_start = Instant::now();
        let dtype = descriptor_dtype(&prepared_entry.descriptor.weight_dtype);
        let descriptor = host_descriptor(
            dtype,
            &prepared_entry.descriptor.entry,
            &prepared_entry.artifact_bytes,
        );
        descriptor
            .validate()
            .expect("GEA1 host descriptor admission");
        let admission_us = descriptor_start.elapsed().as_micros() as u64;
        let weight_range = prepared_entry.descriptor.weight_range;
        let weight_path = if dtype == DeviceDataType::BF16 {
            &source_path
        } else {
            &derived_path
        };
        let range_start = Instant::now();
        let weight_bytes = read_range(weight_path, weight_range);
        let tensor_range_read_us = range_start.elapsed().as_micros() as u64;
        let expected_weight_bytes = usize::try_from(weight_range[1] - weight_range[0])
            .expect("weight range fits host usize");
        assert_eq!(weight_bytes.len(), expected_weight_bytes);
        assert_eq!(weight_bytes.len(), WEIGHT_ELEMENTS * dtype.byte_width());

        let input_bytes = f32_bytes(&activation_values);
        let output_bytes = vec![0; OUTPUT_ELEMENTS * DeviceDataType::F32.byte_width()];
        let allocation_start = Instant::now();
        let weight_handle = session
            .alloc_bytes(weight_bytes.len())
            .expect("resident weight allocation");
        let input_handle = session
            .alloc_bytes(input_bytes.len())
            .expect("resident input allocation");
        let output_handle = session
            .alloc_bytes(output_bytes.len())
            .expect("resident output allocation");
        let allocation_us = allocation_start.elapsed().as_micros() as u64;

        let upload_start = Instant::now();
        session
            .copy_in_bytes(weight_handle, &weight_bytes, dtype)
            .expect("upload selected resident weight range");
        session.record_weight_upload();
        let upload_us = upload_start.elapsed().as_micros() as u64;
        session
            .copy_in_bytes(input_handle, &input_bytes, DeviceDataType::F32)
            .expect("upload activation input");
        session
            .copy_in_bytes(output_handle, &output_bytes, DeviceDataType::F32)
            .expect("initialize output allocation");

        let pipeline_start = Instant::now();
        let module_handle = session
            .load_module(&prepared_entry.artifact_bytes)
            .expect("compile and create Metal pipeline");
        let pipeline_create_us = pipeline_start.elapsed().as_micros() as u64;

        let binding_start = Instant::now();
        let bindings = [
            MetalLaunchBinding {
                handle: weight_handle,
                binding_index: 0,
                byte_offset: 0,
                view_span: weight_bytes.len() as u64,
            },
            MetalLaunchBinding {
                handle: input_handle,
                binding_index: 1,
                byte_offset: 0,
                view_span: input_bytes.len() as u64,
            },
            MetalLaunchBinding {
                handle: output_handle,
                binding_index: 2,
                byte_offset: 0,
                view_span: output_bytes.len() as u64,
            },
        ];
        let binding_us = binding_start.elapsed().as_micros() as u64;
        let launch_start = Instant::now();
        session
            .launch_kernel_bound(
                module_handle,
                &prepared_entry.descriptor.entry,
                &bindings,
                DISPATCH,
                BLOCK,
            )
            .expect("bind and launch GEA1 Metal entry");
        let launch_submit_orchestration_us = launch_start.elapsed().as_micros() as u64;
        let sync_start = Instant::now();
        session.sync().expect("synchronize GEA1 Metal entry");
        let sync_wait_us = sync_start.elapsed().as_micros() as u64;
        let gpu_times = session.take_encoder_gpu_us();
        assert_eq!(gpu_times.len(), 1, "per-entry GPU timestamp is required");
        let gpu_body_us = gpu_times[0];

        let readback_start = Instant::now();
        let output_raw = session
            .readback_bytes(output_handle, DeviceDataType::F32)
            .expect("read only the declared 320-element output allocation");
        let readback_us = readback_start.elapsed().as_micros() as u64;
        assert_eq!(output_raw.len(), output_bytes.len());
        let output_f32 = decode_f32_le(&output_raw);
        let bf16_to_f32_us = 0;
        let weight_values = if dtype == DeviceDataType::BF16 {
            bf16_to_f32(&weight_bytes)
        } else {
            decode_f32_le(&weight_bytes)
        };
        let expected_f32 = reference_gemv(&weight_values, &activation_values);
        let numerical_rows = numerical_rows(&expected_f32, &output_f32);
        runs.push(KernelRun {
            prepared: prepared_entry,
            dtype,
            physical_dtype: dtype.spelling().to_ascii_uppercase(),
            weight_handle,
            input_handle,
            output_handle,
            weight_bytes: weight_bytes.len(),
            input_bytes: input_bytes.len(),
            output_bytes: output_bytes.len(),
            output_raw,
            output_f32,
            admission_us,
            allocation_us,
            upload_us,
            pipeline_create_us,
            binding_us,
            launch_submit_orchestration_us,
            sync_wait_us,
            gpu_body_us,
            readback_us,
            tensor_range_read_us,
            bf16_to_f32_us,
            numerical_rows,
        });
    }
    assert_eq!(runs.len(), 2);
    assert_eq!(runs[0].prepared.descriptor.entry, BF16_ENTRY);
    assert_eq!(runs[1].prepared.descriptor.entry, F32_ENTRY);
    assert_eq!(runs[0].dtype, DeviceDataType::BF16);
    assert_eq!(runs[1].dtype, DeviceDataType::F32);
    assert_eq!(
        runs[0].input_bytes,
        INPUT_ELEMENTS * DeviceDataType::F32.byte_width()
    );
    assert_eq!(
        runs[1].input_bytes,
        INPUT_ELEMENTS * DeviceDataType::F32.byte_width()
    );
    assert_eq!(
        runs[0].output_bytes,
        OUTPUT_ELEMENTS * DeviceDataType::F32.byte_width()
    );
    assert_eq!(
        runs[1].output_bytes,
        OUTPUT_ELEMENTS * DeviceDataType::F32.byte_width()
    );
    assert_eq!(session.driver_counters().uploads, 2);

    let allocations: Vec<Value> = runs
        .iter()
        .map(|run| {
            json!({
                "dtype": run.physical_dtype,
                "weight": {
                    "handle": run.weight_handle.0,
                    "capacity_bytes": run.weight_bytes,
                    "lifetime": "per-program",
                    "initialization": "host-provided",
                    "upload_count": 1,
                },
                "input": {
                    "handle": run.input_handle.0,
                    "capacity_bytes": run.input_bytes,
                    "lifetime": "per-program",
                    "initialization": "host-provided",
                    "upload_count": 1,
                },
                "output": {
                    "handle": run.output_handle.0,
                    "capacity_bytes": run.output_bytes,
                    "lifetime": "observation-point",
                    "initialization": "kernel-initialized",
                    "upload_count": 0,
                },
            })
        })
        .collect();
    let launch_order: Vec<Value> = runs
        .iter()
        .enumerate()
        .map(|(order, run)| {
            json!({
                "order": order,
                "entry": run.prepared.descriptor.entry,
                "artifact_sha256": run.prepared.bundle.artifact_sha256,
                "dispatch": DISPATCH,
                "block": BLOCK,
                "bindings": [
                    {"index": 0, "role": "weight", "handle": run.weight_handle.0},
                    {"index": 1, "role": "input", "handle": run.input_handle.0},
                    {"index": 2, "role": "output", "handle": run.output_handle.0},
                ],
            })
        })
        .collect();
    let e2e_us = e2e_start.elapsed().as_micros() as u64;
    let measurement_entries: BTreeMap<String, Value> = runs
        .iter()
        .map(|run| {
            (
                run.prepared.descriptor.entry.clone(),
                output_receipt(run, e2e_us),
            )
        })
        .collect();
    let receipt = json!({
        "schema": "gea1-metal-receipt-v1",
        "delivery": "GEA1-U4",
        "backend": "Metal",
        "machine": Command::new("hostname").output().ok().map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned()),
        "identities": {
            "entries": runs.iter().map(|run| json!({
                "entry": run.prepared.descriptor.entry,
                "artifact_sha256": run.prepared.bundle.artifact_sha256,
                "descriptor_sha256": run.prepared.bundle.descriptor_sha256,
                "source_sha256": run.prepared.bundle.source_sha256,
            })).collect::<Vec<_>>(),
            "activation_sha256": identity["activation"]["sha256"],
            "source_model_sha256": identity["source"]["sha256"],
            "derived_f32_model_sha256": identity["derived"]["sha256"],
            "logical_tensor_sha256": identity["logical_identity"]["expanded_bf16_sha256"],
        },
        "revisions": {
            "source_model": source_revision,
            "gradus": gradus_revision,
            "radix": radix_revision,
            "hosts": hosts_revision,
            "converter": identity["converter"]["revision"],
        },
        "physical_device": {
            "backend": "Metal",
            "registry_id": device.registry_id,
            "model": device.device_model,
            "api_total_bytes": device.api_total_bytes,
            "max_threads_per_workgroup": device.max_threads_per_workgroup,
            "workgroup_shared_memory_min_bytes": device.workgroup_shared_memory_min_bytes,
            "workgroup_shared_memory_max_bytes": device.workgroup_shared_memory_max_bytes,
            "collective_width": device.collective_width,
            "unified_memory": device.unified_memory,
        },
        "allocations": allocations,
        "launch_order": launch_order,
        "zero_cpu_attestation": {
            "cpu_substitute_count": 0,
            "cpu_bridge_count": 0,
            "cpu_substitutes": [],
            "cpu_bridges": [],
            "execution_session": "MetalHostSession::try_open",
            "fake_driver_used": false,
            "weights_uploaded_once_per_dtype": true,
        },
        "declared_output": {
            "elements": OUTPUT_ELEMENTS,
            "logical_dtype": "F32",
            "logical_bytes": OUTPUT_ELEMENTS * 4,
            "readback_scope": "only declared 320-element output per entry",
        },
        "outputs": runs
            .iter()
            .map(|run| output_receipt(run, e2e_us))
            .collect::<Vec<_>>(),
        "measurement_fields": MEASUREMENT_FIELDS,
        "measurements": measurement_entries,
        "e2e_us": measured(e2e_us),
    });
    let parent = receipt_path.parent().expect("receipt parent");
    fs::create_dir_all(parent).expect("create receipt parent");
    fs::write(
        &receipt_path,
        serde_json::to_vec_pretty(&receipt).expect("serialize receipt"),
    )
    .unwrap_or_else(|error| panic!("write {}: {error}", receipt_path.display()));
    eprintln!("GEA1 real Metal receipt: {}", receipt_path.display());
}
