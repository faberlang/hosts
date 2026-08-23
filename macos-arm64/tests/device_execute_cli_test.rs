//! CLI + wire tests for `device-execute`.

use std::collections::BTreeMap;
use std::fs;
use std::process::Command;

use faber_host_macos_arm64::composite_host::{
    CompositeHost, DeviceByteBuffer, InferenceSessionState,
};
use faber_host_macos_arm64::device_descriptor::{
    DescriptorBuffer, DescriptorBufferVersion, DescriptorKernel, DescriptorLaunch,
    DescriptorResult, DeviceBufferInitialization, DeviceBufferLifetime, DeviceBufferRole,
    DeviceDataType, DeviceDescriptor, DeviceProgramLifetime, E_DEVICE_DESCRIPTOR,
};
use faber_host_macos_arm64::device_execute::{
    admit_v2_load, descriptor_from_json, descriptor_to_json, gguf_region_table, inputs_from_gguf,
    inputs_from_json, inputs_from_mapped_gguf, inputs_to_json, parse_control_request,
    parse_device_execute_args, receipt_to_json, weight_map_from_json, weight_map_to_json,
    DeviceExecuteControlVerb, DeviceExecuteInvocationMode, DeviceExecuteProtocol,
    DeviceExecuteReceipt, WeightFileRange,
};
use faber_host_macos_arm64::device_host::DeviceRuntime;
use faber_host_macos_arm64::metal_host::MappedWeightFile;
use faber_host_macos_arm64::{CudaHostSession, FakeCudaDriver, HostError};
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

fn packed_weight_descriptor(backend: DeviceBackend) -> DeviceDescriptor {
    let mut descriptor = elementwise_add_descriptor();
    descriptor.backend = backend;
    for kernel in &mut descriptor.kernels {
        for buffer in &mut kernel.buffers {
            buffer.element_count = 9;
        }
    }
    for version in &mut descriptor.buffer_versions {
        version.element_count = 9;
    }
    descriptor
}

fn fake_runtime(backend: DeviceBackend) -> DeviceRuntime {
    match backend {
        DeviceBackend::Metal => DeviceRuntime::Metal(
            MetalHostSession::with_driver(Box::new(
                FakeMetalDriver::default().with_known_entry("addita"),
            ))
            .expect("fake Metal admit"),
        ),
        DeviceBackend::Cuda => DeviceRuntime::Cuda(
            CudaHostSession::with_driver(Box::new(
                FakeCudaDriver::default().with_known_entry("addita"),
            ))
            .expect("fake CUDA admit"),
        ),
    }
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
fn parse_device_execute_args_requires_weights_and_map_together() {
    let args = [
        "--descriptor".to_owned(),
        "d.json".to_owned(),
        "--module".to_owned(),
        "m.bin".to_owned(),
        "--inputs".to_owned(),
        "i.json".to_owned(),
        "--weights".to_owned(),
        "model.gguf".to_owned(),
    ];
    let err = parse_device_execute_args(&args).expect_err("weights without map");
    assert!(err.contains("--weight-map"), "{err}");
}

#[test]
fn parse_device_execute_args_accepts_weights_and_map() {
    let args = [
        "--descriptor".to_owned(),
        "d.json".to_owned(),
        "--module".to_owned(),
        "m.bin".to_owned(),
        "--inputs".to_owned(),
        "i.json".to_owned(),
        "--weights".to_owned(),
        "model.gguf".to_owned(),
        "--weight-map".to_owned(),
        "map.json".to_owned(),
    ];
    let parsed = parse_device_execute_args(&args).expect("valid flags");
    assert_eq!(
        parsed.weights.as_deref(),
        Some(std::path::Path::new("model.gguf"))
    );
    assert_eq!(
        parsed.weight_map.as_deref(),
        Some(std::path::Path::new("map.json"))
    );
}

#[test]
fn gguf_weight_map_admits_raw_packed_region() {
    let mut file = Vec::new();
    file.extend_from_slice(&[0xaa, 0xbb, 0xcc, 0xdd]);
    file.extend_from_slice(&1.0f32.to_le_bytes());
    file.extend_from_slice(&2.0f32.to_le_bytes());
    file.extend_from_slice(&u32::to_le_bytes(0xff81_0000));
    let map = BTreeMap::from([(
        7,
        WeightFileRange {
            offset: 4,
            len: 12,
            elems: 3,
        },
    )]);
    let json = weight_map_to_json(&map).expect("encode");
    let decoded = weight_map_from_json(&json).expect("decode");
    assert_eq!(decoded, map);
    let inputs = inputs_from_gguf(&file, &map).expect("fill");
    let values = &inputs.byte_map()[&7];
    assert_eq!(values.dtype, DeviceDataType::U8);
    assert_eq!(values.bytes, file[4..16]);
}

#[test]
fn gguf_weight_map_rejects_logical_f32_padding() {
    let file = vec![0u8; 12];
    let map = BTreeMap::from([(
        7,
        WeightFileRange {
            offset: 0,
            len: 12,
            elems: 4,
        },
    )]);
    let err = inputs_from_gguf(&file, &map).expect_err("logical pad is not admitted");
    assert!(
        err.message.contains("native packed width"),
        "{}",
        err.message
    );
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
    assert!(!parsed.control);
}

#[test]
fn parse_device_execute_args_accepts_control_owner_flag() {
    let args = [
        "--control".to_owned(),
        "--descriptor".to_owned(),
        "d.json".to_owned(),
        "--module".to_owned(),
        "m.bin".to_owned(),
        "--inputs".to_owned(),
        "i.json".to_owned(),
    ];
    let parsed = parse_device_execute_args(&args).expect("valid control flags");
    assert!(parsed.control);
}

#[test]
fn parse_device_execute_args_accepts_distributed_image_and_bind_count() {
    let args = [
        "--backend".to_owned(),
        "cuda".to_owned(),
        "--distributed-image".to_owned(),
        "eight-rank.postcard".to_owned(),
        "--bind-count".to_owned(),
        "1".to_owned(),
    ];
    let parsed = parse_device_execute_args(&args).expect("distributed flags");
    assert_eq!(
        parsed.distributed_image.as_deref(),
        Some(std::path::Path::new("eight-rank.postcard"))
    );
    assert_eq!(parsed.bind_count, Some(1));
    assert_eq!(
        parsed.backend,
        Some(faber_host_macos_arm64::composite_host::DeviceSelection::Cuda)
    );
}

#[test]
fn parse_device_execute_args_requires_distributed_image_and_bind_count_together() {
    let image_only = [
        "--distributed-image".to_owned(),
        "eight-rank.postcard".to_owned(),
    ];
    let err = parse_device_execute_args(&image_only).expect_err("image without bind-count");
    assert!(err.contains("--bind-count"), "{err}");

    let count_only = ["--bind-count".to_owned(), "8".to_owned()];
    let err = parse_device_execute_args(&count_only).expect_err("bind-count without image");
    assert!(err.contains("--distributed-image"), "{err}");
}

#[test]
fn parse_device_execute_args_rejects_distributed_with_control() {
    let args = [
        "--control".to_owned(),
        "--distributed-image".to_owned(),
        "eight-rank.postcard".to_owned(),
        "--bind-count".to_owned(),
        "1".to_owned(),
    ];
    let err = parse_device_execute_args(&args).expect_err("control + distributed");
    assert!(err.contains("one-shot prepare"), "{err}");
}

#[test]
fn cli_device_execute_usage_names_distributed_image() {
    let output = Command::new(env!("CARGO_BIN_EXE_faber-host-macos-arm64"))
        .args(["device-execute"])
        .output()
        .expect("run device-execute");
    assert_eq!(output.status.code(), Some(64));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--distributed-image"), "{stderr}");
    assert!(stderr.contains("--bind-count"), "{stderr}");
}

#[test]
fn cli_distributed_image_missing_file_exits_2() {
    let dir = tempfile::tempdir().expect("tempdir");
    let missing = dir.path().join("missing.postcard");
    let output = Command::new(env!("CARGO_BIN_EXE_faber-host-macos-arm64"))
        .args([
            "device-execute",
            "--backend",
            "metal",
            "--distributed-image",
            missing.to_str().expect("utf8"),
            "--bind-count",
            "1",
        ])
        .output()
        .expect("run device-execute");
    assert_eq!(output.status.code(), Some(2));
    let json: Value = serde_json::from_slice(&output.stdout).expect("error json");
    assert_eq!(json["code"], "E_INVALID_ARGS");
}

#[test]
fn control_protocol_accepts_lifecycle_verbs_and_lossless_inputs() {
    let load = parse_control_request(br#"{"op":"load"}"#).expect("load");
    assert_eq!(load.verb, DeviceExecuteControlVerb::Load);
    let step =
        parse_control_request(br#"{"op":"step","inputs":{"2":["0xff810000"]}}"#).expect("step");
    assert_eq!(step.verb, DeviceExecuteControlVerb::Step);
    assert_eq!(step.inputs.expect("inputs")[&2][0].to_bits(), 0xff81_0000);
    for (raw, expected) in [
        (
            br#"{"op":"reset"}"#.as_slice(),
            DeviceExecuteControlVerb::Reset,
        ),
        (
            br#"{"op":"release"}"#.as_slice(),
            DeviceExecuteControlVerb::Release,
        ),
    ] {
        assert_eq!(parse_control_request(raw).expect("verb").verb, expected);
    }
}

#[test]
fn control_protocol_rejects_unknown_verb_with_structured_error() {
    let error = parse_control_request(br#"{"op":"compile"}"#).expect_err("unknown verb");
    assert_eq!(error.code, "E_INVALID_ARGS");
    assert!(
        error.message.contains("load, step, reset, release"),
        "{error}"
    );
}

#[test]
fn protocol_v2_load_names_both_programs_and_requires_the_pair() {
    let args = [
        "--control".to_owned(),
        "--protocol".to_owned(),
        "v2".to_owned(),
        "--prefill-descriptor".to_owned(),
        "prefill.json".to_owned(),
        "--prefill-module".to_owned(),
        "prefill.msl".to_owned(),
        "--decode-descriptor".to_owned(),
        "decode.json".to_owned(),
        "--decode-module".to_owned(),
        "decode.msl".to_owned(),
        "--model-identity".to_owned(),
        "smollm2".to_owned(),
        "--session-identity".to_owned(),
        "prompt-7".to_owned(),
        "--inputs".to_owned(),
        "prefill-inputs.json".to_owned(),
    ];
    let parsed = parse_device_execute_args(&args).expect("v2 pair");
    assert_eq!(parsed.protocol, DeviceExecuteProtocol::V2);
    assert_eq!(
        parsed.prefill_descriptor.as_deref(),
        Some(std::path::Path::new("prefill.json"))
    );
    assert_eq!(
        parsed.decode_module.as_deref(),
        Some(std::path::Path::new("decode.msl"))
    );
    assert_eq!(parsed.model_identity.as_deref(), Some("smollm2"));
    assert_eq!(parsed.session_identity.as_deref(), Some("prompt-7"));

    let missing = [
        "--protocol".to_owned(),
        "v2".to_owned(),
        "--prefill-descriptor".to_owned(),
        "prefill.json".to_owned(),
        "--prefill-module".to_owned(),
        "prefill.msl".to_owned(),
        "--inputs".to_owned(),
        "inputs.json".to_owned(),
    ];
    let error = parse_device_execute_args(&missing).expect_err("decode pair is mandatory");
    assert!(error.contains("decode-descriptor"), "{error}");

    let legacy_with_pair = [
        "--protocol".to_owned(),
        "v1".to_owned(),
        "--descriptor".to_owned(),
        "legacy.json".to_owned(),
        "--module".to_owned(),
        "legacy.msl".to_owned(),
        "--inputs".to_owned(),
        "inputs.json".to_owned(),
        "--decode-descriptor".to_owned(),
        "decode.json".to_owned(),
    ];
    let error = parse_device_execute_args(&legacy_with_pair)
        .expect_err("v1 cannot carry a second KV program");
    assert!(error.contains("protocol v1"), "{error}");
}

#[test]
fn protocol_v2_decode_carries_token_and_cursor_but_v1_rejects_it() {
    let request = parse_control_request(
        br#"{"protocol":2,"op":"invoke","mode":"scalar_decode","token":42,"position":17,"sequence_epoch":3,"prefix_before":17,"valid_len_after":18,"query_start":17}"#,
    )
    .expect("v2 invoke");
    assert_eq!(request.protocol, DeviceExecuteProtocol::V2);
    assert_eq!(request.verb, DeviceExecuteControlVerb::Step);
    let invocation = request.invocation.expect("invocation");
    assert_eq!(invocation.mode, DeviceExecuteInvocationMode::ScalarDecode);
    assert_eq!(invocation.token, Some(42));
    assert_eq!(invocation.position, 17);
    assert!(request.inputs.is_none(), "decode has no legacy inputs map");

    let error = parse_control_request(
        br#"{"protocol":1,"op":"invoke","mode":"scalar_decode","token":42,"position":17,"sequence_epoch":3,"prefix_before":17,"valid_len_after":18,"query_start":17}"#,
    )
    .expect_err("v1 cannot carry KV execution");
    assert_eq!(error.code, "E_INVALID_ARGS");
    assert!(error.message.contains("protocol v1"), "{error}");

    let error = parse_control_request(
        br#"{"protocol":2,"op":"invoke","mode":"scalar_decode","token":42,"position":17,"sequence_epoch":3,"prefix_before":17,"valid_len_after":18,"query_start":17,"inputs":{"1":[42]}}"#,
    )
    .expect_err("decode must not carry a legacy input map");
    assert!(error.message.contains("token/position only"), "{error}");
}

#[test]
fn protocol_v2_admission_exposes_both_program_identities_and_one_prepare() {
    let prefill = elementwise_add_descriptor();
    let decode = elementwise_add_descriptor();
    let receipt = admit_v2_load(&prefill, &decode, "smollm2", "prompt-7").expect("admit pair");
    assert_eq!(receipt.protocol, 2);
    assert_eq!(receipt.lifecycle.prepares, 1);
    assert_eq!(receipt.lifecycle.reuses, 0);
    assert_eq!(receipt.model_identity.as_deref(), Some("smollm2"));
    assert_eq!(receipt.session_identity.as_deref(), Some("prompt-7"));
    assert_eq!(receipt.program_identities.len(), 2);
    assert!(receipt.program_identities.contains_key("prefill"));
    assert!(receipt.program_identities.contains_key("scalar_decode"));
}

#[test]
fn d1_state_machine_is_public_at_the_composite_host_boundary() {
    let state = InferenceSessionState::new(8).expect("state");
    assert_eq!(state.sequence_epoch(), 1);
    assert_eq!(state.valid_len(), 0);
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
    assert_eq!(decoded.buffer_versions, original.buffer_versions);
    assert_eq!(decoded.data_flow, original.data_flow);
    assert_eq!(decoded.roots, original.roots);
    assert_eq!(decoded.results, original.results);
    assert_eq!(decoded.end_of_run_results, original.end_of_run_results);
    assert_eq!(decoded.program_lifetime, DeviceProgramLifetime::SingleRun);
}

/// Spawn-shaped descriptor (radix `WireDescriptor` field set) decodes
/// bit-identical on the structural fields — no f32 payloads on this file.
#[test]
fn spawn_shaped_descriptor_json_is_structurally_identical() {
    let json = br#"{
      "backend": "metal",
      "kernels": [{
        "entry": "addita",
        "buffers": [
          {"buffer_id":1,"buffer_name":"a","semantic_value":1,"role":"input","lifetime":"per-program","initialization":"host-provided","binding":0,"element_ty":"f32","element_count":2,"version":1},
          {"buffer_id":2,"buffer_name":"b","semantic_value":2,"role":"input","lifetime":"per-program","initialization":"host-provided","binding":1,"element_ty":"f32","element_count":2,"version":1},
          {"buffer_id":3,"buffer_name":"out","semantic_value":3,"role":"output","lifetime":"observation-point","initialization":"kernel-initialized","binding":2,"element_ty":"f32","element_count":2,"version":1}
        ],
        "grid": [1,1,1],
        "block": [2,1,1]
      }],
      "launches": [{"id":1,"kernel_index":0}],
      "buffer_versions": [
        {"buffer_id":1,"version":1,"element_ty":"f32","element_count":2},
        {"buffer_id":2,"version":1,"element_ty":"f32","element_count":2},
        {"buffer_id":3,"version":1,"element_ty":"f32","element_count":2}
      ],
      "program_lifetime": "single-run",
      "data_flow": [],
      "roots": [1],
      "results": [{"buffer_id":3,"version":1,"produced_by":1,"at_launch":1}],
      "end_of_run_results": []
    }"#;
    let decoded = descriptor_from_json(json, MODULE_IMAGE.to_vec()).expect("decode");
    let reencoded = descriptor_to_json(&decoded).expect("encode");
    let again = descriptor_from_json(&reencoded, MODULE_IMAGE.to_vec()).expect("redecode");
    assert_eq!(decoded.kernels, again.kernels);
    assert_eq!(decoded.launches, again.launches);
    assert_eq!(decoded.buffer_versions, again.buffer_versions);
    assert_eq!(decoded.roots, again.roots);
    assert_eq!(decoded.results, again.results);
}

#[test]
fn inputs_json_round_trips_buffer_ids() {
    let inputs = add_inputs();
    let json = inputs_to_json(&inputs).expect("encode");
    let decoded = inputs_from_json(&json).expect("decode");
    assert_eq!(decoded.map(), &inputs);
    assert!(decoded.byte_map().is_empty());
}

#[test]
fn inputs_json_round_trips_non_finite_values() {
    let inputs = BTreeMap::from([(1, vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.5])]);
    let json = inputs_to_json(&inputs).expect("encode");
    let text = String::from_utf8(json.clone()).expect("utf8");
    assert!(
        text.contains(&format!("\"0x{:08x}\"", f32::NAN.to_bits())),
        "{text}"
    );
    assert!(!text.contains("null"), "{text}");
    let decoded = inputs_from_json(&json).expect("decode");
    let values = decoded.map().get(&1).expect("buffer 1");
    assert_eq!(values[0].to_bits(), f32::NAN.to_bits());
    assert_eq!(values[1], f32::INFINITY);
    assert_eq!(values[2], f32::NEG_INFINITY);
    assert_eq!(values[3], 1.5);
}

/// First spawn-wire divergence on SmolLM2: packed GGUF word `0xff810000`
/// (file offset 1_774_948) was encoded as `"NaN"` and came back as
/// canonical `0x7fc00000`. Packed weights ride `f32` slots as raw bits.
#[test]
fn inputs_json_preserves_smollm2_first_nan_payload() {
    const FIRST: u32 = 0xff81_0000;
    let inputs = BTreeMap::from([(1, vec![f32::from_bits(FIRST)])]);
    let json = inputs_to_json(&inputs).expect("encode");
    let text = String::from_utf8(json.clone()).expect("utf8");
    assert!(text.contains("\"0xff810000\""), "{text}");
    assert!(!text.contains("\"NaN\""), "{text}");
    let decoded = inputs_from_json(&json).expect("decode");
    assert_eq!(decoded.map()[&1][0].to_bits(), FIRST);
}

#[test]
fn inputs_json_round_trips_smollm2_token_bit_patterns() {
    let tokens: Vec<f32> = [504u32, 2365, 6354, 16438, 27003, 690, 260, 23790, 2767]
        .into_iter()
        .map(f32::from_bits)
        .collect();
    let inputs = BTreeMap::from([(1, tokens.clone())]);
    let json = inputs_to_json(&inputs).expect("encode");
    let decoded = inputs_from_json(&json).expect("decode");
    let got = decoded.map().get(&1).expect("buffer 1");
    assert_eq!(got.len(), tokens.len());
    for (index, (observed, expected)) in got.iter().zip(&tokens).enumerate() {
        assert_eq!(
            observed.to_bits(),
            expected.to_bits(),
            "token[{index}] bits diverged"
        );
    }
}

#[test]
fn inputs_json_accepts_legacy_nan_string() {
    let decoded = inputs_from_json(br#"{"1":["NaN"]}"#).expect("decode");
    assert!(decoded.map()[&1][0].is_nan());
}

fn tagged_input_json(id: u32, dtype: &str, bytes: &[u8]) -> Vec<u8> {
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!(r#"{{"{id}":{{"dtype":"{dtype}","bytes":"{hex}"}}}}"#).into_bytes()
}

fn assert_misaligned_tail(err: &HostError, dtype: &str, len: usize) {
    assert_eq!(err.code, "E_INVALID_ARGS");
    assert!(
        err.message.contains("misaligned")
            && err.message.contains(dtype)
            && err.message.contains(&len.to_string()),
        "misaligned-tail error must name dtype and length, got {}",
        err.message
    );
}

/// Tagged `--inputs` stay raw bytes. 34-byte f16/bf16 payloads are aligned
/// (width 2); 32-byte f32 is aligned (width 4). Untagged hex still means
/// one f32 word of bits — that legacy array form is back-compat, not a
/// byte blob.
#[test]
fn inputs_json_accepts_dtype_tagged_aligned_bytes() {
    let payload_34: Vec<u8> = (0..34).collect();
    for dtype in [(DeviceDataType::F16, "f16"), (DeviceDataType::BF16, "bf16")] {
        let decoded = inputs_from_json(&tagged_input_json(7, dtype.1, &payload_34))
            .expect("aligned tagged payload");
        let got = &decoded.byte_map()[&7];
        assert_eq!(got.dtype, dtype.0);
        assert_eq!(got.bytes, payload_34);
        assert!(decoded.map().is_empty());
    }

    let payload_32: Vec<u8> = (0..32).collect();
    let decoded =
        inputs_from_json(&tagged_input_json(3, "f32", &payload_32)).expect("aligned f32 payload");
    let got = &decoded.byte_map()[&3];
    assert_eq!(got.dtype, DeviceDataType::F32);
    assert_eq!(got.bytes, payload_32);

    let mixed = br#"{"1":[1.0,2.0],"2":{"dtype":"f16","bytes":"0x0011"}}"#;
    let decoded = inputs_from_json(mixed).expect("mixed array and tagged bytes");
    assert_eq!(decoded.map()[&1], vec![1.0, 2.0]);
    assert_eq!(decoded.byte_map()[&2].dtype, DeviceDataType::F16);
    assert_eq!(decoded.byte_map()[&2].bytes, vec![0x00, 0x11]);
}

/// DSB-2 named rule: payload length must be a multiple of the tag width.
/// 34-byte f32 is the CUDA first-failing oracle (`len % 4 == 2`); 33-byte
/// tails are odd and miss every f32/f16/bf16 width.
#[test]
fn inputs_json_rejects_misaligned_tagged_bytes_by_name() {
    let payload_34: Vec<u8> = (0..34).collect();
    let err = inputs_from_json(&tagged_input_json(7, "f32", &payload_34))
        .expect_err("34-byte f32 tail is misaligned");
    assert_misaligned_tail(&err, "f32", 34);

    let payload_33: Vec<u8> = (0..33).collect();
    for dtype in ["f32", "f16", "bf16"] {
        let err = inputs_from_json(&tagged_input_json(7, dtype, &payload_33))
            .expect_err("33-byte tail is misaligned");
        assert_misaligned_tail(&err, dtype, 33);
    }
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
    for field in ["encode_us", "submit_us", "wait_us"] {
        assert!(parsed.get(field).is_some(), "missing {field} phase field");
    }
}

#[test]
fn packed_weight_bytes_reach_session_buffers_on_both_fake_backends() {
    let packed: Vec<u8> = (0..34).collect();
    let expected: Vec<f32> = packed
        .chunks(4)
        .map(|chunk| {
            let mut word = [0u8; 4];
            word[..chunk.len()].copy_from_slice(chunk);
            f32::from_le_bytes(word)
        })
        .collect();
    let weights = BTreeMap::from([(
        1,
        DeviceByteBuffer {
            bytes: packed,
            dtype: DeviceDataType::U8,
            packed_format: None,
        },
    )]);
    let inputs = BTreeMap::from([(2, vec![0.0; 9])]);

    for backend in [DeviceBackend::Metal, DeviceBackend::Cuda] {
        let descriptor = packed_weight_descriptor(backend);
        let runtime = fake_runtime(backend);
        let mut host = CompositeHost::with_device(runtime, "fake-packed-device").expect("host");
        let mut session = host
            .create_program_session(&descriptor)
            .expect("session create");
        let receipt = session
            .execute_with_weight_bytes(&inputs, &weights)
            .expect("packed weight execute");
        assert_eq!(receipt.outputs.get(&3), Some(&expected));
        session.teardown().expect("teardown");
        assert_eq!(host.device().expect("device").live_handle_count(), 0);
    }
}

#[test]
fn mmap_aliased_weight_bytes_take_metal_wrap_on_cli_session_path() {
    let mut file = vec![0u8; 64];
    for (index, byte) in file.iter_mut().take(34).enumerate() {
        *byte = index as u8;
    }
    let path = unique_temp("mmap-cli-wrap.bin");
    fs::write(&path, &file).expect("write fixture");
    let mapped = MappedWeightFile::open(&path).expect("mmap");
    let map = BTreeMap::from([(
        1,
        WeightFileRange {
            offset: 0,
            len: 34,
            elems: 9,
        },
    )]);
    let table = gguf_region_table(mapped.bytes(), &map).expect("region table");
    let aliased = inputs_from_mapped_gguf(&mapped, &map, &table).expect("alias");
    assert_eq!(
        aliased.byte_map()[&1].bytes.as_ptr(),
        mapped.bytes().as_ptr(),
        "aliased weight bytes must keep the mmap pointer"
    );
    let inputs = BTreeMap::from([(2, vec![0.0; 9])]);
    let expected: Vec<f32> = file[..36]
        .chunks(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("word")))
        .collect();

    let mut metal = fake_runtime(DeviceBackend::Metal);
    metal
        .retain_mapped_weight_file(mapped.clone())
        .expect("retain mapping");
    let mut host = CompositeHost::with_device(metal, "fake-metal-wrap").expect("host");
    let mut session = host
        .create_program_session(&packed_weight_descriptor(DeviceBackend::Metal))
        .expect("session create");
    let receipt = session
        .execute_with_weight_bytes(&inputs, aliased.byte_map())
        .expect("mmap weight execute");
    assert_eq!(receipt.outputs.get(&3), Some(&expected));
    session.teardown().expect("teardown");
    match host.device().expect("device") {
        DeviceRuntime::Metal(runtime) => assert!(
            runtime.mmap_wrap_count() >= 1,
            "CLI weight upload must reach the Metal mmap wrap branch"
        ),
        DeviceRuntime::Cuda(_) => panic!("expected Metal"),
    }

    let mut cuda = fake_runtime(DeviceBackend::Cuda);
    cuda.retain_mapped_weight_file(mapped.clone())
        .expect("cuda retain is a no-op");
    let mut cuda_host = CompositeHost::with_device(cuda, "fake-cuda-wrap").expect("cuda host");
    let mut cuda_session = cuda_host
        .create_program_session(&packed_weight_descriptor(DeviceBackend::Cuda))
        .expect("cuda session");
    let cuda_receipt = cuda_session
        .execute_with_weight_bytes(&inputs, aliased.byte_map())
        .expect("cuda copies the same raw bytes");
    assert_eq!(cuda_receipt.outputs.get(&3), Some(&expected));
    cuda_session.teardown().expect("cuda teardown");

    drop(aliased);
    drop(mapped);
    let _ = fs::remove_file(&path);
}

#[test]
fn mmap_region_table_aliases_raw_packed_bytes() {
    let mut file = Vec::new();
    file.extend_from_slice(&[0xaa, 0xbb, 0xcc, 0xdd]);
    file.extend_from_slice(&1.0f32.to_le_bytes());
    file.extend_from_slice(&2.0f32.to_le_bytes());
    file.extend_from_slice(&u32::to_le_bytes(0xff81_0000));
    let path = unique_temp("mmap-packed.bin");
    fs::write(&path, &file).expect("write fixture");
    let mapped = MappedWeightFile::open(&path).expect("mmap");
    let map = BTreeMap::from([(
        7,
        WeightFileRange {
            offset: 4,
            len: 12,
            elems: 3,
        },
    )]);
    let table = gguf_region_table(mapped.bytes(), &map).expect("region table");
    assert_eq!(table.data_start, 0, "non-GGUF fixture uses data_start 0");
    assert_eq!(table.abs_starts, vec![4]);
    assert_eq!(table.abs_ends, vec![16]);
    let owned = inputs_from_gguf(&file, &map).expect("owned");
    let aliased = inputs_from_mapped_gguf(&mapped, &map, &table).expect("mapped");
    let values = &aliased.byte_map()[&7];
    let owned_values = &owned.byte_map()[&7];
    assert_eq!(values.dtype, DeviceDataType::U8);
    assert_eq!(values.bytes, owned_values.bytes);
    assert_eq!(values.bytes, file[4..16]);
    drop(aliased);
    drop(mapped);
    let _ = fs::remove_file(&path);
}

#[test]
fn mmap_region_table_parses_gguf_data_start() {
    let mut file = b"GGUF".to_vec();
    file.extend_from_slice(&3u32.to_le_bytes());
    file.extend_from_slice(&0u64.to_le_bytes());
    file.extend_from_slice(&0u64.to_le_bytes());
    file.resize(32, 0);
    file.extend_from_slice(&3.0f32.to_le_bytes());
    let path = unique_temp("mmap-gguf.bin");
    fs::write(&path, &file).expect("write gguf");
    let mapped = MappedWeightFile::open(&path).expect("mmap");
    let map = BTreeMap::from([(
        1,
        WeightFileRange {
            offset: 32,
            len: 4,
            elems: 1,
        },
    )]);
    let table = gguf_region_table(mapped.bytes(), &map).expect("region table");
    assert_eq!(table.data_start, 32);
    assert_eq!(table.abs_starts, vec![32]);
    assert_eq!(table.abs_ends, vec![36]);
    let aliased = inputs_from_mapped_gguf(&mapped, &map, &table).expect("mapped");
    assert_eq!(
        aliased.byte_map()[&1].bytes,
        file[32..36],
        "GGUF data bytes stay raw"
    );
    drop(aliased);
    drop(mapped);
    let _ = fs::remove_file(&path);
}

#[test]
fn mmap_region_table_rejects_range_before_data_start() {
    let mut file = b"GGUF".to_vec();
    file.extend_from_slice(&3u32.to_le_bytes());
    file.extend_from_slice(&0u64.to_le_bytes());
    file.extend_from_slice(&0u64.to_le_bytes());
    file.resize(36, 0);
    let map = BTreeMap::from([(
        1,
        WeightFileRange {
            offset: 8,
            len: 4,
            elems: 1,
        },
    )]);
    let err = gguf_region_table(&file, &map).expect_err("header range is not a data region");
    assert!(err.message.contains("data_start"), "{}", err.message);
}

fn unique_temp(name: &str) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("faber-m5-u3-{}-{name}", std::process::id()));
    path
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
