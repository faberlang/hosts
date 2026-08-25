//! Metal host admission + lifecycle sequencing tests (lane M; M1 skeleton +
//! M4 C5 API parity closure).

use faber::Valor;
use faber_host_macos_arm64::device_descriptor::{
    DescriptorRuntimeSource, DeviceDataType, E_DEVICE_ENTRY_MISMATCH, E_DEVICE_SHAPE_MISMATCH,
};
use faber_host_macos_arm64::device_host::{DeviceLaunchBinding, DeviceRuntime, DeviceSession};
use faber_host_macos_arm64::metal_host::E_METAL_DRIVER;
use faber_host_macos_arm64::metal_host::{MappedWeightFile, MetalHandleId, MetalLaunchBinding};
use faber_host_macos_arm64::{
    probe_metal_environment, CudaHostSession, FakeCudaDriver, FakeMetalDriver, MetalHostSession,
    E_METAL_INVALID_HANDLE, E_METAL_UNAVAILABLE, E_METAL_UNSUPPORTED,
};
use host_coordinator::{DeviceBackend, DeviceHandle, DeviceHandleKind};

#[test]
fn probe_reports_structured_admission_without_claiming_product_run() {
    let report = probe_metal_environment();
    // On Apple Silicon macOS the system default Metal device is present and
    // the report admits; anywhere else it must fail closed. Either way the
    // report must be structured and must not imply a completed kernel launch.
    assert!(!report.reason.is_empty());
    if report.admitted {
        assert!(report.mtl_device.is_some());
    } else {
        assert!(report.mtl_device.is_none());
    }
}

#[test]
fn error_codes_mirror_cuda_family() {
    assert_eq!(E_METAL_UNAVAILABLE, "E_METAL_UNAVAILABLE");
    assert_eq!(E_METAL_UNSUPPORTED, "E_METAL_UNSUPPORTED");
    assert_eq!(E_METAL_INVALID_HANDLE, "E_METAL_INVALID_HANDLE");
    assert_eq!(E_METAL_DRIVER, "E_METAL_DRIVER");
}

#[test]
fn fake_driver_sequences_elementwise_add_without_product_label() {
    let mut session =
        MetalHostSession::with_driver(Box::new(FakeMetalDriver::default())).expect("fake admit");
    assert!(session.is_admitted());

    let module = session
        .load_module(b"// fake compiler-owned image bytes")
        .expect("load");
    let a = session.alloc_bytes(8).expect("alloc a");
    let b = session.alloc_bytes(8).expect("alloc b");
    let out = session.alloc_bytes(8).expect("alloc out");

    // Control frames carry only handle ids — never f32 payload bytes.
    match MetalHostSession::handle_frame_data(a) {
        Valor::Tabula(map) => {
            assert!(map.contains_key("metal_handle"));
            assert!(!map.values().any(|v| matches!(v, Valor::Octeti(_))));
        }
        other => panic!("expected tabula handle control: {other:?}"),
    }

    session.copy_in_f32(a, &[1.0, 2.0]).expect("copy a");
    session.copy_in_f32(b, &[3.0, 4.0]).expect("copy b");
    let packed = session.alloc_bytes(4).expect("alloc packed");
    session
        .copy_in_bytes(packed, &[0x11, 0x22, 0x33], DeviceDataType::U8)
        .expect("packed-region 3-byte admit pads to 4");
    session
        .launch_elementwise_add_f32(module, a, b, out)
        .expect("launch");
    let values = session.readback_f32(out).expect("readback");
    assert_eq!(values, vec![4.0, 6.0]);

    session.release(out).expect("release out");
    let err = session
        .readback_f32(out)
        .expect_err("stale handle must fail closed");
    assert_eq!(err.code, E_METAL_INVALID_HANDLE);
}

#[test]
fn device_session_byte_surface_round_trips_metal_and_declares_retention() {
    let mut runtime = DeviceRuntime::Metal(
        MetalHostSession::with_driver(Box::new(FakeMetalDriver::default())).expect("fake admit"),
    );
    assert!(runtime.supports_mapped_weight_retention());

    let cuda_runtime = DeviceRuntime::Cuda(
        CudaHostSession::with_driver(Box::new(FakeCudaDriver::default())).expect("fake admit"),
    );
    assert!(!cuda_runtime.supports_mapped_weight_retention());

    for payload in [
        vec![0x11],
        vec![0x22, 0x33],
        vec![0x44, 0x55, 0x66],
        (0..34).collect(),
    ] {
        let handle = runtime.alloc_bytes(payload.len()).expect("alloc");
        runtime
            .copy_in_bytes(&handle, &payload, DeviceDataType::U8)
            .expect("byte upload");
        assert_eq!(
            runtime
                .readback_bytes(&handle, DeviceDataType::U8)
                .expect("byte readback"),
            payload
        );
    }
}

#[test]
fn fake_unavailable_driver_rejects_session_open() {
    let err = MetalHostSession::with_driver(Box::new(FakeMetalDriver::unavailable()))
        .expect_err("unavailable fake");
    assert_eq!(err.code, E_METAL_UNAVAILABLE);
}

#[test]
fn try_open_opens_live_session_or_fails_closed() {
    // On this machine the system default Metal device is present → the live
    // session opens and a module image that reaches the MSL compiler but is
    // rejected must fail closed as a driver-level error, never product-claim a
    // launch. Without a Metal device, admission fails closed with
    // E_METAL_UNAVAILABLE. Mirrors the CUDA try_open test.
    match MetalHostSession::try_open() {
        Err(error) => {
            assert_eq!(error.code, E_METAL_UNAVAILABLE);
        }
        Ok(mut session) => {
            assert!(session.is_admitted());
            // A kernel void entry exists but the body is invalid MSL, so the
            // runtime compile fails (E_METAL_DRIVER) — not E_INVALID_ARGS,
            // which is reserved for structurally invalid images.
            let err = session
                .load_module(
                    b"kernel void broken(uint id [[thread_position_in_grid]]) { not_valid_msl; }",
                )
                .expect_err("system adapter must not product-launch without valid MSL");
            assert_eq!(err.code, E_METAL_DRIVER);
        }
    }
}

#[test]
fn fake_driver_sequences_generalized_launch_kernel() {
    let mut session =
        MetalHostSession::with_driver(Box::new(FakeMetalDriver::default())).expect("fake admit");

    let module = session
        .load_module(b"// fake compiler-owned image bytes")
        .expect("load");
    let a = session.alloc_bytes(8).expect("alloc a");
    let b = session.alloc_bytes(8).expect("alloc b");
    let out = session.alloc_bytes(8).expect("alloc out");

    session.copy_in_f32(a, &[1.0, 2.0]).expect("copy a");
    session.copy_in_f32(b, &[3.0, 4.0]).expect("copy b");
    session
        .launch_kernel_3d(module, "add_one", &[a, b, out], 2, 2, 1, 8, 8, 1)
        .expect("generalized 3D launch");
    let values = session.readback_f32(out).expect("readback");
    assert_eq!(values, vec![4.0, 6.0]);

    // The explicit session barrier is callable independently of a launch.
    session.sync().expect("session sync barrier");
}

#[test]
fn fake_driver_accepts_all_gea2_entry_arities_from_declared_table() {
    const GEA2_ENTRIES: [(&str, usize); 13] = [
        ("rmsnorm", 3),
        ("gemm_qo", 3),
        ("gemm_kv", 3),
        ("gemm_gate_up", 3),
        ("gemm_down", 3),
        // The rope launches bind the packed output plus every per-head
        // window write (15 q_head / 5 k_head) declared by the GEA2-U5e
        // per-instance window repair (radix 5f96ed340).
        ("rope_q", 18),
        ("rope_k", 8),
        ("transpose", 2),
        ("score_gemm", 4),
        ("causal_softmax", 2),
        ("context_gemm", 3),
        ("swiglu", 3),
        ("residual_add", 3),
    ];

    let mut driver = FakeMetalDriver::default();
    for (entry, _) in GEA2_ENTRIES {
        driver = driver.with_known_entry(entry);
    }
    let mut session = MetalHostSession::with_driver(Box::new(driver)).expect("fake admit");
    let module = session.load_module(b"gea2 fake module").expect("load");

    for (entry, arity) in GEA2_ENTRIES {
        let mut buffers = Vec::with_capacity(arity);
        for _ in 0..arity {
            buffers.push(session.alloc_bytes(4).expect("alloc launch binding"));
        }
        session
            .launch_kernel(module, entry, &buffers, 1, 1)
            .unwrap_or_else(|error| panic!("GEA2 entry {entry} with arity {arity}: {error}"));
    }

    assert_eq!(
        session.command_submit_count(),
        0,
        "fake GEA2 launches stay encode-only until sync"
    );
    assert_eq!(session.blocking_wait_count(), 0);
    session.sync().expect("fake GEA2 sync");
    assert_eq!(session.command_submit_count(), 1);
    assert_eq!(session.blocking_wait_count(), 1);
}

#[test]
fn fake_driver_rejects_unknown_entry_against_declared_gea2_table() {
    let mut driver = FakeMetalDriver::default();
    for entry in [
        "rmsnorm",
        "gemm_qo",
        "gemm_kv",
        "gemm_gate_up",
        "gemm_down",
        "rope_q",
        "rope_k",
        "transpose",
        "score_gemm",
        "causal_softmax",
        "context_gemm",
        "swiglu",
        "residual_add",
    ] {
        driver = driver.with_known_entry(entry);
    }
    let mut session = MetalHostSession::with_driver(Box::new(driver)).expect("fake admit");
    let module = session.load_module(b"gea2 fake module").expect("load");
    let buffers = [
        session.alloc_bytes(4).expect("alloc input a"),
        session.alloc_bytes(4).expect("alloc input b"),
        session.alloc_bytes(4).expect("alloc output"),
    ];

    let error = session
        .launch_kernel(module, "not_a_gea2_entry", &buffers, 1, 1)
        .expect_err("unknown declared-module entry must fail closed");
    assert_eq!(error.code, E_DEVICE_ENTRY_MISMATCH);
    assert_eq!(session.command_submit_count(), 0);
    assert_eq!(session.blocking_wait_count(), 0);
}

/// W8-U1: several kernel encodes stay encode-only until `sync`, which is
/// one command-buffer submit and one blocking wait. Readback after that
/// flush does not add another wait (coalesced).
#[test]
fn fake_driver_batches_encodes_into_one_submit_and_wait() {
    let mut session =
        MetalHostSession::with_driver(Box::new(FakeMetalDriver::default())).expect("fake admit");

    let module = session
        .load_module(b"// fake compiler-owned image bytes")
        .expect("load");
    let a = session.alloc_bytes(8).expect("alloc a");
    let b = session.alloc_bytes(8).expect("alloc b");
    let mid = session.alloc_bytes(8).expect("alloc mid");
    let out = session.alloc_bytes(8).expect("alloc out");

    session.copy_in_f32(a, &[1.0, 2.0]).expect("copy a");
    session.copy_in_f32(b, &[3.0, 4.0]).expect("copy b");

    session
        .launch_kernel(module, "add_one", &[a, b, mid], 1, 2)
        .expect("encode 1");
    session
        .launch_kernel(module, "add_one", &[mid, b, out], 1, 2)
        .expect("encode 2");

    assert_eq!(
        session.command_submit_count(),
        0,
        "encodes must not submit a command buffer"
    );
    assert_eq!(session.blocking_wait_count(), 0, "encodes must not wait");

    session.sync().expect("step-boundary commit+wait");
    assert_eq!(session.command_submit_count(), 1);
    assert_eq!(session.blocking_wait_count(), 1);

    let values = session.readback_f32(out).expect("coalesced readback");
    assert_eq!(values, vec![7.0, 10.0]);
    assert_eq!(
        session.command_submit_count(),
        1,
        "readback after sync must not submit again"
    );
    assert_eq!(
        session.blocking_wait_count(),
        1,
        "readback after sync must not wait again"
    );
}

#[test]
fn session_fails_closed_on_guard_checks() {
    let mut session =
        MetalHostSession::with_driver(Box::new(FakeMetalDriver::default())).expect("fake admit");

    let err = session.load_module(b"").expect_err("empty module image");
    assert_eq!(err.code, "E_INVALID_ARGS");

    let err = session.alloc_bytes(0).expect_err("zero-length buffer");
    assert_eq!(err.code, "E_INVALID_ARGS");

    let a = session.alloc_bytes(8).expect("alloc a");
    let b = session.alloc_bytes(8).expect("alloc b");
    let out = session.alloc_bytes(4).expect("alloc out (wrong size)");

    let err = session
        .copy_in_f32(a, &[1.0, 2.0, 3.0])
        .expect_err("copy_in size mismatch");
    assert_eq!(err.code, "E_INVALID_ARGS");

    let module = session.load_module(b"image").expect("load module");
    let err = session
        .launch_elementwise_add_f32(module, a, b, out)
        .expect_err("unequal buffer sizes");
    assert_eq!(err.code, "E_INVALID_ARGS");

    // A buffer passed as the module handle is also fail-closed, distinct from
    // a stale id.
    let err = session
        .launch_elementwise_add_f32(a, a, b, out)
        .expect_err("module handle must be a module");
    assert_eq!(err.code, "E_INVALID_ARGS");

    // Generalized launch guards: empty entry, a non-buffer in the slice, and
    // a stale handle all fail closed before the driver is touched.
    let err = session
        .launch_kernel(module, "", &[a], 1, 8)
        .expect_err("empty entry name");
    assert_eq!(err.code, "E_INVALID_ARGS");

    let err = session
        .launch_kernel(module, "add_one", &[module], 1, 8)
        .expect_err("non-buffer handle in launch slice");
    assert_eq!(err.code, "E_INVALID_ARGS");

    let released = session.alloc_bytes(8).expect("alloc released");
    session.release(released).expect("release released");
    let err = session
        .launch_kernel(module, "add_one", &[a, released], 1, 8)
        .expect_err("stale handle in launch slice");
    assert_eq!(err.code, E_METAL_INVALID_HANDLE);
}

/// Emitted kernel text (radix metal-text, U2 runtime-extent guard): input at
/// buffer 0, output at buffer 1, element count at buffer 2.
const ADD_ONE_MSL: &str = r#"#include <metal_stdlib>
using namespace metal;

kernel void add_one(
    device const float* x_in [[buffer(0)]],
    device const uint* extent_2 [[buffer(2)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  if (id >= extent_2[0]) {
    return;
  }
  float x = x_in[id];
  output[id] = (x + 1.0f);
}
"#;

#[test]
fn system_driver_compiles_msl_launches_add_one_and_reads_back() {
    // Environment-gated: only runs where a real Metal device exists. On a
    // machine without Metal, admission fails closed and the real-binding proof
    // is skipped (the fake-driver tests cover sequencing everywhere).
    let Ok(mut session) = MetalHostSession::try_open() else {
        return;
    };
    assert!(session.is_admitted());

    let module = session
        .load_module(ADD_ONE_MSL.as_bytes())
        .expect("runtime MSL compile");
    let a = session.alloc_bytes(16 * 4).expect("alloc a");
    let b = session.alloc_bytes(16 * 4).expect("alloc b");
    let out = session.alloc_bytes(16 * 4).expect("alloc out");

    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    session.copy_in_f32(a, &input).expect("copy in a");
    session
        .launch_elementwise_add_f32(module, a, b, out)
        .expect("launch add_one");
    session.sync().expect("explicit session sync barrier");
    let values = session.readback_f32(out).expect("readback out");

    let expected: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    assert_eq!(values, expected);
}

/// MSL kernel for the real-device generalized launch proof: a plain
/// two-buffer kernel (input@0, output@1) with no runtime-extent channel, so
/// the session `launch_kernel` path is exercised directly over an explicit
/// buffer slice and grid/block shape.
const SCALE_TWO_MSL: &str = r#"#include <metal_stdlib>
using namespace metal;

kernel void scale_two(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  y[id] = (x[id] * 2.0f);
}
"#;

#[test]
fn system_session_launch_kernel_dispatches_over_buffer_slice() {
    // Real-device proof of the generalized session launch: a named entry
    // dispatched over a buffer slice with an explicit grid/block shape,
    // without the legacy elementwise-add session method. This is the PAR-1
    // done-when: a caller can launch a named entry over a buffer slice.
    let Ok(mut session) = MetalHostSession::try_open() else {
        return;
    };
    let module = session
        .load_module(SCALE_TWO_MSL.as_bytes())
        .expect("runtime MSL compile");
    let a = session.alloc_bytes(16 * 4).expect("alloc a");
    let out = session.alloc_bytes(16 * 4).expect("alloc out");

    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    session.copy_in_f32(a, &input).expect("copy in a");
    session
        .launch_kernel(module, "scale_two", &[a, out], 1, 16)
        .expect("generalized launch");
    session.sync().expect("session sync barrier");
    let values = session.readback_f32(out).expect("readback out");

    let expected: Vec<f32> = (0..16).map(|i| (i as f32) * 2.0).collect();
    assert_eq!(values, expected);
}

/// MSL module declaring TWO kernel entries (S2-5 multi-kernel modules): the
/// proven `scale_two` and `add_one` bodies in one library, so the driver
/// loads one module and dispatches both entries — the Metal lane of the
/// ordinary two-kernel chain.
const TWO_ENTRY_MSL: &str = r#"#include <metal_stdlib>
using namespace metal;

kernel void scale_two(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  y[id] = (x[id] * 2.0f);
}

kernel void add_one(
    device const float* x [[buffer(0)]],
    device const float* unused [[buffer(2)]],
    device float* y [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  y[id] = (x[id] + 1.0f);
}
"#;

#[test]
fn system_driver_loads_multi_entry_module_and_dispatches_both_entries() {
    // Real-device proof of S2-5 multi-entry Metal modules: one module holds
    // a pipeline per declared `kernel void` entry, and the generalized
    // launch resolves each entry by name (mirroring cuModuleGetFunction on
    // the CUDA lane). Environment-gated like the other real-binding proofs.
    let Ok(mut session) = MetalHostSession::try_open() else {
        return;
    };
    let module = session
        .load_module(TWO_ENTRY_MSL.as_bytes())
        .expect("runtime MSL compile with two entries");
    let a = session.alloc_bytes(16 * 4).expect("alloc a");
    let out = session.alloc_bytes(16 * 4).expect("alloc out");

    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    session.copy_in_f32(a, &input).expect("copy in a");

    // Kernel 1: scale_two (a -> out). Kernel 2: add_one over the SAME
    // device-resident buffer (out -> out): the second entry reads what the
    // first wrote, exactly the two-kernel chain shape.
    session
        .launch_kernel(module, "scale_two", &[a, out], 1, 16)
        .expect("launch scale_two");
    session
        .launch_kernel(module, "add_one", &[out, out, a], 1, 16)
        .expect("launch add_one");
    session.sync().expect("session sync barrier");

    let values = session.readback_f32(out).expect("readback out");
    let expected: Vec<f32> = (0..16).map(|i| (i as f32) * 2.0 + 1.0).collect();
    assert_eq!(values, expected);
    assert_eq!(
        session.command_submit_count(),
        1,
        "W8-U1: two real-device encodes share one command-buffer submit"
    );
    assert_eq!(
        session.blocking_wait_count(),
        1,
        "W8-U1: one blocking wait at the step-boundary sync"
    );

    // An entry the module does not declare fails closed with the typed
    // E_DEVICE_ENTRY_MISMATCH (multi-entry module, unknown name).
    let err = session
        .launch_kernel(module, "nope", &[a, out], 1, 16)
        .expect_err("unknown entry must fail closed");
    assert_eq!(err.code, E_DEVICE_ENTRY_MISMATCH);
    assert!(err.message.contains("nope"));
}

/// MSL kernel for the G4 real-device repeated-write accumulation proof: an
/// in-place `acc[id] = acc[id] + a[id]` kernel — the emitted elementwise
/// accumulation shape over a persistent device buffer.
const ACCUMULATE_MSL: &str = r#"#include <metal_stdlib>
using namespace metal;

kernel void accumulate(
    device const float* a [[buffer(0)]],
    device float* acc [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  acc[id] = acc[id] + a[id];
}
"#;

#[test]
fn system_driver_accumulates_persistent_buffer_across_repeated_launches() {
    // G4 (P2) real-device proof: a persistent device buffer accumulates
    // across repeated launches WITHOUT re-initialization — the production
    // repeated-write lifecycle. The host zero-fills the buffer once (ZeroFill
    // policy), then every launch adds the input into it. Environment-gated
    // like the other real-binding proofs.
    let Ok(mut session) = MetalHostSession::try_open() else {
        return;
    };
    assert!(session.is_admitted());
    let module = session
        .load_module(ACCUMULATE_MSL.as_bytes())
        .expect("runtime MSL compile");
    let a = session.alloc_bytes(4 * 4).expect("alloc a");
    let acc = session.alloc_bytes(4 * 4).expect("alloc acc");

    // ZeroFill initialization: the accumulation buffer starts defined at zero.
    session
        .copy_in_f32(acc, &[0.0, 0.0, 0.0, 0.0])
        .expect("zero-fill acc");
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    session.copy_in_f32(a, &input).expect("copy in a");

    // Repeated-write: two launches WITHOUT re-initializing acc.
    for _ in 0..2 {
        session
            .launch_kernel(module, "accumulate", &[a, acc], 1, 4)
            .expect("accumulate launch");
        session.sync().expect("session sync barrier");
    }
    let values = session.readback_f32(acc).expect("readback acc");
    assert_eq!(
        values,
        vec![2.0, 4.0, 6.0, 8.0],
        "two launches accumulate 2a"
    );
}

#[test]
fn mmap_weight_file_is_read_only_lazy_mapping() {
    let mut bytes = vec![0u8; 64];
    bytes[16..20].copy_from_slice(&7.0f32.to_le_bytes());
    bytes[20..24].copy_from_slice(&8.0f32.to_le_bytes());
    let path = {
        let mut path = std::env::temp_dir();
        path.push(format!("faber-m5-u3-metal-mmap-{}", std::process::id()));
        path
    };
    std::fs::write(&path, &bytes).expect("write");
    let mapped = MappedWeightFile::open(&path).expect("mmap");
    assert_eq!(mapped.len(), 64);
    assert!(mapped.page_size().is_power_of_two());
    assert!(mapped.mapped_len() >= mapped.len());
    assert_eq!(&mapped.bytes()[16..24], &bytes[16..24]);
    drop(mapped);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn system_driver_admits_mmap_region_without_copy_and_launches() {
    let Ok(mut session) = MetalHostSession::try_open() else {
        return;
    };
    let mut bytes = vec![0u8; 64];
    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    for (index, value) in input.iter().enumerate() {
        let off = index * 4;
        bytes[off..off + 4].copy_from_slice(&value.to_le_bytes());
    }
    let path = {
        let mut path = std::env::temp_dir();
        path.push(format!("faber-m5-u3-metal-launch-{}", std::process::id()));
        path
    };
    std::fs::write(&path, &bytes).expect("write");
    let mapped = MappedWeightFile::open(&path).expect("mmap");
    session
        .retain_mapped_file(mapped.clone())
        .expect("retain mapping");

    let module = session
        .load_module(ADD_ONE_MSL.as_bytes())
        .expect("runtime MSL compile");
    let a = session.alloc_bytes(16 * 4).expect("alloc a");
    let b = session.alloc_bytes(16 * 4).expect("alloc b");
    let out = session.alloc_bytes(16 * 4).expect("alloc out");
    session
        .copy_in_bytes(a, &mapped.bytes()[..64], DeviceDataType::F32)
        .expect("mmap admit");
    session
        .launch_elementwise_add_f32(module, a, b, out)
        .expect("launch add_one");
    session.sync().expect("sync");
    let values = session.readback_f32(out).expect("readback");
    let expected: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    assert_eq!(
        values, expected,
        "mmap-backed input launched at offset zero"
    );
    let echoed = session.readback_f32(a).expect("readback mmap region");
    assert_eq!(echoed, input, "no-copy region matches the mapped file");
    drop(mapped);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn system_driver_mmap_unaligned_region_uses_page_remainder() {
    let Ok(mut session) = MetalHostSession::try_open() else {
        return;
    };
    let mut bytes = vec![0u8; 64];
    bytes[16..20].copy_from_slice(&7.0f32.to_le_bytes());
    bytes[20..24].copy_from_slice(&8.0f32.to_le_bytes());
    let path = {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "faber-m5-u3-metal-unaligned-{}",
            std::process::id()
        ));
        path
    };
    std::fs::write(&path, &bytes).expect("write");
    let mapped = MappedWeightFile::open(&path).expect("mmap");
    session
        .retain_mapped_file(mapped.clone())
        .expect("retain mapping");
    let buffer = session.alloc_bytes(8).expect("alloc region");
    session
        .copy_in_bytes(buffer, &mapped.bytes()[16..24], DeviceDataType::F32)
        .expect("mmap unaligned admit");
    let values = session.readback_f32(buffer).expect("readback");
    assert_eq!(
        values,
        vec![7.0, 8.0],
        "page remainder still names tensor start"
    );
    drop(mapped);
    let _ = std::fs::remove_file(&path);
}

/// Observation copy: dest[i] = src[i] from the bound view.
const OBSERVA_MSL: &str = r#"#include <metal_stdlib>
using namespace metal;

kernel void observa(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
  y[id] = x[id];
}
"#;

fn b4_binding(
    handle: MetalHandleId,
    len_bytes: u64,
    binding_index: u32,
    byte_offset: u64,
    view_span: u64,
) -> DeviceLaunchBinding {
    DeviceLaunchBinding {
        handle: DeviceHandle {
            backend: DeviceBackend::Metal,
            kind: DeviceHandleKind::Buffer { len_bytes },
            id: handle.0,
        },
        binding_index,
        byte_offset,
        view_span,
        runtime_source: DescriptorRuntimeSource::Constant,
    }
}

fn metal_from_b4(binding: DeviceLaunchBinding) -> MetalLaunchBinding {
    MetalLaunchBinding {
        handle: MetalHandleId(binding.handle.id),
        binding_index: binding.binding_index,
        byte_offset: binding.byte_offset,
        view_span: binding.view_span,
    }
}

fn bind_copy(
    src: MetalHandleId,
    src_len: u64,
    src_offset: u64,
    dest: MetalHandleId,
    dest_len: u64,
) -> [MetalLaunchBinding; 2] {
    [
        metal_from_b4(b4_binding(src, src_len, 0, src_offset, dest_len)),
        metal_from_b4(b4_binding(dest, dest_len, 1, 0, dest_len)),
    ]
}

#[test]
fn b6_same_allocation_binds_row_0_and_row_n_through_launch_binding_api() {
    let mut session =
        MetalHostSession::with_driver(Box::new(FakeMetalDriver::default())).expect("fake admit");
    let module = session
        .load_module(b"// fake compiler-owned image bytes")
        .expect("load");
    let row_bytes = 16u64;
    let cache = session.alloc_bytes(32).expect("cache");
    let out = session.alloc_bytes(16).expect("out");
    session
        .copy_in_f32(cache, &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
        .expect("rows");
    let allocs_before = session.driver_counters().buffer_allocs;
    let handles_before = session.live_handle_count();

    session
        .launch_kernel_bound(
            module,
            "observa",
            &bind_copy(cache, 32, 0, out, row_bytes),
            [1, 1, 1],
            [4, 1, 1],
        )
        .expect("row 0");
    session.sync().expect("sync row 0");
    assert_eq!(
        session.readback_f32(out).expect("row 0 readback"),
        vec![1.0, 2.0, 3.0, 4.0]
    );

    session
        .launch_kernel_bound(
            module,
            "observa",
            &bind_copy(cache, 32, row_bytes, out, row_bytes),
            [1, 1, 1],
            [4, 1, 1],
        )
        .expect("row N");
    session.sync().expect("sync row N");
    assert_eq!(
        session.readback_f32(out).expect("row N readback"),
        vec![10.0, 20.0, 30.0, 40.0],
        "same allocation bound at a nonzero offset must read row N"
    );
    assert_eq!(
        session.driver_counters().buffer_allocs,
        allocs_before,
        "bound launch must not allocate a per-kernel temp or cache copy"
    );
    assert_eq!(
        session.live_handle_count(),
        handles_before,
        "row 0 and row N reuse the same handles"
    );
}

#[test]
fn b6_one_cursor_upload_serves_the_whole_step() {
    let mut session =
        MetalHostSession::with_driver(Box::new(FakeMetalDriver::default())).expect("fake admit");
    let module = session
        .load_module(b"// fake compiler-owned image bytes")
        .expect("load");
    let cursor = session.alloc_bytes(16).expect("cursor");
    let dest_pos = session.alloc_bytes(4).expect("position");
    let dest_len = session.alloc_bytes(4).expect("valid_len");
    session
        .copy_in_f32(
            cursor,
            &[
                f32::from_bits(7),
                f32::from_bits(11),
                f32::from_bits(1),
                f32::from_bits(3),
            ],
        )
        .expect("one cursor upload");
    let allocs_before = session.driver_counters().buffer_allocs;

    session
        .launch_kernel_bound(
            module,
            "observa",
            &bind_copy(cursor, 16, 0, dest_pos, 4),
            [1, 1, 1],
            [1, 1, 1],
        )
        .expect("position field");
    session
        .launch_kernel_bound(
            module,
            "observa",
            &bind_copy(cursor, 16, 4, dest_len, 4),
            [1, 1, 1],
            [1, 1, 1],
        )
        .expect("valid_len field");
    session.sync().expect("one step barrier");

    assert_eq!(session.readback_f32(dest_pos).expect("pos")[0].to_bits(), 7);
    assert_eq!(
        session.readback_f32(dest_len).expect("len")[0].to_bits(),
        11
    );
    assert_eq!(
        session.driver_counters().buffer_allocs,
        allocs_before,
        "the same uploaded cursor serves both kernels; no per-kernel temp"
    );
    assert_eq!(session.command_submit_count(), 1);
    assert_eq!(session.blocking_wait_count(), 1);
}

#[test]
fn b6_legacy_offset_zero_wrapper_stays_green() {
    let mut session =
        MetalHostSession::with_driver(Box::new(FakeMetalDriver::default())).expect("fake admit");
    let module = session
        .load_module(b"// fake compiler-owned image bytes")
        .expect("load");
    let a = session.alloc_bytes(8).expect("a");
    let b = session.alloc_bytes(8).expect("b");
    let out = session.alloc_bytes(8).expect("out");
    session.copy_in_f32(a, &[1.0, 2.0]).expect("copy a");
    session.copy_in_f32(b, &[3.0, 4.0]).expect("copy b");
    session
        .launch_kernel(module, "add_one", &[a, b, out], 1, 2)
        .expect("legacy offset-zero wrapper");
    assert_eq!(session.readback_f32(out).expect("readback"), vec![4.0, 6.0]);

    session
        .launch_kernel_bound(
            module,
            "add_one",
            &[
                metal_from_b4(b4_binding(a, 8, 0, 0, 8)),
                metal_from_b4(b4_binding(b, 8, 1, 0, 8)),
                metal_from_b4(b4_binding(out, 8, 2, 0, 8)),
            ],
            [1, 1, 1],
            [2, 1, 1],
        )
        .expect("B4 offset-zero bound launch");
    assert_eq!(
        session.readback_f32(out).expect("bound readback"),
        vec![4.0, 6.0]
    );
}

#[test]
fn b6_launch_binding_offset_past_span_fails_closed() {
    let mut session =
        MetalHostSession::with_driver(Box::new(FakeMetalDriver::default())).expect("fake admit");
    let module = session
        .load_module(b"// fake compiler-owned image bytes")
        .expect("load");
    let cache = session.alloc_bytes(16).expect("cache");
    let out = session.alloc_bytes(8).expect("out");
    let err = session
        .launch_kernel_bound(
            module,
            "observa",
            &bind_copy(cache, 16, 16, out, 8),
            [1, 1, 1],
            [2, 1, 1],
        )
        .expect_err("offset past the allocation must fail closed");
    assert_eq!(err.code, E_DEVICE_SHAPE_MISMATCH);
}

#[test]
fn b6_mapped_region_composes_page_remainder_with_binding_offset() {
    let mut session =
        MetalHostSession::with_driver(Box::new(FakeMetalDriver::default())).expect("fake admit");
    let mut bytes = vec![0u8; 64];
    bytes[8..12].copy_from_slice(&1.0f32.to_le_bytes());
    bytes[16..20].copy_from_slice(&2.0f32.to_le_bytes());
    bytes[24..28].copy_from_slice(&3.0f32.to_le_bytes());
    let path = {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "faber-b6-metal-compose-fake-{}",
            std::process::id()
        ));
        path
    };
    std::fs::write(&path, &bytes).expect("write");
    let mapped = MappedWeightFile::open(&path).expect("mmap");
    session
        .retain_mapped_file(mapped.clone())
        .expect("retain mapping");
    let module = session
        .load_module(b"// fake compiler-owned image bytes")
        .expect("load");
    let region = session.alloc_bytes(16).expect("mapped region");
    session
        .copy_in_bytes(region, &mapped.bytes()[16..32], DeviceDataType::F32)
        .expect("mmap unaligned admit");
    assert!(
        session.mmap_wrap_count() >= 1,
        "mapped copy_in must take the wrap branch"
    );
    let echoed = session.readback_f32(region).expect("logical start");
    assert_eq!(
        echoed[..1],
        [2.0],
        "page remainder still names the admitted tensor start"
    );
    let out = session.alloc_bytes(4).expect("out");
    session
        .launch_kernel_bound(
            module,
            "observa",
            &bind_copy(region, 16, 8, out, 4),
            [1, 1, 1],
            [1, 1, 1],
        )
        .expect("nonzero launch offset over mapped region");
    session.sync().expect("sync");
    let values = session.readback_f32(out).expect("composed read");
    assert_eq!(
        values,
        vec![3.0],
        "composed bind must be remainder+offset (file[24]=3), not remainder (2) or offset-only (1)"
    );
    drop(mapped);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn system_driver_same_allocation_binds_row_0_and_row_n() {
    let Ok(mut session) = MetalHostSession::try_open() else {
        return;
    };
    let module = session
        .load_module(OBSERVA_MSL.as_bytes())
        .expect("runtime MSL compile");
    let cache = session.alloc_bytes(32).expect("cache");
    let out = session.alloc_bytes(16).expect("out");
    session
        .copy_in_f32(cache, &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
        .expect("rows");
    session
        .launch_kernel_bound(
            module,
            "observa",
            &bind_copy(cache, 32, 0, out, 16),
            [1, 1, 1],
            [4, 1, 1],
        )
        .expect("row 0");
    session.sync().expect("sync row 0");
    assert_eq!(
        session.readback_f32(out).expect("row 0"),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    session
        .launch_kernel_bound(
            module,
            "observa",
            &bind_copy(cache, 32, 16, out, 16),
            [1, 1, 1],
            [4, 1, 1],
        )
        .expect("row N");
    session.sync().expect("sync row N");
    assert_eq!(
        session.readback_f32(out).expect("row N"),
        vec![10.0, 20.0, 30.0, 40.0]
    );
}

#[test]
fn system_driver_mapped_region_composes_page_remainder_with_binding_offset() {
    let Ok(mut session) = MetalHostSession::try_open() else {
        return;
    };
    let mut bytes = vec![0u8; 64];
    bytes[8..12].copy_from_slice(&1.0f32.to_le_bytes());
    bytes[16..20].copy_from_slice(&2.0f32.to_le_bytes());
    bytes[24..28].copy_from_slice(&3.0f32.to_le_bytes());
    let path = {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "faber-b6-metal-compose-live-{}",
            std::process::id()
        ));
        path
    };
    std::fs::write(&path, &bytes).expect("write");
    let mapped = MappedWeightFile::open(&path).expect("mmap");
    session
        .retain_mapped_file(mapped.clone())
        .expect("retain mapping");
    let module = session
        .load_module(OBSERVA_MSL.as_bytes())
        .expect("runtime MSL compile");
    let region = session.alloc_bytes(16).expect("mapped region");
    session
        .copy_in_bytes(region, &mapped.bytes()[16..32], DeviceDataType::F32)
        .expect("mmap unaligned admit");
    let out = session.alloc_bytes(4).expect("out");
    session
        .launch_kernel_bound(
            module,
            "observa",
            &bind_copy(region, 16, 8, out, 4),
            [1, 1, 1],
            [1, 1, 1],
        )
        .expect("composed setBuffer offset");
    session.sync().expect("sync");
    let values = session.readback_f32(out).expect("composed read");
    assert_eq!(
        values,
        vec![3.0],
        "live setBuffer must add the launch offset to the mmap page remainder"
    );
    drop(mapped);
    let _ = std::fs::remove_file(&path);
}
