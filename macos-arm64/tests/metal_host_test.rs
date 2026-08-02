//! Metal host admission + lifecycle sequencing tests (lane M; M1 skeleton +
//! M4 C5 API parity closure).

use faber::Valor;
use faber_host_macos_arm64::metal_host::E_METAL_DRIVER;
use faber_host_macos_arm64::{
    probe_metal_environment, FakeMetalDriver, MetalHostSession, E_METAL_INVALID_HANDLE,
    E_METAL_UNAVAILABLE, E_METAL_UNSUPPORTED,
};

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
