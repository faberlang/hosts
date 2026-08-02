//! Metal host admission + lifecycle sequencing tests (lane M, M1 skeleton).

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
}
