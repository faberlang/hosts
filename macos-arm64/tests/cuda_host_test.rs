//! CUDA host admission + lifecycle sequencing tests (Track C2 path A).

use faber::Valor;
use faber_host_macos_arm64::cuda_host::E_CUDA_DRIVER;
use faber_host_macos_arm64::{
    probe_cuda_environment, CudaHostSession, FakeCudaDriver, E_CUDA_INVALID_HANDLE,
    E_CUDA_UNAVAILABLE,
};

#[test]
fn probe_reports_structured_admission_without_claiming_product_run() {
    let report = probe_cuda_environment();
    // This machine is expected to be fail-closed for product CUDA; either way
    // the report must be structured and must not imply a completed launch.
    assert!(!report.reason.is_empty());
    if !report.admitted {
        assert!(report.nvidia_smi.is_none());
        assert!(report.libcuda_candidates.is_empty());
    }
}

#[test]
fn try_open_fails_closed_when_environment_unavailable() {
    let result = CudaHostSession::try_open();
    // On this hunter machine nvidia-smi/libcuda are absent → Err
    // (E_CUDA_UNAVAILABLE). If a future proof machine admits (dlopen +
    // cuInit succeed), the real Driver API binding is live and loading a bogus
    // PTX image must fail closed as a driver-level error — never
    // product-claim a launch.
    match result {
        Err(error) => {
            assert_eq!(error.code, E_CUDA_UNAVAILABLE);
        }
        Ok(mut session) => {
            assert!(session.is_admitted());
            let err = session
                .load_module(b"not-a-real-ptx")
                .expect_err("system adapter must not product-launch without valid PTX");
            assert_eq!(err.code, E_CUDA_DRIVER);
        }
    }
}

#[test]
fn fake_driver_sequences_elementwise_add_without_product_label() {
    let mut session =
        CudaHostSession::with_driver(Box::new(FakeCudaDriver::default())).expect("fake admit");
    assert!(session.is_admitted());

    let module = session
        .load_module(b"// fake compiler-owned image bytes")
        .expect("load");
    let a = session.alloc_bytes(8).expect("alloc a");
    let b = session.alloc_bytes(8).expect("alloc b");
    let out = session.alloc_bytes(8).expect("alloc out");

    // Control frames carry only handle ids — never f32 payload bytes.
    match CudaHostSession::handle_frame_data(a) {
        Valor::Tabula(map) => {
            assert!(map.contains_key("cuda_handle"));
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
    assert_eq!(err.code, E_CUDA_INVALID_HANDLE);
}

#[test]
fn fake_unavailable_driver_rejects_session_open() {
    let err = CudaHostSession::with_driver(Box::new(FakeCudaDriver::unavailable()))
        .expect_err("unavailable fake");
    assert_eq!(err.code, E_CUDA_UNAVAILABLE);
}
