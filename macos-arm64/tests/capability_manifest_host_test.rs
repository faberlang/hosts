//! Capability manifest host identity is selected at the call site.

use faber_host_macos_arm64::CapabilityManifest;

#[test]
fn from_parts_records_declared_host_identity() {
    let cuda =
        CapabilityManifest::from_parts(CapabilityManifest::HOST_CUDA_LINUX, Vec::new(), Vec::new());
    assert_eq!(cuda.host, CapabilityManifest::HOST_CUDA_LINUX);
    assert_ne!(cuda.host, "macos-arm64");
    assert_eq!(CapabilityManifest::HOST_CUDA_LINUX, "cuda-linux");

    let macos = CapabilityManifest::from_parts(
        CapabilityManifest::HOST_MACOS_ARM64,
        Vec::new(),
        Vec::new(),
    );
    assert_eq!(macos.host, "macos-arm64");
}
