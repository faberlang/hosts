use serde::{Deserialize, Serialize};

use crate::kernel::SyscallInfo;

/// Machine-readable surface exported by this host.
///
/// Strict compilation will eventually consume a richer version of this manifest.
/// The first slice records only built-in syscalls and registered providers so
/// host capability discovery has a concrete artifact before policy exists.
///
/// `host` is a declared identity selected at the call site from the admitted
/// backend — never a crate-wide hardcoded string.
/// [`Self::HOST_MACOS_ARM64`] is the macOS product (Metal or CPU-only).
/// [`Self::HOST_CUDA_LINUX`] is a CUDA-admitted host.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityManifest {
    /// Product host identity. Selected from the admitted backend at the
    /// [`Self::from_parts`] call site.
    pub host: String,
    pub manifest_version: u32,
    pub builtins: Vec<SyscallManifest>,
    pub providers: Vec<RegisteredProvider>,
}

impl CapabilityManifest {
    /// macOS product host spelling. Metal-admitted and CPU-only hosts keep
    /// this identity.
    pub const HOST_MACOS_ARM64: &'static str = "macos-arm64";

    /// CUDA-admitted host spelling (Linux/CUDA product identity).
    ///
    /// Documented per-backend constant (ELP-08 CLH-3). Not a driver-derived
    /// string. ELP-04 spawn-table spelling is a separate product decision.
    pub const HOST_CUDA_LINUX: &'static str = "cuda-linux";

    pub fn from_parts(
        host: impl Into<String>,
        syscalls: Vec<SyscallInfo>,
        providers: Vec<RegisteredProvider>,
    ) -> Self {
        Self {
            host: host.into(),
            manifest_version: 1,
            builtins: syscalls.into_iter().map(SyscallManifest::from).collect(),
            providers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_parts_records_declared_host_identity() {
        let cuda = CapabilityManifest::from_parts(
            CapabilityManifest::HOST_CUDA_LINUX,
            Vec::new(),
            Vec::new(),
        );
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
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyscallManifest {
    pub name: String,
    pub prefix: String,
    pub summary: String,
}

impl From<SyscallInfo> for SyscallManifest {
    fn from(info: SyscallInfo) -> Self {
        Self {
            name: info.name,
            prefix: info.prefix,
            summary: info.summary,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisteredProvider {
    pub name: String,
    pub owner: String,
    pub prefix: Option<String>,
    pub calls: Vec<String>,
}
