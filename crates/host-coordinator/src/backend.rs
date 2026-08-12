//! Physical device backend discriminator (HOSTS-COORD).
//!
//! The physical backend a device-lifecycle fact or handle belongs to. Moved
//! from `faber-runtime/src/device.rs` (device module split, inventory §3.2):
//! the **physical backend fact** lands with HOSTS-COORD; the
//! build/selection surface (`DeviceSelection`, `from_spelling` selection
//! metadata) is RADIX-ARTIFACT+FABER-BUILD and re-points to the Radix
//! artifact contract + Faber build configuration (S8A).

/// A native device backend admitted by the product host.
///
/// The accepted machines are Apple Metal (M5 Max, burgus) and NVIDIA CUDA
/// (RTX 5070, pharos); these are the only backends the campaign productizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceBackend {
    /// Apple Metal (MSL modules; macOS-only).
    Metal,
    /// NVIDIA CUDA Driver API (PTX modules).
    Cuda,
}

impl DeviceBackend {
    /// Stable diagnostic spelling (`"metal"` / `"cuda"`). Used in control
    /// frames and structured error diagnostics.
    #[must_use]
    pub fn spelling(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }

    /// Parse a backend from its stable spelling.
    #[must_use]
    pub fn from_spelling(spelling: &str) -> Option<Self> {
        match spelling {
            "metal" => Some(Self::Metal),
            "cuda" => Some(Self::Cuda),
            _ => None,
        }
    }
}

impl std::fmt::Display for DeviceBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.spelling())
    }
}
