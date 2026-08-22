//! CUDA host driver binding and descriptor launch adapter.
//!
//! The crate is intentionally independent of any host product so it can build
//! on Linux. Backend-neutral descriptors, lifecycle registries, and kernel
//! error carriers come from [`host_device_core`].

pub use host_device_core::device_descriptor;
pub use host_device_core::device_registry;
pub use host_device_core::kernel;

pub mod cuda_host;
pub mod cuda_launch_adapter;

// Re-export the complete backend surfaces so product aliases preserve the
// old `crate::cuda_host::...` and `crate::cuda_launch_adapter::...` paths.
pub use cuda_host::*;
pub use cuda_launch_adapter::*;
