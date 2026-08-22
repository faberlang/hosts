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

pub use cuda_host::{
    discover_cuda_snapshot, enumerate_cuda_physical_devices, probe_cuda_environment, CudaEnvReport,
    CudaHandleId, CudaHostSession, CudaPhysicalDevice, FakeCudaDriver, E_CUDA_INVALID_HANDLE,
    E_CUDA_UNAVAILABLE, E_CUDA_UNSUPPORTED,
};
pub use cuda_launch_adapter::{
    launch_descriptor, parse_descriptor, AdapterBufferRole, AdapterLaunchReceipt, NumericOracle,
    NvvmElementType, NvvmLaunchBuffer, NvvmLaunchPlan, OracleCheck, NVVM_DESCRIPTOR_SCHEMA_VERSION,
    NVVM_DESCRIPTOR_TARGET,
};
