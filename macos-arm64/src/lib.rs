//! macOS arm64 host runtime primitives for Faber.
//!
//! This crate is the first Faber-owned proof of the host syscall model. It keeps
//! the frame/router/kernel shape local to the macOS host until a second host or
//! concrete duplication justifies extraction. The model is adapted from Muninn's
//! frame and kernel semantics, but this crate intentionally has no Muninn
//! runtime dependency.

pub mod component;
pub mod composite_host;
pub use host_cuda as cuda_host;
pub use host_cuda as cuda_launch_adapter;
pub use host_device_core::device_descriptor;
pub mod device_execute;
pub mod device_host;
pub use host_device_core::device_registry;
pub mod device_runtime_set;
pub mod kernel;
pub mod manifest;
pub mod metal_host;
pub mod syscall_import;
pub mod transaction_backend;
pub mod wasm;

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
pub use device_runtime_set::DeviceRuntimeSet;
pub use kernel::{Conversation, Direction, Frame, HostError, HostKernel, Status};
pub use manifest::{CapabilityManifest, RegisteredProvider, SyscallManifest};
pub use metal_host::{
    discover_metal_snapshot, enumerate_metal_physical_devices, probe_metal_environment,
    FakeMetalDriver, MetalEnvReport, MetalHandleId, MetalHostSession, MetalPhysicalDevice,
    E_METAL_INVALID_HANDLE, E_METAL_UNAVAILABLE, E_METAL_UNSUPPORTED,
};
pub use transaction_backend::{DeviceRuntimeBackend, LaunchProgram};
