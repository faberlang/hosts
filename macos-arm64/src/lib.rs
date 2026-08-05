//! macOS arm64 host runtime primitives for Faber.
//!
//! This crate is the first Faber-owned proof of the host syscall model. It keeps
//! the frame/router/kernel shape local to the macOS host until a second host or
//! concrete duplication justifies extraction. The model is adapted from Muninn's
//! frame and kernel semantics, but this crate intentionally has no Muninn
//! runtime dependency.

pub mod component;
pub mod composite_host;
pub mod cuda_host;
pub mod device_descriptor;
pub mod device_host;
pub mod device_registry;
pub mod kernel;
pub mod manifest;
pub mod metal_host;
pub mod syscall_import;
pub mod wasm;

pub use cuda_host::{
    probe_cuda_environment, CudaEnvReport, CudaHandleId, CudaHostSession, FakeCudaDriver,
    E_CUDA_INVALID_HANDLE, E_CUDA_UNAVAILABLE, E_CUDA_UNSUPPORTED,
};
pub use kernel::{Conversation, Direction, Frame, HostError, HostKernel, Status};
pub use manifest::{CapabilityManifest, RegisteredProvider, SyscallManifest};
pub use metal_host::{
    probe_metal_environment, FakeMetalDriver, MetalEnvReport, MetalHandleId, MetalHostSession,
    E_METAL_INVALID_HANDLE, E_METAL_UNAVAILABLE, E_METAL_UNSUPPORTED,
};
