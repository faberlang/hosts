//! Typed runtime device descriptor + fail-before-launch validation (S1-4).
//!
//! The frozen N1.4 error surface says bad descriptors, ABI mismatches, entry
//! mismatches, and dtype/shape mismatches must **fail before launch** with
//! typed diagnostics. This module owns that surface for the composite host:
//!
//! - [`DeviceDescriptor`] is the host's typed contract for one device program
//!   (backend, module image, ordered kernels with typed buffer slots and
//!   launch shapes). It mirrors the typed facts of the radix S1-1
//!   `DeviceProgram` schema; the faber package layer maps the canonical
//!   payload onto this host descriptor (hosts never infer bindings or shapes
//!   from emitted text — A3).
//! - [`DeviceDescriptor::validate`] enforces the consistency rules **before
//!   any launch**: structural validity (`E_DEVICE_DESCRIPTOR`), binding and
//!   role consistency (`E_DEVICE_ABI_MISMATCH`), cross-reference shape
//!   conflicts (`E_DEVICE_SHAPE_MISMATCH`), and cross-reference dtype
//!   conflicts (`E_DEVICE_DTYPE_MISMATCH`).
//!
//! The stable codes below are the host-side half of the N1.4 error table.
//! Backend-availability failures (`E_BACKEND_UNAVAILABLE`) and the no-device-
//! program refusal (`E_NO_DEVICE_PROGRAM`) live in [`crate::composite_host`]
//! (they are host-construction failures, not descriptor failures).

use faber::device::DeviceBackend;

use crate::kernel::{HostError, HostResult};

/// Stable host error code for a structurally bad or missing device descriptor.
pub const E_DEVICE_DESCRIPTOR: &str = "E_DEVICE_DESCRIPTOR";
/// Stable host error code for a descriptor ABI inconsistency (binding or role
/// conflicts across the kernel slots).
pub const E_DEVICE_ABI_MISMATCH: &str = "E_DEVICE_ABI_MISMATCH";
/// Stable host error code for a kernel entry that the loaded module does not
/// declare.
pub const E_DEVICE_ENTRY_MISMATCH: &str = "E_DEVICE_ENTRY_MISMATCH";
/// Stable host error code for a shape conflict (a buffer referenced with two
/// different shapes, or a launch binding a buffer of the wrong size).
pub const E_DEVICE_SHAPE_MISMATCH: &str = "E_DEVICE_SHAPE_MISMATCH";
/// Stable host error code for a dtype conflict (a buffer referenced with two
/// different element types).
pub const E_DEVICE_DTYPE_MISMATCH: &str = "E_DEVICE_DTYPE_MISMATCH";
/// Stable host error code for a requested backend the machine cannot admit.
/// Never a CPU fallback: an explicit GPU request that cannot be served fails
/// closed before launch (N1.1/N1.4).
pub const E_BACKEND_UNAVAILABLE: &str = "E_BACKEND_UNAVAILABLE";
/// Stable host error code for an explicit device request on a package/route
/// that carries no device program (N1.4: "package has no device program").
pub const E_NO_DEVICE_PROGRAM: &str = "E_NO_DEVICE_PROGRAM";

/// Host-side element data type of a device buffer slot.
///
/// A small typed set mirroring the emitted kernel ABI; hosts never infer an
/// element type from module text (A3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceDataType {
    /// IEEE 754 single precision.
    F32,
    /// IEEE 754 double precision.
    F64,
    /// Signed 32-bit integer.
    I32,
    /// Signed 64-bit integer.
    I64,
    /// Unsigned 8-bit integer.
    U8,
}

impl DeviceDataType {
    /// Stable diagnostic spelling.
    #[must_use]
    pub fn spelling(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
        }
    }

    /// Parse a data type from its stable spelling.
    #[must_use]
    pub fn from_spelling(spelling: &str) -> Option<Self> {
        match spelling {
            "f32" => Some(Self::F32),
            "f64" => Some(Self::F64),
            "i32" => Some(Self::I32),
            "i64" => Some(Self::I64),
            "u8" => Some(Self::U8),
            _ => None,
        }
    }

    /// Byte width of one element.
    #[must_use]
    pub fn byte_width(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
            Self::U8 => 1,
        }
    }
}

/// Slot role of a device buffer at a kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceBufferRole {
    /// Host-provided input; read-only on the device.
    Input,
    /// Device-produced output; read back at an observation point.
    Output,
    /// Device-resident intermediate, written and read across kernels.
    InOut,
}

impl DeviceBufferRole {
    /// Stable diagnostic spelling.
    #[must_use]
    pub fn spelling(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Output => "output",
            Self::InOut => "in-out",
        }
    }
}

/// One typed buffer slot of a kernel.
///
/// `buffer_id`/`buffer_name` are the program-level buffer identity repeated
/// across kernels; validation requires every reference to the same id to
/// agree on dtype and shape (a shape change must be a new version — the S1-1
/// contract; hosts reject in-place reinterpretation at the descriptor).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DescriptorBuffer {
    /// Program-level buffer identity key.
    pub buffer_id: u32,
    /// Logical name for diagnostics.
    pub buffer_name: String,
    /// Slot role at this kernel.
    pub role: DeviceBufferRole,
    /// Target-neutral binding index (backends map it to their binding syntax).
    pub binding: u32,
    /// Element type of this buffer version.
    pub element_ty: DeviceDataType,
    /// Element count of this buffer version.
    pub element_count: u64,
}

impl DescriptorBuffer {
    /// The byte length this slot expects on the device.
    #[must_use]
    pub fn byte_length(&self) -> u64 {
        self.element_count * u64::from(self.element_ty.byte_width() as u32)
    }
}

/// One kernel of a descriptor: entry, typed slots, and launch shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DescriptorKernel {
    /// Target-neutral logical entry name (backends map it to their syntax).
    pub entry: String,
    /// Typed buffer slots bound by this kernel.
    pub buffers: Vec<DescriptorBuffer>,
    /// 3D dispatch grid (workgroup count per axis).
    pub grid: [u32; 3],
    /// 3D block (threadgroup) shape per axis.
    pub block: [u32; 3],
}

/// Typed runtime device descriptor: the host's contract for one device
/// program (module image + ordered kernels). The faber package layer maps the
/// canonical S1-1 payload onto this shape; hosts validate it before launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceDescriptor {
    /// Backend the descriptor targets.
    pub backend: DeviceBackend,
    /// Compiled module image (MSL source for Metal, PTX for CUDA).
    pub module_image: Vec<u8>,
    /// Ordered kernel declarations.
    pub kernels: Vec<DescriptorKernel>,
}

/// FNV-1a 64-bit provenance hash (the campaign's per-blob provenance
/// convention, N1.3 §3.4; radix-mir-fmir uses the same construction).
///
/// # Panics
/// Never panics.
#[must_use]
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

impl DeviceDescriptor {
    /// Validate the descriptor's consistency rules **before any launch**.
    ///
    /// Checks, in order:
    /// 1. structural validity: a non-empty module image, at least one kernel,
    ///    a non-empty entry per kernel, at least one slot per kernel, a
    ///    non-zero grid/block per kernel, a non-zero element count per slot —
    ///    failures are `E_DEVICE_DESCRIPTOR`;
    /// 2. ABI consistency per kernel: unique bindings and role consistency
    ///    across references to the same buffer id — `E_DEVICE_ABI_MISMATCH`;
    /// 3. cross-reference dtype/shape consistency: every reference to the
    ///    same buffer id carries the same element type (`E_DEVICE_DTYPE_MISMATCH`)
    ///    and the same element count (`E_DEVICE_SHAPE_MISMATCH`) — a shape
    ///    change must be a new version, never an in-place reinterpretation.
    ///
    /// # Errors
    /// Returns the first typed [`HostError`] the descriptor violates.
    pub fn validate(&self) -> HostResult<()> {
        if self.module_image.is_empty() {
            return Err(descriptor_error(
                "device descriptor carries an empty module image",
            ));
        }
        if self.kernels.is_empty() {
            return Err(descriptor_error("device descriptor declares no kernels"));
        }

        let mut identities: Vec<(u32, String, DeviceBufferRole)> = Vec::new();
        let mut shapes: Vec<(u32, DeviceDataType, u64)> = Vec::new();

        for kernel in &self.kernels {
            if kernel.entry.trim().is_empty() {
                return Err(descriptor_error(
                    "device descriptor has a kernel with an empty entry name",
                ));
            }
            if kernel.buffers.is_empty() {
                return Err(descriptor_error(format!(
                    "device descriptor kernel `{}` binds no buffers",
                    kernel.entry
                )));
            }
            if kernel.grid.contains(&0) || kernel.block.contains(&0) {
                return Err(descriptor_error(format!(
                    "device descriptor kernel `{}` has a zero grid or block axis",
                    kernel.entry
                )));
            }

            let mut seen_bindings: Vec<u32> = Vec::new();
            for slot in &kernel.buffers {
                if slot.element_count == 0 {
                    return Err(descriptor_error(format!(
                        "device descriptor kernel `{}` binds a zero-count buffer `{}`",
                        kernel.entry, slot.buffer_name
                    )));
                }
                if seen_bindings.contains(&slot.binding) {
                    return Err(abi_error(format!(
                        "device descriptor kernel `{}` binds index {} more than once",
                        kernel.entry, slot.binding
                    )));
                }
                seen_bindings.push(slot.binding);

                if let Some((_, name, role)) =
                    identities.iter().find(|(id, _, _)| *id == slot.buffer_id)
                {
                    if role_conflict(*role, slot.role) {
                        return Err(abi_error(format!(
                            "device buffer `{}` (id {}) is referenced with conflicting roles {} and {}",
                            slot.buffer_name,
                            slot.buffer_id,
                            role.spelling(),
                            slot.role.spelling()
                        )));
                    }
                    if *name != slot.buffer_name {
                        return Err(abi_error(format!(
                            "device buffer id {} is referenced with conflicting names `{}` and `{}`",
                            slot.buffer_id, name, slot.buffer_name
                        )));
                    }
                } else {
                    identities.push((slot.buffer_id, slot.buffer_name.clone(), slot.role));
                }

                if let Some((_, first_ty, first_count)) =
                    shapes.iter().find(|(id, _, _)| *id == slot.buffer_id)
                {
                    if *first_ty != slot.element_ty {
                        return Err(dtype_error(format!(
                            "device buffer `{}` (id {}) is referenced with conflicting element types {} and {}",
                            slot.buffer_name,
                            slot.buffer_id,
                            first_ty.spelling(),
                            slot.element_ty.spelling()
                        )));
                    }
                    if *first_count != slot.element_count {
                        return Err(shape_error(format!(
                            "device buffer `{}` (id {}) is referenced with conflicting element counts {} and {} (a shape change must be a new version)",
                            slot.buffer_name,
                            slot.buffer_id,
                            first_count,
                            slot.element_count
                        )));
                    }
                } else {
                    shapes.push((slot.buffer_id, slot.element_ty, slot.element_count));
                }
            }
        }
        Ok(())
    }
}

/// Whether two slot roles contradict at a shared buffer identity.
///
/// `InOut` composes with anything; a direct `Input`/`Output` split across two
/// kernels is a program-level role conflict (one buffer cannot be read-only
/// host input and device-produced output at the same time).
fn role_conflict(first: DeviceBufferRole, second: DeviceBufferRole) -> bool {
    matches!(
        (first, second),
        (DeviceBufferRole::Input, DeviceBufferRole::Output)
            | (DeviceBufferRole::Output, DeviceBufferRole::Input)
    )
}

fn descriptor_error(message: impl Into<String>) -> HostError {
    HostError {
        code: E_DEVICE_DESCRIPTOR.to_owned(),
        message: message.into(),
        retryable: false,
    }
}

fn abi_error(message: impl Into<String>) -> HostError {
    HostError {
        code: E_DEVICE_ABI_MISMATCH.to_owned(),
        message: message.into(),
        retryable: false,
    }
}

fn shape_error(message: impl Into<String>) -> HostError {
    HostError {
        code: E_DEVICE_SHAPE_MISMATCH.to_owned(),
        message: message.into(),
        retryable: false,
    }
}

fn dtype_error(message: impl Into<String>) -> HostError {
    HostError {
        code: E_DEVICE_DTYPE_MISMATCH.to_owned(),
        message: message.into(),
        retryable: false,
    }
}

/// Stable error constructors shared by the composite host and tests.
pub(crate) mod errors {
    use super::{HostError, E_BACKEND_UNAVAILABLE, E_NO_DEVICE_PROGRAM};

    pub(crate) fn backend_unavailable(message: impl Into<String>) -> HostError {
        HostError {
            code: E_BACKEND_UNAVAILABLE.to_owned(),
            message: message.into(),
            retryable: false,
        }
    }

    pub(crate) fn no_device_program(message: impl Into<String>) -> HostError {
        HostError {
            code: E_NO_DEVICE_PROGRAM.to_owned(),
            message: message.into(),
            retryable: false,
        }
    }

    pub(crate) fn shape_mismatch(message: impl Into<String>) -> HostError {
        HostError {
            code: super::E_DEVICE_SHAPE_MISMATCH.to_owned(),
            message: message.into(),
            retryable: false,
        }
    }
}
