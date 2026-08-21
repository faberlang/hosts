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
//! - [`KvCacheDescriptor`] is the KV storage/binding plan: allocation
//!   capacity and view extent are separate typed facts, multiple views may
//!   share one allocation, launch bindings preserve declared indices and
//!   order, and runtime cursor values never join the graph hash.
//!
//! The stable codes below are the host-side half of the N1.4 error table.
//! Backend-availability failures (`E_BACKEND_UNAVAILABLE`) and the no-device-
//! program refusal (`E_NO_DEVICE_PROGRAM`) live in [`crate::composite_host`]
//! (they are host-construction failures, not descriptor failures).

use std::collections::BTreeMap;

use host_coordinator::DeviceBackend;

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
///
/// Placement-ABI `dtype: u32` (`__faber_gpu_v1_copy_in`) is the
/// `MirScalarLayout` discriminant. Radix `placement-debt-audit` F2 owns that
/// integer; this enum coordinates the host names onto it and does not assign
/// a second numbering. F2 has not yet given `MirScalarLayout` `#[repr(u32)]`,
/// so the numbers below are that enum's declaration-order discriminants:
///
/// | `dtype` | `MirScalarLayout` | [`DeviceDataType`] |
/// | ------- | ----------------- | ------------------ |
/// | 3       | `I32`             | [`Self::I32`]      |
/// | 4       | `I64`             | [`Self::I64`]      |
/// | 6       | `U8`              | [`Self::U8`]       |
/// | 10      | `F16`             | [`Self::F16`]      |
/// | 11      | `F32`             | [`Self::F32`]      |
/// | 12      | `F64`             | [`Self::F64`]      |
///
/// [`Self::BF16`] has no F2 slot yet (`MirScalarLayout` has no BF16 variant).
/// Decode via [`Self::from_placement_discriminant`].
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
    /// IEEE 754 binary16.
    F16,
    /// Brain floating point, 16-bit.
    BF16,
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
            Self::F16 => "f16",
            Self::BF16 => "bf16",
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
            "f16" => Some(Self::F16),
            "bf16" => Some(Self::BF16),
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
            Self::F16 | Self::BF16 => 2,
        }
    }

    /// Decode a placement-ABI `dtype: u32` (`MirScalarLayout` discriminant).
    ///
    /// Radix `placement-debt-audit` F2 owns the integer. Unmapped
    /// discriminants (Bool/I8/I16/I128/U16/U32/U64, and any future slot)
    /// return `None`.
    #[must_use]
    pub fn from_placement_discriminant(discriminant: u32) -> Option<Self> {
        match discriminant {
            3 => Some(Self::I32),
            4 => Some(Self::I64),
            6 => Some(Self::U8),
            10 => Some(Self::F16),
            11 => Some(Self::F32),
            12 => Some(Self::F64),
            _ => None,
        }
    }

    /// Placement-ABI `dtype: u32` for this host type, when F2 names one.
    #[must_use]
    pub fn placement_discriminant(self) -> Option<u32> {
        match self {
            Self::I32 => Some(3),
            Self::I64 => Some(4),
            Self::U8 => Some(6),
            Self::F16 => Some(10),
            Self::F32 => Some(11),
            Self::F64 => Some(12),
            Self::BF16 => None,
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

    /// Parse a role from its stable spelling.
    #[must_use]
    pub fn from_spelling(spelling: &str) -> Option<Self> {
        match spelling {
            "input" => Some(Self::Input),
            "output" => Some(Self::Output),
            "in-out" => Some(Self::InOut),
            _ => None,
        }
    }
}

/// How long a buffer's storage lives in the device program (S2-4).
///
/// Mirrors the radix S1-1 [`BufferLifetime`]: **per-program** buffers are
/// allocated once at session creation and released at program end;
/// **per-step** buffers are live within one step and recycled at the step
/// boundary; an **observation point** buffer is read back at a declared
/// observation point and then released (read-then-release). The host session
/// consumes these typed facts — it never derives a lifetime from slot role
/// alone (that would be coincidence, council 3).
///
/// [`BufferLifetime`]: https://docs.rs/radix-mir/latest/radix_mir/device_program/enum.BufferLifetime.html
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceBufferLifetime {
    /// Allocated once for the whole program; persists across executions.
    PerProgram,
    /// Live within one step; recycled at the step boundary.
    PerStep,
    /// Read back at a declared observation point; read-then-release.
    ObservationPoint,
}

impl DeviceBufferLifetime {
    /// Stable diagnostic spelling.
    #[must_use]
    pub fn spelling(self) -> &'static str {
        match self {
            Self::PerProgram => "per-program",
            Self::PerStep => "per-step",
            Self::ObservationPoint => "observation-point",
        }
    }

    /// Parse a lifetime from its stable spelling.
    #[must_use]
    pub fn from_spelling(spelling: &str) -> Option<Self> {
        match spelling {
            "per-program" => Some(Self::PerProgram),
            "per-step" => Some(Self::PerStep),
            "observation-point" => Some(Self::ObservationPoint),
            _ => None,
        }
    }
}

/// Independent initialization axis (F5): how a buffer's storage is brought
/// to its first defined state. Mirrors the wire's
/// [`WireInitializationPolicy`]: **zero-fill** storage is zeroed at
/// allocation (persistent accumulation state, optimizer state); **host-
/// provided** storage is uploaded from host inputs; **kernel-initialized**
/// storage is fully defined by a device kernel before any read.
///
/// The host honors this fact at allocation — it never re-derives an
/// initialization policy from role or lifetime (that would couple the F5
/// axes).
///
/// [`WireInitializationPolicy`]: https://docs.rs/radix-mir-fmir/latest/radix_mir_fmir/enum.WireInitializationPolicy.html
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DeviceBufferInitialization {
    /// Storage is zero-filled at allocation.
    #[default]
    ZeroFill,
    /// Storage is uploaded from host-provided values.
    HostProvided,
    /// Storage is fully written by a device kernel before any read.
    KernelInitialized,
}

impl DeviceBufferInitialization {
    /// Stable diagnostic spelling.
    #[must_use]
    pub fn spelling(self) -> &'static str {
        match self {
            Self::ZeroFill => "zero-fill",
            Self::HostProvided => "host-provided",
            Self::KernelInitialized => "kernel-initialized",
        }
    }

    /// Parse an initialization policy from its stable spelling.
    #[must_use]
    pub fn from_spelling(spelling: &str) -> Option<Self> {
        match spelling {
            "zero-fill" => Some(Self::ZeroFill),
            "host-provided" => Some(Self::HostProvided),
            "kernel-initialized" => Some(Self::KernelInitialized),
            _ => None,
        }
    }
}

/// Program execution-lifetime regime (S2-4), mirroring the radix S1-1
/// [`DeviceProgramLifetime`]: whether the program runs once (a one-shot-with-
/// repeat surface for the leak proof) or repeats as a training step (when
/// per-step recycling between executions becomes meaningful).
///
/// [`DeviceProgramLifetime`]: https://docs.rs/radix-mir/latest/radix_mir/device_program/enum.DeviceProgramLifetime.html
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DeviceProgramLifetime {
    /// One-shot program run.
    #[default]
    SingleRun,
    /// Repeating training step; per-step buffers recycle between iterations.
    RepeatingStep,
}

impl DeviceProgramLifetime {
    /// Stable diagnostic spelling.
    #[must_use]
    pub fn spelling(self) -> &'static str {
        match self {
            Self::SingleRun => "single-run",
            Self::RepeatingStep => "repeating-step",
        }
    }

    /// Parse a program lifetime from its stable spelling.
    #[must_use]
    pub fn from_spelling(spelling: &str) -> Option<Self> {
        match spelling {
            "single-run" => Some(Self::SingleRun),
            "repeating-step" => Some(Self::RepeatingStep),
            _ => None,
        }
    }
}

/// One typed buffer slot of a kernel.
///
/// `buffer_id`/`buffer_name` are the program-level buffer identity repeated
/// across kernels; validation requires every reference to the same id to
/// agree on identity and lifetime, while every `(buffer_id, version)` pair
/// agrees on dtype and shape. A shape change is a new version — the S1-1
/// contract; hosts reject in-place reinterpretation at the descriptor.
///
/// `semantic_value` is the stable **semantic value identity** the buffer
/// holds (F1): the wire's carried value fact — never derived from names,
/// shapes, binding positions, or declaration coincidence. Two unrelated
/// same-name/same-shape values are distinct; validation requires the same
/// buffer id to always carry the same semantic value and two different
/// buffer ids never to alias one value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DescriptorBuffer {
    /// Program-level buffer identity key.
    pub buffer_id: u32,
    /// Logical name for diagnostics.
    pub buffer_name: String,
    /// The stable semantic value identity this buffer holds (F1).
    pub semantic_value: u32,
    /// Slot role at this kernel.
    pub role: DeviceBufferRole,
    /// How long this buffer's storage lives (S2-4; consumed by the session's
    /// lifetime-distinct allocation/release policy).
    pub lifetime: DeviceBufferLifetime,
    /// Independent initialization axis (F5): how this buffer's storage is
    /// brought to its first defined state — the wire's carried policy,
    /// projected verbatim. The host zero-fills `ZeroFill` buffers at
    /// allocation; never re-derived from role or lifetime.
    pub initialization: DeviceBufferInitialization,
    /// Target-neutral binding index (backends map it to their binding syntax).
    pub binding: u32,
    /// Element type of this buffer version.
    pub element_ty: DeviceDataType,
    /// Element count of this buffer version.
    pub element_count: u64,
    /// Content version of this buffer shape (the wire's carried
    /// `BufferVersion.version` — R2: the host consumes the version fact; it
    /// never re-derives or hardcodes `1`).
    pub version: u32,
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

/// One entry in the ordered launch sequence.
///
/// The launch id is the program identity used by carried data-flow edges;
/// `kernel_index` names the declaration to launch. A kernel declaration may
/// therefore occur more than once in the sequence without duplicating or
/// reinterpreting its typed facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorLaunch {
    /// Program-unique launch identity.
    pub id: u32,
    /// Index into [`DeviceDescriptor::kernels`].
    pub kernel_index: u32,
}

/// One declared observation point (F6): an explicit result row projected
/// from the wire's carried observation fact.
///
/// A result is read back at its producing launch's completion boundary —
/// writable intermediates and persistent state are results only through
/// this declared fact, never because they are writable. Validation admits
/// results only for buffers whose lifetime is [`DeviceBufferLifetime::ObservationPoint`],
/// so the constructor and the host admission agree on one readback rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorResult {
    /// The observed buffer's program-level identity.
    pub buffer_id: u32,
    /// The observed content version.
    pub version: u32,
    /// The launch that produced the observed version.
    pub produced_by: u32,
    /// The launch whose completion boundary makes the observation valid
    /// (F5/F6); the wire carries it as the explicit observation fact.
    pub at_launch: u32,
}

/// One declared end-of-run observation (S5A-U1): a buffer whose FINAL value
/// is read back exactly once at the declared completion boundary — after the
/// step loop of a `RepeatingStep` session — and returned to the caller.
///
/// Distinct from [`DescriptorResult`] (the per-step observations, read back
/// every step): an end-of-run observation may name a **`PerStep`** buffer (the
/// final forward activations, the final gradients) or a **`PerProgram`**
/// buffer (the final trainable params). The params MUST stay `PerProgram` —
/// once-init persistence across steps — so their only readback is this
/// one-shot end-of-run readback; they are never read within a step. The
/// descriptor's validation admits only these two lifetime classes (an
/// [`DeviceBufferLifetime::ObservationPoint`] buffer is a per-step result
/// and is never read both per step and at the end). The set is the wire's
/// DECLARED `EndOfRun` cadence set, carried verbatim by the descriptor —
/// the session reads it back via
/// [`ProgramSession::read_end_of_run`](crate::composite_host::ProgramSession::read_end_of_run)
/// exactly once after the final step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorEndOfRunResult {
    /// The observed buffer's program-level identity.
    pub buffer_id: u32,
    /// The observed content version.
    pub version: u32,
}

/// One version-keyed buffer shape carried by the device-program wire.
///
/// The key is `(buffer_id, version)`, not just `buffer_id`: a buffer identity
/// can carry multiple shape snapshots over a complete program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorBufferVersion {
    /// Program-level buffer identity key.
    pub buffer_id: u32,
    /// Content version within the buffer identity.
    pub version: u32,
    /// Element type of this version's shape.
    pub element_ty: DeviceDataType,
    /// Element count of this version's shape.
    pub element_count: u64,
}

/// One persistent allocation: buffer identity, dtype, capacity bytes,
/// lifetime, and initialization. Capacity is a storage fact and is never a
/// view extent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DescriptorAllocation {
    /// Buffer identity of this allocation (the launch-binding handle).
    pub buffer_id: u32,
    /// Element type of the backing store.
    pub dtype: DeviceDataType,
    /// Fixed capacity in bytes. Distinct from any view's [`DescriptorView::maximum_span`].
    pub capacity_bytes: u64,
    /// How long this allocation lives.
    pub lifetime: DeviceBufferLifetime,
    /// How the allocation is brought to its first defined state.
    pub initialization: DeviceBufferInitialization,
}

/// A bounded view over an allocation. Multiple views may share one
/// allocation (append and prefix over a persistent K or V arena).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DescriptorView {
    /// Allocation this view addresses ([`DescriptorAllocation::buffer_id`]).
    pub allocation_id: u32,
    /// Logical dimensions in axis order (layer, KV-head, position, dim).
    pub logical_dims: Vec<u64>,
    /// Element strides matching [`Self::logical_dims`] rank.
    pub strides: Vec<u64>,
    /// Static base offset in bytes from the allocation start.
    pub static_base: u64,
    /// Maximum span in bytes this view may address. Distinct from allocation
    /// capacity.
    pub maximum_span: u64,
}

/// One typed invocation-state upload. Current cursor values never join
/// static program identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DescriptorInvocationState {
    /// Write position (absolute cache row).
    pub position: u32,
    /// Valid length after this step.
    pub valid_len_after: u32,
    /// Query rows in this step.
    pub query_rows: u32,
    /// Sequence epoch; advances on logical reset.
    pub sequence_epoch: u32,
}

/// Tagged source of a launch binding's dynamic offset or span. Typed tags
/// only — never a magic uniform bit pattern or string sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DescriptorRuntimeSource {
    /// Offset and span are the constant descriptor facts.
    Constant,
    /// Offset or span is produced from [`DescriptorInvocationState::position`].
    Position,
    /// Offset or span is produced from [`DescriptorInvocationState::valid_len_after`].
    ValidLenAfter,
    /// Offset or span is produced from [`DescriptorInvocationState::query_rows`].
    QueryRows,
    /// Offset or span is produced from [`DescriptorInvocationState::sequence_epoch`].
    SequenceEpoch,
}

/// One launch binding: declared binding index is preserved through to the
/// launch record. `handle` names the allocation; `byte_offset` / `view_span`
/// are the static envelope (runtime sources supply the live cursor at launch).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DescriptorLaunchBinding {
    /// Allocation handle ([`DescriptorAllocation::buffer_id`]).
    pub handle: u32,
    /// Declared binding index. Never dropped before launch.
    pub binding_index: u32,
    /// Byte offset into the allocation (static base, or the static envelope
    /// when [`Self::runtime_source`] is a cursor tag).
    pub byte_offset: u64,
    /// View span in bytes for this binding.
    pub view_span: u64,
    /// Whether offset/span are constant or sourced from invocation state.
    pub runtime_source: DescriptorRuntimeSource,
}

/// KV storage and binding plan: two persistent arenas expose bounded views
/// without copying, and one typed invocation-state upload carries the live
/// cursor. Lives beside [`DeviceDescriptor`] so legacy whole-buffer
/// descriptors keep their struct shape.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KvCacheDescriptor {
    /// Persistent allocations (K and V arenas). Capacity is a storage fact.
    pub allocations: Vec<DescriptorAllocation>,
    /// Bounded views over those allocations. Multiple views may share one
    /// allocation.
    pub views: Vec<DescriptorView>,
    /// Current cursor. Deliberately excluded from [`Self::program_graph_hash`].
    pub invocation_state: DescriptorInvocationState,
    /// Ordered launch bindings. Declaration order is the launch-record order.
    pub launch_bindings: Vec<DescriptorLaunchBinding>,
}

/// Typed runtime device descriptor: the host's contract for one device
/// program (module image + kernel declarations + ordered launches +
/// version-keyed resource metadata + program lifetime + carried graph and
/// observation facts). The faber package layer maps the canonical S1-1
/// payload onto this shape; hosts validate it before launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceDescriptor {
    /// Backend the descriptor targets.
    pub backend: DeviceBackend,
    /// Compiled module image (MSL source for Metal, PTX for CUDA).
    pub module_image: Vec<u8>,
    /// Ordered kernel declarations.
    pub kernels: Vec<DescriptorKernel>,
    /// Ordered launch records. The host must not substitute kernel declaration
    /// order for this sequence.
    pub launches: Vec<DescriptorLaunch>,
    /// Complete version-keyed resource shape facts carried by the wire.
    pub buffer_versions: Vec<DescriptorBufferVersion>,
    /// Program execution-lifetime regime (S2-4): the session consumes it as
    /// the declared program fact (single-run one-shot-with-repeat surface for
    /// the leak proof vs a repeating training step).
    pub program_lifetime: DeviceProgramLifetime,
    /// Carried inter-kernel data-flow edges (A10/R2): the wire's
    /// producer/consumer facts per buffer version. The session consumes them
    /// for the declared resource graph and the host schedules the validated
    /// graph — it never re-derives topology from launch order or a
    /// first-writer coincidence rule.
    pub data_flow: Vec<DescriptorDataFlow>,
    /// Declared legal execution roots (F3): the launches the graph may start
    /// from. Never inferred from kernel declaration order. Validation proves
    /// every launch is reachable from a root.
    pub roots: Vec<u32>,
    /// Declared per-step observation points (F6 + S5A-U1): the explicit
    /// result rows the host reads back and releases within every step. Only
    /// these buffers are observable per step — each is an
    /// `ObservationPoint`-lifetime buffer (the loss).
    pub results: Vec<DescriptorResult>,
    /// Declared end-of-run observations (S5A-U1): the result rows the host
    /// reads back exactly ONCE after the step loop of a `RepeatingStep`
    /// session — the final forward, final gradients, final params. Each is a
    /// `PerStep` or `PerProgram` buffer (never read per step, never read-only
    /// input state). This is the wire's declared `EndOfRun` cadence set,
    /// carried by the descriptor — the host never derives it and there is no
    /// runtime declaration seam.
    pub end_of_run_results: Vec<DescriptorEndOfRunResult>,
}

/// One carried inter-kernel data-flow edge (A10): a buffer content version
/// produced by launch `producer` and consumed by launch `consumer`. Mirrors
/// the radix-mir `BufferRegistry::data_flow_pairs` fact; the host renders it
/// as-is from the descriptor (R2 consume — no re-derivation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorDataFlow {
    /// Buffer whose content flows.
    pub buffer_id: u32,
    /// Content version that flows (the wire's `RegistryVersion.version`).
    pub version: u32,
    /// Producing launch id (1-based).
    pub producer: u32,
    /// Consuming launch id (1-based).
    pub consumer: u32,
}

/// FNV-1a 64-bit **module provenance** hash (the campaign's per-blob
/// provenance convention, N1.3 §3.4; radix-mir-fmir uses the same
/// construction). Names only the loaded backend blob — it is NOT the
/// run/session program-graph identity, which is the distinct SHA-256 domain
/// [`DeviceDescriptor::program_graph_hash`].
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
    ///    at least one launch, non-empty entries, at least one slot per
    ///    kernel, non-zero grid/block axes, and non-zero element counts —
    ///    failures are `E_DEVICE_DESCRIPTOR`;
    /// 2. launch identity: launch ids are non-zero and unique, kernel
    ///    references are in range, and carried edge endpoints name launches —
    ///    failures are `E_DEVICE_DESCRIPTOR`;
    /// 3. version metadata: `(buffer_id, version)` keys are non-zero and
    ///    unique, and each key has one shape; slot facts must agree with the
    ///    keyed metadata — dtype conflicts are `E_DEVICE_DTYPE_MISMATCH` and
    ///    shape conflicts are `E_DEVICE_SHAPE_MISMATCH`;
    /// 4. ABI consistency per kernel: unique bindings and role/name
    ///    consistency across references to the same buffer id —
    ///    `E_DEVICE_ABI_MISMATCH`;
    /// 5. cross-reference lifetime consistency (S2-4): every reference to the
    ///    same buffer id carries the same [`DeviceBufferLifetime`] — a
    ///    lifetime is an identity fact of the buffer, not a per-slot choice
    ///    (`E_DEVICE_ABI_MISMATCH`).
    ///
    /// # Errors
    /// Returns the first typed [`HostError`] the descriptor violates.
    pub fn validate(&self) -> HostResult<()> {
        if self.module_image.is_empty() {
            return Err(errors::descriptor(
                "device descriptor carries an empty module image",
            ));
        }
        if self.kernels.is_empty() {
            return Err(errors::descriptor("device descriptor declares no kernels"));
        }
        if self.launches.is_empty() {
            return Err(errors::descriptor("device descriptor declares no launches"));
        }

        let mut launch_ids: Vec<u32> = Vec::with_capacity(self.launches.len());
        for launch in &self.launches {
            if launch.id == 0 {
                return Err(errors::descriptor(
                    "device descriptor has a launch with the reserved zero identity",
                ));
            }
            if launch_ids.contains(&launch.id) {
                return Err(errors::descriptor(format!(
                    "device descriptor repeats launch identity {}",
                    launch.id
                )));
            }
            if self.kernels.get(launch.kernel_index as usize).is_none() {
                return Err(errors::descriptor(format!(
                    "device descriptor launch {} references unknown kernel index {}",
                    launch.id, launch.kernel_index
                )));
            }
            launch_ids.push(launch.id);
        }

        // Declared legal execution roots (F3): non-zero, unique, real launch
        // ids. The host schedules the validated graph from these facts; an
        // empty root set would leave the schedule unanchored.
        let mut root_ids: Vec<u32> = Vec::with_capacity(self.roots.len());
        for root in &self.roots {
            if *root == 0 {
                return Err(errors::descriptor(
                    "device descriptor has a root with the reserved zero identity",
                ));
            }
            if root_ids.contains(root) {
                return Err(errors::descriptor(format!(
                    "device descriptor repeats legal execution root {root}"
                )));
            }
            if !launch_ids.contains(root) {
                return Err(errors::descriptor(format!(
                    "device descriptor root {root} names an unknown launch"
                )));
            }
            root_ids.push(*root);
        }
        if root_ids.is_empty() {
            return Err(errors::descriptor(
                "device descriptor declares no legal execution roots",
            ));
        }

        let mut versions: Vec<DescriptorBufferVersion> =
            Vec::with_capacity(self.buffer_versions.len());
        for version in &self.buffer_versions {
            if version.version == 0 {
                return Err(errors::descriptor(format!(
                    "device descriptor buffer {} uses the reserved zero version",
                    version.buffer_id
                )));
            }
            if version.element_count == 0 {
                return Err(errors::descriptor(format!(
                    "device descriptor buffer {} version {} has a zero element count",
                    version.buffer_id, version.version
                )));
            }
            if let Some(first) = versions.iter().find(|first| {
                first.buffer_id == version.buffer_id && first.version == version.version
            }) {
                if first.element_ty != version.element_ty {
                    return Err(errors::dtype_mismatch(format!(
                        "device buffer {} version {} carries conflicting element types {} and {}",
                        version.buffer_id,
                        version.version,
                        first.element_ty.spelling(),
                        version.element_ty.spelling()
                    )));
                }
                if first.element_count != version.element_count {
                    return Err(errors::shape_mismatch(format!(
                        "device buffer {} version {} carries conflicting element counts {} and {}",
                        version.buffer_id,
                        version.version,
                        first.element_count,
                        version.element_count
                    )));
                }
                return Err(errors::descriptor(format!(
                    "device descriptor repeats buffer {} version {} metadata",
                    version.buffer_id, version.version
                )));
            }
            versions.push(*version);
        }
        if versions.is_empty() {
            return Err(errors::descriptor(
                "device descriptor declares no version-keyed buffer metadata",
            ));
        }

        for edge in &self.data_flow {
            if edge.version == 0 || edge.producer == 0 || edge.consumer == 0 {
                return Err(errors::descriptor(
                    "device descriptor data-flow edge uses a reserved zero identity",
                ));
            }
            if edge.producer == edge.consumer {
                return Err(errors::descriptor(format!(
                    "device descriptor data-flow edge for buffer {} version {} is self-referential at launch {}",
                    edge.buffer_id, edge.version, edge.producer
                )));
            }
            if !launch_ids.contains(&edge.producer) || !launch_ids.contains(&edge.consumer) {
                return Err(errors::descriptor(format!(
                    "device descriptor data-flow edge for buffer {} version {} references an unknown launch",
                    edge.buffer_id, edge.version
                )));
            }
            if !versions.iter().any(|version| {
                version.buffer_id == edge.buffer_id && version.version == edge.version
            }) {
                return Err(errors::descriptor(format!(
                    "device descriptor data-flow edge references unknown buffer {} version {}",
                    edge.buffer_id, edge.version
                )));
            }
        }

        // Carried graph schedule (F3): the launch sequence is the schedule,
        // and the carried dependency edges must be consistent with it.
        //
        // 1. Single definition per value generation: exactly one launch
        //    produces a given `(buffer, version)`. The wire carries one edge
        //    per (producer, consumer) pair (DescriptorDataFlow mirrors
        //    `BufferRegistry::data_flow_pairs`), so a version consumed by
        //    several launches legitimately repeats the same producer — that
        //    fan-out is not a second definition. Only a DIFFERENT producer
        //    for the same `(buffer, version)` is another writer of the same
        //    generation, which the frozen contract forbids (F2).
        // 2. Topological consistency: the carried launch order must place
        //    every consumer launch after all its producers. A cycle or a
        //    missing/inverted dependency fails validation before launch.
        // 3. Complete schedule: every launch is reachable from a declared
        //    root following the dependency edges forward.
        let mut producers: Vec<((u32, u32), u32)> = Vec::new();
        for edge in &self.data_flow {
            if let Some((_, first_producer)) = producers.iter().find(|((buffer_id, version), _)| {
                *buffer_id == edge.buffer_id && *version == edge.version
            }) {
                if *first_producer == edge.producer {
                    // Fan-out: the same value generation feeds several
                    // consumers, one carried edge per consumer. The producer
                    // is unique — admit the repeated edge.
                    continue;
                }
                return Err(errors::descriptor(format!(
                    "device descriptor defines buffer {} version {} twice (producers {} and {}); one value generation has exactly one producer",
                    edge.buffer_id, edge.version, first_producer, edge.producer
                )));
            }
            producers.push(((edge.buffer_id, edge.version), edge.producer));
        }
        let mut position: BTreeMap<u32, usize> = BTreeMap::new();
        for (index, launch) in self.launches.iter().enumerate() {
            position.insert(launch.id, index);
        }
        for edge in &self.data_flow {
            let producer_at = position[&edge.producer];
            let consumer_at = position[&edge.consumer];
            if producer_at >= consumer_at {
                return Err(errors::descriptor(format!(
                    "device descriptor launch order violates the carried dependency graph: launch {} (producer of buffer {} version {}) is not scheduled before launch {} (its consumer)",
                    edge.producer, edge.buffer_id, edge.version, edge.consumer
                )));
            }
        }
        let mut reachable: Vec<u32> = Vec::with_capacity(self.launches.len());
        let mut stack: Vec<u32> = root_ids.clone();
        while let Some(launch) = stack.pop() {
            if reachable.contains(&launch) {
                continue;
            }
            reachable.push(launch);
            for edge in &self.data_flow {
                if edge.producer == launch && !reachable.contains(&edge.consumer) {
                    stack.push(edge.consumer);
                }
            }
        }
        for launch in &launch_ids {
            if !reachable.contains(launch) {
                return Err(errors::descriptor(format!(
                    "device descriptor launch {launch} is not reachable from any declared root; the carried graph is incomplete"
                )));
            }
        }

        let mut identities: Vec<(u32, String, DeviceBufferRole)> = Vec::new();
        let mut semantic_values: Vec<(u32, u32)> = Vec::new();
        let mut lifetimes: Vec<(u32, DeviceBufferLifetime)> = Vec::new();
        let mut initializations: Vec<(u32, DeviceBufferInitialization)> = Vec::new();
        // Resident steps copy each PerStep input once. A later kernel write
        // to that same buffer would clobber the host value (the old
        // per-kernel copy hid this). Admit fails closed.
        let mut per_step_inputs: Vec<u32> = Vec::new();
        let mut per_step_writes: Vec<u32> = Vec::new();

        for kernel in &self.kernels {
            if kernel.entry.trim().is_empty() {
                return Err(errors::descriptor(
                    "device descriptor has a kernel with an empty entry name",
                ));
            }
            if kernel.buffers.is_empty() {
                return Err(errors::descriptor(format!(
                    "device descriptor kernel `{}` binds no buffers",
                    kernel.entry
                )));
            }
            if kernel.grid.contains(&0) || kernel.block.contains(&0) {
                return Err(errors::descriptor(format!(
                    "device descriptor kernel `{}` has a zero grid or block axis",
                    kernel.entry
                )));
            }

            let mut seen_bindings: Vec<u32> = Vec::new();
            for slot in &kernel.buffers {
                if slot.element_count == 0 {
                    return Err(errors::descriptor(format!(
                        "device descriptor kernel `{}` binds a zero-count buffer `{}`",
                        kernel.entry, slot.buffer_name
                    )));
                }
                if slot.version == 0 {
                    return Err(errors::descriptor(format!(
                        "device descriptor kernel `{}` binds buffer `{}` with the reserved zero version",
                        kernel.entry, slot.buffer_name
                    )));
                }
                if seen_bindings.contains(&slot.binding) {
                    return Err(errors::abi_mismatch(format!(
                        "device descriptor kernel `{}` binds index {} more than once",
                        kernel.entry, slot.binding
                    )));
                }
                seen_bindings.push(slot.binding);

                // F1: the stable semantic value identity. The same buffer id
                // always holds the same value, and two different buffer ids
                // never alias one value (two unrelated same-name/same-shape
                // values are distinct).
                if slot.semantic_value == 0 {
                    return Err(errors::descriptor(format!(
                        "device buffer `{}` (id {}) carries the reserved zero semantic value identity",
                        slot.buffer_name, slot.buffer_id
                    )));
                }
                if let Some((_, first_semantic)) =
                    semantic_values.iter().find(|(id, _)| *id == slot.buffer_id)
                {
                    if *first_semantic != slot.semantic_value {
                        return Err(errors::abi_mismatch(format!(
                            "device buffer `{}` (id {}) is referenced with conflicting semantic value identities {} and {}",
                            slot.buffer_name,
                            slot.buffer_id,
                            first_semantic,
                            slot.semantic_value
                        )));
                    }
                } else {
                    if let Some((_, other_id)) = semantic_values
                        .iter()
                        .find(|(_, value)| *value == slot.semantic_value)
                    {
                        return Err(errors::abi_mismatch(format!(
                            "device buffers `{}` (id {}) and id {} alias the same semantic value {}; each value is held by exactly one buffer",
                            slot.buffer_name, slot.buffer_id, other_id, slot.semantic_value
                        )));
                    }
                    semantic_values.push((slot.buffer_id, slot.semantic_value));
                }

                if let Some((_, name, role)) =
                    identities.iter().find(|(id, _, _)| *id == slot.buffer_id)
                {
                    if role_conflict(*role, slot.role) {
                        return Err(errors::abi_mismatch(format!(
                            "device buffer `{}` (id {}) is referenced with conflicting roles {} and {}",
                            slot.buffer_name,
                            slot.buffer_id,
                            role.spelling(),
                            slot.role.spelling()
                        )));
                    }
                    if *name != slot.buffer_name {
                        return Err(errors::abi_mismatch(format!(
                            "device buffer id {} is referenced with conflicting names `{}` and `{}`",
                            slot.buffer_id, name, slot.buffer_name
                        )));
                    }
                } else {
                    identities.push((slot.buffer_id, slot.buffer_name.clone(), slot.role));
                }

                if slot.lifetime == DeviceBufferLifetime::PerStep {
                    match slot.role {
                        DeviceBufferRole::Input => {
                            if !per_step_inputs.contains(&slot.buffer_id) {
                                per_step_inputs.push(slot.buffer_id);
                            }
                        }
                        DeviceBufferRole::Output | DeviceBufferRole::InOut => {
                            if !per_step_writes.contains(&slot.buffer_id) {
                                per_step_writes.push(slot.buffer_id);
                            }
                        }
                    }
                }

                let Some(version) = versions.iter().find(|version| {
                    version.buffer_id == slot.buffer_id && version.version == slot.version
                }) else {
                    return Err(errors::descriptor(format!(
                        "device buffer `{}` (id {}) version {} has no keyed metadata",
                        slot.buffer_name, slot.buffer_id, slot.version
                    )));
                };
                if version.element_ty != slot.element_ty {
                    return Err(errors::dtype_mismatch(format!(
                        "device buffer `{}` (id {}) version {} is referenced with conflicting element types {} and {}",
                        slot.buffer_name,
                        slot.buffer_id,
                        slot.version,
                        version.element_ty.spelling(),
                        slot.element_ty.spelling()
                    )));
                }
                if version.element_count != slot.element_count {
                    return Err(errors::shape_mismatch(format!(
                        "device buffer `{}` (id {}) version {} is referenced with conflicting element counts {} and {}",
                        slot.buffer_name,
                        slot.buffer_id,
                        slot.version,
                        version.element_count,
                        slot.element_count
                    )));
                }

                // S2-4: a lifetime is a buffer identity fact; two references
                // to the same id must agree on it (the session's per-class
                // allocation/release policy is driven by this single fact).
                if let Some((_, first_lifetime)) =
                    lifetimes.iter().find(|(id, _)| *id == slot.buffer_id)
                {
                    if *first_lifetime != slot.lifetime {
                        return Err(errors::abi_mismatch(format!(
                            "device buffer `{}` (id {}) is referenced with conflicting lifetimes {} and {}",
                            slot.buffer_name,
                            slot.buffer_id,
                            first_lifetime.spelling(),
                            slot.lifetime.spelling()
                        )));
                    }
                } else {
                    lifetimes.push((slot.buffer_id, slot.lifetime));
                }

                // F5: the initialization axis is also a buffer identity fact
                // — two references to the same id must agree on how its
                // storage is brought to its first defined state (the
                // once-init / per-allocation policy is driven by this single
                // fact).
                if let Some((_, first_init)) =
                    initializations.iter().find(|(id, _)| *id == slot.buffer_id)
                {
                    if *first_init != slot.initialization {
                        return Err(errors::abi_mismatch(format!(
                            "device buffer `{}` (id {}) is referenced with conflicting initialization policies {} and {}",
                            slot.buffer_name,
                            slot.buffer_id,
                            first_init.spelling(),
                            slot.initialization.spelling()
                        )));
                    }
                } else {
                    initializations.push((slot.buffer_id, slot.initialization));
                }
            }
        }

        for id in &per_step_inputs {
            if per_step_writes.contains(id) {
                let name = identities
                    .iter()
                    .find(|(buffer_id, _, _)| buffer_id == id)
                    .map(|(_, name, _)| name.as_str())
                    .unwrap_or("<unknown>");
                return Err(errors::descriptor(format!(
                    "device buffer `{name}` (id {id}) is a PerStep input written mid-graph; resident steps copy PerStep inputs once, so a later kernel write would clobber the host value"
                )));
            }
        }

        // RepeatingStep once-init contract (S5-U6): a repeating training
        // step copies its HostProvided params into their PerProgram storage
        // exactly once at session creation and never re-copies on later
        // steps — steps copy nothing. A HostProvided buffer outside
        // per-program storage could never receive its values in step-mode,
        // so the combination fails closed here, before any launch.
        if self.program_lifetime == DeviceProgramLifetime::RepeatingStep {
            for (id, init) in &initializations {
                if *init == DeviceBufferInitialization::HostProvided {
                    let lifetime = lifetimes
                        .iter()
                        .find(|(buffer_id, _)| buffer_id == id)
                        .map(|(_, lifetime)| *lifetime);
                    let name = identities
                        .iter()
                        .find(|(buffer_id, _, _)| buffer_id == id)
                        .map(|(_, name, _)| name.as_str())
                        .unwrap_or("<unknown>");
                    if lifetime != Some(DeviceBufferLifetime::PerProgram) {
                        return Err(errors::descriptor(format!(
                            "RepeatingStep buffer `{name}` (id {id}) is host-provided but has lifetime `{}`; a repeating step once-inits its host-provided params at session creation, which is defined only for per-program storage",
                            lifetime
                                .map(DeviceBufferLifetime::spelling)
                                .unwrap_or("(no declared lifetime)")
                        )));
                    }
                }
            }
        }

        // Observation admission (F6): results are DECLARED observation points.
        // A result must name a buffer the program allocates, with the
        // `ObservationPoint` lifetime — the only class the session reads back
        // (a writable intermediate or persistent state exposed as a result
        // without an explicit observation fact is rejected, matching the
        // constructor's rule). Each observation must be anchored at a real
        // launch that completes at or after the producing launch.
        let mut result_buffer_ids: Vec<u32> = Vec::with_capacity(self.results.len());
        for result in &self.results {
            if result.version == 0 {
                return Err(errors::descriptor(format!(
                    "device descriptor result for buffer {} uses the reserved zero version",
                    result.buffer_id
                )));
            }
            if result.produced_by == 0 || result.at_launch == 0 {
                return Err(errors::descriptor(format!(
                    "device descriptor result for buffer {} uses a reserved zero launch identity",
                    result.buffer_id
                )));
            }
            if !launch_ids.contains(&result.produced_by) {
                return Err(errors::descriptor(format!(
                    "device descriptor result for buffer {} names unknown producing launch {}",
                    result.buffer_id, result.produced_by
                )));
            }
            if !launch_ids.contains(&result.at_launch) {
                return Err(errors::descriptor(format!(
                    "device descriptor result for buffer {} names unknown observation launch {}",
                    result.buffer_id, result.at_launch
                )));
            }
            if position[&result.at_launch] < position[&result.produced_by] {
                return Err(errors::descriptor(format!(
                    "device descriptor result for buffer {} is observed at launch {} before its producing launch {}; an observation is valid only at or after the producer",
                    result.buffer_id, result.at_launch, result.produced_by
                )));
            }
            if result_buffer_ids.contains(&result.buffer_id) {
                return Err(errors::descriptor(format!(
                    "device descriptor repeats observation buffer {}; results must be unique in the host receipt",
                    result.buffer_id
                )));
            }
            result_buffer_ids.push(result.buffer_id);

            let Some((_, lifetime)) = lifetimes.iter().find(|(id, _)| *id == result.buffer_id)
            else {
                return Err(errors::descriptor(format!(
                    "device descriptor result names buffer {} which no kernel slot allocates",
                    result.buffer_id
                )));
            };
            if *lifetime != DeviceBufferLifetime::ObservationPoint {
                return Err(errors::descriptor(format!(
                    "device descriptor result names buffer {} with lifetime `{}`; only declared observation-point buffers are read back (no undeclared readback)",
                    result.buffer_id,
                    lifetime.spelling()
                )));
            }
            let keyed = versions.iter().any(|version| {
                version.buffer_id == result.buffer_id && version.version == result.version
            });
            if !keyed {
                return Err(errors::descriptor(format!(
                    "device descriptor result names buffer {} version {} which has no keyed metadata",
                    result.buffer_id, result.version
                )));
            }
        }

        // End-of-run observation admission (S5A-U1): the DECLARED cadence set
        // — the wire's `EndOfRun` result rows — is read back exactly once
        // after the step loop. Each entry must name a buffer the program
        // writes (never read-only input state) with a PerStep or PerProgram
        // lifetime (an ObservationPoint buffer is a per-step result and is
        // never read both per step and at the end), must be unique, must not
        // overlap the per-step results, and must carry keyed version
        // metadata. An UNDECLARED readback — a buffer read back without a
        // declared cadence — fails closed here, before any launch.
        let mut end_of_run_buffer_ids: Vec<u32> = Vec::with_capacity(self.end_of_run_results.len());
        for end_of_run in &self.end_of_run_results {
            if end_of_run.version == 0 {
                return Err(errors::descriptor(format!(
                    "device descriptor end-of-run observation for buffer {} uses the reserved zero version",
                    end_of_run.buffer_id
                )));
            }
            if result_buffer_ids.contains(&end_of_run.buffer_id) {
                return Err(errors::descriptor(format!(
                    "device descriptor end-of-run observation names per-step observation buffer {}; a buffer is never read both per step and at the end",
                    end_of_run.buffer_id
                )));
            }
            if end_of_run_buffer_ids.contains(&end_of_run.buffer_id) {
                return Err(errors::descriptor(format!(
                    "device descriptor end-of-run observation repeats buffer {}; the set must be unique in the host receipt",
                    end_of_run.buffer_id
                )));
            }
            end_of_run_buffer_ids.push(end_of_run.buffer_id);

            let Some(meta) = self
                .kernels
                .iter()
                .flat_map(|kernel| kernel.buffers.iter())
                .find(|slot| {
                    slot.buffer_id == end_of_run.buffer_id && slot.version == end_of_run.version
                })
            else {
                return Err(errors::descriptor(format!(
                    "device descriptor end-of-run observation names buffer {} version {} which no kernel slot allocates",
                    end_of_run.buffer_id, end_of_run.version
                )));
            };
            if meta.role == DeviceBufferRole::Input {
                return Err(errors::descriptor(format!(
                    "device descriptor end-of-run observation names input buffer {}; a final value must be written by the program",
                    end_of_run.buffer_id
                )));
            }
            if meta.lifetime != DeviceBufferLifetime::PerStep
                && meta.lifetime != DeviceBufferLifetime::PerProgram
            {
                return Err(errors::descriptor(format!(
                    "device descriptor end-of-run observation names buffer {} with lifetime `{}`; only per-step and per-program buffers are read back once at the end (observation-point buffers are the per-step results)",
                    end_of_run.buffer_id,
                    meta.lifetime.spelling()
                )));
            }
            let keyed = versions.iter().any(|version| {
                version.buffer_id == end_of_run.buffer_id && version.version == end_of_run.version
            });
            if !keyed {
                return Err(errors::descriptor(format!(
                    "device descriptor end-of-run observation names buffer {} version {} which has no keyed metadata",
                    end_of_run.buffer_id, end_of_run.version
                )));
            }
        }
        Ok(())
    }

    /// SHA-256 receipt of the descriptor's carried **program-graph** facts
    /// (F3/F6): the buffer semantic identities + content versions + the
    /// execution-affecting buffer facts (binding, lifetime, initialization),
    /// the program execution-lifetime regime, the declared roots, the ordered
    /// launch sequence with the full facts of each launched kernel
    /// **including the backend entry-name bytes** (`kernel.entry`), the
    /// carried dependency edges, the declared per-step observation points,
    /// and the declared end-of-run observations (S5A-U1).
    ///
    /// This is the **host program-graph identity** (OQ1 resolution — a
    /// DISTINCT domain from the radix call/region execution-descriptor
    /// identity): a SHA-256 receipt under the distinct name `program_graph_hash`,
    /// computed over the domain-tagged canonical descriptor bytes. The
    /// distinct host-graph domain tag [`HOST_PROGRAM_GRAPH_DOMAIN_TAG`] is
    /// embedded as the first length-prefixed field of the byte stream, so a
    /// host program-graph receipt is never interchangeable with the radix
    /// execution-descriptor receipt over identical bytes. It is
    /// backend-entry-inclusive: the same program compiled to differently-
    /// named backend entries (Metal `mlp_loss__0` vs CUDA `mlp_loss__t60_…`)
    /// yields different receipts. It is NOT a semantic-identity claim — the
    /// backend-neutral semantic identity of a program is the complete-program
    /// SHA (`radix_mir_fmir::device_identity_hash`, the A10 identity),
    /// computed from the canonical wire bytes without backend symbols. The
    /// host may consume the radix execution-descriptor receipt where
    /// call-shape identity is the fact; the run/session identity is THIS
    /// domain-tagged receipt, never a u64 FNV value and never the radix
    /// receipt.
    ///
    /// The byte stream is length-prefixed and deterministic. Kernel facts
    /// are inlined per launch, so reordering kernel DECLARATIONS (which
    /// changes neither the launches, the graph, nor the semantic identities)
    /// produces the same receipt — declaration order is never an execution
    /// authority.
    ///
    /// # Panics
    /// Never panics.
    #[must_use]
    pub fn program_graph_hash(&self) -> String {
        format!("sha256:{}", sha256_hex(&self.program_graph_bytes()))
    }

    /// Program-graph identity of this descriptor plus the KV storage/binding
    /// plan. Runtime invocation-state cursor values are excluded: static
    /// hashes include binding expressions, not current cursor values.
    #[must_use]
    pub fn program_graph_hash_with_kv(&self, kv: &KvCacheDescriptor) -> String {
        let mut bytes = self.program_graph_bytes();
        bytes.extend_from_slice(&kv.static_graph_bytes());
        format!("sha256:{}", sha256_hex(&bytes))
    }

    fn program_graph_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        // The distinct host-graph domain tag (OQ1) is part of the receipt
        // substrate: length-prefixed like every UTF-8 field in the canonical
        // byte stream.
        push_bytes(&mut bytes, HOST_PROGRAM_GRAPH_DOMAIN_TAG.as_bytes());
        // The program execution-lifetime regime (single-run vs repeating
        // training step) changes how the session executes the graph, so it is
        // part of the graph identity (S3-U5 census: M07).
        push_u32(&mut bytes, self.program_lifetime as u32);
        let mut buffers: Vec<(&DescriptorBuffer, &u32)> = self
            .kernels
            .iter()
            .flat_map(|kernel| kernel.buffers.iter())
            .map(|slot| (slot, &slot.version))
            .collect();
        // De-duplicate (buffer_id, version) keys and sort so the hash is
        // declaration-independent.
        buffers.sort_by(|(left, left_version), (right, right_version)| {
            (left.buffer_id, *left_version).cmp(&(right.buffer_id, *right_version))
        });
        buffers.dedup_by(|(left, left_version), (right, right_version)| {
            left.buffer_id == right.buffer_id && left_version == right_version
        });
        for (slot, version) in buffers {
            push_u32(&mut bytes, slot.buffer_id);
            push_u32(&mut bytes, slot.semantic_value);
            push_u32(&mut bytes, *version);
            push_u32(&mut bytes, slot.element_ty as u32);
            bytes.extend_from_slice(&slot.element_count.to_le_bytes());
            // S3-U5 (M07): lifetime and initialization are buffer IDENTITY
            // facts (validation enforces cross-reference consistency), so
            // they join the sorted canonical stream — every deduplicated key
            // carries one validated value, keeping the stream
            // declaration-independent.
            push_u32(&mut bytes, slot.lifetime as u32);
            push_u32(&mut bytes, slot.initialization as u32);
        }
        push_u32(&mut bytes, self.roots.len() as u32);
        for root in &self.roots {
            push_u32(&mut bytes, *root);
        }
        push_u32(&mut bytes, self.launches.len() as u32);
        for launch in &self.launches {
            push_u32(&mut bytes, launch.id);
            if let Some(kernel) = self.kernels.get(launch.kernel_index as usize) {
                // Inline the launched kernel's full facts (declaration-order
                // independent).
                push_bytes(&mut bytes, kernel.entry.as_bytes());
                push_u32(&mut bytes, kernel.buffers.len() as u32);
                // S3-U5 (M07): the per-slot facts are inlined in a
                // declaration-independent order too — sorted by the total
                // order (buffer_id, version, binding) — so reordering buffer
                // slot declarations within a kernel declaration does not
                // change the receipt.
                let mut slots: Vec<&DescriptorBuffer> = kernel.buffers.iter().collect();
                slots.sort_by(|left, right| {
                    (left.buffer_id, left.version, left.binding).cmp(&(
                        right.buffer_id,
                        right.version,
                        right.binding,
                    ))
                });
                for slot in slots {
                    push_u32(&mut bytes, slot.buffer_id);
                    push_u32(&mut bytes, slot.semantic_value);
                    push_u32(&mut bytes, slot.version);
                    push_u32(&mut bytes, slot.role as u32);
                    // S3-U5 (M07): binding is a per-slot ABI fact (unique per
                    // kernel; the same buffer id may bind different indices
                    // in different kernels), and lifetime/initialization are
                    // the slot-level execution policies — all part of the
                    // launched kernel's executed facts.
                    push_u32(&mut bytes, slot.binding);
                    push_u32(&mut bytes, slot.lifetime as u32);
                    push_u32(&mut bytes, slot.initialization as u32);
                    push_u32(&mut bytes, slot.element_ty as u32);
                    bytes.extend_from_slice(&slot.element_count.to_le_bytes());
                }
                for axis in kernel.grid {
                    push_u32(&mut bytes, axis);
                }
                for axis in kernel.block {
                    push_u32(&mut bytes, axis);
                }
            }
        }
        push_u32(&mut bytes, self.data_flow.len() as u32);
        for edge in &self.data_flow {
            push_u32(&mut bytes, edge.buffer_id);
            push_u32(&mut bytes, edge.version);
            push_u32(&mut bytes, edge.producer);
            push_u32(&mut bytes, edge.consumer);
        }
        push_u32(&mut bytes, self.results.len() as u32);
        for result in &self.results {
            push_u32(&mut bytes, result.buffer_id);
            push_u32(&mut bytes, result.version);
            push_u32(&mut bytes, result.produced_by);
            push_u32(&mut bytes, result.at_launch);
        }
        // S5A-U1: the declared end-of-run observations are part of the graph
        // the host executes — the one-shot readback after the step loop.
        push_u32(&mut bytes, self.end_of_run_results.len() as u32);
        for end_of_run in &self.end_of_run_results {
            push_u32(&mut bytes, end_of_run.buffer_id);
            push_u32(&mut bytes, end_of_run.version);
        }
        bytes
    }
}

/// Distinct domain tag for the KV storage/binding plan. Cursor values are
/// never part of this stream.
const HOST_KV_STORAGE_DOMAIN_TAG: &str = "faber.host-kv-storage.v1";

impl KvCacheDescriptor {
    /// Validate allocation/view/binding consistency **before any launch**.
    ///
    /// Allocation capacity and view extent are checked as separate facts: a
    /// view must fit in its allocation, and two views may share one
    /// allocation. Binding records keep declared indices and order.
    ///
    /// # Errors
    /// Returns the first typed [`HostError`] the plan violates.
    pub fn validate(&self) -> HostResult<()> {
        let mut allocations: Vec<DescriptorAllocation> = Vec::with_capacity(self.allocations.len());
        for allocation in &self.allocations {
            if allocation.buffer_id == 0 {
                return Err(errors::descriptor(
                    "device descriptor allocation uses the reserved zero buffer identity",
                ));
            }
            if allocation.capacity_bytes == 0 {
                return Err(errors::descriptor(format!(
                    "device descriptor allocation {} has a zero byte capacity",
                    allocation.buffer_id
                )));
            }
            if allocations
                .iter()
                .any(|first| first.buffer_id == allocation.buffer_id)
            {
                return Err(errors::descriptor(format!(
                    "device descriptor repeats allocation identity {}",
                    allocation.buffer_id
                )));
            }
            allocations.push(*allocation);
        }
        if allocations.is_empty() {
            return Err(errors::descriptor(
                "device descriptor declares no allocations",
            ));
        }

        for view in &self.views {
            if view.logical_dims.is_empty() || view.logical_dims.len() != view.strides.len() {
                return Err(errors::descriptor(format!(
                    "device descriptor view on allocation {} has rank-mismatched dims and strides",
                    view.allocation_id
                )));
            }
            if view.logical_dims.iter().any(|dim| *dim == 0)
                || view.strides.iter().any(|stride| *stride == 0)
            {
                return Err(errors::descriptor(format!(
                    "device descriptor view on allocation {} has a zero dim or stride",
                    view.allocation_id
                )));
            }
            if view.maximum_span == 0 {
                return Err(errors::descriptor(format!(
                    "device descriptor view on allocation {} has a zero maximum span",
                    view.allocation_id
                )));
            }
            let Some(allocation) = allocations
                .iter()
                .find(|allocation| allocation.buffer_id == view.allocation_id)
            else {
                return Err(errors::descriptor(format!(
                    "device descriptor view names unknown allocation {}",
                    view.allocation_id
                )));
            };
            let Some(end) = view.static_base.checked_add(view.maximum_span) else {
                return Err(errors::shape_mismatch(format!(
                    "device descriptor view on allocation {} overflows its static envelope",
                    view.allocation_id
                )));
            };
            if end > allocation.capacity_bytes {
                return Err(errors::shape_mismatch(format!(
                    "device descriptor view on allocation {} spans {} bytes from base {} but allocation capacity is {} bytes",
                    view.allocation_id, view.maximum_span, view.static_base, allocation.capacity_bytes
                )));
            }
        }

        for binding in &self.launch_bindings {
            if binding.view_span == 0 {
                return Err(errors::descriptor(format!(
                    "device descriptor launch binding index {} has a zero view span",
                    binding.binding_index
                )));
            }
            let Some(allocation) = allocations
                .iter()
                .find(|allocation| allocation.buffer_id == binding.handle)
            else {
                return Err(errors::descriptor(format!(
                    "device descriptor launch binding index {} names unknown allocation handle {}",
                    binding.binding_index, binding.handle
                )));
            };
            let Some(end) = binding.byte_offset.checked_add(binding.view_span) else {
                return Err(errors::shape_mismatch(format!(
                    "device descriptor launch binding index {} overflows its static envelope",
                    binding.binding_index
                )));
            };
            if end > allocation.capacity_bytes {
                return Err(errors::shape_mismatch(format!(
                    "device descriptor launch binding index {} spans {} bytes from offset {} but allocation {} capacity is {} bytes",
                    binding.binding_index,
                    binding.view_span,
                    binding.byte_offset,
                    binding.handle,
                    allocation.capacity_bytes
                )));
            }
        }
        Ok(())
    }

    /// Launch records in declared binding order, with indices preserved.
    #[must_use]
    pub fn launch_records(&self) -> &[DescriptorLaunchBinding] {
        &self.launch_bindings
    }

    /// SHA-256 receipt of the static storage/binding plan. Runtime cursor
    /// values on [`Self::invocation_state`] are not hashed.
    #[must_use]
    pub fn program_graph_hash(&self) -> String {
        format!("sha256:{}", sha256_hex(&self.static_graph_bytes()))
    }

    fn static_graph_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        push_bytes(&mut bytes, HOST_KV_STORAGE_DOMAIN_TAG.as_bytes());
        let mut allocations: Vec<&DescriptorAllocation> = self.allocations.iter().collect();
        allocations.sort_by_key(|allocation| allocation.buffer_id);
        push_u32(&mut bytes, allocations.len() as u32);
        for allocation in allocations {
            push_u32(&mut bytes, allocation.buffer_id);
            push_u32(&mut bytes, allocation.dtype as u32);
            bytes.extend_from_slice(&allocation.capacity_bytes.to_le_bytes());
            push_u32(&mut bytes, allocation.lifetime as u32);
            push_u32(&mut bytes, allocation.initialization as u32);
        }
        let mut views: Vec<&DescriptorView> = self.views.iter().collect();
        views.sort_by(|left, right| {
            (
                left.allocation_id,
                left.static_base,
                left.maximum_span,
                left.logical_dims.as_slice(),
            )
                .cmp(&(
                    right.allocation_id,
                    right.static_base,
                    right.maximum_span,
                    right.logical_dims.as_slice(),
                ))
        });
        push_u32(&mut bytes, views.len() as u32);
        for view in views {
            push_u32(&mut bytes, view.allocation_id);
            push_u32(&mut bytes, view.logical_dims.len() as u32);
            for dim in &view.logical_dims {
                bytes.extend_from_slice(&dim.to_le_bytes());
            }
            push_u32(&mut bytes, view.strides.len() as u32);
            for stride in &view.strides {
                bytes.extend_from_slice(&stride.to_le_bytes());
            }
            bytes.extend_from_slice(&view.static_base.to_le_bytes());
            bytes.extend_from_slice(&view.maximum_span.to_le_bytes());
        }
        // Launch bindings hash in declared order: the launch record is the
        // declaration. Binding expressions (index, offset, span, source tag)
        // join the identity; current cursor values do not.
        push_u32(&mut bytes, self.launch_bindings.len() as u32);
        for binding in &self.launch_bindings {
            push_u32(&mut bytes, binding.handle);
            push_u32(&mut bytes, binding.binding_index);
            bytes.extend_from_slice(&binding.byte_offset.to_le_bytes());
            bytes.extend_from_slice(&binding.view_span.to_le_bytes());
            push_u32(&mut bytes, binding.runtime_source as u32);
        }
        bytes
    }
}

/// The distinct host program-graph identity domain tag (OQ1 resolution,
/// head-cto advisory cf45415c): the host run/session identity is a DISTINCT
/// domain from the radix call/region execution-descriptor identity
/// (`faber.execution-descriptor.v1`), re-domained under this tag with a
/// SHA-256 receipt. The tag is embedded in the canonical byte stream, so a
/// host program-graph receipt is never interchangeable with a radix
/// execution-descriptor receipt over identical bytes. A later decision to
/// consume the radix digest as the run/session identity is a recorded
/// contract change; a translation facade preserving both identities is
/// forbidden.
pub const HOST_PROGRAM_GRAPH_DOMAIN_TAG: &str = "faber.host-program-graph.v1";

/// SHA-256 (FIPS 180-4) of `bytes` as the lowercase 64-hex digest body —
/// the hashing substrate of the re-domained program-graph receipt
/// (`sha256:<64-hex>`). Kept as a self-contained, dependency-free
/// implementation so the host leaf crate carries no extra hashing dependency
/// for its run/session identity.
///
/// # Panics
/// Never panics.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex_encode(&sha256(bytes))
}

/// The SHA-256 initial hash state (FIPS 180-4 §5.3.3).
const SHA256_INITIAL_STATE: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// The SHA-256 round constants (FIPS 180-4 §4.2.2): the first 32 bits of the
/// fractional parts of the cube roots of the first 64 primes.
const SHA256_ROUND_CONSTANTS: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA-256 (FIPS 180-4) digest of `bytes` as the 32-byte big-endian state.
fn sha256(bytes: &[u8]) -> [u8; 32] {
    // Padding (FIPS 180-4 §5.1.1): 0x80, zero bytes to 56 mod 64, then the
    // 64-bit big-endian message length in bits.
    let bit_length = (bytes.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity(bytes.len() + 72);
    padded.extend_from_slice(bytes);
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_length.to_be_bytes());

    let mut state = SHA256_INITIAL_STATE;
    for block in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let sigma0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let sigma1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(sigma0)
                .wrapping_add(w[i - 7])
                .wrapping_add(sigma1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for (i, constant) in SHA256_ROUND_CONSTANTS.iter().enumerate() {
            let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(sum1)
                .wrapping_add(ch)
                .wrapping_add(*constant)
                .wrapping_add(w[i]);
            let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = sum0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        state = [
            state[0].wrapping_add(a),
            state[1].wrapping_add(b),
            state[2].wrapping_add(c),
            state[3].wrapping_add(d),
            state[4].wrapping_add(e),
            state[5].wrapping_add(f),
            state[6].wrapping_add(g),
            state[7].wrapping_add(h),
        ];
    }

    let mut digest = [0u8; 32];
    for (i, word) in state.iter().enumerate() {
        let start = i * 4;
        digest[start..start + 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

/// Lowercase hex of `bytes` (the receipt hex spelling, `sha256:<64-hex>`).
fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(char::from(HEX[(byte >> 4) as usize]));
        out.push(char::from(HEX[(byte & 0x0f) as usize]));
    }
    out
}

/// Append a `u32` in little-endian to the canonical byte stream (length-free
/// width — the schema is fixed-field, so no field needs its own length).
fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

/// Append a length-prefixed byte slice to the canonical byte stream.
fn push_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    push_u32(bytes, value.len() as u32);
    bytes.extend_from_slice(value);
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

/// Stable error constructors shared by the descriptor validator, the
/// composite host, and the launch adapters.
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

    pub(crate) fn descriptor(message: impl Into<String>) -> HostError {
        HostError {
            code: super::E_DEVICE_DESCRIPTOR.to_owned(),
            message: message.into(),
            retryable: false,
        }
    }

    pub(crate) fn abi_mismatch(message: impl Into<String>) -> HostError {
        HostError {
            code: super::E_DEVICE_ABI_MISMATCH.to_owned(),
            message: message.into(),
            retryable: false,
        }
    }

    pub(crate) fn dtype_mismatch(message: impl Into<String>) -> HostError {
        HostError {
            code: super::E_DEVICE_DTYPE_MISMATCH.to_owned(),
            message: message.into(),
            retryable: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_bf16_round_trip_spelling_and_byte_width() {
        assert_eq!(DeviceDataType::F16.spelling(), "f16");
        assert_eq!(
            DeviceDataType::from_spelling("f16"),
            Some(DeviceDataType::F16)
        );
        assert_eq!(DeviceDataType::F16.byte_width(), 2);

        assert_eq!(DeviceDataType::BF16.spelling(), "bf16");
        assert_eq!(
            DeviceDataType::from_spelling("bf16"),
            Some(DeviceDataType::BF16)
        );
        assert_eq!(DeviceDataType::BF16.byte_width(), 2);
    }

    #[test]
    fn placement_discriminant_maps_to_device_data_type() {
        // MirScalarLayout declaration-order discriminants (F2 owner).
        assert_eq!(
            DeviceDataType::from_placement_discriminant(3),
            Some(DeviceDataType::I32)
        );
        assert_eq!(
            DeviceDataType::from_placement_discriminant(4),
            Some(DeviceDataType::I64)
        );
        assert_eq!(
            DeviceDataType::from_placement_discriminant(6),
            Some(DeviceDataType::U8)
        );
        assert_eq!(
            DeviceDataType::from_placement_discriminant(10),
            Some(DeviceDataType::F16)
        );
        assert_eq!(
            DeviceDataType::from_placement_discriminant(11),
            Some(DeviceDataType::F32)
        );
        assert_eq!(
            DeviceDataType::from_placement_discriminant(12),
            Some(DeviceDataType::F64)
        );
        assert_eq!(DeviceDataType::from_placement_discriminant(0), None);
        assert_eq!(DeviceDataType::from_placement_discriminant(30), None);

        assert_eq!(DeviceDataType::I32.placement_discriminant(), Some(3));
        assert_eq!(DeviceDataType::I64.placement_discriminant(), Some(4));
        assert_eq!(DeviceDataType::U8.placement_discriminant(), Some(6));
        assert_eq!(DeviceDataType::F16.placement_discriminant(), Some(10));
        assert_eq!(DeviceDataType::F32.placement_discriminant(), Some(11));
        assert_eq!(DeviceDataType::F64.placement_discriminant(), Some(12));
        assert_eq!(DeviceDataType::BF16.placement_discriminant(), None);
    }
}
