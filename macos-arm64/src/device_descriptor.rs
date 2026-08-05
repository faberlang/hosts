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

use std::collections::BTreeMap;

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
    /// Declared observation points (F6): the explicit result rows the host
    /// reads back and releases. Only these buffers are observable.
    pub results: Vec<DescriptorResult>,
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
            return Err(descriptor_error(
                "device descriptor carries an empty module image",
            ));
        }
        if self.kernels.is_empty() {
            return Err(descriptor_error("device descriptor declares no kernels"));
        }
        if self.launches.is_empty() {
            return Err(descriptor_error("device descriptor declares no launches"));
        }

        let mut launch_ids: Vec<u32> = Vec::with_capacity(self.launches.len());
        for launch in &self.launches {
            if launch.id == 0 {
                return Err(descriptor_error(
                    "device descriptor has a launch with the reserved zero identity",
                ));
            }
            if launch_ids.contains(&launch.id) {
                return Err(descriptor_error(format!(
                    "device descriptor repeats launch identity {}",
                    launch.id
                )));
            }
            if self.kernels.get(launch.kernel_index as usize).is_none() {
                return Err(descriptor_error(format!(
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
                return Err(descriptor_error(
                    "device descriptor has a root with the reserved zero identity",
                ));
            }
            if root_ids.contains(root) {
                return Err(descriptor_error(format!(
                    "device descriptor repeats legal execution root {root}"
                )));
            }
            if !launch_ids.contains(root) {
                return Err(descriptor_error(format!(
                    "device descriptor root {root} names an unknown launch"
                )));
            }
            root_ids.push(*root);
        }
        if root_ids.is_empty() {
            return Err(descriptor_error(
                "device descriptor declares no legal execution roots",
            ));
        }

        let mut versions: Vec<DescriptorBufferVersion> =
            Vec::with_capacity(self.buffer_versions.len());
        for version in &self.buffer_versions {
            if version.version == 0 {
                return Err(descriptor_error(format!(
                    "device descriptor buffer {} uses the reserved zero version",
                    version.buffer_id
                )));
            }
            if version.element_count == 0 {
                return Err(descriptor_error(format!(
                    "device descriptor buffer {} version {} has a zero element count",
                    version.buffer_id, version.version
                )));
            }
            if let Some(first) = versions.iter().find(|first| {
                first.buffer_id == version.buffer_id && first.version == version.version
            }) {
                if first.element_ty != version.element_ty {
                    return Err(dtype_error(format!(
                        "device buffer {} version {} carries conflicting element types {} and {}",
                        version.buffer_id,
                        version.version,
                        first.element_ty.spelling(),
                        version.element_ty.spelling()
                    )));
                }
                if first.element_count != version.element_count {
                    return Err(shape_error(format!(
                        "device buffer {} version {} carries conflicting element counts {} and {}",
                        version.buffer_id,
                        version.version,
                        first.element_count,
                        version.element_count
                    )));
                }
                return Err(descriptor_error(format!(
                    "device descriptor repeats buffer {} version {} metadata",
                    version.buffer_id, version.version
                )));
            }
            versions.push(*version);
        }
        if versions.is_empty() {
            return Err(descriptor_error(
                "device descriptor declares no version-keyed buffer metadata",
            ));
        }

        for edge in &self.data_flow {
            if edge.version == 0 || edge.producer == 0 || edge.consumer == 0 {
                return Err(descriptor_error(
                    "device descriptor data-flow edge uses a reserved zero identity",
                ));
            }
            if edge.producer == edge.consumer {
                return Err(descriptor_error(format!(
                    "device descriptor data-flow edge for buffer {} version {} is self-referential at launch {}",
                    edge.buffer_id, edge.version, edge.producer
                )));
            }
            if !launch_ids.contains(&edge.producer) || !launch_ids.contains(&edge.consumer) {
                return Err(descriptor_error(format!(
                    "device descriptor data-flow edge for buffer {} version {} references an unknown launch",
                    edge.buffer_id, edge.version
                )));
            }
            if !versions.iter().any(|version| {
                version.buffer_id == edge.buffer_id && version.version == edge.version
            }) {
                return Err(descriptor_error(format!(
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
                return Err(descriptor_error(format!(
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
                return Err(descriptor_error(format!(
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
                return Err(descriptor_error(format!(
                    "device descriptor launch {launch} is not reachable from any declared root; the carried graph is incomplete"
                )));
            }
        }

        let mut identities: Vec<(u32, String, DeviceBufferRole)> = Vec::new();
        let mut semantic_values: Vec<(u32, u32)> = Vec::new();
        let mut lifetimes: Vec<(u32, DeviceBufferLifetime)> = Vec::new();
        let mut initializations: Vec<(u32, DeviceBufferInitialization)> = Vec::new();

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
                if slot.version == 0 {
                    return Err(descriptor_error(format!(
                        "device descriptor kernel `{}` binds buffer `{}` with the reserved zero version",
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

                // F1: the stable semantic value identity. The same buffer id
                // always holds the same value, and two different buffer ids
                // never alias one value (two unrelated same-name/same-shape
                // values are distinct).
                if slot.semantic_value == 0 {
                    return Err(descriptor_error(format!(
                        "device buffer `{}` (id {}) carries the reserved zero semantic value identity",
                        slot.buffer_name, slot.buffer_id
                    )));
                }
                if let Some((_, first_semantic)) =
                    semantic_values.iter().find(|(id, _)| *id == slot.buffer_id)
                {
                    if *first_semantic != slot.semantic_value {
                        return Err(abi_error(format!(
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
                        return Err(abi_error(format!(
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

                let Some(version) = versions.iter().find(|version| {
                    version.buffer_id == slot.buffer_id && version.version == slot.version
                }) else {
                    return Err(descriptor_error(format!(
                        "device buffer `{}` (id {}) version {} has no keyed metadata",
                        slot.buffer_name, slot.buffer_id, slot.version
                    )));
                };
                if version.element_ty != slot.element_ty {
                    return Err(dtype_error(format!(
                        "device buffer `{}` (id {}) version {} is referenced with conflicting element types {} and {}",
                        slot.buffer_name,
                        slot.buffer_id,
                        slot.version,
                        version.element_ty.spelling(),
                        slot.element_ty.spelling()
                    )));
                }
                if version.element_count != slot.element_count {
                    return Err(shape_error(format!(
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
                        return Err(abi_error(format!(
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
                if let Some((_, first_init)) = initializations
                    .iter()
                    .find(|(id, _)| *id == slot.buffer_id)
                {
                    if *first_init != slot.initialization {
                        return Err(abi_error(format!(
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
                        return Err(descriptor_error(format!(
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
                return Err(descriptor_error(format!(
                    "device descriptor result for buffer {} uses the reserved zero version",
                    result.buffer_id
                )));
            }
            if result.produced_by == 0 || result.at_launch == 0 {
                return Err(descriptor_error(format!(
                    "device descriptor result for buffer {} uses a reserved zero launch identity",
                    result.buffer_id
                )));
            }
            if !launch_ids.contains(&result.produced_by) {
                return Err(descriptor_error(format!(
                    "device descriptor result for buffer {} names unknown producing launch {}",
                    result.buffer_id, result.produced_by
                )));
            }
            if !launch_ids.contains(&result.at_launch) {
                return Err(descriptor_error(format!(
                    "device descriptor result for buffer {} names unknown observation launch {}",
                    result.buffer_id, result.at_launch
                )));
            }
            if position[&result.at_launch] < position[&result.produced_by] {
                return Err(descriptor_error(format!(
                    "device descriptor result for buffer {} is observed at launch {} before its producing launch {}; an observation is valid only at or after the producer",
                    result.buffer_id, result.at_launch, result.produced_by
                )));
            }
            if result_buffer_ids.contains(&result.buffer_id) {
                return Err(descriptor_error(format!(
                    "device descriptor repeats observation buffer {}; results must be unique in the host receipt",
                    result.buffer_id
                )));
            }
            result_buffer_ids.push(result.buffer_id);

            let Some((_, lifetime)) = lifetimes.iter().find(|(id, _)| *id == result.buffer_id)
            else {
                return Err(descriptor_error(format!(
                    "device descriptor result names buffer {} which no kernel slot allocates",
                    result.buffer_id
                )));
            };
            if *lifetime != DeviceBufferLifetime::ObservationPoint {
                return Err(descriptor_error(format!(
                    "device descriptor result names buffer {} with lifetime `{}`; only declared observation-point buffers are read back (no undeclared readback)",
                    result.buffer_id,
                    lifetime.spelling()
                )));
            }
            let keyed = versions.iter().any(|version| {
                version.buffer_id == result.buffer_id && version.version == result.version
            });
            if !keyed {
                return Err(descriptor_error(format!(
                    "device descriptor result names buffer {} version {} which has no keyed metadata",
                    result.buffer_id, result.version
                )));
            }
        }
        Ok(())
    }

    /// FNV-1a hash of the descriptor's carried **semantic graph** (F3/F6):
    /// the buffer semantic identities + content versions, the declared
    /// roots, the ordered launch sequence with the full facts of each
    /// launched kernel, the carried dependency edges, and the declared
    /// observation points. This is the graph identity the host executes —
    /// distinct from the module provenance hash, which only names the
    /// backend blob.
    ///
    /// The byte stream is length-prefixed and deterministic. Kernel facts
    /// are inlined per launch, so reordering kernel DECLARATIONS (which
    /// changes neither the launches, the graph, nor the semantic identities)
    /// produces the same hash — declaration order is never an execution
    /// authority.
    ///
    /// # Panics
    /// Never panics.
    #[must_use]
    pub fn semantic_graph_hash(&self) -> u64 {
        let mut bytes = Vec::new();
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
                for slot in &kernel.buffers {
                    push_u32(&mut bytes, slot.buffer_id);
                    push_u32(&mut bytes, slot.semantic_value);
                    push_u32(&mut bytes, slot.version);
                    push_u32(&mut bytes, slot.role as u32);
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
        fnv1a64(&bytes)
    }
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
