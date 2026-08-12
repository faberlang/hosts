//! Physical device identity + health epoch types (gpu-inference-multi-device,
//! MD1-D1 — identity + discovery schema freeze).
//!
//! Frozen identity vocabulary ([`md0-naming-contract.md`](radix/docs/factory/gpu-inference-multi-device/md0-naming-contract.md)
//! §1):
//!
//! - [`PhysicalDeviceId`] — an **opaque identity class** for one physical
//!   device on one machine. Canonical source is per host/backend (CUDA: the
//!   PCI UUID measured on pharos; Metal: a registry/stable identifier — a
//!   declared procedure to be confirmed on burgus at MD1-H1). Scope is
//!   **machine-local**: ids never travel across machines and never enter
//!   package/program identity (A10 semantic hash, naming contract §2).
//!   **The ordinal is a locator only, never identity**: the same backend +
//!   same ordinal + different identity facts is a *distinct* id (a replaced
//!   device), and renaming an ordinal never changes an id.
//! - [`DeviceHealthGeneration`] — a monotonic machine-local epoch over the
//!   admission-gating fact set (presence, identity, capability set, memory
//!   totals, healthy/degraded transition). It is **distinct from the semantic
//!   `ValueGeneration`** epoch carried by FMIR semantic values (radix-air /
//!   faber `package::device`); the two are never conflated.
//!
//! Discovery facts live in [`crate::discovery`]: they are timestamped samples
//! and never retroactively rewrite a `PhysicalDeviceId`.

use crate::backend::DeviceBackend;

/// Canonical stable identity facts for one physical device, per backend.
///
/// These are the facts [`PhysicalDeviceId`] derives from. **Ordinal, topology,
/// capability, and memory facts never appear here** (naming contract §1):
/// capability/memory facts are discovery samples, not identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DeviceIdentityFacts {
    /// NVIDIA CUDA device. Canonical source: the **PCI UUID** as reported by
    /// nvidia-smi, `GPU-…` prefix included (T1 measured on pharos:
    /// `GPU-3e017562-9ec3-da9a-962d-b8bd5f9e24be`).
    Cuda(CudaIdentityFacts),
    /// Apple Metal device. Canonical source: the device registry/stable
    /// identifier (a declared procedure; T1 measured only CUDA — to be
    /// confirmed on burgus at MD1-H1).
    Metal(MetalIdentityFacts),
}

/// CUDA identity facts.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CudaIdentityFacts {
    /// Canonical source: nvidia-smi PCI UUID (`GPU-3e017562-…`, prefix
    /// included).
    pub pci_uuid: String,
    /// Corroborating driver API UUID without the `GPU-` prefix
    /// (`3e017562-…`, the `device_enum` probe). Both reports are kept
    /// distinct (T1 §2); `None` when a probe exposed only one report. A
    /// differing driver UUID at the same ordinal is a changed identity fact
    /// and yields a distinct id.
    pub driver_uuid: Option<String>,
}

/// Metal identity facts.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MetalIdentityFacts {
    /// Registry/stable device identifier (declared procedure).
    pub registry_id: String,
}

/// An opaque, machine-local identity of one physical device.
///
/// Equality, ordering, and hashing are over the canonical identity facts
/// only — the ordinal locator never participates (naming contract §1). An id
/// is **immutable**: discovery facts never retroactively rewrite one; a
/// changed fact set at the same ordinal is a *new* id (replacement), and the
/// [`DeviceHealthGeneration`] advances.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PhysicalDeviceId {
    facts: DeviceIdentityFacts,
}

impl PhysicalDeviceId {
    /// Build an id from canonical identity facts.
    #[must_use]
    pub fn from_facts(facts: DeviceIdentityFacts) -> Self {
        Self { facts }
    }

    /// CUDA convenience constructor (canonical PCI UUID + optional
    /// corroborating driver API UUID).
    #[must_use]
    pub fn cuda(pci_uuid: impl Into<String>, driver_uuid: Option<String>) -> Self {
        Self::from_facts(DeviceIdentityFacts::Cuda(CudaIdentityFacts {
            pci_uuid: pci_uuid.into(),
            driver_uuid,
        }))
    }

    /// Metal convenience constructor (registry/stable identifier).
    #[must_use]
    pub fn metal(registry_id: impl Into<String>) -> Self {
        Self::from_facts(DeviceIdentityFacts::Metal(MetalIdentityFacts {
            registry_id: registry_id.into(),
        }))
    }

    /// The backend this device belongs to.
    #[must_use]
    pub fn backend(&self) -> DeviceBackend {
        match &self.facts {
            DeviceIdentityFacts::Cuda(_) => DeviceBackend::Cuda,
            DeviceIdentityFacts::Metal(_) => DeviceBackend::Metal,
        }
    }

    /// The canonical identity facts this id derives from (read access; the id
    /// itself stays opaque and immutable).
    #[must_use]
    pub fn facts(&self) -> &DeviceIdentityFacts {
        &self.facts
    }

    /// Deterministic canonical byte encoding of the identity facts — the
    /// encoding used inside discovery snapshot canonical bytes. Stable across
    /// processes and machines; the ordinal never enters it.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match &self.facts {
            DeviceIdentityFacts::Cuda(c) => {
                out.push(0u8); // tag: Cuda
                push_str(&mut out, &c.pci_uuid);
                push_bool(&mut out, c.driver_uuid.is_some());
                if let Some(uuid) = &c.driver_uuid {
                    push_str(&mut out, uuid);
                }
            }
            DeviceIdentityFacts::Metal(m) => {
                out.push(1u8); // tag: Metal
                push_str(&mut out, &m.registry_id);
            }
        }
        out
    }

    /// Replacement detection (naming contract §1): whether `previous` — the
    /// identity observed at the *same ordinal locator* — is a different
    /// device. Any difference in identity facts (or backend) at the same
    /// ordinal means the device was replaced; the caller mints the new id
    /// from the current facts and advances [`DeviceHealthGeneration`].
    #[must_use]
    pub fn change_against(&self, previous: &Self) -> IdentityChange {
        if self == previous {
            IdentityChange::SameDevice
        } else {
            IdentityChange::Replaced
        }
    }
}

impl std::fmt::Display for PhysicalDeviceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.facts {
            DeviceIdentityFacts::Cuda(c) => {
                write!(f, "cuda:{}", c.pci_uuid)?;
                if let Some(uuid) = &c.driver_uuid {
                    write!(f, " (driver {uuid})")?;
                }
            }
            DeviceIdentityFacts::Metal(m) => write!(f, "metal:{}", m.registry_id)?,
        }
        Ok(())
    }
}

/// Whether an observed identity is the same device as a previously recorded
/// one at the same ordinal locator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IdentityChange {
    /// Same backend + identical identity facts: the same device.
    SameDevice,
    /// Different identity facts at the same ordinal: the device was replaced.
    /// Mint the new [`PhysicalDeviceId`] and advance
    /// [`DeviceHealthGeneration`].
    Replaced,
}

/// Locator-only ordinal of a device within one machine.
///
/// **Never identity** (naming contract §1; MD-A2). The ordinal names *where*
/// a device sits in the enumeration order of a probe; it may change without
/// changing the device, and the same ordinal may name different devices over
/// time (replacement). Ordering is over the device id, never this locator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeviceOrdinal(u32);

impl DeviceOrdinal {
    /// Build a locator ordinal.
    #[must_use]
    pub const fn new(ordinal: u32) -> Self {
        Self(ordinal)
    }

    /// The raw ordinal value.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for DeviceOrdinal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Monotonic machine-local epoch over the admission-gating fact set.
///
/// Advances on **any** observed change in facts that gate admission: device
/// presence, identity (replacement), capability set, memory totals, and
/// healthy/degraded transitions (MD1-Q3 default). **Distinct from the
/// semantic `ValueGeneration`** epoch of FMIR semantic values — never
/// conflated. A snapshot, plan, or session bound to a stale generation is
/// rejected before it gates admission or planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeviceHealthGeneration(u64);

impl DeviceHealthGeneration {
    /// The first generation (T1 measured the pharos device healthy at
    /// epoch 1).
    #[must_use]
    pub const fn initial() -> Self {
        Self(1)
    }

    /// The raw epoch value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Advance the epoch by one. Called on any observed admission-gating
    /// change (presence, identity/capability/memory, healthy/degraded).
    #[must_use]
    pub const fn advance(self) -> Self {
        Self(self.0.saturating_add(1))
    }

    /// True when `candidate` is not the current generation — a snapshot or
    /// session recorded under `candidate` is stale and must be rejected.
    #[must_use]
    pub fn is_stale(self, candidate: Self) -> bool {
        candidate != self
    }

    /// True when `candidate` is exactly the current generation.
    #[must_use]
    pub fn is_current(self, candidate: Self) -> bool {
        candidate == self
    }
}

impl std::fmt::Display for DeviceHealthGeneration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// --- deterministic canonical byte helpers (shared with discovery.rs) ---
//
// Lengths are u64 little-endian so the encoding is stable across platforms
// (no `size_t` or endianness dependence).

pub(crate) fn push_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

pub(crate) fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

pub(crate) fn push_bool(out: &mut Vec<u8>, v: bool) {
    out.push(u8::from(v));
}

pub(crate) fn push_str(out: &mut Vec<u8>, s: &str) {
    push_u64(out, s.len() as u64);
    out.extend_from_slice(s.as_bytes());
}

#[cfg(test)]
#[path = "device_identity_test.rs"]
mod tests;
