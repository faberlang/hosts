//! Device-set membership + topology schema (gpu-inference-multi-device,
//! MD1-S1 — `DeviceSet` / `DeviceSetSelection` / `DeviceTopologySnapshot` /
//! `DeviceLink`).
//!
//! Frozen vocabulary ([`md0-naming-contract.md`](radix/docs/factory/gpu-inference-multi-device/md0-naming-contract.md)
//! §1):
//!
//! - [`DeviceSet`] — selected **membership** of [`PhysicalDeviceId`]s.
//!   "Mesh" wording never replaces simple membership: a set is a set of
//!   identities with no shape, link, or ordering semantics.
//! - [`DeviceSetSelection`] — the request type: explicit `PhysicalDeviceId`s
//!   or declared constraints. It **never overloads** the portable
//!   `FmirDeviceSelection` (`Auto|Metal|Cuda`,
//!   [`crate::backend::DeviceBackend`]) with machine-local ids — the
//!   single-device request surface stays unchanged (MD-A15).
//! - [`DeviceTopologySnapshot`] — per-device capabilities/health (carried by
//!   the [`DeviceDiscoverySnapshot`](crate::discovery::DeviceDiscoverySnapshot)
//!   the topology was measured from) plus directed [`DeviceLink`] rows. Live
//!   link facts are **never folded into `PhysicalDeviceId`**: identity is
//!   derived only from stable identity facts, and topology facts are
//!   timestamped samples.
//! - [`DeviceLink`] — a directed pair `i→j`, `i≠j`, with `Admitted` (path
//!   class + measured facts), `NotAttempted`, or `Rejected` state.
//!
//! Membership validation (C2 *healthy*/*degraded*,
//! [`md0-cost-contract.md`](radix/docs/factory/gpu-inference-multi-device/md0-cost-contract.md)
//! §2): only members recorded under the **current** [`DeviceHealthGeneration`]
//! epoch are selectable; a *degraded* member stays selectable but only for
//! gates it satisfies — the degraded fact is exposed so gates can evaluate
//! it. A NOT-ATTEMPTED or rejected link is **never assumed** (C2 topology
//! gate): only an explicitly admitted link passes. A size-1 `DeviceSet` is
//! legal on the single-device host (T1 consequence 1);
//! two-physical-identity and P2P rows stay NOT ATTEMPTED.

use crate::backend::DeviceBackend;
use crate::device_identity::{DeviceHealthGeneration, PhysicalDeviceId};
use crate::discovery::{DeviceDiscoveryEntry, DeviceDiscoverySnapshot};
use std::collections::{BTreeMap, BTreeSet};

/// Selected membership of [`PhysicalDeviceId`]s.
///
/// Membership is exactly a set of stable machine-local identities. Equality
/// and ordering are over the identities only — the ordinal locator never
/// participates (naming contract §1): renaming an ordinal changes neither the
/// id nor any set built from it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeviceSet {
    members: BTreeSet<PhysicalDeviceId>,
}

impl DeviceSet {
    /// Canonical membership from an id iterator. Duplicates collapse; the set
    /// iterates in stable identity order.
    #[must_use]
    pub fn from_members(ids: impl IntoIterator<Item = PhysicalDeviceId>) -> Self {
        Self {
            members: ids.into_iter().collect(),
        }
    }

    /// Whether `id` is a member.
    #[must_use]
    pub fn contains(&self, id: &PhysicalDeviceId) -> bool {
        self.members.contains(id)
    }

    /// The member identities in stable identity order (ordinal-free).
    #[must_use]
    pub fn members(&self) -> &BTreeSet<PhysicalDeviceId> {
        &self.members
    }

    /// Number of members.
    #[must_use]
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// True when the set has no members.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// True when exactly one device is selected — the legal size-1 case on
    /// the single-device host (T1 consequence 1).
    #[must_use]
    pub fn is_singleton(&self) -> bool {
        self.members.len() == 1
    }

    /// Healthy-epoch membership validation (C2 §2 *healthy*/*degraded*).
    ///
    /// Every member must be recorded in `discovery` under the **current**
    /// health generation. An id that is not in the snapshot at all — which
    /// covers both an unknown device and a **replaced** device (same ordinal,
    /// different identity facts; replacement detection, naming contract §1) —
    /// and a member recorded under a stale epoch are rejected.
    ///
    /// # Errors
    ///
    /// Returns [`MembershipError`] when a member is missing from the snapshot
    /// or recorded under a stale health generation.
    pub fn validate(
        &self,
        discovery: &DeviceDiscoverySnapshot,
        current: DeviceHealthGeneration,
    ) -> Result<(), MembershipError> {
        for id in &self.members {
            validate_member(discovery, current, id)?;
        }
        Ok(())
    }
}

/// Why a membership validation failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MembershipError {
    /// The id is not recorded in the snapshot. Covers both a device never
    /// present and a **replaced** device (same ordinal, different identity
    /// facts — naming contract §1): replacement yields a *new* id that no
    /// snapshot entry carries until the next probe.
    UnknownDevice(PhysicalDeviceId),
    /// The member is recorded under a stale health generation — the snapshot
    /// was sampled before the epoch advanced and must be rejected before it
    /// gates admission or planning (MD1-Q3 default).
    StaleEpoch {
        /// The requested member.
        id: PhysicalDeviceId,
        /// Generation the snapshot recorded.
        recorded: DeviceHealthGeneration,
        /// Generation that is current at validation time.
        current: DeviceHealthGeneration,
    },
}

/// The device-set request type.
///
/// Carries **explicit** machine-local ids or **declared constraints**; it
/// never overloads the portable `FmirDeviceSelection` (`Auto|Metal|Cuda`,
/// [`crate::backend::DeviceBackend`]) with machine-local ids (naming
/// contract §1) — the single-device request surface is preserved unchanged
/// (MD-A15).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceSetSelection {
    /// Explicit selection: the caller names the exact [`PhysicalDeviceId`]s.
    /// Every id must be a current-epoch member of the snapshot.
    Explicit(BTreeSet<PhysicalDeviceId>),
    /// Declared membership constraints, resolved against the snapshot.
    Constraints(DeviceSetConstraints),
}

impl DeviceSetSelection {
    /// Build an explicit selection from an id iterator.
    #[must_use]
    pub fn explicit(ids: impl IntoIterator<Item = PhysicalDeviceId>) -> Self {
        Self::Explicit(ids.into_iter().collect())
    }

    /// Build a constraint-based selection.
    #[must_use]
    pub fn constraints(constraints: DeviceSetConstraints) -> Self {
        Self::Constraints(constraints)
    }

    /// Resolve the request into a [`DeviceSet`] and validate membership
    /// against `discovery` under `current`.
    ///
    /// - `Explicit` ids are validated: every id must be a current-epoch
    ///   member; an unknown/replaced or stale-epoch id rejects the whole
    ///   selection.
    /// - `Constraints` selects the snapshot's current-epoch members matching
    ///   the declared shape (backend, exclusions, count bounds).
    ///
    /// # Errors
    ///
    /// Returns [`SelectionError`] when an explicit id fails membership
    /// validation or constraint resolution cannot satisfy the declared
    /// count bounds.
    pub fn resolve(
        &self,
        discovery: &DeviceDiscoverySnapshot,
        current: DeviceHealthGeneration,
    ) -> Result<DeviceSet, SelectionError> {
        match self {
            Self::Explicit(ids) => {
                let set = DeviceSet::from_members(ids.iter().cloned());
                set.validate(discovery, current)
                    .map_err(SelectionError::Membership)?;
                Ok(set)
            }
            Self::Constraints(constraints) => {
                let mut selected = Vec::new();
                for entry in discovery.devices().values() {
                    // Healthy-epoch rule: stale-epoch members are not
                    // selectable (C2 §2).
                    if current.is_stale(entry.health_generation) {
                        continue;
                    }
                    if let Some(backend) = constraints.backend {
                        if entry.identity.backend() != backend {
                            continue;
                        }
                    }
                    if constraints.exclude.contains(&entry.identity) {
                        continue;
                    }
                    selected.push(entry.identity.clone());
                }
                let set = DeviceSet::from_members(selected);
                if set.len() < constraints.min_count {
                    return Err(SelectionError::BelowMinimum {
                        min: constraints.min_count,
                        actual: set.len(),
                    });
                }
                if let Some(max) = constraints.max_count {
                    if set.len() > max {
                        return Err(SelectionError::AboveMaximum {
                            max,
                            actual: set.len(),
                        });
                    }
                }
                Ok(set)
            }
        }
    }
}

/// Declared membership constraints for a
/// [`DeviceSetSelection::Constraints`] request.
///
/// Shape-only constraints (backend, exclusions, member count). They select
/// membership; they never encode priority policy (that is MD1-P1's scope,
/// C2 §1: gates reject, objectives rank only gate-passing plans).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DeviceSetConstraints {
    /// Constrain every member to one backend, when declared.
    pub backend: Option<DeviceBackend>,
    /// Identities that must never be selected.
    pub exclude: BTreeSet<PhysicalDeviceId>,
    /// Minimum member count (0 when unconstrained).
    pub min_count: usize,
    /// Maximum member count (`None` = unconstrained).
    pub max_count: Option<usize>,
}

/// Why a selection could not resolve to a [`DeviceSet`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionError {
    /// Membership validation failed (unknown/replaced or stale-epoch member).
    Membership(MembershipError),
    /// Constraint resolution selected fewer members than `min_count`.
    BelowMinimum {
        /// Declared minimum.
        min: usize,
        /// Actual selected count.
        actual: usize,
    },
    /// Constraint resolution selected more members than `max_count`.
    AboveMaximum {
        /// Declared maximum.
        max: usize,
        /// Actual selected count.
        actual: usize,
    },
}

/// A directed device pair `i→j` with `i≠j` (T1 §3: self is not a P2P row).
///
/// State is `Admitted` (path class + measured facts), `NotAttempted`, or
/// `Rejected`. A NOT-ATTEMPTED or rejected link is **never assumed** (C2
/// topology gate) — only an explicitly admitted link may be traversed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceLink {
    from: PhysicalDeviceId,
    to: PhysicalDeviceId,
    state: DeviceLinkState,
}

/// State of one directed link.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceLinkState {
    /// Measured and explicitly admitted.
    Admitted {
        /// The class of admitted path (peer-access or labeled host-staged).
        path_class: LinkPathClass,
        /// Measured directional facts for the `from → to` direction.
        facts: LinkFacts,
    },
    /// The pair was never probed/attempted. On pharos every directed pair
    /// `i→j, i≠j` is NOT ATTEMPTED (T1 §3). Never assumed.
    NotAttempted,
    /// The pair was probed and explicitly rejected.
    Rejected {
        /// Why the pair was rejected (recorded for the explained receipt).
        reason: String,
    },
}

/// The class of an admitted link path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LinkPathClass {
    /// Measured peer-access-admitted direct device-to-device copy path.
    Peer,
    /// Explicitly labeled host-staged path. Silent host staging is forbidden
    /// (T1 §6 consequence 3): every staged copy is labeled and timed.
    HostStaged,
}

/// Measured directional facts of an admitted link (`from → to`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LinkFacts {
    /// Measured directional bandwidth, bytes per second (T1 §4 rates).
    pub bandwidth_bytes_per_sec: u64,
    /// Measured per-copy fixed latency, nanoseconds (T1 §4).
    pub latency_nanos: u64,
}

impl DeviceLink {
    /// An explicitly admitted directed link with path class + measured facts.
    ///
    /// # Panics
    ///
    /// Panics if `from == to` (a [`DeviceLink`] is a directed pair `i→j`, `i≠j`).
    #[must_use]
    pub fn admitted(
        from: PhysicalDeviceId,
        to: PhysicalDeviceId,
        path_class: LinkPathClass,
        facts: LinkFacts,
    ) -> Self {
        assert_ne!(
            from, to,
            "a DeviceLink is a directed pair i→j with i≠j (T1 §3)"
        );
        Self {
            from,
            to,
            state: DeviceLinkState::Admitted { path_class, facts },
        }
    }

    /// An explicitly NOT-ATTEMPTED directed pair.
    ///
    /// # Panics
    ///
    /// Panics if `from == to` (a [`DeviceLink`] is a directed pair `i→j`, `i≠j`).
    #[must_use]
    pub fn not_attempted(from: PhysicalDeviceId, to: PhysicalDeviceId) -> Self {
        assert_ne!(
            from, to,
            "a DeviceLink is a directed pair i→j with i≠j (T1 §3)"
        );
        Self {
            from,
            to,
            state: DeviceLinkState::NotAttempted,
        }
    }

    /// An explicitly rejected directed pair.
    ///
    /// # Panics
    ///
    /// Panics if `from == to` (a [`DeviceLink`] is a directed pair `i→j`, `i≠j`).
    #[must_use]
    pub fn rejected(
        from: PhysicalDeviceId,
        to: PhysicalDeviceId,
        reason: impl Into<String>,
    ) -> Self {
        assert_ne!(
            from, to,
            "a DeviceLink is a directed pair i→j with i≠j (T1 §3)"
        );
        Self {
            from,
            to,
            state: DeviceLinkState::Rejected {
                reason: reason.into(),
            },
        }
    }

    /// The source device.
    #[must_use]
    pub fn from(&self) -> &PhysicalDeviceId {
        &self.from
    }

    /// The destination device.
    #[must_use]
    pub fn to(&self) -> &PhysicalDeviceId {
        &self.to
    }

    /// The link state.
    #[must_use]
    pub fn state(&self) -> &DeviceLinkState {
        &self.state
    }
}

/// Topology facts for one probe: per-device capabilities/health (from the
/// discovery sample) plus directed [`DeviceLink`] rows.
///
/// Live link facts live here — **never folded into `PhysicalDeviceId`**
/// (naming contract §1): identity is derived only from stable identity facts,
/// and topology is a timestamped sample.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceTopologySnapshot {
    discovery: DeviceDiscoverySnapshot,
    links: BTreeMap<(PhysicalDeviceId, PhysicalDeviceId), DeviceLink>,
}

impl DeviceTopologySnapshot {
    /// Build a topology from the discovery sample that measured the devices
    /// and the directed link rows probed in the same sample.
    ///
    /// Every link endpoint must be a device recorded in `discovery`, and a
    /// link never joins a device to itself (T1 §3: self is not a P2P row) —
    /// violations are programmer errors and fail fast, matching the discovery
    /// snapshot's assertion style.
    ///
    /// # Panics
    ///
    /// Panics if a link joins a device to itself or names an endpoint that is
    /// not in `discovery`.
    #[must_use]
    pub fn new(
        discovery: DeviceDiscoverySnapshot,
        links: impl IntoIterator<Item = DeviceLink>,
    ) -> Self {
        let mut by_pair = BTreeMap::new();
        for link in links {
            assert_ne!(
                link.from(),
                link.to(),
                "a DeviceLink is a directed pair i→j with i≠j (T1 §3)"
            );
            for endpoint in [link.from(), link.to()] {
                assert!(
                    find_entry(&discovery, endpoint).is_some(),
                    "link endpoint {endpoint} is not a device in the discovery snapshot"
                );
            }
            by_pair.insert((link.from().clone(), link.to().clone()), link);
        }
        Self {
            discovery,
            links: by_pair,
        }
    }

    /// The discovery sample the topology was measured from — carries the
    /// per-device capability/memory/health facts.
    #[must_use]
    pub fn discovery(&self) -> &DeviceDiscoverySnapshot {
        &self.discovery
    }

    /// Per-device facts for `id`, if it is a device in this topology.
    #[must_use]
    pub fn member(&self, id: &PhysicalDeviceId) -> Option<&DeviceDiscoveryEntry> {
        find_entry(&self.discovery, id)
    }

    /// The directed link row `from → to`, if one is recorded.
    #[must_use]
    pub fn link(&self, from: &PhysicalDeviceId, to: &PhysicalDeviceId) -> Option<&DeviceLink> {
        self.links.get(&(from.clone(), to.clone()))
    }

    /// All directed link rows, in stable `(from, to)` identity order.
    pub fn links(&self) -> impl Iterator<Item = &DeviceLink> {
        self.links.values()
    }

    /// The C2 topology gate: may a transfer traverse `from → to`?
    ///
    /// Only an explicitly **admitted** link passes. Absent rows, NOT-ATTEMPTED
    /// rows, and rejected rows all fail — an unmeasured or rejected link is
    /// never assumed (C2 §1.2 topology/peer-access; T1 §3). Endpoints outside
    /// the topology fail. A `from == to` self-move is a local copy, not a
    /// link traversal, and passes without a link row.
    ///
    /// # Errors
    ///
    /// Returns [`LinkGateError`] when an endpoint is unknown, no directed
    /// row is recorded, or the row is NOT-ATTEMPTED / rejected.
    pub fn traversal_allowed(
        &self,
        from: &PhysicalDeviceId,
        to: &PhysicalDeviceId,
    ) -> Result<(), LinkGateError> {
        if from == to {
            return Ok(());
        }
        if find_entry(&self.discovery, from).is_none() {
            return Err(LinkGateError::UnknownEndpoint {
                endpoint: from.clone(),
            });
        }
        if find_entry(&self.discovery, to).is_none() {
            return Err(LinkGateError::UnknownEndpoint {
                endpoint: to.clone(),
            });
        }
        match self.links.get(&(from.clone(), to.clone())) {
            None => Err(LinkGateError::NoLinkRecorded {
                from: from.clone(),
                to: to.clone(),
            }),
            Some(link) => match link.state() {
                DeviceLinkState::Admitted { .. } => Ok(()),
                DeviceLinkState::NotAttempted => Err(LinkGateError::NotAttempted {
                    from: from.clone(),
                    to: to.clone(),
                }),
                DeviceLinkState::Rejected { reason } => Err(LinkGateError::Rejected {
                    from: from.clone(),
                    to: to.clone(),
                    reason: reason.clone(),
                }),
            },
        }
    }
}

/// Why the topology gate rejected a traversal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkGateError {
    /// One endpoint is not a device in the topology — no link can be
    /// admitted for an unknown device.
    UnknownEndpoint {
        /// The endpoint that is not a topology device.
        endpoint: PhysicalDeviceId,
    },
    /// No directed row is recorded — an unmeasured pair is never assumed
    /// (C2 topology gate; T1 §3).
    NoLinkRecorded {
        /// Source device.
        from: PhysicalDeviceId,
        /// Destination device.
        to: PhysicalDeviceId,
    },
    /// The pair was explicitly recorded NOT ATTEMPTED.
    NotAttempted {
        /// Source device.
        from: PhysicalDeviceId,
        /// Destination device.
        to: PhysicalDeviceId,
    },
    /// The pair was probed and explicitly rejected.
    Rejected {
        /// Source device.
        from: PhysicalDeviceId,
        /// Destination device.
        to: PhysicalDeviceId,
        /// Recorded rejection reason.
        reason: String,
    },
}

fn find_entry<'a>(
    discovery: &'a DeviceDiscoverySnapshot,
    id: &PhysicalDeviceId,
) -> Option<&'a DeviceDiscoveryEntry> {
    discovery
        .devices()
        .values()
        .find(|entry| &entry.identity == id)
}

fn validate_member(
    discovery: &DeviceDiscoverySnapshot,
    current: DeviceHealthGeneration,
    id: &PhysicalDeviceId,
) -> Result<(), MembershipError> {
    let entry =
        find_entry(discovery, id).ok_or_else(|| MembershipError::UnknownDevice(id.clone()))?;
    if current.is_stale(entry.health_generation) {
        return Err(MembershipError::StaleEpoch {
            id: id.clone(),
            recorded: entry.health_generation,
            current,
        });
    }
    Ok(())
}

#[cfg(test)]
#[path = "device_set_test.rs"]
mod tests;
