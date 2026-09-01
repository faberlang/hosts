//! One [`DeviceRuntime`] per selected [`PhysicalDeviceId`] (MD3H-H2).
//!
//! The set is composition, not a host rewire: H3 will later make the
//! CompositeHost's one-runtime field the M=1 member. Today M=1 is the live
//! shape (burgus Metal, pharos CUDA); M>1 is proven by composing fake
//! sessions. Members are one backend — Metal+CUDA in one job is a campaign
//! non-goal.

use std::collections::BTreeMap;

use host_coordinator::DeviceBackend;
use host_coordinator::device_identity::PhysicalDeviceId;

use crate::cuda_host::enumerate_cuda_physical_devices;
use crate::device_host::DeviceRuntime;
use crate::kernel::{HostError, HostResult};
use crate::metal_host::enumerate_metal_physical_devices;

/// One [`DeviceRuntime`] session per selected [`PhysicalDeviceId`].
pub struct DeviceRuntimeSet {
    members: BTreeMap<PhysicalDeviceId, DeviceRuntime>,
}

impl DeviceRuntimeSet {
    /// Assemble a set from already-opened sessions. Empty and mixed-backend
    /// sets fail closed. Each runtime's backend must match its id.
    pub fn from_members(
        members: impl IntoIterator<Item = (PhysicalDeviceId, DeviceRuntime)>,
    ) -> HostResult<Self> {
        let members: BTreeMap<PhysicalDeviceId, DeviceRuntime> = members.into_iter().collect();
        if members.is_empty() {
            return Err(HostError::invalid_args(
                "DeviceRuntimeSet requires at least one PhysicalDeviceId",
            ));
        }
        if !backends_are_homogeneous(members.keys().map(PhysicalDeviceId::backend)) {
            return Err(HostError::invalid_args(
                "DeviceRuntimeSet members must share one backend (Metal+CUDA in one job is a non-goal)",
            ));
        }
        for (id, runtime) in &members {
            if runtime.backend() != id.backend() {
                return Err(HostError::invalid_args(format!(
                    "runtime backend {} does not match PhysicalDeviceId {id}",
                    runtime.backend().spelling()
                )));
            }
        }
        Ok(Self { members })
    }

    /// Open one live session per selected id. M=1 today: every selected id
    /// of a backend must be the single enumerated device, opened through
    /// [`DeviceRuntime::open`]. Selecting two distinct same-backend ids
    /// fails closed (use [`Self::from_members`] for the M>1 composition
    /// shape).
    pub fn open_live(ids: impl IntoIterator<Item = PhysicalDeviceId>) -> HostResult<Self> {
        let mut unique = BTreeMap::<PhysicalDeviceId, ()>::new();
        for id in ids {
            unique.insert(id, ());
        }
        if unique.is_empty() {
            return Err(HostError::invalid_args(
                "DeviceRuntimeSet::open_live requires at least one PhysicalDeviceId",
            ));
        }
        if !backends_are_homogeneous(unique.keys().map(PhysicalDeviceId::backend)) {
            return Err(HostError::invalid_args(
                "DeviceRuntimeSet members must share one backend (Metal+CUDA in one job is a non-goal)",
            ));
        }
        if unique.len() != 1 {
            return Err(HostError::invalid_args(
                "live DeviceRuntimeSet is M=1 today; M>1 is composition via from_members",
            ));
        }
        let Some(id) = unique.into_keys().next() else {
            return Err(HostError::invalid_args(
                "DeviceRuntimeSet::open_live requires at least one PhysicalDeviceId",
            ));
        };
        match_enumerated_identity(&id)?;
        let runtime = DeviceRuntime::open(id.backend())?;
        Self::from_members([(id, runtime)])
    }

    /// Number of physical sessions in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// True when the set has no members (unreachable after construction).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// The homogeneous backend of every member.
    #[must_use]
    pub fn backend(&self) -> DeviceBackend {
        match self.members.keys().next() {
            Some(id) => id.backend(),
            // from_members rejects empty sets; keep a defined backend so
            // production hygiene stays at zero expects.
            None => DeviceBackend::Metal,
        }
    }

    /// Whether `id` is a member.
    #[must_use]
    pub fn contains(&self, id: &PhysicalDeviceId) -> bool {
        self.members.contains_key(id)
    }

    /// Member identities in stable identity order.
    #[must_use]
    pub fn ids(&self) -> impl Iterator<Item = &PhysicalDeviceId> {
        self.members.keys()
    }

    /// The session bound to `id`.
    #[must_use]
    pub fn get(&self, id: &PhysicalDeviceId) -> Option<&DeviceRuntime> {
        self.members.get(id)
    }

    /// The mutable session bound to `id`.
    pub fn get_mut(&mut self, id: &PhysicalDeviceId) -> Option<&mut DeviceRuntime> {
        self.members.get_mut(id)
    }
}

impl std::fmt::Debug for DeviceRuntimeSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceRuntimeSet")
            .field("backend", &self.backend())
            .field("ids", &self.members.keys().collect::<Vec<_>>())
            .finish()
    }
}

fn backends_are_homogeneous(mut backends: impl Iterator<Item = DeviceBackend>) -> bool {
    let Some(first) = backends.next() else {
        return true;
    };
    backends.all(|backend| backend == first)
}

fn match_enumerated_identity(id: &PhysicalDeviceId) -> HostResult<()> {
    match id.backend() {
        DeviceBackend::Metal => {
            let devices = enumerate_metal_physical_devices()?;
            if devices
                .iter()
                .any(|device| PhysicalDeviceId::metal(&device.registry_id) == *id)
            {
                Ok(())
            } else {
                Err(HostError::invalid_args(format!(
                    "PhysicalDeviceId {id} is not among the enumerated Metal devices"
                )))
            }
        }
        DeviceBackend::Cuda => {
            let devices = enumerate_cuda_physical_devices()?;
            if devices.iter().any(|device| {
                PhysicalDeviceId::cuda(&device.pci_uuid, device.driver_uuid.clone()) == *id
            }) {
                Ok(())
            } else {
                Err(HostError::invalid_args(format!(
                    "PhysicalDeviceId {id} is not among the enumerated CUDA devices"
                )))
            }
        }
    }
}
