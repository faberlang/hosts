//! Opaque physical device-handle carriers (HOSTS-COORD).
//!
//! Intra-module split from `faber-runtime/src/device.rs` (inventory §3.2):
//! [`DeviceHandle`]/[`DeviceHandleKind`] (physical-handle carriers) land with
//! HOSTS-COORD — they must not ride the support crate and never enter
//! generated language values (DDPP0 §1 row 7 rule 2). The selection/build
//! metadata half of `device.rs` (`DeviceBackend`/`DeviceSelection`,
//! `from_spelling`) is RADIX-ARTIFACT+FABER-BUILD.

use faber::{FromValor, Valor};
use std::collections::BTreeMap;

use crate::backend::DeviceBackend;

/// What kind of device object an opaque handle names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceHandleKind {
    /// A compiled module (MSL or PTX image loaded by the driver).
    Module,
    /// A device buffer of the given byte length.
    Buffer {
        /// Allocated byte length on the device.
        len_bytes: u64,
    },
}

impl DeviceHandleKind {
    /// Stable diagnostic spelling (`"module"` / `"buffer"`).
    #[must_use]
    pub fn spelling(self) -> &'static str {
        match self {
            Self::Module => "module",
            Self::Buffer { .. } => "buffer",
        }
    }
}

/// Opaque host-owned handle identity carried across control boundaries.
///
/// A handle is a **carrier, not a payload**: it names the backend, the kind,
/// and the session-local opaque id, and nothing else. Tensor bytes, module
/// text, and shapes never travel inside a handle — they live in the owning
/// host session's registry. Valor-frame integration preserves this invariant:
/// the control frame for a handle is scalar identifiers only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceHandle {
    /// The backend that owns the underlying device object.
    pub backend: DeviceBackend,
    /// What kind of device object the id names.
    pub kind: DeviceHandleKind,
    /// Session-local opaque id (allocated by the owning session's registry).
    pub id: u64,
}

impl DeviceHandle {
    /// The byte length of a buffer handle; `None` for modules.
    #[must_use]
    pub fn len_bytes(self) -> Option<u64> {
        match self.kind {
            DeviceHandleKind::Module => None,
            DeviceHandleKind::Buffer { len_bytes } => Some(len_bytes),
        }
    }
}

impl From<DeviceHandle> for Valor {
    fn from(handle: DeviceHandle) -> Self {
        Valor::from(&handle)
    }
}

impl From<&DeviceHandle> for Valor {
    /// Control-frame representation of a handle: scalar identifiers only, no
    /// payload bytes (a control frame for a handle can never be a tensor).
    fn from(handle: &DeviceHandle) -> Self {
        let mut fields = BTreeMap::new();
        fields.insert(
            "device_backend".to_owned(),
            Valor::Textus(handle.backend.spelling().to_owned()),
        );
        fields.insert(
            "device_kind".to_owned(),
            Valor::Textus(handle.kind.spelling().to_owned()),
        );
        // Session-local opaque ids are small; the existing host control-frame
        // precedent already carries them as Numerus (`id.0 as i64`).
        #[allow(clippy::cast_possible_wrap)]
        fields.insert("device_id".to_owned(), Valor::Numerus(handle.id as i64));
        if let Some(len_bytes) = handle.len_bytes() {
            #[allow(clippy::cast_possible_wrap)]
            fields.insert("len_bytes".to_owned(), Valor::Numerus(len_bytes as i64));
        }
        Valor::Tabula(fields)
    }
}

impl FromValor for DeviceHandle {
    /// Extract a handle from its control-frame representation. Rejects any
    /// frame that is not a scalar-identifier tabula (a frame carrying a
    /// `Octeti` payload is not a handle control frame).
    fn from_valor(value: &Valor) -> Option<Self> {
        let Valor::Tabula(fields) = value else {
            return None;
        };
        if fields
            .values()
            .any(|field| matches!(field, Valor::Octeti(_)))
        {
            return None;
        }
        let backend_spelling = String::from_valor(fields.get("device_backend")?)?;
        let backend = DeviceBackend::from_spelling(&backend_spelling)?;
        let id = u64::try_from(i64::from_valor(fields.get("device_id")?)?).ok()?;
        let kind = String::from_valor(fields.get("device_kind")?)?;
        match kind.as_str() {
            "module" => Some(Self {
                backend,
                kind: DeviceHandleKind::Module,
                id,
            }),
            "buffer" => {
                let len_bytes = u64::try_from(i64::from_valor(fields.get("len_bytes")?)?).ok()?;
                Some(Self {
                    backend,
                    kind: DeviceHandleKind::Buffer { len_bytes },
                    id,
                })
            }
            _ => None,
        }
    }
}
