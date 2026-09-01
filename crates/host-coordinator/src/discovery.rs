//! Timestamped device discovery facts (gpu-inference-multi-device, MD1-D1 —
//! identity + discovery schema freeze).
//!
//! A [`DeviceDiscoverySnapshot`] is a **sample, never timeless** (naming
//! contract §1): every fact (identity, capabilities, memory totals, health
//! generation, ordinal locator, probe provenance) is tied to one probe time
//! and one probe. Determinism: identical input facts produce identical
//! canonical bytes and an identical content-addressed
//! [`DeviceDiscoverySnapshotId`].
//!
//! Memory totals keep both reports **distinct and never conflated** (T1 §8):
//! the vendor tool report (nvidia-smi, MiB) and the driver/runtime API report
//! (`cuDeviceTotalMem`, bytes) are separate fields. Directed P2P/link facts
//! are recorded at probe level here; the detailed per-pair `DeviceLink` rows
//! are MD1-S1 facts (`device_set.rs`).

use crate::backend::DeviceBackend;
use crate::device_identity::{
    DeviceHealthGeneration, DeviceOrdinal, PhysicalDeviceId, push_bool, push_str, push_u32,
    push_u64,
};
use std::collections::BTreeMap;

/// Content-addressed id of one discovery sample.
///
/// SHA-256 over the sample's canonical bytes: identical facts (including the
/// explicit probe time) yield the identical id. Machine-local; never portable
/// package content and never part of the A10 semantic hash (naming
/// contract §2).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceDiscoverySnapshotId([u8; 32]);

impl DeviceDiscoverySnapshotId {
    /// SHA-256 over the sample's canonical bytes.
    #[must_use]
    pub fn from_canonical_bytes(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes))
    }

    /// Raw 32-byte digest.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Lowercase hex digest.
    #[must_use]
    pub fn hex(&self) -> String {
        let mut s = String::with_capacity(64);
        for b in self.0 {
            use std::fmt::Write;
            write!(s, "{b:02x}").expect("writing to a String cannot fail");
        }
        s
    }
}

impl std::fmt::Display for DeviceDiscoverySnapshotId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.hex())
    }
}

impl std::fmt::Debug for DeviceDiscoverySnapshotId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DeviceDiscoverySnapshotId({})", self.hex())
    }
}

/// One device's discovery facts within a snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceDiscoveryEntry {
    /// Locator ordinal within this machine — **never identity**.
    pub ordinal: DeviceOrdinal,
    /// Opaque stable identity (machine-local).
    pub identity: PhysicalDeviceId,
    /// Vendor model descriptor, e.g. `"NVIDIA GeForce RTX 5070"` (T1 §2).
    pub device_model: Option<String>,
    /// Capability facts (CUDA identity, generic launch-resource limits, dtype surface).
    pub capabilities: DeviceCapabilities,
    /// Memory totals — both reports kept distinct (T1 §8).
    pub memory: DeviceMemory,
    /// Healthy/degraded state observed by the probe.
    pub health: DeviceHealth,
    /// Health epoch under which these facts were observed.
    pub health_generation: DeviceHealthGeneration,
    /// Probe provenance (which probe/tool versions produced the sample).
    pub probe_provenance: ProbeProvenance,
}

impl DeviceDiscoveryEntry {
    /// The backend of the identified device (derived from the identity — the
    /// entry never carries an independent backend field that could disagree).
    #[must_use]
    pub fn backend(&self) -> DeviceBackend {
        self.identity.backend()
    }
}

/// Capability facts that gate admission.
///
/// `compute_capability` and `sm_count` are CUDA identity facts. The five
/// launch-resource fields are generic and populated per backend from live
/// device queries (CUDA block / Metal threadgroup; warp / simdgroup).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceCapabilities {
    /// Compute capability (e.g. `12.0` for the RTX 5070, Blackwell).
    pub compute_capability: ComputeCapability,
    /// Streaming-multiprocessor count (48 on the RTX 5070).
    pub sm_count: u32,
    /// Raw arithmetic dtype surface the device executes (T1 §2 smoke).
    pub dtype_surface: DtypeSurface,
    /// Maximum threads in one workgroup (CUDA block / Metal threadgroup).
    pub max_threads_per_workgroup: u32,
    /// Minimum guaranteed workgroup shared memory, bytes.
    pub workgroup_shared_memory_min_bytes: u32,
    /// Maximum opt-in workgroup shared memory, bytes.
    pub workgroup_shared_memory_max_bytes: u32,
    /// Collective width (CUDA warp / Metal simdgroup).
    pub collective_width: u32,
    /// True when the device shares host memory (integrated / unified).
    pub unified_memory: bool,
}

/// Compute capability `major.minor` (12.0 on the RTX 5070).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputeCapability {
    /// Major version.
    pub major: u32,
    /// Minor version.
    pub minor: u32,
}

/// Raw arithmetic dtype surface (T1 §2 empirical kernel smoke on pharos:
/// f32/f64/f16/bf16/i8/i32 all PASS). Independent of host `DeviceDataType`
/// abstractions and never `U8`-as-quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(clippy::struct_excessive_bools)] // independent measured dtype flags, not a config bag
pub struct DtypeSurface {
    /// f32 executed PASS.
    pub f32: bool,
    /// f64 executed PASS.
    pub f64: bool,
    /// f16 executed PASS.
    pub f16: bool,
    /// bf16 executed PASS.
    pub bf16: bool,
    /// i8 executed PASS.
    pub i8: bool,
    /// i32 executed PASS.
    pub i32: bool,
}

impl DtypeSurface {
    /// An empty surface (no dtype measured).
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            f32: false,
            f64: false,
            f16: false,
            bf16: false,
            i8: false,
            i32: false,
        }
    }
}

/// Memory totals from both reports, **kept distinct — never conflated**
/// (T1 §8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceMemory {
    /// Vendor tool report total (nvidia-smi), MiB. `None` for backends with
    /// no such tool report (Metal unified memory is OS-managed). T1:
    /// 12227 MiB.
    pub tool_report_total_mib: Option<u64>,
    /// Driver/runtime API report total (`cuDeviceTotalMem`), bytes. T1:
    /// 12 343 705 600 B (≈ 11772 MiB usable — numerically distinct from the
    /// 12227 MiB tool report; both stay separate).
    pub api_total_bytes: u64,
}

/// Healthy/degraded state observed by a probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceHealth {
    /// Device admitted and healthy.
    Healthy,
    /// Device present but degraded; the reason is recorded.
    Degraded(String),
}

/// Probe provenance: which probe and tool versions produced the facts. Facts
/// are a sample — the provenance names the sample's source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeProvenance {
    /// Named probe, e.g. `"device_enum + nvidia-smi"` (T1 §2).
    pub probe: String,
    /// Tool/driver versions at probe time, e.g.
    /// `"driver 595.71.05 / CUDA 13.2"`.
    pub tool_versions: String,
}

/// Probe-level state of directed P2P facts.
///
/// Detailed per-pair rows (`admitted` / NOT-ATTEMPTED / rejected) are MD1-S1
/// `DeviceLink` facts (`device_set.rs`); this snapshot records only the
/// probe-level state so a missing row can **never be mistaken for a pass**
/// (T1 §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum P2pProbeState {
    /// Fewer than two physical devices: every directed pair `i→j, i≠j` is
    /// NOT ATTEMPTED (T1 §3 — pharos has one CUDA device).
    NotAttempted,
    /// Probe ran with ≥2 devices; directed rows live in MD1-S1 link facts.
    Attempted,
}

/// A timestamped discovery sample — facts, never timeless.
///
/// The id is content-addressed over the canonical bytes of every field, so
/// identical input facts (including the explicit probe time) produce
/// identical bytes and an identical id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceDiscoverySnapshot {
    id: DeviceDiscoverySnapshotId,
    probe_utc_nanos: u64,
    devices: BTreeMap<DeviceOrdinal, DeviceDiscoveryEntry>,
    p2p: P2pProbeState,
}

impl DeviceDiscoverySnapshot {
    /// Build a sample. `probe_utc_nanos` is the explicit probe time (Unix
    /// epoch, nanoseconds) — supplied by the probe, never sampled inside this
    /// type, so determinism holds for identical inputs. Every entry's
    /// [`DeviceDiscoveryEntry::ordinal`] must equal its `BTreeMap` key (the
    /// locator is recorded per device); a mismatch is a programmer error and
    /// panics.
    ///
    /// # Panics
    ///
    /// Panics if any entry's ordinal does not match its map key.
    #[must_use]
    pub fn new(
        probe_utc_nanos: u64,
        devices: BTreeMap<DeviceOrdinal, DeviceDiscoveryEntry>,
        p2p: P2pProbeState,
    ) -> Self {
        for (key, entry) in &devices {
            assert_eq!(
                key, &entry.ordinal,
                "discovery entry ordinal must match its map key"
            );
        }
        let mut out = Self {
            id: DeviceDiscoverySnapshotId([0u8; 32]),
            probe_utc_nanos,
            devices,
            p2p,
        };
        let bytes = out.canonical_bytes_without_id();
        out.id = DeviceDiscoverySnapshotId::from_canonical_bytes(&bytes);
        out
    }

    /// Populate a snapshot from host-enumerated entries.
    ///
    /// Additive seam for product hosts: entries are keyed by their own
    /// locator ordinal so callers do not assemble the `BTreeMap` by hand.
    /// P2P is **not** inferred from device count — this constructor records
    /// [`P2pProbeState::NotAttempted`]. A host that actually probed directed
    /// pairs still uses [`Self::new`].
    #[must_use]
    pub fn from_enumerated(
        probe_utc_nanos: u64,
        entries: impl IntoIterator<Item = DeviceDiscoveryEntry>,
    ) -> Self {
        let devices: BTreeMap<DeviceOrdinal, DeviceDiscoveryEntry> = entries
            .into_iter()
            .map(|entry| (entry.ordinal, entry))
            .collect();
        Self::new(probe_utc_nanos, devices, P2pProbeState::NotAttempted)
    }

    /// Content-addressed id of this sample.
    #[must_use]
    pub fn id(&self) -> DeviceDiscoverySnapshotId {
        self.id
    }

    /// Explicit probe time (Unix epoch, nanoseconds).
    #[must_use]
    pub fn probe_utc_nanos(&self) -> u64 {
        self.probe_utc_nanos
    }

    /// Per-device facts, keyed by ordinal locator (canonical `BTree` order).
    #[must_use]
    pub fn devices(&self) -> &BTreeMap<DeviceOrdinal, DeviceDiscoveryEntry> {
        &self.devices
    }

    /// The entry recorded at the given ordinal locator, if present.
    #[must_use]
    pub fn entry(&self, ordinal: DeviceOrdinal) -> Option<&DeviceDiscoveryEntry> {
        self.devices.get(&ordinal)
    }

    /// Probe-level P2P state.
    #[must_use]
    pub fn p2p_state(&self) -> P2pProbeState {
        self.p2p
    }

    /// Deterministic canonical bytes (identical facts → identical bytes).
    /// The id itself does not participate — it is derived *from* these bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        self.canonical_bytes_without_id()
    }

    /// True when any device entry carries a stale health generation — such a
    /// snapshot must be rejected before it gates admission or planning.
    #[must_use]
    pub fn is_stale(&self, current: DeviceHealthGeneration) -> bool {
        self.devices
            .values()
            .any(|entry| current.is_stale(entry.health_generation))
    }

    /// True when every device entry carries exactly `current`.
    #[must_use]
    pub fn is_current_generation(&self, current: DeviceHealthGeneration) -> bool {
        !self.is_stale(current)
    }

    fn canonical_bytes_without_id(&self) -> Vec<u8> {
        let mut out = Vec::new();
        push_u64(&mut out, self.probe_utc_nanos);
        push_u64(&mut out, self.devices.len() as u64);
        for (ordinal, entry) in &self.devices {
            // Locator (part of the sample facts; never part of identity).
            push_u32(&mut out, ordinal.get());
            out.extend_from_slice(&entry.identity.canonical_bytes());
            push_bool(&mut out, entry.device_model.is_some());
            if let Some(model) = &entry.device_model {
                push_str(&mut out, model);
            }
            // Capabilities.
            push_u32(&mut out, entry.capabilities.compute_capability.major);
            push_u32(&mut out, entry.capabilities.compute_capability.minor);
            push_u32(&mut out, entry.capabilities.sm_count);
            let ds = &entry.capabilities.dtype_surface;
            push_bool(&mut out, ds.f32);
            push_bool(&mut out, ds.f64);
            push_bool(&mut out, ds.f16);
            push_bool(&mut out, ds.bf16);
            push_bool(&mut out, ds.i8);
            push_bool(&mut out, ds.i32);
            push_u32(&mut out, entry.capabilities.max_threads_per_workgroup);
            push_u32(
                &mut out,
                entry.capabilities.workgroup_shared_memory_min_bytes,
            );
            push_u32(
                &mut out,
                entry.capabilities.workgroup_shared_memory_max_bytes,
            );
            push_u32(&mut out, entry.capabilities.collective_width);
            push_bool(&mut out, entry.capabilities.unified_memory);
            // Memory — both reports, kept distinct.
            push_bool(&mut out, entry.memory.tool_report_total_mib.is_some());
            if let Some(mib) = entry.memory.tool_report_total_mib {
                push_u64(&mut out, mib);
            }
            push_u64(&mut out, entry.memory.api_total_bytes);
            // Health + generation.
            match &entry.health {
                DeviceHealth::Healthy => out.push(0u8),
                DeviceHealth::Degraded(reason) => {
                    out.push(1u8);
                    push_str(&mut out, reason);
                }
            }
            push_u64(&mut out, entry.health_generation.get());
            // Provenance.
            push_str(&mut out, &entry.probe_provenance.probe);
            push_str(&mut out, &entry.probe_provenance.tool_versions);
        }
        match self.p2p {
            P2pProbeState::NotAttempted => out.push(0u8),
            P2pProbeState::Attempted => out.push(1u8),
        }
        out
    }
}

/// Minimal SHA-256 (FIPS 180-4), dependency-free and deterministic.
///
/// Used only to content-address discovery snapshot ids; the digest is stable
/// across processes and machines (validated against published test vectors).
#[derive(Debug, Clone, Copy)]
struct Sha256 {
    state: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_len: u64,
}

const SHA256_K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09_e667,
                0xbb67_ae85,
                0x3c6e_f372,
                0xa54f_f53a,
                0x510e_527f,
                0x9b05_688c,
                0x1f83_d9ab,
                0x5be0_cd19,
            ],
            buf: [0; 64],
            buf_len: 0,
            total_len: 0,
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
            }
        }
        while data.len() >= 64 {
            let block: [u8; 64] = (&data[..64]).try_into().expect("slice is exactly 64 bytes");
            self.compress(&block);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    fn finalize(mut self) -> [u8; 32] {
        // Total bit length, big-endian, appended in the final padding block.
        let bit_len = self.total_len.wrapping_mul(8).to_be_bytes();
        let mut tail = Vec::with_capacity(self.buf_len + 9);
        tail.extend_from_slice(&self.buf[..self.buf_len]);
        tail.push(0x80);
        while tail.len() % 64 != 56 {
            tail.push(0);
        }
        tail.extend_from_slice(&bit_len);
        for block in tail.chunks_exact(64) {
            self.compress(block);
        }
        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn digest(data: &[u8]) -> [u8; 32] {
        let mut h = Self::new();
        h.update(data);
        h.finalize()
    }

    #[allow(clippy::many_single_char_names)] // FIPS 180-4 working variables a–h
    fn compress(&mut self, block: &[u8]) {
        let mut w = [0u32; 64];
        for (i, word) in w.iter_mut().enumerate().take(16) {
            let base = i * 4;
            *word = u32::from_be_bytes([
                block[base],
                block[base + 1],
                block[base + 2],
                block[base + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }
}

#[cfg(test)]
#[path = "discovery_test.rs"]
mod tests;
