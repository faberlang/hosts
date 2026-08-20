//! Typed/ranged transport + host-staged adapter + selected-transport receipt
//! records (gpu-inference-multi-device, MD3-T1 — parallel with MD3-F1 on
//! disjoint module files; the `lib.rs` registration line serializes per lane
//! queue §2).
//!
//! ## Typed/ranged transfer semantics (T2 §8.2 — exit-gate bullet 2)
//!
//! A transfer is identified by **byte range + logical dtype/layout/generation**
//! (the mirrored dtype/layout encodings from `execution_transaction.rs` —
//! stable canonical forms, FC6/FC18). [`validate_before_copy`] compares the
//! **declared** facts of a [`TransferSpec`] against the **actual** facts of a
//! [`SourceValue`] and the destination the copy would land on; **dtype,
//! layout, bounds, generation, owner, and destination mismatches reject before
//! copy** — fail-before-copy with a typed diagnostic ([`TransferRejection`])
//! naming the violated class and the exact failing fact. This is the
//! transfer-time authority of MD2-C1 residual 2: typed/ranged validation
//! against actual content.
//!
//! ## `TransportAdapter` trait
//!
//! [`TransportAdapter`] has two implementations:
//!
//! - [`HostStagedAdapter`] — the **host-staged** path (pinned-host staging,
//!   **labeled + timed** — no silent host staging, T2 §7; byte accounting at
//!   the T1 measured constants, FC5). Every copy accumulates against the
//!   declared transfer budget at the measured rates (T2 §8.6 budget
//!   discipline); a materially different fixture-measured rate is recorded
//!   and re-budgeted — never replaced by an assumed P2P/NVLink rate.
//! - [`PeerAdapter`] — the explicit **peer** path, which is **NOT ATTEMPTED**
//!   until a directed pair is measured and admitted (T2 §6 per-pair flip
//!   rule — never a global switch). A peer transfer on an unmeasured pair is
//!   rejected by [`TransferError::PeerNotAdmitted`]; even an admitted pair is
//!   never copied here (real peer execution requires a real same-host ≥2-GPU
//!   topology — lane queue §5b — and this campaign never fabricates a peer
//!   pass).
//!
//! ## Timeout/failure policy per copy (S4)
//!
//! Each [`TransferSpec`] declares a per-copy timeout; a copy that exceeds it
//! surfaces [`TransferError::Timeout`] and a failed copy surfaces
//! [`TransferError::Failed`] — both convert (via
//! [`TransferError::into_backend_error`]) into the coordinator's
//! [`BackendError`] vocabulary, which the `ExecutionTransaction` (MD3-X1)
//! aborts on with no partial publication (Q8 fail-closed; MD-A13).
//!
//! ## Selected-transport receipt records (S4)
//!
//! The **actual selected transport** — copy path, staging buffers,
//! streams/queues/events, timeout/failure policy, bytes, timing — records into
//! [`TransportReceipt`] (the S4 selected-transport section of the transaction
//! receipt; X1's base `TransactionReceipt` doc names MD3-T1 as the home of
//! these records). It **never records into the portable logical plan**: the
//! logical plan's transport surface is the [`TransportPathMirror`]
//! **admissibility** label only (v1 = `{host-staged}`), and the selected
//! transport is a runtime fact the mirror/spec vocabulary cannot express — a
//! structural separation this module's tests assert.
//!
//! ## Mirror vocabulary
//!
//! faber-runtime cannot import radix-mir (FC18), so the transport consumes the
//! X1 mirrored facts ([`TransferOperationMirror`], [`MirroredDtype`],
//! [`MirroredStorageLayout`], [`TransportPathMirror`],
//! [`TransferDirectionMirror`]) and never inference/training vocabulary (S5 —
//! no tokens/sequences/KV/experts).

use crate::bound_plan::LogicalPartitionId;
use crate::execution_transaction::{
    BackendError, MirroredDtype, MirroredStorageLayout, TransferDirectionMirror,
    TransferOperationMirror, TransferRef, TransportPathMirror,
};
use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

// --- T1 measured constants --------------------------------------------------

/// T1 small-copy threshold: below 16 KiB the fixed per-copy latency is
/// ~1.3–2.9 µs (md0-transport §2.2).
pub const T1_SMALL_COPY_BYTES: u64 = 16 * 1024;

/// T1 measured fixed per-copy latency below 16 KiB — the conservative floor
/// of the measured 1.3–2.9 µs band (md0-transport §2.2).
pub const T1_SMALL_COPY_LATENCY_NANOS: u64 = 1_300;

/// T1 measured per-copy latency at 64 KiB — the ~5–6 µs band
/// (md0-transport §2.2). Used for every copy above the 16 KiB threshold
/// (conservative upper step).
pub const T1_LARGE_COPY_LATENCY_NANOS: u64 = 5_000;

/// T1 measured H2D bandwidth — the conservative floor of the measured
/// 9.8–11.6 GB/s band (md0-transport §2.2; raw/host-staging-bench.log).
pub const T1_H2D_BYTES_PER_SEC: u64 = 9_800_000_000;

/// T1 measured D2H bandwidth — the conservative floor of the measured
/// 11.0–12.3 GB/s band (md0-transport §2.2).
pub const T1_D2H_BYTES_PER_SEC: u64 = 11_000_000_000;

/// T1 measured combined BIDI bandwidth — the conservative floor of the
/// measured 12.1–14.1 GB/s band (md0-transport §2.2).
pub const T1_BIDI_BYTES_PER_SEC: u64 = 12_100_000_000;

/// The T1 measured host-staging rates (pharos RTX 5070; FC5). Only measured
/// host-staging rates may feed a budget — never an assumed P2P/NVLink rate
/// (T2 §8.6; C2 §1.2 topology gate).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeasuredRates {
    /// H2D bytes/second (T1 §2.2 H2D column, conservative floor).
    pub h2d_bytes_per_sec: u64,
    /// D2H bytes/second (T1 §2.2 D2H column, conservative floor).
    pub d2h_bytes_per_sec: u64,
    /// Combined BIDI bytes/second (T1 §2.2 BIDI column, conservative floor).
    pub bidi_bytes_per_sec: u64,
}

impl MeasuredRates {
    /// The T1 measured constants.
    #[must_use]
    pub const fn t1() -> Self {
        Self {
            h2d_bytes_per_sec: T1_H2D_BYTES_PER_SEC,
            d2h_bytes_per_sec: T1_D2H_BYTES_PER_SEC,
            bidi_bytes_per_sec: T1_BIDI_BYTES_PER_SEC,
        }
    }

    /// The rate for one direction (BIDI = the combined rate).
    #[must_use]
    pub const fn rate_for(self, direction: TransferDirectionMirror) -> u64 {
        match direction {
            TransferDirectionMirror::H2D => self.h2d_bytes_per_sec,
            TransferDirectionMirror::D2H => self.d2h_bytes_per_sec,
            TransferDirectionMirror::BIDI => self.bidi_bytes_per_sec,
        }
    }

    /// The rates with one direction replaced by a recorded fixture-measured
    /// rate (T2 §8.6 re-budget).
    #[must_use]
    pub const fn with_rate(mut self, direction: TransferDirectionMirror, rate: u64) -> Self {
        match direction {
            TransferDirectionMirror::H2D => self.h2d_bytes_per_sec = rate,
            TransferDirectionMirror::D2H => self.d2h_bytes_per_sec = rate,
            TransferDirectionMirror::BIDI => self.bidi_bytes_per_sec = rate,
        }
        self
    }
}

/// The T2 §4.2 transfer-time estimate at the measured rates:
/// `t(S) = L(S) + S / B_dir`, where `L(S)` is the T1 fixed per-copy latency
/// for a block of size `S` and `B_dir` is the measured per-direction rate.
/// This is the budget-time accounting basis — never an assumed peer rate.
#[must_use]
pub fn expected_copy_time_nanos(
    bytes: u64,
    direction: TransferDirectionMirror,
    rates: MeasuredRates,
) -> u64 {
    let fixed = if bytes <= T1_SMALL_COPY_BYTES {
        T1_SMALL_COPY_LATENCY_NANOS
    } else {
        T1_LARGE_COPY_LATENCY_NANOS
    };
    let transfer = bytes
        .saturating_mul(1_000_000_000)
        .checked_div(rates.rate_for(direction))
        .unwrap_or(0);
    fixed.saturating_add(transfer)
}

// --- byte range + declared/actual transfer facts ----------------------------

/// One half-open byte range of the source value being moved — the byte part
/// of the transfer identity (T2 §8.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ByteRange {
    /// The inclusive start offset within the source value.
    pub offset: u64,
    /// The number of bytes moved.
    pub length: u64,
}

impl ByteRange {
    /// A half-open byte range.
    #[must_use]
    pub const fn new(offset: u64, length: u64) -> Self {
        Self { offset, length }
    }

    /// The exclusive end offset (`offset + length`; the caller validates it
    /// against the source size before copy).
    #[must_use]
    pub const fn end(self) -> u64 {
        self.offset + self.length
    }
}

/// The **declared** facts of one typed/ranged transfer — the runtime-side
/// carrier of the portable logical plan's facts (built from a
/// [`TransferOperationMirror`] via [`TransferSpec::from_mirror`], or authored
/// directly by a backend). The transport surface is exactly the
/// [`TransportPathMirror`] **admissibility** label; the *selected* transport
/// is never part of this type (S4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferSpec {
    transfer_ref: TransferRef,
    source_owner: LogicalPartitionId,
    destination: LogicalPartitionId,
    range: ByteRange,
    direction: TransferDirectionMirror,
    dtype: MirroredDtype,
    layout: MirroredStorageLayout,
    generation: u64,
    path_label: TransportPathMirror,
    timeout: Duration,
}

impl TransferSpec {
    /// A declared typed/ranged transfer. `generation` is the content
    /// generation the transfer moves (mirroring the producer generation);
    /// `timeout` is the declared per-copy timeout (S4).
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        transfer_ref: TransferRef,
        source_owner: LogicalPartitionId,
        destination: LogicalPartitionId,
        range: ByteRange,
        direction: TransferDirectionMirror,
        dtype: MirroredDtype,
        layout: MirroredStorageLayout,
        generation: u64,
        timeout: Duration,
    ) -> Self {
        Self {
            transfer_ref,
            source_owner,
            destination,
            range,
            direction,
            dtype,
            layout,
            generation,
            path_label: TransportPathMirror::HostStaged,
            timeout,
        }
    }

    /// The declared facts derived from a portable logical transfer mirror:
    /// the range is the mirror's full byte contract (`0..byte_count`) and the
    /// generation is the mirror's producer generation. The selected transport
    /// is **not** carried from the mirror (S4) — the mirror only contributes
    /// the admissibility path label.
    #[must_use]
    pub fn from_mirror(mirror: &TransferOperationMirror, timeout: Duration) -> Self {
        Self {
            transfer_ref: mirror.id().clone(),
            source_owner: mirror.source().clone(),
            destination: mirror.destination().clone(),
            range: ByteRange::new(0, mirror.byte_count()),
            direction: mirror.direction(),
            dtype: mirror.element_dtype(),
            layout: mirror.layout(),
            generation: mirror.producer_generation(),
            path_label: mirror.path_label(),
            timeout,
        }
    }

    /// The transfer identity.
    #[must_use]
    pub fn transfer_ref(&self) -> &TransferRef {
        &self.transfer_ref
    }

    /// The declared source owner (the value's declared owner).
    #[must_use]
    pub fn source_owner(&self) -> &LogicalPartitionId {
        &self.source_owner
    }

    /// The declared destination partition.
    #[must_use]
    pub fn destination(&self) -> &LogicalPartitionId {
        &self.destination
    }

    /// The declared byte range of the move.
    #[must_use]
    pub const fn range(&self) -> ByteRange {
        self.range
    }

    /// The declared direction.
    #[must_use]
    pub const fn direction(&self) -> TransferDirectionMirror {
        self.direction
    }

    /// The declared logical dtype.
    #[must_use]
    pub const fn dtype(&self) -> MirroredDtype {
        self.dtype
    }

    /// The declared logical storage layout.
    #[must_use]
    pub const fn layout(&self) -> MirroredStorageLayout {
        self.layout
    }

    /// The declared content generation the transfer moves.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// The transport-**admissibility** label (v1 = `{host-staged}`) — the
    /// only transport surface the declared/portable side carries (S4).
    #[must_use]
    pub const fn path_label(&self) -> TransportPathMirror {
        self.path_label
    }

    /// The declared per-copy timeout (S4).
    #[must_use]
    pub const fn timeout(&self) -> Duration {
        self.timeout
    }
}

/// The **actual** facts of the source content a copy would move, supplied by
/// the backend/host at copy time. The typed/ranged validation compares these
/// against the [`TransferSpec`] before any byte is staged or moved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceValue {
    owner: LogicalPartitionId,
    dtype: MirroredDtype,
    layout: MirroredStorageLayout,
    generation: u64,
    bytes: Vec<u8>,
}

impl SourceValue {
    /// The actual source content: its declared owner, dtype, layout, content
    /// generation, and full byte content.
    #[must_use]
    pub fn new(
        owner: LogicalPartitionId,
        dtype: MirroredDtype,
        layout: MirroredStorageLayout,
        generation: u64,
        bytes: Vec<u8>,
    ) -> Self {
        Self {
            owner,
            dtype,
            layout,
            generation,
            bytes,
        }
    }

    /// The actual owner of the content.
    #[must_use]
    pub fn owner(&self) -> &LogicalPartitionId {
        &self.owner
    }

    /// The actual dtype of the content.
    #[must_use]
    pub const fn dtype(&self) -> MirroredDtype {
        self.dtype
    }

    /// The actual layout of the content.
    #[must_use]
    pub const fn layout(&self) -> MirroredStorageLayout {
        self.layout
    }

    /// The actual content generation.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// The total byte size of the content (the bounds check is against this).
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.bytes.len() as u64
    }

    /// The full content bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The bytes of one range. The range must have passed
    /// [`validate_before_copy`] (bounds-checked before any copy).
    #[must_use]
    pub fn slice(&self, range: ByteRange) -> Vec<u8> {
        debug_assert!(
            range.end() <= self.total_bytes(),
            "range must be validated before slicing"
        );
        self.bytes[range.offset as usize..range.end() as usize].to_vec()
    }
}

// --- fail-before-copy rejection ---------------------------------------------

/// A typed/ranged rejection — the diagnostic naming the **violated class**
/// and the **exact failing fact**, emitted before any byte is staged or moved
/// (T2 §8.2; exit-gate bullet 2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferRejection {
    /// The declared logical dtype does not match the source content.
    DtypeMismatch {
        /// The rejected transfer.
        transfer: TransferRef,
        /// The declared dtype.
        declared: MirroredDtype,
        /// The actual source dtype.
        actual: MirroredDtype,
    },
    /// The declared logical layout does not match the source content.
    LayoutMismatch {
        /// The rejected transfer.
        transfer: TransferRef,
        /// The declared layout.
        declared: MirroredStorageLayout,
        /// The actual source layout.
        actual: MirroredStorageLayout,
    },
    /// The declared byte range exceeds the source content's total size.
    RangeOutOfBounds {
        /// The rejected transfer.
        transfer: TransferRef,
        /// The declared range.
        range: ByteRange,
        /// The actual source size in bytes.
        source_bytes: u64,
    },
    /// The declared content generation does not match the source content.
    GenerationMismatch {
        /// The rejected transfer.
        transfer: TransferRef,
        /// The declared generation.
        declared: u64,
        /// The actual source content generation.
        actual: u64,
    },
    /// The declared source owner does not match the actual content owner.
    OwnerMismatch {
        /// The rejected transfer.
        transfer: TransferRef,
        /// The declared owner.
        declared: LogicalPartitionId,
        /// The actual content owner.
        actual: LogicalPartitionId,
    },
    /// The declared destination does not match the destination the copy
    /// would land on.
    DestinationMismatch {
        /// The rejected transfer.
        transfer: TransferRef,
        /// The declared destination.
        declared: LogicalPartitionId,
        /// The actual destination partition.
        actual: LogicalPartitionId,
    },
}

impl TransferRejection {
    /// The violated class, as a short diagnostic spelling.
    #[must_use]
    pub fn class(&self) -> &'static str {
        match self {
            Self::DtypeMismatch { .. } => "dtype mismatch",
            Self::LayoutMismatch { .. } => "layout mismatch",
            Self::RangeOutOfBounds { .. } => "out-of-bounds range",
            Self::GenerationMismatch { .. } => "generation mismatch",
            Self::OwnerMismatch { .. } => "owner mismatch",
            Self::DestinationMismatch { .. } => "destination mismatch",
        }
    }
}

impl std::fmt::Display for TransferRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DtypeMismatch {
                transfer,
                declared,
                actual,
            } => write!(
                f,
                "dtype mismatch: transfer {transfer} declared {} but the source content is {}",
                declared.spelling(),
                actual.spelling()
            ),
            Self::LayoutMismatch {
                transfer,
                declared,
                actual,
            } => write!(
                f,
                "layout mismatch: transfer {transfer} declared {} but the source content is {}",
                declared.spelling(),
                actual.spelling()
            ),
            Self::RangeOutOfBounds {
                transfer,
                range,
                source_bytes,
            } => write!(
                f,
                "out-of-bounds range: transfer {transfer} declares range {range:?} which exceeds the source's {source_bytes} bytes"
            ),
            Self::GenerationMismatch {
                transfer,
                declared,
                actual,
            } => write!(
                f,
                "generation mismatch: transfer {transfer} declares generation {declared} but the source content is generation {actual}"
            ),
            Self::OwnerMismatch {
                transfer,
                declared,
                actual,
            } => write!(
                f,
                "owner mismatch: transfer {transfer} declares owner {declared} but the source content belongs to {actual}"
            ),
            Self::DestinationMismatch {
                transfer,
                declared,
                actual,
            } => write!(
                f,
                "destination mismatch: transfer {transfer} declares destination {declared} but the copy would land on {actual}"
            ),
        }
    }
}

/// The fail-before-copy typed/ranged validation (T2 §8.2). Compares the
/// declared facts against the actual source content and the actual
/// destination **before any staging is allocated or any byte is moved**.
/// Rejection order is deterministic: dtype, layout, bounds, generation,
/// owner, destination.
pub fn validate_before_copy(
    spec: &TransferSpec,
    source: &SourceValue,
    destination: &LogicalPartitionId,
) -> Result<(), TransferRejection> {
    if spec.dtype() != source.dtype() {
        return Err(TransferRejection::DtypeMismatch {
            transfer: spec.transfer_ref().clone(),
            declared: spec.dtype(),
            actual: source.dtype(),
        });
    }
    if spec.layout() != source.layout() {
        return Err(TransferRejection::LayoutMismatch {
            transfer: spec.transfer_ref().clone(),
            declared: spec.layout(),
            actual: source.layout(),
        });
    }
    if spec.range().end() > source.total_bytes() {
        return Err(TransferRejection::RangeOutOfBounds {
            transfer: spec.transfer_ref().clone(),
            range: spec.range(),
            source_bytes: source.total_bytes(),
        });
    }
    if spec.generation() != source.generation() {
        return Err(TransferRejection::GenerationMismatch {
            transfer: spec.transfer_ref().clone(),
            declared: spec.generation(),
            actual: source.generation(),
        });
    }
    if spec.source_owner() != source.owner() {
        return Err(TransferRejection::OwnerMismatch {
            transfer: spec.transfer_ref().clone(),
            declared: spec.source_owner().clone(),
            actual: source.owner().clone(),
        });
    }
    if spec.destination() != destination {
        return Err(TransferRejection::DestinationMismatch {
            transfer: spec.transfer_ref().clone(),
            declared: spec.destination().clone(),
            actual: destination.clone(),
        });
    }
    Ok(())
}

// --- transfer errors (the coordinator aborts on these) -----------------------

/// A transfer-level error the coordinator aborts on (Q8 fail-closed; the
/// `ExecutionTransaction` surfaces the converted [`BackendError`] and
/// releases/retires every affected resource with no partial publication).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferError {
    /// A typed/ranged mismatch rejected the transfer **before copy**.
    Rejected(TransferRejection),
    /// The copy exceeded its declared per-copy timeout.
    Timeout {
        /// The timed-out transfer.
        transfer: TransferRef,
        /// The declared per-copy timeout (S4).
        declared_timeout: Duration,
        /// The measured elapsed time.
        elapsed_nanos: u64,
    },
    /// The copy failed mid-flight (device/runtime error).
    Failed {
        /// The failed transfer.
        transfer: TransferRef,
        /// What failed, as reported by the runtime.
        detail: String,
    },
    /// A peer transfer was requested on a directed pair that has not been
    /// measured and admitted — NOT ATTEMPTED (T2 §6 per-pair flip rule).
    PeerNotAdmitted {
        /// The unmeasured directed pair.
        pair: DirectedPair,
    },
    /// A peer transfer on an admitted pair is still not attempted in this
    /// campaign: real peer execution requires a real same-host ≥2-GPU
    /// topology (lane queue §5b). Never fabricated.
    PeerNotAttempted {
        /// The admitted directed pair.
        pair: DirectedPair,
        /// Why the copy is not attempted.
        detail: String,
    },
    /// The copy would exceed the declared transfer budget (T2 §8.6).
    BudgetExceeded {
        /// The rejected transfer.
        transfer: TransferRef,
        /// The declared budget bytes.
        budget_bytes: u64,
        /// The bytes already used.
        used_bytes: u64,
        /// The bytes this copy needs.
        needed_bytes: u64,
    },
}

impl TransferError {
    /// Convert into the coordinator's [`BackendError`] vocabulary. A timed-out
    /// copy surfaces `BackendError::Timeout`; every other transfer error
    /// surfaces `BackendError::Operation` (the coordinator aborts on any
    /// backend error with no partial publication — MD3-X1, Q8).
    #[must_use]
    pub fn into_backend_error(self, partition: LogicalPartitionId) -> BackendError {
        match self {
            Self::Rejected(rejection) => BackendError::operation(
                partition,
                format!("typed/ranged transfer rejected before copy: {rejection}"),
            ),
            Self::Timeout {
                transfer,
                declared_timeout,
                elapsed_nanos,
            } => BackendError::timeout(
                partition,
                format!(
                    "transfer {transfer} exceeded its declared {declared_timeout:?} timeout ({elapsed_nanos} ns elapsed)"
                ),
            ),
            Self::Failed { transfer, detail } => BackendError::operation(
                partition,
                format!("transfer {transfer} failed: {detail}"),
            ),
            Self::PeerNotAdmitted { pair } => BackendError::operation(
                partition,
                format!("peer transfer on unmeasured directed pair {pair} is not admitted (T2 §6 — NOT ATTEMPTED)"),
            ),
            Self::PeerNotAttempted { pair, detail } => BackendError::operation(
                partition,
                format!("peer transfer on admitted pair {pair} is not attempted: {detail}"),
            ),
            Self::BudgetExceeded {
                transfer,
                budget_bytes,
                used_bytes,
                needed_bytes,
            } => BackendError::operation(
                partition,
                format!(
                    "transfer {transfer} exceeds the declared transfer budget (used {used_bytes} of {budget_bytes} bytes, needs {needed_bytes} more)"
                ),
            ),
        }
    }
}

impl std::fmt::Display for TransferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rejected(rejection) => write!(f, "{rejection}"),
            Self::Timeout {
                transfer,
                declared_timeout,
                elapsed_nanos,
            } => write!(
                f,
                "transfer {transfer} timed out after {elapsed_nanos} ns (declared {declared_timeout:?})"
            ),
            Self::Failed { transfer, detail } => {
                write!(f, "transfer {transfer} failed: {detail}")
            }
            Self::PeerNotAdmitted { pair } => write!(
                f,
                "peer transfer on unmeasured directed pair {pair} rejected (NOT ATTEMPTED)"
            ),
            Self::PeerNotAttempted { pair, detail } => {
                write!(f, "peer transfer on pair {pair} not attempted: {detail}")
            }
            Self::BudgetExceeded {
                transfer,
                budget_bytes,
                used_bytes,
                needed_bytes,
            } => write!(
                f,
                "transfer {transfer} exceeds the declared transfer budget ({used_bytes}/{budget_bytes} bytes used, needs {needed_bytes})"
            ),
        }
    }
}

// --- directed peer pairs (T2 §6 per-pair flip rule) --------------------------

/// One ordered directed pair `source → destination` of the peer-path
/// admission registry. A **self** pair (`i → i`) is not a P2P row (T1 §2.1).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DirectedPair {
    source: LogicalPartitionId,
    destination: LogicalPartitionId,
}

impl DirectedPair {
    /// A directed pair from `source` to `destination`.
    #[must_use]
    pub fn new(source: LogicalPartitionId, destination: LogicalPartitionId) -> Self {
        Self {
            source,
            destination,
        }
    }

    /// The source partition.
    #[must_use]
    pub fn source(&self) -> &LogicalPartitionId {
        &self.source
    }

    /// The destination partition.
    #[must_use]
    pub fn destination(&self) -> &LogicalPartitionId {
        &self.destination
    }
}

impl std::fmt::Display for DirectedPair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}->{}", self.source, self.destination)
    }
}

/// A **measured** directed pair — the record that may flip a pair to peer
/// copies (T2 §6: the flip requires a `cuDeviceCanAccessPeer`-style probe, a
/// measured directional bandwidth + latency with the T1 §7 receipt
/// conventions, and the measured numbers replacing the host-staging numbers
/// for that pair's transfers). No such measurement exists on the acceptance
/// host (`device_count=1`), so no pair is admitted there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerPairMeasurement {
    pair: DirectedPair,
    probe_provenance: String,
    directional_bandwidth_bytes_per_sec: u64,
    latency_nanos: u64,
    evidence_reference: String,
}

impl PeerPairMeasurement {
    /// A measured pair record.
    #[must_use]
    pub fn new(
        pair: DirectedPair,
        probe_provenance: impl Into<String>,
        directional_bandwidth_bytes_per_sec: u64,
        latency_nanos: u64,
        evidence_reference: impl Into<String>,
    ) -> Self {
        Self {
            pair,
            probe_provenance: probe_provenance.into(),
            directional_bandwidth_bytes_per_sec,
            latency_nanos,
            evidence_reference: evidence_reference.into(),
        }
    }

    /// The measured pair.
    #[must_use]
    pub fn pair(&self) -> &DirectedPair {
        &self.pair
    }

    /// The probe provenance (T1 §7 conventions).
    #[must_use]
    pub fn probe_provenance(&self) -> &str {
        &self.probe_provenance
    }

    /// The measured directional bandwidth.
    #[must_use]
    pub const fn directional_bandwidth_bytes_per_sec(&self) -> u64 {
        self.directional_bandwidth_bytes_per_sec
    }

    /// The measured latency.
    #[must_use]
    pub const fn latency_nanos(&self) -> u64 {
        self.latency_nanos
    }

    /// The evidence reference (raw artifact / receipt per T1 §7).
    #[must_use]
    pub fn evidence_reference(&self) -> &str {
        &self.evidence_reference
    }
}

/// Why a pair measurement was rejected for admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairAdmissionError {
    /// A self pair is not a P2P row (T1 §2.1).
    SelfPair {
        /// The rejected pair.
        pair: DirectedPair,
    },
    /// The measurement lacks a directional bandwidth and/or latency.
    MissingBandwidthOrLatency {
        /// The rejected pair.
        pair: DirectedPair,
    },
    /// The measurement lacks an evidence reference (T1 §7 receipt
    /// conventions — a flip without evidence is never recorded).
    MissingEvidence {
        /// The rejected pair.
        pair: DirectedPair,
    },
}

/// The per-directed-pair admission registry (T2 §6 flip rule — **never a
/// global switch**). Admission records a measured pair; the selection
/// function [`select_copy_path`] reads it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PeerPairRegistry {
    admitted: BTreeMap<DirectedPair, PeerPairMeasurement>,
}

impl PeerPairRegistry {
    /// An empty registry — no pair is admitted.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Admit one measured directed pair. A self pair, a measurement without
    /// a directional bandwidth/latency, or a measurement without an evidence
    /// reference is rejected (T2 §6; T1 §7).
    pub fn admit(&mut self, measurement: PeerPairMeasurement) -> Result<(), PairAdmissionError> {
        let pair = measurement.pair().clone();
        if pair.source() == pair.destination() {
            return Err(PairAdmissionError::SelfPair { pair });
        }
        if measurement.directional_bandwidth_bytes_per_sec() == 0
            || measurement.latency_nanos() == 0
        {
            return Err(PairAdmissionError::MissingBandwidthOrLatency { pair });
        }
        if measurement.evidence_reference().is_empty() {
            return Err(PairAdmissionError::MissingEvidence { pair });
        }
        self.admitted.insert(pair, measurement);
        Ok(())
    }

    /// Whether the directed pair was measured and admitted.
    #[must_use]
    pub fn is_admitted(&self, pair: &DirectedPair) -> bool {
        self.admitted.contains_key(pair)
    }

    /// The measurement record of an admitted pair.
    #[must_use]
    pub fn measurement(&self, pair: &DirectedPair) -> Option<&PeerPairMeasurement> {
        self.admitted.get(pair)
    }

    /// The admitted pairs, in stable order.
    #[must_use]
    pub fn admitted_pairs(&self) -> BTreeSet<DirectedPair> {
        self.admitted.keys().cloned().collect()
    }
}

/// The **selected** copy path of one transfer — a runtime fact that records
/// to the transaction receipt (S4), never to the portable logical plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CopyPath {
    /// Pinned host memory ↔ device over `PCIe` — the admitted path (T2 §2).
    HostStaged,
    /// A per-directed-pair peer copy — only after measurement + admission
    /// (T2 §6); NOT ATTEMPTED on the acceptance host.
    Peer,
}

impl CopyPath {
    /// The path spelling (the "labeled" part of a labeled + timed copy).
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::HostStaged => "host-staged",
            Self::Peer => "peer",
        }
    }
}

/// The T2 §6 per-pair flip rule: a directed pair that was measured and
/// admitted selects [`CopyPath::Peer`]; every other pair stays
/// [`CopyPath::HostStaged`]. There is **no global switch** — the flip is per
/// directed pair.
#[must_use]
pub fn select_copy_path(pair: &DirectedPair, registry: &PeerPairRegistry) -> CopyPath {
    if registry.is_admitted(pair) {
        CopyPath::Peer
    } else {
        CopyPath::HostStaged
    }
}

// --- staging / streams / events ---------------------------------------------

/// Stable opaque id of one pinned host staging buffer.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StagingBufferId(u64);

impl StagingBufferId {
    /// Build a staging buffer id from its raw value.
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// The raw id value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for StagingBufferId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "staging:{}", self.0)
    }
}

/// One pinned host staging buffer allocation. Staging buffers inside a
/// partition budget are accounted at **full size**; pinned host allocations
/// do not consume device budget (T1 §2.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagingBufferRecord {
    /// The buffer's stable id.
    pub id: StagingBufferId,
    /// The capacity in bytes (the full size of the in-flight copy).
    pub capacity_bytes: u64,
    /// Always true for host staging — pinned host memory (T1 §2.3).
    pub pinned: bool,
}

/// The pinned-host staging pool of the host-staged adapter. Buffers are
/// allocated at copy start (in-flight staging, accounted at full size) and
/// released at copy end.
#[derive(Debug, Clone, Default)]
pub struct StagingPool {
    next_id: u64,
    allocations: u64,
    active: BTreeMap<StagingBufferId, StagingBufferRecord>,
}

impl StagingPool {
    /// An empty staging pool.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate a pinned host staging buffer of `bytes` capacity.
    #[must_use]
    pub fn allocate(&mut self, bytes: u64) -> StagingBufferRecord {
        self.next_id += 1;
        self.allocations += 1;
        let record = StagingBufferRecord {
            id: StagingBufferId::new(self.next_id),
            capacity_bytes: bytes,
            pinned: true,
        };
        self.active.insert(record.id.clone(), record.clone());
        record
    }

    /// Release a staging buffer.
    pub fn release(&mut self, id: StagingBufferId) {
        self.active.remove(&id);
    }

    /// The currently in-flight staging buffers.
    #[must_use]
    pub fn active_buffers(&self) -> &BTreeMap<StagingBufferId, StagingBufferRecord> {
        &self.active
    }

    /// The number of staging buffers ever allocated (a fail-before-copy
    /// proof: a rejection allocates nothing).
    #[must_use]
    pub const fn allocations(&self) -> u64 {
        self.allocations
    }

    /// The number of in-flight staging buffers.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.active.len()
    }
}

/// Stable opaque id of one engine (stream/queue) the copy ran on.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EngineId(u64);

impl EngineId {
    /// Build an engine id from its raw value.
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// The raw id value.
    #[must_use]
    pub const fn get(&self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for EngineId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "engine:{}", self.0)
    }
}

/// Stable opaque id of one completion event the copy recorded.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EventId(u64);

impl EventId {
    /// Build an event id from its raw value.
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// The raw id value.
    #[must_use]
    pub const fn get(&self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for EventId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "event:{}", self.0)
    }
}

// --- the S4 selected-transport record ----------------------------------------

/// The actual selected transport of one executed copy — the S4 receipt
/// record (copy path, staging buffers, streams/queues/events, timeout/failure
/// policy, bytes, timing). Produced only by a [`TransportAdapter`] at copy
/// time; never part of the portable logical plan (S4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectedTransferRecord {
    /// The transfer identity.
    pub transfer_ref: TransferRef,
    /// The selected copy path.
    pub copy_path: CopyPath,
    /// The pinned host staging buffer used (full size, pinned).
    pub staging: StagingBufferRecord,
    /// The stream/queue the copy ran on.
    pub engine: EngineId,
    /// The completion event the copy recorded.
    pub event: EventId,
    /// The declared per-copy timeout policy.
    pub timeout: Duration,
    /// The exact bytes moved.
    pub bytes: u64,
    /// The wall time of the copy.
    pub elapsed_nanos: u64,
    /// The budget-time estimate at the measured rates.
    pub expected_nanos: u64,
    /// The direction of the copy.
    pub direction: TransferDirectionMirror,
    /// The destination partition the bytes landed on.
    pub destination: LogicalPartitionId,
}

/// The outcome of one executed copy: the S4 record plus the exact bytes
/// landed at the destination (byte-exact accounting).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferOutcome {
    /// The selected-transport record.
    pub record: SelectedTransferRecord,
    /// The bytes landed at the destination (byte-identical to the validated
    /// source range).
    pub destination_bytes: Vec<u8>,
}

// --- declared transfer budget ------------------------------------------------

/// The plan's **declared transfer budget** (C2 §1.2 transfer budget: bytes
/// and time). Every copy accumulates against it at the measured rates
/// (T2 §8.6); this is distinct from the class-6 reservation the transaction
/// holds at prepare (X1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferBudget {
    /// The declared budget bytes.
    pub bytes: u64,
    /// The declared budget time (nanos).
    pub time_nanos: u64,
}

impl TransferBudget {
    /// A declared transfer budget.
    #[must_use]
    pub const fn declared(bytes: u64, time_nanos: u64) -> Self {
        Self { bytes, time_nanos }
    }
}

/// One recorded fixture rate observation (T2 §8.6): a materially different
/// fixture-measured rate is recorded and re-budgeted — never replaced by an
/// assumed P2P/NVLink rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateObservation {
    /// The measured direction.
    pub direction: TransferDirectionMirror,
    /// The measured bytes moved.
    pub bytes: u64,
    /// The measured elapsed time.
    pub elapsed_nanos: u64,
    /// The measured bytes/second.
    pub measured_bytes_per_sec: u64,
}

// --- the adapter trait -------------------------------------------------------

/// The transport adapter surface (T2 §8; MD3-T1). [`HostStagedAdapter`] is
/// the admitted implementation; [`PeerAdapter`] is the explicit NOT-ATTEMPTED
/// peer path (T2 §6 per-pair flip rule). A copy is always typed/ranged
/// validated **before** any byte is staged or moved, and a failed/timed-out
/// copy surfaces a [`TransferError`] the coordinator aborts on.
pub trait TransportAdapter {
    /// Execute one typed/ranged transfer. Rejects before copy on any
    /// dtype/layout/bounds/generation/owner/destination mismatch; surfaces
    /// [`TransferError::Timeout`] / [`TransferError::Failed`] /
    /// [`TransferError::BudgetExceeded`] / peer-admission errors.
    fn copy(
        &mut self,
        spec: &TransferSpec,
        source: &SourceValue,
        destination: &LogicalPartitionId,
    ) -> Result<TransferOutcome, TransferError>;

    /// The recorded selected-transfer records (the S4 receipt input).
    fn selected_transfer_records(&self) -> &[SelectedTransferRecord];
}

// --- the host-staged adapter -------------------------------------------------

/// The **host-staged** transport adapter (admitted, T2 §2): pinned-host
/// staging, **labeled + timed** (no silent host staging, T2 §7), byte
/// accounting at the T1 measured constants (FC5), budget discipline at the
/// measured rates (T2 §8.6), and a declared per-copy timeout (S4).
///
/// In the unit layer the copy is the byte-exact staged move of the validated
/// source range into pinned host staging and out to the destination; the
/// [`SelectedTransferRecord`] records the actual selected transport for the
/// S4 [`TransportReceipt`]. `simulated_delay` / `simulated_failure` are
/// test-injection seams for the timeout/failure policy.
#[derive(Debug, Clone)]
pub struct HostStagedAdapter {
    declared_budget: TransferBudget,
    used_bytes: u64,
    used_time_nanos: u64,
    staging_pool: StagingPool,
    rates: MeasuredRates,
    next_engine: u64,
    next_event: u64,
    records: Vec<SelectedTransferRecord>,
    rate_observations: Vec<RateObservation>,
    simulated_delay: Duration,
    simulated_failure: Option<String>,
}

impl HostStagedAdapter {
    /// A host-staged adapter over the declared transfer budget, accounting at
    /// the T1 measured rates.
    #[must_use]
    pub fn new(declared_budget: TransferBudget) -> Self {
        Self {
            declared_budget,
            used_bytes: 0,
            used_time_nanos: 0,
            staging_pool: StagingPool::new(),
            rates: MeasuredRates::t1(),
            next_engine: 0,
            next_event: 0,
            records: Vec::new(),
            rate_observations: Vec::new(),
            simulated_delay: Duration::ZERO,
            simulated_failure: None,
        }
    }

    /// Test-injection seam: make every copy take `delay` wall time (the
    /// timeout policy check is against the measured elapsed). `Duration::ZERO`
    /// (the default) runs the copy immediately.
    pub fn set_simulated_delay(&mut self, delay: Duration) {
        self.simulated_delay = delay;
    }

    /// Test-injection seam: make every copy fail mid-flight with `detail`
    /// (surfaces [`TransferError::Failed`]). `None` (the default) runs the
    /// copy normally.
    pub fn set_simulated_failure(&mut self, detail: Option<String>) {
        self.simulated_failure = detail;
    }

    /// The declared transfer budget.
    #[must_use]
    pub const fn budget(&self) -> TransferBudget {
        self.declared_budget
    }

    /// The bytes accumulated against the declared budget.
    #[must_use]
    pub const fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    /// The budget-time accumulated at the measured rates.
    #[must_use]
    pub const fn used_time_nanos(&self) -> u64 {
        self.used_time_nanos
    }

    /// The rates currently used for budget accounting.
    #[must_use]
    pub const fn rates(&self) -> MeasuredRates {
        self.rates
    }

    /// The recorded fixture rate observations (T2 §8.6).
    #[must_use]
    pub fn rate_observations(&self) -> &[RateObservation] {
        &self.rate_observations
    }

    /// The recorded selected-transfer records (S4), in copy order.
    #[must_use]
    pub fn selected_transfer_records(&self) -> &[SelectedTransferRecord] {
        &self.records
    }

    /// The staging pool (fail-before-copy proof: a rejection allocates
    /// nothing).
    #[must_use]
    pub fn staging_pool(&self) -> &StagingPool {
        &self.staging_pool
    }

    /// Record a materially different **fixture-measured** rate and re-budget
    /// at it (T2 §8.6). The recorded rate replaces the assumed constant for
    /// that direction's future accounting — never an assumed P2P/NVLink rate.
    pub fn record_fixture_measurement(
        &mut self,
        direction: TransferDirectionMirror,
        bytes: u64,
        elapsed_nanos: u64,
    ) {
        if elapsed_nanos == 0 {
            return;
        }
        let measured = bytes.saturating_mul(1_000_000_000) / elapsed_nanos;
        self.rate_observations.push(RateObservation {
            direction,
            bytes,
            elapsed_nanos,
            measured_bytes_per_sec: measured,
        });
        self.rates = self.rates.with_rate(direction, measured);
    }

    /// The S4 selected-transport receipt records of this adapter: the actual
    /// selected transports (path/staging/events/timeout/bytes/timing) plus
    /// the budget accounting at the measured rates. This is the
    /// selected-transport section of the transaction receipt (S4).
    #[must_use]
    pub fn transport_receipt(&self) -> TransportReceipt {
        TransportReceipt {
            records: self.records.clone(),
            budget_bytes: self.declared_budget.bytes,
            budget_time_nanos: self.declared_budget.time_nanos,
            used_bytes: self.used_bytes,
            used_time_nanos: self.used_time_nanos,
            rates: self.rates,
            rate_observations: self.rate_observations.clone(),
        }
    }

    /// Execute the labeled + timed staged copy (T2 §7) for a validated spec:
    /// pinned host staging allocated at copy start (accounted at full size,
    /// T1 §2.3), the byte-exact move of the validated range, and the S4
    /// timeout/failure policy against the measured elapsed. The in-flight
    /// staging is released on **every** path; validation and budget checks
    /// happen before this helper runs (a rejection allocates nothing).
    /// Returns the staging record (for the S4 selected-transfer record), the
    /// destination content, and the measured elapsed time.
    fn run_staged_copy(
        &mut self,
        spec: &TransferSpec,
        source: &SourceValue,
        bytes: u64,
    ) -> Result<(StagingBufferRecord, Vec<u8>, Duration), TransferError> {
        let staging = self.staging_pool.allocate(bytes);

        let start = Instant::now();
        if !self.simulated_delay.is_zero() {
            std::thread::sleep(self.simulated_delay);
        }
        let destination_bytes = match &self.simulated_failure {
            Some(detail) => {
                self.staging_pool.release(staging.id.clone());
                return Err(TransferError::Failed {
                    transfer: spec.transfer_ref().clone(),
                    detail: detail.clone(),
                });
            }
            None => source.slice(spec.range()),
        };
        let elapsed = start.elapsed();
        self.staging_pool.release(staging.id.clone());

        if elapsed > spec.timeout() {
            return Err(TransferError::Timeout {
                transfer: spec.transfer_ref().clone(),
                declared_timeout: spec.timeout(),
                elapsed_nanos: elapsed.as_nanos() as u64,
            });
        }
        Ok((staging, destination_bytes, elapsed))
    }
}

impl TransportAdapter for HostStagedAdapter {
    fn copy(
        &mut self,
        spec: &TransferSpec,
        source: &SourceValue,
        destination: &LogicalPartitionId,
    ) -> Result<TransferOutcome, TransferError> {
        // 1. Typed/ranged validation — fail before copy, before any staging
        //    allocation and before any byte is moved.
        validate_before_copy(spec, source, destination).map_err(TransferError::Rejected)?;

        let bytes = spec.range().length;
        let direction = spec.direction();

        // 2. Budget discipline at the measured rates (T2 §8.6): the copy
        //    must fit the declared transfer budget (bytes and time).
        let expected = expected_copy_time_nanos(bytes, direction, self.rates);
        if self.used_bytes.saturating_add(bytes) > self.declared_budget.bytes
            || self.used_time_nanos.saturating_add(expected) > self.declared_budget.time_nanos
        {
            return Err(TransferError::BudgetExceeded {
                transfer: spec.transfer_ref().clone(),
                budget_bytes: self.declared_budget.bytes,
                used_bytes: self.used_bytes,
                needed_bytes: bytes,
            });
        }

        // 3. The labeled + timed staged copy (T2 §7 — no silent host
        //    staging) with the S4 timeout/failure policy. In-flight staging
        //    is allocated at copy start and released on every path inside
        //    the helper.
        let (staging, destination_bytes, elapsed) = self.run_staged_copy(spec, source, bytes)?;

        // 4. Account the budget at the measured rates (T2 §8.6).
        self.used_bytes = self.used_bytes.saturating_add(bytes);
        self.used_time_nanos = self.used_time_nanos.saturating_add(expected);

        // 5. Label the selected transport and record it (S4) — the actual
        //    selected transport records to the receipt, never to the
        //    portable logical plan.
        self.next_engine += 1;
        self.next_event += 1;
        let record = SelectedTransferRecord {
            transfer_ref: spec.transfer_ref().clone(),
            copy_path: CopyPath::HostStaged,
            staging,
            engine: EngineId::new(self.next_engine),
            event: EventId::new(self.next_event),
            timeout: spec.timeout(),
            bytes,
            elapsed_nanos: elapsed.as_nanos() as u64,
            expected_nanos: expected,
            direction,
            destination: destination.clone(),
        };
        self.records.push(record.clone());
        Ok(TransferOutcome {
            record,
            destination_bytes,
        })
    }

    fn selected_transfer_records(&self) -> &[SelectedTransferRecord] {
        &self.records
    }
}

// --- the explicit peer path (NOT ATTEMPTED) ----------------------------------

/// The explicit **peer** transport (T2 §6 — NOT ATTEMPTED until a directed
/// pair is measured and admitted; never a global switch). Typed/ranged
/// validation still runs **before** the admission check (fail-before-copy),
/// then a peer transfer on an **unmeasured** pair is rejected with
/// [`TransferError::PeerNotAdmitted`]. Even an admitted pair is never copied
/// by this adapter: real peer execution requires a real same-host ≥2-GPU
/// topology (lane queue §5b), and this campaign never fabricates a peer pass
/// — the flip is recorded, the execution row stays NOT ATTEMPTED.
#[derive(Debug, Clone, Default)]
pub struct PeerAdapter {
    registry: PeerPairRegistry,
}

impl PeerAdapter {
    /// A peer adapter with an empty admission registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Admit one measured directed pair (per-pair flip rule — T2 §6).
    pub fn admit_pair(
        &mut self,
        measurement: PeerPairMeasurement,
    ) -> Result<(), PairAdmissionError> {
        self.registry.admit(measurement)
    }

    /// Whether a directed pair was measured and admitted.
    #[must_use]
    pub fn pair_admitted(&self, pair: &DirectedPair) -> bool {
        self.registry.is_admitted(pair)
    }

    /// The admission registry.
    #[must_use]
    pub fn registry(&self) -> &PeerPairRegistry {
        &self.registry
    }
}

impl TransportAdapter for PeerAdapter {
    fn copy(
        &mut self,
        spec: &TransferSpec,
        source: &SourceValue,
        destination: &LogicalPartitionId,
    ) -> Result<TransferOutcome, TransferError> {
        // Fail-before-copy typed/ranged validation first.
        validate_before_copy(spec, source, destination).map_err(TransferError::Rejected)?;

        let pair = DirectedPair::new(spec.source_owner().clone(), destination.clone());

        // Per-directed-pair admission check (T2 §6): a peer transfer on an
        // unmeasured pair is rejected — NOT ATTEMPTED.
        if !self.registry.is_admitted(&pair) {
            return Err(TransferError::PeerNotAdmitted { pair });
        }

        // An admitted pair's flip is recorded, but the copy itself is still
        // NOT ATTEMPTED: real peer execution needs a real same-host ≥2-GPU
        // topology (lane queue §5b). No fabricated peer pass.
        Err(TransferError::PeerNotAttempted {
            pair,
            detail: "real peer copies require a real same-host ≥2-GPU topology (lane queue §5b); never fabricated".to_owned(),
        })
    }

    // The peer path never executes a copy (NOT ATTEMPTED), so there is
    // never a selected-transport record (S4) — always empty.
    fn selected_transfer_records(&self) -> &[SelectedTransferRecord] {
        &[]
    }
}

// --- the S4 selected-transport receipt ---------------------------------------

/// The S4 selected-transport section of the transaction receipt: the **actual
/// selected transport** records (copy path, staging buffers,
/// streams/queues/events, timeout/failure policy, bytes, timing) plus the
/// transfer-budget accounting at the measured rates. Landed at MD3-T1 (X1's
/// base `TransactionReceipt` doc: "the selected-transport records (S4) land
/// at MD3-T1"); the coordinator folds this section into the
/// `TransactionReceipt`. The portable logical plan never carries it (S4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransportReceipt {
    /// The selected-transfer records, in copy order.
    pub records: Vec<SelectedTransferRecord>,
    /// The declared transfer-budget bytes.
    pub budget_bytes: u64,
    /// The declared transfer-budget time (nanos).
    pub budget_time_nanos: u64,
    /// The bytes accumulated against the budget.
    pub used_bytes: u64,
    /// The budget-time accumulated at the measured rates.
    pub used_time_nanos: u64,
    /// The rates used for accounting (T1 measured; re-budgeted on recorded
    /// fixture measurements — never an assumed P2P/NVLink rate).
    pub rates: MeasuredRates,
    /// The recorded fixture rate observations (T2 §8.6).
    pub rate_observations: Vec<RateObservation>,
}

#[cfg(test)]
#[path = "transport_test.rs"]
mod tests;
