//! GI3-2 — S1 per-tensor-class representation/repack plan types.
//!
//! The first of the two CTO shared contracts frozen before backend fan-out
//! (`6badaa01` S1; `correct_before_next_phase`; gi3-delivery §GI3-2). One
//! pinned row only: SmolLM2-360M-Instruct Q4_K_M (model contract v1.0.0 —
//! quant mix Q4_K 16 / Q5_0 176 / Q6_K 16 / Q8_0 17 / F32 65 = 290 tensors).
//!
//! CONTRACT
//! - `QuantizedTensorLayout` stays the **stored-layout authority**: it is
//!   never widened into the physical plan (`repack_plan.rs` only *consumes*
//!   its facts — `PinnedDtype`, `ByteRange`).
//! - The GI2-1 `purpose=cpu-oracle` dequant (`model_widen.rs`) is explicitly
//!   NOT a backend representation; the declared f32 conversion **reuses its
//!   exact dequant semantics** (`ggml/src/ggml-quants.c @ a957b7747`,
//!   [`crate::model_widen::ORACLE_TRANSFORM_IMPL`]) as the initial admitted
//!   representation (Q4 default).
//! - Any conversion carries an independent identity, destination layout,
//!   algorithm family, shape/padding/alignment/byte extent, transformation
//!   implementation + version, output digest, setup time + peak temporary
//!   memory, persistence/cache policy, and executable compatibility
//!   (llama-lessons §7.2 field list).
//! - **Never presented as direct GGUF quantized execution** (campaign
//!   "quantized means native" posture; §7.4 anti-patterns).
//!
//! COUNCIL G3 TRIM — the declared f32-conversion candidate is the only path
//! exercised by this unit. Every descriptor field that a **second
//! representation** would determine (selected backend, persistence/cache
//! policy, executable compatibility) is explicitly marked
//! [`PendingSecondRepresentation`] and NOT populated; that depth belongs to
//! the unit where a second representation actually exists.

use sha2::{Digest, Sha256};

/// Whole-file digest of the pinned SmolLM2 row.
pub const PINNED_SHA256_HEX: &str =
    "2fa3f013dcdd7b99f9b237717fa0b12d75bbb89984cc1274be1471a465bac9c2";

/// Dequantization implementation used by the declared conversion.
pub const ORACLE_TRANSFORM_IMPL: &str = "ggml/src/ggml-quants.c @ a957b7747";

/// Admitted GGUF tensor types for the pinned capability row.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PinnedDtype {
    F32,
    Q5_0,
    Q8_0,
    Q4_K,
    Q6_K,
}

impl PinnedDtype {
    /// Canonical GGUF type name used in capability receipts.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::Q5_0 => "Q5_0",
            Self::Q8_0 => "Q8_0",
            Self::Q4_K => "Q4_K",
            Self::Q6_K => "Q6_K",
        }
    }
}

/// Absolute source-byte range for one packed tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteRange {
    pub start: u64,
    pub end: u64,
}

/// SHA-256 used for content-addressed coordinator plans.
#[must_use]
pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// Byte width of the f32 destination element (IEEE-754 binary32).
pub const F32_ELEMENT_BYTES: u64 = 4;

/// Exact whole-file size of the pinned row (goldens evidence
/// `gi2-dequant-goldens.json` → `model.bytes`).
pub const PINNED_FILE_BYTES: u64 = 270_590_880;

/// Explicit marker for a descriptor field the current candidate does not
/// exercise (council G3 trim): the declared f32-conversion candidate
/// populates only the fields the conversion determines; every other field is
/// explicitly marked pending until a **second representation** actually
/// exists (a native quantized matmul path, a block reinterpretation, …).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PendingSecondRepresentation;

// ---------------------------------------------------------------------------
// Row identity
// ---------------------------------------------------------------------------

/// Identity of the admitted row the repack plan and capability record bind
/// to (model contract v1.0.0; `PINNED_SHA256_HEX`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowIdentity {
    /// Canonical model name.
    pub model_name: &'static str,
    /// Whole-file SHA-256 (hex) — the pinned digest `2fa3f013…bac9c2`.
    pub sha256_hex: &'static str,
    /// Whole-file size in bytes.
    pub file_bytes: u64,
}

impl RowIdentity {
    /// The pinned row's identity (SmolLM2-360M-Instruct Q4_K_M).
    #[must_use]
    pub const fn pinned_row() -> Self {
        Self {
            model_name: "SmolLM2-360M-Instruct Q4_K_M",
            sha256_hex: PINNED_SHA256_HEX,
            file_bytes: PINNED_FILE_BYTES,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-tensor-class facts
// ---------------------------------------------------------------------------

/// Per-tensor-class facts of the admitted row (FC5 + the committed contract
/// metadata evidence `contract-gguf-metadata.txt` → "TENSOR TYPE AGGREGATE").
/// Class-aggregate counts, never per-tensor claims.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorClassFacts {
    /// The closed pinned-row GGML type (F32 / Q5_0 / Q8_0 / Q4_K / Q6_K).
    pub ggml_type: PinnedDtype,
    /// Tensor count of this class in the admitted file.
    pub tensor_count: u64,
    /// Total logical elements across the class.
    pub total_elements: u64,
    /// Total packed bytes across the class (the class's source byte extent).
    pub source_byte_extent: u64,
    /// Representative source tensor of the class (the deterministic-fixture
    /// tensor from `gi2-dequant-goldens.json` — class identity only).
    pub representative_tensor: &'static str,
}

impl TensorClassFacts {
    /// Destination byte extent of the declared f32 conversion
    /// (`total_elements × 4` — tight contiguous f32).
    #[must_use]
    pub const fn f32_destination_byte_extent(self) -> u64 {
        self.total_elements * F32_ELEMENT_BYTES
    }
}

/// The five tensor classes of the pinned row with their contract facts.
#[must_use]
pub const fn pinned_row_class_facts() -> [TensorClassFacts; 5] {
    [
        TensorClassFacts {
            ggml_type: PinnedDtype::Q4_K,
            tensor_count: 16,
            total_elements: 39_321_600,
            source_byte_extent: 22_118_400,
            representative_tensor: "blk.3.ffn_down.weight",
        },
        TensorClassFacts {
            ggml_type: PinnedDtype::Q5_0,
            tensor_count: 176,
            total_elements: 231_014_400,
            source_byte_extent: 158_822_400,
            representative_tensor: "blk.0.attn_q.weight",
        },
        TensorClassFacts {
            ggml_type: PinnedDtype::Q6_K,
            tensor_count: 16,
            total_elements: 39_321_600,
            source_byte_extent: 32_256_000,
            representative_tensor: "blk.0.ffn_down.weight",
        },
        TensorClassFacts {
            ggml_type: PinnedDtype::Q8_0,
            tensor_count: 17,
            total_elements: 52_101_120,
            source_byte_extent: 55_357_440,
            representative_tensor: "blk.0.attn_v.weight",
        },
        TensorClassFacts {
            ggml_type: PinnedDtype::F32,
            tensor_count: 65,
            total_elements: 62_400,
            source_byte_extent: 249_600,
            representative_tensor: "output_norm.weight",
        },
    ]
}

// ---------------------------------------------------------------------------
// Destination / algorithm vocabulary (frozen field list)
// ---------------------------------------------------------------------------

/// Destination layout of a repack. The declared f32 conversion's destination
/// is tight contiguous f32 (row-major, no padding). A second representation's
/// destination (packed/vectorized/padded layouts) adds variants here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DestinationLayout {
    /// Contiguous f32, no padding; `byte_extent = elements × 4`.
    ContiguousF32,
}

/// Element/block interpretation of the destination tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementInterpretation {
    /// Every destination element is a widened IEEE-754 binary32 logical
    /// element (the dequant widening).
    F32LogicalElement,
}

/// Algorithm family of the repack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgorithmFamily {
    /// The declared f32 conversion, reusing the GI2-1 dequant semantics
    /// (`ggml-quants.c` exact integer/half math at the pinned checkout).
    DeclaredF32Conversion,
}

/// Destination shape (class-aggregate for the initial record; per-tensor
/// shapes are resolved at execution from the tensor view).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    /// Total logical elements of the converted destination.
    pub element_count: u64,
}

/// Destination padding. The declared f32 conversion is tight (no padding).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Padding {
    /// Tight contiguous storage — no padding.
    None,
}

// ---------------------------------------------------------------------------
// The repack descriptor
// ---------------------------------------------------------------------------

/// The full repack descriptor (llama-lessons §7.2 / CTO S1 field list):
/// independent identity, destination layout + element/block interpretation,
/// selected backend + algorithm family, shape/padding/alignment/byte extent,
/// transformation implementation + version, output digest, setup time + peak
/// temporary memory, persistence/cache policy, and executable compatibility.
///
/// A class-level descriptor carries the class-aggregate shape/extent plus
/// the deterministic-fixture evidence; per-tensor byte ranges and
/// execution-time digests are resolved at upload (GI3-5) and recorded on the
/// execution receipts. The unexercised selection fields
/// (`backend`, `persistence_policy`, `executable_compatibility`) are
/// explicitly [`PendingSecondRepresentation`] — populated only when a second
/// representation exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepackDescriptor {
    // -- identity (source tensor + encoding + range) --
    /// Source tensor identity — the class's representative fixture tensor
    /// at the contract level; the actual tensor name per tensor at execution.
    pub source_tensor: &'static str,
    /// Source GGML encoding (the closed pinned-row set).
    pub source_encoding: PinnedDtype,
    /// Source packed byte range within the admitted file — **per-tensor**,
    /// resolved at execution from the tensor view (GI3-5); `None` at the
    /// class level is per-tensor variance, not a pending field.
    pub source_byte_range: Option<ByteRange>,
    // -- destination layout + element/block interpretation --
    /// Destination layout of the converted tensor.
    pub destination_layout: DestinationLayout,
    /// Element/block interpretation of the destination.
    pub element_interpretation: ElementInterpretation,
    // -- selected backend + algorithm family --
    /// Selected backend — explicitly pending: the per-backend dimensions
    /// land in the GI3-3/GI3-4-owned record files
    /// (`gi3-representation-record-{metal,cuda}.json`).
    pub backend: PendingSecondRepresentation,
    /// Algorithm family of the repack.
    pub algorithm_family: AlgorithmFamily,
    // -- shape / padding / alignment / byte extent --
    /// Destination shape (class-aggregate at the contract level).
    pub shape: Shape,
    /// Destination padding.
    pub padding: Padding,
    /// Byte alignment of the destination (f32 element width for the declared
    /// conversion).
    pub alignment_bytes: u64,
    /// Destination byte extent (`elements × 4` for the declared conversion).
    pub byte_extent: u64,
    // -- transformation implementation + version --
    /// The pinned transform the conversion reuses (`model_widen.rs`
    /// semantics: `ggml/src/ggml-quants.c @ a957b7747`).
    pub transform_impl: &'static str,
    // -- output digest --
    /// SHA-256 of the converted f32 LE byte stream — deterministic-fixture
    /// evidence only (the committed goldens); `None` on a live
    /// materialization (digests are recorded at GI3-5).
    pub output_digest: Option<[u8; 32]>,
    // -- setup time + peak temporary memory (setup evidence, never decode metrics) --
    /// Tensor-level conversion wall time (µs) — deterministic-fixture
    /// evidence only.
    pub setup_time_us: Option<u64>,
    /// Peak temporary bytes during conversion — deterministic-fixture
    /// evidence only.
    pub peak_temp_bytes: Option<u64>,
    // -- persistence / cache policy + executable compatibility --
    /// Persistence/cache policy (persistent upload, rebuilt, cached) —
    /// explicitly pending until a second representation / the GI3-5 upload
    /// path exists.
    pub persistence_policy: PendingSecondRepresentation,
    /// Executable compatibility (which backends/kernels may execute the
    /// converted bytes) — explicitly pending until the emitters land
    /// (GI3-3/GI3-4).
    pub executable_compatibility: PendingSecondRepresentation,
}

impl RepackDescriptor {
    /// The declared f32-conversion descriptor for one tensor class — the
    /// initial admitted representation (Q4 default; reuses the GI2-1 dequant
    /// semantics). Fixture evidence starts unset; attach it with
    /// [`Self::with_fixture_evidence`] for the deterministic record.
    #[must_use]
    pub const fn declared_f32_conversion(facts: TensorClassFacts) -> Self {
        Self {
            source_tensor: facts.representative_tensor,
            source_encoding: facts.ggml_type,
            source_byte_range: None,
            destination_layout: DestinationLayout::ContiguousF32,
            element_interpretation: ElementInterpretation::F32LogicalElement,
            backend: PendingSecondRepresentation,
            algorithm_family: AlgorithmFamily::DeclaredF32Conversion,
            shape: Shape {
                element_count: facts.total_elements,
            },
            padding: Padding::None,
            alignment_bytes: F32_ELEMENT_BYTES,
            byte_extent: facts.f32_destination_byte_extent(),
            transform_impl: ORACLE_TRANSFORM_IMPL,
            output_digest: None,
            setup_time_us: None,
            peak_temp_bytes: None,
            persistence_policy: PendingSecondRepresentation,
            executable_compatibility: PendingSecondRepresentation,
        }
    }

    /// Attach deterministic-fixture evidence (the golden digest + the
    /// tensor-level fixture-generation timing/peak) on a copy. Setup evidence
    /// only — never a decode metric.
    #[must_use]
    pub fn with_fixture_evidence(
        &self,
        output_digest: [u8; 32],
        setup_time_us: u64,
        peak_temp_bytes: u64,
    ) -> Self {
        Self {
            output_digest: Some(output_digest),
            setup_time_us: Some(setup_time_us),
            peak_temp_bytes: Some(peak_temp_bytes),
            ..self.clone()
        }
    }

    /// Whether the descriptor is for the declared f32 conversion.
    #[must_use]
    pub const fn is_declared_f32_conversion(&self) -> bool {
        matches!(
            self.algorithm_family,
            AlgorithmFamily::DeclaredF32Conversion
        )
    }
}

// ---------------------------------------------------------------------------
// Per-class selection
// ---------------------------------------------------------------------------

/// A selected representation for one tensor class of the admitted row.
///
/// Direct native GGML block execution is a **valid selected candidate** (CTO
/// S1) but is NOT selected for the initial record: only the declared f32
/// conversion is exercised, and any direct-native path is explicitly pending
/// a second representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectedRepresentation {
    /// The declared f32 conversion (the initial admitted representation,
    /// carrying the full repack descriptor).
    DeclaredF32Conversion(RepackDescriptor),
    /// Direct native GGML block execution — valid as a candidate, explicitly
    /// pending until a second representation exists.
    DirectNative(PendingSecondRepresentation),
    /// Explicit `unsupported(reason)`.
    Unsupported { reason: String },
}

impl SelectedRepresentation {
    /// Whether this path claims direct GGUF quantized execution of the
    /// tensor. A converted tensor **never** claims it (campaign "quantized
    /// means native"; §7.4); only the explicitly-pending native path could,
    /// and it is never selected for a converted tensor.
    #[must_use]
    pub fn claims_direct_quantized_execution(&self) -> bool {
        matches!(self, Self::DirectNative(_))
    }
}

/// One tensor class's selection within the plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerClassSelection {
    /// Class facts (identity + aggregate counts).
    pub facts: TensorClassFacts,
    /// The selected representation.
    pub representation: SelectedRepresentation,
}

/// The S1 per-tensor-class representation/repack plan for the admitted row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepackSelection {
    /// Row identity the plan binds to.
    pub row: RowIdentity,
    /// One selection per tensor class (all five classes of the closed set).
    pub per_class: Vec<PerClassSelection>,
}

impl RepackSelection {
    /// The initial admitted selection (Q4 default): every tensor class
    /// selects the declared f32 conversion with the full repack descriptor.
    #[must_use]
    pub fn initial_declared_f32_conversion(row: RowIdentity) -> Self {
        let per_class = pinned_row_class_facts()
            .into_iter()
            .map(|facts| PerClassSelection {
                facts,
                representation: SelectedRepresentation::DeclaredF32Conversion(
                    RepackDescriptor::declared_f32_conversion(facts),
                ),
            })
            .collect();
        Self { row, per_class }
    }

    /// The declared f32-conversion descriptor for one tensor class.
    #[must_use]
    pub fn f32_conversion_descriptor(&self, class: PinnedDtype) -> Option<&RepackDescriptor> {
        self.per_class.iter().find_map(|sel| {
            if sel.facts.ggml_type != class {
                return None;
            }
            match &sel.representation {
                SelectedRepresentation::DeclaredF32Conversion(d) => Some(d),
                _ => None,
            }
        })
    }

    /// The selection for one tensor class.
    #[must_use]
    pub fn class(&self, class: PinnedDtype) -> Option<&PerClassSelection> {
        self.per_class
            .iter()
            .find(|sel| sel.facts.ggml_type == class)
    }

    /// The selected representation of one tensor class.
    #[must_use]
    pub fn representation(&self, class: PinnedDtype) -> Option<&SelectedRepresentation> {
        self.class(class).map(|sel| &sel.representation)
    }

    /// Whether every tensor class carries a selected representation (no
    /// class is left without a decision).
    #[must_use]
    pub fn every_class_selected(&self) -> bool {
        self.per_class.len() == pinned_row_class_facts().len()
            && self.per_class.iter().all(|sel| {
                !matches!(
                    sel.representation,
                    SelectedRepresentation::Unsupported { .. }
                )
            })
    }
}
