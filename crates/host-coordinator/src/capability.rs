//! GI3-2 — S2 structured backend capability result types.
//!
//! The second CTO shared contract frozen before backend fan-out (`6badaa01`
//! S2; gi3-delivery §GI3-2): the tri-state result
//! `unsupported(reason) | supported_direct(candidates) |
//! supported_with_explicit_conversion(candidates, conversion_plan)` over the
//! 12-question dimension surface (llama-lessons §8.3), with the initial
//! populated record for the pinned row.
//!
//! CONTRACT
//! - **No silent CPU fallback for an explicit GPU route**: the result is
//!   exactly one of the three variants — an explicit GPU route may fail or
//!   take a declared conversion candidate, never silently fall back.
//! - The conversion dimension consumes the S1 repack plan (`repack_plan.rs`):
//!   the two contracts are one concern and freeze together.
//! - Recipe implementation / compiled-specialization dimensions are marked
//!   pending here and populated by GI3-3/GI3-4 into the **per-backend record
//!   files** (`gi3-representation-record-{metal,cuda}.json`; F2 decision,
//!   audit `112dc81a` — one writer per file).
//!
//! COUNCIL G3 TRIM — only the dimensions the declared f32-conversion path
//! exercises are assessed in the initial record (op/shape/dtype legality;
//! conversion-required; the direct-native alternative); every other dimension
//! is explicitly [`PendingSecondRepresentation`].

use crate::repack_plan::{PinnedDtype, RepackDescriptor, RepackSelection, RowIdentity};

// ---------------------------------------------------------------------------
// Op families
// ---------------------------------------------------------------------------

/// The pinned row's op families: the GI3-1 frozen recipe surface (Gather /
/// RmsNormalization / Rope / CausalMaskedSoftmax + the verified SiLU
/// composition) plus reuse (TiledMatMul) and the tied-head logits projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpFamily {
    /// Embedding/gather over `token_embd.weight` (Q8_0, `[960,49152]`).
    Gather,
    /// `x/sqrt(mean(x²)+eps)·γ` over the F32 norm weights (eps 1e-5).
    RmsNormalization,
    /// Llama-arch NORM consecutive-pair rotation (dim 64, freq_base 100000).
    Rope,
    /// Causal masked row softmax over the attention scores.
    CausalMaskedSoftmax,
    /// `SiLU(x) = x/(1+exp(−x))` elementwise composition (GI3-1 verified).
    SiluComposition,
    /// Quantized projection/matmul (attn Q/K/V/O + FFN up/gate/down).
    QuantizedMatmul,
    /// Full-vocab logits via the tied output head (`token_embd.weight`).
    LogitsHead,
}

/// All seven op families in record order.
pub const ALL_OP_FAMILIES: [OpFamily; 7] = [
    OpFamily::Gather,
    OpFamily::RmsNormalization,
    OpFamily::Rope,
    OpFamily::CausalMaskedSoftmax,
    OpFamily::SiluComposition,
    OpFamily::QuantizedMatmul,
    OpFamily::LogitsHead,
];

impl OpFamily {
    /// Canonical spelling (the record's `op_family` key).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Gather => "gather",
            Self::RmsNormalization => "rms_normalization",
            Self::Rope => "rope",
            Self::CausalMaskedSoftmax => "causal_masked_softmax",
            Self::SiluComposition => "silu_composition",
            Self::QuantizedMatmul => "quantized_matmul",
            Self::LogitsHead => "logits_head",
        }
    }

    /// The tensor classes this family's weight route consumes. Empty for the
    /// pure-compute families (no quantized weight tensors); `F32` norm
    /// weights need no conversion (F32 is already f32).
    #[must_use]
    pub const fn consumed_tensor_classes(self) -> &'static [PinnedDtype] {
        match self {
            Self::Gather | Self::LogitsHead => &[PinnedDtype::Q8_0],
            Self::QuantizedMatmul => &[
                PinnedDtype::Q4_K,
                PinnedDtype::Q5_0,
                PinnedDtype::Q6_K,
                PinnedDtype::Q8_0,
            ],
            Self::RmsNormalization
            | Self::Rope
            | Self::CausalMaskedSoftmax
            | Self::SiluComposition => &[],
        }
    }
}

// ---------------------------------------------------------------------------
// Tri-state result
// ---------------------------------------------------------------------------

/// A candidate for an op family's execution route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Candidate {
    /// The frozen recipe (or reuse) the family executes on (GI3-1 — the
    /// shared-plan surface; emitter bodies pending in GI3-3/GI3-4).
    Recipe(OpFamily),
    /// The declared f32 conversion (the initial admitted weight
    /// representation).
    DeclaredF32Conversion,
    /// Direct native GGML block execution — a correct alternative whose
    /// implementation is explicitly pending a second representation.
    DirectNative,
}

/// The explicit conversion plan an op family's weight route requires — the
/// per-class declared repack descriptors (the S1 repack plan, consumed
/// directly).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversionPlan {
    /// One declared repack descriptor per consumed quantized tensor class.
    pub per_class: Vec<RepackDescriptor>,
}

/// The S2 tri-state structured backend capability result (CTO S2; §8.3).
///
/// No variant exists for silent CPU fallback: an explicit GPU route either
/// is unsupported (with a reason), is directly supported, or requires the
/// declared conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityResult {
    /// The family is unsupported with an explicit reason.
    Unsupported { reason: String },
    /// The family executes directly on the frozen recipe surface.
    SupportedDirect { candidates: Vec<Candidate> },
    /// The family's weight route requires the declared conversion plan.
    SupportedWithExplicitConversion {
        candidates: Vec<Candidate>,
        conversion_plan: ConversionPlan,
    },
}

// ---------------------------------------------------------------------------
// The 12-question dimension surface (§8.3)
// ---------------------------------------------------------------------------

/// An assessed fact for one capability dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityAssessment {
    /// The semantic operation is legal for the pinned shapes/dtypes per the
    /// GI3-1 recipe contract (dimension 1).
    LegalForPinnedShapes,
    /// The family's weight route requires the declared f32 conversion
    /// (dimension 9).
    ConversionRequired,
    /// The family consumes no quantized weight tensors (F32 weights or no
    /// weights); no conversion is required (dimension 9).
    NoConversionRequired,
    /// Direct native block execution remains a correct, not-yet-implemented
    /// alternative (dimension 10).
    DirectNativeCandidateCorrect,
}

/// One dimension of the 12-question surface (§8.3): assessed by the initial
/// record, or explicitly unexercised until a second representation / backend
/// implementation exists (council G3 trim).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityDimension {
    /// The initial record assesses the dimension.
    Assessed(CapabilityAssessment),
    /// Explicitly `pending_second_representation`.
    PendingSecondRepresentation,
}

/// The 12-question dimension surface (llama-lessons §8.3):
///
/// 1. legal for these shapes/dtypes?    7. alignment/alias satisfied?
/// 2. recipe implemented?               8. capture/reuse safe?
/// 3. compiled specialization present?  9. conversion/repack required?
/// 4. device features available?       10. alternatives correct?
/// 5. layouts compatible?              11. profitability class?
/// 6. workspace feasible?              12. receipt/fallback?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilityDimensions {
    /// Q1 — legal for these shapes and dtypes?
    pub legal_for_shapes_and_dtypes: CapabilityDimension,
    /// Q2 — does the backend implement the required recipe?
    pub recipe_implemented: CapabilityDimension,
    /// Q3 — was the specialization compiled into this build?
    pub compiled_specialization: CapabilityDimension,
    /// Q4 — does the physical device expose the features?
    pub device_features: CapabilityDimension,
    /// Q5 — are input/output layouts compatible?
    pub layouts_compatible: CapabilityDimension,
    /// Q6 — is the workspace feasible under the budget?
    pub workspace_feasible: CapabilityDimension,
    /// Q7 — are alignment and alias requirements satisfied?
    pub alignment_aliasing: CapabilityDimension,
    /// Q8 — is the operation safe under the capture/reuse mode?
    pub capture_reuse_safe: CapabilityDimension,
    /// Q9 — is a conversion or repack required?
    pub conversion_or_repack_required: CapabilityDimension,
    /// Q10 — which alternatives remain correct?
    pub alternatives: CapabilityDimension,
    /// Q11 — which candidate is predicted to be profitable?
    pub profitability_class: CapabilityDimension,
    /// Q12 — what decision and fallback appear in the receipt?
    pub receipt_fallback: CapabilityDimension,
}

impl CapabilityDimensions {
    /// The initial record's dimension surface (council G3 trim): only the
    /// dimensions the declared f32-conversion path exercises are assessed —
    /// legality (Q1), conversion-required (Q9), and the direct-native
    /// alternative (Q10, weight families only). Every other dimension is
    /// explicitly [`CapabilityDimension::PendingSecondRepresentation`].
    #[must_use]
    pub fn initial(consumes_quantized_weights: bool) -> Self {
        let conversion = if consumes_quantized_weights {
            CapabilityDimension::Assessed(CapabilityAssessment::ConversionRequired)
        } else {
            CapabilityDimension::Assessed(CapabilityAssessment::NoConversionRequired)
        };
        let alternatives = if consumes_quantized_weights {
            CapabilityDimension::Assessed(CapabilityAssessment::DirectNativeCandidateCorrect)
        } else {
            CapabilityDimension::PendingSecondRepresentation
        };
        Self {
            legal_for_shapes_and_dtypes: CapabilityDimension::Assessed(
                CapabilityAssessment::LegalForPinnedShapes,
            ),
            recipe_implemented: CapabilityDimension::PendingSecondRepresentation,
            compiled_specialization: CapabilityDimension::PendingSecondRepresentation,
            device_features: CapabilityDimension::PendingSecondRepresentation,
            layouts_compatible: CapabilityDimension::PendingSecondRepresentation,
            workspace_feasible: CapabilityDimension::PendingSecondRepresentation,
            alignment_aliasing: CapabilityDimension::PendingSecondRepresentation,
            capture_reuse_safe: CapabilityDimension::PendingSecondRepresentation,
            conversion_or_repack_required: conversion,
            alternatives,
            profitability_class: CapabilityDimension::PendingSecondRepresentation,
            receipt_fallback: CapabilityDimension::PendingSecondRepresentation,
        }
    }

    /// Count of assessed dimensions.
    #[must_use]
    pub fn assessed_count(&self) -> usize {
        self.dimensions()
            .filter(|d| matches!(d, CapabilityDimension::Assessed(_)))
            .count()
    }

    /// Count of explicitly-pending dimensions.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.dimensions()
            .filter(|d| matches!(d, CapabilityDimension::PendingSecondRepresentation))
            .count()
    }

    fn dimensions(&self) -> impl Iterator<Item = &CapabilityDimension> {
        [
            &self.legal_for_shapes_and_dtypes,
            &self.recipe_implemented,
            &self.compiled_specialization,
            &self.device_features,
            &self.layouts_compatible,
            &self.workspace_feasible,
            &self.alignment_aliasing,
            &self.capture_reuse_safe,
            &self.conversion_or_repack_required,
            &self.alternatives,
            &self.profitability_class,
            &self.receipt_fallback,
        ]
        .into_iter()
    }
}

// ---------------------------------------------------------------------------
// The capability record
// ---------------------------------------------------------------------------

/// One op family's structured capability result: the tri-state plus the
/// 12-question dimension surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpFamilyCapability {
    /// The op family.
    pub op_family: OpFamily,
    /// The tri-state result.
    pub result: CapabilityResult,
    /// The 12-question dimension surface.
    pub dimensions: CapabilityDimensions,
}

/// The S2 structured capability record for the admitted row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityRecord {
    /// Row identity the record binds to.
    pub row: RowIdentity,
    /// One result per op family.
    pub per_family: Vec<OpFamilyCapability>,
}

impl CapabilityRecord {
    /// The initial capability record for the pinned row (Q4 default):
    /// weight-consuming families (Gather, QuantizedMatmul, LogitsHead) are
    /// `supported_with_explicit_conversion` on the declared f32 conversion
    /// (consuming the S1 selection); pure-compute families are
    /// `supported_direct` on the frozen GI3-1 recipe surface. No variant
    /// permits a silent CPU fallback.
    ///
    /// # Panics
    ///
    /// Panics if a consumed tensor class has no declared f32 conversion in
    /// `selection` — the conversion dimension only consumes selected plans.
    #[must_use]
    pub fn initial(row: RowIdentity, selection: &RepackSelection) -> Self {
        let per_family = ALL_OP_FAMILIES
            .iter()
            .map(|family| {
                let consumed = family.consumed_tensor_classes();
                let consumes_quantized_weights = !consumed.is_empty();
                let dimensions = CapabilityDimensions::initial(consumes_quantized_weights);
                let result = if consumes_quantized_weights {
                    let per_class = consumed
                        .iter()
                        .map(|class| {
                            selection
                                .f32_conversion_descriptor(*class)
                                .expect("every consumed class selects the declared f32 conversion")
                                .clone()
                        })
                        .collect();
                    CapabilityResult::SupportedWithExplicitConversion {
                        candidates: vec![Candidate::DeclaredF32Conversion, Candidate::DirectNative],
                        conversion_plan: ConversionPlan { per_class },
                    }
                } else {
                    CapabilityResult::SupportedDirect {
                        candidates: vec![Candidate::Recipe(*family)],
                    }
                };
                OpFamilyCapability {
                    op_family: *family,
                    result,
                    dimensions,
                }
            })
            .collect();
        Self { row, per_family }
    }

    /// One op family's capability.
    #[must_use]
    pub fn family(&self, family: OpFamily) -> Option<&OpFamilyCapability> {
        self.per_family.iter().find(|cap| cap.op_family == family)
    }

    /// Whether every family carries a tri-state result (no family is left
    /// without a decision).
    #[must_use]
    pub fn every_family_decided(&self) -> bool {
        self.per_family.len() == ALL_OP_FAMILIES.len()
            && self
                .per_family
                .iter()
                .all(|cap| !matches!(cap.result, CapabilityResult::Unsupported { .. }))
    }
}
