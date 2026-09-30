//! GI3-2 — S2 structured backend capability result tests.
//!
//! Families:
//! 1. The initial record decides every op family with a valid tri-state
//!    result; no family is left without a decision.
//! 2. Weight-consuming families (gather / `quantized_matmul` / `logits_head`) are
//!    `supported_with_explicit_conversion` carrying the per-class conversion
//!    plan consumed from the S1 repack selection.
//! 3. Pure-compute families (`rms_normalization` / rope / `causal_masked_softmax`
//!    / `silu_composition`) are `supported_direct` on the frozen GI3-1 recipe
//!    surface.
//! 4. The tri-state is exactly three variants — no silent CPU fallback path
//!    exists.
//! 5. Unexercised capability dimensions are explicitly
//!    `pending_second_representation` (council G3 trim); the assessed
//!    dimensions are the ones the declared f32-conversion path exercises.
//! 6. The conversion plan consumes the S1 selection (same descriptors).
//! 7. Row identity binds to the pinned digest.

use crate::capability::*;
use crate::repack_plan::{PinnedDtype, RepackSelection, RowIdentity};

fn initial_record() -> CapabilityRecord {
    let selection = RepackSelection::initial_declared_f32_conversion(RowIdentity::pinned_row());
    CapabilityRecord::initial(RowIdentity::pinned_row(), &selection)
}

// ---------------------------------------------------------------------------
// 1. Every family decided
// ---------------------------------------------------------------------------

#[test]
fn initial_record_decides_every_family() {
    let record = initial_record();
    assert!(record.every_family_decided(), "no family may be undecided");
    assert_eq!(record.per_family.len(), 7);
    for family in ALL_OP_FAMILIES {
        assert!(
            record.family(family).is_some(),
            "{}: family must have a capability row",
            family.name()
        );
    }
    assert_eq!(record.row, RowIdentity::pinned_row());
}

// ---------------------------------------------------------------------------
// 2. Weight families require the declared conversion
// ---------------------------------------------------------------------------

#[test]
fn weight_families_are_supported_with_explicit_conversion() {
    let selection = RepackSelection::initial_declared_f32_conversion(RowIdentity::pinned_row());
    let record = CapabilityRecord::initial(RowIdentity::pinned_row(), &selection);
    for family in [
        OpFamily::Gather,
        OpFamily::QuantizedMatmul,
        OpFamily::LogitsHead,
    ] {
        let cap = record.family(family).expect("family present");
        let CapabilityResult::SupportedWithExplicitConversion {
            candidates,
            conversion_plan,
        } = &cap.result
        else {
            panic!(
                "{}: expected supported_with_explicit_conversion",
                family.name()
            );
        };
        assert_eq!(
            candidates,
            &vec![Candidate::DeclaredF32Conversion, Candidate::DirectNative]
        );
        // The conversion plan covers exactly the family's consumed classes.
        let consumed = family.consumed_tensor_classes();
        assert_eq!(
            conversion_plan.per_class.len(),
            consumed.len(),
            "{}: conversion plan must cover every consumed class",
            family.name()
        );
        for class in consumed {
            assert!(
                conversion_plan
                    .per_class
                    .iter()
                    .any(|d| d.source_encoding == *class),
                "{}: plan must include a descriptor for {}",
                family.name(),
                class.name()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Pure-compute families are supported directly
// ---------------------------------------------------------------------------

#[test]
fn pure_compute_families_are_supported_direct_on_the_frozen_recipe() {
    let record = initial_record();
    for family in [
        OpFamily::RmsNormalization,
        OpFamily::Rope,
        OpFamily::CausalMaskedSoftmax,
        OpFamily::SiluComposition,
    ] {
        let cap = record.family(family).expect("family present");
        let CapabilityResult::SupportedDirect { candidates } = &cap.result else {
            panic!("{}: expected supported_direct", family.name());
        };
        assert_eq!(candidates, &vec![Candidate::Recipe(family)]);
    }
}

// ---------------------------------------------------------------------------
// 4. No silent CPU fallback
// ---------------------------------------------------------------------------

#[test]
fn tri_state_has_exactly_three_variants_and_no_cpu_fallback() {
    let record = initial_record();
    // The tri-state is the only result vocabulary: every family carries one
    // of the three variants — a silent-CPU-fallback path cannot be expressed.
    for cap in &record.per_family {
        let decided = match &cap.result {
            CapabilityResult::Unsupported { reason } => !reason.is_empty(),
            CapabilityResult::SupportedDirect { .. }
            | CapabilityResult::SupportedWithExplicitConversion { .. } => true,
        };
        assert!(
            decided,
            "{}: invalid tri-state result",
            cap.op_family.name()
        );
    }
}

// ---------------------------------------------------------------------------
// 5. Dimension trim: assessed vs explicitly pending
// ---------------------------------------------------------------------------

#[test]
fn unexercised_dimensions_are_explicitly_pending_second_representation() {
    let record = initial_record();
    for family in ALL_OP_FAMILIES {
        let cap = record.family(family).expect("family present");
        let consumes_quantized = !family.consumed_tensor_classes().is_empty();
        let (assessed_expected, pending_expected) = if consumes_quantized {
            (3, 9) // Q1 legality + Q9 conversion + Q10 direct-native alternative
        } else {
            (2, 10) // Q1 legality + Q9 no-conversion-required
        };
        assert_eq!(
            cap.dimensions.assessed_count(),
            assessed_expected,
            "{}: assessed dimension count",
            family.name()
        );
        assert_eq!(
            cap.dimensions.pending_count(),
            pending_expected,
            "{}: pending dimension count",
            family.name()
        );
        // Q1 — legality is assessed for every family (GI3-1 contract).
        assert_eq!(
            cap.dimensions.legal_for_shapes_and_dtypes,
            CapabilityDimension::Assessed(CapabilityAssessment::LegalForPinnedShapes)
        );
        // Q9 — conversion required only where the family consumes quantized
        // classes.
        let expected_conversion = if consumes_quantized {
            CapabilityDimension::Assessed(CapabilityAssessment::ConversionRequired)
        } else {
            CapabilityDimension::Assessed(CapabilityAssessment::NoConversionRequired)
        };
        assert_eq!(
            cap.dimensions.conversion_or_repack_required,
            expected_conversion,
            "{}: Q9",
            family.name()
        );
        // Q10 — the direct-native alternative is assessed only for weight
        // families.
        let expected_alternatives = if consumes_quantized {
            CapabilityDimension::Assessed(CapabilityAssessment::DirectNativeCandidateCorrect)
        } else {
            CapabilityDimension::PendingSecondRepresentation
        };
        assert_eq!(
            cap.dimensions.alternatives,
            expected_alternatives,
            "{}: Q10",
            family.name()
        );
        // Q2/Q3 — recipe implementation + compiled specialization stay
        // explicitly pending here; GI3-3/GI3-4 populate the per-backend
        // record files.
        assert_eq!(
            cap.dimensions.recipe_implemented,
            CapabilityDimension::PendingSecondRepresentation
        );
        assert_eq!(
            cap.dimensions.compiled_specialization,
            CapabilityDimension::PendingSecondRepresentation
        );
    }
}

// ---------------------------------------------------------------------------
// 6. The conversion plan consumes the S1 selection
// ---------------------------------------------------------------------------

#[test]
fn conversion_plan_consumes_the_s1_selection() {
    let selection = RepackSelection::initial_declared_f32_conversion(RowIdentity::pinned_row());
    let record = CapabilityRecord::initial(RowIdentity::pinned_row(), &selection);
    let cap = record
        .family(OpFamily::QuantizedMatmul)
        .expect("family present");
    let CapabilityResult::SupportedWithExplicitConversion {
        conversion_plan, ..
    } = &cap.result
    else {
        panic!("quantized_matmul must carry a conversion plan");
    };
    // Same descriptors as the S1 selection (the two contracts are one
    // concern).
    for class in [
        PinnedDtype::Q4_K,
        PinnedDtype::Q5_0,
        PinnedDtype::Q6_K,
        PinnedDtype::Q8_0,
    ] {
        let from_plan = conversion_plan
            .per_class
            .iter()
            .find(|d| d.source_encoding == class)
            .expect("plan descriptor present");
        let from_selection = selection
            .f32_conversion_descriptor(class)
            .expect("selection descriptor present");
        assert_eq!(from_plan, from_selection);
        assert_eq!(
            from_plan.byte_extent,
            from_selection.byte_extent,
            "{}: plan and selection agree on byte extent",
            class.name()
        );
    }
}

// ---------------------------------------------------------------------------
// 7. F32-only families need no conversion
// ---------------------------------------------------------------------------

#[test]
fn f32_only_families_need_no_conversion() {
    // RMSNorm consumes the F32 norm weights; F32 is already f32, so the
    // declared f32 conversion is an identity — no lossy conversion is
    // required on its route.
    let record = initial_record();
    let cap = record
        .family(OpFamily::RmsNormalization)
        .expect("family present");
    assert_eq!(
        cap.dimensions.conversion_or_repack_required,
        CapabilityDimension::Assessed(CapabilityAssessment::NoConversionRequired)
    );
    assert_eq!(
        cap.dimensions.legal_for_shapes_and_dtypes,
        CapabilityDimension::Assessed(CapabilityAssessment::LegalForPinnedShapes)
    );
}
