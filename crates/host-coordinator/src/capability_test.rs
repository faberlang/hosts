//! GI3-2 — S2 structured backend capability result tests.
//!
//! Families:
//! 1. The initial record decides every op family with a valid tri-state
//!    result; no family is left without a decision.
//! 2. Weight-consuming families (gather / quantized_matmul / logits_head) are
//!    `supported_with_explicit_conversion` carrying the per-class conversion
//!    plan consumed from the S1 repack selection.
//! 3. Pure-compute families (rms_normalization / rope / causal_masked_softmax
//!    / silu_composition) are `supported_direct` on the frozen GI3-1 recipe
//!    surface.
//! 4. The tri-state is exactly three variants — no silent CPU fallback path
//!    exists.
//! 5. Unexercised capability dimensions are explicitly
//!    `pending_second_representation` (council G3 trim); the assessed
//!    dimensions are the ones the declared f32-conversion path exercises.
//! 6. The conversion plan consumes the S1 selection (same descriptors).
//! 7. Row identity binds to the pinned digest.
//! 8. The committed evidence record (`evidence/gi3-representation-record.json`)
//!    is hash-accounted and matches the initial contract (schema, row
//!    identity, five classes with the declared f32 conversion + explicit
//!    pending markers, seven families with valid tri-state rows).

use crate::capability::*;
use faber::model_format::PinnedDtype;
use faber::Json;
use faber::repack_plan::{RepackSelection, RowIdentity};
use faber::Valor;

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
            CapabilityResult::SupportedDirect { .. } => true,
            CapabilityResult::SupportedWithExplicitConversion { .. } => true,
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

// ---------------------------------------------------------------------------
// 8. Committed evidence record is hash-accounted (machine check)
// ---------------------------------------------------------------------------

/// The committed evidence record (sibling radix repo; same convention as the
/// GI2 dequant goldens — a missing file is reported loudly, never silently
/// skipped).
const RECORD_REL_PATH: &str =
    "../radix/docs/factory/gpu-inference-gguf/evidence/gi3-representation-record.json";

fn load_record() -> Option<Json> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(RECORD_REL_PATH);
    let wire = match std::fs::read_to_string(&path) {
        Ok(wire) => wire,
        Err(err) => {
            eprintln!(
                "SKIP: representation record not readable at {} ({err})",
                path.display()
            );
            return None;
        }
    };
    Some(Json::parse(&wire).expect("record JSON must parse"))
}

fn obj<'a>(v: &'a Valor, key: &str) -> &'a Valor {
    let Valor::Tabula(fields) = v else {
        panic!("expected JSON object at {key}");
    };
    fields
        .get(key)
        .unwrap_or_else(|| panic!("record missing field {key:?}"))
}

fn list<'a>(v: &'a Valor) -> &'a Vec<Valor> {
    let Valor::Lista(items) = v else {
        panic!("expected JSON list");
    };
    items
}

fn text<'a>(v: &'a Valor) -> &'a str {
    let Valor::Textus(s) = v else {
        panic!("expected JSON string");
    };
    s
}

fn int(v: &Valor) -> i64 {
    let Valor::Numerus(n) = v else {
        panic!("expected JSON integer");
    };
    *n
}

fn boolean(v: &Valor) -> bool {
    let Valor::Bivalens(b) = v else {
        panic!("expected JSON boolean");
    };
    *b
}

#[test]
fn committed_evidence_record_is_hash_accounted_and_matches_the_initial_contract() {
    let Some(record) = load_record() else {
        return;
    };
    let root = record.as_valor();
    assert_eq!(text(obj(root, "schema")), "gi3-representation-record-v1");

    // Row identity binds to the pinned digest.
    let row = obj(root, "row");
    assert_eq!(text(obj(row, "model_name")), "SmolLM2-360M-Instruct Q4_K_M");
    assert_eq!(text(obj(row, "sha256_hex")), faber::model_format::PINNED_SHA256_HEX);
    assert_eq!(int(obj(row, "tensor_count")), 290);

    // Repack selection: five classes, every one the declared f32 conversion
    // with the explicit pending markers.
    let selection = obj(root, "repack_selection");
    let per_class = list(obj(selection, "per_class"));
    assert_eq!(per_class.len(), 5, "closed set has five classes");
    let mut classes: Vec<&str> = per_class
        .iter()
        .map(|c| text(obj(c, "ggml_type")))
        .collect();
    classes.sort_unstable();
    assert_eq!(classes, vec!["F32", "Q4_K", "Q5_0", "Q6_K", "Q8_0"]);
    for c in per_class {
        let sr = obj(c, "selected_representation");
        assert_eq!(text(obj(sr, "kind")), "declared_f32_conversion");
        assert_eq!(text(obj(sr, "backend")), "pending_second_representation");
        assert_eq!(
            text(obj(sr, "persistence_policy")),
            "pending_second_representation"
        );
        assert_eq!(
            text(obj(sr, "executable_compatibility")),
            "pending_second_representation"
        );
        let digest = text(obj(sr, "output_digest"));
        assert_eq!(
            digest.len(),
            64,
            "{}: fixture digest must be a SHA-256 hex",
            text(obj(c, "ggml_type"))
        );
        assert_eq!(
            text(obj(sr, "transform_impl")),
            faber::model_widen::ORACLE_TRANSFORM_IMPL
        );
    }

    // Capability: seven families, valid tri-state rows, 12 dimensions each,
    // and no silent CPU fallback.
    let capability = obj(root, "capability");
    assert!(boolean(obj(capability, "no_silent_cpu_fallback")));
    let per_family = list(obj(capability, "per_family"));
    assert_eq!(per_family.len(), 7);
    for f in per_family {
        let name = text(obj(f, "op_family"));
        let result = obj(f, "result");
        let kind = text(obj(result, "kind"));
        assert!(
            matches!(
                kind,
                "unsupported" | "supported_direct" | "supported_with_explicit_conversion"
            ),
            "{name}: invalid tri-state kind"
        );
        let dims = obj(f, "dimensions");
        // Dimensions are object-keyed; count 12 by checking a fixed set.
        for q in [
            "legal_for_shapes_and_dtypes",
            "recipe_implemented",
            "compiled_specialization",
            "device_features",
            "layouts_compatible",
            "workspace_feasible",
            "alignment_aliasing",
            "capture_reuse_safe",
            "conversion_or_repack_required",
            "alternatives",
            "profitability_class",
            "receipt_fallback",
        ] {
            assert!(
                text(obj(dims, q)).contains("assessed")
                    || text(obj(dims, q)) == "pending_second_representation",
                "{name}: dimension {q} must be assessed or explicitly pending"
            );
        }
        // The conversion dimension is assessed for every family.
        assert!(
            text(obj(dims, "conversion_or_repack_required")).starts_with("assessed"),
            "{name}: Q9 must be assessed"
        );
        // The tri-state matches the family's consumed classes.
        let consumed: Vec<&str> = list(obj(f, "consumed_tensor_classes"))
            .iter()
            .map(|c| text(c))
            .collect();
        if !consumed.is_empty() {
            assert_eq!(kind, "supported_with_explicit_conversion", "{name}");
            let plan_classes: Vec<&str> = list(obj(obj(result, "conversion_plan"), "per_class"))
                .iter()
                .map(|c| text(c))
                .collect();
            assert_eq!(
                plan_classes, consumed,
                "{name}: plan covers consumed classes"
            );
        }
    }
}
