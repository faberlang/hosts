//! MD1-D1 identity tests: distinctness, ordinal-locator-only equality and
//! ordering, replacement detection, and the health-epoch contract.

use crate::backend::DeviceBackend;
use crate::device_identity::{
    CudaIdentityFacts, DeviceHealthGeneration, DeviceIdentityFacts, DeviceOrdinal, IdentityChange,
    PhysicalDeviceId,
};
use std::cmp::Ordering;

const UUID_A: &str = "GPU-3e017562-9ec3-da9a-962d-b8bd5f9e24be";
const UUID_B: &str = "GPU-11111111-2222-3333-4444-555555555555";
const UUID_C: &str = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";

/// Two `PhysicalDeviceId`s with the same backend and the same ordinal locator
/// but different identity facts are distinct ids (naming contract §1; exit
/// gate: "two same-backend physical devices are distinguishable despite
/// backend and ordinal reuse").
#[test]
fn same_backend_same_ordinal_different_facts_are_distinct() {
    // Both devices sit at ordinal locator 0 — the ordinal is the caller's
    // context and never enters the id.
    let id_a = PhysicalDeviceId::cuda(UUID_A, None);
    let id_b = PhysicalDeviceId::cuda(UUID_B, None);

    assert_eq!(id_a.backend(), DeviceBackend::Cuda);
    assert_eq!(id_b.backend(), DeviceBackend::Cuda);
    assert_ne!(id_a, id_b);
    assert!(id_a < id_b || id_b < id_a); // total order, distinct ids never tie
    assert_eq!(id_b.change_against(&id_a), IdentityChange::Replaced);
    assert_eq!(id_a.change_against(&id_b), IdentityChange::Replaced);
}

/// A differing corroborating driver UUID is itself a changed identity fact.
#[test]
fn differing_driver_uuid_is_a_distinct_identity_fact() {
    let with_driver = PhysicalDeviceId::cuda(
        UUID_A,
        Some("3e017562-9ec3-da9a-962d-b8bd5f9e24be".to_owned()),
    );
    let without_driver = PhysicalDeviceId::cuda(UUID_A, None);

    assert_ne!(with_driver, without_driver);
    assert_eq!(
        with_driver.change_against(&without_driver),
        IdentityChange::Replaced
    );
}

/// CUDA and Metal identity classes are never confused by construction.
#[test]
fn cuda_and_metal_ids_are_distinct_classes() {
    let cuda = PhysicalDeviceId::cuda(UUID_A, None);
    let metal = PhysicalDeviceId::metal("device-registry-1");

    assert_ne!(cuda, metal);
    assert_eq!(metal.backend(), DeviceBackend::Metal);
    assert_eq!(metal.change_against(&cuda), IdentityChange::Replaced);
}

/// Identity equality and ordering ignore the ordinal (ordinal-rename
/// locator-only proof, naming contract §1 / MD-A2).
#[test]
fn identity_equality_and_ordering_ignore_ordinal() {
    let facts = DeviceIdentityFacts::Cuda(CudaIdentityFacts {
        pci_uuid: UUID_A.to_owned(),
        driver_uuid: Some("3e017562-9ec3-da9a-962d-b8bd5f9e24be".to_owned()),
    });
    // The same physical device observed at two different ordinal locators is
    // the same id.
    let at_ordinal_0 = PhysicalDeviceId::from_facts(facts.clone());
    let at_ordinal_1 = PhysicalDeviceId::from_facts(facts);
    assert_eq!(at_ordinal_0, at_ordinal_1);
    assert_eq!(Ord::cmp(&at_ordinal_0, &at_ordinal_1), Ordering::Equal);

    // The locator domain itself compares by value and is independent of ids.
    assert_ne!(DeviceOrdinal::new(0), DeviceOrdinal::new(1));

    // Ordering over ids is total, ordinal-free, and stable under renaming.
    let mut ids = [
        PhysicalDeviceId::cuda(UUID_C, None),
        at_ordinal_0.clone(),
        PhysicalDeviceId::cuda(UUID_A, Some("different-driver".to_owned())),
        at_ordinal_1.clone(),
    ];
    ids.sort();
    assert!(ids.windows(2).all(|w| w[0] <= w[1]));
    assert_eq!(
        ids.iter().filter(|id| **id == at_ordinal_0).count(),
        2,
        "the renamed id still equals the original — ordinal never participates"
    );
}

/// Identical facts at the same ordinal mean the same device (no false
/// replacement).
#[test]
fn replacement_detection_same_facts_is_same_device() {
    let a = PhysicalDeviceId::cuda(UUID_A, None);
    let a_again = PhysicalDeviceId::cuda(UUID_A, None);

    assert_eq!(a.change_against(&a_again), IdentityChange::SameDevice);
    assert_eq!(a.change_against(&a), IdentityChange::SameDevice);
}

/// The health epoch is a monotonic machine-local generation, advanced on any
/// admission-gating change, and always distinct from the semantic
/// `ValueGeneration` domain.
#[test]
fn health_generation_is_a_monotonic_epoch() {
    let gen1 = DeviceHealthGeneration::initial();
    assert_eq!(gen1.get(), 1);
    let gen2 = gen1.advance();
    assert_eq!(gen2.get(), 2);
    let gen3 = gen2.advance();

    assert!(gen1 < gen2 && gen2 < gen3);
    assert!(gen2.is_current(gen2));
    assert!(!gen2.is_stale(gen2));
    assert!(gen1.is_stale(gen2));
    assert!(gen1.is_stale(gen3));
}

/// `DeviceHealthGeneration` is a different type from the FMIR semantic
/// `ValueGeneration` epoch (radix-air / faber `package::device`) — they are
/// never conflated. This test pins the identity-domain accessors.
#[test]
fn health_generation_exposes_raw_epoch_value() {
    let generation = DeviceHealthGeneration::initial().advance().advance();
    assert_eq!(generation.get(), 3);
    assert_eq!(format!("{generation}"), "3");
}
