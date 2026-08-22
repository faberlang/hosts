//! Companion tests for [`super::moe`] (repo companion-test convention).
//!
//! The device-side router selection family mirrors the PM3 host-seam policy
//! and the family Metal module minting fails closed on unminted formats.

use super::*;

fn q8_0_block(value: i8) -> Vec<u8> {
    let mut block = vec![0x00, 0x3c]; // f16 scale = 1.0
    block.extend(std::iter::repeat_n(
        value as u8,
        Q8_0_BLOCK_ELEMENTS as usize,
    ));
    block
}

#[test]
fn router_selection_mirrors_host_seam_policy_on_tied_logits() {
    let rows = 2u64;
    let experts = 4u64;
    let active = 2u64;
    let bind = RouterSelectionBind::packed(
        rows,
        Q8_0_BLOCK_ELEMENTS,
        experts,
        active,
        QuantizedFormat::Q8_0,
        [rows as u32, 1, 1],
    );
    // Two experts share the max logit; the lower id must win the tie.
    let weights = [1i8, 1, 3, 3];
    let packed = weights
        .iter()
        .flat_map(|value| q8_0_block(*value))
        .collect::<Vec<_>>();
    let activation = (0..rows as usize * Q8_0_BLOCK_ELEMENTS as usize)
        .map(|index| (index % Q8_0_BLOCK_ELEMENTS as usize) as f32 * 0.5 + 1.0)
        .collect::<Vec<_>>();
    let mut ids = vec![0u32; (rows * active) as usize];
    let mut out_weights = vec![0.0f32; (rows * active) as usize];

    dispatch_router_selection(
        RouterSelectionKernel::Device,
        &bind,
        &activation,
        &packed,
        &mut ids,
        &mut out_weights,
    )
    .expect("router selection");

    for row in 0..rows as usize {
        let row_ids = &ids[row * active as usize..(row + 1) * active as usize];
        assert_eq!(row_ids, &[2, 3], "lower id must win the equal-logit tie");
        let row_weights = &out_weights[row * active as usize..(row + 1) * active as usize];
        let dot = activation
            [row * Q8_0_BLOCK_ELEMENTS as usize..(row + 1) * Q8_0_BLOCK_ELEMENTS as usize]
            .iter()
            .sum::<f32>();
        let max_logit = 3.0f32 * dot;
        let expected = [0.5f32, 0.5];
        assert_eq!(
            row_weights, &expected,
            "softmax over the tied selected experts (max logit {max_logit})"
        );
    }
}

#[test]
fn router_selection_is_byte_deterministic_on_ties() {
    let bind = RouterSelectionBind::packed(
        1,
        Q8_0_BLOCK_ELEMENTS,
        3,
        2,
        QuantizedFormat::Q8_0,
        [1, 1, 1],
    );
    let packed = [1i8, 1, 2]
        .iter()
        .flat_map(|value| q8_0_block(*value))
        .collect::<Vec<_>>();
    let activation = vec![1.0f32; Q8_0_BLOCK_ELEMENTS as usize];
    let mut first_ids = vec![0u32; 2];
    let mut first_weights = vec![0.0f32; 2];
    let mut second_ids = vec![0u32; 2];
    let mut second_weights = vec![0.0f32; 2];
    for (ids, weights) in [
        (&mut first_ids, &mut first_weights),
        (&mut second_ids, &mut second_weights),
    ] {
        dispatch_router_selection(
            RouterSelectionKernel::Device,
            &bind,
            &activation,
            &packed,
            ids,
            weights,
        )
        .expect("router selection");
    }
    assert_eq!(first_ids, second_ids);
    assert_eq!(first_weights, second_weights);
}

#[test]
fn router_selection_non_finite_logit_fails_closed() {
    let bind = RouterSelectionBind::packed(
        1,
        Q8_0_BLOCK_ELEMENTS,
        2,
        1,
        QuantizedFormat::Q8_0,
        [1, 1, 1],
    );
    // A Q8_0 block whose f16 scale is positive infinity dequantizes to a
    // non-finite logit when the activation is nonzero.
    let mut packed = q8_0_block(1);
    packed[0] = 0x00;
    packed[1] = 0x7c;
    packed.extend(q8_0_block(1));
    let activation = vec![1.0f32; Q8_0_BLOCK_ELEMENTS as usize];
    let mut ids = vec![0u32; 1];
    let mut weights = vec![0.0f32; 1];
    let error = dispatch_router_selection(
        RouterSelectionKernel::Device,
        &bind,
        &activation,
        &packed,
        &mut ids,
        &mut weights,
    )
    .expect_err("non-finite logit must fail closed");
    assert!(matches!(
        error,
        KernelBodyError::NonFiniteLogit { row: 0, expert: 0 }
    ));
}

#[test]
fn moe_family_msl_rejects_unminted_formats() {
    let facts = MoeFamilyMslFacts {
        rows: 1,
        k: 32,
        n: 5,
        experts: 4,
        active: 2,
        format: QuantizedFormat::Q4K,
    };
    let error = moe_family_msl(&facts).expect_err("unminted format must fail closed");
    assert!(matches!(
        error,
        KernelBodyError::InvalidBind(message) if message.contains("Q8_0 only")
    ));
}
