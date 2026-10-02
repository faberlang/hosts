use super::from_tensor_i32;
use faber::Tensor;
use faber::tensor::{ERR_TENSOR_EDGE_POLICY_INVALID, TensorEdgePolicy};

#[test]
fn tensor_conversion_propagates_unreadable_view_error() {
    let invalid = Tensor::structa(vec![1_i32, 2, 3], &[3])
        .expect("dense tensor")
        .shift(&[1])
        .expect("shifted tensor")
        .limes(TensorEdgePolicy::Custom {
            remap: |_| vec![99],
            transform: None,
        })
        .expect("custom edge mapping");

    assert_eq!(
        from_tensor_i32(&invalid).err(),
        Some(ERR_TENSOR_EDGE_POLICY_INVALID)
    );
}
