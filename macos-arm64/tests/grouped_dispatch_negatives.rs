//! EXEC02-PM4 grouped-dispatch fail-closed negative matrix (Metal seam).
//!
//! Every mutation probe must reject with the typed `KernelBodyError`
//! diagnostic before any buffer access — never a silent repair, an inferred
//! layout, or an FNV fallback.  The mixed-shape, truncated rank-3 slice, and
//! wrong-layout families are covered here on `GroupedExpertGemmBind`.

use faber_host_macos_arm64::kernel::library::{
    dispatch_grouped_expert_gemm, GroupedExpertGemmBind, GroupedExpertGemmKernel,
    GroupedExpertGemmLayout, KernelBodyError, QuantizedFormat,
};

const ROWS: u64 = 7;
const K: u64 = 32;
const N: u64 = 2;
const EXPERTS: u64 = 2;

fn bind() -> GroupedExpertGemmBind {
    GroupedExpertGemmBind::contiguous(ROWS, K, N, EXPERTS, QuantizedFormat::Q8_0, [N as u32, ROWS as u32, 1])
}

fn uniform_shapes() -> Vec<[u64; 2]> {
    vec![[K, N]; EXPERTS as usize]
}

#[test]
fn grouped_dispatch_mixed_shape_batch_fails_closed() {
    let mut shapes = uniform_shapes();
    shapes[1] = [31, 1];
    let mut output = [f32::NAN; 14];
    let error = dispatch_grouped_expert_gemm(
        GroupedExpertGemmKernel::Packed,
        &bind(),
        &shapes,
        &[],
        &[],
        &mut output,
    )
    .expect_err("mixed-shape expert batch must fail closed");
    assert!(
        matches!(error, KernelBodyError::ShapeMismatch(message) if message.contains("mixed-shape")),
        "typed mixed-shape diagnostic expected, got: {error}"
    );
    assert!(output.iter().all(|value| value.is_nan()));
}

#[test]
fn grouped_dispatch_expert_count_disagreement_fails_closed() {
    let short_shapes = uniform_shapes()[..1].to_vec();
    let mut output = [f32::NAN; 14];
    let error = dispatch_grouped_expert_gemm(
        GroupedExpertGemmKernel::Packed,
        &bind(),
        &short_shapes,
        &[],
        &[],
        &mut output,
    )
    .expect_err("expert-count disagreement must fail closed");
    assert!(
        matches!(
            error,
            KernelBodyError::ShapeMismatch(message)
                if message.contains("expert count disagrees with the bind")
        ),
        "typed expert-count diagnostic expected, got: {error}"
    );
    assert!(output.iter().all(|value| value.is_nan()));
}

#[test]
fn grouped_dispatch_truncated_rank3_slice_fails_closed() {
    let full_bytes = (bind().packed_expert_stride_bytes * EXPERTS) as usize;
    let truncated = vec![0u8; full_bytes - 1];
    let activation = vec![0.0f32; (ROWS * K) as usize];
    let mut output = [f32::NAN; 14];
    let error = dispatch_grouped_expert_gemm(
        GroupedExpertGemmKernel::Packed,
        &bind(),
        &uniform_shapes(),
        &activation,
        &truncated,
        &mut output,
    )
    .expect_err("truncated rank-3 expert slice must fail closed");
    assert!(
        matches!(
            error,
            KernelBodyError::BufferTooShort { buffer: "grouped expert packed weights", required, actual }
                if required == full_bytes as u64 && actual == truncated.len()
        ),
        "typed packed-weight span diagnostic expected, got: {error}"
    );
    assert!(output.iter().all(|value| value.is_nan()));
}

#[test]
fn grouped_dispatch_wrong_layout_expert_tensor_fails_closed() {
    let mut wrong_layout = bind();
    wrong_layout.layout = GroupedExpertGemmLayout::Unsupported;
    let mut output = [f32::NAN; 14];
    let error = dispatch_grouped_expert_gemm(
        GroupedExpertGemmKernel::Packed,
        &wrong_layout,
        &uniform_shapes(),
        &[],
        &[],
        &mut output,
    )
    .expect_err("wrong-layout expert tensor must fail closed");
    assert!(
        matches!(
            error,
            KernelBodyError::InvalidBind(message) if message.contains("layout is not servable")
        ),
        "typed layout diagnostic expected, got: {error}"
    );
    assert!(output.iter().all(|value| value.is_nan()));
}
