//! Gradient storage ABI (`gradus` — step/accumulator).
//!
//! v1 supports `f32` gradients only, stored as flat `Vec<f32>` with an
//! explicit shape. Functions handle: create, accumulate, read, zero.
//! Storage uses `StableBox` pointer handles (same pattern as [`tensor`]).
//! Accumulate validates shape/dtype match at the byte level; the caller
//! (generated backward code) is responsible for end-to-end correctness.
//!
//! [`gradient_read`] returns a handle to a `#[repr(C)]` [`GradientViewV1`]
//! carrier instead of a raw pointer to the Rust struct — safe to dereference
//! from C/LLVM code.

use super::{RuntimeContext, StableBox};
use crate::abi::{
    FaberRtPtrResultV1, FaberRtStatusV1, STATUS_INVALID_ARGUMENT, STATUS_OK, STATUS_PANIC,
};
use radix_host_abi::FaberRtValueKindV1;
use crate::abi::FaberRtContextV1;
use std::ffi::c_void;

pub(super) struct GradientStorage {
    pub(super) data: Vec<f32>,
    pub(super) shape: Vec<i64>,
}

/// `#[repr(C)]` view carrier returned by [`gradient_read`].
///
/// Exposes gradient data as flat `f32` pointer + element count and shape as
/// `i64` pointer + rank. Safe to dereference from C/LLVM — no Rust layout
/// or Vec internals leak across the ABI boundary.
#[repr(C)]
pub(super) struct GradientViewV1 {
    pub(super) data: *const f32,
    pub(super) len: u64,
    pub(super) shape: *const i64,
    pub(super) rank: u64,
}

fn ffi_ptr(operation: impl FnOnce() -> FaberRtPtrResultV1) -> FaberRtPtrResultV1 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation))
        .unwrap_or(FaberRtPtrResultV1::failure(STATUS_PANIC))
}

fn ffi_status(operation: impl FnOnce() -> FaberRtStatusV1) -> FaberRtStatusV1 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation)).unwrap_or(STATUS_PANIC)
}

fn runtime(context: *mut FaberRtContextV1) -> Option<&'static mut RuntimeContext> {
    (!context.is_null()).then(|| unsafe { &mut *context.cast::<RuntimeContext>() })
}

fn find_gradient(runtime: &RuntimeContext, handle: *mut c_void) -> Option<&GradientStorage> {
    runtime
        .gradients
        .iter()
        .find(|g| std::ptr::eq(g.as_ref(), handle.cast()))
        .map(StableBox::as_ref)
}

fn find_gradient_mut(
    runtime: &mut RuntimeContext,
    handle: *mut c_void,
) -> Option<&mut GradientStorage> {
    runtime
        .gradients
        .iter_mut()
        .find(|g| std::ptr::eq(g.as_ref(), handle.cast()))
        .map(StableBox::as_mut)
}

fn store_gradient(
    runtime: &mut RuntimeContext,
    data: Vec<f32>,
    shape: Vec<i64>,
) -> FaberRtPtrResultV1 {
    let gradient = StableBox::new(GradientStorage { data, shape });
    let handle = gradient.handle();
    runtime.gradients.push(gradient);
    FaberRtPtrResultV1::success(handle)
}

fn checked_element_count(shape: &[i64]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, d| acc.checked_mul(*d as usize))
}

/// Create a zero-filled gradient storage for the given shape and element kind.
///
/// v1 admits only `VALUE_KIND_F32`. Returns an opaque handle or
/// `STATUS_INVALID_ARGUMENT` on null/negative shape, unknown kind, or
/// overflow.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_gradient_create(
    context: *mut FaberRtContextV1,
    shape: *const i64,
    rank: i64,
    kind: FaberRtValueKindV1,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        if kind != radix_host_abi::VALUE_KIND_F32 {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        if shape.is_null() || rank < 0 {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let Ok(rank_usize) = usize::try_from(rank) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank_usize) };
        for dim in shape_slice {
            if *dim < 0 {
                return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
            }
        }
        let Some(element_count) = checked_element_count(shape_slice) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_gradient(runtime, vec![0.0f32; element_count], shape_slice.to_vec())
    })
}

/// Accumulate an incoming gradient tensor into stored gradient storage.
///
/// Validates shape match element-by-element. Accumulation is element-wise add:
/// `stored[i] += incoming[i]`. Returns `STATUS_INVALID_ARGUMENT` on unknown
/// handle, shape mismatch, or null addresses.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_gradient_accumulate(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
    data: *const f32,
    shape: *const i64,
    rank: i64,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = runtime(context) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(gradient) = find_gradient_mut(runtime, handle) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if data.is_null() || shape.is_null() || rank < 0 {
            return STATUS_INVALID_ARGUMENT;
        }
        let Ok(rank_usize) = usize::try_from(rank) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if rank_usize != gradient.shape.len() {
            return STATUS_INVALID_ARGUMENT;
        }
        let incoming_shape = unsafe { std::slice::from_raw_parts(shape, rank_usize) };
        if incoming_shape != gradient.shape.as_slice() {
            return STATUS_INVALID_ARGUMENT;
        }
        let Some(element_count) = checked_element_count(incoming_shape) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let incoming = unsafe { std::slice::from_raw_parts(data, element_count) };
        for (stored, inc) in gradient.data.iter_mut().zip(incoming.iter()) {
            *stored += inc;
        }
        STATUS_OK
    })
}

/// Return a `#[repr(C)]` [`GradientViewV1`] carrier referencing the gradient
/// storage data.
///
/// The returned handle points to a stable allocation with flat `f32` data
/// pointer + length and `i64` shape pointer + rank — safe to dereference from
/// C/LLVM without exposing Rust layout or Vec internals. Returns
/// `FaberRtPtrResultV1::failure` on unknown handle.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_gradient_read(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(gradient) = find_gradient(runtime, handle) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let view = GradientViewV1 {
            data: gradient.data.as_ptr(),
            len: gradient.data.len() as u64,
            shape: gradient.shape.as_ptr(),
            rank: gradient.shape.len() as u64,
        };
        let boxed = StableBox::new(view);
        let handle = boxed.handle();
        runtime.gradient_views.push(boxed);
        FaberRtPtrResultV1::success(handle)
    })
}

/// Zero all elements of the gradient storage.
///
/// Sets every element in the flat buffer to `0.0`. Returns
/// `STATUS_INVALID_ARGUMENT` on unknown handle.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_gradient_zero(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = runtime(context) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(gradient) = find_gradient_mut(runtime, handle) else {
            return STATUS_INVALID_ARGUMENT;
        };
        for slot in &mut gradient.data {
            *slot = 0.0;
        }
        STATUS_OK
    })
}
