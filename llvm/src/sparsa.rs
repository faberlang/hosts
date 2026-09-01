//! Arena-owned sparse tensor carrier for the LLVM host ABI (Stage 4Y).
//!
//! Sparse storage reuses `faber::Sparsa<T>` (COO map of non-default entries).
//! Dense bridges go through the Stage 4V tensor arena carrier.

use super::RuntimeContext;
use super::array::{RuntimeValue, find_array, read_value, write_value};
use super::tensor::{find_tensor, store_tensor_from_parts, tensor_to_runtime_values};
use crate::abi::FaberRtContextV1;
use crate::abi::{
    FaberRtPtrResultV1, FaberRtStatusV1, STATUS_INVALID_ARGUMENT, STATUS_OK, STATUS_PANIC,
};
use faber::{Sparsa, Tensor};
use radix_host_abi::{
    FaberRtValueKindV1, VALUE_KIND_F32, VALUE_KIND_F64, VALUE_KIND_I32, VALUE_KIND_I64,
};
use std::ffi::c_void;
use std::panic::{self, AssertUnwindSafe};

pub(super) enum RuntimeSparse {
    F32(Sparsa<f32>),
    F64(Sparsa<f64>),
    I32(Sparsa<i32>),
    I64(Sparsa<i64>),
}

impl RuntimeSparse {
    fn kind(&self) -> FaberRtValueKindV1 {
        match self {
            Self::F32(_) => VALUE_KIND_F32,
            Self::F64(_) => VALUE_KIND_F64,
            Self::I32(_) => VALUE_KIND_I32,
            Self::I64(_) => VALUE_KIND_I64,
        }
    }

    fn rank(&self) -> i64 {
        match self {
            Self::F32(value) => value.longitudo(),
            Self::F64(value) => value.longitudo(),
            Self::I32(value) => value.longitudo(),
            Self::I64(value) => value.longitudo(),
        }
    }

    fn nonzero(&self) -> Option<i64> {
        match self {
            Self::F32(value) => value.nonnihil().ok(),
            Self::F64(value) => value.nonnihil().ok(),
            Self::I32(value) => value.nonnihil().ok(),
            Self::I64(value) => value.nonnihil().ok(),
        }
    }
}

fn ffi_ptr(operation: impl FnOnce() -> FaberRtPtrResultV1) -> FaberRtPtrResultV1 {
    panic::catch_unwind(AssertUnwindSafe(operation))
        .unwrap_or(FaberRtPtrResultV1::failure(STATUS_PANIC))
}

fn ffi_status(operation: impl FnOnce() -> FaberRtStatusV1) -> FaberRtStatusV1 {
    panic::catch_unwind(AssertUnwindSafe(operation)).unwrap_or(STATUS_PANIC)
}

fn runtime(context: *mut FaberRtContextV1) -> Option<&'static mut RuntimeContext> {
    (!context.is_null()).then(|| unsafe { &mut *context.cast::<RuntimeContext>() })
}

fn store_sparse(runtime: &mut RuntimeContext, value: RuntimeSparse) -> FaberRtPtrResultV1 {
    let boxed = super::StableBox::new(value);
    let handle = boxed.handle();
    runtime.sparses.push(boxed);
    FaberRtPtrResultV1::success(handle)
}

fn find_sparse(runtime: &RuntimeContext, handle: *mut c_void) -> Option<&RuntimeSparse> {
    runtime
        .sparses
        .iter()
        .find(|value| std::ptr::eq(value.as_ref(), handle.cast()))
        .map(super::StableBox::as_ref)
}

fn find_sparse_mut(
    runtime: &mut RuntimeContext,
    handle: *mut c_void,
) -> Option<&mut RuntimeSparse> {
    runtime
        .sparses
        .iter_mut()
        .find(|value| std::ptr::eq(value.as_ref(), handle.cast()))
        .map(super::StableBox::as_mut)
}

fn shape_from_array(array: &super::array::RuntimeArray) -> Option<Vec<i64>> {
    if array.kind != VALUE_KIND_I64 {
        return None;
    }
    array
        .values
        .iter()
        .map(|value| match value {
            RuntimeValue::I64(dim) => Some(dim),
            _ => None,
        })
        .collect()
}

fn indices_from_array(array: &super::array::RuntimeArray) -> Option<Vec<i64>> {
    shape_from_array(array)
}

fn empty_sparse(kind: FaberRtValueKindV1, shape: &[i64]) -> Option<RuntimeSparse> {
    Some(match kind {
        VALUE_KIND_F32 => RuntimeSparse::F32(Sparsa::vacua(shape).ok()?),
        VALUE_KIND_F64 => RuntimeSparse::F64(Sparsa::vacua(shape).ok()?),
        VALUE_KIND_I32 => RuntimeSparse::I32(Sparsa::vacua(shape).ok()?),
        VALUE_KIND_I64 => RuntimeSparse::I64(Sparsa::vacua(shape).ok()?),
        _ => return None,
    })
}

/// Empty sparse tensor with explicit shape lista.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_sparse_new(
    context: *mut FaberRtContextV1,
    kind: FaberRtValueKindV1,
    shape: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(shape) = find_array(runtime, shape).and_then(shape_from_array) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(sparse) = empty_sparse(kind, &shape) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_sparse(runtime, sparse)
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_sparse_get(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
    indices: *mut c_void,
    kind: FaberRtValueKindV1,
    output: *mut c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = runtime(context) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(indices) = find_array(runtime, indices).and_then(indices_from_array) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(sparse) = find_sparse(runtime, handle) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if sparse.kind() != kind {
            return STATUS_INVALID_ARGUMENT;
        }
        let value = match sparse {
            RuntimeSparse::F32(value) => value.accipe(&indices).ok().map(RuntimeValue::F32),
            RuntimeSparse::F64(value) => value.accipe(&indices).ok().map(RuntimeValue::F64),
            RuntimeSparse::I32(value) => value.accipe(&indices).ok().map(RuntimeValue::I32),
            RuntimeSparse::I64(value) => value.accipe(&indices).ok().map(RuntimeValue::I64),
        };
        let Some(value) = value else {
            return STATUS_INVALID_ARGUMENT;
        };
        if !(unsafe { write_value(value, output) }) {
            return STATUS_INVALID_ARGUMENT;
        }
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_sparse_set(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
    indices: *mut c_void,
    kind: FaberRtValueKindV1,
    value: *const c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = runtime(context) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(indices) = find_array(runtime, indices).and_then(indices_from_array) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(value) = (unsafe { read_value(kind, value) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(sparse) = find_sparse_mut(runtime, handle) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if sparse.kind() != kind {
            return STATUS_INVALID_ARGUMENT;
        }
        let ok = match (sparse, value) {
            (RuntimeSparse::F32(sparse), RuntimeValue::F32(value)) => {
                sparse.ponde(&indices, value).is_ok()
            }
            (RuntimeSparse::F64(sparse), RuntimeValue::F64(value)) => {
                sparse.ponde(&indices, value).is_ok()
            }
            (RuntimeSparse::I32(sparse), RuntimeValue::I32(value)) => {
                sparse.ponde(&indices, value).is_ok()
            }
            (RuntimeSparse::I64(sparse), RuntimeValue::I64(value)) => {
                sparse.ponde(&indices, value).is_ok()
            }
            _ => false,
        };
        if ok {
            STATUS_OK
        } else {
            STATUS_INVALID_ARGUMENT
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_sparse_nonzero(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
    output: *mut i64,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = runtime(context) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(sparse) = find_sparse(runtime, handle) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(count) = sparse.nonzero() else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(output) = (unsafe { output.as_mut() }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        *output = count;
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_sparse_rank(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
    output: *mut i64,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = runtime(context) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(sparse) = find_sparse(runtime, handle) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(output) = (unsafe { output.as_mut() }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        *output = sparse.rank();
        STATUS_OK
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_sparse_densify(
    context: *mut FaberRtContextV1,
    handle: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(sparse) = find_sparse(runtime, handle) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let densified = match sparse {
            RuntimeSparse::F32(value) => value.densata().ok().map(|dense| {
                (
                    VALUE_KIND_F32,
                    dense.magnitudines(),
                    dense.planata().into_iter().map(RuntimeValue::F32).collect(),
                )
            }),
            RuntimeSparse::F64(value) => value.densata().ok().map(|dense| {
                (
                    VALUE_KIND_F64,
                    dense.magnitudines(),
                    dense.planata().into_iter().map(RuntimeValue::F64).collect(),
                )
            }),
            RuntimeSparse::I32(value) => value.densata().ok().map(|dense| {
                (
                    VALUE_KIND_I32,
                    dense.magnitudines(),
                    dense.planata().into_iter().map(RuntimeValue::I32).collect(),
                )
            }),
            RuntimeSparse::I64(value) => value.densata().ok().map(|dense| {
                (
                    VALUE_KIND_I64,
                    dense.magnitudines(),
                    dense.planata().into_iter().map(RuntimeValue::I64).collect(),
                )
            }),
        };
        let Some((kind, shape, data)) = densified else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_tensor_from_parts(runtime, kind, shape, data)
    })
}

#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_sparse_from_tensor(
    context: *mut FaberRtContextV1,
    tensor: *mut c_void,
    kind: FaberRtValueKindV1,
) -> FaberRtPtrResultV1 {
    ffi_ptr(|| {
        let Some(runtime) = runtime(context) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(tensor) = find_tensor(runtime, tensor) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        if tensor.kind != kind {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let Some((shape, values)) = tensor_to_runtime_values(tensor) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let sparse = match kind {
            VALUE_KIND_F32 => match values
                .iter()
                .map(|value| match value {
                    RuntimeValue::F32(value) => Some(*value),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
                .and_then(|data| Tensor::structa(data, &shape).ok())
            {
                Some(dense) => RuntimeSparse::F32(Sparsa::from_tensor(&dense)),
                None => return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT),
            },
            VALUE_KIND_F64 => match values
                .iter()
                .map(|value| match value {
                    RuntimeValue::F64(value) => Some(*value),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
                .and_then(|data| Tensor::structa(data, &shape).ok())
            {
                Some(dense) => RuntimeSparse::F64(Sparsa::from_tensor(&dense)),
                None => return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT),
            },
            VALUE_KIND_I32 => match values
                .iter()
                .map(|value| match value {
                    RuntimeValue::I32(value) => Some(*value),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
                .and_then(|data| Tensor::structa(data, &shape).ok())
            {
                Some(dense) => RuntimeSparse::I32(Sparsa::from_tensor(&dense)),
                None => return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT),
            },
            VALUE_KIND_I64 => match values
                .iter()
                .map(|value| match value {
                    RuntimeValue::I64(value) => Some(*value),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
                .and_then(|data| Tensor::structa(data, &shape).ok())
            {
                Some(dense) => RuntimeSparse::I64(Sparsa::from_tensor(&dense)),
                None => return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT),
            },
            _ => return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT),
        };
        store_sparse(runtime, sparse)
    })
}
