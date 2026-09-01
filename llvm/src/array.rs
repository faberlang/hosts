//! Arena-owned typed arrays for the LLVM host ABI.

use super::RuntimeContext;
use super::option::store_option;
use crate::abi::FaberRtContextV1;
use crate::abi::{
    FaberRtPtrResultV1, FaberRtStatusV1, STATUS_INVALID_ARGUMENT, STATUS_OK, STATUS_PANIC,
};
use radix_host_abi::{
    ARRAY_OPTION_FIRST, ARRAY_OPTION_INDEX, ARRAY_OPTION_LAST, ARRAY_OPTION_REMOVE_FIRST,
    ARRAY_OPTION_REMOVE_LAST, ARRAY_RANGE_DROP_FIRST, ARRAY_RANGE_SLICE, ARRAY_RANGE_TAKE,
    ARRAY_RANGE_TAKE_LAST, FaberRtArrayOptionModeV1, FaberRtArrayRangeModeV1, FaberRtValueKindV1,
    VALUE_KIND_ASCII, VALUE_KIND_F16, VALUE_KIND_F32, VALUE_KIND_F64, VALUE_KIND_I1, VALUE_KIND_I8,
    VALUE_KIND_I16, VALUE_KIND_I32, VALUE_KIND_I64, VALUE_KIND_INSTANS, VALUE_KIND_OPTION_I64,
    VALUE_KIND_PTR, VALUE_KIND_TEXT, VALUE_KIND_U8, VALUE_KIND_U16, VALUE_KIND_U32, VALUE_KIND_U64,
    VALUE_KIND_VALOR,
};
use std::ffi::c_void;
use std::panic::{self, AssertUnwindSafe};

#[derive(Clone, Copy, PartialEq)]
pub(super) enum RuntimeValue {
    I1(u8),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F16(u16),
    F32(f32),
    F64(f64),
    Ptr(*mut c_void),
}

/// Homogeneous backing store for one arena array or tensor.
///
/// Arrays are kind-tagged at the ABI boundary, so a tagged `RuntimeValue`
/// cell wasted a 16-byte enum slot on every element. Typed vectors keep
/// the scalar width (4 bytes for `f32`, 1 byte for `u8`, …) and let
/// extend/flatten/slice memcpy compact payloads.
#[derive(Clone)]
pub(super) enum RuntimeCells {
    I1(Vec<u8>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    F16(Vec<u16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    Ptr(Vec<*mut c_void>),
}

pub(super) struct RuntimeArray {
    pub(super) kind: FaberRtValueKindV1,
    pub(super) values: RuntimeCells,
}

impl RuntimeCells {
    pub(super) fn empty(kind: FaberRtValueKindV1) -> Option<Self> {
        Some(match kind {
            VALUE_KIND_I1 => Self::I1(Vec::new()),
            VALUE_KIND_I8 => Self::I8(Vec::new()),
            VALUE_KIND_I16 => Self::I16(Vec::new()),
            VALUE_KIND_I32 => Self::I32(Vec::new()),
            VALUE_KIND_I64 => Self::I64(Vec::new()),
            VALUE_KIND_U8 => Self::U8(Vec::new()),
            VALUE_KIND_U16 => Self::U16(Vec::new()),
            VALUE_KIND_U32 => Self::U32(Vec::new()),
            VALUE_KIND_U64 => Self::U64(Vec::new()),
            VALUE_KIND_F16 => Self::F16(Vec::new()),
            VALUE_KIND_F32 => Self::F32(Vec::new()),
            VALUE_KIND_F64 => Self::F64(Vec::new()),
            VALUE_KIND_PTR
            | VALUE_KIND_TEXT
            | VALUE_KIND_VALOR
            | VALUE_KIND_OPTION_I64
            | VALUE_KIND_INSTANS
            | VALUE_KIND_ASCII => Self::Ptr(Vec::new()),
            _ => return None,
        })
    }

    pub(super) fn from_values(kind: FaberRtValueKindV1, values: Vec<RuntimeValue>) -> Option<Self> {
        let mut cells = Self::empty(kind)?;
        cells.reserve(values.len());
        for value in values {
            if !cells.push(value) {
                return None;
            }
        }
        Some(cells)
    }

    pub(super) fn repeat(
        kind: FaberRtValueKindV1,
        value: RuntimeValue,
        count: usize,
    ) -> Option<Self> {
        Some(match (kind, value) {
            (VALUE_KIND_I1, RuntimeValue::I1(value)) => Self::I1(vec![value; count]),
            (VALUE_KIND_I8, RuntimeValue::I8(value)) => Self::I8(vec![value; count]),
            (VALUE_KIND_I16, RuntimeValue::I16(value)) => Self::I16(vec![value; count]),
            (VALUE_KIND_I32, RuntimeValue::I32(value)) => Self::I32(vec![value; count]),
            (VALUE_KIND_I64, RuntimeValue::I64(value)) => Self::I64(vec![value; count]),
            (VALUE_KIND_U8, RuntimeValue::U8(value)) => Self::U8(vec![value; count]),
            (VALUE_KIND_U16, RuntimeValue::U16(value)) => Self::U16(vec![value; count]),
            (VALUE_KIND_U32, RuntimeValue::U32(value)) => Self::U32(vec![value; count]),
            (VALUE_KIND_U64, RuntimeValue::U64(value)) => Self::U64(vec![value; count]),
            (VALUE_KIND_F16, RuntimeValue::F16(value)) => Self::F16(vec![value; count]),
            (VALUE_KIND_F32, RuntimeValue::F32(value)) => Self::F32(vec![value; count]),
            (VALUE_KIND_F64, RuntimeValue::F64(value)) => Self::F64(vec![value; count]),
            (
                VALUE_KIND_PTR
                | VALUE_KIND_TEXT
                | VALUE_KIND_VALOR
                | VALUE_KIND_OPTION_I64
                | VALUE_KIND_INSTANS
                | VALUE_KIND_ASCII,
                RuntimeValue::Ptr(value),
            ) => Self::Ptr(vec![value; count]),
            _ => return None,
        })
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::I1(values) | Self::U8(values) => values.len(),
            Self::I8(values) => values.len(),
            Self::I16(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::F16(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::F32(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::U64(values) => values.len(),
            Self::F64(values) => values.len(),
            Self::Ptr(values) => values.len(),
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(super) fn reserve(&mut self, additional: usize) {
        match self {
            Self::I1(values) | Self::U8(values) => values.reserve(additional),
            Self::I8(values) => values.reserve(additional),
            Self::I16(values) => values.reserve(additional),
            Self::U16(values) => values.reserve(additional),
            Self::F16(values) => values.reserve(additional),
            Self::I32(values) => values.reserve(additional),
            Self::U32(values) => values.reserve(additional),
            Self::F32(values) => values.reserve(additional),
            Self::I64(values) => values.reserve(additional),
            Self::U64(values) => values.reserve(additional),
            Self::F64(values) => values.reserve(additional),
            Self::Ptr(values) => values.reserve(additional),
        }
    }

    pub(super) fn get(&self, index: usize) -> Option<RuntimeValue> {
        Some(match self {
            Self::I1(values) => RuntimeValue::I1(*values.get(index)?),
            Self::I8(values) => RuntimeValue::I8(*values.get(index)?),
            Self::I16(values) => RuntimeValue::I16(*values.get(index)?),
            Self::I32(values) => RuntimeValue::I32(*values.get(index)?),
            Self::I64(values) => RuntimeValue::I64(*values.get(index)?),
            Self::U8(values) => RuntimeValue::U8(*values.get(index)?),
            Self::U16(values) => RuntimeValue::U16(*values.get(index)?),
            Self::U32(values) => RuntimeValue::U32(*values.get(index)?),
            Self::U64(values) => RuntimeValue::U64(*values.get(index)?),
            Self::F16(values) => RuntimeValue::F16(*values.get(index)?),
            Self::F32(values) => RuntimeValue::F32(*values.get(index)?),
            Self::F64(values) => RuntimeValue::F64(*values.get(index)?),
            Self::Ptr(values) => RuntimeValue::Ptr(*values.get(index)?),
        })
    }

    pub(super) fn first(&self) -> Option<RuntimeValue> {
        self.get(0)
    }

    pub(super) fn last(&self) -> Option<RuntimeValue> {
        let len = self.len();
        (len > 0).then(|| self.get(len - 1)).flatten()
    }

    pub(super) fn set(&mut self, index: usize, value: RuntimeValue) -> bool {
        match (self, value) {
            (Self::I1(values), RuntimeValue::I1(value)) => {
                values.get_mut(index).map(|slot| *slot = value).is_some()
            }
            (Self::I8(values), RuntimeValue::I8(value)) => {
                values.get_mut(index).map(|slot| *slot = value).is_some()
            }
            (Self::I16(values), RuntimeValue::I16(value)) => {
                values.get_mut(index).map(|slot| *slot = value).is_some()
            }
            (Self::I32(values), RuntimeValue::I32(value)) => {
                values.get_mut(index).map(|slot| *slot = value).is_some()
            }
            (Self::I64(values), RuntimeValue::I64(value)) => {
                values.get_mut(index).map(|slot| *slot = value).is_some()
            }
            (Self::U8(values), RuntimeValue::U8(value)) => {
                values.get_mut(index).map(|slot| *slot = value).is_some()
            }
            (Self::U16(values), RuntimeValue::U16(value)) => {
                values.get_mut(index).map(|slot| *slot = value).is_some()
            }
            (Self::U32(values), RuntimeValue::U32(value)) => {
                values.get_mut(index).map(|slot| *slot = value).is_some()
            }
            (Self::U64(values), RuntimeValue::U64(value)) => {
                values.get_mut(index).map(|slot| *slot = value).is_some()
            }
            (Self::F16(values), RuntimeValue::F16(value)) => {
                values.get_mut(index).map(|slot| *slot = value).is_some()
            }
            (Self::F32(values), RuntimeValue::F32(value)) => {
                values.get_mut(index).map(|slot| *slot = value).is_some()
            }
            (Self::F64(values), RuntimeValue::F64(value)) => {
                values.get_mut(index).map(|slot| *slot = value).is_some()
            }
            (Self::Ptr(values), RuntimeValue::Ptr(value)) => {
                values.get_mut(index).map(|slot| *slot = value).is_some()
            }
            _ => false,
        }
    }

    pub(super) fn push(&mut self, value: RuntimeValue) -> bool {
        match (self, value) {
            (Self::I1(values), RuntimeValue::I1(value)) => values.push(value),
            (Self::I8(values), RuntimeValue::I8(value)) => values.push(value),
            (Self::I16(values), RuntimeValue::I16(value)) => values.push(value),
            (Self::I32(values), RuntimeValue::I32(value)) => values.push(value),
            (Self::I64(values), RuntimeValue::I64(value)) => values.push(value),
            (Self::U8(values), RuntimeValue::U8(value)) => values.push(value),
            (Self::U16(values), RuntimeValue::U16(value)) => values.push(value),
            (Self::U32(values), RuntimeValue::U32(value)) => values.push(value),
            (Self::U64(values), RuntimeValue::U64(value)) => values.push(value),
            (Self::F16(values), RuntimeValue::F16(value)) => values.push(value),
            (Self::F32(values), RuntimeValue::F32(value)) => values.push(value),
            (Self::F64(values), RuntimeValue::F64(value)) => values.push(value),
            (Self::Ptr(values), RuntimeValue::Ptr(value)) => values.push(value),
            _ => return false,
        }
        true
    }

    pub(super) fn pop(&mut self) -> Option<RuntimeValue> {
        Some(match self {
            Self::I1(values) => RuntimeValue::I1(values.pop()?),
            Self::I8(values) => RuntimeValue::I8(values.pop()?),
            Self::I16(values) => RuntimeValue::I16(values.pop()?),
            Self::I32(values) => RuntimeValue::I32(values.pop()?),
            Self::I64(values) => RuntimeValue::I64(values.pop()?),
            Self::U8(values) => RuntimeValue::U8(values.pop()?),
            Self::U16(values) => RuntimeValue::U16(values.pop()?),
            Self::U32(values) => RuntimeValue::U32(values.pop()?),
            Self::U64(values) => RuntimeValue::U64(values.pop()?),
            Self::F16(values) => RuntimeValue::F16(values.pop()?),
            Self::F32(values) => RuntimeValue::F32(values.pop()?),
            Self::F64(values) => RuntimeValue::F64(values.pop()?),
            Self::Ptr(values) => RuntimeValue::Ptr(values.pop()?),
        })
    }

    pub(super) fn remove(&mut self, index: usize) -> Option<RuntimeValue> {
        if index >= self.len() {
            return None;
        }
        Some(match self {
            Self::I1(values) => RuntimeValue::I1(values.remove(index)),
            Self::I8(values) => RuntimeValue::I8(values.remove(index)),
            Self::I16(values) => RuntimeValue::I16(values.remove(index)),
            Self::I32(values) => RuntimeValue::I32(values.remove(index)),
            Self::I64(values) => RuntimeValue::I64(values.remove(index)),
            Self::U8(values) => RuntimeValue::U8(values.remove(index)),
            Self::U16(values) => RuntimeValue::U16(values.remove(index)),
            Self::U32(values) => RuntimeValue::U32(values.remove(index)),
            Self::U64(values) => RuntimeValue::U64(values.remove(index)),
            Self::F16(values) => RuntimeValue::F16(values.remove(index)),
            Self::F32(values) => RuntimeValue::F32(values.remove(index)),
            Self::F64(values) => RuntimeValue::F64(values.remove(index)),
            Self::Ptr(values) => RuntimeValue::Ptr(values.remove(index)),
        })
    }

    pub(super) fn contains(&self, value: &RuntimeValue) -> bool {
        match (self, value) {
            (Self::I1(values), RuntimeValue::I1(value)) => values.contains(value),
            (Self::I8(values), RuntimeValue::I8(value)) => values.contains(value),
            (Self::I16(values), RuntimeValue::I16(value)) => values.contains(value),
            (Self::I32(values), RuntimeValue::I32(value)) => values.contains(value),
            (Self::I64(values), RuntimeValue::I64(value)) => values.contains(value),
            (Self::U8(values), RuntimeValue::U8(value)) => values.contains(value),
            (Self::U16(values), RuntimeValue::U16(value)) => values.contains(value),
            (Self::U32(values), RuntimeValue::U32(value)) => values.contains(value),
            (Self::U64(values), RuntimeValue::U64(value)) => values.contains(value),
            (Self::F16(values), RuntimeValue::F16(value)) => values.contains(value),
            (Self::F32(values), RuntimeValue::F32(value)) => values.contains(value),
            (Self::F64(values), RuntimeValue::F64(value)) => values.contains(value),
            (Self::Ptr(values), RuntimeValue::Ptr(value)) => values.contains(value),
            _ => false,
        }
    }

    pub(super) fn reverse(&mut self) {
        match self {
            Self::I1(values) | Self::U8(values) => values.reverse(),
            Self::I8(values) => values.reverse(),
            Self::I16(values) => values.reverse(),
            Self::U16(values) => values.reverse(),
            Self::F16(values) => values.reverse(),
            Self::I32(values) => values.reverse(),
            Self::U32(values) => values.reverse(),
            Self::F32(values) => values.reverse(),
            Self::I64(values) => values.reverse(),
            Self::U64(values) => values.reverse(),
            Self::F64(values) => values.reverse(),
            Self::Ptr(values) => values.reverse(),
        }
    }

    pub(super) fn sort(&mut self) -> bool {
        match self {
            Self::I8(values) => values.sort_unstable(),
            Self::I16(values) => values.sort_unstable(),
            Self::I32(values) => values.sort_unstable(),
            Self::I64(values) => values.sort_unstable(),
            Self::U8(values) => values.sort_unstable(),
            Self::U16(values) => values.sort_unstable(),
            Self::U32(values) => values.sort_unstable(),
            Self::U64(values) => values.sort_unstable(),
            Self::F32(values) => values.sort_unstable_by(f32::total_cmp),
            Self::F64(values) => values.sort_unstable_by(f64::total_cmp),
            _ => return false,
        }
        true
    }

    pub(super) fn slice(&self, start: usize, end: usize) -> Self {
        match self {
            Self::I1(values) => Self::I1(values[start..end].to_vec()),
            Self::I8(values) => Self::I8(values[start..end].to_vec()),
            Self::I16(values) => Self::I16(values[start..end].to_vec()),
            Self::I32(values) => Self::I32(values[start..end].to_vec()),
            Self::I64(values) => Self::I64(values[start..end].to_vec()),
            Self::U8(values) => Self::U8(values[start..end].to_vec()),
            Self::U16(values) => Self::U16(values[start..end].to_vec()),
            Self::U32(values) => Self::U32(values[start..end].to_vec()),
            Self::U64(values) => Self::U64(values[start..end].to_vec()),
            Self::F16(values) => Self::F16(values[start..end].to_vec()),
            Self::F32(values) => Self::F32(values[start..end].to_vec()),
            Self::F64(values) => Self::F64(values[start..end].to_vec()),
            Self::Ptr(values) => Self::Ptr(values[start..end].to_vec()),
        }
    }

    pub(super) fn extend_from(&mut self, other: &Self) -> bool {
        match (self, other) {
            (Self::I1(dst), Self::I1(src)) | (Self::U8(dst), Self::U8(src)) => {
                dst.extend_from_slice(src)
            }
            (Self::I8(dst), Self::I8(src)) => dst.extend_from_slice(src),
            (Self::I16(dst), Self::I16(src)) => dst.extend_from_slice(src),
            (Self::U16(dst), Self::U16(src)) => dst.extend_from_slice(src),
            (Self::F16(dst), Self::F16(src)) => dst.extend_from_slice(src),
            (Self::I32(dst), Self::I32(src)) => dst.extend_from_slice(src),
            (Self::U32(dst), Self::U32(src)) => dst.extend_from_slice(src),
            (Self::F32(dst), Self::F32(src)) => dst.extend_from_slice(src),
            (Self::I64(dst), Self::I64(src)) => dst.extend_from_slice(src),
            (Self::U64(dst), Self::U64(src)) => dst.extend_from_slice(src),
            (Self::F64(dst), Self::F64(src)) => dst.extend_from_slice(src),
            (Self::Ptr(dst), Self::Ptr(src)) => dst.extend_from_slice(src),
            _ => return false,
        }
        true
    }

    pub(super) fn iter(&self) -> RuntimeCellsIter<'_> {
        RuntimeCellsIter {
            cells: self,
            index: 0,
        }
    }

    pub(super) fn to_values(&self) -> Vec<RuntimeValue> {
        self.iter().collect()
    }

    pub(super) fn as_i32(&self) -> Option<&[i32]> {
        match self {
            Self::I32(values) => Some(values),
            _ => None,
        }
    }

    pub(super) fn as_i64(&self) -> Option<&[i64]> {
        match self {
            Self::I64(values) => Some(values),
            _ => None,
        }
    }

    pub(super) fn as_u8(&self) -> Option<&[u8]> {
        match self {
            Self::U8(values) => Some(values),
            _ => None,
        }
    }

    pub(super) fn as_u16(&self) -> Option<&[u16]> {
        match self {
            Self::U16(values) => Some(values),
            _ => None,
        }
    }

    pub(super) fn as_f32(&self) -> Option<&[f32]> {
        match self {
            Self::F32(values) => Some(values),
            _ => None,
        }
    }

    pub(super) fn as_f64(&self) -> Option<&[f64]> {
        match self {
            Self::F64(values) => Some(values),
            _ => None,
        }
    }

    pub(super) fn wrapping_sum(&self) -> Option<RuntimeValue> {
        Some(match self {
            Self::I8(values) => RuntimeValue::I8(values.iter().copied().fold(0, i8::wrapping_add)),
            Self::I16(values) => {
                RuntimeValue::I16(values.iter().copied().fold(0, i16::wrapping_add))
            }
            Self::I32(values) => {
                RuntimeValue::I32(values.iter().copied().fold(0, i32::wrapping_add))
            }
            Self::I64(values) => {
                RuntimeValue::I64(values.iter().copied().fold(0, i64::wrapping_add))
            }
            Self::U8(values) => RuntimeValue::U8(values.iter().copied().fold(0, u8::wrapping_add)),
            Self::U16(values) => {
                RuntimeValue::U16(values.iter().copied().fold(0, u16::wrapping_add))
            }
            Self::U32(values) => {
                RuntimeValue::U32(values.iter().copied().fold(0, u32::wrapping_add))
            }
            Self::U64(values) => {
                RuntimeValue::U64(values.iter().copied().fold(0, u64::wrapping_add))
            }
            Self::F32(values) => RuntimeValue::F32(values.iter().copied().sum()),
            Self::F64(values) => RuntimeValue::F64(values.iter().copied().sum()),
            _ => return None,
        })
    }
}

pub(super) struct RuntimeCellsIter<'a> {
    cells: &'a RuntimeCells,
    index: usize,
}

impl Iterator for RuntimeCellsIter<'_> {
    type Item = RuntimeValue;

    fn next(&mut self) -> Option<Self::Item> {
        let value = self.cells.get(self.index)?;
        self.index += 1;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.cells.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for RuntimeCellsIter<'_> {}

impl<'a> IntoIterator for &'a RuntimeCells {
    type Item = RuntimeValue;
    type IntoIter = RuntimeCellsIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_array_new(
    context: *mut FaberRtContextV1,
    kind: FaberRtValueKindV1,
) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        if !valid_kind(kind) {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let Some(values) = RuntimeCells::empty(kind) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        store_array_cells(runtime, kind, values)
    })
}

pub(super) fn store_array(
    runtime: &mut RuntimeContext,
    kind: FaberRtValueKindV1,
    values: Vec<RuntimeValue>,
) -> FaberRtPtrResultV1 {
    let Some(values) = RuntimeCells::from_values(kind, values) else {
        return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
    };
    store_array_cells(runtime, kind, values)
}

pub(super) fn store_array_cells(
    runtime: &mut RuntimeContext,
    kind: FaberRtValueKindV1,
    values: RuntimeCells,
) -> FaberRtPtrResultV1 {
    let array = super::StableBox::new(RuntimeArray { kind, values });
    let handle = array.handle();
    let index = runtime.arrays.len();
    runtime.array_by_handle.insert(handle as usize, index);
    runtime.arrays.push(array);
    FaberRtPtrResultV1::success(handle)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_array_push(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
    kind: FaberRtValueKindV1,
    value: *const c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(array) = find_array_mut(runtime, array) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(value) = (unsafe { read_value(kind, value) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if array.kind != kind || !array.values.push(value) {
            return STATUS_INVALID_ARGUMENT;
        }
        STATUS_OK
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_array_extend(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
    source: *mut c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(dest_index) = find_array_index(runtime, array) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(source_index) = find_array_index(runtime, source) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if runtime.arrays[dest_index].kind != runtime.arrays[source_index].kind {
            return STATUS_INVALID_ARGUMENT;
        }
        if dest_index == source_index {
            let extra = runtime.arrays[dest_index].values.clone();
            let dest = &mut runtime.arrays[dest_index].values;
            dest.reserve(extra.len());
            if !dest.extend_from(&extra) {
                return STATUS_INVALID_ARGUMENT;
            }
        } else {
            let (dest, source) = two_array_refs(&mut runtime.arrays, dest_index, source_index);
            dest.values.reserve(source.values.len());
            if !dest.values.extend_from(&source.values) {
                return STATUS_INVALID_ARGUMENT;
            }
        }
        STATUS_OK
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_array_length(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
    output: *mut i64,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(array) = find_array(runtime, array) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Ok(length) = i64::try_from(array.values.len()) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if !(unsafe { write_typed(output.cast(), length) }) {
            return STATUS_INVALID_ARGUMENT;
        }
        STATUS_OK
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_array_get(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
    index: i64,
    kind: FaberRtValueKindV1,
    output: *mut c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(array) = find_array(runtime, array) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Ok(index) = usize::try_from(index) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(value) = array.values.get(index) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if array.kind != kind || !(unsafe { write_value(value, output) }) {
            return STATUS_INVALID_ARGUMENT;
        }
        STATUS_OK
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_array_set(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
    index: i64,
    kind: FaberRtValueKindV1,
    value: *const c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(value) = (unsafe { read_value(kind, value) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(array) = find_array_mut(runtime, array) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Ok(index) = usize::try_from(index) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if array.kind != kind || !array.values.set(index, value) {
            return STATUS_INVALID_ARGUMENT;
        }
        STATUS_OK
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_array_clone(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(source_index) = find_array_index(runtime, array) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let kind = runtime.arrays[source_index].kind;
        let values = runtime.arrays[source_index].values.clone();
        store_array_cells(runtime, kind, values)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_array_contains(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
    kind: FaberRtValueKindV1,
    value: *const c_void,
    output: *mut u8,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(value) = (unsafe { read_value(kind, value) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(array) = find_array(runtime, array) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if array.kind != kind
            || !(unsafe { write_typed(output.cast(), u8::from(array.values.contains(&value))) })
        {
            return STATUS_INVALID_ARGUMENT;
        }
        STATUS_OK
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_array_is_empty(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
    output: *mut u8,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(array) = find_array(runtime, array) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if !(unsafe { write_typed(output.cast(), u8::from(array.values.is_empty())) }) {
            return STATUS_INVALID_ARGUMENT;
        }
        STATUS_OK
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_array_reverse(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
) -> FaberRtStatusV1 {
    ffi_status(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let Some(array) = find_array_mut(runtime, array) else {
            return STATUS_INVALID_ARGUMENT;
        };
        array.values.reverse();
        STATUS_OK
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_array_range(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
    mode: FaberRtArrayRangeModeV1,
    first: i64,
    second: i64,
) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(source_index) = find_array_index(runtime, array) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let source = &runtime.arrays[source_index];
        let Some((start, end)) = range_bounds(mode, first, second, source.values.len()) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let kind = source.kind;
        let values = source.values.slice(start, end);
        store_array_cells(runtime, kind, values)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __faber_rt_v1_array_option(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
    mode: FaberRtArrayOptionModeV1,
    index: i64,
) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| {
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let Some(array_index) = find_array_index(runtime, array) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let array = &mut runtime.arrays[array_index];
        let kind = array.kind;
        let value = match mode {
            ARRAY_OPTION_INDEX => usize::try_from(index)
                .ok()
                .and_then(|index| array.values.get(index)),
            ARRAY_OPTION_FIRST => array.values.first(),
            ARRAY_OPTION_LAST => array.values.last(),
            ARRAY_OPTION_REMOVE_FIRST => (!array.values.is_empty())
                .then(|| array.values.remove(0))
                .flatten(),
            ARRAY_OPTION_REMOVE_LAST => array.values.pop(),
            _ => return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT),
        };
        store_option(runtime, kind, value)
    })
}

fn range_bounds(
    mode: FaberRtArrayRangeModeV1,
    first: i64,
    second: i64,
    len: usize,
) -> Option<(usize, usize)> {
    let clamp = |value: i64| usize::try_from(value).ok().map(|value| value.min(len));
    Some(match mode {
        ARRAY_RANGE_SLICE => {
            let end = clamp(second)?;
            let start = clamp(first)?.min(end);
            (start, end)
        }
        ARRAY_RANGE_TAKE => (0, clamp(first)?),
        ARRAY_RANGE_TAKE_LAST => (len.saturating_sub(clamp(first)?), len),
        ARRAY_RANGE_DROP_FIRST => (clamp(first)?, len),
        _ => return None,
    })
}

pub(super) fn valid_kind(kind: FaberRtValueKindV1) -> bool {
    matches!(
        kind,
        VALUE_KIND_I1
            | VALUE_KIND_I8
            | VALUE_KIND_I16
            | VALUE_KIND_I32
            | VALUE_KIND_I64
            | VALUE_KIND_U8
            | VALUE_KIND_U16
            | VALUE_KIND_U32
            | VALUE_KIND_U64
            | VALUE_KIND_F16
            | VALUE_KIND_F32
            | VALUE_KIND_F64
            | VALUE_KIND_PTR
            | VALUE_KIND_TEXT
            | VALUE_KIND_VALOR
            | VALUE_KIND_OPTION_I64
            | VALUE_KIND_INSTANS
            | VALUE_KIND_ASCII
    )
}

pub(super) unsafe fn runtime_mut<'a>(
    context: *mut FaberRtContextV1,
) -> Option<&'a mut RuntimeContext> {
    (!context.is_null()).then(|| unsafe { &mut *context.cast::<RuntimeContext>() })
}

pub(super) fn find_array(runtime: &RuntimeContext, handle: *mut c_void) -> Option<&RuntimeArray> {
    let index = *runtime.array_by_handle.get(&(handle as usize))?;
    runtime.arrays.get(index).map(super::StableBox::as_ref)
}

pub(super) fn find_array_mut(
    runtime: &mut RuntimeContext,
    handle: *mut c_void,
) -> Option<&mut RuntimeArray> {
    let index = *runtime.array_by_handle.get(&(handle as usize))?;
    runtime.arrays.get_mut(index).map(super::StableBox::as_mut)
}

fn find_array_index(runtime: &RuntimeContext, handle: *mut c_void) -> Option<usize> {
    runtime.array_by_handle.get(&(handle as usize)).copied()
}

fn two_array_refs(
    arrays: &mut [super::StableBox<RuntimeArray>],
    dest_index: usize,
    source_index: usize,
) -> (&mut RuntimeArray, &RuntimeArray) {
    if dest_index < source_index {
        let (left, right) = arrays.split_at_mut(source_index);
        (left[dest_index].as_mut(), right[0].as_ref())
    } else {
        let (left, right) = arrays.split_at_mut(dest_index);
        (right[0].as_mut(), left[source_index].as_ref())
    }
}

pub(super) unsafe fn read_value(
    kind: FaberRtValueKindV1,
    value: *const c_void,
) -> Option<RuntimeValue> {
    Some(match kind {
        VALUE_KIND_I1 => RuntimeValue::I1(unsafe { read_typed(value) }?),
        VALUE_KIND_I8 => RuntimeValue::I8(unsafe { read_typed(value) }?),
        VALUE_KIND_I16 => RuntimeValue::I16(unsafe { read_typed(value) }?),
        VALUE_KIND_I32 => RuntimeValue::I32(unsafe { read_typed(value) }?),
        VALUE_KIND_I64 => RuntimeValue::I64(unsafe { read_typed(value) }?),
        VALUE_KIND_U8 => RuntimeValue::U8(unsafe { read_typed(value) }?),
        VALUE_KIND_U16 => RuntimeValue::U16(unsafe { read_typed(value) }?),
        VALUE_KIND_U32 => RuntimeValue::U32(unsafe { read_typed(value) }?),
        VALUE_KIND_U64 => RuntimeValue::U64(unsafe { read_typed(value) }?),
        VALUE_KIND_F16 => RuntimeValue::F16(unsafe { read_typed(value) }?),
        VALUE_KIND_F32 => RuntimeValue::F32(unsafe { read_typed(value) }?),
        VALUE_KIND_F64 => RuntimeValue::F64(unsafe { read_typed(value) }?),
        VALUE_KIND_PTR
        | VALUE_KIND_TEXT
        | VALUE_KIND_VALOR
        | VALUE_KIND_OPTION_I64
        | VALUE_KIND_INSTANS
        | VALUE_KIND_ASCII => RuntimeValue::Ptr(unsafe { read_typed(value) }?),
        _ => return None,
    })
}

#[allow(clippy::similar_names)]
pub(super) unsafe fn write_value(value: RuntimeValue, output: *mut c_void) -> bool {
    match value {
        RuntimeValue::I1(value) => unsafe { write_typed(output, value) },
        RuntimeValue::I8(value) => unsafe { write_typed(output, value) },
        RuntimeValue::I16(value) => unsafe { write_typed(output, value) },
        RuntimeValue::I32(value) => unsafe { write_typed(output, value) },
        RuntimeValue::I64(value) => unsafe { write_typed(output, value) },
        RuntimeValue::U8(value) => unsafe { write_typed(output, value) },
        RuntimeValue::U16(value) => unsafe { write_typed(output, value) },
        RuntimeValue::U32(value) => unsafe { write_typed(output, value) },
        RuntimeValue::U64(value) => unsafe { write_typed(output, value) },
        RuntimeValue::F16(value) => unsafe { write_typed(output, value) },
        RuntimeValue::F32(value) => unsafe { write_typed(output, value) },
        RuntimeValue::F64(value) => unsafe { write_typed(output, value) },
        RuntimeValue::Ptr(value) => unsafe { write_typed(output, value) },
    }
}

#[allow(clippy::unnecessary_wraps)]
unsafe fn read_typed<T: Copy>(value: *const c_void) -> Option<T> {
    let value = value.cast::<T>();
    (!value.is_null() && value.is_aligned()).then(|| unsafe { value.read() })
}

unsafe fn write_typed<T>(output: *mut c_void, value: T) -> bool {
    let output = output.cast::<T>();
    if output.is_null() || !output.is_aligned() {
        return false;
    }
    unsafe { output.write(value) };
    true
}

fn ffi_status(operation: impl FnOnce() -> FaberRtStatusV1) -> FaberRtStatusV1 {
    panic::catch_unwind(AssertUnwindSafe(operation)).unwrap_or(STATUS_PANIC)
}

fn ffi_ptr_result(operation: impl FnOnce() -> FaberRtPtrResultV1) -> FaberRtPtrResultV1 {
    panic::catch_unwind(AssertUnwindSafe(operation))
        .unwrap_or(FaberRtPtrResultV1::failure(STATUS_PANIC))
}
