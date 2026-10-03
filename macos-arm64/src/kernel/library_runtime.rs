//! Metal runtime bridge for the fused projection library entries.
//!
//! `library.rs` owns the shape-checked CPU bodies and their plan-selection ABI.
//! This module is the production call path used by the Metal family materializer:
//! it accepts the runtime's fused request, forwards it through
//! [`super::library::dispatch_selected`], and mints the matching device entry
//! names.  Keeping the bridge here prevents the runtime from reconstructing a
//! selection from buffer lengths or from silently accepting an unowned entry.
//!
//! The MSL below is a small dense-f32 family materializer.  Packed weight
//! materializers remain owned by the target emitter, but the entry names and
//! grouped output layout are the same ABI that the carrier binds to.

use host_coordinator::DeviceHandle;
use serde::{Deserialize, Serialize};

use crate::device_descriptor::{DeviceDataType, PackedStorageFormat};
use crate::device_host::{DeviceRuntime, DeviceSession};
use crate::kernel::HostResult;

use super::library::{
    BindDescriptor, BindLayout, KernelBodyError, LibraryDispatch, QkvProjectionBind,
    QkvProjectionLayout, QkvProjectionWeight, QuantizedFormat, QuantizedGemvBind,
    dispatch_selected,
};

/// Concrete dimensions baked into the fused Metal family module.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LibraryFamilyMslFacts {
    /// Number of activation rows.
    pub rows: u64,
    /// Activation and RMS row width.
    pub hidden: u64,
    /// Number of key/value groups.
    pub kv_heads: u64,
    /// Query heads sharing one key/value group.
    pub q_per_kv: u64,
    /// Elements in one attention head.
    pub head_dim: u64,
    /// RMS epsilon for the residual body.
    pub epsilon: f32,
}

impl LibraryFamilyMslFacts {
    fn validate(self) -> Result<(), KernelBodyError> {
        if self.rows == 0
            || self.hidden == 0
            || self.kv_heads == 0
            || self.q_per_kv == 0
            || self.head_dim == 0
        {
            return Err(KernelBodyError::InvalidBind(
                "fused library Metal module has a zero dimension",
            ));
        }
        self.kv_heads
            .checked_mul(self.q_per_kv)
            .and_then(|heads| heads.checked_mul(self.head_dim))
            .ok_or(KernelBodyError::InvalidBind(
                "fused library Metal module Q width overflows",
            ))?;
        self.kv_heads
            .checked_mul(self.head_dim)
            .ok_or(KernelBodyError::InvalidBind(
                "fused library Metal module KV width overflows",
            ))?;
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(KernelBodyError::InvalidEpsilon);
        }
        self.rows
            .checked_mul(self.hidden)
            .and_then(|span| span.checked_mul(3))
            .ok_or(KernelBodyError::InvalidBind(
                "fused library Metal module span overflow",
            ))?;
        Ok(())
    }

    #[must_use]
    pub fn q_width(self) -> u64 {
        self.kv_heads * self.q_per_kv * self.head_dim
    }

    #[must_use]
    pub fn kv_width(self) -> u64 {
        self.kv_heads * self.head_dim
    }
}

/// Mint the QKV and residual/RMS entries consumed by the Metal runtime.
///
/// The Q output is grouped as `[kv_group, q_head, row, head_dim]`; K and V are
/// grouped as `[kv_group, row, head_dim]`.  The three dense weights use the
/// library ABI's column-major `[output, hidden]` view.  One Q lane per
/// group/row also writes the corresponding K and V lanes, so the single
/// device dispatch has the same one-body shape as the host reference.
pub fn library_family_msl(facts: &LibraryFamilyMslFacts) -> Result<String, KernelBodyError> {
    facts.validate()?;
    let rows = facts.rows;
    let hidden = facts.hidden;
    let kv_heads = facts.kv_heads;
    let q_per_kv = facts.q_per_kv;
    let head_dim = facts.head_dim;
    let q_width = facts.q_width();
    let kv_width = facts.kv_width();
    let epsilon = facts.epsilon;
    Ok(format!(
        r#"#include <metal_stdlib>
using namespace metal;

constant uint ROWS = {rows}u;
constant uint HIDDEN = {hidden}u;
constant uint KV_HEADS = {kv_heads}u;
constant uint Q_PER_KV = {q_per_kv}u;
constant uint HEAD_DIM = {head_dim}u;
constant uint Q_WIDTH = {q_width}u;
constant uint KV_WIDTH = {kv_width}u;
constant float RMS_EPSILON = {epsilon:.9e}f;

kernel void QkvProjection(
    device const float* activation [[buffer(0)]],
    device const float* q_weight [[buffer(1)]],
    device const float* k_weight [[buffer(2)]],
    device const float* v_weight [[buffer(3)]],
    device float* q_output [[buffer(4)]],
    device float* k_output [[buffer(5)]],
    device float* v_output [[buffer(6)]],
    uint id [[thread_position_in_grid]]) {{
  if (id >= ROWS * Q_WIDTH) {{ return; }}
  uint dim = id % HEAD_DIM;
  uint row = (id / HEAD_DIM) % ROWS;
  uint query_head = (id / (ROWS * HEAD_DIM)) % Q_PER_KV;
  uint group = id / (Q_PER_KV * ROWS * HEAD_DIM);
  uint q_column = (group * Q_PER_KV + query_head) * HEAD_DIM + dim;
  uint input_base = row * HIDDEN;
  float q_sum = 0.0f;
  for (uint k = 0u; k < HIDDEN; ++k) {{
    q_sum += activation[input_base + k] * q_weight[q_column * HIDDEN + k];
  }}
  q_output[id] = q_sum;

  // The first query head owns the grouped K/V row for this group.
  if (query_head == 0u) {{
    uint kv_id = group * ROWS * HEAD_DIM + row * HEAD_DIM + dim;
    uint kv_column = group * HEAD_DIM + dim;
    float k_sum = 0.0f;
    float v_sum = 0.0f;
    for (uint k = 0u; k < HIDDEN; ++k) {{
      float x = activation[input_base + k];
      k_sum += x * k_weight[kv_column * HIDDEN + k];
      v_sum += x * v_weight[kv_column * HIDDEN + k];
    }}
    k_output[kv_id] = k_sum;
    v_output[kv_id] = v_sum;
  }}
}}

kernel void ResidualRmsNorm(
    device const float* residual [[buffer(0)]],
    device const float* skip [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device float* output [[buffer(3)]],
    uint id [[thread_position_in_grid]]) {{
  if (id >= ROWS * HIDDEN) {{ return; }}
  uint row = id / HIDDEN;
  uint col = id % HIDDEN;
  float sumsq = 0.0f;
  for (uint j = 0u; j < HIDDEN; ++j) {{
    float value = residual[row * HIDDEN + j] + skip[row * HIDDEN + j];
    sumsq += value * value;
  }}
  float scale = 1.0f / sqrt(sumsq / float(HIDDEN) + RMS_EPSILON);
  output[id] = (residual[row * HIDDEN + col] + skip[row * HIDDEN + col])
      * scale * gamma[col];
}}
"#,
        rows = rows,
        hidden = hidden,
        kv_heads = kv_heads,
        q_per_kv = q_per_kv,
        head_dim = head_dim,
        q_width = q_width,
        kv_width = kv_width,
        epsilon = epsilon,
    ))
}

/// One fused request entering the Metal runtime library route.
///
/// Unlike the generic library enum, this route admits only the two fused
/// entries whose device module is materialized here.  Each arm still carries
/// the complete selector facts so unsupported layouts and uniform drift reach
/// the existing fail-closed selector before a body reads or writes a buffer.
#[allow(
    clippy::large_enum_variant,
    reason = "the runtime dispatch carries complete borrowed fused requests on the launch path; boxing would add allocation and erase the inline boundary"
)]
pub enum MetalLibraryDispatch<'a> {
    /// Grouped Q/K/V projection.
    QkvProjection {
        library_entry: Option<&'a str>,
        decode_gemv: u32,
        layout: QkvProjectionLayout,
        bind: &'a QkvProjectionBind,
        activation: &'a [f32],
        weights: [QkvProjectionWeight<'a>; 3],
        biases: [Option<&'a [f32]>; 3],
        rope: Option<(&'a [f32], &'a [f32])>,
        outputs: [&'a mut [f32]; 3],
    },
    /// Residual addition followed by RMS normalization.
    ResidualRmsNorm {
        library_entry: Option<&'a str>,
        layout: BindLayout,
        bind: &'a BindDescriptor,
        residual: &'a [f32],
        skip: &'a [f32],
        gamma: &'a [f32],
        output: &'a mut [f32],
        epsilon: f32,
    },
}

/// One device buffer participating in a fused library dispatch.
///
/// The binding index is retained separately from the vector position.  In
/// particular, K/V cache targets are extra resources on the carrier launch;
/// they must not be mistaken for inputs or dropped when the owning body runs.
#[derive(Debug, Clone, Copy)]
pub struct FusedLibraryDeviceBuffer<'a> {
    /// Device allocation carrying the logical view.
    pub handle: &'a DeviceHandle,
    /// Dtype carried by the descriptor for this view.
    pub dtype: DeviceDataType,
    /// View offset in bytes.
    pub byte_offset: u64,
    /// View span in bytes.
    pub view_span: u64,
    /// Explicit descriptor binding index.
    pub binding_index: u32,
    /// GGML block/pack geometry for a packed view. Unknown formats remain
    /// explicit so the CPU bridge can fail closed before reinterpretation.
    pub packed_format: Option<PackedStorageFormat>,
}

/// A producer fact emitted when the owning fused body executes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FusedLibraryDispatchReceipt {
    /// Derived carrier entry that selected the library route.
    pub entry: String,
    /// Owning body used by the route.
    pub body: String,
    /// Q destination binding.
    pub q_output_binding: u32,
    /// K destination binding.
    pub k_output_binding: u32,
    /// V destination binding.
    pub v_output_binding: u32,
}

/// A fully bound CPU bridge for one derived QKV library entry.
///
/// This is the production fallback for Metal modules whose carrier function
/// only publishes Q. The bridge reads typed dense views and native packed
/// views, executes the same selected body used by the focused library tests,
/// and uploads all three destinations. Packed views require an admitted GGML
/// format fact; absent or unsupported formats fail closed rather than being
/// reinterpreted as f32.
pub struct FusedQkvDeviceDispatch<'a> {
    /// Canonical library selection entry.
    pub library_entry: &'a str,
    /// Derived carrier entry that produced this dispatch fact.
    pub derived_entry: &'a str,
    /// Decode uniform carried by the plan.
    pub decode_gemv: u32,
    /// Fully validated Q/K/V shape and stride facts.
    pub bind: QkvProjectionBind,
    /// Activation view.
    pub activation: FusedLibraryDeviceBuffer<'a>,
    /// Q/K/V weight views.
    pub weights: [FusedLibraryDeviceBuffer<'a>; 3],
    /// Optional Q/K/V bias views.
    pub biases: [Option<FusedLibraryDeviceBuffer<'a>>; 3],
    /// Optional cosine/sine RoPE table views.
    pub rope: Option<(FusedLibraryDeviceBuffer<'a>, FusedLibraryDeviceBuffer<'a>)>,
    /// Q/K/V output views. K/V are the launch's extra Write resources.
    pub outputs: [FusedLibraryDeviceBuffer<'a>; 3],
}

fn dense_storage(
    runtime: &mut DeviceRuntime,
    view: FusedLibraryDeviceBuffer<'_>,
    label: &'static str,
) -> HostResult<(Vec<f32>, usize, usize)> {
    if view.dtype != DeviceDataType::F32 {
        return Err(crate::kernel::HostError::invalid_args(format!(
            "fused library CPU bridge requires f32 {label}, got {}",
            view.dtype.spelling()
        )));
    }
    if !view.byte_offset.is_multiple_of(4) || !view.view_span.is_multiple_of(4) {
        return Err(crate::kernel::HostError::invalid_args(format!(
            "fused library {label} view is not f32 aligned"
        )));
    }
    let values = runtime.readback_f32(view.handle)?;
    let start = usize::try_from(view.byte_offset / 4).map_err(|_| {
        crate::kernel::HostError::invalid_args(format!(
            "fused library {label} offset overflows host"
        ))
    })?;
    let span = usize::try_from(view.view_span / 4).map_err(|_| {
        crate::kernel::HostError::invalid_args(format!("fused library {label} span overflows host"))
    })?;
    let end = start.checked_add(span).ok_or_else(|| {
        crate::kernel::HostError::invalid_args(format!("fused library {label} view overflows host"))
    })?;
    if end > values.len() {
        return Err(crate::kernel::HostError::invalid_args(format!(
            "fused library {label} view [{start}..{end}] exceeds {} f32 values",
            values.len()
        )));
    }
    Ok((values, start, end))
}

fn dense_view(
    runtime: &mut DeviceRuntime,
    view: FusedLibraryDeviceBuffer<'_>,
    label: &'static str,
) -> HostResult<Vec<f32>> {
    let (values, start, end) = dense_storage(runtime, view, label)?;
    Ok(values[start..end].to_vec())
}

fn packed_view(
    runtime: &mut DeviceRuntime,
    view: FusedLibraryDeviceBuffer<'_>,
    label: &'static str,
) -> HostResult<Vec<u8>> {
    if !matches!(view.dtype, DeviceDataType::F32 | DeviceDataType::U8) {
        return Err(crate::kernel::HostError::invalid_args(format!(
            "fused library packed {label} requires an f32-word or u8 region, got {}",
            view.dtype.spelling()
        )));
    }
    // The f32 tag marks a padded word region (byte length / 4 elements);
    // the view span stays in bytes either way.
    if view.packed_format.is_none() {
        return Err(crate::kernel::HostError::invalid_args(format!(
            "fused library packed {label} has no known GGML format"
        )));
    }
    let values = runtime.readback_bytes(view.handle, DeviceDataType::U8)?;
    let start = usize::try_from(view.byte_offset).map_err(|_| {
        crate::kernel::HostError::invalid_args(format!(
            "fused library packed {label} offset overflows host"
        ))
    })?;
    let span = usize::try_from(view.view_span).map_err(|_| {
        crate::kernel::HostError::invalid_args(format!(
            "fused library packed {label} span overflows host"
        ))
    })?;
    let end = start.checked_add(span).ok_or_else(|| {
        crate::kernel::HostError::invalid_args(format!(
            "fused library packed {label} view overflows host"
        ))
    })?;
    if end > values.len() {
        return Err(crate::kernel::HostError::invalid_args(format!(
            "fused library packed {label} view [{start}..{end}] exceeds {} bytes",
            values.len()
        )));
    }
    Ok(values[start..end].to_vec())
}

fn quantized_format(
    view: FusedLibraryDeviceBuffer<'_>,
    label: &'static str,
) -> HostResult<QuantizedFormat> {
    let format = view.packed_format.ok_or_else(|| {
        crate::kernel::HostError::invalid_args(format!(
            "fused library packed {label} has no known GGML format"
        ))
    })?;
    QuantizedFormat::from_ggml_type_id(format.ggml_type_id()).ok_or_else(|| {
        crate::kernel::HostError::invalid_args(format!(
            "fused library packed {label} format {} is not servable by the CPU bridge",
            format.spelling()
        ))
    })
}

/// Execute one fully bound fused QKV library body on the Metal session.
///
/// The call is deliberately below the generic carrier launch: K/V output
/// bindings are explicit inputs to this function and are copied back only
/// after the selected body has written them.  A caller can therefore prove
/// that the extra Write resources, not a census or a carrier side effect,
/// produced the cache bytes.
pub fn dispatch_fused_qkv_device(
    runtime: &mut DeviceRuntime,
    request: FusedQkvDeviceDispatch<'_>,
) -> HostResult<FusedLibraryDispatchReceipt> {
    if runtime.backend() != host_coordinator::DeviceBackend::Metal {
        return Err(crate::kernel::HostError::invalid_args(
            "fused library CPU bridge is a Metal-only route",
        ));
    }
    let activation = dense_view(runtime, request.activation, "activation")?;
    enum WeightStorage {
        Dense(Vec<f32>),
        Packed {
            values: Vec<u8>,
            format: QuantizedFormat,
        },
    }
    let mut weight_storage = [
        WeightStorage::Dense(Vec::new()),
        WeightStorage::Dense(Vec::new()),
        WeightStorage::Dense(Vec::new()),
    ];
    for ((storage, view), label) in weight_storage
        .iter_mut()
        .zip(request.weights)
        .zip(["Q weight", "K weight", "V weight"])
    {
        // A packed region is recognized by its uploaded GGML format fact,
        // not by its dtype tag: the wire types padded packed regions as f32
        // words (byte length / 4) while the bytes keep the native layout.
        *storage = if view.packed_format.is_some() {
            if !matches!(view.dtype, DeviceDataType::F32 | DeviceDataType::U8) {
                return Err(crate::kernel::HostError::invalid_args(format!(
                    "fused library packed {label} requires an f32-word or u8 region, got {}",
                    view.dtype.spelling()
                )));
            }
            WeightStorage::Packed {
                values: packed_view(runtime, view, label)?,
                format: quantized_format(view, label)?,
            }
        } else {
            match view.dtype {
                DeviceDataType::F32 => WeightStorage::Dense(dense_view(runtime, view, label)?),
                DeviceDataType::U8 => {
                    return Err(crate::kernel::HostError::invalid_args(format!(
                        "fused library {label} is a u8 region with no known GGML format"
                    )));
                }
                dtype => {
                    return Err(crate::kernel::HostError::invalid_args(format!(
                        "fused library {label} requires f32 or packed u8, got {}",
                        dtype.spelling()
                    )));
                }
            }
        };
    }
    let q_width = request
        .bind
        .kv_heads
        .checked_mul(request.bind.q_per_kv)
        .and_then(|heads| heads.checked_mul(request.bind.head_dim))
        .ok_or_else(|| crate::kernel::HostError::invalid_args("QKV Q width overflows"))?;
    let kv_width = request
        .bind
        .kv_heads
        .checked_mul(request.bind.head_dim)
        .ok_or_else(|| crate::kernel::HostError::invalid_args("QKV KV width overflows"))?;
    let packed_bind = |n: u64, format: QuantizedFormat| {
        QuantizedGemvBind::decode(request.bind.hidden, n, format, request.bind.grid)
    };
    let weight_refs = [
        match &weight_storage[0] {
            WeightStorage::Dense(values) => QkvProjectionWeight::Dense(values),
            WeightStorage::Packed { values, format } => QkvProjectionWeight::Quantized {
                bind: packed_bind(q_width, *format),
                packed: values,
            },
        },
        match &weight_storage[1] {
            WeightStorage::Dense(values) => QkvProjectionWeight::Dense(values),
            WeightStorage::Packed { values, format } => QkvProjectionWeight::Quantized {
                bind: packed_bind(kv_width, *format),
                packed: values,
            },
        },
        match &weight_storage[2] {
            WeightStorage::Dense(values) => QkvProjectionWeight::Dense(values),
            WeightStorage::Packed { values, format } => QkvProjectionWeight::Quantized {
                bind: packed_bind(kv_width, *format),
                packed: values,
            },
        },
    ];
    let biases = [
        request.biases[0]
            .map(|view| dense_view(runtime, view, "Q bias"))
            .transpose()?,
        request.biases[1]
            .map(|view| dense_view(runtime, view, "K bias"))
            .transpose()?,
        request.biases[2]
            .map(|view| dense_view(runtime, view, "V bias"))
            .transpose()?,
    ];
    let rope = request
        .rope
        .map(|(cos, sin)| {
            Ok((
                dense_view(runtime, cos, "RoPE cosine")?,
                dense_view(runtime, sin, "RoPE sine")?,
            ))
        })
        .transpose()?;
    let (mut q_output, q_start, q_end) = dense_storage(runtime, request.outputs[0], "Q output")?;
    let (mut k_output, k_start, k_end) = dense_storage(runtime, request.outputs[1], "K output")?;
    let (mut v_output, v_start, v_end) = dense_storage(runtime, request.outputs[2], "V output")?;
    let bias_refs = [
        biases[0].as_deref(),
        biases[1].as_deref(),
        biases[2].as_deref(),
    ];
    let rope_refs = rope
        .as_ref()
        .map(|(cos, sin)| (cos.as_slice(), sin.as_slice()));
    dispatch_metal_library(MetalLibraryDispatch::QkvProjection {
        library_entry: Some(request.library_entry),
        decode_gemv: request.decode_gemv,
        layout: request.bind.layout,
        bind: &request.bind,
        activation: &activation,
        weights: weight_refs,
        biases: bias_refs,
        rope: rope_refs,
        outputs: [
            &mut q_output[q_start..q_end],
            &mut k_output[k_start..k_end],
            &mut v_output[v_start..v_end],
        ],
    })
    .map_err(|error| crate::kernel::HostError::invalid_args(error.to_string()))?;
    for (view, values) in request
        .outputs
        .into_iter()
        .zip([q_output, k_output, v_output])
    {
        runtime.copy_in_f32(view.handle, &values)?;
    }
    Ok(FusedLibraryDispatchReceipt {
        entry: request.derived_entry.to_owned(),
        body: "qkv_projection_cpu".to_owned(),
        q_output_binding: request.outputs[0].binding_index,
        k_output_binding: request.outputs[1].binding_index,
        v_output_binding: request.outputs[2].binding_index,
    })
}

/// A fully bound device view set for one fused residual-plus-RMS dispatch.
///
/// The views are dense f32 activations and the gamma weight; the owning body
/// is the library's residual-add + RMS normalization under a row-major bind.
pub struct FusedResidualRmsDeviceDispatch<'a> {
    /// Canonical library entry (`"ResidualRmsNorm"`).
    pub library_entry: &'a str,
    /// Activation rows in this invocation.
    pub rows: u64,
    /// Activation and RMS row width.
    pub hidden: u64,
    /// RMS epsilon carried by the compiled carrier.
    pub epsilon: f32,
    /// Residual (pre-attention stream) view.
    pub residual: FusedLibraryDeviceBuffer<'a>,
    /// Skip (attention output projection) view.
    pub skip: FusedLibraryDeviceBuffer<'a>,
    /// Gamma weight view.
    pub gamma: FusedLibraryDeviceBuffer<'a>,
    /// Destination view.
    pub output: FusedLibraryDeviceBuffer<'a>,
}

/// Execute one fully bound fused residual-plus-RMS library body on the Metal
/// session.
///
/// The carrier launch publishes the RMS-normalized residual only; the skip
/// stream rides the launch as an extra read resource the carrier kernel never
/// touches.  This bridge runs the fused body that consumes both streams and
/// uploads the destination, so the written bytes prove the residual add ran.
///
/// PB-8: the bridge also writes the accumulated stream (`residual + skip`)
/// back into the residual buffer.  The carrier's layer-end residual add
/// composes against the residual view it was bound with, so without this
/// write-back the attention output is normalized but never joins the
/// residual stream and every layer loses one attention projection
/// (`h_next = h + down` instead of the reference `h + o + down`).
pub fn dispatch_fused_residual_rms_device(
    runtime: &mut DeviceRuntime,
    request: FusedResidualRmsDeviceDispatch<'_>,
) -> HostResult<()> {
    if runtime.backend() != host_coordinator::DeviceBackend::Metal {
        return Err(crate::kernel::HostError::invalid_args(
            "fused library CPU bridge is a Metal-only route",
        ));
    }
    if request.library_entry != "ResidualRmsNorm" {
        return Err(crate::kernel::HostError::invalid_args(
            "fused residual/RMS dispatch disagrees with library_entry",
        ));
    }
    if request.rows == 0 || request.hidden == 0 {
        return Err(crate::kernel::HostError::invalid_args(
            "fused residual/RMS dispatch has a zero dimension",
        ));
    }
    let (mut residual_buffer, residual_start, residual_end) =
        dense_storage(runtime, request.residual, "residual")?;
    let residual_len = residual_end - residual_start;
    let skip = dense_view(runtime, request.skip, "skip")?;
    if skip.len() != residual_len {
        return Err(crate::kernel::HostError::invalid_args(format!(
            "fused residual/RMS dispatch stream widths disagree: residual {residual_len}, skip {}",
            skip.len()
        )));
    }
    let gamma = dense_view(runtime, request.gamma, "gamma")?;
    let (mut output, out_start, out_end) = dense_storage(runtime, request.output, "output")?;
    let bind = BindDescriptor::row_major([request.rows, request.hidden], [1, 1, 1]);
    dispatch_metal_library(MetalLibraryDispatch::ResidualRmsNorm {
        library_entry: Some(request.library_entry),
        layout: BindLayout::RowMajor,
        bind: &bind,
        residual: &residual_buffer[residual_start..residual_end],
        skip: &skip,
        gamma: &gamma,
        output: &mut output[out_start..out_end],
        epsilon: request.epsilon,
    })
    .map_err(|error| crate::kernel::HostError::invalid_args(error.to_string()))?;
    runtime.copy_in_f32(request.output.handle, &output)?;
    // PB-8: fold the skip into the residual stream so the carrier's layer-end
    // residual add composes h + o + down. The write covers the whole device
    // buffer with the view span updated in place, preserving any bytes
    // outside the view.
    for (slot, add) in residual_buffer[residual_start..residual_end]
        .iter_mut()
        .zip(&skip)
    {
        *slot += *add;
    }
    runtime.copy_in_f32(request.residual.handle, &residual_buffer)?;
    Ok(())
}

/// Dispatch one admitted Metal runtime request through the existing library
/// selector and body.
///
/// This is intentionally the sole runtime bridge to `dispatch_selected`.
/// The runtime does not call `select_*` directly, infer layouts from buffer
/// lengths, or bypass the library ABI for a convenient fallback.
pub fn dispatch_metal_library(request: MetalLibraryDispatch<'_>) -> Result<(), KernelBodyError> {
    match request {
        MetalLibraryDispatch::QkvProjection {
            library_entry,
            decode_gemv,
            layout,
            bind,
            activation,
            weights,
            biases,
            rope,
            outputs,
        } => dispatch_selected(LibraryDispatch::QkvProjection {
            library_entry,
            decode_gemv,
            layout,
            bind,
            activation,
            weights,
            biases,
            rope,
            outputs,
        }),
        MetalLibraryDispatch::ResidualRmsNorm {
            library_entry,
            layout,
            bind,
            residual,
            skip,
            gamma,
            output,
            epsilon,
        } => dispatch_selected(LibraryDispatch::ResidualRmsNorm {
            library_entry,
            layout,
            bind,
            residual,
            skip,
            gamma,
            output,
            epsilon,
        }),
    }
}

#[cfg(test)]
#[path = "library_runtime_test.rs"]
mod tests;
