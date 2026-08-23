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
    dispatch_selected, BindDescriptor, BindLayout, KernelBodyError, LibraryDispatch,
    QkvProjectionBind, QkvProjectionLayout, QkvProjectionWeight, QuantizedFormat,
    QuantizedGemvBind,
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
    if view.byte_offset % 4 != 0 || view.view_span % 4 != 0 {
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
                    )))
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
mod tests {
    use super::*;

    #[test]
    fn runtime_msl_admits_activation_width_distinct_from_q_width() {
        let module = library_family_msl(&LibraryFamilyMslFacts {
            rows: 1,
            hidden: 8,
            kv_heads: 1,
            q_per_kv: 1,
            head_dim: 4,
            epsilon: 1.0e-5,
        })
        .expect("activation width and Q width are independent facts");
        assert!(module.contains("constant uint HIDDEN = 8u;"));
        assert!(module.contains("constant uint Q_WIDTH = 4u;"));
    }

    #[test]
    fn runtime_qkv_entry_drift_fails_before_output_write() {
        let bind = QkvProjectionBind::grouped(1, 4, 1, 1, 4, [4, 1, 1]);
        let activation = [1.0; 4];
        let weight = [1.0; 16];
        let mut q = [f32::NAN; 4];
        let mut k = [f32::NAN; 4];
        let mut v = [f32::NAN; 4];
        let error = dispatch_metal_library(MetalLibraryDispatch::QkvProjection {
            library_entry: Some("ResidualRmsNorm"),
            decode_gemv: 0,
            layout: QkvProjectionLayout::Grouped,
            bind: &bind,
            activation: &activation,
            weights: [
                QkvProjectionWeight::Dense(&weight),
                QkvProjectionWeight::Dense(&weight),
                QkvProjectionWeight::Dense(&weight),
            ],
            biases: [None, None, None],
            rope: None,
            outputs: [&mut q, &mut k, &mut v],
        })
        .expect_err("wrong runtime entry must fail closed");
        assert!(matches!(
            error,
            KernelBodyError::InvalidBind(message) if message.contains("QKV projection selection")
        ));
        assert!(q.iter().all(|value| value.is_nan()));
        assert!(k.iter().all(|value| value.is_nan()));
        assert!(v.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn production_bridge_binds_kv_write_targets_for_both_gqa_pairings() {
        use crate::device_host::DeviceRuntime;
        use crate::metal_host::{FakeMetalDriver, MetalHostSession};
        use host_coordinator::{DeviceBackend, DeviceHandle};

        fn view<'a>(
            handle: &'a DeviceHandle,
            count: u64,
            binding: u32,
        ) -> FusedLibraryDeviceBuffer<'a> {
            FusedLibraryDeviceBuffer {
                handle,
                dtype: DeviceDataType::F32,
                byte_offset: 0,
                view_span: count * 4,
                binding_index: binding,
                packed_format: None,
            }
        }

        for (kv_heads, q_per_kv, k_binding, v_binding) in [(2, 2, 5, 6), (1, 4, 7, 8)] {
            let mut runtime = DeviceRuntime::Metal(
                MetalHostSession::with_driver(Box::new(FakeMetalDriver::default()))
                    .expect("fake Metal admission"),
            );
            let head_dim = 2u64;
            let hidden = kv_heads * q_per_kv * head_dim;
            let q_width = hidden;
            let kv_width = kv_heads * head_dim;
            let activation_values = vec![1.0f32; hidden as usize];
            let q_weight_values = vec![1.0f32; (hidden * q_width) as usize];
            let k_weight_values = vec![2.0f32; (hidden * kv_width) as usize];
            let v_weight_values = vec![3.0f32; (hidden * kv_width) as usize];
            let q_output_values = vec![0.0f32; q_width as usize];
            let k_output_values = vec![0.0f32; kv_width as usize];
            let v_output_values = vec![0.0f32; kv_width as usize];
            let alloc = |runtime: &mut DeviceRuntime, values: &[f32]| {
                let handle = runtime
                    .alloc_bytes(values.len() * 4)
                    .expect("allocate bridge buffer");
                runtime
                    .copy_in_f32(&handle, values)
                    .expect("initialize bridge buffer");
                handle
            };
            let activation = alloc(&mut runtime, &activation_values);
            let q_weight = alloc(&mut runtime, &q_weight_values);
            let k_weight = alloc(&mut runtime, &k_weight_values);
            let v_weight = alloc(&mut runtime, &v_weight_values);
            let q_output = alloc(&mut runtime, &q_output_values);
            let k_output = alloc(&mut runtime, &k_output_values);
            let v_output = alloc(&mut runtime, &v_output_values);
            let receipt = dispatch_fused_qkv_device(
                &mut runtime,
                FusedQkvDeviceDispatch {
                    library_entry: "QkvProjection",
                    derived_entry: "prefill_blk_0_QkvProjection",
                    decode_gemv: 1,
                    bind: QkvProjectionBind::grouped(
                        1,
                        hidden,
                        kv_heads,
                        q_per_kv,
                        head_dim,
                        [q_width as u32, 1, 1],
                    ),
                    activation: view(&activation, hidden, 0),
                    weights: [
                        view(&q_weight, hidden * q_width, 1),
                        view(&k_weight, hidden * kv_width, 2),
                        view(&v_weight, hidden * kv_width, 3),
                    ],
                    biases: [None, None, None],
                    rope: None,
                    outputs: [
                        view(&q_output, q_width, 4),
                        view(&k_output, kv_width, k_binding),
                        view(&v_output, kv_width, v_binding),
                    ],
                },
            )
            .expect("fused body executes");
            assert_eq!(receipt.entry, "prefill_blk_0_QkvProjection");
            assert_eq!(receipt.body, "qkv_projection_cpu");
            assert_eq!(receipt.k_output_binding, k_binding);
            assert_eq!(receipt.v_output_binding, v_binding);
            let k_values = runtime.readback_f32(&k_output).expect("read K output");
            let v_values = runtime.readback_f32(&v_output).expect("read V output");
            assert!(k_values.iter().all(|value| *value == hidden as f32 * 2.0));
            assert!(v_values.iter().all(|value| *value == hidden as f32 * 3.0));
            assert_eq!(runtime.backend(), DeviceBackend::Metal);
        }
    }

    #[test]
    fn residual_rms_bridge_adds_skip_before_normalization() {
        use crate::device_host::DeviceRuntime;
        use crate::metal_host::{FakeMetalDriver, MetalHostSession};
        use host_coordinator::DeviceHandle;

        fn view<'a>(
            handle: &'a DeviceHandle,
            count: u64,
            binding: u32,
        ) -> FusedLibraryDeviceBuffer<'a> {
            FusedLibraryDeviceBuffer {
                handle,
                dtype: DeviceDataType::F32,
                byte_offset: 0,
                view_span: count * 4,
                binding_index: binding,
                packed_format: None,
            }
        }

        let mut runtime = DeviceRuntime::Metal(
            MetalHostSession::with_driver(Box::new(FakeMetalDriver::default()))
                .expect("fake Metal admission"),
        );
        let rows = 2u64;
        let hidden = 4u64;
        let epsilon = 1.0e-6f32;
        let residual = [1.0f32, -2.0, 3.0, -4.0, 0.5, 0.5, -1.5, 2.5];
        let skip = [0.25f32, 0.25, -0.25, -0.25, 1.0, -1.0, 2.0, -2.0];
        let gamma = [1.0f32, 0.5, 2.0, 1.5];
        let mut expected = vec![0.0f32; (rows * hidden) as usize];
        for row in 0..rows as usize {
            let base = row * hidden as usize;
            let summed: Vec<f32> = (0..hidden as usize)
                .map(|col| residual[base + col] + skip[base + col])
                .collect();
            let sumsq: f32 = summed.iter().map(|value| value * value).sum();
            let scale = 1.0 / ((sumsq / hidden as f32 + epsilon).sqrt());
            for col in 0..hidden as usize {
                expected[base + col] = summed[col] * scale * gamma[col];
            }
        }
        let alloc = |runtime: &mut DeviceRuntime, values: &[f32]| {
            let handle = runtime
                .alloc_bytes(values.len() * 4)
                .expect("allocate bridge buffer");
            runtime
                .copy_in_f32(&handle, values)
                .expect("initialize bridge buffer");
            handle
        };
        let residual_handle = alloc(&mut runtime, &residual);
        let skip_handle = alloc(&mut runtime, &skip);
        let gamma_handle = alloc(&mut runtime, &gamma);
        let output_handle = alloc(&mut runtime, &vec![0.0f32; expected.len()]);
        dispatch_fused_residual_rms_device(
            &mut runtime,
            FusedResidualRmsDeviceDispatch {
                library_entry: "ResidualRmsNorm",
                rows,
                hidden,
                epsilon,
                residual: view(&residual_handle, rows * hidden, 0),
                skip: view(&skip_handle, rows * hidden, 1),
                gamma: view(&gamma_handle, hidden, 2),
                output: view(&output_handle, rows * hidden, 3),
            },
        )
        .expect("fused residual/RMS body executes");
        let written = runtime
            .readback_f32(&output_handle)
            .expect("read output buffer");
        for (actual, want) in written.iter().zip(&expected) {
            assert!(
                (actual - want).abs() <= 1.0e-6 * want.abs().max(1.0),
                "expected {want}, wrote {actual}"
            );
        }
        // Without the skip the first row's scale would normalize the bare
        // residual: prove the skip changed the bytes.
        let bare = {
            let summed: Vec<f32> = residual[..hidden as usize].to_vec();
            let sumsq: f32 = summed.iter().map(|value| value * value).sum();
            1.0 / ((sumsq / hidden as f32 + epsilon).sqrt())
        };
        let with_skip = expected[0] / gamma[0];
        assert!((with_skip - bare).abs() > 1.0e-3);
        // PB-8: the residual stream must carry the skip so the layer-end
        // residual add composes h + o + down.
        let written_residual = runtime
            .readback_f32(&residual_handle)
            .expect("read residual buffer");
        for (actual, want) in written_residual
            .iter()
            .zip(residual.iter().zip(&skip).map(|(a, b)| a + b))
        {
            assert!(
                (actual - want).abs() <= 1.0e-6 * want.abs().max(1.0),
                "expected accumulated {want}, wrote {actual}"
            );
        }
    }

    #[test]
    fn residual_rms_bridge_writeback_splices_a_view_span_not_the_whole_buffer() {
        use crate::device_host::DeviceRuntime;
        use crate::metal_host::{FakeMetalDriver, MetalHostSession};

        let mut runtime = DeviceRuntime::Metal(
            MetalHostSession::with_driver(Box::new(FakeMetalDriver::default()))
                .expect("fake Metal admission"),
        );
        // One pooled buffer holding two spans; the residual view is the
        // second span, so a whole-buffer overwrite would corrupt the first.
        let pooled = [9.0f32, 9.0, 9.0, 9.0, 1.0, -2.0, 3.0, -4.0];
        let skip = [0.25f32, 0.25, -0.25, -0.25];
        let gamma = [1.0f32, 0.5, 2.0, 1.5];
        let alloc = |runtime: &mut DeviceRuntime, values: &[f32]| {
            let handle = runtime
                .alloc_bytes(values.len() * 4)
                .expect("allocate bridge buffer");
            runtime
                .copy_in_f32(&handle, values)
                .expect("initialize bridge buffer");
            handle
        };
        let pooled_handle = alloc(&mut runtime, &pooled);
        let skip_handle = alloc(&mut runtime, &skip);
        let gamma_handle = alloc(&mut runtime, &gamma);
        let output_handle = alloc(&mut runtime, &vec![0.0f32; 4]);
        fn view<'a>(
            handle: &'a host_coordinator::DeviceHandle,
            count: u64,
            offset: u64,
            binding: u32,
        ) -> FusedLibraryDeviceBuffer<'a> {
            FusedLibraryDeviceBuffer {
                handle,
                dtype: DeviceDataType::F32,
                byte_offset: offset,
                view_span: count * 4,
                binding_index: binding,
                packed_format: None,
            }
        }
        dispatch_fused_residual_rms_device(
            &mut runtime,
            FusedResidualRmsDeviceDispatch {
                library_entry: "ResidualRmsNorm",
                rows: 1,
                hidden: 4,
                epsilon: 1.0e-6,
                residual: view(&pooled_handle, 4, 16, 0),
                skip: view(&skip_handle, 4, 0, 1),
                gamma: view(&gamma_handle, 4, 0, 2),
                output: view(&output_handle, 4, 0, 3),
            },
        )
        .expect("fused residual/RMS body executes");
        let pooled_after = runtime
            .readback_f32(&pooled_handle)
            .expect("read pooled buffer");
        for (actual, want) in pooled_after.iter().zip(
            pooled
                .iter()
                .enumerate()
                .map(|(index, value)| value + skip.get(index.wrapping_sub(4)).copied().unwrap_or(0.0)),
        ) {
            assert!(
                (actual - want).abs() <= 1.0e-6 * want.abs().max(1.0),
                "expected pooled {want}, wrote {actual}"
            );
        }
    }

    #[test]
    fn residual_rms_bridge_rejects_foreign_entry() {
        use crate::device_host::DeviceRuntime;
        use crate::metal_host::{FakeMetalDriver, MetalHostSession};
        use host_coordinator::DeviceHandle;

        let mut runtime = DeviceRuntime::Metal(
            MetalHostSession::with_driver(Box::new(FakeMetalDriver::default()))
                .expect("fake Metal admission"),
        );
        let handle = runtime
            .alloc_bytes(16)
            .expect("allocate bridge buffer");
        let view = FusedLibraryDeviceBuffer {
            handle: &handle,
            dtype: DeviceDataType::F32,
            byte_offset: 0,
            view_span: 16,
            binding_index: 0,
            packed_format: None,
        };
        let error = dispatch_fused_residual_rms_device(
            &mut runtime,
            FusedResidualRmsDeviceDispatch {
                library_entry: "QkvProjection",
                rows: 1,
                hidden: 4,
                epsilon: 1.0e-6,
                residual: view,
                skip: view,
                gamma: view,
                output: view,
            },
        )
        .expect_err("wrong runtime entry must fail closed");
        assert!(error.message.contains("disagrees with library_entry"));
    }

    #[test]
    fn production_bridge_consumes_packed_qkv_views_with_explicit_format() {
        use crate::device_host::DeviceRuntime;
        use crate::metal_host::{FakeMetalDriver, MetalHostSession};
        use host_coordinator::{DeviceBackend, DeviceHandle};

        fn f32_view<'a>(
            handle: &'a DeviceHandle,
            count: u64,
            binding: u32,
        ) -> FusedLibraryDeviceBuffer<'a> {
            FusedLibraryDeviceBuffer {
                handle,
                dtype: DeviceDataType::F32,
                byte_offset: 0,
                view_span: count * 4,
                binding_index: binding,
                packed_format: None,
            }
        }

        fn packed_view<'a>(
            handle: &'a DeviceHandle,
            bytes: u64,
            binding: u32,
        ) -> FusedLibraryDeviceBuffer<'a> {
            FusedLibraryDeviceBuffer {
                handle,
                dtype: DeviceDataType::U8,
                byte_offset: 0,
                view_span: bytes,
                binding_index: binding,
                packed_format: Some(PackedStorageFormat::Q8_0),
            }
        }

        fn q8_columns(value: u8, columns: usize) -> Vec<u8> {
            let mut bytes = Vec::with_capacity(columns * 34);
            for _ in 0..columns {
                bytes.extend_from_slice(&[0x00, 0x3c]);
                bytes.extend(std::iter::repeat_n(value, 32));
            }
            bytes
        }

        let mut runtime = DeviceRuntime::Metal(
            MetalHostSession::with_driver(Box::new(FakeMetalDriver::default()))
                .expect("fake Metal admission"),
        );
        let hidden = 32u64;
        let alloc_f32 = |runtime: &mut DeviceRuntime, values: &[f32]| {
            let handle = runtime
                .alloc_bytes(values.len() * 4)
                .expect("allocate f32 bridge buffer");
            runtime
                .copy_in_f32(&handle, values)
                .expect("initialize f32 bridge buffer");
            handle
        };
        let alloc_bytes = |runtime: &mut DeviceRuntime, values: &[u8]| {
            let handle = runtime
                .alloc_bytes(values.len())
                .expect("allocate packed bridge buffer");
            runtime
                .copy_in_bytes(&handle, values, DeviceDataType::U8)
                .expect("initialize packed bridge buffer");
            handle
        };
        let activation = alloc_f32(&mut runtime, &[1.0; 32]);
        let q_weight_bytes = q8_columns(1, 32);
        let k_weight_bytes = q8_columns(2, 32);
        let v_weight_bytes = q8_columns(3, 32);
        let q_weight = alloc_bytes(&mut runtime, &q_weight_bytes);
        let k_weight = alloc_bytes(&mut runtime, &k_weight_bytes);
        let v_weight = alloc_bytes(&mut runtime, &v_weight_bytes);
        let q_output = alloc_f32(&mut runtime, &[0.0; 32]);
        let k_output = alloc_f32(&mut runtime, &[0.0; 32]);
        let v_output = alloc_f32(&mut runtime, &[0.0; 32]);

        dispatch_fused_qkv_device(
            &mut runtime,
            FusedQkvDeviceDispatch {
                library_entry: "QkvProjection",
                derived_entry: "prefill_blk_0_QkvProjection",
                decode_gemv: 1,
                bind: QkvProjectionBind::grouped(1, hidden, 1, 1, 32, [32, 1, 1]),
                activation: FusedLibraryDeviceBuffer {
                    handle: &activation,
                    dtype: DeviceDataType::F32,
                    byte_offset: 0,
                    view_span: hidden * 4,
                    binding_index: 0,
                    packed_format: None,
                },
                weights: [
                    packed_view(&q_weight, q_weight_bytes.len() as u64, 1),
                    packed_view(&k_weight, k_weight_bytes.len() as u64, 2),
                    packed_view(&v_weight, v_weight_bytes.len() as u64, 3),
                ],
                biases: [None, None, None],
                rope: None,
                outputs: [
                    f32_view(&q_output, hidden, 4),
                    f32_view(&k_output, hidden, 5),
                    f32_view(&v_output, hidden, 6),
                ],
            },
        )
        .expect("packed QKV bridge executes");

        let q = runtime.readback_f32(&q_output).expect("read packed Q");
        let k = runtime.readback_f32(&k_output).expect("read packed K");
        let v = runtime.readback_f32(&v_output).expect("read packed V");
        assert!(q.iter().all(|value| *value == 32.0));
        assert!(k.iter().all(|value| *value == 64.0));
        assert!(v.iter().all(|value| *value == 96.0));
        assert_eq!(runtime.backend(), DeviceBackend::Metal);
    }

    #[test]
    fn production_bridge_admits_f32_word_tagged_packed_regions() {
        use crate::device_host::DeviceRuntime;
        use crate::metal_host::{FakeMetalDriver, MetalHostSession};
        use host_coordinator::{DeviceBackend, DeviceHandle};

        // The wire types a padded packed region as f32 words (byte length/4)
        // while the bytes keep the native GGML layout; the uploaded format
        // fact, not the dtype tag, selects the packed path.
        fn f32_word_view<'a>(
            handle: &'a DeviceHandle,
            words: u64,
            binding: u32,
        ) -> FusedLibraryDeviceBuffer<'a> {
            FusedLibraryDeviceBuffer {
                handle,
                dtype: DeviceDataType::F32,
                byte_offset: 0,
                view_span: words * 4,
                binding_index: binding,
                packed_format: Some(PackedStorageFormat::Q8_0),
            }
        }

        fn f32_view<'a>(
            handle: &'a DeviceHandle,
            count: u64,
            binding: u32,
        ) -> FusedLibraryDeviceBuffer<'a> {
            FusedLibraryDeviceBuffer {
                handle,
                dtype: DeviceDataType::F32,
                byte_offset: 0,
                view_span: count * 4,
                binding_index: binding,
                packed_format: None,
            }
        }

        fn q8_columns(value: u8, columns: usize) -> Vec<u8> {
            let mut bytes = Vec::with_capacity(columns * 34);
            for _ in 0..columns {
                bytes.extend_from_slice(&[0x00, 0x3c]);
                bytes.extend(std::iter::repeat_n(value, 32));
            }
            bytes
        }

        let mut runtime = DeviceRuntime::Metal(
            MetalHostSession::with_driver(Box::new(FakeMetalDriver::default()))
                .expect("fake Metal admission"),
        );
        let hidden = 32u64;
        let alloc_f32 = |runtime: &mut DeviceRuntime, values: &[f32]| {
            let handle = runtime
                .alloc_bytes(values.len() * 4)
                .expect("allocate f32 bridge buffer");
            runtime
                .copy_in_f32(&handle, values)
                .expect("initialize f32 bridge buffer");
            handle
        };
        let alloc_bytes = |runtime: &mut DeviceRuntime, values: &[u8]| {
            let handle = runtime
                .alloc_bytes(values.len())
                .expect("allocate packed bridge buffer");
            runtime
                .copy_in_bytes(&handle, values, DeviceDataType::U8)
                .expect("initialize packed bridge buffer");
            handle
        };
        let activation = alloc_f32(&mut runtime, &[1.0; 32]);
        let q_weight_bytes = q8_columns(1, 32);
        let k_weight_bytes = q8_columns(2, 32);
        let v_weight_bytes = q8_columns(3, 32);
        // Pad each region to a whole f32 word count, as the wire does.
        let words = |bytes: &[u8]| -> u64 { bytes.len().div_ceil(4) as u64 };
        let q_weight = alloc_bytes(&mut runtime, &q_weight_bytes);
        let k_weight = alloc_bytes(&mut runtime, &k_weight_bytes);
        let v_weight = alloc_bytes(&mut runtime, &v_weight_bytes);
        let q_output = alloc_f32(&mut runtime, &[0.0; 32]);
        let k_output = alloc_f32(&mut runtime, &[0.0; 32]);
        let v_output = alloc_f32(&mut runtime, &[0.0; 32]);

        dispatch_fused_qkv_device(
            &mut runtime,
            FusedQkvDeviceDispatch {
                library_entry: "QkvProjection",
                derived_entry: "prefill_blk_0_QkvProjection",
                decode_gemv: 1,
                bind: QkvProjectionBind::grouped(1, hidden, 1, 1, 32, [32, 1, 1]),
                activation: f32_view(&activation, hidden, 0),
                weights: [
                    f32_word_view(&q_weight, words(&q_weight_bytes), 1),
                    f32_word_view(&k_weight, words(&k_weight_bytes), 2),
                    f32_word_view(&v_weight, words(&v_weight_bytes), 3),
                ],
                biases: [None, None, None],
                rope: None,
                outputs: [
                    f32_view(&q_output, hidden, 4),
                    f32_view(&k_output, hidden, 5),
                    f32_view(&v_output, hidden, 6),
                ],
            },
        )
        .expect("f32-word-tagged packed QKV bridge executes");

        let q = runtime.readback_f32(&q_output).expect("read packed Q");
        let k = runtime.readback_f32(&k_output).expect("read packed K");
        let v = runtime.readback_f32(&v_output).expect("read packed V");
        assert!(q.iter().all(|value| *value == 32.0));
        assert!(k.iter().all(|value| *value == 64.0));
        assert!(v.iter().all(|value| *value == 96.0));
        assert_eq!(runtime.backend(), DeviceBackend::Metal);
    }
}
