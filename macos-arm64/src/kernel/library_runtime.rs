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

use super::library::{
    dispatch_selected, BindDescriptor, BindLayout, KernelBodyError, LibraryDispatch,
    QkvProjectionBind, QkvProjectionLayout, QkvProjectionWeight,
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
        let expected_hidden = self
            .kv_heads
            .checked_mul(self.q_per_kv)
            .and_then(|heads| heads.checked_mul(self.head_dim))
            .ok_or(KernelBodyError::InvalidBind(
                "fused library Metal module hidden width overflow",
            ))?;
        if expected_hidden != self.hidden {
            return Err(KernelBodyError::ShapeMismatch(
                "fused library Metal module hidden width does not match GQA facts",
            ));
        }
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
    fn runtime_msl_rejects_gqa_width_drift() {
        let error = library_family_msl(&LibraryFamilyMslFacts {
            rows: 1,
            hidden: 8,
            kv_heads: 1,
            q_per_kv: 1,
            head_dim: 4,
            epsilon: 1.0e-5,
        })
        .expect_err("hidden/GQA drift must fail closed");
        assert!(matches!(
            error,
            KernelBodyError::ShapeMismatch(message) if message.contains("hidden width")
        ));
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
}
