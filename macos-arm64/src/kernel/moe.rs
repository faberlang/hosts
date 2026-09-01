//! MoE router-selection family + the minted Metal grouped-dispatch bodies.
//!
//! This module owns the device-side router selection body (M8-U1b placement
//! ruling): one family body computes the packed router GEMV logits and then
//! the deterministic top-k/softmax selection, writing the grouped-dispatch
//! `expert_ids` / `expert_weights` device buffers.  The selection policy
//! mirrors the PM3 host seam (`radix/crates/faber/src/package/device/
//! router_selection.rs`) exactly: descending logit order, lower expert id
//! wins an equal-logit tie, softmax is evaluated only over the selected
//! experts after subtracting the selected-row maximum, and any non-finite
//! logit fails closed.
//!
//! The minted Metal bodies are the real grouped-dispatch kernels for this
//! backend: `router_selection` writes the ids/weights buffers on the device
//! and `grouped_expert_gemm` reads them back from device buffers.  The CPU
//! bodies in this module and in [`super::library`] are the parity references
//! those Metal bodies must match.  Plan recognition stays in radix-mir
//! (M8-U1a); this module owns the family bodies and the module assembly.

use super::library::{KernelBodyError, QuantizedFormat, block_value};

/// The only router-weight layout currently minted by the family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterSelectionLayout {
    /// Router weights are packed per expert: `[expert, k]` blocks in the
    /// landed per-format block layout.
    Packed,
    /// Sentinel for a caller that has not proved a servable layout.
    Unsupported,
}

/// Bind facts for one device-side router selection invocation.
///
/// Activations are logical `[rows, k]`, router weights are packed
/// `[experts, k]`, and the selection writes `[rows, active]` expert ids and
/// weights.  The body never infers a span from a byte-buffer length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouterSelectionBind {
    /// Number of activation/selection rows (`M`).
    pub rows: u64,
    /// Contracted activation width (`K`).
    pub k: u64,
    /// Declared expert count (`E`, the router logit width).
    pub experts: u64,
    /// Declared active experts per row (top-k width).
    pub active: u64,
    /// Physical element stride between activation rows.
    pub input_row_stride: u64,
    /// Physical byte stride between packed expert weight rows.
    pub packed_expert_stride_bytes: u64,
    /// Packed router-weight format delegated to the landed per-format
    /// substrate.
    pub format: QuantizedFormat,
    /// Shape family selected by the executor.
    pub layout: RouterSelectionLayout,
    /// Backend-neutral launch grid carried into the plan record.
    pub grid: [u32; 3],
}

impl RouterSelectionBind {
    /// Construct a contiguous packed router bind.
    #[must_use]
    pub fn packed(
        rows: u64,
        k: u64,
        experts: u64,
        active: u64,
        format: QuantizedFormat,
        grid: [u32; 3],
    ) -> Self {
        let packed_expert_stride_bytes = k
            .div_ceil(format.block_elements())
            .saturating_mul(format.block_bytes());
        Self {
            rows,
            k,
            experts,
            active,
            input_row_stride: k,
            packed_expert_stride_bytes,
            format,
            layout: RouterSelectionLayout::Packed,
            grid,
        }
    }

    /// Validate shape, layout, stride, policy, and dispatch facts before a
    /// router body touches any buffer.
    pub fn validate(&self) -> Result<(), KernelBodyError> {
        if self.rows == 0 || self.k == 0 || self.experts == 0 || self.active == 0 {
            return Err(KernelBodyError::InvalidBind(
                "router selection has a zero dimension",
            ));
        }
        if self.active > self.experts {
            return Err(KernelBodyError::InvalidBind(
                "router selection active experts exceed the declared expert count",
            ));
        }
        if self.layout != RouterSelectionLayout::Packed {
            return Err(KernelBodyError::InvalidBind(
                "router selection layout is not servable",
            ));
        }
        if self.input_row_stride == 0 || self.packed_expert_stride_bytes == 0 {
            return Err(KernelBodyError::InvalidBind(
                "router selection has a zero stride",
            ));
        }
        if self.grid.iter().any(|axis| *axis == 0) {
            return Err(KernelBodyError::InvalidBind(
                "router selection has a zero dispatch axis",
            ));
        }
        if !self.k.is_multiple_of(self.format.block_elements()) {
            return Err(KernelBodyError::InvalidBind(
                "router selection K is not block aligned",
            ));
        }
        let blocks_per_expert = self.k / self.format.block_elements();
        let minimum_expert_bytes = blocks_per_expert
            .checked_mul(self.format.block_bytes())
            .ok_or(KernelBodyError::InvalidBind(
                "router selection packed expert span overflow",
            ))?;
        if self.packed_expert_stride_bytes < minimum_expert_bytes {
            return Err(KernelBodyError::InvalidBind(
                "router selection packed expert stride is smaller than its block span",
            ));
        }
        Ok(())
    }
}

/// The selected router body dispatch key.
///
/// Plan recognition stays outside this module; the executor provides the
/// selection only after the expert policy is declared (M8-U1a fail-closed
/// recognition).  Unknown layouts fail in [`RouterSelectionBind::validate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterSelectionKernel {
    /// Device-side router GEMV + deterministic top-k/softmax writing the
    /// grouped-dispatch ids/weights device buffers.
    Device,
}

fn checked_usize(value: u64) -> Result<usize, KernelBodyError> {
    usize::try_from(value)
        .map_err(|_| KernelBodyError::InvalidBind("router selection index exceeds host usize"))
}

fn input_span(bind: &RouterSelectionBind) -> Result<u64, KernelBodyError> {
    bind.rows
        .checked_sub(1)
        .and_then(|last| last.checked_mul(bind.input_row_stride))
        .and_then(|last| last.checked_add(bind.k))
        .ok_or(KernelBodyError::InvalidBind(
            "router selection input span overflow",
        ))
}

fn weight_span(bind: &RouterSelectionBind) -> Result<u64, KernelBodyError> {
    let blocks_per_expert = bind.k / bind.format.block_elements();
    let per_expert = blocks_per_expert
        .checked_mul(bind.format.block_bytes())
        .ok_or(KernelBodyError::InvalidBind(
            "router selection packed expert span overflow",
        ))?;
    bind.experts
        .checked_sub(1)
        .and_then(|last| last.checked_mul(bind.packed_expert_stride_bytes))
        .and_then(|last| last.checked_add(per_expert))
        .ok_or(KernelBodyError::InvalidBind(
            "router selection weight span overflow",
        ))
}

fn selection_span(rows: u64, active: u64) -> Result<u64, KernelBodyError> {
    rows.checked_mul(active).ok_or(KernelBodyError::InvalidBind(
        "router selection span overflow",
    ))
}

fn checked_buffer(name: &'static str, required: u64, actual: usize) -> Result<(), KernelBodyError> {
    if required > actual as u64 {
        return Err(KernelBodyError::BufferTooShort {
            buffer: name,
            required,
            actual,
        });
    }
    Ok(())
}

/// Deterministic top-k + softmax over one row of router logits.
///
/// This is the CPU parity reference for the minted `router_selection` Metal
/// body.  The arithmetic mirrors the PM3 host seam exactly: larger logits
/// win, a lower expert id wins an equal-logit tie, and softmax is evaluated
/// only over the selected experts after subtracting the selected-row
/// maximum.  Any non-finite logit fails closed before selection.
fn select_row(
    logits: &[f32],
    active: usize,
    expert_ids: &mut [u32],
    expert_weights: &mut [f32],
    row: u64,
) -> Result<(), KernelBodyError> {
    if let Some((expert, _)) = logits
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(KernelBodyError::NonFiniteLogit {
            row,
            expert: expert as u64,
        });
    }
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|(left_index, left), (right_index, right)| {
        // Non-finite logits were rejected above, so `partial_cmp` is total
        // here; the fallback is never taken and keeps the seam's ordering.
        right
            .partial_cmp(left)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left_index.cmp(right_index))
    });
    ranked.truncate(active);

    let max_logit = ranked[0].1;
    let mut weights: Vec<f32> = ranked
        .iter()
        .map(|(_, logit)| (*logit - max_logit).exp())
        .collect();
    let normalizer: f32 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= normalizer;
    }
    for (slot, ((expert, _), weight)) in ranked.iter().zip(&weights).enumerate() {
        expert_ids[slot] = *expert as u32;
        expert_weights[slot] = *weight;
    }
    Ok(())
}

/// Execute the device-side router selection body: packed router GEMV over
/// the activation producing one logit per declared expert, then the
/// deterministic top-k/softmax selection writing `expert_ids` /
/// `expert_weights`.
pub fn router_selection(
    bind: &RouterSelectionBind,
    activation: &[f32],
    router_weight: &[u8],
    expert_ids: &mut [u32],
    expert_weights: &mut [f32],
) -> Result<(), KernelBodyError> {
    bind.validate()?;
    let input_span = input_span(bind)?;
    let weight_span = weight_span(bind)?;
    let selection_span = selection_span(bind.rows, bind.active)?;
    checked_buffer("router selection activation", input_span, activation.len())?;
    checked_buffer(
        "router selection packed weights",
        weight_span,
        router_weight.len(),
    )?;
    checked_buffer(
        "router selection expert ids",
        selection_span,
        expert_ids.len(),
    )?;
    checked_buffer(
        "router selection expert weights",
        selection_span,
        expert_weights.len(),
    )?;

    let rows = checked_usize(bind.rows)?;
    let k = checked_usize(bind.k)?;
    let experts = checked_usize(bind.experts)?;
    let active = checked_usize(bind.active)?;
    let input_row_stride = checked_usize(bind.input_row_stride)?;
    let expert_stride = checked_usize(bind.packed_expert_stride_bytes)?;
    let block_elements = checked_usize(bind.format.block_elements())?;
    let block_bytes = checked_usize(bind.format.block_bytes())?;
    let blocks = k / block_elements;

    let mut logits = vec![0.0f32; experts];
    for row in 0..rows {
        for expert in 0..experts {
            let mut logit = 0.0f32;
            for block_index in 0..blocks {
                let block_base = expert * expert_stride + block_index * block_bytes;
                let block = &router_weight[block_base..block_base + block_bytes];
                let activation_base = row * input_row_stride + block_index * block_elements;
                for element in 0..block_elements {
                    let weight = block_value(bind.format, block, element)?;
                    logit += activation[activation_base + element] * weight;
                }
            }
            logits[expert] = logit;
        }
        let row_ids = &mut expert_ids[row * active..(row + 1) * active];
        let row_weights = &mut expert_weights[row * active..(row + 1) * active];
        select_row(&logits, active, row_ids, row_weights, row as u64)?;
    }
    Ok(())
}

/// Dispatch the already-selected router body.
pub fn dispatch_router_selection(
    kernel: RouterSelectionKernel,
    bind: &RouterSelectionBind,
    activation: &[f32],
    router_weight: &[u8],
    expert_ids: &mut [u32],
    expert_weights: &mut [f32],
) -> Result<(), KernelBodyError> {
    match kernel {
        RouterSelectionKernel::Device => {
            router_selection(bind, activation, router_weight, expert_ids, expert_weights)
        }
    }
}

/// Concrete geometry minted into the family Metal module.  The Metal bodies
/// follow the concrete-dim emitter convention: the plan facts are baked into
/// the MSL text, never read from a runtime uniform buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeFamilyMslFacts {
    /// Activation/output rows (`M`).
    pub rows: u64,
    /// Contracted activation width (`K`).
    pub k: u64,
    /// Output width per expert (`N`).
    pub n: u64,
    /// Total expert slices in the packed weight region (`E`).
    pub experts: u64,
    /// Active experts per row selected by the router (top-k width).
    pub active: u64,
    /// Packed weight format minted into the module.
    pub format: QuantizedFormat,
}

const Q8_0_BLOCK_ELEMENTS: u64 = 32;
const Q8_0_BLOCK_BYTES: u64 = 34;

/// Mint the real Metal grouped-dispatch module: the `router_selection` and
/// `grouped_expert_gemm` MSL kernels as one module image.
///
/// Only `Q8_0` is minted today (the R-PACK-03 probe format); any other
/// format fails closed rather than guessing a block geometry.  The router
/// kernel writes the grouped-dispatch ids/weights device buffers and the
/// grouped kernel reads them back from device buffers, so one resident step
/// never round-trips the router logits to the host.
pub fn moe_family_msl(facts: &MoeFamilyMslFacts) -> Result<String, KernelBodyError> {
    if facts.format != QuantizedFormat::Q8_0 {
        return Err(KernelBodyError::InvalidBind(
            "MoE family Metal module mints Q8_0 only; other formats are not servable",
        ));
    }
    if facts.rows == 0 || facts.k == 0 || facts.n == 0 || facts.experts == 0 || facts.active == 0 {
        return Err(KernelBodyError::InvalidBind(
            "MoE family Metal module has a zero dimension",
        ));
    }
    if facts.active > facts.experts {
        return Err(KernelBodyError::InvalidBind(
            "MoE family Metal module active experts exceed the declared expert count",
        ));
    }
    if !facts.k.is_multiple_of(Q8_0_BLOCK_ELEMENTS) {
        return Err(KernelBodyError::InvalidBind(
            "MoE family Metal module K is not Q8_0 block aligned",
        ));
    }
    let k_blocks = facts.k / Q8_0_BLOCK_ELEMENTS;
    let router_expert_stride =
        k_blocks
            .checked_mul(Q8_0_BLOCK_BYTES)
            .ok_or(KernelBodyError::InvalidBind(
                "MoE family Metal module router expert stride overflow",
            ))?;
    let column_stride = router_expert_stride;
    let expert_stride = facts
        .n
        .checked_mul(column_stride)
        .ok_or(KernelBodyError::InvalidBind(
            "MoE family Metal module expert stride overflow",
        ))?;

    Ok(format!(
        r#"#include <metal_stdlib>
using namespace metal;

constant uint ROWS = {rows}u;
constant uint K = {k}u;
constant uint N = {n}u;
constant uint EXPERTS = {experts}u;
constant uint ACTIVE = {active}u;
constant uint K_BLOCKS = {k_blocks}u;
constant uint BLOCK_ELEMENTS = {block_elements}u;
constant uint BLOCK_BYTES = {block_bytes}u;
constant uint ROUTER_EXPERT_STRIDE_BYTES = {router_expert_stride}u;
constant uint COLUMN_STRIDE_BYTES = {column_stride}u;
constant uint EXPERT_STRIDE_BYTES = {expert_stride}u;

static float q8_0_value(const device uchar* block, uint element) {{
    ushort hbits = ushort(block[0]) | (ushort(block[1]) << 8);
    float d = float(as_type<half>(hbits));
    int q = int(block[2 + element]);
    if (q >= 128) {{ q -= 256; }}
    return float(q) * d;
}}

kernel void router_selection(
    device const float* activation [[buffer(0)]],
    device const uchar* router_weight [[buffer(1)]],
    device uint* expert_ids [[buffer(2)]],
    device float* expert_weights [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {{
  if (id >= ROWS) {{ return; }}
  float logits[EXPERTS];
  bool finite = true;
  for (uint e = 0; e < EXPERTS; ++e) {{
    float sum = 0.0f;
    for (uint blk = 0; blk < K_BLOCKS; ++blk) {{
      const device uchar* block = router_weight + (e * ROUTER_EXPERT_STRIDE_BYTES) + (blk * BLOCK_BYTES);
      uint a_base = id * K + blk * BLOCK_ELEMENTS;
      for (uint el = 0; el < BLOCK_ELEMENTS; ++el) {{
        sum += activation[a_base + el] * q8_0_value(block, el);
      }}
    }}
    logits[e] = sum;
    if (!isfinite(sum)) {{ finite = false; }}
  }}
  if (!finite) {{
    for (uint slot = 0; slot < ACTIVE; ++slot) {{
      expert_ids[id * ACTIVE + slot] = 0xFFFFFFFFu;
      expert_weights[id * ACTIVE + slot] = NAN;
    }}
    return;
  }}
  bool picked[EXPERTS];
  for (uint e = 0; e < EXPERTS; ++e) {{ picked[e] = false; }}
  for (uint slot = 0; slot < ACTIVE; ++slot) {{
    int best = -1;
    float best_logit = -INFINITY;
    for (uint e = 0; e < EXPERTS; ++e) {{
      if (picked[e]) {{ continue; }}
      if (best < 0 || logits[e] > best_logit || (logits[e] == best_logit && e < uint(best))) {{
        best = int(e);
        best_logit = logits[e];
      }}
    }}
    picked[uint(best)] = true;
    expert_ids[id * ACTIVE + slot] = uint(best);
  }}
  float max_logit = logits[expert_ids[id * ACTIVE]];
  float normalizer = 0.0f;
  for (uint slot = 0; slot < ACTIVE; ++slot) {{
    float w = exp(logits[expert_ids[id * ACTIVE + slot]] - max_logit);
    expert_weights[id * ACTIVE + slot] = w;
    normalizer += w;
  }}
  for (uint slot = 0; slot < ACTIVE; ++slot) {{
    expert_weights[id * ACTIVE + slot] /= normalizer;
  }}
}}

kernel void grouped_expert_gemm(
    device const float* activation [[buffer(0)]],
    device const uint* expert_ids [[buffer(1)]],
    device const float* expert_weights [[buffer(2)]],
    device const uchar* packed_weight [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {{
  if (id >= ROWS * N) {{ return; }}
  uint row = id / N;
  uint col = id % N;
  float accumulated = 0.0f;
  bool poisoned = false;
  for (uint slot = 0; slot < ACTIVE; ++slot) {{
    uint expert = expert_ids[row * ACTIVE + slot];
    if (expert >= EXPERTS) {{ poisoned = true; break; }}
    float weight = expert_weights[row * ACTIVE + slot];
    float intermediate = 0.0f;
    for (uint blk = 0; blk < K_BLOCKS; ++blk) {{
      const device uchar* block = packed_weight
          + (expert * EXPERT_STRIDE_BYTES)
          + (col * COLUMN_STRIDE_BYTES)
          + (blk * BLOCK_BYTES);
      uint a_base = row * K + blk * BLOCK_ELEMENTS;
      for (uint el = 0; el < BLOCK_ELEMENTS; ++el) {{
        intermediate += activation[a_base + el] * q8_0_value(block, el);
      }}
    }}
    accumulated += weight * intermediate;
  }}
  output[row * N + col] = poisoned ? NAN : accumulated;
}}
"#,
        rows = facts.rows,
        k = facts.k,
        n = facts.n,
        experts = facts.experts,
        active = facts.active,
        k_blocks = k_blocks,
        block_elements = Q8_0_BLOCK_ELEMENTS,
        block_bytes = Q8_0_BLOCK_BYTES,
        router_expert_stride = router_expert_stride,
        column_stride = column_stride,
        expert_stride = expert_stride,
    ))
}

#[cfg(test)]
#[path = "moe_test.rs"]
mod tests;
