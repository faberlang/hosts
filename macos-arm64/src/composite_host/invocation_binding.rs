//! Pure v2 cursor-to-input projection (PPE-P2).
//!
//! A protocol-v2 step carries cursor facts, not a second JSON input map. This
//! module turns those facts into the small set of values declared by the
//! selected dense descriptor. It deliberately has no device or session
//! dependency, so malformed cursor facts fail before dispatch.

use std::collections::{BTreeMap, BTreeSet};

use crate::device_descriptor::{DeviceBufferRole, DeviceDataType, DeviceDescriptor};
use crate::device_execute::{DeviceExecuteInvocation, DeviceExecuteInvocationMode};
use crate::kernel::{HostError, HostResult};

/// Dense descriptor input name for the token row.
pub const PROMPT_TOKENS: &str = "prompt_tokens";
/// Dense descriptor input name for the RoPE cosine table.
pub const ROPE_COS: &str = "prefill.rope.cos";
/// Dense descriptor input name for the RoPE sine table.
pub const ROPE_SIN: &str = "prefill.rope.sin";
/// Dense descriptor input name for Q projection prefix ids.
pub const Q_PREFIX_IDS: &str = "decode.q_prefix_ids";
/// Dense descriptor input name for K/V projection prefix ids.
pub const KV_PREFIX_IDS: &str = "decode.kv_prefix_ids";
/// Dense descriptor input name for the four-field B1 invocation cursor.
pub const INVOCATION_STATE: &str = "kv.invocation_state";

/// Parameters needed to materialize one or more RoPE rows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeConfig {
    /// Full attention head width. The descriptor carries the resulting row
    /// width; this value supplies the numeric RoPE contract.
    pub head_dim: u32,
    /// RoPE frequency base (for example, 10_000.0).
    pub theta: f64,
}

impl RopeConfig {
    fn validate(self) -> HostResult<()> {
        if self.head_dim == 0 || self.head_dim % 2 != 0 {
            return Err(invalid_args(format!(
                "RoPE head_dim must be a nonzero even number; got {}",
                self.head_dim
            )));
        }
        if !self.theta.is_finite() || self.theta <= 0.0 {
            return Err(invalid_args(format!(
                "RoPE theta must be finite and positive; got {}",
                self.theta
            )));
        }
        Ok(())
    }

    fn row_width(self) -> usize {
        (self.head_dim / 2) as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InputSpec {
    buffer_id: u32,
    element_count: u64,
    element_ty: DeviceDataType,
}

fn invalid_args(message: impl Into<String>) -> HostError {
    HostError::invalid_args(message)
}

/// Project one v2 invocation into the selected descriptor's dynamic inputs.
///
/// The result is keyed by descriptor buffer id, matching the host session's
/// input surface. Per-program model weights and every other declared input are
/// intentionally excluded: they are prepared once by the resident owner.
///
/// Prefill emits the supplied prompt and the complete prompt-length RoPE
/// table. Scalar decode emits one token, one RoPE row at the explicit absolute
/// `position`, and the declared gather-index tables (`decode.q_prefix_ids` /
/// `decode.kv_prefix_ids`) as `0..element_count`. Those ids address the
/// GEMV-padded logical Q/K/V span, not the sequence cursor. The supplied
/// prompt is ignored for decode, so a caller cannot accidentally upload a
/// full prompt on a scalar step.
///
/// # Errors
/// Returns `E_INVALID_ARGS` for malformed cursor arithmetic, a mode/query-row
/// mismatch, missing or conflicting declared dynamic inputs, and shape/dtype
/// mismatches. No device operation occurs in this function.
pub fn project_invocation_bindings(
    descriptor: &DeviceDescriptor,
    invocation: &DeviceExecuteInvocation,
    prompt_tokens: &[u32],
    rope: RopeConfig,
) -> HostResult<BTreeMap<u32, Vec<f32>>> {
    rope.validate()?;
    let query_rows = query_rows(invocation, prompt_tokens)?;
    validate_cursor(invocation, query_rows)?;

    let required = match invocation.mode {
        DeviceExecuteInvocationMode::Prefill => [PROMPT_TOKENS, ROPE_COS, ROPE_SIN, "", ""],
        DeviceExecuteInvocationMode::ScalarDecode => {
            [PROMPT_TOKENS, ROPE_COS, ROPE_SIN, Q_PREFIX_IDS, ""]
        }
    };
    let required = required.into_iter().filter(|name| !name.is_empty());
    let inputs = declared_input_specs(descriptor, required)?;

    let mut projected = BTreeMap::new();
    let token_values = match invocation.mode {
        DeviceExecuteInvocationMode::Prefill => encode_ids(prompt_tokens),
        DeviceExecuteInvocationMode::ScalarDecode => {
            let token = invocation
                .token
                .ok_or_else(|| invalid_args("scalar decode projection requires one token row"))?;
            encode_ids(&[token])
        }
    };
    insert_checked(
        &mut projected,
        PROMPT_TOKENS,
        inputs.get(PROMPT_TOKENS),
        token_values,
    )?;

    let cos_spec = inputs.get(ROPE_COS);
    let pairs = rope.row_width();
    let declared_rope = cos_spec.map(|spec| spec.element_count).unwrap_or(0);
    let rope_rows_count = if pairs == 0 {
        0
    } else {
        declared_rope / pairs as u64
    };
    let positions: Vec<u32> =
        if invocation.mode == DeviceExecuteInvocationMode::ScalarDecode && rope_rows_count <= 1 {
            vec![invocation.position]
        } else if invocation.mode == DeviceExecuteInvocationMode::Prefill {
            (0..query_rows).collect()
        } else {
            (0..u32::try_from(rope_rows_count).unwrap_or(u32::MAX)).collect()
        };
    let (cos, sin) = rope_rows(&positions, rope);
    insert_checked(&mut projected, ROPE_COS, inputs.get(ROPE_COS), cos)?;
    insert_checked(&mut projected, ROPE_SIN, inputs.get(ROPE_SIN), sin)?;

    if let Some(cursor) = optional_input_spec(descriptor, INVOCATION_STATE)? {
        let values = vec![
            invocation.position as f32,
            invocation.valid_len_after as f32,
            query_rows as f32,
            invocation.sequence_epoch as f32,
        ];
        insert_checked(&mut projected, INVOCATION_STATE, Some(&cursor), values)?;
    }

    if invocation.mode == DeviceExecuteInvocationMode::ScalarDecode {
        insert_checked(
            &mut projected,
            Q_PREFIX_IDS,
            inputs.get(Q_PREFIX_IDS),
            prefix_ids_for(inputs.get(Q_PREFIX_IDS), Q_PREFIX_IDS)?,
        )?;
        if let Some(spec) = optional_input_spec(descriptor, KV_PREFIX_IDS)? {
            insert_checked(
                &mut projected,
                KV_PREFIX_IDS,
                Some(&spec),
                prefix_ids_for(Some(&spec), KV_PREFIX_IDS)?,
            )?;
        }
    }
    Ok(projected)
}

/// Alias emphasizing that the projection is the map consumed by a session's
/// input-copy surface.
pub fn project_invocation_inputs(
    descriptor: &DeviceDescriptor,
    invocation: &DeviceExecuteInvocation,
    prompt_tokens: &[u32],
    rope: RopeConfig,
) -> HostResult<BTreeMap<u32, Vec<f32>>> {
    project_invocation_bindings(descriptor, invocation, prompt_tokens, rope)
}

fn query_rows(invocation: &DeviceExecuteInvocation, prompt_tokens: &[u32]) -> HostResult<u32> {
    match invocation.mode {
        DeviceExecuteInvocationMode::Prefill => u32::try_from(prompt_tokens.len())
            .map_err(|_| invalid_args("prefill prompt length does not fit the v2 query-row field")),
        DeviceExecuteInvocationMode::ScalarDecode => {
            if invocation.token.is_none() {
                return Err(invalid_args(
                    "scalar decode projection requires one token row",
                ));
            }
            Ok(1)
        }
    }
}

fn validate_cursor(invocation: &DeviceExecuteInvocation, query_rows: u32) -> HostResult<()> {
    if query_rows == 0 {
        return Err(invalid_args(
            "v2 invocation must contain at least one query row",
        ));
    }
    if invocation.position != invocation.prefix_before {
        return Err(invalid_args(format!(
            "v2 cursor position {} must equal prefix_before {} for contiguous append",
            invocation.position, invocation.prefix_before
        )));
    }
    if invocation.query_start != invocation.position {
        return Err(invalid_args(format!(
            "v2 cursor query_start {} must equal position {} for contiguous append",
            invocation.query_start, invocation.position
        )));
    }
    if invocation.mode == DeviceExecuteInvocationMode::Prefill
        && (invocation.prefix_before != 0 || invocation.position != 0)
    {
        return Err(invalid_args(
            "prefill projection requires a fresh cursor at position 0",
        ));
    }
    let expected_after = invocation
        .prefix_before
        .checked_add(query_rows)
        .ok_or_else(|| {
            invalid_args(format!(
                "v2 cursor overflow: prefix_before {} + query_rows {}",
                invocation.prefix_before, query_rows
            ))
        })?;
    if invocation.valid_len_after != expected_after {
        return Err(invalid_args(format!(
            "v2 cursor valid_len_after {} does not equal prefix_before {} + query_rows {}",
            invocation.valid_len_after, invocation.prefix_before, query_rows
        )));
    }
    Ok(())
}

fn declared_input_specs<I>(
    descriptor: &DeviceDescriptor,
    required: I,
) -> HostResult<BTreeMap<&'static str, InputSpec>>
where
    I: Iterator<Item = &'static str>,
{
    let required: BTreeSet<&'static str> = required.collect();
    let mut specs = BTreeMap::new();
    for kernel in &descriptor.kernels {
        for slot in &kernel.buffers {
            if slot.role != DeviceBufferRole::Input {
                continue;
            }
            let Some(name) = required
                .iter()
                .copied()
                .find(|name| *name == slot.buffer_name)
            else {
                continue;
            };
            let next = InputSpec {
                buffer_id: slot.buffer_id,
                element_count: slot.element_count,
                element_ty: slot.element_ty,
            };
            if let Some(previous) = specs.get(name) {
                if *previous != next {
                    return Err(invalid_args(format!(
                        "declared input `{name}` has conflicting buffer identity or shape"
                    )));
                }
            } else {
                specs.insert(name, next);
            }
        }
    }
    for name in required {
        if !specs.contains_key(name) {
            return Err(invalid_args(format!(
                "selected descriptor does not declare required input `{name}`"
            )));
        }
    }
    let mut ids = BTreeMap::new();
    for (name, spec) in &specs {
        if let Some(previous) = ids.insert(spec.buffer_id, *name) {
            return Err(invalid_args(format!(
                "declared dynamic inputs `{previous}` and `{name}` alias buffer id {}",
                spec.buffer_id
            )));
        }
    }
    Ok(specs)
}

fn optional_input_spec(
    descriptor: &DeviceDescriptor,
    name: &'static str,
) -> HostResult<Option<InputSpec>> {
    let mut found = None;
    for kernel in &descriptor.kernels {
        for slot in &kernel.buffers {
            if slot.role != DeviceBufferRole::Input || slot.buffer_name != name {
                continue;
            }
            let next = InputSpec {
                buffer_id: slot.buffer_id,
                element_count: slot.element_count,
                element_ty: slot.element_ty,
            };
            if let Some(previous) = found {
                if previous != next {
                    return Err(invalid_args(format!(
                        "declared input `{name}` has conflicting buffer identity or shape"
                    )));
                }
            } else {
                found = Some(next);
            }
        }
    }
    Ok(found)
}

fn prefix_ids_for(spec: Option<&InputSpec>, name: &str) -> HostResult<Vec<f32>> {
    let spec = spec.ok_or_else(|| invalid_args(format!("missing projected input `{name}`")))?;
    let count = u32::try_from(spec.element_count).map_err(|_| {
        invalid_args(format!(
            "declared input `{name}` element count does not fit the host"
        ))
    })?;
    Ok(encode_ids(&(0..count).collect::<Vec<_>>()))
}

fn insert_checked(
    projected: &mut BTreeMap<u32, Vec<f32>>,
    name: &str,
    spec: Option<&InputSpec>,
    values: Vec<f32>,
) -> HostResult<()> {
    let spec = spec.ok_or_else(|| invalid_args(format!("missing projected input `{name}`")))?;
    if spec.element_ty != DeviceDataType::F32 {
        return Err(invalid_args(format!(
            "projected input `{name}` uses {}; v2 projection only supports f32",
            spec.element_ty.spelling()
        )));
    }
    let expected = usize::try_from(spec.element_count).map_err(|_| {
        invalid_args(format!(
            "declared input `{name}` element count does not fit the host"
        ))
    })?;
    if values.len() != expected {
        return Err(invalid_args(format!(
            "projected input `{name}` has {} elements but the descriptor declares {}",
            values.len(),
            expected
        )));
    }
    projected.insert(spec.buffer_id, values);
    Ok(())
}

fn encode_ids(ids: &[u32]) -> Vec<f32> {
    ids.iter().map(|id| f32::from_bits(*id)).collect()
}

fn rope_rows(positions: &[u32], rope: RopeConfig) -> (Vec<f32>, Vec<f32>) {
    let pairs = rope.row_width();
    let mut cos = Vec::with_capacity(positions.len() * pairs);
    let mut sin = Vec::with_capacity(positions.len() * pairs);
    for position in positions {
        for pair in 0..pairs {
            let angle = f64::from(*position)
                * rope
                    .theta
                    .powf(-(2.0 * pair as f64) / f64::from(rope.head_dim));
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    (cos, sin)
}
