//! SSM state family module assembly for the resident schedule (M8-U2b).
//!
//! This module owns the minted Metal bodies for the SSM conv/scan state
//! families (R-PACK-04): `ssm_conv1d` (causal shared-kernel convolution over
//! `[length, channels]` state rows) and `ssm_scan` (additive linear
//! recurrence over `[length, state_dim]` rows).  The bodies are the real
//! device kernels; the CPU bodies in [`super::ssm_conv1d`] and
//! [`super::ssm_scan`] are the parity references they must match.  Plan
//! recognition stays in radix-mir (M8-U2a); this module owns the family
//! module assembly and the plan-dispatch seams.

use super::library::KernelBodyError;
use super::ssm_conv1d::{SsmConv1dBind, SsmConv1dKernel, dispatch_ssm_conv1d};
use super::ssm_scan::{SsmScanBind, SsmScanKernel, dispatch_ssm_scan};

/// Concrete geometry minted into the SSM family Metal module.  The Metal
/// bodies follow the concrete-dim emitter convention: the plan facts are
/// baked into the MSL text, never read from a runtime uniform buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SsmFamilyMslFacts {
    /// Number of time rows (`L`).
    pub length: u64,
    /// State channels (conv) / state values (scan) per time row (`D`).
    pub state_dim: u64,
    /// Number of causal convolution taps (`K`).
    pub kernel_width: u64,
}

/// Mint the real Metal SSM family module: the `ssm_conv1d` and `ssm_scan`
/// MSL kernels as one module image.
///
/// Only the plain F32 state layout the landed bodies prove is minted; the
/// state spans are `length * state_dim` f32 elements and the conv kernel is
/// `kernel_width` f32 elements, exactly the canonical `[length, state]`
/// channels-last storage admitted by the binds.
pub fn ssm_family_msl(facts: &SsmFamilyMslFacts) -> Result<String, KernelBodyError> {
    if facts.length == 0 || facts.state_dim == 0 || facts.kernel_width == 0 {
        return Err(KernelBodyError::InvalidBind(
            "SSM family Metal module has a zero dimension",
        ));
    }
    Ok(format!(
        r#"#include <metal_stdlib>
using namespace metal;

constant uint LENGTH = {length}u;
constant uint STATE_DIM = {state_dim}u;
constant uint KERNEL_WIDTH = {kernel_width}u;

kernel void ssm_conv1d(
    device const float* input [[buffer(0)]],
    device const float* conv_kernel [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {{
  if (id >= LENGTH * STATE_DIM) {{ return; }}
  uint t = id / STATE_DIM;
  uint c = id % STATE_DIM;
  float acc = 0.0f;
  for (uint tap = 0; tap < KERNEL_WIDTH; ++tap) {{
    if (tap <= t) {{
      acc += input[(t - tap) * STATE_DIM + c] * conv_kernel[tap];
    }}
  }}
  output[id] = acc;
}}

kernel void ssm_scan(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {{
  if (id >= STATE_DIM) {{ return; }}
  float carry = 0.0f;
  for (uint t = 0; t < LENGTH; ++t) {{
    carry += input[t * STATE_DIM + id];
    output[t * STATE_DIM + id] = carry;
  }}
}}
"#,
        length = facts.length,
        state_dim = facts.state_dim,
        kernel_width = facts.kernel_width,
    ))
}

/// The SSM state-op families admitted through the plan path (M8-U2a
/// recognition rows map to these dispatch seams).
#[derive(Debug)]
pub enum SsmFamilyDispatch<'a> {
    /// Causal SSM convolution over `[length, channels]` state rows.
    SsmConv1d {
        /// Declared library entry name; must name this family.
        library_entry: Option<&'a str>,
        /// Validated conv bind.
        bind: &'a SsmConv1dBind,
        /// State input `[length, channels]`.
        input: &'a [f32],
        /// Rank-1 causal kernel `[kernel_width]`.
        kernel: &'a [f32],
        /// Convolved state output `[length, channels]`.
        output: &'a mut [f32],
    },
    /// SSM linear recurrence over `[length, state_dim]` state rows.
    SsmScan {
        /// Declared library entry name; must name this family.
        library_entry: Option<&'a str>,
        /// Validated scan bind.
        bind: &'a SsmScanBind,
        /// State input `[length, state_dim]`.
        input: &'a [f32],
        /// Scanned state output `[length, state_dim]`.
        output: &'a mut [f32],
    },
}

/// Dispatch an admitted SSM family selection through the plan seam.
///
/// Mirrors the library dispatch contract: the declared library entry must
/// name the family or the selection fails closed before any buffer access.
pub fn ssm_family_dispatch(request: SsmFamilyDispatch<'_>) -> Result<(), KernelBodyError> {
    match request {
        SsmFamilyDispatch::SsmConv1d {
            library_entry,
            bind,
            input,
            kernel,
            output,
        } => {
            if let Some(entry) = library_entry {
                if entry != "SsmConv1d" {
                    return Err(KernelBodyError::InvalidBind(
                        "selection entry disagrees with library_entry SsmConv1d",
                    ));
                }
            }
            dispatch_ssm_conv1d(SsmConv1dKernel::Causal, bind, input, kernel, output)
        }
        SsmFamilyDispatch::SsmScan {
            library_entry,
            bind,
            input,
            output,
        } => {
            if let Some(entry) = library_entry {
                if entry != "SsmScan" {
                    return Err(KernelBodyError::InvalidBind(
                        "selection entry disagrees with library_entry SsmScan",
                    ));
                }
            }
            dispatch_ssm_scan(SsmScanKernel::Additive, bind, input, output)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssm_family_msl_fails_closed_on_zero_dimension() {
        let error = ssm_family_msl(&SsmFamilyMslFacts {
            length: 0,
            state_dim: 8,
            kernel_width: 4,
        })
        .expect_err("zero length must fail closed");
        assert!(matches!(
            error,
            KernelBodyError::InvalidBind(message) if message.contains("zero dimension")
        ));
    }

    #[test]
    fn ssm_family_dispatch_rejects_a_mismatched_library_entry() {
        let bind = SsmScanBind::decode(2, [2, 1, 1]);
        let input = [1.0f32, 2.0];
        let mut output = [0.0f32; 2];
        let error = ssm_family_dispatch(SsmFamilyDispatch::SsmScan {
            library_entry: Some("SsmConv1d"),
            bind: &bind,
            input: &input,
            output: &mut output,
        })
        .expect_err("wrong entry must fail closed");
        assert!(matches!(
            error,
            KernelBodyError::InvalidBind(message) if message.contains("disagrees with library_entry")
        ));
        assert_eq!(output, [0.0; 2], "failed selection must not write");
    }
}
