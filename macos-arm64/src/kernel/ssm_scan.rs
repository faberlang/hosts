//! Shape-generic Metal-library reference for the synthetic SSM scan.
//!
//! The body carries one rank-2 `[time, state]` state tensor and implements the
//! admitted additive recurrence `state[t, d] = state[t - 1, d] + input[t, d]`.
//! A sequence-length launch is the prefill arm; a length-one launch is the
//! decode/state-update arm.  Both arms share this bind-validated body, and no
//! state dimensions are inferred from buffer lengths.

use super::library::KernelBodyError;

/// The only state layout currently proven by this body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SsmScanLayout {
    /// Canonical rank-2 `[length, state_dim]` storage with state as the inner axis.
    StateLast,
    /// Sentinel for a caller that has not proved a servable state layout.
    Unsupported,
}

/// Explicit launch regime carried by the SSM scan bind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SsmScanRegime {
    /// Sequence scan over a prefill input.
    Prefill,
    /// One-token state update for decode.
    Decode,
}

/// Immutable facts bound to one SSM scan invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SsmScanBind {
    /// Number of time rows in the state input and output.
    pub length: u64,
    /// Number of independent state values in each time row.
    pub state_dim: u64,
    /// Physical element strides for logical input `[time, state]`.
    pub input_strides: [u64; 2],
    /// Physical element strides for logical output `[time, state]`.
    pub output_strides: [u64; 2],
    /// State layout selected by the executor.
    pub layout: SsmScanLayout,
    /// Prefill or length-one decode arm selected by the executor.
    pub regime: SsmScanRegime,
    /// Backend-neutral dispatch grid carried by the launch record.
    pub grid: [u32; 3],
}

impl SsmScanBind {
    /// Construct the canonical rank-2 prefill state layout.
    #[must_use]
    pub fn prefill(length: u64, state_dim: u64, grid: [u32; 3]) -> Self {
        Self {
            length,
            state_dim,
            input_strides: [state_dim, 1],
            output_strides: [state_dim, 1],
            layout: SsmScanLayout::StateLast,
            regime: SsmScanRegime::Prefill,
            grid,
        }
    }

    /// Construct the canonical rank-2 length-one decode state layout.
    #[must_use]
    pub fn decode(state_dim: u64, grid: [u32; 3]) -> Self {
        Self {
            length: 1,
            state_dim,
            input_strides: [state_dim, 1],
            output_strides: [state_dim, 1],
            layout: SsmScanLayout::StateLast,
            regime: SsmScanRegime::Decode,
            grid,
        }
    }

    /// Validate all shape, regime, layout, stride, and grid facts before
    /// either scan arm touches a buffer.
    pub fn validate(&self) -> Result<(), KernelBodyError> {
        if self.length == 0 || self.state_dim == 0 {
            return Err(KernelBodyError::InvalidBind(
                "SSM scan has a zero dimension",
            ));
        }
        if self.layout != SsmScanLayout::StateLast {
            return Err(KernelBodyError::InvalidBind(
                "SSM scan state layout is not servable",
            ));
        }
        if self.grid.iter().any(|axis| *axis == 0) {
            return Err(KernelBodyError::InvalidBind(
                "SSM scan bind has a zero dispatch axis",
            ));
        }
        let expected_state_strides = [self.state_dim, 1];
        if self.input_strides != expected_state_strides
            || self.output_strides != expected_state_strides
        {
            return Err(KernelBodyError::InvalidBind(
                "SSM scan state layout has non-canonical strides",
            ));
        }
        match self.regime {
            SsmScanRegime::Prefill if self.length < 2 => {
                return Err(KernelBodyError::InvalidBind(
                    "SSM scan prefill regime requires a sequence length greater than one",
                ));
            }
            SsmScanRegime::Decode if self.length != 1 => {
                return Err(KernelBodyError::InvalidBind(
                    "SSM scan decode regime requires a length-one launch",
                ));
            }
            SsmScanRegime::Prefill | SsmScanRegime::Decode => {}
        }
        state_span(self.length, self.state_dim, self.input_strides)?;
        state_span(self.length, self.state_dim, self.output_strides)?;
        Ok(())
    }
}

/// The selected body dispatch key.  Plan recognition remains outside this
/// module; this enum only prevents a caller from bypassing the body contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SsmScanKernel {
    /// Additive linear-recurrence scan used by the synthetic fixture tier.
    Additive,
}

fn checked_usize(value: u64) -> Result<usize, KernelBodyError> {
    usize::try_from(value)
        .map_err(|_| KernelBodyError::InvalidBind("SSM scan index exceeds host usize"))
}

fn state_span(length: u64, state_dim: u64, strides: [u64; 2]) -> Result<u64, KernelBodyError> {
    length
        .checked_sub(1)
        .and_then(|last| last.checked_mul(strides[0]))
        .and_then(|last| {
            state_dim
                .checked_sub(1)
                .and_then(|last_state| last_state.checked_mul(strides[1]))
                .and_then(|last_state| last.checked_add(last_state))
        })
        .and_then(|last| last.checked_add(1))
        .ok_or(KernelBodyError::InvalidBind("SSM scan state span overflow"))
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

/// Execute the bind-parameterized SSM scan body.
///
/// Prefill walks each state channel through the sequence in order.  Decode is
/// the length-one specialization of the same update: the incoming token is
/// the sole update and is written to the output state row.  The explicit
/// regime check is performed before buffer access so a mislabeled launch fails
/// closed instead of silently using the wrong state arithmetic.
pub fn ssm_scan(
    bind: &SsmScanBind,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), KernelBodyError> {
    bind.validate()?;
    let input_span = state_span(bind.length, bind.state_dim, bind.input_strides)?;
    let output_span = state_span(bind.length, bind.state_dim, bind.output_strides)?;
    checked_buffer("SSM scan input", input_span, input.len())?;
    checked_buffer("SSM scan output", output_span, output.len())?;

    let length = checked_usize(bind.length)?;
    let state_dim = checked_usize(bind.state_dim)?;
    let input_row_stride = checked_usize(bind.input_strides[0])?;
    let input_state_stride = checked_usize(bind.input_strides[1])?;
    let output_row_stride = checked_usize(bind.output_strides[0])?;
    let output_state_stride = checked_usize(bind.output_strides[1])?;

    match bind.regime {
        SsmScanRegime::Decode => {
            for state in 0..state_dim {
                let input_index = state * input_state_stride;
                let output_index = state * output_state_stride;
                output[output_index] = input[input_index];
            }
        }
        SsmScanRegime::Prefill => {
            for state in 0..state_dim {
                let mut carry = 0.0f32;
                for time in 0..length {
                    let input_index = time * input_row_stride + state * input_state_stride;
                    carry += input[input_index];
                    let output_index = time * output_row_stride + state * output_state_stride;
                    output[output_index] = carry;
                }
            }
        }
    }
    Ok(())
}

/// Dispatch the already-selected SSM scan body.
///
/// This is intentionally not a library-entry selector.  Plan recognition
/// remains outside the host body seam; unsupported state layouts and regime
/// mismatches fail in [`SsmScanBind::validate`].
pub fn dispatch_ssm_scan(
    kernel: SsmScanKernel,
    bind: &SsmScanBind,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), KernelBodyError> {
    match kernel {
        SsmScanKernel::Additive => ssm_scan(bind, input, output),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn additive_prefill_scan_matches_state_rows() {
        let input = [
            1.0f32, 2.0, 3.0, // t0
            0.5, -1.0, 2.0, // t1
            2.0, 4.0, -0.5, // t2
            -1.0, 0.25, 1.5, // t3
        ];
        let bind = SsmScanBind::prefill(4, 3, [12, 1, 1]);
        let mut output = [0.0f32; 12];

        dispatch_ssm_scan(SsmScanKernel::Additive, &bind, &input, &mut output)
            .expect("prefill SSM scan");

        assert_eq!(
            output,
            [
                1.0, 2.0, 3.0, // t0
                1.5, 1.0, 5.0, // t1
                3.5, 5.0, 4.5, // t2
                2.5, 5.25, 6.0, // t3
            ]
        );
    }

    #[test]
    fn decode_arm_is_the_length_one_state_update() {
        let input = [2.5f32, -0.75];
        let bind = SsmScanBind::decode(2, [2, 1, 1]);
        let mut output = [0.0f32; 2];

        dispatch_ssm_scan(SsmScanKernel::Additive, &bind, &input, &mut output)
            .expect("decode SSM scan");

        assert_eq!(output, input);
    }
}
