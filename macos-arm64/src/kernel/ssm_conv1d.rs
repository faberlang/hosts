//! Shape-generic Metal-library reference for causal SSM `conv1d`.
//!
//! The body owns one `[time, channel]` output row per logical invocation.  It
//! consumes a shared rank-1 causal kernel and keeps the state channel layout
//! explicit in the bind.  This module deliberately has no plan recognition;
//! M8-U2 owns mapping a resolved plan entry to this dispatch seam.

use super::library::KernelBodyError;

/// The only state layout currently proven by this body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SsmConv1dLayout {
    /// Canonical `[length, channels]` storage with channels as the inner axis.
    ChannelsLast,
    /// Sentinel for a caller that has not proved a servable state layout.
    Unsupported,
}

/// Immutable facts bound to one causal SSM convolution invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SsmConv1dBind {
    /// Number of time rows in the state input and output.
    pub length: u64,
    /// Number of independent state channels in each time row.
    pub channels: u64,
    /// Number of causal taps in the shared kernel.
    pub kernel_width: u64,
    /// Physical element strides for logical input `[time, channel]`.
    pub input_strides: [u64; 2],
    /// Physical element strides for logical output `[time, channel]`.
    pub output_strides: [u64; 2],
    /// Physical element stride for the rank-1 kernel.
    pub kernel_stride: u64,
    /// State channel layout selected by the executor.
    pub layout: SsmConv1dLayout,
    /// Backend-neutral dispatch grid carried by the launch record.
    pub grid: [u32; 3],
}

impl SsmConv1dBind {
    /// Construct the canonical synthetic SSM state-channel layout.
    #[must_use]
    pub fn channels_last(length: u64, channels: u64, kernel_width: u64, grid: [u32; 3]) -> Self {
        Self {
            length,
            channels,
            kernel_width,
            input_strides: [channels, 1],
            output_strides: [channels, 1],
            kernel_stride: 1,
            layout: SsmConv1dLayout::ChannelsLast,
            grid,
        }
    }

    /// Validate all shape, layout, stride, and grid facts before buffer access.
    pub fn validate(&self) -> Result<(), KernelBodyError> {
        if self.length == 0 || self.channels == 0 || self.kernel_width == 0 {
            return Err(KernelBodyError::InvalidBind(
                "SSM conv1d has a zero dimension",
            ));
        }
        if self.layout != SsmConv1dLayout::ChannelsLast {
            return Err(KernelBodyError::InvalidBind(
                "SSM conv1d state channel layout is not servable",
            ));
        }
        if self.grid.iter().any(|axis| *axis == 0) {
            return Err(KernelBodyError::InvalidBind(
                "SSM conv1d bind has a zero dispatch axis",
            ));
        }
        if self
            .input_strides
            .iter()
            .chain(self.output_strides.iter())
            .any(|stride| *stride == 0)
            || self.kernel_stride == 0
        {
            return Err(KernelBodyError::InvalidBind(
                "SSM conv1d bind has a zero stride",
            ));
        }
        let expected_state_strides = [self.channels, 1];
        if self.input_strides != expected_state_strides
            || self.output_strides != expected_state_strides
            || self.kernel_stride != 1
        {
            return Err(KernelBodyError::InvalidBind(
                "SSM conv1d state channel layout has non-canonical strides",
            ));
        }
        state_span(self.length, self.channels, self.input_strides)?;
        state_span(self.length, self.channels, self.output_strides)?;
        kernel_span(self.kernel_width, self.kernel_stride)?;
        Ok(())
    }
}

/// The selected body dispatch key.  Plan recognition remains outside this
/// module; this enum only prevents a caller from bypassing the body contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SsmConv1dKernel {
    /// Causal shared-kernel convolution over the state channels.
    Causal,
}

fn checked_usize(value: u64) -> Result<usize, KernelBodyError> {
    usize::try_from(value)
        .map_err(|_| KernelBodyError::InvalidBind("SSM conv1d index exceeds host usize"))
}

fn state_span(length: u64, channels: u64, strides: [u64; 2]) -> Result<u64, KernelBodyError> {
    length
        .checked_sub(1)
        .and_then(|last| last.checked_mul(strides[0]))
        .and_then(|last| {
            channels
                .checked_sub(1)
                .and_then(|last_channel| last_channel.checked_mul(strides[1]))
                .and_then(|last_channel| last.checked_add(last_channel))
        })
        .and_then(|last| last.checked_add(1))
        .ok_or(KernelBodyError::InvalidBind(
            "SSM conv1d state span overflow",
        ))
}

fn kernel_span(kernel_width: u64, kernel_stride: u64) -> Result<u64, KernelBodyError> {
    kernel_width
        .checked_sub(1)
        .and_then(|last| last.checked_mul(kernel_stride))
        .and_then(|last| last.checked_add(1))
        .ok_or(KernelBodyError::InvalidBind(
            "SSM conv1d kernel span overflow",
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

/// Execute the bind-parameterized causal convolution body.
///
/// The arithmetic is the M2 reference contract:
/// `out[t,c] = sum(k <= t) input[t-k,c] * kernel[k]`.  The invalid causal
/// prefix is omitted rather than read from a guessed state buffer, so the
/// same body proves both the initial prefix and steady-state rows.
pub fn ssm_conv1d(
    bind: &SsmConv1dBind,
    input: &[f32],
    kernel: &[f32],
    output: &mut [f32],
) -> Result<(), KernelBodyError> {
    bind.validate()?;
    let input_span = state_span(bind.length, bind.channels, bind.input_strides)?;
    let output_span = state_span(bind.length, bind.channels, bind.output_strides)?;
    let kernel_span = kernel_span(bind.kernel_width, bind.kernel_stride)?;
    checked_buffer("SSM conv1d input", input_span, input.len())?;
    checked_buffer("SSM conv1d kernel", kernel_span, kernel.len())?;
    checked_buffer("SSM conv1d output", output_span, output.len())?;

    let length = checked_usize(bind.length)?;
    let channels = checked_usize(bind.channels)?;
    let kernel_width = checked_usize(bind.kernel_width)?;
    let input_row_stride = checked_usize(bind.input_strides[0])?;
    let input_channel_stride = checked_usize(bind.input_strides[1])?;
    let output_row_stride = checked_usize(bind.output_strides[0])?;
    let output_channel_stride = checked_usize(bind.output_strides[1])?;
    let kernel_stride = checked_usize(bind.kernel_stride)?;

    for time in 0..length {
        for channel in 0..channels {
            let mut accumulator = 0.0f32;
            for offset in 0..kernel_width.min(time + 1) {
                let input_time = time - offset;
                let input_index = input_time * input_row_stride + channel * input_channel_stride;
                let kernel_index = offset * kernel_stride;
                accumulator += input[input_index] * kernel[kernel_index];
            }
            let output_index = time * output_row_stride + channel * output_channel_stride;
            output[output_index] = accumulator;
        }
    }
    Ok(())
}

/// Dispatch the already-selected SSM convolution body.
///
/// This is intentionally not a library-entry selector.  M8-U2 owns plan
/// recognition and must provide the selection only after it proves the state
/// layout; unknown layout families fail in [`SsmConv1dBind::validate`].
pub fn dispatch_ssm_conv1d(
    kernel: SsmConv1dKernel,
    bind: &SsmConv1dBind,
    input: &[f32],
    weights: &[f32],
    output: &mut [f32],
) -> Result<(), KernelBodyError> {
    match kernel {
        SsmConv1dKernel::Causal => ssm_conv1d(bind, input, weights, output),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_conv_matches_per_channel_reference_rows() {
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weights = [0.5f32, -1.0, 2.0];
        let bind = SsmConv1dBind::channels_last(4, 2, 3, [8, 1, 1]);
        let mut output = [0.0f32; 8];

        dispatch_ssm_conv1d(
            SsmConv1dKernel::Causal,
            &bind,
            &input,
            &weights,
            &mut output,
        )
        .expect("causal SSM convolution");

        assert_eq!(output, [0.5, 1.0, 0.5, 0.0, 1.5, 3.0, 4.5, 6.0]);
    }

    #[test]
    fn unservable_state_layout_fails_closed_before_buffer_access() {
        let mut bind = SsmConv1dBind::channels_last(2, 2, 2, [4, 1, 1]);
        bind.layout = SsmConv1dLayout::Unsupported;
        let mut output = [91.0f32; 4];

        let error = ssm_conv1d(&bind, &[], &[], &mut output)
            .expect_err("unsupported state layout must fail closed");
        assert!(matches!(
            error,
            KernelBodyError::InvalidBind(message) if message.contains("not servable")
        ));
        assert_eq!(output, [91.0; 4]);
    }
}
