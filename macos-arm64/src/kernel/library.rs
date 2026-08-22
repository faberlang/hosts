//! Shape-generic Metal-library body references (M2 library v0).
//!
//! The bodies in this module deliberately contain no model-specific constants
//! or concrete tensor extents.  A [`BindDescriptor`] supplies the logical
//! dimensions, physical strides, layout family, and dispatch grid.  Keeping
//! the body separate from its bind facts lets the executor select one library
//! entry for many shapes without changing the arithmetic or the operation
//! order used by the existing dense path.
//!
//! These functions are the host-side reference for the corresponding device
//! bodies.  They are also useful for focused numeric tests: a contiguous and
//! a strided bind must produce the same logical bytes.  Device materializers
//! are responsible for turning the descriptor into backend binding/uniform
//! arguments; this module never guesses a shape from a buffer length.

/// Version of the target-neutral library-v0 bind descriptor consumed by these
/// bodies.  This is an additive executor-plan extension, not a replacement for
/// the existing device descriptor schema.
pub const LIBRARY_BIND_ABI_VERSION: u16 = 1;

/// A body layout family.  `Strided` permits arbitrary positive strides;
/// `RowMajor` additionally checks the canonical contiguous stride chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindLayout {
    /// A one-dimensional logical buffer.
    Flat,
    /// Contiguous row-major storage.
    RowMajor,
    /// A non-contiguous view described by [`BindDescriptor::strides`].
    Strided,
    /// RoPE with consecutive pairs (`[x0, x1]`, `[x2, x3]`, ...).
    RopeConsecutivePair { rotated_width: u64 },
    /// RoPE with rotate-half pairing (the Qwen/NeoX convention).
    RopeRotateHalf { rotated_width: u64 },
}

/// The immutable facts bound to one library-body invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindDescriptor {
    /// Logical extents in outer-to-inner order.
    pub dims: Vec<u64>,
    /// Physical element strides in the same order as [`Self::dims`].
    pub strides: Vec<u64>,
    /// The interpretation of the dimensions and strides.
    pub layout: BindLayout,
    /// Backend-neutral dispatch grid.  The body itself does not derive or
    /// modify this value; it is carried for the executor's launch record.
    pub grid: [u32; 3],
}

impl BindDescriptor {
    /// Construct a contiguous row-major bind with a caller-owned grid.
    #[must_use]
    pub fn row_major(dims: impl Into<Vec<u64>>, grid: [u32; 3]) -> Self {
        let dims = dims.into();
        let mut strides = vec![1u64; dims.len()];
        for axis in (0..dims.len()).rev().skip(1) {
            strides[axis] = strides[axis + 1].saturating_mul(dims[axis + 1]);
        }
        Self {
            dims,
            strides,
            layout: BindLayout::RowMajor,
            grid,
        }
    }

    /// Construct an arbitrary positive-stride bind.
    #[must_use]
    pub fn strided(
        dims: impl Into<Vec<u64>>,
        strides: impl Into<Vec<u64>>,
        grid: [u32; 3],
    ) -> Self {
        Self {
            dims: dims.into(),
            strides: strides.into(),
            layout: BindLayout::Strided,
            grid,
        }
    }

    /// Number of logical elements covered by this bind.
    #[must_use]
    pub fn element_count(&self) -> Option<u64> {
        self.dims
            .iter()
            .try_fold(1u64, |count, dim| count.checked_mul(*dim))
    }

    /// Number of physical elements addressed by this bind, including holes
    /// introduced by strides.
    #[must_use]
    pub fn physical_span(&self) -> Option<u64> {
        if self.dims.is_empty() || self.dims.len() != self.strides.len() {
            return None;
        }
        self.dims
            .iter()
            .zip(&self.strides)
            .try_fold(1u64, |span, (dim, stride)| {
                dim.checked_sub(1)
                    .and_then(|last| last.checked_mul(*stride))
                    .and_then(|last| span.checked_add(last))
            })
    }

    /// Validate the descriptor before a body touches any buffer.
    pub fn validate(&self) -> Result<(), KernelBodyError> {
        if self.dims.is_empty() {
            return Err(KernelBodyError::InvalidBind(
                "bind descriptor has no dimensions",
            ));
        }
        if self.dims.len() != self.strides.len() {
            return Err(KernelBodyError::InvalidBind(
                "bind dimensions and strides have different ranks",
            ));
        }
        if self.dims.iter().any(|dim| *dim == 0) {
            return Err(KernelBodyError::InvalidBind(
                "bind descriptor has a zero dimension",
            ));
        }
        if self.strides.iter().any(|stride| *stride == 0) {
            return Err(KernelBodyError::InvalidBind(
                "bind descriptor has a zero stride",
            ));
        }
        if self.grid.iter().any(|axis| *axis == 0) {
            return Err(KernelBodyError::InvalidBind(
                "bind descriptor has a zero dispatch axis",
            ));
        }
        if matches!(self.layout, BindLayout::RowMajor)
            && self.strides != row_major_strides(&self.dims)
        {
            return Err(KernelBodyError::InvalidBind(
                "row-major bind has non-canonical strides",
            ));
        }
        if let BindLayout::RopeConsecutivePair { rotated_width }
        | BindLayout::RopeRotateHalf { rotated_width } = self.layout
        {
            let width = *self.dims.last().unwrap_or(&0);
            if rotated_width == 0 || rotated_width % 2 != 0 || rotated_width > width {
                return Err(KernelBodyError::InvalidBind(
                    "rope rotated width must be positive, even, and within the row width",
                ));
            }
        }
        if self.element_count().is_none() || self.physical_span().is_none() {
            return Err(KernelBodyError::InvalidBind(
                "bind descriptor dimensions overflow the element index",
            ));
        }
        Ok(())
    }
}

fn row_major_strides(dims: &[u64]) -> Vec<u64> {
    let mut strides = vec![1u64; dims.len()];
    for axis in (0..dims.len()).rev().skip(1) {
        strides[axis] = strides[axis + 1].saturating_mul(dims[axis + 1]);
    }
    strides
}

/// Quantized packed-weight format supported by the decode GEMV library.
///
/// The byte layout and block arithmetic mirror the R-PACK-02 Metal bodies.
/// A GEMV dequantizes one block at the load that consumes it; it never builds
/// a whole f32 weight matrix.  `Q5_K` is retained here because it is an
/// admitted R-PACK-02 format even though the two dense reference rungs use
/// `Q4_K`, `Q5_0`, `Q6_K`, and `Q8_0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizedFormat {
    /// GGML Q4_K: 256 elements in a 144-byte superblock.
    Q4K,
    /// GGML Q5_0: 32 elements in a 22-byte block.
    Q5_0,
    /// GGML Q5_K: 256 elements in a 176-byte superblock.
    Q5K,
    /// GGML Q6_K: 256 elements in a 210-byte superblock.
    Q6K,
    /// GGML Q8_0: 32 elements in a 34-byte block.
    Q8_0,
}

impl QuantizedFormat {
    /// Logical elements covered by one packed block.
    #[must_use]
    pub const fn block_elements(self) -> u64 {
        match self {
            Self::Q4K | Self::Q5K | Self::Q6K => 256,
            Self::Q5_0 | Self::Q8_0 => 32,
        }
    }

    /// Packed bytes occupied by one block.
    #[must_use]
    pub const fn block_bytes(self) -> u64 {
        match self {
            Self::Q4K => 144,
            Self::Q5_0 => 22,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8_0 => 34,
        }
    }

    /// Canonical GGML spelling used in receipts and diagnostics.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Q4K => "Q4_K",
            Self::Q5_0 => "Q5_0",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8_0 => "Q8_0",
        }
    }

    /// Resolve a GEMV format from a packed-layout GGML type id.
    ///
    /// Unknown ids fail closed — never a guessed block geometry. F32/F16/BF16
    /// are not GEMV packed formats on this path.
    #[must_use]
    pub const fn from_ggml_type_id(id: u32) -> Option<Self> {
        match id {
            6 => Some(Self::Q5_0),
            8 => Some(Self::Q8_0),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            _ => None,
        }
    }
}

/// Bind facts for one sequence-length-one quantized projection.
///
/// Packed columns use the R-PACK-02 layout: column `n` owns
/// `ceil(k / block_elements) * block_bytes` bytes, and each block is decoded
/// at use.  Strides are explicit so a plan can bind views without baking in
/// model-specific dimensions or contiguous storage assumptions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizedGemvBind {
    /// Contracted activation width.
    pub k: u64,
    /// Projection output width.
    pub n: u64,
    /// Physical stride between activation elements.
    pub input_stride: u64,
    /// Physical stride between output elements.
    pub output_stride: u64,
    /// Physical byte stride between packed weight columns.
    pub packed_column_stride_bytes: u64,
    /// Packed weight format.
    pub format: QuantizedFormat,
    /// Backend-neutral dispatch grid carried into the plan record.
    pub grid: [u32; 3],
}

impl QuantizedGemvBind {
    /// Construct a contiguous M=1 decode bind.
    #[must_use]
    pub fn decode(k: u64, n: u64, format: QuantizedFormat, grid: [u32; 3]) -> Self {
        let blocks = k.div_ceil(format.block_elements());
        Self {
            k,
            n,
            input_stride: 1,
            output_stride: 1,
            packed_column_stride_bytes: blocks.saturating_mul(format.block_bytes()),
            format,
            grid,
        }
    }

    /// Construct a decode bind over strided activation/output views.
    #[must_use]
    pub fn strided(
        k: u64,
        n: u64,
        input_stride: u64,
        output_stride: u64,
        packed_column_stride_bytes: u64,
        format: QuantizedFormat,
        grid: [u32; 3],
    ) -> Self {
        Self {
            k,
            n,
            input_stride,
            output_stride,
            packed_column_stride_bytes,
            format,
            grid,
        }
    }

    /// Validate all bind facts before a body reads or writes a buffer.
    pub fn validate(&self) -> Result<(), KernelBodyError> {
        if self.k == 0 || self.n == 0 {
            return Err(KernelBodyError::InvalidBind(
                "quantized GEMV has a zero dimension",
            ));
        }
        if self.input_stride == 0 || self.output_stride == 0 || self.packed_column_stride_bytes == 0
        {
            return Err(KernelBodyError::InvalidBind(
                "quantized GEMV has a zero stride",
            ));
        }
        if self.grid.iter().any(|axis| *axis == 0) {
            return Err(KernelBodyError::InvalidBind(
                "quantized GEMV has a zero dispatch axis",
            ));
        }
        let block_elements = self.format.block_elements();
        if !self.k.is_multiple_of(block_elements) {
            return Err(KernelBodyError::InvalidBind(
                "quantized GEMV K is not block aligned",
            ));
        }
        let blocks_per_column = self.k / block_elements;
        let minimum_column_bytes = blocks_per_column
            .checked_mul(self.format.block_bytes())
            .ok_or(KernelBodyError::InvalidBind(
                "quantized GEMV packed column span overflow",
            ))?;
        if self.packed_column_stride_bytes < minimum_column_bytes {
            return Err(KernelBodyError::InvalidBind(
                "quantized GEMV packed column stride is too small",
            ));
        }
        self.k
            .checked_sub(1)
            .and_then(|last| last.checked_mul(self.input_stride))
            .and_then(|last| last.checked_add(1))
            .ok_or(KernelBodyError::InvalidBind(
                "quantized GEMV input span overflow",
            ))?;
        self.n
            .checked_sub(1)
            .and_then(|last| last.checked_mul(self.output_stride))
            .and_then(|last| last.checked_add(1))
            .ok_or(KernelBodyError::InvalidBind(
                "quantized GEMV output span overflow",
            ))?;
        self.n
            .checked_sub(1)
            .and_then(|last| last.checked_mul(self.packed_column_stride_bytes))
            .and_then(|last| last.checked_add(minimum_column_bytes))
            .ok_or(KernelBodyError::InvalidBind(
                "quantized GEMV weight span overflow",
            ))?;
        Ok(())
    }

    fn input_span(self) -> u64 {
        1 + (self.k - 1) * self.input_stride
    }

    fn output_span(self) -> u64 {
        1 + (self.n - 1) * self.output_stride
    }

    fn weight_span(self) -> u64 {
        (self.n - 1) * self.packed_column_stride_bytes
            + (self.k / self.format.block_elements()) * self.format.block_bytes()
    }
}

/// Layout family for one grouped Q/K/V projection body.
///
/// The grouped layout is the only layout the body can materialize without
/// inventing a permutation or a second launch.  Unknown and unsupported
/// layout families remain explicit so selection fails closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QkvProjectionLayout {
    /// Q is `[kv_group, q_head, row, d]`; K/V are `[kv_group, row, d]`.
    Grouped,
    /// Sentinel used by callers that have not proved a servable layout.
    Unsupported,
}

/// One weight source consumed by the shape-generic Q/K/V body.
///
/// Dense f32 is retained for the GI2-2 numeric class.  Packed sources use the
/// same R-PACK-02 bind and dequantization path as decode GEMV; the body never
/// infers a packed layout from a byte length.
#[derive(Debug, Clone, Copy)]
pub enum QkvProjectionWeight<'a> {
    /// Column-major logical `[hidden, output]` f32 weights, with each output
    /// column stored contiguously (`column * hidden + k`).
    Dense(&'a [f32]),
    /// One packed column stream per output column.
    Quantized {
        /// Typed shape/format facts for the packed source.
        bind: QuantizedGemvBind,
        /// R-PACK-02 bytes addressed by `bind`.
        packed: &'a [u8],
    },
}

/// Bind facts for the single-launch grouped Q/K/V projection body.
///
/// All shape and physical-stride facts are carried here.  In particular, a
/// caller cannot turn a Q/K/V selection into a body by assuming contiguous
/// output or by deriving GQA dimensions from a supplied slice length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QkvProjectionBind {
    /// Number of activation rows in this invocation.
    pub rows: u64,
    /// Activation width / contracted weight dimension.
    pub hidden: u64,
    /// Number of KV groups.
    pub kv_heads: u64,
    /// Query heads sharing one KV group.
    pub q_per_kv: u64,
    /// Elements in one attention head.
    pub head_dim: u64,
    /// Physical row stride of the activation view.
    pub input_row_stride: u64,
    /// Physical element stride within one activation row.
    pub input_element_stride: u64,
    /// Physical strides for logical Q `[group, q_head, row, d]`.
    pub q_output_strides: [u64; 4],
    /// Physical strides for logical K/V `[group, row, d]`.
    pub kv_output_strides: [u64; 3],
    /// Shape family selected by the executor.
    pub layout: QkvProjectionLayout,
    /// Whether Q/K use the rotate-half (NeoX/Qwen) pairing.
    pub rotate_half: bool,
    /// Backend-neutral launch grid carried with the bind.
    pub grid: [u32; 3],
}

impl QkvProjectionBind {
    /// Construct the canonical grouped contiguous output layout.
    #[must_use]
    pub fn grouped(
        rows: u64,
        hidden: u64,
        kv_heads: u64,
        q_per_kv: u64,
        head_dim: u64,
        grid: [u32; 3],
    ) -> Self {
        let q_output_strides = [
            q_per_kv.saturating_mul(rows).saturating_mul(head_dim),
            rows.saturating_mul(head_dim),
            head_dim,
            1,
        ];
        let kv_output_strides = [rows.saturating_mul(head_dim), head_dim, 1];
        Self {
            rows,
            hidden,
            kv_heads,
            q_per_kv,
            head_dim,
            input_row_stride: hidden,
            input_element_stride: 1,
            q_output_strides,
            kv_output_strides,
            layout: QkvProjectionLayout::Grouped,
            rotate_half: false,
            grid,
        }
    }

    /// Validate all static shape, stride, and grid facts before any buffer is
    /// accessed by the body.
    pub fn validate(&self) -> Result<(), KernelBodyError> {
        if self.rows == 0
            || self.hidden == 0
            || self.kv_heads == 0
            || self.q_per_kv == 0
            || self.head_dim == 0
        {
            return Err(KernelBodyError::InvalidBind(
                "QKV projection bind has a zero dimension",
            ));
        }
        if self.layout != QkvProjectionLayout::Grouped {
            return Err(KernelBodyError::InvalidBind(
                "QKV projection layout is not servable",
            ));
        }
        if self.grid.iter().any(|axis| *axis == 0) {
            return Err(KernelBodyError::InvalidBind(
                "QKV projection bind has a zero dispatch axis",
            ));
        }
        if self
            .q_output_strides
            .iter()
            .chain(self.kv_output_strides.iter())
            .any(|stride| *stride == 0)
        {
            return Err(KernelBodyError::InvalidBind(
                "QKV projection bind has a zero output stride",
            ));
        }
        self.kv_heads
            .checked_mul(self.q_per_kv)
            .and_then(|heads| heads.checked_mul(self.head_dim))
            .filter(|width| *width == self.hidden)
            .ok_or(KernelBodyError::ShapeMismatch(
                "QKV hidden width does not equal kv_heads * q_per_kv * head_dim",
            ))?;
        self.rows
            .checked_sub(1)
            .and_then(|last| last.checked_mul(self.input_row_stride))
            .and_then(|last| {
                self.hidden
                    .checked_sub(1)
                    .and_then(|width| width.checked_mul(self.input_element_stride))
                    .and_then(|width| last.checked_add(width))
            })
            .and_then(|last| last.checked_add(1))
            .ok_or(KernelBodyError::InvalidBind("QKV activation span overflow"))?;
        attention_span(
            &[self.kv_heads, self.q_per_kv, self.rows, self.head_dim],
            &self.q_output_strides,
        )?;
        attention_span(
            &[self.kv_heads, self.rows, self.head_dim],
            &self.kv_output_strides,
        )?;
        if self.rotate_half && !self.head_dim.is_multiple_of(2) {
            return Err(KernelBodyError::ShapeMismatch(
                "QKV rotate-half head width must be even",
            ));
        }
        Ok(())
    }

    fn q_width(self) -> usize {
        (self.kv_heads * self.q_per_kv * self.head_dim) as usize
    }

    fn kv_width(self) -> usize {
        (self.kv_heads * self.head_dim) as usize
    }
}

fn qkv_weight_values(
    bind: &QkvProjectionBind,
    weight: QkvProjectionWeight<'_>,
    activation: &[f32],
    row_base: usize,
    output_width: usize,
) -> Result<Vec<f32>, KernelBodyError> {
    match weight {
        QkvProjectionWeight::Dense(values) => {
            let required = checked_usize(bind.hidden.checked_mul(output_width as u64).ok_or(
                KernelBodyError::InvalidBind("QKV dense weight span overflow"),
            )?)?;
            if values.len() < required {
                return Err(KernelBodyError::BufferTooShort {
                    buffer: "QKV dense weight",
                    required: required as u64,
                    actual: values.len(),
                });
            }
            let hidden = checked_usize(bind.hidden)?;
            let input_stride = checked_usize(bind.input_element_stride)?;
            let mut output = vec![0.0f32; output_width];
            for column in 0..output_width {
                let mut sum = 0.0f32;
                for element in 0..hidden {
                    sum += activation[row_base + element * input_stride]
                        * values[column * hidden + element];
                }
                output[column] = sum;
            }
            Ok(output)
        }
        QkvProjectionWeight::Quantized {
            bind: weight_bind,
            packed,
        } => {
            if weight_bind.k != bind.hidden || weight_bind.n != output_width as u64 {
                return Err(KernelBodyError::ShapeMismatch(
                    "QKV packed weight dimensions do not match the projection bind",
                ));
            }
            let mut row_bind = weight_bind;
            row_bind.input_stride = bind.input_element_stride;
            row_bind.output_stride = 1;
            row_bind.grid = bind.grid;
            let mut output = vec![0.0f32; output_width];
            dispatch_gemv(
                GemvKernel::Quantized,
                &row_bind,
                &activation[row_base..],
                packed,
                &mut output,
            )?;
            Ok(output)
        }
    }
}

fn qkv_add_bias(
    values: &mut [f32],
    bias: Option<&[f32]>,
    name: &'static str,
) -> Result<(), KernelBodyError> {
    let Some(bias) = bias else {
        return Ok(());
    };
    if bias.len() < values.len() {
        return Err(KernelBodyError::BufferTooShort {
            buffer: name,
            required: values.len() as u64,
            actual: bias.len(),
        });
    }
    for (value, offset) in values.iter_mut().zip(bias) {
        *value += *offset;
    }
    Ok(())
}

fn qkv_rotate(values: &mut [f32], head_dim: usize, rotate_half: bool, cos: &[f32], sin: &[f32]) {
    let half = head_dim / 2;
    for head in values.chunks_exact_mut(head_dim) {
        let source = head.to_vec();
        for pair in 0..half {
            let (left, right) = if rotate_half {
                (pair, pair + half)
            } else {
                (pair * 2, pair * 2 + 1)
            };
            head[left] = source[left] * cos[pair] - source[right] * sin[pair];
            head[right] = source[left] * sin[pair] + source[right] * cos[pair];
        }
    }
}

/// One bind-parameterized Q/K/V projection body.
///
/// Q and K receive the optional bias before the optional RoPE rotation; V
/// receives its bias but no rotation.  The three matrices are reduced in one
/// body and written directly to their grouped output views.  A mismatch in
/// the optional bias or RoPE bind is rejected before touching output memory.
pub fn qkv_projection(
    bind: &QkvProjectionBind,
    activation: &[f32],
    q_weight: QkvProjectionWeight<'_>,
    k_weight: QkvProjectionWeight<'_>,
    v_weight: QkvProjectionWeight<'_>,
    q_bias: Option<&[f32]>,
    k_bias: Option<&[f32]>,
    v_bias: Option<&[f32]>,
    cos: Option<&[f32]>,
    sin: Option<&[f32]>,
    q_output: &mut [f32],
    k_output: &mut [f32],
    v_output: &mut [f32],
) -> Result<(), KernelBodyError> {
    bind.validate()?;
    let input_span = checked_usize(
        (bind.rows - 1)
            .checked_mul(bind.input_row_stride)
            .and_then(|last| {
                (bind.hidden - 1)
                    .checked_mul(bind.input_element_stride)
                    .and_then(|width| last.checked_add(width))
            })
            .and_then(|last| last.checked_add(1))
            .ok_or(KernelBodyError::InvalidBind("QKV activation span overflow"))?,
    )?;
    if activation.len() < input_span {
        return Err(KernelBodyError::BufferTooShort {
            buffer: "QKV activation",
            required: input_span as u64,
            actual: activation.len(),
        });
    }
    let q_span = checked_usize(attention_span(
        &[bind.kv_heads, bind.q_per_kv, bind.rows, bind.head_dim],
        &bind.q_output_strides,
    )?)?;
    let kv_span = checked_usize(attention_span(
        &[bind.kv_heads, bind.rows, bind.head_dim],
        &bind.kv_output_strides,
    )?)?;
    if q_output.len() < q_span {
        return Err(KernelBodyError::BufferTooShort {
            buffer: "QKV Q output",
            required: q_span as u64,
            actual: q_output.len(),
        });
    }
    if k_output.len() < kv_span {
        return Err(KernelBodyError::BufferTooShort {
            buffer: "QKV K output",
            required: kv_span as u64,
            actual: k_output.len(),
        });
    }
    if v_output.len() < kv_span {
        return Err(KernelBodyError::BufferTooShort {
            buffer: "QKV V output",
            required: kv_span as u64,
            actual: v_output.len(),
        });
    }
    let q_width = bind.q_width();
    let kv_width = bind.kv_width();
    let bias_present = [q_bias.is_some(), k_bias.is_some(), v_bias.is_some()];
    if bias_present.iter().any(|present| *present) && !bias_present.iter().all(|present| *present) {
        return Err(KernelBodyError::InvalidBind(
            "QKV bias bind must provide Q, K, and V together",
        ));
    }
    if cos.is_some() != sin.is_some() {
        return Err(KernelBodyError::InvalidBind(
            "QKV RoPE bind must provide cosine and sine tables together",
        ));
    }
    let table_width = checked_usize(bind.head_dim / 2)?;
    if let (Some(cos), Some(sin)) = (cos, sin) {
        let table_len = checked_usize(bind.rows * bind.head_dim / 2)?;
        if cos.len() < table_len || sin.len() < table_len {
            return Err(KernelBodyError::BufferTooShort {
                buffer: "QKV RoPE table",
                required: table_len as u64,
                actual: cos.len().min(sin.len()),
            });
        }
    }
    for row in 0..checked_usize(bind.rows)? {
        let row_base = checked_usize(
            (row as u64)
                .checked_mul(bind.input_row_stride)
                .ok_or(KernelBodyError::InvalidBind("QKV input row index overflow"))?,
        )?;
        let mut q = qkv_weight_values(bind, q_weight, activation, row_base, q_width)?;
        let mut k = qkv_weight_values(bind, k_weight, activation, row_base, kv_width)?;
        let mut v = qkv_weight_values(bind, v_weight, activation, row_base, kv_width)?;
        qkv_add_bias(&mut q, q_bias, "QKV Q bias")?;
        qkv_add_bias(&mut k, k_bias, "QKV K bias")?;
        qkv_add_bias(&mut v, v_bias, "QKV V bias")?;
        if let (Some(cos), Some(sin)) = (cos, sin) {
            let table_base = row * table_width;
            qkv_rotate(
                &mut q,
                checked_usize(bind.head_dim)?,
                bind.rotate_half,
                &cos[table_base..table_base + table_width],
                &sin[table_base..table_base + table_width],
            );
            qkv_rotate(
                &mut k,
                checked_usize(bind.head_dim)?,
                bind.rotate_half,
                &cos[table_base..table_base + table_width],
                &sin[table_base..table_base + table_width],
            );
        }
        let row_u64 = row as u64;
        for group in 0..checked_usize(bind.kv_heads)? {
            for query_head in 0..checked_usize(bind.q_per_kv)? {
                for dimension in 0..checked_usize(bind.head_dim)? {
                    let column = (group * checked_usize(bind.q_per_kv)? + query_head)
                        * checked_usize(bind.head_dim)?
                        + dimension;
                    let offset = checked_usize(
                        (group as u64)
                            .checked_mul(bind.q_output_strides[0])
                            .and_then(|value| {
                                (query_head as u64)
                                    .checked_mul(bind.q_output_strides[1])
                                    .and_then(|head| value.checked_add(head))
                            })
                            .and_then(|value| {
                                row_u64
                                    .checked_mul(bind.q_output_strides[2])
                                    .and_then(|row| value.checked_add(row))
                            })
                            .and_then(|value| {
                                (dimension as u64)
                                    .checked_mul(bind.q_output_strides[3])
                                    .and_then(|dimension| value.checked_add(dimension))
                            })
                            .ok_or(KernelBodyError::InvalidBind("QKV Q output index overflow"))?,
                    )?;
                    q_output[offset] = q[column];
                }
            }
            for dimension in 0..checked_usize(bind.head_dim)? {
                let column = group * checked_usize(bind.head_dim)? + dimension;
                let offset = checked_usize(
                    (group as u64)
                        .checked_mul(bind.kv_output_strides[0])
                        .and_then(|value| {
                            row_u64
                                .checked_mul(bind.kv_output_strides[1])
                                .and_then(|row| value.checked_add(row))
                        })
                        .and_then(|value| {
                            (dimension as u64)
                                .checked_mul(bind.kv_output_strides[2])
                                .and_then(|dimension| value.checked_add(dimension))
                        })
                        .ok_or(KernelBodyError::InvalidBind("QKV KV output index overflow"))?,
                )?;
                k_output[offset] = k[column];
                v_output[offset] = v[column];
            }
        }
    }
    Ok(())
}

/// Select the single QKV body from executor-plan facts.
///
/// Both prefill (`decode_gemv = 0`) and scalar decode (`decode_gemv = 1`)
/// select the same bind-parameterized body.  Unknown entries and unservable
/// layouts fail closed rather than silently falling back to three launches.
pub fn select_qkv_projection(
    library_entry: Option<&str>,
    decode_gemv: u32,
    layout: QkvProjectionLayout,
) -> Result<Option<LibraryKernel>, KernelBodyError> {
    if layout != QkvProjectionLayout::Grouped {
        return Err(KernelBodyError::InvalidBind(
            "QKV projection layout is not servable",
        ));
    }
    match (library_entry, decode_gemv) {
        (Some("QkvProjection"), 0 | 1) => Ok(Some(LibraryKernel::QkvProjection)),
        (Some("QkvProjection"), _) => Err(KernelBodyError::InvalidBind(
            "QKV projection decode uniform is not 0 or 1",
        )),
        (None, 0 | 1) => Ok(None),
        _ => Err(KernelBodyError::InvalidBind(
            "QKV projection selection disagrees with library_entry",
        )),
    }
}

/// Dispatch the selected single-body QKV projection.
pub fn dispatch_qkv_projection(
    kernel: LibraryKernel,
    bind: &QkvProjectionBind,
    activation: &[f32],
    weights: [QkvProjectionWeight<'_>; 3],
    biases: [Option<&[f32]>; 3],
    rope: Option<(&[f32], &[f32])>,
    outputs: [&mut [f32]; 3],
) -> Result<(), KernelBodyError> {
    match kernel {
        LibraryKernel::QkvProjection => {
            let [q_output, k_output, v_output] = outputs;
            qkv_projection(
                bind,
                activation,
                weights[0],
                weights[1],
                weights[2],
                biases[0],
                biases[1],
                biases[2],
                rope.map(|(cos, _)| cos),
                rope.map(|(_, sin)| sin),
                q_output,
                k_output,
                v_output,
            )
        }
        LibraryKernel::CausalAttention => Err(KernelBodyError::InvalidBind(
            "QKV dispatch received CausalAttention",
        )),
    }
}

/// Layout family for the fused attention library body.
///
/// `Grouped` is the canonical `[kv_group, q_head, query, head_dim]` query and
/// output layout with `[kv_group, sequence, head_dim]` key/value layout.
/// `Strided` keeps those logical axes while allowing a caller to bind views
/// with arbitrary positive strides.  The arithmetic is shared by both
/// layouts; only the bind facts change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalAttentionLayout {
    /// Canonical group-major, sequence-major storage.
    Grouped,
    /// Explicit strides describe each input and output view.
    Strided,
}

/// Bind facts for one shape-generic `CausalAttention` library invocation.
///
/// One invocation owns one KV-group batch.  Its `q_per_kv` query heads use
/// separate softmax accumulators, while the body is independent of model
/// dimensions.  Query tensors are logically
/// `[kv_batch, q_per_kv, query_seq, head_dim]`; key and value tensors are
/// `[kv_batch, seq_block, head_dim]`.  The causal window is the final
/// `query_seq` rows of the KV block, which covers both prompt and one-token
/// decode calls without a model-specific position argument.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CausalAttentionBind {
    /// Head width used by the dot product and `1/sqrt(head_dim)` scale.
    pub head_dim: u64,
    /// Number of key/value sequence rows in this invocation.
    pub seq_block: u64,
    /// Number of query heads sharing each KV head.
    pub q_per_kv: u64,
    /// Number of KV groups in the bound batch.
    pub kv_batch: u64,
    /// Number of query rows.  They address the final rows of `seq_block`.
    pub query_seq: u64,
    /// Query strides for logical `[group, q_head, query, dim]` axes.
    pub q_strides: [u64; 4],
    /// Key strides for logical `[group, sequence, dim]` axes.
    pub k_strides: [u64; 3],
    /// Value strides for logical `[group, sequence, dim]` axes.
    pub v_strides: [u64; 3],
    /// Output strides for logical `[group, q_head, query, dim]` axes.
    pub output_strides: [u64; 4],
    /// Logical storage family for this bind.
    pub layout: CausalAttentionLayout,
    /// Backend-neutral dispatch grid carried into the plan record.
    pub grid: [u32; 3],
}

impl CausalAttentionBind {
    /// Construct canonical grouped storage for one prompt or decode block.
    #[must_use]
    pub fn grouped(
        head_dim: u64,
        seq_block: u64,
        q_per_kv: u64,
        kv_batch: u64,
        query_seq: u64,
        grid: [u32; 3],
    ) -> Self {
        let q_row = q_per_kv.saturating_mul(query_seq).saturating_mul(head_dim);
        let q_head = query_seq.saturating_mul(head_dim);
        let k_row = seq_block.saturating_mul(head_dim);
        let q_strides = [q_row, q_head, head_dim, 1];
        let k_strides = [k_row, head_dim, 1];
        Self {
            head_dim,
            seq_block,
            q_per_kv,
            kv_batch,
            query_seq,
            q_strides,
            k_strides,
            v_strides: k_strides,
            output_strides: q_strides,
            layout: CausalAttentionLayout::Grouped,
            grid,
        }
    }

    /// Construct a strided attention bind while retaining the same logical
    /// axes as [`Self::grouped`].
    #[must_use]
    pub fn strided(
        head_dim: u64,
        seq_block: u64,
        q_per_kv: u64,
        kv_batch: u64,
        query_seq: u64,
        q_strides: [u64; 4],
        k_strides: [u64; 3],
        v_strides: [u64; 3],
        output_strides: [u64; 4],
        grid: [u32; 3],
    ) -> Self {
        Self {
            head_dim,
            seq_block,
            q_per_kv,
            kv_batch,
            query_seq,
            q_strides,
            k_strides,
            v_strides,
            output_strides,
            layout: CausalAttentionLayout::Strided,
            grid,
        }
    }

    /// Validate the attention shape and all bound physical views before any
    /// input or output buffer is touched.
    pub fn validate(&self) -> Result<(), KernelBodyError> {
        if self.head_dim == 0
            || self.seq_block == 0
            || self.q_per_kv == 0
            || self.kv_batch == 0
            || self.query_seq == 0
        {
            return Err(KernelBodyError::InvalidBind(
                "causal attention bind has a zero dimension",
            ));
        }
        if self.query_seq > self.seq_block {
            return Err(KernelBodyError::InvalidBind(
                "causal attention query rows exceed the sequence block",
            ));
        }
        if self.grid.iter().any(|axis| *axis == 0) {
            return Err(KernelBodyError::InvalidBind(
                "causal attention bind has a zero dispatch axis",
            ));
        }
        if self
            .q_strides
            .iter()
            .chain(self.k_strides.iter())
            .chain(self.v_strides.iter())
            .chain(self.output_strides.iter())
            .any(|stride| *stride == 0)
        {
            return Err(KernelBodyError::InvalidBind(
                "causal attention bind has a zero stride",
            ));
        }
        let q_head =
            self.query_seq
                .checked_mul(self.head_dim)
                .ok_or(KernelBodyError::InvalidBind(
                    "causal attention query stride overflow",
                ))?;
        let q_group = self
            .q_per_kv
            .checked_mul(q_head)
            .ok_or(KernelBodyError::InvalidBind(
                "causal attention query stride overflow",
            ))?;
        let k_group =
            self.seq_block
                .checked_mul(self.head_dim)
                .ok_or(KernelBodyError::InvalidBind(
                    "causal attention KV stride overflow",
                ))?;
        let expected_q = [q_group, q_head, self.head_dim, 1];
        let expected_k = [k_group, self.head_dim, 1];
        if matches!(self.layout, CausalAttentionLayout::Grouped)
            && (self.q_strides != expected_q
                || self.k_strides != expected_k
                || self.v_strides != self.k_strides
                || self.output_strides != self.q_strides)
        {
            return Err(KernelBodyError::InvalidBind(
                "grouped causal attention bind has non-canonical strides",
            ));
        }
        attention_span(
            &[self.kv_batch, self.q_per_kv, self.query_seq, self.head_dim],
            &self.q_strides,
        )?;
        attention_span(
            &[self.kv_batch, self.seq_block, self.head_dim],
            &self.k_strides,
        )?;
        attention_span(
            &[self.kv_batch, self.seq_block, self.head_dim],
            &self.v_strides,
        )?;
        attention_span(
            &[self.kv_batch, self.q_per_kv, self.query_seq, self.head_dim],
            &self.output_strides,
        )?;
        Ok(())
    }
}

fn attention_span(dims: &[u64], strides: &[u64]) -> Result<u64, KernelBodyError> {
    if dims.len() != strides.len() {
        return Err(KernelBodyError::InvalidBind(
            "causal attention dimensions and strides have different ranks",
        ));
    }
    dims.iter()
        .zip(strides)
        .try_fold(1u64, |span, (dim, stride)| {
            dim.checked_sub(1)
                .and_then(|last| last.checked_mul(*stride))
                .and_then(|last| span.checked_add(last))
        })
        .ok_or(KernelBodyError::InvalidBind(
            "causal attention physical span overflow",
        ))
}

/// Errors returned before a body performs a read or write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelBodyError {
    /// The bind facts are structurally invalid.
    InvalidBind(&'static str),
    /// A buffer is shorter than the descriptor's addressed span.
    BufferTooShort {
        /// Logical body input/output name.
        buffer: &'static str,
        /// Required physical element count.
        required: u64,
        /// Supplied physical element count.
        actual: usize,
    },
    /// A body argument has a shape or length incompatible with the bind.
    ShapeMismatch(&'static str),
    /// RMSNorm epsilon is not an accepted finite positive f32.
    InvalidEpsilon,
    /// RoPE tables do not match the bound row and pair dimensions.
    InvalidRopeTable,
    /// A packed weight block is shorter than its format requires.
    PackedBlockTooShort {
        /// Canonical packed format spelling.
        format: &'static str,
        /// Required bytes for one block.
        required: usize,
        /// Supplied bytes available to the block decoder.
        actual: usize,
    },
}

impl std::fmt::Display for KernelBodyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidBind(message) => write!(f, "invalid library bind: {message}"),
            Self::BufferTooShort {
                buffer,
                required,
                actual,
            } => write!(
                f,
                "library buffer {buffer} has {actual} elements, needs {required}"
            ),
            Self::ShapeMismatch(message) => write!(f, "library shape mismatch: {message}"),
            Self::InvalidEpsilon => {
                write!(f, "library RMSNorm epsilon must be finite and positive")
            }
            Self::InvalidRopeTable => write!(f, "library RoPE table does not match the bind"),
            Self::PackedBlockTooShort {
                format,
                required,
                actual,
            } => write!(
                f,
                "library packed {format} block has {actual} bytes, needs {required}"
            ),
        }
    }
}

impl std::error::Error for KernelBodyError {}

/// Decode one Q6_K nibble and its scale slot.
///
/// A Q6_K block is 256 elements in eight groups of 32, so `lane` is `0..=3`.
/// Any other lane fails closed instead of panicking.
fn q6_k_lane_quant(
    lane: usize,
    ql0: u8,
    ql1: u8,
    qh: u8,
    scale_index: usize,
) -> Result<(u8, usize), KernelBodyError> {
    match lane {
        0 => Ok(((ql0 & 0x0f) | ((qh & 3) << 4), scale_index)),
        1 => Ok(((ql1 & 0x0f) | (((qh >> 2) & 3) << 4), scale_index + 2)),
        2 => Ok(((ql0 >> 4) | (((qh >> 4) & 3) << 4), scale_index + 4)),
        3 => Ok(((ql1 >> 4) | (((qh >> 6) & 3) << 4), scale_index + 6)),
        _ => Err(KernelBodyError::ShapeMismatch(
            "Q6_K lane is bounded by 256-element blocks",
        )),
    }
}

fn checked_usize(value: u64) -> Result<usize, KernelBodyError> {
    usize::try_from(value)
        .map_err(|_| KernelBodyError::InvalidBind("bind index exceeds host usize"))
}

fn checked_span(
    bind: &BindDescriptor,
    buffer: &'static str,
    len: usize,
) -> Result<(), KernelBodyError> {
    bind.validate()?;
    let required = bind
        .physical_span()
        .ok_or(KernelBodyError::InvalidBind("bind physical span overflow"))?;
    if required > len as u64 {
        return Err(KernelBodyError::BufferTooShort {
            buffer,
            required,
            actual: len,
        });
    }
    Ok(())
}

/// Visit every logical element in row-major logical order, yielding its
/// physical offset and the last-axis column.  The logical order is independent
/// of the physical strides, which is the key parameterization invariant.
fn for_each_element(
    bind: &BindDescriptor,
    mut visit: impl FnMut(usize, usize, usize),
) -> Result<(), KernelBodyError> {
    bind.validate()?;
    let dims: Vec<usize> = bind
        .dims
        .iter()
        .map(|dim| checked_usize(*dim))
        .collect::<Result<_, _>>()?;
    let strides: Vec<usize> = bind
        .strides
        .iter()
        .map(|stride| checked_usize(*stride))
        .collect::<Result<_, _>>()?;
    let count = bind
        .element_count()
        .ok_or(KernelBodyError::InvalidBind("bind element count overflow"))?;
    let count = checked_usize(count)?;
    let width = *dims.last().unwrap_or(&1);
    for logical in 0..count {
        let mut remainder = logical;
        let mut offset = 0usize;
        for axis in (0..dims.len()).rev() {
            let coordinate = remainder % dims[axis];
            remainder /= dims[axis];
            offset = offset.saturating_add(coordinate.saturating_mul(strides[axis]));
        }
        visit(logical, offset, logical % width);
    }
    Ok(())
}

/// Visit all logical rows and their physical element offsets.
fn for_each_row(
    bind: &BindDescriptor,
    mut visit: impl FnMut(usize, usize, usize, usize),
) -> Result<(), KernelBodyError> {
    bind.validate()?;
    let dims: Vec<usize> = bind
        .dims
        .iter()
        .map(|dim| checked_usize(*dim))
        .collect::<Result<_, _>>()?;
    let strides: Vec<usize> = bind
        .strides
        .iter()
        .map(|stride| checked_usize(*stride))
        .collect::<Result<_, _>>()?;
    let width = *dims.last().unwrap_or(&1);
    let rows = dims[..dims.len() - 1]
        .iter()
        .try_fold(1usize, |count, dim| count.checked_mul(*dim))
        .ok_or(KernelBodyError::InvalidBind("bind row count overflow"))?;
    for row in 0..rows {
        let mut remainder = row;
        let mut base = 0usize;
        for axis in (0..dims.len() - 1).rev() {
            let coordinate = remainder % dims[axis];
            remainder /= dims[axis];
            base = base.saturating_add(coordinate.saturating_mul(strides[axis]));
        }
        visit(row, base, width, strides[dims.len() - 1]);
    }
    Ok(())
}

fn ensure_output(bind: &BindDescriptor, output: &[f32]) -> Result<(), KernelBodyError> {
    checked_span(bind, "output", output.len())
}

const SIMDGROUP_WIDTH: usize = 32;

fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) & 1;
    let exponent = u32::from(bits >> 10) & 0x1f;
    let fraction = u32::from(bits & 0x03ff);
    let bits32 = if exponent == 0 {
        if fraction == 0 {
            sign << 31
        } else {
            let position = 32 - fraction.leading_zeros();
            (sign << 31) | ((position + 102) << 23) | ((fraction << (24 - position)) - (1 << 23))
        }
    } else if exponent == 0x1f {
        (sign << 31) | (0xff << 23) | (fraction << 13)
    } else {
        (sign << 31) | ((exponent + 112) << 23) | (fraction << 13)
    };
    f32::from_bits(bits32)
}

fn get_scale_min_k4(index: usize, scales: &[u8]) -> (u8, u8) {
    if index < 4 {
        (scales[index] & 63, scales[index + 4] & 63)
    } else {
        (
            (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4),
            (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
        )
    }
}

fn block_value(
    format: QuantizedFormat,
    block: &[u8],
    element: usize,
) -> Result<f32, KernelBodyError> {
    let expected = checked_usize(format.block_bytes())?;
    if block.len() < expected {
        return Err(KernelBodyError::PackedBlockTooShort {
            format: format.spelling(),
            required: expected,
            actual: block.len(),
        });
    }
    match format {
        QuantizedFormat::Q4K => {
            let d = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin = half_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let scales = &block[4..16];
            let group = element / 64;
            let within = element % 64;
            let scale_index = group * 2 + usize::from(within >= 32);
            let (scale, min) = get_scale_min_k4(scale_index, scales);
            let q = block[16 + group * 32 + (within % 32)];
            let nibble = if within < 32 { q & 0x0f } else { q >> 4 };
            Ok(d * f32::from(scale) * f32::from(nibble) - dmin * f32::from(min))
        }
        QuantizedFormat::Q5_0 => {
            let d = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
            let pair = element % 16;
            let q = block[6 + pair];
            let high = if element < 16 {
                ((qh >> pair) << 4) & 0x10
            } else {
                (qh >> (pair + 12)) & 0x10
            };
            let nibble = if element < 16 { q & 0x0f } else { q >> 4 };
            Ok((f32::from((nibble | high as u8) as i8) - 16.0) * d)
        }
        QuantizedFormat::Q5K => {
            let d = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin = half_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let scale_index = element / 32;
            let lane = element % 32;
            let pair = element / 64;
            let (scale, min) = if scale_index < 4 {
                (block[4 + scale_index] & 63, block[8 + scale_index] & 63)
            } else {
                (
                    (block[4 + scale_index + 4] & 0x0f) | ((block[4 + scale_index - 4] >> 6) << 4),
                    (block[4 + scale_index + 4] >> 4) | ((block[4 + scale_index] >> 6) << 4),
                )
            };
            let q = block[48 + pair * 32 + lane];
            let nibble = if scale_index.is_multiple_of(2) {
                q & 0x0f
            } else {
                q >> 4
            };
            let mask = if scale_index.is_multiple_of(2) {
                1u8 << (2 * pair)
            } else {
                2u8 << (2 * pair)
            };
            let high = if block[16 + lane] & mask != 0 { 16 } else { 0 };
            Ok(d * f32::from(scale) * f32::from(nibble + high) - dmin * f32::from(min))
        }
        QuantizedFormat::Q6K => {
            let d = half_to_f32(u16::from_le_bytes([block[208], block[209]]));
            let half = (element / 128) * 128;
            let remainder = element % 128;
            let lane = remainder / 32;
            let l = remainder % 32;
            let scale_index = half / 16 + l / 16;
            let ql_offset = half / 2;
            let qh_offset = half / 4;
            let ql0 = block[ql_offset + l];
            let ql1 = block[ql_offset + l + 32];
            let qh = block[128 + qh_offset + l];
            let (q, scale_slot) = q6_k_lane_quant(lane, ql0, ql1, qh, scale_index)?;
            let scale = i8::from_ne_bytes([block[192 + scale_slot]]) as f32;
            Ok(d * scale * (i32::from(q) - 32) as f32)
        }
        QuantizedFormat::Q8_0 => {
            let d = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let q = i8::from_ne_bytes([block[2 + element]]) as f32;
            Ok(d * q)
        }
    }
}

fn simdgroup_reduce(mut lanes: [f32; SIMDGROUP_WIDTH]) -> f32 {
    let mut width = SIMDGROUP_WIDTH / 2;
    while width != 0 {
        for lane in 0..width {
            lanes[lane] += lanes[lane + width];
        }
        width /= 2;
    }
    lanes[0]
}

/// Fused quantized GEMV for an M=1 decode projection.
///
/// Routes through [`dispatch_gemv`]'s per-format table.  Each specialized
/// body streams one packed block at a time (header in registers, coalesced
/// qs), then reduces 32 simdgroup lanes.  Dequant matches the R-PACK-02
/// formulas; the body never expands a whole f32 weight matrix.
pub fn quantized_gemv(
    bind: &QuantizedGemvBind,
    activation: &[f32],
    packed_weight: &[u8],
    output: &mut [f32],
) -> Result<(), KernelBodyError> {
    dispatch_gemv(
        GemvKernel::Quantized,
        bind,
        activation,
        packed_weight,
        output,
    )
}

/// Dispatch key for a plan-selected decode GEMV body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemvKernel {
    /// Per-format packed dequant fused with the GEMV reduction.
    Quantized,
}

struct GemvBuffers<'a> {
    k: usize,
    n: usize,
    input_stride: usize,
    output_stride: usize,
    column_stride: usize,
    activation: &'a [f32],
    packed_weight: &'a [u8],
    output: &'a mut [f32],
}

fn prepare_gemv_buffers<'a>(
    bind: &QuantizedGemvBind,
    activation: &'a [f32],
    packed_weight: &'a [u8],
    output: &'a mut [f32],
) -> Result<GemvBuffers<'a>, KernelBodyError> {
    bind.validate()?;
    let input_span = checked_usize(bind.input_span())?;
    let output_span = checked_usize(bind.output_span())?;
    let weight_span = checked_usize(bind.weight_span())?;
    if activation.len() < input_span {
        return Err(KernelBodyError::BufferTooShort {
            buffer: "activation",
            required: input_span as u64,
            actual: activation.len(),
        });
    }
    if output.len() < output_span {
        return Err(KernelBodyError::BufferTooShort {
            buffer: "output",
            required: output_span as u64,
            actual: output.len(),
        });
    }
    if packed_weight.len() < weight_span {
        return Err(KernelBodyError::BufferTooShort {
            buffer: "packed_weight",
            required: weight_span as u64,
            actual: packed_weight.len(),
        });
    }
    Ok(GemvBuffers {
        k: checked_usize(bind.k)?,
        n: checked_usize(bind.n)?,
        input_stride: checked_usize(bind.input_stride)?,
        output_stride: checked_usize(bind.output_stride)?,
        column_stride: checked_usize(bind.packed_column_stride_bytes)?,
        activation,
        packed_weight,
        output,
    })
}

/// Dispatch a plan-selected decode GEMV body.
///
/// The format table is the specialization surface: Q4_K / Q5_0 / Q6_K /
/// Q8_0 (and Q5_K, retained from R-PACK-02) each own a block-streaming
/// body.  Lane mapping stays 32-wide so the f32 reduction order matches
/// the previous generic simdgroup loop.
pub fn dispatch_gemv(
    kernel: GemvKernel,
    bind: &QuantizedGemvBind,
    activation: &[f32],
    packed_weight: &[u8],
    output: &mut [f32],
) -> Result<(), KernelBodyError> {
    match kernel {
        GemvKernel::Quantized => {
            let mut buffers = prepare_gemv_buffers(bind, activation, packed_weight, output)?;
            match bind.format {
                QuantizedFormat::Q4K => gemv_q4_k(&mut buffers),
                QuantizedFormat::Q5_0 => gemv_q5_0(&mut buffers),
                QuantizedFormat::Q5K => gemv_q5_k(&mut buffers),
                QuantizedFormat::Q6K => gemv_q6_k(&mut buffers),
                QuantizedFormat::Q8_0 => gemv_q8_0(&mut buffers),
            }
        }
    }
}

fn gemv_q8_0(buffers: &mut GemvBuffers<'_>) -> Result<(), KernelBodyError> {
    let blocks = buffers.k / 32;
    for column in 0..buffers.n {
        let mut lane_sums = [0.0f32; SIMDGROUP_WIDTH];
        let col_start = column * buffers.column_stride;
        for ib in 0..blocks {
            let base = col_start + ib * 34;
            let block = &buffers.packed_weight[base..base + 34];
            let d = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let k_base = ib * 32;
            for lane in 0..SIMDGROUP_WIDTH {
                let q = i8::from_ne_bytes([block[2 + lane]]) as f32;
                lane_sums[lane] +=
                    buffers.activation[(k_base + lane) * buffers.input_stride] * (d * q);
            }
        }
        buffers.output[column * buffers.output_stride] = simdgroup_reduce(lane_sums);
    }
    Ok(())
}

fn gemv_q5_0(buffers: &mut GemvBuffers<'_>) -> Result<(), KernelBodyError> {
    let blocks = buffers.k / 32;
    for column in 0..buffers.n {
        let mut lane_sums = [0.0f32; SIMDGROUP_WIDTH];
        let col_start = column * buffers.column_stride;
        for ib in 0..blocks {
            let base = col_start + ib * 22;
            let block = &buffers.packed_weight[base..base + 22];
            let d = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
            let k_base = ib * 32;
            for lane in 0..SIMDGROUP_WIDTH {
                let pair = lane % 16;
                let q = block[6 + pair];
                let high = if lane < 16 {
                    ((qh >> pair) << 4) & 0x10
                } else {
                    (qh >> (pair + 12)) & 0x10
                };
                let nibble = if lane < 16 { q & 0x0f } else { q >> 4 };
                let weight = (f32::from((nibble | high as u8) as i8) - 16.0) * d;
                lane_sums[lane] +=
                    buffers.activation[(k_base + lane) * buffers.input_stride] * weight;
            }
        }
        buffers.output[column * buffers.output_stride] = simdgroup_reduce(lane_sums);
    }
    Ok(())
}

fn gemv_q4_k(buffers: &mut GemvBuffers<'_>) -> Result<(), KernelBodyError> {
    let blocks = buffers.k / 256;
    for column in 0..buffers.n {
        let mut lane_sums = [0.0f32; SIMDGROUP_WIDTH];
        let col_start = column * buffers.column_stride;
        for ib in 0..blocks {
            let base = col_start + ib * 144;
            let block = &buffers.packed_weight[base..base + 144];
            let d = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin = half_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let scales = &block[4..16];
            let k_base = ib * 256;
            for lane in 0..SIMDGROUP_WIDTH {
                for group in 0..8 {
                    let (scale, min) = get_scale_min_k4(group, scales);
                    let qs = block[16 + (group / 2) * 32 + lane];
                    let nibble = if group % 2 == 0 { qs & 0x0f } else { qs >> 4 };
                    let weight = d * f32::from(scale) * f32::from(nibble) - dmin * f32::from(min);
                    let index = k_base + group * 32 + lane;
                    lane_sums[lane] += buffers.activation[index * buffers.input_stride] * weight;
                }
            }
        }
        buffers.output[column * buffers.output_stride] = simdgroup_reduce(lane_sums);
    }
    Ok(())
}

fn gemv_q5_k(buffers: &mut GemvBuffers<'_>) -> Result<(), KernelBodyError> {
    let blocks = buffers.k / 256;
    for column in 0..buffers.n {
        let mut lane_sums = [0.0f32; SIMDGROUP_WIDTH];
        let col_start = column * buffers.column_stride;
        for ib in 0..blocks {
            let base = col_start + ib * 176;
            let block = &buffers.packed_weight[base..base + 176];
            let k_base = ib * 256;
            for lane in 0..SIMDGROUP_WIDTH {
                for group in 0..8 {
                    let element = group * 32 + lane;
                    let weight = block_value(QuantizedFormat::Q5K, block, element)?;
                    lane_sums[lane] +=
                        buffers.activation[(k_base + element) * buffers.input_stride] * weight;
                }
            }
        }
        buffers.output[column * buffers.output_stride] = simdgroup_reduce(lane_sums);
    }
    Ok(())
}

fn gemv_q6_k(buffers: &mut GemvBuffers<'_>) -> Result<(), KernelBodyError> {
    let blocks = buffers.k / 256;
    for column in 0..buffers.n {
        let mut lane_sums = [0.0f32; SIMDGROUP_WIDTH];
        let col_start = column * buffers.column_stride;
        for ib in 0..blocks {
            let base = col_start + ib * 210;
            let block = &buffers.packed_weight[base..base + 210];
            let d = half_to_f32(u16::from_le_bytes([block[208], block[209]]));
            let k_base = ib * 256;
            for lane in 0..SIMDGROUP_WIDTH {
                for group in 0..8 {
                    let element = group * 32 + lane;
                    let half = (element / 128) * 128;
                    let remainder = element % 128;
                    let q_lane = remainder / 32;
                    let l = remainder % 32;
                    let scale_index = half / 16 + l / 16;
                    let ql_offset = half / 2;
                    let qh_offset = half / 4;
                    let ql0 = block[ql_offset + l];
                    let ql1 = block[ql_offset + l + 32];
                    let qh = block[128 + qh_offset + l];
                    let (q, scale_slot) = q6_k_lane_quant(q_lane, ql0, ql1, qh, scale_index)?;
                    let scale = i8::from_ne_bytes([block[192 + scale_slot]]) as f32;
                    let weight = d * scale * (i32::from(q) - 32) as f32;
                    lane_sums[lane] +=
                        buffers.activation[(k_base + element) * buffers.input_stride] * weight;
                }
            }
        }
        buffers.output[column * buffers.output_stride] = simdgroup_reduce(lane_sums);
    }
    Ok(())
}

/// Select the decode GEMV library body from executor-plan facts.
///
/// The generic M5-U2 spellings (`quantized_gemv` / `quantized_gemm`) and the
/// amended transformer projection entries (`QkvProjection`, `OutputProjection`,
/// `SwiGlu`) share the same packed simdgroup body.  `decode_gemv` is the
/// matching uniform (`1` when M/seq is 1); prefill keeps GEMM.  The two facts
/// must agree, and unknown entries fail closed.
pub fn select_decode_gemv(
    library_entry: Option<&str>,
    decode_gemv: u32,
) -> Result<Option<GemvKernel>, KernelBodyError> {
    let transformer_projection = matches!(
        library_entry,
        Some("QkvProjection" | "OutputProjection" | "SwiGlu")
    );
    let gemv_entry = library_entry == Some("quantized_gemv");
    let gemm_entry = library_entry == Some("quantized_gemm");
    match (gemv_entry, gemm_entry, transformer_projection, decode_gemv) {
        (true, false, false, 1) | (false, false, true, 1) => Ok(Some(GemvKernel::Quantized)),
        (false, true, false, 0) | (false, false, true, 0) => Ok(None),
        (false, false, false, 0) => Ok(None),
        (false, false, false, 1) if library_entry.is_none() => Ok(Some(GemvKernel::Quantized)),
        _ => Err(KernelBodyError::InvalidBind(
            "decode GEMV selection disagrees with library_entry",
        )),
    }
}

/// RMS normalization over the last logical axis, with affine `gamma`.
///
/// Arithmetic and operation order match the existing body: sum `x*x`, divide
/// by the row width, add epsilon inside the square root, then write
/// `x * (1/sqrt(mean + eps)) * gamma[col]`.
pub fn rms(
    bind: &BindDescriptor,
    input: &[f32],
    gamma: &[f32],
    output: &mut [f32],
    epsilon: f32,
) -> Result<(), KernelBodyError> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(KernelBodyError::InvalidEpsilon);
    }
    checked_span(bind, "input", input.len())?;
    ensure_output(bind, output)?;
    let width = checked_usize(*bind.dims.last().unwrap_or(&0))?;
    if gamma.len() < width {
        return Err(KernelBodyError::BufferTooShort {
            buffer: "gamma",
            required: width as u64,
            actual: gamma.len(),
        });
    }
    for_each_row(bind, |_, base, width, stride| {
        let mut sumsq = 0.0f32;
        for col in 0..width {
            let x = input[base + col * stride];
            sumsq += x * x;
        }
        let mean = sumsq / width as f32;
        let scale = 1.0f32 / (mean + epsilon).sqrt();
        for col in 0..width {
            let offset = base + col * stride;
            output[offset] = input[offset] * scale * gamma[col];
        }
    })?;
    Ok(())
}

/// Pointwise residual addition: `output = left + right`.
pub fn residual(
    bind: &BindDescriptor,
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
) -> Result<(), KernelBodyError> {
    checked_span(bind, "left", left.len())?;
    checked_span(bind, "right", right.len())?;
    ensure_output(bind, output)?;
    for_each_element(bind, |_, offset, _| {
        output[offset] = left[offset] + right[offset]
    })?;
    Ok(())
}

/// SwiGLU body: `silu(gate) * up`, retaining the existing scalar operation
/// order (`neg`, `exp`, `add`, `div`, then the two multiplies).
pub fn swiglu(
    bind: &BindDescriptor,
    gate: &[f32],
    up: &[f32],
    output: &mut [f32],
) -> Result<(), KernelBodyError> {
    checked_span(bind, "gate", gate.len())?;
    checked_span(bind, "up", up.len())?;
    ensure_output(bind, output)?;
    for_each_element(bind, |_, offset, _| {
        let neg = -gate[offset];
        let exp = neg.exp();
        let add1 = exp + 1.0;
        let div = 1.0 / add1;
        let silu = gate[offset] * div;
        output[offset] = silu * up[offset];
    })?;
    Ok(())
}

/// RoPE body using host-precomputed cosine and sine tables.
///
/// A `RopeConsecutivePair` bind uses `(2p, 2p+1)` pairs.  A
/// `RopeRotateHalf` bind uses the NeoX pair `(p, p + dim/2)`.  Tables may be
/// rank-1 (one position for every row) or rank-2 (one position per row); the
/// descriptor's dimensions determine the row count and pair width.
pub fn rope(
    bind: &BindDescriptor,
    input: &[f32],
    cos: &[f32],
    sin: &[f32],
    output: &mut [f32],
) -> Result<(), KernelBodyError> {
    checked_span(bind, "input", input.len())?;
    ensure_output(bind, output)?;
    let (dim, rotate_half) = match bind.layout {
        BindLayout::RopeConsecutivePair { rotated_width } => (checked_usize(rotated_width)?, false),
        BindLayout::RopeRotateHalf { rotated_width } => (checked_usize(rotated_width)?, true),
        _ => {
            return Err(KernelBodyError::ShapeMismatch(
                "rope bind has no rope layout",
            ))
        }
    };
    let pairs = dim / 2;
    let rows = bind
        .element_count()
        .ok_or(KernelBodyError::InvalidBind("rope element count overflow"))?
        / bind.dims.last().copied().unwrap_or(1);
    let rows = checked_usize(rows)?;
    let per_row_tables = match (cos.len(), sin.len()) {
        (a, b) if a == pairs && b == pairs => false,
        (a, b) if a == rows.saturating_mul(pairs) && b == rows.saturating_mul(pairs) => true,
        _ => return Err(KernelBodyError::InvalidRopeTable),
    };
    let mut row_number = 0usize;
    for_each_row(bind, |_, base, row_width, stride| {
        let table_base = if per_row_tables {
            row_number * pairs
        } else {
            0
        };
        for col in 0..row_width {
            let offset = base + col * stride;
            if col >= dim {
                output[offset] = input[offset];
                continue;
            }
            let pair = if rotate_half { col % pairs } else { col / 2 };
            let cos_t = cos[table_base + pair];
            let sin_t = sin[table_base + pair];
            if rotate_half {
                if col < pairs {
                    let x0 = input[base + col * stride];
                    let x1 = input[base + (col + pairs) * stride];
                    output[offset] = (x0 * cos_t) - (x1 * sin_t);
                } else {
                    let x0 = input[base + (col - pairs) * stride];
                    let x1 = input[offset];
                    output[offset] = (x0 * sin_t) + (x1 * cos_t);
                }
            } else if col % 2 == 0 {
                let x0 = input[offset];
                let x1 = input[base + (col + 1) * stride];
                output[offset] = (x0 * cos_t) - (x1 * sin_t);
            } else {
                let x0 = input[base + (col - 1) * stride];
                let x1 = input[offset];
                output[offset] = (x0 * sin_t) + (x1 * cos_t);
            }
        }
        row_number += 1;
    })?;
    debug_assert_eq!(row_number, rows);
    Ok(())
}

/// Numerically stable row-wise softmax over the last logical axis.
pub fn softmax(
    bind: &BindDescriptor,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), KernelBodyError> {
    checked_span(bind, "input", input.len())?;
    ensure_output(bind, output)?;
    for_each_row(bind, |_, base, width, stride| {
        let mut row_max = input[base];
        for col in 1..width {
            row_max = row_max.max(input[base + col * stride]);
        }
        let mut row_sum = 0.0f32;
        for col in 0..width {
            row_sum += (input[base + col * stride] - row_max).exp();
        }
        for col in 0..width {
            let offset = base + col * stride;
            output[offset] = (input[offset] - row_max).exp() / row_sum;
        }
    })?;
    Ok(())
}

/// Causal variant used by the existing dense attention path.  It shares the
/// same stable row-softmax body and only changes the logical row extent.
pub fn causal_softmax(
    bind: &BindDescriptor,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), KernelBodyError> {
    checked_span(bind, "input", input.len())?;
    ensure_output(bind, output)?;
    for_each_row(bind, |row, base, width, stride| {
        let masked = (row + 1).min(width);
        let mut row_max = input[base];
        for col in 1..masked {
            row_max = row_max.max(input[base + col * stride]);
        }
        let mut row_sum = 0.0f32;
        for col in 0..masked {
            row_sum += (input[base + col * stride] - row_max).exp();
        }
        for col in 0..width {
            let offset = base + col * stride;
            output[offset] = if col >= masked {
                0.0
            } else {
                (input[offset] - row_max).exp() / row_sum
            };
        }
    })?;
    Ok(())
}

fn attention_offset(coordinates: &[u64], strides: &[u64]) -> Result<usize, KernelBodyError> {
    if coordinates.len() != strides.len() {
        return Err(KernelBodyError::InvalidBind(
            "causal attention coordinate rank does not match its strides",
        ));
    }
    let offset = coordinates
        .iter()
        .zip(strides)
        .try_fold(0u64, |offset, (coordinate, stride)| {
            coordinate
                .checked_mul(*stride)
                .and_then(|term| offset.checked_add(term))
        })
        .ok_or(KernelBodyError::InvalidBind(
            "causal attention element offset overflow",
        ))?;
    checked_usize(offset)
}

fn ensure_attention_buffer(
    bind: &CausalAttentionBind,
    name: &'static str,
    dims: &[u64],
    strides: &[u64],
    len: usize,
) -> Result<(), KernelBodyError> {
    let required = attention_span(dims, strides)?;
    if required > len as u64 {
        return Err(KernelBodyError::BufferTooShort {
            buffer: name,
            required,
            actual: len,
        });
    }
    // Keep the bind in this helper's contract: the dimensions and strides are
    // never inferred from the supplied slice length.
    bind.validate()
}

/// Fused causal attention over every query head in each KV group.
///
/// The body performs the existing M3 operation order in one library call:
/// dot-product scores, scale by `1/sqrt(head_dim)`, causal mask, online
/// softmax, and the value/context reduction.  The running max, sum, and
/// per-dimension context accumulators are scoped to one query head, so a
/// head's softmax cannot observe another head's score or value.
///
/// The function is the host reference for the plan-selected library entry.
/// It is intentionally not wired into the legacy descriptor decomposition;
/// M4-U2 owns the selection that makes this entry reachable from a device
/// plan.
pub fn causal_attention(
    bind: &CausalAttentionBind,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
) -> Result<(), KernelBodyError> {
    bind.validate()?;
    let q_dims = [bind.kv_batch, bind.q_per_kv, bind.query_seq, bind.head_dim];
    let kv_dims = [bind.kv_batch, bind.seq_block, bind.head_dim];
    ensure_attention_buffer(bind, "q", &q_dims, &bind.q_strides, q.len())?;
    ensure_attention_buffer(bind, "k", &kv_dims, &bind.k_strides, k.len())?;
    ensure_attention_buffer(bind, "v", &kv_dims, &bind.v_strides, v.len())?;
    ensure_attention_buffer(bind, "output", &q_dims, &bind.output_strides, output.len())?;

    let scale = 1.0f32 / (bind.head_dim as f32).sqrt();
    let head_dim = checked_usize(bind.head_dim)?;
    let seq_block = checked_usize(bind.seq_block)?;
    let q_per_kv = checked_usize(bind.q_per_kv)?;
    let kv_batch = checked_usize(bind.kv_batch)?;
    let query_seq = checked_usize(bind.query_seq)?;
    let query_start = seq_block - query_seq;

    // Online softmax is the host reference for the streaming Metal body.  It
    // keeps one accumulator per output dimension and never materializes the
    // query-by-sequence score tile.  The max/sum rescale is the standard flash
    // recurrence and preserves per-head independence.
    for group in 0..kv_batch {
        for q_head in 0..q_per_kv {
            for query in 0..query_seq {
                let query_position = query_start + query;
                let visible = (query_position + 1).min(seq_block);
                let mut row_max = f32::NEG_INFINITY;
                let mut row_sum = 0.0f32;
                let mut context = vec![0.0f32; head_dim];

                for token in 0..visible {
                    let mut dot = 0.0f32;
                    for dimension in 0..head_dim {
                        let q_offset = attention_offset(
                            &[group as u64, q_head as u64, query as u64, dimension as u64],
                            &bind.q_strides,
                        )?;
                        let k_offset = attention_offset(
                            &[group as u64, token as u64, dimension as u64],
                            &bind.k_strides,
                        )?;
                        dot += q[q_offset] * k[k_offset];
                    }
                    let score = dot * scale;
                    let next_max = row_max.max(score);
                    let old_scale = (row_max - next_max).exp();
                    let token_scale = (score - next_max).exp();
                    row_sum = row_sum * old_scale + token_scale;
                    for (dimension, value) in context.iter_mut().enumerate() {
                        let v_offset = attention_offset(
                            &[group as u64, token as u64, dimension as u64],
                            &bind.v_strides,
                        )?;
                        *value = *value * old_scale + token_scale * v[v_offset];
                    }
                    row_max = next_max;
                }

                for (dimension, value) in context.into_iter().enumerate() {
                    let output_offset = attention_offset(
                        &[group as u64, q_head as u64, query as u64, dimension as u64],
                        &bind.output_strides,
                    )?;
                    output[output_offset] = value / row_sum;
                }
            }
        }
    }
    Ok(())
}

/// The plan-path library selections currently materialized by this host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LibraryKernel {
    /// Scale + causal scores + softmax + context for one KV-group batch.
    CausalAttention,
    /// One grouped Q/K/V projection body with bind-supplied weights/views.
    QkvProjection,
}

/// Dispatch one plan-selected library kernel.
///
/// Keeping this narrow dispatcher separate from the legacy descriptor
/// executor prevents a second, dead attention route before M4-U2's planner
/// selection lands.
pub fn dispatch(
    kernel: LibraryKernel,
    bind: &CausalAttentionBind,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
) -> Result<(), KernelBodyError> {
    match kernel {
        LibraryKernel::CausalAttention => causal_attention(bind, q, k, v, output),
        LibraryKernel::QkvProjection => Err(KernelBodyError::InvalidBind(
            "QKV projection requires its grouped bind and three outputs",
        )),
    }
}

/// Compatibility alias for callers that name the fused body explicitly.
pub use causal_attention as causal_attention_fused;

/// Compatibility aliases using the names used by the MIR recipes.
pub use causal_softmax as causal_masked_softmax;
pub use residual as residual_add;
pub use rms as rms_norm;

#[cfg(test)]
#[path = "library_test.rs"]
mod tests;
