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
        }
    }
}

impl std::error::Error for KernelBodyError {}

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
/// dot-product scores, scale by `1/sqrt(head_dim)`, causal mask, stable
/// softmax, and the value/context reduction.  `shared_max` and `shared_sum`
/// model the device threadgroup's per-head shared-memory accumulators.  They
/// are indexed by `q_head`, never shared between query heads, so a head's
/// softmax cannot observe another head's score or value.
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

    // These vectors stand in for threadgroup shared memory on the Metal
    // materializer.  There is one max and one sum slot for every q head in the
    // group, rather than one accumulator for the whole group.
    let mut shared_max = vec![0.0f32; q_per_kv];
    let mut shared_sum = vec![0.0f32; q_per_kv];
    let mut scores = vec![0.0f32; seq_block];

    for group in 0..kv_batch {
        for q_head in 0..q_per_kv {
            for query in 0..query_seq {
                let query_position = query_start + query;
                let visible = (query_position + 1).min(seq_block);
                let mut row_max = f32::NEG_INFINITY;

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
                    scores[token] = score;
                    row_max = row_max.max(score);
                }
                shared_max[q_head] = row_max;

                let mut row_sum = 0.0f32;
                for token in 0..visible {
                    row_sum += (scores[token] - shared_max[q_head]).exp();
                }
                shared_sum[q_head] = row_sum;

                for dimension in 0..head_dim {
                    let mut context = 0.0f32;
                    for token in 0..visible {
                        let probability =
                            (scores[token] - shared_max[q_head]).exp() / shared_sum[q_head];
                        let v_offset = attention_offset(
                            &[group as u64, token as u64, dimension as u64],
                            &bind.v_strides,
                        )?;
                        context += probability * v[v_offset];
                    }
                    let output_offset = attention_offset(
                        &[group as u64, q_head as u64, query as u64, dimension as u64],
                        &bind.output_strides,
                    )?;
                    output[output_offset] = context;
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
    }
}

/// Compatibility alias for callers that name the fused body explicitly.
pub use causal_attention as causal_attention_fused;

/// Compatibility aliases using the names used by the MIR recipes.
pub use causal_softmax as causal_masked_softmax;
pub use residual as residual_add;
pub use rms as rms_norm;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strided_residual_matches_contiguous_logical_values() {
        let contiguous = BindDescriptor::row_major(vec![2, 3], [1, 1, 1]);
        let strided = BindDescriptor::strided(vec![2, 3], vec![4, 1], [1, 1, 1]);
        let left = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let right = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut expected = [0.0; 6];
        residual(&contiguous, &left, &right, &mut expected).expect("contiguous residual");
        let left_strided = [1.0, 2.0, 3.0, 99.0, 4.0, 5.0, 6.0];
        let right_strided = [6.0, 5.0, 4.0, 88.0, 3.0, 2.0, 1.0];
        let mut actual = [0.0; 7];
        residual(&strided, &left_strided, &right_strided, &mut actual).expect("strided residual");
        assert_eq!(&actual[..3], &expected[..3]);
        assert_eq!(&actual[4..7], &expected[3..6]);
    }

    #[test]
    fn all_library_bodies_consume_bind_facts() {
        let bind = BindDescriptor::row_major(vec![2, 4], [1, 1, 1]);
        let input = [0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0];
        let gamma = [1.0, 0.9, 1.1, 0.8];
        let mut output = [0.0; 8];
        rms(&bind, &input, &gamma, &mut output, 1e-5).expect("rms");
        residual(&bind, &input, &output, &mut [0.0; 8]).expect("residual");
        swiglu(&bind, &input, &output, &mut [0.0; 8]).expect("swiglu");
        softmax(&bind, &input, &mut output).expect("softmax");
        let rope_bind = BindDescriptor {
            dims: vec![2, 4],
            strides: vec![4, 1],
            layout: BindLayout::RopeConsecutivePair { rotated_width: 4 },
            grid: [1, 1, 1],
        };
        rope(&rope_bind, &input, &[1.0, 1.0], &[0.0, 0.0], &mut output).expect("rope");
    }

    #[test]
    fn invalid_bind_fails_before_buffer_access() {
        let bind = BindDescriptor::strided(vec![2, 3], vec![0, 1], [1, 1, 1]);
        let mut output = [0.0; 6];
        assert!(matches!(
            residual(&bind, &[], &[], &mut output),
            Err(KernelBodyError::InvalidBind(_))
        ));
    }
}
