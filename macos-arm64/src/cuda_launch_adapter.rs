//! CUDA host launch adapter: emitted NVVM descriptor v2 + PTX → launch (U-03).
//!
//! The adapter is the host half of the Stage 2 launch route
//! (`docs/factory/codex-gap-campaign/delivery-stage2.md`, U-03). It consumes
//! the compiler-emitted NVVM kernel descriptor sidecar (`schema_version` 2,
//! `radix-mir-llvm/src/nvvm_descriptor.rs`) plus the emitted PTX bytes and
//! drives the existing generalized [`CudaHostSession::launch_kernel_3d`]
//! surface in one ordered sequence:
//!
//! 1. **Parse + validate fail-closed** — the sidecar JSON is parsed into a
//!    typed launch plan; every structural rule (schema version, target tag,
//!    buffer shapes/counts/roles/bindings, kernel entry names, NVVM dtype
//!    family, launch grid/block) is enforced with the typed `E_DEVICE_*`
//!    codes from [`crate::device_descriptor`] before any driver call. Grid/
//!    block axes must be non-zero and fit `u32` — an out-of-range axis is
//!    rejected, never saturated to `u32::MAX`.
//! 2. **Load module** — the PTX bytes are loaded once per launch.
//! 3. **Allocate from the descriptor** — every storage buffer is allocated
//!    with the byte length derived from its descriptor element count and the
//!    kernel's byte width (never from re-derived text).
//! 4. **Copy in host inputs** — `input` / `extra-input` buffers receive the
//!    host f32 values keyed by binding slot; a missing or wrong-sized input
//!    fails closed with `E_DEVICE_SHAPE_MISMATCH`. `accumulation` buffers are
//!    zero-filled at allocation (the host ZeroFill convention; the v2 sidecar
//!    carries no initialization axis).
//! 5. **Launch** — exactly one `launch_kernel_3d` call with the descriptor's
//!    plan-driven grid/block. The plan facts (`tiled_matmul` `m/k/n` and
//!    `workgroup_x/y`, `tree_reduction` `length`/`partials`/`workgroup_x`)
//!    are cross-checked against the buffer contract and the launch geometry,
//!    so a tampered descriptor cannot carry two conflicting launch
//!    authorities (single launch authority).
//! 6. **Sync** — the explicit step-boundary barrier after the launch.
//! 7. **Read back** — `output` / `extra-output` buffers are read back as f32
//!    rows; an optional numeric oracle (`|a−b| ≤ atol + rtol·|b|`) verifies
//!    the first output row.
//!
//! The adapter releases every handle it allocates (buffers + module) on
//! success AND on error, so a failed launch never leaks at the driver
//! boundary (S2-2 posture).
//!
//! # Schema-constant tracking (U-01 residual a)
//!
//! [`NVVM_DESCRIPTOR_SCHEMA_VERSION`] tracks `NVVM_DESCRIPTOR_SCHEMA_VERSION`
//! in `radix-mir-llvm/src/nvvm_descriptor.rs`; a bump to a newer additive
//! schema lands in the same change set. The v1 consumer lane
//! `hosts/scripta/cuda-tier-f-proof` still asserts `schema_version == 1` and
//! must be updated before that lane is re-run — that script is
//! auditor/operator-owned and is not edited here.
//!
//! Unit tests drive the sequencing fake (`FakeCudaDriver`), never a real
//! device; real-device execution is the runpod-gated U-05 step.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::cuda_host::{CudaHandleId, CudaHostSession};
use crate::device_descriptor::errors;
use crate::device_descriptor::{E_DEVICE_DESCRIPTOR, E_DEVICE_DTYPE_MISMATCH};
use crate::kernel::{HostError, HostResult};

/// The emitted NVVM descriptor schema version this adapter consumes. Tracks
/// `NVVM_DESCRIPTOR_SCHEMA_VERSION` in `radix-mir-llvm/src/nvvm_descriptor.rs`
/// (U-01); a schema bump must land here in the same change set.
pub const NVVM_DESCRIPTOR_SCHEMA_VERSION: u32 = 2;

/// The emitted descriptor target tag this adapter consumes.
pub const NVVM_DESCRIPTOR_TARGET: &str = "llvm-nvvm";

/// Closed set of kernel-plan identities the adapter knows (mirrors the radix
/// `CollectionKernelPlan` closed set from `radix-mir/src/kernel_plan/plan.rs`).
/// A plan kind outside the set fails closed at parse.
const KNOWN_PLAN_KINDS: &[&str] = &[
    "elementwise",
    "tiled_matmul",
    "tree_reduction",
    "transpose",
    "axis_reduction",
    "row_softmax",
    "layer_normalization",
    "gather",
    "rms_normalization",
    "rope",
    "causal_masked_softmax",
];

/// The NVVM kernel scalar family (`radix-mir-llvm` emits f32/i32/u32 kernels
/// only). The adapter validates the descriptor's dtype against this family
/// with `E_DEVICE_DTYPE_MISMATCH`. The session's transfer surface is f32-only
/// today, so a non-f32 descriptor passes family validation and then fails
/// closed at the transfer stage with the same typed code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvvmElementType {
    /// IEEE 754 single precision.
    F32,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 32-bit integer.
    U32,
}

impl NvvmElementType {
    /// Stable diagnostic spelling.
    #[must_use]
    pub fn spelling(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::I32 => "i32",
            Self::U32 => "u32",
        }
    }

    /// Parse an element type from its stable spelling.
    #[must_use]
    pub fn from_spelling(spelling: &str) -> Option<Self> {
        match spelling {
            "f32" => Some(Self::F32),
            "i32" => Some(Self::I32),
            "u32" => Some(Self::U32),
            _ => None,
        }
    }

    /// Byte width of one element.
    #[must_use]
    pub fn byte_width(self) -> u32 {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
        }
    }
}

/// Buffer role at the kernel — the v2 sidecar's role tags in binding order
/// (inputs, output, extra outputs, accumulation buffers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterBufferRole {
    /// Host-provided read-only input (the first input buffer).
    Input,
    /// Host-provided read-only extra input.
    ExtraInput,
    /// Device-produced output, read back after the launch.
    Output,
    /// Device-produced extra output, read back after the launch.
    ExtraOutput,
    /// Read-write accumulation buffer, zero-filled at allocation.
    Accumulation,
}

impl AdapterBufferRole {
    /// Parse a role from its sidecar tag.
    #[must_use]
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "input" => Some(Self::Input),
            "extra-input" => Some(Self::ExtraInput),
            "output" => Some(Self::Output),
            "extra-output" => Some(Self::ExtraOutput),
            "accumulation" => Some(Self::Accumulation),
            _ => None,
        }
    }

    /// Whether this role receives host-provided values.
    #[must_use]
    pub fn is_input(self) -> bool {
        matches!(self, Self::Input | Self::ExtraInput)
    }

    /// Whether this role is read back as an output row.
    #[must_use]
    pub fn is_output(self) -> bool {
        matches!(self, Self::Output | Self::ExtraOutput)
    }

    /// The sidecar tag spelling.
    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::ExtraInput => "extra-input",
            Self::Output => "output",
            Self::ExtraOutput => "extra-output",
            Self::Accumulation => "accumulation",
        }
    }
}

/// One typed storage buffer of the launch plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NvvmLaunchBuffer {
    /// Role at the kernel.
    pub role: AdapterBufferRole,
    /// Binding slot.
    pub binding: u32,
    /// Flat element count (the product of `shape`).
    pub element_count: u64,
    /// Literal shape dims.
    pub shape: Vec<u64>,
}

/// The validated plan the adapter launches from: the typed projection of one
/// descriptor kernel entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NvvmLaunchPlan {
    /// Kernel entry symbol (the `@` name in the emitted PTX).
    pub entry: String,
    /// Element type of the kernel's buffers (the NVVM scalar family).
    pub element_ty: NvvmElementType,
    /// Storage buffers in binding order.
    pub buffers: Vec<NvvmLaunchBuffer>,
    /// Dispatch (grid) axes.
    pub grid: [u32; 3],
    /// Workgroup (block) axes.
    pub block: [u32; 3],
    /// Closed-set plan identity (`None` for elementwise kernels).
    pub plan_kind: Option<String>,
}

/// Numeric oracle: the expected output row with the tolerance rule
/// `|a−b| ≤ atol + rtol·|b|` (the U-04 matmul numeric-policy rule).
#[derive(Debug, Clone, PartialEq)]
pub struct NumericOracle {
    /// Expected output row.
    pub expected: Vec<f32>,
    /// Absolute tolerance.
    pub atol: f64,
    /// Relative tolerance.
    pub rtol: f64,
}

impl NumericOracle {
    /// A new oracle with the given expected row and tolerances.
    #[must_use]
    pub fn new(expected: Vec<f32>, atol: f64, rtol: f64) -> Self {
        Self {
            expected,
            atol,
            rtol,
        }
    }

    /// Apply the tolerance rule `|a−b| ≤ atol + rtol·|b|` per element.
    /// Returns `(matched, max_abs_delta)`.
    #[must_use]
    pub fn matches(&self, actual: &[f32]) -> (bool, f64) {
        if actual.len() != self.expected.len() {
            return (false, f64::INFINITY);
        }
        let mut max_delta = 0.0f64;
        for (left, right) in actual.iter().zip(self.expected.iter()) {
            let delta = (f64::from(*left) - f64::from(*right)).abs();
            max_delta = max_delta.max(delta);
            if delta > self.atol + self.rtol * f64::from(*right).abs() {
                return (false, max_delta);
            }
        }
        (true, max_delta)
    }
}

/// Numeric oracle verdict recorded in the receipt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OracleCheck {
    /// Whether the readback row satisfied the tolerance rule.
    pub matched: bool,
    /// Largest per-element absolute deviation observed.
    pub max_abs_delta: f64,
}

/// Outcome of one adapter launch: the observable receipt of the
/// load → alloc → copy → launch → sync → readback sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct AdapterLaunchReceipt {
    /// Kernel entry launched.
    pub entry: String,
    /// FNV-1a provenance hash of the loaded PTX bytes.
    pub module_hash: u64,
    /// Number of launches dispatched (always 1; the adapter is a single
    /// launch authority).
    pub launches: usize,
    /// Storage buffers allocated from the descriptor.
    pub allocated_buffers: usize,
    /// Host→device f32 copies performed.
    pub copy_ins: usize,
    /// Accumulation buffers zero-filled at allocation.
    pub zero_fills: usize,
    /// Output buffers read back.
    pub readbacks: usize,
    /// Handles released after the launch (buffers + module).
    pub releases: usize,
    /// Readback rows keyed by output binding.
    pub outputs: BTreeMap<u32, Vec<f32>>,
    /// Oracle verdict when one was supplied.
    pub oracle: Option<OracleCheck>,
}

/// JSON mirror of the emitted v2 descriptor. Fields that validation rules on
/// with a typed code carry serde defaults so a structurally bad sidecar
/// produces a typed `E_DEVICE_*` error instead of a raw parse error.
#[derive(Debug, Deserialize)]
struct NvvmDescriptorDocument {
    #[serde(default)]
    schema_version: u32,
    #[serde(default)]
    target: String,
    #[serde(default)]
    kernels: Vec<NvvmKernelJson>,
}

#[derive(Debug, Deserialize)]
struct NvvmKernelJson {
    #[serde(default)]
    entry: String,
    #[serde(default)]
    element_type: String,
    #[serde(default)]
    element_byte_width: u32,
    #[serde(default)]
    element_count: u64,
    #[serde(default)]
    element_counts: Vec<u64>,
    #[serde(default)]
    input_buffers: usize,
    #[serde(default)]
    output_buffers: usize,
    #[serde(default)]
    accumulation_buffers: usize,
    #[serde(default)]
    buffers: Vec<NvvmBufferJson>,
    #[serde(default)]
    launch: Option<NvvmLaunchJson>,
    #[serde(default)]
    plan: Option<NvvmPlanJson>,
}

#[derive(Debug, Deserialize)]
struct NvvmBufferJson {
    #[serde(default)]
    role: String,
    #[serde(default)]
    binding: u32,
    #[serde(default)]
    element_count: u64,
    #[serde(default)]
    shape: Vec<u64>,
}

#[derive(Debug, Deserialize)]
struct NvvmLaunchJson {
    workgroup: NvvmAxisJson,
    dispatch: NvvmAxisJson,
}

#[derive(Debug, Deserialize)]
struct NvvmAxisJson {
    #[serde(default)]
    x: u64,
    #[serde(default)]
    y: u64,
    #[serde(default)]
    z: u64,
}

#[derive(Debug, Deserialize)]
struct NvvmPlanJson {
    #[serde(default)]
    kind: String,
    m: Option<u64>,
    k: Option<u64>,
    n: Option<u64>,
    workgroup_x: Option<u32>,
    workgroup_y: Option<u32>,
    op: Option<String>,
    length: Option<u64>,
    partials: Option<u64>,
}

/// Parse and validate an emitted NVVM descriptor v2 sidecar into a typed
/// launch plan. Every rule fails closed before any driver call:
///
/// - JSON structure / schema version / target tag / empty kernel list /
///   missing launch geometry / zero or `u32`-out-of-range grid/block axes →
///   `E_DEVICE_DESCRIPTOR`;
/// - dtype outside the NVVM scalar family or a byte width contradicting it →
///   `E_DEVICE_DTYPE_MISMATCH`;
/// - buffer count vs shape, cross-referenced counts, or plan count
///   contradictions → `E_DEVICE_SHAPE_MISMATCH`;
/// - role/count/binding ABI contradictions → `E_DEVICE_ABI_MISMATCH`.
///
/// # Errors
/// Typed `E_DEVICE_*` host errors (see above).
pub fn parse_descriptor(descriptor_json: &[u8]) -> HostResult<NvvmLaunchPlan> {
    let document: NvvmDescriptorDocument =
        serde_json::from_slice(descriptor_json).map_err(|error| HostError {
            code: E_DEVICE_DESCRIPTOR.to_owned(),
            message: format!("nvvm descriptor JSON does not parse: {error}"),
            retryable: false,
        })?;
    if document.schema_version != NVVM_DESCRIPTOR_SCHEMA_VERSION {
        return Err(errors::descriptor(format!(
            "nvvm descriptor schema_version {} is not supported by this host adapter (expected {NVVM_DESCRIPTOR_SCHEMA_VERSION})",
            document.schema_version
        )));
    }
    if document.target != NVVM_DESCRIPTOR_TARGET {
        return Err(errors::descriptor(format!(
            "nvvm descriptor target `{}` is not this adapter's target (`{NVVM_DESCRIPTOR_TARGET}`)",
            document.target
        )));
    }
    if document.kernels.is_empty() {
        return Err(errors::descriptor(
            "nvvm descriptor declares no kernels",
        ));
    }
    if document.kernels.len() > 1 {
        return Err(errors::descriptor(format!(
            "nvvm descriptor declares {} kernels; this host adapter launches exactly one kernel per descriptor",
            document.kernels.len()
        )));
    }
    validate_kernel(&document.kernels[0])
}

/// Structural validation of one descriptor kernel entry (fail-closed).
fn validate_kernel(kernel: &NvvmKernelJson) -> HostResult<NvvmLaunchPlan> {
    if kernel.entry.trim().is_empty() {
        return Err(errors::descriptor(
            "nvvm descriptor has a kernel with an empty entry name",
        ));
    }
    let element_ty = NvvmElementType::from_spelling(&kernel.element_type).ok_or_else(|| {
        HostError {
            code: E_DEVICE_DTYPE_MISMATCH.to_owned(),
            message: format!(
                "nvvm descriptor kernel `{}` declares element type `{}` outside the NVVM scalar family (f32/i32/u32)",
                kernel.entry, kernel.element_type
            ),
            retryable: false,
        }
    })?;
    if element_ty.byte_width() != kernel.element_byte_width {
        return Err(errors::dtype_mismatch(format!(
            "nvvm descriptor kernel `{}` declares element type `{}` with byte width {} (expected {})",
            kernel.entry,
            kernel.element_type,
            kernel.element_byte_width,
            element_ty.byte_width()
        )));
    }
    if kernel.buffers.is_empty() {
        return Err(errors::descriptor(format!(
            "nvvm descriptor kernel `{}` declares no storage buffers",
            kernel.entry
        )));
    }

    let mut buffers: Vec<NvvmLaunchBuffer> = Vec::with_capacity(kernel.buffers.len());
    let mut bindings: Vec<u32> = Vec::with_capacity(kernel.buffers.len());
    let mut input_count = 0usize;
    let mut output_count = 0usize;
    let mut accumulation_count = 0usize;
    for buffer in &kernel.buffers {
        let role = AdapterBufferRole::from_tag(&buffer.role).ok_or_else(|| {
            errors::descriptor(format!(
                "nvvm descriptor kernel `{}` declares unknown buffer role `{}`",
                kernel.entry, buffer.role
            ))
        })?;
        if buffer.element_count == 0 {
            return Err(errors::descriptor(format!(
                "nvvm descriptor kernel `{}` declares a zero-count buffer at binding {}",
                kernel.entry, buffer.binding
            )));
        }
        if bindings.contains(&buffer.binding) {
            return Err(errors::abi_mismatch(format!(
                "nvvm descriptor kernel `{}` binds index {} more than once",
                kernel.entry, buffer.binding
            )));
        }
        bindings.push(buffer.binding);
        // Bounds sanity on descriptor inputs: the flat count must equal the
        // product of the literal shape dims.
        let shape_product = buffer
            .shape
            .iter()
            .try_fold(1u64, |acc, dim| acc.checked_mul(*dim))
            .ok_or_else(|| {
                errors::descriptor(format!(
                    "nvvm descriptor kernel `{}` buffer `{}` shape dims overflow",
                    kernel.entry, buffer.role
                ))
            })?;
        if shape_product != buffer.element_count {
            return Err(errors::shape_mismatch(format!(
                "nvvm descriptor kernel `{}` buffer `{}` (binding {}) declares element_count {} but its shape {:?} product is {}",
                kernel.entry,
                buffer.role,
                buffer.binding,
                buffer.element_count,
                buffer.shape,
                shape_product
            )));
        }
        match role {
            AdapterBufferRole::Input | AdapterBufferRole::ExtraInput => input_count += 1,
            AdapterBufferRole::Output | AdapterBufferRole::ExtraOutput => output_count += 1,
            AdapterBufferRole::Accumulation => accumulation_count += 1,
        }
        buffers.push(NvvmLaunchBuffer {
            role,
            binding: buffer.binding,
            element_count: buffer.element_count,
            shape: buffer.shape.clone(),
        });
    }

    if output_count == 0 {
        return Err(errors::descriptor(format!(
            "nvvm descriptor kernel `{}` declares no output buffer to read back",
            kernel.entry
        )));
    }
    if input_count != kernel.input_buffers
        || output_count != kernel.output_buffers
        || accumulation_count != kernel.accumulation_buffers
    {
        return Err(errors::abi_mismatch(format!(
            "nvvm descriptor kernel `{}` declares {} input / {} output / {} accumulation buffers but its buffer roles total {} / {} / {}",
            kernel.entry,
            kernel.input_buffers,
            kernel.output_buffers,
            kernel.accumulation_buffers,
            input_count,
            output_count,
            accumulation_count
        )));
    }
    // Cross-referenced counts (v1 field preserved verbatim): the per-buffer
    // `element_counts` list must agree with the per-buffer contract in
    // length and value.
    if kernel.element_counts.len() != buffers.len() {
        return Err(errors::shape_mismatch(format!(
            "nvvm descriptor kernel `{}` records {} element_counts entries for {} storage buffers",
            kernel.entry,
            kernel.element_counts.len(),
            buffers.len()
        )));
    }
    for (index, (declared, buffer)) in kernel
        .element_counts
        .iter()
        .zip(buffers.iter())
        .enumerate()
    {
        if *declared != buffer.element_count {
            return Err(errors::shape_mismatch(format!(
                "nvvm descriptor kernel `{}` records element_counts[{index}] = {declared} but buffer binding {} declares {}",
                kernel.entry, buffer.binding, buffer.element_count
            )));
        }
    }
    if let Some(first_input) = buffers.iter().find(|buffer| buffer.role.is_input()) {
        if kernel.element_count != first_input.element_count {
            return Err(errors::shape_mismatch(format!(
                "nvvm descriptor kernel `{}` records kernel element_count {} but its first input buffer (binding {}) declares {}",
                kernel.entry,
                kernel.element_count,
                first_input.binding,
                first_input.element_count
            )));
        }
    }

    // Launch geometry: dispatch (grid) + workgroup (block). Every axis must
    // be non-zero and fit `u32` — an out-of-range axis is rejected, never
    // saturated to `u32::MAX`.
    let launch = kernel.launch.as_ref().ok_or_else(|| {
        errors::descriptor(format!(
            "nvvm descriptor kernel `{}` carries no launch geometry",
            kernel.entry
        ))
    })?;
    let grid = [
        launch_axis(launch.dispatch.x, "dispatch x", &kernel.entry)?,
        launch_axis(launch.dispatch.y, "dispatch y", &kernel.entry)?,
        launch_axis(launch.dispatch.z, "dispatch z", &kernel.entry)?,
    ];
    let block = [
        launch_axis(launch.workgroup.x, "workgroup x", &kernel.entry)?,
        launch_axis(launch.workgroup.y, "workgroup y", &kernel.entry)?,
        launch_axis(launch.workgroup.z, "workgroup z", &kernel.entry)?,
    ];

    // Plan identity: closed set only; the plan facts are the dispatch
    // authority and are cross-checked against the buffer contract and the
    // launch geometry (single launch authority).
    let plan_kind = match &kernel.plan {
        Some(plan) => {
            if !KNOWN_PLAN_KINDS.contains(&plan.kind.as_str()) {
                return Err(errors::descriptor(format!(
                    "nvvm descriptor kernel `{}` declares unknown plan kind `{}`",
                    kernel.entry, plan.kind
                )));
            }
            Some(plan.kind.clone())
        }
        None => None,
    };
    if let Some(plan) = &kernel.plan {
        match plan.kind.as_str() {
            "tiled_matmul" => {
                validate_tiled_matmul_plan(plan, &buffers, &block, &kernel.entry)?
            }
            "tree_reduction" => {
                validate_tree_reduction_plan(plan, &buffers, &block, &kernel.entry)?
            }
            _ => {}
        }
    }

    Ok(NvvmLaunchPlan {
        entry: kernel.entry.clone(),
        element_ty,
        buffers,
        grid,
        block,
        plan_kind,
    })
}

/// One launch axis, fail-closed on zero and on values that do not fit `u32`.
fn launch_axis(value: u64, axis: &str, entry: &str) -> HostResult<u32> {
    if value == 0 {
        return Err(errors::descriptor(format!(
            "nvvm descriptor kernel `{entry}` has a zero {axis} launch axis"
        )));
    }
    u32::try_from(value).map_err(|_| {
        errors::descriptor(format!(
            "nvvm descriptor kernel `{entry}` {axis} launch axis {value} does not fit u32 (rejected, not saturated)"
        ))
    })
}

/// Cross-check the `tiled_matmul` plan facts against the buffer contract and
/// the launch geometry: positions 0/1/2 must be the `input` `M·K`,
/// `extra-input` `K·N`, and `output` `M·N` buffers, and the plan's workgroup
/// facts must agree with the launch workgroup (single launch authority).
fn validate_tiled_matmul_plan(
    plan: &NvvmPlanJson,
    buffers: &[NvvmLaunchBuffer],
    block: &[u32; 3],
    entry: &str,
) -> HostResult<()> {
    let m = plan.m.ok_or_else(|| {
        errors::descriptor(format!(
            "nvvm descriptor plan tiled_matmul (kernel `{entry}`) is missing m"
        ))
    })?;
    let k = plan.k.ok_or_else(|| {
        errors::descriptor(format!(
            "nvvm descriptor plan tiled_matmul (kernel `{entry}`) is missing k"
        ))
    })?;
    let n = plan.n.ok_or_else(|| {
        errors::descriptor(format!(
            "nvvm descriptor plan tiled_matmul (kernel `{entry}`) is missing n"
        ))
    })?;
    let (mk, kn, mn) = (
        m.checked_mul(k).ok_or_else(|| {
            errors::descriptor(format!(
                "nvvm descriptor plan tiled_matmul (kernel `{entry}`) M·K overflows"
            ))
        })?,
        k.checked_mul(n).ok_or_else(|| {
            errors::descriptor(format!(
                "nvvm descriptor plan tiled_matmul (kernel `{entry}`) K·N overflows"
            ))
        })?,
        m.checked_mul(n).ok_or_else(|| {
            errors::descriptor(format!(
                "nvvm descriptor plan tiled_matmul (kernel `{entry}`) M·N overflows"
            ))
        })?,
    );
    if buffers.len() < 3 {
        return Err(errors::abi_mismatch(format!(
            "nvvm descriptor plan tiled_matmul (kernel `{entry}`) requires the M·K/K·N/M·N buffer contract, but the kernel binds {} buffers",
            buffers.len()
        )));
    }
    plan_buffer_position(buffers, 0, AdapterBufferRole::Input, mk, "tiled_matmul", entry)?;
    plan_buffer_position(buffers, 1, AdapterBufferRole::ExtraInput, kn, "tiled_matmul", entry)?;
    plan_buffer_position(buffers, 2, AdapterBufferRole::Output, mn, "tiled_matmul", entry)?;
    plan_workgroup_consistency(plan, block, entry)
}

/// Cross-check the `tree_reduction` plan facts against the buffer contract:
/// positions 0/1 must be the `input` `length` and `output` `partials`
/// buffers, with a known combination operator.
fn validate_tree_reduction_plan(
    plan: &NvvmPlanJson,
    buffers: &[NvvmLaunchBuffer],
    block: &[u32; 3],
    entry: &str,
) -> HostResult<()> {
    let length = plan.length.ok_or_else(|| {
        errors::descriptor(format!(
            "nvvm descriptor plan tree_reduction (kernel `{entry}`) is missing length"
        ))
    })?;
    let partials = plan.partials.ok_or_else(|| {
        errors::descriptor(format!(
            "nvvm descriptor plan tree_reduction (kernel `{entry}`) is missing partials"
        ))
    })?;
    match plan.op.as_deref() {
        Some("sum") | Some("mean") => {}
        Some(op) => {
            return Err(errors::descriptor(format!(
                "nvvm descriptor plan tree_reduction (kernel `{entry}`) declares unknown op `{op}`"
            )))
        }
        None => {
            return Err(errors::descriptor(format!(
                "nvvm descriptor plan tree_reduction (kernel `{entry}`) is missing op"
            )))
        }
    }
    if buffers.len() < 2 {
        return Err(errors::abi_mismatch(format!(
            "nvvm descriptor plan tree_reduction (kernel `{entry}`) requires the length/partials buffer contract, but the kernel binds {} buffers",
            buffers.len()
        )));
    }
    plan_buffer_position(buffers, 0, AdapterBufferRole::Input, length, "tree_reduction", entry)?;
    plan_buffer_position(buffers, 1, AdapterBufferRole::Output, partials, "tree_reduction", entry)?;
    plan_workgroup_consistency(plan, block, entry)
}

/// One position of a plan-shaped buffer contract: the role must match and the
/// flat count must equal the plan's expected product.
fn plan_buffer_position(
    buffers: &[NvvmLaunchBuffer],
    position: usize,
    expected_role: AdapterBufferRole,
    expected_count: u64,
    kind: &str,
    entry: &str,
) -> HostResult<()> {
    let buffer = &buffers[position];
    if buffer.role != expected_role {
        return Err(errors::abi_mismatch(format!(
            "nvvm descriptor plan {kind} (kernel `{entry}`) expects buffer position {position} to be `{}` but it is `{}`",
            expected_role.tag(),
            buffer.role.tag()
        )));
    }
    if buffer.element_count != expected_count {
        return Err(errors::shape_mismatch(format!(
            "nvvm descriptor plan {kind} (kernel `{entry}`) buffer position {position} declares {} elements but the plan expects {}",
            buffer.element_count, expected_count
        )));
    }
    Ok(())
}

/// Single launch authority: the plan's workgroup facts must agree with the
/// launch geometry the adapter dispatches from.
fn plan_workgroup_consistency(
    plan: &NvvmPlanJson,
    block: &[u32; 3],
    entry: &str,
) -> HostResult<()> {
    if let Some(workgroup_x) = plan.workgroup_x {
        if workgroup_x != block[0] {
            return Err(errors::descriptor(format!(
                "nvvm descriptor kernel `{entry}` plan workgroup_x {workgroup_x} contradicts launch workgroup x {} (single launch authority)",
                block[0]
            )));
        }
    }
    if let Some(workgroup_y) = plan.workgroup_y {
        if workgroup_y != block[1] {
            return Err(errors::descriptor(format!(
                "nvvm descriptor kernel `{entry}` plan workgroup_y {workgroup_y} contradicts launch workgroup y {} (single launch authority)",
                block[1]
            )));
        }
    }
    Ok(())
}

/// Launch one validated plan on a session with the given PTX bytes and host
/// inputs, then sync and read back. `inputs` maps input-buffer bindings to
/// host f32 values; every `input` / `extra-input` buffer must have an entry
/// whose length equals its declared element count (`E_DEVICE_SHAPE_MISMATCH`
/// otherwise). An optional [`NumericOracle`] is checked against the first
/// output row.
///
/// The adapter releases every handle it allocates on success AND on error, so
/// a failed launch never leaks at the driver boundary.
///
/// # Errors
/// - `E_DEVICE_DTYPE_MISMATCH` — the kernel is not f32 (the session transfer
///   surface is f32-only);
/// - `E_DEVICE_DESCRIPTOR` — empty PTX, or a buffer whose byte length does
///   not fit the host address space;
/// - `E_DEVICE_SHAPE_MISMATCH` — a declared input is missing or its length
///   contradicts its element count;
/// - `E_DEVICE_ENTRY_MISMATCH` / session-level failures bubble through
///   unchanged.
pub fn execute_launch_plan(
    session: &mut CudaHostSession,
    ptx: &[u8],
    plan: &NvvmLaunchPlan,
    inputs: &BTreeMap<u32, Vec<f32>>,
    oracle: Option<&NumericOracle>,
) -> HostResult<AdapterLaunchReceipt> {
    if plan.element_ty != NvvmElementType::F32 {
        return Err(errors::dtype_mismatch(format!(
            "host launch adapter transfers are f32-only; descriptor kernel `{}` declares `{}`",
            plan.entry,
            plan.element_ty.spelling()
        )));
    }
    if ptx.is_empty() {
        return Err(errors::descriptor(
            "host launch adapter received empty PTX bytes",
        ));
    }
    let mut handles: Vec<CudaHandleId> = Vec::with_capacity(plan.buffers.len());
    let mut module: Option<CudaHandleId> = None;
    // Error-path teardown (S2-3 posture): a failure at any stage runs the
    // ordered release below before the error escapes.
    let outcome = (|| -> HostResult<AdapterLaunchReceipt> {
        let module_handle = session.load_module(ptx)?;
        module = Some(module_handle);
        for buffer in &plan.buffers {
            let byte_length = buffer_byte_length(buffer, plan.element_ty, &plan.entry)?;
            handles.push(session.alloc_bytes(byte_length)?);
        }
        let mut copy_ins = 0usize;
        let mut zero_fills = 0usize;
        for (buffer, handle) in plan.buffers.iter().zip(handles.iter()) {
            match buffer.role {
                AdapterBufferRole::Input | AdapterBufferRole::ExtraInput => {
                    let values = inputs.get(&buffer.binding).ok_or_else(|| {
                        errors::shape_mismatch(format!(
                            "no host input provided for descriptor kernel `{}` buffer `{}` (binding {})",
                            plan.entry,
                            buffer.role.tag(),
                            buffer.binding
                        ))
                    })?;
                    if u64::try_from(values.len()).ok() != Some(buffer.element_count) {
                        return Err(errors::shape_mismatch(format!(
                            "input for descriptor kernel `{}` buffer `{}` (binding {}) has {} f32 elements but the descriptor declares {}",
                            plan.entry,
                            buffer.role.tag(),
                            buffer.binding,
                            values.len(),
                            buffer.element_count
                        )));
                    }
                    session.copy_in_f32(*handle, values)?;
                    copy_ins += 1;
                }
                AdapterBufferRole::Accumulation => {
                    // Zero-fill accumulation storage at allocation (the host
                    // ZeroFill convention; the v2 sidecar carries no
                    // initialization axis).
                    let element_count = usize::try_from(buffer.element_count).map_err(|_| {
                        errors::descriptor(format!(
                            "descriptor kernel `{}` accumulation buffer binding {} element count does not fit host usize",
                            plan.entry, buffer.binding
                        ))
                    })?;
                    let zeros = vec![0.0f32; element_count];
                    session.copy_in_f32(*handle, &zeros)?;
                    zero_fills += 1;
                }
                AdapterBufferRole::Output | AdapterBufferRole::ExtraOutput => {}
            }
        }
        session.launch_kernel_3d(
            module_handle,
            &plan.entry,
            handles.as_slice(),
            plan.grid[0],
            plan.grid[1],
            plan.grid[2],
            plan.block[0],
            plan.block[1],
            plan.block[2],
        )?;
        session.sync()?;
        let mut readbacks = 0usize;
        let mut outputs: BTreeMap<u32, Vec<f32>> = BTreeMap::new();
        for (buffer, handle) in plan.buffers.iter().zip(handles.iter()) {
            if buffer.role.is_output() {
                outputs.insert(buffer.binding, session.readback_f32(*handle)?);
                readbacks += 1;
            }
        }
        let oracle_check = oracle.map(|expected| {
            match outputs.values().next() {
                Some(values) => {
                    let (matched, max_abs_delta) = expected.matches(values);
                    OracleCheck {
                        matched,
                        max_abs_delta,
                    }
                }
                None => OracleCheck {
                    matched: false,
                    max_abs_delta: f64::INFINITY,
                },
            }
        });
        Ok(AdapterLaunchReceipt {
            entry: plan.entry.clone(),
            module_hash: crate::device_descriptor::fnv1a64(ptx),
            launches: 1,
            allocated_buffers: plan.buffers.len(),
            copy_ins,
            zero_fills,
            readbacks,
            releases: 0,
            outputs,
            oracle: oracle_check,
        })
    })();

    // Teardown: release every allocated buffer and the module, success or
    // failure, before the receipt (or the error) escapes.
    let mut releases = 0usize;
    for handle in &handles {
        if session.release(*handle).is_ok() {
            releases += 1;
        }
    }
    if let Some(module_handle) = module {
        if session.release(module_handle).is_ok() {
            releases += 1;
        }
    }
    match outcome {
        Ok(mut receipt) => {
            receipt.releases = releases;
            Ok(receipt)
        }
        Err(error) => Err(error),
    }
}

/// Parse a v2 descriptor sidecar and launch it in one call: parse + validate
/// fail-closed, then load / alloc / copy / launch / sync / readback. See
/// [`parse_descriptor`] and [`execute_launch_plan`].
///
/// # Errors
/// The union of [`parse_descriptor`] and [`execute_launch_plan`] errors.
pub fn launch_descriptor(
    session: &mut CudaHostSession,
    descriptor_json: &[u8],
    ptx: &[u8],
    inputs: &BTreeMap<u32, Vec<f32>>,
    oracle: Option<&NumericOracle>,
) -> HostResult<AdapterLaunchReceipt> {
    let plan = parse_descriptor(descriptor_json)?;
    execute_launch_plan(session, ptx, &plan, inputs, oracle)
}

/// Byte length one buffer must be allocated with, sized from the descriptor
/// element count and the kernel's byte width (never from re-derived text).
fn buffer_byte_length(
    buffer: &NvvmLaunchBuffer,
    element_ty: NvvmElementType,
    entry: &str,
) -> HostResult<usize> {
    let bytes = buffer
        .element_count
        .checked_mul(u64::from(element_ty.byte_width()))
        .ok_or_else(|| {
            errors::descriptor(format!(
                "descriptor kernel `{entry}` buffer `{}` (binding {}) byte length overflows u64",
                buffer.role.tag(),
                buffer.binding
            ))
        })?;
    usize::try_from(bytes).map_err(|_| {
        errors::descriptor(format!(
            "descriptor kernel `{entry}` buffer `{}` (binding {}) byte length {bytes} does not fit host usize",
            buffer.role.tag(),
            buffer.binding
        ))
    })
}
