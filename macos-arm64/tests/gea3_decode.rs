//! GEA3-U5a: hosts fake ladder for the landed full-model plans.
//!
//! This test owns the hosts-side half of the GEA3 transport boundary.  It
//! mirror-parses both exported programs, validates the carried full-model and
//! KV facts, maps each program onto the target-neutral `DeviceDescriptor`, and
//! runs the ordered launch graph through the sequencing fake.  The fake never
//! reads tensor values and never stands in for a Metal kernel body.

#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use faber_host_macos_arm64::composite_host::{CompositeHost, CompositeHostConfig, DeviceSelection};
use faber_host_macos_arm64::device_descriptor::{
    sha256_hex, DescriptorBuffer, DescriptorBufferVersion, DescriptorDataFlow,
    DescriptorEndOfRunResult, DescriptorKernel, DescriptorLaunch, DescriptorResult,
    DescriptorRuntimeSource, DeviceBufferInitialization, DeviceBufferLifetime, DeviceBufferRole,
    DeviceDataType, DeviceDescriptor, DeviceProgramLifetime,
};
use faber_host_macos_arm64::device_host::{DeviceLaunchBinding, DeviceRuntime, DeviceSession};
use faber_host_macos_arm64::metal_host::MappedWeightFile;
use faber_host_macos_arm64::{enumerate_metal_physical_devices, FakeMetalDriver, MetalHostSession};
use host_coordinator::{DeviceBackend, DeviceHandle};
use serde::Deserialize;
use serde_json::{json, Value};

const PLAN_SCHEMA: &str = "gea3-program-plan-v1";
// GEA3-WIRE-BUFFER-V2 (Amendment 3 / GEA3-A1): the named successor of
// GEA3-WIRE-BUFFER-V1, mirrored from the schema authority
// `radix/crates/radix-mir-fmir/src/schema/wire.rs`
// (`GEA3_WIRE_BUFFER_ABI_V2` + `WireSubWindowProjection`).  The allocation
// law stands — one (buffer id, version), one allocation count — and a read
// or write slot may carry a bounded sub-window of that one allocation.
const WIRE_BUFFER_ABI: &str = "gea3-wire-buffer-v2";
const PLAN_MEMBER: &str = "gea3-program-plan.json";
const MODULE_IMAGE_RULE: &str =
    "module_members are independently selectable; the plan binds them by entry identity";
const SOURCE: &str = "gradus/src/kernel.fab";
const LAYERS: usize = 32;
const BLOCK_LAUNCHES_PER_LAYER: usize = 62;
const LAUNCHES_PER_PROGRAM: usize = LAYERS * BLOCK_LAUNCHES_PER_LAYER + 3;
const DEPENDENCIES_PER_PROGRAM: usize = LAUNCHES_PER_PROGRAM - 1;
const PREFILL_ROWS: u64 = 36;
const HISTORY_CAPACITY: u64 = 76;
const KV_WIDTH: u64 = 320;
const VOCAB: u64 = 49_152;
const HIDDEN: u64 = 960;
const DECODE_STEPS: usize = 8;

// PPB-U7 (delivery row PPB-U7): one compile-time admission identity per
// statue.  The frozen statue stays the byte-stable regression row; the soak
// statue `metal-m5max-soak-l2000` is a separate compile-time fact — 2000-row
// fixed capacity, 1900 decode steps, margin 64 — never a runtime capacity
// argument, never a session lifetime, and `DeviceProgramLifetime::SingleRun`
// is retained for both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Gea3Identity {
    name: &'static str,
    n_predict: usize,
    margin: usize,
    history_capacity: u64,
    decode_steps: usize,
}

impl Gea3Identity {
    /// `l_max = prompt(36) + N + margin`, derived — never hand-copied.
    const fn l_max(self) -> u64 {
        PREFILL_ROWS + self.n_predict as u64 + self.margin as u64
    }

    /// Fixed F32 KV footprint: 32 layers × 2 (K/V) × l_max rows × 320 × 4.
    const fn kv_bytes(self) -> u64 {
        LAYERS as u64 * 2 * self.l_max() * KV_WIDTH * 4
    }

    /// The mechanical derivation the receipt names beside the measured value.
    fn kv_basis(self) -> String {
        format!("32 * 2 * {} * {KV_WIDTH} * sizeof(F32)", self.l_max())
    }
}

const GEA3_FROZEN_SHORT: Gea3Identity = Gea3Identity {
    name: "metal-m5max",
    n_predict: 8,
    margin: 32,
    history_capacity: HISTORY_CAPACITY,
    decode_steps: DECODE_STEPS,
};

const GEA3_SOAK_L2000: Gea3Identity = Gea3Identity {
    name: "metal-m5max-soak-l2000",
    n_predict: 1_900,
    margin: 64,
    history_capacity: 2_000,
    decode_steps: 1_900,
};

// GLP-U1b: the fixed-output-length parity statue is a separate compile-time
// identity — N=1000, margin 64, l_max = 36 + 1000 + 64 = 1100, fixed F32 KV
// footprint 32 × 2 × 1100 × 320 × 4 = 90,112,000 bytes.
const GEA3_FIXED1000: Gea3Identity = Gea3Identity {
    name: "metal-m5max-fixed1000",
    n_predict: 1_000,
    margin: 64,
    history_capacity: 1_100,
    decode_steps: 1_000,
};

// PPB-U3: the optional parity timing companion the physical test emits in
// addition to — never instead of — its `gea3-metal-receipt-v1` receipt.  The
// environment names the output path; an unset variable is the opt-out.
const PARITY_COMPANION_SCHEMA: &str = "gea3-parity-timing-companion-v1";
const PARITY_COMPANION_ENV: &str = "GEA3_PARITY_TIMING_COMPANION";
// PPB-U7: the soak runner rewrites the receipt on this cadence during decode
// so a cap-killed process still leaves its latest measured state on disk
// (KV residency and allocation cost are written before the first step).
const SOAK_RECEIPT_REWRITE_STEPS: usize = 25;

/// PPB-U7: the soak receipt stream.  The runner emits the U6 monotonic
/// produced-token counter as line-delimited stdout progress during decode
/// (one strict `{"produced_tokens": N}` JSON object per line, so a 60s-capped
/// kill still yields the count and the fresh-at-cap evidence) and rewrites
/// the receipt file on a fixed cadence so the kill also leaves the latest
/// measured KV residency, allocation cost, and step state on disk.  The
/// frozen statue passes `None` and its receipt bytes are unchanged.
struct Gea3SoakStream {
    identity: Gea3Identity,
    base_receipt: Value,
    receipt_path: PathBuf,
    companion_path: Option<PathBuf>,
}

impl Gea3SoakStream {
    fn emit_progress(&self, produced_tokens: u64) {
        // One strict JSON object per line; Rust's Stdout is line-buffered,
        // so each line reaches the polled file before the next step starts.
        println!("{{\"produced_tokens\": {produced_tokens}}}");
    }

    /// Write the latest measured state.  No fsync is needed: the arm is
    /// killed by a signal, not a power loss, and the page cache preserves
    /// the last complete rewrite.
    fn write_partial(&self, evidence: &SoakPartialEvidence) {
        let mut receipt = self.base_receipt.clone();
        receipt["status"] = json!("partial_streamed");
        receipt["residency"] = evidence.residency.clone();
        receipt["execution"] = evidence.execution.clone();
        receipt["steps"] = evidence.steps.clone();
        receipt["launch_plans"] = evidence.launch_plans.clone();
        receipt["throughput"] = evidence.throughput.clone();
        receipt["partial_stream"] = json!({
            "law": "the hard cap may kill this run before n_predict; this file is the last cadence rewrite, not a completion claim",
            "decode_steps_observed": evidence.decode_steps,
            "produced_tokens": evidence.produced_tokens,
        });
        if let Some(parent) = self.receipt_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Ok(bytes) = serde_json::to_vec_pretty(&receipt) {
            let _ = fs::write(&self.receipt_path, bytes);
        }
    }
}

/// The measured-so-far cells a streamed soak receipt carries.
struct SoakPartialEvidence {
    residency: Value,
    execution: Value,
    steps: Value,
    launch_plans: Value,
    throughput: Value,
    decode_steps: usize,
    produced_tokens: usize,
}

// ---------------------------------------------------------------------------
// Hosts' typed serde mirror of the GEA3 transport envelope.  This is a
// consumer-owned schema.  Unknown and missing fields fail the decode rather
// than silently dropping a producer fact.
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3ProgramPlanEnvelope {
    schema: String,
    wire_buffer_abi: String,
    source: String,
    programs: Gea3Programs,
    module_members: Vec<String>,
    module_image_rule: String,
    instance_expansion: Gea3InstanceExpansion,
    kv_geometry: Gea3KvGeometry,
    declared_outputs: Gea3DeclaredOutputs,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3Programs {
    prefill: Gea3Program,
    decode_step: Gea3Program,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3InstanceExpansion {
    layers: usize,
    block_launches_per_layer: usize,
    attention_heads: usize,
    kv_heads: usize,
    per_layer_weight_identities: bool,
    per_layer_intermediates: bool,
    per_layer_kv_buffers: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3KvGeometry {
    capacity: u64,
    declared_history_length: u64,
    dtype: String,
    mask_beyond_length: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3DeclaredOutputs {
    prefill_logits: Vec<u64>,
    decode_logits: Vec<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3Program {
    program: String,
    kernels: Vec<Gea3KernelUnit>,
    launches: Vec<Gea3LaunchUnit>,
    lifetime: Gea3ProgramLifetime,
    results: Vec<Gea3ResultBuffer>,
    allocations: Vec<Gea3Allocation>,
    semantic_values: Vec<Gea3SemanticValue>,
    roots: Vec<u32>,
    dependencies: Vec<Gea3DependencyEdge>,
    relations: Vec<Gea3Relation>,
    state_buffers: Vec<Gea3StateBuffer>,
    declared_history_length: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3KernelUnit {
    function: u32,
    entry: String,
    plan: Gea3Plan,
    resources: Vec<Gea3DeviceResource>,
    launch: Gea3KernelLaunchPlan,
    layer: usize,
    ordinal: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
enum Gea3Plan {
    Elementwise,
    TiledMatMul(Gea3MatMulPlan),
    ComposedMatMul(Gea3ComposedMatMulPlan),
    Transpose(Gea3TransposePlan),
    RmsNormalization(Gea3RmsNormalizationPlan),
    Rope(Gea3RopePlan),
    CausalMaskedSoftmax(Gea3CausalMaskedSoftmaxPlan),
    Gather(Gea3GatherPlan),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3ComposedMatMulPlan {
    stages: Vec<Value>,
    edges: Vec<Value>,
    handoff: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3MatMulPlan {
    m: u64,
    k: u64,
    n: u64,
    left_layout: Gea3MatMulLayout,
    right_layout: Gea3MatMulLayout,
    right_operand_layout: Gea3RightOperandLayout,
    tile: u32,
    workgroup_x: u32,
    workgroup_y: u32,
    shared_memory: Gea3MatMulSharedMemory,
    barriers: Vec<Gea3BarrierPoint>,
    oob_padding: Gea3OobPaddingPolicy,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Gea3RightOperandLayout {
    OutIn,
    RowMajor,
}

#[derive(Debug, Clone, Copy, Deserialize)]
enum Gea3MatMulLayout {
    F32,
    Bf16,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3MatMulSharedMemory {
    shared_a: Gea3SharedMemoryLayout,
    shared_b: Gea3SharedMemoryLayout,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3SharedMemoryLayout {
    element_byte_width: u32,
    slot_count: u32,
    buffer_name: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3BarrierPoint {
    after: Gea3BarrierPhase,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Gea3BarrierPhase {
    SharedMemoryLoad,
    ReductionStep,
    InnerProductStep,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Gea3OobPaddingPolicy {
    ZeroFill,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3TransposePlan {
    m: u64,
    n: u64,
    workgroup_x: u32,
    dispatch_x: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3RmsNormalizationPlan {
    axis: u64,
    epsilon_bits: u32,
    width: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3RopePlan {
    pos: u64,
    dim: u64,
    width: u64,
    per_row: bool,
    rows: u64,
    rotate_half: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3CausalMaskedSoftmaxPlan {
    rows: u64,
    cols: u64,
}

/// PGC-R1/PGC-R3: the row-gather plan mirror — one indexed row copy per
/// token id, superseding the one-hot selector matmul. The wire resource
/// stamps `element_ty: "f32"` for the ids buffer (the hosts GEA3 statue is
/// F32-only; 4 bytes per id) while the emitted MSL truth is
/// `device const uint* ids`, pinned by `gea3_decode_pgc_r1.rs`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3GatherPlan {
    id_count: u64,
    table_cols: u64,
    table_rows: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3DeviceResource {
    buffer: Gea3BufferIdentity,
    version: Gea3BufferVersion,
    binding: Gea3Binding,
    access: Gea3ResourceAccess,
    generation: u32,
    initialization: Gea3Initialization,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3BufferIdentity {
    id: u32,
    name: String,
    role: Gea3BufferRole,
    storage: Gea3StorageLayout,
    lifetime: Gea3BufferLifetime,
    semantic_value: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3BufferVersion {
    version: u32,
    element_ty: String,
    element_count: u64,
    reduced_projection: Option<Gea3ReducedProjection>,
    #[serde(default)]
    sub_window: Option<Gea3SubWindowProjection>,
    /// GEA3-U6 num-10: the chunked-geometry declaration — a head-major
    /// producer read whose logical rows are discontiguous `block`-wide
    /// chunks (the prefill attention concat the o-projection consumes).
    #[serde(default)]
    chunked_window: Option<Gea3ChunkedWindow>,
}

/// The hosts' typed mirror of the GEA3-U6 num-10 chunked-geometry
/// declaration: `row_count` rows of `block` contiguous elements per
/// chunk, chunks stacked head-major at `row_count · block` pitch.
/// Admission mirrors the emitter's operand-pitch law fail-closed:
/// read-only, never beside a sub-window projection, and the allocation
/// must be a whole number of chunk cells (a flat read of a chunked truth
/// is the same defect as a strided write of a flat truth).
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3ChunkedWindow {
    block: u64,
    row_count: u64,
}

/// The hosts' typed mirror of the GEA3-WIRE-BUFFER-V2 sub-window
/// projection (Amendment 3 / GEA3-A1): `row_count` rows of `row_width`
/// contiguous elements at `row_stride` producer pitch, starting at
/// `element_offset`, with the carried derived element count beside the
/// geometry so admission can cross-check the two truths fail-closed.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3SubWindowProjection {
    buffer: u32,
    element_offset: u64,
    row_width: u64,
    row_count: u64,
    row_stride: u64,
    derived_element_count: u64,
}

impl Gea3SubWindowProjection {
    fn window_element_count(&self) -> Option<u64> {
        self.row_width.checked_mul(self.row_count)
    }

    fn covering_span(&self) -> Option<u64> {
        self.row_count
            .checked_sub(1)
            .and_then(|rows| rows.checked_mul(self.row_stride))
            .and_then(|span| span.checked_add(self.row_width))
    }

    fn is_contiguous(&self) -> bool {
        self.row_stride == self.row_width
    }

    /// The byte offset and view span of the window's device binding: the
    /// covering span is exact for a contiguous window and bounds a strided
    /// read-side window (never out of allocation).
    fn byte_binding(&self) -> Option<(u64, u64)> {
        let span = self.covering_span()?;
        let offset_bytes = self.element_offset.checked_mul(4)?;
        let span_bytes = span.checked_mul(4)?;
        Some((offset_bytes, span_bytes))
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3ReducedProjection {
    axis_extent: u64,
    inner_stride: u64,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3Binding {
    group: u32,
    binding: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3KernelLaunchPlan {
    workgroup: Gea3WorkgroupSize,
    dispatch_size: Gea3DispatchSize,
    workgroup_count: Gea3WorkgroupCount,
    declared_input_count: usize,
    declared_output_count: u64,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3WorkgroupSize {
    x: u32,
    y: u32,
    z: u32,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3DispatchSize {
    x: u64,
    y: u64,
    z: u64,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3WorkgroupCount {
    x: u64,
    y: u64,
    z: u64,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3LaunchUnit {
    id: u32,
    kernel_index: u32,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3DependencyEdge {
    producer: u32,
    consumer: u32,
    buffer: u32,
    version: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3ResultBuffer {
    buffer: Gea3BufferIdentity,
    version: Gea3BufferVersion,
    role: Gea3ResultRole,
    produced_by: u32,
    observation: Gea3ObservationFact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Gea3ResultRole {
    Output,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3ObservationFact {
    at_launch: u32,
    cadence: Gea3ObservationCadence,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Gea3ObservationCadence {
    PerStep,
    EndOfRun,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3Allocation {
    id: u32,
    name: String,
    role: Gea3BufferRole,
    storage: Gea3StorageLayout,
    lifetime: Gea3BufferLifetime,
    element_count: u64,
    dtype: String,
    initialization: Gea3Initialization,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3SemanticValue {
    id: u32,
    name: String,
    origin: Gea3SemanticValueOrigin,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
enum Gea3SemanticValueOrigin {
    Synthetic { label: String },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3Relation {}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3StateBuffer {
    #[serde(default)]
    id: Option<u32>,
    #[serde(default)]
    name: Option<String>,
    layer: usize,
    #[serde(default)]
    role: Option<Gea3BufferRole>,
    #[serde(default)]
    dtype: Option<String>,
    #[serde(default)]
    shape: Option<Vec<u64>>,
    #[serde(default)]
    history_capacity: Option<u64>,
    #[serde(default)]
    declared_history_length: Option<u64>,
    #[serde(default)]
    producer_launch: Option<u32>,
    #[serde(default)]
    entry: Option<String>,
    #[serde(default)]
    cache_buffer: Option<u32>,
    #[serde(default)]
    history_length: Option<u64>,
    #[serde(default)]
    row_shape: Option<Vec<u64>>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Gea3ProgramLifetime {
    SingleRun,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
enum Gea3BufferRole {
    Input,
    Output,
    InOut,
}

#[derive(Debug, Clone, Copy, Deserialize)]
enum Gea3StorageLayout {
    #[serde(rename = "device-handle")]
    DeviceHandle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
enum Gea3BufferLifetime {
    PerProgram,
    PerStep,
    ObservationPoint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Gea3ResourceAccess {
    #[serde(alias = "Read")]
    Read,
    #[serde(alias = "Write")]
    Write,
    #[serde(alias = "ReadWrite", alias = "read-write")]
    ReadWrite,
}

#[derive(Debug, Clone, Copy, Deserialize)]
enum Gea3Initialization {
    ZeroFill,
    HostProvided,
    KernelInitialized,
}

// ---------------------------------------------------------------------------
// Envelope admission and DeviceDescriptor mapping.
// ---------------------------------------------------------------------------

fn gea3_artifact_dir() -> PathBuf {
    let root = std::env::var_os("GEA3_ARTIFACT_DIR")
        .map(PathBuf::from)
        .expect("GEA3_ARTIFACT_DIR must identify the exported GEA3 bundle");
    assert!(
        root.is_dir(),
        "missing GEA3 artifact directory {}",
        root.display()
    );
    root
}

fn load_gea3_plan(artifact_dir: &Path) -> Gea3ProgramPlanEnvelope {
    let path = artifact_dir.join(PLAN_MEMBER);
    let bytes = fs::read(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|error| panic!("mirror-parse {}: {error}", path.display()))
}

/// The per-kernel window bindings derived from the carried sub-window
/// projections: (kernel index, binding index) → (byte offset, view span).
/// A slot without an entry binds its whole handle.
type Gea3WindowBindings = BTreeMap<(u32, u32), (u64, u64)>;

/// Admit one carried sub-window against the allocation it nests in (the
/// two-truths guard mirrored from the schema authority
/// `validate_carried_sub_window_projection`).  Returns the window's byte
/// binding when the resource carries one.
/// Admit one carried chunked-geometry declaration against the allocation
/// it reads (the GEA3-U6 num-10 mirror of the emitter's admission law):
/// read-only, never beside a sub-window projection, never degenerate, and
/// the allocation must partition into whole `block · row_count` chunk
/// cells with the carried count spanning every cell.  A flat read of a
/// head-major truth is not admissible — the consumer must declare the
/// geometry it addresses through.
fn admit_chunked_window(
    resource: &Gea3DeviceResource,
    chunked: Gea3ChunkedWindow,
    allocation: u64,
) -> Result<(), String> {
    if matches!(resource.access, Gea3ResourceAccess::Write) {
        return Err(format!(
            "resource `{}` carries a chunked window on a write slot; chunked reads are read-side operand-pitch facts only (GEA3-U6 num-10)",
            resource.buffer.name
        ));
    }
    if resource.version.sub_window.is_some() {
        return Err(format!(
            "resource `{}` carries both a sub-window and a chunked geometry; one binding carries one read geometry (GEA3-U6 num-10)",
            resource.buffer.name
        ));
    }
    let chunk_cells = chunked
        .block
        .checked_mul(chunked.row_count)
        .ok_or_else(|| {
            format!(
                "resource `{}` carries a chunked window whose block · rows saturates u64",
                resource.buffer.name
            )
        })?;
    if chunked.block == 0 || chunked.row_count == 0 || resource.version.element_count == 0 {
        return Err(format!(
            "resource `{}` carries a degenerate chunked window (block {}, rows {})",
            resource.buffer.name, chunked.block, chunked.row_count
        ));
    }
    if resource.version.element_count != allocation || allocation % chunk_cells != 0 {
        return Err(format!(
            "resource `{}` carries a chunked window (block {} · rows {} = {chunk_cells}) that does not partition its {allocation}-element allocation; the declared geometry must equal the producer's head-major truth (GEA3-U6 num-10)",
            resource.buffer.name, chunked.block, chunked.row_count
        ));
    }
    Ok(())
}

fn admit_sub_window(
    resource: &Gea3DeviceResource,
    allocation: u64,
) -> Result<Option<(u64, u64)>, String> {
    let Some(window) = resource.version.sub_window else {
        return Ok(None);
    };
    if window.buffer != resource.buffer.id {
        return Err(format!(
            "resource `{}` binds buffer {} but its sub-window names buffer {}; the window addresses its own producer (GEA3-WIRE-BUFFER-V2)",
            resource.buffer.name, resource.buffer.id, window.buffer
        ));
    }
    if window.row_width == 0 || window.row_count == 0 || window.row_stride < window.row_width {
        return Err(format!(
            "resource `{}` carries a sub-window with degenerate geometry (width {}, rows {}, stride {})",
            resource.buffer.name, window.row_width, window.row_count, window.row_stride
        ));
    }
    let derived = window.window_element_count().ok_or_else(|| {
        format!(
            "resource `{}` carries a sub-window whose width · rows saturates u64",
            resource.buffer.name
        )
    })?;
    if window.derived_element_count != derived || resource.version.element_count != derived {
        return Err(format!(
            "resource `{}` binds {} elements with a sub-window deriving {} (width {} · rows {}); the carried and derived truths must agree (GEA3-WIRE-BUFFER-V2)",
            resource.buffer.name,
            resource.version.element_count,
            window.derived_element_count,
            window.row_width,
            window.row_count
        ));
    }
    if matches!(resource.access, Gea3ResourceAccess::ReadWrite) {
        return Err(format!(
            "resource `{}` carries a sub-window on a read-write slot; windows are read- or write-side facts only (GEA3-WIRE-BUFFER-V2)",
            resource.buffer.name
        ));
    }
    if matches!(resource.access, Gea3ResourceAccess::Write) && !window.is_contiguous() {
        return Err(format!(
            "resource `{}` carries a strided sub-window on a write slot; a scattered write is not one binding (GEA3-WIRE-BUFFER-V2)",
            resource.buffer.name
        ));
    }
    // GEA3-U6 num-3 / PGC-B1 pitch-truth law (the hosts mirror of the
    // harness admission, B1 fold-state repair): a STRIDED read window's
    // declared row pitch must equal the producer's actual pitch — but a B1
    // window may be a bounded prefix of the larger fixed-capacity producer
    // (the kv arena's 1088-row bucket inside the 1100-row capacity
    // allocation), so `row_stride · row_count` need only nest inside a
    // whole-row allocation at the KV row width; a lying pitch remains
    // rejected even when count and nesting laws pass.  Contiguous windows
    // (stride == width) are dense by construction and skip the law.
    if !matches!(resource.access, Gea3ResourceAccess::Write) && !window.is_contiguous() {
        let pitch_span = window
            .row_stride
            .checked_mul(window.row_count)
            .ok_or_else(|| {
                format!(
                    "resource `{}` carries a strided read window whose pitch·rows saturates u64",
                    resource.buffer.name
                )
            })?;
        let bounded_prefix = pitch_span < allocation;
        if allocation % window.row_stride != 0
            || pitch_span > allocation
            || (bounded_prefix && window.row_stride != KV_WIDTH)
        {
            return Err(format!(
                "resource `{}` carries a strided read window declaring row_stride {} · row_count {} = {pitch_span} beyond or incompatible with the producer's {allocation}-element allocation; the declared stride must equal the producer's actual pitch (GEA3-U6 num-3 pitch-truth law)",
                resource.buffer.name, window.row_stride, window.row_count
            ));
        }
    }
    let span = window.covering_span().ok_or_else(|| {
        format!(
            "resource `{}` carries a sub-window whose covering span saturates u64",
            resource.buffer.name
        )
    })?;
    if window
        .element_offset
        .checked_add(span)
        .is_none_or(|end| end > allocation)
    {
        return Err(format!(
            "resource `{}` carries a sub-window spanning {} of {allocation} elements from offset {}; the window must nest inside its producer buffer (GEA3-WIRE-BUFFER-V2)",
            resource.buffer.name, span, window.element_offset
        ));
    }
    window.byte_binding().map(Some).ok_or_else(|| {
        format!(
            "resource `{}` carries a sub-window whose byte binding overflows",
            resource.buffer.name
        )
    })
}

fn map_envelope_to_descriptor(
    envelope: &Gea3ProgramPlanEnvelope,
    program: &Gea3Program,
    program_name: &str,
    artifact_dir: &Path,
    identity: Gea3Identity,
) -> Result<(DeviceDescriptor, Gea3WindowBindings), String> {
    admit_envelope(envelope, identity)?;
    admit_program(envelope, program, program_name, identity)?;

    let mut module_image = Vec::new();
    for member in &envelope.module_members {
        if !member.ends_with(".metal") || Path::new(member).components().count() != 1 {
            return Err(format!("invalid Metal module member `{member}`"));
        }
        let path = artifact_dir.join(member);
        let bytes = fs::read(&path)
            .map_err(|error| format!("module member `{}` is missing: {error}", path.display()))?;
        if bytes.is_empty() {
            return Err(format!("module member `{member}` is empty"));
        }
        module_image.extend_from_slice(&bytes);
    }

    // GEA3-A1 (GEA3-WIRE-BUFFER-V2): full-width bindings establish the
    // allocation law — one (buffer id, version), one allocation count;
    // sub-window bindings never mint a second count.  The windows are
    // validated after every full-width declaration is known (a window may
    // precede the full-width binding that bounds it — the attention-output
    // concat's 15 window writes precede its full read).
    let mut shapes: BTreeMap<(u32, u32), (DeviceDataType, u64)> = BTreeMap::new();
    let mut identities: BTreeMap<u32, (String, u32, DeviceBufferRole, DeviceBufferLifetime)> =
        BTreeMap::new();
    for kernel in &program.kernels {
        for resource in &kernel.resources {
            if resource.generation != 1 {
                return Err(format!(
                    "resource `{}` has unsupported generation {}",
                    resource.buffer.name, resource.generation
                ));
            }
            if resource.version.reduced_projection.is_some() {
                return Err(format!(
                    "resource `{}` carries an undeclared projection",
                    resource.buffer.name
                ));
            }
            if resource.binding.group != 0 {
                return Err(format!(
                    "resource `{}` binds unsupported group {}",
                    resource.buffer.name, resource.binding.group
                ));
            }
            let dtype =
                DeviceDataType::from_spelling(&resource.version.element_ty).ok_or_else(|| {
                    format!(
                        "resource `{}` has unsupported dtype `{}`",
                        resource.buffer.name, resource.version.element_ty
                    )
                })?;
            if dtype != DeviceDataType::F32 {
                return Err(format!("resource `{}` is not F32", resource.buffer.name));
            }
            let key = (resource.buffer.id, resource.version.version);
            if resource.version.sub_window.is_none() {
                if let Some((previous, count)) = shapes.get(&key) {
                    if *previous != dtype {
                        return Err(format!("buffer {} version {} changes dtype", key.0, key.1));
                    }
                    if *count != resource.version.element_count {
                        return Err(format!(
                            "buffer {} version {} carries conflicting element counts {} and {}",
                            key.0, key.1, count, resource.version.element_count
                        ));
                    }
                } else {
                    shapes.insert(key, (dtype, resource.version.element_count));
                }
            }
            let identity = (
                resource.buffer.name.clone(),
                resource.buffer.semantic_value,
                map_role(resource.buffer.role),
                map_lifetime(resource.buffer.lifetime),
            );
            if let Some(previous) = identities.get(&resource.buffer.id) {
                if previous != &identity {
                    return Err(format!(
                        "buffer {} changes carried identity across slots",
                        resource.buffer.id
                    ));
                }
            } else {
                identities.insert(resource.buffer.id, identity);
            }
        }
    }
    let mut windows: Gea3WindowBindings = BTreeMap::new();
    for (kernel_index, kernel) in program.kernels.iter().enumerate() {
        for resource in &kernel.resources {
            let key = (resource.buffer.id, resource.version.version);
            let allocation = shapes.get(&key).map(|(_, count)| *count).ok_or_else(|| {
                format!(
                    "resource `{}` binds a sub-window of buffer {} version {} with no full-width declaration (GEA3-WIRE-BUFFER-V2)",
                    resource.buffer.name, key.0, key.1
                )
            })?;
            if let Some(chunked) = resource.version.chunked_window {
                admit_chunked_window(resource, chunked, allocation)?;
            }
            if let Some(binding) = admit_sub_window(resource, allocation)? {
                if windows
                    .insert(
                        (
                            u32::try_from(kernel_index).expect("kernel index fits u32"),
                            resource.binding.binding,
                        ),
                        binding,
                    )
                    .is_some()
                {
                    return Err(format!(
                        "kernel `{}` repeats a window binding at {}",
                        kernel.entry, resource.binding.binding
                    ));
                }
            }
        }
    }

    let mut kernels = Vec::with_capacity(program.kernels.len());
    for kernel in &program.kernels {
        let mut bindings = BTreeSet::new();
        let mut buffers = Vec::with_capacity(kernel.resources.len());
        for resource in &kernel.resources {
            if !bindings.insert(resource.binding.binding) {
                return Err(format!(
                    "kernel `{}` repeats binding {}",
                    kernel.entry, resource.binding.binding
                ));
            }
            let key = (resource.buffer.id, resource.version.version);
            let (_, full_count) = shapes
                .get(&key)
                .ok_or_else(|| format!("resource `{}` has no keyed shape", resource.buffer.name))?;
            if resource.version.element_count != *full_count
                && resource.version.sub_window.is_none()
            {
                return Err(format!(
                    "resource `{}` binds a sub-window without a projection fact",
                    resource.buffer.name
                ));
            }
            if matches!(
                resource.access,
                Gea3ResourceAccess::Write | Gea3ResourceAccess::ReadWrite
            ) && resource.version.element_count != *full_count
                && !resource
                    .version
                    .sub_window
                    .is_some_and(|window| window.is_contiguous())
            {
                return Err(format!(
                    "resource `{}` writes a partial shape",
                    resource.buffer.name
                ));
            }
            buffers.push(DescriptorBuffer {
                buffer_id: resource.buffer.id,
                buffer_name: resource.buffer.name.clone(),
                semantic_value: resource.buffer.semantic_value,
                role: map_role(resource.buffer.role),
                lifetime: map_lifetime(resource.buffer.lifetime),
                initialization: map_initialization(resource.initialization),
                binding: resource.binding.binding,
                element_ty: DeviceDataType::F32,
                element_count: *full_count,
                version: resource.version.version,
            });
        }
        let grid = [
            u32::try_from(kernel.launch.workgroup_count.x)
                .map_err(|_| format!("kernel `{}` grid x overflows host axis", kernel.entry))?,
            u32::try_from(kernel.launch.workgroup_count.y)
                .map_err(|_| format!("kernel `{}` grid y overflows host axis", kernel.entry))?,
            u32::try_from(kernel.launch.workgroup_count.z)
                .map_err(|_| format!("kernel `{}` grid z overflows host axis", kernel.entry))?,
        ];
        let block = [
            kernel.launch.workgroup.x,
            kernel.launch.workgroup.y,
            kernel.launch.workgroup.z,
        ];
        if grid.contains(&0) || block.contains(&0) {
            return Err(format!(
                "kernel `{}` has a zero dispatch axis",
                kernel.entry
            ));
        }
        kernels.push(DescriptorKernel {
            entry: kernel.entry.clone(),
            buffers,
            grid,
            block,
        });
    }

    let result = program
        .results
        .first()
        .ok_or_else(|| format!("{program_name} declares no logits result"))?;
    let result_count = if program_name == "prefill" {
        PREFILL_ROWS * VOCAB
    } else {
        VOCAB
    };
    if result.role as u8 != Gea3ResultRole::Output as u8
        || result.observation.cadence as u8 != Gea3ObservationCadence::EndOfRun as u8
        || result.version.element_ty != "f32"
        || result.version.element_count != result_count
        || result.produced_by as usize != LAUNCHES_PER_PROGRAM
        || result.observation.at_launch as usize != LAUNCHES_PER_PROGRAM
    {
        return Err(format!(
            "{program_name} result is not the declared logits row"
        ));
    }
    let result_key = (result.buffer.id, result.version.version);
    if shapes.get(&result_key).map(|(_, count)| *count) != Some(result_count) {
        return Err(format!(
            "{program_name} result is not backed by its final logits buffer"
        ));
    }

    let buffer_versions = shapes
        .iter()
        .map(
            |(&(buffer_id, version), &(element_ty, element_count))| DescriptorBufferVersion {
                buffer_id,
                version,
                element_ty,
                element_count,
            },
        )
        .collect();
    let data_flow = program
        .dependencies
        .iter()
        .map(|edge| DescriptorDataFlow {
            buffer_id: edge.buffer,
            version: edge.version,
            producer: edge.producer,
            consumer: edge.consumer,
        })
        .collect();
    let mut results = Vec::new();
    let mut end_of_run_results = Vec::new();
    for row in &program.results {
        let key = (row.buffer.id, row.version.version);
        let Some((element_ty, element_count)) = shapes.get(&key) else {
            return Err(format!(
                "result names buffer {} version {} which has no keyed metadata",
                row.buffer.id, row.version.version
            ));
        };
        let row_dtype = DeviceDataType::from_spelling(&row.version.element_ty)
            .ok_or_else(|| format!("result `{}` has unsupported dtype", row.buffer.name))?;
        if row_dtype != *element_ty || row.version.element_count != *element_count {
            return Err(format!(
                "result `{}` does not match buffer {} version {} metadata",
                row.buffer.name, row.buffer.id, row.version.version
            ));
        }
        match row.observation.cadence {
            Gea3ObservationCadence::PerStep => results.push(DescriptorResult {
                buffer_id: row.buffer.id,
                version: row.version.version,
                produced_by: row.produced_by,
                at_launch: row.observation.at_launch,
            }),
            Gea3ObservationCadence::EndOfRun => end_of_run_results.push(DescriptorEndOfRunResult {
                buffer_id: row.buffer.id,
                version: row.version.version,
            }),
        }
    }

    Ok((
        DeviceDescriptor {
            backend: DeviceBackend::Metal,
            module_image,
            kernels,
            launches: program
                .launches
                .iter()
                .map(|launch| DescriptorLaunch {
                    id: launch.id,
                    kernel_index: launch.kernel_index,
                })
                .collect(),
            buffer_versions,
            program_lifetime: DeviceProgramLifetime::SingleRun,
            data_flow,
            roots: program.roots.clone(),
            results,
            end_of_run_results,
        },
        windows,
    ))
}

fn admit_envelope(
    envelope: &Gea3ProgramPlanEnvelope,
    identity: Gea3Identity,
) -> Result<(), String> {
    if envelope.schema != PLAN_SCHEMA {
        return Err(format!("unexpected envelope schema `{}`", envelope.schema));
    }
    if envelope.wire_buffer_abi != WIRE_BUFFER_ABI {
        return Err(format!(
            "unexpected wire-buffer ABI `{}`; the plan must carry {WIRE_BUFFER_ABI} sub-window projection facts (GEA3-A1)",
            envelope.wire_buffer_abi
        ));
    }
    if envelope.source != SOURCE {
        return Err(format!("unexpected plan source `{}`", envelope.source));
    }
    // GEA3-A2g: 33 members — the composed MLP parents replace the six
    // previously public MLP leaves while the chunked o-projection remains.
    if envelope.module_image_rule != MODULE_IMAGE_RULE || envelope.module_members.len() != 33 {
        return Err("module assembly facts drifted".to_owned());
    }
    if envelope.instance_expansion.layers != LAYERS
        || envelope.instance_expansion.block_launches_per_layer != BLOCK_LAUNCHES_PER_LAYER
        || envelope.instance_expansion.attention_heads != 15
        || envelope.instance_expansion.kv_heads != 5
        || !envelope.instance_expansion.per_layer_weight_identities
        || !envelope.instance_expansion.per_layer_intermediates
        || !envelope.instance_expansion.per_layer_kv_buffers
    {
        return Err("full-model instance expansion facts drifted".to_owned());
    }
    if envelope.kv_geometry.capacity != identity.history_capacity
        || envelope.kv_geometry.declared_history_length != identity.history_capacity
        || envelope.kv_geometry.dtype != "F32"
        || !envelope.kv_geometry.mask_beyond_length
    {
        return Err(format!(
            "KV geometry is not the {} fixed-capacity contract",
            identity.name
        ));
    }
    if envelope.declared_outputs.prefill_logits != vec![PREFILL_ROWS, VOCAB]
        || envelope.declared_outputs.decode_logits != vec![VOCAB]
    {
        return Err("declared logits shapes drifted".to_owned());
    }
    Ok(())
}

fn admit_program(
    envelope: &Gea3ProgramPlanEnvelope,
    program: &Gea3Program,
    program_name: &str,
    identity: Gea3Identity,
) -> Result<(), String> {
    let expected = LAUNCHES_PER_PROGRAM;
    if program.program
        != if program_name == "prefill" {
            "prefill"
        } else {
            "decode-step"
        }
        || !matches!(program.lifetime, Gea3ProgramLifetime::SingleRun)
        || program.kernels.len() != expected
        || program.launches.len() != expected
        || program.dependencies.len() != DEPENDENCIES_PER_PROGRAM
        || program.roots != vec![1]
        || program.declared_history_length != identity.history_capacity
    {
        return Err(format!("{program_name} program structural facts drifted"));
    }
    let mut allocation_ids = BTreeSet::new();
    for allocation in &program.allocations {
        if !allocation_ids.insert(allocation.id)
            || allocation.dtype != "f32"
            || allocation.element_count == 0
            || !matches!(allocation.storage, Gea3StorageLayout::DeviceHandle)
        {
            return Err(format!("{program_name} allocation facts are invalid"));
        }
    }
    let mut semantic_ids = BTreeSet::new();
    for semantic in &program.semantic_values {
        if !semantic_ids.insert(semantic.id) {
            return Err(format!(
                "{program_name} repeats semantic identity {}",
                semantic.id
            ));
        }
    }
    for (index, launch) in program.launches.iter().enumerate() {
        if launch.id != u32::try_from(index + 1).unwrap()
            || launch.kernel_index != u32::try_from(index).unwrap()
        {
            return Err(format!(
                "{program_name} launch order is not carried explicitly"
            ));
        }
    }
    for (index, kernel) in program.kernels.iter().enumerate() {
        // Function identities are zero-based in the producer's exported
        // module table, so function 0 is a valid entry identity.
        if kernel.resources.is_empty() {
            return Err(format!("{program_name} kernel {index} is incomplete"));
        }
        if !allocation_ids.iter().any(|id| {
            kernel
                .resources
                .iter()
                .any(|resource| resource.buffer.id == *id)
        }) {
            return Err(format!(
                "{program_name} kernel {index} has no allocated resource"
            ));
        }
        check_recipe(&kernel.entry, &kernel.plan)?;
        if kernel.launch.declared_input_count == 0
            || kernel.launch.declared_input_count >= kernel.resources.len()
        {
            return Err(format!(
                "{program_name} `{}` input arity fact drifted",
                kernel.entry
            ));
        }
        if kernel.launch.declared_output_count == 0 {
            return Err(format!(
                "{program_name} `{}` has no declared output",
                kernel.entry
            ));
        }
        // GEA4 (a) mirror: the carried launch facts must agree with each
        // other and with the kernel's own write slot — the threadgroup
        // requirements themselves are radix-admitted against the emitted
        // MSL; the hosts own the two carried copies never contradicting.
        if (
            kernel.launch.dispatch_size.x,
            kernel.launch.dispatch_size.y,
            kernel.launch.dispatch_size.z,
        ) != (
            kernel.launch.workgroup_count.x,
            kernel.launch.workgroup_count.y,
            kernel.launch.workgroup_count.z,
        ) {
            return Err(format!(
                "{program_name} `{}` launch facts disagree with each other: dispatch_size {:?} != workgroup_count {:?} (GEA4 derived-geometry law)",
                kernel.entry,
                (
                    kernel.launch.dispatch_size.x,
                    kernel.launch.dispatch_size.y,
                    kernel.launch.dispatch_size.z
                ),
                (
                    kernel.launch.workgroup_count.x,
                    kernel.launch.workgroup_count.y,
                    kernel.launch.workgroup_count.z
                )
            ));
        }
        if kernel.launch.workgroup.x == 0
            || kernel.launch.workgroup.y == 0
            || kernel.launch.workgroup.z == 0
            || kernel.launch.workgroup_count.x == 0
            || kernel.launch.workgroup_count.y == 0
            || kernel.launch.workgroup_count.z == 0
        {
            return Err(format!(
                "{program_name} `{}` launch carries a zero dispatch axis (GEA4 derived-geometry law)",
                kernel.entry
            ));
        }
        // The declared output count must agree with the kernel's
        // full-width write slot.  Windowed writes are owned by the
        // two-truths and scattered-write laws (their carried width is a
        // projection fact, not the output declaration).
        let write_width = kernel
            .resources
            .iter()
            .filter(|resource| {
                matches!(resource.access, Gea3ResourceAccess::Write)
                    && resource.version.sub_window.is_none()
            })
            .map(|resource| resource.version.element_count)
            .max();
        match write_width {
            Some(width) if width == kernel.launch.declared_output_count => {}
            Some(width) => {
                return Err(format!(
                    "{program_name} `{}` declared output count {} disagrees with its write slot width {width} (GEA4 derived-geometry law)",
                    kernel.entry, kernel.launch.declared_output_count
                ));
            }
            // A kernel whose only writes are windowed projections (the KV
            // mutators, the attention-concat tiles) declares its output
            // through the window laws, not a full-width slot.
            None => {}
        }
    }
    for edge in &program.dependencies {
        if edge.producer == 0
            || edge.consumer <= edge.producer
            || edge.consumer > expected as u32
            || !allocation_ids.contains(&edge.buffer)
            || edge.version == 0
        {
            return Err(format!("{program_name} carries an invalid dependency edge"));
        }
    }
    let mut seen_edges = BTreeSet::new();
    for edge in &program.dependencies {
        if !seen_edges.insert((edge.producer, edge.consumer, edge.buffer, edge.version)) {
            return Err(format!("{program_name} repeats a dependency edge"));
        }
    }
    // GEA4 (b) mirror: binding-vs-edge agreement on EVERY edge — the
    // producer launch must write the edge's (buffer, version) and the
    // consumer launch must bind it at a read (or read-write) slot.  This
    // replaces the KV-pair-scoped special case: a mis-binding anywhere in
    // the ABI (the launch-6 exact-zero class, the seam class num-3/num-10
    // proved — a flat read of a chunked truth and a strided write of a
    // flat truth are the same defect) fails closed before any device work.
    for edge in &program.dependencies {
        let producer = &program.kernels[(edge.producer - 1) as usize];
        let consumer = &program.kernels[(edge.consumer - 1) as usize];
        let producer_writes = producer.resources.iter().any(|resource| {
            matches!(
                resource.access,
                Gea3ResourceAccess::Write | Gea3ResourceAccess::ReadWrite
            ) && resource.buffer.id == edge.buffer
                && resource.version.version == edge.version
        });
        let consumer_binds = consumer.resources.iter().any(|resource| {
            matches!(
                resource.access,
                Gea3ResourceAccess::Read | Gea3ResourceAccess::ReadWrite
            ) && resource.buffer.id == edge.buffer
                && resource.version.version == edge.version
        });
        if !producer_writes || !consumer_binds {
            return Err(format!(
                "{program_name} binding-vs-edge mismatch at edge {} → {}: edge names buffer {} version {} (producer writes: {}, consumer binds: {})",
                edge.producer,
                edge.consumer,
                edge.buffer,
                edge.version,
                producer_writes,
                consumer_binds
            ));
        }
    }
    // GEA4 (d) mirror: the staged-composition non-zero-intermediates
    // readback (b7c1db3) as a per-family structural gate.  Every
    // kernel-initialized intermediate a launch reads must be covered by
    // prior writes at the same (buffer, version): full-width and chunked
    // reads need the whole allocation tiled with no gap; windowed reads
    // need each row interval inside one prior write span.  Host-provided
    // and zero-fill buffers are initialized truths — excluded.
    gea4_intermediate_coverage(program, program_name)?;
    admit_state_buffers(program, program_name, identity)?;
    let expected_output_count = if program_name == "prefill" {
        PREFILL_ROWS * VOCAB
    } else {
        VOCAB
    };
    let output = program
        .results
        .first()
        .ok_or_else(|| format!("{program_name} logits output is undeclared"))?;
    if output.buffer.role as u8 != Gea3BufferRole::Output as u8
        || output.version.element_count != expected_output_count
        || output.version.element_ty != "f32"
        || output.produced_by as usize != expected
        || output.observation.at_launch as usize != expected
        || !matches!(output.observation.cadence, Gea3ObservationCadence::EndOfRun)
    {
        return Err(format!(
            "{program_name} logits output is not declared at the final launch"
        ));
    }
    let _ = envelope;
    Ok(())
}

/// GEA4 (d): the zero-intermediate structural gate (hosts mirror of the
/// harness admission law).  Walks the family's launches in order and
/// requires every READ of a kernel-initialized buffer to be covered by
/// prior writes of the same (buffer, version) — an unwritten region reads
/// the zero-fill, the exact-zero-logits class the device readback had to
/// discover after execution.  Plan-carried intervals prove coverage; no
/// symbolic execution.
fn gea4_intermediate_coverage(program: &Gea3Program, program_name: &str) -> Result<(), String> {
    let mut full_width: BTreeMap<(u32, u32), u64> = BTreeMap::new();
    for kernel in &program.kernels {
        for resource in &kernel.resources {
            if resource.version.sub_window.is_none() {
                // First declaration wins; conflicting counts are the
                // mapper's WIRE-BUFFER-V1 law, not this gate's.
                full_width
                    .entry((resource.buffer.id, resource.version.version))
                    .or_insert(resource.version.element_count);
            }
        }
    }
    let kernel_initialized: BTreeMap<u32, ()> = program
        .allocations
        .iter()
        .filter(|allocation| {
            matches!(
                allocation.initialization,
                Gea3Initialization::KernelInitialized
            )
        })
        .map(|allocation| (allocation.id, ()))
        .collect();
    let mut written: BTreeMap<(u32, u32), Vec<(u64, u64)>> = BTreeMap::new();
    for (index, kernel) in program.kernels.iter().enumerate() {
        let launch = index + 1;
        for resource in &kernel.resources {
            let key = (resource.buffer.id, resource.version.version);
            let Some(&allocation) = full_width.get(&key) else {
                return Err(format!(
                    "{program_name} `{}` binds buffer {} version {} with no full-width declaration (GEA4 structural readback gate)",
                    kernel.entry, key.0, key.1
                ));
            };
            let window = resource.version.sub_window;
            match resource.access {
                Gea3ResourceAccess::Write | Gea3ResourceAccess::ReadWrite => {
                    let span = window.map_or(Some((0, allocation)), |window| {
                        window.covering_span().map(|span| {
                            (
                                window.element_offset,
                                window.element_offset.saturating_add(span),
                            )
                        })
                    });
                    let Some(span) = span else {
                        return Err(format!(
                            "{program_name} `{}` write window overflows (GEA4 structural readback gate)",
                            kernel.entry
                        ));
                    };
                    written.entry(key).or_default().push(span);
                }
                Gea3ResourceAccess::Read => {
                    if !kernel_initialized.contains_key(&key.0) {
                        continue;
                    }
                    let Some(spans) = written.get(&key) else {
                        return Err(format!(
                            "{program_name} zero-intermediate composition: launch {launch} (`{}`) reads kernel-initialized buffer {} version {} that no prior launch wrote (GEA4 structural readback gate)",
                            kernel.entry, key.0, key.1
                        ));
                    };
                    match window {
                        // Full-width and chunked reads span the whole
                        // allocation: prior writes must tile it with no gap.
                        None => {
                            let mut ordered = spans.clone();
                            ordered.sort_unstable();
                            let mut cursor = 0_u64;
                            for (start, end) in ordered {
                                if start > cursor {
                                    break;
                                }
                                cursor = cursor.max(end);
                            }
                            if cursor < allocation {
                                return Err(format!(
                                    "{program_name} zero-intermediate composition: launch {launch} (`{}`) reads kernel-initialized buffer {} version {} whole, but prior writes cover only [0, {cursor}) of {allocation} — an unwritten gap reads zeros (GEA4 structural readback gate)",
                                    kernel.entry, key.0, key.1
                                ));
                            }
                        }
                        Some(window) => {
                            for row in 0..window.row_count {
                                let start = window.element_offset + row * window.row_stride;
                                let end = start + window.row_width;
                                if !spans.iter().any(|(write_start, write_end)| {
                                    *write_start <= start && end <= *write_end
                                }) {
                                    return Err(format!(
                                        "{program_name} zero-intermediate composition: launch {launch} (`{}`) reads kernel-initialized buffer {} version {} window row {row} [{start}, {end}) that no prior launch wrote (GEA4 structural readback gate)",
                                        kernel.entry, key.0, key.1
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn admit_state_buffers(
    program: &Gea3Program,
    program_name: &str,
    identity: Gea3Identity,
) -> Result<(), String> {
    let cache_rows: Vec<&Gea3StateBuffer> = program
        .state_buffers
        .iter()
        .filter(|row| {
            row.name
                .as_deref()
                .is_some_and(|name| name.contains(".kv_"))
        })
        .collect();
    if cache_rows.len() != LAYERS * 2 {
        return Err(format!(
            "{program_name} does not declare two KV buffers per layer"
        ));
    }
    for layer in 0..LAYERS {
        for cache in ["kv_k", "kv_v"] {
            let name = format!("blk.{layer:02}.{cache}");
            let row = cache_rows
                .iter()
                .find(|row| row.name.as_deref() == Some(name.as_str()))
                .ok_or_else(|| format!("{program_name} is missing {name}"))?;
            if row.layer != layer
                || row.role != Some(Gea3BufferRole::InOut)
                || row.dtype.as_deref() != Some("f32")
                || row.shape.as_deref() != Some(&[identity.history_capacity, KV_WIDTH][..])
                || row.history_capacity != Some(identity.history_capacity)
                || row.declared_history_length != Some(identity.history_capacity)
            {
                return Err(format!("{program_name} KV geometry drifted for {name}"));
            }
        }
    }
    Ok(())
}

fn check_recipe(entry: &str, plan: &Gea3Plan) -> Result<(), String> {
    let admitted = match entry {
        "decode_gemv_qo"
        | "decode_gemv_kv"
        | "decode_score_gemm"
        | "decode_context_gemm"
        | "prefill_gemm_qo"
        | "prefill_gemm_o"
        | "prefill_gemm_kv"
        | "prefill_score_gemm"
        | "prefill_context_gemm"
        | "lm_head_gemv"
        | "prefill_lm_head_gemv" => matches!(plan, Gea3Plan::TiledMatMul(_)),
        "decode_mlp" | "prefill_mlp" => matches!(plan, Gea3Plan::ComposedMatMul(_)),
        "head_rmsnorm" | "prefill_head_rmsnorm" | "decode_rmsnorm" | "prefill_rmsnorm" => {
            matches!(plan, Gea3Plan::RmsNormalization(_))
        }
        "decode_rope_q" | "decode_rope_k" | "prefill_rope_q" | "prefill_rope_k" => {
            matches!(plan, Gea3Plan::Rope(_))
        }
        "decode_key_transpose" | "prefill_key_transpose" => {
            matches!(plan, Gea3Plan::Transpose(_))
        }
        "decode_masked_softmax" | "prefill_causal_softmax" => {
            matches!(plan, Gea3Plan::CausalMaskedSoftmax(_))
        }
        "embedding_gather" | "prefill_embedding_gather" => matches!(plan, Gea3Plan::Gather(_)),
        "decode_residual_add" | "prefill_residual_add" => {
            matches!(plan, Gea3Plan::Elementwise)
        }
        "kv_append_k" | "kv_append_v" | "prefill_kv_write_k" | "prefill_kv_write_v" => {
            matches!(plan, Gea3Plan::TiledMatMul(_))
        }
        other => return Err(format!("unknown GEA3 entry `{other}`")),
    };
    if admitted {
        Ok(())
    } else {
        Err(format!("GEA3 entry `{entry}` carries the wrong recipe"))
    }
}

fn map_role(role: Gea3BufferRole) -> DeviceBufferRole {
    match role {
        Gea3BufferRole::Input => DeviceBufferRole::Input,
        Gea3BufferRole::Output => DeviceBufferRole::Output,
        Gea3BufferRole::InOut => DeviceBufferRole::InOut,
    }
}

fn map_lifetime(lifetime: Gea3BufferLifetime) -> DeviceBufferLifetime {
    match lifetime {
        Gea3BufferLifetime::PerProgram => DeviceBufferLifetime::PerProgram,
        Gea3BufferLifetime::PerStep => DeviceBufferLifetime::PerStep,
        Gea3BufferLifetime::ObservationPoint => DeviceBufferLifetime::ObservationPoint,
    }
}

fn map_initialization(initialization: Gea3Initialization) -> DeviceBufferInitialization {
    match initialization {
        Gea3Initialization::ZeroFill => DeviceBufferInitialization::ZeroFill,
        Gea3Initialization::HostProvided => DeviceBufferInitialization::HostProvided,
        Gea3Initialization::KernelInitialized => DeviceBufferInitialization::KernelInitialized,
    }
}

// ---------------------------------------------------------------------------
// Fake structural execution.  The driver receives one tiny placeholder buffer
// per declared binding.  No model weights are allocated, copied, or read: the
// plan/descriptor admission above is the only source of shape and residency
// facts in this U5a structural proof.
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct StructuralLoopReceipt {
    prefill_runs: usize,
    decode_runs: usize,
    launches: usize,
    readbacks: usize,
}

fn fake_entry_names() -> impl Iterator<Item = &'static str> {
    [
        "embedding_gather",
        "prefill_embedding_gather",
        "decode_rmsnorm",
        "decode_gemv_qo",
        "decode_gemv_kv",
        "decode_mlp",
        "decode_rope_q",
        "decode_rope_k",
        "kv_append_k",
        "kv_append_v",
        "decode_key_transpose",
        "decode_score_gemm",
        "decode_masked_softmax",
        "decode_context_gemm",
        "decode_residual_add",
        "prefill_rmsnorm",
        "prefill_gemm_qo",
        "prefill_gemm_o",
        "prefill_gemm_kv",
        "prefill_mlp",
        "prefill_rope_q",
        "prefill_rope_k",
        "prefill_key_transpose",
        "prefill_score_gemm",
        "prefill_causal_softmax",
        "prefill_context_gemm",
        "prefill_residual_add",
        "prefill_kv_write_k",
        "prefill_kv_write_v",
        "head_rmsnorm",
        "lm_head_gemv",
        "prefill_head_rmsnorm",
        "prefill_lm_head_gemv",
    ]
    .into_iter()
}

fn fake_execute_program(
    session: &mut MetalHostSession,
    descriptor: &DeviceDescriptor,
    runs: usize,
) -> Result<usize, String> {
    let module = session
        .load_module(&descriptor.module_image)
        .map_err(|error| error.message.clone())?;
    let placeholder = session
        .alloc_bytes(4)
        .map_err(|error| error.message.clone())?;
    let mut launches = 0usize;
    let result = (|| {
        for _ in 0..runs {
            for launch in &descriptor.launches {
                let kernel = &descriptor.kernels[launch.kernel_index as usize];
                let bindings = vec![placeholder; kernel.buffers.len()];
                session
                    .launch_kernel_3d(
                        module,
                        &kernel.entry,
                        &bindings,
                        kernel.grid[0],
                        kernel.grid[1],
                        kernel.grid[2],
                        kernel.block[0],
                        kernel.block[1],
                        kernel.block[2],
                    )
                    .map_err(|error| error.message.clone())?;
                launches += 1;
            }
            session.sync().map_err(|error| error.message.clone())?;
        }
        Ok(launches)
    })();
    let _ = session.release(placeholder);
    let _ = session.release(module);
    result
}

fn assert_declared_logits_only(
    descriptor: &DeviceDescriptor,
    buffer_id: u32,
) -> Result<(), String> {
    if descriptor
        .end_of_run_results
        .iter()
        .any(|result| result.buffer_id == buffer_id)
    {
        Ok(())
    } else {
        Err(format!(
            "buffer {buffer_id} is not a declared logits observation"
        ))
    }
}

fn map_both(
    envelope: &Gea3ProgramPlanEnvelope,
    artifact_dir: &Path,
    identity: Gea3Identity,
) -> Result<
    (
        (DeviceDescriptor, Gea3WindowBindings),
        (DeviceDescriptor, Gea3WindowBindings),
    ),
    String,
> {
    let prefill = map_envelope_to_descriptor(
        envelope,
        &envelope.programs.prefill,
        "prefill",
        artifact_dir,
        identity,
    )?;
    let decode = map_envelope_to_descriptor(
        envelope,
        &envelope.programs.decode_step,
        "decode_step",
        artifact_dir,
        identity,
    )?;
    prefill
        .0
        .validate()
        .map_err(|error| format!("prefill descriptor rejected: {}", error.message))?;
    decode
        .0
        .validate()
        .map_err(|error| format!("decode descriptor rejected: {}", error.message))?;
    Ok((prefill, decode))
}

fn model_weight_names(program: &Gea3Program) -> BTreeSet<String> {
    program
        .allocations
        .iter()
        .filter(|allocation| {
            allocation.lifetime as u8 == Gea3BufferLifetime::PerProgram as u8
                && allocation.initialization as u8 == Gea3Initialization::HostProvided as u8
                && (allocation.name.starts_with("blk.")
                    || allocation.name == "token_embd.weight"
                    || allocation.name == "output_norm.weight")
        })
        .map(|allocation| allocation.name.clone())
        .collect()
}

#[test]
fn gea3_descriptor_admission() {
    let artifact_dir = gea3_artifact_dir();
    let envelope = load_gea3_plan(&artifact_dir);
    let ((prefill, prefill_windows), (decode, decode_windows)) = map_both(&envelope, &artifact_dir, GEA3_FROZEN_SHORT)
        .unwrap_or_else(|error| panic!("GEA3 plan → DeviceDescriptor mapping failed: {error}"));

    assert_eq!(prefill.kernels.len(), LAUNCHES_PER_PROGRAM);
    assert_eq!(decode.kernels.len(), LAUNCHES_PER_PROGRAM);
    assert_eq!(prefill.data_flow.len(), DEPENDENCIES_PER_PROGRAM);
    assert_eq!(decode.data_flow.len(), DEPENDENCIES_PER_PROGRAM);
    assert_eq!(prefill.end_of_run_results.len(), 1);
    assert_eq!(decode.end_of_run_results.len(), 1);
    assert_eq!(prefill.end_of_run_results[0].version, 1);
    assert_eq!(decode.end_of_run_results[0].version, 1);
    assert_eq!(
        model_weight_names(&envelope.programs.decode_step).len(),
        290
    );

    // GEA3-A1: the mapper accepts the carried sub-window projections.  The
    // decode program carries the per-head read-side windows (5 key
    // transposes + 15 score queries + 15 context values per layer) plus the
    // 15 attention-output write windows per layer; prefill mirrors the
    // geometry at T_p (windows are always in-bounds bindings of the one
    // allocation).  GEA3-U6 num-1 adds the two full-span arena write
    // windows per layer (kv_append / prefill_kv_write).
    let expected_decode_windows = LAYERS * (5 + 15 + 15 + 15 + 2);
    let expected_prefill_windows = LAYERS * (5 + 15 + 15 + 15 + 2);
    assert_eq!(
        decode_windows.len(),
        expected_decode_windows,
        "decode window binding count"
    );
    assert_eq!(
        prefill_windows.len(),
        expected_prefill_windows,
        "prefill window binding count"
    );
    for (byte_offset, view_span) in decode_windows.values().chain(prefill_windows.values()) {
        assert_eq!(
            byte_offset % 4,
            0,
            "window byte offsets are element aligned"
        );
        assert_eq!(view_span % 4, 0, "window spans are element aligned");
        assert!(*view_span > 0, "a window binds a non-empty span");
    }

    // The cache declarations are not inferred from resource extents. They
    // remain an independently admitted, capacity-bounded fact.
    assert_eq!(
        envelope
            .programs
            .decode_step
            .state_buffers
            .iter()
            .filter(|row| row
                .name
                .as_deref()
                .is_some_and(|name| name.contains(".kv_")))
            .count(),
        LAYERS * 2
    );
}

#[test]
fn gea3_mirror_parse_rejects_malformed_plans() {
    let artifact_dir = gea3_artifact_dir();
    let bytes = fs::read(artifact_dir.join(PLAN_MEMBER)).expect("read exported GEA3 plan");
    let value: Value = serde_json::from_slice(&bytes).expect("exported plan is JSON");

    let mut unknown = value.clone();
    unknown["mystery"] = json!(true);
    assert!(serde_json::from_value::<Gea3ProgramPlanEnvelope>(unknown).is_err());

    let mut missing = value.clone();
    missing
        .as_object_mut()
        .expect("envelope object")
        .remove("kv_geometry");
    assert!(serde_json::from_value::<Gea3ProgramPlanEnvelope>(missing).is_err());

    let mut unknown_kernel = value.clone();
    unknown_kernel["programs"]["decode_step"]["kernels"][0]["mystery"] = json!(1);
    assert!(serde_json::from_value::<Gea3ProgramPlanEnvelope>(unknown_kernel).is_err());

    let mut unknown_recipe = value.clone();
    unknown_recipe["programs"]["prefill"]["kernels"][0]["plan"] =
        json!({ "Gather": { "rows": 49152 } });
    assert!(serde_json::from_value::<Gea3ProgramPlanEnvelope>(unknown_recipe).is_err());

    let parsed: Gea3ProgramPlanEnvelope = serde_json::from_value(value).expect("mirror parse");
    let mut wrong_schema = parsed;
    wrong_schema.schema = "gea3-program-plan-v2".to_owned();
    assert!(admit_envelope(&wrong_schema, GEA3_FROZEN_SHORT).is_err());
}

/// PPB-U7: the statue admission law is capacity-exact and cross-rejecting.
/// A soak envelope (2000 rows) admits only under the soak identity; the
/// frozen identity rejects it; a 1999/2001-row envelope rejects under both;
/// and the per-layer KV state-buffer shapes follow the statue mechanically.
/// This is the identity gate only — the full bundle admission runs in the
/// §6-style physical pair against the real exported soak plan.
#[test]
fn gea3_soak_statue_admission_is_capacity_exact() {
    assert_eq!(GEA3_SOAK_L2000.l_max(), 2_000);
    assert_eq!(GEA3_SOAK_L2000.kv_bytes(), 163_840_000);
    assert_eq!(
        GEA3_SOAK_L2000.kv_basis(),
        "32 * 2 * 2000 * 320 * sizeof(F32)"
    );
    assert_eq!(GEA3_FROZEN_SHORT.kv_bytes(), 6_225_920);

    let envelope_for = |capacity: u64| -> Gea3ProgramPlanEnvelope {
        // A2g re-derive trigger: admit_envelope is capacity-exact on module
        // assembly ("module assembly facts drifted") — 33 members post-A2g,
        // was 37 pre-collapse; re-derive this builder from the admission
        // law if the composition moves again.
        let members: Vec<String> = (0..33).map(|index| format!("m{index}.metal")).collect();
        let value = json!({
            "schema": PLAN_SCHEMA,
            "wire_buffer_abi": WIRE_BUFFER_ABI,
            "source": SOURCE,
            "programs": {
                "prefill": empty_program("prefill", capacity),
                "decode_step": empty_program("decode-step", capacity),
            },
            "module_members": members,
            "module_image_rule": MODULE_IMAGE_RULE,
            "instance_expansion": {
                "layers": LAYERS,
                "block_launches_per_layer": BLOCK_LAUNCHES_PER_LAYER,
                "attention_heads": 15,
                "kv_heads": 5,
                "per_layer_weight_identities": true,
                "per_layer_intermediates": true,
                "per_layer_kv_buffers": true,
            },
            "kv_geometry": {
                "capacity": capacity,
                "declared_history_length": capacity,
                "dtype": "F32",
                "mask_beyond_length": true,
            },
            "declared_outputs": {
                "prefill_logits": [PREFILL_ROWS, VOCAB],
                "decode_logits": [VOCAB],
            },
        });
        serde_json::from_value(value).expect("synthetic statue envelope parses")
    };

    let soak = envelope_for(GEA3_SOAK_L2000.history_capacity);
    assert!(admit_envelope(&soak, GEA3_SOAK_L2000).is_ok());
    // Cross-statue rejection: the frozen identity must refuse the soak plan.
    let cross = admit_envelope(&soak, GEA3_FROZEN_SHORT)
        .expect_err("the frozen identity must reject a 2000-row KV geometry");
    assert!(cross.contains("metal-m5max"), "diagnostic {cross}");
    let frozen = envelope_for(GEA3_FROZEN_SHORT.history_capacity);
    assert!(admit_envelope(&frozen, GEA3_FROZEN_SHORT).is_ok());
    assert!(admit_envelope(&frozen, GEA3_SOAK_L2000).is_err());
    for wrong_capacity in [1_999_u64, 2_001] {
        let envelope = envelope_for(wrong_capacity);
        assert!(
            admit_envelope(&envelope, GEA3_SOAK_L2000).is_err(),
            "soak admission must reject capacity {wrong_capacity}"
        );
    }
    // The state-buffer geometry law follows the same statue fact.
    assert!(admit_state_buffers(&soak.programs.decode_step, "decode_step", GEA3_SOAK_L2000).is_ok());
    assert!(admit_state_buffers(&soak.programs.decode_step, "decode_step", GEA3_FROZEN_SHORT).is_err());
    assert!(admit_state_buffers(&frozen.programs.decode_step, "decode_step", GEA3_FROZEN_SHORT).is_ok());
}

/// A structurally minimal program for the statue admission law: the envelope
/// and KV state facts only.  Kernel/launch counts are the full-bundle
/// admission's business, proven by the physical pair.
fn empty_program(spelling: &str, capacity: u64) -> Value {
    let state_buffers: Vec<Value> = (0..LAYERS)
        .flat_map(|layer| {
            ["kv_k", "kv_v"].into_iter().map(move |cache| {
                json!({
                    "id": u32::try_from(layer * 2 + if cache == "kv_k" { 0 } else { 1 }).unwrap(),
                    "name": format!("blk.{layer:02}.{cache}"),
                    "layer": layer,
                    "role": "InOut",
                    "dtype": "f32",
                    "shape": [capacity, KV_WIDTH],
                    "history_capacity": capacity,
                    "declared_history_length": capacity,
                })
            })
        })
        .collect();
    json!({
        "program": spelling,
        "kernels": [],
        "launches": [],
        "lifetime": "single-run",
        "results": [],
        "allocations": [],
        "semantic_values": [],
        "roots": [1],
        "dependencies": [],
        "relations": [],
        "state_buffers": state_buffers,
        "declared_history_length": capacity,
    })
}

#[test]
fn gea3_negative_rows_fail_closed() {    let artifact_dir = gea3_artifact_dir();
    let bytes = fs::read(artifact_dir.join(PLAN_MEMBER)).expect("read exported GEA3 plan");
    let original: Value = serde_json::from_slice(&bytes).expect("exported plan is JSON");

    let mut missing_edge = original.clone();
    missing_edge["programs"]["decode_step"]["dependencies"]
        .as_array_mut()
        .expect("dependencies")
        .pop();
    let parsed: Gea3ProgramPlanEnvelope =
        serde_json::from_value(missing_edge).expect("mirror parse");
    assert!(admit_program(&parsed, &parsed.programs.decode_step, "decode_step", GEA3_FROZEN_SHORT).is_err());

    let mut wrong_dtype = original.clone();
    wrong_dtype["programs"]["prefill"]["kernels"][0]["resources"][0]["version"]["element_ty"] =
        json!("bf16");
    let parsed: Gea3ProgramPlanEnvelope =
        serde_json::from_value(wrong_dtype).expect("mirror parse");
    assert!(map_envelope_to_descriptor(
        &parsed,
        &parsed.programs.prefill,
        "prefill",
        &artifact_dir,
        GEA3_FROZEN_SHORT,
    )
    .is_err());

    let mut conflicting_shape = original.clone();
    conflicting_shape["programs"]["decode_step"]["kernels"][1]["resources"][0]["version"]
        ["element_count"] = json!(961);
    let parsed: Gea3ProgramPlanEnvelope =
        serde_json::from_value(conflicting_shape).expect("mirror parse");
    let error = map_envelope_to_descriptor(
        &parsed,
        &parsed.programs.decode_step,
        "decode_step",
        &artifact_dir,
        GEA3_FROZEN_SHORT,
    )
    .expect_err("one buffer identity cannot carry two element counts");
    assert!(
        error.contains("conflicting element counts"),
        "diagnostic must name the GEA3-WIRE-BUFFER-V1 conflict: {error}"
    );

    let mut intermediate_readback = original.clone();
    let intermediate_id = intermediate_readback["programs"]["decode_step"]["kernels"][0]
        ["resources"]
        .as_array()
        .expect("resources")
        .iter()
        .find(|resource| resource["access"] == "write")
        .and_then(|resource| resource["buffer"]["id"].as_u64())
        .expect("first intermediate output");
    intermediate_readback["programs"]["decode_step"]["results"][0]["buffer"]["id"] =
        json!(intermediate_id);
    let parsed: Gea3ProgramPlanEnvelope =
        serde_json::from_value(intermediate_readback).expect("mirror parse");
    let error = map_envelope_to_descriptor(
        &parsed,
        &parsed.programs.decode_step,
        "decode_step",
        &artifact_dir,
        GEA3_FROZEN_SHORT,
    )
    .expect_err("intermediate readback must fail closed");
    assert!(
        error.contains("logits"),
        "diagnostic must name logits observation: {error}"
    );

    let envelope = load_gea3_plan(&artifact_dir);
    let ((_, _), (decode, _)) = map_both(&envelope, &artifact_dir, GEA3_FROZEN_SHORT).expect("real plan maps");
    assert!(assert_declared_logits_only(&decode, decode.end_of_run_results[0].buffer_id).is_ok());
    assert!(assert_declared_logits_only(&decode, u32::MAX).is_err());

    // GEA3-A1 negatives: the sub-window two-truths guard at the mapper.
    // Launch 10 (kernels[9], after the num-7 append reorder) is the first
    // key transpose; its canonical input carries the strided [76,64]
    // k-cache window.
    let mut wrong_derived = original.clone();
    wrong_derived["programs"]["decode_step"]["kernels"][9]["resources"][0]["version"]
        ["sub_window"]["derived_element_count"] = json!(4_863);
    let parsed: Gea3ProgramPlanEnvelope =
        serde_json::from_value(wrong_derived).expect("mirror parse");
    let error = map_envelope_to_descriptor(
        &parsed,
        &parsed.programs.decode_step,
        "decode_step",
        &artifact_dir,
        GEA3_FROZEN_SHORT,
    )
    .expect_err("a corrupted derived count must fail the mapper");
    assert!(
        error.contains("carried and derived truths must agree"),
        "diagnostic must name the two-truths law: {error}"
    );

    let mut off_the_end = original.clone();
    off_the_end["programs"]["decode_step"]["kernels"][9]["resources"][0]["version"]["sub_window"]
        ["element_offset"] = json!(5 * 64);
    let parsed: Gea3ProgramPlanEnvelope =
        serde_json::from_value(off_the_end).expect("mirror parse");
    let error = map_envelope_to_descriptor(
        &parsed,
        &parsed.programs.decode_step,
        "decode_step",
        &artifact_dir,
        GEA3_FROZEN_SHORT,
    )
    .expect_err("an off-the-end window must fail the mapper");
    assert!(
        error.contains("nest inside its producer buffer"),
        "diagnostic must name the nesting law: {error}"
    );

    // GEA3-U6 num-3 pitch-truth negatives (the hosts mirror): a strided
    // read window whose declared stride lies about the producer's actual
    // pitch fails the mapper even though both count laws pass — a stride
    // below the 320 pitch and a stride above it are both rejected.
    for (label, lying_stride) in [("below", 76u64), ("above", 640u64)] {
        let mut lying = original.clone();
        lying["programs"]["decode_step"]["kernels"][9]["resources"][0]["version"]["sub_window"]
            ["row_stride"] = json!(lying_stride);
        let parsed: Gea3ProgramPlanEnvelope = serde_json::from_value(lying).expect("mirror parse");
        let error = map_envelope_to_descriptor(
            &parsed,
            &parsed.programs.decode_step,
            "decode_step",
            &artifact_dir,
        GEA3_FROZEN_SHORT,
        )
        .expect_err(&format!(
            "a lying stride {label} the pitch must fail the mapper"
        ));
        assert!(
            error.contains("pitch-truth"),
            "a stride {label} the producer pitch must name the pitch-truth law: {error}"
        );
    }

    // The attention-output hybrid write window (first context launch,
    // kernels[44] after the num-7 reorder; its LAST resource is the
    // windowed write) must stay contiguous.
    let mut scattered_write = original.clone();
    let context_resources = scattered_write["programs"]["decode_step"]["kernels"][44]["resources"]
        .as_array_mut()
        .expect("context resources");
    let write_index = context_resources.len() - 1;
    context_resources[write_index]["version"]["sub_window"]["row_count"] = json!(15);
    context_resources[write_index]["version"]["sub_window"]["row_stride"] = json!(960);
    context_resources[write_index]["version"]["sub_window"]["derived_element_count"] =
        json!(15 * 64);
    context_resources[write_index]["version"]["element_count"] = json!(15 * 64);
    let parsed: Gea3ProgramPlanEnvelope =
        serde_json::from_value(scattered_write).expect("mirror parse");
    let error = map_envelope_to_descriptor(
        &parsed,
        &parsed.programs.decode_step,
        "decode_step",
        &artifact_dir,
        GEA3_FROZEN_SHORT,
    )
    .expect_err("a scattered write window must fail the mapper");
    assert!(
        error.contains("scattered write"),
        "diagnostic must name the write-window law: {error}"
    );

    // A sub-window without its projection fact stays rejected (the V1 law).
    let mut bare_sub_window = original.clone();
    bare_sub_window["programs"]["decode_step"]["kernels"][9]["resources"][0]["version"]
        .as_object_mut()
        .expect("version object")
        .remove("sub_window");
    let parsed: Gea3ProgramPlanEnvelope =
        serde_json::from_value(bare_sub_window).expect("mirror parse");
    let error = map_envelope_to_descriptor(
        &parsed,
        &parsed.programs.decode_step,
        "decode_step",
        &artifact_dir,
        GEA3_FROZEN_SHORT,
    )
    .expect_err("a window count without the projection fact must fail the mapper");
    assert!(
        error.contains("sub-window without a projection fact")
            || error.contains("conflicting element counts"),
        "diagnostic must name the missing-fact or one-count law: {error}"
    );

    // The ABI tag is fail-closed: a V1 plan (no successor tag) cannot map.
    let mut v1_plan = original.clone();
    v1_plan["wire_buffer_abi"] = json!("gea3-wire-buffer-v1");
    let parsed: Gea3ProgramPlanEnvelope = serde_json::from_value(v1_plan).expect("mirror parse");
    let error = map_envelope_to_descriptor(
        &parsed,
        &parsed.programs.decode_step,
        "decode_step",
        &artifact_dir,
        GEA3_FROZEN_SHORT,
    )
    .expect_err("a V1-tagged plan must fail the mapper");
    assert!(
        error.contains("wire-buffer ABI"),
        "diagnostic must name the ABI law: {error}"
    );
}

/// GEA4 pre-Metal plan-admission pass (hosts mirror): the derived gates
/// red-green against the exported proven-good bundle — (a) launch-facts
/// consistency, (b) binding-vs-edge agreement on every edge, (d) the
/// staged-composition non-zero-intermediates readback (b7c1db3) promoted
/// to a structural gate.  The count vocabulary (c) is the radix harness
/// law; the hosts mirror carries no family spec table.
#[test]
fn gea4_admission_gates_fail_closed() {
    let artifact_dir = gea3_artifact_dir();
    let bytes = fs::read(artifact_dir.join(PLAN_MEMBER)).expect("read exported GEA3 plan");
    let original: Value = serde_json::from_slice(&bytes).expect("exported plan is JSON");

    // Green control: both family programs admit as exported.
    let envelope = load_gea3_plan(&artifact_dir);
    assert!(admit_program(&envelope, &envelope.programs.decode_step, "decode_step", GEA3_FROZEN_SHORT).is_ok());
    assert!(admit_program(&envelope, &envelope.programs.prefill, "prefill", GEA3_FROZEN_SHORT).is_ok());

    // (a) red — the two carried copies of the grid disagree with each
    // other (dispatch_size mutated, workgroup_count left alone).
    let mut contradictory = original.clone();
    contradictory["programs"]["decode_step"]["kernels"][1]["launch"]["dispatch_size"]["x"] =
        json!(480);
    let parsed: Gea3ProgramPlanEnvelope =
        serde_json::from_value(contradictory).expect("mirror parse");
    let error = admit_program(&parsed, &parsed.programs.decode_step, "decode_step", GEA3_FROZEN_SHORT)
        .expect_err("contradictory launch facts must fail closed");
    assert!(
        error.contains("launch facts disagree with each other"),
        "diagnostic must name the carried-facts law: {error}"
    );

    // (b) red — non-KV binding-vs-edge mismatch on the CONSUMER side:
    // launch 6 (`decode_rope_q`) rewires its canonical input to the
    // embedding output while the edge still names the q projection.
    let mut non_kv_misbinding = original.clone();
    let embedding_output_id = original["programs"]["decode_step"]["kernels"][0]["resources"]
        .as_array()
        .expect("embedding resources")
        .last()
        .expect("embedding output")["buffer"]["id"]
        .clone();
    non_kv_misbinding["programs"]["decode_step"]["kernels"][5]["resources"][0]["buffer"]["id"] =
        embedding_output_id;
    let parsed: Gea3ProgramPlanEnvelope =
        serde_json::from_value(non_kv_misbinding).expect("mirror parse");
    let error = admit_program(&parsed, &parsed.programs.decode_step, "decode_step", GEA3_FROZEN_SHORT)
        .expect_err("a non-KV mis-binding must fail closed");
    assert!(
        error.contains("binding-vs-edge mismatch"),
        "diagnostic must name the every-edge law: {error}"
    );

    // (b) red — the same law from the PRODUCER side: the rope_q edge
    // rewired to launch 1, which never writes the edge buffer.
    let mut producer_mismatch = original.clone();
    let rope_edge = producer_mismatch["programs"]["decode_step"]["dependencies"]
        .as_array_mut()
        .expect("dependencies")
        .iter_mut()
        .find(|edge| edge["consumer"] == json!(6))
        .expect("rope_q edge");
    rope_edge["producer"] = json!(1);
    let parsed: Gea3ProgramPlanEnvelope =
        serde_json::from_value(producer_mismatch).expect("mirror parse");
    let error = admit_program(&parsed, &parsed.programs.decode_step, "decode_step", GEA3_FROZEN_SHORT)
        .expect_err("a producer that never writes the edge buffer must fail closed");
    assert!(
        error.contains("binding-vs-edge mismatch"),
        "diagnostic must name the every-edge law: {error}"
    );

    // (d) red — zero-intermediate composition: context tile 5's write
    // window shifts one head off its slot (still nested, carried count
    // untouched), leaving [320, 384) of the attention concat unwritten;
    // the o-projection's full-width read then observes the zero-fill.
    let mut zero_intermediate = original.clone();
    let context_write_index = original["programs"]["decode_step"]["kernels"][49]["resources"]
        .as_array()
        .expect("context resources")
        .len()
        - 1;
    zero_intermediate["programs"]["decode_step"]["kernels"][49]["resources"][context_write_index]
        ["version"]["sub_window"]["element_offset"] = json!(6 * 64);
    let parsed: Gea3ProgramPlanEnvelope =
        serde_json::from_value(zero_intermediate).expect("mirror parse");
    let error = admit_program(&parsed, &parsed.programs.decode_step, "decode_step", GEA3_FROZEN_SHORT)
        .expect_err("an unwritten concat gap must fail closed at admission");
    assert!(
        error.contains("zero-intermediate composition"),
        "diagnostic must name the structural readback gate: {error}"
    );
}

#[test]
fn gea3_fake_multi_step_structural_loop() {
    let artifact_dir = gea3_artifact_dir();
    let envelope = load_gea3_plan(&artifact_dir);
    let ((prefill, _), (decode, _)) = map_both(&envelope, &artifact_dir, GEA3_FROZEN_SHORT)
        .unwrap_or_else(|error| panic!("GEA3 plan → DeviceDescriptor mapping failed: {error}"));

    // The model and KV facts are admitted once, before the fake loop starts.
    assert_eq!(
        model_weight_names(&envelope.programs.decode_step).len(),
        290
    );
    assert_eq!(
        envelope
            .programs
            .decode_step
            .state_buffers
            .iter()
            .filter(|row| row
                .name
                .as_deref()
                .is_some_and(|name| name.contains(".kv_")))
            .count(),
        LAYERS * 2
    );

    let mut driver = FakeMetalDriver::default();
    for entry in fake_entry_names() {
        driver = driver.with_known_entry(entry);
    }
    let mut session =
        MetalHostSession::with_driver(Box::new(driver)).expect("fake Metal admission");
    let prefill_launches = fake_execute_program(&mut session, &prefill, 1)
        .unwrap_or_else(|error| panic!("fake prefill failed: {error}"));
    let decode_launches = fake_execute_program(&mut session, &decode, DECODE_STEPS)
        .unwrap_or_else(|error| panic!("fake decode loop failed: {error}"));
    let counters = session.driver_counters();
    let receipt = StructuralLoopReceipt {
        prefill_runs: 1,
        decode_runs: DECODE_STEPS,
        launches: prefill_launches + decode_launches,
        // Structural U5a intentionally does not issue a device readback. The
        // declared logits observation is checked separately, and the negative
        // row above proves an intermediate cannot become one by omission.
        readbacks: 0,
    };

    assert_eq!(receipt.prefill_runs, 1);
    assert_eq!(receipt.decode_runs, DECODE_STEPS);
    assert_eq!(receipt.launches, LAUNCHES_PER_PROGRAM * (DECODE_STEPS + 1));
    assert_eq!(receipt.readbacks, 0);
    assert_eq!(
        counters.module_loads, 2,
        "prefill and decode modules load once"
    );
    assert_eq!(
        counters.module_releases, 2,
        "both modules release after the fake loop"
    );
    assert_eq!(
        counters.buffer_allocs, 2,
        "placeholder bindings do not fake model residency"
    );
    assert_eq!(counters.buffer_releases, 2);
    assert_eq!(
        session.live_handle_count(),
        0,
        "fake loop teardown releases every handle"
    );
    assert!(assert_declared_logits_only(&prefill, prefill.end_of_run_results[0].buffer_id).is_ok());
    assert!(assert_declared_logits_only(&decode, decode.end_of_run_results[0].buffer_id).is_ok());
}

// ---------------------------------------------------------------------------
// Physical U5b preflight.  This intentionally stops before Metal dispatch when
// the exported producer facts cannot carry the declared KV state or when the
// emitted module's buffer ABI disagrees with those facts.  A fake launch cannot
// discharge either red, and this test must never turn either one into a CPU
// substitute or an unmeasured physical receipt.
// ---------------------------------------------------------------------------

fn gea3_unmeasured(reason: &str) -> Value {
    json!({"value": Value::Null, "status": "unmeasured", "reason": reason})
}

fn gea3_git_revision(path: &Path) -> String {
    let output = Command::new("git")
        .args(["-C"])
        .arg(path)
        .args(["rev-parse", "HEAD"])
        .output()
        .unwrap_or_else(|error| panic!("read Git revision for {}: {error}", path.display()));
    assert!(
        output.status.success(),
        "Git revision failed for {}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_owned()
}

fn gea3_emitted_buffer_arity(module_image: &[u8], entry: &str) -> Option<usize> {
    let source = std::str::from_utf8(module_image).ok()?;
    let marker = format!("kernel void {entry}(");
    let start = source.find(&marker)?;
    let function_header = &source[start..];
    let end = function_header.find(") {")?;
    Some(function_header[..end].matches("[[buffer(").count())
}

fn gea3_physical_plan_reds(
    envelope: &Gea3ProgramPlanEnvelope,
    prefill: &DeviceDescriptor,
    decode: &DeviceDescriptor,
) -> Vec<String> {
    let mut reds = Vec::new();
    for (program_name, program, descriptor) in [
        ("prefill", &envelope.programs.prefill, prefill),
        ("decode-step", &envelope.programs.decode_step, decode),
    ] {
        let mut checked_entries = BTreeSet::new();
        for kernel in &program.kernels {
            if !checked_entries.insert(kernel.entry.clone()) {
                continue;
            }
            let Some(emitted) = gea3_emitted_buffer_arity(&descriptor.module_image, &kernel.entry)
            else {
                reds.push(format!(
                    "{program_name} entry `{}` is absent from the emitted Metal module",
                    kernel.entry
                ));
                continue;
            };
            if emitted != kernel.resources.len() {
                reds.push(format!(
                    "{program_name} entry `{}` carries {} plan resources but emitted MSL declares {emitted} buffer arguments",
                    kernel.entry,
                    kernel.resources.len()
                ));
            }
        }
    }

    let decode_resource_names: BTreeSet<&str> = envelope
        .programs
        .decode_step
        .kernels
        .iter()
        .flat_map(|kernel| kernel.resources.iter())
        .map(|resource| resource.buffer.name.as_str())
        .collect();
    let missing_kv: Vec<String> = envelope
        .programs
        .decode_step
        .state_buffers
        .iter()
        .filter_map(|state| state.name.as_deref())
        .filter(|name| name.contains(".kv_"))
        .filter(|name| !decode_resource_names.contains(name))
        .map(str::to_owned)
        .collect();
    if !missing_kv.is_empty() {
        reds.push(format!(
            "decode-step declares {} fixed-capacity KV state buffers, but none are carried as launch resources (first missing: {})",
            missing_kv.len(),
            missing_kv.first().expect("non-empty missing KV row")
        ));
    }
    reds
}

#[derive(Debug)]
struct Gea3PhysicalProgram {
    descriptor: DeviceDescriptor,
    /// GEA3-A1: the per-kernel sub-window bindings of this program —
    /// (kernel index, binding index) → (byte offset, view span).  A slot
    /// without an entry binds its whole handle.
    windows: Gea3WindowBindings,
    module: DeviceHandle,
    buffers: BTreeMap<(u32, u32), DeviceHandle>,
    output: (u32, u32),
    weight_alloc_us: u64,
    kv_alloc_us: u64,
}

fn gea3_elapsed_us(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX)
}

fn gea3_canonical_model_name(name: &str) -> Option<String> {
    let name = if let Some(name) = name.strip_prefix("plan_extra_") {
        if name.ends_with("_token_embd.weight") {
            return Some("token_embd.weight".to_owned());
        }
        if name.ends_with("_output_norm.weight") {
            return Some("output_norm.weight".to_owned());
        }
        let marker = name.find("_blk.")?;
        &name[marker + 1..]
    } else {
        name
    };
    if name == "token_embd.weight" || name == "output_norm.weight" {
        return Some(name.to_owned());
    }
    let rest = name.strip_prefix("blk.")?;
    let (layer, suffix) = rest.split_once('.')?;
    if !suffix.ends_with(".weight") {
        return None;
    }
    let layer = layer.parse::<u32>().ok()?;
    Some(format!("blk.{layer}.{suffix}"))
}

fn gea3_model_ranges(manifest: &Value) -> Result<BTreeMap<String, (u64, u64)>, String> {
    let tensors = manifest["tensors"]
        .as_array()
        .ok_or_else(|| "GEA3 input manifest has no tensor table".to_owned())?;
    let mut ranges = BTreeMap::new();
    for tensor in tensors {
        let name = tensor["name"]
            .as_str()
            .ok_or_else(|| "GEA3 tensor is missing its name".to_owned())?
            .to_owned();
        let range = tensor["absolute_range"]
            .as_array()
            .ok_or_else(|| format!("GEA3 tensor `{name}` is missing its absolute range"))?;
        let start = range
            .first()
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("GEA3 tensor `{name}` has no range start"))?;
        let end = range
            .get(1)
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("GEA3 tensor `{name}` has no range end"))?;
        if end <= start || ranges.insert(name.clone(), (start, end)).is_some() {
            return Err(format!(
                "GEA3 tensor `{name}` has an invalid or duplicate range"
            ));
        }
    }
    if ranges.len() != 290 {
        return Err(format!(
            "GEA3 input manifest carries {} tensors, expected 290",
            ranges.len()
        ));
    }
    Ok(ranges)
}

fn gea3_unique_slots(descriptor: &DeviceDescriptor) -> BTreeMap<(u32, u32), DescriptorBuffer> {
    let mut slots = BTreeMap::new();
    for kernel in &descriptor.kernels {
        for slot in &kernel.buffers {
            slots
                .entry((slot.buffer_id, slot.version))
                .or_insert_with(|| slot.clone());
        }
    }
    slots
}

/// GEA3-A1: one launch binding for a kernel slot — the carried sub-window's
/// byte offset and view span when the plan declares one, the whole handle
/// otherwise.  The B4 launch-binding surface carries (handle, binding
/// index, byte offset, view span, source); the Metal session bounds-checks
/// offset + span against the allocation before dispatch.
fn gea3_launch_binding(
    handle: DeviceHandle,
    program: &Gea3PhysicalProgram,
    kernel_index: u32,
    slot: &DescriptorBuffer,
) -> Result<DeviceLaunchBinding, String> {
    match program.windows.get(&(kernel_index, slot.binding)).copied() {
        Some((byte_offset, view_span)) => Ok(DeviceLaunchBinding {
            handle,
            binding_index: slot.binding,
            byte_offset,
            view_span,
            runtime_source: DescriptorRuntimeSource::Constant,
        }),
        None => DeviceLaunchBinding::whole_handle(handle, slot.binding)
            .map_err(|error| error.message.clone()),
    }
}

fn gea3_prepare_physical_program(
    runtime: &mut DeviceRuntime,
    descriptor: DeviceDescriptor,
    windows: Gea3WindowBindings,
    shared: &mut BTreeMap<String, DeviceHandle>,
) -> Result<Gea3PhysicalProgram, String> {
    let module = runtime
        .load_module(&descriptor.module_image)
        .map_err(|error| error.message.clone())?;
    let slots = gea3_unique_slots(&descriptor);
    let mut buffers = BTreeMap::new();
    let mut weight_alloc_us = 0_u64;
    let mut kv_alloc_us = 0_u64;
    let result = (|| {
        for (key, slot) in slots {
            let shared_key = if let Some(weight) = gea3_canonical_model_name(&slot.buffer_name) {
                Some(format!("weight:{weight}"))
            } else if slot.buffer_name.starts_with("blk.") && slot.buffer_name.contains(".kv_") {
                Some(format!("kv:{}", slot.buffer_name))
            } else {
                None
            };
            let handle = if let Some(shared_key) = shared_key {
                if let Some(handle) = shared.get(&shared_key).copied() {
                    handle
                } else {
                    let bytes = usize::try_from(
                        slot.element_count
                            .checked_mul(slot.element_ty.byte_width() as u64)
                            .ok_or_else(|| {
                                format!("buffer `{}` byte length overflows", slot.buffer_name)
                            })?,
                    )
                    .map_err(|_| format!("buffer `{}` is too large", slot.buffer_name))?;
                    let started = Instant::now();
                    let handle = runtime
                        .alloc_bytes(bytes)
                        .map_err(|error| error.message.clone())?;
                    let elapsed = gea3_elapsed_us(started);
                    if shared_key.starts_with("weight:") {
                        weight_alloc_us = weight_alloc_us.saturating_add(elapsed);
                    } else {
                        kv_alloc_us = kv_alloc_us.saturating_add(elapsed);
                    }
                    shared.insert(shared_key, handle);
                    handle
                }
            } else {
                let bytes = usize::try_from(
                    slot.element_count
                        .checked_mul(slot.element_ty.byte_width() as u64)
                        .ok_or_else(|| {
                            format!("buffer `{}` byte length overflows", slot.buffer_name)
                        })?,
                )
                .map_err(|_| format!("buffer `{}` is too large", slot.buffer_name))?;
                runtime
                    .alloc_bytes(bytes)
                    .map_err(|error| error.message.clone())?
            };
            buffers.insert(key, handle);
        }
        let output = descriptor
            .end_of_run_results
            .first()
            .map(|result| (result.buffer_id, result.version))
            .ok_or_else(|| "GEA3 program declares no logits observation".to_owned())?;
        if !buffers.contains_key(&output) {
            return Err(format!(
                "GEA3 logits observation names unallocated buffer {} version {}",
                output.0, output.1
            ));
        }
        Ok(Gea3PhysicalProgram {
            descriptor,
            windows,
            module,
            buffers: buffers.clone(),
            output,
            weight_alloc_us,
            kv_alloc_us,
        })
    })();
    if result.is_err() {
        let mut released = BTreeSet::new();
        for handle in buffers.values().copied().chain(std::iter::once(module)) {
            if released.insert(handle.id) {
                drop(runtime.release(&handle));
            }
        }
    }
    result
}

fn gea3_f32_bytes(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn gea3_f32_from_bytes(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "F32 readback has {} non-word-aligned bytes",
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn gea3_zero_handle(runtime: &mut DeviceRuntime, handle: &DeviceHandle) -> Result<(), String> {
    let bytes = usize::try_from(
        handle
            .len_bytes()
            .ok_or_else(|| "buffer has no byte length".to_owned())?,
    )
    .map_err(|_| "buffer byte length overflows host usize".to_owned())?;
    runtime
        .copy_in_bytes(handle, &vec![0; bytes], DeviceDataType::U8)
        .map_err(|error| error.message.clone())
}

fn gea3_copy_f32(
    runtime: &mut DeviceRuntime,
    handle: &DeviceHandle,
    values: &[f32],
) -> Result<(), String> {
    let expected = handle
        .len_bytes()
        .ok_or_else(|| "buffer has no byte length".to_owned())?;
    if expected != u64::try_from(values.len() * 4).unwrap_or(u64::MAX) {
        return Err(format!(
            "dynamic input has {} bytes but device buffer has {expected}",
            values.len() * 4
        ));
    }
    runtime
        .copy_in_bytes(handle, gea3_f32_bytes(values), DeviceDataType::F32)
        .map_err(|error| error.message.clone())
}

/// Observed dynamic-input uploads for one step (PPB-U3): the copy count the
/// GEA3 receipt has always carried, plus the staged byte total the parity
/// companion reports as a host-to-device transfer observation.
struct Gea3InputUploads {
    copies: usize,
    bytes: u64,
}

fn gea3_update_inputs(
    runtime: &mut DeviceRuntime,
    program: &Gea3PhysicalProgram,
    tokens: &[u32],
    position: u32,
    valid_len: u32,
    prefill: bool,
    capacity: u64,
) -> Result<Gea3InputUploads, String> {
    let mut copied = BTreeSet::new();
    let mut uploaded_bytes = 0_u64;
    let mut copy = |handle: DeviceHandle, values: Vec<f32>| -> Result<(), String> {
        if copied.insert(handle.id) {
            uploaded_bytes =
                uploaded_bytes.saturating_add(u64::try_from(values.len() * 4).unwrap_or(u64::MAX));
            gea3_copy_f32(runtime, &handle, &values)?;
        }
        Ok(())
    };
    for kernel in &program.descriptor.kernels {
        for slot in &kernel.buffers {
            if slot.role != DeviceBufferRole::Input {
                continue;
            }
            let handle = *program
                .buffers
                .get(&(slot.buffer_id, slot.version))
                .ok_or_else(|| "dynamic input handle disappeared".to_owned())?;
            if (kernel.entry == "embedding_gather" || kernel.entry == "prefill_embedding_gather")
                && slot.binding == 1
            {
                // PGC-R1/PGC-R3: compact token-id binding — the gather
                // consumes one u32 id per row (staged through the statue's
                // F32 element path, 4 bytes per id), never the superseded
                // [rows, VOCAB] one-hot selector.
                let rows = if prefill { tokens.len() } else { 1 };
                if slot.element_count != rows as u64 {
                    return Err(format!(
                        "embedding ids declare {} values, expected {rows}",
                        slot.element_count
                    ));
                }
                let ids: Vec<f32> = tokens[..rows]
                    .iter()
                    .map(|token| {
                        let token = usize::try_from(*token)
                            .map_err(|_| "token id does not fit host usize".to_owned())?;
                        if token >= VOCAB as usize {
                            return Err(format!("token id {token} is outside vocab {VOCAB}"));
                        }
                        Ok(f32::from_bits(u32::try_from(token).map_err(|_| "token id does not fit u32".to_owned())?))
                    })
                    .collect::<Result<Vec<f32>, String>>()?;
                copy(handle, ids)?;
            } else if (kernel.entry == "embedding_gather" || kernel.entry == "prefill_embedding_gather")
                && slot.binding == 0
            {
                // The resident [VOCAB, 960] table is weight-shaped and
                // once-resident (PGC-C5); it is never staged per step here.
                continue;
            } else if (kernel.entry == "prefill_rope_q" || kernel.entry == "prefill_rope_k")
                && slot.binding == 1
                || (kernel.entry == "decode_rope_q" || kernel.entry == "decode_rope_k")
                    && slot.binding == 1
            {
                let positions: Vec<u32> = if prefill {
                    (0..u32::try_from(tokens.len()).map_err(|_| "prompt is too long".to_owned())?)
                        .collect()
                } else {
                    vec![position]
                };
                let mut table = Vec::with_capacity(positions.len() * 32 * 3);
                for position in positions {
                    for pair in 0..32 {
                        let angle =
                            f64::from(position) * 100_000.0_f64.powf(-(2.0 * pair as f64) / 64.0);
                        table.extend([0.0, angle.cos() as f32, angle.sin() as f32]);
                    }
                }
                copy(handle, table)?;
            } else if (kernel.entry == "prefill_score_gemm" || kernel.entry == "decode_score_gemm")
                && slot.binding == 2
            {
                let count = usize::try_from(slot.element_count)
                    .map_err(|_| "attention scale count overflows host usize".to_owned())?;
                copy(handle, vec![0.125; count])?;
            } else if kernel.entry == "decode_masked_softmax" && slot.binding == 1 {
                let count = usize::try_from(slot.element_count)
                    .map_err(|_| "decode mask count overflows host usize".to_owned())?;
                let valid = usize::try_from(valid_len).unwrap_or(usize::MAX).min(count);
                let mut mask = vec![f32::NEG_INFINITY; count];
                mask[..valid].fill(0.0);
                copy(handle, mask)?;
            } else if kernel.entry == "prefill_causal_softmax" && slot.binding == 1 {
                // GEA3-U6 num-9: the resident causal mask — 0 at or below
                // the diagonal, negative infinity above it (the
                // decode_masked_softmax additive idiom).  The [36,36]
                // count is the frozen prompt geometry.
                let count = usize::try_from(slot.element_count)
                    .map_err(|_| "prefill causal mask count overflows host usize".to_owned())?;
                let extent = (count as f64).sqrt();
                if extent.fract() != 0.0 || extent < 1.0 {
                    return Err(format!(
                        "prefill causal mask count {count} is not a square extent"
                    ));
                }
                let extent = extent as usize;
                let mut mask = vec![f32::NEG_INFINITY; count];
                for row in 0..extent {
                    let row_start = row * extent;
                    let causal = (row + 1).min(extent);
                    mask[row_start..row_start + causal].fill(0.0);
                }
                copy(handle, mask)?;
            } else if (kernel.entry == "kv_append_k" || kernel.entry == "kv_append_v")
                && slot.binding == 1
            {
                // GEA3-U6 num-1: the resident one-hot row selector at the
                // decode position — `slot · row` writes the incoming [1,320]
                // row into exactly that history slot.
                let count = usize::try_from(slot.element_count)
                    .map_err(|_| "append slot count overflows host usize".to_owned())?;
                let position = usize::try_from(position)
                    .map_err(|_| "decode position overflows host usize".to_owned())?;
                if position >= count {
                    return Err(format!(
                        "decode position {position} is outside the {}-slot arena",
                        count
                    ));
                }
                let mut slot_values = vec![0.0; count];
                slot_values[position] = 1.0;
                copy(handle, slot_values)?;
            } else if (kernel.entry == "prefill_kv_write_k" || kernel.entry == "prefill_kv_write_v")
                && slot.binding == 1
            {
                // The resident [76,36] 0/1 block indicator: rows 0..35
                // select the incoming rope'd K (or raw V) rows.
                let count = usize::try_from(slot.element_count)
                    .map_err(|_| "prefill block count overflows host usize".to_owned())?;
                let mut block = vec![0.0; count];
                let columns = count / capacity as usize;
                for diagonal in 0..columns.min(capacity as usize) {
                    block[diagonal * columns + diagonal] = 1.0;
                }
                copy(handle, block)?;
            }
        }
    }
    Ok(Gea3InputUploads {
        copies: copied.len(),
        bytes: uploaded_bytes,
    })
}

fn gea3_diagnostic_buffer_summary(
    runtime: &mut DeviceRuntime,
    handle: &DeviceHandle,
    slot: &Gea3DeviceResource,
) -> Result<Value, String> {
    let bytes = runtime
        .readback_bytes(handle, DeviceDataType::F32)
        .map_err(|error| error.message.clone())?;
    let values = gea3_f32_from_bytes(&bytes)?;
    let mut finite = 0usize;
    let mut non_zero = 0usize;
    let mut nan_count = 0usize;
    let mut pos_inf_count = 0usize;
    let mut neg_inf_count = 0usize;
    let mut first_non_finite = None;
    let mut min = None;
    let mut max = None;
    let mut abs_sum = 0.0_f64;
    for (index, value) in values.iter().copied().enumerate() {
        if value.is_finite() {
            finite += 1;
            min = Some(min.map_or(value, |previous: f32| previous.min(value)));
            max = Some(max.map_or(value, |previous: f32| previous.max(value)));
            abs_sum += f64::from(value.abs());
        } else {
            if value.is_nan() {
                nan_count += 1;
            } else if value > 0.0 {
                pos_inf_count += 1;
            } else {
                neg_inf_count += 1;
            }
            if first_non_finite.is_none() {
                first_non_finite = Some(json!({
                    "index": index,
                    "value": format!("{value:?}"),
                }));
            }
        }
        if value != 0.0 {
            non_zero += 1;
        }
    }
    let first_values = values
        .iter()
        .take(8)
        .map(|value| {
            if value.is_finite() {
                json!(*value)
            } else {
                json!(format!("{value:?}"))
            }
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "buffer_id": slot.buffer.id,
        "version": slot.version.version,
        "name": slot.buffer.name,
        "role": format!("{:?}", slot.buffer.role),
        "access": format!("{:?}", slot.access),
        "element_count": slot.version.element_count,
        "readback_bytes": bytes.len(),
        "sha256": gea3_readback_hash(&values),
        "finite_count": finite,
        "non_finite_count": nan_count + pos_inf_count + neg_inf_count,
        "non_finite": nan_count + pos_inf_count + neg_inf_count != 0,
        "nan_count": nan_count,
        "pos_inf_count": pos_inf_count,
        "neg_inf_count": neg_inf_count,
        "first_non_finite": first_non_finite,
        "non_zero_count": non_zero,
        "non_zero": non_zero != 0,
        "min": min,
        "max": max,
        "mean_abs": if values.is_empty() {
            0.0
        } else {
            abs_sum / values.len() as f64
        },
        "first_values": first_values,
    }))
}

fn gea3_diagnostic_model_upload(
    runtime: &mut DeviceRuntime,
    program: &Gea3PhysicalProgram,
    shared: &BTreeMap<String, DeviceHandle>,
    model_ranges: &BTreeMap<String, (u64, u64)>,
    mapped: &MappedWeightFile,
) -> Result<(u64, usize), String> {
    let mut weight_bytes = 0_u64;
    let mut weight_uploads = 0usize;
    for (name, handle) in shared
        .iter()
        .filter_map(|(key, handle)| key.strip_prefix("weight:").map(|name| (name, *handle)))
    {
        let (start, end) = model_ranges
            .get(name)
            .copied()
            .ok_or_else(|| format!("no frozen GGUF range for model tensor `{name}`"))?;
        let start =
            usize::try_from(start).map_err(|_| format!("tensor `{name}` offset overflows"))?;
        let end = usize::try_from(end).map_err(|_| format!("tensor `{name}` end overflows"))?;
        let bytes = mapped
            .bytes()
            .get(start..end)
            .ok_or_else(|| format!("tensor `{name}` range is outside GGUF"))?;
        let expected = handle
            .len_bytes()
            .ok_or_else(|| format!("tensor `{name}` handle has no byte length"))?;
        if expected != bytes.len() as u64 {
            return Err(format!(
                "tensor `{name}` declares {expected} bytes but GGUF carries {}",
                bytes.len()
            ));
        }
        runtime
            .copy_in_bytes(&handle, bytes, DeviceDataType::U8)
            .map_err(|error| error.message.clone())?;
        weight_bytes = weight_bytes.saturating_add(bytes.len() as u64);
        weight_uploads += 1;
    }
    if weight_uploads != 290 {
        return Err(format!(
            "uploaded {weight_uploads} model weights, expected 290"
        ));
    }

    // A diagnostic run must not inherit allocator contents for the many
    // synthetic input/output slots in the exported composition.  Model
    // weights are uploaded above; every other slot starts from an explicit
    // zero so a zero readback names the route rather than stale device data.
    let slots = gea3_unique_slots(&program.descriptor);
    let mut zeroed = BTreeSet::new();
    for (key, handle) in &program.buffers {
        let slot = slots
            .get(key)
            .ok_or_else(|| "diagnostic slot metadata disappeared".to_owned())?;
        if gea3_canonical_model_name(&slot.buffer_name).is_some() {
            continue;
        }
        if zeroed.insert(handle.id) {
            gea3_zero_handle(runtime, handle)?;
        }
    }
    Ok((weight_bytes, weight_uploads))
}

fn gea3_diagnostic_resource_is_weight(resource: &Gea3DeviceResource) -> bool {
    gea3_canonical_model_name(&resource.buffer.name).is_some()
}

fn gea3_diagnostic_read_slots(
    runtime: &mut DeviceRuntime,
    program: &Gea3PhysicalProgram,
    kernel: &Gea3KernelUnit,
    input: bool,
) -> Result<Vec<Value>, String> {
    kernel
        .resources
        .iter()
        .filter(|resource| {
            if gea3_diagnostic_resource_is_weight(resource) {
                return false;
            }
            if input {
                matches!(
                    resource.access,
                    Gea3ResourceAccess::Read | Gea3ResourceAccess::ReadWrite
                )
            } else {
                matches!(
                    resource.access,
                    Gea3ResourceAccess::Write | Gea3ResourceAccess::ReadWrite
                )
            }
        })
        .map(|resource| {
            let handle = program
                .buffers
                .get(&(resource.buffer.id, resource.version.version))
                .ok_or_else(|| {
                    format!(
                        "diagnostic buffer {} version {} disappeared",
                        resource.buffer.id, resource.version.version
                    )
                })?;
            gea3_diagnostic_buffer_summary(runtime, handle, resource)
        })
        .collect()
}

fn gea3_diagnostic_primary_slot<'a>(
    kernel: &'a Gea3KernelUnit,
    input: bool,
) -> Option<&'a Gea3DeviceResource> {
    // Prefer the direct read edge.  A ReadWrite KV arena is a side effect of
    // the K/V launch, not the activation consumed by the entry.  Falling back
    // to ReadWrite keeps the helper useful for entries whose only input is a
    // state arena.
    kernel
        .resources
        .iter()
        .filter(|resource| !gea3_diagnostic_resource_is_weight(resource))
        .find(|resource| {
            if input {
                resource.access == Gea3ResourceAccess::Read
                    && resource.buffer.role != Gea3BufferRole::Input
            } else {
                resource.access == Gea3ResourceAccess::Write
            }
        })
        .or_else(|| {
            kernel.resources.iter().find(|resource| {
                !gea3_diagnostic_resource_is_weight(resource)
                    && if input {
                        resource.access == Gea3ResourceAccess::Read
                    } else {
                        resource.access == Gea3ResourceAccess::Write
                    }
            })
        })
        .or_else(|| {
            kernel.resources.iter().find(|resource| {
                !gea3_diagnostic_resource_is_weight(resource)
                    && if input {
                        matches!(
                            resource.access,
                            Gea3ResourceAccess::Read | Gea3ResourceAccess::ReadWrite
                        )
                    } else {
                        matches!(
                            resource.access,
                            Gea3ResourceAccess::Write | Gea3ResourceAccess::ReadWrite
                        )
                    }
            })
        })
}

fn gea3_diagnostic_summary_non_zero(summary: &Value) -> bool {
    summary["non_zero"].as_bool().unwrap_or(false)
}

fn gea3_diagnostic_expected_buffer(
    descriptor: &DeviceDescriptor,
    launch_id: u32,
) -> Option<(u32, u32)> {
    descriptor
        .data_flow
        .iter()
        .find(|edge| edge.consumer == launch_id)
        .map(|edge| (edge.producer, edge.buffer_id))
}

fn gea3_launch_rows(descriptor: &DeviceDescriptor) -> Vec<Value> {
    descriptor
        .launches
        .iter()
        .map(|launch| {
            let kernel = &descriptor.kernels[launch.kernel_index as usize];
            json!({
                "id": launch.id,
                "kernel_index": launch.kernel_index,
                "entry": kernel.entry,
                "grid": kernel.grid,
                "block": kernel.block,
                "binding_indices": kernel.buffers.iter().map(|buffer| buffer.binding).collect::<Vec<_>>(),
                "buffer_ids": kernel.buffers.iter().map(|buffer| buffer.buffer_id).collect::<Vec<_>>(),
            })
        })
        .collect()
}

fn gea3_data_flow_satisfied(descriptor: &DeviceDescriptor) -> bool {
    let positions: BTreeMap<u32, usize> = descriptor
        .launches
        .iter()
        .enumerate()
        .map(|(index, launch)| (launch.id, index))
        .collect();
    descriptor.data_flow.iter().all(|edge| {
        positions
            .get(&edge.producer)
            .zip(positions.get(&edge.consumer))
            .is_some_and(|(producer, consumer)| producer < consumer)
    })
}

fn gea3_readback_hash(values: &[f32]) -> String {
    sha256_hex(gea3_f32_bytes(values))
}

fn gea3_argmax(values: &[f32]) -> Result<u32, String> {
    let mut best = None;
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(format!("logits contain non-finite value at index {index}"));
        }
        if best.is_none_or(|(_, previous)| value > previous) {
            best = Some((index, value));
        }
    }
    best.map(|(index, _)| index as u32)
        .ok_or_else(|| "logits readback is empty".to_owned())
}

fn gea3_release_programs(
    runtime: &mut DeviceRuntime,
    programs: &mut Vec<Gea3PhysicalProgram>,
) -> Result<(), String> {
    let mut handles = BTreeMap::new();
    for program in programs.drain(..) {
        handles.insert(program.module.id, program.module);
        for handle in program.buffers.into_values() {
            handles.insert(handle.id, handle);
        }
    }
    let mut first_error = None;
    for handle in handles.into_values() {
        if let Err(error) = runtime.release(&handle) {
            first_error.get_or_insert(error.message);
        }
    }
    first_error.map_or(Ok(()), Err)
}

// ---------------------------------------------------------------------------
// PPB-U3: the optional parity timing companion (gea3-parity-timing-
// companion-v1).  The companion is disjoint from `gea3-metal-receipt-v1`:
// the physical test emits the versioned receipt and additionally emits this
// companion when the environment names an output path.  Every phase is
// directly measured on its own clock and the
// boundaries never overlap: the launch/encode clock ends before the queue
// wait begins, and the queue wait ends at the declared completion point
// (the explicit step `sync` return).  No category is a total minus other
// terms, and a unified-memory allocation is never called a transfer without
// an observed copy event.
// ---------------------------------------------------------------------------

/// One directly measured host-wall phase: the phase clock's own duration
/// plus the phase boundaries on the step clock.  The duration is always an
/// independent `elapsed` reading of the phase clock, never a difference of
/// the two boundaries.
#[derive(Debug, Clone, Copy)]
struct Gea3ParityPhase {
    start_since_step_us: u64,
    end_since_step_us: u64,
    duration_us: u64,
}

/// Per-step companion observations (prefill step 0, then each decode step).
#[derive(Debug, Clone)]
struct Gea3ParityStep {
    mode: &'static str,
    step: u64,
    input_upload: Gea3ParityPhase,
    input_upload_bytes: u64,
    input_upload_copies: usize,
    launch_encode: Gea3ParityPhase,
    queue_wait: Gea3ParityPhase,
    gpu_encoder_us: Option<u64>,
    gpu_timestamp_count: usize,
    gpu_start_timestamp_count: usize,
    gpu_total_encoder_count: usize,
    command_submits: usize,
    blocking_waits: usize,
    readback: Gea3ParityPhase,
    readback_bytes: u64,
}

/// Run-level residency and transfer observations collected by the physical
/// run before the prefill/decode steps.
struct Gea3ParityResidency {
    weight_alloc_us: u64,
    weight_upload_us: u64,
    weight_bytes: u64,
    weight_uploads: u64,
    kv_alloc_us: u64,
    kv_zero_fill_us: u64,
    kv_zero_fill_bytes: u64,
    other_zero_fills: u64,
}

/// The complete PPB-U3 observation set handed to the companion builder.
struct Gea3ParityTiming {
    residency: Gea3ParityResidency,
    steps: Vec<Gea3ParityStep>,
}

fn gea3_parity_measured(value: u64, basis: &str) -> Value {
    json!({"value": value, "status": "measured", "basis": basis})
}

fn gea3_parity_count(value: u64, basis: &str) -> Value {
    json!({"value": value, "status": "measured", "basis": basis})
}

fn gea3_gpu_timing_cell(
    value: Option<u64>,
    sampled_encoders: usize,
    total_encoders: usize,
    basis: &str,
) -> Value {
    let (value, status) = match value {
        Some(value) => (json!(value), "measured"),
        None => (Value::Null, "not_measured"),
    };
    json!({
        "value": value,
        "status": status,
        "sampled_encoders": sampled_encoders,
        "total_encoders": total_encoders,
        "coverage": format!("{sampled_encoders}/{total_encoders}"),
        "reason": if status == "not_measured" {
            json!("the driver sampled no GPU timestamps for this step; no encoder time is inferred")
        } else {
            Value::Null
        },
        "basis": basis,
    })
}

fn gea3_gpu_count_cell(value: usize, total_encoders: usize, basis: &str) -> Value {
    json!({
        "value": value,
        "status": "measured",
        "sampled_encoders": value,
        "total_encoders": total_encoders,
        "coverage": format!("{value}/{total_encoders}"),
        "basis": basis,
    })
}

fn gea3_gpu_not_measured(reason: &str, total_encoders: usize) -> Value {
    let mut cell = gea3_gpu_timing_cell(
        None,
        0,
        total_encoders,
        "no GPU timestamp duration was observed",
    );
    cell["reason"] = json!(reason);
    cell
}

fn gea3_parity_not_measured(reason: &str) -> Value {
    json!({"value": Value::Null, "status": "not_measured", "reason": reason})
}

fn gea3_parity_not_observable(reason: &str) -> Value {
    json!({"value": Value::Null, "status": "not_observable", "reason": reason})
}

fn gea3_parity_boundaries(phase: &Gea3ParityPhase) -> Value {
    json!({
        "start_us_since_step_start": phase.start_since_step_us,
        "end_us_since_step_start": phase.end_since_step_us,
        "status": "measured",
        "basis": "monotonic step clock read at each phase boundary",
    })
}

/// The non-overlap law, fail closed: the input-upload clock ends no later
/// than the launch/encode clock begins, the launch/encode clock ends before
/// the queue wait begins, and the queue wait ends at the declared
/// completion point before the readback begins.
fn gea3_parity_admit_step_boundaries(step: &Gea3ParityStep) -> Result<(), String> {
    let name = format!("{} step {}", step.mode, step.step);
    if step.input_upload.end_since_step_us > step.launch_encode.start_since_step_us {
        return Err(format!(
            "{name}: input-upload boundary {} overlaps launch/encode start {}",
            step.input_upload.end_since_step_us, step.launch_encode.start_since_step_us
        ));
    }
    if step.launch_encode.end_since_step_us > step.queue_wait.start_since_step_us {
        return Err(format!(
            "{name}: launch/encode clock ends at {} but queue wait begins at {}; the encode clock must end before the queue wait begins",
            step.launch_encode.end_since_step_us, step.queue_wait.start_since_step_us
        ));
    }
    if step.queue_wait.end_since_step_us > step.readback.start_since_step_us {
        return Err(format!(
            "{name}: queue wait ends at {} but readback begins at {}; the queue wait must end at the declared completion point first",
            step.queue_wait.end_since_step_us, step.readback.start_since_step_us
        ));
    }
    Ok(())
}

fn gea3_parity_step_json(step: &Gea3ParityStep) -> Value {
    let gpu_encoder = gea3_gpu_timing_cell(
        step.gpu_encoder_us,
        step.gpu_timestamp_count,
        step.gpu_total_encoder_count,
        "sum of sampled per-encoder device GPU timestamps for this step (FABER_PER_OP_TIMING); coverage is explicit",
    );
    json!({
        "mode": step.mode,
        "step": step.step,
        "host_input_upload": {
            "duration_us": gea3_parity_measured(step.input_upload.duration_us, "phase clock around dynamic-input staging copy_in_bytes (host to device)"),
            "bytes": gea3_parity_measured(step.input_upload_bytes, "sum of staged f32 payloads actually copied in"),
            "copies": gea3_parity_count(step.input_upload_copies as u64, "distinct device handles staged this step"),
            "transfer": {"status": "measured", "basis": "observed copy_in_bytes events into regular allocations"},
            "boundary": gea3_parity_boundaries(&step.input_upload),
        },
        "host_launch_encode": {
            "duration_us": gea3_parity_measured(step.launch_encode.duration_us, "launch encode wall; this clock ends before the queue wait begins"),
            "boundary": gea3_parity_boundaries(&step.launch_encode),
        },
        "queue_wait": {
            "duration_us": gea3_parity_measured(step.queue_wait.duration_us, "explicit step sync wall (command-buffer commit plus wait, one DeviceRuntime call)"),
            "ends_at": {"value": "explicit runtime.sync() return — the declared completion point", "status": "measured"},
            "blocking_waits": gea3_parity_count(step.blocking_waits as u64, "blocking waits inside this phase"),
            "boundary": gea3_parity_boundaries(&step.queue_wait),
        },
        "gpu_encoder": {
            "duration_us": gpu_encoder,
            "timestamp_count": gea3_gpu_count_cell(step.gpu_timestamp_count, step.gpu_total_encoder_count, "sampled per-encoder GPU end timestamps"),
            "gpu_start_timestamp_count": gea3_gpu_count_cell(step.gpu_start_timestamp_count, step.gpu_total_encoder_count, "sampled per-encoder GPU start timestamps"),
            "clock": "device GPU timestamps (independent of the host wall phases)",
            "coverage": {
                "sampled_encoders": step.gpu_timestamp_count,
                "total_encoders": step.gpu_total_encoder_count,
                "fraction": format!("{}/{}", step.gpu_timestamp_count, step.gpu_total_encoder_count),
            },
        },
        "command_submits": {
            "value": step.command_submits,
            "status": "measured",
            "basis": "pending command-buffer commits at the step boundary",
        },
        "command_submit_wall": gea3_parity_not_observable(
            "commit and waitUntilCompleted are one DeviceRuntime::sync call on this surface; a submit-only wall cannot be separated without inferring it",
        ),
        "device_to_host_readback": {
            "duration_us": gea3_parity_measured(step.readback.duration_us, "phase clock around the declared logits readback"),
            "bytes": gea3_parity_measured(step.readback_bytes, "declared logits observation bytes read back"),
            "transfer": {"status": "measured", "basis": "observed copy_out event after the step flush"},
            "boundary": gea3_parity_boundaries(&step.readback),
        },
    })
}

fn gea3_parity_summed(label: &str, values: &[u64], total_steps: usize) -> Value {
    if values.is_empty() {
        return gea3_parity_not_measured(&format!("no measured {label} observations to sum"));
    }
    json!({
        "value": values.iter().copied().sum::<u64>(),
        "status": "derived",
        "steps_measured": values.len(),
        "steps_not_measured": total_steps - values.len(),
        "basis": format!("sum of directly measured per-step {label} phases; no term is a total minus other terms"),
    })
}

fn gea3_parity_gpu_summed(steps: &[&Gea3ParityStep]) -> Value {
    let values: Vec<u64> = steps.iter().filter_map(|step| step.gpu_encoder_us).collect();
    let sampled_encoders: Vec<usize> = steps
        .iter()
        .map(|step| step.gpu_timestamp_count)
        .collect();
    let total_encoders: Vec<usize> = steps
        .iter()
        .map(|step| step.gpu_total_encoder_count)
        .collect();
    let status = if values.is_empty() {
        "not_measured"
    } else {
        "derived"
    };
    json!({
        "value": values.first().map(|_| values.iter().copied().sum::<u64>()),
        "status": status,
        "steps_measured": values.len(),
        "steps_not_measured": steps.len() - values.len(),
        "sampled_encoders_per_step": sampled_encoders,
        "total_encoders_per_step": total_encoders,
        "coverage_per_step": steps.iter().map(|step| format!("{}/{}", step.gpu_timestamp_count, step.gpu_total_encoder_count)).collect::<Vec<_>>(),
        "basis": "sum of sampled per-encoder GPU timestamps; each step carries its sampled/total encoder coverage",
    })
}

/// Build the versioned companion from directly measured observations.  The
/// boundary law is admitted fail-closed; nothing here subtracts a total and
/// every absent fact keeps an explicit status.
fn gea3_build_parity_companion(
    timing: &Gea3ParityTiming,
    source_receipt: &str,
) -> Result<Value, String> {
    for step in &timing.steps {
        gea3_parity_admit_step_boundaries(step)?;
    }
    let prefill_steps: Vec<&Gea3ParityStep> = timing
        .steps
        .iter()
        .filter(|step| step.mode == "prefill")
        .collect();
    let decode_steps: Vec<&Gea3ParityStep> = timing
        .steps
        .iter()
        .filter(|step| step.mode == "decode")
        .collect();
    let summarize = |steps: &[&Gea3ParityStep]| -> Value {
        json!({
            "input_upload_us": gea3_parity_summed("input upload", &steps.iter().map(|step| step.input_upload.duration_us).collect::<Vec<_>>(), steps.len()),
            "launch_encode_us": gea3_parity_summed("launch encode", &steps.iter().map(|step| step.launch_encode.duration_us).collect::<Vec<_>>(), steps.len()),
            "queue_wait_us": gea3_parity_summed("queue wait (submit plus sync)", &steps.iter().map(|step| step.queue_wait.duration_us).collect::<Vec<_>>(), steps.len()),
            "submit_sync_us": gea3_parity_summed("submit plus sync", &steps.iter().map(|step| step.queue_wait.duration_us).collect::<Vec<_>>(), steps.len()),
            "gpu_encoder_us": gea3_parity_gpu_summed(steps),
            "readback_us": gea3_parity_summed("readback", &steps.iter().map(|step| step.readback.duration_us).collect::<Vec<_>>(), steps.len()),
        })
    };
    let residency = &timing.residency;
    Ok(json!({
        "schema": PARITY_COMPANION_SCHEMA,
        "delivery": "PPB-U3",
        "source_test": "gea3_real_metal_decode_receipt",
        "source_receipt_schema": "gea3-metal-receipt-v1",
        "source_receipt_path": source_receipt,
        "measurement_laws": [
            "the launch/encode clock ends before the queue wait begins",
            "the queue wait ends at the declared completion point (the explicit step sync return)",
            "every category is directly measured on its own clock; no category is a total minus other terms",
            "unified-memory allocation is never called a transfer without an observed copy event",
        ],
        "phases": {
            "prefill": prefill_steps.iter().map(|step| gea3_parity_step_json(step)).collect::<Vec<_>>(),
            "decode": decode_steps.iter().map(|step| gea3_parity_step_json(step)).collect::<Vec<_>>(),
        },
        "transfers": {
            "weight_residency_upload": {
                "duration_us": gea3_parity_measured(residency.weight_upload_us, "phase clock around the 290 copy_in_bytes admissions"),
                "bytes": gea3_parity_measured(residency.weight_bytes, "GEA3 input manifest absolute GGUF ranges for the frozen model tensors"),
                "admissions": gea3_parity_count(residency.weight_uploads, "distinct frozen model tensor identities admitted this run"),
                "host_to_device_transfer": gea3_parity_not_observable(
                    "admissions run over a retained unified-memory GGUF mapping and may wrap pages zero-copy; copy-versus-wrap is not observable through the DeviceRuntime surface",
                ),
            },
            "kv_zero_fill": {
                "duration_us": gea3_parity_measured(residency.kv_zero_fill_us, "phase clocks around the copy_in_bytes zero-fill of the KV arenas"),
                "bytes": gea3_parity_measured(residency.kv_zero_fill_bytes, "32 * 2 * 76 * 320 * sizeof(F32) zero-filled on first touch"),
                "host_to_device_transfer": {"status": "measured", "basis": "observed copy_in_bytes zero-fill events"},
            },
            "other_setup_zero_fills": {
                "count": gea3_parity_count(residency.other_zero_fills, "non-KV handles zero-filled during residency setup"),
                "duration_us": gea3_parity_not_measured(
                    "interleaved with the residency loop and not separately clocked; no value is inferred",
                ),
            },
            "residency_allocations": {
                "weight_allocation_us": gea3_parity_measured(residency.weight_alloc_us, "device allocation clocks for the 290 shared model tensors"),
                "kv_allocation_us": gea3_parity_measured(residency.kv_alloc_us, "device allocation clocks for the shared KV arenas"),
                "transfer_classification": {"value": "allocation only; a unified-memory allocation is not a transfer without an observed copy event", "status": "measured"},
            },
        },
        "summary": {
            "prefill": summarize(&prefill_steps),
            "decode": summarize(&decode_steps),
        },
    }))
}

fn gea3_parity_companion_path() -> Option<PathBuf> {
    std::env::var_os(PARITY_COMPANION_ENV).map(PathBuf::from)
}

/// Write the companion when an output path is present; an absent path is
/// the opt-out and leaves the physical run untouched.
fn gea3_write_parity_companion(target: Option<&Path>, companion: &Value) {
    let Some(path) = target else {
        return;
    };
    let parent = path.parent().expect("parity companion parent");
    fs::create_dir_all(parent).expect("create parity companion parent");
    fs::write(
        path,
        serde_json::to_vec_pretty(companion).expect("serialize parity timing companion"),
    )
    .unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
    eprintln!("GEA3 parity timing companion: {}", path.display());
}

// PPB-U3 focused fake/admission proof.  It never touches the physical
// route: it drives the production builder and writer with synthetic fake
// observations, the way the fake ladder proves the launch graph.  It covers
// the absent and the present optional output, inspects the emitted
// companion schema, and proves the non-overlap boundary law fails closed.
fn gea3_fake_parity_step(
    mode: &'static str,
    step: u64,
    gpu_encoder_us: Option<u64>,
) -> Gea3ParityStep {
    Gea3ParityStep {
        mode,
        step,
        input_upload: Gea3ParityPhase {
            start_since_step_us: 0,
            end_since_step_us: 40,
            duration_us: 40,
        },
        input_upload_bytes: 1_048_576,
        input_upload_copies: 5,
        launch_encode: Gea3ParityPhase {
            start_since_step_us: 40,
            end_since_step_us: 240,
            duration_us: 200,
        },
        queue_wait: Gea3ParityPhase {
            start_since_step_us: 240,
            end_since_step_us: 600,
            duration_us: 360,
        },
        gpu_encoder_us,
        gpu_timestamp_count: u64::from(gpu_encoder_us.is_some()) as usize * LAUNCHES_PER_PROGRAM,
        gpu_start_timestamp_count: u64::from(gpu_encoder_us.is_some()) as usize * LAUNCHES_PER_PROGRAM,
        gpu_total_encoder_count: LAUNCHES_PER_PROGRAM,
        command_submits: 1,
        blocking_waits: 1,
        readback: Gea3ParityPhase {
            start_since_step_us: 600,
            end_since_step_us: 640,
            duration_us: 40,
        },
        readback_bytes: VOCAB * 4,
    }
}

fn gea3_fake_parity_timing(steps: Vec<Gea3ParityStep>) -> Gea3ParityTiming {
    Gea3ParityTiming {
        residency: Gea3ParityResidency {
            weight_alloc_us: 900,
            weight_upload_us: 4_000,
            weight_bytes: 1_447_284_480,
            weight_uploads: 290,
            kv_alloc_us: 60,
            kv_zero_fill_us: 120,
            kv_zero_fill_bytes: LAYERS as u64 * 2 * HISTORY_CAPACITY * KV_WIDTH * 4,
            other_zero_fills: 3,
        },
        steps,
    }
}

#[test]
fn gea3_parity_timing_companion_optional_emission() {
    let timing = gea3_fake_parity_timing(vec![
        gea3_fake_parity_step("prefill", 0, Some(150)),
        gea3_fake_parity_step("decode", 1, Some(140)),
        // A step whose GPU timestamps were not sampled keeps the explicit
        // not_measured status; no encoder time is inferred for it.
        gea3_fake_parity_step("decode", 2, None),
    ]);
    let companion = gea3_build_parity_companion(&timing, "evidence/gea3-metal-receipt.json")
        .expect("fake companion observations admit");

    // (a) the companion is versioned and disjoint from the GEA3 receipt.
    assert_eq!(companion["schema"], json!(PARITY_COMPANION_SCHEMA));
    assert_eq!(companion["delivery"], json!("PPB-U3"));
    assert_eq!(
        companion["source_receipt_schema"],
        json!("gea3-metal-receipt-v1")
    );

    // (b) boundary law, green: encode ends before the queue wait begins,
    // and the queue wait ends before the readback begins, on every step.
    let all_phase_rows = companion["phases"]["prefill"]
        .as_array()
        .into_iter()
        .flatten()
        .chain(
            companion["phases"]["decode"]
                .as_array()
                .into_iter()
                .flatten(),
        );
    for phase_row in all_phase_rows {
        let encode_end = phase_row["host_launch_encode"]["boundary"]["end_us_since_step_start"]
            .as_u64()
            .expect("encode end boundary");
        let queue_start = phase_row["queue_wait"]["boundary"]["start_us_since_step_start"]
            .as_u64()
            .expect("queue start boundary");
        let queue_end = phase_row["queue_wait"]["boundary"]["end_us_since_step_start"]
            .as_u64()
            .expect("queue end boundary");
        let readback_start = phase_row["device_to_host_readback"]["boundary"]
            ["start_us_since_step_start"]
            .as_u64()
            .expect("readback start boundary");
        assert!(
            encode_end <= queue_start,
            "encode clock must end before the queue wait begins"
        );
        assert!(
            queue_end <= readback_start,
            "queue wait must end at the declared completion point before readback"
        );
        assert_eq!(
            phase_row["queue_wait"]["ends_at"]["value"],
            json!("explicit runtime.sync() return — the declared completion point")
        );
    }

    // (c) every phase carries an evidence status; the unsampled GPU step is
    // explicitly not_measured and the submit-only wall is not_observable.
    let unsampled = &companion["phases"]["decode"][1];
    assert_eq!(
        unsampled["gpu_encoder"]["duration_us"]["status"],
        json!("not_measured")
    );
    assert_eq!(
        unsampled["gpu_encoder"]["duration_us"]["sampled_encoders"],
        json!(0)
    );
    assert_eq!(
        unsampled["gpu_encoder"]["duration_us"]["total_encoders"],
        json!(LAUNCHES_PER_PROGRAM)
    );
    assert!(
        unsampled["gpu_encoder"]["duration_us"]["reason"]
            .as_str()
            .is_some_and(|reason| reason.contains("no GPU timestamps")),
        "the not_measured reason must name the missing fact"
    );
    assert_eq!(
        unsampled["command_submit_wall"]["status"],
        json!("not_observable")
    );
    let sampled = &companion["phases"]["decode"][0];
    assert_eq!(
        sampled["gpu_encoder"]["duration_us"]["status"],
        json!("measured")
    );
    assert_eq!(
        sampled["gpu_encoder"]["duration_us"]["sampled_encoders"],
        json!(LAUNCHES_PER_PROGRAM)
    );
    assert_eq!(
        sampled["gpu_encoder"]["duration_us"]["total_encoders"],
        json!(LAUNCHES_PER_PROGRAM)
    );

    // A partial sample must remain honest at the duration field itself, not
    // merely in a sibling count.  This is the fixed-1000 shape when Metal's
    // 2,048-slot counter buffer samples 1,024 of the encoders.
    let mut partial = gea3_fake_parity_step("decode", 4, Some(125));
    partial.gpu_timestamp_count = 1_024;
    partial.gpu_start_timestamp_count = 1_024;
    let partial_companion = gea3_build_parity_companion(
        &gea3_fake_parity_timing(vec![partial]),
        "evidence/gea3-metal-receipt.json",
    )
    .expect("partial GPU sample remains admissible");
    // A2g re-derive trigger: composed MLP parents collapse 4 chain
    // edges/layer x 32 layers, so the decode encoder total is re-derived
    // 2,115 -> 1,987 (= 32 * 62 + 3); re-derive both rows from the census
    // export if the A2g admission composition moves again.
    assert_eq!(
        partial_companion["phases"]["decode"][0]["gpu_encoder"]["duration_us"]["coverage"],
        json!("1024/1987")
    );
    assert_eq!(
        partial_companion["summary"]["decode"]["gpu_encoder_us"]["coverage_per_step"][0],
        json!("1024/1987")
    );

    // (d) transfer honesty: the weight residency never claims a transfer it
    // cannot observe, and allocation is not a transfer.
    assert_eq!(
        companion["transfers"]["weight_residency_upload"]["host_to_device_transfer"]["status"],
        json!("not_observable")
    );
    assert_eq!(
        companion["transfers"]["kv_zero_fill"]["host_to_device_transfer"]["status"],
        json!("measured")
    );
    assert_eq!(
        companion["transfers"]["other_setup_zero_fills"]["duration_us"]["status"],
        json!("not_measured")
    );
    assert!(
        companion["transfers"]["residency_allocations"]["transfer_classification"]["value"]
            .as_str()
            .is_some_and(|value| value.contains("not a transfer")),
        "allocation must be classified apart from transfer"
    );

    // (e) red control: overlapping encode/queue boundaries fail closed and
    // name the law, so the boundary admission is a gate, not decoration.
    let mut overlapping = gea3_fake_parity_step("decode", 3, None);
    overlapping.launch_encode.end_since_step_us = 300;
    overlapping.queue_wait.start_since_step_us = 250;
    let error = gea3_build_parity_companion(
        &gea3_fake_parity_timing(vec![overlapping]),
        "evidence/gea3-metal-receipt.json",
    )
    .expect_err("overlapping boundaries must fail closed");
    assert!(
        error.contains("queue wait begins"),
        "diagnostic must name the boundary law: {error}"
    );

    // (f) absent optional output: no path means no companion, and nothing
    // is written.  Present optional output: the production writer emits the
    // companion and the written file parses back to the same schema.
    let absent_dir = tempfile::tempdir().expect("absent-output tempdir");
    std::env::remove_var(PARITY_COMPANION_ENV);
    assert!(
        gea3_parity_companion_path().is_none(),
        "unset environment is the opt-out"
    );
    gea3_write_parity_companion(gea3_parity_companion_path().as_deref(), &companion);
    assert_eq!(
        fs::read_dir(absent_dir.path())
            .expect("read absent-output tempdir")
            .count(),
        0,
        "no companion may be written without a path"
    );

    let present_dir = tempfile::tempdir().expect("present-output tempdir");
    let companion_path = present_dir.path().join("gea3-parity-timing-companion.json");
    std::env::set_var(PARITY_COMPANION_ENV, &companion_path);
    let target = gea3_parity_companion_path().expect("companion path from environment");
    gea3_write_parity_companion(Some(&target), &companion);
    std::env::remove_var(PARITY_COMPANION_ENV);
    let written: Value =
        serde_json::from_slice(&fs::read(&companion_path).expect("read written companion"))
            .expect("written companion is JSON");
    assert_eq!(written["schema"], json!(PARITY_COMPANION_SCHEMA));
    assert_eq!(
        written["phases"]["decode"][1]["gpu_encoder"]["duration_us"]["status"],
        json!("not_measured")
    );
    assert_eq!(
        written["summary"]["decode"]["queue_wait_us"]["status"],
        json!("derived")
    );
    assert_eq!(
        written["summary"]["decode"]["submit_sync_us"]["status"],
        json!("derived")
    );
}

fn gea3_run_physical(
    runtime: &mut DeviceRuntime,
    prefill: (DeviceDescriptor, Gea3WindowBindings),
    decode: (DeviceDescriptor, Gea3WindowBindings),
    input_manifest: &Value,
    model_path: &Path,
    prompt_tokens: &[u32],
    identity: Gea3Identity,
    stream: Option<&Gea3SoakStream>,
) -> Result<(Value, Gea3ParityTiming), String> {
    let model_ranges = gea3_model_ranges(input_manifest)?;
    let mapped = MappedWeightFile::open(model_path).map_err(|error| error.message.clone())?;
    runtime
        .retain_mapped_weight_file(mapped.clone())
        .map_err(|error| error.message.clone())?;
    let mut shared = BTreeMap::new();
    let mut programs = Vec::new();
    let prepare = (|| {
        let prefill_program =
            gea3_prepare_physical_program(runtime, prefill.0, prefill.1, &mut shared)?;
        let decode_program =
            gea3_prepare_physical_program(runtime, decode.0, decode.1, &mut shared)?;
        programs.push(prefill_program);
        programs.push(decode_program);
        Ok::<(), String>(())
    })();
    if let Err(error) = prepare {
        drop(gea3_release_programs(runtime, &mut programs));
        return Err(error);
    }
    let weight_allocations_us = programs
        .iter()
        .map(|program| program.weight_alloc_us)
        .sum::<u64>();
    let kv_allocations_us = programs
        .iter()
        .map(|program| program.kv_alloc_us)
        .sum::<u64>();
    let weight_upload_started = Instant::now();
    let mut weight_bytes = 0_u64;
    let mut weight_uploads = 0_u64;
    for (name, handle) in shared
        .iter()
        .filter_map(|(key, handle)| key.strip_prefix("weight:").map(|name| (name, *handle)))
    {
        let (start, end) = model_ranges
            .get(name)
            .copied()
            .ok_or_else(|| format!("no frozen GGUF range for model tensor `{name}`"))?;
        let start =
            usize::try_from(start).map_err(|_| format!("tensor `{name}` offset overflows"))?;
        let end = usize::try_from(end).map_err(|_| format!("tensor `{name}` end overflows"))?;
        let bytes = mapped
            .bytes()
            .get(start..end)
            .ok_or_else(|| format!("tensor `{name}` range is outside GGUF"))?;
        let expected = handle
            .len_bytes()
            .ok_or_else(|| format!("tensor `{name}` handle has no byte length"))?;
        if expected != bytes.len() as u64 {
            return Err(format!(
                "tensor `{name}` declares {expected} bytes but GGUF carries {}",
                bytes.len()
            ));
        }
        runtime
            .copy_in_bytes(&handle, bytes, DeviceDataType::U8)
            .map_err(|error| error.message.clone())?;
        weight_bytes = weight_bytes.saturating_add(bytes.len() as u64);
        weight_uploads += 1;
    }
    if weight_uploads != 290 {
        return Err(format!(
            "uploaded {weight_uploads} model weights, expected 290"
        ));
    }
    let weight_upload_us = gea3_elapsed_us(weight_upload_started);
    let mut kv_bytes = 0_u64;
    let mut kv_zero_us = 0_u64;
    let mut other_zero_fills = 0_u64;
    let mut zeroed = BTreeSet::new();
    for program in &programs {
        let resident_slots = gea3_unique_slots(&program.descriptor);
        for (key, handle) in &program.buffers {
            let slot = resident_slots
                .get(key)
                .ok_or_else(|| "resident slot metadata disappeared")?;
            if gea3_canonical_model_name(&slot.buffer_name).is_some() {
                continue;
            }
            let first_zero = zeroed.insert(handle.id);
            if slot.buffer_name.starts_with("blk.") && slot.buffer_name.contains(".kv_") {
                kv_bytes = kv_bytes.saturating_add(if first_zero {
                    handle.len_bytes().unwrap_or(0)
                } else {
                    0
                });
            }
            if (slot.lifetime == DeviceBufferLifetime::PerProgram
                && slot.initialization == DeviceBufferInitialization::HostProvided)
                || slot.initialization == DeviceBufferInitialization::ZeroFill
            {
                if first_zero {
                    let started = Instant::now();
                    gea3_zero_handle(runtime, handle)?;
                    if slot.buffer_name.starts_with("blk.") && slot.buffer_name.contains(".kv_") {
                        kv_zero_us = kv_zero_us.saturating_add(gea3_elapsed_us(started));
                    } else {
                        other_zero_fills += 1;
                    }
                }
            }
        }
    }
    let kv_setup_us = kv_allocations_us.saturating_add(kv_zero_us);
    // The allocation pass above counted every zero-fill handle in `zeroed`;
    // the model weights are deliberately excluded from that set.
    let expected_kv_bytes = identity.kv_bytes();
    if kv_bytes != expected_kv_bytes {
        return Err(format!(
            "KV residency is {kv_bytes} bytes, expected {expected_kv_bytes} ({})",
            identity.kv_basis()
        ));
    }
    let launch_rows_prefill = gea3_launch_rows(&programs[0].descriptor);
    let launch_rows_decode = gea3_launch_rows(&programs[1].descriptor);
    let edge_prefill = gea3_data_flow_satisfied(&programs[0].descriptor);
    let edge_decode = gea3_data_flow_satisfied(&programs[1].descriptor);
    if !edge_prefill || !edge_decode {
        return Err("GEA3 carried data-flow edges are not topologically satisfied".to_owned());
    }
    // PPB-U7: the residency cells are final here — weights uploaded, both KV
    // arenas allocated and zero-filled — so a soak stream writes its first
    // receipt state before any decode step runs.
    let residency = json!({
        "weight_allocations": {"value": 290, "status": "measured", "basis": "distinct frozen model tensor identities"},
        "weight_bytes": {"value": weight_bytes, "status": "measured", "basis": "GEA3 input manifest absolute ranges"},
        "weight_upload_count": {"value": weight_uploads, "status": "measured"},
        "weight_residency_us": {"value": weight_allocations_us.saturating_add(weight_upload_us), "status": "measured", "components": {"allocation_and_program_setup": weight_allocations_us, "mapped_upload": weight_upload_us}},
        "kv_allocations": {"value": LAYERS * 2, "status": "measured", "basis": "one shared fixed-capacity K/V arena per layer"},
        "kv_bytes": {"value": kv_bytes, "status": "measured", "basis": identity.kv_basis()},
        "kv_alloc_us": {"value": kv_setup_us, "status": "measured", "basis": "KV handle allocation plus first zero-fill"},
        "zero_cpu_substitutes": {"value": 0, "status": "measured", "basis": "all model work was submitted to Metal"},
        "zero_cpu_bridges": {"value": 0, "status": "measured", "basis": "only one-hot staging, mask/rope constants, and host argmax ran on CPU"},
    });
    if let Some(stream) = stream {
        stream.write_partial(&SoakPartialEvidence {
            residency: residency.clone(),
            execution: json!({
                "step_count": {"value": 0, "status": "measured", "basis": "no decode step had run when this streamed receipt was written"},
                "prefill_wall_us": gea3_unmeasured("prefill had not run when this streamed receipt was written"),
            }),
            steps: json!([]),
            launch_plans: json!({}),
            throughput: json!({}),
            decode_steps: 0,
            produced_tokens: 0,
        });
    }
    let mut step_receipts = Vec::new();
    let mut parity_steps = Vec::new();
    let mut greedy = Vec::new();
    let prefill_started = Instant::now();
    let prefill_uploads = gea3_update_inputs(
        runtime,
        &programs[0],
        prompt_tokens,
        0,
        u32::try_from(prompt_tokens.len()).map_err(|_| "prompt is too long".to_owned())?,
        true,
        identity.history_capacity,
    )?;
    // PPB-U3: the input-upload phase clock starts with the step clock, so
    // one elapsed reading is both its duration and its end boundary.
    let prefill_upload_us = gea3_elapsed_us(prefill_started);
    let prefill_before_submit = runtime.command_submit_count();
    let prefill_before_wait = runtime.blocking_wait_count();
    let prefill_launch_started = Instant::now();
    let prefill_encode_start_us = gea3_elapsed_us(prefill_started);
    for launch in &programs[0].descriptor.launches {
        let kernel = &programs[0].descriptor.kernels[launch.kernel_index as usize];
        let bindings: Vec<DeviceLaunchBinding> = kernel
            .buffers
            .iter()
            .map(|slot| {
                let handle = programs[0]
                    .buffers
                    .get(&(slot.buffer_id, slot.version))
                    .copied()
                    .ok_or_else(|| "prefill launch buffer disappeared".to_owned())?;
                gea3_launch_binding(handle, &programs[0], launch.kernel_index, slot)
            })
            .collect::<Result<_, _>>()?;
        runtime
            .launch_kernel_bound(
                &programs[0].module,
                &kernel.entry,
                &bindings,
                kernel.grid,
                kernel.block,
            )
            .map_err(|error| error.message.clone())?;
    }
    let prefill_encode_sync_us = gea3_elapsed_us(prefill_launch_started);
    let prefill_encode_end_us = gea3_elapsed_us(prefill_started);
    // PPB-U3: the queue-wait clock starts only after the encode clock has
    // ended and stops at the declared completion point — the sync return.
    let prefill_queue_started = Instant::now();
    let prefill_queue_start_us = gea3_elapsed_us(prefill_started);
    runtime.sync().map_err(|error| error.message.clone())?;
    let prefill_queue_wait_us = gea3_elapsed_us(prefill_queue_started);
    let prefill_queue_end_us = gea3_elapsed_us(prefill_started);
    let prefill_gpu_us = runtime.take_encoder_gpu_us();
    let prefill_gpu_start_us = runtime.take_encoder_gpu_start_us();
    let prefill_submit_count = runtime
        .command_submit_count()
        .saturating_sub(prefill_before_submit);
    let prefill_wait_count = runtime
        .blocking_wait_count()
        .saturating_sub(prefill_before_wait);
    let prefill_readback_started = Instant::now();
    let prefill_readback_start_us = gea3_elapsed_us(prefill_started);
    let prefill_handle = programs[0]
        .buffers
        .get(&programs[0].output)
        .copied()
        .ok_or_else(|| "prefill logits handle disappeared".to_owned())?;
    let prefill_values = gea3_f32_from_bytes(
        &runtime
            .readback_bytes(&prefill_handle, DeviceDataType::F32)
            .map_err(|error| error.message.clone())?,
    )?;
    let prefill_readback_us = gea3_elapsed_us(prefill_readback_started);
    let prefill_readback_end_us = gea3_elapsed_us(prefill_started);
    let prefill_vocab = usize::try_from(VOCAB).unwrap();
    let prefill_last = prefill_values
        .get((prompt_tokens.len().saturating_sub(1) * prefill_vocab)..)
        .ok_or_else(|| "prefill logits are shorter than the final prompt row".to_owned())?;
    let mut next_token = gea3_argmax(prefill_last)?;
    greedy.push(next_token);
    parity_steps.push(Gea3ParityStep {
        mode: "prefill",
        step: 0,
        input_upload: Gea3ParityPhase {
            start_since_step_us: 0,
            end_since_step_us: prefill_upload_us,
            duration_us: prefill_upload_us,
        },
        input_upload_bytes: prefill_uploads.bytes,
        input_upload_copies: prefill_uploads.copies,
        launch_encode: Gea3ParityPhase {
            start_since_step_us: prefill_encode_start_us,
            end_since_step_us: prefill_encode_end_us,
            duration_us: prefill_encode_sync_us,
        },
        queue_wait: Gea3ParityPhase {
            start_since_step_us: prefill_queue_start_us,
            end_since_step_us: prefill_queue_end_us,
            duration_us: prefill_queue_wait_us,
        },
        gpu_encoder_us: (!prefill_gpu_us.is_empty()).then(|| prefill_gpu_us.iter().copied().sum()),
        gpu_timestamp_count: prefill_gpu_us.len(),
        gpu_start_timestamp_count: prefill_gpu_start_us.len(),
        gpu_total_encoder_count: programs[0].descriptor.launches.len(),
        command_submits: prefill_submit_count,
        blocking_waits: prefill_wait_count,
        readback: Gea3ParityPhase {
            start_since_step_us: prefill_readback_start_us,
            end_since_step_us: prefill_readback_end_us,
            duration_us: prefill_readback_us,
        },
        readback_bytes: u64::try_from(prefill_values.len() * 4).unwrap_or(u64::MAX),
    });
    step_receipts.push(json!({
        "mode": "prefill",
        "step": 0,
        "input_uploads": prefill_uploads.copies,
        "launch_plan": "prefill",
        "launch_count": programs[0].descriptor.launches.len(),
        "data_flow_edges": {"declared": programs[0].descriptor.data_flow.len(), "satisfied": edge_prefill},
        "dispatch": {"launches": programs[0].descriptor.launches.len(), "command_submits": prefill_submit_count, "blocking_waits": prefill_wait_count},
        "timing_us": {
            "wall": gea3_elapsed_us(prefill_started),
            "launch_encode_us": prefill_encode_sync_us,
            "submit_sync_us": prefill_queue_wait_us,
            "gpu_body_sum": gea3_gpu_timing_cell(
                (!prefill_gpu_us.is_empty()).then(|| prefill_gpu_us.iter().copied().sum()),
                prefill_gpu_us.len(),
                programs[0].descriptor.launches.len(),
                "sum of sampled per-encoder Metal GPU timestamps; coverage is explicit",
            ),
            "gpu_timestamp_count": gea3_gpu_count_cell(
                prefill_gpu_us.len(),
                programs[0].descriptor.launches.len(),
                "sampled per-encoder GPU end timestamps",
            ),
            "gpu_start_timestamp_count": gea3_gpu_count_cell(
                prefill_gpu_start_us.len(),
                programs[0].descriptor.launches.len(),
                "sampled per-encoder GPU start timestamps",
            ),
            "readback": prefill_readback_us,
        },
        "readback": {"buffer_id": programs[0].output.0, "version": programs[0].output.1, "elements": prefill_values.len(), "bytes": prefill_values.len() * 4, "sha256": gea3_readback_hash(&prefill_values), "finite": true},
        "next_token": next_token,
    }));
    // PPB-U7: progress counts decoded output tokens.  The prefill logits
    // choose the first input to decode but do not count as generated output.
    for step in 0..identity.decode_steps {
        let valid_len = u32::try_from(prompt_tokens.len() + step + 1)
            .map_err(|_| "decode valid length overflows".to_owned())?;
        let position = valid_len - 1;
        let started = Instant::now();
        let uploads = gea3_update_inputs(
            runtime,
            &programs[1],
            &[next_token],
            position,
            valid_len,
            false,
            identity.history_capacity,
        )?;
        // PPB-U3: the input-upload phase clock starts with the step clock.
        let upload_us = gea3_elapsed_us(started);
        let before_submit = runtime.command_submit_count();
        let before_wait = runtime.blocking_wait_count();
        let launch_started = Instant::now();
        let encode_start_us = gea3_elapsed_us(started);
        for launch in &programs[1].descriptor.launches {
            let kernel = &programs[1].descriptor.kernels[launch.kernel_index as usize];
            let bindings: Vec<DeviceLaunchBinding> = kernel
                .buffers
                .iter()
                .map(|slot| {
                    let handle = programs[1]
                        .buffers
                        .get(&(slot.buffer_id, slot.version))
                        .copied()
                        .ok_or_else(|| "decode launch buffer disappeared".to_owned())?;
                    gea3_launch_binding(handle, &programs[1], launch.kernel_index, slot)
                })
                .collect::<Result<_, _>>()?;
            runtime
                .launch_kernel_bound(
                    &programs[1].module,
                    &kernel.entry,
                    &bindings,
                    kernel.grid,
                    kernel.block,
                )
                .map_err(|error| error.message.clone())?;
        }
        let encode_sync_us = gea3_elapsed_us(launch_started);
        let encode_end_us = gea3_elapsed_us(started);
        // PPB-U3: the queue-wait clock starts only after the encode clock
        // has ended and stops at the declared completion point.
        let queue_started = Instant::now();
        let queue_start_us = gea3_elapsed_us(started);
        runtime.sync().map_err(|error| error.message.clone())?;
        let queue_wait_us = gea3_elapsed_us(queue_started);
        let queue_end_us = gea3_elapsed_us(started);
        let gpu_us = runtime.take_encoder_gpu_us();
        let gpu_start_us = runtime.take_encoder_gpu_start_us();
        let submit_count = runtime.command_submit_count().saturating_sub(before_submit);
        let wait_count = runtime.blocking_wait_count().saturating_sub(before_wait);
        let readback_started = Instant::now();
        let readback_start_us = gea3_elapsed_us(started);
        let output_handle = programs[1]
            .buffers
            .get(&programs[1].output)
            .copied()
            .ok_or_else(|| "decode logits handle disappeared".to_owned())?;
        let values = gea3_f32_from_bytes(
            &runtime
                .readback_bytes(&output_handle, DeviceDataType::F32)
                .map_err(|error| error.message.clone())?,
        )?;
        let readback_us = gea3_elapsed_us(readback_started);
        let readback_end_us = gea3_elapsed_us(started);
        next_token = gea3_argmax(&values)?;
        greedy.push(next_token);
        parity_steps.push(Gea3ParityStep {
            mode: "decode",
            step: (step + 1) as u64,
            input_upload: Gea3ParityPhase {
                start_since_step_us: 0,
                end_since_step_us: upload_us,
                duration_us: upload_us,
            },
            input_upload_bytes: uploads.bytes,
            input_upload_copies: uploads.copies,
            launch_encode: Gea3ParityPhase {
                start_since_step_us: encode_start_us,
                end_since_step_us: encode_end_us,
                duration_us: encode_sync_us,
            },
            queue_wait: Gea3ParityPhase {
                start_since_step_us: queue_start_us,
                end_since_step_us: queue_end_us,
                duration_us: queue_wait_us,
            },
            gpu_encoder_us: (!gpu_us.is_empty()).then(|| gpu_us.iter().copied().sum()),
            gpu_timestamp_count: gpu_us.len(),
            gpu_start_timestamp_count: gpu_start_us.len(),
            gpu_total_encoder_count: programs[1].descriptor.launches.len(),
            command_submits: submit_count,
            blocking_waits: wait_count,
            readback: Gea3ParityPhase {
                start_since_step_us: readback_start_us,
                end_since_step_us: readback_end_us,
                duration_us: readback_us,
            },
            readback_bytes: u64::try_from(values.len() * 4).unwrap_or(u64::MAX),
        });
        step_receipts.push(json!({
            "mode": "decode",
            "step": step + 1,
            "position": position,
            "valid_len_after": valid_len,
            "input_uploads": uploads.copies,
            "launch_plan": "decode",
            "launch_count": programs[1].descriptor.launches.len(),
            "data_flow_edges": {"declared": programs[1].descriptor.data_flow.len(), "satisfied": edge_decode},
            "dispatch": {"launches": programs[1].descriptor.launches.len(), "command_submits": submit_count, "blocking_waits": wait_count},
            "timing_us": {
                "wall": gea3_elapsed_us(started),
                "launch_encode_us": encode_sync_us,
                "submit_sync_us": queue_wait_us,
                "gpu_body_sum": gea3_gpu_timing_cell(
                    (!gpu_us.is_empty()).then(|| gpu_us.iter().copied().sum()),
                    gpu_us.len(),
                    programs[1].descriptor.launches.len(),
                    "sum of sampled per-encoder Metal GPU timestamps; coverage is explicit",
                ),
                "gpu_timestamp_count": gea3_gpu_count_cell(
                    gpu_us.len(),
                    programs[1].descriptor.launches.len(),
                    "sampled per-encoder GPU end timestamps",
                ),
                "gpu_start_timestamp_count": gea3_gpu_count_cell(
                    gpu_start_us.len(),
                    programs[1].descriptor.launches.len(),
                    "sampled per-encoder GPU start timestamps",
                ),
                "readback": readback_us,
            },
            "readback": {"buffer_id": programs[1].output.0, "version": programs[1].output.1, "elements": values.len(), "bytes": values.len() * 4, "sha256": gea3_readback_hash(&values), "finite": true},
            "next_token": next_token,
        }));
        // PPB-U7: the U6 monotonic produced-token counter — one strict JSON
        // line per produced token on stdout, then the cadence receipt
        // rewrite so a cap-killed arm leaves its latest measured state.
        if let Some(stream) = stream {
            stream.emit_progress((step + 1) as u64);
            if (step + 1) % SOAK_RECEIPT_REWRITE_STEPS == 0 || step + 1 == identity.decode_steps {
                let decode_walls: Vec<u64> = step_receipts
                    .iter()
                    .skip(1)
                    .filter_map(|row| row["timing_us"]["wall"].as_u64())
                    .collect();
                let decode_wall: u64 = decode_walls.iter().sum();
                stream.write_partial(&SoakPartialEvidence {
                    residency: residency.clone(),
                    execution: json!({
                        "prefill_wall_us": step_receipts.first()
                            .and_then(|row| row["timing_us"]["wall"].as_u64())
                            .map(|wall| json!({"value": wall, "status": "measured"}))
                            .unwrap_or_else(|| gea3_unmeasured("prefill wall not yet recorded")),
                        "decode_wall_us": {"value": decode_wall, "status": "measured"},
                        "step_count": {"value": step + 1, "status": "measured", "basis": "decode steps completed when this streamed receipt was written"},
                        "greedy_token_sequence": {"value": greedy, "status": "measured", "basis": "first-index host argmax of declared logits readback"},
                    }),
                    steps: json!(step_receipts),
                    launch_plans: json!({"prefill": launch_rows_prefill, "decode": launch_rows_decode}),
                    throughput: json!({
                        "tg_ts": {"value": (step + 1) as f64 * 1_000_000.0 / decode_wall.max(1) as f64, "status": "derived", "basis": "observed decode steps / summed observed decode wall"},
                    }),
                    decode_steps: step + 1,
                    produced_tokens: step + 1,
                });
                if let Some(companion_target) = stream.companion_path.as_ref() {
                    let timing = Gea3ParityTiming {
                        residency: Gea3ParityResidency {
                            weight_alloc_us: weight_allocations_us,
                            weight_upload_us,
                            weight_bytes,
                            weight_uploads,
                            kv_alloc_us: kv_allocations_us,
                            kv_zero_fill_us: kv_zero_us,
                            kv_zero_fill_bytes: kv_bytes,
                            other_zero_fills,
                        },
                        steps: parity_steps.clone(),
                    };
                    if let Ok(companion) = gea3_build_parity_companion(
                        &timing,
                        &stream.receipt_path.display().to_string(),
                    ) {
                        gea3_write_parity_companion(Some(companion_target), &companion);
                    }
                }
            }
        }
    }
    let prefill_wall_us = step_receipts
        .first()
        .and_then(|row| row["timing_us"]["wall"].as_u64())
        .unwrap_or_else(|| gea3_elapsed_us(prefill_started));
    let decode_wall_us: u64 = step_receipts
        .iter()
        .skip(1)
        .filter_map(|row| row["timing_us"]["wall"].as_u64())
        .sum();
    let decode_gpu_cells: Vec<Value> = step_receipts
        .iter()
        .skip(1)
        .map(|row| row["timing_us"]["gpu_body_sum"].clone())
        .collect();
    let decode_launch_encode_us: Vec<u64> = step_receipts
        .iter()
        .skip(1)
        .filter_map(|row| row["timing_us"]["launch_encode_us"].as_u64())
        .collect();
    let decode_submit_sync_us: Vec<u64> = step_receipts
        .iter()
        .skip(1)
        .filter_map(|row| row["timing_us"]["submit_sync_us"].as_u64())
        .collect();
    let kv_alloc_us = kv_setup_us;
    // Every statue uses the same receipt schema; a non-frozen statue names
    // its own completed-step basis without changing the timing field shape.
    let step_count_cell = if identity.history_capacity == HISTORY_CAPACITY {
        json!({"value": identity.decode_steps, "status": "assumed", "basis": "frozen GEA3 n_predict"})
    } else {
        json!({"value": identity.decode_steps, "status": "measured", "basis": "completed decode steps (the statue's pinned step count)"})
    };
    let evidence = json!({
        "residency": residency,
        "execution": {
            "prefill_wall_us": {"value": prefill_wall_us, "status": "measured"},
            "per_step_gpu_body_us": {"value": decode_gpu_cells, "status": "measured", "basis": "each per-step sampled Metal encoder sum carries its sampled/total encoder coverage"},
            "launch_encode_us_per_step": {"value": decode_launch_encode_us, "status": "measured", "basis": "host launch encode wall before the explicit step sync"},
            "submit_sync_us_per_step": {"value": decode_submit_sync_us, "status": "measured", "basis": "host wall around explicit Metal command-buffer submit plus wait"},
            "launches_per_step": {"value": programs[1].descriptor.launches.len(), "status": "derived", "basis": "descriptor launch list"},
            "step_count": step_count_cell,
            "submit_sync_count": {"value": step_receipts.iter().skip(1).map(|row| row["dispatch"]["blocking_waits"].as_u64().unwrap_or(0)).sum::<u64>(), "status": "measured", "basis": "blocking waits at the explicit Metal step boundary"},
            "launch_encode_us": {"value": decode_launch_encode_us.iter().copied().sum::<u64>(), "status": "derived", "basis": "sum of per-step launch encode clocks"},
            "submit_sync_us": {"value": decode_submit_sync_us.iter().copied().sum::<u64>(), "status": "derived", "basis": "sum of per-step explicit Metal command-buffer submit plus wait clocks"},
            "logits_readback_bytes_per_step": {"value": VOCAB * 4, "status": "derived", "basis": "declared decode logits shape [49152] F32"},
            "greedy_token_sequence": {"value": greedy, "status": "measured", "basis": "first-index host argmax of declared logits readback"},
            "intermediate_readbacks": {"value": 0, "status": "measured", "basis": "only declared logits observation was read back per invocation"},
            "decode_wall_us": {"value": decode_wall_us, "status": "measured"},
        },
        "launch_plans": {"prefill": launch_rows_prefill, "decode": launch_rows_decode},
        "steps": step_receipts,
        "throughput": {
            "pp_ts": {"value": PREFILL_ROWS as f64 * 1_000_000.0 / prefill_wall_us.max(1) as f64, "status": "derived", "basis": "prompt rows / prefill wall"},
            "tg_ts": {"value": identity.decode_steps as f64 * 1_000_000.0 / decode_wall_us.max(1) as f64, "status": "derived", "basis": "decode steps / summed decode wall"},
        },
    });
    drop(gea3_release_programs(runtime, &mut programs));
    let parity = Gea3ParityTiming {
        residency: Gea3ParityResidency {
            weight_alloc_us: weight_allocations_us,
            weight_upload_us,
            weight_bytes,
            weight_uploads,
            kv_alloc_us: kv_allocations_us,
            kv_zero_fill_us: kv_zero_us,
            kv_zero_fill_bytes: kv_bytes,
            other_zero_fills,
        },
        steps: parity_steps,
    };
    Ok((evidence, parity))
}

#[test]
#[ignore = "physical Metal gate; run only with the exact §6 command"]
fn gea3_real_metal_decode_receipt() {
    gea3_physical_receipt_run(GEA3_FROZEN_SHORT);
}

/// GLP-U1b fixed-output-length runner: the same §6 physical path
/// re-specialized at the `metal-m5max-fixed1000` statue.  The capacity and
/// step count are compile-time statue facts; no runtime capacity argument or
/// session lifetime is introduced, and `DeviceProgramLifetime::SingleRun` is
/// retained by admission.
#[test]
#[ignore = "physical Metal gate; run only with the exact §6 command"]
fn gea3_fixed1000_metal_decode_receipt() {
    gea3_physical_receipt_run(GEA3_FIXED1000);
}

/// PPB-U7 soak runner: the same §6 physical path re-specialized at the
/// `metal-m5max-soak-l2000` statue.  The runner emits the U6 monotonic
/// produced-token counter as line-delimited stdout progress during decode
/// and streams the receipt, so a 60s-capped kill still yields the count, the
/// fresh-at-cap evidence, the KV residency, and the allocation cost.  The
/// capacity and step count are compile-time statue facts; no runtime
/// capacity argument or session lifetime is introduced, and
/// `DeviceProgramLifetime::SingleRun` is retained by admission.
#[test]
#[ignore = "physical Metal gate; run only with the exact §6 command"]
fn gea3_soak_metal_decode_receipt() {
    gea3_physical_receipt_run(GEA3_SOAK_L2000);
}

fn gea3_physical_receipt_run(identity: Gea3Identity) {
    std::env::set_var("FABER_PER_OP_TIMING", "1");
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("faberlang workspace root");
    let artifact_dir = gea3_artifact_dir();
    let receipt_path = PathBuf::from(
        std::env::var_os("GEA3_METAL_RECEIPT")
            .expect("GEA3_METAL_RECEIPT must identify the receipt output"),
    );
    let model_path = PathBuf::from(
        std::env::var_os("GEA3_F32_GGUF").expect("GEA3_F32_GGUF must identify the frozen F32 GGUF"),
    );
    assert!(
        model_path.is_file(),
        "missing GEA3 F32 GGUF {}",
        model_path.display()
    );

    let plan_started = Instant::now();
    let envelope = load_gea3_plan(&artifact_dir);
    let (prefill, decode) = map_both(&envelope, &artifact_dir, identity)
        .unwrap_or_else(|error| panic!("GEA3 plan → DeviceDescriptor mapping failed: {error}"));
    let plan_admission_us = u64::try_from(plan_started.elapsed().as_micros()).unwrap_or(u64::MAX);
    let production_reds = gea3_physical_plan_reds(&envelope, &prefill.0, &decode.0);

    let devices = enumerate_metal_physical_devices().expect("Metal device enumeration");
    assert!(
        !devices.is_empty(),
        "Metal selected but no physical device identity exists"
    );
    let device = &devices[0];
    let session_started = Instant::now();
    let session_result = CompositeHost::new(CompositeHostConfig::device(DeviceSelection::Metal))
        .and_then(|host| {
            host.require_implicit_local()?;
            Ok(host)
        });
    let session_admission_us =
        u64::try_from(session_started.elapsed().as_micros()).unwrap_or(u64::MAX);
    let session_admitted = session_result.is_ok();

    let bundle_manifest_path = artifact_dir.join("gea3-artifact-bundle-manifest.json");
    let bundle_manifest: Value = serde_json::from_slice(
        &fs::read(&bundle_manifest_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", bundle_manifest_path.display())),
    )
    .unwrap_or_else(|error| panic!("parse {}: {error}", bundle_manifest_path.display()));
    let input_manifest_path = workspace
        .join("radix/docs/factory/gpu-execution-architecture/evidence/gea3-input-manifest.json");
    let input_manifest: Value = serde_json::from_slice(
        &fs::read(&input_manifest_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", input_manifest_path.display())),
    )
    .unwrap_or_else(|error| panic!("parse {}: {error}", input_manifest_path.display()));
    let prompt_tokens: Vec<u32> = input_manifest["prompt_fixture"]["comparator_token_ids"]
        .as_array()
        .expect("frozen comparator prompt token ids")
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .expect("prompt token fits u32")
        })
        .collect();
    assert_eq!(prompt_tokens.len(), PREFILL_ROWS as usize);

    let mut receipt = json!({
        "schema": "gea3-metal-receipt-v1",
        "delivery": "GEA3-U5b",
        "status": if production_reds.is_empty() && session_admitted { "physical-run" } else { "blocked" },
        "machine": Command::new("hostname").output().ok().map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned()),
        "identities": {
            "model": input_manifest["model"],
            "plan_schema": envelope.schema,
            "source": envelope.source,
            "module_members": bundle_manifest["entries"],
            "module_image_rule": envelope.module_image_rule,
            "wire_buffer_v1_mirror_validation": {
                "status": "measured",
                "value": "admit_envelope + admit_program + DeviceDescriptor::validate",
                "basis": "GEA3-WIRE-BUFFER-V1 carried (buffer_id, content_version) identity; emitted MSL arity checked against every launch resource list",
            },
        },
        "revisions": {
            "gradus": gea3_git_revision(&workspace.join("gradus")),
            "radix": gea3_git_revision(&workspace.join("radix")),
            "hosts": gea3_git_revision(&workspace.join("hosts")),
        },
        "physical_device": {
            "backend": "Metal",
            "ordinal": device.ordinal,
            "registry_id": device.registry_id,
            "model": device.device_model,
            "api_total_bytes": device.api_total_bytes,
            "max_threads_per_workgroup": device.max_threads_per_workgroup,
            "workgroup_shared_memory_min_bytes": device.workgroup_shared_memory_min_bytes,
            "workgroup_shared_memory_max_bytes": device.workgroup_shared_memory_max_bytes,
            "collective_width": device.collective_width,
            "unified_memory": device.unified_memory,
        },
        "shapes": {
            "prefill_logits": envelope.declared_outputs.prefill_logits,
            "decode_logits": envelope.declared_outputs.decode_logits,
            "kv_capacity": envelope.kv_geometry.capacity,
            "kv_dtype": envelope.kv_geometry.dtype,
        },
        "preflight": {
            "plan_admission_us": {"value": plan_admission_us, "status": "measured"},
            "session_admitted": {"value": session_admitted, "status": "measured"},
            "session_admission_us": {"value": session_admission_us, "status": "measured"},
            "production_reds": production_reds,
        },
        "blocked_reason": production_reds,
        "measurement_policy": "No unmeasured physical field is promoted; production reds stop-and-amend.",
    });
    // PPB-U7: a non-frozen statue pins its identity facts in the receipt —
    // including a cap-killed partial — so the reducer validates the pinned
    // workload while the observed step count stays a measurement.
    if identity != GEA3_FROZEN_SHORT {
        receipt["statue"] = json!({
            "name": identity.name,
            "n_predict": identity.n_predict,
            "margin": identity.margin,
            "l_max": identity.l_max(),
            "l_max_formula": format!("prompt({PREFILL_ROWS}) + n_predict({}) + margin({})", identity.n_predict, identity.margin),
            "kv_capacity_rows": identity.history_capacity,
            "kv_bytes": identity.kv_bytes(),
            "kv_bytes_derivation": identity.kv_basis(),
            "decode_steps": identity.decode_steps,
            "single_run_lifetime_retained": true,
        });
    }
    let soak_stream = (identity != GEA3_FROZEN_SHORT).then(|| Gea3SoakStream {
        identity,
        base_receipt: receipt.clone(),
        receipt_path: receipt_path.clone(),
        companion_path: gea3_parity_companion_path(),
    });
    let mut parity_timing: Option<Gea3ParityTiming> = None;

    if production_reds.is_empty() && session_admitted {
        let mut host = session_result.expect("admitted Metal session disappeared");
        let execution = host
            .device_mut()
            .ok_or_else(|| "physical admission has no device runtime".to_owned())
            .and_then(|runtime| {
                gea3_run_physical(
                    runtime,
                    prefill,
                    decode,
                    &input_manifest,
                    &model_path,
                    &prompt_tokens,
                    identity,
                    soak_stream.as_ref(),
                )
            });
        match execution {
            Ok((execution, timing)) => {
                receipt["status"] = json!("green");
                receipt["residency"] = execution["residency"].clone();
                receipt["execution"] = execution["execution"].clone();
                receipt["launch_plans"] = execution["launch_plans"].clone();
                receipt["steps"] = execution["steps"].clone();
                receipt["throughput"] = execution["throughput"].clone();
                parity_timing = Some(timing);
            }
            Err(error) => {
                receipt["status"] = json!("blocked");
                receipt["blocked_reason"] = json!([error]);
                receipt["residency"] = json!({
                    "weight_allocations": gea3_unmeasured("physical execution failed before a complete receipt"),
                    "weight_bytes": gea3_unmeasured("physical execution failed before a complete receipt"),
                    "weight_upload_count": gea3_unmeasured("physical execution failed before a complete receipt"),
                    "kv_allocations": gea3_unmeasured("physical execution failed before a complete receipt"),
                    "kv_bytes": gea3_unmeasured("physical execution failed before a complete receipt"),
                    "kv_alloc_us": gea3_unmeasured("physical execution failed before a complete receipt"),
                    "zero_cpu_substitutes": {"value": 0, "status": "measured", "basis": "no CPU model execution was attempted"},
                    "zero_cpu_bridges": {"value": 0, "status": "measured", "basis": "no CPU model execution was attempted"},
                });
                receipt["execution"] = json!({
                    "prefill_wall_us": gea3_unmeasured("physical execution failed before a complete receipt"),
                    "per_step_gpu_body_us": gea3_gpu_not_measured("physical execution failed before a complete receipt", LAUNCHES_PER_PROGRAM),
                    "launch_encode_us_per_step": gea3_unmeasured("physical execution failed before a complete receipt"),
                    "submit_sync_us_per_step": gea3_unmeasured("physical execution failed before a complete receipt"),
                    "launches_per_step": gea3_unmeasured("physical execution failed before a complete receipt"),
                    "submit_sync_count": gea3_unmeasured("physical execution failed before a complete receipt"),
                    "launch_encode_us": gea3_unmeasured("physical execution failed before a complete receipt"),
                    "submit_sync_us": gea3_unmeasured("physical execution failed before a complete receipt"),
                    "logits_readback_bytes_per_step": gea3_unmeasured("physical execution failed before a complete receipt"),
                    "greedy_token_sequence": gea3_unmeasured("physical execution failed before a complete receipt"),
                    "intermediate_readbacks": {"value": 0, "status": "measured", "basis": "no complete physical execution receipt"},
                });
            }
        }
    } else {
        receipt["residency"] = json!({
            "weight_allocations": gea3_unmeasured("blocked before residency"),
            "weight_bytes": gea3_unmeasured("blocked before residency"),
            "weight_upload_count": gea3_unmeasured("blocked before residency"),
            "kv_allocations": gea3_unmeasured("blocked before residency"),
            "kv_bytes": gea3_unmeasured("blocked before residency"),
            "kv_alloc_us": gea3_unmeasured("blocked before residency"),
            "zero_cpu_substitutes": {"value": 0, "status": "measured", "basis": "no fake or CPU execution was attempted"},
            "zero_cpu_bridges": {"value": 0, "status": "measured", "basis": "no fake or CPU execution was attempted"},
        });
        receipt["execution"] = json!({
            "prefill_wall_us": gea3_unmeasured("blocked before dispatch"),
            "per_step_gpu_body_us": gea3_gpu_not_measured("blocked before dispatch", LAUNCHES_PER_PROGRAM),
            "launch_encode_us_per_step": gea3_unmeasured("blocked before dispatch"),
            "submit_sync_us_per_step": gea3_unmeasured("blocked before dispatch"),
            "launches_per_step": gea3_unmeasured("blocked before dispatch"),
            "submit_sync_count": gea3_unmeasured("blocked before dispatch"),
            "launch_encode_us": gea3_unmeasured("blocked before dispatch"),
            "submit_sync_us": gea3_unmeasured("blocked before dispatch"),
            "logits_readback_bytes_per_step": gea3_unmeasured("blocked before dispatch"),
            "greedy_token_sequence": gea3_unmeasured("blocked before dispatch"),
            "intermediate_readbacks": {"value": 0, "status": "measured", "basis": "no dispatch occurred"},
        });
    }

    let parent = receipt_path.parent().expect("receipt parent");
    fs::create_dir_all(parent).expect("create receipt parent");
    fs::write(
        &receipt_path,
        serde_json::to_vec_pretty(&receipt).expect("serialize GEA3 receipt"),
    )
    .unwrap_or_else(|error| panic!("write {}: {error}", receipt_path.display()));
    eprintln!("GEA3 physical Metal receipt: {}", receipt_path.display());

    // PPB-U3: the optional parity timing companion is emitted in addition
    // to the unchanged GEA3 receipt above, only when the environment names
    // an output path.  The GEA3 receipt is already complete and on disk.
    if let Some(companion_target) = gea3_parity_companion_path() {
        let timing = parity_timing
            .as_ref()
            .expect("parity companion requested but no physical timing was observed");
        let companion = gea3_build_parity_companion(timing, &receipt_path.display().to_string())
            .unwrap_or_else(|error| panic!("PPB-U3 companion boundary law violated: {error}"));
        gea3_write_parity_companion(Some(&companion_target), &companion);
    }

    assert_eq!(
        receipt["status"], "green",
        "GEA3-U5b physical receipt is blocked; receipt was written first"
    );
}

// ---------------------------------------------------------------------------
// GEA3 composition diagnostic.  This is deliberately separate from the U5b
// receipt path: it reads every produced intermediate after each dependency-
// ordered launch and stops at the first route transition whose upstream value
// was non-zero but whose declared consumer/output path is zero.  It is a
// diagnostic harness only, not a production receipt path.
// ---------------------------------------------------------------------------

fn gea3_run_staged_diagnostic(
    runtime: &mut DeviceRuntime,
    prefill: (DeviceDescriptor, Gea3WindowBindings),
    decode: (DeviceDescriptor, Gea3WindowBindings),
    decode_plan: &Gea3Program,
    prefill_plan: &Gea3Program,
    input_manifest: &Value,
    model_path: &Path,
    prompt_tokens: &[u32],
) -> Result<Value, String> {
    let model_ranges = gea3_model_ranges(input_manifest)?;
    let mapped = MappedWeightFile::open(model_path).map_err(|error| error.message.clone())?;
    runtime
        .retain_mapped_weight_file(mapped.clone())
        .map_err(|error| error.message.clone())?;
    let mut shared = BTreeMap::new();
    let mut programs = Vec::new();
    let result = (|| {
        // GEA3-U6 rerun #7: the staged route runs prefill first so the KV
        // arenas the decode route reads are the arenas prefill wrote (the
        // shared per-layer handles), not the zero-fill the decode-only
        // diagnostic of rerun #6 observed.
        let prefill_program =
            gea3_prepare_physical_program(runtime, prefill.0, prefill.1, &mut shared)?;
        programs.push(prefill_program);
        let program = gea3_prepare_physical_program(runtime, decode.0, decode.1, &mut shared)?;
        programs.push(program);
        let (weight_bytes, weight_uploads) = gea3_diagnostic_model_upload(
            runtime,
            programs.first().expect("diagnostic program prepared"),
            &shared,
            &model_ranges,
            &mapped,
        )?;
        // The decode program's own synthetic slots must not inherit
        // allocator contents either; the shared arenas are zeroed here (the
        // second zero is idempotent and precedes the prefill run).
        {
            let decode_program = programs.last().expect("decode program prepared");
            let slots = gea3_unique_slots(&decode_program.descriptor);
            let mut zeroed = BTreeSet::new();
            for (key, handle) in &decode_program.buffers {
                let slot = slots
                    .get(key)
                    .ok_or_else(|| "diagnostic slot metadata disappeared".to_owned())?;
                if gea3_canonical_model_name(&slot.buffer_name).is_some() {
                    continue;
                }
                if zeroed.insert(handle.id) {
                    gea3_zero_handle(runtime, handle)?;
                }
            }
        }
        let prefill_uploads = gea3_update_inputs(
            runtime,
            programs.first().expect("prefill program retained"),
            prompt_tokens,
            0,
            u32::try_from(prompt_tokens.len()).map_err(|_| "prompt is too long".to_owned())?,
            true,
            HISTORY_CAPACITY,
        )?;
        for launch in &programs[0].descriptor.launches {
            let kernel = &programs[0].descriptor.kernels[launch.kernel_index as usize];
            let bindings: Vec<DeviceLaunchBinding> = kernel
                .buffers
                .iter()
                .map(|slot| {
                    let handle = programs[0]
                        .buffers
                        .get(&(slot.buffer_id, slot.version))
                        .copied()
                        .ok_or_else(|| "prefill launch buffer disappeared".to_owned())?;
                    gea3_launch_binding(handle, &programs[0], launch.kernel_index, slot)
                })
                .collect::<Result<_, String>>()?;
            runtime
                .launch_kernel_bound(
                    &programs[0].module,
                    &kernel.entry,
                    &bindings,
                    kernel.grid,
                    kernel.block,
                )
                .map_err(|error| error.message.clone())?;
        }
        runtime.sync().map_err(|error| error.message.clone())?;
        // Arena evidence: read every shared KV arena through the decode
        // program's handles right after prefill — the mutator launches must
        // have left them non-zero.
        let decode_program = programs.last().expect("decode program retained");
        let mut arena_summaries = Vec::new();
        let mut arena_handles = BTreeSet::new();
        for kernel in &decode_plan.kernels {
            if kernel.entry != "kv_append_k" && kernel.entry != "kv_append_v" {
                continue;
            }
            let resource = &kernel.resources[0];
            let handle = decode_program
                .buffers
                .get(&(resource.buffer.id, resource.version.version))
                .copied()
                .ok_or_else(|| "arena handle disappeared".to_owned())?;
            if !arena_handles.insert(handle.id) {
                continue;
            }
            arena_summaries.push(gea3_diagnostic_buffer_summary(runtime, &handle, resource)?);
        }
        let arenas_non_zero = arena_summaries.len() == LAYERS * 2
            && arena_summaries
                .iter()
                .all(|summary| summary["non_zero"].as_bool().unwrap_or(false));
        // GEA3-U6 num-2 localization: layer 0's prefill write lands only one
        // row while layers 1-31 write all 36 rows with byte-identical plan
        // bindings.  The plan cannot be the delta, so the runtime inputs are:
        // record the layer-0 vs layer-1 producer chain (embedding gather,
        // attention rmsnorm, both kv projections, rope_k, residual add) and
        // the per-row non-zero census of both layers' arenas right after
        // prefill.
        let prefill_program = programs.first().expect("prefill program retained");
        let mut chain_summaries = Vec::new();
        let probed_entries = [
            "embedding_gather",
            "prefill_rmsnorm",
            "prefill_gemm_kv",
            "prefill_rope_k",
            "prefill_residual_add",
        ];
        for kernel in &prefill_plan.kernels {
            if !probed_entries.contains(&kernel.entry.as_str()) {
                continue;
            }
            let layer = kernel.layer;
            if layer > 1 {
                continue;
            }
            for resource in &kernel.resources {
                if resource.access != Gea3ResourceAccess::Write
                    || gea3_diagnostic_resource_is_weight(resource)
                {
                    continue;
                }
                let handle = prefill_program
                    .buffers
                    .get(&(resource.buffer.id, resource.version.version))
                    .copied()
                    .ok_or_else(|| "probe handle disappeared".to_owned())?;
                chain_summaries.push(gea3_diagnostic_buffer_summary(runtime, &handle, resource)?);
            }
        }
        let mut arena_row_census = Vec::new();
        for kernel in &decode_plan.kernels {
            if kernel.entry != "kv_append_k" && kernel.entry != "kv_append_v" {
                continue;
            }
            let layer = kernel.layer;
            if layer > 1 {
                continue;
            }
            let resource = &kernel.resources[0];
            let handle = decode_program
                .buffers
                .get(&(resource.buffer.id, resource.version.version))
                .copied()
                .ok_or_else(|| "arena handle disappeared".to_owned())?;
            let bytes = runtime
                .readback_bytes(&handle, DeviceDataType::F32)
                .map_err(|error| error.message.clone())?;
            let values = gea3_f32_from_bytes(&bytes)?;
            let rows: Vec<u64> = values
                .chunks(KV_WIDTH as usize)
                .map(|row| row.iter().filter(|value| **value != 0.0).count() as u64)
                .collect();
            arena_row_census.push(json!({
                "name": resource.buffer.name,
                "entry": kernel.entry,
                "layer": layer,
                "non_zero_per_row": rows,
                "rows_written": rows.iter().filter(|count| **count > 0).count(),
            }));
        }
        // GEA3-U6 num-9 write-side probe (env-gated, full f32 arrays): the
        // num-8 receipts proved the layer-1 arena bytes equal the layer-1
        // prefill rope_k / k-projection outputs, so the WRITE is faithful —
        // the divergence is upstream in the prefill attention chain.  This
        // probe dumps every non-weight write buffer of layers 0..1 (q/k/v
        // projections, rope_q/rope_k, per-head score/softmax/transpose/
        // context, both residuals) plus all four probed arenas so the
        // offline comparator can diff per element against the scalar oracle.
        let mut write_side_probe = Vec::new();
        if std::env::var_os("GEA3_WRITE_SIDE_PROBE").is_some() {
            let probe_entries = [
                "prefill_rmsnorm",
                "prefill_gemm_qo",
                "prefill_gemm_o",
                "prefill_gemm_kv",
                "prefill_rope_q",
                "prefill_rope_k",
                "prefill_key_transpose",
                "prefill_score_gemm",
                "prefill_causal_softmax",
                "prefill_context_gemm",
                "prefill_residual_add",
                "prefill_swiglu",
            ];
            for kernel in &prefill_plan.kernels {
                if !probe_entries.contains(&kernel.entry.as_str()) || kernel.layer > 1 {
                    continue;
                }
                for resource in &kernel.resources {
                    if resource.access != Gea3ResourceAccess::Write
                        || gea3_diagnostic_resource_is_weight(resource)
                    {
                        continue;
                    }
                    let handle = prefill_program
                        .buffers
                        .get(&(resource.buffer.id, resource.version.version))
                        .copied()
                        .ok_or_else(|| "probe handle disappeared".to_owned())?;
                    let bytes = runtime
                        .readback_bytes(&handle, DeviceDataType::F32)
                        .map_err(|error| error.message.clone())?;
                    let values = gea3_f32_from_bytes(&bytes)?;
                    write_side_probe.push(json!({
                        "entry": kernel.entry,
                        "layer": kernel.layer,
                        "ordinal": kernel.ordinal,
                        "name": resource.buffer.name,
                        "element_count": values.len(),
                        "values": values,
                    }));
                }
            }
            for kernel in &decode_plan.kernels {
                if kernel.entry != "kv_append_k" && kernel.entry != "kv_append_v" {
                    continue;
                }
                let layer = kernel.layer;
                if layer > 1 {
                    continue;
                }
                let resource = &kernel.resources[0];
                let handle = decode_program
                    .buffers
                    .get(&(resource.buffer.id, resource.version.version))
                    .copied()
                    .ok_or_else(|| "arena handle disappeared".to_owned())?;
                let bytes = runtime
                    .readback_bytes(&handle, DeviceDataType::F32)
                    .map_err(|error| error.message.clone())?;
                let values = gea3_f32_from_bytes(&bytes)?;
                write_side_probe.push(json!({
                    "entry": kernel.entry,
                    "layer": layer,
                    "ordinal": kernel.ordinal,
                    "name": resource.buffer.name,
                    "element_count": values.len(),
                    "values": values,
                }));
            }
        }
        let prefill_launch_count = programs[0].descriptor.launches.len();
        let program = programs.last().expect("decode program retained");
        let token = *prompt_tokens
            .last()
            .ok_or_else(|| "frozen comparator prompt is empty".to_owned())?;
        let input_uploads = gea3_update_inputs(
            runtime,
            program,
            &[token],
            u32::try_from(prompt_tokens.len()).map_err(|_| "prompt is too long".to_owned())?,
            u32::try_from(prompt_tokens.len() + 1)
                .map_err(|_| "decode valid length overflows".to_owned())?,
            false,
            HISTORY_CAPACITY,
        )?;

        let mut stages = Vec::new();
        let mut previous_output = None;
        let mut first_bad = None;
        let mut first_non_finite = None;
        for launch in &program.descriptor.launches {
            let descriptor_kernel = &program.descriptor.kernels[launch.kernel_index as usize];
            let plan_kernel = &decode_plan.kernels[launch.kernel_index as usize];
            let bindings: Vec<DeviceLaunchBinding> = descriptor_kernel
                .buffers
                .iter()
                .map(|slot| {
                    let handle = program
                        .buffers
                        .get(&(slot.buffer_id, slot.version))
                        .copied()
                        .ok_or_else(|| "diagnostic launch buffer disappeared".to_owned())?;
                    gea3_launch_binding(handle, program, launch.kernel_index, slot)
                })
                .collect::<Result<_, String>>()?;
            runtime
                .launch_kernel_bound(
                    &program.module,
                    &descriptor_kernel.entry,
                    &bindings,
                    descriptor_kernel.grid,
                    descriptor_kernel.block,
                )
                .map_err(|error| error.message.clone())?;
            runtime.sync().map_err(|error| error.message.clone())?;

            let inputs = gea3_diagnostic_read_slots(runtime, program, plan_kernel, true)?;
            let outputs = gea3_diagnostic_read_slots(runtime, program, plan_kernel, false)?;
            let primary_input = gea3_diagnostic_primary_slot(plan_kernel, true)
                .ok_or_else(|| format!("{} has no diagnostic input", plan_kernel.entry))?;
            let primary_output = gea3_diagnostic_primary_slot(plan_kernel, false)
                .ok_or_else(|| format!("{} has no diagnostic output", plan_kernel.entry))?;
            let primary_input_summary = inputs
                .iter()
                .find(|summary| {
                    summary["buffer_id"].as_u64() == Some(u64::from(primary_input.buffer.id))
                        && summary["version"].as_u64()
                            == Some(u64::from(primary_input.version.version))
                })
                .cloned()
                .ok_or_else(|| "primary diagnostic input summary disappeared".to_owned())?;
            let primary_output_summary = outputs
                .iter()
                .find(|summary| {
                    summary["buffer_id"].as_u64() == Some(u64::from(primary_output.buffer.id))
                        && summary["version"].as_u64()
                            == Some(u64::from(primary_output.version.version))
                })
                .cloned()
                .ok_or_else(|| "primary diagnostic output summary disappeared".to_owned())?;
            let expected_buffer = gea3_diagnostic_expected_buffer(&program.descriptor, launch.id);
            let upstream_non_zero = previous_output
                .as_ref()
                .is_some_and(gea3_diagnostic_summary_non_zero);
            let output_non_zero = gea3_diagnostic_summary_non_zero(&primary_output_summary);
            let wiring_mismatch =
                expected_buffer.is_some_and(|(_, buffer_id)| buffer_id != primary_input.buffer.id);
            let stage_is_bad =
                launch.id > 1 && upstream_non_zero && (!output_non_zero || wiring_mismatch);
            let abi = plan_kernel
                .resources
                .iter()
                .map(|resource| {
                    json!({
                        "binding": resource.binding.binding,
                        "access": format!("{:?}", resource.access),
                        "buffer_id": resource.buffer.id,
                        "version": resource.version.version,
                        "name": resource.buffer.name,
                        "role": format!("{:?}", resource.buffer.role),
                        "element_count": resource.version.element_count,
                    })
                })
                .collect::<Vec<_>>();
            // Non-finite localization: any produced output slot carrying a
            // NaN or +/-inf flags the launch.  The first such launch is named
            // with every non-weight input slot's finiteness state so the seam
            // (finite inputs -> non-finite output) is machine-readable.
            let non_finite_outputs: Vec<Value> = outputs
                .iter()
                .filter(|summary| summary["non_finite"].as_bool().unwrap_or(false))
                .map(|summary| {
                    json!({
                        "name": summary["name"],
                        "buffer_id": summary["buffer_id"],
                        "element_count": summary["element_count"],
                        "finite_count": summary["finite_count"],
                        "non_finite_count": summary["non_finite_count"],
                        "nan_count": summary["nan_count"],
                        "pos_inf_count": summary["pos_inf_count"],
                        "neg_inf_count": summary["neg_inf_count"],
                        "first_non_finite": summary["first_non_finite"],
                    })
                })
                .collect();
            let stage = json!({
                "launch_id": launch.id,
                "kernel_index": launch.kernel_index,
                "layer": plan_kernel.layer,
                "ordinal": plan_kernel.ordinal,
                "entry": plan_kernel.entry,
                "abi": {
                    "binding_count": plan_kernel.resources.len(),
                    "bindings": abi,
                },
                "expected_dependency": expected_buffer.map(|(producer, buffer_id)| json!({
                    "producer_launch": producer,
                    "buffer_id": buffer_id,
                })),
                "input": primary_input_summary,
                "inputs_finiteness": inputs
                    .iter()
                    .map(|summary| {
                        json!({
                            "name": summary["name"],
                            "buffer_id": summary["buffer_id"],
                            "element_count": summary["element_count"],
                            "finite_count": summary["finite_count"],
                            "non_finite_count": summary["non_finite_count"],
                        })
                    })
                    .collect::<Vec<_>>(),
                "non_finite_outputs": non_finite_outputs.clone(),
                "outputs": outputs,
                "primary_output": primary_output_summary,
                "upstream_primary_output": previous_output,
                "wiring_mismatch": wiring_mismatch,
                "stage_is_bad": stage_is_bad,
                "readback_policy": "sync after this launch; read all non-weight inputs and produced outputs",
            });
            if stage_is_bad && first_bad.is_none() {
                first_bad = Some(json!({
                    "launch_id": launch.id,
                    "entry": plan_kernel.entry,
                    "abi": stage["abi"].clone(),
                    "buffer_pair": {
                        "expected_source": stage["expected_dependency"].clone(),
                        "upstream_output": stage["upstream_primary_output"].clone(),
                        "actual_consumer_input": stage["input"].clone(),
                        "destination": stage["primary_output"].clone(),
                    },
                    "evidence": {
                        "upstream_non_zero": upstream_non_zero,
                        "actual_input_non_zero": gea3_diagnostic_summary_non_zero(&stage["input"]),
                        "destination_non_zero": output_non_zero,
                        "wiring_mismatch": wiring_mismatch,
                    },
                }));
            }
            if !non_finite_outputs.is_empty() && first_non_finite.is_none() {
                first_non_finite = Some(json!({
                    "launch_id": launch.id,
                    "layer": plan_kernel.layer,
                    "ordinal": plan_kernel.ordinal,
                    "entry": plan_kernel.entry,
                    "outputs": non_finite_outputs,
                    "inputs_finiteness": stage["inputs_finiteness"].clone(),
                    "primary_input": stage["input"].clone(),
                }));
            }
            stages.push(stage);
            previous_output = Some(primary_output_summary);
            if first_bad.is_some() {
                break;
            }
        }

        // GEA3-U6 num-11 terminal probe (env-gated, full f32 arrays): the
        // rerun-#14 offline join names the terminal logits drift (L32
        // lm_head_gemv, ~4.4e-5 vs the frozen 2e-5 terminal policy) as the
        // only non-match row, but the per-stage summaries echo first-values
        // only and cannot decompose a drift that small.  This probe dumps
        // the post-L31 path in full — the layer-31 residual stream entering
        // the head (both occurrences), the final-norm output, and the logits
        // row — so the offline comparator can walk the terminal ops per op
        // (final norm vs lm_head gemm vs writeback/dtype seams) on the
        // device's own bytes.
        let mut terminal_probe = Vec::new();
        if std::env::var_os("GEA3_WRITE_SIDE_PROBE").is_some() {
            for kernel in &decode_plan.kernels {
                let probed = match kernel.entry.as_str() {
                    "decode_residual_add" => kernel.layer == 31,
                    "head_rmsnorm" | "lm_head_gemv" => true,
                    _ => false,
                };
                if !probed {
                    continue;
                }
                for resource in &kernel.resources {
                    if resource.access != Gea3ResourceAccess::Write
                        || gea3_diagnostic_resource_is_weight(resource)
                    {
                        continue;
                    }
                    let handle = program
                        .buffers
                        .get(&(resource.buffer.id, resource.version.version))
                        .copied()
                        .ok_or_else(|| "terminal probe handle disappeared".to_owned())?;
                    let bytes = runtime
                        .readback_bytes(&handle, DeviceDataType::F32)
                        .map_err(|error| error.message.clone())?;
                    let values = gea3_f32_from_bytes(&bytes)?;
                    terminal_probe.push(json!({
                        "entry": kernel.entry,
                        "layer": kernel.layer,
                        "ordinal": kernel.ordinal,
                        "name": resource.buffer.name,
                        "element_count": values.len(),
                        "values": values,
                    }));
                }
            }
        }

        let terminal = stages.last().map(|stage| {
            json!({
                "launch_id": stage["launch_id"],
                "entry": stage["entry"],
                "buffer": stage["primary_output"],
                "non_zero": gea3_diagnostic_summary_non_zero(&stage["primary_output"]),
            })
        });
        Ok(json!({
            "status": if first_bad.is_some() {
                "first-bad-entry"
            } else {
                "composition-through-logits-writeback"
            },
            "route": "prefill-then-decode-step",
            "token": token,
            "prefill": {
                "launch_count": prefill_launch_count,
                "input_uploads": prefill_uploads.copies,
                "arena_count": arena_summaries.len(),
                "arenas_non_zero": arenas_non_zero,
                "arena_summaries_after_prefill": arena_summaries,
                "layer0_localization": {
                    "chain_summaries": chain_summaries,
                    "arena_row_census": arena_row_census,
                },
                "write_side_probe": write_side_probe,
            },
            "input_uploads": input_uploads.copies,
            "weights": {
                "upload_count": weight_uploads,
                "bytes": weight_bytes,
            },
            "stages_executed": stages.len(),
            "stages": stages,
            "terminal_probe": terminal_probe,
            "first_bad": first_bad,
            "first_non_finite": first_non_finite,
            "terminal": terminal,
            "diagnostic_policy": "real Metal, real F32 GGUF weights, dependency-order launches, internal F32 readback after every stage; no CPU model substitute",
        }))
    })();
    let release = gea3_release_programs(runtime, &mut programs);
    match (result, release) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Err(error), Err(release_error)) => Err(format!(
            "{error}; diagnostic teardown failed: {release_error}"
        )),
    }
}

#[test]
#[ignore = "diagnostic physical Metal gate; requires exact GEA3 artifact/model/receipt env"]
fn gea3_real_metal_staged_composition_diagnostic() {
    std::env::set_var("FABER_PER_OP_TIMING", "1");
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("faberlang workspace root");
    let artifact_dir = gea3_artifact_dir();
    let receipt_path = PathBuf::from(
        std::env::var_os("GEA3_DIAGNOSTIC_RECEIPT")
            .expect("GEA3_DIAGNOSTIC_RECEIPT must identify the diagnostic output"),
    );
    let model_path = PathBuf::from(
        std::env::var_os("GEA3_F32_GGUF").expect("GEA3_F32_GGUF must identify the frozen F32 GGUF"),
    );
    assert!(
        model_path.is_file(),
        "missing GEA3 F32 GGUF {}",
        model_path.display()
    );
    let envelope = load_gea3_plan(&artifact_dir);
    let (prefill, decode) = map_both(&envelope, &artifact_dir, GEA3_FROZEN_SHORT)
        .unwrap_or_else(|error| panic!("GEA3 plan → DeviceDescriptor mapping failed: {error}"));
    let input_manifest_path = workspace
        .join("radix/docs/factory/gpu-execution-architecture/evidence/gea3-input-manifest.json");
    let input_manifest: Value = serde_json::from_slice(
        &fs::read(&input_manifest_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", input_manifest_path.display())),
    )
    .unwrap_or_else(|error| panic!("parse {}: {error}", input_manifest_path.display()));
    let prompt_tokens: Vec<u32> = input_manifest["prompt_fixture"]["comparator_token_ids"]
        .as_array()
        .expect("frozen comparator prompt token ids")
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .expect("prompt token fits u32")
        })
        .collect();

    let devices = enumerate_metal_physical_devices().expect("Metal device enumeration");
    assert!(
        !devices.is_empty(),
        "Metal selected but no physical device identity exists"
    );
    let device = &devices[0];
    let session_result = CompositeHost::new(CompositeHostConfig::device(DeviceSelection::Metal))
        .and_then(|host| {
            host.require_implicit_local()?;
            Ok(host)
        });
    let mut receipt = json!({
        "schema": "gea3-staged-composition-diagnostic-v1",
        "delivery": "GEA3-U7-GEA3-numerical-closeout-diagnostic",
        "status": if session_result.is_ok() { "physical-run" } else { "blocked" },
        "diagnostic_only": true,
        "machine": Command::new("hostname").output().ok().map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned()),
        "revisions": {
            "gradus": gea3_git_revision(&workspace.join("gradus")),
            "radix": gea3_git_revision(&workspace.join("radix")),
            "hosts": gea3_git_revision(&workspace.join("hosts")),
        },
        "physical_device": {
            "backend": "Metal",
            "ordinal": device.ordinal,
            "registry_id": device.registry_id,
            "model": device.device_model,
        },
        "plan": {
            "schema": envelope.schema,
            "source": envelope.source,
            "program": "decode-step",
            "launch_count": decode.0.launches.len(),
            "dependency_count": decode.0.data_flow.len(),
        },
        "measurement_policy": "No production receipt fields are changed; this harness stops only after naming a first bad route entry or proving every stage through logits writeback.",
    });
    if let Ok(mut host) = session_result {
        let execution = host
            .device_mut()
            .ok_or_else(|| "physical admission has no device runtime".to_owned())
            .and_then(|runtime| {
                gea3_run_staged_diagnostic(
                    runtime,
                    prefill,
                    decode,
                    &envelope.programs.decode_step,
                    &envelope.programs.prefill,
                    &input_manifest,
                    &model_path,
                    &prompt_tokens,
                )
            });
        match execution {
            Ok(execution) => {
                receipt["status"] = json!(execution["status"]);
                receipt["execution"] = execution;
            }
            Err(error) => {
                receipt["status"] = json!("blocked");
                receipt["blocked_reason"] = json!([error]);
            }
        }
    } else {
        receipt["blocked_reason"] = json!(["physical Metal session admission failed"]);
    }
    let parent = receipt_path.parent().expect("diagnostic receipt parent");
    fs::create_dir_all(parent).expect("create diagnostic receipt parent");
    fs::write(
        &receipt_path,
        serde_json::to_vec_pretty(&receipt).expect("serialize staged diagnostic receipt"),
    )
    .unwrap_or_else(|error| panic!("write {}: {error}", receipt_path.display()));
    eprintln!(
        "GEA3 staged composition diagnostic: {}",
        receipt_path.display()
    );

    // GEA3-U6 rerun #5: the repaired plan composes through logits writeback
    // (with NaN logits until the per-head shape residual is amended), so the
    // diagnostic law is the disjunction: name a first-bad entry OR prove the
    // route through the logits writeback.
    assert!(
        receipt["status"] == "first-bad-entry"
            || receipt["status"] == "composition-through-logits-writeback",
        "diagnostic must name the first bad composition entry or prove the route through logits"
    );
    assert!(
        receipt["execution"]["first_bad"].is_object()
            || receipt["execution"]["status"] == "composition-through-logits-writeback",
        "diagnostic receipt lacks first-bad evidence or terminal proof"
    );
    // GEA3-U6 rerun #7: the KV mutators launch, so the arenas the decode
    // route reads must be non-zero after prefill — the empty-arena artifact
    // of rerun #6 is closed.  Recorded honestly: if this fires, the append
    // launches or their slot/block operands are wrong.
    assert!(
        receipt["execution"]["prefill"]["arenas_non_zero"] == json!(true),
        "KV arenas are zero after prefill; the mutator launches did not write"
    );
    assert_eq!(
        receipt["execution"]["prefill"]["arena_count"],
        json!(LAYERS * 2)
    );
    // GEA3-U6 rerun #6: non-finite localization.  When any launch produced a
    // non-finite element, the receipt must name that FIRST launch with its
    // inputs' finiteness state — never leave the NaN origin anonymous.
    assert!(
        receipt["execution"]["first_non_finite"].is_null()
            || receipt["execution"]["first_non_finite"]["launch_id"]
                .as_u64()
                .is_some(),
        "first non-finite launch must be named with its launch id"
    );
}
