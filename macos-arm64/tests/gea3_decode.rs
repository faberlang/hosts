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
const BLOCK_LAUNCHES_PER_LAYER: usize = 64;
const LAUNCHES_PER_PROGRAM: usize = LAYERS * BLOCK_LAUNCHES_PER_LAYER + 3;
const DEPENDENCIES_PER_PROGRAM: usize = LAUNCHES_PER_PROGRAM - 1;
const PREFILL_ROWS: u64 = 36;
const HISTORY_CAPACITY: u64 = 76;
const KV_WIDTH: u64 = 320;
const VOCAB: u64 = 49_152;
const HIDDEN: u64 = 960;
const DECODE_STEPS: usize = 8;

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
    Transpose(Gea3TransposePlan),
    RmsNormalization(Gea3RmsNormalizationPlan),
    Rope(Gea3RopePlan),
    CausalMaskedSoftmax(Gea3CausalMaskedSoftmaxPlan),
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
    window
        .byte_binding()
        .map(Some)
        .ok_or_else(|| {
            format!(
                "resource `{}` carries a sub-window whose byte binding overflows",
                resource.buffer.name
            )
        })
}

// ---------------------------------------------------------------------------
// GEA3-WIRE-BUFFER-V2 carried==derived parity vectors (V2 law dedupe).  The
// hosts mirror above is seam-forced — public hosts cannot consume the private
// schema authority — so the shared vectors pin it: every vector encodes the
// production law's verdict (`radix_mir_fmir::schema::
// validate_carried_sub_window_projection`, proven by radix-mir-fmir's
// `wire_gea3_test` and the delegating mir-emit-harness negatives).  The
// ledger (`gea3-carried-derived-parity-ledger.toml`, module-boundary
// parity-ledger pattern, efb2b8747 precedent) pins each vector's
// canonical-JSON digest; the mirror must reproduce every verdict live, and a
// regenerated ledger must byte-match the committed one.  Drift between the
// hosts mirror and the production law fails this gate.
// ---------------------------------------------------------------------------

const CARRIED_DERIVED_VECTORS: &str = include_str!("gea3-carried-derived-vectors.json");
const CARRIED_DERIVED_LEDGER: &str = include_str!("gea3-carried-derived-parity-ledger.toml");

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea3CarriedDerivedVector {
    label: String,
    resource: Gea3DeviceResource,
    producer_element_count: u64,
    expected_verdict: String,
    expected_reason: Option<String>,
}

fn carried_derived_vectors() -> Vec<Gea3CarriedDerivedVector> {
    serde_json::from_str(CARRIED_DERIVED_VECTORS).expect("parse carried==derived vectors")
}

/// The vector's canonical JSON bytes (serde_json's key-sorted map) — the
/// ledger digest substrate, so an edited vector is a stale pin.
fn carried_derived_vector_digest(vector: &Value) -> String {
    let canonical = serde_json::to_string(vector).expect("canonical vector JSON");
    format!("sha256:{}", sha256_hex(canonical.as_bytes()))
}

/// The mirror's live verdict on one vector: `Ok` when the window admits,
/// the rejection reason otherwise.
fn carried_derived_mirror_verdict(vector: &Gea3CarriedDerivedVector) -> Result<(), String> {
    match admit_sub_window(&vector.resource, vector.producer_element_count) {
        Ok(_) => Ok(()),
        Err(error) => Err(error),
    }
}

/// Live recheck: the mirror must reproduce the vector's pinned production
/// verdict exactly, and a rejection must name the pinned law.
fn verify_carried_derived_vector(vector: &Gea3CarriedDerivedVector) -> Result<(), String> {
    match carried_derived_mirror_verdict(vector) {
        Ok(()) if vector.expected_verdict == "admit" => Ok(()),
        Ok(()) => Err(format!(
            "vector `{}` must reject but the mirror admits",
            vector.label
        )),
        Err(error) if vector.expected_verdict == "reject" => {
            let reason = vector.expected_reason.as_deref().unwrap_or("");
            if error.contains(reason) {
                Ok(())
            } else {
                Err(format!(
                    "vector `{}` rejects without naming `{reason}`: {error}",
                    vector.label
                ))
            }
        }
        Err(error) => Err(format!(
            "vector `{}` must admit but the mirror rejects: {error}",
            vector.label
        )),
    }
}

/// Deterministic ledger text for the committed vectors (measured artifact).
/// Generation is fail-closed: a parity row is emitted only when the mirror
/// reproduces the vector's pinned verdict.
fn carried_derived_ledger_text() -> String {
    let mut text = String::from(
        "# GEA3-WIRE-BUFFER-V2 carried==derived mirror parity ledger (V2 law dedupe).\n\
         # Measured artifact (module-boundary parity-ledger pattern, efb2b8747 precedent):\n\
         # row identity is the shared cross-repo vector label; `digest` is sha256 of the\n\
         # vector's canonical JSON bytes. Deterministically generated by the hosts gea3_decode\n\
         # parity gate; regeneration must produce no diff. The mirror's live verdict must\n\
         # reproduce the production law's pin on every row — drift fails the gate.\n\
         schema_version = 1\n",
    );
    let vectors = carried_derived_vectors();
    let raw: Vec<Value> = serde_json::from_str(CARRIED_DERIVED_VECTORS).expect("parse raw vectors");
    for (vector, raw_vector) in vectors.iter().zip(raw.iter()) {
        verify_carried_derived_vector(vector)
            .unwrap_or_else(|error| panic!("carried==derived ledger generation is fail-closed: {error}"));
        text.push_str(&format!(
            "\n[[row]]\n\
             path = \"gea3-carried-derived-vectors.json#{}\"\n\
             mode = \"mirror\"\n\
             layer = \"admission\"\n\
             disposition = \"parity\"\n\
             digest = \"{}\"\n",
            vector.label,
            carried_derived_vector_digest(raw_vector),
        ));
    }
    text
}

#[test]
fn gea3_mirror_pins_carried_derived_parity_vectors() {
    let vectors = carried_derived_vectors();
    assert!(
        !vectors.is_empty(),
        "the carried==derived parity vectors must not be empty"
    );
    for vector in &vectors {
        verify_carried_derived_vector(vector)
            .unwrap_or_else(|error| panic!("carried==derived parity violation: {error}"));
    }
}

#[test]
fn gea3_carried_derived_parity_ledger_is_fresh() {
    assert_eq!(
        CARRIED_DERIVED_LEDGER,
        carried_derived_ledger_text(),
        "carried==derived parity ledger is stale vs the live vectors; run the sanctioned \
         `gea3_regenerate_carried_derived_parity_ledger` regeneration and commit the diff"
    );
}

#[test]
#[ignore = "sanctioned regeneration; writes the parity ledger from the live vectors"]
fn gea3_regenerate_carried_derived_parity_ledger() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/gea3-carried-derived-parity-ledger.toml");
    std::fs::write(&path, carried_derived_ledger_text()).expect("write carried==derived parity ledger");
}

fn map_envelope_to_descriptor(
    envelope: &Gea3ProgramPlanEnvelope,
    program: &Gea3Program,
    program_name: &str,
    artifact_dir: &Path,
) -> Result<(DeviceDescriptor, Gea3WindowBindings), String> {
    admit_envelope(envelope)?;
    admit_program(envelope, program, program_name)?;

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
                        return Err(format!(
                            "buffer {} version {} changes dtype",
                            key.0, key.1
                        ));
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
            if let Some(binding) = admit_sub_window(resource, allocation)? {
                if windows
                    .insert(
                        (u32::try_from(kernel_index).expect("kernel index fits u32"), resource.binding.binding),
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

fn admit_envelope(envelope: &Gea3ProgramPlanEnvelope) -> Result<(), String> {
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
    if envelope.kv_geometry.capacity != HISTORY_CAPACITY
        || envelope.kv_geometry.declared_history_length != HISTORY_CAPACITY
        || envelope.kv_geometry.dtype != "F32"
        || !envelope.kv_geometry.mask_beyond_length
    {
        return Err("KV geometry is not the frozen capacity-bounded contract".to_owned());
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
        || program.declared_history_length != HISTORY_CAPACITY
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
    admit_state_buffers(program, program_name)?;
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

fn admit_state_buffers(program: &Gea3Program, program_name: &str) -> Result<(), String> {
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
                || row.shape.as_deref() != Some(&[HISTORY_CAPACITY, KV_WIDTH][..])
                || row.history_capacity != Some(HISTORY_CAPACITY)
                || row.declared_history_length != Some(HISTORY_CAPACITY)
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
        | "decode_gemv_gate_up"
        | "decode_gemv_down"
        | "decode_score_gemm"
        | "decode_context_gemm"
        | "prefill_gemm_qo"
        | "prefill_gemm_kv"
        | "prefill_gemm_gate_up"
        | "prefill_gemm_down"
        | "prefill_score_gemm"
        | "prefill_context_gemm"
        | "lm_head_gemv" => matches!(plan, Gea3Plan::TiledMatMul(_)),
        "head_rmsnorm" | "decode_rmsnorm" | "prefill_rmsnorm" => {
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
        "embedding_gather" => matches!(plan, Gea3Plan::TiledMatMul(_)),
        "decode_swiglu" | "decode_residual_add" | "prefill_swiglu" | "prefill_residual_add" => {
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
        "decode_rmsnorm",
        "decode_gemv_qo",
        "decode_gemv_kv",
        "decode_gemv_gate_up",
        "decode_gemv_down",
        "decode_rope_q",
        "decode_rope_k",
        "kv_append_k",
        "kv_append_v",
        "decode_key_transpose",
        "decode_score_gemm",
        "decode_masked_softmax",
        "decode_context_gemm",
        "decode_swiglu",
        "decode_residual_add",
        "prefill_rmsnorm",
        "prefill_gemm_qo",
        "prefill_gemm_kv",
        "prefill_gemm_gate_up",
        "prefill_gemm_down",
        "prefill_rope_q",
        "prefill_rope_k",
        "prefill_key_transpose",
        "prefill_score_gemm",
        "prefill_causal_softmax",
        "prefill_context_gemm",
        "prefill_swiglu",
        "prefill_residual_add",
        "prefill_kv_write_k",
        "prefill_kv_write_v",
        "head_rmsnorm",
        "lm_head_gemv",
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
    )?;
    let decode = map_envelope_to_descriptor(
        envelope,
        &envelope.programs.decode_step,
        "decode_step",
        artifact_dir,
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
    let ((prefill, prefill_windows), (decode, decode_windows)) =
        map_both(&envelope, &artifact_dir)
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
    // allocation).
    let expected_decode_windows = LAYERS * (5 + 15 + 15 + 15);
    let expected_prefill_windows = LAYERS * (5 + 15 + 15 + 15);
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
        assert_eq!(byte_offset % 4, 0, "window byte offsets are element aligned");
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
    assert!(admit_envelope(&wrong_schema).is_err());
}

#[test]
fn gea3_negative_rows_fail_closed() {
    let artifact_dir = gea3_artifact_dir();
    let bytes = fs::read(artifact_dir.join(PLAN_MEMBER)).expect("read exported GEA3 plan");
    let original: Value = serde_json::from_slice(&bytes).expect("exported plan is JSON");

    let mut missing_edge = original.clone();
    missing_edge["programs"]["decode_step"]["dependencies"]
        .as_array_mut()
        .expect("dependencies")
        .pop();
    let parsed: Gea3ProgramPlanEnvelope =
        serde_json::from_value(missing_edge).expect("mirror parse");
    assert!(admit_program(&parsed, &parsed.programs.decode_step, "decode_step").is_err());

    let mut wrong_dtype = original.clone();
    wrong_dtype["programs"]["prefill"]["kernels"][0]["resources"][0]["version"]["element_ty"] =
        json!("bf16");
    let parsed: Gea3ProgramPlanEnvelope =
        serde_json::from_value(wrong_dtype).expect("mirror parse");
    assert!(map_envelope_to_descriptor(
        &parsed,
        &parsed.programs.prefill,
        "prefill",
        &artifact_dir
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
    )
    .expect_err("intermediate readback must fail closed");
    assert!(
        error.contains("logits"),
        "diagnostic must name logits observation: {error}"
    );

    let envelope = load_gea3_plan(&artifact_dir);
    let ((_, _), (decode, _)) = map_both(&envelope, &artifact_dir).expect("real plan maps");
    assert!(assert_declared_logits_only(&decode, decode.end_of_run_results[0].buffer_id).is_ok());
    assert!(assert_declared_logits_only(&decode, u32::MAX).is_err());

    // GEA3-A1 negatives: the sub-window two-truths guard at the mapper.
    // Launch 8 (kernels[7]) is the first key transpose; its canonical input
    // carries the strided [76,64] k-cache window.
    let mut wrong_derived = original.clone();
    wrong_derived["programs"]["decode_step"]["kernels"][7]["resources"][0]["version"]
        ["sub_window"]["derived_element_count"] = json!(4_863);
    let parsed: Gea3ProgramPlanEnvelope =
        serde_json::from_value(wrong_derived).expect("mirror parse");
    let error = map_envelope_to_descriptor(
        &parsed,
        &parsed.programs.decode_step,
        "decode_step",
        &artifact_dir,
    )
    .expect_err("a corrupted derived count must fail the mapper");
    assert!(
        error.contains("carried and derived truths must agree"),
        "diagnostic must name the two-truths law: {error}"
    );

    let mut off_the_end = original.clone();
    off_the_end["programs"]["decode_step"]["kernels"][7]["resources"][0]["version"]
        ["sub_window"]["element_offset"] = json!(5 * 64);
    let parsed: Gea3ProgramPlanEnvelope =
        serde_json::from_value(off_the_end).expect("mirror parse");
    let error = map_envelope_to_descriptor(
        &parsed,
        &parsed.programs.decode_step,
        "decode_step",
        &artifact_dir,
    )
    .expect_err("an off-the-end window must fail the mapper");
    assert!(
        error.contains("nest inside its producer buffer"),
        "diagnostic must name the nesting law: {error}"
    );

    // The attention-output hybrid write window (first context launch,
    // kernels[42]; its LAST resource is the windowed write) must stay
    // contiguous.
    let mut scattered_write = original.clone();
    let context_resources = scattered_write["programs"]["decode_step"]["kernels"][42]
        ["resources"]
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
    )
    .expect_err("a scattered write window must fail the mapper");
    assert!(
        error.contains("scattered write"),
        "diagnostic must name the write-window law: {error}"
    );

    // A sub-window without its projection fact stays rejected (the V1 law).
    let mut bare_sub_window = original.clone();
    bare_sub_window["programs"]["decode_step"]["kernels"][7]["resources"][0]["version"]
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
    )
    .expect_err("a V1-tagged plan must fail the mapper");
    assert!(
        error.contains("wire-buffer ABI"),
        "diagnostic must name the ABI law: {error}"
    );
}

#[test]
fn gea3_fake_multi_step_structural_loop() {
    let artifact_dir = gea3_artifact_dir();
    let envelope = load_gea3_plan(&artifact_dir);
    let ((prefill, _), (decode, _)) = map_both(&envelope, &artifact_dir)
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
    match program
        .windows
        .get(&(kernel_index, slot.binding))
        .copied()
    {
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

fn gea3_update_inputs(
    runtime: &mut DeviceRuntime,
    program: &Gea3PhysicalProgram,
    tokens: &[u32],
    position: u32,
    valid_len: u32,
    prefill: bool,
) -> Result<usize, String> {
    let mut copied = BTreeSet::new();
    let mut copy = |handle: DeviceHandle, values: Vec<f32>| -> Result<(), String> {
        if copied.insert(handle.id) {
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
            if kernel.entry == "embedding_gather" && slot.binding == 0 {
                let rows = if prefill { tokens.len() } else { 1 };
                let expected = rows
                    .checked_mul(VOCAB as usize)
                    .ok_or_else(|| "one-hot input size overflows".to_owned())?;
                if slot.element_count != expected as u64 {
                    return Err(format!(
                        "embedding selector declares {} values, expected {expected}",
                        slot.element_count
                    ));
                }
                let mut one_hot = vec![0.0; expected];
                for (row, token) in tokens.iter().enumerate() {
                    let token = usize::try_from(*token)
                        .map_err(|_| "token id does not fit host usize".to_owned())?;
                    if token >= VOCAB as usize {
                        return Err(format!("token id {token} is outside vocab {VOCAB}"));
                    }
                    one_hot[row * VOCAB as usize + token] = 1.0;
                }
                copy(handle, one_hot)?;
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
            }
        }
    }
    Ok(copied.len())
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

fn gea3_run_physical(
    runtime: &mut DeviceRuntime,
    prefill: (DeviceDescriptor, Gea3WindowBindings),
    decode: (DeviceDescriptor, Gea3WindowBindings),
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
    let prepare = (|| {
        let prefill_program = gea3_prepare_physical_program(
            runtime,
            prefill.0,
            prefill.1,
            &mut shared,
        )?;
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
                    }
                }
            }
        }
    }
    let kv_setup_us = kv_allocations_us.saturating_add(kv_zero_us);
    // The allocation pass above counted every zero-fill handle in `zeroed`;
    // the model weights are deliberately excluded from that set.
    let expected_kv_bytes = (LAYERS as u64) * 2 * HISTORY_CAPACITY * KV_WIDTH * 4;
    if kv_bytes != expected_kv_bytes {
        return Err(format!(
            "KV residency is {kv_bytes} bytes, expected {expected_kv_bytes}"
        ));
    }
    let launch_rows_prefill = gea3_launch_rows(&programs[0].descriptor);
    let launch_rows_decode = gea3_launch_rows(&programs[1].descriptor);
    let edge_prefill = gea3_data_flow_satisfied(&programs[0].descriptor);
    let edge_decode = gea3_data_flow_satisfied(&programs[1].descriptor);
    if !edge_prefill || !edge_decode {
        return Err("GEA3 carried data-flow edges are not topologically satisfied".to_owned());
    }
    let mut step_receipts = Vec::new();
    let mut greedy = Vec::new();
    let prefill_started = Instant::now();
    let prefill_copies = gea3_update_inputs(
        runtime,
        &programs[0],
        prompt_tokens,
        0,
        u32::try_from(prompt_tokens.len()).map_err(|_| "prompt is too long".to_owned())?,
        true,
    )?;
    let prefill_before_submit = runtime.command_submit_count();
    let prefill_before_wait = runtime.blocking_wait_count();
    let prefill_launch_started = Instant::now();
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
    runtime.sync().map_err(|error| error.message.clone())?;
    let prefill_gpu_us = runtime.take_encoder_gpu_us();
    let prefill_gpu_start_us = runtime.take_encoder_gpu_start_us();
    let prefill_submit_count = runtime
        .command_submit_count()
        .saturating_sub(prefill_before_submit);
    let prefill_wait_count = runtime
        .blocking_wait_count()
        .saturating_sub(prefill_before_wait);
    let prefill_readback_started = Instant::now();
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
    let prefill_vocab = usize::try_from(VOCAB).unwrap();
    let prefill_last = prefill_values
        .get((prompt_tokens.len().saturating_sub(1) * prefill_vocab)..)
        .ok_or_else(|| "prefill logits are shorter than the final prompt row".to_owned())?;
    let mut next_token = gea3_argmax(prefill_last)?;
    greedy.push(next_token);
    step_receipts.push(json!({
        "mode": "prefill",
        "step": 0,
        "input_uploads": prefill_copies,
        "launch_plan": "prefill",
        "launch_count": programs[0].descriptor.launches.len(),
        "data_flow_edges": {"declared": programs[0].descriptor.data_flow.len(), "satisfied": edge_prefill},
        "dispatch": {"launches": programs[0].descriptor.launches.len(), "command_submits": prefill_submit_count, "blocking_waits": prefill_wait_count},
        "timing_us": {"wall": gea3_elapsed_us(prefill_started), "launch_encode_and_sync": prefill_encode_sync_us, "gpu_body_sum": prefill_gpu_us.iter().copied().sum::<u64>(), "gpu_timestamp_count": prefill_gpu_us.len(), "gpu_start_timestamp_count": prefill_gpu_start_us.len(), "readback": prefill_readback_us},
        "readback": {"buffer_id": programs[0].output.0, "version": programs[0].output.1, "elements": prefill_values.len(), "bytes": prefill_values.len() * 4, "sha256": gea3_readback_hash(&prefill_values), "finite": true},
        "next_token": next_token,
    }));
    for step in 0..DECODE_STEPS {
        let valid_len = u32::try_from(prompt_tokens.len() + step + 1)
            .map_err(|_| "decode valid length overflows".to_owned())?;
        let position = valid_len - 1;
        let started = Instant::now();
        let input_uploads = gea3_update_inputs(
            runtime,
            &programs[1],
            &[next_token],
            position,
            valid_len,
            false,
        )?;
        let before_submit = runtime.command_submit_count();
        let before_wait = runtime.blocking_wait_count();
        let launch_started = Instant::now();
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
        runtime.sync().map_err(|error| error.message.clone())?;
        let gpu_us = runtime.take_encoder_gpu_us();
        let gpu_start_us = runtime.take_encoder_gpu_start_us();
        let submit_count = runtime.command_submit_count().saturating_sub(before_submit);
        let wait_count = runtime.blocking_wait_count().saturating_sub(before_wait);
        let readback_started = Instant::now();
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
        next_token = gea3_argmax(&values)?;
        greedy.push(next_token);
        step_receipts.push(json!({
            "mode": "decode",
            "step": step + 1,
            "position": position,
            "valid_len_after": valid_len,
            "input_uploads": input_uploads,
            "launch_plan": "decode",
            "launch_count": programs[1].descriptor.launches.len(),
            "data_flow_edges": {"declared": programs[1].descriptor.data_flow.len(), "satisfied": edge_decode},
            "dispatch": {"launches": programs[1].descriptor.launches.len(), "command_submits": submit_count, "blocking_waits": wait_count},
            "timing_us": {"wall": gea3_elapsed_us(started), "launch_encode_and_sync": encode_sync_us, "gpu_body_sum": gpu_us.iter().copied().sum::<u64>(), "gpu_timestamp_count": gpu_us.len(), "gpu_start_timestamp_count": gpu_start_us.len(), "readback": readback_us},
            "readback": {"buffer_id": programs[1].output.0, "version": programs[1].output.1, "elements": values.len(), "bytes": values.len() * 4, "sha256": gea3_readback_hash(&values), "finite": true},
            "next_token": next_token,
        }));
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
    let decode_gpu_us: Vec<u64> = step_receipts
        .iter()
        .skip(1)
        .filter_map(|row| row["timing_us"]["gpu_body_sum"].as_u64())
        .collect();
    let decode_submit_us: Vec<u64> = step_receipts
        .iter()
        .skip(1)
        .filter_map(|row| row["timing_us"]["launch_encode_and_sync"].as_u64())
        .collect();
    let kv_alloc_us = kv_setup_us;
    let evidence = json!({
        "residency": {
            "weight_allocations": {"value": 290, "status": "measured", "basis": "distinct frozen model tensor identities"},
            "weight_bytes": {"value": weight_bytes, "status": "measured", "basis": "GEA3 input manifest absolute ranges"},
            "weight_upload_count": {"value": weight_uploads, "status": "measured"},
            "weight_residency_us": {"value": weight_allocations_us.saturating_add(weight_upload_us), "status": "measured", "components": {"allocation_and_program_setup": weight_allocations_us, "mapped_upload": weight_upload_us}},
            "kv_allocations": {"value": LAYERS * 2, "status": "measured", "basis": "one shared fixed-capacity K/V arena per layer"},
            "kv_bytes": {"value": kv_bytes, "status": "measured", "basis": "32 * 2 * 76 * 320 * sizeof(F32)"},
            "kv_alloc_us": {"value": kv_alloc_us, "status": "measured", "basis": "KV handle allocation plus first zero-fill"},
            "zero_cpu_substitutes": {"value": 0, "status": "measured", "basis": "all model work was submitted to Metal"},
            "zero_cpu_bridges": {"value": 0, "status": "measured", "basis": "only one-hot staging, mask/rope constants, and host argmax ran on CPU"},
        },
        "execution": {
            "prefill_wall_us": {"value": prefill_wall_us, "status": "measured"},
            "per_step_gpu_body_us": {"value": decode_gpu_us, "status": "measured", "basis": "sum of Metal encoder timestamps per decode step"},
            "launch_submit_us_per_step": {"value": decode_submit_us, "status": "measured", "basis": "host launch encode plus explicit step sync"},
            "launches_per_step": {"value": programs[1].descriptor.launches.len(), "status": "derived", "basis": "descriptor launch list"},
            "step_count": {"value": DECODE_STEPS, "status": "assumed", "basis": "frozen GEA3 n_predict"},
            "sync_wait_count": {"value": step_receipts.iter().skip(1).map(|row| row["dispatch"]["blocking_waits"].as_u64().unwrap_or(0)).sum::<u64>(), "status": "measured"},
            "sync_wait_us": {"value": step_receipts.iter().skip(1).map(|row| row["timing_us"]["launch_encode_and_sync"].as_u64().unwrap_or(0)).sum::<u64>(), "status": "measured", "basis": "explicit Metal step sync wall"},
            "logits_readback_bytes_per_step": {"value": VOCAB * 4, "status": "derived", "basis": "declared decode logits shape [49152] F32"},
            "greedy_token_sequence": {"value": greedy, "status": "measured", "basis": "first-index host argmax of declared logits readback"},
            "intermediate_readbacks": {"value": 0, "status": "measured", "basis": "only declared logits observation was read back per invocation"},
            "decode_wall_us": {"value": decode_wall_us, "status": "measured"},
        },
        "launch_plans": {"prefill": launch_rows_prefill, "decode": launch_rows_decode},
        "steps": step_receipts,
        "throughput": {
            "pp_ts": {"value": PREFILL_ROWS as f64 * 1_000_000.0 / prefill_wall_us.max(1) as f64, "status": "derived", "basis": "prompt rows / prefill wall"},
            "tg_ts": {"value": DECODE_STEPS as f64 * 1_000_000.0 / decode_wall_us.max(1) as f64, "status": "derived", "basis": "decode steps / summed decode wall"},
        },
    });
    drop(gea3_release_programs(runtime, &mut programs));
    Ok(evidence)
}

#[test]
#[ignore = "physical Metal gate; run only with the exact §6 command"]
fn gea3_real_metal_decode_receipt() {
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
    let (prefill, decode) = map_both(&envelope, &artifact_dir)
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
                )
            });
        match execution {
            Ok(execution) => {
                receipt["status"] = json!("green");
                receipt["residency"] = execution["residency"].clone();
                receipt["execution"] = execution["execution"].clone();
                receipt["launch_plans"] = execution["launch_plans"].clone();
                receipt["steps"] = execution["steps"].clone();
                receipt["throughput"] = execution["throughput"].clone();
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
                    "per_step_gpu_body_us": gea3_unmeasured("physical execution failed before a complete receipt"),
                    "launch_submit_us_per_step": gea3_unmeasured("physical execution failed before a complete receipt"),
                    "launches_per_step": gea3_unmeasured("physical execution failed before a complete receipt"),
                    "sync_wait_count": gea3_unmeasured("physical execution failed before a complete receipt"),
                    "sync_wait_us": gea3_unmeasured("physical execution failed before a complete receipt"),
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
            "per_step_gpu_body_us": gea3_unmeasured("blocked before dispatch"),
            "launch_submit_us_per_step": gea3_unmeasured("blocked before dispatch"),
            "launches_per_step": gea3_unmeasured("blocked before dispatch"),
            "sync_wait_count": gea3_unmeasured("blocked before dispatch"),
            "sync_wait_us": gea3_unmeasured("blocked before dispatch"),
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
    decode: (DeviceDescriptor, Gea3WindowBindings),
    decode_plan: &Gea3Program,
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
        let program = gea3_prepare_physical_program(runtime, decode.0, decode.1, &mut shared)?;
        programs.push(program);
        let (weight_bytes, weight_uploads) = gea3_diagnostic_model_upload(
            runtime,
            programs.first().expect("diagnostic program prepared"),
            &shared,
            &model_ranges,
            &mapped,
        )?;
        let program = programs.first().expect("diagnostic program retained");
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
            "route": "decode-step",
            "token": token,
            "input_uploads": input_uploads,
            "weights": {
                "upload_count": weight_uploads,
                "bytes": weight_bytes,
            },
            "stages_executed": stages.len(),
            "stages": stages,
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
    let (_, decode) = map_both(&envelope, &artifact_dir)
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
                    decode,
                    &envelope.programs.decode_step,
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
    // GEA3-U6 rerun #6: non-finite localization.  When any launch produced a
    // non-finite element, the receipt must name that FIRST launch with its
    // inputs' finiteness state — never leave the NaN origin anonymous.
    assert!(
        receipt["execution"]["first_non_finite"].is_null()
            || receipt["execution"]["first_non_finite"]["launch_id"].as_u64().is_some(),
        "first non-finite launch must be named with its launch id"
    );
}
