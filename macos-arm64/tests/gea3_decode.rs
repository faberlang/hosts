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

use faber_host_macos_arm64::device_descriptor::{
    DescriptorBuffer, DescriptorBufferVersion, DescriptorDataFlow, DescriptorEndOfRunResult,
    DescriptorKernel, DescriptorLaunch, DescriptorResult, DeviceBufferInitialization,
    DeviceBufferLifetime, DeviceBufferRole, DeviceDataType, DeviceDescriptor,
    DeviceProgramLifetime,
};
use faber_host_macos_arm64::{FakeMetalDriver, MetalHostSession};
use host_coordinator::DeviceBackend;
use serde::Deserialize;
use serde_json::{json, Value};

const PLAN_SCHEMA: &str = "gea3-program-plan-v1";
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
    #[serde(alias = "ReadWrite")]
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

fn map_envelope_to_descriptor(
    envelope: &Gea3ProgramPlanEnvelope,
    program: &Gea3Program,
    program_name: &str,
    artifact_dir: &Path,
) -> Result<DeviceDescriptor, String> {
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
            if resource.version.element_count != *full_count {
                return Err(format!(
                    "resource `{}` binds a sub-window without a projection fact",
                    resource.buffer.name
                ));
            }
            if matches!(
                resource.access,
                Gea3ResourceAccess::Write | Gea3ResourceAccess::ReadWrite
            ) && resource.version.element_count != *full_count
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

    Ok(DeviceDescriptor {
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
    })
}

fn admit_envelope(envelope: &Gea3ProgramPlanEnvelope) -> Result<(), String> {
    if envelope.schema != PLAN_SCHEMA {
        return Err(format!("unexpected envelope schema `{}`", envelope.schema));
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
        "embedding_gather"
        | "decode_swiglu"
        | "decode_residual_add"
        | "prefill_swiglu"
        | "prefill_residual_add" => matches!(plan, Gea3Plan::Elementwise),
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
) -> Result<(DeviceDescriptor, DeviceDescriptor), String> {
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
        .validate()
        .map_err(|error| format!("prefill descriptor rejected: {}", error.message))?;
    decode
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
    let (prefill, decode) = map_both(&envelope, &artifact_dir)
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
    conflicting_shape["programs"]["decode_step"]["kernels"][1]["resources"][1]["version"]
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

    let mut intermediate_readback = original;
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
    let (_, decode) = map_both(&envelope, &artifact_dir).expect("real plan maps");
    assert!(assert_declared_logits_only(&decode, decode.end_of_run_results[0].buffer_id).is_ok());
    assert!(assert_declared_logits_only(&decode, u32::MAX).is_err());
}

#[test]
fn gea3_fake_multi_step_structural_loop() {
    let artifact_dir = gea3_artifact_dir();
    let envelope = load_gea3_plan(&artifact_dir);
    let (prefill, decode) = map_both(&envelope, &artifact_dir)
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
