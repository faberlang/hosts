//! GEA2-U5b: host edge consumption — mirror-parse the exported program plan,
//! map it onto a [`DeviceDescriptor`], and admit it fail-closed.
//!
//! The bundle's `gea2-program-plan.json` member (envelope schema
//! `gea2-program-plan-v1`) carries the radix `WireDeviceProgram` in its native
//! serde JSON form, instance-expanded (64 kernels/launches, 78 dependency
//! edges, roots `[1]`). Hosts owns the consumption half of that ABI: these
//! mirror structs parse the envelope (unknown/missing fields fail closed),
//! [`map_envelope_to_descriptor`] resolves the wire's per-slot bound shapes
//! onto the host descriptor's version-keyed shape table — failing closed on
//! any slot that binds fewer elements than its buffer's full shape — while
//! carrying role/lifetime/initialization verbatim, and
//! [`DeviceDescriptor::validate`] runs before any launch. Neither side infers
//! the other's facts (GEA2 seam-lowering transport decision).
#![allow(dead_code)] // the mirror fields are the fail-closed decode contract; serde reads the full wire shape while the mapper consumes the mapped subset

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use faber_host_macos_arm64::composite_host::{CompositeHost, DeviceByteBuffer};
use faber_host_macos_arm64::device_descriptor::{
    DescriptorBuffer, DescriptorBufferVersion, DescriptorDataFlow, DescriptorEndOfRunResult,
    DescriptorKernel, DescriptorLaunch, DescriptorResult, DeviceBufferInitialization,
    DeviceBufferLifetime, DeviceBufferRole, DeviceDataType, DeviceDescriptor,
    DeviceProgramLifetime, E_DEVICE_DESCRIPTOR,
};
use faber_host_macos_arm64::device_host::DeviceRuntime;
use faber_host_macos_arm64::{enumerate_metal_physical_devices, FakeMetalDriver, MetalHostSession};
use host_coordinator::DeviceBackend;
use serde::Deserialize;
use serde_json::{json, Value};

const PLAN_ENVELOPE_SCHEMA: &str = "gea2-program-plan-v1";
const PLAN_MEMBER: &str = "gea2-program-plan.json";
const MODULE_IMAGE_RULE: &str =
    "module_image is the concatenation of module_members in listed order";

/// The 15-entry block kernel table and its instance counts (GEA2 §5 / U5a
/// admission facts, mirrored here for consumption admission; the gathered
/// o-projection is its own entry since radix `6a0e3780a`, and the windowed
/// v-projection since GEA2-U5g).
const GEA2_ENTRY_TABLE: [(&str, usize); 15] = [
    ("rmsnorm", 2),
    ("gemm_qo", 1),
    ("gemm_qo_gathered", 1),
    ("gemm_kv", 1),
    ("gemm_kv_windows", 1),
    ("gemm_gate_up", 2),
    ("gemm_down", 1),
    ("rope_q", 1),
    ("rope_k", 1),
    ("transpose", 5),
    ("score_gemm", 15),
    ("causal_softmax", 15),
    ("context_gemm", 15),
    ("swiglu", 1),
    ("residual_add", 2),
];

// ---------------------------------------------------------------------------
// Hosts' own serde mirror of the gea2-program-plan-v1 envelope. The mirror is
// the consumption half of the transport ABI: unknown or missing fields fail
// the decode (the GEA1 ArtifactDescriptor/BundleManifest mirror precedent,
// strengthened with deny_unknown_fields).
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2ProgramPlanEnvelope {
    schema: String,
    program: Gea2Program,
    module_members: Vec<String>,
    module_image_rule: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2Program {
    kernels: Vec<Gea2KernelUnit>,
    launches: Vec<Gea2LaunchUnit>,
    lifetime: Gea2ProgramLifetime,
    results: Vec<Gea2ResultBuffer>,
    semantic_values: Vec<Gea2SemanticValue>,
    roots: Vec<u32>,
    dependencies: Vec<Gea2DependencyEdge>,
    relations: Vec<Gea2CompanionRelation>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2KernelUnit {
    function: u32,
    entry: String,
    plan: Gea2Plan,
    resources: Vec<Gea2DeviceResource>,
    launch: Gea2KernelLaunchPlan,
}

/// The wire's collection-kernel recipe. Only the GEA2 recipes are admitted;
/// an unknown variant is an ABI change and fails the mirror closed.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
enum Gea2Plan {
    Elementwise,
    TiledMatMul(Gea2MatMulPlan),
    Transpose(Gea2TransposePlan),
    RmsNormalization(Gea2RmsNormalizationPlan),
    Rope(Gea2RopePlan),
    CausalMaskedSoftmax(Gea2CausalMaskedSoftmaxPlan),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2MatMulPlan {
    m: u64,
    k: u64,
    n: u64,
    left_layout: Gea2MatMulLayout,
    right_layout: Gea2MatMulLayout,
    right_operand_layout: Gea2RightOperandLayout,
    tile: u32,
    workgroup_x: u32,
    workgroup_y: u32,
    shared_memory: Gea2MatMulSharedMemory,
    barriers: Vec<Gea2BarrierPoint>,
    oob_padding: Gea2OobPaddingPolicy,
}

/// The tiled matmul right operand's declared physical memory layout
/// (GEA2-U5i): GGUF `[out][in]` out-major weights vs the natural row-major
/// `[k][n]` tensor the transpose entry produces.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Gea2RightOperandLayout {
    OutIn,
    RowMajor,
}

#[derive(Debug, Clone, Copy, Deserialize)]
enum Gea2MatMulLayout {
    F32,
    Bf16,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2MatMulSharedMemory {
    shared_a: Gea2SharedMemoryLayout,
    shared_b: Gea2SharedMemoryLayout,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2SharedMemoryLayout {
    element_byte_width: u32,
    slot_count: u32,
    buffer_name: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2BarrierPoint {
    after: Gea2BarrierPhase,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Gea2BarrierPhase {
    SharedMemoryLoad,
    ReductionStep,
    InnerProductStep,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Gea2OobPaddingPolicy {
    ZeroFill,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2TransposePlan {
    m: u64,
    n: u64,
    workgroup_x: u32,
    dispatch_x: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2RmsNormalizationPlan {
    axis: u64,
    epsilon_bits: u32,
    width: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2RopePlan {
    pos: u64,
    dim: u64,
    width: u64,
    per_row: bool,
    rows: u64,
    rotate_half: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2CausalMaskedSoftmaxPlan {
    rows: u64,
    cols: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2DeviceResource {
    buffer: Gea2BufferIdentity,
    version: Gea2BufferVersion,
    binding: Gea2Binding,
    access: Gea2ResourceAccess,
    generation: u32,
    initialization: Gea2Initialization,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2BufferIdentity {
    id: u32,
    name: String,
    role: Gea2BufferRole,
    storage: Gea2StorageLayout,
    lifetime: Gea2BufferLifetime,
    semantic_value: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2BufferVersion {
    version: u32,
    element_ty: String,
    element_count: u64,
    reduced_projection: Option<Gea2ReducedProjection>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2ReducedProjection {
    axis_extent: u64,
    inner_stride: u64,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2Binding {
    group: u32,
    binding: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2KernelLaunchPlan {
    workgroup: Gea2WorkgroupSize,
    dispatch_size: Gea2DispatchSize,
    workgroup_count: Gea2WorkgroupCount,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2WorkgroupSize {
    x: u32,
    y: u32,
    z: u32,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2DispatchSize {
    x: u64,
    y: u64,
    z: u64,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2WorkgroupCount {
    x: u64,
    y: u64,
    z: u64,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2LaunchUnit {
    id: u32,
    kernel_index: u32,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2DependencyEdge {
    producer: u32,
    consumer: u32,
    buffer: u32,
    version: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2ResultBuffer {
    buffer: Gea2BufferIdentity,
    version: Gea2BufferVersion,
    role: Gea2BufferRole,
    produced_by: u32,
    observation: Gea2ObservationFact,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2ObservationFact {
    at_launch: u32,
    cadence: Gea2ObservationCadence,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2SemanticValue {
    id: u32,
    name: String,
    origin: Gea2SemanticValueOrigin,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
enum Gea2SemanticValueOrigin {
    MirLocal { function: u32, local: u32 },
    HostInput,
    Synthetic { label: String },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2CompanionRelation {
    primal: u32,
    companion: u32,
    derivative: Gea2CompanionDerivativeKind,
    device_resident: bool,
    selected_inputs: Vec<Gea2CompanionSelectedInput>,
    selected_outputs: Vec<Gea2CompanionSelectedOutput>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Gea2CompanionDerivativeKind {
    ReverseModeVjp,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2CompanionSelectedInput {
    param: u32,
    position: u32,
    ty: u32,
    gradient_slot: u32,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gea2CompanionSelectedOutput {
    position: u32,
    ty: u32,
    upstream_gradient_ty: u32,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Gea2BufferRole {
    Input,
    Output,
    InOut,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Gea2StorageLayout {
    HostOwned,
    DeviceHandle,
    Sparse,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Gea2BufferLifetime {
    PerProgram,
    PerStep,
    ObservationPoint,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Gea2ResourceAccess {
    Read,
    Write,
    ReadWrite,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Gea2Initialization {
    ZeroFill,
    HostProvided,
    KernelInitialized,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Gea2ObservationCadence {
    PerStep,
    EndOfRun,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Gea2ProgramLifetime {
    SingleRun,
    RepeatingStep(u32),
}

// ---------------------------------------------------------------------------
// Envelope → DeviceDescriptor mapping (the ABI's consumption half).
// ---------------------------------------------------------------------------

/// Map the mirror-parsed envelope onto a [`DeviceDescriptor`], preserving the
/// instance expansion. Every fact the host cannot carry is rejected rather
/// than dropped; every carried identity fact (role/lifetime/initialization/
/// semantic value) survives verbatim.
fn map_envelope_to_descriptor(
    envelope: &Gea2ProgramPlanEnvelope,
    artifact_dir: &Path,
) -> Result<DeviceDescriptor, String> {
    if envelope.schema != PLAN_ENVELOPE_SCHEMA {
        return Err(format!(
            "unexpected program-plan envelope schema `{}`; expected `{PLAN_ENVELOPE_SCHEMA}`",
            envelope.schema
        ));
    }
    if envelope.module_image_rule != MODULE_IMAGE_RULE {
        return Err(format!(
            "unexpected module-image assembly rule `{}`",
            envelope.module_image_rule
        ));
    }
    if envelope.module_members.is_empty() {
        return Err("program plan envelope declares no module members".to_owned());
    }

    // module_image is the concatenation of the listed members, in order. A
    // missing or empty member fails closed.
    let mut module_image = Vec::new();
    for member in &envelope.module_members {
        if !member.ends_with(".metal") {
            return Err(format!(
                "module member `{member}` is not a Metal source member"
            ));
        }
        let path = artifact_dir.join(member);
        let bytes = fs::read(&path).map_err(|error| {
            format!(
                "module member `{}` is missing from the exported bundle: {error}",
                path.display()
            )
        })?;
        if bytes.is_empty() {
            return Err(format!("module member `{member}` is empty"));
        }
        module_image.extend_from_slice(&bytes);
    }

    let program = &envelope.program;

    // Instance-expansion facts are the plan of record (U5a admission).
    if program.kernels.len() != 64 || program.launches.len() != 64 {
        return Err(format!(
            "GEA2 plan must carry 64 instance-expanded kernels and launches ({} kernels, {} launches)",
            program.kernels.len(),
            program.launches.len()
        ));
    }
    if program.dependencies.len() != 78 {
        return Err(format!(
            "missing dependency edge: {} present, 78 expected",
            program.dependencies.len()
        ));
    }
    if program.roots != vec![1] {
        return Err(format!(
            "GEA2 roots must be exactly [1], got {:?}",
            program.roots
        ));
    }
    for (index, launch) in program.launches.iter().enumerate() {
        let expected_id = u32::try_from(index + 1).expect("launch id fits u32");
        let expected_kernel = u32::try_from(index).expect("kernel index fits u32");
        if launch.id != expected_id || launch.kernel_index != expected_kernel {
            return Err(format!("launch {expected_id} is not instance-expanded"));
        }
    }
    check_entry_table(&program.kernels)?;

    // The wire carries per-slot bound shapes. The host descriptor's keyed
    // metadata carries ONE shape per (buffer_id, version), so the mapper
    // resolves each key to the buffer's full shape (the largest bound
    // count) — and a slot that binds FEWER elements than that full shape
    // fails closed (GEA2-U5g): the descriptor ABI carries no projection
    // fact, so an undeclared sub-window read would silently widen to a
    // mis-strided prefix of the packed buffer. A window a launch means to
    // consume must be declared as its own full-shape buffer (the per-head
    // q/k/v windows), never a slice of a packed one.
    let mut shapes: BTreeMap<(u32, u32), (DeviceDataType, u64)> = BTreeMap::new();
    for kernel in &program.kernels {
        for resource in &kernel.resources {
            let key = (resource.buffer.id, resource.version.version);
            let element_ty = DeviceDataType::from_spelling(&resource.version.element_ty)
                .ok_or_else(|| {
                    format!(
                        "buffer `{}` (id {}) carries unsupported element type `{}`",
                        resource.buffer.name, resource.buffer.id, resource.version.element_ty
                    )
                })?;
            let entry = shapes.entry(key).or_insert((element_ty, 0));
            if entry.0 != element_ty {
                return Err(format!(
                    "buffer {} version {} carries conflicting element types {} and {}",
                    key.0,
                    key.1,
                    entry.0.spelling(),
                    element_ty.spelling()
                ));
            }
            entry.1 = entry.1.max(resource.version.element_count);
        }
    }

    let mut kernels = Vec::with_capacity(program.kernels.len());
    for kernel in &program.kernels {
        check_entry_recipe(&kernel.entry, &kernel.plan)?;
        if kernel.resources.is_empty() {
            return Err(format!("GEA2 kernel `{}` binds no resources", kernel.entry));
        }
        let grid = [
            to_grid_axis(kernel.launch.workgroup_count.x, &kernel.entry)?,
            to_grid_axis(kernel.launch.workgroup_count.y, &kernel.entry)?,
            to_grid_axis(kernel.launch.workgroup_count.z, &kernel.entry)?,
        ];
        let block = [
            to_block_axis(kernel.launch.workgroup.x, &kernel.entry)?,
            to_block_axis(kernel.launch.workgroup.y, &kernel.entry)?,
            to_block_axis(kernel.launch.workgroup.z, &kernel.entry)?,
        ];
        if grid.contains(&0) || block.contains(&0) {
            return Err(format!(
                "GEA2 kernel `{}` has a zero grid or block axis",
                kernel.entry
            ));
        }
        let mut buffers = Vec::with_capacity(kernel.resources.len());
        let mut seen_bindings: Vec<u32> = Vec::with_capacity(kernel.resources.len());
        for resource in &kernel.resources {
            if resource.version.reduced_projection.is_some() {
                return Err(format!(
                    "buffer `{}` (id {}) carries a reduced-resource projection; the host device descriptor carries no projection fact, so the mapping fails closed",
                    resource.buffer.name, resource.buffer.id
                ));
            }
            if resource.binding.group != 0 {
                return Err(format!(
                    "buffer `{}` (id {}) binds group {}; the host ABI binds group 0 only",
                    resource.buffer.name, resource.buffer.id, resource.binding.group
                ));
            }
            if seen_bindings.contains(&resource.binding.binding) {
                return Err(format!(
                    "GEA2 kernel `{}` binds index {} more than once",
                    kernel.entry, resource.binding.binding
                ));
            }
            seen_bindings.push(resource.binding.binding);

            let key = (resource.buffer.id, resource.version.version);
            let (element_ty, full_count) = shapes[&key];
            let bound_count = resource.version.element_count;
            if bound_count > full_count {
                return Err(format!(
                    "buffer `{}` (id {}) version {} binds {} elements, more than the buffer's declared {}; a slot cannot exceed its buffer",
                    resource.buffer.name,
                    resource.buffer.id,
                    resource.version.version,
                    bound_count,
                    full_count
                ));
            }
            if matches!(
                resource.access,
                Gea2ResourceAccess::Write | Gea2ResourceAccess::ReadWrite
            ) && bound_count != full_count
            {
                return Err(format!(
                    "buffer `{}` (id {}) version {} is written as {} elements, not the buffer's declared {}; a write must define the full shape",
                    resource.buffer.name,
                    resource.buffer.id,
                    resource.version.version,
                    bound_count,
                    full_count
                ));
            }
            if bound_count < full_count {
                return Err(format!(
                    "buffer `{}` (id {}) version {} is read as {} elements of the buffer's declared {}; an undeclared sub-window read is not a carried ABI fact — declare the window as its own full-shape buffer",
                    resource.buffer.name,
                    resource.buffer.id,
                    resource.version.version,
                    bound_count,
                    full_count
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
                element_ty,
                element_count: full_count,
                version: resource.version.version,
            });
        }
        kernels.push(DescriptorKernel {
            entry: kernel.entry.clone(),
            buffers,
            grid,
            block,
        });
    }

    let (results, end_of_run_results) = map_results(&program.results, &shapes)?;

    let buffer_versions = shapes
        .into_iter()
        .map(
            |((buffer_id, version), (element_ty, element_count))| DescriptorBufferVersion {
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

    let program_lifetime = match program.lifetime {
        Gea2ProgramLifetime::SingleRun => DeviceProgramLifetime::SingleRun,
        Gea2ProgramLifetime::RepeatingStep(count) => {
            if count == 0 {
                return Err(
                    "GEA2 plan declares a repeating-step lifetime with a zero step count"
                        .to_owned(),
                );
            }
            DeviceProgramLifetime::RepeatingStep
        }
    };

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
        program_lifetime,
        data_flow,
        roots: program.roots.clone(),
        results,
        end_of_run_results,
    })
}

/// Project the declared observation rows, splitting on the carried cadence
/// (per-step observations vs end-of-run). A result whose declared version
/// shape disagrees with the buffer's keyed shape fails closed.
fn map_results(
    results: &[Gea2ResultBuffer],
    shapes: &BTreeMap<(u32, u32), (DeviceDataType, u64)>,
) -> Result<(Vec<DescriptorResult>, Vec<DescriptorEndOfRunResult>), String> {
    let mut per_step = Vec::with_capacity(results.len());
    let mut end_of_run = Vec::with_capacity(results.len());
    for result in results {
        let key = (result.buffer.id, result.version.version);
        let Some((_, full_count)) = shapes.get(&key) else {
            return Err(format!(
                "result names buffer {} version {} which has no keyed metadata",
                result.buffer.id, result.version.version
            ));
        };
        if *full_count != result.version.element_count {
            return Err(format!(
                "result buffer `{}` (id {}) declares {} elements but the buffer's keyed version shape is {}",
                result.buffer.name, result.buffer.id, result.version.element_count, full_count
            ));
        }
        match result.observation.cadence {
            Gea2ObservationCadence::PerStep => per_step.push(DescriptorResult {
                buffer_id: result.buffer.id,
                version: result.version.version,
                produced_by: result.produced_by,
                at_launch: result.observation.at_launch,
            }),
            Gea2ObservationCadence::EndOfRun => {
                end_of_run.push(DescriptorEndOfRunResult {
                    buffer_id: result.buffer.id,
                    version: result.version.version,
                });
            }
        }
    }
    Ok((per_step, end_of_run))
}

fn check_entry_table(kernels: &[Gea2KernelUnit]) -> Result<(), String> {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for kernel in kernels {
        *counts.entry(kernel.entry.as_str()).or_insert(0) += 1;
    }
    for (entry, expected) in GEA2_ENTRY_TABLE {
        let actual = counts.get(entry).copied().unwrap_or(0);
        if actual != expected {
            return Err(format!(
                "GEA2 plan instance count drifted for `{entry}`: {actual} present, {expected} expected"
            ));
        }
    }
    if counts.len() != GEA2_ENTRY_TABLE.len() {
        return Err(format!(
            "GEA2 plan carries {} distinct entries; the {}-entry block table expects {}",
            counts.len(),
            GEA2_ENTRY_TABLE.len(),
            GEA2_ENTRY_TABLE.len()
        ));
    }
    Ok(())
}

/// The entry's recipe identity must match its plan variant (a carried fact,
/// never inferred from the entry name alone).
fn check_entry_recipe(entry: &str, plan: &Gea2Plan) -> Result<(), String> {
    let admitted = match entry {
        "rmsnorm" => matches!(plan, Gea2Plan::RmsNormalization(_)),
        "gemm_qo" | "gemm_qo_gathered" | "gemm_kv" | "gemm_kv_windows" | "gemm_gate_up"
        | "gemm_down" | "score_gemm" | "context_gemm" => {
            matches!(plan, Gea2Plan::TiledMatMul(_))
        }
        "rope_q" | "rope_k" => matches!(plan, Gea2Plan::Rope(_)),
        "transpose" => matches!(plan, Gea2Plan::Transpose(_)),
        "causal_softmax" => matches!(plan, Gea2Plan::CausalMaskedSoftmax(_)),
        "swiglu" | "residual_add" => matches!(plan, Gea2Plan::Elementwise),
        other => {
            return Err(format!("GEA2 plan names unknown kernel entry `{other}`"));
        }
    };
    if !admitted {
        return Err(format!(
            "GEA2 kernel `{entry}` carries a recipe that does not match its entry identity"
        ));
    }
    Ok(())
}

fn to_grid_axis(axis: u64, entry: &str) -> Result<u32, String> {
    u32::try_from(axis).map_err(|_| {
        format!("GEA2 kernel `{entry}` has a workgroup count that overflows the host dispatch axis")
    })
}

fn to_block_axis(axis: u32, entry: &str) -> Result<u32, String> {
    if axis == 0 {
        return Err(format!("GEA2 kernel `{entry}` has a zero workgroup axis"));
    }
    Ok(axis)
}

fn map_role(role: Gea2BufferRole) -> DeviceBufferRole {
    match role {
        Gea2BufferRole::Input => DeviceBufferRole::Input,
        Gea2BufferRole::Output => DeviceBufferRole::Output,
        Gea2BufferRole::InOut => DeviceBufferRole::InOut,
    }
}

fn map_lifetime(lifetime: Gea2BufferLifetime) -> DeviceBufferLifetime {
    match lifetime {
        Gea2BufferLifetime::PerProgram => DeviceBufferLifetime::PerProgram,
        Gea2BufferLifetime::PerStep => DeviceBufferLifetime::PerStep,
        Gea2BufferLifetime::ObservationPoint => DeviceBufferLifetime::ObservationPoint,
    }
}

fn map_initialization(initialization: Gea2Initialization) -> DeviceBufferInitialization {
    match initialization {
        Gea2Initialization::ZeroFill => DeviceBufferInitialization::ZeroFill,
        Gea2Initialization::HostProvided => DeviceBufferInitialization::HostProvided,
        Gea2Initialization::KernelInitialized => DeviceBufferInitialization::KernelInitialized,
    }
}

// ---------------------------------------------------------------------------
// Bundle loading (the real exported bundle via GEA2_ARTIFACT_DIR).
// ---------------------------------------------------------------------------

fn gea2_artifact_dir() -> PathBuf {
    let root = std::env::var_os("GEA2_ARTIFACT_DIR")
        .map(PathBuf::from)
        .expect("GEA2_ARTIFACT_DIR must identify the exported GEA2 bundle");
    assert!(
        root.is_dir(),
        "missing GEA2 artifact directory {}",
        root.display()
    );
    root
}

fn load_gea2_plan(artifact_dir: &Path) -> Gea2ProgramPlanEnvelope {
    let path = artifact_dir.join(PLAN_MEMBER);
    let bytes = fs::read(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|error| panic!("mirror-parse {}: {error}", path.display()))
}

// ---------------------------------------------------------------------------
// The admission filter (U5b done-when).
// ---------------------------------------------------------------------------

#[test]
fn gea2_descriptor_admission() {
    let artifact_dir = gea2_artifact_dir();
    let envelope = load_gea2_plan(&artifact_dir);
    let descriptor = map_envelope_to_descriptor(&envelope, &artifact_dir)
        .unwrap_or_else(|error| panic!("GEA2 plan → DeviceDescriptor mapping failed: {error}"));
    descriptor.validate().unwrap_or_else(|error| {
        panic!(
            "mapped GEA2 descriptor must validate: {} ({})",
            error.message, error.code
        )
    });

    // Instance-expansion facts survive the mapping exactly.
    assert_eq!(descriptor.backend, DeviceBackend::Metal);
    assert_eq!(descriptor.kernels.len(), 64);
    assert_eq!(descriptor.launches.len(), 64);
    assert_eq!(descriptor.data_flow.len(), 78);
    assert_eq!(descriptor.roots, vec![1]);
    assert_eq!(
        descriptor.buffer_versions.len(),
        101,
        "101 distinct buffer version keys"
    );
    assert_eq!(descriptor.results.len(), 1);
    assert_eq!(descriptor.end_of_run_results.len(), 0);
    assert!(!descriptor.module_image.is_empty());
    assert_eq!(
        descriptor
            .launches
            .iter()
            .map(|launch| launch.id)
            .collect::<Vec<_>>(),
        (1..=64).collect::<Vec<_>>()
    );

    // role/lifetime/initialization survive instance expansion exactly:
    // weights are HostProvided/PerProgram, intermediates are
    // KernelInitialized/PerStep, the output is an ObservationPoint. The
    // class counts are distinct buffer identities (each weight is read by
    // several launches, so slot counts would over-count).
    let mut weight_ids = Vec::new();
    let mut intermediate_ids = Vec::new();
    let mut output = None;
    for kernel in &descriptor.kernels {
        for slot in &kernel.buffers {
            match slot.lifetime {
                DeviceBufferLifetime::PerProgram => {
                    if !weight_ids.contains(&slot.buffer_id) {
                        weight_ids.push(slot.buffer_id);
                    }
                }
                DeviceBufferLifetime::PerStep => {
                    if !intermediate_ids.contains(&slot.buffer_id) {
                        intermediate_ids.push(slot.buffer_id);
                    }
                }
                DeviceBufferLifetime::ObservationPoint => output = Some(slot),
            }
        }
    }
    assert_eq!(
        weight_ids.len(),
        12,
        "twelve block tensors are per-program inputs"
    );
    for weight in weight_ids {
        let slots = descriptor
            .kernels
            .iter()
            .flat_map(|kernel| kernel.buffers.iter())
            .filter(|slot| slot.buffer_id == weight);
        for slot in slots {
            assert_eq!(slot.role, DeviceBufferRole::Input);
            assert_eq!(
                slot.initialization,
                DeviceBufferInitialization::HostProvided
            );
            assert_eq!(slot.element_ty, DeviceDataType::F32);
        }
    }
    assert!(
        !intermediate_ids.is_empty(),
        "device intermediates must survive the instance expansion"
    );
    for intermediate in intermediate_ids {
        let slots = descriptor
            .kernels
            .iter()
            .flat_map(|kernel| kernel.buffers.iter())
            .filter(|slot| slot.buffer_id == intermediate);
        for slot in slots {
            assert_eq!(slot.role, DeviceBufferRole::InOut);
            assert_eq!(
                slot.initialization,
                DeviceBufferInitialization::KernelInitialized
            );
        }
    }
    let output = output.expect("the block output is an observation point");
    assert_eq!(output.role, DeviceBufferRole::Output);
    assert_eq!(output.element_count, 7680);
    assert_eq!(output.element_ty, DeviceDataType::F32);

    // Declared result: the F32 [8,960] block output at the final launch.
    let result = &descriptor.results[0];
    assert_eq!(result.buffer_id, output.buffer_id);
    assert_eq!(result.version, 1);
    assert_eq!(result.produced_by, 64);
    assert_eq!(result.at_launch, 64);

    // Per-instance attention windows (GEA2-U5e repair, radix 5f96ed340):
    // the plan declares a DISTINCT `q_head_<h>` buffer per score_gemm
    // instance and a DISTINCT `k_head_<g>` buffer per transpose instance —
    // produced by the rope launches as full-shape 512-element writes, never
    // a shared window of the packed rope outputs. The consumption side
    // mirrors the plan admission: the windows stay distinct through the
    // mapping, and the plan carries no shared query/key window.
    let mut score_query_windows = Vec::new();
    let mut transpose_key_windows = Vec::new();
    for kernel in &descriptor.kernels {
        match kernel.entry.as_str() {
            "score_gemm" => score_query_windows.push(kernel.buffers[0].buffer_id),
            "transpose" => transpose_key_windows.push(kernel.buffers[0].buffer_id),
            _ => {}
        }
    }
    assert_eq!(score_query_windows.len(), 15);
    assert_eq!(
        score_query_windows
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        15,
        "every score_gemm instance binds its own q_head window"
    );
    assert_eq!(transpose_key_windows.len(), 5);
    assert_eq!(
        transpose_key_windows
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        5,
        "every transpose instance binds its own k_head window"
    );
    let window_extents: std::collections::BTreeSet<(u32, u64)> = envelope
        .program
        .kernels
        .iter()
        .flat_map(|kernel| &kernel.resources)
        .filter(|resource| {
            resource.buffer.name.starts_with("q_head_")
                || resource.buffer.name.starts_with("k_head_")
                || resource.buffer.name.starts_with("v_head_")
        })
        .map(|resource| (resource.buffer.id, resource.version.element_count))
        .collect();
    assert_eq!(
        window_extents.len(),
        25,
        "15 q_head + 5 k_head + 5 v_head windows, each a distinct full-shape buffer"
    );
    assert!(
        window_extents.iter().all(|&(_, extent)| extent == 512),
        "every per-head window is the full [8,64] shape: {window_extents:?}"
    );

    // GEA2-U5g layout truth: every context_gemm consumes ITS OWN KV head's
    // value window (§2: head h → v_head_{h/3}), declared as a distinct
    // full-shape [8,64] buffer produced by the windowed v-projection launch
    // — never a sub-window read of the packed [8,320] `v` (the pre-U5g
    // defect bound the packed buffer at element_count 512 into all 15
    // instances, ignoring the GQA kv-head; the mapper now rejects that
    // shape outright). Pinned here so drift in either direction — a shared
    // window, a wrong kv-head mapping, or a sub-window read — fails this
    // gate closed rather than passing silently.
    let mut value_windows = Vec::new();
    for kernel in &descriptor.kernels {
        if kernel.entry == "context_gemm" {
            value_windows.push(kernel.buffers[1].buffer_name.clone());
        }
    }
    assert_eq!(value_windows.len(), 15);
    for (position, name) in value_windows.iter().enumerate() {
        assert_eq!(
            name,
            &format!("v_head_{}", position / 3),
            "context_gemm {position} must bind its GQA kv-head value window"
        );
    }
    assert_eq!(
        value_windows
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        5,
        "the 15 context_gemm instances resolve to the 5 distinct per-KV-head v windows"
    );
}

// ---------------------------------------------------------------------------
// Negative rows — mirror-parse fails closed (hosts' consumption half).
// ---------------------------------------------------------------------------

#[test]
fn gea2_mirror_parse_rejects_malformed_plans() {
    let artifact_dir = gea2_artifact_dir();
    let path = artifact_dir.join(PLAN_MEMBER);
    let bytes = fs::read(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let value: Value =
        serde_json::from_slice(&bytes).expect("the real exported plan is valid JSON");

    // An unknown envelope field fails the mirror closed.
    let mut unknown_field = value.clone();
    unknown_field["unexpected_field"] = json!(true);
    assert!(
        serde_json::from_value::<Gea2ProgramPlanEnvelope>(unknown_field).is_err(),
        "unknown envelope field must fail closed"
    );

    // A missing envelope field fails the mirror closed.
    let mut missing_field = value.clone();
    missing_field
        .as_object_mut()
        .expect("envelope object")
        .remove("module_image_rule");
    assert!(
        serde_json::from_value::<Gea2ProgramPlanEnvelope>(missing_field).is_err(),
        "missing envelope field must fail closed"
    );

    // An unknown program field fails the mirror closed.
    let mut unknown_program_field = value.clone();
    unknown_program_field["program"]["mystery"] = json!(1);
    assert!(
        serde_json::from_value::<Gea2ProgramPlanEnvelope>(unknown_program_field).is_err(),
        "unknown program field must fail closed"
    );

    // An unknown kernel field fails the mirror closed.
    let mut unknown_kernel_field = value.clone();
    unknown_kernel_field["program"]["kernels"][0]["mystery"] = json!(1);
    assert!(
        serde_json::from_value::<Gea2ProgramPlanEnvelope>(unknown_kernel_field).is_err(),
        "unknown kernel field must fail closed"
    );

    // An unknown plan variant (a recipe the host ABI does not admit) fails
    // the mirror closed.
    let mut unknown_plan_variant = value.clone();
    unknown_plan_variant["program"]["kernels"][0]["plan"] =
        json!({ "Gather": { "table_rows": 8, "table_cols": 960, "id_count": 8 } });
    assert!(
        serde_json::from_value::<Gea2ProgramPlanEnvelope>(unknown_plan_variant).is_err(),
        "unknown plan variant must fail closed"
    );

    // An unknown field inside a known plan variant fails the mirror closed.
    let mut unknown_plan_field = value.clone();
    unknown_plan_field["program"]["kernels"][0]["plan"]["RmsNormalization"]["mystery"] = json!(1);
    assert!(
        serde_json::from_value::<Gea2ProgramPlanEnvelope>(unknown_plan_field).is_err(),
        "unknown plan field must fail closed"
    );

    // A changed envelope schema tag is admitted by the mirror (a string) but
    // fails the envelope admission fail-closed.
    let mut wrong_schema = value.clone();
    wrong_schema["schema"] = json!("gea2-program-plan-v2");
    let parsed: Gea2ProgramPlanEnvelope =
        serde_json::from_value(wrong_schema).expect("schema tag is a mirror string");
    let error = map_envelope_to_descriptor(&parsed, &artifact_dir)
        .expect_err("wrong schema tag must fail closed");
    assert!(
        error.contains(PLAN_ENVELOPE_SCHEMA),
        "schema rejection must name the expected tag: {error}"
    );

    // A plan that is not instance-expanded fails the mapper closed.
    let mut truncated = value.clone();
    truncated["program"]["launches"]
        .as_array_mut()
        .expect("launches array")
        .truncate(63);
    let parsed: Gea2ProgramPlanEnvelope =
        serde_json::from_value(truncated).expect("truncated launches still mirror");
    let error = map_envelope_to_descriptor(&parsed, &artifact_dir)
        .expect_err("a 63-launch plan must fail closed");
    assert!(
        error.contains("64"),
        "instance expansion must fail closed: {error}"
    );

    // A missing dependency edge fails the mapper closed.
    let mut missing_edge = value.clone();
    missing_edge["program"]["dependencies"]
        .as_array_mut()
        .expect("dependencies array")
        .pop();
    let parsed: Gea2ProgramPlanEnvelope =
        serde_json::from_value(missing_edge).expect("77-edge plan still mirrors");
    let error = map_envelope_to_descriptor(&parsed, &artifact_dir)
        .expect_err("a 77-edge plan must fail closed");
    assert!(error.contains("78"), "edge count must fail closed: {error}");

    // GEA2-U5g rejection proof: an UNDECLARED sub-window read — a
    // context_gemm binding the packed 2560-element `v` at 512 elements
    // with no projection (the pre-U5g defect shape) — fails the mapping
    // closed instead of silently widening to a mis-strided prefix.
    let packed_v = value["program"]["kernels"]
        .as_array()
        .expect("kernels")
        .iter()
        .find(|kernel| kernel["entry"] == "gemm_kv_windows")
        .and_then(|kernel| kernel["resources"].as_array())
        .and_then(|resources| resources.get(2))
        .and_then(|resource| resource.get("buffer"))
        .expect("the windowed v-projection's packed v write")
        .clone();
    let context_index = value["program"]["kernels"]
        .as_array()
        .expect("kernels")
        .iter()
        .position(|kernel| kernel["entry"] == "context_gemm")
        .expect("a context_gemm kernel");
    let mut subwindow_read = value.clone();
    subwindow_read["program"]["kernels"][context_index]["resources"][1]["buffer"] = packed_v;
    // element_count stays 512 — smaller than the bound buffer's 2560.
    let parsed: Gea2ProgramPlanEnvelope =
        serde_json::from_value(subwindow_read).expect("the sub-window plan still mirrors");
    let error = map_envelope_to_descriptor(&parsed, &artifact_dir)
        .expect_err("an undeclared sub-window read must fail the mapping closed");
    assert!(
        error.contains("sub-window read"),
        "the rejection must name the sub-window rule: {error}"
    );

    // A module member absent from the bundle fails the assembly closed.
    let mut missing_member = value.clone();
    missing_member["module_members"][0] = json!("absent.metal");
    let parsed: Gea2ProgramPlanEnvelope =
        serde_json::from_value(missing_member).expect("member list still mirrors");
    let error = map_envelope_to_descriptor(&parsed, &artifact_dir)
        .expect_err("a missing module member must fail closed");
    assert!(
        error.contains("absent.metal"),
        "member rejection names the member: {error}"
    );
}

// ---------------------------------------------------------------------------
// Negative rows — DeviceDescriptor::validate fails closed on plan-shaped
// violations (single producer, topological order, root reachability).
// ---------------------------------------------------------------------------

#[test]
fn gea2_descriptor_validate_rejects_plan_shape_violations() {
    let artifact_dir = gea2_artifact_dir();
    let envelope = load_gea2_plan(&artifact_dir);
    let descriptor = map_envelope_to_descriptor(&envelope, &artifact_dir)
        .expect("the real GEA2 plan maps to a descriptor");
    descriptor
        .validate()
        .expect("the mapped descriptor validates");

    // Single producer: one value generation has exactly one producer. Give
    // ln1 (buffer 100, produced by launch 1) a second producer.
    let mut double_producer = descriptor.clone();
    double_producer.data_flow.push(DescriptorDataFlow {
        buffer_id: 100,
        version: 1,
        producer: 2,
        consumer: 5,
    });
    let error = double_producer
        .validate()
        .expect_err("a second producer for one generation must fail closed");
    assert_eq!(error.code, E_DEVICE_DESCRIPTOR);
    assert!(
        error.message.contains("exactly one producer"),
        "diagnostic names the single-producer rule: {}",
        error.message
    );

    // Topological order: a consumer must be scheduled after its producer.
    // Invert the final edge (down, 63 → 64) so the producer follows its
    // consumer.
    let mut inverted = descriptor.clone();
    let last = inverted.data_flow.len() - 1;
    let edge = &mut inverted.data_flow[last];
    edge.producer = 64;
    edge.consumer = 63;
    let error = inverted
        .validate()
        .expect_err("a non-topological edge must fail closed");
    assert_eq!(error.code, E_DEVICE_DESCRIPTOR);
    assert!(
        error.message.contains("not scheduled before"),
        "diagnostic names the topological rule: {}",
        error.message
    );

    // Root reachability: every launch is reachable from a declared root.
    // Remove the edge that anchors the final launch.
    let mut unreachable = descriptor.clone();
    unreachable.data_flow.pop();
    let error = unreachable
        .validate()
        .expect_err("an unreachable launch must fail closed");
    assert_eq!(error.code, E_DEVICE_DESCRIPTOR);
    assert!(
        error.message.contains("not reachable"),
        "diagnostic names the reachability rule: {}",
        error.message
    );

    // Undeclared root: roots must name real launches.
    let mut undeclared_root = descriptor.clone();
    undeclared_root.roots = vec![99];
    let error = undeclared_root
        .validate()
        .expect_err("an undeclared root must fail closed");
    assert_eq!(error.code, E_DEVICE_DESCRIPTOR);
    assert!(
        error.message.contains("unknown launch"),
        "diagnostic names the unknown root: {}",
        error.message
    );

    // Undeclared intermediate: a data-flow edge must name a buffer the plan
    // carries.
    let mut unknown_buffer = descriptor.clone();
    unknown_buffer.data_flow[0].buffer_id = u32::MAX;
    let error = unknown_buffer
        .validate()
        .expect_err("an undeclared intermediate must fail closed");
    assert_eq!(error.code, E_DEVICE_DESCRIPTOR);
    assert!(
        error.message.contains("unknown buffer"),
        "diagnostic names the unknown buffer: {}",
        error.message
    );
}

// ---------------------------------------------------------------------------
// The fake sequence ladder (U5c + U5d): the 64-launch block program executes
// on the FakeMetalDriver through the composite program session. The driver's
// declared-function table carries the 15-entry block ABI, so every launch
// takes the structural, encode-only GEA2 dispatch — no kernel-library body
// and no CPU oracle is ever consulted (the values are never evidence).
// ---------------------------------------------------------------------------

/// The fake-metal composite whose module declares every GEA2 block entry, so
/// each of the 64 launches takes the driver's declared-entry GEA2 dispatch.
fn gea2_fake_host() -> CompositeHost {
    let mut driver = FakeMetalDriver::default();
    for (entry, _) in GEA2_ENTRY_TABLE {
        driver = driver.with_known_entry(entry);
    }
    let runtime = DeviceRuntime::Metal(
        MetalHostSession::with_driver(Box::new(driver)).expect("fake Metal admission"),
    );
    CompositeHost::with_device(runtime, "fake-metal-device").expect("fake-metal composite")
}

/// The real exported bundle's plan, mapped onto a validated descriptor.
fn gea2_sequence_descriptor() -> DeviceDescriptor {
    let artifact_dir = gea2_artifact_dir();
    let envelope = load_gea2_plan(&artifact_dir);
    let descriptor = map_envelope_to_descriptor(&envelope, &artifact_dir)
        .unwrap_or_else(|error| panic!("GEA2 plan → DeviceDescriptor mapping failed: {error}"));
    descriptor.validate().unwrap_or_else(|error| {
        panic!(
            "mapped GEA2 descriptor must validate: {} ({})",
            error.message, error.code
        )
    });
    descriptor
}

/// The declared HostProvided PerProgram tensors as zero-valued F32 byte
/// payloads. The sequence ladder is structural: the bytes only satisfy the
/// declared shapes; no value is read or interpreted.
fn gea2_host_input_bytes(
    descriptor: &DeviceDescriptor,
) -> std::collections::BTreeMap<u32, DeviceByteBuffer> {
    let mut uploads = Vec::new();
    for kernel in &descriptor.kernels {
        for slot in &kernel.buffers {
            if slot.lifetime == DeviceBufferLifetime::PerProgram
                && slot.initialization == DeviceBufferInitialization::HostProvided
                && !uploads.contains(&slot.buffer_id)
            {
                uploads.push(slot.buffer_id);
            }
        }
    }
    uploads
        .into_iter()
        .map(|buffer_id| {
            let version = descriptor
                .buffer_versions
                .iter()
                .find(|version| version.buffer_id == buffer_id)
                .unwrap_or_else(|| panic!("host input buffer {buffer_id} has no keyed version"));
            let byte_len = usize::try_from(version.element_count)
                .expect("element count fits usize")
                * DeviceDataType::F32.byte_width();
            (
                buffer_id,
                DeviceByteBuffer {
                    bytes: vec![0; byte_len],
                    dtype: DeviceDataType::F32,
                    packed_format: None,
                },
            )
        })
        .collect()
}

/// One structural execution of the 64-launch block sequence on the fake
/// driver, returning the receipt after an ordered teardown.
fn gea2_execute_fake_sequence(
    host: &mut CompositeHost,
    descriptor: &DeviceDescriptor,
) -> faber_host_macos_arm64::composite_host::DeviceExecutionReceipt {
    let mut session = host
        .create_program_session(descriptor)
        .expect("fake GEA2 program session");
    let inputs = gea2_host_input_bytes(descriptor);
    let receipt = session
        .execute_with_weight_bytes(&BTreeMap::new(), &inputs)
        .expect("the 64-launch fake GEA2 sequence executes");
    session.teardown().expect("ordered GEA2 session teardown");
    receipt
}

#[test]
fn gea2_fake_sequence_has_no_cpu_substitute() {
    let descriptor = gea2_sequence_descriptor();
    let mut host = gea2_fake_host();
    let receipt = gea2_execute_fake_sequence(&mut host, &descriptor);

    // 64 launches, in the declared instance-expanded order.
    assert_eq!(receipt.launch_ids, (1..=64).collect::<Vec<u32>>());
    let declared_entries: Vec<String> = descriptor
        .launches
        .iter()
        .map(|launch| {
            descriptor.kernels[launch.kernel_index as usize]
                .entry
                .clone()
        })
        .collect();
    assert_eq!(receipt.launch_entries, declared_entries);

    // Uploads exactly once: the twelve declared HostProvided PerProgram
    // tensors (nine weights + activation + rope table + attention_scale —
    // the [8,8] scale resource the emitter repair 5f96ed340 added; the plan
    // carries no separate positions buffer — positions are rope-recipe
    // facts).
    let device = host.device().expect("device present");
    assert_eq!(
        device.driver_counters().uploads,
        12,
        "each declared host tensor uploads exactly once, at the once-upload site"
    );
    assert_eq!(device.driver_counters().module_loads, 1, "one module load");

    // Zero CPU substitutes: every launch took the driver's declared-entry
    // structural GEA2 dispatch; no fused-library body or CPU oracle ran and
    // no kernel values were produced or interpreted.
    assert!(receipt.fused_library_dispatches.is_empty());
    assert_eq!(
        receipt.allocated_buffer_versions.len(),
        101,
        "the plan's 101 version-keyed buffers are the whole allocation set"
    );
    assert_eq!(
        receipt.readbacks, 1,
        "the declared output is the only readback"
    );
    assert_eq!(receipt.outputs.len(), 1);
    let output_id = descriptor.results[0].buffer_id;
    let output = &receipt.outputs[&output_id];
    assert_eq!(
        output.len(),
        7680,
        "the declared [8,960] F32 block output is read back in full"
    );
    assert!(
        output.iter().all(|value| value.is_finite()),
        "structural launches produce no garbage bytes"
    );
    assert_eq!(
        host.device().expect("device present").live_handle_count(),
        0,
        "teardown released every handle"
    );
}

#[test]
fn gea2_fake_sequence_rejects_intermediate_readback() {
    let descriptor = gea2_sequence_descriptor();
    let mut host = gea2_fake_host();
    let receipt = gea2_execute_fake_sequence(&mut host, &descriptor);

    // Zero intermediate readbacks on the green path: the only device→host
    // transfer is the declared output at its declared observation point.
    let output_id = descriptor.results[0].buffer_id;
    assert_eq!(receipt.readbacks, 1);
    assert_eq!(receipt.outputs.len(), 1);
    assert!(receipt.outputs.contains_key(&output_id));
    // The twelve uploads are once-init at session creation (driver upload
    // counter, asserted above); the step-boundary transfer set is exactly
    // the one declared readback.
    let device = host.device().expect("device present");
    assert_eq!(
        device.driver_counters().uploads,
        12,
        "twelve declared host tensors upload once"
    );
    assert_eq!(
        receipt.transfers, 1,
        "one declared readback is the only step-boundary transfer"
    );

    // A readback attempted outside the declared output fails closed before
    // any launch: declaring the PerStep intermediate ln1 (buffer 100,
    // produced by launch 1) as a result is an undeclared readback.
    let mut intermediate_result = descriptor.clone();
    intermediate_result.results = vec![DescriptorResult {
        buffer_id: 100,
        version: 1,
        produced_by: 1,
        at_launch: 1,
    }];
    let error = match host.create_program_session(&intermediate_result) {
        Ok(_) => panic!("reading back a PerStep intermediate must fail closed"),
        Err(error) => error,
    };
    assert_eq!(error.code, E_DEVICE_DESCRIPTOR);
    assert!(
        error.message.contains("observation-point"),
        "diagnostic names the observation-point rule: {}",
        error.message
    );
    assert_eq!(
        host.device().expect("device present").live_handle_count(),
        0,
        "the rejected readback leaves no handles behind"
    );

    // A result naming a buffer the plan does not carry fails closed at
    // descriptor validation, before a session exists.
    let mut undeclared_result = descriptor.clone();
    undeclared_result.results = vec![DescriptorResult {
        buffer_id: u32::MAX,
        version: 1,
        produced_by: 64,
        at_launch: 64,
    }];
    let error = undeclared_result
        .validate()
        .expect_err("a result naming an undeclared buffer must fail closed");
    assert_eq!(error.code, E_DEVICE_DESCRIPTOR);
    assert!(
        error.message.contains("no kernel slot allocates"),
        "diagnostic names the unknown buffer: {}",
        error.message
    );
}

// ---------------------------------------------------------------------------
// GEA2-U5e: the physical Metal block receipt. The ignored test executes the
// exported 64-launch block program on the real Metal driver with the frozen
// U1 inputs — nine GGUF tensor ranges SHA-checked against the U1 manifest,
// the activation fixture, and the Gradus rope table — reads back only the
// declared [8,960] output after one step-boundary sync, and compares the
// physical values against the scalar block oracle authored below from the
// §2 block arithmetic (the GEA1 `reference_gemv` mirror pattern: no gradus,
// radix, or host-helper calls). Per-kernel rows and seam classification are
// U6's; the receipt lands at `GEA2_METAL_RECEIPT`.
// ---------------------------------------------------------------------------

/// Frozen block geometry (gea2-delivery §2 / U1 manifest facts).
const GEA2_T: usize = 8;
const GEA2_D: usize = 960;
const GEA2_H: usize = 15;
const GEA2_KV: usize = 5;
const GEA2_HD: usize = 64;
const GEA2_F: usize = 2560;
const GEA2_EPS: f32 = 1e-5;
const GEA2_SCALE: f32 = 0.125;

/// The U1 manifest's nine frozen tensors: GGUF name → logical element count.
const GEA2_FROZEN_TENSORS: [(&str, usize); 9] = [
    ("blk.0.attn_norm.weight", 960),
    ("blk.0.attn_q.weight", 921_600),
    ("blk.0.attn_k.weight", 307_200),
    ("blk.0.attn_v.weight", 307_200),
    ("blk.0.attn_output.weight", 921_600),
    ("blk.0.ffn_norm.weight", 960),
    ("blk.0.ffn_gate.weight", 2_457_600),
    ("blk.0.ffn_up.weight", 2_457_600),
    ("blk.0.ffn_down.weight", 2_457_600),
];

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("faberlang workspace root")
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut child = Command::new("shasum")
        .args(["-a", "256"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn shasum for bytes");
    child
        .stdin
        .take()
        .expect("shasum stdin")
        .write_all(bytes)
        .expect("write bytes to shasum");
    let output = child.wait_with_output().expect("wait for shasum");
    assert!(
        output.status.success(),
        "shasum stdin failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()
        .expect("shasum digest")
        .to_owned()
}

fn sha256_file(path: &Path) -> String {
    let output = Command::new("shasum")
        .args(["-a", "256"])
        .arg(path)
        .output()
        .unwrap_or_else(|error| panic!("run shasum for {}: {error}", path.display()));
    assert!(
        output.status.success(),
        "shasum failed for {}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_owned()
}

fn git_revision(path: &Path) -> String {
    let output = Command::new("git")
        .args(["-C"])
        .arg(path)
        .args(["rev-parse", "HEAD"])
        .output()
        .unwrap_or_else(|error| panic!("git revision {}: {error}", path.display()));
    assert!(
        output.status.success(),
        "git revision failed for {}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_owned()
}

fn read_range(path: &Path, range: [u64; 2]) -> Vec<u8> {
    assert!(range[1] > range[0], "empty range in {}", path.display());
    let length = usize::try_from(range[1] - range[0]).expect("range fits host usize");
    let mut file =
        File::open(path).unwrap_or_else(|error| panic!("open {}: {error}", path.display()));
    file.seek(SeekFrom::Start(range[0]))
        .unwrap_or_else(|error| panic!("seek {}: {error}", path.display()));
    let mut bytes = vec![0; length];
    file.read_exact(&mut bytes)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    bytes
}

fn decode_f32_le(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0, "F32 payload is not word aligned");
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn measured(value: impl serde::Serialize) -> Value {
    json!({"value": value, "status": "measured"})
}

fn derived(value: impl serde::Serialize, formula: &str) -> Value {
    json!({"value": value, "status": "derived", "formula": formula})
}

fn unmeasured(reason: &str) -> Value {
    json!({"value": Value::Null, "status": "unmeasured", "reason": reason})
}

/// The frozen U1 input set for the scalar block oracle: the activation row
/// block, the nine decoded GGUF tensors, and the rope table.
struct Gea2ScalarBlockInputs {
    x: Vec<f32>,
    attn_norm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    o: Vec<f32>,
    ffn_norm: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
    rope: Vec<f32>,
}

fn gea2_f32_gguf_path() -> PathBuf {
    if let Some(path) = std::env::var_os("GEA2_F32_GGUF") {
        return PathBuf::from(path);
    }
    // The U1 manifest freezes the identity by digest, not path; the derived
    // model cache location is the frozen run's default (GEA1 §7 precedent).
    PathBuf::from(
        "/Users/ianzepp/ai/models/derived/HuggingFaceTB/SmolLM2-360M-Instruct/\
a10cc1512eabd3dde888204e902eca88bddb4951/SmolLM2-360M-Instruct-f32.gguf",
    )
}

/// Range-read and SHA-check the nine frozen tensors plus the activation and
/// rope fixtures against the U1 manifest. Any drift is a stop, not a new
/// fixture.
fn gea2_scalar_block_inputs(manifest: &Value) -> (Gea2ScalarBlockInputs, u64, usize) {
    let workspace = workspace_root();
    let gguf_path = gea2_f32_gguf_path();
    assert!(
        gguf_path.is_file(),
        "missing F32 GGUF {}",
        gguf_path.display()
    );
    assert_eq!(
        sha256_file(&gguf_path),
        manifest["model"]["derived_f32_gguf"]["sha256"]
            .as_str()
            .expect("manifest gguf digest"),
        "the F32 GGUF no longer matches the frozen U1 identity"
    );

    let tensors = manifest["tensors"].as_array().expect("manifest tensors");
    assert_eq!(tensors.len(), GEA2_FROZEN_TENSORS.len());
    let mut range_read_us = 0_u64;
    let mut range_read_bytes = 0_usize;
    let mut decoded: Vec<Vec<f32>> = Vec::with_capacity(tensors.len());
    for (row, (gguf_name, elements)) in GEA2_FROZEN_TENSORS.iter().enumerate() {
        let tensor = &tensors[row];
        assert_eq!(
            tensor["name"].as_str().expect("tensor name"),
            *gguf_name,
            "U1 manifest tensor order drifted at row {row}"
        );
        assert_eq!(
            tensor["values"].as_u64().expect("values") as usize,
            *elements
        );
        let range = [
            tensor["absolute_range"][0].as_u64().expect("range start"),
            tensor["absolute_range"][1].as_u64().expect("range end"),
        ];
        let started = Instant::now();
        let bytes = read_range(&gguf_path, range);
        range_read_us += started.elapsed().as_micros() as u64;
        range_read_bytes += bytes.len();
        assert_eq!(
            bytes.len(),
            *elements * DeviceDataType::F32.byte_width(),
            "tensor {gguf_name} range length disagrees with the frozen element count"
        );
        assert_eq!(
            sha256_bytes(&bytes),
            tensor["tensor_sha256"].as_str().expect("tensor digest"),
            "tensor {gguf_name} drifted against the U1 manifest digest"
        );
        decoded.push(decode_f32_le(&bytes));
    }

    let activation_path = workspace.join("gradus/fixtures/activations/gea2-block-input.bin");
    let activation = fs::read(&activation_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", activation_path.display()));
    assert_eq!(activation.len(), GEA2_T * GEA2_D * 4);
    assert_eq!(
        sha256_bytes(&activation),
        manifest["activation"]["sha256"]
            .as_str()
            .expect("activation digest"),
        "activation fixture drifted against the U1 manifest"
    );
    let rope_path = workspace.join("gradus/fixtures/rope/gea2-rope-table.f32");
    let rope_bytes = fs::read(&rope_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", rope_path.display()));
    assert_eq!(rope_bytes.len(), GEA2_T * 32 * 3 * 4);
    assert_eq!(
        sha256_bytes(&rope_bytes),
        manifest["rope_table"]["sha256"]
            .as_str()
            .expect("rope digest"),
        "rope table drifted against the U1 manifest"
    );

    let inputs = Gea2ScalarBlockInputs {
        x: decode_f32_le(&activation),
        attn_norm: decoded[0].clone(),
        q: decoded[1].clone(),
        k: decoded[2].clone(),
        v: decoded[3].clone(),
        o: decoded[4].clone(),
        ffn_norm: decoded[5].clone(),
        gate: decoded[6].clone(),
        up: decoded[7].clone(),
        down: decoded[8].clone(),
        rope: decode_f32_le(&rope_bytes),
    };
    (inputs, range_read_us, range_read_bytes)
}

// --- the mirrored scalar block oracle (§2 arithmetic, independent of every
// gradus/radix/host body; F32 left-to-right accumulation throughout) ---

fn oracle_rmsnorm(x: &[f32], gamma: &[f32], rows: usize, width: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * width];
    for row in 0..rows {
        let mut sumsq = 0.0_f32;
        for j in 0..width {
            let value = x[row * width + j];
            sumsq += value * value;
        }
        let mean = sumsq / width as f32;
        let scale = 1.0 / (mean + GEA2_EPS).sqrt();
        for col in 0..width {
            out[row * width + col] = x[row * width + col] * scale * gamma[col];
        }
    }
    out
}

/// `a [rows, k] · W` where `w_raw` is the GGUF's `[out][in]` row-major
/// weight layout: out-row `col` holds the `k` input weights for output
/// `col` (`w_raw[col * k + i]`). This is the container's one weight
/// contract — the U1/GEA3 oracle `matmul` and every exported gemm kernel
/// index the same layout (GEA2-U5g layout-truth ruling).
fn oracle_gemm(a: &[f32], w_raw: &[f32], rows: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), rows * k);
    assert_eq!(w_raw.len(), k * n);
    let mut out = vec![0.0_f32; rows * n];
    for row in 0..rows {
        for col in 0..n {
            let mut acc = 0.0_f32;
            for i in 0..k {
                acc += a[row * k + i] * w_raw[col * k + i];
            }
            out[row * n + col] = acc;
        }
    }
    out
}

/// Apply consecutive-pair RoPE to every head dimension of a packed
/// `[T, heads*64]` projection. Row `t` carries position `t` (positions
/// 0..7); the table row is `[pos, pair, {angle, cos, sin}]`.
fn oracle_rope_packed(packed: &[f32], width: usize, rope: &[f32]) -> Vec<f32> {
    let mut out = packed.to_vec();
    for row in 0..GEA2_T {
        for col in 0..width {
            let within = col % GEA2_HD;
            let pair = within / 2;
            let cos_t = rope[(row * 32 + pair) * 3 + 1];
            let sin_t = rope[(row * 32 + pair) * 3 + 2];
            let base = row * width + col - within;
            if within % 2 == 0 {
                let next = packed[base + within + 1];
                out[base + within] = packed[base + within] * cos_t - next * sin_t;
            } else {
                let prev = packed[base + within - 1];
                out[base + within] = prev * sin_t + packed[base + within] * cos_t;
            }
        }
    }
    out
}

fn oracle_softmax_causal(scores: &[f32], row: usize) -> Vec<f32> {
    let mut row_max = scores[row * GEA2_T];
    for j in 1..=row {
        row_max = row_max.max(scores[row * GEA2_T + j]);
    }
    let mut row_sum = 0.0_f32;
    for j in 0..=row {
        row_sum += (scores[row * GEA2_T + j] - row_max).exp();
    }
    (0..=row)
        .map(|j| (scores[row * GEA2_T + j] - row_max).exp() / row_sum)
        .collect()
}

/// The independent scalar mirror of the §2 block forward at T=8: returns
/// the [8,960] block output.
fn oracle_scalar_block(inputs: &Gea2ScalarBlockInputs) -> Vec<f32> {
    oracle_scalar_block_rows(inputs).block_output
}

/// A head window `[rows, width]` carved from a packed `[rows, heads*width]`
/// projection (the plan's per-head windowed entries write exactly these).
fn oracle_head_window(
    packed: &[f32],
    rows: usize,
    head: usize,
    heads: usize,
    width: usize,
) -> Vec<f32> {
    let mut window = vec![0.0_f32; rows * width];
    for row in 0..rows {
        for d in 0..width {
            window[row * width + d] = packed[row * heads * width + head * width + d];
        }
    }
    window
}

/// Every §2 policy-row intermediate of the block forward at T=8, produced by
/// the scalar mirror (the instrumented diagnostic's reference rows; the
/// plan's per-head window buffers carry slices of the packed projections and
/// attention rows).
struct Gea2ScalarBlockRows {
    ln1: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    rope_q: Vec<f32>,
    rope_k: Vec<f32>,
    q_head: Vec<Vec<f32>>,
    k_head: Vec<Vec<f32>>,
    v_head: Vec<Vec<f32>>,
    scores: Vec<Vec<f32>>,
    probabilities: Vec<Vec<f32>>,
    contexts: Vec<Vec<f32>>,
    o_projection: Vec<f32>,
    residual1: Vec<f32>,
    ln2: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    swiglu: Vec<f32>,
    down: Vec<f32>,
    block_output: Vec<f32>,
}

fn oracle_scalar_block_rows(inputs: &Gea2ScalarBlockInputs) -> Gea2ScalarBlockRows {
    let ln1 = oracle_rmsnorm(&inputs.x, &inputs.attn_norm, GEA2_T, GEA2_D);
    let q = oracle_gemm(&ln1, &inputs.q, GEA2_T, GEA2_D, GEA2_D);
    let k = oracle_gemm(&ln1, &inputs.k, GEA2_T, GEA2_D, GEA2_KV * GEA2_HD);
    let v = oracle_gemm(&ln1, &inputs.v, GEA2_T, GEA2_D, GEA2_KV * GEA2_HD);
    let rope_q = oracle_rope_packed(&q, GEA2_D, &inputs.rope);
    let rope_k = oracle_rope_packed(&k, GEA2_KV * GEA2_HD, &inputs.rope);

    let q_head = (0..GEA2_H)
        .map(|head| oracle_head_window(&rope_q, GEA2_T, head, GEA2_H, GEA2_HD))
        .collect();
    let k_head = (0..GEA2_KV)
        .map(|head| oracle_head_window(&rope_k, GEA2_T, head, GEA2_KV, GEA2_HD))
        .collect();
    let v_head = (0..GEA2_KV)
        .map(|head| oracle_head_window(&v, GEA2_T, head, GEA2_KV, GEA2_HD))
        .collect();

    // Attention: 15 query heads over 5 KV heads (GQA ratio 3, head h → h/3).
    // The device score_gemm writes the FULL 8×8 product scaled by
    // GEA2_SCALE (the causal mask lives in causal_softmax, which zeroes the
    // upper triangle), so the reference score row is the full product.
    let mut ctx = vec![0.0_f32; GEA2_T * GEA2_D];
    let mut scores = Vec::with_capacity(GEA2_H);
    let mut probabilities = Vec::with_capacity(GEA2_H);
    let mut contexts = Vec::with_capacity(GEA2_H);
    for h in 0..GEA2_H {
        let kv = h / 3;
        let mut score = vec![0.0_f32; GEA2_T * GEA2_T];
        for row in 0..GEA2_T {
            for col in 0..GEA2_T {
                let mut acc = 0.0_f32;
                for d in 0..GEA2_HD {
                    acc += rope_q[row * GEA2_D + h * GEA2_HD + d]
                        * rope_k[col * GEA2_KV * GEA2_HD + kv * GEA2_HD + d];
                }
                score[row * GEA2_T + col] = acc * GEA2_SCALE;
            }
        }
        let mut prob = vec![0.0_f32; GEA2_T * GEA2_T];
        for row in 0..GEA2_T {
            let window = oracle_softmax_causal(&score, row);
            prob[row * GEA2_T..row * GEA2_T + row + 1].copy_from_slice(&window);
        }
        let mut context = vec![0.0_f32; GEA2_T * GEA2_HD];
        for row in 0..GEA2_T {
            for d in 0..GEA2_HD {
                let mut acc = 0.0_f32;
                for col in 0..GEA2_T {
                    acc += prob[row * GEA2_T + col] * v[col * GEA2_KV * GEA2_HD + kv * GEA2_HD + d];
                }
                context[row * GEA2_HD + d] = acc;
            }
        }
        scores.push(score);
        probabilities.push(prob);
        contexts.push(context);
    }
    for h in 0..GEA2_H {
        for row in 0..GEA2_T {
            for d in 0..GEA2_HD {
                ctx[row * GEA2_D + h * GEA2_HD + d] = contexts[h][row * GEA2_HD + d];
            }
        }
    }

    let o_projection = oracle_gemm(&ctx, &inputs.o, GEA2_T, GEA2_D, GEA2_D);
    let residual1: Vec<f32> = inputs
        .x
        .iter()
        .zip(&o_projection)
        .map(|(left, right)| left + right)
        .collect();
    let ln2 = oracle_rmsnorm(&residual1, &inputs.ffn_norm, GEA2_T, GEA2_D);
    let gate = oracle_gemm(&ln2, &inputs.gate, GEA2_T, GEA2_D, GEA2_F);
    let up = oracle_gemm(&ln2, &inputs.up, GEA2_T, GEA2_D, GEA2_F);
    let swiglu: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| (g / ((-g).exp() + 1.0)) * u)
        .collect();
    let down = oracle_gemm(&swiglu, &inputs.down, GEA2_T, GEA2_F, GEA2_D);
    let block_output: Vec<f32> = residual1
        .iter()
        .zip(&down)
        .map(|(left, right)| left + right)
        .collect();
    Gea2ScalarBlockRows {
        ln1,
        q,
        k,
        v,
        rope_q,
        rope_k,
        q_head,
        k_head,
        v_head,
        scores,
        probabilities,
        contexts,
        o_projection,
        residual1,
        ln2,
        gate,
        up,
        swiglu,
        down,
        block_output,
    }
}

// --- frozen-policy comparison (mirrors the U1 `compare_f32` semantics) ---

fn ulp_distance(left: f32, right: f32) -> u32 {
    if left.to_bits() == right.to_bits() || left == right {
        return 0;
    }
    fn ordered_bits(bits: u32) -> u32 {
        if bits & 0x8000_0000 != 0 {
            0x8000_0000_u32 - (bits & 0x7fff_ffff)
        } else {
            0x8000_0000_u32 + bits
        }
    }
    ordered_bits(left.to_bits()).abs_diff(ordered_bits(right.to_bits()))
}

/// The frozen `block_output` policy row: abs ≤ 5e-4, rel ≤ 2e-5 (denominator
/// `max(|expected|, |observed|, MIN_POSITIVE)`), ULP ≤ 1024.
/// GEA2-U5g layout-truth pin: the scalar mirror oracle reads the GGUF
/// weights in the container's `[out][in]` row-major layout — the same
/// contract the U1/GEA3 oracles and every exported gemm kernel index. The
/// pre-U5g oracle read `[k][n]` and reproduced the retired receipt's
/// −5.8359156 row 0 (to 1.8e-6); the correct row-0 class is −1.0494539
/// (digest-verified bytes, llama-cli-exact semantics). No device — the
/// ignored marker is the frozen F32 GGUF's local model-cache identity.
#[test]
#[ignore = "reads the frozen F32 GGUF from the local model cache (the §6 fixture identity)"]
fn gea2_scalar_block_oracle_pins_row_zero_under_out_in_weights() {
    let workspace = workspace_root();
    let manifest_path = workspace
        .join("radix/docs/factory/gpu-execution-architecture/evidence/gea2-input-manifest.json");
    let manifest: Value = serde_json::from_slice(
        &fs::read(&manifest_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", manifest_path.display())),
    )
    .expect("valid GEA2 U1 input manifest");
    let (inputs, _range_read_us, _range_read_bytes) = gea2_scalar_block_inputs(&manifest);
    let expected = oracle_scalar_block(&inputs);
    assert!(
        expected.iter().all(|value| value.is_finite()),
        "the scalar block oracle produces finite F32 throughout"
    );
    // The digest-verified reference value is −1.0494539 (GEA3-U1 oracle
    // bytes); this mirror is an independent F32 implementation (std `exp`,
    // its own accumulation order), so the pin admits cross-implementation
    // ulp slack — while staying six orders of magnitude from the wrong
    // transposed class (−5.8359156).
    assert!(
        (expected[0] - (-1.049_453_9_f32)).abs() <= 2e-6,
        "row 0 must be the [out][in]-weighted block output class −1.0494539, got {}",
        expected[0]
    );
}

fn gea2_compare_block_output(expected: &[f32], observed: &[f32]) -> (f32, f32, u32, usize) {
    assert_eq!(expected.len(), observed.len());
    assert!(
        expected
            .iter()
            .chain(observed)
            .all(|value| value.is_finite()),
        "the block comparison rejects non-finite values"
    );
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut max_ulp = 0_u32;
    let mut first_index = 0;
    for (index, (&want, &got)) in expected.iter().zip(observed).enumerate() {
        let absolute = (got - want).abs();
        if absolute > max_abs {
            max_abs = absolute;
            first_index = index;
        }
        let denominator = want.abs().max(got.abs()).max(f32::MIN_POSITIVE);
        max_rel = max_rel.max(absolute / denominator);
        max_ulp = max_ulp.max(ulp_distance(want, got));
    }
    (max_abs, max_rel, max_ulp, first_index)
}

/// Build the real 12-tensor HostProvided upload set: every declared
/// HostProvided PerProgram slot resolved by name to its frozen U1 bytes (the
/// attention_scale resource added by the emitter repair 5f96ed340 carries
/// the §2 frozen 0.125 constant).
fn gea2_real_host_inputs(
    descriptor: &DeviceDescriptor,
    inputs: &Gea2ScalarBlockInputs,
) -> BTreeMap<u32, DeviceByteBuffer> {
    let mut uploads = BTreeMap::new();
    for kernel in &descriptor.kernels {
        for slot in &kernel.buffers {
            if slot.lifetime != DeviceBufferLifetime::PerProgram
                || slot.initialization != DeviceBufferInitialization::HostProvided
                || uploads.contains_key(&slot.buffer_id)
            {
                continue;
            }
            let values: Vec<f32> = match slot.buffer_name.as_str() {
                "activation_x" => inputs.x.clone(),
                "attn_norm_weight" => inputs.attn_norm.clone(),
                "q_weight" => inputs.q.clone(),
                "k_weight" => inputs.k.clone(),
                "v_weight" => inputs.v.clone(),
                "o_weight" => inputs.o.clone(),
                "ffn_norm_weight" => inputs.ffn_norm.clone(),
                "gate_weight" => inputs.gate.clone(),
                "up_weight" => inputs.up.clone(),
                "down_weight" => inputs.down.clone(),
                "rope_table" => inputs.rope.clone(),
                "attention_scale" => vec![GEA2_SCALE; 64],
                other => panic!("unknown HostProvided GEA2 tensor `{other}`"),
            };
            assert_eq!(
                values.len() as u64,
                slot.element_count,
                "tensor `{}` byte length disagrees with the plan's declared shape",
                slot.buffer_name
            );
            uploads.insert(
                slot.buffer_id,
                DeviceByteBuffer {
                    bytes: f32_le_bytes(&values),
                    dtype: DeviceDataType::F32,
                    packed_format: None,
                },
            );
        }
    }
    assert_eq!(uploads.len(), 12, "the frozen upload set is twelve tensors");
    uploads
}

#[test]
#[ignore = "physical Metal gate; run only with the exact §6 command pair"]
fn gea2_real_metal_block_receipt() {
    let e2e_start = Instant::now();
    std::env::set_var("FABER_PER_OP_TIMING", "1");
    let workspace = workspace_root();
    let receipt_path = PathBuf::from(
        std::env::var_os("GEA2_METAL_RECEIPT")
            .expect("GEA2_METAL_RECEIPT must identify the receipt output"),
    );

    // Frozen identities: the U1 manifest gates every input byte.
    let manifest_path = workspace
        .join("radix/docs/factory/gpu-execution-architecture/evidence/gea2-input-manifest.json");
    let manifest: Value = serde_json::from_slice(
        &fs::read(&manifest_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", manifest_path.display())),
    )
    .expect("valid GEA2 U1 input manifest");
    assert_eq!(manifest["schema"], "gea2-input-manifest-v1");
    assert_eq!(manifest["delivery"], "GEA2-U1");
    assert_eq!(
        manifest["geometry"]["token_rows"].as_u64(),
        Some(GEA2_T as u64)
    );
    assert_eq!(
        manifest["geometry"]["hidden_dim"].as_u64(),
        Some(GEA2_D as u64)
    );
    assert_eq!(
        manifest["geometry"]["query_heads"].as_u64(),
        Some(GEA2_H as u64)
    );
    assert_eq!(
        manifest["geometry"]["kv_heads"].as_u64(),
        Some(GEA2_KV as u64)
    );
    assert_eq!(
        manifest["geometry"]["intermediate_dim"].as_u64(),
        Some(GEA2_F as u64)
    );
    let (inputs, tensor_range_read_us, tensor_range_read_bytes) =
        gea2_scalar_block_inputs(&manifest);

    // Entry identities: the exported bundle's fourteen members, hash-verified.
    let artifact_dir = gea2_artifact_dir();
    let bundle_path = artifact_dir.join("gea2-artifact-bundle-manifest.json");
    let bundle: Value = serde_json::from_slice(
        &fs::read(&bundle_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", bundle_path.display())),
    )
    .expect("valid GEA2 artifact bundle manifest");
    assert_eq!(bundle["delivery"], "GEA2-U5a");
    let entries = bundle["entries"].as_array().expect("bundle entries");
    assert_eq!(entries.len(), GEA2_ENTRY_TABLE.len());
    let entry_identities: Vec<Value> = entries
        .iter()
        .map(|entry| {
            let artifact = artifact_dir.join(entry["artifact"].as_str().expect("artifact member"));
            let bytes = fs::read(&artifact)
                .unwrap_or_else(|error| panic!("read {}: {error}", artifact.display()));
            assert_eq!(
                sha256_bytes(&bytes),
                entry["artifact_sha256"].as_str().expect("artifact digest"),
                "bundle member {} drifted",
                artifact.display()
            );
            json!({
                "entry": entry["entry"],
                "provenance": entry["provenance"],
                "artifact": entry["artifact"],
                "artifact_sha256": entry["artifact_sha256"],
                "artifact_bytes": bytes.len(),
                "descriptor_sha256": entry["descriptor_sha256"],
                "source": entry["source"],
                "source_sha256": entry["source_sha256"],
            })
        })
        .collect();
    let plan_bytes = fs::read(artifact_dir.join(PLAN_MEMBER))
        .unwrap_or_else(|error| panic!("read plan member: {error}"));
    let plan_sha256 = sha256_bytes(&plan_bytes);

    // Revisions and physical device identity.
    let gradus_revision = git_revision(&workspace.join("gradus"));
    let radix_revision = git_revision(&workspace.join("radix"));
    let hosts_revision = git_revision(&workspace.join("hosts"));
    let devices = enumerate_metal_physical_devices().expect("Metal device enumeration");
    assert!(
        !devices.is_empty(),
        "Metal selected but no physical device identity exists"
    );
    let device = &devices[0];
    assert!(!device.registry_id.is_empty(), "registry identity required");
    assert!(device.api_total_bytes > 0, "memory capability required");

    // The plan, mapped and admitted exactly as the fake ladder admits it.
    let descriptor = gea2_sequence_descriptor();

    // The real session: physical Metal, one module compile, one execution.
    let runtime =
        DeviceRuntime::Metal(MetalHostSession::try_open().expect("physical Metal admission"));
    let mut host =
        CompositeHost::with_device(runtime, "metal-device").expect("real-metal composite");
    let mut session = host
        .create_program_session(&descriptor)
        .expect("real GEA2 program session");
    let module_hash = session.module_hash();
    let program_graph_hash = session.program_graph_hash().to_owned();
    let module_compile_us = unmeasured("session module-compile timing is crate-private");
    let per_program_alloc_us = unmeasured("session per-program allocation timing is crate-private");
    let uploads = gea2_real_host_inputs(&descriptor, &inputs);
    let weight_bytes_total: usize = uploads.values().map(|buffer| buffer.bytes.len()).sum();
    let receipt = session
        .execute_with_weight_bytes(&BTreeMap::new(), &uploads)
        .expect("the 64-launch GEA2 block executes on physical Metal");
    session.teardown().expect("ordered GEA2 session teardown");
    assert_eq!(
        host.device().expect("device present").live_handle_count(),
        0,
        "teardown released every handle"
    );

    // Structural facts: the §6 sequence contract, on the real driver.
    assert_eq!(receipt.launch_ids, (1..=64).collect::<Vec<u32>>());
    let declared_entries: Vec<String> = descriptor
        .launches
        .iter()
        .map(|launch| {
            descriptor.kernels[launch.kernel_index as usize]
                .entry
                .clone()
        })
        .collect();
    assert_eq!(receipt.launch_entries, declared_entries);
    assert_eq!(
        receipt.fused_library_dispatches.len(),
        0,
        "zero CPU substitutes"
    );
    assert!(
        receipt.allocated_buffer_versions.len() == 101,
        "the plan's 101 version-keyed buffers are the whole allocation set"
    );
    let counters = host.device().expect("device present").driver_counters();
    assert_eq!(
        counters.uploads, 12,
        "each declared host tensor uploads exactly once"
    );
    // S2-8 real-device counter contract: the system driver reports module
    // lifecycle counters as zero (only the fake drivers count them); the
    // single module compile is proven by the 64 launches executing against
    // the session's compiled pipeline and recorded as `module_hash_fnv1a`.
    assert_eq!(
        counters.module_loads, 0,
        "system driver reports module counters as zero (S2-8 real-device gate)"
    );
    assert_eq!(
        receipt.readbacks, 1,
        "the declared output is the only readback"
    );
    assert_eq!(receipt.outputs.len(), 1);
    assert_eq!(
        receipt.transfers, 1,
        "one step-boundary transfer: the declared readback"
    );
    assert_eq!(
        receipt.syncs, 1,
        "one step-boundary sync after the last launch"
    );
    let output_id = descriptor.results[0].buffer_id;
    let observed = &receipt.outputs[&output_id];
    assert_eq!(
        observed.len(),
        GEA2_T * GEA2_D,
        "the declared [8,960] output in full"
    );
    assert!(
        observed.iter().all(|value| value.is_finite()),
        "finite physical output"
    );

    // Dependency-edge satisfaction: 78 declared edges, producer before consumer.
    assert_eq!(descriptor.data_flow.len(), 78);
    let edge_rows: Vec<Value> = descriptor
        .data_flow
        .iter()
        .map(|edge| {
            let satisfied = receipt.launch_ids.contains(&edge.producer)
                && receipt.launch_ids.contains(&edge.consumer)
                && edge.producer < edge.consumer;
            assert!(
                satisfied,
                "edge {}→{} unsatisfied",
                edge.producer, edge.consumer
            );
            json!({
                "producer": edge.producer,
                "consumer": edge.consumer,
                "buffer": edge.buffer_id,
                "version": edge.version,
            })
        })
        .collect();

    // The block numerical row: physical output vs the independent scalar
    // oracle. GEA2-U5j pre-timing amendment (CTO ruling f51387e4): the
    // verdict is the element-wise disjunction `abs(e) <= atol OR (rel(e) <=
    // rtol AND ulp(e) <= ulp_row)` — every frozen constant unchanged (abs
    // 5e-4 / rel 2e-5 / ulp 1024); only the composition changes from the
    // aggregate-conjunction-of-maxima form. The aggregate metrics remain in
    // the receipt for the record.
    let expected = oracle_scalar_block(&inputs);
    let (max_abs, max_rel, max_ulp, first_index) = gea2_compare_block_output(&expected, observed);
    let (policy_pass, failing_elements) =
        gea2_element_wise_passes(&expected, observed, 5e-4, 2e-5, 1024);
    assert_eq!(
        failing_elements, 0,
        "the amended element-wise gate must pass every block element"
    );
    let sample_rows: Vec<Value> = (0..8)
        .map(|index| {
            json!({
                "index": index,
                "expected_f32": expected[index],
                "observed_f32": observed[index],
                "absolute_error": (observed[index] - expected[index]).abs(),
            })
        })
        .collect();

    // Stage split (derived): rmsnorm-attn | attention (rope..o_proj/residual1)
    // | ffn (post-attention rmsnorm..block_output), by launch position.
    let entries_flat = &receipt.launch_entries;
    let attention_start = entries_flat
        .iter()
        .position(|entry| entry == "rope_q")
        .expect("rope_q launch");
    let ffn_start = (0..entries_flat.len())
        .find(|&index| index > attention_start && entries_flat[index] == "rmsnorm")
        .expect("post-attention rmsnorm launch");
    let stage_of = |index: usize| -> &'static str {
        if index < attention_start {
            "rmsnorm_attn"
        } else if index < ffn_start {
            "attention"
        } else {
            "ffn"
        }
    };
    let launch_rows: Vec<Value> = descriptor
        .launches
        .iter()
        .enumerate()
        .map(|(order, launch)| {
            let kernel = &descriptor.kernels[launch.kernel_index as usize];
            json!({
                "order": order,
                "launch_id": launch.id,
                "entry": kernel.entry,
                "stage": stage_of(order),
                "grid": kernel.grid,
                "block": kernel.block,
            })
        })
        .collect();
    let gpu_total: u64 = receipt.launch_gpu_us.iter().sum();
    let per_stage_gpu_us = if receipt.launch_gpu_us.len() == 64 {
        let mut stages: BTreeMap<&str, u64> = BTreeMap::new();
        for (index, gpu) in receipt.launch_gpu_us.iter().enumerate() {
            *stages.entry(stage_of(index)).or_default() += gpu;
        }
        derived(
            stages,
            "sum of per-encoder GPU timestamps grouped by launch-position stage",
        )
    } else {
        unmeasured("per-encoder GPU timestamps were not sampled")
    };

    let output_bytes = f32_le_bytes(observed);
    let weight_allocations: Vec<Value> = uploads
        .iter()
        .map(|(buffer_id, buffer)| {
            let name = descriptor
                .kernels
                .iter()
                .flat_map(|kernel| &kernel.buffers)
                .find(|slot| slot.buffer_id == *buffer_id)
                .map(|slot| slot.buffer_name.as_str())
                .unwrap_or("unknown");
            json!({
                "buffer_id": buffer_id,
                "name": name,
                "bytes": buffer.bytes.len(),
                "lifetime": "per-program",
                "initialization": "host-provided",
                "upload_count": 1,
            })
        })
        .collect();
    let intermediate_allocations: Vec<Value> = descriptor
        .buffer_versions
        .iter()
        .filter(|version| !uploads.contains_key(&version.buffer_id))
        .map(|version| {
            json!({
                "buffer_id": version.buffer_id,
                "version": version.version,
                "elements": version.element_count,
                "bytes": version.element_count as usize * DeviceDataType::F32.byte_width(),
                "initialization": "kernel-initialized",
            })
        })
        .collect();
    assert_eq!(
        intermediate_allocations.len(),
        89,
        "101 version-keyed buffers minus the 12 host uploads (the U5g windowed v-projection grew the plan by five v_head windows)"
    );

    let hostname = Command::new("hostname")
        .output()
        .ok()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned());
    let e2e_us = e2e_start.elapsed().as_micros() as u64;
    let receipt_json = json!({
        "schema": "gea2-metal-receipt-v1",
        "delivery": "GEA2-U5e",
        "backend": "Metal",
        "machine": hostname,
        "identities": {
            "entries": entry_identities,
            "program_plan_sha256": plan_sha256,
            "module_hash_fnv1a": module_hash,
            "program_graph_sha256": program_graph_hash,
            "activation_sha256": manifest["activation"]["sha256"],
            "rope_table_sha256": manifest["rope_table"]["sha256"],
            "derived_f32_gguf_sha256": manifest["model"]["derived_f32_gguf"]["sha256"],
        },
        "revisions": {
            "source_model": manifest["model"]["revision"],
            "gradus": gradus_revision,
            "radix": radix_revision,
            "hosts": hosts_revision,
        },
        "physical_device": {
            "backend": "Metal",
            "registry_id": device.registry_id,
            "model": device.device_model,
            "api_total_bytes": device.api_total_bytes,
            "max_threads_per_workgroup": device.max_threads_per_workgroup,
            "workgroup_shared_memory_min_bytes": device.workgroup_shared_memory_min_bytes,
            "workgroup_shared_memory_max_bytes": device.workgroup_shared_memory_max_bytes,
            "collective_width": device.collective_width,
            "unified_memory": device.unified_memory,
        },
        "allocations": {
            "host_tensors": weight_allocations,
            "intermediates": intermediate_allocations,
        },
        "launch_sequence": launch_rows,
        "dependency_edges": edge_rows,
        "zero_cpu_attestation": {
            "cpu_substitute_count": 0,
            "cpu_bridge_count": 0,
            "fused_library_dispatches": receipt.fused_library_dispatches.len(),
            "execution_session": "MetalHostSession::try_open",
            "fake_driver_used": false,
            "uploads": counters.uploads,
            "module_loads": counters.module_loads,
            "intermediate_readbacks": 0,
            "step_boundary_syncs": receipt.syncs,
            "completion_boundary": receipt.completion_boundary.spelling(),
        },
        "declared_output": {
            "buffer_id": output_id,
            "elements": GEA2_T * GEA2_D,
            "dtype": "F32",
            "bytes": output_bytes.len(),
            "sha256": sha256_bytes(&output_bytes),
        },
        "block_output_comparison": {
            "oracle": "independent scalar F32 mirror of gea2-delivery §2 authored in this test (no gradus/radix/host-helper calls)",
            "policy": {"max_absolute_error": 5e-4, "max_relative_error": 2e-5, "max_ulp_distance": 1024},
            "composition": "element-wise disjunction (GEA2-U5j pre-timing amendment, CTO ruling f51387e4): abs(e) <= atol OR (rel(e) <= rtol AND ulp(e) <= ulp_row); every frozen constant unchanged",
            "max_absolute_error": max_abs,
            "max_relative_error": max_rel,
            "max_ulp_distance": max_ulp,
            "first_largest_error_index": first_index,
            "failing_elements": failing_elements,
            "policy_pass": policy_pass,
            "sample_rows": sample_rows,
            "v5_comparison_verbatim": {
                "commit": "65c952196",
                "max_absolute_error": 4.9591064453125e-05,
                "max_relative_error": 1.133144460618496e-2,
                "max_ulp_distance": 131072,
                "first_largest_error_index": 1047,
            },
        },
        "amendment": {
            "id": "GEA2-U5j",
            "ruling": "head-cto task f51387e4, mail cd41bea1",
            "record": "radix/docs/factory/gpu-execution-architecture/evidence/gea2-tolerance-amendment.md",
            "pre_timing": "warmups=0, repetitions=1, block_steady_state unmeasured — the amendment lands before any timing (delivery §5: no tolerance widening after observation; a pre-timing numerical amendment only)",
            "mechanism": "deterministic tiled-vs-serial F32 accumulation-order difference (8x8-tile partial-sum order vs the scalar oracle's left-to-right serial dot); the U6b diagnostic run reproduced receipt v5's block metrics bit-identically",
            "v5_retained": "receipt v5 (commit 65c952196) stays in git history as the red justification; v5's block comparison row is carried verbatim above",
            "prose_corrections": [
                "all 89 observed rows are sub-5e-5 abs (not 89 sub-1e-5): 87 rows are sub-1e-5; gemm_down and block_output sit at 4.959e-5",
                "the 8 block_output elements >= 1e-5 are the same 8 elements as the gemm_down row's (the add is exact; the error is inherited)",
                "the U6b diagnostic's earlier diverged heuristic (abs>1e-2 || rel>1e-3) has small-magnitude blindness and tripped on noise-class rows — the amended element-wise gate replaces it"
            ],
            "watch_items": [
                "gemm_down is the amplitude leader (K=2560): single-block abs margin 10.1x; multi-block or larger-K evidence must re-derive the bound, not inherit it",
                "swiglu is the tightest family (3.815e-6 vs atol 5e-6 = 1.31x margin): deterministic, but any sigmoid/SiLU emitter change re-tests it immediately",
                "the residual zero-rows compare the add on device operands (block_output == residual1 + down, residual1 == activation_x + o_projection — bit-exact), not the end-to-end mirror reading",
                "transpose/window rows carry their producer family's bounds via the declared family mapping"
            ],
        },
        "measurements": {
            "tensor_range_read_us": measured(tensor_range_read_us),
            "tensor_range_read_bytes": measured(tensor_range_read_bytes),
            "weight_residency_us": measured(receipt.copy_in_us),
            "weight_bytes": measured(weight_bytes_total),
            "module_compile_us": module_compile_us,
            "per_program_alloc_us": per_program_alloc_us,
            "intermediate_alloc_us": unmeasured(
                "intermediate pool checkout is not separately timed by the session; count and per_program_alloc_us recorded",
            ),
            "intermediate_count": measured(89),
            "launch_sequence_us": derived(
                receipt.gpu_encode_submit_wait_us.saturating_sub(gpu_total),
                "encode+submit+wait wall minus summed per-encoder GPU timestamps",
            ),
            "gpu_body_us": measured(gpu_total),
            "per_stage_gpu_us": per_stage_gpu_us,
            "block_steady_state_us": unmeasured(
                "single cold evidence run; warm steady-state sampling is deferred with U6",
            ),
            "sync_wait_count": measured(receipt.syncs),
            "sync_wait_us": unmeasured(
                "not separately timed inside the single step-boundary wait",
            ),
            "readback_us": measured(receipt.readback_us),
            "readback_bytes": measured(output_bytes.len()),
            "warmups": measured(0),
            "repetitions": measured(1),
            "effective_bandwidth_bytes_per_s": if gpu_total == 0 {
                unmeasured("GPU timestamps rounded to zero")
            } else {
                derived(
                    weight_bytes_total as f64 * 1_000_000.0 / gpu_total as f64,
                    "declared resident weight bytes / summed gpu_body_us * 1_000_000",
                )
            },
            "e2e_us": measured(e2e_us),
        },
    });
    let parent = receipt_path.parent().expect("receipt parent");
    fs::create_dir_all(parent).expect("create receipt parent");
    fs::write(
        &receipt_path,
        serde_json::to_vec_pretty(&receipt_json).expect("serialize receipt"),
    )
    .unwrap_or_else(|error| panic!("write {}: {error}", receipt_path.display()));
    eprintln!("GEA2 real Metal block receipt: {}", receipt_path.display());

    assert!(
        policy_pass,
        "physical [8,960] block output disagrees with the scalar oracle beyond the frozen \
block_output policy: max_abs={max_abs:.3e} (≤5e-4), max_rel={max_rel:.3e} (≤2e-5), \
max_ulp={max_ulp} (≤1024), first_largest_error_index={first_index}; receipt at {}",
        receipt_path.display()
    );
}

// ---------------------------------------------------------------------------
// GEA2-U6b instrumented localization (the U5g successor's diagnostic): one
// real-Metal execution reclassifies every policy-row intermediate as an
// ObservationPoint result at its producing launch (the descriptor ABI's own
// lifetime/cadence surface; no wire or schema change) and reads them all
// back in the same run, comparing each against the scalar oracle's reference
// row to name the first launch whose output diverges (the CTO ruling's
// method — the bounded executor residual localizes at the diverging launch).
// The five transpose launches carry no policy row and are not observed.
// ---------------------------------------------------------------------------

/// The diagnostic descriptor: every kernel-produced buffer of all 64
/// launches observed at its producing launch — 89 buffers including the
/// per-head windows and the five transpose outputs (a superset of U6b's 59
/// policy rows; the windowed entries' fan-out and the key transpose are
/// exactly where the U5h layout truth lives).
fn gea2_diagnostic_descriptor() -> DeviceDescriptor {
    let artifact_dir = gea2_artifact_dir();
    let envelope = load_gea2_plan(&artifact_dir);
    let mut descriptor = gea2_sequence_descriptor();
    // The observed set comes from the wire's access facts (the descriptor
    // slot model carries the buffer's storage role, not the per-slot access).
    let mut observed: BTreeMap<u32, (u32, u32)> = BTreeMap::new();
    for (index, kernel) in envelope.program.kernels.iter().enumerate() {
        let launch_id = u32::try_from(index + 1).expect("launch id fits u32");
        for resource in &kernel.resources {
            if matches!(
                resource.access,
                Gea2ResourceAccess::Write | Gea2ResourceAccess::ReadWrite
            ) {
                observed.insert(resource.buffer.id, (resource.version.version, launch_id));
            }
        }
    }
    // One lifetime per buffer across every slot that references it (the
    // descriptor's lifetime rule), so consumers of an observed intermediate
    // see the same observation-point class as its producer.
    for kernel in descriptor.kernels.iter_mut() {
        for slot in kernel.buffers.iter_mut() {
            if observed.contains_key(&slot.buffer_id) {
                slot.lifetime = DeviceBufferLifetime::ObservationPoint;
            }
        }
    }
    descriptor.results = observed
        .iter()
        .map(|(&buffer_id, &(version, launch_id))| DescriptorResult {
            buffer_id,
            version,
            produced_by: launch_id,
            at_launch: launch_id,
        })
        .collect();
    descriptor
        .validate()
        .expect("the diagnostic descriptor validates (84 observed rows)");
    descriptor
}

/// The reference row for every observed buffer, keyed by buffer name
/// (resolved through the descriptor so the plan's buffer ids stay the wire's
/// facts, never hardcoded here).
fn gea2_diagnostic_reference_rows(
    descriptor: &DeviceDescriptor,
    inputs: &Gea2ScalarBlockInputs,
) -> BTreeMap<u32, Vec<f32>> {
    let rows = oracle_scalar_block_rows(inputs);
    let mut named: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    named.insert("ln1".to_owned(), rows.ln1);
    named.insert("q".to_owned(), rows.q);
    named.insert("k".to_owned(), rows.k);
    named.insert("v".to_owned(), rows.v);
    named.insert("rope_q".to_owned(), rows.rope_q);
    named.insert("rope_k".to_owned(), rows.rope_k);
    named.insert("o_projection".to_owned(), rows.o_projection);
    named.insert("residual1".to_owned(), rows.residual1);
    named.insert("ln2".to_owned(), rows.ln2);
    named.insert("gate".to_owned(), rows.gate);
    named.insert("up".to_owned(), rows.up);
    named.insert("swiglu".to_owned(), rows.swiglu);
    named.insert("down".to_owned(), rows.down);
    named.insert("block_output".to_owned(), rows.block_output);
    for (head, row) in rows.q_head.iter().enumerate() {
        named.insert(format!("q_head_{head}"), row.clone());
    }
    for (head, row) in rows.k_head.iter().enumerate() {
        named.insert(format!("k_head_{head}"), row.clone());
    }
    for (head, row) in rows.v_head.iter().enumerate() {
        named.insert(format!("v_head_{head}"), row.clone());
    }
    // The transpose entries produce the real [8,64]→[64,8] key transpose
    // (GEA2-U5i layout truth — the explicit device transpose, decision 4):
    // output flat `k` (row `j = k/8`, col `i = k%8`) holds `k_head[i*64+j]`.
    for (head, row) in rows.k_head.iter().enumerate() {
        let mut transposed = vec![0.0_f32; row.len()];
        for (k, value) in transposed.iter_mut().enumerate() {
            let i = k % 8;
            let j = k / 8;
            *value = row[i * 64 + j];
        }
        named.insert(format!("key_transpose_{head}"), transposed);
    }
    for (head, row) in rows.scores.iter().enumerate() {
        named.insert(format!("score_{head}"), row.clone());
    }
    for (head, row) in rows.probabilities.iter().enumerate() {
        named.insert(format!("probabilities_{head}"), row.clone());
    }
    for (head, row) in rows.contexts.iter().enumerate() {
        named.insert(format!("context_{head}"), row.clone());
    }

    let mut reference = BTreeMap::new();
    for result in &descriptor.results {
        let name = descriptor
            .kernels
            .iter()
            .flat_map(|kernel| kernel.buffers.iter())
            .find(|slot| slot.buffer_id == result.buffer_id && slot.version == result.version)
            .map(|slot| slot.buffer_name.as_str())
            .expect("observed buffer has a name");
        let row = named
            .get(name)
            .unwrap_or_else(|| panic!("no oracle reference row for `{name}`"));
        assert_eq!(
            row.len(),
            descriptor
                .buffer_versions
                .iter()
                .find(|version| {
                    version.buffer_id == result.buffer_id && version.version == result.version
                })
                .expect("observed buffer has version metadata")
                .element_count as usize,
            "reference row `{name}` has the declared element count"
        );
        reference.insert(result.buffer_id, row.clone());
    }
    reference
}

fn gea2_row_metrics(expected: &[f32], observed: &[f32]) -> (f32, f32, u32) {
    assert_eq!(expected.len(), observed.len());
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut max_ulp = 0_u32;
    for (&want, &got) in expected.iter().zip(observed) {
        let absolute = (got - want).abs();
        max_abs = max_abs.max(absolute);
        let denominator = want.abs().max(got.abs()).max(f32::MIN_POSITIVE);
        max_rel = max_rel.max(absolute / denominator);
        max_ulp = max_ulp.max(ulp_distance(want, got));
    }
    (max_abs, max_rel, max_ulp)
}

/// One frozen per-family tolerance (the U1-frozen table, mirror of the
/// radix oracle's `FROZEN_TOLERANCES`; constants byte-identical).
#[derive(Debug, Clone, Copy)]
struct Gea2FrozenTolerance {
    family: &'static str,
    atol: f32,
    rtol: f32,
    ulp: u32,
}

/// GEA2-U5j: the 18-row frozen family table — constants unchanged, only the
/// composition changes (aggregate-conjunction-of-maxima → element-wise
/// disjunction, CTO ruling f51387e4).
const GEA2_FROZEN_TOLERANCES: [Gea2FrozenTolerance; 18] = [
    Gea2FrozenTolerance {
        family: "rmsnorm_pre_attention",
        atol: 2.0e-5,
        rtol: 2.0e-5,
        ulp: 256,
    },
    Gea2FrozenTolerance {
        family: "q_projection_gemm",
        atol: 1.0e-4,
        rtol: 1.0e-5,
        ulp: 512,
    },
    Gea2FrozenTolerance {
        family: "k_projection_gemm",
        atol: 1.0e-4,
        rtol: 1.0e-5,
        ulp: 512,
    },
    Gea2FrozenTolerance {
        family: "v_projection_gemm",
        atol: 1.0e-4,
        rtol: 1.0e-5,
        ulp: 512,
    },
    Gea2FrozenTolerance {
        family: "rope_q",
        atol: 2.0e-6,
        rtol: 2.0e-6,
        ulp: 64,
    },
    Gea2FrozenTolerance {
        family: "rope_k",
        atol: 2.0e-6,
        rtol: 2.0e-6,
        ulp: 64,
    },
    Gea2FrozenTolerance {
        family: "score_gemm",
        atol: 2.0e-5,
        rtol: 1.0e-5,
        ulp: 256,
    },
    Gea2FrozenTolerance {
        family: "causal_softmax",
        atol: 2.0e-6,
        rtol: 2.0e-6,
        ulp: 128,
    },
    Gea2FrozenTolerance {
        family: "context_gemm",
        atol: 2.0e-5,
        rtol: 1.0e-5,
        ulp: 256,
    },
    Gea2FrozenTolerance {
        family: "o_projection_gemm",
        atol: 1.0e-4,
        rtol: 1.0e-5,
        ulp: 512,
    },
    Gea2FrozenTolerance {
        family: "residual_1",
        atol: 0.0,
        rtol: 0.0,
        ulp: 0,
    },
    Gea2FrozenTolerance {
        family: "rmsnorm_post_attention",
        atol: 2.0e-5,
        rtol: 2.0e-5,
        ulp: 256,
    },
    Gea2FrozenTolerance {
        family: "gate_projection_gemm",
        atol: 1.0e-4,
        rtol: 1.0e-5,
        ulp: 512,
    },
    Gea2FrozenTolerance {
        family: "up_projection_gemm",
        atol: 1.0e-4,
        rtol: 1.0e-5,
        ulp: 512,
    },
    Gea2FrozenTolerance {
        family: "swiglu",
        atol: 5.0e-6,
        rtol: 5.0e-6,
        ulp: 128,
    },
    Gea2FrozenTolerance {
        family: "down_projection_gemm",
        atol: 1.0e-4,
        rtol: 1.0e-5,
        ulp: 512,
    },
    Gea2FrozenTolerance {
        family: "residual_2",
        atol: 0.0,
        rtol: 0.0,
        ulp: 0,
    },
    Gea2FrozenTolerance {
        family: "block_output",
        atol: 5.0e-4,
        rtol: 2.0e-5,
        ulp: 1024,
    },
];

fn gea2_frozen_tolerance(family: &str) -> Gea2FrozenTolerance {
    GEA2_FROZEN_TOLERANCES
        .iter()
        .copied()
        .find(|tolerance| tolerance.family == family)
        .unwrap_or_else(|| panic!("no frozen GEA2 family `{family}`"))
}

/// GEA2-U5j companion (ii): the declared family mapping for every observed
/// buffer. Per-head windows inherit their producer family's bounds; the
/// transpose outputs are a declared permutation (geometry move) of the
/// rope_k head window and carry the rope_k family's bounds. The residual
/// buffers are compared on device operands, not end-to-end (companion (i)).
fn gea2_diagnostic_family(buffer_name: &str) -> &'static str {
    if buffer_name.starts_with("key_transpose_") {
        "rope_k" // declared transpose permutation class (geometry move)
    } else if buffer_name.starts_with("q_head_") {
        "rope_q"
    } else if buffer_name.starts_with("k_head_") {
        "rope_k"
    } else if buffer_name.starts_with("v_head_") {
        "v_projection_gemm"
    } else if buffer_name.starts_with("score_") {
        "score_gemm"
    } else if buffer_name.starts_with("probabilities_") {
        "causal_softmax"
    } else if buffer_name.starts_with("context_") {
        "context_gemm"
    } else {
        match buffer_name {
            "ln1" => "rmsnorm_pre_attention",
            "q" => "q_projection_gemm",
            "k" => "k_projection_gemm",
            "v" => "v_projection_gemm",
            "rope_q" => "rope_q",
            "rope_k" => "rope_k",
            "o_projection" => "o_projection_gemm",
            "residual1" => "residual_1",
            "ln2" => "rmsnorm_post_attention",
            "gate" => "gate_projection_gemm",
            "up" => "up_projection_gemm",
            "swiglu" => "swiglu",
            "down" => "down_projection_gemm",
            "block_output" => "block_output",
            other => panic!("no declared GEA2 family for `{other}`"),
        }
    }
}

/// GEA2-U5j element-wise frozen-policy verdict (CTO ruling f51387e4): a row
/// passes iff EVERY element satisfies `abs(e) <= atol OR (rel(e) <= rtol
/// AND ulp(e) <= ulp_row)`. Every frozen constant is unchanged; only the
/// composition changes from the aggregate-conjunction-of-maxima form. The
/// ulp bound stays scoped to the rel branch (load-binding for rmsnorm).
/// Returns `(pass, failing_elements)`.
fn gea2_element_wise_passes(
    expected: &[f32],
    observed: &[f32],
    atol: f32,
    rtol: f32,
    ulp_row: u32,
) -> (bool, usize) {
    assert_eq!(expected.len(), observed.len());
    let mut failing = 0usize;
    for (&want, &got) in expected.iter().zip(observed) {
        let absolute = (got - want).abs();
        let denominator = want.abs().max(got.abs()).max(f32::MIN_POSITIVE);
        let relative = absolute / denominator;
        let ulps = ulp_distance(want, got);
        let passes = absolute <= atol || (relative <= rtol && ulps <= ulp_row);
        if !passes {
            failing += 1;
        }
    }
    (failing == 0, failing)
}

/// GEA2-U5j red-green proof (hosts side of the amendment record): the
/// element-wise disjunction passes the deterministic v5 accumulation-order
/// noise class (a 2.1e-5-magnitude element at abs 2.4e-7, and the
/// amplitude leader at |expected| 62.71 at abs 4.959e-5 — receipt v5's
/// exact block metrics) while still failing a semantic-scale error and the
/// historical transposed-weight class. The aggregate conjunction this
/// amendment replaces failed the same green case (max_rel 1.13e-2,
/// max_ulp 131072).
#[test]
fn gea2_amended_block_gate_passes_v5_noise_and_fails_semantic_errors() {
    let expected = [
        -1.049_453_3_f32,
        2.104_044_0e-5_f32,
        -62.708_534_f32,
        1.0_f32,
        -0.5_f32,
    ];
    let v5_noise = [
        -1.049_453_3_f32,
        2.080_202_0e-5_f32, // abs 2.384e-7 on a 2.1e-5 element
        -62.708_584_f32,    // abs 4.959e-5 on a 62.7 element
        1.0_f32,
        -0.5_f32,
    ];
    let (passes, failing) = gea2_element_wise_passes(&expected, &v5_noise, 5e-4, 2e-5, 1024);
    assert!(
        passes,
        "the v5 noise class must pass the amended block gate"
    );
    assert_eq!(failing, 0);
    // The old aggregate conjunction fails the same row (max_rel > rtol).
    let (max_abs, max_rel, max_ulp) = gea2_row_metrics(&expected, &v5_noise);
    assert!(max_abs <= 5e-4, "the abs channel governs the noise class");
    assert!(
        max_rel > 2e-5 && max_ulp > 1024,
        "the rel/ulp channels are the old failures"
    );
    // Semantic-scale errors fail.
    let mut semantic = expected;
    semantic[0] += 0.5;
    assert!(
        !gea2_element_wise_passes(&expected, &semantic, 5e-4, 2e-5, 1024).0,
        "a ~5e-1-scale element error must fail the amended gate"
    );
    let mut transposed_class = expected;
    transposed_class[0] = -5.835_915_6_f32;
    assert!(
        !gea2_element_wise_passes(&expected, &transposed_class, 5e-4, 2e-5, 1024).0,
        "the pre-U5g transposed-weight class must fail the amended gate"
    );
}

/// The instrumented physical localization run: reads back every policy-row
/// intermediate at its producing launch in one real-Metal execution and
/// writes the raw device-rows artifact (schema `gea2-device-rows-v1`: F32 LE
/// values + launch id + entry name — the device side never sees the policy)
/// under the artifact dir via `GEA2_DEVICE_ROWS` (uncommitted; Mind
/// integrates). The comparison against the scalar mirror's reference rows
/// names the first diverging launch.
#[test]
#[ignore = "physical Metal gate; run only with GEA2_ARTIFACT_DIR + GEA2_DEVICE_ROWS set"]
fn gea2_real_metal_diagnostic_rows() {
    let workspace = workspace_root();
    let manifest_path = workspace
        .join("radix/docs/factory/gpu-execution-architecture/evidence/gea2-input-manifest.json");
    let manifest: Value = serde_json::from_slice(
        &fs::read(&manifest_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", manifest_path.display())),
    )
    .expect("valid GEA2 U1 input manifest");
    let (inputs, _, _) = gea2_scalar_block_inputs(&manifest);

    let descriptor = gea2_diagnostic_descriptor();
    assert_eq!(
        descriptor.results.len(),
        89,
        "89 kernel-produced buffers observed across all 64 launches (58 policy intermediates + the block output + 25 per-head windows + 5 transpose outputs)"
    );
    let reference = gea2_diagnostic_reference_rows(&descriptor, &inputs);
    assert_eq!(reference.len(), 89, "every observed row has a reference");

    let runtime =
        DeviceRuntime::Metal(MetalHostSession::try_open().expect("physical Metal admission"));
    let mut host =
        CompositeHost::with_device(runtime, "metal-device").expect("real-metal composite");
    let mut session = host
        .create_program_session(&descriptor)
        .expect("diagnostic GEA2 program session");
    let uploads = gea2_real_host_inputs(&descriptor, &inputs);
    let receipt = session
        .execute_with_weight_bytes(&BTreeMap::new(), &uploads)
        .expect("the 64-launch GEA2 block executes on physical Metal");
    session.teardown().expect("ordered diagnostic teardown");
    assert_eq!(receipt.outputs.len(), 89, "every observed row is read back");

    // Per-launch comparison in launch order under the GEA2-U5j amended
    // gate: each observed buffer maps to its declared frozen family (the
    // per-head windows and the transpose outputs carry their producer
    // family's bounds — companion (ii)), and every element must satisfy
    // `abs(e) <= atol OR (rel(e) <= rtol AND ulp(e) <= ulp_row)`. The
    // residual_1 buffer is compared on device operands, not end-to-end
    // (companion (i) — the end-to-end reading is a false gate for a correct
    // device once producers carry accumulation-order noise). The first
    // launch with a failing element is the localization.
    let mut rows_out: Vec<Value> = Vec::new();
    let mut first_divergent: Option<(u32, String, f32, f32)> = None;
    let mut total_failing = 0usize;
    for result in &descriptor.results {
        let launch = descriptor
            .launches
            .iter()
            .find(|launch| launch.id == result.produced_by)
            .expect("observed result names its producing launch");
        let kernel = &descriptor.kernels[launch.kernel_index as usize];
        let slot = descriptor
            .kernels
            .iter()
            .flat_map(|kernel| kernel.buffers.iter())
            .find(|slot| slot.buffer_id == result.buffer_id && slot.version == result.version)
            .expect("observed buffer has a descriptor slot");
        let family = gea2_diagnostic_family(&slot.buffer_name);
        let expected = &reference[&result.buffer_id];
        let observed = &receipt.outputs[&result.buffer_id];
        let (max_abs, max_rel, max_ulp) = gea2_row_metrics(expected, observed);
        if family == "residual_1" {
            // Companion (i): the residual add is verified bit-exact on the
            // device operands below; the end-to-end reading of the 0/0/0
            // residual family is the false gate the U5j amendment replaces.
            rows_out.push(json!({
                "launch_id": launch.id,
                "entry": kernel.entry,
                "buffer": slot.buffer_name,
                "buffer_id": slot.buffer_id,
                "elements": slot.element_count,
                "family": family,
                "comparison": "device-operand-add",
                "max_absolute_error": max_abs,
                "max_relative_error": max_rel,
                "max_ulp_distance": max_ulp,
                "failing_elements": 0,
                "diverged": false,
                "observed_f32": observed,
                "expected_f32": expected,
            }));
            continue;
        }
        let tolerance = gea2_frozen_tolerance(family);
        let (passes, failing) = gea2_element_wise_passes(
            expected,
            observed,
            tolerance.atol,
            tolerance.rtol,
            tolerance.ulp,
        );
        total_failing += failing;
        rows_out.push(json!({
            "launch_id": launch.id,
            "entry": kernel.entry,
            "buffer": slot.buffer_name,
            "buffer_id": slot.buffer_id,
            "elements": slot.element_count,
            "family": family,
            "comparison": "end-to-end-element-wise",
            "max_absolute_error": max_abs,
            "max_relative_error": max_rel,
            "max_ulp_distance": max_ulp,
            "failing_elements": failing,
            "diverged": !passes,
            "observed_f32": observed,
            "expected_f32": expected,
        }));
        if !passes && first_divergent.is_none() {
            first_divergent = Some((
                launch.id,
                format!("{} ({})", kernel.entry, slot.buffer_name),
                max_abs,
                max_rel,
            ));
        }
    }
    assert!(
        rows_out.len() == 89,
        "all 89 observed rows were compared (got {})",
        rows_out.len()
    );

    // Companion (i) — the residual zero-rows compare the ADD on device
    // operands (device_out == F32(device_in_a + device_in_b), bitwise).
    // Launch 64: block_output == residual1 + down; launch 58: residual1 ==
    // F32(activation_x + o_projection) — activation_x is the host-provided
    // input, so the pairing is the frozen `inputs.x` + device o_projection.
    let buffer_by_name = |name: &str| -> u32 {
        descriptor
            .kernels
            .iter()
            .flat_map(|kernel| kernel.buffers.iter())
            .find(|slot| slot.buffer_name == name)
            .unwrap_or_else(|| panic!("no descriptor slot named `{name}`"))
            .buffer_id
    };
    let block_output = &receipt.outputs[&buffer_by_name("block_output")];
    let residual1 = &receipt.outputs[&buffer_by_name("residual1")];
    let down = &receipt.outputs[&buffer_by_name("down")];
    let o_projection = &receipt.outputs[&buffer_by_name("o_projection")];
    let add64_exact = block_output
        .iter()
        .zip(residual1.iter().zip(down.iter()))
        .all(|(&out, (&a, &b))| out == a + b);
    assert!(
        add64_exact,
        "launch 64 block_output must equal F32(residual1 + down) bitwise on device operands"
    );
    let add58_exact = residual1
        .iter()
        .zip(inputs.x.iter().zip(o_projection.iter()))
        .all(|(&out, (&a, &b))| out == a + b);
    assert!(
        add58_exact,
        "launch 58 residual1 must equal F32(activation_x + o_projection) bitwise on device operands"
    );
    let residual_rows = json!([
        {
            "launch_id": 58,
            "entry": "residual_add",
            "buffer": "residual1",
            "elements": 7680,
            "family": "residual_1",
            "comparison": "device-operand-add",
            "bit_exact": add58_exact,
            "operands": ["activation_x", "o_projection"],
        },
        {
            "launch_id": 64,
            "entry": "residual_add",
            "buffer": "block_output",
            "elements": 7680,
            "family": "residual_2",
            "comparison": "device-operand-add",
            "bit_exact": add64_exact,
            "operands": ["residual1", "down"],
        },
    ]);

    let artifact_path = match std::env::var_os("GEA2_DEVICE_ROWS") {
        Some(path) => PathBuf::from(path),
        None => gea2_artifact_dir().join("gea2-device-rows.json"),
    };
    let artifact = json!({
        "schema": "gea2-device-rows-v1",
        "delivery": "GEA2-U6b",
        "backend": "Metal",
        "revisions": {
            "gradus": git_revision(&workspace.join("gradus")),
            "radix": git_revision(&workspace.join("radix")),
            "hosts": git_revision(&workspace.join("hosts")),
        },
        "policy": {
            "composition": "element-wise disjunction (GEA2-U5j, CTO ruling f51387e4): abs(e) <= atol OR (rel(e) <= rtol AND ulp(e) <= ulp_row); every frozen constant unchanged",
            "family_bounds": "the 18-row frozen table; per-head windows and the transpose outputs map to their producer family (declared mapping, companion ii)",
        },
        "failing_elements_total": total_failing,
        "residual_device_add": residual_rows,
        "launches": rows_out,
    });
    let parent = artifact_path.parent().expect("artifact parent");
    fs::create_dir_all(parent).expect("create artifact parent");
    fs::write(
        &artifact_path,
        serde_json::to_vec_pretty(&artifact).expect("serialize device rows"),
    )
    .unwrap_or_else(|error| panic!("write {}: {error}", artifact_path.display()));
    eprintln!("GEA2 device rows: {}", artifact_path.display());

    match first_divergent {
        Some((launch_id, entry, max_abs, max_rel)) => panic!(
            "localization: launch {launch_id} ({entry}) diverges from the scalar oracle under \
the amended element-wise gate — max_abs={max_abs:.3e}, max_rel={max_rel:.3e}; device rows at {}",
            artifact_path.display()
        ),
        None => {
            let worst = rows_out
                .iter()
                .max_by(|a, b| {
                    a["max_absolute_error"]
                        .as_f64()
                        .unwrap_or(0.0)
                        .partial_cmp(&b["max_absolute_error"].as_f64().unwrap_or(0.0))
                        .expect("comparable")
                })
                .expect("non-empty rows");
            assert!(
                total_failing == 0,
                "the amended element-wise gate must pass every element (got {total_failing} failing)"
            );
            assert!(
                add64_exact && add58_exact,
                "the residual device-operand adds must be bit-exact"
            );
            eprintln!(
                "GEA2 diagnostic: no launch diverges under the amended gate (worst row {} max_abs={})",
                worst["entry"], worst["max_absolute_error"]
            );
        }
    }
}
