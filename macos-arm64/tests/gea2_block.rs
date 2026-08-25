//! GEA2-U5b: host edge consumption — mirror-parse the exported program plan,
//! map it onto a [`DeviceDescriptor`], and admit it fail-closed.
//!
//! The bundle's `gea2-program-plan.json` member (envelope schema
//! `gea2-program-plan-v1`) carries the radix `WireDeviceProgram` in its native
//! serde JSON form, instance-expanded (64 kernels/launches, 63 dependency
//! edges, roots `[1]`). Hosts owns the consumption half of that ABI: these
//! mirror structs parse the envelope (unknown/missing fields fail closed),
//! [`map_envelope_to_descriptor`] resolves the wire's per-slot bound shapes
//! onto the host descriptor's version-keyed shape table while carrying
//! role/lifetime/initialization verbatim, and
//! [`DeviceDescriptor::validate`] runs before any launch. Neither side infers
//! the other's facts (GEA2 seam-lowering transport decision).
#![allow(dead_code)] // the mirror fields are the fail-closed decode contract; serde reads the full wire shape while the mapper consumes the mapped subset

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use faber_host_macos_arm64::device_descriptor::{
    DescriptorBuffer, DescriptorBufferVersion, DescriptorDataFlow, DescriptorEndOfRunResult,
    DescriptorKernel, DescriptorLaunch, DescriptorResult, DeviceBufferInitialization,
    DeviceBufferLifetime, DeviceBufferRole, DeviceDataType, DeviceDescriptor,
    DeviceProgramLifetime, E_DEVICE_DESCRIPTOR,
};
use host_coordinator::DeviceBackend;
use serde::Deserialize;
use serde_json::{json, Value};

const PLAN_ENVELOPE_SCHEMA: &str = "gea2-program-plan-v1";
const PLAN_MEMBER: &str = "gea2-program-plan.json";
const MODULE_IMAGE_RULE: &str =
    "module_image is the concatenation of module_members in listed order";

/// The 13-entry block kernel table and its instance counts (GEA2 §5 / U5a
/// admission facts, mirrored here for consumption admission).
const GEA2_ENTRY_TABLE: [(&str, usize); 13] = [
    ("rmsnorm", 2),
    ("gemm_qo", 2),
    ("gemm_kv", 2),
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
    tile: u32,
    workgroup_x: u32,
    workgroup_y: u32,
    shared_memory: Gea2MatMulSharedMemory,
    barriers: Vec<Gea2BarrierPoint>,
    oob_padding: Gea2OobPaddingPolicy,
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
            format!("module member `{}` is missing from the exported bundle: {error}", path.display())
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
    if program.dependencies.len() != 63 {
        return Err(format!(
            "missing dependency edge: {} present, 63 expected",
            program.dependencies.len()
        ));
    }
    if program.roots != vec![1] {
        return Err(format!("GEA2 roots must be exactly [1], got {:?}", program.roots));
    }
    for (index, launch) in program.launches.iter().enumerate() {
        let expected_id = u32::try_from(index + 1).expect("launch id fits u32");
        let expected_kernel = u32::try_from(index).expect("kernel index fits u32");
        if launch.id != expected_id || launch.kernel_index != expected_kernel {
            return Err(format!("launch {expected_id} is not instance-expanded"));
        }
    }
    check_entry_table(&program.kernels)?;

    // The wire carries per-slot bound shapes: a kernel may bind a window of a
    // buffer (score_gemm binds a 512-element window of the 7680-element
    // rope_q). The host descriptor's keyed metadata carries ONE shape per
    // (buffer_id, version), so the mapper resolves each key to the buffer's
    // full shape (the largest bound count) and admits window reads
    // fail-closed as views of that shape.
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
        .map(|((buffer_id, version), (element_ty, element_count))| DescriptorBufferVersion {
            buffer_id,
            version,
            element_ty,
            element_count,
        })
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
                return Err("GEA2 plan declares a repeating-step lifetime with a zero step count"
                    .to_owned());
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
            "GEA2 plan carries {} distinct entries; the 13-entry block table expects {}",
            counts.len(),
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
        "gemm_qo" | "gemm_kv" | "gemm_gate_up" | "gemm_down" | "score_gemm" | "context_gemm" => {
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
    let bytes = fs::read(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
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
    assert_eq!(descriptor.data_flow.len(), 63);
    assert_eq!(descriptor.roots, vec![1]);
    assert_eq!(descriptor.buffer_versions.len(), 75, "75 distinct buffer version keys");
    assert_eq!(descriptor.results.len(), 1);
    assert_eq!(descriptor.end_of_run_results.len(), 0);
    assert!(!descriptor.module_image.is_empty());
    assert_eq!(
        descriptor.launches.iter().map(|launch| launch.id).collect::<Vec<_>>(),
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
    assert_eq!(weight_ids.len(), 11, "eleven block tensors are per-program inputs");
    for weight in weight_ids {
        let slots = descriptor
            .kernels
            .iter()
            .flat_map(|kernel| kernel.buffers.iter())
            .filter(|slot| slot.buffer_id == weight);
        for slot in slots {
            assert_eq!(slot.role, DeviceBufferRole::Input);
            assert_eq!(slot.initialization, DeviceBufferInitialization::HostProvided);
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
    assert!(error.contains("64"), "instance expansion must fail closed: {error}");

    // A missing dependency edge fails the mapper closed.
    let mut missing_edge = value.clone();
    missing_edge["program"]["dependencies"]
        .as_array_mut()
        .expect("dependencies array")
        .pop();
    let parsed: Gea2ProgramPlanEnvelope =
        serde_json::from_value(missing_edge).expect("62-edge plan still mirrors");
    let error = map_envelope_to_descriptor(&parsed, &artifact_dir)
        .expect_err("a 62-edge plan must fail closed");
    assert!(error.contains("63"), "edge count must fail closed: {error}");

    // A module member absent from the bundle fails the assembly closed.
    let mut missing_member = value.clone();
    missing_member["module_members"][0] = json!("absent.metal");
    let parsed: Gea2ProgramPlanEnvelope =
        serde_json::from_value(missing_member).expect("member list still mirrors");
    let error = map_envelope_to_descriptor(&parsed, &artifact_dir)
        .expect_err("a missing module member must fail closed");
    assert!(error.contains("absent.metal"), "member rejection names the member: {error}");
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
    descriptor.validate().expect("the mapped descriptor validates");

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
