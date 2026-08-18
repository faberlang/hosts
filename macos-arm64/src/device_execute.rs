//! CLI device-descriptor execute surface.
//!
//! The command `faber-host-macos-arm64 device-execute` is the packed device
//! run exposed over files + JSON (the same `construct_composite_host` /
//! `create_program_session` / `session.execute` path the composite host
//! already owns). The wire is the CLI arg surface:
//!
//! ```text
//! faber-host-macos-arm64 device-execute \
//!   --backend metal|cuda|auto \
//!   --descriptor <descriptor.json> \
//!   --module <module.bin> \
//!   --inputs <inputs.json>
//! ```
//!
//! `--backend` is optional; when omitted the descriptor's `backend` field
//! is the explicit selection. The module image is a raw file (not encoded
//! in the descriptor JSON). Inputs are `{ "<buffer-id>": [f32, ...] }`.
//!
//! Success prints a receipt JSON and exits 0. A host failure prints a
//! [`HostError`] JSON and exits 2. Usage / parse failures exit 64.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use host_coordinator::DeviceBackend;
use serde::{Deserialize, Serialize};

use crate::composite_host::{
    CompositeHost, CompositeHostConfig, DeviceExecutionReceipt, DeviceSelection,
};
use crate::device_descriptor::{
    DescriptorBuffer, DescriptorBufferVersion, DescriptorDataFlow, DescriptorEndOfRunResult,
    DescriptorKernel, DescriptorLaunch, DescriptorResult, DeviceBufferInitialization,
    DeviceBufferLifetime, DeviceBufferRole, DeviceDataType, DeviceDescriptor,
    DeviceProgramLifetime,
};
use crate::kernel::{HostError, HostResult};

/// CLI flags for `device-execute`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceExecuteArgs {
    /// Optional selection override (`auto` / `metal` / `cuda`).
    pub backend: Option<DeviceSelection>,
    /// Descriptor JSON path (no module image).
    pub descriptor: PathBuf,
    /// Raw module-image path (MSL source or PTX).
    pub module: PathBuf,
    /// Inputs JSON path (`{ "<id>": [f32, ...] }`).
    pub inputs: PathBuf,
}

/// Parse `device-execute` CLI flags. Unknown or missing flags are usage
/// errors (the caller maps them to exit 64).
pub fn parse_device_execute_args(args: &[String]) -> Result<DeviceExecuteArgs, String> {
    let mut backend = None;
    let mut descriptor = None;
    let mut module = None;
    let mut inputs = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--backend" => {
                let value = next_flag_value(args, &mut index, "--backend")?;
                backend = Some(parse_selection(&value)?);
            }
            "--descriptor" => {
                let value = next_flag_value(args, &mut index, "--descriptor")?;
                descriptor = Some(PathBuf::from(value));
            }
            "--module" => {
                let value = next_flag_value(args, &mut index, "--module")?;
                module = Some(PathBuf::from(value));
            }
            "--inputs" => {
                let value = next_flag_value(args, &mut index, "--inputs")?;
                inputs = Some(PathBuf::from(value));
            }
            other => return Err(format!("unknown device-execute argument: {other}")),
        }
        index += 1;
    }
    Ok(DeviceExecuteArgs {
        backend,
        descriptor: descriptor.ok_or_else(|| usage_text().to_owned())?,
        module: module.ok_or_else(|| usage_text().to_owned())?,
        inputs: inputs.ok_or_else(|| usage_text().to_owned())?,
    })
}

/// Usage line for the command.
#[must_use]
pub fn usage_text() -> &'static str {
    "usage: faber-host-macos-arm64 device-execute [--backend auto|metal|cuda] --descriptor <json> --module <bin> --inputs <json>"
}

/// Load files, validate the descriptor, construct the composite host, and
/// execute one packed device run.
pub fn run_device_execute(args: &DeviceExecuteArgs) -> HostResult<DeviceExecuteReceipt> {
    let descriptor_bytes = read_file(&args.descriptor)?;
    let module_image = read_file(&args.module)?;
    let inputs_bytes = read_file(&args.inputs)?;
    let descriptor = descriptor_from_json(&descriptor_bytes, module_image)?;
    let inputs = inputs_from_json(&inputs_bytes)?;
    descriptor.validate()?;
    let selection = args
        .backend
        .unwrap_or_else(|| selection_for_backend(descriptor.backend));
    let mut host = CompositeHost::new(CompositeHostConfig {
        selection,
        requires_device: true,
    })?;
    let mut session = host.create_program_session(&descriptor)?;
    let receipt = session.execute(&inputs)?;
    session.teardown()?;
    Ok(DeviceExecuteReceipt::from_host(&receipt))
}

/// Decode a descriptor JSON plus a raw module image.
pub fn descriptor_from_json(bytes: &[u8], module_image: Vec<u8>) -> HostResult<DeviceDescriptor> {
    let wire: WireDescriptor = serde_json::from_slice(bytes).map_err(|error| {
        HostError::invalid_args(format!(
            "device-execute descriptor JSON is invalid: {error}"
        ))
    })?;
    wire.into_descriptor(module_image)
}

/// Encode a descriptor as the CLI JSON (module image omitted).
pub fn descriptor_to_json(descriptor: &DeviceDescriptor) -> HostResult<Vec<u8>> {
    let wire = WireDescriptor::from_descriptor(descriptor);
    serde_json::to_vec_pretty(&wire).map_err(|error| {
        HostError::internal(format!(
            "device-execute failed to encode descriptor: {error}"
        ))
    })
}

/// Decode `{ "<buffer-id>": [f32, ...] }`.
///
/// Finite values are JSON numbers (`f32` → `f64` is injective). NaN
/// payloads are `"0x"` + 8 hex bits so packed GGUF words survive the
/// file; `"NaN"` still decodes as the canonical quiet NaN. Infinities
/// stay `"Infinity"` / `"-Infinity"`.
pub fn inputs_from_json(bytes: &[u8]) -> HostResult<BTreeMap<u32, Vec<f32>>> {
    let wire: BTreeMap<String, Vec<serde_json::Value>> =
        serde_json::from_slice(bytes).map_err(|error| {
            HostError::invalid_args(format!("device-execute inputs JSON is invalid: {error}"))
        })?;
    let mut inputs = BTreeMap::new();
    for (key, values) in wire {
        let id = key.parse::<u32>().map_err(|_| {
            HostError::invalid_args(format!(
                "device-execute inputs key `{key}` is not a buffer id"
            ))
        })?;
        let mut parsed = Vec::with_capacity(values.len());
        for (index, value) in values.into_iter().enumerate() {
            parsed.push(f32_from_json(&value).map_err(|detail| {
                HostError::invalid_args(format!(
                    "device-execute inputs[{id}][{index}] is invalid: {detail}"
                ))
            })?);
        }
        inputs.insert(id, parsed);
    }
    Ok(inputs)
}

/// Encode inputs as `{ "<buffer-id>": [f32, ...] }`.
pub fn inputs_to_json(inputs: &BTreeMap<u32, Vec<f32>>) -> HostResult<Vec<u8>> {
    let wire: BTreeMap<String, Vec<serde_json::Value>> = inputs
        .iter()
        .map(|(id, values)| {
            (
                id.to_string(),
                values.iter().copied().map(f32_to_json).collect(),
            )
        })
        .collect();
    serde_json::to_vec_pretty(&wire).map_err(|error| {
        HostError::internal(format!("device-execute failed to encode inputs: {error}"))
    })
}

fn f32_to_json(value: f32) -> serde_json::Value {
    if value.is_nan() {
        serde_json::Value::String(hex_f32_bits(value))
    } else if value == f32::INFINITY {
        serde_json::Value::String("Infinity".to_owned())
    } else if value == f32::NEG_INFINITY {
        serde_json::Value::String("-Infinity".to_owned())
    } else {
        serde_json::Number::from_f64(f64::from(value))
            .map(serde_json::Value::Number)
            .unwrap_or_else(|| serde_json::Value::String(hex_f32_bits(value)))
    }
}

fn f32_from_json(value: &serde_json::Value) -> Result<f32, String> {
    match value {
        serde_json::Value::Number(number) => number
            .as_f64()
            .map(|wide| wide as f32)
            .ok_or_else(|| "number is not finite".to_owned()),
        serde_json::Value::String(spelling) => parse_f32_string(spelling),
        serde_json::Value::Null => Ok(f32::NAN),
        other => Err(format!("expected number or non-finite string, got {other}")),
    }
}

fn hex_f32_bits(value: f32) -> String {
    format!("0x{:08x}", value.to_bits())
}

fn parse_f32_string(spelling: &str) -> Result<f32, String> {
    match spelling {
        "NaN" => Ok(f32::NAN),
        "Infinity" => Ok(f32::INFINITY),
        "-Infinity" => Ok(f32::NEG_INFINITY),
        hex if hex.len() == 10 && hex.as_bytes()[..2].eq_ignore_ascii_case(b"0x") => {
            u32::from_str_radix(&hex[2..], 16)
                .map(f32::from_bits)
                .map_err(|_| format!("f32 bit string `{spelling}` is not hex"))
        }
        other => Err(format!("unknown f32 spelling `{other}`")),
    }
}

/// Encode a receipt for stdout.
pub fn receipt_to_json(receipt: &DeviceExecuteReceipt) -> HostResult<Vec<u8>> {
    serde_json::to_vec_pretty(receipt).map_err(|error| {
        HostError::internal(format!("device-execute failed to encode receipt: {error}"))
    })
}

/// CLI receipt: the facts the spawn client needs (outputs + launch counts).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceExecuteReceipt {
    /// Selected backend spelling.
    pub backend: String,
    /// Selected-hardware name.
    pub device_name: String,
    /// Launches dispatched.
    pub launches: usize,
    /// Descriptor launch identities, in order.
    pub launch_ids: Vec<u32>,
    /// Kernel entries dispatched, in order.
    pub launch_entries: Vec<String>,
    /// Host→device copy-ins.
    pub copy_ins: usize,
    /// Declared observation outputs (`buffer id` → f32 values).
    pub outputs: BTreeMap<String, Vec<f32>>,
    /// Allocated program-level buffer ids.
    pub allocated_buffers: Vec<u32>,
    /// Observed device syncs.
    pub syncs: usize,
    /// Observed transfers (copy-ins + readbacks).
    pub transfers: usize,
    /// Device→host readbacks.
    pub readbacks: usize,
    /// Program-graph SHA-256 receipt.
    pub program_graph_hash: String,
}

impl DeviceExecuteReceipt {
    /// Project the host receipt onto the CLI wire.
    #[must_use]
    pub fn from_host(receipt: &DeviceExecutionReceipt) -> Self {
        Self {
            backend: receipt.backend.spelling().to_owned(),
            device_name: receipt.device_name.clone(),
            launches: receipt.launches,
            launch_ids: receipt.launch_ids.clone(),
            launch_entries: receipt.launch_entries.clone(),
            copy_ins: receipt.copy_ins,
            outputs: receipt
                .outputs
                .iter()
                .map(|(id, values)| (id.to_string(), values.clone()))
                .collect(),
            allocated_buffers: receipt.allocated_buffers.clone(),
            syncs: receipt.syncs,
            transfers: receipt.transfers,
            readbacks: receipt.readbacks,
            program_graph_hash: receipt.program_graph_hash.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct WireDescriptor {
    backend: String,
    kernels: Vec<WireKernel>,
    launches: Vec<WireLaunch>,
    buffer_versions: Vec<WireBufferVersion>,
    program_lifetime: String,
    data_flow: Vec<WireDataFlow>,
    roots: Vec<u32>,
    results: Vec<WireResult>,
    #[serde(default)]
    end_of_run_results: Vec<WireEndOfRun>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct WireKernel {
    entry: String,
    buffers: Vec<WireBuffer>,
    grid: [u32; 3],
    block: [u32; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct WireBuffer {
    buffer_id: u32,
    buffer_name: String,
    semantic_value: u32,
    role: String,
    lifetime: String,
    initialization: String,
    binding: u32,
    element_ty: String,
    element_count: u64,
    version: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct WireLaunch {
    id: u32,
    kernel_index: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct WireBufferVersion {
    buffer_id: u32,
    version: u32,
    element_ty: String,
    element_count: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct WireDataFlow {
    buffer_id: u32,
    version: u32,
    producer: u32,
    consumer: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct WireResult {
    buffer_id: u32,
    version: u32,
    produced_by: u32,
    at_launch: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct WireEndOfRun {
    buffer_id: u32,
    version: u32,
}

impl WireDescriptor {
    fn from_descriptor(descriptor: &DeviceDescriptor) -> Self {
        Self {
            backend: descriptor.backend.spelling().to_owned(),
            kernels: descriptor
                .kernels
                .iter()
                .map(|kernel| WireKernel {
                    entry: kernel.entry.clone(),
                    buffers: kernel.buffers.iter().map(WireBuffer::from_slot).collect(),
                    grid: kernel.grid,
                    block: kernel.block,
                })
                .collect(),
            launches: descriptor
                .launches
                .iter()
                .map(|launch| WireLaunch {
                    id: launch.id,
                    kernel_index: launch.kernel_index,
                })
                .collect(),
            buffer_versions: descriptor
                .buffer_versions
                .iter()
                .map(|version| WireBufferVersion {
                    buffer_id: version.buffer_id,
                    version: version.version,
                    element_ty: version.element_ty.spelling().to_owned(),
                    element_count: version.element_count,
                })
                .collect(),
            program_lifetime: descriptor.program_lifetime.spelling().to_owned(),
            data_flow: descriptor
                .data_flow
                .iter()
                .map(|edge| WireDataFlow {
                    buffer_id: edge.buffer_id,
                    version: edge.version,
                    producer: edge.producer,
                    consumer: edge.consumer,
                })
                .collect(),
            roots: descriptor.roots.clone(),
            results: descriptor
                .results
                .iter()
                .map(|result| WireResult {
                    buffer_id: result.buffer_id,
                    version: result.version,
                    produced_by: result.produced_by,
                    at_launch: result.at_launch,
                })
                .collect(),
            end_of_run_results: descriptor
                .end_of_run_results
                .iter()
                .map(|result| WireEndOfRun {
                    buffer_id: result.buffer_id,
                    version: result.version,
                })
                .collect(),
        }
    }

    fn into_descriptor(self, module_image: Vec<u8>) -> HostResult<DeviceDescriptor> {
        let backend = DeviceBackend::from_spelling(&self.backend).ok_or_else(|| {
            HostError::invalid_args(format!(
                "device-execute descriptor backend `{}` is not metal or cuda",
                self.backend
            ))
        })?;
        let program_lifetime = DeviceProgramLifetime::from_spelling(&self.program_lifetime)
            .ok_or_else(|| {
                HostError::invalid_args(format!(
                    "device-execute descriptor program_lifetime `{}` is not single-run or repeating-step",
                    self.program_lifetime
                ))
            })?;
        let mut kernels = Vec::with_capacity(self.kernels.len());
        for kernel in self.kernels {
            let mut buffers = Vec::with_capacity(kernel.buffers.len());
            for slot in kernel.buffers {
                buffers.push(slot.into_slot()?);
            }
            kernels.push(DescriptorKernel {
                entry: kernel.entry,
                buffers,
                grid: kernel.grid,
                block: kernel.block,
            });
        }
        let mut buffer_versions = Vec::with_capacity(self.buffer_versions.len());
        for version in self.buffer_versions {
            let element_ty = parse_dtype(&version.element_ty)?;
            buffer_versions.push(DescriptorBufferVersion {
                buffer_id: version.buffer_id,
                version: version.version,
                element_ty,
                element_count: version.element_count,
            });
        }
        Ok(DeviceDescriptor {
            backend,
            module_image,
            kernels,
            launches: self
                .launches
                .into_iter()
                .map(|launch| DescriptorLaunch {
                    id: launch.id,
                    kernel_index: launch.kernel_index,
                })
                .collect(),
            buffer_versions,
            program_lifetime,
            data_flow: self
                .data_flow
                .into_iter()
                .map(|edge| DescriptorDataFlow {
                    buffer_id: edge.buffer_id,
                    version: edge.version,
                    producer: edge.producer,
                    consumer: edge.consumer,
                })
                .collect(),
            roots: self.roots,
            results: self
                .results
                .into_iter()
                .map(|result| DescriptorResult {
                    buffer_id: result.buffer_id,
                    version: result.version,
                    produced_by: result.produced_by,
                    at_launch: result.at_launch,
                })
                .collect(),
            end_of_run_results: self
                .end_of_run_results
                .into_iter()
                .map(|result| DescriptorEndOfRunResult {
                    buffer_id: result.buffer_id,
                    version: result.version,
                })
                .collect(),
        })
    }
}

impl WireBuffer {
    fn from_slot(slot: &DescriptorBuffer) -> Self {
        Self {
            buffer_id: slot.buffer_id,
            buffer_name: slot.buffer_name.clone(),
            semantic_value: slot.semantic_value,
            role: slot.role.spelling().to_owned(),
            lifetime: slot.lifetime.spelling().to_owned(),
            initialization: slot.initialization.spelling().to_owned(),
            binding: slot.binding,
            element_ty: slot.element_ty.spelling().to_owned(),
            element_count: slot.element_count,
            version: slot.version,
        }
    }

    fn into_slot(self) -> HostResult<DescriptorBuffer> {
        let role = DeviceBufferRole::from_spelling(&self.role).ok_or_else(|| {
            HostError::invalid_args(format!(
                "device-execute descriptor role `{}` is not input, output, or in-out",
                self.role
            ))
        })?;
        let lifetime = DeviceBufferLifetime::from_spelling(&self.lifetime).ok_or_else(|| {
            HostError::invalid_args(format!(
                "device-execute descriptor lifetime `{}` is not per-program, per-step, or observation-point",
                self.lifetime
            ))
        })?;
        let initialization = DeviceBufferInitialization::from_spelling(&self.initialization)
            .ok_or_else(|| {
                HostError::invalid_args(format!(
                    "device-execute descriptor initialization `{}` is not zero-fill, host-provided, or kernel-initialized",
                    self.initialization
                ))
            })?;
        Ok(DescriptorBuffer {
            buffer_id: self.buffer_id,
            buffer_name: self.buffer_name,
            semantic_value: self.semantic_value,
            role,
            lifetime,
            initialization,
            binding: self.binding,
            element_ty: parse_dtype(&self.element_ty)?,
            element_count: self.element_count,
            version: self.version,
        })
    }
}

fn parse_dtype(spelling: &str) -> HostResult<DeviceDataType> {
    DeviceDataType::from_spelling(spelling).ok_or_else(|| {
        HostError::invalid_args(format!(
            "device-execute descriptor element_ty `{spelling}` is outside the host dtype surface"
        ))
    })
}

fn parse_selection(spelling: &str) -> Result<DeviceSelection, String> {
    match spelling {
        "auto" => Ok(DeviceSelection::Auto),
        "metal" => Ok(DeviceSelection::Metal),
        "cuda" => Ok(DeviceSelection::Cuda),
        other => Err(format!(
            "device-execute --backend must be auto, metal, or cuda (got `{other}`)"
        )),
    }
}

fn selection_for_backend(backend: DeviceBackend) -> DeviceSelection {
    match backend {
        DeviceBackend::Metal => DeviceSelection::Metal,
        DeviceBackend::Cuda => DeviceSelection::Cuda,
    }
}

fn next_flag_value(args: &[String], index: &mut usize, flag: &str) -> Result<String, String> {
    let value = args
        .get(*index + 1)
        .ok_or_else(|| format!("{flag} requires a value; {}", usage_text()))?;
    *index += 1;
    Ok(value.clone())
}

fn read_file(path: &Path) -> HostResult<Vec<u8>> {
    fs::read(path).map_err(|error| {
        HostError::invalid_args(format!(
            "device-execute failed to read {}: {error}",
            path.display()
        ))
    })
}
