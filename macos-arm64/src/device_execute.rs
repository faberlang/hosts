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
//!   --inputs <inputs.json> \
//!   [--weights <model.gguf> --weight-map <map.json>]
//! ```
//!
//! `--backend` is optional; when omitted the descriptor's `backend` field
//! is the explicit selection. The module image is a raw file (not encoded
//! in the descriptor JSON). `--inputs` is `{ "<buffer-id>": [f32, ...] }`
//! for tiny per-step values (tokens, rope, synthesized tables). Packed
//! weights are not JSON: `--weights` is the GGUF file and `--weight-map`
//! names byte ranges inside it (`offset` / `len` / `elems` per buffer id).
//!
//! Success prints a receipt JSON and exits 0. A host failure prints a
//! [`HostError`] JSON and exits 2. Usage / parse failures exit 64.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use host_coordinator::DeviceBackend;
use serde::{Deserialize, Serialize};

use crate::composite_host::{
    CompositeHost, CompositeHostConfig, DeviceExecutionReceipt, DeviceSelection,
    PreparedResidentSession,
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
    /// Inputs JSON path (`{ "<id>": [f32, ...] }`) — tiny per-step values.
    pub inputs: PathBuf,
    /// Optional GGUF file the child maps for packed-weight buffers.
    pub weights: Option<PathBuf>,
    /// Optional `{ "<id>": { offset, len, elems } }` map into `--weights`.
    pub weight_map: Option<PathBuf>,
    /// Keep the host process alive and accept load/step/reset/release JSON
    /// commands on stdin. The default remains the legacy one-shot command.
    pub control: bool,
}

/// Lifecycle facts returned by the explicit resident-session control entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct DeviceExecuteLifecycle {
    /// Sessions admitted by `load`.
    pub prepares: usize,
    /// Steps executed through the admitted session.
    pub reuses: usize,
    /// Resident-state resets (zero for the M1 no-KV graph).
    pub resets: usize,
    /// Explicit releases.
    pub releases: usize,
    /// Module reloads after load.
    pub module_reloads: usize,
    /// PerProgram allocations after load.
    pub per_program_reallocs: usize,
    /// Device handles still owned by the session.
    pub live_handles: usize,
}

/// One response from the explicit resident-session control protocol.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceExecuteControlReceipt {
    /// The accepted operation (`load`, `step`, `reset`, or `release`).
    pub operation: String,
    /// Session lifecycle evidence after the operation.
    pub lifecycle: DeviceExecuteLifecycle,
    /// Number of declared physical kernel bodies in the loaded descriptor.
    #[serde(default)]
    pub kernel_count: usize,
    /// Number of state buffers cleared by `reset`.
    #[serde(default)]
    pub reset_cleared: usize,
    /// Device execution facts for a `step`; absent for control-only verbs.
    #[serde(default)]
    pub receipt: Option<DeviceExecuteReceipt>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceExecuteControlVerb {
    Load,
    Step,
    Reset,
    Release,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeviceExecuteControlRequest {
    pub verb: DeviceExecuteControlVerb,
    pub inputs: Option<BTreeMap<u32, Vec<f32>>>,
}

/// Parse `device-execute` CLI flags. Unknown or missing flags are usage
/// errors (the caller maps them to exit 64).
pub fn parse_device_execute_args(args: &[String]) -> Result<DeviceExecuteArgs, String> {
    let mut backend = None;
    let mut descriptor = None;
    let mut module = None;
    let mut inputs = None;
    let mut weights = None;
    let mut weight_map = None;
    let mut control = false;
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
            "--weights" => {
                let value = next_flag_value(args, &mut index, "--weights")?;
                weights = Some(PathBuf::from(value));
            }
            "--weight-map" => {
                let value = next_flag_value(args, &mut index, "--weight-map")?;
                weight_map = Some(PathBuf::from(value));
            }
            "--control" => control = true,
            other => return Err(format!("unknown device-execute argument: {other}")),
        }
        index += 1;
    }
    match (&weights, &weight_map) {
        (None, None) | (Some(_), Some(_)) => {}
        _ => {
            return Err(format!(
                "--weights and --weight-map must be passed together; {}",
                usage_text()
            ));
        }
    }
    Ok(DeviceExecuteArgs {
        backend,
        descriptor: descriptor.ok_or_else(|| usage_text().to_owned())?,
        module: module.ok_or_else(|| usage_text().to_owned())?,
        inputs: inputs.ok_or_else(|| usage_text().to_owned())?,
        weights,
        weight_map,
        control,
    })
}

/// Usage line for the command.
#[must_use]
pub fn usage_text() -> &'static str {
    "usage: faber-host-macos-arm64 device-execute [--control] [--backend auto|metal|cuda] --descriptor <json> --module <bin> --inputs <json> [--weights <gguf> --weight-map <json>]"
}

#[derive(Debug, Deserialize)]
struct WireControlRequest {
    op: String,
    #[serde(default)]
    inputs: Option<serde_json::Value>,
}

/// Decode one line of the resident-session control protocol.
///
/// The protocol is deliberately a local stdin/stdout stream, not HTTP. Input
/// values use the same lossless JSON spellings as the legacy `--inputs` file.
pub fn parse_control_request(bytes: &[u8]) -> HostResult<DeviceExecuteControlRequest> {
    let wire: WireControlRequest = serde_json::from_slice(bytes).map_err(|error| {
        HostError::invalid_args(format!("device-execute control JSON is invalid: {error}"))
    })?;
    let verb = match wire.op.as_str() {
        "load" => DeviceExecuteControlVerb::Load,
        "step" => DeviceExecuteControlVerb::Step,
        "reset" => DeviceExecuteControlVerb::Reset,
        "release" => DeviceExecuteControlVerb::Release,
        other => {
            return Err(HostError::invalid_args(format!(
                "device-execute control verb `{other}` is not one of load, step, reset, release"
            )))
        }
    };
    let inputs = wire
        .inputs
        .map(|value| {
            let bytes = serde_json::to_vec(&value).map_err(|error| {
                HostError::invalid_args(format!(
                    "device-execute control inputs are invalid: {error}"
                ))
            })?;
            inputs_from_json(&bytes)
        })
        .transpose()?;
    Ok(DeviceExecuteControlRequest { verb, inputs })
}

/// Load files, validate the descriptor, construct the composite host, and
/// execute one packed device run.
pub fn run_device_execute(args: &DeviceExecuteArgs) -> HostResult<DeviceExecuteReceipt> {
    let cli_started = Instant::now();
    let read_started = Instant::now();
    let descriptor_bytes = read_file(&args.descriptor)?;
    let module_image = read_file(&args.module)?;
    let inputs_bytes = read_file(&args.inputs)?;
    let weight_map_bytes = match &args.weight_map {
        Some(path) => Some(read_file(path)?),
        None => None,
    };
    let gguf_bytes = match &args.weights {
        Some(path) => Some(read_file(path)?),
        None => None,
    };
    let mut inputs = match (gguf_bytes.as_deref(), weight_map_bytes.as_deref()) {
        (Some(gguf), Some(map_json)) => {
            let map = weight_map_from_json(map_json)?;
            inputs_from_gguf(gguf, &map)?
        }
        _ => BTreeMap::new(),
    };
    let file_read_us = elapsed_us(read_started);
    let descriptor_started = Instant::now();
    let descriptor = descriptor_from_json(&descriptor_bytes, module_image)?;
    let descriptor_decode_us = elapsed_us(descriptor_started);
    let inputs_started = Instant::now();
    let json_inputs = inputs_from_json(&inputs_bytes)?;
    for (id, values) in json_inputs {
        if inputs.contains_key(&id) {
            return Err(HostError::invalid_args(format!(
                "device-execute buffer {id} is in both --weight-map and --inputs"
            )));
        }
        inputs.insert(id, values);
    }
    let json_decode_us = elapsed_us(inputs_started);
    descriptor.validate()?;
    let selection = args
        .backend
        .unwrap_or_else(|| selection_for_backend(descriptor.backend));
    let host_started = Instant::now();
    let mut host = CompositeHost::new(CompositeHostConfig {
        selection,
        requires_device: true,
    })?;
    let host_construct_us = elapsed_us(host_started);
    let session_started = Instant::now();
    let mut session = host.create_program_session(&descriptor)?;
    let session_create_us = elapsed_us(session_started);
    let load_module_us = session.load_module_us;
    let per_program_alloc_us = session.per_program_alloc_us;
    let step_started = Instant::now();
    let receipt = session.execute(&inputs)?;
    let step_wall_us = elapsed_us(step_started);
    session.teardown()?;
    let mut wire = DeviceExecuteReceipt::from_host(&receipt);
    wire.kernel_count = descriptor.kernels.len();
    wire.stage_timing = stage_timing(step_wall_us, &receipt);
    wire.file_read_us = file_read_us;
    wire.descriptor_decode_us = descriptor_decode_us;
    wire.json_decode_us = json_decode_us;
    wire.host_construct_us = host_construct_us;
    wire.session_create_us = session_create_us;
    wire.load_module_us = load_module_us;
    wire.per_program_alloc_us = per_program_alloc_us;
    wire.cli_internal_us = elapsed_us(cli_started);
    Ok(wire)
}

/// Run the explicit resident-session control stream.
///
/// The descriptor/module/weight map are read once, then the first command must
/// be `load`. The resulting prepared session stays owned by this one host
/// process until an explicit `release`; each `step` only receives invocation
/// inputs and uses the already admitted resident adapter.
pub fn run_device_execute_control(args: &DeviceExecuteArgs) -> HostResult<()> {
    let descriptor_bytes = read_file(&args.descriptor)?;
    let module_image = read_file(&args.module)?;
    let base_inputs = inputs_from_json(&read_file(&args.inputs)?)?;
    let weight_map = match &args.weight_map {
        Some(path) => weight_map_from_json(&read_file(path)?)?,
        None => BTreeMap::new(),
    };
    let gguf = match &args.weights {
        Some(path) => read_file(path)?,
        None => Vec::new(),
    };
    let weight_inputs = if args.weights.is_some() {
        inputs_from_gguf(&gguf, &weight_map)?
    } else {
        BTreeMap::new()
    };
    let descriptor = descriptor_from_json(&descriptor_bytes, module_image)?;
    descriptor.validate()?;
    let selection = args
        .backend
        .unwrap_or_else(|| selection_for_backend(descriptor.backend));

    let stdin = std::io::stdin();
    let mut lines = BufReader::new(stdin.lock()).lines();
    let first = lines
        .next()
        .transpose()
        .map_err(|error| {
            HostError::internal(format!("device-execute control read failed: {error}"))
        })?
        .ok_or_else(|| {
            HostError::invalid_args("device-execute control requires load before end of input")
        })?;
    let first = parse_control_request(first.as_bytes())?;
    if first.verb != DeviceExecuteControlVerb::Load || first.inputs.is_some() {
        return Err(HostError::invalid_args(
            "device-execute control stream must begin with {\\\"op\\\":\\\"load\\\"}",
        ));
    }

    let mut host = CompositeHost::new(CompositeHostConfig {
        selection,
        requires_device: true,
    })?;
    let mut session = host.prepare_resident_session(&descriptor, &weight_inputs)?;
    let stdout = std::io::stdout();
    let mut stdout = std::io::BufWriter::new(stdout.lock());
    write_control_receipt(
        &mut stdout,
        control_receipt("load", &session, descriptor.kernels.len(), None, 0),
    )?;

    for line in lines {
        let line = line.map_err(|error| {
            HostError::internal(format!("device-execute control read failed: {error}"))
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let request = parse_control_request(line.as_bytes())?;
        match request.verb {
            DeviceExecuteControlVerb::Load => {
                return Err(HostError::invalid_args(
                    "device-execute control load is only valid once per process",
                ));
            }
            DeviceExecuteControlVerb::Step => {
                let inputs = request.inputs.unwrap_or_else(|| base_inputs.clone());
                let started = Instant::now();
                let host_receipt = session.execute_step(&inputs)?;
                let wall_us = elapsed_us(started);
                let mut receipt = DeviceExecuteReceipt::from_host(&host_receipt);
                receipt.kernel_count = descriptor.kernels.len();
                receipt.stage_timing = stage_timing(wall_us, &host_receipt);
                write_control_receipt(
                    &mut stdout,
                    control_receipt("step", &session, descriptor.kernels.len(), Some(receipt), 0),
                )?;
            }
            DeviceExecuteControlVerb::Reset => {
                if request.inputs.is_some() {
                    return Err(HostError::invalid_args(
                        "device-execute control reset does not accept inputs",
                    ));
                }
                let cleared = session.reset_prompt()?;
                write_control_receipt(
                    &mut stdout,
                    control_receipt("reset", &session, descriptor.kernels.len(), None, cleared),
                )?;
            }
            DeviceExecuteControlVerb::Release => {
                if request.inputs.is_some() {
                    return Err(HostError::invalid_args(
                        "device-execute control release does not accept inputs",
                    ));
                }
                let lifecycle = session.teardown()?;
                write_control_receipt(
                    &mut stdout,
                    DeviceExecuteControlReceipt {
                        operation: "release".to_owned(),
                        lifecycle: DeviceExecuteLifecycle {
                            prepares: lifecycle.counters.prepares,
                            reuses: lifecycle.counters.reuses,
                            resets: lifecycle.counters.resets,
                            releases: lifecycle.counters.releases,
                            module_reloads: lifecycle.module_reloads,
                            per_program_reallocs: lifecycle.per_program_reallocs,
                            live_handles: lifecycle.live_handles,
                        },
                        kernel_count: descriptor.kernels.len(),
                        reset_cleared: 0,
                        receipt: None,
                    },
                )?;
                return Ok(());
            }
        }
    }
    Err(HostError::invalid_args(
        "device-execute control stream ended before explicit release",
    ))
}

fn control_receipt(
    operation: &str,
    session: &PreparedResidentSession<'_>,
    kernel_count: usize,
    receipt: Option<DeviceExecuteReceipt>,
    reset_cleared: usize,
) -> DeviceExecuteControlReceipt {
    let lifecycle = session.receipt();
    DeviceExecuteControlReceipt {
        operation: operation.to_owned(),
        lifecycle: DeviceExecuteLifecycle {
            prepares: lifecycle.counters.prepares,
            reuses: lifecycle.counters.reuses,
            resets: lifecycle.counters.resets,
            releases: lifecycle.counters.releases,
            module_reloads: lifecycle.module_reloads,
            per_program_reallocs: lifecycle.per_program_reallocs,
            live_handles: lifecycle.live_handles,
        },
        kernel_count,
        reset_cleared,
        receipt,
    }
}

fn write_control_receipt(
    stdout: &mut impl Write,
    receipt: DeviceExecuteControlReceipt,
) -> HostResult<()> {
    serde_json::to_writer(&mut *stdout, &receipt).map_err(|error| {
        HostError::internal(format!("device-execute control write failed: {error}"))
    })?;
    stdout.write_all(b"\n").map_err(|error| {
        HostError::internal(format!("device-execute control write failed: {error}"))
    })?;
    stdout.flush().map_err(|error| {
        HostError::internal(format!("device-execute control flush failed: {error}"))
    })
}

fn elapsed_us(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX)
}

fn stage_timing(wall_us: u64, receipt: &DeviceExecutionReceipt) -> StageTimingReceipt {
    let kernel_us = receipt.gpu_encode_submit_wait_us;
    let transfer_us = receipt.copy_in_us.saturating_add(receipt.readback_us);
    StageTimingReceipt {
        kernel_us,
        transfer_us,
        host_round_trip_us: wall_us
            .saturating_sub(kernel_us)
            .saturating_sub(transfer_us),
    }
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

/// One GGUF byte range copied into a host f32 buffer (`elems` logical f32s).
///
/// Packed Q4_K words occupy the prefix as raw bits; the tail is zero pad
/// so `copy_in_f32` sees the MIR logical element count. This is a file
/// map, not a weight codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightFileRange {
    /// Byte offset into `--weights`.
    pub offset: u64,
    /// Packed-byte length to copy.
    pub len: u64,
    /// Logical f32 element count of the host buffer.
    pub elems: u64,
}

/// Decode `{ "<buffer-id>": { "offset", "len", "elems" } }`.
pub fn weight_map_from_json(bytes: &[u8]) -> HostResult<BTreeMap<u32, WeightFileRange>> {
    let wire: BTreeMap<String, WeightFileRange> =
        serde_json::from_slice(bytes).map_err(|error| {
            HostError::invalid_args(format!(
                "device-execute weight-map JSON is invalid: {error}"
            ))
        })?;
    let mut map = BTreeMap::new();
    for (key, range) in wire {
        let id = key.parse::<u32>().map_err(|_| {
            HostError::invalid_args(format!(
                "device-execute weight-map key `{key}` is not a buffer id"
            ))
        })?;
        map.insert(id, range);
    }
    Ok(map)
}

/// Encode a weight map as `{ "<buffer-id>": { offset, len, elems } }`.
pub fn weight_map_to_json(map: &BTreeMap<u32, WeightFileRange>) -> HostResult<Vec<u8>> {
    let wire: BTreeMap<String, WeightFileRange> = map
        .iter()
        .map(|(id, range)| (id.to_string(), *range))
        .collect();
    serde_json::to_vec(&wire).map_err(|error| {
        HostError::internal(format!(
            "device-execute failed to encode weight-map: {error}"
        ))
    })
}

/// Copy GGUF byte ranges into host f32 buffers (prefix copy, zero pad).
pub fn inputs_from_gguf(
    gguf: &[u8],
    map: &BTreeMap<u32, WeightFileRange>,
) -> HostResult<BTreeMap<u32, Vec<f32>>> {
    let mut inputs = BTreeMap::new();
    for (id, range) in map {
        let start = usize::try_from(range.offset).map_err(|_| {
            HostError::invalid_args(format!("device-execute weight-map[{id}] offset overflows"))
        })?;
        let len = usize::try_from(range.len).map_err(|_| {
            HostError::invalid_args(format!("device-execute weight-map[{id}] len overflows"))
        })?;
        let end = start.checked_add(len).ok_or_else(|| {
            HostError::invalid_args(format!("device-execute weight-map[{id}] range overflows"))
        })?;
        let slice = gguf.get(start..end).ok_or_else(|| {
            HostError::invalid_args(format!(
                "device-execute weight-map[{id}] range [{start}, {end}) exceeds {} bytes",
                gguf.len()
            ))
        })?;
        inputs.insert(*id, packed_bytes_as_f32_padded(slice, range.elems));
    }
    Ok(inputs)
}

fn packed_bytes_as_f32_padded(bytes: &[u8], logical_elems: u64) -> Vec<f32> {
    let mut padded = bytes.to_vec();
    while !padded.len().is_multiple_of(4) {
        padded.push(0);
    }
    let mut values: Vec<f32> = padded
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let want = logical_elems as usize;
    if values.len() < want {
        values.resize(want, 0.0);
    }
    values
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
    /// Child file-read wall (descriptor + module + inputs).
    #[serde(default)]
    pub file_read_us: u64,
    /// Descriptor JSON decode wall.
    #[serde(default)]
    pub descriptor_decode_us: u64,
    /// Inputs JSON decode wall (tiny per-step values only).
    #[serde(default)]
    pub json_decode_us: u64,
    /// Composite-host construction wall (Metal device admit).
    #[serde(default)]
    pub host_construct_us: u64,
    /// Session create wall (module compile + PerProgram alloc).
    #[serde(default)]
    pub session_create_us: u64,
    /// Module compile + pipeline-create wall inside session create.
    #[serde(default)]
    pub load_module_us: u64,
    /// PerProgram allocation wall inside session create.
    #[serde(default)]
    pub per_program_alloc_us: u64,
    /// Host→device copy-in wall (packed-weight upload on SingleRun).
    #[serde(default)]
    pub copy_in_us: u64,
    /// Kernel encode + submit + blocking wait (true GPU step time).
    #[serde(default)]
    pub gpu_encode_submit_wait_us: u64,
    /// Observation readback wall.
    #[serde(default)]
    pub readback_us: u64,
    /// Child wall from first file read through teardown (excludes dyld).
    #[serde(default)]
    pub cli_internal_us: u64,
    /// Number of declared physical kernel bodies in the admitted descriptor.
    #[serde(default)]
    pub kernel_count: usize,
    /// Stage timing for this one-shot or control `step` operation.
    #[serde(default)]
    pub stage_timing: StageTimingReceipt,
}

/// Receipt timing split shared by the legacy one-shot and resident control
/// paths. The total is the sum of the three measured columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct StageTimingReceipt {
    pub kernel_us: u64,
    pub transfer_us: u64,
    pub host_round_trip_us: u64,
}

impl StageTimingReceipt {
    #[must_use]
    pub const fn total_us(self) -> u64 {
        self.kernel_us
            .saturating_add(self.transfer_us)
            .saturating_add(self.host_round_trip_us)
    }
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
            file_read_us: 0,
            descriptor_decode_us: 0,
            json_decode_us: 0,
            host_construct_us: 0,
            session_create_us: 0,
            load_module_us: 0,
            per_program_alloc_us: 0,
            copy_in_us: receipt.copy_in_us,
            gpu_encode_submit_wait_us: receipt.gpu_encode_submit_wait_us,
            readback_us: receipt.readback_us,
            cli_internal_us: 0,
            kernel_count: 0,
            stage_timing: StageTimingReceipt::default(),
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
