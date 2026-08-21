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
//! for tiny per-step values (tokens, rope, synthesized tables), or
//! `{ "<buffer-id>": { "dtype": "f32"|"f16"|"bf16", "bytes": "<hex>" } }`
//! for dtype-tagged raw payloads. Packed weights are not JSON: `--weights`
//! is the GGUF file and `--weight-map` names byte ranges inside it
//! (`offset` / `len` / `elems` per buffer id).
//!
//! Success prints a receipt JSON and exits 0. A host failure prints a
//! [`HostError`] JSON and exits 2. Usage / parse failures exit 64.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use host_coordinator::DeviceBackend;
use serde::{Deserialize, Serialize};

use crate::composite_host::invocation_binding::RopeConfig;
use crate::composite_host::{
    CompositeHost, CompositeHostConfig, DeviceByteBuffer, DeviceExecutionReceipt, DeviceSelection,
    PreparedResidentSession, PreparedSessionReceipt,
};
use crate::device_descriptor::{
    DescriptorBuffer, DescriptorBufferVersion, DescriptorDataFlow, DescriptorEndOfRunResult,
    DescriptorKernel, DescriptorLaunch, DescriptorResult, DeviceBufferInitialization,
    DeviceBufferLifetime, DeviceBufferRole, DeviceDataType, DeviceDescriptor,
    DeviceProgramLifetime,
};
use crate::device_host::DeviceSession;
use crate::kernel::{HostError, HostResult};
use crate::metal_host::{process_resident_bytes, MappedWeightFile, MappedWeightPaging};

/// The versioned control protocol carried by `device-execute`.
///
/// Version 1 is the existing one-program stream. It deliberately has no KV
/// invocation surface: a v1 request containing a mode or cursor fact is
/// rejected instead of being guessed into the v2 runtime-binding shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
pub enum DeviceExecuteProtocol {
    V1 = 1,
    V2 = 2,
}

impl DeviceExecuteProtocol {
    #[must_use]
    pub const fn version(self) -> u32 {
        self as u32
    }

    fn from_version(version: u32) -> Option<Self> {
        match version {
            1 => Some(Self::V1),
            2 => Some(Self::V2),
            _ => None,
        }
    }
}

/// Explicit v2 invocation regime. The spelling is part of the control wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceExecuteInvocationMode {
    Prefill,
    ScalarDecode,
}

impl DeviceExecuteInvocationMode {
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::ScalarDecode => "scalar_decode",
        }
    }
}

/// Runtime facts for one v2 invocation. Cursor facts are explicit and are
/// never inferred from the token or from the current step count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceExecuteInvocation {
    pub mode: DeviceExecuteInvocationMode,
    /// Decode's one token. Prefill may omit this when its token rows are in
    /// the declared input stream.
    #[serde(default)]
    pub token: Option<u32>,
    /// Absolute cache position written by this invocation.
    pub position: u32,
    pub sequence_epoch: u32,
    pub prefix_before: u32,
    pub valid_len_after: u32,
    pub query_start: u32,
}

/// CLI flags for `device-execute`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceExecuteArgs {
    /// Optional selection override (`auto` / `metal` / `cuda`).
    pub backend: Option<DeviceSelection>,
    /// Control protocol. Omitted means the legacy v1 surface.
    pub protocol: DeviceExecuteProtocol,
    /// Descriptor JSON path (no module image). For v2 this is the prefill
    /// descriptor alias, retained so old callers can still inspect the base
    /// paths.
    pub descriptor: PathBuf,
    /// Raw module-image path (MSL source or PTX).
    pub module: PathBuf,
    /// Inputs JSON path: f32 arrays or dtype-tagged hex bytes.
    pub inputs: PathBuf,
    /// Optional second v2 program descriptor/module pair.
    pub prefill_descriptor: Option<PathBuf>,
    pub prefill_module: Option<PathBuf>,
    pub decode_descriptor: Option<PathBuf>,
    pub decode_module: Option<PathBuf>,
    /// Explicit v2 model/session identities.
    pub model_identity: Option<String>,
    pub session_identity: Option<String>,
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
#[serde(default)]
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
    /// Protocol version that admitted this operation.
    #[serde(default = "default_protocol_version")]
    pub protocol: u32,
    /// Session lifecycle evidence after the operation.
    pub lifecycle: DeviceExecuteLifecycle,
    /// Explicit model/session identities carried by v2 load.
    #[serde(default)]
    pub model_identity: Option<String>,
    #[serde(default)]
    pub session_identity: Option<String>,
    /// Program-graph identities for both admitted v2 programs.
    #[serde(default)]
    pub program_identities: BTreeMap<String, String>,
    /// Number of declared physical kernel bodies in the loaded descriptor.
    #[serde(default)]
    pub kernel_count: usize,
    /// Number of state buffers cleared by `reset`.
    #[serde(default)]
    pub reset_cleared: usize,
    /// Device execution facts for a `step`; absent for control-only verbs.
    #[serde(default)]
    pub receipt: Option<DeviceExecuteReceipt>,
    /// mmap paging facts after `load` (zero on other verbs).
    #[serde(default)]
    pub mmap: MappedWeightPaging,
    /// Gradus `data_start` of the mapped GGUF (0 when the file is not GGUF).
    #[serde(default)]
    pub mmap_data_start: u64,
    /// Number of admitted `abs_starts`/`abs_ends` regions.
    #[serde(default)]
    pub mmap_regions: usize,
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
    pub protocol: DeviceExecuteProtocol,
    pub verb: DeviceExecuteControlVerb,
    pub inputs: Option<BTreeMap<u32, Vec<f32>>>,
    pub invocation: Option<DeviceExecuteInvocation>,
}

/// Parse `device-execute` CLI flags. Unknown or missing flags are usage
/// errors (the caller maps them to exit 64).
pub fn parse_device_execute_args(args: &[String]) -> Result<DeviceExecuteArgs, String> {
    let mut backend = None;
    let mut protocol = DeviceExecuteProtocol::V1;
    let mut descriptor = None;
    let mut module = None;
    let mut inputs = None;
    let mut prefill_descriptor = None;
    let mut prefill_module = None;
    let mut decode_descriptor = None;
    let mut decode_module = None;
    let mut model_identity = None;
    let mut session_identity = None;
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
            "--protocol" => {
                let value = next_flag_value(args, &mut index, "--protocol")?;
                protocol = parse_protocol(&value)?;
            }
            "--descriptor" => {
                let value = next_flag_value(args, &mut index, "--descriptor")?;
                descriptor = Some(PathBuf::from(value));
            }
            "--module" => {
                let value = next_flag_value(args, &mut index, "--module")?;
                module = Some(PathBuf::from(value));
            }
            "--prefill-descriptor" => {
                let value = next_flag_value(args, &mut index, "--prefill-descriptor")?;
                prefill_descriptor = Some(PathBuf::from(value));
            }
            "--prefill-module" => {
                let value = next_flag_value(args, &mut index, "--prefill-module")?;
                prefill_module = Some(PathBuf::from(value));
            }
            "--decode-descriptor" => {
                let value = next_flag_value(args, &mut index, "--decode-descriptor")?;
                decode_descriptor = Some(PathBuf::from(value));
            }
            "--decode-module" => {
                let value = next_flag_value(args, &mut index, "--decode-module")?;
                decode_module = Some(PathBuf::from(value));
            }
            "--model-identity" => {
                model_identity = Some(next_flag_value(args, &mut index, "--model-identity")?);
            }
            "--session-identity" => {
                session_identity = Some(next_flag_value(args, &mut index, "--session-identity")?);
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
    let descriptor = descriptor.or_else(|| prefill_descriptor.clone());
    let module = module.or_else(|| prefill_module.clone());
    let descriptor = descriptor.ok_or_else(|| usage_text().to_owned())?;
    let module = module.ok_or_else(|| usage_text().to_owned())?;
    let inputs = inputs.ok_or_else(|| usage_text().to_owned())?;
    let has_v2_program_flags = prefill_descriptor.is_some()
        || prefill_module.is_some()
        || decode_descriptor.is_some()
        || decode_module.is_some()
        || model_identity.is_some()
        || session_identity.is_some();
    if protocol == DeviceExecuteProtocol::V1 && has_v2_program_flags {
        return Err(format!(
            "protocol v1 cannot carry KV execution; request --protocol v2; {}",
            usage_text()
        ));
    }
    if protocol == DeviceExecuteProtocol::V2
        && (prefill_descriptor.is_none()
            || prefill_module.is_none()
            || decode_descriptor.is_none()
            || decode_module.is_none())
    {
        return Err(format!(
            "protocol v2 requires --prefill-descriptor/--prefill-module and --decode-descriptor/--decode-module; {}",
            usage_text()
        ));
    }
    Ok(DeviceExecuteArgs {
        backend,
        protocol,
        descriptor,
        module,
        inputs,
        prefill_descriptor,
        prefill_module,
        decode_descriptor,
        decode_module,
        model_identity,
        session_identity,
        weights,
        weight_map,
        control,
    })
}

/// Usage line for the command.
#[must_use]
pub fn usage_text() -> &'static str {
    "usage: faber-host-macos-arm64 device-execute [--control] [--protocol v1|v2] [--backend auto|metal|cuda] --descriptor <json> --module <bin> --inputs <json> [--prefill-descriptor <json> --prefill-module <bin> --decode-descriptor <json> --decode-module <bin>] [--weights <gguf> --weight-map <json>]"
}

#[derive(Debug, Deserialize)]
struct WireControlRequest {
    op: String,
    #[serde(default)]
    protocol: Option<u32>,
    #[serde(default)]
    inputs: Option<serde_json::Value>,
    #[serde(default)]
    mode: Option<DeviceExecuteInvocationMode>,
    #[serde(default)]
    token: Option<u32>,
    #[serde(default)]
    position: Option<u32>,
    #[serde(default)]
    sequence_epoch: Option<u32>,
    #[serde(default)]
    prefix_before: Option<u32>,
    #[serde(default)]
    valid_len_after: Option<u32>,
    #[serde(default)]
    query_start: Option<u32>,
}

fn default_protocol_version() -> u32 {
    DeviceExecuteProtocol::V1.version()
}

/// Decode one line of the resident-session control protocol.
///
/// The protocol is deliberately a local stdin/stdout stream, not HTTP. Input
/// values use the same lossless JSON spellings as the legacy `--inputs` file.
pub fn parse_control_request(bytes: &[u8]) -> HostResult<DeviceExecuteControlRequest> {
    let wire: WireControlRequest = serde_json::from_slice(bytes).map_err(|error| {
        HostError::invalid_args(format!("device-execute control JSON is invalid: {error}"))
    })?;
    let protocol = wire.protocol.unwrap_or(DeviceExecuteProtocol::V1.version());
    let protocol = DeviceExecuteProtocol::from_version(protocol).ok_or_else(|| {
        HostError::invalid_args(format!(
            "device-execute control protocol {protocol} is unsupported; expected 1 or 2"
        ))
    })?;
    let verb = match wire.op.as_str() {
        "load" => DeviceExecuteControlVerb::Load,
        "step" | "invoke" => DeviceExecuteControlVerb::Step,
        "reset" => DeviceExecuteControlVerb::Reset,
        "release" => DeviceExecuteControlVerb::Release,
        other => {
            return Err(HostError::invalid_args(format!(
                "device-execute control verb `{other}` is not one of load, step, reset, release"
            )))
        }
    };
    let has_v2_fields = wire.mode.is_some()
        || wire.token.is_some()
        || wire.position.is_some()
        || wire.sequence_epoch.is_some()
        || wire.prefix_before.is_some()
        || wire.valid_len_after.is_some()
        || wire.query_start.is_some();
    if protocol == DeviceExecuteProtocol::V1 && (has_v2_fields || wire.op == "invoke") {
        return Err(HostError::invalid_args(
            "device-execute protocol v1 cannot carry KV execution; request protocol v2",
        ));
    }
    if protocol == DeviceExecuteProtocol::V2
        && verb == DeviceExecuteControlVerb::Load
        && has_v2_fields
    {
        return Err(HostError::invalid_args(
            "device-execute v2 load admits programs; invocation facts belong to invoke",
        ));
    }
    let inputs = wire
        .inputs
        .map(|value| {
            let bytes = serde_json::to_vec(&value).map_err(|error| {
                HostError::invalid_args(format!(
                    "device-execute control inputs are invalid: {error}"
                ))
            })?;
            inputs_from_json(&bytes)?.into_f32_map()
        })
        .transpose()?;
    let invocation = if protocol == DeviceExecuteProtocol::V2
        && verb == DeviceExecuteControlVerb::Step
    {
        let mode = wire.mode.ok_or_else(|| {
            HostError::invalid_args(
                "device-execute protocol v2 invoke requires explicit mode and cursor facts",
            )
        })?;
        let invocation = DeviceExecuteInvocation {
            mode,
            token: wire.token,
            position: wire.position.ok_or_else(|| {
                HostError::invalid_args("device-execute v2 invoke requires position")
            })?,
            sequence_epoch: wire.sequence_epoch.ok_or_else(|| {
                HostError::invalid_args("device-execute v2 invoke requires sequence_epoch")
            })?,
            prefix_before: wire.prefix_before.ok_or_else(|| {
                HostError::invalid_args("device-execute v2 invoke requires prefix_before")
            })?,
            valid_len_after: wire.valid_len_after.ok_or_else(|| {
                HostError::invalid_args("device-execute v2 invoke requires valid_len_after")
            })?,
            query_start: wire.query_start.ok_or_else(|| {
                HostError::invalid_args("device-execute v2 invoke requires query_start")
            })?,
        };
        if mode == DeviceExecuteInvocationMode::ScalarDecode {
            if invocation.token.is_none() {
                return Err(HostError::invalid_args(
                    "device-execute scalar_decode invoke requires token",
                ));
            }
            if inputs.is_some() {
                return Err(HostError::invalid_args(
                    "device-execute scalar_decode invoke carries token/position only; inputs are not accepted",
                ));
            }
        }
        Some(invocation)
    } else {
        if has_v2_fields {
            return Err(HostError::invalid_args(
                "device-execute v2 mode/cursor facts are only valid on invoke",
            ));
        }
        None
    };
    Ok(DeviceExecuteControlRequest {
        protocol,
        verb,
        inputs,
        invocation,
    })
}

/// Admit the two static programs carried by a v2 load.
///
/// This seam is intentionally pure. It validates both descriptors before the
/// paired resident owner is constructed and returns the identities that the
/// owner must preserve. Cursor values are not part of either identity.
pub fn admit_v2_load(
    prefill: &DeviceDescriptor,
    decode: &DeviceDescriptor,
    model_identity: impl Into<String>,
    session_identity: impl Into<String>,
) -> HostResult<DeviceExecuteControlReceipt> {
    prefill.validate()?;
    decode.validate()?;
    if prefill.backend != decode.backend {
        return Err(HostError::invalid_args(format!(
            "device-execute protocol v2 programs target different backends: {} and {}",
            prefill.backend.spelling(),
            decode.backend.spelling()
        )));
    }
    let model_identity = model_identity.into();
    let session_identity = session_identity.into();
    if model_identity.is_empty() || session_identity.is_empty() {
        return Err(HostError::invalid_args(
            "device-execute protocol v2 load requires non-empty model_identity and session_identity",
        ));
    }
    let program_identities = BTreeMap::from([
        ("prefill".to_owned(), prefill.program_graph_hash()),
        ("scalar_decode".to_owned(), decode.program_graph_hash()),
    ]);
    Ok(DeviceExecuteControlReceipt {
        operation: "load".to_owned(),
        protocol: DeviceExecuteProtocol::V2.version(),
        lifecycle: DeviceExecuteLifecycle {
            prepares: 1,
            ..DeviceExecuteLifecycle::default()
        },
        model_identity: Some(model_identity),
        session_identity: Some(session_identity),
        program_identities,
        kernel_count: prefill.kernels.len() + decode.kernels.len(),
        reset_cleared: 0,
        receipt: None,
        mmap: MappedWeightPaging::default(),
        mmap_data_start: 0,
        mmap_regions: 0,
    })
}

/// Load files, validate the descriptor, construct the composite host, and
/// execute one packed device run.
pub fn run_device_execute(args: &DeviceExecuteArgs) -> HostResult<DeviceExecuteReceipt> {
    if args.protocol == DeviceExecuteProtocol::V2 {
        return Err(HostError::invalid_args(
            "protocol v2 paired programs require the --control load/invoke stream",
        ));
    }
    let cli_started = Instant::now();
    let read_started = Instant::now();
    let descriptor_bytes = read_file(&args.descriptor)?;
    let module_image = read_file(&args.module)?;
    let inputs_bytes = read_file(&args.inputs)?;
    let weight_map_bytes = match &args.weight_map {
        Some(path) => Some(read_file(path)?),
        None => None,
    };
    let mapped_weights = match &args.weights {
        Some(path) => Some(MappedWeightFile::open(path)?),
        None => None,
    };
    let mut weight_inputs = WeightInputs::default();
    let mut mmap_paging = MappedWeightPaging::default();
    let mut mmap_data_start = 0u64;
    let mut mmap_regions = 0usize;
    if let (Some(mapped), Some(map_json)) = (&mapped_weights, weight_map_bytes.as_deref()) {
        let map = weight_map_from_json(map_json)?;
        let table = gguf_region_table(mapped.bytes(), &map)?;
        weight_inputs = inputs_from_mapped_gguf(mapped, &map, &table)?;
        mmap_paging = mapped_paging(mapped);
        mmap_data_start = table.data_start;
        mmap_regions = table.abs_starts.len();
    }
    let file_read_us = elapsed_us(read_started);
    let descriptor_started = Instant::now();
    let descriptor = descriptor_from_json(&descriptor_bytes, module_image)?;
    let descriptor_decode_us = elapsed_us(descriptor_started);
    let inputs_started = Instant::now();
    let json_inputs = inputs_from_json(&inputs_bytes)?;
    merge_json_inputs(&mut weight_inputs, json_inputs)?;
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
    retain_mapped_weights(&mut host, mapped_weights.as_ref())?;
    let session_started = Instant::now();
    let mut session = host.create_program_session(&descriptor)?;
    let session_create_us = elapsed_us(session_started);
    let load_module_us = session.load_module_us();
    let per_program_alloc_us = session.per_program_alloc_us();
    let step_started = Instant::now();
    let receipt =
        session.execute_with_weight_bytes(weight_inputs.map(), weight_inputs.byte_map())?;
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
    wire.mmap = mmap_paging;
    wire.mmap_data_start = mmap_data_start;
    wire.mmap_regions = mmap_regions;
    Ok(wire)
}

/// Run the v2 paired-program control stream.
///
/// D5 owns the protocol boundary: both static programs are admitted by one
/// load. P1 constructs the paired owner after that pure admission and routes
/// each mode-selected invoke through its real resident dispatch rail, without
/// changing the v1 single-program path.
fn run_device_execute_control_v2(args: &DeviceExecuteArgs) -> HostResult<()> {
    let prefill_descriptor = args
        .prefill_descriptor
        .as_ref()
        .ok_or_else(|| HostError::invalid_args("protocol v2 requires a prefill descriptor"))?;
    let prefill_module = args
        .prefill_module
        .as_ref()
        .ok_or_else(|| HostError::invalid_args("protocol v2 requires a prefill module"))?;
    let decode_descriptor = args
        .decode_descriptor
        .as_ref()
        .ok_or_else(|| HostError::invalid_args("protocol v2 requires a decode descriptor"))?;
    let decode_module = args
        .decode_module
        .as_ref()
        .ok_or_else(|| HostError::invalid_args("protocol v2 requires a decode module"))?;
    let prefill =
        descriptor_from_json(&read_file(prefill_descriptor)?, read_file(prefill_module)?)?;
    let decode = descriptor_from_json(&read_file(decode_descriptor)?, read_file(decode_module)?)?;
    let model_identity = args
        .model_identity
        .clone()
        .ok_or_else(|| HostError::invalid_args("protocol v2 requires --model-identity"))?;
    let session_identity = args
        .session_identity
        .clone()
        .ok_or_else(|| HostError::invalid_args("protocol v2 requires --session-identity"))?;

    let json_inputs = inputs_from_json(&read_file(&args.inputs)?)?;
    let weight_map = match &args.weight_map {
        Some(path) => weight_map_from_json(&read_file(path)?)?,
        None => BTreeMap::new(),
    };
    let mapped_weights = match &args.weights {
        Some(path) => Some(MappedWeightFile::open(path)?),
        None => None,
    };
    let mut weight_inputs = if let Some(mapped) = &mapped_weights {
        let table = gguf_region_table(mapped.bytes(), &weight_map)?;
        inputs_from_mapped_gguf(mapped, &weight_map, &table)?
    } else {
        WeightInputs::default()
    };
    merge_json_inputs(&mut weight_inputs, json_inputs)?;
    let prompt_tokens = prompt_tokens_from_inputs(&prefill, weight_inputs.map())?;
    let rope = rope_config_from_inputs(&prefill, weight_inputs.map(), prompt_tokens.len())?;

    let mut load = admit_v2_load(
        &prefill,
        &decode,
        model_identity.clone(),
        session_identity.clone(),
    )?;
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
    if first.protocol != DeviceExecuteProtocol::V2
        || first.verb != DeviceExecuteControlVerb::Load
        || first.inputs.is_some()
    {
        return Err(HostError::invalid_args(
            "device-execute protocol v2 stream must begin with {\\\"protocol\\\":2,\\\"op\\\":\\\"load\\\"}",
        ));
    }

    let selection = args
        .backend
        .unwrap_or_else(|| selection_for_backend(prefill.backend));
    let mut host = CompositeHost::new(CompositeHostConfig {
        selection,
        requires_device: true,
    })?;
    retain_mapped_weights(&mut host, mapped_weights.as_ref())?;
    let mut paired = host.prepare_paired_session(
        &prefill,
        &decode,
        prompt_tokens,
        rope,
        weight_inputs.map(),
        weight_inputs.byte_map(),
        model_identity,
        session_identity,
    )?;
    load.lifecycle.live_handles = paired.live_handles();
    load.lifecycle.module_reloads = paired.module_reloads();
    load.lifecycle.per_program_reallocs = paired.per_program_reallocs();

    let stdout = std::io::stdout();
    let mut stdout = std::io::BufWriter::new(stdout.lock());
    write_control_receipt(&mut stdout, load.clone())?;
    for line in lines {
        let line = line.map_err(|error| {
            HostError::internal(format!("device-execute control read failed: {error}"))
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let request = parse_control_request(line.as_bytes())?;
        if request.protocol != DeviceExecuteProtocol::V2 {
            return Err(HostError::invalid_args(
                "device-execute control protocol cannot change after v2 load",
            ));
        }
        match request.verb {
            DeviceExecuteControlVerb::Load => {
                return Err(HostError::invalid_args(
                    "device-execute protocol v2 load is only valid once per process",
                ));
            }
            DeviceExecuteControlVerb::Step => {
                let invocation = request.invocation.ok_or_else(|| {
                    HostError::invalid_args(
                        "device-execute protocol v2 invoke requires invocation facts",
                    )
                })?;
                if request.inputs.is_some() {
                    return Err(HostError::invalid_args(
                        "device-execute v2 invoke carries cursor facts only; load owns prefill inputs",
                    ));
                }
                let started = Instant::now();
                let receipt = paired.execute_invocation(&invocation)?;
                let wall_us = elapsed_us(started);
                let mut wire = DeviceExecuteReceipt::from_host(&receipt);
                wire.kernel_count = match invocation.mode {
                    DeviceExecuteInvocationMode::Prefill => prefill.kernels.len(),
                    DeviceExecuteInvocationMode::ScalarDecode => decode.kernels.len(),
                };
                wire.stage_timing = stage_timing(wall_us, &receipt);
                load.operation = "invoke".to_owned();
                load.lifecycle.reuses = paired.reuses();
                load.lifecycle.live_handles = paired.live_handles();
                load.receipt = Some(wire);
                write_control_receipt(&mut stdout, load.clone())?;
            }
            DeviceExecuteControlVerb::Reset => {
                if request.inputs.is_some() || request.invocation.is_some() {
                    return Err(HostError::invalid_args(
                        "device-execute protocol v2 reset does not accept invocation inputs",
                    ));
                }
                load.operation = "reset".to_owned();
                load.lifecycle.resets += 1;
                load.lifecycle.live_handles = paired.live_handles();
                load.receipt = None;
                write_control_receipt(&mut stdout, load.clone())?;
            }
            DeviceExecuteControlVerb::Release => {
                if request.inputs.is_some() || request.invocation.is_some() {
                    return Err(HostError::invalid_args(
                        "device-execute protocol v2 release does not accept invocation inputs",
                    ));
                }
                let reuses = paired.reuses();
                paired.teardown()?;
                load.operation = "release".to_owned();
                load.lifecycle.reuses = reuses;
                load.lifecycle.releases += 1;
                load.lifecycle.live_handles = 0;
                load.receipt = None;
                write_control_receipt(&mut stdout, load)?;
                return Ok(());
            }
        }
    }
    Err(HostError::invalid_args(
        "device-execute control stream ended before explicit release",
    ))
}

fn input_id_by_name(descriptor: &DeviceDescriptor, name: &str) -> HostResult<u32> {
    let mut found = None;
    for kernel in &descriptor.kernels {
        for slot in &kernel.buffers {
            if slot.role != DeviceBufferRole::Input || slot.buffer_name != name {
                continue;
            }
            if let Some(previous) = found {
                if previous != slot.buffer_id {
                    return Err(HostError::invalid_args(format!(
                        "v2 input `{name}` has conflicting buffer identities"
                    )));
                }
            } else {
                found = Some(slot.buffer_id);
            }
        }
    }
    found.ok_or_else(|| HostError::invalid_args(format!("v2 input `{name}` is not declared")))
}

fn prompt_tokens_from_inputs(
    descriptor: &DeviceDescriptor,
    inputs: &BTreeMap<u32, Vec<f32>>,
) -> HostResult<Vec<u32>> {
    let id = input_id_by_name(
        descriptor,
        crate::composite_host::invocation_binding::PROMPT_TOKENS,
    )?;
    let values = inputs.get(&id).ok_or_else(|| {
        HostError::invalid_args(format!(
            "v2 prefill input `{}` is missing from the load inputs",
            crate::composite_host::invocation_binding::PROMPT_TOKENS
        ))
    })?;
    if values.is_empty() {
        return Err(HostError::invalid_args(
            "v2 prefill prompt_tokens must contain at least one row",
        ));
    }
    Ok(values.iter().map(|value| value.to_bits()).collect())
}

fn rope_config_from_inputs(
    descriptor: &DeviceDescriptor,
    inputs: &BTreeMap<u32, Vec<f32>>,
    prompt_len: usize,
) -> HostResult<RopeConfig> {
    let cos_id = input_id_by_name(
        descriptor,
        crate::composite_host::invocation_binding::ROPE_COS,
    )?;
    let sin_id = input_id_by_name(
        descriptor,
        crate::composite_host::invocation_binding::ROPE_SIN,
    )?;
    let cos = inputs.get(&cos_id).ok_or_else(|| {
        HostError::invalid_args("v2 prefill rope cosine input is missing from the load inputs")
    })?;
    let sin = inputs.get(&sin_id).ok_or_else(|| {
        HostError::invalid_args("v2 prefill rope sine input is missing from the load inputs")
    })?;
    if prompt_len == 0 || cos.len() != sin.len() || cos.len() % prompt_len != 0 {
        return Err(HostError::invalid_args(
            "v2 prefill RoPE inputs do not contain one equal-width row per prompt token",
        ));
    }
    let row_width = cos.len() / prompt_len;
    if row_width == 0 || row_width > (u32::MAX as usize / 2) {
        return Err(HostError::invalid_args(
            "v2 prefill RoPE row width is outside the host surface",
        ));
    }
    let head_dim = u32::try_from(row_width * 2)
        .map_err(|_| HostError::invalid_args("v2 prefill RoPE head dimension overflows"))?;
    let theta = infer_rope_theta(cos, sin, row_width, prompt_len).unwrap_or(10_000.0);
    Ok(RopeConfig { head_dim, theta })
}

fn infer_rope_theta(cos: &[f32], sin: &[f32], row_width: usize, rows: usize) -> Option<f64> {
    if row_width < 2 || rows < 2 {
        return None;
    }
    let pair = row_width - 1;
    let angle = f64::from(sin[row_width + pair]).atan2(f64::from(cos[row_width + pair]));
    if !angle.is_finite() || angle <= 0.0 {
        return None;
    }
    let head_dim = (row_width * 2) as f64;
    let exponent = -head_dim / (2.0 * pair as f64);
    let theta = angle.powf(exponent);
    (theta.is_finite() && theta > 0.0).then_some(theta)
}

/// Run the explicit resident-session control stream.
///
/// The descriptor/module/weight map are read once, then the first command must
/// be `load`. The resulting prepared session stays owned by this one host
/// process until an explicit `release`; each `step` only receives invocation
/// inputs and uses the already admitted resident adapter.
pub fn run_device_execute_control(args: &DeviceExecuteArgs) -> HostResult<()> {
    if args.protocol == DeviceExecuteProtocol::V2 {
        return run_device_execute_control_v2(args);
    }
    let descriptor_bytes = read_file(&args.descriptor)?;
    let module_image = read_file(&args.module)?;
    let json_inputs = inputs_from_json(&read_file(&args.inputs)?)?;
    let weight_map = match &args.weight_map {
        Some(path) => weight_map_from_json(&read_file(path)?)?,
        None => BTreeMap::new(),
    };
    let mapped_weights = match &args.weights {
        Some(path) => Some(MappedWeightFile::open(path)?),
        None => None,
    };
    let mut mmap_paging = MappedWeightPaging::default();
    let mut mmap_data_start = 0u64;
    let mut mmap_regions = 0usize;
    let mut weight_inputs = if let Some(mapped) = &mapped_weights {
        let table = gguf_region_table(mapped.bytes(), &weight_map)?;
        mmap_paging = mapped_paging(mapped);
        mmap_data_start = table.data_start;
        mmap_regions = table.abs_starts.len();
        inputs_from_mapped_gguf(mapped, &weight_map, &table)?
    } else {
        WeightInputs::default()
    };
    merge_json_inputs(&mut weight_inputs, json_inputs)?;
    let base_inputs = weight_inputs.take_f32_map();
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
    if first.protocol != args.protocol
        || first.verb != DeviceExecuteControlVerb::Load
        || first.inputs.is_some()
        || first.invocation.is_some()
    {
        return Err(HostError::invalid_args(
            "device-execute control stream must begin with {\\\"op\\\":\\\"load\\\"}",
        ));
    }

    let mut host = CompositeHost::new(CompositeHostConfig {
        selection,
        requires_device: true,
    })?;
    retain_mapped_weights(&mut host, mapped_weights.as_ref())?;
    let mut session = PreparedResidentSession::prepare_with_weight_bytes(
        &mut host,
        &descriptor,
        weight_inputs.map(),
        weight_inputs.byte_map(),
    )?;
    let stdout = std::io::stdout();
    let mut stdout = std::io::BufWriter::new(stdout.lock());
    let mut load_receipt = control_receipt("load", &session, descriptor.kernels.len(), None, 0);
    load_receipt.mmap = mmap_paging;
    load_receipt.mmap_data_start = mmap_data_start;
    load_receipt.mmap_regions = mmap_regions;
    write_control_receipt(&mut stdout, load_receipt)?;

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
                let timing = session.receipt();
                let mut receipt = DeviceExecuteReceipt::from_host_with_phase_timing(
                    &host_receipt,
                    f4h1_phase_timing(&timing),
                );
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
                        protocol: args.protocol.version(),
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
                        model_identity: None,
                        session_identity: None,
                        program_identities: BTreeMap::new(),
                        mmap: MappedWeightPaging::default(),
                        mmap_data_start: 0,
                        mmap_regions: 0,
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
        protocol: DeviceExecuteProtocol::V1.version(),
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
        model_identity: None,
        session_identity: None,
        program_identities: BTreeMap::new(),
        mmap: MappedWeightPaging::default(),
        mmap_data_start: 0,
        mmap_regions: 0,
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

/// Numeric phase values carried from the F4H1 steady-state receipt.
#[derive(Debug, Clone, Copy, Default)]
struct DevicePhaseTiming {
    encode_us: u64,
    submit_us: u64,
    wait_us: u64,
}

/// Project one F4H1 measurement onto the numeric compatibility wire.
///
/// `NotMeasured` stays the wire's zero value. Measured and derived values are
/// copied from the receipt; this helper never recomputes a phase from another
/// timing field.
fn f4h1_measurement_us<T: Serialize>(measurement: T) -> u64 {
    serde_json::to_value(measurement)
        .ok()
        .and_then(|value| value.get("value_us").and_then(serde_json::Value::as_u64))
        .unwrap_or_default()
}

fn f4h1_phase_timing(receipt: &PreparedSessionReceipt) -> DevicePhaseTiming {
    let phase = &receipt.timing.steady_state;
    DevicePhaseTiming {
        encode_us: f4h1_measurement_us(phase.encode.duration_us),
        submit_us: f4h1_measurement_us(phase.submit.duration_us),
        wait_us: f4h1_measurement_us(phase.wait.duration_us),
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

/// Decode `{ "<buffer-id>": [f32, ...] }` or dtype-tagged hex bytes.
///
/// Untagged arrays keep the legacy f32 wire: finite values are JSON
/// numbers (`f32` → `f64` is injective); NaN payloads are `"0x"` + 8 hex
/// bits so packed GGUF words survive the file; `"NaN"` still decodes as
/// the canonical quiet NaN; infinities stay `"Infinity"` / `"-Infinity"`.
/// Tagged objects `{ "dtype", "bytes" }` stay raw bytes with a
/// [`DeviceDataType`] tag; a tail that is not a multiple of the tag's
/// byte width fails closed by name.
pub fn inputs_from_json(bytes: &[u8]) -> HostResult<WeightInputs> {
    let wire: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(bytes).map_err(|error| {
            HostError::invalid_args(format!("device-execute inputs JSON is invalid: {error}"))
        })?;
    let mut inputs = WeightInputs::default();
    for (key, value) in wire {
        let id = key.parse::<u32>().map_err(|_| {
            HostError::invalid_args(format!(
                "device-execute inputs key `{key}` is not a buffer id"
            ))
        })?;
        match value {
            serde_json::Value::Array(values) => {
                let mut parsed = Vec::with_capacity(values.len());
                for (index, item) in values.into_iter().enumerate() {
                    parsed.push(f32_from_json(&item).map_err(|detail| {
                        HostError::invalid_args(format!(
                            "device-execute inputs[{id}][{index}] is invalid: {detail}"
                        ))
                    })?);
                }
                inputs.insert_owned(id, parsed);
            }
            serde_json::Value::Object(_) => {
                let tagged: WireTaggedInput = serde_json::from_value(value).map_err(|error| {
                    HostError::invalid_args(format!(
                        "device-execute inputs[{id}] tagged bytes are invalid: {error}"
                    ))
                })?;
                let dtype = parse_input_dtype(&tagged.dtype)?;
                let payload = parse_hex_bytes(&tagged.bytes).map_err(|detail| {
                    HostError::invalid_args(format!(
                        "device-execute inputs[{id}] bytes are invalid: {detail}"
                    ))
                })?;
                if payload.len() % dtype.byte_width() != 0 {
                    return Err(misaligned_input_tail(id, dtype, payload.len()));
                }
                inputs.insert_bytes_owned(
                    id,
                    DeviceByteBuffer {
                        bytes: payload,
                        dtype,
                    },
                );
            }
            other => {
                return Err(HostError::invalid_args(format!(
                    "device-execute inputs[{id}] expected an f32 array or dtype-tagged bytes, got {other}"
                )));
            }
        }
    }
    Ok(inputs)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireTaggedInput {
    dtype: String,
    bytes: String,
}

fn parse_input_dtype(spelling: &str) -> HostResult<DeviceDataType> {
    DeviceDataType::from_spelling(spelling).ok_or_else(|| {
        HostError::invalid_args(format!(
            "device-execute inputs dtype `{spelling}` is outside the host dtype surface"
        ))
    })
}

fn misaligned_input_tail(id: u32, dtype: DeviceDataType, len: usize) -> HostError {
    HostError::invalid_args(format!(
        "device-execute inputs[{id}] rejects a misaligned {} tail of {} bytes",
        dtype.spelling(),
        len
    ))
}

fn merge_json_inputs(dst: &mut WeightInputs, mut src: WeightInputs) -> HostResult<()> {
    let ids: Vec<u32> = src.values.keys().chain(src.bytes.keys()).copied().collect();
    for id in ids {
        if dst.contains(id) {
            return Err(HostError::invalid_args(format!(
                "device-execute buffer {id} is in both --weight-map and --inputs"
            )));
        }
    }
    dst.values.append(&mut src.values);
    dst.bytes.append(&mut src.bytes);
    Ok(())
}

/// One GGUF byte range admitted as a native packed device region.
///
/// Packed words occupy the prefix as raw bits. `elems` is the admitted
/// packed-region width in f32 words (`ceil(len / 4)`), never the logical
/// F32 element count. Extra logical-F32 padding fails closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightFileRange {
    /// Byte offset into `--weights`.
    pub offset: u64,
    /// Packed-byte length to copy.
    pub len: u64,
    /// Admitted packed-region width in f32 words.
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

/// Admit GGUF byte ranges as raw dtype-tagged weight regions. The source
/// bytes stay bytes all the way to the session-owned PerProgram allocation;
/// the old f32 reinterpretation path is deliberately not used.
pub fn inputs_from_gguf(
    gguf: &[u8],
    map: &BTreeMap<u32, WeightFileRange>,
) -> HostResult<WeightInputs> {
    let mut inputs = WeightInputs::default();
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
        validate_packed_range(*id, range)?;
        inputs.insert_bytes_owned(
            *id,
            DeviceByteBuffer {
                bytes: slice.to_vec(),
                dtype: DeviceDataType::U8,
            },
        );
    }
    Ok(inputs)
}

fn validate_packed_range(id: u32, range: &WeightFileRange) -> HostResult<()> {
    let packed_elems = packed_f32_count(range.len);
    if packed_elems != range.elems {
        return Err(HostError::invalid_args(format!(
            "device-execute weight-map[{id}] elems {} is not the native packed width {packed_elems} (len {})",
            range.elems, range.len
        )));
    }
    Ok(())
}

/// Gradus region table: `data_start` plus the admitted `abs_starts` /
/// `abs_ends` ranges. Weight-map offsets are already absolute file offsets
/// (`data_start + relative_offset`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufRegionTable {
    pub data_start: u64,
    pub abs_starts: Vec<u64>,
    pub abs_ends: Vec<u64>,
}

/// Host-side inputs for one device-execute request. Untagged JSON arrays
/// remain f32 values; tagged JSON objects and GGUF weight-map entries are
/// raw dtype-tagged bytes. Mapped GGUF ranges may alias a retained mmap.
#[derive(Debug, Default)]
pub struct WeightInputs {
    values: BTreeMap<u32, Vec<f32>>,
    bytes: BTreeMap<u32, DeviceByteBuffer>,
    aliased_bytes: BTreeSet<u32>,
}

impl Drop for WeightInputs {
    fn drop(&mut self) {
        for id in std::mem::take(&mut self.aliased_bytes) {
            if let Some(values) = self.bytes.remove(&id) {
                std::mem::forget(values.bytes);
            }
        }
    }
}

impl WeightInputs {
    #[must_use]
    pub fn map(&self) -> &BTreeMap<u32, Vec<f32>> {
        &self.values
    }

    #[must_use]
    pub fn byte_map(&self) -> &BTreeMap<u32, DeviceByteBuffer> {
        &self.bytes
    }

    #[must_use]
    pub fn contains(&self, id: u32) -> bool {
        self.values.contains_key(&id) || self.bytes.contains_key(&id)
    }

    fn insert_owned(&mut self, id: u32, values: Vec<f32>) {
        self.values.insert(id, values);
    }

    fn insert_bytes_owned(&mut self, id: u32, values: DeviceByteBuffer) {
        self.bytes.insert(id, values);
    }

    fn insert_bytes_aliased(&mut self, id: u32, bytes: &[u8]) {
        let values =
            unsafe { Vec::from_raw_parts(bytes.as_ptr() as *mut u8, bytes.len(), bytes.len()) };
        self.bytes.insert(
            id,
            DeviceByteBuffer {
                bytes: values,
                dtype: DeviceDataType::U8,
            },
        );
        self.aliased_bytes.insert(id);
    }

    fn take_f32_map(&mut self) -> BTreeMap<u32, Vec<f32>> {
        std::mem::take(&mut self.values)
    }

    fn into_f32_map(mut self) -> HostResult<BTreeMap<u32, Vec<f32>>> {
        if !self.bytes.is_empty() {
            return Err(HostError::invalid_args(
                "device-execute control inputs do not accept dtype-tagged bytes",
            ));
        }
        Ok(self.take_f32_map())
    }
}

/// Build the region table from a mapped GGUF (or a raw fixture file).
///
/// A GGUF v3 file supplies `data_start` from the header table end aligned
/// to 32. Ranges that start before `data_start` or past the file fail
/// closed. Non-GGUF fixtures use `data_start = 0` so existing packed-map
/// tests keep their byte offsets.
pub fn gguf_region_table(
    bytes: &[u8],
    map: &BTreeMap<u32, WeightFileRange>,
) -> HostResult<GgufRegionTable> {
    let data_start = match gguf_data_start(bytes)? {
        Some(start) => start,
        None => 0,
    };
    let file_len = bytes.len() as u64;
    let mut abs_starts = Vec::with_capacity(map.len());
    let mut abs_ends = Vec::with_capacity(map.len());
    for (id, range) in map {
        let start = range.offset;
        let end = start.checked_add(range.len).ok_or_else(|| {
            HostError::invalid_args(format!("device-execute weight-map[{id}] range overflows"))
        })?;
        if start < data_start {
            return Err(HostError::invalid_args(format!(
                "device-execute weight-map[{id}] offset {start} is before GGUF data_start {data_start}"
            )));
        }
        if end > file_len {
            return Err(HostError::invalid_args(format!(
                "device-execute weight-map[{id}] range [{start}, {end}) exceeds {} bytes",
                bytes.len()
            )));
        }
        abs_starts.push(start);
        abs_ends.push(end);
    }
    Ok(GgufRegionTable {
        data_start,
        abs_starts,
        abs_ends,
    })
}

/// Admit mapped GGUF ranges as raw dtype-tagged byte regions without copying.
/// The retained mmap keeps the aliased bytes live until the device session has
/// finished uploading them; CUDA copies from the same raw slice.
pub fn inputs_from_mapped_gguf(
    mapped: &MappedWeightFile,
    map: &BTreeMap<u32, WeightFileRange>,
    table: &GgufRegionTable,
) -> HostResult<WeightInputs> {
    let bytes = mapped.bytes();
    let mut inputs = WeightInputs::default();
    for (id, range) in map {
        if range.offset < table.data_start {
            return Err(HostError::invalid_args(format!(
                "device-execute weight-map[{id}] offset {} is before GGUF data_start {}",
                range.offset, table.data_start
            )));
        }
        let start = usize::try_from(range.offset).map_err(|_| {
            HostError::invalid_args(format!("device-execute weight-map[{id}] offset overflows"))
        })?;
        let end = start
            .checked_add(usize::try_from(range.len).map_err(|_| {
                HostError::invalid_args(format!("device-execute weight-map[{id}] len overflows"))
            })?)
            .ok_or_else(|| {
                HostError::invalid_args(format!("device-execute weight-map[{id}] range overflows"))
            })?;
        let slice = bytes.get(start..end).ok_or_else(|| {
            HostError::invalid_args(format!(
                "device-execute weight-map[{id}] range [{start}, {end}) exceeds {} bytes",
                bytes.len()
            ))
        })?;
        validate_packed_range(*id, range)?;
        inputs.insert_bytes_aliased(*id, slice);
    }
    Ok(inputs)
}

fn gguf_data_start(bytes: &[u8]) -> HostResult<Option<u64>> {
    if bytes.len() < 24 || &bytes[..4] != b"GGUF" {
        return Ok(None);
    }
    let version = read_u32_at(bytes, 4, "GGUF version")?;
    if version != 3 {
        return Err(HostError::invalid_args(format!(
            "device-execute GGUF version {version} is not v3"
        )));
    }
    let n_tensors = read_u64_at(bytes, 8, "GGUF tensor count")?;
    let n_kv = read_u64_at(bytes, 16, "GGUF metadata count")?;
    let mut off = 24usize;
    for _ in 0..n_kv {
        off = skip_gguf_kv(bytes, off)?;
    }
    for _ in 0..n_tensors {
        off = skip_gguf_tensor_info(bytes, off)?;
    }
    const ALIGN: usize = 32;
    let data_start = off.div_ceil(ALIGN).saturating_mul(ALIGN);
    if data_start > bytes.len() {
        return Err(HostError::invalid_args(
            "device-execute GGUF data_start is past the mapped file",
        ));
    }
    Ok(Some(data_start as u64))
}

fn skip_gguf_string(bytes: &[u8], off: usize) -> HostResult<usize> {
    let len = read_u64_at(bytes, off, "GGUF string length")?;
    let start = off + 8;
    let end = start
        .checked_add(len as usize)
        .ok_or_else(|| HostError::invalid_args("device-execute GGUF string overflows"))?;
    if end > bytes.len() {
        return Err(HostError::invalid_args(
            "device-execute GGUF string exceeds the mapped file",
        ));
    }
    Ok(end)
}

fn skip_gguf_kv(bytes: &[u8], off: usize) -> HostResult<usize> {
    let after_key = skip_gguf_string(bytes, off)?;
    let tag = read_u32_at(bytes, after_key, "GGUF metadata type")?;
    skip_gguf_value(bytes, after_key + 4, tag)
}

fn skip_gguf_tensor_info(bytes: &[u8], off: usize) -> HostResult<usize> {
    let after_name = skip_gguf_string(bytes, off)?;
    let ndims = read_u32_at(bytes, after_name, "GGUF tensor ndims")?;
    let mut next = after_name + 4;
    for _ in 0..ndims {
        read_u64_at(bytes, next, "GGUF tensor dim")?;
        next += 8;
    }
    read_u32_at(bytes, next, "GGUF tensor type")?;
    next += 4;
    read_u64_at(bytes, next, "GGUF tensor offset")?;
    Ok(next + 8)
}

fn skip_gguf_value(bytes: &[u8], off: usize, tag: u32) -> HostResult<usize> {
    match tag {
        0 | 1 | 7 => Ok(off + 1),
        2 | 3 => Ok(off + 2),
        4 | 5 | 6 => Ok(off + 4),
        8 => skip_gguf_string(bytes, off),
        10 | 11 | 12 => Ok(off + 8),
        9 => {
            let elem = read_u32_at(bytes, off, "GGUF array elem type")?;
            let count = read_u64_at(bytes, off + 4, "GGUF array count")?;
            let mut next = off + 12;
            for _ in 0..count {
                next = skip_gguf_value(bytes, next, elem)?;
            }
            Ok(next)
        }
        other => Err(HostError::invalid_args(format!(
            "device-execute GGUF metadata type {other} is unhandled"
        ))),
    }
}

fn read_u32_at(bytes: &[u8], off: usize, what: &str) -> HostResult<u32> {
    let slice = bytes.get(off..off + 4).ok_or_else(|| {
        HostError::invalid_args(format!("device-execute {what} exceeds the mapped file"))
    })?;
    let mut raw = [0u8; 4];
    raw.copy_from_slice(slice);
    Ok(u32::from_le_bytes(raw))
}

fn read_u64_at(bytes: &[u8], off: usize, what: &str) -> HostResult<u64> {
    let slice = bytes.get(off..off + 8).ok_or_else(|| {
        HostError::invalid_args(format!("device-execute {what} exceeds the mapped file"))
    })?;
    let mut raw = [0u8; 8];
    raw.copy_from_slice(slice);
    Ok(u64::from_le_bytes(raw))
}

fn retain_mapped_weights(
    host: &mut CompositeHost,
    mapped: Option<&MappedWeightFile>,
) -> HostResult<()> {
    let Some(mapped) = mapped else {
        return Ok(());
    };
    if let Some(runtime) = host.device_mut() {
        if runtime.supports_mapped_weight_retention() {
            runtime.retain_mapped_weight_file(mapped.clone())?;
        }
    }
    Ok(())
}

fn mapped_paging(mapped: &MappedWeightFile) -> MappedWeightPaging {
    MappedWeightPaging {
        page_size: mapped.page_size() as u64,
        mapped_len: mapped.mapped_len() as u64,
        file_len: mapped.len() as u64,
        rss_bytes: process_resident_bytes(),
    }
}

fn packed_f32_count(len: u64) -> u64 {
    len.div_ceil(4).max(1)
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

fn parse_hex_bytes(spelling: &str) -> Result<Vec<u8>, String> {
    let hex = if spelling.len() >= 2 && spelling.as_bytes()[..2].eq_ignore_ascii_case(b"0x") {
        &spelling[2..]
    } else {
        spelling
    };
    if hex.len() % 2 != 0 {
        return Err(format!("hex payload `{spelling}` is not whole bytes"));
    }
    if !hex.as_bytes().iter().all(u8::is_ascii_hexdigit) {
        return Err(format!("hex payload `{spelling}` is not hex"));
    }
    let mut bytes = Vec::with_capacity(hex.len() / 2);
    for chunk in hex.as_bytes().chunks_exact(2) {
        let text = std::str::from_utf8(chunk).expect("ascii hex digits");
        bytes.push(u8::from_str_radix(text, 16).expect("ascii hex digits"));
    }
    Ok(bytes)
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
    /// Protocol version that produced this receipt.
    #[serde(default = "default_protocol_version")]
    pub protocol: u32,
    /// Explicit model/session identities for a v2 route.
    #[serde(default)]
    pub model_identity: Option<String>,
    #[serde(default)]
    pub session_identity: Option<String>,
    #[serde(default)]
    pub program_identities: BTreeMap<String, String>,
    /// Lifecycle counters for resident control operations.
    #[serde(default)]
    pub lifecycle: DeviceExecuteLifecycle,
    /// Explicit invocation mode when this is a v2 invocation receipt.
    #[serde(default)]
    pub invocation_mode: Option<DeviceExecuteInvocationMode>,
    /// Launches dispatched.
    pub launches: usize,
    /// Descriptor launch identities, in order.
    pub launch_ids: Vec<u32>,
    /// Kernel entries dispatched, in order.
    pub launch_entries: Vec<String>,
    /// Host→device copy-ins.
    pub copy_ins: usize,
    /// Temporary PerStep/ObservationPoint handles allocated by this step.
    #[serde(default)]
    pub pool_allocations: usize,
    /// Temporary handles reused from the session-scoped pool this step.
    #[serde(default)]
    pub pool_reuses: usize,
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
    /// Host command encoding wall copied from the F4H1 steady-state phase.
    #[serde(default)]
    pub encode_us: u64,
    /// Command-buffer or driver submit wall copied from the F4H1 steady-state phase.
    #[serde(default)]
    pub submit_us: u64,
    /// Blocking device wait wall copied from the F4H1 steady-state phase.
    #[serde(default)]
    pub wait_us: u64,
    /// Deprecated fused encode + submit + wait wall. Prefer the phase fields.
    #[serde(default)]
    pub gpu_encode_submit_wait_us: u64,
    /// Per-encoder GPU timestamps in launch order (µs). Empty when unsampled.
    #[serde(default)]
    pub launch_gpu_us: Vec<u64>,
    /// Per-encoder GPU start times in launch order (µs, relative to the
    /// first encoder start). Empty when unsampled.
    #[serde(default)]
    pub launch_gpu_start_us: Vec<u64>,
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
    /// mmap paging facts when `--weights` was mapped rather than copied.
    #[serde(default)]
    pub mmap: MappedWeightPaging,
    /// Gradus `data_start` of the mapped GGUF (0 when the file is not GGUF).
    #[serde(default)]
    pub mmap_data_start: u64,
    /// Number of admitted `abs_starts`/`abs_ends` regions.
    #[serde(default)]
    pub mmap_regions: usize,
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
    /// Project the host receipt onto the CLI wire without an F4H1 phase
    /// projection. This is retained for one-shot callers whose public host
    /// receipt predates the prepared-session timing projection.
    #[must_use]
    pub fn from_host(receipt: &DeviceExecutionReceipt) -> Self {
        Self::from_host_with_phase_timing(receipt, DevicePhaseTiming::default())
    }

    /// Project the host receipt and the F4H1 phase values onto the CLI wire.
    #[must_use]
    fn from_host_with_phase_timing(
        receipt: &DeviceExecutionReceipt,
        timing: DevicePhaseTiming,
    ) -> Self {
        Self {
            backend: receipt.backend.spelling().to_owned(),
            device_name: receipt.device_name.clone(),
            protocol: DeviceExecuteProtocol::V1.version(),
            model_identity: None,
            session_identity: None,
            program_identities: BTreeMap::new(),
            lifecycle: DeviceExecuteLifecycle::default(),
            invocation_mode: None,
            launches: receipt.launches,
            launch_ids: receipt.launch_ids.clone(),
            launch_entries: receipt.launch_entries.clone(),
            copy_ins: receipt.copy_ins,
            pool_allocations: receipt.pool_allocations,
            pool_reuses: receipt.pool_reuses,
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
            encode_us: timing.encode_us,
            submit_us: timing.submit_us,
            wait_us: timing.wait_us,
            gpu_encode_submit_wait_us: receipt.gpu_encode_submit_wait_us,
            launch_gpu_us: receipt.launch_gpu_us.clone(),
            launch_gpu_start_us: receipt.launch_gpu_start_us.clone(),
            readback_us: receipt.readback_us,
            cli_internal_us: 0,
            kernel_count: 0,
            stage_timing: StageTimingReceipt::default(),
            mmap: MappedWeightPaging::default(),
            mmap_data_start: 0,
            mmap_regions: 0,
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

fn parse_protocol(spelling: &str) -> Result<DeviceExecuteProtocol, String> {
    match spelling {
        "1" | "v1" => Ok(DeviceExecuteProtocol::V1),
        "2" | "v2" => Ok(DeviceExecuteProtocol::V2),
        other => Err(format!(
            "device-execute --protocol must be v1 or v2 (got `{other}`)"
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
