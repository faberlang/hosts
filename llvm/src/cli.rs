//! Static CLI descriptor decode + argv parse for the LLVM host (Stage 8 S8.2).
//!
//! The emitted program entry calls [`__faber_rt_v1_cli_parse`] with the
//! compiler-emitted static descriptor (radix `cli_descriptor` byte format v1).
//! This module decodes that descriptor, parses the process argumenta captured
//! at `__faber_rt_v1_init` against it, and returns a typed value table the
//! emitted entry adapter reads with the `__faber_rt_v1_cli_field_*` getters
//! before invoking the selected MIR function (campaign D12).
//!
//! Parse diagnostics mirror the Rust CLI oracle (`radix-hir-rust` cli.rs):
//! parse errors print `error: {message}` to stderr and exit with code 2; help
//! and version output print to stdout and exit 0.

use super::array::{runtime_mut, store_array, RuntimeValue};
use super::format::store_text_owned;
use super::option::store_option;
use super::valor_aggregate::store_octeti;
use super::{RuntimeContext, StableBox};
use crate::abi::{FaberRtPtrResultV1, STATUS_INVALID_ARGUMENT, STATUS_PANIC};
use radix_host_abi::{VALUE_KIND_I1,
    VALUE_KIND_I64, VALUE_KIND_PTR, VALUE_KIND_TEXT,
};
use crate::abi::FaberRtContextV1;
use std::ffi::c_void;
use std::io::Write;
use std::panic::{self, AssertUnwindSafe};

/// CLI descriptor byte-format constants (mirror of radix `cli_descriptor`).
const CLI_DESCRIPTOR_MAGIC: &[u8; 4] = b"FCLI";
const CLI_DESCRIPTOR_VERSION: u8 = 1;

/// CLI value type tags (radix `cli_descriptor::tags`).
const T_TEXTUS: u8 = 0;
const T_NUMERUS: u8 = 1;
const T_FRACTUS: u8 = 2;
const T_BIVALENS: u8 = 3;
const T_OCTETI: u8 = 4;
const T_LISTA_TEXTUS: u8 = 6;
const T_LISTA_NUMERUS: u8 = 7;

const MODE_SUBCOMMAND: u8 = 1;

/// Exit policy tags (radix `cli_descriptor::exit_tags`).
const EXIT_NONE: u8 = 0;
const EXIT_FIXED: u8 = 1;
const EXIT_BINDING: u8 = 2;
const EXIT_FIELD: u8 = 3;
const EXIT_UNSUPPORTED: u8 = 4;

/// Default payload tags (radix `cli_descriptor::default_tags`).
const DEF_TEXT: u8 = 0;
const DEF_INTEGER: u8 = 1;
const DEF_FLOAT: u8 = 2;
const DEF_BOOL: u8 = 3;
const DEF_NIL: u8 = 4;
const DEF_EXPR: u8 = 5;

/// The parse-error process exit code (Rust oracle parity: `cli_parse_error`
/// prints `error: {message}` and `std::process::exit(2)`).
const CLI_PARSE_ERROR_EXIT: i32 = 2;

#[derive(Debug, Clone)]
enum DescriptorExit {
    None,
    Fixed(i64),
    Binding(String),
    /// EXIT_FIELD policy. Only the `field` half carries runtime behavior on
    /// this host; the decoder still consumes the `object` half (v1 byte
    /// contract) but does not retain it.
    Field {
        field: String,
    },
    Unsupported,
}

#[derive(Debug, Clone)]
enum DescriptorDefault {
    Text(String),
    Integer(i64),
    Float(f64),
    Bool(bool),
    Nil,
    Expr(String),
}

#[derive(Debug, Clone)]
struct DescriptorOption {
    binding: String,
    ty: u8,
    short: Option<String>,
    long: Option<String>,
    flag: bool,
    default: Option<DescriptorDefault>,
    description: Option<String>,
}

#[derive(Debug, Clone)]
struct DescriptorOperand {
    binding: String,
    ty: u8,
    rest: bool,
    default: Option<DescriptorDefault>,
    description: Option<String>,
}

#[derive(Debug, Clone)]
struct DescriptorCommand {
    path: Vec<String>,
    aliases: Vec<String>,
    options: Vec<DescriptorOption>,
    operands: Vec<DescriptorOperand>,
    description: Option<String>,
}

#[derive(Debug, Clone)]
struct CliDescriptor {
    name: String,
    mode: u8,
    version: Option<String>,
    description: Option<String>,
    exit: DescriptorExit,
    global_options: Vec<DescriptorOption>,
    global_operands: Vec<DescriptorOperand>,
    options: Vec<DescriptorOption>,
    operands: Vec<DescriptorOperand>,
    commands: Vec<DescriptorCommand>,
}

/// One parsed CLI table entry, shaped for the emitted adapter's typed getter:
/// scalar fields carry the raw scalar; text/list/octeti and optional fields
/// carry a pre-built carrier handle (text handle, option carrier, or array
/// carrier).
#[derive(Clone, Copy)]
enum CliPayload {
    Handle(*mut c_void),
    Integer(i64),
    Float(f64),
    Bool(bool),
}

#[derive(Clone, Copy)]
struct CliEntry {
    kind: u8,
    /// True when the payload is an option carrier (`optio<T>` record field).
    carrier: bool,
    value: CliPayload,
}

pub(super) struct RuntimeCliTable {
    /// Record-order entries (options then operands) for the selected surface.
    entries: Vec<CliEntry>,
    /// Parallel binding names used to resolve exit-policy field references.
    binding_names: Vec<String>,
    exit: DescriptorExit,
    /// Selected command index (subcommand mode) or -1 (single command).
    selected_command: i64,
}

/// Decode + parse argv against the static descriptor, storing the typed value
/// table in the runtime context. Parse errors print the Rust-oracle-shaped
/// diagnostic to stderr and exit with code 2.
///
/// # Safety
///
/// `context` must be live. `descriptor` must point to `descriptor_len` bytes
/// of compiler-emitted descriptor data (radix `cli_descriptor` v1).
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_cli_parse(
    context: *mut FaberRtContextV1,
    descriptor: *const u8,
    descriptor_len: usize,
) -> FaberRtPtrResultV1 {
    ffi_ptr_result(|| {
        if descriptor.is_null() {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        }
        let bytes = std::slice::from_raw_parts(descriptor, descriptor_len);
        let decoded = match decode_descriptor(bytes) {
            Ok(decoded) => decoded,
            Err(reason) => cli_parse_exit(format!("invalid CLI descriptor: {reason}")),
        };
        let Some(runtime) = (unsafe { runtime_mut(context) }) else {
            return FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT);
        };
        let arguments = runtime
            .arguments
            .iter()
            .map(|argument| String::from_utf8_lossy(argument).into_owned())
            .collect::<Vec<_>>();
        let parsed = parse_descriptor(context, &decoded, &arguments);
        let table = match parsed {
            Ok(table) => table,
            Err(message) => cli_parse_exit(message),
        };
        let table = StableBox::new(table);
        let handle = table.handle();
        runtime.cli_table = Some(table);
        FaberRtPtrResultV1::success(handle)
    })
}

/// Return the stored CLI typed value table handle.
///
/// # Safety
///
/// `context` must be live and `__faber_rt_v1_cli_parse` must have run.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_cli_table(context: *mut FaberRtContextV1) -> *mut c_void {
    let Some(runtime) = runtime(context) else {
        return std::ptr::null_mut();
    };
    runtime
        .cli_table
        .as_ref()
        .map_or(std::ptr::null_mut(), |table| table.handle())
}

/// Selected command index (subcommand mode), or -1 (single-command).
///
/// # Safety
///
/// `context` must be live and `__faber_rt_v1_cli_parse` must have run.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_cli_selected_command(context: *mut FaberRtContextV1) -> i64 {
    let Some(runtime) = runtime(context) else {
        return -1;
    };
    runtime
        .cli_table
        .as_ref()
        .map_or(-1, |table| table.selected_command)
}

/// Process exit code derived from the descriptor exit policy + parse table.
///
/// # Safety
///
/// `context` must be live and `__faber_rt_v1_cli_parse` must have run.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_cli_exit_code(context: *mut FaberRtContextV1) -> i64 {
    let Some(runtime) = runtime(context) else {
        return 0;
    };
    let Some(table) = runtime.cli_table.as_ref() else {
        return 0;
    };
    match &table.exit {
        DescriptorExit::None => 0,
        DescriptorExit::Fixed(code) => *code,
        DescriptorExit::Binding(binding) => table_entry_by_binding(table, binding).unwrap_or(0),
        DescriptorExit::Field { field } => table_entry_by_binding(table, field).unwrap_or(0),
        DescriptorExit::Unsupported => 0,
    }
}

/// Extract the field at `index` as a handle (text, list, octeti, or option
/// carrier).
///
/// # Safety
///
/// `context` and `table` must be live; `index` must be an in-range pointer
/// field.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_cli_field_ptr(
    context: *mut FaberRtContextV1,
    table: *mut c_void,
    index: i64,
) -> *mut c_void {
    let Some(runtime) = runtime(context) else {
        return std::ptr::null_mut();
    };
    let Some(entry) = find_entry(runtime, table, index) else {
        return std::ptr::null_mut();
    };
    if !entry.carrier
        && !matches!(
            entry.kind,
            T_TEXTUS | T_OCTETI | T_LISTA_TEXTUS | T_LISTA_NUMERUS
        )
    {
        return std::ptr::null_mut();
    }
    match entry.value {
        CliPayload::Handle(handle) => handle,
        _ => std::ptr::null_mut(),
    }
}

/// Extract the field at `index` as an i64 scalar.
///
/// # Safety
///
/// `context` and `table` must be live; `index` must be an in-range scalar
/// field.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_cli_field_i64(
    context: *mut FaberRtContextV1,
    table: *mut c_void,
    index: i64,
) -> i64 {
    let Some(runtime) = runtime(context) else {
        return 0;
    };
    let Some(entry) = find_entry(runtime, table, index) else {
        return 0;
    };
    match (entry.kind, entry.value) {
        (T_NUMERUS, CliPayload::Integer(value)) => value,
        _ => 0,
    }
}

/// Extract the field at `index` as an f64 scalar.
///
/// # Safety
///
/// `context` and `table` must be live; `index` must be an in-range scalar
/// field.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_cli_field_f64(
    context: *mut FaberRtContextV1,
    table: *mut c_void,
    index: i64,
) -> f64 {
    let Some(runtime) = runtime(context) else {
        return 0.0;
    };
    let Some(entry) = find_entry(runtime, table, index) else {
        return 0.0;
    };
    match (entry.kind, entry.value) {
        (T_FRACTUS, CliPayload::Float(value)) => value,
        _ => 0.0,
    }
}

/// Extract the field at `index` as an i1 boolean scalar (the runtime returns
/// `u8`; LLVM declares the function `i1` — the established bivalens carrier
/// pattern).
///
/// # Safety
///
/// `context` and `table` must be live; `index` must be an in-range scalar
/// field.
#[no_mangle]
pub unsafe extern "C" fn __faber_rt_v1_cli_field_i1(
    context: *mut FaberRtContextV1,
    table: *mut c_void,
    index: i64,
) -> u8 {
    let Some(runtime) = runtime(context) else {
        return 0;
    };
    let Some(entry) = find_entry(runtime, table, index) else {
        return 0;
    };
    match (entry.kind, entry.value) {
        (T_BIVALENS, CliPayload::Bool(value)) => u8::from(value),
        _ => 0,
    }
}

fn find_entry<'a>(
    runtime: &'a RuntimeContext,
    table: *mut c_void,
    index: i64,
) -> Option<&'a CliEntry> {
    let stored = runtime.cli_table.as_ref()?;
    if stored.handle() != table {
        return None;
    }
    let index = usize::try_from(index).ok()?;
    stored.entries.get(index)
}

fn table_entry_by_binding(table: &RuntimeCliTable, binding: &str) -> Option<i64> {
    let binding_index = table
        .binding_names
        .iter()
        .position(|name| name == binding)?;
    match table.entries.get(binding_index).map(|entry| entry.value) {
        Some(CliPayload::Integer(value)) => Some(value),
        Some(CliPayload::Bool(value)) => Some(i64::from(value)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// argv parsing (Rust-oracle parity)
// ---------------------------------------------------------------------------

fn parse_descriptor(
    context: *mut FaberRtContextV1,
    descriptor: &CliDescriptor,
    arguments: &[String],
) -> Result<RuntimeCliTable, String> {
    if descriptor.mode == MODE_SUBCOMMAND {
        parse_subcommand(context, descriptor, arguments)
    } else {
        let options = descriptor
            .global_options
            .iter()
            .chain(descriptor.options.iter())
            .collect::<Vec<_>>();
        let operands = descriptor
            .global_operands
            .iter()
            .chain(descriptor.operands.iter())
            .collect::<Vec<_>>();
        let parsed = parse_surface(context, descriptor, &options, &operands, arguments)?;
        Ok(RuntimeCliTable {
            entries: parsed.entries,
            binding_names: parsed.binding_names,
            exit: descriptor.exit.clone(),
            selected_command: -1,
        })
    }
}

fn parse_subcommand(
    context: *mut FaberRtContextV1,
    descriptor: &CliDescriptor,
    arguments: &[String],
) -> Result<RuntimeCliTable, String> {
    let mut global_entries = Vec::with_capacity(descriptor.global_options.len());
    for option in &descriptor.global_options {
        global_entries.push(initial_option_entry(context, option));
    }
    let mut index = 0;
    let mut command_parts = Vec::new();
    while index < arguments.len() {
        let arg = &arguments[index];
        if arg == "--help" || arg == "-h" {
            print_root_subcommand_help(descriptor);
            std::process::exit(0);
        }
        if descriptor.version.is_some() && arg == "--version" {
            println!("{}", descriptor.version.as_deref().unwrap_or_default());
            std::process::exit(0);
        }
        if arg.starts_with("--") {
            let global_refs = descriptor.global_options.iter().collect::<Vec<_>>();
            parse_long_option(
                context,
                &global_refs,
                &mut global_entries,
                arg,
                &mut index,
                arguments,
            )?;
            index += 1;
            continue;
        }
        if arg.starts_with('-') && arg.len() > 1 {
            let global_refs = descriptor.global_options.iter().collect::<Vec<_>>();
            parse_short_option(
                context,
                &global_refs,
                &mut global_entries,
                arg,
                &mut index,
                arguments,
            )?;
            index += 1;
            continue;
        }
        command_parts.push(arg.clone());
        command_parts.extend(arguments[index + 1..].iter().cloned());
        break;
    }
    if command_parts.is_empty() {
        print_root_subcommand_help(descriptor);
        return Err(String::new());
    }
    let Some((command_index, command)) = select_command(descriptor, &command_parts) else {
        let message = format!("unknown command '{}'", command_parts[0]);
        print_root_subcommand_help(descriptor);
        return Err(message);
    };
    let consumed = command.path.len().min(command_parts.len());
    let command_arguments = command_parts[consumed..].to_vec();
    let options = descriptor
        .global_options
        .iter()
        .chain(command.options.iter())
        .collect::<Vec<_>>();
    let operands = descriptor
        .global_operands
        .iter()
        .chain(command.operands.iter())
        .collect::<Vec<_>>();
    // Global entries already carry the dispatcher-level decisions; command
    // local options start at their defaults.
    let mut option_entries = global_entries;
    for option in &command.options {
        option_entries.push(initial_option_entry(context, option));
    }
    let parsed = parse_surface_with_entries(
        context,
        descriptor,
        &options,
        &operands,
        &command_arguments,
        &mut option_entries,
    )?;
    // The parsed surface already carries the full command record: the global
    // option decisions from the dispatcher (the command parse may additionally
    // re-see a global option after the command name, matching the oracle) plus
    // command-local options and the merged operand list.
    Ok(RuntimeCliTable {
        entries: parsed.entries,
        binding_names: parsed.binding_names,
        exit: descriptor.exit.clone(),
        selected_command: command_index as i64,
    })
}

fn select_command<'a>(
    descriptor: &'a CliDescriptor,
    command_parts: &[String],
) -> Option<(usize, &'a DescriptorCommand)> {
    let mut best: Option<(usize, usize, &DescriptorCommand)> = None;
    for (index, command) in descriptor.commands.iter().enumerate() {
        if command_parts.len() >= command.path.len()
            && command.path.iter().enumerate().all(|(part_index, part)| {
                command_parts
                    .get(part_index)
                    .is_some_and(|value| value == part)
            })
        {
            let candidate = (index, command.path.len(), command);
            best = match best {
                None => Some(candidate),
                Some(current) if candidate.1 > current.1 => Some(candidate),
                _ => best,
            };
        }
        for alias in &command.aliases {
            let alias_parts = alias_path(alias);
            if command_parts.len() >= alias_parts.len()
                && alias_parts.iter().enumerate().all(|(part_index, part)| {
                    command_parts
                        .get(part_index)
                        .is_some_and(|value| value == part)
                })
            {
                let candidate = (index, alias_parts.len(), command);
                best = match best {
                    None => Some(candidate),
                    Some(current) if candidate.1 > current.1 => Some(candidate),
                    _ => best,
                };
            }
        }
    }
    best.map(|(index, _, command)| (index, command))
}

fn alias_path(alias: &str) -> Vec<&str> {
    alias.split('/').filter(|part| !part.is_empty()).collect()
}

struct ParsedSurface {
    entries: Vec<CliEntry>,
    binding_names: Vec<String>,
}

fn parse_surface(
    context: *mut FaberRtContextV1,
    descriptor: &CliDescriptor,
    options: &[&DescriptorOption],
    operands: &[&DescriptorOperand],
    arguments: &[String],
) -> Result<ParsedSurface, String> {
    let mut option_entries = Vec::with_capacity(options.len());
    for option in options {
        option_entries.push(initial_option_entry(context, option));
    }
    parse_surface_with_entries(
        context,
        descriptor,
        options,
        operands,
        arguments,
        &mut option_entries,
    )
}

fn parse_surface_with_entries(
    context: *mut FaberRtContextV1,
    descriptor: &CliDescriptor,
    options: &[&DescriptorOption],
    operands: &[&DescriptorOperand],
    arguments: &[String],
    option_entries: &mut [CliEntry],
) -> Result<ParsedSurface, String> {
    let mut positionals = Vec::new();
    let mut index = 0;
    while index < arguments.len() {
        let arg = &arguments[index];
        if arg == "--" {
            positionals.extend(arguments[index + 1..].iter().cloned());
            break;
        }
        if arg == "--help" || arg == "-h" {
            print_surface_help(descriptor, options, operands);
            std::process::exit(0);
        }
        if descriptor.version.is_some() && arg == "--version" {
            println!("{}", descriptor.version.as_deref().unwrap_or_default());
            std::process::exit(0);
        }
        if arg.starts_with("--") {
            parse_long_option(context, options, option_entries, arg, &mut index, arguments)?;
            index += 1;
            continue;
        }
        if arg.starts_with('-') && arg.len() > 1 {
            parse_short_option(context, options, option_entries, arg, &mut index, arguments)?;
            index += 1;
            continue;
        }
        positionals.push(arg.clone());
        index += 1;
    }
    let operand_entries = assign_operands(context, operands, &positionals)?;
    let mut entries = Vec::with_capacity(options.len() + operands.len());
    let mut binding_names = Vec::with_capacity(options.len() + operands.len());
    for (index, option) in options.iter().enumerate() {
        entries.push(option_entries[index]);
        binding_names.push(option.binding.clone());
    }
    for (index, operand) in operands.iter().enumerate() {
        entries.push(operand_entries[index]);
        binding_names.push(operand.binding.clone());
    }
    Ok(ParsedSurface {
        entries,
        binding_names,
    })
}

fn initial_option_entry(context: *mut FaberRtContextV1, option: &DescriptorOption) -> CliEntry {
    if option.flag {
        let enabled = matches!(&option.default, Some(DescriptorDefault::Bool(true)));
        return CliEntry {
            kind: option.ty,
            carrier: false,
            value: CliPayload::Bool(enabled),
        };
    }
    match &option.default {
        Some(default) => default_value_entry(context, option.ty, default),
        None => CliEntry {
            kind: option.ty,
            carrier: true,
            value: CliPayload::Handle(option_none_carrier(context, option.ty)),
        },
    }
}

fn option_none_carrier(context: *mut FaberRtContextV1, ty: u8) -> *mut c_void {
    let Some(runtime) = (unsafe { runtime_mut(context) }) else {
        return std::ptr::null_mut();
    };
    let kind = option_value_kind(ty);
    let result = store_option(runtime, kind, None);
    if result.status.is_ok() {
        result.value
    } else {
        std::ptr::null_mut()
    }
}

fn default_value_entry(
    context: *mut FaberRtContextV1,
    ty: u8,
    default: &DescriptorDefault,
) -> CliEntry {
    match (ty, default) {
        (T_NUMERUS, DescriptorDefault::Integer(value)) => CliEntry {
            kind: ty,
            carrier: false,
            value: CliPayload::Integer(*value),
        },
        (T_FRACTUS, DescriptorDefault::Float(value)) => CliEntry {
            kind: ty,
            carrier: false,
            value: CliPayload::Float(*value),
        },
        (T_BIVALENS, DescriptorDefault::Bool(value)) => CliEntry {
            kind: ty,
            carrier: false,
            value: CliPayload::Bool(*value),
        },
        (T_OCTETI, DescriptorDefault::Text(value)) => CliEntry {
            kind: ty,
            carrier: false,
            value: CliPayload::Handle(octeti_handle(context, value.as_bytes())),
        },
        (_, DescriptorDefault::Text(value)) | (_, DescriptorDefault::Expr(value)) => CliEntry {
            kind: ty,
            carrier: false,
            value: CliPayload::Handle(text_handle(context, value)),
        },
        (T_NUMERUS, _) => CliEntry {
            kind: ty,
            carrier: false,
            value: CliPayload::Integer(0),
        },
        (T_FRACTUS, _) => CliEntry {
            kind: ty,
            carrier: false,
            value: CliPayload::Float(0.0),
        },
        (T_BIVALENS, _) => CliEntry {
            kind: ty,
            carrier: false,
            value: CliPayload::Bool(false),
        },
        (_, DescriptorDefault::Nil) => CliEntry {
            kind: ty,
            carrier: false,
            value: CliPayload::Handle(match ty {
                T_OCTETI => octeti_handle(context, &[]),
                _ => text_handle(context, ""),
            }),
        },
        (_, _) => CliEntry {
            kind: ty,
            carrier: false,
            value: CliPayload::Integer(0),
        },
    }
}

fn parse_long_option(
    context: *mut FaberRtContextV1,
    options: &[&DescriptorOption],
    option_entries: &mut [CliEntry],
    arg: &str,
    index: &mut usize,
    arguments: &[String],
) -> Result<(), String> {
    let (name, inline) = match arg.split_once('=') {
        Some((name, value)) => (name, Some(value.to_owned())),
        None => (arg, None),
    };
    for (option_index, option) in options.iter().enumerate() {
        if option
            .long
            .as_deref()
            .is_some_and(|long| name == format!("--{long}"))
        {
            apply_option(
                context,
                option,
                option_index,
                option_entries,
                &name,
                inline,
                index,
                arguments,
            )?;
            return Ok(());
        }
    }
    Err(format!("unknown option '{arg}'"))
}

fn parse_short_option(
    context: *mut FaberRtContextV1,
    options: &[&DescriptorOption],
    option_entries: &mut [CliEntry],
    arg: &str,
    index: &mut usize,
    arguments: &[String],
) -> Result<(), String> {
    for (option_index, option) in options.iter().enumerate() {
        if option
            .short
            .as_deref()
            .is_some_and(|short| arg == format!("-{short}"))
        {
            apply_option(
                context,
                option,
                option_index,
                option_entries,
                arg,
                None,
                index,
                arguments,
            )?;
            return Ok(());
        }
    }
    Err(format!("unknown option '{arg}'"))
}

fn apply_option(
    context: *mut FaberRtContextV1,
    option: &DescriptorOption,
    option_index: usize,
    option_entries: &mut [CliEntry],
    label: &str,
    inline: Option<String>,
    index: &mut usize,
    arguments: &[String],
) -> Result<(), String> {
    if option.flag {
        option_entries[option_index] = CliEntry {
            kind: option.ty,
            carrier: false,
            value: CliPayload::Bool(true),
        };
        return Ok(());
    }
    let raw = match inline {
        Some(value) => value,
        None => {
            *index += 1;
            arguments
                .get(*index)
                .cloned()
                .ok_or_else(|| format!("missing value for {label}"))?
        }
    };
    option_entries[option_index] = parse_option_value(context, option, &raw)?;
    Ok(())
}

fn parse_option_value(
    context: *mut FaberRtContextV1,
    option: &DescriptorOption,
    raw: &str,
) -> Result<CliEntry, String> {
    let optional = option.default.is_none();
    let value = match option.ty {
        T_NUMERUS => raw
            .parse::<i64>()
            .map(CliPayload::Integer)
            .map_err(|_| format!("invalid numeric value '{raw}'"))?,
        T_FRACTUS => raw
            .parse::<f64>()
            .map(CliPayload::Float)
            .map_err(|_| format!("invalid numeric value '{raw}'"))?,
        T_BIVALENS => raw
            .parse::<bool>()
            .map(CliPayload::Bool)
            .map_err(|_| format!("invalid boolean value '{raw}'"))?,
        T_OCTETI => CliPayload::Handle(octeti_handle(context, raw.as_bytes())),
        _ => CliPayload::Handle(text_handle(context, raw)),
    };
    if optional {
        Ok(CliEntry {
            kind: option.ty,
            carrier: true,
            value: CliPayload::Handle(option_some_carrier(context, option.ty, value)?),
        })
    } else {
        Ok(CliEntry {
            kind: option.ty,
            carrier: false,
            value,
        })
    }
}

fn assign_operands(
    context: *mut FaberRtContextV1,
    operands: &[&DescriptorOperand],
    positionals: &[String],
) -> Result<Vec<CliEntry>, String> {
    let mut entries = Vec::with_capacity(operands.len());
    let mut positional_iter = positionals.iter();
    let has_rest = operands.iter().any(|operand| operand.rest);
    for operand in operands {
        if operand.rest {
            let mut values = Vec::new();
            for raw in positional_iter.by_ref() {
                values.push(parse_operand_payload(context, operand, raw)?);
            }
            entries.push(list_value_entry(context, operand.ty, values)?);
            continue;
        }
        let value = match positional_iter.next() {
            Some(raw) => parse_operand_payload(context, operand, raw)?,
            None => match &operand.default {
                Some(default) => default_value_payload(context, operand.ty, default),
                None => return Err(format!("missing operand '{}'", operand.binding)),
            },
        };
        entries.push(entry_from_payload(context, operand.ty, value));
    }
    if !has_rest {
        if let Some(extra) = positional_iter.next() {
            return Err(format!("unexpected operand '{extra}'"));
        }
    }
    Ok(entries)
}

fn parse_operand_payload(
    context: *mut FaberRtContextV1,
    operand: &DescriptorOperand,
    raw: &str,
) -> Result<CliPayload, String> {
    match operand.ty {
        T_NUMERUS => raw
            .parse::<i64>()
            .map(CliPayload::Integer)
            .map_err(|_| format!("invalid numeric value '{raw}'")),
        T_FRACTUS => raw
            .parse::<f64>()
            .map(CliPayload::Float)
            .map_err(|_| format!("invalid numeric value '{raw}'")),
        T_BIVALENS => raw
            .parse::<bool>()
            .map(CliPayload::Bool)
            .map_err(|_| format!("invalid boolean value '{raw}'")),
        T_OCTETI => Ok(CliPayload::Handle(octeti_handle(context, raw.as_bytes()))),
        T_LISTA_NUMERUS => raw
            .parse::<i64>()
            .map(CliPayload::Integer)
            .map_err(|_| format!("invalid numeric value '{raw}'")),
        _ => Ok(CliPayload::Handle(text_handle(context, raw))),
    }
}

fn entry_from_payload(_context: *mut FaberRtContextV1, ty: u8, value: CliPayload) -> CliEntry {
    CliEntry {
        kind: ty,
        carrier: false,
        value,
    }
}

fn list_value_entry(
    context: *mut FaberRtContextV1,
    ty: u8,
    values: Vec<CliPayload>,
) -> Result<CliEntry, String> {
    let Some(runtime) = (unsafe { runtime_mut(context) }) else {
        return Err(String::new());
    };
    let (kind, runtime_values) = match ty {
        T_LISTA_NUMERUS => {
            let values = values
                .into_iter()
                .map(|value| match value {
                    CliPayload::Integer(value) => RuntimeValue::I64(value),
                    _ => RuntimeValue::I64(0),
                })
                .collect();
            (VALUE_KIND_I64, values)
        }
        _ => {
            let values = values
                .into_iter()
                .map(|value| match value {
                    CliPayload::Handle(handle) => RuntimeValue::Ptr(handle),
                    _ => RuntimeValue::Ptr(std::ptr::null_mut()),
                })
                .collect();
            (VALUE_KIND_PTR, values)
        }
    };
    let result = store_array(runtime, kind, runtime_values);
    if result.status.is_ok() {
        Ok(CliEntry {
            kind: ty,
            carrier: false,
            value: CliPayload::Handle(result.value),
        })
    } else {
        Err(String::new())
    }
}

fn default_value_payload(
    context: *mut FaberRtContextV1,
    ty: u8,
    default: &DescriptorDefault,
) -> CliPayload {
    match (ty, default) {
        (T_NUMERUS, DescriptorDefault::Integer(value)) => CliPayload::Integer(*value),
        (T_FRACTUS, DescriptorDefault::Float(value)) => CliPayload::Float(*value),
        (T_BIVALENS, DescriptorDefault::Bool(value)) => CliPayload::Bool(*value),
        (T_OCTETI, DescriptorDefault::Text(value)) => {
            CliPayload::Handle(octeti_handle(context, value.as_bytes()))
        }
        (_, DescriptorDefault::Text(value)) => CliPayload::Handle(text_handle(context, value)),
        _ => match ty {
            T_NUMERUS => CliPayload::Integer(0),
            T_FRACTUS => CliPayload::Float(0.0),
            T_BIVALENS => CliPayload::Bool(false),
            _ => CliPayload::Handle(text_handle(context, "")),
        },
    }
}

/// The option-carrier value kind for a CLI type tag (mirrors the emitter's
/// `runtime_value_abi` mapping for scalar/text/list fields).
fn option_value_kind(ty: u8) -> radix_host_abi::FaberRtValueKindV1 {
    match ty {
        T_NUMERUS => VALUE_KIND_I64,
        T_FRACTUS => radix_host_abi::VALUE_KIND_F64,
        T_BIVALENS => VALUE_KIND_I1,
        T_OCTETI | T_LISTA_TEXTUS | T_LISTA_NUMERUS => VALUE_KIND_PTR,
        _ => VALUE_KIND_TEXT,
    }
}

fn option_some_carrier(
    context: *mut FaberRtContextV1,
    ty: u8,
    payload: CliPayload,
) -> Result<*mut c_void, String> {
    let Some(runtime) = (unsafe { runtime_mut(context) }) else {
        return Err(String::new());
    };
    let kind = option_value_kind(ty);
    let value = match payload {
        CliPayload::Integer(value) => RuntimeValue::I64(value),
        CliPayload::Float(value) => RuntimeValue::F64(value),
        CliPayload::Bool(value) => RuntimeValue::I1(u8::from(value)),
        CliPayload::Handle(handle) => RuntimeValue::Ptr(handle),
    };
    let result = store_option(runtime, kind, Some(value));
    if result.status.is_ok() {
        Ok(result.value)
    } else {
        Err(String::new())
    }
}

fn text_handle(context: *mut FaberRtContextV1, value: &str) -> *mut c_void {
    // SAFETY: `context` is live on this path; the returned handle lives in the
    // context arena for the process lifetime.
    unsafe { store_text_owned(context, value.to_owned()) }
}

fn octeti_handle(context: *mut FaberRtContextV1, bytes: &[u8]) -> *mut c_void {
    let Some(runtime) = (unsafe { runtime_mut(context) }) else {
        return std::ptr::null_mut();
    };
    let result = store_octeti(runtime, bytes.to_vec());
    if result.status.is_ok() {
        result.value
    } else {
        std::ptr::null_mut()
    }
}

// ---------------------------------------------------------------------------
// help/version printing (Rust-oracle format parity)
// ---------------------------------------------------------------------------

fn print_surface_help(
    descriptor: &CliDescriptor,
    options: &[&DescriptorOption],
    operands: &[&DescriptorOperand],
) {
    println!(
        "Usage: {}{}",
        descriptor.name,
        usage_suffix(options, operands)
    );
    if let Some(description) = &descriptor.description {
        println!();
        println!("{description}");
    }
    if !options.is_empty() {
        println!();
        println!("Options:");
        for option in options {
            println!("  {:<24}{}", option_label(option), option.description());
        }
    }
    if !operands.is_empty() {
        println!();
        println!("Operands:");
        for operand in operands {
            println!("  {:<24}{}", operand_label(operand), operand.description());
        }
    }
    println!();
    println!("  -h, --help              Print help");
    if descriptor.version.is_some() {
        println!("      --version           Print version");
    }
}

fn print_root_subcommand_help(descriptor: &CliDescriptor) {
    println!("Usage: {} [OPTIONS] <COMMAND>", descriptor.name);
    if let Some(description) = &descriptor.description {
        println!();
        println!("{description}");
    }
    if !descriptor.global_options.is_empty() {
        println!();
        println!("Global Options:");
        for option in &descriptor.global_options {
            println!("  {:<24}{}", option_label(option), option.description());
        }
    }
    println!();
    println!("Commands:");
    for command in &descriptor.commands {
        let path = command.path.join(" ");
        let aliases = if command.aliases.is_empty() {
            String::new()
        } else {
            format!(" (alias: {})", command.aliases.join(", "))
        };
        println!("  {:<24}{}{}", path, command.description(), aliases);
    }
    println!();
    println!("  -h, --help              Print help");
    if descriptor.version.is_some() {
        println!("      --version           Print version");
    }
}

fn usage_suffix(options: &[&DescriptorOption], operands: &[&DescriptorOperand]) -> String {
    let mut parts = Vec::new();
    if !options.is_empty() {
        parts.push("[OPTIONS]".to_owned());
    }
    for operand in operands {
        if operand.rest {
            parts.push(format!("[{}...]", operand.binding));
        } else if operand.default.is_some() {
            parts.push(format!("[{}]", operand.binding));
        } else {
            parts.push(format!("<{}>", operand.binding));
        }
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!(" {}", parts.join(" "))
    }
}

fn option_label(option: &DescriptorOption) -> String {
    let mut names = Vec::new();
    if let Some(short) = &option.short {
        names.push(format!("-{short}"));
    }
    if let Some(long) = &option.long {
        names.push(format!("--{long}"));
    }
    let mut label = names.join(", ");
    if !option.flag {
        label.push(' ');
        label.push_str(value_name(option.ty));
    }
    label
}

fn operand_label(operand: &DescriptorOperand) -> String {
    if operand.rest {
        format!("{}...", operand.binding)
    } else {
        operand.binding.clone()
    }
}

fn value_name(ty: u8) -> &'static str {
    match ty {
        T_NUMERUS | T_LISTA_NUMERUS => "<NUMERUS>",
        T_FRACTUS => "<FRACTUS>",
        T_BIVALENS => "<BIVALENS>",
        T_OCTETI => "<OCTETI>",
        _ => "<TEXTUS>",
    }
}

impl DescriptorOption {
    fn description(&self) -> String {
        self.description.clone().unwrap_or_default()
    }
}

impl DescriptorOperand {
    fn description(&self) -> String {
        self.description.clone().unwrap_or_default()
    }
}

impl DescriptorCommand {
    fn description(&self) -> String {
        self.description.clone().unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// descriptor decode (radix `cli_descriptor` v1 mirror)
// ---------------------------------------------------------------------------

struct DescrReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

fn decode_descriptor(bytes: &[u8]) -> Result<CliDescriptor, String> {
    let mut reader = DescrReader::new(bytes)?;
    if reader.bytes(4)? != CLI_DESCRIPTOR_MAGIC {
        return Err("bad magic".to_owned());
    }
    let version = reader.u8()?;
    if version != CLI_DESCRIPTOR_VERSION {
        return Err(format!("unsupported version {version}"));
    }
    let mode = reader.u8()?;
    let has_version = reader.u8()? != 0;
    let has_description = reader.u8()? != 0;
    let name = reader.string()?;
    let version = if has_version {
        Some(reader.string()?)
    } else {
        None
    };
    let description = if has_description {
        Some(reader.string()?)
    } else {
        None
    };
    let exit = decode_exit(&mut reader)?;
    let global_options = decode_options(&mut reader)?;
    let global_operands = decode_operands(&mut reader)?;
    let options = decode_options(&mut reader)?;
    let operands = decode_operands(&mut reader)?;
    let command_count = reader.u16()? as usize;
    let mut commands = Vec::with_capacity(command_count);
    for _ in 0..command_count {
        commands.push(decode_command(&mut reader)?);
    }
    Ok(CliDescriptor {
        name,
        mode,
        version,
        description,
        exit,
        global_options,
        global_operands,
        options,
        operands,
        commands,
    })
}

fn decode_exit(reader: &mut DescrReader<'_>) -> Result<DescriptorExit, String> {
    match reader.u8()? {
        EXIT_NONE => Ok(DescriptorExit::None),
        EXIT_FIXED => Ok(DescriptorExit::Fixed(reader.i64()?)),
        EXIT_BINDING => Ok(DescriptorExit::Binding(reader.string()?)),
        EXIT_FIELD => {
            // The `object` half of the EXIT_FIELD policy is part of the v1
            // descriptor byte contract but carries no runtime behavior on
            // this host; consume it to keep the reader aligned.
            reader.string()?;
            Ok(DescriptorExit::Field {
                field: reader.string()?,
            })
        }
        EXIT_UNSUPPORTED => Ok(DescriptorExit::Unsupported),
        other => Err(format!("unknown exit tag {other}")),
    }
}

fn decode_options(reader: &mut DescrReader<'_>) -> Result<Vec<DescriptorOption>, String> {
    let count = reader.u16()? as usize;
    let mut options = Vec::with_capacity(count);
    for _ in 0..count {
        options.push(DescriptorOption {
            ty: reader.u8()?,
            flag: reader.u8()? != 0,
            short: reader.opt_string()?,
            long: reader.opt_string()?,
            description: reader.opt_string()?,
            default: reader.opt_default()?,
            binding: reader.string()?,
        });
    }
    Ok(options)
}

fn decode_operands(reader: &mut DescrReader<'_>) -> Result<Vec<DescriptorOperand>, String> {
    let count = reader.u16()? as usize;
    let mut operands = Vec::with_capacity(count);
    for _ in 0..count {
        operands.push(DescriptorOperand {
            ty: reader.u8()?,
            rest: reader.u8()? != 0,
            description: reader.opt_string()?,
            default: reader.opt_default()?,
            binding: reader.string()?,
        });
    }
    Ok(operands)
}

fn decode_command(reader: &mut DescrReader<'_>) -> Result<DescriptorCommand, String> {
    let path_len = reader.u8()? as usize;
    let mut path = Vec::with_capacity(path_len);
    for _ in 0..path_len {
        path.push(reader.string()?);
    }
    let alias_count = reader.u8()? as usize;
    let mut aliases = Vec::with_capacity(alias_count);
    for _ in 0..alias_count {
        aliases.push(reader.string()?);
    }
    let description = reader.opt_string()?;
    // The v1 command record also carries the target `function` identity and
    // an `args_binding`, but neither drives runtime behavior on this host;
    // consume both to keep the reader aligned for options/operands.
    reader.string()?;
    reader.opt_string()?;
    let options = decode_options(reader)?;
    let operands = decode_operands(reader)?;
    Ok(DescriptorCommand {
        path,
        aliases,
        options,
        operands,
        description,
    })
}

impl<'a> DescrReader<'a> {
    fn new(bytes: &'a [u8]) -> Result<Self, String> {
        if bytes.len() < 5 {
            return Err("truncated descriptor".to_owned());
        }
        Ok(Self { bytes, offset: 0 })
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], String> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| "descriptor overflow".to_owned())?;
        if end > self.bytes.len() {
            return Err("truncated descriptor".to_owned());
        }
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn bytes(&mut self, len: usize) -> Result<&'a [u8], String> {
        self.take(len)
    }

    fn u16(&mut self) -> Result<u16, String> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn i64(&mut self) -> Result<i64, String> {
        let bytes = self.take(8)?;
        Ok(i64::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String, String> {
        let len = self.u16()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| "non-UTF-8 descriptor string".to_owned())
    }

    fn opt_string(&mut self) -> Result<Option<String>, String> {
        if self.u8()? == 0 {
            Ok(None)
        } else {
            self.string().map(Some)
        }
    }

    fn opt_default(&mut self) -> Result<Option<DescriptorDefault>, String> {
        if self.u8()? == 0 {
            return Ok(None);
        }
        let default = match self.u8()? {
            DEF_TEXT => DescriptorDefault::Text(self.string()?),
            DEF_EXPR => DescriptorDefault::Expr(self.string()?),
            DEF_INTEGER => DescriptorDefault::Integer(self.i64()?),
            DEF_FLOAT => {
                DescriptorDefault::Float(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
            }
            DEF_BOOL => DescriptorDefault::Bool(self.u8()? != 0),
            DEF_NIL => DescriptorDefault::Nil,
            other => return Err(format!("unknown default tag {other}")),
        };
        Ok(Some(default))
    }
}

/// Parse failure path: print the Rust-oracle-shaped `error: {message}` line
/// to stderr and exit with the oracle's parse-error code 2. An empty message
/// (the no-command subcommand case) prints no error line — the oracle prints
/// only the help before exiting 2.
fn cli_parse_exit(message: String) -> ! {
    let mut stderr = std::io::stderr().lock();
    if !message.is_empty() {
        drop(stderr.write_all(format!("error: {message}\n").as_bytes()));
        drop(stderr.flush());
    }
    std::process::exit(CLI_PARSE_ERROR_EXIT);
}

fn ffi_ptr_result(operation: impl FnOnce() -> FaberRtPtrResultV1) -> FaberRtPtrResultV1 {
    panic::catch_unwind(AssertUnwindSafe(operation))
        .unwrap_or(FaberRtPtrResultV1::failure(STATUS_PANIC))
}

fn runtime(context: *mut FaberRtContextV1) -> Option<&'static RuntimeContext> {
    (!context.is_null()).then(|| unsafe { &*context.cast::<RuntimeContext>() })
}
