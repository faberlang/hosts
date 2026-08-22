use std::env;
use std::process::ExitCode;

use faber_host_macos_arm64::component::ComponentHost;
use faber_host_macos_arm64::kernel::frame_data;
use faber_host_macos_arm64::kernel::valor_wire;
use faber_host_macos_arm64::wasm::WasmHost;
use faber_host_macos_arm64::{Frame, HostKernel, Status};

fn main() -> ExitCode {
    match run(env::args().skip(1).collect()) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(64)
        }
    }
}

fn run(args: Vec<String>) -> Result<ExitCode, String> {
    let Some(command) = args.first().map(String::as_str) else {
        print_usage();
        return Ok(ExitCode::SUCCESS);
    };

    match command {
        "manifest" => print_manifest(),
        "call" => call(&args[1..]),
        "wasm-call" => wasm_call(&args[1..]),
        "component-call" => component_call(&args[1..]),
        "device-execute" => device_execute(&args[1..]),
        "help" | "-h" | "--help" => {
            print_usage();
            Ok(ExitCode::SUCCESS)
        }
        other => Err(format!("unknown command: {other}")),
    }
}

fn print_manifest() -> Result<ExitCode, String> {
    let kernel = HostKernel::new();
    print_json(&kernel.manifest())?;
    Ok(ExitCode::SUCCESS)
}

fn call(args: &[String]) -> Result<ExitCode, String> {
    let Some(call) = args.first() else {
        return Err("usage: faber-host-macos-arm64 call <name> [json-object]".into());
    };

    let data = match args.get(1) {
        Some(raw) => valor_wire::parse_json_object(raw)?,
        None => frame_data::empty(),
    };

    let kernel = HostKernel::new();
    let request = Frame::request_with(call, data).with_from("cli");
    let response = kernel.route(&request);
    let status = response.status;
    print_json(&response)?;

    if status == Status::Error {
        Ok(ExitCode::from(2))
    } else {
        Ok(ExitCode::SUCCESS)
    }
}

fn component_call(args: &[String]) -> Result<ExitCode, String> {
    let [path, export_name, route_code] = args else {
        return Err(
            "usage: faber-host-macos-arm64 component-call <component> <export> <route-code>".into(),
        );
    };

    let route_code = route_code
        .parse::<u32>()
        .map_err(|error| format!("component route code must be a u32: {error}"))?;
    let host =
        ComponentHost::new().map_err(|error| format!("component host init failed: {error}"))?;
    let output = host
        .call_export_from_file(path, export_name, route_code)
        .map_err(|error| format!("component call failed: {error}"))?;
    print_json(&output.response)?;

    if output.response.status == Status::Error {
        Ok(ExitCode::from(2))
    } else {
        Ok(ExitCode::SUCCESS)
    }
}

fn wasm_call(args: &[String]) -> Result<ExitCode, String> {
    let [path, export_name, route_code] = args else {
        return Err(
            "usage: faber-host-macos-arm64 wasm-call <module> <export> <route-code>".into(),
        );
    };

    let route_code = route_code
        .parse::<u32>()
        .map_err(|error| format!("wasm route code must be a u32: {error}"))?;
    let host = WasmHost::new().map_err(|error| format!("wasm host init failed: {error}"))?;
    let output = host
        .call_export_from_file(path, export_name, route_code)
        .map_err(|error| format!("wasm call failed: {error}"))?;
    print_json(&output.response)?;

    if output.response.status == Status::Error {
        Ok(ExitCode::from(2))
    } else {
        Ok(ExitCode::SUCCESS)
    }
}

fn print_json(value: &impl serde::Serialize) -> Result<(), String> {
    serde_json::to_writer_pretty(std::io::stdout(), value)
        .map_err(|error| format!("failed to write JSON: {error}"))?;
    println!();
    Ok(())
}

fn device_execute(args: &[String]) -> Result<ExitCode, String> {
    let parsed = faber_host_macos_arm64::device_execute::parse_device_execute_args(args)?;
    if parsed.distributed_image.is_some() {
        return match faber_host_macos_arm64::device_execute::run_distributed_prepare(&parsed) {
            Ok(receipt) => {
                let json =
                    faber_host_macos_arm64::device_execute::distributed_prepare_receipt_to_json(
                        &receipt,
                    )
                    .map_err(|error| error.to_string())?;
                std::io::Write::write_all(&mut std::io::stdout(), &json)
                    .map_err(|error| format!("failed to write JSON: {error}"))?;
                println!();
                Ok(ExitCode::SUCCESS)
            }
            Err(error) => {
                eprintln!("{error}");
                print_json(&error)?;
                Ok(ExitCode::from(2))
            }
        };
    }
    if parsed.control {
        return match faber_host_macos_arm64::device_execute::run_device_execute_control(&parsed) {
            Ok(()) => Ok(ExitCode::SUCCESS),
            Err(error) => {
                print_json(&error)?;
                Ok(ExitCode::from(2))
            }
        };
    }
    match faber_host_macos_arm64::device_execute::run_device_execute(&parsed) {
        Ok(receipt) => {
            let json = faber_host_macos_arm64::device_execute::receipt_to_json(&receipt)
                .map_err(|error| error.to_string())?;
            std::io::Write::write_all(&mut std::io::stdout(), &json)
                .map_err(|error| format!("failed to write JSON: {error}"))?;
            println!();
            Ok(ExitCode::SUCCESS)
        }
        Err(error) => {
            eprintln!("{error}");
            print_json(&error)?;
            Ok(ExitCode::from(2))
        }
    }
}

fn print_usage() {
    println!("usage:");
    println!("  faber-host-macos-arm64 manifest");
    println!("  faber-host-macos-arm64 call <name> [json-object]");
    println!("  faber-host-macos-arm64 wasm-call <module> <export> <route-code>");
    println!("  faber-host-macos-arm64 component-call <component> <export> <route-code>");
    println!(
        "  faber-host-macos-arm64 device-execute [--control] [--backend auto|metal|cuda] --descriptor <json> --module <bin> --inputs <json> [--weights <gguf> --weight-map <json>]"
    );
    println!(
        "  faber-host-macos-arm64 device-execute [--backend auto|metal|cuda] --distributed-image <postcard> --bind-count <n>"
    );
}
