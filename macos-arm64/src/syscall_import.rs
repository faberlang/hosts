//! Shared Wasm import routing for Faber capability calls.
//!
//! TARGET: Generated Rust currently lowers `ad` calls to a tiny route-code ABI.
//! This module keeps that temporary ABI aligned for core Wasm modules and
//! Component Model wrappers while both paths route through the same frame
//! kernel internally.

use faber::Valor;

use crate::kernel::frame_data;
use crate::{Frame, HostError, HostKernel};

pub const CAPABILITY_CALL_IMPORT: &str = "capability-call";
pub const COMPONENT_CODE_HOST_ECHO: u32 = 1;
pub const COMPONENT_CODE_PG_QUERY: u32 = 2;

pub fn route_capability_code(kernel: &HostKernel, route_code: i32) -> Frame {
    let (call, data) = match route_code {
        code if code == COMPONENT_CODE_HOST_ECHO as i32 => (
            "host:echo",
            frame_data::tabula([("value", Valor::Textus("salve".into()))]),
        ),
        code if code == COMPONENT_CODE_PG_QUERY as i32 => ("pg:query", frame_data::empty()),
        other => {
            let request = Frame::request("host:unknown").with_from("wasm");
            return request.error(&HostError::invalid_args(format!(
                "unknown capability route code: {other}"
            )));
        }
    };
    let request = Frame::request_with(call, data).with_from("wasm");
    kernel.route(&request)
}
