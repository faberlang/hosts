use crate::kernel::frame_data;
use crate::kernel::{Frame, FrameData, HostError, HostResult};

/// Describes a syscall exposed by the host manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyscallInfo {
    pub name: String,
    pub prefix: String,
    pub summary: String,
}

impl SyscallInfo {
    pub fn new(
        name: impl Into<String>,
        prefix: impl Into<String>,
        summary: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            prefix: prefix.into(),
            summary: summary.into(),
        }
    }
}

/// Built-in host syscall handler.
///
/// Handlers own a prefix such as `host` or `fs`. The router dispatches by
/// prefix first, while each handler remains responsible for validating exact
/// call names under that prefix.
pub trait Syscall: Send + Sync {
    fn prefix(&self) -> &'static str;

    fn syscalls(&self) -> Vec<SyscallInfo>;

    fn dispatch(&self, request: &Frame) -> HostResult<Frame>;
}

/// Minimal built-in host namespace used to prove routing.
pub struct HostEcho;

impl Syscall for HostEcho {
    fn prefix(&self) -> &'static str {
        "host"
    }

    fn syscalls(&self) -> Vec<SyscallInfo> {
        vec![
            SyscallInfo::new("host:echo", "host", "Return the request payload unchanged."),
            SyscallInfo::new(
                "host:bytes",
                "host",
                "Return a byte-status frame with opaque byte-shaped payload.",
            ),
        ]
    }

    fn dispatch(&self, request: &Frame) -> HostResult<Frame> {
        match request.call.as_str() {
            "host:echo" => Ok(request.done_with(echo_data(&request.data))),
            "host:bytes" => Ok(request.byte_with(byte_data())),
            other => Err(HostError::no_route(format!(
                "no built-in host syscall registered for {other}"
            ))),
        }
    }
}

fn echo_data(data: &FrameData) -> FrameData {
    frame_data::tabula([("echo", data.clone())])
}

fn byte_data() -> FrameData {
    FrameData::Lista(vec![1_i64.into(), 2_i64.into(), 3_i64.into()])
}
