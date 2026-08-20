use std::collections::VecDeque;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use host_kernel::{
    CancellationProbe, DispatchContext, Kernel, ProviderContent, ProviderReply, RequestFrame,
};

use crate::kernel::{
    Conversation, Frame, HostEcho, HostError, HostResult, Status, Syscall, SyscallInfo,
};
use crate::manifest::{CapabilityManifest, RegisteredProvider};

/// Faber-owned host kernel for macOS route proofs.
///
/// Core Norma families are the public provider crates registered on
/// `host_kernel::Kernel`. This type keeps the private Frame/Conversation/
/// Wasm adapter surface while deleting duplicated family implementations.
pub struct HostKernel {
    providers: Kernel,
    host_echo: HostEcho,
    registered_providers: Vec<RegisteredProvider>,
}

impl HostKernel {
    pub fn new() -> Self {
        let mut providers = Kernel::new();
        // Registration failure is a wiring bug (duplicate prefix/route). Prefer a
        // fail-closed empty provider table over a process panic so production
        // hygiene stays zero-panic and routes report E_NO_ROUTE.
        if let Err(error) = register_core_providers(&mut providers) {
            eprintln!("faber-host-macos-arm64: public host provider registration failed: {error}");
            providers = Kernel::new();
        }
        Self {
            providers,
            host_echo: HostEcho,
            registered_providers: Vec::new(),
        }
    }

    pub fn route(&self, request: &Frame) -> Frame {
        if request.prefix() == "host" {
            return match self.host_echo.dispatch(request) {
                Ok(response) => response,
                Err(error) => request.error(&error),
            };
        }
        match self.dispatch_public(request) {
            Ok(reply) => route_frame_from_reply(request, reply),
            Err(error) => request.error(&error),
        }
    }

    pub fn open(&self, request: Frame) -> Conversation {
        if request.prefix() == "host" {
            return match self.host_echo.dispatch(&request) {
                Ok(response) => Conversation::new(request, response),
                Err(error) => Conversation::from_gateway_frames(
                    request.clone(),
                    VecDeque::from([request.error(&error)]),
                ),
            };
        }
        match self.dispatch_public(&request) {
            Ok(reply) => Conversation::from_gateway_frames(
                request.clone(),
                gateway_frames_from_reply(&request, reply),
            ),
            Err(error) => Conversation::from_gateway_frames(
                request.clone(),
                VecDeque::from([request.error(&error)]),
            ),
        }
    }

    pub fn attach_sermo(&self, sermo: &mut faber::Sermo) -> HostResult<()> {
        let request = sermo
            .first_outgoing()
            .ok_or_else(|| HostError::invalid_args("sermo has no request frame"))?;
        if request.status != faber::FrameStatus::Request {
            return Err(HostError::invalid_args(
                "sermo first outgoing frame must be a request",
            ));
        }

        let mut conversation = self.open(Frame::from(request));
        while let Some(frame) = conversation.recv() {
            sermo.push_incoming(frame.into());
        }
        Ok(())
    }

    pub fn manifest(&self) -> CapabilityManifest {
        CapabilityManifest::from_parts(self.syscalls(), self.registered_providers.clone())
    }

    pub fn syscalls(&self) -> Vec<SyscallInfo> {
        let mut syscalls = self.host_echo.syscalls();
        for provider in self.providers.manifest().providers {
            for call in provider.calls {
                let prefix = call
                    .route
                    .split_once(':')
                    .map(|(prefix, _)| prefix.to_owned())
                    .unwrap_or_else(|| call.route.clone());
                syscalls.push(SyscallInfo {
                    name: call.route,
                    prefix,
                    summary: call.summary,
                });
            }
        }
        syscalls.sort_by(|left, right| left.name.cmp(&right.name));
        syscalls
    }

    fn dispatch_public(&self, request: &Frame) -> HostResult<ProviderReply> {
        let context = DispatchContext {
            cancellation: CancellationProbe::from_flag(Arc::new(AtomicBool::new(false))),
        };
        let public_request = RequestFrame {
            conversation_id: request.id.clone(),
            route: request.call.clone(),
            opener: request.data.clone(),
            target: None,
        };
        self.providers
            .dispatch(&public_request, &context)
            .map_err(map_public_error)
    }
}

impl Default for HostKernel {
    fn default() -> Self {
        Self::new()
    }
}

fn register_core_providers(kernel: &mut Kernel) -> host_kernel::HostResult<()> {
    aleator::register(kernel)?;
    tempus::register(kernel)?;
    consolum::register(kernel)?;
    solum::register(kernel)?;
    processus::register(kernel)?;
    http::register(kernel)?;
    Ok(())
}

fn map_public_error(error: host_kernel::HostError) -> HostError {
    HostError {
        code: error.code,
        message: error.message,
        retryable: error.retryable,
    }
}

fn route_frame_from_reply(request: &Frame, reply: ProviderReply) -> Frame {
    match reply.contents.as_slice() {
        [] => request.done(),
        [ProviderContent::Item(data)] => request.done_with(data.clone()),
        [ProviderContent::Byte(bytes)] => request.byte_with(faber::Valor::Octeti(bytes.clone())),
        [ProviderContent::Bulk(data)] => request.response_status(Status::Bulk, data.clone()),
        items
            if items
                .iter()
                .all(|item| matches!(item, ProviderContent::Item(_))) =>
        {
            let list = items
                .iter()
                .filter_map(|item| match item {
                    ProviderContent::Item(data) => Some(data.clone()),
                    _ => None,
                })
                .collect();
            request.done_with(faber::Valor::Lista(list))
        }
        [first, ..] => match first {
            ProviderContent::Item(data) => request.done_with(data.clone()),
            ProviderContent::Byte(bytes) => request.byte_with(faber::Valor::Octeti(bytes.clone())),
            ProviderContent::Bulk(data) => request.response_status(Status::Bulk, data.clone()),
        },
    }
}

fn gateway_frames_from_reply(request: &Frame, reply: ProviderReply) -> VecDeque<Frame> {
    if reply.contents.is_empty() {
        return VecDeque::from([request.done()]);
    }

    let mut frames = VecDeque::new();
    for content in reply.contents {
        match content {
            ProviderContent::Item(data) => frames.push_back(request.item_with(data)),
            ProviderContent::Byte(bytes) => {
                frames.push_back(request.byte_with(faber::Valor::Octeti(bytes)))
            }
            ProviderContent::Bulk(data) => {
                frames.push_back(request.response_status(Status::Bulk, data))
            }
        }
    }
    if frames
        .back()
        .is_none_or(|frame| !frame.status.is_terminal())
    {
        frames.push_back(Frame::terminal(&request.id, &request.call, Status::Done));
    }
    frames
}
