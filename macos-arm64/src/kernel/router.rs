use std::collections::{BTreeMap, VecDeque};

use faber::Valor;

use crate::kernel::{frame_data, Frame, HostError, Status, Syscall, SyscallInfo};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    CallerToGateway,
    GatewayToCaller,
}

pub struct Conversation {
    conversation_id: String,
    route: String,
    sent: Vec<Frame>,
    incoming: VecDeque<Frame>,
    caller_done: bool,
    gateway_done: bool,
    detached: bool,
}

impl Conversation {
    pub(crate) fn new(request: Frame, response: Frame) -> Self {
        Self::from_gateway_frames(request.clone(), conversation_incoming(&request, response))
    }

    /// Build a conversation from already-materialized gateway frames.
    pub(crate) fn from_gateway_frames(request: Frame, incoming: VecDeque<Frame>) -> Self {
        let gateway_done = incoming
            .back()
            .is_some_and(|frame| frame.status.is_terminal());
        Self {
            conversation_id: request.id.clone(),
            route: request.call.clone(),
            sent: vec![request],
            incoming,
            caller_done: false,
            gateway_done,
            detached: false,
        }
    }

    pub fn id(&self) -> &str {
        &self.conversation_id
    }

    pub fn route(&self) -> &str {
        &self.route
    }

    pub fn push(&mut self, mut frame: Frame) {
        frame.parent_id = Some(self.conversation_id.clone());
        frame.call = self.route.clone();
        if frame.status.is_terminal() {
            self.caller_done = true;
        }
        self.sent.push(frame);
    }

    pub fn recv(&mut self) -> Option<Frame> {
        if self.detached {
            return None;
        }
        let frame = self.incoming.pop_front()?;
        if frame.status.is_terminal() {
            self.gateway_done = true;
        }
        Some(frame)
    }

    pub fn done(&mut self, direction: Direction) {
        match direction {
            Direction::CallerToGateway => {
                if !self.caller_done {
                    self.push(self.local_terminal(Status::Done));
                }
            }
            Direction::GatewayToCaller => {
                if !self.gateway_done {
                    self.incoming.push_back(self.local_terminal(Status::Done));
                    self.gateway_done = true;
                }
            }
        }
    }

    pub fn detach(&mut self) {
        self.detached = true;
        self.incoming.clear();
    }

    pub fn sent(&self) -> &[Frame] {
        &self.sent
    }

    pub fn is_complete(&self) -> bool {
        self.caller_done && (self.gateway_done || self.detached)
    }

    fn local_terminal(&self, status: Status) -> Frame {
        Frame::terminal(&self.conversation_id, &self.route, status)
    }
}

fn conversation_incoming(request: &Frame, response: Frame) -> VecDeque<Frame> {
    if response.status == Status::Done && !frame_data::is_empty(&response.data) {
        if let Valor::Lista(items) = response.data {
            let mut frames = items
                .into_iter()
                .map(|item| request.item_with(item))
                .collect::<VecDeque<_>>();
            frames.push_back(Frame::terminal(&request.id, &request.call, Status::Done));
            return frames;
        }
        return VecDeque::from([
            request.item_with(response.data),
            Frame::terminal(&request.id, &request.call, Status::Done),
        ]);
    }
    if !response.status.is_terminal() {
        return VecDeque::from([
            response,
            Frame::terminal(&request.id, &request.call, Status::Done),
        ]);
    }

    VecDeque::from([response])
}

/// Prefix router for host syscalls.
///
/// The current router is deliberately synchronous and in-process. That is enough
/// to prove the host route contract while keeping daemon transport, cancellation
/// tokens, and streaming backpressure out of the first slice.
pub struct Router {
    routes: BTreeMap<&'static str, Box<dyn Syscall>>,
}

impl Router {
    pub fn new() -> Self {
        Self {
            routes: BTreeMap::new(),
        }
    }

    pub fn register(&mut self, syscall: impl Syscall + 'static) {
        self.routes.insert(syscall.prefix(), Box::new(syscall));
    }

    pub fn route(&self, request: &Frame) -> Frame {
        let Some(syscall) = self.routes.get(request.prefix()) else {
            let error = HostError::no_route(format!("no handler for call: {}", request.call));
            return request.error(&error);
        };

        match syscall.dispatch(request) {
            Ok(response) => response,
            Err(error) => request.error(&error),
        }
    }

    pub fn open(&self, request: Frame) -> Conversation {
        let response = self.route(&request);
        Conversation::new(request, response)
    }

    pub fn syscalls(&self) -> Vec<SyscallInfo> {
        self.routes
            .values()
            .flat_map(|syscall| syscall.syscalls())
            .collect()
    }
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}
