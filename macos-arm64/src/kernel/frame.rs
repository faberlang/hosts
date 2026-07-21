use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use faber::Valor;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::kernel::frame_data::{empty, is_empty_tabula};
use crate::kernel::valor_wire::{json_to_valor, serde_field, valor_to_json};
use crate::kernel::HostError;

pub type FrameData = Valor;

static NEXT_FRAME_ID: AtomicU64 = AtomicU64::new(1);

/// Lifecycle status for host frames.
///
/// The first kernel slice only emits `Request`, `Done`, and `Error`, but the
/// fuller lifecycle is present now so future streaming, cancellation, and daemon
/// transport work do not need to reshape the core frame envelope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Request,
    Item,
    Byte,
    Bulk,
    Done,
    Error,
    Cancel,
}

impl Status {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Done | Self::Error | Self::Cancel)
    }
}

impl From<faber::FrameStatus> for Status {
    fn from(status: faber::FrameStatus) -> Self {
        match status {
            faber::FrameStatus::Request => Self::Request,
            faber::FrameStatus::Item => Self::Item,
            faber::FrameStatus::Byte => Self::Byte,
            faber::FrameStatus::Bulk => Self::Bulk,
            faber::FrameStatus::Done => Self::Done,
            faber::FrameStatus::Error => Self::Error,
            faber::FrameStatus::Cancel => Self::Cancel,
        }
    }
}

impl From<Status> for faber::FrameStatus {
    fn from(status: Status) -> Self {
        match status {
            Status::Request => Self::Request,
            Status::Item => Self::Item,
            Status::Byte => Self::Byte,
            Status::Bulk => Self::Bulk,
            Status::Done => Self::Done,
            Status::Error => Self::Error,
            Status::Cancel => Self::Cancel,
        }
    }
}

/// Universal in-memory host message.
///
/// A `Frame` is intentionally serializable even though the first proof routes it
/// in-process. That keeps the same contract usable for JSON debugging, compact
/// binary streams, local sockets, and eventual provider processes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Frame {
    pub id: String,
    pub parent_id: Option<String>,
    pub created_ms: u128,
    pub expires_in: u64,
    pub from: Option<String>,
    pub call: String,
    pub status: Status,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace: Option<Value>,
    #[serde(
        default = "default_frame_data",
        skip_serializing_if = "is_empty_tabula"
    )]
    #[serde(with = "serde_field")]
    pub data: FrameData,
}

fn default_frame_data() -> FrameData {
    empty()
}

impl Frame {
    pub fn request(call: impl Into<String>) -> Self {
        Self::request_with(call, empty())
    }

    pub fn request_with(call: impl Into<String>, data: FrameData) -> Self {
        Self {
            id: next_frame_id(),
            parent_id: None,
            created_ms: now_millis(),
            expires_in: 0,
            from: None,
            call: call.into(),
            status: Status::Request,
            trace: None,
            data,
        }
    }

    pub fn prefix(&self) -> &str {
        self.call
            .split_once(':')
            .map_or(&self.call, |(prefix, _)| prefix)
    }

    pub fn done(&self) -> Self {
        self.response(Status::Done, empty())
    }

    pub fn done_with(&self, data: FrameData) -> Self {
        self.response(Status::Done, data)
    }

    pub fn item_with(&self, data: FrameData) -> Self {
        self.response(Status::Item, data)
    }

    pub fn byte_with(&self, data: FrameData) -> Self {
        self.response(Status::Byte, data)
    }

    pub fn response_status(&self, status: Status, data: FrameData) -> Self {
        self.response(status, data)
    }

    pub fn error(&self, error: &HostError) -> Self {
        self.response(Status::Error, error.to_data())
    }

    pub fn terminal(parent_id: impl Into<String>, call: impl Into<String>, status: Status) -> Self {
        Self {
            id: next_frame_id(),
            parent_id: Some(parent_id.into()),
            created_ms: now_millis(),
            expires_in: 0,
            from: Some("faber-host-macos-arm64".into()),
            call: call.into(),
            status,
            trace: None,
            data: empty(),
        }
    }

    pub fn with_from(mut self, from: impl Into<String>) -> Self {
        self.from = Some(from.into());
        self
    }

    pub fn with_trace(mut self, trace: Value) -> Self {
        self.trace = Some(trace);
        self
    }

    fn response(&self, status: Status, data: FrameData) -> Self {
        Self {
            id: next_frame_id(),
            parent_id: Some(self.id.clone()),
            created_ms: now_millis(),
            expires_in: 0,
            from: Some("faber-host-macos-arm64".into()),
            call: self.call.clone(),
            status,
            trace: self.trace.clone(),
            data,
        }
    }
}

#[allow(clippy::manual_ok_err)]
impl From<faber::Scrinium> for Frame {
    fn from(frame: faber::Scrinium) -> Self {
        let trace = frame.trace.and_then(|trace| {
            if let Ok(json) = valor_to_json(&trace) {
                Some(json)
            } else {
                None
            }
        });
        Self {
            id: if frame.id.is_empty() {
                next_frame_id()
            } else {
                frame.id
            },
            parent_id: frame.parent_id,
            created_ms: u128::try_from(frame.created_ms).unwrap_or_else(|_| now_millis()),
            expires_in: 0,
            from: frame.from,
            call: frame.call,
            status: Status::from(frame.status),
            trace,
            data: frame.data,
        }
    }
}

#[allow(clippy::manual_ok_err)]
impl From<Frame> for faber::Scrinium {
    fn from(frame: Frame) -> Self {
        let trace = frame.trace.and_then(|trace| {
            if let Ok(valor) = json_to_valor(trace) {
                Some(valor)
            } else {
                None
            }
        });
        faber::Scrinium {
            id: frame.id,
            parent_id: frame.parent_id,
            call: frame.call,
            status: faber::FrameStatus::from(frame.status),
            data: frame.data,
            created_ms: i64::try_from(frame.created_ms).unwrap_or(i64::MAX),
            from: frame.from,
            trace,
        }
    }
}

fn next_frame_id() -> String {
    let seq = NEXT_FRAME_ID.fetch_add(1, Ordering::Relaxed);
    format!("frame-{seq}")
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}
