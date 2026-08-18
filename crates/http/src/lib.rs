//! Target-specific HTTP/1.1 provider for the generic `http:*` family.
//!
//! The provider exposes only the manifest contract. Framework routing,
//! middleware, and request/response application semantics remain Faber code.
//!
//! The crate also carries the concrete HTTP client effects
//! ([`client`]) — the single home for generated-Rust HTTP client surface.

pub mod client;

use faber::Valor;
use host_kernel::{
    DispatchContext, HostError, HostResult, Kernel, Provider, ProviderRegistration, ProviderReply,
    RequestFrame,
};
use std::collections::{BTreeMap, HashMap};
use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

pub const DEFAULT_MAX_BODY_BYTES: usize = 1024 * 1024;
pub const DEFAULT_BIND_HOST: &str = "127.0.0.1";
pub(crate) const MAX_CONFIGURED_BODY_BYTES: usize = 16 * 1024 * 1024;
const MAX_HEADER_BYTES: usize = 64 * 1024;
const ACCEPT_BACKLOG: i32 = 32;
const POLL_INTERVAL: Duration = Duration::from_millis(5);
const WRITE_TIMEOUT: Duration = Duration::from_millis(250);
const STREAM_WRITE_TIMEOUT: Duration = Duration::from_secs(5);
const REQUEST_ID_PREFIX: &str = "http-";

pub struct Http {
    registration: ProviderRegistration,
    state: Arc<HttpState>,
}

struct HttpState {
    listeners: Mutex<HashMap<i64, Arc<ListenerState>>>,
    connections: Mutex<HashMap<i64, Arc<ConnectionSlot>>>,
    requests: Mutex<HashMap<String, i64>>,
    writers: Mutex<HashMap<i64, i64>>,
    next_listener: AtomicI64,
    next_request: AtomicU64,
    next_connection: AtomicI64,
    next_writer: AtomicI64,
    client: client::ClientState,
}

struct ListenerState {
    listener: TcpListener,
    max_body_bytes: usize,
    stopped: AtomicBool,
}

struct ConnectionSlot {
    id: i64,
    listener: i64,
    closer: TcpStream,
    inner: Mutex<Connection>,
}

struct Connection {
    stream: TcpStream,
    leftover: Vec<u8>,
    phase: Phase,
}

enum Phase {
    Idle,
    Pending { request_id: String },
    Streaming { request_id: String, writer: i64 },
}

struct RequestParts {
    id: String,
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

struct ParsedHeaders {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    content_length: usize,
}

enum PollRead {
    Ready(RequestParts),
    Idle,
    Closed,
}

impl Http {
    /// Create a new [`Http`] provider.
    ///
    /// # Errors
    ///
    /// Returns [`HostError`] if the embedded manifest JSON cannot be parsed.
    pub fn new() -> HostResult<Self> {
        Ok(Self {
            registration: ProviderRegistration::new(host_kernel::parse_manifest(manifest_json())?),
            state: Arc::new(HttpState {
                listeners: Mutex::new(HashMap::new()),
                connections: Mutex::new(HashMap::new()),
                requests: Mutex::new(HashMap::new()),
                writers: Mutex::new(HashMap::new()),
                next_listener: AtomicI64::new(1),
                next_request: AtomicU64::new(1),
                next_connection: AtomicI64::new(1),
                next_writer: AtomicI64::new(1),
                client: client::ClientState::new(),
            }),
        })
    }
}

/// Register the [`Http`] provider with the kernel.
///
/// # Errors
///
/// Returns [`HostError`] if the provider cannot be created
/// (manifest parsing failure) or if registration fails.
pub fn register(kernel: &mut Kernel) -> HostResult<()> {
    kernel.register(Arc::new(Http::new()?))
}

#[must_use]
pub fn manifest_json() -> &'static str {
    include_str!("manifest.json")
}

impl Provider for Http {
    fn registration(&self) -> &ProviderRegistration {
        &self.registration
    }

    fn dispatch(
        &self,
        request: &RequestFrame,
        context: &DispatchContext,
    ) -> HostResult<ProviderReply> {
        match request.route.as_str() {
            "http:listen" => self.listen(&request.opener),
            "http:accept" => self.accept(&request.opener, context),
            "http:respond" => self.respond(&request.opener),
            "http:respond_open" => self.respond_open(&request.opener),
            "http:respond_chunk" => self.respond_chunk(&request.opener, context),
            "http:respond_finish" => self.respond_finish(&request.opener, context),
            "http:stop" => self.stop(&request.opener),
            "http:agent" => self.state.client.agent(&request.opener),
            "http:get" => self.state.client.verb_get(&request.opener),
            "http:post" => self.state.client.verb_post(&request.opener),
            "http:put" => self.state.client.verb_put(&request.opener),
            "http:delete" => self.state.client.verb_delete(&request.opener),
            "http:patch" => self.state.client.verb_patch(&request.opener),
            "http:request" => self.state.client.request(&request.opener),
            "http:request_open" => self.state.client.request_open(&request.opener),
            "http:read" => self.state.client.read(&request.opener, context),
            other => Err(HostError::no_route(format!(
                "no built-in http syscall registered for {other}"
            ))),
        }
    }
}

impl Http {
    fn listen(&self, opener: &Valor) -> HostResult<ProviderReply> {
        let values = list_args(opener, "http:listen")?;
        if !(1..=3).contains(&values.len()) {
            return Err(HostError::invalid_args(
                "http:listen requires [port], [port, max_body_bytes], or [port, max_body_bytes, bind_host]",
            ));
        }
        let port = integer_arg(&values[0], "port")?;
        let port = u16::try_from(port)
            .map_err(|_| HostError::invalid_args("http:listen port must be between 0 and 65535"))?;
        let (max_body_bytes, bind_host) = match values.len() {
            1 => (DEFAULT_MAX_BODY_BYTES, DEFAULT_BIND_HOST.to_owned()),
            2 => match &values[1] {
                Valor::Textus(host) | Valor::Instans(host) => {
                    (DEFAULT_MAX_BODY_BYTES, parse_bind_host(host)?)
                }
                other => (
                    bounded_body_size(integer_arg(other, "max_body_bytes")?)?,
                    DEFAULT_BIND_HOST.to_owned(),
                ),
            },
            _ => (
                bounded_body_size(integer_arg(&values[1], "max_body_bytes")?)?,
                parse_bind_host(&text_arg(&values[2], "bind_host")?)?,
            ),
        };
        let listener = TcpListener::bind((bind_host.as_str(), port))
            .map_err(|error| HostError::internal(format!("http:listen bind failed: {error}")))?;
        apply_accept_backlog(&listener)?;
        listener
            .set_nonblocking(true)
            .map_err(|error| HostError::internal(format!("http:listen setup failed: {error}")))?;
        let state = Arc::new(ListenerState {
            listener,
            max_body_bytes,
            stopped: AtomicBool::new(false),
        });
        let handle = self.state.next_listener.fetch_add(1, Ordering::SeqCst);
        let mut listeners = lock(&self.state.listeners, "http listeners")?;
        listeners.insert(handle, state);
        Ok(ProviderReply::item(Valor::Numerus(handle)))
    }

    fn accept(&self, opener: &Valor, context: &DispatchContext) -> HostResult<ProviderReply> {
        let handle = integer_arg(positional(opener, 0, "listener")?, "listener")?;
        let listener = {
            let listeners = lock(&self.state.listeners, "http listeners")?;
            listeners.get(&handle).cloned()
        }
        .ok_or_else(|| HostError::invalid_args(format!("http:accept unknown listener {handle}")))?;

        loop {
            if context.cancellation.is_cancelled() {
                return Err(HostError::cancelled());
            }
            if listener.stopped.load(Ordering::SeqCst) {
                return Err(HostError::invalid_args("http:accept listener is stopped"));
            }
            if let Some(request) = self.poll_idle_connection(handle, &listener, context)? {
                return Ok(ProviderReply::item(request));
            }
            match listener.listener.accept() {
                Ok((stream, _peer)) => {
                    let request = self.admit_connection(handle, stream, &listener, context)?;
                    return Ok(ProviderReply::item(request));
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    thread::sleep(POLL_INTERVAL);
                }
                Err(error) => {
                    return Err(HostError::internal(format!("http:accept failed: {error}")));
                }
            }
        }
    }

    fn admit_connection(
        &self,
        listener_handle: i64,
        stream: TcpStream,
        listener: &ListenerState,
        context: &DispatchContext,
    ) -> HostResult<Valor> {
        let closer = stream
            .try_clone()
            .map_err(|error| HostError::internal(format!("http:accept clone failed: {error}")))?;
        let mut connection = Connection {
            stream,
            leftover: Vec::new(),
            phase: Phase::Idle,
        };
        let parts = read_request(
            &mut connection,
            listener.max_body_bytes,
            &listener.stopped,
            context,
            &self.state.next_request,
        )?;
        if listener.stopped.load(Ordering::SeqCst) {
            return Err(HostError::invalid_args("http:accept listener is stopped"));
        }
        let connection_id = self.state.next_connection.fetch_add(1, Ordering::SeqCst);
        let request_id = parts.id.clone();
        let slot = Arc::new(ConnectionSlot {
            id: connection_id,
            listener: listener_handle,
            closer,
            inner: Mutex::new({
                connection.phase = Phase::Pending {
                    request_id: request_id.clone(),
                };
                connection
            }),
        });
        {
            let mut connections = lock(&self.state.connections, "http connections")?;
            connections.insert(connection_id, slot);
        }
        {
            let mut requests = lock(&self.state.requests, "http requests")?;
            requests.insert(request_id, connection_id);
        }
        Ok(request_carrier(&parts, connection_id))
    }

    fn poll_idle_connection(
        &self,
        listener_handle: i64,
        listener: &ListenerState,
        context: &DispatchContext,
    ) -> HostResult<Option<Valor>> {
        let idle = {
            let connections = lock(&self.state.connections, "http connections")?;
            connections
                .values()
                .filter(|slot| slot.listener == listener_handle)
                .filter_map(|slot| {
                    let connection = slot.inner.try_lock().ok()?;
                    matches!(connection.phase, Phase::Idle).then(|| Arc::clone(slot))
                })
                .collect::<Vec<_>>()
        };
        let mut closed = Vec::new();
        for slot in idle {
            let outcome = {
                let mut connection = lock(&slot.inner, "http connection")?;
                if !matches!(connection.phase, Phase::Idle) {
                    continue;
                }
                poll_request(
                    &mut connection,
                    listener.max_body_bytes,
                    &listener.stopped,
                    context,
                    &self.state.next_request,
                    false,
                )?
            };
            match outcome {
                PollRead::Ready(parts) => {
                    let request_id = parts.id.clone();
                    {
                        let mut connection = lock(&slot.inner, "http connection")?;
                        connection.phase = Phase::Pending {
                            request_id: request_id.clone(),
                        };
                    }
                    let mut requests = lock(&self.state.requests, "http requests")?;
                    requests.insert(request_id, slot.id);
                    return Ok(Some(request_carrier(&parts, slot.id)));
                }
                PollRead::Closed => closed.push(slot.id),
                PollRead::Idle => {}
            }
        }
        for connection_id in closed {
            self.drop_connection(connection_id)?;
        }
        Ok(None)
    }

    fn respond(&self, opener: &Valor) -> HostResult<ProviderReply> {
        let values = list_args(opener, "http:respond")?;
        if values.len() != 4 {
            return Err(HostError::invalid_args(
                "http:respond requires [request_id, status, headers, body]",
            ));
        }
        let request_id = text_arg(&values[0], "request_id")?;
        let status = status_arg(&values[1], "http:respond")?;
        let headers = response_headers(&values[2])?;
        let body = bytes_value(&values[3], "body")?;
        let connection_id = self.take_request(&request_id)?;
        let slot = self.take_connection(connection_id)?;
        let mut connection = lock(&slot.inner, "http connection")?;
        match &connection.phase {
            Phase::Pending {
                request_id: pending,
            } if pending == &request_id => {}
            _ => {
                return Err(HostError::invalid_args(format!(
                    "http:respond unknown request {request_id}"
                )));
            }
        }
        connection.phase = Phase::Idle;
        let write_result =
            write_oneshot(&mut connection.stream, status, &request_id, &headers, &body);
        let _ = slot.closer.shutdown(Shutdown::Both);
        let _ = connection.stream.shutdown(Shutdown::Both);
        write_result?;
        Ok(ProviderReply::vacuum())
    }

    fn respond_open(&self, opener: &Valor) -> HostResult<ProviderReply> {
        let values = list_args(opener, "http:respond_open")?;
        if values.len() != 3 {
            return Err(HostError::invalid_args(
                "http:respond_open requires [request_id, status, headers]",
            ));
        }
        let request_id = text_arg(&values[0], "request_id")?;
        let status = status_arg(&values[1], "http:respond_open")?;
        let headers = response_headers(&values[2])?;
        let connection_id = self.take_request(&request_id)?;
        let slot = self.connection(connection_id)?;
        let writer = self.state.next_writer.fetch_add(1, Ordering::SeqCst);
        {
            let mut connection = lock(&slot.inner, "http connection")?;
            match &connection.phase {
                Phase::Pending {
                    request_id: pending,
                } if pending == &request_id => {}
                _ => {
                    self.restore_request(request_id.clone(), connection_id)?;
                    return Err(HostError::invalid_args(format!(
                        "http:respond_open unknown request {request_id}"
                    )));
                }
            }
            let head = format_chunked_open(status, &request_id, &headers);
            if let Err(error) = write_blocking(&mut connection.stream, &head) {
                let _ = slot.closer.shutdown(Shutdown::Both);
                let _ = connection.stream.shutdown(Shutdown::Both);
                drop(connection);
                self.drop_connection(connection_id)?;
                return Err(error);
            }
            connection.phase = Phase::Streaming { request_id, writer };
        }
        let mut writers = lock(&self.state.writers, "http writers")?;
        writers.insert(writer, connection_id);
        Ok(ProviderReply::item(Valor::Numerus(writer)))
    }

    fn respond_chunk(
        &self,
        opener: &Valor,
        context: &DispatchContext,
    ) -> HostResult<ProviderReply> {
        let values = list_args(opener, "http:respond_chunk")?;
        if values.len() != 2 {
            return Err(HostError::invalid_args(
                "http:respond_chunk requires [writer, bytes]",
            ));
        }
        let writer = integer_arg(&values[0], "writer")?;
        let bytes = bytes_value(&values[1], "bytes")?;
        if bytes.is_empty() {
            return Err(HostError::invalid_args(
                "http:respond_chunk bytes must not be empty",
            ));
        }
        let slot = self.writer_connection(writer)?;
        let frame = format_chunk(&bytes);
        {
            let mut connection = lock(&slot.inner, "http connection")?;
            match connection.phase {
                Phase::Streaming {
                    writer: open_writer,
                    ..
                } if open_writer == writer => {}
                _ => {
                    return Err(HostError::invalid_args(format!(
                        "http:respond_chunk unknown writer {writer}"
                    )));
                }
            }
            self.write_all_backpressure(&mut connection.stream, &frame, slot.listener, context)?;
        }
        Ok(ProviderReply::vacuum())
    }

    fn respond_finish(
        &self,
        opener: &Valor,
        context: &DispatchContext,
    ) -> HostResult<ProviderReply> {
        let values = list_args(opener, "http:respond_finish")?;
        if values.len() != 2 {
            return Err(HostError::invalid_args(
                "http:respond_finish requires [writer, keep_alive]",
            ));
        }
        let writer = integer_arg(&values[0], "writer")?;
        let keep_alive = bool_arg(&values[1], "keep_alive")?;
        let slot = self.take_writer(writer)?;
        {
            let mut connection = lock(&slot.inner, "http connection")?;
            match connection.phase {
                Phase::Streaming {
                    writer: open_writer,
                    ..
                } if open_writer == writer => {}
                _ => {
                    return Err(HostError::invalid_args(format!(
                        "http:respond_finish unknown writer {writer}"
                    )));
                }
            }
            self.write_all_backpressure(
                &mut connection.stream,
                b"0\r\n\r\n",
                slot.listener,
                context,
            )?;
            if keep_alive {
                connection.phase = Phase::Idle;
                connection.stream.set_nonblocking(true).map_err(|error| {
                    HostError::internal(format!("http:respond_finish setup failed: {error}"))
                })?;
            } else {
                let _ = slot.closer.shutdown(Shutdown::Both);
                let _ = connection.stream.shutdown(Shutdown::Both);
                drop(connection);
                self.drop_connection(slot.id)?;
            }
        }
        Ok(ProviderReply::vacuum())
    }

    fn stop(&self, opener: &Valor) -> HostResult<ProviderReply> {
        let handle = integer_arg(positional(opener, 0, "listener")?, "listener")?;
        let listener = {
            let mut listeners = lock(&self.state.listeners, "http listeners")?;
            listeners.remove(&handle)
        }
        .ok_or_else(|| HostError::invalid_args(format!("http:stop unknown listener {handle}")))?;
        listener.stopped.store(true, Ordering::SeqCst);
        let slots = {
            let mut connections = lock(&self.state.connections, "http connections")?;
            let ids = connections
                .iter()
                .filter_map(|(id, slot)| (slot.listener == handle).then_some(*id))
                .collect::<Vec<_>>();
            ids.into_iter()
                .filter_map(|id| connections.remove(&id))
                .collect::<Vec<_>>()
        };
        let mut request_ids = Vec::new();
        let mut writer_ids = Vec::new();
        for slot in &slots {
            let _ = slot.closer.shutdown(Shutdown::Both);
            if let Ok(connection) = slot.inner.lock() {
                match &connection.phase {
                    Phase::Pending { request_id } => request_ids.push(request_id.clone()),
                    Phase::Streaming {
                        request_id, writer, ..
                    } => {
                        request_ids.push(request_id.clone());
                        writer_ids.push(*writer);
                    }
                    Phase::Idle => {}
                }
            }
        }
        {
            let mut requests = lock(&self.state.requests, "http requests")?;
            for request_id in request_ids {
                requests.remove(&request_id);
            }
        }
        {
            let mut writers = lock(&self.state.writers, "http writers")?;
            for writer in writer_ids {
                writers.remove(&writer);
            }
        }
        Ok(ProviderReply::vacuum())
    }

    fn take_request(&self, request_id: &str) -> HostResult<i64> {
        let mut requests = lock(&self.state.requests, "http requests")?;
        requests.remove(request_id).ok_or_else(|| {
            HostError::invalid_args(format!("http:respond unknown request {request_id}"))
        })
    }

    fn restore_request(&self, request_id: String, connection_id: i64) -> HostResult<()> {
        let mut requests = lock(&self.state.requests, "http requests")?;
        requests.insert(request_id, connection_id);
        Ok(())
    }

    fn connection(&self, connection_id: i64) -> HostResult<Arc<ConnectionSlot>> {
        let connections = lock(&self.state.connections, "http connections")?;
        connections.get(&connection_id).cloned().ok_or_else(|| {
            HostError::invalid_args(format!("http unknown connection {connection_id}"))
        })
    }

    fn take_connection(&self, connection_id: i64) -> HostResult<Arc<ConnectionSlot>> {
        let mut connections = lock(&self.state.connections, "http connections")?;
        connections.remove(&connection_id).ok_or_else(|| {
            HostError::invalid_args(format!("http unknown connection {connection_id}"))
        })
    }

    fn writer_connection(&self, writer: i64) -> HostResult<Arc<ConnectionSlot>> {
        let connection_id = {
            let writers = lock(&self.state.writers, "http writers")?;
            *writers.get(&writer).ok_or_else(|| {
                HostError::invalid_args(format!("http:respond_chunk unknown writer {writer}"))
            })?
        };
        self.connection(connection_id)
    }

    fn take_writer(&self, writer: i64) -> HostResult<Arc<ConnectionSlot>> {
        let connection_id = {
            let mut writers = lock(&self.state.writers, "http writers")?;
            writers.remove(&writer).ok_or_else(|| {
                HostError::invalid_args(format!("http:respond_finish unknown writer {writer}"))
            })?
        };
        self.connection(connection_id)
    }

    fn drop_connection(&self, connection_id: i64) -> HostResult<()> {
        if let Some(slot) = {
            let mut connections = lock(&self.state.connections, "http connections")?;
            connections.remove(&connection_id)
        } {
            self.forget_connection_indexes(&slot)?;
        }
        Ok(())
    }

    fn forget_connection_indexes(&self, slot: &ConnectionSlot) -> HostResult<()> {
        let (request_id, writer) = {
            let connection = lock(&slot.inner, "http connection")?;
            match &connection.phase {
                Phase::Pending { request_id } => (Some(request_id.clone()), None),
                Phase::Streaming {
                    request_id, writer, ..
                } => (Some(request_id.clone()), Some(*writer)),
                Phase::Idle => (None, None),
            }
        };
        if let Some(request_id) = request_id {
            lock(&self.state.requests, "http requests")?.remove(&request_id);
        }
        if let Some(writer) = writer {
            lock(&self.state.writers, "http writers")?.remove(&writer);
        }
        Ok(())
    }

    fn listener_is_stopped(&self, handle: i64) -> HostResult<bool> {
        let listeners = lock(&self.state.listeners, "http listeners")?;
        Ok(listeners
            .get(&handle)
            .is_none_or(|listener| listener.stopped.load(Ordering::SeqCst)))
    }

    fn write_all_backpressure(
        &self,
        stream: &mut TcpStream,
        mut bytes: &[u8],
        listener: i64,
        context: &DispatchContext,
    ) -> HostResult<()> {
        stream.set_nonblocking(true).map_err(|error| {
            HostError::internal(format!("http stream write setup failed: {error}"))
        })?;
        let deadline = Instant::now() + STREAM_WRITE_TIMEOUT;
        while !bytes.is_empty() {
            if context.cancellation.is_cancelled() {
                return Err(HostError::cancelled());
            }
            if self.listener_is_stopped(listener)? {
                return Err(HostError::invalid_args("http listener is stopped"));
            }
            if Instant::now() >= deadline {
                return Err(HostError::internal(
                    "http write timed out under backpressure",
                ));
            }
            match stream.write(bytes) {
                Ok(0) => return Err(HostError::internal("http write closed")),
                Ok(count) => bytes = &bytes[count..],
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    thread::sleep(POLL_INTERVAL);
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => {
                    return Err(HostError::internal(format!("http write failed: {error}")));
                }
            }
        }
        Ok(())
    }
}

impl RequestParts {
    fn new(
        id: String,
        method: String,
        path: String,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    ) -> Self {
        Self {
            id,
            method,
            path,
            headers,
            body,
        }
    }
}

fn read_request(
    connection: &mut Connection,
    max_body_bytes: usize,
    stopped: &AtomicBool,
    context: &DispatchContext,
    next_request: &AtomicU64,
) -> HostResult<RequestParts> {
    loop {
        match poll_request(
            connection,
            max_body_bytes,
            stopped,
            context,
            next_request,
            true,
        )? {
            PollRead::Ready(parts) => return Ok(parts),
            PollRead::Idle => {
                check_read_state(stopped, context)?;
                thread::sleep(POLL_INTERVAL);
            }
            PollRead::Closed => {
                return Err(HostError::invalid_args(
                    "http:accept request ended before headers",
                ));
            }
        }
    }
}

fn poll_request(
    connection: &mut Connection,
    max_body_bytes: usize,
    stopped: &AtomicBool,
    context: &DispatchContext,
    next_request: &AtomicU64,
    block: bool,
) -> HostResult<PollRead> {
    connection
        .stream
        .set_nonblocking(true)
        .map_err(|error| HostError::internal(format!("http:accept setup failed: {error}")))?;
    loop {
        if let Some(end) = find_header_end(&connection.leftover) {
            let ParsedHeaders {
                method,
                path,
                headers,
                content_length,
            } = parse_headers(&connection.leftover[..end])?;
            if content_length > max_body_bytes {
                return Err(HostError::invalid_args(format!(
                    "http:accept request body exceeds max_body_bytes {max_body_bytes}"
                )));
            }
            let body_start = end + 4;
            let needed = body_start + content_length;
            if connection.leftover.len() >= needed {
                let body = connection.leftover[body_start..needed].to_vec();
                connection.leftover.drain(..needed);
                let id = format!(
                    "{REQUEST_ID_PREFIX}{}",
                    next_request.fetch_add(1, Ordering::SeqCst)
                );
                return Ok(PollRead::Ready(RequestParts::new(
                    id, method, path, headers, body,
                )));
            }
        } else if connection.leftover.len() > MAX_HEADER_BYTES {
            return Err(HostError::invalid_args(
                "http:accept request headers exceed limit",
            ));
        }
        let mut chunk = [0_u8; 4096];
        match connection.stream.read(&mut chunk) {
            Ok(0) => {
                if connection.leftover.is_empty() {
                    return Ok(PollRead::Closed);
                }
                return Err(HostError::invalid_args(
                    "http:accept request ended before headers",
                ));
            }
            Ok(count) => connection.leftover.extend_from_slice(&chunk[..count]),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                if block {
                    check_read_state(stopped, context)?;
                    thread::sleep(POLL_INTERVAL);
                    continue;
                }
                return Ok(PollRead::Idle);
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => {
                return Err(HostError::internal(format!(
                    "http:accept read failed: {error}"
                )));
            }
        }
    }
}

fn check_read_state(stopped: &AtomicBool, context: &DispatchContext) -> HostResult<()> {
    if context.cancellation.is_cancelled() {
        return Err(HostError::cancelled());
    }
    if stopped.load(Ordering::SeqCst) {
        return Err(HostError::invalid_args("http:accept listener is stopped"));
    }
    Ok(())
}

fn parse_headers(bytes: &[u8]) -> HostResult<ParsedHeaders> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| HostError::invalid_args("http:accept headers are not UTF-8"))?;
    let mut lines = text.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| HostError::invalid_args("http:accept request line is missing"))?;
    let mut request_parts = request_line.split_ascii_whitespace();
    let method = request_parts
        .next()
        .filter(|method| !method.is_empty() && method.bytes().all(is_token_byte))
        .ok_or_else(|| HostError::invalid_args("http:accept request method is malformed"))?;
    let path = request_parts
        .next()
        .filter(|path| !path.is_empty() && !path.contains(['\r', '\n']))
        .ok_or_else(|| HostError::invalid_args("http:accept request path is malformed"))?;
    if request_parts.next() != Some("HTTP/1.1") || request_parts.next().is_some() {
        return Err(HostError::invalid_args("http:accept requires HTTP/1.1"));
    }
    let mut headers = Vec::new();
    let mut content_length = None;
    let mut has_host = false;
    for line in lines {
        if line.is_empty() {
            break;
        }
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| HostError::invalid_args("http:accept header is malformed"))?;
        let name = name.trim();
        let value = value.trim();
        if name.is_empty() || !name.bytes().all(is_token_byte) || !is_valid_header_value(value) {
            return Err(HostError::invalid_args("http:accept header is malformed"));
        }
        if name.eq_ignore_ascii_case("host") {
            has_host = true;
        }
        if name.eq_ignore_ascii_case("transfer-encoding") {
            return Err(HostError::invalid_args(
                "http:accept transfer-encoding is unsupported",
            ));
        }
        if name.eq_ignore_ascii_case("content-length") {
            let parsed = value
                .parse::<usize>()
                .map_err(|_| HostError::invalid_args("http:accept content-length is malformed"))?;
            if content_length.replace(parsed).is_some() {
                return Err(HostError::invalid_args(
                    "http:accept repeats content-length",
                ));
            }
        }
        headers.push((name.to_owned(), value.to_owned()));
    }
    if !has_host {
        return Err(HostError::invalid_args(
            "http:accept HTTP/1.1 request requires a Host header",
        ));
    }
    Ok(ParsedHeaders {
        method: method.to_owned(),
        path: path.to_owned(),
        headers,
        content_length: content_length.unwrap_or(0),
    })
}

fn request_carrier(parts: &RequestParts, connection: i64) -> Valor {
    let mut fields = BTreeMap::new();
    fields.insert("id".to_owned(), Valor::Textus(parts.id.clone()));
    fields.insert("method".to_owned(), Valor::Textus(parts.method.clone()));
    fields.insert("path".to_owned(), Valor::Textus(parts.path.clone()));
    fields.insert("headers".to_owned(), headers_value(&parts.headers));
    fields.insert("body".to_owned(), Valor::Octeti(parts.body.clone()));
    fields.insert("connection".to_owned(), Valor::Numerus(connection));
    Valor::Tabula(fields)
}

pub(crate) fn headers_value(headers: &[(String, String)]) -> Valor {
    Valor::Lista(
        headers
            .iter()
            .map(|(name, value)| {
                let mut fields = BTreeMap::new();
                fields.insert("name".to_owned(), Valor::Textus(name.clone()));
                fields.insert("value".to_owned(), Valor::Textus(value.clone()));
                Valor::Tabula(fields)
            })
            .collect(),
    )
}

pub(crate) fn response_headers(value: &Valor) -> HostResult<Vec<(String, String)>> {
    let Valor::Lista(items) = value else {
        return Err(HostError::invalid_args(
            "http:respond headers must be a list",
        ));
    };
    items.iter().map(header_value).collect()
}

fn header_value(value: &Valor) -> HostResult<(String, String)> {
    let Valor::Tabula(fields) = value else {
        return Err(HostError::invalid_args(
            "HTTP headers must be tables with name and value",
        ));
    };
    let name = text_field(fields, "name")?;
    let value = text_field(fields, "value")?;
    if name.is_empty() || !name.bytes().all(is_token_byte) || !is_valid_header_value(&value) {
        return Err(HostError::invalid_args(
            "HTTP header name or value is malformed",
        ));
    }
    if name.eq_ignore_ascii_case("connection")
        || name.eq_ignore_ascii_case("content-length")
        || name.eq_ignore_ascii_case("transfer-encoding")
    {
        return Err(HostError::invalid_args(
            "HTTP connection, content-length, and transfer-encoding headers are provider-owned",
        ));
    }
    Ok((name, value))
}

fn format_response(
    status: u16,
    request_id: &str,
    headers: &[(String, String)],
    body: &[u8],
) -> Vec<u8> {
    let mut response = format!(
        "HTTP/1.1 {status} {}\r\nconnection: close\r\nx-faber-request-id: {request_id}\r\ncontent-length: {}\r\n",
        reason_phrase(status),
        body.len()
    )
    .into_bytes();
    append_headers(&mut response, headers);
    response.extend_from_slice(body);
    response
}

fn format_chunked_open(status: u16, request_id: &str, headers: &[(String, String)]) -> Vec<u8> {
    let mut response = format!(
        "HTTP/1.1 {status} {}\r\ntransfer-encoding: chunked\r\nx-faber-request-id: {request_id}\r\n",
        reason_phrase(status),
    )
    .into_bytes();
    append_headers(&mut response, headers);
    response
}

fn append_headers(response: &mut Vec<u8>, headers: &[(String, String)]) {
    for (name, value) in headers {
        response.extend_from_slice(name.as_bytes());
        response.extend_from_slice(b": ");
        response.extend_from_slice(value.as_bytes());
        response.extend_from_slice(b"\r\n");
    }
    response.extend_from_slice(b"\r\n");
}

fn format_chunk(data: &[u8]) -> Vec<u8> {
    let mut frame = format!("{:x}\r\n", data.len()).into_bytes();
    frame.extend_from_slice(data);
    frame.extend_from_slice(b"\r\n");
    frame
}

fn write_oneshot(
    stream: &mut TcpStream,
    status: u16,
    request_id: &str,
    headers: &[(String, String)],
    body: &[u8],
) -> HostResult<()> {
    write_blocking(stream, &format_response(status, request_id, headers, body))
}

fn write_blocking(stream: &mut TcpStream, bytes: &[u8]) -> HostResult<()> {
    stream
        .set_nonblocking(false)
        .map_err(|error| HostError::internal(format!("http write setup failed: {error}")))?;
    stream
        .set_write_timeout(Some(WRITE_TIMEOUT))
        .map_err(|error| {
            HostError::internal(format!("http write timeout setup failed: {error}"))
        })?;
    stream
        .write_all(bytes)
        .map_err(|error| HostError::internal(format!("http write failed: {error}")))
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        202 => "Accepted",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "HTTP Status",
    }
}

pub(crate) fn list_args<'a>(value: &'a Valor, route: &str) -> HostResult<&'a [Valor]> {
    match value {
        Valor::Lista(values) => Ok(values),
        _ => Err(HostError::invalid_args(format!(
            "{route} opener must be a list"
        ))),
    }
}

fn positional<'a>(value: &'a Valor, index: usize, name: &str) -> HostResult<&'a Valor> {
    if index == 0 && !matches!(value, Valor::Lista(_)) {
        return Ok(value);
    }
    let values = list_args(value, "HTTP route")?;
    values.get(index).ok_or_else(|| {
        HostError::invalid_args(format!("missing positional argument {index} ({name})"))
    })
}

pub(crate) fn integer_arg(value: &Valor, name: &str) -> HostResult<i64> {
    match value {
        Valor::Numerus(value) => Ok(*value),
        _ => Err(HostError::invalid_args(format!("{name} must be numerus"))),
    }
}

fn bool_arg(value: &Valor, name: &str) -> HostResult<bool> {
    match value {
        Valor::Bivalens(value) => Ok(*value),
        _ => Err(HostError::invalid_args(format!("{name} must be bivalens"))),
    }
}

fn status_arg(value: &Valor, route: &str) -> HostResult<u16> {
    let status = integer_arg(value, "status")?;
    u16::try_from(status)
        .ok()
        .filter(|status| (100..=599).contains(status))
        .ok_or_else(|| {
            HostError::invalid_args(format!("{route} status must be between 100 and 599"))
        })
}

pub(crate) fn text_arg(value: &Valor, name: &str) -> HostResult<String> {
    match value {
        Valor::Textus(value) | Valor::Instans(value) => Ok(value.clone()),
        _ => Err(HostError::invalid_args(format!("{name} must be textus"))),
    }
}

fn text_field(fields: &BTreeMap<String, Valor>, name: &str) -> HostResult<String> {
    fields
        .get(name)
        .ok_or_else(|| HostError::invalid_args(format!("HTTP header field `{name}` is missing")))
        .and_then(|value| text_arg(value, name))
}

pub(crate) fn bytes_value(value: &Valor, name: &str) -> HostResult<Vec<u8>> {
    match value {
        Valor::Octeti(bytes) => Ok(bytes.clone()),
        Valor::Textus(text) => Ok(text.as_bytes().to_vec()),
        _ => Err(HostError::invalid_args(format!(
            "{name} must be octeti or textus"
        ))),
    }
}

fn bounded_body_size(value: i64) -> HostResult<usize> {
    if value <= 0 {
        return Err(HostError::invalid_args("max_body_bytes must be positive"));
    }
    let value = usize::try_from(value)
        .map_err(|_| HostError::invalid_args("max_body_bytes is too large"))?;
    if value > MAX_CONFIGURED_BODY_BYTES {
        return Err(HostError::invalid_args(format!(
            "max_body_bytes must be at most {MAX_CONFIGURED_BODY_BYTES}"
        )));
    }
    Ok(value)
}

fn parse_bind_host(host: &str) -> HostResult<String> {
    if host.is_empty() || host.contains(['\0', '\r', '\n']) {
        return Err(HostError::invalid_args(
            "http:listen bind_host is malformed",
        ));
    }
    Ok(host.to_owned())
}

fn find_header_end(bytes: &[u8]) -> Option<usize> {
    bytes.windows(4).position(|window| window == b"\r\n\r\n")
}

fn is_valid_header_value(value: &str) -> bool {
    !value
        .bytes()
        .any(|byte| (byte < 0x20 && byte != b'\t') || byte == 0x7f)
}

pub(crate) fn is_token_byte(byte: u8) -> bool {
    matches!(
        byte,
        b'0'..=b'9'
            | b'a'..=b'z'
            | b'A'..=b'Z'
            | b'!'
            | b'#'
            | b'$'
            | b'%'
            | b'&'
            | b'\''
            | b'*'
            | b'+'
            | b'-'
            | b'.'
            | b'^'
            | b'_'
            | b'`'
            | b'|'
            | b'~'
    )
}

fn apply_accept_backlog(listener: &TcpListener) -> HostResult<()> {
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        // SAFETY: `listener` is an open TCP socket. A second listen(2) updates
        // the accept backlog on Darwin and Linux; excess SYNs are refused.
        let result = unsafe { libc::listen(listener.as_raw_fd(), ACCEPT_BACKLOG) };
        if result != 0 {
            return Err(HostError::internal(format!(
                "http:listen backlog failed: {}",
                io::Error::last_os_error()
            )));
        }
    }
    #[cfg(not(unix))]
    let _ = listener;
    Ok(())
}

pub(crate) fn lock<'a, T>(mutex: &'a Mutex<T>, label: &str) -> HostResult<MutexGuard<'a, T>> {
    mutex
        .lock()
        .map_err(|_| HostError::internal(format!("{label} lock poisoned")))
}

#[cfg(test)]
#[path = "http_test.rs"]
mod tests;
