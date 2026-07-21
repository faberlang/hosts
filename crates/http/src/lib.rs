//! Target-specific loopback HTTP/1.1 provider for the generic `http:*` family.
//!
//! The provider exposes only the manifest contract. Framework routing,
//! middleware, and request/response application semantics remain Faber code.

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
use std::time::Duration;

pub const DEFAULT_MAX_BODY_BYTES: usize = 1024 * 1024;
const MAX_CONFIGURED_BODY_BYTES: usize = 16 * 1024 * 1024;
const MAX_HEADER_BYTES: usize = 64 * 1024;
const POLL_INTERVAL: Duration = Duration::from_millis(5);
const WRITE_TIMEOUT: Duration = Duration::from_millis(250);
const REQUEST_ID_PREFIX: &str = "http-";

pub struct Http {
    registration: ProviderRegistration,
    state: Arc<HttpState>,
}

struct HttpState {
    listeners: Mutex<HashMap<i64, Arc<ListenerState>>>,
    pending: Mutex<BTreeMap<String, PendingRequest>>,
    next_listener: AtomicI64,
    next_request: AtomicU64,
}

struct ListenerState {
    listener: TcpListener,
    max_body_bytes: usize,
    stopped: AtomicBool,
}

struct PendingRequest {
    listener: i64,
    stream: TcpStream,
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
                pending: Mutex::new(BTreeMap::new()),
                next_listener: AtomicI64::new(1),
                next_request: AtomicU64::new(1),
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
            "http:stop" => self.stop(&request.opener),
            other => Err(HostError::no_route(format!(
                "no built-in http syscall registered for {other}"
            ))),
        }
    }
}

impl Http {
    fn listen(&self, opener: &Valor) -> HostResult<ProviderReply> {
        let values = list_args(opener, "http:listen")?;
        if !(1..=2).contains(&values.len()) {
            return Err(HostError::invalid_args(
                "http:listen requires [port] or [port, max_body_bytes]",
            ));
        }
        let port = integer_arg(&values[0], "port")?;
        let port = u16::try_from(port)
            .map_err(|_| HostError::invalid_args("http:listen port must be between 0 and 65535"))?;
        let max_body_bytes = match values.get(1) {
            Some(value) => bounded_body_size(integer_arg(value, "max_body_bytes")?)?,
            None => DEFAULT_MAX_BODY_BYTES,
        };
        let listener = TcpListener::bind(("127.0.0.1", port))
            .map_err(|error| HostError::internal(format!("http:listen bind failed: {error}")))?;
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
            match listener.listener.accept() {
                Ok((stream, _peer)) => {
                    let (parts, stream) = read_request(
                        stream,
                        listener.max_body_bytes,
                        &listener.stopped,
                        context,
                        &self.state.next_request,
                    )?;
                    let mut pending = lock(&self.state.pending, "http pending requests")?;
                    if listener.stopped.load(Ordering::SeqCst) {
                        return Err(HostError::invalid_args("http:accept listener is stopped"));
                    }
                    let request = request_carrier(&parts);
                    pending.insert(
                        parts.id.clone(),
                        PendingRequest {
                            listener: handle,
                            stream,
                        },
                    );
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

    fn respond(&self, opener: &Valor) -> HostResult<ProviderReply> {
        let values = list_args(opener, "http:respond")?;
        if values.len() != 4 {
            return Err(HostError::invalid_args(
                "http:respond requires [request_id, status, headers, body]",
            ));
        }
        let request_id = text_arg(&values[0], "request_id")?;
        let status = integer_arg(&values[1], "status")?;
        let status = u16::try_from(status)
            .ok()
            .filter(|status| (100..=599).contains(status))
            .ok_or_else(|| {
                HostError::invalid_args("http:respond status must be between 100 and 599")
            })?;
        let headers = response_headers(&values[2])?;
        let body = bytes_value(&values[3], "body")?;
        let mut pending = lock(&self.state.pending, "http pending requests")?;
        let mut request = pending.remove(&request_id).ok_or_else(|| {
            HostError::invalid_args(format!("http:respond unknown request {request_id}"))
        })?;
        drop(pending);

        request
            .stream
            .set_nonblocking(false)
            .map_err(|error| HostError::internal(format!("http:respond setup failed: {error}")))?;
        request
            .stream
            .set_write_timeout(Some(WRITE_TIMEOUT))
            .map_err(|error| {
                HostError::internal(format!("http:respond timeout setup failed: {error}"))
            })?;
        let response = format_response(status, &request_id, &headers, &body);
        let write_result = request.stream.write_all(&response);
        let _ = request.stream.shutdown(Shutdown::Both);
        write_result
            .map_err(|error| HostError::internal(format!("http:respond write failed: {error}")))?;
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
        let mut pending = lock(&self.state.pending, "http pending requests")?;
        let request_ids = pending
            .iter()
            .filter(|(_, request)| request.listener == handle)
            .map(|(request_id, _)| request_id.clone())
            .collect::<Vec<_>>();
        let requests = request_ids
            .into_iter()
            .filter_map(|request_id| pending.remove(&request_id))
            .map(|request| request.stream)
            .collect::<Vec<_>>();
        drop(pending);
        for stream in requests {
            let _ = stream.shutdown(Shutdown::Both);
        }
        Ok(ProviderReply::vacuum())
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
    mut stream: TcpStream,
    max_body_bytes: usize,
    stopped: &AtomicBool,
    context: &DispatchContext,
    next_request: &AtomicU64,
) -> HostResult<(RequestParts, TcpStream)> {
    stream
        .set_nonblocking(true)
        .map_err(|error| HostError::internal(format!("http:accept setup failed: {error}")))?;
    let mut bytes = Vec::new();
    let header_end = loop {
        check_read_state(stopped, context)?;
        let mut chunk = [0_u8; 4096];
        match stream.read(&mut chunk) {
            Ok(0) => {
                return Err(HostError::invalid_args(
                    "http:accept request ended before headers",
                ))
            }
            Ok(count) => {
                bytes.extend_from_slice(&chunk[..count]);
                if let Some(end) = find_header_end(&bytes) {
                    break end;
                }
                if bytes.len() > MAX_HEADER_BYTES {
                    return Err(HostError::invalid_args(
                        "http:accept request headers exceed limit",
                    ));
                }
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => thread::sleep(POLL_INTERVAL),
            Err(error) => {
                return Err(HostError::internal(format!(
                    "http:accept read failed: {error}"
                )))
            }
        }
    };

    let ParsedHeaders {
        method,
        path,
        headers,
        content_length,
    } = parse_headers(&bytes[..header_end])?;
    if content_length > max_body_bytes {
        return Err(HostError::invalid_args(format!(
            "http:accept request body exceeds max_body_bytes {max_body_bytes}"
        )));
    }
    let body_start = header_end + 4;
    while bytes.len() < body_start + content_length {
        check_read_state(stopped, context)?;
        let mut chunk = [0_u8; 4096];
        match stream.read(&mut chunk) {
            Ok(0) => {
                return Err(HostError::invalid_args(
                    "http:accept request ended before body",
                ))
            }
            Ok(count) => {
                bytes.extend_from_slice(&chunk[..count]);
                if bytes.len() > body_start + content_length {
                    break;
                }
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => thread::sleep(POLL_INTERVAL),
            Err(error) => {
                return Err(HostError::internal(format!(
                    "http:accept body read failed: {error}"
                )))
            }
        }
    }
    let body = bytes[body_start..body_start + content_length].to_vec();
    let id = format!(
        "{REQUEST_ID_PREFIX}{}",
        next_request.fetch_add(1, Ordering::SeqCst)
    );
    Ok((RequestParts::new(id, method, path, headers, body), stream))
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

fn request_carrier(parts: &RequestParts) -> Valor {
    let mut fields = BTreeMap::new();
    fields.insert("id".to_owned(), Valor::Textus(parts.id.clone()));
    fields.insert("method".to_owned(), Valor::Textus(parts.method.clone()));
    fields.insert("path".to_owned(), Valor::Textus(parts.path.clone()));
    fields.insert("headers".to_owned(), headers_value(&parts.headers));
    fields.insert("body".to_owned(), Valor::Octeti(parts.body.clone()));
    Valor::Tabula(fields)
}

fn headers_value(headers: &[(String, String)]) -> Valor {
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

fn response_headers(value: &Valor) -> HostResult<Vec<(String, String)>> {
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
    if name.eq_ignore_ascii_case("connection") || name.eq_ignore_ascii_case("content-length") {
        return Err(HostError::invalid_args(
            "HTTP connection and content-length headers are provider-owned",
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
    for (name, value) in headers {
        response.extend_from_slice(name.as_bytes());
        response.extend_from_slice(b": ");
        response.extend_from_slice(value.as_bytes());
        response.extend_from_slice(b"\r\n");
    }
    response.extend_from_slice(b"\r\n");
    response.extend_from_slice(body);
    response
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

fn list_args<'a>(value: &'a Valor, route: &str) -> HostResult<&'a [Valor]> {
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

fn integer_arg(value: &Valor, name: &str) -> HostResult<i64> {
    match value {
        Valor::Numerus(value) => Ok(*value),
        _ => Err(HostError::invalid_args(format!("{name} must be numerus"))),
    }
}

fn text_arg(value: &Valor, name: &str) -> HostResult<String> {
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

fn bytes_value(value: &Valor, name: &str) -> HostResult<Vec<u8>> {
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

fn find_header_end(bytes: &[u8]) -> Option<usize> {
    bytes.windows(4).position(|window| window == b"\r\n\r\n")
}

fn is_valid_header_value(value: &str) -> bool {
    !value
        .bytes()
        .any(|byte| (byte < 0x20 && byte != b'\t') || byte == 0x7f)
}

fn is_token_byte(byte: u8) -> bool {
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

fn lock<'a, T>(mutex: &'a Mutex<T>, label: &str) -> HostResult<MutexGuard<'a, T>> {
    mutex
        .lock()
        .map_err(|_| HostError::internal(format!("{label} lock poisoned")))
}

#[cfg(test)]
#[path = "http_test.rs"]
mod tests;
