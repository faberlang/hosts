//! HTTP client effects for generated Faber code.
//!
//! Moved from `faber-runtime/src/http.rs` in the faber-target-runtime S1-U3
//! split: the concrete HTTP client surface now has exactly one home — the
//! hosts `http` native package (server provider + client effects). The
//! faber/runtime/rust package keeps no HTTP implementation.

use crate::{
    bytes_value, headers_value, integer_arg, is_token_byte, list_args, lock, response_headers,
    text_arg, DEFAULT_MAX_BODY_BYTES,
};
use faber::Valor;
use host_kernel::{DispatchContext, HostError, HostResult, ProviderReply};
use std::collections::{BTreeMap, HashMap};
use std::io::Read;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Replicatio {
    status: i64,
    corpus: String,
    corpus_octeti: Vec<u8>,
    capita: HashMap<String, String>,
}

impl Replicatio {
    #[must_use]
    pub fn new(status: i64, corpus_octeti: Vec<u8>, capita: HashMap<String, String>) -> Self {
        let corpus = String::from_utf8_lossy(&corpus_octeti).into_owned();
        Self {
            status,
            corpus,
            corpus_octeti,
            capita: normalize_headers(capita),
        }
    }

    #[must_use]
    pub fn status(&self) -> i64 {
        self.status
    }

    #[must_use]
    pub fn corpus(&self) -> String {
        self.corpus.clone()
    }

    #[must_use]
    pub fn corpus_octeti(&self) -> Vec<u8> {
        self.corpus_octeti.clone()
    }

    pub fn corpus_json(&self) -> Valor {
        faber::Json::parse(&self.corpus).map_or(Valor::Nihil, Valor::from)
    }

    #[must_use]
    pub fn capita(&self) -> HashMap<String, String> {
        self.capita.clone()
    }

    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn caput(&self, nomen: String) -> Option<String> {
        self.capita.get(&nomen.to_ascii_lowercase()).cloned()
    }

    #[must_use]
    pub fn bene(&self) -> bool {
        (200..=299).contains(&self.status)
    }
}

pub async fn petet(url: String) -> Replicatio {
    rogabit("GET".to_owned(), url, HashMap::new(), String::new()).await
}

pub async fn mittet(url: String, corpus: String) -> Replicatio {
    rogabit("POST".to_owned(), url, HashMap::new(), corpus).await
}

pub async fn ponet(url: String, corpus: String) -> Replicatio {
    rogabit("PUT".to_owned(), url, HashMap::new(), corpus).await
}

pub async fn delet(url: String) -> Replicatio {
    rogabit("DELETE".to_owned(), url, HashMap::new(), String::new()).await
}

pub async fn mutabit(url: String, corpus: String) -> Replicatio {
    rogabit("PATCH".to_owned(), url, HashMap::new(), corpus).await
}

#[allow(clippy::implicit_hasher, clippy::unused_async)]
pub async fn rogabit(
    modus: String,
    url: String,
    capita: HashMap<String, String>,
    corpus: String,
) -> Replicatio {
    match http_request(&modus, &url, &capita, corpus.as_bytes()) {
        Ok(response) => response,
        Err(error) => Replicatio::new(
            599,
            error.into_bytes(),
            HashMap::from([("x-faber-error".to_owned(), "http-client".to_owned())]),
        ),
    }
}

fn http_request(
    method: &str,
    url: &str,
    headers: &HashMap<String, String>,
    body: &[u8],
) -> Result<Replicatio, String> {
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(30))
        .build();
    let mut request = agent.request(method, url);
    for (name, value) in headers {
        request = request.set(name, value);
    }

    let result = if body.is_empty() {
        request.call()
    } else {
        request.send_bytes(body)
    };
    match result {
        Ok(response) | Err(ureq::Error::Status(_, response)) => response_to_replicatio(response),
        Err(error) => Err(format!("http request failed: {error}")),
    }
}

fn response_to_replicatio(response: ureq::Response) -> Result<Replicatio, String> {
    let status = i64::from(response.status());
    let headers = response
        .headers_names()
        .into_iter()
        .filter_map(|name| response.header(&name).map(|value| (name, value.to_owned())))
        .collect::<HashMap<_, _>>();
    let mut body = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut body)
        .map_err(|error| format!("http body read failed: {error}"))?;
    Ok(Replicatio::new(status, body, headers))
}

fn normalize_headers(headers: HashMap<String, String>) -> HashMap<String, String> {
    headers
        .into_iter()
        .map(|(name, value)| (name.to_ascii_lowercase(), value))
        .collect()
}

const DEFAULT_AGENT_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_AGENT_TIMEOUT: Duration = Duration::from_mins(2);
const DEFAULT_READ_BYTES: usize = 64 * 1024;
const MAX_READ_BYTES: usize = 1024 * 1024;

pub(crate) struct ClientState {
    agents: Mutex<HashMap<i64, ureq::Agent>>,
    readers: Mutex<HashMap<i64, Arc<Mutex<ClientReader>>>>,
    next_agent: AtomicI64,
    next_reader: AtomicI64,
}

struct ClientReader {
    reader: Box<dyn Read + Send>,
    finished: bool,
}

enum AgentRef {
    Ephemeral,
    Handle(i64),
}

struct RequestArgs {
    agent: AgentRef,
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl ClientState {
    pub(crate) fn new() -> Self {
        Self {
            agents: Mutex::new(HashMap::new()),
            readers: Mutex::new(HashMap::new()),
            next_agent: AtomicI64::new(1),
            next_reader: AtomicI64::new(1),
        }
    }

    pub(crate) fn agent(&self, opener: &Valor) -> HostResult<ProviderReply> {
        let values = list_args(opener, "http:agent")?;
        if values.len() > 1 {
            return Err(HostError::invalid_args(
                "http:agent requires [] or [timeout_ms]",
            ));
        }
        let timeout = if values.is_empty() {
            DEFAULT_AGENT_TIMEOUT
        } else {
            agent_timeout(integer_arg(&values[0], "timeout_ms")?)?
        };
        let handle = self.next_agent.fetch_add(1, Ordering::SeqCst);
        let mut agents = lock(&self.agents, "http agents")?;
        agents.insert(handle, build_agent(timeout));
        Ok(ProviderReply::item(Valor::Numerus(handle)))
    }

    pub(crate) fn verb_get(&self, opener: &Valor) -> HostResult<ProviderReply> {
        self.oneshot(&parse_no_body_verb(opener, "GET")?)
    }

    pub(crate) fn verb_delete(&self, opener: &Valor) -> HostResult<ProviderReply> {
        self.oneshot(&parse_no_body_verb(opener, "DELETE")?)
    }

    pub(crate) fn verb_post(&self, opener: &Valor) -> HostResult<ProviderReply> {
        self.oneshot(&parse_body_verb(opener, "POST")?)
    }

    pub(crate) fn verb_put(&self, opener: &Valor) -> HostResult<ProviderReply> {
        self.oneshot(&parse_body_verb(opener, "PUT")?)
    }

    pub(crate) fn verb_patch(&self, opener: &Valor) -> HostResult<ProviderReply> {
        self.oneshot(&parse_body_verb(opener, "PATCH")?)
    }

    pub(crate) fn request(&self, opener: &Valor) -> HostResult<ProviderReply> {
        self.oneshot(&parse_generic_request(opener, "http:request")?)
    }

    pub(crate) fn request_open(&self, opener: &Valor) -> HostResult<ProviderReply> {
        let args = parse_generic_request(opener, "http:request_open")?;
        let agent = self.resolve_agent(&args.agent)?;
        let response = call_ureq(&agent, &args)?;
        let status = i64::from(response.status());
        let headers = collect_response_headers(&response);
        let reader = response.into_reader();
        let handle = self.next_reader.fetch_add(1, Ordering::SeqCst);
        {
            let mut readers = lock(&self.readers, "http readers")?;
            readers.insert(
                handle,
                Arc::new(Mutex::new(ClientReader {
                    reader: Box::new(reader),
                    finished: false,
                })),
            );
        }
        let mut fields = BTreeMap::new();
        fields.insert("reader".to_owned(), Valor::Numerus(handle));
        fields.insert("status".to_owned(), Valor::Numerus(status));
        fields.insert("headers".to_owned(), headers_value(&headers));
        Ok(ProviderReply::item(Valor::Tabula(fields)))
    }

    pub(crate) fn read(
        &self,
        opener: &Valor,
        context: &DispatchContext,
    ) -> HostResult<ProviderReply> {
        if context.cancellation.is_cancelled() {
            return Err(HostError::cancelled());
        }
        let values = list_args(opener, "http:read")?;
        if !(1..=2).contains(&values.len()) {
            return Err(HostError::invalid_args(
                "http:read requires [reader] or [reader, max_bytes]",
            ));
        }
        let handle = integer_arg(&values[0], "reader")?;
        let max_bytes = if values.len() == 2 {
            bounded_read_size(integer_arg(&values[1], "max_bytes")?)?
        } else {
            DEFAULT_READ_BYTES
        };
        let slot = {
            let readers = lock(&self.readers, "http readers")?;
            readers.get(&handle).cloned().ok_or_else(|| {
                HostError::invalid_args(format!("http:read unknown reader {handle}"))
            })?
        };
        let mut inner = lock(&slot, "http reader")?;
        if inner.finished {
            return Err(HostError::invalid_args(format!(
                "http:read reader {handle} is finished"
            )));
        }
        let mut buf = vec![0_u8; max_bytes];
        let count = loop {
            match inner.reader.read(&mut buf) {
                Ok(count) => break count,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => {
                    inner.finished = true;
                    drop(inner);
                    self.drop_reader(handle)?;
                    return Err(HostError::internal(format!("http:read failed: {error}")));
                }
            }
        };
        let done = count == 0;
        if done {
            inner.finished = true;
        }
        let bytes = buf[..count].to_vec();
        drop(inner);
        if done {
            self.drop_reader(handle)?;
        }
        let mut fields = BTreeMap::new();
        fields.insert("bytes".to_owned(), Valor::Octeti(bytes));
        fields.insert("done".to_owned(), Valor::Bivalens(done));
        Ok(ProviderReply::item(Valor::Tabula(fields)))
    }

    fn oneshot(&self, args: &RequestArgs) -> HostResult<ProviderReply> {
        let agent = self.resolve_agent(&args.agent)?;
        let response = call_ureq(&agent, args)?;
        let status = i64::from(response.status());
        if content_length_exceeds(&response, DEFAULT_MAX_BODY_BYTES) {
            return Err(HostError::invalid_args(format!(
                "http response body exceeds max_body_bytes {DEFAULT_MAX_BODY_BYTES}"
            )));
        }
        let headers = collect_response_headers(&response);
        let body = read_bounded(&mut response.into_reader(), DEFAULT_MAX_BODY_BYTES)?;
        Ok(ProviderReply::item(oneshot_carrier(status, &headers, body)))
    }

    fn resolve_agent(&self, agent: &AgentRef) -> HostResult<ureq::Agent> {
        match agent {
            AgentRef::Ephemeral => Ok(build_agent(DEFAULT_AGENT_TIMEOUT)),
            AgentRef::Handle(handle) => {
                let agents = lock(&self.agents, "http agents")?;
                agents
                    .get(handle)
                    .cloned()
                    .ok_or_else(|| HostError::invalid_args(format!("http unknown agent {handle}")))
            }
        }
    }

    fn drop_reader(&self, handle: i64) -> HostResult<()> {
        let mut readers = lock(&self.readers, "http readers")?;
        readers.remove(&handle);
        Ok(())
    }
}

fn parse_no_body_verb(opener: &Valor, method: &str) -> HostResult<RequestArgs> {
    let values = list_args(opener, &format!("http:{}", method.to_ascii_lowercase()))?;
    let (agent, rest) = take_agent(values);
    match rest {
        [url] => Ok(RequestArgs {
            agent,
            method: method.to_owned(),
            url: parse_url(url)?,
            headers: Vec::new(),
            body: Vec::new(),
        }),
        [url, headers] => Ok(RequestArgs {
            agent,
            method: method.to_owned(),
            url: parse_url(url)?,
            headers: response_headers(headers)?,
            body: Vec::new(),
        }),
        _ => Err(HostError::invalid_args(format!(
            "http:{} requires [url], [url, headers], [agent, url], or [agent, url, headers]",
            method.to_ascii_lowercase()
        ))),
    }
}

fn parse_body_verb(opener: &Valor, method: &str) -> HostResult<RequestArgs> {
    let values = list_args(opener, &format!("http:{}", method.to_ascii_lowercase()))?;
    let (agent, rest) = take_agent(values);
    match rest {
        [url, body] => Ok(RequestArgs {
            agent,
            method: method.to_owned(),
            url: parse_url(url)?,
            headers: Vec::new(),
            body: bytes_value(body, "body")?,
        }),
        [url, headers, body] => Ok(RequestArgs {
            agent,
            method: method.to_owned(),
            url: parse_url(url)?,
            headers: response_headers(headers)?,
            body: bytes_value(body, "body")?,
        }),
        _ => Err(HostError::invalid_args(format!(
            "http:{} requires [url, body], [url, headers, body], [agent, url, body], or [agent, url, headers, body]",
            method.to_ascii_lowercase()
        ))),
    }
}

fn parse_generic_request(opener: &Valor, route: &str) -> HostResult<RequestArgs> {
    let values = list_args(opener, route)?;
    let (agent, rest) = take_agent(values);
    match rest {
        [method, url, headers, body] => Ok(RequestArgs {
            agent,
            method: parse_method(method)?,
            url: parse_url(url)?,
            headers: response_headers(headers)?,
            body: bytes_value(body, "body")?,
        }),
        _ => Err(HostError::invalid_args(format!(
            "{route} requires [method, url, headers, body] or [agent, method, url, headers, body]"
        ))),
    }
}

fn take_agent(values: &[Valor]) -> (AgentRef, &[Valor]) {
    match values.first() {
        Some(Valor::Numerus(handle)) => (AgentRef::Handle(*handle), &values[1..]),
        _ => (AgentRef::Ephemeral, values),
    }
}

fn parse_method(value: &Valor) -> HostResult<String> {
    let method = text_arg(value, "method")?;
    if method.is_empty() || !method.bytes().all(is_token_byte) {
        return Err(HostError::invalid_args("http method is malformed"));
    }
    Ok(method)
}

fn parse_url(value: &Valor) -> HostResult<String> {
    let url = text_arg(value, "url")?;
    if url.is_empty() || url.contains(['\0', '\r', '\n', ' ']) {
        return Err(HostError::invalid_args("http url is malformed"));
    }
    Ok(url)
}

fn agent_timeout(timeout_ms: i64) -> HostResult<Duration> {
    if timeout_ms <= 0 {
        return Err(HostError::invalid_args("timeout_ms must be positive"));
    }
    let timeout_ms = u64::try_from(timeout_ms)
        .map_err(|_| HostError::invalid_args("timeout_ms is too large"))?;
    let timeout = Duration::from_millis(timeout_ms);
    if timeout > MAX_AGENT_TIMEOUT {
        return Err(HostError::invalid_args(format!(
            "timeout_ms must be at most {}",
            MAX_AGENT_TIMEOUT.as_millis()
        )));
    }
    Ok(timeout)
}

fn bounded_read_size(value: i64) -> HostResult<usize> {
    if value <= 0 {
        return Err(HostError::invalid_args("max_bytes must be positive"));
    }
    let value =
        usize::try_from(value).map_err(|_| HostError::invalid_args("max_bytes is too large"))?;
    if value > MAX_READ_BYTES {
        return Err(HostError::invalid_args(format!(
            "max_bytes must be at most {MAX_READ_BYTES}"
        )));
    }
    Ok(value)
}

fn build_agent(timeout: Duration) -> ureq::Agent {
    ureq::AgentBuilder::new().timeout(timeout).build()
}

fn call_ureq(agent: &ureq::Agent, args: &RequestArgs) -> HostResult<ureq::Response> {
    let mut request = agent.request(&args.method, &args.url);
    for (name, value) in &args.headers {
        request = request.set(name, value);
    }
    let result = if args.body.is_empty() {
        request.call()
    } else {
        request.send_bytes(&args.body)
    };
    match result {
        Ok(response) | Err(ureq::Error::Status(_, response)) => Ok(response),
        Err(error) => Err(HostError::internal(format!("http request failed: {error}"))),
    }
}

fn collect_response_headers(response: &ureq::Response) -> Vec<(String, String)> {
    response
        .headers_names()
        .into_iter()
        .filter_map(|name| response.header(&name).map(|value| (name, value.to_owned())))
        .collect()
}

fn content_length_exceeds(response: &ureq::Response, max: usize) -> bool {
    response
        .header("content-length")
        .and_then(|value| value.parse::<usize>().ok())
        .is_some_and(|length| length > max)
}

fn read_bounded(reader: &mut impl Read, max: usize) -> HostResult<Vec<u8>> {
    let mut body = Vec::new();
    let mut buf = [0_u8; 8192];
    loop {
        match reader.read(&mut buf) {
            Ok(0) => return Ok(body),
            Ok(count) => {
                if body.len().saturating_add(count) > max {
                    return Err(HostError::invalid_args(format!(
                        "http response body exceeds max_body_bytes {max}"
                    )));
                }
                body.extend_from_slice(&buf[..count]);
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => {
                return Err(HostError::internal(format!(
                    "http body read failed: {error}"
                )));
            }
        }
    }
}

fn oneshot_carrier(status: i64, headers: &[(String, String)], body: Vec<u8>) -> Valor {
    let mut fields = BTreeMap::new();
    fields.insert("status".to_owned(), Valor::Numerus(status));
    fields.insert("headers".to_owned(), headers_value(headers));
    fields.insert("body".to_owned(), Valor::Octeti(body));
    Valor::Tabula(fields)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_response_carrier() {
        let response = Replicatio::new(
            201,
            b"{\"ok\":true}".to_vec(),
            HashMap::from([("X-Faber-Test".to_owned(), "yes".to_owned())]),
        );

        assert_eq!(response.status(), 201);
        assert_eq!(
            response.caput("x-faber-test".to_owned()),
            Some("yes".to_owned())
        );
        assert!(matches!(response.corpus_json(), Valor::Tabula(_)));
        assert!(response.bene());
    }

    #[test]
    fn error_response_carrier_has_correct_fields() {
        let response = Replicatio::new(
            500,
            b"internal error".to_vec(),
            HashMap::from([("x-faber-error".to_owned(), "http-client".to_owned())]),
        );

        assert_eq!(response.status(), 500);
        assert_eq!(response.corpus(), "internal error");
        assert!(!response.bene());
        assert_eq!(
            response.caput("x-faber-error".to_owned()),
            Some("http-client".to_owned())
        );
    }

    #[test]
    fn empty_body_response_produces_nihil_json() {
        let response = Replicatio::new(200, b"".to_vec(), HashMap::new());

        assert_eq!(response.status(), 200);
        assert_eq!(response.corpus(), "");
        assert!(response.bene());
        assert_eq!(response.corpus_json(), Valor::Nihil);
    }

    #[test]
    fn error_status_not_between_200_and_299() {
        for status in [100, 199, 300, 404, 599] {
            let response = Replicatio::new(status, b"".to_vec(), HashMap::new());
            assert!(!response.bene(), "status {status} should not be bene");
        }
    }

    #[test]
    fn header_lookup_is_case_insensitive() {
        let response = Replicatio::new(
            200,
            b"ok".to_vec(),
            HashMap::from([("X-Custom-Header".to_owned(), "value".to_owned())]),
        );

        assert_eq!(
            response.caput("x-custom-header".to_owned()),
            Some("value".to_owned())
        );
        assert_eq!(
            response.caput("X-CUSTOM-HEADER".to_owned()),
            Some("value".to_owned())
        );
        assert_eq!(
            response.caput("X-Custom-Header".to_owned()),
            Some("value".to_owned())
        );
    }

    #[test]
    fn missing_header_returns_none() {
        let response = Replicatio::new(200, b"ok".to_vec(), HashMap::new());
        assert_eq!(response.caput("nonexistent".to_owned()), None);
    }

    #[test]
    fn status_code_is_exposed() {
        for status in [200, 201, 404, 500, 599] {
            let response = Replicatio::new(status, b"".to_vec(), HashMap::new());
            assert_eq!(response.status(), status);
        }
    }

    #[test]
    fn text_corpus_preserves_body_as_string() {
        let response = Replicatio::new(200, b"hello world".to_vec(), HashMap::new());
        assert_eq!(response.corpus(), "hello world");
    }

    #[test]
    fn binary_body_corpus_octeti_is_preserved() {
        let bytes: Vec<u8> = vec![0x00, 0x01, 0xFF, 0xFE];
        let response = Replicatio::new(200, bytes.clone(), HashMap::new());
        assert_eq!(response.corpus_octeti(), bytes);
    }
}
