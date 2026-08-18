use super::*;
use host_kernel::{CancellationProbe, ProviderContent};
use std::collections::BTreeMap;
use std::io::Read;
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

fn context() -> DispatchContext {
    DispatchContext {
        cancellation: CancellationProbe::new(|| false),
    }
}

fn request(route: &str, opener: Valor) -> RequestFrame {
    RequestFrame {
        conversation_id: route.to_owned(),
        route: route.to_owned(),
        opener,
        target: None,
    }
}

fn listen(provider: &Http, port: i64) -> i64 {
    let reply = provider
        .dispatch(
            &request("http:listen", Valor::Lista(vec![Valor::Numerus(port)])),
            &context(),
        )
        .expect("listen");
    let [ProviderContent::Item(Valor::Numerus(handle))] = reply.contents.as_slice() else {
        panic!("http:listen must return one numerus handle");
    };
    *handle
}

fn free_port() -> u16 {
    TcpListener::bind(("127.0.0.1", 0))
        .expect("reserve loopback port")
        .local_addr()
        .expect("local address")
        .port()
}

fn headers(entries: &[(&str, &str)]) -> Valor {
    Valor::Lista(
        entries
            .iter()
            .map(|(name, value)| {
                let mut fields = BTreeMap::new();
                fields.insert("name".to_owned(), Valor::Textus((*name).to_owned()));
                fields.insert("value".to_owned(), Valor::Textus((*value).to_owned()));
                Valor::Tabula(fields)
            })
            .collect(),
    )
}

fn accept_bounded(provider: &Arc<Http>, handle: i64) -> HostResult<ProviderReply> {
    let cancelled = Arc::new(AtomicBool::new(false));
    let accept_provider = Arc::clone(provider);
    let cancellation = Arc::clone(&cancelled);
    let (result_tx, result_rx) = mpsc::channel();
    let accepted = thread::spawn(move || {
        let result = accept_provider.dispatch(
            &request("http:accept", Valor::Numerus(handle)),
            &DispatchContext {
                cancellation: CancellationProbe::from_flag(cancellation),
            },
        );
        result_tx.send(result).expect("send accept result");
    });
    let result = match result_rx.recv_timeout(Duration::from_secs(1)) {
        Ok(result) => result,
        Err(mpsc::RecvTimeoutError::Timeout) => {
            cancelled.store(true, Ordering::SeqCst);
            result_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("cancelled accept must finish")
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            cancelled.store(true, Ordering::SeqCst);
            accepted.join().expect("accept thread");
            panic!("accept thread disconnected before returning");
        }
    };
    accepted.join().expect("accept thread");
    result
}

#[test]
fn manifest_registers_http_contract_and_routes() {
    let mut kernel = Kernel::new();
    register(&mut kernel).expect("register http");
    let manifest = &kernel.manifest().providers[0];
    assert_eq!(manifest.provider, "http");
    assert_eq!(manifest.prefixes, ["http"]);
    assert_eq!(
        manifest
            .calls
            .iter()
            .map(|call| (
                call.route.as_str(),
                call.opener.as_str(),
                call.result.as_str()
            ))
            .collect::<Vec<_>>(),
        vec![
            ("http:listen", "lista<valor>", "numerus"),
            ("http:accept", "numerus", "valor"),
            ("http:respond", "lista<valor>", "vacuum"),
            ("http:respond_open", "lista<valor>", "numerus"),
            ("http:respond_chunk", "lista<valor>", "vacuum"),
            ("http:respond_finish", "lista<valor>", "vacuum"),
            ("http:stop", "numerus", "vacuum"),
        ]
    );

    let reply = kernel
        .dispatch(
            &request(
                "http:listen",
                Valor::Lista(vec![Valor::Numerus(i64::from(free_port()))]),
            ),
            &context(),
        )
        .expect("kernel manifest-admitted listen");
    let [ProviderContent::Item(Valor::Numerus(handle))] = reply.contents.as_slice() else {
        panic!("kernel listen reply must be one numerus handle");
    };
    kernel
        .dispatch(&request("http:stop", Valor::Numerus(*handle)), &context())
        .expect("kernel manifest-admitted stop");
}

#[test]
fn localhost_request_accepts_and_responds_once() {
    let provider = Arc::new(Http::new().expect("provider"));
    let port = free_port();
    let handle = listen(&provider, i64::from(port));
    let accept_provider = Arc::clone(&provider);
    let accepted = thread::spawn(move || {
        accept_provider
            .dispatch(&request("http:accept", Valor::Numerus(handle)), &context())
            .expect("accept")
    });

    thread::sleep(Duration::from_millis(10));
    let mut client = TcpStream::connect(("127.0.0.1", port)).expect("connect localhost");
    client
        .write_all(b"POST /salve HTTP/1.1\r\nHost: localhost\r\nContent-Length: 5\r\n\r\nhello")
        .expect("write request");
    let reply = accepted.join().expect("accept thread");
    let [ProviderContent::Item(Valor::Tabula(fields))] = reply.contents.as_slice() else {
        panic!("http:accept must return one request table");
    };
    assert_eq!(fields.get("method"), Some(&Valor::Textus("POST".into())));
    assert_eq!(fields.get("path"), Some(&Valor::Textus("/salve".into())));
    assert_eq!(fields.get("body"), Some(&Valor::Octeti(b"hello".to_vec())));
    let request_id = match fields.get("id") {
        Some(Valor::Textus(value)) => value.clone(),
        other => panic!("missing request id: {other:?}"),
    };

    provider
        .dispatch(
            &request(
                "http:respond",
                Valor::Lista(vec![
                    Valor::Textus(request_id.clone()),
                    Valor::Numerus(200),
                    headers(&[("content-type", "text/plain")]),
                    Valor::Textus("world".into()),
                ]),
            ),
            &context(),
        )
        .expect("respond");
    let mut response = String::new();
    client
        .set_read_timeout(Some(Duration::from_secs(1)))
        .expect("read timeout");
    client.read_to_string(&mut response).expect("read response");
    assert!(response.starts_with("HTTP/1.1 200 OK\r\n"));
    assert!(response.contains("x-faber-request-id: http-"));
    assert!(response.ends_with("\r\nworld"));

    provider
        .dispatch(&request("http:stop", Valor::Numerus(handle)), &context())
        .expect("stop");
}

#[test]
fn accept_cancellation_is_bounded() {
    let provider = Arc::new(Http::new().expect("provider"));
    let handle = listen(&provider, i64::from(free_port()));
    let cancelled = Arc::new(AtomicBool::new(false));
    let accept_provider = Arc::clone(&provider);
    let cancellation = Arc::clone(&cancelled);
    let accepted = thread::spawn(move || {
        accept_provider.dispatch(
            &request("http:accept", Valor::Numerus(handle)),
            &DispatchContext {
                cancellation: CancellationProbe::from_flag(cancellation),
            },
        )
    });
    thread::sleep(Duration::from_millis(20));
    cancelled.store(true, Ordering::SeqCst);
    let error = accepted
        .join()
        .expect("accept thread")
        .expect_err("cancelled accept");
    assert_eq!(error.code, "E_CANCELLED");
}

#[test]
fn malformed_and_oversized_requests_fail_closed() {
    let provider = Arc::new(Http::new().expect("provider"));
    let port = free_port();
    let handle = listen(&provider, i64::from(port));
    let accept_provider = Arc::clone(&provider);
    let accepted = thread::spawn(move || {
        accept_provider.dispatch(&request("http:accept", Valor::Numerus(handle)), &context())
    });
    thread::sleep(Duration::from_millis(10));
    let mut malformed = TcpStream::connect(("127.0.0.1", port)).expect("connect malformed");
    malformed
        .write_all(b"GET / HTTP/1.0\r\n\r\n")
        .expect("write malformed");
    let error = accepted
        .join()
        .expect("accept thread")
        .expect_err("malformed request");
    assert_eq!(error.code, "E_INVALID_ARGS");

    provider
        .dispatch(&request("http:stop", Valor::Numerus(handle)), &context())
        .expect("stop malformed listener");

    let limited_port = free_port();
    let limited_handle = provider
        .dispatch(
            &request(
                "http:listen",
                Valor::Lista(vec![
                    Valor::Numerus(i64::from(limited_port)),
                    Valor::Numerus(3),
                ]),
            ),
            &context(),
        )
        .expect("listen limited")
        .contents;
    let ProviderContent::Item(Valor::Numerus(limited_handle)) =
        limited_handle.into_iter().next().expect("handle")
    else {
        panic!("limited listener handle");
    };
    let accept_provider = Arc::clone(&provider);
    let accepted = thread::spawn(move || {
        accept_provider.dispatch(
            &request("http:accept", Valor::Numerus(limited_handle)),
            &context(),
        )
    });
    thread::sleep(Duration::from_millis(10));
    let mut oversized = TcpStream::connect(("127.0.0.1", limited_port)).expect("connect oversized");
    oversized
        .write_all(b"POST / HTTP/1.1\r\nContent-Length: 4\r\n\r\ntoolong")
        .expect("write oversized");
    let error = accepted
        .join()
        .expect("accept thread")
        .expect_err("oversized request");
    assert_eq!(error.code, "E_INVALID_ARGS");
    provider
        .dispatch(
            &request("http:stop", Valor::Numerus(limited_handle)),
            &context(),
        )
        .expect("stop limited listener");
}

#[test]
fn stop_unblocks_pending_accept() {
    let provider = Arc::new(Http::new().expect("provider"));
    let handle = listen(&provider, i64::from(free_port()));
    let accept_provider = Arc::clone(&provider);
    let accepted = thread::spawn(move || {
        accept_provider.dispatch(&request("http:accept", Valor::Numerus(handle)), &context())
    });
    thread::sleep(Duration::from_millis(20));
    provider
        .dispatch(&request("http:stop", Valor::Numerus(handle)), &context())
        .expect("stop");
    let error = accepted
        .join()
        .expect("accept thread")
        .expect_err("stopped accept");
    assert_eq!(error.code, "E_INVALID_ARGS");
}

#[test]
fn http11_without_host_is_rejected() {
    let provider = Arc::new(Http::new().expect("provider"));
    let port = free_port();
    let handle = listen(&provider, i64::from(port));
    let mut client = TcpStream::connect(("127.0.0.1", port)).expect("connect no-host request");
    client
        .write_all(b"GET / HTTP/1.1\r\n\r\n")
        .expect("write no-host request");
    let result = accept_bounded(&provider, handle);
    provider
        .dispatch(&request("http:stop", Valor::Numerus(handle)), &context())
        .expect("stop no-host listener");
    let error = result.expect_err("HTTP/1.1 without Host must be rejected");
    assert_eq!(error.code, "E_INVALID_ARGS");
}

#[test]
fn request_header_control_bytes_are_rejected() {
    let provider = Arc::new(Http::new().expect("provider"));
    let port = free_port();
    let handle = listen(&provider, i64::from(port));
    let mut client = TcpStream::connect(("127.0.0.1", port)).expect("connect control-byte request");
    client
        .write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\nX-Test: valid\x01invalid\r\n\r\n")
        .expect("write control-byte request");
    let result = accept_bounded(&provider, handle);
    provider
        .dispatch(&request("http:stop", Valor::Numerus(handle)), &context())
        .expect("stop control-byte listener");
    let error = result.expect_err("request header control byte must be rejected");
    assert_eq!(error.code, "E_INVALID_ARGS");
}

#[test]
fn response_header_control_bytes_are_rejected() {
    let provider = Arc::new(Http::new().expect("provider"));
    let port = free_port();
    let handle = listen(&provider, i64::from(port));
    let mut client = TcpStream::connect(("127.0.0.1", port)).expect("connect response request");
    client
        .write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .expect("write response request");
    let reply = accept_bounded(&provider, handle).expect("valid request");
    let [ProviderContent::Item(Valor::Tabula(fields))] = reply.contents.as_slice() else {
        panic!("http:accept must return one request table");
    };
    let request_id = match fields.get("id") {
        Some(Valor::Textus(value)) => value.clone(),
        other => panic!("missing request id: {other:?}"),
    };
    let result = provider.dispatch(
        &request(
            "http:respond",
            Valor::Lista(vec![
                Valor::Textus(request_id),
                Valor::Numerus(200),
                headers(&[("x-test", "valid\u{1}invalid")]),
                Valor::Textus("body".into()),
            ]),
        ),
        &context(),
    );
    provider
        .dispatch(&request("http:stop", Valor::Numerus(handle)), &context())
        .expect("stop response listener");
    let error = result.expect_err("response header control byte must be rejected");
    assert_eq!(error.code, "E_INVALID_ARGS");
}

#[test]
fn respond_with_unknown_id_returns_error() {
    let provider = Http::new().expect("provider");
    let result = provider.dispatch(
        &request(
            "http:respond",
            Valor::Lista(vec![
                Valor::Textus("http-no-such-request".into()),
                Valor::Numerus(200),
                headers(&[("content-type", "text/plain")]),
                Valor::Textus("body".into()),
            ]),
        ),
        &context(),
    );
    let error = result.expect_err("response to unknown id must fail");
    assert_eq!(error.code, "E_INVALID_ARGS");
}

#[test]
fn stop_invalid_handle_returns_error() {
    let provider = Http::new().expect("provider");
    let result = provider.dispatch(&request("http:stop", Valor::Numerus(9999)), &context());
    let error = result.expect_err("stop invalid handle must fail");
    assert_eq!(error.code, "E_INVALID_ARGS");
}

#[test]
fn listen_on_port_below_zero_or_above_u16_rejected() {
    let provider = Http::new().expect("provider");
    // Port 0 is valid (ephemeral). Reject only out-of-range ports.
    for port in [-1, -8080, 65536, i64::MIN] {
        let result = provider.dispatch(
            &request("http:listen", Valor::Lista(vec![Valor::Numerus(port)])),
            &context(),
        );
        let error = result.expect_err("invalid port must fail");
        assert_eq!(error.code, "E_INVALID_ARGS", "port {port}");
    }
}

fn request_table(reply: &ProviderReply) -> &BTreeMap<String, Valor> {
    let [ProviderContent::Item(Valor::Tabula(fields))] = reply.contents.as_slice() else {
        panic!("http:accept must return one request table");
    };
    fields
}

fn text_field<'a>(fields: &'a BTreeMap<String, Valor>, name: &str) -> &'a str {
    match fields.get(name) {
        Some(Valor::Textus(value)) => value,
        other => panic!("missing {name}: {other:?}"),
    }
}

fn numerus_field(fields: &BTreeMap<String, Valor>, name: &str) -> i64 {
    match fields.get(name) {
        Some(Valor::Numerus(value)) => *value,
        other => panic!("missing {name}: {other:?}"),
    }
}

fn writer_handle(reply: &ProviderReply) -> i64 {
    let [ProviderContent::Item(Valor::Numerus(handle))] = reply.contents.as_slice() else {
        panic!("http:respond_open must return one numerus writer");
    };
    *handle
}

fn respond_open(provider: &Http, request_id: &str, status: i64, header_list: Valor) -> i64 {
    writer_handle(
        &provider
            .dispatch(
                &request(
                    "http:respond_open",
                    Valor::Lista(vec![
                        Valor::Textus(request_id.to_owned()),
                        Valor::Numerus(status),
                        header_list,
                    ]),
                ),
                &context(),
            )
            .expect("respond_open"),
    )
}

fn respond_chunk(provider: &Http, writer: i64, bytes: &[u8]) -> HostResult<ProviderReply> {
    provider.dispatch(
        &request(
            "http:respond_chunk",
            Valor::Lista(vec![Valor::Numerus(writer), Valor::Octeti(bytes.to_vec())]),
        ),
        &context(),
    )
}

fn respond_finish(provider: &Http, writer: i64, keep_alive: bool) -> HostResult<ProviderReply> {
    provider.dispatch(
        &request(
            "http:respond_finish",
            Valor::Lista(vec![Valor::Numerus(writer), Valor::Bivalens(keep_alive)]),
        ),
        &context(),
    )
}

struct Wire {
    stream: TcpStream,
    buf: Vec<u8>,
}

impl Wire {
    fn connect(port: u16) -> Self {
        let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("read timeout");
        stream
            .set_write_timeout(Some(Duration::from_secs(2)))
            .expect("write timeout");
        Self {
            stream,
            buf: Vec::new(),
        }
    }

    fn send(&mut self, bytes: &[u8]) {
        self.stream.write_all(bytes).expect("write request");
    }

    fn send_get(&mut self, path: &str) {
        self.send(format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n").as_bytes());
    }

    fn read_headers(&mut self) -> String {
        loop {
            if let Some(end) = find_header_end(&self.buf) {
                let headers = String::from_utf8_lossy(&self.buf[..=end + 3]).into_owned();
                self.buf.drain(..=end + 3);
                return headers;
            }
            self.pull();
        }
    }

    fn read_chunked_body(&mut self) -> Vec<u8> {
        let mut body = Vec::new();
        loop {
            let line = self.read_line();
            let size = usize::from_str_radix(line.trim(), 16).expect("chunk size");
            if size == 0 {
                let trailer = self.read_line();
                assert!(trailer.is_empty(), "chunked trailer must be empty");
                return body;
            }
            while self.buf.len() < size + 2 {
                self.pull();
            }
            body.extend_from_slice(&self.buf[..size]);
            assert_eq!(&self.buf[size..size + 2], b"\r\n");
            self.buf.drain(..size + 2);
        }
    }

    fn read_line(&mut self) -> String {
        loop {
            if let Some(pos) = self.buf.windows(2).position(|window| window == b"\r\n") {
                let line = String::from_utf8_lossy(&self.buf[..pos]).into_owned();
                self.buf.drain(..pos + 2);
                return line;
            }
            self.pull();
        }
    }

    fn pull(&mut self) {
        let mut chunk = [0_u8; 4096];
        let count = self.stream.read(&mut chunk).expect("read wire");
        assert_ne!(count, 0, "peer closed while reading");
        self.buf.extend_from_slice(&chunk[..count]);
    }

    fn drain(&mut self, want: usize) -> usize {
        let mut got = 0;
        while got < want {
            if !self.buf.is_empty() {
                let take = want.saturating_sub(got).min(self.buf.len());
                self.buf.drain(..take);
                got += take;
                continue;
            }
            let mut chunk = [0_u8; 4096];
            match self.stream.read(&mut chunk) {
                Ok(0) => break,
                Ok(count) => got += count,
                Err(error)
                    if error.kind() == std::io::ErrorKind::WouldBlock
                        || error.kind() == std::io::ErrorKind::TimedOut =>
                {
                    break
                }
                Err(error) => panic!("drain failed: {error}"),
            }
        }
        got
    }
}

fn accept_one(provider: &Arc<Http>, handle: i64) -> (String, i64) {
    let accept_provider = Arc::clone(provider);
    let accepted = thread::spawn(move || {
        accept_provider
            .dispatch(&request("http:accept", Valor::Numerus(handle)), &context())
            .expect("accept")
    });
    let reply = accepted.join().expect("accept thread");
    let fields = request_table(&reply);
    (
        text_field(fields, "id").to_owned(),
        numerus_field(fields, "connection"),
    )
}

#[test]
fn streaming_response_writes_chunked_frames() {
    let provider = Arc::new(Http::new().expect("provider"));
    let port = free_port();
    let handle = listen(&provider, i64::from(port));
    let mut client = Wire::connect(port);
    client.send_get("/stream");
    let (request_id, _connection) = accept_one(&provider, handle);
    let writer = respond_open(
        &provider,
        &request_id,
        200,
        headers(&[("content-type", "text/plain")]),
    );
    respond_chunk(&provider, writer, b"hello").expect("first chunk");
    respond_chunk(&provider, writer, b" world").expect("second chunk");
    respond_finish(&provider, writer, false).expect("finish stream");
    let head = client.read_headers();
    assert!(head.starts_with("HTTP/1.1 200 OK\r\n"));
    assert!(head
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked"));
    assert!(head.contains("x-faber-request-id: http-"));
    assert_eq!(client.read_chunked_body(), b"hello world");
    provider
        .dispatch(&request("http:stop", Valor::Numerus(handle)), &context())
        .expect("stop stream listener");
}

#[test]
fn keep_alive_reuses_one_connection_for_two_requests() {
    let provider = Arc::new(Http::new().expect("provider"));
    let port = free_port();
    let handle = listen(&provider, i64::from(port));
    let mut client = Wire::connect(port);
    client.send_get("/one");
    let (first_id, first_connection) = accept_one(&provider, handle);
    let writer = respond_open(
        &provider,
        &first_id,
        200,
        headers(&[("content-type", "text/plain")]),
    );
    respond_chunk(&provider, writer, b"one").expect("keep-alive chunk one");
    respond_finish(&provider, writer, true).expect("keep-alive finish");
    let head = client.read_headers();
    assert!(head.starts_with("HTTP/1.1 200 OK\r\n"));
    assert_eq!(client.read_chunked_body(), b"one");

    client.send_get("/two");
    let (second_id, second_connection) = accept_one(&provider, handle);
    assert_eq!(first_connection, second_connection);
    assert_ne!(first_id, second_id);
    let writer = respond_open(
        &provider,
        &second_id,
        200,
        headers(&[("content-type", "text/plain")]),
    );
    respond_chunk(&provider, writer, b"two").expect("keep-alive chunk two");
    respond_finish(&provider, writer, true).expect("keep-alive finish two");
    let head = client.read_headers();
    assert!(head.starts_with("HTTP/1.1 200 OK\r\n"));
    assert_eq!(client.read_chunked_body(), b"two");
    provider
        .dispatch(&request("http:stop", Valor::Numerus(handle)), &context())
        .expect("stop keep-alive listener");
}

#[test]
fn respond_chunk_blocks_under_backpressure() {
    let provider = Arc::new(Http::new().expect("provider"));
    let port = free_port();
    let handle = listen(&provider, i64::from(port));
    let mut client = Wire::connect(port);
    client.send_get("/backpressure");
    let (request_id, _connection) = accept_one(&provider, handle);
    let writer = respond_open(
        &provider,
        &request_id,
        200,
        headers(&[("content-type", "application/octet-stream")]),
    );
    let payload = vec![b'x'; 8 * 1024 * 1024];
    let chunk_provider = Arc::clone(&provider);
    let chunk_bytes = payload.clone();
    let (done_tx, done_rx) = mpsc::channel();
    let started = Instant::now();
    let writer_thread = thread::spawn(move || {
        let result = respond_chunk(&chunk_provider, writer, &chunk_bytes);
        done_tx
            .send(started.elapsed())
            .expect("send backpressure elapsed");
        result
    });
    thread::sleep(Duration::from_millis(80));
    assert!(
        done_rx.try_recv().is_err(),
        "respond_chunk must stay blocked while the client is not reading"
    );
    let _head = client.read_headers();
    let drained = client.drain(payload.len());
    assert!(drained > 0, "client must relieve send-buffer backpressure");
    let blocked_for = done_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("blocked chunk must finish after the client reads");
    writer_thread
        .join()
        .expect("chunk thread")
        .expect("chunk after drain");
    assert!(
        blocked_for >= Duration::from_millis(50),
        "respond_chunk returned too quickly: {blocked_for:?}"
    );
    respond_finish(&provider, writer, false).expect("finish backpressure stream");
    provider
        .dispatch(&request("http:stop", Valor::Numerus(handle)), &context())
        .expect("stop backpressure listener");
}

#[test]
fn stop_during_stream_fails_the_in_flight_write() {
    let provider = Arc::new(Http::new().expect("provider"));
    let port = free_port();
    let handle = listen(&provider, i64::from(port));
    let mut client = Wire::connect(port);
    client.send_get("/stop-stream");
    let (request_id, _connection) = accept_one(&provider, handle);
    let writer = respond_open(
        &provider,
        &request_id,
        200,
        headers(&[("content-type", "application/octet-stream")]),
    );
    let chunk_provider = Arc::clone(&provider);
    let payload = vec![b'y'; 8 * 1024 * 1024];
    let writer_thread = thread::spawn(move || respond_chunk(&chunk_provider, writer, &payload));
    thread::sleep(Duration::from_millis(20));
    provider
        .dispatch(&request("http:stop", Valor::Numerus(handle)), &context())
        .expect("stop during stream");
    let error = writer_thread
        .join()
        .expect("chunk thread")
        .expect_err("stop must fail the in-flight chunk");
    assert!(
        error.code == "E_INVALID_ARGS" || error.code == "E_INTERNAL" || error.code == "E_CANCELLED",
        "unexpected stop-during-stream code {}",
        error.code
    );
}

#[test]
fn listen_bind_host_and_empty_host_are_honored() {
    let provider = Http::new().expect("provider");
    let port = free_port();
    let handle = provider
        .dispatch(
            &request(
                "http:listen",
                Valor::Lista(vec![
                    Valor::Numerus(i64::from(port)),
                    Valor::Numerus(
                        i64::try_from(DEFAULT_MAX_BODY_BYTES).expect("default body bound"),
                    ),
                    Valor::Textus("127.0.0.1".into()),
                ]),
            ),
            &context(),
        )
        .expect("listen bind_host");
    let [ProviderContent::Item(Valor::Numerus(handle))] = handle.contents.as_slice() else {
        panic!("bind_host listen must return a handle");
    };
    TcpStream::connect(("127.0.0.1", port)).expect("connect explicit bind_host");
    provider
        .dispatch(&request("http:stop", Valor::Numerus(*handle)), &context())
        .expect("stop bind_host listener");

    let error = provider
        .dispatch(
            &request(
                "http:listen",
                Valor::Lista(vec![
                    Valor::Numerus(i64::from(free_port())),
                    Valor::Numerus(1024),
                    Valor::Textus(String::new()),
                ]),
            ),
            &context(),
        )
        .expect_err("empty bind_host must fail");
    assert_eq!(error.code, "E_INVALID_ARGS");
}
