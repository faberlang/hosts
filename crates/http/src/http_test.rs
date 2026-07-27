use super::*;
use host_kernel::{CancellationProbe, ProviderContent};
use std::collections::BTreeMap;
use std::io::Read;
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

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
