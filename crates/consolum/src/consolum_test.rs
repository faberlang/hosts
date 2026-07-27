use super::*;
use host_kernel::ProviderContent;
use std::io::{Cursor, Read};

fn active_context() -> DispatchContext {
    DispatchContext {
        cancellation: host_kernel::CancellationProbe::new(|| false),
    }
}

fn dispatch_line(reader: impl Read + Send + 'static) -> ProviderReply {
    let provider = Consolum::with_line_reader_for_tests(reader).expect("provider");
    provider
        .dispatch(
            &RequestFrame {
                conversation_id: "read-line".into(),
                route: "consolum:lege".into(),
                opener: Valor::Nihil,
                target: None,
            },
            &active_context(),
        )
        .expect("read line")
}

#[test]
fn manifest_omits_fundet_alias_and_registers_canonical_routes() {
    let mut kernel = Kernel::new();
    register(&mut kernel).expect("register consolum");
    let calls = &kernel.manifest().providers[0].calls;
    assert_eq!(calls.len(), 16);
    assert!(calls.iter().any(|call| call.route == "consolum:funde"));
    assert!(!calls.iter().any(|call| call.route == "consolum:fundet"));
}

#[test]
fn terminal_predicate_returns_one_boolean_item() {
    let provider = Consolum::new().expect("provider");
    let reply = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "audit".into(),
                route: "consolum:audit".into(),
                opener: Valor::Nihil,
                target: None,
            },
            &DispatchContext {
                cancellation: host_kernel::CancellationProbe::new(|| false),
            },
        )
        .expect("audit");
    assert!(matches!(
        reply.contents.as_slice(),
        [ProviderContent::Item(Valor::Bivalens(_))]
    ));
}

#[test]
fn byte_and_string_arguments_decode_from_ordered_openers() {
    assert_eq!(
        bytes_arg(&Valor::Octeti(vec![1, 2]), 0, "data").unwrap(),
        vec![1, 2]
    );
    assert_eq!(
        string_arg(&Valor::Lista(vec![Valor::Textus("ok".into())]), 0, "msg").unwrap(),
        "ok"
    );
    assert!(i64_arg(&Valor::Textus("bad".into()), 0, "n").is_err());
}

#[test]
#[allow(clippy::cast_possible_wrap)]
fn hauri_rejects_over_limit_before_allocation_and_keeps_zero_policy() {
    let provider = Consolum::new().expect("provider");

    let zero = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "zero".into(),
                route: "consolum:hauri".into(),
                opener: Valor::Numerus(0),
                target: None,
            },
            &active_context(),
        )
        .expect("zero-byte stdin read");
    assert!(matches!(zero.contents.as_slice(), [ProviderContent::Byte(value)] if value.is_empty()));

    let negative = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "negative".into(),
                route: "consolum:hauri".into(),
                opener: Valor::Numerus(-1),
                target: None,
            },
            &active_context(),
        )
        .expect("negative stdin read clamps to zero");
    assert!(
        matches!(negative.contents.as_slice(), [ProviderContent::Byte(value)] if value.is_empty())
    );

    let error = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "too-many".into(),
                route: "consolum:hauri".into(),
                opener: Valor::Numerus(MAX_STDIN_READ_BYTES as i64 + 1),
                target: None,
            },
            &active_context(),
        )
        .expect_err("over-limit stdin read must fail before allocation");
    assert_eq!(error.code, "E_INVALID_ARGS");
    assert!(error.message.contains("consolum:hauri"));
    assert!(error.message.contains(&MAX_STDIN_READ_BYTES.to_string()));
}

#[test]
fn read_line_returns_text_before_newline() {
    let reply = dispatch_line(Cursor::new(b"salve\nreliquum".to_vec()));

    assert!(matches!(
        reply.contents.as_slice(),
        [ProviderContent::Item(Valor::Textus(line))] if line == "salve"
    ));
}

#[test]
fn read_line_returns_eof_terminated_text() {
    let reply = dispatch_line(Cursor::new(b"salve".to_vec()));

    assert!(matches!(
        reply.contents.as_slice(),
        [ProviderContent::Item(Valor::Textus(line))] if line == "salve"
    ));
}

#[test]
fn read_line_returns_empty_text_at_empty_eof() {
    let reply = dispatch_line(Cursor::new(Vec::new()));

    assert!(matches!(
        reply.contents.as_slice(),
        [ProviderContent::Item(Valor::Textus(line))] if line.is_empty()
    ));
}

#[cfg(unix)]
#[test]
fn fd_wait_honors_cancellation() {
    use std::os::fd::AsRawFd;
    use std::os::unix::net::UnixStream;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;
    use std::time::{Duration, Instant};

    let (reader, _writer) = UnixStream::pair().expect("socket pair");
    let cancelled = Arc::new(AtomicBool::new(false));
    let trigger = Arc::clone(&cancelled);
    let fd = reader.as_raw_fd();
    let started = Instant::now();
    let waiter = thread::spawn(move || {
        let _reader = reader;
        wait_for_fd(
            fd,
            libc::POLLIN as libc::c_short,
            &DispatchContext {
                cancellation: host_kernel::CancellationProbe::from_flag(cancelled),
            },
            "test:read",
        )
    });
    thread::sleep(Duration::from_millis(25));
    trigger.store(true, Ordering::SeqCst);
    let error = waiter
        .join()
        .expect("waiter thread")
        .expect_err("blocked fd wait must cancel");
    assert_eq!(error.code, "E_CANCELLED");
    assert!(started.elapsed() < Duration::from_secs(1));
}

#[cfg(unix)]
#[test]
fn nonblocking_write_honors_cancellation_and_restores_flags() {
    use std::os::fd::AsRawFd;
    use std::os::unix::net::UnixStream;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;
    use std::time::{Duration, Instant};

    let (writer, _reader) = UnixStream::pair().expect("socket pair");
    let fd = writer.as_raw_fd();
    let original_flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    assert!(original_flags >= 0, "read original fd flags");
    let cancelled = Arc::new(AtomicBool::new(false));
    let trigger = Arc::clone(&cancelled);
    let timer = thread::spawn(move || {
        thread::sleep(Duration::from_millis(25));
        trigger.store(true, Ordering::SeqCst);
    });
    let started = Instant::now();
    let error = write_fd_cancellable(
        fd,
        &vec![b'x'; 8 * 1024 * 1024],
        &DispatchContext {
            cancellation: host_kernel::CancellationProbe::from_flag(cancelled),
        },
        "test:write",
    )
    .expect_err("blocked fd write must cancel");
    timer.join().expect("cancellation timer");
    assert_eq!(error.code, "E_CANCELLED");
    assert!(started.elapsed() < Duration::from_secs(1));
    let restored_flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    assert_eq!(
        restored_flags & libc::O_NONBLOCK,
        original_flags & libc::O_NONBLOCK
    );
}

#[test]
fn scribe_with_empty_string_emits_empty_frame_content() {
    let provider = Consolum::new().expect("provider");
    let reply = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "scribe-empty".into(),
                route: "consolum:scribe".into(),
                opener: Valor::Textus(String::new()),
                target: None,
            },
            &active_context(),
        )
        .expect("scribe empty");
    assert!(reply.contents.is_empty());
}

#[test]
fn string_arg_rejects_non_string_positional() {
    assert!(
        string_arg(&Valor::Numerus(42), 0, "msg").is_err(),
        "numerus must be rejected as string arg"
    );
}

#[test]
#[allow(clippy::cast_possible_wrap)]
fn hauri_rejects_non_numeric_magnitudo() {
    let provider = Consolum::new().expect("provider");
    let error = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "hauri-string".into(),
                route: "consolum:hauri".into(),
                opener: Valor::Textus("many".into()),
                target: None,
            },
            &active_context(),
        )
        .expect_err("string magnitudo must fail");
    assert_eq!(error.code, "E_INVALID_ARGS");
}
