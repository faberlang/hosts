use super::*;
use host_kernel::ProviderContent;
use std::sync::atomic::{AtomicBool, Ordering};

#[test]
fn manifest_registers_all_process_routes() {
    let mut kernel = Kernel::new();
    register(&mut kernel).expect("register processus");
    let calls = &kernel.manifest().providers[0].calls;
    assert_eq!(calls.len(), 10);
    assert!(calls.iter().any(|call| call.route == "processus:scribe"));
    assert!(
        calls.iter().all(|call| call.route != "processus:exi"),
        "processus:exi must stay unmanifested until host exit has a protocol-visible terminal response"
    );
}

#[test]
fn unsafe_unmanifested_routes_are_not_dispatchable_through_provider() {
    let provider = Processus::new().expect("provider");
    let route = "processus:exi";
    let error = provider
        .dispatch(
            &RequestFrame {
                conversation_id: route.into(),
                route: route.into(),
                opener: Valor::Numerus(7),
                target: None,
            },
            &DispatchContext {
                cancellation: host_kernel::CancellationProbe::new(|| true),
            },
        )
        .expect_err("unsafe route must be rejected as an ordinary host error");

    assert_eq!(error.code, "E_NO_ROUTE");
}

#[test]
fn environment_mutation_round_trip_uses_textus_carriers() {
    let provider = Processus::new().expect("provider");
    let name = format!("FABER_PROCESSUS_TEST_{}", std::process::id());
    let context = DispatchContext {
        cancellation: host_kernel::CancellationProbe::new(|| false),
    };
    provider
        .dispatch(
            &RequestFrame {
                conversation_id: "scribe".into(),
                route: "processus:scribe".into(),
                opener: Valor::Lista(vec![
                    Valor::Textus(name.clone()),
                    Valor::Textus("salve".into()),
                ]),
                target: None,
            },
            &context,
        )
        .expect("environment write");
    let reply = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "lege".into(),
                route: "processus:lege".into(),
                opener: Valor::Textus(name.clone()),
                target: None,
            },
            &context,
        )
        .expect("environment read");
    assert!(matches!(
        reply.contents.as_slice(),
        [ProviderContent::Item(Valor::Textus(value))] if value == "salve"
    ));
    std::env::remove_var(name);
}

#[test]
fn capture_returns_structured_status_stdout_and_stderr() {
    let provider = Processus::new().expect("provider");
    let reply = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "capture".into(),
                route: "processus:captura".into(),
                opener: Valor::Lista(vec![
                    Valor::Textus("sh".into()),
                    Valor::Textus("-c".into()),
                    Valor::Textus("printf out; printf err >&2; exit 7".into()),
                ]),
                target: None,
            },
            &DispatchContext {
                cancellation: host_kernel::CancellationProbe::new(|| false),
            },
        )
        .expect("capture");
    let [ProviderContent::Item(Valor::Tabula(fields))] = reply.contents.as_slice() else {
        panic!("capture must return one tabula item");
    };
    assert_eq!(fields.get("status"), Some(&Valor::Numerus(7)));
    assert_eq!(fields.get("stdout"), Some(&Valor::Textus("out".into())));
    assert_eq!(fields.get("stderr"), Some(&Valor::Textus("err".into())));
}

#[test]
fn shell_route_returns_stdout_item() {
    let provider = Processus::new().expect("provider");
    let reply = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "shell".into(),
                route: "processus:exsequi".into(),
                opener: Valor::Textus("printf salve".into()),
                target: None,
            },
            &DispatchContext {
                cancellation: host_kernel::CancellationProbe::new(|| false),
            },
        )
        .expect("shell");
    assert!(
        matches!(reply.contents.as_slice(), [ProviderContent::Item(Valor::Textus(text))] if text == "salve")
    );
}

fn dispatch_until_cancelled(provider: &Processus, route: &str, opener: Valor) -> HostError {
    let cancelled = Arc::new(AtomicBool::new(false));
    let trigger = Arc::clone(&cancelled);
    let timer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(25));
        trigger.store(true, Ordering::SeqCst);
    });
    let result = provider.dispatch(
        &RequestFrame {
            conversation_id: route.into(),
            route: route.into(),
            opener,
            target: None,
        },
        &DispatchContext {
            cancellation: host_kernel::CancellationProbe::from_flag(cancelled),
        },
    );
    timer.join().expect("cancellation timer");
    result.expect_err("running process must be cancelled")
}

#[test]
fn cancellation_terminates_shell_and_capture_children() {
    let provider = Processus::new().expect("provider");
    let started = std::time::Instant::now();
    let shell_error = dispatch_until_cancelled(
        &provider,
        "processus:exsequi",
        Valor::Textus("while :; do :; done".into()),
    );
    assert_eq!(shell_error.code, "E_CANCELLED");
    let capture_error = dispatch_until_cancelled(
        &provider,
        "processus:captura",
        Valor::Lista(vec![
            Valor::Textus("sh".into()),
            Valor::Textus("-c".into()),
            Valor::Textus("while :; do :; done".into()),
        ]),
    );
    assert_eq!(capture_error.code, "E_CANCELLED");
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "cancelled child operations must not block indefinitely"
    );
}

#[cfg(unix)]
#[test]
fn cancellation_terminates_shell_descendants() {
    let provider = Processus::new().expect("provider");
    let started = std::time::Instant::now();
    let error = dispatch_until_cancelled(
        &provider,
        "processus:exsequi",
        Valor::Textus("sleep 30".into()),
    );
    assert_eq!(error.code, "E_CANCELLED");
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "cancellation must terminate shell descendants with the owned process"
    );
}

#[test]
fn exsequi_empty_command_returns_textus_empty_string() {
    let provider = Processus::new().expect("provider");
    let reply = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "empty".into(),
                route: "processus:exsequi".into(),
                opener: Valor::Textus(String::new()),
                target: None,
            },
            &DispatchContext {
                cancellation: host_kernel::CancellationProbe::new(|| false),
            },
        )
        .expect("empty shell command");
    assert!(matches!(
        reply.contents.as_slice(),
        [ProviderContent::Item(Valor::Textus(text))] if text.is_empty()
    ));
}

#[test]
fn captura_empty_args_list_rejected() {
    let provider = Processus::new().expect("provider");
    let error = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "capture-empty".into(),
                route: "processus:captura".into(),
                opener: Valor::Lista(Vec::new()),
                target: None,
            },
            &DispatchContext {
                cancellation: host_kernel::CancellationProbe::new(|| false),
            },
        )
        .expect_err("empty captura args must fail");
    assert_eq!(error.code, "E_INVALID_ARGS");
}

#[test]
fn scribe_rejects_empty_env_name() {
    let provider = Processus::new().expect("provider");
    let error = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "scribe-empty".into(),
                route: "processus:scribe".into(),
                opener: Valor::Lista(vec![
                    Valor::Textus(String::new()),
                    Valor::Textus("val".into()),
                ]),
                target: None,
            },
            &DispatchContext {
                cancellation: host_kernel::CancellationProbe::new(|| false),
            },
        )
        .expect_err("empty env name must fail");
    assert_eq!(error.code, "E_INVALID_ARGS");
}
