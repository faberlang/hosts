use super::*;
use faber::frame;
use host_kernel::{
    HostError, ManifestCall, Provider, ProviderContent, ProviderManifest, ProviderRegistration,
    ProviderReply,
};
use std::future::Future;
use std::pin::pin;
use std::sync::Barrier;
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

#[allow(clippy::struct_excessive_bools)]
struct TestProvider {
    registration: ProviderRegistration,
    delay: Duration,
    panic: bool,
    wait_for_cancellation: bool,
    return_cancelled_after_cancellation: bool,
    retryable_error: bool,
    fixed_reply: Option<ProviderReply>,
    started: Option<Arc<AtomicBool>>,
    observed_request: Option<Arc<Mutex<Option<RequestFrame>>>>,
}

impl TestProvider {
    fn new(route: &str) -> Self {
        Self {
            registration: ProviderRegistration::new(ProviderManifest {
                manifest_version: 1,
                provider: "test".to_owned(),
                owner: "test".to_owned(),
                prefixes: vec!["test".to_owned()],
                calls: vec![ManifestCall {
                    route: route.to_owned(),
                    summary: "test route".to_owned(),
                    opener: "valor".to_owned(),
                    result: "valor".to_owned(),
                }],
                native_dependencies: Vec::new(),
            }),
            delay: Duration::ZERO,
            panic: false,
            wait_for_cancellation: false,
            return_cancelled_after_cancellation: false,
            retryable_error: false,
            fixed_reply: None,
            started: None,
            observed_request: None,
        }
    }

    fn with_reply(mut self, result: &str, reply: ProviderReply) -> Self {
        self.registration.manifest.calls[0].result = result.to_owned();
        self.fixed_reply = Some(reply);
        self
    }
}

impl Provider for TestProvider {
    fn registration(&self) -> &ProviderRegistration {
        &self.registration
    }

    fn dispatch(
        &self,
        request: &RequestFrame,
        context: &DispatchContext,
    ) -> host_kernel::HostResult<ProviderReply> {
        assert!(!self.panic, "test provider panic");
        if let Some(started) = &self.started {
            started.store(true, Ordering::SeqCst);
        }
        if let Some(observed_request) = &self.observed_request {
            *observed_request
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(request.clone());
        }
        if self.wait_for_cancellation {
            let deadline = Instant::now() + Duration::from_millis(250);
            while !context.cancellation.is_cancelled() && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(1));
            }
        }
        if self.return_cancelled_after_cancellation && context.cancellation.is_cancelled() {
            return Err(HostError::cancelled());
        }
        if self.retryable_error {
            return Err(HostError::try_new("E_TEMPORARY", "retry later", true)
                .expect("valid retryable test error"));
        }
        if !self.delay.is_zero() {
            thread::sleep(self.delay);
        }
        if let Some(reply) = &self.fixed_reply {
            return Ok(reply.clone());
        }
        Ok(ProviderReply::item(request.opener.clone()))
    }
}

fn host(provider: TestProvider, workers: usize, capacity: usize) -> NativeHost {
    let mut kernel = Kernel::new();
    kernel
        .register(Arc::new(provider))
        .expect("register test provider");
    NativeHost::with_limits(kernel, workers, capacity)
}

fn start_fixed_reply(provider: TestProvider) -> (NativeHost, frame::Sermo) {
    let host = host(provider, 1, 1);
    let (sermo, responses, cancellation) = frame::test_response_sender("test:reply");
    host.start(
        SermoRequest {
            conversation_id: sermo.conversation_id(),
            route: "test:reply".to_owned(),
            opener: faber::Valor::Nihil,
            target: None,
        },
        responses,
        cancellation,
    )
    .expect("enqueue fixed reply job");
    (host, sermo)
}

#[test]
fn start_forwards_host_protocol_frame_fields() {
    let observed_request = Arc::new(Mutex::new(None));
    let provider = TestProvider {
        observed_request: Some(Arc::clone(&observed_request)),
        ..TestProvider::new("test:echo")
    };
    let host = host(provider, 1, 1);
    let (mut sermo, responses, cancellation) = frame::test_response_sender("test:echo");

    host.start(
        SermoRequest {
            conversation_id: sermo.conversation_id(),
            route: "test:echo".to_owned(),
            opener: faber::Valor::Textus("payload".to_owned()),
            target: Some("target-handle"),
        },
        responses,
        cancellation,
    )
    .expect("enqueue frame forwarding job");

    assert_eq!(
        frame::sermo_recv(&mut sermo).expect("item").status,
        FrameStatus::Item
    );
    let request = observed_request
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
        .expect("provider should observe request frame");
    assert_eq!(request.conversation_id, sermo.conversation_id());
    assert_eq!(request.route, "test:echo");
    assert_eq!(request.opener, faber::Valor::Textus("payload".to_owned()));
    assert_eq!(request.target.as_deref(), Some("target-handle"));

    host.shutdown();
}

#[test]
fn start_only_enqueues_and_worker_replies() {
    let host = host(TestProvider::new("test:echo"), 1, 1);
    let (mut sermo, responses, cancellation) = frame::test_response_sender("test:echo");
    let began = Instant::now();
    host.start(
        SermoRequest {
            conversation_id: sermo.conversation_id(),
            route: "test:echo".to_owned(),
            opener: faber::Valor::Textus("ok".to_owned()),
            target: None,
        },
        responses,
        cancellation,
    )
    .expect("enqueue");
    assert!(began.elapsed() < Duration::from_millis(20));
    assert_eq!(
        frame::sermo_recv(&mut sermo).expect("item").status,
        FrameStatus::Item
    );
    assert_eq!(
        frame::sermo_recv(&mut sermo).expect("done").status,
        FrameStatus::Done
    );
}

#[test]
fn provider_vacuum_reply_emits_only_terminal_done() {
    let (host, mut sermo) = start_fixed_reply(
        TestProvider::new("test:reply").with_reply("vacuum", ProviderReply::vacuum()),
    );

    let done = frame::sermo_recv(&mut sermo).expect("vacuum terminal");
    assert_eq!(done.status, FrameStatus::Done);
    assert_eq!(done.data, faber::Valor::Nihil);
    assert!(
        frame::sermo_recv(&mut sermo).is_none(),
        "vacuum reply must not emit content frames before terminal done"
    );
    host.shutdown();
}

#[test]
fn provider_byte_reply_maps_to_byte_frame_with_octeti_payload() {
    let bytes = vec![1, 2, 3, 5, 8];
    let (host, mut sermo) = start_fixed_reply(
        TestProvider::new("test:reply").with_reply("bytes", ProviderReply::byte(bytes.clone())),
    );

    let byte = frame::sermo_recv(&mut sermo).expect("byte content");
    assert_eq!(byte.status, FrameStatus::Byte);
    assert_eq!(byte.data, faber::Valor::Octeti(bytes));
    assert_eq!(
        frame::sermo_recv(&mut sermo).expect("byte terminal").status,
        FrameStatus::Done
    );
    host.shutdown();
}

#[test]
fn provider_bulk_reply_maps_to_bulk_frame_with_expected_payload() {
    let payload = faber::Valor::Tabula(BTreeMap::from([(
        "rows".to_owned(),
        faber::Valor::Numerus(2),
    )]));
    let (host, mut sermo) = start_fixed_reply(TestProvider::new("test:reply").with_reply(
        "bulk-valor",
        ProviderReply {
            contents: vec![ProviderContent::Bulk(payload.clone())],
        },
    ));

    let bulk = frame::sermo_recv(&mut sermo).expect("bulk content");
    assert_eq!(bulk.status, FrameStatus::Bulk);
    assert_eq!(bulk.data, payload);
    assert_eq!(
        frame::sermo_recv(&mut sermo).expect("bulk terminal").status,
        FrameStatus::Done
    );
    host.shutdown();
}

#[test]
fn provider_multi_item_reply_preserves_order_and_terminal_done_follows() {
    let (host, mut sermo) = start_fixed_reply(TestProvider::new("test:reply").with_reply(
        "lista-valor",
        ProviderReply {
            contents: vec![
                ProviderContent::Item(faber::Valor::Textus("first".to_owned())),
                ProviderContent::Item(faber::Valor::Numerus(2)),
                ProviderContent::Item(faber::Valor::Bivalens(true)),
            ],
        },
    ));

    let first = frame::sermo_recv(&mut sermo).expect("first item");
    assert_eq!(first.status, FrameStatus::Item);
    assert_eq!(first.data, faber::Valor::Textus("first".to_owned()));

    let second = frame::sermo_recv(&mut sermo).expect("second item");
    assert_eq!(second.status, FrameStatus::Item);
    assert_eq!(second.data, faber::Valor::Numerus(2));

    let third = frame::sermo_recv(&mut sermo).expect("third item");
    assert_eq!(third.status, FrameStatus::Item);
    assert_eq!(third.data, faber::Valor::Bivalens(true));

    let done = frame::sermo_recv(&mut sermo).expect("multi-item terminal");
    assert_eq!(done.status, FrameStatus::Done);
    assert_eq!(done.data, faber::Valor::Nihil);
    assert!(frame::sermo_recv(&mut sermo).is_none());
    host.shutdown();
}

#[test]
fn saturation_is_immediate_and_shutdown_rejects() {
    let provider = TestProvider {
        delay: Duration::from_millis(100),
        ..TestProvider::new("test:echo")
    };
    let host = host(provider, 1, 1);
    let mut held = Vec::new();
    for _ in 0..3 {
        let (sermo, responses, cancellation) = frame::test_response_sender("test:echo");
        let result = host.start(
            SermoRequest {
                conversation_id: sermo.conversation_id(),
                route: "test:echo".to_owned(),
                opener: faber::Valor::Nihil,
                target: None,
            },
            responses,
            cancellation,
        );
        held.push(result);
    }
    assert!(held.iter().any(|result| {
        result
            .as_ref()
            .is_err_and(|error| error.issue == "host_queue_saturated")
    }));
    host.shutdown();
    let (_sermo, responses, cancellation) = frame::test_response_sender("test:echo");
    let error = host
        .start(
            SermoRequest {
                conversation_id: "after".to_owned(),
                route: "test:echo".to_owned(),
                opener: faber::Valor::Nihil,
                target: None,
            },
            responses,
            cancellation,
        )
        .expect_err("shutdown must reject");
    assert_eq!(error.issue, "host_shutting_down");
}

#[test]
fn unsupported_routes_are_rejected_before_enqueue() {
    let started = Arc::new(AtomicBool::new(false));
    let observed_request = Arc::new(Mutex::new(None));
    let provider = TestProvider {
        started: Some(Arc::clone(&started)),
        observed_request: Some(Arc::clone(&observed_request)),
        ..TestProvider::new("test:echo")
    };
    let host = host(provider, 1, 1);
    let (mut sermo, responses, cancellation) = frame::test_response_sender("test:missing");
    let retained_responses = responses.clone();
    let error = host
        .start(
            SermoRequest {
                conversation_id: "unsupported".to_owned(),
                route: "test:missing".to_owned(),
                opener: faber::Valor::Nihil,
                target: None,
            },
            responses,
            cancellation,
        )
        .expect_err("unsupported routes must fail before enqueue");

    assert_eq!(error.issue, "host_unsupported_route");
    assert_eq!(host.state.next_job_id.load(Ordering::SeqCst), 0);
    assert!(
        host.state
            .active_jobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_empty(),
        "unsupported route must not register an active job"
    );
    assert!(
        !started.load(Ordering::SeqCst),
        "unsupported route must not dispatch to the provider"
    );
    assert!(
        observed_request
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_none(),
        "unsupported route must not construct a provider request"
    );

    {
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        let mut receive = pin!(frame::sermo_recv_async(&mut sermo));
        assert!(
            matches!(receive.as_mut().poll(&mut context), Poll::Pending),
            "unsupported route rejection must not emit content or terminal frames"
        );
    }
    drop(retained_responses);
    host.shutdown();
}

#[test]
fn enqueue_rechecks_shutdown_state_while_sender_is_installed() {
    let host = host(TestProvider::new("test:echo"), 1, 1);
    host.state.shutting_down.store(true, Ordering::SeqCst);
    let (sermo, responses, cancellation) = frame::test_response_sender("test:echo");
    let error = enqueue_job(
        &host.state,
        NativeJob {
            id: 0,
            request: SermoRequest {
                conversation_id: sermo.conversation_id(),
                route: "test:echo".to_owned(),
                opener: faber::Valor::Nihil,
                target: None,
            },
            responses,
            cancellation,
        },
    )
    .expect_err("shutdown must reject before enqueue even when queue sender remains");

    assert_eq!(error.issue, "host_shutting_down");
    host.state.shutting_down.store(false, Ordering::SeqCst);
    host.shutdown();
}

#[test]
fn concurrent_enqueue_admits_exactly_bounded_capacity() {
    const PRODUCERS: usize = 12;
    const CAPACITY: usize = 5;

    let (queue, receiver) = sync_channel(CAPACITY);
    let state = Arc::new(NativeState {
        kernel: Arc::new(Kernel::new()),
        queue: Mutex::new(Some(queue)),
        shutting_down: AtomicBool::new(false),
        public_handles: AtomicUsize::new(1),
        next_job_id: AtomicU64::new(0),
        active_jobs: Mutex::new(BTreeMap::new()),
        workers: Mutex::new(Vec::new()),
    });
    let barrier = Arc::new(Barrier::new(PRODUCERS));
    let mut producers = Vec::new();

    for index in 0..PRODUCERS {
        let state = Arc::clone(&state);
        let barrier = Arc::clone(&barrier);
        producers.push(thread::spawn(move || {
            let (_sermo, responses, cancellation) = frame::test_response_sender("test:echo");
            barrier.wait();
            let result = enqueue_job(
                &state,
                NativeJob {
                    id: index as u64,
                    request: SermoRequest {
                        conversation_id: format!("enqueue-{index}"),
                        route: "test:echo".to_owned(),
                        opener: faber::Valor::Nihil,
                        target: None,
                    },
                    responses,
                    cancellation,
                },
            );
            (index as u64, result)
        }));
    }

    let mut accepted = Vec::new();
    let mut saturated = 0;
    for producer in producers {
        let (job_id, result) = producer.join().expect("producer should not panic");
        match result {
            Ok(()) => accepted.push(job_id),
            Err(error) if error.issue == "host_queue_saturated" => saturated += 1,
            Err(error) => panic!("unexpected enqueue error: {}", error.issue),
        }
    }

    assert_eq!(accepted.len(), CAPACITY);
    assert_eq!(saturated, PRODUCERS - CAPACITY);

    let mut queued = Vec::new();
    while let Ok(job) = receiver.try_recv() {
        queued.push(job.id);
    }
    accepted.sort_unstable();
    queued.sort_unstable();
    assert_eq!(
        queued, accepted,
        "every successful producer must correspond to one queued job"
    );
}

#[test]
fn construction_rejects_zero_workers() {
    let Err(error) = NativeHost::try_with_limits(Kernel::new(), 0, 1) else {
        panic!("zero workers must be rejected");
    };

    assert_eq!(error.issue, "native_host_zero_workers");
    assert_eq!(
        error.to_string(),
        "native host requires at least one worker"
    );
}

#[test]
fn construction_spawn_failure_shuts_down_and_joins_partial_workers() {
    let exits = Arc::new(AtomicUsize::new(0));
    let worker_exits = Arc::clone(&exits);
    let mut attempts = 0;

    let result = build_with_worker_spawner(Kernel::new(), 4, 1, move |name, state, receiver| {
        if attempts == 2 {
            return Err(std::io::Error::other("injected worker spawn failure"));
        }
        attempts += 1;
        let worker_exits = Arc::clone(&worker_exits);
        thread::Builder::new().name(name).spawn(move || {
            run_worker(&state, &receiver);
            worker_exits.fetch_add(1, Ordering::SeqCst);
        })
    });

    let Err(error) = result else {
        panic!("injected spawn failure should abort construction");
    };
    assert_eq!(error.issue, "native_host_worker_spawn_failed");
    assert!(
        error.to_string().contains("injected worker spawn failure"),
        "unexpected error: {error}"
    );
    assert_eq!(
        exits.load(Ordering::SeqCst),
        2,
        "partial worker cleanup must close the queue and join already spawned workers"
    );
}

#[test]
fn pre_cancelled_job_emits_cancel_frame() {
    let host = host(TestProvider::new("test:echo"), 1, 1);
    let (mut sermo, responses, cancellation) = frame::test_response_sender("test:echo");
    cancellation.cancel();
    host.start(
        SermoRequest {
            conversation_id: sermo.conversation_id(),
            route: "test:echo".to_owned(),
            opener: faber::Valor::Nihil,
            target: None,
        },
        responses,
        cancellation,
    )
    .expect("enqueue cancelled job");
    assert_eq!(
        frame::sermo_recv(&mut sermo).expect("cancel").status,
        FrameStatus::Cancel
    );
}

#[test]
fn provider_panic_emits_error_frame_with_code_and_retryable_false() {
    let host = host(
        TestProvider {
            panic: true,
            ..TestProvider::new("test:panic")
        },
        1,
        1,
    );
    let (mut sermo, responses, cancellation) = frame::test_response_sender("test:panic");
    host.start(
        SermoRequest {
            conversation_id: sermo.conversation_id(),
            route: "test:panic".to_owned(),
            opener: faber::Valor::Nihil,
            target: None,
        },
        responses,
        cancellation,
    )
    .expect("enqueue panic job");
    let terminal = frame::sermo_recv(&mut sermo).expect("error");
    assert_eq!(terminal.status, FrameStatus::Error);
    let faber::Valor::Tabula(fields) = terminal.data else {
        panic!("provider panic error payload must be a tabula");
    };
    assert_eq!(
        fields.get("code"),
        Some(&faber::Valor::Textus("E_PROVIDER_PANIC".to_owned()))
    );
    assert_eq!(
        fields.get("retryable"),
        Some(&faber::Valor::Bivalens(false))
    );
    let Some(faber::Valor::Textus(message)) = fields.get("message") else {
        panic!("provider panic error payload must include a text message");
    };
    assert!(
        !message.is_empty(),
        "provider panic error message must be nonempty"
    );
}

#[test]
fn provider_cancelled_error_emits_cancel_terminal_frame() {
    let started = Arc::new(AtomicBool::new(false));
    let provider = TestProvider {
        wait_for_cancellation: true,
        return_cancelled_after_cancellation: true,
        started: Some(Arc::clone(&started)),
        ..TestProvider::new("test:cancel")
    };
    let host = host(provider, 1, 1);
    let (mut sermo, responses, cancellation) = frame::test_response_sender("test:cancel");
    host.start(
        SermoRequest {
            conversation_id: sermo.conversation_id(),
            route: "test:cancel".to_owned(),
            opener: faber::Valor::Nihil,
            target: None,
        },
        responses,
        cancellation.clone(),
    )
    .expect("enqueue cancellation-observing job");

    wait_for_provider_start(&started);
    cancellation.cancel();

    assert_eq!(
        frame::sermo_recv(&mut sermo)
            .expect("provider cancellation terminal")
            .status,
        FrameStatus::Cancel
    );
}

#[test]
fn provider_retryable_error_preserves_retryable_field() {
    let provider = TestProvider {
        retryable_error: true,
        ..TestProvider::new("test:retry")
    };
    let host = host(provider, 1, 1);
    let (mut sermo, responses, cancellation) = frame::test_response_sender("test:retry");
    host.start(
        SermoRequest {
            conversation_id: sermo.conversation_id(),
            route: "test:retry".to_owned(),
            opener: faber::Valor::Nihil,
            target: None,
        },
        responses,
        cancellation,
    )
    .expect("enqueue retryable-error job");

    let terminal = frame::sermo_recv(&mut sermo).expect("retryable error terminal");
    assert_eq!(terminal.status, FrameStatus::Error);
    let faber::Valor::Tabula(fields) = terminal.data else {
        panic!("error payload must be a tabula");
    };
    assert_eq!(fields.get("retryable"), Some(&faber::Valor::Bivalens(true)));
}

fn wait_for_provider_start(started: &Arc<AtomicBool>) {
    let deadline = Instant::now() + Duration::from_secs(1);
    while !started.load(Ordering::SeqCst) && Instant::now() < deadline {
        thread::yield_now();
    }
    assert!(started.load(Ordering::SeqCst), "provider did not start");
}

#[test]
fn shutdown_cancels_active_provider_before_joining_workers() {
    let started = Arc::new(AtomicBool::new(false));
    let provider = TestProvider {
        wait_for_cancellation: true,
        started: Some(Arc::clone(&started)),
        ..TestProvider::new("test:echo")
    };
    let host = host(provider, 1, 1);
    let (mut sermo, responses, cancellation) = frame::test_response_sender("test:echo");
    host.start(
        SermoRequest {
            conversation_id: sermo.conversation_id(),
            route: "test:echo".to_owned(),
            opener: faber::Valor::Nihil,
            target: None,
        },
        responses,
        cancellation,
    )
    .expect("enqueue active job");
    wait_for_provider_start(&started);

    let began = Instant::now();
    host.shutdown();
    assert!(
        began.elapsed() < Duration::from_millis(100),
        "shutdown should cancel an active provider before its fallback deadline"
    );
    assert_eq!(
        frame::sermo_recv(&mut sermo)
            .expect("shutdown terminal")
            .status,
        FrameStatus::Cancel
    );
}

#[test]
fn shutdown_resolves_queued_and_active_terminal_obligations() {
    let started = Arc::new(AtomicBool::new(false));
    let provider = TestProvider {
        delay: Duration::from_millis(40),
        started: Some(Arc::clone(&started)),
        ..TestProvider::new("test:echo")
    };
    let host = host(provider, 1, 2);
    let (mut active_sermo, active_responses, active_cancellation) =
        frame::test_response_sender("test:echo");
    host.start(
        SermoRequest {
            conversation_id: active_sermo.conversation_id(),
            route: "test:echo".to_owned(),
            opener: faber::Valor::Nihil,
            target: None,
        },
        active_responses,
        active_cancellation,
    )
    .expect("enqueue active job");
    wait_for_provider_start(&started);

    let (mut queued_sermo, queued_responses, queued_cancellation) =
        frame::test_response_sender("test:echo");
    host.start(
        SermoRequest {
            conversation_id: queued_sermo.conversation_id(),
            route: "test:echo".to_owned(),
            opener: faber::Valor::Nihil,
            target: None,
        },
        queued_responses,
        queued_cancellation,
    )
    .expect("enqueue queued job");

    host.shutdown();
    assert_eq!(
        frame::sermo_recv(&mut active_sermo)
            .expect("active terminal")
            .status,
        FrameStatus::Cancel
    );
    assert_eq!(
        frame::sermo_recv(&mut queued_sermo)
            .expect("queued terminal")
            .status,
        FrameStatus::Error
    );
}

#[test]
fn dropping_last_public_handle_begins_shutdown_without_joining() {
    let started = Arc::new(AtomicBool::new(false));
    let provider = TestProvider {
        delay: Duration::from_millis(40),
        started: Some(Arc::clone(&started)),
        ..TestProvider::new("test:echo")
    };
    let host = host(provider, 1, 1);
    let (mut sermo, responses, cancellation) = frame::test_response_sender("test:echo");
    host.start(
        SermoRequest {
            conversation_id: sermo.conversation_id(),
            route: "test:echo".to_owned(),
            opener: faber::Valor::Nihil,
            target: None,
        },
        responses,
        cancellation,
    )
    .expect("enqueue active job");
    wait_for_provider_start(&started);

    let began = Instant::now();
    drop(host);
    assert!(
        began.elapsed() < Duration::from_millis(20),
        "drop must not join active provider work"
    );
    assert_eq!(
        frame::sermo_recv(&mut sermo).expect("drop terminal").status,
        FrameStatus::Cancel
    );
}

#[test]
fn dropping_last_public_handle_gives_active_cancel_and_queued_error() {
    let started = Arc::new(AtomicBool::new(false));
    let provider = TestProvider {
        delay: Duration::from_millis(40),
        started: Some(Arc::clone(&started)),
        ..TestProvider::new("test:echo")
    };
    let host = host(provider, 1, 2);
    let (mut active_sermo, active_responses, active_cancellation) =
        frame::test_response_sender("test:echo");
    host.start(
        SermoRequest {
            conversation_id: active_sermo.conversation_id(),
            route: "test:echo".to_owned(),
            opener: faber::Valor::Nihil,
            target: None,
        },
        active_responses,
        active_cancellation,
    )
    .expect("enqueue active job");
    wait_for_provider_start(&started);

    let (mut queued_sermo, queued_responses, queued_cancellation) =
        frame::test_response_sender("test:echo");
    host.start(
        SermoRequest {
            conversation_id: queued_sermo.conversation_id(),
            route: "test:echo".to_owned(),
            opener: faber::Valor::Nihil,
            target: None,
        },
        queued_responses,
        queued_cancellation,
    )
    .expect("enqueue queued job");

    let began = Instant::now();
    drop(host);
    assert!(
        began.elapsed() < Duration::from_millis(20),
        "dropping the last public handle must not join active provider work"
    );
    assert_eq!(
        frame::sermo_recv(&mut active_sermo)
            .expect("drop active terminal")
            .status,
        FrameStatus::Cancel
    );
    assert_eq!(
        frame::sermo_recv(&mut queued_sermo)
            .expect("drop queued terminal")
            .status,
        FrameStatus::Error
    );
}

#[test]
fn dropping_one_public_clone_keeps_host_alive() {
    let host = host(TestProvider::new("test:echo"), 1, 1);
    let retained = host.clone();
    drop(host);
    let (mut sermo, responses, cancellation) = frame::test_response_sender("test:echo");
    retained
        .start(
            SermoRequest {
                conversation_id: sermo.conversation_id(),
                route: "test:echo".to_owned(),
                opener: faber::Valor::Textus("alive".to_owned()),
                target: None,
            },
            responses,
            cancellation,
        )
        .expect("retained clone remains usable");
    assert_eq!(
        frame::sermo_recv(&mut sermo).expect("item").status,
        FrameStatus::Item
    );
    assert_eq!(
        frame::sermo_recv(&mut sermo).expect("done").status,
        FrameStatus::Done
    );
    retained.shutdown();
}
