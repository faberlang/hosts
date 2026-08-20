use super::*;
use host_kernel::ProviderContent;

#[test]
fn manifest_omits_legacy_expectet_alias() {
    let mut kernel = Kernel::new();
    register(&mut kernel).expect("register tempus");
    let calls = &kernel.manifest().providers[0].calls;
    assert_eq!(calls.len(), 4);
    assert!(!calls.iter().any(|call| call.route == "tempus:expectet"));
}

#[test]
fn sleep_returns_vacuum_and_honors_cancellation() {
    let provider = Tempus::new().expect("provider");
    let request = RequestFrame {
        conversation_id: "sleep".into(),
        route: "tempus:dormiet".into(),
        opener: Valor::Numerus(0),
        target: None,
    };
    let ok_context = DispatchContext {
        cancellation: host_kernel::CancellationProbe::new(|| false),
    };
    let reply = provider.dispatch(&request, &ok_context).expect("sleep");
    assert!(reply.contents.is_empty());

    let cancelled_context = DispatchContext {
        cancellation: host_kernel::CancellationProbe::new(|| true),
    };
    let error = provider
        .dispatch(&request, &cancelled_context)
        .expect_err("cancelled sleep");
    assert_eq!(error.code, "E_CANCELLED");
}

#[test]
fn clock_routes_return_scalar_items() {
    let provider = Tempus::new().expect("provider");
    let context = DispatchContext {
        cancellation: host_kernel::CancellationProbe::new(|| false),
    };
    let now = provider
        .dispatch(
            &RequestFrame {
                conversation_id: "tempus:nunc".into(),
                route: "tempus:nunc".into(),
                opener: Valor::Nihil,
                target: None,
            },
            &context,
        )
        .expect("wall clock route");
    assert!(matches!(
        now.contents.as_slice(),
        [ProviderContent::Item(Valor::Instans(_))]
    ));
    assert!(faber::Instans::try_from_valor(
        match &now.contents[0] {
            ProviderContent::Item(value) => value,
            _ => unreachable!("wall clock must return an item"),
        },
        InstansPraecisio::Nanosecunda,
    )
    .is_some());

    for route in ["tempus:monotonicum", "tempus:activum"] {
        let reply = provider
            .dispatch(
                &RequestFrame {
                    conversation_id: route.into(),
                    route: route.into(),
                    opener: Valor::Nihil,
                    target: None,
                },
                &context,
            )
            .expect("clock route");
        assert!(matches!(
            reply.contents.as_slice(),
            [ProviderContent::Item(Valor::Numerus(_))]
        ));
    }
}

#[test]
fn sleep_rejects_invalid_duration() {
    let provider = Tempus::new().expect("provider");
    let context = DispatchContext {
        cancellation: host_kernel::CancellationProbe::new(|| false),
    };
    for opener in [Valor::Textus("slow".into()), Valor::Numerus(-1)] {
        let error = provider
            .dispatch(
                &RequestFrame {
                    conversation_id: "invalid-sleep".into(),
                    route: "tempus:dormiet".into(),
                    opener,
                    target: None,
                },
                &context,
            )
            .expect_err("invalid sleep duration");
        assert_eq!(error.code, "E_INVALID_ARGS");
    }
}

#[test]
fn sleep_accepts_zero_and_small_positive_ms() {
    let provider = Tempus::new().expect("provider");
    let context = DispatchContext {
        cancellation: host_kernel::CancellationProbe::new(|| false),
    };
    for ms in [0, 1] {
        let reply = provider
            .dispatch(
                &RequestFrame {
                    conversation_id: format!("sleep-{ms}"),
                    route: "tempus:dormiet".into(),
                    opener: Valor::Numerus(ms),
                    target: None,
                },
                &context,
            )
            .unwrap_or_else(|error| panic!("sleep({ms}) must succeed: {error}"));
        assert!(reply.contents.is_empty(), "sleep({ms}) must return vacuum");
    }
}
