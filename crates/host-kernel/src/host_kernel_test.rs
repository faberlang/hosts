use super::*;

struct TestProvider {
    registration: ProviderRegistration,
    reply_with_provider: bool,
    fixed_reply: Option<ProviderReply>,
    fixed_error: Option<HostError>,
    panic_on_dispatch: Option<PanicPayload>,
}

#[derive(Clone, Copy)]
enum PanicPayload {
    Message,
    Number,
}

impl TestProvider {
    fn new(prefix: &str, route: &str) -> Self {
        Self::named(prefix, prefix, route)
    }

    fn named(provider: &str, prefix: &str, route: &str) -> Self {
        Self {
            registration: ProviderRegistration::new(manifest(provider, &[prefix], &[route])),
            reply_with_provider: false,
            fixed_reply: None,
            fixed_error: None,
            panic_on_dispatch: None,
        }
    }

    fn routed(provider: &str, prefixes: &[&str], routes: &[&str]) -> Self {
        Self {
            registration: ProviderRegistration::new(manifest(provider, prefixes, routes)),
            reply_with_provider: true,
            fixed_reply: None,
            fixed_error: None,
            panic_on_dispatch: None,
        }
    }

    fn with_reply(mut self, reply: ProviderReply) -> Self {
        self.fixed_reply = Some(reply);
        self
    }

    fn with_error(mut self, error: HostError) -> Self {
        self.fixed_error = Some(error);
        self
    }

    fn panicking(mut self) -> Self {
        self.panic_on_dispatch = Some(PanicPayload::Message);
        self
    }

    fn panicking_with_number(mut self) -> Self {
        self.panic_on_dispatch = Some(PanicPayload::Number);
        self
    }
}

fn manifest(provider: &str, prefixes: &[&str], routes: &[&str]) -> ProviderManifest {
    ProviderManifest {
        manifest_version: 1,
        provider: provider.to_owned(),
        owner: "test".to_owned(),
        prefixes: prefixes.iter().map(|prefix| (*prefix).to_owned()).collect(),
        calls: routes
            .iter()
            .map(|route| ManifestCall {
                route: (*route).to_owned(),
                summary: "test route".to_owned(),
                opener: "valor".to_owned(),
                result: "valor".to_owned(),
            })
            .collect(),
        native_dependencies: Vec::new(),
    }
}

fn invalid_manifest_provider(mut update: impl FnMut(&mut ProviderManifest)) -> TestProvider {
    let mut provider = TestProvider::new("test", "test:echo");
    update(&mut provider.registration.manifest);
    provider
}

impl Provider for TestProvider {
    fn registration(&self) -> &ProviderRegistration {
        &self.registration
    }

    fn dispatch(
        &self,
        request: &RequestFrame,
        _context: &DispatchContext,
    ) -> HostResult<ProviderReply> {
        match self.panic_on_dispatch {
            Some(PanicPayload::Message) => panic!("provider panic fixture"),
            Some(PanicPayload::Number) => std::panic::panic_any(17_u32),
            None => {}
        }
        if let Some(error) = &self.fixed_error {
            return Err(error.clone());
        }
        if let Some(reply) = &self.fixed_reply {
            return Ok(reply.clone());
        }
        if self.reply_with_provider {
            return Ok(ProviderReply::item(Valor::Textus(
                self.registration.manifest.provider.clone(),
            )));
        }
        Ok(ProviderReply::item(request.opener.clone()))
    }
}

fn context() -> DispatchContext {
    DispatchContext {
        cancellation: CancellationProbe::new(|| false),
    }
}

fn request(route: &str) -> RequestFrame {
    RequestFrame {
        conversation_id: format!("conversation-{route}"),
        route: route.to_owned(),
        opener: Valor::Textus(route.to_owned()),
        target: None,
    }
}

fn request_with_opener(route: &str, opener: Valor) -> RequestFrame {
    RequestFrame {
        conversation_id: format!("conversation-{route}"),
        route: route.to_owned(),
        opener,
        target: None,
    }
}

#[test]
fn registration_and_dispatch_are_manifest_checked() {
    let provider = Arc::new(TestProvider::new("test", "test:echo"));
    let mut kernel = Kernel::new();
    kernel.register(provider).expect("register provider");
    let reply = kernel
        .dispatch(
            &RequestFrame {
                conversation_id: "c1".to_owned(),
                route: "test:echo".to_owned(),
                opener: Valor::Textus("ok".to_owned()),
                target: None,
            },
            &context(),
        )
        .expect("dispatch route");
    assert_eq!(reply, ProviderReply::item(Valor::Textus("ok".to_owned())));
    assert!(kernel
        .dispatch(
            &RequestFrame {
                conversation_id: "c2".to_owned(),
                route: "test:missing".to_owned(),
                opener: Valor::Nihil,
                target: None,
            },
            &context(),
        )
        .is_err());
}

#[test]
fn duplicate_prefix_between_distinct_providers_fails_closed() {
    let mut kernel = Kernel::new();
    kernel
        .register(Arc::new(TestProvider::named(
            "first-provider",
            "shared",
            "shared:echo",
        )))
        .expect("first provider");
    let error = kernel
        .register(Arc::new(TestProvider::named(
            "second-provider",
            "shared",
            "shared:other",
        )))
        .expect_err("duplicate provider prefixes must fail closed");

    assert_eq!(error.code, "E_INTERNAL");
    assert!(error.message.contains("duplicate provider prefix `shared`"));
    assert!(kernel.supports_route("shared:echo"));
    assert!(
        !kernel.supports_route("shared:other"),
        "rejected provider routes must not be admitted"
    );
    assert_eq!(kernel.manifest().providers.len(), 1);
}

#[test]
fn multi_prefix_provider_dispatches_all_families_and_aggregates_once() {
    let mut kernel = Kernel::new();
    kernel
        .register(Arc::new(TestProvider::routed(
            "multi",
            &["alpha", "beta"],
            &["alpha:echo", "beta:echo"],
        )))
        .expect("multi-prefix provider");

    assert!(kernel.supports_route("alpha:echo"));
    assert!(kernel.supports_route("beta:echo"));
    assert_eq!(
        kernel
            .dispatch(&request("alpha:echo"), &context())
            .expect("dispatch alpha family"),
        ProviderReply::item(Valor::Textus("multi".to_owned()))
    );
    assert_eq!(
        kernel
            .dispatch(&request("beta:echo"), &context())
            .expect("dispatch beta family"),
        ProviderReply::item(Valor::Textus("multi".to_owned()))
    );

    let manifest = kernel.manifest();
    assert_eq!(manifest.providers.len(), 1);
    assert_eq!(manifest.providers[0].provider, "multi");
    assert_eq!(
        manifest.providers[0].prefixes,
        vec!["alpha".to_owned(), "beta".to_owned()]
    );
    assert_eq!(manifest.providers[0].calls.len(), 2);
}

#[test]
fn multi_prefix_conflict_rejects_provider_without_partial_admission() {
    let mut kernel = Kernel::new();
    kernel
        .register(Arc::new(TestProvider::routed(
            "first",
            &["alpha", "beta"],
            &["alpha:echo", "beta:echo"],
        )))
        .expect("first multi-prefix provider");

    let error = kernel
        .register(Arc::new(TestProvider::routed(
            "second",
            &["gamma", "beta"],
            &["gamma:echo", "beta:other"],
        )))
        .expect_err("conflict on any prefix should reject provider");

    assert_eq!(error.code, "E_INTERNAL");
    assert!(error.message.contains("duplicate provider prefix `beta`"));
    assert!(kernel.supports_route("alpha:echo"));
    assert!(kernel.supports_route("beta:echo"));
    assert!(
        !kernel.supports_route("gamma:echo"),
        "non-conflicting prefix from rejected provider must not be admitted"
    );
    assert!(!kernel.supports_route("beta:other"));
    assert_eq!(
        kernel
            .dispatch(&request("alpha:echo"), &context())
            .expect("dispatch admitted alpha family"),
        ProviderReply::item(Valor::Textus("first".to_owned()))
    );
    assert_eq!(kernel.manifest().providers.len(), 1);
}

#[test]
fn unsupported_manifest_versions_are_rejected() {
    let mut kernel = Kernel::new();
    let error = kernel
        .register(Arc::new(invalid_manifest_provider(|manifest| {
            manifest.manifest_version = 2;
        })))
        .expect_err("unsupported manifest versions must fail closed");

    assert_eq!(error.code, "E_INVALID_ARGS");
    assert!(error
        .message
        .contains("unsupported provider manifest version 2"));
    assert!(!kernel.supports_route("test:echo"));
}

#[test]
fn invalid_prefixes_are_rejected() {
    for prefix in [
        "",
        "Upper",
        "bad.prefix",
        "bad/prefix",
        "-",
        "_",
        "123",
        "-bad",
        "_bad",
        "1bad",
    ] {
        let mut kernel = Kernel::new();
        let result = kernel.register(Arc::new(invalid_manifest_provider(|manifest| {
            manifest.prefixes = vec![prefix.to_owned()];
            manifest.calls[0].route = format!("{prefix}:echo");
        })));
        let error = match result {
            Ok(()) => panic!("invalid prefix `{prefix}` must fail closed"),
            Err(error) => error,
        };

        assert_eq!(error.code, "E_INVALID_ARGS");
        assert!(error.message.contains("invalid provider prefix"));
        assert!(!kernel.supports_route(&format!("{prefix}:echo")));
    }
}

#[test]
fn prefix_names_may_use_internal_separators_after_ascii_letter() {
    for prefix in ["test", "test2", "test-case", "test_case", "a-b_c3"] {
        let mut kernel = Kernel::new();
        kernel
            .register(Arc::new(TestProvider::new(
                prefix,
                &format!("{prefix}:echo"),
            )))
            .unwrap_or_else(|error| panic!("valid prefix `{prefix}` rejected: {error}"));
        assert!(kernel.supports_route(&format!("{prefix}:echo")));
    }
}

#[test]
fn duplicate_prefixes_inside_manifest_are_rejected() {
    let mut kernel = Kernel::new();
    let error = kernel
        .register(Arc::new(invalid_manifest_provider(|manifest| {
            manifest.prefixes = vec!["test".to_owned(), "test".to_owned()];
        })))
        .expect_err("duplicate prefixes inside one manifest must fail closed");

    assert_eq!(error.code, "E_INVALID_ARGS");
    assert!(error.message.contains("provider prefixes must be unique"));
    assert!(!kernel.supports_route("test:echo"));
}

#[test]
fn repeated_routes_inside_manifest_are_rejected() {
    let mut kernel = Kernel::new();
    let error = kernel
        .register(Arc::new(invalid_manifest_provider(|manifest| {
            manifest.calls.push(manifest.calls[0].clone());
        })))
        .expect_err("repeated routes inside one manifest must fail closed");

    assert_eq!(error.code, "E_INTERNAL");
    assert!(error.message.contains("provider manifest repeats route"));
    assert!(!kernel.supports_route("test:echo"));
}

#[test]
fn duplicate_provider_identity_fails_before_manifest_aggregation() {
    let mut kernel = Kernel::new();
    kernel
        .register(Arc::new(TestProvider::named("same", "one", "one:echo")))
        .expect("first provider");
    let error = kernel
        .register(Arc::new(TestProvider::named("same", "two", "two:echo")))
        .expect_err("provider identity must be unique");

    assert_eq!(error.code, "E_INTERNAL");
    assert!(error.message.contains("duplicate provider identity `same`"));
    assert!(kernel.supports_route("one:echo"));
    assert!(!kernel.supports_route("two:echo"));
    assert_eq!(kernel.manifest().providers.len(), 1);
}

#[test]
fn empty_call_manifests_are_rejected() {
    let mut provider = TestProvider::new("empty", "empty:call");
    provider.registration.manifest.calls.clear();
    let mut kernel = Kernel::new();
    let error = kernel
        .register(Arc::new(provider))
        .expect_err("empty call manifests must be rejected");

    assert_eq!(error.code, "E_INVALID_ARGS");
    assert!(error.message.contains("at least one call"));
}

#[test]
fn invalid_route_grammar_cases_are_rejected() {
    for (route, message) in [
        ("testecho", "has no prefix"),
        ("test:", "does not match provider prefixes"),
        ("other:echo", "does not match provider prefixes"),
        (":echo", "does not match provider prefixes"),
        ("test:echo:extra", "does not match provider prefixes"),
    ] {
        let mut kernel = Kernel::new();
        let error = kernel
            .register(Arc::new(invalid_manifest_provider(|manifest| {
                manifest.calls[0].route = route.to_owned();
            })))
            .unwrap_err();

        assert_eq!(error.code, "E_INVALID_ARGS", "{route}");
        assert!(
            error.message.contains(message),
            "route `{route}` produced unexpected error: {}",
            error.message
        );
        assert!(
            !kernel.supports_route(route),
            "rejected route `{route}` must not be admitted"
        );
        assert!(!kernel.supports_route("test:echo"));
        assert!(kernel.manifest().providers.is_empty());
    }
}

#[test]
fn invalid_opener_contracts_are_rejected() {
    for opener in ["", "bytes", "genus", "valor "] {
        let mut kernel = Kernel::new();
        let error = kernel
            .register(Arc::new(invalid_manifest_provider(|manifest| {
                manifest.calls[0].opener = opener.to_owned();
            })))
            .unwrap_err();

        assert_eq!(error.code, "E_INVALID_ARGS", "{opener:?}");
        assert!(
            error.message.contains("unsupported opener contract"),
            "opener `{opener}` produced unexpected error: {}",
            error.message
        );
        assert!(!kernel.supports_route("test:echo"));
    }
}

#[test]
fn invalid_result_contracts_are_rejected() {
    for result in ["", "byte", "genus", "valor "] {
        let mut kernel = Kernel::new();
        let error = kernel
            .register(Arc::new(invalid_manifest_provider(|manifest| {
                manifest.calls[0].result = result.to_owned();
            })))
            .unwrap_err();

        assert_eq!(error.code, "E_INVALID_ARGS", "{result:?}");
        assert!(
            error.message.contains("unsupported result contract"),
            "result `{result}` produced unexpected error: {}",
            error.message
        );
        assert!(!kernel.supports_route("test:echo"));
    }
}

#[test]
fn dispatch_rejects_provider_reply_that_violates_manifest_result_contract() {
    let mut provider =
        TestProvider::new("test", "test:echo").with_reply(ProviderReply::byte(vec![1, 2, 3]));
    provider.registration.manifest.calls[0].result = "valor".to_owned();
    let mut kernel = Kernel::new();
    kernel
        .register(Arc::new(provider))
        .expect("register provider with valor result contract");

    let error = kernel
        .dispatch(&request("test:echo"), &context())
        .expect_err("byte reply must not satisfy valor contract");

    assert_eq!(error.code, "E_INTERNAL");
    assert!(error
        .message
        .contains("does not match declared result contract"));
}

#[test]
fn dispatch_converts_provider_panic_to_host_error() {
    let provider = TestProvider::new("test", "test:echo").panicking();
    let mut kernel = Kernel::new();
    kernel
        .register(Arc::new(provider))
        .expect("register panicking provider");

    let error = kernel
        .dispatch(&request("test:echo"), &context())
        .expect_err("provider panic must be converted at kernel boundary");

    assert_eq!(error.code, "E_PROVIDER_PANIC");
    assert_eq!(error.message, "provider panicked");
    assert!(!error.retryable);
}

#[test]
fn dispatch_contains_non_string_panic_payload_and_kernel_remains_usable() {
    let mut kernel = Kernel::new();
    kernel
        .register(Arc::new(
            TestProvider::new("panic", "panic:boom").panicking_with_number(),
        ))
        .expect("register panicking provider");
    kernel
        .register(Arc::new(TestProvider::new("ok", "ok:echo")))
        .expect("register healthy provider");

    let dispatch = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        kernel.dispatch(&request("panic:boom"), &context())
    }))
    .expect("provider panic must not unwind across kernel boundary");
    let error = dispatch.expect_err("contained provider panic must become host error");

    assert_eq!(error.code, "E_PROVIDER_PANIC");
    assert_eq!(
        kernel
            .dispatch(&request("ok:echo"), &context())
            .expect("kernel should dispatch after contained panic"),
        ProviderReply::item(Valor::Textus("ok:echo".to_owned()))
    );
}

#[test]
fn dispatch_preserves_provider_errors_without_rewriting_as_panic() {
    let provider_error = HostError::try_new("E_PROVIDER_DOWN", "provider is down", true)
        .expect("valid provider error");
    let provider = TestProvider::new("test", "test:echo").with_error(provider_error.clone());
    let mut kernel = Kernel::new();
    kernel
        .register(Arc::new(provider))
        .expect("register erroring provider");

    let error = kernel
        .dispatch(&request("test:echo"), &context())
        .expect_err("provider error must pass through unchanged");

    assert_eq!(error, provider_error);
}

#[test]
fn dispatch_accepts_declared_non_valor_reply_contracts() {
    for (result, reply) in [
        ("vacuum", ProviderReply::vacuum()),
        (
            "textus",
            ProviderReply::item(Valor::Textus("text".to_owned())),
        ),
        ("numerus", ProviderReply::item(Valor::Numerus(7))),
        ("fractus", ProviderReply::item(Valor::Fractus(0.5))),
        ("bivalens", ProviderReply::item(Valor::Bivalens(true))),
        ("octeti", ProviderReply::byte(vec![1, 2, 3])),
        (
            "instans<ns>",
            ProviderReply::item(Valor::Instans("2026-07-14T00:00:00Z".to_owned())),
        ),
        ("bytes", ProviderReply::byte(vec![1, 2, 3])),
        (
            "lista<textus>",
            ProviderReply::list([
                Valor::Textus("one".to_owned()),
                Valor::Textus("two".to_owned()),
            ]),
        ),
        (
            "lista-valor",
            ProviderReply::list([
                Valor::Textus("one".to_owned()),
                Valor::Textus("two".to_owned()),
            ]),
        ),
        (
            "bulk-valor",
            ProviderReply {
                contents: vec![ProviderContent::Bulk(Valor::Textus("bulk".to_owned()))],
            },
        ),
    ] {
        let mut provider = TestProvider::new("test", "test:echo").with_reply(reply.clone());
        provider.registration.manifest.calls[0].result = result.to_owned();
        let mut kernel = Kernel::new();
        kernel
            .register(Arc::new(provider))
            .unwrap_or_else(|error| panic!("register provider with `{result}` result: {error}"));

        assert_eq!(
            kernel
                .dispatch(&request("test:echo"), &context())
                .unwrap_or_else(|error| panic!("dispatch `{result}` result: {error}")),
            reply
        );
    }
}

#[test]
fn vacuum_contract_accepts_empty_frame_carrier_and_optional_numerus() {
    let mut vacuum_provider = TestProvider::new("vacuum", "vacuum:echo")
        .with_reply(ProviderReply::item(Valor::Textus("ok".to_owned())));
    vacuum_provider.registration.manifest.calls[0].opener = "vacuum".to_owned();
    vacuum_provider.registration.manifest.calls[0].result = "textus".to_owned();
    let mut kernel = Kernel::new();
    kernel
        .register(Arc::new(vacuum_provider))
        .expect("register vacuum provider");
    assert_eq!(
        kernel
            .dispatch(
                &request_with_opener("vacuum:echo", Valor::Tabula(BTreeMap::new())),
                &context(),
            )
            .expect("empty frame carrier is vacuum"),
        ProviderReply::item(Valor::Textus("ok".to_owned()))
    );

    let mut optional_provider = TestProvider::new("optional", "optional:echo")
        .with_reply(ProviderReply::item(Valor::Textus("ok".to_owned())));
    optional_provider.registration.manifest.calls[0].opener = "sponte<numerus>".to_owned();
    optional_provider.registration.manifest.calls[0].result = "textus".to_owned();
    let mut kernel = Kernel::new();
    kernel
        .register(Arc::new(optional_provider))
        .expect("register optional numerus provider");
    for opener in [Valor::Tabula(BTreeMap::new()), Valor::Numerus(7)] {
        assert_eq!(
            kernel
                .dispatch(&request_with_opener("optional:echo", opener), &context())
                .expect("optional numerus opener"),
            ProviderReply::item(Valor::Textus("ok".to_owned()))
        );
    }
}

#[test]
fn solum_lege_textus_opener_and_result_contract_are_supported() {
    let mut provider = TestProvider::new("solum", "solum:lege")
        .with_reply(ProviderReply::item(Valor::Textus("body".to_owned())));
    provider.registration.manifest.calls[0].opener = "textus".to_owned();
    provider.registration.manifest.calls[0].result = "textus".to_owned();
    let mut kernel = Kernel::new();
    kernel
        .register(Arc::new(provider))
        .expect("register solum textus contract");

    assert_eq!(
        kernel
            .dispatch(
                &request_with_opener("solum:lege", Valor::Textus("data.txt".to_owned())),
                &context(),
            )
            .expect("dispatch solum:lege textus opener"),
        ProviderReply::item(Valor::Textus("body".to_owned()))
    );
}

#[test]
fn dispatch_rejects_request_opener_that_violates_manifest_contract() {
    let mut provider = TestProvider::new("solum", "solum:lege")
        .with_reply(ProviderReply::item(Valor::Textus("body".to_owned())));
    provider.registration.manifest.calls[0].opener = "textus".to_owned();
    provider.registration.manifest.calls[0].result = "textus".to_owned();
    let mut kernel = Kernel::new();
    kernel
        .register(Arc::new(provider))
        .expect("register provider");

    let error = kernel
        .dispatch(
            &request_with_opener("solum:lege", Valor::Numerus(7)),
            &context(),
        )
        .expect_err("numerus opener must not satisfy textus contract");

    assert_eq!(error.code, "E_INVALID_ARGS");
    assert!(error
        .message
        .contains("does not match declared opener contract"));
}

#[test]
fn unknown_manifest_fields_are_rejected() {
    let error = parse_manifest(
        r#"{"manifest_version":1,"provider":"test","owner":"x","prefixes":["test"],"calls":[],"native_dependencies":[],"extra":true}"#,
    )
    .expect_err("unknown fields must fail");
    assert_eq!(error.code, "E_INVALID_ARGS");
}

#[test]
fn cancellation_is_checked_before_provider_dispatch() {
    let mut kernel = Kernel::new();
    kernel
        .register(Arc::new(TestProvider::new("test", "test:echo")))
        .expect("register provider");
    let context = DispatchContext {
        cancellation: CancellationProbe::new(|| true),
    };
    let error = kernel
        .dispatch(
            &RequestFrame {
                conversation_id: "c1".to_owned(),
                route: "test:echo".to_owned(),
                opener: Valor::Nihil,
                target: None,
            },
            &context,
        )
        .expect_err("cancelled request");
    assert_eq!(error.code, "E_CANCELLED");
}

#[test]
fn dispatch_contains_cancellation_probe_panic() {
    let mut kernel = Kernel::new();
    kernel
        .register(Arc::new(TestProvider::new("test", "test:echo")))
        .expect("register provider");
    let panicking_context = DispatchContext {
        cancellation: CancellationProbe::new(|| panic!("cancellation probe panic fixture")),
    };

    let dispatch = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        kernel.dispatch(&request("test:echo"), &panicking_context)
    }))
    .expect("cancellation probe panic must not unwind across kernel boundary");
    let error = dispatch.expect_err("contained cancellation panic must become host error");

    assert_eq!(error.code, "E_INTERNAL");
    assert_eq!(error.message, "cancellation probe panicked");
    assert!(!error.retryable);
    assert_eq!(
        kernel
            .dispatch(&request("test:echo"), &context())
            .expect("kernel should dispatch after contained cancellation panic"),
        ProviderReply::item(Valor::Textus("test:echo".to_owned()))
    );
}
