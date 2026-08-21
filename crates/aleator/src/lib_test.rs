use super::*;
use host_kernel::{Kernel, ProviderContent};

fn test_provider() -> (Aleator, host_kernel::DispatchContext) {
    let provider = Aleator::new().expect("provider");
    let context = host_kernel::DispatchContext {
        cancellation: host_kernel::CancellationProbe::new(|| false),
    };
    (provider, context)
}

fn request_frame(route: &str, opener: Valor) -> RequestFrame {
    RequestFrame {
        conversation_id: route.into(),
        route: route.into(),
        opener,
        target: None,
    }
}

fn item_data(reply: &ProviderReply) -> Valor {
    match reply.contents.as_slice() {
        [ProviderContent::Item(value)] => value.clone(),
        other => panic!("expected item reply, got {other:?}"),
    }
}

fn seed_provider(provider: &Aleator, context: &host_kernel::DispatchContext, n: i64) {
    provider
        .dispatch(&request_frame("aleator:semina", Valor::Numerus(n)), context)
        .expect("seed");
}

fn fractum(provider: &Aleator, context: &host_kernel::DispatchContext) -> Valor {
    item_data(
        &provider
            .dispatch(&request_frame("aleator:fractum", Valor::Nihil), context)
            .expect("fractum"),
    )
}

#[test]
fn manifest_registers_all_canonical_routes() {
    let mut kernel = Kernel::new();
    register(&mut kernel).expect("register aleator");
    let routes = &kernel.manifest().providers[0].calls;
    assert_eq!(routes.len(), 5);
    assert!(routes.iter().any(|call| call.route == "aleator:octetos"));
}

#[test]
fn seed_returns_vacuum() {
    let (provider, context) = test_provider();
    let reply = provider
        .dispatch(
            &request_frame("aleator:semina", Valor::Numerus(42)),
            &context,
        )
        .expect("seed");
    assert!(reply.contents.is_empty(), "seed reply should be vacuum");
}

#[test]
fn same_seed_produces_the_same_fractum_stream() {
    let (provider, context) = test_provider();
    seed_provider(&provider, &context, 42);
    let first = [fractum(&provider, &context), fractum(&provider, &context)];
    seed_provider(&provider, &context, 42);
    let second = [fractum(&provider, &context), fractum(&provider, &context)];
    assert_eq!(first, second);
}

#[test]
fn independent_providers_do_not_share_rng_state() {
    let (left, context) = test_provider();
    let right = Aleator::new().expect("right provider");
    let interloper = Aleator::new().expect("interloper");

    seed_provider(&left, &context, 42);
    seed_provider(&right, &context, 42);
    seed_provider(&interloper, &context, 7);
    let _ = fractum(&interloper, &context);
    let _ = fractum(&interloper, &context);

    let from_left = [
        fractum(&left, &context),
        fractum(&left, &context),
        fractum(&left, &context),
    ];
    let from_right = [
        fractum(&right, &context),
        fractum(&right, &context),
        fractum(&right, &context),
    ];
    assert_eq!(from_left, from_right);

    seed_provider(&left, &context, 42);
    seed_provider(&interloper, &context, 7);
    let _ = fractum(&interloper, &context);
    assert_eq!(fractum(&left, &context), from_left[0]);
    assert_ne!(fractum(&interloper, &context), from_left[0]);
}

#[test]
fn seeded_octetos_returns_four_bytes() {
    let (provider, context) = test_provider();
    provider
        .dispatch(
            &request_frame("aleator:semina", Valor::Numerus(42)),
            &context,
        )
        .expect("seed");
    let reply = provider
        .dispatch(
            &request_frame("aleator:octetos", Valor::Numerus(4)),
            &context,
        )
        .expect("octetos");
    assert!(
        matches!(reply.contents.as_slice(), [ProviderContent::Byte(value)] if value.len() == 4),
        "expected 4 random bytes"
    );
}

#[test]
fn octetos_zero_returns_empty() {
    let (provider, context) = test_provider();
    let reply = provider
        .dispatch(
            &request_frame("aleator:octetos", Valor::Numerus(0)),
            &context,
        )
        .expect("zero bytes");
    assert!(
        matches!(reply.contents.as_slice(), [ProviderContent::Byte(value)] if value.is_empty()),
        "zero byte request should return empty"
    );
}

#[test]
fn octetos_negative_clamps_to_zero() {
    let (provider, context) = test_provider();
    let reply = provider
        .dispatch(
            &request_frame("aleator:octetos", Valor::Numerus(-1)),
            &context,
        )
        .expect("negative byte count");
    assert!(
        matches!(reply.contents.as_slice(), [ProviderContent::Byte(value)] if value.is_empty()),
        "negative byte request should clamp to empty"
    );
}

#[test]
fn octetos_over_limit_rejected() {
    let (provider, context) = test_provider();
    // SAFETY: `MAX_RANDOM_BYTES` (1 MiB) fits easily in `i64`.
    #[allow(clippy::cast_possible_wrap)]
    let error = provider
        .dispatch(
            &request_frame(
                "aleator:octetos",
                Valor::Numerus(MAX_RANDOM_BYTES as i64 + 1),
            ),
            &context,
        )
        .expect_err("over-limit request must fail");
    assert_eq!(error.code, "E_INVALID_ARGS");
    assert!(error.message.contains("aleator:octetos"));
    assert!(error.message.contains(&MAX_RANDOM_BYTES.to_string()));
}

#[test]
fn bounded_non_negative_len_returns_zero_for_non_positive() {
    assert_eq!(bounded_non_negative_len(-5, 100, "test", "n").unwrap(), 0);
    assert_eq!(bounded_non_negative_len(0, 100, "test", "n").unwrap(), 0);
}

#[test]
fn bounded_non_negative_len_accepts_valid() {
    assert_eq!(bounded_non_negative_len(1, 100, "test", "n").unwrap(), 1);
    assert_eq!(
        bounded_non_negative_len(100, 100, "test", "n").unwrap(),
        100
    );
    assert_eq!(bounded_non_negative_len(50, 100, "test", "n").unwrap(), 50);
}

#[test]
fn bounded_non_negative_len_rejects_over_max() {
    let error = bounded_non_negative_len(101, 100, "aleator:octetos", "n").expect_err("over max");
    assert_eq!(error.code, "E_INVALID_ARGS");
    assert!(error.message.contains("aleator:octetos"));
    assert!(error.message.contains("must be at most 100"));
}
