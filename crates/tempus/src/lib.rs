//! Public `tempus` provider.

use faber::{Instans, InstansPraecisio, Valor};
use host_kernel::{
    DispatchContext, HostError, HostResult, Kernel, Provider, ProviderRegistration, ProviderReply,
    RequestFrame,
};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub struct Tempus {
    registration: ProviderRegistration,
}

impl Tempus {
    /// Create a new [`Tempus`] provider.
    ///
    /// # Errors
    ///
    /// Returns [`HostError`] if the embedded manifest JSON cannot be parsed.
    pub fn new() -> HostResult<Self> {
        Ok(Self {
            registration: ProviderRegistration::new(host_kernel::parse_manifest(manifest_json())?),
        })
    }
}

/// Register the [`Tempus`] provider with the kernel.
///
/// # Errors
///
/// Returns [`HostError`] if the provider cannot be created
/// (manifest parsing failure) or if registration fails.
pub fn register(kernel: &mut Kernel) -> HostResult<()> {
    kernel.register(Arc::new(Tempus::new()?))
}

#[must_use]
pub fn manifest_json() -> &'static str {
    include_str!("manifest.json")
}

impl Provider for Tempus {
    fn registration(&self) -> &ProviderRegistration {
        &self.registration
    }

    fn dispatch(
        &self,
        request: &RequestFrame,
        context: &DispatchContext,
    ) -> HostResult<ProviderReply> {
        match request.route.as_str() {
            "tempus:nunc" => {
                let instant = Instans::from_nanos(epoch_nanos()?, InstansPraecisio::Nanosecunda);
                Ok(ProviderReply::item(instant.into()))
            }
            "tempus:monotonicum" | "tempus:activum" => {
                Ok(ProviderReply::item(elapsed_nanos()?.into()))
            }
            "tempus:dormiet" => sleep(&request.opener, context),
            other => Err(HostError::no_route(format!(
                "no built-in tempus syscall registered for {other}"
            ))),
        }
    }
}

static START: OnceLock<Instant> = OnceLock::new();

fn sleep(opener: &Valor, context: &DispatchContext) -> HostResult<ProviderReply> {
    if context.cancellation.is_cancelled() {
        return Err(HostError::cancelled());
    }
    let ms = i64_arg(opener, 0, "ms")?;
    if ms < 0 {
        return Err(HostError::invalid_args("ms must be non-negative"));
    }
    if ms > 0 {
        // SAFETY: `ms >= 0` was checked above.
        #[allow(clippy::cast_sign_loss)]
        let deadline = Instant::now() + Duration::from_millis(ms as u64);
        while Instant::now() < deadline {
            if context.cancellation.is_cancelled() {
                return Err(HostError::cancelled());
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            std::thread::sleep(remaining.min(Duration::from_millis(5)));
        }
    }
    Ok(ProviderReply::vacuum())
}

fn epoch_nanos() -> HostResult<i64> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| HostError::internal(format!("tempus:nunc failed: {error}")))?
        .as_nanos();
    i64::try_from(nanos)
        .map_err(|_| HostError::internal("tempus:nunc exceeded i64 nanosecond range"))
}

fn elapsed_nanos() -> HostResult<i64> {
    let nanos = START.get_or_init(Instant::now).elapsed().as_nanos();
    i64::try_from(nanos)
        .map_err(|_| HostError::internal("tempus elapsed time exceeded i64 nanosecond range"))
}

fn i64_arg(value: &Valor, index: usize, name: &str) -> HostResult<i64> {
    let value = match value {
        Valor::Lista(values) => values.get(index),
        value if index == 0 => Some(value),
        _ => None,
    };
    match value {
        Some(Valor::Numerus(number)) => Ok(*number),
        Some(_) => Err(HostError::invalid_args(format!("{name} must be numerus"))),
        None => Err(HostError::invalid_args(format!("missing {name}"))),
    }
}

#[cfg(test)]
mod tests {
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
}
