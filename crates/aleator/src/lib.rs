//! Public `aleator` provider.

use faber::Valor;
use host_kernel::{
    DispatchContext, HostError, HostResult, Kernel, Provider, ProviderRegistration, ProviderReply,
    RequestFrame,
};
use std::fs::File;
use std::io::Read;
use std::sync::Arc;
use std::sync::{Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_RANDOM_BYTES: usize = 1024 * 1024;

pub struct Aleator {
    registration: ProviderRegistration,
}

impl Aleator {
    /// Create a new [`Aleator`] provider.
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

/// Register the [`Aleator`] provider with the kernel.
///
/// # Errors
///
/// Returns [`HostError`] if the provider cannot be created
/// (manifest parsing failure) or if registration fails.
pub fn register(kernel: &mut Kernel) -> HostResult<()> {
    kernel.register(Arc::new(Aleator::new()?))
}

#[must_use]
pub fn manifest_json() -> &'static str {
    include_str!("manifest.json")
}

impl Provider for Aleator {
    fn registration(&self) -> &ProviderRegistration {
        &self.registration
    }

    fn dispatch(
        &self,
        request: &RequestFrame,
        _context: &DispatchContext,
    ) -> HostResult<ProviderReply> {
        match request.route.as_str() {
            "aleator:fractum" => Ok(ProviderReply::item(Valor::Fractus(random_fraction()))),
            "aleator:sortire" => sort_integer(&request.opener),
            "aleator:octetos" => random_bytes_route(&request.opener),
            "aleator:uuid" => uuid_route(),
            "aleator:semina" => seed(&request.opener),
            other => Err(HostError::no_route(format!(
                "no built-in aleator syscall registered for {other}"
            ))),
        }
    }
}

#[derive(Clone, Copy)]
struct Prng {
    state: u64,
}

impl Prng {
    fn next_u64(&mut self) -> u64 {
        if self.state == 0 {
            self.state = default_seed();
        }
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

static RNG: Mutex<Prng> = Mutex::new(Prng { state: 0 });

fn rng() -> MutexGuard<'static, Prng> {
    RNG.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[allow(clippy::cast_precision_loss)]
fn random_fraction() -> f64 {
    let bits = rng().next_u64() >> 11;
    (bits as f64) / ((1_u64 << 53) as f64)
}

#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation
)]
fn sort_integer(opener: &Valor) -> HostResult<ProviderReply> {
    let min = i64_arg(opener, 0, "min")?;
    let max = i64_arg(opener, 1, "max")?;
    let (lo, hi) = if min <= max { (min, max) } else { (max, min) };
    // SAFETY: `span = hi - lo + 1` with `hi >= lo` so the result is non-negative
    // and fits `u128`. `offset < span` crosses u128 → i128, then `lo + offset`
    // is bounded by [lo, hi], which fits `i64` by construction.
    let span = (i128::from(hi) - i128::from(lo) + 1) as u128;
    let offset = (u128::from(rng().next_u64()) % span) as i128;
    Ok(ProviderReply::item(Valor::Numerus(
        (i128::from(lo) + offset) as i64,
    )))
}

fn random_bytes_route(opener: &Valor) -> HostResult<ProviderReply> {
    let n = i64_arg(opener, 0, "n")?;
    let len = bounded_non_negative_len(n, MAX_RANDOM_BYTES, "aleator:octetos", "n")?;
    let mut bytes = vec![0_u8; len];
    if len > 0 {
        File::open("/dev/urandom")
            .and_then(|mut file| file.read_exact(&mut bytes))
            .map_err(|error| {
                HostError::internal(format!("aleator random bytes failed: {error}"))
            })?;
    }
    Ok(ProviderReply::byte(bytes))
}

fn bounded_non_negative_len(value: i64, max: usize, route: &str, name: &str) -> HostResult<usize> {
    if value <= 0 {
        return Ok(0);
    }
    let len = usize::try_from(value)
        .map_err(|_| HostError::invalid_args(format!("{route} {name} is too large")))?;
    if len > max {
        return Err(HostError::invalid_args(format!(
            "{route} {name} must be at most {max} bytes"
        )));
    }
    Ok(len)
}

fn uuid_route() -> HostResult<ProviderReply> {
    let mut bytes = vec![0_u8; 16];
    File::open("/dev/urandom")
        .and_then(|mut file| file.read_exact(&mut bytes))
        .map_err(|error| HostError::internal(format!("aleator random bytes failed: {error}")))?;
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Ok(ProviderReply::item(Valor::Textus(format_uuid(&bytes))))
}

fn seed(opener: &Valor) -> HostResult<ProviderReply> {
    let n = i64_arg(opener, 0, "n")?;
    // SAFETY: `n > 0` guards the cast; negative values use `default_seed()`.
    #[allow(clippy::cast_sign_loss)]
    let state = if n > 0 { n as u64 } else { default_seed() };
    rng().state = state;
    Ok(ProviderReply::vacuum())
}

#[allow(clippy::cast_possible_truncation)]
fn default_seed() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos() as u64);
    nanos ^ u64::from(std::process::id())
}

fn format_uuid(bytes: &[u8]) -> String {
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
    )
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
        let error =
            bounded_non_negative_len(101, 100, "aleator:octetos", "n").expect_err("over max");
        assert_eq!(error.code, "E_INVALID_ARGS");
        assert!(error.message.contains("aleator:octetos"));
        assert!(error.message.contains("must be at most 100"));
    }
}
