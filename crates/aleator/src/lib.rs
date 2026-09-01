//! Public `aleator` provider.

use faber::Valor;
use host_kernel::{
    DispatchContext, HostError, HostResult, Kernel, Provider, ProviderRegistration, ProviderReply,
    RequestFrame,
};
use std::fs::File;
use std::io::Read;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_RANDOM_BYTES: usize = 1024 * 1024;

pub struct Aleator {
    registration: ProviderRegistration,
    rng: Mutex<Prng>,
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
            rng: Mutex::new(Prng { state: 0 }),
        })
    }

    fn rng(&self) -> MutexGuard<'_, Prng> {
        self.rng
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[allow(clippy::cast_precision_loss)]
    fn random_fraction(&self) -> f64 {
        let bits = self.rng().next_u64() >> 11;
        (bits as f64) / ((1_u64 << 53) as f64)
    }

    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation
    )]
    fn sort_integer(&self, opener: &Valor) -> HostResult<ProviderReply> {
        let min = i64_arg(opener, 0, "min")?;
        let max = i64_arg(opener, 1, "max")?;
        let (lo, hi) = if min <= max { (min, max) } else { (max, min) };
        // SAFETY: `span = hi - lo + 1` with `hi >= lo` so the result is non-negative
        // and fits `u128`. `offset < span` crosses u128 → i128, then `lo + offset`
        // is bounded by [lo, hi], which fits `i64` by construction.
        let span = (i128::from(hi) - i128::from(lo) + 1) as u128;
        let offset = (u128::from(self.rng().next_u64()) % span) as i128;
        Ok(ProviderReply::item(Valor::Numerus(
            (i128::from(lo) + offset) as i64,
        )))
    }

    fn seed(&self, opener: &Valor) -> HostResult<ProviderReply> {
        let n = i64_arg(opener, 0, "n")?;
        // SAFETY: `n > 0` guards the cast; negative values use `default_seed()`.
        #[allow(clippy::cast_sign_loss)]
        let state = if n > 0 { n as u64 } else { default_seed() };
        self.rng().state = state;
        Ok(ProviderReply::vacuum())
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
            "aleator:fractum" => Ok(ProviderReply::item(Valor::Fractus(self.random_fraction()))),
            "aleator:sortire" => self.sort_integer(&request.opener),
            "aleator:octetos" => random_bytes_route(&request.opener),
            "aleator:uuid" => uuid_route(),
            "aleator:semina" => self.seed(&request.opener),
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

fn random_bytes_route(opener: &Valor) -> HostResult<ProviderReply> {
    let n = i64_arg(opener, 0, "n")?;
    let len = bounded_non_negative_len(n, MAX_RANDOM_BYTES, "aleator:octetos", "n")?;
    Ok(ProviderReply::byte(random_bytes(len)?))
}

fn random_bytes(len: usize) -> HostResult<Vec<u8>> {
    let mut bytes = vec![0_u8; len];
    if len > 0 {
        File::open("/dev/urandom")
            .and_then(|mut file| file.read_exact(&mut bytes))
            .map_err(|error| {
                HostError::internal(format!("aleator random bytes failed: {error}"))
            })?;
    }
    Ok(bytes)
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
    let mut bytes = random_bytes(16)?;
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Ok(ProviderReply::item(Valor::Textus(format_uuid(&bytes))))
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
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15]
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
#[path = "lib_test.rs"]
mod tests;
