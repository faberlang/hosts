//! Checked, transport-neutral routing for public Faber host providers.

use faber::Valor;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

pub type HostResult<T> = Result<T, HostError>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostError {
    pub code: String,
    pub message: String,
    pub retryable: bool,
}

impl HostError {
    /// Returns a new `HostError` with the given code, message, and retryable flag.
    ///
    /// # Errors
    ///
    /// Returns `E_INVALID_ARGS` if `code` does not match the expected format
    /// (starts with `E_`, followed by uppercase ASCII letters, digits, or underscores).
    pub fn try_new(
        code: impl Into<String>,
        message: impl Into<String>,
        retryable: bool,
    ) -> HostResult<Self> {
        let code = code.into();
        if !valid_error_code(&code) {
            return Err(Self::invalid_args(format!(
                "invalid host error code `{code}`"
            )));
        }
        Ok(Self {
            code,
            message: message.into(),
            retryable,
        })
    }

    pub fn invalid_args(message: impl Into<String>) -> Self {
        Self::unchecked("E_INVALID_ARGS", message, false)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::unchecked("E_INTERNAL", message, false)
    }

    pub fn no_route(message: impl Into<String>) -> Self {
        Self::unchecked("E_NO_ROUTE", message, false)
    }

    #[must_use]
    pub fn cancelled() -> Self {
        Self::unchecked("E_CANCELLED", "operation cancelled", false)
    }

    #[must_use]
    pub fn provider_panic() -> Self {
        Self::unchecked("E_PROVIDER_PANIC", "provider panicked", false)
    }

    #[must_use]
    pub fn to_valor(&self) -> Valor {
        let mut fields = BTreeMap::new();
        fields.insert("code".to_owned(), Valor::Textus(self.code.clone()));
        fields.insert("message".to_owned(), Valor::Textus(self.message.clone()));
        fields.insert("retryable".to_owned(), Valor::Bivalens(self.retryable));
        Valor::Tabula(fields)
    }

    fn unchecked(code: &'static str, message: impl Into<String>, retryable: bool) -> Self {
        Self {
            code: code.to_owned(),
            message: message.into(),
            retryable,
        }
    }
}

impl fmt::Display for HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for HostError {}

fn valid_error_code(code: &str) -> bool {
    let mut chars = code.chars();
    matches!(chars.next(), Some('E'))
        && matches!(chars.next(), Some('_'))
        && chars.all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '_')
}

#[derive(Clone, Debug)]
pub struct RequestFrame {
    pub conversation_id: String,
    pub route: String,
    pub opener: Valor,
    pub target: Option<String>,
}

#[derive(Clone)]
pub struct CancellationProbe(Arc<dyn Fn() -> bool + Send + Sync>);

impl CancellationProbe {
    pub fn new(is_cancelled: impl Fn() -> bool + Send + Sync + 'static) -> Self {
        Self(Arc::new(is_cancelled))
    }

    pub fn from_flag(flag: Arc<AtomicBool>) -> Self {
        Self::new(move || flag.load(std::sync::atomic::Ordering::SeqCst))
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        (self.0)()
    }

    fn try_is_cancelled(&self) -> HostResult<bool> {
        catch_unwind(AssertUnwindSafe(|| self.is_cancelled()))
            .map_err(|_panic| HostError::internal("cancellation probe panicked"))
    }
}

impl fmt::Debug for CancellationProbe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CancellationProbe").finish_non_exhaustive()
    }
}

#[derive(Clone, Debug)]
pub struct DispatchContext {
    pub cancellation: CancellationProbe,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ProviderContent {
    Item(Valor),
    Byte(Vec<u8>),
    Bulk(Valor),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProviderReply {
    pub contents: Vec<ProviderContent>,
}

impl ProviderReply {
    #[must_use]
    pub fn vacuum() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn item(data: Valor) -> Self {
        Self {
            contents: vec![ProviderContent::Item(data)],
        }
    }

    #[must_use]
    pub fn byte(data: Vec<u8>) -> Self {
        Self {
            contents: vec![ProviderContent::Byte(data)],
        }
    }

    pub fn list(items: impl IntoIterator<Item = Valor>) -> Self {
        Self {
            contents: items.into_iter().map(ProviderContent::Item).collect(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProviderManifest {
    pub manifest_version: u32,
    pub provider: String,
    pub owner: String,
    pub prefixes: Vec<String>,
    pub calls: Vec<ManifestCall>,
    #[serde(default)]
    pub native_dependencies: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ManifestCall {
    pub route: String,
    pub summary: String,
    pub opener: String,
    pub result: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OpenerContract {
    Vacuum,
    SponteNumerus,
    Textus,
    Numerus,
    Octeti,
    ListaTextus,
    ListaNumerus,
    ListaValor,
    Valor,
}

impl OpenerContract {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "vacuum" => Some(Self::Vacuum),
            "sponte<numerus>" => Some(Self::SponteNumerus),
            "textus" => Some(Self::Textus),
            "numerus" => Some(Self::Numerus),
            "octeti" => Some(Self::Octeti),
            "lista<textus>" => Some(Self::ListaTextus),
            "lista<numerus>" => Some(Self::ListaNumerus),
            "lista<valor>" => Some(Self::ListaValor),
            "valor" => Some(Self::Valor),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResultContract {
    Vacuum,
    Textus,
    Numerus,
    Fractus,
    Bivalens,
    Octeti,
    InstansNs,
    ListaTextus,
    Valor,
    Bytes,
    ListaValor,
    BulkValor,
}

impl ResultContract {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "vacuum" => Some(Self::Vacuum),
            "textus" => Some(Self::Textus),
            "numerus" => Some(Self::Numerus),
            "fractus" => Some(Self::Fractus),
            "bivalens" => Some(Self::Bivalens),
            "octeti" => Some(Self::Octeti),
            "instans<ns>" => Some(Self::InstansNs),
            "lista<textus>" => Some(Self::ListaTextus),
            "valor" => Some(Self::Valor),
            "bytes" => Some(Self::Bytes),
            "lista-valor" => Some(Self::ListaValor),
            "bulk-valor" => Some(Self::BulkValor),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RouteContract {
    opener: OpenerContract,
    result: ResultContract,
}

impl RouteContract {
    fn parse(call: &ManifestCall) -> HostResult<Self> {
        let Some(opener) = OpenerContract::parse(&call.opener) else {
            return Err(HostError::invalid_args(format!(
                "manifest route `{}` declares unsupported opener contract `{}`",
                call.route, call.opener
            )));
        };
        let Some(result) = ResultContract::parse(&call.result) else {
            return Err(HostError::invalid_args(format!(
                "manifest route `{}` declares unsupported result contract `{}`",
                call.route, call.result
            )));
        };
        Ok(Self { opener, result })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderRegistration {
    pub manifest: ProviderManifest,
}

impl ProviderRegistration {
    #[must_use]
    pub fn new(manifest: ProviderManifest) -> Self {
        Self { manifest }
    }
}

pub trait Provider: Send + Sync {
    fn registration(&self) -> &ProviderRegistration;

    /// Dispatch a request to this provider.
    ///
    /// # Errors
    ///
    /// Returns any `HostError` variant that the provider implementation chooses
    /// to produce, such as `E_INTERNAL`, `E_INVALID_ARGS`, or `E_NO_ROUTE`.
    fn dispatch(
        &self,
        request: &RequestFrame,
        context: &DispatchContext,
    ) -> HostResult<ProviderReply>;
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct KernelManifest {
    pub manifest_version: u32,
    pub providers: Vec<ProviderManifest>,
}

pub struct Kernel {
    providers: BTreeMap<String, Arc<dyn Provider>>,
    routes: BTreeSet<String>,
    route_contracts: BTreeMap<String, RouteContract>,
}

impl fmt::Debug for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Kernel")
            .field("providers", &self.providers.keys().collect::<Vec<_>>())
            .field("routes", &self.routes.iter().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl Default for Kernel {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernel {
    #[must_use]
    pub fn new() -> Self {
        Self {
            providers: BTreeMap::new(),
            routes: BTreeSet::new(),
            route_contracts: BTreeMap::new(),
        }
    }

    /// Register a provider with this kernel.
    ///
    /// # Errors
    ///
    /// Returns `E_INVALID_ARGS` if the provider manifest fails validation (e.g.
    /// unsupported version, missing provider name, empty prefixes, no calls,
    /// duplicate prefixes, invalid prefix format, or malformed route contracts).
    ///
    /// Returns `E_INTERNAL` if a provider with the same identity, prefix, or
    /// route is already registered.
    // Arc-by-value is the ownership transfer type for shared providers; the
    // body clones into maps. `&Arc` would only push noise to every caller.
    #[allow(clippy::needless_pass_by_value)]
    pub fn register(&mut self, provider: Arc<dyn Provider>) -> HostResult<()> {
        let registration = provider.registration();
        validate_manifest(&registration.manifest)?;
        let manifest = &registration.manifest;
        if self
            .providers
            .values()
            .any(|registered| registered.registration().manifest.provider == manifest.provider)
        {
            return Err(HostError::internal(format!(
                "duplicate provider identity `{}`",
                manifest.provider
            )));
        }
        for prefix in &manifest.prefixes {
            if self.providers.contains_key(prefix) {
                return Err(HostError::internal(format!(
                    "duplicate provider prefix `{prefix}`"
                )));
            }
        }
        for call in &manifest.calls {
            if self.routes.contains(&call.route) {
                return Err(HostError::internal(format!(
                    "duplicate provider route `{}`",
                    call.route
                )));
            }
        }
        let route_contracts = manifest
            .calls
            .iter()
            .map(|call| Ok((call.route.clone(), RouteContract::parse(call)?)))
            .collect::<HostResult<Vec<_>>>()?;
        for prefix in &manifest.prefixes {
            self.providers.insert(prefix.clone(), Arc::clone(&provider));
        }
        for call in &manifest.calls {
            self.routes.insert(call.route.clone());
        }
        for (route, contract) in route_contracts {
            self.route_contracts.insert(route, contract);
        }
        Ok(())
    }

    /// Dispatch a request to the registered provider that handles the given route.
    ///
    /// # Errors
    ///
    /// Returns `E_CANCELLED` if the cancellation probe signals cancellation.
    ///
    /// Returns `E_NO_ROUTE` if the request route has no provider prefix or no
    /// provider is registered for that prefix, or if the provider manifest does
    /// not export the route.
    ///
    /// Returns `E_INTERNAL` if a route contract is missing from the manifest.
    ///
    /// Returns `E_INVALID_ARGS` if the request opener does not match the
    /// declared opener contract, or if the provider reply does not match the
    /// declared result contract.
    ///
    /// Returns `E_PROVIDER_PANIC` if the provider panics during dispatch.
    pub fn dispatch(
        &self,
        request: &RequestFrame,
        context: &DispatchContext,
    ) -> HostResult<ProviderReply> {
        if context.cancellation.try_is_cancelled()? {
            return Err(HostError::cancelled());
        }
        let Some((prefix, _verb)) = request.route.split_once(':') else {
            return Err(HostError::no_route(format!(
                "route `{}` has no provider prefix",
                request.route
            )));
        };
        let Some(provider) = self.providers.get(prefix) else {
            return Err(HostError::no_route(format!(
                "no provider registered for `{prefix}`"
            )));
        };
        if !self.routes.contains(&request.route) {
            return Err(HostError::no_route(format!(
                "provider manifest does not export `{}`",
                request.route
            )));
        }
        let contract = self
            .route_contracts
            .get(&request.route)
            .copied()
            .ok_or_else(|| {
                HostError::internal(format!(
                    "provider manifest route `{}` has no call contract",
                    request.route
                ))
            })?;
        validate_request_contract(&request.route, request, contract.opener)?;
        let reply = catch_unwind(AssertUnwindSafe(|| provider.dispatch(request, context)))
            .map_err(|_panic| HostError::provider_panic())??;
        validate_reply_contract(&request.route, &reply, contract.result)?;
        Ok(reply)
    }

    #[must_use]
    pub fn supports_route(&self, route: &str) -> bool {
        self.routes.contains(route)
    }

    #[must_use]
    pub fn manifest(&self) -> KernelManifest {
        let mut providers = self
            .providers
            .values()
            .map(|provider| provider.registration().manifest.clone())
            .collect::<Vec<_>>();
        providers.sort_by(|left, right| left.provider.cmp(&right.provider));
        providers.dedup_by(|left, right| left.provider == right.provider);
        KernelManifest {
            manifest_version: 1,
            providers,
        }
    }
}

fn validate_manifest(manifest: &ProviderManifest) -> HostResult<()> {
    if manifest.manifest_version != 1 {
        return Err(HostError::invalid_args(format!(
            "unsupported provider manifest version {}",
            manifest.manifest_version
        )));
    }
    if manifest.provider.is_empty() || manifest.prefixes.is_empty() {
        return Err(HostError::invalid_args(
            "provider manifest requires provider and prefixes",
        ));
    }
    if manifest.calls.is_empty() {
        return Err(HostError::invalid_args(
            "provider manifest requires at least one call",
        ));
    }
    let prefixes = manifest.prefixes.iter().collect::<BTreeSet<_>>();
    if prefixes.len() != manifest.prefixes.len() {
        return Err(HostError::invalid_args("provider prefixes must be unique"));
    }
    for prefix in &manifest.prefixes {
        if !valid_prefix(prefix) {
            return Err(HostError::invalid_args(format!(
                "invalid provider prefix `{prefix}`"
            )));
        }
    }
    let mut routes = BTreeSet::new();
    for call in &manifest.calls {
        RouteContract::parse(call)?;
        let Some((prefix, verb)) = call.route.split_once(':') else {
            return Err(HostError::invalid_args(format!(
                "manifest route `{}` has no prefix",
                call.route
            )));
        };
        if verb.is_empty() || verb.contains(':') || !prefixes.contains(&prefix.to_owned()) {
            return Err(HostError::invalid_args(format!(
                "manifest route `{}` does not match provider prefixes",
                call.route
            )));
        }
        if !routes.insert(&call.route) {
            return Err(HostError::internal(format!(
                "provider manifest repeats route `{}`",
                call.route
            )));
        }
    }
    Ok(())
}

fn validate_reply_contract(
    route: &str,
    reply: &ProviderReply,
    result: ResultContract,
) -> HostResult<()> {
    let valid = match result {
        ResultContract::Vacuum => reply.contents.is_empty(),
        ResultContract::Textus => matches!(
            reply.contents.as_slice(),
            [ProviderContent::Item(Valor::Textus(_))]
        ),
        ResultContract::Numerus => matches!(
            reply.contents.as_slice(),
            [ProviderContent::Item(Valor::Numerus(_))]
        ),
        ResultContract::Fractus => matches!(
            reply.contents.as_slice(),
            [ProviderContent::Item(Valor::Fractus(_))]
        ),
        ResultContract::Bivalens => matches!(
            reply.contents.as_slice(),
            [ProviderContent::Item(Valor::Bivalens(_))]
        ),
        ResultContract::Octeti | ResultContract::Bytes => matches!(
            reply.contents.as_slice(),
            [ProviderContent::Byte(_) | ProviderContent::Item(Valor::Octeti(_))]
        ),
        ResultContract::InstansNs => matches!(
            reply.contents.as_slice(),
            [ProviderContent::Item(Valor::Instans(_))]
        ),
        ResultContract::ListaTextus => reply
            .contents
            .iter()
            .all(|content| matches!(content, ProviderContent::Item(Valor::Textus(_)))),
        ResultContract::Valor => matches!(reply.contents.as_slice(), [ProviderContent::Item(_)]),
        ResultContract::ListaValor => reply
            .contents
            .iter()
            .all(|content| matches!(content, ProviderContent::Item(_))),
        ResultContract::BulkValor => {
            matches!(reply.contents.as_slice(), [ProviderContent::Bulk(_)])
        }
    };
    if valid {
        Ok(())
    } else {
        Err(HostError::internal(format!(
            "provider reply for route `{route}` does not match declared result contract"
        )))
    }
}

fn validate_request_contract(
    route: &str,
    request: &RequestFrame,
    opener: OpenerContract,
) -> HostResult<()> {
    let valid = match opener {
        OpenerContract::Vacuum => is_vacuum_carrier(&request.opener),
        OpenerContract::SponteNumerus => {
            is_vacuum_carrier(&request.opener) || matches!(&request.opener, Valor::Numerus(_))
        }
        OpenerContract::Textus => matches!(&request.opener, Valor::Textus(_)),
        OpenerContract::Numerus => matches!(&request.opener, Valor::Numerus(_)),
        OpenerContract::Octeti => matches!(&request.opener, Valor::Octeti(_)),
        OpenerContract::ListaTextus => matches!(
            &request.opener,
            Valor::Lista(items) if items.iter().all(|item| matches!(item, Valor::Textus(_)))
        ),
        OpenerContract::ListaNumerus => matches!(
            &request.opener,
            Valor::Lista(items) if items.iter().all(|item| matches!(item, Valor::Numerus(_)))
        ),
        OpenerContract::ListaValor => matches!(&request.opener, Valor::Lista(_)),
        OpenerContract::Valor => true,
    };
    if valid {
        Ok(())
    } else {
        Err(HostError::invalid_args(format!(
            "request opener for route `{route}` does not match declared opener contract"
        )))
    }
}

fn is_vacuum_carrier(value: &Valor) -> bool {
    matches!(value, Valor::Nihil) || matches!(value, Valor::Tabula(fields) if fields.is_empty())
}

fn valid_prefix(prefix: &str) -> bool {
    let mut chars = prefix.chars();
    matches!(chars.next(), Some(ch) if ch.is_ascii_lowercase())
        && chars.all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || matches!(ch, '_' | '-'))
}

/// Parse a provider manifest from a JSON string.
///
/// # Errors
///
/// Returns `E_INVALID_ARGS` if the JSON cannot be deserialized into a
/// `ProviderManifest`.
pub fn parse_manifest(json: &str) -> HostResult<ProviderManifest> {
    serde_json::from_str(json)
        .map_err(|error| HostError::invalid_args(format!("invalid provider manifest: {error}")))
}

#[cfg(test)]
#[path = "host_kernel_test.rs"]
mod tests;
