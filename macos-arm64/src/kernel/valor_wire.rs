//! JSON transport encoding for [`faber::Valor`] at frame wire boundaries.
//!
//! WHY: `scrinium.data` is `valor` in the language contract; `serde_json` is one
//! transport encoding layered on top, not the in-memory carrier type.
//!
//! Wire contract (DEFER-051): variants JSON can represent natively keep their
//! plain shape (`Nihil`/`Bivalens`/`Numerus`/`Fractus`/`Textus`/`Lista`/
//! `Tabula`). Variants JSON cannot represent use a tagged single-key envelope:
//!
//! - `Valor::Instans(t)` ↔ `{"$instans": "<rfc3339>"}`
//! - `Valor::Octeti(bytes)` ↔ `{"$octeti": [0, 1, ...]}` (array of byte
//!   numbers, matching the legacy `Octeti` wire shape plus a tag)
//!
//! Collision contract: the `$`-prefixed single-key namespace is **reserved**
//! for wire tags. A `Tabula` consisting of a single `$`-prefixed key is
//! escaped on encode by wrapping it as `{"$tabula": {...}}`, and unwrapped on
//! decode; decoding a single-key `$`-prefixed object with an unknown tag is an
//! error. Multi-key `Tabula` maps may use `$`-prefixed keys freely.

use std::collections::BTreeMap;

use faber::Valor;

/// Wire tag for the `Valor::Instans` envelope.
const INSTANS_TAG: &str = "$instans";
/// Wire tag for the `Valor::Octeti` envelope.
const OCTETI_TAG: &str = "$octeti";
/// Escape tag wrapping a single-key `$`-prefixed `Tabula` so it is not
/// misdecoded as a typed envelope.
const ESCAPE_TAG: &str = "$tabula";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValorWireError(pub String);

impl std::fmt::Display for ValorWireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for ValorWireError {}

pub fn json_to_valor(value: serde_json::Value) -> Result<Valor, ValorWireError> {
    match value {
        serde_json::Value::Null => Ok(Valor::Nihil),
        serde_json::Value::Bool(b) => Ok(Valor::Bivalens(b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(Valor::Numerus(i))
            } else if let Some(f) = n.as_f64() {
                Ok(Valor::Fractus(f))
            } else {
                Err(ValorWireError(format!(
                    "JSON number out of representable range: {n}"
                )))
            }
        }
        serde_json::Value::String(s) => Ok(Valor::Textus(s)),
        serde_json::Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(json_to_valor(item)?);
            }
            Ok(Valor::Lista(out))
        }
        serde_json::Value::Object(map) => decode_object(map),
    }
}

fn decode_object(map: serde_json::Map<String, serde_json::Value>) -> Result<Valor, ValorWireError> {
    if map.len() == 1 {
        return match map.into_iter().next() {
            Some((key, value)) if key.starts_with('$') => decode_tagged(key, value),
            Some((key, value)) => plain_tabula(std::iter::once((key, value))),
            None => Ok(Valor::Tabula(BTreeMap::new())),
        };
    }
    plain_tabula(map)
}

fn decode_tagged(tag: String, value: serde_json::Value) -> Result<Valor, ValorWireError> {
    match tag.as_str() {
        INSTANS_TAG => match value {
            serde_json::Value::String(s) => Ok(Valor::Instans(s)),
            other => Err(ValorWireError(format!(
                "{INSTANS_TAG} envelope must hold a string, got {other}"
            ))),
        },
        OCTETI_TAG => match value {
            serde_json::Value::Array(items) => {
                let mut bytes = Vec::with_capacity(items.len());
                for item in items {
                    let byte = item
                        .as_u64()
                        .and_then(|n| u8::try_from(n).ok())
                        .ok_or_else(|| {
                            ValorWireError(format!(
                                "{OCTETI_TAG} envelope holds a non-byte: {item}"
                            ))
                        })?;
                    bytes.push(byte);
                }
                Ok(Valor::Octeti(bytes))
            }
            other => Err(ValorWireError(format!(
                "{OCTETI_TAG} envelope must hold an array, got {other}"
            ))),
        },
        ESCAPE_TAG => match value {
            serde_json::Value::Object(inner) => plain_tabula(inner),
            other => Err(ValorWireError(format!(
                "{ESCAPE_TAG} escape must hold an object, got {other}"
            ))),
        },
        _ => Err(ValorWireError(format!(
            "unknown valor wire tag {tag:?}: single-key \"$\"-prefixed objects are reserved"
        ))),
    }
}

fn plain_tabula(
    map: impl IntoIterator<Item = (String, serde_json::Value)>,
) -> Result<Valor, ValorWireError> {
    let mut out = BTreeMap::new();
    for (key, value) in map {
        out.insert(key, json_to_valor(value)?);
    }
    Ok(Valor::Tabula(out))
}

pub fn valor_to_json(valor: &Valor) -> Result<serde_json::Value, ValorWireError> {
    match valor {
        Valor::Nihil => Ok(serde_json::Value::Null),
        Valor::Bivalens(b) => Ok(serde_json::Value::Bool(*b)),
        Valor::Numerus(n) => Ok(serde_json::Value::Number((*n).into())),
        Valor::Fractus(f) => {
            if f.is_finite() {
                match serde_json::Number::from_f64(*f) {
                    Some(num) => Ok(serde_json::Value::Number(num)),
                    None => Err(ValorWireError(format!(
                        "fractus {f} cannot be represented as a JSON number"
                    ))),
                }
            } else {
                Err(ValorWireError(format!(
                    "fractus value is NaN or infinite: {f}"
                )))
            }
        }
        Valor::Textus(s) => Ok(serde_json::Value::String(s.clone())),
        Valor::Octeti(bytes) => {
            let mut out = serde_json::Map::new();
            out.insert(
                OCTETI_TAG.to_string(),
                serde_json::Value::Array(
                    bytes
                        .iter()
                        .map(|byte| serde_json::Value::Number(i64::from(*byte).into()))
                        .collect(),
                ),
            );
            Ok(serde_json::Value::Object(out))
        }
        Valor::Lista(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(valor_to_json(item)?);
            }
            Ok(serde_json::Value::Array(out))
        }
        Valor::Tabula(tab) => {
            let mut out = serde_json::Map::new();
            for (key, value) in tab {
                out.insert(key.clone(), valor_to_json(value)?);
            }
            if out.len() == 1 && out.keys().next().is_some_and(|key| key.starts_with('$')) {
                let mut escaped = serde_json::Map::new();
                escaped.insert(ESCAPE_TAG.to_string(), serde_json::Value::Object(out));
                return Ok(serde_json::Value::Object(escaped));
            }
            Ok(serde_json::Value::Object(out))
        }
        Valor::Instans(t) => {
            let mut out = serde_json::Map::new();
            out.insert(
                INSTANS_TAG.to_string(),
                serde_json::Value::String(t.clone()),
            );
            Ok(serde_json::Value::Object(out))
        }
    }
}

pub fn parse_json_object(raw: &str) -> Result<Valor, String> {
    let value: serde_json::Value =
        serde_json::from_str(raw).map_err(|error| format!("invalid JSON payload: {error}"))?;
    match value {
        serde_json::Value::Object(_) => {
            json_to_valor(value).map_err(|error| format!("invalid frame data payload: {error}"))
        }
        _ => Err("call payload must be a JSON object".into()),
    }
}

pub(crate) mod serde_field {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::{json_to_valor, valor_to_json};

    pub fn serialize<S>(valor: &faber::Valor, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let json = valor_to_json(valor).map_err(serde::ser::Error::custom)?;
        json.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<faber::Valor, D::Error>
    where
        D: Deserializer<'de>,
    {
        let json = serde_json::Value::deserialize(deserializer)?;
        json_to_valor(json).map_err(serde::de::Error::custom)
    }
}
