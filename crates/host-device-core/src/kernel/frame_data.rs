//! Construction and projection helpers for frame `data` payloads.
//!
//! TARGET: syscall arguments are ordinary `valor` payloads. Generated Faber
//! usually sends empty, scalar, or ordered-list openers; CLI/debug calls may use
//! named tabula fields.

use std::collections::BTreeMap;

use faber::{FromValor, Valor};

use crate::kernel::{HostError, HostResult};

#[must_use]
pub fn empty() -> Valor {
    Valor::Tabula(BTreeMap::new())
}

#[must_use]
pub fn is_empty_tabula(data: &Valor) -> bool {
    matches!(data, Valor::Tabula(tab) if tab.is_empty())
}

#[must_use]
pub fn is_empty(data: &Valor) -> bool {
    matches!(data, Valor::Nihil) || is_empty_tabula(data)
}

pub fn tabula(fields: impl IntoIterator<Item = (impl Into<String>, impl Into<Valor>)>) -> Valor {
    Valor::Tabula(
        fields
            .into_iter()
            .map(|(key, value)| (key.into(), value.into()))
            .collect(),
    )
}

/// Returns the positional argument or named field.
///
/// # Errors
/// Returns an error when the requested argument is absent.
pub fn positional_or_field<'a>(data: &'a Valor, index: usize, key: &str) -> HostResult<&'a Valor> {
    match data {
        Valor::Tabula(tab) => tab
            .get(key)
            .ok_or_else(|| HostError::invalid_args(format!("missing {key}"))),
        Valor::Lista(items) => items.get(index).ok_or_else(|| {
            HostError::invalid_args(format!("missing positional argument {index} ({key})"))
        }),
        value if index == 0 => Ok(value),
        _ => Err(HostError::invalid_args(format!(
            "missing positional argument {index} ({key})"
        ))),
    }
}

/// Returns a named field from tabular frame data.
///
/// # Errors
/// Returns an error when `data` is not tabular or the field is absent.
pub fn field<'a>(data: &'a Valor, key: &str) -> HostResult<&'a Valor> {
    let Valor::Tabula(tab) = data else {
        return Err(HostError::invalid_args("frame data must be a tabula"));
    };
    tab.get(key)
        .ok_or_else(|| HostError::invalid_args(format!("missing {key}")))
}

/// Returns a string positional argument or named field.
///
/// # Errors
/// Returns an error when the argument is absent or is not a string.
pub fn string_arg(data: &Valor, index: usize, key: &str) -> HostResult<String> {
    let value = positional_or_field(data, index, key)?;
    String::from_valor(value)
        .ok_or_else(|| HostError::invalid_args(format!("{key} must be a string")))
}

/// Returns a named string field.
///
/// # Errors
/// Returns an error when the field is absent or is not a string.
pub fn string_field(data: &Valor, key: &str) -> HostResult<String> {
    let value = field(data, key)?;
    String::from_valor(value)
        .ok_or_else(|| HostError::invalid_args(format!("{key} must be a string")))
}

/// Returns an integer positional argument or named field.
///
/// # Errors
/// Returns an error when the argument is absent or is not an integer.
pub fn i64_arg(data: &Valor, index: usize, key: &str) -> HostResult<i64> {
    let value = positional_or_field(data, index, key)?;
    i64::from_valor(value)
        .ok_or_else(|| HostError::invalid_args(format!("{key} must be an integer")))
}

/// Returns a named integer field.
///
/// # Errors
/// Returns an error when the field is absent or is not an integer.
pub fn i64_field(data: &Valor, key: &str) -> HostResult<i64> {
    let value = field(data, key)?;
    i64::from_valor(value)
        .ok_or_else(|| HostError::invalid_args(format!("{key} must be an integer")))
}

/// Returns a string-list positional argument or named field.
///
/// # Errors
/// Returns an error when the argument is absent or is not a string list.
pub fn string_list_arg(data: &Valor, index: usize, key: &str) -> HostResult<Vec<String>> {
    if index == 0
        && let Some(items) = Vec::<String>::from_valor(data)
    {
        return Ok(items);
    }

    let value = positional_or_field(data, index, key)?;
    Vec::<String>::from_valor(value)
        .ok_or_else(|| HostError::invalid_args(format!("{key} must be a list of strings")))
}

/// Returns a named boolean field.
///
/// # Errors
/// Returns an error when the field is absent or is not a boolean.
pub fn bool_field(data: &Valor, key: &str) -> HostResult<bool> {
    let value = field(data, key)?;
    bool::from_valor(value)
        .ok_or_else(|| HostError::invalid_args(format!("{key} must be a boolean")))
}

/// Returns a byte-array positional argument or named field.
///
/// # Errors
/// Returns an error when the argument is absent or cannot be decoded as bytes.
pub fn bytes_arg(data: &Valor, index: usize, key: &str) -> HostResult<Vec<u8>> {
    bytes_from_valor(positional_or_field(data, index, key)?, key)
}

/// Returns a named byte-array field.
///
/// # Errors
/// Returns an error when the field is absent or cannot be decoded as bytes.
pub fn bytes_field(data: &Valor, key: &str) -> HostResult<Vec<u8>> {
    let value = field(data, key)?;
    bytes_from_valor(value, key)
}

fn bytes_from_valor(value: &Valor, key: &str) -> HostResult<Vec<u8>> {
    match value {
        Valor::Lista(items) => items
            .iter()
            .map(|item| match item {
                Valor::Numerus(byte) if (0..=i64::from(u8::MAX)).contains(byte) => Ok(*byte as u8),
                _ => Err(HostError::invalid_args(format!("{key} must contain bytes"))),
            })
            .collect(),
        Valor::Textus(text) => Ok(text.as_bytes().to_vec()),
        _ => Err(HostError::invalid_args(format!(
            "{key} must be a byte array or string"
        ))),
    }
}

#[must_use]
pub fn single_text(value: String) -> Valor {
    Valor::Textus(value)
}

#[must_use]
pub fn single_bool(value: bool) -> Valor {
    Valor::Bivalens(value)
}

#[must_use]
pub fn single_bytes(value: Vec<u8>) -> Valor {
    Valor::Octeti(value)
}

pub fn string_list(items: Vec<String>) -> Valor {
    Valor::Lista(items.into_iter().map(Valor::Textus).collect())
}
