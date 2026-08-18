//! Public `solum` filesystem provider.

use faber::Valor;
use host_kernel::{
    DispatchContext, HostError, HostResult, Kernel, Provider, ProviderRegistration, ProviderReply,
    RequestFrame,
};
use sha2::{Digest, Sha256};
use std::fs::{self, File, FileTimes, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::Arc;
use std::time::SystemTime;

const MAX_RANGE_READ_BYTES: usize = 64 * 1024 * 1024;

pub struct Solum {
    registration: ProviderRegistration,
}

impl Solum {
    /// Create a new [`Solum`] provider.
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

/// Register the [`Solum`] provider with the kernel.
///
/// # Errors
///
/// Returns [`HostError`] if the provider cannot be created
/// (manifest parsing failure) or if registration fails.
pub fn register(kernel: &mut Kernel) -> HostResult<()> {
    kernel.register(Arc::new(Solum::new()?))
}

#[must_use]
pub fn manifest_json() -> &'static str {
    include_str!("manifest.json")
}

impl Provider for Solum {
    fn registration(&self) -> &ProviderRegistration {
        &self.registration
    }

    fn dispatch(
        &self,
        request: &RequestFrame,
        _context: &DispatchContext,
    ) -> HostResult<ProviderReply> {
        match request.route.as_str() {
            "solum:lege" => read_text(&request.opener, request.target.as_deref()),
            "solum:hauri" | "solum:hauriet" => read_bytes(&request.opener),
            "solum:digestio" => digest_file(&request.opener),
            "solum:partem" => read_byte_range(&request.opener),
            "solum:inveni" => find_text_range(&request.opener),
            "solum:carpe" | "solum:carpiet" => read_lines(&request.opener),
            "solum:scribe" | "solum:scribet" => write_text(&request.opener),
            "solum:funde" => write_bytes(&request.opener),
            "solum:appone" | "solum:apponet" => append_text(&request.opener),
            "solum:exstat" | "solum:exstabit" => exists(&request.opener),
            "solum:directoriumne" => is_dir(&request.opener),
            "solum:regularene" => is_file(&request.opener),
            "solum:legibilene" => is_readable(&request.opener),
            "solum:vinculumne" => is_symlink(&request.opener),
            "solum:mensura" => file_size(&request.opener),
            "solum:modum" => set_file_mode(&request.opener),
            "solum:modus" => file_mode(&request.opener),
            "solum:vincula" => create_symlink(&request.opener),
            "solum:dele" | "solum:delet" => delete_file(&request.opener),
            "solum:exscribe" | "solum:exscribet" => copy_file(&request.opener),
            "solum:renomina" | "solum:renominabit" => rename_file(&request.opener),
            "solum:tange" | "solum:tanget" => touch(&request.opener),
            "solum:sequere" | "solum:sequetur" => follow_symlink(&request.opener),
            "solum:crea" | "solum:creabit" => create_dir(&request.opener),
            "solum:enumera" | "solum:enumerabit" => list_dir(&request.opener),
            "solum:amputa" | "solum:amputabit" => remove_dir(&request.opener),
            "solum:domus" => home_dir(),
            "solum:temporarium" => Ok(ProviderReply::item(Valor::Textus(
                std::env::temp_dir().to_string_lossy().into_owned(),
            ))),
            "solum:iunge" => join_paths(&request.opener),
            "solum:parens" => parent_path(&request.opener),
            "solum:nomen" => file_name(&request.opener),
            "solum:suffixum" => extension(&request.opener),
            "solum:absolve" => canonicalize(&request.opener),
            other => Err(HostError::no_route(format!(
                "no built-in solum syscall registered for {other}"
            ))),
        }
    }
}

/// One Item per line — shape for `try_sermo_materialize_lista` through `solum:carpe`.
fn read_file_lines(path: &str, err_label: &str) -> HostResult<ProviderReply> {
    let text = fs::read_to_string(path)
        .map_err(|error| HostError::internal(format!("{err_label}: {error}")))?;
    Ok(ProviderReply::list(
        text.lines().map(|line| Valor::Textus(line.to_owned())),
    ))
}

/// Read file for `solum:lege`.
///
/// The manifest declares a single `textus` result contract for this route.
/// List and byte reads use `solum:carpe` and `solum:hauri` so kernel result
/// validation observes one stable result shape per route.
fn read_text(opener: &Valor, target: Option<&str>) -> HostResult<ProviderReply> {
    let path = string_arg(opener, 0, "via")?;
    let textus = std::any::type_name::<String>();
    if target.is_some_and(|target| target != textus) {
        return Err(HostError::internal(format!(
            "solum:lege target `{}` is not supported; use solum:carpe for lista<textus> or solum:hauri for octeti",
            target.unwrap_or("<unknown>")
        )));
    }
    let text = fs::read_to_string(&path)
        .map_err(|error| HostError::internal(format!("solum:lege failed: {error}")))?;
    Ok(ProviderReply::item(Valor::Textus(text)))
}

fn read_bytes(opener: &Valor) -> HostResult<ProviderReply> {
    let path = string_arg(opener, 0, "via")?;
    let bytes = fs::read(&path)
        .map_err(|error| HostError::internal(format!("solum:hauri failed: {error}")))?;
    Ok(ProviderReply::byte(bytes))
}

/// SHA-256 of the file at `via`, streamed so large artifacts are not loaded whole.
///
/// The hex body is the 64-digit lowercase digest Gradus admission takes as
/// `digestio`. The algorithm name (`sha-256`) stays with the caller.
fn digest_file(opener: &Valor) -> HostResult<ProviderReply> {
    let path = string_arg(opener, 0, "via")?;
    let mut file = File::open(&path)
        .map_err(|error| HostError::internal(format!("solum:digestio open failed: {error}")))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buf)
            .map_err(|error| HostError::internal(format!("solum:digestio read failed: {error}")))?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    Ok(ProviderReply::item(Valor::Textus(hex_lower(
        hasher.finalize().as_slice(),
    ))))
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        write!(out, "{byte:02x}").expect("writing to a String cannot fail");
    }
    out
}

fn read_byte_range(opener: &Valor) -> HostResult<ProviderReply> {
    let path = string_arg(opener, 0, "via")?;
    let start = non_negative_offset(i64_arg(opener, 1, "initium")?, "initium")?;
    let length = bounded_range_length(i64_arg(opener, 2, "longitudo")?, "solum:partem")?;
    let mut file = File::open(&path)
        .map_err(|error| HostError::internal(format!("solum:partem open failed: {error}")))?;
    file.seek(SeekFrom::Start(start))
        .map_err(|error| HostError::internal(format!("solum:partem seek failed: {error}")))?;
    let mut bytes = Vec::with_capacity(length);
    file.take(length as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| HostError::internal(format!("solum:partem read failed: {error}")))?;
    Ok(ProviderReply::byte(bytes))
}

fn find_text_range(opener: &Valor) -> HostResult<ProviderReply> {
    let path = string_arg(opener, 0, "via")?;
    let pattern = string_arg(opener, 1, "exemplar")?;
    let start = non_negative_offset(i64_arg(opener, 2, "initium")?, "initium")?;
    let length = bounded_range_length(i64_arg(opener, 3, "longitudo")?, "solum:inveni")?;
    let mut file = File::open(&path)
        .map_err(|error| HostError::internal(format!("solum:inveni open failed: {error}")))?;
    file.seek(SeekFrom::Start(start))
        .map_err(|error| HostError::internal(format!("solum:inveni seek failed: {error}")))?;
    let mut bytes = Vec::with_capacity(length);
    file.take(length as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| HostError::internal(format!("solum:inveni read failed: {error}")))?;
    let needle = pattern.as_bytes();
    let offset = if needle.is_empty() {
        i64::try_from(start).unwrap_or(i64::MAX)
    } else {
        bytes
            .windows(needle.len())
            .position(|window| window == needle)
            .map_or(-1, |position| {
                // SAFETY: `position` is bounded by `MAX_RANGE_READ_BYTES`,
                // which fits in `i64`.
                #[allow(clippy::cast_possible_wrap)]
                let position = position as i64;
                i64::try_from(start)
                    .unwrap_or(i64::MAX)
                    .saturating_add(position)
            })
    };
    Ok(ProviderReply::item(Valor::Numerus(offset)))
}

fn read_lines(opener: &Valor) -> HostResult<ProviderReply> {
    let path = string_arg(opener, 0, "via")?;
    read_file_lines(&path, "solum:carpe failed")
}

fn write_text(opener: &Valor) -> HostResult<ProviderReply> {
    let path = string_arg(opener, 0, "via")?;
    let data = string_arg(opener, 1, "data")?;
    fs::write(&path, data)
        .map_err(|error| HostError::internal(format!("solum:scribe failed: {error}")))?;
    Ok(ProviderReply::vacuum())
}

fn write_bytes(opener: &Valor) -> HostResult<ProviderReply> {
    let path = string_arg(opener, 0, "via")?;
    let data = bytes_arg(opener, 1, "data")?;
    fs::write(&path, data)
        .map_err(|error| HostError::internal(format!("solum:funde failed: {error}")))?;
    Ok(ProviderReply::vacuum())
}

fn append_text(opener: &Valor) -> HostResult<ProviderReply> {
    let path = string_arg(opener, 0, "via")?;
    let data = string_arg(opener, 1, "data")?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|error| HostError::internal(format!("solum:appone open failed: {error}")))?;
    file.write_all(data.as_bytes())
        .map_err(|error| HostError::internal(format!("solum:appone write failed: {error}")))?;
    Ok(ProviderReply::vacuum())
}

fn exists(opener: &Valor) -> HostResult<ProviderReply> {
    Ok(ProviderReply::item(Valor::Bivalens(
        Path::new(&string_arg(opener, 0, "via")?).exists(),
    )))
}

fn is_dir(opener: &Valor) -> HostResult<ProviderReply> {
    let path = string_arg(opener, 0, "via")?;
    Ok(ProviderReply::item(Valor::Bivalens(
        fs::metadata(path).is_ok_and(|metadata| metadata.is_dir()),
    )))
}

fn is_file(opener: &Valor) -> HostResult<ProviderReply> {
    let path = string_arg(opener, 0, "via")?;
    Ok(ProviderReply::item(Valor::Bivalens(
        fs::metadata(path).is_ok_and(|metadata| metadata.is_file()),
    )))
}

fn is_readable(opener: &Valor) -> HostResult<ProviderReply> {
    let path = string_arg(opener, 0, "via")?;
    Ok(ProviderReply::item(Valor::Bivalens(
        File::open(path).is_ok(),
    )))
}

fn is_symlink(opener: &Valor) -> HostResult<ProviderReply> {
    let path = string_arg(opener, 0, "via")?;
    Ok(ProviderReply::item(Valor::Bivalens(
        fs::symlink_metadata(path).is_ok_and(|metadata| metadata.file_type().is_symlink()),
    )))
}

fn file_size(opener: &Valor) -> HostResult<ProviderReply> {
    let path = string_arg(opener, 0, "via")?;
    let size = fs::metadata(path)
        .map_err(|error| HostError::internal(format!("solum:mensura failed: {error}")))?
        .len();
    // SAFETY: file sizes returned by the OS fit `i64` for practical files;
    // Valor::Numerus uses `i64`.
    #[allow(clippy::cast_possible_wrap)]
    let size = size as i64;
    Ok(ProviderReply::item(Valor::Numerus(size)))
}

fn file_mode(opener: &Valor) -> HostResult<ProviderReply> {
    let path = string_arg(opener, 0, "via")?;
    let mode = fs::metadata(path)
        .map_err(|error| HostError::internal(format!("solum:modus failed: {error}")))?
        .permissions()
        .mode();
    Ok(ProviderReply::item(Valor::Numerus(i64::from(mode))))
}

fn set_file_mode(opener: &Valor) -> HostResult<ProviderReply> {
    let path = string_arg(opener, 0, "via")?;
    let mode = i64_arg(opener, 1, "modus")?;
    if !(0..=0o7777).contains(&mode) {
        return Err(HostError::invalid_args(
            "modus must be between 0 and 0o7777",
        ));
    }
    // SAFETY: `mode` was checked to be in `0..=0o7777`, which fits `u32`.
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let permissions = fs::Permissions::from_mode(mode as u32);
    fs::set_permissions(&path, permissions)
        .map_err(|error| HostError::internal(format!("solum:modum failed: {error}")))?;
    Ok(ProviderReply::vacuum())
}

fn create_symlink(opener: &Valor) -> HostResult<ProviderReply> {
    let source = string_arg(opener, 0, "fons")?;
    let destination = string_arg(opener, 1, "destinatio")?;
    std::os::unix::fs::symlink(&source, &destination)
        .map_err(|error| HostError::internal(format!("solum:vincula failed: {error}")))?;
    Ok(ProviderReply::vacuum())
}

fn delete_file(opener: &Valor) -> HostResult<ProviderReply> {
    let path = string_arg(opener, 0, "via")?;
    // Parity with faber-runtime: missing path is success (idempotent dele).
    match fs::remove_file(&path) {
        Ok(()) => Ok(ProviderReply::vacuum()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(ProviderReply::vacuum()),
        Err(error) => Err(HostError::internal(format!("solum:dele failed: {error}"))),
    }
}

fn copy_file(opener: &Valor) -> HostResult<ProviderReply> {
    let source = string_arg(opener, 0, "fons")?;
    let destination = string_arg(opener, 1, "destinatio")?;
    fs::copy(&source, &destination)
        .map_err(|error| HostError::internal(format!("solum:exscribe failed: {error}")))?;
    Ok(ProviderReply::vacuum())
}

fn rename_file(opener: &Valor) -> HostResult<ProviderReply> {
    let source = string_arg(opener, 0, "fons")?;
    let destination = string_arg(opener, 1, "destinatio")?;
    fs::rename(&source, &destination)
        .map_err(|error| HostError::internal(format!("solum:renomina failed: {error}")))?;
    Ok(ProviderReply::vacuum())
}

fn touch(opener: &Valor) -> HostResult<ProviderReply> {
    let path = string_arg(opener, 0, "via")?;
    touch_path(Path::new(&path))
        .map_err(|error| HostError::internal(format!("solum:tange failed for {path}: {error}")))?;
    Ok(ProviderReply::vacuum())
}

fn touch_path(path: &Path) -> std::io::Result<()> {
    if !path.exists() {
        OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)?;
    }
    let handle = File::open(path)?;
    let now = SystemTime::now();
    let times = FileTimes::new().set_modified(now).set_accessed(now);
    handle.set_times(times)
}

fn follow_symlink(opener: &Valor) -> HostResult<ProviderReply> {
    let path = string_arg(opener, 0, "via")?;
    let target = fs::read_link(&path)
        .map_err(|error| HostError::internal(format!("solum:sequere failed: {error}")))?
        .to_string_lossy()
        .into_owned();
    Ok(ProviderReply::item(Valor::Textus(target)))
}

fn create_dir(opener: &Valor) -> HostResult<ProviderReply> {
    let path = string_arg(opener, 0, "via")?;
    fs::create_dir_all(&path)
        .map_err(|error| HostError::internal(format!("solum:crea failed: {error}")))?;
    Ok(ProviderReply::vacuum())
}

fn list_dir(opener: &Valor) -> HostResult<ProviderReply> {
    let path = string_arg(opener, 0, "via")?;
    let mut entries = fs::read_dir(&path)
        .map_err(|error| HostError::internal(format!("solum:enumera failed: {error}")))?
        .map(|entry| {
            entry
                .map(|entry| Valor::Textus(entry.file_name().to_string_lossy().into_owned()))
                .map_err(|error| {
                    HostError::internal(format!("solum:enumera entry failed: {error}"))
                })
        })
        .collect::<HostResult<Vec<_>>>()?;
    entries.sort_by(|left, right| left_string(left).cmp(left_string(right)));
    Ok(ProviderReply::list(entries))
}

fn left_string(value: &Valor) -> &str {
    match value {
        Valor::Textus(text) => text,
        _ => "",
    }
}

fn remove_dir(opener: &Valor) -> HostResult<ProviderReply> {
    let path = string_arg(opener, 0, "via")?;
    fs::remove_dir_all(&path)
        .map_err(|error| HostError::internal(format!("solum:amputa failed: {error}")))?;
    Ok(ProviderReply::vacuum())
}

fn home_dir() -> HostResult<ProviderReply> {
    let home = home_value(
        std::env::var("HOME").ok(),
        std::env::var("USERPROFILE").ok(),
    )
    .map_err(|message| HostError::internal(format!("solum:domus failed: {message}")))?;
    Ok(ProviderReply::item(Valor::Textus(home)))
}

fn home_value(home: Option<String>, userprofile: Option<String>) -> Result<String, &'static str> {
    home.or(userprofile)
        .ok_or("no home directory environment variable")
}

fn join_paths(opener: &Valor) -> HostResult<ProviderReply> {
    let parts = string_list_arg(opener, 0, "partes")?;
    Ok(ProviderReply::item(Valor::Textus(parts.join("/"))))
}

fn parent_path(opener: &Valor) -> HostResult<ProviderReply> {
    let path = string_arg(opener, 0, "via")?;
    Ok(ProviderReply::item(Valor::Textus(
        Path::new(&path)
            .parent()
            .map(|parent| parent.to_string_lossy().into_owned())
            .unwrap_or_default(),
    )))
}

fn file_name(opener: &Valor) -> HostResult<ProviderReply> {
    let path = string_arg(opener, 0, "via")?;
    Ok(ProviderReply::item(Valor::Textus(
        Path::new(&path)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default(),
    )))
}

fn extension(opener: &Valor) -> HostResult<ProviderReply> {
    let path = string_arg(opener, 0, "via")?;
    Ok(ProviderReply::item(Valor::Textus(
        Path::new(&path)
            .extension()
            .map(|extension| format!(".{}", extension.to_string_lossy()))
            .unwrap_or_default(),
    )))
}

fn canonicalize(opener: &Valor) -> HostResult<ProviderReply> {
    let path = string_arg(opener, 0, "via")?;
    let resolved = fs::canonicalize(&path)
        .map_err(|error| HostError::internal(format!("solum:absolve failed: {error}")))?
        .to_string_lossy()
        .into_owned();
    Ok(ProviderReply::item(Valor::Textus(resolved)))
}

fn non_negative_offset(value: i64, key: &str) -> HostResult<u64> {
    u64::try_from(value).map_err(|_| HostError::invalid_args(format!("{key} must be non-negative")))
}

fn non_negative_length(value: i64, key: &str) -> HostResult<usize> {
    usize::try_from(value)
        .map_err(|_| HostError::invalid_args(format!("{key} must be non-negative")))
}

fn bounded_range_length(value: i64, route: &str) -> HostResult<usize> {
    let length = non_negative_length(value, "longitudo")?;
    if length > MAX_RANGE_READ_BYTES {
        return Err(HostError::invalid_args(format!(
            "{route} longitudo must be at most {MAX_RANGE_READ_BYTES} bytes"
        )));
    }
    Ok(length)
}

fn positional<'a>(value: &'a Valor, index: usize, name: &str) -> HostResult<&'a Valor> {
    match value {
        Valor::Lista(values) => values.get(index).ok_or_else(|| {
            HostError::invalid_args(format!("missing positional argument {index} ({name})"))
        }),
        value if index == 0 => Ok(value),
        _ => Err(HostError::invalid_args(format!(
            "missing positional argument {index} ({name})"
        ))),
    }
}

fn i64_arg(value: &Valor, index: usize, name: &str) -> HostResult<i64> {
    match positional(value, index, name)? {
        Valor::Numerus(number) => Ok(*number),
        _ => Err(HostError::invalid_args(format!(
            "{name} must be an integer"
        ))),
    }
}

fn string_arg(value: &Valor, index: usize, name: &str) -> HostResult<String> {
    match positional(value, index, name)? {
        Valor::Textus(text) | Valor::Instans(text) => Ok(text.clone()),
        _ => Err(HostError::invalid_args(format!("{name} must be a string"))),
    }
}

fn string_list_arg(value: &Valor, index: usize, name: &str) -> HostResult<Vec<String>> {
    let value = if index == 0 {
        match value {
            Valor::Lista(values) if values.iter().all(|item| matches!(item, Valor::Textus(_))) => {
                value
            }
            _ => positional(value, index, name)?,
        }
    } else {
        positional(value, index, name)?
    };
    match value {
        Valor::Lista(values) => values
            .iter()
            .map(|item| match item {
                Valor::Textus(text) | Valor::Instans(text) => Ok(text.clone()),
                _ => Err(HostError::invalid_args(format!(
                    "{name} must contain strings"
                ))),
            })
            .collect(),
        _ => Err(HostError::invalid_args(format!(
            "{name} must be a list of strings"
        ))),
    }
}

fn bytes_arg(value: &Valor, index: usize, name: &str) -> HostResult<Vec<u8>> {
    match positional(value, index, name)? {
        Valor::Octeti(bytes) => Ok(bytes.clone()),
        Valor::Textus(text) => Ok(text.as_bytes().to_vec()),
        Valor::Lista(items) => items
            .iter()
            .map(|item| match item {
                Valor::Numerus(byte) if (0..=i64::from(u8::MAX)).contains(byte) => {
                    // SAFETY: range check `0..=i64::from(u8::MAX)` guarantees
                    // the value fits `u8`.
                    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                    let byte = *byte as u8;
                    Ok(byte)
                }
                _ => Err(HostError::invalid_args(format!(
                    "{name} must contain bytes"
                ))),
            })
            .collect(),
        _ => Err(HostError::invalid_args(format!(
            "{name} must be a byte array or string"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use host_kernel::ProviderContent;

    fn context() -> DispatchContext {
        DispatchContext {
            cancellation: host_kernel::CancellationProbe::new(|| false),
        }
    }

    #[test]
    fn manifest_contains_canonical_routes_and_omits_legacy_aliases() {
        let mut kernel = Kernel::new();
        register(&mut kernel).expect("register solum");
        let calls = &kernel.manifest().providers[0].calls;
        assert_eq!(calls.len(), 46);
        assert!(calls.iter().any(|call| call.route == "solum:modum"));
        assert!(calls.iter().any(|call| call.route == "solum:vincula"));
        assert!(calls.iter().any(|call| call.route == "solum:digestio"));
        assert!(!calls.iter().any(|call| call.route == "solum:fundet"));
        assert!(!calls.iter().any(|call| call.route == "solum:leget"));
    }

    #[test]
    fn mode_and_relative_symlink_operations_preserve_contract() {
        let provider = Solum::new().expect("provider");
        let dir = std::env::temp_dir().join(format!("faber-public-solum-{}", std::process::id()));
        let file = dir.join("payload.txt");
        let link = dir.join("payload-link.txt");
        std::fs::create_dir(&dir).expect("fixture directory");
        std::fs::write(&file, "salve").expect("fixture file");

        let set_mode = provider
            .dispatch(
                &RequestFrame {
                    conversation_id: "mode".into(),
                    route: "solum:modum".into(),
                    opener: Valor::Lista(vec![
                        Valor::Textus(file.to_string_lossy().into_owned()),
                        Valor::Numerus(0o640),
                    ]),
                    target: None,
                },
                &context(),
            )
            .expect("set mode");
        assert!(set_mode.contents.is_empty());
        assert_eq!(
            std::fs::metadata(&file).expect("stat").permissions().mode() & 0o7777,
            0o640
        );

        let link_reply = provider
            .dispatch(
                &RequestFrame {
                    conversation_id: "link".into(),
                    route: "solum:vincula".into(),
                    opener: Valor::Lista(vec![
                        Valor::Textus("payload.txt".into()),
                        Valor::Textus(link.to_string_lossy().into_owned()),
                    ]),
                    target: None,
                },
                &context(),
            )
            .expect("symlink");
        assert!(link_reply.contents.is_empty());
        assert_eq!(
            std::fs::read_link(&link).expect("read link"),
            Path::new("payload.txt")
        );
        assert!(std::fs::symlink_metadata(&link)
            .expect("stat link")
            .file_type()
            .is_symlink());

        std::fs::remove_file(&link).expect("cleanup link");
        std::fs::remove_file(&file).expect("cleanup file");
        std::fs::remove_dir(&dir).expect("cleanup dir");
    }

    #[test]
    fn bounded_partem_and_inveni_return_byte_and_scalar_shapes() {
        let provider = Solum::new().expect("provider");
        let path =
            std::env::temp_dir().join(format!("faber-public-solum-range-{}", std::process::id()));
        std::fs::write(&path, "salve munde").expect("fixture");
        let part = provider
            .dispatch(
                &RequestFrame {
                    conversation_id: "part".into(),
                    route: "solum:partem".into(),
                    opener: Valor::Lista(vec![
                        Valor::Textus(path.to_string_lossy().into_owned()),
                        Valor::Numerus(6),
                        Valor::Numerus(5),
                    ]),
                    target: None,
                },
                &context(),
            )
            .expect("part");
        assert!(
            matches!(part.contents.as_slice(), [ProviderContent::Byte(bytes)] if bytes == b"munde")
        );
        let found = provider
            .dispatch(
                &RequestFrame {
                    conversation_id: "find".into(),
                    route: "solum:inveni".into(),
                    opener: Valor::Lista(vec![
                        Valor::Textus(path.to_string_lossy().into_owned()),
                        Valor::Textus("munde".into()),
                        Valor::Numerus(0),
                        Valor::Numerus(32),
                    ]),
                    target: None,
                },
                &context(),
            )
            .expect("find");
        assert!(matches!(
            found.contents.as_slice(),
            [ProviderContent::Item(Valor::Numerus(6))]
        ));
        std::fs::remove_file(path).expect("cleanup");
    }

    #[test]
    #[allow(clippy::cast_possible_wrap)]
    fn partem_and_inveni_reject_over_limit_ranges_before_allocation() {
        let provider = Solum::new().expect("provider");
        let path =
            std::env::temp_dir().join(format!("faber-public-solum-limit-{}", std::process::id()));
        std::fs::write(&path, b"payload").expect("fixture");
        let path_s = path.to_string_lossy().into_owned();

        let zero_part = provider
            .dispatch(
                &RequestFrame {
                    conversation_id: "part-zero".into(),
                    route: "solum:partem".into(),
                    opener: Valor::Lista(vec![
                        Valor::Textus(path_s.clone()),
                        Valor::Numerus(0),
                        Valor::Numerus(0),
                    ]),
                    target: None,
                },
                &context(),
            )
            .expect("zero-length partem");
        assert!(
            matches!(zero_part.contents.as_slice(), [ProviderContent::Byte(bytes)] if bytes.is_empty())
        );

        for (route, opener) in [
            (
                "solum:partem",
                Valor::Lista(vec![
                    Valor::Textus(path_s.clone()),
                    Valor::Numerus(0),
                    Valor::Numerus(MAX_RANGE_READ_BYTES as i64 + 1),
                ]),
            ),
            (
                "solum:inveni",
                Valor::Lista(vec![
                    Valor::Textus(path_s.clone()),
                    Valor::Textus("pay".into()),
                    Valor::Numerus(0),
                    Valor::Numerus(MAX_RANGE_READ_BYTES as i64 + 1),
                ]),
            ),
        ] {
            let error = provider
                .dispatch(
                    &RequestFrame {
                        conversation_id: format!("{route}-too-long"),
                        route: route.to_owned(),
                        opener,
                        target: None,
                    },
                    &context(),
                )
                .expect_err("over-limit range must fail before allocation");
            assert_eq!(error.code, "E_INVALID_ARGS");
            assert!(error.message.contains(route));
            assert!(error.message.contains(&MAX_RANGE_READ_BYTES.to_string()));
        }

        for (route, opener) in [
            (
                "solum:partem",
                Valor::Lista(vec![
                    Valor::Textus(path_s.clone()),
                    Valor::Numerus(0),
                    Valor::Numerus(-1),
                ]),
            ),
            (
                "solum:inveni",
                Valor::Lista(vec![
                    Valor::Textus(path_s),
                    Valor::Textus("pay".into()),
                    Valor::Numerus(0),
                    Valor::Numerus(-1),
                ]),
            ),
        ] {
            let error = provider
                .dispatch(
                    &RequestFrame {
                        conversation_id: format!("{route}-negative"),
                        route: route.to_owned(),
                        opener,
                        target: None,
                    },
                    &context(),
                )
                .expect_err("negative range length must remain invalid");
            assert_eq!(error.code, "E_INVALID_ARGS");
            assert!(error.message.contains("longitudo"));
            assert!(error.message.contains("non-negative"));
        }

        std::fs::remove_file(path).expect("cleanup");
    }

    #[test]
    fn lege_is_textus_only_and_rejects_list_or_byte_targets() {
        let provider = Solum::new().expect("provider");
        let path =
            std::env::temp_dir().join(format!("faber-public-solum-lege-{}", std::process::id()));
        std::fs::write(&path, "prima\nsecunda\n").expect("fixture");
        let path_s = path.to_string_lossy().into_owned();

        let text = provider
            .dispatch(
                &RequestFrame {
                    conversation_id: "lege-text".into(),
                    route: "solum:lege".into(),
                    opener: Valor::Textus(path_s.clone()),
                    target: Some(std::any::type_name::<String>().to_owned()),
                },
                &context(),
            )
            .expect("lege text");
        assert!(matches!(
            text.contents.as_slice(),
            [ProviderContent::Item(Valor::Textus(s))] if s == "prima\nsecunda\n"
        ));

        for target in [
            std::any::type_name::<Vec<String>>(),
            std::any::type_name::<Vec<u8>>(),
        ] {
            let error = provider
                .dispatch(
                    &RequestFrame {
                        conversation_id: "lege-target".into(),
                        route: "solum:lege".into(),
                        opener: Valor::Textus(path_s.clone()),
                        target: Some(target.to_owned()),
                    },
                    &context(),
                )
                .expect_err("non-text solum:lege target must not bypass manifest contract");
            assert_eq!(error.code, "E_INTERNAL");
            assert!(error.message.contains("solum:lege target"));
            assert!(error.message.contains("solum:carpe"));
            assert!(error.message.contains("solum:hauri"));
        }
        std::fs::remove_file(&path).expect("cleanup");
    }

    #[test]
    fn inveni_empty_pattern_is_found_at_start() {
        let provider = Solum::new().expect("provider");
        let path =
            std::env::temp_dir().join(format!("faber-public-solum-empty-{}", std::process::id()));
        std::fs::write(&path, b"payload").expect("fixture");
        let found = provider
            .dispatch(
                &RequestFrame {
                    conversation_id: "empty".into(),
                    route: "solum:inveni".into(),
                    opener: Valor::Lista(vec![
                        Valor::Textus(path.to_string_lossy().into_owned()),
                        Valor::Textus(String::new()),
                        Valor::Numerus(3),
                        Valor::Numerus(8),
                    ]),
                    target: None,
                },
                &context(),
            )
            .expect("empty inveni");
        assert!(matches!(
            found.contents.as_slice(),
            [ProviderContent::Item(Valor::Numerus(3))]
        ));
        std::fs::remove_file(&path).expect("cleanup");
    }

    #[test]
    fn dele_missing_path_is_success() {
        let provider = Solum::new().expect("provider");
        let missing =
            std::env::temp_dir().join(format!("faber-public-solum-missing-{}", std::process::id()));
        assert!(!missing.exists());
        let reply = provider
            .dispatch(
                &RequestFrame {
                    conversation_id: "dele-missing".into(),
                    route: "solum:dele".into(),
                    opener: Valor::Textus(missing.to_string_lossy().into_owned()),
                    target: None,
                },
                &context(),
            )
            .expect("dele missing");
        assert!(reply.contents.is_empty());
    }

    #[test]
    fn tange_existing_socket_returns_internal_error_instead_of_success() {
        use std::os::unix::net::UnixListener;

        let provider = Solum::new().expect("provider");
        let path =
            std::env::temp_dir().join(format!("faber-public-solum-socket-{}", std::process::id()));
        let listener = UnixListener::bind(&path).expect("bind socket fixture");
        let path_s = path.to_string_lossy().into_owned();
        let error = provider
            .dispatch(
                &RequestFrame {
                    conversation_id: "tange-socket".into(),
                    route: "solum:tange".into(),
                    opener: Valor::Textus(path_s.clone()),
                    target: None,
                },
                &context(),
            )
            .expect_err("touching an unopenable existing path must fail");
        assert_eq!(error.code, "E_INTERNAL");
        assert!(error.message.contains("solum:tange"));
        assert!(error.message.contains(&path_s));

        drop(listener);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn home_value_prefers_home_then_userprofile_and_errors_without_either() {
        assert_eq!(
            home_value(Some("/home/faber".into()), Some("C:\\Users\\faber".into())),
            Ok("/home/faber".into())
        );
        assert_eq!(
            home_value(None, Some("C:\\Users\\faber".into())),
            Ok("C:\\Users\\faber".into())
        );
        assert_eq!(
            home_value(None, None),
            Err("no home directory environment variable")
        );
    }

    // FIPS 180-4 SHA-256("abc") — the pinned file-digest oracle.
    const FIPS_ABC_SHA256: &str =
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    #[test]
    fn digestio_of_known_file_matches_fips_180_4_abc() {
        let provider = Solum::new().expect("provider");
        let path = std::env::temp_dir().join(format!(
            "faber-public-solum-digestio-{}",
            std::process::id()
        ));
        std::fs::write(&path, b"abc").expect("fixture");
        let reply = provider
            .dispatch(
                &RequestFrame {
                    conversation_id: "digestio-abc".into(),
                    route: "solum:digestio".into(),
                    opener: Valor::Textus(path.to_string_lossy().into_owned()),
                    target: None,
                },
                &context(),
            )
            .expect("solum:digestio");
        assert!(
            matches!(
                reply.contents.as_slice(),
                [ProviderContent::Item(Valor::Textus(hex))] if hex == FIPS_ABC_SHA256
            ),
            "solum:digestio must return the pinned SHA-256 hex, got {reply:?}"
        );
        std::fs::remove_file(path).expect("cleanup");
    }

    #[test]
    fn digestio_of_empty_file_matches_fips_180_4_empty() {
        let provider = Solum::new().expect("provider");
        let path = std::env::temp_dir().join(format!(
            "faber-public-solum-digestio-empty-{}",
            std::process::id()
        ));
        std::fs::write(&path, b"").expect("fixture");
        let reply = provider
            .dispatch(
                &RequestFrame {
                    conversation_id: "digestio-empty".into(),
                    route: "solum:digestio".into(),
                    opener: Valor::Textus(path.to_string_lossy().into_owned()),
                    target: None,
                },
                &context(),
            )
            .expect("solum:digestio empty");
        assert!(matches!(
            reply.contents.as_slice(),
            [ProviderContent::Item(Valor::Textus(hex))]
                if hex == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ));
        std::fs::remove_file(path).expect("cleanup");
    }

    #[test]
    fn digestio_missing_path_is_internal_error() {
        let provider = Solum::new().expect("provider");
        let missing = std::env::temp_dir().join(format!(
            "faber-public-solum-digestio-missing-{}",
            std::process::id()
        ));
        assert!(!missing.exists());
        let error = provider
            .dispatch(
                &RequestFrame {
                    conversation_id: "digestio-missing".into(),
                    route: "solum:digestio".into(),
                    opener: Valor::Textus(missing.to_string_lossy().into_owned()),
                    target: None,
                },
                &context(),
            )
            .expect_err("missing file must fail");
        assert_eq!(error.code, "E_INTERNAL");
        assert!(error.message.contains("solum:digestio"));
    }

    #[test]
    fn exstat_nonexistent_path_returns_false() {
        let provider = Solum::new().expect("provider");
        let dir = tempfile::tempdir().expect("temp dir");
        let missing = dir.path().join("solum-nonexistent");
        assert!(!missing.exists());
        let reply = provider
            .dispatch(
                &RequestFrame {
                    conversation_id: "exstat-missing".into(),
                    route: "solum:exstat".into(),
                    opener: Valor::Textus(missing.to_string_lossy().into_owned()),
                    target: None,
                },
                &context(),
            )
            .expect("exstat missing");
        assert!(matches!(
            reply.contents.as_slice(),
            [ProviderContent::Item(Valor::Bivalens(false))]
        ));
    }

    #[test]
    fn crea_existing_directory_is_idempotent() {
        let provider = Solum::new().expect("provider");
        let dir = tempfile::tempdir().expect("temp dir");
        let existing = dir.path().join("solum-crea-exist");
        std::fs::create_dir(&existing).expect("first create");
        let reply = provider
            .dispatch(
                &RequestFrame {
                    conversation_id: "crea-existing".into(),
                    route: "solum:crea".into(),
                    opener: Valor::Textus(existing.to_string_lossy().into_owned()),
                    target: None,
                },
                &context(),
            )
            .expect("crea existing");
        assert!(reply.contents.is_empty());
        assert!(existing.exists());
    }

    #[test]
    fn regula_rejects_non_existent_path() {
        let provider = Solum::new().expect("provider");
        let dir = tempfile::tempdir().expect("temp dir");
        let missing = dir.path().join("solum-regula-missing");
        let reply = provider
            .dispatch(
                &RequestFrame {
                    conversation_id: "regula-missing".into(),
                    route: "solum:regularene".into(),
                    opener: Valor::Textus(missing.to_string_lossy().into_owned()),
                    target: None,
                },
                &context(),
            )
            .expect("regula missing");
        assert!(matches!(
            reply.contents.as_slice(),
            [ProviderContent::Item(Valor::Bivalens(false))]
        ));
    }
}
