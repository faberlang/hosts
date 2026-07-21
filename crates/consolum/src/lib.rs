//! Public `consolum` provider.

use faber::Valor;
use host_kernel::{
    DispatchContext, HostError, HostResult, Kernel, Provider, ProviderRegistration, ProviderReply,
    RequestFrame,
};
use std::io::{self, IsTerminal, Read, Write};
#[cfg(unix)]
use std::os::fd::{AsRawFd, RawFd};
use std::sync::{Arc, Mutex};

const MAX_STDIN_READ_BYTES: usize = 1024 * 1024;

pub struct Consolum {
    registration: ProviderRegistration,
    line_reader: Option<Mutex<Box<dyn Read + Send>>>,
}

impl Consolum {
    /// Create a new [`Consolum`] provider.
    ///
    /// # Errors
    ///
    /// Returns [`HostError`] if the embedded manifest JSON cannot be parsed.
    pub fn new() -> HostResult<Self> {
        Ok(Self {
            registration: ProviderRegistration::new(host_kernel::parse_manifest(manifest_json())?),
            line_reader: None,
        })
    }

    #[doc(hidden)]
    pub fn with_line_reader_for_tests<R>(reader: R) -> HostResult<Self>
    where
        R: Read + Send + 'static,
    {
        Ok(Self {
            registration: ProviderRegistration::new(host_kernel::parse_manifest(manifest_json())?),
            line_reader: Some(Mutex::new(Box::new(reader))),
        })
    }
}

/// Register the [`Consolum`] provider with the kernel.
///
/// # Errors
///
/// Returns [`HostError`] if the provider cannot be created
/// (manifest parsing failure) or if registration fails.
pub fn register(kernel: &mut Kernel) -> HostResult<()> {
    kernel.register(Arc::new(Consolum::new()?))
}

#[must_use]
pub fn manifest_json() -> &'static str {
    include_str!("manifest.json")
}

impl Provider for Consolum {
    fn registration(&self) -> &ProviderRegistration {
        &self.registration
    }

    fn dispatch(
        &self,
        request: &RequestFrame,
        context: &DispatchContext,
    ) -> HostResult<ProviderReply> {
        match request.route.as_str() {
            "consolum:hauri" | "consolum:hauriet" => read_stdin(&request.opener, context),
            "consolum:lege" | "consolum:leget" => self.read_line(context),
            "consolum:funde" => write_stdout_bytes(&request.opener, context),
            "consolum:scribe" | "consolum:scribet" => write_stdout_line(&request.opener, context),
            "consolum:dic" | "consolum:dicet" => write_stdout(&request.opener, context),
            "consolum:mone" | "consolum:monet" | "consolum:vide" | "consolum:videbit" => {
                write_stderr_line(&request.opener, context)
            }
            "consolum:audit" => Ok(ProviderReply::item(Valor::Bivalens(
                io::stdin().is_terminal(),
            ))),
            "consolum:loquitur" => Ok(ProviderReply::item(Valor::Bivalens(
                io::stdout().is_terminal(),
            ))),
            "consolum:admonet" => Ok(ProviderReply::item(Valor::Bivalens(
                io::stderr().is_terminal(),
            ))),
            other => Err(HostError::no_route(format!(
                "no built-in consolum syscall registered for {other}"
            ))),
        }
    }
}

fn read_stdin(opener: &Valor, context: &DispatchContext) -> HostResult<ProviderReply> {
    let magnitude = bounded_non_negative_len(
        i64_arg(opener, 0, "magnitudo")?,
        MAX_STDIN_READ_BYTES,
        "consolum:hauri",
        "magnitudo",
    )?;
    ensure_active(context)?;
    if magnitude == 0 {
        return Ok(ProviderReply::byte(Vec::new()));
    }
    let mut buffer = vec![0_u8; magnitude];
    let mut stdin = io::stdin().lock();
    #[cfg(unix)]
    wait_for_fd(
        stdin.as_raw_fd(),
        libc::POLLIN as libc::c_short,
        context,
        "consolum:hauri",
    )?;
    let bytes_read = stdin
        .read(&mut buffer)
        .map_err(|error| HostError::internal(format!("failed to read stdin: {error}")))?;
    ensure_active(context)?;
    buffer.truncate(bytes_read);
    Ok(ProviderReply::byte(buffer))
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

impl Consolum {
    fn read_line(&self, context: &DispatchContext) -> HostResult<ProviderReply> {
        if let Some(reader) = &self.line_reader {
            let mut reader = reader
                .lock()
                .map_err(|_error| HostError::internal("consolum line reader lock poisoned"))?;
            return read_line_from(&mut **reader, context, || Ok(()));
        }

        let mut stdin = io::stdin().lock();
        #[cfg(unix)]
        let stdin_fd = stdin.as_raw_fd();
        #[cfg(unix)]
        return read_line_from(&mut stdin, context, || {
            wait_for_fd(
                stdin_fd,
                libc::POLLIN as libc::c_short,
                context,
                "consolum:lege",
            )
        });
        #[cfg(not(unix))]
        read_line_from(&mut stdin, context, || Ok(()))
    }
}

fn read_line_from<R, W>(
    reader: &mut R,
    context: &DispatchContext,
    mut wait_until_readable: W,
) -> HostResult<ProviderReply>
where
    R: Read + ?Sized,
    W: FnMut() -> HostResult<()>,
{
    let mut bytes = Vec::new();
    loop {
        ensure_active(context)?;
        wait_until_readable()?;
        let mut byte = [0_u8; 1];
        let count = reader
            .read(&mut byte)
            .map_err(|error| HostError::internal(format!("failed to read stdin line: {error}")))?;
        if count == 0 {
            break;
        }
        bytes.push(byte[0]);
        if byte[0] == b'\n' {
            break;
        }
    }
    ensure_active(context)?;
    let mut line = String::from_utf8(bytes)
        .map_err(|error| HostError::internal(format!("failed to decode stdin line: {error}")))?;
    trim_line_ending(&mut line);
    Ok(ProviderReply::item(Valor::Textus(line)))
}

fn write_stdout_bytes(opener: &Valor, context: &DispatchContext) -> HostResult<ProviderReply> {
    let data = bytes_arg(opener, 0, "data")?;
    let mut stdout = io::stdout().lock();
    write_stream(&mut stdout, &data, context, "consolum:funde")?;
    Ok(ProviderReply::vacuum())
}

fn write_stdout_line(opener: &Valor, context: &DispatchContext) -> HostResult<ProviderReply> {
    let message = string_arg(opener, 0, "msg")?;
    let data = format!("{message}\n");
    let mut stdout = io::stdout().lock();
    write_stream(&mut stdout, data.as_bytes(), context, "consolum:scribe")?;
    Ok(ProviderReply::vacuum())
}

fn write_stdout(opener: &Valor, context: &DispatchContext) -> HostResult<ProviderReply> {
    let message = string_arg(opener, 0, "msg")?;
    let mut stdout = io::stdout().lock();
    write_stream(&mut stdout, message.as_bytes(), context, "consolum:dic")?;
    Ok(ProviderReply::vacuum())
}

fn write_stderr_line(opener: &Valor, context: &DispatchContext) -> HostResult<ProviderReply> {
    let message = string_arg(opener, 0, "msg")?;
    let data = format!("{message}\n");
    let mut stderr = io::stderr().lock();
    write_stream(&mut stderr, data.as_bytes(), context, "consolum:mone")?;
    Ok(ProviderReply::vacuum())
}

fn ensure_active(context: &DispatchContext) -> HostResult<()> {
    if context.cancellation.is_cancelled() {
        return Err(HostError::cancelled());
    }
    Ok(())
}

#[cfg(unix)]
const IO_POLL_TIMEOUT_MS: libc::c_int = 5;

#[cfg(unix)]
fn wait_for_fd(
    fd: RawFd,
    events: libc::c_short,
    context: &DispatchContext,
    operation: &str,
) -> HostResult<()> {
    loop {
        ensure_active(context)?;
        let mut descriptor = libc::pollfd {
            fd,
            events,
            revents: 0,
        };
        // SAFETY: `descriptor` is a valid one-element pollfd array for the
        // borrowed file descriptor, and libc writes only within that array.
        let result = unsafe { libc::poll(&raw mut descriptor, 1, IO_POLL_TIMEOUT_MS) };
        if result > 0 {
            let error_events = libc::POLLNVAL as libc::c_short;
            if descriptor.revents & error_events != 0 {
                return Err(HostError::internal(format!("{operation} fd is invalid")));
            }
            let ready_events =
                events | libc::POLLERR as libc::c_short | libc::POLLHUP as libc::c_short;
            if descriptor.revents & ready_events != 0 {
                return Ok(());
            }
            continue;
        }
        if result == 0 {
            continue;
        }
        let error = io::Error::last_os_error();
        if error.kind() == io::ErrorKind::Interrupted {
            continue;
        }
        return Err(HostError::internal(format!(
            "{operation} poll failed: {error}"
        )));
    }
}

#[cfg(unix)]
fn write_stream<W: Write + AsRawFd>(
    writer: &mut W,
    data: &[u8],
    context: &DispatchContext,
    operation: &str,
) -> HostResult<()> {
    ensure_active(context)?;
    writer
        .flush()
        .map_err(|error| HostError::internal(format!("{operation} flush failed: {error}")))?;
    write_fd_cancellable(writer.as_raw_fd(), data, context, operation)
}

#[cfg(not(unix))]
fn write_stream<W: Write>(
    writer: &mut W,
    data: &[u8],
    context: &DispatchContext,
    operation: &str,
) -> HostResult<()> {
    ensure_active(context)?;
    writer
        .write_all(data)
        .and_then(|()| writer.flush())
        .map_err(|error| HostError::internal(format!("{operation} write failed: {error}")))?;
    ensure_active(context)
}

#[cfg(unix)]
fn write_fd_cancellable(
    fd: RawFd,
    data: &[u8],
    context: &DispatchContext,
    operation: &str,
) -> HostResult<()> {
    ensure_active(context)?;
    // SAFETY: the caller keeps the resource owning `fd` alive for this call.
    let original_flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if original_flags < 0 {
        return Err(HostError::internal(format!(
            "{operation} could not inspect fd flags: {}",
            io::Error::last_os_error()
        )));
    }
    let nonblocking_flags = original_flags | libc::O_NONBLOCK;
    // SAFETY: `fd` is the same live descriptor whose flags were just read.
    if unsafe { libc::fcntl(fd, libc::F_SETFL, nonblocking_flags) } < 0 {
        return Err(HostError::internal(format!(
            "{operation} could not enable nonblocking writes: {}",
            io::Error::last_os_error()
        )));
    }
    let result = write_fd_nonblocking(fd, data, context, operation);
    // SAFETY: `fd` remains owned by the caller while the write operation is
    // being finalized and its original flags are restored.
    let restore = unsafe { libc::fcntl(fd, libc::F_SETFL, original_flags) };
    if restore < 0 {
        return Err(HostError::internal(format!(
            "{operation} could not restore fd flags: {}",
            io::Error::last_os_error()
        )));
    }
    result
}

#[cfg(unix)]
fn write_fd_nonblocking(
    fd: RawFd,
    data: &[u8],
    context: &DispatchContext,
    operation: &str,
) -> HostResult<()> {
    let mut offset = 0;
    while offset < data.len() {
        wait_for_fd(fd, libc::POLLOUT as libc::c_short, context, operation)?;
        ensure_active(context)?;
        let remaining = &data[offset..];
        // SAFETY: `remaining` is a live slice for the duration of this call,
        // and `fd` remains owned by the caller.
        let written = unsafe { libc::write(fd, remaining.as_ptr().cast(), remaining.len()) };
        // SAFETY: `libc::write` returns the positive byte count when `written > 0`.
        if written > 0 {
            #[allow(clippy::cast_sign_loss)]
            let written = written as usize;
            offset += written;
            continue;
        }
        if written == 0 {
            return Err(HostError::internal(format!(
                "{operation} write returned zero"
            )));
        }
        let error = io::Error::last_os_error();
        if error.kind() == io::ErrorKind::Interrupted || error.kind() == io::ErrorKind::WouldBlock {
            continue;
        }
        return Err(HostError::internal(format!(
            "{operation} write failed: {error}"
        )));
    }
    Ok(())
}

fn trim_line_ending(line: &mut String) {
    if line.ends_with('\n') {
        line.pop();
        if line.ends_with('\r') {
            line.pop();
        }
    }
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
#[path = "consolum_test.rs"]
mod tests;
