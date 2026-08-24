//! Public `processus` provider.

use faber::Valor;
use host_kernel::{
    DispatchContext, HostError, HostResult, Kernel, Provider, ProviderRegistration, ProviderReply,
    RequestFrame,
};
use std::collections::BTreeMap;
use std::io::{self, Read};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::Duration;

pub struct Processus {
    registration: ProviderRegistration,
}

impl Processus {
    /// Create a new [`Processus`] provider.
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

/// Register the [`Processus`] provider with the kernel.
///
/// # Errors
///
/// Returns [`HostError`] if the provider cannot be created
/// (manifest parsing failure) or if registration fails.
pub fn register(kernel: &mut Kernel) -> HostResult<()> {
    kernel.register(Arc::new(Processus::new()?))
}

#[must_use]
pub fn manifest_json() -> &'static str {
    include_str!("manifest.json")
}

impl Provider for Processus {
    fn registration(&self) -> &ProviderRegistration {
        &self.registration
    }

    fn dispatch(
        &self,
        request: &RequestFrame,
        context: &DispatchContext,
    ) -> HostResult<ProviderReply> {
        match request.route.as_str() {
            "processus:exsequi" | "processus:exsequetur" => execute_shell(&request.opener, context),
            "processus:dimitte" => spawn_detached(&request.opener),
            "processus:lege" => read_env(&request.opener),
            "processus:scribe" => write_env(&request.opener),
            "processus:sedes" => current_dir(),
            "processus:muta" => set_current_dir(&request.opener),
            "processus:identitas" => Ok(ProviderReply::item(Valor::Numerus(i64::from(
                std::process::id(),
            )))),
            "processus:argumenta" => Ok(ProviderReply::list(
                std::env::args().skip(1).map(Valor::Textus),
            )),
            "processus:captura" => capture_process(&request.opener, context),
            other => Err(HostError::no_route(format!(
                "no built-in processus syscall registered for {other}"
            ))),
        }
    }
}

fn execute_shell(opener: &Valor, context: &DispatchContext) -> HostResult<ProviderReply> {
    let command = string_arg(opener, 0, "imperium")?;
    let mut process = Command::new("sh");
    process.arg("-c").arg(command);
    let output = run_command(process, context, "processus:exsequi")?;
    Ok(ProviderReply::item(Valor::Textus(
        String::from_utf8_lossy(&output.stdout).into_owned(),
    )))
}

fn capture_process(opener: &Valor, context: &DispatchContext) -> HostResult<ProviderReply> {
    let args = string_list_arg(opener, 0, "args")?;
    let (program, program_args) = args.split_first().ok_or_else(|| {
        HostError::invalid_args("processus:captura requires a non-empty args list")
    })?;
    let mut process = Command::new(program);
    process.args(program_args);
    let output = run_command(process, context, "processus:captura")?;
    let mut fields = BTreeMap::new();
    fields.insert(
        "status".to_owned(),
        Valor::Numerus(output.status.code().map_or(-1, i64::from)),
    );
    fields.insert(
        "stdout".to_owned(),
        Valor::Textus(String::from_utf8_lossy(&output.stdout).into_owned()),
    );
    fields.insert(
        "stderr".to_owned(),
        Valor::Textus(String::from_utf8_lossy(&output.stderr).into_owned()),
    );
    Ok(ProviderReply::item(Valor::Tabula(fields)))
}

struct CommandOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

fn run_command(
    mut command: Command,
    context: &DispatchContext,
    operation: &str,
) -> HostResult<CommandOutput> {
    if context.cancellation.is_cancelled() {
        return Err(HostError::cancelled());
    }

    configure_process_group(&mut command);
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| HostError::internal(format!("{operation} failed: {error}")))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| pipe_unavailable(&mut child, operation, "stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| pipe_unavailable(&mut child, operation, "stderr"))?;
    let stdout_reader = thread::spawn(move || read_pipe(stdout));
    let stderr_reader = thread::spawn(move || read_pipe(stderr));

    let status = loop {
        if context.cancellation.is_cancelled() {
            let cleanup = abort_command(&mut child, stdout_reader, stderr_reader, operation);
            return match cleanup {
                Ok(()) => Err(HostError::cancelled()),
                Err(error) => Err(HostError::internal(format!(
                    "{operation} cancellation cleanup failed: {}",
                    error.message
                ))),
            };
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => thread::sleep(Duration::from_millis(5)),
            Err(error) => {
                let cleanup = abort_command(&mut child, stdout_reader, stderr_reader, operation);
                return match cleanup {
                    Ok(()) => Err(HostError::internal(format!(
                        "{operation} wait failed: {error}"
                    ))),
                    Err(cleanup_error) => Err(HostError::internal(format!(
                        "{operation} wait failed: {error}; cleanup failed: {}",
                        cleanup_error.message
                    ))),
                };
            }
        }
    };
    let (stdout, stderr) = join_readers(stdout_reader, stderr_reader, operation)?;
    Ok(CommandOutput {
        status,
        stdout,
        stderr,
    })
}

fn read_pipe(mut pipe: impl Read) -> io::Result<Vec<u8>> {
    let mut data = Vec::new();
    pipe.read_to_end(&mut data)?;
    Ok(data)
}

fn join_readers(
    stdout_reader: thread::JoinHandle<io::Result<Vec<u8>>>,
    stderr_reader: thread::JoinHandle<io::Result<Vec<u8>>>,
    operation: &str,
) -> HostResult<(Vec<u8>, Vec<u8>)> {
    let stdout = join_reader(stdout_reader, operation, "stdout")?;
    let stderr = join_reader(stderr_reader, operation, "stderr")?;
    Ok((stdout, stderr))
}

fn join_reader(
    reader: thread::JoinHandle<io::Result<Vec<u8>>>,
    operation: &str,
    stream: &str,
) -> HostResult<Vec<u8>> {
    match reader.join() {
        Ok(Ok(data)) => Ok(data),
        Ok(Err(error)) => Err(HostError::internal(format!(
            "{operation} failed reading {stream}: {error}"
        ))),
        Err(_) => Err(HostError::internal(format!(
            "{operation} {stream} reader panicked"
        ))),
    }
}

fn abort_command(
    child: &mut Child,
    stdout_reader: thread::JoinHandle<io::Result<Vec<u8>>>,
    stderr_reader: thread::JoinHandle<io::Result<Vec<u8>>>,
    operation: &str,
) -> HostResult<()> {
    let termination = terminate_child(child);
    let stdout = join_reader(stdout_reader, operation, "stdout");
    let stderr = join_reader(stderr_reader, operation, "stderr");
    termination?;
    stdout?;
    stderr?;
    Ok(())
}

fn terminate_child(child: &mut Child) -> HostResult<()> {
    let group_termination = terminate_process_group(child);
    if !matches!(&group_termination, Ok(true)) {
        match child.kill() {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(HostError::internal(format!(
                    "failed to terminate child: {error}"
                )));
            }
        }
    }
    let reap = child
        .wait()
        .map(|_| ())
        .map_err(|error| HostError::internal(format!("failed to reap child: {error}")));
    reap?;
    group_termination.map(|_| ())
}

fn configure_process_group(command: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;

        // WHY: shell commands may fork descendants that inherit our output
        // pipes. Keep the operation in its own group so cancellation closes
        // the whole owned process tree before reader threads are joined.
        command.process_group(0);
    }
    #[cfg(not(unix))]
    drop(command);
}

fn terminate_process_group(child: &mut Child) -> HostResult<bool> {
    #[cfg(unix)]
    {
        // SAFETY: `child.id()` returns `u32` which fits in `libc::pid_t` (`i32`);
        // the negation produces a valid negative process group ID.
        #[allow(clippy::cast_possible_wrap)]
        let group = -(child.id() as libc::pid_t);
        let signal_result = unsafe { libc::kill(group, libc::SIGKILL) };
        if signal_result == 0 {
            return Ok(true);
        }
        let error = io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH) {
            return Ok(false);
        }
        Err(HostError::internal(format!(
            "failed to signal child group: {error}"
        )))
    }
    #[cfg(not(unix))]
    {
        drop(child);
        Ok(false)
    }
}

fn pipe_unavailable(child: &mut Child, operation: &str, stream: &str) -> HostError {
    let message = format!("{operation} did not provide a {stream} pipe");
    match terminate_child(child) {
        Ok(()) => HostError::internal(message),
        Err(error) => HostError::internal(format!("{message}; cleanup failed: {error}")),
    }
}

fn spawn_detached(opener: &Valor) -> HostResult<ProviderReply> {
    let args = string_list_arg(opener, 0, "args")?;
    let (program, program_args) = args.split_first().ok_or_else(|| {
        HostError::invalid_args("processus:dimitte requires a non-empty args list")
    })?;
    let child = Command::new(program)
        .args(program_args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| HostError::internal(format!("processus:dimitte failed: {error}")))?;
    Ok(ProviderReply::item(Valor::Numerus(i64::from(child.id()))))
}

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn environment_lock() -> HostResult<MutexGuard<'static, ()>> {
    ENV_LOCK
        .lock()
        .map_err(|_error| HostError::internal("processus environment lock poisoned"))
}

fn read_env(opener: &Valor) -> HostResult<ProviderReply> {
    let name = string_arg(opener, 0, "nomen")?;
    let _guard = environment_lock()?;
    match std::env::var(&name) {
        Ok(value) => Ok(ProviderReply::item(Valor::Textus(value))),
        Err(_) => Err(HostError::internal(format!(
            "processus:lege: environment variable `{name}` is not set"
        ))),
    }
}

fn write_env(opener: &Valor) -> HostResult<ProviderReply> {
    let values = string_list_arg(opener, 0, "args")?;
    let [name, value] = values.as_slice() else {
        return Err(HostError::invalid_args(
            "processus:scribe requires [nomen, valor]",
        ));
    };
    if name.is_empty() || name.contains('=') || name.contains('\0') {
        return Err(HostError::invalid_args(
            "processus:scribe nomen must be non-empty and contain neither `=` nor NUL",
        ));
    }
    let _guard = environment_lock()?;
    std::env::set_var(name, value);
    Ok(ProviderReply::vacuum())
}

fn current_dir() -> HostResult<ProviderReply> {
    let path = std::env::current_dir()
        .map_err(|error| HostError::internal(format!("processus:sedes failed: {error}")))?;
    Ok(ProviderReply::item(Valor::Textus(
        path.to_string_lossy().into_owned(),
    )))
}

fn set_current_dir(opener: &Valor) -> HostResult<ProviderReply> {
    let path = string_arg(opener, 0, "via")?;
    std::env::set_current_dir(&path)
        .map_err(|error| HostError::internal(format!("processus:muta failed: {error}")))?;
    Ok(ProviderReply::vacuum())
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

#[cfg(test)]
#[path = "lib_test.rs"]
mod tests;
