//! Bounded native host dispatch for public Faber providers.

use faber::{Cancellation, DispatchError, FrameStatus, HostDispatch, ResponseSender, SermoRequest};
use host_kernel::{CancellationProbe, DispatchContext, Kernel, ProviderContent, RequestFrame};
use std::collections::BTreeMap;
use std::io;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

pub const DEFAULT_WORKERS: usize = 16;
pub const DEFAULT_QUEUE_CAPACITY: usize = 256;

type JobReceiver = Arc<Mutex<Receiver<NativeJob>>>;

struct NativeJob {
    id: u64,
    request: SermoRequest,
    responses: ResponseSender,
    cancellation: Cancellation,
}

struct NativeState {
    kernel: Arc<Kernel>,
    queue: Mutex<Option<SyncSender<NativeJob>>>,
    shutting_down: AtomicBool,
    public_handles: AtomicUsize,
    next_job_id: AtomicU64,
    active_jobs: Mutex<BTreeMap<u64, Cancellation>>,
    workers: Mutex<Vec<JoinHandle<()>>>,
}

pub struct NativeHost {
    state: Arc<NativeState>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeHostConstructionError {
    pub issue: &'static str,
    pub message: String,
}

impl NativeHostConstructionError {
    fn new(issue: &'static str, message: impl Into<String>) -> Self {
        Self {
            issue,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for NativeHostConstructionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for NativeHostConstructionError {}

impl Clone for NativeHost {
    fn clone(&self) -> Self {
        self.state.public_handles.fetch_add(1, Ordering::SeqCst);
        Self {
            state: Arc::clone(&self.state),
        }
    }
}

impl NativeHost {
    /// Panics if construction fails; prefer `try_new` when the caller needs to
    /// surface startup failures as diagnostics instead of aborting.
    #[must_use]
    pub fn new(kernel: Kernel) -> Self {
        Self::with_limits(kernel, DEFAULT_WORKERS, DEFAULT_QUEUE_CAPACITY)
    }

    /// Panics if construction fails; prefer `try_with_limits` for fallible
    /// startup paths and tests.
    ///
    /// # Panics
    ///
    /// Panics if `try_with_limits` returns an error, which occurs when
    /// `workers` is zero or when worker threads cannot be spawned.
    #[must_use]
    pub fn with_limits(kernel: Kernel, workers: usize, queue_capacity: usize) -> Self {
        Self::try_with_limits(kernel, workers, queue_capacity)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// Constructs a [`NativeHost`] with default worker and queue limits.
    ///
    /// # Errors
    ///
    /// Returns [`NativeHostConstructionError`] if worker threads cannot be
    /// spawned.
    pub fn try_new(kernel: Kernel) -> Result<Self, NativeHostConstructionError> {
        Self::try_with_limits(kernel, DEFAULT_WORKERS, DEFAULT_QUEUE_CAPACITY)
    }

    /// Constructs a [`NativeHost`] with the given number of workers and queue
    /// capacity.
    ///
    /// # Errors
    ///
    /// Returns [`NativeHostConstructionError`] if `workers` is zero or if
    /// worker threads cannot be spawned.
    pub fn try_with_limits(
        kernel: Kernel,
        workers: usize,
        queue_capacity: usize,
    ) -> Result<Self, NativeHostConstructionError> {
        build_with_worker_spawner(kernel, workers, queue_capacity, spawn_worker)
    }

    pub fn shutdown(&self) {
        begin_shutdown(&self.state);
        join_workers(&self.state);
    }
}

fn build_with_worker_spawner(
    kernel: Kernel,
    workers: usize,
    queue_capacity: usize,
    mut spawn: impl FnMut(String, Arc<NativeState>, JobReceiver) -> io::Result<JoinHandle<()>>,
) -> Result<NativeHost, NativeHostConstructionError> {
    if workers == 0 {
        return Err(NativeHostConstructionError::new(
            "native_host_zero_workers",
            "native host requires at least one worker",
        ));
    }
    let (queue, receiver) = sync_channel(queue_capacity);
    let state = Arc::new(NativeState {
        kernel: Arc::new(kernel),
        queue: Mutex::new(Some(queue)),
        shutting_down: AtomicBool::new(false),
        public_handles: AtomicUsize::new(1),
        next_job_id: AtomicU64::new(0),
        active_jobs: Mutex::new(BTreeMap::new()),
        workers: Mutex::new(Vec::with_capacity(workers)),
    });
    let receiver = Arc::new(Mutex::new(receiver));
    for index in 0..workers {
        let name = format!("faber-native-host-{index}");
        let worker_state = Arc::clone(&state);
        let worker_receiver = Arc::clone(&receiver);
        let handle = match spawn(name, worker_state, worker_receiver) {
            Ok(handle) => handle,
            Err(error) => {
                cleanup_partial_workers(&state);
                return Err(NativeHostConstructionError::new(
                    "native_host_worker_spawn_failed",
                    format!("spawn native host worker: {error}"),
                ));
            }
        };
        state
            .workers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(handle);
    }
    Ok(NativeHost { state })
}

fn spawn_worker(
    name: String,
    state: Arc<NativeState>,
    receiver: JobReceiver,
) -> io::Result<JoinHandle<()>> {
    thread::Builder::new()
        .name(name)
        .spawn(move || run_worker(&state, &receiver))
}

impl HostDispatch for NativeHost {
    fn start(
        &self,
        request: SermoRequest,
        responses: ResponseSender,
        cancellation: Cancellation,
    ) -> Result<(), DispatchError> {
        if !self.state.kernel.supports_route(&request.route) {
            return Err(DispatchError::new(
                "host_unsupported_route",
                format!("unsupported native host route `{}`", request.route),
            ));
        }
        let job = NativeJob {
            id: self.state.next_job_id.fetch_add(1, Ordering::SeqCst),
            request,
            responses,
            cancellation,
        };
        enqueue_job(&self.state, job)
    }
}

impl Drop for NativeHost {
    fn drop(&mut self) {
        if self.state.public_handles.fetch_sub(1, Ordering::SeqCst) == 1 {
            begin_shutdown(&self.state);
        }
    }
}

fn enqueue_job(state: &NativeState, job: NativeJob) -> Result<(), DispatchError> {
    let queue = state
        .queue
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if state.shutting_down.load(Ordering::SeqCst) {
        return Err(DispatchError::new(
            "host_shutting_down",
            "native host is shutting down",
        ));
    }
    let Some(queue) = queue.as_ref() else {
        return Err(DispatchError::new(
            "host_shutting_down",
            "native host is shutting down",
        ));
    };
    match queue.try_send(job) {
        Ok(()) => Ok(()),
        Err(TrySendError::Full(_)) => Err(DispatchError::new(
            "host_queue_saturated",
            "native host worker queue is saturated",
        )),
        Err(TrySendError::Disconnected(_)) => Err(DispatchError::new(
            "host_shutting_down",
            "native host worker queue is shut down",
        )),
    }
}

fn begin_shutdown(state: &Arc<NativeState>) {
    let mut queue = state
        .queue
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !state.shutting_down.swap(true, Ordering::SeqCst) {
        queue.take();
        drop(queue);
        let active_jobs = state
            .active_jobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for cancellation in active_jobs.values() {
            cancellation.cancel();
        }
    }
}

fn cleanup_partial_workers(state: &Arc<NativeState>) {
    begin_shutdown(state);
    join_workers(state);
}

fn join_workers(state: &NativeState) {
    let handles = std::mem::take(
        &mut *state
            .workers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    );
    for handle in handles {
        let _ = handle.join();
    }
}

fn run_worker(state: &Arc<NativeState>, receiver: &JobReceiver) {
    loop {
        let job = {
            let receiver = receiver
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            receiver.recv()
        };
        let Ok(job) = job else {
            return;
        };
        if state.shutting_down.load(Ordering::SeqCst) || !register_active_job(state, &job) {
            send_error(
                &job.responses,
                "host_shutting_down",
                "native host is shutting down",
            );
        } else {
            let job_id = job.id;
            run_job(state, job);
            unregister_active_job(state, job_id);
        }
    }
}

fn register_active_job(state: &NativeState, job: &NativeJob) -> bool {
    let mut active_jobs = state
        .active_jobs
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if state.shutting_down.load(Ordering::SeqCst) {
        return false;
    }
    active_jobs.insert(job.id, job.cancellation.clone());
    true
}

fn unregister_active_job(state: &NativeState, job_id: u64) {
    state
        .active_jobs
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(&job_id);
}

fn run_job(state: &NativeState, job: NativeJob) {
    if job.cancellation.is_cancelled() {
        let _ = job.responses.cancel();
        return;
    }
    let responses = job.responses.clone();
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        let request = RequestFrame {
            conversation_id: job.request.conversation_id,
            route: job.request.route,
            opener: job.request.opener,
            target: job.request.target.map(str::to_owned),
        };
        let cancellation = job.cancellation.clone();
        let context = DispatchContext {
            cancellation: CancellationProbe::new(move || cancellation.is_cancelled()),
        };
        match state.kernel.dispatch(&request, &context) {
            Ok(reply) => send_reply(reply.contents, &job.responses, &job.cancellation),
            Err(error) => send_host_error(&job.responses, &error),
        }
    }));
    if result.is_err() {
        send_error(&responses, "E_PROVIDER_PANIC", "provider panicked");
    }
}

fn send_reply(
    contents: Vec<ProviderContent>,
    responses: &ResponseSender,
    cancellation: &Cancellation,
) {
    for content in contents {
        if cancellation.is_cancelled() {
            let _ = responses.cancel();
            return;
        }
        let result = match content {
            ProviderContent::Item(data) => responses.item(data),
            ProviderContent::Byte(bytes) => responses.byte(bytes),
            ProviderContent::Bulk(data) => responses.send(FrameStatus::Bulk, data),
        };
        if result.is_err() {
            return;
        }
    }
    if cancellation.is_cancelled() {
        let _ = responses.cancel();
    } else {
        let _ = responses.done();
    }
}

fn send_error(responses: &ResponseSender, code: &str, message: &str) {
    let mut fields = std::collections::BTreeMap::new();
    fields.insert("code".to_owned(), faber::Valor::Textus(code.to_owned()));
    fields.insert(
        "message".to_owned(),
        faber::Valor::Textus(message.to_owned()),
    );
    fields.insert("retryable".to_owned(), faber::Valor::Bivalens(false));
    let _ = responses.send(FrameStatus::Error, faber::Valor::Tabula(fields));
}

fn send_host_error(responses: &ResponseSender, error: &host_kernel::HostError) {
    if error.code == "E_CANCELLED" {
        let _ = responses.cancel();
    } else {
        let _ = responses.send(FrameStatus::Error, error.to_valor());
    }
}

#[cfg(test)]
#[path = "host_native_test.rs"]
mod tests;
