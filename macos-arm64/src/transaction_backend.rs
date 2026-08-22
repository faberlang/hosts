//! Real [`DeviceExecutionBackend`] over a [`DeviceRuntimeSet`] (MD3H-H2;
//! absorbs MD3-S1).
//!
//! Launches run against the owning session; transfers and broadcasts move
//! through [`HostStagedAdapter`] (labeled + timed, T2 §7) as host-staged
//! D2H then H2D copies; barriers `sync` the involved sessions. Staging /
//! publish / retire map onto the frozen transaction contract. No
//! host-coordinator trait extension.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use host_coordinator::bound_plan::LogicalPartitionId;
use host_coordinator::device_identity::PhysicalDeviceId;
use host_coordinator::execution_transaction::{
    BackendError, DeviceExecutionBackend, MirroredDtype, MirroredStorageLayout, OperationEvent,
    OutputRef, ReservationRecord, StagedWrite, TransactionFailure, TransactionOperation,
    TransferDirectionMirror, TransferRef,
};
use host_coordinator::transport::{
    ByteRange, HostStagedAdapter, SourceValue, TransferBudget, TransferError, TransferSpec,
    TransportAdapter, TransportReceipt,
};
use host_coordinator::DeviceHandle;

use crate::device_descriptor::DeviceDataType;
use crate::device_host::{DeviceRuntime, DeviceSession};
use crate::device_runtime_set::DeviceRuntimeSet;
use crate::kernel::{HostError, HostResult};

/// Kernel image + entry the backend launches for every `Launch` operation.
#[derive(Debug, Clone)]
pub struct LaunchProgram {
    image: Vec<u8>,
    entry: String,
}

impl LaunchProgram {
    /// A compiled module image (MSL or PTX) and its launch entry.
    #[must_use]
    pub fn new(image: impl Into<Vec<u8>>, entry: impl Into<String>) -> Self {
        Self {
            image: image.into(),
            entry: entry.into(),
        }
    }

    /// The module image bytes.
    #[must_use]
    pub fn image(&self) -> &[u8] {
        &self.image
    }

    /// The kernel entry name.
    #[must_use]
    pub fn entry(&self) -> &str {
        &self.entry
    }
}

/// Real device-execution backend over one [`DeviceRuntimeSet`].
pub struct DeviceRuntimeBackend {
    set: DeviceRuntimeSet,
    partition_devices: BTreeMap<LogicalPartitionId, PhysicalDeviceId>,
    launch: LaunchProgram,
    modules: BTreeMap<PhysicalDeviceId, DeviceHandle>,
    transport: HostStagedAdapter,
    transfer_timeout: Duration,
    reservations: BTreeMap<LogicalPartitionId, ReservationRecord>,
    resources: BTreeMap<LogicalPartitionId, Vec<DeviceHandle>>,
    staged_writes: BTreeMap<OutputRef, StagedWrite>,
    published_writes: BTreeMap<OutputRef, StagedWrite>,
    staged_total_bytes: u64,
    published_total_bytes: u64,
    published_once: bool,
    completed_events: BTreeSet<OperationEvent>,
}

impl DeviceRuntimeBackend {
    /// Bind partitions onto sessions. Every partition must name a member of
    /// `set`. The transfer budget is the adapter's declared class (distinct
    /// from the transaction's class-6 reservation).
    pub fn new(
        set: DeviceRuntimeSet,
        partition_devices: BTreeMap<LogicalPartitionId, PhysicalDeviceId>,
        launch: LaunchProgram,
        transfer_budget: TransferBudget,
        transfer_timeout: Duration,
    ) -> HostResult<Self> {
        if partition_devices.is_empty() {
            return Err(HostError::invalid_args(
                "DeviceRuntimeBackend requires at least one partition binding",
            ));
        }
        for (partition, device) in &partition_devices {
            if !set.contains(device) {
                return Err(HostError::invalid_args(format!(
                    "partition {partition} binds {device}, which is not a DeviceRuntimeSet member"
                )));
            }
        }
        Ok(Self {
            set,
            partition_devices,
            launch,
            modules: BTreeMap::new(),
            transport: HostStagedAdapter::new(transfer_budget),
            transfer_timeout,
            reservations: BTreeMap::new(),
            resources: BTreeMap::new(),
            staged_writes: BTreeMap::new(),
            published_writes: BTreeMap::new(),
            staged_total_bytes: 0,
            published_total_bytes: 0,
            published_once: false,
            completed_events: BTreeSet::new(),
        })
    }

    /// The composed runtime set.
    #[must_use]
    pub fn set(&self) -> &DeviceRuntimeSet {
        &self.set
    }

    /// The S4 selected-transport receipt (labeled + timed host-staged copies).
    #[must_use]
    pub fn transport_receipt(&self) -> TransportReceipt {
        self.transport.transport_receipt()
    }

    /// Bytes currently staged (same contract as the fake backend).
    #[must_use]
    pub fn staged_write_set(&self) -> &BTreeMap<OutputRef, StagedWrite> {
        &self.staged_writes
    }

    /// Bytes published by the last `publish`.
    #[must_use]
    pub fn published_write_set(&self) -> &BTreeMap<OutputRef, StagedWrite> {
        &self.published_writes
    }

    fn device_of(&self, partition: &LogicalPartitionId) -> Result<&PhysicalDeviceId, BackendError> {
        self.partition_devices.get(partition).ok_or_else(|| {
            BackendError::operation(
                partition.clone(),
                format!("partition {partition} is not bound to a PhysicalDeviceId"),
            )
        })
    }

    fn runtime_mut(
        &mut self,
        partition: &LogicalPartitionId,
    ) -> Result<&mut DeviceRuntime, BackendError> {
        let device = self.device_of(partition)?.clone();
        self.set.get_mut(&device).ok_or_else(|| {
            BackendError::device_loss(
                partition.clone(),
                format!("PhysicalDeviceId {device} is not in the runtime set"),
            )
        })
    }

    fn track(&mut self, partition: &LogicalPartitionId, handle: DeviceHandle) {
        self.resources
            .entry(partition.clone())
            .or_default()
            .push(handle);
    }

    fn alloc(
        &mut self,
        partition: &LogicalPartitionId,
        len_bytes: usize,
    ) -> Result<DeviceHandle, BackendError> {
        let handle = self
            .runtime_mut(partition)?
            .alloc_bytes(len_bytes)
            .map_err(|error| map_host_error(partition, error, FaultClass::Allocation))?;
        self.track(partition, handle);
        Ok(handle)
    }

    fn ensure_module(
        &mut self,
        partition: &LogicalPartitionId,
    ) -> Result<DeviceHandle, BackendError> {
        let device = self.device_of(partition)?.clone();
        if let Some(module) = self.modules.get(&device) {
            return Ok(*module);
        }
        let image = self.launch.image.clone();
        let module = self
            .runtime_mut(partition)?
            .load_module(&image)
            .map_err(|error| map_host_error(partition, error, FaultClass::Operation))?;
        self.modules.insert(device, module);
        Ok(module)
    }

    fn free_partition(&mut self, partition: &LogicalPartitionId) {
        let handles = self.resources.remove(partition).unwrap_or_default();
        if let Ok(runtime) = self.runtime_mut(partition) {
            for handle in handles {
                drop(runtime.release(&handle));
            }
        }
        self.reservations.remove(partition);
    }

    fn host_stage_copy(
        &mut self,
        source: &LogicalPartitionId,
        destination: &LogicalPartitionId,
        spec: TransferSpec,
        bytes: u64,
    ) -> Result<(), BackendError> {
        let len = usize::try_from(bytes).map_err(|_| {
            BackendError::operation(
                source.clone(),
                format!("transfer byte count {bytes} overflows usize"),
            )
        })?;
        let pattern = pattern_bytes(len);
        let src_handle = self.alloc(source, len)?;
        self.runtime_mut(source)?
            .copy_in_bytes(&src_handle, &pattern, DeviceDataType::U8)
            .map_err(|error| map_host_error(source, error, FaultClass::Operation))?;
        let source_bytes = self
            .runtime_mut(source)?
            .readback_bytes(&src_handle, DeviceDataType::U8)
            .map_err(|error| map_host_error(source, error, FaultClass::Operation))?;
        if source_bytes != pattern {
            return Err(BackendError::operation(
                source.clone(),
                "host-staged source readback was not byte-exact",
            ));
        }
        let source_value = SourceValue::new(
            source.clone(),
            spec.dtype(),
            spec.layout(),
            spec.generation(),
            source_bytes,
        );
        let outcome = self
            .transport
            .copy(&spec, &source_value, destination)
            .map_err(|error| transfer_to_backend(destination, error))?;
        if outcome.destination_bytes != pattern {
            return Err(BackendError::operation(
                destination.clone(),
                "host-staged copy was not byte-exact",
            ));
        }
        let dest_handle = self.alloc(destination, len)?;
        self.runtime_mut(destination)?
            .copy_in_bytes(&dest_handle, &outcome.destination_bytes, DeviceDataType::U8)
            .map_err(|error| map_host_error(destination, error, FaultClass::Operation))?;
        let landed = self
            .runtime_mut(destination)?
            .readback_bytes(&dest_handle, DeviceDataType::U8)
            .map_err(|error| map_host_error(destination, error, FaultClass::Operation))?;
        if landed != pattern {
            return Err(BackendError::operation(
                destination.clone(),
                "destination device buffer was not byte-exact after host-staged copy",
            ));
        }
        Ok(())
    }

    fn complete_events(&mut self, operation: &TransactionOperation) {
        self.completed_events.extend(operation.completed_events());
    }
}

impl DeviceExecutionBackend for DeviceRuntimeBackend {
    fn reserve(
        &mut self,
        partition: &LogicalPartitionId,
        reservation: &ReservationRecord,
    ) -> Result<(), BackendError> {
        if self.reservations.contains_key(partition) {
            return Err(BackendError::allocation(
                partition.clone(),
                format!("partition {partition} already holds a reservation"),
            ));
        }
        self.device_of(partition)?;
        if reservation.output_buffer_bytes() > 0 {
            let len = usize::try_from(reservation.output_buffer_bytes()).map_err(|_| {
                BackendError::allocation(partition.clone(), "output_buffer_bytes overflows usize")
            })?;
            self.alloc(partition, len)?;
        }
        if reservation.transfer_staging_bytes() > 0 {
            let len = usize::try_from(reservation.transfer_staging_bytes()).map_err(|_| {
                BackendError::allocation(
                    partition.clone(),
                    "transfer_staging_bytes overflows usize",
                )
            })?;
            self.alloc(partition, len)?;
        }
        if reservation.transaction_scratch_bytes() > 0 {
            let len = usize::try_from(reservation.transaction_scratch_bytes()).map_err(|_| {
                BackendError::allocation(
                    partition.clone(),
                    "transaction_scratch_bytes overflows usize",
                )
            })?;
            self.alloc(partition, len)?;
        }
        self.reservations.insert(partition.clone(), *reservation);
        Ok(())
    }

    fn run_operation(&mut self, operation: &TransactionOperation) -> Result<(), BackendError> {
        match operation {
            TransactionOperation::Launch {
                partition,
                output_bytes,
                ..
            } => {
                let len = usize::try_from(*output_bytes).map_err(|_| {
                    BackendError::operation(
                        partition.clone(),
                        format!("launch output_bytes {output_bytes} overflows usize"),
                    )
                })?;
                if len > 0 {
                    let module = self.ensure_module(partition)?;
                    let src = self.alloc(partition, len)?;
                    let dst = self.alloc(partition, len)?;
                    let pattern = pattern_bytes(len);
                    self.runtime_mut(partition)?
                        .copy_in_bytes(&src, &pattern, DeviceDataType::U8)
                        .map_err(|error| map_host_error(partition, error, FaultClass::Operation))?;
                    let n = (len / 4).max(1) as u32;
                    let entry = self.launch.entry.clone();
                    self.runtime_mut(partition)?
                        .launch_kernel(&module, &entry, &[src, dst], [n, 1, 1], [1, 1, 1])
                        .map_err(|error| map_host_error(partition, error, FaultClass::Operation))?;
                    self.runtime_mut(partition)?
                        .sync()
                        .map_err(|error| map_host_error(partition, error, FaultClass::Operation))?;
                }
                self.complete_events(operation);
                Ok(())
            }
            TransactionOperation::Transfer(transfer) => {
                let spec = TransferSpec::from_mirror(transfer, self.transfer_timeout);
                self.host_stage_copy(
                    transfer.source(),
                    transfer.destination(),
                    spec,
                    transfer.byte_count(),
                )?;
                self.complete_events(operation);
                Ok(())
            }
            TransactionOperation::CollectiveBroadcast(broadcast) => {
                for participant in broadcast.participants() {
                    if participant == broadcast.source() {
                        continue;
                    }
                    let spec = TransferSpec::new(
                        TransferRef::new(format!("{}:{participant}", broadcast.id())),
                        broadcast.source().clone(),
                        participant.clone(),
                        ByteRange::new(0, broadcast.byte_count()),
                        TransferDirectionMirror::BIDI,
                        MirroredDtype::F32,
                        MirroredStorageLayout::Dense,
                        0,
                        self.transfer_timeout,
                    );
                    self.host_stage_copy(
                        broadcast.source(),
                        participant,
                        spec,
                        broadcast.byte_count(),
                    )?;
                }
                self.complete_events(operation);
                Ok(())
            }
            TransactionOperation::Barrier { partitions, .. } => {
                let mut synced = BTreeSet::new();
                for partition in partitions {
                    let device = self.device_of(partition)?.clone();
                    if !synced.insert(device.clone()) {
                        continue;
                    }
                    self.runtime_mut(partition)?
                        .sync()
                        .map_err(|error| map_host_error(partition, error, FaultClass::Operation))?;
                }
                self.complete_events(operation);
                Ok(())
            }
        }
    }

    fn event_completed(&self, event: &OperationEvent) -> bool {
        self.completed_events.contains(event)
    }

    fn stage_write(&mut self, write: &StagedWrite) -> Result<(), BackendError> {
        if self.staged_writes.contains_key(write.output_ref()) {
            return Err(BackendError::allocation(
                write.partition().clone(),
                format!("output {} staged twice", write.output_ref()),
            ));
        }
        self.staged_total_bytes = self.staged_total_bytes.saturating_add(write.byte_count());
        self.staged_writes
            .insert(write.output_ref().clone(), write.clone());
        Ok(())
    }

    fn publish(&mut self) -> Result<(), BackendError> {
        if self.published_once {
            return Err(BackendError::operation(
                LogicalPartitionId::new("unknown"),
                "publish called twice — publication is one-shot",
            ));
        }
        self.published_writes = self.staged_writes.clone();
        self.published_total_bytes = self.staged_total_bytes;
        self.published_once = true;
        Ok(())
    }

    fn staged_bytes(&self) -> u64 {
        self.staged_total_bytes
    }

    fn published_bytes(&self) -> u64 {
        self.published_total_bytes
    }

    fn release(&mut self, partition: &LogicalPartitionId) {
        self.free_partition(partition);
    }

    fn retire(&mut self, partition: &LogicalPartitionId, _failure: &TransactionFailure) {
        let retired: u64 = self
            .staged_writes
            .values()
            .filter(|write| write.partition() == partition)
            .map(StagedWrite::byte_count)
            .sum();
        self.staged_total_bytes = self.staged_total_bytes.saturating_sub(retired);
        self.staged_writes
            .retain(|_, write| write.partition() != partition);
        self.free_partition(partition);
    }
}

enum FaultClass {
    Allocation,
    Operation,
}

fn map_host_error(
    partition: &LogicalPartitionId,
    error: HostError,
    class: FaultClass,
) -> BackendError {
    match error.code.as_str() {
        "E_TIMEOUT" => BackendError::timeout(partition.clone(), error.message),
        "E_CANCELLED" => BackendError::cancelled(partition.clone(), error.message),
        "E_CUDA_UNAVAILABLE" | "E_METAL_UNAVAILABLE" => {
            BackendError::device_loss(partition.clone(), error.message)
        }
        _ => match class {
            FaultClass::Allocation => {
                BackendError::allocation(partition.clone(), format!("{error}"))
            }
            FaultClass::Operation => BackendError::operation(partition.clone(), format!("{error}")),
        },
    }
}

fn transfer_to_backend(partition: &LogicalPartitionId, error: TransferError) -> BackendError {
    error.into_backend_error(partition.clone())
}

fn pattern_bytes(len: usize) -> Vec<u8> {
    (0..len).map(|index| (index % 256) as u8).collect()
}
