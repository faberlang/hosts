//! Companion tests for [`super::library_runtime`] (repo companion-test convention).

use super::*;

#[test]
fn runtime_msl_admits_activation_width_distinct_from_q_width() {
    let module = library_family_msl(&LibraryFamilyMslFacts {
        rows: 1,
        hidden: 8,
        kv_heads: 1,
        q_per_kv: 1,
        head_dim: 4,
        epsilon: 1.0e-5,
    })
    .expect("activation width and Q width are independent facts");
    assert!(module.contains("constant uint HIDDEN = 8u;"));
    assert!(module.contains("constant uint Q_WIDTH = 4u;"));
}

#[test]
fn runtime_qkv_entry_drift_fails_before_output_write() {
    let bind = QkvProjectionBind::grouped(1, 4, 1, 1, 4, [4, 1, 1]);
    let activation = [1.0; 4];
    let weight = [1.0; 16];
    let mut q = [f32::NAN; 4];
    let mut k = [f32::NAN; 4];
    let mut v = [f32::NAN; 4];
    let error = dispatch_metal_library(MetalLibraryDispatch::QkvProjection {
        library_entry: Some("ResidualRmsNorm"),
        decode_gemv: 0,
        layout: QkvProjectionLayout::Grouped,
        bind: &bind,
        activation: &activation,
        weights: [
            QkvProjectionWeight::Dense(&weight),
            QkvProjectionWeight::Dense(&weight),
            QkvProjectionWeight::Dense(&weight),
        ],
        biases: [None, None, None],
        rope: None,
        outputs: [&mut q, &mut k, &mut v],
    })
    .expect_err("wrong runtime entry must fail closed");
    assert!(matches!(
        error,
        KernelBodyError::InvalidBind(message) if message.contains("QKV projection selection")
    ));
    assert!(q.iter().all(|value| value.is_nan()));
    assert!(k.iter().all(|value| value.is_nan()));
    assert!(v.iter().all(|value| value.is_nan()));
}

#[test]
fn production_bridge_binds_kv_write_targets_for_both_gqa_pairings() {
    use crate::device_host::DeviceRuntime;
    use crate::metal_host::{FakeMetalDriver, MetalHostSession};
    use host_coordinator::{DeviceBackend, DeviceHandle};

    fn view<'a>(
        handle: &'a DeviceHandle,
        count: u64,
        binding: u32,
    ) -> FusedLibraryDeviceBuffer<'a> {
        FusedLibraryDeviceBuffer {
            handle,
            dtype: DeviceDataType::F32,
            byte_offset: 0,
            view_span: count * 4,
            binding_index: binding,
            packed_format: None,
        }
    }

    for (kv_heads, q_per_kv, k_binding, v_binding) in [(2, 2, 5, 6), (1, 4, 7, 8)] {
        let mut runtime = DeviceRuntime::Metal(
            MetalHostSession::with_driver(Box::new(FakeMetalDriver::default()))
                .expect("fake Metal admission"),
        );
        let head_dim = 2u64;
        let hidden = kv_heads * q_per_kv * head_dim;
        let q_width = hidden;
        let kv_width = kv_heads * head_dim;
        let activation_values = vec![1.0f32; hidden as usize];
        let q_weight_values = vec![1.0f32; (hidden * q_width) as usize];
        let k_weight_values = vec![2.0f32; (hidden * kv_width) as usize];
        let v_weight_values = vec![3.0f32; (hidden * kv_width) as usize];
        let q_output_values = vec![0.0f32; q_width as usize];
        let k_output_values = vec![0.0f32; kv_width as usize];
        let v_output_values = vec![0.0f32; kv_width as usize];
        let alloc = |runtime: &mut DeviceRuntime, values: &[f32]| {
            let handle = runtime
                .alloc_bytes(values.len() * 4)
                .expect("allocate bridge buffer");
            runtime
                .copy_in_f32(&handle, values)
                .expect("initialize bridge buffer");
            handle
        };
        let activation = alloc(&mut runtime, &activation_values);
        let q_weight = alloc(&mut runtime, &q_weight_values);
        let k_weight = alloc(&mut runtime, &k_weight_values);
        let v_weight = alloc(&mut runtime, &v_weight_values);
        let q_output = alloc(&mut runtime, &q_output_values);
        let k_output = alloc(&mut runtime, &k_output_values);
        let v_output = alloc(&mut runtime, &v_output_values);
        let receipt = dispatch_fused_qkv_device(
            &mut runtime,
            FusedQkvDeviceDispatch {
                library_entry: "QkvProjection",
                derived_entry: "prefill_blk_0_QkvProjection",
                decode_gemv: 1,
                bind: QkvProjectionBind::grouped(
                    1,
                    hidden,
                    kv_heads,
                    q_per_kv,
                    head_dim,
                    [q_width as u32, 1, 1],
                ),
                activation: view(&activation, hidden, 0),
                weights: [
                    view(&q_weight, hidden * q_width, 1),
                    view(&k_weight, hidden * kv_width, 2),
                    view(&v_weight, hidden * kv_width, 3),
                ],
                biases: [None, None, None],
                rope: None,
                outputs: [
                    view(&q_output, q_width, 4),
                    view(&k_output, kv_width, k_binding),
                    view(&v_output, kv_width, v_binding),
                ],
            },
        )
        .expect("fused body executes");
        assert_eq!(receipt.entry, "prefill_blk_0_QkvProjection");
        assert_eq!(receipt.body, "qkv_projection_cpu");
        assert_eq!(receipt.k_output_binding, k_binding);
        assert_eq!(receipt.v_output_binding, v_binding);
        let k_values = runtime.readback_f32(&k_output).expect("read K output");
        let v_values = runtime.readback_f32(&v_output).expect("read V output");
        assert!(k_values.iter().all(|value| *value == hidden as f32 * 2.0));
        assert!(v_values.iter().all(|value| *value == hidden as f32 * 3.0));
        assert_eq!(runtime.backend(), DeviceBackend::Metal);
    }
}

#[test]
fn residual_rms_bridge_adds_skip_before_normalization() {
    use crate::device_host::DeviceRuntime;
    use crate::metal_host::{FakeMetalDriver, MetalHostSession};
    use host_coordinator::DeviceHandle;

    fn view<'a>(
        handle: &'a DeviceHandle,
        count: u64,
        binding: u32,
    ) -> FusedLibraryDeviceBuffer<'a> {
        FusedLibraryDeviceBuffer {
            handle,
            dtype: DeviceDataType::F32,
            byte_offset: 0,
            view_span: count * 4,
            binding_index: binding,
            packed_format: None,
        }
    }

    let mut runtime = DeviceRuntime::Metal(
        MetalHostSession::with_driver(Box::new(FakeMetalDriver::default()))
            .expect("fake Metal admission"),
    );
    let rows = 2u64;
    let hidden = 4u64;
    let epsilon = 1.0e-6f32;
    let residual = [1.0f32, -2.0, 3.0, -4.0, 0.5, 0.5, -1.5, 2.5];
    let skip = [0.25f32, 0.25, -0.25, -0.25, 1.0, -1.0, 2.0, -2.0];
    let gamma = [1.0f32, 0.5, 2.0, 1.5];
    let mut expected = vec![0.0f32; (rows * hidden) as usize];
    for row in 0..rows as usize {
        let base = row * hidden as usize;
        let summed: Vec<f32> = (0..hidden as usize)
            .map(|col| residual[base + col] + skip[base + col])
            .collect();
        let sumsq: f32 = summed.iter().map(|value| value * value).sum();
        let scale = 1.0 / ((sumsq / hidden as f32 + epsilon).sqrt());
        for col in 0..hidden as usize {
            expected[base + col] = summed[col] * scale * gamma[col];
        }
    }
    let alloc = |runtime: &mut DeviceRuntime, values: &[f32]| {
        let handle = runtime
            .alloc_bytes(values.len() * 4)
            .expect("allocate bridge buffer");
        runtime
            .copy_in_f32(&handle, values)
            .expect("initialize bridge buffer");
        handle
    };
    let residual_handle = alloc(&mut runtime, &residual);
    let skip_handle = alloc(&mut runtime, &skip);
    let gamma_handle = alloc(&mut runtime, &gamma);
    let output_handle = alloc(&mut runtime, &vec![0.0f32; expected.len()]);
    dispatch_fused_residual_rms_device(
        &mut runtime,
        FusedResidualRmsDeviceDispatch {
            library_entry: "ResidualRmsNorm",
            rows,
            hidden,
            epsilon,
            residual: view(&residual_handle, rows * hidden, 0),
            skip: view(&skip_handle, rows * hidden, 1),
            gamma: view(&gamma_handle, hidden, 2),
            output: view(&output_handle, rows * hidden, 3),
        },
    )
    .expect("fused residual/RMS body executes");
    let written = runtime
        .readback_f32(&output_handle)
        .expect("read output buffer");
    for (actual, want) in written.iter().zip(&expected) {
        assert!(
            (actual - want).abs() <= 1.0e-6 * want.abs().max(1.0),
            "expected {want}, wrote {actual}"
        );
    }
    // Without the skip the first row's scale would normalize the bare
    // residual: prove the skip changed the bytes.
    let bare = {
        let summed: Vec<f32> = residual[..hidden as usize].to_vec();
        let sumsq: f32 = summed.iter().map(|value| value * value).sum();
        1.0 / ((sumsq / hidden as f32 + epsilon).sqrt())
    };
    let with_skip = expected[0] / gamma[0];
    assert!((with_skip - bare).abs() > 1.0e-3);
    // PB-8: the residual stream must carry the skip so the layer-end
    // residual add composes h + o + down.
    let written_residual = runtime
        .readback_f32(&residual_handle)
        .expect("read residual buffer");
    for (actual, want) in written_residual
        .iter()
        .zip(residual.iter().zip(&skip).map(|(a, b)| a + b))
    {
        assert!(
            (actual - want).abs() <= 1.0e-6 * want.abs().max(1.0),
            "expected accumulated {want}, wrote {actual}"
        );
    }
}

#[test]
fn residual_rms_bridge_writeback_splices_a_view_span_not_the_whole_buffer() {
    use crate::device_host::DeviceRuntime;
    use crate::metal_host::{FakeMetalDriver, MetalHostSession};

    let mut runtime = DeviceRuntime::Metal(
        MetalHostSession::with_driver(Box::new(FakeMetalDriver::default()))
            .expect("fake Metal admission"),
    );
    // One pooled buffer holding two spans; the residual view is the
    // second span, so a whole-buffer overwrite would corrupt the first.
    let pooled = [9.0f32, 9.0, 9.0, 9.0, 1.0, -2.0, 3.0, -4.0];
    let skip = [0.25f32, 0.25, -0.25, -0.25];
    let gamma = [1.0f32, 0.5, 2.0, 1.5];
    let alloc = |runtime: &mut DeviceRuntime, values: &[f32]| {
        let handle = runtime
            .alloc_bytes(values.len() * 4)
            .expect("allocate bridge buffer");
        runtime
            .copy_in_f32(&handle, values)
            .expect("initialize bridge buffer");
        handle
    };
    let pooled_handle = alloc(&mut runtime, &pooled);
    let skip_handle = alloc(&mut runtime, &skip);
    let gamma_handle = alloc(&mut runtime, &gamma);
    let output_handle = alloc(&mut runtime, &[0.0f32; 4]);
    fn view<'a>(
        handle: &'a host_coordinator::DeviceHandle,
        count: u64,
        offset: u64,
        binding: u32,
    ) -> FusedLibraryDeviceBuffer<'a> {
        FusedLibraryDeviceBuffer {
            handle,
            dtype: DeviceDataType::F32,
            byte_offset: offset,
            view_span: count * 4,
            binding_index: binding,
            packed_format: None,
        }
    }
    dispatch_fused_residual_rms_device(
        &mut runtime,
        FusedResidualRmsDeviceDispatch {
            library_entry: "ResidualRmsNorm",
            rows: 1,
            hidden: 4,
            epsilon: 1.0e-6,
            residual: view(&pooled_handle, 4, 16, 0),
            skip: view(&skip_handle, 4, 0, 1),
            gamma: view(&gamma_handle, 4, 0, 2),
            output: view(&output_handle, 4, 0, 3),
        },
    )
    .expect("fused residual/RMS body executes");
    let pooled_after = runtime
        .readback_f32(&pooled_handle)
        .expect("read pooled buffer");
    for (actual, want) in pooled_after.iter().zip(
        pooled
            .iter()
            .enumerate()
            .map(|(index, value)| value + skip.get(index.wrapping_sub(4)).copied().unwrap_or(0.0)),
    ) {
        assert!(
            (actual - want).abs() <= 1.0e-6 * want.abs().max(1.0),
            "expected pooled {want}, wrote {actual}"
        );
    }
}

#[test]
fn residual_rms_bridge_rejects_foreign_entry() {
    use crate::device_host::DeviceRuntime;
    use crate::metal_host::{FakeMetalDriver, MetalHostSession};

    let mut runtime = DeviceRuntime::Metal(
        MetalHostSession::with_driver(Box::new(FakeMetalDriver::default()))
            .expect("fake Metal admission"),
    );
    let handle = runtime.alloc_bytes(16).expect("allocate bridge buffer");
    let view = FusedLibraryDeviceBuffer {
        handle: &handle,
        dtype: DeviceDataType::F32,
        byte_offset: 0,
        view_span: 16,
        binding_index: 0,
        packed_format: None,
    };
    let error = dispatch_fused_residual_rms_device(
        &mut runtime,
        FusedResidualRmsDeviceDispatch {
            library_entry: "QkvProjection",
            rows: 1,
            hidden: 4,
            epsilon: 1.0e-6,
            residual: view,
            skip: view,
            gamma: view,
            output: view,
        },
    )
    .expect_err("wrong runtime entry must fail closed");
    assert!(error.message.contains("disagrees with library_entry"));
}

#[test]
fn production_bridge_consumes_packed_qkv_views_with_explicit_format() {
    use crate::device_host::DeviceRuntime;
    use crate::metal_host::{FakeMetalDriver, MetalHostSession};
    use host_coordinator::{DeviceBackend, DeviceHandle};

    fn f32_view<'a>(
        handle: &'a DeviceHandle,
        count: u64,
        binding: u32,
    ) -> FusedLibraryDeviceBuffer<'a> {
        FusedLibraryDeviceBuffer {
            handle,
            dtype: DeviceDataType::F32,
            byte_offset: 0,
            view_span: count * 4,
            binding_index: binding,
            packed_format: None,
        }
    }

    fn packed_view<'a>(
        handle: &'a DeviceHandle,
        bytes: u64,
        binding: u32,
    ) -> FusedLibraryDeviceBuffer<'a> {
        FusedLibraryDeviceBuffer {
            handle,
            dtype: DeviceDataType::U8,
            byte_offset: 0,
            view_span: bytes,
            binding_index: binding,
            packed_format: Some(PackedStorageFormat::Q8_0),
        }
    }

    fn q8_columns(value: u8, columns: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(columns * 34);
        for _ in 0..columns {
            bytes.extend_from_slice(&[0x00, 0x3c]);
            bytes.extend(std::iter::repeat_n(value, 32));
        }
        bytes
    }

    let mut runtime = DeviceRuntime::Metal(
        MetalHostSession::with_driver(Box::new(FakeMetalDriver::default()))
            .expect("fake Metal admission"),
    );
    let hidden = 32u64;
    let alloc_f32 = |runtime: &mut DeviceRuntime, values: &[f32]| {
        let handle = runtime
            .alloc_bytes(values.len() * 4)
            .expect("allocate f32 bridge buffer");
        runtime
            .copy_in_f32(&handle, values)
            .expect("initialize f32 bridge buffer");
        handle
    };
    let alloc_bytes = |runtime: &mut DeviceRuntime, values: &[u8]| {
        let handle = runtime
            .alloc_bytes(values.len())
            .expect("allocate packed bridge buffer");
        runtime
            .copy_in_bytes(&handle, values, DeviceDataType::U8)
            .expect("initialize packed bridge buffer");
        handle
    };
    let activation = alloc_f32(&mut runtime, &[1.0; 32]);
    let q_weight_bytes = q8_columns(1, 32);
    let k_weight_bytes = q8_columns(2, 32);
    let v_weight_bytes = q8_columns(3, 32);
    let q_weight = alloc_bytes(&mut runtime, &q_weight_bytes);
    let k_weight = alloc_bytes(&mut runtime, &k_weight_bytes);
    let v_weight = alloc_bytes(&mut runtime, &v_weight_bytes);
    let q_output = alloc_f32(&mut runtime, &[0.0; 32]);
    let k_output = alloc_f32(&mut runtime, &[0.0; 32]);
    let v_output = alloc_f32(&mut runtime, &[0.0; 32]);

    dispatch_fused_qkv_device(
        &mut runtime,
        FusedQkvDeviceDispatch {
            library_entry: "QkvProjection",
            derived_entry: "prefill_blk_0_QkvProjection",
            decode_gemv: 1,
            bind: QkvProjectionBind::grouped(1, hidden, 1, 1, 32, [32, 1, 1]),
            activation: FusedLibraryDeviceBuffer {
                handle: &activation,
                dtype: DeviceDataType::F32,
                byte_offset: 0,
                view_span: hidden * 4,
                binding_index: 0,
                packed_format: None,
            },
            weights: [
                packed_view(&q_weight, q_weight_bytes.len() as u64, 1),
                packed_view(&k_weight, k_weight_bytes.len() as u64, 2),
                packed_view(&v_weight, v_weight_bytes.len() as u64, 3),
            ],
            biases: [None, None, None],
            rope: None,
            outputs: [
                f32_view(&q_output, hidden, 4),
                f32_view(&k_output, hidden, 5),
                f32_view(&v_output, hidden, 6),
            ],
        },
    )
    .expect("packed QKV bridge executes");

    let q = runtime.readback_f32(&q_output).expect("read packed Q");
    let k = runtime.readback_f32(&k_output).expect("read packed K");
    let v = runtime.readback_f32(&v_output).expect("read packed V");
    assert!(q.iter().all(|value| *value == 32.0));
    assert!(k.iter().all(|value| *value == 64.0));
    assert!(v.iter().all(|value| *value == 96.0));
    assert_eq!(runtime.backend(), DeviceBackend::Metal);
}

#[test]
fn production_bridge_admits_f32_word_tagged_packed_regions() {
    use crate::device_host::DeviceRuntime;
    use crate::metal_host::{FakeMetalDriver, MetalHostSession};
    use host_coordinator::{DeviceBackend, DeviceHandle};

    // The wire types a padded packed region as f32 words (byte length/4)
    // while the bytes keep the native GGML layout; the uploaded format
    // fact, not the dtype tag, selects the packed path.
    fn f32_word_view<'a>(
        handle: &'a DeviceHandle,
        words: u64,
        binding: u32,
    ) -> FusedLibraryDeviceBuffer<'a> {
        FusedLibraryDeviceBuffer {
            handle,
            dtype: DeviceDataType::F32,
            byte_offset: 0,
            view_span: words * 4,
            binding_index: binding,
            packed_format: Some(PackedStorageFormat::Q8_0),
        }
    }

    fn f32_view<'a>(
        handle: &'a DeviceHandle,
        count: u64,
        binding: u32,
    ) -> FusedLibraryDeviceBuffer<'a> {
        FusedLibraryDeviceBuffer {
            handle,
            dtype: DeviceDataType::F32,
            byte_offset: 0,
            view_span: count * 4,
            binding_index: binding,
            packed_format: None,
        }
    }

    fn q8_columns(value: u8, columns: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(columns * 34);
        for _ in 0..columns {
            bytes.extend_from_slice(&[0x00, 0x3c]);
            bytes.extend(std::iter::repeat_n(value, 32));
        }
        bytes
    }

    let mut runtime = DeviceRuntime::Metal(
        MetalHostSession::with_driver(Box::new(FakeMetalDriver::default()))
            .expect("fake Metal admission"),
    );
    let hidden = 32u64;
    let alloc_f32 = |runtime: &mut DeviceRuntime, values: &[f32]| {
        let handle = runtime
            .alloc_bytes(values.len() * 4)
            .expect("allocate f32 bridge buffer");
        runtime
            .copy_in_f32(&handle, values)
            .expect("initialize f32 bridge buffer");
        handle
    };
    let alloc_bytes = |runtime: &mut DeviceRuntime, values: &[u8]| {
        let handle = runtime
            .alloc_bytes(values.len())
            .expect("allocate packed bridge buffer");
        runtime
            .copy_in_bytes(&handle, values, DeviceDataType::U8)
            .expect("initialize packed bridge buffer");
        handle
    };
    let activation = alloc_f32(&mut runtime, &[1.0; 32]);
    let q_weight_bytes = q8_columns(1, 32);
    let k_weight_bytes = q8_columns(2, 32);
    let v_weight_bytes = q8_columns(3, 32);
    // Pad each region to a whole f32 word count, as the wire does.
    let words = |bytes: &[u8]| -> u64 { bytes.len().div_ceil(4) as u64 };
    let q_weight = alloc_bytes(&mut runtime, &q_weight_bytes);
    let k_weight = alloc_bytes(&mut runtime, &k_weight_bytes);
    let v_weight = alloc_bytes(&mut runtime, &v_weight_bytes);
    let q_output = alloc_f32(&mut runtime, &[0.0; 32]);
    let k_output = alloc_f32(&mut runtime, &[0.0; 32]);
    let v_output = alloc_f32(&mut runtime, &[0.0; 32]);

    dispatch_fused_qkv_device(
        &mut runtime,
        FusedQkvDeviceDispatch {
            library_entry: "QkvProjection",
            derived_entry: "prefill_blk_0_QkvProjection",
            decode_gemv: 1,
            bind: QkvProjectionBind::grouped(1, hidden, 1, 1, 32, [32, 1, 1]),
            activation: f32_view(&activation, hidden, 0),
            weights: [
                f32_word_view(&q_weight, words(&q_weight_bytes), 1),
                f32_word_view(&k_weight, words(&k_weight_bytes), 2),
                f32_word_view(&v_weight, words(&v_weight_bytes), 3),
            ],
            biases: [None, None, None],
            rope: None,
            outputs: [
                f32_view(&q_output, hidden, 4),
                f32_view(&k_output, hidden, 5),
                f32_view(&v_output, hidden, 6),
            ],
        },
    )
    .expect("f32-word-tagged packed QKV bridge executes");

    let q = runtime.readback_f32(&q_output).expect("read packed Q");
    let k = runtime.readback_f32(&k_output).expect("read packed K");
    let v = runtime.readback_f32(&v_output).expect("read packed V");
    assert!(q.iter().all(|value| *value == 32.0));
    assert!(k.iter().all(|value| *value == 64.0));
    assert!(v.iter().all(|value| *value == 96.0));
    assert_eq!(runtime.backend(), DeviceBackend::Metal);
}
