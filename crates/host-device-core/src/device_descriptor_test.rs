use super::*;

#[test]
fn f16_bf16_round_trip_spelling_and_byte_width() {
    assert_eq!(DeviceDataType::F16.spelling(), "f16");
    assert_eq!(
        DeviceDataType::from_spelling("f16"),
        Some(DeviceDataType::F16)
    );
    assert_eq!(DeviceDataType::F16.byte_width(), 2);

    assert_eq!(DeviceDataType::BF16.spelling(), "bf16");
    assert_eq!(
        DeviceDataType::from_spelling("bf16"),
        Some(DeviceDataType::BF16)
    );
    assert_eq!(DeviceDataType::BF16.byte_width(), 2);
}

#[test]
fn placement_discriminant_maps_to_device_data_type() {
    // MirScalarLayout declaration-order discriminants (F2 owner).
    assert_eq!(
        DeviceDataType::from_placement_discriminant(3),
        Some(DeviceDataType::I32)
    );
    assert_eq!(
        DeviceDataType::from_placement_discriminant(4),
        Some(DeviceDataType::I64)
    );
    assert_eq!(
        DeviceDataType::from_placement_discriminant(6),
        Some(DeviceDataType::U8)
    );
    assert_eq!(
        DeviceDataType::from_placement_discriminant(10),
        Some(DeviceDataType::F16)
    );
    assert_eq!(
        DeviceDataType::from_placement_discriminant(11),
        Some(DeviceDataType::F32)
    );
    assert_eq!(
        DeviceDataType::from_placement_discriminant(12),
        Some(DeviceDataType::F64)
    );
    assert_eq!(DeviceDataType::from_placement_discriminant(0), None);
    assert_eq!(DeviceDataType::from_placement_discriminant(30), None);

    assert_eq!(DeviceDataType::I32.placement_discriminant(), Some(3));
    assert_eq!(DeviceDataType::I64.placement_discriminant(), Some(4));
    assert_eq!(DeviceDataType::U8.placement_discriminant(), Some(6));
    assert_eq!(DeviceDataType::F16.placement_discriminant(), Some(10));
    assert_eq!(DeviceDataType::F32.placement_discriminant(), Some(11));
    assert_eq!(DeviceDataType::F64.placement_discriminant(), Some(12));
    assert_eq!(DeviceDataType::BF16.placement_discriminant(), None);
}

#[test]
fn byte_length_rejects_u64_overflow() {
    let buffer = DescriptorBuffer {
        buffer_id: 1,
        buffer_name: "overflow".to_owned(),
        semantic_value: 1,
        role: DeviceBufferRole::InOut,
        lifetime: DeviceBufferLifetime::PerProgram,
        initialization: DeviceBufferInitialization::ZeroFill,
        binding: 0,
        element_ty: DeviceDataType::F32,
        element_count: u64::MAX,
        version: 1,
    };

    assert_eq!(buffer.byte_length(), None);
}

#[test]
fn validate_rejects_overflowing_buffer_byte_length() {
    let count = u64::MAX;
    let slot = DescriptorBuffer {
        buffer_id: 1,
        buffer_name: "overflow".to_owned(),
        semantic_value: 1,
        role: DeviceBufferRole::InOut,
        lifetime: DeviceBufferLifetime::PerProgram,
        initialization: DeviceBufferInitialization::ZeroFill,
        binding: 0,
        element_ty: DeviceDataType::F32,
        element_count: count,
        version: 1,
    };
    let descriptor = DeviceDescriptor {
        backend: DeviceBackend::Metal,
        module_image: vec![1],
        kernels: vec![DescriptorKernel {
            entry: "kernel".to_owned(),
            buffers: vec![slot],
            grid: [1, 1, 1],
            block: [1, 1, 1],
        }],
        launches: vec![DescriptorLaunch {
            id: 1,
            kernel_index: 0,
        }],
        buffer_versions: vec![DescriptorBufferVersion {
            buffer_id: 1,
            version: 1,
            element_ty: DeviceDataType::F32,
            element_count: count,
        }],
        program_lifetime: DeviceProgramLifetime::SingleRun,
        data_flow: Vec::new(),
        roots: vec![1],
        results: Vec::new(),
        end_of_run_results: Vec::new(),
    };

    let error = descriptor.validate().expect_err("overflowing descriptor");
    assert_eq!(error.code, E_DEVICE_SHAPE_MISMATCH);
}
