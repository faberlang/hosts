//! DCG-3: `DtypeSurface` ↔ `DeviceDataType` consistency ratchet.
//!
//! Every dtype `DtypeSurface` claims executable must be nameable by
//! [`DeviceDataType`] on the transfer surface. F16 is named (DSB-3, `563ea2b`).
//! BF16 is the documented slotless exception: radix `placement-debt-audit` F2
//! owns the placement-ABI discriminant. Lands green — regression guard, not
//! a red-first slot add.

use faber_host_macos_arm64::device_descriptor::DeviceDataType;
use host_coordinator::discovery::DtypeSurface;

/// Exhaustive `DtypeSurface` flag names. A new field fails to compile here.
fn surface_flags(surface: DtypeSurface) -> [(&'static str, bool); 6] {
    let DtypeSurface {
        f32,
        f64,
        f16,
        bf16,
        i8,
        i32,
    } = surface;
    [
        ("f32", f32),
        ("f64", f64),
        ("f16", f16),
        ("bf16", bf16),
        ("i8", i8),
        ("i32", i32),
    ]
}

/// Transfer-surface spelling for a `DtypeSurface` flag, when one exists.
///
/// `i8` is T1 kernel-smoke only (`DtypeSurface` is independent of
/// `DeviceDataType` and never `U8`-as-quantization).
fn transfer_type(surface_spelling: &str) -> Option<DeviceDataType> {
    match surface_spelling {
        "f32" => Some(DeviceDataType::F32),
        "f64" => Some(DeviceDataType::F64),
        "f16" => Some(DeviceDataType::F16),
        "bf16" => Some(DeviceDataType::BF16),
        "i8" => None,
        "i32" => Some(DeviceDataType::I32),
        other => panic!("unmapped DtypeSurface flag {other}"),
    }
}

/// Reverse map: which `DeviceDataType` variants are `DtypeSurface` flags.
///
/// `I64`/`U8` are transfer types the surface does not claim as executable
/// arithmetic. A new variant fails to compile until it is classified.
fn surface_spelling(dtype: DeviceDataType) -> Option<&'static str> {
    match dtype {
        DeviceDataType::F32 => Some("f32"),
        DeviceDataType::F64 => Some("f64"),
        DeviceDataType::F16 => Some("f16"),
        DeviceDataType::BF16 => Some("bf16"),
        DeviceDataType::I32 => Some("i32"),
        DeviceDataType::I64 | DeviceDataType::U8 => None,
    }
}

#[test]
fn dtype_surface_flags_map_to_device_data_type_spellings() {
    let claimed = DtypeSurface {
        f32: true,
        f64: true,
        f16: true,
        bf16: true,
        i8: true,
        i32: true,
    };

    for (spelling, is_claimed) in surface_flags(claimed) {
        assert!(is_claimed, "{spelling} fixture claims executable");
        assert_eq!(
            DeviceDataType::from_spelling(spelling),
            transfer_type(spelling),
            "{spelling} DtypeSurface flag ↔ DeviceDataType spelling",
        );
    }

    for dtype in [
        DeviceDataType::F32,
        DeviceDataType::F64,
        DeviceDataType::I32,
        DeviceDataType::I64,
        DeviceDataType::U8,
        DeviceDataType::F16,
        DeviceDataType::BF16,
    ] {
        if let Some(spelling) = surface_spelling(dtype) {
            assert_eq!(dtype.spelling(), spelling);
            assert_eq!(DeviceDataType::from_spelling(spelling), Some(dtype));
        }
    }

    assert_eq!(
        DeviceDataType::from_spelling("f16"),
        Some(DeviceDataType::F16),
        "F16 is named on the transfer surface",
    );
    assert_eq!(
        DeviceDataType::F16.placement_discriminant(),
        Some(10),
        "F16 is named on the placement ABI",
    );

    assert_eq!(
        DeviceDataType::from_spelling("bf16"),
        Some(DeviceDataType::BF16),
        "BF16 is nameable on the transfer surface",
    );
    assert_eq!(
        DeviceDataType::BF16.placement_discriminant(),
        None,
        "BF16 is slotless pending radix placement-debt-audit F2 (MirScalarLayout has no BF16 variant; F2 owns the discriminant)",
    );
}
