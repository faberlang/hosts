//! LLVM host integration tests.
//!
//! Float comparisons here are against compile-time exact IEEE values after
//! parse round-trips (e.g. `"1.25"` → `1.25`), not fuzzy numeric algorithms.
#![allow(clippy::float_cmp)]
// Test-only cast noise: truncation/wrap/precision-loss sites in test helpers
// that mirror ABI call signatures (i64 → narrower types, float ↔ int, etc.).
// Production code in sibling modules is kept clean per Pro cast policy.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use super::cli::{
    __faber_rt_v1_cli_exit_code, __faber_rt_v1_cli_field_i64, __faber_rt_v1_cli_parse,
    __faber_rt_v1_cli_selected_command,
};
use super::*;
use std::ffi::{c_void, CStr};

#[test]
fn abi_status_codes_match_radix_host_abi_authority() {
    // Runtime owns the struct layouts; radix-host-abi owns the code values.
    let ours = [
        ("STATUS_OK", STATUS_OK.code),
        ("STATUS_INVALID_ARGUMENT", STATUS_INVALID_ARGUMENT.code),
        ("STATUS_IO_ERROR", STATUS_IO_ERROR.code),
        ("STATUS_PANIC", STATUS_PANIC.code),
        ("STATUS_UNSUPPORTED", STATUS_UNSUPPORTED.code),
        ("STATUS_FALLIBLE", STATUS_FALLIBLE.code),
    ];
    assert_eq!(ours.as_slice(), radix_host_abi::STATUS_CODES);
}

#[test]
fn fallible_error_pairs_status_first_with_payload_handle() {
    let mut context = ptr::null_mut();
    let status = unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) };
    assert_eq!(status, STATUS_OK);

    let stored = format::store_text(context, "tensor".to_owned());
    assert!(stored.status.is_ok());
    let result = unsafe { __faber_rt_v1_fallible_error(context, stored.value) };
    assert_eq!(result.status, STATUS_FALLIBLE);
    assert_eq!(result.value, stored.value);
    assert!(!result.status.is_ok());

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn fallible_error_rejects_null_context() {
    let result = unsafe { __faber_rt_v1_fallible_error(ptr::null_mut(), ptr::dangling_mut()) };
    assert_eq!(result, FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT));
}

#[test]
fn init_write_and_shutdown_round_trip() {
    let mut context = ptr::null_mut();
    let status = unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) };
    assert_eq!(status, STATUS_OK);
    assert!(!context.is_null());
    let status =
        unsafe { __faber_rt_v1_write_nota_text(context, FaberRtSliceV1::from_static(b"")) };
    assert_eq!(status, STATUS_OK);
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn arguments_excludes_host_argv0_and_returns_text_lista() {
    let program = std::ffi::CString::new("prog").unwrap();
    let first = std::ffi::CString::new("alpha").unwrap();
    let second = std::ffi::CString::new("beta").unwrap();
    let argv = [program.as_ptr(), first.as_ptr(), second.as_ptr()];
    let mut context = ptr::null_mut();
    let status = unsafe { __faber_rt_v1_init(3, argv.as_ptr(), &raw mut context) };
    assert_eq!(status, STATUS_OK);

    // Faber argumenta semantics: the captured arguments exclude argv[0].
    let runtime = unsafe { &*context.cast::<RuntimeContext>() };
    assert_eq!(
        runtime.arguments,
        vec![b"alpha".to_vec(), b"beta".to_vec()],
        "process argumenta must exclude the host argv[0] program path"
    );

    let result = unsafe { __faber_rt_v1_arguments(context) };
    assert!(result.status.is_ok(), "arguments symbol must succeed");
    assert!(
        !result.value.is_null(),
        "arguments must return a lista handle"
    );
    let array = array::find_array(runtime, result.value).expect("arguments handle in arena");
    assert_eq!(
        array.kind,
        radix_host_abi::VALUE_KIND_PTR,
        "argumenta list must be a lista<textus>"
    );
    let rendered = array
        .values
        .iter()
        .filter_map(|value| match value {
            array::RuntimeValue::Ptr(handle) => {
                let text = format::find_text(runtime, handle)?;
                Some(text.value.clone())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        rendered,
        vec!["alpha".to_owned(), "beta".to_owned()],
        "argumenta list must carry the exact argv[1..] text elements"
    );

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn arguments_empty_without_process_args() {
    let mut context = ptr::null_mut();
    let status = unsafe { __faber_rt_v1_init(1, std::ptr::null(), &raw mut context) };
    assert_eq!(
        status, STATUS_INVALID_ARGUMENT,
        "argc>0 with null argv rejects"
    );
    let status = unsafe { __faber_rt_v1_init(0, std::ptr::null(), &raw mut context) };
    assert_eq!(status, STATUS_OK);
    let result = unsafe { __faber_rt_v1_arguments(context) };
    assert!(result.status.is_ok());
    let runtime = unsafe { &*context.cast::<RuntimeContext>() };
    let array = array::find_array(runtime, result.value).expect("empty arguments handle");
    assert!(array.values.is_empty());
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn invalid_slice_fails_closed() {
    let status = unsafe {
        __faber_rt_v1_write_nota_text(
            ptr::dangling_mut(),
            FaberRtSliceV1 {
                data: ptr::null(),
                len: 1,
            },
        )
    };
    assert_eq!(status, STATUS_INVALID_ARGUMENT);
}

#[test]
fn diagnostic_nota_scalar_variants_all_return_ok() {
    let mut context = ptr::null_mut();
    let status = unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) };
    assert_eq!(status, STATUS_OK);

    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_nota_i64(context, 42) },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_nota_i1(context, 1) },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_nota_f32(context, 1.25) },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_nota_f64(context, 2.5) },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_nota_i8(context, -8) },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_nota_i32(context, -32) },
        STATUS_OK
    );

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn diagnostic_nota_text_and_ascii_accept_valid_inputs_reject_nulls() {
    let mut context = ptr::null_mut();
    let status = unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) };
    assert_eq!(status, STATUS_OK);

    let text = FaberRtSliceV1::from_static(b"nota text");
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_nota_text(context, &raw const text) },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_nota_text(context, ptr::null()) },
        STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_nota_ascii(context, c"nota ascii".as_ptr().cast()) },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_nota_ascii(context, ptr::null()) },
        STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_nota_ptr(context, ptr::null()) },
        STATUS_UNSUPPORTED
    );

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn diagnostic_nota_option_renders_payload_or_nihil() {
    let mut context = ptr::null_mut();
    let status = unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) };
    assert_eq!(status, STATUS_OK);

    // Present raw null-encoded option: the pointer bits ARE the i64 payload
    // (the optional-chain `inttoptr` representation).
    let present = 100usize as *mut c_void;
    assert_eq!(
        unsafe {
            __faber_rt_v1_diagnostic_nota_option(context, present, radix_host_abi::VALUE_KIND_I64)
        },
        STATUS_OK
    );

    // Nihil option: the null handle renders `nihil`.
    assert_eq!(
        unsafe {
            __faber_rt_v1_diagnostic_nota_option(
                context,
                ptr::null_mut(),
                radix_host_abi::VALUE_KIND_I64,
            )
        },
        STATUS_OK
    );

    // Unknown non-null raw handle with an unsupported kind stays fail-closed.
    assert_eq!(
        unsafe {
            __faber_rt_v1_diagnostic_nota_option(context, present, radix_host_abi::VALUE_KIND_I8)
        },
        STATUS_UNSUPPORTED
    );

    unsafe { __faber_rt_v1_shutdown(context) };
}

/// L11 (fc9be27a): family 3 — `mone`/`scribe`/`vide` of an option value render
/// through the option diagnostic carrier like `nota`: the payload for a
/// present raw null-encoded option, `nihil` for the null handle, on the
/// stream's channel.
#[test]
fn diagnostic_mone_scribe_vide_option_carriers_render_payload_or_nihil() {
    let mut context = ptr::null_mut();
    let status = unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) };
    assert_eq!(status, STATUS_OK);

    // Present raw null-encoded option: the pointer bits ARE the i64 payload.
    let present = 100usize as *mut c_void;
    for carrier in [
        __faber_rt_v1_diagnostic_mone_option,
        __faber_rt_v1_diagnostic_scribe_option,
        __faber_rt_v1_diagnostic_vide_option,
    ] {
        assert_eq!(
            unsafe { carrier(context, present, radix_host_abi::VALUE_KIND_I64) },
            STATUS_OK
        );
    }

    // Nihil option: the null handle renders `nihil` on every stream.
    for carrier in [
        __faber_rt_v1_diagnostic_mone_option,
        __faber_rt_v1_diagnostic_scribe_option,
        __faber_rt_v1_diagnostic_vide_option,
    ] {
        assert_eq!(
            unsafe { carrier(context, ptr::null_mut(), radix_host_abi::VALUE_KIND_I64) },
            STATUS_OK
        );
    }

    // Unknown non-null raw handle with an unsupported kind stays fail-closed.
    assert_eq!(
        unsafe {
            __faber_rt_v1_diagnostic_mone_option(context, present, radix_host_abi::VALUE_KIND_I8)
        },
        STATUS_UNSUPPORTED
    );

    unsafe { __faber_rt_v1_shutdown(context) };
}

/// L11 (fc9be27a): family 3 — raw null-encoded options decode the pointer bits
/// per value-kind. `est nihil` / `non est nihil` and `vel` on literal-built
/// scalar unions (`numerus ∪ nihil`) previously failed because the raw carrier
/// only accepted the pointer kind.
#[test]
fn option_raw_scalar_presence_get_and_coalesce_decode_pointer_bits() {
    let mut context = ptr::null_mut();
    let status = unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) };
    assert_eq!(status, STATUS_OK);

    // Raw scalar option: the pointer bits ARE the i64 payload (chain/literal
    // result). Presence is the non-null pointer regardless of payload kind.
    let present = 100usize as *mut c_void;
    let mut is_present = 99_u8;
    assert_eq!(
        unsafe {
            __faber_rt_v1_option_is_present(
                context,
                present,
                VALUE_KIND_I64,
                std::ptr::from_mut(&mut is_present).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(is_present, 1);

    // A null raw scalar option is absent (not an argument error).
    is_present = 99;
    assert_eq!(
        unsafe {
            __faber_rt_v1_option_is_present(
                context,
                ptr::null_mut(),
                VALUE_KIND_I64,
                std::ptr::from_mut(&mut is_present).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(is_present, 0);

    // `get` decodes the raw scalar bits into the output slot.
    let mut output = 0_i64;
    assert_eq!(
        unsafe {
            __faber_rt_v1_option_get(
                context,
                present,
                VALUE_KIND_I64,
                std::ptr::from_mut(&mut output).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(output, 100);

    // `get_or` decodes the raw scalar bits; a null raw option coalesces to the
    // fallback.
    let fallback = 9_i64;
    output = 0;
    assert_eq!(
        unsafe {
            __faber_rt_v1_option_get_or(
                context,
                present,
                VALUE_KIND_I64,
                std::ptr::from_ref(&fallback).cast(),
                std::ptr::from_mut(&mut output).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(output, 100);
    output = 0;
    assert_eq!(
        unsafe {
            __faber_rt_v1_option_get_or(
                context,
                ptr::null_mut(),
                VALUE_KIND_I64,
                std::ptr::from_ref(&fallback).cast(),
                std::ptr::from_mut(&mut output).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(output, 9);

    // A null raw scalar option still fails closed on `get` (no payload).
    assert_eq!(
        unsafe {
            __faber_rt_v1_option_get(
                context,
                ptr::null_mut(),
                VALUE_KIND_I64,
                std::ptr::from_mut(&mut output).cast(),
            )
        },
        STATUS_INVALID_ARGUMENT
    );

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn diagnostic_mone_and_vide_families_return_expected_statuses() {
    let mut context = ptr::null_mut();
    let status = unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) };
    assert_eq!(status, STATUS_OK);

    let text = FaberRtSliceV1::from_static(b"nota text");
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_mone_ptr(context, ptr::null()) },
        STATUS_UNSUPPORTED
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_mone_text(context, &raw const text) },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_mone_ascii(context, c"mone ascii".as_ptr().cast()) },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_mone_i64(context, -64) },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_vide_ptr(context, ptr::null()) },
        STATUS_UNSUPPORTED
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_vide_text(context, &raw const text) },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_vide_ascii(context, c"vide ascii".as_ptr().cast()) },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_vide_i64(context, 64) },
        STATUS_OK
    );

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn assertion_family_returns_handled_statuses() {
    let mut context = ptr::null_mut();
    let status = unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) };
    assert_eq!(status, STATUS_OK);

    assert_eq!(unsafe { __faber_rt_v1_assert(context, 1) }, STATUS_OK);
    assert_eq!(unsafe { __faber_rt_v1_assert(context, 0) }, STATUS_PANIC);
    assert_eq!(
        unsafe { __faber_rt_v1_assert_message(context, 1, FaberRtSliceV1::from_static(b"unused")) },
        STATUS_OK
    );
    assert_eq!(
        unsafe {
            __faber_rt_v1_assert_message(
                context,
                0,
                FaberRtSliceV1::from_static(b"assertion failed"),
            )
        },
        STATUS_PANIC
    );

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn format_single_substitution_scalars_renders_correct_text() {
    let mut context = ptr::null_mut();
    let status = unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) };
    assert_eq!(status, STATUS_OK);

    let one = unsafe {
        __faber_rt_v1_format_i64(context, FaberRtSliceV1::from_static("n=§".as_bytes()), 42)
    };
    let float = unsafe {
        __faber_rt_v1_format_f64(context, FaberRtSliceV1::from_static("x=§".as_bytes()), 1.5)
    };
    let boolean = unsafe {
        __faber_rt_v1_format_i1(context, FaberRtSliceV1::from_static("b=§".as_bytes()), 1)
    };

    assert_eq!(one.status, STATUS_OK);
    assert_eq!(float.status, STATUS_OK);
    assert_eq!(boolean.status, STATUS_OK);
    assert_eq!(unsafe { &*one.value.cast::<RuntimeText>() }.value, "n=42");
    assert_eq!(
        unsafe { &*float.value.cast::<RuntimeText>() }.value,
        "x=1.5"
    );
    assert_eq!(
        unsafe { &*boolean.value.cast::<RuntimeText>() }.value,
        "b=verum"
    );

    unsafe { __faber_rt_v1_shutdown(context) };
}

/// L28 (ab91f49f, W16): the f32 format carrier keeps the f32 precision —
/// `0.1f32` renders `0.1`, NOT the `0.10000000149011612` an f64-widened
/// carrier would produce — and integral f32s keep the `.0` decimal marker
/// (`display_fractus` semantics, matching the HIR-Rust lane).
#[test]
fn format_f32_keeps_f32_precision_and_decimal_marker() {
    let mut context = ptr::null_mut();
    let status = unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) };
    assert_eq!(status, STATUS_OK);

    let integral = unsafe {
        __faber_rt_v1_format_f32(context, FaberRtSliceV1::from_static("§".as_bytes()), 4.0f32)
    };
    let fractional = unsafe {
        __faber_rt_v1_format_f32(context, FaberRtSliceV1::from_static("§".as_bytes()), 0.1f32)
    };
    assert_eq!(integral.status, STATUS_OK);
    assert_eq!(fractional.status, STATUS_OK);
    assert_eq!(
        unsafe { &*integral.value.cast::<RuntimeText>() }.value,
        "4.0"
    );
    assert_eq!(
        unsafe { &*fractional.value.cast::<RuntimeText>() }.value,
        "0.1"
    );

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn format_multi_arg_and_text_wrapping_renders_combined_text() {
    let mut context = ptr::null_mut();
    let status = unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) };
    assert_eq!(status, STATUS_OK);

    let one = unsafe {
        __faber_rt_v1_format_i64(context, FaberRtSliceV1::from_static("n=§".as_bytes()), 42)
    };
    let float = unsafe {
        __faber_rt_v1_format_f64(context, FaberRtSliceV1::from_static("x=§".as_bytes()), 1.5)
    };
    let reordered = unsafe {
        __faber_rt_v1_format_i64_i64(
            context,
            FaberRtSliceV1::from_static("§1/§0/§9".as_bytes()),
            3,
            7,
        )
    };
    let paired = unsafe {
        __faber_rt_v1_format_text_text(
            context,
            FaberRtSliceV1::from_static("§ + §".as_bytes()),
            one.value.cast(),
            float.value.cast(),
        )
    };
    let single = unsafe {
        __faber_rt_v1_format_text(
            context,
            FaberRtSliceV1::from_static("[§]".as_bytes()),
            one.value.cast(),
        )
    };
    let mixed = unsafe {
        __faber_rt_v1_format_text_i64(
            context,
            FaberRtSliceV1::from_static("§:§".as_bytes()),
            one.value.cast(),
            9,
        )
    };
    let mixed_bool = unsafe {
        __faber_rt_v1_format_text_i64_i1(
            context,
            FaberRtSliceV1::from_static("§:§:§".as_bytes()),
            one.value.cast(),
            9,
            1,
        )
    };
    let three = unsafe {
        __faber_rt_v1_format_i64_i64_i64(
            context,
            FaberRtSliceV1::from_static("§/§/§".as_bytes()),
            1,
            2,
            3,
        )
    };

    assert_eq!(reordered.status, STATUS_OK);
    assert_eq!(paired.status, STATUS_OK);
    assert_eq!(single.status, STATUS_OK);
    assert_eq!(mixed.status, STATUS_OK);
    assert_eq!(mixed_bool.status, STATUS_OK);
    assert_eq!(three.status, STATUS_OK);
    assert_eq!(
        unsafe { &*reordered.value.cast::<RuntimeText>() }.value,
        "7/3/§9"
    );
    assert_eq!(
        unsafe { &*paired.value.cast::<RuntimeText>() }.value,
        "n=42 + x=1.5"
    );
    assert_eq!(
        unsafe { &*single.value.cast::<RuntimeText>() }.value,
        "[n=42]"
    );
    assert_eq!(
        unsafe { &*mixed.value.cast::<RuntimeText>() }.value,
        "n=42:9"
    );
    assert_eq!(
        unsafe { &*mixed_bool.value.cast::<RuntimeText>() }.value,
        "n=42:9:verum"
    );
    assert_eq!(
        unsafe { &*three.value.cast::<RuntimeText>() }.value,
        "1/2/3"
    );

    let mut length = -1;
    let length_status =
        unsafe { __faber_rt_v1_text_length(context, paired.value.cast(), &raw mut length) };
    assert_eq!(length_status, STATUS_OK);
    assert_eq!(length, 12);

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn format_invalid_utf8_pattern_returns_invalid_argument() {
    let mut context = ptr::null_mut();
    let status = unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) };
    assert_eq!(status, STATUS_OK);

    let invalid =
        unsafe { __faber_rt_v1_format_i64(context, FaberRtSliceV1::from_static(&[0xff]), 42) };
    assert_eq!(
        invalid,
        FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT)
    );

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn text_query_family_is_empty_contains_starts_ends_works() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let text = FaberRtSliceV1::from_static("  Rōma/AVĒ  ".as_bytes());
    let empty = FaberRtSliceV1::from_static(b"");
    let roma = FaberRtSliceV1::from_static("Rōma".as_bytes());
    let mut answer = 0;

    assert_eq!(
        unsafe { __faber_rt_v1_text_is_empty(context, &raw const empty, &raw mut answer) },
        STATUS_OK
    );
    assert_eq!(answer, 1);
    assert_eq!(
        unsafe {
            __faber_rt_v1_text_contains(context, &raw const text, &raw const roma, &raw mut answer)
        },
        STATUS_OK
    );
    assert_eq!(answer, 1);
    assert_eq!(
        unsafe {
            __faber_rt_v1_text_starts_with(
                context,
                &raw const text,
                &raw const empty,
                &raw mut answer,
            )
        },
        STATUS_OK
    );
    assert_eq!(answer, 1);
    assert_eq!(
        unsafe {
            __faber_rt_v1_text_ends_with(
                context,
                &raw const text,
                &raw const empty,
                &raw mut answer,
            )
        },
        STATUS_OK
    );
    assert_eq!(answer, 1);

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn text_transform_family_trim_lower_upper_slice_replace_split_works() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let text = FaberRtSliceV1::from_static("  Rōma/AVĒ  ".as_bytes());
    let roma = FaberRtSliceV1::from_static("Rōma".as_bytes());
    let slash = FaberRtSliceV1::from_static(b"/");
    let ave = FaberRtSliceV1::from_static("AVĒ".as_bytes());

    let trimmed = unsafe { __faber_rt_v1_text_trim(context, &raw const text) };
    let lower = unsafe { __faber_rt_v1_text_lowercase(context, trimmed.value.cast()) };
    let upper = unsafe { __faber_rt_v1_text_uppercase(context, lower.value.cast()) };
    let sliced = unsafe { __faber_rt_v1_text_slice(context, trimmed.value.cast(), 1, 5) };
    let replaced = unsafe {
        __faber_rt_v1_text_replace(
            context,
            trimmed.value.cast(),
            &raw const ave,
            &raw const roma,
        )
    };
    let split_result =
        unsafe { __faber_rt_v1_text_split(context, trimmed.value.cast(), &raw const slash) };

    for result in [trimmed, lower, upper, sliced, replaced, split_result] {
        assert_eq!(result.status, STATUS_OK);
    }
    assert_eq!(
        unsafe { &*trimmed.value.cast::<RuntimeText>() }.value,
        "Rōma/AVĒ"
    );
    assert_eq!(
        unsafe { &*lower.value.cast::<RuntimeText>() }.value,
        "rōma/avē"
    );
    assert_eq!(
        unsafe { &*upper.value.cast::<RuntimeText>() }.value,
        "RŌMA/AVĒ"
    );
    assert_eq!(
        unsafe { &*sliced.value.cast::<RuntimeText>() }.value,
        "ōma/"
    );
    assert_eq!(
        unsafe { &*replaced.value.cast::<RuntimeText>() }.value,
        "Rōma/Rōma"
    );
    let split = unsafe { &*split_result.value.cast::<RuntimeArray>() };
    assert_eq!(split.values.len(), 2);
    let parts = split
        .values
        .iter()
        .map(|value| match value {
            array::RuntimeValue::Ptr(value) => {
                unsafe { &*value.cast::<RuntimeText>() }.value.as_str()
            }
            _ => panic!("split produced non-text carrier"),
        })
        .collect::<Vec<_>>();
    assert_eq!(parts, ["Rōma", "AVĒ"]);

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn text_slice_negative_bounds_return_invalid_argument() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let text = FaberRtSliceV1::from_static("  Rōma/AVĒ  ".as_bytes());
    let trimmed = unsafe { __faber_rt_v1_text_trim(context, &raw const text) };

    assert_eq!(
        unsafe { __faber_rt_v1_text_slice(context, trimmed.value.cast(), -1, 2) }.status,
        STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe { __faber_rt_v1_text_slice(context, trimmed.value.cast(), 0, -1) }.status,
        STATUS_INVALID_ARGUMENT
    );

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn text_concat_family_owns_contextual_text_result() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let first = FaberRtSliceV1::from_static("salve".as_bytes());
    let second = FaberRtSliceV1::from_static(" munde".as_bytes());
    let result = unsafe { __faber_rt_v1_text_concat(context, &raw const first, &raw const second) };
    assert!(result.status.is_ok());
    assert_eq!(
        unsafe { &*result.value.cast::<RuntimeText>() }.value,
        "salve munde"
    );
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn text_scalar_conversion_family_honors_width_radix_recovery_status() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let hex = FaberRtSliceV1::from_static(b"ff");
    let negative = FaberRtSliceV1::from_static(b"-8");
    let decimal = FaberRtSliceV1::from_static(b"1.25");
    let invalid = FaberRtSliceV1::from_static(b"invalid");
    let empty = FaberRtSliceV1::from_static(b"");
    let mut i32_value = 0i32;
    let mut i8_value = 0i8;
    let mut i64_value = 0i64;
    let mut f64_value = 0.0f64;
    let mut truthy = 1u8;

    assert_eq!(
        unsafe {
            __faber_rt_v1_text_parse_integer(
                context,
                &raw const hex,
                16,
                VALUE_KIND_I32,
                std::ptr::from_mut(&mut i32_value).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(i32_value, 255);
    assert_eq!(
        unsafe {
            __faber_rt_v1_text_parse_integer(
                context,
                &raw const negative,
                10,
                VALUE_KIND_I8,
                std::ptr::from_mut(&mut i8_value).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(i8_value, -8);
    assert_eq!(
        unsafe {
            __faber_rt_v1_text_parse_float(
                context,
                &raw const decimal,
                VALUE_KIND_F64,
                std::ptr::from_mut(&mut f64_value).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(f64_value, 1.25);
    assert_eq!(
        unsafe {
            __faber_rt_v1_text_parse_integer(
                context,
                &raw const invalid,
                10,
                VALUE_KIND_I64,
                std::ptr::from_mut(&mut i64_value).cast(),
            )
        },
        STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe { __faber_rt_v1_text_truthy(context, &raw const empty, &raw mut truthy) },
        STATUS_OK
    );
    assert_eq!(truthy, 0);
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn scalar_text_conversion_family_preserves_rust_conversion_spellings() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );

    let zero = unsafe { __faber_rt_v1_text_f64(context, 0.0) };
    let truth = unsafe { __faber_rt_v1_text_i1(context, 1) };
    let falsehood = unsafe { __faber_rt_v1_text_i1(context, 0) };

    assert_eq!(zero.status, STATUS_OK);
    assert_eq!(truth.status, STATUS_OK);
    assert_eq!(falsehood.status, STATUS_OK);
    assert_eq!(unsafe { &*zero.value.cast::<RuntimeText>() }.value, "0.0");
    assert_eq!(unsafe { &*truth.value.cast::<RuntimeText>() }.value, "true");
    assert_eq!(
        unsafe { &*falsehood.value.cast::<RuntimeText>() }.value,
        "false"
    );

    let empty = c"";
    let nonempty = c"yes";
    let mut answer = 0;
    assert_eq!(
        unsafe { __faber_rt_v1_ascii_truthy(context, empty.as_ptr(), &raw mut answer) },
        STATUS_OK
    );
    assert_eq!(answer, 0);
    assert_eq!(
        unsafe { __faber_rt_v1_ascii_truthy(context, nonempty.as_ptr(), &raw mut answer) },
        STATUS_OK
    );
    assert_eq!(answer, 1);
    assert_eq!(
        unsafe { __faber_rt_v1_ascii_truthy(context, ptr::null(), &raw mut answer) },
        STATUS_INVALID_ARGUMENT
    );

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn typed_map_preserves_value_semantics() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let map = unsafe { __faber_rt_v1_map_new(context, VALUE_KIND_TEXT, VALUE_KIND_I64) };
    assert_eq!(map.status, STATUS_OK);
    let first_key = FaberRtSliceV1::from_static("aelia".as_bytes());
    let equal_key = FaberRtSliceV1::from_static("aelia".as_bytes());
    let missing_key = FaberRtSliceV1::from_static("balbus".as_bytes());
    let first_handle = std::ptr::from_ref(&first_key).cast_mut().cast::<c_void>();
    let equal_handle = std::ptr::from_ref(&equal_key).cast_mut().cast::<c_void>();
    let missing_handle = std::ptr::from_ref(&missing_key).cast_mut().cast::<c_void>();
    let value = 95i64;
    assert_eq!(
        unsafe {
            __faber_rt_v1_map_put(
                context,
                map.value,
                VALUE_KIND_TEXT,
                std::ptr::from_ref(&first_handle).cast(),
                VALUE_KIND_I64,
                std::ptr::from_ref(&value).cast(),
            )
        },
        STATUS_OK
    );
    let mut answer = 0u8;
    assert_eq!(
        unsafe {
            __faber_rt_v1_map_contains(
                context,
                map.value,
                VALUE_KIND_TEXT,
                std::ptr::from_ref(&equal_handle).cast(),
                &raw mut answer,
            )
        },
        STATUS_OK
    );
    assert_eq!(
        answer, 1,
        "distinct text descriptors compare by UTF-8 value"
    );
    let present = unsafe {
        __faber_rt_v1_map_option(
            context,
            map.value,
            VALUE_KIND_TEXT,
            std::ptr::from_ref(&equal_handle).cast(),
            VALUE_KIND_I64,
        )
    };
    let missing = unsafe {
        __faber_rt_v1_map_option(
            context,
            map.value,
            VALUE_KIND_TEXT,
            std::ptr::from_ref(&missing_handle).cast(),
            VALUE_KIND_I64,
        )
    };
    assert!(unsafe { &*present.value.cast::<RuntimeOption>() }
        .value
        .is_some());
    assert!(unsafe { &*missing.value.cast::<RuntimeOption>() }
        .value
        .is_none());
    let mut length = 0i64;
    assert_eq!(
        unsafe { __faber_rt_v1_map_length(context, map.value, &raw mut length) },
        STATUS_OK
    );
    assert_eq!(length, 1);
    let keys = unsafe { __faber_rt_v1_map_keys(context, map.value) };
    let values = unsafe { __faber_rt_v1_map_values(context, map.value) };
    // L19: key snapshots are stored with the canonical element kind array
    // consumers use (`array_get`/`array_set` reject a kind mismatch). The
    // emitter canonicalizes every pointer-carried element (textus/ascii/
    // valor/instans/octeti …) to VALUE_KIND_PTR, so a `tabula<textus, T>`
    // key snapshot is VALUE_KIND_PTR — the raw VALUE_KIND_TEXT made
    // `itera de <tabula>` fail every element read.
    assert_eq!(
        unsafe { &*keys.value.cast::<RuntimeArray>() }.kind,
        VALUE_KIND_PTR
    );
    assert_eq!(
        unsafe { &*values.value.cast::<RuntimeArray>() }.kind,
        VALUE_KIND_I64
    );
    assert_eq!(
        unsafe { &*keys.value.cast::<RuntimeArray>() }.values.len(),
        1
    );
    assert_eq!(
        unsafe { &*values.value.cast::<RuntimeArray>() }
            .values
            .len(),
        1
    );
    assert_eq!(
        unsafe {
            __faber_rt_v1_map_delete(
                context,
                map.value,
                VALUE_KIND_TEXT,
                std::ptr::from_ref(&equal_handle).cast(),
                &raw mut answer,
            )
        },
        STATUS_OK
    );
    assert_eq!(answer, 1);
    assert_eq!(
        unsafe { __faber_rt_v1_map_is_empty(context, map.value, &raw mut answer) },
        STATUS_OK
    );
    assert_eq!(answer, 1);

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn typed_set_preserves_value_semantics() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let left = unsafe { __faber_rt_v1_set_new(context, VALUE_KIND_I64) };
    let right = unsafe { __faber_rt_v1_set_new(context, VALUE_KIND_I64) };
    for (set, values) in [
        (left.value, &[1i64, 2, 3][..]),
        (right.value, &[2i64, 4][..]),
    ] {
        for value in values {
            assert_eq!(
                unsafe {
                    __faber_rt_v1_set_add(
                        context,
                        set,
                        VALUE_KIND_I64,
                        std::ptr::from_ref(value).cast(),
                    )
                },
                STATUS_OK
            );
        }
    }
    let union = unsafe { __faber_rt_v1_set_union(context, left.value, right.value) };
    let intersection = unsafe { __faber_rt_v1_set_intersection(context, left.value, right.value) };
    let difference = unsafe { __faber_rt_v1_set_difference(context, left.value, right.value) };
    let symmetric =
        unsafe { __faber_rt_v1_set_symmetric_difference(context, left.value, right.value) };
    assert_eq!(
        unsafe { &*union.value.cast::<RuntimeSet>() }.values.len(),
        4
    );
    assert_eq!(
        unsafe { &*intersection.value.cast::<RuntimeSet>() }
            .values
            .len(),
        1
    );
    assert_eq!(
        unsafe { &*difference.value.cast::<RuntimeSet>() }
            .values
            .len(),
        2
    );
    assert_eq!(
        unsafe { &*symmetric.value.cast::<RuntimeSet>() }
            .values
            .len(),
        3
    );
    let mut answer = 0u8;
    assert_eq!(
        unsafe {
            __faber_rt_v1_set_is_subset(context, intersection.value, union.value, &raw mut answer)
        },
        STATUS_OK
    );
    assert_eq!(answer, 1);
    assert_eq!(
        unsafe {
            __faber_rt_v1_set_is_superset(context, union.value, intersection.value, &raw mut answer)
        },
        STATUS_OK
    );
    assert_eq!(answer, 1);
    let two = 2i64;
    assert_eq!(
        unsafe {
            __faber_rt_v1_set_contains(
                context,
                left.value,
                VALUE_KIND_I64,
                std::ptr::from_ref(&two).cast(),
                &raw mut answer,
            )
        },
        STATUS_OK
    );
    assert_eq!(answer, 1);
    assert_eq!(
        unsafe {
            __faber_rt_v1_set_delete(
                context,
                left.value,
                VALUE_KIND_I64,
                std::ptr::from_ref(&two).cast(),
                &raw mut answer,
            )
        },
        STATUS_OK
    );
    let mut length = 0i64;
    assert_eq!(
        unsafe { __faber_rt_v1_set_length(context, left.value, &raw mut length) },
        STATUS_OK
    );
    assert_eq!(length, 2);
    assert_eq!(
        unsafe { __faber_rt_v1_set_is_empty(context, left.value, &raw mut answer) },
        STATUS_OK
    );
    assert_eq!(answer, 0);

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn scalar_text_conversion_family_owns_canonical_values() {
    let mut context = ptr::null_mut();
    let status = unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) };
    assert_eq!(status, STATUS_OK);

    let integer = unsafe { __faber_rt_v1_text_i64(context, -42) };
    let float = unsafe { __faber_rt_v1_text_f64(context, 3.25) };
    let boolean = unsafe { __faber_rt_v1_text_i1(context, 1) };
    let false_boolean = unsafe { __faber_rt_v1_text_i1(context, 0) };

    assert_eq!(integer.status, STATUS_OK);
    assert_eq!(float.status, STATUS_OK);
    assert_eq!(boolean.status, STATUS_OK);
    assert_eq!(false_boolean.status, STATUS_OK);
    assert_eq!(
        unsafe { &*integer.value.cast::<RuntimeText>() }.value,
        "-42"
    );
    assert_eq!(unsafe { &*float.value.cast::<RuntimeText>() }.value, "3.25");
    assert_eq!(
        unsafe { &*boolean.value.cast::<RuntimeText>() }.value,
        "true"
    );
    assert_eq!(
        unsafe { &*false_boolean.value.cast::<RuntimeText>() }.value,
        "false"
    );

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn scalar_valor_conversion_family_owns_typed_values() {
    let mut context = ptr::null_mut();
    let status = unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) };
    assert_eq!(status, STATUS_OK);

    let integer = unsafe { __faber_rt_v1_valor_i64(context, -42) };
    let float = unsafe { __faber_rt_v1_valor_f64(context, 3.25) };
    let boolean = unsafe { __faber_rt_v1_valor_i1(context, 1) };

    assert_eq!(
        unsafe { &*integer.value.cast::<Valor>() },
        &Valor::Numerus(-42)
    );
    assert_eq!(
        unsafe { &*float.value.cast::<Valor>() },
        &Valor::Fractus(3.25)
    );
    assert_eq!(
        unsafe { &*boolean.value.cast::<Valor>() },
        &Valor::Bivalens(true)
    );

    let text = FaberRtSliceV1::from_static(b"salve");
    let boxed_text = unsafe { __faber_rt_v1_valor_text(context, &raw const text) };
    let boxed_ascii = unsafe { __faber_rt_v1_valor_ascii(context, c"roma".as_ptr()) };
    let boxed_nihil = unsafe { __faber_rt_v1_valor_nihil(context) };
    assert_eq!(
        unsafe { &*boxed_text.value.cast::<Valor>() },
        &Valor::Textus("salve".into())
    );
    assert_eq!(
        unsafe { &*boxed_ascii.value.cast::<Valor>() },
        &Valor::Textus("roma".into())
    );
    assert_eq!(
        unsafe { &*boxed_nihil.value.cast::<Valor>() },
        &Valor::Nihil
    );

    let mut integer_out = 0;
    let mut float_out = 0.0;
    let mut boolean_out = 0;
    assert_eq!(
        unsafe { __faber_rt_v1_valor_get_i64(context, integer.value.cast(), &raw mut integer_out) },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_rt_v1_valor_get_f64(context, integer.value.cast(), &raw mut float_out) },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_rt_v1_valor_get_i1(context, boolean.value.cast(), &raw mut boolean_out) },
        STATUS_OK
    );
    assert_eq!((integer_out, float_out, boolean_out), (-42, -42.0, 1));

    let extracted_text = unsafe { __faber_rt_v1_valor_get_text(context, boxed_text.value.cast()) };
    let descriptor = unsafe { &*extracted_text.value.cast::<FaberRtSliceV1>() };
    assert_eq!(
        unsafe { std::slice::from_raw_parts(descriptor.data, descriptor.len as usize) },
        b"salve"
    );
    let extracted_ascii =
        unsafe { __faber_rt_v1_valor_get_ascii(context, boxed_ascii.value.cast()) };
    assert_eq!(
        unsafe { CStr::from_ptr(extracted_ascii.value.cast()) }.to_bytes(),
        b"roma"
    );
    assert_eq!(
        unsafe { __faber_rt_v1_valor_get_nihil(context, boxed_nihil.value.cast()) },
        STATUS_OK
    );

    let mismatch = unsafe {
        __faber_rt_v1_valor_get_i64(context, boxed_text.value.cast(), &raw mut integer_out)
    };
    assert_eq!(mismatch, STATUS_INVALID_ARGUMENT);
    let foreign = unsafe {
        __faber_rt_v1_valor_get_i64(context, ptr::dangling::<Valor>(), &raw mut integer_out)
    };
    assert_eq!(foreign, STATUS_INVALID_ARGUMENT);

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn valor_octeti_round_trip_preserves_bytes() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );

    let bytes = FaberRtSliceV1::from_static(&[0xde, 0xad]);
    let octeti = unsafe { __faber_rt_v1_octeti_new(context, &raw const bytes) };
    let octeti_valor = unsafe { __faber_rt_v1_valor_octeti(context, octeti.value) };
    assert_eq!(
        unsafe { &*octeti_valor.value.cast::<Valor>() },
        &Valor::Octeti(vec![0xde, 0xad])
    );
    let octeti_again =
        unsafe { __faber_rt_v1_valor_get_octeti(context, octeti_valor.value.cast()) };
    assert_eq!(
        unsafe { &*octeti_again.value.cast::<Vec<u8>>() },
        &[0xde, 0xad]
    );

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn valor_array_round_trip_preserves_elements() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );

    let array = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    for value in [1_i64, 2] {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    array.value,
                    VALUE_KIND_I64,
                    ptr::from_ref(&value).cast(),
                )
            },
            STATUS_OK
        );
    }
    let array_valor = unsafe { __faber_rt_v1_valor_array(context, array.value) };
    assert_eq!(
        unsafe { &*array_valor.value.cast::<Valor>() },
        &Valor::Lista(vec![Valor::Numerus(1), Valor::Numerus(2)])
    );
    let array_again =
        unsafe { __faber_rt_v1_valor_get_array(context, array_valor.value.cast(), VALUE_KIND_I64) };
    for (index, expected) in [1_i64, 2].into_iter().enumerate() {
        let mut actual = 0_i64;
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_get(
                    context,
                    array_again.value,
                    index as i64,
                    VALUE_KIND_I64,
                    ptr::from_mut(&mut actual).cast(),
                )
            },
            STATUS_OK
        );
        assert_eq!(actual, expected);
    }

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn valor_array_text_elements_round_trip_preserves_textus() {
    // L27 (d31792f5): the array element kind for textus is VALUE_KIND_PTR
    // (`runtime_value_abi`), so `↦ valor` of a `lista<textus>` stored raw
    // handle pointers and `valor_array` previously failed to decode them
    // (STATUS_INVALID_ARGUMENT), latching a nonzero process exit for
    // stdout-correct programs (est/est exit-code row). Ptr-kind elements now
    // resolve arena text / literal descriptors / nested aggregates.
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );

    let array = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_PTR) };
    // Distinct live descriptors with stable addresses (stack locals kept
    // alive past the pushes; a Vec would reallocate and move them).
    let prima = FaberRtSliceV1::from_static(b"prima");
    let secunda = FaberRtSliceV1::from_static(b"secunda");
    let handles = [
        ptr::from_ref(&prima).cast::<c_void>(),
        ptr::from_ref(&secunda).cast::<c_void>(),
    ];
    for handle in &handles {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    array.value,
                    VALUE_KIND_PTR,
                    ptr::from_ref(handle).cast(),
                )
            },
            STATUS_OK
        );
    }
    let array_valor = unsafe { __faber_rt_v1_valor_array(context, array.value) };
    assert!(
        array_valor.status.is_ok(),
        "lista<textus> ↦ valor must succeed: {:?}",
        array_valor.status
    );
    assert_eq!(
        unsafe { &*array_valor.value.cast::<Valor>() },
        &Valor::Lista(vec![
            Valor::Textus("prima".to_owned()),
            Valor::Textus("secunda".to_owned())
        ])
    );

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn valor_map_round_trip_preserves_entries() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );

    let map = unsafe { __faber_rt_v1_map_new(context, VALUE_KIND_TEXT, VALUE_KIND_I64) };
    let key = FaberRtSliceV1::from_static(b"alpha");
    let key_handle = ptr::from_ref(&key).cast_mut().cast::<c_void>();
    let value = 10_i64;
    assert_eq!(
        unsafe {
            __faber_rt_v1_map_put(
                context,
                map.value,
                VALUE_KIND_TEXT,
                ptr::from_ref(&key_handle).cast(),
                VALUE_KIND_I64,
                ptr::from_ref(&value).cast(),
            )
        },
        STATUS_OK
    );
    let map_valor = unsafe { __faber_rt_v1_valor_map(context, map.value) };
    let mut expected = std::collections::BTreeMap::new();
    expected.insert("alpha".to_owned(), Valor::Numerus(10));
    assert_eq!(
        unsafe { &*map_valor.value.cast::<Valor>() },
        &Valor::Tabula(expected)
    );
    let map_again = unsafe {
        __faber_rt_v1_valor_get_map(
            context,
            map_valor.value.cast(),
            VALUE_KIND_TEXT,
            VALUE_KIND_I64,
        )
    };
    let map_again = unsafe { &*map_again.value.cast::<RuntimeMap>() };
    assert_eq!(map_again.entries.len(), 1);

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn octeti_family_mutates_indexes_and_converts_text() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let text = FaberRtSliceV1::from_static(b"hi");
    let bytes = unsafe { __faber_rt_v1_octeti_from_text(context, &raw const text) };
    assert_eq!(
        unsafe { __faber_rt_v1_octeti_append(context, bytes.value, b'!') },
        STATUS_OK
    );
    let mut length = 0_i64;
    assert_eq!(
        unsafe { __faber_rt_v1_octeti_length(context, bytes.value, &raw mut length) },
        STATUS_OK
    );
    assert_eq!(length, 3);
    let last = unsafe { __faber_rt_v1_octeti_get(context, bytes.value, 2) };
    let mut value = 0_u8;
    assert_eq!(
        unsafe {
            __faber_rt_v1_option_get(
                context,
                last.value,
                VALUE_KIND_U8,
                ptr::from_mut(&mut value).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(value, b'!');
    let decoded = unsafe { __faber_rt_v1_octeti_get_text(context, bytes.value) };
    let decoded = unsafe { &*decoded.value.cast::<FaberRtSliceV1>() };
    assert_eq!(
        unsafe { std::slice::from_raw_parts(decoded.data, decoded.len as usize) },
        b"hi!"
    );

    let ascii = unsafe { __faber_rt_v1_octeti_from_ascii(context, c"SPQR".as_ptr()) };
    let decoded = unsafe { __faber_rt_v1_octeti_get_ascii(context, ascii.value) };
    assert_eq!(
        unsafe { CStr::from_ptr(decoded.value.cast()) }.to_bytes(),
        b"SPQR"
    );
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn instans_family_preserves_precision_and_valor_provenance() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let wire = FaberRtSliceV1::from_static(b"1979-05-27T07:32:00.123456Z");
    let micros = unsafe {
        __faber_rt_v1_instans_from_text(context, &raw const wire, INSTANS_PRECISION_MICROS)
    };
    let millis =
        unsafe { __faber_rt_v1_instans_retag(context, micros.value, INSTANS_PRECISION_MILLIS) };
    let rendered = unsafe { __faber_rt_v1_instans_get_text(context, millis.value) };
    let rendered = unsafe { &*rendered.value.cast::<FaberRtSliceV1>() };
    assert_eq!(
        unsafe { std::slice::from_raw_parts(rendered.data, rendered.len as usize) },
        b"1979-05-27T07:32:00.123Z"
    );
    let valor = unsafe { __faber_rt_v1_valor_text(context, &raw const wire) };
    let seconds = unsafe {
        __faber_rt_v1_instans_from_valor(context, valor.value.cast(), INSTANS_PRECISION_SECONDS)
    };
    let rendered = unsafe { __faber_rt_v1_instans_get_text(context, seconds.value) };
    let rendered = unsafe { &*rendered.value.cast::<FaberRtSliceV1>() };
    assert_eq!(
        unsafe { std::slice::from_raw_parts(rendered.data, rendered.len as usize) },
        b"1979-05-27T07:32:00Z"
    );
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn genus_valor_field_table_boxes_and_extracts_atomically() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let name_text = FaberRtSliceV1::from_static(b"name");
    let age_text = FaberRtSliceV1::from_static(b"age");
    let code_text = FaberRtSliceV1::from_static(b"code");
    let names = [
        ptr::from_ref(&name_text),
        ptr::from_ref(&age_text),
        ptr::from_ref(&code_text),
    ];
    let kinds = [VALUE_KIND_TEXT, VALUE_KIND_I64, VALUE_KIND_ASCII];
    let name_value = FaberRtSliceV1::from_static(b"Marcus");
    let name_handle = ptr::from_ref(&name_value).cast_mut().cast::<c_void>();
    let age = 42_i64;
    let code = c"SPQR".as_ptr().cast_mut().cast::<c_void>();
    let values = [
        ptr::from_ref(&name_handle).cast(),
        ptr::from_ref(&age).cast(),
        ptr::from_ref(&code).cast(),
    ];
    let boxed = unsafe {
        __faber_rt_v1_valor_genus(context, 3, names.as_ptr(), kinds.as_ptr(), values.as_ptr())
    };
    let mut expected = std::collections::BTreeMap::new();
    expected.insert("age".to_owned(), Valor::Numerus(42));
    expected.insert("name".to_owned(), Valor::Textus("Marcus".to_owned()));
    expected.insert("code".to_owned(), Valor::Textus("SPQR".to_owned()));
    assert_eq!(
        unsafe { &*boxed.value.cast::<Valor>() },
        &Valor::Tabula(expected)
    );

    let mut extracted_name: *mut c_void = ptr::null_mut();
    let mut extracted_age = 0_i64;
    let mut extracted_code: *mut c_void = ptr::null_mut();
    let outputs = [
        ptr::from_mut(&mut extracted_name).cast(),
        ptr::from_mut(&mut extracted_age).cast(),
        ptr::from_mut(&mut extracted_code).cast(),
    ];
    let defaultable = [0_u8, 0, 0];
    assert_eq!(
        unsafe {
            __faber_rt_v1_valor_get_genus(
                context,
                boxed.value.cast(),
                3,
                names.as_ptr(),
                kinds.as_ptr(),
                defaultable.as_ptr(),
                outputs.as_ptr(),
            )
        },
        STATUS_OK
    );
    let extracted_name = unsafe { &*extracted_name.cast::<FaberRtSliceV1>() };
    assert_eq!(
        unsafe { std::slice::from_raw_parts(extracted_name.data, extracted_name.len as usize) },
        b"Marcus"
    );
    assert_eq!(extracted_age, 42);
    assert_eq!(
        unsafe { CStr::from_ptr(extracted_code.cast()) }.to_bytes(),
        b"SPQR"
    );

    let missing_name = FaberRtSliceV1::from_static(b"missing");
    let missing_names = [ptr::from_ref(&missing_name)];
    let missing_kinds = [VALUE_KIND_I64];
    let missing_defaultable = [1_u8];
    let mut retained = 7_i64;
    let missing_outputs = [ptr::from_mut(&mut retained).cast()];
    assert_eq!(
        unsafe {
            __faber_rt_v1_valor_get_genus(
                context,
                boxed.value.cast(),
                1,
                missing_names.as_ptr(),
                missing_kinds.as_ptr(),
                missing_defaultable.as_ptr(),
                missing_outputs.as_ptr(),
            )
        },
        STATUS_OK
    );
    assert_eq!(retained, 7);
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn valor_genus_defaulted_extraction_matches_valor_genus_fixture() {
    // P7 (promotion packet valor-genus-field-layout): the
    // `conversio/valor-genus.fab` fixture's expected behavior through the
    // host ABI. The `Persona` genus has a mandatory field (`nomen`), a
    // sponte field (`aetas optional` → nihil seed), a declared default
    // (`regio = "Roma"`), and a mandatory instant (`born`); the payload
    // omits `aetas`/`regio` and carries an extra ignored key. The
    // `valor_get_genus` row must keep pre-seeded output slots on
    // DEFAULTABLE-policy missing keys, fail the whole extraction (the `⇥`
    // recovery latch) on a MANDATORY-policy missing key, and ignore keys
    // outside the descriptor table.
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );

    let nomen_text = FaberRtSliceV1::from_static(b"nomen");
    let aetas_text = FaberRtSliceV1::from_static(b"aetas");
    let regio_text = FaberRtSliceV1::from_static(b"regio");
    let born_text = FaberRtSliceV1::from_static(b"born");
    let extra_text = FaberRtSliceV1::from_static(b"extra");
    let persona_names = [
        ptr::from_ref(&nomen_text),
        ptr::from_ref(&aetas_text),
        ptr::from_ref(&regio_text),
        ptr::from_ref(&born_text),
    ];
    let persona_kinds = [
        VALUE_KIND_TEXT,
        radix_host_abi::VALUE_KIND_OPTION_I64,
        VALUE_KIND_TEXT,
        radix_host_abi::VALUE_KIND_INSTANS,
    ];
    let persona_policies = [
        radix_host_abi::GENUS_FIELD_POLICY_MANDATORY,
        radix_host_abi::GENUS_FIELD_POLICY_DEFAULTABLE,
        radix_host_abi::GENUS_FIELD_POLICY_DEFAULTABLE,
        radix_host_abi::GENUS_FIELD_POLICY_MANDATORY,
    ];

    // `good` payload: `{"nomen": "Marcus", "born": "1979-05-27T07:32:00Z",
    // "extra": "ignored"}` — `aetas` and `regio` keys are absent, `extra`
    // is outside the descriptor table and must be ignored.
    let nomen_value = FaberRtSliceV1::from_static(b"Marcus");
    let nomen_handle = ptr::from_ref(&nomen_value).cast_mut().cast::<c_void>();
    let born_wire = FaberRtSliceV1::from_static(b"1979-05-27T07:32:00Z");
    let born = unsafe {
        __faber_rt_v1_instans_from_text(context, &raw const born_wire, INSTANS_PRECISION_SECONDS)
    };
    assert_eq!(born.status, STATUS_OK);
    let born_handle = born.value;
    let extra_value = FaberRtSliceV1::from_static(b"ignored");
    let extra_handle = ptr::from_ref(&extra_value).cast_mut().cast::<c_void>();
    let good_names = [
        ptr::from_ref(&nomen_text),
        ptr::from_ref(&born_text),
        ptr::from_ref(&extra_text),
    ];
    let good_kinds = [
        VALUE_KIND_TEXT,
        radix_host_abi::VALUE_KIND_INSTANS,
        VALUE_KIND_TEXT,
    ];
    let good_values = [
        ptr::from_ref(&nomen_handle).cast(),
        ptr::from_ref(&born_handle).cast(),
        ptr::from_ref(&extra_handle).cast(),
    ];
    let good = unsafe {
        __faber_rt_v1_valor_genus(
            context,
            3,
            good_names.as_ptr(),
            good_kinds.as_ptr(),
            good_values.as_ptr(),
        )
    };
    assert_eq!(good.status, STATUS_OK);

    // Extraction with pre-seeded construction-default slots: `aetas` seeded
    // nihil, `regio` seeded "Roma".
    let regio_seed = FaberRtSliceV1::from_static(b"Roma");
    let mut out_nomen: *mut c_void = ptr::null_mut();
    let mut out_aetas: *mut c_void = ptr::null_mut();
    let mut out_regio: *mut c_void = ptr::from_ref(&regio_seed).cast_mut().cast();
    let mut out_born: *mut c_void = ptr::null_mut();
    let outputs = [
        ptr::from_mut(&mut out_nomen).cast(),
        ptr::from_mut(&mut out_aetas).cast(),
        ptr::from_mut(&mut out_regio).cast(),
        ptr::from_mut(&mut out_born).cast(),
    ];
    assert_eq!(
        unsafe {
            __faber_rt_v1_valor_get_genus(
                context,
                good.value.cast(),
                4,
                persona_names.as_ptr(),
                persona_kinds.as_ptr(),
                persona_policies.as_ptr(),
                outputs.as_ptr(),
            )
        },
        STATUS_OK
    );
    let extracted_nomen = unsafe { &*out_nomen.cast::<FaberRtSliceV1>() };
    assert_eq!(
        unsafe { std::slice::from_raw_parts(extracted_nomen.data, extracted_nomen.len as usize) },
        b"Marcus"
    );
    assert!(out_aetas.is_null(), "sponte `aetas` keeps its nihil seed");
    let extracted_regio = unsafe { &*out_regio.cast::<FaberRtSliceV1>() };
    assert_eq!(
        unsafe { std::slice::from_raw_parts(extracted_regio.data, extracted_regio.len as usize) },
        b"Roma",
        "missing `regio` keeps the declared default seed"
    );
    let rendered = unsafe { __faber_rt_v1_instans_get_text(context, out_born) };
    assert_eq!(rendered.status, STATUS_OK);
    let rendered = unsafe { &*rendered.value.cast::<FaberRtSliceV1>() };
    assert_eq!(
        unsafe { std::slice::from_raw_parts(rendered.data, rendered.len as usize) },
        b"1979-05-27T07:32:00Z"
    );

    // `stale ↦ Persona` failure: `{"nomen": "Livia"}` lacks the mandatory
    // `born` key, so the whole extraction fails (the `⇥` recovery latch).
    let livia_value = FaberRtSliceV1::from_static(b"Livia");
    let livia_handle = ptr::from_ref(&livia_value).cast_mut().cast::<c_void>();
    let stale_names = [ptr::from_ref(&nomen_text)];
    let stale_kinds = [VALUE_KIND_TEXT];
    let stale_values = [ptr::from_ref(&livia_handle).cast()];
    let stale = unsafe {
        __faber_rt_v1_valor_genus(
            context,
            1,
            stale_names.as_ptr(),
            stale_kinds.as_ptr(),
            stale_values.as_ptr(),
        )
    };
    assert_eq!(stale.status, STATUS_OK);
    let stale_outputs = [
        ptr::from_mut(&mut out_nomen).cast(),
        ptr::from_mut(&mut out_aetas).cast(),
        ptr::from_mut(&mut out_regio).cast(),
        ptr::from_mut(&mut out_born).cast(),
    ];
    assert_eq!(
        unsafe {
            __faber_rt_v1_valor_get_genus(
                context,
                stale.value.cast(),
                4,
                persona_names.as_ptr(),
                persona_kinds.as_ptr(),
                persona_policies.as_ptr(),
                stale_outputs.as_ptr(),
            )
        },
        STATUS_INVALID_ARGUMENT,
        "missing mandatory `born` must fail the whole genus extraction"
    );

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn array_family_round_trips_every_value_kind_and_spreads() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );

    let i1 = 1_u8;
    let i8_value = -8_i8;
    let i16_value = -16_i16;
    let i32_value = -32_i32;
    let i64_value = -64_i64;
    let u8_value = 8_u8;
    let u16_value = 16_u16;
    let u32_value = 32_u32;
    let u64_value = 64_u64;
    let f16_value = 0x3c00_u16;
    let f32_value = 3.25_f32;
    let f64_value = 6.5_f64;
    let pointer_value = context.cast::<std::ffi::c_void>();
    let cases = [
        (VALUE_KIND_I1, std::ptr::from_ref(&i1).cast()),
        (VALUE_KIND_I8, std::ptr::from_ref(&i8_value).cast()),
        (VALUE_KIND_I16, std::ptr::from_ref(&i16_value).cast()),
        (VALUE_KIND_I32, std::ptr::from_ref(&i32_value).cast()),
        (VALUE_KIND_I64, std::ptr::from_ref(&i64_value).cast()),
        (VALUE_KIND_U8, std::ptr::from_ref(&u8_value).cast()),
        (VALUE_KIND_U16, std::ptr::from_ref(&u16_value).cast()),
        (VALUE_KIND_U32, std::ptr::from_ref(&u32_value).cast()),
        (VALUE_KIND_U64, std::ptr::from_ref(&u64_value).cast()),
        (VALUE_KIND_F16, std::ptr::from_ref(&f16_value).cast()),
        (VALUE_KIND_F32, std::ptr::from_ref(&f32_value).cast()),
        (VALUE_KIND_F64, std::ptr::from_ref(&f64_value).cast()),
        (VALUE_KIND_PTR, std::ptr::from_ref(&pointer_value).cast()),
    ];

    for (kind, input) in cases {
        let array = unsafe { __faber_rt_v1_array_new(context, kind) };
        assert_eq!(array.status, STATUS_OK);
        assert_eq!(
            unsafe { __faber_rt_v1_array_push(context, array.value, kind, input) },
            STATUS_OK
        );

        let mut length = -1_i64;
        assert_eq!(
            unsafe { __faber_rt_v1_array_length(context, array.value, &raw mut length) },
            STATUS_OK
        );
        assert_eq!(length, 1);

        let mut output = 0_u64;
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_get(
                    context,
                    array.value,
                    0,
                    kind,
                    std::ptr::from_mut(&mut output).cast(),
                )
            },
            STATUS_OK
        );
        let width = match kind {
            VALUE_KIND_I1 | VALUE_KIND_I8 | VALUE_KIND_U8 => 1,
            VALUE_KIND_I16 | VALUE_KIND_U16 | VALUE_KIND_F16 => 2,
            VALUE_KIND_I32 | VALUE_KIND_U32 | VALUE_KIND_F32 => 4,
            VALUE_KIND_I64 | VALUE_KIND_U64 | VALUE_KIND_F64 | VALUE_KIND_PTR => 8,
            _ => unreachable!(),
        };
        assert_eq!(&output.to_ne_bytes()[..width], unsafe {
            std::slice::from_raw_parts(input.cast::<u8>(), width)
        });
    }

    let source = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    let target = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    let first = 1_i64;
    let second = 2_i64;
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_push(
                context,
                source.value,
                VALUE_KIND_I64,
                std::ptr::from_ref(&first).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_rt_v1_array_extend(context, target.value, source.value) },
        STATUS_OK
    );
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_set(
                context,
                target.value,
                0,
                VALUE_KIND_I64,
                std::ptr::from_ref(&second).cast(),
            )
        },
        STATUS_OK
    );
    let mut output = 0_i64;
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_get(
                context,
                target.value,
                0,
                VALUE_KIND_I64,
                std::ptr::from_mut(&mut output).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(output, second);

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn array_family_rejects_foreign_handles_kinds_and_bounds() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let array = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    let value = 1_i64;
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_push(
                context,
                array.value,
                VALUE_KIND_I64,
                std::ptr::from_ref(&value).cast(),
            )
        },
        STATUS_OK
    );

    let mut output = 0_i64;
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_get(
                context,
                context.cast(),
                0,
                VALUE_KIND_I64,
                std::ptr::from_mut(&mut output).cast(),
            )
        },
        STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_get(
                context,
                array.value,
                -1,
                VALUE_KIND_I64,
                std::ptr::from_mut(&mut output).cast(),
            )
        },
        STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_get(
                context,
                array.value,
                1,
                VALUE_KIND_I64,
                std::ptr::from_mut(&mut output).cast(),
            )
        },
        STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_get(
                context,
                array.value,
                0,
                VALUE_KIND_F64,
                std::ptr::from_mut(&mut output).cast(),
            )
        },
        STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe { __faber_rt_v1_array_length(context, array.value, ptr::null_mut()) },
        STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe { __faber_rt_v1_array_push(context, array.value, VALUE_KIND_I64, ptr::null()) },
        STATUS_INVALID_ARGUMENT
    );
    let mut aligned = [0_u64; 2];
    let misaligned = unsafe { aligned.as_mut_ptr().cast::<u8>().add(1).cast() };
    assert_eq!(
        unsafe { __faber_rt_v1_array_push(context, array.value, VALUE_KIND_I64, misaligned) },
        STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe { __faber_rt_v1_array_get(context, array.value, 0, VALUE_KIND_I64, misaligned) },
        STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe { __faber_rt_v1_array_length(context, array.value, misaligned.cast()) },
        STATUS_INVALID_ARGUMENT
    );

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn array_value_preserving_methods_clone_query_reverse_and_range() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let array = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    for value in [1_i64, 2, 3, 4, 5] {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    array.value,
                    VALUE_KIND_I64,
                    std::ptr::from_ref(&value).cast(),
                )
            },
            STATUS_OK
        );
    }

    let mut output = 0_u8;
    let three = 3_i64;
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_contains(
                context,
                array.value,
                VALUE_KIND_I64,
                std::ptr::from_ref(&three).cast(),
                &raw mut output,
            )
        },
        STATUS_OK
    );
    assert_eq!(output, 1);
    assert_eq!(
        unsafe { __faber_rt_v1_array_is_empty(context, array.value, &raw mut output) },
        STATUS_OK
    );
    assert_eq!(output, 0);

    let clone = unsafe { __faber_rt_v1_array_clone(context, array.value) };
    assert_eq!(clone.status, STATUS_OK);
    assert_eq!(
        unsafe { __faber_rt_v1_array_reverse(context, clone.value) },
        STATUS_OK
    );
    assert_array_i64(context, clone.value, &[5, 4, 3, 2, 1]);
    assert_array_i64(context, array.value, &[1, 2, 3, 4, 5]);

    for (mode, first, second, expected) in [
        (ARRAY_RANGE_SLICE, 1, 4, &[2_i64, 3, 4][..]),
        (ARRAY_RANGE_TAKE, 2, 0, &[1_i64, 2][..]),
        (ARRAY_RANGE_TAKE_LAST, 2, 0, &[4_i64, 5][..]),
        (ARRAY_RANGE_DROP_FIRST, 2, 0, &[3_i64, 4, 5][..]),
    ] {
        let result =
            unsafe { __faber_rt_v1_array_range(context, array.value, mode, first, second) };
        assert_eq!(result.status, STATUS_OK);
        assert_array_i64(context, result.value, expected);
    }
    for (mode, first, second) in [
        (ARRAY_RANGE_TAKE, -1, 0),
        (ARRAY_RANGE_SLICE, 0, -1),
        (99, 0, 0),
    ] {
        let result =
            unsafe { __faber_rt_v1_array_range(context, array.value, mode, first, second) };
        assert_eq!(result.status, STATUS_INVALID_ARGUMENT);
        assert!(result.value.is_null());
    }

    unsafe { __faber_rt_v1_shutdown(context) };
}

fn assert_array_i64(
    context: *mut FaberRtContextV1,
    array: *mut std::ffi::c_void,
    expected: &[i64],
) {
    let mut length = -1_i64;
    assert_eq!(
        unsafe { __faber_rt_v1_array_length(context, array, &raw mut length) },
        STATUS_OK
    );
    assert_eq!(usize::try_from(length), Ok(expected.len()));
    for (index, expected) in expected.iter().enumerate() {
        let mut actual = 0_i64;
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_get(
                    context,
                    array,
                    index as i64,
                    VALUE_KIND_I64,
                    std::ptr::from_mut(&mut actual).cast(),
                )
            },
            STATUS_OK
        );
        assert_eq!(&actual, expected);
    }
}

#[test]
fn array_option_methods_cover_access_empty_and_removal() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let array = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    for value in [10_i64, 20, 30] {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    array.value,
                    VALUE_KIND_I64,
                    std::ptr::from_ref(&value).cast(),
                )
            },
            STATUS_OK
        );
    }

    for (mode, index, expected) in [
        (ARRAY_OPTION_INDEX, 1, Some(20_i64)),
        (ARRAY_OPTION_FIRST, 0, Some(10)),
        (ARRAY_OPTION_LAST, 0, Some(30)),
        (ARRAY_OPTION_INDEX, -1, None),
        (ARRAY_OPTION_INDEX, 99, None),
        (ARRAY_OPTION_REMOVE_FIRST, 0, Some(10)),
        (ARRAY_OPTION_REMOVE_LAST, 0, Some(30)),
    ] {
        let result = unsafe { __faber_rt_v1_array_option(context, array.value, mode, index) };
        assert_eq!(result.status, STATUS_OK);
        let option = unsafe { &*result.value.cast::<RuntimeOption>() };
        assert_eq!(option.kind, VALUE_KIND_I64);
        assert_eq!(option_i64(option), expected);
    }
    assert_array_i64(context, array.value, &[20]);

    let invalid = unsafe { __faber_rt_v1_array_option(context, array.value, 99, 0) };
    assert_eq!(invalid.status, STATUS_INVALID_ARGUMENT);
    assert!(invalid.value.is_null());
    unsafe { __faber_rt_v1_shutdown(context) };
}

fn option_i64(option: &RuntimeOption) -> Option<i64> {
    match option.value {
        Some(array::RuntimeValue::I64(value)) => Some(value),
        None => None,
        _ => panic!("unexpected runtime option kind"),
    }
}

#[test]
fn option_family_produces_queries_unwraps_and_coalesces_shared_handles() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let value = 42_i64;
    let fallback = 9_i64;
    let none = unsafe { __faber_rt_v1_option_none(context, VALUE_KIND_I64) };
    let some = unsafe {
        __faber_rt_v1_option_some(context, VALUE_KIND_I64, std::ptr::from_ref(&value).cast())
    };
    assert_eq!(none.status, STATUS_OK);
    assert_eq!(some.status, STATUS_OK);

    for (option, expected) in [(none.value, 0_u8), (some.value, 1_u8)] {
        let mut present = 99_u8;
        assert_eq!(
            unsafe {
                __faber_rt_v1_option_is_present(
                    context,
                    option,
                    VALUE_KIND_I64,
                    std::ptr::from_mut(&mut present).cast(),
                )
            },
            STATUS_OK
        );
        assert_eq!(present, expected);
    }

    let mut output = 0_i64;
    assert_eq!(
        unsafe {
            __faber_rt_v1_option_get(
                context,
                some.value,
                VALUE_KIND_I64,
                std::ptr::from_mut(&mut output).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(output, value);
    assert_eq!(
        unsafe {
            __faber_rt_v1_option_get_or(
                context,
                none.value,
                VALUE_KIND_I64,
                std::ptr::from_ref(&fallback).cast(),
                std::ptr::from_mut(&mut output).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(output, fallback);
    assert_eq!(
        unsafe {
            __faber_rt_v1_option_get(
                context,
                none.value,
                VALUE_KIND_I64,
                std::ptr::from_mut(&mut output).cast(),
            )
        },
        STATUS_INVALID_ARGUMENT
    );

    let array = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_push(
                context,
                array.value,
                VALUE_KIND_I64,
                std::ptr::from_ref(&value).cast(),
            )
        },
        STATUS_OK
    );
    let endpoint =
        unsafe { __faber_rt_v1_array_option(context, array.value, ARRAY_OPTION_FIRST, 0) };
    assert_eq!(
        unsafe {
            __faber_rt_v1_option_get(
                context,
                endpoint.value,
                VALUE_KIND_I64,
                std::ptr::from_mut(&mut output).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(output, value);
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn array_numeric_sort_and_sum_works_for_unsigned_float_and_empty() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );

    let unsigned = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_U32) };
    for value in [u32::MAX, 1] {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    unsigned.value,
                    VALUE_KIND_U32,
                    std::ptr::from_ref(&value).cast(),
                )
            },
            STATUS_OK
        );
    }
    assert_eq!(
        unsafe { __faber_rt_v1_array_sort(context, unsigned.value) },
        STATUS_OK
    );
    let mut first = 0_u32;
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_get(
                context,
                unsigned.value,
                0,
                VALUE_KIND_U32,
                std::ptr::from_mut(&mut first).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(first, 1);
    let mut unsigned_sum = 0_u32;
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_sum(
                context,
                unsigned.value,
                VALUE_KIND_U32,
                std::ptr::from_mut(&mut unsigned_sum).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(unsigned_sum, 0);

    let floats = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_F64) };
    for value in [3.5_f64, -1.0, 2.0] {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    floats.value,
                    VALUE_KIND_F64,
                    std::ptr::from_ref(&value).cast(),
                )
            },
            STATUS_OK
        );
    }
    assert_eq!(
        unsafe { __faber_rt_v1_array_sort(context, floats.value) },
        STATUS_OK
    );
    let mut float_sum = 0.0_f64;
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_sum(
                context,
                floats.value,
                VALUE_KIND_F64,
                std::ptr::from_mut(&mut float_sum).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(float_sum, 4.5);

    let empty = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    let mut empty_sum = -1_i64;
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_sum(
                context,
                empty.value,
                VALUE_KIND_I64,
                std::ptr::from_mut(&mut empty_sum).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(empty_sum, 0);

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn array_sort_rejects_unsupported_value_kind() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );

    let unsupported = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_PTR) };
    assert_eq!(
        unsafe { __faber_rt_v1_array_sort(context, unsupported.value) },
        STATUS_INVALID_ARGUMENT
    );

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn tensor_core_shape_rank_set_get_and_empty_tensor() {
    let mut context = std::ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, std::ptr::null(), &raw mut context) },
        STATUS_OK
    );

    let shape = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    for dim in [2_i64, 3_i64] {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    shape.value,
                    VALUE_KIND_I64,
                    std::ptr::from_ref(&dim).cast(),
                )
            },
            STATUS_OK
        );
    }

    let flat = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_F32) };
    for value in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    flat.value,
                    VALUE_KIND_F32,
                    std::ptr::from_ref(&value).cast(),
                )
            },
            STATUS_OK
        );
    }

    let tensor =
        unsafe { __faber_rt_v1_tensor_from_flat(context, VALUE_KIND_F32, flat.value, shape.value) };
    assert_eq!(tensor.status, STATUS_OK);

    let mut rank = -1_i64;
    assert_eq!(
        unsafe { __faber_rt_v1_tensor_rank(context, tensor.value, &raw mut rank) },
        STATUS_OK
    );
    assert_eq!(rank, 2);

    let dims = unsafe { __faber_rt_v1_tensor_shape(context, tensor.value) };
    assert_eq!(dims.status, STATUS_OK);
    let mut length = 0_i64;
    assert_eq!(
        unsafe { __faber_rt_v1_array_length(context, dims.value, &raw mut length) },
        STATUS_OK
    );
    assert_eq!(length, 2);

    let origin = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    for dim in [0_i64, 0_i64] {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    origin.value,
                    VALUE_KIND_I64,
                    std::ptr::from_ref(&dim).cast(),
                )
            },
            STATUS_OK
        );
    }
    let nine = 9.0_f32;
    assert_eq!(
        unsafe {
            __faber_rt_v1_tensor_set(
                context,
                tensor.value,
                origin.value,
                VALUE_KIND_F32,
                std::ptr::from_ref(&nine).cast(),
            )
        },
        STATUS_OK
    );
    let got = unsafe { __faber_rt_v1_tensor_get(context, tensor.value, origin.value) };
    assert_eq!(got.status, STATUS_OK);
    let mut present = 0_u8;
    assert_eq!(
        unsafe {
            __faber_rt_v1_option_is_present(
                context,
                got.value,
                VALUE_KIND_F32,
                std::ptr::from_mut(&mut present).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(present, 1);

    let empty = unsafe { __faber_rt_v1_tensor_new(context, VALUE_KIND_F32) };
    assert_eq!(empty.status, STATUS_OK);
    let mut empty_rank = -1_i64;
    assert_eq!(
        unsafe { __faber_rt_v1_tensor_rank(context, empty.value, &raw mut empty_rank) },
        STATUS_OK
    );
    assert_eq!(empty_rank, 0);

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn tensor_slice_materialize_new_fill_reshape() {
    let mut context = std::ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, std::ptr::null(), &raw mut context) },
        STATUS_OK
    );

    let shape = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    for dim in [2_i64, 3_i64] {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    shape.value,
                    VALUE_KIND_I64,
                    std::ptr::from_ref(&dim).cast(),
                )
            },
            STATUS_OK
        );
    }

    let flat = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_F32) };
    for value in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    flat.value,
                    VALUE_KIND_F32,
                    std::ptr::from_ref(&value).cast(),
                )
            },
            STATUS_OK
        );
    }

    let tensor =
        unsafe { __faber_rt_v1_tensor_from_flat(context, VALUE_KIND_F32, flat.value, shape.value) };
    assert_eq!(tensor.status, STATUS_OK);

    let slice = unsafe { __faber_rt_v1_tensor_slice(context, tensor.value, 0, 1) };
    assert_eq!(slice.status, STATUS_OK);
    let owned = unsafe { __faber_rt_v1_tensor_materialize(context, slice.value) };
    assert_eq!(owned.status, STATUS_OK);
    let flat2 = unsafe { __faber_rt_v1_tensor_flatten(context, owned.value) };
    assert_eq!(flat2.status, STATUS_OK);
    let mut flat_len = 0_i64;
    assert_eq!(
        unsafe { __faber_rt_v1_array_length(context, flat2.value, &raw mut flat_len) },
        STATUS_OK
    );
    assert_eq!(flat_len, 3);

    let zero = 0.0_f32;
    let filled = unsafe {
        __faber_rt_v1_tensor_create(
            context,
            VALUE_KIND_F32,
            std::ptr::from_ref(&zero).cast(),
            shape.value,
        )
    };
    assert_eq!(filled.status, STATUS_OK);
    let four = 4.0_f32;
    let refilled = unsafe {
        __faber_rt_v1_tensor_fill(
            context,
            filled.value,
            VALUE_KIND_F32,
            std::ptr::from_ref(&four).cast(),
        )
    };
    assert_eq!(refilled.status, STATUS_OK);

    let newshape = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    for dim in [3_i64, 2_i64] {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    newshape.value,
                    VALUE_KIND_I64,
                    std::ptr::from_ref(&dim).cast(),
                )
            },
            STATUS_OK
        );
    }
    let reshaped = unsafe { __faber_rt_v1_tensor_reshape(context, refilled.value, newshape.value) };
    assert_eq!(reshaped.status, STATUS_OK);
    let mut reshaped_rank = 0_i64;
    assert_eq!(
        unsafe { __faber_rt_v1_tensor_rank(context, reshaped.value, &raw mut reshaped_rank) },
        STATUS_OK
    );
    assert_eq!(reshaped_rank, 2);

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn tensor_arithmetic_add_sub_mul_and_reduce_round_trip() {
    let mut context = std::ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, std::ptr::null(), &raw mut context) },
        STATUS_OK
    );

    let shape = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    for dim in [2_i64, 2_i64] {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    shape.value,
                    VALUE_KIND_I64,
                    std::ptr::from_ref(&dim).cast(),
                )
            },
            STATUS_OK
        );
    }
    let flat_a = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_F32) };
    for value in [1.0_f32, 2.0, 3.0, 4.0] {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    flat_a.value,
                    VALUE_KIND_F32,
                    std::ptr::from_ref(&value).cast(),
                )
            },
            STATUS_OK
        );
    }
    let flat_b = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_F32) };
    for value in [10.0_f32, 20.0, 30.0, 40.0] {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    flat_b.value,
                    VALUE_KIND_F32,
                    std::ptr::from_ref(&value).cast(),
                )
            },
            STATUS_OK
        );
    }
    let a = unsafe {
        __faber_rt_v1_tensor_from_flat(context, VALUE_KIND_F32, flat_a.value, shape.value)
    };
    let b = unsafe {
        __faber_rt_v1_tensor_from_flat(context, VALUE_KIND_F32, flat_b.value, shape.value)
    };
    assert_eq!(a.status, STATUS_OK);
    assert_eq!(b.status, STATUS_OK);

    let sum = unsafe { __faber_rt_v1_tensor_add(context, a.value, b.value) };
    assert_eq!(sum.status, STATUS_OK);
    let diff = unsafe { __faber_rt_v1_tensor_sub(context, sum.value, b.value) };
    assert_eq!(diff.status, STATUS_OK);
    let prod = unsafe { __faber_rt_v1_tensor_mul(context, diff.value, a.value) };
    assert_eq!(prod.status, STATUS_OK);
    let flat = unsafe { __faber_rt_v1_tensor_flatten(context, prod.value) };
    assert_eq!(flat.status, STATUS_OK);

    let mut total = 0.0_f32;
    assert_eq!(
        unsafe {
            __faber_rt_v1_tensor_sum(
                context,
                a.value,
                VALUE_KIND_F32,
                std::ptr::from_mut(&mut total).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(total, 10.0);
    let mut mean = 0.0_f32;
    assert_eq!(
        unsafe {
            __faber_rt_v1_tensor_mean(
                context,
                a.value,
                VALUE_KIND_F32,
                std::ptr::from_mut(&mut mean).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(mean, 2.5);

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn tensor_matmul_rank2_produces_rank2_result() {
    let mut context = std::ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, std::ptr::null(), &raw mut context) },
        STATUS_OK
    );

    let shape_a = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    for dim in [2_i64, 3_i64] {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    shape_a.value,
                    VALUE_KIND_I64,
                    std::ptr::from_ref(&dim).cast(),
                )
            },
            STATUS_OK
        );
    }
    let shape_b = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    for dim in [3_i64, 2_i64] {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    shape_b.value,
                    VALUE_KIND_I64,
                    std::ptr::from_ref(&dim).cast(),
                )
            },
            STATUS_OK
        );
    }
    let flat_m = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_F32) };
    for value in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    flat_m.value,
                    VALUE_KIND_F32,
                    std::ptr::from_ref(&value).cast(),
                )
            },
            STATUS_OK
        );
    }
    let left = unsafe {
        __faber_rt_v1_tensor_from_flat(context, VALUE_KIND_F32, flat_m.value, shape_a.value)
    };
    let right = unsafe {
        __faber_rt_v1_tensor_from_flat(context, VALUE_KIND_F32, flat_m.value, shape_b.value)
    };
    let product = unsafe { __faber_rt_v1_tensor_matmul(context, left.value, right.value) };
    assert_eq!(product.status, STATUS_OK);
    let mut rank = 0_i64;
    assert_eq!(
        unsafe { __faber_rt_v1_tensor_rank(context, product.value, &raw mut rank) },
        STATUS_OK
    );
    assert_eq!(rank, 2);

    unsafe { __faber_rt_v1_shutdown(context) };
}

fn push_host_tensor_value(
    context: *mut FaberRtContextV1,
    array: *mut c_void,
    kind: FaberRtValueKindV1,
    value: i64,
) {
    match kind {
        VALUE_KIND_F32 => {
            let value = value as f32;
            assert_eq!(
                unsafe {
                    __faber_rt_v1_array_push(
                        context,
                        array,
                        kind,
                        std::ptr::from_ref(&value).cast(),
                    )
                },
                STATUS_OK
            );
        }
        VALUE_KIND_F64 => {
            let value = value as f64;
            assert_eq!(
                unsafe {
                    __faber_rt_v1_array_push(
                        context,
                        array,
                        kind,
                        std::ptr::from_ref(&value).cast(),
                    )
                },
                STATUS_OK
            );
        }
        VALUE_KIND_I32 => {
            let value = value as i32;
            assert_eq!(
                unsafe {
                    __faber_rt_v1_array_push(
                        context,
                        array,
                        kind,
                        std::ptr::from_ref(&value).cast(),
                    )
                },
                STATUS_OK
            );
        }
        VALUE_KIND_I64 => {
            assert_eq!(
                unsafe {
                    __faber_rt_v1_array_push(
                        context,
                        array,
                        kind,
                        std::ptr::from_ref(&value).cast(),
                    )
                },
                STATUS_OK
            );
        }
        _ => panic!("unsupported test tensor kind {kind}"),
    }
}

fn read_host_tensor_sum(
    context: *mut FaberRtContextV1,
    tensor: *mut c_void,
    kind: FaberRtValueKindV1,
) -> i64 {
    match kind {
        VALUE_KIND_F32 => {
            let mut total = 0.0_f32;
            assert_eq!(
                unsafe {
                    __faber_rt_v1_tensor_sum(
                        context,
                        tensor,
                        kind,
                        std::ptr::from_mut(&mut total).cast(),
                    )
                },
                STATUS_OK
            );
            total as i64
        }
        VALUE_KIND_F64 => {
            let mut total = 0.0_f64;
            assert_eq!(
                unsafe {
                    __faber_rt_v1_tensor_sum(
                        context,
                        tensor,
                        kind,
                        std::ptr::from_mut(&mut total).cast(),
                    )
                },
                STATUS_OK
            );
            total as i64
        }
        VALUE_KIND_I32 => {
            let mut total = 0_i32;
            assert_eq!(
                unsafe {
                    __faber_rt_v1_tensor_sum(
                        context,
                        tensor,
                        kind,
                        std::ptr::from_mut(&mut total).cast(),
                    )
                },
                STATUS_OK
            );
            total.into()
        }
        VALUE_KIND_I64 => {
            let mut total = 0_i64;
            assert_eq!(
                unsafe {
                    __faber_rt_v1_tensor_sum(
                        context,
                        tensor,
                        kind,
                        std::ptr::from_mut(&mut total).cast(),
                    )
                },
                STATUS_OK
            );
            total
        }
        _ => panic!("unsupported test tensor kind {kind}"),
    }
}

fn host_tensor_shape(context: *mut FaberRtContextV1, dims: &[i64]) -> FaberRtPtrResultV1 {
    let shape = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    assert_eq!(shape.status, STATUS_OK);
    for dim in dims {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    shape.value,
                    VALUE_KIND_I64,
                    std::ptr::from_ref(dim).cast(),
                )
            },
            STATUS_OK
        );
    }
    shape
}

fn host_tensor_from_i64_values(
    context: *mut FaberRtContextV1,
    kind: FaberRtValueKindV1,
    values: &[i64],
    dims: &[i64],
) -> FaberRtPtrResultV1 {
    let flat = unsafe { __faber_rt_v1_array_new(context, kind) };
    assert_eq!(flat.status, STATUS_OK);
    for value in values {
        push_host_tensor_value(context, flat.value, kind, *value);
    }
    let shape = host_tensor_shape(context, dims);
    unsafe { __faber_rt_v1_tensor_from_flat(context, kind, flat.value, shape.value) }
}

#[test]
fn tensor_host_arithmetic_boundary_supports_f32_f64_i32_i64() {
    let mut context = std::ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, std::ptr::null(), &raw mut context) },
        STATUS_OK
    );

    for kind in [
        VALUE_KIND_F32,
        VALUE_KIND_F64,
        VALUE_KIND_I32,
        VALUE_KIND_I64,
    ] {
        let left = host_tensor_from_i64_values(context, kind, &[1, 2, 3], &[3]);
        let right = host_tensor_from_i64_values(context, kind, &[10, 20, 30], &[3]);
        assert_eq!(left.status, STATUS_OK);
        assert_eq!(right.status, STATUS_OK);

        let sum = unsafe { __faber_rt_v1_tensor_add(context, left.value, right.value) };
        assert_eq!(sum.status, STATUS_OK);
        assert_eq!(read_host_tensor_sum(context, sum.value, kind), 66);

        let diff = unsafe { __faber_rt_v1_tensor_sub(context, sum.value, right.value) };
        assert_eq!(diff.status, STATUS_OK);
        assert_eq!(read_host_tensor_sum(context, diff.value, kind), 6);

        let product = unsafe { __faber_rt_v1_tensor_mul(context, diff.value, left.value) };
        assert_eq!(product.status, STATUS_OK);
        assert_eq!(read_host_tensor_sum(context, product.value, kind), 14);
    }

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn tensor_host_rejects_kinds_without_arithmetic_dispatch_at_admission() {
    let mut context = std::ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, std::ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let shape = host_tensor_shape(context, &[1]);
    let one_i16 = 1_i16;
    #[allow(clippy::similar_names)]
    let one_u32 = 1_u32;
    let one_f16_bits = 0x3c00_u16;
    let one_i1 = 1_u8;

    for (kind, value) in [
        (VALUE_KIND_I1, std::ptr::from_ref(&one_i1).cast()),
        (VALUE_KIND_I16, std::ptr::from_ref(&one_i16).cast()),
        (VALUE_KIND_U32, std::ptr::from_ref(&one_u32).cast()),
        (VALUE_KIND_F16, std::ptr::from_ref(&one_f16_bits).cast()),
    ] {
        let flat = unsafe { __faber_rt_v1_array_new(context, kind) };
        assert_eq!(flat.status, STATUS_OK);
        assert_eq!(
            unsafe { __faber_rt_v1_array_push(context, flat.value, kind, value) },
            STATUS_OK
        );

        assert_eq!(
            unsafe { __faber_rt_v1_tensor_new(context, kind) }.status,
            STATUS_INVALID_ARGUMENT
        );
        assert_eq!(
            unsafe { __faber_rt_v1_tensor_from_flat(context, kind, flat.value, shape.value) }
                .status,
            STATUS_INVALID_ARGUMENT
        );
        assert_eq!(
            unsafe { __faber_rt_v1_tensor_create(context, kind, value, shape.value) }.status,
            STATUS_INVALID_ARGUMENT
        );
    }

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn tensor_host_add_broadcasts_zero_extent_without_panic() {
    let mut context = std::ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, std::ptr::null(), &raw mut context) },
        STATUS_OK
    );

    let empty_shape = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    for dim in [0_i64, 3_i64] {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    empty_shape.value,
                    VALUE_KIND_I64,
                    std::ptr::from_ref(&dim).cast(),
                )
            },
            STATUS_OK
        );
    }
    let empty_flat = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_F32) };
    let empty = unsafe {
        __faber_rt_v1_tensor_from_flat(context, VALUE_KIND_F32, empty_flat.value, empty_shape.value)
    };
    assert_eq!(empty.status, STATUS_OK);

    let row_shape = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    for dim in [1_i64, 3_i64] {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    row_shape.value,
                    VALUE_KIND_I64,
                    std::ptr::from_ref(&dim).cast(),
                )
            },
            STATUS_OK
        );
    }
    let row_flat = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_F32) };
    for value in [1.0_f32, 2.0, 3.0] {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    row_flat.value,
                    VALUE_KIND_F32,
                    std::ptr::from_ref(&value).cast(),
                )
            },
            STATUS_OK
        );
    }
    let row = unsafe {
        __faber_rt_v1_tensor_from_flat(context, VALUE_KIND_F32, row_flat.value, row_shape.value)
    };
    assert_eq!(row.status, STATUS_OK);

    let sum = unsafe { __faber_rt_v1_tensor_add(context, empty.value, row.value) };
    assert_eq!(sum.status, STATUS_OK);
    let flat = unsafe { __faber_rt_v1_tensor_flatten(context, sum.value) };
    assert_eq!(flat.status, STATUS_OK);
    let mut flat_len = -1_i64;
    assert_eq!(
        unsafe { __faber_rt_v1_array_length(context, flat.value, &raw mut flat_len) },
        STATUS_OK
    );
    assert_eq!(flat_len, 0);

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn tensor_convert_widens_and_narrows_element_kinds() {
    let mut context = std::ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, std::ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let shape = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    let dim = 2_i64;
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_push(
                context,
                shape.value,
                VALUE_KIND_I64,
                std::ptr::from_ref(&dim).cast(),
            )
        },
        STATUS_OK
    );
    let flat = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    for value in [1_i64, 2_i64] {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    flat.value,
                    VALUE_KIND_I64,
                    std::ptr::from_ref(&value).cast(),
                )
            },
            STATUS_OK
        );
    }
    let ints =
        unsafe { __faber_rt_v1_tensor_from_flat(context, VALUE_KIND_I64, flat.value, shape.value) };
    assert_eq!(ints.status, STATUS_OK);
    let floats = unsafe {
        __faber_rt_v1_tensor_convert(context, ints.value, VALUE_KIND_I64, VALUE_KIND_F64)
    };
    assert_eq!(floats.status, STATUS_OK);
    let back = unsafe {
        __faber_rt_v1_tensor_convert(context, floats.value, VALUE_KIND_F64, VALUE_KIND_I64)
    };
    assert_eq!(back.status, STATUS_OK);
    let flat2 = unsafe { __faber_rt_v1_tensor_flatten(context, back.value) };
    assert_eq!(flat2.status, STATUS_OK);
    let mut length = 0_i64;
    assert_eq!(
        unsafe { __faber_rt_v1_array_length(context, flat2.value, &raw mut length) },
        STATUS_OK
    );
    assert_eq!(length, 2);
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn sparse_core_carrier_sets_gets_and_densifies() {
    let mut context = std::ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, std::ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let shape = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    for dim in [2_i64, 3_i64] {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    shape.value,
                    VALUE_KIND_I64,
                    std::ptr::from_ref(&dim).cast(),
                )
            },
            STATUS_OK
        );
    }
    let sparse = unsafe { __faber_rt_v1_sparse_new(context, VALUE_KIND_F32, shape.value) };
    assert_eq!(sparse.status, STATUS_OK);
    let idx = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    for dim in [0_i64, 1_i64] {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    idx.value,
                    VALUE_KIND_I64,
                    std::ptr::from_ref(&dim).cast(),
                )
            },
            STATUS_OK
        );
    }
    let four = 4.0_f32;
    assert_eq!(
        unsafe {
            __faber_rt_v1_sparse_set(
                context,
                sparse.value,
                idx.value,
                VALUE_KIND_F32,
                std::ptr::from_ref(&four).cast(),
            )
        },
        STATUS_OK
    );
    let mut got = 0.0_f32;
    assert_eq!(
        unsafe {
            __faber_rt_v1_sparse_get(
                context,
                sparse.value,
                idx.value,
                VALUE_KIND_F32,
                std::ptr::from_mut(&mut got).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(got, 4.0);
    let mut count = 0_i64;
    assert_eq!(
        unsafe { __faber_rt_v1_sparse_nonzero(context, sparse.value, &raw mut count) },
        STATUS_OK
    );
    assert_eq!(count, 1);
    let dense = unsafe { __faber_rt_v1_sparse_densify(context, sparse.value) };
    assert_eq!(dense.status, STATUS_OK);
    let mut rank = 0_i64;
    assert_eq!(
        unsafe { __faber_rt_v1_tensor_rank(context, dense.value, &raw mut rank) },
        STATUS_OK
    );
    assert_eq!(rank, 2);
    let back = unsafe { __faber_rt_v1_sparse_from_tensor(context, dense.value, VALUE_KIND_F32) };
    assert_eq!(back.status, STATUS_OK);
    let mut count2 = 0_i64;
    assert_eq!(
        unsafe { __faber_rt_v1_sparse_nonzero(context, back.value, &raw mut count2) },
        STATUS_OK
    );
    assert_eq!(count2, 1);
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn regex_conversion_preserves_pattern_text() {
    let mut context = std::ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, std::ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let digits = b"\\d+\0";
    let slice = FaberRtSliceV1 {
        data: digits.as_ptr().cast_mut().cast(),
        len: 3,
    };
    let regex = unsafe { __faber_rt_v1_regex_from_text(context, &raw const slice) };
    assert_eq!(regex.status, STATUS_OK);
    let text = unsafe { __faber_rt_v1_regex_get_text(context, regex.value) };
    assert_eq!(text.status, STATUS_OK);
    let ascii = unsafe { __faber_rt_v1_regex_from_ascii(context, c"(?i)[a-z]+".as_ptr()) };
    assert_eq!(ascii.status, STATUS_OK);
    let rendered = unsafe { __faber_rt_v1_regex_get_text(context, ascii.value) };
    assert_eq!(rendered.status, STATUS_OK);
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn set_array_collection_conversion_dedupes() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let array = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    assert_eq!(array.status, STATUS_OK);
    for value in [1_i64, 2, 2, 3] {
        let mut slot = value;
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    array.value,
                    VALUE_KIND_I64,
                    &raw mut slot as *const _,
                )
            },
            STATUS_OK
        );
    }
    let set = unsafe { __faber_rt_v1_set_from_array(context, array.value) };
    assert_eq!(set.status, STATUS_OK);
    let mut length = 0_i64;
    assert_eq!(
        unsafe { __faber_rt_v1_set_length(context, set.value, &raw mut length) },
        STATUS_OK
    );
    assert_eq!(length, 3);
    let back = unsafe { __faber_rt_v1_array_from_set(context, set.value) };
    assert_eq!(back.status, STATUS_OK);
    length = 0;
    assert_eq!(
        unsafe { __faber_rt_v1_array_length(context, back.value, &raw mut length) },
        STATUS_OK
    );
    assert_eq!(length, 3);
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn interval_carrier_algebra_and_materialize() {
    let mut context = std::ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, std::ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let morning = unsafe { __faber_rt_v1_interval_new(context, 0, 6, 0) };
    let afternoon = unsafe { __faber_rt_v1_interval_new(context, 4, 10, 0) };
    assert_eq!(morning.status, STATUS_OK);
    assert_eq!(afternoon.status, STATUS_OK);
    let overlap =
        unsafe { __faber_rt_v1_interval_intersect(context, morning.value, afternoon.value) };
    assert_eq!(overlap.status, STATUS_OK);
    let mut present = 0_u8;
    assert_eq!(
        unsafe {
            __faber_rt_v1_option_is_present(
                context,
                overlap.value,
                VALUE_KIND_PTR,
                (&raw mut present).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(present, 1);
    let mut interval_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            __faber_rt_v1_option_get(
                context,
                overlap.value,
                VALUE_KIND_PTR,
                (&raw mut interval_ptr).cast(),
            )
        },
        STATUS_OK
    );
    let mut length = 0_i64;
    assert_eq!(
        unsafe { __faber_rt_v1_interval_length(context, interval_ptr, &raw mut length) },
        STATUS_OK
    );
    assert_eq!(length, 2);

    let gap_left = unsafe { __faber_rt_v1_interval_new(context, 0, 3, 0) };
    let gap_right = unsafe { __faber_rt_v1_interval_new(context, 6, 9, 0) };
    let no_union =
        unsafe { __faber_rt_v1_interval_union(context, gap_left.value, gap_right.value) };
    assert_eq!(no_union.status, STATUS_OK);
    present = 1;
    assert_eq!(
        unsafe {
            __faber_rt_v1_option_is_present(
                context,
                no_union.value,
                VALUE_KIND_PTR,
                (&raw mut present).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(present, 0);

    let half = unsafe { __faber_rt_v1_interval_new(context, 0, 10, 0) };
    let mut clamped = 0_i64;
    assert_eq!(
        unsafe { __faber_rt_v1_interval_clamp_i64(context, 15, half.value, &raw mut clamped) },
        STATUS_OK
    );
    assert_eq!(clamped, 9);
    let mut contains = 0_u8;
    assert_eq!(
        unsafe { __faber_rt_v1_interval_contains(context, half.value, 5, &raw mut contains) },
        STATUS_OK
    );
    assert_eq!(contains, 1);

    let list = unsafe { __faber_rt_v1_interval_materialize_array(context, half.value) };
    assert_eq!(list.status, STATUS_OK);
    let mut list_len = 0_i64;
    assert_eq!(
        unsafe { __faber_rt_v1_array_length(context, list.value, &raw mut list_len) },
        STATUS_OK
    );
    assert_eq!(list_len, 10);

    let closed = unsafe { __faber_rt_v1_interval_new(context, 0, 3, 1) };
    let tensor = unsafe { __faber_rt_v1_interval_materialize_tensor(context, closed.value) };
    assert_eq!(tensor.status, STATUS_OK);
    let wide = unsafe { __faber_rt_v1_interval_new(context, 0, 100, 0) };
    let narrow_target = unsafe { __faber_rt_v1_interval_new(context, 10, 50, 1) };
    let narrow = unsafe { __faber_rt_v1_interval_clamp(context, wide.value, narrow_target.value) };
    assert_eq!(narrow.status, STATUS_OK);
    length = 0;
    assert_eq!(
        unsafe { __faber_rt_v1_interval_length(context, narrow.value, &raw mut length) },
        STATUS_OK
    );
    assert_eq!(length, 41);
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn gradient_create_accumulate_read_zero_round_trip() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );

    // Create a gradient with shape [2, 3].
    let shape = [2_i64, 3];
    let gradient =
        unsafe { __faber_rt_v1_gradient_create(context, shape.as_ptr(), 2, VALUE_KIND_F32) };
    assert_eq!(gradient.status, STATUS_OK);
    assert!(!gradient.value.is_null());

    // Accumulate incoming data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].
    let incoming: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_eq!(
        unsafe {
            __faber_rt_v1_gradient_accumulate(
                context,
                gradient.value,
                incoming.as_ptr(),
                shape.as_ptr(),
                2,
            )
        },
        STATUS_OK
    );

    // Read the gradient view and verify data through the repr(C) carrier.
    let view = unsafe { __faber_rt_v1_gradient_read(context, gradient.value) };
    assert_eq!(view.status, STATUS_OK);
    let view_ptr = view.value.cast::<gradient::GradientViewV1>();
    let view_ref = unsafe { &*view_ptr };
    assert_eq!(
        unsafe { std::slice::from_raw_parts(view_ref.data, view_ref.len as usize) },
        incoming
    );
    assert_eq!(
        unsafe { std::slice::from_raw_parts(view_ref.shape, view_ref.rank as usize) },
        shape
    );

    // Accumulate again: same shape, different values.
    let more: [f32; 6] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    assert_eq!(
        unsafe {
            __faber_rt_v1_gradient_accumulate(
                context,
                gradient.value,
                more.as_ptr(),
                shape.as_ptr(),
                2,
            )
        },
        STATUS_OK
    );

    // Read again — view should reflect the accumulated values.
    let view2 = unsafe { __faber_rt_v1_gradient_read(context, gradient.value) };
    assert_eq!(view2.status, STATUS_OK);
    let view2_ptr = view2.value.cast::<gradient::GradientViewV1>();
    let view2_ref = unsafe { &*view2_ptr };
    let expected: [f32; 6] = [11.0, 22.0, 33.0, 44.0, 55.0, 66.0];
    assert_eq!(
        unsafe { std::slice::from_raw_parts(view2_ref.data, view2_ref.len as usize) },
        expected
    );

    // Zero the gradient.
    assert_eq!(
        unsafe { __faber_rt_v1_gradient_zero(context, gradient.value) },
        STATUS_OK
    );

    // Read after zero — all elements should be 0.0.
    let view3 = unsafe { __faber_rt_v1_gradient_read(context, gradient.value) };
    assert_eq!(view3.status, STATUS_OK);
    let view3_ptr = view3.value.cast::<gradient::GradientViewV1>();
    let view3_ref = unsafe { &*view3_ptr };
    let zeros: [f32; 6] = [0.0; 6];
    assert_eq!(
        unsafe { std::slice::from_raw_parts(view3_ref.data, view3_ref.len as usize) },
        zeros
    );

    // Verify shape is still preserved after zero.
    assert_eq!(
        unsafe { std::slice::from_raw_parts(view3_ref.shape, view3_ref.rank as usize) },
        shape
    );

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn gpu_placement_copy_in_readback_round_trip() {
    let input: [u8; 16] =
        unsafe { std::mem::transmute::<[f32; 4], [u8; 16]>([1.0_f32, 2.0, 3.0, 4.0]) };

    let status = unsafe { __faber_gpu_v1_copy_in(42, input.as_ptr(), input.len() as u64, 0) };
    assert_eq!(status, STATUS_OK);

    let mut dest = [0_u8; 16];
    let mut actual_len = 0_u64;
    let status = unsafe {
        __faber_gpu_v1_readback(
            42,
            dest.as_mut_ptr(),
            dest.len() as u64,
            &raw mut actual_len,
        )
    };
    assert_eq!(status, STATUS_OK);
    assert_eq!(actual_len, 16);
    assert_eq!(dest, input);

    let result: [f32; 4] = unsafe { std::mem::transmute::<[u8; 16], [f32; 4]>(dest) };
    assert_eq!(result, [1.0_f32, 2.0, 3.0, 4.0]);
}

#[test]
fn gpu_placement_copy_in_overwrites_existing_buffer() {
    let first: [u8; 4] = f32::to_ne_bytes(42.0_f32);
    let second: [u8; 4] = f32::to_ne_bytes(99.0_f32);

    unsafe { __faber_gpu_v1_copy_in(1, first.as_ptr(), 4, 0) };
    unsafe { __faber_gpu_v1_copy_in(1, second.as_ptr(), 4, 0) };

    let mut dest = [0_u8; 4];
    let mut actual_len = 0_u64;
    let status = unsafe { __faber_gpu_v1_readback(1, dest.as_mut_ptr(), 4, &raw mut actual_len) };
    assert_eq!(status, STATUS_OK);
    assert_eq!(actual_len, 4);

    let value: f32 = f32::from_ne_bytes(dest);
    assert_eq!(value, 99.0_f32);
}

#[test]
fn gpu_placement_readback_unknown_buffer_fails() {
    let mut dest = [0_u8; 4];
    let mut actual_len = 0_u64;
    let status = unsafe { __faber_gpu_v1_readback(999, dest.as_mut_ptr(), 4, &raw mut actual_len) };
    assert_eq!(status, STATUS_INVALID_ARGUMENT);
}

#[test]
fn gpu_placement_readback_capacity_too_small_fails() {
    let input: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    unsafe { __faber_gpu_v1_copy_in(77, input.as_ptr(), 8, 0) };

    let mut dest = [0_u8; 4];
    let mut actual_len = 0_u64;
    let status = unsafe { __faber_gpu_v1_readback(77, dest.as_mut_ptr(), 4, &raw mut actual_len) };
    assert_eq!(status, STATUS_IO_ERROR);
}

#[test]
fn gpu_placement_copy_in_null_ptr_with_positive_length_fails() {
    let status = unsafe { __faber_gpu_v1_copy_in(1, std::ptr::null(), 8, 0) };
    assert_eq!(status, STATUS_INVALID_ARGUMENT);
}

#[test]
fn gpu_placement_copy_in_zero_length_allocates_empty_buffer() {
    let status = unsafe { __faber_gpu_v1_copy_in(0, std::ptr::null(), 0, 0) };
    assert_eq!(status, STATUS_OK);

    let mut dest = [0_u8; 1];
    let mut actual_len = 0_u64;
    let status = unsafe { __faber_gpu_v1_readback(0, dest.as_mut_ptr(), 1, &raw mut actual_len) };
    assert_eq!(status, STATUS_OK);
    assert_eq!(actual_len, 0);
}

#[test]
fn gpu_placement_readback_null_dest_fails() {
    let mut actual_len = 0_u64;
    let status =
        unsafe { __faber_gpu_v1_readback(1, std::ptr::null_mut(), 8, &raw mut actual_len) };
    assert_eq!(status, STATUS_INVALID_ARGUMENT);
}

#[test]
fn gpu_placement_readback_null_actual_len_fails() {
    let mut dest = [0_u8; 4];
    let status = unsafe { __faber_gpu_v1_readback(1, dest.as_mut_ptr(), 4, std::ptr::null_mut()) };
    assert_eq!(status, STATUS_INVALID_ARGUMENT);
}

#[test]
fn gpu_placement_sync_is_noop_and_returns_ok() {
    let status = unsafe { __faber_gpu_v1_sync(0) };
    assert_eq!(status, STATUS_OK);

    let status = unsafe { __faber_gpu_v1_sync(u64::MAX) };
    assert_eq!(status, STATUS_OK);
}

#[test]
fn gpu_placement_multiple_buffers_independent() {
    let a: [u8; 4] = unsafe { std::mem::transmute::<[f32; 1], [u8; 4]>([10.0_f32]) };
    let b: [u8; 4] = unsafe { std::mem::transmute::<[f32; 1], [u8; 4]>([20.0_f32]) };

    unsafe { __faber_gpu_v1_copy_in(100, a.as_ptr(), 4, 0) };
    unsafe { __faber_gpu_v1_copy_in(200, b.as_ptr(), 4, 0) };

    let mut dest_a = [0_u8; 4];
    let mut len_a = 0_u64;
    let mut dest_b = [0_u8; 4];
    let mut len_b = 0_u64;

    assert_eq!(
        unsafe { __faber_gpu_v1_readback(100, dest_a.as_mut_ptr(), 4, &raw mut len_a) },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_gpu_v1_readback(200, dest_b.as_mut_ptr(), 4, &raw mut len_b) },
        STATUS_OK
    );

    assert_eq!(len_a, 4);
    assert_eq!(len_b, 4);
    assert_eq!(dest_a, a);
    assert_eq!(dest_b, b);
}

// ── G-SPINE-08 Stage 2: LLVM device execution exemplar ──────────────────

/// Exemplar integration test for the full placement lifecycle:
/// copy-in → kernel dispatch → readback → verify.
///
/// # Gating issue
///
/// This test is `#[ignore]` because the full compilation pipeline
/// (`.faber` → MIR → `radix-mir-llvm` → LLVM IR → native code) is not yet
/// available from within `cargo test`. The exemplar kernel source lives at
/// `faber-runtime/hosts/llvm/exempla/tensor/llvm-placement-v1.fab`.
///
/// **What is needed to unblock:**
/// 1. `faber build --target llvm-host` producing a native binary or
///    object file from `.fab` source, OR
/// 2. An `inkwell` / `llvm-sys` dev-dependency in `faber-host-llvm` for
///    in-process JIT compilation of the emitted LLVM IR, OR
/// 3. A build script that invokes the LLVM toolchain (`llc`, `clang`) on
///    PATH to compile the emitted IR to a `.o` and link it as a
///    `#[link]` dependency.
///
/// # Honest device execution
///
/// The LLVM emitter (`radix-mir-llvm`) lowers tensor elementwise
/// multiplication to `@__faber_rt_v1_tensor_mul` — a runtime FFI call.
/// This test calls the same runtime FFI, following the standard JIT ABI
/// pattern. The distinguishing factor (per the delivery spec) is the
/// compilation pipeline, not the absence of runtime support calls.
/// No Rust-side elementwise arithmetic is used — the multiply runs
/// through the same `tensor_mul` runtime helper that the emitted native
/// code would invoke.
///
/// # Compilation PATH requirement
///
/// When a LLVM toolchain is available, run with:
/// `cargo test -p faber-host-llvm -- --ignored --nocapture`
#[test]
#[ignore = "G-SPINE-08 S2: requires faber build --target llvm-host or LLVM toolchain on PATH. \
           Exemplar source at exempla/tensor/llvm-placement-v1.fab"]
fn llvm_device_execution_exemplar_multiply_by_two() {
    // ── Step 1: Copy-in input data ──────────────────────────────────
    // Stage f32 input [1.0, 2.0, 3.0, 4.0] as raw bytes.
    let input_f32: [f32; 4] = [1.0_f32, 2.0, 3.0, 4.0];
    let input_bytes: [u8; 16] = unsafe { std::mem::transmute::<[f32; 4], [u8; 16]>(input_f32) };

    let copy_in_status =
        unsafe { __faber_gpu_v1_copy_in(42, input_bytes.as_ptr(), input_bytes.len() as u64, 0) };
    assert_eq!(copy_in_status, STATUS_OK, "copy_in failed");

    // ── Step 2: Kernel dispatch (runtime FFI pattern) ──────────────
    // In a full build, the LLVM-emitted native kernel would:
    //   a. Load the runtime context
    //   b. Create tensors from device buffer data
    //   c. Call @__faber_rt_v1_tensor_mul(context, input_tensor, two_scalar)
    //   d. Flatten the result back to raw bytes
    //   e. Write result bytes to the device output buffer
    //
    // This test demonstrates steps (a)–(d) through the same runtime
    // FFI that the emitted native code calls.

    let mut context = ptr::null_mut();
    let init_status = unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) };
    assert_eq!(init_status, STATUS_OK);
    assert!(!context.is_null());

    // Build the shape [4] for a rank-1 tensor.
    let shape = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    assert_eq!(shape.status, STATUS_OK);
    {
        let dim = 4_i64;
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    shape.value,
                    VALUE_KIND_I64,
                    std::ptr::from_ref(&dim).cast(),
                )
            },
            STATUS_OK
        );
    }

    // Build the flat element list for the input tensor.
    let flat_input = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_F32) };
    assert_eq!(flat_input.status, STATUS_OK);
    for value in &input_f32 {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    flat_input.value,
                    VALUE_KIND_F32,
                    std::ptr::from_ref(value).cast(),
                )
            },
            STATUS_OK
        );
    }

    // Build the flat element list for the scalar multiplier [2.0].
    let flat_two = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_F32) };
    assert_eq!(flat_two.status, STATUS_OK);
    let two = 2.0_f32;
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_push(
                context,
                flat_two.value,
                VALUE_KIND_F32,
                std::ptr::from_ref(&two).cast(),
            )
        },
        STATUS_OK
    );

    // Shape for the scalar tensor [1].
    let scalar_shape = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    assert_eq!(scalar_shape.status, STATUS_OK);
    let one_dim = 1_i64;
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_push(
                context,
                scalar_shape.value,
                VALUE_KIND_I64,
                std::ptr::from_ref(&one_dim).cast(),
            )
        },
        STATUS_OK
    );

    // Create the input tensor [1.0, 2.0, 3.0, 4.0] shape [4].
    let input_tensor = unsafe {
        __faber_rt_v1_tensor_from_flat(context, VALUE_KIND_F32, flat_input.value, shape.value)
    };
    assert_eq!(input_tensor.status, STATUS_OK);

    // Create the scalar multiplier tensor [2.0] shape [1].
    let two_tensor = unsafe {
        __faber_rt_v1_tensor_from_flat(context, VALUE_KIND_F32, flat_two.value, scalar_shape.value)
    };
    assert_eq!(two_tensor.status, STATUS_OK);

    // Multiply: this is the @__faber_rt_v1_tensor_mul call that the
    // LLVM-emitted native kernel would make. Broadcast semantics in
    // the runtime handle the scalar broadcast automatically.
    let product =
        unsafe { __faber_rt_v1_tensor_mul(context, input_tensor.value, two_tensor.value) };
    assert_eq!(product.status, STATUS_OK);

    // Flatten the result tensor to a lista<f32> to extract raw bytes.
    let flat_result = unsafe { __faber_rt_v1_tensor_flatten(context, product.value) };
    assert_eq!(flat_result.status, STATUS_OK);

    // Read the flattened result array to extract f32 values as bytes.
    let runtime_array = unsafe { &*flat_result.value.cast::<RuntimeArray>() };
    assert_eq!(runtime_array.values.len(), 4);
    let mut output_bytes = [0_u8; 16];
    for (i, value) in runtime_array.values.iter().enumerate() {
        let array::RuntimeValue::F32(f) = value else {
            panic!("expected F32 at index {i}, got non-F32 RuntimeValue");
        };
        let f_bytes: [u8; 4] = f32::to_ne_bytes(f);
        output_bytes[i * 4..(i + 1) * 4].copy_from_slice(&f_bytes);
    }

    // ── Step 3: Copy-out result to device buffer ───────────────────
    // In a real kernel, the native code writes directly to device
    // memory. Here we simulate by copying the kernel output back to a
    // device buffer so readback can retrieve it.
    let copy_out_status =
        unsafe { __faber_gpu_v1_copy_in(43, output_bytes.as_ptr(), output_bytes.len() as u64, 0) };
    assert_eq!(copy_out_status, STATUS_OK);

    // ── Step 4: Readback and verify ─────────────────────────────────
    let mut dest = [0_u8; 16];
    let mut actual_len = 0_u64;
    let readback_status = unsafe {
        __faber_gpu_v1_readback(
            43,
            dest.as_mut_ptr(),
            dest.len() as u64,
            &raw mut actual_len,
        )
    };
    assert_eq!(readback_status, STATUS_OK);
    assert_eq!(actual_len, 16);
    assert_eq!(dest, output_bytes, "readback bytes mismatch");

    let result: [f32; 4] = unsafe { std::mem::transmute::<[u8; 16], [f32; 4]>(dest) };
    assert_eq!(result, [2.0_f32, 4.0, 6.0, 8.0], "kernel output incorrect");

    // ── Step 5: Sync and cleanup ────────────────────────────────────
    let sync_status = unsafe { __faber_gpu_v1_sync(43) };
    assert_eq!(sync_status, STATUS_OK);

    unsafe { __faber_rt_v1_shutdown(context) };
}

// ── G-SPINE-08 Stage 3: Cross-backend golden reference oracle ─────────

// ── Golden provenance ─────────────────────────────────────────────────
//
// Honesty note (2026-07-26, hand-11 residual): No live WebGPU capture was
// performed because the WebGPU host pipeline is not available in the LLVM
// test environment. The golden constants below are a PRE-COMPUTED
// MATHEMATICAL REFERENCE for the deterministic elementwise multiply-by-2
// kernel. Expected output [2.0, 4.0, 6.0, 8.0] is trivially correct for
// input [1.0, 2.0, 3.0, 4.0] and does not depend on a particular backend
// capture. When the WebGPU host pipeline is available, re-capture by
// running the same kernel on the WebGPU host and update this comment.
//
//   Kernel:   elementwise multiply-by-2, rank-1 f32, 4 elements
//   Source:   exempla/tensor/llvm-placement-v1.fab
//   Input:    [1.0, 2.0, 3.0, 4.0]
//   Output:   [2.0, 4.0, 6.0, 8.0]
//   Authored: 2026-07-22 by hand-6 (G-SPINE-08 S3)
//   Honesty:  2026-07-26 by hand-11 (G-SPINE-08 residual)
//
// The test below runs the LLVM exemplar (same copy-in → dispatch →
// readback pipeline as Stage 2) and compares each output element against
// this golden reference. Mismatches report index, expected (golden), and
// actual (LLVM) values.
const GOLDEN_MULTIPLY_BY_TWO: [f32; 4] = [2.0_f32, 4.0, 6.0, 8.0];

/// Cross-backend golden reference oracle: LLVM output vs WebGPU golden.
///
/// Runs the LLVM device execution exemplar and compares each output element
/// against the golden reference captured from an expected WebGPU run.
///
/// # Gating
///
/// `#[ignore]` — requires `faber build --target llvm-host` or LLVM
/// toolchain on PATH. Run with:
/// `cargo test -p faber-host-llvm -- --ignored --nocapture`
#[test]
#[ignore = "G-SPINE-08 S3: requires faber build --target llvm-host or LLVM toolchain on PATH"]
fn llvm_golden_oracle_multiply_by_two() {
    // ── Golden reference ────────────────────────────────────────────
    // Provenance: see GOLDEN_MULTIPLY_BY_TWO const and block comment above.
    let golden: &[f32; 4] = &GOLDEN_MULTIPLY_BY_TWO;

    // ── Step 1: Copy-in input data ──────────────────────────────────
    let input_f32: [f32; 4] = [1.0_f32, 2.0, 3.0, 4.0];
    let input_bytes: [u8; 16] = unsafe { std::mem::transmute::<[f32; 4], [u8; 16]>(input_f32) };

    let copy_in_status =
        unsafe { __faber_gpu_v1_copy_in(42, input_bytes.as_ptr(), input_bytes.len() as u64, 0) };
    assert_eq!(copy_in_status, STATUS_OK, "copy_in failed");

    // ── Step 2: Kernel dispatch (runtime FFI pattern) ──────────────
    let mut context = ptr::null_mut();
    let init_status = unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) };
    assert_eq!(init_status, STATUS_OK);
    assert!(!context.is_null());

    // Shape [4] for rank-1 tensor.
    let shape = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    assert_eq!(shape.status, STATUS_OK);
    {
        let dim = 4_i64;
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    shape.value,
                    VALUE_KIND_I64,
                    std::ptr::from_ref(&dim).cast(),
                )
            },
            STATUS_OK
        );
    }

    // Flat input elements.
    let flat_input = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_F32) };
    assert_eq!(flat_input.status, STATUS_OK);
    for value in &input_f32 {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    flat_input.value,
                    VALUE_KIND_F32,
                    std::ptr::from_ref(value).cast(),
                )
            },
            STATUS_OK
        );
    }

    // Flat scalar multiplier [2.0].
    let flat_two = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_F32) };
    assert_eq!(flat_two.status, STATUS_OK);
    let two = 2.0_f32;
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_push(
                context,
                flat_two.value,
                VALUE_KIND_F32,
                std::ptr::from_ref(&two).cast(),
            )
        },
        STATUS_OK
    );

    // Scalar shape [1].
    let scalar_shape = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    assert_eq!(scalar_shape.status, STATUS_OK);
    let one_dim = 1_i64;
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_push(
                context,
                scalar_shape.value,
                VALUE_KIND_I64,
                std::ptr::from_ref(&one_dim).cast(),
            )
        },
        STATUS_OK
    );

    // Build input tensor.
    let input_tensor = unsafe {
        __faber_rt_v1_tensor_from_flat(context, VALUE_KIND_F32, flat_input.value, shape.value)
    };
    assert_eq!(input_tensor.status, STATUS_OK);

    // Build scalar tensor [2.0].
    let two_tensor = unsafe {
        __faber_rt_v1_tensor_from_flat(context, VALUE_KIND_F32, flat_two.value, scalar_shape.value)
    };
    assert_eq!(two_tensor.status, STATUS_OK);

    // Multiply: @__faber_rt_v1_tensor_mul — same FFI path emitted code uses.
    let product =
        unsafe { __faber_rt_v1_tensor_mul(context, input_tensor.value, two_tensor.value) };
    assert_eq!(product.status, STATUS_OK);

    // Flatten to lista<f32>.
    let flat_result = unsafe { __faber_rt_v1_tensor_flatten(context, product.value) };
    assert_eq!(flat_result.status, STATUS_OK);

    // Extract f32 values from flattened result.
    let runtime_array = unsafe { &*flat_result.value.cast::<RuntimeArray>() };
    assert_eq!(
        runtime_array.values.len(),
        golden.len(),
        "output element count mismatch"
    );
    let mut output_f32 = [0.0_f32; 4];
    for (i, value) in runtime_array.values.iter().enumerate() {
        let array::RuntimeValue::F32(f) = value else {
            panic!("expected F32 at index {i}, got non-F32 RuntimeValue");
        };
        output_f32[i] = f;
    }

    // ── Step 3: Copy-out result to device buffer ───────────────────
    let mut output_bytes = [0_u8; 16];
    for (i, f) in output_f32.iter().enumerate() {
        output_bytes[i * 4..(i + 1) * 4].copy_from_slice(&f32::to_ne_bytes(*f));
    }
    let copy_out_status =
        unsafe { __faber_gpu_v1_copy_in(43, output_bytes.as_ptr(), output_bytes.len() as u64, 0) };
    assert_eq!(copy_out_status, STATUS_OK);

    // ── Step 4: Readback ────────────────────────────────────────────
    let mut dest = [0_u8; 16];
    let mut actual_len = 0_u64;
    let readback_status = unsafe {
        __faber_gpu_v1_readback(
            43,
            dest.as_mut_ptr(),
            dest.len() as u64,
            &raw mut actual_len,
        )
    };
    assert_eq!(readback_status, STATUS_OK);
    assert_eq!(actual_len, 16);
    let llvm_result: [f32; 4] = unsafe { std::mem::transmute::<[u8; 16], [f32; 4]>(dest) };

    // ── Step 5: Elementwise oracle comparison ───────────────────────
    // Compare each element individually. On mismatch, report the index,
    // expected golden value, and actual LLVM output value.
    for (i, (actual, expected)) in llvm_result.iter().zip(golden.iter()).enumerate() {
        assert_eq!(
            actual, expected,
            "oracle mismatch at index {i}: LLVM output {actual} != golden {expected}"
        );
    }

    // ── Step 6: Sync and cleanup ────────────────────────────────────
    let sync_status = unsafe { __faber_gpu_v1_sync(43) };
    assert_eq!(sync_status, STATUS_OK);

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn solum_read_lines_splits_file_into_text_lista() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let path =
        std::env::temp_dir().join(format!("faber-solum-read-lines-{}.txt", std::process::id()));
    std::fs::write(&path, "prima\nsecunda\n").expect("write fixture file");
    let path = path.to_string_lossy().into_owned();
    let path_slice = FaberRtSliceV1 {
        data: path.as_ptr(),
        len: path.len() as u64,
    };

    let result = unsafe { __faber_rt_v1_solum_read_lines(context, &raw const path_slice) };
    assert_eq!(result.status, STATUS_OK);
    let array = unsafe { &*result.value.cast::<RuntimeArray>() };
    assert_eq!(array.kind, VALUE_KIND_PTR);
    let lines = array
        .values
        .iter()
        .map(|value| match value {
            array::RuntimeValue::Ptr(handle) => {
                unsafe { &*handle.cast::<RuntimeText>() }.value.as_str()
            }
            _ => panic!("read_lines produced non-text carrier"),
        })
        .collect::<Vec<_>>();
    assert_eq!(lines, ["prima", "secunda"]);

    std::fs::remove_file(&path).ok();
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn solum_read_bytes_reads_raw_file_bytes() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let path =
        std::env::temp_dir().join(format!("faber-solum-read-bytes-{}.txt", std::process::id()));
    std::fs::write(&path, b"\x00\x01prima\nsecunda\n").expect("write fixture file");
    let path = path.to_string_lossy().into_owned();
    let path_slice = FaberRtSliceV1 {
        data: path.as_ptr(),
        len: path.len() as u64,
    };

    let result = unsafe { __faber_rt_v1_solum_read_bytes(context, &raw const path_slice) };
    assert_eq!(result.status, STATUS_OK);
    assert_eq!(
        unsafe { &*result.value.cast::<Vec<u8>>() },
        b"\x00\x01prima\nsecunda\n"
    );

    std::fs::remove_file(&path).ok();
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn opaque_nota_renders_lista_textus_and_octeti_in_rust_debug_shape() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let runtime = unsafe { &mut *context.cast::<RuntimeContext>() };

    // lista<textus>: array of text handles renders ["prima", "secunda"].
    let mut handles = Vec::new();
    for line in ["prima", "secunda"] {
        let text = format::store_text(context, line.to_owned());
        assert_eq!(text.status, STATUS_OK);
        handles.push(array::RuntimeValue::Ptr(text.value));
    }
    let array = array::store_array(runtime, radix_host_abi::VALUE_KIND_PTR, handles);
    assert_eq!(array.status, STATUS_OK);
    assert_eq!(
        super::opaque_diagnostic_text(runtime, array.value),
        Some(r#"["prima", "secunda"]"#.to_owned())
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_nota_ptr(context, array.value.cast()) },
        STATUS_OK
    );

    // octeti: byte payload renders as decimal byte list.
    let octeti = valor_aggregate::store_octeti(runtime, b"prima\n".to_vec());
    assert_eq!(octeti.status, STATUS_OK);
    assert_eq!(
        super::opaque_diagnostic_text(runtime, octeti.value),
        Some("[112, 114, 105, 109, 97, 10]".to_owned())
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_nota_ptr(context, octeti.value.cast()) },
        STATUS_OK
    );

    // Unrecognized handle stays fail-closed.
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_nota_ptr(context, ptr::null()) },
        STATUS_UNSUPPORTED
    );

    unsafe { __faber_rt_v1_shutdown(context) };
}

/// L10 (fa1a5d8c): numeric `lista` elements render in the Rust oracle's Debug
/// shape (`[1.0, 4.0, 9.0, 16.0]` / `[2, 3]`), and a `valor` renders via the
/// oracle's `display_valor` (`42`, `{"alpha": 10}`) with octeti payloads
/// rendering as byte lists (`[222, 173]`, the oracle's `bytes ↦ valor` Lista
/// Debug shape).
#[test]
fn opaque_nota_renders_numeric_lista_and_valor_in_rust_debug_shape() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let runtime = unsafe { &mut *context.cast::<RuntimeContext>() };

    // lista<f32>: array of f32 elements renders [1.0, 4.0, 9.0, 16.0].
    let f32_array = array::store_array(
        runtime,
        radix_host_abi::VALUE_KIND_F32,
        vec![
            array::RuntimeValue::F32(1.0),
            array::RuntimeValue::F32(4.0),
            array::RuntimeValue::F32(9.0),
            array::RuntimeValue::F32(16.0),
        ],
    );
    assert_eq!(f32_array.status, STATUS_OK);
    assert_eq!(
        super::opaque_diagnostic_text(runtime, f32_array.value),
        Some("[1.0, 4.0, 9.0, 16.0]".to_owned())
    );

    // lista<numerus>: renders [2, 3].
    let i64_array = array::store_array(
        runtime,
        radix_host_abi::VALUE_KIND_I64,
        vec![array::RuntimeValue::I64(2), array::RuntimeValue::I64(3)],
    );
    assert_eq!(i64_array.status, STATUS_OK);
    assert_eq!(
        super::opaque_diagnostic_text(runtime, i64_array.value),
        Some("[2, 3]".to_owned())
    );

    // valor numerus renders its displayed magnitude.
    let numerus = convert::store_valor(context, faber::Valor::Numerus(42));
    assert_eq!(numerus.status, STATUS_OK);
    assert_eq!(
        super::opaque_diagnostic_text(runtime, numerus.value),
        Some("42".to_owned())
    );

    // valor octeti renders the byte-list Debug shape ([222, 173]).
    let octeti_valor = convert::store_valor(context, faber::Valor::Octeti(vec![0xde, 0xad]));
    assert_eq!(octeti_valor.status, STATUS_OK);
    assert_eq!(
        super::opaque_diagnostic_text(runtime, octeti_valor.value),
        Some("[222, 173]".to_owned())
    );

    // valor tabula renders display_valor's map shape.
    let mut tabula = std::collections::BTreeMap::new();
    tabula.insert("alpha".to_owned(), faber::Valor::Numerus(10));
    let tabula_valor = convert::store_valor(context, faber::Valor::Tabula(tabula));
    assert_eq!(tabula_valor.status, STATUS_OK);
    assert_eq!(
        super::opaque_diagnostic_text(runtime, tabula_valor.value),
        Some(r#"{"alpha": 10}"#.to_owned())
    );

    unsafe { __faber_rt_v1_shutdown(context) };
}

/// L10 (fa1a5d8c): a `tabula` handle renders in the Rust oracle's derived
/// `Json(Valor::Tabula({...}))` Debug shape, and a `copia` handle renders
/// `{1, 2, 3}` in stored order.
#[test]
fn opaque_nota_renders_tabula_and_copia_in_rust_debug_shape() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let runtime = unsafe { &mut *context.cast::<RuntimeContext>() };

    // tabula<textus, numerus>: JSON-literal map renders Json(Tabula({...})).
    let key = format::store_text(context, "alpha".to_owned());
    assert_eq!(key.status, STATUS_OK);
    let map = collection_map::store_map(
        runtime,
        radix_host_abi::VALUE_KIND_TEXT,
        radix_host_abi::VALUE_KIND_I64,
        vec![(
            array::RuntimeValue::Ptr(key.value),
            array::RuntimeValue::I64(10),
        )],
    );
    assert_eq!(map.status, STATUS_OK);
    assert_eq!(
        super::opaque_diagnostic_text(runtime, map.value),
        Some(r#"Json(Tabula({"alpha": Numerus(10)}))"#.to_owned())
    );

    // copia<numerus>: renders {1, 2, 3}.
    let set = collection_map::store_set(
        runtime,
        radix_host_abi::VALUE_KIND_I64,
        vec![
            array::RuntimeValue::I64(1),
            array::RuntimeValue::I64(2),
            array::RuntimeValue::I64(3),
        ],
    );
    assert_eq!(set.status, STATUS_OK);
    assert_eq!(
        super::opaque_diagnostic_text(runtime, set.value),
        Some("{1, 2, 3}".to_owned())
    );

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn instans_compare_family_orders_handles() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let earlier = FaberRtSliceV1::from_static(b"1979-05-27T07:32:00Z");
    let later = FaberRtSliceV1::from_static(b"1980-01-01T00:00:00Z");
    let earlier = unsafe {
        __faber_rt_v1_instans_from_text(context, &raw const earlier, INSTANS_PRECISION_SECONDS)
    };
    let later = unsafe {
        __faber_rt_v1_instans_from_text(context, &raw const later, INSTANS_PRECISION_SECONDS)
    };
    assert_eq!(earlier.status, STATUS_OK);
    assert_eq!(later.status, STATUS_OK);
    let (a, b) = (earlier.value, later.value);
    assert_eq!(unsafe { __faber_rt_v1_compare_lt_2_ptr_ptr_to_i1(a, b) }, 1);
    assert_eq!(unsafe { __faber_rt_v1_compare_lt_2_ptr_ptr_to_i1(b, a) }, 0);
    assert_eq!(unsafe { __faber_rt_v1_compare_gt_2_ptr_ptr_to_i1(b, a) }, 1);
    assert_eq!(unsafe { __faber_rt_v1_compare_gt_2_ptr_ptr_to_i1(a, b) }, 0);
    assert_eq!(
        unsafe { __faber_rt_v1_compare_lte_2_ptr_ptr_to_i1(a, b) },
        1
    );
    assert_eq!(
        unsafe { __faber_rt_v1_compare_lte_2_ptr_ptr_to_i1(a, a) },
        1
    );
    assert_eq!(
        unsafe { __faber_rt_v1_compare_gte_2_ptr_ptr_to_i1(b, a) },
        1
    );
    assert_eq!(
        unsafe { __faber_rt_v1_compare_gte_2_ptr_ptr_to_i1(a, b) },
        0
    );
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn tempus_nunc_returns_current_instant_handle() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let nunc = unsafe { __faber_rt_v1_tempus_nunc(context) };
    assert_eq!(nunc.status, STATUS_OK);
    assert!(!nunc.value.is_null());
    let rendered = unsafe { __faber_rt_v1_instans_get_text(context, nunc.value) };
    assert_eq!(rendered.status, STATUS_OK);
    let rendered = unsafe { &*rendered.value.cast::<FaberRtSliceV1>() };
    let text = unsafe {
        std::str::from_utf8_unchecked(std::slice::from_raw_parts(
            rendered.data,
            rendered.len as usize,
        ))
    };
    assert!(
        text.ends_with('Z'),
        "RFC3339 wire should end with Z, got {text}"
    );
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn provider_valor_cape_reads_tabula_field() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let map = unsafe { __faber_rt_v1_map_new(context, VALUE_KIND_TEXT, VALUE_KIND_I64) };
    assert_eq!(map.status, STATUS_OK);
    let key = FaberRtSliceV1::from_static(b"creatus");
    let key_handle = ptr::from_ref(&key).cast_mut().cast::<c_void>();
    let value = 42_i64;
    assert_eq!(
        unsafe {
            __faber_rt_v1_map_put(
                context,
                map.value,
                VALUE_KIND_TEXT,
                ptr::from_ref(&key_handle).cast(),
                VALUE_KIND_I64,
                ptr::from_ref(&value).cast(),
            )
        },
        STATUS_OK
    );
    let valor = unsafe { __faber_rt_v1_valor_map(context, map.value) };
    assert_eq!(valor.status, STATUS_OK);
    let field = unsafe { __faber_rt_v1_valor_cape(context, valor.value.cast(), &raw const key) };
    assert_eq!(field.status, STATUS_OK);
    assert_eq!(
        unsafe { &*field.value.cast::<Valor>() },
        &Valor::Numerus(42)
    );
    let missing = FaberRtSliceV1::from_static(b"absentia");
    let field =
        unsafe { __faber_rt_v1_valor_cape(context, valor.value.cast(), &raw const missing) };
    assert_eq!(field.status, STATUS_INVALID_ARGUMENT);
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn provider_json_solve_pange_round_trip() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let wire = FaberRtSliceV1::from_static(br#"{"nomen":"Ada"}"#);
    let solved = unsafe { __faber_rt_v1_json_solve(context, &raw const wire) };
    assert_eq!(solved.status, STATUS_OK);
    let panged = unsafe { __faber_rt_v1_json_pange(context, solved.value.cast()) };
    assert_eq!(panged.status, STATUS_OK);
    let panged = unsafe { &*panged.value.cast::<FaberRtSliceV1>() };
    assert_eq!(
        unsafe { std::slice::from_raw_parts(panged.data, panged.len as usize) },
        br#"{"nomen":"Ada"}"#
    );
    let bad = FaberRtSliceV1::from_static(b"{ nope");
    let solved = unsafe { __faber_rt_v1_json_solve(context, &raw const bad) };
    assert_eq!(solved.status, STATUS_INVALID_ARGUMENT);
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn provider_json_tempta_boxes_text_payload() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let wire = FaberRtSliceV1::from_static(br#"{"ok":true}"#);
    let tentativa = unsafe { __faber_rt_v1_json_tempta(context, &raw const wire) };
    assert_eq!(tentativa.status, STATUS_OK);
    assert!(!tentativa.value.is_null());
    // The union box holds the text form of the payload.
    let payload = unsafe { *(tentativa.value.cast::<*mut c_void>()) };
    let payload = unsafe { &*payload.cast::<FaberRtSliceV1>() };
    assert_eq!(
        unsafe { std::slice::from_raw_parts(payload.data, payload.len as usize) },
        br#"{"ok":true}"#
    );
    let nil = FaberRtSliceV1::from_static(b"{ nope");
    let tentativa = unsafe { __faber_rt_v1_json_tempta(context, &raw const nil) };
    assert_eq!(tentativa.status, STATUS_OK);
    let payload = unsafe { *(tentativa.value.cast::<*mut c_void>()) };
    let payload = unsafe { &*payload.cast::<FaberRtSliceV1>() };
    assert_eq!(
        unsafe { std::slice::from_raw_parts(payload.data, payload.len as usize) },
        b"nihil"
    );
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn provider_toml_solve_fails_closed_unsupported() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let wire = FaberRtSliceV1::from_static(b"creatus = 1979-05-27T07:32:00Z");
    let solved = unsafe { __faber_rt_v1_toml_solve(context, &raw const wire) };
    assert_eq!(solved.status, STATUS_UNSUPPORTED);
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn opaque_ptr_conversion_preserves_handle() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let wire = FaberRtSliceV1::from_static(b"ada");
    let text = unsafe { __faber_rt_v1_valor_text(context, &raw const wire) };
    assert_eq!(text.status, STATUS_OK);
    let converted = unsafe { __faber_rt_v1_convert_runtime_1_ptr_to_ptr(context, text.value) };
    assert_eq!(converted.status, STATUS_OK);
    assert_eq!(converted.value, text.value);
    let null = unsafe { __faber_rt_v1_convert_runtime_1_ptr_to_ptr(context, ptr::null_mut()) };
    assert_eq!(null.status, STATUS_INVALID_ARGUMENT);
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn option_unwrap_ptr_passes_through_box() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let boxed = super::StableBox::new(ptr::null_mut::<c_void>());
    let handle = boxed.handle();
    assert_eq!(unsafe { __faber_rt_v1_option_unwrap_ptr(handle) }, handle);
    drop(boxed);
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn aggregate_set_index_ptr_i64_sets_text_i64_map_entry() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let map = unsafe { __faber_rt_v1_map_new(context, VALUE_KIND_TEXT, VALUE_KIND_I64) };
    assert_eq!(map.status, STATUS_OK);
    let key = FaberRtSliceV1::from_static(b"alpha");
    unsafe { __faber_rt_v1_aggregate_set_index_ptr_i64(map.value, &raw const key, 7) };
    let mut output = 0_i64;
    let key_handle = ptr::from_ref(&key).cast_mut().cast::<c_void>();
    let status = unsafe {
        __faber_rt_v1_map_get(
            context,
            map.value,
            VALUE_KIND_TEXT,
            ptr::from_ref(&key_handle).cast(),
            VALUE_KIND_I64,
            ptr::from_mut(&mut output).cast(),
        )
    };
    assert_eq!(status, STATUS_OK);
    assert_eq!(output, 7);
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn format_1_ptr_to_ptr_renders_opaque_lista_debug_shape() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let array = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    assert_eq!(array.status, STATUS_OK);
    for value in [1_i64, 2, 3] {
        let status = unsafe {
            __faber_rt_v1_array_push(
                context,
                array.value,
                VALUE_KIND_I64,
                ptr::from_ref(&value).cast(),
            )
        };
        assert_eq!(status, STATUS_OK);
    }
    let formatted = unsafe {
        __faber_rt_v1_format_1_ptr_to_ptr(
            context,
            FaberRtSliceV1::from_static(b"nums=\xC2\xA7"),
            array.value,
        )
    };
    assert_eq!(formatted.status, STATUS_OK);
    assert_eq!(
        unsafe { &*formatted.value.cast::<RuntimeText>() }.value,
        "nums=[1, 2, 3]"
    );
    // Unknown handles fail closed (STATUS_UNSUPPORTED), never panic.
    let bogus = unsafe {
        __faber_rt_v1_format_1_ptr_to_ptr(
            context,
            FaberRtSliceV1::from_static(b"x"),
            ptr::null_mut(),
        )
    };
    assert_eq!(bogus.status, STATUS_UNSUPPORTED);
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn regex_literal_1_ptr_to_ptr_builds_pattern_carrier() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );
    let pattern = b"\\d+\0";
    let flags = b"i\0";
    let descriptor = super::regex_rt::RegexLiteralDescriptorV1 {
        pattern: pattern.as_ptr().cast(),
        flags: flags.as_ptr().cast(),
    };
    let result =
        unsafe { __faber_rt_v1_regex_literal_1_ptr_to_ptr(context, &raw const descriptor) };
    assert_eq!(result.status, STATUS_OK);
    let text = unsafe { __faber_rt_v1_regex_get_text(context, result.value) };
    assert_eq!(text.status, STATUS_OK);
    assert_eq!(unsafe { &*text.value.cast::<RuntimeText>() }.value, "\\d+");
    // Null descriptor fails closed with STATUS_INVALID_ARGUMENT.
    let invalid = unsafe { __faber_rt_v1_regex_literal_1_ptr_to_ptr(context, ptr::null()) };
    assert_eq!(invalid.status, STATUS_INVALID_ARGUMENT);
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn read_line_0_to_ptr_rejects_null_context_and_links() {
    // The symbol must be exported and fail closed on a null context; the
    // live-stdin read path is exercised by the repro link pipeline (stdin is
    // a process-global the test runner cannot inject).
    let result = unsafe { __faber_rt_v1_read_line_0_to_ptr(ptr::null_mut()) };
    assert_eq!(result.status, STATUS_INVALID_ARGUMENT);
}

// ---------------------------------------------------------------------------
// Stage 8 S8.2 — static CLI descriptor decode, argv parse, exit policy.
// ---------------------------------------------------------------------------

/// Hand-constructed `cli_descriptor` v1 bytes for a single-command program
/// with one numeric operand (`exitum`) and a `Binding` exit policy, in the
/// radix `cli_descriptor` byte format (an independent decoder check).
fn descriptor_single_numerus_binding() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"FCLI");
    bytes.push(1); // version
    bytes.push(0); // mode single
    bytes.push(0); // no version
    bytes.push(0); // no description
                   // name
    bytes.extend_from_slice(&5u16.to_le_bytes());
    bytes.extend_from_slice(b"smoke");
    // exit: Binding("exitum")
    bytes.push(2);
    bytes.extend_from_slice(&6u16.to_le_bytes());
    bytes.extend_from_slice(b"exitum");
    // global options: 0
    bytes.extend_from_slice(&0u16.to_le_bytes());
    // global operands: 0
    bytes.extend_from_slice(&0u16.to_le_bytes());
    // options: 0
    bytes.extend_from_slice(&0u16.to_le_bytes());
    // operands: 1 (numerus, rest=false, no desc, no default, "exitum")
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.push(1); // ty numerus
    bytes.push(0); // rest
    bytes.push(0); // has_description
    bytes.push(0); // has_default
    bytes.extend_from_slice(&6u16.to_le_bytes());
    bytes.extend_from_slice(b"exitum");
    // commands: 0
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes
}

#[test]
fn cli_parse_returns_typed_value_table_and_binding_exit_code() {
    let descriptor = descriptor_single_numerus_binding();
    let program = std::ffi::CString::new("prog").unwrap();
    let operand = std::ffi::CString::new("7").unwrap();
    let argv = [program.as_ptr(), operand.as_ptr()];
    let mut context = ptr::null_mut();
    let status = unsafe { __faber_rt_v1_init(2, argv.as_ptr(), &raw mut context) };
    assert_eq!(status, STATUS_OK);

    let result = unsafe { __faber_rt_v1_cli_parse(context, descriptor.as_ptr(), descriptor.len()) };
    assert!(
        result.status.is_ok(),
        "descriptor decode + argv parse must succeed"
    );
    assert!(
        !result.value.is_null(),
        "typed value table must be returned"
    );
    assert_eq!(
        unsafe { __faber_rt_v1_cli_field_i64(context, result.value, 0) },
        7,
        "numeric operand must parse to i64"
    );
    // The Binding exit policy resolves the record field by binding name.
    assert_eq!(
        unsafe { __faber_rt_v1_cli_exit_code(context) },
        7,
        "Binding exit policy must resolve the numeric record field"
    );
    assert_eq!(
        unsafe { __faber_rt_v1_cli_selected_command(context) },
        -1,
        "single-command mode has no selected command"
    );
    unsafe { __faber_rt_v1_shutdown(context) };
}

// ===========================================================================
// P8 — sermo materialization surface (promotion packet sermo-runtime-surface)
// ===========================================================================
//
// The five ad/sermo-* fixtures' captured shapes each get a row here:
//   - sermo-conversio  (convert_1_aggregate_to_text):  SermoOpen + sermo->textus
//   - sermo-recovery   (convert_2_aggregate_i64_to_i64): sermo->valor + the
//     _or scalar recovery row (fallback substitutes on type mismatch)
//   - sermo-live-directional / sermo-tuus / sermo-vacuum (value-returning
//     runtime call): SermoOpen now returns an opaque stream handle

fn sermo_context() -> *mut FaberRtContextV1 {
    let mut context = ptr::null_mut();
    let status = unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) };
    assert_eq!(status, STATUS_OK);
    assert!(!context.is_null());
    context
}

/// Store a payload valor in the arena so the binding can read it by pointer.
fn store_payload(context: *mut FaberRtContextV1, value: Valor) -> *const Valor {
    let result = convert::store_valor(context, value);
    assert!(result.status.is_ok(), "payload valor must store");
    result.value.cast::<Valor>()
}

fn find_numeric(runtime: &RuntimeContext, handle: *mut c_void) -> Option<i64> {
    runtime
        .numeric_boxes
        .iter()
        .find(|value| std::ptr::eq(value.as_ref(), handle.cast()))
        .map(StableBox::as_ref)
        .copied()
}

#[test]
fn sermo_open_returns_an_opaque_stream_handle() {
    let context = sermo_context();
    let payload = store_payload(context, Valor::Nihil);
    let opened = unsafe {
        __faber_rt_v1_sermo_open(
            context,
            &FaberRtSliceV1::from_static(b"runtime:echo"),
            payload,
        )
    };
    assert!(opened.status.is_ok(), "SermoOpen must succeed");
    assert!(
        !opened.value.is_null(),
        "SermoOpen must return an opaque stream handle (value-returning runtime call)"
    );
    let runtime = unsafe { &*context.cast::<RuntimeContext>() };
    assert_eq!(
        runtime.sermos.len(),
        1,
        "stream handle registered in the arena"
    );
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn sermo_open_and_materialize_text_echoes_the_opener() {
    let context = sermo_context();
    let payload = store_payload(context, Valor::Textus("salve, munde".into()));
    let opened = unsafe {
        __faber_rt_v1_sermo_open(
            context,
            &FaberRtSliceV1::from_static(b"runtime:echo"),
            payload,
        )
    };
    assert!(opened.status.is_ok(), "SermoOpen must succeed");

    // ad/sermo-conversio.fab shape: `ad 'runtime:echo'(payload) ↦ textus`.
    let materialized = unsafe { __faber_rt_v1_sermo_materialize_text(context, opened.value) };
    assert!(materialized.status.is_ok(), "sermo -> textus must succeed");
    let runtime = unsafe { &*context.cast::<RuntimeContext>() };
    let text = format::find_text(runtime, materialized.value).expect("textus in arena");
    assert_eq!(text.value, "salve, munde");
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn sermo_materialize_valor_returns_the_frame_payload() {
    let context = sermo_context();
    let payload = store_payload(context, Valor::Textus("salve".into()));
    let opened = unsafe {
        __faber_rt_v1_sermo_open(
            context,
            &FaberRtSliceV1::from_static(b"runtime:echo"),
            payload,
        )
    };
    assert!(opened.status.is_ok());

    // ad/sermo-* shape: `sermo ↦ valor` — the stream materializes to a valor
    // carrier (recovery fixtures then extract scalars from it).
    let result = unsafe { __faber_rt_v1_sermo_materialize_valor(context, opened.value) };
    assert!(result.status.is_ok(), "sermo -> valor must succeed");
    let runtime = unsafe { &*context.cast::<RuntimeContext>() };
    let valor = convert::find_valor(runtime, result.value).expect("valor in arena");
    assert_eq!(valor, &Valor::Textus("salve".into()));
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn sermo_set_opener_replaces_the_request_payload() {
    let context = sermo_context();
    let payload = store_payload(context, Valor::Nihil);
    let opened = unsafe {
        __faber_rt_v1_sermo_open(
            context,
            &FaberRtSliceV1::from_static(b"runtime:echo"),
            payload,
        )
    };
    assert!(opened.status.is_ok());

    // SermoSetOpener replaces the first request frame's payload before the
    // stream is consumed; runtime:echo echoes the opener back.
    let ping = store_payload(context, Valor::Textus("ping".into()));
    let status = unsafe { __faber_rt_v1_sermo_set_opener(context, opened.value, ping) };
    assert!(status.is_ok(), "SermoSetOpener must succeed");

    let materialized = unsafe { __faber_rt_v1_sermo_materialize_text(context, opened.value) };
    assert!(materialized.status.is_ok());
    let runtime = unsafe { &*context.cast::<RuntimeContext>() };
    let text = format::find_text(runtime, materialized.value).expect("textus in arena");
    assert_eq!(text.value, "ping");
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn sermo_materialize_i64_or_extracts_a_numeric_payload() {
    let context = sermo_context();
    let payload = store_payload(context, Valor::Numerus(42));
    let opened = unsafe {
        __faber_rt_v1_sermo_open(
            context,
            &FaberRtSliceV1::from_static(b"runtime:echo"),
            payload,
        )
    };
    assert!(opened.status.is_ok());

    let result = unsafe { __faber_rt_v1_sermo_materialize_i64_or(context, opened.value, 0) };
    assert!(result.status.is_ok());
    let runtime = unsafe { &*context.cast::<RuntimeContext>() };
    assert_eq!(find_numeric(runtime, result.value), Some(42));
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn sermo_materialize_i64_or_recovers_on_type_mismatch() {
    let context = sermo_context();
    let payload = store_payload(context, Valor::Textus("non-numeric".into()));
    let opened = unsafe {
        __faber_rt_v1_sermo_open(
            context,
            &FaberRtSliceV1::from_static(b"runtime:echo"),
            payload,
        )
    };
    assert!(opened.status.is_ok());

    // ad/sermo-recovery.fab shape: `ad 'runtime:echo'(payload) ↦ i64 ⇥ 0` —
    // the echo returns textus, the scalar extraction fails, and the `_or`
    // fallback substitutes instead of aborting (convert_2_aggregate_i64_to_i64).
    let result = unsafe { __faber_rt_v1_sermo_materialize_i64_or(context, opened.value, 0) };
    assert!(result.status.is_ok(), "recovery row must not abort");
    let runtime = unsafe { &*context.cast::<RuntimeContext>() };
    assert_eq!(find_numeric(runtime, result.value), Some(0));
    unsafe { __faber_rt_v1_shutdown(context) };
}

/// ABI round-trip timings for the hosts runtime-cell / handle-scan work.
///
/// Prints wall times and the in-arena f32 payload footprint. This is a
/// measurement, not a performance gate.
#[test]
fn abi_roundtrip_perf_measurement() {
    use std::time::Instant;

    eprintln!(
        "RuntimeValue size: {} bytes",
        std::mem::size_of::<array::RuntimeValue>()
    );

    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &raw mut context) },
        STATUS_OK
    );

    const N: i64 = 20_000;
    const HANDLE_COUNT: usize = 2_000;

    let array = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_F32) };
    assert!(array.status.is_ok());
    let started = Instant::now();
    for index in 0..N {
        let value = index as f32;
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    array.value,
                    VALUE_KIND_F32,
                    std::ptr::from_ref(&value).cast(),
                )
            },
            STATUS_OK
        );
    }
    let mut out = 0.0_f32;
    for index in 0..N {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_get(
                    context,
                    array.value,
                    index,
                    VALUE_KIND_F32,
                    std::ptr::from_mut(&mut out).cast(),
                )
            },
            STATUS_OK
        );
    }
    eprintln!("array push+get {N} f32: {:?}", started.elapsed());

    let runtime_array = unsafe { &*array.value.cast::<RuntimeArray>() };
    eprintln!(
        "f32 array {N} payload bytes: {}",
        runtime_array.values.payload_bytes()
    );

    let source = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_F32) };
    let target = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_F32) };
    for index in 0..10_000 {
        let value = index as f32;
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    source.value,
                    VALUE_KIND_F32,
                    std::ptr::from_ref(&value).cast(),
                )
            },
            STATUS_OK
        );
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    target.value,
                    VALUE_KIND_F32,
                    std::ptr::from_ref(&value).cast(),
                )
            },
            STATUS_OK
        );
    }
    let started = Instant::now();
    assert_eq!(
        unsafe { __faber_rt_v1_array_extend(context, target.value, source.value) },
        STATUS_OK
    );
    eprintln!("array_extend 10000+10000 f32: {:?}", started.elapsed());

    let mut handles = Vec::with_capacity(HANDLE_COUNT);
    for _ in 0..HANDLE_COUNT {
        let created = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
        assert!(created.status.is_ok());
        handles.push(created.value);
    }
    let started = Instant::now();
    let mut length = 0_i64;
    for _ in 0..20 {
        for handle in &handles {
            assert_eq!(
                unsafe { __faber_rt_v1_array_length(context, *handle, &raw mut length) },
                STATUS_OK
            );
        }
    }
    eprintln!(
        "array_length {HANDLE_COUNT} handles x 20: {:?}",
        started.elapsed()
    );

    let shape = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    let dim = 64_i64;
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_push(
                context,
                shape.value,
                VALUE_KIND_I64,
                std::ptr::from_ref(&dim).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_push(
                context,
                shape.value,
                VALUE_KIND_I64,
                std::ptr::from_ref(&dim).cast(),
            )
        },
        STATUS_OK
    );
    let fill = 1.0_f32;
    let tensor = unsafe {
        __faber_rt_v1_tensor_create(
            context,
            VALUE_KIND_F32,
            std::ptr::from_ref(&fill).cast(),
            shape.value,
        )
    };
    assert!(tensor.status.is_ok());

    let started = Instant::now();
    for _ in 0..200 {
        let flat = unsafe { __faber_rt_v1_tensor_flatten(context, tensor.value) };
        assert!(flat.status.is_ok());
    }
    eprintln!("tensor_flatten 64x64 f32 x200: {:?}", started.elapsed());

    let started = Instant::now();
    for _ in 0..200 {
        let sliced = unsafe { __faber_rt_v1_tensor_slice(context, tensor.value, 0, 32) };
        assert!(sliced.status.is_ok());
    }
    eprintln!(
        "tensor_slice 64x64->32x64 f32 x200: {:?}",
        started.elapsed()
    );

    let started = Instant::now();
    for _ in 0..100 {
        let sum = unsafe { __faber_rt_v1_tensor_add(context, tensor.value, tensor.value) };
        assert!(sum.status.is_ok());
    }
    eprintln!("tensor_add 64x64 f32 x100: {:?}", started.elapsed());

    unsafe { __faber_rt_v1_shutdown(context) };
}
