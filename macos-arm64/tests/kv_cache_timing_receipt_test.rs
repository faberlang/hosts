//! F4H1 proof: the KV-cache receipt keeps wall terms and lifecycle facts
//! explicit without wiring the later F4H2 session hookup.

#[allow(dead_code)]
mod device_descriptor {
    pub use faber_host_macos_arm64::device_descriptor::{
        DeviceBufferLifetime, DeviceBufferRole, DeviceDataType, DeviceProgramLifetime,
    };
}

#[allow(dead_code)]
mod kernel {
    pub mod library_runtime {
        pub use faber_host_macos_arm64::kernel::library_runtime::FusedLibraryDispatchReceipt;
    }
}

#[path = "../src/composite_host/receipt.rs"]
mod receipt;

use receipt::{
    KvCacheLifecycleReceipt, KvCacheMeasurement, KvCachePhaseTiming, KvCacheTimingReceipt,
    KvCacheTimingSpan,
};

fn measured_span(start_us: u64, end_us: u64, duration_us: u64) -> KvCacheTimingSpan {
    KvCacheTimingSpan {
        start_us: KvCacheMeasurement::measured(start_us),
        end_us: KvCacheMeasurement::measured(end_us),
        duration_us: KvCacheMeasurement::measured(duration_us),
    }
}

#[test]
fn missing_timestamps_are_explicit_and_never_zero_defaults() {
    let missing = KvCacheTimingSpan::not_measured();

    assert_eq!(missing.start_us, KvCacheMeasurement::NotMeasured);
    assert_eq!(missing.end_us, KvCacheMeasurement::NotMeasured);
    assert_eq!(missing.duration_us, KvCacheMeasurement::NotMeasured);

    let encoded = serde_json::to_value(KvCacheMeasurement::NotMeasured).expect("encode status");
    assert_eq!(encoded["status"], "not_measured");
    assert!(encoded.get("value_us").is_none());
}

#[test]
fn setup_and_steady_state_keep_body_encode_submit_and_wait_separate() {
    let setup = KvCachePhaseTiming {
        gpu_body: measured_span(10, 20, 10),
        encode: measured_span(21, 25, 4),
        submit: measured_span(26, 27, 1),
        wait: measured_span(28, 35, 7),
    };
    let steady = KvCachePhaseTiming {
        gpu_body: measured_span(100, 140, 40),
        encode: measured_span(141, 145, 4),
        submit: measured_span(146, 147, 1),
        wait: KvCacheTimingSpan::not_measured(),
    };
    let receipt = KvCacheTimingReceipt {
        setup_phase: setup,
        steady_state: steady,
        slack_us: KvCacheMeasurement::derived(3),
        lifecycle: KvCacheLifecycleReceipt::zero(),
    };

    assert_eq!(
        receipt.setup_phase.gpu_body.duration_us,
        KvCacheMeasurement::measured(10)
    );
    assert_eq!(
        receipt.setup_phase.encode.duration_us,
        KvCacheMeasurement::measured(4)
    );
    assert_eq!(
        receipt.setup_phase.submit.duration_us,
        KvCacheMeasurement::measured(1)
    );
    assert_eq!(
        receipt.setup_phase.wait.duration_us,
        KvCacheMeasurement::measured(7)
    );
    assert_eq!(
        receipt.steady_state.gpu_body.duration_us,
        KvCacheMeasurement::measured(40)
    );
    assert_eq!(
        receipt.steady_state.encode.duration_us,
        KvCacheMeasurement::measured(4)
    );
    assert_eq!(
        receipt.steady_state.submit.duration_us,
        KvCacheMeasurement::measured(1)
    );
    assert_eq!(receipt.steady_state.wait, KvCacheTimingSpan::not_measured());
    assert_eq!(receipt.slack_us, KvCacheMeasurement::derived(3));
}

#[test]
fn lifecycle_receipt_carries_residency_and_cache_byte_facts() {
    let receipt = KvCacheTimingReceipt {
        setup_phase: KvCachePhaseTiming::not_measured(),
        steady_state: KvCachePhaseTiming::not_measured(),
        slack_us: KvCacheMeasurement::NotMeasured,
        lifecycle: KvCacheLifecycleReceipt {
            module_reloads: 2,
            persistent_reallocations: 1,
            weight_uploads: 3,
            old_prefix_copy_bytes: 4096,
            full_cache_clear_bytes: 8192,
        },
    };

    assert_eq!(receipt.lifecycle.module_reloads, 2);
    assert_eq!(receipt.lifecycle.persistent_reallocations, 1);
    assert_eq!(receipt.lifecycle.weight_uploads, 3);
    assert_eq!(receipt.lifecycle.old_prefix_copy_bytes, 4096);
    assert_eq!(receipt.lifecycle.full_cache_clear_bytes, 8192);
}

#[test]
fn f7_receipt_addendum_names_option_a_and_fused_sync_ruling() {
    let addendum = include_str!("kv_cache_timing_receipt.receipt.md");

    for needle in [
        "Option **(a)** is accepted",
        "host_product_work_us",
        "fused term without an independent clock",
        "stays `not_measured` with\nthat fusion reason",
        "fused-sync separation is deferred",
        "`unattributed` also stays",
        "`not_measured`; this unit does not invent a residual",
    ] {
        assert!(
            addendum.contains(needle),
            "receipt addendum missing {needle}"
        );
    }
}
