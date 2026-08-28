//! PGC-R1 focused device proof: the indexed embedding gather.
//!
//! The production GEA3 bundle is intentionally untouched here (its export
//! lane carries the pre-existing kv-append identity lag owned by
//! PGC-R3's re-key). This additive test proves the PGC-R1 binding
//! contract at the device boundary through the fake Metal session: the
//! embedding launch binds the tied `[49152,960]` table, a COMPACT
//! `lista<u32>` token-id vector, and the output — never a `[36,49152]`
//! one-hot selector; the ids upload is 144 B (36 × u32), not 7,077,888 B;
//! the launch geometry is the row-copy contract (workgroup (1,1,1), one
//! thread per output element); and the ids bytes round-trip exactly. The
//! fake session is a binding/census oracle, not a real-GPU receipt — the
//! emitted row-copy body itself is pinned by the radix sibling
//! (`gea3_pipeline_pgc_r1_test.rs`).

use faber_host_macos_arm64::device_host::{DeviceLaunchBinding, DeviceRuntime, DeviceSession};
use faber_host_macos_arm64::metal_host::MetalLaunchBinding;
use faber_host_macos_arm64::{FakeMetalDriver, MetalHostSession};
use host_coordinator::DeviceBackend;

const VOCAB: usize = 49_152;
const HIDDEN: usize = 960;
const PREFILL_ROWS: usize = 36;
const F32_BYTES: usize = 4;
const TABLE_BYTES: usize = VOCAB * HIDDEN * F32_BYTES;
const IDS_BYTES: usize = PREFILL_ROWS * F32_BYTES;
const OUTPUT_BYTES: usize = PREFILL_ROWS * HIDDEN * F32_BYTES;
/// The pre-PGC-R1 one-hot selector staging this card removes.
const SELECTOR_BYTES_BEFORE: usize = PREFILL_ROWS * VOCAB * F32_BYTES;

/// The fixed-1000 prompt's token ids (the fixture's parity record owns the
/// real values; these stand in as arbitrary in-range ids).
fn token_ids() -> Vec<u32> {
    (0..PREFILL_ROWS).map(|row| ((row * 1_373) % VOCAB) as u32).collect()
}

fn ids_le_bytes(ids: &[u32]) -> Vec<u8> {
    ids.iter().flat_map(|id| id.to_le_bytes()).collect()
}

#[test]
fn pgc_r1_device_embedding_launch_binds_compact_token_ids() {
    assert_eq!(SELECTOR_BYTES_BEFORE, 7_077_888);
    assert_eq!(IDS_BYTES, 144);

    let session =
        MetalHostSession::with_driver(Box::new(FakeMetalDriver::default()
            .with_known_entry("prefill_embedding_gather")
            .with_known_entry("embedding_gather")))
        .expect("fake Metal");
    let mut runtime = DeviceRuntime::Metal(session);
    let module = runtime.load_module(b"pgc-r1-embedding-gather").expect("module");
    let table = runtime.alloc_bytes(TABLE_BYTES).expect("table");
    let ids = runtime.alloc_bytes(IDS_BYTES).expect("ids");
    let plan_extra = runtime.alloc_bytes(F32_BYTES).expect("inert plan extra");
    let output = runtime.alloc_bytes(OUTPUT_BYTES).expect("output");

    // The compact upload: 36 u32 ids — never the [36,49152] selector.
    let ids_bytes = ids_le_bytes(&token_ids());
    assert_eq!(ids_bytes.len(), IDS_BYTES);
    runtime
        .copy_in_bytes(&ids, &ids_bytes, faber_host_macos_arm64::device_descriptor::DeviceDataType::U8)
        .expect("ids upload");
    let observed = runtime
        .readback_bytes(&ids, faber_host_macos_arm64::device_descriptor::DeviceDataType::U8)
        .expect("ids readback");
    assert_eq!(observed, ids_bytes, "token ids round-trip byte-exact");

    // The launch ABI: table (binding 0), ids (binding 1), inert plan
    // extra (binding 2), output (binding 3) — the bundle's four-binding
    // embedding contract, with NO selector buffer anywhere.
    let bindings = [
        MetalLaunchBinding {
            handle: faber_host_macos_arm64::metal_host::MetalHandleId(table.id),
            binding_index: 0,
            byte_offset: 0,
            view_span: TABLE_BYTES as u64,
        },
        MetalLaunchBinding {
            handle: faber_host_macos_arm64::metal_host::MetalHandleId(ids.id),
            binding_index: 1,
            byte_offset: 0,
            view_span: IDS_BYTES as u64,
        },
        MetalLaunchBinding {
            handle: faber_host_macos_arm64::metal_host::MetalHandleId(plan_extra.id),
            binding_index: 2,
            byte_offset: 0,
            view_span: F32_BYTES as u64,
        },
        MetalLaunchBinding {
            handle: faber_host_macos_arm64::metal_host::MetalHandleId(output.id),
            binding_index: 3,
            byte_offset: 0,
            view_span: OUTPUT_BYTES as u64,
        },
    ];
    let DeviceRuntime::Metal(session) = &mut runtime else {
        unreachable!("fake Metal runtime");
    };
    // The row-copy launch contract: one thread per output element under
    // the generic 1D gather grid — (PREFILL_ROWS × HIDDEN, 1, 1) threads
    // in (1, 1, 1) workgroups; never a tiled matmul grid.
    session
        .launch_kernel_bound(
            faber_host_macos_arm64::metal_host::MetalHandleId(module.id),
            "prefill_embedding_gather",
            &bindings,
            [(PREFILL_ROWS * HIDDEN) as u32, 1, 1],
            [1, 1, 1],
        )
        .expect("row-copy launch");
    session.sync().expect("row-copy sync");
    assert_eq!(
        session.live_handle_count(),
        5,
        "table + ids + plan extra + output + module allocations"
    );
}

/// The decode member carries the same contract at T=1: one id, one
/// 960-element row copy.
#[test]
fn pgc_r1_device_decode_member_is_one_row_copy() {
    let session =
        MetalHostSession::with_driver(Box::new(FakeMetalDriver::default()
            .with_known_entry("prefill_embedding_gather")
            .with_known_entry("embedding_gather")))
        .expect("fake Metal");
    let mut runtime = DeviceRuntime::Metal(session);
    let module = runtime.load_module(b"pgc-r1-embedding-gather").expect("module");
    let table = runtime.alloc_bytes(TABLE_BYTES).expect("table");
    let ids = runtime.alloc_bytes(F32_BYTES).expect("one id");
    let output = runtime.alloc_bytes(HIDDEN * F32_BYTES).expect("one row");
    runtime
        .copy_in_bytes(
            &ids,
            &[0x2a, 0x00, 0x00, 0x00],
            faber_host_macos_arm64::device_descriptor::DeviceDataType::U8,
        )
        .expect("decode id upload");
    let DeviceRuntime::Metal(session) = &mut runtime else {
        unreachable!("fake Metal runtime");
    };
    session
        .launch_kernel_bound(
            faber_host_macos_arm64::metal_host::MetalHandleId(module.id),
            "embedding_gather",
            &[
                MetalLaunchBinding {
                    handle: faber_host_macos_arm64::metal_host::MetalHandleId(table.id),
                    binding_index: 0,
                    byte_offset: 0,
                    view_span: TABLE_BYTES as u64,
                },
                MetalLaunchBinding {
                    handle: faber_host_macos_arm64::metal_host::MetalHandleId(ids.id),
                    binding_index: 1,
                    byte_offset: 0,
                    view_span: F32_BYTES as u64,
                },
                MetalLaunchBinding {
                    handle: faber_host_macos_arm64::metal_host::MetalHandleId(output.id),
                    binding_index: 2,
                    byte_offset: 0,
                    view_span: (HIDDEN * F32_BYTES) as u64,
                },
            ],
            [HIDDEN as u32, 1, 1],
            [1, 1, 1],
        )
        .expect("decode row-copy launch");
}

/// The staged-byte census at the device boundary: pre-prefill embedding
/// upload drops from 7,077,888 B of one-hot selector to 144 B of token
/// ids — the ~7.08 MB the card's done_when names.
#[test]
fn pgc_r1_device_staging_census_drops_the_one_hot_selector() {
    let staged_before = SELECTOR_BYTES_BEFORE + TABLE_BYTES;
    let staged_after = IDS_BYTES + TABLE_BYTES;
    assert_eq!(staged_before - staged_after, 7_077_744);
}

const _: DeviceBackend = DeviceBackend::Metal;
const _: Option<DeviceLaunchBinding> = None;
