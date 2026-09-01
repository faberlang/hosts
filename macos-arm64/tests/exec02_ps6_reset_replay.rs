//! Focused EXEC02-PS6 reset/replay determinism proof.
//!
//! The SSM arm runs the landed conv1d and scan bodies over a fixed synthetic
//! layer fixture. The KV arm uses the landed transactional cursor and a small
//! K/V row overlay so that reset/replay compares live bytes while retaining
//! the physical rows that reset logically retires.

#[path = "../src/composite_host/inference_state.rs"]
mod inference_state;

use faber_host_macos_arm64::kernel::ssm_conv1d::{
    SsmConv1dBind, SsmConv1dKernel, dispatch_ssm_conv1d,
};
use faber_host_macos_arm64::kernel::ssm_scan::{SsmScanBind, SsmScanKernel, dispatch_ssm_scan};
use inference_state::{E_KV_STALE, InferenceSessionState, InvocationMode, SequencePhase};

const SSM_LENGTH: usize = 3;
const SSM_STATE_DIM: usize = 4;
const SSM_KERNEL_WIDTH: usize = 3;
const KV_CAPACITY: u32 = 8;

fn byte_identity(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn run_ssm_replay(state: &mut Vec<f32>) -> Vec<u8> {
    let input = [
        0.25f32, -0.5, 0.75, 1.0, // prefill t0
        0.5, 0.25, -0.25, 0.125, // prefill t1
        -0.75, 0.5, 0.125, -0.25, // prefill t2
    ];
    let kernel = [0.5f32, -0.25, 0.125];
    let conv_bind = SsmConv1dBind::channels_last(
        SSM_LENGTH as u64,
        SSM_STATE_DIM as u64,
        SSM_KERNEL_WIDTH as u64,
        [(SSM_LENGTH * SSM_STATE_DIM) as u32, 1, 1],
    );
    let mut convolved = vec![0.0; input.len()];
    dispatch_ssm_conv1d(
        SsmConv1dKernel::Causal,
        &conv_bind,
        &input,
        &kernel,
        &mut convolved,
    )
    .expect("PS6 SSM conv body");

    let scan_bind = SsmScanBind::prefill(
        SSM_LENGTH as u64,
        SSM_STATE_DIM as u64,
        [(SSM_LENGTH * SSM_STATE_DIM) as u32, 1, 1],
    );
    let mut prefill_state = vec![0.0; convolved.len()];
    dispatch_ssm_scan(
        SsmScanKernel::Additive,
        &scan_bind,
        &convolved,
        &mut prefill_state,
    )
    .expect("PS6 SSM prefill body");

    let decode_input = [0.125f32, -0.25, 0.5, 0.75];
    let decode_bind = SsmScanBind::decode(SSM_STATE_DIM as u64, [SSM_STATE_DIM as u32, 1, 1]);
    let mut decode_state = vec![0.0; decode_input.len()];
    dispatch_ssm_scan(
        SsmScanKernel::Additive,
        &decode_bind,
        &decode_input,
        &mut decode_state,
    )
    .expect("PS6 SSM decode body");

    state.clear();
    state.extend_from_slice(&prefill_state);
    state.extend_from_slice(&decode_state);
    byte_identity(state)
}

#[test]
fn exec02_ps6_ssm_reset_replay_is_byte_identical() {
    let mut state = vec![91.0f32; SSM_LENGTH * SSM_STATE_DIM + SSM_STATE_DIM];

    state.clear();
    let first_pass = run_ssm_replay(&mut state);
    assert!(
        !first_pass.is_empty(),
        "SSM replay must produce state bytes"
    );

    state.fill(17.0);
    state.clear();
    let second_pass = run_ssm_replay(&mut state);

    assert_eq!(first_pass, second_pass, "SSM reset/replay bytes must match");
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct KvRow {
    position: u32,
    token: u32,
}

#[derive(Debug)]
struct KvReplay {
    state: InferenceSessionState,
    k_rows: Vec<KvRow>,
    v_rows: Vec<KvRow>,
}

impl KvReplay {
    fn new() -> Self {
        Self {
            state: InferenceSessionState::new(KV_CAPACITY).expect("KV capacity"),
            k_rows: Vec::new(),
            v_rows: Vec::new(),
        }
    }

    fn write_row(rows: &mut Vec<KvRow>, row: KvRow) {
        if let Some(existing) = rows
            .iter_mut()
            .find(|existing| existing.position == row.position)
        {
            *existing = row;
        } else {
            rows.push(row);
        }
    }

    fn commit_tokens(&mut self, mode: InvocationMode, query_rows: u32, token_base: u32) {
        let transaction = self
            .state
            .begin_transaction(mode, query_rows)
            .expect("KV transaction admits");
        let state_before = self.state.clone();
        let k_before = self.k_rows.clone();
        let v_before = self.v_rows.clone();

        assert_eq!(
            self.state, state_before,
            "admission must not advance KV state"
        );
        assert_eq!(self.k_rows, k_before, "admission must not write K rows");
        assert_eq!(self.v_rows, v_before, "admission must not write V rows");

        let facts = transaction.plan().coordinates();
        self.state
            .commit_transaction(&transaction)
            .expect("KV transaction commits");
        for offset in 0..query_rows {
            let position = facts.write_position + offset;
            let token = token_base + offset;
            Self::write_row(&mut self.k_rows, KvRow { position, token });
            Self::write_row(
                &mut self.v_rows,
                KvRow {
                    position,
                    token: token + 10_000,
                },
            );
        }
        assert_eq!(
            self.state.valid_len(),
            facts.valid_len_after,
            "KV cursor advances only at commit"
        );
    }

    fn live_bytes(&self) -> Vec<u8> {
        let valid_len = self.state.valid_len();
        let mut bytes = Vec::new();
        for row in self
            .k_rows
            .iter()
            .chain(self.v_rows.iter())
            .filter(|row| row.position < valid_len)
        {
            bytes.extend_from_slice(&row.position.to_le_bytes());
            bytes.extend_from_slice(&row.token.to_le_bytes());
        }
        bytes
    }

    fn replay_pass(&mut self) -> Vec<u8> {
        self.commit_tokens(InvocationMode::Prefill, 3, 100);
        self.commit_tokens(InvocationMode::ScalarDecode, 1, 103);
        self.live_bytes()
    }
}

#[test]
fn exec02_ps6_kv_reset_replay_is_byte_identical_and_mid_step_stays_uncommitted() {
    let mut replay = KvReplay::new();
    let first_pass = replay.replay_pass();
    assert!(
        !first_pass.is_empty(),
        "KV replay must produce live K/V bytes"
    );

    let reset = replay.state.logical_reset().expect("KV reset");
    assert_eq!(reset.previous_valid_len, 4);
    assert_eq!(reset.valid_len, 0);
    assert_eq!(replay.state.phase(), SequencePhase::Fresh);
    assert_eq!(replay.state.valid_len(), 0);

    let second_pass = replay.replay_pass();
    assert_eq!(first_pass, second_pass, "KV reset/replay bytes must match");

    let mut mid_step = KvReplay::new();
    mid_step.commit_tokens(InvocationMode::Prefill, 2, 7);
    let transaction = mid_step
        .state
        .begin_transaction(InvocationMode::ScalarDecode, 1)
        .expect("mid-step decode admits");
    let before_reset = mid_step.state.clone();
    let reset = mid_step.state.logical_reset().expect("mid-step reset");
    assert_eq!(reset.previous_valid_len, 2);
    assert_eq!(mid_step.state.valid_len(), 0);
    assert_eq!(mid_step.state.phase(), SequencePhase::Fresh);

    let stale = mid_step
        .state
        .commit_transaction(&transaction)
        .expect_err("pre-reset token must not commit after reset");
    assert_eq!(stale.code, E_KV_STALE);
    assert_eq!(mid_step.state.valid_len(), 0);
    assert_eq!(mid_step.state.phase(), SequencePhase::Fresh);
    assert_ne!(mid_step.state, before_reset, "reset must advance the epoch");
}
