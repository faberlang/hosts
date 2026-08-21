use super::{encoder_gpu_timeline_from_samples, EncoderGpuTimeline};

#[test]
fn timeline_converts_ticks_relative_to_first_start() {
    // cpu 1 ms ↔ gpu 1_000_000 ticks, so 1000 ticks = 1 µs.
    let samples = [1000, 6000, 8000, 13_000];
    let timeline = encoder_gpu_timeline_from_samples(&samples, 2, 0, 1_000_000, 0, 1_000_000);
    assert_eq!(
        timeline,
        EncoderGpuTimeline {
            duration_us: vec![5, 5],
            start_us: vec![0, 7],
        }
    );
    let end0 = timeline.start_us[0].saturating_add(timeline.duration_us[0]);
    let gap = timeline.start_us[1].saturating_sub(end0);
    assert_eq!(gap, 2);
}

#[test]
fn timeline_rejects_empty_or_zero_span() {
    let samples = [0, 10];
    assert_eq!(
        encoder_gpu_timeline_from_samples(&samples, 0, 0, 1, 0, 1),
        EncoderGpuTimeline::default()
    );
    assert_eq!(
        encoder_gpu_timeline_from_samples(&samples, 1, 5, 5, 0, 1),
        EncoderGpuTimeline::default()
    );
    assert_eq!(
        encoder_gpu_timeline_from_samples(&samples, 1, 0, 1, 9, 9),
        EncoderGpuTimeline::default()
    );
}
