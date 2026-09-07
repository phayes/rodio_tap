use arrayvec::ArrayVec;
use std::time::Duration;

pub(crate) struct ReadyBatch<const C: usize> {
    pub(crate) frames: Vec<ArrayVec<f32, C>>,
    pub(crate) channels: usize,
    pub(crate) sample_rate_hz: u32,
    pub(crate) duration: Duration,
}

pub(crate) fn duration_to_frames(duration: Duration, sample_rate_hz: u32) -> usize {
    ((sample_rate_hz as u128 * duration.as_nanos() + 500_000_000) / 1_000_000_000)
        .max(1)
        .try_into()
        .unwrap_or(usize::MAX)
}

pub(crate) fn frames_to_duration(frames: usize, sample_rate_hz: u32) -> Duration {
    if sample_rate_hz == 0 {
        return Duration::ZERO;
    }

    let nanos = (frames as u128 * 1_000_000_000) / sample_rate_hz as u128;
    Duration::from_nanos(nanos.min(u64::MAX as u128) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_duration_and_frames() {
        assert_eq!(duration_to_frames(Duration::from_millis(10), 48_000), 480);
        assert_eq!(frames_to_duration(480, 48_000), Duration::from_millis(10));
    }
}
