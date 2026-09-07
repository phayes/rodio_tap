use crate::batch::{ReadyBatch, duration_to_frames, frames_to_duration};
use crate::{FrameFormat, FrameReaderConfig, TapPacket, TapReader};
use arrayvec::ArrayVec;
use rtrb::Consumer;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub(crate) struct ReaderCore<const C: usize> {
    config: FrameReaderConfig,
    active_consumer: Option<Consumer<TapPacket<C>>>,
    active_tap: Option<Arc<TapReader<C>>>,
    channels: usize,
    sample_rate_hz: u32,
    has_format: bool,
    batch_buf: Vec<ArrayVec<f32, C>>,
    batch_len_frames: usize,
    next_deadline: Option<Instant>,
}

impl<const C: usize> ReaderCore<C> {
    pub(crate) fn new(config: FrameReaderConfig) -> Self {
        Self {
            config,
            active_consumer: None,
            active_tap: None,
            channels: 0,
            sample_rate_hz: 0,
            has_format: false,
            batch_buf: Vec::new(),
            batch_len_frames: 1,
            next_deadline: None,
        }
    }

    pub(crate) fn config(&self) -> &FrameReaderConfig {
        &self.config
    }

    pub(crate) fn has_consumer(&self) -> bool {
        self.active_consumer.is_some()
    }

    pub(crate) fn try_attach_or_switch(&mut self, tap: Arc<TapReader<C>>) -> bool {
        let tap_changed = self
            .active_tap
            .as_ref()
            .is_none_or(|active| !Arc::ptr_eq(active, &tap));

        if self.active_consumer.is_none() || tap_changed {
            if let Ok(mut slot) = tap.consumer.lock()
                && let Some(consumer) = slot.take()
            {
                self.active_consumer = Some(consumer);
                self.active_tap = Some(Arc::clone(&tap));
                self.reset_stream();
                return true;
            }
            self.active_consumer = None;
            self.active_tap = None;
        }
        false
    }

    pub(crate) fn switch_if_changed(&mut self, tap: Arc<TapReader<C>>) -> bool {
        let changed = self
            .active_tap
            .as_ref()
            .is_none_or(|active| !Arc::ptr_eq(active, &tap));
        if changed {
            self.batch_buf.clear();
            self.has_format = false;
            let _ = self.try_attach_or_switch(tap);
        }
        changed
    }

    fn reset_stream(&mut self) {
        self.channels = 0;
        self.sample_rate_hz = 0;
        self.has_format = false;
        self.batch_buf.clear();
        self.batch_len_frames = 1;
        self.next_deadline = None;
    }

    fn recompute_batch_size(&mut self) {
        self.batch_len_frames = if let Some(frames) = self.config.frames_per_batch {
            frames as usize
        } else if self.sample_rate_hz == 0 {
            1
        } else {
            duration_to_frames(
                self.config
                    .time_per_batch
                    .expect("time_per_batch is required without frames_per_batch"),
                self.sample_rate_hz,
            )
        }
        .max(1);
        self.batch_buf
            .reserve(self.batch_len_frames.saturating_sub(self.batch_buf.len()));
    }

    fn take_batch(&mut self) -> ReadyBatch<C> {
        let frames = std::mem::take(&mut self.batch_buf);
        ReadyBatch {
            duration: frames_to_duration(frames.len(), self.sample_rate_hz),
            frames,
            channels: self.channels,
            sample_rate_hz: self.sample_rate_hz,
        }
    }

    pub(crate) fn recycle(&mut self, mut frames: Vec<ArrayVec<f32, C>>) {
        frames.clear();
        if self.batch_buf.is_empty() && frames.capacity() >= self.batch_len_frames {
            self.batch_buf = frames;
        }
    }

    fn handle_format(&mut self, format: FrameFormat) -> Option<ReadyBatch<C>> {
        let new_channels = format.channels as usize;
        if new_channels == 0 || new_channels > C || format.sample_rate_hz == 0 {
            return None;
        }
        if self.has_format
            && self.channels == new_channels
            && self.sample_rate_hz == format.sample_rate_hz
        {
            return None;
        }

        let ready = (self.has_format && !self.batch_buf.is_empty()).then(|| self.take_batch());
        self.channels = new_channels;
        self.sample_rate_hz = format.sample_rate_hz;
        self.has_format = true;
        self.recompute_batch_size();
        ready
    }

    fn handle_frame(&mut self, frame: &ArrayVec<f32, C>) -> Option<ReadyBatch<C>> {
        if !self.has_format || frame.len() != self.channels {
            return None;
        }
        self.batch_buf.push(frame.clone());
        (self.batch_buf.len() == self.batch_len_frames).then(|| self.take_batch())
    }

    /// Consume at most enough packets to produce one callback batch.
    pub(crate) fn read_ready_batch(&mut self) -> Option<ReadyBatch<C>> {
        let mut consumer = self.active_consumer.take()?;
        let mut ready = None;

        'chunks: loop {
            let available = consumer.slots();
            if available == 0 {
                break;
            }
            let missing = self
                .batch_len_frames
                .saturating_sub(self.batch_buf.len())
                .max(1);
            let want = available.min(missing.saturating_add(1));
            let Ok(chunk) = consumer.read_chunk(want) else {
                break;
            };
            let (first, second) = chunk.as_slices();
            let mut processed = 0;
            for packet in first.iter().chain(second.iter()) {
                processed += 1;
                ready = match packet {
                    TapPacket::Format(format) => self.handle_format(*format),
                    TapPacket::Frame(frame) => self.handle_frame(frame),
                };
                if ready.is_some() {
                    break;
                }
            }
            chunk.commit(processed);
            if ready.is_some() {
                break 'chunks;
            }
        }

        self.active_consumer = Some(consumer);
        ready
    }

    /// Place each batch at the end of its represented audio interval.
    ///
    /// If work falls behind, rebasing to `now` prevents a run of past deadlines from
    /// turning into a callback burst.
    pub(crate) fn schedule(&mut self, duration: Duration, now: Instant) -> Instant {
        let candidate = self
            .next_deadline
            .and_then(|deadline| deadline.checked_add(duration))
            .or_else(|| now.checked_add(duration))
            .unwrap_or(now);
        let deadline = candidate.max(now);
        self.next_deadline = Some(deadline);
        deadline
    }

    pub(crate) fn sleep_for_missing(&self) -> Option<Duration> {
        if self.active_consumer.is_none()
            || self.sample_rate_hz == 0
            || self.channels == 0
            || self.batch_buf.is_empty()
            || self.batch_buf.len() >= self.batch_len_frames
        {
            return None;
        }
        let missing = self.batch_len_frames - self.batch_buf.len();
        let predicted =
            frames_to_duration(missing, self.sample_rate_hz).mul_f64(self.config.sleep_bias as f64);
        Some(predicted.clamp(self.config.min_sleep, self.config.max_sleep))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> FrameReaderConfig {
        FrameReaderConfig {
            time_per_batch: None,
            frames_per_batch: Some(2),
            no_tap_sleep: Duration::from_millis(1),
            sleep_bias: 0.75,
            min_sleep: Duration::from_micros(100),
            max_sleep: Duration::from_millis(5),
        }
    }

    #[test]
    fn timeline_delays_first_batch_and_rebases_late_work() {
        let mut core = ReaderCore::<2>::new(config());
        let start = Instant::now();
        let period = Duration::from_millis(10);
        assert_eq!(core.schedule(period, start), start + period);
        assert_eq!(core.schedule(period, start + period), start + period * 2);
        let late = start + Duration::from_millis(50);
        assert_eq!(core.schedule(period, late), late);
        assert_eq!(core.schedule(period, late), late + period);
    }

    #[test]
    fn format_change_returns_partial_old_format_batch() {
        let mut core = ReaderCore::<2>::new(config());
        core.handle_format(FrameFormat {
            channels: 2,
            sample_rate_hz: 48_000,
        });
        let mut frame = ArrayVec::new();
        frame.extend([1.0, 2.0]);
        assert!(core.handle_frame(&frame).is_none());

        let ready = core
            .handle_format(FrameFormat {
                channels: 1,
                sample_rate_hz: 44_100,
            })
            .unwrap();
        assert_eq!(ready.frames.len(), 1);
        assert_eq!(ready.channels, 2);
        assert_eq!(ready.sample_rate_hz, 48_000);
    }

    #[test]
    fn ring_backlog_is_returned_one_batch_at_a_time() {
        let source = rodio::buffer::SamplesBuffer::new(
            std::num::NonZeroU16::new(2).unwrap(),
            std::num::NonZeroU32::new(48_000).unwrap(),
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        let (tap, adapter) = TapReader::<2>::new(source);
        let forwarded: Vec<_> = adapter.collect();
        assert_eq!(forwarded.len(), 8);

        let mut core = ReaderCore::new(config());
        assert!(core.try_attach_or_switch(tap));
        let first = core.read_ready_batch().unwrap();
        assert_eq!(first.frames.len(), 2);
        let second = core.read_ready_batch().unwrap();
        assert_eq!(second.frames.len(), 2);
        assert!(core.read_ready_batch().is_none());
    }
}
