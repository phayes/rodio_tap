use crate::reader_core::ReaderCore;
use crate::{FrameReaderConfig, TapReader};
use arrayvec::ArrayVec;
use std::sync::Arc;
use std::time::Instant;

/// Synchronous high-level reader for playback-paced tapped frame batches.
///
/// The reader consumes only one callback batch at a time and delays its delivery
/// until the end of the audio interval represented by that batch. This prevents
/// buffered audio from producing several visualization callbacks at once.
///
/// This method blocks its thread. For Tokio, enable `async` and use
/// [`crate::AsyncFrameReader`].
pub struct FrameReader<const C: usize = 2> {
    tap_fn: Box<dyn Fn() -> Option<Arc<TapReader<C>>> + Send + Sync>,
    core: ReaderCore<C>,
}

impl<const C: usize> FrameReader<C> {
    /// Create a reader with [`FrameReaderConfig::default`].
    pub fn new<G>(tap_fn: G) -> Self
    where
        G: Fn() -> Option<Arc<TapReader<C>>> + Send + Sync + 'static,
    {
        Self::new_with_config(FrameReaderConfig::default(), tap_fn)
    }

    /// Create a reader with explicit batching and pacing configuration.
    pub fn new_with_config<G>(config: FrameReaderConfig, tap_fn: G) -> Self
    where
        G: Fn() -> Option<Arc<TapReader<C>>> + Send + Sync + 'static,
    {
        assert!(C > 0, "FrameReader requires C > 0");
        assert!(
            config.frames_per_batch.is_some() || config.time_per_batch.is_some(),
            "FrameReaderConfig requires frames_per_batch or time_per_batch"
        );
        assert!(
            (0.0..=1.0).contains(&config.sleep_bias),
            "FrameReaderConfig.sleep_bias must be between 0 and 1"
        );
        assert!(
            config.min_sleep <= config.max_sleep,
            "FrameReaderConfig.min_sleep must not exceed max_sleep"
        );
        Self {
            tap_fn: Box::new(tap_fn),
            core: ReaderCore::new(config),
        }
    }

    /// Run forever, delivering each batch on its playback-time deadline.
    pub fn run<F>(&mut self, mut on_batch: F) -> !
    where
        F: FnMut(&[ArrayVec<f32, C>], usize, u32) + Send + 'static,
    {
        loop {
            if !self.core.has_consumer() {
                let Some(tap) = (self.tap_fn)() else {
                    std::thread::sleep(self.core.config().no_tap_sleep);
                    continue;
                };
                if !self.core.try_attach_or_switch(tap) {
                    std::thread::sleep(self.core.config().no_tap_sleep);
                    continue;
                }
            }

            if let Some(batch) = self.core.read_ready_batch() {
                let deadline = self.core.schedule(batch.duration, Instant::now());
                let wait = deadline.saturating_duration_since(Instant::now());
                if !wait.is_zero() {
                    std::thread::sleep(wait);
                }
                on_batch(&batch.frames, batch.channels, batch.sample_rate_hz);
                self.core.recycle(batch.frames);
                continue;
            }

            if let Some(tap) = (self.tap_fn)()
                && self.core.switch_if_changed(tap)
            {
                continue;
            }

            if let Some(wait) = self.core.sleep_for_missing() {
                std::thread::sleep(wait);
            } else {
                std::thread::yield_now();
            }
        }
    }
}
