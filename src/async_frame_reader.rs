use crate::reader_core::ReaderCore;
use crate::{FrameReaderConfig, TapReader};
use arrayvec::ArrayVec;
use std::sync::Arc;
use std::time::Instant;

/// Tokio high-level reader for playback-paced tapped frame batches.
///
/// Its waiting implementation is independent from [`crate::FrameReader`]: this
/// type directly uses Tokio sleep and yield operations and never blocks a runtime
/// worker thread.
pub struct AsyncFrameReader<const C: usize = 2> {
    tap_fn: Box<dyn Fn() -> Option<Arc<TapReader<C>>> + Send + Sync>,
    core: ReaderCore<C>,
}

impl<const C: usize> AsyncFrameReader<C> {
    /// Create an async reader with [`FrameReaderConfig::default`].
    pub fn new<G>(tap_fn: G) -> Self
    where
        G: Fn() -> Option<Arc<TapReader<C>>> + Send + Sync + 'static,
    {
        Self::new_with_config(FrameReaderConfig::default(), tap_fn)
    }

    /// Create an async reader with explicit batching and pacing configuration.
    pub fn new_with_config<G>(config: FrameReaderConfig, tap_fn: G) -> Self
    where
        G: Fn() -> Option<Arc<TapReader<C>>> + Send + Sync + 'static,
    {
        assert!(C > 0, "AsyncFrameReader requires C > 0");
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

    /// Run forever, asynchronously delivering each batch on its playback deadline.
    pub async fn run<F>(&mut self, mut on_batch: F) -> !
    where
        F: FnMut(&[ArrayVec<f32, C>], usize, u32) + Send + 'static,
    {
        loop {
            if !self.core.has_consumer() {
                let Some(tap) = (self.tap_fn)() else {
                    tokio::time::sleep(self.core.config().no_tap_sleep).await;
                    continue;
                };
                if !self.core.try_attach_or_switch(tap) {
                    tokio::time::sleep(self.core.config().no_tap_sleep).await;
                    continue;
                }
            }

            if let Some(batch) = self.core.read_ready_batch() {
                let deadline = self.core.schedule(batch.duration, Instant::now());
                let wait = deadline.saturating_duration_since(Instant::now());
                if !wait.is_zero() {
                    tokio::time::sleep(wait).await;
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
                tokio::time::sleep(wait).await;
            } else {
                tokio::task::yield_now().await;
            }
        }
    }
}
