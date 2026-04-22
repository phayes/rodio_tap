#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]

mod tap;
use std::time::Duration;
pub use tap::*;

/// Configuration shared by [`FrameReader`] and [`AsyncFrameReader`].
///
/// `FrameReader` is the default synchronous reader.
/// `AsyncFrameReader` is available behind the `async` feature for Tokio/async runtimes.
///
/// You must specify at least one of `time_per_batch` or `frames_per_batch`
pub struct FrameReaderConfig {
    /// Target batch duration.
    ///
    /// Default: `Some(Duration::from_millis(10))`.
    pub time_per_batch: Option<Duration>,
    /// Preferred fixed frame count per batch.
    ///
    /// If set, this takes precedence over `time_per_batch`.
    /// Default: `None`.
    pub frames_per_batch: Option<u32>,
    /// Sleep duration when there is no active tap.
    ///
    /// Default: `Duration::from_millis(100)`.
    pub no_tap_sleep: Duration,
    /// Pacing bias in the range `0.0 < sleep_bias <= 1.0`.
    ///
    /// Used when a batch is partially filled to predict how long to sleep before polling
    /// again: `actual_sleep = sleep_bias * predicted_missing_time`.
    ///
    /// Tuning guidance:
    /// - Lower values (for example `0.2..0.6`) wake up earlier and poll more often.
    ///   Prefer this for low-latency / real-time-ish processing where callback jitter
    ///   matters more than CPU efficiency.
    /// - Higher values (for example `0.7..1.0`) sleep closer to the full predicted
    ///   time. Prefer this for latency-tolerant workloads where fewer wakeups and
    ///   better CPU efficiency are more important.
    ///
    /// The final sleep is still clamped by `min_sleep` and `max_sleep`.
    ///
    /// Default: `0.75`.
    pub sleep_bias: f32,
    /// Lower clamp for tiny sleeps.
    ///
    /// Default: `Duration::from_micros(150)`.
    pub min_sleep: Duration,
    /// Upper clamp for pacing sleeps.
    ///
    /// Default: `Duration::from_millis(100)`.
    pub max_sleep: Duration,
}

impl Default for FrameReaderConfig {
    fn default() -> Self {
        Self {
            time_per_batch: Some(Duration::from_millis(10)),
            frames_per_batch: None,
            no_tap_sleep: Duration::from_millis(100),
            sleep_bias: 0.75,
            min_sleep: Duration::from_micros(150), // tiny but nonzero to be cooperative
            max_sleep: Duration::from_millis(100),
        }
    }
}

#[cfg(feature = "async")]
mod async_frame_reader;

#[cfg(feature = "async")]
pub use async_frame_reader::*;

#[cfg(feature = "visualizer")]
mod visualizer;

#[cfg(feature = "visualizer")]
pub use visualizer::*;

#[cfg(all(feature = "visualizer", feature = "async"))]
mod async_visualizer;

#[cfg(all(feature = "visualizer", feature = "async"))]
pub use async_visualizer::*;

mod frame_reader;
pub use frame_reader::*;
