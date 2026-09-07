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
///
/// A completed batch is delivered at the end of the audio duration it
/// represents. Readers consume at most one ready batch per deadline, so a
/// producer-side buffer or accumulated ring backlog does not cause callback
/// bursts. If callback work misses a deadline, pacing rebases instead of
/// emitting multiple catch-up callbacks.
///
/// Real-time tuning (suggested starting point for very low-latency use cases):
/// - `frames_per_batch: Some(64)` (equivalent to 128 sample buffer size in stereo)
/// - `time_per_batch: None` (use fixed frame batches)
/// - `sleep_bias: 0.5` (wake earlier to avoid late batch delivery)
/// - `min_sleep: Duration::from_micros(5)` (tiny cooperative sleep)
#[derive(Debug, Clone)]
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
    /// Pacing bias in the range `0.0 <= sleep_bias <= 1.0`.
    ///
    /// Used when a batch is partially filled to predict how long to sleep before polling
    /// again: `actual_sleep = sleep_bias * predicted_missing_time`.
    /// A value of `0.0` means the predicted sleep becomes zero and the reader will
    /// always use `min_sleep` after clamping.
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
    /// Whether to discard queued callback batches when processing falls behind.
    ///
    /// When enabled, the reader skips one queued batch for each complete batch
    /// interval of lateness. It does not drain future prebuffered audio. This is
    /// useful for live meters and visualizers that should recover toward the
    /// current playback position.
    /// Keep it disabled for recording or analysis that must preserve every frame.
    ///
    /// Default: `false`.
    pub drop_late_batches: bool,
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
            drop_late_batches: false,
        }
    }
}

/// A playback-paced frame batch delivered by [`FrameReader`] or [`AsyncFrameReader`].
#[derive(Debug)]
pub struct FrameBatch<'a, const C: usize> {
    /// Audio frames in this callback batch.
    pub frames: &'a [arrayvec::ArrayVec<f32, C>],
    /// Active channel count represented by each frame.
    pub channels: usize,
    /// Active stream sample rate.
    pub sample_rate_hz: u32,
    /// Number of older callback batches discarded immediately before this batch.
    ///
    /// This is always zero when [`FrameReaderConfig::drop_late_batches`] is disabled.
    pub dropped_batches: usize,
    /// Audio duration represented by the discarded batches.
    pub dropped_duration: Duration,
}

#[cfg(feature = "async")]
mod async_frame_reader;

#[cfg(feature = "async")]
pub use async_frame_reader::*;

#[cfg(feature = "visualizer")]
mod visualizer;

#[cfg(feature = "visualizer")]
pub use visualizer::*;

mod batch;
mod frame_reader;
mod reader_core;
pub use frame_reader::*;
