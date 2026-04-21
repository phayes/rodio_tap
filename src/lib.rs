mod tap;
use std::time::Duration;
pub use tap::*;

/// Configuration shared by [`FrameReader`] and [`AsyncFrameReader`].
///
/// `FrameReader` is the default synchronous reader.
/// `AsyncFrameReader` is available behind the `async` feature for Tokio/async runtimes.
///
/// You must specify at least one of `ms_per_batch` or `frames_per_batch`
pub struct FrameReaderConfig {
    /// target batch duration (e.g., Some(10) = ~10 ms)
    pub ms_per_batch: Option<u32>,
    /// preferred fixed frame count per batch
    pub frames_per_batch: Option<u32>,
    /// cap for read_chunk()
    pub max_per_read_samples: usize,
    /// sleep when there is no active tap
    pub no_tap_sleep: Duration,
    /// 0< bias <=1; actual sleep = bias * predicted_missing_time
    pub sleep_bias: f32,
    /// lower clamp for tiny sleeps
    pub min_sleep: Duration,
    /// upper clamp (safety)
    pub max_sleep: Duration,
}

impl Default for FrameReaderConfig {
    fn default() -> Self {
        Self {
            ms_per_batch: Some(10),
            frames_per_batch: None,
            max_per_read_samples: 64 * 1024,
            no_tap_sleep: Duration::from_secs(1),
            sleep_bias: 0.75,
            min_sleep: Duration::from_micros(150), // tiny but nonzero to be cooperative
            max_sleep: Duration::from_millis(5),
        }
    }
}

#[cfg(feature = "async")]
mod async_frame_reader;
#[cfg(feature = "async")]
pub use async_frame_reader::*;

pub mod frame_reader;
pub use frame_reader::*;