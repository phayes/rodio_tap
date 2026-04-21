use crate::{FrameReaderConfig, TapReader};
use rtrb::Consumer;
use std::sync::Arc;
use std::time::Duration;

/// High-level reader for tapped PCM batches.
///
/// `FrameReader` wraps the low-level `TapReader` consumer loop and provides:
///
/// - automatic attach/switch when the active tap changes
/// - frame-aligned ring-buffer reads
/// - configurable batch sizing (time-based or frame-count-based)
/// - cooperative pacing to avoid busy-spin when data is incomplete
///
/// The callback receives interleaved `f32` PCM where:
///
/// - `batch.len() == frames_per_batch * channels`
/// - samples are laid out per frame (`[L, R, L, R, ...]` for stereo)
/// - every delivered batch is channel-aligned
///
/// This is the recommended API for most users. If you need direct ring-buffer control,
/// use the lower-level `TapReader`/`TapAdapter` pair.
///
/// This reader is synchronous and blocks the current thread. Run it in a dedicated thread.
/// For Tokio/async usage, enable the `async` feature and use `AsyncFrameReader`.
pub struct FrameReader {
    tap_fn: Box<dyn Fn() -> Option<Arc<TapReader>> + Send + Sync>,
    config: FrameReaderConfig,

    // live state
    active_consumer: Option<Consumer<f32>>,
    active_tap: Option<Arc<TapReader>>,
    ch: usize,
    sr: u32,

    // batch buffer (interleaved)
    batch_buf: Vec<f32>,
    batch_len_samples: usize, // = frames_per_batch * ch (recomputed per tap)
    filled: usize,            // samples written so far (multiple of ch)
}

impl FrameReader {
    /// Create a reader with [`FrameReaderConfig::default`].
    ///
    /// The returned reader will call `tap_fn` to discover the currently active tap.
    /// The closure should return the same `Arc<TapReader>` while a track is active, and
    /// a different one when playback switches.
    pub fn new<G>(tap_fn: G) -> Self
    where
        G: Fn() -> Option<Arc<TapReader>> + Send + Sync + 'static,
    {
        let config = FrameReaderConfig::default();
        Self::new_with_config(config, tap_fn)
    }

    /// Create a reader with an explicit [`FrameReaderConfig`].
    ///
    /// At least one of `frames_per_batch` or `ms_per_batch` must be set in the config.
    /// If both are set, `frames_per_batch` takes precedence.
    pub fn new_with_config<G>(config: FrameReaderConfig, tap_fn: G) -> Self
    where
        G: Fn() -> Option<Arc<TapReader>> + Send + Sync + 'static,
    {
        assert!(
            config.frames_per_batch.is_some() || config.ms_per_batch.is_some(),
            "FrameReaderConfig requires at least one batch target: frames_per_batch or ms_per_batch"
        );

        Self {
            tap_fn: Box::new(tap_fn),
            config,
            active_consumer: None,
            active_tap: None,
            ch: 0,
            sr: 0,
            batch_buf: Vec::new(),
            batch_len_samples: 0,
            filled: 0,
        }
    }

    /// Set the maximum number of samples requested from the ring per read attempt.
    ///
    /// Larger values can improve throughput, while smaller values can reduce per-iteration
    /// latency and callback burst size.
    pub fn with_max_per_read_samples(mut self, n: usize) -> Self {
        self.config.max_per_read_samples = n;
        self
    }

    /// Set sleep duration used when no active tap is available.
    pub fn with_no_tap_sleep(mut self, d: Duration) -> Self {
        self.config.no_tap_sleep = d;
        self
    }

    /// Set pacing bias used to predict sleep while a batch is partially filled.
    ///
    /// Expected range is `0.0 < bias <= 1.0`.
    pub fn with_sleep_bias(mut self, bias_0to1: f32) -> Self {
        self.config.sleep_bias = bias_0to1;
        self
    }

    /// Clamp the pacing sleep between `min` and `max`.
    pub fn with_sleep_clamp(mut self, min: Duration, max: Duration) -> Self {
        self.config.min_sleep = min;
        self.config.max_sleep = max;
        self
    }

    #[inline]
    fn recompute_batch_size(&mut self) {
        let frames = if let Some(frames) = self.config.frames_per_batch {
            frames as usize
        } else {
            let ms = self
                .config
                .ms_per_batch
                .expect("FrameReaderConfig must set ms_per_batch when frames_per_batch is not set");
            // frames = round(sr * ms / 1000)
            ((self.sr as u64 * ms as u64 + 500) / 1000) as usize
        }
        .max(1);
        let samples = frames * self.ch;
        self.batch_len_samples = samples.max(self.ch); // ensure ≥ one frame
        self.batch_buf.resize(self.batch_len_samples, 0.0);
        self.filled = 0;
    }

    /// Attach to current tap or switch if the current tap changed.
    fn try_attach_or_switch(&mut self, tap: Arc<TapReader>) -> bool {
        let tap_changed = match &self.active_tap {
            Some(active) => !Arc::ptr_eq(active, &tap),
            None => true,
        };

        if self.active_consumer.is_none() || tap_changed {
            if let Ok(mut slot) = tap.consumer.lock() {
                if let Some(cons) = slot.take() {
                    self.active_consumer = Some(cons);
                    self.active_tap = Some(Arc::clone(&tap));
                    self.ch = tap.channels as usize;
                    self.sr = tap.sample_rate_hz;
                    self.recompute_batch_size();

                    #[cfg(feature = "log")]
                    log::debug!(
                        "FrameReader attached tap ({} ch @ {} Hz, {} frames/batch)",
                        tap.channels,
                        tap.sample_rate_hz,
                        self.batch_len_samples / self.ch
                    );
                    return true;
                }
            }
            self.active_consumer = None;
            self.active_tap = None;
        }
        false
    }

    /// Copy `take` samples from `src` into the batch; invoke `on_batch` on each full batch.
    #[inline]
    fn ingest<F>(&mut self, mut src: &[f32], mut take: usize, on_batch: &mut F)
    where
        F: FnMut(&[f32], usize /*channels*/, u32 /*sr*/) + Send + 'static,
    {
        debug_assert_eq!(take % self.ch, 0);
        while take > 0 {
            let room = self.batch_len_samples - self.filled;
            let n = room.min(take);
            self.batch_buf[self.filled..self.filled + n].copy_from_slice(&src[..n]);
            self.filled += n;
            src = &src[n..];
            take -= n;

            if self.filled == self.batch_len_samples {
                on_batch(&self.batch_buf, self.ch, self.sr);
                self.filled = 0;
            }
        }
    }

    /// Predict a conservative sleep to let missing frames arrive.
    #[inline]
    fn sleep_for_missing(&self) -> Option<Duration> {
        if self.active_consumer.is_none() || self.sr == 0 || self.ch == 0 {
            return None;
        }
        let missing = self.batch_len_samples - self.filled; // samples
        if missing < self.ch {
            return None;
        } // less than one full frame missing → just yield
        let missing_frames = missing / self.ch; // integer frames
        // time = frames / sr seconds
        let nanos_f = (missing_frames as f64 / self.sr as f64)
            * 1_000_000_000.0
            * (self.config.sleep_bias as f64);
        let mut d = Duration::from_nanos(nanos_f as u64);
        if d < self.config.min_sleep {
            d = self.config.min_sleep;
        }
        if d > self.config.max_sleep {
            d = self.config.max_sleep;
        }
        Some(d)
    }

    /// Run forever and deliver full, channel-aligned batches to `on_batch`.
    ///
    /// The callback is invoked only for complete batches. Partial data is buffered
    /// internally until enough samples arrive.
    ///
    /// The chunks are interleaved. Use `batch.chunks_exact(ch)` to iterate per-frame.
    ///
    /// If `tap_fn` reports a different tap (for example, track switch), any partial batch
    /// in progress is dropped to preserve the exact batch-size contract.
    ///
    /// ```norun
    /// use std::sync::Arc;
    /// use arc_swap::ArcSwapOption;
    /// use rodio_tap::FrameReader;
    ///
    /// // Example: your app stores the current tap in ArcSwapOption.
    /// let current_tap = Arc::new(ArcSwapOption::empty());
    ///
    /// // Build reader that fetches the current tap each loop.
    /// let mut reader = FrameReader::new({
    ///     let current_tap = Arc::clone(&current_tap);
    ///     move || current_tap.load_full()
    /// });
    ///
    /// // Drive the reader and process channel-interleaved batches.
    /// reader.run(|batch, ch, sr| {
    ///     let frames = batch.len() / ch;
    ///     let mut sums = vec![0.0f32; ch];
    ///
    ///     for frame in batch.chunks_exact(ch) {
    ///         for (i, &s) in frame.iter().enumerate() {
    ///             sums[i] += s;
    ///         }
    ///     }
    ///
    ///     let avgs: Vec<f32> = sums.into_iter().map(|s| s / frames as f32).collect();
    ///     println!(
    ///         "[{} Hz] {} frames (~{} ms), avg/ch = {:?}",
    ///         sr,
    ///         frames,
    ///         1000 * frames as u64 / sr as u64,
    ///         avgs
    ///     );
    /// });
    /// ```
    pub fn run<F>(&mut self, mut on_batch: F) -> !
    where
        F: FnMut(&[f32], usize /*channels*/, u32 /*sr*/) + Send + 'static,
    {
        loop {
            if self.active_consumer.is_none() {
                let Some(tap) = (self.tap_fn)() else {
                    std::thread::sleep(self.config.no_tap_sleep);
                    continue;
                };
                if !self.try_attach_or_switch(tap) {
                    std::thread::sleep(self.config.no_tap_sleep);
                    continue;
                }
            }

            let mut made_progress = false;

            // Move the consumer out so we can borrow `self` mutably inside the loop.
            if self.active_consumer.is_some() {
                let mut cons = self.active_consumer.take().unwrap();

                loop {
                    let avail = cons.slots();
                    if avail == 0 {
                        break;
                    }

                    let want = avail.min(self.config.max_per_read_samples);
                    let Ok(chunk) = cons.read_chunk(want) else {
                        break;
                    };

                    let (a, b) = chunk.as_slices();
                    let total = a.len() + b.len();
                    if total == 0 {
                        break;
                    }

                    // Keep alignment: only commit multiples of channels.
                    let aligned = total - (total % self.ch);
                    if aligned == 0 {
                        break;
                    } // not enough for a full frame yet

                    // Portion inside A
                    let a_take = a.len().min(aligned);
                    let a_aligned = a_take - (a_take % self.ch);
                    if a_aligned > 0 {
                        self.ingest(&a[..a_aligned], a_aligned, &mut on_batch);
                    }

                    // Portion inside B
                    let b_take = aligned - a_aligned; // multiple of ch
                    if b_take > 0 {
                        self.ingest(&b[..b_take], b_take, &mut on_batch);
                    }

                    // Advance by an aligned amount only.
                    chunk.commit(aligned);
                    made_progress = true;

                    // If we left an unaligned tail (< ch), wait for more data.
                    if aligned < total {
                        break;
                    }

                    // If we can immediately produce more, keep looping (no sleep).
                    if self.filled == 0 {
                        continue; // delivered a full batch; try for another
                    } else {
                        break; // batch in progress; drop to pacing logic
                    }
                }

                // Put the consumer back.
                self.active_consumer = Some(cons);
            }

            // Track boundary: reattach & recompute batch size; drop partial batch.
            if let Some(tap) = (self.tap_fn)() {
                let tap_changed = match &self.active_tap {
                    Some(active) => !Arc::ptr_eq(active, &tap),
                    None => true,
                };
                if tap_changed {
                    #[cfg(feature = "log")]
                    log::trace!("FrameReader switching tap / tracks");

                    self.filled = 0; // drop partial to keep exact batch contract
                    let _ = self.try_attach_or_switch(tap);
                    continue;
                }
            }

            // Pacing: if a batch is in-progress and we didn't fill it this turn, sleep a bit.
            if self.filled > 0 && !made_progress {
                if let Some(d) = self.sleep_for_missing() {
                    std::thread::sleep(d);
                    continue;
                }
            }

            // Otherwise, be cooperative but eager.
            std::thread::yield_now();
        }
    }
}
