use crate::{FrameReaderConfig, FrameFormat, TapPacket, TapReader};
use arrayvec::ArrayVec;
use rtrb::Consumer;
use std::sync::Arc;
use std::time::Duration;

/// High-level reader for tapped frame batches.
///
/// `FrameReader` wraps the low-level `TapReader` consumer loop and provides:
///
/// - automatic attach/switch when the active tap changes
/// - packet-based ring-buffer reads (`TapPacket::Format` / `TapPacket::Frame`)
/// - configurable batch sizing (time-based or frame-count-based)
/// - cooperative pacing to avoid busy-spin when data is incomplete
///
/// The callback receives frame batches where each frame is an interleaved
/// `ArrayVec<f32, C>`:
///
/// - usually `batch.len() == frames_per_batch`
/// - each frame length is `channels` for that callback
/// - on in-band format change, a partial batch may be emitted before switching format
///
/// This is the recommended API for most users. If you need direct ring-buffer control,
/// use the lower-level `TapReader`/`TapAdapter` pair.
///
/// This reader is synchronous and blocks the current thread. Run it in a dedicated thread.
/// For Tokio/async usage, enable the `async` feature and use `AsyncFrameReader`.
pub struct FrameReader<const C: usize = 2> {
    tap_fn: Box<dyn Fn() -> Option<Arc<TapReader<C>>> + Send + Sync>,
    config: FrameReaderConfig,

    // live state
    active_consumer: Option<Consumer<TapPacket<C>>>,
    active_tap: Option<Arc<TapReader<C>>>,
    ch: usize,
    sr: u32,
    has_format: bool,

    // batch buffer (interleaved frames)
    batch_buf: Vec<ArrayVec<f32, C>>,
    batch_len_frames: usize,
}

impl<const C: usize> FrameReader<C> {
    /// Create a reader with [`FrameReaderConfig::default`].
    ///
    /// The returned reader will call `tap_fn` to discover the currently active tap.
    /// The closure should return the same `Arc<TapReader>` while a track is active, and
    /// a different one when playback switches.
    pub fn new<G>(tap_fn: G) -> Self
    where
        G: Fn() -> Option<Arc<TapReader<C>>> + Send + Sync + 'static,
    {
        let config = FrameReaderConfig::default();
        Self::new_with_config(config, tap_fn)
    }

    /// Create a reader with an explicit [`FrameReaderConfig`].
    ///
    /// At least one of `frames_per_batch` or `time_per_batch` must be set in the config.
    /// If both are set, `frames_per_batch` takes precedence.
    pub fn new_with_config<G>(config: FrameReaderConfig, tap_fn: G) -> Self
    where
        G: Fn() -> Option<Arc<TapReader<C>>> + Send + Sync + 'static,
    {
        assert!(C > 0, "FrameReader requires C > 0");
        assert!(
            config.frames_per_batch.is_some() || config.time_per_batch.is_some(),
            "FrameReaderConfig requires at least one batch target: frames_per_batch or time_per_batch"
        );

        Self {
            tap_fn: Box::new(tap_fn),
            config,
            active_consumer: None,
            active_tap: None,
            ch: 0,
            sr: 0,
            has_format: false,
            batch_buf: Vec::new(),
            batch_len_frames: 0,
        }
    }

    #[inline]
    fn recompute_batch_size(&mut self) {
        self.batch_len_frames = if let Some(frames) = self.config.frames_per_batch {
            frames as usize
        } else {
            if self.sr == 0 {
                // We may attach before seeing the first Format packet.
                1
            } else {
            let batch_duration = self
                .config
                .time_per_batch
                .expect("FrameReaderConfig must set time_per_batch when frames_per_batch is not set");
            // frames = round(sr * duration_secs)
            ((self.sr as u128 * batch_duration.as_nanos() + 500_000_000) / 1_000_000_000) as usize
            }
        }
        .max(1);
        self.batch_buf.clear();
        self.batch_buf.reserve(self.batch_len_frames);
    }

    /// Attach to current tap or switch if the current tap changed.
    fn try_attach_or_switch(&mut self, tap: Arc<TapReader<C>>) -> bool {
        let tap_changed = match &self.active_tap {
            Some(active) => !Arc::ptr_eq(active, &tap),
            None => true,
        };

        if self.active_consumer.is_none() || tap_changed {
            if let Ok(mut slot) = tap.consumer.lock() {
                if let Some(cons) = slot.take() {
                    self.active_consumer = Some(cons);
                    self.active_tap = Some(Arc::clone(&tap));
                    self.ch = 0;
                    self.sr = 0;
                    self.has_format = false;
                    self.recompute_batch_size();

                    #[cfg(feature = "log")]
                    log::debug!("FrameReader attached tap (awaiting first Format packet)");
                    return true;
                }
            }
            self.active_consumer = None;
            self.active_tap = None;
        }
        false
    }

    #[inline]
    fn emit_batch<F>(&mut self, on_batch: &mut F)
    where
        F: FnMut(&[ArrayVec<f32, C>], usize /*channels*/, u32 /*sr*/),
    {
        if self.batch_buf.is_empty() {
            return;
        }
        on_batch(&self.batch_buf, self.ch, self.sr);
        self.batch_buf.clear();
    }

    #[inline]
    fn handle_format<F>(&mut self, fmt: FrameFormat, on_batch: &mut F)
    where
        F: FnMut(&[ArrayVec<f32, C>], usize /*channels*/, u32 /*sr*/),
    {
        let new_ch = fmt.channels as usize;
        if new_ch == 0 || new_ch > C {
            #[cfg(feature = "log")]
            log::debug!(
                "FrameReader ignoring invalid format packet ({} ch @ {} Hz)",
                fmt.channels,
                fmt.sample_rate_hz
            );
            return;
        }

        let format_changed = !self.has_format || self.ch != new_ch || self.sr != fmt.sample_rate_hz;
        if !format_changed {
            return;
        }

        if self.has_format && !self.batch_buf.is_empty() {
            // On in-band format changes, emit the partial batch in the old format.
            self.emit_batch(on_batch);
        }

        self.ch = new_ch;
        self.sr = fmt.sample_rate_hz;
        self.has_format = true;
        self.recompute_batch_size();
    }

    #[inline]
    fn handle_frame<F>(&mut self, frame: &ArrayVec<f32, C>, on_batch: &mut F)
    where
        F: FnMut(&[ArrayVec<f32, C>], usize /*channels*/, u32 /*sr*/),
    {
        // Late attach can observe frames before the first format packet.
        if !self.has_format {
            #[cfg(feature = "log")]
            log::debug!("FrameReader dropping frame because it has not received a format packet yet");
            return;
        }

        if frame.len() != self.ch {
            #[cfg(feature = "log")]
            log::debug!(
                "FrameReader dropping frame with len {} (expected {})",
                frame.len(),
                self.ch
            );
            return;
        }

        self.batch_buf.push(frame.clone());
        if self.batch_buf.len() == self.batch_len_frames {
            self.emit_batch(on_batch);
        }
    }

    /// Predict a conservative sleep to let missing frames arrive.
    #[inline]
    fn sleep_for_missing(&self) -> Option<Duration> {
        if self.active_consumer.is_none() || self.sr == 0 || self.ch == 0 {
            return None;
        }
        if self.batch_buf.len() >= self.batch_len_frames {
            return None;
        }
        let missing_frames = self.batch_len_frames - self.batch_buf.len();
        if missing_frames == 0 {
            return None;
        }
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

    /// Run forever and deliver frame batches to `on_batch`.
    ///
    /// The callback is usually invoked for complete batches. If an in-band format change
    /// arrives while a batch is in progress, the partial batch is emitted before the
    /// new format starts.
    ///
    /// Each item in `batch` is one interleaved frame (`ArrayVec<f32, C>`) with length `ch`.
    ///
    /// If `tap_fn` reports a different tap (for example, track switch), any partial batch
    /// in progress is dropped.
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use arrayvec::ArrayVec;
    /// use arc_swap::ArcSwapOption;
    /// use rodio_tap::{FrameReader, TapReader};
    ///
    /// // Example: your app stores the current tap in ArcSwapOption.
    /// let current_tap = Arc::new(ArcSwapOption::<TapReader<2>>::empty());
    ///
    /// // Build reader that fetches the current tap each loop.
    /// let mut reader = FrameReader::<2>::new({
    ///     let current_tap = Arc::clone(&current_tap);
    ///     move || current_tap.load_full()
    /// });
    ///
    /// // Drive the reader and process frame batches.
    /// reader.run(|batch, ch, sr| {
    ///     let frames = batch.len();
    ///     let mut sums = vec![0.0f32; ch];
    ///
    ///     for frame in batch {
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
        F: FnMut(&[ArrayVec<f32, C>], usize /*channels*/, u32 /*sr*/) + Send + 'static,
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

                    // Prefer reading roughly one callback worth of packets at a time.
                    // Add one packet of slack to accommodate an in-band `Format` packet.
                    let missing_frames = self.batch_len_frames.saturating_sub(self.batch_buf.len());
                    let target_packets = missing_frames.max(1).saturating_add(1);
                    let want = avail.min(target_packets);
                    let Ok(chunk) = cons.read_chunk(want) else {
                        break;
                    };

                    let (a, b) = chunk.as_slices();
                    let total = a.len() + b.len();
                    if total == 0 {
                        break;
                    }

                    for packet in a.iter().chain(b.iter()) {
                        match packet {
                            TapPacket::Format(fmt) => self.handle_format(*fmt, &mut on_batch),
                            TapPacket::Frame(frame) => self.handle_frame(frame, &mut on_batch),
                        }
                        made_progress = true;
                    }

                    // Commit all packets we processed from this chunk.
                    chunk.commit(total);
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

                    self.batch_buf.clear(); // drop partial across taps
                    self.has_format = false;
                    let _ = self.try_attach_or_switch(tap);
                    continue;
                }
            }

            // Pacing: if a batch is in-progress and we didn't fill it this turn, sleep a bit.
            if !self.batch_buf.is_empty() && !made_progress {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn frame2(a: f32, b: f32) -> ArrayVec<f32, 2> {
        let mut f = ArrayVec::<f32, 2>::new();
        f.push(a);
        f.push(b);
        f
    }

    fn test_reader() -> FrameReader<2> {
        let cfg = FrameReaderConfig {
            frames_per_batch: Some(2),
            time_per_batch: None,
            no_tap_sleep: Duration::from_millis(1),
            sleep_bias: 0.75,
            min_sleep: Duration::from_micros(100),
            max_sleep: Duration::from_millis(5),
        };
        FrameReader::new_with_config(cfg, || None)
    }

    #[test]
    fn ignores_frame_before_first_format() {
        let mut reader = test_reader();
        let mut called = 0usize;
        reader.handle_frame(&frame2(1.0, 2.0), &mut |_, _, _| called += 1);
        assert_eq!(called, 0);
        assert!(reader.batch_buf.is_empty());
    }

    #[test]
    fn emits_full_batch_after_format() {
        let mut reader = test_reader();
        let mut observed = Vec::new();

        reader.handle_format(
            FrameFormat {
                channels: 2,
                sample_rate_hz: 48_000,
            },
            &mut |batch, ch, sr| observed.push((batch.to_vec(), ch, sr)),
        );
        reader.handle_frame(&frame2(1.0, 2.0), &mut |batch, ch, sr| {
            observed.push((batch.to_vec(), ch, sr))
        });
        reader.handle_frame(&frame2(3.0, 4.0), &mut |batch, ch, sr| {
            observed.push((batch.to_vec(), ch, sr))
        });

        assert_eq!(observed.len(), 1);
        assert_eq!(observed[0].1, 2);
        assert_eq!(observed[0].2, 48_000);
        assert_eq!(observed[0].0.len(), 2);
    }

    #[test]
    fn emits_partial_batch_on_format_change() {
        let mut reader = test_reader();
        let mut observed = Vec::new();

        reader.handle_format(
            FrameFormat {
                channels: 2,
                sample_rate_hz: 48_000,
            },
            &mut |batch, ch, sr| observed.push((batch.to_vec(), ch, sr)),
        );
        reader.handle_frame(&frame2(1.0, 2.0), &mut |batch, ch, sr| {
            observed.push((batch.to_vec(), ch, sr))
        });

        reader.handle_format(
            FrameFormat {
                channels: 1,
                sample_rate_hz: 44_100,
            },
            &mut |batch, ch, sr| observed.push((batch.to_vec(), ch, sr)),
        );

        assert_eq!(observed.len(), 1);
        assert_eq!(observed[0].1, 2);
        assert_eq!(observed[0].2, 48_000);
        assert_eq!(observed[0].0.len(), 1);
        assert_eq!(reader.ch, 1);
        assert_eq!(reader.sr, 44_100);
    }

    #[test]
    fn drops_frame_on_channel_mismatch() {
        let mut reader = test_reader();
        reader.handle_format(
            FrameFormat {
                channels: 2,
                sample_rate_hz: 48_000,
            },
            &mut |_, _, _| {},
        );

        let mut bad = ArrayVec::<f32, 2>::new();
        bad.push(1.0);
        reader.handle_frame(&bad, &mut |_, _, _| {});
        assert!(reader.batch_buf.is_empty());
    }
}
