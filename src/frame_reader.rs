use rtrb::Consumer;
use std::time::Duration;
use crate::TapReader;
use std::sync::Arc;

/// Reads interleaved PCM in time-based batches.
/// On each callback, you get a slice of length = frames_per_batch * channels (for the current tap).
pub struct FrameReader {
    // config
    ms_per_batch: u32,           // target batch duration (e.g., 10 = ~10 ms)
    max_per_read_samples: usize, // cap for read_chunk()
    no_tap_sleep: Duration,      // sleep when there is no active tap
    sleep_bias: f32,             // 0< bias <=1; actual sleep = bias * predicted_missing_time
    min_sleep: Duration,         // lower clamp for tiny sleeps
    max_sleep: Duration,         // upper clamp (safety)

    // live state
    active_consumer: Option<Consumer<f32>>,
    active_generation: u32,
    ch: usize,
    sr: u32,
    replay_gain: Option<f64>,
    peak_amplitude: Option<f64>,

    // batch buffer (interleaved)
    batch_buf: Vec<f32>,
    batch_len_samples: usize, // = frames_per_batch * ch (recomputed per tap)
    filled: usize,            // samples written so far (multiple of ch)
}

impl FrameReader {
    pub fn new(ms_per_batch: u32) -> Self {
        Self {
            ms_per_batch,
            max_per_read_samples: 64 * 1024,
            no_tap_sleep: Duration::from_secs(1),
            sleep_bias: 0.75,
            min_sleep: Duration::from_micros(150), // tiny but nonzero to be cooperative
            max_sleep: Duration::from_millis(5),
            active_consumer: None,
            active_generation: 0,
            ch: 0,
            sr: 0,
            replay_gain: None,
            peak_amplitude: None,
            batch_buf: Vec::new(),
            batch_len_samples: 0,
            filled: 0,
        }
    }

    pub fn with_max_per_read_samples(mut self, n: usize) -> Self {
        self.max_per_read_samples = n;
        self
    }

    pub fn with_no_tap_sleep(mut self, d: Duration) -> Self {
        self.no_tap_sleep = d;
        self
    }

    pub fn with_sleep_bias(mut self, bias_0to1: f32) -> Self {
        self.sleep_bias = bias_0to1;
        self
    }

    pub fn with_sleep_clamp(mut self, min: Duration, max: Duration) -> Self {
        self.min_sleep = min;
        self.max_sleep = max;
        self
    }

    #[inline]
    fn recompute_batch_size(&mut self) {
        // frames = round(sr * ms / 1000), clamp to at least 1
        let frames = ((self.sr as u64 * self.ms_per_batch as u64 + 500) / 1000).max(1) as usize;
        let samples = frames * self.ch;
        self.batch_len_samples = samples.max(self.ch); // ensure ≥ one frame
        self.batch_buf.resize(self.batch_len_samples, 0.0);
        self.filled = 0;
    }

    /// Attach to current tap or switch if generation changed.
    fn try_attach_or_switch(&mut self, tap: Arc<TapReader>) -> bool {
        if self.active_consumer.is_none() || tap.generation != self.active_generation {
            if let Ok(mut slot) = tap.consumer.lock() {
                if let Some(cons) = slot.take() {
                    self.active_consumer = Some(cons);
                    self.active_generation = tap.generation;
                    self.ch = tap.channels as usize;
                    self.sr = tap.sample_rate_hz;
                    self.replay_gain = tap.replay_gain;
                    self.peak_amplitude = tap.peak_amplitude;
                    self.recompute_batch_size();
                    log::debug!(
                        "FrameReader attached gen {} ({} ch @ {} Hz, ~{} ms/batch ≈ {} frames)",
                        tap.generation,
                        tap.channels,
                        tap.sample_rate_hz,
                        self.ms_per_batch,
                        self.batch_len_samples / self.ch
                    );
                    return true;
                }
            }
            self.active_consumer = None;
        }
        false
    }

    /// Copy `take` samples from `src` into the batch; invoke `on_batch` on each full batch.
    #[inline]
    fn ingest<F>(&mut self, mut src: &[f32], mut take: usize, on_batch: &mut F)
    where
        F: FnMut(
                &[f32],
                usize,       /*channels*/
                u32,         /*sr*/
                u32,         /*generation*/
                Option<f64>, /*Replay Gain*/
                Option<f64>, /*Peak Amplitude*/
            ) + Send
            + 'static,
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
                on_batch(
                    &self.batch_buf,
                    self.ch,
                    self.sr,
                    self.active_generation,
                    self.replay_gain,
                    self.peak_amplitude,
                );
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
        let nanos_f =
            (missing_frames as f64 / self.sr as f64) * 1_000_000_000.0 * (self.sleep_bias as f64);
        let mut d = Duration::from_nanos(nanos_f as u64);
        if d < self.min_sleep {
            d = self.min_sleep;
        }
        if d > self.max_sleep {
            d = self.max_sleep;
        }
        Some(d)
    }

    /// Run forever, delivering batches close to `ms_per_batch`.
    ///
    /// The chunks will be interleaved, so you can use the channels parameter to map the samples to the channels using the Iter::chunks_exact(channels) method
    ///
    /// ```norun
    ///  // Aim for ~33 ms batches regardless of sr/ch
    ///  let mut pump = PcmFramePump::new(33);
    ///
    /// pump.run(|batch, ch, sr, gen| {
    ///    let frames = batch.len() / ch;
    ///    // example: per-channel averages across this time slice
    ///    let mut sums = vec![0.0f32; ch];
    ///    for frame in batch.chunks_exact(ch) {
    ///        for (i, &s) in frame.iter().enumerate() { sums[i] += s; }
    ///    }
    ///    let avgs: Vec<f32> = sums.into_iter().map(|s| s / frames as f32).collect();
    ///    println!("[gen {} @ {}Hz] ~{} ms ({} frames) | avg/ch = {:?}",
    ///             gen, sr, 1000 * frames as u64 / sr as u64, frames, avgs);
    /// }).await;
    pub async fn run<F>(&mut self, mut on_batch: F) -> !
    where
        F: FnMut(
                &[f32],
                usize,       /*channels*/
                u32,         /*sr*/
                u32,         /*gen*/
                Option<f64>, /*replay_gain*/
                Option<f64>, /*peak_amplitude*/
            ) + Send
            + 'static,
    {
        loop {
            if self.active_consumer.is_none() && !self.try_attach_or_switch() {
                tokio::time::sleep(self.no_tap_sleep).await;
                continue;
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

                    let want = avail.min(self.max_per_read_samples);
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
            if let Some(tap) = PLAYER.current_track_tap() {
                if tap.generation != self.active_generation {
                    self.filled = 0; // drop partial to keep exact batch contract
                    let _ = self.try_attach_or_switch();
                    continue;
                }
            }

            // Pacing: if a batch is in-progress and we didn't fill it this turn, sleep a bit.
            if self.filled > 0 && !made_progress {
                if let Some(d) = self.sleep_for_missing() {
                    tokio::time::sleep(d).await;
                    continue;
                }
            }

            // Otherwise, be cooperative but eager.
            tokio::task::yield_now().await;
        }
    }
}
