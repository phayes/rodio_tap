use arc_swap::ArcSwapOption;
use arrayvec::ArrayVec;
use cpal::Sample;
use rodio::source::SeekError;
use rtrb::chunks::ChunkError;
use rtrb::{Consumer, Producer, RingBuffer};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Duration;
use rodio::SampleRate;

/// Taps PCM from an inner Source into a lock-free ring, preserving frame alignment.
pub struct TapAdapter<S: rodio::Source> {
    inner: S,
    prod: Producer<f32>,
    /// Staging buffer (only holds whole frames; flushed to ring in aligned chunks)
    batch: ArrayVec<f32, 1024>, // ~4 KiB at f32; tune as you like
    /// Number of interleaved channels
    num_channels: usize,
    /// Publish-on-first-sample hook (sets Player.current_tap once audio starts)
    on_start: Option<OnFirstSample>,
    /// Collect exactly one frame before touching the ring/batch
    frame_buf: ArrayVec<f32, 8>, // supports up to 8 ch (7.1). Increase if you need more.
}

impl<S: rodio::Source> TapAdapter<S> {
    pub fn new(
        inner: S,
        prod: Producer<f32>,
        num_channels: u16,
        on_start: Option<OnFirstSample>,
    ) -> Self {
        let ch = num_channels as usize;
        debug_assert!(
            ch <= 8,
            "Tap::frame_buf capacity (8) < channels ({}) — increase ArrayVec capacity",
            ch
        );
        Self {
            inner,
            prod,
            batch: ArrayVec::new(),
            num_channels: ch,
            on_start,
            frame_buf: ArrayVec::new(),
        }
    }

    /// Flush staged samples to the ring in **multiples of num_channels**.
    /// Attempts partial flush if the ring is tight.
    #[inline]
    fn flush(&mut self) {
        if self.batch.len() < self.num_channels {
            return; // not even one full frame staged
        }

        // Try to write as much as we have; if ring is tight, fall back to a smaller aligned chunk.
        let mut try_len = self.batch.len();

        loop {
            match self.prod.write_chunk(try_len) {
                Ok(mut wchunk) => {
                    // We got `try_len` slots; copy & commit all of them.
                    let (a, b) = wchunk.as_mut_slices();
                    let n_a = a.len();
                    if n_a > 0 {
                        a.copy_from_slice(&self.batch[..n_a]);
                    }
                    let n_b = b.len();
                    if n_b > 0 {
                        b.copy_from_slice(&self.batch[n_a..n_a + n_b]);
                    }
                    wchunk.commit(try_len);
                    // drop exactly what we committed; keep any tail (< num_channels)
                    self.batch.drain(..try_len);
                    break;
                }
                Err(ChunkError::TooFewSlots(avail)) => {
                    // Align down to whole frames; if still zero, bail out.
                    let aligned = (avail / self.num_channels) * self.num_channels;
                    if aligned == 0 {
                        return;
                    }
                    try_len = aligned;
                    // loop and retry with the smaller aligned size
                }
            }
        }
    }

    /// Handle exactly one whole frame (length == num_channels) in a way that never breaks alignment:
    /// - Prefer staging it into `batch`
    /// - If batch has no room, try flushing then staging
    /// - If still tight, try writing the single frame directly to the ring
    /// - Otherwise drop the **whole frame** (never a single sample)
    #[inline]
    fn push_frame_aligned(&mut self, frame: &[f32]) {
        debug_assert_eq!(frame.len(), self.num_channels);

        // Try to stage into batch; if room < one frame, flush first.
        if self.batch.remaining_capacity() < self.num_channels {
            self.flush();
        }
        if self.batch.remaining_capacity() >= self.num_channels {
            // Safe append of a whole frame
            let _ = self.batch.try_extend_from_slice(frame);
            return;
        }

        // No batch space (ring still tight). Try direct write of a single frame.
        match self.prod.write_chunk(self.num_channels) {
            Ok(mut wchunk) => {
                let (a, b) = wchunk.as_mut_slices();
                if a.len() >= self.num_channels {
                    a[..self.num_channels].copy_from_slice(frame);
                } else {
                    let n_a = a.len();
                    a[..n_a].copy_from_slice(&frame[..n_a]);
                    b[..(self.num_channels - n_a)].copy_from_slice(&frame[n_a..]);
                }
                wchunk.commit(self.num_channels);
                return;
            }
            Err(ChunkError::TooFewSlots(_)) => {
                #[cfg(all(debug_assertions, feature = "log"))]
                log::debug!("TapAdapter: No space in ring buffer, dropped the frame.");
            }
        }
    }
}

impl<S: rodio::Source> Iterator for TapAdapter<S>
where
    S::Item: cpal::Sample + Send + 'static,
    f32: cpal::FromSample<S::Item>,
{
    type Item = S::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let s = self.inner.next()?; // rodio mixer pulls one sample

        // Publish-on-first-sample for queued tracks (if configured)
        if let Some(cb) = &self.on_start {
            cb.maybe_publish();
        }

        // Collect exactly one frame before touching the ring/batch.
        let f: f32 = f32::from_sample(s);
        if self.frame_buf.try_push(f).is_err() {
            // Shouldn't happen with capacity >= channels; reset defensively.
            self.frame_buf.clear();
            let _ = self.frame_buf.try_push(f);
        }

        if self.frame_buf.len() == self.num_channels {
            // Move the full frame out, leaving an empty ArrayVec in self
            let frame = std::mem::take(&mut self.frame_buf); // requires ArrayVec: Default (it has it)
            debug_assert_eq!(frame.len(), self.num_channels);
            self.push_frame_aligned(&frame[..]);
            // self.frame_buf is now empty and ready for the next frame
        }

        Some(s)
    }
}

impl<S: rodio::Source> rodio::Source for TapAdapter<S>
where
    S::Item: cpal::Sample + Send + 'static,
    f32: cpal::FromSample<S::Item>,
{
    #[inline]
    fn current_span_len(&self) -> Option<usize> {
        self.inner.current_span_len()
    }
    #[inline]
    fn channels(&self) -> std::num::NonZero<u16> {
        self.inner.channels()
    }
    #[inline]
    fn sample_rate(&self) -> SampleRate {
        self.inner.sample_rate()
    }

    #[inline]
    fn total_duration(&self) -> Option<Duration> {
        self.inner.total_duration()
    }

    // Forward seeks and clear staged state so no pre-seek samples leak out.
    #[inline]
    fn try_seek(&mut self, pos: Duration) -> Result<(), SeekError> {
        self.inner.try_seek(pos)?;
        self.frame_buf.clear();
        self.batch.clear();
        Ok(())
    }
}

impl<S: rodio::Source> Drop for TapAdapter<S> {
    fn drop(&mut self) {
        // Discard any partial frame quietly (keeps alignment guarantee).
        self.frame_buf.clear();
        // Final aligned flush so the consumer never sees a broken frame tail.
        self.flush();
    }
}

pub struct TapReader {
    pub consumer: Mutex<Option<Consumer<f32>>>, // taken by consumer thread once
    pub sample_rate_hz: u32,
    pub channels: u16,
}

impl TapReader {
    /// Build a TapReader + TapAdapter pair.
    pub fn new<S>(
        decoder: S,
    ) -> (Arc<TapReader>, TapAdapter<S>)
    where
        S: rodio::Source + Send + 'static,
        S::Item: cpal::Sample + Send + 'static,
        f32: cpal::FromSample<S::Item>,
    {
        Self::new_with_publish_target_inner(None, decoder)
    }

    /// Build a TapReader + TapAdapter pair and wire publish-on-first-sample.
    ///
    /// - `publish_target` is the ArcSwap where the reader will be published on the first decoded sample.
    pub fn new_with_publish_target<S>(
        publish_target: &Arc<ArcSwapOption<TapReader>>,
        decoder: S,
    ) -> (Arc<TapReader>, TapAdapter<S>)
    where
        S: rodio::Source + Send + 'static,
        S::Item: cpal::Sample + Send + 'static,
        f32: cpal::FromSample<S::Item>,
    {
        Self::new_with_publish_target_inner(Some(publish_target), decoder)
    }

    fn new_with_publish_target_inner<S>(
        publish_target: Option<&Arc<ArcSwapOption<TapReader>>>,
        decoder: S,
    ) -> (Arc<TapReader>, TapAdapter<S>)
    where
        S: rodio::Source + Send + 'static,
        S::Item: cpal::Sample + Send + 'static,
        f32: cpal::FromSample<S::Item>,
    {
        let sr = decoder.sample_rate();
        let ch = decoder.channels();

        // We could get into trouble with 7.1 surround at 192kHz (which needs 204Kb for 30 reads-per-second - so no room for error at all)
        // TODO: Make the dynamic based on the sample rate and number of channels
        let cap = 65_536;
        let (prod, cons) = RingBuffer::<f32>::new(cap);

        // Reader side (for consumers)
        let reader = Arc::new(TapReader {
            consumer: Mutex::new(Some(cons)),
            sample_rate_hz: sr.into(),
            channels: ch.into(),
        });

        let on_start = publish_target.map(|target| OnFirstSample {
            fired: AtomicBool::new(false),
            target: Arc::clone(target),
            tap: Arc::clone(&reader),
        });

        // Writer side (for the mixer)
        let adapter = TapAdapter::new(decoder, prod, ch.into(), on_start);

        (reader, adapter)
    }
}

pub struct OnFirstSample {
    fired: AtomicBool,
    target: Arc<ArcSwapOption<TapReader>>,
    tap: Arc<TapReader>,
}

impl OnFirstSample {
    #[inline]
    fn maybe_publish(&self) {
        if self
            .fired
            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            self.target.store(Some(self.tap.clone()));

            #[cfg(feature = "log")]
            log::trace!("OnFirstSample published tap");
        }
    }

    pub fn is_fired(&self, order: Ordering) -> bool {
        self.fired.load(order)
    }

    pub fn get_tap(&self) -> Arc<TapReader> {
        self.tap.clone()
    }

    pub fn get_target(&self) -> Arc<ArcSwapOption<TapReader>> {
        self.target.clone()
    }
}
