use arc_swap::ArcSwapOption;
use arrayvec::ArrayVec;
use cpal::Sample;
use rodio::source::SeekError;
use rodio::SampleRate;
use rtrb::{Consumer, Producer, RingBuffer};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

/// Runtime audio format metadata for tapped packets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameFormat {
    pub channels: u8,
    pub sample_rate_hz: u32,
}

/// Packet emitted by the low-level tap ring. Generic C over the maximum number of channels supported.
///
/// Invariants:
/// - The first packet for a stream is always `Format`.
/// - Whenever format changes, a new `Format` packet is emitted before the first `Frame`.
/// - `Frame.len()` is always `<= C`, and equals the active format channels for this tap.
#[derive(Debug, Clone, PartialEq)]
pub enum TapPacket<const C: usize = 2> {
    /// Format packet announces the current `{channels, sample_rate_hz}`.
    Format(FrameFormat),
    /// Frame packet carries one interleaved frame (up to `C` channels).
    /// For example, the frame is interleaved `[L, R, L, R, ...]` for stereo.
    Frame(ArrayVec<f32, C>),
}

impl<const C: usize> Default for TapPacket<C> {
    fn default() -> Self {
        if C == 0 {
            panic!("TapPacket requires generic C > 0");
        }
        let channels = C.max(1) as u8;
        TapPacket::Format(FrameFormat {
            channels,
            sample_rate_hz: 48_000,
        })
    }
}

/// Adapts a `rodio::Source` and taps packets into a lock-free ring buffer. 
///
/// This is the write side of the low-level tap API. It forwards all samples to the
/// playback pipeline while mirroring packets into an `rtrb` ring:
/// - `TapPacket::Format` announces the active format.
/// - `TapPacket::Frame` carries one interleaved frame (up to `C` channels).
///
/// TapAdapter is generic over C: the maximum number of channels supported. 
/// If the source reports more channels than `C`, extra channels are dropped.
pub struct TapAdapter<S: rodio::Source, const C: usize = 2> {
    inner: S,
    prod: Producer<TapPacket<C>>,
    /// Publish-on-first-sample hook (sets Player.current_tap once audio starts)
    on_start: Option<OnFirstSample<C>>,
    /// Build one output frame before packet emission.
    frame_buf: ArrayVec<f32, C>,
    /// Source frame position in current source channel layout.
    src_frame_pos: usize,
    /// Active source channel count.
    active_src_channels: usize,
    /// Active output channel count (`min(source_channels, C)`).
    active_out_channels: usize,
    /// Active source sample-rate.
    active_sample_rate_hz: u32,
    /// Remaining samples in current span.
    span_remaining: Option<usize>,
    /// True until a `Format` packet is emitted for current params.
    format_dirty: bool,
}

impl<S: rodio::Source, const C: usize> TapAdapter<S, C> {
    pub fn new(inner: S, prod: Producer<TapPacket<C>>, on_start: Option<OnFirstSample<C>>) -> Self {
        assert!(C > 0, "TapAdapter requires generic C > 0");

        let src_channels = inner.channels().get() as usize;
        let out_channels = src_channels.min(C).max(1);
        let sample_rate_hz: u32 = inner.sample_rate().into();
        let span_remaining = inner.current_span_len();

        Self {
            inner,
            prod,
            on_start,
            frame_buf: ArrayVec::new(),
            src_frame_pos: 0,
            active_src_channels: src_channels,
            active_out_channels: out_channels,
            active_sample_rate_hz: sample_rate_hz,
            span_remaining,
            format_dirty: true,
        }
    }

    #[inline]
    fn current_output_format(&self) -> FrameFormat {
        FrameFormat {
            channels: self.active_out_channels as u8,
            sample_rate_hz: self.active_sample_rate_hz,
        }
    }

    #[inline]
    fn push_packet(&mut self, packet: TapPacket<C>) -> bool {
        if self.prod.push(packet).is_err() {
            #[cfg(all(debug_assertions, feature = "log"))]
            log::debug!("TapAdapter: No space in ring buffer, dropped packet.");
            false
        } else {
            true
        }
    }

    #[inline]
    fn emit_format_if_needed(&mut self) -> bool {
        if !self.format_dirty {
            return true;
        }
        if self.push_packet(TapPacket::Format(self.current_output_format())) {
            self.format_dirty = false;
            true
        } else {
            false
        }
    }

    #[inline]
    fn emit_frame_packet(&mut self) {
        if self.active_out_channels == 0 {
            self.frame_buf.clear();
            return;
        }

        if self.frame_buf.len() != self.active_out_channels {
            self.frame_buf.clear();
            return;
        }

        // Never emit a frame before its matching format packet is accepted.
        let format_ok = self.emit_format_if_needed();
        if !format_ok {
            // Drop this frame and retry format on the next frame boundary.
            self.frame_buf.clear();
            return;
        }

        let frame = std::mem::take(&mut self.frame_buf);
        debug_assert_eq!(frame.len(), self.active_out_channels);
        let _ = self.push_packet(TapPacket::Frame(frame));
    }

    #[inline]
    fn note_span_sample(&mut self) {
        if let Some(rem) = self.span_remaining.as_mut() {
            if *rem > 0 {
                *rem -= 1;
            }
            if *rem == 0 {
                self.handle_span_boundary();
            }
        }
    }

    #[inline]
    fn handle_span_boundary(&mut self) {
        // Discard partial frame to avoid any cross-boundary mixing.
        self.frame_buf.clear();
        self.src_frame_pos = 0;

        let new_src_channels = self.inner.channels().get() as usize;
        let new_out_channels = new_src_channels.min(C).max(1);
        let new_sample_rate_hz: u32 = self.inner.sample_rate().into();

        if new_out_channels != self.active_out_channels || new_sample_rate_hz != self.active_sample_rate_hz {
            self.format_dirty = true;
        }

        self.active_src_channels = new_src_channels;
        self.active_out_channels = new_out_channels;
        self.active_sample_rate_hz = new_sample_rate_hz;
        self.span_remaining = self.inner.current_span_len();
    }
}

impl<S: rodio::Source, const C: usize> Iterator for TapAdapter<S, C>
where
    S::Item: cpal::Sample + Send + 'static,
    f32: cpal::FromSample<S::Item>,
{
    type Item = S::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // `Some(0)` means current span is exhausted, so refresh format/span before
        // associating the next emitted sample with metadata.
        if self.span_remaining == Some(0) {
            self.handle_span_boundary();
        }

        let s = self.inner.next()?;

        if let Some(cb) = &self.on_start {
            cb.maybe_publish();
        }

        let f: f32 = f32::from_sample(s);
        if self.src_frame_pos < self.active_out_channels {
            let _ = self.frame_buf.try_push(f);
        }
        self.src_frame_pos += 1;

        if self.src_frame_pos == self.active_src_channels {
            self.src_frame_pos = 0;
            self.emit_frame_packet();
        }

        self.note_span_sample();
        Some(s)
    }
}

impl<S: rodio::Source, const C: usize> rodio::Source for TapAdapter<S, C>
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

    #[inline]
    fn try_seek(&mut self, pos: Duration) -> Result<(), SeekError> {
        self.inner.try_seek(pos)?;
        self.frame_buf.clear();
        self.src_frame_pos = 0;
        self.active_src_channels = self.inner.channels().get() as usize;
        self.active_out_channels = self.active_src_channels.min(C).max(1);
        self.active_sample_rate_hz = self.inner.sample_rate().into();
        self.span_remaining = self.inner.current_span_len();
        self.format_dirty = true;
        Ok(())
    }
}

impl<S: rodio::Source, const C: usize> Drop for TapAdapter<S, C> {
    fn drop(&mut self) {
        self.frame_buf.clear();
    }
}

/// Read side for packets produced by [`TapAdapter`].
///
/// # Low-level API warning
///
/// `TapReader`/`TapAdapter` are intentionally low-level building blocks:
///
/// - You are responsible for manually taking and polling the ring-buffer consumer.
/// - You must process `TapPacket::Format` boundaries before consuming `TapPacket::Frame`.
/// - You should decide your own pacing strategy to avoid busy waiting.
///
/// If you want a higher-level API with batching and built-in pacing, use
/// [`crate::FrameReader`] instead. For Tokio/async runtimes, enable the `async`
/// feature and use [`crate::AsyncFrameReader`].
///
/// # Example
///
/// ```no_run
/// use rodio_tap::{TapPacket, TapReader};
///
/// fn run<S>(decoder: S)
/// where
///     S: rodio::Source + Send + 'static,
///     S::Item: cpal::Sample + Send + 'static,
///     f32: cpal::FromSample<S::Item>,
/// {
/// let (tap_reader, adapter) = TapReader::<2>::new(decoder);
/// let _ = adapter;
///
/// let mut consumer = tap_reader
///     .consumer
///     .lock()
///     .expect("tap consumer lock poisoned")
///     .take()
///     .expect("tap consumer already taken");
///
/// let mut current_format = None;
///
/// loop {
///     let avail = consumer.slots();
///     if avail == 0 {
///         std::thread::sleep(std::time::Duration::from_millis(1));
///         continue;
///     }
///
///     let chunk = consumer.read_chunk(avail).expect("read_chunk failed");
///     let (a_len, b_len) = {
///         let (a, b) = chunk.as_slices();
///
///         for packet in a.iter().chain(b.iter()) {
///             match packet {
///                 TapPacket::<2>::Format(fmt) => current_format = Some(*fmt),
///                 TapPacket::<2>::Frame(frame) => {
///                     let _ = (frame, current_format);
///                     // Process one interleaved frame for the active format.
///                 }
///             }
///         }
///         (a.len(), b.len())
///     };
///
///     chunk.commit(a_len + b_len);
/// }
/// # }
/// ```
pub struct TapReader<const C: usize = 2> {
    /// Single-use ring-buffer consumer.
    ///
    /// This is wrapped in `Mutex<Option<_>>` so one thread/task can take ownership once
    /// and then poll it directly.
    pub consumer: Mutex<Option<Consumer<TapPacket<C>>>>,
}

impl<const C: usize> TapReader<C> {
    /// Build a `TapReader` + `TapAdapter` pair.
    pub fn new<S>(decoder: S) -> (Arc<TapReader<C>>, TapAdapter<S, C>)
    where
        S: rodio::Source + Send + 'static,
        S::Item: cpal::Sample + Send + 'static,
        f32: cpal::FromSample<S::Item>,
    {
        Self::new_with_publish_target_inner(None, decoder)
    }

    /// Build a TapReader + TapAdapter pair and wire publish-on-first-sample.
    pub fn new_with_publish_target<S>(
        publish_target: &Arc<ArcSwapOption<TapReader<C>>>,
        decoder: S,
    ) -> (Arc<TapReader<C>>, TapAdapter<S, C>)
    where
        S: rodio::Source + Send + 'static,
        S::Item: cpal::Sample + Send + 'static,
        f32: cpal::FromSample<S::Item>,
    {
        Self::new_with_publish_target_inner(Some(publish_target), decoder)
    }

    fn new_with_publish_target_inner<S>(
        publish_target: Option<&Arc<ArcSwapOption<TapReader<C>>>>,
        decoder: S,
    ) -> (Arc<TapReader<C>>, TapAdapter<S, C>)
    where
        S: rodio::Source + Send + 'static,
        S::Item: cpal::Sample + Send + 'static,
        f32: cpal::FromSample<S::Item>,
    {
        assert!(C > 0, "TapReader requires C > 0");

        let cap = 65_536;
        let (prod, cons) = RingBuffer::<TapPacket<C>>::new(cap);

        let reader = Arc::new(TapReader {
            consumer: Mutex::new(Some(cons)),
        });

        let on_start = publish_target.map(|target| OnFirstSample {
            fired: AtomicBool::new(false),
            target: Arc::clone(target),
            tap: Arc::clone(&reader),
        });

        let adapter = TapAdapter::new(decoder, prod, on_start);
        (reader, adapter)
    }
}

pub struct OnFirstSample<const C: usize = 2> {
    fired: AtomicBool,
    target: Arc<ArcSwapOption<TapReader<C>>>,
    tap: Arc<TapReader<C>>,
}

impl<const C: usize> OnFirstSample<C> {
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

    pub fn get_tap(&self) -> Arc<TapReader<C>> {
        self.tap.clone()
    }

    pub fn get_target(&self) -> Arc<ArcSwapOption<TapReader<C>>> {
        self.target.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::num::NonZero;
    use rodio::Source;

    #[derive(Clone)]
    struct SpanChunk {
        channels: u16,
        sample_rate_hz: u32,
        samples: Vec<f32>,
    }

    struct MockSpanSource {
        chunks: VecDeque<SpanChunk>,
        current: Option<SpanChunk>,
        idx: usize,
    }

    impl MockSpanSource {
        fn new(chunks: Vec<SpanChunk>) -> Self {
            let mut q = VecDeque::from(chunks);
            let current = q.pop_front();
            Self {
                chunks: q,
                current,
                idx: 0,
            }
        }
    }

    impl Iterator for MockSpanSource {
        type Item = f32;

        fn next(&mut self) -> Option<Self::Item> {
            loop {
                let cur = self.current.as_ref()?;
                if self.idx < cur.samples.len() {
                    let s = cur.samples[self.idx];
                    self.idx += 1;
                    if self.idx == cur.samples.len() {
                        self.current = self.chunks.pop_front();
                        self.idx = 0;
                    }
                    return Some(s);
                }
                self.current = self.chunks.pop_front();
                self.idx = 0;
            }
        }
    }

    impl rodio::Source for MockSpanSource {
        fn current_span_len(&self) -> Option<usize> {
            self.current.as_ref().map(|c| c.samples.len())
        }

        fn channels(&self) -> NonZero<u16> {
            let c = self.current.as_ref().map(|c| c.channels).unwrap_or(2);
            NonZero::new(c).unwrap()
        }

        fn sample_rate(&self) -> SampleRate {
            NonZero::new(
                self.current
                    .as_ref()
                    .map(|c| c.sample_rate_hz)
                    .unwrap_or(48_000),
            )
            .unwrap()
        }

        fn total_duration(&self) -> Option<Duration> {
            None
        }
    }

    fn drain_packets<const C: usize>(consumer: &mut Consumer<TapPacket<C>>) -> Vec<TapPacket<C>> {
        let avail = consumer.slots();
        if avail == 0 {
            return Vec::new();
        }
        let chunk = consumer.read_chunk(avail).expect("read_chunk");
        let (a_len, b_len, out) = {
            let (a, b) = chunk.as_slices();
            let mut out = Vec::with_capacity(a.len() + b.len());
            out.extend(a.iter().cloned());
            out.extend(b.iter().cloned());
            (a.len(), b.len(), out)
        };
        chunk.commit(a_len + b_len);
        out
    }

    #[test]
    fn emits_initial_format_before_frames() {
        let src = MockSpanSource::new(vec![SpanChunk {
            channels: 2,
            sample_rate_hz: 48_000,
            samples: vec![0.1, 0.2, 0.3, 0.4],
        }]);

        let (reader, mut adapter) = TapReader::<2>::new(src);
        while adapter.next().is_some() {}
        drop(adapter);

        let mut cons = reader.consumer.lock().unwrap().take().unwrap();
        let packets = drain_packets(&mut cons);
        assert!(!packets.is_empty());
        assert!(matches!(
            packets[0],
            TapPacket::Format(FrameFormat {
                channels: 2,
                sample_rate_hz: 48_000
            })
        ));
    }

    #[test]
    fn emits_format_on_change() {
        let src = MockSpanSource::new(vec![
            SpanChunk {
                channels: 2,
                sample_rate_hz: 44_100,
                samples: vec![0.0, 0.1, 0.2, 0.3],
            },
            SpanChunk {
                channels: 1,
                sample_rate_hz: 48_000,
                samples: vec![0.4, 0.5],
            },
        ]);

        let (reader, mut adapter) = TapReader::<2>::new(src);
        while adapter.next().is_some() {}
        drop(adapter);

        let mut cons = reader.consumer.lock().unwrap().take().unwrap();
        let packets = drain_packets(&mut cons);
        let formats: Vec<_> = packets
            .iter()
            .filter_map(|p| match p {
                TapPacket::Format(f) => Some(*f),
                TapPacket::Frame(_) => None,
            })
            .collect();

        assert!(formats.len() >= 2);
        assert_eq!(
            formats[0],
            FrameFormat {
                channels: 2,
                sample_rate_hz: 44_100
            }
        );
        assert_eq!(
            formats[1],
            FrameFormat {
                channels: 1,
                sample_rate_hz: 48_000
            }
        );
    }

    #[test]
    fn frame_lengths_match_output_channels() {
        let src = MockSpanSource::new(vec![SpanChunk {
            channels: 4,
            sample_rate_hz: 48_000,
            samples: vec![1.0, 2.0, 3.0, 4.0],
        }]);

        let (reader, mut adapter) = TapReader::<2>::new(src);
        while adapter.next().is_some() {}
        drop(adapter);

        let mut cons = reader.consumer.lock().unwrap().take().unwrap();
        let packets = drain_packets(&mut cons);
        for packet in packets {
            if let TapPacket::Frame(frame) = packet {
                assert_eq!(frame.len(), 2);
            }
        }
    }

    #[test]
    fn seek_resets_format_and_partial_state() {
        let src = MockSpanSource::new(vec![SpanChunk {
            channels: 2,
            sample_rate_hz: 44_100,
            samples: vec![0.0, 0.1, 0.2, 0.3],
        }]);

        let (reader, mut adapter) = TapReader::<2>::new(src);
        let _ = adapter.next();
        let _ = adapter.try_seek(Duration::from_secs(0));
        while adapter.next().is_some() {}
        drop(adapter);

        let mut cons = reader.consumer.lock().unwrap().take().unwrap();
        let packets = drain_packets(&mut cons);
        assert!(packets.iter().any(|p| matches!(p, TapPacket::Format(_))));
    }
}
