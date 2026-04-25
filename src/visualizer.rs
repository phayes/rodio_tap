//! Real-time spectrum analysis utilities built on top of [`crate::FrameReader`].
//!
//! This module provides a callback-oriented visualizer pipeline:
//!
//! 1. Pull tapped frame batches from a `TapReader` via `FrameReader`.
//! 2. Maintain per-channel rolling sample history.
//! 3. Run FFT analysis at a cadence derived from [`VisualizerConfig::period`].
//! 4. Emit per-channel peak/RMS plus configured frequency-bin magnitudes.
//!
//! The bin layout (`hz_lo` / `hz_hi`) is fixed by config, while each callback's effective
//! analyzable maximum is still clamped by stream Nyquist (`sample_rate_hz / 2`).
//! Bins above Nyquist are emitted as `0.0`.
//!
//! For async runtime integration, use [`Visualizer::run_with_frame_reader_async`] (requires the
//! `async` feature).
//!
//! # Full Example
//!
//! ```no_run
//! use rodio::source::SineWave;
//! use rodio::{DeviceSinkBuilder, Player, Source};
//! use std::sync::Arc;
//! use std::thread;
//! use std::time::Duration;
//! use rodio_tap::{TapReader, Visualizer, VisualizerConfig};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Build a simple test tone and loop it forever.
//!     let tone = SineWave::new(440.0).amplify(0.2).repeat_infinite();
//!
//!     // Tap the source before sending it to playback.
//!     let (tap_reader, tap_adapter) = TapReader::<2>::new(tone);
//!
//!     // Play audio through rodio.
//!     let mut sink = DeviceSinkBuilder::open_default_sink()?;
//!     sink.log_on_drop(false);
//!     let player = Player::connect_new(sink.mixer());
//!     player.append(tap_adapter);
//!     player.play();
//!
//!     // Visualizer callback runs forever, so run it on a worker thread.
//!     let tap_for_visualizer = Arc::clone(&tap_reader);
//!     thread::spawn(move || {
//!         let config = VisualizerConfig {
//!             period: Duration::from_millis(33), // ~30 FPS updates
//!             ..Default::default()
//!         };
//!         let bins = config.frequency_bins(); // stable hz ranges for each bin
//!
//!         Visualizer::<2>::run_with_frame_reader(
//!             move || Some(Arc::clone(&tap_for_visualizer)),
//!             config,
//!             move |channels, sample_rate_hz| {
//!                 if let Some(ch0) = channels.first() {
//!                     // Print only the first few bins for demo purposes.
//!                     for (i, magnitude) in ch0.bins.iter().copied().take(5).enumerate() {
//!                         let range = &bins[i];
//!                         println!(
//!                             "[{} Hz] {:>6.0}..{:>6.0} Hz => {:.4}",
//!                             sample_rate_hz, range.hz_lo, range.hz_hi, magnitude
//!                         );
//!                     }
//!                     println!("---");
//!                 }
//!             },
//!         );
//!     });
//!
//!     // Keep main alive while audio + visualizer run.
//!     thread::sleep(Duration::from_secs(1));
//!     Ok(())
//! }
//! ```

#[cfg(feature = "async")]
use crate::AsyncFrameReader;
use crate::{FrameReader, FrameReaderConfig, TapReader};
use arrayvec::ArrayVec;
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

/// Approximate lower bound of typical human hearing.
pub const LOW_FREQUENCY_HUMAN: f32 = 20.0;
/// Sub-audible bass floor useful for visualizing rumble/very low end.
pub const LOW_FREQUENCY_SUB_BASS: f32 = 10.0;
/// Practical top-end for human hearing.
///
/// Note: effective analyzed maximum is always clamped by stream Nyquist (`sample_rate_hz / 2`).
pub const TOP_FREQUENCY_HUMAN: f32 = 20_000.0;
/// Nyquist limit of 44.1 kHz audio (CD sample rate).
///
/// Note: for lower sample rates, bins above stream Nyquist are emitted as `0.0`.
pub const TOP_FREQUENCY_CD: f32 = 22_050.0;
/// Nyquist limit of 48 kHz audio.
///
/// Note: this exceeds Nyquist for 44.1 kHz streams, so for those streams the upper range is zero-filled.
pub const TOP_FREQUENCY_48K: f32 = 24_000.0;

#[derive(Debug, Clone)]
/// Frequency-domain bin transform configuration.
pub enum Transform {
    /// Log-spaced usize number of bins between a minimum and maximum frequency.
    FourierLog(usize),
    /// Linearly-spaced usize number of bins between a minimum and maximum frequency.
    FourierLinear(usize),
    /// User-provided bin ranges used as-is.
    FourierCustom(Vec<FrequencyBin>),
}

/// Error returned by visualizer configuration validation.
#[derive(Debug, Clone, PartialEq)]
pub enum VisualizerError {
    PeriodMustBePositive,
    BinCountMustBePositive,
    MinFrequencyMustBePositive {
        min_frequency_hz: f32,
    },
    MaxFrequencyMustExceedMin {
        min_frequency_hz: f32,
        max_frequency_hz: f32,
    },
    CustomBinsEmpty,
    CustomBinLowerEdgeMustBePositive {
        index: usize,
        hz_lo: f32,
    },
    CustomBinUpperEdgeMustExceedLower {
        index: usize,
        hz_lo: f32,
        hz_hi: f32,
    },
    CustomBinsMustBeSortedAndNonOverlapping {
        previous_index: usize,
        current_index: usize,
        previous_hz_hi: f32,
        current_hz_lo: f32,
    },
}

impl std::fmt::Display for VisualizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VisualizerError::PeriodMustBePositive => {
                write!(f, "VisualizerConfig.period must be > 0")
            }
            VisualizerError::BinCountMustBePositive => {
                write!(f, "transform bin count must be > 0")
            }
            VisualizerError::MinFrequencyMustBePositive { min_frequency_hz } => {
                write!(f, "min_frequency_hz must be > 0 (got {min_frequency_hz})")
            }
            VisualizerError::MaxFrequencyMustExceedMin {
                min_frequency_hz,
                max_frequency_hz,
            } => write!(
                f,
                "max_frequency_hz must be > min_frequency_hz (got min={min_frequency_hz}, max={max_frequency_hz})"
            ),
            VisualizerError::CustomBinsEmpty => {
                write!(f, "Transform::FourierCustom bins must not be empty")
            }
            VisualizerError::CustomBinLowerEdgeMustBePositive { index, hz_lo } => write!(
                f,
                "custom bin at index {index} must have hz_lo > 0 (got {hz_lo})"
            ),
            VisualizerError::CustomBinUpperEdgeMustExceedLower {
                index,
                hz_lo,
                hz_hi,
            } => write!(
                f,
                "custom bin at index {index} must have hz_hi > hz_lo (got hz_lo={hz_lo}, hz_hi={hz_hi})"
            ),
            VisualizerError::CustomBinsMustBeSortedAndNonOverlapping {
                previous_index,
                current_index,
                previous_hz_hi,
                current_hz_lo,
            } => write!(
                f,
                "custom bins must be sorted and non-overlapping (bin {current_index}.hz_lo={current_hz_lo} < bin {previous_index}.hz_hi={previous_hz_hi})"
            ),
        }
    }
}

impl std::error::Error for VisualizerError {}

impl Default for Transform {
    fn default() -> Self {
        Self::FourierLog(28)
    }
}

impl Transform {
    /// Validate this transform against the provided generated-bin frequency range.
    ///
    /// For `FourierLog` and `FourierLinear`, checks:
    /// - bin count is greater than zero
    /// - `min_frequency_hz > 0.0`
    /// - `max_frequency_hz > min_frequency_hz`
    ///
    /// For `FourierCustom`, checks:
    /// - at least one bin is provided
    /// - each bin has `hz_lo > 0.0` and `hz_hi > hz_lo`
    /// - bins are sorted and non-overlapping (`next.hz_lo >= prev.hz_hi`)
    pub fn validate(
        &self,
        min_frequency_hz: f32,
        max_frequency_hz: f32,
    ) -> Result<(), VisualizerError> {
        match self {
            Transform::FourierLog(num_bins) | Transform::FourierLinear(num_bins) => {
                if *num_bins == 0 {
                    return Err(VisualizerError::BinCountMustBePositive);
                }
                if min_frequency_hz <= 0.0 {
                    return Err(VisualizerError::MinFrequencyMustBePositive { min_frequency_hz });
                }
                if max_frequency_hz <= min_frequency_hz {
                    return Err(VisualizerError::MaxFrequencyMustExceedMin {
                        min_frequency_hz,
                        max_frequency_hz,
                    });
                }
            }
            Transform::FourierCustom(bins) => {
                if bins.is_empty() {
                    return Err(VisualizerError::CustomBinsEmpty);
                }
                for (idx, bin) in bins.iter().enumerate() {
                    if bin.hz_lo <= 0.0 {
                        return Err(VisualizerError::CustomBinLowerEdgeMustBePositive {
                            index: idx,
                            hz_lo: bin.hz_lo,
                        });
                    }
                    if bin.hz_hi <= bin.hz_lo {
                        return Err(VisualizerError::CustomBinUpperEdgeMustExceedLower {
                            index: idx,
                            hz_lo: bin.hz_lo,
                            hz_hi: bin.hz_hi,
                        });
                    }
                    if let Some(prev) = idx.checked_sub(1).and_then(|i| bins.get(i)) {
                        if bin.hz_lo < prev.hz_hi {
                            return Err(VisualizerError::CustomBinsMustBeSortedAndNonOverlapping {
                                previous_index: idx - 1,
                                current_index: idx,
                                previous_hz_hi: prev.hz_hi,
                                current_hz_lo: bin.hz_lo,
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Compute frequency-bands for this transform.
    pub fn frequency_bins(
        &self,
        min_frequency_hz: f32,
        max_frequency_hz: f32,
    ) -> Vec<FrequencyBin> {
        match self {
            Transform::FourierLog(_) => {
                let edges = self.compute_log_edges(min_frequency_hz, max_frequency_hz);
                edges_to_frequency_bins(&edges)
            }
            Transform::FourierLinear(_) => {
                let edges = self.compute_linear_edges(min_frequency_hz, max_frequency_hz);
                edges_to_frequency_bins(&edges)
            }
            Transform::FourierCustom(bins) => bins.clone(),
        }
    }

    fn configured_max_frequency_hz(&self, max_frequency_hz: f32) -> f32 {
        match self {
            Transform::FourierLog(_) | Transform::FourierLinear(_) => max_frequency_hz,
            Transform::FourierCustom(bins) => bins.iter().map(|bin| bin.hz_hi).fold(0.0, f32::max),
        }
    }

    /// Build geometrically spaced band edges in Hz.
    ///
    /// Uses a constant-ratio progression (`min_hz * (max_hz / min_hz).powf(t)`), so it is
    /// logarithmic in the broad sense but not tied to any specific log base.
    fn compute_log_edges(&self, min_hz: f32, max_hz: f32) -> Vec<f32> {
        let num_bands = match self {
            Transform::FourierLog(n) | Transform::FourierLinear(n) => *n,
            Transform::FourierCustom(_) => {
                panic!("Transform::FourierCustom does not generate computed log edges")
            }
        };
        let mut edges = Vec::with_capacity(num_bands + 1);
        let ratio = max_hz / min_hz;
        for idx in 0..=num_bands {
            let t = idx as f32 / num_bands as f32;
            edges.push(min_hz * ratio.powf(t));
        }
        edges
    }

    /// Build linearly-spaced band edges in Hz.
    fn compute_linear_edges(&self, min_hz: f32, max_hz: f32) -> Vec<f32> {
        let num_bands = match self {
            Transform::FourierLog(n) | Transform::FourierLinear(n) => *n,
            Transform::FourierCustom(_) => {
                panic!("Transform::FourierCustom does not generate computed linear edges")
            }
        };
        let mut edges = Vec::with_capacity(num_bands + 1);
        let span = max_hz - min_hz;
        for idx in 0..=num_bands {
            let t = idx as f32 / num_bands as f32;
            edges.push(min_hz + span * t);
        }
        edges
    }
}

#[derive(Debug, Clone)]
/// Configuration for spectrum analysis and callback cadence.
pub struct VisualizerConfig {
    /// Target callback cadence.
    ///
    /// Also used to derive the internal FFT sample window at the current sample rate.
    ///
    /// Default: `Duration::from_millis(33)`.
    pub period: Duration,
    /// Frequency transform used to build output bins.
    ///
    /// Default: `Transform::FourierLog(28)`.
    pub transform: Transform,
    /// Lower bound (Hz) for generated bin construction (`FourierLog` / `FourierLinear`).
    ///
    /// Default: `LOW_FREQUENCY_HUMAN` (`20.0`).
    pub min_frequency_hz: f32,
    /// Upper bound (Hz) for generated bin construction (`FourierLog` / `FourierLinear`).
    ///
    /// Per-callback effective max is `min(max_frequency_hz, sample_rate_hz / 2)`.
    ///
    /// Default: `TOP_FREQUENCY_HUMAN` (`20_000.0`).
    pub max_frequency_hz: f32,
    /// If `true`, divide FFT magnitudes by FFT length.
    ///
    /// This helps make magnitudes less dependent on FFT size.
    ///
    /// Default: `false`.
    pub normalize_by_fft_size: bool,
    /// If `false`, do not emit callback frames until each channel has accumulated
    /// a full FFT window of samples.
    ///
    /// Default: `false`.
    pub emit_before_fft_window_full: bool,
}

impl Default for VisualizerConfig {
    fn default() -> Self {
        Self {
            period: Duration::from_millis(33),
            transform: Transform::default(),
            min_frequency_hz: LOW_FREQUENCY_HUMAN,
            max_frequency_hz: TOP_FREQUENCY_HUMAN,
            normalize_by_fft_size: false,
            emit_before_fft_window_full: false,
        }
    }
}

impl VisualizerConfig {
    pub fn validate(&self) -> Result<(), VisualizerError> {
        if self.period.as_nanos() == 0 {
            return Err(VisualizerError::PeriodMustBePositive);
        }
        self.transform
            .validate(self.min_frequency_hz, self.max_frequency_hz)
    }

    /// Compute fixed frequency-band metadata for this config.
    ///
    /// Magnitudes are not included here; they are emitted in `ChannelSpectrum.bins`.
    pub fn frequency_bins(&self) -> Vec<FrequencyBin> {
        self.transform
            .frequency_bins(self.min_frequency_hz, self.max_frequency_hz)
    }

    /// Maximum configured frequency target for this transform.
    fn configured_max_frequency_hz(&self) -> f32 {
        self.transform
            .configured_max_frequency_hz(self.max_frequency_hz)
    }
}

#[derive(Debug, Clone)]
/// Frequency range metadata for one visualizer band.
pub struct FrequencyBin {
    /// Inclusive-ish lower edge of the band in Hz.
    pub hz_lo: f32,
    /// Upper edge of the band in Hz.
    pub hz_hi: f32,
}

impl FrequencyBin {
    /// Create a frequency bin from lower/upper edges in Hz.
    ///
    /// # Panics
    ///
    /// Panics when `hz_lo <= 0.0` or `hz_hi <= hz_lo`.
    pub fn new(hz_lo: f32, hz_hi: f32) -> Self {
        assert!(hz_lo > 0.0, "FrequencyBin.hz_lo must be > 0");
        assert!(hz_hi > hz_lo, "FrequencyBin.hz_hi must be > hz_lo");
        Self { hz_lo, hz_hi }
    }
}

#[derive(Debug, Clone)]
/// Per-channel visualizer output for one callback frame.
pub struct ChannelSpectrum {
    /// Absolute sample peak over the current callback batch.
    pub peak: f32,
    /// RMS value over the current callback batch.
    pub rms: f32,
    /// Magnitude per configured frequency band.
    ///
    /// Index-aligned with `Visualizer::frequency_bins()` / `VisualizerConfig::frequency_bins()`.
    pub bins: Vec<f32>,
}

#[derive(Debug, Clone)]
/// Full visualizer output for one callback frame.
pub struct VisualizerFrame {
    /// Stream sample rate observed for this callback frame.
    pub sample_rate_hz: u32,
    /// One `ChannelSpectrum` per channel.
    pub channels: Vec<ChannelSpectrum>,
}

/// Stateful real-time spectrum analyzer.
///
/// `C` is the maximum supported channel count and should match your `TapReader<C>` / `FrameReader<C>`.
///
/// See a full runnable example:
/// [examples/wav_visualizer_simple.rs](https://github.com/phayes/rodio_tap/blob/master/examples/wav_visualizer_simple.rs)
///
/// # Example
///
/// ```
/// use rodio::source::SineWave;
/// use rodio::{DeviceSinkBuilder, Player, Source};
/// use std::sync::Arc;
/// use std::thread;
/// use std::time::Duration;
/// use rodio_tap::{TapReader, Visualizer, VisualizerConfig};
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let tone = SineWave::new(440.0).amplify(0.2).repeat_infinite();
///     let (tap_reader, tap_adapter) = TapReader::<2>::new(tone);
///
///     let mut sink = DeviceSinkBuilder::open_default_sink()?;
///     sink.log_on_drop(false);
///     let player = Player::connect_new(sink.mixer());
///     player.append(tap_adapter);
///     player.play();
///
///     let tap_for_visualizer = Arc::clone(&tap_reader);
///     thread::spawn(move || {
///         let config = VisualizerConfig {
///             period: Duration::from_millis(33),
///             ..Default::default()
///         };
///         let bins = config.frequency_bins();
///
///         Visualizer::<2>::run_with_frame_reader(
///             move || Some(Arc::clone(&tap_for_visualizer)),
///             config,
///             move |channels, sample_rate_hz| {
///                 if let Some(ch0) = channels.first() {
///                     for (i, magnitude) in ch0.bins.iter().copied().take(5).enumerate() {
///                         let range = &bins[i];
///                         println!(
///                             "[{} Hz] {:>6.0}..{:>6.0} Hz => {:.4}",
///                             sample_rate_hz, range.hz_lo, range.hz_hi, magnitude
///                         );
///                     }
///                 }
///             },
///         );
///     });
///
///     thread::sleep(Duration::from_secs(1));
///     Ok(())
/// }
/// ```
pub struct Visualizer<const C: usize = 2> {
    config: VisualizerConfig,
    frequency_bins: Vec<FrequencyBin>,
    histories: Vec<VecDeque<f32>>,
    fft_planner: FftPlanner<f32>,
    fft: Arc<dyn Fft<f32>>,
    fft_input: Vec<Complex32>,
    fft_spectrum: Vec<Complex32>,
    fft_len: usize,
    last_sample_rate_hz: Option<u32>,
}

impl<const C: usize> Visualizer<C> {
    /// Create a new visualizer with the given config.
    ///
    /// Returns an error if the config is invalid.
    ///
    /// # Panics
    ///
    /// Panics if `C == 0`.
    pub fn new(config: VisualizerConfig) -> Result<Self, VisualizerError> {
        assert!(C > 0, "Visualizer requires C > 0");
        config.validate()?;

        let fft_len = 1usize;
        let mut fft_planner = FftPlanner::new();
        let fft = fft_planner.plan_fft_forward(fft_len);

        Ok(Self {
            frequency_bins: config.frequency_bins(),
            histories: (0..C).map(|_| VecDeque::new()).collect(),
            fft_planner,
            fft,
            fft_input: vec![Complex32::new(0.0, 0.0); fft_len],
            fft_spectrum: vec![Complex32::new(0.0, 0.0); fft_len],
            fft_len,
            last_sample_rate_hz: None,
            config,
        })
    }

    /// Access the effective config.
    pub fn config(&self) -> &VisualizerConfig {
        &self.config
    }

    /// Access fixed frequency-band metadata used by this visualizer.
    ///
    /// These ranges are static for the life of the visualizer instance.
    pub fn frequency_bins(&self) -> &[FrequencyBin] {
        &self.frequency_bins
    }

    /// Runner that wires `FrameReader` and visualizer processing together.
    ///
    /// This method never returns and should generally be run on a dedicated thread.
    ///
    /// Callback receives per-channel spectrum data and the current sample rate.
    pub fn run_with_frame_reader<G, F>(tap_fn: G, config: VisualizerConfig, mut callback: F) -> !
    where
        G: Fn() -> Option<Arc<TapReader<C>>> + Send + Sync + 'static,
        F: FnMut(&[ChannelSpectrum], u32) + Send + 'static,
    {
        let reader_config = FrameReaderConfig {
            time_per_batch: Some(config.period),
            frames_per_batch: None,
            ..Default::default()
        };
        let mut reader = FrameReader::<C>::new_with_config(reader_config, tap_fn);
        let mut visualizer = Visualizer::<C>::new(config)
            .unwrap_or_else(|err| panic!("Visualizer config is invalid: {err}"));

        reader.run(move |batch, channels, sample_rate_hz| {
            if let Some(frame) = visualizer.process_batch(batch, channels, sample_rate_hz) {
                callback(&frame.channels, frame.sample_rate_hz);
            }
        });
    }

    /// Async runner that wires `AsyncFrameReader` and visualizer processing together.
    ///
    /// Requires crate feature `async`.
    ///
    /// This method never returns and should be spawned on a runtime task.
    #[cfg(feature = "async")]
    pub async fn run_with_frame_reader_async<G, F>(
        tap_fn: G,
        config: VisualizerConfig,
        mut callback: F,
    ) -> !
    where
        G: Fn() -> Option<Arc<TapReader<C>>> + Send + Sync + 'static,
        F: FnMut(&[ChannelSpectrum], u32) + Send + 'static,
    {
        let reader_config = FrameReaderConfig {
            time_per_batch: Some(config.period),
            frames_per_batch: None,
            ..Default::default()
        };
        let mut reader = AsyncFrameReader::<C>::new_with_config(reader_config, tap_fn);
        let mut visualizer = Visualizer::<C>::new(config)
            .unwrap_or_else(|err| panic!("Visualizer config is invalid: {err}"));

        reader
            .run(move |batch, channels, sample_rate_hz| {
                if let Some(frame) = visualizer.process_batch(batch, channels, sample_rate_hz) {
                    callback(&frame.channels, frame.sample_rate_hz);
                }
            })
            .await
    }

    /// Process one `FrameReader` callback batch into a visualizer frame.
    ///
    /// Returns `None` when input is invalid/empty or when `emit_before_fft_window_full` is
    /// disabled and there is not yet enough history for FFT analysis.
    pub fn process_batch(
        &mut self,
        batch: &[ArrayVec<f32, C>],
        channels: usize,
        sample_rate_hz: u32,
    ) -> Option<VisualizerFrame> {
        if channels == 0 || channels > C || sample_rate_hz == 0 || batch.is_empty() {
            return None;
        }

        let fft_len = derive_fft_len(self.config.period, sample_rate_hz);
        let sample_rate_changed = self.last_sample_rate_hz != Some(sample_rate_hz);
        if sample_rate_changed {
            for history in &mut self.histories {
                history.clear();
            }
            self.last_sample_rate_hz = Some(sample_rate_hz);
        }
        if fft_len != self.fft_len {
            self.reconfigure_fft(fft_len);
        }

        let mut peak = vec![0.0_f32; channels];
        let mut sum_sq = vec![0.0_f32; channels];
        let mut count = vec![0usize; channels];

        for frame in batch {
            if frame.len() < channels {
                continue;
            }
            for ch in 0..channels {
                let sample = frame[ch];
                let abs = sample.abs();
                if abs > peak[ch] {
                    peak[ch] = abs;
                }
                sum_sq[ch] += sample * sample;
                count[ch] += 1;

                let history = &mut self.histories[ch];
                if history.len() == self.fft_len {
                    history.pop_front();
                }
                history.push_back(sample);
            }
        }

        if !self.config.emit_before_fft_window_full
            && self.histories[..channels]
                .iter()
                .any(|history| history.len() < self.fft_len)
        {
            return None;
        }

        let mut channel_spectra = Vec::with_capacity(channels);
        for ch in 0..channels {
            let magnitudes = self.compute_fft_magnitudes(ch, sample_rate_hz);
            let bins = self.compute_bin_magnitudes(sample_rate_hz, &magnitudes);
            let rms = if count[ch] > 0 {
                (sum_sq[ch] / count[ch] as f32).sqrt()
            } else {
                0.0
            };

            channel_spectra.push(ChannelSpectrum {
                peak: peak[ch],
                rms,
                bins,
            });
        }

        Some(VisualizerFrame {
            sample_rate_hz,
            channels: channel_spectra,
        })
    }

    /// Rebuild FFT plan and buffers for a new FFT length.
    fn reconfigure_fft(&mut self, fft_len: usize) {
        self.fft_len = fft_len.max(1);
        self.fft = self.fft_planner.plan_fft_forward(self.fft_len);
        self.fft_input = vec![Complex32::new(0.0, 0.0); self.fft_len];
        self.fft_spectrum = vec![Complex32::new(0.0, 0.0); self.fft_len];
        for history in &mut self.histories {
            while history.len() > self.fft_len {
                history.pop_front();
            }
        }
    }

    /// Compute linear magnitudes for FFT bins `0..=N/2` for one channel.
    fn compute_fft_magnitudes(&mut self, channel: usize, sample_rate_hz: u32) -> Vec<f32> {
        let history = &self.histories[channel];
        self.fft_input.fill(Complex32::new(0.0, 0.0));

        let start = self.fft_len.saturating_sub(history.len());
        for (i, sample) in history.iter().enumerate() {
            let idx = start + i;
            let windowed = *sample * hann_window(idx, self.fft_len);
            self.fft_input[idx] = Complex32::new(windowed, 0.0);
        }

        self.fft_spectrum.copy_from_slice(&self.fft_input);
        self.fft.process(&mut self.fft_spectrum);

        let max_bin = (self.fft_len / 2).max(1);
        let mut magnitudes = vec![0.0_f32; max_bin + 1];
        for (idx, slot) in magnitudes.iter_mut().enumerate() {
            let c = self.fft_spectrum[idx];
            let mut mag = (c.re * c.re + c.im * c.im).sqrt();
            if self.config.normalize_by_fft_size {
                mag /= self.fft_len as f32;
            }
            *slot = mag;
        }

        // Keep the effective upper analysis limit tied to stream Nyquist.
        let effective_max_hz = self
            .config
            .configured_max_frequency_hz()
            .min(sample_rate_hz as f32 * 0.5);
        if effective_max_hz <= 0.0 {
            magnitudes.fill(0.0);
        }

        magnitudes
    }

    /// Aggregate FFT bin magnitudes into configured frequency bands.
    ///
    /// Bins entirely above effective Nyquist are emitted as `0.0`.
    fn compute_bin_magnitudes(&self, sample_rate_hz: u32, magnitudes: &[f32]) -> Vec<f32> {
        let mut bins = Vec::with_capacity(self.frequency_bins.len());
        let nyquist = sample_rate_hz as f32 * 0.5;
        let effective_max = self.config.configured_max_frequency_hz().min(nyquist);
        let max_bin = magnitudes.len().saturating_sub(1);

        for band in &self.frequency_bins {
            let hz_lo = band.hz_lo;
            let hz_hi = band.hz_hi;

            let magnitude = if hz_lo >= effective_max || max_bin == 0 {
                0.0
            } else {
                let analyze_hi = hz_hi.min(effective_max);
                if analyze_hi <= hz_lo {
                    0.0
                } else {
                    let bin_lo = hz_to_bin(hz_lo, self.fft_len, sample_rate_hz).min(max_bin);
                    let bin_hi = hz_to_bin(analyze_hi, self.fft_len, sample_rate_hz).min(max_bin);
                    if bin_hi < bin_lo {
                        0.0
                    } else {
                        let mut sum = 0.0_f32;
                        let mut n = 0usize;
                        for &mag in &magnitudes[bin_lo..=bin_hi] {
                            sum += mag;
                            n += 1;
                        }
                        if n == 0 { 0.0 } else { sum / n as f32 }
                    }
                }
            };

            bins.push(magnitude);
        }

        bins
    }
}

/// Derive internal FFT size from callback period and sample rate.
///
/// The result is rounded to a power-of-two for FFT efficiency.
pub(crate) fn derive_fft_len(period: Duration, sample_rate_hz: u32) -> usize {
    let frames =
        ((sample_rate_hz as u128 * period.as_nanos() + 500_000_000) / 1_000_000_000).max(1);

    // Keep analysis size power-of-two for FFT efficiency.
    let target = usize::try_from(frames).unwrap_or(usize::MAX / 2).max(1);
    target.next_power_of_two().max(1)
}

/// Convert edge list into `[lo, hi]` frequency ranges.
pub(crate) fn edges_to_frequency_bins(edges: &[f32]) -> Vec<FrequencyBin> {
    edges
        .windows(2)
        .map(|range| FrequencyBin::new(range[0], range[1]))
        .collect()
}

/// Map frequency (Hz) to FFT bin index.
pub(crate) fn hz_to_bin(hz: f32, fft_len: usize, sample_rate_hz: u32) -> usize {
    if sample_rate_hz == 0 {
        return 0;
    }
    (((hz * fft_len as f32) / sample_rate_hz as f32).floor() as usize).min(fft_len / 2)
}

/// Hann window coefficient at `index` for window length `len`.
pub(crate) fn hann_window(index: usize, len: usize) -> f32 {
    if len <= 1 {
        1.0
    } else {
        let n = index as f32;
        let denom = (len - 1) as f32;
        0.5 - 0.5 * (2.0 * std::f32::consts::PI * n / denom).cos()
    }
}
