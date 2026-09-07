//! Real-time spectrum analysis utilities built on top of [`crate::FrameReader`].
//!
//! This module provides a callback-oriented visualizer pipeline:
//!
//! 1. Pull tapped frame batches from a `TapReader` via `FrameReader`.
//! 2. Maintain per-channel rolling sample history.
//! 3. Run long bass and short upper-frequency FFT windows at
//!    [`VisualizerConfig::period`] cadence.
//! 4. Emit per-hop peak/RMS plus normalized multi-resolution frequency magnitudes.
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
//!             bass_window_duration: Duration::from_millis(170),
//!             upper_window_duration: Duration::from_millis(33),
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
//!                     for (i, value) in ch0.bins.iter().take(5).enumerate() {
//!                         let range = &bins[i];
//!                         println!(
//!                             "[{} Hz] {:>6.0}..{:>6.0} Hz => {:.4}",
//!                             sample_rate_hz, range.hz_lo, range.hz_hi, value.magnitude
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
use crate::batch::duration_to_frames;
use crate::{FrameReader, FrameReaderConfig, TapReader};
use arrayvec::ArrayVec;
use realfft::num_complex::Complex32;
use realfft::{RealFftPlanner, RealToComplex};
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
    BassWindowMustBePositive,
    UpperWindowMustBePositive,
    WindowMustNotBeShorterThanPeriod,
    BassWindowMustNotBeShorterThanUpperWindow,
    CrossoverMustBeWithinFrequencyRange {
        crossover_frequency_hz: f32,
    },
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
}

impl std::fmt::Display for VisualizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VisualizerError::PeriodMustBePositive => {
                write!(f, "VisualizerConfig.period must be > 0")
            }
            VisualizerError::BassWindowMustBePositive => {
                write!(f, "VisualizerConfig.bass_window_duration must be > 0")
            }
            VisualizerError::UpperWindowMustBePositive => {
                write!(f, "VisualizerConfig.upper_window_duration must be > 0")
            }
            VisualizerError::WindowMustNotBeShorterThanPeriod => {
                write!(f, "visualizer FFT windows must be at least period")
            }
            VisualizerError::BassWindowMustNotBeShorterThanUpperWindow => {
                write!(
                    f,
                    "bass_window_duration must be at least upper_window_duration"
                )
            }
            VisualizerError::CrossoverMustBeWithinFrequencyRange {
                crossover_frequency_hz,
            } => {
                write!(
                    f,
                    "crossover_frequency_hz must fall inside the configured frequency range (got {crossover_frequency_hz})"
                )
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
    ///
    /// Note: custom bins may overlap.
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

    fn configured_min_frequency_hz(&self, min_frequency_hz: f32) -> f32 {
        match self {
            Transform::FourierLog(_) | Transform::FourierLinear(_) => min_frequency_hz,
            Transform::FourierCustom(bins) => bins
                .iter()
                .map(|bin| bin.hz_lo)
                .fold(f32::INFINITY, f32::min),
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
///
/// At 48 kHz, the default power-of-two windows are 8192 samples for bass
/// (about 5.9 Hz per FFT bin) and 2048 samples for upper frequencies
/// (about 23.4 Hz per bin). Both are recalculated every 33 ms. A generated
/// frequency band that crosses 250 Hz combines normalized low and high
/// subranges from the appropriate FFTs. Hann coherent-gain normalization keeps
/// equal-amplitude tones comparable across the two FFT sizes.
///
/// [`Visualizer::process_batch`] itself never sleeps. When called directly
/// with more than one hop, it returns every completed frame in sample order.
/// The built-in synchronous and asynchronous runners provide playback pacing.
pub struct VisualizerConfig {
    /// Target callback cadence and analysis hop. Default: 33 ms.
    pub period: Duration,
    /// Long analysis window used below `crossover_frequency_hz`. Default: 170 ms.
    pub bass_window_duration: Duration,
    /// Short analysis window used at and above `crossover_frequency_hz`. Default: 33 ms.
    pub upper_window_duration: Duration,
    /// Frequency where analysis changes from the bass FFT to the upper FFT. Default: 250 Hz.
    pub crossover_frequency_hz: f32,
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
    /// Whether FFT input should be decimated when the configured frequency range permits it.
    ///
    /// When enabled, the visualizer applies as many anti-aliased 2:1 stages as possible
    /// while retaining Nyquist headroom above the highest configured frequency bin.
    ///
    /// Peak/RMS calculations and the sample rate reported in [`VisualizerFrame`]
    /// continue to use the original stream.
    ///
    /// Default: `true`.
    pub decimation: bool,
    /// Whether visualizer runners discard queued batches when analysis falls behind.
    ///
    /// When enabled, one queued batch is skipped for each complete emission interval
    /// of lateness, allowing callbacks to recover toward current playback without
    /// draining future prebuffered audio. Analysis history is reset after a skip so
    /// samples from opposite sides of the discontinuity are not placed next to each
    /// other in an FFT window. This setting affects
    /// [`Visualizer::run_with_frame_reader`] and
    /// [`Visualizer::run_with_frame_reader_async`], but not direct
    /// [`Visualizer::process_batch`] calls.
    ///
    /// Default: `true`.
    pub drop_late_batches: bool,
}

impl Default for VisualizerConfig {
    fn default() -> Self {
        Self {
            period: Duration::from_millis(33),
            bass_window_duration: Duration::from_millis(170),
            upper_window_duration: Duration::from_millis(33),
            crossover_frequency_hz: 250.0,
            transform: Transform::default(),
            min_frequency_hz: LOW_FREQUENCY_HUMAN,
            max_frequency_hz: TOP_FREQUENCY_HUMAN,
            decimation: true,
            drop_late_batches: true,
        }
    }
}

impl VisualizerConfig {
    pub fn validate(&self) -> Result<(), VisualizerError> {
        if self.period.is_zero() {
            return Err(VisualizerError::PeriodMustBePositive);
        }
        if self.bass_window_duration.is_zero() {
            return Err(VisualizerError::BassWindowMustBePositive);
        }
        if self.upper_window_duration.is_zero() {
            return Err(VisualizerError::UpperWindowMustBePositive);
        }
        if self.bass_window_duration < self.period || self.upper_window_duration < self.period {
            return Err(VisualizerError::WindowMustNotBeShorterThanPeriod);
        }
        if self.bass_window_duration < self.upper_window_duration {
            return Err(VisualizerError::BassWindowMustNotBeShorterThanUpperWindow);
        }
        self.transform
            .validate(self.min_frequency_hz, self.max_frequency_hz)?;
        let configured_min = self
            .transform
            .configured_min_frequency_hz(self.min_frequency_hz);
        let configured_max = self
            .transform
            .configured_max_frequency_hz(self.max_frequency_hz);
        if !self.crossover_frequency_hz.is_finite()
            || self.crossover_frequency_hz <= configured_min
            || self.crossover_frequency_hz >= configured_max
        {
            return Err(VisualizerError::CrossoverMustBeWithinFrequencyRange {
                crossover_frequency_hz: self.crossover_frequency_hz,
            });
        }
        Ok(())
    }

    /// Compute fixed frequency-band metadata for this config.
    ///
    /// Measurements are not included here; they are emitted in `ChannelSpectrum.bins`.
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
    /// Magnitude and power measurements per configured frequency band.
    ///
    /// Index-aligned with `Visualizer::frequency_bins()` / `VisualizerConfig::frequency_bins()`.
    pub bins: Vec<FrequencyData>,
}

/// Measurements for one configured frequency band.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FrequencyData {
    /// Mean normalized FFT magnitude across the band.
    ///
    /// Use magnitude for spectrum-bar height and other displays where a tone's
    /// amplitude should be visually intuitive and comparable between FFT window
    /// sizes. Because this is a mean, a narrow tone contributes less to a wider
    /// band containing many otherwise-quiet FFT lines.
    pub magnitude: f32,
    /// Largest normalized FFT magnitude found within the band.
    ///
    /// Use peak magnitude for spectrum bars or narrow-tone detection when a strong
    /// frequency should remain visible inside a wide band. Unlike [`Self::power`],
    /// this value is not additive, and it is more sensitive to FFT resolution and
    /// spectral leakage than the mean [`Self::magnitude`].
    pub peak_magnitude: f32,
    /// Integrated mean-square signal power within the band.
    ///
    /// Use power when comparing how much signal is present in different frequency
    /// regions, such as RGB color balance, band-level meters, or effect triggers.
    /// Power values are additive across non-overlapping bands and are normalized
    /// for FFT length and Hann-window power, allowing bass and upper FFT results
    /// to be combined. This is a linear amplitude-squared value, not decibels.
    pub power: f32,
}

impl FrequencyData {
    /// Convert [`Self::power`] to decibels relative to `reference_power`.
    ///
    /// This uses `10 * log10(power / reference_power)`. A power equal to the
    /// reference is `0 dB`, half the reference power is approximately `-3.01 dB`,
    /// and zero power returns negative infinity.
    ///
    /// Choose the reference according to the units and convention used by the
    /// input samples:
    ///
    /// - `0.5` for AES17-style dBFS with normalized `-1.0..=1.0` samples, where a
    ///   full-scale sine wave is `0 dBFS`.
    /// - `1.0` for a mean-square full-scale reference, where a full-scale sine is
    ///   approximately `-3.01 dB` and a constant full-scale signal is `0 dB`.
    /// - `4.0e-10` for dB SPL when samples are calibrated in pascals, corresponding
    ///   to `(20 µPa RMS)^2`.
    ///
    /// Normalized digital audio samples do not by themselves contain the
    /// calibration needed for dB SPL or other physical-unit standards.
    ///
    /// Use the linear [`Self::power`] value when summing bands or computing
    /// proportions. Use this conversion for logarithmically scaled displays.
    ///
    /// # Panics
    ///
    /// Panics if `reference_power` is not finite and greater than zero.
    pub fn power_db(&self, reference_power: f32) -> f32 {
        assert!(
            reference_power.is_finite() && reference_power > 0.0,
            "reference_power must be finite and greater than zero"
        );
        10.0 * (self.power / reference_power).log10()
    }
}

#[derive(Debug, Clone)]
/// Full visualizer output for one callback frame.
pub struct VisualizerFrame {
    /// Stream sample rate observed for this callback frame.
    pub sample_rate_hz: u32,
    /// One `ChannelSpectrum` per channel.
    pub channels: Vec<ChannelSpectrum>,
}

struct FftState {
    fft: Arc<dyn RealToComplex<f32>>,
    input: Vec<f32>,
    spectrum: Vec<Complex32>,
    len: usize,
}

impl FftState {
    fn new(planner: &mut RealFftPlanner<f32>, len: usize) -> Self {
        let len = len.max(1);
        let fft = planner.plan_fft_forward(len);
        let input = fft.make_input_vec();
        let spectrum = fft.make_output_vec();
        Self {
            fft,
            input,
            spectrum,
            len,
        }
    }

    fn values(&mut self, history: &VecDeque<f32>) -> Vec<FrequencyData> {
        self.input.fill(0.0);
        let available = history.len().min(self.len);
        let history_start = history.len().saturating_sub(available);
        let mut coherent_gain = 0.0;
        let mut window_power = 0.0;
        for (offset, sample) in history.iter().skip(history_start).enumerate() {
            let window = hann_window(offset, available);
            coherent_gain += window;
            window_power += window * window;
            self.input[offset] = *sample * window;
        }
        let coherent_gain = coherent_gain.max(f32::EPSILON);
        let power_normalization = (self.len as f32 * window_power).max(f32::EPSILON);
        self.fft
            .process(&mut self.input, &mut self.spectrum)
            .expect("realfft buffers must match the planned FFT length");

        let last = self.spectrum.len().saturating_sub(1);
        self.spectrum
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let one_sided_power_scale = if index == 0 || index == last {
                    1.0
                } else {
                    2.0
                };
                let magnitude = value.norm() * one_sided_power_scale / coherent_gain;
                FrequencyData {
                    magnitude,
                    peak_magnitude: magnitude,
                    power: value.norm_sqr() * one_sided_power_scale / power_normalization,
                }
            })
            .collect()
    }
}

struct FirDecimator {
    coefficients: Arc<[f32]>,
    delay: VecDeque<f32>,
    emit_next: bool,
}

impl FirDecimator {
    fn new(coefficients: Arc<[f32]>) -> Self {
        let delay = std::iter::repeat_n(0.0, coefficients.len()).collect();
        Self {
            coefficients,
            delay,
            emit_next: false,
        }
    }

    fn process(&mut self, sample: f32) -> Option<f32> {
        self.delay.pop_front();
        self.delay.push_back(sample);
        self.emit_next = !self.emit_next;
        if self.emit_next {
            return None;
        }

        Some(
            self.delay
                .iter()
                .zip(self.coefficients.iter())
                .map(|(sample, coefficient)| sample * coefficient)
                .sum(),
        )
    }
}

struct ChannelDecimator {
    stages: Vec<FirDecimator>,
}

impl ChannelDecimator {
    fn new(stage_coefficients: &[Arc<[f32]>]) -> Self {
        Self {
            stages: stage_coefficients
                .iter()
                .cloned()
                .map(FirDecimator::new)
                .collect(),
        }
    }

    fn process(&mut self, sample: f32) -> Option<f32> {
        let mut output = Some(sample);
        for stage in &mut self.stages {
            output = stage.process(output?);
        }
        output
    }
}

/// Stateful real-time spectrum analyzer.
///
/// `C` is the maximum supported channel count and should match your `TapReader<C>` / `FrameReader<C>`.
///
/// This type is available with the crate feature `visualizer`.
///
/// FFT processing uses `realfft` internally (real-to-complex transform), and this crate forwards
/// SIMD-related feature flags to `realfft`:
/// - `avx`
/// - `sse`
/// - `neon`
/// - `wasm_simd`
///
/// Example dependency setup:
/// `rodio_tap = { version = "0.2.0", features = ["visualizer", "avx"] }`
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
///             bass_window_duration: Duration::from_millis(170),
///             upper_window_duration: Duration::from_millis(33),
///             ..Default::default()
///         };
///         let bins = config.frequency_bins();
///
///         Visualizer::<2>::run_with_frame_reader(
///             move || Some(Arc::clone(&tap_for_visualizer)),
///             config,
///             move |channels, sample_rate_hz| {
///                 if let Some(ch0) = channels.first() {
///                     for (i, value) in ch0.bins.iter().take(5).enumerate() {
///                         let range = &bins[i];
///                         println!(
///                             "[{} Hz] {:>6.0}..{:>6.0} Hz => {:.4}",
///                             sample_rate_hz, range.hz_lo, range.hz_hi, value.magnitude
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
    decimators: Vec<ChannelDecimator>,
    fft_planner: RealFftPlanner<f32>,
    bass_fft: FftState,
    upper_fft: FftState,
    analysis_sample_rate_hz: u32,
    hop_frames: usize,
    hop_collected: usize,
    hop_peak: Vec<f32>,
    hop_sum_sq: Vec<f32>,
    last_sample_rate_hz: Option<u32>,
    last_channels: Option<usize>,
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

        let mut fft_planner = RealFftPlanner::<f32>::new();
        let bass_fft = FftState::new(&mut fft_planner, 1);
        let upper_fft = FftState::new(&mut fft_planner, 1);

        Ok(Self {
            frequency_bins: config.frequency_bins(),
            histories: (0..C).map(|_| VecDeque::new()).collect(),
            decimators: (0..C)
                .map(|_| ChannelDecimator { stages: Vec::new() })
                .collect(),
            fft_planner,
            bass_fft,
            upper_fft,
            analysis_sample_rate_hz: 0,
            hop_frames: 1,
            hop_collected: 0,
            hop_peak: vec![0.0; C],
            hop_sum_sq: vec![0.0; C],
            last_sample_rate_hz: None,
            last_channels: None,
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
            drop_late_batches: config.drop_late_batches,
            ..Default::default()
        };
        let mut reader = FrameReader::<C>::new_with_config(reader_config, tap_fn);
        let mut visualizer = Visualizer::<C>::new(config)
            .unwrap_or_else(|err| panic!("Visualizer config is invalid: {err}"));

        reader.run(move |batch| {
            if batch.dropped_batches > 0 {
                visualizer.reset_format(batch.channels, batch.sample_rate_hz);
            }
            for frame in
                visualizer.process_batch(batch.frames, batch.channels, batch.sample_rate_hz)
            {
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
            drop_late_batches: config.drop_late_batches,
            ..Default::default()
        };
        let mut reader = AsyncFrameReader::<C>::new_with_config(reader_config, tap_fn);
        let mut visualizer = Visualizer::<C>::new(config)
            .unwrap_or_else(|err| panic!("Visualizer config is invalid: {err}"));

        reader
            .run(move |batch| {
                if batch.dropped_batches > 0 {
                    visualizer.reset_format(batch.channels, batch.sample_rate_hz);
                }
                for frame in
                    visualizer.process_batch(batch.frames, batch.channels, batch.sample_rate_hz)
                {
                    callback(&frame.channels, frame.sample_rate_hz);
                }
            })
            .await
    }

    /// Process tapped frames and return one output for every completed emission hop.
    ///
    /// The built-in runners pass one paced hop at a time. Direct callers may pass a
    /// larger slice; every hop is retained in sample order, but this analysis method
    /// does not perform wall-clock pacing.
    pub fn process_batch(
        &mut self,
        batch: &[ArrayVec<f32, C>],
        channels: usize,
        sample_rate_hz: u32,
    ) -> Vec<VisualizerFrame> {
        if channels == 0 || channels > C || sample_rate_hz == 0 || batch.is_empty() {
            return Vec::new();
        }
        if self.last_sample_rate_hz != Some(sample_rate_hz) || self.last_channels != Some(channels)
        {
            self.reset_format(channels, sample_rate_hz);
        }

        let mut output = Vec::new();
        for frame in batch {
            if frame.len() < channels {
                continue;
            }
            for channel in 0..channels {
                let sample = frame[channel];
                self.hop_peak[channel] = self.hop_peak[channel].max(sample.abs());
                self.hop_sum_sq[channel] += sample * sample;
                if let Some(sample) = self.decimators[channel].process(sample) {
                    let history = &mut self.histories[channel];
                    if history.len() == self.bass_fft.len {
                        history.pop_front();
                    }
                    history.push_back(sample);
                }
            }
            self.hop_collected += 1;
            if self.hop_collected == self.hop_frames {
                output.push(self.build_frame(channels, sample_rate_hz));
                self.hop_collected = 0;
                self.hop_peak.fill(0.0);
                self.hop_sum_sq.fill(0.0);
            }
        }
        output
    }

    fn reset_format(&mut self, channels: usize, sample_rate_hz: u32) {
        self.last_sample_rate_hz = Some(sample_rate_hz);
        self.last_channels = Some(channels);
        self.hop_frames = duration_to_frames(self.config.period, sample_rate_hz);
        self.hop_collected = 0;
        self.hop_peak.fill(0.0);
        self.hop_sum_sq.fill(0.0);
        for history in &mut self.histories {
            history.clear();
        }
        let configured_max_frequency_hz = self
            .config
            .configured_max_frequency_hz()
            .min(sample_rate_hz as f32 * 0.5);
        let decimation_stages = decimation_stages(
            sample_rate_hz,
            self.config.decimation,
            configured_max_frequency_hz,
        );
        self.analysis_sample_rate_hz = sample_rate_hz >> decimation_stages;
        let stage_coefficients = decimation_filter_stages(
            sample_rate_hz,
            decimation_stages,
            configured_max_frequency_hz,
        );
        self.decimators = (0..C)
            .map(|_| ChannelDecimator::new(&stage_coefficients))
            .collect();
        self.bass_fft = FftState::new(
            &mut self.fft_planner,
            derive_fft_len(
                self.config.bass_window_duration,
                self.analysis_sample_rate_hz,
            ),
        );
        self.upper_fft = FftState::new(
            &mut self.fft_planner,
            derive_fft_len(
                self.config.upper_window_duration,
                self.analysis_sample_rate_hz,
            ),
        );
    }

    fn build_frame(&mut self, channels: usize, sample_rate_hz: u32) -> VisualizerFrame {
        let mut spectra = Vec::with_capacity(channels);
        for channel in 0..channels {
            let bass = self.bass_fft.values(&self.histories[channel]);
            let upper = self.upper_fft.values(&self.histories[channel]);
            let bins = self.compute_bin_values(self.analysis_sample_rate_hz, &bass, &upper);
            spectra.push(ChannelSpectrum {
                peak: self.hop_peak[channel],
                rms: (self.hop_sum_sq[channel] / self.hop_frames as f32).sqrt(),
                bins,
            });
        }
        VisualizerFrame {
            sample_rate_hz,
            channels: spectra,
        }
    }

    fn compute_bin_values(
        &self,
        analysis_sample_rate_hz: u32,
        bass: &[FrequencyData],
        upper: &[FrequencyData],
    ) -> Vec<FrequencyData> {
        let effective_max = self
            .config
            .configured_max_frequency_hz()
            .min(analysis_sample_rate_hz as f32 * 0.5);
        let crossover = self.config.crossover_frequency_hz;

        self.frequency_bins
            .iter()
            .map(|band| {
                let lo = band.hz_lo;
                let hi = band.hz_hi.min(effective_max);
                if hi <= lo {
                    return FrequencyData::default();
                }
                if hi <= crossover {
                    return aggregate_fft_range(
                        bass,
                        self.bass_fft.len,
                        analysis_sample_rate_hz,
                        lo,
                        hi,
                    );
                }
                if lo >= crossover {
                    return aggregate_fft_range(
                        upper,
                        self.upper_fft.len,
                        analysis_sample_rate_hz,
                        lo,
                        hi,
                    );
                }

                let bass_width = crossover - lo;
                let upper_width = hi - crossover;
                let bass_value = aggregate_fft_range(
                    bass,
                    self.bass_fft.len,
                    analysis_sample_rate_hz,
                    lo,
                    crossover,
                );
                let upper_value = aggregate_fft_range(
                    upper,
                    self.upper_fft.len,
                    analysis_sample_rate_hz,
                    crossover,
                    hi,
                );
                FrequencyData {
                    magnitude: (bass_value.magnitude * bass_width
                        + upper_value.magnitude * upper_width)
                        / (bass_width + upper_width),
                    peak_magnitude: bass_value.peak_magnitude.max(upper_value.peak_magnitude),
                    power: bass_value.power + upper_value.power,
                }
            })
            .collect()
    }
}

const DECIMATION_NYQUIST_HEADROOM: f32 = 1.05;

fn decimation_stages(
    source_sample_rate_hz: u32,
    enabled: bool,
    preserved_max_frequency_hz: f32,
) -> u32 {
    if !enabled {
        return 0;
    }

    let mut stages = 0;
    loop {
        let next_stages = stages + 1;
        let Some(factor) = 1_u32.checked_shl(next_stages) else {
            break;
        };
        if !source_sample_rate_hz.is_multiple_of(factor) {
            break;
        }
        let candidate_rate = source_sample_rate_hz / factor;
        if candidate_rate as f32 * 0.5 < preserved_max_frequency_hz * DECIMATION_NYQUIST_HEADROOM {
            break;
        }

        stages = next_stages;
    }
    stages
}

fn decimation_filter_stages(
    source_sample_rate_hz: u32,
    stages: u32,
    preserved_max_frequency_hz: f32,
) -> Vec<Arc<[f32]>> {
    let mut input_rate_hz = source_sample_rate_hz as f32;
    (0..stages)
        .map(|_| {
            let output_nyquist_hz = input_rate_hz * 0.25;
            let coefficients = design_decimation_filter(
                input_rate_hz,
                preserved_max_frequency_hz,
                output_nyquist_hz,
            );
            input_rate_hz *= 0.5;
            Arc::from(coefficients)
        })
        .collect()
}

fn design_decimation_filter(
    input_rate_hz: f32,
    passband_hz: f32,
    output_nyquist_hz: f32,
) -> Vec<f32> {
    let transition_hz = output_nyquist_hz - passband_hz;
    debug_assert!(transition_hz > 0.0);

    // A Blackman-windowed sinc needs approximately six samples per normalized
    // transition-width to place its first sidelobes beyond the stopband edge.
    let normalized_transition = transition_hz / input_rate_hz;
    let mut taps = (6.0 / normalized_transition).ceil() as usize;
    taps = taps.max(31);
    if taps.is_multiple_of(2) {
        taps += 1;
    }

    let cutoff = (passband_hz + output_nyquist_hz) * 0.5 / input_rate_hz;
    let center = (taps - 1) as f32 * 0.5;
    let mut coefficients = (0..taps)
        .map(|index| {
            let offset = index as f32 - center;
            let sinc = if offset == 0.0 {
                2.0 * cutoff
            } else {
                (2.0 * std::f32::consts::PI * cutoff * offset).sin()
                    / (std::f32::consts::PI * offset)
            };
            let phase = 2.0 * std::f32::consts::PI * index as f32 / (taps - 1) as f32;
            let blackman = 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos();
            sinc * blackman
        })
        .collect::<Vec<_>>();
    let dc_gain = coefficients.iter().sum::<f32>();
    for coefficient in &mut coefficients {
        *coefficient /= dc_gain;
    }
    coefficients
}

/// Derive internal FFT size from callback period and sample rate.
///
/// The result is rounded to a power-of-two for FFT efficiency.
pub(crate) fn derive_fft_len(period: Duration, sample_rate_hz: u32) -> usize {
    let frames =
        ((sample_rate_hz as u128 * period.as_nanos() + 500_000_000) / 1_000_000_000).max(1);

    // Keep analysis size power-of-two for FFT efficiency.
    let target = usize::try_from(frames)
        .unwrap_or(usize::MAX / 2)
        .clamp(1, usize::MAX / 2);
    target.next_power_of_two().max(1)
}

fn aggregate_fft_range(
    values: &[FrequencyData],
    fft_len: usize,
    sample_rate_hz: u32,
    hz_lo: f32,
    hz_hi: f32,
) -> FrequencyData {
    if values.is_empty() || hz_hi <= hz_lo {
        return FrequencyData::default();
    }
    let last = values.len() - 1;
    let first_bin = hz_to_bin(hz_lo, fft_len, sample_rate_hz).min(last);
    let last_bin = hz_to_bin(hz_hi, fft_len, sample_rate_hz).min(last);
    if last_bin < first_bin {
        return FrequencyData::default();
    }
    let range = &values[first_bin..=last_bin];
    FrequencyData {
        magnitude: range.iter().map(|value| value.magnitude).sum::<f32>() / range.len() as f32,
        peak_magnitude: range
            .iter()
            .map(|value| value.peak_magnitude)
            .fold(0.0, f32::max),
        power: range.iter().map(|value| value.power).sum(),
    }
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
    if len <= 2 {
        1.0
    } else {
        let n = index as f32;
        let denom = (len - 1) as f32;
        0.5 - 0.5 * (2.0 * std::f32::consts::PI * n / denom).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn frequency_data_converts_power_to_decibels() {
        assert!(approx_eq(
            FrequencyData {
                magnitude: 0.0,
                peak_magnitude: 0.0,
                power: 1.0,
            }
            .power_db(1.0),
            0.0,
            1e-6,
        ));
        assert!(approx_eq(
            FrequencyData {
                magnitude: 0.0,
                peak_magnitude: 0.0,
                power: 2.0,
            }
            .power_db(4.0),
            -3.010_300_2,
            1e-5,
        ));
        let silence_db = FrequencyData {
            magnitude: 0.0,
            peak_magnitude: 0.0,
            power: 0.0,
        }
        .power_db(1.0);
        assert!(silence_db.is_infinite() && silence_db.is_sign_negative());
    }

    #[test]
    fn transform_validate_rejects_zero_bins() {
        let err = Transform::FourierLog(0)
            .validate(LOW_FREQUENCY_HUMAN, TOP_FREQUENCY_HUMAN)
            .unwrap_err();
        assert_eq!(err, VisualizerError::BinCountMustBePositive);
    }

    #[test]
    fn transform_validate_custom_allows_overlapping_bins() {
        let transform = Transform::FourierCustom(vec![
            FrequencyBin::new(20.0, 100.0),
            FrequencyBin::new(90.0, 200.0),
        ]);

        assert!(
            transform
                .validate(LOW_FREQUENCY_HUMAN, TOP_FREQUENCY_HUMAN)
                .is_ok()
        );
    }

    #[test]
    fn transform_validate_custom_accepts_valid_bins() {
        let transform = Transform::FourierCustom(vec![
            FrequencyBin::new(20.0, 100.0),
            FrequencyBin::new(100.0, 300.0),
            FrequencyBin::new(300.0, 1_000.0),
        ]);

        assert!(
            transform
                .validate(LOW_FREQUENCY_HUMAN, TOP_FREQUENCY_HUMAN)
                .is_ok()
        );
    }

    #[test]
    fn transform_frequency_bins_linear_have_even_spacing() {
        let bins = Transform::FourierLinear(4).frequency_bins(20.0, 220.0);
        assert_eq!(bins.len(), 4);

        let widths: Vec<f32> = bins.iter().map(|bin| bin.hz_hi - bin.hz_lo).collect();
        assert!(approx_eq(widths[0], widths[1], 1e-5));
        assert!(approx_eq(widths[1], widths[2], 1e-5));
        assert!(approx_eq(widths[2], widths[3], 1e-5));
        assert!(approx_eq(bins[0].hz_lo, 20.0, 1e-5));
        assert!(approx_eq(bins[3].hz_hi, 220.0, 1e-5));
    }

    #[test]
    fn transform_frequency_bins_log_have_constant_ratio_edges() {
        let bins = Transform::FourierLog(4).frequency_bins(10.0, 160.0);
        assert_eq!(bins.len(), 4);

        let edges = [
            bins[0].hz_lo,
            bins[0].hz_hi,
            bins[1].hz_hi,
            bins[2].hz_hi,
            bins[3].hz_hi,
        ];
        let r0 = edges[1] / edges[0];
        let r1 = edges[2] / edges[1];
        let r2 = edges[3] / edges[2];
        let r3 = edges[4] / edges[3];

        assert!(approx_eq(r0, r1, 1e-5));
        assert!(approx_eq(r1, r2, 1e-5));
        assert!(approx_eq(r2, r3, 1e-5));
        assert!(approx_eq(edges[0], 10.0, 1e-5));
        assert!(approx_eq(edges[4], 160.0, 1e-4));
    }

    #[test]
    fn visualizer_config_validate_checks_period() {
        let config = VisualizerConfig {
            period: Duration::from_nanos(0),
            ..Default::default()
        };
        assert_eq!(
            config.validate(),
            Err(VisualizerError::PeriodMustBePositive)
        );
    }

    #[test]
    fn visualizer_config_requires_bass_window_to_be_longest() {
        let config = VisualizerConfig {
            bass_window_duration: Duration::from_millis(33),
            upper_window_duration: Duration::from_millis(100),
            ..Default::default()
        };
        assert_eq!(
            config.validate(),
            Err(VisualizerError::BassWindowMustNotBeShorterThanUpperWindow)
        );
    }

    #[test]
    fn decimation_selects_safe_power_of_two_rates() {
        let defaults = VisualizerConfig::default();
        assert!(defaults.decimation);
        assert!(defaults.drop_late_batches);
        assert_eq!(decimation_stages(44_100, true, 20_000.0), 0);
        assert_eq!(decimation_stages(96_000, true, 20_000.0), 1);
        assert_eq!(decimation_stages(192_000, true, 20_000.0), 2);
        assert_eq!(decimation_stages(48_000, true, 1_500.0), 3);
        assert_eq!(decimation_stages(96_000, false, 20_000.0), 0);
        assert_eq!(decimation_stages(96_000, true, 24_000.0), 0);
    }

    #[test]
    fn decimation_filter_rejects_aliases() {
        let coefficients = decimation_filter_stages(96_000, 1, 20_000.0);
        let mut decimator = ChannelDecimator::new(&coefficients);
        let output = (0..9_600)
            .filter_map(|index| {
                let phase = 2.0 * std::f32::consts::PI * 30_000.0 * index as f32 / 96_000.0;
                decimator.process(phase.sin())
            })
            .skip(500)
            .collect::<Vec<_>>();
        let rms =
            (output.iter().map(|sample| sample * sample).sum::<f32>() / output.len() as f32).sqrt();
        assert!(rms < 0.001, "aliased stopband RMS was {rms}");
    }

    #[test]
    fn visualizer_new_returns_error_for_invalid_transform() {
        let config = VisualizerConfig {
            transform: Transform::FourierLog(0),
            ..Default::default()
        };
        assert!(matches!(
            Visualizer::<2>::new(config),
            Err(VisualizerError::BinCountMustBePositive)
        ));
    }

    fn mono_batch(samples: impl IntoIterator<Item = f32>) -> Vec<ArrayVec<f32, 1>> {
        samples
            .into_iter()
            .map(|sample| {
                let mut frame = ArrayVec::new();
                frame.push(sample);
                frame
            })
            .collect()
    }

    #[test]
    fn oversized_input_emits_every_completed_hop() {
        let config = VisualizerConfig {
            period: Duration::from_millis(2),
            bass_window_duration: Duration::from_millis(8),
            upper_window_duration: Duration::from_millis(2),
            crossover_frequency_hz: 200.0,
            min_frequency_hz: 10.0,
            max_frequency_hz: 400.0,
            ..Default::default()
        };
        let mut visualizer = Visualizer::<1>::new(config).unwrap();
        let frames = visualizer.process_batch(&mono_batch([0.0; 10]), 1, 1_000);
        assert_eq!(frames.len(), 5);
        assert_eq!(visualizer.histories[0].len(), 8);
        assert_eq!(visualizer.bass_fft.len, 8);
        assert_eq!(visualizer.upper_fft.len, 2);
    }

    #[test]
    fn partial_window_uses_available_history_for_normalization() {
        let mut planner = RealFftPlanner::new();
        let mut fft = FftState::new(&mut planner, 8);
        let history = VecDeque::from([1.0, 1.0, 1.0, 1.0]);
        let values = fft.values(&history);
        assert!(approx_eq(values[0].magnitude, 1.0, 1e-5));
    }

    #[test]
    fn integrated_fft_power_matches_sine_mean_square() {
        let sample_rate = 4_096;
        let mut planner = RealFftPlanner::new();
        let mut fft = FftState::new(&mut planner, sample_rate);
        let history = (0..sample_rate)
            .map(|index| {
                let phase = 2.0 * std::f32::consts::PI * 125.0 * index as f32 / sample_rate as f32;
                0.5 * phase.sin()
            })
            .collect();
        let power = fft
            .values(&history)
            .iter()
            .map(|value| value.power)
            .sum::<f32>();
        assert!(approx_eq(power, 0.125, 1e-4), "power was {power}");
    }

    #[test]
    fn bass_and_upper_windows_have_comparable_normalized_gain() {
        let sample_rate = 4_096_u32;
        let config = VisualizerConfig {
            period: Duration::from_millis(125),
            bass_window_duration: Duration::from_secs(1),
            upper_window_duration: Duration::from_millis(125),
            crossover_frequency_hz: 250.0,
            min_frequency_hz: 20.0,
            max_frequency_hz: 1_500.0,
            transform: Transform::FourierCustom(vec![
                FrequencyBin::new(124.5, 125.5),
                FrequencyBin::new(999.0, 1_001.0),
            ]),
            decimation: true,
            drop_late_batches: true,
        };
        let samples = (0..sample_rate)
            .map(|index| {
                let t = index as f32 / sample_rate as f32;
                0.5 * (2.0 * std::f32::consts::PI * 125.0 * t).sin()
                    + 0.5 * (2.0 * std::f32::consts::PI * 1_000.0 * t).sin()
            })
            .collect::<Vec<_>>();
        let mut visualizer = Visualizer::<1>::new(config).unwrap();
        let output = visualizer.process_batch(&mono_batch(samples), 1, sample_rate);
        let bins = &output.last().unwrap().channels[0].bins;
        assert!(
            bins[0].magnitude > 0.35,
            "bass magnitude was {}",
            bins[0].magnitude
        );
        assert!(
            bins[1].magnitude > 0.35,
            "upper magnitude was {}",
            bins[1].magnitude
        );
        assert!(
            (bins[0].magnitude - bins[1].magnitude).abs() < 0.1,
            "values: {bins:?}"
        );
        assert!(bins[0].peak_magnitude > bins[0].magnitude);
        assert!(bins[1].peak_magnitude > bins[1].magnitude);
        assert!(
            (bins[0].power - bins[1].power).abs() < 0.02,
            "values: {bins:?}"
        );
    }

    #[test]
    fn decimation_preserves_fft_bin_magnitude_and_source_metrics() {
        let sample_rate = 96_000_u32;
        let tone_hz = 10_007.812_5;
        let samples = (0..sample_rate / 5)
            .map(|index| {
                let phase =
                    2.0 * std::f32::consts::PI * tone_hz * index as f32 / sample_rate as f32;
                0.5 * phase.sin()
            })
            .collect::<Vec<_>>();
        let make_config = |decimation| VisualizerConfig {
            period: Duration::from_millis(16),
            bass_window_duration: Duration::from_millis(32),
            upper_window_duration: Duration::from_millis(32),
            crossover_frequency_hz: 1_000.0,
            min_frequency_hz: 100.0,
            max_frequency_hz: 20_000.0,
            transform: Transform::FourierCustom(vec![
                FrequencyBin::new(100.0, 500.0),
                FrequencyBin::new(10_007.0, 10_008.0),
            ]),
            decimation,
            drop_late_batches: true,
        };

        let mut source_rate = Visualizer::<1>::new(make_config(false)).unwrap();
        let mut decimated = Visualizer::<1>::new(make_config(true)).unwrap();
        let source_output = source_rate
            .process_batch(&mono_batch(samples.iter().copied()), 1, sample_rate)
            .pop()
            .unwrap();
        let decimated_output = decimated
            .process_batch(&mono_batch(samples), 1, sample_rate)
            .pop()
            .unwrap();
        let source_channel = &source_output.channels[0];
        let decimated_channel = &decimated_output.channels[0];

        assert_eq!(source_rate.analysis_sample_rate_hz, 96_000);
        assert_eq!(decimated.analysis_sample_rate_hz, 24_000);
        assert!(
            (source_channel.bins[1].magnitude - decimated_channel.bins[1].magnitude).abs() < 0.02,
            "source={} decimated={}",
            source_channel.bins[1].magnitude,
            decimated_channel.bins[1].magnitude
        );
        assert!(
            (source_channel.bins[1].power - decimated_channel.bins[1].power).abs() < 0.02,
            "source={} decimated={}",
            source_channel.bins[1].power,
            decimated_channel.bins[1].power
        );
        assert!(approx_eq(source_channel.peak, decimated_channel.peak, 1e-6));
        assert!(approx_eq(source_channel.rms, decimated_channel.rms, 1e-6));
        assert_eq!(source_output.sample_rate_hz, sample_rate);
        assert_eq!(decimated_output.sample_rate_hz, sample_rate);
    }
}
