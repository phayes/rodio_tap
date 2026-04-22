use crate::{AsyncFrameReader, FrameReaderConfig, TapReader};
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
pub struct VisualizerConfig {
    pub period: Duration,
    pub num_bands: usize,
    pub min_frequency_hz: f32,
    pub max_frequency_hz: f32,
    pub normalize_by_fft_size: bool,
    pub emit_before_fft_window_full: bool,
}

impl Default for VisualizerConfig {
    fn default() -> Self {
        Self {
            period: Duration::from_millis(33),
            num_bands: 28,
            min_frequency_hz: LOW_FREQUENCY_HUMAN,
            max_frequency_hz: TOP_FREQUENCY_HUMAN,
            normalize_by_fft_size: false,
            emit_before_fft_window_full: false,
        }
    }
}

impl VisualizerConfig {
    fn validate(&self) {
        assert!(
            self.period.as_nanos() > 0,
            "VisualizerConfig.period must be > 0"
        );
        assert!(self.num_bands > 0, "VisualizerConfig.num_bands must be > 0");
        assert!(
            self.min_frequency_hz > 0.0,
            "VisualizerConfig.min_frequency_hz must be > 0"
        );
        assert!(
            self.max_frequency_hz > self.min_frequency_hz,
            "VisualizerConfig.max_frequency_hz must be > min_frequency_hz"
        );
    }

    pub fn frequency_bins(&self) -> Vec<FrequencyBin> {
        let edges = compute_log_edges(
            self.min_frequency_hz,
            self.max_frequency_hz,
            self.num_bands,
        );
        edges_to_frequency_bins(&edges)
    }
}

#[derive(Debug, Clone)]
pub struct FrequencyBin {
    pub hz_lo: f32,
    pub hz_hi: f32,
}

#[derive(Debug, Clone)]
pub struct ChannelSpectrum {
    pub peak: f32,
    pub rms: f32,
    pub bins: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct VisualizerFrame {
    pub sample_rate_hz: u32,
    pub channels: Vec<ChannelSpectrum>,
}

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
    pub fn new(config: VisualizerConfig) -> Self {
        assert!(C > 0, "Visualizer requires C > 0");
        config.validate();

        let fft_len = 1usize;
        let mut fft_planner = FftPlanner::new();
        let fft = fft_planner.plan_fft_forward(fft_len);

        Self {
            frequency_bins: config.frequency_bins(),
            histories: (0..C).map(|_| VecDeque::new()).collect(),
            fft_planner,
            fft,
            fft_input: vec![Complex32::new(0.0, 0.0); fft_len],
            fft_spectrum: vec![Complex32::new(0.0, 0.0); fft_len],
            fft_len,
            last_sample_rate_hz: None,
            config,
        }
    }

    pub fn config(&self) -> &VisualizerConfig {
        &self.config
    }

    pub fn frequency_bins(&self) -> &[FrequencyBin] {
        &self.frequency_bins
    }

    pub async fn run_with_async_frame_reader<G, F>(
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
        let mut visualizer = Visualizer::<C>::new(config);

        reader
            .run(move |batch, channels, sample_rate_hz| {
            if let Some(frame) = visualizer.process_batch(batch, channels, sample_rate_hz) {
                callback(&frame.channels, frame.sample_rate_hz);
            }
            })
            .await
    }

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
            .max_frequency_hz
            .min(sample_rate_hz as f32 * 0.5);
        if effective_max_hz <= 0.0 {
            magnitudes.fill(0.0);
        }

        magnitudes
    }

    fn compute_bin_magnitudes(&self, sample_rate_hz: u32, magnitudes: &[f32]) -> Vec<f32> {
        let mut bins = Vec::with_capacity(self.frequency_bins.len());
        let nyquist = sample_rate_hz as f32 * 0.5;
        let effective_max = self.config.max_frequency_hz.min(nyquist);
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

fn derive_fft_len(period: Duration, sample_rate_hz: u32) -> usize {
    let frames =
        ((sample_rate_hz as u128 * period.as_nanos() + 500_000_000) / 1_000_000_000).max(1);

    // Keep analysis size power-of-two for FFT efficiency.
    let target = usize::try_from(frames).unwrap_or(usize::MAX / 2).max(1);
    target.next_power_of_two().max(1)
}

fn compute_log_edges(min_hz: f32, max_hz: f32, num_bands: usize) -> Vec<f32> {
    let mut edges = Vec::with_capacity(num_bands + 1);
    let ratio = max_hz / min_hz;
    for idx in 0..=num_bands {
        let t = idx as f32 / num_bands as f32;
        edges.push(min_hz * ratio.powf(t));
    }
    edges
}

fn edges_to_frequency_bins(edges: &[f32]) -> Vec<FrequencyBin> {
    edges
        .windows(2)
        .map(|range| FrequencyBin {
            hz_lo: range[0],
            hz_hi: range[1],
        })
        .collect()
}

fn hz_to_bin(hz: f32, fft_len: usize, sample_rate_hz: u32) -> usize {
    if sample_rate_hz == 0 {
        return 0;
    }
    (((hz * fft_len as f32) / sample_rate_hz as f32).floor() as usize).min(fft_len / 2)
}

fn hann_window(index: usize, len: usize) -> f32 {
    if len <= 1 {
        1.0
    } else {
        let n = index as f32;
        let denom = (len - 1) as f32;
        0.5 - 0.5 * (2.0 * std::f32::consts::PI * n / denom).cos()
    }
}
