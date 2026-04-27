use arc_swap::ArcSwapOption;
use realfft::num_complex::Complex32;
use realfft::{RealFftPlanner, RealToComplex};
use rodio::{Decoder, DeviceSinkBuilder, Player};
use rodio_tap::{FrameReader, FrameReaderConfig, TapReader};
use std::collections::VecDeque;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufReader, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

const FFT_SIZE: usize = 2048;
const NUM_BANDS: usize = 28;
const BAR_WIDTH: usize = 48;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TapMode {
    /// Build one persistent tap around a rodio queue source (default).
    SingleTap,
    /// Build one tap per track and swap active tap as tracks start.
    OneTapPerTrack,
}

#[derive(Debug)]
struct CliArgs {
    mode: TapMode,
    wav_paths: Vec<PathBuf>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_cli_args()?;

    let mut sink_handle = DeviceSinkBuilder::open_default_sink()?;
    sink_handle.log_on_drop(false);
    let player = Player::connect_new(sink_handle.mixer());

    // Build one of two example pipelines:
    //
    // 1) --single_tap (default / preferred):
    //    - queue all tracks into one rodio queue source
    //    - wrap queue output in one TapAdapter/TapReader pair
    //    - append that single tap-adapter once to the player
    //
    // 2) --one_tap_per_track:
    //    - create one TapAdapter/TapReader pair per track
    //    - append each tap-adapter to the same player queue
    //    - use ArcSwap to point FrameReader at the currently active tap
    let tap_fn: Arc<dyn Fn() -> Option<Arc<TapReader<2>>> + Send + Sync> = match args.mode {
        TapMode::SingleTap => {
            let (queue_in, queue_out) = rodio::queue::queue(false);
            let (tap_reader, tap_adapter) = TapReader::<2>::new(queue_out);
            player.append(tap_adapter);

            for wav_path in &args.wav_paths {
                let file = File::open(wav_path)?;
                let decoder = Decoder::new(BufReader::new(file))?;
                queue_in.append(decoder);
            }

            let tap_reader = Arc::clone(&tap_reader);
            Arc::new(move || Some(Arc::clone(&tap_reader)))
        }
        TapMode::OneTapPerTrack => {
            let current_tap = Arc::new(ArcSwapOption::<TapReader<2>>::empty());
            for wav_path in &args.wav_paths {
                let file = File::open(wav_path)?;
                let decoder = Decoder::new(BufReader::new(file))?;
                let (_tap_reader, tap_adapter) =
                    TapReader::<2>::new_with_publish_target(&current_tap, decoder);
                player.append(tap_adapter);
            }

            let current_tap = Arc::clone(&current_tap);
            Arc::new(move || current_tap.load_full())
        }
    };

    player.play();

    let _terminal = TerminalGuard::new()?;
    let visualizer = Arc::new(Mutex::new(Visualizer::new(FFT_SIZE, NUM_BANDS)));
    let viz_for_cb = Arc::clone(&visualizer);
    let tap_fn_for_reader = Arc::clone(&tap_fn);

    thread::spawn(move || {
        let config = FrameReaderConfig {
            time_per_batch: Some(Duration::from_millis(33)),
            ..Default::default()
        };
        let mut frame_reader =
            FrameReader::<2>::new_with_config(config, move || tap_fn_for_reader());

        frame_reader.run(move |batch, channels, sample_rate_hz| {
            if channels == 0 || batch.is_empty() {
                return;
            }

            let mut mono = Vec::with_capacity(batch.len());
            for frame in batch {
                let sum = frame.iter().copied().sum::<f32>();
                mono.push(sum / channels as f32);
            }

            if let Ok(mut viz) = viz_for_cb.lock() {
                let _ = viz.consume_and_render(&mono, sample_rate_hz);
            }
        });
    });

    wait_for_playback_end(&player);

    Ok(())
}

fn wait_for_playback_end(player: &Player) {
    while !player.empty() {
        thread::sleep(Duration::from_millis(80));
    }
    thread::sleep(Duration::from_millis(100));
}

fn parse_cli_args() -> Result<CliArgs, Box<dyn Error>> {
    let mut args = std::env::args_os();
    let program = args.next().unwrap_or_default();
    let mut mode = TapMode::SingleTap;
    let mut paths = Vec::new();

    for arg in args {
        let s = arg.to_string_lossy();
        match s.as_ref() {
            "--single_tap" => mode = TapMode::SingleTap,
            "--one_tap_per_track" => mode = TapMode::OneTapPerTrack,
            flag if flag.starts_with("--") => {
                return Err(format!("Unknown flag: {}\n\n{}", flag, usage_text(&program)).into());
            }
            _ => paths.push(PathBuf::from(arg)),
        }
    }

    if paths.is_empty() {
        return Err(usage_text(&program).into());
    }

    for path in &paths {
        if !path.exists() {
            return Err(format!("Input file does not exist: {}", path.display()).into());
        }
    }

    Ok(CliArgs {
        mode,
        wav_paths: paths,
    })
}

fn usage_text(program: &std::ffi::OsStr) -> String {
    format!(
        "Usage: {:?} [--single_tap|--one_tap_per_track] <path/to/file1.wav> [path/to/file2.wav ...]\n\n\
Modes:\n\
  --single_tap        (default) queue tracks into one tapped source\n\
  --one_tap_per_track create one tap per queued track and swap active tap\n\n\
Example:\n\
  cargo run --example wav_visualizer -- --single_tap examples/example.wav examples/sweep_5s_22050_mono.wav",
        program
    )
}

struct TerminalGuard;

impl TerminalGuard {
    fn new() -> io::Result<Self> {
        let mut out = io::stdout().lock();
        write!(out, "\x1B[2J\x1B[H\x1B[?25l")?;
        out.flush()?;
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = writeln!(io::stdout(), "\x1B[?25h");
    }
}

struct Visualizer {
    analyzer: SpectrumAnalyzer,
    renderer: SpectrumRenderer,
}

impl Visualizer {
    fn new(n_fft: usize, num_bands: usize) -> Self {
        Self {
            analyzer: SpectrumAnalyzer::new(n_fft, num_bands),
            renderer: SpectrumRenderer::new(),
        }
    }

    fn consume_and_render(&mut self, mono: &[f32], sample_rate_hz: u32) -> io::Result<()> {
        if let Some(frame) = self.analyzer.process(mono, sample_rate_hz) {
            self.renderer.render(&frame)?;
        }
        Ok(())
    }
}

struct SpectrumFrame {
    sample_rate_hz: u32,
    rms: f32,
    peak: f32,
    bands: Vec<BandReading>,
}

struct BandReading {
    hz_lo: f32,
    hz_hi: f32,
    magnitude: f32,
}

struct SpectrumAnalyzer {
    n_fft: usize,
    num_bands: usize,
    history: VecDeque<f32>,
    fft_planner: RealFftPlanner<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
    fft_input: Vec<f32>,
    fft_spectrum: Vec<Complex32>,
}

impl SpectrumAnalyzer {
    fn new(n_fft: usize, num_bands: usize) -> Self {
        let mut fft_planner = RealFftPlanner::<f32>::new();
        let fft = fft_planner.plan_fft_forward(n_fft);
        let fft_input = fft.make_input_vec();
        let fft_spectrum = fft.make_output_vec();
        Self {
            n_fft,
            num_bands,
            history: VecDeque::with_capacity(n_fft),
            fft_planner,
            fft,
            fft_input,
            fft_spectrum,
        }
    }

    fn process(&mut self, mono: &[f32], sample_rate_hz: u32) -> Option<SpectrumFrame> {
        if mono.is_empty() || sample_rate_hz == 0 {
            return None;
        }

        for sample in mono {
            if self.history.len() == self.n_fft {
                self.history.pop_front();
            }
            self.history.push_back(*sample);
        }

        if self.history.len() < self.n_fft {
            return None;
        }

        self.ensure_plan();
        self.fill_fft_input_hann();
        self.fft
            .process(&mut self.fft_input, &mut self.fft_spectrum)
            .expect("realfft input/output lengths must match planned FFT length");

        let mut bands = Vec::with_capacity(self.num_bands);
        let nyquist = sample_rate_hz as f32 * 0.5;
        let min_hz = 30.0_f32;
        let min_bin = 1_usize;
        let max_bin = self.n_fft / 2;
        let hz_to_bin = |hz: f32| -> usize {
            let clamped = hz.clamp(0.0, nyquist);
            (((clamped * self.n_fft as f32) / sample_rate_hz as f32).floor() as usize)
                .clamp(min_bin, max_bin)
        };

        for i in 0..self.num_bands {
            let start_t = i as f32 / self.num_bands as f32;
            let end_t = (i + 1) as f32 / self.num_bands as f32;
            let hz_lo = min_hz * (nyquist / min_hz).powf(start_t);
            let hz_hi = min_hz * (nyquist / min_hz).powf(end_t);
            let bin_lo = hz_to_bin(hz_lo);
            let mut bin_hi = hz_to_bin(hz_hi);
            if bin_hi <= bin_lo {
                bin_hi = (bin_lo + 1).min(max_bin);
            }

            let mut sum = 0.0_f32;
            for k in bin_lo..=bin_hi {
                let c = self.fft_spectrum[k];
                sum += (c.re * c.re + c.im * c.im).sqrt();
            }
            let magnitude = sum / (bin_hi - bin_lo + 1) as f32;
            bands.push(BandReading {
                hz_lo,
                hz_hi,
                magnitude,
            });
        }

        let mut sum_sq = 0.0_f32;
        let mut peak = 0.0_f32;
        for sample in mono {
            let abs = sample.abs();
            peak = peak.max(abs);
            sum_sq += sample * sample;
        }

        Some(SpectrumFrame {
            sample_rate_hz,
            rms: (sum_sq / mono.len() as f32).sqrt(),
            peak,
            bands,
        })
    }

    fn ensure_plan(&mut self) {
        if self.fft.len() != self.n_fft {
            self.fft = self.fft_planner.plan_fft_forward(self.n_fft);
            self.fft_input = self.fft.make_input_vec();
            self.fft_spectrum = self.fft.make_output_vec();
        }
    }

    fn fill_fft_input_hann(&mut self) {
        self.fft_input.fill(0.0);
        let len = self.history.len();
        for (i, sample) in self.history.iter().enumerate() {
            let window = if len > 1 {
                let n = i as f32;
                let denom = (len - 1) as f32;
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * n / denom).cos()
            } else {
                1.0
            };
            self.fft_input[i] = sample * window;
        }
    }
}

struct SpectrumRenderer {
    level_floor: f32,
}

impl SpectrumRenderer {
    fn new() -> Self {
        Self { level_floor: 1e-3 }
    }

    fn render(&mut self, frame: &SpectrumFrame) -> io::Result<()> {
        let mut out = io::stdout().lock();
        let observed_peak = frame
            .bands
            .iter()
            .map(|b| b.magnitude)
            .fold(self.level_floor, f32::max);
        self.level_floor = (self.level_floor * 0.94).max(observed_peak).max(1e-3);

        write!(out, "\x1B[H")?;
        writeln!(
            out,
            "WAV FFT terminal spectrum  |  sr: {} Hz  |  peak: {:.3}  |  rms: {:.3}",
            frame.sample_rate_hz, frame.peak, frame.rms
        )?;

        for band in &frame.bands {
            let normalized = (band.magnitude / self.level_floor).clamp(0.0, 1.0);
            let bars = (normalized * BAR_WIDTH as f32).round() as usize;
            let bar = "#".repeat(bars);
            writeln!(
                out,
                "{:>5.0} - {:>5.0} Hz | {:<width$}",
                band.hz_lo,
                band.hz_hi,
                bar,
                width = BAR_WIDTH
            )?;
        }

        out.flush()
    }
}
