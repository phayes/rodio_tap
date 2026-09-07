use arc_swap::ArcSwapOption;
use rodio::{Decoder, DeviceSinkBuilder, Player};
use rodio_tap::{
    ChannelSpectrum, FrequencyBin, TapReader, Transform, Visualizer, VisualizerConfig,
};
use std::error::Error;
use std::fs::File;
use std::io::{self, BufReader, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

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
    let renderer = Arc::new(Mutex::new(SpectrumRenderer::new()));
    let renderer_for_callback = Arc::clone(&renderer);
    let tap_fn_for_reader = Arc::clone(&tap_fn);

    thread::spawn(move || {
        let config = VisualizerConfig {
            emit_period: Duration::from_millis(33),
            bass_window_duration: Duration::from_millis(170),
            upper_window_duration: Duration::from_millis(33),
            crossover_frequency_hz: 250.0,
            transform: Transform::FourierLog(NUM_BANDS),
            ..Default::default()
        };
        let bins = config.frequency_bins();
        Visualizer::<2>::run_with_frame_reader(
            move || tap_fn_for_reader(),
            config,
            move |channels, sample_rate_hz| {
                if let (Some(channel), Ok(mut renderer)) =
                    (channels.first(), renderer_for_callback.lock())
                {
                    let _ = renderer.render(&bins, channel, sample_rate_hz);
                }
            },
        );
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

struct SpectrumRenderer {
    level_floor: f32,
}

impl SpectrumRenderer {
    fn new() -> Self {
        Self { level_floor: 1e-3 }
    }

    fn render(
        &mut self,
        bins: &[FrequencyBin],
        channel: &ChannelSpectrum,
        sample_rate_hz: u32,
    ) -> io::Result<()> {
        let mut out = io::stdout().lock();
        let observed_peak = channel
            .bins
            .iter()
            .map(|value| value.magnitude)
            .fold(self.level_floor, f32::max);
        self.level_floor = (self.level_floor * 0.94).max(observed_peak).max(1e-3);

        write!(out, "\x1B[H")?;
        writeln!(
            out,
            "WAV multi-resolution spectrum  |  sr: {} Hz  |  peak: {:.3}  |  rms: {:.3}",
            sample_rate_hz, channel.peak, channel.rms
        )?;

        for (band, value) in bins.iter().zip(&channel.bins) {
            let normalized = (value.magnitude / self.level_floor).clamp(0.0, 1.0);
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
