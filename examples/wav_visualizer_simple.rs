use rodio::{Decoder, DeviceSinkBuilder, Player};
use rodio_tap::{TapReader, Visualizer, VisualizerConfig};
use std::error::Error;
use std::fs::File;
use std::io::{self, BufReader, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

const BAR_WIDTH: usize = 40;

fn main() -> Result<(), Box<dyn Error>> {
    let wav_paths = parse_cli_paths()?;

    let mut sink_handle = DeviceSinkBuilder::open_default_sink()?;
    sink_handle.log_on_drop(false);
    let player = Player::connect_new(sink_handle.mixer());

    let (queue_in, queue_out) = rodio::queue::queue(false);
    let (tap_reader, tap_adapter) = TapReader::<2>::new(queue_out);
    player.append(tap_adapter);

    for wav_path in &wav_paths {
        let file = File::open(wav_path)?;
        let decoder = Decoder::new(BufReader::new(file))?;
        queue_in.append(decoder);
    }

    let _terminal = TerminalGuard::new()?;

    let tap_for_reader = Arc::clone(&tap_reader);
    thread::spawn(move || {
        let config = VisualizerConfig {
            period: Duration::from_millis(33),
            ..Default::default()
        };
        let frequency_bins = config.frequency_bins();

        Visualizer::<2>::run_with_frame_reader(
            move || Some(Arc::clone(&tap_for_reader)),
            config,
            move |channels, sample_rate_hz| {
                let _ = render(&frequency_bins, channels, sample_rate_hz);
            },
        );
    });

    player.play();
    while !player.empty() {
        thread::sleep(Duration::from_millis(80));
    }
    thread::sleep(Duration::from_millis(100));

    Ok(())
}

fn render(
    frequency_bins: &[rodio_tap::FrequencyBin],
    channels: &[rodio_tap::ChannelSpectrum],
    sample_rate_hz: u32,
) -> io::Result<()> {
    let Some(channel) = channels.first() else {
        return Ok(());
    };

    let mut out = io::stdout().lock();
    write!(out, "\x1B[H")?;
    writeln!(
        out,
        "WAV visualizer (simple API)  |  sr: {} Hz  |  peak: {:.3}  |  rms: {:.3}{}",
        sample_rate_hz,
        channel.peak,
        channel.rms,
        if channels.len() > 1 {
            "  |  showing ch0"
        } else {
            ""
        }
    )?;

    for (idx, &magnitude) in channel.bins.iter().enumerate() {
        let bars = magnitude.round().clamp(0.0, BAR_WIDTH as f32) as usize;
        let bar = "#".repeat(bars);
        let freq = frequency_bins.get(idx);
        writeln!(
            out,
            "{:>5.0} - {:>5.0} Hz | {:<width$}",
            freq.map(|f| f.hz_lo).unwrap_or(0.0),
            freq.map(|f| f.hz_hi).unwrap_or(0.0),
            bar,
            width = BAR_WIDTH
        )?;
    }

    out.flush()
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

fn parse_cli_paths() -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut args = std::env::args_os();
    let program = args.next().unwrap_or_default();
    let paths: Vec<PathBuf> = args.map(PathBuf::from).collect();

    if paths.is_empty() {
        return Err(format!(
            "Usage: {:?} <path/to/file1.wav> [path/to/file2.wav ...]",
            program
        )
        .into());
    }

    for path in &paths {
        if !path.exists() {
            return Err(format!("Input file does not exist: {}", path.display()).into());
        }
    }

    Ok(paths)
}
