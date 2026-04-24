use rodio::{Decoder, DeviceSinkBuilder, Player, Source};
use rodio_tap::{FrameReader, FrameReaderConfig, TapReader};
use std::error::Error;
use std::fs::File;
use std::io::{self, BufReader, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const FRAMES_PER_BATCH: u32 = 64;
const SLEEP_BIAS: f32 = 0.5;
const MIN_SLEEP_US: u64 = 5;
const MAX_CHANNELS: usize = 64;

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_cli_args()?;

    let mut sink_handle = DeviceSinkBuilder::open_default_sink()?;
    sink_handle.log_on_drop(false);
    let player = Player::connect_new(sink_handle.mixer());

    let (queue_in, queue_out) = rodio::queue::queue(false);
    let (tap_reader, tap_adapter) = TapReader::<MAX_CHANNELS>::new(queue_out);
    player.append(tap_adapter);

    if args.loop_playback {
        if args.wav_paths.len() != 1 {
            return Err(
                "Loop mode currently supports exactly one input file: use --loop <file.wav>".into(),
            );
        }
        let file = File::open(&args.wav_paths[0])?;
        let decoder = Decoder::new(BufReader::new(file))?;
        queue_in.append(decoder.repeat_infinite());
    } else {
        for wav_path in &args.wav_paths {
            let file = File::open(wav_path)?;
            let decoder = Decoder::new(BufReader::new(file))?;
            queue_in.append(decoder);
        }
    }

    // Keep terminal output simple: clear once, then redraw report in place.
    print!("\x1B[2J\x1B[H");
    println!("waiting for enough samples to measure average latency");
    io::stdout().flush()?;

    let tap_for_reader = Arc::clone(&tap_reader);
    thread::spawn(move || {
        let config = FrameReaderConfig {
            frames_per_batch: Some(FRAMES_PER_BATCH),
            time_per_batch: None,
            sleep_bias: SLEEP_BIAS,
            min_sleep: Duration::from_micros(MIN_SLEEP_US),
            ..Default::default()
        };

        let mut reader = FrameReader::<MAX_CHANNELS>::new_with_config(config, move || {
            Some(Arc::clone(&tap_for_reader))
        });
        let mut meter = CallbackLatencyMeter::new(args.window_seconds);

        reader.run(move |batch, channels, sample_rate_hz| {
            if batch.is_empty() || sample_rate_hz == 0 {
                return;
            }
            meter.on_batch(batch.len() as u32, channels, sample_rate_hz);
        });
    });

    player.play();
    if args.loop_playback {
        loop {
            thread::sleep(Duration::from_secs(1));
        }
    } else {
        wait_for_playback_end(&player);
        println!("\nPlayback finished.");
    }
    Ok(())
}

struct CallbackLatencyMeter {
    window_seconds: f64,
    window_target_us: f64,
    last_batch_at: Option<Instant>,
    intervals_total_us: f64,
    intervals_count: u64,
}

impl CallbackLatencyMeter {
    fn new(window_seconds: f64) -> Self {
        let safe_window_seconds = if window_seconds.is_finite() && window_seconds > 0.0 {
            window_seconds
        } else {
            1.0
        };
        Self {
            window_seconds: safe_window_seconds,
            window_target_us: safe_window_seconds * 1_000_000.0,
            last_batch_at: None,
            intervals_total_us: 0.0,
            intervals_count: 0,
        }
    }

    fn on_batch(&mut self, frames_in_batch: u32, channels: usize, sample_rate_hz: u32) {
        let now = Instant::now();

        if let Some(last) = self.last_batch_at {
            let dt_us = now.duration_since(last).as_secs_f64() * 1_000_000.0;
            self.intervals_total_us += dt_us;
            self.intervals_count += 1;
        }
        self.last_batch_at = Some(now);

        // Wait until we have a full configured window of observed interval data.
        if self.intervals_total_us < self.window_target_us {
            return;
        }

        let window_interval_count = self.intervals_count;
        let avg_interval_us = if window_interval_count == 0 {
            0.0
        } else {
            self.intervals_total_us / window_interval_count as f64
        };
        let theoretical_min_us = (frames_in_batch as f64 * 1_000_000.0) / sample_rate_hz as f64;

        let overhead_ns = ((avg_interval_us - theoretical_min_us).max(0.0)) * 1_000.0;

        let mut out = io::stdout().lock();
        let _ = write!(out, "\x1B[H\x1B[J");
        let _ = writeln!(
            out,
            "WAV low-latency timing report ({:.3}s windows)\n  Avg interval    : {:>9.1} us\n  Theoretical min : {:>9.1} us\n  Overhead        : {:>9.0} ns\n  Intervals       : {}\n  Batch config    : {} frames, {} ch, {} Hz",
            self.window_seconds,
            avg_interval_us,
            theoretical_min_us,
            overhead_ns,
            window_interval_count,
            frames_in_batch,
            channels,
            sample_rate_hz
        );
        let _ = out.flush();

        // Reset rolling stats so the next print is a fresh 1-second chunk.
        self.intervals_total_us = 0.0;
        self.intervals_count = 0;
    }
}

fn wait_for_playback_end(player: &Player) {
    while !player.empty() {
        thread::sleep(Duration::from_millis(80));
    }
    thread::sleep(Duration::from_millis(100));
}

#[derive(Debug)]
struct CliArgs {
    wav_paths: Vec<PathBuf>,
    window_seconds: f64,
    loop_playback: bool,
}

fn parse_cli_args() -> Result<CliArgs, Box<dyn Error>> {
    let mut args = std::env::args_os();
    let program = args.next().unwrap_or_default();
    let mut paths = Vec::new();
    let mut window_seconds = 1.0_f64;
    let mut loop_playback = false;

    for arg in args {
        let arg_str = arg.to_string_lossy();
        if let Some(v) = arg_str.strip_prefix("--window=") {
            window_seconds = v
                .parse::<f64>()
                .map_err(|_| format!("Invalid --window value: {v}. Expected a positive number."))?;
            if !window_seconds.is_finite() || window_seconds <= 0.0 {
                return Err(
                    format!("Invalid --window value: {v}. Expected a positive number.").into(),
                );
            }
            continue;
        }
        if arg_str == "--loop" {
            loop_playback = true;
            continue;
        }
        if arg_str.starts_with("--") {
            return Err(format!("Unknown flag: {}\n\n{}", arg_str, usage_text(&program)).into());
        }
        paths.push(PathBuf::from(arg));
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
        wav_paths: paths,
        window_seconds,
        loop_playback,
    })
}

fn usage_text(program: &std::ffi::OsStr) -> String {
    format!(
        "Usage: {:?} [--window=<num_seconds>] [--loop] <path/to/file1.wav> [path/to/file2.wav ...]\n\n\
Flags:\n\
  --window=<num_seconds>  Reporting window size in seconds (default: 1)\n\
  --loop                  Infinitely loop playback (currently supports one input file)\n",
        program
    )
}
