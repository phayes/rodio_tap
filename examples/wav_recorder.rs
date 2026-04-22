use rodio::{Decoder, DeviceSinkBuilder, Player};
use rodio_tap::{FrameReader, FrameReaderConfig, TapReader};
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::io::BufWriter;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const WRITE_FRAMES_PER_BATCH: u32 = 16_384;
const MAX_INPUT_CHANNELS: usize = 64;
const OUTPUT_SAMPLE_RATE_HZ: u32 = 48_000;

fn main() -> Result<(), Box<dyn Error>> {
    let wav_paths = parse_cli_paths()?;
    let output_path = temp_output_path();
    println!("Recording to {}", output_path.display());

    let mut sink_handle = DeviceSinkBuilder::open_default_sink()?;
    sink_handle.log_on_drop(false);
    let player = Player::connect_new(sink_handle.mixer());

    // Queue lets us chain any number of input decoders into one continuous source stream.
    let (queue_in, queue_out) = rodio::queue::queue(false);

    // Tap the mixed queue output so we can both hear it and inspect frames.
    let (tap_reader, tap_adapter) = TapReader::<MAX_INPUT_CHANNELS>::new(queue_out);
    player.append(tap_adapter);

    for wav_path in &wav_paths {
        let file = File::open(wav_path)?;
        let decoder = Decoder::new(BufReader::new(file))?;
        queue_in.append(decoder);
    }

    let tap_for_reader = Arc::clone(&tap_reader);
    let output_path_for_reader = output_path.clone();

    // Run recording in a worker thread because `FrameReader::run` is a blocking loop.
    thread::spawn(move || {
        let mut recorder = IncrementalRecorder::new(output_path_for_reader);
        let config = FrameReaderConfig {
            // Keep the frame reader callback size aligned with our write chunk size.
            frames_per_batch: Some(WRITE_FRAMES_PER_BATCH),
            time_per_batch: None,
            ..Default::default()
        };
        let mut frame_reader =
            FrameReader::<MAX_INPUT_CHANNELS>::new_with_config(config, move || {
                Some(Arc::clone(&tap_for_reader))
            });

        frame_reader.run(move |batch, channels, sample_rate_hz| {
            if batch.is_empty() || channels == 0 {
                return;
            }
            // Convert and write incrementally each callback
            recorder.process_batch(batch, channels, sample_rate_hz);
        });
    });

    player.play();
    wait_for_playback_end(&player);

    // Give the reader thread a final moment to drain buffered packets.
    thread::sleep(Duration::from_millis(150));

    Ok(())
}

/// Small state holder for incremental WAV recording.
///
/// `IncrementalRecorder` is used by the `FrameReader` callback thread:
/// - lazily opens a mono `hound::WavWriter` on the first valid batch,
/// - converts each incoming batch to fixed-rate mono (via helper resampler),
/// - writes converted samples immediately so memory usage stays bounded.
struct IncrementalRecorder {
    output_path: PathBuf,
    writer: Option<hound::WavWriter<BufWriter<File>>>,
}

impl IncrementalRecorder {
    fn new(output_path: PathBuf) -> Self {
        Self {
            output_path,
            writer: None,
        }
    }

    fn process_batch(
        &mut self,
        batch: &[arrayvec::ArrayVec<f32, MAX_INPUT_CHANNELS>],
        channels: usize,
        sample_rate_hz: u32,
    ) {
        if batch.is_empty() || channels == 0 || sample_rate_hz == 0 {
            return;
        }

        if self.writer.is_none() {
            self.open_writer();
        }

        // This is intentionally a naive sample-rate conversion for the example:
        // linear interpolation when upsampling, and sample dropping when downsampling.
        // Proper resampling should include anti-alias low-pass filtering.
        let out_batch = resample_batch_to_mono_fixed_rate(batch, channels, sample_rate_hz);
        // Write each converted sample immediately so memory usage stays bounded.
        for sample in out_batch {
            self.write_sample(sample);
        }
    }

    fn open_writer(&mut self) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: OUTPUT_SAMPLE_RATE_HZ,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let writer =
            hound::WavWriter::create(&self.output_path, spec).expect("failed to create WAV file");
        self.writer = Some(writer);
    }

    fn write_sample(&mut self, sample: f32) {
        let writer = self
            .writer
            .as_mut()
            .expect("attempted to write before writer was initialized");

        writer
            .write_sample(sample)
            .expect("failed to write WAV sample");
    }
}

// Simple fold-down mono mixer: sum all channels.
//
// A production-ready mixer would apply scaling/weighting to avoid clipping.
fn sum_to_mono(frame: &arrayvec::ArrayVec<f32, MAX_INPUT_CHANNELS>, channels: usize) -> f32 {
    frame.iter().take(channels).copied().sum::<f32>()
}

// Resample the batch to a fixed output sample rate in mono.
//
// NOTE: This is an intentionally a naive resampler.
// - Upsampling path: linear interpolation between neighboring samples.
// - Downsampling path: sample dropping (stride-based decimation).
//
// A production-ready sample-rate conversion should instead:
// 1) use a low-pass anti-alias filter with cutoff <= min(in_sr, out_sr)/2,
// 2) apply a polyphase FIR (or windowed-sinc) resampler for fractional-ratio support,
// 3) carry filter state across batches to avoid boundary artifacts/clicks,
// 4) optionally use dithering/noise-shaping depending on final format.
//
// Keeping this simple makes the example easier to follow, but this is not production DSP.
fn resample_batch_to_mono_fixed_rate(
    batch: &[arrayvec::ArrayVec<f32, MAX_INPUT_CHANNELS>],
    channels: usize,
    input_sample_rate_hz: u32,
) -> Vec<f32> {
    // First normalize channel layout: any N-channel input -> mono stream.
    let mono_in: Vec<f32> = batch.iter().map(|f| sum_to_mono(f, channels)).collect();
    if mono_in.is_empty() {
        return Vec::new();
    }

    if input_sample_rate_hz == OUTPUT_SAMPLE_RATE_HZ {
        return mono_in;
    }

    if input_sample_rate_hz < OUTPUT_SAMPLE_RATE_HZ {
        // Upsample via linear interpolation:
        // map evenly spaced output points into input index-space.
        let output_len = ((mono_in.len() as u128 * OUTPUT_SAMPLE_RATE_HZ as u128
            + (input_sample_rate_hz as u128 / 2))
            / input_sample_rate_hz as u128)
            .max(1) as usize;

        if output_len == 1 {
            return vec![mono_in[0]];
        }

        let mut out = Vec::with_capacity(output_len);
        // `scale` maps output sample indices to floating-point input positions.
        let scale = (mono_in.len() - 1) as f64 / (output_len - 1) as f64;
        for i in 0..output_len {
            let pos = i as f64 * scale;
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(mono_in.len() - 1);
            let frac = (pos - lo as f64) as f32;
            // Blend neighboring samples by fractional distance.
            out.push(mono_in[lo] + (mono_in[hi] - mono_in[lo]) * frac);
        }
        return out;
    }

    // Downsample by striding through the input and dropping intermediate samples.
    let mut out = Vec::new();
    let step = input_sample_rate_hz as f64 / OUTPUT_SAMPLE_RATE_HZ as f64;
    let mut pos = 0.0_f64;
    while (pos as usize) < mono_in.len() {
        out.push(mono_in[pos as usize]);
        pos += step;
    }
    out
}

fn wait_for_playback_end(player: &Player) {
    while !player.empty() {
        thread::sleep(Duration::from_millis(80));
    }
    thread::sleep(Duration::from_millis(100));
}

fn temp_output_path() -> PathBuf {
    let ts_millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let mut path = std::env::temp_dir();
    path.push(format!("rodio_tap_wav_recorder_{ts_millis}.wav"));
    path
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
