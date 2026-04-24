# rodio_tap

[![crates.io](https://img.shields.io/crates/v/rodio_tap.svg)](https://crates.io/crates/rodio_tap)
[![docs.rs](https://docs.rs/rodio_tap/badge.svg)](https://docs.rs/rodio_tap)

`rodio_tap` taps `rodio::Source` audio while still passing the source through to playback.

Use it when you want to analyze, visualize, meter, or record playback data in real time.

https://github.com/user-attachments/assets/54d66615-4ef3-4876-af7b-6dc5886b64ff

## What it provides

- `TapReader` + `TapAdapter`: low-level packet ring-buffer access.
- `FrameReader`: synchronous high-level reader that yields frame batches.
- `AsyncFrameReader` (feature `async`): async high-level reader for Tokio runtimes.
- `Visualizer` (feature `visualizer`): callback-driven FFT bins + peak/rms per channel.

## Installation

In your `Cargo.toml`:

```toml
[dependencies]
rodio_tap = "0.1.1"
```

Enable the visualizer module:

```toml
[dependencies]
rodio_tap = { version = "0.1.1", features = ["visualizer"] }
```

Enable async support if tokio support is needed:

```toml
[dependencies]
rodio_tap = { version = "0.1.1", features = ["async"] }
```

## Quick start

```rust
use std::sync::Arc;
use std::thread;
use rodio::Decoder;
use rodio_tap::{FrameReader, TapReader};

fn run<S>(rodio_source: S)
where
     S: rodio::Source + Send + 'static,
     S::Item: cpal::Sample + Send + 'static,
     f32: cpal::FromSample<S::Item>,
{

// Create a stereo (2-channel) tap reader from any rodio source.
let (tap_reader, tap_adapter) = TapReader::<2>::new(rodio_source);

// Send `tap_adapter` into your rodio playback pipeline.
let _ = tap_adapter;

let tap_for_reader = Arc::clone(&tap_reader);
thread::spawn(move || {
    // Create a stereo (2 channel) frame reader
    let mut reader = FrameReader::<2>::new(move || Some(Arc::clone(&tap_for_reader)));
    reader.run(|batch, channels, sample_rate_hz| {
        let frames = batch.len();
        println!("{} frames @ {} Hz", frames, sample_rate_hz);

        for frame in batch {
            let _ = frame;
            // Process one interleaved frame.
        }
    });
});
}
```

### Multiple tracks:

```rust
use std::sync::Arc;
use rodio::queue;
use rodio_tap::TapReader;

// One queue source for all tracks.
let (queue_in, queue_out) = queue::queue(false);

// One persistent tap around the queue output.
let (tap_reader, tap_adapter) = TapReader::<2>::new(queue_out);
let _ = (tap_reader, tap_adapter);

// Append each decoder to queue_in.append(decoder).
```

## Async reader

With the `async` feature enabled:

```rust
use std::sync::Arc;
use rodio_tap::AsyncFrameReader;

async fn run_reader(tap: Arc<rodio_tap::TapReader<2>>) {
    let mut reader = AsyncFrameReader::<2>::new(move || Some(Arc::clone(&tap)));
    reader
        .run(|batch, channels, sample_rate_hz| {
            let frames = batch.len();
            println!("{} frames @ {} Hz", frames, sample_rate_hz);
        })
        .await;
}
```


## Visualizer

`Visualizer` (feature `visualizer`) provides an abstract for building music visualizers. 

```rust
use rodio::source::SineWave;
use rodio::{DeviceSinkBuilder, Player, Source};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use rodio_tap::{Visualizer, VisualizerConfig, TapReader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a simple test tone and loop it forever.
    let tone = SineWave::new(440.0).amplify(0.2).repeat_infinite();

    // Tap the source before sending it to playback.
    let (tap_reader, tap_adapter) = TapReader::<2>::new(tone);

    // Play audio through rodio.
    let mut sink = DeviceSinkBuilder::open_default_sink()?;
    sink.log_on_drop(false);
    let player = Player::connect_new(sink.mixer());
    player.append(tap_adapter);
    player.play();

    // Visualizer callback runs forever, so run it on a worker thread.
    let tap_for_visualizer = Arc::clone(&tap_reader);
    thread::spawn(move || {
        let config = VisualizerConfig {
            period: Duration::from_millis(33), // ~30 FPS updates
            ..Default::default()
        };
        let bins = config.frequency_bins(); // stable hz ranges for each bin

        // Run visualizer with the frame reader
        // You will get a list of channels on your callback, and each channel will have been frequency magnitures
        // Use `run_with_frame_reader_async()` for use with tokio
        Visualizer::<2>::run_with_frame_reader(
            move || Some(Arc::clone(&tap_for_visualizer)),
            config,
            move |channels, sample_rate_hz| {
                if let Some(ch0) = channels.first() {
                    // Print only the first few bins for demo purposes.
                    for (i, magnitude) in ch0.bins.iter().copied().take(5).enumerate() {
                        let range = &bins[i];
                        println!(
                            "[{} Hz] {:>6.0}..{:>6.0} Hz => {:.4}",
                            sample_rate_hz, range.hz_lo, range.hz_hi, magnitude
                        );
                    }
                    println!("---");
                }
            },
        );
    });

    // Keep main alive while audio + visualizer run.
    thread::sleep(Duration::from_secs(1));
    Ok(())
}
```

## Real-Time Use

`rodio_tap` is suitable for real-time monitoring and low-latency audio
pipeline paths when configured for low latency. In release mode, typical 
overhead is very small (often around ~100 ns).

Suggested low-latency `FrameReaderConfig` starting point:

- `frames_per_batch: Some(64)` (equivalent to 128 sample buffer size in stereo)
- `time_per_batch: None` (use fixed frame batches)
- `sleep_bias: 0.5` (wake earlier to avoid late batch delivery)
- `min_sleep: Duration::from_micros(5)` (tiny cooperative sleep)
- Run in `--release` mode for realistic performance numbers

This profile is generally appropriate for real-time use cases such as ASIO,
CoreAudio, WASAPI, and JACK style pipelines. 

## Examples

### [`wav_visualizer_full`](https://github.com/phayes/rodio_tap/blob/master/examples/wav_visualizer_full.rs)

Terminal FFT visualizer with explicit pipeline wiring and rendering logic.

```bash
cargo run --example wav_visualizer_full -- examples/example.wav
```

### [`wav_visualizer_simple`](https://github.com/phayes/rodio_tap/blob/master/examples/wav_visualizer_simple.rs) (feature: `visualizer`)

Higher-level visualizer API example with less boilerplate.

```bash
cargo run --example wav_visualizer_simple --features visualizer -- examples/example.wav
```

### [`wav_recorder`](https://github.com/phayes/rodio_tap/blob/master/examples/wav_recorder.rs)

Queues one or more WAV files, captures frames with `FrameReader`, and writes a
single output WAV file.

```bash
cargo run --example wav_recorder -- examples/example.wav examples/sweep.wav
```

### [`wav_low_latency`](https://github.com/phayes/rodio_tap/blob/master/examples/wav_low_latency.rs)

Low-latency timing monitor for callback interval and overhead measurement.

```bash
cargo run --release --example wav_low_latency -- --window=5 --loop examples/example.wav
```
