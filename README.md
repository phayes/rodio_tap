# rodio_tap

`rodio_tap` taps `rodio::Source` audio into `f32` PCM batches while still passing the source through to playback.

Use it when you want to analyze, visualize, meter, or record playback data in real time.

## What it provides

- `TapReader` + `TapAdapter`: low-level ring-buffer access.
- `FrameReader`: synchronous high-level reader that yields channel-aligned batches.
- `AsyncFrameReader` (feature `async`): async high-level reader for Tokio runtimes.

All batches are interleaved (`[L, R, L, R, ...]` for stereo) and channel-aligned.

## Installation

In your `Cargo.toml`:

```toml
[dependencies]
rodio_tap = "0.1.0"
```

Enable async support if needed:

```toml
[dependencies]
rodio_tap = { version = "0.1.0", features = ["async"] }
```

## Quick start (synchronous)

```rust
use std::sync::Arc;
use std::thread;
use rodio::Decoder;
use rodio_tap::{FrameReader, TapReader};

# fn run<S>(rodio_source: S)
# where
#     S: rodio::Source + Send + 'static,
#     S::Item: cpal::Sample + Send + 'static,
#     f32: cpal::FromSample<S::Item>,
# {
let (tap_reader, tap_adapter) = TapReader::new(rodio_source);

// Send `tap_adapter` into your rodio playback pipeline.
let _ = tap_adapter;

let tap_for_reader = Arc::clone(&tap_reader);
thread::spawn(move || {
    let mut reader = FrameReader::new(move || Some(Arc::clone(&tap_for_reader)));
    reader.run(|batch, channels, sample_rate_hz| {
        let frames = batch.len() / channels;
        println!("{} frames @ {} Hz", frames, sample_rate_hz);

        for frame in batch.chunks_exact(channels) {
            let _ = frame;
            // Process one interleaved frame.
        }
    });
});
# }
```

## Async reader

With the `async` feature enabled:

```rust
use std::sync::Arc;
use rodio_tap::AsyncFrameReader;

async fn run_reader(tap: Arc<rodio_tap::TapReader>) {
    let mut reader = AsyncFrameReader::new(move || Some(Arc::clone(&tap)));
    reader
        .run(|batch, channels, sample_rate_hz| {
            let frames = batch.len() / channels;
            println!("{} frames @ {} Hz", frames, sample_rate_hz);
        })
        .await;
}
```

## Example visualizer

This repository includes `examples/wav_visualizer.rs`, which plays a WAV file and shows a terminal FFT view.

Run it with:

```bash
cargo run --example wav_visualizer -- examples/example.wav
```
