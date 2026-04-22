# rodio_tap

`rodio_tap` taps `rodio::Source` audio while still passing the source through to playback.

Use it when you want to analyze, visualize, meter, or record playback data in real time.

https://github.com/user-attachments/assets/54d66615-4ef3-4876-af7b-6dc5886b64ff

## What it provides

- `TapReader` + `TapAdapter`: low-level packet ring-buffer access.
- `FrameReader`: synchronous high-level reader that yields frame batches.
- `AsyncFrameReader` (feature `async`): async high-level reader for Tokio runtimes.

`FrameReader` batches are arrays of interleaved frames (`[L, R, ...]` per frame).

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

fn run<S>(rodio_source: S)
where
     S: rodio::Source + Send + 'static,
     S::Item: cpal::Sample + Send + 'static,
     f32: cpal::FromSample<S::Item>,
{
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

When stream format changes (sample rate / channel count) inside a tap, `FrameReader`
emits any in-progress partial batch before switching to the new format.

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

## Example visualizer

This repository includes `examples/wav_visualizer.rs`, which plays a WAV file and shows a terminal FFT view.

Run it with:

```bash
cargo run --example wav_visualizer -- examples/example.wav
```
