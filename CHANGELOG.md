# Changelog


## [Unreleased]

### Added

- Added multi-resolution spectrum analysis. `Visualizer` now uses a long
  `bass_window_duration` below `crossover_frequency_hz` and a shorter
  `upper_window_duration` above it. The defaults are 170 ms, 250 Hz, and 33 ms
  respectively.
- Added `VisualizerConfig::allow_decimation`, enabled by default. The visualizer
  automatically applies anti-aliased power-of-two decimation when the highest
  configured frequency bin leaves at least 20% Nyquist headroom. Peak and RMS
  measurements, callback cadence, and the reported stream sample rate continue
  to use the original source rate.
- Added `FrequencyData` for each configured frequency band:
  - `magnitude` is the mean normalized FFT magnitude and is useful for ordinary
    spectrum bars.
  - `peak_magnitude` is the strongest normalized FFT line in the band and is
    useful when a narrow tone should remain visible inside a wide band.
  - `power` is normalized, additive band power and is useful for comparing
    frequency regions, color balance, and level-based effects.
- Added `FrequencyData::power_db(reference_power)` for converting linear band
  power to decibels using `10 * log10(power / reference_power)`.
- Added `avx`, `sse`, `neon`, and `wasm_simd` crate features, forwarded to the
  FFT implementation.
- Added `FrameReaderConfig::drop_late_batches`, disabled by default. When
  enabled, synchronous and asynchronous readers skip queued batches for whole
  elapsed batch intervals, allowing live analysis to recover toward the
  current playback position without draining future prebuffered audio.
- Added `FrameBatch`, which reports the delivered frames and stream format
  together with the number and audio duration of batches discarded immediately
  before delivery.
- Added `VisualizerConfig::drop_late_batches`, enabled by default. Built-in
  visualizer runners forward this policy to their frame readers so live
  visualizations recover from missed emission intervals. Rolling analysis state
  is reset after a skip to avoid joining samples across the discontinuity.
  Direct `Visualizer::process_batch` calls remain caller-controlled.

### Changed

- Replaced `rustfft` with `realfft` for real-to-complex visualizer FFTs.
- FFT magnitudes are now always normalized using Hann coherent gain. This makes
  equal-amplitude tones comparable across the bass and upper FFT window sizes.
- FFT power is normalized using FFT length and Hann window power, allowing
  power from the two FFT sizes to be combined meaningfully.
- `ChannelSpectrum::bins` changed from `Vec<f32>` to `Vec<FrequencyData>`.
  Existing spectrum displays should read `bin.magnitude`; energy-based
  consumers should read `bin.power`.
- `Visualizer::process_batch` now returns `Vec<VisualizerFrame>` instead of
  `Option<VisualizerFrame>`. Direct callers must process every returned frame;
  oversized batches can complete multiple emission hops.
- Partial startup windows are emitted with normalization based on the available
  history.
- `FrameReader` and `AsyncFrameReader` now deliver completed batches at the end
  of the playback interval represented by the batch. Buffered input is paced
  instead of being emitted in callback bursts, and missed deadlines rebase
  rather than triggering catch-up bursts.
- `FrameReader::run` and `AsyncFrameReader::run` callbacks now receive one
  `FrameBatch` argument instead of separate frame, channel-count, and
  sample-rate arguments.
- Synchronous and asynchronous readers now share batching, tap-switching, and
  format-change behavior.
- On an in-band format change, readers still emit a partial batch in the old
  format before switching. On a tap change, partial state is discarded.

### Removed

- Removed `VisualizerConfig::normalize_by_fft_size`; coherent-gain
  normalization is now always applied.

### Migration guide

Update visualizer configuration:

```rust
let config = VisualizerConfig {
    period: Duration::from_millis(33),
    bass_window_duration: Duration::from_millis(170),
    upper_window_duration: Duration::from_millis(33),
    crossover_frequency_hz: 250.0,
    allow_decimation: true,
    drop_late_batches: true,
    ..Default::default()
};
```

Read the desired measurement from each frequency band:

```rust
for data in &channel.bins {
    render_bar(data.magnitude);
    update_band_balance(data.power);
}
```

Handle every frame returned by direct batch processing:

```rust
for frame in visualizer.process_batch(batch, channels, sample_rate_hz) {
    consume(frame);
}
```

Update frame-reader callbacks:

```rust
reader.run(|batch| {
    process(batch.frames, batch.channels, batch.sample_rate_hz);

    if batch.dropped_batches > 0 {
        handle_discontinuity(batch.dropped_duration);
    }
});
```

## [0.2.0] - 2026-04-25

- Previous published release.

[Unreleased]: https://github.com/phayes/rodio_tap/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/phayes/rodio_tap/releases/tag/v0.2.0
