# Feature Extraction

Detailed guide to `FeatureExtractor` and `StreamingFeatureExtractor` for audio processing.

## What is Feature Extraction?

Feature extraction converts raw audio waveforms into compact representations that capture the essential characteristics of speech. The process reduces a 16kHz audio signal (16,000 samples/second) to approximately 100 feature frames/second, each with 40-80 dimensions.

## FeatureConfig

The `FeatureConfig` struct controls all aspects of feature extraction.

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sample_rate` | `u32` | 16000 | Audio sample rate in Hz |
| `frame_size` | `usize` | 400 | Samples per frame (25ms at 16kHz) |
| `frame_shift` | `usize` | 160 | Hop between frames (10ms at 16kHz) |
| `fft_size` | `usize` | 512 | FFT size (auto: next power of 2) |
| `num_mels` | `usize` | 40 | Mel filterbank channels |
| `num_mfcc` | `usize` | 13 | MFCC coefficients to retain |
| `pre_emphasis` | `f32` | 0.97 | Pre-emphasis coefficient (0 = disabled) |
| `window_type` | `WindowType` | Hanning | Window function |
| `low_freq` | `f32` | 20.0 | Lower frequency bound (Hz) |
| `high_freq` | `f32` | 8000.0 | Upper frequency bound (Hz) |
| `use_power` | `bool` | true | Use power spectrum vs magnitude |
| `normalize_mean` | `bool` | true | Per-utterance mean normalization |
| `normalize_variance` | `bool` | false | Per-utterance variance normalization |
| `include_delta` | `bool` | false | Add velocity features |
| `include_delta_delta` | `bool` | false | Add acceleration features |
| `delta_window` | `usize` | 2 | Window for delta computation |

### Creating Configurations

```rust
use libgrammstein::acoustic::{FeatureConfig, WindowType};

// Default configuration for 16kHz wideband speech
let config = FeatureConfig::default();

// Telephony (8kHz narrowband)
let config = FeatureConfig::telephony();

// Music analysis (44.1kHz)
let config = FeatureConfig::music();

// Custom configuration
let config = FeatureConfig {
    sample_rate: 16000,
    num_mels: 80,
    include_delta: true,
    include_delta_delta: true,
    window_type: WindowType::Hamming,
    ..Default::default()
};

// Query frame timing
println!("Frame duration: {}ms", config.frame_duration_ms());  // 25.0
println!("Frame shift: {}ms", config.frame_shift_ms());        // 10.0
println!("Feature dimension: {}", config.feature_dim());       // 240 (80 + 80 + 80)
```

## Window Functions

The window function shapes each audio frame before FFT to reduce spectral leakage.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Window Function Shapes                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Hanning (default)            Hamming                   Rectangular        │
│                                                                             │
│       ╭──────╮                   ╭──────╮                  ┌──────┐        │
│      ╱        ╲                 ╱        ╲                 │      │        │
│     ╱          ╲               ╱          ╲                │      │        │
│    ╱            ╲             ╱            ╲               │      │        │
│   ╱              ╲           ╱              ╲              │      │        │
│  ─                ─         ─                ─             ─      ─        │
│                                                                             │
│   Zero at edges              Small non-zero edges         Causes leakage   │
│   Best balance               Better sidelobe rejection    Special cases    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Window | Formula | Use Case |
|--------|---------|----------|
| **Hanning** | `0.5(1 - cos(2πn/(N-1)))` | General ASR (default) |
| **Hamming** | `0.54 - 0.46·cos(2πn/(N-1))` | Alternative to Hanning |
| **Blackman** | `0.42 - 0.5·cos(...) + 0.08·cos(...)` | Maximum frequency resolution |
| **Rectangular** | `1.0` | Avoid unless necessary |

```rust
use libgrammstein::acoustic::WindowType;

let config = FeatureConfig {
    window_type: WindowType::Hanning,  // Default, recommended
    ..Default::default()
};
```

## Mel Filterbank

The mel filterbank applies triangular filters spaced on the perceptual mel scale.

### Mel Scale

The mel scale maps frequency to perceived pitch:

```
mel = 2595 × log₁₀(1 + f/700)
f = 700 × (10^(mel/2595) - 1)
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Mel Scale vs Linear Frequency                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Mel │                                    ╱                                 │
│      │                               ╱                                      │
│      │                          ╱                                           │
│      │                     ╱         ← Logarithmic above ~1000 Hz           │
│      │                ╱                                                     │
│      │           ╱         ← Nearly linear below ~1000 Hz                   │
│      │      ╱                                                               │
│      │ ╱                                                                    │
│      └─────────────────────────────────────────────────────────────────── Hz│
│       0    1000   2000   3000   4000   5000   6000   7000   8000           │
│                                                                             │
│  Human hearing is more sensitive to frequency differences at low frequencies│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Triangular Filters

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Mel Filterbank (40 filters)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Gain│   ╱╲    ╱╲    ╱╲     ╱╲     ╱╲      ╱╲       ╱╲        ╱╲           │
│      │  ╱  ╲  ╱  ╲  ╱  ╲   ╱  ╲   ╱  ╲    ╱  ╲     ╱  ╲      ╱  ╲          │
│      │ ╱    ╲╱    ╲╱    ╲ ╱    ╲ ╱    ╲  ╱    ╲   ╱    ╲    ╱    ╲         │
│      │╱                   ╲      ╲      ╲       ╲        ╲         ╲        │
│      └──────────────────────────────────────────────────────────────────── Hz│
│       20                                                              8000  │
│       ◄── Narrow filters ──►           ◄──── Wide filters ────►            │
│           (low frequency)                    (high frequency)               │
│                                                                             │
│  Filters are equally spaced on mel scale, but widen on Hz scale             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

```rust
use libgrammstein::acoustic::MelFilterbank;

// Create filterbank directly (usually done internally)
let fb = MelFilterbank::new(
    40,      // num_mels
    512,     // fft_size
    16000,   // sample_rate
    20.0,    // low_freq
    8000.0,  // high_freq
);

// Convert frequencies
let mel = MelFilterbank::hz_to_mel(1000.0);  // ~1000 mel
let hz = MelFilterbank::mel_to_hz(mel);      // ~1000 Hz

// Apply to power spectrum
let spectrum: Vec<f32> = compute_fft(&frame);  // [257] for 512-point FFT
let mel_energies = fb.apply(&spectrum);         // [40] mel energies
```

## FeatureExtractor

The main feature extraction interface for batch processing.

### Creating an Extractor

```rust
use libgrammstein::acoustic::{FeatureExtractor, FeatureConfig};

let config = FeatureConfig::default();
let extractor = FeatureExtractor::new(config);

// Access configuration
let sample_rate = extractor.config().sample_rate;
let num_mels = extractor.filterbank().num_mels();
```

### Extraction Methods

```rust
// Load 16kHz mono audio (values in range [-1.0, 1.0])
let audio: Vec<f32> = load_audio_file("speech.wav");

// Check expected frame count
let num_frames = extractor.num_frames(audio.len());
println!("Will extract {} frames from {} samples", num_frames, audio.len());

// Extract mel filterbank features (recommended for neural models)
let filterbank = extractor.extract_filterbank(&audio);
// Shape: [num_frames, num_mels]
// Includes: log compression, optional normalization/deltas

// Extract MFCC features (traditional systems)
let mfcc = extractor.extract_mfcc(&audio);
// Shape: [num_frames, num_mfcc]
// Includes: DCT after log mel

// Extract log-mel (no normalization, for streaming)
let log_mel = extractor.extract_log_mel(&audio);
// Shape: [num_frames, num_mels]
// Frame-level only, no utterance normalization

// Extract power spectrogram (debugging, visualization)
let spectrogram = extractor.extract_spectrogram(&audio);
// Shape: [num_frames, fft_size/2 + 1]
```

### Complete Example

```rust
use libgrammstein::acoustic::{FeatureExtractor, FeatureConfig};

fn extract_features_for_asr(audio_path: &str) -> Vec<Vec<f32>> {
    // Configure for neural ASR
    let config = FeatureConfig {
        sample_rate: 16000,
        num_mels: 80,
        include_delta: true,
        include_delta_delta: true,
        normalize_mean: true,
        normalize_variance: true,
        ..Default::default()
    };

    let extractor = FeatureExtractor::new(config);

    // Load audio (implementation depends on your audio library)
    let audio = load_wav_mono_16khz(audio_path);

    // Extract features
    let features = extractor.extract_filterbank(&audio);

    println!(
        "Extracted {} frames, {} dimensions each",
        features.len(),
        features.first().map(|f| f.len()).unwrap_or(0)
    );
    // Output: "Extracted 98 frames, 240 dimensions each"
    // (80 base + 80 delta + 80 delta-delta)

    features
}
```

## StreamingFeatureExtractor

Real-time feature extraction for live audio streams.

### How Streaming Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Streaming Feature Extraction                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Time ───────────────────────────────────────────────────────────────────►  │
│                                                                             │
│  Audio chunks arrive:                                                       │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐                                │
│  │ C1 │ │ C2 │ │ C3 │ │ C4 │ │ C5 │ │ C6 │ ...                            │
│  └────┘ └────┘ └────┘ └────┘ └────┘ └────┘                                │
│                                                                             │
│  Internal buffer accumulates:                                               │
│  ┌────────────────────────┐                                                │
│  │ C1 │ C2 │ C3 │ C4 │    │  Buffer                                       │
│  └────────────────────────┘                                                │
│  ├─Frame─┤                    Complete frames extracted                     │
│       ├─Frame─┤                                                             │
│            ├─Frame─┤                                                        │
│                 ├─Frame─┤                                                   │
│                      ├──────┤  Incomplete (buffered)                        │
│                                                                             │
│  Output: Features for complete frames only                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Creating a Streaming Extractor

```rust
use libgrammstein::acoustic::{StreamingFeatureExtractor, FeatureConfig};

let config = FeatureConfig::default();
let mut streaming = StreamingFeatureExtractor::new(config);
```

### Streaming API

```rust
// Add audio samples (returns number of complete frames available)
let frames_ready = streaming.add_samples(&audio_chunk);

// Check available frames without extracting
let available = streaming.available_frames();

// Extract available filterbank features (consumes from buffer)
let features = streaming.extract_filterbank();
if !features.is_empty() {
    process_features(&features);
}

// Extract MFCC instead
let mfcc = streaming.extract_mfcc();

// At end of stream, flush remaining samples (with padding)
let final_features = streaming.flush_filterbank();

// Reset for new utterance
streaming.reset();

// Query state
let processed = streaming.samples_processed();
let buffered = streaming.buffer_len();
```

### Real-Time Microphone Example

```rust
use libgrammstein::acoustic::{StreamingFeatureExtractor, FeatureConfig};

fn real_time_asr(microphone: &mut Microphone) {
    let config = FeatureConfig::default();
    let mut streaming = StreamingFeatureExtractor::new(config);

    // Process audio in 100ms chunks (1600 samples at 16kHz)
    let chunk_size = 1600;

    loop {
        // Read from microphone
        let chunk = microphone.read(chunk_size);

        if chunk.is_empty() {
            break;  // End of stream
        }

        // Add samples to buffer
        streaming.add_samples(&chunk);

        // Extract any complete frames
        let features = streaming.extract_filterbank();

        if !features.is_empty() {
            // Send to acoustic model for real-time recognition
            let posteriors = acoustic_model.forward(&features);

            // Update decoder state
            for posterior in posteriors {
                decoder.process_frame(&posterior);
            }
        }
    }

    // Process remaining audio
    let final_features = streaming.flush_filterbank();
    if !final_features.is_empty() {
        let posteriors = acoustic_model.forward(&final_features);
        for posterior in posteriors {
            decoder.process_frame(&posterior);
        }
    }

    // Get final transcription
    let result = decoder.finalize();
    println!("Transcription: {}", result);
}
```

## Delta Features

Delta (velocity) and delta-delta (acceleration) features capture temporal dynamics.

### Computation

```
         Σᵢ₌₁ⁿ i × (cₜ₊ᵢ - cₜ₋ᵢ)
Δcₜ = ─────────────────────────
           2 × Σᵢ₌₁ⁿ i²

Δ²cₜ = Δ(Δcₜ)  (apply delta to deltas)
```

Where `n` is the delta window (default: 2).

### Enabling Deltas

```rust
let config = FeatureConfig {
    num_mels: 40,
    include_delta: true,         // Adds 40 more dimensions
    include_delta_delta: true,   // Adds 40 more dimensions
    delta_window: 2,             // Context frames for delta
    ..Default::default()
};

// Feature dimension: 40 + 40 + 40 = 120
let dim = config.feature_dim();
```

### When to Use Deltas

| Model Type | Delta | Delta-Delta | Notes |
|------------|-------|-------------|-------|
| GMM-HMM | Yes | Yes | Captures dynamics |
| LSTM/GRU | Optional | Optional | RNN learns dynamics |
| Transformer | No | No | Self-attention sees all frames |
| CTC | No | No | Usually not needed |

## Performance Considerations

### Memory Usage

```rust
// Estimate memory for extraction
let audio_samples = 16000 * 60;  // 60 seconds at 16kHz
let config = FeatureConfig::default();
let extractor = FeatureExtractor::new(config);

let num_frames = extractor.num_frames(audio_samples);
let feature_dim = config.feature_dim();
let memory_bytes = num_frames * feature_dim * 4;  // f32 = 4 bytes

println!("Features will use {} MB", memory_bytes / (1024 * 1024));
// ~2.4 MB for 60s audio with 40-dim features
```

### Batch vs Streaming Trade-offs

| Aspect | Batch | Streaming |
|--------|-------|-----------|
| **Normalization** | Per-utterance | Per-frame |
| **Latency** | Full audio required | ~25ms per frame |
| **Memory** | O(audio length) | O(frame size) |
| **Use Case** | Offline processing | Real-time ASR |

## Related Documentation

- [Acoustic Overview](overview.md) - Module introduction
- [Acoustic Models](models.md) - Neural model inference
- [lling-llang Integration](../../../lling-llang/docs/acoustic/overview.md)
