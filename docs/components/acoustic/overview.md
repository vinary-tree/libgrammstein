# Acoustic Processing

Audio feature extraction for speech recognition and acoustic modeling.

## What is Acoustic Processing?

Acoustic processing transforms raw audio waveforms into compact feature representations suitable for speech recognition. Just as humans perceive sound through the cochlea's frequency analysis, acoustic feature extraction mimics this process computationally.

The acoustic module provides:

- **Mel Filterbank Features**: Perceptually-motivated frequency representation
- **MFCC**: Mel-frequency cepstral coefficients for traditional ASR
- **Streaming Extraction**: Real-time processing for live audio
- **Neural Models**: Transformer-based acoustic models via Candle

## Terminology

| Term | Definition |
|------|------------|
| **Frame** | A short segment of audio (typically 25ms) analyzed as a unit |
| **Frame Shift** | Time between consecutive frames (typically 10ms), creating overlap |
| **Mel Scale** | Perceptual frequency scale where equal distances sound equally spaced |
| **Filterbank** | Set of triangular filters that bin frequency energy |
| **MFCC** | Cepstral coefficients from DCT of log mel filterbank |
| **Spectrogram** | Time-frequency representation of audio |

## Feature Extraction Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Audio Feature Extraction Pipeline                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │   Raw    │───►│   Pre-   │───►│ Framing  │───►│ Windowing│             │
│  │  Audio   │    │ emphasis │    │ (25ms)   │    │ (Hanning)│             │
│  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘             │
│                                                       │                     │
│                                                       ▼                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │  Output  │◄───│   Log    │◄───│   Mel    │◄───│   FFT    │             │
│  │ Features │    │ Compress │    │Filterbank│    │ (Power)  │             │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘             │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Optional Processing                           │   │
│  │   ┌────────┐    ┌────────────┐    ┌───────────────────────────┐    │   │
│  │   │  DCT   │    │   Delta    │    │  Mean/Variance            │    │   │
│  │   │(→MFCC) │    │  Features  │    │  Normalization            │    │   │
│  │   └────────┘    └────────────┘    └───────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Stages

1. **Pre-emphasis**: Boosts high frequencies to compensate for spectral tilt
   - Formula: `y[n] = x[n] - α·x[n-1]` where α ≈ 0.97

2. **Framing**: Segments audio into overlapping frames
   - Default: 25ms frames with 10ms shift (15ms overlap)

3. **Windowing**: Applies window function to reduce spectral leakage
   - Hanning (default), Hamming, Blackman, or Rectangular

4. **FFT**: Computes frequency spectrum via Fast Fourier Transform

5. **Mel Filterbank**: Applies triangular filters on mel scale
   - Mel formula: `mel = 2595 × log₁₀(1 + f/700)`

6. **Log Compression**: Applies logarithm for dynamic range compression

7. **DCT** (optional): Discrete Cosine Transform produces MFCC

## Feature Types Comparison

| Feature Type | Dimensions | Best For | Normalization |
|--------------|------------|----------|---------------|
| **Mel Filterbank** | 40-80 | Neural models (Conformer, Whisper) | Per-utterance |
| **Log-Mel** | 40-80 | Streaming ASR, real-time | Frame-level |
| **MFCC** | 13-39 | GMM-HMM, legacy systems | Per-utterance |
| **Spectrogram** | FFT/2+1 | Visualization, debugging | None |

## Quick Start

```rust
use libgrammstein::acoustic::{FeatureExtractor, FeatureConfig};

// Create feature extractor for 16kHz audio
let config = FeatureConfig::default();
let extractor = FeatureExtractor::new(config);

// Load mono 16kHz audio (values in [-1.0, 1.0])
let audio: Vec<f32> = load_audio_file("speech.wav");

// Extract 40-dimensional mel filterbank features
let filterbank = extractor.extract_filterbank(&audio);
println!("Extracted {} frames of {} dimensions",
    filterbank.len(),        // e.g., 98 frames for 1s audio
    filterbank[0].len()      // 40 dimensions
);

// Or extract 13-dimensional MFCC
let mfcc = extractor.extract_mfcc(&audio);
```

## Common Configurations

### Wideband Speech (Default)

```rust
let config = FeatureConfig::default();
// sample_rate: 16000 Hz
// frame_size: 400 samples (25ms)
// frame_shift: 160 samples (10ms)
// num_mels: 40
// low_freq: 20 Hz, high_freq: 8000 Hz
```

### Telephony (Narrowband)

```rust
let config = FeatureConfig::telephony();
// sample_rate: 8000 Hz
// high_freq: 4000 Hz (Nyquist limit)
// Suitable for phone audio
```

### Music / High-Fidelity

```rust
let config = FeatureConfig::music();
// sample_rate: 44100 Hz
// num_mels: 80
// Suitable for music analysis
```

## Integration with ASR

The acoustic module integrates with lling-llang's ASR cascade:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ASR Pipeline Integration                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   libgrammstein                        lling-llang                          │
│  ┌─────────────────┐                ┌──────────────────────────────────┐   │
│  │FeatureExtractor │───────────────►│        AcousticModel             │   │
│  │  (MFCC/FB)      │  features      │   (TransformerAcousticModel)    │   │
│  └─────────────────┘                └────────────┬─────────────────────┘   │
│                                                  │ posteriors              │
│                                                  ▼                         │
│                                     ┌──────────────────────────────────┐   │
│                                     │      CTC/HMM Decoder             │   │
│                                     │   (Compose with LM WFST)         │   │
│                                     └────────────┬─────────────────────┘   │
│                                                  │                         │
│                                                  ▼                         │
│                                            Transcription                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Feature Flags

| Flag | Dependencies | Description |
|------|--------------|-------------|
| `acoustic` | rustfft, realfft | Audio feature extraction |
| `candle-model` | candle-core, candle-nn | Neural acoustic models |

Enable in `Cargo.toml`:

```toml
[dependencies]
libgrammstein = { version = "0.1", features = ["acoustic"] }

# Or with neural models:
libgrammstein = { version = "0.1", features = ["candle-model"] }
```

## Related Documentation

- [Feature Extraction](features.md) - Detailed FeatureExtractor API
- [Acoustic Models](models.md) - Candle-based neural models
- [lling-llang Acoustic Integration](../../../lling-llang/docs/acoustic/overview.md)
