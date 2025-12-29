//! Audio processing and acoustic features for speech recognition.
//!
//! This module provides audio feature extraction capabilities for:
//!
//! - **Mel Filterbank Features**: Perceptually-motivated frequency representation
//! - **MFCC**: Mel-frequency cepstral coefficients (classic ASR features)
//! - **Spectrogram**: Raw power/magnitude spectrum
//! - **Log-Mel**: Log-compressed mel spectrum (neural model input)
//!
//! # Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                           Acoustic Processing                               │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │                                                                              │
//! │   ┌──────────────────────────────────────────────────────────────────────┐  │
//! │   │                    Audio Feature Extraction                          │  │
//! │   │                                                                      │  │
//! │   │   Raw Audio ─► Pre-emphasis ─► Framing ─► Windowing ─► FFT          │  │
//! │   │        ─► Power Spectrum ─► Mel Filterbank ─► Log ─► (DCT)          │  │
//! │   │                                                                      │  │
//! │   │   Features:                                                          │  │
//! │   │   • FeatureExtractor: Batch extraction                               │  │
//! │   │   • StreamingFeatureExtractor: Real-time extraction                  │  │
//! │   │   • MelFilterbank: Triangular filter bank                            │  │
//! │   │                                                                      │  │
//! │   └──────────────────────────────────────────────────────────────────────┘  │
//! │                                                                              │
//! │   Integration with lling-llang:                                             │
//! │   • Features feed into AcousticModel implementations                        │
//! │   • CTC decoder consumes frame posteriors                                   │
//! │   • ASR cascade (H∘C∘L∘G) uses emission probabilities                       │
//! │                                                                              │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::acoustic::{FeatureExtractor, FeatureConfig};
//!
//! // Create feature extractor for 16kHz audio
//! let config = FeatureConfig::default();
//! let extractor = FeatureExtractor::new(config);
//!
//! // Load audio (mono, 16kHz)
//! let audio: Vec<f32> = load_audio("speech.wav");
//!
//! // Extract 40-dim mel filterbank features
//! let filterbank = extractor.extract_filterbank(&audio);
//! println!("Extracted {} frames of {} dimensions", filterbank.len(), filterbank[0].len());
//!
//! // Extract 13-dim MFCC
//! let mfcc = extractor.extract_mfcc(&audio);
//! println!("MFCC: {} frames", mfcc.len());
//! ```
//!
//! # Streaming Example
//!
//! ```ignore
//! use libgrammstein::acoustic::{StreamingFeatureExtractor, FeatureConfig};
//!
//! let mut streaming = StreamingFeatureExtractor::new(FeatureConfig::default());
//!
//! // Process audio in chunks (e.g., from microphone)
//! loop {
//!     let chunk = read_audio_chunk();
//!     streaming.add_samples(&chunk);
//!
//!     // Extract available frames
//!     let features = streaming.extract_filterbank();
//!     if !features.is_empty() {
//!         process_features(&features);
//!     }
//! }
//!
//! // Flush remaining audio at end of stream
//! let final_features = streaming.flush_filterbank();
//! ```
//!
//! # Feature Types
//!
//! | Feature Type | Dimensions | Use Case |
//! |--------------|------------|----------|
//! | **Filterbank** | 40-80 | Neural acoustic models (Conformer, Whisper) |
//! | **MFCC** | 13-39 | GMM-HMM systems, some neural models |
//! | **Log-Mel** | 40-80 | Transformer models, streaming ASR |
//! | **Spectrogram** | FFT/2+1 | Visualization, debugging |
//!
//! # Configuration
//!
//! Common configurations:
//!
//! - `FeatureConfig::default()`: 16kHz wideband speech
//! - `FeatureConfig::telephony()`: 8kHz narrowband (phone)
//! - `FeatureConfig::music()`: 44.1kHz high-fidelity
//!
//! # References
//!
//! - Davis & Mermelstein (1980) - MFCC
//! - Stevens et al. (1937) - Mel scale
//! - Povey et al. (2011) - Kaldi speech recognition toolkit

mod features;

#[cfg(feature = "candle-model")]
mod model;

pub use features::{
    // Constants
    DEFAULT_FRAME_SHIFT,
    DEFAULT_FRAME_SIZE,
    DEFAULT_HIGH_FREQ,
    DEFAULT_LOW_FREQ,
    DEFAULT_NUM_MFCC,
    DEFAULT_NUM_MELS,
    DEFAULT_PRE_EMPHASIS,
    DEFAULT_SAMPLE_RATE,
    LOG_EPSILON,
    // Types
    FeatureConfig,
    FeatureExtractor,
    MelFilterbank,
    StreamingFeatureExtractor,
    WindowType,
};

#[cfg(feature = "candle-model")]
pub use model::{
    AcousticModel, AcousticModelConfig, LinearAcousticModel, MockAcousticModel,
    TransformerAcousticModel,
};
