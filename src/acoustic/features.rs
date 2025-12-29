//! Audio feature extraction for acoustic models.
//!
//! This module provides standard audio feature extraction methods:
//!
//! - **Mel Filterbank**: Perceptually-motivated frequency binning
//! - **MFCC**: Mel-frequency cepstral coefficients
//! - **Spectrogram**: Power or magnitude spectrogram
//!
//! # Signal Processing Pipeline
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                      Audio Feature Extraction Pipeline                       │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │                                                                              │
//! │   Raw Audio  ─►  Pre-emphasis  ─►  Framing  ─►  Windowing  ─►  FFT          │
//! │                                                                              │
//! │        ─►  Power Spectrum  ─►  Mel Filterbank  ─►  Log  ─►  (DCT for MFCC)  │
//! │                                                                              │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::acoustic::{FeatureExtractor, FeatureConfig};
//!
//! let config = FeatureConfig::default();
//! let extractor = FeatureExtractor::new(config);
//!
//! // Extract 40-dim filterbank features
//! let audio: Vec<f32> = load_audio("speech.wav");
//! let features = extractor.extract_filterbank(&audio);
//!
//! // Extract 13-dim MFCC
//! let mfcc = extractor.extract_mfcc(&audio);
//! ```
//!
//! # References
//!
//! - Davis & Mermelstein, "Comparison of Parametric Representations for
//!   Monosyllabic Word Recognition" (1980) - MFCC
//! - Stevens et al., "A Scale for the Measurement of the Psychological
//!   Magnitude Pitch" (1937) - Mel scale

use std::f32::consts::PI;

use realfft::{RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;

/// Default sample rate (16 kHz, common for speech).
pub const DEFAULT_SAMPLE_RATE: u32 = 16000;

/// Default frame size in samples (25ms at 16kHz = 400 samples).
pub const DEFAULT_FRAME_SIZE: usize = 400;

/// Default frame shift in samples (10ms at 16kHz = 160 samples).
pub const DEFAULT_FRAME_SHIFT: usize = 160;

/// Default number of mel filterbanks.
pub const DEFAULT_NUM_MELS: usize = 40;

/// Default number of MFCC coefficients.
pub const DEFAULT_NUM_MFCC: usize = 13;

/// Default pre-emphasis coefficient.
pub const DEFAULT_PRE_EMPHASIS: f32 = 0.97;

/// Default low frequency bound (Hz).
pub const DEFAULT_LOW_FREQ: f32 = 20.0;

/// Default high frequency bound (Hz) - Nyquist for 16kHz.
pub const DEFAULT_HIGH_FREQ: f32 = 8000.0;

/// Small epsilon for log stability.
pub const LOG_EPSILON: f32 = 1e-10;

/// Window function type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowType {
    /// Hanning (Hann) window: `0.5 * (1 - cos(2πn/(N-1)))`
    Hanning,
    /// Hamming window: `0.54 - 0.46 * cos(2πn/(N-1))`
    Hamming,
    /// Rectangular window (no windowing).
    Rectangular,
    /// Blackman window: `0.42 - 0.5*cos(2πn/(N-1)) + 0.08*cos(4πn/(N-1))`
    Blackman,
}

impl Default for WindowType {
    fn default() -> Self {
        Self::Hanning
    }
}

/// Configuration for audio feature extraction.
#[derive(Clone, Debug)]
pub struct FeatureConfig {
    /// Sample rate in Hz (typically 16000 or 8000).
    pub sample_rate: u32,

    /// Frame size in samples.
    pub frame_size: usize,

    /// Frame shift (hop) in samples.
    pub frame_shift: usize,

    /// FFT size (power of 2, >= frame_size).
    pub fft_size: usize,

    /// Number of mel filterbank channels.
    pub num_mels: usize,

    /// Number of MFCC coefficients to extract.
    pub num_mfcc: usize,

    /// Pre-emphasis coefficient (0 to disable).
    pub pre_emphasis: f32,

    /// Window function type.
    pub window_type: WindowType,

    /// Lower frequency bound for filterbank (Hz).
    pub low_freq: f32,

    /// Upper frequency bound for filterbank (Hz).
    pub high_freq: f32,

    /// Whether to use power spectrum (true) or magnitude (false).
    pub use_power: bool,

    /// Whether to apply mean normalization per utterance.
    pub normalize_mean: bool,

    /// Whether to apply variance normalization per utterance.
    pub normalize_variance: bool,

    /// Whether to include delta (velocity) features.
    pub include_delta: bool,

    /// Whether to include delta-delta (acceleration) features.
    pub include_delta_delta: bool,

    /// Window size for delta computation.
    pub delta_window: usize,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        let frame_size = DEFAULT_FRAME_SIZE;
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            frame_size,
            frame_shift: DEFAULT_FRAME_SHIFT,
            fft_size: frame_size.next_power_of_two(),
            num_mels: DEFAULT_NUM_MELS,
            num_mfcc: DEFAULT_NUM_MFCC,
            pre_emphasis: DEFAULT_PRE_EMPHASIS,
            window_type: WindowType::Hanning,
            low_freq: DEFAULT_LOW_FREQ,
            high_freq: DEFAULT_HIGH_FREQ,
            use_power: true,
            normalize_mean: true,
            normalize_variance: false,
            include_delta: false,
            include_delta_delta: false,
            delta_window: 2,
        }
    }
}

impl FeatureConfig {
    /// Create configuration for 8kHz audio (telephony).
    pub fn telephony() -> Self {
        Self {
            sample_rate: 8000,
            frame_size: 200, // 25ms at 8kHz
            frame_shift: 80, // 10ms at 8kHz
            fft_size: 256,
            high_freq: 4000.0, // Nyquist for 8kHz
            ..Default::default()
        }
    }

    /// Create configuration for 16kHz audio (wideband speech).
    pub fn wideband() -> Self {
        Self::default()
    }

    /// Create configuration for 44.1kHz audio (music/CD quality).
    pub fn music() -> Self {
        Self {
            sample_rate: 44100,
            frame_size: 1102, // ~25ms at 44.1kHz
            frame_shift: 441, // ~10ms at 44.1kHz
            fft_size: 2048,
            high_freq: 22050.0, // Nyquist for 44.1kHz
            num_mels: 80,       // More bins for wider bandwidth
            ..Default::default()
        }
    }

    /// Frame duration in milliseconds.
    pub fn frame_duration_ms(&self) -> f32 {
        1000.0 * self.frame_size as f32 / self.sample_rate as f32
    }

    /// Frame shift in milliseconds.
    pub fn frame_shift_ms(&self) -> f32 {
        1000.0 * self.frame_shift as f32 / self.sample_rate as f32
    }

    /// Get the output feature dimension.
    pub fn feature_dim(&self) -> usize {
        let base = self.num_mels;
        let mut dim = base;
        if self.include_delta {
            dim += base;
        }
        if self.include_delta_delta {
            dim += base;
        }
        dim
    }
}

/// Triangular mel filterbank for frequency binning.
///
/// The mel scale is a perceptual scale of pitches judged by listeners
/// to be equal in distance from one another.
#[derive(Clone, Debug)]
pub struct MelFilterbank {
    /// Number of mel channels.
    num_mels: usize,

    /// FFT size (determines number of frequency bins).
    fft_size: usize,

    /// Sample rate for frequency calculations.
    sample_rate: u32,

    /// Lower frequency bound (Hz).
    low_freq: f32,

    /// Upper frequency bound (Hz).
    high_freq: f32,

    /// Filter weights: `[num_mels, num_fft_bins]` stored as sparse representation.
    /// Each mel channel has (start_bin, weights) for non-zero weights.
    filters: Vec<MelFilter>,
}

/// A single triangular mel filter.
#[derive(Clone, Debug)]
struct MelFilter {
    /// Starting FFT bin index.
    start_bin: usize,
    /// Filter weights for consecutive bins starting at start_bin.
    weights: Vec<f32>,
}

impl MelFilterbank {
    /// Create a new mel filterbank.
    ///
    /// # Arguments
    ///
    /// * `num_mels` - Number of mel channels
    /// * `fft_size` - FFT size (must be power of 2)
    /// * `sample_rate` - Audio sample rate in Hz
    /// * `low_freq` - Lower frequency bound in Hz
    /// * `high_freq` - Upper frequency bound in Hz
    pub fn new(
        num_mels: usize,
        fft_size: usize,
        sample_rate: u32,
        low_freq: f32,
        high_freq: f32,
    ) -> Self {
        let mut filterbank = Self {
            num_mels,
            fft_size,
            sample_rate,
            low_freq,
            high_freq,
            filters: Vec::with_capacity(num_mels),
        };
        filterbank.build_filters();
        filterbank
    }

    /// Convert frequency in Hz to mel scale.
    ///
    /// `mel = 2595 * log10(1 + f/700)`
    #[inline]
    pub fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    /// Convert mel scale to frequency in Hz.
    ///
    /// `f = 700 * (10^(mel/2595) - 1)`
    #[inline]
    pub fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    /// Build the triangular filter bank.
    fn build_filters(&mut self) {
        let num_fft_bins = self.fft_size / 2 + 1;
        let hz_per_bin = self.sample_rate as f32 / self.fft_size as f32;

        // Convert frequency bounds to mel scale
        let low_mel = Self::hz_to_mel(self.low_freq);
        let high_mel = Self::hz_to_mel(self.high_freq);

        // Create equally spaced mel points (including two extra for edges)
        let num_points = self.num_mels + 2;
        let mel_points: Vec<f32> = (0..num_points)
            .map(|i| low_mel + (high_mel - low_mel) * i as f32 / (num_points - 1) as f32)
            .collect();

        // Convert mel points to Hz
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| Self::mel_to_hz(m)).collect();

        // Convert Hz points to FFT bin indices (can be fractional)
        let bin_points: Vec<f32> = hz_points.iter().map(|&f| f / hz_per_bin).collect();

        // Build triangular filters
        self.filters.clear();
        for m in 0..self.num_mels {
            let left = bin_points[m];
            let center = bin_points[m + 1];
            let right = bin_points[m + 2];

            let start_bin = left.floor() as usize;
            let end_bin = (right.ceil() as usize).min(num_fft_bins);

            if start_bin >= end_bin {
                // Edge case: very narrow filter
                self.filters.push(MelFilter {
                    start_bin: 0,
                    weights: vec![],
                });
                continue;
            }

            let mut weights = Vec::with_capacity(end_bin - start_bin);

            for bin in start_bin..end_bin {
                let bin_f = bin as f32;
                let weight = if bin_f < left {
                    0.0
                } else if bin_f < center {
                    // Rising edge
                    (bin_f - left) / (center - left)
                } else if bin_f < right {
                    // Falling edge
                    (right - bin_f) / (right - center)
                } else {
                    0.0
                };
                weights.push(weight);
            }

            self.filters.push(MelFilter { start_bin, weights });
        }
    }

    /// Apply filterbank to power/magnitude spectrum.
    ///
    /// # Arguments
    ///
    /// * `spectrum` - Power or magnitude spectrum `[fft_size/2 + 1]`
    ///
    /// # Returns
    ///
    /// Mel-scale energies `[num_mels]`
    pub fn apply(&self, spectrum: &[f32]) -> Vec<f32> {
        let mut mel_energies = vec![0.0f32; self.num_mels];

        for (m, filter) in self.filters.iter().enumerate() {
            let mut energy = 0.0f32;
            for (i, &weight) in filter.weights.iter().enumerate() {
                let bin = filter.start_bin + i;
                if bin < spectrum.len() {
                    energy += weight * spectrum[bin];
                }
            }
            mel_energies[m] = energy;
        }

        mel_energies
    }

    /// Get number of mel channels.
    pub fn num_mels(&self) -> usize {
        self.num_mels
    }

    /// Get the filter bank as a dense matrix (for debugging/visualization).
    pub fn to_dense(&self) -> Vec<Vec<f32>> {
        let num_fft_bins = self.fft_size / 2 + 1;
        let mut dense = vec![vec![0.0f32; num_fft_bins]; self.num_mels];

        for (m, filter) in self.filters.iter().enumerate() {
            for (i, &weight) in filter.weights.iter().enumerate() {
                let bin = filter.start_bin + i;
                if bin < num_fft_bins {
                    dense[m][bin] = weight;
                }
            }
        }

        dense
    }
}

/// DCT (Discrete Cosine Transform) for MFCC computation.
///
/// Uses DCT-II (the most common type for MFCC).
#[derive(Clone, Debug)]
struct DctTransform {
    /// Number of input features (mel channels).
    num_input: usize,
    /// Number of output coefficients.
    num_output: usize,
    /// DCT matrix: `[num_output, num_input]`
    matrix: Vec<Vec<f32>>,
}

impl DctTransform {
    /// Create a new DCT transform.
    fn new(num_input: usize, num_output: usize) -> Self {
        let mut matrix = vec![vec![0.0f32; num_input]; num_output];

        let scale = (2.0 / num_input as f32).sqrt();

        for k in 0..num_output {
            for n in 0..num_input {
                // DCT-II formula: cos(π * k * (n + 0.5) / N)
                matrix[k][n] = scale * (PI * k as f32 * (n as f32 + 0.5) / num_input as f32).cos();
            }
        }

        // First coefficient has different normalization
        if !matrix.is_empty() {
            let scale0 = (1.0 / num_input as f32).sqrt();
            for n in 0..num_input {
                matrix[0][n] = scale0;
            }
        }

        Self {
            num_input,
            num_output,
            matrix,
        }
    }

    /// Apply DCT to input vector.
    fn apply(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0f32; self.num_output];

        for k in 0..self.num_output {
            let mut sum = 0.0f32;
            for (n, &x) in input.iter().enumerate().take(self.num_input) {
                sum += self.matrix[k][n] * x;
            }
            output[k] = sum;
        }

        output
    }
}

/// Audio feature extractor.
///
/// Extracts MFCC, mel filterbank, or spectrogram features from raw audio.
pub struct FeatureExtractor {
    /// Configuration.
    config: FeatureConfig,

    /// Pre-computed window function.
    window: Vec<f32>,

    /// Mel filterbank.
    filterbank: MelFilterbank,

    /// DCT transform for MFCC.
    dct: DctTransform,

    /// Real FFT planner.
    fft: std::sync::Arc<dyn RealToComplex<f32>>,
}

impl FeatureExtractor {
    /// Create a new feature extractor with the given configuration.
    pub fn new(config: FeatureConfig) -> Self {
        // Build window function
        let window = Self::build_window(config.frame_size, config.window_type);

        // Build mel filterbank
        let filterbank = MelFilterbank::new(
            config.num_mels,
            config.fft_size,
            config.sample_rate,
            config.low_freq,
            config.high_freq,
        );

        // Build DCT for MFCC
        let dct = DctTransform::new(config.num_mels, config.num_mfcc);

        // Create FFT planner
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(config.fft_size);

        Self {
            config,
            window,
            filterbank,
            dct,
            fft,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &FeatureConfig {
        &self.config
    }

    /// Build window function.
    fn build_window(size: usize, window_type: WindowType) -> Vec<f32> {
        (0..size)
            .map(|n| {
                let x = 2.0 * PI * n as f32 / (size - 1) as f32;
                match window_type {
                    WindowType::Hanning => 0.5 * (1.0 - x.cos()),
                    WindowType::Hamming => 0.54 - 0.46 * x.cos(),
                    WindowType::Rectangular => 1.0,
                    WindowType::Blackman => 0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos(),
                }
            })
            .collect()
    }

    /// Apply pre-emphasis filter.
    ///
    /// `y[n] = x[n] - α * x[n-1]`
    fn apply_pre_emphasis(&self, audio: &[f32]) -> Vec<f32> {
        if self.config.pre_emphasis == 0.0 || audio.is_empty() {
            return audio.to_vec();
        }

        let mut output = Vec::with_capacity(audio.len());
        output.push(audio[0]);

        for i in 1..audio.len() {
            output.push(audio[i] - self.config.pre_emphasis * audio[i - 1]);
        }

        output
    }

    /// Extract a single frame and apply windowing.
    fn extract_frame(&self, audio: &[f32], start: usize) -> Vec<f32> {
        let mut frame = vec![0.0f32; self.config.fft_size];

        // Copy audio samples and apply window
        for i in 0..self.config.frame_size {
            let sample_idx = start + i;
            if sample_idx < audio.len() {
                frame[i] = audio[sample_idx] * self.window[i];
            }
        }

        frame
    }

    /// Compute power spectrum from windowed frame.
    fn compute_spectrum(&self, frame: &mut [f32]) -> Vec<f32> {
        let mut spectrum = vec![Complex::new(0.0f32, 0.0f32); self.config.fft_size / 2 + 1];

        // Perform FFT
        self.fft
            .process(frame, &mut spectrum)
            .expect("FFT failed");

        // Compute power or magnitude spectrum
        spectrum
            .iter()
            .map(|c| {
                if self.config.use_power {
                    c.norm_sqr() // |c|^2
                } else {
                    c.norm() // |c|
                }
            })
            .collect()
    }

    /// Compute delta features using regression.
    ///
    /// `d[t] = Σ_{n=-N}^{N} n * c[t+n] / (2 * Σ_{n=1}^{N} n²)`
    fn compute_delta(&self, features: &[Vec<f32>], window: usize) -> Vec<Vec<f32>> {
        let num_frames = features.len();
        if num_frames == 0 {
            return vec![];
        }

        let dim = features[0].len();
        let mut delta = vec![vec![0.0f32; dim]; num_frames];

        // Normalization factor
        let norm: f32 = 2.0 * (1..=window).map(|n| (n * n) as f32).sum::<f32>();

        for t in 0..num_frames {
            for d in 0..dim {
                let mut sum = 0.0f32;
                for n in 1..=window {
                    // Handle edge cases with padding
                    let t_minus = if t >= n { t - n } else { 0 };
                    let t_plus = (t + n).min(num_frames - 1);

                    sum += n as f32 * (features[t_plus][d] - features[t_minus][d]);
                }
                delta[t][d] = sum / norm;
            }
        }

        delta
    }

    /// Normalize features (mean and/or variance normalization).
    fn normalize(&self, features: &mut [Vec<f32>]) {
        if features.is_empty() {
            return;
        }

        let num_frames = features.len();
        let dim = features[0].len();

        if self.config.normalize_mean {
            // Compute mean
            let mut mean = vec![0.0f32; dim];
            for frame in features.iter() {
                for (d, &v) in frame.iter().enumerate() {
                    mean[d] += v;
                }
            }
            for m in mean.iter_mut() {
                *m /= num_frames as f32;
            }

            // Subtract mean
            for frame in features.iter_mut() {
                for (d, v) in frame.iter_mut().enumerate() {
                    *v -= mean[d];
                }
            }
        }

        if self.config.normalize_variance {
            // Compute variance
            let mut var = vec![0.0f32; dim];
            for frame in features.iter() {
                for (d, &v) in frame.iter().enumerate() {
                    var[d] += v * v;
                }
            }
            for v in var.iter_mut() {
                *v = (*v / num_frames as f32).sqrt().max(1e-10);
            }

            // Divide by std
            for frame in features.iter_mut() {
                for (d, v) in frame.iter_mut().enumerate() {
                    *v /= var[d];
                }
            }
        }
    }

    /// Extract mel filterbank features from audio.
    ///
    /// # Arguments
    ///
    /// * `audio` - Raw audio samples (mono, at configured sample rate)
    ///
    /// # Returns
    ///
    /// Feature matrix `[num_frames, num_mels]` (or larger with deltas)
    pub fn extract_filterbank(&self, audio: &[f32]) -> Vec<Vec<f32>> {
        if audio.is_empty() {
            return vec![];
        }

        // Apply pre-emphasis
        let emphasized = self.apply_pre_emphasis(audio);

        // Calculate number of frames
        let num_frames = if emphasized.len() > self.config.frame_size {
            (emphasized.len() - self.config.frame_size) / self.config.frame_shift + 1
        } else {
            1
        };

        // Extract features for each frame
        let mut features: Vec<Vec<f32>> = Vec::with_capacity(num_frames);

        for i in 0..num_frames {
            let start = i * self.config.frame_shift;
            let mut frame = self.extract_frame(&emphasized, start);
            let spectrum = self.compute_spectrum(&mut frame);
            let mel_energies = self.filterbank.apply(&spectrum);

            // Apply log compression
            let log_mel: Vec<f32> = mel_energies
                .iter()
                .map(|&e| (e + LOG_EPSILON).ln())
                .collect();

            features.push(log_mel);
        }

        // Apply normalization
        self.normalize(&mut features);

        // Add delta features if configured
        if self.config.include_delta || self.config.include_delta_delta {
            let delta = self.compute_delta(&features, self.config.delta_window);

            if self.config.include_delta_delta {
                let delta_delta = self.compute_delta(&delta, self.config.delta_window);

                // Concatenate features
                for (i, frame) in features.iter_mut().enumerate() {
                    frame.extend_from_slice(&delta[i]);
                    frame.extend_from_slice(&delta_delta[i]);
                }
            } else {
                for (i, frame) in features.iter_mut().enumerate() {
                    frame.extend_from_slice(&delta[i]);
                }
            }
        }

        features
    }

    /// Extract MFCC features from audio.
    ///
    /// # Arguments
    ///
    /// * `audio` - Raw audio samples (mono, at configured sample rate)
    ///
    /// # Returns
    ///
    /// Feature matrix `[num_frames, num_mfcc]` (or larger with deltas)
    pub fn extract_mfcc(&self, audio: &[f32]) -> Vec<Vec<f32>> {
        if audio.is_empty() {
            return vec![];
        }

        // Apply pre-emphasis
        let emphasized = self.apply_pre_emphasis(audio);

        // Calculate number of frames
        let num_frames = if emphasized.len() > self.config.frame_size {
            (emphasized.len() - self.config.frame_size) / self.config.frame_shift + 1
        } else {
            1
        };

        // Extract features for each frame
        let mut features: Vec<Vec<f32>> = Vec::with_capacity(num_frames);

        for i in 0..num_frames {
            let start = i * self.config.frame_shift;
            let mut frame = self.extract_frame(&emphasized, start);
            let spectrum = self.compute_spectrum(&mut frame);
            let mel_energies = self.filterbank.apply(&spectrum);

            // Apply log compression
            let log_mel: Vec<f32> = mel_energies
                .iter()
                .map(|&e| (e + LOG_EPSILON).ln())
                .collect();

            // Apply DCT to get MFCC
            let mfcc = self.dct.apply(&log_mel);
            features.push(mfcc);
        }

        // Apply normalization
        self.normalize(&mut features);

        // Add delta features if configured
        if self.config.include_delta || self.config.include_delta_delta {
            let delta = self.compute_delta(&features, self.config.delta_window);

            if self.config.include_delta_delta {
                let delta_delta = self.compute_delta(&delta, self.config.delta_window);

                // Concatenate features
                for (i, frame) in features.iter_mut().enumerate() {
                    frame.extend_from_slice(&delta[i]);
                    frame.extend_from_slice(&delta_delta[i]);
                }
            } else {
                for (i, frame) in features.iter_mut().enumerate() {
                    frame.extend_from_slice(&delta[i]);
                }
            }
        }

        features
    }

    /// Extract power spectrogram from audio.
    ///
    /// # Arguments
    ///
    /// * `audio` - Raw audio samples (mono, at configured sample rate)
    ///
    /// # Returns
    ///
    /// Spectrogram `[num_frames, fft_size/2 + 1]`
    pub fn extract_spectrogram(&self, audio: &[f32]) -> Vec<Vec<f32>> {
        if audio.is_empty() {
            return vec![];
        }

        // Apply pre-emphasis
        let emphasized = self.apply_pre_emphasis(audio);

        // Calculate number of frames
        let num_frames = if emphasized.len() > self.config.frame_size {
            (emphasized.len() - self.config.frame_size) / self.config.frame_shift + 1
        } else {
            1
        };

        // Extract spectrogram
        let mut spectrogram: Vec<Vec<f32>> = Vec::with_capacity(num_frames);

        for i in 0..num_frames {
            let start = i * self.config.frame_shift;
            let mut frame = self.extract_frame(&emphasized, start);
            let spectrum = self.compute_spectrum(&mut frame);
            spectrogram.push(spectrum);
        }

        spectrogram
    }

    /// Extract log-mel spectrogram (commonly used for neural models).
    ///
    /// This is similar to filterbank features but without per-utterance normalization,
    /// making it more suitable for streaming applications.
    ///
    /// # Arguments
    ///
    /// * `audio` - Raw audio samples (mono, at configured sample rate)
    ///
    /// # Returns
    ///
    /// Log-mel spectrogram `[num_frames, num_mels]`
    pub fn extract_log_mel(&self, audio: &[f32]) -> Vec<Vec<f32>> {
        if audio.is_empty() {
            return vec![];
        }

        // Apply pre-emphasis
        let emphasized = self.apply_pre_emphasis(audio);

        // Calculate number of frames
        let num_frames = if emphasized.len() > self.config.frame_size {
            (emphasized.len() - self.config.frame_size) / self.config.frame_shift + 1
        } else {
            1
        };

        // Extract features for each frame
        let mut features: Vec<Vec<f32>> = Vec::with_capacity(num_frames);

        for i in 0..num_frames {
            let start = i * self.config.frame_shift;
            let mut frame = self.extract_frame(&emphasized, start);
            let spectrum = self.compute_spectrum(&mut frame);
            let mel_energies = self.filterbank.apply(&spectrum);

            // Apply log compression
            let log_mel: Vec<f32> = mel_energies
                .iter()
                .map(|&e| (e + LOG_EPSILON).ln())
                .collect();

            features.push(log_mel);
        }

        features
    }

    /// Get number of frames that would be extracted from audio of given length.
    pub fn num_frames(&self, audio_length: usize) -> usize {
        if audio_length <= self.config.frame_size {
            return if audio_length > 0 { 1 } else { 0 };
        }
        (audio_length - self.config.frame_size) / self.config.frame_shift + 1
    }

    /// Get the mel filterbank (for visualization or debugging).
    pub fn filterbank(&self) -> &MelFilterbank {
        &self.filterbank
    }
}

/// Streaming feature extractor for real-time applications.
///
/// Buffers audio samples and extracts features as frames become available.
pub struct StreamingFeatureExtractor {
    /// Base extractor.
    extractor: FeatureExtractor,

    /// Audio buffer.
    buffer: Vec<f32>,

    /// Number of samples processed (for tracking).
    samples_processed: usize,
}

impl StreamingFeatureExtractor {
    /// Create a new streaming feature extractor.
    pub fn new(config: FeatureConfig) -> Self {
        Self {
            extractor: FeatureExtractor::new(config),
            buffer: Vec::new(),
            samples_processed: 0,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &FeatureConfig {
        self.extractor.config()
    }

    /// Add audio samples to the buffer.
    ///
    /// Returns the number of complete frames available.
    pub fn add_samples(&mut self, samples: &[f32]) -> usize {
        self.buffer.extend_from_slice(samples);
        self.available_frames()
    }

    /// Get number of complete frames available in buffer.
    pub fn available_frames(&self) -> usize {
        self.extractor.num_frames(self.buffer.len())
    }

    /// Extract available frames as filterbank features.
    ///
    /// Consumes processed samples from the buffer.
    pub fn extract_filterbank(&mut self) -> Vec<Vec<f32>> {
        let num_frames = self.available_frames();
        if num_frames == 0 {
            return vec![];
        }

        // Extract features
        let features = self.extractor.extract_filterbank(&self.buffer);

        // Remove processed samples (keep overlap for next extraction)
        let consumed = num_frames * self.extractor.config.frame_shift;
        self.buffer.drain(..consumed);
        self.samples_processed += consumed;

        features
    }

    /// Extract available frames as MFCC features.
    pub fn extract_mfcc(&mut self) -> Vec<Vec<f32>> {
        let num_frames = self.available_frames();
        if num_frames == 0 {
            return vec![];
        }

        let features = self.extractor.extract_mfcc(&self.buffer);

        let consumed = num_frames * self.extractor.config.frame_shift;
        self.buffer.drain(..consumed);
        self.samples_processed += consumed;

        features
    }

    /// Flush remaining audio (for end of stream).
    ///
    /// Pads the buffer if necessary to extract final frame.
    pub fn flush_filterbank(&mut self) -> Vec<Vec<f32>> {
        if self.buffer.is_empty() {
            return vec![];
        }

        // Pad to minimum frame size
        while self.buffer.len() < self.extractor.config.frame_size {
            self.buffer.push(0.0);
        }

        let features = self.extractor.extract_filterbank(&self.buffer);
        self.buffer.clear();

        features
    }

    /// Flush remaining audio as MFCC.
    pub fn flush_mfcc(&mut self) -> Vec<Vec<f32>> {
        if self.buffer.is_empty() {
            return vec![];
        }

        while self.buffer.len() < self.extractor.config.frame_size {
            self.buffer.push(0.0);
        }

        let features = self.extractor.extract_mfcc(&self.buffer);
        self.buffer.clear();

        features
    }

    /// Reset the extractor state.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.samples_processed = 0;
    }

    /// Get total samples processed.
    pub fn samples_processed(&self) -> usize {
        self.samples_processed
    }

    /// Get current buffer length.
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_scale_conversion() {
        // Test Hz to Mel conversion
        assert!((MelFilterbank::hz_to_mel(0.0) - 0.0).abs() < 0.01);
        assert!((MelFilterbank::hz_to_mel(1000.0) - 1000.0).abs() < 50.0); // Roughly linear at 1kHz

        // Test round-trip
        for hz in [100.0, 500.0, 1000.0, 4000.0, 8000.0] {
            let mel = MelFilterbank::hz_to_mel(hz);
            let hz_back = MelFilterbank::mel_to_hz(mel);
            assert!((hz - hz_back).abs() < 0.1, "Round-trip failed for {}", hz);
        }
    }

    #[test]
    fn test_mel_filterbank_creation() {
        let fb = MelFilterbank::new(40, 512, 16000, 20.0, 8000.0);
        assert_eq!(fb.num_mels(), 40);

        // Each filter should have non-zero weights
        let dense = fb.to_dense();
        for (m, filter) in dense.iter().enumerate() {
            let sum: f32 = filter.iter().sum();
            assert!(sum > 0.0, "Filter {} has zero weights", m);
        }
    }

    #[test]
    fn test_filterbank_apply() {
        let fb = MelFilterbank::new(10, 512, 16000, 20.0, 8000.0);

        // Create a flat spectrum
        let spectrum = vec![1.0f32; 257]; // 512/2 + 1

        let mel_energies = fb.apply(&spectrum);
        assert_eq!(mel_energies.len(), 10);

        // All energies should be positive
        for e in &mel_energies {
            assert!(*e > 0.0);
        }
    }

    #[test]
    fn test_feature_config_defaults() {
        let config = FeatureConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.frame_size, 400);
        assert_eq!(config.frame_shift, 160);
        assert_eq!(config.num_mels, 40);
        assert_eq!(config.num_mfcc, 13);
    }

    #[test]
    fn test_feature_extractor_creation() {
        let config = FeatureConfig::default();
        let extractor = FeatureExtractor::new(config);
        assert_eq!(extractor.config().num_mels, 40);
    }

    #[test]
    fn test_extract_filterbank() {
        let config = FeatureConfig {
            normalize_mean: false,
            normalize_variance: false,
            ..Default::default()
        };
        let extractor = FeatureExtractor::new(config);

        // Create synthetic audio: 1 second at 16kHz
        let audio: Vec<f32> = (0..16000)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();

        let features = extractor.extract_filterbank(&audio);

        // Should have ~98 frames for 1 second: (16000 - 400) / 160 + 1 = 98
        assert!(!features.is_empty());
        assert!(features.len() >= 90 && features.len() <= 100);

        // Each frame should have 40 mel features
        assert_eq!(features[0].len(), 40);
    }

    #[test]
    fn test_extract_mfcc() {
        let config = FeatureConfig {
            normalize_mean: false,
            ..Default::default()
        };
        let extractor = FeatureExtractor::new(config);

        let audio: Vec<f32> = (0..16000)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();

        let features = extractor.extract_mfcc(&audio);

        assert!(!features.is_empty());
        // Each frame should have 13 MFCC
        assert_eq!(features[0].len(), 13);
    }

    #[test]
    fn test_extract_spectrogram() {
        let config = FeatureConfig::default();
        let fft_size = config.fft_size;
        let extractor = FeatureExtractor::new(config);

        let audio: Vec<f32> = (0..8000)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / 16000.0).sin())
            .collect();

        let spectrogram = extractor.extract_spectrogram(&audio);

        assert!(!spectrogram.is_empty());
        // FFT size / 2 + 1 = 256 (for fft_size=512)
        assert_eq!(spectrogram[0].len(), fft_size / 2 + 1);
    }

    #[test]
    fn test_delta_features() {
        let config = FeatureConfig {
            include_delta: true,
            include_delta_delta: true,
            normalize_mean: false,
            ..Default::default()
        };
        let extractor = FeatureExtractor::new(config);

        let audio: Vec<f32> = (0..16000)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();

        let features = extractor.extract_filterbank(&audio);

        // With delta and delta-delta: 40 + 40 + 40 = 120
        assert_eq!(features[0].len(), 120);
    }

    #[test]
    fn test_streaming_extractor() {
        let config = FeatureConfig::default();
        let mut streaming = StreamingFeatureExtractor::new(config);

        // Add audio in chunks
        let audio: Vec<f32> = (0..8000)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();

        // Add first half
        let frames1 = streaming.add_samples(&audio[..4000]);
        assert!(frames1 > 0);

        let features1 = streaming.extract_filterbank();
        assert_eq!(features1.len(), frames1);

        // Add second half
        let frames2 = streaming.add_samples(&audio[4000..]);
        assert!(frames2 > 0);

        let features2 = streaming.extract_filterbank();
        assert_eq!(features2.len(), frames2);

        // Flush remaining
        let final_features = streaming.flush_filterbank();
        // May have 0 or 1 final frame
        assert!(final_features.len() <= 1);
    }

    #[test]
    fn test_empty_audio() {
        let config = FeatureConfig::default();
        let extractor = FeatureExtractor::new(config);

        let features = extractor.extract_filterbank(&[]);
        assert!(features.is_empty());

        let mfcc = extractor.extract_mfcc(&[]);
        assert!(mfcc.is_empty());
    }

    #[test]
    fn test_short_audio() {
        let config = FeatureConfig::default();
        let extractor = FeatureExtractor::new(config);

        // Audio shorter than frame size
        let audio = vec![0.5f32; 200];

        let features = extractor.extract_filterbank(&audio);
        assert_eq!(features.len(), 1); // Should produce 1 frame
    }

    #[test]
    fn test_pre_emphasis() {
        let config = FeatureConfig {
            pre_emphasis: 0.97,
            ..Default::default()
        };
        let extractor = FeatureExtractor::new(config);

        let audio = vec![1.0f32; 100];
        let emphasized = extractor.apply_pre_emphasis(&audio);

        // First sample unchanged
        assert_eq!(emphasized[0], 1.0);
        // Subsequent samples should be smaller due to pre-emphasis
        assert!((emphasized[1] - 0.03).abs() < 0.01);
    }

    #[test]
    fn test_window_functions() {
        let size = 400;

        // Hanning should be 0 at edges
        let hanning = FeatureExtractor::build_window(size, WindowType::Hanning);
        assert!(hanning[0].abs() < 0.01);
        assert!(hanning[size - 1].abs() < 0.01);
        assert!((hanning[size / 2] - 1.0).abs() < 0.01); // Max at center

        // Hamming has non-zero edges
        let hamming = FeatureExtractor::build_window(size, WindowType::Hamming);
        assert!((hamming[0] - 0.08).abs() < 0.01);

        // Rectangular is all 1s
        let rect = FeatureExtractor::build_window(size, WindowType::Rectangular);
        assert!(rect.iter().all(|&w| (w - 1.0).abs() < 0.001));
    }

    #[test]
    fn test_num_frames_calculation() {
        let config = FeatureConfig::default();
        let extractor = FeatureExtractor::new(config);

        // 1 second at 16kHz
        let num_frames = extractor.num_frames(16000);
        // (16000 - 400) / 160 + 1 = 98.75 → 98
        assert_eq!(num_frames, 98);

        // Exact frame size
        let num_frames_single = extractor.num_frames(400);
        assert_eq!(num_frames_single, 1);

        // Empty
        assert_eq!(extractor.num_frames(0), 0);
    }

    #[test]
    fn test_telephony_config() {
        let config = FeatureConfig::telephony();
        assert_eq!(config.sample_rate, 8000);
        assert_eq!(config.high_freq, 4000.0);
    }

    #[test]
    fn test_music_config() {
        let config = FeatureConfig::music();
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.num_mels, 80);
    }
}
