//! Acoustic word embeddings for audio-to-embedding projection.
//!
//! This module provides fixed-dimensional embeddings of variable-length audio
//! segments, enabling:
//!
//! - **Query-by-example**: Find words similar to an audio query
//! - **Audio-text alignment**: Joint embedding space for speech and text
//! - **Acoustic similarity**: Compare pronunciation of words
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    Acoustic Word Embedding Pipeline                     │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │   Audio Frames    ──────►  Encoder  ──────►  Pooling  ──────►  Embedding
//! │   [T, F]                  (BiLSTM/      (Mean/Max/      [D]
//! │                            Transformer)  Attention)
//! │                                                                         │
//! │   Where:                                                                │
//! │     T = variable time steps                                             │
//! │     F = feature dimension (e.g., 40 filterbank)                        │
//! │     D = embedding dimension (e.g., 128)                                │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Integration with Text Embeddings
//!
//! Acoustic embeddings can be aligned with text embeddings for cross-modal retrieval:
//!
//! ```text
//! Audio: "hello" ──► [acoustic embedding] ──┐
//!                                           ├──► cosine similarity
//! Text:  "hello" ──► [text embedding]    ──┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::embedding::AcousticWordEmbedding;
//!
//! let awe = AcousticWordEmbedding::new(128);
//!
//! // Encode audio segment
//! let frames = vec![vec![0.0f32; 40]; 100]; // 100 frames of 40-dim features
//! let embedding = awe.encode(&frames);
//!
//! // Query similar words
//! let results = awe.query_by_example(&frames, 10);
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{Array1, Array2, Axis};
use ordered_float::OrderedFloat;

/// Pooling strategy for aggregating frame-level features.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Average all frame embeddings.
    Mean,
    /// Take maximum across frames for each dimension.
    Max,
    /// Use last frame embedding (for recurrent encoders).
    Last,
    /// Weighted average using attention mechanism.
    Attention,
    /// Concatenate mean and max pooling.
    MeanMax,
}

impl Default for PoolingStrategy {
    fn default() -> Self {
        Self::Mean
    }
}

/// Configuration for acoustic word embeddings.
#[derive(Clone, Debug)]
pub struct AcousticEmbeddingConfig {
    /// Output embedding dimension.
    pub embedding_dim: usize,

    /// Input feature dimension (e.g., 40 for filterbank).
    pub feature_dim: usize,

    /// Pooling strategy for variable-length sequences.
    pub pooling: PoolingStrategy,

    /// Whether to L2-normalize output embeddings.
    pub normalize: bool,

    /// Optional projection matrix dimensions for text alignment.
    pub text_projection_dim: Option<usize>,
}

impl Default for AcousticEmbeddingConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            feature_dim: 40,
            pooling: PoolingStrategy::Mean,
            normalize: true,
            text_projection_dim: None,
        }
    }
}

/// Core trait for acoustic encoders that process variable-length audio.
///
/// Implementations wrap neural network encoders (BiLSTM, Transformer, etc.)
/// that convert frame sequences to fixed-dimensional embeddings.
pub trait AcousticEncoder: Send + Sync {
    /// Encode a sequence of frames to frame-level representations.
    ///
    /// Input: `[num_frames, feature_dim]`
    /// Output: `[num_frames, hidden_dim]`
    fn encode_frames(&self, frames: &[Vec<f32>]) -> Vec<Vec<f32>>;

    /// Get the hidden dimension of frame-level outputs.
    fn hidden_dim(&self) -> usize;

    /// Get the expected input feature dimension.
    fn feature_dim(&self) -> usize;
}

/// Simple linear encoder for testing and baseline.
///
/// Projects input features through a linear layer.
#[derive(Clone, Debug)]
pub struct LinearEncoder {
    /// Projection matrix: [feature_dim, hidden_dim]
    weights: Array2<f32>,

    /// Bias: [hidden_dim]
    bias: Array1<f32>,
}

impl LinearEncoder {
    /// Create a new linear encoder with random initialization.
    pub fn new(feature_dim: usize, hidden_dim: usize) -> Self {
        // Xavier initialization
        let scale = (2.0 / (feature_dim + hidden_dim) as f32).sqrt();
        let weights = Array2::from_shape_fn((feature_dim, hidden_dim), |_| {
            (rand::random::<f32>() - 0.5) * 2.0 * scale
        });
        let bias = Array1::zeros(hidden_dim);

        Self { weights, bias }
    }

    /// Create from existing weights.
    pub fn from_weights(weights: Array2<f32>, bias: Array1<f32>) -> Self {
        Self { weights, bias }
    }
}

impl AcousticEncoder for LinearEncoder {
    fn encode_frames(&self, frames: &[Vec<f32>]) -> Vec<Vec<f32>> {
        frames
            .iter()
            .map(|frame| {
                let input = Array1::from_vec(frame.clone());
                let output = input.dot(&self.weights) + &self.bias;
                output.to_vec()
            })
            .collect()
    }

    fn hidden_dim(&self) -> usize {
        self.weights.ncols()
    }

    fn feature_dim(&self) -> usize {
        self.weights.nrows()
    }
}

/// Fixed-dimensional embedding of variable-length audio.
///
/// Combines an acoustic encoder with pooling to produce fixed-size embeddings
/// that can be used for similarity search and query-by-example.
pub struct AcousticWordEmbedding {
    /// The acoustic encoder.
    encoder: Arc<dyn AcousticEncoder>,

    /// Configuration.
    config: AcousticEmbeddingConfig,

    /// Optional projection matrix for text alignment: [hidden_dim, text_dim]
    text_projection: Option<Array2<f32>>,

    /// Word-to-embedding cache (for known words).
    word_cache: HashMap<String, Array1<f32>>,

    /// Audio-to-word index for query-by-example.
    /// Stores (word, embedding) pairs for nearest neighbor search.
    word_index: Vec<(String, Array1<f32>)>,
}

impl AcousticWordEmbedding {
    /// Create a new acoustic word embedding model.
    pub fn new(config: AcousticEmbeddingConfig) -> Self {
        let encoder = Arc::new(LinearEncoder::new(config.feature_dim, config.embedding_dim));
        Self::with_encoder(encoder, config)
    }

    /// Create with a custom encoder.
    pub fn with_encoder(encoder: Arc<dyn AcousticEncoder>, config: AcousticEmbeddingConfig) -> Self {
        let text_projection = config.text_projection_dim.map(|text_dim| {
            let hidden = encoder.hidden_dim();
            let scale = (2.0 / (hidden + text_dim) as f32).sqrt();
            Array2::from_shape_fn((hidden, text_dim), |_| {
                (rand::random::<f32>() - 0.5) * 2.0 * scale
            })
        });

        Self {
            encoder,
            config,
            text_projection,
            word_cache: HashMap::new(),
            word_index: Vec::new(),
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &AcousticEmbeddingConfig {
        &self.config
    }

    /// Get embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        if self.text_projection.is_some() {
            self.config.text_projection_dim.unwrap_or(self.encoder.hidden_dim())
        } else {
            self.encoder.hidden_dim()
        }
    }

    /// Encode an audio segment to a fixed-dimension embedding.
    ///
    /// # Arguments
    ///
    /// * `frames` - Sequence of frames: `[num_frames][feature_dim]`
    ///
    /// # Returns
    ///
    /// Fixed-dimension embedding vector.
    pub fn encode(&self, frames: &[Vec<f32>]) -> Vec<f32> {
        if frames.is_empty() {
            return vec![0.0; self.embedding_dim()];
        }

        // Get frame-level encodings
        let encoded = self.encoder.encode_frames(frames);

        // Apply pooling
        let pooled = self.apply_pooling(&encoded);

        // Apply text projection if configured
        let projected = if let Some(ref proj) = self.text_projection {
            pooled.dot(proj)
        } else {
            pooled
        };

        // Normalize if configured
        if self.config.normalize {
            let norm = projected.dot(&projected).sqrt();
            if norm > 1e-8 {
                (projected / norm).to_vec()
            } else {
                projected.to_vec()
            }
        } else {
            projected.to_vec()
        }
    }

    /// Apply pooling strategy to frame-level encodings.
    fn apply_pooling(&self, frames: &[Vec<f32>]) -> Array1<f32> {
        if frames.is_empty() {
            return Array1::zeros(self.encoder.hidden_dim());
        }

        let hidden_dim = frames[0].len();
        let num_frames = frames.len();

        match self.config.pooling {
            PoolingStrategy::Mean => {
                let mut sum = Array1::zeros(hidden_dim);
                for frame in frames {
                    sum += &Array1::from_vec(frame.clone());
                }
                sum / num_frames as f32
            }
            PoolingStrategy::Max => {
                let mut max = Array1::from_vec(frames[0].clone());
                for frame in frames.iter().skip(1) {
                    for (i, &v) in frame.iter().enumerate() {
                        if v > max[i] {
                            max[i] = v;
                        }
                    }
                }
                max
            }
            PoolingStrategy::Last => Array1::from_vec(frames[num_frames - 1].clone()),
            PoolingStrategy::Attention => {
                // Simple self-attention with learned query
                // For now, use uniform attention (equivalent to mean)
                let mut sum = Array1::zeros(hidden_dim);
                for frame in frames {
                    sum += &Array1::from_vec(frame.clone());
                }
                sum / num_frames as f32
            }
            PoolingStrategy::MeanMax => {
                // Concatenate mean and max
                let mut mean = Array1::zeros(hidden_dim);
                let mut max = Array1::from_vec(frames[0].clone());

                for frame in frames {
                    let arr = Array1::from_vec(frame.clone());
                    mean += &arr;
                    for (i, &v) in frame.iter().enumerate() {
                        if v > max[i] {
                            max[i] = v;
                        }
                    }
                }
                mean /= num_frames as f32;

                // Concatenate (note: this changes the embedding dimension)
                let mut concat = Vec::with_capacity(hidden_dim * 2);
                concat.extend(mean.iter().copied());
                concat.extend(max.iter().copied());
                Array1::from_vec(concat)
            }
        }
    }

    /// Compute similarity between two audio segments.
    pub fn audio_similarity(&self, audio1: &[Vec<f32>], audio2: &[Vec<f32>]) -> f64 {
        let emb1 = self.encode(audio1);
        let emb2 = self.encode(audio2);
        self.cosine_similarity(&emb1, &emb2)
    }

    /// Compute cosine similarity between two embeddings.
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f64 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a < 1e-8 || norm_b < 1e-8 {
            0.0
        } else {
            (dot / (norm_a * norm_b)) as f64
        }
    }

    /// Add a word to the index with its audio embedding.
    pub fn add_word(&mut self, word: &str, frames: &[Vec<f32>]) {
        let embedding = Array1::from_vec(self.encode(frames));
        self.word_cache.insert(word.to_string(), embedding.clone());
        self.word_index.push((word.to_string(), embedding));
    }

    /// Add a word with pre-computed embedding.
    pub fn add_word_embedding(&mut self, word: &str, embedding: Vec<f32>) {
        let arr = Array1::from_vec(embedding);
        self.word_cache.insert(word.to_string(), arr.clone());
        self.word_index.push((word.to_string(), arr));
    }

    /// Get embedding for a known word.
    pub fn get_word_embedding(&self, word: &str) -> Option<&Array1<f32>> {
        self.word_cache.get(word)
    }

    /// Query-by-example: find similar words in the index.
    ///
    /// # Arguments
    ///
    /// * `audio` - Query audio segment
    /// * `k` - Number of results to return
    ///
    /// # Returns
    ///
    /// Top-k (word, similarity) pairs sorted by decreasing similarity.
    pub fn query_by_example(&self, audio: &[Vec<f32>], k: usize) -> Vec<(String, f64)> {
        let query_emb = self.encode(audio);
        self.query_by_embedding(&query_emb, k)
    }

    /// Query by pre-computed embedding.
    pub fn query_by_embedding(&self, query_emb: &[f32], k: usize) -> Vec<(String, f64)> {
        let mut scores: Vec<(String, f64)> = self
            .word_index
            .iter()
            .map(|(word, emb)| {
                let sim = self.cosine_similarity(query_emb, emb.as_slice().unwrap());
                (word.clone(), sim)
            })
            .collect();

        // Sort by similarity (descending)
        scores.sort_by(|a, b| OrderedFloat(b.1).cmp(&OrderedFloat(a.1)));

        scores.into_iter().take(k).collect()
    }

    /// Get number of indexed words.
    pub fn index_size(&self) -> usize {
        self.word_index.len()
    }

    /// Clear the word index.
    pub fn clear_index(&mut self) {
        self.word_cache.clear();
        self.word_index.clear();
    }

    /// Compute pairwise similarities for all indexed words.
    pub fn all_pairwise_similarities(&self) -> Array2<f32> {
        let n = self.word_index.len();
        let mut sims = Array2::zeros((n, n));

        for i in 0..n {
            for j in i..n {
                let sim = self.cosine_similarity(
                    self.word_index[i].1.as_slice().unwrap(),
                    self.word_index[j].1.as_slice().unwrap(),
                ) as f32;
                sims[[i, j]] = sim;
                sims[[j, i]] = sim;
            }
        }

        sims
    }
}

/// Statistics about acoustic embeddings.
#[derive(Clone, Debug, Default)]
pub struct AcousticEmbeddingStats {
    /// Total words in index.
    pub num_words: usize,

    /// Total audio frames processed.
    pub total_frames: usize,

    /// Average embedding norm.
    pub avg_norm: f64,

    /// Average pairwise similarity.
    pub avg_similarity: f64,
}

impl AcousticWordEmbedding {
    /// Compute statistics about the current index.
    pub fn compute_stats(&self) -> AcousticEmbeddingStats {
        let num_words = self.word_index.len();

        if num_words == 0 {
            return AcousticEmbeddingStats::default();
        }

        // Compute average norm
        let avg_norm: f64 = self
            .word_index
            .iter()
            .map(|(_, emb)| emb.dot(emb).sqrt() as f64)
            .sum::<f64>()
            / num_words as f64;

        // Compute average pairwise similarity (for small indices)
        let avg_similarity = if num_words <= 1000 {
            let sims = self.all_pairwise_similarities();
            let total: f32 = sims.sum();
            let count = (num_words * num_words) as f32;
            (total / count) as f64
        } else {
            // Sample for large indices
            0.0
        };

        AcousticEmbeddingStats {
            num_words,
            total_frames: 0, // Would need to track this during encoding
            avg_norm,
            avg_similarity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_encoder() {
        let encoder = LinearEncoder::new(40, 128);

        assert_eq!(encoder.feature_dim(), 40);
        assert_eq!(encoder.hidden_dim(), 128);

        let frames = vec![vec![0.0f32; 40]; 10];
        let encoded = encoder.encode_frames(&frames);

        assert_eq!(encoded.len(), 10);
        assert_eq!(encoded[0].len(), 128);
    }

    #[test]
    fn test_acoustic_word_embedding_encode() {
        let config = AcousticEmbeddingConfig {
            embedding_dim: 64,
            feature_dim: 40,
            pooling: PoolingStrategy::Mean,
            normalize: true,
            text_projection_dim: None,
        };

        let awe = AcousticWordEmbedding::new(config);

        let frames = vec![vec![1.0f32; 40]; 20];
        let embedding = awe.encode(&frames);

        assert_eq!(embedding.len(), 64);

        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_pooling_strategies() {
        let config = AcousticEmbeddingConfig::default();
        let awe = AcousticWordEmbedding::new(config);

        let frames = vec![
            vec![1.0f32; 128],
            vec![2.0f32; 128],
            vec![3.0f32; 128],
        ];

        // Test mean pooling
        let mean = awe.apply_pooling(&frames);
        assert!((mean[0] - 2.0).abs() < 0.01);

        // Test max pooling
        let config_max = AcousticEmbeddingConfig {
            pooling: PoolingStrategy::Max,
            ..Default::default()
        };
        let awe_max = AcousticWordEmbedding::new(config_max);
        let max = awe_max.apply_pooling(&frames);
        assert!((max[0] - 3.0).abs() < 0.01);

        // Test last pooling
        let config_last = AcousticEmbeddingConfig {
            pooling: PoolingStrategy::Last,
            ..Default::default()
        };
        let awe_last = AcousticWordEmbedding::new(config_last);
        let last = awe_last.apply_pooling(&frames);
        assert!((last[0] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_audio_similarity() {
        let config = AcousticEmbeddingConfig::default();
        let awe = AcousticWordEmbedding::new(config);

        // Same audio should have high similarity
        let frames1 = vec![vec![1.0f32; 40]; 10];
        let sim_self = awe.audio_similarity(&frames1, &frames1);
        assert!(sim_self > 0.99);

        // Different audio may have lower similarity
        let frames2 = vec![vec![-1.0f32; 40]; 10];
        let sim_diff = awe.audio_similarity(&frames1, &frames2);
        assert!(sim_diff < sim_self);
    }

    #[test]
    fn test_query_by_example() {
        let config = AcousticEmbeddingConfig::default();
        let mut awe = AcousticWordEmbedding::new(config);

        // Add some words
        awe.add_word("hello", &vec![vec![1.0f32; 40]; 10]);
        awe.add_word("world", &vec![vec![2.0f32; 40]; 10]);
        awe.add_word("foo", &vec![vec![-1.0f32; 40]; 10]);

        assert_eq!(awe.index_size(), 3);

        // Query with similar audio to "hello"
        let query = vec![vec![1.0f32; 40]; 10];
        let results = awe.query_by_example(&query, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "hello"); // Most similar
    }

    #[test]
    fn test_empty_audio() {
        let config = AcousticEmbeddingConfig::default();
        let awe = AcousticWordEmbedding::new(config);

        let embedding = awe.encode(&[]);
        assert_eq!(embedding.len(), awe.embedding_dim());
    }

    #[test]
    fn test_word_embedding_cache() {
        let config = AcousticEmbeddingConfig::default();
        let mut awe = AcousticWordEmbedding::new(config);

        awe.add_word("test", &vec![vec![1.0f32; 40]; 5]);

        let emb = awe.get_word_embedding("test");
        assert!(emb.is_some());

        let emb_none = awe.get_word_embedding("missing");
        assert!(emb_none.is_none());
    }

    #[test]
    fn test_compute_stats() {
        let config = AcousticEmbeddingConfig::default();
        let mut awe = AcousticWordEmbedding::new(config);

        // Empty stats
        let stats_empty = awe.compute_stats();
        assert_eq!(stats_empty.num_words, 0);

        // Add words
        awe.add_word("a", &vec![vec![1.0f32; 40]; 5]);
        awe.add_word("b", &vec![vec![2.0f32; 40]; 5]);

        let stats = awe.compute_stats();
        assert_eq!(stats.num_words, 2);
        assert!(stats.avg_norm > 0.0);
    }

    #[test]
    fn test_text_projection() {
        let config = AcousticEmbeddingConfig {
            embedding_dim: 64,
            feature_dim: 40,
            text_projection_dim: Some(100), // Project to text embedding space
            ..Default::default()
        };

        let awe = AcousticWordEmbedding::new(config);

        let frames = vec![vec![1.0f32; 40]; 10];
        let embedding = awe.encode(&frames);

        // Embedding should be in projected dimension
        assert_eq!(embedding.len(), 100);
    }
}
