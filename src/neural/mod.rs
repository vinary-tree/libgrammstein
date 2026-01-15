//! Neural rescoring module for ModernBERT-based language model scoring.
//!
//! This module provides neural rescoring capabilities using ModernBERT,
//! a 149M parameter model with 8K context length.
//!
//! # Features
//!
//! - **ModernBERT Rescoring**: Score candidate sentences using MLM pseudo-perplexity
//! - **Batch Scoring**: Efficient batch scoring of multiple candidates
//! - **Path Rescoring**: Rescore top-k paths from n-gram beam search
//! - **Document Embedding**: Generate embeddings for RAG retrieval
//! - **Extractive Summarization**: Select representative sentences
//! - **Code Embeddings**: Neural code embeddings (CodeT5+, UniXcoder, GraphCodeBERT)
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::neural::{ModernBertRescorer, Device};
//!
//! let rescorer = ModernBertRescorer::load("answerdotai/ModernBERT-base", Device::Cuda(0))?;
//! let score = rescorer.score_sentence("The quick brown fox jumps over the lazy dog.")?;
//! ```
//!
//! # Code Embeddings Example
//!
//! ```ignore
//! use libgrammstein::neural::code::{CodeT5Embedder, CodeLanguage, CodeEmbedder};
//!
//! let embedder = CodeT5Embedder::from_directory("/path/to/codet5p-110m")?;
//! let embedding = embedder.embed_code("fn main() {}", CodeLanguage::Rust)?;
//! ```

mod modernbert;
mod rescorer;
mod embedder;
mod summarizer;
mod cache;

#[cfg(feature = "code-neural")]
pub mod code;

pub use modernbert::{ModernBertModel, ModernBertConfig, Device};
pub use rescorer::{ModernBertRescorer, RescoringConfig, ScoredPath};
pub use embedder::{ModernBertEmbedder, EmbeddingConfig};
pub use summarizer::{Summarizer, SummarizerConfig, Synopsis, SynopsisSource};
pub use cache::{KvCache, CacheConfig};

/// Result type for neural module operations.
pub type Result<T> = std::result::Result<T, NeuralError>;

/// Error type for neural module operations.
#[derive(Debug, thiserror::Error)]
pub enum NeuralError {
    /// Model loading failed.
    #[error("Failed to load model: {0}")]
    ModelLoad(String),

    /// Tokenization failed.
    #[error("Tokenization error: {0}")]
    Tokenization(String),

    /// Inference failed.
    #[error("Inference error: {0}")]
    Inference(String),

    /// Device not available.
    #[error("Device not available: {0}")]
    DeviceNotAvailable(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Candle error.
    #[error("Candle error: {0}")]
    Candle(String),
}

impl From<candle_core::Error> for NeuralError {
    fn from(e: candle_core::Error) -> Self {
        NeuralError::Candle(e.to_string())
    }
}
