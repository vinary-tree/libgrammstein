//! Topic extraction and modeling using BERTopic-like algorithms.
//!
//! This module provides automatic topic extraction from document collections
//! using hierarchical agglomerative clustering and c-TF-IDF keyword extraction.
//!
//! # Features
//!
//! - **Hierarchical Clustering**: Ward linkage agglomerative clustering on document embeddings
//! - **c-TF-IDF Keywords**: Class-based TF-IDF for extracting representative keywords per topic
//! - **Dendrogram Navigation**: Explore topic hierarchy at different levels
//! - **Checkpointing**: Save/resume long-running topic extraction
//! - **Lock-free Parallelization**: Non-blocking algorithms using atomics
//!
//! # Architecture
//!
//! The topic extraction pipeline:
//!
//! 1. **Distance Matrix**: Compute pairwise cosine distances between document embeddings
//! 2. **Clustering**: Hierarchical agglomerative clustering to group similar documents
//! 3. **Dendrogram**: Build hierarchical tree structure from clustering
//! 4. **c-TF-IDF**: Extract representative keywords for each topic cluster
//! 5. **Summarization**: Generate natural language descriptions
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::topic::{TopicExtractor, TopicConfig};
//! use libgrammstein::rag::RagIndex;
//!
//! // After building a RAG index with documents...
//! let config = TopicConfig::default();
//! let extractor = TopicExtractor::new(config);
//!
//! // Extract topics from the index
//! let topic_model = extractor.extract(&index)?;
//!
//! // Explore topics
//! for topic in topic_model.leaf_topics() {
//!     println!("{}: {}", topic.id, topic.keyword_summary(5));
//! }
//! ```
//!
//! # Thread Safety
//!
//! All public APIs are designed to be thread-safe:
//! - Distance matrix computation uses lock-free atomic operations
//! - Vocabulary building uses atomic counters
//! - Topic model is immutable after extraction
//!
//! # Checkpointing
//!
//! Long-running extractions can be checkpointed:
//!
//! ```ignore
//! let extractor = TopicExtractor::new(config.with_checkpointing(100));
//! let model = extractor.extract_with_checkpoints(&index, "checkpoints/topic")?;
//!
//! // Resume from checkpoint if interrupted
//! let model = TopicExtractor::resume("checkpoints/topic")?;
//! ```

mod checkpoint;
mod clustering;
mod config;
mod ctfidf;
mod dendrogram;
mod extractor;
mod model;
pub mod paradigm;
mod summarizer;
mod types;

pub use checkpoint::*;
pub use clustering::*;
pub use config::*;
pub use ctfidf::*;
pub use dendrogram::*;
pub use extractor::*;
pub use model::*;
pub use paradigm::{
    ApiPattern, ApiPatternConfig, ApiPatternMiner, DetectionResult, DomainPatternDetector,
    IndicatorCategory, IndicatorMatch, LanguageHints, MettaPattern, MettaPatternCatalog,
    MettaPatternCategory, MettaPatternMatch, MiningStats, Paradigm, ParadigmConfig,
    ParadigmDetector, ParadigmIndicator, ParadigmProfile, ParadigmWeights, RholangPattern,
    RholangPatternCatalog, RholangPatternCategory, RholangPatternMatch,
};
pub use summarizer::*;
pub use types::*;

use thiserror::Error;

/// Result type for topic operations.
pub type Result<T> = std::result::Result<T, TopicError>;

/// Errors that can occur during topic extraction and modeling.
#[derive(Error, Debug)]
pub enum TopicError {
    /// Error during clustering.
    #[error("Clustering error: {0}")]
    ClusteringError(String),

    /// Error during c-TF-IDF computation.
    #[error("c-TF-IDF error: {0}")]
    CtfidfError(String),

    /// Error during summarization.
    #[error("Summarization error: {0}")]
    SummarizationError(String),

    /// Not enough documents for topic extraction.
    #[error("Not enough documents: need at least {minimum}, have {actual}")]
    InsufficientDocuments {
        /// Minimum number of documents required.
        minimum: usize,
        /// Actual number of documents supplied.
        actual: usize,
    },

    /// Embedding dimension mismatch.
    #[error("Embedding dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected embedding dimension.
        expected: usize,
        /// Actual embedding dimension.
        actual: usize,
    },

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Checkpoint error.
    #[error("Checkpoint error: {0}")]
    CheckpointError(String),

    /// Paradigm detection error.
    #[error("Paradigm detection error: {0}")]
    ParadigmError(String),

    /// IO error.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

impl From<crate::bincode_compat::Error> for TopicError {
    fn from(err: crate::bincode_compat::Error) -> Self {
        TopicError::SerializationError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = TopicError::InsufficientDocuments {
            minimum: 10,
            actual: 5,
        };
        assert!(err.to_string().contains("10"));
        assert!(err.to_string().contains("5"));
    }

    #[test]
    fn test_config_exports() {
        // Verify configs are exported
        let _ = TopicConfig::default();
        let _ = ClusteringConfig::default();
        let _ = CtfidfConfig::default();
    }

    #[test]
    fn test_topic_exports() {
        // Verify topic types are exported
        let id = TopicId::new(0);
        assert_eq!(id.as_u32(), 0);
    }
}
