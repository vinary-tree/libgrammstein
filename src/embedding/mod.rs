//! Subword embeddings (FastText-style).
//!
//! This module provides:
//! - BPE tokenizer for subword segmentation
//! - Subword extraction for FastText-style embeddings
//! - Skip-gram training with negative sampling
//! - Word embedding lookup with subword fallback
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::embedding::{SubwordEmbedding, EmbeddingTrainerBuilder};
//! use libgrammstein::corpus::PlaintextReader;
//!
//! // Train embeddings from corpus
//! let reader = PlaintextReader::from_file("corpus.txt")?;
//! let model = EmbeddingTrainerBuilder::new()
//!     .dim(100)
//!     .window_size(5)
//!     .min_count(5)
//!     .epochs(5)
//!     .train(&reader)?;
//!
//! // Get word vectors
//! let vec = model.word_vector("hello");
//!
//! // Find similar words
//! let similar = model.most_similar("king", 10);
//!
//! // Compute analogy: "king" - "man" + "woman" ≈ "queen"
//! let results = model.analogy("man", "king", "woman", 5);
//! ```

mod bpe;
mod model;
mod trainer;

pub use bpe::{
    extract_subwords, hash_subword, BpeTokenizer, BpeTrainer, MergeOp,
    BPE_END_OF_WORD, BPE_UNKNOWN,
};

pub use model::{
    SubwordEmbedding, DEFAULT_BUCKET_COUNT, DEFAULT_EMBEDDING_DIM,
    DEFAULT_MAX_SUBWORD_LEN, DEFAULT_MIN_SUBWORD_LEN,
};

pub use trainer::{
    EmbeddingConfig, EmbeddingProgress, EmbeddingTrainer, EmbeddingTrainerBuilder,
};

// TODO: Implement in Phase 8
// mod simd;
// mod serialization;
