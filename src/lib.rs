//! # libgrammstein
//!
//! A hybrid language model library combining N-gram models with subword embeddings,
//! designed for integration with WFST-based text correction systems.
//!
//! ## Overview
//!
//! libgrammstein provides:
//!
//! - **N-gram Language Models**: Modified Kneser-Ney smoothing with efficient trie-based storage
//! - **Subword Embeddings**: FastText-style embeddings with BPE tokenization
//! - **Hybrid Model**: Combines n-gram and embedding scores for robust OOV handling
//! - **WFST Integration**: Implements `LanguageModel` trait for lling-llang lattice rescoring
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │ lling-llang (WFST Framework)                                 │
//! │   - Lattices, CFG parsing, composition                       │
//! │   - LanguageModelLayer (uses trait below)                    │
//! └─────────────────────────────────────────────────────────────┘
//!                               │
//!                               │ implements LanguageModel trait
//!                               ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │ libgrammstein (this crate)                                   │
//! │   - NgramModel with Modified Kneser-Ney                      │
//! │   - SubwordEmbedding (FastText-style)                        │
//! │   - HybridLanguageModel                                      │
//! └─────────────────────────────────────────────────────────────┘
//!                               │
//!                               │ uses dictionary backends
//!                               ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │ liblevenshtein-rust                                          │
//! │   - DynamicDawgChar, PathMapDictionary                       │
//! │   - MutableMappedDictionary trait for n-gram storage         │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Example Usage
//!
//! ```ignore
//! use libgrammstein::ngram::NgramModel;
//! use libgrammstein::corpus::WikipediaReader;
//!
//! // Train n-gram model from Wikipedia
//! let reader = WikipediaReader::from_dump("enwiki-latest-pages-articles.xml.bz2")?;
//! let model = NgramModel::train(reader, 5)?;
//!
//! // Query log probability
//! let log_prob = model.log_prob("world", &["hello"]);
//! println!("log P(world | hello) = {}", log_prob);
//!
//! // Sentence scoring
//! let sentence_log_prob = model.sentence_log_prob(&["the", "quick", "brown", "fox"]);
//! ```

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

pub mod corpus;
pub mod embedding;
pub mod generation;
pub mod hybrid;
pub mod ngram;
pub mod scoring;

#[cfg(feature = "lling-llang-integration")]
pub mod integration;

/// Error types for libgrammstein operations.
pub mod error {
    use thiserror::Error;

    /// Main error type for libgrammstein operations.
    #[derive(Error, Debug)]
    pub enum Error {
        /// I/O error during corpus reading or model loading.
        #[error("I/O error: {0}")]
        Io(#[from] std::io::Error),

        /// XML parsing error (Wikipedia dump).
        #[error("XML parsing error: {0}")]
        Xml(#[from] quick_xml::Error),

        /// Invalid n-gram order (must be >= 1).
        #[error("Invalid n-gram order: {0} (must be >= 1)")]
        InvalidOrder(usize),

        /// Empty corpus provided for training.
        #[error("Empty corpus: no sentences found")]
        EmptyCorpus,

        /// Model not trained.
        #[error("Model not trained: {0}")]
        NotTrained(String),

        /// Serialization error.
        #[cfg(feature = "serde")]
        #[error("Serialization error: {0}")]
        Serialization(#[from] bincode::Error),
    }

    /// Result type alias for libgrammstein operations.
    pub type Result<T> = std::result::Result<T, Error>;
}

pub use error::{Error, Result};

/// Re-export commonly used types.
pub mod prelude {
    pub use crate::corpus::CorpusReader;
    pub use crate::error::{Error, Result};
    pub use crate::ngram::{NgramEntry, NgramModel};
    pub use crate::scoring::Perplexity;

    #[cfg(feature = "lling-llang-integration")]
    pub use crate::integration::GrammsteinLanguageModel;
}
