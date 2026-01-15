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
pub mod dictionary;
pub mod embedding;
pub mod generation;
pub mod hybrid;
pub mod ngram;
pub mod scoring;

#[cfg(feature = "lling-llang-integration")]
pub mod integration;

#[cfg(feature = "cli")]
pub mod cli;

#[cfg(feature = "cli")]
pub mod language;

#[cfg(feature = "acoustic")]
pub mod acoustic;

#[cfg(feature = "neural-rescore")]
pub mod neural;

#[cfg(feature = "rag")]
pub mod rag;

#[cfg(feature = "rag")]
pub mod topic;

#[cfg(any(feature = "google-books", feature = "pdf-extraction"))]
pub mod sources;

#[cfg(feature = "code")]
pub mod code;

#[cfg(feature = "latex")]
pub mod latex;

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

        /// Serialization error (bincode).
        #[cfg(feature = "serde-extras")]
        #[error("Serialization error: {0}")]
        Serialization(#[from] bincode::Error),

        /// Serialization error (general, e.g., JSON).
        #[error("Serialization error: {0}")]
        SerializationMessage(String),
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

    #[cfg(feature = "acoustic")]
    pub use crate::acoustic::{
        FeatureConfig, FeatureExtractor, MelFilterbank, StreamingFeatureExtractor, WindowType,
    };

    #[cfg(feature = "candle-model")]
    pub use crate::acoustic::{
        AcousticModel, AcousticModelConfig, LinearAcousticModel, MockAcousticModel,
        TransformerAcousticModel,
    };

    #[cfg(feature = "neural-rescore")]
    pub use crate::neural::{
        Device, EmbeddingConfig, ModernBertConfig, ModernBertEmbedder, ModernBertModel,
        ModernBertRescorer, RescoringConfig, ScoredPath, Summarizer, SummarizerConfig, Synopsis,
        SynopsisSource,
    };

    #[cfg(feature = "rag")]
    pub use crate::rag::{
        Document, DocumentId, DocumentMetadata, ExactCosineBackend, IndexBuilder,
        IndexBuilderConfig, LanguageTag, RagIndex, RagIndexConfig, Retriever, RetrievalConfig,
        RetrievalResult,
    };

    #[cfg(feature = "rag")]
    pub use crate::topic::{Topic, TopicConfig, TopicExtractor, TopicId, TopicModel};

    // Code module exports
    #[cfg(feature = "code")]
    pub use crate::code::{
        // Core traits and types
        CodeLanguage, TokenType, TokenContext,
        // Tokenization
        CodeTokenizer, CodeToken,
        // Parsing
        ParsedCode, AstNode, AstError,
        // Code Property Graph
        CodePropertyGraph, CpgNode, CpgNodeKind, CpgEdge, CpgEdgeKind,
        // PCFG
        WeightedCFG, Production, PcfgTrainer,
        // Corpus
        CodeCorpusReader, CodeSnippet,
        // Correction trait and types
        CodeCorrector, Correction, CorrectionKind, CorrectionSource, CorrectionCandidates,
        // Concrete corrector implementations
        LexicalCorrector, GrammarCorrector, SemanticCorrector, EnsembleCorrector,
        // End-to-end pipeline
        CorrectionPipeline, PipelineConfig, PipelineError, AnalysisResult, Diagnostic, DiagnosticSeverity,
        // GNN semantic scoring
        GnnSemanticScorer, GnnConfig, GnnFeatures, SemanticIssue, IssueType,
        // Grammar-constrained decoding
        GrammarConstraint, ConstrainedDecodingConfig, TokenMask, DecodingVocabulary,
    };

    // Language-specific re-exports
    #[cfg(feature = "code-python")]
    pub use crate::code::Python;

    #[cfg(feature = "code-rust")]
    pub use crate::code::Rust;

    #[cfg(feature = "code-javascript")]
    pub use crate::code::JavaScript;

    #[cfg(feature = "code-rholang")]
    pub use crate::code::Rholang;

    #[cfg(feature = "code-metta")]
    pub use crate::code::MeTTa;

    // LaTeX module exports
    #[cfg(feature = "latex")]
    pub use crate::latex::{
        // Tokenization
        LaTeXTokenizer, LaTeXToken, LaTeXTokenKind, TokenizerConfig, MathMode, BraceKind,
        // N-gram models
        LaTeXNgramModel, ModeDetector, LaTeXMode, NgramConfig,
        // Embeddings
        LaTeXEmbedder, CommandEmbedding, EquationEmbedding, LaTeXEmbeddingConfig,
        // Neural rescoring
        LaTeXRescorer, RescorerConfig, RescoreResult,
        // Equation RAG
        EquationRagIndex, EquationDocument, EquationRetriever, EquationRetrievalConfig,
        // Combined scorer
        LaTeXScorer, LaTeXScorerBuilder, ScorerConfig, ScoringResult, ComponentScore,
    };
}
