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
//! │ libdictenstein (+ liblevenshtein-rust)                       │
//! │   - DynamicDawgChar, PathMapDictionary, persistent ARTries   │
//! │   - MappedDictionary / MutableMappedDictionary trait bounds  │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Example Usage
//!
//! ```ignore
//! use libgrammstein::ngram::{NgramEntry, TrainerBuilder};
//! use libgrammstein::corpus::PlaintextReader;
//! use libdictenstein::dynamic_dawg::char::DynamicDawgChar;
//!
//! // Train a 5-gram Modified Kneser-Ney model over a trie backend.
//! // (Training goes through `TrainerBuilder`; `NgramModel` has no `train` constructor.)
//! let reader = PlaintextReader::from_file("corpus.txt")?;
//! let model = TrainerBuilder::new(DynamicDawgChar::<NgramEntry>::new())
//!     .order(5)
//!     .train(reader)?;
//!
//! // Query a log probability.
//! let log_prob = model.log_prob("world", &["hello"]);
//! println!("log P(world | hello) = {}", log_prob);
//! ```

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

/// bincode 1.x-compatible serialization shim over bincode 2.x (see the module
/// docs: the `legacy()` fixint-LE config keeps persisted model files readable).
///
/// Gated on the same features that make the optional `bincode` dependency
/// available, since the shim is a thin wrapper over it.
#[cfg(any(feature = "serde-extras", feature = "rag"))]
pub mod bincode_compat;
pub mod corpus;
pub mod dictionary;
pub mod embedding;
pub mod generation;
pub mod hybrid;
pub mod ngram;
pub mod scoring;
pub mod util;

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

#[cfg(feature = "google-books")]
pub mod aggregated;

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
        Serialization(#[from] crate::bincode_compat::Error),

        /// Serialization error (general, e.g., JSON).
        #[error("Serialization error: {0}")]
        SerializationMessage(String),

        /// Model construction / vocabulary consistency error (e.g. a term-id
        /// model loaded against a vocabulary whose fingerprint does not match).
        #[error("Model error: {0}")]
        Model(String),

        /// Neural model error.
        #[cfg(feature = "neural-rescore")]
        #[error("Neural error: {0}")]
        Neural(#[from] crate::neural::NeuralError),
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
        IndexBuilderConfig, LanguageTag, RagIndex, RagIndexConfig, RetrievalConfig,
        RetrievalResult, Retriever,
    };

    #[cfg(feature = "rag")]
    pub use crate::topic::{Topic, TopicConfig, TopicExtractor, TopicId, TopicModel};

    // Code module exports
    #[cfg(feature = "code")]
    pub use crate::code::{
        AnalysisResult,
        AstError,
        AstNode,
        // Corpus
        CodeCorpusReader,
        // Correction trait and types
        CodeCorrector,
        // Core traits and types
        CodeLanguage,
        // Code Property Graph
        CodePropertyGraph,
        CodeSnippet,
        CodeToken,
        // Tokenization
        CodeTokenizer,
        ConstrainedDecodingConfig,
        Correction,
        CorrectionCandidates,
        CorrectionKind,
        // End-to-end pipeline
        CorrectionPipeline,
        CorrectionSource,
        CpgEdge,
        CpgEdgeKind,
        CpgNode,
        CpgNodeKind,
        DecodingVocabulary,
        Diagnostic,
        DiagnosticSeverity,
        EnsembleCorrector,
        GnnConfig,
        GnnFeatures,
        // GNN semantic scoring
        GnnSemanticScorer,
        // Grammar-constrained decoding
        GrammarConstraint,
        GrammarCorrector,
        IssueType,
        // Concrete corrector implementations
        LexicalCorrector,
        // Parsing
        ParsedCode,
        PcfgTrainer,
        PipelineConfig,
        PipelineError,
        Production,
        SemanticCorrector,
        SemanticIssue,
        TokenContext,
        TokenMask,
        TokenType,
        // PCFG
        WeightedCFG,
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
        BraceKind,
        CommandEmbedding,
        ComponentScore,
        EquationDocument,
        EquationEmbedding,
        // Equation RAG
        EquationRagIndex,
        EquationRetrievalConfig,
        EquationRetriever,
        // Embeddings
        LaTeXEmbedder,
        LaTeXEmbeddingConfig,
        LaTeXMode,
        // N-gram models
        LaTeXNgramModel,
        // Neural rescoring
        LaTeXRescorer,
        // Combined scorer
        LaTeXScorer,
        LaTeXScorerBuilder,
        LaTeXToken,
        LaTeXTokenKind,
        // Tokenization
        LaTeXTokenizer,
        MathMode,
        ModeDetector,
        NgramConfig,
        RescoreResult,
        RescorerConfig,
        ScorerConfig,
        ScoringResult,
        TokenizerConfig,
    };
}
