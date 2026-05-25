//! Programming language modeling for syntactic and semantic code correction.
//!
//! This module provides a framework for modeling programming languages with:
//!
//! - **Tree-sitter integration**: Incremental parsing with error recovery
//! - **Code Property Graphs**: AST + CFG + DFG unified representation
//! - **PCFG training**: Probabilistic context-free grammars from parsed corpora
//! - **Neural embeddings**: UniXcoder/GraphCodeBERT integration (with `code-neural` feature)
//! - **WFST export**: Grammar-weighted transducers for lling-llang integration
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Source Code                               │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │              Tree-sitter (incremental parsing)               │
//! │         AST with ERROR nodes for partial/invalid code        │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!          ┌───────────────────┼───────────────────┐
//!          ▼                   ▼                   ▼
//! ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
//! │  Token-Level    │ │   Structural    │ │    Semantic     │
//! │  Correction     │ │    Analysis     │ │    Analysis     │
//! │ (liblevenshtein)│ │   (CPG + GNN)   │ │  (UniXcoder)    │
//! └─────────────────┘ └─────────────────┘ └─────────────────┘
//!          │                   │                   │
//!          └───────────────────┼───────────────────┘
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    WFST Pipeline                             │
//! │    lexical ∘ grammar(PCFG) ∘ semantic(neural)                │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Example
//!
//! ```ignore
//! use libgrammstein::code::{CodeLanguage, Python, CodeCorpusReader};
//! use libgrammstein::code::pcfg::PcfgTrainer;
//!
//! // Create a Python language handler
//! let python = Python::new();
//!
//! // Parse some code
//! let tree = python.parse("def foo(x): return x + 1")?;
//!
//! // Train a PCFG from a corpus
//! let corpus = DirectoryCorpusReader::new("./python_corpus", &python)?;
//! let mut trainer = PcfgTrainer::new(&python);
//! trainer.train_from_corpus(&corpus)?;
//! let pcfg = trainer.to_weighted_cfg();
//! ```

pub mod ast;
pub mod constrained_decoding;
pub mod corpus;
pub mod correction;
pub mod correctors;
pub mod cpg;
pub mod gnn;
pub mod language;
pub mod pcfg;
pub mod pipeline;
pub mod subtree;
pub mod tokenizer;

#[cfg(feature = "code-neural")]
pub mod embeddings;

#[cfg(feature = "lling-llang-integration")]
pub mod wfst_export;

pub mod languages;

// Re-export core types
pub use ast::{byte_offset_to_position, AstError, AstNode, ParsedCode};
pub use constrained_decoding::{
    ConstrainedDecodingConfig, DecodingVocabulary, EarleyChart, EarleyParser, EarleyState,
    GrammarConstraint, TokenMask,
};
pub use corpus::{CodeCorpusReader, CodeSnippet};
pub use correction::{
    CodeCorrector, Correction, CorrectionCandidates, CorrectionKind, CorrectionSource,
};
pub use correctors::{EnsembleCorrector, GrammarCorrector, LexicalCorrector, SemanticCorrector};
pub use cpg::{CodePropertyGraph, CpgEdge, CpgEdgeKind, CpgNode, CpgNodeKind};
pub use gnn::{GnnConfig, GnnFeatures, GnnSemanticScorer, IssueType, SemanticIssue};
pub use language::{CodeLanguage, TokenContext, TokenType};
pub use pcfg::{PcfgTrainer, Production, WeightedCFG};
pub use pipeline::{
    AnalysisResult, CorrectionPipeline, Diagnostic, DiagnosticSeverity, PipelineConfig,
    PipelineError,
};
pub use subtree::{
    FlatNode, FlatTree, MiningResult, PatternNode, SubtreePattern, TreeminerConfig, TreeminerD,
};
pub use tokenizer::{CodeToken, CodeTokenizer};

#[cfg(feature = "code-neural")]
pub use embeddings::{CodeEmbedder, EmbeddingModel};

// Re-export language implementations when specific language features are enabled

// Mainstream languages
#[cfg(feature = "code-python")]
pub use languages::Python;

#[cfg(feature = "code-rust")]
pub use languages::Rust;

#[cfg(feature = "code-javascript")]
pub use languages::JavaScript;

// Domain-specific languages
#[cfg(feature = "code-rholang")]
pub use languages::Rholang;

#[cfg(feature = "code-metta")]
pub use languages::MeTTa;

// Re-export WFST types when integration is enabled
#[cfg(feature = "lling-llang-integration")]
pub use wfst_export::{PcfgScorer, PcfgWfstConfig, SymbolVocabulary};

#[cfg(feature = "lling-llang-integration")]
pub use wfst_export::PcfgWfstExport;
