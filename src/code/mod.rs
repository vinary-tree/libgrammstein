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

pub mod language;
pub mod tokenizer;
pub mod ast;
pub mod cpg;
pub mod pcfg;
pub mod corpus;
pub mod correction;
pub mod correctors;
pub mod gnn;
pub mod constrained_decoding;
pub mod pipeline;
pub mod subtree;

#[cfg(feature = "code-neural")]
pub mod embeddings;

#[cfg(feature = "lling-llang-integration")]
pub mod wfst_export;

pub mod languages;

// Re-export core types
pub use language::{CodeLanguage, TokenType, TokenContext};
pub use tokenizer::{CodeTokenizer, CodeToken};
pub use ast::{ParsedCode, AstNode, AstError, byte_offset_to_position};
pub use cpg::{CodePropertyGraph, CpgNode, CpgNodeKind, CpgEdge, CpgEdgeKind};
pub use pcfg::{WeightedCFG, Production, PcfgTrainer};
pub use corpus::{CodeCorpusReader, CodeSnippet};
pub use correction::{CodeCorrector, Correction, CorrectionKind, CorrectionSource, CorrectionCandidates};
pub use correctors::{LexicalCorrector, GrammarCorrector, SemanticCorrector, EnsembleCorrector};
pub use gnn::{GnnSemanticScorer, GnnConfig, GnnFeatures, SemanticIssue, IssueType};
pub use pipeline::{CorrectionPipeline, PipelineConfig, PipelineError, AnalysisResult, Diagnostic, DiagnosticSeverity};
pub use constrained_decoding::{
    GrammarConstraint, ConstrainedDecodingConfig, TokenMask, DecodingVocabulary,
    EarleyParser, EarleyState, EarleyChart,
};
pub use subtree::{TreeminerD, TreeminerConfig, MiningResult, FlatTree, FlatNode, SubtreePattern, PatternNode};

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
pub use wfst_export::{PcfgWfstConfig, SymbolVocabulary, PcfgScorer};

#[cfg(feature = "lling-llang-integration")]
pub use wfst_export::PcfgWfstExport;
