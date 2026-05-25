//! LaTeX-specific statistical and neural scoring for correction pipelines.
//!
//! This module provides specialized language models and scoring functions
//! for LaTeX documents, designed to integrate with the lling-llang WFST
//! correction framework.
//!
//! # Overview
//!
//! The LaTeX module consists of several components:
//!
//! - **Tokenizer** (`tokenizer`): LaTeX-aware tokenization that distinguishes between
//!   commands, environments, math mode, and text content.
//!
//! - **N-gram Model** (`ngram`): Mode-aware n-gram models trained separately on
//!   command sequences, mathematical expressions, and natural text.
//!
//! - **Embeddings** (`embedding`): LaTeX command and equation embeddings for
//!   semantic similarity scoring.
//!
//! - **Rescorer** (`rescorer`): Neural rescoring using fine-tuned ModernBERT
//!   for mathematical text.
//!
//! - **RAG** (`rag`): Equation retrieval for finding similar correct equations
//!   in a reference corpus.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                       LaTeX Scoring Pipeline                        │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                     │
//! │  Input: LaTeX Token Stream                                          │
//! │           │                                                         │
//! │           ▼                                                         │
//! │  ┌───────────────────┐                                              │
//! │  │   Mode Detector   │ ← Identifies command/math/text regions       │
//! │  └─────────┬─────────┘                                              │
//! │            │                                                        │
//! │  ┌─────────┴─────────┬───────────────────┬───────────────────┐     │
//! │  ▼                   ▼                   ▼                   ▼     │
//! │ Command N-gram    Math N-gram       Text N-gram       Equation     │
//! │ Model             Model             Model             Embeddings   │
//! │  │                   │                   │               │         │
//! │  └───────────────────┴───────────────────┴───────────────┘         │
//! │                              │                                      │
//! │                              ▼                                      │
//! │                    Combined Score (weighted)                        │
//! │                              │                                      │
//! │                              ▼                                      │
//! │                    Neural Rescore (optional)                        │
//! │                              │                                      │
//! │                              ▼                                      │
//! │                       Final Score                                   │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::latex::{LaTeXTokenizer, LaTeXNgramModel, LaTeXScorer};
//!
//! // Tokenize LaTeX input
//! let tokenizer = LaTeXTokenizer::new();
//! let tokens = tokenizer.tokenize(r"\begin{equation} x^2 + y^2 = z^2 \end{equation}");
//!
//! // Score with mode-aware n-gram model
//! let model = LaTeXNgramModel::load("latex_model.bin")?;
//! let score = model.score(&tokens);
//!
//! // Or use the combined scorer
//! let scorer = LaTeXScorer::builder()
//!     .with_ngram_model(model)
//!     .with_equation_embeddings(embeddings)
//!     .build();
//!
//! let final_score = scorer.score(&tokens);
//! ```
//!
//! # Training
//!
//! Models can be trained on arXiv LaTeX source files:
//!
//! ```ignore
//! use libgrammstein::latex::{LaTeXCorpusReader, LaTeXTrainer};
//!
//! let corpus = LaTeXCorpusReader::from_arxiv_bulk("/path/to/arxiv")?;
//! let trainer = LaTeXTrainer::new()
//!     .ngram_order(5)
//!     .math_weight(2.0)
//!     .command_weight(1.5);
//!
//! let model = trainer.train(corpus)?;
//! model.save("latex_model.bin")?;
//! ```

pub mod embedding;
pub mod ngram;
pub mod rag;
pub mod rescorer;
pub mod scorer;
pub mod tokenizer;

// Re-export main types
pub use tokenizer::{
    BraceKind, LaTeXToken, LaTeXTokenKind, LaTeXTokenizer, MathMode, TokenizerConfig,
};

pub use ngram::{LaTeXMode, LaTeXNgramModel, ModeDetector, NgramConfig};

pub use embedding::{
    CommandEmbedding, EmbeddingConfig as LaTeXEmbeddingConfig, EquationEmbedding, LaTeXEmbedder,
};

pub use rescorer::{LaTeXRescorer, RescoreResult, RescorerConfig};

pub use rag::{
    EquationDocument, EquationRagIndex, EquationRetriever,
    RetrievalConfig as EquationRetrievalConfig,
};

pub use scorer::{ComponentScore, LaTeXScorer, LaTeXScorerBuilder, ScorerConfig, ScoringResult};
