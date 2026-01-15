//! Concrete corrector implementations for code correction.
//!
//! This module provides implementations of the `CodeCorrector` trait:
//!
//! - **Lexical**: Token-level fuzzy matching using liblevenshtein
//! - **Grammar**: PCFG-based correction with Earley parsing
//! - **Semantic**: GNN/embedding-based semantic analysis
//! - **Ensemble**: Combined scoring from all sources

pub mod lexical;
pub mod grammar;
pub mod semantic;
pub mod ensemble;

pub use lexical::LexicalCorrector;
pub use grammar::GrammarCorrector;
pub use semantic::SemanticCorrector;
pub use ensemble::EnsembleCorrector;
