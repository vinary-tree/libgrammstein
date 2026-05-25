//! Concrete corrector implementations for code correction.
//!
//! This module provides implementations of the `CodeCorrector` trait:
//!
//! - **Lexical**: Token-level fuzzy matching using liblevenshtein
//! - **Grammar**: PCFG-based correction with Earley parsing
//! - **Semantic**: GNN/embedding-based semantic analysis
//! - **Ensemble**: Combined scoring from all sources

pub mod ensemble;
pub mod grammar;
pub mod lexical;
pub mod semantic;

pub use ensemble::EnsembleCorrector;
pub use grammar::GrammarCorrector;
pub use lexical::LexicalCorrector;
pub use semantic::SemanticCorrector;
