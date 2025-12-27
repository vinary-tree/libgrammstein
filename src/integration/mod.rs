//! Integration with lling-llang WFST framework.
//!
//! This module provides implementations of lling-llang's `LanguageModel` trait,
//! enabling libgrammstein models to be used for lattice rescoring in WFST-based
//! text correction pipelines.
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::integration::GrammsteinLanguageModel;
//! use libgrammstein::ngram::NgramModel;
//! use lling_llang::layers::LanguageModel;
//!
//! let ngram_model = NgramModel::load("model.bin")?;
//! let lm = GrammsteinLanguageModel::from_ngram(ngram_model);
//!
//! // Use with lling-llang's LanguageModelLayer
//! let score = lm.score_sequence(&["the", "quick", "brown", "fox"]);
//! ```

mod lling_llang;

pub use self::lling_llang::GrammsteinLanguageModel;
