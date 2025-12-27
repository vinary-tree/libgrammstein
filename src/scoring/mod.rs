//! Perplexity and sentence scoring.
//!
//! This module provides:
//! - Sentence log-probability computation
//! - Corpus perplexity evaluation
//! - OOV rate calculation
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::scoring::Perplexity;
//! use libgrammstein::ngram::NgramModel;
//!
//! let model = NgramModel::load("model.bin")?;
//! let perplexity = Perplexity::new(&model);
//!
//! // Score a sentence
//! let log_prob = perplexity.sentence_log_prob(&["the", "quick", "brown", "fox"]);
//!
//! // Evaluate corpus perplexity
//! let corpus_ppl = perplexity.corpus_perplexity(&corpus_reader)?;
//! ```

mod perplexity;
mod sentence;

pub use perplexity::Perplexity;
pub use sentence::SentenceScorer;
