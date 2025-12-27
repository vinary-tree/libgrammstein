//! Hybrid language model combining n-grams and embeddings.
//!
//! This module provides:
//! - Combined n-gram + embedding scoring
//! - OOV handling via embedding similarity
//! - Configurable interpolation strategies
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::hybrid::{HybridLanguageModel, HybridConfig, InterpolationStrategy};
//! use libgrammstein::ngram::NgramModel;
//! use libgrammstein::embedding::SubwordEmbedding;
//!
//! // Create hybrid model from trained components
//! let config = HybridConfig {
//!     strategy: InterpolationStrategy::Linear { alpha: 0.8 },
//!     ..Default::default()
//! };
//! let hybrid = HybridLanguageModel::new(ngram_model, embedding_model, config);
//!
//! // Score a word in context
//! let score = hybrid.score("fox", &["the", "quick", "brown"]);
//!
//! // Compute perplexity
//! let ppl = hybrid.perplexity(&["the", "quick", "brown", "fox"]);
//! ```

mod model;
mod oov;

pub use model::{HybridConfig, HybridLanguageModel, InterpolationStrategy};
pub use oov::{OovHandler, OovStrategy};
