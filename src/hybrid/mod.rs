//! Hybrid language model combining n-grams and embeddings.
//!
//! This module provides:
//! - Combined n-gram + embedding scoring
//! - OOV handling via embedding similarity
//! - Configurable interpolation strategies
//!
//! # Store Backend Type Alias
//!
//! - [`InMemoryTermIdHybrid`]: the hybrid model over the byte-native
//!   [`TermIdStore`] on an in-memory `DynamicDawg<NgramEntry>` — the sole hybrid
//!   model type.
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

#[cfg(feature = "serde-extras")]
pub use model::PortableHybridModel;
pub use model::{HybridConfig, HybridLanguageModel, InterpolationStrategy};
pub use oov::{OovHandler, OovStrategy};

use crate::ngram::store::TermIdStore;
use crate::ngram::NgramEntry;

/// In-memory byte-native hybrid model — the sole hybrid model type.
///
/// Backed by [`TermIdStore`] over an in-memory `DynamicDawg<NgramEntry>`; its
/// portable format is the term-id encoding (with an embedded vocabulary), loaded
/// via [`HybridLanguageModel::load_portable`] with a `DynamicDawg::new` factory.
pub type InMemoryTermIdHybrid =
    HybridLanguageModel<TermIdStore<libdictenstein::dynamic_dawg::DynamicDawg<NgramEntry>>>;
