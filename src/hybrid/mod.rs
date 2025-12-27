//! Hybrid language model combining n-grams and embeddings.
//!
//! This module provides:
//! - Combined n-gram + embedding scoring
//! - OOV handling via embedding similarity
//! - Configurable interpolation strategies
//!
//! # Dictionary Backend Type Aliases
//!
//! Two type aliases are provided for common use cases:
//!
//! - [`SerializableHybridModel`]: Uses `DynamicDawgChar` backend for models that need
//!   to be saved/loaded. This backend supports full serde serialization.
//!
//! - [`PathMapHybridModel`]: Uses `PathMapDictionary` backend for integration with
//!   lling-llang's shared lattice architecture.
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

use crate::ngram::NgramEntry;

/// Serializable hybrid model using DynamicDawgChar backend.
///
/// Use this when you need to save/load models to/from disk.
/// This backend supports full serde serialization.
///
/// # Example
///
/// ```ignore
/// use libgrammstein::hybrid::SerializableHybridModel;
///
/// // Train and save
/// model.save("hybrid_model.bin")?;
///
/// // Load later
/// let model: SerializableHybridModel = SerializableHybridModel::load("hybrid_model.bin")?;
/// ```
pub type SerializableHybridModel =
    HybridLanguageModel<liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar<NgramEntry>>;

/// Memory-efficient hybrid model using PathMapDictionary backend.
///
/// Use this for lling-llang integration with shared lattice structures.
/// This backend does NOT support serde serialization but provides
/// better memory sharing characteristics.
pub type PathMapHybridModel =
    HybridLanguageModel<liblevenshtein::dictionary::pathmap::PathMapDictionary<NgramEntry>>;
