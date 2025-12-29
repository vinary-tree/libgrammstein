//! N-gram language model with Modified Kneser-Ney smoothing.
//!
//! This module provides a complete n-gram language model implementation that uses
//! liblevenshtein-rust's dictionary backends for efficient storage and retrieval.
//!
//! # Overview
//!
//! The n-gram model supports:
//! - Orders 1-5 (unigrams through 5-grams)
//! - Modified Kneser-Ney smoothing for probability estimation
//! - Streaming corpus training with Rayon parallelism
//! - Efficient probability queries via trie navigation
//!
//! # Dictionary Backend Type Aliases
//!
//! Two type aliases are provided for common use cases:
//!
//! - [`SerializableNgramModel`]: Uses `DynamicDawgChar` backend for models that need
//!   to be saved/loaded. This backend supports full serde serialization.
//!
//! - [`PathMapNgramModel`]: Uses `PathMapDictionary` backend for integration with
//!   lling-llang's shared lattice architecture. This backend does NOT support serde
//!   serialization but provides better memory sharing characteristics.
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::ngram::NgramModel;
//! use libgrammstein::corpus::PlaintextReader;
//!
//! let reader = PlaintextReader::from_directory("corpus/")?;
//! let model = NgramModel::train(reader, 3)?; // trigram model
//!
//! let log_prob = model.log_prob("fox", &["quick", "brown"]);
//! ```

mod entry;
mod model;
pub mod smoothing;
mod trainer;
mod trie;

#[cfg(feature = "serde-extras")]
pub mod accumulator;

pub use entry::{NgramEntry, NgramEntrySnapshot};
pub use model::NgramModel;
#[cfg(feature = "serde-extras")]
pub use model::PortableNgramModel;
pub use trainer::{NgramTrainer, TrainerBuilder, TrainingConfig, TrainingProgress, TrainingStats};
pub use trie::{IterableDictionary, NgramTrie, NGRAM_SEPARATOR};

#[cfg(feature = "serde-extras")]
pub use accumulator::{AccumulatorError, AccumulatorResult, NgramAccumulator};

// Dictionary backend type aliases for common use cases

/// Serializable n-gram model using DynamicDawgChar backend.
///
/// Use this when you need to save/load models to/from disk.
/// This backend supports full serde serialization.
///
/// # Example
///
/// ```ignore
/// use libgrammstein::ngram::SerializableNgramModel;
/// use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
///
/// // Train and save
/// let dictionary = DynamicDawgChar::<NgramEntry>::new();
/// let model = TrainerBuilder::new(dictionary).order(5).train(reader)?;
/// model.save("model.bin")?;
///
/// // Load later
/// let model: SerializableNgramModel = SerializableNgramModel::load("model.bin")?;
/// ```
pub type SerializableNgramModel =
    NgramModel<liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar<NgramEntry>>;

/// Memory-efficient n-gram model using PathMapDictionary backend.
///
/// Use this for lling-llang integration with shared lattice structures.
/// This backend does NOT support serde serialization but provides
/// better memory sharing characteristics.
///
/// # Example
///
/// ```ignore
/// use libgrammstein::ngram::PathMapNgramModel;
/// use liblevenshtein::dictionary::pathmap::PathMapDictionary;
///
/// let dictionary = PathMapDictionary::<NgramEntry>::new();
/// let model = TrainerBuilder::new(dictionary).order(5).train(reader)?;
///
/// // Use with lling-llang's LanguageModelLayer
/// let lm = GrammsteinLanguageModel::from_ngram(model);
/// ```
pub type PathMapNgramModel =
    NgramModel<liblevenshtein::dictionary::pathmap::PathMapDictionary<NgramEntry>>;
