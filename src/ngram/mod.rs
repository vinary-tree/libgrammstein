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
//! # Key Encoding
//!
//! N-gram keys can be encoded in two ways:
//!
//! 1. **Legacy (pipe-separated)**: Simple `"the|quick|brown"` encoding. Deprecated
//!    because it can corrupt data if tokens contain `|`.
//!
//! 2. **Vocabulary-indexed (varint)**: Each word maps to a u64 index, encoded as
//!    LEB128 varint bytes stored as Latin-1 characters. This produces compact keys
//!    and supports unlimited vocabulary size. See [`vocabulary`] module.
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
pub mod metadata_filtering_zipper;
mod model;
pub mod smoothing;
mod trainer;
mod trie;
pub mod vocabulary;
pub mod vocabulary_indexed;

#[cfg(feature = "serde-extras")]
pub mod accumulator;

pub use entry::{NgramEntry, NgramEntrySnapshot};
pub use metadata_filtering_zipper::{MetadataFilteringZipper, METADATA_PREFIX};
pub use model::NgramModel;
#[cfg(feature = "serde-extras")]
pub use model::{PortableNgramModel, PortableVocabulary};
pub use trainer::{
    NgramTrainer, TrainerBuilder, TrainingConfig, TrainingProgress, TrainingStats, VocabularyMode,
};
#[allow(deprecated)]
pub use trie::{IterableDictionary, NgramTrie, NGRAM_SEPARATOR};
pub use vocabulary::{
    create_vocabulary, create_vocabulary_with_bloom, decode_ngram_key, decode_varint,
    encode_ngram_key, encode_ngram_key_batch, encode_ngram_key_existing, encode_varint,
    ngram_order, open_or_create_vocabulary, open_or_create_vocabulary_with_bloom, open_vocabulary,
    open_vocabulary_with_recovery, try_encode_ngram_key, try_encode_ngram_key_batch,
    DurabilityPolicy, PersistentVocabARTrie, RecoveryReport, SharedVocabARTrie, VocabSyncHandle,
    VocabularyError, VocabularyResult, FIRST_VALID_INDEX,
};
pub use vocabulary_indexed::{
    decode_key_to_indices, VocabularyIndexedDictionary, VocabularyIndexedNode,
};

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
/// use libdictenstein::dynamic_dawg::char::DynamicDawgChar;
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
    NgramModel<libdictenstein::dynamic_dawg::char::DynamicDawgChar<NgramEntry>>;

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
/// use libdictenstein::pathmap::PathMapDictionary;
///
/// let dictionary = PathMapDictionary::<NgramEntry>::new();
/// let model = TrainerBuilder::new(dictionary).order(5).train(reader)?;
///
/// // Use with lling-llang's LanguageModelLayer
/// let lm = GrammsteinLanguageModel::from_ngram(model);
/// ```
pub type PathMapNgramModel = NgramModel<libdictenstein::pathmap::PathMapDictionary<NgramEntry>>;
