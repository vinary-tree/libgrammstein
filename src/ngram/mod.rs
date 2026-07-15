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
//! - Efficient probability queries via term-id trie navigation
//!
//! # Key Encoding
//!
//! There is exactly one live n-gram store, [`TermIdStore`]: n-grams are keyed as
//! **LEB128-varint-encoded `u64` term-id sequences stored as raw bytes** — the
//! same self-delimiting encoding the Google-Books count store uses. Each word
//! maps to a `u64` term-id (see the [`vocabulary`] module); the encoding is
//! compact (1-3 bytes for common ids) and delimiter-collision-free. No
//! `'|'`-joined string model exists; an old `'|'`-keyed portable file is re-keyed
//! onto term-ids by a single one-shot reader at load time.
//!
//! # Model Type Aliases
//!
//! - [`InMemoryTermIdModel`]: the byte-native model over an in-memory
//!   `DynamicDawg<NgramEntry>` — the training + inference target.
//! - [`PersistentTermIdModel`]: the byte-native model over a disk-backed
//!   `Arc<PersistentARTrie<NgramEntry>>`.
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
pub mod store;
mod tempdir;
mod trainer;
pub mod u64_view;
pub mod vocabulary;

#[cfg(test)]
mod migration_tests;

#[cfg(feature = "serde-extras")]
pub mod accumulator;

pub use entry::{NgramEntry, NgramEntrySnapshot};
pub use metadata_filtering_zipper::{MetadataFilteringZipper, METADATA_PREFIX};
pub use model::NgramModel;
#[cfg(feature = "serde-extras")]
pub use model::{KeyEncoding, PortableNgramModel, PortableVocabulary};
pub use store::{
    ByteMappedDictionary, IterableNgramStore, MutableByteMappedDictionary, MutableNgramStore,
    NgramLookup, TermIdStore,
};
pub use trainer::{
    NgramTrainer, TrainerBuilder, TrainingConfig, TrainingProgress, TrainingStats, VocabularyMode,
};
pub use u64_view::{U64NgramNode, U64NgramView, VarintByteUnit};
pub use vocabulary::{
    create_vocabulary, create_vocabulary_with_bloom, decode_ngram_key, decode_varint,
    encode_ngram_key, encode_ngram_key_batch, encode_ngram_key_existing, encode_varint,
    ngram_order, open_or_create_vocabulary, open_or_create_vocabulary_with_bloom, open_vocabulary,
    open_vocabulary_with_recovery, try_encode_ngram_key, try_encode_ngram_key_batch,
    DurabilityPolicy, PersistentVocabARTrie, RecoveryReport, SharedVocabARTrie, VocabSyncHandle,
    VocabularyError, VocabularyResult, FIRST_VALID_INDEX,
};

#[cfg(feature = "serde-extras")]
pub use accumulator::{AccumulatorError, AccumulatorResult, NgramAccumulator};

// Store-backend type aliases for common use cases

/// In-memory byte-native n-gram model — the natural training + inference target.
///
/// Backed by [`TermIdStore`] over an in-memory `DynamicDawg<NgramEntry>` (byte
/// unit): LEB128 term-id keys, 1× memory (the char-lift's ~4× gone), and
/// delimiter-collision-free. This is what [`TrainerBuilder::new`] with a
/// `DynamicDawg<NgramEntry>` backend produces.
///
/// # Example
///
/// ```ignore
/// use libgrammstein::ngram::{InMemoryTermIdModel, TrainerBuilder};
/// use libdictenstein::dynamic_dawg::DynamicDawg;
///
/// let model = TrainerBuilder::new(DynamicDawg::new()).order(5).train(reader)?;
/// model.save_portable("model.bin")?; // portable term-id format (+ vocabulary)
///
/// let model: InMemoryTermIdModel =
///     InMemoryTermIdModel::load_portable("model.bin", DynamicDawg::new)?;
/// ```
pub type InMemoryTermIdModel =
    NgramModel<TermIdStore<libdictenstein::dynamic_dawg::DynamicDawg<NgramEntry>>>;

/// Disk-backed byte-native n-gram model.
///
/// Backed by [`TermIdStore`] over a persistent `Arc<PersistentARTrie<NgramEntry>>`
/// (byte unit) — parallels the Google-Books count store and composes directly
/// with [`U64NgramView`] for fuzzy term-id correction.
pub type PersistentTermIdModel = NgramModel<
    TermIdStore<std::sync::Arc<libdictenstein::persistent_artrie::PersistentARTrie<NgramEntry>>>,
>;
