//! Dictionary extraction and building for WFST text correction.
//!
//! This module provides functionality for extracting word frequency dictionaries
//! from corpora and building optimized spelling dictionaries for use with
//! WFST-based error correction systems.
//!
//! # Architecture
//!
//! The dictionary pipeline uses two trie implementations:
//!
//! - **Extraction**: Uses liblevenshtein's `PersistentARTrieChar` for concurrent
//!   word counting during corpus processing. This trie supports UTF-8 (char-based)
//!   nodes for multilingual support and atomic increment operations.
//!
//! - **Final Dictionary**: Converts to `DoubleArrayTrieChar` for fast read-only
//!   lookups during WFST rescoring. This representation is compact and optimized
//!   for prefix matching operations.
//!
//! # Vocabulary-Based Dictionary (Recommended)
//!
//! For n-gram language models using vocabulary-indexed encoding, the vocabulary
//! already contains all words and the n-gram storage contains unigram frequencies.
//! Use [`VocabularyDictionary`] for dictionary lookups in this case:
//!
//! ```ignore
//! use libgrammstein::dictionary::VocabularyDictionary;
//!
//! let dict = VocabularyDictionary::new(&vocabulary, &storage);
//!
//! // Check if word exists
//! if dict.contains("hello") {
//!     // Get unigram frequency
//!     let freq = dict.frequency("hello");
//! }
//! ```
//!
//! # Legacy Example
//!
//! For standalone dictionary extraction (without n-gram models):
//!
//! ```ignore
//! use libgrammstein::dictionary::{WordExtractor, DictionaryBuilder};
//! use libgrammstein::corpus::PlaintextReader;
//!
//! // Extract words from corpus
//! let mut extractor = WordExtractor::new();
//! let reader = PlaintextReader::from_file("corpus.txt")?;
//! for sentence in reader.sentences() {
//!     extractor.add_sentence(&sentence);
//! }
//!
//! // Build final dictionary with minimum frequency threshold
//! let dictionary = DictionaryBuilder::new()
//!     .min_frequency(5)
//!     .build_from_extractor(&extractor)?;
//!
//! // Save dictionary
//! dictionary.save("words.dict")?;
//! ```

mod builder;
mod extractor;
mod types;

#[cfg(feature = "google-books")]
mod vocabulary_backed;

pub use builder::{DictionaryBuilder, SpellingDictionary};
pub use extractor::{ExtractionConfig, WordExtractor};
pub use types::{DictionaryMetadata, DictionaryStats, WordEntry};

#[cfg(feature = "google-books")]
pub use vocabulary_backed::VocabularyDictionary;
