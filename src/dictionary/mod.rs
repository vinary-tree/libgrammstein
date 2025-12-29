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
//! # Example
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

mod types;
mod extractor;
mod builder;

pub use types::{DictionaryMetadata, WordEntry, DictionaryStats};
pub use extractor::{WordExtractor, ExtractionConfig};
pub use builder::{DictionaryBuilder, SpellingDictionary};
