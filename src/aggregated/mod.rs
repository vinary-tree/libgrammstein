//! Aggregated Dictionary abstraction over sharded language models.
//!
//! This module provides [`AggregatedLanguageModelDictionary`], which offers a unified
//! [`Dictionary`](libdictenstein::Dictionary) view over multiple sharded n-gram tries
//! with transparent vocabulary encoding/decoding.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │            AggregatedLanguageModelDictionary            │
//! │            (this module)                                │
//! ├─────────────────────────────────────────────────────────┤
//! │  - vocabulary: SharedVocabARTrie  ← SHARED across  │
//! │  - coordinator: Arc<ShardCoordinator>     all shards   │
//! │  - config: AggregationConfig                           │
//! └──────────────────────────┬──────────────────────────────┘
//!                            │
//!                            │  Routes by first token prefix
//!                            │  (a-z for 1-grams, aa-zz for 2-5 grams)
//!                            ▼
//!     ┌──────────────────────┴──────────────────────┐
//!     │              ShardCoordinator               │
//!     │  ┌─────────┐ ┌─────────┐     ┌─────────┐   │
//!     │  │Shard "a"│ │Shard "b"│ ... │Shard "zz"│  │
//!     │  └────┬────┘ └────┬────┘     └────┬────┘   │
//!     │       │           │               │        │
//!     │       ▼           ▼               ▼        │
//!     │  DiskBacked  DiskBacked      DiskBacked   │
//!     │  CharTrie    CharTrie        CharTrie     │
//!     │  (vocab-     (vocab-         (vocab-      │
//!     │   indexed     indexed         indexed     │
//!     │   keys)       keys)           keys)       │
//!     └────────────────────────────────────────────┘
//! ```
//!
//! # Sharding + Vocabulary Encoding Interaction
//!
//! The sharding strategy and vocabulary encoding work together at different layers:
//!
//! 1. **Routing layer** (ShardCoordinator): Routes n-grams based on **original token strings**
//!    - Example: `["the", "quick", "brown"]` → routes to shard based on first token "the"
//!    - Routing key: `"th"` (first two chars of first token)
//!
//! 2. **Storage layer** (per-shard trie): Stores **vocabulary-indexed keys**
//!    - After routing, each shard encodes the n-gram using vocabulary indices
//!    - Example: `["the", "quick", "brown"]` → indices `[1, 2, 3]` → varint key `"\x01\x02\x03"`
//!    - The vocabulary is **shared** across all shards for consistent encoding
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::aggregated::AggregatedLanguageModelDictionary;
//! use libgrammstein::sources::google_books::sharding::ShardCoordinator;
//! use libgrammstein::ngram::vocabulary::SharedVocabulary;
//!
//! // Load existing sharded language model
//! let config = ShardConfig::new("english_shards");
//! let coordinator = ShardCoordinator::open(config)?;
//! let vocabulary = SharedVocabulary::open(&vocab_path)?;
//!
//! // Create aggregated dictionary
//! let dict = AggregatedLanguageModelDictionary::new(
//!     Arc::new(coordinator),
//!     Arc::new(vocabulary),
//! );
//!
//! // Query (transparent routing + vocabulary encoding)
//! if let Some(count) = dict.get_ngram(&["the", "quick", "brown"]) {
//!     println!("n-gram count: {}", count);
//! }
//! ```

use crate::ngram::vocabulary::{
    encode_ngram_key_bytes, encode_ngram_key_existing_bytes, SharedVocabARTrie,
};
use crate::sources::google_books::sharding::coordinator::ShardCoordinator;
#[allow(deprecated)]
use liblevenshtein::dictionary::{
    Dictionary, DictionaryNode, MappedDictionary, MappedDictionaryNode, SyncStrategy,
};
use std::sync::Arc;

// Note: SharedVocabulary provides the word-to-index mapping methods directly.
// We use it without requiring any trait abstraction.

// ============================================================================
// AggregationConfig
// ============================================================================

/// Configuration for aggregated dictionary behavior.
#[derive(Debug, Clone)]
pub struct AggregationConfig {
    /// Default n-gram delimiter (typically space).
    pub delimiter: char,
    /// Maximum shards to keep open simultaneously.
    pub max_open_shards: usize,
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            delimiter: ' ',
            max_open_shards: 100,
        }
    }
}

// ============================================================================
// AggregatedLanguageModelDictionary
// ============================================================================

/// Unified Dictionary view over multiple sharded n-gram tries.
///
/// This struct provides transparent:
/// - **Routing**: Directs queries to the appropriate shard based on n-gram prefix
/// - **Vocabulary encoding**: Converts words to vocabulary indices automatically
/// - **Shard management**: Lazy loading with configurable limits
///
/// # Thread Safety
///
/// This type is `Send + Sync`. The underlying `ShardCoordinator` and `SharedVocabulary`
/// handle their own synchronization.
pub struct AggregatedLanguageModelDictionary {
    /// Shared vocabulary for word ↔ index mapping.
    vocabulary: SharedVocabARTrie,
    /// Shard coordinator for routing and storage.
    coordinator: Arc<ShardCoordinator>,
    /// Aggregation configuration.
    config: AggregationConfig,
}

impl std::fmt::Debug for AggregatedLanguageModelDictionary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AggregatedLanguageModelDictionary")
            .field("vocabulary_size", &self.vocabulary.len())
            .field("open_shards", &self.coordinator.open_shard_count())
            .field("config", &self.config)
            .finish()
    }
}

impl AggregatedLanguageModelDictionary {
    /// Create a new aggregated dictionary with default configuration.
    ///
    /// # Arguments
    ///
    /// * `coordinator` - The shard coordinator managing n-gram storage
    /// * `vocabulary` - The shared vocabulary for word-to-index mapping
    pub fn new(coordinator: Arc<ShardCoordinator>, vocabulary: SharedVocabARTrie) -> Self {
        Self {
            vocabulary,
            coordinator,
            config: AggregationConfig::default(),
        }
    }

    /// Create a new aggregated dictionary with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `coordinator` - The shard coordinator managing n-gram storage
    /// * `vocabulary` - The shared vocabulary for word-to-index mapping
    /// * `config` - Aggregation configuration
    pub fn with_config(
        coordinator: Arc<ShardCoordinator>,
        vocabulary: SharedVocabARTrie,
        config: AggregationConfig,
    ) -> Self {
        Self {
            vocabulary,
            coordinator,
            config,
        }
    }

    /// Get the vocabulary.
    pub fn vocabulary(&self) -> &SharedVocabARTrie {
        &self.vocabulary
    }

    /// Get the shard coordinator.
    pub fn coordinator(&self) -> &Arc<ShardCoordinator> {
        &self.coordinator
    }

    /// Get the aggregation configuration.
    pub fn config(&self) -> &AggregationConfig {
        &self.config
    }

    /// Get the delimiter used for splitting terms.
    pub fn delimiter(&self) -> char {
        self.config.delimiter
    }

    /// Split a term into tokens using the configured delimiter.
    fn split_term<'a>(&self, term: &'a str) -> Vec<&'a str> {
        term.split(self.config.delimiter).collect()
    }

    // ========================================================================
    // N-gram specific methods
    // ========================================================================

    /// Check if an n-gram exists in the sharded store.
    ///
    /// Routes to the appropriate shard based on the first token's prefix.
    /// Returns `false` if any word is OOV (out of vocabulary).
    pub fn contains_ngram(&self, words: &[&str]) -> bool {
        if words.is_empty() {
            return false;
        }

        // Encode using existing vocabulary only (OOV → None)
        let encoded_key = match encode_ngram_key_existing_bytes(words, &self.vocabulary) {
            Some(k) => k,
            None => return false,
        };

        // Route to shard based on token strings
        let shard_key = self.coordinator.route_tokens(words);

        // Check in shard
        self.coordinator
            .get_in_shard(&shard_key, &encoded_key)
            .is_some()
    }

    /// Get the count associated with an n-gram.
    ///
    /// Routes to the appropriate shard based on the first token's prefix.
    /// Returns `None` if the n-gram doesn't exist or any word is OOV.
    pub fn get_ngram(&self, words: &[&str]) -> Option<u64> {
        if words.is_empty() {
            return None;
        }

        // Encode using existing vocabulary only
        let encoded_key = encode_ngram_key_existing_bytes(words, &self.vocabulary)?;

        // Route to shard
        let shard_key = self.coordinator.route_tokens(words);

        // Lookup in shard
        self.coordinator.get_in_shard(&shard_key, &encoded_key)
    }

    /// Insert an n-gram with a count.
    ///
    /// New words are automatically added to the vocabulary.
    ///
    /// # Returns
    ///
    /// `Ok(true)` if this is a new n-gram, `Ok(false)` if updating an existing one.
    /// Returns error if the shard operation fails.
    pub fn insert_ngram(
        &self,
        words: &[&str],
        count: u64,
    ) -> Result<bool, crate::sources::google_books::sharding::coordinator::CoordinatorError> {
        if words.is_empty() {
            return Ok(false);
        }

        // Encode with vocabulary acquisition
        let encoded_key = encode_ngram_key_bytes(words, &self.vocabulary);

        // Route to shard
        let shard_key = self.coordinator.route_tokens(words);

        // Store in shard
        self.coordinator
            .store_in_shard(&shard_key, &encoded_key, count)
    }

    /// Get the total number of n-grams across all shards.
    pub fn total_ngram_count(&self) -> u64 {
        self.coordinator.total_entry_count()
    }

    /// Decode an encoded key back to word indices.
    ///
    /// Useful for debugging and reverse lookups.
    pub fn decode_key(&self, key: &[u8]) -> Vec<u64> {
        crate::ngram::vocabulary::decode_ngram_key_bytes(key)
    }

    /// Build a reverse vocabulary map (index → word).
    ///
    /// Useful for decoding n-gram keys back to human-readable form.
    ///
    /// Note: Prefer using `vocabulary().get_term(index)` for O(1) lookups
    /// instead of building a full HashMap when only a few lookups are needed.
    pub fn build_reverse_vocabulary(&self) -> std::collections::HashMap<u64, String> {
        let guard = self.vocabulary.read();
        let len = guard.len();
        let mut map = std::collections::HashMap::with_capacity(len);
        for i in 1..=(len as u64) {
            if let Some(term) = guard.get_term(i) {
                map.insert(i, term);
            }
        }
        map
    }

    /// Checkpoint all shards and vocabulary.
    pub fn checkpoint(&self) -> Result<(), String> {
        // Checkpoint vocabulary first
        self.vocabulary
            .write()
            .checkpoint()
            .map_err(|e| format!("Vocabulary checkpoint failed: {}", e))?;

        // Checkpoint all shards
        self.coordinator
            .checkpoint_all()
            .map_err(|e| format!("Shard checkpoint failed: {:?}", e))?;

        Ok(())
    }

    // ========================================================================
    // Iteration methods (metadata-filtered)
    // ========================================================================

    /// Iterate all n-grams (excluding metadata) with decoded word sequences.
    ///
    /// This method iterates over all loaded shards and decodes vocabulary-indexed
    /// n-gram keys back to human-readable word sequences. Metadata entries
    /// (prefixed with `\x00`) are automatically filtered out.
    ///
    /// # Returns
    ///
    /// An iterator yielding `(Vec<String>, u64)` pairs where:
    /// - `Vec<String>` is the n-gram word sequence
    /// - `u64` is the count for that n-gram
    ///
    /// # Note
    ///
    /// This method collects all n-grams from currently loaded shards. If you
    /// need complete coverage, call `coordinator.load_all_shards()` first
    /// (not exposed publicly - intended for internal use).
    ///
    /// # Example
    ///
    /// ```ignore
    /// for (words, count) in dict.iter_ngrams()? {
    ///     println!("{:?} -> {}", words, count);
    /// }
    /// ```
    pub fn iter_ngrams(&self) -> Result<impl Iterator<Item = (Vec<String>, u64)> + '_, String> {
        let reverse_vocab = self.build_reverse_vocabulary();

        // Collect all n-gram entries from all open shards
        let mut all_entries: Vec<(Vec<u8>, u64)> = Vec::new();

        for shard_key in self.coordinator.open_shard_keys() {
            if let Ok(shard) = self.coordinator.get_or_create_shard(&shard_key) {
                let guard = shard.read();
                // Use iter_with_counts which already filters checkpoint metadata
                if let Ok(entries) = guard.iter_with_counts() {
                    all_entries.extend(entries);
                }
            }
        }

        // Filter and decode entries
        Ok(all_entries
            .into_iter()
            .filter(|(key, _)| !key.starts_with(&[0x00]))
            .filter_map(move |(key, count)| {
                // Decode the vocabulary-indexed key back to words
                let indices = crate::ngram::vocabulary::decode_ngram_key_bytes(&key);
                let words: Option<Vec<String>> = indices
                    .into_iter()
                    .map(|idx| reverse_vocab.get(&idx).cloned())
                    .collect();
                words.map(|w| (w, count))
            }))
    }

    /// Iterate all n-grams (excluding metadata) as raw encoded keys with counts.
    ///
    /// This is a lower-level method that returns the varint-encoded keys
    /// without decoding them. Useful for bulk operations where decoding
    /// overhead should be avoided.
    ///
    /// # Returns
    ///
    /// An iterator yielding `(Vec<u8>, u64)` pairs where:
    /// - `Vec<u8>` is the varint-encoded n-gram key (raw bytes)
    /// - `u64` is the count for that n-gram
    pub fn iter_ngrams_raw(&self) -> Result<impl Iterator<Item = (Vec<u8>, u64)> + '_, String> {
        // Collect all n-gram entries from all open shards
        let mut all_entries: Vec<(Vec<u8>, u64)> = Vec::new();

        for shard_key in self.coordinator.open_shard_keys() {
            if let Ok(shard) = self.coordinator.get_or_create_shard(&shard_key) {
                let guard = shard.read();
                // Use iter_with_counts which already filters checkpoint metadata
                if let Ok(entries) = guard.iter_with_counts() {
                    all_entries.extend(entries);
                }
            }
        }

        // Filter out metadata entries
        Ok(all_entries
            .into_iter()
            .filter(|(key, _)| !key.starts_with(&[0x00])))
    }

    /// Get the number of n-grams in all open shards (excluding metadata).
    ///
    /// This is a cached count from the coordinator, which may not account
    /// for metadata entries. For an exact count excluding metadata, use
    /// `iter_ngrams_raw().count()`.
    pub fn ngram_count(&self) -> u64 {
        self.coordinator.total_entry_count()
    }
}

// ============================================================================
// Dictionary Trait Implementation
// ============================================================================

impl Dictionary for AggregatedLanguageModelDictionary {
    type Node = AggregatedDictionaryNode;

    fn root(&self) -> Self::Node {
        // The root node represents a view into all shards
        // Character-level traversal would need to fan out across shards
        AggregatedDictionaryNode {
            // Root node - no specific shard yet
            _phantom: std::marker::PhantomData,
        }
    }

    fn contains(&self, term: &str) -> bool {
        // Split by delimiter and check as n-gram
        let words = self.split_term(term);
        let refs: Vec<&str> = words.iter().map(|s| *s).collect();
        self.contains_ngram(&refs)
    }

    fn len(&self) -> Option<usize> {
        // Total across all shards
        Some(self.coordinator.total_entry_count() as usize)
    }

    fn is_empty(&self) -> bool {
        self.coordinator.total_entry_count() == 0
    }

    fn sync_strategy(&self) -> SyncStrategy {
        SyncStrategy::InternalSync
    }
}

// ============================================================================
// MappedDictionary Trait Implementation
// ============================================================================

impl MappedDictionary for AggregatedLanguageModelDictionary {
    type Value = u64;

    fn get_value(&self, term: &str) -> Option<Self::Value> {
        let words = self.split_term(term);
        let refs: Vec<&str> = words.iter().map(|s| *s).collect();
        self.get_ngram(&refs)
    }

    fn contains_with_value<F>(&self, term: &str, predicate: F) -> bool
    where
        F: Fn(&Self::Value) -> bool,
    {
        self.get_value(term).is_some_and(|v| predicate(&v))
    }
}

// ============================================================================
// AggregatedDictionaryNode
// ============================================================================

/// Node for aggregated dictionary traversal.
///
/// Note: Character-level traversal over an aggregated sharded dictionary is
/// complex because it requires fanning out across multiple shards. This
/// implementation provides basic functionality; for Levenshtein automaton
/// compatibility, use individual shard-level traversal via
/// [`VocabularyIndexedDictionary`](libdictenstein::VocabularyIndexedDictionary).
#[derive(Clone)]
pub struct AggregatedDictionaryNode {
    _phantom: std::marker::PhantomData<()>,
}

impl std::fmt::Debug for AggregatedDictionaryNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AggregatedDictionaryNode").finish()
    }
}

impl DictionaryNode for AggregatedDictionaryNode {
    type Unit = char;

    fn is_final(&self) -> bool {
        // Aggregated node traversal is limited
        // For full traversal, use individual shard access
        false
    }

    fn transition(&self, _label: Self::Unit) -> Option<Self> {
        // Character-level transitions across aggregated shards
        // would require complex fanout logic
        None
    }

    fn edges(&self) -> Box<dyn Iterator<Item = (Self::Unit, Self)> + '_> {
        // No edges at aggregated level - use shard-specific access
        Box::new(std::iter::empty())
    }
}

impl MappedDictionaryNode for AggregatedDictionaryNode {
    type Value = u64;

    fn value(&self) -> Option<Self::Value> {
        None
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full integration tests require setting up ShardCoordinator and SharedVocabulary
    // with actual file paths. Unit tests here focus on the vocabulary provider trait.

    #[test]
    fn test_aggregation_config_default() {
        let config = AggregationConfig::default();
        assert_eq!(config.delimiter, ' ');
        assert_eq!(config.max_open_shards, 100);
    }

    #[test]
    fn test_shared_vocabulary_methods() {
        // Verify SharedVocabARTrie has the methods we need
        // This test verifies the API we depend on exists
        fn _check_api(v: &SharedVocabARTrie) {
            let _: libdictenstein::persistent_artrie::error::Result<u64> =
                v.write().insert("word");
            let _: Option<u64> = v.read().get_index("word");
            let _: bool = v.read().contains("word");
            let _: usize = v.read().len();
            let _: bool = v.read().is_empty();
        }
    }

    #[test]
    fn test_aggregated_node_is_placeholder() {
        let node = AggregatedDictionaryNode {
            _phantom: std::marker::PhantomData,
        };

        // Aggregated node is a placeholder
        assert!(!node.is_final());
        assert!(node.transition('a').is_none());
        assert_eq!(node.edges().count(), 0);
        assert!(node.value().is_none());
    }
}
