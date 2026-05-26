//! N-gram trie wrapper over liblevenshtein dictionary backends.
//!
//! This module provides a high-level interface for storing and querying n-grams
//! using liblevenshtein-rust's dictionary implementations.
//!
//! # Key Encoding
//!
//! N-gram keys can be encoded in two ways:
//!
//! 1. **Legacy (pipe-separated)**: `"the|quick|brown"` - Simple but vulnerable to
//!    corruption if tokens contain the pipe character.
//!
//! 2. **Vocabulary-indexed (PUA)**: Each word maps to a Unicode Private Use Area
//!    character, and n-gram keys are sequences of these characters. This eliminates
//!    the delimiter bug entirely.
//!
//! New code should use vocabulary-indexed encoding via [`crate::ngram::vocabulary`].

use super::entry::NgramEntry;
use liblevenshtein::dictionary::MutableMappedDictionary;
use std::marker::PhantomData;
use std::sync::Arc;

/// Trait for dictionaries that support iteration over (key, value) pairs.
///
/// This is used for portable serialization, allowing models to be saved
/// and loaded without requiring the dictionary to implement serde traits.
pub trait IterableDictionary: MutableMappedDictionary<Value = NgramEntry> {
    /// Iterate over all (key, value) pairs in the dictionary.
    fn iter_all(&self) -> Box<dyn Iterator<Item = (String, NgramEntry)> + '_>;
}

// Implement IterableDictionary for DynamicDawgChar
impl IterableDictionary
    for liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar<NgramEntry>
{
    fn iter_all(&self) -> Box<dyn Iterator<Item = (String, NgramEntry)> + '_> {
        Box::new(self.iter())
    }
}

// Implement IterableDictionary for PathMapDictionary
impl IterableDictionary for liblevenshtein::dictionary::pathmap::PathMapDictionary<NgramEntry> {
    fn iter_all(&self) -> Box<dyn Iterator<Item = (String, NgramEntry)> + '_> {
        Box::new(self.iter())
    }
}

// Implement IterableDictionary for the disk-backed char ARTrie (shared handle).
// `SharedCharARTrie<V> = Arc<RwLock<PersistentARTrieChar<V>>>` already implements
// `MutableMappedDictionary<Value = NgramEntry>` (libdictenstein
// persistent_artrie_char/mod.rs:718), so the supertrait holds; this adds the
// portable-serialization iteration hook so the type can back HybridLanguageModel /
// NgramModel / TrainerBuilder.
impl IterableDictionary for libdictenstein::persistent_artrie_char::SharedCharARTrie<NgramEntry> {
    fn iter_all(&self) -> Box<dyn Iterator<Item = (String, NgramEntry)> + '_> {
        // `iter_with_values()` borrows the RwLock read guard; materialize into an
        // owned Vec so the returned iterator does not borrow a dropped guard. The
        // collect is load-bearing (lifetime detach), not a `needless_collect`.
        let entries: Vec<(String, NgramEntry)> = self.read().iter_with_values().collect();
        Box::new(entries.into_iter())
    }
}

// Implement IterableDictionary for the vocabulary-indexed wrapper. The backend `D`
// stores varint-encoded latin1 keys; reconstruct each word string by decoding the
// key to vocabulary indices and reverse-looking-up each index, joined with the
// wrapper's delimiter (pgmcp pins it to '|' to match LEGACY_NGRAM_SEPARATOR, so the
// portable keys round-trip through NgramTrie's legacy split/join). A key whose index
// is missing from the vocabulary is skipped defensively (no panic).
impl<D> IterableDictionary for super::vocabulary_indexed::VocabularyIndexedDictionary<D>
where
    D: IterableDictionary,
{
    fn iter_all(&self) -> Box<dyn Iterator<Item = (String, NgramEntry)> + '_> {
        let delimiter = self.delimiter().to_string();
        // Hold one vocab read guard across all reverse lookups.
        let guard = self.vocabulary().read();
        let decoded: Vec<(String, NgramEntry)> = self
            .backend()
            .iter_all()
            .filter_map(|(key, entry)| {
                let indices = super::vocabulary_indexed::decode_key_to_indices(&key);
                if indices.is_empty() {
                    return None;
                }
                let mut words = Vec::with_capacity(indices.len());
                for idx in indices {
                    words.push(guard.get_term(idx)?);
                }
                Some((words.join(&delimiter), entry))
            })
            .collect();
        drop(guard);
        Box::new(decoded.into_iter())
    }
}

/// Separator used between tokens in legacy n-gram keys.
///
/// # Deprecation Notice
///
/// This encoding scheme is deprecated because it can cause silent data corruption
/// if a token contains the pipe character. For example:
///
/// ```text
/// ["foo|bar", "baz"] → "foo|bar|baz" → ["foo", "bar", "baz"]  // WRONG!
/// ```
///
/// Use [`crate::ngram::vocabulary`] for the new vocabulary-indexed encoding
/// that avoids this issue.
#[deprecated(
    since = "0.3.0",
    note = "Use vocabulary-indexed encoding via crate::ngram::vocabulary instead. \
            Pipe-separated keys can corrupt data if tokens contain '|'."
)]
pub const NGRAM_SEPARATOR: char = '|';

// Re-export the non-deprecated version for internal use during migration
pub(crate) const LEGACY_NGRAM_SEPARATOR: char = '|';

/// N-gram trie wrapper providing high-level n-gram operations.
///
/// Wraps a dictionary backend (like `DynamicDawgChar` or `PathMapDictionary`)
/// to provide n-gram-specific operations like key encoding and batch updates.
///
/// # Type Parameters
///
/// * `D` - The dictionary backend type, must implement `MutableMappedDictionary<Value = NgramEntry>`
///
/// # Example
///
/// ```ignore
/// use libgrammstein::ngram::{NgramTrie, NgramEntry};
/// use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
///
/// let dict = DynamicDawgChar::<NgramEntry>::new();
/// let trie = NgramTrie::new(dict);
///
/// trie.insert(&["the", "quick", "brown"]);
/// assert_eq!(trie.get(&["the", "quick", "brown"]).map(|e| e.count()), Some(1));
/// ```
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(bound = "D: serde::Serialize + serde::de::DeserializeOwned")]
pub struct NgramTrie<D>
where
    D: MutableMappedDictionary<Value = NgramEntry>,
{
    /// The underlying dictionary backend.
    dictionary: Arc<D>,

    /// Maximum n-gram order stored in this trie.
    max_order: usize,

    /// Phantom data for type parameter.
    #[serde(skip)]
    _marker: PhantomData<D>,
}

impl<D> NgramTrie<D>
where
    D: MutableMappedDictionary<Value = NgramEntry>,
{
    /// Create a new n-gram trie wrapping the given dictionary.
    pub fn new(dictionary: D, max_order: usize) -> Self {
        Self {
            dictionary: Arc::new(dictionary),
            max_order,
            _marker: PhantomData,
        }
    }

    /// Create from an existing Arc-wrapped dictionary.
    pub fn from_arc(dictionary: Arc<D>, max_order: usize) -> Self {
        Self {
            dictionary,
            max_order,
            _marker: PhantomData,
        }
    }

    /// Get the maximum n-gram order.
    #[inline]
    pub fn max_order(&self) -> usize {
        self.max_order
    }

    /// Get a reference to the underlying dictionary.
    #[inline]
    pub fn dictionary(&self) -> &D {
        &self.dictionary
    }

    /// Get a clone of the Arc-wrapped dictionary.
    #[inline]
    pub fn dictionary_arc(&self) -> Arc<D> {
        Arc::clone(&self.dictionary)
    }

    /// Encode an n-gram as a dictionary key using legacy pipe-separated format.
    ///
    /// # Deprecation Notice
    ///
    /// This function is deprecated because pipe-separated encoding can cause
    /// silent data corruption if tokens contain the pipe character.
    ///
    /// Use [`crate::ngram::vocabulary::encode_ngram_key`] for the new
    /// vocabulary-indexed encoding.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let key = NgramTrie::<D>::encode_key(&["the", "quick", "brown"]);
    /// assert_eq!(key, "the|quick|brown");
    /// ```
    #[inline]
    #[deprecated(
        since = "0.3.0",
        note = "Use vocabulary::encode_ngram_key() instead. \
                Pipe-separated keys can corrupt data if tokens contain '|'."
    )]
    pub fn encode_key(tokens: &[&str]) -> String {
        Self::encode_key_legacy(tokens)
    }

    /// Encode an n-gram as a dictionary key using legacy pipe-separated format.
    ///
    /// This is the internal implementation used during migration. New code should
    /// use [`crate::ngram::vocabulary::encode_ngram_key`] instead.
    #[inline]
    pub(crate) fn encode_key_legacy(tokens: &[&str]) -> String {
        tokens.join(&LEGACY_NGRAM_SEPARATOR.to_string())
    }

    /// Insert or increment an n-gram count.
    ///
    /// If the n-gram exists, increments its count. Otherwise, inserts it with count 1.
    ///
    /// # Note
    ///
    /// This method uses legacy pipe-separated encoding. For vocabulary-indexed
    /// encoding, use [`Self::insert_with_key`] with a key from
    /// [`crate::ngram::vocabulary::encode_ngram_key`].
    ///
    /// # Returns
    ///
    /// `true` if this was a new n-gram (inserted), `false` if it already existed (incremented).
    pub fn insert(&self, tokens: &[&str]) -> bool {
        let key = Self::encode_key_legacy(tokens);
        self.dictionary
            .update_or_insert(&key, NgramEntry::new(1), |entry| entry.increment())
    }

    /// Insert or increment an n-gram using a pre-encoded key.
    ///
    /// Use this with vocabulary-indexed keys from [`crate::ngram::vocabulary::encode_ngram_key`].
    ///
    /// # Returns
    ///
    /// `true` if this was a new n-gram (inserted), `false` if it already existed (incremented).
    pub fn insert_with_key(&self, key: &str) -> bool {
        self.dictionary
            .update_or_insert(key, NgramEntry::new(1), |entry| entry.increment())
    }

    /// Insert an n-gram with a specific count.
    ///
    /// # Note
    ///
    /// This method uses legacy pipe-separated encoding. For vocabulary-indexed
    /// encoding, use [`Self::insert_with_key_and_count`].
    pub fn insert_with_count(&self, tokens: &[&str], count: u64) -> bool {
        let key = Self::encode_key_legacy(tokens);
        self.dictionary
            .insert_with_value(&key, NgramEntry::new(count))
    }

    /// Insert an n-gram with a specific count using a pre-encoded key.
    pub fn insert_with_key_and_count(&self, key: &str, count: u64) -> bool {
        self.dictionary
            .insert_with_value(key, NgramEntry::new(count))
    }

    /// Get the entry for an n-gram, if it exists.
    ///
    /// # Note
    ///
    /// This method uses legacy pipe-separated encoding. For vocabulary-indexed
    /// encoding, use [`Self::get_by_key`].
    pub fn get(&self, tokens: &[&str]) -> Option<NgramEntry> {
        let key = Self::encode_key_legacy(tokens);
        self.dictionary.get_value(&key)
    }

    /// Get the entry for an n-gram using a pre-encoded key.
    pub fn get_by_key(&self, key: &str) -> Option<NgramEntry> {
        self.dictionary.get_value(key)
    }

    /// Check if an n-gram exists in the trie.
    ///
    /// # Note
    ///
    /// This method uses legacy pipe-separated encoding. For vocabulary-indexed
    /// encoding, use [`Self::contains_key`].
    pub fn contains(&self, tokens: &[&str]) -> bool {
        let key = Self::encode_key_legacy(tokens);
        self.dictionary.contains(&key)
    }

    /// Check if an n-gram exists in the trie using a pre-encoded key.
    pub fn contains_key(&self, key: &str) -> bool {
        self.dictionary.contains(key)
    }

    /// Get the count for an n-gram, or 0 if it doesn't exist.
    #[inline]
    pub fn count(&self, tokens: &[&str]) -> u64 {
        self.get(tokens).map(|e| e.count()).unwrap_or(0)
    }

    /// Get the count for an n-gram using a pre-encoded key, or 0 if it doesn't exist.
    #[inline]
    pub fn count_by_key(&self, key: &str) -> u64 {
        self.get_by_key(key).map(|e| e.count()).unwrap_or(0)
    }

    /// Update continuation count for an n-gram.
    ///
    /// This is called during the second pass of training to set
    /// the number of unique preceding contexts.
    ///
    /// # Note
    ///
    /// This method uses legacy pipe-separated encoding. For vocabulary-indexed
    /// encoding, use [`Self::update_continuation_count_by_key`].
    pub fn update_continuation_count(&self, tokens: &[&str], continuation_count: u32) {
        let key = Self::encode_key_legacy(tokens);
        self.dictionary.update_or_insert(
            &key,
            NgramEntry::with_stats(0, continuation_count, 0),
            |entry| entry.set_continuation_count(continuation_count),
        );
    }

    /// Update continuation count for an n-gram using a pre-encoded key.
    pub fn update_continuation_count_by_key(&self, key: &str, continuation_count: u32) {
        self.dictionary.update_or_insert(
            key,
            NgramEntry::with_stats(0, continuation_count, 0),
            |entry| entry.set_continuation_count(continuation_count),
        );
    }

    /// Update unique continuations count for an n-gram.
    ///
    /// # Note
    ///
    /// This method uses legacy pipe-separated encoding. For vocabulary-indexed
    /// encoding, use [`Self::update_unique_continuations_by_key`].
    pub fn update_unique_continuations(&self, tokens: &[&str], unique_continuations: u32) {
        let key = Self::encode_key_legacy(tokens);
        self.dictionary.update_or_insert(
            &key,
            NgramEntry::with_stats(0, 0, unique_continuations),
            |entry| entry.set_unique_continuations(unique_continuations),
        );
    }

    /// Update unique continuations count for an n-gram using a pre-encoded key.
    pub fn update_unique_continuations_by_key(&self, key: &str, unique_continuations: u32) {
        self.dictionary.update_or_insert(
            key,
            NgramEntry::with_stats(0, 0, unique_continuations),
            |entry| entry.set_unique_continuations(unique_continuations),
        );
    }

    /// Get the total number of n-grams stored.
    ///
    /// Returns `None` if the dictionary doesn't support length queries.
    pub fn len(&self) -> usize {
        self.dictionary.len().unwrap_or(0)
    }

    /// Check if the trie is empty.
    pub fn is_empty(&self) -> bool {
        self.dictionary.len().map_or(true, |len| len == 0)
    }

    /// Iterate over all (key, entry) pairs in the trie.
    ///
    /// This is available when the dictionary implements `IterableDictionary`.
    pub fn iter_entries(&self) -> impl Iterator<Item = (String, NgramEntry)> + '_
    where
        D: IterableDictionary,
    {
        self.dictionary.iter_all()
    }
}

impl<D> Clone for NgramTrie<D>
where
    D: MutableMappedDictionary<Value = NgramEntry>,
{
    fn clone(&self) -> Self {
        Self {
            dictionary: Arc::clone(&self.dictionary),
            max_order: self.max_order,
            _marker: PhantomData,
        }
    }
}

/// Fast position-aware hash for n-gram keys.
///
/// Uses position-aware hashing to distinguish n-grams with the same
/// tokens in different orders (e.g., ["a", "b"] vs ["b", "a"]).
///
/// Based on MeTTaTron's collision-resistant hashing pattern.
#[inline]
#[allow(dead_code)]
pub fn hash_ngram_key(tokens: &[&str]) -> u64 {
    use crate::util::hash::safe_hash_with_seed;

    const GOLDEN_RATIO: u64 = 0x9e3779b97f4a7c15;
    const NGRAM_SEED: u64 = 0x6e6772616d5f7365; // "ngram_se"

    let mut hash = NGRAM_SEED;
    for (i, token) in tokens.iter().enumerate() {
        let token_hash = safe_hash_with_seed(token.as_bytes(), i as u64);
        hash = hash.wrapping_add(token_hash).wrapping_mul(GOLDEN_RATIO);
    }
    hash ^ (hash >> 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_key_legacy() {
        type Trie = NgramTrie<liblevenshtein::dictionary::pathmap::PathMapDictionary<NgramEntry>>;

        assert_eq!(Trie::encode_key_legacy(&["the"]), "the");
        assert_eq!(Trie::encode_key_legacy(&["the", "quick"]), "the|quick");
        assert_eq!(
            Trie::encode_key_legacy(&["the", "quick", "brown"]),
            "the|quick|brown"
        );
    }

    #[test]
    fn test_legacy_encoding_pipe_bug() {
        // This test demonstrates the bug that vocabulary-indexed encoding fixes
        type Trie = NgramTrie<liblevenshtein::dictionary::pathmap::PathMapDictionary<NgramEntry>>;

        // A token containing a pipe character
        let tokens = ["foo|bar", "baz"];
        let encoded = Trie::encode_key_legacy(&tokens);

        // When decoded by splitting on pipe, we get the wrong number of tokens!
        let decoded: Vec<_> = encoded.split(LEGACY_NGRAM_SEPARATOR).collect();
        assert_eq!(decoded.len(), 3, "Bug: pipe in token causes wrong split");
        assert_eq!(
            decoded,
            ["foo", "bar", "baz"],
            "Bug: original tokens corrupted"
        );
    }

    #[test]
    fn test_hash_ngram_key_order_matters() {
        let hash1 = hash_ngram_key(&["a", "b"]);
        let hash2 = hash_ngram_key(&["b", "a"]);
        assert_ne!(
            hash1, hash2,
            "Different orderings should have different hashes"
        );
    }

    #[test]
    fn test_hash_ngram_key_deterministic() {
        let hash1 = hash_ngram_key(&["the", "quick", "brown"]);
        let hash2 = hash_ngram_key(&["the", "quick", "brown"]);
        assert_eq!(hash1, hash2, "Same input should produce same hash");
    }

    // ── IterableDictionary impls for the persistent ARTrie backends ──

    #[test]
    fn iter_all_shared_char_artrie_roundtrip() {
        use libdictenstein::persistent_artrie_char::{PersistentARTrieChar, SharedCharARTrie};
        use parking_lot::RwLock;
        use std::collections::HashMap;
        use std::sync::Arc;

        let dir = tempfile::tempdir().expect("tempdir");
        let trie = PersistentARTrieChar::<NgramEntry>::create(dir.path().join("c.artrie"))
            .expect("create counts trie");
        let backend: SharedCharARTrie<NgramEntry> = Arc::new(RwLock::new(trie));
        backend.insert_with_value("ab", NgramEntry::new(3));
        backend.insert_with_value("cd", NgramEntry::with_stats(5, 2, 1));

        let got: HashMap<String, u64> = backend.iter_all().map(|(k, v)| (k, v.count())).collect();
        assert_eq!(got.get("ab"), Some(&3));
        assert_eq!(got.get("cd"), Some(&5));
        assert_eq!(got.len(), 2);
    }

    #[test]
    fn iter_all_vocab_indexed_reconstructs_words() {
        use crate::ngram::vocabulary::create_vocabulary;
        use crate::ngram::vocabulary_indexed::VocabularyIndexedDictionary;
        use libdictenstein::persistent_artrie_char::{PersistentARTrieChar, SharedCharARTrie};
        use parking_lot::RwLock;
        use std::collections::HashMap;
        use std::sync::Arc;

        let dir = tempfile::tempdir().expect("tempdir");
        let vocab = create_vocabulary(&dir.path().join("v.artrie")).expect("vocab");
        let counts: SharedCharARTrie<NgramEntry> = Arc::new(RwLock::new(
            PersistentARTrieChar::<NgramEntry>::create(dir.path().join("c.artrie"))
                .expect("counts"),
        ));
        let dict = VocabularyIndexedDictionary::with_delimiter(counts, vocab, '|');

        // Insert via the MutableMappedDictionary surface (splits on '|' →
        // assigns vocab ids → stores latin1 varint keys in the counts trie).
        dict.insert_with_value("the|quick|brown", NgramEntry::new(2));
        dict.insert_with_value("the|lazy", NgramEntry::new(5));

        // iter_all must decode the integer keys back to the exact word strings.
        let got: HashMap<String, u64> = dict.iter_all().map(|(k, v)| (k, v.count())).collect();
        assert_eq!(
            got.get("the|quick|brown"),
            Some(&2),
            "trigram reconstructed"
        );
        assert_eq!(got.get("the|lazy"), Some(&5), "bigram reconstructed");
        assert_eq!(got.len(), 2);
    }

    #[test]
    fn iter_all_vocab_indexed_skips_missing_index() {
        use crate::ngram::vocabulary::{create_vocabulary, encode_varint};
        use crate::ngram::vocabulary_indexed::{
            decode_key_to_indices, VocabularyIndexedDictionary,
        };
        use libdictenstein::persistent_artrie_char::{PersistentARTrieChar, SharedCharARTrie};
        use parking_lot::RwLock;
        use std::sync::Arc;

        let dir = tempfile::tempdir().expect("tempdir");
        let vocab = create_vocabulary(&dir.path().join("v.artrie")).expect("vocab");
        let counts: SharedCharARTrie<NgramEntry> = Arc::new(RwLock::new(
            PersistentARTrieChar::<NgramEntry>::create(dir.path().join("c.artrie"))
                .expect("counts"),
        ));
        let dict = VocabularyIndexedDictionary::with_delimiter(counts.clone(), vocab, '|');

        // One valid n-gram (assigns vocab ids 1,2), …
        dict.insert_with_value("alpha|beta", NgramEntry::new(1));
        // … plus a forged backend key for index 9999 that was never assigned.
        let mut buf = Vec::new();
        encode_varint(9999, &mut buf);
        let bogus_key: String = buf.iter().map(|&b| b as char).collect(); // latin1
        assert_eq!(decode_key_to_indices(&bogus_key), vec![9999]);
        counts.insert_with_value(&bogus_key, NgramEntry::new(7));

        // The missing-index key must be skipped (no panic), the valid one kept.
        let got: Vec<String> = dict.iter_all().map(|(k, _)| k).collect();
        assert_eq!(got, vec!["alpha|beta".to_string()]);
    }
}
