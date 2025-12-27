//! N-gram trie wrapper over liblevenshtein dictionary backends.
//!
//! This module provides a high-level interface for storing and querying n-grams
//! using liblevenshtein-rust's dictionary implementations.

use crate::ngram::NgramEntry;
use gxhash::GxHasher;
use liblevenshtein::dictionary::MutableMappedDictionary;
use std::hash::Hasher;
use std::marker::PhantomData;
use std::sync::Arc;

/// Separator used between tokens in n-gram keys.
///
/// Using pipe character as it's unlikely to appear in natural text tokens.
pub const NGRAM_SEPARATOR: char = '|';

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
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound = "D: serde::Serialize + serde::de::DeserializeOwned")
)]
pub struct NgramTrie<D>
where
    D: MutableMappedDictionary<Value = NgramEntry>,
{
    /// The underlying dictionary backend.
    dictionary: Arc<D>,

    /// Maximum n-gram order stored in this trie.
    max_order: usize,

    /// Phantom data for type parameter.
    #[cfg_attr(feature = "serde", serde(skip))]
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

    /// Encode an n-gram as a dictionary key.
    ///
    /// Tokens are joined with the separator character.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let key = NgramTrie::<D>::encode_key(&["the", "quick", "brown"]);
    /// assert_eq!(key, "the|quick|brown");
    /// ```
    #[inline]
    pub fn encode_key(tokens: &[&str]) -> String {
        tokens.join(&NGRAM_SEPARATOR.to_string())
    }

    /// Insert or increment an n-gram count.
    ///
    /// If the n-gram exists, increments its count. Otherwise, inserts it with count 1.
    ///
    /// # Returns
    ///
    /// `true` if this was a new n-gram (inserted), `false` if it already existed (incremented).
    pub fn insert(&self, tokens: &[&str]) -> bool {
        let key = Self::encode_key(tokens);
        self.dictionary.update_or_insert(
            &key,
            NgramEntry::new(1),
            |entry| entry.increment(),
        )
    }

    /// Insert an n-gram with a specific count.
    pub fn insert_with_count(&self, tokens: &[&str], count: u64) -> bool {
        let key = Self::encode_key(tokens);
        self.dictionary.insert_with_value(&key, NgramEntry::new(count))
    }

    /// Get the entry for an n-gram, if it exists.
    pub fn get(&self, tokens: &[&str]) -> Option<NgramEntry> {
        let key = Self::encode_key(tokens);
        self.dictionary.get_value(&key)
    }

    /// Check if an n-gram exists in the trie.
    pub fn contains(&self, tokens: &[&str]) -> bool {
        let key = Self::encode_key(tokens);
        self.dictionary.contains(&key)
    }

    /// Get the count for an n-gram, or 0 if it doesn't exist.
    #[inline]
    pub fn count(&self, tokens: &[&str]) -> u64 {
        self.get(tokens).map(|e| e.count()).unwrap_or(0)
    }

    /// Update continuation count for an n-gram.
    ///
    /// This is called during the second pass of training to set
    /// the number of unique preceding contexts.
    pub fn update_continuation_count(&self, tokens: &[&str], continuation_count: u32) {
        let key = Self::encode_key(tokens);
        self.dictionary.update_or_insert(
            &key,
            NgramEntry::with_stats(0, continuation_count, 0),
            |entry| entry.set_continuation_count(continuation_count),
        );
    }

    /// Update unique continuations count for an n-gram.
    pub fn update_unique_continuations(&self, tokens: &[&str], unique_continuations: u32) {
        let key = Self::encode_key(tokens);
        self.dictionary.update_or_insert(
            &key,
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
/// Uses GxHash with position encoding to distinguish n-grams with the same
/// tokens in different orders (e.g., ["a", "b"] vs ["b", "a"]).
///
/// Based on MeTTaTron's collision-resistant hashing pattern.
#[inline]
pub fn hash_ngram_key(tokens: &[&str]) -> u64 {
    const GOLDEN_RATIO: u64 = 0x9e3779b97f4a7c15;
    const NGRAM_SEED: u64 = 0x6e6772616d5f7365; // "ngram_se"

    let mut hash = NGRAM_SEED;
    for (i, token) in tokens.iter().enumerate() {
        // Position-aware hashing
        let mut hasher = GxHasher::with_seed(i as i64);
        hasher.write(token.as_bytes());
        let token_hash = hasher.finish();
        hash = hash.wrapping_add(token_hash).wrapping_mul(GOLDEN_RATIO);
    }
    hash ^ (hash >> 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_key() {
        assert_eq!(
            NgramTrie::<liblevenshtein::dictionary::pathmap::PathMapDictionary<NgramEntry>>::encode_key(&["the"]),
            "the"
        );
        assert_eq!(
            NgramTrie::<liblevenshtein::dictionary::pathmap::PathMapDictionary<NgramEntry>>::encode_key(&["the", "quick"]),
            "the|quick"
        );
        assert_eq!(
            NgramTrie::<liblevenshtein::dictionary::pathmap::PathMapDictionary<NgramEntry>>::encode_key(&["the", "quick", "brown"]),
            "the|quick|brown"
        );
    }

    #[test]
    fn test_hash_ngram_key_order_matters() {
        let hash1 = hash_ngram_key(&["a", "b"]);
        let hash2 = hash_ngram_key(&["b", "a"]);
        assert_ne!(hash1, hash2, "Different orderings should have different hashes");
    }

    #[test]
    fn test_hash_ngram_key_deterministic() {
        let hash1 = hash_ngram_key(&["the", "quick", "brown"]);
        let hash2 = hash_ngram_key(&["the", "quick", "brown"]);
        assert_eq!(hash1, hash2, "Same input should produce same hash");
    }
}
