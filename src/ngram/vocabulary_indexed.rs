//! Vocabulary-indexed dictionary wrapper for transparent n-gram encoding.
//!
//! This module provides [`VocabularyIndexedDictionary`], which wraps an underlying
//! [`MutableMappedDictionary`] with transparent word-to-index translation. N-grams
//! are stored as varint-encoded vocabulary indices, but the API accepts and returns
//! human-readable word sequences.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │   VocabularyIndexedDictionary<D>                    │
//! │   (this module)                                     │
//! ├─────────────────────────────────────────────────────┤
//! │  - vocabulary: Arc<SharedVocabARTrie>  ← word → u64  │
//! │  - backend: D                 ← underlying trie     │
//! │  - delimiter: char                                  │
//! └─────────────────────────────────────────────────────┘
//!                     │
//!     ┌───────────────┴───────────────┐
//!     ▼                               ▼
//! ┌──────────────────┐    ┌───────────────────────────┐
//! │ SharedVocabARTrie │    │  Any MutableMappedDictionary │
//! │ (word → u64)     │    │  (DynamicDawgChar, etc.)     │
//! └──────────────────┘    └───────────────────────────────┘
//! ```
//!
//! # Key Encoding
//!
//! N-grams are encoded as concatenated LEB128 varints, with each byte stored
//! as a Latin-1 character (0x00-0xFF → U+0000-U+00FF):
//!
//! - Word "the" → index 1 → varint `[0x01]` → char `'\x01'`
//! - Word "quick" → index 128 → varint `[0x80, 0x01]` → chars `'\x80\x01'`
//!
//! # Usage
//!
//! ```ignore
//! use libgrammstein::ngram::vocabulary_indexed::VocabularyIndexedDictionary;
//! use libgrammstein::ngram::SharedVocabARTrie;
//! use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
//!
//! // Create or open vocabulary
//! let vocab = Arc::new(SharedVocabARTrie::open_or_create(&vocab_path)?);
//!
//! // Create underlying trie
//! let backend = DynamicDawgChar::new();
//!
//! // Wrap with vocabulary indexing
//! let dict = VocabularyIndexedDictionary::new(backend, vocab);
//!
//! // Insert n-gram (words are transparently encoded)
//! dict.insert_ngram(&["the", "quick", "brown"], 42);
//!
//! // Lookup works the same way
//! assert_eq!(dict.get_ngram(&["the", "quick", "brown"]), Some(42));
//! ```
//!
//! # OOV Handling
//!
//! - **Read operations**: Return `None`/`false` for OOV words without modifying vocabulary
//! - **Write operations**: Acquire new vocabulary indices for unknown words

use super::vocabulary::{decode_varint, encode_varint, SharedVocabARTrie};
use liblevenshtein::dictionary::{
    Dictionary, DictionaryNode, MappedDictionary, MappedDictionaryNode, MutableMappedDictionary,
    SyncStrategy,
};

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert varint bytes to a Latin-1 encoded string.
///
/// Each byte 0x00-0xFF is stored as char U+0000-U+00FF.
#[inline]
fn bytes_to_latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| char::from(b)).collect()
}

/// Convert a Latin-1 encoded string back to bytes.
#[inline]
fn latin1_to_bytes(s: &str) -> Vec<u8> {
    s.chars().map(|c| c as u8).collect()
}

/// Decode a Latin-1 encoded varint key back to word indices.
pub fn decode_key_to_indices(key: &str) -> Vec<u64> {
    let bytes = latin1_to_bytes(key);
    let mut indices = Vec::new();
    let mut offset = 0;
    while offset < bytes.len() {
        if let Some((index, consumed)) = decode_varint(&bytes[offset..]) {
            indices.push(index);
            offset += consumed;
        } else {
            break;
        }
    }
    indices
}

// ============================================================================
// VocabularyIndexedDictionary
// ============================================================================

/// A dictionary wrapper that transparently encodes/decodes vocabulary-indexed keys.
///
/// This struct wraps an underlying [`MutableMappedDictionary`] backend (like
/// [`DynamicDawgChar`](liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar))
/// and provides n-gram-aware insertion and lookup using a shared vocabulary.
///
/// # Type Parameters
///
/// - `D`: The underlying dictionary backend
///
/// # Thread Safety
///
/// This type is `Send + Sync` if `D` is. The vocabulary is shared via `Arc`
/// across multiple dictionary instances.
///
/// # Example
///
/// ```ignore
/// use libgrammstein::ngram::vocabulary_indexed::VocabularyIndexedDictionary;
/// use libgrammstein::ngram::SharedVocabARTrie;
/// use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
///
/// let vocab = Arc::new(SharedVocabARTrie::open_or_create(&path)?);
/// let backend: DynamicDawgChar<u64> = DynamicDawgChar::new();
/// let dict = VocabularyIndexedDictionary::new(backend, vocab);
///
/// // Insert n-gram with count
/// dict.insert_ngram(&["the", "quick"], 100);
///
/// // Query (returns None for OOV)
/// assert_eq!(dict.get_ngram(&["the", "quick"]), Some(100));
/// assert_eq!(dict.get_ngram(&["unknown", "word"]), None);
/// ```
#[derive(Clone)]
pub struct VocabularyIndexedDictionary<D> {
    /// The underlying dictionary backend.
    backend: D,
    /// The shared vocabulary for word-to-index mapping.
    vocabulary: SharedVocabARTrie,
    /// Delimiter for splitting terms into words.
    delimiter: char,
}

impl<D> std::fmt::Debug for VocabularyIndexedDictionary<D>
where
    D: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VocabularyIndexedDictionary")
            .field("backend", &self.backend)
            .field("delimiter", &self.delimiter)
            .finish_non_exhaustive()
    }
}

impl<D> VocabularyIndexedDictionary<D> {
    /// Create a new vocabulary-indexed dictionary with default space delimiter.
    ///
    /// # Arguments
    ///
    /// * `backend` - The underlying dictionary backend
    /// * `vocabulary` - The shared vocabulary for word-to-index mapping
    pub fn new(backend: D, vocabulary: SharedVocabARTrie) -> Self {
        Self {
            backend,
            vocabulary,
            delimiter: ' ',
        }
    }

    /// Create a new vocabulary-indexed dictionary with a custom delimiter.
    ///
    /// # Arguments
    ///
    /// * `backend` - The underlying dictionary backend
    /// * `vocabulary` - The shared vocabulary for word-to-index mapping
    /// * `delimiter` - The character used to split terms into words
    pub fn with_delimiter(backend: D, vocabulary: SharedVocabARTrie, delimiter: char) -> Self {
        Self {
            backend,
            vocabulary,
            delimiter,
        }
    }

    /// Get a reference to the underlying backend.
    pub fn backend(&self) -> &D {
        &self.backend
    }

    /// Get a reference to the vocabulary.
    pub fn vocabulary(&self) -> &SharedVocabARTrie {
        &self.vocabulary
    }

    /// Get the delimiter used for splitting terms.
    pub fn delimiter(&self) -> char {
        self.delimiter
    }

    /// Split a term into words using the configured delimiter.
    fn split_term<'a>(&self, term: &'a str) -> impl Iterator<Item = &'a str> {
        term.split(self.delimiter)
    }

    /// Encode words to a varint key using existing vocabulary only.
    ///
    /// Returns `None` if any word is OOV (out of vocabulary).
    fn encode_key_existing(&self, words: &[&str]) -> Option<String> {
        let mut buf = Vec::with_capacity(words.len() * 2);
        let guard = self.vocabulary.read();
        for word in words {
            let index = guard.get_index(word)?;
            encode_varint(index, &mut buf);
        }
        Some(bytes_to_latin1(&buf))
    }

    /// Encode words to a varint key, acquiring new indices as needed.
    fn encode_key_inserting(&self, words: &[&str]) -> String {
        let mut buf = Vec::with_capacity(words.len() * 2);
        let mut guard = self.vocabulary.write();
        for word in words {
            let index = guard
                .insert(word)
                .expect("vocabulary insert: persistent ARTrie I/O failed");
            encode_varint(index, &mut buf);
        }
        bytes_to_latin1(&buf)
    }
}

// ============================================================================
// N-gram specific methods
// ============================================================================

impl<D> VocabularyIndexedDictionary<D>
where
    D: MappedDictionary,
{
    /// Check if an n-gram exists in the dictionary.
    ///
    /// Returns `false` if any word is OOV (out of vocabulary).
    pub fn contains_ngram(&self, words: &[&str]) -> bool {
        self.encode_key_existing(words)
            .map(|key| self.backend.contains(&key))
            .unwrap_or(false)
    }

    /// Get the value associated with an n-gram.
    ///
    /// Returns `None` if the n-gram doesn't exist or any word is OOV.
    pub fn get_ngram(&self, words: &[&str]) -> Option<D::Value> {
        let key = self.encode_key_existing(words)?;
        self.backend.get_value(&key)
    }
}

impl<D> VocabularyIndexedDictionary<D>
where
    D: MutableMappedDictionary,
{
    /// Insert an n-gram with a value.
    ///
    /// New words are automatically added to the vocabulary.
    ///
    /// # Returns
    ///
    /// `true` if this is a new n-gram, `false` if updating an existing one.
    pub fn insert_ngram(&self, words: &[&str], value: D::Value) -> bool {
        let key = self.encode_key_inserting(words);
        self.backend.insert_with_value(&key, value)
    }

    /// Update an existing n-gram's value or insert a new one.
    ///
    /// # Arguments
    ///
    /// * `words` - The n-gram word sequence
    /// * `default_value` - The value to use if the n-gram doesn't exist
    /// * `update_fn` - Function to apply to the existing value if it exists
    ///
    /// # Returns
    ///
    /// `true` if this was a new n-gram (inserted with default), `false` if updated.
    pub fn update_or_insert_ngram<F>(
        &self,
        words: &[&str],
        default_value: D::Value,
        update_fn: F,
    ) -> bool
    where
        F: FnOnce(&mut D::Value),
    {
        let key = self.encode_key_inserting(words);
        self.backend
            .update_or_insert(&key, default_value, update_fn)
    }
}

// ============================================================================
// Dictionary Trait Implementation
// ============================================================================

impl<D> Dictionary for VocabularyIndexedDictionary<D>
where
    D: Dictionary,
{
    type Node = VocabularyIndexedNode<D::Node>;

    fn root(&self) -> Self::Node {
        VocabularyIndexedNode {
            inner: self.backend.root(),
        }
    }

    fn contains(&self, term: &str) -> bool {
        // Split term and encode using existing vocabulary
        let words: Vec<&str> = self.split_term(term).collect();
        self.encode_key_existing(&words)
            .map(|key| self.backend.contains(&key))
            .unwrap_or(false)
    }

    fn len(&self) -> Option<usize> {
        self.backend.len()
    }

    fn is_empty(&self) -> bool {
        self.backend.is_empty()
    }

    fn sync_strategy(&self) -> SyncStrategy {
        self.backend.sync_strategy()
    }
}

// ============================================================================
// MappedDictionary Trait Implementation
// ============================================================================

impl<D> MappedDictionary for VocabularyIndexedDictionary<D>
where
    D: MappedDictionary,
{
    type Value = D::Value;

    fn get_value(&self, term: &str) -> Option<Self::Value> {
        let words: Vec<&str> = self.split_term(term).collect();
        let key = self.encode_key_existing(&words)?;
        self.backend.get_value(&key)
    }

    fn contains_with_value<F>(&self, term: &str, predicate: F) -> bool
    where
        F: Fn(&Self::Value) -> bool,
    {
        let words: Vec<&str> = self.split_term(term).collect();
        self.encode_key_existing(&words)
            .map(|key| self.backend.contains_with_value(&key, predicate))
            .unwrap_or(false)
    }
}

// ============================================================================
// MutableMappedDictionary Trait Implementation
// ============================================================================

impl<D> MutableMappedDictionary for VocabularyIndexedDictionary<D>
where
    D: MutableMappedDictionary,
{
    fn insert_with_value(&self, term: &str, value: Self::Value) -> bool {
        let words: Vec<&str> = self.split_term(term).collect();
        let key = self.encode_key_inserting(&words);
        self.backend.insert_with_value(&key, value)
    }

    fn union_with<F>(&self, other: &Self, merge_fn: F) -> usize
    where
        F: Fn(&Self::Value, &Self::Value) -> Self::Value,
        Self::Value: Clone,
    {
        // Note: This assumes both dictionaries share the same vocabulary
        // If they don't, keys won't match correctly
        self.backend.union_with(&other.backend, merge_fn)
    }

    fn update_or_insert<F>(&self, term: &str, default_value: Self::Value, update_fn: F) -> bool
    where
        F: FnOnce(&mut Self::Value),
    {
        let words: Vec<&str> = self.split_term(term).collect();
        let key = self.encode_key_inserting(&words);
        self.backend
            .update_or_insert(&key, default_value, update_fn)
    }
}

// ============================================================================
// VocabularyIndexedNode
// ============================================================================

/// Node wrapper for vocabulary-indexed dictionary traversal.
///
/// This wrapper exposes the underlying trie's character-level transitions,
/// enabling compatibility with Levenshtein automata and other traversal algorithms.
///
/// # Character Semantics
///
/// The node exposes Latin-1 characters (U+0000-U+00FF) representing varint-encoded
/// vocabulary indices. Traversal operates at the character level, not the word level.
#[derive(Clone)]
pub struct VocabularyIndexedNode<N> {
    inner: N,
}

impl<N: std::fmt::Debug> std::fmt::Debug for VocabularyIndexedNode<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VocabularyIndexedNode")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<N: DictionaryNode> DictionaryNode for VocabularyIndexedNode<N> {
    type Unit = N::Unit;

    fn is_final(&self) -> bool {
        self.inner.is_final()
    }

    fn transition(&self, label: Self::Unit) -> Option<Self> {
        self.inner.transition(label).map(|inner| Self { inner })
    }

    fn edges(&self) -> Box<dyn Iterator<Item = (Self::Unit, Self)> + '_> {
        Box::new(
            self.inner
                .edges()
                .map(|(label, inner)| (label, Self { inner })),
        )
    }

    fn has_edge(&self, label: Self::Unit) -> bool {
        self.inner.has_edge(label)
    }

    fn edge_count(&self) -> Option<usize> {
        self.inner.edge_count()
    }
}

impl<N: MappedDictionaryNode> MappedDictionaryNode for VocabularyIndexedNode<N> {
    type Value = N::Value;

    fn value(&self) -> Option<Self::Value> {
        self.inner.value()
    }
}

// ============================================================================
// Zipper Support
// ============================================================================

use super::metadata_filtering_zipper::MetadataFilteringZipper;
use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
use liblevenshtein::dictionary::dynamic_dawg_char_zipper::DynamicDawgCharZipper;
use liblevenshtein::dictionary::value::DictionaryValue;

/// Type alias for the zipper over a DynamicDawgChar backend.
///
/// This zipper wraps the backend's zipper with metadata filtering,
/// ensuring that `\x00` prefixed metadata entries are not exposed
/// through iteration or navigation.
pub type VocabularyIndexedDictionaryZipper<V> = MetadataFilteringZipper<DynamicDawgCharZipper<V>>;

impl<V: DictionaryValue> VocabularyIndexedDictionary<DynamicDawgChar<V>> {
    /// Create a metadata-filtering zipper for iteration.
    ///
    /// The zipper automatically excludes `\x00` prefixed metadata entries
    /// from `children()` iteration and `descend()` operations at the root level.
    ///
    /// # Returns
    ///
    /// A `VocabularyIndexedDictionaryZipper` positioned at the root of the
    /// underlying trie, with metadata filtering enabled.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use libgrammstein::ngram::vocabulary_indexed::VocabularyIndexedDictionary;
    /// use libdictenstein::zipper::{DictZipper, ValuedDictZipper};
    ///
    /// let dict: VocabularyIndexedDictionary<DynamicDawgChar<u64>> = /* ... */;
    ///
    /// // Iterate all n-grams (metadata filtered)
    /// let zipper = dict.zipper();
    /// for (label, child) in zipper.children() {
    ///     // '\x00' prefixed metadata is not included
    ///     println!("Label: {:?}", label);
    /// }
    /// ```
    pub fn zipper(&self) -> VocabularyIndexedDictionaryZipper<V> {
        let backend_zipper = DynamicDawgCharZipper::new_from_dict(self.backend());
        MetadataFilteringZipper::new(backend_zipper)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ngram::vocabulary::create_vocabulary;
    use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
    use tempfile::TempDir;

    fn create_test_dict() -> (TempDir, VocabularyIndexedDictionary<DynamicDawgChar<u64>>) {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let vocab_path = dir.path().join("vocab.artrie");
        let vocab = create_vocabulary(&vocab_path).expect("Failed to create vocabulary");
        let backend: DynamicDawgChar<u64> = DynamicDawgChar::new();
        let dict = VocabularyIndexedDictionary::new(backend, vocab);
        (dir, dict)
    }

    #[test]
    fn test_insert_ngram() {
        let (_dir, dict) = create_test_dict();

        assert!(dict.insert_ngram(&["the", "quick"], 100));
        assert!(!dict.insert_ngram(&["the", "quick"], 200)); // Update

        assert_eq!(dict.get_ngram(&["the", "quick"]), Some(200));
    }

    #[test]
    fn test_contains_ngram() {
        let (_dir, dict) = create_test_dict();

        assert!(!dict.contains_ngram(&["the", "quick"]));

        dict.insert_ngram(&["the", "quick"], 100);
        assert!(dict.contains_ngram(&["the", "quick"]));
        assert!(!dict.contains_ngram(&["unknown", "word"]));
    }

    #[test]
    fn test_oov_returns_none() {
        let (_dir, dict) = create_test_dict();

        // OOV words should return None without modifying vocabulary
        assert!(dict.get_ngram(&["unknown", "word"]).is_none());

        // Vocabulary should still be empty (no side effects)
        assert_eq!(dict.vocabulary().read().len(), 0);
    }

    #[test]
    fn test_update_or_insert_ngram() {
        let (_dir, dict) = create_test_dict();

        // Insert new
        let is_new = dict.update_or_insert_ngram(&["the", "quick"], 10, |v| *v += 5);
        assert!(is_new);
        assert_eq!(dict.get_ngram(&["the", "quick"]), Some(10));

        // Update existing
        let is_new = dict.update_or_insert_ngram(&["the", "quick"], 10, |v| *v += 5);
        assert!(!is_new);
        assert_eq!(dict.get_ngram(&["the", "quick"]), Some(15));
    }

    #[test]
    fn test_dictionary_trait() {
        let (_dir, dict) = create_test_dict();

        // Insert via ngram API
        dict.insert_ngram(&["the", "quick"], 100);

        // Check via Dictionary trait
        assert!(dict.len().is_some());
        assert!(!dict.is_empty());
    }

    #[test]
    fn test_mapped_dictionary_trait() {
        let (_dir, dict) = create_test_dict();

        // Insert via ngram API
        dict.insert_ngram(&["hello", "world"], 42);

        // Lookup via MappedDictionary trait (with delimiter)
        assert_eq!(dict.get_value("hello world"), Some(42));
        assert!(dict.get_value("unknown word").is_none());
    }

    #[test]
    fn test_mutable_mapped_dictionary_trait() {
        let (_dir, dict) = create_test_dict();

        // Insert via trait
        assert!(dict.insert_with_value("hello world", 100));
        assert_eq!(dict.get_value("hello world"), Some(100));

        // Update
        let is_new = dict.update_or_insert("hello world", 50, |v| *v *= 2);
        assert!(!is_new);
        assert_eq!(dict.get_value("hello world"), Some(200));
    }

    #[test]
    fn test_custom_delimiter() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let vocab_path = dir.path().join("vocab.artrie");
        let vocab = create_vocabulary(&vocab_path).expect("Failed to create vocabulary");
        let backend: DynamicDawgChar<u64> = DynamicDawgChar::new();
        let dict = VocabularyIndexedDictionary::with_delimiter(backend, vocab, '|');

        dict.insert_with_value("the|quick|brown", 123);
        assert_eq!(dict.get_value("the|quick|brown"), Some(123));
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let dir = TempDir::new().expect("Failed to create temp dir");
        let vocab_path = dir.path().join("vocab.artrie");
        let vocab = create_vocabulary(&vocab_path).expect("Failed to create vocabulary");
        let backend: DynamicDawgChar<u64> = DynamicDawgChar::new();
        let dict = Arc::new(VocabularyIndexedDictionary::new(backend, vocab));
        let mut handles = vec![];

        // Multiple threads inserting the same n-gram
        for _ in 0..10 {
            let dict = Arc::clone(&dict);
            handles.push(thread::spawn(move || {
                dict.insert_ngram(&["shared", "ngram"], 42);
            }));
        }

        for handle in handles {
            handle.join().expect("thread should complete");
        }

        // Should have exactly one entry
        assert_eq!(dict.get_ngram(&["shared", "ngram"]), Some(42));
    }

    #[test]
    fn test_node_traversal() {
        let (_dir, dict) = create_test_dict();
        dict.insert_ngram(&["abc"], 1);

        let root = dict.root();
        assert!(!root.is_final());

        // Traversal works at character level (over encoded key)
        // This is useful for Levenshtein automata compatibility
        let edges: Vec<_> = root.edges().collect();
        assert!(!edges.is_empty());
    }

    #[test]
    fn test_large_vocabulary_indices() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let vocab_path = dir.path().join("vocab.artrie");
        let vocab = create_vocabulary(&vocab_path).expect("Failed to create vocabulary");
        let backend: DynamicDawgChar<u64> = DynamicDawgChar::new();
        let dict = VocabularyIndexedDictionary::new(backend, vocab.clone());

        // Insert enough words to require multi-byte varints
        {
            let mut guard = vocab.write();
            for i in 0..200 {
                guard.insert(&format!("word{}", i));
            }
        }

        // Insert n-gram with high-index words
        dict.insert_ngram(&["word0", "word127", "word199"], 999);
        assert_eq!(dict.get_ngram(&["word0", "word127", "word199"]), Some(999));
    }

    #[test]
    fn test_decode_key_to_indices() {
        let indices = vec![1u64, 127, 128, 16383];
        let mut buf = Vec::new();
        for &idx in &indices {
            encode_varint(idx, &mut buf);
        }
        let key = bytes_to_latin1(&buf);

        let decoded = decode_key_to_indices(&key);
        assert_eq!(decoded, indices);
    }

    #[test]
    fn test_empty_ngram() {
        let (_dir, dict) = create_test_dict();

        // Empty n-gram should work
        assert!(dict.insert_ngram(&[], 42));
        assert_eq!(dict.get_ngram(&[]), Some(42));
    }

    #[test]
    fn test_single_word_ngram() {
        let (_dir, dict) = create_test_dict();

        dict.insert_ngram(&["unigram"], 1);
        assert_eq!(dict.get_ngram(&["unigram"]), Some(1));
    }

    #[test]
    fn test_zipper_excludes_metadata() {
        use liblevenshtein::dictionary::zipper::DictZipper;

        let (_dir, dict) = create_test_dict();

        // Insert regular n-grams
        dict.insert_ngram(&["hello"], 1);
        dict.insert_ngram(&["world"], 2);

        // Simulate metadata by inserting directly to backend with \x00 prefix
        // (In production, metadata would be inserted via vocabulary methods)
        dict.backend().insert_with_value("\x00__meta__", 999);

        // Create zipper - metadata should be filtered
        let zipper = dict.zipper();

        // Check that \x00 is not in the children
        let children: Vec<char> = zipper.children().map(|(c, _)| c).collect();
        assert!(
            !children.contains(&'\x00'),
            "Metadata prefix should be filtered from children"
        );

        // Verify we can't descend to metadata
        assert!(
            zipper.descend('\x00').is_none(),
            "Should not be able to descend to metadata at root"
        );

        // Verify we can still descend to regular data
        // (The encoded key for "hello" starts with the varint-encoded index)
        assert!(
            !children.is_empty(),
            "Should have children for regular n-grams"
        );
    }
}
