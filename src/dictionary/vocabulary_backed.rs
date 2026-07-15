//! Vocabulary-backed dictionary for n-gram models.
//!
//! When using vocabulary-indexed n-gram encoding, the vocabulary already contains
//! all words encountered during training, and unigram frequencies are stored in
//! the n-gram storage. This module provides a dictionary interface backed by
//! these existing data structures, eliminating the need for separate dictionary
//! extraction.
//!
//! # Advantages
//!
//! - **No duplicate data**: Reuses vocabulary and n-gram storage
//! - **Always consistent**: Dictionary is always in sync with n-gram model
//! - **Efficient**: No separate extraction or building step required

use crate::ngram::vocabulary::SharedConcurrentVocab;
use crate::sources::google_books::NgramStorage;

/// A dictionary backed by shared vocabulary and n-gram storage.
///
/// This provides dictionary functionality (word existence checks, frequency lookups)
/// using the vocabulary and unigram counts from a vocabulary-indexed n-gram model.
///
/// # Example
///
/// ```ignore
/// use libgrammstein::dictionary::VocabularyDictionary;
/// use libgrammstein::ngram::vocabulary::{open_or_create_vocabulary, SharedVocabARTrie};
/// use libgrammstein::sources::google_books::storage::NgramStorage;
///
/// let vocab = open_or_create_vocabulary("vocab.artrie")?;
/// let storage = NgramStorage::create_single_trie("ngrams.artrie")?;
///
/// let dict = VocabularyDictionary::new(vocab, &storage);
///
/// // Check if word exists in vocabulary
/// if dict.contains("hello") {
///     // Get unigram frequency from n-gram storage
///     if let Some(freq) = dict.frequency("hello") {
///         println!("'hello' appears {} times", freq);
///     }
/// }
/// ```
pub struct VocabularyDictionary<'a> {
    /// Shared vocabulary mapping words to indices.
    vocabulary: SharedConcurrentVocab,
    /// N-gram storage containing unigram counts.
    storage: &'a NgramStorage,
}

impl<'a> VocabularyDictionary<'a> {
    /// Create a new vocabulary-backed dictionary.
    ///
    /// # Arguments
    ///
    /// * `vocabulary` - The shared vocabulary from n-gram training
    /// * `storage` - The n-gram storage containing unigram counts
    pub fn new(vocabulary: SharedConcurrentVocab, storage: &'a NgramStorage) -> Self {
        Self {
            vocabulary,
            storage,
        }
    }

    /// Check if a word exists in the vocabulary.
    ///
    /// This is O(k) in word length due to trie lookup.
    pub fn contains(&self, word: &str) -> bool {
        self.vocabulary.get_index(word).is_some()
    }

    /// Get the unigram frequency of a word.
    ///
    /// Returns `None` if the word is not in the vocabulary or has no unigram
    /// count recorded in storage.
    ///
    /// # Note
    ///
    /// This encodes the word to a varint key and looks it up in storage.
    /// For sharded storage, ensure the storage was created with vocabulary
    /// support to guarantee correct routing.
    pub fn frequency(&self, word: &str) -> Option<u64> {
        // Check if word is in vocabulary
        self.vocabulary.get_index(word)?;

        // Byte-native unigram lookup: `get_tokens` encodes `[word]` to a LEB128
        // term-id key and routes by the token — matching exactly what
        // `store_tokens` wrote. (The former `encode_ngram_key_lockfree` + `get`
        // path char-lifted each varint byte into a Latin-1 `char`, whose UTF-8
        // bytes diverged from the byte-native stored key for any term-id >= 128.)
        self.storage.get_tokens(&[word])
    }

    /// Get the vocabulary size (number of unique words).
    pub fn len(&self) -> u64 {
        self.vocabulary.len() as u64
    }

    /// Check if the dictionary is empty.
    pub fn is_empty(&self) -> bool {
        self.vocabulary.len() == 0
    }

    /// Get a reference to the underlying vocabulary.
    pub fn vocabulary(&self) -> &SharedConcurrentVocab {
        &self.vocabulary
    }

    /// Get a reference to the underlying n-gram storage.
    pub fn storage(&self) -> &NgramStorage {
        self.storage
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ngram::vocabulary::open_or_create_concurrent_vocabulary_lockfree;
    use crate::sources::google_books::NgramStorage;
    use tempfile::TempDir;

    #[test]
    fn test_vocabulary_dictionary() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let vocab_path = dir.path().join("vocab.artrie");
        let trie_path = dir.path().join("ngrams.artrie");

        // Create vocabulary and storage
        let vocabulary = open_or_create_concurrent_vocabulary_lockfree(&vocab_path)
            .expect("Failed to create vocabulary");

        let storage =
            NgramStorage::create_single_trie_with_vocabulary(&trie_path, Some(vocabulary.clone()))
                .expect("Failed to create storage");

        // Store some unigrams
        storage
            .store_tokens(&["hello"], 100)
            .expect("Failed to store");
        storage
            .store_tokens(&["world"], 50)
            .expect("Failed to store");
        storage
            .store_tokens(&["test"], 25)
            .expect("Failed to store");

        // Merge lock-free vocabulary entries into persistent layer
        // (required before VocabularyDictionary can see them)
        storage
            .sync_vocabulary()
            .expect("Failed to sync vocabulary");

        // Create dictionary
        let dict = VocabularyDictionary::new(vocabulary, &storage);

        // Check contains
        assert!(dict.contains("hello"));
        assert!(dict.contains("world"));
        assert!(dict.contains("test"));
        assert!(!dict.contains("nonexistent"));

        // Check frequencies
        assert_eq!(dict.frequency("hello"), Some(100));
        assert_eq!(dict.frequency("world"), Some(50));
        assert_eq!(dict.frequency("test"), Some(25));
        assert_eq!(dict.frequency("nonexistent"), None);

        // Check length
        assert_eq!(dict.len(), 3);
        assert!(!dict.is_empty());
    }

    #[test]
    fn vocabulary_query_frequency_requires_vocabulary_and_storage_count() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let vocab_path = dir.path().join("vocab.artrie");
        let trie_path = dir.path().join("ngrams.artrie");

        let vocabulary = open_or_create_concurrent_vocabulary_lockfree(&vocab_path)
            .expect("Failed to create vocabulary");

        let storage =
            NgramStorage::create_single_trie_with_vocabulary(&trie_path, Some(vocabulary.clone()))
                .expect("Failed to create storage");

        storage
            .store_tokens(&["known"], 11)
            .expect("Failed to store unigram");
        storage
            .sync_vocabulary()
            .expect("Failed to sync vocabulary");

        vocabulary
            .insert("vocab_only")
            .expect("Failed to insert vocabulary-only word");

        let dict = VocabularyDictionary::new(vocabulary, &storage);

        assert!(dict.contains("known"));
        assert_eq!(dict.frequency("known"), Some(11));
        assert!(dict.contains("vocab_only"));
        assert_eq!(
            dict.frequency("vocab_only"),
            None,
            "frequency requires both vocabulary membership and a unigram count"
        );
        assert!(!dict.contains("missing"));
        assert_eq!(dict.frequency("missing"), None);
    }

    #[test]
    fn test_vocabulary_dictionary_empty() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let vocab_path = dir.path().join("vocab.artrie");
        let trie_path = dir.path().join("ngrams.artrie");

        let vocabulary = open_or_create_concurrent_vocabulary_lockfree(&vocab_path)
            .expect("Failed to create vocabulary");

        let storage =
            NgramStorage::create_single_trie_with_vocabulary(&trie_path, Some(vocabulary.clone()))
                .expect("Failed to create storage");

        let dict = VocabularyDictionary::new(vocabulary, &storage);

        assert!(dict.is_empty());
        assert!(!dict.contains("anything"));
        assert_eq!(dict.frequency("anything"), None);
    }
}
