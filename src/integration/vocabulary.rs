//! Word vocabulary for WFST export.
//!
//! This module provides a bidirectional mapping between words (strings)
//! and compact word IDs for efficient WFST representation.

use std::collections::HashMap;

/// Type alias for word IDs in the WFST.
pub type WordId = u32;

/// Special word ID for the end-of-sentence marker.
pub const EOS_WORD_ID: WordId = 0;

/// Special word ID for the start-of-sentence / unknown word marker.
pub const UNK_WORD_ID: WordId = 1;

/// End-of-sentence token string.
pub const EOS_TOKEN: &str = "</s>";

/// Unknown word token string.
pub const UNK_TOKEN: &str = "<unk>";

/// Bidirectional mapping between words and compact word IDs.
///
/// The vocabulary is built from the n-gram model's trie by iterating
/// over all unique unigrams. Special tokens (<unk>, </s>) are reserved
/// at fixed IDs.
///
/// # Example
///
/// ```ignore
/// use libgrammstein::integration::vocabulary::WordVocabulary;
///
/// let mut vocab = WordVocabulary::new();
/// let id = vocab.add_word("hello");
/// assert_eq!(vocab.get_word(id), Some("hello"));
/// assert_eq!(vocab.get_id("hello"), Some(id));
/// ```
#[derive(Clone, Debug)]
pub struct WordVocabulary {
    /// Mapping from word string to ID.
    word_to_id: HashMap<String, WordId>,
    /// Mapping from ID to word string.
    id_to_word: Vec<String>,
}

impl WordVocabulary {
    /// Create a new empty vocabulary with reserved special tokens.
    ///
    /// Reserved IDs:
    /// - 0: `</s>` (end-of-sentence)
    /// - 1: `<unk>` (unknown word)
    pub fn new() -> Self {
        let mut vocab = Self {
            word_to_id: HashMap::new(),
            id_to_word: Vec::new(),
        };

        // Reserve special tokens
        vocab.add_word(EOS_TOKEN);
        vocab.add_word(UNK_TOKEN);

        vocab
    }

    /// Create a vocabulary with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut vocab = Self {
            word_to_id: HashMap::with_capacity(capacity + 2), // +2 for special tokens
            id_to_word: Vec::with_capacity(capacity + 2),
        };

        // Reserve special tokens
        vocab.add_word(EOS_TOKEN);
        vocab.add_word(UNK_TOKEN);

        vocab
    }

    /// Add a word to the vocabulary, returning its ID.
    ///
    /// If the word already exists, returns its existing ID.
    pub fn add_word(&mut self, word: &str) -> WordId {
        if let Some(&id) = self.word_to_id.get(word) {
            return id;
        }

        let id = self.id_to_word.len() as WordId;
        self.id_to_word.push(word.to_string());
        self.word_to_id.insert(word.to_string(), id);
        id
    }

    /// Get the ID for a word, if it exists.
    #[inline]
    pub fn get_id(&self, word: &str) -> Option<WordId> {
        self.word_to_id.get(word).copied()
    }

    /// Get the ID for a word, or the unknown word ID if not found.
    #[inline]
    pub fn get_id_or_unk(&self, word: &str) -> WordId {
        self.word_to_id.get(word).copied().unwrap_or(UNK_WORD_ID)
    }

    /// Get the word for an ID, if it exists.
    #[inline]
    pub fn get_word(&self, id: WordId) -> Option<&str> {
        self.id_to_word.get(id as usize).map(|s| s.as_str())
    }

    /// Get the number of words in the vocabulary.
    #[inline]
    pub fn len(&self) -> usize {
        self.id_to_word.len()
    }

    /// Check if the vocabulary is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.id_to_word.len() <= 2 // Only special tokens
    }

    /// Check if a word exists in the vocabulary.
    #[inline]
    pub fn contains(&self, word: &str) -> bool {
        self.word_to_id.contains_key(word)
    }

    /// Iterate over (word, id) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, WordId)> + '_ {
        self.id_to_word
            .iter()
            .enumerate()
            .map(|(id, word)| (word.as_str(), id as WordId))
    }

    /// Iterate over word IDs (excluding special tokens).
    pub fn word_ids(&self) -> impl Iterator<Item = WordId> + '_ {
        2..self.id_to_word.len() as WordId
    }

    /// Get the end-of-sentence ID.
    #[inline]
    pub const fn eos_id(&self) -> WordId {
        EOS_WORD_ID
    }

    /// Get the unknown word ID.
    #[inline]
    pub const fn unk_id(&self) -> WordId {
        UNK_WORD_ID
    }

    /// Convert a sequence of words to a sequence of IDs.
    pub fn encode(&self, words: &[&str]) -> Vec<WordId> {
        words.iter().map(|w| self.get_id_or_unk(w)).collect()
    }

    /// Convert a sequence of IDs to a sequence of words.
    ///
    /// Returns `None` if any ID is invalid.
    pub fn decode(&self, ids: &[WordId]) -> Option<Vec<&str>> {
        ids.iter().map(|&id| self.get_word(id)).collect()
    }
}

impl Default for WordVocabulary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_vocabulary() {
        let vocab = WordVocabulary::new();
        assert_eq!(vocab.len(), 2); // EOS and UNK
        assert_eq!(vocab.get_id(EOS_TOKEN), Some(EOS_WORD_ID));
        assert_eq!(vocab.get_id(UNK_TOKEN), Some(UNK_WORD_ID));
    }

    #[test]
    fn test_add_word() {
        let mut vocab = WordVocabulary::new();
        let id1 = vocab.add_word("hello");
        let id2 = vocab.add_word("world");
        let id3 = vocab.add_word("hello"); // Duplicate

        assert_eq!(id1, 2); // First word after special tokens
        assert_eq!(id2, 3);
        assert_eq!(id3, id1); // Same ID for duplicate
        assert_eq!(vocab.len(), 4);
    }

    #[test]
    fn test_get_word_and_id() {
        let mut vocab = WordVocabulary::new();
        let id = vocab.add_word("test");

        assert_eq!(vocab.get_id("test"), Some(id));
        assert_eq!(vocab.get_word(id), Some("test"));
        assert_eq!(vocab.get_id("nonexistent"), None);
        assert_eq!(vocab.get_word(999), None);
    }

    #[test]
    fn test_get_id_or_unk() {
        let mut vocab = WordVocabulary::new();
        vocab.add_word("known");

        assert_eq!(vocab.get_id_or_unk("known"), 2);
        assert_eq!(vocab.get_id_or_unk("unknown"), UNK_WORD_ID);
    }

    #[test]
    fn test_encode_decode() {
        let mut vocab = WordVocabulary::new();
        vocab.add_word("the");
        vocab.add_word("quick");
        vocab.add_word("fox");

        let words = ["the", "quick", "fox"];
        let ids = vocab.encode(&words);
        let decoded = vocab.decode(&ids).expect("decode failed");

        assert_eq!(decoded, words);
    }

    #[test]
    fn test_iter() {
        let mut vocab = WordVocabulary::new();
        vocab.add_word("a");
        vocab.add_word("b");

        let pairs: Vec<_> = vocab.iter().collect();
        assert_eq!(pairs.len(), 4);
        assert_eq!(pairs[0], (EOS_TOKEN, EOS_WORD_ID));
        assert_eq!(pairs[1], (UNK_TOKEN, UNK_WORD_ID));
        assert_eq!(pairs[2], ("a", 2));
        assert_eq!(pairs[3], ("b", 3));
    }
}
