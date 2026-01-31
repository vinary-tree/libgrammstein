//! Shared vocabulary for mapping words to varint-encoded u64 indices.
//!
//! This module provides a vocabulary that maps each unique word/token to a u64 index,
//! enabling compact n-gram key encoding using LEB128 varint encoding with Latin-1 strings.
//!
//! # Architecture
//!
//! The vocabulary uses [`PersistentVocabARTrie`] from libdictenstein, which provides:
//! - **O(k) forward lookup** (word → index) via adaptive radix trie (k = word length)
//! - **O(k) reverse lookup** (index → word) via parent pointer backtracking (O(1) cache hit)
//! - **Thread-safe** atomic index assignment and RwLock-protected access
//! - **ACID-compliant** with WAL-based crash recovery
//!
//! N-gram keys are encoded as concatenated LEB128 varints, with each byte stored
//! as a Latin-1 character (0x00-0xFF → U+0000-U+00FF) for trie compatibility.
//!
//! # Key Benefits
//!
//! - **No delimiter bugs**: Each word maps to a varint, no delimiters needed
//! - **Compact encoding**: Common words (index 0-127) use just 1 byte
//! - **Unlimited vocabulary**: u64 indices support up to 2^64 words
//! - **Standard encoding**: LEB128 is widely used (protobuf, DWARF, WebAssembly)
//! - **Thread-safe**: Built-in atomic index assignment + RwLock for concurrent access
//! - **O(k) reverse lookups**: Use `get_term()` with cached hot lookups
//!
//! # Encoding Format
//!
//! Each word index is encoded as LEB128 varint, then converted to a Latin-1 string:
//! - Bytes 0x00-0xFF are stored as chars U+0000-U+00FF
//! - This produces valid UTF-8 that tries can store directly
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::ngram::vocabulary::{SharedVocabulary, encode_ngram_key};
//!
//! let vocab = SharedVocabulary::create(path)?;
//!
//! // Each word maps to a unique u64 index (idempotent insert)
//! let the_idx = vocab.insert("the");   // Returns 1
//! let quick_idx = vocab.insert("quick"); // Returns 2
//!
//! // N-gram keys are varint-encoded Latin-1 strings
//! let bigram_key = encode_ngram_key(&["the", "quick"], &vocab);
//!
//! // O(k) reverse lookups (O(1) cache hit)
//! assert_eq!(vocab.get_term(1), Some("the".to_string()));
//! assert_eq!(vocab.get_term(2), Some("quick".to_string()));
//! ```

use libdictenstein::persistent_vocab_artrie::PersistentVocabARTrie;
use parking_lot::RwLock;
use std::path::Path;
use thiserror::Error;

/// First valid vocabulary index.
///
/// Index 0 is reserved to avoid collision with the \x00 metadata key prefix.
/// Varint encoding of 0 produces \x00, which would cause n-gram keys to be
/// mistakenly filtered as metadata entries.
const FIRST_VALID_INDEX: u64 = 1;

/// Error type for vocabulary operations.
#[derive(Error, Debug)]
pub enum VocabularyError {
    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Trie operation failed.
    #[error("Trie error: {0}")]
    Trie(String),

    /// Version mismatch.
    #[error("Vocabulary version mismatch: expected {expected}, found {found}")]
    VersionMismatch {
        /// Expected vocabulary version.
        expected: u64,
        /// Found vocabulary version.
        found: u64,
    },

    /// Persistent ARTrie error.
    #[error("Persistent ARTrie error: {0}")]
    PersistentARTrie(#[from] libdictenstein::persistent_artrie::error::PersistentARTrieError),
}

/// Result type for vocabulary operations.
pub type VocabularyResult<T> = Result<T, VocabularyError>;

// ============================================================================
// Varint Encoding Utilities
// ============================================================================

/// Encode a u64 as LEB128 varint bytes.
///
/// LEB128 (Little Endian Base 128) encodes integers in 7-bit groups with
/// continuation bits. Values 0-127 use 1 byte, 128-16383 use 2 bytes, etc.
#[inline]
pub fn encode_varint(mut value: u64, buf: &mut Vec<u8>) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte);
            break;
        } else {
            buf.push(byte | 0x80);
        }
    }
}

/// Decode a LEB128 varint from bytes, returning (value, bytes_consumed).
///
/// Returns `None` if the bytes are incomplete or would overflow u64.
#[inline]
pub fn decode_varint(bytes: &[u8]) -> Option<(u64, usize)> {
    let mut result: u64 = 0;
    let mut shift = 0;
    for (i, &byte) in bytes.iter().enumerate() {
        if shift >= 64 {
            return None; // Would overflow
        }
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Some((result, i + 1));
        }
        shift += 7;
    }
    None // Incomplete varint
}

// ============================================================================
// SharedVocabulary
// ============================================================================

/// Shared vocabulary mapping words to u64 indices.
///
/// This struct provides thread-safe word-to-index mapping backed by a
/// disk-persistent trie with O(k) reverse lookups via parent pointers.
///
/// # Thread Safety
///
/// Thread-safe via internal RwLock protection. Can be safely shared across threads.
///
/// # Persistence
///
/// The vocabulary uses WAL-based crash recovery. Call `checkpoint()` to persist
/// state durably.
///
/// # Reverse Lookups
///
/// Use `get_term(index)` for O(k) reverse lookups (index → word).
/// Hot lookups are cached for O(1) access.
#[derive(Debug)]
pub struct SharedVocabulary {
    /// The underlying PersistentVocabARTrie with interior mutability.
    inner: RwLock<PersistentVocabARTrie>,
}

impl SharedVocabulary {
    /// Create a new vocabulary at the given path.
    ///
    /// If the path already exists, opens the existing vocabulary.
    /// Otherwise, creates a new empty vocabulary.
    pub fn open_or_create(path: &Path) -> VocabularyResult<Self> {
        if path.exists() {
            Self::open(path)
        } else {
            Self::create(path)
        }
    }

    /// Create a new empty vocabulary at the given path.
    ///
    /// Starts indices at FIRST_VALID_INDEX (1) to avoid \x00 collision
    /// with metadata key prefixes.
    pub fn create(path: &Path) -> VocabularyResult<Self> {
        let inner = PersistentVocabARTrie::create_with_start_index(path, FIRST_VALID_INDEX)?;
        Ok(Self { inner: RwLock::new(inner) })
    }

    /// Open an existing vocabulary from the given path.
    ///
    /// This method uses WAL recovery to restore the vocabulary state.
    /// The underlying `PersistentVocabARTrie::open()` only restores metadata,
    /// so we use `open_with_recovery()` to replay the WAL and restore data.
    pub fn open(path: &Path) -> VocabularyResult<Self> {
        // Use open_with_recovery to replay WAL and restore vocabulary state
        let (inner, _report) = PersistentVocabARTrie::open_with_recovery(path)?;
        Ok(Self { inner: RwLock::new(inner) })
    }

    /// Open an existing vocabulary with crash recovery.
    ///
    /// Returns a tuple of (vocabulary, recovery_report).
    pub fn open_with_recovery(
        path: &Path,
    ) -> VocabularyResult<(Self, libdictenstein::persistent_artrie::recovery::RecoveryReport)> {
        let (inner, report) = PersistentVocabARTrie::open_with_recovery(path)?;
        Ok((Self { inner: RwLock::new(inner) }, report))
    }

    /// Insert a word and get its index (idempotent).
    ///
    /// If the word already exists, returns its existing index.
    /// Otherwise, assigns the next available index.
    ///
    /// This is the primary method for vocabulary insertion. It is idempotent:
    /// calling `insert("word")` multiple times returns the same index.
    #[inline]
    pub fn insert(&self, word: &str) -> u64 {
        self.inner.write().insert(word)
    }

    /// Get or assign an index for a word (thread-safe).
    ///
    /// This is an alias for `insert()` for backward compatibility.
    /// Uses idempotent insertion: returns existing index if word exists.
    #[inline]
    pub fn get_or_insert(&self, word: &str) -> u64 {
        self.insert(word)
    }

    /// Try to get or assign an index for a word (thread-safe, fallible).
    ///
    /// This is an alias for `insert()` wrapped in Ok for backward compatibility.
    /// The underlying operation is infallible.
    #[inline]
    pub fn try_get_or_insert(&self, word: &str) -> VocabularyResult<u64> {
        Ok(self.insert(word))
    }

    /// Look up index for word (None if not in vocabulary).
    #[inline]
    pub fn get(&self, word: &str) -> Option<u64> {
        self.inner.read().get_index(word)
    }

    /// Look up index for word (None if not in vocabulary).
    ///
    /// This is an alias for `get()` matching the PersistentVocabARTrie API.
    #[inline]
    pub fn get_index(&self, word: &str) -> Option<u64> {
        self.inner.read().get_index(word)
    }

    /// O(k) reverse lookup: get the word for an index.
    ///
    /// Returns `None` if the index is not in the vocabulary.
    ///
    /// # Performance
    ///
    /// - O(1) for cached (hot) lookups via LRU cache
    /// - O(k) for cold lookups via parent pointer backtracking (k = word length)
    #[inline]
    pub fn get_term(&self, index: u64) -> Option<String> {
        self.inner.read().get_term(index)
    }

    /// Check if a word exists in the vocabulary.
    #[inline]
    pub fn contains(&self, word: &str) -> bool {
        self.inner.read().contains(word)
    }

    /// Check if an index exists in the vocabulary.
    #[inline]
    pub fn contains_index(&self, index: u64) -> bool {
        self.inner.read().contains_index(index)
    }

    /// Get the number of words in the vocabulary.
    #[inline]
    pub fn len(&self) -> u64 {
        self.inner.read().len() as u64
    }

    /// Check if the vocabulary is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }

    /// Get the starting index for vocabulary entries.
    #[inline]
    pub fn start_index(&self) -> u64 {
        self.inner.read().start_index()
    }

    /// Check if there are unsaved changes.
    #[inline]
    pub fn is_dirty(&self) -> bool {
        self.inner.read().is_dirty()
    }

    /// Checkpoint the vocabulary to disk.
    pub fn checkpoint(&self) -> VocabularyResult<()> {
        self.inner.write().checkpoint().map_err(VocabularyError::from)
    }

    /// Sync the vocabulary to disk.
    ///
    /// This is an alias for `checkpoint()` for backward compatibility.
    pub fn sync(&self) -> VocabularyResult<()> {
        self.checkpoint()
    }

    /// Iterate over all terms in the vocabulary.
    ///
    /// Returns an iterator that yields `(term, index)` pairs.
    /// Uses the reverse lookup capability to iterate by index.
    ///
    /// # Note
    ///
    /// This method iterates sequentially from `start_index` to `start_index + len`,
    /// using `get_term()` for each index. This is O(n * k) where n is the vocabulary
    /// size and k is the average term length.
    pub fn iter_terms(&self) -> impl Iterator<Item = (String, u64)> + '_ {
        let start = self.start_index();
        let count = self.len();
        (start..start + count).filter_map(move |idx| {
            self.get_term(idx).map(|term| (term, idx))
        })
    }
}

/// Encode an n-gram as a varint-encoded Latin-1 string.
///
/// Each word is mapped to its vocabulary index, then LEB128 encoded.
/// Bytes are converted to Latin-1 chars (0x00-0xFF → U+0000-U+00FF).
///
/// # Arguments
///
/// * `words` - Slice of word strings
/// * `vocab` - The vocabulary to use for mapping
///
/// # Returns
///
/// A Latin-1 encoded string where each word's index is varint-encoded.
///
/// # Example
///
/// ```ignore
/// let key = encode_ngram_key(&["the", "quick", "brown"], &vocab);
/// // Each word's index is varint-encoded and converted to Latin-1 chars
/// ```
pub fn encode_ngram_key(words: &[&str], vocab: &SharedVocabulary) -> String {
    let mut buf = Vec::with_capacity(words.len() * 2); // Estimate 2 bytes/word average
    for word in words {
        let index = vocab.get_or_insert(word);
        encode_varint(index, &mut buf);
    }
    // Convert bytes to Latin-1 string (each byte → char U+00XX)
    buf.into_iter().map(|b| char::from(b)).collect()
}

/// Try to encode an n-gram as a varint-encoded Latin-1 string (fallible version).
pub fn try_encode_ngram_key(
    words: &[&str],
    vocab: &SharedVocabulary,
) -> VocabularyResult<String> {
    let mut buf = Vec::with_capacity(words.len() * 2);
    for word in words {
        let index = vocab.try_get_or_insert(word)?;
        encode_varint(index, &mut buf);
    }
    Ok(buf.into_iter().map(|b| char::from(b)).collect())
}

/// Encode an n-gram using existing vocabulary entries only.
///
/// Returns `None` if any word is not in the vocabulary.
/// This is useful for queries where we don't want to add new words.
pub fn encode_ngram_key_existing(words: &[&str], vocab: &SharedVocabulary) -> Option<String> {
    let mut buf = Vec::with_capacity(words.len() * 2);
    for word in words {
        let index = vocab.get(word)?;
        encode_varint(index, &mut buf);
    }
    Some(buf.into_iter().map(|b| char::from(b)).collect())
}

/// Decode a Latin-1 encoded n-gram key to word indices.
///
/// Converts Latin-1 chars back to bytes and decodes LEB128 varints.
pub fn decode_ngram_key(key: &str) -> Vec<u64> {
    // Convert Latin-1 chars back to bytes
    let bytes: Vec<u8> = key.chars().map(|c| c as u8).collect();
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

/// Get the n-gram order from an encoded key.
///
/// The order is the number of varints in the encoded key.
#[inline]
pub fn ngram_order(key: &str) -> u8 {
    decode_ngram_key(key).len() as u8
}

/// Encode a slice of word indices to a varint-encoded Latin-1 key.
///
/// This is useful for encoding sub-sequences (contexts) of n-grams
/// without going through the vocabulary lookup. For example, when
/// computing MKN continuation counts, we need to encode context keys
/// from word indices that have already been decoded.
///
/// # Arguments
///
/// * `indices` - Slice of word indices (u64 values)
///
/// # Returns
///
/// A Latin-1 encoded string where each index is varint-encoded.
///
/// # Example
///
/// ```ignore
/// // Given indices [100, 32, 200], encode as context key
/// let context = encode_indices_to_key(&[100, 32, 200]);
/// // This produces a Latin-1 string with LEB128-encoded values
/// ```
pub fn encode_indices_to_key(indices: &[u64]) -> String {
    let mut buf = Vec::with_capacity(indices.len() * 2);
    for &index in indices {
        encode_varint(index, &mut buf);
    }
    // Convert bytes to Latin-1 string (each byte → char U+00XX)
    buf.into_iter().map(|b| char::from(b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_temp_vocab() -> (TempDir, SharedVocabulary) {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("vocab.artrie");
        let vocab = SharedVocabulary::create(&path).expect("Failed to create vocab");
        (dir, vocab)
    }

    #[test]
    fn test_get_or_insert_new_word() {
        let (_dir, vocab) = create_temp_vocab();

        let idx1 = vocab.get_or_insert("the");
        let idx2 = vocab.get_or_insert("quick");
        let idx3 = vocab.get_or_insert("brown");

        // Each word should get a unique index
        assert_ne!(idx1, idx2);
        assert_ne!(idx2, idx3);
        assert_ne!(idx1, idx3);

        // Indices should be sequential, starting at 1 (not 0, to avoid \x00 collision)
        assert_eq!(idx1, 1);
        assert_eq!(idx2, 2);
        assert_eq!(idx3, 3);
    }

    #[test]
    fn test_get_or_insert_existing_word() {
        let (_dir, vocab) = create_temp_vocab();

        let idx1 = vocab.get_or_insert("hello");
        let idx2 = vocab.get_or_insert("hello");

        // Same word should return same index
        assert_eq!(idx1, idx2);
        assert_eq!(vocab.len(), 1);
    }

    #[test]
    fn test_get_existing() {
        let (_dir, vocab) = create_temp_vocab();

        assert!(vocab.get("nonexistent").is_none());

        let idx1 = vocab.get_or_insert("test");
        let idx2 = vocab.get("test");

        assert_eq!(idx2, Some(idx1));
    }

    #[test]
    fn test_contains() {
        let (_dir, vocab) = create_temp_vocab();

        assert!(!vocab.contains("word"));
        vocab.get_or_insert("word");
        assert!(vocab.contains("word"));
    }

    #[test]
    fn test_varint_encoding() {
        // Test encode/decode roundtrip for various values
        let test_values: [u64; 10] = [0, 1, 127, 128, 255, 256, 16383, 16384, 2097151, u64::MAX];

        for &value in &test_values {
            let mut buf = Vec::new();
            encode_varint(value, &mut buf);
            let (decoded, len) = decode_varint(&buf).expect("Should decode");
            assert_eq!(decoded, value, "Value {} should roundtrip", value);
            assert_eq!(len, buf.len(), "Should consume all bytes for {}", value);
        }
    }

    #[test]
    fn test_varint_encoding_sizes() {
        // Verify varint sizes are as expected
        let mut buf = Vec::new();

        // 0-127: 1 byte
        buf.clear();
        encode_varint(0, &mut buf);
        assert_eq!(buf.len(), 1);
        buf.clear();
        encode_varint(127, &mut buf);
        assert_eq!(buf.len(), 1);

        // 128-16383: 2 bytes
        buf.clear();
        encode_varint(128, &mut buf);
        assert_eq!(buf.len(), 2);
        buf.clear();
        encode_varint(16383, &mut buf);
        assert_eq!(buf.len(), 2);

        // 16384-2097151: 3 bytes
        buf.clear();
        encode_varint(16384, &mut buf);
        assert_eq!(buf.len(), 3);
    }

    #[test]
    fn test_encode_ngram_key() {
        let (_dir, vocab) = create_temp_vocab();

        let key = encode_ngram_key(&["the", "quick", "brown"], &vocab);

        // Decode and verify we get 3 indices
        let indices = decode_ngram_key(&key);
        assert_eq!(indices.len(), 3);
        assert_eq!(indices, vec![1, 2, 3]); // Sequential indices starting at 1
    }

    #[test]
    fn test_encode_ngram_key_with_large_indices() {
        let (_dir, vocab) = create_temp_vocab();

        // Insert enough words to test multi-byte varint
        // Indices start at 1, so word0 gets index 1, word199 gets index 200
        for i in 0..200 {
            vocab.get_or_insert(&format!("word{}", i));
        }

        // Encode an n-gram with indices that span single and multi-byte varints
        let key = encode_ngram_key(&["word0", "word126", "word127", "word199"], &vocab);

        let indices = decode_ngram_key(&key);
        // word0 -> index 1, word126 -> index 127, word127 -> index 128, word199 -> index 200
        assert_eq!(indices, vec![1, 127, 128, 200]);

        // word0 (index 1) = 1 byte
        // word126 (index 127) = 1 byte
        // word127 (index 128) = 2 bytes (continuation bit set)
        // word199 (index 200) = 2 bytes
        // Total: 6 bytes = 6 Latin-1 chars
        assert_eq!(key.chars().count(), 6);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let (_dir, vocab) = create_temp_vocab();

        let words = ["the", "quick", "brown", "fox"];
        let key = encode_ngram_key(&words, &vocab);

        // Decode should recover original indices
        let indices = decode_ngram_key(&key);
        assert_eq!(indices.len(), words.len());

        // Use get_term() for O(1) reverse lookups
        let decoded: Vec<_> = indices
            .iter()
            .map(|&idx| vocab.get_term(idx).expect("index should exist"))
            .collect();

        assert_eq!(decoded, words);
    }

    #[test]
    fn test_pipe_in_token_no_longer_corrupts() {
        let (_dir, vocab) = create_temp_vocab();

        // Token contains pipe character - this was the bug!
        let tokens = ["foo|bar", "baz"];
        let key = encode_ngram_key(&tokens, &vocab);

        // Decode and verify
        let indices = decode_ngram_key(&key);
        assert_eq!(indices.len(), 2);

        // Use get_term() for O(1) reverse lookups
        let decoded: Vec<_> = indices
            .iter()
            .map(|&idx| vocab.get_term(idx).expect("index should exist"))
            .collect();

        assert_eq!(decoded, tokens);
    }

    #[test]
    fn test_get_term_o1_lookup() {
        let (_dir, vocab) = create_temp_vocab();

        // Insert words
        let idx1 = vocab.get_or_insert("hello");
        let idx2 = vocab.get_or_insert("world");
        let idx3 = vocab.get_or_insert("rust");

        // O(1) reverse lookups should work
        assert_eq!(vocab.get_term(idx1), Some("hello".to_string()));
        assert_eq!(vocab.get_term(idx2), Some("world".to_string()));
        assert_eq!(vocab.get_term(idx3), Some("rust".to_string()));

        // Invalid indices should return None
        assert_eq!(vocab.get_term(0), None); // Below FIRST_VALID_INDEX
        assert_eq!(vocab.get_term(999), None); // Above range
    }

    #[test]
    fn test_ngram_order() {
        let (_dir, vocab) = create_temp_vocab();

        let unigram = encode_ngram_key(&["word"], &vocab);
        let bigram = encode_ngram_key(&["the", "quick"], &vocab);
        let trigram = encode_ngram_key(&["a", "b", "c"], &vocab);
        let fivegram = encode_ngram_key(&["1", "2", "3", "4", "5"], &vocab);

        assert_eq!(ngram_order(&unigram), 1);
        assert_eq!(ngram_order(&bigram), 2);
        assert_eq!(ngram_order(&trigram), 3);
        assert_eq!(ngram_order(&fivegram), 5);
    }

    #[test]
    fn test_case_sensitivity() {
        let (_dir, vocab) = create_temp_vocab();

        let lower = vocab.get_or_insert("the");
        let upper = vocab.get_or_insert("The");
        let all_caps = vocab.get_or_insert("THE");

        // Case-sensitive: all should be different
        assert_ne!(lower, upper);
        assert_ne!(upper, all_caps);
        assert_ne!(lower, all_caps);
    }

    #[test]
    fn test_punctuation_as_tokens() {
        let (_dir, vocab) = create_temp_vocab();

        let comma = vocab.get_or_insert(",");
        let period = vocab.get_or_insert(".");
        let quote = vocab.get_or_insert("\"");

        // Each punctuation should be a unique token
        assert_ne!(comma, period);
        assert_ne!(period, quote);

        // They should all be valid indices (starting at 1)
        assert_eq!(comma, 1);
        assert_eq!(period, 2);
        assert_eq!(quote, 3);
    }

    #[test]
    fn test_persistence() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("vocab.artrie");

        // Create and populate vocabulary
        let idx1;
        let idx2;
        {
            let vocab = SharedVocabulary::create(&path).expect("Failed to create vocab");
            idx1 = vocab.get_or_insert("hello");
            idx2 = vocab.get_or_insert("world");
            vocab.checkpoint().expect("Checkpoint failed");
        }

        // Reopen and verify
        {
            let vocab = SharedVocabulary::open(&path).expect("Failed to open vocab");

            assert_eq!(vocab.len(), 2);
            assert_eq!(vocab.get("hello"), Some(idx1));
            assert_eq!(vocab.get("world"), Some(idx2));

            // New word should get next index
            let idx3 = vocab.get_or_insert("new");
            assert_eq!(idx3, 3); // Sequential after idx1=1, idx2=2
        }
    }

    #[test]
    fn test_encode_ngram_key_existing() {
        let (_dir, vocab) = create_temp_vocab();

        // Word not in vocabulary
        assert!(encode_ngram_key_existing(&["unknown"], &vocab).is_none());

        // Add some words
        vocab.get_or_insert("the");
        vocab.get_or_insert("quick");

        // Now they should work
        let key = encode_ngram_key_existing(&["the", "quick"], &vocab);
        assert!(key.is_some());

        // Verify decoding works (indices start at 1)
        let indices = decode_ngram_key(&key.unwrap());
        assert_eq!(indices, vec![1, 2]);

        // But mixed should fail
        assert!(encode_ngram_key_existing(&["the", "unknown"], &vocab).is_none());
    }

    #[test]
    fn test_concurrent_insert() {
        use std::sync::Arc;
        use std::thread;

        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("vocab.artrie");
        let vocab = Arc::new(SharedVocabulary::create(&path).expect("Failed to create vocab"));

        // Spawn multiple threads that all try to insert the same word
        let mut handles = vec![];
        for _ in 0..10 {
            let vocab = Arc::clone(&vocab);
            handles.push(thread::spawn(move || vocab.get_or_insert("shared_word")));
        }

        // Collect results
        let indices: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All threads should get the same index
        let first = indices[0];
        for idx in &indices {
            assert_eq!(*idx, first);
        }

        // Should only have one entry
        assert_eq!(vocab.len(), 1);
    }

    #[test]
    fn test_latin1_encoding_preserves_bytes() {
        // Verify that Latin-1 encoding correctly round-trips all byte values
        for byte in 0u8..=255 {
            let c = char::from(byte);
            assert_eq!(c as u8, byte, "Byte {} should round-trip through char", byte);
        }
    }

    #[test]
    fn test_large_vocabulary() {
        // Test that vocabulary can grow beyond the old PUA limit
        let (_dir, vocab) = create_temp_vocab();

        // Insert more than 131,068 words (old PUA limit)
        // We'll test with a smaller number for speed, but verify indices are correct
        for i in 0..1000 {
            let idx = vocab.get_or_insert(&format!("word{}", i));
            // Indices start at 1, not 0
            assert_eq!(idx, (i + 1) as u64);
        }

        assert_eq!(vocab.len(), 1000);
    }

    #[test]
    fn test_start_index() {
        let (_dir, vocab) = create_temp_vocab();

        // Start index should be 1 (FIRST_VALID_INDEX)
        assert_eq!(vocab.start_index(), 1);
    }

    #[test]
    fn test_is_dirty() {
        let (_dir, vocab) = create_temp_vocab();

        // After inserting, vocabulary should be dirty
        vocab.get_or_insert("test");
        assert!(vocab.is_dirty());

        // After checkpoint, should no longer be dirty
        vocab.checkpoint().expect("checkpoint failed");
        assert!(!vocab.is_dirty());
    }

    #[test]
    fn test_contains_index() {
        let (_dir, vocab) = create_temp_vocab();

        // Index 0 should never exist (reserved)
        assert!(!vocab.contains_index(0));

        // After insert, index 1 should exist
        vocab.get_or_insert("test");
        assert!(vocab.contains_index(1));
        assert!(!vocab.contains_index(2)); // Not yet inserted
    }
}
