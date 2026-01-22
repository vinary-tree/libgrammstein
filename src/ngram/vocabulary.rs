//! Shared vocabulary for mapping words to PUA (Private Use Area) characters.
//!
//! This module provides a vocabulary that maps each unique word/token directly to a
//! Unicode Private Use Area character, which enables compact n-gram key encoding
//! without delimiters.
//!
//! # Architecture
//!
//! The vocabulary maps each unique word to a PUA character in the range U+F0000-U+FFFFD
//! (Supplementary Private Use Area-A). This range provides 65,534 code points, which
//! is sufficient for most vocabularies. For larger vocabularies, the range can be
//! extended to include Supplementary Private Use Area-B (U+100000-U+10FFFD).
//!
//! # Key Benefits
//!
//! - **No delimiter bugs**: Each word maps to a single character, so no delimiter
//!   character can appear within a token
//! - **Compact encoding**: N-gram keys are sequences of 4-byte UTF-8 characters
//! - **DAWG suffix sharing**: Common n-gram endings share trie nodes
//! - **Thread-safe**: Uses double-checked locking for concurrent access
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::ngram::vocabulary::SharedVocabulary;
//!
//! let vocab = SharedVocabulary::create(path)?;
//!
//! // Each word maps to a unique PUA character
//! let the_char = vocab.get_or_insert("the");
//! let quick_char = vocab.get_or_insert("quick");
//!
//! // N-gram keys are sequences of PUA characters
//! let bigram_key: String = vec![the_char, quick_char].into_iter().collect();
//! ```

use liblevenshtein::dictionary::persistent_artrie_char::DiskBackedCharTrieInner;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use thiserror::Error;

/// Start of the Private Use Area-A range (U+F0000).
const PUA_START: u32 = 0xF0000;

/// End of the Private Use Area-A range (U+FFFFD).
const PUA_END_A: u32 = 0xFFFFD;

/// End of the Private Use Area-B range (U+10FFFD).
const PUA_END_B: u32 = 0x10FFFD;

/// Maximum vocabulary size (PUA-A + PUA-B = 131,068 code points).
pub const MAX_VOCABULARY_SIZE: u32 = (PUA_END_A - PUA_START + 1) + (PUA_END_B - 0x100000 + 1);

/// Metadata key for the next character code point.
const META_NEXT_CHAR: &str = "\x00__meta__:next_char";

/// Metadata key for the vocabulary version.
const META_VERSION: &str = "\x00__meta__:version";

/// Current vocabulary format version.
const VOCABULARY_VERSION: u64 = 1;

/// Error type for vocabulary operations.
#[derive(Error, Debug)]
pub enum VocabularyError {
    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Trie operation failed.
    #[error("Trie error: {0}")]
    Trie(String),

    /// Vocabulary capacity exceeded.
    #[error("Vocabulary capacity exceeded: {current} >= {max}")]
    CapacityExceeded {
        /// Current vocabulary size.
        current: u32,
        /// Maximum allowed vocabulary size.
        max: u32,
    },

    /// Invalid PUA character stored in trie.
    #[error("Invalid PUA character stored: {0}")]
    InvalidPuaChar(u64),

    /// Version mismatch.
    #[error("Vocabulary version mismatch: expected {expected}, found {found}")]
    VersionMismatch {
        /// Expected vocabulary version.
        expected: u64,
        /// Found vocabulary version.
        found: u64,
    },
}

/// Result type for vocabulary operations.
pub type VocabularyResult<T> = Result<T, VocabularyError>;

/// Shared vocabulary mapping words to PUA characters.
///
/// This struct provides thread-safe word-to-character mapping backed by a
/// disk-persistent trie. Each unique word is assigned a unique character
/// from the Unicode Private Use Area.
///
/// # Thread Safety
///
/// Uses double-checked locking to ensure only one thread assigns a character
/// for each word. The `fetch_add` on `next_char` is only called while holding
/// the write lock, preventing TOCTOU race conditions.
///
/// # Persistence
///
/// The vocabulary is stored in a `DiskBackedCharTrieInner<u64>` with WAL-based
/// crash recovery. The next character code point is stored in the trie as
/// metadata to survive restarts.
#[derive(Debug)]
pub struct SharedVocabulary {
    /// The underlying trie mapping words to PUA characters (stored as u64).
    trie: Arc<RwLock<DiskBackedCharTrieInner<u64>>>,

    /// Next PUA code point to assign (relative to PUA_START).
    next_char: AtomicU32,
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
    pub fn create(path: &Path) -> VocabularyResult<Self> {
        let trie = DiskBackedCharTrieInner::create(path)
            .map_err(|e| VocabularyError::Trie(format!("Failed to create vocabulary: {}", e)))?;

        let trie = Arc::new(RwLock::new(trie));

        // Initialize metadata
        {
            let mut guard = trie.write();
            guard.upsert(META_VERSION, VOCABULARY_VERSION).map_err(|e| {
                VocabularyError::Trie(format!("Failed to initialize version: {}", e))
            })?;
            guard.upsert(META_NEXT_CHAR, 0u64).map_err(|e| {
                VocabularyError::Trie(format!("Failed to initialize next_char: {}", e))
            })?;
        }

        Ok(Self {
            trie,
            next_char: AtomicU32::new(0),
        })
    }

    /// Open an existing vocabulary from the given path.
    pub fn open(path: &Path) -> VocabularyResult<Self> {
        let trie = DiskBackedCharTrieInner::open(path)
            .map_err(|e| VocabularyError::Trie(format!("Failed to open vocabulary: {}", e)))?;

        // Check version
        if let Some(&version) = trie.get(META_VERSION) {
            if version != VOCABULARY_VERSION {
                return Err(VocabularyError::VersionMismatch {
                    expected: VOCABULARY_VERSION,
                    found: version,
                });
            }
        }

        // Load next_char from metadata
        let next_char = trie
            .get(META_NEXT_CHAR)
            .map(|&v| v as u32)
            .unwrap_or(0);

        Ok(Self {
            trie: Arc::new(RwLock::new(trie)),
            next_char: AtomicU32::new(next_char),
        })
    }

    /// Get or assign a PUA character for a word (thread-safe).
    ///
    /// Uses double-checked locking to avoid TOCTOU race conditions.
    /// If the word already exists, returns its existing PUA character.
    /// Otherwise, assigns the next available PUA character.
    ///
    /// # Panics
    ///
    /// Panics if the vocabulary capacity is exceeded. Use `try_get_or_insert`
    /// for fallible insertion.
    pub fn get_or_insert(&self, word: &str) -> char {
        self.try_get_or_insert(word)
            .expect("Vocabulary capacity exceeded")
    }

    /// Try to get or assign a PUA character for a word (thread-safe, fallible).
    ///
    /// Returns `Err` if the vocabulary capacity is exceeded.
    pub fn try_get_or_insert(&self, word: &str) -> VocabularyResult<char> {
        // Fast path: read lock to check if word exists
        {
            let guard = self.trie.read();
            if let Some(&code) = guard.get(word) {
                return Self::u64_to_pua_char(code);
            }
        }

        // Slow path: acquire write lock and re-check
        let mut guard = self.trie.write();

        // Re-check after acquiring write lock (another thread may have inserted)
        if let Some(&code) = guard.get(word) {
            return Self::u64_to_pua_char(code);
        }

        // This thread wins - assign the next PUA character
        let relative_code = self.next_char.fetch_add(1, Ordering::SeqCst);
        let pua_char = Self::relative_to_pua_char(relative_code)?;

        // Store the word -> PUA char mapping
        guard.upsert(word, pua_char as u64).map_err(|e| {
            VocabularyError::Trie(format!("Failed to insert word '{}': {}", word, e))
        })?;

        // Update the metadata for next_char
        guard.upsert(META_NEXT_CHAR, (relative_code + 1) as u64).map_err(|e| {
            VocabularyError::Trie(format!("Failed to update next_char metadata: {}", e))
        })?;

        Ok(pua_char)
    }

    /// Look up PUA character for word (None if not in vocabulary).
    pub fn get(&self, word: &str) -> Option<char> {
        self.trie
            .read()
            .get(word)
            .and_then(|&code| Self::u64_to_pua_char(code).ok())
    }

    /// Check if a word exists in the vocabulary.
    pub fn contains(&self, word: &str) -> bool {
        self.trie.read().get(word).is_some()
    }

    /// Get the number of words in the vocabulary.
    pub fn len(&self) -> u32 {
        self.next_char.load(Ordering::Relaxed)
    }

    /// Check if the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Build a reverse lookup cache (PUA char -> word).
    ///
    /// This is useful for decoding n-gram keys back to words.
    /// Note: This iterates over all entries and allocates a HashMap.
    pub fn build_reverse_cache(&self) -> HashMap<char, String> {
        let guard = self.trie.read();
        let mut cache = HashMap::new();

        // Iterate over all entries using prefix iteration with empty prefix
        // This returns all (term, value) pairs in the trie
        if let Ok(Some(entries)) = guard.iter_prefix_with_values("") {
            for (key, value) in entries {
                // Skip metadata keys (start with \x00)
                if key.starts_with('\x00') {
                    continue;
                }

                if let Ok(pua_char) = Self::u64_to_pua_char(value) {
                    cache.insert(pua_char, key);
                }
            }
        }

        cache
    }

    /// Checkpoint the vocabulary to disk.
    pub fn checkpoint(&self) -> VocabularyResult<()> {
        let mut guard = self.trie.write();

        // Ensure next_char metadata is up to date
        let current = self.next_char.load(Ordering::Relaxed);
        guard.upsert(META_NEXT_CHAR, current as u64).map_err(|e| {
            VocabularyError::Trie(format!("Failed to update next_char metadata: {}", e))
        })?;

        guard.checkpoint().map_err(|e| {
            VocabularyError::Trie(format!("Checkpoint failed: {}", e))
        })
    }

    /// Sync the vocabulary to disk (WAL flush).
    pub fn sync(&self) -> VocabularyResult<()> {
        self.trie.write().sync().map_err(|e| {
            VocabularyError::Trie(format!("Sync failed: {}", e))
        })
    }

    /// Convert a relative code point (0-based) to a PUA character.
    fn relative_to_pua_char(relative: u32) -> VocabularyResult<char> {
        let absolute = if relative <= (PUA_END_A - PUA_START) {
            // Within PUA-A range
            PUA_START + relative
        } else {
            // Overflow to PUA-B range
            let pua_b_offset = relative - (PUA_END_A - PUA_START + 1);
            let pua_b_start = 0x100000u32;
            if pua_b_start + pua_b_offset > PUA_END_B {
                return Err(VocabularyError::CapacityExceeded {
                    current: relative,
                    max: MAX_VOCABULARY_SIZE,
                });
            }
            pua_b_start + pua_b_offset
        };

        char::from_u32(absolute).ok_or_else(|| VocabularyError::InvalidPuaChar(absolute as u64))
    }

    /// Convert a u64 stored in the trie to a PUA character.
    fn u64_to_pua_char(code: u64) -> VocabularyResult<char> {
        let code_u32 = code as u32;
        char::from_u32(code_u32).ok_or_else(|| VocabularyError::InvalidPuaChar(code))
    }

    /// Get a clone of the Arc-wrapped trie (for sharing across threads).
    pub fn trie_arc(&self) -> Arc<RwLock<DiskBackedCharTrieInner<u64>>> {
        Arc::clone(&self.trie)
    }
}

/// Encode an n-gram as a trie key (sequence of PUA chars).
///
/// Each word in the n-gram is mapped to its PUA character, and the
/// resulting characters are collected into a String.
///
/// # Arguments
///
/// * `words` - Slice of word strings
/// * `vocab` - The vocabulary to use for mapping
///
/// # Returns
///
/// A String where each character is a PUA character representing one word.
///
/// # Example
///
/// ```ignore
/// let key = encode_ngram_key(&["the", "quick", "brown"], &vocab);
/// assert_eq!(key.chars().count(), 3);  // 3 PUA chars
/// ```
pub fn encode_ngram_key(words: &[&str], vocab: &SharedVocabulary) -> String {
    words.iter().map(|w| vocab.get_or_insert(w)).collect()
}

/// Try to encode an n-gram as a trie key (fallible version).
pub fn try_encode_ngram_key(
    words: &[&str],
    vocab: &SharedVocabulary,
) -> VocabularyResult<String> {
    words
        .iter()
        .map(|w| vocab.try_get_or_insert(w))
        .collect()
}

/// Encode an n-gram using existing vocabulary entries only.
///
/// Returns `None` if any word is not in the vocabulary.
/// This is useful for queries where we don't want to add new words.
pub fn encode_ngram_key_existing(words: &[&str], vocab: &SharedVocabulary) -> Option<String> {
    words
        .iter()
        .map(|w| vocab.get(w))
        .collect::<Option<String>>()
}

/// Decode an n-gram key to its PUA character sequence.
///
/// This is a simple wrapper that returns an iterator over the characters.
/// To get the original words, use a reverse lookup cache from `SharedVocabulary::build_reverse_cache`.
pub fn decode_ngram_key(key: &str) -> impl Iterator<Item = char> + '_ {
    key.chars()
}

/// Get the n-gram order from an encoded key.
///
/// The order is simply the number of characters in the key.
#[inline]
pub fn ngram_order(key: &str) -> u8 {
    key.chars().count() as u8
}

/// Check if a character is a PUA character.
#[inline]
pub fn is_pua_char(c: char) -> bool {
    let code = c as u32;
    (PUA_START..=PUA_END_A).contains(&code) || (0x100000..=PUA_END_B).contains(&code)
}

/// Convert a PUA character to its relative index (0-based).
///
/// Returns `None` if the character is not a PUA character.
pub fn pua_char_to_index(c: char) -> Option<u32> {
    let code = c as u32;
    if (PUA_START..=PUA_END_A).contains(&code) {
        Some(code - PUA_START)
    } else if (0x100000..=PUA_END_B).contains(&code) {
        Some((PUA_END_A - PUA_START + 1) + (code - 0x100000))
    } else {
        None
    }
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

        let char1 = vocab.get_or_insert("the");
        let char2 = vocab.get_or_insert("quick");
        let char3 = vocab.get_or_insert("brown");

        // Each word should get a unique character
        assert_ne!(char1, char2);
        assert_ne!(char2, char3);
        assert_ne!(char1, char3);

        // All should be PUA characters
        assert!(is_pua_char(char1));
        assert!(is_pua_char(char2));
        assert!(is_pua_char(char3));
    }

    #[test]
    fn test_get_or_insert_existing_word() {
        let (_dir, vocab) = create_temp_vocab();

        let char1 = vocab.get_or_insert("hello");
        let char2 = vocab.get_or_insert("hello");

        // Same word should return same character
        assert_eq!(char1, char2);
        assert_eq!(vocab.len(), 1);
    }

    #[test]
    fn test_get_existing() {
        let (_dir, vocab) = create_temp_vocab();

        assert!(vocab.get("nonexistent").is_none());

        let char1 = vocab.get_or_insert("test");
        let char2 = vocab.get("test");

        assert_eq!(char2, Some(char1));
    }

    #[test]
    fn test_contains() {
        let (_dir, vocab) = create_temp_vocab();

        assert!(!vocab.contains("word"));
        vocab.get_or_insert("word");
        assert!(vocab.contains("word"));
    }

    #[test]
    fn test_encode_ngram_key() {
        let (_dir, vocab) = create_temp_vocab();

        let key = encode_ngram_key(&["the", "quick", "brown"], &vocab);

        // Should have 3 characters
        assert_eq!(key.chars().count(), 3);

        // All characters should be PUA
        for c in key.chars() {
            assert!(is_pua_char(c));
        }
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let (_dir, vocab) = create_temp_vocab();

        let words = ["the", "quick", "brown", "fox"];
        let key = encode_ngram_key(&words, &vocab);

        // Build reverse cache
        let reverse = vocab.build_reverse_cache();

        // Decode should recover original words
        let decoded: Vec<_> = decode_ngram_key(&key)
            .map(|c| reverse.get(&c).unwrap().as_str())
            .collect();

        assert_eq!(decoded, words);
    }

    #[test]
    fn test_pipe_in_token_no_longer_corrupts() {
        let (_dir, vocab) = create_temp_vocab();

        // Token contains pipe character - this was the bug!
        let tokens = ["foo|bar", "baz"];
        let key = encode_ngram_key(&tokens, &vocab);

        // Key should have exactly 2 characters (one per word)
        assert_eq!(key.chars().count(), 2);

        // Build reverse cache and verify roundtrip
        let reverse = vocab.build_reverse_cache();
        let decoded: Vec<_> = decode_ngram_key(&key)
            .map(|c| reverse.get(&c).unwrap().as_str())
            .collect();

        assert_eq!(decoded, tokens);
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

        // They should all be valid PUA chars
        assert!(is_pua_char(comma));
        assert!(is_pua_char(period));
        assert!(is_pua_char(quote));
    }

    #[test]
    fn test_persistence() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("vocab.artrie");

        // Create and populate vocabulary
        let char1;
        let char2;
        {
            let vocab = SharedVocabulary::create(&path).expect("Failed to create vocab");
            char1 = vocab.get_or_insert("hello");
            char2 = vocab.get_or_insert("world");
            vocab.checkpoint().expect("Checkpoint failed");
        }

        // Reopen and verify
        {
            let vocab = SharedVocabulary::open(&path).expect("Failed to open vocab");

            assert_eq!(vocab.len(), 2);
            assert_eq!(vocab.get("hello"), Some(char1));
            assert_eq!(vocab.get("world"), Some(char2));

            // New word should get next index
            let char3 = vocab.get_or_insert("new");
            assert_ne!(char3, char1);
            assert_ne!(char3, char2);
        }
    }

    #[test]
    fn test_pua_char_to_index_roundtrip() {
        for i in [0, 1, 100, 1000, 10000, 65533] {
            let pua_char =
                SharedVocabulary::relative_to_pua_char(i).expect("Should be valid");
            let index = pua_char_to_index(pua_char).expect("Should have index");
            assert_eq!(index, i, "Index {} should roundtrip", i);
        }
    }

    #[test]
    fn test_is_pua_char() {
        // Valid PUA-A chars
        assert!(is_pua_char('\u{F0000}'));
        assert!(is_pua_char('\u{F0001}'));
        assert!(is_pua_char('\u{FFFFD}'));

        // Valid PUA-B chars
        assert!(is_pua_char('\u{100000}'));
        assert!(is_pua_char('\u{10FFFD}'));

        // Invalid (regular chars)
        assert!(!is_pua_char('a'));
        assert!(!is_pua_char('Z'));
        assert!(!is_pua_char('|'));
        assert!(!is_pua_char('\u{1F600}')); // Emoji
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
        assert_eq!(key.unwrap().chars().count(), 2);

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
            handles.push(thread::spawn(move || {
                vocab.get_or_insert("shared_word")
            }));
        }

        // Collect results
        let chars: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All threads should get the same character
        let first = chars[0];
        for c in &chars {
            assert_eq!(*c, first);
        }

        // Should only have one entry
        assert_eq!(vocab.len(), 1);
    }
}
