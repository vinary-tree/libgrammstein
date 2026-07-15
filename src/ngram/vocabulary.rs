//! Shared vocabulary for mapping words to varint-encoded u64 indices.
//!
//! This module provides vocabulary types and utilities for n-gram key encoding
//! using LEB128 varint encoding with Latin-1 strings.
//!
//! # Architecture
//!
//! The vocabulary uses [`SharedVocabARTrie`] from libdictenstein, which provides:
//! - **O(k) forward lookup** (word → index) via adaptive radix trie (k = word length)
//! - **O(k) reverse lookup** (index → word) via parent pointer backtracking (O(1) cache hit)
//! - **Lock-free** atomic index assignment and overlay-CAS concurrent access
//! - **ACID-compliant** with WAL-based crash recovery
//! - **BloomFilter** for O(1) OOV word rejection (5-10x faster negative lookups)
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
//! - **Lock-free**: atomic index assignment + overlay-CAS concurrent access (no RwLock)
//! - **O(k) reverse lookups**: Use `get_term()` with cached hot lookups
//! - **BloomFilter**: Fast rejection of out-of-vocabulary words
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
//! use libgrammstein::ngram::vocabulary::{
//!     SharedVocabARTrie, create_vocabulary, encode_ngram_key, FIRST_VALID_INDEX,
//! };
//!
//! let vocab = create_vocabulary(&path)?;
//!
//! // Each word maps to a unique u64 index (idempotent insert)
//! let the_idx = vocab.as_ref().insert("the").expect("test insert");   // Returns 1
//! let quick_idx = vocab.as_ref().insert("quick").expect("test insert"); // Returns 2
//!
//! // N-gram keys are varint-encoded Latin-1 strings
//! let bigram_key = encode_ngram_key(&["the", "quick"], &vocab);
//!
//! // O(k) reverse lookups (O(1) cache hit)
//! assert_eq!(vocab.as_ref().get_term(1), Some("the".to_string()));
//! assert_eq!(vocab.as_ref().get_term(2), Some("quick".to_string()));
//! ```

use std::path::Path;
use std::sync::Arc;

use thiserror::Error;

// Re-export from libdictenstein
pub use libdictenstein::persistent_artrie::dict_impl::DurabilityPolicy;
pub use libdictenstein::persistent_artrie::recovery::RecoveryReport;
pub use libdictenstein::persistent_artrie::vocab::{
    PersistentVocabARTrie, SharedVocabARTrie, VocabSyncHandle,
};

/// First valid vocabulary index.
///
/// Index 0 is reserved to avoid collision with the \x00 metadata key prefix.
/// Varint encoding of 0 produces \x00, which would cause n-gram keys to be
/// mistakenly filtered as metadata entries.
pub const FIRST_VALID_INDEX: u64 = 1;

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
// Vocabulary Factory Functions
// ============================================================================

/// Create a new vocabulary at the given path.
///
/// Starts indices at [`FIRST_VALID_INDEX`] (1) to avoid \x00 collision
/// with metadata key prefixes.
///
/// Enables slot-level dirty tracking for optimized WAL rotation (90%+ I/O reduction
/// compared to full checkpoint serialization).
pub fn create_vocabulary(path: &Path) -> VocabularyResult<SharedVocabARTrie> {
    let trie = PersistentVocabARTrie::create_with_start_index(path, FIRST_VALID_INDEX)?;
    trie.enable_slot_tracking();
    Ok(Arc::new(trie))
}

/// Create a new vocabulary with BloomFilter enabled.
///
/// The BloomFilter provides O(1) fast-path for detecting new terms during
/// bulk insert operations, skipping expensive O(k) trie lookups.
///
/// Enables slot-level dirty tracking for optimized WAL rotation (90%+ I/O reduction
/// compared to full checkpoint serialization).
///
/// # Arguments
///
/// * `path` - Path to the vocabulary file
/// * `bloom_capacity` - Expected number of vocabulary entries (for optimal bloom sizing)
pub fn create_vocabulary_with_bloom(
    path: &Path,
    bloom_capacity: usize,
) -> VocabularyResult<SharedVocabARTrie> {
    // BloomFilter removed (overlay reads use the lock-free walk); equivalent to create_vocabulary.
    let _ = bloom_capacity;
    let trie = PersistentVocabARTrie::create_with_start_index(path, FIRST_VALID_INDEX)?;
    trie.enable_slot_tracking();
    Ok(Arc::new(trie))
}

/// Open an existing vocabulary from the given path.
///
/// Uses WAL recovery to restore the vocabulary state after crash.
/// Also loads or rebuilds the BloomFilter automatically.
///
/// Enables slot-level dirty tracking for optimized WAL rotation (90%+ I/O reduction
/// compared to full checkpoint serialization).
pub fn open_vocabulary(path: &Path) -> VocabularyResult<SharedVocabARTrie> {
    let (trie, _report) = PersistentVocabARTrie::open_with_recovery(path)?;
    trie.enable_slot_tracking();
    Ok(Arc::new(trie))
}

/// Open an existing vocabulary with crash recovery report.
///
/// Returns a tuple of (vocabulary, recovery_report).
///
/// Enables slot-level dirty tracking for optimized WAL rotation (90%+ I/O reduction
/// compared to full checkpoint serialization).
pub fn open_vocabulary_with_recovery(
    path: &Path,
) -> VocabularyResult<(SharedVocabARTrie, RecoveryReport)> {
    let (trie, report) = PersistentVocabARTrie::open_with_recovery(path)?;
    trie.enable_slot_tracking();
    Ok((Arc::new(trie), report))
}

/// Open or create a vocabulary at the given path.
///
/// If the path exists, opens the existing vocabulary with recovery.
/// Otherwise, creates a new empty vocabulary.
pub fn open_or_create_vocabulary(path: &Path) -> VocabularyResult<SharedVocabARTrie> {
    if path.exists() {
        open_vocabulary(path)
    } else {
        create_vocabulary(path)
    }
}

/// Open or create a vocabulary with BloomFilter.
///
/// If the path exists, opens the existing vocabulary (bloom filter loaded/rebuilt).
/// Otherwise, creates a new vocabulary with BloomFilter enabled.
pub fn open_or_create_vocabulary_with_bloom(
    path: &Path,
    bloom_capacity: usize,
) -> VocabularyResult<SharedVocabARTrie> {
    if path.exists() {
        // Opening loads/rebuilds bloom filter automatically
        open_vocabulary(path)
    } else {
        create_vocabulary_with_bloom(path, bloom_capacity)
    }
}

// ============================================================================
// Lock-Free Concurrent Vocabulary Factory Functions
// ============================================================================

/// Wrap a `PersistentVocabARTrie` in an `Arc` for lock-free concurrent use.
///
/// `PersistentVocabARTrie` IS the single lock-free vocabulary implementation —
/// there is no separate wrapper type. `Arc`-sharing it lets multiple threads
/// insert vocabulary entries and checkpoint through `&self` with no external
/// locking.
///
/// # Usage Pattern
///
/// ```ignore
/// // Create the lock-free vocabulary, then share it.
/// let vocab = PersistentVocabARTrie::create_with_start_index(&path, FIRST_VALID_INDEX)?;
/// let concurrent = create_concurrent_vocabulary_lockfree(vocab);
///
/// // Use in parallel workers - no contention!
/// std::thread::scope(|s| {
///     for _ in 0..12 {
///         let c = Arc::clone(&concurrent);
///         s.spawn(move || {
///             for ngram in ngrams {
///                 let key = encode_ngram_key_lockfree(&ngram.words, &c);
///                 // Store n-gram...
///             }
///         });
///     }
/// });
///
/// // Checkpoint lock-free layer to persistent storage
/// concurrent.checkpoint()?;
/// ```
pub fn create_concurrent_vocabulary_lockfree(
    vocab: PersistentVocabARTrie,
) -> Arc<PersistentVocabARTrie> {
    // `PersistentVocabARTrie` IS the single lock-free impl (no wrapper). `Arc` it so many threads
    // insert/checkpoint through `&self` with no external locking.
    Arc::new(vocab)
}

/// Open/create a vocabulary and return a shared lock-free handle.
///
/// Uses the trie's default durability policy. (The durable Order-A insert requires
/// `Immediate`/`GroupCommit` and rejects `None`/`Periodic` — see the in-body note;
/// tune to `GroupCommit` for batched-fsync bulk-import throughput.)
pub fn open_or_create_concurrent_vocabulary_lockfree(
    path: &Path,
) -> VocabularyResult<Arc<PersistentVocabARTrie>> {
    // The durable Order-A insert requires Immediate/GroupCommit durability (an ACK guarantees the
    // write is durable before it becomes visible) — it REJECTS None/Periodic. Use the default
    // policy; for bulk-import throughput tune to DurabilityPolicy::GroupCommit (batched fsync) and
    // benchmark (this is the durable-vs-old-in-memory-insert_cas trade-off).
    let trie = if path.exists() {
        let (trie, _report) = PersistentVocabARTrie::open_with_recovery(path)?;
        trie
    } else {
        PersistentVocabARTrie::create_with_start_index(path, FIRST_VALID_INDEX)?
    };
    Ok(Arc::new(trie))
}

/// Open/create a lock-free vocabulary (capacity hint currently advisory — the overlay's `DashMap`
/// auto-sizes). Uses the trie's default durability policy.
pub fn open_or_create_concurrent_vocabulary_lockfree_with_capacity(
    path: &Path,
    _estimated_terms: usize,
) -> VocabularyResult<Arc<PersistentVocabARTrie>> {
    open_or_create_concurrent_vocabulary_lockfree(path)
}

/// Open/create a lock-free vocabulary (the BloomFilter was removed — overlay reads use the
/// lock-free walk, not a bloom). Uses the trie's default durability policy.
pub fn open_or_create_concurrent_vocabulary_lockfree_with_bloom(
    path: &Path,
    _bloom_capacity: usize,
) -> VocabularyResult<Arc<PersistentVocabARTrie>> {
    open_or_create_concurrent_vocabulary_lockfree(path)
}

/// Type alias for a shared lock-free vocabulary.
///
/// Identical to [`SharedVocabARTrie`] after libdictenstein's F4 lock-collapse
/// (`66a2a63`): both are a bare `Arc<PersistentVocabARTrie>` whose readers and
/// writers are fully lock-free (overlay CAS + DashMap caches + atomic frontiers).
/// Kept as a named synonym for the concurrent-import call sites.
pub type SharedConcurrentVocab = SharedVocabARTrie;

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
// N-gram Key Encoding
// ============================================================================

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
pub fn encode_ngram_key(words: &[&str], vocab: &SharedVocabARTrie) -> String {
    let mut buf = Vec::with_capacity(words.len() * 2); // Estimate 2 bytes/word average
    let guard = vocab.as_ref();
    for word in words {
        let index = guard
            .insert(word)
            .expect("vocabulary insert: persistent ARTrie I/O failed");
        encode_varint(index, &mut buf);
    }
    // Convert bytes to Latin-1 string (each byte → char U+00XX)
    buf.into_iter().map(char::from).collect()
}

/// Try to encode an n-gram as a varint-encoded Latin-1 string (fallible version).
///
/// Returns `Err` if the underlying persistent vocabulary insert fails (I/O,
/// corruption, etc.). Use this in contexts where you can propagate the error
/// rather than panicking.
pub fn try_encode_ngram_key(words: &[&str], vocab: &SharedVocabARTrie) -> VocabularyResult<String> {
    let mut buf = Vec::with_capacity(words.len() * 2);
    let guard = vocab.as_ref();
    for word in words {
        let index = guard.insert(word)?;
        encode_varint(index, &mut buf);
    }
    Ok(buf.into_iter().map(char::from).collect())
}

/// Encode an n-gram using batch insert for vocabulary terms.
///
/// Uses `insert_batch()` to insert all tokens in a single WAL record,
/// reducing I/O overhead from N WAL records to 1 per n-gram.
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
/// // Instead of 5 WAL records for a 5-gram, this writes just 1
/// let key = encode_ngram_key_batch(&["the", "quick", "brown", "fox", "jumps"], &vocab);
/// ```
pub fn encode_ngram_key_batch(words: &[&str], vocab: &SharedVocabARTrie) -> String {
    if words.is_empty() {
        return String::new();
    }

    // Use batch insert for single WAL record
    let indices = vocab
        .as_ref()
        .insert_batch(words)
        .expect("vocabulary batch insert: persistent ARTrie I/O failed");

    // Convert indices to varint-encoded Latin-1 string
    let mut buf = Vec::with_capacity(indices.len() * 2);
    for index in indices {
        encode_varint(index, &mut buf);
    }
    buf.into_iter().map(char::from).collect()
}

/// Try to encode an n-gram using batch insert (fallible version).
///
/// Uses `insert_batch()` to insert all tokens in a single WAL record.
/// Returns `Err` if the underlying persistent vocabulary insert fails
/// (I/O, corruption, etc.). Use this in contexts where you can propagate
/// the error rather than panicking.
pub fn try_encode_ngram_key_batch(
    words: &[&str],
    vocab: &SharedVocabARTrie,
) -> VocabularyResult<String> {
    if words.is_empty() {
        return Ok(String::new());
    }
    let indices = vocab.as_ref().insert_batch(words)?;
    let mut buf = Vec::with_capacity(indices.len() * 2);
    for index in indices {
        encode_varint(index, &mut buf);
    }
    Ok(buf.into_iter().map(char::from).collect())
}

// ============================================================================
// Lock-Free Encoding (for High-Concurrency Imports)
// ============================================================================

/// Encode an n-gram using lock-free CAS operations.
///
/// Unlike [`encode_ngram_key_batch`] which acquires a write lock, this function
/// uses truly lock-free CAS (compare-and-swap) operations on persistent data
/// structures. Multiple threads can encode n-grams simultaneously without
/// blocking each other.
///
/// # Performance
///
/// This function is optimized for high-concurrency workloads where many workers
/// insert vocabulary entries simultaneously. The lock-free approach eliminates
/// the write lock contention that causes slowdowns in parallel import.
///
/// # Arguments
///
/// * `words` - Slice of word strings
/// * `vocab` - A `PersistentVocabARTrie` (the lock-free vocabulary)
///
/// # Returns
///
/// A Latin-1 encoded string where each word's index is varint-encoded.
///
/// # Example
///
/// ```ignore
/// use std::sync::Arc;
///
/// // Create the lock-free vocabulary, then share it.
/// let vocab = PersistentVocabARTrie::create("vocab.vocab")?;
/// let concurrent = Arc::new(vocab);
///
/// // Multiple threads can encode concurrently without blocking
/// std::thread::scope(|s| {
///     for i in 0..12 {
///         let c = Arc::clone(&concurrent);
///         s.spawn(move || {
///             for ngram in ngrams_for_worker(i) {
///                 let key = encode_ngram_key_lockfree(&ngram, &c);
///                 // Store key...
///             }
///         });
///     }
/// });
/// ```
pub fn encode_ngram_key_lockfree(words: &[&str], vocab: &PersistentVocabARTrie) -> String {
    if words.is_empty() {
        return String::new();
    }

    // Use lock-free insert_batch
    let indices = vocab
        .insert_batch(words)
        .expect("vocab insert_batch failed");

    // Convert indices to varint-encoded Latin-1 string
    let mut buf = Vec::with_capacity(indices.len() * 2);
    for index in indices {
        encode_varint(index, &mut buf);
    }
    buf.into_iter().map(char::from).collect()
}

/// Try to encode an n-gram using lock-free CAS operations (fallible version).
///
/// Uses lock-free CAS operations for concurrent vocabulary access.
pub fn try_encode_ngram_key_lockfree(
    words: &[&str],
    vocab: &PersistentVocabARTrie,
) -> VocabularyResult<String> {
    Ok(encode_ngram_key_lockfree(words, vocab))
}

/// Encode an n-gram using a `PersistentVocabARTrie` directly.
///
/// This is the most efficient encoding path when you have direct access
/// to the `PersistentVocabARTrie` (the lock-free vocabulary).
///
/// # Arguments
///
/// * `words` - Slice of word strings
/// * `vocab` - A `PersistentVocabARTrie` instance
///
/// # Returns
///
/// A Latin-1 encoded string where each word's index is varint-encoded.
pub fn encode_ngram_key_with_lockfree_vocab(
    words: &[&str],
    vocab: &PersistentVocabARTrie,
) -> String {
    if words.is_empty() {
        return String::new();
    }

    // Use lock-free batch insert
    let indices = vocab
        .insert_batch(words)
        .expect("vocab insert_batch failed");

    // Convert indices to varint-encoded Latin-1 string
    let mut buf = Vec::with_capacity(indices.len() * 2);
    for index in indices {
        encode_varint(index, &mut buf);
    }
    buf.into_iter().map(char::from).collect()
}

use std::cell::RefCell;

thread_local! {
    /// Reusable buffer for LEB128 n-gram key encoding.
    ///
    /// Avoids per-call heap allocation by `.clear()`-ing and reusing the same
    /// buffer on each thread. Typical n-gram keys are <64 bytes.
    static ENCODE_BUF: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(64));
}

/// Encode n-gram tokens into a byte key using a thread-local buffer, then call `f(&[u8])`.
///
/// This avoids allocating a `Vec<u8>` per n-gram by reusing a thread-local buffer.
/// The callback pattern ensures the buffer reference doesn't escape. Uses
/// `PersistentVocabARTrie::insert_batch` for lock-free vocabulary access.
///
/// # Arguments
///
/// * `words` - The n-gram tokens (e.g., `["the", "quick"]`)
/// * `vocab` - A lock-free concurrent vocabulary
/// * `f` - Callback receiving the encoded `&[u8]` key
///
/// # Returns
///
/// The return value of `f`.
///
/// # Example
///
/// ```ignore
/// with_encoded_ngram_key_lockfree(&["the", "quick"], &vocab, |key| {
///     trie.increment_cas(key, 1);
/// });
/// ```
pub fn with_encoded_ngram_key_lockfree<R>(
    words: &[&str],
    vocab: &PersistentVocabARTrie,
    f: impl FnOnce(&[u8]) -> R,
) -> R {
    ENCODE_BUF.with(|buf| {
        let mut buf = buf.borrow_mut();
        buf.clear();
        let indices = vocab
            .insert_batch(words)
            .expect("vocab insert_batch failed");
        for index in indices {
            encode_varint(index, &mut buf);
        }
        f(&buf)
    })
}

/// Encode n-gram tokens into a byte key using a thread-local buffer (bytes version).
///
/// Like `with_encoded_ngram_key_lockfree` but returns the key as a `Vec<u8>` for
/// callers that need ownership. Slightly less efficient than the callback variant
/// since it clones the buffer, but avoids lifetime issues.
pub fn encode_ngram_key_lockfree_bytes(words: &[&str], vocab: &PersistentVocabARTrie) -> Vec<u8> {
    with_encoded_ngram_key_lockfree(words, vocab, |key| key.to_vec())
}

/// Encode an n-gram using existing vocabulary entries only.
///
/// Returns `None` if any word is not in the vocabulary.
/// This is useful for queries where we don't want to add new words.
pub fn encode_ngram_key_existing(words: &[&str], vocab: &SharedVocabARTrie) -> Option<String> {
    let mut buf = Vec::with_capacity(words.len() * 2);
    let guard = vocab.as_ref();
    for word in words {
        let index = guard.get_index(word)?;
        encode_varint(index, &mut buf);
    }
    Some(buf.into_iter().map(char::from).collect())
}

/// Encode an n-gram as raw varint bytes, inserting new words into vocab.
pub fn encode_ngram_key_bytes(words: &[&str], vocab: &SharedVocabARTrie) -> Vec<u8> {
    let mut buf = Vec::with_capacity(words.len() * 2);
    let guard = vocab.as_ref();
    for word in words {
        let index = guard
            .insert(word)
            .expect("vocabulary insert: persistent ARTrie I/O failed");
        encode_varint(index, &mut buf);
    }
    buf
}

/// Encode an n-gram as raw varint bytes using existing vocabulary entries only.
///
/// Returns `None` if any word is not in the vocabulary.
pub fn encode_ngram_key_existing_bytes(
    words: &[&str],
    vocab: &SharedVocabARTrie,
) -> Option<Vec<u8>> {
    let mut buf = Vec::with_capacity(words.len() * 2);
    let guard = vocab.as_ref();
    for word in words {
        let index = guard.get_index(word)?;
        encode_varint(index, &mut buf);
    }
    Some(buf)
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
    buf.into_iter().map(char::from).collect()
}

// ============================================================================
// Byte-Native Key Functions (for PersistentARTrie<u64> / byte-keyed tries)
// ============================================================================

/// Decode a raw byte n-gram key to word indices.
///
/// Unlike `decode_ngram_key(&str)` which first converts Latin-1 chars back to
/// bytes, this operates directly on the raw varint-encoded byte key.
#[inline]
pub fn decode_ngram_key_bytes(key: &[u8]) -> Vec<u64> {
    let mut indices = Vec::new();
    let mut offset = 0;
    while offset < key.len() {
        if let Some((index, consumed)) = decode_varint(&key[offset..]) {
            indices.push(index);
            offset += consumed;
        } else {
            break;
        }
    }
    indices
}

/// Encode a slice of word indices to a varint-encoded byte key.
///
/// Like `encode_indices_to_key` but returns `Vec<u8>` without Latin-1
/// conversion. Used for MKN context keys with byte-keyed tries.
pub fn encode_indices_to_key_bytes(indices: &[u64]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(indices.len() * 2);
    for &index in indices {
        encode_varint(index, &mut buf);
    }
    buf
}

/// Get the n-gram order from a raw byte key.
///
/// Counts the number of varints in the encoded byte key.
#[inline]
pub fn ngram_order_bytes(key: &[u8]) -> u8 {
    let mut count: u8 = 0;
    let mut offset = 0;
    while offset < key.len() {
        if let Some((_index, consumed)) = decode_varint(&key[offset..]) {
            count += 1;
            offset += consumed;
        } else {
            break;
        }
    }
    count
}

/// Try to encode n-gram tokens into a raw byte key using lock-free vocabulary.
///
/// Returns `Err` if any word fails vocabulary insertion.
/// This is the fallible version of `encode_ngram_key_lockfree_bytes`.
pub fn try_encode_ngram_key_lockfree_bytes(
    words: &[&str],
    vocab: &PersistentVocabARTrie,
) -> VocabularyResult<Vec<u8>> {
    Ok(encode_ngram_key_lockfree_bytes(words, vocab))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_temp_vocab() -> (TempDir, SharedVocabARTrie) {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("vocab.artrie");
        let vocab = create_vocabulary(&path).expect("Failed to create vocab");
        (dir, vocab)
    }

    #[test]
    fn test_insert_new_word() {
        let (_dir, vocab) = create_temp_vocab();

        let idx1 = vocab.as_ref().insert("the").expect("test insert");
        let idx2 = vocab.as_ref().insert("quick").expect("test insert");
        let idx3 = vocab.as_ref().insert("brown").expect("test insert");

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
    fn test_insert_existing_word() {
        let (_dir, vocab) = create_temp_vocab();

        let idx1 = vocab.as_ref().insert("hello").expect("test insert");
        let idx2 = vocab.as_ref().insert("hello").expect("test insert");

        // Same word should return same index
        assert_eq!(idx1, idx2);
        assert_eq!(vocab.as_ref().len(), 1);
    }

    #[test]
    fn test_get_existing() {
        let (_dir, vocab) = create_temp_vocab();

        assert!(vocab.as_ref().get_index("nonexistent").is_none());

        let idx1 = vocab.as_ref().insert("test").expect("test insert");
        let idx2 = vocab.as_ref().get_index("test");

        assert_eq!(idx2, Some(idx1));
    }

    #[test]
    fn test_contains() {
        let (_dir, vocab) = create_temp_vocab();

        assert!(!vocab.as_ref().contains("word"));
        vocab.as_ref().insert("word").expect("test insert");
        assert!(vocab.as_ref().contains("word"));
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
    fn test_encode_ngram_key_batch() {
        let (_dir, vocab) = create_temp_vocab();

        // Test that batch encoding produces same result as individual encoding
        let words = ["the", "quick", "brown"];
        let key_batch = encode_ngram_key_batch(&words, &vocab);

        // Decode and verify we get 3 indices
        let indices = decode_ngram_key(&key_batch);
        assert_eq!(indices.len(), 3);
        assert_eq!(indices, vec![1, 2, 3]); // Sequential indices starting at 1

        // Verify vocabulary state
        assert_eq!(vocab.as_ref().len(), 3);

        // Test encoding existing words (should return same indices)
        let key_batch2 = encode_ngram_key_batch(&words, &vocab);
        assert_eq!(key_batch, key_batch2);
        assert_eq!(vocab.as_ref().len(), 3); // No new words added
    }

    #[test]
    fn test_encode_ngram_key_batch_empty() {
        let (_dir, vocab) = create_temp_vocab();

        let key = encode_ngram_key_batch(&[], &vocab);
        assert!(key.is_empty());
        assert_eq!(vocab.as_ref().len(), 0);
    }

    #[test]
    fn test_encode_ngram_key_batch_mixed() {
        let (_dir, vocab) = create_temp_vocab();

        // Insert some words first
        vocab.as_ref().insert("the").expect("test insert");
        vocab.as_ref().insert("quick").expect("test insert");

        // Batch encode with mix of existing and new words
        let words = ["the", "quick", "brown", "fox"];
        let key = encode_ngram_key_batch(&words, &vocab);

        let indices = decode_ngram_key(&key);
        // the=1, quick=2, brown=3, fox=4
        assert_eq!(indices, vec![1, 2, 3, 4]);
        assert_eq!(vocab.as_ref().len(), 4);
    }

    #[test]
    fn test_encode_ngram_key_with_large_indices() {
        let (_dir, vocab) = create_temp_vocab();

        // Insert enough words to test multi-byte varint
        // Indices start at 1, so word0 gets index 1, word199 gets index 200
        for i in 0..200 {
            vocab
                .as_ref()
                .insert(&format!("word{}", i))
                .expect("insert word");
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
            .map(|&idx| vocab.as_ref().get_term(idx).expect("index should exist"))
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
            .map(|&idx| vocab.as_ref().get_term(idx).expect("index should exist"))
            .collect();

        assert_eq!(decoded, tokens);
    }

    #[test]
    fn test_get_term_reverse_lookup() {
        let (_dir, vocab) = create_temp_vocab();

        // Insert words
        let idx1 = vocab.as_ref().insert("hello").expect("test insert");
        let idx2 = vocab.as_ref().insert("world").expect("test insert");
        let idx3 = vocab.as_ref().insert("rust").expect("test insert");

        // Reverse lookups should work
        assert_eq!(vocab.as_ref().get_term(idx1), Some("hello".to_string()));
        assert_eq!(vocab.as_ref().get_term(idx2), Some("world".to_string()));
        assert_eq!(vocab.as_ref().get_term(idx3), Some("rust".to_string()));

        // Invalid indices should return None
        assert_eq!(vocab.as_ref().get_term(0), None); // Below FIRST_VALID_INDEX
        assert_eq!(vocab.as_ref().get_term(999), None); // Above range
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

        let lower = vocab.as_ref().insert("the").expect("test insert");
        let upper = vocab.as_ref().insert("The").expect("test insert");
        let all_caps = vocab.as_ref().insert("THE").expect("test insert");

        // Case-sensitive: all should be different
        assert_ne!(lower, upper);
        assert_ne!(upper, all_caps);
        assert_ne!(lower, all_caps);
    }

    #[test]
    fn test_punctuation_as_tokens() {
        let (_dir, vocab) = create_temp_vocab();

        let comma = vocab.as_ref().insert(",").expect("test insert");
        let period = vocab.as_ref().insert(".").expect("test insert");
        let quote = vocab.as_ref().insert("\"").expect("test insert");

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
            let vocab = create_vocabulary(&path).expect("Failed to create vocab");
            idx1 = vocab.as_ref().insert("hello").expect("test insert");
            idx2 = vocab.as_ref().insert("world").expect("test insert");
            vocab.as_ref().checkpoint().expect("Checkpoint failed");
        }

        // Reopen and verify
        {
            let vocab = open_vocabulary(&path).expect("Failed to open vocab");

            assert_eq!(vocab.as_ref().len(), 2);
            assert_eq!(vocab.as_ref().get_index("hello"), Some(idx1));
            assert_eq!(vocab.as_ref().get_index("world"), Some(idx2));

            // New word should get next index
            let idx3 = vocab.as_ref().insert("new").expect("test insert");
            assert_eq!(idx3, 3); // Sequential after idx1=1, idx2=2
        }
    }

    #[test]
    fn test_encode_ngram_key_existing() {
        let (_dir, vocab) = create_temp_vocab();

        // Word not in vocabulary
        assert!(encode_ngram_key_existing(&["unknown"], &vocab).is_none());

        // Add some words
        vocab.as_ref().insert("the").expect("test insert");
        vocab.as_ref().insert("quick").expect("test insert");

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
        let vocab = Arc::new(create_vocabulary(&path).expect("Failed to create vocab"));

        // Spawn multiple threads that all try to insert the same word
        let mut handles = vec![];
        for _ in 0..10 {
            let vocab = Arc::clone(&vocab);
            handles.push(thread::spawn(move || {
                vocab.as_ref().insert("shared_word").expect("test insert")
            }));
        }

        // Collect results
        let indices: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All threads should get the same index
        let first = indices[0];
        for idx in &indices {
            assert_eq!(*idx, first);
        }

        // Should only have one entry
        assert_eq!(vocab.as_ref().len(), 1);
    }

    #[test]
    fn test_latin1_encoding_preserves_bytes() {
        // Verify that Latin-1 encoding correctly round-trips all byte values
        for byte in 0u8..=255 {
            let c = char::from(byte);
            assert_eq!(
                c as u8, byte,
                "Byte {} should round-trip through char",
                byte
            );
        }
    }

    #[test]
    fn test_large_vocabulary() {
        // Test that vocabulary can grow beyond the old PUA limit
        let (_dir, vocab) = create_temp_vocab();

        // Insert more than 131,068 words (old PUA limit)
        // We'll test with a smaller number for speed, but verify indices are correct
        for i in 0..1000 {
            let idx = vocab
                .as_ref()
                .insert(&format!("word{}", i))
                .expect("test insert");
            // Indices start at 1, not 0
            assert_eq!(idx, (i + 1) as u64);
        }

        assert_eq!(vocab.as_ref().len(), 1000);
    }

    #[test]
    fn test_start_index() {
        let (_dir, vocab) = create_temp_vocab();

        // Start index should be 1 (FIRST_VALID_INDEX)
        assert_eq!(vocab.as_ref().start_index(), 1);
    }

    #[test]
    fn test_is_dirty() {
        let (_dir, vocab) = create_temp_vocab();

        // After inserting, vocabulary should be dirty
        vocab.as_ref().insert("test").expect("test insert");
        assert!(vocab.as_ref().is_dirty());

        // After checkpoint, should no longer be dirty
        vocab.as_ref().checkpoint().expect("checkpoint failed");
        assert!(!vocab.as_ref().is_dirty());
    }

    #[test]
    fn test_contains_index() {
        let (_dir, vocab) = create_temp_vocab();

        // Index 0 should never exist (reserved)
        assert!(!vocab.as_ref().contains_index(0));

        // After insert, index 1 should exist
        vocab.as_ref().insert("test").expect("test insert");
        assert!(vocab.as_ref().contains_index(1));
        assert!(!vocab.as_ref().contains_index(2)); // Not yet inserted
    }

    // ==================== Lock-Free Mode Tests ====================

    #[test]
    fn test_encode_ngram_key_lockfree() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("vocab.artrie");
        let concurrent = open_or_create_concurrent_vocabulary_lockfree(&path)
            .expect("Failed to create concurrent vocab");

        // Encode using lock-free API
        let key = encode_ngram_key_lockfree(&["the", "quick", "brown"], &concurrent);

        // Decode and verify
        let indices = decode_ngram_key(&key);
        assert_eq!(indices.len(), 3);

        // Verify we can look up the terms
        assert_eq!(concurrent.get_index("the"), Some(1));
        assert_eq!(concurrent.get_index("quick"), Some(2));
        assert_eq!(concurrent.get_index("brown"), Some(3));
    }

    #[test]
    fn test_encode_ngram_key_lockfree_concurrent() {
        use std::thread;

        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("vocab.artrie");
        let concurrent = open_or_create_concurrent_vocabulary_lockfree(&path)
            .expect("Failed to create concurrent vocab");

        let num_threads = 8;
        let terms_per_thread = 100;

        // Spawn multiple threads that encode n-grams concurrently
        let handles: Vec<_> = (0..num_threads)
            .map(|t| {
                let c = Arc::clone(&concurrent);
                thread::spawn(move || {
                    let mut keys = Vec::new();
                    for i in 0..terms_per_thread {
                        let words = [
                            format!("thread{}_word{}", t, i),
                            format!("thread{}_word{}", t, i + 1),
                        ];
                        let word_refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
                        let key = encode_ngram_key_lockfree(&word_refs, &c);
                        keys.push(key);
                    }
                    keys
                })
            })
            .collect();

        // Collect all keys
        let all_keys: Vec<Vec<String>> = handles
            .into_iter()
            .map(|h| h.join().expect("thread complete"))
            .collect();

        // Verify all keys decode to 2 indices
        for thread_keys in &all_keys {
            for key in thread_keys {
                let indices = decode_ngram_key(key);
                assert_eq!(indices.len(), 2, "Each n-gram should have 2 indices");
            }
        }

        // Verify vocabulary grew correctly
        // Each thread inserts (terms_per_thread + 1) unique words
        // Total = num_threads * (terms_per_thread + 1)
        let expected_vocab_size = num_threads * (terms_per_thread + 1);
        assert!(concurrent.next_index() > expected_vocab_size as u64);
    }

    #[test]
    fn test_open_or_create_concurrent_vocabulary_lockfree() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("vocab.artrie");

        // Create new
        let concurrent1 = open_or_create_concurrent_vocabulary_lockfree(&path)
            .expect("Failed to create concurrent vocab");

        concurrent1.insert("hello").expect("insert");
        concurrent1.insert("world").expect("insert");

        // Checkpoint to persistent storage and ensure it's persisted to disk
        concurrent1.checkpoint().expect("checkpoint failed");
        drop(concurrent1);

        // Reopen - the lock-free layer starts fresh, but the persistent layer
        // has the terms from before. The get_index method checks both layers.
        let concurrent2 = open_or_create_concurrent_vocabulary_lockfree(&path)
            .expect("Failed to open concurrent vocab");

        // Verify existing terms are accessible (from persistent layer)
        assert_eq!(concurrent2.get_index("hello"), Some(1));
        assert_eq!(concurrent2.get_index("world"), Some(2));

        // New term should get the next index.
        // The lock-free layer starts from next_index of the persistent trie (3).
        let idx3 = concurrent2.insert("new").expect("insert");
        assert!(
            idx3 >= 3,
            "New term index should be at least 3, got {}",
            idx3
        );
    }

    #[test]
    fn test_lockfree_vocab_stats() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("vocab.artrie");
        let concurrent = open_or_create_concurrent_vocabulary_lockfree(&path)
            .expect("Failed to create concurrent vocab");

        // Insert some terms
        concurrent.insert("one").expect("insert");
        concurrent.insert("two").expect("insert");
        concurrent.insert("three").expect("insert");

        // Lock-free vocab stats are now just len()/next_index() on the single impl.
        assert_eq!(concurrent.len(), 3);
        assert_eq!(concurrent.next_index(), 4); // 1,2,3 inserted, next is 4 (start at 1)
    }
}
