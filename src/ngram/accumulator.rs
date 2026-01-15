//! Persistent N-gram accumulator using liblevenshtein's DiskBackedCharTrieInner.
//!
//! This module provides crash-safe N-gram counting for training with:
//! - WAL-based durability (automatic crash recovery)
//! - Atomic increment operations
//! - Checkpointing for training resume
//!
//! # Architecture
//!
//! The accumulator uses a three-tier persistence strategy:
//!
//! 1. **Hot State (DiskBackedCharTrieInner)**: WAL-based crash recovery during training
//! 2. **Checkpoints (metadata)**: Periodic snapshots of training progress
//! 3. **Final Model (PathMap)**: Convert to optimized inference structure
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::ngram::NgramAccumulator;
//! use std::path::Path;
//!
//! // Create new accumulator
//! let mut acc = NgramAccumulator::create(Path::new("ngrams.artrie"))?;
//!
//! // Atomic increment (crash-safe)
//! acc.increment("the|quick|brown")?;
//! acc.increment("quick|brown|fox")?;
//!
//! // Ensure durability
//! acc.sync()?;
//!
//! // Periodic checkpoint (truncates WAL)
//! acc.checkpoint()?;
//!
//! // Export for final model
//! for (ngram, count) in acc.iter_with_counts() {
//!     println!("{}: {}", ngram, count);
//! }
//! ```

use liblevenshtein::dictionary::persistent_artrie_char::DiskBackedCharTrieInner;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Error type for NgramAccumulator operations.
#[derive(Error, Debug)]
pub enum AccumulatorError {
    /// I/O error from persistent storage.
    #[error("Storage I/O error: {0}")]
    Storage(#[from] std::io::Error),

    /// Liblevenshtein dictionary error.
    #[error("Dictionary error: {0}")]
    Dictionary(String),

    /// Checkpoint error.
    #[error("Checkpoint error: {0}")]
    Checkpoint(String),
}

/// Result type for accumulator operations.
pub type AccumulatorResult<T> = std::result::Result<T, AccumulatorError>;

/// N-gram count accumulator using persistent ART with WAL.
///
/// Provides atomic increment for frequency counting with crash recovery.
/// The underlying DiskBackedCharTrieInner uses Write-Ahead Logging (WAL) for durability
/// and automatic ARIES-style recovery on open.
///
/// This uses the character-based trie variant for proper Unicode support in
/// multilingual n-grams.
///
/// # Thread Safety
///
/// The `increment()` method takes `&mut self`. For concurrent access, wrap in
/// a synchronization primitive like `parking_lot::RwLock`.
pub struct NgramAccumulator {
    /// Persistent ART-based trie with WAL (stores i64 for increment compatibility)
    /// Uses character-based trie for proper Unicode n-gram support.
    trie: DiskBackedCharTrieInner<i64>,

    /// File path for the trie
    path: PathBuf,

    /// Count of unique n-grams (cached)
    unique_count: std::sync::atomic::AtomicUsize,
}

impl NgramAccumulator {
    /// Create a new persistent accumulator.
    ///
    /// Creates a new DiskBackedCharTrieInner at the specified path. If the path
    /// already exists, it will be overwritten.
    ///
    /// # Arguments
    ///
    /// * `path` - Path for the persistent storage files
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created or initialized.
    pub fn create(path: &Path) -> AccumulatorResult<Self> {
        let trie = DiskBackedCharTrieInner::create(path).map_err(|e| {
            AccumulatorError::Dictionary(format!("Failed to create persistent trie: {}", e))
        })?;

        Ok(Self {
            trie,
            path: path.to_path_buf(),
            unique_count: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Open an existing accumulator with automatic crash recovery.
    ///
    /// Opens an existing DiskBackedCharTrieInner with automatic corruption detection
    /// and recovery from WAL archive segments. If corruption is detected,
    /// the trie is rebuilt from archived WAL segments.
    ///
    /// # Recovery Process
    ///
    /// 1. **Check existence** - If file doesn't exist, returns an error
    /// 2. **Detect corruption** - Validates file header and checksums
    /// 3. **If corrupted** - Rebuilds from WAL archive segments
    /// 4. **Open normally** - If no corruption detected
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the existing persistent storage files
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be opened
    /// - Corruption is detected but no WAL archive segments exist
    /// - Recovery fails
    pub fn open(path: &Path) -> AccumulatorResult<Self> {
        let (trie, report) = DiskBackedCharTrieInner::open_with_recovery(path).map_err(|e| {
            AccumulatorError::Dictionary(format!("Failed to open/recover persistent trie: {}", e))
        })?;

        // Log recovery if it occurred
        if report.mode.recovered() {
            log::info!(
                "Recovered accumulator from crash: mode={:?}, {} records replayed, {} terms recovered",
                report.mode,
                report.records_replayed,
                report.terms_recovered
            );
        }

        // Get current count (len is a direct field access)
        let count = trie.len;

        Ok(Self {
            trie,
            path: path.to_path_buf(),
            unique_count: std::sync::atomic::AtomicUsize::new(count),
        })
    }

    /// Get the storage path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Atomically increment count for an n-gram (WAL-logged, crash-safe).
    ///
    /// Uses the atomic `increment()` method which is a single operation:
    /// - Initializes to 0 if the n-gram doesn't exist
    /// - Adds the delta atomically
    /// - Logs the operation to WAL for crash recovery
    ///
    /// # Arguments
    ///
    /// * `ngram` - The n-gram key (e.g., "the|quick|brown" pipe-separated)
    ///
    /// # Returns
    ///
    /// The new count value after incrementing.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation cannot be persisted.
    pub fn increment(&mut self, ngram: &str) -> AccumulatorResult<i64> {
        let result = self.trie.increment(ngram, 1).map_err(|e| {
            AccumulatorError::Dictionary(format!("Failed to increment n-gram: {}", e))
        })?;

        // If this is a new entry (count went from 0 to 1), update unique count
        if result == 1 {
            self.unique_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        Ok(result)
    }

    /// Increment by a specific delta.
    ///
    /// Useful for merging counts from batch processing.
    pub fn increment_by(&mut self, ngram: &str, delta: i64) -> AccumulatorResult<i64> {
        self.trie.increment(ngram, delta).map_err(|e| {
            AccumulatorError::Dictionary(format!("Failed to increment n-gram: {}", e))
        })
    }

    /// Get the current count for an n-gram.
    ///
    /// Returns `None` if the n-gram has never been seen.
    pub fn get(&self, ngram: &str) -> Option<i64> {
        self.trie.get(ngram).copied()
    }

    /// Check if an n-gram exists in the accumulator.
    pub fn contains(&self, ngram: &str) -> bool {
        self.trie.contains(ngram)
    }

    /// Flush WAL to ensure durability.
    ///
    /// Ensures all pending operations are written to the WAL on disk.
    /// Call this periodically during training to reduce data loss on crash.
    ///
    /// # Errors
    ///
    /// Returns an error if the sync operation fails.
    pub fn sync(&mut self) -> AccumulatorResult<()> {
        self.trie.sync().map_err(|e| {
            AccumulatorError::Dictionary(format!("Failed to sync WAL: {}", e))
        })
    }

    /// Checkpoint to disk and truncate WAL.
    ///
    /// This operation:
    /// 1. Persists all in-memory data to the main storage file
    /// 2. Truncates the WAL (safe because all data is now in main file)
    ///
    /// After a checkpoint, recovery only needs to replay WAL records with
    /// LSN > checkpoint_lsn.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint operation fails.
    pub fn checkpoint(&mut self) -> AccumulatorResult<()> {
        self.trie.checkpoint().map_err(|e| {
            AccumulatorError::Dictionary(format!("Failed to checkpoint: {}", e))
        })
    }

    /// Iterate over all (ngram, count) pairs.
    ///
    /// Used for exporting to the final model format.
    /// Returns an empty iterator if the trie is empty or an error occurs.
    pub fn iter_with_counts(&self) -> impl Iterator<Item = (String, i64)> {
        self.trie
            .iter_prefix_with_values("")
            .ok()
            .flatten()
            .unwrap_or_default()
            .into_iter()
    }

    /// Iterate over all n-gram keys.
    ///
    /// Returns an empty iterator if the trie is empty or an error occurs.
    pub fn iter(&self) -> impl Iterator<Item = String> {
        self.trie
            .iter_prefix("")
            .ok()
            .flatten()
            .unwrap_or_default()
            .into_iter()
    }

    /// Number of unique n-grams.
    ///
    /// Note: This is a cached approximate count. For the exact count,
    /// iterate and count.
    pub fn len(&self) -> usize {
        self.unique_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Check if the accumulator is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the exact count of unique n-grams.
    ///
    /// This directly accesses the trie's length field.
    pub fn exact_len(&self) -> usize {
        self.trie.len
    }
}

/// N-gram key format utilities.
pub mod key_format {
    /// Separator used between tokens in n-gram keys.
    pub const SEPARATOR: char = '|';

    /// Build an n-gram key from tokens.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let key = build_key(&["the", "quick", "brown"]);
    /// assert_eq!(key, "the|quick|brown");
    /// ```
    pub fn build_key(tokens: &[&str]) -> String {
        tokens.join(&SEPARATOR.to_string())
    }

    /// Build an n-gram key from owned tokens.
    pub fn build_key_owned(tokens: &[String]) -> String {
        tokens.join(&SEPARATOR.to_string())
    }

    /// Parse an n-gram key into tokens.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let tokens = parse_key("the|quick|brown");
    /// assert_eq!(tokens, vec!["the", "quick", "brown"]);
    /// ```
    pub fn parse_key(key: &str) -> Vec<&str> {
        key.split(SEPARATOR).collect()
    }

    /// Get the order (n) of an n-gram from its key.
    pub fn order(key: &str) -> usize {
        key.split(SEPARATOR).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_create_and_increment() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("test.artrie");

        let mut acc = NgramAccumulator::create(&path).expect("Failed to create accumulator");

        // First increment creates entry with count 1
        let count = acc.increment("the|quick").expect("Failed to increment");
        assert_eq!(count, 1);

        // Second increment should return 2
        let count = acc.increment("the|quick").expect("Failed to increment");
        assert_eq!(count, 2);

        // Different n-gram
        let count = acc.increment("quick|brown").expect("Failed to increment");
        assert_eq!(count, 1);

        assert_eq!(acc.len(), 2);
    }

    #[test]
    fn test_persistence_and_recovery() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("test.artrie");

        // Create and populate
        {
            let mut acc = NgramAccumulator::create(&path).expect("Failed to create accumulator");
            acc.increment("the|quick").expect("Failed to increment");
            acc.increment("the|quick").expect("Failed to increment");
            acc.increment("quick|brown").expect("Failed to increment");
            acc.sync().expect("Failed to sync");
        }

        // Reopen and verify
        {
            let acc = NgramAccumulator::open(&path).expect("Failed to open accumulator");
            assert_eq!(acc.get("the|quick"), Some(2));
            assert_eq!(acc.get("quick|brown"), Some(1));
            assert_eq!(acc.get("nonexistent"), None);
        }
    }

    #[test]
    fn test_iteration() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("test.artrie");

        let mut acc = NgramAccumulator::create(&path).expect("Failed to create accumulator");
        acc.increment("a|b").expect("Failed to increment");
        acc.increment("a|b").expect("Failed to increment");
        acc.increment("c|d").expect("Failed to increment");

        let mut entries: Vec<_> = acc.iter_with_counts().collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0], ("a|b".to_string(), 2));
        assert_eq!(entries[1], ("c|d".to_string(), 1));
    }

    #[test]
    fn test_key_format() {
        use key_format::*;

        assert_eq!(build_key(&["the", "quick", "brown"]), "the|quick|brown");
        assert_eq!(parse_key("the|quick|brown"), vec!["the", "quick", "brown"]);
        assert_eq!(order("the|quick|brown"), 3);
        assert_eq!(order("unigram"), 1);
    }
}
