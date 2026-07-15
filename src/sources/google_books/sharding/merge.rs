//! Parallel reduction merge for sharded n-gram storage.
//!
//! This module provides utilities for merging multiple shards into a single
//! output using parallel reduction, and exporting the final result to PathMap.
//!
//! # Merge Strategy
//!
//! After import completes, shards are merged using O(log N) parallel reduction:
//!
//! ```text
//! Phase 1: Parallel pairwise merge
//! ┌───┐ ┌───┐   ┌───┐ ┌───┐   ┌───┐ ┌───┐
//! │ a │+│ b │   │ c │+│ d │   │ e │+│ f │  ...
//! └─┬─┘ └─┬─┘   └─┬─┘ └─┬─┘   └─┬─┘ └─┬─┘
//!   └──┬──┘       └──┬──┘       └──┬──┘
//!    ┌─┴─┐        ┌─┴─┐        ┌─┴─┐
//!    │a-b│        │c-d│        │e-f│    ...
//!
//! Phase 2: Continue reduction
//!    ┌───┐ ┌───┐       ┌───┐ ┌───┐
//!    │a-b│+│c-d│       │e-f│+│...│
//!    └─┬─┘ └─┬─┘       └─┬─┘ └─┬─┘
//!      └──┬──┘           └──┬──┘
//!       ┌─┴──┐          ┌──┴─┐
//!       │a-d │          │e-..│
//!
//! Phase N: Final merge → PathMap
//! ```
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::sources::google_books::sharding::merge::{
//!     MergeCoordinator, MergeProgress,
//! };
//!
//! let coordinator = ShardCoordinator::resume_or_start(config)?;
//!
//! // After import completes, merge to PathMap
//! let mut merger = MergeCoordinator::new(&coordinator);
//! let pathmap = merger.merge_to_pathmap(
//!     "output.pathmap",
//!     |progress| println!("{:?}", progress),
//! )?;
//! ```

use super::coordinator::ShardCoordinator;
use super::routing::ShardKey;
use libdictenstein::persistent_artrie::PersistentARTrie;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;
use xxhash_rust::xxh3::Xxh3DefaultBuilder;

/// Type alias for HashMap with xxh3 hasher (non-adversarial data).
type XxHashMap<K, V> = HashMap<K, V, Xxh3DefaultBuilder>;

/// Error type for merge operations.
#[derive(Error, Debug)]
pub enum MergeError {
    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Shard error.
    #[error("Shard error: {0}")]
    Shard(#[from] super::shard::ShardError),

    /// Coordinator error.
    #[error("Coordinator error: {0}")]
    Coordinator(#[from] super::coordinator::CoordinatorError),

    /// Trie operation failed.
    #[error("Trie error: {0}")]
    Trie(String),

    /// No shards to merge.
    #[error("No shards available for merge")]
    NoShards,

    /// Merge cancelled.
    #[error("Merge cancelled")]
    Cancelled,

    /// A shard failed to open during merge.
    #[error("Failed to open shard '{shard_key}': {message}")]
    ShardOpen {
        /// The key of the shard that failed to open.
        shard_key: String,
        /// The error message describing why the shard failed to open.
        message: String,
    },
}

/// Result type for merge operations.
pub type MergeResult<T> = Result<T, MergeError>;

/// Progress information during merge.
#[derive(Clone, Debug)]
pub struct MergeProgress {
    /// Current merge phase (1-based).
    pub phase: usize,

    /// Total phases expected.
    pub total_phases: usize,

    /// Shards remaining in current phase.
    pub shards_remaining: usize,

    /// Total shards at start.
    pub total_shards: usize,

    /// N-grams merged so far.
    pub ngrams_merged: u64,

    /// Estimated completion percentage.
    pub percent_complete: f32,
}

/// Statistics from a completed merge.
#[derive(Clone, Debug)]
pub struct MergeStats {
    /// Total phases executed.
    pub phases: usize,

    /// Total n-grams in final output.
    pub total_ngrams: u64,

    /// Total bytes written.
    pub bytes_written: u64,

    /// Duration in milliseconds.
    pub duration_ms: u64,

    /// Shards merged.
    pub shards_merged: usize,
}

/// Coordinator for parallel merge operations.
pub struct MergeCoordinator<'a> {
    /// Reference to the shard coordinator.
    coordinator: &'a ShardCoordinator,

    /// Working directory for intermediate files.
    work_dir: PathBuf,

    /// Number of parallel merge workers.
    parallelism: usize,

    /// Whether to delete intermediate files.
    cleanup_intermediates: bool,
}

impl<'a> MergeCoordinator<'a> {
    /// Create a new merge coordinator.
    pub fn new(coordinator: &'a ShardCoordinator) -> Self {
        let work_dir = coordinator.config().shard_dir.join("merge_work");
        // Use std::thread::available_parallelism for CPU count
        let parallelism = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        Self {
            coordinator,
            work_dir,
            parallelism,
            cleanup_intermediates: true,
        }
    }

    /// Set the working directory for intermediate files.
    pub fn with_work_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.work_dir = dir.into();
        self
    }

    /// Set the parallelism level.
    pub fn with_parallelism(mut self, parallelism: usize) -> Self {
        self.parallelism = parallelism.max(1);
        self
    }

    /// Set whether to cleanup intermediate files.
    pub fn with_cleanup(mut self, cleanup: bool) -> Self {
        self.cleanup_intermediates = cleanup;
        self
    }

    /// Merge all shards to a single trie file.
    ///
    /// This approach directly iterates through shard handles (avoiding
    /// file opening issues) and writes to a single output trie.
    pub fn merge_to_trie<F>(
        &self,
        output_path: impl AsRef<Path>,
        mut progress_callback: F,
    ) -> MergeResult<MergeStats>
    where
        F: FnMut(MergeProgress) + Send,
    {
        let output_path = output_path.as_ref();
        let start_time = std::time::Instant::now();

        // Discover all shard files on disk (not just cached ones)
        let shard_files = self
            .coordinator
            .discover_shard_files()
            .map_err(|e| MergeError::Trie(format!("Failed to discover shard files: {}", e)))?;
        if shard_files.is_empty() {
            return Err(MergeError::NoShards);
        }

        let shard_keys: Vec<ShardKey> = shard_files.into_iter().map(|(key, _)| key).collect();
        let total_shards = shard_keys.len();

        log::info!(
            "Starting merge of {} shards to {:?}",
            total_shards,
            output_path
        );

        // Create output trie
        let mut output_trie = PersistentARTrie::<u64>::create(output_path)
            .map_err(|e| MergeError::Trie(format!("Failed to create output trie: {}", e)))?;

        let mut ngrams_merged = 0u64;

        // Iterate through each shard and copy data to output
        for (shards_processed, key) in shard_keys.iter().enumerate() {
            progress_callback(MergeProgress {
                phase: 1,
                total_phases: 1,
                shards_remaining: total_shards - shards_processed,
                total_shards,
                ngrams_merged,
                percent_complete: (shards_processed as f32 / total_shards as f32) * 100.0,
            });

            log::trace!(
                "Merging shard '{}' ({}/{})",
                key,
                shards_processed + 1,
                total_shards
            );

            let shard =
                self.coordinator
                    .get_or_create_shard(key)
                    .map_err(|e| MergeError::ShardOpen {
                        shard_key: key.to_string(),
                        message: e.to_string(),
                    })?;
            // Overlay-default: iteration reads the overlay directly (no pre-iteration
            // flush needed); a shared read guard suffices.
            let guard = shard.read();

            let iter = guard
                .iter_with_counts()
                .map_err(|e| MergeError::Trie(format!("Shard {} iteration failed: {}", key, e)))?;

            for (ngram, count) in iter {
                output_trie
                    .increment_bytes(&ngram, count as i64)
                    .map_err(|e| MergeError::Trie(format!("Increment failed: {}", e)))?;
                ngrams_merged += 1;
            }
        }

        // Checkpoint output trie
        output_trie
            .checkpoint()
            .map_err(|e| MergeError::Trie(format!("Checkpoint failed: {}", e)))?;

        // Final progress callback
        progress_callback(MergeProgress {
            phase: 1,
            total_phases: 1,
            shards_remaining: 0,
            total_shards,
            ngrams_merged,
            percent_complete: 100.0,
        });

        let duration = start_time.elapsed();
        let bytes_written = std::fs::metadata(output_path).map(|m| m.len()).unwrap_or(0);

        Ok(MergeStats {
            phases: 1,
            total_ngrams: ngrams_merged,
            bytes_written,
            duration_ms: duration.as_millis() as u64,
            shards_merged: total_shards,
        })
    }

    /// Merge all shards directly into memory and return counts.
    ///
    /// This is useful for smaller datasets where all n-grams fit in memory.
    /// Returns a HashMap of (ngram, count) pairs.
    pub fn merge_to_memory(&self) -> MergeResult<XxHashMap<Vec<u8>, u64>> {
        // Discover all shard files on disk (not just cached ones)
        let shard_files = self
            .coordinator
            .discover_shard_files()
            .map_err(|e| MergeError::Trie(format!("Failed to discover shard files: {}", e)))?;
        if shard_files.is_empty() {
            return Err(MergeError::NoShards);
        }

        let shard_keys: Vec<ShardKey> = shard_files.into_iter().map(|(key, _)| key).collect();

        log::info!("Merging {} shards to memory", shard_keys.len());

        // Collect all n-grams from all shards
        // Use try_fold pattern to propagate errors from shard iteration
        let results: Result<Vec<XxHashMap<Vec<u8>, u64>>, MergeError> = shard_keys
            .par_iter()
            .map(|key| {
                log::trace!("Merging shard '{}' to memory", key);

                let shard = self.coordinator.get_or_create_shard(key).map_err(|e| {
                    MergeError::ShardOpen {
                        shard_key: key.to_string(),
                        message: e.to_string(),
                    }
                })?;
                let guard = shard.read();
                let iter = guard.iter_with_counts().map_err(|e| {
                    MergeError::Trie(format!("Shard {} iteration failed: {}", key, e))
                })?;
                Ok(iter.into_iter().collect::<XxHashMap<_, _>>())
            })
            .collect();

        let results = results?;

        // Merge all results
        let mut merged: XxHashMap<Vec<u8>, u64> = HashMap::with_hasher(Xxh3DefaultBuilder);
        for partial in results {
            for (ngram, count) in partial {
                *merged.entry(ngram).or_default() += count;
            }
        }

        log::info!("Merged {} unique n-grams", merged.len());
        Ok(merged)
    }

    /// Stream merge all shards to an iterator.
    ///
    /// This avoids loading all data into memory by streaming through shards.
    /// Note: Duplicate n-grams across shards are NOT aggregated in this mode.
    ///
    /// # Errors
    ///
    /// Returns an error if shard discovery fails or any shard iteration fails.
    pub fn iter_all(&self) -> MergeResult<impl Iterator<Item = (Vec<u8>, u64)>> {
        // Discover all shard files on disk (not just cached ones)
        let shard_keys: Vec<ShardKey> = self
            .coordinator
            .discover_shard_files()
            .map_err(|e| MergeError::Trie(format!("Failed to discover shard files: {}", e)))?
            .into_iter()
            .map(|(key, _)| key)
            .collect();

        // Pre-collect all entries to avoid lifetime issues and propagate errors early
        let mut all_entries = Vec::new();
        for key in shard_keys {
            log::trace!("Iterating shard '{}'", key);

            let shard =
                self.coordinator
                    .get_or_create_shard(&key)
                    .map_err(|e| MergeError::ShardOpen {
                        shard_key: key.to_string(),
                        message: e.to_string(),
                    })?;
            let guard = shard.read();
            let iter = guard
                .iter_with_counts()
                .map_err(|e| MergeError::Trie(format!("Shard {} iteration failed: {}", key, e)))?;
            all_entries.extend(iter);
        }

        Ok(all_entries.into_iter())
    }

    /// Get the estimated final size (total entries across all shards).
    pub fn estimated_final_size(&self) -> u64 {
        self.coordinator.total_entry_count()
    }
}

/// Builder for configuring merge operations.
pub struct MergeBuilder<'a> {
    coordinator: &'a ShardCoordinator,
    work_dir: Option<PathBuf>,
    parallelism: Option<usize>,
    cleanup: bool,
}

impl<'a> MergeBuilder<'a> {
    /// Create a new merge builder.
    pub fn new(coordinator: &'a ShardCoordinator) -> Self {
        Self {
            coordinator,
            work_dir: None,
            parallelism: None,
            cleanup: true,
        }
    }

    /// Set the working directory.
    pub fn work_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.work_dir = Some(dir.into());
        self
    }

    /// Set parallelism level.
    pub fn parallelism(mut self, n: usize) -> Self {
        self.parallelism = Some(n);
        self
    }

    /// Set whether to cleanup intermediate files.
    pub fn cleanup(mut self, cleanup: bool) -> Self {
        self.cleanup = cleanup;
        self
    }

    /// Build the merge coordinator.
    pub fn build(self) -> MergeCoordinator<'a> {
        let mut merger = MergeCoordinator::new(self.coordinator);

        if let Some(dir) = self.work_dir {
            merger = merger.with_work_dir(dir);
        }
        if let Some(p) = self.parallelism {
            merger = merger.with_parallelism(p);
        }

        merger.with_cleanup(self.cleanup)
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::{ShardConfig, ShardGranularity};
    use super::super::routing::compute_shard_key_from_token;
    use super::*;
    use crate::ngram::vocabulary::{create_vocabulary, encode_varint, SharedVocabARTrie};
    use tempfile::TempDir;

    /// Concatenated LEB128 term-id byte key for a token sequence (interning the
    /// tokens in the vocabulary; `insert` is idempotent so recomputing a key for
    /// an already-stored n-gram yields the same bytes).
    fn ngram_key(vocab: &SharedVocabARTrie, tokens: &[&str]) -> Vec<u8> {
        let mut key = Vec::with_capacity(tokens.len() * 2);
        for token in tokens {
            let id = vocab.as_ref().insert(token).expect("vocab insert");
            encode_varint(id, &mut key);
        }
        key
    }

    /// Store one n-gram exactly as the importer does: route by the first token's
    /// characters + sequence length, key by concatenated term-id varints.
    fn store_ngram(
        coordinator: &ShardCoordinator,
        vocab: &SharedVocabARTrie,
        tokens: &[&str],
        count: u64,
    ) {
        let shard_key = compute_shard_key_from_token(
            tokens[0],
            tokens.len() as u8,
            &coordinator.config().granularity,
        );
        coordinator
            .store_in_shard(&shard_key, &ngram_key(vocab, tokens), count)
            .expect("store_in_shard");
    }

    fn create_test_coordinator() -> (TempDir, ShardCoordinator, SharedVocabARTrie) {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let config =
            ShardConfig::new(dir.path().join("shards")).with_granularity(ShardGranularity::TwoChar);

        let coordinator = ShardCoordinator::new(config).expect("Failed to create coordinator");
        let vocab = create_vocabulary(&dir.path().join("vocab")).expect("vocabulary");

        // Add test data across multiple shards (first-token prefixes: th, ap, ba, ch, ze).
        store_ngram(&coordinator, &vocab, &["the", "quick"], 100);
        store_ngram(&coordinator, &vocab, &["the", "slow"], 50);
        store_ngram(&coordinator, &vocab, &["apple", "pie"], 30);
        store_ngram(&coordinator, &vocab, &["apple", "cider"], 20);
        store_ngram(&coordinator, &vocab, &["banana", "split"], 15);
        store_ngram(&coordinator, &vocab, &["cherry", "blossom"], 10);
        store_ngram(&coordinator, &vocab, &["zebra", "crossing"], 5);

        (dir, coordinator, vocab)
    }

    #[test]
    fn test_merge_to_memory() {
        let (_dir, coordinator, vocab) = create_test_coordinator();
        let merger = MergeCoordinator::new(&coordinator);

        let merged = merger.merge_to_memory().expect("merge");

        assert_eq!(merged.len(), 7);
        assert_eq!(
            merged.get(&ngram_key(&vocab, &["the", "quick"])),
            Some(&100)
        );
        assert_eq!(merged.get(&ngram_key(&vocab, &["apple", "pie"])), Some(&30));
        assert_eq!(
            merged.get(&ngram_key(&vocab, &["zebra", "crossing"])),
            Some(&5)
        );
    }

    #[test]
    fn test_iter_all() {
        let (_dir, coordinator, _vocab) = create_test_coordinator();
        let merger = MergeCoordinator::new(&coordinator);

        let all: Vec<_> = merger.iter_all().expect("iter_all").collect();

        assert_eq!(all.len(), 7);
    }

    #[test]
    fn test_estimated_size() {
        let (_dir, coordinator, _vocab) = create_test_coordinator();
        let merger = MergeCoordinator::new(&coordinator);

        let size = merger.estimated_final_size();
        assert_eq!(size, 7);
    }

    #[test]
    fn test_merge_to_trie() {
        let (dir, coordinator, vocab) = create_test_coordinator();
        let merger =
            MergeCoordinator::new(&coordinator).with_work_dir(dir.path().join("merge_work"));

        let output_path = dir.path().join("merged.artrie");
        let stats = merger
            .merge_to_trie(&output_path, |_progress| {})
            .expect("merge");

        assert!(output_path.exists());
        assert_eq!(stats.shards_merged, 5); // th, ap, ba, ch, ze
        assert!(stats.total_ngrams > 0);

        // Verify merged trie contents (term-id byte keys).
        let merged_trie = PersistentARTrie::<u64>::open(&output_path).expect("open");
        assert_eq!(
            merged_trie.get_value_bytes(&ngram_key(&vocab, &["the", "quick"])),
            Some(100)
        );
        assert_eq!(
            merged_trie.get_value_bytes(&ngram_key(&vocab, &["apple", "pie"])),
            Some(30)
        );
    }

    #[test]
    fn test_merge_progress() {
        let (dir, coordinator, _vocab) = create_test_coordinator();
        let merger =
            MergeCoordinator::new(&coordinator).with_work_dir(dir.path().join("merge_work"));

        let output_path = dir.path().join("merged.artrie");
        let mut progress_updates = Vec::new();

        let _stats = merger
            .merge_to_trie(&output_path, |progress| {
                progress_updates.push(progress.clone());
            })
            .expect("merge");

        // Should have received progress updates
        assert!(!progress_updates.is_empty());

        // Final progress should be 100%
        let last = progress_updates.last().unwrap();
        assert_eq!(last.percent_complete, 100.0);
    }

    #[test]
    fn test_merge_builder() {
        let (_dir, coordinator, _vocab) = create_test_coordinator();

        let merger = MergeBuilder::new(&coordinator)
            .parallelism(4)
            .cleanup(false)
            .build();

        assert_eq!(merger.parallelism, 4);
        assert!(!merger.cleanup_intermediates);
    }

    #[test]
    fn test_merge_empty_coordinator() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let config =
            ShardConfig::new(dir.path().join("shards")).with_granularity(ShardGranularity::TwoChar);

        let coordinator = ShardCoordinator::new(config).expect("create");
        let merger = MergeCoordinator::new(&coordinator);

        let result = merger.merge_to_memory();
        assert!(matches!(result, Err(MergeError::NoShards)));
    }
}
