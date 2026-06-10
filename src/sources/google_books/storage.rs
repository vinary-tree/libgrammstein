//! N-gram storage backend abstraction for Google Books import.
//!
//! This module provides a unified interface for storing n-grams during import,
//! supporting both single-trie and sharded storage backends.
//!
//! # Backends
//!
//! - **SingleTrie**: A single `PersistentARTrie<u64>` (byte-keyed) protected by
//!   `Arc<RwLock>`. Simple but has write contention with multiple workers.
//!
//! - **Sharded**: Distributes n-grams across multiple byte-keyed tries based on
//!   prefix routing. Eliminates write contention for parallel imports.
//!
//! # Vocabulary-Indexed Encoding
//!
//! When a `SharedVocabulary` is provided, n-gram keys are encoded as raw LEB128
//! varint byte sequences stored directly in the byte-keyed trie. This:
//! - Fixes the delimiter bug when tokens contain `|`
//! - Provides compact storage (1-2 bytes per word for common words)
//! - Supports unlimited vocabulary size
//! - Eliminates Latin-1 char conversion overhead

use super::checkpoint::{CheckpointError, ImportCheckpoint};
use super::config::GoogleBooksConfig;
use super::sharding::{CheckpointHandle, ShardCoordinator, ShardKey};
use crate::ngram::vocabulary::{
    try_encode_ngram_key_lockfree_bytes, with_encoded_ngram_key_lockfree, SharedConcurrentVocab,
    VocabularyError,
};
use libdictenstein::persistent_artrie::PersistentARTrie;
use liblevenshtein::dictionary::Dictionary;
use parking_lot::RwLock;
use smallvec::SmallVec;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use thiserror::Error;

/// Error type for storage operations.
#[derive(Error, Debug)]
pub enum StorageError {
    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Trie operation failed.
    #[error("Trie error: {0}")]
    Trie(String),

    /// Shard coordinator error.
    #[error("Shard coordinator error: {0}")]
    Coordinator(#[from] super::sharding::CoordinatorError),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Vocabulary error.
    #[error("Vocabulary error: {0}")]
    Vocabulary(#[from] VocabularyError),
}

/// Result type for storage operations.
pub type StorageResult<T> = Result<T, StorageError>;

/// Statistics for storage operations.
#[derive(Debug, Default)]
pub struct StorageStats {
    /// Total n-grams stored (including duplicates).
    pub total_ngrams: AtomicU64,

    /// Unique n-grams stored.
    pub unique_ngrams: AtomicU64,
}

impl StorageStats {
    /// Record n-grams stored.
    pub fn record(&self, total: u64, unique: u64) {
        self.total_ngrams.fetch_add(total, Ordering::Relaxed);
        self.unique_ngrams.fetch_add(unique, Ordering::Relaxed);
    }
}

/// Unified storage backend for n-gram import.
///
/// This enum allows the importer to use either single-trie or sharded storage
/// without changing the import logic.
///
/// When a `SharedVocabulary` is attached, n-gram keys are encoded using
/// vocabulary-indexed (PUA character) encoding instead of pipe-separated tokens.
pub enum NgramStorage {
    /// Single trie storage (byte-keyed).
    SingleTrie {
        /// The trie instance (byte-keyed). Holds both n-gram data and
        /// `ImportCheckpoint` metadata (metadata keys are prefixed with
        /// `\x00` per checkpoint.rs convention).
        trie: Arc<RwLock<PersistentARTrie<u64>>>,
        /// Optional lock-free concurrent vocabulary for encoding.
        vocabulary: Option<SharedConcurrentVocab>,
        /// Storage statistics.
        stats: Arc<StorageStats>,
    },

    /// Sharded storage using prefix-based routing.
    Sharded {
        /// The shard coordinator.
        coordinator: ShardCoordinator,
        /// Auxiliary checkpoint-metadata trie. Stores `ImportCheckpoint`
        /// progress separately from the n-gram data shards. Created at
        /// `{output_path}.checkpoint.artrie`.
        checkpoint_trie: Arc<RwLock<PersistentARTrie<u64>>>,
        /// Optional lock-free concurrent vocabulary for encoding.
        vocabulary: Option<SharedConcurrentVocab>,
        /// Storage statistics.
        stats: Arc<StorageStats>,
    },
}

impl NgramStorage {
    /// Create storage based on configuration and estimated n-gram count.
    ///
    /// Automatically chooses single-trie or sharded based on configuration
    /// and dataset size.
    pub fn create(config: &GoogleBooksConfig, estimated_ngrams: u64) -> StorageResult<Self> {
        let use_sharding = config.should_use_sharding(estimated_ngrams);

        if use_sharding {
            Self::create_sharded(config)
        } else {
            Self::create_single_trie(&config.output_path)
        }
    }

    /// Create single-trie storage.
    pub fn create_single_trie(output_path: &Path) -> StorageResult<Self> {
        Self::create_single_trie_with_vocabulary(output_path, None)
    }

    /// Create single-trie storage with optional vocabulary.
    ///
    /// Enables slot-level dirty tracking for optimized checkpoints (90%+ I/O reduction).
    pub fn create_single_trie_with_vocabulary(
        output_path: &Path,
        vocabulary: Option<SharedConcurrentVocab>,
    ) -> StorageResult<Self> {
        let trie = if output_path.exists() {
            log::info!("Opening existing trie at {:?}", output_path);
            PersistentARTrie::open_with_slot_tracking(output_path)
                .map_err(|e| StorageError::Trie(format!("Failed to open trie: {}", e)))?
        } else {
            log::info!("Creating new trie at {:?}", output_path);
            PersistentARTrie::create_with_slot_tracking(output_path)
                .map_err(|e| StorageError::Trie(format!("Failed to create trie: {}", e)))?
        };
        // The lock-free overlay is always-on now (libdictenstein flips to it on
        // create); the explicit enable_lockfree() toggle was removed.

        Ok(Self::SingleTrie {
            trie: Arc::new(RwLock::new(trie)),
            vocabulary,
            stats: Arc::new(StorageStats::default()),
        })
    }

    /// Create sharded storage.
    pub fn create_sharded(config: &GoogleBooksConfig) -> StorageResult<Self> {
        Self::create_sharded_with_vocabulary(config, None)
    }

    /// Create sharded storage with optional vocabulary.
    pub fn create_sharded_with_vocabulary(
        config: &GoogleBooksConfig,
        vocabulary: Option<SharedConcurrentVocab>,
    ) -> StorageResult<Self> {
        let shard_config = config.to_shard_config();

        log::info!(
            "Creating sharded storage at {:?} with {:?} granularity",
            shard_config.shard_dir,
            shard_config.granularity
        );

        let coordinator = ShardCoordinator::new_with_checkpoints(shard_config)?;

        // Auxiliary checkpoint-metadata trie at {output_path}.checkpoint.artrie.
        // Holds ImportCheckpoint progress separately from the n-gram data
        // shards so checkpoint metadata can be atomically persisted on its
        // own cadence. Created if missing; opened with WAL recovery if present.
        let checkpoint_trie_path = config.output_path.with_extension("checkpoint.artrie");
        let checkpoint_trie = if checkpoint_trie_path.exists() {
            PersistentARTrie::open(&checkpoint_trie_path).map_err(|e| {
                StorageError::Trie(format!(
                    "Failed to open checkpoint metadata trie at {}: {}",
                    checkpoint_trie_path.display(),
                    e
                ))
            })?
        } else {
            PersistentARTrie::create(&checkpoint_trie_path).map_err(|e| {
                StorageError::Trie(format!(
                    "Failed to create checkpoint metadata trie at {}: {}",
                    checkpoint_trie_path.display(),
                    e
                ))
            })?
        };

        Ok(Self::Sharded {
            coordinator,
            checkpoint_trie: Arc::new(RwLock::new(checkpoint_trie)),
            vocabulary,
            stats: Arc::new(StorageStats::default()),
        })
    }

    /// Access the trie that holds `ImportCheckpoint` metadata.
    ///
    /// In single-trie mode this returns the inner data trie (which also holds
    /// the `\x00`-prefixed checkpoint keys). In sharded mode this returns the
    /// dedicated auxiliary checkpoint trie.
    ///
    /// Callers that just need to save/load/delete the standard
    /// `ImportCheckpoint` should prefer the high-level methods
    /// (`save_import_checkpoint` / `load_import_checkpoint` /
    /// `delete_import_checkpoint`) — direct trie access is reserved for the
    /// MKN compute path in single-trie mode (which iterates n-gram data
    /// living in the same trie).
    pub fn checkpoint_trie(&self) -> &Arc<RwLock<PersistentARTrie<u64>>> {
        match self {
            Self::SingleTrie { trie, .. } => trie,
            Self::Sharded {
                checkpoint_trie, ..
            } => checkpoint_trie,
        }
    }

    /// Persist an `ImportCheckpoint` to the storage's checkpoint trie and
    /// flush it to disk (truncating its WAL).
    pub fn save_import_checkpoint(&self, checkpoint: &ImportCheckpoint) -> StorageResult<()> {
        let trie_arc = self.checkpoint_trie();
        let mut trie = trie_arc.write();
        checkpoint.save_to_trie(&mut *trie).map_err(|e| {
            StorageError::Trie(format!("Failed to save import checkpoint to trie: {}", e))
        })?;
        trie.checkpoint()
            .map_err(|e| StorageError::Trie(format!("Failed to flush checkpoint trie: {}", e)))?;
        Ok(())
    }

    /// Save an `ImportCheckpoint` to the checkpoint trie using an async-WAL
    /// sync (durable but without truncating the WAL). Used by periodic
    /// checkpoints where WAL replay on next open is acceptable.
    pub fn save_import_checkpoint_async(&self, checkpoint: &ImportCheckpoint) -> StorageResult<()> {
        let trie_arc = self.checkpoint_trie();
        let mut trie = trie_arc.write();
        checkpoint.save_to_trie(&mut *trie).map_err(|e| {
            StorageError::Trie(format!("Failed to save import checkpoint to trie: {}", e))
        })?;
        trie.sync().map_err(|e| {
            StorageError::Trie(format!("Failed to sync checkpoint trie WAL: {}", e))
        })?;
        Ok(())
    }

    /// Load an `ImportCheckpoint` from the storage's checkpoint trie, if one
    /// exists. Returns `Ok(None)` when no checkpoint has been saved.
    pub fn load_import_checkpoint(&self) -> Result<Option<ImportCheckpoint>, CheckpointError> {
        let trie_arc = self.checkpoint_trie();
        let trie = trie_arc.read();
        ImportCheckpoint::load_from_trie(&*trie)
    }

    /// Delete the persisted `ImportCheckpoint` (call after a successful
    /// finalize so that the next invocation does not see stale progress).
    /// Returns the number of checkpoint keys removed.
    pub fn delete_import_checkpoint(&self) -> Result<usize, CheckpointError> {
        let trie_arc = self.checkpoint_trie();
        let mut trie = trie_arc.write();
        ImportCheckpoint::delete_from_trie(&mut *trie)
    }

    /// Resume or start storage based on configuration.
    ///
    /// For sharded storage, this loads existing checkpoint state.
    pub fn resume_or_start(
        config: &GoogleBooksConfig,
        estimated_ngrams: u64,
    ) -> StorageResult<Self> {
        Self::resume_or_start_with_vocabulary(config, estimated_ngrams, None)
    }

    /// Resume or start storage with optional vocabulary.
    ///
    /// For sharded storage, this loads existing checkpoint state.
    pub fn resume_or_start_with_vocabulary(
        config: &GoogleBooksConfig,
        estimated_ngrams: u64,
        vocabulary: Option<SharedConcurrentVocab>,
    ) -> StorageResult<Self> {
        let use_sharding = config.should_use_sharding(estimated_ngrams);

        if use_sharding {
            let shard_config = config.to_shard_config();

            log::info!(
                "Resuming/starting sharded storage at {:?}",
                shard_config.shard_dir
            );

            let coordinator = ShardCoordinator::resume_or_start(shard_config)?;

            // Open or create the auxiliary checkpoint-metadata trie. See the
            // doc comment on `Sharded::checkpoint_trie` for why this lives
            // alongside the coordinator.
            let checkpoint_trie_path = config.output_path.with_extension("checkpoint.artrie");
            let checkpoint_trie = if checkpoint_trie_path.exists() {
                PersistentARTrie::open(&checkpoint_trie_path).map_err(|e| {
                    StorageError::Trie(format!(
                        "Failed to open checkpoint metadata trie at {}: {}",
                        checkpoint_trie_path.display(),
                        e
                    ))
                })?
            } else {
                PersistentARTrie::create(&checkpoint_trie_path).map_err(|e| {
                    StorageError::Trie(format!(
                        "Failed to create checkpoint metadata trie at {}: {}",
                        checkpoint_trie_path.display(),
                        e
                    ))
                })?
            };

            Ok(Self::Sharded {
                coordinator,
                checkpoint_trie: Arc::new(RwLock::new(checkpoint_trie)),
                vocabulary,
                stats: Arc::new(StorageStats::default()),
            })
        } else {
            Self::create_single_trie_with_vocabulary(&config.output_path, vocabulary)
        }
    }

    /// Check if this is sharded storage.
    pub fn is_sharded(&self) -> bool {
        matches!(self, Self::Sharded { .. })
    }

    /// Check if vocabulary-indexed encoding is enabled.
    pub fn has_vocabulary(&self) -> bool {
        match self {
            Self::SingleTrie { vocabulary, .. } => vocabulary.is_some(),
            Self::Sharded { vocabulary, .. } => vocabulary.is_some(),
        }
    }

    /// Get the vocabulary reference, if any.
    pub fn vocabulary(&self) -> Option<&SharedConcurrentVocab> {
        match self {
            Self::SingleTrie { vocabulary, .. } => vocabulary.as_ref(),
            Self::Sharded { vocabulary, .. } => vocabulary.as_ref(),
        }
    }

    /// Check if the vocabulary has unsaved changes.
    ///
    /// This replaces `vocabulary_current_lsn()` and `vocabulary_synced_lsn()`.
    /// Returns `false` if no vocabulary is configured or vocabulary is clean.
    pub fn vocabulary_is_dirty(&self) -> bool {
        self.vocabulary().is_some_and(|v| v.is_dirty())
    }

    /// Encode tokens to a raw byte n-gram key using vocabulary.
    ///
    /// Uses lock-free CAS for concurrent vocabulary access and returns
    /// the raw LEB128 varint-encoded byte key directly.
    ///
    /// Vocabulary is required. Returns an error if no vocabulary is configured.
    pub fn encode_tokens(&self, tokens: &[&str]) -> StorageResult<Vec<u8>> {
        let vocab = self
            .vocabulary()
            .ok_or_else(|| StorageError::Config("Vocabulary required for token encoding".into()))?;
        try_encode_ngram_key_lockfree_bytes(tokens, vocab).map_err(StorageError::from)
    }

    /// Store tokens as an n-gram with count, using vocabulary encoding if available.
    ///
    /// This is the recommended method for storing n-grams when you have tokens.
    /// If vocabulary is enabled, the tokens are encoded to a varint key and routed
    /// based on the original tokens (not the encoded key).
    ///
    /// For single-trie mode, uses lock-free `increment_cas()` — workers only acquire
    /// a shared read lock, eliminating the write contention that previously serialized
    /// all 12+ parallel workers.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The n-gram tokens (e.g., ["the", "quick", "brown"])
    /// * `count` - The n-gram count
    ///
    /// # Returns
    ///
    /// `true` if this was a new n-gram.
    pub fn store_tokens(&self, tokens: &[&str], count: u64) -> StorageResult<bool> {
        let vocab = self
            .vocabulary()
            .ok_or_else(|| StorageError::Config("Vocabulary required for token encoding".into()))?;

        match self {
            Self::SingleTrie { trie, stats, .. } => {
                let guard = trie.read();
                // Zero-alloc path via thread-local buffer
                let is_new = with_encoded_ngram_key_lockfree(tokens, vocab, |encoded_key| {
                    // Single overlay read (overlay-default single source of truth).
                    let is_new = guard.get_value_bytes(encoded_key).is_none();
                    guard.increment_cas(encoded_key, count);
                    is_new
                });

                stats.record(count, if is_new { 1 } else { 0 });
                Ok(is_new)
            }
            Self::Sharded {
                coordinator, stats, ..
            } => {
                // Route based on original tokens, store encoded byte key
                let shard_key = coordinator.route_tokens(tokens);
                let is_new = with_encoded_ngram_key_lockfree(tokens, vocab, |encoded_key| {
                    coordinator.store_in_shard(&shard_key, encoded_key, count)
                })?;
                stats.record(count, if is_new { 1 } else { 0 });
                Ok(is_new)
            }
        }
    }

    /// Store an n-gram string with count, splitting on spaces into tokens.
    ///
    /// This is a convenience wrapper around `store_tokens` that avoids heap
    /// allocation by using `SmallVec<[&str; 5]>` for the token split (n-gram
    /// orders 1-5 fit inline on the stack).
    ///
    /// # Arguments
    ///
    /// * `ngram` - Space-separated n-gram string (e.g., "the quick brown")
    /// * `count` - The n-gram count
    pub fn store_ngram(&self, ngram: &str, count: u64) -> StorageResult<bool> {
        let tokens: SmallVec<[&str; 5]> = ngram.split(' ').collect();
        self.store_tokens(&tokens, count)
    }

    /// Insert an n-gram string into a pending prefix transaction.
    ///
    /// This is a convenience wrapper around `tx_insert_tokens` that avoids heap
    /// allocation by using `SmallVec<[&str; 5]>` for the token split.
    ///
    /// # Arguments
    ///
    /// * `tx` - The active transaction from `begin_prefix_tx()`
    /// * `ngram` - Space-separated n-gram string (e.g., "the quick brown")
    /// * `count` - The n-gram count
    pub fn tx_insert_ngram(
        &self,
        tx: &mut StoragePrefixTx,
        ngram: &str,
        count: u64,
    ) -> StorageResult<()> {
        let tokens: SmallVec<[&str; 5]> = ngram.split(' ').collect();
        self.tx_insert_tokens(tx, &tokens, count)
    }

    /// Route tokens to their shard key (for sharded mode with vocabulary encoding).
    ///
    /// Returns `None` for single-trie mode.
    pub fn route_tokens(&self, tokens: &[&str]) -> Option<ShardKey> {
        match self {
            Self::SingleTrie { .. } => None,
            Self::Sharded { coordinator, .. } => Some(coordinator.route_tokens(tokens)),
        }
    }

    /// Get storage statistics.
    pub fn stats(&self) -> &StorageStats {
        match self {
            Self::SingleTrie { stats, .. } => stats,
            Self::Sharded { stats, .. } => stats,
        }
    }

    /// Store an n-gram with count.
    ///
    /// For single-trie mode, uses lock-free `increment_cas()` — workers only acquire
    /// a shared read lock, eliminating write contention.
    ///
    /// Returns `true` if this was a new n-gram.
    pub fn store(&self, ngram: &str, count: u64) -> StorageResult<bool> {
        let ngram_bytes = ngram.as_bytes();
        match self {
            Self::SingleTrie { trie, stats, .. } => {
                let guard = trie.read();
                // Single overlay read (overlay-default single source of truth).
                let is_new = guard.get_value_bytes(ngram_bytes).is_none();
                guard.increment_cas(ngram_bytes, count);

                stats.record(count, if is_new { 1 } else { 0 });
                Ok(is_new)
            }
            Self::Sharded {
                coordinator, stats, ..
            } => {
                let is_new = coordinator.store_ngram(ngram, count)?;
                stats.record(count, if is_new { 1 } else { 0 });
                Ok(is_new)
            }
        }
    }

    /// Store multiple n-grams to the same shard efficiently.
    ///
    /// For single-trie mode, this is equivalent to calling `store` repeatedly.
    /// For sharded mode, this batches writes to the same shard.
    ///
    /// # Arguments
    ///
    /// * `shard_key` - The shard key (ignored in single-trie mode)
    /// * `ngrams` - Iterator of (ngram, count) pairs
    ///
    /// # Returns
    ///
    /// Number of new (unique) n-grams stored.
    pub fn store_batch<'a, I>(&self, shard_key: Option<&ShardKey>, ngrams: I) -> StorageResult<u64>
    where
        I: Iterator<Item = (&'a [u8], u64)>,
    {
        match self {
            Self::SingleTrie { trie, stats, .. } => {
                let guard = trie.read();
                let mut new_count = 0u64;
                let mut total_count = 0u64;

                for (ngram, count) in ngrams {
                    // Single overlay read (overlay-default single source of truth).
                    let is_new = guard.get_value_bytes(ngram).is_none();
                    guard.increment_cas(ngram, count);

                    if is_new {
                        new_count += 1;
                    }
                    total_count += count;
                }

                stats.record(total_count, new_count);
                Ok(new_count)
            }
            Self::Sharded {
                coordinator, stats, ..
            } => {
                let key = shard_key.ok_or_else(|| {
                    StorageError::Config("Shard key required for sharded storage batch".to_string())
                })?;

                let ngrams_vec: Vec<_> = ngrams.collect();
                let total_count: u64 = ngrams_vec.iter().map(|(_, c)| *c).sum();

                let new_count = coordinator.store_ngrams_batch(key, ngrams_vec.into_iter())?;

                stats.record(total_count, new_count);
                Ok(new_count)
            }
        }
    }

    /// Get the count for an n-gram.
    ///
    /// For single-trie mode, checks both the lock-free and persistent layers and
    /// sums their values (the lock-free layer accumulates counts that haven't been
    /// merged to persistent yet).
    ///
    /// For vocabulary-indexed encoding with sharded storage, use `get_tokens` instead
    /// to ensure correct routing based on original tokens.
    pub fn get(&self, ngram: &str) -> Option<u64> {
        let ngram_bytes = ngram.as_bytes();
        match self {
            Self::SingleTrie { trie, .. } => {
                let guard = trie.read();
                // Single overlay read (overlay-default single source of truth); the
                // prior get_lockfree + get_value_bytes sum read the same leaf twice.
                match guard.get_value_bytes(ngram_bytes).unwrap_or(0) {
                    0 => None,
                    total => Some(total),
                }
            }
            Self::Sharded { coordinator, .. } => coordinator.get(ngram),
        }
    }

    /// Get the count for an n-gram by tokens (for vocabulary-indexed encoding).
    ///
    /// This encodes the tokens to a varint key and routes based on the original tokens,
    /// ensuring correct shard routing for vocabulary-indexed storage.
    ///
    /// For single-trie mode, reads the overlay (the single source of truth).
    ///
    /// # Arguments
    ///
    /// * `tokens` - The n-gram tokens (e.g., ["the", "quick", "brown"])
    pub fn get_tokens(&self, tokens: &[&str]) -> Option<u64> {
        let encoded_key = self.encode_tokens(tokens).ok()?;

        match self {
            Self::SingleTrie { trie, .. } => {
                let guard = trie.read();
                // Single overlay read (overlay-default single source of truth).
                match guard.get_value_bytes(&encoded_key).unwrap_or(0) {
                    0 => None,
                    total => Some(total),
                }
            }
            Self::Sharded { coordinator, .. } => {
                let shard_key = coordinator.route_tokens(tokens);
                coordinator.get_in_shard(&shard_key, &encoded_key)
            }
        }
    }

    /// Check if an n-gram exists.
    pub fn contains(&self, ngram: &str) -> bool {
        self.get(ngram).is_some()
    }

    /// Check if an n-gram exists by tokens (for vocabulary-indexed encoding).
    pub fn contains_tokens(&self, tokens: &[&str]) -> bool {
        self.get_tokens(tokens).is_some()
    }

    /// Checkpoint the storage.
    ///
    /// For single-trie mode, `checkpoint()` serializes the lock-free overlay snapshot
    /// (the durable production state under the overlay-default write mode) into the
    /// on-disk image. The obsolete `merge_lockfree_values_to_persistent` pre-step has
    /// been removed.
    pub fn checkpoint(&self) -> StorageResult<()> {
        match self {
            Self::SingleTrie { trie, .. } => {
                let guard = trie.write();
                guard
                    .checkpoint()
                    .map_err(|e| StorageError::Trie(format!("Checkpoint failed: {}", e)))
            }
            Self::Sharded { coordinator, .. } => {
                coordinator.coordinated_checkpoint()?;
                Ok(())
            }
        }
    }

    /// Persist to disk.
    ///
    /// For single-trie mode, `checkpoint()` serializes the overlay snapshot (the
    /// durable state under overlay-default) — a bare WAL sync would not capture the
    /// lock-free `increment_cas` counts. The obsolete merge pre-step has been removed.
    pub fn sync(&self) -> StorageResult<()> {
        match self {
            Self::SingleTrie { trie, .. } => {
                let guard = trie.write();
                guard
                    .checkpoint()
                    .map_err(|e| StorageError::Trie(format!("Sync failed: {}", e)))
            }
            Self::Sharded { coordinator, .. } => {
                coordinator.sync_all()?;
                Ok(())
            }
        }
    }

    /// Flush lock-free overlays for shards exceeding the entry threshold.
    ///
    /// This bounds per-shard lock-free memory usage during high-parallelism
    /// imports. Only acquires write locks on shards that need flushing.
    ///
    /// For single-trie mode, flushes the single trie if it has lock-free data.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Maximum lock-free entries per shard before flushing.
    ///
    /// # Returns
    ///
    /// The number of shards/tries that were flushed.
    pub fn flush_lockfree_over_threshold(&self, threshold: u64) -> StorageResult<usize> {
        match self {
            Self::SingleTrie { trie, .. } => {
                // Single trie: checkpoint to persist the overlay and reclaim its
                // memory. (We don't track per-entry counts on the single trie, so we
                //  always checkpoint when this is called, which is fine since
                //  single-trie mode is not the OOM-prone path.)
                let guard = trie.write();
                guard
                    .checkpoint()
                    .map_err(|e| StorageError::Trie(format!("Lock-free flush failed: {}", e)))?;
                Ok(1)
            }
            Self::Sharded { coordinator, .. } => coordinator
                .flush_lockfree_over_threshold(threshold)
                .map_err(StorageError::from),
        }
    }

    /// Sync the vocabulary WAL to disk (if present).
    ///
    /// Lightweight durability: syncs the WAL (the overlay image is published only
    /// by `checkpoint_vocabulary`). Crash recovery replays the WAL tail on reopen.
    pub fn sync_vocabulary(&self) -> StorageResult<()> {
        let vocab = match self {
            Self::SingleTrie { vocabulary, .. } => vocabulary,
            Self::Sharded { vocabulary, .. } => vocabulary,
        };
        if let Some(v) = vocab {
            v.sync()
                .map_err(|e| StorageError::Trie(format!("Vocabulary sync failed: {}", e)))?;
        }
        Ok(())
    }

    /// Checkpoint the vocabulary (if present).
    ///
    /// Publishes the vocabulary's overlay snapshot to disk. The migrated single
    /// lock-free `PersistentVocabARTrie` has no separate persistent layer to merge
    /// into — `checkpoint()` is the durable image publication.
    pub fn checkpoint_vocabulary(&self) -> StorageResult<()> {
        let vocab = match self {
            Self::SingleTrie { vocabulary, .. } => vocabulary,
            Self::Sharded { vocabulary, .. } => vocabulary,
        };
        if let Some(v) = vocab {
            v.checkpoint()
                .map_err(|e| StorageError::Trie(format!("Vocabulary checkpoint failed: {}", e)))?;
        }
        Ok(())
    }

    /// Rotate vocabulary WAL without full checkpoint serialization.
    ///
    /// Unlike [`checkpoint_vocabulary()`], which publishes a full overlay image
    /// (causing file growth), this syncs the WAL and retains it for replay — no
    /// overlay image is written — so it provides crash-recovery durability during
    /// bulk imports without vocabulary file bloat.
    ///
    /// # When to Use
    ///
    /// Use `rotate_vocabulary_wal()` during bulk imports:
    /// - Periodic durability without file bloat
    /// - WAL ensures crash recovery
    ///
    /// Use `checkpoint_vocabulary()` for final compaction:
    /// - After import completes successfully
    /// - Reduces recovery time by avoiding WAL replay
    pub fn rotate_vocabulary_wal(&self) -> StorageResult<()> {
        let vocab = match self {
            Self::SingleTrie { vocabulary, .. } => vocabulary,
            Self::Sharded { vocabulary, .. } => vocabulary,
        };
        if let Some(v) = vocab {
            // Sync + rotate the WAL while retaining it for replay; no overlay image
            // is published (that is checkpoint_vocabulary's job), so no file bloat.
            v.rotate_wal().map_err(|e| {
                StorageError::Trie(format!("Vocabulary WAL rotation failed: {}", e))
            })?;
        }
        Ok(())
    }

    /// Rotate the vocabulary WAL for periodic bulk-import durability.
    ///
    /// Under the migrated single lock-free `PersistentVocabARTrie` this is now
    /// equivalent to [`rotate_vocabulary_wal()`]: it syncs + retains the WAL with
    /// no overlay image published. The historical two-layer "merge into the
    /// persistent `reverse_index` HashMap" step (and its ~1.7 GB rebuild spike)
    /// no longer exists — the overlay *is* the vocabulary, so there is nothing to
    /// merge. Retained as a distinct entry point for its existing callers.
    pub fn merge_and_rotate_vocabulary_wal(&self) -> StorageResult<()> {
        let vocab = match self {
            Self::SingleTrie { vocabulary, .. } => vocabulary,
            Self::Sharded { vocabulary, .. } => vocabulary,
        };
        if let Some(v) = vocab {
            // Sync + rotate the WAL while retaining it for replay; no overlay image
            // is published, so no file bloat (see rotate_vocabulary_wal).
            v.rotate_wal().map_err(|e| {
                StorageError::Trie(format!("Vocabulary WAL rotation failed: {}", e))
            })?;
        }
        Ok(())
    }

    /// Check if a shard is currently syncing (for defer-and-continue pattern).
    ///
    /// Workers can use this to check if their target shard is syncing and
    /// defer to another job if so, avoiding blocking.
    ///
    /// For single-trie mode, always returns `false`.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The n-gram tokens to check routing for
    ///
    /// # Returns
    ///
    /// `true` if the shard that would store these tokens is currently syncing.
    pub fn is_shard_syncing(&self, tokens: &[&str]) -> bool {
        match self {
            Self::SingleTrie { .. } => false, // Single trie is never "syncing"
            Self::Sharded { coordinator, .. } => {
                let shard_key = coordinator.route_tokens(tokens);
                coordinator.is_shard_syncing(&shard_key)
            }
        }
    }

    /// Check if a shard (by key) is currently syncing.
    ///
    /// For single-trie mode, always returns `false`.
    pub fn is_shard_key_syncing(&self, shard_key: &ShardKey) -> bool {
        match self {
            Self::SingleTrie { .. } => false,
            Self::Sharded { coordinator, .. } => coordinator.is_shard_syncing(shard_key),
        }
    }

    /// Check if the shard for a given file prefix is currently syncing.
    ///
    /// This is used by workers to check if they should defer a job
    /// because the target shard is being synced.
    ///
    /// For single-trie mode, always returns `false`.
    ///
    /// # Arguments
    ///
    /// * `prefix` - The file prefix (e.g., "th", "to")
    /// * `order` - The n-gram order (1-5)
    ///
    /// # Returns
    ///
    /// `true` if the shard that would store n-grams from this prefix is syncing.
    pub fn is_prefix_shard_syncing(&self, prefix: &str, order: u8) -> bool {
        match self {
            Self::SingleTrie { .. } => false,
            Self::Sharded { coordinator, .. } => {
                let shard_key = super::sharding::shard_key_for_file_prefix(
                    prefix,
                    order,
                    &coordinator.config().granularity,
                );
                coordinator.is_shard_syncing(&shard_key)
            }
        }
    }

    /// Sync all shards in parallel.
    ///
    /// This enables non-blocking checkpoints where workers can continue
    /// on non-syncing shards while the sync is in progress.
    ///
    /// For single-trie mode, performs a regular sync.
    ///
    /// # Arguments
    ///
    /// * `max_concurrent` - Maximum shards to sync in parallel (recommended: 8)
    ///
    /// # Returns
    ///
    /// Number of shards synced, or error if any sync failed.
    pub fn sync_parallel(&self, max_concurrent: usize) -> StorageResult<usize> {
        match self {
            Self::SingleTrie { trie, .. } => {
                let guard = trie.write();
                guard
                    .checkpoint()
                    .map_err(|e| StorageError::Trie(format!("Sync failed: {}", e)))?;
                Ok(1) // Single trie counts as 1 sync
            }
            Self::Sharded { coordinator, .. } => {
                coordinator.sync_all_parallel(max_concurrent)?;
                Ok(coordinator.open_shard_count())
            }
        }
    }

    /// Checkpoint with parallel WAL flushing for non-blocking operation.
    ///
    /// This provides the same guarantees as `checkpoint()` but with better
    /// performance for large shard counts:
    ///
    /// 1. Parallel WAL sync across all shards
    /// 2. Sequential checkpoint/truncate (fast since data is synced)
    /// 3. Global checkpoint save
    ///
    /// Workers can continue on non-syncing shards during the sync phase.
    ///
    /// For single-trie mode, performs a regular checkpoint.
    ///
    /// # Arguments
    ///
    /// * `max_concurrent_syncs` - Maximum shards to sync in parallel (recommended: 8)
    pub fn checkpoint_parallel(&self, max_concurrent_syncs: usize) -> StorageResult<()> {
        match self {
            Self::SingleTrie { trie, .. } => {
                let guard = trie.write();
                guard
                    .checkpoint()
                    .map_err(|e| StorageError::Trie(format!("Checkpoint failed: {}", e)))
            }
            Self::Sharded { coordinator, .. } => {
                coordinator.coordinated_checkpoint_parallel(max_concurrent_syncs)?;
                Ok(())
            }
        }
    }

    /// Start async checkpoint (non-blocking).
    ///
    /// This initiates WAL rotation on all dirty shards and returns immediately.
    /// Workers can continue writing to the new WAL segments while background
    /// threads sync the old segments.
    ///
    /// # Returns
    ///
    /// A `CheckpointHandle` that can be used to:
    /// - Check if all syncs completed (`all_synced()`)
    /// - Wait for all syncs to complete (`wait_all()` or `wait_all_parallel()`)
    ///
    /// # Usage Pattern
    ///
    /// ```ignore
    /// let handle = storage.checkpoint_async()?;
    /// // Workers continue...
    /// handle.wait_all_parallel()?;
    /// storage.checkpoint_async_finish(8)?;
    /// ```
    pub fn checkpoint_async(&self) -> StorageResult<CheckpointHandle> {
        match self {
            Self::SingleTrie { trie, .. } => {
                // Single trie: checkpoint synchronously (the overlay snapshot is the
                // durable state; the obsolete merge pre-step has been removed)
                let guard = trie.write();
                guard
                    .checkpoint()
                    .map_err(|e| StorageError::Trie(format!("Checkpoint failed: {}", e)))?;
                // Return a dummy handle (no shards to track)
                Ok(CheckpointHandle::empty())
            }
            Self::Sharded { coordinator, .. } => coordinator
                .coordinated_checkpoint_async()
                .map_err(StorageError::from),
        }
    }

    /// Finish async checkpoint after sync is complete.
    ///
    /// Call this after `checkpoint_async().wait_all_parallel()` to truncate WALs
    /// and update checkpoint metadata.
    ///
    /// # Arguments
    ///
    /// * `max_concurrent` - Maximum shards to persist in parallel.
    pub fn checkpoint_async_finish(&self, max_concurrent: usize) -> StorageResult<()> {
        self.checkpoint_async_finish_with_progress(max_concurrent, None::<fn(usize, usize)>)
    }

    /// Finish async checkpoint with progress callback.
    ///
    /// Same as `checkpoint_async_finish()` but invokes a callback after each
    /// shard completes.
    ///
    /// # Arguments
    ///
    /// * `max_concurrent` - Maximum shards to persist in parallel.
    /// * `progress_callback` - Optional callback receiving (processed, total).
    pub fn checkpoint_async_finish_with_progress<F>(
        &self,
        max_concurrent: usize,
        progress_callback: Option<F>,
    ) -> StorageResult<()>
    where
        F: Fn(usize, usize) + Send + Sync,
    {
        match self {
            Self::SingleTrie { .. } => {
                // Already checkpointed in checkpoint_async()
                Ok(())
            }
            Self::Sharded { coordinator, .. } => coordinator
                .coordinated_checkpoint_finish_with_progress(max_concurrent, progress_callback)
                .map_err(StorageError::from),
        }
    }

    /// Close the storage (checkpoint and release resources).
    pub fn close(&self) -> StorageResult<()> {
        match self {
            Self::SingleTrie { trie, .. } => {
                let guard = trie.write();
                guard
                    .checkpoint()
                    .map_err(|e| StorageError::Trie(format!("Close checkpoint failed: {}", e)))
            }
            Self::Sharded { coordinator, .. } => {
                coordinator.close_all()?;
                Ok(())
            }
        }
    }

    /// Get total entry count.
    pub fn len(&self) -> u64 {
        match self {
            Self::SingleTrie { trie, .. } => {
                let guard = trie.read();
                guard.len().unwrap_or(0) as u64
            }
            Self::Sharded { coordinator, .. } => coordinator.total_entry_count(),
        }
    }

    /// Check if storage is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Route an n-gram to its shard key (for sharded mode).
    ///
    /// Returns `None` for single-trie mode.
    pub fn route_ngram(&self, ngram: &str) -> Option<ShardKey> {
        match self {
            Self::SingleTrie { .. } => None,
            Self::Sharded { coordinator, .. } => Some(coordinator.route_ngram(ngram)),
        }
    }

    /// Get the underlying trie (for single-trie mode only).
    ///
    /// Returns `None` for sharded mode.
    pub fn as_single_trie(&self) -> Option<&Arc<RwLock<PersistentARTrie<u64>>>> {
        match self {
            Self::SingleTrie { trie, .. } => Some(trie),
            Self::Sharded { .. } => None,
        }
    }

    /// Get the shard coordinator (for sharded mode only).
    ///
    /// Returns `None` for single-trie mode.
    pub fn as_sharded(&self) -> Option<&ShardCoordinator> {
        match self {
            Self::SingleTrie { .. } => None,
            Self::Sharded { coordinator, .. } => Some(coordinator),
        }
    }

    /// Mark a prefix as completed (for sharded mode).
    pub fn mark_prefix_completed(&self, prefix: &str, order: u8) -> StorageResult<()> {
        if let Self::Sharded { coordinator, .. } = self {
            let key = super::sharding::shard_key_for_file_prefix(
                prefix,
                order,
                &coordinator.config().granularity,
            );
            coordinator.mark_prefix_completed(&key, prefix)?;
        }
        Ok(())
    }

    /// Check if a prefix is completed (for sharded mode).
    pub fn is_prefix_completed(&self, prefix: &str) -> bool {
        match self {
            Self::SingleTrie { .. } => false, // Single trie doesn't track prefixes
            Self::Sharded { coordinator, .. } => coordinator.is_prefix_completed(prefix),
        }
    }

    // ========================================================================
    // Document Transaction API (for idempotent prefix imports)
    // ========================================================================

    /// Begin a prefix transaction for atomic, idempotent n-gram import.
    ///
    /// This creates a document transaction that buffers all n-gram inserts
    /// until `commit_prefix_tx()` is called. If interrupted before commit,
    /// the transaction is automatically discarded on recovery.
    ///
    /// **Only supported for sharded storage.** Returns `None` for single-trie mode.
    ///
    /// # Key Properties
    ///
    /// - **Atomicity**: Either all n-grams are committed or none are
    /// - **Idempotency**: Uses SET semantics, so re-imports produce the same result
    /// - **Crash Safety**: Uncommitted transactions are discarded on WAL recovery
    ///
    /// # Arguments
    ///
    /// * `prefix` - The prefix file being imported (used as document ID)
    /// * `order` - The n-gram order (1-5), used for shard routing
    ///
    /// # Returns
    ///
    /// A `StoragePrefixTx` that must be passed to `tx_insert()` and
    /// eventually to `commit_prefix_tx()`, or `None` for single-trie mode
    /// OR for hash-based sharding granularities (see safety note below).
    ///
    /// # Safety constraint: only prefix-based granularities support chunked tx
    ///
    /// Chunked transactions bind to a single shard at `begin_prefix_tx` time,
    /// then route every subsequent `tx_insert_ngram` to that shard. This is
    /// correct under prefix-based granularities (`TwoChar`, `Adaptive` for
    /// the orders that resolve to prefix routing): all n-grams in a Google
    /// Books file share a 2-char prefix, so they all route to the same shard,
    /// and `route_tokens(first_token)` agrees with
    /// `shard_key_for_file_prefix(file_prefix)`.
    ///
    /// Under `CpuProportional` (hash-based), file prefix and first-token
    /// routing diverge — `hash("th")` and `hash("the")` produce different
    /// shard indices. If we bound the tx to `hash("th")` but n-grams write
    /// based on the first token's hash, get-by-tokens would later read from
    /// the wrong shard. Returning `None` here forces the caller to fall back
    /// to per-record `store_ngram`/`store_tokens`, which routes each n-gram
    /// individually via `route_tokens(tokens)` — correct for all
    /// granularities at the cost of losing the chunked-tx memory bound.
    pub fn begin_prefix_tx(
        &self,
        prefix: &str,
        order: u8,
    ) -> StorageResult<Option<StoragePrefixTx>> {
        match self {
            Self::SingleTrie { .. } => {
                // Single-trie mode doesn't support transactions
                // Caller should use store() directly
                Ok(None)
            }
            Self::Sharded { coordinator, .. } => {
                // Hash-based granularities can't use chunked-tx safely
                // (see doc comment above). Fall through to per-record path.
                if coordinator.config().granularity.is_hash_based() {
                    return Ok(None);
                }
                let shard_key = super::sharding::shard_key_for_file_prefix(
                    prefix,
                    order,
                    &coordinator.config().granularity,
                );
                let inner = coordinator.begin_prefix_tx(&shard_key, prefix)?;
                Ok(Some(StoragePrefixTx { inner }))
            }
        }
    }

    /// Insert an n-gram (as tokens) into a pending prefix transaction.
    ///
    /// The n-gram is encoded using the vocabulary (if present) and buffered
    /// in memory. It will be written atomically when the transaction is committed.
    /// Uses SET semantics (not increment), making re-imports idempotent.
    ///
    /// # Arguments
    ///
    /// * `tx` - The active transaction from `begin_prefix_tx()`
    /// * `tokens` - The n-gram tokens (e.g., ["the", "quick"])
    /// * `count` - The n-gram count
    pub fn tx_insert_tokens(
        &self,
        tx: &mut StoragePrefixTx,
        tokens: &[&str],
        count: u64,
    ) -> StorageResult<()> {
        // Encode tokens to raw byte key using vocabulary
        let encoded_key = self.encode_tokens(tokens)?;

        match self {
            Self::Sharded { coordinator, .. } => {
                coordinator.tx_insert(&mut tx.inner, &encoded_key, count);
                Ok(())
            }
            Self::SingleTrie { .. } => {
                // Should not happen - caller should check begin_prefix_tx result
                Err(StorageError::Config(
                    "Cannot use transaction with single-trie storage".to_string(),
                ))
            }
        }
    }

    /// Commit a prefix transaction atomically and mark the prefix as completed.
    ///
    /// This:
    /// 1. Writes all buffered n-grams to the WAL as a single batch
    /// 2. Applies them to the trie atomically
    /// 3. Marks the prefix as completed in the checkpoint state
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to commit (consumed)
    ///
    /// # Returns
    ///
    /// The number of n-grams that were committed.
    pub fn commit_prefix_tx(&self, tx: StoragePrefixTx) -> StorageResult<usize> {
        match self {
            Self::Sharded {
                coordinator, stats, ..
            } => {
                let count = coordinator.commit_prefix_tx(tx.inner)?;
                stats.record(count as u64, count as u64);
                Ok(count)
            }
            Self::SingleTrie { .. } => Err(StorageError::Config(
                "Cannot use transaction with single-trie storage".to_string(),
            )),
        }
    }

    /// Commit a prefix transaction chunk and begin a new transaction for
    /// the same prefix/shard.
    ///
    /// This commits the current chunk's buffered n-grams to the WAL WITHOUT
    /// marking the prefix as complete, then begins a fresh transaction for
    /// the next chunk. Used for chunked imports of large prefix files to
    /// bound per-transaction memory usage.
    ///
    /// # Arguments
    ///
    /// * `tx` - The current transaction (consumed)
    /// * `prefix` - The prefix being imported (needed to begin a new tx)
    /// * `order` - The n-gram order (needed for shard routing)
    ///
    /// # Returns
    ///
    /// The number of n-grams committed in this chunk. On success, `tx` is
    /// replaced with a fresh transaction for the next chunk. On error, `tx`
    /// is left in an indeterminate state (inner consumed) and should not be
    /// used further.
    pub fn commit_and_renew_prefix_tx(
        &self,
        tx: &mut StoragePrefixTx,
        prefix: &str,
        order: u8,
    ) -> StorageResult<usize> {
        match self {
            Self::Sharded {
                coordinator, stats, ..
            } => {
                let count = coordinator.commit_chunk_tx(&mut tx.inner)?;
                stats.record(count as u64, count as u64);

                // Begin a new transaction for the next chunk
                let shard_key = super::sharding::shard_key_for_file_prefix(
                    prefix,
                    order,
                    &coordinator.config().granularity,
                );
                let new_inner = coordinator.begin_prefix_tx(&shard_key, prefix)?;
                tx.inner = new_inner;
                Ok(count)
            }
            Self::SingleTrie { .. } => Err(StorageError::Config(
                "Cannot use chunked transactions with single-trie storage".to_string(),
            )),
        }
    }

    /// Abort a prefix transaction, discarding all buffered n-grams.
    ///
    /// Use this if an error occurs during processing and you want to
    /// discard the partial work without committing it.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to abort (consumed)
    pub fn abort_prefix_tx(&self, tx: StoragePrefixTx) -> StorageResult<()> {
        match self {
            Self::Sharded { coordinator, .. } => {
                coordinator.abort_prefix_tx(tx.inner)?;
                Ok(())
            }
            Self::SingleTrie { .. } => Err(StorageError::Config(
                "Cannot use transaction with single-trie storage".to_string(),
            )),
        }
    }

    // ========================================================================
    // File Transaction API (for single-trie mode with INCREMENT semantics)
    // ========================================================================

    /// Begin a file transaction for single-trie mode with INCREMENT semantics.
    ///
    /// Unlike `begin_prefix_tx()` which uses SET semantics (for sharded mode
    /// where each prefix file is complete), this uses INCREMENT semantics
    /// for cross-file count accumulation.
    ///
    /// **Only supported for single-trie storage.** Returns `Err` for sharded mode.
    ///
    /// # Key Properties
    ///
    /// - **Atomicity**: Either all n-gram increments are committed or none are
    /// - **Accumulation**: Counts add to existing values (not replace)
    /// - **Crash Safety**: Uncommitted transactions are discarded on WAL recovery
    ///
    /// # Arguments
    ///
    /// * `file_id` - Identifier for the file being imported (used as document ID)
    ///
    /// # Returns
    ///
    /// A `StorageFileTx` that must be passed to `tx_increment_tokens()` and
    /// eventually to `commit_file_tx()`.
    pub fn begin_file_tx(&self, file_id: &str) -> StorageResult<StorageFileTx> {
        match self {
            Self::SingleTrie {
                trie, vocabulary, ..
            } => {
                let tx = trie
                    .write()
                    .begin_document(file_id)
                    .map_err(|e| StorageError::Trie(format!("Failed to begin file tx: {}", e)))?;
                Ok(StorageFileTx {
                    inner: tx,
                    vocabulary: vocabulary.clone(),
                    ngram_count: 0,
                })
            }
            Self::Sharded { .. } => Err(StorageError::Config(
                "File transactions not supported in sharded mode; use begin_prefix_tx()".into(),
            )),
        }
    }

    /// Add an n-gram increment to a file transaction (tokens version).
    ///
    /// The n-gram is encoded using the vocabulary (if present) and buffered
    /// in memory. It will be applied atomically when the transaction is committed.
    /// Uses INCREMENT semantics (counts accumulate across files).
    ///
    /// # Arguments
    ///
    /// * `tx` - The active transaction from `begin_file_tx()`
    /// * `tokens` - The n-gram tokens (e.g., ["the", "quick"])
    /// * `count` - The count to add
    pub fn tx_increment_tokens(
        &self,
        tx: &mut StorageFileTx,
        tokens: &[&str],
        count: u64,
    ) -> StorageResult<()> {
        match self {
            Self::SingleTrie { trie, .. } => {
                // Encode tokens to raw byte key using vocabulary
                let encoded_key = tx.encode_tokens(tokens)?;

                // Buffer the increment in the transaction
                trie.read()
                    .tx_increment_bytes(&mut tx.inner, &encoded_key, count as i64);
                tx.ngram_count += 1;
                Ok(())
            }
            Self::Sharded { .. } => Err(StorageError::Config(
                "Cannot use file transaction with sharded storage".to_string(),
            )),
        }
    }

    /// Add an n-gram increment to a file transaction (raw ngram string version).
    ///
    /// # Arguments
    ///
    /// * `tx` - The active transaction from `begin_file_tx()`
    /// * `ngram` - The n-gram string (tokens joined with spaces or encoded)
    /// * `count` - The count to add
    pub fn tx_increment_ngram(
        &self,
        tx: &mut StorageFileTx,
        ngram: &str,
        count: u64,
    ) -> StorageResult<()> {
        match self {
            Self::SingleTrie { trie, .. } => {
                // Buffer the increment in the transaction
                trie.read()
                    .tx_increment_bytes(&mut tx.inner, ngram.as_bytes(), count as i64);
                tx.ngram_count += 1;
                Ok(())
            }
            Self::Sharded { .. } => Err(StorageError::Config(
                "Cannot use file transaction with sharded storage".to_string(),
            )),
        }
    }

    /// Commit a file transaction atomically.
    ///
    /// This writes all buffered increments to the WAL as a single batch
    /// and applies them to the trie atomically.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to commit (consumed)
    ///
    /// # Returns
    ///
    /// The number of n-gram operations that were committed.
    pub fn commit_file_tx(&self, tx: StorageFileTx) -> StorageResult<usize> {
        match self {
            Self::SingleTrie { trie, stats, .. } => {
                let ngram_count = tx.ngram_count;
                let committed = trie
                    .write()
                    .commit_document(tx.inner)
                    .map_err(|e| StorageError::Trie(format!("Failed to commit file tx: {}", e)))?;

                stats.record(ngram_count as u64, committed as u64);
                Ok(committed)
            }
            Self::Sharded { .. } => Err(StorageError::Config(
                "Cannot use file transaction with sharded storage".to_string(),
            )),
        }
    }

    /// Abort a file transaction, discarding all buffered increments.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to abort (consumed)
    pub fn abort_file_tx(&self, tx: StorageFileTx) -> StorageResult<()> {
        match self {
            Self::SingleTrie { trie, .. } => trie
                .read()
                .abort_document(tx.inner)
                .map_err(|e| StorageError::Trie(format!("Failed to abort file tx: {}", e))),
            Self::Sharded { .. } => Err(StorageError::Config(
                "Cannot use file transaction with sharded storage".to_string(),
            )),
        }
    }
}

/// A file transaction for atomic n-gram imports with INCREMENT semantics.
///
/// This is used for single-trie mode where n-gram counts need to accumulate
/// across multiple files. Unlike `StoragePrefixTx` which uses SET semantics
/// (for sharded mode), this uses INCREMENT semantics.
///
/// # Usage
///
/// ```ignore
/// let mut tx = storage.begin_file_tx("file_001.txt")?;
/// for (tokens, count) in ngrams {
///     storage.tx_increment_tokens(&mut tx, &tokens, count)?;
/// }
/// storage.commit_file_tx(tx)?;
/// ```
pub struct StorageFileTx {
    inner: libdictenstein::persistent_artrie::DocumentTransaction<u64>,
    vocabulary: Option<SharedConcurrentVocab>,
    ngram_count: usize,
}

impl StorageFileTx {
    /// Get the document ID (file identifier).
    pub fn file_id(&self) -> &str {
        self.inner.document_id()
    }

    /// Get the number of n-grams buffered so far.
    pub fn ngram_count(&self) -> usize {
        self.ngram_count
    }

    /// Check if the transaction is still active.
    pub fn is_active(&self) -> bool {
        self.inner.is_active()
    }

    /// Encode tokens to a raw byte key using the vocabulary.
    ///
    /// Vocabulary is required. Returns an error if no vocabulary is configured.
    fn encode_tokens(&self, tokens: &[&str]) -> StorageResult<Vec<u8>> {
        let vocab = self
            .vocabulary
            .as_ref()
            .ok_or_else(|| StorageError::Config("Vocabulary required for token encoding".into()))?;
        try_encode_ngram_key_lockfree_bytes(tokens, vocab).map_err(StorageError::from)
    }
}

/// A prefix transaction for atomic n-gram imports via NgramStorage.
///
/// This wraps the coordinator-level transaction and provides the same
/// atomicity, idempotency, and crash-safety guarantees.
///
/// # Usage
///
/// ```ignore
/// if let Some(mut tx) = storage.begin_prefix_tx("th", 2)? {
///     for (tokens, count) in ngrams {
///         storage.tx_insert_tokens(&mut tx, &tokens, count)?;
///     }
///     storage.commit_prefix_tx(tx)?;
/// }
/// ```
pub struct StoragePrefixTx {
    inner: super::sharding::CoordinatorPrefixTx,
}

impl StoragePrefixTx {
    /// Get the prefix being imported.
    pub fn prefix(&self) -> Option<&str> {
        self.inner.prefix()
    }

    /// Get the number of n-grams buffered so far.
    pub fn ngram_count(&self) -> usize {
        self.inner.ngram_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ngram::vocabulary::{
        decode_ngram_key_bytes, open_or_create_concurrent_vocabulary_lockfree,
        open_or_create_vocabulary,
    };
    use crate::sources::google_books::config::ShardingMode;
    use tempfile::TempDir;

    #[test]
    fn test_single_trie_storage() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("test.artrie");

        let storage = NgramStorage::create_single_trie(&path).expect("Failed to create storage");

        assert!(!storage.is_sharded());

        // Store some n-grams
        assert!(storage.store("the|quick", 10).expect("Failed to store"));
        assert!(!storage.store("the|quick", 5).expect("Failed to store"));

        // Query
        assert_eq!(storage.get("the|quick"), Some(15));
        assert!(storage.contains("the|quick"));
        assert!(!storage.contains("nonexistent"));

        // Stats
        assert_eq!(storage.stats().total_ngrams.load(Ordering::Relaxed), 15);
        assert_eq!(storage.stats().unique_ngrams.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_sharded_storage() {
        let dir = TempDir::new().expect("Failed to create temp dir");

        let config = GoogleBooksConfig {
            output_path: dir.path().join("output.artrie"),
            sharding: ShardingMode::Enabled(super::super::config::ShardingOptions::default()),
            ..Default::default()
        };

        let storage = NgramStorage::create_sharded(&config).expect("Failed to create storage");

        assert!(storage.is_sharded());

        // Store some n-grams
        assert!(storage.store("the|quick", 10).expect("Failed to store"));
        assert!(!storage.store("the|quick", 5).expect("Failed to store"));
        assert!(storage.store("apple|pie", 3).expect("Failed to store"));

        // Query
        assert_eq!(storage.get("the|quick"), Some(15));
        assert_eq!(storage.get("apple|pie"), Some(3));
        assert!(storage.contains("the|quick"));
        assert!(!storage.contains("nonexistent"));
    }

    #[test]
    fn test_batch_storage() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("test.artrie");

        let storage = NgramStorage::create_single_trie(&path).expect("Failed to create storage");

        let ngrams: Vec<(&[u8], u64)> = vec![
            (b"the|quick" as &[u8], 10u64),
            (b"the|slow" as &[u8], 5),
            (b"this|is" as &[u8], 3),
        ];

        let new_count = storage
            .store_batch(None, ngrams.into_iter())
            .expect("Failed to batch store");

        assert_eq!(new_count, 3);
        assert_eq!(storage.get("the|quick"), Some(10));
        assert_eq!(storage.get("the|slow"), Some(5));
        assert_eq!(storage.get("this|is"), Some(3));
    }

    #[test]
    fn test_vocabulary_encoding_single_trie() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let trie_path = dir.path().join("test.artrie");
        let vocab_path = dir.path().join("vocab.artrie");

        // Create lock-free concurrent vocabulary
        let vocabulary = open_or_create_concurrent_vocabulary_lockfree(&vocab_path)
            .expect("Failed to create vocabulary");

        // Create storage with vocabulary
        let storage =
            NgramStorage::create_single_trie_with_vocabulary(&trie_path, Some(vocabulary))
                .expect("Failed to create storage");

        assert!(storage.has_vocabulary());

        // Store tokens - should encode to varint Latin-1 strings
        let tokens1 = ["the", "quick"];
        let tokens2 = ["the", "slow"];
        let tokens3 = ["apple", "pie"];

        assert!(storage.store_tokens(&tokens1, 10).expect("Failed to store"));
        assert!(!storage.store_tokens(&tokens1, 5).expect("Failed to store")); // Duplicate
        assert!(storage.store_tokens(&tokens2, 3).expect("Failed to store"));
        assert!(storage.store_tokens(&tokens3, 7).expect("Failed to store"));

        // Verify encoded keys decode to correct number of indices
        let encoded1 = storage.encode_tokens(&tokens1).unwrap();
        let encoded3 = storage.encode_tokens(&tokens3).unwrap();

        assert_eq!(decode_ngram_key_bytes(&encoded1).len(), 2); // 2 word indices
        assert_eq!(decode_ngram_key_bytes(&encoded3).len(), 2); // 2 word indices

        // Query using tokens (routes through encode_tokens internally)
        assert_eq!(storage.get_tokens(&tokens1), Some(15)); // 10 + 5
        assert_eq!(storage.get_tokens(&tokens2), Some(3));
        assert_eq!(storage.get_tokens(&tokens3), Some(7));

        // Stats
        assert_eq!(storage.stats().total_ngrams.load(Ordering::Relaxed), 25);
        assert_eq!(storage.stats().unique_ngrams.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_vocabulary_encoding_sharded() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let vocab_path = dir.path().join("vocab.artrie");

        // Create lock-free concurrent vocabulary
        let vocabulary = open_or_create_concurrent_vocabulary_lockfree(&vocab_path)
            .expect("Failed to create vocabulary");

        // Create sharded storage with vocabulary using Adaptive granularity
        // to test prefix-based routing behavior
        let config = GoogleBooksConfig {
            output_path: dir.path().join("output.artrie"),
            sharding: ShardingMode::Enabled(super::super::config::ShardingOptions {
                granularity: super::super::config::ShardingGranularity::Adaptive,
                ..Default::default()
            }),
            ..Default::default()
        };

        let storage = NgramStorage::create_sharded_with_vocabulary(&config, Some(vocabulary))
            .expect("Failed to create storage");

        assert!(storage.is_sharded());
        assert!(storage.has_vocabulary());

        // Store tokens - should route based on original tokens, not encoded key
        let tokens_th = ["the", "quick", "brown"];
        let tokens_ap = ["apple", "pie"];
        let tokens_ze = ["zebra", "crossing"];

        assert!(storage
            .store_tokens(&tokens_th, 10)
            .expect("Failed to store"));
        assert!(storage
            .store_tokens(&tokens_ap, 5)
            .expect("Failed to store"));
        assert!(storage
            .store_tokens(&tokens_ze, 3)
            .expect("Failed to store"));

        // Verify routing is based on original tokens
        let shard_th = storage.route_tokens(&tokens_th).unwrap();
        let shard_ap = storage.route_tokens(&tokens_ap).unwrap();
        let shard_ze = storage.route_tokens(&tokens_ze).unwrap();

        // With Adaptive granularity (default), trigrams use 2-char prefixes
        assert_eq!(shard_th.prefix, "th");
        assert_eq!(shard_ap.prefix, "ap");
        assert_eq!(shard_ze.prefix, "ze");

        // Query using get_tokens (which routes based on original tokens)
        assert_eq!(storage.get_tokens(&tokens_th), Some(10));
        assert_eq!(storage.get_tokens(&tokens_ap), Some(5));
        assert_eq!(storage.get_tokens(&tokens_ze), Some(3));

        // Verify contains_tokens works
        assert!(storage.contains_tokens(&tokens_th));
        assert!(storage.contains_tokens(&tokens_ap));
        assert!(storage.contains_tokens(&tokens_ze));
        assert!(!storage.contains_tokens(&["nonexistent", "ngram"]));

        // Verify encoded keys decode to correct number of indices
        let encoded_th = storage.encode_tokens(&tokens_th).unwrap();
        assert_eq!(decode_ngram_key_bytes(&encoded_th).len(), 3); // 3 word indices for trigram
    }

    #[test]
    fn test_vocabulary_encoding_with_pipe_in_token() {
        // This tests the bug fix: tokens containing | should not corrupt data
        let dir = TempDir::new().expect("Failed to create temp dir");
        let trie_path = dir.path().join("test.artrie");
        let vocab_path = dir.path().join("vocab.artrie");

        let vocabulary = open_or_create_concurrent_vocabulary_lockfree(&vocab_path)
            .expect("Failed to create vocabulary");

        let storage =
            NgramStorage::create_single_trie_with_vocabulary(&trie_path, Some(vocabulary))
                .expect("Failed to create storage");

        // Token with pipe - this would corrupt with pipe-separated encoding
        let tokens = ["foo|bar", "baz"];

        assert!(storage.store_tokens(&tokens, 10).expect("Failed to store"));

        // Verify we can retrieve it
        let encoded = storage.encode_tokens(&tokens).unwrap();
        assert_eq!(storage.get_tokens(&tokens), Some(10));

        // The encoded key should decode to 2 indices, not affected by the | in the token
        assert_eq!(decode_ngram_key_bytes(&encoded).len(), 2);

        // Store different tokens - should NOT conflict
        let tokens2 = ["foo", "bar", "baz"]; // 3 separate tokens
        assert!(storage.store_tokens(&tokens2, 5).expect("Failed to store"));

        let encoded2 = storage.encode_tokens(&tokens2).unwrap();
        assert_eq!(decode_ngram_key_bytes(&encoded2).len(), 3); // 3 word indices
        assert_ne!(encoded, encoded2); // Different keys
    }

    // ---- Chunked transactions ----

    /// Diagnostic: same volume of inserts but single commit (no renewal).
    /// If this passes but the renew variant fails, the bug is in the renew
    /// path. If both fail, the bug is in some other part of the pipeline.
    #[test]
    fn test_single_commit_at_150_entries() {
        let dir = TempDir::new().expect("tempdir");
        let vocab_path = dir.path().join("vocab.artrie");
        let vocabulary = open_or_create_concurrent_vocabulary_lockfree(&vocab_path).expect("vocab");

        let config = GoogleBooksConfig {
            output_path: dir.path().join("output.artrie"),
            sharding: ShardingMode::Enabled(super::super::config::ShardingOptions {
                granularity: super::super::config::ShardingGranularity::TwoChar,
                ..Default::default()
            }),
            ..Default::default()
        };
        let storage = NgramStorage::create_sharded_with_vocabulary(&config, Some(vocabulary))
            .expect("create_sharded_with_vocabulary");

        let mut tx = storage
            .begin_prefix_tx("th", 2)
            .expect("begin_prefix_tx")
            .expect("Some for prefix-based");

        for i in 0..150 {
            let ngram = format!("the w{:04}", i);
            storage
                .tx_insert_ngram(&mut tx, &ngram, 100 + i as u64)
                .expect("tx_insert_ngram");
        }

        let committed = storage.commit_prefix_tx(tx).expect("commit_prefix_tx");
        assert_eq!(committed, 150);

        let mut missing = Vec::new();
        for i in 0..150 {
            let suffix = format!("w{:04}", i);
            let tokens = ["the", suffix.as_str()];
            if storage.get_tokens(&tokens) != Some(100 + i as u64) {
                missing.push((i, format!("the {}", suffix), storage.get_tokens(&tokens)));
            }
        }
        assert!(
            missing.is_empty(),
            "single-commit variant: missing {} of 150: first: {:?}",
            missing.len(),
            &missing[..missing.len().min(3)]
        );
    }

    #[test]
    fn test_commit_and_renew_prefix_tx_continues_inserts() {
        // Multi-chunk workflow at the storage layer: chunked commits via
        // commit_and_renew_prefix_tx, then a final commit_prefix_tx.
        //
        // Uses TwoChar (prefix-based) granularity. Under hash-based
        // CpuProportional, `begin_prefix_tx` returns None and this code
        // path is bypassed at the storage layer (per the safety constraint
        // documented on NgramStorage::begin_prefix_tx).
        let dir = TempDir::new().expect("tempdir");
        let vocab_path = dir.path().join("vocab.artrie");
        let vocabulary = open_or_create_concurrent_vocabulary_lockfree(&vocab_path).expect("vocab");

        let config = GoogleBooksConfig {
            output_path: dir.path().join("output.artrie"),
            sharding: ShardingMode::Enabled(super::super::config::ShardingOptions {
                granularity: super::super::config::ShardingGranularity::TwoChar,
                ..Default::default()
            }),
            ..Default::default()
        };
        let storage = NgramStorage::create_sharded_with_vocabulary(&config, Some(vocabulary))
            .expect("create_sharded_with_vocabulary");

        // Begin a prefix tx for "th" (2-gram order). tx_insert_ngram splits
        // on ASCII space, so we use space-delimited strings — matching the
        // Google Books n-gram format the importer consumes.
        let mut tx = storage
            .begin_prefix_tx("th", 2)
            .expect("begin_prefix_tx")
            .expect("sharded mode with prefix-based granularity should return Some");

        // Insert 100 n-grams, then chunk-commit
        for i in 0..100 {
            let ngram = format!("the w{:04}", i);
            storage
                .tx_insert_ngram(&mut tx, &ngram, 100 + i as u64)
                .expect("tx_insert_ngram chunk 1");
        }
        let committed_chunk = storage
            .commit_and_renew_prefix_tx(&mut tx, "th", 2)
            .expect("commit_and_renew_prefix_tx");
        assert_eq!(
            committed_chunk, 100,
            "first chunk should commit 100 n-grams"
        );

        // Insert 50 more
        for i in 100..150 {
            let ngram = format!("the w{:04}", i);
            storage
                .tx_insert_ngram(&mut tx, &ngram, 100 + i as u64)
                .expect("tx_insert_ngram chunk 2");
        }

        // Final commit marks the prefix complete
        let committed_final = storage.commit_prefix_tx(tx).expect("commit_prefix_tx");
        assert_eq!(committed_final, 50, "final chunk should commit 50 n-grams");

        // All 150 n-grams should be queryable via get_tokens — the prefix-based
        // routing means both the chunked-tx target shard (hash of "th") and
        // the read-time routing (first 2 chars of "the") resolve to the same
        // ShardKey("th"), so the data is findable end-to-end.
        let mut missing = Vec::new();
        for i in 0..150 {
            let suffix = format!("w{:04}", i);
            let tokens = ["the", suffix.as_str()];
            if storage.get_tokens(&tokens) != Some(100 + i as u64) {
                missing.push((i, format!("the {}", suffix), storage.get_tokens(&tokens)));
            }
        }
        assert!(
            missing.is_empty(),
            "missing or wrong n-grams after multi-chunk commit ({} total): first few: {:?}",
            missing.len(),
            &missing[..missing.len().min(5)]
        );
    }

    #[test]
    fn test_begin_prefix_tx_returns_none_for_hash_based_sharding() {
        // Safety constraint: chunked transactions can't be used under
        // hash-based granularities because file prefix and first-token
        // hashes diverge (e.g., hash("th") ≠ hash("the")). begin_prefix_tx
        // must return None for these, forcing the caller to fall back to
        // per-record store_ngram which routes correctly per-token.
        //
        // Without this guard, the importer's chunked-tx path would bind
        // to one shard per file but the data would belong on many — and
        // get_tokens reads would later miss the data entirely.
        let dir = TempDir::new().expect("tempdir");
        let vocab_path = dir.path().join("vocab.artrie");
        let vocabulary = open_or_create_concurrent_vocabulary_lockfree(&vocab_path).expect("vocab");

        let config = GoogleBooksConfig {
            output_path: dir.path().join("output.artrie"),
            sharding: ShardingMode::Enabled(super::super::config::ShardingOptions {
                granularity: super::super::config::ShardingGranularity::CpuProportional {
                    multiplier: 2,
                    minimum: 8,
                },
                ..Default::default()
            }),
            ..Default::default()
        };
        let storage = NgramStorage::create_sharded_with_vocabulary(&config, Some(vocabulary))
            .expect("create_sharded_with_vocabulary");

        let result = storage
            .begin_prefix_tx("th", 2)
            .expect("begin_prefix_tx should not error on hash-based granularity");
        assert!(
            result.is_none(),
            "hash-based granularity must return None from begin_prefix_tx to \
             prevent the file-prefix-vs-first-token routing divergence bug"
        );

        // The per-record path (store_tokens) still works correctly: it routes
        // by first-token hash both at write time and at read time.
        storage
            .store_tokens(&["the", "quick"], 42)
            .expect("store_tokens (per-record fallback)");
        assert_eq!(storage.get_tokens(&["the", "quick"]), Some(42));
    }

    #[test]
    fn test_commit_and_renew_prefix_tx_single_trie_errors() {
        // Single-trie mode doesn't support chunked transactions — should
        // surface a clear configuration error.
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("test.artrie");
        let storage = NgramStorage::create_single_trie(&path).expect("create_single_trie");

        // begin_prefix_tx itself returns None for single-trie mode, which is
        // the upstream invariant that prevents
        // anyone from constructing a StoragePrefixTx against single-trie
        // storage in the first place.
        let result = storage
            .begin_prefix_tx("th", 2)
            .expect("begin_prefix_tx should not error for single-trie");
        assert!(
            result.is_none(),
            "single-trie mode must return None from begin_prefix_tx — \
             chunked transactions require sharded storage"
        );
    }

    #[test]
    fn test_merge_and_rotate_vocabulary_wal_idempotent() {
        // Calling merge_and_rotate_vocabulary_wal twice in succession on a
        // populated vocabulary should not error or lose entries. This is the
        // core durability contract — periodic checkpoints invoke this method
        // and it must be safe to call repeatedly.
        let dir = TempDir::new().expect("tempdir");
        let vocab_path = dir.path().join("vocab.artrie");
        let vocabulary = open_or_create_concurrent_vocabulary_lockfree(&vocab_path).expect("vocab");

        let config = GoogleBooksConfig {
            output_path: dir.path().join("output.artrie"),
            sharding: ShardingMode::Enabled(super::super::config::ShardingOptions::default()),
            ..Default::default()
        };
        let storage = NgramStorage::create_sharded_with_vocabulary(&config, Some(vocabulary))
            .expect("create_sharded_with_vocabulary");

        // Populate the vocabulary by storing a few token-based n-grams
        storage
            .store_tokens(&["hello", "world"], 1)
            .expect("store_tokens 1");
        storage
            .store_tokens(&["foo", "bar"], 2)
            .expect("store_tokens 2");
        storage
            .store_tokens(&["baz", "qux"], 3)
            .expect("store_tokens 3");

        // Two back-to-back calls should both succeed
        storage
            .merge_and_rotate_vocabulary_wal()
            .expect("first merge_and_rotate_vocabulary_wal");
        storage
            .merge_and_rotate_vocabulary_wal()
            .expect("second merge_and_rotate_vocabulary_wal (idempotent)");

        // Tokens still resolvable after the double-rotate
        assert_eq!(storage.get_tokens(&["hello", "world"]), Some(1));
        assert_eq!(storage.get_tokens(&["foo", "bar"]), Some(2));
        assert_eq!(storage.get_tokens(&["baz", "qux"]), Some(3));
    }

    #[test]
    fn test_flush_lockfree_over_threshold_single_trie_flushes_unconditionally() {
        // Per the doc comment on NgramStorage::flush_lockfree_over_threshold:
        // single-trie mode does not track per-entry lock-free counts, so it
        // always flushes when the method is called (returning 1).
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("test.artrie");
        let storage = NgramStorage::create_single_trie(&path).expect("create_single_trie");

        // Store a small number of entries (well under any threshold)
        for i in 0..5 {
            let ngram = format!("the|w{}", i);
            storage.store(&ngram, 1).expect("store");
        }

        // Even with a high threshold, single-trie mode unconditionally flushes
        let flushed = storage
            .flush_lockfree_over_threshold(10_000)
            .expect("flush_lockfree_over_threshold");
        assert_eq!(
            flushed, 1,
            "single-trie mode should always flush (returns 1) regardless of threshold"
        );
    }

    // ---- Checkpoint-resume regression (docs/debugging/checkpoint-resume-bug.md) ----

    #[test]
    fn test_checkpoint_resume_no_count_doubling() {
        // Regression test for the documented checkpoint-resume bug. The
        // original failure mode: the vocabulary was NOT checkpointed
        // alongside the n-gram trie/shards, so on reopen the vocab restarted
        // from a stale index point, causing orphaned n-grams and doubled
        // counts. The fix wires merge_and_rotate_vocabulary_wal into both
        // periodic and save checkpoints — this test guards that invariant.
        let dir = TempDir::new().expect("tempdir");
        let vocab_path = dir.path().join("vocab.artrie");

        // Known n-grams spanning multiple shards (different first-token
        // 2-char prefixes) so we exercise the cross-shard durability path.
        let ngrams: Vec<(Vec<&str>, u64)> = vec![
            (vec!["the", "quick"], 1000),
            (vec!["the", "brown"], 500),
            (vec!["the", "fox"], 250),
            (vec!["apple", "pie"], 100),
            (vec!["apple", "tart"], 50),
            (vec!["zebra", "crossing"], 25),
            (vec!["zebra", "stripes"], 12),
        ];

        // ---- Phase 1: import, checkpoint, drop ----
        {
            let vocab =
                open_or_create_concurrent_vocabulary_lockfree(&vocab_path).expect("vocab create");

            let config = GoogleBooksConfig {
                output_path: dir.path().join("english.artrie"),
                sharding: ShardingMode::Enabled(super::super::config::ShardingOptions {
                    granularity: super::super::config::ShardingGranularity::TwoChar,
                    ..Default::default()
                }),
                ..Default::default()
            };
            let storage = NgramStorage::create_sharded_with_vocabulary(&config, Some(vocab))
                .expect("create_sharded_with_vocabulary");

            for (tokens, count) in &ngrams {
                storage.store_tokens(tokens, *count).expect("store_tokens");
            }

            // Mirror the importer's periodic-checkpoint flow:
            // merge vocab + rotate WAL, then sync + checkpoint shards.
            storage
                .merge_and_rotate_vocabulary_wal()
                .expect("merge_and_rotate_vocabulary_wal");
            storage.sync().expect("storage sync");
            storage.checkpoint().expect("storage checkpoint");

            // storage drops here, releasing all file handles
        }

        // ---- Phase 2: reopen and verify ----
        {
            let vocab =
                open_or_create_concurrent_vocabulary_lockfree(&vocab_path).expect("vocab reopen");

            let config = GoogleBooksConfig {
                output_path: dir.path().join("english.artrie"),
                sharding: ShardingMode::Enabled(super::super::config::ShardingOptions {
                    granularity: super::super::config::ShardingGranularity::TwoChar,
                    ..Default::default()
                }),
                ..Default::default()
            };
            let storage = NgramStorage::create_sharded_with_vocabulary(&config, Some(vocab))
                .expect("reopen storage");

            // Every n-gram must be queryable with its EXACT original count.
            // Any count doubling, missing entry, or vocab-index drift would
            // fail this loop.
            for (tokens, expected_count) in &ngrams {
                let got = storage.get_tokens(tokens);
                let token_slice: Vec<&str> = tokens.iter().copied().collect();
                assert_eq!(
                    got,
                    Some(*expected_count),
                    "n-gram {:?} expected count {} but got {:?} after \
                     checkpoint+drop+reopen — indicates vocab or shard data \
                     was lost (the documented checkpoint-resume bug class)",
                    token_slice,
                    expected_count,
                    got
                );
            }
        }
    }

    #[test]
    fn test_vocabulary_indices_stable_across_reopen() {
        // Independently of count doubling, the vocab's term→index mapping
        // must be STABLE across reopen. Re-inserting a known term should
        // return the same index it had originally — not a fresh sequential
        // one. The original bug had the vocab restart numbering after a
        // crash, orphaning all n-grams that had been encoded with the
        // pre-crash indices.
        let dir = TempDir::new().expect("tempdir");
        let vocab_path = dir.path().join("vocab.artrie");

        let terms = ["the", "quick", "brown", "fox"];
        let original_indices: Vec<u64>;

        // Phase 1: insert + checkpoint
        {
            let vocab = open_or_create_vocabulary(&vocab_path).expect("vocab create");
            let mut indices = Vec::new();
            {
                let guard = vocab.write();
                for term in &terms {
                    indices.push(guard.insert(term).expect("test vocab insert"));
                }
            }
            original_indices = indices;

            vocab.write().checkpoint().expect("vocab checkpoint");
        }

        // Phase 2: reopen + re-insert
        let vocab = open_or_create_vocabulary(&vocab_path).expect("vocab reopen");
        let mut reopened_indices = Vec::new();
        {
            let guard = vocab.write();
            for term in &terms {
                reopened_indices.push(guard.insert(term).expect("test vocab insert"));
            }
        }

        assert_eq!(
            original_indices, reopened_indices,
            "vocabulary indices must be stable across reopen — the documented \
             checkpoint-resume bug arose when re-insertion assigned new \
             indices, orphaning previously-encoded n-grams"
        );
    }

    fn persistent_storage_bridge_sharded_config(dir: &TempDir) -> GoogleBooksConfig {
        GoogleBooksConfig {
            output_path: dir.path().join("bridge.artrie"),
            sharding: ShardingMode::Enabled(super::super::config::ShardingOptions {
                granularity: super::super::config::ShardingGranularity::TwoChar,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn persistent_storage_bridge_checkpoint(order: u8, prefix: &str) -> ImportCheckpoint {
        let mut checkpoint = ImportCheckpoint::new();
        checkpoint.complete_prefix(order, prefix);
        checkpoint.add_ngrams(order, 1);
        checkpoint
    }

    fn assert_persistent_storage_bridge_completed(
        checkpoint: &ImportCheckpoint,
        order: u8,
        prefix: &str,
    ) {
        assert!(
            !checkpoint.needs_prefix(order, prefix),
            "completed prefix should not need processing after checkpoint reopen"
        );
    }

    #[test]
    fn persistent_storage_bridge_single_trie_checkpoint_metadata_reopens() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("single.artrie");

        {
            let storage = NgramStorage::create_single_trie(&path).expect("create single storage");
            storage.store("the quick", 7).expect("store data");
            storage.checkpoint().expect("checkpoint data");

            let checkpoint = persistent_storage_bridge_checkpoint(2, "th");
            storage
                .save_import_checkpoint(&checkpoint)
                .expect("save durable checkpoint metadata");
        }

        let reopened = NgramStorage::create_single_trie(&path).expect("reopen single storage");
        assert_eq!(
            reopened.get("the quick"),
            Some(7),
            "single-trie data should recover with checkpoint metadata"
        );

        let loaded = reopened
            .load_import_checkpoint()
            .expect("load checkpoint metadata")
            .expect("checkpoint metadata should exist");
        assert_persistent_storage_bridge_completed(&loaded, 2, "th");
    }

    #[test]
    fn persistent_storage_bridge_sharded_checkpoint_metadata_reopens() {
        let dir = TempDir::new().expect("tempdir");
        let config = persistent_storage_bridge_sharded_config(&dir);

        {
            let storage = NgramStorage::create_sharded(&config).expect("create sharded storage");
            storage.store("the quick", 11).expect("store data");
            storage.sync().expect("sync sharded data");
            storage.checkpoint().expect("checkpoint sharded data");

            let checkpoint = persistent_storage_bridge_checkpoint(2, "th");
            storage
                .save_import_checkpoint(&checkpoint)
                .expect("save sharded checkpoint metadata");
        }

        let reopened =
            NgramStorage::resume_or_start(&config, 1_000_000).expect("reopen sharded storage");
        assert_eq!(
            reopened.get("the quick"),
            Some(11),
            "sharded data should recover with auxiliary checkpoint metadata"
        );

        let loaded = reopened
            .load_import_checkpoint()
            .expect("load sharded checkpoint metadata")
            .expect("checkpoint metadata should exist");
        assert_persistent_storage_bridge_completed(&loaded, 2, "th");
    }

    #[test]
    fn persistent_storage_bridge_completed_prefix_data_and_metadata_recover() {
        let dir = TempDir::new().expect("tempdir");
        let config = persistent_storage_bridge_sharded_config(&dir);
        let vocab_path = dir.path().join("vocab.artrie");

        {
            let vocabulary =
                open_or_create_concurrent_vocabulary_lockfree(&vocab_path).expect("vocab create");
            let storage = NgramStorage::create_sharded_with_vocabulary(&config, Some(vocabulary))
                .expect("create sharded storage with vocabulary");

            let mut tx = storage
                .begin_prefix_tx("th", 2)
                .expect("begin prefix tx")
                .expect("prefix-based sharding should support prefix tx");
            storage
                .tx_insert_ngram(&mut tx, "the quick", 13)
                .expect("insert prefix n-gram");
            assert_eq!(storage.commit_prefix_tx(tx).expect("commit prefix tx"), 1);

            storage
                .merge_and_rotate_vocabulary_wal()
                .expect("durable vocabulary evidence");
            storage.sync().expect("sync shard data");
            storage.checkpoint().expect("checkpoint shard data");

            let checkpoint = persistent_storage_bridge_checkpoint(2, "th");
            storage
                .save_import_checkpoint(&checkpoint)
                .expect("save checkpoint claim after data durability");
        }

        let vocabulary =
            open_or_create_concurrent_vocabulary_lockfree(&vocab_path).expect("vocab reopen");
        let reopened = NgramStorage::create_sharded_with_vocabulary(&config, Some(vocabulary))
            .expect("reopen sharded storage with vocabulary");

        assert_eq!(
            reopened.get_tokens(&["the", "quick"]),
            Some(13),
            "completed prefix data should recover through stable vocabulary keys"
        );
        assert!(
            reopened.is_prefix_completed("th"),
            "shard prefix completion should survive checkpoint/reopen"
        );

        let loaded = reopened
            .load_import_checkpoint()
            .expect("load checkpoint metadata")
            .expect("checkpoint metadata should exist");
        assert_persistent_storage_bridge_completed(&loaded, 2, "th");
    }

    #[test]
    fn persistent_storage_bridge_graceful_cancel_checkpoint_recoverable() {
        let dir = TempDir::new().expect("tempdir");
        let config = persistent_storage_bridge_sharded_config(&dir);

        {
            let storage = NgramStorage::create_sharded(&config).expect("create sharded storage");
            storage.store("cancel path", 5).expect("store data");
            storage.sync().expect("sync data before graceful cancel");
            storage.checkpoint().expect("checkpoint data before cancel");

            let mut checkpoint = persistent_storage_bridge_checkpoint(2, "ca");
            checkpoint.current_prefix = None;
            storage
                .save_import_checkpoint(&checkpoint)
                .expect("save graceful-cancel checkpoint");
        }

        let reopened =
            NgramStorage::resume_or_start(&config, 1_000_000).expect("reopen sharded storage");
        assert_eq!(
            reopened.get("cancel path"),
            Some(5),
            "graceful cancel checkpoint should recover drained data"
        );

        let loaded = reopened
            .load_import_checkpoint()
            .expect("load graceful-cancel checkpoint")
            .expect("checkpoint metadata should exist");
        assert_persistent_storage_bridge_completed(&loaded, 2, "ca");
    }

    #[test]
    fn persistent_storage_bridge_force_quit_does_not_publish_new_claim() {
        let dir = TempDir::new().expect("tempdir");
        let config = persistent_storage_bridge_sharded_config(&dir);

        {
            let storage = NgramStorage::create_sharded(&config).expect("create sharded storage");
            let old_checkpoint = persistent_storage_bridge_checkpoint(2, "aa");
            storage
                .save_import_checkpoint(&old_checkpoint)
                .expect("save pre-existing checkpoint");

            storage.store("the quick", 17).expect("store later data");
            storage.sync().expect("sync later data");
            storage.checkpoint().expect("checkpoint later data");
        }

        let reopened =
            NgramStorage::resume_or_start(&config, 1_000_000).expect("reopen sharded storage");
        assert_eq!(
            reopened.get("the quick"),
            Some(17),
            "force-quit path may leave durable data without publishing a checkpoint claim"
        );

        let loaded = reopened
            .load_import_checkpoint()
            .expect("load checkpoint metadata")
            .expect("old checkpoint metadata should remain");
        assert_persistent_storage_bridge_completed(&loaded, 2, "aa");
        assert!(
            loaded.needs_prefix(2, "th"),
            "later force-quit data must not appear as a new checkpoint claim"
        );
    }

    #[test]
    fn persistent_storage_bridge_vocabulary_indices_stable_across_reopen() {
        let dir = TempDir::new().expect("tempdir");
        let vocab_path = dir.path().join("vocab.artrie");
        let terms = ["the", "quick", "bridge", "recover"];
        let first_indices: Vec<u64>;

        {
            let vocab = open_or_create_vocabulary(&vocab_path).expect("vocab create");
            let guard = vocab.write();
            first_indices = terms
                .iter()
                .map(|term| guard.insert(term).expect("insert vocab term"))
                .collect();
            guard.checkpoint().expect("checkpoint vocabulary");
        }

        let vocab = open_or_create_vocabulary(&vocab_path).expect("vocab reopen");
        let guard = vocab.write();
        let reopened_indices: Vec<u64> = terms
            .iter()
            .map(|term| guard.insert(term).expect("reinsert vocab term"))
            .collect();

        assert_eq!(
            first_indices, reopened_indices,
            "vocabulary term indices should remain stable after checkpoint/reopen"
        );
    }
}
