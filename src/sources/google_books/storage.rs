//! N-gram storage backend abstraction for Google Books import.
//!
//! This module provides a unified interface for storing n-grams during import,
//! supporting both single-trie and sharded storage backends.
//!
//! # Backends
//!
//! - **SingleTrie**: Original behavior using a single `PersistentARTrieChar<u64>`
//!   protected by `Arc<RwLock>`. Simple but has write contention with multiple workers.
//!
//! - **Sharded**: Distributes n-grams across multiple tries based on prefix routing.
//!   Eliminates write contention for parallel imports.
//!
//! # Vocabulary-Indexed Encoding
//!
//! When a `SharedVocabulary` is provided, n-gram keys are encoded using varint-encoded
//! u64 indices stored as Latin-1 strings instead of pipe-separated tokens. This:
//! - Fixes the delimiter bug when tokens contain `|`
//! - Provides compact storage (1-2 bytes per word for common words)
//! - Supports unlimited vocabulary size

use super::config::GoogleBooksConfig;
use super::sharding::{ShardCoordinator, ShardKey};
use crate::ngram::vocabulary::{try_encode_ngram_key, SharedVocabulary, VocabularyError};
use libdictenstein::persistent_artrie_char::PersistentARTrieChar;
use parking_lot::RwLock;
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
    /// Single trie storage (original behavior).
    SingleTrie {
        /// The trie instance.
        trie: Arc<RwLock<PersistentARTrieChar<u64>>>,
        /// Optional shared vocabulary for PUA encoding.
        vocabulary: Option<Arc<SharedVocabulary>>,
        /// Storage statistics.
        stats: Arc<StorageStats>,
    },

    /// Sharded storage using prefix-based routing.
    Sharded {
        /// The shard coordinator.
        coordinator: ShardCoordinator,
        /// Optional shared vocabulary for PUA encoding.
        vocabulary: Option<Arc<SharedVocabulary>>,
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
    pub fn create_single_trie_with_vocabulary(
        output_path: &Path,
        vocabulary: Option<Arc<SharedVocabulary>>,
    ) -> StorageResult<Self> {
        let trie = if output_path.exists() {
            log::info!("Opening existing trie at {:?}", output_path);
            PersistentARTrieChar::open(output_path)
                .map_err(|e| StorageError::Trie(format!("Failed to open trie: {}", e)))?
        } else {
            log::info!("Creating new trie at {:?}", output_path);
            PersistentARTrieChar::create(output_path)
                .map_err(|e| StorageError::Trie(format!("Failed to create trie: {}", e)))?
        };

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
        vocabulary: Option<Arc<SharedVocabulary>>,
    ) -> StorageResult<Self> {
        let shard_config = config.to_shard_config();

        log::info!(
            "Creating sharded storage at {:?} with {:?} granularity",
            shard_config.shard_dir,
            shard_config.granularity
        );

        let coordinator = ShardCoordinator::new_with_checkpoints(shard_config)?;

        Ok(Self::Sharded {
            coordinator,
            vocabulary,
            stats: Arc::new(StorageStats::default()),
        })
    }

    /// Resume or start storage based on configuration.
    ///
    /// For sharded storage, this loads existing checkpoint state.
    pub fn resume_or_start(config: &GoogleBooksConfig, estimated_ngrams: u64) -> StorageResult<Self> {
        Self::resume_or_start_with_vocabulary(config, estimated_ngrams, None)
    }

    /// Resume or start storage with optional vocabulary.
    ///
    /// For sharded storage, this loads existing checkpoint state.
    pub fn resume_or_start_with_vocabulary(
        config: &GoogleBooksConfig,
        estimated_ngrams: u64,
        vocabulary: Option<Arc<SharedVocabulary>>,
    ) -> StorageResult<Self> {
        let use_sharding = config.should_use_sharding(estimated_ngrams);

        if use_sharding {
            let shard_config = config.to_shard_config();

            log::info!(
                "Resuming/starting sharded storage at {:?}",
                shard_config.shard_dir
            );

            let coordinator = ShardCoordinator::resume_or_start(shard_config)?;

            Ok(Self::Sharded {
                coordinator,
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
    pub fn vocabulary(&self) -> Option<&Arc<SharedVocabulary>> {
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

    /// Encode tokens to an n-gram key using vocabulary.
    ///
    /// Returns a Latin-1 encoded string of varint bytes if vocabulary is set.
    /// If vocabulary is not set, returns the tokens joined with spaces.
    pub fn encode_tokens(&self, tokens: &[&str]) -> StorageResult<String> {
        match self.vocabulary() {
            Some(vocab) => try_encode_ngram_key(tokens, vocab).map_err(StorageError::from),
            None => Ok(tokens.join(" ")),
        }
    }

    /// Store tokens as an n-gram with count, using vocabulary encoding if available.
    ///
    /// This is the recommended method for storing n-grams when you have tokens.
    /// If vocabulary is enabled, the tokens are encoded to a varint key and routed
    /// based on the original tokens (not the encoded key).
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
        let encoded_key = self.encode_tokens(tokens)?;

        match self {
            Self::SingleTrie { trie, stats, .. } => {
                let mut guard = trie.write();
                let is_new = guard.get(&encoded_key).is_none();
                guard.increment(&encoded_key, count as i64).map_err(|e| {
                    StorageError::Trie(format!("Failed to store n-gram: {}", e))
                })?;

                stats.record(count, if is_new { 1 } else { 0 });
                Ok(is_new)
            }
            Self::Sharded { coordinator, stats, .. } => {
                // Route based on original tokens, store encoded key
                let shard_key = coordinator.route_tokens(tokens);
                let is_new = coordinator.store_in_shard(&shard_key, &encoded_key, count)?;
                stats.record(count, if is_new { 1 } else { 0 });
                Ok(is_new)
            }
        }
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
    /// Returns `true` if this was a new n-gram.
    pub fn store(&self, ngram: &str, count: u64) -> StorageResult<bool> {
        match self {
            Self::SingleTrie { trie, stats, .. } => {
                let mut guard = trie.write();
                let is_new = guard.get(ngram).is_none();
                guard.increment(ngram, count as i64).map_err(|e| {
                    StorageError::Trie(format!("Failed to store n-gram: {}", e))
                })?;

                stats.record(count, if is_new { 1 } else { 0 });
                Ok(is_new)
            }
            Self::Sharded { coordinator, stats, .. } => {
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
        I: Iterator<Item = (&'a str, u64)>,
    {
        match self {
            Self::SingleTrie { trie, stats, .. } => {
                let mut guard = trie.write();
                let mut new_count = 0u64;
                let mut total_count = 0u64;

                for (ngram, count) in ngrams {
                    let is_new = guard.get(ngram).is_none();
                    guard.increment(ngram, count as i64).map_err(|e| {
                        StorageError::Trie(format!("Failed to store n-gram: {}", e))
                    })?;

                    if is_new {
                        new_count += 1;
                    }
                    total_count += count;
                }

                stats.record(total_count, new_count);
                Ok(new_count)
            }
            Self::Sharded { coordinator, stats, .. } => {
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
    /// For vocabulary-indexed encoding with sharded storage, use `get_tokens` instead
    /// to ensure correct routing based on original tokens.
    pub fn get(&self, ngram: &str) -> Option<u64> {
        match self {
            Self::SingleTrie { trie, .. } => {
                let guard = trie.read();
                guard.get(ngram).map(|v| *v as u64)
            }
            Self::Sharded { coordinator, .. } => coordinator.get(ngram),
        }
    }

    /// Get the count for an n-gram by tokens (for vocabulary-indexed encoding).
    ///
    /// This encodes the tokens to a varint key and routes based on the original tokens,
    /// ensuring correct shard routing for vocabulary-indexed storage.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The n-gram tokens (e.g., ["the", "quick", "brown"])
    pub fn get_tokens(&self, tokens: &[&str]) -> Option<u64> {
        let encoded_key = self.encode_tokens(tokens).ok()?;

        match self {
            Self::SingleTrie { trie, .. } => {
                let guard = trie.read();
                guard.get(&encoded_key).map(|v| *v as u64)
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
    pub fn checkpoint(&self) -> StorageResult<()> {
        match self {
            Self::SingleTrie { trie, .. } => {
                let mut guard = trie.write();
                guard.checkpoint().map_err(|e| {
                    StorageError::Trie(format!("Checkpoint failed: {}", e))
                })
            }
            Self::Sharded { coordinator, .. } => {
                coordinator.coordinated_checkpoint()?;
                Ok(())
            }
        }
    }

    /// Sync to disk (WAL flush).
    pub fn sync(&self) -> StorageResult<()> {
        match self {
            Self::SingleTrie { trie, .. } => {
                let mut guard = trie.write();
                guard.checkpoint().map_err(|e| {
                    StorageError::Trie(format!("Sync failed: {}", e))
                })
            }
            Self::Sharded { coordinator, .. } => {
                coordinator.sync_all()?;
                Ok(())
            }
        }
    }

    /// Sync the vocabulary WAL (if present).
    pub fn sync_vocabulary(&self) -> StorageResult<()> {
        let vocab = match self {
            Self::SingleTrie { vocabulary, .. } => vocabulary,
            Self::Sharded { vocabulary, .. } => vocabulary,
        };
        if let Some(v) = vocab {
            v.sync().map_err(|e| {
                StorageError::Trie(format!("Vocabulary sync failed: {}", e))
            })?;
        }
        Ok(())
    }

    /// Checkpoint the vocabulary (if present).
    pub fn checkpoint_vocabulary(&self) -> StorageResult<()> {
        let vocab = match self {
            Self::SingleTrie { vocabulary, .. } => vocabulary,
            Self::Sharded { vocabulary, .. } => vocabulary,
        };
        if let Some(v) = vocab {
            v.checkpoint().map_err(|e| {
                StorageError::Trie(format!("Vocabulary checkpoint failed: {}", e))
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
                let mut guard = trie.write();
                guard.checkpoint().map_err(|e| {
                    StorageError::Trie(format!("Sync failed: {}", e))
                })?;
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
                let mut guard = trie.write();
                guard.checkpoint().map_err(|e| {
                    StorageError::Trie(format!("Checkpoint failed: {}", e))
                })
            }
            Self::Sharded { coordinator, .. } => {
                coordinator.coordinated_checkpoint_parallel(max_concurrent_syncs)?;
                Ok(())
            }
        }
    }

    /// Close the storage (checkpoint and release resources).
    pub fn close(&self) -> StorageResult<()> {
        match self {
            Self::SingleTrie { trie, .. } => {
                let mut guard = trie.write();
                guard.checkpoint().map_err(|e| {
                    StorageError::Trie(format!("Close checkpoint failed: {}", e))
                })
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
                guard.len() as u64
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
    pub fn as_single_trie(&self) -> Option<&Arc<RwLock<PersistentARTrieChar<u64>>>> {
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
    /// eventually to `commit_prefix_tx()`, or `None` for single-trie mode.
    pub fn begin_prefix_tx(&self, prefix: &str, order: u8) -> StorageResult<Option<StoragePrefixTx>> {
        match self {
            Self::SingleTrie { .. } => {
                // Single-trie mode doesn't support transactions
                // Caller should use store() directly
                Ok(None)
            }
            Self::Sharded { coordinator, .. } => {
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
        // Encode tokens using vocabulary
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
            Self::Sharded { coordinator, stats, .. } => {
                let count = coordinator.commit_prefix_tx(tx.inner)?;
                stats.record(count as u64, count as u64);
                Ok(count)
            }
            Self::SingleTrie { .. } => Err(StorageError::Config(
                "Cannot use transaction with single-trie storage".to_string(),
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
    use crate::ngram::vocabulary::decode_ngram_key;
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

        let ngrams = vec![("the|quick", 10u64), ("the|slow", 5), ("this|is", 3)];

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

        // Create vocabulary
        let vocabulary = Arc::new(
            SharedVocabulary::open_or_create(&vocab_path).expect("Failed to create vocabulary"),
        );

        // Create storage with vocabulary
        let storage = NgramStorage::create_single_trie_with_vocabulary(&trie_path, Some(vocabulary))
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

        assert_eq!(decode_ngram_key(&encoded1).len(), 2); // 2 word indices
        assert_eq!(decode_ngram_key(&encoded3).len(), 2); // 2 word indices

        // Query using encoded keys
        assert_eq!(storage.get(&encoded1), Some(15)); // 10 + 5
        assert_eq!(
            storage.get(&storage.encode_tokens(&tokens2).unwrap()),
            Some(3)
        );
        assert_eq!(storage.get(&encoded3), Some(7));

        // Stats
        assert_eq!(storage.stats().total_ngrams.load(Ordering::Relaxed), 25);
        assert_eq!(storage.stats().unique_ngrams.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_vocabulary_encoding_sharded() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let vocab_path = dir.path().join("vocab.artrie");

        // Create vocabulary
        let vocabulary = Arc::new(
            SharedVocabulary::open_or_create(&vocab_path).expect("Failed to create vocabulary"),
        );

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

        assert!(storage.store_tokens(&tokens_th, 10).expect("Failed to store"));
        assert!(storage.store_tokens(&tokens_ap, 5).expect("Failed to store"));
        assert!(storage.store_tokens(&tokens_ze, 3).expect("Failed to store"));

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
        assert_eq!(decode_ngram_key(&encoded_th).len(), 3); // 3 word indices for trigram
    }

    #[test]
    fn test_vocabulary_encoding_with_pipe_in_token() {
        // This tests the bug fix: tokens containing | should not corrupt data
        let dir = TempDir::new().expect("Failed to create temp dir");
        let trie_path = dir.path().join("test.artrie");
        let vocab_path = dir.path().join("vocab.artrie");

        let vocabulary = Arc::new(
            SharedVocabulary::open_or_create(&vocab_path).expect("Failed to create vocabulary"),
        );

        let storage = NgramStorage::create_single_trie_with_vocabulary(&trie_path, Some(vocabulary))
            .expect("Failed to create storage");

        // Token with pipe - this would corrupt with pipe-separated encoding
        let tokens = ["foo|bar", "baz"];

        assert!(storage.store_tokens(&tokens, 10).expect("Failed to store"));

        // Verify we can retrieve it
        let encoded = storage.encode_tokens(&tokens).unwrap();
        assert_eq!(storage.get(&encoded), Some(10));

        // The encoded key should decode to 2 indices, not affected by the | in the token
        assert_eq!(decode_ngram_key(&encoded).len(), 2);

        // Store different tokens - should NOT conflict
        let tokens2 = ["foo", "bar", "baz"]; // 3 separate tokens
        assert!(storage.store_tokens(&tokens2, 5).expect("Failed to store"));

        let encoded2 = storage.encode_tokens(&tokens2).unwrap();
        assert_eq!(decode_ngram_key(&encoded2).len(), 3); // 3 word indices
        assert_ne!(encoded, encoded2); // Different keys
    }
}
