//! N-gram storage backend abstraction for Google Books import.
//!
//! This module provides a unified interface for storing n-grams during import,
//! supporting both single-trie and sharded storage backends.
//!
//! # Backends
//!
//! - **SingleTrie**: Original behavior using a single `DiskBackedCharTrieInner<u64>`
//!   protected by `Arc<RwLock>`. Simple but has write contention with multiple workers.
//!
//! - **Sharded**: Distributes n-grams across multiple tries based on prefix routing.
//!   Eliminates write contention for parallel imports.

use super::config::{GoogleBooksConfig, ShardingMode};
use super::sharding::{ShardCoordinator, ShardKey};
use liblevenshtein::dictionary::persistent_artrie_char::DiskBackedCharTrieInner;
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
pub enum NgramStorage {
    /// Single trie storage (original behavior).
    SingleTrie {
        /// The trie instance.
        trie: Arc<RwLock<DiskBackedCharTrieInner<u64>>>,
        /// Storage statistics.
        stats: Arc<StorageStats>,
    },

    /// Sharded storage using prefix-based routing.
    Sharded {
        /// The shard coordinator.
        coordinator: ShardCoordinator,
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
        let trie = if output_path.exists() {
            log::info!("Opening existing trie at {:?}", output_path);
            DiskBackedCharTrieInner::open(output_path)
                .map_err(|e| StorageError::Trie(format!("Failed to open trie: {}", e)))?
        } else {
            log::info!("Creating new trie at {:?}", output_path);
            DiskBackedCharTrieInner::create(output_path)
                .map_err(|e| StorageError::Trie(format!("Failed to create trie: {}", e)))?
        };

        Ok(Self::SingleTrie {
            trie: Arc::new(RwLock::new(trie)),
            stats: Arc::new(StorageStats::default()),
        })
    }

    /// Create sharded storage.
    pub fn create_sharded(config: &GoogleBooksConfig) -> StorageResult<Self> {
        let shard_config = config.to_shard_config();

        log::info!(
            "Creating sharded storage at {:?} with {:?} granularity",
            shard_config.shard_dir,
            shard_config.granularity
        );

        let coordinator = ShardCoordinator::new_with_checkpoints(shard_config)?;

        Ok(Self::Sharded {
            coordinator,
            stats: Arc::new(StorageStats::default()),
        })
    }

    /// Resume or start storage based on configuration.
    ///
    /// For sharded storage, this loads existing checkpoint state.
    pub fn resume_or_start(config: &GoogleBooksConfig, estimated_ngrams: u64) -> StorageResult<Self> {
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
                stats: Arc::new(StorageStats::default()),
            })
        } else {
            Self::create_single_trie(&config.output_path)
        }
    }

    /// Check if this is sharded storage.
    pub fn is_sharded(&self) -> bool {
        matches!(self, Self::Sharded { .. })
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
            Self::SingleTrie { trie, stats } => {
                let mut guard = trie.write();
                let is_new = guard.get(ngram).is_none();
                guard.increment(ngram, count as i64).map_err(|e| {
                    StorageError::Trie(format!("Failed to store n-gram: {}", e))
                })?;

                stats.record(count, if is_new { 1 } else { 0 });
                Ok(is_new)
            }
            Self::Sharded { coordinator, stats } => {
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
            Self::SingleTrie { trie, stats } => {
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
            Self::Sharded { coordinator, stats } => {
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
    pub fn get(&self, ngram: &str) -> Option<u64> {
        match self {
            Self::SingleTrie { trie, .. } => {
                let guard = trie.read();
                guard.get(ngram).map(|v| *v as u64)
            }
            Self::Sharded { coordinator, .. } => coordinator.get(ngram),
        }
    }

    /// Check if an n-gram exists.
    pub fn contains(&self, ngram: &str) -> bool {
        self.get(ngram).is_some()
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
                guard.sync().map_err(|e| {
                    StorageError::Trie(format!("Sync failed: {}", e))
                })
            }
            Self::Sharded { coordinator, .. } => {
                coordinator.sync_all()?;
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
                guard.len as u64
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
    pub fn as_single_trie(&self) -> Option<&Arc<RwLock<DiskBackedCharTrieInner<u64>>>> {
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
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
