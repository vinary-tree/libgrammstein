//! ShardCoordinator manages multiple sharded tries for parallel n-gram import.
//!
//! The coordinator handles:
//! - Shard lifecycle (create, open, close)
//! - N-gram routing to appropriate shards
//! - Writer acquisition/release for exclusive access
//! - Checkpoint coordination across all shards
//! - Query fanout for read operations

use super::checkpoint::{CheckpointManager, CheckpointResult, GlobalCheckpoint, ImportPhase, ImportState};
use super::config::{ShardConfig, ShardGranularity};
use super::routing::{compute_shard_key, ngram_order, ShardKey};
use super::shard::{ShardError, ShardHandle};

use dashmap::DashMap;
use lru::LruCache;
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;

/// Error type for coordinator operations.
#[derive(Error, Debug)]
pub enum CoordinatorError {
    /// Shard operation failed.
    #[error("Shard error: {0}")]
    Shard(#[from] ShardError),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Checkpoint error.
    #[error("Checkpoint error: {0}")]
    Checkpoint(String),

    /// Global checkpoint error.
    #[error("Global checkpoint error: {0}")]
    GlobalCheckpoint(#[from] super::checkpoint::CheckpointError),

    /// Writer acquisition timeout.
    #[error("Timeout acquiring writer for shard {shard_key}")]
    WriterTimeout {
        /// The shard key that timed out.
        shard_key: String,
    },

    /// Shard not found.
    #[error("Shard not found: {shard_key}")]
    ShardNotFound {
        /// The missing shard key.
        shard_key: String,
    },

    /// Recovery required but not performed.
    #[error("Recovery required: import was interrupted with in-progress shards")]
    RecoveryRequired,
}

/// Result type for coordinator operations.
pub type CoordinatorResult<T> = Result<T, CoordinatorError>;

/// Statistics for the coordinator.
#[derive(Debug, Default)]
pub struct CoordinatorStats {
    /// Total n-grams stored across all shards.
    pub total_ngrams: AtomicU64,

    /// Total unique n-grams.
    pub unique_ngrams: AtomicU64,

    /// Number of active shards.
    pub active_shards: AtomicUsize,

    /// Number of writer acquisitions.
    pub writer_acquisitions: AtomicU64,

    /// Cumulative writer wait time in microseconds.
    pub writer_wait_us: AtomicU64,

    /// Number of shard evictions (LRU).
    pub shard_evictions: AtomicU64,
}

impl CoordinatorStats {
    /// Record a writer acquisition with wait time.
    pub fn record_writer_acquisition(&self, wait_us: u64) {
        self.writer_acquisitions.fetch_add(1, Ordering::Relaxed);
        self.writer_wait_us.fetch_add(wait_us, Ordering::Relaxed);
    }

    /// Record n-grams stored.
    pub fn record_ngrams(&self, total: u64, unique: u64) {
        self.total_ngrams.fetch_add(total, Ordering::Relaxed);
        self.unique_ngrams.fetch_add(unique, Ordering::Relaxed);
    }
}

/// Manages multiple sharded tries for parallel n-gram import.
///
/// The coordinator provides:
/// - Automatic shard routing based on n-gram prefix
/// - Lazy shard creation/opening
/// - LRU eviction for memory management
/// - Writer token management for exclusive access
/// - Coordinated checkpointing with crash recovery
pub struct ShardCoordinator {
    /// Configuration.
    config: ShardConfig,

    /// Active shards (open and ready for I/O).
    /// Uses DashMap for concurrent access.
    shards: DashMap<ShardKey, Arc<RwLock<ShardHandle>>>,

    /// LRU cache for tracking shard access order (for eviction).
    /// Only used when max_open_shards > 0.
    lru_tracker: Option<Mutex<LruCache<ShardKey, ()>>>,

    /// Global checkpoint manager (optional).
    checkpoint_manager: Option<Mutex<CheckpointManager>>,

    /// Statistics.
    stats: Arc<CoordinatorStats>,

    /// Shutdown flag.
    shutdown: std::sync::atomic::AtomicBool,
}

impl ShardCoordinator {
    /// Create a new coordinator with the given configuration.
    ///
    /// This creates the shard directory if it doesn't exist.
    pub fn new(config: ShardConfig) -> CoordinatorResult<Self> {
        Self::new_internal(config, false)
    }

    /// Create a new coordinator with checkpoint management enabled.
    ///
    /// Use this when you want automatic checkpoint coordination and crash recovery.
    pub fn new_with_checkpoints(config: ShardConfig) -> CoordinatorResult<Self> {
        Self::new_internal(config, true)
    }

    /// Internal constructor.
    fn new_internal(config: ShardConfig, enable_checkpoints: bool) -> CoordinatorResult<Self> {
        // Create shard directory
        std::fs::create_dir_all(&config.shard_dir)?;

        let lru_tracker = if config.max_open_shards > 0 {
            Some(Mutex::new(LruCache::new(
                NonZeroUsize::new(config.max_open_shards)
                    .unwrap_or(NonZeroUsize::new(100).unwrap()),
            )))
        } else {
            None
        };

        let checkpoint_manager = if enable_checkpoints {
            let checkpoint_path = config.global_checkpoint_path();
            Some(Mutex::new(CheckpointManager::new(
                checkpoint_path,
                config.checkpoint_interval_ms,
            )?))
        } else {
            None
        };

        Ok(Self {
            config,
            shards: DashMap::new(),
            lru_tracker,
            checkpoint_manager,
            stats: Arc::new(CoordinatorStats::default()),
            shutdown: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Open an existing coordinator from a shard directory.
    ///
    /// Discovers existing shards from the directory but doesn't open them
    /// (they are opened lazily on first access).
    pub fn open(config: ShardConfig) -> CoordinatorResult<Self> {
        if !config.shard_dir.exists() {
            return Err(CoordinatorError::Config(format!(
                "Shard directory does not exist: {}",
                config.shard_dir.display()
            )));
        }

        Self::new(config)
    }

    /// Open an existing coordinator with checkpoint management.
    ///
    /// If checkpoint exists and shows interrupted import, marks for recovery.
    pub fn open_with_checkpoints(config: ShardConfig) -> CoordinatorResult<Self> {
        if !config.shard_dir.exists() {
            return Err(CoordinatorError::Config(format!(
                "Shard directory does not exist: {}",
                config.shard_dir.display()
            )));
        }

        let mut coordinator = Self::new_internal(config, true)?;

        // Detect if recovery is needed
        if let Some(ref manager) = coordinator.checkpoint_manager {
            let mut mgr = manager.lock();
            mgr.detect_recovery();
        }

        Ok(coordinator)
    }

    /// Resume or start a new import with checkpoint support.
    ///
    /// If a previous import was interrupted, this resumes from the last checkpoint.
    /// Otherwise, it starts a fresh import.
    pub fn resume_or_start(config: ShardConfig) -> CoordinatorResult<Self> {
        let exists = config.shard_dir.exists() && config.global_checkpoint_path().exists();

        if exists {
            Self::open_with_checkpoints(config)
        } else {
            Self::new_with_checkpoints(config)
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &ShardConfig {
        &self.config
    }

    /// Get statistics.
    pub fn stats(&self) -> &CoordinatorStats {
        &self.stats
    }

    /// Get the number of currently open shards.
    pub fn open_shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Get or create a shard for the given key.
    ///
    /// This method is thread-safe and handles concurrent requests for
    /// the same shard.
    pub fn get_or_create_shard(
        &self,
        key: &ShardKey,
    ) -> CoordinatorResult<Arc<RwLock<ShardHandle>>> {
        // Fast path: shard already exists
        if let Some(shard) = self.shards.get(key) {
            self.touch_lru(key);
            return Ok(Arc::clone(&shard));
        }

        // Slow path: need to create/open the shard
        self.create_or_open_shard(key)
    }

    /// Create or open a shard (internal, handles race conditions).
    fn create_or_open_shard(&self, key: &ShardKey) -> CoordinatorResult<Arc<RwLock<ShardHandle>>> {
        // Check LRU eviction first
        self.maybe_evict_shard();

        let path = self.config.shard_path(&key.as_file_stem());

        // Try to create/open the shard, handling race conditions.
        // Multiple threads may try to create the same shard simultaneously.
        // If path.exists() is false but create fails with "AlreadyExists",
        // another thread won the race - fall back to open.
        let shard = if path.exists() {
            ShardHandle::open(key.clone(), &path)?
        } else {
            match ShardHandle::create(key.clone(), &path) {
                Ok(shard) => shard,
                Err(e) => {
                    // Check if this is a race condition (another thread created it)
                    let error_msg = e.to_string();
                    if error_msg.contains("AlreadyExists") || error_msg.contains("already exists") {
                        // Race condition: another thread created it first, open instead
                        ShardHandle::open(key.clone(), &path)?
                    } else {
                        return Err(e.into());
                    }
                }
            }
        };

        let shard = Arc::new(RwLock::new(shard));

        // Insert into map (may race with another thread)
        let entry = self.shards.entry(key.clone()).or_insert(Arc::clone(&shard));

        // Update LRU
        self.touch_lru(key);
        self.stats.active_shards.fetch_add(1, Ordering::Relaxed);

        Ok(Arc::clone(&entry))
    }

    /// Update LRU tracker for a shard.
    fn touch_lru(&self, key: &ShardKey) {
        if let Some(ref lru) = self.lru_tracker {
            let mut lru = lru.lock();
            lru.get_or_insert(key.clone(), || ());
        }
    }

    /// Evict least recently used shard if at capacity.
    fn maybe_evict_shard(&self) {
        if let Some(ref lru) = self.lru_tracker {
            let mut lru = lru.lock();
            if lru.len() >= lru.cap().get() {
                // Get LRU key
                if let Some((key, _)) = lru.peek_lru() {
                    let key = key.clone();
                    drop(lru); // Release lock before removing shard

                    // Remove from active shards (flushes to disk via Drop)
                    if let Some((_, shard)) = self.shards.remove(&key) {
                        // Checkpoint before eviction
                        if let Some(mut guard) = shard.try_write() {
                            let _ = guard.checkpoint();
                        }
                        self.stats.shard_evictions.fetch_add(1, Ordering::Relaxed);
                        self.stats.active_shards.fetch_sub(1, Ordering::Relaxed);
                    }

                    // Re-acquire lock and pop
                    let mut lru = self.lru_tracker.as_ref().unwrap().lock();
                    lru.pop_lru();
                }
            }
        }
    }

    /// Compute the shard key for an n-gram.
    pub fn route_ngram(&self, ngram: &str) -> ShardKey {
        let order = ngram_order(ngram);
        compute_shard_key(ngram, order, &self.config.granularity)
    }

    /// Store an n-gram count.
    ///
    /// Routes the n-gram to the appropriate shard and increments its count.
    /// This method acquires a write lock on the shard, so it should be used
    /// for batch operations where possible.
    ///
    /// # Arguments
    ///
    /// * `ngram` - The n-gram string (pipe-separated tokens)
    /// * `count` - The count to add
    ///
    /// # Returns
    ///
    /// `true` if this was a new n-gram, `false` if it already existed.
    pub fn store_ngram(&self, ngram: &str, count: u64) -> CoordinatorResult<bool> {
        let key = self.route_ngram(ngram);
        let shard = self.get_or_create_shard(&key)?;

        // Acquire write lock (blocking)
        let mut guard = shard.write();

        // Need to acquire write token for the shard
        let start = Instant::now();
        let token = loop {
            if let Some(token) = guard.try_acquire_write(0) {
                break token;
            }
            // Spin with tiny yield (shouldn't happen often since we hold the RwLock)
            std::thread::yield_now();

            // Timeout after 1 second (something is wrong)
            if start.elapsed().as_secs() > 1 {
                return Err(CoordinatorError::WriterTimeout {
                    shard_key: key.to_string(),
                });
            }
        };

        let wait_us = start.elapsed().as_micros() as u64;
        self.stats.record_writer_acquisition(wait_us);

        let was_new = guard.increment(ngram, count, &token)?;
        guard.release_write(token);

        if was_new {
            self.stats.unique_ngrams.fetch_add(1, Ordering::Relaxed);
        }
        self.stats.total_ngrams.fetch_add(count, Ordering::Relaxed);

        Ok(was_new)
    }

    /// Store multiple n-grams to the same shard efficiently.
    ///
    /// All n-grams must route to the same shard. This is more efficient
    /// than calling `store_ngram` repeatedly because it only acquires
    /// the write lock once.
    ///
    /// # Arguments
    ///
    /// * `key` - The shard key (all n-grams must route to this shard)
    /// * `ngrams` - Iterator of (ngram, count) pairs
    ///
    /// # Returns
    ///
    /// Number of new (unique) n-grams stored.
    pub fn store_ngrams_batch<'a, I>(&self, key: &ShardKey, ngrams: I) -> CoordinatorResult<u64>
    where
        I: Iterator<Item = (&'a str, u64)>,
    {
        let shard = self.get_or_create_shard(key)?;
        let mut guard = shard.write();

        let start = Instant::now();
        let token = loop {
            if let Some(token) = guard.try_acquire_write(0) {
                break token;
            }
            std::thread::yield_now();
            if start.elapsed().as_secs() > 1 {
                return Err(CoordinatorError::WriterTimeout {
                    shard_key: key.to_string(),
                });
            }
        };

        let wait_us = start.elapsed().as_micros() as u64;
        self.stats.record_writer_acquisition(wait_us);

        let mut new_count = 0u64;
        let mut total_count = 0u64;

        for (ngram, count) in ngrams {
            if guard.increment(ngram, count, &token)? {
                new_count += 1;
            }
            total_count += count;
        }

        guard.release_write(token);

        self.stats.record_ngrams(total_count, new_count);

        Ok(new_count)
    }

    /// Get the count for an n-gram.
    pub fn get(&self, ngram: &str) -> Option<u64> {
        let key = self.route_ngram(ngram);

        if let Some(shard) = self.shards.get(&key) {
            let guard = shard.read();
            return guard.get(ngram);
        }

        // Shard not loaded - check if file exists
        let path = self.config.shard_path(&key.as_file_stem());
        if path.exists() {
            // Load shard and query
            if let Ok(shard) = self.get_or_create_shard(&key) {
                let guard = shard.read();
                return guard.get(ngram);
            }
        }

        None
    }

    /// Check if an n-gram exists.
    pub fn contains(&self, ngram: &str) -> bool {
        self.get(ngram).is_some()
    }

    /// Checkpoint all open shards.
    pub fn checkpoint_all(&self) -> CoordinatorResult<()> {
        let mut errors = Vec::new();

        for entry in self.shards.iter() {
            let key = entry.key().clone();
            let shard = entry.value();

            if let Some(mut guard) = shard.try_write() {
                if let Err(e) = guard.checkpoint() {
                    errors.push(format!("Shard {}: {}", key, e));
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(CoordinatorError::Checkpoint(errors.join("; ")))
        }
    }

    /// Sync all open shards (flush WAL).
    pub fn sync_all(&self) -> CoordinatorResult<()> {
        let mut errors = Vec::new();

        for entry in self.shards.iter() {
            let key = entry.key().clone();
            let shard = entry.value();

            if let Some(mut guard) = shard.try_write() {
                if let Err(e) = guard.sync() {
                    errors.push(format!("Shard {}: {}", key, e));
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(CoordinatorError::Checkpoint(errors.join("; ")))
        }
    }

    /// Close all shards (checkpoint and remove from memory).
    pub fn close_all(&self) -> CoordinatorResult<()> {
        self.shutdown
            .store(true, std::sync::atomic::Ordering::SeqCst);

        // Checkpoint all shards
        self.checkpoint_all()?;

        // Clear shards (Drop will handle cleanup)
        self.shards.clear();

        if let Some(ref lru) = self.lru_tracker {
            lru.lock().clear();
        }

        self.stats.active_shards.store(0, Ordering::Relaxed);

        Ok(())
    }

    /// Get total entry count across all open shards.
    pub fn total_entry_count(&self) -> u64 {
        self.shards
            .iter()
            .map(|entry| entry.value().read().len() as u64)
            .sum()
    }

    /// Iterate over all shard keys that have been opened.
    pub fn open_shard_keys(&self) -> Vec<ShardKey> {
        self.shards.iter().map(|e| e.key().clone()).collect()
    }

    /// Get all shard file paths in the shard directory.
    pub fn discover_shard_files(&self) -> CoordinatorResult<Vec<(ShardKey, std::path::PathBuf)>> {
        let mut shards = Vec::new();

        for entry in std::fs::read_dir(&self.config.shard_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|e| e.to_str()) == Some("artrie") {
                // Extract shard key from filename: shard_XX.artrie
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Some(prefix) = stem.strip_prefix("shard_") {
                        let key = ShardKey::new(prefix);
                        shards.push((key, path));
                    }
                }
            }
        }

        Ok(shards)
    }

    /// Mark a prefix as completed in its corresponding shard.
    pub fn complete_prefix(&self, file_prefix: &str, order: u8) -> CoordinatorResult<()> {
        let key = super::routing::shard_key_for_file_prefix(
            file_prefix,
            order,
            &self.config.granularity,
        );

        if let Some(shard) = self.shards.get(&key) {
            let mut guard = shard.write();
            guard.complete_prefix(file_prefix);
        }

        Ok(())
    }

    /// Get a summary of all shards for checkpoint purposes.
    pub fn shard_summaries(&self) -> HashMap<ShardKey, ShardSummary> {
        self.shards
            .iter()
            .map(|entry| {
                let key = entry.key().clone();
                let guard = entry.value().read();
                let summary = ShardSummary {
                    path: guard.path().to_path_buf(),
                    entry_count: guard.len() as u64,
                    completed_prefixes: guard
                        .checkpoint_state()
                        .completed_prefixes
                        .iter()
                        .cloned()
                        .collect(),
                    ngrams_processed: guard.checkpoint_state().ngrams_processed,
                };
                (key, summary)
            })
            .collect()
    }

    // ========== Checkpoint Coordination Methods ==========

    /// Check if recovery is needed from a previous interrupted import.
    pub fn needs_recovery(&self) -> bool {
        if let Some(ref manager) = self.checkpoint_manager {
            manager.lock().needs_recovery()
        } else {
            false
        }
    }

    /// Get the current import state.
    pub fn import_state(&self) -> Option<ImportState> {
        self.checkpoint_manager.as_ref().map(|mgr| {
            mgr.lock().checkpoint().import_state.clone()
        })
    }

    /// Start a new import (sets checkpoint state to InProgress).
    pub fn start_import(&self) -> CoordinatorResult<()> {
        if let Some(ref manager) = self.checkpoint_manager {
            let mut mgr = manager.lock();

            // Check if recovery is needed
            if mgr.needs_recovery() {
                return Err(CoordinatorError::RecoveryRequired);
            }

            mgr.checkpoint_mut().start_import();
            mgr.save()?;
        }
        Ok(())
    }

    /// Resume an interrupted import.
    pub fn resume_import(&self) -> CoordinatorResult<()> {
        if let Some(ref manager) = self.checkpoint_manager {
            let mut mgr = manager.lock();
            mgr.resume();
            mgr.save()?;
        }
        Ok(())
    }

    /// Set the current import phase.
    pub fn set_import_phase(&self, phase: ImportPhase) -> CoordinatorResult<()> {
        if let Some(ref manager) = self.checkpoint_manager {
            let mut mgr = manager.lock();
            mgr.checkpoint_mut().set_phase(phase);
            mgr.maybe_save()?;
        }
        Ok(())
    }

    /// Mark import as completed.
    pub fn complete_import(&self) -> CoordinatorResult<()> {
        if let Some(ref manager) = self.checkpoint_manager {
            let mut mgr = manager.lock();
            let total = self.stats.total_ngrams.load(Ordering::Relaxed);
            let unique = self.stats.unique_ngrams.load(Ordering::Relaxed);
            mgr.checkpoint_mut().complete_import(total, unique);
            mgr.save()?;
        }
        Ok(())
    }

    /// Mark import as failed.
    pub fn fail_import(&self, error: impl Into<String>) -> CoordinatorResult<()> {
        if let Some(ref manager) = self.checkpoint_manager {
            let mut mgr = manager.lock();
            mgr.checkpoint_mut().fail_import(error);
            mgr.save()?;
        }
        Ok(())
    }

    /// Get all completed prefixes across all shards.
    pub fn all_completed_prefixes(&self) -> HashSet<String> {
        if let Some(ref manager) = self.checkpoint_manager {
            manager.lock().checkpoint().all_completed_prefixes()
        } else {
            // Fall back to querying shards directly
            self.shards
                .iter()
                .flat_map(|e| {
                    e.value()
                        .read()
                        .checkpoint_state()
                        .completed_prefixes
                        .iter()
                        .cloned()
                        .collect::<Vec<_>>()
                })
                .collect()
        }
    }

    /// Check if a prefix has been completed.
    pub fn is_prefix_completed(&self, prefix: &str) -> bool {
        self.all_completed_prefixes().contains(prefix)
    }

    /// Mark a prefix as currently being processed.
    pub fn set_current_prefix(&self, shard_key: &ShardKey, prefix: Option<&str>) -> CoordinatorResult<()> {
        // Update shard's checkpoint state
        if let Some(shard) = self.shards.get(shard_key) {
            let mut guard = shard.write();
            guard.set_current_prefix(prefix);
        }

        // Update global checkpoint
        if let Some(ref manager) = self.checkpoint_manager {
            let mut mgr = manager.lock();
            mgr.checkpoint_mut().set_current_prefix(shard_key, prefix);
            mgr.maybe_save()?;
        }

        Ok(())
    }

    /// Mark a prefix as completed in a shard and update global checkpoint.
    pub fn mark_prefix_completed(&self, shard_key: &ShardKey, prefix: &str) -> CoordinatorResult<()> {
        // Update shard's checkpoint state
        if let Some(shard) = self.shards.get(shard_key) {
            let mut guard = shard.write();
            guard.complete_prefix(prefix);
        }

        // Update global checkpoint
        if let Some(ref manager) = self.checkpoint_manager {
            let mut mgr = manager.lock();
            mgr.checkpoint_mut().complete_prefix(shard_key, prefix);
            mgr.maybe_save()?;
        }

        Ok(())
    }

    /// Perform a coordinated checkpoint of all shards and the global state.
    ///
    /// This checkpoints each shard individually, then updates and saves the
    /// global checkpoint atomically.
    pub fn coordinated_checkpoint(&self) -> CoordinatorResult<()> {
        // First, checkpoint all open shards
        let mut errors = Vec::new();

        for entry in self.shards.iter() {
            let key = entry.key().clone();
            let shard = entry.value();

            if let Some(mut guard) = shard.try_write() {
                // Update shard checkpoint
                if let Err(e) = guard.checkpoint() {
                    errors.push(format!("Shard {}: {}", key, e));
                    continue;
                }

                // Update global checkpoint with shard state
                if let Some(ref manager) = self.checkpoint_manager {
                    let mut mgr = manager.lock();
                    let ckpt = mgr.checkpoint_mut();
                    let state = guard.checkpoint_state();

                    // Create or update shard record
                    let record = ckpt.get_or_create_shard(&key, guard.path());
                    record.entry_count = guard.len() as u64;
                    record.ngrams_processed = state.ngrams_processed;
                    record.completed_prefixes = state.completed_prefixes.clone();
                    record.current_prefix = state.current_prefix.clone();
                    record.last_checkpoint_time = std::time::SystemTime::now()
                        .duration_since(std::time::SystemTime::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                }
            }
        }

        // Save global checkpoint
        if let Some(ref manager) = self.checkpoint_manager {
            manager.lock().save()?;
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(CoordinatorError::Checkpoint(errors.join("; ")))
        }
    }

    /// Perform automatic checkpoint if enough time has passed.
    ///
    /// Returns `true` if a checkpoint was performed.
    pub fn maybe_checkpoint(&self) -> CoordinatorResult<bool> {
        if let Some(ref manager) = self.checkpoint_manager {
            let mut mgr = manager.lock();
            if mgr.maybe_save()? {
                drop(mgr);
                // Do a full coordinated checkpoint
                self.checkpoint_all()?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Set metadata in the global checkpoint.
    pub fn set_checkpoint_metadata(&self, key: impl Into<String>, value: impl Into<String>) -> CoordinatorResult<()> {
        if let Some(ref manager) = self.checkpoint_manager {
            let mut mgr = manager.lock();
            mgr.checkpoint_mut().set_metadata(key, value);
            mgr.maybe_save()?;
        }
        Ok(())
    }

    /// Get metadata from the global checkpoint.
    pub fn get_checkpoint_metadata(&self, key: &str) -> Option<String> {
        self.checkpoint_manager.as_ref().and_then(|mgr| {
            mgr.lock()
                .checkpoint()
                .get_metadata(key)
                .cloned()
        })
    }

    /// Get a summary of the checkpoint state for logging.
    pub fn checkpoint_summary(&self) -> Option<super::checkpoint::CheckpointSummary> {
        self.checkpoint_manager.as_ref().map(|mgr| {
            mgr.lock().checkpoint().summary()
        })
    }
}

impl Drop for ShardCoordinator {
    fn drop(&mut self) {
        // Best-effort checkpoint on drop
        let _ = self.close_all();
    }
}

/// Summary of a shard's state for checkpointing.
#[derive(Clone, Debug)]
pub struct ShardSummary {
    /// Path to the shard file.
    pub path: std::path::PathBuf,

    /// Number of entries in the shard.
    pub entry_count: u64,

    /// Completed prefixes.
    pub completed_prefixes: Vec<String>,

    /// N-grams processed through this shard.
    pub ngrams_processed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_coordinator_create_and_store() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let config = ShardConfig::new(dir.path().join("shards"));

        let coordinator = ShardCoordinator::new(config).expect("Failed to create coordinator");

        // Store some n-grams
        let was_new = coordinator
            .store_ngram("the|quick", 10)
            .expect("Failed to store");
        assert!(was_new);

        let was_new = coordinator
            .store_ngram("the|quick", 5)
            .expect("Failed to store");
        assert!(!was_new);

        let was_new = coordinator
            .store_ngram("apple|pie", 3)
            .expect("Failed to store");
        assert!(was_new);

        // Query
        assert_eq!(coordinator.get("the|quick"), Some(15));
        assert_eq!(coordinator.get("apple|pie"), Some(3));
        assert_eq!(coordinator.get("nonexistent"), None);

        // Stats
        assert_eq!(
            coordinator.stats().unique_ngrams.load(Ordering::Relaxed),
            2
        );
        assert_eq!(
            coordinator.stats().total_ngrams.load(Ordering::Relaxed),
            18
        );
    }

    #[test]
    fn test_coordinator_routing() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let config = ShardConfig::new(dir.path().join("shards"))
            .with_granularity(ShardGranularity::TwoChar);

        let coordinator = ShardCoordinator::new(config).expect("Failed to create coordinator");

        // Store n-grams that should go to different shards
        coordinator
            .store_ngram("the|quick", 1)
            .expect("Failed to store");
        coordinator
            .store_ngram("apple|pie", 1)
            .expect("Failed to store");
        coordinator
            .store_ngram("zebra|crossing", 1)
            .expect("Failed to store");

        // Should have 3 different shards open
        assert_eq!(coordinator.open_shard_count(), 3);

        // Verify routing
        assert_eq!(coordinator.route_ngram("the|quick").prefix, "th");
        assert_eq!(coordinator.route_ngram("apple|pie").prefix, "ap");
        assert_eq!(coordinator.route_ngram("zebra|crossing").prefix, "ze");
    }

    #[test]
    fn test_coordinator_batch_store() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        // Use TwoChar granularity for prefix-based routing tests
        let config = ShardConfig::new(dir.path().join("shards"))
            .with_granularity(ShardGranularity::TwoChar);

        let coordinator = ShardCoordinator::new(config).expect("Failed to create coordinator");

        let key = ShardKey::new("th");
        let ngrams = vec![("the|quick", 10u64), ("the|slow", 5), ("this|is", 3)];

        let new_count = coordinator
            .store_ngrams_batch(&key, ngrams.into_iter())
            .expect("Failed to batch store");

        assert_eq!(new_count, 3);
        assert_eq!(coordinator.get("the|quick"), Some(10));
        assert_eq!(coordinator.get("the|slow"), Some(5));
        assert_eq!(coordinator.get("this|is"), Some(3));
    }

    #[test]
    fn test_coordinator_checkpoint() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let config = ShardConfig::new(dir.path().join("shards"));

        {
            let coordinator = ShardCoordinator::new(config.clone()).expect("Failed to create");
            coordinator
                .store_ngram("the|quick", 10)
                .expect("Failed to store");
            coordinator.checkpoint_all().expect("Failed to checkpoint");
        }

        // Reopen and verify
        {
            let coordinator = ShardCoordinator::open(config).expect("Failed to open");
            // Need to explicitly open the shard since we didn't query it yet
            let key = ShardKey::new("th");
            let _ = coordinator.get_or_create_shard(&key);
            assert_eq!(coordinator.get("the|quick"), Some(10));
        }
    }

    #[test]
    fn test_coordinator_with_global_checkpoint() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        // Use TwoChar granularity for prefix-based routing tests
        let config = ShardConfig::new(dir.path().join("shards"))
            .with_granularity(ShardGranularity::TwoChar);

        {
            let coordinator = ShardCoordinator::new_with_checkpoints(config.clone())
                .expect("Failed to create");

            // Start import
            coordinator.start_import().expect("Failed to start import");
            // Verify import is in progress
            if let Some(state) = coordinator.import_state() {
                assert!(matches!(state, ImportState::InProgress { .. }));
            } else {
                panic!("Expected import state to be set");
            }

            // Store some data
            coordinator.store_ngram("the|quick", 10).expect("Failed to store");

            // Mark prefix as in-progress
            let key = ShardKey::new("th");
            coordinator.set_current_prefix(&key, Some("th")).expect("Failed to set prefix");

            // Set metadata
            coordinator.set_checkpoint_metadata("language", "en").expect("Failed to set metadata");

            // Perform coordinated checkpoint
            coordinator.coordinated_checkpoint().expect("Failed to checkpoint");

            // Mark prefix as completed
            coordinator.mark_prefix_completed(&key, "th").expect("Failed to complete prefix");

            // Complete import
            coordinator.complete_import().expect("Failed to complete import");
        }

        // Reopen and verify
        {
            let coordinator = ShardCoordinator::open_with_checkpoints(config)
                .expect("Failed to open");

            // Should not need recovery (import completed successfully)
            assert!(!coordinator.needs_recovery());

            // Check metadata persisted
            assert_eq!(
                coordinator.get_checkpoint_metadata("language"),
                Some("en".to_string())
            );

            // Check completed prefixes
            assert!(coordinator.is_prefix_completed("th"));
        }
    }

    #[test]
    fn test_coordinator_recovery_detection() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let config = ShardConfig::new(dir.path().join("shards"));

        // Simulate interrupted import
        {
            let coordinator = ShardCoordinator::new_with_checkpoints(config.clone())
                .expect("Failed to create");

            coordinator.start_import().expect("Failed to start import");
            coordinator.store_ngram("the|quick", 10).expect("Failed to store");

            let key = ShardKey::new("th");
            coordinator.set_current_prefix(&key, Some("th")).expect("Failed to set prefix");

            // Checkpoint but don't complete import - simulates crash
            coordinator.coordinated_checkpoint().expect("Failed to checkpoint");

            // Drop coordinator without completing import
        }

        // Reopen should detect recovery needed
        {
            let coordinator = ShardCoordinator::open_with_checkpoints(config.clone())
                .expect("Failed to open");

            assert!(coordinator.needs_recovery());

            // Starting a new import should fail
            assert!(coordinator.start_import().is_err());

            // Resume import
            coordinator.resume_import().expect("Failed to resume");
            assert!(!coordinator.needs_recovery());

            // Now we can complete
            coordinator.complete_import().expect("Failed to complete");
        }
    }
}
