//! ShardCoordinator manages multiple sharded tries for parallel n-gram import.
//!
//! The coordinator handles:
//! - Shard lifecycle (create, open, close)
//! - N-gram routing to appropriate shards
//! - Writer acquisition/release for exclusive access
//! - Checkpoint coordination across all shards
//! - Query fanout for read operations

use super::checkpoint::{CheckpointManager, ImportPhase, ImportState};
use super::config::{ShardConfig, ShardGranularity};
use super::routing::{compute_shard_key, compute_shard_key_from_token, ngram_order, ShardKey};
use super::shard::{PrefixTransaction, ShardError, ShardHandle, ShardSyncHandle, ShardSyncState};

use dashmap::DashMap;
use lru::LruCache;
use parking_lot::{Mutex, RwLock};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
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

    /// Parallel sync failed for one or more shards.
    #[error("Parallel sync failed: {errors}")]
    ParallelSyncFailed {
        /// Error messages from failed shards.
        errors: String,
    },
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

    /// Per-shard creation locks to prevent TOCTOU races during shard creation.
    ///
    /// When multiple threads attempt to create the same shard simultaneously,
    /// this lock ensures only one thread performs the actual file creation
    /// while others wait and then use the already-created shard.
    ///
    /// This prevents the race condition where:
    /// 1. Thread A checks if file exists (false)
    /// 2. Thread B checks if file exists (false)
    /// 3. Both threads try to create the file, causing corruption
    creation_locks: DashMap<ShardKey, Arc<parking_lot::Mutex<()>>>,

    /// LRU cache for tracking shard access order (for eviction).
    /// Only used when max_open_shards > 0.
    lru_tracker: Option<Mutex<LruCache<ShardKey, ()>>>,

    /// Global checkpoint manager (optional).
    checkpoint_manager: Option<Mutex<CheckpointManager>>,

    /// Statistics.
    stats: Arc<CoordinatorStats>,

    /// Persistent rayon thread pool reused across `sync_all_parallel` /
    /// `coordinated_checkpoint_finish_with_progress` calls. Previously a
    /// fresh `ThreadPoolBuilder::new().build()` was created on every
    /// call — spawning N worker threads each time. Holding one pool for
    /// the coordinator's lifetime amortizes the thread-creation cost across
    /// the many periodic checkpoints that run during a long import.
    ///
    /// Wrapped in `OnceCell` (via `OnceLock`) so the pool is built lazily
    /// on first use — at which point `max_concurrent` is known. Pool size
    /// is fixed to the first caller's value; subsequent callers with a
    /// smaller `max_concurrent` still get correct behavior (rayon
    /// internally schedules at most `num_threads` parallel tasks).
    parallel_pool: std::sync::OnceLock<rayon::ThreadPool>,

    /// Shutdown flag.
    shutdown: std::sync::atomic::AtomicBool,
}

/// A prefix transaction held at the coordinator level.
///
/// This wraps a shard-level `PrefixTransaction` with the shard reference needed
/// to commit or abort the transaction. The transaction provides atomic,
/// idempotent import semantics for prefix files.
///
/// # Atomicity
///
/// Either all n-grams buffered in this transaction are committed to the shard,
/// or none are. There is no partial state.
///
/// # Idempotency
///
/// Uses SET semantics (not increment), so if the same prefix is re-imported
/// after a crash, the result is identical to a single import.
///
/// # Crash Safety
///
/// If the process crashes before `commit_prefix_tx()` is called, the buffered
/// n-grams are discarded during WAL recovery. Only committed transactions
/// survive crashes.
pub struct CoordinatorPrefixTx {
    /// The shard key this transaction belongs to.
    pub shard_key: ShardKey,

    /// Reference to the shard (needed for commit/abort).
    shard: Arc<RwLock<ShardHandle>>,

    /// The inner shard-level transaction.
    /// Option because it's taken during commit/abort.
    inner: Option<PrefixTransaction<u64>>,
}

impl CoordinatorPrefixTx {
    /// Get the prefix being imported.
    pub fn prefix(&self) -> Option<&str> {
        self.inner.as_ref().map(|tx| tx.prefix.as_str())
    }

    /// Get the number of n-grams buffered so far.
    pub fn ngram_count(&self) -> usize {
        self.inner.as_ref().map(|tx| tx.ngram_count).unwrap_or(0)
    }
}

/// Handle for tracking completion of a coordinated async checkpoint.
///
/// This holds handles for all shards that were syncing when the checkpoint
/// was initiated. The checkpoint is durable once all handles report synced.
///
/// # Non-blocking Checkpoint Pattern
///
/// ```ignore
/// // 1. Start async checkpoint (fast - O(1) rotation per shard)
/// let handle = coordinator.coordinated_checkpoint_async()?;
///
/// // 2. Workers continue immediately on new WAL segments
/// // 3. Background threads sync the old segments
///
/// // 4. When durability is needed (e.g., before reporting completion)
/// handle.wait_all()?;
/// ```
///
/// # Performance
///
/// With 100 shards at 50ms fsync each:
/// - **Blocking checkpoint**: ~5000ms sequential or ~625ms parallel (8 threads)
/// - **Async checkpoint**: ~1-10ms rotation, workers continue immediately
///
/// The async pattern reduces checkpoint blocking by **40-50x**.
pub struct CheckpointHandle {
    /// Handles for each shard's async sync.
    handles: Vec<ShardSyncHandle>,
}

impl CheckpointHandle {
    /// Create an empty checkpoint handle (for single-trie mode).
    pub fn empty() -> Self {
        Self { handles: Vec::new() }
    }

    /// Check if all shards have completed sync (non-blocking).
    ///
    /// Returns `true` when all shards have their target LSNs durable on disk.
    pub fn all_synced(&self) -> bool {
        self.handles.iter().all(|h| h.is_synced())
    }

    /// Wait for all shards to complete sync (blocking).
    ///
    /// Waits for each shard's sync to complete in order. After this returns,
    /// all data written before `coordinated_checkpoint_async()` is durable.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any shard's sync fails. The error includes which
    /// shard failed.
    pub fn wait_all(self) -> CoordinatorResult<()> {
        for handle in self.handles {
            handle.wait()?;
        }
        Ok(())
    }

    /// Wait for all shards using parallel waiting (blocking but faster).
    ///
    /// Uses rayon to wait on all shard handles concurrently rather than
    /// sequentially.
    pub fn wait_all_parallel(self) -> CoordinatorResult<()> {
        use rayon::prelude::*;
        let errors: Vec<_> = self.handles
            .into_par_iter()
            .filter_map(|h| h.wait().err())
            .collect();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(CoordinatorError::Checkpoint(
                errors.into_iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; ")
            ))
        }
    }

    /// Get the number of shards being synced.
    pub fn count(&self) -> usize {
        self.handles.len()
    }

    /// Check if this handle has no shards (all were clean).
    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }

    /// Get the total target LSNs being synced (for debugging).
    pub fn total_target_lsns(&self) -> u64 {
        self.handles.iter().map(|h| h.target_lsn()).sum()
    }
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
            creation_locks: DashMap::new(),
            lru_tracker,
            checkpoint_manager,
            stats: Arc::new(CoordinatorStats::default()),
            parallel_pool: std::sync::OnceLock::new(),
            shutdown: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Get the persistent rayon thread pool, building it on first call.
    ///
    /// The pool is sized to `max_workers` and reused for every subsequent
    /// `sync_all_parallel` / `coordinated_checkpoint_finish_with_progress`
    /// call — avoiding the prior cost of building a fresh pool (spawning
    /// N worker threads) on each invocation. The first caller decides the
    /// pool size; later callers with a different `max_workers` share the
    /// existing pool (rayon caps concurrency at `num_threads` internally,
    /// so a smaller request is naturally bounded).
    fn get_or_build_parallel_pool(&self, max_workers: usize) -> CoordinatorResult<&rayon::ThreadPool> {
        // Fast path: pool already built.
        if let Some(pool) = self.parallel_pool.get() {
            return Ok(pool);
        }

        // Slow path: build and install. `set` may race with another caller —
        // the first to win wins, and we re-fetch via `get_or_init` semantics
        // by checking `get()` after `set` returns Err (meaning someone else
        // already initialized).
        let new_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(max_workers.max(1))
            .build()
            .map_err(|e| {
                CoordinatorError::Config(format!("Failed to build parallel thread pool: {}", e))
            })?;

        match self.parallel_pool.set(new_pool) {
            Ok(()) => Ok(self.parallel_pool.get().expect("just set above")),
            Err(_) => Ok(self
                .parallel_pool
                .get()
                .expect("OnceLock observed as Some after Err on set")),
        }
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
    /// Also reconciles global checkpoint state with actual shard state to detect
    /// any inconsistencies from interrupted imports.
    pub fn open_with_checkpoints(config: ShardConfig) -> CoordinatorResult<Self> {
        if !config.shard_dir.exists() {
            return Err(CoordinatorError::Config(format!(
                "Shard directory does not exist: {}",
                config.shard_dir.display()
            )));
        }

        let coordinator = Self::new_internal(config, true)?;

        // CRITICAL: Reconcile global checkpoint with actual shard state.
        // The global checkpoint may be stale if the process crashed after committing
        // shard data but before saving the global checkpoint, or vice versa.
        // Shard state (from WAL replay) is authoritative.
        let shard_files = coordinator.discover_shard_files()?;

        if !shard_files.is_empty() {
            log::info!(
                "Reconciling global checkpoint with {} shard files",
                shard_files.len()
            );

            let mut reconciled_prefixes = 0usize;

            for (key, _path) in &shard_files {
                // Opening the shard triggers WAL replay and loads checkpoint state
                let shard = coordinator.get_or_create_shard(key)?;
                let guard = shard.read();
                let state = guard.checkpoint_state();

                // Merge shard's completed_prefixes into global checkpoint
                // Shard state is authoritative (it's what's actually on disk after WAL replay)
                if let Some(ref manager) = coordinator.checkpoint_manager {
                    let mut mgr = manager.lock();
                    let ckpt = mgr.checkpoint_mut();
                    let record = ckpt.get_or_create_shard(key, guard.path());

                    // Add any prefixes that are in shard but not in global checkpoint
                    for prefix in &state.completed_prefixes {
                        if !record.completed_prefixes.contains(prefix) {
                            log::debug!(
                                "Reconciling: adding prefix '{}' to shard {} (found in shard WAL)",
                                prefix,
                                key
                            );
                            record.completed_prefixes.insert(prefix.clone());
                            reconciled_prefixes += 1;
                        }
                    }

                    // Update entry count and n-gram count from shard
                    record.entry_count = guard.len() as u64;
                    record.ngrams_processed = state.ngrams_processed;
                }
            }

            if reconciled_prefixes > 0 {
                log::info!(
                    "Reconciled {} prefixes from shard WALs into global checkpoint",
                    reconciled_prefixes
                );
            }

            // Save reconciled checkpoint
            if let Some(ref manager) = coordinator.checkpoint_manager {
                manager.lock().save()?;
            }
        }

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
    ///
    /// When the shard directory exists but the global checkpoint doesn't, this will
    /// attempt to recover from existing shard WAL files (see `recover_from_shards`).
    pub fn resume_or_start(config: ShardConfig) -> CoordinatorResult<Self> {
        let has_global_checkpoint =
            config.shard_dir.exists() && config.global_checkpoint_path().exists();

        if has_global_checkpoint {
            Self::open_with_checkpoints(config)
        } else if config.shard_dir.exists() {
            // Shard directory exists but no global checkpoint - recover from shard WALs
            Self::recover_from_shards(config)
        } else {
            Self::new_with_checkpoints(config)
        }
    }

    /// Recover coordinator state from existing shard files when global checkpoint is missing.
    ///
    /// This handles the case where an import was interrupted before the global checkpoint
    /// was saved. Shards may have WAL data that needs to be replayed and their checkpoint
    /// states aggregated into a new global checkpoint.
    fn recover_from_shards(config: ShardConfig) -> CoordinatorResult<Self> {
        log::info!(
            "No global checkpoint found, attempting recovery from shard WAL files in {}",
            config.shard_dir.display()
        );

        // Create coordinator with checkpoint management
        let coordinator = Self::new_internal(config, true)?;

        // Discover and open all existing shards to trigger WAL replay
        let shard_files = coordinator.discover_shard_files()?;

        if shard_files.is_empty() {
            log::info!("No existing shard files found, starting fresh import");
            return Ok(coordinator);
        }

        log::info!(
            "Found {} existing shard files, replaying WALs and recovering checkpoint state",
            shard_files.len()
        );

        let mut total_recovered_prefixes = 0usize;
        let mut total_ngrams_recovered = 0u64;

        for (key, _path) in &shard_files {
            // Opening the shard triggers WAL replay via open_with_recovery()
            // and loads checkpoint state via load_checkpoint_state()
            let shard = coordinator.get_or_create_shard(key)?;
            let guard = shard.read();

            let state = guard.checkpoint_state();
            let prefix_count = state.completed_prefixes.len();
            let ngram_count = state.ngrams_processed;

            if prefix_count > 0 || ngram_count > 0 {
                log::info!(
                    "  Shard {}: recovered {} completed prefixes, {} n-grams",
                    key,
                    prefix_count,
                    ngram_count
                );
            }

            total_recovered_prefixes += prefix_count;
            total_ngrams_recovered += ngram_count;

            // Update global checkpoint with shard state
            if let Some(ref manager) = coordinator.checkpoint_manager {
                let mut mgr = manager.lock();
                let ckpt = mgr.checkpoint_mut();

                let record = ckpt.get_or_create_shard(key, guard.path());
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

        // Save the recovered global checkpoint
        if let Some(ref manager) = coordinator.checkpoint_manager {
            let mut mgr = manager.lock();
            mgr.detect_recovery();
            mgr.save()?;
        }

        log::info!(
            "Recovery complete: {} prefixes, {} n-grams recovered from {} shards",
            total_recovered_prefixes,
            total_ngrams_recovered,
            shard_files.len()
        );

        // Update stats
        coordinator
            .stats
            .unique_ngrams
            .store(total_ngrams_recovered, Ordering::Relaxed);

        Ok(coordinator)
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
    ///
    /// Uses a per-shard mutex to serialize shard creation attempts. This prevents
    /// TOCTOU race conditions where multiple workers might both see the file
    /// doesn't exist and then both try to create it, leading to file corruption.
    ///
    /// The double-check pattern ensures that:
    /// 1. We first acquire the creation lock for this specific shard key
    /// 2. We re-check if another thread created the shard while we waited
    /// 3. Only if still needed, we perform the actual create/open operation
    fn create_or_open_shard(&self, key: &ShardKey) -> CoordinatorResult<Arc<RwLock<ShardHandle>>> {
        // Get or create a mutex for this specific shard key.
        // This serializes creation attempts for the same shard while allowing
        // parallel creation of different shards.
        let lock = self
            .creation_locks
            .entry(key.clone())
            .or_insert_with(|| Arc::new(parking_lot::Mutex::new(())))
            .clone();

        // Acquire the lock - blocks other threads trying to create the same
        // shard. parking_lot::Mutex doesn't poison on panic, so we never lose
        // an import to a poisoned creation lock (the prior `std::sync::Mutex`
        // would have aborted with `expect("shard creation lock poisoned")`).
        let _guard = lock.lock();

        // Double-check pattern: another thread may have created while we waited
        if let Some(shard) = self.shards.get(key) {
            self.touch_lru(key);
            return Ok(Arc::clone(&shard));
        }

        // Now safe to create/open - we hold the exclusive lock for this key
        self.maybe_evict_shard();
        let path = self.config.shard_path(&key.as_file_stem());

        let shard = ShardHandle::open_or_create(key.clone(), &path)?;
        let shard = Arc::new(RwLock::new(shard));

        // Insert into map - no race now since we hold the lock
        self.shards.insert(key.clone(), Arc::clone(&shard));

        // Update LRU
        self.touch_lru(key);
        self.stats.active_shards.fetch_add(1, Ordering::Relaxed);

        Ok(shard)
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
            let lru = lru.lock();
            if lru.len() >= lru.cap().get() {
                // Get LRU key
                if let Some((key, _)) = lru.peek_lru() {
                    let key = key.clone();
                    drop(lru); // Release lock before removing shard

                    // Remove from active shards (flushes to disk via Drop)
                    if let Some((_, shard)) = self.shards.remove(&key) {
                        // CRITICAL: Block on checkpoint to ensure data is persisted before eviction.
                        // Using try_write() could skip checkpoint if shard is busy, leading to data loss.
                        // Blocking here is safe because we've already removed from the map,
                        // so no new operations will start on this shard.
                        let mut guard = shard.write();
                        if let Err(e) = guard.checkpoint() {
                            log::error!(
                                "Failed to checkpoint shard {} during eviction: {}",
                                key,
                                e
                            );
                        }
                        drop(guard);
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

// Sub-modules each provide additional `impl ShardCoordinator { ... }` blocks
// grouped by concern (split out of this file's earlier god-module form).
mod routing;
mod discovery;
mod import_state;
mod sync;
mod transactions;

#[cfg(test)]
mod tests;
