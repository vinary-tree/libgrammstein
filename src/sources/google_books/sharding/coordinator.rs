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
    creation_locks: DashMap<ShardKey, Arc<std::sync::Mutex<()>>>,

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
            .or_insert_with(|| Arc::new(std::sync::Mutex::new(())))
            .clone();

        // Acquire the lock - blocks other threads trying to create the same shard
        let _guard = lock.lock().expect("shard creation lock poisoned");

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

    /// Compute the shard key for an n-gram.
    pub fn route_ngram(&self, ngram: &str) -> ShardKey {
        let order = ngram_order(ngram);
        compute_shard_key(ngram, order, &self.config.granularity)
    }

    /// Compute the shard key from tokens (for vocabulary-indexed encoding).
    ///
    /// This routes based on the first token before encoding to PUA characters.
    /// Use this when storing vocabulary-indexed n-grams where the key is a
    /// sequence of PUA characters rather than pipe-separated tokens.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The n-gram tokens (e.g., ["the", "quick", "brown"])
    ///
    /// # Returns
    ///
    /// A `ShardKey` identifying which shard should store this n-gram.
    pub fn route_tokens(&self, tokens: &[&str]) -> ShardKey {
        let first_token = tokens.first().map(|s| *s).unwrap_or("");
        let order = tokens.len() as u8;
        compute_shard_key_from_token(first_token, order, &self.config.granularity)
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
        self.store_in_shard(&key, ngram.as_bytes(), count)
    }

    /// Store an encoded n-gram key in a specific shard.
    ///
    /// Use this when you have already computed the shard key and the encoded
    /// n-gram key (e.g., for vocabulary-indexed encoding).
    ///
    /// # Arguments
    ///
    /// * `shard_key` - The shard key (from `route_tokens` or `route_ngram`)
    /// * `encoded_key` - The encoded n-gram key to store
    /// * `count` - The count to add
    ///
    /// # Returns
    ///
    /// `true` if this was a new n-gram, `false` if it already existed.
    pub fn store_in_shard(
        &self,
        shard_key: &ShardKey,
        encoded_key: &[u8],
        count: u64,
    ) -> CoordinatorResult<bool> {
        let shard = self.get_or_create_shard(shard_key)?;

        // Lock-free path: shared read guard + CAS increment
        let guard = shard.read();
        let was_new = guard.increment_lockfree(encoded_key, count)?;

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
        I: Iterator<Item = (&'a [u8], u64)>,
    {
        let shard = self.get_or_create_shard(key)?;

        // Lock-free path: shared read guard + CAS increments
        let guard = shard.read();

        let mut new_count = 0u64;
        let mut total_count = 0u64;

        for (ngram, count) in ngrams {
            if guard.increment_lockfree(ngram, count)? {
                new_count += 1;
            }
            total_count += count;
        }

        self.stats.record_ngrams(total_count, new_count);

        Ok(new_count)
    }

    /// Get the count for an n-gram (text-based routing, byte-keyed lookup).
    pub fn get(&self, ngram: &str) -> Option<u64> {
        let key = self.route_ngram(ngram);
        let ngram_bytes = ngram.as_bytes();

        if let Some(shard) = self.shards.get(&key) {
            let guard = shard.read();
            return guard.get(ngram_bytes);
        }

        // Shard not loaded - check if file exists
        let path = self.config.shard_path(&key.as_file_stem());
        if path.exists() {
            // Load shard and query
            if let Ok(shard) = self.get_or_create_shard(&key) {
                let guard = shard.read();
                return guard.get(ngram_bytes);
            }
        }

        None
    }

    /// Check if an n-gram exists.
    pub fn contains(&self, ngram: &str) -> bool {
        self.get(ngram).is_some()
    }

    /// Get the count for an encoded key in a specific shard.
    ///
    /// Use this when you have already computed the shard key and the encoded
    /// n-gram key (e.g., for vocabulary-indexed encoding).
    ///
    /// # Arguments
    ///
    /// * `shard_key` - The shard key (from `route_tokens` or `route_ngram`)
    /// * `encoded_key` - The encoded n-gram key to look up
    pub fn get_in_shard(&self, shard_key: &ShardKey, encoded_key: &[u8]) -> Option<u64> {
        if let Some(shard) = self.shards.get(shard_key) {
            let guard = shard.read();
            return guard.get(encoded_key);
        }

        // Shard not loaded - check if file exists
        let path = self.config.shard_path(&shard_key.as_file_stem());
        if path.exists() {
            // Load shard and query
            if let Ok(shard) = self.get_or_create_shard(shard_key) {
                let guard = shard.read();
                return guard.get(encoded_key);
            }
        }

        None
    }

    /// Checkpoint all open shards.
    /// Merge lock-free overlays into persistent tries for all open shards.
    ///
    /// Call this before operations that iterate shard data (merge, query, MKN)
    /// to ensure they see values written via the lock-free path.
    pub fn flush_all_lockfree(&self) -> CoordinatorResult<()> {
        let mut errors = Vec::new();

        for entry in self.shards.iter() {
            let key = entry.key().clone();
            let shard = entry.value();
            let mut guard = shard.write();
            if let Err(e) = guard.flush_lockfree() {
                errors.push(format!("Shard {}: {}", key, e));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(CoordinatorError::Checkpoint(errors.join("; ")))
        }
    }

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

            // Use blocking write() to ensure all shards are synced.
            // This prevents the bug where locked shards are silently skipped,
            // causing WAL replay to double n-gram counts on resume.
            let mut guard = shard.write();
            if let Err(e) = guard.sync() {
                errors.push(format!("Shard {}: {}", key, e));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(CoordinatorError::Checkpoint(errors.join("; ")))
        }
    }

    /// Check if a specific shard is currently syncing.
    ///
    /// Workers can use this to defer writes to non-syncing shards,
    /// implementing the "defer-and-continue" pattern.
    ///
    /// # Arguments
    ///
    /// * `key` - The shard key to check
    ///
    /// # Returns
    ///
    /// `true` if the shard is currently syncing, `false` otherwise.
    /// Returns `false` if the shard doesn't exist (not yet created).
    pub fn is_shard_syncing(&self, key: &ShardKey) -> bool {
        self.shards
            .get(key)
            .map(|s| s.read().is_syncing())
            .unwrap_or(false)
    }

    /// Wait for a shard to finish syncing (with timeout).
    ///
    /// Returns immediately if the shard is not syncing.
    ///
    /// # Arguments
    ///
    /// * `key` - The shard key to wait on
    /// * `timeout` - Maximum time to wait
    ///
    /// # Returns
    ///
    /// `Ok(())` if sync completed or shard wasn't syncing,
    /// `Err(ShardError::SyncTimeout)` if timeout elapsed.
    pub fn wait_if_syncing(&self, key: &ShardKey, timeout: Duration) -> CoordinatorResult<()> {
        if let Some(shard) = self.shards.get(key) {
            let guard = shard.read();
            if guard.is_syncing() {
                guard.wait_for_sync(timeout)?;
            }
        }
        Ok(())
    }

    /// Sync all dirty shards in parallel using rayon.
    ///
    /// This method enables non-blocking checkpoint operations:
    /// - Workers can continue on non-syncing shards
    /// - Only workers targeting a syncing shard need to defer
    ///
    /// # Arguments
    ///
    /// * `max_concurrent` - Maximum number of shards to sync concurrently.
    ///   Higher values increase I/O parallelism but may overwhelm slow disks.
    ///   Recommended: 4-8 for SSDs, 1-2 for HDDs.
    ///
    /// # Returns
    ///
    /// Number of shards synced, or error if any sync failed.
    ///
    /// # Thread Safety
    ///
    /// Uses `sync_tracked()` which employs CAS to ensure only one sync
    /// operation per shard at a time. Formally verified in TLA+ spec.
    pub fn sync_all_parallel(&self, max_concurrent: usize) -> CoordinatorResult<usize> {
        // Collect dirty shard keys (read locks only, very fast)
        let dirty_shards: Vec<ShardKey> = self
            .shards
            .iter()
            .filter(|e| e.value().read().sync_state() == ShardSyncState::Dirty)
            .map(|e| e.key().clone())
            .collect();

        if dirty_shards.is_empty() {
            return Ok(0);
        }

        log::debug!(
            "Parallel sync: {} dirty shards with max {} concurrent",
            dirty_shards.len(),
            max_concurrent
        );

        // Parallel sync using rayon with bounded concurrency
        // Use a thread pool with limited parallelism
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(max_concurrent.min(dirty_shards.len()))
            .build()
            .map_err(|e| CoordinatorError::Config(format!("Failed to create thread pool: {}", e)))?;

        let errors: Vec<(ShardKey, String)> = pool.install(|| {
            dirty_shards
                .par_iter()
                .filter_map(|key| {
                    let shard = self.shards.get(key)?;

                    // Try to start sync (CAS: Dirty -> Syncing)
                    // If this returns false, another thread started the sync
                    {
                        let guard = shard.read();
                        if !guard.sync_coordinator().try_start_sync() {
                            return None; // Already syncing or clean
                        }
                    }

                    // Perform the actual sync with write lock
                    let mut guard = shard.write();
                    match guard.sync() {
                        Ok(()) => {
                            // Success: mark clean
                            let lsn = guard.stats().write_count.load(Ordering::Relaxed);
                            guard.sync_coordinator().complete_sync(lsn);
                            None
                        }
                        Err(e) => {
                            // Failure: mark failed
                            guard.sync_coordinator().fail_sync(&e.to_string());
                            Some((key.clone(), e.to_string()))
                        }
                    }
                })
                .collect()
        });

        let synced_count = dirty_shards.len() - errors.len();

        if errors.is_empty() {
            log::debug!("Parallel sync completed: {} shards synced", synced_count);
            Ok(synced_count)
        } else {
            let error_messages: Vec<String> = errors
                .iter()
                .map(|(k, e)| format!("{}: {}", k, e))
                .collect();
            Err(CoordinatorError::ParallelSyncFailed {
                errors: error_messages.join("; "),
            })
        }
    }

    /// Start async sync on all dirty shards (non-blocking).
    ///
    /// This initiates WAL segment rotation on each dirty shard:
    /// - Each shard rotates its WAL to a new segment (O(1) operation)
    /// - New writes go to the new segment immediately
    /// - Background threads sync the old segments
    /// - Workers can continue without waiting for fsync
    ///
    /// # Returns
    ///
    /// A `CheckpointHandle` that can be used to:
    /// - Check if all syncs completed (`all_synced()`)
    /// - Wait for all syncs to complete (`wait_all()`)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Start async sync on all dirty shards
    /// let handle = coordinator.sync_all_async()?;
    ///
    /// // Workers continue immediately...
    /// process_more_data();
    ///
    /// // When durability is needed
    /// handle.wait_all()?;
    /// ```
    pub fn sync_all_async(&self) -> CoordinatorResult<CheckpointHandle> {
        let mut handles = Vec::new();

        for entry in self.shards.iter() {
            let shard = entry.value();
            let guard = shard.read();

            // Only sync shards that have dirty data
            if guard.is_dirty() {
                if let Some(handle) = guard.sync_async()? {
                    handles.push(handle);
                }
            }
        }

        log::debug!(
            "Async sync initiated: {} shards rotating WAL",
            handles.len()
        );

        Ok(CheckpointHandle { handles })
    }

    /// Check if any shard is currently syncing.
    ///
    /// This can be used by workers to decide whether to defer operations
    /// during a checkpoint.
    pub fn any_shard_syncing(&self) -> bool {
        self.shards
            .iter()
            .any(|entry| entry.value().read().is_syncing())
    }

    /// Coordinated async checkpoint for maximum throughput.
    ///
    /// This is the recommended checkpoint method for high-throughput workloads.
    /// It provides the same durability guarantees as `coordinated_checkpoint()`
    /// but with minimal blocking:
    ///
    /// 1. **Fast rotation (~1ms total)**: Rotate all shards to new WAL segments
    /// 2. **Non-blocking**: Return immediately - workers continue writing
    /// 3. **Background sync**: Old segments sync in background threads
    /// 4. **Explicit wait**: Call `handle.wait_all()` when durability is needed
    ///
    /// # Performance
    ///
    /// With 100 shards at 50ms fsync each:
    /// - `coordinated_checkpoint()`: ~5000ms blocking (sequential)
    /// - `coordinated_checkpoint_parallel(8)`: ~625ms blocking
    /// - `coordinated_checkpoint_async()`: ~1-10ms, then workers continue
    ///
    /// The async checkpoint provides **40-50x less blocking** than sequential,
    /// and **~60x less blocking** than parallel checkpoint.
    ///
    /// # Usage Pattern
    ///
    /// ```ignore
    /// // Periodic checkpoint during import
    /// let handle = coordinator.coordinated_checkpoint_async()?;
    ///
    /// // Workers continue immediately on new WAL segments
    /// // ... more imports ...
    ///
    /// // At end of batch or before reporting progress
    /// handle.wait_all()?;
    ///
    /// // Now safe to update checkpoint metadata
    /// coordinator.coordinated_checkpoint_finish(8)?;
    /// ```
    pub fn coordinated_checkpoint_async(&self) -> CoordinatorResult<CheckpointHandle> {
        let start = Instant::now();

        // Start async sync on all dirty shards
        let handle = self.sync_all_async()?;

        let elapsed = start.elapsed();
        log::debug!(
            "Async checkpoint rotation completed: {} shards in {:.2}ms",
            handle.count(),
            elapsed.as_secs_f64() * 1000.0
        );

        Ok(handle)
    }

    /// Finish a checkpoint after async sync is complete.
    ///
    /// Call this after `coordinated_checkpoint_async().wait_all()` to:
    /// 1. Truncate the WALs (fast - data already synced)
    /// 2. Update global checkpoint state
    /// 3. Save checkpoint metadata
    ///
    /// This is the "commit" phase of an async checkpoint.
    ///
    /// # Arguments
    ///
    /// * `max_concurrent` - Maximum shards to persist in parallel.
    ///   Recommended: 4-8 for NVMe SSDs, 2-4 for SATA SSDs, 1-2 for HDDs.
    ///
    /// # Parallelization
    ///
    /// Uses a bounded rayon thread pool to checkpoint shards in parallel.
    /// Since the expensive fsync was already done during the async phase,
    /// this is primarily truncating WALs and updating state, which is fast
    /// (~1ms per shard). Bounded parallelism prevents overwhelming disk I/O
    /// when many dirty shards need to persist simultaneously.
    pub fn coordinated_checkpoint_finish(&self, max_concurrent: usize) -> CoordinatorResult<()> {
        self.coordinated_checkpoint_finish_with_progress(max_concurrent, None::<fn(usize, usize)>)
    }

    /// Finish a checkpoint with progress reporting.
    ///
    /// Same as `coordinated_checkpoint_finish()` but emits `CheckpointProgress` events
    /// through the provided channel.
    ///
    /// # Arguments
    ///
    /// * `max_concurrent` - Maximum shards to persist in parallel.
    /// * `progress_callback` - Optional callback invoked after each shard completes.
    ///   Receives (shards_processed, total_shards).
    pub fn coordinated_checkpoint_finish_with_progress<F>(
        &self,
        max_concurrent: usize,
        progress_callback: Option<F>,
    ) -> CoordinatorResult<()>
    where
        F: Fn(usize, usize) + Send + Sync,
    {
        let start = Instant::now();

        // Collect shard keys to process in parallel
        let shard_keys: Vec<ShardKey> = self.shards.iter().map(|e| e.key().clone()).collect();
        let total_shards = shard_keys.len();

        // Create bounded thread pool for I/O parallelism
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(max_concurrent.min(shard_keys.len()).max(1))
            .build()
            .map_err(|e| CoordinatorError::Config(format!("Failed to create thread pool: {}", e)))?;

        // Atomic counter for progress tracking
        let shards_processed = AtomicUsize::new(0);

        // Parallel checkpoint (truncate WALs) - collect state updates
        // OPTIMIZATION: Skip shards that are already clean (no dirty data) to avoid
        // expensive persist_to_disk() calls. Clean shards still contribute their state
        // to the checkpoint metadata.
        let results: Vec<Result<(ShardKey, u64, u64, HashSet<String>, Option<String>), String>> =
            pool.install(|| {
                shard_keys
                    .par_iter()
                    .filter_map(|key| {
                    let shard = self.shards.get(key)?;

                    // Check if shard is clean with a read lock first (fast, non-blocking)
                    let result = {
                        let guard = shard.read();
                        if !guard.is_dirty() {
                            // Shard is clean - still collect its state for checkpoint metadata
                            let state = guard.checkpoint_state();
                            Some(Ok((
                                key.clone(),
                                guard.len() as u64,
                                state.ngrams_processed,
                                state.completed_prefixes.clone(),
                                state.current_prefix.clone(),
                            )))
                        } else {
                            None
                        }
                    };

                    let result = result.unwrap_or_else(|| {
                        // Shard is dirty - need to checkpoint (requires write lock)
                        let mut guard = shard.write();

                        // Checkpoint (truncate WAL) - fast since data already synced
                        if let Err(e) = guard.checkpoint() {
                            return Err(format!("Shard {}: {}", key, e));
                        }

                        // Collect state for global checkpoint update (done outside parallel section
                        // to avoid lock contention on checkpoint_manager)
                        let state = guard.checkpoint_state();
                        Ok((
                            key.clone(),
                            guard.len() as u64,
                            state.ngrams_processed,
                            state.completed_prefixes.clone(),
                            state.current_prefix.clone(),
                        ))
                    });

                    // Update progress counter and emit event
                    let processed = shards_processed.fetch_add(1, Ordering::Relaxed) + 1;
                    if let Some(ref callback) = progress_callback {
                        callback(processed, total_shards);
                    }

                    Some(result)
                })
                .collect()
            });

        // Separate errors from successful results
        let mut errors = Vec::new();
        let mut shard_states = Vec::new();

        for result in results {
            match result {
                Ok(state) => shard_states.push(state),
                Err(e) => errors.push(e),
            }
        }

        // Update global checkpoint sequentially (single lock, fast operations)
        if let Some(ref manager) = self.checkpoint_manager {
            let mut mgr = manager.lock();
            let ckpt = mgr.checkpoint_mut();
            let now = std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);

            for (key, entry_count, ngrams_processed, completed_prefixes, current_prefix) in
                shard_states
            {
                // Get shard path for the record
                let path = self.config.shard_path(&key.as_file_stem());
                let record = ckpt.get_or_create_shard(&key, &path);
                record.entry_count = entry_count;
                record.ngrams_processed = ngrams_processed;
                record.completed_prefixes = completed_prefixes;
                record.current_prefix = current_prefix;
                record.last_checkpoint_time = now;
            }

            // Save global checkpoint
            mgr.save()?;
        }

        let elapsed = start.elapsed();
        log::debug!(
            "Async checkpoint finish completed: {} shards in {:.2}ms",
            self.shards.len(),
            elapsed.as_secs_f64() * 1000.0
        );

        if errors.is_empty() {
            Ok(())
        } else {
            Err(CoordinatorError::Checkpoint(errors.join("; ")))
        }
    }

    /// Perform a coordinated checkpoint with parallel WAL flushing.
    ///
    /// This method provides the same guarantees as `coordinated_checkpoint()`
    /// but with significantly better performance for large shard counts:
    ///
    /// 1. Vocabulary checkpoint (synchronous - single resource)
    /// 2. Parallel WAL sync via `sync_all_parallel()`
    /// 3. Sequential state collection from all shards (read locks, fast)
    /// 4. Sequential WAL checkpoint/truncate (write locks)
    /// 5. Atomic global checkpoint JSON save
    ///
    /// Workers can continue on non-syncing shards during step 2, only
    /// blocking when they need to access a shard that is currently syncing.
    ///
    /// # Arguments
    ///
    /// * `max_concurrent_syncs` - Maximum shards to sync in parallel.
    ///   Recommended: 8 for SSDs, 2 for HDDs.
    ///
    /// # Performance
    ///
    /// With 100 shards @ 50ms each:
    /// - Sequential: ~5000ms total blocking
    /// - Parallel (8 concurrent): ~625ms + workers continue on other shards
    pub fn coordinated_checkpoint_parallel(
        &self,
        max_concurrent_syncs: usize,
    ) -> CoordinatorResult<()> {
        let start = Instant::now();

        // Step 1: Parallel WAL sync
        let synced_count = self.sync_all_parallel(max_concurrent_syncs)?;

        // Step 2: Sequential state collection and checkpoint
        let mut errors = Vec::new();

        for entry in self.shards.iter() {
            let key = entry.key().clone();
            let shard = entry.value();

            // Acquire write lock for checkpoint (truncates WAL)
            let mut guard = shard.write();

            // Checkpoint (truncate WAL) - this is fast since data is already synced
            if let Err(e) = guard.checkpoint() {
                errors.push(format!("Shard {}: {}", key, e));
                continue;
            }

            // Update global checkpoint with shard state
            if let Some(ref manager) = self.checkpoint_manager {
                let mut mgr = manager.lock();
                let ckpt = mgr.checkpoint_mut();
                let state = guard.checkpoint_state();

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

        // Step 3: Save global checkpoint
        if let Some(ref manager) = self.checkpoint_manager {
            manager.lock().save()?;
        }

        let elapsed = start.elapsed();
        log::debug!(
            "Parallel checkpoint completed: {} shards synced, {} total shards, {}ms elapsed",
            synced_count,
            self.shards.len(),
            elapsed.as_millis()
        );

        if errors.is_empty() {
            Ok(())
        } else {
            Err(CoordinatorError::Checkpoint(errors.join("; ")))
        }
    }

    /// Retry failed syncs (after `sync_all_parallel()` returned errors).
    ///
    /// This resets shards from SyncFailed to Dirty so they can be synced again.
    pub fn retry_failed_syncs(&self) -> usize {
        let mut retried = 0;
        for entry in self.shards.iter() {
            let guard = entry.value().read();
            if guard.sync_coordinator().retry_sync() {
                retried += 1;
            }
        }
        retried
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
            guard.complete_prefix(file_prefix)?;
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

    /// Get completed prefixes for a specific n-gram order.
    ///
    /// Returns all prefixes that have been marked complete in shards
    /// associated with the given order.
    pub fn completed_prefixes_for_order(&self, order: u8) -> HashSet<String> {
        if let Some(ref manager) = self.checkpoint_manager {
            manager.lock().checkpoint().completed_prefixes_for_order(order)
        } else {
            // Fall back to querying shards directly
            self.shards
                .iter()
                .filter(|e| {
                    // Only include shards for this order or shards that contain all orders
                    e.key().order.is_none() || e.key().order == Some(order)
                })
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
            guard.complete_prefix(prefix)?;
        }

        // Update global checkpoint
        if let Some(ref manager) = self.checkpoint_manager {
            let mut mgr = manager.lock();
            mgr.checkpoint_mut().complete_prefix(shard_key, prefix);
            mgr.maybe_save()?;
        }

        Ok(())
    }

    // ========================================================================
    // Document Transaction API (for idempotent prefix imports)
    // ========================================================================

    /// Begin a prefix transaction for atomic, idempotent n-gram import.
    ///
    /// This creates a document transaction on the appropriate shard that buffers
    /// all n-gram inserts until `commit_prefix_tx()` is called. If interrupted
    /// before commit, the transaction is automatically discarded on recovery.
    ///
    /// # Key Properties
    ///
    /// - **Atomicity**: Either all n-grams are committed or none are
    /// - **Idempotency**: Uses SET semantics, so re-imports produce the same result
    /// - **Crash Safety**: Uncommitted transactions are discarded on WAL recovery
    ///
    /// # Arguments
    ///
    /// * `shard_key` - The shard key for this prefix (from `route_tokens`)
    /// * `prefix` - The prefix file being imported (used as document ID)
    ///
    /// # Returns
    ///
    /// A `CoordinatorPrefixTx` that must be passed to `tx_insert()` and
    /// eventually to `commit_prefix_tx()`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let shard_key = coordinator.route_tokens(&tokens);
    /// let mut tx = coordinator.begin_prefix_tx(&shard_key, "th")?;
    /// for (ngram, count) in ngrams {
    ///     coordinator.tx_insert(&mut tx, &ngram, count);
    /// }
    /// coordinator.commit_prefix_tx(tx)?;
    /// ```
    pub fn begin_prefix_tx(
        &self,
        shard_key: &ShardKey,
        prefix: &str,
    ) -> CoordinatorResult<CoordinatorPrefixTx> {
        let shard = self.get_or_create_shard(shard_key)?;
        let guard = shard.read();
        let inner_tx = guard.begin_prefix(prefix)?;

        Ok(CoordinatorPrefixTx {
            shard_key: shard_key.clone(),
            shard: Arc::clone(&shard),
            inner: Some(inner_tx),
        })
    }

    /// Insert an n-gram into a pending prefix transaction.
    ///
    /// The n-gram is buffered in memory and will be written atomically when
    /// the transaction is committed. Uses SET semantics (not increment),
    /// making re-imports idempotent.
    ///
    /// # Arguments
    ///
    /// * `tx` - The active transaction from `begin_prefix_tx()`
    /// * `ngram` - The n-gram key to insert
    /// * `count` - The n-gram count
    pub fn tx_insert(&self, tx: &mut CoordinatorPrefixTx, ngram: &[u8], count: u64) {
        let guard = tx.shard.read();
        if let Some(ref mut inner) = tx.inner {
            guard.tx_insert(inner, ngram, count);
        }
    }

    /// Commit a prefix transaction atomically and mark the prefix as completed.
    ///
    /// This:
    /// 1. Writes all buffered n-grams to the WAL as a single batch
    /// 2. Applies them to the trie atomically
    /// 3. Marks the prefix as completed in the shard's checkpoint state
    /// 4. Persists the shard checkpoint state to WAL (done by commit_prefix)
    /// 5. Updates and saves the global checkpoint
    ///
    /// # Durability Guarantee
    ///
    /// After this method returns successfully, the prefix completion is durable:
    /// - Shard checkpoint state is persisted to the shard's WAL
    /// - Global checkpoint is saved to disk
    ///
    /// This ensures that if the process crashes after commit_prefix_tx() returns,
    /// the prefix will be correctly marked as complete during recovery.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to commit (consumed)
    ///
    /// # Returns
    ///
    /// The number of n-grams that were committed.
    pub fn commit_prefix_tx(&self, mut tx: CoordinatorPrefixTx) -> CoordinatorResult<usize> {
        let inner_tx = tx.inner.take().expect("Transaction already consumed");
        let prefix = inner_tx.prefix.clone();

        // Commit the transaction (this now also updates and persists shard checkpoint state)
        let ngram_count = {
            let mut guard = tx.shard.write();
            guard.commit_prefix(inner_tx)?
        };

        // Update stats
        self.stats.unique_ngrams.fetch_add(ngram_count as u64, Ordering::Relaxed);

        // Mark prefix as completed in global checkpoint and force save.
        // We use save() instead of maybe_save() to ensure durability - this is
        // critical because the shard's checkpoint state has already recorded
        // the prefix as complete. If we skip saving the global checkpoint and
        // crash, recovery would rebuild the global checkpoint from shard states
        // correctly, but forcing the save here provides an extra safety margin
        // and keeps the global checkpoint in sync.
        if let Some(ref manager) = self.checkpoint_manager {
            let mut mgr = manager.lock();
            mgr.checkpoint_mut().complete_prefix(&tx.shard_key, &prefix);
            mgr.save()?;
        }

        Ok(ngram_count)
    }

    /// Abort a prefix transaction, discarding all buffered n-grams.
    ///
    /// Use this if an error occurs during processing and you want to
    /// discard the partial work without committing it.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to abort (consumed)
    pub fn abort_prefix_tx(&self, mut tx: CoordinatorPrefixTx) -> CoordinatorResult<()> {
        if let Some(inner_tx) = tx.inner.take() {
            let guard = tx.shard.read();
            guard.abort_prefix(inner_tx)?;
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

            // Use blocking write() to ensure all shards are checkpointed.
            // This prevents the bug where locked shards are silently skipped,
            // causing WAL replay to double n-gram counts on resume.
            let mut guard = shard.write();

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
        let ngrams: Vec<(&[u8], u64)> = vec![
            (b"the|quick" as &[u8], 10u64),
            (b"the|slow" as &[u8], 5),
            (b"this|is" as &[u8], 3),
        ];

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

    #[test]
    fn test_parallel_sync() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let config = ShardConfig::new(dir.path().join("shards"))
            .with_granularity(ShardGranularity::TwoChar);

        let coordinator = ShardCoordinator::new(config).expect("Failed to create coordinator");

        // Store n-grams in multiple shards
        coordinator.store_ngram("the|quick", 10).expect("Failed to store");
        coordinator.store_ngram("apple|pie", 5).expect("Failed to store");
        coordinator.store_ngram("zebra|crossing", 3).expect("Failed to store");

        // Should have 3 shards open
        assert_eq!(coordinator.open_shard_count(), 3);

        // All shards should be dirty after writes
        for key in coordinator.open_shard_keys() {
            let shard = coordinator.get_or_create_shard(&key).expect("Failed to get shard");
            assert!(shard.read().is_dirty(), "Shard {} should be dirty", key);
        }

        // Parallel sync should sync all dirty shards
        let synced = coordinator.sync_all_parallel(4).expect("Failed to parallel sync");
        assert_eq!(synced, 3, "Should have synced 3 shards");

        // All shards should be clean after sync
        for key in coordinator.open_shard_keys() {
            let shard = coordinator.get_or_create_shard(&key).expect("Failed to get shard");
            assert!(!shard.read().is_dirty(), "Shard {} should be clean after sync", key);
        }

        // Syncing again should return 0 (nothing to sync)
        let synced = coordinator.sync_all_parallel(4).expect("Failed to parallel sync");
        assert_eq!(synced, 0, "Should have synced 0 shards (all clean)");
    }

    #[test]
    fn test_is_shard_syncing() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let config = ShardConfig::new(dir.path().join("shards"))
            .with_granularity(ShardGranularity::TwoChar);

        let coordinator = ShardCoordinator::new(config).expect("Failed to create coordinator");

        // Store some data
        coordinator.store_ngram("the|quick", 10).expect("Failed to store");

        let key = ShardKey::new("th");

        // Not syncing initially
        assert!(!coordinator.is_shard_syncing(&key));

        // Manually start sync on the shard
        {
            let shard = coordinator.get_or_create_shard(&key).expect("Failed to get shard");
            let guard = shard.read();
            assert!(guard.sync_coordinator().try_start_sync());
        }

        // Now should report syncing
        assert!(coordinator.is_shard_syncing(&key));

        // Complete sync
        {
            let shard = coordinator.get_or_create_shard(&key).expect("Failed to get shard");
            let guard = shard.read();
            guard.sync_coordinator().complete_sync(42);
        }

        // Not syncing anymore
        assert!(!coordinator.is_shard_syncing(&key));
    }

    #[test]
    fn test_parallel_checkpoint() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let config = ShardConfig::new(dir.path().join("shards"))
            .with_granularity(ShardGranularity::TwoChar);

        let coordinator = ShardCoordinator::new_with_checkpoints(config.clone())
            .expect("Failed to create coordinator");

        // Store n-grams in multiple shards
        coordinator.store_ngram("the|quick", 10).expect("Failed to store");
        coordinator.store_ngram("apple|pie", 5).expect("Failed to store");
        coordinator.store_ngram("zebra|crossing", 3).expect("Failed to store");

        // Parallel checkpoint
        coordinator.coordinated_checkpoint_parallel(4).expect("Failed to parallel checkpoint");

        // All shards should be clean
        for key in coordinator.open_shard_keys() {
            let shard = coordinator.get_or_create_shard(&key).expect("Failed to get shard");
            assert!(!shard.read().is_dirty(), "Shard {} should be clean after checkpoint", key);
        }

        // Verify data survives checkpoint
        assert_eq!(coordinator.get("the|quick"), Some(10));
        assert_eq!(coordinator.get("apple|pie"), Some(5));
        assert_eq!(coordinator.get("zebra|crossing"), Some(3));
    }
}
