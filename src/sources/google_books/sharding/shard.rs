//! Individual shard wrapper around PersistentARTrie.
//!
//! Each shard manages a subset of n-grams based on prefix routing.
//! Shards are written concurrently via the lock-free overlay (`increment_cas`),
//! so workers need no exclusive write lock.

use super::routing::ShardKey;
use libdictenstein::persistent_artrie::eviction::{EvictionConfig, EvictionStats};
use libdictenstein::persistent_artrie::wal::SyncHandle;
use libdictenstein::persistent_artrie::wal_managed::WalManaged;
use libdictenstein::persistent_artrie::{DocumentTransaction, PersistentARTrie, SharedARTrie};
use libdictenstein::EvictableARTrie;
use liblevenshtein::dictionary::Dictionary;
use parking_lot::{Condvar, Mutex};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;

/// Error type for shard operations.
#[derive(Error, Debug)]
pub enum ShardError {
    /// Failed to create or open the shard file.
    #[error("Failed to create/open shard at {path}: {message}")]
    Open {
        /// Filesystem path for the shard.
        path: PathBuf,
        /// Human-readable error context.
        message: String,
    },

    /// Read operation failed.
    #[error("Read failed for shard {shard_key}: {message}")]
    Read {
        /// Shard key being read.
        shard_key: String,
        /// Human-readable error context.
        message: String,
    },

    /// Write operation failed.
    #[error("Write failed for shard {shard_key}: {message}")]
    Write {
        /// Shard key being written.
        shard_key: String,
        /// Human-readable error context.
        message: String,
    },

    /// Checkpoint operation failed.
    #[error("Checkpoint failed for shard {shard_key}: {message}")]
    Checkpoint {
        /// Shard key being checkpointed.
        shard_key: String,
        /// Human-readable error context.
        message: String,
    },

    /// Shard is locked by another writer.
    #[error("Shard {shard_key} is locked by worker {holder}")]
    Locked {
        /// Shard key that is currently locked.
        shard_key: String,
        /// Worker ID holding the active write token.
        holder: usize,
    },

    /// Writer token is invalid or expired.
    #[error("Invalid write token for shard {shard_key}")]
    InvalidToken {
        /// Shard key whose write token was rejected.
        shard_key: String,
    },

    /// Sync operation failed.
    #[error("Sync failed for shard {shard_key}: {message}")]
    Sync {
        /// Shard key being synchronized.
        shard_key: String,
        /// Human-readable error context.
        message: String,
    },

    /// Sync operation timed out.
    #[error("Sync timed out for shard {shard_key}")]
    SyncTimeout {
        /// Shard key whose sync timed out.
        shard_key: String,
    },
}

/// Result type for shard operations.
pub type ShardResult<T> = Result<T, ShardError>;

/// Sync state for per-shard WAL flushing.
///
/// This state machine tracks whether a shard has dirty data that needs to be
/// synced to disk. The state transitions are:
///
/// ```text
///     write()                mark_clean()
/// +-----------> Clean <---------------+
/// |               |                   |
/// |               | (first write)     |
/// |               v                   |
/// |            Dirty -----------------+
/// |               |                   |
/// |               | try_start_sync()  |
/// |               v                   |
/// +----------- Syncing               |
/// complete_sync()  |                  |
///                  | fail_sync()      |
///                  v                  |
///              SyncFailed ------------+
///                  retry_sync()
/// ```
///
/// Formally verified in `formal/tla/AsyncShardSync.tla`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ShardSyncState {
    /// No pending WAL writes - shard is fully persisted.
    Clean = 0,

    /// Has WAL entries not yet synced to disk.
    Dirty = 1,

    /// Currently being synced by a checkpoint operation.
    Syncing = 2,

    /// Sync failed, needs retry.
    SyncFailed = 3,
}

impl ShardSyncState {
    /// Convert from u8 (for atomic operations).
    fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Clean,
            1 => Self::Dirty,
            2 => Self::Syncing,
            3 => Self::SyncFailed,
            _ => Self::Dirty, // Default to dirty for safety
        }
    }
}

/// Per-shard sync coordinator for async WAL flushing.
///
/// This coordinator manages the sync state machine for a single shard,
/// enabling parallel checkpoint operations across multiple shards.
///
/// Workers can check `is_syncing()` and defer their writes to avoid
/// blocking on a syncing shard, implementing the "defer-and-continue"
/// pattern for non-blocking checkpoints.
///
/// # Thread Safety
///
/// All methods use atomic operations or properly synchronized primitives.
/// The sync state uses CAS (compare-and-swap) to prevent races during
/// state transitions.
///
/// Formally verified in `formal/tla/AsyncShardSync.tla`.
pub struct ShardSyncCoordinator {
    /// Current sync state (atomic for lock-free reads).
    state: AtomicU8,

    /// Condition variable for waiting on sync completion.
    /// Tuple of (completed flag, condvar).
    sync_complete: Arc<(Mutex<bool>, Condvar)>,

    /// LSN of last successful sync (for incremental tracking).
    last_synced_lsn: AtomicU64,

    /// Error message from failed sync (if any).
    last_error: Mutex<Option<String>>,
}

impl Default for ShardSyncCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl ShardSyncCoordinator {
    /// Create a new sync coordinator in Clean state.
    pub fn new() -> Self {
        Self {
            state: AtomicU8::new(ShardSyncState::Clean as u8),
            sync_complete: Arc::new((Mutex::new(true), Condvar::new())),
            last_synced_lsn: AtomicU64::new(0),
            last_error: Mutex::new(None),
        }
    }

    /// Get the current sync state.
    pub fn state(&self) -> ShardSyncState {
        ShardSyncState::from_u8(self.state.load(Ordering::Acquire))
    }

    /// Mark the shard as dirty (has pending WAL writes).
    ///
    /// Called after write operations to indicate the shard has data
    /// that needs to be synced. Only transitions from Clean to Dirty.
    pub fn mark_dirty(&self) {
        // CAS loop: only transition Clean -> Dirty
        loop {
            let current = self.state.load(Ordering::Acquire);
            let current_state = ShardSyncState::from_u8(current);

            // Already dirty or syncing - no change needed
            if current_state != ShardSyncState::Clean {
                return;
            }

            // Try to transition Clean -> Dirty
            if self
                .state
                .compare_exchange(
                    current,
                    ShardSyncState::Dirty as u8,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                return;
            }
            // CAS failed, retry
        }
    }

    /// Try to start syncing this shard.
    ///
    /// Returns `true` if the transition Dirty -> Syncing succeeded.
    /// Returns `false` if the shard is not dirty or already syncing.
    ///
    /// This uses CAS to ensure only one syncer can be active at a time.
    pub fn try_start_sync(&self) -> bool {
        // CAS: Dirty -> Syncing
        let dirty = ShardSyncState::Dirty as u8;
        let syncing = ShardSyncState::Syncing as u8;

        if self
            .state
            .compare_exchange(dirty, syncing, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            // Mark sync as not complete
            let (lock, _) = &*self.sync_complete;
            let mut completed = lock.lock();
            *completed = false;
            true
        } else {
            false
        }
    }

    /// Complete the sync operation successfully.
    ///
    /// Transitions Syncing -> Clean and notifies waiters.
    pub fn complete_sync(&self, new_lsn: u64) {
        // Update LSN first
        self.last_synced_lsn.store(new_lsn, Ordering::Release);

        // Clear any previous error
        {
            let mut error = self.last_error.lock();
            *error = None;
        }

        // Transition Syncing -> Clean
        self.state
            .store(ShardSyncState::Clean as u8, Ordering::Release);

        // Notify waiters
        let (lock, cvar) = &*self.sync_complete;
        let mut completed = lock.lock();
        *completed = true;
        cvar.notify_all();
    }

    /// Mark sync as failed.
    ///
    /// Transitions Syncing -> SyncFailed and notifies waiters.
    pub fn fail_sync(&self, error: impl Into<String>) {
        // Store error message
        {
            let mut last_error = self.last_error.lock();
            *last_error = Some(error.into());
        }

        // Transition Syncing -> SyncFailed
        self.state
            .store(ShardSyncState::SyncFailed as u8, Ordering::Release);

        // Notify waiters (they'll see the failure state)
        let (lock, cvar) = &*self.sync_complete;
        let mut completed = lock.lock();
        *completed = true; // Sync is "complete" (just failed)
        cvar.notify_all();
    }

    /// Reset from SyncFailed to Dirty for retry.
    ///
    /// Returns `true` if the transition succeeded.
    pub fn retry_sync(&self) -> bool {
        let failed = ShardSyncState::SyncFailed as u8;
        let dirty = ShardSyncState::Dirty as u8;

        self.state
            .compare_exchange(failed, dirty, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Check if the shard is currently syncing.
    ///
    /// Workers can use this to defer writes to non-syncing shards.
    pub fn is_syncing(&self) -> bool {
        self.state() == ShardSyncState::Syncing
    }

    /// Check if the shard is dirty (has unsync'd data).
    pub fn is_dirty(&self) -> bool {
        self.state() == ShardSyncState::Dirty
    }

    /// Check if sync failed.
    pub fn is_sync_failed(&self) -> bool {
        self.state() == ShardSyncState::SyncFailed
    }

    /// Wait for sync to complete (with timeout).
    ///
    /// Returns `Ok(())` if sync completed, `Err(())` if timeout.
    pub fn wait_for_sync(&self, timeout: Duration) -> Result<(), ()> {
        let (lock, cvar) = &*self.sync_complete;
        let mut completed = lock.lock();

        if *completed {
            return Ok(());
        }

        // Wait with timeout
        let result = cvar.wait_for(&mut completed, timeout);

        if result.timed_out() {
            Err(())
        } else if *completed {
            Ok(())
        } else {
            Err(())
        }
    }

    /// Get the last synced LSN.
    pub fn last_synced_lsn(&self) -> u64 {
        self.last_synced_lsn.load(Ordering::Acquire)
    }

    /// Get the last error message (if any).
    pub fn last_error(&self) -> Option<String> {
        self.last_error.lock().clone()
    }
}

/// Per-shard checkpoint state.
///
/// Stored within the shard's trie using reserved key prefixes.
#[derive(Clone, Debug, Default)]
pub struct ShardCheckpointState {
    /// Prefixes that have been fully imported to this shard.
    pub completed_prefixes: HashSet<String>,

    /// Prefix currently being imported (if any).
    pub current_prefix: Option<String>,

    /// Total n-grams processed through this shard.
    pub ngrams_processed: u64,

    /// LSN of last checkpoint.
    pub last_checkpoint_lsn: u64,
}

/// Statistics for a single shard.
#[derive(Debug, Default)]
pub struct ShardStats {
    /// Number of entries in the shard.
    pub entry_count: AtomicU64,

    /// Number of write operations.
    pub write_count: AtomicU64,

    /// Number of read operations.
    pub read_count: AtomicU64,

    /// Cumulative time spent waiting for write lock (microseconds).
    pub lock_wait_us: AtomicU64,
}

impl ShardStats {
    /// Record a write operation.
    pub fn record_write(&self) {
        self.write_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a read operation.
    pub fn record_read(&self) {
        self.read_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record lock wait time.
    pub fn record_lock_wait(&self, micros: u64) {
        self.lock_wait_us.fetch_add(micros, Ordering::Relaxed);
    }

    /// Update entry count.
    pub fn set_entry_count(&self, count: u64) {
        self.entry_count.store(count, Ordering::Relaxed);
    }

    /// Increment entry count by delta.
    pub fn add_entries(&self, delta: u64) {
        self.entry_count.fetch_add(delta, Ordering::Relaxed);
    }
}

/// Handle for tracking completion of an async shard WAL sync.
///
/// This wraps the underlying `SyncHandle` from libdictenstein with shard-specific
/// context (the shard key) for error messages and logging.
///
/// # Non-blocking Sync Pattern
///
/// The async sync pattern enables non-blocking checkpoints:
/// 1. Call `sync_async()` on each shard - this rotates the WAL segment (O(1))
/// 2. New writes go to the new segment - workers continue without blocking
/// 3. The old segment is synced in the background
/// 4. Call `wait()` when durability is needed (e.g., before marking checkpoint complete)
///
/// # Performance
///
/// With 100 shards at 50ms sync each:
/// - **Blocking sync**: ~5000ms total (sequential) or ~625ms (8 concurrent)
/// - **Async sync**: ~1-10ms rotation, workers continue immediately
///
/// The async pattern provides ~40-50x less blocking during checkpoints.
pub struct ShardSyncHandle {
    /// The underlying sync handle from libdictenstein.
    inner: SyncHandle,

    /// The shard key this handle belongs to.
    shard_key: ShardKey,
}

impl ShardSyncHandle {
    /// Check if sync has completed (non-blocking).
    ///
    /// Returns `true` if the target LSN is now durable on disk.
    pub fn is_synced(&self) -> bool {
        self.inner.is_synced()
    }

    /// Wait for sync to complete (blocking).
    ///
    /// Blocks until the target LSN is durable on disk.
    ///
    /// # Errors
    ///
    /// Returns `Err(ShardError::Sync)` if the sync failed or the
    /// background sync thread crashed.
    pub fn wait(self) -> ShardResult<()> {
        self.inner.wait().map_err(|e| ShardError::Sync {
            shard_key: self.shard_key.to_string(),
            message: format!("Async sync wait failed: {}", e),
        })
    }

    /// Wait for sync with timeout (blocking).
    ///
    /// # Arguments
    ///
    /// * `timeout` - Maximum time to wait
    ///
    /// # Returns
    ///
    /// - `Ok(true)` - Sync completed within timeout
    /// - `Ok(false)` - Timeout elapsed, sync not yet complete
    /// - `Err(...)` - Sync failed or thread crashed
    pub fn wait_timeout(&self, timeout: Duration) -> ShardResult<bool> {
        self.inner
            .wait_timeout(timeout)
            .map_err(|e| ShardError::Sync {
                shard_key: self.shard_key.to_string(),
                message: format!("Async sync wait_timeout failed: {}", e),
            })
    }

    /// Get the shard key this handle belongs to.
    pub fn shard_key(&self) -> &ShardKey {
        &self.shard_key
    }

    /// Get the target LSN that this handle is waiting for.
    pub fn target_lsn(&self) -> u64 {
        self.inner.target_lsn()
    }
}

/// Handle to an individual shard.
///
/// Wraps a `PersistentARTrie<u64>` (byte-keyed) with checkpoint state and
/// exclusive write access control. N-gram keys are raw LEB128 varint-encoded
/// byte sequences stored directly without Latin-1 char conversion.
pub struct ShardHandle {
    /// The shard key identifying this shard.
    key: ShardKey,

    /// The underlying trie (byte-keyed), shared via `Arc` so the lock-free
    /// overlay's eviction coordinator can hold a weak self-reference.
    trie: SharedARTrie<u64>,

    /// File path for this shard.
    path: PathBuf,

    /// Checkpoint state for this shard.
    checkpoint_state: ShardCheckpointState,

    /// Shard statistics.
    stats: ShardStats,

    /// Sync coordinator for async WAL flushing.
    ///
    /// Tracks sync state (Clean/Dirty/Syncing/SyncFailed) and provides
    /// synchronization primitives for parallel checkpoint operations.
    sync_coordinator: ShardSyncCoordinator,

    /// Number of entries currently in the lock-free overlay (not yet merged).
    ///
    /// Tracked via `Relaxed` ordering since it's an approximate count used
    /// only for threshold-based flush decisions. Incremented on new entries
    /// in `increment_lockfree()`, reset to 0 after merge in
    /// `flush_lockfree()`, `sync()`, and `checkpoint()`.
    lockfree_entries: AtomicU64,
}

impl ShardHandle {
    /// Reserved key prefix for checkpoint data within the trie.
    const CHECKPOINT_PREFIX: &'static str = "\x00__shard_ckpt__:";

    /// Byte variant of CHECKPOINT_PREFIX for filtering iteration results.
    const CHECKPOINT_PREFIX_BYTES: &'static [u8] = b"\x00__shard_ckpt__:";

    /// Create a new shard at the given path.
    ///
    /// Creates a new trie file, overwriting if it exists.
    /// Uses slot-level dirty tracking for optimized checkpoints (90%+ I/O reduction).
    pub fn create(key: ShardKey, path: impl AsRef<Path>) -> ShardResult<Self> {
        let path = path.as_ref().to_path_buf();

        // Use create_with_slot_tracking for optimized incremental checkpoints.
        // The lock-free overlay is always-on now (libdictenstein flips to it on
        // create), so no explicit enable_lockfree() toggle is needed.
        let trie =
            PersistentARTrie::create_with_slot_tracking(&path).map_err(|e| ShardError::Open {
                path: path.clone(),
                message: e.to_string(),
            })?;

        Ok(Self {
            key,
            trie: Arc::new(trie),
            path,
            checkpoint_state: ShardCheckpointState::default(),
            stats: ShardStats::default(),
            sync_coordinator: ShardSyncCoordinator::new(),
            lockfree_entries: AtomicU64::new(0),
        })
    }

    /// Open an existing shard with automatic crash recovery.
    ///
    /// Enables slot-level dirty tracking for optimized checkpoints (90%+ I/O reduction).
    pub fn open(key: ShardKey, path: impl AsRef<Path>) -> ShardResult<Self> {
        let path = path.as_ref().to_path_buf();

        let (trie, recovery_report) = PersistentARTrie::open_with_recovery_and_slot_tracking(&path)
            .map_err(|e| {
                let msg = format!(
                    "Failed to open shard at {:?}. If this shard was created with an older format \
                     (PersistentARTrieChar), it must be re-imported. Error: {}",
                    path, e
                );
                ShardError::Open {
                    path: path.clone(),
                    message: msg,
                }
            })?;
        // The lock-free overlay is always-on now; no explicit enable_lockfree().

        if recovery_report.mode.recovered() {
            log::info!(
                "Shard {} recovered from crash: {:?}, {} records replayed",
                key,
                recovery_report.mode,
                recovery_report.records_replayed
            );
        }

        let mut handle = Self {
            key,
            trie: Arc::new(trie),
            path,
            checkpoint_state: ShardCheckpointState::default(),
            stats: ShardStats::default(),
            sync_coordinator: ShardSyncCoordinator::new(),
            lockfree_entries: AtomicU64::new(0),
        };

        // Load checkpoint state from trie
        handle.load_checkpoint_state()?;
        handle
            .stats
            .set_entry_count(handle.trie.len().unwrap_or(0) as u64);

        Ok(handle)
    }

    /// Open an existing shard or create a new one.
    ///
    /// This method is designed to be called under a per-shard creation lock
    /// managed by `ShardCoordinator`. The coordinator serializes creation
    /// attempts for the same shard key, eliminating TOCTOU race conditions.
    ///
    /// # Arguments
    ///
    /// * `key` - The shard key identifying this shard.
    /// * `path` - File path for the shard.
    ///
    /// # Returns
    ///
    /// A `ShardHandle` for the opened or newly created shard.
    ///
    /// # Thread Safety
    ///
    /// Callers MUST ensure this is called under appropriate synchronization
    /// (e.g., via `ShardCoordinator::create_or_open_shard`) to prevent
    /// concurrent creation attempts on the same path.
    pub fn open_or_create(key: ShardKey, path: impl AsRef<Path>) -> ShardResult<Self> {
        let path = path.as_ref();

        if path.exists() {
            Self::open(key, path)
        } else {
            Self::create(key, path)
        }
    }

    /// Arm overlay-heap eviction on this shard's trie.
    ///
    /// When `config` is `Some`, the checkpoint tail evicts the coldest resident
    /// overlay nodes down to `config.resident_budget_bytes` (lossless — evicted
    /// nodes fault back on read). `None` leaves the overlay unbounded (legacy
    /// behavior). Called once by the coordinator after open/create; the installed
    /// eviction coordinator holds a weak ref to this shard's `Arc`-d trie, so it
    /// is torn down when the shard's last `Arc` drops.
    pub fn arm_eviction(&self, config: Option<EvictionConfig>) -> ShardResult<()> {
        if let Some(config) = config {
            self.trie
                .enable_eviction(config)
                .map_err(|e| ShardError::Open {
                    path: self.path.clone(),
                    message: format!("failed to enable overlay eviction: {e}"),
                })?;
        }
        Ok(())
    }

    /// Overlay-eviction statistics for this shard's trie.
    ///
    /// `nodes_evicted` / `bytes_freed` accumulate across the checkpoint-tail
    /// budget evictions; `resident_bytes` is the live resident-overlay estimate.
    pub fn eviction_stats(&self) -> EvictionStats {
        self.trie.eviction_stats()
    }

    /// Get the shard key.
    pub fn key(&self) -> &ShardKey {
        &self.key
    }

    /// Get the file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the entry count.
    pub fn len(&self) -> usize {
        self.stats
            .entry_count
            .load(std::sync::atomic::Ordering::Relaxed) as usize
    }

    /// Check if the shard is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the approximate number of entries in the lock-free overlay.
    ///
    /// This count is incremented when new entries are added via
    /// `increment_lockfree()` and reset to 0 when the overlay is merged
    /// into the persistent trie (via `flush_lockfree()`, `sync()`, or
    /// `checkpoint()`).
    ///
    /// Used by the coordinator to decide when to flush individual shards
    /// to bound memory usage during high-parallelism imports.
    pub fn lockfree_entry_count(&self) -> u64 {
        self.lockfree_entries.load(Ordering::Relaxed)
    }

    /// Get the checkpoint state.
    pub fn checkpoint_state(&self) -> &ShardCheckpointState {
        &self.checkpoint_state
    }

    /// Get shard statistics.
    pub fn stats(&self) -> &ShardStats {
        &self.stats
    }

    /// Lock-free increment using CAS. Only needs `&self` (shared access).
    ///
    /// Uses the lock-free overlay's `increment_cas` — no exclusive write lock
    /// required, so multiple workers can increment the same shard concurrently
    /// without serialization.
    pub fn increment_lockfree(&self, ngram: &[u8], count: u64) -> ShardResult<bool> {
        // Single overlay read: under the overlay-default write mode `get_value_bytes`
        // and `get_lockfree` observe the same overlay leaf, so one read is the source
        // of truth (summing them would double-count).
        let was_new = self.trie.get_value_bytes(ngram).is_none();

        self.trie.increment_cas(ngram, count);

        self.stats.record_write();
        if was_new {
            self.stats.add_entries(1);
            self.lockfree_entries.fetch_add(1, Ordering::Relaxed);
        }

        self.sync_coordinator.mark_dirty();

        Ok(was_new)
    }

    /// Get the count for an n-gram (overlay-default: single source of truth).
    pub fn get(&self, ngram: &[u8]) -> Option<u64> {
        self.stats.record_read();
        // `get_value_bytes` is the single source of truth under the overlay-default
        // write mode; the prior `get_lockfree + get_value_bytes` sum read the same
        // overlay leaf twice and double-counted.
        match self.trie.get_value_bytes(ngram).unwrap_or(0) {
            0 => None,
            total => Some(total),
        }
    }

    /// Check if an n-gram exists (overlay-default: `contains_bytes` routes to the overlay).
    pub fn contains(&self, ngram: &[u8]) -> bool {
        self.trie.contains_bytes(ngram)
    }

    /// Iterate over all n-grams with their counts.
    ///
    /// Returns `(Vec<u8>, u64)` pairs where the key is the raw varint-encoded
    /// byte key and the value is the n-gram count.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying trie iteration fails. This can happen
    /// due to I/O errors, corrupted data, or other trie-level issues.
    pub fn iter_with_counts(&self) -> ShardResult<Vec<(Vec<u8>, u64)>> {
        // iter_prefix_with_values on PersistentARTrie returns Option<impl Iterator<Item=(Vec<u8>, V)>>
        // We collect into a Vec to avoid lifetime issues with borrowed iterators.
        match self.trie.iter_prefix_with_values(b"") {
            Some(iter) => Ok(iter
                .filter(|(k, _)| !k.starts_with(Self::CHECKPOINT_PREFIX_BYTES))
                .collect()),
            None => Ok(Vec::new()),
        }
    }

    /// Persist the shard to disk.
    ///
    /// Under the overlay-default write mode the lock-free overlay IS the durable
    /// production state, so `checkpoint()` (which serializes the overlay snapshot into
    /// the on-disk image) is what makes `increment_cas` counts crash-durable — a bare
    /// WAL `sync()` would not capture them. The former
    /// `merge_lockfree_values_to_persistent()` pre-step is obsolete (it rejects under
    /// the overlay) and has been removed.
    pub fn sync(&self) -> ShardResult<()> {
        self.trie.checkpoint().map_err(|e| ShardError::Checkpoint {
            shard_key: self.key.to_string(),
            message: format!("sync failed: {}", e),
        })?;
        self.lockfree_entries.store(0, Ordering::Relaxed);
        Ok(())
    }

    /// Sync WAL to disk with state tracking for parallel checkpoints.
    ///
    /// This method:
    /// 1. Attempts to start sync (Dirty -> Syncing)
    /// 2. Performs the actual sync
    /// 3. Marks completion (Syncing -> Clean) or failure (Syncing -> SyncFailed)
    ///
    /// Returns `Ok(true)` if sync was performed, `Ok(false)` if no sync needed
    /// (shard was clean or already syncing).
    pub fn sync_tracked(&self) -> ShardResult<bool> {
        // Try to start sync (CAS: Dirty -> Syncing)
        if !self.sync_coordinator.try_start_sync() {
            // Either clean (no sync needed) or already syncing
            return Ok(false);
        }

        // Persist the overlay to disk. Under the overlay-default write mode only a
        // `checkpoint()` makes the lock-free `increment_cas` counts crash-durable (a
        // bare WAL `sync()` does not capture the overlay); the obsolete
        // `merge_lockfree_values_to_persistent()` pre-step has been removed.
        match self.trie.checkpoint() {
            Ok(()) => {
                self.lockfree_entries.store(0, Ordering::Relaxed);
                // Use actual synced LSN from the ARTrie WAL
                let lsn = self.trie.synced_lsn().unwrap_or(0);
                self.sync_coordinator.complete_sync(lsn);
                Ok(true)
            }
            Err(e) => {
                // Failure: mark failed and notify waiters
                let error_msg = format!("sync failed: {}", e);
                self.sync_coordinator.fail_sync(&error_msg);
                Err(ShardError::Sync {
                    shard_key: self.key.to_string(),
                    message: error_msg,
                })
            }
        }
    }

    /// Mark the shard as dirty (has pending WAL writes).
    ///
    /// Call this after write operations to indicate the shard needs sync.
    /// Note: `increment()` calls this automatically.
    pub fn mark_dirty(&self) {
        self.sync_coordinator.mark_dirty();
    }

    /// Check if the shard is currently syncing.
    ///
    /// Workers can use this to defer writes to avoid blocking.
    pub fn is_syncing(&self) -> bool {
        self.sync_coordinator.is_syncing()
    }

    /// Check if the shard has dirty (unsync'd) data.
    pub fn is_dirty(&self) -> bool {
        self.sync_coordinator.is_dirty()
    }

    /// Get the current sync state.
    pub fn sync_state(&self) -> ShardSyncState {
        self.sync_coordinator.state()
    }

    /// Get a reference to the sync coordinator.
    pub fn sync_coordinator(&self) -> &ShardSyncCoordinator {
        &self.sync_coordinator
    }

    /// Wait for sync to complete (with timeout).
    ///
    /// Returns `Ok(())` if sync completed, `Err(ShardError::SyncTimeout)` if timeout.
    pub fn wait_for_sync(&self, timeout: Duration) -> ShardResult<()> {
        self.sync_coordinator
            .wait_for_sync(timeout)
            .map_err(|()| ShardError::SyncTimeout {
                shard_key: self.key.to_string(),
            })
    }

    /// Start async WAL sync - returns immediately, sync happens in background.
    ///
    /// This uses the WAL's segment rotation to enable non-blocking sync:
    /// - O(1) rotation creates a new segment for new writes
    /// - Previous segment is synced in the background
    /// - Writers can continue immediately without blocking
    ///
    /// The returned `ShardSyncHandle` can be used to:
    /// - Check sync status with `is_synced()` (non-blocking)
    /// - Wait for completion with `wait()` (blocking)
    /// - Wait with timeout via `wait_timeout()` (blocking with timeout)
    ///
    /// # Returns
    ///
    /// - `Ok(Some(handle))` - Async sync initiated, use handle to track completion
    /// - `Ok(None)` - No WAL configured (in-memory mode), nothing to sync
    /// - `Err(...)` - Failed to initiate async sync
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Start async sync
    /// if let Some(handle) = shard.sync_async()? {
    ///     // Continue processing while sync happens in background
    ///     process_more_data();
    ///
    ///     // Check if done (non-blocking)
    ///     if !handle.is_synced() {
    ///         // Still syncing, do other work
    ///     }
    ///
    ///     // Wait when durability is needed
    ///     handle.wait()?;
    /// }
    /// ```
    pub fn sync_async(&self) -> ShardResult<Option<ShardSyncHandle>> {
        // Use WalManaged trait method to initiate async sync
        let handle = self.trie.wal_sync_async().map_err(|e| ShardError::Sync {
            shard_key: self.key.to_string(),
            message: format!("Failed to initiate async sync: {}", e),
        })?;

        Ok(handle.map(|inner| ShardSyncHandle {
            inner,
            shard_key: self.key.clone(),
        }))
    }

    /// Get the current LSN (Log Sequence Number) of this shard's trie.
    ///
    /// This returns the next LSN to be assigned. It increases monotonically
    /// with each write operation.
    pub fn current_lsn(&self) -> u64 {
        self.trie.current_lsn()
    }

    /// Get the highest durable LSN (synced to disk) of this shard's trie.
    ///
    /// Operations with LSN ≤ synced_lsn are guaranteed to survive crashes.
    /// Returns `None` if WAL is not active (in-memory trie).
    pub fn synced_lsn(&self) -> Option<u64> {
        self.trie.synced_lsn()
    }

    /// Persist the lock-free overlay to disk and reclaim its resident memory.
    ///
    /// Under the overlay-default write mode the overlay IS the durable state, so a
    /// `checkpoint()` (overlay snapshot → on-disk image) both persists the counts and
    /// lets the overlay reclaim memory. The obsolete `merge_lockfree_values_to_persistent`
    /// pre-step (which rejects under the overlay) has been removed; point lookups and
    /// iteration already read the overlay directly, so no pre-iteration flush is needed.
    pub fn flush_lockfree(&self) -> ShardResult<()> {
        self.trie.checkpoint().map_err(|e| ShardError::Checkpoint {
            shard_key: self.key.to_string(),
            message: format!("flush_lockfree failed: {}", e),
        })?;
        self.lockfree_entries.store(0, Ordering::Relaxed);
        Ok(())
    }

    /// Checkpoint the shard (persist to disk and truncate WAL).
    ///
    /// Uses sequential flush for optimized disk I/O (5-15% faster checkpoints).
    pub fn checkpoint(&self) -> ShardResult<()> {
        // Save checkpoint state to trie
        self.save_checkpoint_state()?;

        // Flush dirty arenas in sequential order for optimal I/O
        self.trie
            .flush_sequential()
            .map_err(|e| ShardError::Checkpoint {
                shard_key: self.key.to_string(),
                message: format!("flush_sequential failed: {}", e),
            })?;

        // Checkpoint the trie. Under the overlay-default write mode this serializes the
        // overlay snapshot (the durable production state) into the on-disk image — the
        // obsolete `merge_lockfree_values_to_persistent` pre-step has been removed.
        self.trie.checkpoint().map_err(|e| ShardError::Checkpoint {
            shard_key: self.key.to_string(),
            message: e.to_string(),
        })?;
        self.lockfree_entries.store(0, Ordering::Relaxed);
        Ok(())
    }

    /// Mark a prefix as completed in this shard.
    ///
    /// This immediately persists the checkpoint state to the WAL so that
    /// the completion survives crashes even if a full checkpoint hasn't occurred.
    pub fn complete_prefix(&mut self, prefix: &str) -> ShardResult<()> {
        self.checkpoint_state
            .completed_prefixes
            .insert(prefix.to_string());
        self.checkpoint_state.current_prefix = None;

        // Immediately persist to WAL so this survives crashes
        self.persist_checkpoint_state()
    }

    /// Set the current prefix being processed.
    pub fn set_current_prefix(&mut self, prefix: Option<&str>) {
        self.checkpoint_state.current_prefix = prefix.map(String::from);
    }

    /// Add to the n-gram count.
    pub fn add_ngrams_processed(&mut self, count: u64) {
        self.checkpoint_state.ngrams_processed += count;
    }

    /// Load checkpoint state from the trie.
    fn load_checkpoint_state(&mut self) -> ShardResult<()> {
        // Load n-grams processed count
        let ngrams_key = format!("{}ngrams_processed", Self::CHECKPOINT_PREFIX);
        if let Some(value) = self.trie.get_value_bytes(ngrams_key.as_bytes()) {
            self.checkpoint_state.ngrams_processed = value as u64;
        }

        // Load completed prefixes by scanning for prefix keys
        // Key format: \x00__shard_ckpt__:prefix:XX
        let prefix_pattern = format!("{}prefix:", Self::CHECKPOINT_PREFIX);
        let prefix_pattern_bytes = prefix_pattern.as_bytes();

        if let Some(iter) = self.trie.iter_prefix_with_values(prefix_pattern_bytes) {
            for (key, _value) in iter {
                // Extract prefix name from key: \x00__shard_ckpt__:prefix:XX -> XX
                if key.starts_with(prefix_pattern_bytes) {
                    let suffix = &key[prefix_pattern_bytes.len()..];
                    if let Ok(prefix) = std::str::from_utf8(suffix) {
                        self.checkpoint_state
                            .completed_prefixes
                            .insert(prefix.to_string());
                    }
                }
            }
        }

        log::debug!(
            "Shard {}: loaded {} completed prefixes, {} ngrams processed",
            self.key,
            self.checkpoint_state.completed_prefixes.len(),
            self.checkpoint_state.ngrams_processed
        );

        Ok(())
    }

    /// Save checkpoint state to the trie.
    fn save_checkpoint_state(&self) -> ShardResult<()> {
        // Save n-grams processed count
        let ngrams_key = format!("{}ngrams_processed", Self::CHECKPOINT_PREFIX);
        self.trie
            .upsert_bytes(
                ngrams_key.as_bytes(),
                self.checkpoint_state.ngrams_processed,
            )
            .map_err(|e| ShardError::Checkpoint {
                shard_key: self.key.to_string(),
                message: format!("failed to save ngrams_processed: {}", e),
            })?;

        // Save completed prefix count (for backward compatibility)
        let completed_key = format!("{}completed", Self::CHECKPOINT_PREFIX);
        self.trie
            .upsert_bytes(
                completed_key.as_bytes(),
                self.checkpoint_state.completed_prefixes.len() as u64,
            )
            .map_err(|e| ShardError::Checkpoint {
                shard_key: self.key.to_string(),
                message: format!("failed to save completed count: {}", e),
            })?;

        // Save each completed prefix name as a separate key
        // Key format: \x00__shard_ckpt__:prefix:XX where XX is the prefix
        for prefix in &self.checkpoint_state.completed_prefixes {
            let prefix_key = format!("{}prefix:{}", Self::CHECKPOINT_PREFIX, prefix);
            self.trie
                .upsert_bytes(prefix_key.as_bytes(), 1) // Value 1 = completed marker
                .map_err(|e| ShardError::Checkpoint {
                    shard_key: self.key.to_string(),
                    message: format!("failed to save prefix {}: {}", prefix, e),
                })?;
        }

        Ok(())
    }

    /// Persist checkpoint state to WAL immediately (without full checkpoint).
    ///
    /// Call this after marking a prefix as complete to ensure the state
    /// survives crashes. This writes to the WAL but doesn't truncate it.
    pub fn persist_checkpoint_state(&self) -> ShardResult<()> {
        self.save_checkpoint_state()?;
        self.trie.sync().map_err(|e| ShardError::Checkpoint {
            shard_key: self.key.to_string(),
            message: format!("failed to sync checkpoint state: {}", e),
        })
    }

    // ========================================================================
    // Document Transaction API (for idempotent prefix imports)
    // ========================================================================

    /// Begin a document transaction for a prefix file.
    ///
    /// This creates an atomic transaction that buffers all n-gram inserts
    /// until `commit_prefix()` is called. If interrupted before commit,
    /// the transaction is automatically discarded on recovery.
    ///
    /// # Arguments
    ///
    /// * `prefix` - The prefix file being imported (used as document ID)
    ///
    /// # Returns
    ///
    /// A `PrefixTransaction` that must be passed to `tx_insert()` and
    /// eventually to `commit_prefix()`.
    pub fn begin_prefix(&self, prefix: &str) -> ShardResult<PrefixTransaction<u64>> {
        let document_id = format!("prefix:{}", prefix);
        let tx = self
            .trie
            .begin_document(&document_id)
            .map_err(|e| ShardError::Write {
                shard_key: self.key.to_string(),
                message: format!("Failed to begin transaction for prefix '{}': {}", prefix, e),
            })?;
        Ok(PrefixTransaction {
            prefix: prefix.to_string(),
            tx,
            ngram_count: 0,
        })
    }

    /// Insert an n-gram into a pending prefix transaction.
    ///
    /// The n-gram is buffered in memory and will be written atomically
    /// when the transaction is committed. Uses SET semantics (not increment),
    /// making re-imports idempotent.
    ///
    /// # Arguments
    ///
    /// * `tx` - The active transaction from `begin_prefix()`
    /// * `ngram` - The n-gram key (raw varint-encoded bytes)
    /// * `count` - The n-gram count
    pub fn tx_insert(&self, tx: &mut PrefixTransaction<u64>, ngram: &[u8], count: u64) {
        self.trie.tx_insert_bytes(&mut tx.tx, ngram, Some(count));
        tx.ngram_count += 1;
    }

    /// Commit a prefix transaction atomically.
    ///
    /// This writes all buffered n-grams to the WAL as a single batch record,
    /// then applies them to the trie. The transaction is committed atomically -
    /// either all n-grams are persisted or none are.
    ///
    /// After commit, marks the shard as dirty for eventual sync.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to commit (consumed)
    ///
    /// # Returns
    ///
    /// The number of n-grams that were committed.
    pub fn commit_prefix(&mut self, tx: PrefixTransaction<u64>) -> ShardResult<usize> {
        let ngram_count = tx.ngram_count;
        let prefix = tx.prefix.clone();

        let inserted = self
            .trie
            .commit_document(tx.tx)
            .map_err(|e| ShardError::Write {
                shard_key: self.key.to_string(),
                message: format!(
                    "Failed to commit transaction for prefix '{}': {}",
                    prefix, e
                ),
            })?;

        // Update stats
        self.stats.add_entries(inserted as u64);
        self.stats.record_write();

        // Update checkpoint state with completed prefix.
        // This is CRITICAL for crash recovery - without this, the shard's checkpoint
        // state won't record the prefix as complete, causing data loss on resume.
        self.checkpoint_state
            .completed_prefixes
            .insert(prefix.clone());
        self.checkpoint_state.current_prefix = None;

        // Persist checkpoint state to WAL so it survives crashes.
        // This ensures that even if the process crashes after commit_document()
        // but before the global checkpoint is saved, the shard's local state
        // will record this prefix as complete during WAL recovery.
        self.persist_checkpoint_state()?;

        // Mark shard as dirty after write
        self.sync_coordinator.mark_dirty();

        log::trace!(
            "Shard {}: committed prefix '{}' with {} n-grams ({} newly inserted)",
            self.key,
            prefix,
            ngram_count,
            inserted
        );

        Ok(ngram_count)
    }

    /// Commit a prefix transaction chunk WITHOUT marking the prefix as complete.
    ///
    /// This writes all buffered n-grams to the WAL as a single batch record,
    /// then applies them to the trie. Unlike `commit_prefix()`, this does NOT:
    /// - Update `checkpoint_state.completed_prefixes`
    /// - Persist checkpoint state to WAL
    ///
    /// This is used for chunked imports of large prefix files (e.g., 2-gram
    /// files with 50-100M entries). The caller commits chunks periodically
    /// to bound memory usage, then calls `commit_prefix()` on the final chunk
    /// to mark the prefix as complete.
    ///
    /// # Crash Recovery
    ///
    /// If the process crashes between chunk commits:
    /// - Committed chunks are durable in the WAL
    /// - The prefix is NOT marked as complete
    /// - On resume, the prefix is re-imported from scratch (SET semantics
    ///   make this idempotent — already-committed n-grams are overwritten
    ///   with the same values)
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to commit (consumed). Caller must begin a
    ///   new transaction for the next chunk.
    ///
    /// # Returns
    ///
    /// The number of n-grams that were committed in this chunk.
    pub fn commit_chunk(&mut self, tx: PrefixTransaction<u64>) -> ShardResult<usize> {
        let ngram_count = tx.ngram_count;
        let prefix = tx.prefix.clone();

        let inserted = self
            .trie
            .commit_document(tx.tx)
            .map_err(|e| ShardError::Write {
                shard_key: self.key.to_string(),
                message: format!("Failed to commit chunk for prefix '{}': {}", prefix, e),
            })?;

        // Update stats
        self.stats.add_entries(inserted as u64);
        self.stats.record_write();

        // NOTE: We intentionally do NOT update checkpoint_state.completed_prefixes
        // or persist checkpoint state here. The prefix is only marked complete
        // when the final chunk is committed via commit_prefix().

        // Mark shard as dirty after write
        self.sync_coordinator.mark_dirty();

        log::trace!(
            "Shard {}: committed chunk for prefix '{}' with {} n-grams ({} newly inserted)",
            self.key,
            prefix,
            ngram_count,
            inserted
        );

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
    pub fn abort_prefix(&self, tx: PrefixTransaction<u64>) -> ShardResult<()> {
        let prefix = tx.prefix.clone();
        self.trie
            .abort_document(tx.tx)
            .map_err(|e| ShardError::Write {
                shard_key: self.key.to_string(),
                message: format!("Failed to abort transaction for prefix '{}': {}", prefix, e),
            })?;

        log::trace!(
            "Shard {}: aborted prefix '{}' transaction ({} n-grams discarded)",
            self.key,
            prefix,
            tx.ngram_count
        );

        Ok(())
    }
}

/// A pending prefix transaction for atomic n-gram imports.
///
/// This wraps a `DocumentTransaction` with prefix-specific metadata
/// and provides idempotent import semantics:
///
/// - **Atomicity**: All n-grams are committed together or none are
/// - **Idempotency**: Uses SET semantics, so re-imports produce the same result
/// - **Crash safety**: Uncommitted transactions are discarded on recovery
///
/// # Usage
///
/// ```ignore
/// let mut tx = shard.begin_prefix("th")?;
/// for (ngram, count) in ngrams {
///     shard.tx_insert(&mut tx, &ngram, count);
/// }
/// shard.commit_prefix(tx)?;
/// ```
pub struct PrefixTransaction<V: liblevenshtein::dictionary::DictionaryValue> {
    /// The prefix file being imported.
    pub prefix: String,

    /// The underlying document transaction.
    tx: DocumentTransaction<V>,

    /// Number of n-grams buffered in this transaction.
    pub ngram_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_shard_create_and_write() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("test_shard.artrie");
        let key = ShardKey::new("th");

        let shard = ShardHandle::create(key.clone(), &path).expect("Failed to create shard");

        // Write some data via the lock-free path.
        let was_new = shard
            .increment_lockfree(b"the|quick", 5)
            .expect("Failed to increment");
        assert!(was_new);

        let was_new = shard
            .increment_lockfree(b"the|quick", 3)
            .expect("Failed to increment");
        assert!(!was_new);

        // Read back
        assert_eq!(shard.get(b"the|quick"), Some(8));
        assert_eq!(shard.len(), 1);
    }

    #[test]
    fn test_shard_persistence() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("test_shard.artrie");
        let key = ShardKey::new("th");

        // Create and write
        {
            let shard = ShardHandle::create(key.clone(), &path).expect("Failed to create shard");
            shard.increment_lockfree(b"the|quick", 10).unwrap();
            shard.sync().unwrap();
        }

        // Reopen and verify
        {
            let shard = ShardHandle::open(key, &path).expect("Failed to open shard");
            assert_eq!(shard.get(b"the|quick"), Some(10));
        }
    }

    #[test]
    fn test_open_or_create_new_shard() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("new_shard.artrie");
        let key = ShardKey::new("ab");

        // File doesn't exist - should create
        assert!(!path.exists());
        let shard = ShardHandle::open_or_create(key.clone(), &path)
            .expect("Failed to open_or_create shard");
        assert!(path.exists());

        // Write data via the lock-free path.
        shard.increment_lockfree(b"apple|pie", 5).unwrap();
        shard.sync().unwrap();

        assert_eq!(shard.get(b"apple|pie"), Some(5));
    }

    #[test]
    fn test_open_or_create_existing_shard() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("existing_shard.artrie");
        let key = ShardKey::new("cd");

        // Create initial shard with data
        {
            let shard = ShardHandle::create(key.clone(), &path).expect("Failed to create shard");
            shard.increment_lockfree(b"cat|dog", 7).unwrap();
            shard.sync().unwrap();
        }

        // open_or_create should open existing shard
        let shard = ShardHandle::open_or_create(key, &path)
            .expect("Failed to open_or_create existing shard");

        // Verify data is preserved
        assert_eq!(shard.get(b"cat|dog"), Some(7));
    }

    // ========== Sync Coordinator Tests ==========

    #[test]
    fn test_sync_state_machine() {
        let coordinator = ShardSyncCoordinator::new();

        // Initial state is Clean
        assert_eq!(coordinator.state(), ShardSyncState::Clean);
        assert!(!coordinator.is_syncing());
        assert!(!coordinator.is_dirty());

        // Mark dirty
        coordinator.mark_dirty();
        assert_eq!(coordinator.state(), ShardSyncState::Dirty);
        assert!(coordinator.is_dirty());
        assert!(!coordinator.is_syncing());

        // Marking dirty again is idempotent
        coordinator.mark_dirty();
        assert_eq!(coordinator.state(), ShardSyncState::Dirty);

        // Start sync (Dirty -> Syncing)
        assert!(coordinator.try_start_sync());
        assert_eq!(coordinator.state(), ShardSyncState::Syncing);
        assert!(coordinator.is_syncing());
        assert!(!coordinator.is_dirty());

        // Can't start sync again while syncing
        assert!(!coordinator.try_start_sync());

        // Complete sync (Syncing -> Clean)
        coordinator.complete_sync(100);
        assert_eq!(coordinator.state(), ShardSyncState::Clean);
        assert!(!coordinator.is_syncing());
        assert_eq!(coordinator.last_synced_lsn(), 100);
    }

    #[test]
    fn test_sync_state_failure() {
        let coordinator = ShardSyncCoordinator::new();

        // Mark dirty and start sync
        coordinator.mark_dirty();
        assert!(coordinator.try_start_sync());
        assert!(coordinator.is_syncing());

        // Fail sync
        coordinator.fail_sync("disk full");
        assert_eq!(coordinator.state(), ShardSyncState::SyncFailed);
        assert!(coordinator.is_sync_failed());
        assert!(!coordinator.is_syncing());
        assert_eq!(coordinator.last_error(), Some("disk full".to_string()));

        // Retry (SyncFailed -> Dirty)
        assert!(coordinator.retry_sync());
        assert_eq!(coordinator.state(), ShardSyncState::Dirty);
        assert!(coordinator.is_dirty());

        // Can start sync again after retry
        assert!(coordinator.try_start_sync());
        coordinator.complete_sync(200);
        assert_eq!(coordinator.state(), ShardSyncState::Clean);
    }

    #[test]
    fn test_sync_tracked_marks_dirty() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("test_shard.artrie");
        let key = ShardKey::new("th");

        let shard = ShardHandle::create(key, &path).expect("Failed to create shard");

        // Initially clean
        assert_eq!(shard.sync_state(), ShardSyncState::Clean);

        // Lock-free write marks dirty
        shard
            .increment_lockfree(b"the|quick", 5)
            .expect("Failed to increment");
        assert_eq!(shard.sync_state(), ShardSyncState::Dirty);

        // sync_tracked transitions through Syncing to Clean
        assert!(shard.sync_tracked().expect("sync_tracked failed"));
        assert_eq!(shard.sync_state(), ShardSyncState::Clean);

        // sync_tracked on clean shard returns false (no sync needed)
        assert!(!shard.sync_tracked().expect("sync_tracked failed"));
    }

    #[test]
    fn test_sync_coordinator_wait() {
        use std::thread;

        let coordinator = Arc::new(ShardSyncCoordinator::new());

        // Mark dirty and start sync
        coordinator.mark_dirty();
        assert!(coordinator.try_start_sync());

        // Spawn a thread to complete sync after a delay
        let coordinator_clone = Arc::clone(&coordinator);
        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            coordinator_clone.complete_sync(42);
        });

        // Wait for sync (should succeed)
        let result = coordinator.wait_for_sync(Duration::from_millis(200));
        assert!(result.is_ok());
        assert_eq!(coordinator.state(), ShardSyncState::Clean);

        handle.join().expect("Thread panicked");
    }

    #[test]
    fn test_sync_coordinator_timeout() {
        let coordinator = ShardSyncCoordinator::new();

        // Mark dirty and start sync (but don't complete it)
        coordinator.mark_dirty();
        assert!(coordinator.try_start_sync());

        // Wait should timeout
        let result = coordinator.wait_for_sync(Duration::from_millis(10));
        assert!(result.is_err());
        assert!(coordinator.is_syncing()); // Still syncing
    }

    // ========== Document Transaction API Tests ==========

    #[test]
    fn test_commit_prefix_updates_checkpoint_state() {
        // This test verifies the fix for data loss on interrupt/resume.
        // The transaction-based commit_prefix() must update the shard's checkpoint
        // state so that completed prefixes survive crashes.
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("test_shard.artrie");
        let key = ShardKey::new("th");

        // Create shard and commit a prefix via transaction API
        {
            let mut shard =
                ShardHandle::create(key.clone(), &path).expect("Failed to create shard");

            // Begin transaction
            let mut tx = shard.begin_prefix("th").expect("Failed to begin prefix");

            // Insert some n-grams
            shard.tx_insert(&mut tx, b"the|quick", 10);
            shard.tx_insert(&mut tx, b"the|brown", 5);

            // Verify checkpoint state is empty before commit
            assert!(
                shard.checkpoint_state().completed_prefixes.is_empty(),
                "Checkpoint state should be empty before commit"
            );

            // Commit the transaction
            let count = shard.commit_prefix(tx).expect("Failed to commit prefix");
            assert_eq!(count, 2, "Should have committed 2 n-grams");

            // Verify checkpoint state is updated after commit
            assert!(
                shard.checkpoint_state().completed_prefixes.contains("th"),
                "Checkpoint state should contain 'th' after commit"
            );
            assert!(
                shard.checkpoint_state().current_prefix.is_none(),
                "Current prefix should be None after commit"
            );
        }

        // Reopen shard and verify checkpoint state was persisted
        {
            let shard = ShardHandle::open(key, &path).expect("Failed to open shard");

            // Checkpoint state should be loaded from WAL
            assert!(
                shard.checkpoint_state().completed_prefixes.contains("th"),
                "Checkpoint state should persist 'th' across reopen - this is the fix!"
            );

            // Data should also be present
            assert_eq!(shard.get(b"the|quick"), Some(10));
            assert_eq!(shard.get(b"the|brown"), Some(5));
        }
    }

    #[test]
    fn test_commit_prefix_multiple_prefixes() {
        // Test that multiple prefixes can be committed and all are persisted
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("test_shard.artrie");
        let key = ShardKey::new("th");

        {
            let mut shard =
                ShardHandle::create(key.clone(), &path).expect("Failed to create shard");

            // Commit first prefix
            let mut tx1 = shard.begin_prefix("th").expect("Failed to begin prefix");
            shard.tx_insert(&mut tx1, b"the|quick", 10);
            shard.commit_prefix(tx1).expect("Failed to commit prefix");

            // Commit second prefix (different one routed to same shard)
            let mut tx2 = shard.begin_prefix("ti").expect("Failed to begin prefix");
            shard.tx_insert(&mut tx2, b"time|flies", 3);
            shard.commit_prefix(tx2).expect("Failed to commit prefix");

            // Both prefixes should be in checkpoint state
            assert!(shard.checkpoint_state().completed_prefixes.contains("th"));
            assert!(shard.checkpoint_state().completed_prefixes.contains("ti"));
        }

        // Verify persistence
        {
            let shard = ShardHandle::open(key, &path).expect("Failed to open shard");
            assert!(shard.checkpoint_state().completed_prefixes.contains("th"));
            assert!(shard.checkpoint_state().completed_prefixes.contains("ti"));
            assert_eq!(shard.checkpoint_state().completed_prefixes.len(), 2);
        }
    }

    #[test]
    fn test_abort_prefix_does_not_update_checkpoint() {
        // Verify that aborted transactions don't affect checkpoint state
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("test_shard.artrie");
        let key = ShardKey::new("th");

        {
            let shard = ShardHandle::create(key.clone(), &path).expect("Failed to create shard");

            // Begin and abort a transaction
            let mut tx = shard.begin_prefix("th").expect("Failed to begin prefix");
            shard.tx_insert(&mut tx, b"the|quick", 10);
            shard.abort_prefix(tx).expect("Failed to abort prefix");

            // Checkpoint state should not be updated
            assert!(
                shard.checkpoint_state().completed_prefixes.is_empty(),
                "Aborted prefix should not appear in checkpoint state"
            );

            // Data should not be present
            assert_eq!(shard.get(b"the|quick"), None);
        }
    }

    // ---- Lock-free overlay entry-count tracking ----

    #[test]
    fn test_lockfree_entry_count_increments_on_increment() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("test_shard.artrie");
        let shard = ShardHandle::create(ShardKey::new("th"), &path).expect("create");

        assert_eq!(shard.lockfree_entry_count(), 0);

        shard.increment_lockfree(b"the|quick", 1).unwrap();
        shard.increment_lockfree(b"the|brown", 1).unwrap();
        shard.increment_lockfree(b"the|fox", 1).unwrap();

        assert_eq!(shard.lockfree_entry_count(), 3);
    }

    #[test]
    fn test_lockfree_entry_count_resets_on_sync() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("test_shard.artrie");
        let shard = ShardHandle::create(ShardKey::new("th"), &path).expect("create");

        shard.increment_lockfree(b"the|quick", 1).unwrap();
        shard.increment_lockfree(b"the|brown", 1).unwrap();
        assert_eq!(shard.lockfree_entry_count(), 2);

        shard.sync().expect("sync");
        assert_eq!(
            shard.lockfree_entry_count(),
            0,
            "sync() merges lock-free overlay into persistent and should reset the counter"
        );
    }

    #[test]
    fn test_lockfree_entry_count_resets_on_flush_lockfree() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("test_shard.artrie");
        let shard = ShardHandle::create(ShardKey::new("th"), &path).expect("create");

        shard.increment_lockfree(b"the|quick", 1).unwrap();
        shard.increment_lockfree(b"the|brown", 1).unwrap();
        shard.increment_lockfree(b"the|fox", 1).unwrap();
        assert_eq!(shard.lockfree_entry_count(), 3);

        shard.flush_lockfree().expect("flush_lockfree");
        assert_eq!(
            shard.lockfree_entry_count(),
            0,
            "flush_lockfree() should reset the counter"
        );
    }

    #[test]
    fn test_lockfree_entry_count_resets_on_checkpoint() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("test_shard.artrie");
        let shard = ShardHandle::create(ShardKey::new("th"), &path).expect("create");

        shard.increment_lockfree(b"the|quick", 1).unwrap();
        assert_eq!(shard.lockfree_entry_count(), 1);

        shard.checkpoint().expect("checkpoint");
        assert_eq!(
            shard.lockfree_entry_count(),
            0,
            "checkpoint() should reset the counter"
        );
    }

    // ---- Chunked commit (commit_chunk) ----

    #[test]
    fn test_commit_chunk_persists_data() {
        // commit_chunk persists buffered n-grams to the WAL but must NOT mark
        // the prefix as complete. This is essential for the crash-recovery
        // contract documented on `ShardHandle::commit_chunk`.
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("test_shard.artrie");
        let mut shard = ShardHandle::create(ShardKey::new("th"), &path).expect("create");

        let mut tx = shard.begin_prefix("th").expect("begin_prefix");
        for n in 0..5 {
            let key = format!("the|word{}", n);
            shard.tx_insert(&mut tx, key.as_bytes(), 100 + n as u64);
        }

        let inserted = shard.commit_chunk(tx).expect("commit_chunk");
        assert_eq!(inserted, 5, "should commit all 5 buffered n-grams");

        // Data is queryable after chunk commit
        for n in 0..5 {
            let key = format!("the|word{}", n);
            assert_eq!(shard.get(key.as_bytes()), Some(100 + n as u64));
        }

        // Prefix is NOT yet marked complete
        assert!(
            !shard.checkpoint_state().completed_prefixes.contains("th"),
            "commit_chunk must NOT mark the prefix as complete — that's commit_prefix's job"
        );
    }

    #[test]
    fn test_commit_chunk_then_commit_prefix_marks_complete() {
        // Multi-chunk workflow: commit_chunk for intermediate chunks, then
        // commit_prefix on the final chunk to mark the prefix complete.
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("test_shard.artrie");
        let key = ShardKey::new("th");

        {
            let mut shard = ShardHandle::create(key.clone(), &path).expect("create");

            // Chunk 1: insert 5 entries, commit_chunk
            let mut tx1 = shard.begin_prefix("th").expect("begin_prefix");
            for n in 0..5 {
                let k = format!("the|word{}", n);
                shard.tx_insert(&mut tx1, k.as_bytes(), 100 + n as u64);
            }
            shard.commit_chunk(tx1).expect("commit_chunk 1");

            // Chunk 2: insert 3 more entries, commit_prefix (final)
            let mut tx2 = shard.begin_prefix("th").expect("begin_prefix");
            for n in 5..8 {
                let k = format!("the|word{}", n);
                shard.tx_insert(&mut tx2, k.as_bytes(), 100 + n as u64);
            }
            let inserted = shard.commit_prefix(tx2).expect("commit_prefix");
            assert_eq!(inserted, 3);

            // All 8 entries should be present
            for n in 0..8 {
                let k = format!("the|word{}", n);
                assert_eq!(shard.get(k.as_bytes()), Some(100 + n as u64));
            }

            // Now the prefix IS marked complete
            assert!(
                shard.checkpoint_state().completed_prefixes.contains("th"),
                "commit_prefix on the final chunk should mark the prefix complete"
            );
        }

        // Reopen and verify checkpoint state + data persist
        {
            let shard = ShardHandle::open(key, &path).expect("open");
            assert!(shard.checkpoint_state().completed_prefixes.contains("th"));
            for n in 0..8 {
                let k = format!("the|word{}", n);
                assert_eq!(shard.get(k.as_bytes()), Some(100 + n as u64));
            }
        }
    }

    #[test]
    fn test_commit_chunk_handles_large_batch_and_varint_boundary() {
        // Diagnostic: insert 200 keys that mimic vocab-encoded n-gram keys,
        // including the LEB128 1-byte → 2-byte boundary at index 128. The
        // shard's persistent ART must handle arbitrary byte sequences and
        // arbitrary commit-batch sizes.
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("test_shard.artrie");
        let mut shard = ShardHandle::create(ShardKey::new("th"), &path).expect("create");

        // Keys: [0x01, idx_varint] mimicking ["the" → 1, word_N → N+2]
        fn encode_varint(mut v: u64, out: &mut Vec<u8>) {
            loop {
                let b = (v & 0x7f) as u8;
                v >>= 7;
                if v == 0 {
                    out.push(b);
                    break;
                } else {
                    out.push(b | 0x80);
                }
            }
        }

        let mut tx = shard.begin_prefix("th").expect("begin_prefix");
        let mut keys = Vec::new();
        for i in 0..200u64 {
            let mut key = vec![0x01u8];
            encode_varint(i + 2, &mut key); // start at index 2 (1 is "the")
            shard.tx_insert(&mut tx, &key, 1000 + i);
            keys.push(key);
        }
        let inserted = shard.commit_prefix(tx).expect("commit_prefix");
        assert_eq!(inserted, 200);

        let mut missing = Vec::new();
        for (i, key) in keys.iter().enumerate() {
            if shard.get(key) != Some(1000 + i as u64) {
                missing.push((i, key.clone(), shard.get(key)));
            }
        }
        assert!(
            missing.is_empty(),
            "shard-level: missing {} keys: first: {:?}",
            missing.len(),
            &missing[..missing.len().min(3)]
        );
    }

    #[test]
    fn test_commit_chunk_set_semantics_idempotent() {
        // The crash-recovery contract: SET semantics make re-inserting the
        // same (key, value) idempotent. After two identical commit_chunk
        // sequences, get() returns the value (not 2x the value).
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("test_shard.artrie");
        let mut shard = ShardHandle::create(ShardKey::new("th"), &path).expect("create");

        // First commit_chunk
        let mut tx1 = shard.begin_prefix("th").expect("begin_prefix");
        shard.tx_insert(&mut tx1, b"the|fox", 10);
        shard.commit_chunk(tx1).expect("commit_chunk 1");
        assert_eq!(shard.get(b"the|fox"), Some(10));

        // Identical re-commit (simulating a resume after crash)
        let mut tx2 = shard.begin_prefix("th").expect("begin_prefix");
        shard.tx_insert(&mut tx2, b"the|fox", 10);
        shard.commit_chunk(tx2).expect("commit_chunk 2");

        // SET semantics: value remains 10, not 20
        assert_eq!(
            shard.get(b"the|fox"),
            Some(10),
            "commit_chunk uses SET semantics — re-inserting the same value must not double it"
        );
    }

    #[test]
    fn test_overlay_eviction_is_lossless_and_observable() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("evict_shard.artrie");
        let shard = ShardHandle::create(ShardKey::new("th"), &path).expect("create");

        // Tiny resident budget so a modest insert provably exceeds it and the
        // checkpoint tail evicts cold overlay nodes. (min_eviction_depth pins the
        // shallow fan-out, so the budget itself may be structurally unreachable —
        // we assert eviction OCCURRED and was lossless, not a strict residency.)
        shard
            .arm_eviction(Some(EvictionConfig {
                resident_budget_bytes: Some(4096),
                ..EvictionConfig::without_memory_monitor()
            }))
            .expect("arm eviction");

        const N: u64 = 2000;
        for i in 0..N {
            shard
                .increment_lockfree(format!("th|w{:05}", i).as_bytes(), i + 1)
                .expect("increment");
        }
        // #1 registers the overlay; #2's tail evicts the now-cold nodes to budget.
        shard.checkpoint().expect("checkpoint 1");
        shard.checkpoint().expect("checkpoint 2");

        // Observable (libdictenstein e2f7681 records the checkpoint-tail eviction).
        let stats = shard.eviction_stats();
        assert!(
            stats.nodes_evicted > 0,
            "budget eviction should have reclaimed cold overlay nodes (nodes_evicted={})",
            stats.nodes_evicted
        );

        // Lossless: every evicted value still faults back on read with its count.
        for i in 0..N {
            assert_eq!(
                shard.get(format!("th|w{:05}", i).as_bytes()),
                Some(i + 1),
                "evicted key th|w{:05} must fault back losslessly",
                i
            );
        }
    }

    #[test]
    fn test_overlay_eviction_under_concurrent_writers() {
        // Red-team coverage the loom proofs do not reach: many lock-free
        // increment_cas writers per shard racing the checkpoint-tail budget
        // eviction (root-CAS unswizzle + 1c stamp guard). Unique keys => each
        // final count is deterministically 1 regardless of interleaving.
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("evict_concurrent.artrie");
        let shard = Arc::new(ShardHandle::create(ShardKey::new("th"), &path).expect("create"));
        shard
            .arm_eviction(Some(EvictionConfig {
                resident_budget_bytes: Some(4096),
                ..EvictionConfig::without_memory_monitor()
            }))
            .expect("arm eviction");

        const WRITERS: u64 = 4;
        const PER_WRITER: u64 = 500;

        let writers: Vec<_> = (0..WRITERS)
            .map(|w| {
                let shard = Arc::clone(&shard);
                std::thread::spawn(move || {
                    for i in 0..PER_WRITER {
                        shard
                            .increment_lockfree(format!("th|w{}_{:04}", w, i).as_bytes(), 1)
                            .expect("concurrent increment");
                    }
                })
            })
            .collect();

        let checkpointer = {
            let shard = Arc::clone(&shard);
            std::thread::spawn(move || {
                for _ in 0..4 {
                    shard.checkpoint().expect("concurrent checkpoint");
                }
            })
        };

        for writer in writers {
            writer.join().expect("writer thread panicked");
        }
        checkpointer.join().expect("checkpoint thread panicked");

        // Final checkpoint, then verify no write was lost under concurrent eviction.
        shard.checkpoint().expect("final checkpoint");
        for w in 0..WRITERS {
            for i in 0..PER_WRITER {
                let key = format!("th|w{}_{:04}", w, i);
                assert_eq!(
                    shard.get(key.as_bytes()),
                    Some(1),
                    "write {} lost under concurrent budget eviction",
                    key
                );
            }
        }
    }
}
