//! Individual shard wrapper around PersistentARTrie.
//!
//! Each shard manages a subset of n-grams based on prefix routing.
//! Shards provide exclusive write access via WriteToken.

use super::routing::ShardKey;
use libdictenstein::persistent_artrie::wal::SyncHandle;
use libdictenstein::persistent_artrie::wal_managed::WalManaged;
use libdictenstein::persistent_artrie::{
    DocumentTransaction, PersistentARTrie,
};
use liblevenshtein::dictionary::Dictionary;
use parking_lot::{Condvar, Mutex};
use std::collections::HashSet;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Error type for shard operations.
#[derive(Error, Debug)]
pub enum ShardError {
    /// Failed to create or open the shard file.
    #[error("Failed to create/open shard at {path}: {message}")]
    Open { path: PathBuf, message: String },

    /// Read operation failed.
    #[error("Read failed for shard {shard_key}: {message}")]
    Read { shard_key: String, message: String },

    /// Write operation failed.
    #[error("Write failed for shard {shard_key}: {message}")]
    Write { shard_key: String, message: String },

    /// Checkpoint operation failed.
    #[error("Checkpoint failed for shard {shard_key}: {message}")]
    Checkpoint { shard_key: String, message: String },

    /// Shard is locked by another writer.
    #[error("Shard {shard_key} is locked by worker {holder}")]
    Locked { shard_key: String, holder: usize },

    /// Writer token is invalid or expired.
    #[error("Invalid write token for shard {shard_key}")]
    InvalidToken { shard_key: String },

    /// Sync operation failed.
    #[error("Sync failed for shard {shard_key}: {message}")]
    Sync { shard_key: String, message: String },

    /// Sync operation timed out.
    #[error("Sync timed out for shard {shard_key}")]
    SyncTimeout { shard_key: String },
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

/// Write token for exclusive shard access.
///
/// Ensures single-writer constraint per shard. A worker must hold
/// a WriteToken to perform write operations on a shard.
///
/// # Thread Safety
///
/// `WriteToken` is intentionally `!Send` (via `PhantomData<*const ()>`).
/// This compile-time constraint ensures tokens cannot be passed between
/// threads, which allows the use of `Relaxed` ordering when reading
/// `write_generation` in `release_write()`. Since the token can only be
/// used on the thread that acquired it, program order guarantees visibility
/// of the generation counter without explicit synchronization.
///
/// This design is formally verified in `formal/tla/ShardWriteToken.tla`.
#[derive(Debug)]
pub struct WriteToken {
    /// The shard this token grants access to.
    pub shard_key: ShardKey,

    /// When the token was acquired.
    pub acquired_at: Instant,

    /// ID of the worker holding this token.
    pub worker_id: usize,

    /// Generation counter to detect stale tokens.
    generation: u64,

    /// Marker to make WriteToken `!Send`.
    ///
    /// This ensures `Relaxed` ordering on `write_generation` reads is correct
    /// because tokens cannot be passed between threads. Within a single thread,
    /// program order ensures visibility of the generation counter.
    _not_send: PhantomData<*const ()>,
}

impl WriteToken {
    /// Create a new write token.
    fn new(shard_key: ShardKey, worker_id: usize, generation: u64) -> Self {
        Self {
            shard_key,
            acquired_at: Instant::now(),
            worker_id,
            generation,
            _not_send: PhantomData,
        }
    }

    /// Check if this token is valid for the given shard and generation.
    fn is_valid(&self, shard_key: &ShardKey, current_generation: u64) -> bool {
        &self.shard_key == shard_key && self.generation == current_generation
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
        self.inner.wait_timeout(timeout).map_err(|e| ShardError::Sync {
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

    /// The underlying trie (byte-keyed).
    trie: PersistentARTrie<u64>,

    /// File path for this shard.
    path: PathBuf,

    /// Checkpoint state for this shard.
    checkpoint_state: ShardCheckpointState,

    /// Write lock: true if a writer holds the lock.
    write_locked: AtomicBool,

    /// ID of the worker holding the write lock (if any).
    write_holder: AtomicUsize,

    /// Generation counter for write tokens.
    write_generation: AtomicU64,

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

        // Use create_with_slot_tracking for optimized incremental checkpoints
        let mut trie =
            PersistentARTrie::create_with_slot_tracking(&path).map_err(|e| ShardError::Open {
                path: path.clone(),
                message: e.to_string(),
            })?;

        // Enable lock-free overlay for concurrent CAS-based increments
        trie.enable_lockfree();

        Ok(Self {
            key,
            trie,
            path,
            checkpoint_state: ShardCheckpointState::default(),
            write_locked: AtomicBool::new(false),
            write_holder: AtomicUsize::new(usize::MAX),
            write_generation: AtomicU64::new(0),
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

        let (mut trie, recovery_report) =
            PersistentARTrie::open_with_recovery_and_slot_tracking(&path).map_err(|e| {
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

        // Enable lock-free overlay for concurrent CAS-based increments
        trie.enable_lockfree();

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
            trie,
            path,
            checkpoint_state: ShardCheckpointState::default(),
            write_locked: AtomicBool::new(false),
            write_holder: AtomicUsize::new(usize::MAX),
            write_generation: AtomicU64::new(0),
            stats: ShardStats::default(),
            sync_coordinator: ShardSyncCoordinator::new(),
            lockfree_entries: AtomicU64::new(0),
        };

        // Load checkpoint state from trie
        handle.load_checkpoint_state()?;
        handle.stats.set_entry_count(handle.trie.len().unwrap_or(0) as u64);

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
        self.stats.entry_count.load(std::sync::atomic::Ordering::Relaxed) as usize
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

    /// Try to acquire exclusive write access.
    ///
    /// Returns `Some(WriteToken)` if successful, `None` if another worker
    /// holds the lock.
    pub fn try_acquire_write(&self, worker_id: usize) -> Option<WriteToken> {
        // Try to set write_locked from false to true
        if self
            .write_locked
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            // We got the lock
            self.write_holder.store(worker_id, Ordering::Relaxed);
            let generation = self.write_generation.fetch_add(1, Ordering::Relaxed);
            Some(WriteToken::new(self.key.clone(), worker_id, generation + 1))
        } else {
            None
        }
    }

    /// Release exclusive write access.
    ///
    /// Returns `true` if the token was valid and the lock was released.
    ///
    /// # Memory Ordering
    ///
    /// Uses `Relaxed` ordering for loading the generation counter. This is safe
    /// because `WriteToken` is `!Send`:
    ///
    /// 1. The token can only be used on the thread that acquired it
    /// 2. Within a single thread, program order ensures the generation read
    ///    sees the value written during `try_acquire_write()`
    /// 3. The subsequent `Release` store on `write_locked` publishes all writes
    ///    to the next acquirer via the Acquire-Release pair on the CAS
    ///
    /// This optimization is formally verified in `formal/tla/ShardWriteToken.tla`.
    pub fn release_write(&self, token: WriteToken) -> bool {
        // Relaxed is safe because WriteToken is !Send (PhantomData<*const ()>).
        // The token cannot be passed between threads, so program order guarantees
        // we see the generation value from our own try_acquire_write() call.
        let current_gen = self.write_generation.load(Ordering::Relaxed);
        if token.is_valid(&self.key, current_gen) {
            self.write_holder.store(usize::MAX, Ordering::Relaxed);
            self.write_locked.store(false, Ordering::Release);
            true
        } else {
            false
        }
    }

    /// Check if the shard is currently write-locked.
    pub fn is_write_locked(&self) -> bool {
        self.write_locked.load(Ordering::Relaxed)
    }

    /// Get the worker ID holding the write lock (if any).
    pub fn write_holder(&self) -> Option<usize> {
        if self.is_write_locked() {
            let holder = self.write_holder.load(Ordering::Relaxed);
            if holder != usize::MAX {
                return Some(holder);
            }
        }
        None
    }

    /// Increment an n-gram count (requires write token).
    ///
    /// The caller must hold a valid WriteToken for this shard.
    pub fn increment(
        &mut self,
        ngram: &[u8],
        count: u64,
        _token: &WriteToken,
    ) -> ShardResult<bool> {
        let was_new = self.trie.get_value_bytes(ngram).is_none();

        self.trie
            .increment_bytes(ngram, count as i64)
            .map_err(|e| ShardError::Write {
                shard_key: self.key.to_string(),
                message: e.to_string(),
            })?;

        self.stats.record_write();
        if was_new {
            self.stats.add_entries(1);
        }

        // Mark shard as dirty after write
        self.sync_coordinator.mark_dirty();

        Ok(was_new)
    }

    /// Lock-free increment using CAS. Only needs `&self` (shared access).
    ///
    /// Uses the lock-free overlay's `increment_cas` — no WriteToken or
    /// exclusive RwLock required, so multiple workers can increment the
    /// same shard concurrently without serialization.
    pub fn increment_lockfree(&self, ngram: &[u8], count: u64) -> ShardResult<bool> {
        let lockfree_exists = self.trie.get_lockfree(ngram).is_some();
        let persistent_exists = self.trie.get_value_bytes(ngram).is_some();
        let was_new = !lockfree_exists && !persistent_exists;

        self.trie.increment_cas(ngram, count);

        self.stats.record_write();
        if was_new {
            self.stats.add_entries(1);
            self.lockfree_entries.fetch_add(1, Ordering::Relaxed);
        }

        self.sync_coordinator.mark_dirty();

        Ok(was_new)
    }

    /// Get the count for an n-gram (dual-layer: lock-free overlay + persistent).
    pub fn get(&self, ngram: &[u8]) -> Option<u64> {
        self.stats.record_read();
        let lockfree_val = self.trie.get_lockfree(ngram).unwrap_or(0);
        let persistent_val = self.trie.get_value_bytes(ngram).unwrap_or(0);
        let total = lockfree_val + persistent_val;
        if total > 0 {
            Some(total)
        } else {
            None
        }
    }

    /// Check if an n-gram exists (dual-layer: lock-free overlay + persistent).
    pub fn contains(&self, ngram: &[u8]) -> bool {
        self.trie.get_lockfree(ngram).is_some() || self.trie.contains_bytes(ngram)
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

    /// Sync WAL to disk (merges lock-free overlay first).
    pub fn sync(&mut self) -> ShardResult<()> {
        self.trie
            .merge_lockfree_values_to_persistent()
            .map_err(|e| ShardError::Checkpoint {
                shard_key: self.key.to_string(),
                message: format!("merge before sync failed: {}", e),
            })?;
        self.lockfree_entries.store(0, Ordering::Relaxed);
        self.trie.checkpoint().map_err(|e| ShardError::Checkpoint {
            shard_key: self.key.to_string(),
            message: format!("sync failed: {}", e),
        })
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
    pub fn sync_tracked(&mut self) -> ShardResult<bool> {
        // Try to start sync (CAS: Dirty -> Syncing)
        if !self.sync_coordinator.try_start_sync() {
            // Either clean (no sync needed) or already syncing
            return Ok(false);
        }

        // Merge lock-free overlay into persistent trie before syncing
        if let Err(e) = self.trie.merge_lockfree_values_to_persistent() {
            let error_msg = format!("merge before sync failed: {}", e);
            self.sync_coordinator.fail_sync(&error_msg);
            return Err(ShardError::Sync {
                shard_key: self.key.to_string(),
                message: error_msg,
            });
        }
        self.lockfree_entries.store(0, Ordering::Relaxed);

        // Perform the actual sync
        match self.trie.sync() {
            Ok(()) => {
                // Success: mark clean and notify waiters
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
        self.sync_coordinator.wait_for_sync(timeout).map_err(|()| {
            ShardError::SyncTimeout {
                shard_key: self.key.to_string(),
            }
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

    /// Merge the lock-free overlay into the persistent trie.
    ///
    /// Call this before operations that iterate the persistent trie
    /// (e.g., merge, query, MKN computation) to ensure they see all data.
    /// Point lookups (`get`, `contains`) already check both layers.
    pub fn flush_lockfree(&mut self) -> ShardResult<()> {
        self.trie
            .merge_lockfree_values_to_persistent()
            .map_err(|e| ShardError::Checkpoint {
                shard_key: self.key.to_string(),
                message: format!("flush_lockfree failed: {}", e),
            })?;
        self.lockfree_entries.store(0, Ordering::Relaxed);
        Ok(())
    }

    /// Checkpoint the shard (persist to disk and truncate WAL).
    ///
    /// Uses sequential flush for optimized disk I/O (5-15% faster checkpoints).
    pub fn checkpoint(&mut self) -> ShardResult<()> {
        // Merge lock-free overlay into persistent trie before checkpointing
        self.trie
            .merge_lockfree_values_to_persistent()
            .map_err(|e| ShardError::Checkpoint {
                shard_key: self.key.to_string(),
                message: format!("merge before checkpoint failed: {}", e),
            })?;
        self.lockfree_entries.store(0, Ordering::Relaxed);

        // Save checkpoint state to trie
        self.save_checkpoint_state()?;

        // Flush dirty arenas in sequential order for optimal I/O
        self.trie.flush_sequential().map_err(|e| ShardError::Checkpoint {
            shard_key: self.key.to_string(),
            message: format!("flush_sequential failed: {}", e),
        })?;

        // Checkpoint the trie
        self.trie.checkpoint().map_err(|e| ShardError::Checkpoint {
            shard_key: self.key.to_string(),
            message: e.to_string(),
        })
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
    fn save_checkpoint_state(&mut self) -> ShardResult<()> {
        // Save n-grams processed count
        let ngrams_key = format!("{}ngrams_processed", Self::CHECKPOINT_PREFIX);
        self.trie
            .upsert_bytes(ngrams_key.as_bytes(), self.checkpoint_state.ngrams_processed)
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
    pub fn persist_checkpoint_state(&mut self) -> ShardResult<()> {
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
        let tx = self.trie.begin_document(&document_id).map_err(|e| ShardError::Write {
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

        let inserted = self.trie.commit_document(tx.tx).map_err(|e| ShardError::Write {
            shard_key: self.key.to_string(),
            message: format!("Failed to commit transaction for prefix '{}': {}", prefix, e),
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
            self.key, prefix, ngram_count, inserted
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

        let inserted = self.trie.commit_document(tx.tx).map_err(|e| ShardError::Write {
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
            self.key, prefix, ngram_count, inserted
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
        self.trie.abort_document(tx.tx).map_err(|e| ShardError::Write {
            shard_key: self.key.to_string(),
            message: format!("Failed to abort transaction for prefix '{}': {}", prefix, e),
        })?;

        log::trace!(
            "Shard {}: aborted prefix '{}' transaction ({} n-grams discarded)",
            self.key, prefix, tx.ngram_count
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

        let mut shard = ShardHandle::create(key.clone(), &path).expect("Failed to create shard");

        // Acquire write token
        let token = shard.try_acquire_write(0).expect("Failed to acquire write");

        // Write some data
        let was_new = shard
            .increment(b"the|quick", 5, &token)
            .expect("Failed to increment");
        assert!(was_new);

        let was_new = shard
            .increment(b"the|quick", 3, &token)
            .expect("Failed to increment");
        assert!(!was_new);

        // Read back
        assert_eq!(shard.get(b"the|quick"), Some(8));
        assert_eq!(shard.len(), 1);

        // Release token
        assert!(shard.release_write(token));
    }

    #[test]
    fn test_write_token_exclusivity() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("test_shard.artrie");
        let key = ShardKey::new("th");

        let shard = ShardHandle::create(key, &path).expect("Failed to create shard");

        // First worker acquires
        let token1 = shard.try_acquire_write(0).expect("Failed to acquire");
        assert!(shard.is_write_locked());
        assert_eq!(shard.write_holder(), Some(0));

        // Second worker cannot acquire
        assert!(shard.try_acquire_write(1).is_none());

        // Release
        assert!(shard.release_write(token1));
        assert!(!shard.is_write_locked());

        // Now second worker can acquire
        let token2 = shard.try_acquire_write(1).expect("Failed to acquire");
        assert_eq!(shard.write_holder(), Some(1));
        shard.release_write(token2);
    }

    #[test]
    fn test_shard_persistence() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("test_shard.artrie");
        let key = ShardKey::new("th");

        // Create and write
        {
            let mut shard =
                ShardHandle::create(key.clone(), &path).expect("Failed to create shard");
            let token = shard.try_acquire_write(0).unwrap();
            shard.increment(b"the|quick", 10, &token).unwrap();
            shard.sync().unwrap();
            shard.release_write(token);
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
        let mut shard = ShardHandle::open_or_create(key.clone(), &path)
            .expect("Failed to open_or_create shard");
        assert!(path.exists());

        // Write data
        let token = shard.try_acquire_write(0).unwrap();
        shard.increment(b"apple|pie", 5, &token).unwrap();
        shard.sync().unwrap();
        shard.release_write(token);

        assert_eq!(shard.get(b"apple|pie"), Some(5));
    }

    #[test]
    fn test_open_or_create_existing_shard() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("existing_shard.artrie");
        let key = ShardKey::new("cd");

        // Create initial shard with data
        {
            let mut shard = ShardHandle::create(key.clone(), &path)
                .expect("Failed to create shard");
            let token = shard.try_acquire_write(0).unwrap();
            shard.increment(b"cat|dog", 7, &token).unwrap();
            shard.sync().unwrap();
            shard.release_write(token);
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

        let mut shard = ShardHandle::create(key, &path).expect("Failed to create shard");

        // Initially clean
        assert_eq!(shard.sync_state(), ShardSyncState::Clean);

        // Write marks dirty
        let token = shard.try_acquire_write(0).expect("Failed to acquire write");
        shard.increment(b"the|quick", 5, &token).expect("Failed to increment");
        assert_eq!(shard.sync_state(), ShardSyncState::Dirty);
        shard.release_write(token);

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
            let mut shard = ShardHandle::create(key.clone(), &path).expect("Failed to create shard");

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
            let mut shard = ShardHandle::create(key.clone(), &path).expect("Failed to create shard");

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
            let mut shard = ShardHandle::create(key.clone(), &path).expect("Failed to create shard");

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
}
