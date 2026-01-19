//! Individual shard wrapper around DiskBackedCharTrieInner.
//!
//! Each shard manages a subset of n-grams based on prefix routing.
//! Shards provide exclusive write access via WriteToken.

use super::routing::ShardKey;
use liblevenshtein::dictionary::persistent_artrie_char::DiskBackedCharTrieInner;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;
use thiserror::Error;

/// Error type for shard operations.
#[derive(Error, Debug)]
pub enum ShardError {
    /// Failed to create or open the shard file.
    #[error("Failed to create/open shard at {path}: {message}")]
    Open { path: PathBuf, message: String },

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
}

/// Result type for shard operations.
pub type ShardResult<T> = Result<T, ShardError>;

/// Write token for exclusive shard access.
///
/// Ensures single-writer constraint per shard. A worker must hold
/// a WriteToken to perform write operations on a shard.
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
}

impl WriteToken {
    /// Create a new write token.
    fn new(shard_key: ShardKey, worker_id: usize, generation: u64) -> Self {
        Self {
            shard_key,
            acquired_at: Instant::now(),
            worker_id,
            generation,
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

/// Handle to an individual shard.
///
/// Wraps a `DiskBackedCharTrieInner<u64>` with checkpoint state and
/// exclusive write access control.
pub struct ShardHandle {
    /// The shard key identifying this shard.
    key: ShardKey,

    /// The underlying trie.
    trie: DiskBackedCharTrieInner<u64>,

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
}

impl ShardHandle {
    /// Reserved key prefix for checkpoint data within the trie.
    const CHECKPOINT_PREFIX: &'static str = "\x00__shard_ckpt__:";

    /// Create a new shard at the given path.
    ///
    /// Creates a new trie file, overwriting if it exists.
    pub fn create(key: ShardKey, path: impl AsRef<Path>) -> ShardResult<Self> {
        let path = path.as_ref().to_path_buf();

        let trie = DiskBackedCharTrieInner::create(&path).map_err(|e| ShardError::Open {
            path: path.clone(),
            message: e.to_string(),
        })?;

        Ok(Self {
            key,
            trie,
            path,
            checkpoint_state: ShardCheckpointState::default(),
            write_locked: AtomicBool::new(false),
            write_holder: AtomicUsize::new(usize::MAX),
            write_generation: AtomicU64::new(0),
            stats: ShardStats::default(),
        })
    }

    /// Open an existing shard with automatic crash recovery.
    pub fn open(key: ShardKey, path: impl AsRef<Path>) -> ShardResult<Self> {
        let path = path.as_ref().to_path_buf();

        let (trie, recovery_report) =
            DiskBackedCharTrieInner::open_with_recovery(&path).map_err(|e| ShardError::Open {
                path: path.clone(),
                message: e.to_string(),
            })?;

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
        };

        // Load checkpoint state from trie
        handle.load_checkpoint_state()?;
        handle.stats.set_entry_count(handle.trie.len as u64);

        Ok(handle)
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
        self.trie.len
    }

    /// Check if the shard is empty.
    pub fn is_empty(&self) -> bool {
        self.trie.len == 0
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
    pub fn release_write(&self, token: WriteToken) -> bool {
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
        ngram: &str,
        count: u64,
        _token: &WriteToken,
    ) -> ShardResult<bool> {
        let was_new = self.trie.get(ngram).is_none();

        self.trie
            .increment(ngram, count as i64)
            .map_err(|e| ShardError::Write {
                shard_key: self.key.to_string(),
                message: e.to_string(),
            })?;

        self.stats.record_write();
        if was_new {
            self.stats.add_entries(1);
        }

        Ok(was_new)
    }

    /// Get the count for an n-gram.
    pub fn get(&self, ngram: &str) -> Option<u64> {
        self.stats.record_read();
        self.trie.get(ngram).map(|v| *v as u64)
    }

    /// Check if an n-gram exists.
    pub fn contains(&self, ngram: &str) -> bool {
        self.trie.contains(ngram)
    }

    /// Iterate over all n-grams with their counts.
    pub fn iter_with_counts(&self) -> impl Iterator<Item = (String, u64)> {
        self.trie
            .iter_prefix_with_values("")
            .ok()
            .flatten()
            .unwrap_or_default()
            .into_iter()
            .filter(|(k, _)| !k.starts_with(Self::CHECKPOINT_PREFIX))
            .map(|(k, v)| (k, v as u64))
    }

    /// Sync WAL to disk.
    pub fn sync(&mut self) -> ShardResult<()> {
        self.trie.sync().map_err(|e| ShardError::Checkpoint {
            shard_key: self.key.to_string(),
            message: format!("sync failed: {}", e),
        })
    }

    /// Checkpoint the shard (persist to disk and truncate WAL).
    pub fn checkpoint(&mut self) -> ShardResult<()> {
        // Save checkpoint state to trie
        self.save_checkpoint_state()?;

        // Checkpoint the trie
        self.trie.checkpoint().map_err(|e| ShardError::Checkpoint {
            shard_key: self.key.to_string(),
            message: e.to_string(),
        })
    }

    /// Mark a prefix as completed in this shard.
    pub fn complete_prefix(&mut self, prefix: &str) {
        self.checkpoint_state
            .completed_prefixes
            .insert(prefix.to_string());
        self.checkpoint_state.current_prefix = None;
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
        // Load completed prefixes
        let completed_key = format!("{}completed", Self::CHECKPOINT_PREFIX);
        if let Some(value) = self.trie.get(&completed_key) {
            // Value encodes the count of completed prefixes
            let count = *value as usize;
            for i in 0..count {
                let prefix_key = format!("{}completed:{}", Self::CHECKPOINT_PREFIX, i);
                // The prefix is stored as a separate key (we'd need string storage)
                // For now, we'll reconstruct from iteration
                if self.trie.contains(&prefix_key) {
                    // In a real implementation, we'd store the prefix string
                    // For now, this is a placeholder
                }
            }
        }

        // Load n-grams processed count
        let ngrams_key = format!("{}ngrams_processed", Self::CHECKPOINT_PREFIX);
        if let Some(value) = self.trie.get(&ngrams_key) {
            self.checkpoint_state.ngrams_processed = *value as u64;
        }

        Ok(())
    }

    /// Save checkpoint state to the trie.
    fn save_checkpoint_state(&mut self) -> ShardResult<()> {
        // Save n-grams processed count
        let ngrams_key = format!("{}ngrams_processed", Self::CHECKPOINT_PREFIX);
        self.trie
            .upsert(&ngrams_key, self.checkpoint_state.ngrams_processed)
            .map_err(|e| ShardError::Checkpoint {
                shard_key: self.key.to_string(),
                message: format!("failed to save ngrams_processed: {}", e),
            })?;

        // Save completed prefix count
        let completed_key = format!("{}completed", Self::CHECKPOINT_PREFIX);
        self.trie
            .upsert(
                &completed_key,
                self.checkpoint_state.completed_prefixes.len() as u64,
            )
            .map_err(|e| ShardError::Checkpoint {
                shard_key: self.key.to_string(),
                message: format!("failed to save completed count: {}", e),
            })?;

        Ok(())
    }
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
            .increment("the|quick", 5, &token)
            .expect("Failed to increment");
        assert!(was_new);

        let was_new = shard
            .increment("the|quick", 3, &token)
            .expect("Failed to increment");
        assert!(!was_new);

        // Read back
        assert_eq!(shard.get("the|quick"), Some(8));
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
            shard.increment("the|quick", 10, &token).unwrap();
            shard.sync().unwrap();
            shard.release_write(token);
        }

        // Reopen and verify
        {
            let shard = ShardHandle::open(key, &path).expect("Failed to open shard");
            assert_eq!(shard.get("the|quick"), Some(10));
        }
    }
}
