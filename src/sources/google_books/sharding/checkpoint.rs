//! Global checkpoint coordination for sharded imports.
//!
//! Provides coordinated checkpointing across all shards with crash recovery.
//! Each shard maintains its own WAL for local recovery, while the global
//! checkpoint coordinates the overall import state.
//!
//! # Checkpoint Structure
//!
//! ```text
//! GlobalCheckpoint
//! ├── version: u32
//! ├── created_at: DateTime
//! ├── last_updated: DateTime
//! ├── import_state: ImportState
//! ├── shard_records: HashMap<ShardKey, ShardCheckpointRecord>
//! └── metadata: HashMap<String, String>
//! ```
//!
//! # Recovery Flow
//!
//! 1. Load global checkpoint (if exists)
//! 2. Verify each shard's state matches the global checkpoint
//! 3. Mark any in-progress shards for recovery
//! 4. Resume from last known good state

use super::routing::ShardKey;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use thiserror::Error;

/// Current checkpoint format version.
const CHECKPOINT_VERSION: u32 = 1;

/// Error type for checkpoint operations.
#[derive(Error, Debug)]
pub enum CheckpointError {
    /// I/O error during checkpoint operations.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Checkpoint version mismatch.
    #[error("Checkpoint version mismatch: expected {expected}, found {found}")]
    VersionMismatch { expected: u32, found: u32 },

    /// Checkpoint integrity error.
    #[error("Checkpoint integrity error: {0}")]
    Integrity(String),

    /// Shard state inconsistency.
    #[error("Shard {shard_key} state inconsistency: {message}")]
    ShardInconsistency { shard_key: String, message: String },
}

/// Result type for checkpoint operations.
pub type CheckpointResult<T> = Result<T, CheckpointError>;

/// Overall import state.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ImportState {
    /// Import has not started.
    NotStarted,

    /// Import is in progress.
    InProgress {
        /// When the import started.
        started_at: u64,

        /// Current phase of import.
        phase: ImportPhase,
    },

    /// Import completed successfully.
    Completed {
        /// When the import completed.
        completed_at: u64,

        /// Total n-grams imported.
        total_ngrams: u64,

        /// Total unique n-grams.
        unique_ngrams: u64,
    },

    /// Import failed with an error.
    Failed {
        /// When the failure occurred.
        failed_at: u64,

        /// Error message.
        error: String,
    },

    /// Import was interrupted and needs recovery.
    RequiresRecovery {
        /// Last checkpoint timestamp.
        last_checkpoint_at: u64,

        /// Shards that were in progress when interrupted.
        in_progress_shards: Vec<String>,
    },
}

/// Import phase for progress tracking.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ImportPhase {
    /// Downloading and parsing n-gram files.
    Importing,

    /// Computing MKN statistics.
    ComputingMkn,

    /// Merging shards into PathMap.
    Merging,

    /// Finalizing and cleanup.
    Finalizing,
}

/// Per-shard checkpoint record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardCheckpointRecord {
    /// Shard prefix (e.g., "th", "aa").
    pub prefix: String,

    /// Optional n-gram order if using order-specific sharding.
    pub order: Option<u8>,

    /// Path to the shard file.
    pub path: PathBuf,

    /// Number of entries in the shard at checkpoint.
    pub entry_count: u64,

    /// Prefixes that have been fully imported to this shard.
    pub completed_prefixes: HashSet<String>,

    /// Prefix currently being imported (if any).
    pub current_prefix: Option<String>,

    /// Total n-grams processed through this shard.
    pub ngrams_processed: u64,

    /// Last WAL LSN for this shard.
    pub last_lsn: u64,

    /// When this shard was last checkpointed.
    pub last_checkpoint_time: u64,
}

impl ShardCheckpointRecord {
    /// Create a new shard checkpoint record.
    pub fn new(key: &ShardKey, path: impl Into<PathBuf>) -> Self {
        Self {
            prefix: key.prefix.clone(),
            order: key.order,
            path: path.into(),
            entry_count: 0,
            completed_prefixes: HashSet::new(),
            current_prefix: None,
            ngrams_processed: 0,
            last_lsn: 0,
            last_checkpoint_time: current_timestamp(),
        }
    }

    /// Convert back to a ShardKey.
    pub fn to_shard_key(&self) -> ShardKey {
        if let Some(order) = self.order {
            ShardKey::with_order(&self.prefix, order)
        } else {
            ShardKey::new(&self.prefix)
        }
    }

    /// Check if this shard was in progress when checkpointed.
    pub fn is_in_progress(&self) -> bool {
        self.current_prefix.is_some()
    }
}

/// Global checkpoint for coordinating all shards.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GlobalCheckpoint {
    /// Checkpoint format version.
    version: u32,

    /// When the checkpoint was first created.
    created_at: u64,

    /// When the checkpoint was last updated.
    last_updated: u64,

    /// Overall import state.
    pub import_state: ImportState,

    /// Per-shard checkpoint records.
    pub shards: HashMap<String, ShardCheckpointRecord>,

    /// Additional metadata (language, orders, etc.).
    pub metadata: HashMap<String, String>,
}

impl Default for GlobalCheckpoint {
    fn default() -> Self {
        Self::new()
    }
}

impl GlobalCheckpoint {
    /// Create a new empty checkpoint.
    pub fn new() -> Self {
        let now = current_timestamp();
        Self {
            version: CHECKPOINT_VERSION,
            created_at: now,
            last_updated: now,
            import_state: ImportState::NotStarted,
            shards: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Load a checkpoint from file, or create a new one if it doesn't exist.
    pub fn load_or_create(path: impl AsRef<Path>) -> CheckpointResult<Self> {
        let path = path.as_ref();
        if path.exists() {
            Self::load(path)
        } else {
            Ok(Self::new())
        }
    }

    /// Load a checkpoint from file.
    pub fn load(path: impl AsRef<Path>) -> CheckpointResult<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let checkpoint: GlobalCheckpoint = serde_json::from_reader(reader)?;

        // Verify version
        if checkpoint.version != CHECKPOINT_VERSION {
            return Err(CheckpointError::VersionMismatch {
                expected: CHECKPOINT_VERSION,
                found: checkpoint.version,
            });
        }

        Ok(checkpoint)
    }

    /// Save the checkpoint to file atomically.
    ///
    /// Uses write-to-temp + rename pattern for crash safety.
    pub fn save(&self, path: impl AsRef<Path>) -> CheckpointResult<()> {
        let path = path.as_ref();
        let temp_path = path.with_extension("json.tmp");

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Write to temporary file
        {
            let file = File::create(&temp_path)?;
            let writer = BufWriter::new(file);
            serde_json::to_writer_pretty(writer, self)?;
        }

        // Atomic rename
        fs::rename(&temp_path, path)?;

        Ok(())
    }

    /// Update the checkpoint timestamp.
    pub fn touch(&mut self) {
        self.last_updated = current_timestamp();
    }

    /// Set import to in-progress state.
    pub fn start_import(&mut self) {
        self.import_state = ImportState::InProgress {
            started_at: current_timestamp(),
            phase: ImportPhase::Importing,
        };
        self.touch();
    }

    /// Set the current import phase.
    pub fn set_phase(&mut self, phase: ImportPhase) {
        if let ImportState::InProgress { started_at, .. } = &self.import_state {
            self.import_state = ImportState::InProgress {
                started_at: *started_at,
                phase,
            };
            self.touch();
        }
    }

    /// Mark import as completed.
    pub fn complete_import(&mut self, total_ngrams: u64, unique_ngrams: u64) {
        self.import_state = ImportState::Completed {
            completed_at: current_timestamp(),
            total_ngrams,
            unique_ngrams,
        };
        self.touch();
    }

    /// Mark import as failed.
    pub fn fail_import(&mut self, error: impl Into<String>) {
        self.import_state = ImportState::Failed {
            failed_at: current_timestamp(),
            error: error.into(),
        };
        self.touch();
    }

    /// Check if recovery is needed (was interrupted).
    pub fn needs_recovery(&self) -> bool {
        matches!(self.import_state, ImportState::RequiresRecovery { .. })
    }

    /// Check if import is in progress.
    pub fn is_in_progress(&self) -> bool {
        matches!(self.import_state, ImportState::InProgress { .. })
    }

    /// Check if import has completed.
    pub fn is_completed(&self) -> bool {
        matches!(self.import_state, ImportState::Completed { .. })
    }

    /// Get or create a shard record.
    pub fn get_or_create_shard(&mut self, key: &ShardKey, path: impl Into<PathBuf>) -> &mut ShardCheckpointRecord {
        let key_str = key.to_string();
        self.shards
            .entry(key_str)
            .or_insert_with(|| ShardCheckpointRecord::new(key, path))
    }

    /// Update a shard's checkpoint record.
    pub fn update_shard(&mut self, key: &ShardKey, entry_count: u64, ngrams_processed: u64) {
        let key_str = key.to_string();
        if let Some(record) = self.shards.get_mut(&key_str) {
            record.entry_count = entry_count;
            record.ngrams_processed = ngrams_processed;
            record.last_checkpoint_time = current_timestamp();
        }
        self.touch();
    }

    /// Mark a prefix as completed for a shard.
    pub fn complete_prefix(&mut self, key: &ShardKey, prefix: &str) {
        let key_str = key.to_string();
        if let Some(record) = self.shards.get_mut(&key_str) {
            record.completed_prefixes.insert(prefix.to_string());
            record.current_prefix = None;
        }
        self.touch();
    }

    /// Set the current prefix being processed for a shard.
    pub fn set_current_prefix(&mut self, key: &ShardKey, prefix: Option<&str>) {
        let key_str = key.to_string();
        if let Some(record) = self.shards.get_mut(&key_str) {
            record.current_prefix = prefix.map(String::from);
        }
        self.touch();
    }

    /// Get all completed prefixes across all shards.
    pub fn all_completed_prefixes(&self) -> HashSet<String> {
        self.shards
            .values()
            .flat_map(|r| r.completed_prefixes.iter().cloned())
            .collect()
    }

    /// Get shards that were in progress when checkpointed.
    pub fn in_progress_shards(&self) -> Vec<&ShardCheckpointRecord> {
        self.shards.values().filter(|r| r.is_in_progress()).collect()
    }

    /// Detect if recovery is needed and update state accordingly.
    ///
    /// This should be called when opening an existing checkpoint.
    pub fn detect_recovery_needed(&mut self) {
        // If import was in progress, it was interrupted
        if let ImportState::InProgress { .. } = &self.import_state {
            let in_progress: Vec<String> = self
                .shards
                .iter()
                .filter(|(_, r)| r.is_in_progress())
                .map(|(k, _)| k.clone())
                .collect();

            self.import_state = ImportState::RequiresRecovery {
                last_checkpoint_at: self.last_updated,
                in_progress_shards: in_progress,
            };
        }
    }

    /// Mark recovery complete and resume import.
    pub fn resume_import(&mut self) {
        if let ImportState::RequiresRecovery { .. } = &self.import_state {
            self.import_state = ImportState::InProgress {
                started_at: current_timestamp(),
                phase: ImportPhase::Importing,
            };
            self.touch();
        }
    }

    /// Get total n-grams across all shards.
    pub fn total_ngrams(&self) -> u64 {
        self.shards.values().map(|r| r.ngrams_processed).sum()
    }

    /// Get total unique entries across all shards.
    pub fn total_entries(&self) -> u64 {
        self.shards.values().map(|r| r.entry_count).sum()
    }

    /// Set metadata value.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
        self.touch();
    }

    /// Get metadata value.
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// Get a summary for logging.
    pub fn summary(&self) -> CheckpointSummary {
        CheckpointSummary {
            state: format!("{:?}", self.import_state),
            shard_count: self.shards.len(),
            total_entries: self.total_entries(),
            total_ngrams: self.total_ngrams(),
            completed_prefixes: self.all_completed_prefixes().len(),
            last_updated: self.last_updated,
        }
    }
}

/// Summary of checkpoint state for logging.
#[derive(Clone, Debug)]
pub struct CheckpointSummary {
    /// State description.
    pub state: String,

    /// Number of shards.
    pub shard_count: usize,

    /// Total unique entries.
    pub total_entries: u64,

    /// Total n-grams processed.
    pub total_ngrams: u64,

    /// Number of completed prefixes.
    pub completed_prefixes: usize,

    /// Last update timestamp.
    pub last_updated: u64,
}

/// Get current timestamp as seconds since UNIX epoch.
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Checkpoint manager that handles save/restore operations.
pub struct CheckpointManager {
    /// Path to the global checkpoint file.
    checkpoint_path: PathBuf,

    /// The current checkpoint state.
    checkpoint: GlobalCheckpoint,

    /// Interval between auto-saves (in milliseconds).
    auto_save_interval_ms: u64,

    /// Last auto-save time.
    last_save_time: std::time::Instant,
}

impl CheckpointManager {
    /// Create a new checkpoint manager.
    pub fn new(checkpoint_path: impl Into<PathBuf>, auto_save_interval_ms: u64) -> CheckpointResult<Self> {
        let checkpoint_path = checkpoint_path.into();
        let checkpoint = GlobalCheckpoint::load_or_create(&checkpoint_path)?;

        Ok(Self {
            checkpoint_path,
            checkpoint,
            auto_save_interval_ms,
            last_save_time: std::time::Instant::now(),
        })
    }

    /// Get a reference to the checkpoint.
    pub fn checkpoint(&self) -> &GlobalCheckpoint {
        &self.checkpoint
    }

    /// Get a mutable reference to the checkpoint.
    pub fn checkpoint_mut(&mut self) -> &mut GlobalCheckpoint {
        &mut self.checkpoint
    }

    /// Save the checkpoint immediately.
    pub fn save(&mut self) -> CheckpointResult<()> {
        self.checkpoint.save(&self.checkpoint_path)?;
        self.last_save_time = std::time::Instant::now();
        Ok(())
    }

    /// Save if enough time has passed since the last save.
    pub fn maybe_save(&mut self) -> CheckpointResult<bool> {
        let elapsed = self.last_save_time.elapsed().as_millis() as u64;
        if elapsed >= self.auto_save_interval_ms {
            self.save()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Check if recovery is needed.
    pub fn needs_recovery(&self) -> bool {
        self.checkpoint.needs_recovery()
    }

    /// Detect and mark recovery if needed.
    pub fn detect_recovery(&mut self) {
        self.checkpoint.detect_recovery_needed();
    }

    /// Resume import after recovery.
    pub fn resume(&mut self) {
        self.checkpoint.resume_import();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_checkpoint_create_and_save() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("checkpoint.json");

        let mut checkpoint = GlobalCheckpoint::new();
        checkpoint.start_import();
        checkpoint.set_metadata("language", "en");

        checkpoint.save(&path).expect("Failed to save");

        // Load and verify
        let loaded = GlobalCheckpoint::load(&path).expect("Failed to load");
        assert!(loaded.is_in_progress());
        assert_eq!(loaded.get_metadata("language"), Some(&"en".to_string()));
    }

    #[test]
    fn test_checkpoint_shard_tracking() {
        let mut checkpoint = GlobalCheckpoint::new();
        checkpoint.start_import();

        let key = ShardKey::new("th");
        checkpoint.get_or_create_shard(&key, "/tmp/shard_th.artrie");

        checkpoint.set_current_prefix(&key, Some("th"));
        assert_eq!(checkpoint.in_progress_shards().len(), 1);

        checkpoint.complete_prefix(&key, "th");
        assert!(checkpoint.all_completed_prefixes().contains("th"));
        assert!(checkpoint.in_progress_shards().is_empty());
    }

    #[test]
    fn test_recovery_detection() {
        let mut checkpoint = GlobalCheckpoint::new();
        checkpoint.start_import();

        let key = ShardKey::new("th");
        checkpoint.get_or_create_shard(&key, "/tmp/shard_th.artrie");
        checkpoint.set_current_prefix(&key, Some("th"));

        // Simulate crash detection
        checkpoint.detect_recovery_needed();
        assert!(checkpoint.needs_recovery());

        if let ImportState::RequiresRecovery { in_progress_shards, .. } = &checkpoint.import_state {
            assert_eq!(in_progress_shards.len(), 1);
            assert_eq!(in_progress_shards[0], "th");
        } else {
            panic!("Expected RequiresRecovery state");
        }

        // Resume
        checkpoint.resume_import();
        assert!(checkpoint.is_in_progress());
    }

    #[test]
    fn test_checkpoint_completion() {
        let mut checkpoint = GlobalCheckpoint::new();
        checkpoint.start_import();

        checkpoint.complete_import(1000000, 500000);
        assert!(checkpoint.is_completed());

        if let ImportState::Completed { total_ngrams, unique_ngrams, .. } = &checkpoint.import_state {
            assert_eq!(*total_ngrams, 1000000);
            assert_eq!(*unique_ngrams, 500000);
        } else {
            panic!("Expected Completed state");
        }
    }

    #[test]
    fn test_checkpoint_atomic_save() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("checkpoint.json");
        let temp_path = path.with_extension("json.tmp");

        let mut checkpoint = GlobalCheckpoint::new();
        checkpoint.start_import();
        checkpoint.save(&path).expect("Failed to save");

        // Temp file should not exist after successful save
        assert!(!temp_path.exists());
        assert!(path.exists());
    }

    #[test]
    fn test_checkpoint_manager() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = dir.path().join("checkpoint.json");

        let mut manager = CheckpointManager::new(&path, 1000).expect("Failed to create manager");

        manager.checkpoint_mut().start_import();
        manager.save().expect("Failed to save");

        // Reload and verify
        let manager2 = CheckpointManager::new(&path, 1000).expect("Failed to create manager");
        assert!(manager2.checkpoint().is_in_progress());
    }

    #[test]
    fn test_shard_checkpoint_record() {
        let key = ShardKey::with_order("th", 2);
        let record = ShardCheckpointRecord::new(&key, "/tmp/shard.artrie");

        assert_eq!(record.prefix, "th");
        assert_eq!(record.order, Some(2));
        assert!(!record.is_in_progress());

        let restored_key = record.to_shard_key();
        assert_eq!(restored_key.prefix, "th");
        assert_eq!(restored_key.order, Some(2));
    }
}
