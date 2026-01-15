//! Checkpoint and resume support for Google Books import.
//!
//! Long-running imports (hours for full datasets) need checkpoint support
//! to handle interruptions gracefully without losing progress.
//!
//! ## Checkpoint Format Versions
//!
//! - **Version 1**: Single-order tracking with `current_order` and `completed_prefixes`
//! - **Version 2**: Per-order tracking with `order_progress` HashMap for overlapping order processing

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Progress tracking for a single n-gram order.
///
/// Allows multiple orders to be tracked independently for overlapping
/// order processing (e.g., workers can process 2-grams while others
/// finish 1-grams).
///
/// ## Prefix States
///
/// A prefix can be in one of four states:
/// - **Not started**: Not in any list (needs processing)
/// - **In progress**: In `in_progress_prefixes` (started but not finished)
/// - **Completed**: In `completed_prefixes` (successfully processed)
/// - **Failed**: In `failed_prefixes` (failed after exhausting retries)
///
/// On resume:
/// - Completed prefixes are skipped
/// - In-progress prefixes are cleared and retried (data may be partial)
/// - Failed prefixes are retried on subsequent runs
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OrderProgress {
    /// Completed prefix files for this order.
    ///
    /// For 1-grams: ["a", "b", "c", ...]
    /// For 2-5 grams: ["aa", "ab", "ac", ...]
    pub completed_prefixes: Vec<String>,

    /// Prefixes that started processing but didn't complete.
    ///
    /// On crash recovery, these prefixes have potentially partial data
    /// in the trie. They should be cleared and retried to ensure
    /// data integrity and prevent double-counting.
    #[serde(default)]
    pub in_progress_prefixes: Vec<String>,

    /// Prefixes that failed after exhausting all retries.
    ///
    /// These are skipped with warnings on the current run but will be
    /// retried on subsequent runs. Permanently failed prefixes should
    /// be manually removed if they can never succeed.
    #[serde(default)]
    pub failed_prefixes: Vec<String>,

    /// Whether this order is fully complete (all prefixes processed).
    pub is_complete: bool,

    /// N-grams processed for this order.
    pub ngrams_processed: u64,
}

/// Import checkpoint for resume support.
///
/// Checkpoints are saved after each prefix file completes and on
/// graceful shutdown (SIGINT/SIGTERM).
///
/// ## Version 2 Changes
///
/// Version 2 supports overlapping order processing where workers can
/// process multiple n-gram orders concurrently. The `order_progress`
/// HashMap tracks progress for each order independently, replacing
/// the single `current_order` + `completed_prefixes` approach.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImportCheckpoint {
    /// Version of checkpoint format (for future compatibility).
    pub version: u32,

    /// Per-order progress tracking.
    ///
    /// Key: n-gram order (1-5)
    /// Value: Progress for that order
    ///
    /// This replaces `completed_orders`, `current_order`, and `completed_prefixes`
    /// from version 1, enabling overlapping order processing.
    pub order_progress: HashMap<u8, OrderProgress>,

    /// Current prefix file being processed (if any).
    ///
    /// Note: With overlapping orders, multiple prefixes may be in progress
    /// across different workers. This field is kept for byte-level resume
    /// of a single file (future enhancement).
    pub current_prefix: Option<String>,

    /// Byte offset within current prefix file.
    pub byte_offset: u64,

    /// MKN computation phase.
    pub mkn_phase: MknPhase,

    /// Statistics for completed work.
    pub stats: CheckpointStats,

    /// Timestamp when checkpoint was saved.
    pub timestamp: DateTime<Utc>,
}

/// Version 1 checkpoint structure for migration.
///
/// This is kept for backward compatibility - when loading a v1 checkpoint,
/// it's automatically migrated to v2 format.
#[derive(Clone, Debug, Deserialize)]
struct ImportCheckpointV1 {
    pub version: u32,
    pub completed_orders: Vec<u8>,
    pub current_order: u8,
    pub completed_prefixes: Vec<String>,
    pub current_prefix: Option<String>,
    pub byte_offset: u64,
    pub mkn_phase: MknPhase,
    pub stats: CheckpointStats,
    pub timestamp: DateTime<Utc>,
}

/// Phase of MKN statistics computation.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub enum MknPhase {
    /// MKN computation not started.
    #[default]
    NotStarted,

    /// Pass 1: Counting raw frequencies.
    Pass1InProgress {
        /// Current order being processed.
        current_order: u8,
    },

    /// Pass 1 complete.
    Pass1Complete,

    /// Pass 2: Computing continuation counts.
    Pass2InProgress {
        /// Current order being processed.
        current_order: u8,
    },

    /// MKN computation complete.
    Complete,
}

/// Statistics tracked in checkpoint.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CheckpointStats {
    /// Total n-grams processed.
    pub ngrams_processed: u64,

    /// Unique n-grams inserted (new entries, not duplicates).
    #[serde(default)]
    pub unique_ngrams: u64,

    /// N-grams per order.
    pub ngrams_by_order: [u64; 5],

    /// Bytes downloaded (HTTP mode).
    pub bytes_downloaded: u64,

    /// Files processed.
    pub files_processed: u32,

    /// Elapsed time in seconds.
    pub elapsed_seconds: u64,
}

impl ImportCheckpoint {
    /// Current checkpoint format version.
    pub const CURRENT_VERSION: u32 = 2;

    /// Create a new empty checkpoint.
    pub fn new() -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            order_progress: HashMap::new(),
            current_prefix: None,
            byte_offset: 0,
            mkn_phase: MknPhase::NotStarted,
            stats: CheckpointStats::default(),
            timestamp: Utc::now(),
        }
    }

    /// Load checkpoint from file.
    ///
    /// Automatically migrates v1 checkpoints to v2 format.
    pub fn load(path: &Path) -> Result<Self, CheckpointError> {
        let file = File::open(path).map_err(CheckpointError::Io)?;
        let reader = BufReader::new(file);

        // First, try to read as raw JSON to check version
        let value: serde_json::Value =
            serde_json::from_reader(reader).map_err(CheckpointError::Json)?;

        let version = value
            .get("version")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as u32;

        if version > Self::CURRENT_VERSION {
            return Err(CheckpointError::UnsupportedVersion {
                found: version,
                max: Self::CURRENT_VERSION,
            });
        }

        if version == 1 {
            // Parse as v1 and migrate
            let v1: ImportCheckpointV1 =
                serde_json::from_value(value).map_err(CheckpointError::Json)?;
            log::info!(
                "Migrating checkpoint from v1 to v2 format (current_order={}, prefixes={})",
                v1.current_order,
                v1.completed_prefixes.len()
            );
            Ok(Self::migrate_from_v1(v1))
        } else {
            // Parse as current version
            serde_json::from_value(value).map_err(CheckpointError::Json)
        }
    }

    /// Migrate a v1 checkpoint to v2 format.
    fn migrate_from_v1(v1: ImportCheckpointV1) -> Self {
        let mut order_progress = HashMap::new();

        // Mark completed orders as complete with empty prefixes
        // (we don't have the prefix list, but they're done)
        for order in v1.completed_orders {
            order_progress.insert(
                order,
                OrderProgress {
                    completed_prefixes: Vec::new(),
                    in_progress_prefixes: Vec::new(),
                    failed_prefixes: Vec::new(),
                    is_complete: true,
                    ngrams_processed: 0, // We don't have per-order stats in v1
                },
            );
        }

        // Add current order's progress (if it has any completed prefixes)
        if !v1.completed_prefixes.is_empty() || v1.current_prefix.is_some() {
            order_progress.insert(
                v1.current_order,
                OrderProgress {
                    completed_prefixes: v1.completed_prefixes,
                    in_progress_prefixes: Vec::new(),
                    failed_prefixes: Vec::new(),
                    is_complete: false,
                    ngrams_processed: 0,
                },
            );
        }

        Self {
            version: Self::CURRENT_VERSION,
            order_progress,
            current_prefix: v1.current_prefix,
            byte_offset: v1.byte_offset,
            mkn_phase: v1.mkn_phase,
            stats: v1.stats,
            timestamp: v1.timestamp,
        }
    }

    /// Save checkpoint to file.
    pub fn save(&self, path: &Path) -> Result<(), CheckpointError> {
        // Write to temp file first, then rename for atomicity
        let temp_path = path.with_extension("checkpoint.tmp");
        let file = File::create(&temp_path).map_err(CheckpointError::Io)?;
        let writer = BufWriter::new(file);

        let mut checkpoint = self.clone();
        checkpoint.timestamp = Utc::now();

        serde_json::to_writer_pretty(writer, &checkpoint).map_err(CheckpointError::Json)?;

        // Atomic rename
        std::fs::rename(&temp_path, path).map_err(CheckpointError::Io)?;

        Ok(())
    }

    /// Check if checkpoint file exists.
    pub fn exists(path: &Path) -> bool {
        path.exists()
    }

    /// Delete checkpoint file.
    pub fn delete(path: &Path) -> Result<(), CheckpointError> {
        if path.exists() {
            std::fs::remove_file(path).map_err(CheckpointError::Io)?;
        }
        Ok(())
    }

    /// Mark a prefix as in-progress (started but not finished).
    ///
    /// This should be called BEFORE any n-grams are written to the trie,
    /// and the checkpoint should be saved immediately. This allows crash
    /// recovery to detect partial data.
    pub fn start_prefix(&mut self, order: u8, prefix: &str) {
        let progress = self.order_progress.entry(order).or_default();
        let prefix_str = prefix.to_string();

        // Ensure not already in any list
        progress.completed_prefixes.retain(|p| p != &prefix_str);
        progress.failed_prefixes.retain(|p| p != &prefix_str);

        if !progress.in_progress_prefixes.contains(&prefix_str) {
            progress.in_progress_prefixes.push(prefix_str);
        }
    }

    /// Mark a prefix as completed for a specific order.
    ///
    /// Moves the prefix from in_progress to completed.
    /// This is the v2 version that supports overlapping order processing.
    pub fn complete_prefix(&mut self, order: u8, prefix: &str) {
        let progress = self.order_progress.entry(order).or_default();
        let prefix_str = prefix.to_string();

        // Move from in_progress to completed
        progress.in_progress_prefixes.retain(|p| p != &prefix_str);

        if !progress.completed_prefixes.contains(&prefix_str) {
            progress.completed_prefixes.push(prefix_str);
        }
        self.stats.files_processed += 1;
    }

    /// Mark a prefix as failed (exhausted all retries).
    ///
    /// Moves the prefix from in_progress to failed.
    /// Failed prefixes are skipped on the current run but will be
    /// retried on subsequent runs.
    pub fn fail_prefix(&mut self, order: u8, prefix: &str) {
        let progress = self.order_progress.entry(order).or_default();
        let prefix_str = prefix.to_string();

        // Move from in_progress to failed
        progress.in_progress_prefixes.retain(|p| p != &prefix_str);

        if !progress.failed_prefixes.contains(&prefix_str) {
            progress.failed_prefixes.push(prefix_str);
        }
    }

    /// Clear a prefix from the failed list (for retry on subsequent run).
    pub fn clear_failed(&mut self, order: u8, prefix: &str) {
        if let Some(progress) = self.order_progress.get_mut(&order) {
            progress.failed_prefixes.retain(|p| p != prefix);
        }
    }

    /// Check if a prefix is currently marked as in-progress.
    pub fn is_in_progress(&self, order: u8, prefix: &str) -> bool {
        self.order_progress
            .get(&order)
            .map(|p| p.in_progress_prefixes.contains(&prefix.to_string()))
            .unwrap_or(false)
    }

    /// Check if a prefix has failed (exhausted retries).
    pub fn is_failed_prefix(&self, order: u8, prefix: &str) -> bool {
        self.order_progress
            .get(&order)
            .map(|p| p.failed_prefixes.contains(&prefix.to_string()))
            .unwrap_or(false)
    }

    /// Get all in-progress prefixes for an order.
    ///
    /// On resume, these should be cleared and retried.
    pub fn in_progress_prefixes(&self, order: u8) -> Vec<String> {
        self.order_progress
            .get(&order)
            .map(|p| p.in_progress_prefixes.clone())
            .unwrap_or_default()
    }

    /// Get all failed prefixes for an order.
    ///
    /// These should be retried on subsequent runs.
    pub fn failed_prefixes(&self, order: u8) -> Vec<String> {
        self.order_progress
            .get(&order)
            .map(|p| p.failed_prefixes.clone())
            .unwrap_or_default()
    }

    /// Move all in-progress prefixes to failed (for crash recovery).
    ///
    /// Call this on resume when in-progress prefixes are detected.
    /// The caller should clear partial data from the trie before retrying.
    pub fn recover_in_progress_as_failed(&mut self, order: u8) {
        if let Some(progress) = self.order_progress.get_mut(&order) {
            for prefix in progress.in_progress_prefixes.drain(..) {
                if !progress.failed_prefixes.contains(&prefix) {
                    progress.failed_prefixes.push(prefix);
                }
            }
        }
    }

    /// Get count of failed prefixes for an order.
    pub fn failed_prefix_count(&self, order: u8) -> usize {
        self.order_progress
            .get(&order)
            .map(|p| p.failed_prefixes.len())
            .unwrap_or(0)
    }

    /// Get total count of failed prefixes across all orders.
    pub fn total_failed_prefix_count(&self) -> usize {
        self.order_progress
            .values()
            .map(|p| p.failed_prefixes.len())
            .sum()
    }

    /// Add n-grams processed count to an order.
    pub fn add_ngrams(&mut self, order: u8, count: u64) {
        let progress = self.order_progress.entry(order).or_default();
        progress.ngrams_processed += count;
        self.stats.ngrams_processed += count;
    }

    /// Mark an order as fully completed.
    ///
    /// Unlike v1, this does NOT clear the completed_prefixes - they're kept
    /// for resume verification and debugging.
    pub fn complete_order(&mut self, order: u8) {
        let progress = self.order_progress.entry(order).or_default();
        progress.is_complete = true;
        // Don't clear completed_prefixes - keep for resume verification
    }

    /// Check if a specific prefix needs processing.
    ///
    /// Returns `true` if the prefix is neither completed nor in-progress.
    /// Failed prefixes are considered to "need processing" since they
    /// should be retried on subsequent runs.
    ///
    /// Note: In-progress prefixes are excluded because they need special
    /// handling (clear partial data before retry). Use `in_progress_prefixes()`
    /// to get the list of prefixes that need recovery.
    pub fn needs_prefix(&self, order: u8, prefix: &str) -> bool {
        self.order_progress
            .get(&order)
            .map(|p| {
                let prefix_str = prefix.to_string();
                !p.is_complete
                    && !p.completed_prefixes.contains(&prefix_str)
                    && !p.in_progress_prefixes.contains(&prefix_str)
            })
            .unwrap_or(true) // If no progress recorded, needs processing
    }

    /// Check if an order is fully complete.
    pub fn is_order_complete(&self, order: u8) -> bool {
        self.order_progress
            .get(&order)
            .map(|p| p.is_complete)
            .unwrap_or(false)
    }

    /// Get the count of completed prefixes for an order.
    pub fn completed_prefix_count(&self, order: u8) -> usize {
        self.order_progress
            .get(&order)
            .map(|p| p.completed_prefixes.len())
            .unwrap_or(0)
    }

    /// Get the total count of completed prefixes across all orders.
    pub fn total_completed_prefix_count(&self) -> usize {
        self.order_progress
            .values()
            .map(|p| p.completed_prefixes.len())
            .sum()
    }

    /// Get all orders that have any progress (started but not necessarily complete).
    pub fn orders_in_progress(&self) -> Vec<u8> {
        self.order_progress
            .iter()
            .filter(|(_, p)| !p.is_complete)
            .map(|(order, _)| *order)
            .collect()
    }

    /// Get all completed orders.
    pub fn completed_orders(&self) -> Vec<u8> {
        self.order_progress
            .iter()
            .filter(|(_, p)| p.is_complete)
            .map(|(order, _)| *order)
            .collect()
    }

    /// Update byte offset for current prefix.
    pub fn update_offset(&mut self, prefix: &str, offset: u64) {
        self.current_prefix = Some(prefix.to_string());
        self.byte_offset = offset;
    }

    /// Get human-readable progress summary.
    pub fn progress_summary(&self) -> String {
        let completed: Vec<_> = self.completed_orders();
        let in_progress: Vec<_> = self.orders_in_progress();

        let prefix_counts: Vec<String> = self
            .order_progress
            .iter()
            .filter(|(_, p)| !p.is_complete)
            .map(|(order, p)| format!("{}:{}", order, p.completed_prefixes.len()))
            .collect();

        let failed_count = self.total_failed_prefix_count();
        let failed_str = if failed_count > 0 {
            format!(", Failed: {}", failed_count)
        } else {
            String::new()
        };

        format!(
            "Completed: {:?}, In progress: {:?}, Prefixes: [{}], N-grams: {}, Files: {}{}",
            completed,
            in_progress,
            prefix_counts.join(", "),
            self.stats.ngrams_processed,
            self.stats.files_processed,
            failed_str,
        )
    }
}

impl Default for ImportCheckpoint {
    fn default() -> Self {
        Self::new()
    }
}

/// Checkpoint errors.
#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[source] std::io::Error),

    /// JSON serialization error.
    #[error("JSON error: {0}")]
    Json(#[source] serde_json::Error),

    /// Unsupported checkpoint version.
    #[error("Unsupported checkpoint version: found {found}, max supported {max}")]
    UnsupportedVersion { found: u32, max: u32 },

    /// Trie operation failed.
    #[error("Trie error: {0}")]
    Trie(String),
}

// =============================================================================
// Trie-based Checkpoint Storage
// =============================================================================
//
// Since DiskBackedCharTrieInner<u64> stores u64 values (not byte arrays),
// we use a key-based storage approach where each piece of checkpoint data
// becomes its own key with a u64 value encoding status or count.
//
// Key namespace (NUL byte prefix ensures no collision with n-grams):
//
// Metadata (u64 encoded values):
//   \x00__ckpt__:version         -> version number (u64)
//   \x00__ckpt__:mkn_phase       -> MKN phase ordinal (0-4)
//   \x00__ckpt__:byte_offset     -> byte offset in current file
//   \x00__ckpt__:timestamp       -> Unix timestamp (seconds)
//   \x00__ckpt__:ngrams_processed -> total ngrams processed
//   \x00__ckpt__:files_processed -> total files processed
//   \x00__ckpt__:bytes_downloaded -> total bytes downloaded
//   \x00__ckpt__:elapsed_seconds -> elapsed time in seconds
//
// Per-order n-gram counts:
//   \x00__ckpt__:ngrams_by_order:{order} -> ngram count for order
//
// Prefix status (value = status code: 1=completed, 2=in_progress, 3=failed):
//   \x00__ckpt__:prefix:{order}:{prefix} -> status code
//
// Order completion status:
//   \x00__ckpt__:order_complete:{order} -> 1 if complete
// =============================================================================

/// Reserved key prefix for checkpoint metadata (NUL byte makes it invalid as n-gram).
///
/// N-grams never start with NUL, so this namespace is guaranteed to be separate.
pub const CHECKPOINT_KEY_PREFIX: &str = "\x00__ckpt__";

/// Key for checkpoint version.
pub const CHECKPOINT_VERSION_KEY: &str = "\x00__ckpt__:version";

/// Key for MKN phase (encoded as ordinal).
pub const CHECKPOINT_MKN_PHASE_KEY: &str = "\x00__ckpt__:mkn_phase";

/// Key for byte offset in current file.
pub const CHECKPOINT_BYTE_OFFSET_KEY: &str = "\x00__ckpt__:byte_offset";

/// Key for timestamp (Unix seconds).
pub const CHECKPOINT_TIMESTAMP_KEY: &str = "\x00__ckpt__:timestamp";

/// Key for total ngrams processed.
pub const CHECKPOINT_NGRAMS_PROCESSED_KEY: &str = "\x00__ckpt__:ngrams_processed";

/// Key for unique ngrams inserted.
pub const CHECKPOINT_UNIQUE_NGRAMS_KEY: &str = "\x00__ckpt__:unique_ngrams";

/// Key for files processed count.
pub const CHECKPOINT_FILES_PROCESSED_KEY: &str = "\x00__ckpt__:files_processed";

/// Key for bytes downloaded.
pub const CHECKPOINT_BYTES_DOWNLOADED_KEY: &str = "\x00__ckpt__:bytes_downloaded";

/// Key for elapsed seconds.
pub const CHECKPOINT_ELAPSED_KEY: &str = "\x00__ckpt__:elapsed_seconds";

/// Key prefix for ngrams by order. Format: "\x00__ckpt__:ngrams_by_order:{order}"
pub const CHECKPOINT_NGRAMS_BY_ORDER_PREFIX: &str = "\x00__ckpt__:ngrams_by_order:";

/// Key prefix for prefix status. Format: "\x00__ckpt__:prefix:{order}:{prefix}"
pub const CHECKPOINT_PREFIX_KEY_PREFIX: &str = "\x00__ckpt__:prefix:";

/// Key prefix for order completion. Format: "\x00__ckpt__:order_complete:{order}"
pub const CHECKPOINT_ORDER_COMPLETE_PREFIX: &str = "\x00__ckpt__:order_complete:";

/// Prefix status codes for trie storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum PrefixStatusCode {
    /// Prefix has been fully processed.
    Completed = 1,
    /// Prefix is currently being processed.
    InProgress = 2,
    /// Prefix failed after exhausting retries.
    Failed = 3,
}

impl PrefixStatusCode {
    /// Convert from u64.
    pub fn from_u64(value: u64) -> Option<Self> {
        match value {
            1 => Some(Self::Completed),
            2 => Some(Self::InProgress),
            3 => Some(Self::Failed),
            _ => None,
        }
    }
}

impl MknPhase {
    /// Convert MKN phase to ordinal for storage.
    pub fn to_ordinal(&self) -> u64 {
        match self {
            MknPhase::NotStarted => 0,
            MknPhase::Pass1InProgress { current_order } => 1 + (*current_order as u64) * 10,
            MknPhase::Pass1Complete => 100,
            MknPhase::Pass2InProgress { current_order } => 101 + (*current_order as u64) * 10,
            MknPhase::Complete => 200,
        }
    }

    /// Convert ordinal back to MKN phase.
    pub fn from_ordinal(ordinal: u64) -> Self {
        match ordinal {
            0 => MknPhase::NotStarted,
            100 => MknPhase::Pass1Complete,
            200 => MknPhase::Complete,
            n if n >= 1 && n < 100 => MknPhase::Pass1InProgress {
                current_order: ((n - 1) / 10) as u8,
            },
            n if n >= 101 && n < 200 => MknPhase::Pass2InProgress {
                current_order: ((n - 101) / 10) as u8,
            },
            _ => MknPhase::NotStarted,
        }
    }
}

impl ImportCheckpoint {
    /// Save checkpoint to a trie using reserved keys.
    ///
    /// This stores the checkpoint data atomically with the n-gram data,
    /// ensuring consistency between data and progress tracking.
    ///
    /// # Arguments
    ///
    /// * `trie` - The trie to store the checkpoint in
    ///
    /// # Returns
    ///
    /// The number of keys written.
    pub fn save_to_trie<T>(&self, trie: &mut T) -> Result<usize, CheckpointError>
    where
        T: TrieCheckpointStorage,
    {
        let mut keys_written = 0;

        // Store metadata as individual keys with u64 values
        trie.store_checkpoint_u64(CHECKPOINT_VERSION_KEY, self.version as u64)
            .map_err(|e| CheckpointError::Trie(e.to_string()))?;
        keys_written += 1;

        trie.store_checkpoint_u64(CHECKPOINT_MKN_PHASE_KEY, self.mkn_phase.to_ordinal())
            .map_err(|e| CheckpointError::Trie(e.to_string()))?;
        keys_written += 1;

        trie.store_checkpoint_u64(CHECKPOINT_BYTE_OFFSET_KEY, self.byte_offset)
            .map_err(|e| CheckpointError::Trie(e.to_string()))?;
        keys_written += 1;

        // Store timestamp as Unix timestamp
        let timestamp_secs = self.timestamp.timestamp() as u64;
        trie.store_checkpoint_u64(CHECKPOINT_TIMESTAMP_KEY, timestamp_secs)
            .map_err(|e| CheckpointError::Trie(e.to_string()))?;
        keys_written += 1;

        // Store stats
        trie.store_checkpoint_u64(CHECKPOINT_NGRAMS_PROCESSED_KEY, self.stats.ngrams_processed)
            .map_err(|e| CheckpointError::Trie(e.to_string()))?;
        keys_written += 1;

        trie.store_checkpoint_u64(CHECKPOINT_UNIQUE_NGRAMS_KEY, self.stats.unique_ngrams)
            .map_err(|e| CheckpointError::Trie(e.to_string()))?;
        keys_written += 1;

        trie.store_checkpoint_u64(CHECKPOINT_FILES_PROCESSED_KEY, self.stats.files_processed as u64)
            .map_err(|e| CheckpointError::Trie(e.to_string()))?;
        keys_written += 1;

        trie.store_checkpoint_u64(CHECKPOINT_BYTES_DOWNLOADED_KEY, self.stats.bytes_downloaded)
            .map_err(|e| CheckpointError::Trie(e.to_string()))?;
        keys_written += 1;

        trie.store_checkpoint_u64(CHECKPOINT_ELAPSED_KEY, self.stats.elapsed_seconds)
            .map_err(|e| CheckpointError::Trie(e.to_string()))?;
        keys_written += 1;

        // Store ngrams by order
        for (idx, &count) in self.stats.ngrams_by_order.iter().enumerate() {
            let key = format!("{}{}", CHECKPOINT_NGRAMS_BY_ORDER_PREFIX, idx + 1);
            trie.store_checkpoint_u64(&key, count)
                .map_err(|e| CheckpointError::Trie(e.to_string()))?;
            keys_written += 1;
        }

        // Store per-order progress
        for (order, progress) in &self.order_progress {
            // Store order completion status
            if progress.is_complete {
                let key = format!("{}{}", CHECKPOINT_ORDER_COMPLETE_PREFIX, order);
                trie.store_checkpoint_u64(&key, 1)
                    .map_err(|e| CheckpointError::Trie(e.to_string()))?;
                keys_written += 1;
            }

            // Store prefix statuses
            for prefix in &progress.completed_prefixes {
                let key = format!("{}{}:{}", CHECKPOINT_PREFIX_KEY_PREFIX, order, prefix);
                trie.store_checkpoint_u64(&key, PrefixStatusCode::Completed as u64)
                    .map_err(|e| CheckpointError::Trie(e.to_string()))?;
                keys_written += 1;
            }

            for prefix in &progress.in_progress_prefixes {
                let key = format!("{}{}:{}", CHECKPOINT_PREFIX_KEY_PREFIX, order, prefix);
                trie.store_checkpoint_u64(&key, PrefixStatusCode::InProgress as u64)
                    .map_err(|e| CheckpointError::Trie(e.to_string()))?;
                keys_written += 1;
            }

            for prefix in &progress.failed_prefixes {
                let key = format!("{}{}:{}", CHECKPOINT_PREFIX_KEY_PREFIX, order, prefix);
                trie.store_checkpoint_u64(&key, PrefixStatusCode::Failed as u64)
                    .map_err(|e| CheckpointError::Trie(e.to_string()))?;
                keys_written += 1;
            }
        }

        Ok(keys_written)
    }

    /// Load checkpoint from a trie.
    ///
    /// # Arguments
    ///
    /// * `trie` - The trie to load the checkpoint from
    ///
    /// # Returns
    ///
    /// `Some(checkpoint)` if checkpoint data exists, `None` if no checkpoint in trie.
    pub fn load_from_trie<T>(trie: &T) -> Result<Option<Self>, CheckpointError>
    where
        T: TrieCheckpointStorage,
    {
        // Try to load version - if not present, no checkpoint exists
        let version = match trie.load_checkpoint_u64(CHECKPOINT_VERSION_KEY) {
            Ok(Some(v)) => v as u32,
            Ok(None) => return Ok(None),
            Err(e) => return Err(CheckpointError::Trie(e.to_string())),
        };

        // Check version
        if version > Self::CURRENT_VERSION {
            return Err(CheckpointError::UnsupportedVersion {
                found: version,
                max: Self::CURRENT_VERSION,
            });
        }

        // Load metadata
        let mkn_phase_ordinal = trie
            .load_checkpoint_u64(CHECKPOINT_MKN_PHASE_KEY)
            .map_err(|e| CheckpointError::Trie(e.to_string()))?
            .unwrap_or(0);
        let mkn_phase = MknPhase::from_ordinal(mkn_phase_ordinal);

        let byte_offset = trie
            .load_checkpoint_u64(CHECKPOINT_BYTE_OFFSET_KEY)
            .map_err(|e| CheckpointError::Trie(e.to_string()))?
            .unwrap_or(0);

        let timestamp_secs = trie
            .load_checkpoint_u64(CHECKPOINT_TIMESTAMP_KEY)
            .map_err(|e| CheckpointError::Trie(e.to_string()))?
            .unwrap_or(0);
        let timestamp = DateTime::from_timestamp(timestamp_secs as i64, 0)
            .unwrap_or_else(Utc::now);

        // Load stats
        let ngrams_processed = trie
            .load_checkpoint_u64(CHECKPOINT_NGRAMS_PROCESSED_KEY)
            .map_err(|e| CheckpointError::Trie(e.to_string()))?
            .unwrap_or(0);

        let unique_ngrams = trie
            .load_checkpoint_u64(CHECKPOINT_UNIQUE_NGRAMS_KEY)
            .map_err(|e| CheckpointError::Trie(e.to_string()))?
            .unwrap_or(0);

        let files_processed = trie
            .load_checkpoint_u64(CHECKPOINT_FILES_PROCESSED_KEY)
            .map_err(|e| CheckpointError::Trie(e.to_string()))?
            .unwrap_or(0) as u32;

        let bytes_downloaded = trie
            .load_checkpoint_u64(CHECKPOINT_BYTES_DOWNLOADED_KEY)
            .map_err(|e| CheckpointError::Trie(e.to_string()))?
            .unwrap_or(0);

        let elapsed_seconds = trie
            .load_checkpoint_u64(CHECKPOINT_ELAPSED_KEY)
            .map_err(|e| CheckpointError::Trie(e.to_string()))?
            .unwrap_or(0);

        // Load ngrams by order
        let mut ngrams_by_order = [0u64; 5];
        for order in 1..=5u8 {
            let key = format!("{}{}", CHECKPOINT_NGRAMS_BY_ORDER_PREFIX, order);
            if let Ok(Some(count)) = trie.load_checkpoint_u64(&key) {
                ngrams_by_order[order as usize - 1] = count;
            }
        }

        let stats = CheckpointStats {
            ngrams_processed,
            unique_ngrams,
            ngrams_by_order,
            bytes_downloaded,
            files_processed,
            elapsed_seconds,
        };

        // Load per-order progress using prefix iteration
        let mut order_progress = HashMap::new();

        for order in 1..=5u8 {
            // Check if order is complete
            let complete_key = format!("{}{}", CHECKPOINT_ORDER_COMPLETE_PREFIX, order);
            let is_complete = trie
                .load_checkpoint_u64(&complete_key)
                .map_err(|e| CheckpointError::Trie(e.to_string()))?
                .map(|v| v == 1)
                .unwrap_or(false);

            // Get all prefix statuses for this order
            let prefix_key_prefix = format!("{}{}:", CHECKPOINT_PREFIX_KEY_PREFIX, order);
            let prefix_entries = trie
                .iter_checkpoint_prefix(&prefix_key_prefix)
                .map_err(|e| CheckpointError::Trie(e.to_string()))?;

            let mut completed = Vec::new();
            let mut in_progress = Vec::new();
            let mut failed = Vec::new();

            for (key, status_code) in prefix_entries {
                // Extract prefix from key (after "...:order:prefix")
                if let Some(prefix) = key.strip_prefix(&prefix_key_prefix) {
                    match PrefixStatusCode::from_u64(status_code) {
                        Some(PrefixStatusCode::Completed) => completed.push(prefix.to_string()),
                        Some(PrefixStatusCode::InProgress) => in_progress.push(prefix.to_string()),
                        Some(PrefixStatusCode::Failed) => failed.push(prefix.to_string()),
                        None => {}
                    }
                }
            }

            // Only add order if it has any data
            if is_complete || !completed.is_empty() || !in_progress.is_empty() || !failed.is_empty() {
                order_progress.insert(
                    order,
                    OrderProgress {
                        completed_prefixes: completed,
                        in_progress_prefixes: in_progress,
                        failed_prefixes: failed,
                        is_complete,
                        ngrams_processed: ngrams_by_order[order as usize - 1],
                    },
                );
            }
        }

        Ok(Some(Self {
            version,
            order_progress,
            current_prefix: None, // Current prefix not stored in trie-based checkpoint
            byte_offset,
            mkn_phase,
            stats,
            timestamp,
        }))
    }

    /// Check if a trie contains checkpoint data.
    pub fn exists_in_trie<T>(trie: &T) -> bool
    where
        T: TrieCheckpointStorage,
    {
        trie.load_checkpoint_u64(CHECKPOINT_VERSION_KEY)
            .map(|opt| opt.is_some())
            .unwrap_or(false)
    }

    /// Delete checkpoint data from a trie.
    ///
    /// Uses prefix deletion to remove all checkpoint keys efficiently.
    pub fn delete_from_trie<T>(trie: &mut T) -> Result<usize, CheckpointError>
    where
        T: TrieCheckpointStorage,
    {
        trie.delete_checkpoint_prefix(CHECKPOINT_KEY_PREFIX)
            .map_err(|e| CheckpointError::Trie(e.to_string()))
    }

    /// Update a single prefix status in the trie.
    ///
    /// This is more efficient than saving the entire checkpoint when
    /// only a single prefix status changes.
    pub fn save_prefix_status_to_trie<T>(
        &self,
        trie: &mut T,
        order: u8,
        prefix: &str,
        status: PrefixStatusCode,
    ) -> Result<(), CheckpointError>
    where
        T: TrieCheckpointStorage,
    {
        let key = format!("{}{}:{}", CHECKPOINT_PREFIX_KEY_PREFIX, order, prefix);
        trie.store_checkpoint_u64(&key, status as u64)
            .map_err(|e| CheckpointError::Trie(e.to_string()))?;
        Ok(())
    }

    /// Remove a prefix status from the trie (e.g., when retrying a failed prefix).
    pub fn remove_prefix_status_from_trie<T>(
        trie: &mut T,
        order: u8,
        prefix: &str,
    ) -> Result<bool, CheckpointError>
    where
        T: TrieCheckpointStorage,
    {
        let key = format!("{}{}:{}", CHECKPOINT_PREFIX_KEY_PREFIX, order, prefix);
        trie.delete_checkpoint_key(&key)
            .map_err(|e| CheckpointError::Trie(e.to_string()))
    }
}

/// Trait for tries that support checkpoint storage.
///
/// This trait abstracts the trie operations needed for checkpoint storage,
/// allowing the checkpoint module to work with different trie implementations.
///
/// Since tries typically store u64 values, this trait uses u64 for storage.
pub trait TrieCheckpointStorage {
    /// Error type for trie operations.
    type Error: std::error::Error;

    /// Store a checkpoint key with u64 value.
    fn store_checkpoint_u64(&mut self, key: &str, value: u64) -> Result<(), Self::Error>;

    /// Load a checkpoint key's u64 value.
    fn load_checkpoint_u64(&self, key: &str) -> Result<Option<u64>, Self::Error>;

    /// Delete a checkpoint key.
    fn delete_checkpoint_key(&mut self, key: &str) -> Result<bool, Self::Error>;

    /// Delete all checkpoint keys with a given prefix.
    fn delete_checkpoint_prefix(&mut self, prefix: &str) -> Result<usize, Self::Error>;

    /// Iterate over all checkpoint keys with a given prefix.
    fn iter_checkpoint_prefix(&self, prefix: &str) -> Result<Vec<(String, u64)>, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_checkpoint_new() {
        let cp = ImportCheckpoint::new();
        assert_eq!(cp.version, ImportCheckpoint::CURRENT_VERSION);
        assert!(cp.order_progress.is_empty());
    }

    #[test]
    fn test_checkpoint_save_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.checkpoint.json");

        let mut cp = ImportCheckpoint::new();
        cp.complete_order(1); // Mark order 1 as complete
        cp.complete_prefix(2, "aa"); // Order 2 has prefix "aa" done
        cp.add_ngrams(2, 12345);

        cp.save(&path).unwrap();
        assert!(path.exists());

        let loaded = ImportCheckpoint::load(&path).unwrap();
        assert!(loaded.is_order_complete(1));
        assert!(!loaded.is_order_complete(2));
        assert!(!loaded.needs_prefix(2, "aa")); // Already done
        assert!(loaded.needs_prefix(2, "ab")); // Not done yet
        assert_eq!(loaded.stats.ngrams_processed, 12345);
    }

    #[test]
    fn test_needs_prefix() {
        let mut cp = ImportCheckpoint::new();
        cp.complete_prefix(2, "aa");
        cp.complete_prefix(2, "ab");

        // Order with no progress - always needs
        assert!(cp.needs_prefix(1, "a"));
        assert!(cp.needs_prefix(3, "aaa"));

        // Order 2 has some prefixes completed
        assert!(!cp.needs_prefix(2, "aa"));
        assert!(!cp.needs_prefix(2, "ab"));

        // Order 2, not completed
        assert!(cp.needs_prefix(2, "ac"));

        // Mark order 2 as complete - now all prefixes are "done"
        cp.complete_order(2);
        assert!(!cp.needs_prefix(2, "ac"));
        assert!(!cp.needs_prefix(2, "zz"));
    }

    #[test]
    fn test_complete_prefix() {
        let mut cp = ImportCheckpoint::new();

        cp.complete_prefix(2, "aa");

        assert_eq!(cp.completed_prefix_count(2), 1);
        assert!(!cp.needs_prefix(2, "aa"));
        assert_eq!(cp.stats.files_processed, 1);
    }

    #[test]
    fn test_complete_order() {
        let mut cp = ImportCheckpoint::new();
        cp.complete_prefix(1, "a");
        cp.complete_prefix(1, "b");

        cp.complete_order(1);

        assert!(cp.is_order_complete(1));
        // v2: Prefixes are NOT cleared, kept for verification
        assert_eq!(cp.completed_prefix_count(1), 2);
    }

    #[test]
    fn test_overlapping_orders() {
        let mut cp = ImportCheckpoint::new();

        // Simulate overlapping processing: some 1-grams done, some 2-grams started
        cp.complete_prefix(1, "a");
        cp.complete_prefix(1, "b");
        cp.complete_prefix(2, "aa"); // 2-grams started before 1-grams finished!

        assert!(!cp.is_order_complete(1));
        assert!(!cp.is_order_complete(2));
        assert!(!cp.needs_prefix(1, "a"));
        assert!(!cp.needs_prefix(2, "aa"));
        assert!(cp.needs_prefix(1, "c"));
        assert!(cp.needs_prefix(2, "ab"));

        // Finish order 1
        cp.complete_order(1);
        assert!(cp.is_order_complete(1));
        assert!(!cp.is_order_complete(2));

        // Order 2 still in progress
        assert_eq!(cp.orders_in_progress(), vec![2]);
        assert_eq!(cp.completed_orders(), vec![1]);
    }

    #[test]
    fn test_v1_migration() {
        // Create a v1-style checkpoint JSON
        let v1_json = r#"{
            "version": 1,
            "completed_orders": [1],
            "current_order": 2,
            "completed_prefixes": ["aa", "ab"],
            "current_prefix": null,
            "byte_offset": 0,
            "mkn_phase": "NotStarted",
            "stats": {
                "ngrams_processed": 12345,
                "ngrams_by_order": [0, 0, 0, 0, 0],
                "bytes_downloaded": 0,
                "files_processed": 3,
                "elapsed_seconds": 100
            },
            "timestamp": "2024-01-01T00:00:00Z"
        }"#;

        let dir = tempdir().unwrap();
        let path = dir.path().join("v1.checkpoint.json");
        std::fs::write(&path, v1_json).unwrap();

        let loaded = ImportCheckpoint::load(&path).unwrap();

        // Should be migrated to v2
        assert_eq!(loaded.version, ImportCheckpoint::CURRENT_VERSION);

        // Order 1 should be marked complete
        assert!(loaded.is_order_complete(1));

        // Order 2 should have the prefixes but not be complete
        assert!(!loaded.is_order_complete(2));
        assert!(!loaded.needs_prefix(2, "aa"));
        assert!(!loaded.needs_prefix(2, "ab"));
        assert!(loaded.needs_prefix(2, "ac"));

        // Stats preserved
        assert_eq!(loaded.stats.ngrams_processed, 12345);
        assert_eq!(loaded.stats.files_processed, 3);
    }

    #[test]
    fn test_prefix_lifecycle() {
        let mut cp = ImportCheckpoint::new();

        // Start a prefix (in_progress)
        cp.start_prefix(2, "aa");
        assert!(cp.is_in_progress(2, "aa"));
        assert!(!cp.needs_prefix(2, "aa")); // In progress, don't requeue
        assert_eq!(cp.in_progress_prefixes(2), vec!["aa".to_string()]);

        // Complete the prefix
        cp.complete_prefix(2, "aa");
        assert!(!cp.is_in_progress(2, "aa"));
        assert!(!cp.needs_prefix(2, "aa")); // Completed
        assert!(cp.in_progress_prefixes(2).is_empty());
        assert_eq!(cp.completed_prefix_count(2), 1);
    }

    #[test]
    fn test_prefix_failure() {
        let mut cp = ImportCheckpoint::new();

        // Start a prefix
        cp.start_prefix(2, "aa");
        assert!(cp.is_in_progress(2, "aa"));

        // Fail the prefix
        cp.fail_prefix(2, "aa");
        assert!(!cp.is_in_progress(2, "aa"));
        assert!(cp.is_failed_prefix(2, "aa"));
        assert_eq!(cp.failed_prefix_count(2), 1);

        // Failed prefixes still "need" processing (for retry)
        assert!(cp.needs_prefix(2, "aa"));
    }

    #[test]
    fn test_recover_in_progress() {
        let mut cp = ImportCheckpoint::new();

        // Simulate crash with in-progress prefixes
        cp.start_prefix(2, "aa");
        cp.start_prefix(2, "ab");
        assert_eq!(cp.in_progress_prefixes(2).len(), 2);

        // Recover: move to failed
        cp.recover_in_progress_as_failed(2);
        assert!(cp.in_progress_prefixes(2).is_empty());
        assert_eq!(cp.failed_prefix_count(2), 2);
        assert!(cp.is_failed_prefix(2, "aa"));
        assert!(cp.is_failed_prefix(2, "ab"));
    }

    #[test]
    fn test_clear_failed() {
        let mut cp = ImportCheckpoint::new();

        cp.start_prefix(2, "aa");
        cp.fail_prefix(2, "aa");
        assert!(cp.is_failed_prefix(2, "aa"));

        // Clear for retry
        cp.clear_failed(2, "aa");
        assert!(!cp.is_failed_prefix(2, "aa"));
        assert!(cp.needs_prefix(2, "aa")); // Now needs processing again
    }

    #[test]
    fn test_start_prefix_clears_other_states() {
        let mut cp = ImportCheckpoint::new();

        // Complete a prefix
        cp.complete_prefix(2, "aa");
        assert!(!cp.needs_prefix(2, "aa"));

        // Starting it again should move it back to in_progress
        cp.start_prefix(2, "aa");
        assert!(cp.is_in_progress(2, "aa"));
        assert!(!cp.order_progress.get(&2).unwrap().completed_prefixes.contains(&"aa".to_string()));
    }

    #[test]
    fn test_failed_prefix_save_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.checkpoint.json");

        let mut cp = ImportCheckpoint::new();
        cp.start_prefix(2, "aa");
        cp.fail_prefix(2, "aa");
        cp.start_prefix(2, "ab");  // In progress

        cp.save(&path).unwrap();

        let loaded = ImportCheckpoint::load(&path).unwrap();
        assert!(loaded.is_failed_prefix(2, "aa"));
        assert!(loaded.is_in_progress(2, "ab"));
        assert_eq!(loaded.failed_prefix_count(2), 1);
    }

    #[test]
    fn test_progress_summary_with_failures() {
        let mut cp = ImportCheckpoint::new();
        cp.complete_prefix(2, "aa");
        cp.fail_prefix(2, "ab");

        let summary = cp.progress_summary();
        assert!(summary.contains("Failed: 1"));
    }
}
