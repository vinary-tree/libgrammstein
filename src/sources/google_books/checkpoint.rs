//! Checkpoint and resume support for Google Books import.
//!
//! Long-running imports (hours for full datasets) need checkpoint support
//! to handle interruptions gracefully without losing progress.
//!
//! ## Checkpoint Format Versions
//!
//! - **Version 1**: Single-order tracking with `current_order` and `completed_prefixes`
//! - **Version 2**: Per-order tracking with `order_progress` HashMap for overlapping order processing
//!
//! ## Performance Optimization
//!
//! The `CheckpointStateMachine.tla` proof establishes the `DisjointSets` invariant:
//! each prefix has exactly one state at any time. This enables O(1) state lookups
//! using a HashMap instead of O(n) Vec searches. The optimization maintains
//! backward-compatible JSON serialization.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// State of a prefix in the checkpoint.
///
/// The TLA+ `DisjointSets` invariant proves each prefix is in exactly one state.
/// This enables O(1) lookups via HashMap<String, PrefixState> instead of
/// O(n) Vec::contains() on three separate vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixState {
    /// Prefix is currently being processed.
    InProgress,
    /// Prefix has been successfully processed.
    Completed,
    /// Prefix failed after exhausting retries.
    Failed,
}

/// Progress tracking for a single n-gram order.
///
/// Allows multiple orders to be tracked independently for overlapping
/// order processing (e.g., workers can process 2-grams while others
/// finish 1-grams).
///
/// ## Prefix States
///
/// A prefix can be in one of four states:
/// - **Not started**: Not in the state map (needs processing)
/// - **In progress**: `PrefixState::InProgress` (started but not finished)
/// - **Completed**: `PrefixState::Completed` (successfully processed)
/// - **Failed**: `PrefixState::Failed` (failed after exhausting retries)
///
/// On resume:
/// - Completed prefixes are skipped
/// - In-progress prefixes are cleared and retried (data may be partial)
/// - Failed prefixes are retried on subsequent runs
///
/// ## Implementation Note
///
/// Uses HashMap<String, PrefixState> internally for O(1) state lookups.
/// The TLA+ `DisjointSets` invariant guarantees each prefix has exactly
/// one state, making this representation sound.
/// JSON serialization uses Vec format for backward compatibility.
#[derive(Clone, Debug, Default)]
pub struct OrderProgress {
    /// Map from prefix to its current state.
    ///
    /// Prefixes not in the map are in "NotStarted" state.
    /// This provides O(1) state lookup instead of O(n) Vec::contains().
    prefix_states: HashMap<String, PrefixState>,

    /// Whether this order is fully complete (all prefixes processed).
    pub is_complete: bool,

    /// N-grams processed for this order.
    pub ngrams_processed: u64,
}

impl OrderProgress {
    /// Get all completed prefixes.
    pub fn completed_prefixes(&self) -> impl Iterator<Item = &String> {
        self.prefix_states.iter()
            .filter(|(_, s)| **s == PrefixState::Completed)
            .map(|(p, _)| p)
    }

    /// Get all in-progress prefixes.
    pub fn in_progress_prefixes(&self) -> impl Iterator<Item = &String> {
        self.prefix_states.iter()
            .filter(|(_, s)| **s == PrefixState::InProgress)
            .map(|(p, _)| p)
    }

    /// Get all failed prefixes.
    pub fn failed_prefixes(&self) -> impl Iterator<Item = &String> {
        self.prefix_states.iter()
            .filter(|(_, s)| **s == PrefixState::Failed)
            .map(|(p, _)| p)
    }

    /// Get the state of a prefix.
    pub fn get_state(&self, prefix: &str) -> Option<PrefixState> {
        self.prefix_states.get(prefix).copied()
    }

    /// Set the state of a prefix.
    pub fn set_state(&mut self, prefix: String, state: PrefixState) {
        self.prefix_states.insert(prefix, state);
    }

    /// Remove a prefix from tracking (return to NotStarted).
    pub fn clear_state(&mut self, prefix: &str) {
        self.prefix_states.remove(prefix);
    }

    /// Count prefixes in a given state.
    pub fn count_state(&self, state: PrefixState) -> usize {
        self.prefix_states.values().filter(|s| **s == state).count()
    }
}

/// Serialization helper for backward-compatible JSON format.
#[derive(Serialize, Deserialize)]
struct OrderProgressSerde {
    completed_prefixes: Vec<String>,
    #[serde(default)]
    in_progress_prefixes: Vec<String>,
    #[serde(default)]
    failed_prefixes: Vec<String>,
    is_complete: bool,
    ngrams_processed: u64,
}

impl Serialize for OrderProgress {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let serde_repr = OrderProgressSerde {
            completed_prefixes: self.completed_prefixes().cloned().collect(),
            in_progress_prefixes: self.in_progress_prefixes().cloned().collect(),
            failed_prefixes: self.failed_prefixes().cloned().collect(),
            is_complete: self.is_complete,
            ngrams_processed: self.ngrams_processed,
        };
        serde_repr.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for OrderProgress {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let serde_repr = OrderProgressSerde::deserialize(deserializer)?;

        let mut prefix_states = HashMap::with_capacity(
            serde_repr.completed_prefixes.len()
                + serde_repr.in_progress_prefixes.len()
                + serde_repr.failed_prefixes.len(),
        );

        for prefix in serde_repr.completed_prefixes {
            prefix_states.insert(prefix, PrefixState::Completed);
        }
        for prefix in serde_repr.in_progress_prefixes {
            prefix_states.insert(prefix, PrefixState::InProgress);
        }
        for prefix in serde_repr.failed_prefixes {
            prefix_states.insert(prefix, PrefixState::Failed);
        }

        Ok(OrderProgress {
            prefix_states,
            is_complete: serde_repr.is_complete,
            ngrams_processed: serde_repr.ngrams_processed,
        })
    }
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
    ///
    /// - Version 1: Single-order tracking (legacy)
    /// - Version 2: Per-order tracking with key-per-prefix trie storage
    /// - Version 3: Bitmap-based trie storage (22 u64s per order vs 676 keys)
    pub const CURRENT_VERSION: u32 = 3;

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
            let progress = OrderProgress {
                prefix_states: HashMap::new(),
                is_complete: true,
                ngrams_processed: 0, // We don't have per-order stats in v1
            };
            order_progress.insert(order, progress);
        }

        // Add current order's progress (if it has any completed prefixes)
        if !v1.completed_prefixes.is_empty() || v1.current_prefix.is_some() {
            let mut prefix_states = HashMap::with_capacity(v1.completed_prefixes.len());
            for prefix in v1.completed_prefixes {
                prefix_states.insert(prefix, PrefixState::Completed);
            }
            let progress = OrderProgress {
                prefix_states,
                is_complete: false,
                ngrams_processed: 0,
            };
            order_progress.insert(v1.current_order, progress);
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
    ///
    /// O(1) operation due to HashMap-based state storage (TLA+ DisjointSets invariant).
    pub fn start_prefix(&mut self, order: u8, prefix: &str) {
        let progress = self.order_progress.entry(order).or_default();
        // DisjointSets invariant: prefix can only be in one state
        // HashMap insert replaces any existing state
        progress.set_state(prefix.to_string(), PrefixState::InProgress);
    }

    /// Mark a prefix as completed for a specific order.
    ///
    /// Moves the prefix from in_progress to completed.
    /// This is the v2 version that supports overlapping order processing.
    ///
    /// O(1) operation due to HashMap-based state storage (TLA+ DisjointSets invariant).
    pub fn complete_prefix(&mut self, order: u8, prefix: &str) {
        let progress = self.order_progress.entry(order).or_default();
        // DisjointSets invariant: HashMap insert replaces any existing state
        progress.set_state(prefix.to_string(), PrefixState::Completed);
        self.stats.files_processed += 1;
    }

    /// Mark a prefix as failed (exhausted all retries).
    ///
    /// Moves the prefix from in_progress to failed.
    /// Failed prefixes are skipped on the current run but will be
    /// retried on subsequent runs.
    ///
    /// O(1) operation due to HashMap-based state storage (TLA+ DisjointSets invariant).
    pub fn fail_prefix(&mut self, order: u8, prefix: &str) {
        let progress = self.order_progress.entry(order).or_default();
        // DisjointSets invariant: HashMap insert replaces any existing state
        progress.set_state(prefix.to_string(), PrefixState::Failed);
    }

    /// Clear a prefix from the failed list (for retry on subsequent run).
    ///
    /// O(1) operation due to HashMap-based state storage.
    pub fn clear_failed(&mut self, order: u8, prefix: &str) {
        if let Some(progress) = self.order_progress.get_mut(&order) {
            // Only clear if currently Failed (don't disturb other states)
            if progress.get_state(prefix) == Some(PrefixState::Failed) {
                progress.clear_state(prefix);
            }
        }
    }

    /// Check if a prefix is currently marked as in-progress.
    ///
    /// O(1) operation due to HashMap-based state storage.
    pub fn is_in_progress(&self, order: u8, prefix: &str) -> bool {
        self.order_progress
            .get(&order)
            .and_then(|p| p.get_state(prefix))
            .map(|s| s == PrefixState::InProgress)
            .unwrap_or(false)
    }

    /// Check if a prefix has failed (exhausted retries).
    ///
    /// O(1) operation due to HashMap-based state storage.
    pub fn is_failed_prefix(&self, order: u8, prefix: &str) -> bool {
        self.order_progress
            .get(&order)
            .and_then(|p| p.get_state(prefix))
            .map(|s| s == PrefixState::Failed)
            .unwrap_or(false)
    }

    /// Get all in-progress prefixes for an order.
    ///
    /// On resume, these should be cleared and retried.
    pub fn in_progress_prefixes(&self, order: u8) -> Vec<String> {
        self.order_progress
            .get(&order)
            .map(|p| p.in_progress_prefixes().cloned().collect())
            .unwrap_or_default()
    }

    /// Get all failed prefixes for an order.
    ///
    /// These should be retried on subsequent runs.
    pub fn failed_prefixes(&self, order: u8) -> Vec<String> {
        self.order_progress
            .get(&order)
            .map(|p| p.failed_prefixes().cloned().collect())
            .unwrap_or_default()
    }

    /// Move all in-progress prefixes to failed (for crash recovery).
    ///
    /// Call this on resume when in-progress prefixes are detected.
    /// The caller should clear partial data from the trie before retrying.
    pub fn recover_in_progress_as_failed(&mut self, order: u8) {
        if let Some(progress) = self.order_progress.get_mut(&order) {
            // Collect in-progress prefixes first to avoid borrow conflicts
            let in_progress: Vec<String> = progress.in_progress_prefixes().cloned().collect();
            for prefix in in_progress {
                progress.set_state(prefix, PrefixState::Failed);
            }
        }
    }

    /// Get count of failed prefixes for an order.
    ///
    /// O(n) where n is number of tracked prefixes for this order.
    pub fn failed_prefix_count(&self, order: u8) -> usize {
        self.order_progress
            .get(&order)
            .map(|p| p.count_state(PrefixState::Failed))
            .unwrap_or(0)
    }

    /// Get total count of failed prefixes across all orders.
    pub fn total_failed_prefix_count(&self) -> usize {
        self.order_progress
            .values()
            .map(|p| p.count_state(PrefixState::Failed))
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
    ///
    /// # Errors
    ///
    /// Returns `CheckpointError::OrderHasInProgressPrefixes` if any prefixes
    /// for this order are still in progress. This enforces the TLA+ invariant:
    /// `order_complete[o] => in_progress_prefixes[o] = {}`
    ///
    /// # Precondition
    ///
    /// All prefixes for this order must be either completed or failed (not in progress).
    /// Call `complete_prefix()` or `fail_prefix()` for all in-progress prefixes first.
    pub fn complete_order(&mut self, order: u8) -> Result<(), CheckpointError> {
        // TLA+ invariant: CompletedOrderNoInProgress
        // order_complete[o] => in_progress_prefixes[o] = {}
        if let Some(progress) = self.order_progress.get(&order) {
            let in_progress_count = progress.count_state(PrefixState::InProgress);
            if in_progress_count > 0 {
                return Err(CheckpointError::OrderHasInProgressPrefixes {
                    order,
                    count: in_progress_count,
                });
            }
        }
        let progress = self.order_progress.entry(order).or_default();
        progress.is_complete = true;
        // Don't clear completed_prefixes - keep for resume verification
        Ok(())
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
    ///
    /// O(1) operation due to HashMap-based state storage (TLA+ DisjointSets invariant).
    pub fn needs_prefix(&self, order: u8, prefix: &str) -> bool {
        self.order_progress
            .get(&order)
            .map(|p| {
                if p.is_complete {
                    return false;
                }
                match p.get_state(prefix) {
                    None => true,           // Not started - needs processing
                    Some(PrefixState::Failed) => true,  // Failed - retry
                    Some(PrefixState::Completed) => false,
                    Some(PrefixState::InProgress) => false,
                }
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
            .map(|p| p.count_state(PrefixState::Completed))
            .unwrap_or(0)
    }

    /// Get the total count of completed prefixes across all orders.
    pub fn total_completed_prefix_count(&self) -> usize {
        self.order_progress
            .values()
            .map(|p| p.count_state(PrefixState::Completed))
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
            .map(|(order, p)| format!("{}:{}", order, p.count_state(PrefixState::Completed)))
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

    /// Cannot complete order with in-progress prefixes.
    ///
    /// TLA+ invariant: `order_complete[o] => in_progress_prefixes[o] = {}`
    /// An order cannot be marked complete while any prefixes are still in progress.
    #[error("Cannot complete order {order}: {count} prefixes still in progress")]
    OrderHasInProgressPrefixes { order: u8, count: usize },
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

/// Key prefix for bitmap chunks. Format: "\x00__ckpt__:bitmap:{order}:{chunk}"
pub const CHECKPOINT_BITMAP_PREFIX: &str = "\x00__ckpt__:bitmap:";

/// Key for per-order n-gram count in bitmap format. Format: "\x00__ckpt__:order_ngrams:{order}"
pub const CHECKPOINT_ORDER_NGRAMS_PREFIX: &str = "\x00__ckpt__:order_ngrams:";

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

// =============================================================================
// Bitmap Encoding/Decoding for Compact Trie Storage
// =============================================================================
//
// Version 3 uses a bitmap format where each prefix state occupies 2 bits:
//   0b00 = NotStarted (not in map)
//   0b01 = InProgress
//   0b10 = Completed
//   0b11 = Failed
//
// Each u64 holds 32 prefix states (64 bits / 2 bits = 32 prefixes).
// For 676 prefixes (order 2+), we need ceil(676/32) = 22 u64 chunks.
//
// This reduces storage from ~22KB (676 keys) to ~180 bytes (22 u64s).
// =============================================================================

/// Bitmap state encoding (2 bits per prefix).
const BITMAP_STATE_NOT_STARTED: u8 = 0b00;
const BITMAP_STATE_IN_PROGRESS: u8 = 0b01;
const BITMAP_STATE_COMPLETED: u8 = 0b10;
const BITMAP_STATE_FAILED: u8 = 0b11;

/// Number of prefix states that fit in one u64 (64 bits / 2 bits = 32).
const PREFIXES_PER_CHUNK: usize = 32;

/// Convert prefix string to index (0-based).
///
/// - Single char: "a" -> 0, "z" -> 25
/// - Two char: "aa" -> 0, "ab" -> 1, "zz" -> 675
///
/// Returns `None` for invalid prefixes.
fn prefix_to_index(prefix: &str) -> Option<u16> {
    let bytes = prefix.as_bytes();
    match bytes.len() {
        1 => {
            let c = bytes[0];
            if c >= b'a' && c <= b'z' {
                Some((c - b'a') as u16)
            } else {
                None
            }
        }
        2 => {
            let c1 = bytes[0];
            let c2 = bytes[1];
            if c1 >= b'a' && c1 <= b'z' && c2 >= b'a' && c2 <= b'z' {
                Some(((c1 - b'a') as u16) * 26 + ((c2 - b'a') as u16))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Convert index back to prefix string.
///
/// - `prefix_len=1`: 0 -> "a", 25 -> "z"
/// - `prefix_len=2`: 0 -> "aa", 1 -> "ab", 675 -> "zz"
fn index_to_prefix(index: u16, prefix_len: u8) -> String {
    match prefix_len {
        1 => {
            debug_assert!(index < 26, "Index {} out of range for single-char prefix", index);
            let c = (b'a' + index as u8) as char;
            c.to_string()
        }
        2 => {
            debug_assert!(index < 676, "Index {} out of range for two-char prefix", index);
            let c1 = (b'a' + (index / 26) as u8) as char;
            let c2 = (b'a' + (index % 26) as u8) as char;
            format!("{}{}", c1, c2)
        }
        _ => panic!("Unsupported prefix length: {}", prefix_len),
    }
}

/// Get the prefix length for a given n-gram order.
///
/// - Order 1: single-char prefixes (26 total)
/// - Order 2+: two-char prefixes (676 total)
fn prefix_len_for_order(order: u8) -> u8 {
    if order == 1 { 1 } else { 2 }
}

/// Get the maximum index for a given prefix length.
fn max_index_for_prefix_len(prefix_len: u8) -> u16 {
    if prefix_len == 1 { 26 } else { 676 }
}

/// Get the number of u64 chunks needed for a given prefix length.
fn num_chunks_for_prefix_len(prefix_len: u8) -> usize {
    let max_index = max_index_for_prefix_len(prefix_len) as usize;
    (max_index + PREFIXES_PER_CHUNK - 1) / PREFIXES_PER_CHUNK
}

/// Pack prefix states into u64 bitmap chunks.
///
/// Each prefix state occupies 2 bits within a u64. States are packed
/// in index order, with lower indices in lower bits.
fn pack_states(states: &HashMap<String, PrefixState>, prefix_len: u8) -> Vec<u64> {
    let num_chunks = num_chunks_for_prefix_len(prefix_len);
    let mut chunks = vec![0u64; num_chunks];

    for (prefix, state) in states {
        if let Some(index) = prefix_to_index(prefix) {
            let chunk_idx = (index as usize) / PREFIXES_PER_CHUNK;
            let bit_pos = ((index as usize) % PREFIXES_PER_CHUNK) * 2;
            let state_bits = match state {
                PrefixState::InProgress => BITMAP_STATE_IN_PROGRESS as u64,
                PrefixState::Completed => BITMAP_STATE_COMPLETED as u64,
                PrefixState::Failed => BITMAP_STATE_FAILED as u64,
            };
            chunks[chunk_idx] |= state_bits << bit_pos;
        } else {
            // Non-standard prefix - log warning and skip
            log::warn!(
                "Skipping non-standard prefix '{}' during bitmap packing",
                prefix
            );
        }
    }
    chunks
}

/// Unpack u64 bitmap chunks into prefix states.
///
/// Returns a HashMap containing only prefixes with non-zero states
/// (NotStarted prefixes are not included in the map).
fn unpack_states(chunks: &[u64], prefix_len: u8) -> HashMap<String, PrefixState> {
    let max_index = max_index_for_prefix_len(prefix_len);
    let mut states = HashMap::new();

    for index in 0..max_index {
        let chunk_idx = (index as usize) / PREFIXES_PER_CHUNK;
        let bit_pos = ((index as usize) % PREFIXES_PER_CHUNK) * 2;

        if chunk_idx < chunks.len() {
            let state_bits = ((chunks[chunk_idx] >> bit_pos) & 0b11) as u8;
            let state = match state_bits {
                BITMAP_STATE_NOT_STARTED => continue, // Don't add to map
                BITMAP_STATE_IN_PROGRESS => PrefixState::InProgress,
                BITMAP_STATE_COMPLETED => PrefixState::Completed,
                BITMAP_STATE_FAILED => PrefixState::Failed,
                _ => unreachable!("Invalid state bits: {}", state_bits),
            };
            let prefix = index_to_prefix(index, prefix_len);
            states.insert(prefix, state);
        }
    }
    states
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
    /// ## Version 3 Bitmap Format
    ///
    /// Version 3 uses a compact bitmap format where each prefix state occupies
    /// 2 bits within u64 chunks. This reduces storage from ~22KB (676 keys per
    /// order) to ~180 bytes (22 u64 chunks per order).
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

        // Always save as current version (v3 bitmap format)
        trie.store_checkpoint_u64(CHECKPOINT_VERSION_KEY, Self::CURRENT_VERSION as u64)
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

        // Store ngrams by order (global stats)
        for (idx, &count) in self.stats.ngrams_by_order.iter().enumerate() {
            let key = format!("{}{}", CHECKPOINT_NGRAMS_BY_ORDER_PREFIX, idx + 1);
            trie.store_checkpoint_u64(&key, count)
                .map_err(|e| CheckpointError::Trie(e.to_string()))?;
            keys_written += 1;
        }

        // Store per-order progress using bitmap format (v3)
        for (order, progress) in &self.order_progress {
            // Store order completion status
            if progress.is_complete {
                let key = format!("{}{}", CHECKPOINT_ORDER_COMPLETE_PREFIX, order);
                trie.store_checkpoint_u64(&key, 1)
                    .map_err(|e| CheckpointError::Trie(e.to_string()))?;
                keys_written += 1;
            }

            // Store per-order n-gram count
            let ngrams_key = format!("{}{}", CHECKPOINT_ORDER_NGRAMS_PREFIX, order);
            trie.store_checkpoint_u64(&ngrams_key, progress.ngrams_processed)
                .map_err(|e| CheckpointError::Trie(e.to_string()))?;
            keys_written += 1;

            // Pack prefix states into bitmap chunks
            let prefix_len = prefix_len_for_order(*order);
            let chunks = pack_states(&progress.prefix_states, prefix_len);

            // Store each chunk
            for (chunk_idx, &chunk_value) in chunks.iter().enumerate() {
                // Only store non-zero chunks (optimization for sparse states)
                if chunk_value != 0 {
                    let key = format!("{}{}:{}", CHECKPOINT_BITMAP_PREFIX, order, chunk_idx);
                    trie.store_checkpoint_u64(&key, chunk_value)
                        .map_err(|e| CheckpointError::Trie(e.to_string()))?;
                    keys_written += 1;
                }
            }
        }

        Ok(keys_written)
    }

    /// Load checkpoint from a trie.
    ///
    /// Supports both v2 (key-per-prefix) and v3 (bitmap) formats.
    /// v2 checkpoints are automatically migrated to v3 on next save.
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

        // Load metadata (common to all versions)
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

        // Load ngrams by order (global stats)
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

        // Load per-order progress based on version
        let order_progress = if version >= 3 {
            Self::load_order_progress_v3(trie, &ngrams_by_order)?
        } else {
            // v2: key-per-prefix format (backward compatibility)
            log::info!("Loading v2 checkpoint (key-per-prefix format), will migrate to v3 on save");
            Self::load_order_progress_v2(trie, &ngrams_by_order)?
        };

        Ok(Some(Self {
            version: Self::CURRENT_VERSION, // Always report as current version
            order_progress,
            current_prefix: None, // Current prefix not stored in trie-based checkpoint
            byte_offset,
            mkn_phase,
            stats,
            timestamp,
        }))
    }

    /// Load per-order progress using v3 bitmap format.
    fn load_order_progress_v3<T>(
        trie: &T,
        ngrams_by_order: &[u64; 5],
    ) -> Result<HashMap<u8, OrderProgress>, CheckpointError>
    where
        T: TrieCheckpointStorage,
    {
        let mut order_progress = HashMap::new();

        for order in 1..=5u8 {
            // Check if order is complete
            let complete_key = format!("{}{}", CHECKPOINT_ORDER_COMPLETE_PREFIX, order);
            let is_complete = trie
                .load_checkpoint_u64(&complete_key)
                .map_err(|e| CheckpointError::Trie(e.to_string()))?
                .map(|v| v == 1)
                .unwrap_or(false);

            // Load per-order n-gram count (v3 stores this separately)
            let ngrams_key = format!("{}{}", CHECKPOINT_ORDER_NGRAMS_PREFIX, order);
            let order_ngrams = trie
                .load_checkpoint_u64(&ngrams_key)
                .map_err(|e| CheckpointError::Trie(e.to_string()))?
                .unwrap_or(ngrams_by_order[order as usize - 1]);

            // Load bitmap chunks for this order
            let prefix_len = prefix_len_for_order(order);
            let num_chunks = num_chunks_for_prefix_len(prefix_len);
            let mut chunks = vec![0u64; num_chunks];
            let mut has_any_chunks = false;

            for chunk_idx in 0..num_chunks {
                let key = format!("{}{}:{}", CHECKPOINT_BITMAP_PREFIX, order, chunk_idx);
                if let Ok(Some(chunk_value)) = trie.load_checkpoint_u64(&key) {
                    chunks[chunk_idx] = chunk_value;
                    if chunk_value != 0 {
                        has_any_chunks = true;
                    }
                }
            }

            // Unpack bitmap chunks into prefix states
            let prefix_states = if has_any_chunks {
                unpack_states(&chunks, prefix_len)
            } else {
                HashMap::new()
            };

            // Only add order if it has any data
            if is_complete || !prefix_states.is_empty() {
                order_progress.insert(
                    order,
                    OrderProgress {
                        prefix_states,
                        is_complete,
                        ngrams_processed: order_ngrams,
                    },
                );
            }
        }

        Ok(order_progress)
    }

    /// Load per-order progress using v2 key-per-prefix format (backward compatibility).
    fn load_order_progress_v2<T>(
        trie: &T,
        ngrams_by_order: &[u64; 5],
    ) -> Result<HashMap<u8, OrderProgress>, CheckpointError>
    where
        T: TrieCheckpointStorage,
    {
        let mut order_progress = HashMap::new();

        for order in 1..=5u8 {
            // Check if order is complete
            let complete_key = format!("{}{}", CHECKPOINT_ORDER_COMPLETE_PREFIX, order);
            let is_complete = trie
                .load_checkpoint_u64(&complete_key)
                .map_err(|e| CheckpointError::Trie(e.to_string()))?
                .map(|v| v == 1)
                .unwrap_or(false);

            // Get all prefix statuses for this order using key iteration
            let prefix_key_prefix = format!("{}{}:", CHECKPOINT_PREFIX_KEY_PREFIX, order);
            let prefix_entries = trie
                .iter_checkpoint_prefix(&prefix_key_prefix)
                .map_err(|e| CheckpointError::Trie(e.to_string()))?;

            let mut prefix_states = HashMap::new();

            for (key, status_code) in prefix_entries {
                // Extract prefix from key (after "...:order:prefix")
                if let Some(prefix) = key.strip_prefix(&prefix_key_prefix) {
                    match PrefixStatusCode::from_u64(status_code) {
                        Some(PrefixStatusCode::Completed) => {
                            prefix_states.insert(prefix.to_string(), PrefixState::Completed);
                        }
                        Some(PrefixStatusCode::InProgress) => {
                            prefix_states.insert(prefix.to_string(), PrefixState::InProgress);
                        }
                        Some(PrefixStatusCode::Failed) => {
                            prefix_states.insert(prefix.to_string(), PrefixState::Failed);
                        }
                        None => {}
                    }
                }
            }

            // Only add order if it has any data
            if is_complete || !prefix_states.is_empty() {
                order_progress.insert(
                    order,
                    OrderProgress {
                        prefix_states,
                        is_complete,
                        ngrams_processed: ngrams_by_order[order as usize - 1],
                    },
                );
            }
        }

        Ok(order_progress)
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
        cp.complete_order(1).expect("no in-progress prefixes for order 1"); // Mark order 1 as complete
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
        cp.complete_order(2).expect("no in-progress prefixes for order 2");
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

        cp.complete_order(1).expect("no in-progress prefixes for order 1");

        assert!(cp.is_order_complete(1));
        // v2: Prefixes are NOT cleared, kept for verification
        assert_eq!(cp.completed_prefix_count(1), 2);
    }

    #[test]
    fn test_complete_order_with_in_progress_fails() {
        let mut cp = ImportCheckpoint::new();
        cp.start_prefix(1, "a");  // Mark prefix as in-progress

        // Attempting to complete order should fail (TLA+ invariant violation)
        let result = cp.complete_order(1);
        assert!(result.is_err());
        match result {
            Err(CheckpointError::OrderHasInProgressPrefixes { order, count }) => {
                assert_eq!(order, 1);
                assert_eq!(count, 1);
            }
            _ => panic!("Expected OrderHasInProgressPrefixes error"),
        }

        // After completing the prefix, order completion should succeed
        cp.complete_prefix(1, "a");
        cp.complete_order(1).expect("all prefixes completed");
        assert!(cp.is_order_complete(1));
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
        cp.complete_order(1).expect("no in-progress prefixes for order 1");
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
        // Verify it's no longer in Completed state
        let progress = cp.order_progress.get(&2).unwrap();
        assert_eq!(progress.get_state("aa"), Some(PrefixState::InProgress));
        assert_eq!(progress.count_state(PrefixState::Completed), 0);
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

    // =========================================================================
    // Bitmap Encoding/Decoding Tests (v3 format)
    // =========================================================================

    #[test]
    fn test_prefix_to_index_single_char() {
        // Test all 26 single-char prefixes
        assert_eq!(prefix_to_index("a"), Some(0));
        assert_eq!(prefix_to_index("b"), Some(1));
        assert_eq!(prefix_to_index("m"), Some(12));
        assert_eq!(prefix_to_index("z"), Some(25));

        // Invalid
        assert_eq!(prefix_to_index("A"), None); // uppercase
        assert_eq!(prefix_to_index("1"), None); // digit
        assert_eq!(prefix_to_index(""), None);  // empty
    }

    #[test]
    fn test_prefix_to_index_two_char() {
        // First row: aa-az (0-25)
        assert_eq!(prefix_to_index("aa"), Some(0));
        assert_eq!(prefix_to_index("ab"), Some(1));
        assert_eq!(prefix_to_index("az"), Some(25));

        // Second row: ba-bz (26-51)
        assert_eq!(prefix_to_index("ba"), Some(26));
        assert_eq!(prefix_to_index("bb"), Some(27));

        // Last: zz (675)
        assert_eq!(prefix_to_index("zz"), Some(675));

        // Formula verification: (c1 - 'a') * 26 + (c2 - 'a')
        // "th" = (19 * 26) + 7 = 494 + 7 = 501
        assert_eq!(prefix_to_index("th"), Some(501));

        // Invalid
        assert_eq!(prefix_to_index("AA"), None);
        assert_eq!(prefix_to_index("a1"), None);
        assert_eq!(prefix_to_index("abc"), None); // too long
    }

    #[test]
    fn test_prefix_to_index_exhaustive_single_char() {
        // Verify all 26 single-char prefixes map to unique indices 0-25
        for (i, c) in ('a'..='z').enumerate() {
            let prefix = c.to_string();
            assert_eq!(prefix_to_index(&prefix), Some(i as u16));
        }
    }

    #[test]
    fn test_prefix_to_index_exhaustive_two_char() {
        // Verify all 676 two-char prefixes map to unique indices 0-675
        let mut expected_index = 0u16;
        for c1 in 'a'..='z' {
            for c2 in 'a'..='z' {
                let prefix = format!("{}{}", c1, c2);
                assert_eq!(
                    prefix_to_index(&prefix),
                    Some(expected_index),
                    "prefix '{}' should map to index {}",
                    prefix,
                    expected_index
                );
                expected_index += 1;
            }
        }
        assert_eq!(expected_index, 676);
    }

    #[test]
    fn test_index_to_prefix_single_char() {
        assert_eq!(index_to_prefix(0, 1), "a");
        assert_eq!(index_to_prefix(1, 1), "b");
        assert_eq!(index_to_prefix(12, 1), "m");
        assert_eq!(index_to_prefix(25, 1), "z");
    }

    #[test]
    fn test_index_to_prefix_two_char() {
        assert_eq!(index_to_prefix(0, 2), "aa");
        assert_eq!(index_to_prefix(1, 2), "ab");
        assert_eq!(index_to_prefix(25, 2), "az");
        assert_eq!(index_to_prefix(26, 2), "ba");
        assert_eq!(index_to_prefix(501, 2), "th");
        assert_eq!(index_to_prefix(675, 2), "zz");
    }

    #[test]
    fn test_index_prefix_roundtrip_single_char() {
        for i in 0..26u16 {
            let prefix = index_to_prefix(i, 1);
            assert_eq!(prefix_to_index(&prefix), Some(i));
        }
    }

    #[test]
    fn test_index_prefix_roundtrip_two_char() {
        for i in 0..676u16 {
            let prefix = index_to_prefix(i, 2);
            assert_eq!(prefix_to_index(&prefix), Some(i));
        }
    }

    #[test]
    fn test_pack_states_empty() {
        let states: HashMap<String, PrefixState> = HashMap::new();

        // Single-char: 1 chunk (26 prefixes / 32 per chunk = 1)
        let chunks = pack_states(&states, 1);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], 0);

        // Two-char: 22 chunks (676 prefixes / 32 per chunk = 22)
        let chunks = pack_states(&states, 2);
        assert_eq!(chunks.len(), 22);
        assert!(chunks.iter().all(|&c| c == 0));
    }

    #[test]
    fn test_pack_states_single_prefix() {
        let mut states = HashMap::new();
        states.insert("aa".to_string(), PrefixState::Completed);

        let chunks = pack_states(&states, 2);

        // "aa" = index 0, chunk 0, bit position 0
        // Completed = 0b10
        assert_eq!(chunks[0], 0b10);
    }

    #[test]
    fn test_pack_states_all_state_types() {
        let mut states = HashMap::new();
        states.insert("aa".to_string(), PrefixState::InProgress); // index 0
        states.insert("ab".to_string(), PrefixState::Completed);  // index 1
        states.insert("ac".to_string(), PrefixState::Failed);     // index 2

        let chunks = pack_states(&states, 2);

        // Bit layout: [ac][ab][aa] = [11][10][01] = 0b110100 + 01 = 0b110101
        // Actually: bit positions are index * 2
        // index 0 (aa): bits 0-1 = InProgress (0b01)
        // index 1 (ab): bits 2-3 = Completed (0b10)
        // index 2 (ac): bits 4-5 = Failed (0b11)
        let expected = 0b01 | (0b10 << 2) | (0b11 << 4);
        assert_eq!(chunks[0], expected);
    }

    #[test]
    fn test_pack_unpack_roundtrip_sparse() {
        let mut states = HashMap::new();
        states.insert("aa".to_string(), PrefixState::Completed);
        states.insert("th".to_string(), PrefixState::InProgress);
        states.insert("zz".to_string(), PrefixState::Failed);

        let chunks = pack_states(&states, 2);
        let unpacked = unpack_states(&chunks, 2);

        assert_eq!(unpacked.len(), 3);
        assert_eq!(unpacked.get("aa"), Some(&PrefixState::Completed));
        assert_eq!(unpacked.get("th"), Some(&PrefixState::InProgress));
        assert_eq!(unpacked.get("zz"), Some(&PrefixState::Failed));
    }

    #[test]
    fn test_pack_unpack_roundtrip_full() {
        // Pack all 676 prefixes as Completed
        let mut states = HashMap::new();
        for c1 in 'a'..='z' {
            for c2 in 'a'..='z' {
                states.insert(format!("{}{}", c1, c2), PrefixState::Completed);
            }
        }

        let chunks = pack_states(&states, 2);
        let unpacked = unpack_states(&chunks, 2);

        assert_eq!(unpacked.len(), 676);
        for (prefix, state) in &unpacked {
            assert_eq!(
                *state,
                PrefixState::Completed,
                "prefix '{}' should be Completed",
                prefix
            );
        }
    }

    #[test]
    fn test_pack_unpack_roundtrip_mixed_states() {
        let mut states = HashMap::new();

        // Assign different states based on index
        for i in 0..676u16 {
            let prefix = index_to_prefix(i, 2);
            let state = match i % 3 {
                0 => PrefixState::Completed,
                1 => PrefixState::InProgress,
                2 => PrefixState::Failed,
                _ => unreachable!(),
            };
            states.insert(prefix, state);
        }

        let chunks = pack_states(&states, 2);
        let unpacked = unpack_states(&chunks, 2);

        assert_eq!(unpacked.len(), 676);
        for i in 0..676u16 {
            let prefix = index_to_prefix(i, 2);
            let expected = match i % 3 {
                0 => PrefixState::Completed,
                1 => PrefixState::InProgress,
                2 => PrefixState::Failed,
                _ => unreachable!(),
            };
            assert_eq!(
                unpacked.get(&prefix),
                Some(&expected),
                "prefix '{}' (index {}) state mismatch",
                prefix,
                i
            );
        }
    }

    #[test]
    fn test_unpack_states_not_started_excluded() {
        // Empty chunks = all NotStarted = empty HashMap
        let chunks = vec![0u64; 22];
        let unpacked = unpack_states(&chunks, 2);
        assert!(unpacked.is_empty());
    }

    #[test]
    fn test_bitmap_chunk_boundaries() {
        // Test prefixes at chunk boundaries (indices 31, 32, 63, 64, etc.)
        let mut states = HashMap::new();

        // Indices at chunk boundaries
        let boundary_indices = [0u16, 31, 32, 63, 64, 95, 96, 671, 672, 675];

        for &idx in &boundary_indices {
            let prefix = index_to_prefix(idx, 2);
            states.insert(prefix, PrefixState::Completed);
        }

        let chunks = pack_states(&states, 2);
        let unpacked = unpack_states(&chunks, 2);

        assert_eq!(unpacked.len(), boundary_indices.len());
        for &idx in &boundary_indices {
            let prefix = index_to_prefix(idx, 2);
            assert_eq!(
                unpacked.get(&prefix),
                Some(&PrefixState::Completed),
                "boundary prefix '{}' (index {}) should be present",
                prefix,
                idx
            );
        }
    }

    #[test]
    fn test_prefix_len_for_order() {
        assert_eq!(prefix_len_for_order(1), 1);
        assert_eq!(prefix_len_for_order(2), 2);
        assert_eq!(prefix_len_for_order(3), 2);
        assert_eq!(prefix_len_for_order(4), 2);
        assert_eq!(prefix_len_for_order(5), 2);
    }

    #[test]
    fn test_num_chunks_for_prefix_len() {
        // Single char: 26 prefixes / 32 = 1 chunk
        assert_eq!(num_chunks_for_prefix_len(1), 1);

        // Two char: 676 prefixes / 32 = 21.125 = 22 chunks
        assert_eq!(num_chunks_for_prefix_len(2), 22);
    }

    #[test]
    fn test_bitmap_storage_size() {
        // Verify the claimed storage reduction
        // v2: 676 keys * ~33 bytes = ~22KB per order
        // v3: 22 chunks * 8 bytes = 176 bytes per order

        let num_chunks = num_chunks_for_prefix_len(2);
        let v3_bytes = num_chunks * 8;

        assert_eq!(num_chunks, 22);
        assert_eq!(v3_bytes, 176);
    }
}
