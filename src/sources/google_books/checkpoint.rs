//! Checkpoint and resume support for Google Books import.
//!
//! Long-running imports (hours for full datasets) need checkpoint support
//! to handle interruptions gracefully without losing progress.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Import checkpoint for resume support.
///
/// Checkpoints are saved after each prefix file completes and on
/// graceful shutdown (SIGINT/SIGTERM).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImportCheckpoint {
    /// Version of checkpoint format (for future compatibility).
    pub version: u32,

    /// Last fully processed n-gram orders.
    pub completed_orders: Vec<u8>,

    /// Current order being processed.
    pub current_order: u8,

    /// Completed prefix files for current order.
    ///
    /// For 1-grams: ["a", "b", "c", ...]
    /// For 2-5 grams: ["aa", "ab", "ac", ...]
    pub completed_prefixes: Vec<String>,

    /// Current prefix file being processed (if any).
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
    pub const CURRENT_VERSION: u32 = 1;

    /// Create a new empty checkpoint.
    pub fn new() -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            completed_orders: Vec::new(),
            current_order: 1,
            completed_prefixes: Vec::new(),
            current_prefix: None,
            byte_offset: 0,
            mkn_phase: MknPhase::NotStarted,
            stats: CheckpointStats::default(),
            timestamp: Utc::now(),
        }
    }

    /// Load checkpoint from file.
    pub fn load(path: &Path) -> Result<Self, CheckpointError> {
        let file = File::open(path).map_err(CheckpointError::Io)?;
        let reader = BufReader::new(file);
        let checkpoint: Self = serde_json::from_reader(reader).map_err(CheckpointError::Json)?;

        // Version check
        if checkpoint.version > Self::CURRENT_VERSION {
            return Err(CheckpointError::UnsupportedVersion {
                found: checkpoint.version,
                max: Self::CURRENT_VERSION,
            });
        }

        Ok(checkpoint)
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

    /// Mark a prefix as completed.
    pub fn complete_prefix(&mut self, prefix: &str) {
        self.completed_prefixes.push(prefix.to_string());
        self.current_prefix = None;
        self.byte_offset = 0;
        self.stats.files_processed += 1;
    }

    /// Mark an order as completed.
    pub fn complete_order(&mut self, order: u8) {
        if !self.completed_orders.contains(&order) {
            self.completed_orders.push(order);
        }
        self.completed_prefixes.clear();
        self.current_prefix = None;
        self.byte_offset = 0;
    }

    /// Check if a specific prefix needs processing.
    pub fn needs_prefix(&self, order: u8, prefix: &str) -> bool {
        if self.completed_orders.contains(&order) {
            return false;
        }
        if self.current_order != order {
            return true;
        }
        !self.completed_prefixes.contains(&prefix.to_string())
    }

    /// Get resume point within current prefix file.
    pub fn resume_offset(&self, order: u8, prefix: &str) -> Option<u64> {
        if self.current_order == order && self.current_prefix.as_deref() == Some(prefix) {
            Some(self.byte_offset)
        } else {
            None
        }
    }

    /// Update byte offset for current prefix.
    pub fn update_offset(&mut self, prefix: &str, offset: u64) {
        self.current_prefix = Some(prefix.to_string());
        self.byte_offset = offset;
    }

    /// Get human-readable progress summary.
    pub fn progress_summary(&self) -> String {
        format!(
            "Orders: {:?}, Current: {}, Prefixes: {}, N-grams: {}, Files: {}",
            self.completed_orders,
            self.current_order,
            self.completed_prefixes.len(),
            self.stats.ngrams_processed,
            self.stats.files_processed,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_checkpoint_new() {
        let cp = ImportCheckpoint::new();
        assert_eq!(cp.version, ImportCheckpoint::CURRENT_VERSION);
        assert!(cp.completed_orders.is_empty());
        assert_eq!(cp.current_order, 1);
    }

    #[test]
    fn test_checkpoint_save_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.checkpoint.json");

        let mut cp = ImportCheckpoint::new();
        cp.completed_orders.push(1);
        cp.current_order = 2;
        cp.completed_prefixes.push("aa".to_string());
        cp.stats.ngrams_processed = 12345;

        cp.save(&path).unwrap();
        assert!(path.exists());

        let loaded = ImportCheckpoint::load(&path).unwrap();
        assert_eq!(loaded.completed_orders, vec![1]);
        assert_eq!(loaded.current_order, 2);
        assert_eq!(loaded.completed_prefixes, vec!["aa".to_string()]);
        assert_eq!(loaded.stats.ngrams_processed, 12345);
    }

    #[test]
    fn test_needs_prefix() {
        let mut cp = ImportCheckpoint::new();
        cp.current_order = 2;
        cp.completed_prefixes.push("aa".to_string());
        cp.completed_prefixes.push("ab".to_string());

        // Different order, always needs
        assert!(cp.needs_prefix(1, "a"));
        assert!(cp.needs_prefix(3, "aaa"));

        // Same order, completed prefix
        assert!(!cp.needs_prefix(2, "aa"));
        assert!(!cp.needs_prefix(2, "ab"));

        // Same order, not completed
        assert!(cp.needs_prefix(2, "ac"));
    }

    #[test]
    fn test_complete_prefix() {
        let mut cp = ImportCheckpoint::new();
        cp.current_prefix = Some("aa".to_string());
        cp.byte_offset = 12345;

        cp.complete_prefix("aa");

        assert_eq!(cp.completed_prefixes, vec!["aa".to_string()]);
        assert!(cp.current_prefix.is_none());
        assert_eq!(cp.byte_offset, 0);
        assert_eq!(cp.stats.files_processed, 1);
    }

    #[test]
    fn test_complete_order() {
        let mut cp = ImportCheckpoint::new();
        cp.current_order = 1;
        cp.completed_prefixes.push("a".to_string());
        cp.completed_prefixes.push("b".to_string());

        cp.complete_order(1);

        assert_eq!(cp.completed_orders, vec![1]);
        assert!(cp.completed_prefixes.is_empty());
    }
}
