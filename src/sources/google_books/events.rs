//! Domain events for Google Books import progress.
//!
//! These events are TUI-agnostic and describe what happened in domain terms.
//! Any UI (TUI, progress bar, logging) can subscribe and render appropriately.
//!
//! ## Architecture
//!
//! The import logic emits domain events via a broadcast channel. Multiple
//! subscribers can consume these events:
//! - TUI: Renders progress dashboard with ratatui
//! - Logging: Writes to file or console
//! - Metrics: Collects statistics for later analysis
//!
//! Commands flow in the opposite direction, from UI to the importer:
//! - Pause/Resume: Gracefully pause workers
//! - Cancel: Stop import with checkpoint
//! - SetParallelism: Adjust worker count at runtime

use std::time::Duration;

/// Domain events emitted during import.
///
/// These events describe what happened without any UI-specific details.
/// The TUI (or any other subscriber) maps these to its own state.
#[derive(Clone, Debug)]
pub enum ImportEvent {
    /// Import started for a specific n-gram order.
    OrderStarted {
        order: u8,
        total_files: u64,
    },

    /// An order completed successfully.
    OrderCompleted {
        order: u8,
        ngram_count: u64,
        duration: Duration,
    },

    /// Worker began downloading a prefix file.
    WorkerStarted {
        worker_id: usize,
        /// N-gram order being processed (1-5).
        order: u8,
        prefix: String,
    },

    /// Worker download progress (bytes received).
    WorkerProgress {
        worker_id: usize,
        bytes_downloaded: u64,
        total_bytes: Option<u64>,
    },

    /// Worker n-gram processing progress (periodic update).
    WorkerNgramProgress {
        worker_id: usize,
        ngram_count: u64,
    },

    /// Worker finished processing a file.
    WorkerFinished {
        worker_id: usize,
        /// N-gram order that was processed (1-5).
        order: u8,
        prefix: String,
        ngram_count: u64,
        duration: Duration,
    },

    /// Per-order progress update for TUI multi-order display.
    ///
    /// Emitted periodically to update the TUI with per-order progress,
    /// enabling display of multiple concurrent orders.
    OrderProgress {
        /// N-gram order (1-5).
        order: u8,
        /// Files completed for this order.
        files_completed: u64,
        /// Total files for this order.
        total_files: u64,
        /// N-grams processed for this order.
        ngrams_processed: u64,
        /// Whether this order is fully complete.
        is_complete: bool,
    },

    /// Worker is retrying after transient error.
    WorkerRetrying {
        worker_id: usize,
        prefix: String,
        attempt: u32,
        max_attempts: u32,
        error: String,
    },

    /// Worker exited (shutdown signal received or queue empty).
    ///
    /// Emitted when a worker task exits, either because it received a shutdown
    /// signal (parallelism decreased) or because the job queue is empty.
    WorkerExited {
        worker_id: usize,
    },

    /// Periodic statistics update.
    StatsSnapshot {
        files_completed: u64,
        total_files: u64,
        total_ngrams: u64,
        unique_ngrams: u64,
        ngrams_per_second: f64,
        elapsed: Duration,
    },

    /// Checkpoint saved.
    CheckpointSaved {
        prefix: String,
    },

    /// Import completed (all orders).
    ImportCompleted {
        total_ngrams: u64,
        duration: Duration,
    },

    /// Import cancelled by user.
    ImportCancelled,

    /// Import paused.
    ImportPaused,

    /// Import resumed.
    ImportResumed,

    /// Fatal error occurred during import.
    ///
    /// This event is sent when an unrecoverable error occurs (e.g., network
    /// failure after retries exhausted, disk I/O error). The TUI should
    /// display this prominently and allow the user to quit gracefully.
    Error {
        /// Human-readable error message.
        message: String,
    },

    /// Log message (for debugging/info).
    Log {
        level: LogLevel,
        message: String,
    },

    /// A prefix file failed after exhausting all retries.
    ///
    /// This event indicates that a prefix could not be processed and will
    /// be skipped for the current run. The prefix is marked as failed in
    /// the checkpoint and will be retried on subsequent runs.
    PrefixFailed {
        /// N-gram order (1-5).
        order: u8,
        /// The prefix that failed.
        prefix: String,
        /// Human-readable error message.
        error: String,
        /// Number of retry attempts made.
        attempts: u32,
    },

    /// Previously failed prefixes are being retried.
    ///
    /// Emitted at the start of an order when there are failed prefixes
    /// from a previous run that will be retried.
    RetryingFailedPrefixes {
        /// N-gram order (1-5).
        order: u8,
        /// Number of failed prefixes being retried.
        count: usize,
        /// List of prefix strings being retried.
        prefixes: Vec<String>,
    },

    /// In-progress prefixes detected on resume (partial data).
    ///
    /// Emitted when resuming and detecting prefixes that were being
    /// processed when the previous run crashed. These prefixes have
    /// potentially partial data that will be cleared before retrying.
    RecoveringInProgressPrefixes {
        /// N-gram order (1-5).
        order: u8,
        /// Number of in-progress prefixes being recovered.
        count: usize,
        /// List of prefix strings with partial data.
        prefixes: Vec<String>,
    },
}

/// Log level for log events.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

/// Commands sent to control the import.
///
/// Commands flow from the UI (or any controller) to the importer.
/// The importer processes these asynchronously and emits corresponding events.
#[derive(Clone, Debug)]
pub enum ImportCommand {
    /// Pause all workers (graceful, waits for current n-gram).
    Pause,

    /// Resume paused workers.
    Resume,

    /// Cancel import (save checkpoint first).
    Cancel,

    /// Force quit without saving checkpoint.
    ForceQuit,

    /// Adjust parallelism at runtime.
    SetParallelism(usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_is_clone() {
        let event = ImportEvent::WorkerStarted {
            worker_id: 0,
            order: 2,
            prefix: "th".to_string(),
        };
        let _cloned = event.clone();
    }

    #[test]
    fn command_is_clone() {
        let cmd = ImportCommand::Pause;
        let _cloned = cmd.clone();
    }

    #[test]
    fn order_progress_event() {
        let event = ImportEvent::OrderProgress {
            order: 2,
            files_completed: 50,
            total_files: 676,
            ngrams_processed: 1_000_000,
            is_complete: false,
        };
        let _cloned = event.clone();
    }
}
