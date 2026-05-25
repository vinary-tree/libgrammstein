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
    OrderStarted { order: u8, total_files: u64 },

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
    WorkerNgramProgress { worker_id: usize, ngram_count: u64 },

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
        /// Files completed for this order (success + skipped).
        files_completed: u64,
        /// Total files for this order.
        total_files: u64,
        /// N-grams processed for this order.
        ngrams_processed: u64,
        /// Whether this order is fully complete.
        is_complete: bool,
        /// Files successfully completed (for green progress bar segment).
        files_succeeded: u64,
        /// Files skipped/failed (for yellow progress bar segment, will retry next session).
        files_skipped: u64,
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
    WorkerExited { worker_id: usize },

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
    CheckpointSaved { prefix: String },

    /// Checkpoint progress update.
    ///
    /// Emitted during async checkpoint finish phase to track progress
    /// of persisting individual shards.
    CheckpointProgress {
        /// Number of shards that have been checkpointed.
        shards_processed: usize,
        /// Total number of shards to checkpoint.
        total_shards: usize,
        /// Completion percentage (0.0 to 100.0).
        percent_complete: f32,
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
    Log { level: LogLevel, message: String },

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

    /// Job requeued with exponential backoff (current session retry).
    ///
    /// Emitted when a job encounters a retryable error and is requeued
    /// with a delay (exponential backoff). These jobs will be retried
    /// within the current session after the backoff delay expires.
    DeferredRetry {
        /// The prefix being retried.
        prefix: String,
        /// Current attempt number (1-indexed).
        attempt: u32,
        /// N-gram order (1-5).
        order: u8,
    },

    /// Deferred retry job started processing (decrement backoff queue).
    ///
    /// Emitted when a job that was previously deferred (requeued with backoff)
    /// starts processing again. This allows the TUI to decrement the backoff
    /// queue counter.
    DeferredRetryStarted {
        /// The prefix starting its retry.
        prefix: String,
        /// N-gram order (1-5).
        order: u8,
    },

    /// Merge phase started (sharded storage only).
    ///
    /// Emitted when the shard merge phase begins after n-gram collection completes.
    MergeStarted {
        /// Number of shards to merge.
        shard_count: usize,
        /// Estimated total n-grams across all shards.
        estimated_ngrams: u64,
    },

    /// Merge progress update.
    ///
    /// Emitted periodically during the merge phase to update progress display.
    MergeProgress {
        /// Number of shards processed so far.
        shards_processed: usize,
        /// Total number of shards.
        total_shards: usize,
        /// N-grams merged so far.
        ngrams_merged: u64,
        /// Completion percentage (0.0 to 100.0).
        percent_complete: f32,
    },

    /// Merge completed successfully.
    ///
    /// Emitted when all shards have been merged into the final output.
    MergeCompleted {
        /// Total n-grams in merged output.
        total_ngrams: u64,
        /// Bytes written to output file.
        bytes_written: u64,
        /// Duration of merge phase.
        duration: Duration,
    },

    /// Merge failed with error.
    ///
    /// Emitted when merge encounters an unrecoverable error.
    MergeFailed {
        /// Human-readable error message.
        error: String,
    },

    /// Shard cleanup started.
    ///
    /// Emitted when temporary shard files are being deleted after merge.
    ShardCleanupStarted {
        /// Number of shard files to delete.
        shard_count: usize,
    },

    /// Shard cleanup completed.
    ///
    /// Emitted when all temporary shard files have been deleted.
    ShardCleanupCompleted {
        /// Number of shard files deleted.
        shards_deleted: usize,
        /// Total bytes freed by deletion.
        bytes_freed: u64,
    },

    /// All work completed - triggers completion dialog.
    ///
    /// Emitted after import, merge (if applicable), and cleanup are all done.
    /// The TUI should display a completion dialog requiring user acknowledgment.
    AllWorkCompleted {
        /// Total n-grams in final output.
        total_ngrams: u64,
        /// Total duration of entire import process.
        total_duration: Duration,
        /// Whether shard files were kept (--keep-shards flag).
        shards_kept: bool,
    },

    /// MKN statistics computation started.
    ///
    /// Emitted when Modified Kneser-Ney statistics computation begins.
    MknStarted {
        /// Source of n-gram data for MKN computation.
        /// Either "shards" (parallel computation) or "single_trie" (sequential).
        source: String,
        /// Estimated total n-grams to process.
        estimated_ngrams: u64,
    },

    /// MKN computation progress update.
    ///
    /// Emitted periodically during MKN statistics computation.
    MknProgress {
        /// Current phase (1 = collecting pairs, 2 = writing stats).
        phase: u8,
        /// Total phases.
        total_phases: u8,
        /// Items processed in current phase.
        items_processed: u64,
        /// Estimated total items in current phase.
        total_items: u64,
        /// Completion percentage (0.0 to 100.0).
        percent_complete: f32,
    },

    /// MKN computation completed successfully.
    ///
    /// Emitted when MKN statistics have been computed and written.
    MknCompleted {
        /// Number of continuation count entries written.
        continuation_entries: u64,
        /// Number of frequency count entries written.
        frequency_entries: u64,
        /// Duration of MKN computation.
        duration: Duration,
    },

    /// MKN computation failed.
    ///
    /// Emitted when MKN computation encounters an unrecoverable error.
    MknFailed {
        /// Human-readable error message.
        error: String,
    },

    /// Import phase changed.
    ///
    /// Emitted when the import pipeline transitions to a new phase.
    /// Enables TUI to display current phase and track progress.
    PhaseChanged {
        /// Name of the new phase (e.g., "Downloading", "Computing MKN Statistics").
        phase: String,
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
            files_succeeded: 48,
            files_skipped: 2,
        };
        let _cloned = event.clone();
    }
}
