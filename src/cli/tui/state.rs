//! TUI state derived from domain events.
//!
//! The TUI state is completely decoupled from the import logic. It receives
//! `ImportEvent`s and maps them to UI-specific state that the renderer uses.

use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use crate::sources::google_books::{ImportEvent, LogLevel};

/// Maximum number of log entries to keep in the scrollback buffer.
const MAX_LOG_ENTRIES: usize = 1000;

/// Maximum number of throughput samples for the sparkline.
const MAX_THROUGHPUT_SAMPLES: usize = 60;

/// TUI state derived from ImportEvents.
///
/// This struct holds all the state needed to render the TUI. It is updated
/// by calling `apply_event` with incoming domain events.
#[derive(Debug)]
pub struct TuiState {
    /// Language being imported.
    pub language: String,

    /// N-gram orders being imported (e.g., 1..=5).
    pub min_order: u8,
    pub max_order: u8,

    /// Current n-gram order being processed.
    pub current_order: u8,

    /// Files completed for current order.
    pub files_completed: u64,

    /// Total files for current order.
    pub total_files: u64,

    /// Total n-grams processed across all orders.
    pub total_ngrams: u64,

    /// Unique n-grams (after deduplication).
    pub unique_ngrams: u64,

    /// Current processing rate (n-grams per second).
    pub ngrams_per_second: f64,

    /// Elapsed time since import started.
    pub elapsed: Duration,

    /// Worker states (one per parallel download slot).
    pub workers: Vec<WorkerState>,

    /// Log entries (most recent first).
    pub log_entries: VecDeque<LogEntry>,

    /// Throughput history for sparkline visualization.
    pub throughput_history: VecDeque<f64>,

    /// Whether import is currently paused.
    pub paused: bool,

    /// Whether import has completed.
    pub completed: bool,

    /// Whether import was cancelled.
    pub cancelled: bool,

    /// Log scroll offset (0 = most recent at bottom).
    pub log_scroll: usize,

    /// Number of parallel download workers.
    pub parallel_downloads: usize,

    /// Fatal error message (if any).
    ///
    /// When set, indicates that a fatal error occurred and the import
    /// cannot continue. The TUI should display this prominently.
    pub error_message: Option<String>,

    /// Per-order progress tracking for overlapping order processing.
    pub order_progress: HashMap<u8, OrderProgressState>,

    /// Total count of failed prefixes across all orders.
    ///
    /// Displayed as a warning in the TUI when non-zero.
    pub failed_prefixes_count: usize,

    /// Count of prefixes being retried from previous failures.
    pub retrying_prefixes_count: usize,

    /// Count of in-progress prefixes being recovered (partial data).
    pub recovering_prefixes_count: usize,

    /// Count of jobs waiting in the backoff queue (deferred retries in current session).
    ///
    /// These are jobs that encountered retryable errors and are waiting to be
    /// retried after their exponential backoff delay expires.
    pub backoff_queue_count: usize,

    /// Whether merge phase is currently in progress.
    pub merge_in_progress: bool,

    /// Current merge progress (if merge is in progress).
    pub merge_progress: Option<MergeProgressState>,

    /// Whether MKN statistics computation is in progress.
    pub mkn_in_progress: bool,

    /// Current MKN progress (if MKN is in progress).
    pub mkn_progress: Option<MknProgressState>,

    /// Whether to show the completion dialog.
    pub show_completion_dialog: bool,

    /// Statistics to display in completion dialog.
    pub completion_stats: Option<CompletionStats>,

    /// Current import phase (e.g., "Importing", "Computing MKN Statistics", "Merging Shards").
    ///
    /// This provides user feedback during long-running phases after file downloads complete.
    pub current_phase: String,
}

/// Progress state for a single n-gram order.
#[derive(Debug, Clone, Default)]
pub struct OrderProgressState {
    /// Files completed for this order (success + skipped).
    pub files_completed: u64,

    /// Total files for this order.
    pub total_files: u64,

    /// N-grams processed for this order.
    pub ngrams_processed: u64,

    /// Whether this order is fully complete.
    pub is_complete: bool,

    /// Files successfully completed (for green progress bar segment).
    pub files_succeeded: u64,

    /// Files skipped/failed (for yellow progress bar segment).
    pub files_skipped: u64,
}

/// Progress state for merge phase.
#[derive(Debug, Clone, Default)]
pub struct MergeProgressState {
    /// Number of shards processed so far.
    pub shards_processed: usize,

    /// Total number of shards.
    pub total_shards: usize,

    /// N-grams merged so far.
    pub ngrams_merged: u64,

    /// Completion percentage (0.0 to 100.0).
    pub percent_complete: f32,
}

/// Progress state for MKN statistics computation.
#[derive(Debug, Clone, Default)]
pub struct MknProgressState {
    /// Current phase (1 = collecting pairs, 2 = writing stats).
    pub phase: u8,

    /// Total phases.
    pub total_phases: u8,

    /// Items processed in current phase.
    pub items_processed: u64,

    /// Estimated total items in current phase.
    pub total_items: u64,

    /// Completion percentage (0.0 to 100.0).
    pub percent_complete: f32,
}

/// Statistics displayed in completion dialog.
#[derive(Debug, Clone)]
pub struct CompletionStats {
    /// Total n-grams in final output.
    pub total_ngrams: u64,

    /// Total duration of entire import process.
    pub total_duration: Duration,

    /// Whether shard files were kept.
    pub shards_kept: bool,

    /// Files processed during import.
    pub files_processed: u64,

    /// Whether merge was performed.
    pub merge_performed: bool,
}

/// State of a single download worker.
#[derive(Debug, Clone)]
pub struct WorkerState {
    /// Worker ID (0-indexed).
    pub id: usize,

    /// Current status.
    pub status: WorkerStatus,

    /// Current prefix being processed (if any).
    pub prefix: Option<String>,

    /// Current n-gram order being processed (if any).
    pub order: Option<u8>,

    /// Download progress (0.0 to 1.0).
    pub progress: f32,

    /// Bytes downloaded for current file.
    pub bytes_downloaded: u64,

    /// Total bytes for current file (if known).
    pub total_bytes: Option<u64>,

    /// N-grams processed in current file.
    pub ngram_count: u64,
}

/// Worker status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerStatus {
    /// Worker is idle, waiting for next task.
    Idle,

    /// Worker is downloading a prefix file.
    Downloading,

    /// Worker has completed current file.
    Completed,

    /// Worker is retrying after an error.
    Retrying { attempt: u32, max_attempts: u32 },
}

/// A log entry for the TUI log panel.
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// Timestamp (formatted as HH:MM:SS).
    pub timestamp: String,

    /// Log level.
    pub level: LogLevel,

    /// Log message.
    pub message: String,
}

impl TuiState {
    /// Create a new TUI state.
    pub fn new(language: &str, min_order: u8, max_order: u8, parallel_downloads: usize) -> Self {
        log::debug!(
            "[TUI-STATE] Creating TuiState: language={}, orders={}-{}, workers={}",
            language,
            min_order,
            max_order,
            parallel_downloads
        );

        let workers = (0..parallel_downloads)
            .map(|id| WorkerState {
                id,
                status: WorkerStatus::Idle,
                prefix: None,
                order: None,
                progress: 0.0,
                bytes_downloaded: 0,
                total_bytes: None,
                ngram_count: 0,
            })
            .collect();

        let state = Self {
            language: language.to_string(),
            min_order,
            max_order,
            current_order: min_order,
            files_completed: 0,
            total_files: 0,
            total_ngrams: 0,
            unique_ngrams: 0,
            ngrams_per_second: 0.0,
            elapsed: Duration::ZERO,
            workers,
            log_entries: VecDeque::with_capacity(MAX_LOG_ENTRIES),
            throughput_history: VecDeque::with_capacity(MAX_THROUGHPUT_SAMPLES),
            paused: false,
            completed: false,
            cancelled: false,
            log_scroll: 0,
            parallel_downloads,
            error_message: None,
            order_progress: HashMap::new(),
            failed_prefixes_count: 0,
            retrying_prefixes_count: 0,
            recovering_prefixes_count: 0,
            backoff_queue_count: 0,
            merge_in_progress: false,
            merge_progress: None,
            mkn_in_progress: false,
            mkn_progress: None,
            show_completion_dialog: false,
            completion_stats: None,
            current_phase: "Initializing".to_string(),
        };

        log::debug!(
            "[TUI-STATE] Initial state: current_phase='{}', mkn_in_progress={}, merge_in_progress={}, completed={}",
            state.current_phase,
            state.mkn_in_progress,
            state.merge_in_progress,
            state.completed
        );

        state
    }

    /// Apply a domain event to update the state.
    pub fn apply_event(&mut self, event: &ImportEvent) {
        match event {
            ImportEvent::OrderStarted { order, total_files } => {
                self.current_order = *order;
                self.total_files = *total_files;
                self.files_completed = 0;

                // Initialize order_progress entry so display shows ACTIVE immediately
                let progress = self.order_progress.entry(*order).or_default();
                progress.total_files = *total_files;

                self.add_log(
                    LogLevel::Info,
                    format!("Started order {} ({} files)", order, total_files),
                );
            }

            ImportEvent::OrderCompleted {
                order,
                ngram_count,
                duration,
            } => {
                self.add_log(
                    LogLevel::Info,
                    format!(
                        "Completed order {}: {} n-grams in {:.1}s",
                        order,
                        format_count(*ngram_count),
                        duration.as_secs_f64()
                    ),
                );
            }

            ImportEvent::WorkerStarted { worker_id, order, prefix } => {
                if let Some(worker) = self.workers.get_mut(*worker_id) {
                    worker.status = WorkerStatus::Downloading;
                    worker.prefix = Some(prefix.clone());
                    worker.order = Some(*order);
                    worker.progress = 0.0;
                    worker.bytes_downloaded = 0;
                    worker.total_bytes = None;
                    worker.ngram_count = 0;
                }
            }

            ImportEvent::WorkerProgress {
                worker_id,
                bytes_downloaded,
                total_bytes,
            } => {
                if let Some(worker) = self.workers.get_mut(*worker_id) {
                    worker.bytes_downloaded = *bytes_downloaded;
                    worker.total_bytes = *total_bytes;
                    if let Some(total) = total_bytes {
                        if *total > 0 {
                            worker.progress = *bytes_downloaded as f32 / *total as f32;
                        }
                    }
                }
            }

            ImportEvent::WorkerNgramProgress {
                worker_id,
                ngram_count,
            } => {
                if let Some(worker) = self.workers.get_mut(*worker_id) {
                    worker.ngram_count = *ngram_count;
                }
            }

            ImportEvent::WorkerFinished {
                worker_id,
                order,
                prefix,
                ngram_count,
                duration,
            } => {
                if let Some(worker) = self.workers.get_mut(*worker_id) {
                    worker.status = WorkerStatus::Completed;
                    worker.progress = 1.0;
                    worker.ngram_count = *ngram_count;
                }
                // Note: files_completed is updated by StatsSnapshot events, not here.
                // Incrementing here would cause race conditions where counts exceed
                // total_files when events arrive out of order.
                self.add_log(
                    LogLevel::Debug,
                    format!(
                        "Completed: {} (order {}, {} n-grams in {:.1}s)",
                        prefix,
                        order,
                        format_count(*ngram_count),
                        duration.as_secs_f64()
                    ),
                );
            }

            ImportEvent::WorkerRetrying {
                worker_id,
                prefix,
                attempt,
                max_attempts,
                error,
            } => {
                if let Some(worker) = self.workers.get_mut(*worker_id) {
                    worker.status = WorkerStatus::Retrying {
                        attempt: *attempt,
                        max_attempts: *max_attempts,
                    };
                }
                self.add_log(
                    LogLevel::Warn,
                    format!(
                        "Retry {}/{}: {} - {}",
                        attempt, max_attempts, prefix, error
                    ),
                );
            }

            ImportEvent::StatsSnapshot {
                files_completed,
                total_files,
                total_ngrams,
                unique_ngrams,
                ngrams_per_second,
                elapsed,
            } => {
                self.files_completed = *files_completed;
                self.total_files = *total_files;
                self.total_ngrams = *total_ngrams;
                self.unique_ngrams = *unique_ngrams;
                self.ngrams_per_second = *ngrams_per_second;
                self.elapsed = *elapsed;

                // Update throughput history for sparkline
                self.throughput_history.push_back(*ngrams_per_second);
                if self.throughput_history.len() > MAX_THROUGHPUT_SAMPLES {
                    self.throughput_history.pop_front();
                }
            }

            ImportEvent::CheckpointSaved { prefix } => {
                self.add_log(LogLevel::Debug, format!("Checkpoint saved: {}", prefix));
            }

            ImportEvent::ImportCompleted {
                total_ngrams,
                duration,
            } => {
                self.completed = true;
                self.total_ngrams = *total_ngrams;
                self.elapsed = *duration;
                self.add_log(
                    LogLevel::Info,
                    format!(
                        "Import completed: {} n-grams in {:.1}s",
                        format_count(*total_ngrams),
                        duration.as_secs_f64()
                    ),
                );
            }

            ImportEvent::ImportCancelled => {
                self.cancelled = true;
                self.add_log(LogLevel::Warn, "Import cancelled by user".to_string());
            }

            ImportEvent::ImportPaused => {
                self.paused = true;
                self.add_log(LogLevel::Info, "Import paused".to_string());
            }

            ImportEvent::ImportResumed => {
                self.paused = false;
                self.add_log(LogLevel::Info, "Import resumed".to_string());
            }

            ImportEvent::Error { message } => {
                log::error!("Import error: {}", message);
                self.error_message = Some(message.clone());
                self.add_log(LogLevel::Error, format!("FATAL: {}", message));
            }

            ImportEvent::Log { level, message } => {
                self.add_log(*level, message.clone());
            }

            ImportEvent::WorkerExited { worker_id } => {
                log::debug!(
                    "[TUI-STATE] WorkerExited: worker_id={}, workers_remaining={}",
                    worker_id,
                    self.workers.len().saturating_sub(1)
                );
                // Remove the worker from the display list
                self.workers.retain(|w| w.id != *worker_id);
                self.add_log(LogLevel::Info, format!("Worker {} exited", worker_id));
            }

            ImportEvent::OrderProgress {
                order,
                files_completed,
                total_files,
                ngrams_processed,
                is_complete,
                files_succeeded,
                files_skipped,
            } => {
                let progress = self.order_progress.entry(*order).or_default();
                progress.files_completed = *files_completed;
                progress.total_files = *total_files;
                progress.ngrams_processed = *ngrams_processed;
                progress.is_complete = *is_complete;
                progress.files_succeeded = *files_succeeded;
                progress.files_skipped = *files_skipped;
            }

            ImportEvent::PrefixFailed {
                order,
                prefix,
                error,
                attempts,
            } => {
                self.failed_prefixes_count += 1;
                self.add_log(
                    LogLevel::Error,
                    format!(
                        "FAILED: {} (order {}) after {} attempts - {}",
                        prefix, order, attempts, error
                    ),
                );
            }

            ImportEvent::RetryingFailedPrefixes {
                order,
                count,
                prefixes,
            } => {
                self.retrying_prefixes_count += *count;
                self.add_log(
                    LogLevel::Warn,
                    format!(
                        "Retrying {} failed prefixes from previous run (order {}): {:?}",
                        count, order, prefixes
                    ),
                );
            }

            ImportEvent::RecoveringInProgressPrefixes {
                order,
                count,
                prefixes,
            } => {
                self.recovering_prefixes_count += *count;
                self.add_log(
                    LogLevel::Warn,
                    format!(
                        "Recovering {} in-progress prefixes with partial data (order {}): {:?}",
                        count, order, prefixes
                    ),
                );
            }

            ImportEvent::DeferredRetry {
                prefix,
                attempt,
                order,
            } => {
                self.backoff_queue_count += 1;
                self.add_log(
                    LogLevel::Debug,
                    format!(
                        "Deferred retry queued: {} (order {}, attempt {})",
                        prefix, order, attempt
                    ),
                );
            }

            ImportEvent::DeferredRetryStarted { prefix, order } => {
                self.backoff_queue_count = self.backoff_queue_count.saturating_sub(1);
                self.add_log(
                    LogLevel::Debug,
                    format!("Deferred retry started: {} (order {})", prefix, order),
                );
            }

            ImportEvent::MergeStarted {
                shard_count,
                estimated_ngrams,
            } => {
                log::debug!(
                    "[TUI-STATE] MergeStarted: merge_in_progress {} -> true, shards={}",
                    self.merge_in_progress,
                    shard_count
                );
                self.merge_in_progress = true;
                self.merge_progress = Some(MergeProgressState {
                    shards_processed: 0,
                    total_shards: *shard_count,
                    ngrams_merged: 0,
                    percent_complete: 0.0,
                });
                self.add_log(
                    LogLevel::Info,
                    format!(
                        "Merging {} shards (~{} n-grams)...",
                        shard_count, estimated_ngrams
                    ),
                );
            }

            ImportEvent::MergeProgress {
                shards_processed,
                total_shards,
                ngrams_merged,
                percent_complete,
            } => {
                self.merge_progress = Some(MergeProgressState {
                    shards_processed: *shards_processed,
                    total_shards: *total_shards,
                    ngrams_merged: *ngrams_merged,
                    percent_complete: *percent_complete,
                });
            }

            ImportEvent::MergeCompleted {
                total_ngrams,
                bytes_written,
                duration,
            } => {
                log::debug!(
                    "[TUI-STATE] MergeCompleted: merge_in_progress {} -> false",
                    self.merge_in_progress
                );
                self.merge_in_progress = false;
                self.merge_progress = None;
                self.add_log(
                    LogLevel::Info,
                    format!(
                        "Merge completed: {} n-grams ({} bytes) in {:.1}s",
                        total_ngrams,
                        bytes_written,
                        duration.as_secs_f64()
                    ),
                );
            }

            ImportEvent::MergeFailed { error } => {
                self.merge_in_progress = false;
                self.merge_progress = None;
                self.error_message = Some(format!("Merge failed: {}", error));
                self.add_log(LogLevel::Error, format!("MERGE FAILED: {}", error));
            }

            ImportEvent::ShardCleanupStarted { shard_count } => {
                self.add_log(
                    LogLevel::Info,
                    format!("Cleaning up {} shard files...", shard_count),
                );
            }

            ImportEvent::ShardCleanupCompleted {
                shards_deleted,
                bytes_freed,
            } => {
                self.add_log(
                    LogLevel::Info,
                    format!(
                        "Cleanup complete: deleted {} shards, freed {} bytes",
                        shards_deleted, bytes_freed
                    ),
                );
            }

            ImportEvent::AllWorkCompleted {
                total_ngrams,
                total_duration,
                shards_kept,
            } => {
                log::debug!(
                    "[TUI-STATE] AllWorkCompleted: completed {} -> true, show_completion_dialog {} -> true",
                    self.completed,
                    self.show_completion_dialog
                );
                self.completed = true;
                self.show_completion_dialog = true;
                self.completion_stats = Some(CompletionStats {
                    total_ngrams: *total_ngrams,
                    total_duration: *total_duration,
                    shards_kept: *shards_kept,
                    files_processed: self.files_completed,
                    merge_performed: self.merge_progress.is_some() || self.merge_in_progress,
                });
                self.add_log(
                    LogLevel::Info,
                    format!(
                        "All work completed: {} n-grams in {:.1}s",
                        total_ngrams,
                        total_duration.as_secs_f64()
                    ),
                );
            }

            ImportEvent::MknStarted {
                source,
                estimated_ngrams,
            } => {
                log::debug!(
                    "[TUI-STATE] MknStarted: mkn_in_progress {} -> true",
                    self.mkn_in_progress
                );
                self.mkn_in_progress = true;
                self.mkn_progress = Some(MknProgressState {
                    phase: 1,
                    total_phases: 2,
                    items_processed: 0,
                    total_items: *estimated_ngrams,
                    percent_complete: 0.0,
                });
                self.add_log(
                    LogLevel::Info,
                    format!(
                        "Computing MKN statistics from {} (~{} n-grams)...",
                        source,
                        format_count(*estimated_ngrams)
                    ),
                );
            }

            ImportEvent::MknProgress {
                phase,
                total_phases,
                items_processed,
                total_items,
                percent_complete,
            } => {
                self.mkn_progress = Some(MknProgressState {
                    phase: *phase,
                    total_phases: *total_phases,
                    items_processed: *items_processed,
                    total_items: *total_items,
                    percent_complete: *percent_complete,
                });
            }

            ImportEvent::MknCompleted {
                continuation_entries,
                frequency_entries,
                duration,
            } => {
                log::debug!(
                    "[TUI-STATE] MknCompleted: mkn_in_progress {} -> false",
                    self.mkn_in_progress
                );
                self.mkn_in_progress = false;
                self.mkn_progress = None;
                self.add_log(
                    LogLevel::Info,
                    format!(
                        "MKN statistics complete: {} continuation, {} frequency entries in {:.1}s",
                        continuation_entries,
                        frequency_entries,
                        duration.as_secs_f64()
                    ),
                );
            }

            ImportEvent::MknFailed { error } => {
                self.mkn_in_progress = false;
                self.mkn_progress = None;
                self.error_message = Some(format!("MKN computation failed: {}", error));
                self.add_log(LogLevel::Error, format!("MKN FAILED: {}", error));
            }

            ImportEvent::PhaseChanged { phase } => {
                log::debug!(
                    "[TUI-STATE] PhaseChanged event received: '{}' -> '{}'",
                    self.current_phase,
                    phase
                );
                self.current_phase = phase.clone();
                self.add_log(LogLevel::Info, format!("Phase: {}", phase));
            }
        }
    }

    /// Add a log entry.
    fn add_log(&mut self, level: LogLevel, message: String) {
        let timestamp = chrono::Local::now().format("%H:%M:%S").to_string();

        self.log_entries.push_front(LogEntry {
            timestamp,
            level,
            message,
        });

        // Trim old entries
        while self.log_entries.len() > MAX_LOG_ENTRIES {
            self.log_entries.pop_back();
        }
    }

    /// Calculate progress percentage for current order.
    pub fn progress_percent(&self) -> f64 {
        if self.total_files == 0 {
            0.0
        } else {
            self.files_completed as f64 / self.total_files as f64 * 100.0
        }
    }

    /// Estimate time remaining.
    pub fn eta(&self) -> Option<Duration> {
        if self.files_completed == 0 || self.files_completed >= self.total_files {
            return None;
        }

        let elapsed_secs = self.elapsed.as_secs_f64();
        let rate = self.files_completed as f64 / elapsed_secs;
        let remaining_files = self.total_files - self.files_completed;
        let remaining_secs = remaining_files as f64 / rate;

        Some(Duration::from_secs_f64(remaining_secs))
    }

    /// Scroll log up.
    pub fn scroll_log_up(&mut self) {
        if self.log_scroll < self.log_entries.len().saturating_sub(1) {
            self.log_scroll += 1;
        }
    }

    /// Scroll log down.
    pub fn scroll_log_down(&mut self) {
        if self.log_scroll > 0 {
            self.log_scroll -= 1;
        }
    }

    /// Check if import is finished (completed, cancelled, or errored).
    pub fn is_finished(&self) -> bool {
        self.completed || self.cancelled || self.error_message.is_some()
    }

    /// Check if import encountered a fatal error.
    pub fn has_error(&self) -> bool {
        self.error_message.is_some()
    }

    /// Check if there are any warnings about failed or recovering prefixes.
    pub fn has_prefix_warnings(&self) -> bool {
        self.failed_prefixes_count > 0
            || self.retrying_prefixes_count > 0
            || self.recovering_prefixes_count > 0
    }

    /// Compute total files completed across all orders.
    ///
    /// Derives from per-order progress data, ensuring consistency
    /// with the Order Progress display.
    pub fn total_files_completed(&self) -> u64 {
        self.order_progress.values().map(|p| p.files_completed).sum()
    }

    /// Get the display phase, inferring from state if PhaseChanged event was missed.
    ///
    /// This provides resilience against broadcast channel lagging by deriving
    /// the phase from state flags (mkn_in_progress, merge_in_progress, etc.)
    /// rather than relying solely on the PhaseChanged event.
    pub fn display_phase(&self) -> &str {
        // Priority order: most specific state first
        let result = if self.merge_in_progress {
            "Merging Shards"
        } else if self.mkn_in_progress {
            "Computing MKN Statistics"
        } else if self.completed {
            "Complete"
        } else {
            // Check if all orders are complete but not yet finalized
            let all_orders_done = !self.order_progress.is_empty()
                && self.order_progress.values().all(|p| p.is_complete);
            if all_orders_done && self.workers.is_empty() {
                // Workers exited, orders complete - we're in cleanup or later
                if self.current_phase == "Cleaning Up" || self.current_phase == "Initializing" {
                    "Finalizing..."
                } else {
                    &self.current_phase
                }
            } else {
                // Fall back to the event-set phase
                &self.current_phase
            }
        };

        log::trace!(
            "[TUI-STATE] display_phase(): merge={}, mkn={}, completed={}, workers={}, current='{}' -> '{}'",
            self.merge_in_progress,
            self.mkn_in_progress,
            self.completed,
            self.workers.len(),
            self.current_phase,
            result
        );

        result
    }

    /// Get a summary of prefix warnings for display.
    pub fn prefix_warnings_summary(&self) -> Option<String> {
        if !self.has_prefix_warnings() {
            return None;
        }

        let mut parts = Vec::new();

        if self.failed_prefixes_count > 0 {
            parts.push(format!("{} failed", self.failed_prefixes_count));
        }
        if self.retrying_prefixes_count > 0 {
            parts.push(format!("{} retrying", self.retrying_prefixes_count));
        }
        if self.recovering_prefixes_count > 0 {
            parts.push(format!("{} recovering", self.recovering_prefixes_count));
        }

        Some(parts.join(", "))
    }
}

/// Format a large count with commas.
fn format_count(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_count() {
        assert_eq!(format_count(0), "0");
        assert_eq!(format_count(100), "100");
        assert_eq!(format_count(1000), "1,000");
        assert_eq!(format_count(1234567), "1,234,567");
    }

    #[test]
    fn test_state_apply_order_started() {
        let mut state = TuiState::new("en", 1, 5, 4);
        state.apply_event(&ImportEvent::OrderStarted {
            order: 2,
            total_files: 678,
        });

        assert_eq!(state.current_order, 2);
        assert_eq!(state.total_files, 678);
        assert_eq!(state.files_completed, 0);
        assert_eq!(state.log_entries.len(), 1);
    }

    #[test]
    fn test_state_apply_worker_started() {
        let mut state = TuiState::new("en", 1, 5, 4);
        state.apply_event(&ImportEvent::WorkerStarted {
            worker_id: 1,
            order: 2,
            prefix: "th".to_string(),
        });

        assert_eq!(state.workers[1].status, WorkerStatus::Downloading);
        assert_eq!(state.workers[1].prefix, Some("th".to_string()));
        assert_eq!(state.workers[1].order, Some(2));
    }

    #[test]
    fn test_progress_percent() {
        let mut state = TuiState::new("en", 1, 5, 4);
        state.total_files = 100;
        state.files_completed = 25;

        assert!((state.progress_percent() - 25.0).abs() < 0.001);
    }

    #[test]
    fn test_log_scrolling() {
        let mut state = TuiState::new("en", 1, 5, 4);

        // Add some log entries
        for i in 0..10 {
            state.add_log(LogLevel::Info, format!("Message {}", i));
        }

        assert_eq!(state.log_scroll, 0);

        state.scroll_log_up();
        assert_eq!(state.log_scroll, 1);

        state.scroll_log_down();
        assert_eq!(state.log_scroll, 0);

        // Can't scroll below 0
        state.scroll_log_down();
        assert_eq!(state.log_scroll, 0);
    }

    #[test]
    fn test_phase_changed_updates_current_phase() {
        let mut state = TuiState::new("en", 1, 5, 4);

        // Initial state should be "Initializing"
        assert_eq!(state.current_phase, "Initializing");

        // Apply PhaseChanged event
        state.apply_event(&ImportEvent::PhaseChanged {
            phase: "Importing N-grams".to_string(),
        });

        // current_phase should be updated
        assert_eq!(state.current_phase, "Importing N-grams");

        // Log should contain the phase change
        assert!(state
            .log_entries
            .iter()
            .any(|e| e.message.contains("Importing N-grams")));

        // Apply another phase change
        state.apply_event(&ImportEvent::PhaseChanged {
            phase: "Computing MKN Statistics".to_string(),
        });

        assert_eq!(state.current_phase, "Computing MKN Statistics");
    }

    #[test]
    fn test_display_phase_infers_mkn_in_progress() {
        let mut state = TuiState::new("en", 1, 5, 4);

        // Set mkn_in_progress but don't send PhaseChanged event
        // (simulating missed event due to lagging)
        state.mkn_in_progress = true;
        state.current_phase = "Cleaning Up".to_string(); // Stale phase

        // display_phase should infer the correct phase
        assert_eq!(state.display_phase(), "Computing MKN Statistics");
    }

    #[test]
    fn test_display_phase_infers_merge_in_progress() {
        let mut state = TuiState::new("en", 1, 5, 4);

        // Set merge_in_progress but don't send PhaseChanged event
        state.merge_in_progress = true;
        state.current_phase = "Computing MKN Statistics".to_string(); // Stale phase

        // display_phase should infer the correct phase
        assert_eq!(state.display_phase(), "Merging Shards");
    }

    #[test]
    fn test_display_phase_merge_takes_priority_over_mkn() {
        let mut state = TuiState::new("en", 1, 5, 4);

        // Both flags set (shouldn't happen in practice, but tests priority)
        state.mkn_in_progress = true;
        state.merge_in_progress = true;

        // Merge should take priority
        assert_eq!(state.display_phase(), "Merging Shards");
    }

    #[test]
    fn test_display_phase_completed() {
        let mut state = TuiState::new("en", 1, 5, 4);

        state.completed = true;
        state.current_phase = "Merging Shards".to_string(); // Stale phase

        assert_eq!(state.display_phase(), "Complete");
    }

    #[test]
    fn test_display_phase_falls_back_to_current_phase() {
        let mut state = TuiState::new("en", 1, 5, 4);

        // Normal case: no special flags, use current_phase
        state.current_phase = "Importing N-grams".to_string();

        assert_eq!(state.display_phase(), "Importing N-grams");
    }

    #[test]
    fn test_display_phase_finalizing_when_all_orders_done() {
        let mut state = TuiState::new("en", 1, 1, 4);

        // Simulate all orders complete, workers exited, but phase still shows cleanup
        state.order_progress.insert(1, OrderProgressState {
            files_completed: 27,
            total_files: 27,
            is_complete: true,
            files_succeeded: 27,
            files_skipped: 0,
            ngrams_processed: 1000,
        });
        state.workers.clear(); // All workers exited
        state.current_phase = "Cleaning Up".to_string();

        // Should infer we're finalizing
        assert_eq!(state.display_phase(), "Finalizing...");
    }

    #[test]
    fn test_total_files_completed_derives_from_order_progress() {
        let mut state = TuiState::new("en", 1, 3, 4);

        // Simulate OrderProgress events for multiple orders
        state.order_progress.insert(1, OrderProgressState {
            files_completed: 10,
            total_files: 27,
            is_complete: false,
            files_succeeded: 8,
            files_skipped: 2,
            ngrams_processed: 1000,
        });
        state.order_progress.insert(2, OrderProgressState {
            files_completed: 15,
            total_files: 676,
            is_complete: false,
            files_succeeded: 15,
            files_skipped: 0,
            ngrams_processed: 5000,
        });
        state.order_progress.insert(3, OrderProgressState {
            files_completed: 5,
            total_files: 676,
            is_complete: false,
            files_succeeded: 5,
            files_skipped: 0,
            ngrams_processed: 2000,
        });

        // total_files_completed should sum across all orders
        assert_eq!(state.total_files_completed(), 30); // 10 + 15 + 5

        // Verify it matches what Order Progress bar would show
        let sum: u64 = state.order_progress.values().map(|p| p.files_completed).sum();
        assert_eq!(state.total_files_completed(), sum);
    }

    #[test]
    fn test_total_files_completed_empty_when_no_orders() {
        let state = TuiState::new("en", 1, 5, 4);

        // No orders have started yet
        assert_eq!(state.total_files_completed(), 0);
    }
}
