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
}

/// Progress state for a single n-gram order.
#[derive(Debug, Clone, Default)]
pub struct OrderProgressState {
    /// Files completed for this order.
    pub files_completed: u64,

    /// Total files for this order.
    pub total_files: u64,

    /// N-grams processed for this order.
    pub ngrams_processed: u64,

    /// Whether this order is fully complete.
    pub is_complete: bool,
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

        Self {
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
        }
    }

    /// Apply a domain event to update the state.
    pub fn apply_event(&mut self, event: &ImportEvent) {
        match event {
            ImportEvent::OrderStarted { order, total_files } => {
                self.current_order = *order;
                self.total_files = *total_files;
                self.files_completed = 0;
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
                self.error_message = Some(message.clone());
                self.add_log(LogLevel::Error, format!("FATAL: {}", message));
            }

            ImportEvent::Log { level, message } => {
                self.add_log(*level, message.clone());
            }

            ImportEvent::WorkerExited { worker_id } => {
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
            } => {
                let progress = self.order_progress.entry(*order).or_default();
                progress.files_completed = *files_completed;
                progress.total_files = *total_files;
                progress.ngrams_processed = *ngrams_processed;
                progress.is_complete = *is_complete;
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
}
