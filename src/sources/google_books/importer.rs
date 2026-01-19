//! Google Books N-gram importer.
//!
//! Orchestrates the import process from Google Books N-grams into a PersistentARTrie,
//! with checkpoint/resume support for long-running imports.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use liblevenshtein::dictionary::persistent_artrie_char::DiskBackedCharTrieInner;
use parking_lot::RwLock;

use super::sharding::MknAggregator;
use super::storage::{NgramStorage, StorageError};


use super::aggregator::YearAggregator;
use super::checkpoint::{
    CheckpointError, CheckpointStats, ImportCheckpoint, MknPhase, TrieCheckpointStorage,
};
use super::config::GoogleBooksConfig;
use super::events::{ImportCommand, ImportEvent, LogLevel};
use super::languages::{get_file_url, get_metadata, get_prefixes, is_supported};
use super::parser::NgramRecord;
use super::reader::{FileNgramReader, ReaderError};
#[cfg(feature = "google-books")]
use super::task_manager::RetryAfter;

// ============================================================================
// N-gram Count Estimation
// ============================================================================

/// Estimate the number of n-grams for a given configuration.
///
/// This is used to decide whether sharding should be enabled in auto mode.
/// The estimates are based on empirical data from Google Books n-gram corpus.
fn estimate_ngram_count(config: &GoogleBooksConfig) -> u64 {
    // Estimates for English (other languages have fewer)
    // These are rough estimates based on Google Books v3 dataset
    let per_order: &[u64] = match config.language.as_str() {
        "en" | "eng" => &[
            0,          // Order 0 (unused)
            13_000_000, // 1-grams: ~13M
            314_000_000, // 2-grams: ~314M
            977_000_000, // 3-grams: ~977M
            1_313_000_000, // 4-grams: ~1.3B
            1_176_000_000, // 5-grams: ~1.2B
        ],
        _ => &[
            0,          // Order 0 (unused)
            5_000_000,  // 1-grams (estimate for non-English)
            100_000_000, // 2-grams
            300_000_000, // 3-grams
            500_000_000, // 4-grams
            400_000_000, // 5-grams
        ],
    };

    let mut total = 0u64;
    for order in config.orders.clone() {
        if let Some(&count) = per_order.get(order as usize) {
            // Apply min_count filter estimate (higher min_count = fewer n-grams)
            // This is a rough estimate: each 10x increase in min_count
            // reduces count by ~60-70%
            let factor = match config.min_count {
                0..=1 => 1.0,
                2..=10 => 0.4,
                11..=40 => 0.2,
                41..=100 => 0.1,
                _ => 0.05,
            };
            total += (count as f64 * factor) as u64;
        }
    }

    total
}

// ============================================================================
// Free functions for parallel processing
// ============================================================================

/// Check if an error is retryable (transient network issues).
///
/// This is a free function to enable use from both the importer methods
/// and the parallel processing function.
///
/// Detects various transient error patterns including:
/// - Explicit timeouts ("timeout", "timed out")
/// - Tokio/hyper timeout indicators ("elapsed", "deadline")
/// - Connection errors ("connection", "reset", "refused", "unreachable")
/// - Network errors ("network", "temporarily", "broken pipe")
fn is_retryable_error(e: &ImportError) -> bool {
    match e {
        ImportError::Reader(reader_err) => {
            let msg = reader_err.to_string().to_lowercase();
            // Retry on connection timeouts, network errors, temporary failures
            msg.contains("timeout")
                || msg.contains("timed out")
                || msg.contains("elapsed") // tokio/hyper: "deadline has elapsed"
                || msg.contains("deadline") // tokio/hyper: "deadline elapsed"
                || msg.contains("connection")
                || msg.contains("connect") // catches "error trying to connect"
                || msg.contains("network")
                || msg.contains("temporarily")
                || msg.contains("reset")
                || msg.contains("broken pipe")
                || msg.contains("refused") // connection refused
                || msg.contains("unreachable") // host unreachable
                || msg.contains("error sending request") // reqwest generic request failure
                || msg.contains("request") // broader request failures
                || msg.contains("dns") // DNS resolution failures
                || msg.contains("resolve") // name resolution failures
                || msg.contains("decoding") // gzip decode errors from truncated responses
                || msg.contains("decode") // general decode failures
        }
        ImportError::Io(io_err) => {
            // Handle I/O errors that may wrap network errors
            let msg = io_err.to_string().to_lowercase();
            msg.contains("timeout")
                || msg.contains("timed out")
                || msg.contains("elapsed")
                || msg.contains("deadline")
                || msg.contains("connection")
                || msg.contains("connect") // catches "error trying to connect"
                || msg.contains("network")
                || msg.contains("temporarily")
                || msg.contains("reset")
                || msg.contains("broken pipe")
                || msg.contains("refused")
                || msg.contains("unreachable")
                || msg.contains("error sending request") // reqwest generic request failure
                || msg.contains("request") // broader request failures
                || msg.contains("dns") // DNS resolution failures
                || msg.contains("resolve") // name resolution failures
                || msg.contains("decoding") // gzip decode errors from truncated responses
                || msg.contains("decode") // general decode failures
                // Also check ErrorKind for structured detection
                || io_err.kind() == std::io::ErrorKind::TimedOut
                || io_err.kind() == std::io::ErrorKind::ConnectionReset
                || io_err.kind() == std::io::ErrorKind::ConnectionRefused
                || io_err.kind() == std::io::ErrorKind::ConnectionAborted
                || io_err.kind() == std::io::ErrorKind::NotConnected
        }
        _ => false,
    }
}

/// Extract RetryAfter from an ImportError if it's a rate limiting error.
///
/// This inspects the underlying ReaderError to check for the RateLimited variant
/// and extracts the Retry-After header value if present.
#[cfg(feature = "google-books")]
fn extract_retry_after(error: &ImportError) -> Option<RetryAfter> {
    match error {
        ImportError::Reader(ReaderError::RateLimited { retry_after, .. }) => retry_after.clone(),
        _ => None,
    }
}

/// Check if an error is specifically a rate limit error (HTTP 429).
#[cfg(feature = "google-books")]
fn is_rate_limit_error(error: &ImportError) -> bool {
    matches!(
        error,
        ImportError::Reader(ReaderError::RateLimited { .. })
    )
}

/// Result of storing an n-gram, with counter deltas for batched updates.
///
/// This enables callers to batch atomic counter updates instead of
/// updating on every n-gram, reducing cache-line bouncing across workers.
#[derive(Debug, Clone, Copy)]
pub struct NgramStorageResult {
    /// Whether the n-gram was new (first occurrence).
    pub is_new: bool,
}

/// Batch size for atomic counter updates.
///
/// With 8 workers processing millions of n-grams, batching every 10,000
/// reduces atomic operations by ~1000x and eliminates cache-line bouncing.
pub const COUNTER_BATCH_SIZE: u64 = 10_000;

/// Store an n-gram using shared Arc references (for parallel processing).
///
/// This is extracted as a free function to enable parallel HTTP downloads,
/// where multiple tasks need to store n-grams concurrently. The function
/// takes Arc references to the shared trie rather than `&self`.
///
/// Returns `NgramStorageResult` so callers can batch atomic counter updates.
/// Callers should accumulate local counts and flush to atomics periodically
/// using `COUNTER_BATCH_SIZE`.
///
/// Note: MKN statistics are computed as a post-processing step after import
/// completes, not during n-gram storage. This eliminates lock contention
/// from the dedup tries that were previously required for on-the-fly MKN.
fn store_ngram_shared(
    ngram: &str,
    count: u64,
    storage: &Arc<NgramStorage>,
) -> Result<NgramStorageResult, ImportError> {
    // Store using the storage abstraction (handles both single-trie and sharded)
    let is_new = storage.store(ngram, count).map_err(|e| {
        ImportError::Trie(format!("Failed to store ngram '{}': {}", ngram, e))
    })?;

    Ok(NgramStorageResult { is_new })
}

/// Legacy version for direct trie access (used during migration).
#[allow(dead_code)]
fn store_ngram_shared_legacy(
    ngram: &str,
    count: u64,
    trie: &Arc<RwLock<DiskBackedCharTrieInner<u64>>>,
) -> Result<NgramStorageResult, ImportError> {
    let mut trie_guard = trie.write();
    let is_new = trie_guard.get(ngram).is_none();
    trie_guard.increment(ngram, count as i64).map_err(|e| {
        ImportError::Trie(format!("Failed to store ngram '{}': {}", ngram, e))
    })?;
    Ok(NgramStorageResult { is_new })
}

// ============================================================================
// TrieCheckpointStorage Implementation
// ============================================================================

/// Error type for trie checkpoint operations.
#[derive(Debug, thiserror::Error)]
pub enum TrieCheckpointError {
    /// Trie operation failed.
    #[error("Trie operation failed: {0}")]
    TrieError(String),
}

impl TrieCheckpointStorage for DiskBackedCharTrieInner<u64> {
    type Error = TrieCheckpointError;

    fn store_checkpoint_u64(&mut self, key: &str, value: u64) -> Result<(), Self::Error> {
        // Use upsert to store or update the value
        self.upsert(key, value)
            .map_err(|e| TrieCheckpointError::TrieError(e.to_string()))?;
        Ok(())
    }

    fn load_checkpoint_u64(&self, key: &str) -> Result<Option<u64>, Self::Error> {
        // Use get to retrieve the value
        Ok(self.get(key).copied())
    }

    fn delete_checkpoint_key(&mut self, key: &str) -> Result<bool, Self::Error> {
        // Use remove to delete the key
        self.remove(key)
            .map_err(|e| TrieCheckpointError::TrieError(e.to_string()))
    }

    fn delete_checkpoint_prefix(&mut self, prefix: &str) -> Result<usize, Self::Error> {
        // Use remove_prefix to delete all keys with the given prefix
        self.remove_prefix(prefix)
            .map_err(|e| TrieCheckpointError::TrieError(e.to_string()))
    }

    fn iter_checkpoint_prefix(&self, prefix: &str) -> Result<Vec<(String, u64)>, Self::Error> {
        // Use iter_prefix_with_values to get all keys and values with the given prefix
        match self.iter_prefix_with_values(prefix) {
            Ok(Some(entries)) => Ok(entries),
            Ok(None) => Ok(Vec::new()),
            Err(e) => Err(TrieCheckpointError::TrieError(e.to_string())),
        }
    }
}

// ============================================================================
// Worker Pool Infrastructure
// ============================================================================

/// Maximum retry attempts for transient failures.
#[cfg(feature = "google-books")]
const MAX_RETRIES: u8 = 5;

/// Initial backoff delay in milliseconds (doubles each retry).
#[cfg(feature = "google-books")]
const INITIAL_BACKOFF_MS: u64 = 1000;

/// A job for the worker pool to process.
#[cfg(feature = "google-books")]
#[derive(Clone)]
struct Job {
    /// URL of the prefix file to download.
    url: Arc<str>,
    /// The prefix being downloaded (e.g., "th", "to").
    prefix: Arc<str>,
    /// N-gram order for this job (1-5).
    order: u8,
    /// Current retry attempt (0 = first attempt).
    attempt: u8,
    /// Backoff duration in ms for next retry (doubles each attempt).
    backoff_ms: u64,
    /// When this job can be executed (None = ready immediately).
    /// Used for deferred retries without blocking the worker.
    ready_at: Option<std::time::Instant>,
}

#[cfg(feature = "google-books")]
impl Job {
    /// Create a new job for first attempt.
    fn new(url: impl Into<Arc<str>>, prefix: impl Into<Arc<str>>, order: u8) -> Self {
        Self {
            url: url.into(),
            prefix: prefix.into(),
            order,
            attempt: 0,
            backoff_ms: INITIAL_BACKOFF_MS,
            ready_at: None, // Ready immediately
        }
    }

    /// Create a retry job with incremented attempt, doubled backoff, and ready_at set.
    /// Arc<str> cloning is cheap (just a pointer increment).
    fn with_retry(&self) -> Self {
        let new_backoff = self.backoff_ms.saturating_mul(2);
        Self {
            url: Arc::clone(&self.url),
            prefix: Arc::clone(&self.prefix),
            order: self.order,
            attempt: self.attempt + 1,
            backoff_ms: new_backoff,
            ready_at: Some(std::time::Instant::now() + Duration::from_millis(new_backoff)),
        }
    }

    /// Create a retry job using the Retry-After header value if available.
    ///
    /// If `retry_after` is `Some`, uses that duration for the retry delay.
    /// Otherwise falls back to the exponential backoff (doubled from previous).
    fn with_retry_after(&self, retry_after: Option<RetryAfter>) -> Self {
        let new_backoff = self.backoff_ms.saturating_mul(2);
        let ready_at = match retry_after {
            Some(ra) => {
                // Use Retry-After header value
                let duration = ra.to_duration();
                // Also update backoff_ms for future retries (if Retry-After is larger)
                std::time::Instant::now() + duration
            }
            None => {
                // Fall back to exponential backoff
                std::time::Instant::now() + Duration::from_millis(new_backoff)
            }
        };

        Self {
            url: Arc::clone(&self.url),
            prefix: Arc::clone(&self.prefix),
            order: self.order,
            attempt: self.attempt + 1,
            backoff_ms: new_backoff,
            ready_at: Some(ready_at),
        }
    }

    /// Check if this job is ready to execute (ready_at has elapsed or is None).
    fn is_ready(&self) -> bool {
        self.ready_at
            .map(|t| t <= std::time::Instant::now())
            .unwrap_or(true)
    }

    /// Get the duration until this job is ready, or None if already ready.
    fn time_until_ready(&self) -> Option<Duration> {
        self.ready_at
            .and_then(|t| t.checked_duration_since(std::time::Instant::now()))
    }
}

/// Result of a job processing attempt.
///
/// Always includes the order and prefix so the main task knows which prefix
/// succeeded or failed. This enables the main task to mark failed prefixes
/// in the checkpoint and continue with other prefixes instead of aborting.
#[cfg(feature = "google-books")]
#[derive(Debug)]
struct JobResult {
    /// N-gram order for this job (1-5).
    order: u8,
    /// The prefix that was processed (e.g., "th", "to").
    prefix: Arc<str>,
    /// The result: n-gram count on success, or error details on failure.
    outcome: JobOutcome,
}

/// Outcome of processing a single prefix file.
#[cfg(feature = "google-books")]
#[derive(Debug)]
enum JobOutcome {
    /// Successfully processed the prefix file.
    Success {
        /// Number of n-grams processed from this file.
        ngram_count: u64,
    },
    /// Failed after exhausting all retry attempts.
    Failed {
        /// The error that caused the failure.
        error: ImportError,
        /// Number of retry attempts made.
        attempts: u32,
    },
    /// Skipped after exhausting retries - will be retried next session.
    Skipped {
        /// The error that caused the skip.
        error: ImportError,
        /// Number of retry attempts made.
        attempts: u32,
    },
}

/// Debug information captured from a failed HTTP request.
///
/// Used to provide detailed logging for retry and skip decisions.
#[cfg(feature = "google-books")]
#[derive(Debug, Clone)]
struct RequestDebugInfo {
    /// URL that was requested.
    url: String,
    /// HTTP status code (if available).
    status_code: Option<u16>,
    /// Relevant response headers.
    headers: Vec<(String, String)>,
    /// Time taken for the request in milliseconds.
    response_time_ms: u64,
    /// Error message.
    error_message: String,
}

#[cfg(feature = "google-books")]
impl RequestDebugInfo {
    /// Create debug info from an error and request timing.
    fn from_error(url: &str, error: &ImportError, response_time: Duration) -> Self {
        // Extract status code and headers if available from the error
        let (status_code, headers) = match error {
            ImportError::Reader(e) => {
                // Try to extract HTTP info from the error message
                let msg = e.to_string();
                let status = if msg.contains("404") {
                    Some(404)
                } else if msg.contains("429") {
                    Some(429)
                } else if msg.contains("500") {
                    Some(500)
                } else if msg.contains("503") {
                    Some(503)
                } else {
                    None
                };
                (status, vec![])
            }
            _ => (None, vec![]),
        };

        Self {
            url: url.to_string(),
            status_code,
            headers,
            response_time_ms: response_time.as_millis() as u64,
            error_message: error.to_string(),
        }
    }
}

/// Outcome of processing a single prefix file (for buffer_unordered pattern).
///
/// Unlike `JobOutcome` which is used by the worker_task pattern with persistent workers,
/// this enum supports the one-shot future pattern used by `process_prefix_file`.
#[cfg(feature = "google-books")]
enum PrefixOutcome {
    /// Successfully processed the prefix file.
    Success {
        /// The prefix that was processed.
        prefix: Arc<str>,
        /// Number of n-grams processed from this file.
        ngram_count: u64,
    },
    /// Failed with retryable error - should be retried after delay.
    Deferred {
        /// URL of the prefix file.
        url: Arc<str>,
        /// The prefix being processed.
        prefix: Arc<str>,
        /// N-gram order.
        order: u8,
        /// Current retry attempt (will be incremented).
        attempt: u8,
        /// Backoff duration in ms for next retry.
        backoff_ms: u64,
        /// The error that triggered the retry.
        error: ImportError,
    },
    /// Failed permanently (non-retryable error or max retries exceeded).
    Failed {
        /// The prefix that failed.
        prefix: Arc<str>,
        /// The error that caused the failure.
        error: ImportError,
        /// Number of retry attempts made.
        attempts: u32,
    },
}

/// Shared state for worker tasks.
#[cfg(feature = "google-books")]
struct WorkerSharedState {
    config: GoogleBooksConfig,
    storage: Arc<NgramStorage>,
    total_ngrams: Arc<AtomicU64>,
    unique_ngrams: Arc<AtomicU64>,
    progress_tx: tokio::sync::mpsc::Sender<WorkerUpdate>,
    paused: Arc<AtomicBool>,
    /// Current number of jobs in the queue (for all-deferred detection)
    queue_size: Arc<AtomicUsize>,
}

/// Process a single job attempt (no retry loop - single attempt only).
///
/// This helper extracts the core processing logic from worker_task to enable
/// non-blocking retry with DelayQueue.
#[cfg(feature = "google-books")]
async fn process_single_attempt(
    job: &Job,
    shared: &WorkerSharedState,
    worker_id: usize,
) -> Result<u64, ImportError> {
    use super::reader::HttpNgramReader;
    use tokio_stream::StreamExt;

    // Add small random delay to stagger connection starts (reduces rate limiting)
    let jitter_ms = rand::random::<u64>() % 500;
    tokio::time::sleep(Duration::from_millis(jitter_ms)).await;

    let mut reader = HttpNgramReader::with_options(
        &job.url,
        shared.config.skip_pos_tags,
        shared.config.min_count,
    );

    let stream = reader.stream_aggregated(shared.config.year_range);
    tokio::pin!(stream);

    // Progress is emitted every 5 seconds (time-based, not count-based)
    // Longer interval reduces channel pressure and TUI overhead
    let mut last_progress_time = Instant::now();
    let progress_interval = Duration::from_secs(5);

    // Local counters for batched atomic updates (reduces cache-line bouncing)
    let mut local_total: u64 = 0;
    let mut local_unique: u64 = 0;

    let mut count = 0u64;
    while let Some(result) = stream.next().await {
        let agg = result?;
        let storage_result = store_ngram_shared(
            &agg.ngram,
            agg.total_count,
            &shared.storage,
        )?;
        count += 1;
        local_total += 1;
        if storage_result.is_new {
            local_unique += 1;
        }

        // Batch flush atomic counters every COUNTER_BATCH_SIZE n-grams
        if local_total >= COUNTER_BATCH_SIZE {
            shared.total_ngrams.fetch_add(local_total, Ordering::Relaxed);
            if local_unique > 0 {
                shared.unique_ngrams.fetch_add(local_unique, Ordering::Relaxed);
            }
            local_total = 0;
            local_unique = 0;
        }

        // Emit progress at 5-second intervals to reduce channel pressure
        if last_progress_time.elapsed() >= progress_interval {
            let _ = shared.progress_tx.try_send(WorkerUpdate::NgramProgress {
                worker_id,
                ngram_count: count,
            });
            last_progress_time = Instant::now();
        }
    }

    // Flush remaining counts
    if local_total > 0 {
        shared.total_ngrams.fetch_add(local_total, Ordering::Relaxed);
    }
    if local_unique > 0 {
        shared.unique_ngrams.fetch_add(local_unique, Ordering::Relaxed);
    }

    Ok(count)
}

/// Persistent worker task that polls jobs from a shared queue.
///
/// This function runs in a loop, processing jobs until:
/// - The job queue is empty (all work completed)
/// - A shutdown signal is received (worker should exit)
///
/// ## Retry with Exponential Backoff
///
/// When a job fails with a retryable error, the worker sleeps for an exponential
/// backoff period and retries the same job. This blocks the worker during the
/// backoff period, but ensures reliable completion of each job.
///
/// For higher throughput with non-blocking retry, use `import_http_with_progress`
/// which implements deferred retry at the caller level.
///
/// # Arguments
///
/// * `worker_id` - Static ID for this worker (for logging/tracking)
/// * `job_rx` - Shared receiver for the job queue
/// * `shutdown_rx` - Watch channel to signal worker shutdown
/// * `shared` - Shared state including tries, config, progress channel
/// * `result_tx` - Channel to send job results back to main task
/// * `worker_exit_tx` - Channel to notify main task when this worker exits
#[cfg(feature = "google-books")]
async fn worker_task(
    worker_id: usize,
    job_rx: Arc<tokio::sync::Mutex<tokio::sync::mpsc::Receiver<Job>>>,
    job_tx: tokio::sync::mpsc::Sender<Job>,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
    shared: Arc<WorkerSharedState>,
    result_tx: tokio::sync::mpsc::Sender<JobResult>,
    worker_exit_tx: tokio::sync::mpsc::Sender<usize>,
) {
    // Track consecutive deferred jobs for all-deferred detection
    let mut consecutive_deferred = 0usize;
    let mut earliest_ready: Option<Instant> = None;
    loop {
        // Check shutdown signal BEFORE polling for work
        if *shutdown_rx.borrow() {
            log::debug!("Worker {} shutting down", worker_id);
            break;
        }

        // Get next job from queue
        let job = {
            let mut rx = job_rx.lock().await;
            tokio::select! {
                biased;
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        log::debug!("Worker {} received shutdown signal while waiting for job", worker_id);
                        break;
                    }
                    continue;
                }
                job = rx.recv() => job,
            }
        };

        let Some(job) = job else {
            // Queue closed with no more jobs - but we might have deferred jobs pending.
            // If all jobs were deferred (waiting on retry backoff), we need to check.
            if consecutive_deferred > 0 {
                log::debug!(
                    "Worker {} queue closed with {} deferred jobs pending",
                    worker_id, consecutive_deferred
                );
            }
            log::debug!("Worker {} finished - queue empty", worker_id);
            break;
        };

        // Check if job is ready to execute (not waiting on retry backoff)
        if !job.is_ready() {
            // Job not ready - track for all-deferred detection
            consecutive_deferred += 1;
            if let Some(ready_at) = job.ready_at {
                earliest_ready = Some(match earliest_ready {
                    Some(e) => e.min(ready_at),
                    None => ready_at,
                });
            }

            // Requeue to back of queue
            let _ = job_tx.send(job).await;

            // Check if we've cycled through entire queue (all jobs deferred)
            let queue_size = shared.queue_size.load(Ordering::SeqCst);
            if queue_size > 0 && consecutive_deferred >= queue_size {
                // All jobs are deferred - block until earliest is ready
                if let Some(ready_at) = earliest_ready {
                    let wait = ready_at.saturating_duration_since(Instant::now());
                    if !wait.is_zero() {
                        log::debug!(
                            "Worker {} blocking {}ms - all {} jobs deferred",
                            worker_id, wait.as_millis(), queue_size
                        );
                        tokio::time::sleep(wait).await;
                    }
                }
                consecutive_deferred = 0;
                earliest_ready = None;
            }
            continue;
        }

        // Job is ready - reset deferred tracking
        consecutive_deferred = 0;
        earliest_ready = None;

        // Decrement queue size since we're executing this job (not requeuing)
        shared.queue_size.fetch_sub(1, Ordering::SeqCst);

        // Check for pause before processing
        while shared.paused.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(100)).await;
            if *shutdown_rx.borrow() {
                break;
            }
        }

        // Send "Started" update
        let _ = shared.progress_tx.try_send(WorkerUpdate::Started {
            worker_id,
            order: job.order,
            prefix: job.prefix.clone(),
        });

        // Single attempt - no blocking retry loop
        // On retryable error, requeue with ready_at set and pick up next job
        let start_time = Instant::now();
        let result = process_single_attempt(&job, &shared, worker_id).await;
        let elapsed = start_time.elapsed();

        match result {
            Ok(count) => {
                // Success - send completion update and result
                let _ = shared.progress_tx.try_send(WorkerUpdate::Finished {
                    worker_id,
                    order: job.order,
                    prefix: job.prefix.clone(),
                    ngram_count: count,
                });
                let job_result = JobResult {
                    order: job.order,
                    prefix: job.prefix,
                    outcome: JobOutcome::Success { ngram_count: count },
                };
                if result_tx.send(job_result).await.is_err() {
                    // Main task dropped, exit worker
                    let _ = worker_exit_tx.send(worker_id).await;
                    let _ = shared.progress_tx.try_send(WorkerUpdate::Exited { worker_id });
                    return;
                }
            }
            Err(e) if is_retryable_error(&e) && job.attempt < MAX_RETRIES => {
                // Retryable error - requeue with ready_at set, pick up next job immediately
                // Extract Retry-After header if this was a rate limit error
                let retry_after = extract_retry_after(&e);
                let retry_job = job.with_retry_after(retry_after.clone());
                let debug_info = RequestDebugInfo::from_error(&job.url, &e, elapsed);

                // Calculate actual delay for logging
                let delay_ms = retry_job.ready_at
                    .map(|ra| ra.saturating_duration_since(std::time::Instant::now()).as_millis() as u64)
                    .unwrap_or(retry_job.backoff_ms);

                // Log detailed debug info (including Retry-After if present)
                log::debug!(
                    "Worker {} deferring {} (order {}) - attempt {}/{}, retry at +{}ms{}\n\
                     URL: {}\n\
                     Error: {}\n\
                     Status code: {:?}\n\
                     Response time: {}ms",
                    worker_id,
                    retry_job.prefix,
                    retry_job.order,
                    retry_job.attempt,
                    MAX_RETRIES,
                    delay_ms,
                    if retry_after.is_some() { " (from Retry-After header)" } else { "" },
                    debug_info.url,
                    debug_info.error_message,
                    debug_info.status_code,
                    debug_info.response_time_ms,
                );

                // Emit deferred event (using Retrying for UI compatibility)
                let _ = shared.progress_tx.try_send(WorkerUpdate::Retrying {
                    worker_id,
                    prefix: Arc::clone(&retry_job.prefix),
                    attempt: retry_job.attempt as u32,
                    error: Arc::from(e.to_string()),
                });

                // Requeue with ready_at set - will be picked up after delay
                // Increment queue size since we're adding back
                shared.queue_size.fetch_add(1, Ordering::SeqCst);
                let _ = job_tx.send(retry_job).await;

                // Worker immediately picks up next job (non-blocking)
            }
            Err(error) => {
                // Non-retryable error or max retries exceeded - skip for this session
                let debug_info = RequestDebugInfo::from_error(&job.url, &error, elapsed);

                // Determine if this was max retries exceeded (retryable) or non-retryable
                let is_max_retries = is_retryable_error(&error) && job.attempt >= MAX_RETRIES;

                if is_max_retries {
                    // Max retries exceeded - skip for this session, will retry next run
                    log::warn!(
                        "Worker {} SKIPPING prefix {} (order {}) after {} failed attempts - will retry next session\n\
                         URL: {}\n\
                         Final error: {}\n\
                         Status code: {:?}\n\
                         Response time: {}ms",
                        worker_id,
                        job.prefix,
                        job.order,
                        job.attempt + 1,
                        debug_info.url,
                        debug_info.error_message,
                        debug_info.status_code,
                        debug_info.response_time_ms,
                    );

                    let job_result = JobResult {
                        order: job.order,
                        prefix: job.prefix,
                        outcome: JobOutcome::Skipped {
                            error,
                            attempts: (job.attempt + 1) as u32,
                        },
                    };
                    if result_tx.send(job_result).await.is_err() {
                        let _ = worker_exit_tx.send(worker_id).await;
                        let _ = shared.progress_tx.try_send(WorkerUpdate::Exited { worker_id });
                        return;
                    }
                } else {
                    // Non-retryable error - permanent failure
                    log::warn!(
                        "Worker {} FAILED on prefix {} (order {}) - non-retryable error after {} attempts\n\
                         URL: {}\n\
                         Error: {}\n\
                         Status code: {:?}\n\
                         Response time: {}ms",
                        worker_id,
                        job.prefix,
                        job.order,
                        job.attempt + 1,
                        debug_info.url,
                        debug_info.error_message,
                        debug_info.status_code,
                        debug_info.response_time_ms,
                    );

                    let job_result = JobResult {
                        order: job.order,
                        prefix: job.prefix,
                        outcome: JobOutcome::Failed {
                            error,
                            attempts: (job.attempt + 1) as u32,
                        },
                    };
                    if result_tx.send(job_result).await.is_err() {
                        let _ = worker_exit_tx.send(worker_id).await;
                        let _ = shared.progress_tx.try_send(WorkerUpdate::Exited { worker_id });
                        return;
                    }
                }
            }
        }
    }

    // Notify main task that this worker is exiting (for active worker tracking)
    let _ = worker_exit_tx.send(worker_id).await;

    // Emit exited event so TUI can remove the worker from display
    let _ = shared.progress_tx.try_send(WorkerUpdate::Exited { worker_id });
    log::debug!("Worker {} exited", worker_id);
}

/// Process a single prefix file and store n-grams (for parallel processing).
///
/// This is extracted as a standalone async function to enable parallel HTTP
/// downloads using `futures::stream::buffer_unordered`. Each task downloads
/// and parses a single prefix file, storing n-grams to the shared tries.
///
/// Uses streaming to avoid buffering entire files in memory. Large 2-gram files
/// can contain 50-100M n-grams (6-8GB in memory), so streaming is essential.
///
/// ## Non-Blocking Retry Pattern
///
/// This function performs a SINGLE attempt and returns immediately with a
/// `PrefixOutcome`. On retryable errors, it returns `Deferred` with retry
/// metadata instead of blocking with sleep. The caller is responsible for
/// collecting deferred items and re-processing them after a delay.
///
/// This pattern prevents all `buffer_unordered` slots from being blocked by
/// sleeping workers, which was causing progress to halt when many requests
/// needed retry simultaneously.
///
/// # Worker ID Pool
///
/// Worker IDs are claimed dynamically from a shared pool at the start of processing
/// and returned to the pool when done. This ensures that concurrent workers always
/// have unique IDs, even when using `buffer_unordered` which can interleave futures
/// in unpredictable order.
///
/// # Arguments
///
/// * `worker_id_pool_tx` - Channel to return worker ID when done (send ID back to pool)
/// * `worker_id_pool_rx` - Shared receiver to claim worker ID at start
/// * `url` - URL of the prefix file to download
/// * `prefix` - The prefix being downloaded (e.g., "th", "to")
/// * `order` - N-gram order (1-5)
/// * `attempt` - Current retry attempt (0 = first attempt)
/// * `backoff_ms` - Backoff delay in ms if this attempt fails (for next retry)
/// * `config` - Import configuration
/// * `storage` - Storage backend for n-grams (single-trie or sharded)
/// * `total_ngrams` - Atomic counter for total n-grams
/// * `unique_ngrams` - Atomic counter for unique n-grams
/// * `progress_tx` - Optional channel for sending progress updates
#[cfg(feature = "google-books")]
async fn process_prefix_file(
    worker_id_pool_tx: tokio::sync::mpsc::Sender<usize>,
    worker_id_pool_rx: Arc<tokio::sync::Mutex<tokio::sync::mpsc::Receiver<usize>>>,
    url: Arc<str>,
    prefix: Arc<str>,
    order: u8,
    attempt: u8,
    backoff_ms: u64,
    config: GoogleBooksConfig,
    storage: Arc<NgramStorage>,
    total_ngrams: Arc<AtomicU64>,
    unique_ngrams: Arc<AtomicU64>,
    progress_tx: Option<tokio::sync::mpsc::Sender<WorkerUpdate>>,
) -> PrefixOutcome {
    use super::reader::HttpNgramReader;
    use tokio_stream::StreamExt;

    // Claim a worker ID from the pool - this blocks until a slot is available.
    // This ensures each concurrent worker has a unique ID for display purposes.
    let worker_id = {
        let mut rx = worker_id_pool_rx.lock().await;
        rx.recv().await.expect("Worker ID pool closed unexpectedly")
    };

    // Helper to return worker ID to pool (used on both success and error)
    let return_worker_id = |pool_tx: tokio::sync::mpsc::Sender<usize>, id: usize| async move {
        let _ = pool_tx.send(id).await;
    };

    // Send "Started" update (include retry info if this is a retry)
    // Using try_send for backpressure - dropping updates is acceptable for progress
    if let Some(ref tx) = progress_tx {
        if attempt > 0 {
            let _ = tx.try_send(WorkerUpdate::Retrying {
                worker_id,
                prefix: Arc::clone(&prefix),
                attempt: attempt as u32,
                error: Arc::from("Resuming deferred retry"),
            });
        } else {
            let _ = tx.try_send(WorkerUpdate::Started {
                worker_id,
                order,
                prefix: Arc::clone(&prefix),
            });
        }
    }

    // Add small random delay to stagger connection starts (reduces rate limiting)
    let jitter_ms = rand::random::<u64>() % 500;
    tokio::time::sleep(Duration::from_millis(jitter_ms)).await;

    // Single attempt processing
    let result: Result<u64, ImportError> = async {
        let mut reader = HttpNgramReader::with_options(
            &url,
            config.skip_pos_tags,
            config.min_count,
        );

        // Stream n-grams instead of buffering entire file in memory.
        // This is critical for large 2-gram files (50-100M n-grams, 6-8GB).
        let stream = reader.stream_aggregated(config.year_range);
        tokio::pin!(stream);

        // Emit progress every 50,000 n-grams for TUI updates.
        // Larger interval reduces channel pressure for high-volume files.
        const NGRAM_PROGRESS_INTERVAL: u64 = 50_000;

        // Local counters for batched atomic updates (reduces cache-line bouncing)
        let mut local_total: u64 = 0;
        let mut local_unique: u64 = 0;

        let mut count = 0u64;
        while let Some(result) = stream.next().await {
            let agg = result?;
            let storage_result = store_ngram_shared(
                &agg.ngram,
                agg.total_count,
                &storage,
            )?;
            count += 1;
            local_total += 1;
            if storage_result.is_new {
                local_unique += 1;
            }

            // Batch flush atomic counters every COUNTER_BATCH_SIZE n-grams
            if local_total >= COUNTER_BATCH_SIZE {
                total_ngrams.fetch_add(local_total, Ordering::Relaxed);
                if local_unique > 0 {
                    unique_ngrams.fetch_add(local_unique, Ordering::Relaxed);
                }
                local_total = 0;
                local_unique = 0;
            }

            // Emit periodic progress for TUI display
            if count % NGRAM_PROGRESS_INTERVAL == 0 {
                if let Some(ref tx) = progress_tx {
                    let _ = tx.try_send(WorkerUpdate::NgramProgress {
                        worker_id,
                        ngram_count: count,
                    });
                }
            }
        }

        // Flush remaining counts
        if local_total > 0 {
            total_ngrams.fetch_add(local_total, Ordering::Relaxed);
        }
        if local_unique > 0 {
            unique_ngrams.fetch_add(local_unique, Ordering::Relaxed);
        }

        Ok(count)
    }
    .await;

    // Return worker ID to pool before returning result
    return_worker_id(worker_id_pool_tx, worker_id).await;

    match result {
        Ok(count) => {
            // Send "Finished" update
            if let Some(ref tx) = progress_tx {
                let _ = tx.try_send(WorkerUpdate::Finished {
                    worker_id,
                    order,
                    prefix: Arc::clone(&prefix),
                    ngram_count: count,
                });
            }
            PrefixOutcome::Success {
                prefix,
                ngram_count: count,
            }
        }
        Err(e) if attempt < MAX_RETRIES && is_retryable_error(&e) => {
            // Retryable error - return Deferred for caller to handle
            let next_backoff_ms = backoff_ms * 2;
            tracing::debug!(
                "Prefix '{}' (order {}) failed attempt {} with retryable error, deferring: {}",
                prefix, order, attempt + 1, e
            );
            if let Some(ref tx) = progress_tx {
                let _ = tx.try_send(WorkerUpdate::Deferred {
                    worker_id,
                    order,
                    prefix: Arc::clone(&prefix),
                    attempt: (attempt + 1) as u32,
                    delay_seconds: backoff_ms / 1000,
                    error: Arc::from(e.to_string()),
                });
            }
            PrefixOutcome::Deferred {
                url,
                prefix,
                order,
                attempt: attempt + 1,
                backoff_ms: next_backoff_ms,
                error: e,
            }
        }
        Err(e) => {
            // Non-retryable error or max retries exceeded
            tracing::warn!(
                "Prefix '{}' (order {}) failed permanently after {} attempts: {}",
                prefix, order, attempt + 1, e
            );
            PrefixOutcome::Failed {
                prefix,
                error: e,
                attempts: (attempt + 1) as u32,
            }
        }
    }
}

// ============================================================================
// Type definitions
// ============================================================================

/// Import progress information.
#[derive(Clone, Debug)]
pub struct ImportProgress {
    /// Current n-gram order being processed (1-5).
    pub current_order: u8,

    /// Current prefix file being processed.
    pub current_prefix: String,

    /// N-grams processed in current file.
    pub ngrams_in_file: u64,

    /// Total n-grams processed across all files.
    pub total_ngrams: u64,

    /// Files completed for current order.
    pub files_completed: u32,

    /// Total files for current order.
    pub total_files: u32,

    /// Bytes downloaded (HTTP mode).
    pub bytes_downloaded: u64,

    /// Processing rate (n-grams per second).
    pub ngrams_per_second: f64,

    /// Estimated time remaining.
    pub eta_seconds: Option<u64>,

    /// Current phase description.
    pub phase: ImportPhase,
}

/// Current import phase.
#[derive(Clone, Debug, PartialEq)]
pub enum ImportPhase {
    /// Downloading and parsing n-gram files.
    Importing,
    /// Computing MKN continuation counts (pass 1).
    MknPass1,
    /// Computing MKN continuation counts (pass 2).
    MknPass2,
    /// Finalizing and flushing to disk.
    Finalizing,
    /// Import complete.
    Complete,
}

/// Progress update sent from parallel download workers.
///
/// These updates are sent via a channel to allow real-time progress
/// display while downloads are in progress.
///
/// Uses `Arc<str>` for prefix fields to avoid string cloning overhead when
/// sending updates through the channel. Cloning Arc<str> is just a pointer
/// increment, not a full string copy.
#[derive(Clone, Debug)]
pub enum WorkerUpdate {
    /// Worker started downloading a prefix file.
    Started {
        /// Worker slot ID (0 to parallel_downloads-1).
        worker_id: usize,
        /// N-gram order being processed (1-5).
        order: u8,
        /// Prefix being downloaded (e.g., "th", "to").
        prefix: Arc<str>,
    },
    /// Worker finished downloading a prefix file.
    Finished {
        /// Worker slot ID.
        worker_id: usize,
        /// N-gram order that was processed (1-5).
        order: u8,
        /// Prefix that was downloaded.
        prefix: Arc<str>,
        /// Number of n-grams processed from this file.
        ngram_count: u64,
    },
    /// Periodic n-gram processing progress.
    NgramProgress {
        /// Worker slot ID.
        worker_id: usize,
        /// Number of n-grams processed so far.
        ngram_count: u64,
    },
    /// Worker encountered an error and is retrying.
    Retrying {
        /// Worker slot ID.
        worker_id: usize,
        /// Prefix being retried.
        prefix: Arc<str>,
        /// Current retry attempt (1-based).
        attempt: u32,
        /// Error message.
        error: Arc<str>,
    },
    /// Job deferred for later retry (worker freed to process other jobs).
    ///
    /// Unlike `Retrying`, this indicates the worker has released the job to a
    /// deferred queue and is immediately available to process other work.
    Deferred {
        /// Worker slot ID that deferred the job.
        worker_id: usize,
        /// N-gram order for this job.
        order: u8,
        /// Prefix being deferred.
        prefix: Arc<str>,
        /// Retry attempt number (1-based).
        attempt: u32,
        /// Seconds until retry.
        delay_seconds: u64,
        /// Error that triggered the retry.
        error: Arc<str>,
    },
    /// Worker exited (shutdown signal received or queue empty).
    Exited {
        /// Worker slot ID.
        worker_id: usize,
    },
}

/// Import statistics.
#[derive(Clone, Debug, Default)]
pub struct ImportStats {
    /// Total n-grams imported.
    pub total_ngrams: u64,

    /// N-grams per order.
    pub ngrams_by_order: [u64; 5],

    /// Unique n-grams (after aggregation).
    pub unique_ngrams: u64,

    /// Total bytes downloaded (HTTP mode).
    pub bytes_downloaded: u64,

    /// Files processed.
    pub files_processed: u32,

    /// Elapsed time in seconds.
    pub elapsed_seconds: u64,

    /// Average n-grams per second.
    pub ngrams_per_second: f64,
}

/// Errors that can occur during import.
#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Unsupported language.
    #[error("Unsupported language: {0}")]
    UnsupportedLanguage(String),

    /// Reader error.
    #[error("Reader error: {0}")]
    Reader(#[from] ReaderError),

    /// Checkpoint error.
    #[error("Checkpoint error: {0}")]
    Checkpoint(#[from] CheckpointError),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Import was interrupted.
    #[error("Import interrupted (checkpoint saved)")]
    Interrupted,

    /// Trie error.
    #[error("Trie error: {0}")]
    Trie(String),
}

/// Google Books N-gram importer.
///
/// Imports n-grams from Google Books dataset into a PersistentARTrie
/// with full MKN smoothing statistics.
///
/// # Concurrency
///
/// - Uses atomic counters for thread-safe progress tracking
/// - Parallel HTTP downloads for multiple prefix files
/// - Lock-free aggregation with streaming parser
///
/// # Checkpoint Support
///
/// - Saves checkpoint after each prefix file completes
/// - Handles graceful shutdown on SIGINT/SIGTERM
/// - Automatically resumes from checkpoint if present
///
/// # Example
///
/// ```ignore
/// use libgrammstein::sources::google_books::{GoogleBooksConfig, GoogleBooksImporter};
///
/// let config = GoogleBooksConfig::builder()
///     .language("en")
///     .orders(1..=5)
///     .output_path("english.artrie")
///     .build()?;
///
/// let mut importer = GoogleBooksImporter::resume_or_start(config).await?;
/// importer.import_http(|progress| {
///     println!("Order {}: {}/{} files",
///         progress.current_order,
///         progress.files_completed,
///         progress.total_files);
/// }).await?;
///
/// let stats = importer.finalize()?;
/// println!("Imported {} n-grams", stats.total_ngrams);
/// ```
pub struct GoogleBooksImporter {
    /// Import configuration.
    config: GoogleBooksConfig,

    /// Current checkpoint state.
    checkpoint: ImportCheckpoint,

    /// Path to checkpoint file.
    checkpoint_path: PathBuf,

    /// Atomic counter for total n-grams.
    total_ngrams: AtomicU64,

    /// Atomic counter for unique n-grams.
    unique_ngrams: AtomicU64,

    /// Atomic flag for interruption.
    interrupted: AtomicBool,

    /// Start time.
    start_time: Instant,

    /// N-gram storage backend.
    /// Can be single-trie (original behavior) or sharded storage.
    storage: Arc<NgramStorage>,

    /// Legacy trie field for checkpoint compatibility.
    /// Only used when storage is single-trie mode.
    /// TODO: Remove once checkpoint migration to storage is complete.
    trie: Arc<RwLock<DiskBackedCharTrieInner<u64>>>,
}

impl GoogleBooksImporter {
    /// Create a new importer with the given configuration.
    pub fn new(config: GoogleBooksConfig) -> Result<Self, ImportError> {
        // Validate language
        if !is_supported(&config.language) {
            return Err(ImportError::UnsupportedLanguage(config.language.clone()));
        }

        let checkpoint_path = config.output_path.with_extension("checkpoint.json");
        let output_path = &config.output_path;

        // Estimate n-gram count based on language and orders
        // For English 1-3 grams, expect ~500M n-grams; 1-5 grams ~2B
        let estimated_ngrams = estimate_ngram_count(&config);
        log::info!("Estimated n-gram count: {}", estimated_ngrams);

        // Create storage backend based on sharding configuration
        let storage = NgramStorage::create(&config, estimated_ngrams).map_err(|e| {
            ImportError::Trie(format!("Failed to create storage: {}", e))
        })?;

        // Log which storage mode is being used
        if storage.is_sharded() {
            log::info!("Using sharded storage for parallel writes");
        } else {
            log::info!("Using single-trie storage");
        }

        // For checkpoint compatibility, we also need a trie reference
        // In sharded mode, create a separate trie just for checkpoint metadata
        let trie = if let Some(inner_trie) = storage.as_single_trie() {
            Arc::clone(inner_trie)
        } else {
            // Sharded mode: create a checkpoint-only trie at output path
            let checkpoint_trie_path = config.output_path.with_extension("checkpoint.artrie");
            let trie = if checkpoint_trie_path.exists() {
                DiskBackedCharTrieInner::open(&checkpoint_trie_path).map_err(|e| {
                    ImportError::Trie(format!("Failed to open checkpoint trie: {}", e))
                })?
            } else {
                DiskBackedCharTrieInner::create(&checkpoint_trie_path).map_err(|e| {
                    ImportError::Trie(format!("Failed to create checkpoint trie: {}", e))
                })?
            };
            Arc::new(RwLock::new(trie))
        };

        Ok(Self {
            config,
            checkpoint: ImportCheckpoint::new(),
            checkpoint_path,
            total_ngrams: AtomicU64::new(0),
            unique_ngrams: AtomicU64::new(0),
            interrupted: AtomicBool::new(false),
            start_time: Instant::now(),
            storage: Arc::new(storage),
            trie,
        })
    }

    /// Resume from checkpoint if it exists, otherwise start fresh.
    ///
    /// This method checks for checkpoint data in the following order:
    /// 1. **Trie-based checkpoint** (preferred): More reliable as it's stored
    ///    atomically with n-gram data via WAL.
    /// 2. **JSON checkpoint** (fallback): For backwards compatibility with
    ///    older imports that only have JSON checkpoints.
    ///
    /// If a JSON checkpoint exists but no trie checkpoint, the JSON data
    /// is migrated to trie storage for future consistency.
    pub fn resume_or_start(config: GoogleBooksConfig) -> Result<Self, ImportError> {
        let checkpoint_path = config.output_path.with_extension("checkpoint.json");

        // First, create the importer to get access to the trie
        let mut importer = Self::new(config)?;

        // Try to load checkpoint from trie first (more reliable)
        let trie_checkpoint = {
            let trie = importer.trie.read();
            ImportCheckpoint::load_from_trie(&*trie)?
        };

        if let Some(checkpoint) = trie_checkpoint {
            log::info!(
                "Resuming from trie checkpoint: {} orders in progress, {} total prefixes completed",
                checkpoint.orders_in_progress().len(),
                checkpoint.total_completed_prefix_count()
            );

            importer.checkpoint = checkpoint;
            importer.total_ngrams.store(
                importer.checkpoint.stats.ngrams_processed,
                Ordering::Relaxed,
            );
            importer.unique_ngrams.store(
                importer.checkpoint.stats.unique_ngrams,
                Ordering::Relaxed,
            );

            // Clean up JSON checkpoint if it exists (we have trie data now)
            if ImportCheckpoint::exists(&checkpoint_path) {
                if let Err(e) = ImportCheckpoint::delete(&checkpoint_path) {
                    log::warn!("Failed to delete legacy JSON checkpoint: {}", e);
                } else {
                    log::info!("Deleted legacy JSON checkpoint (migrated to trie)");
                }
            }

            return Ok(importer);
        }

        // Fallback: try JSON checkpoint for backwards compatibility
        if ImportCheckpoint::exists(&checkpoint_path) {
            let checkpoint = ImportCheckpoint::load(&checkpoint_path)?;
            log::info!(
                "Resuming from JSON checkpoint: {} orders in progress, {} total prefixes completed",
                checkpoint.orders_in_progress().len(),
                checkpoint.total_completed_prefix_count()
            );

            importer.checkpoint = checkpoint;
            importer.total_ngrams.store(
                importer.checkpoint.stats.ngrams_processed,
                Ordering::Relaxed,
            );
            importer.unique_ngrams.store(
                importer.checkpoint.stats.unique_ngrams,
                Ordering::Relaxed,
            );

            // Migrate JSON checkpoint to trie for future consistency
            log::info!("Migrating JSON checkpoint to trie-based storage...");
            {
                let mut trie = importer.trie.write();
                importer.checkpoint.save_to_trie(&mut *trie).map_err(|e| {
                    ImportError::Trie(format!("Failed to migrate checkpoint to trie: {}", e))
                })?;
            }

            return Ok(importer);
        }

        // No checkpoint exists - fresh start
        Ok(importer)
    }

    /// Signal the importer to stop gracefully.
    pub fn interrupt(&self) {
        self.interrupted.store(true, Ordering::Release);
    }

    /// Check if import was interrupted.
    pub fn is_interrupted(&self) -> bool {
        self.interrupted.load(Ordering::Acquire)
    }

    /// Save current checkpoint.
    ///
    /// This persists both the trie data (via WAL checkpoint) and the import
    /// progress. The checkpoint data is stored in both:
    /// 1. The trie itself (with reserved key namespace for atomic consistency)
    /// 2. A JSON file (for backwards compatibility and easy inspection)
    ///
    /// The trie checkpoint truncates the WAL to prevent unbounded growth.
    pub fn save_checkpoint(&mut self) -> Result<(), ImportError> {
        // Update stats before saving
        self.checkpoint.stats.ngrams_processed = self.total_ngrams.load(Ordering::Relaxed);
        self.checkpoint.stats.unique_ngrams = self.unique_ngrams.load(Ordering::Relaxed);
        self.checkpoint.stats.elapsed_seconds = self.start_time.elapsed().as_secs();

        // Save checkpoint to trie FIRST (atomic with n-gram data)
        // This ensures consistency between data and progress tracking.
        {
            let mut trie = self.trie.write();

            // Save checkpoint data to trie
            self.checkpoint.save_to_trie(&mut *trie).map_err(|e| {
                ImportError::Trie(format!("Failed to save checkpoint to trie: {}", e))
            })?;

            // Checkpoint the trie (persists data and truncates WAL)
            trie.checkpoint().map_err(|e| {
                ImportError::Trie(format!("Failed to checkpoint trie: {}", e))
            })?;
        }

        log::debug!("Checkpoint saved: {}", self.checkpoint.progress_summary());
        Ok(())
    }

    /// Delete checkpoint file and trie-based checkpoint data (call after successful completion).
    pub fn cleanup_checkpoint(&mut self) -> Result<(), ImportError> {
        // Delete JSON checkpoint
        ImportCheckpoint::delete(&self.checkpoint_path)?;

        // Delete trie-based checkpoint data
        {
            let mut trie = self.trie.write();
            ImportCheckpoint::delete_from_trie(&mut *trie).map_err(|e| {
                ImportError::Trie(format!("Failed to delete checkpoint from trie: {}", e))
            })?;
        }

        Ok(())
    }

    /// Import from local gzip files.
    ///
    /// # Arguments
    ///
    /// * `file_dir` - Directory containing n-gram files
    /// * `progress` - Progress callback
    pub fn import_files<F>(
        &mut self,
        file_dir: &Path,
        mut progress: F,
    ) -> Result<ImportStats, ImportError>
    where
        F: FnMut(ImportProgress),
    {
        for order in self.config.orders.clone() {
            if self.checkpoint.is_order_complete(order) {
                log::info!("Skipping order {} (already completed)", order);
                continue;
            }

            let prefixes = get_prefixes(order);
            let total_files = prefixes.len() as u32;

            // Get the corpus ID for filename construction
            let metadata = get_metadata(&self.config.language)
                .ok_or_else(|| ImportError::UnsupportedLanguage(self.config.language.clone()))?;

            for (idx, prefix) in prefixes.iter().enumerate() {
                if self.is_interrupted() {
                    self.save_checkpoint()?;
                    return Err(ImportError::Interrupted);
                }

                if !self.checkpoint.needs_prefix(order, prefix) {
                    continue;
                }

                // Build file path using corpus_id (e.g., "eng" for English)
                let filename = format!(
                    "googlebooks-{}-all-{}gram-{}-{}.gz",
                    metadata.corpus_id, order, "20200217", prefix
                );
                let file_path = file_dir.join(&filename);

                if !file_path.exists() {
                    log::warn!("File not found: {:?}", file_path);
                    continue;
                }

                // Process file
                let ngrams_in_file = self.process_file(&file_path)?;

                self.checkpoint.complete_prefix(order, prefix);
                self.checkpoint.stats.ngrams_by_order[(order - 1) as usize] += ngrams_in_file;

                // Report progress
                progress(ImportProgress {
                    current_order: order,
                    current_prefix: prefix.clone(),
                    ngrams_in_file,
                    total_ngrams: self.total_ngrams.load(Ordering::Relaxed),
                    files_completed: idx as u32 + 1,
                    total_files,
                    bytes_downloaded: 0,
                    ngrams_per_second: self.calculate_rate(),
                    eta_seconds: self.estimate_eta(idx as u32 + 1, total_files),
                    phase: ImportPhase::Importing,
                });

                // Save checkpoint periodically
                if (idx + 1) % 10 == 0 {
                    self.save_checkpoint()?;
                }
            }

            self.checkpoint.complete_order(order);
            self.save_checkpoint()?;
        }

        // Finalize: compute MKN stats, sync storage, and return final stats
        self.finalize()
    }

    /// Import from HTTP (streaming from Google's servers).
    ///
    /// # Arguments
    ///
    /// * `progress` - Progress callback
    ///
    /// # Parallelism
    ///
    /// Downloads and processes up to `parallel_downloads` (default: 4) prefix files
    /// concurrently using `futures::stream::buffer_unordered`. This provides ~Nx
    /// throughput improvement for network-bound imports.
    ///
    /// This is a convenience wrapper around `import_http_with_progress` that doesn't
    /// send worker updates. Use `import_http_with_progress` if you need real-time
    /// per-worker progress updates for display purposes.
    #[cfg(feature = "google-books")]
    pub async fn import_http<F>(&mut self, progress: F) -> Result<ImportStats, ImportError>
    where
        F: FnMut(ImportProgress),
    {
        self.import_http_with_progress(progress, None).await
    }

    /// Downloads and processes prefix files with optional real-time worker updates.
    ///
    /// This method provides the same functionality as `import_http`, but additionally
    /// accepts an optional channel for receiving real-time progress updates from
    /// parallel download workers. This enables rich progress display showing what
    /// each worker is currently downloading.
    ///
    /// # Arguments
    ///
    /// * `progress` - Callback invoked after each file completes with overall progress
    /// * `worker_updates` - Optional channel for real-time per-worker status updates
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tokio::sync::mpsc;
    ///
    /// // Use bounded channel with backpressure
    /// let (tx, mut rx) = mpsc::channel(1024);
    ///
    /// // Spawn task to handle worker updates
    /// tokio::spawn(async move {
    ///     while let Some(update) = rx.recv().await {
    ///         match update {
    ///             WorkerUpdate::Started { worker_id, prefix } => {
    ///                 println!("[{}] Downloading: {}", worker_id, prefix);
    ///             }
    ///             WorkerUpdate::Finished { worker_id, prefix, ngram_count } => {
    ///                 println!("[{}] Done: {} ({} n-grams)", worker_id, prefix, ngram_count);
    ///             }
    ///             WorkerUpdate::Retrying { worker_id, prefix, attempt, error } => {
    ///                 println!("[{}] Retry {}: {} - {}", worker_id, attempt, prefix, error);
    ///             }
    ///         }
    ///     }
    /// });
    ///
    /// importer.import_http_with_progress(|progress| { /* ... */ }, Some(tx)).await?;
    /// ```
    #[cfg(feature = "google-books")]
    pub async fn import_http_with_progress<F>(
        &mut self,
        mut progress: F,
        worker_updates: Option<tokio::sync::mpsc::Sender<WorkerUpdate>>,
    ) -> Result<ImportStats, ImportError>
    where
        F: FnMut(ImportProgress),
    {
        use futures::stream::{self, StreamExt};

        let parallel_downloads = self.config.parallel_downloads;

        for order in self.config.orders.clone() {
            if self.checkpoint.is_order_complete(order) {
                log::info!("Skipping order {} (already completed)", order);
                continue;
            }

            let prefixes = get_prefixes(order);
            let total_files = prefixes.len() as u32;

            // Filter to only prefixes that need processing
            let pending_prefixes: Vec<String> = prefixes
                .iter()
                .filter(|p| self.checkpoint.needs_prefix(order, p))
                .cloned()
                .collect();

            if pending_prefixes.is_empty() {
                log::info!("Order {} already complete", order);
                self.checkpoint.complete_order(order);
                continue;
            }

            log::info!(
                "Processing order {} with {} pending files ({} parallel)",
                order,
                pending_prefixes.len(),
                self.config.parallel_downloads
            );

            // Clone Arc references for parallel processing
            let storage = Arc::clone(&self.storage);

            // Create new atomic counters for this batch (we'll sync back after)
            let total_ngrams = Arc::new(AtomicU64::new(self.total_ngrams.load(Ordering::Relaxed)));
            let unique_ngrams = Arc::new(AtomicU64::new(self.unique_ngrams.load(Ordering::Relaxed)));

            let config = self.config.clone();
            let language = self.config.language.clone();

            // Create worker ID pool for dynamic assignment.
            // Worker IDs are claimed when a future starts and returned when it finishes,
            // ensuring each concurrent worker has a unique ID for display purposes.
            let (worker_id_pool_tx, worker_id_pool_rx) =
                tokio::sync::mpsc::channel::<usize>(parallel_downloads);
            // Pre-populate pool with available worker IDs
            for id in 0..parallel_downloads {
                worker_id_pool_tx
                    .send(id)
                    .await
                    .expect("Failed to populate worker ID pool");
            }
            // Wrap receiver in Arc<Mutex> for sharing across futures
            let worker_id_pool_rx = Arc::new(tokio::sync::Mutex::new(worker_id_pool_rx));

            // Track deferred items for retry after initial pass
            let mut deferred_items: Vec<(Arc<str>, Arc<str>, u8, u8, u64)> = Vec::new();
            let mut failed_prefixes: Vec<(Arc<str>, ImportError, u32)> = Vec::new();

            // Create futures for each prefix - worker IDs will be claimed dynamically
            // Initial attempt uses attempt=0 and INITIAL_BACKOFF_MS
            let futures: Vec<_> = pending_prefixes
                .into_iter()
                .filter_map(|prefix| {
                    get_file_url(&language, order, &prefix).map(|url| {
                        let url: Arc<str> = Arc::from(url);
                        let prefix: Arc<str> = Arc::from(prefix);
                        process_prefix_file(
                            worker_id_pool_tx.clone(),
                            Arc::clone(&worker_id_pool_rx),
                            url,
                            prefix,
                            order,
                            0,                    // First attempt
                            INITIAL_BACKOFF_MS,   // Initial backoff
                            config.clone(),
                            Arc::clone(&storage),
                            Arc::clone(&total_ngrams),
                            Arc::clone(&unique_ngrams),
                            worker_updates.clone(),
                        )
                    })
                })
                .collect();

            let pending_count = futures.len() as u32;

            // Process results as they arrive (streaming) to avoid OOM from buffering
            // Note: Previously used .collect().await which buffered all results (~4GB for 2-grams)
            let mut result_stream = stream::iter(futures)
                .buffer_unordered(self.config.parallel_downloads);

            let already_completed = total_files - pending_count;
            let mut completed_in_order = 0u32;

            while let Some(outcome) = result_stream.next().await {
                // Sync atomic counters periodically (not just at end)
                self.total_ngrams
                    .store(total_ngrams.load(Ordering::Relaxed), Ordering::Relaxed);
                self.unique_ngrams
                    .store(unique_ngrams.load(Ordering::Relaxed), Ordering::Relaxed);

                if self.is_interrupted() {
                    self.save_checkpoint()?;
                    return Err(ImportError::Interrupted);
                }

                match outcome {
                    PrefixOutcome::Success { prefix, ngram_count } => {
                        self.checkpoint.complete_prefix(order, &prefix);
                        self.checkpoint.stats.ngrams_by_order[(order - 1) as usize] += ngram_count;
                        completed_in_order += 1;

                        // Report progress (convert Arc<str> to String for public API)
                        progress(ImportProgress {
                            current_order: order,
                            current_prefix: prefix.to_string(),
                            ngrams_in_file: ngram_count,
                            total_ngrams: self.total_ngrams.load(Ordering::Relaxed),
                            files_completed: already_completed + completed_in_order,
                            total_files,
                            bytes_downloaded: self.checkpoint.stats.bytes_downloaded,
                            ngrams_per_second: self.calculate_rate(),
                            eta_seconds: self.estimate_eta(already_completed + completed_in_order, total_files),
                            phase: ImportPhase::Importing,
                        });
                    }
                    PrefixOutcome::Deferred { url, prefix, order: o, attempt, backoff_ms, error: _ } => {
                        // Collect deferred item for retry later (Arc<str> is cheap to store)
                        deferred_items.push((url, prefix, o, attempt, backoff_ms));
                    }
                    PrefixOutcome::Failed { prefix, error, attempts } => {
                        // Collect permanent failures (will be reported at end)
                        failed_prefixes.push((prefix, error, attempts));
                    }
                }

                // Save checkpoint periodically
                if completed_in_order % 10 == 0 {
                    self.save_checkpoint()?;
                }
            }

            // Process deferred items in additional passes until all complete or fail
            while !deferred_items.is_empty() {
                // Wait for the minimum backoff time before retry pass
                let min_backoff = deferred_items.iter().map(|(_, _, _, _, b)| *b).min().unwrap_or(1000);
                tracing::info!(
                    "Processing {} deferred prefixes for order {} after {}ms delay",
                    deferred_items.len(), order, min_backoff
                );
                tokio::time::sleep(Duration::from_millis(min_backoff)).await;

                if self.is_interrupted() {
                    self.save_checkpoint()?;
                    return Err(ImportError::Interrupted);
                }

                // Create futures for deferred items
                let retry_futures: Vec<_> = deferred_items
                    .drain(..)
                    .map(|(url, prefix, o, attempt, backoff_ms)| {
                        process_prefix_file(
                            worker_id_pool_tx.clone(),
                            Arc::clone(&worker_id_pool_rx),
                            url,
                            prefix,
                            o,
                            attempt,
                            backoff_ms,
                            config.clone(),
                            Arc::clone(&storage),
                            Arc::clone(&total_ngrams),
                            Arc::clone(&unique_ngrams),
                            worker_updates.clone(),
                        )
                    })
                    .collect();

                let mut retry_stream = stream::iter(retry_futures)
                    .buffer_unordered(self.config.parallel_downloads);

                while let Some(outcome) = retry_stream.next().await {
                    self.total_ngrams
                        .store(total_ngrams.load(Ordering::Relaxed), Ordering::Relaxed);
                    self.unique_ngrams
                        .store(unique_ngrams.load(Ordering::Relaxed), Ordering::Relaxed);

                    if self.is_interrupted() {
                        self.save_checkpoint()?;
                        return Err(ImportError::Interrupted);
                    }

                    match outcome {
                        PrefixOutcome::Success { prefix, ngram_count } => {
                            self.checkpoint.complete_prefix(order, &prefix);
                            self.checkpoint.stats.ngrams_by_order[(order - 1) as usize] += ngram_count;
                            completed_in_order += 1;

                            // Convert Arc<str> to String for public API
                            progress(ImportProgress {
                                current_order: order,
                                current_prefix: prefix.to_string(),
                                ngrams_in_file: ngram_count,
                                total_ngrams: self.total_ngrams.load(Ordering::Relaxed),
                                files_completed: already_completed + completed_in_order,
                                total_files,
                                bytes_downloaded: self.checkpoint.stats.bytes_downloaded,
                                ngrams_per_second: self.calculate_rate(),
                                eta_seconds: self.estimate_eta(already_completed + completed_in_order, total_files),
                                phase: ImportPhase::Importing,
                            });
                        }
                        PrefixOutcome::Deferred { url, prefix, order: o, attempt, backoff_ms, error: _ } => {
                            // Re-defer for another pass (Arc<str> is cheap to clone)
                            deferred_items.push((url, prefix, o, attempt, backoff_ms));
                        }
                        PrefixOutcome::Failed { prefix, error, attempts } => {
                            failed_prefixes.push((prefix, error, attempts));
                        }
                    }
                }
            }

            // Report any permanent failures (but don't fail the entire import)
            if !failed_prefixes.is_empty() {
                tracing::warn!(
                    "Order {} completed with {} failed prefixes:",
                    order, failed_prefixes.len()
                );
                for (prefix, error, attempts) in &failed_prefixes {
                    tracing::warn!("  {} (after {} attempts): {}", prefix, attempts, error);
                }
            }

            // Final sync of atomic counters
            self.total_ngrams
                .store(total_ngrams.load(Ordering::Relaxed), Ordering::Relaxed);
            self.unique_ngrams
                .store(unique_ngrams.load(Ordering::Relaxed), Ordering::Relaxed);

            self.checkpoint.complete_order(order);
            self.save_checkpoint()?;
        }

        // Finalize: compute MKN stats, sync storage, and return final stats
        self.finalize()
    }

    /// Import from HTTP with reactive event/command channels.
    ///
    /// This method provides a clean reactive interface for UIs that want to subscribe
    /// to import progress events without coupling to any specific UI framework.
    ///
    /// # Architecture
    ///
    /// Events flow down (importer → subscribers):
    /// - `ImportEvent::OrderStarted` - Beginning to process an n-gram order
    /// - `ImportEvent::WorkerStarted` - Worker began downloading a prefix file
    /// - `ImportEvent::WorkerProgress` - Periodic download progress
    /// - `ImportEvent::WorkerFinished` - Worker completed a prefix file
    /// - `ImportEvent::WorkerRetrying` - Worker retrying after transient error
    /// - `ImportEvent::StatsSnapshot` - Periodic statistics update
    /// - `ImportEvent::CheckpointSaved` - Checkpoint was saved
    /// - `ImportEvent::OrderCompleted` - Order completed
    /// - `ImportEvent::ImportCompleted` - All orders completed
    ///
    /// Commands flow up (subscribers → importer):
    /// - `ImportCommand::Pause` - Pause all workers (graceful)
    /// - `ImportCommand::Resume` - Resume paused workers
    /// - `ImportCommand::Cancel` - Cancel import (save checkpoint first)
    /// - `ImportCommand::ForceQuit` - Force quit without saving checkpoint
    /// - `ImportCommand::SetParallelism` - Adjust worker count at runtime
    ///
    /// # Arguments
    ///
    /// * `event_tx` - Broadcast sender for emitting domain events
    /// * `command_rx` - Receiver for control commands from UI
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tokio::sync::{broadcast, mpsc};
    ///
    /// let (event_tx, _) = broadcast::channel::<ImportEvent>(1024);
    /// let (command_tx, command_rx) = mpsc::channel::<ImportCommand>(16);
    ///
    /// // Subscribe to events from multiple consumers
    /// let tui_rx = event_tx.subscribe();
    /// let log_rx = event_tx.subscribe();
    ///
    /// // Run import
    /// let stats = importer.import_http_reactive(event_tx, command_rx).await?;
    /// ```
    #[cfg(feature = "google-books")]
    pub async fn import_http_reactive(
        &mut self,
        event_tx: tokio::sync::broadcast::Sender<ImportEvent>,
        mut command_rx: tokio::sync::mpsc::Receiver<ImportCommand>,
    ) -> Result<ImportStats, ImportError> {
        use futures::stream::{self, StreamExt};
        use std::time::{Duration, Instant};

        let parallel_downloads = self.config.parallel_downloads;

        // Atomics for pause/cancel control
        let paused = Arc::new(AtomicBool::new(false));
        let cancelled = Arc::new(AtomicBool::new(false));
        let force_quit = Arc::new(AtomicBool::new(false));
        let current_parallelism = Arc::new(std::sync::atomic::AtomicUsize::new(parallel_downloads));

        // Channel to notify main loop immediately when parallelism changes
        // This allows spawning/stopping workers without waiting for file completion
        let (parallelism_change_tx, parallelism_change_rx) =
            tokio::sync::mpsc::channel::<usize>(16);
        let parallelism_change_rx = Arc::new(tokio::sync::Mutex::new(parallelism_change_rx));

        // Spawn command handler task and store handle for cleanup
        let paused_clone = paused.clone();
        let cancelled_clone = cancelled.clone();
        let force_quit_clone = force_quit.clone();
        let parallelism_clone = current_parallelism.clone();
        let event_tx_clone = event_tx.clone();
        let command_handler = tokio::spawn(async move {
            while let Some(cmd) = command_rx.recv().await {
                match cmd {
                    ImportCommand::Pause => {
                        paused_clone.store(true, Ordering::SeqCst);
                        let _ = event_tx_clone.send(ImportEvent::ImportPaused);
                    }
                    ImportCommand::Resume => {
                        paused_clone.store(false, Ordering::SeqCst);
                        let _ = event_tx_clone.send(ImportEvent::ImportResumed);
                    }
                    ImportCommand::Cancel => {
                        cancelled_clone.store(true, Ordering::SeqCst);
                    }
                    ImportCommand::ForceQuit => {
                        force_quit_clone.store(true, Ordering::SeqCst);
                    }
                    ImportCommand::SetParallelism(n) => {
                        parallelism_clone.store(n, Ordering::SeqCst);
                        // Notify main loop immediately so it can spawn/stop workers
                        let _ = parallelism_change_tx.send(n).await;
                        let _ = event_tx_clone.send(ImportEvent::Log {
                            level: LogLevel::Info,
                            message: format!("Parallelism adjusted to {}", n),
                        });
                    }
                }
            }
        });

        // ======================================================================
        // UNIFIED JOB QUEUE: Collect all orders' jobs for overlapping processing
        // ======================================================================
        //
        // Instead of processing orders sequentially, we create a unified job queue
        // containing jobs from ALL orders. Workers pull jobs from this queue,
        // enabling them to start processing 2-grams before all 1-grams are done.
        //
        // Per-order progress is tracked separately to maintain checkpoint integrity.

        // Track per-order progress
        let mut jobs_per_order: std::collections::HashMap<u8, u64> =
            std::collections::HashMap::new();
        let mut order_files_completed: std::collections::HashMap<u8, u64> =
            std::collections::HashMap::new();
        let mut order_total_files: std::collections::HashMap<u8, u64> =
            std::collections::HashMap::new();
        let mut order_start_times: std::collections::HashMap<u8, Instant> =
            std::collections::HashMap::new();

        // Pre-calculate job counts and emit OrderStarted events
        let language = self.config.language.clone();
        for order in self.config.orders.clone() {
            if self.checkpoint.is_order_complete(order) {
                log::info!("Skipping order {} (already completed)", order);
                continue;
            }

            let prefixes = get_prefixes(order);
            let total_files = prefixes.len() as u64;
            order_total_files.insert(order, total_files);

            // Filter to only prefixes that need processing
            let pending_count = prefixes
                .iter()
                .filter(|p| self.checkpoint.needs_prefix(order, p))
                .count() as u64;

            if pending_count == 0 {
                log::info!("Order {} already complete", order);
                self.checkpoint.complete_order(order);
                continue;
            }

            let already_completed = total_files - pending_count;
            order_files_completed.insert(order, already_completed);
            jobs_per_order.insert(order, pending_count);
            order_start_times.insert(order, Instant::now());

            // Emit OrderStarted event
            let _ = event_tx.send(ImportEvent::OrderStarted {
                order,
                total_files,
            });

            log::info!(
                "Queued {} pending files for order {} ({} already complete)",
                pending_count,
                order,
                already_completed
            );
        }

        // Check if all orders are already complete
        if jobs_per_order.is_empty() {
            log::info!("All orders already complete");
            command_handler.abort();
            let _ = command_handler.await;
            return self.build_stats();
        }

        let total_pending: u64 = jobs_per_order.values().sum();
        log::info!(
            "Starting overlapping import with {} total pending files across {} orders ({} parallel)",
            total_pending,
            jobs_per_order.len(),
            parallel_downloads
        );

        // Clone Arc references for parallel processing
        let storage = Arc::clone(&self.storage);

        // Create atomic counters for shared state
        let total_ngrams = Arc::new(AtomicU64::new(self.total_ngrams.load(Ordering::Relaxed)));
        let unique_ngrams = Arc::new(AtomicU64::new(self.unique_ngrams.load(Ordering::Relaxed)));

        let config = self.config.clone();

        // Create internal worker update channel that we'll convert to domain events
        // Using bounded channel with backpressure to prevent memory growth when TUI lags
        let (worker_tx, mut worker_rx) = tokio::sync::mpsc::channel::<WorkerUpdate>(1024);

        // Spawn task to convert WorkerUpdate to ImportEvent
        // Note: Converting Arc<str> to String for public API (ImportEvent uses String)
        let event_tx_worker = event_tx.clone();
        let worker_converter = tokio::spawn(async move {
            while let Some(update) = worker_rx.recv().await {
                let event = match update {
                    WorkerUpdate::Started { worker_id, order, prefix } => {
                        ImportEvent::WorkerStarted { worker_id, order, prefix: prefix.to_string() }
                    }
                    WorkerUpdate::Finished { worker_id, order, prefix, ngram_count } => {
                        ImportEvent::WorkerFinished {
                            worker_id,
                            order,
                            prefix: prefix.to_string(),
                            ngram_count,
                            duration: Duration::ZERO, // Duration tracked at higher level
                        }
                    }
                    WorkerUpdate::NgramProgress { worker_id, ngram_count } => {
                        ImportEvent::WorkerNgramProgress { worker_id, ngram_count }
                    }
                    WorkerUpdate::Retrying { worker_id, prefix, attempt, error } => {
                        ImportEvent::WorkerRetrying {
                            worker_id,
                            prefix: prefix.to_string(),
                            attempt,
                            max_attempts: 5,
                            error: error.to_string(),
                        }
                    }
                    WorkerUpdate::Deferred { worker_id, order: _, prefix, attempt, delay_seconds: _, error } => {
                        // Deferred is similar to Retrying - map to same event
                        ImportEvent::WorkerRetrying {
                            worker_id,
                            prefix: prefix.to_string(),
                            attempt,
                            max_attempts: MAX_RETRIES as u32,
                            error: error.to_string(),
                        }
                    }
                    WorkerUpdate::Exited { worker_id } => {
                        ImportEvent::WorkerExited { worker_id }
                    }
                };
                let _ = event_tx_worker.send(event);
            }
        });

        // Create unified job queue for worker pool (all orders)
        // Add extra capacity for failed prefix retries AND in-flight requeued jobs
        // Workers may requeue failed jobs, so we need space for:
        // - Initial jobs + failed retries
        // - Additional capacity for requeued jobs (workers * max_retries)
        let failed_retry_count: usize = self.config.orders.clone()
            .map(|o| self.checkpoint.failed_prefix_count(o))
            .sum();
        let requeue_capacity = parallel_downloads * MAX_RETRIES as usize;
        let (job_tx, job_rx) = tokio::sync::mpsc::channel::<Job>(
            total_pending as usize + failed_retry_count + requeue_capacity + 1
        );
        let job_rx = Arc::new(tokio::sync::Mutex::new(job_rx));

        // Populate job queue with jobs from ALL orders (in priority order: 1-grams first)
        // Jobs are sorted by order so workers process lower orders first
        for order in self.config.orders.clone() {
            // Check for failed prefixes from previous run to retry
            let failed_prefixes = self.checkpoint.failed_prefixes(order);
            if !failed_prefixes.is_empty() {
                log::info!(
                    "Retrying {} previously failed prefixes for order {}: {:?}",
                    failed_prefixes.len(),
                    order,
                    &failed_prefixes
                );

                // Emit event for TUI
                let _ = event_tx.send(ImportEvent::RetryingFailedPrefixes {
                    order,
                    count: failed_prefixes.len(),
                    prefixes: failed_prefixes.clone(),
                });

                // Queue the failed prefixes for retry
                for prefix in failed_prefixes.iter() {
                    // Clear from failed list so it can be retried
                    self.checkpoint.clear_failed(order, prefix);

                    if let Some(url) = get_file_url(&language, order, prefix) {
                        let _ = job_tx.send(Job::new(url, prefix.clone(), order)).await;
                    }
                }
            }

            if !jobs_per_order.contains_key(&order) {
                continue; // Already complete or no pending jobs
            }

            let prefixes = get_prefixes(order);
            for prefix in prefixes.iter() {
                if self.checkpoint.needs_prefix(order, prefix) {
                    if let Some(url) = get_file_url(&language, order, prefix) {
                        let _ = job_tx.send(Job::new(url, prefix.clone(), order)).await;
                    }
                }
            }
        }
        // NOTE: We keep job_tx alive - workers need it to requeue failed jobs.
        // Workers will detect queue exhaustion via the all-deferred blocking logic.

        // Track queue size for all-deferred detection
        let queue_size = Arc::new(AtomicUsize::new(total_pending as usize + failed_retry_count));

        // Create shared state for workers
        let shared_state = Arc::new(WorkerSharedState {
            config: config.clone(),
            storage: Arc::clone(&storage),
            total_ngrams: Arc::clone(&total_ngrams),
            unique_ngrams: Arc::clone(&unique_ngrams),
            progress_tx: worker_tx.clone(),
            paused: Arc::clone(&paused),
            queue_size: Arc::clone(&queue_size),
        });

        // Create result channel for receiving job completions
        // Results always include order and prefix, plus success/failure outcome.
        // We keep result_tx alive for dynamic worker spawning.
        let (result_tx, mut result_rx) =
            tokio::sync::mpsc::channel::<JobResult>(parallel_downloads * 2);

        // Create worker exit notification channel - workers send their ID when exiting
        // This allows the main loop to track active workers and detect when all have exited
        let (worker_exit_tx, mut worker_exit_rx) =
            tokio::sync::mpsc::channel::<usize>(parallel_downloads * 2);

        // Per-worker shutdown channels for individual control
        // Each worker gets its own shutdown signal so we can stop specific workers
        // when parallelism decreases
        let mut worker_shutdown_txs: std::collections::HashMap<
            usize,
            tokio::sync::watch::Sender<bool>,
        > = std::collections::HashMap::new();
        let mut worker_handles: std::collections::HashMap<usize, tokio::task::JoinHandle<()>> =
            std::collections::HashMap::new();

        // Spawn initial workers, each with their own shutdown channel
        for worker_id in 0..parallel_downloads {
            let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
            let handle = tokio::spawn(worker_task(
                worker_id,
                Arc::clone(&job_rx),
                job_tx.clone(),
                shutdown_rx,
                Arc::clone(&shared_state),
                result_tx.clone(),
                worker_exit_tx.clone(),
            ));
            worker_handles.insert(worker_id, handle);
            worker_shutdown_txs.insert(worker_id, shutdown_tx);
        }
        // Keep job_tx alive for dynamic worker spawning - workers need it for requeue

        // Track number of active workers to detect when all have exited
        let mut active_workers = parallel_downloads;

        // Track next worker ID for dynamic spawning
        let mut next_worker_id = parallel_downloads;

        // Keep result_tx alive for dynamic worker spawning (will drop after loop)
        drop(worker_tx);

        // Calculate total already completed across all orders
        let total_already_completed: u64 = order_files_completed.values().sum();
        let files_completed = Arc::new(AtomicU64::new(total_already_completed));
        let import_start = Instant::now();

        // Total files across all orders for stats display
        let grand_total_files: u64 = order_total_files.values().sum();

        // Spawn periodic stats emitter task (1 second interval)
        // This ensures the TUI receives real-time updates even when no files are completing
        let stats_event_tx = event_tx.clone();
        let stats_total_ngrams = Arc::clone(&total_ngrams);
        let stats_unique_ngrams = Arc::clone(&unique_ngrams);
        let stats_files_completed = Arc::clone(&files_completed);
        let stats_start_time = self.start_time;
        let stats_cancelled = Arc::clone(&cancelled);
        let stats_force_quit = Arc::clone(&force_quit);
        let stats_total_files = grand_total_files;

        let stats_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));

            loop {
                interval.tick().await;

                // Check for cancellation
                if stats_cancelled.load(Ordering::Relaxed)
                    || stats_force_quit.load(Ordering::Relaxed)
                {
                    break;
                }

                let total = stats_total_ngrams.load(Ordering::Relaxed);
                let unique = stats_unique_ngrams.load(Ordering::Relaxed);
                let completed = stats_files_completed.load(Ordering::Relaxed);
                let elapsed = stats_start_time.elapsed();

                // Calculate rate
                let ngrams_per_second = if elapsed.as_secs_f64() > 0.0 {
                    total as f64 / elapsed.as_secs_f64()
                } else {
                    0.0
                };

                let _ = stats_event_tx.send(ImportEvent::StatsSnapshot {
                    files_completed: completed,
                    total_files: stats_total_files,
                    total_ngrams: total,
                    unique_ngrams: unique,
                    ngrams_per_second,
                    elapsed,
                });
            }
        });

        // Process results from workers using tokio::select! to handle parallelism
        // changes immediately without waiting for file completion
        let mut results_received = 0u64;

        // Helper closure to signal all workers to shut down
        let signal_all_shutdown = |shutdown_txs: &std::collections::HashMap<
            usize,
            tokio::sync::watch::Sender<bool>,
        >| {
            for tx in shutdown_txs.values() {
                let _ = tx.send(true);
            }
        };

        // Helper closure to handle parallelism changes
        // Returns the number of new workers spawned (for active_workers tracking)
        let handle_parallelism_change = |target: usize,
                                         worker_handles: &mut std::collections::HashMap<
            usize,
            tokio::task::JoinHandle<()>,
        >,
                                         worker_shutdown_txs: &mut std::collections::HashMap<
            usize,
            tokio::sync::watch::Sender<bool>,
        >,
                                         next_worker_id: &mut usize,
                                         job_rx: &Arc<tokio::sync::Mutex<tokio::sync::mpsc::Receiver<Job>>>,
                                         job_tx: &tokio::sync::mpsc::Sender<Job>,
                                         shared_state: &Arc<WorkerSharedState>,
                                         result_tx: &tokio::sync::mpsc::Sender<JobResult>,
                                         worker_exit_tx: &tokio::sync::mpsc::Sender<usize>,
                                         event_tx: &tokio::sync::broadcast::Sender<ImportEvent>|
         -> usize {
            let current_count = worker_handles.len();
            let mut spawned = 0usize;

            if target > current_count {
                // Spawn additional workers immediately
                for _ in 0..(target - current_count) {
                    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
                    let handle = tokio::spawn(worker_task(
                        *next_worker_id,
                        Arc::clone(job_rx),
                        job_tx.clone(),
                        shutdown_rx,
                        Arc::clone(shared_state),
                        result_tx.clone(),
                        worker_exit_tx.clone(),
                    ));
                    worker_handles.insert(*next_worker_id, handle);
                    worker_shutdown_txs.insert(*next_worker_id, shutdown_tx);
                    let _ = event_tx.send(ImportEvent::Log {
                        level: LogLevel::Info,
                        message: format!("Spawned worker {}", *next_worker_id),
                    });
                    *next_worker_id += 1;
                    spawned += 1;
                }
            } else if target < current_count {
                // Signal excess workers to shut down (highest IDs first)
                let workers_to_remove = current_count - target;
                let mut ids_to_remove: Vec<_> = worker_handles.keys().copied().collect();
                ids_to_remove.sort_by(|a, b| b.cmp(a)); // Descending order

                for &worker_id in ids_to_remove.iter().take(workers_to_remove) {
                    if let Some(shutdown_tx) = worker_shutdown_txs.get(&worker_id) {
                        let _ = shutdown_tx.send(true);
                        let _ = event_tx.send(ImportEvent::Log {
                            level: LogLevel::Info,
                            message: format!(
                                "Signaling worker {} to stop after current job",
                                worker_id
                            ),
                        });
                    }
                }
            }

            spawned
        };

        while results_received < total_pending {
            // Check if all workers have exited before any results can arrive
            // This prevents infinite hang when workers exit without producing all results
            if active_workers == 0 {
                log::error!(
                    "All workers exited with {} results missing (received {}/{})",
                    total_pending - results_received,
                    results_received,
                    total_pending
                );
                let _ = event_tx.send(ImportEvent::Log {
                    level: LogLevel::Error,
                    message: format!(
                        "All workers exited with {} results missing",
                        total_pending - results_received
                    ),
                });
                // Save checkpoint before breaking to preserve progress
                if let Err(e) = self.save_checkpoint() {
                    log::error!("Failed to save checkpoint on worker exit: {}", e);
                } else {
                    let _ = event_tx.send(ImportEvent::CheckpointSaved {
                        prefix: "emergency".to_string(),
                    });
                }
                break;
            }

            // Use tokio::select! to race between result, parallelism change, and worker exit
            let mut parallelism_rx = parallelism_change_rx.lock().await;

            tokio::select! {
                biased;

                // Check for cancellation first (highest priority)
                _ = async {}, if force_quit.load(Ordering::SeqCst) => {
                    drop(parallelism_rx);
                    signal_all_shutdown(&worker_shutdown_txs);
                    let _ = event_tx.send(ImportEvent::ImportCancelled);
                    return Err(ImportError::Interrupted);
                }

                _ = async {}, if cancelled.load(Ordering::SeqCst) => {
                    drop(parallelism_rx);
                    signal_all_shutdown(&worker_shutdown_txs);
                    self.save_checkpoint()?;
                    let _ = event_tx.send(ImportEvent::CheckpointSaved {
                        prefix: "all".to_string(),
                    });
                    let _ = event_tx.send(ImportEvent::ImportCancelled);
                    return Err(ImportError::Interrupted);
                }

                // Check for worker exits (high priority - track active workers)
                Some(exited_worker_id) = worker_exit_rx.recv() => {
                    drop(parallelism_rx);
                    active_workers = active_workers.saturating_sub(1);
                    worker_handles.remove(&exited_worker_id);
                    worker_shutdown_txs.remove(&exited_worker_id);
                    log::debug!(
                        "Worker {} exited, {} active workers remaining",
                        exited_worker_id,
                        active_workers
                    );

                    // Save checkpoint when workers exit to preserve progress
                    if let Err(e) = self.save_checkpoint() {
                        log::error!("Checkpoint save failed on worker exit: {}", e);
                    }
                    continue;
                }

                // Check for parallelism changes
                Some(target) = parallelism_rx.recv() => {
                    drop(parallelism_rx); // Release lock before handling
                    let spawned = handle_parallelism_change(
                        target,
                        &mut worker_handles,
                        &mut worker_shutdown_txs,
                        &mut next_worker_id,
                        &job_rx,
                        &job_tx,
                        &shared_state,
                        &result_tx,
                        &worker_exit_tx,
                        &event_tx,
                    );
                    active_workers += spawned;
                    continue; // Don't block on result, loop again
                }

                // Then process results from workers
                result = result_rx.recv() => {
                    drop(parallelism_rx); // Release lock

                    let job_result = match result {
                        Some(r) => r,
                        None => {
                            log::warn!(
                                "Result channel closed unexpectedly after {} results",
                                results_received
                            );
                            break;
                        }
                    };
                    results_received += 1;

                    let result_order = job_result.order;
                    let prefix = job_result.prefix;

                    // Handle job outcome: success, failure, or skipped
                    let ngrams_in_file = match job_result.outcome {
                        JobOutcome::Success { ngram_count } => ngram_count,
                        JobOutcome::Failed { error, attempts } => {
                            // Non-retryable error - mark prefix as failed in checkpoint (for retry on next run)
                            self.checkpoint.fail_prefix(result_order, &prefix);

                            // Emit PrefixFailed event for TUI display (convert Arc<str> to String)
                            let _ = event_tx.send(ImportEvent::PrefixFailed {
                                order: result_order,
                                prefix: prefix.to_string(),
                                error: error.to_string(),
                                attempts,
                            });

                            log::error!(
                                "Prefix {} (order {}) failed after {} attempts: {}. Skipping and continuing.",
                                prefix, result_order, attempts, error
                            );

                            // Save checkpoint immediately to preserve failed state
                            if let Err(e) = self.save_checkpoint() {
                                let _ = event_tx.send(ImportEvent::Error {
                                    message: format!("Checkpoint failed after prefix failure: {}", e),
                                });
                                log::error!("Failed to save checkpoint after prefix failure: {}", e);
                            }

                            // Count this as "processed" for progress purposes (even though it failed)
                            // The prefix will be retried on the next import run
                            files_completed.fetch_add(1, Ordering::Relaxed);

                            // Continue to next result (don't abort the import!)
                            continue;
                        }
                        JobOutcome::Skipped { error, attempts } => {
                            // Max retries exceeded - mark prefix as failed for retry next session
                            self.checkpoint.fail_prefix(result_order, &prefix);

                            // Emit PrefixFailed event for TUI display (convert Arc<str> to String)
                            let _ = event_tx.send(ImportEvent::PrefixFailed {
                                order: result_order,
                                prefix: prefix.to_string(),
                                error: error.to_string(),
                                attempts,
                            });

                            log::warn!(
                                "Prefix {} (order {}) skipped after {} attempts: {}. Will retry next session.",
                                prefix, result_order, attempts, error
                            );

                            // Save checkpoint immediately to preserve failed state
                            if let Err(e) = self.save_checkpoint() {
                                let _ = event_tx.send(ImportEvent::Error {
                                    message: format!("Checkpoint failed after prefix skip: {}", e),
                                });
                                log::error!("Failed to save checkpoint after prefix skip: {}", e);
                            }

                            // Count this as "processed" for progress purposes
                            files_completed.fetch_add(1, Ordering::Relaxed);

                            // Continue to next result
                            continue;
                        }
                    };

                    // Update per-order progress tracking (success case)
                    *order_files_completed.entry(result_order).or_insert(0) += 1;
                    self.checkpoint.complete_prefix(result_order, &prefix);
                    self.checkpoint.add_ngrams(result_order, ngrams_in_file);
                    self.checkpoint.stats.ngrams_by_order[(result_order - 1) as usize] += ngrams_in_file;
                    files_completed.fetch_add(1, Ordering::Relaxed);

                    // Emit per-order progress event
                    let order_done = order_files_completed.get(&result_order).copied().unwrap_or(0);
                    let order_total = order_total_files.get(&result_order).copied().unwrap_or(0);
                    let order_ngrams = self.checkpoint.stats.ngrams_by_order[(result_order - 1) as usize];
                    let order_pending = jobs_per_order.get(&result_order).copied().unwrap_or(0);
                    let order_already_complete = order_total - order_pending;

                    // Order is complete when all successfully processed + failed = pending
                    // Note: failed prefixes count toward completion (they'll be retried next run)
                    let failed_count = self.checkpoint.failed_prefix_count(result_order) as u64;
                    let is_order_complete = order_done + failed_count >= order_pending;

                    let _ = event_tx.send(ImportEvent::OrderProgress {
                        order: result_order,
                        files_completed: order_already_complete + order_done,
                        total_files: order_total,
                        ngrams_processed: order_ngrams,
                        is_complete: is_order_complete,
                    });

                    // Check if order is now complete
                    if is_order_complete && !self.checkpoint.is_order_complete(result_order) {
                        self.checkpoint.complete_order(result_order);
                        let order_duration = order_start_times
                            .get(&result_order)
                            .map(|t| t.elapsed())
                            .unwrap_or_else(|| import_start.elapsed());

                        let _ = event_tx.send(ImportEvent::OrderCompleted {
                            order: result_order,
                            ngram_count: order_ngrams,
                            duration: order_duration,
                        });

                        // Log if there were failures in this order
                        if failed_count > 0 {
                            log::warn!(
                                "Order {} completed with {} failed prefixes (will be retried on next run): {} n-grams in {:?}",
                                result_order,
                                failed_count,
                                order_ngrams,
                                order_duration
                            );
                        } else {
                            log::info!(
                                "Order {} completed: {} n-grams in {:?}",
                                result_order,
                                order_ngrams,
                                order_duration
                            );
                        }
                    }

                    // Save checkpoint periodically
                    if files_completed.load(Ordering::Relaxed) % 10 == 0 {
                        if let Err(e) = self.save_checkpoint() {
                            log::error!("Checkpoint failed: {}", e);
                            let _ = event_tx.send(ImportEvent::Error {
                                message: format!("Checkpoint failed: {}", e),
                            });
                            return Err(e);
                        }
                        let _ = event_tx.send(ImportEvent::CheckpointSaved {
                            prefix: prefix.to_string(),
                        });
                    }
                }
            }
        }

        // Drop channel senders to allow proper cleanup
        drop(result_tx);
        drop(worker_exit_tx);

        // Wait for all workers to finish
        for (_, handle) in worker_handles {
            let _ = handle.await;
        }

        // Wait for worker converter to finish
        let _ = worker_converter.await;

        // Stop periodic stats emitter
        stats_task.abort();
        let _ = stats_task.await;

        // Sync atomic counters back to self
        self.total_ngrams
            .store(total_ngrams.load(Ordering::Relaxed), Ordering::Relaxed);
        self.unique_ngrams
            .store(unique_ngrams.load(Ordering::Relaxed), Ordering::Relaxed);

        // Final checkpoint save
        if let Err(e) = self.save_checkpoint() {
            log::error!("Final checkpoint failed: {}", e);
            let _ = event_tx.send(ImportEvent::Error {
                message: format!("Final checkpoint failed: {}", e),
            });
            return Err(e);
        }

        // Emit ImportCompleted event
        let total_duration = self.start_time.elapsed();
        let total = self.total_ngrams.load(Ordering::Relaxed);
        let _ = event_tx.send(ImportEvent::ImportCompleted {
            total_ngrams: total,
            duration: total_duration,
        });

        // Abort command handler to ensure clean shutdown
        command_handler.abort();
        let _ = command_handler.await;

        // Finalize: compute MKN stats, sync storage, and return final stats
        self.finalize()
    }

    /// Process a single local file.
    fn process_file(&mut self, path: &Path) -> Result<u64, ImportError> {
        let reader = FileNgramReader::open_with_options(
            path,
            self.config.skip_pos_tags,
            self.config.min_count,
        )?;

        let mut aggregator = YearAggregator::new(self.config.year_range);
        let mut ngrams_in_file = 0u64;

        for result in reader {
            let record = result?;

            if let Some(aggregated) = aggregator.push(record) {
                self.store_ngram(&aggregated.ngram, aggregated.total_count)?;
                ngrams_in_file += 1;
            }
        }

        // Flush final n-gram
        if let Some(aggregated) = aggregator.flush() {
            self.store_ngram(&aggregated.ngram, aggregated.total_count)?;
            ngrams_in_file += 1;
        }

        self.total_ngrams.fetch_add(ngrams_in_file, Ordering::Relaxed);
        Ok(ngrams_in_file)
    }

    /// Store an n-gram with its count.
    ///
    /// Uses the storage backend (single-trie or sharded).
    /// MKN statistics are computed as a post-processing step after import completes.
    fn store_ngram(&self, ngram: &str, count: u64) -> Result<(), ImportError> {
        let is_new = self.storage.store(ngram, count).map_err(|e| {
            ImportError::Trie(format!("Failed to store ngram '{}': {}", ngram, e))
        })?;

        if is_new {
            self.unique_ngrams.fetch_add(1, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Calculate current processing rate.
    fn calculate_rate(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.total_ngrams.load(Ordering::Relaxed) as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Estimate time remaining.
    fn estimate_eta(&self, completed: u32, total: u32) -> Option<u64> {
        if completed == 0 || completed >= total {
            return None;
        }

        let elapsed = self.start_time.elapsed().as_secs_f64();
        let rate = completed as f64 / elapsed;
        let remaining = (total - completed) as f64 / rate;

        Some(remaining as u64)
    }

    /// Finalize import: compute MKN statistics, sync storage, and return stats.
    pub fn finalize(&mut self) -> Result<ImportStats, ImportError> {
        log::info!("Finalizing import...");

        // Compute MKN continuation counts
        self.compute_mkn_stats()?;

        // Sync and checkpoint the storage to ensure all data is persisted
        log::info!("Syncing storage to disk...");
        self.storage.sync().map_err(|e| {
            ImportError::Trie(format!("Failed to sync storage: {}", e))
        })?;

        log::info!("Creating storage checkpoint...");
        self.storage.checkpoint().map_err(|e| {
            ImportError::Trie(format!("Failed to checkpoint storage: {}", e))
        })?;

        // Build final stats
        let stats = self.build_stats()?;

        // Clean up checkpoint
        self.cleanup_checkpoint()?;

        log::info!(
            "Import complete: {} n-grams in {} seconds",
            stats.total_ngrams,
            stats.elapsed_seconds
        );

        Ok(stats)
    }

    /// Compute Modified Kneser-Ney smoothing statistics as a post-processing step.
    ///
    /// This function computes MKN statistics differently based on storage mode:
    /// - **Single-trie**: Iterates over the trie and stores stats inline with special keys
    /// - **Sharded**: Uses MknAggregator to compute stats across all shards
    ///
    /// The statistics include:
    /// - N1+(suffix): Count of unique preceding words for each suffix
    /// - N1+prefix(context): Count of unique following words for each context
    ///
    /// These statistics are used by MKN smoothing to estimate probabilities
    /// for unseen n-grams based on lower-order distributions.
    fn compute_mkn_stats(&mut self) -> Result<(), ImportError> {
        if self.checkpoint.mkn_phase == MknPhase::Complete {
            log::info!("MKN statistics already computed");
            return Ok(());
        }

        log::info!("Computing MKN statistics (post-processing)...");

        if self.storage.is_sharded() {
            // Sharded mode: use MknAggregator which iterates over all shards
            self.compute_mkn_stats_sharded()?;
        } else {
            // Single-trie mode: iterate over trie and store stats inline
            self.compute_mkn_stats_single_trie()?;
        }

        self.checkpoint.mkn_phase = MknPhase::Complete;
        self.save_checkpoint()?;

        log::info!("MKN statistics computed successfully");
        Ok(())
    }

    /// Compute MKN stats for sharded storage using MknAggregator.
    fn compute_mkn_stats_sharded(&self) -> Result<(), ImportError> {
        let coordinator = self.storage.as_sharded().ok_or_else(|| {
            ImportError::Trie("Expected sharded storage".to_string())
        })?;

        let aggregator = MknAggregator::new(coordinator);
        let mkn_stats = aggregator.compute_all().map_err(|e| {
            ImportError::Trie(format!("Failed to compute MKN statistics: {}", e))
        })?;

        // Save MKN stats to a separate file alongside the shards
        let mkn_path = self.config.output_path.with_extension("mkn.artrie");
        log::info!("Saving MKN statistics to {:?}...", mkn_path);

        let mkn_trie = DiskBackedCharTrieInner::create(&mkn_path).map_err(|e| {
            ImportError::Trie(format!("Failed to create MKN trie: {}", e))
        })?;
        let mkn_trie = Arc::new(RwLock::new(mkn_trie));

        {
            let mut trie = mkn_trie.write();

            // Store frequency counts for each order
            for (order, counts) in mkn_stats.frequency_counts.iter().enumerate() {
                let prefix = format!("\x00order{}\x00", order);
                trie.upsert(&format!("{}n1", prefix), counts.n1).map_err(|e| {
                    ImportError::Trie(format!("Failed to write n1: {}", e))
                })?;
                trie.upsert(&format!("{}n2", prefix), counts.n2).map_err(|e| {
                    ImportError::Trie(format!("Failed to write n2: {}", e))
                })?;
                trie.upsert(&format!("{}n3", prefix), counts.n3).map_err(|e| {
                    ImportError::Trie(format!("Failed to write n3: {}", e))
                })?;
                trie.upsert(&format!("{}n4", prefix), counts.n4).map_err(|e| {
                    ImportError::Trie(format!("Failed to write n4: {}", e))
                })?;
                trie.upsert(&format!("{}total_unique", prefix), counts.total_unique).map_err(|e| {
                    ImportError::Trie(format!("Failed to write total_unique: {}", e))
                })?;
                trie.upsert(&format!("{}total_count", prefix), counts.total_count).map_err(|e| {
                    ImportError::Trie(format!("Failed to write total_count: {}", e))
                })?;
            }

            // Store continuation counts for each order
            for (order, conts) in mkn_stats.continuation_counts.iter().enumerate() {
                // Store predecessor counts (N1+(•w) - unique predecessors for each context)
                for (context, count) in &conts.predecessor_counts {
                    let key = format!("\x00N1+predecessor\x00{}\x00{}", order, context);
                    trie.upsert(&key, *count).map_err(|e| {
                        ImportError::Trie(format!("Failed to write predecessor count: {}", e))
                    })?;
                }

                // Store successor counts (N1+(w•) - unique successors for each context)
                for (context, count) in &conts.successor_counts {
                    let key = format!("\x00N1+successor\x00{}\x00{}", order, context);
                    trie.upsert(&key, *count).map_err(|e| {
                        ImportError::Trie(format!("Failed to write successor count: {}", e))
                    })?;
                }
            }

            // Checkpoint to persist
            trie.checkpoint().map_err(|e| {
                ImportError::Trie(format!("Failed to checkpoint MKN trie: {}", e))
            })?;
        }

        log::info!(
            "MKN statistics saved: {} orders with frequency and continuation counts",
            mkn_stats.max_order
        );

        Ok(())
    }

    /// Compute MKN stats for single-trie storage (original behavior).
    fn compute_mkn_stats_single_trie(&self) -> Result<(), ImportError> {
        // Collect unique (suffix, prefix) and (context, following) pairs
        // using HashSets for deduplication
        use std::collections::HashSet;
        let mut continuation_pairs: HashSet<(String, String)> = HashSet::new();
        let mut unique_cont_pairs: HashSet<(String, String)> = HashSet::new();

        // Phase 1: Iterate all n-grams and collect pairs
        log::info!("Phase 1: Collecting continuation pairs from n-grams...");
        {
            let trie = self.trie.read();
            // Use iter_prefix_with_values("") to iterate all entries
            if let Some(entries) = trie.iter_prefix_with_values("").map_err(|e| {
                ImportError::Trie(format!("Failed to iterate trie: {}", e))
            })? {
                for (ngram, _count) in entries {
                    // Skip metadata keys (they start with \x00)
                    if ngram.starts_with('\x00') {
                        continue;
                    }

                    let words: Vec<&str> = ngram.split_whitespace().collect();
                    if words.len() >= 2 {
                        // MKN Pass 1: continuation counts (suffix → unique prefixes)
                        let prefix = words[0].to_string();
                        let suffix = words[1..].join(" ");
                        continuation_pairs.insert((suffix, prefix));

                        // MKN Pass 2: unique continuations (context → unique following)
                        let context = words[..words.len() - 1].join(" ");
                        let following = words[words.len() - 1].to_string();
                        unique_cont_pairs.insert((context, following));
                    }
                }
            }
        }

        log::info!(
            "Collected {} continuation pairs and {} unique continuation pairs",
            continuation_pairs.len(),
            unique_cont_pairs.len()
        );

        // Phase 2: Compute and write MKN counts
        log::info!("Phase 2: Writing MKN statistics to trie...");
        {
            let mut trie = self.trie.write();

            // Count unique prefixes per suffix (N1+(suffix))
            let mut suffix_counts: std::collections::HashMap<String, u64> =
                std::collections::HashMap::new();
            for (suffix, _prefix) in &continuation_pairs {
                *suffix_counts.entry(suffix.clone()).or_insert(0) += 1;
            }

            // Write continuation counts
            for (suffix, count) in &suffix_counts {
                let count_key = format!("\x00N1+\x00{}", suffix);
                trie.upsert(&count_key, *count).map_err(|e| {
                    ImportError::Trie(format!(
                        "Failed to write MKN continuation count for '{}': {}",
                        suffix, e
                    ))
                })?;
            }

            // Count unique following words per context (N1+prefix(context))
            let mut context_counts: std::collections::HashMap<String, u64> =
                std::collections::HashMap::new();
            for (context, _following) in &unique_cont_pairs {
                *context_counts.entry(context.clone()).or_insert(0) += 1;
            }

            // Write unique continuation counts
            for (context, count) in &context_counts {
                let count_key = format!("\x00N1+prefix\x00{}", context);
                trie.upsert(&count_key, *count).map_err(|e| {
                    ImportError::Trie(format!(
                        "Failed to write MKN unique continuation count for '{}': {}",
                        context, e
                    ))
                })?;
            }
        }

        Ok(())
    }

    /// Build final statistics.
    fn build_stats(&self) -> Result<ImportStats, ImportError> {
        let elapsed = self.start_time.elapsed().as_secs();
        let total = self.total_ngrams.load(Ordering::Relaxed);

        Ok(ImportStats {
            total_ngrams: total,
            ngrams_by_order: self.checkpoint.stats.ngrams_by_order,
            unique_ngrams: self.unique_ngrams.load(Ordering::Relaxed),
            bytes_downloaded: self.checkpoint.stats.bytes_downloaded,
            files_processed: self.checkpoint.stats.files_processed,
            elapsed_seconds: elapsed,
            ngrams_per_second: if elapsed > 0 {
                total as f64 / elapsed as f64
            } else {
                0.0
            },
        })
    }

    /// Get current checkpoint state (for inspection).
    pub fn checkpoint(&self) -> &ImportCheckpoint {
        &self.checkpoint
    }

    /// Get the configuration.
    pub fn config(&self) -> &GoogleBooksConfig {
        &self.config
    }
}

/// Install a signal handler for graceful shutdown.
///
/// Returns a future that completes when SIGINT or SIGTERM is received.
#[cfg(feature = "google-books")]
pub async fn shutdown_signal() {
    use tokio::signal;

    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

/// Run import with graceful shutdown handling.
#[cfg(feature = "google-books")]
pub async fn run_import_with_shutdown<F>(
    mut importer: GoogleBooksImporter,
    progress: F,
) -> Result<ImportStats, ImportError>
where
    F: FnMut(ImportProgress) + Send + 'static,
{
    let importer_ref = Arc::new(parking_lot::Mutex::new(importer));
    let importer_clone = Arc::clone(&importer_ref);

    // Spawn shutdown handler
    let shutdown_handle = tokio::spawn(async move {
        shutdown_signal().await;
        log::warn!("Received shutdown signal, saving checkpoint...");
        if let Some(mut importer) = importer_clone.try_lock() {
            importer.interrupt();
        }
    });

    // Run import
    let result = {
        let mut importer = importer_ref.lock();
        importer.import_http(progress).await
    };

    // Cancel shutdown handler if import completed normally
    shutdown_handle.abort();

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_import_progress() {
        let progress = ImportProgress {
            current_order: 3,
            current_prefix: "th".to_string(),
            ngrams_in_file: 1000,
            total_ngrams: 50000,
            files_completed: 10,
            total_files: 678,
            bytes_downloaded: 1024 * 1024,
            ngrams_per_second: 5000.0,
            eta_seconds: Some(3600),
            phase: ImportPhase::Importing,
        };

        assert_eq!(progress.current_order, 3);
        assert_eq!(progress.phase, ImportPhase::Importing);
    }

    #[test]
    fn test_import_stats_default() {
        let stats = ImportStats::default();
        assert_eq!(stats.total_ngrams, 0);
        assert_eq!(stats.files_processed, 0);
    }

    #[test]
    fn test_importer_creation() {
        let dir = tempdir().expect("Failed to create temp dir");
        let output_path = dir.path().join("test.artrie");

        let config = GoogleBooksConfig::builder()
            .language("en")
            .orders(1..=1)
            .output_path(output_path)
            .build()
            .expect("Failed to build config");

        let importer = GoogleBooksImporter::new(config);
        assert!(importer.is_ok());
    }

    #[test]
    fn test_unsupported_language() {
        let dir = tempdir().expect("Failed to create temp dir");
        let output_path = dir.path().join("test.artrie");

        let config = GoogleBooksConfig::builder()
            .language("invalid")
            .orders(1..=1)
            .output_path(output_path)
            .build()
            .expect("Failed to build config");

        let importer = GoogleBooksImporter::new(config);
        assert!(matches!(importer, Err(ImportError::UnsupportedLanguage(_))));
    }

    #[test]
    fn test_interrupt_flag() {
        let dir = tempdir().expect("Failed to create temp dir");
        let output_path = dir.path().join("test.artrie");

        let config = GoogleBooksConfig::builder()
            .language("en")
            .orders(1..=1)
            .output_path(output_path)
            .build()
            .expect("Failed to build config");

        let importer = GoogleBooksImporter::new(config).expect("Failed to create importer");

        assert!(!importer.is_interrupted());
        importer.interrupt();
        assert!(importer.is_interrupted());
    }

    /// Create a mock Google Books n-gram gzip file for testing.
    fn create_mock_ngram_file(path: &std::path::Path, ngrams: &[(&str, u16, u64, u32)]) {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        let file = std::fs::File::create(path).expect("Failed to create file");
        let mut encoder = GzEncoder::new(file, Compression::default());

        for (ngram, year, count, volume_count) in ngrams {
            writeln!(encoder, "{}\t{}\t{}\t{}", ngram, year, count, volume_count)
                .expect("Failed to write");
        }

        encoder.finish().expect("Failed to finish compression");
    }

    #[test]
    fn test_file_import_with_mock_data() {
        let dir = tempdir().expect("Failed to create temp dir");
        let output_path = dir.path().join("test.artrie");
        let ngram_dir = dir.path().join("ngrams");
        std::fs::create_dir(&ngram_dir).expect("Failed to create ngram dir");

        // Create a mock 1-gram file with test data
        let file_path = ngram_dir.join("googlebooks-eng-all-1gram-20200217-t.gz");
        create_mock_ngram_file(
            &file_path,
            &[
                ("the", 2000, 50000, 1000),
                ("the", 2001, 55000, 1100),
                ("the", 2002, 60000, 1200),
                ("this", 2000, 10000, 500),
                ("this", 2001, 11000, 550),
                ("that", 2000, 20000, 800),
                ("that", 2001, 21000, 850),
                ("test", 2000, 5000, 200),
            ],
        );

        let config = GoogleBooksConfig::builder()
            .language("en")
            .orders(1..=1)
            .min_count(1)
            .output_path(output_path)
            .build()
            .expect("Failed to build config");

        let mut importer = GoogleBooksImporter::new(config).expect("Failed to create importer");

        // Import from local files
        let result = importer.import_files(&ngram_dir, |progress| {
            assert!(progress.current_order >= 1);
        });

        assert!(result.is_ok());
        let stats = result.unwrap();
        assert!(stats.total_ngrams > 0, "Should have imported n-grams");
    }

    #[test]
    fn test_year_filtering() {
        let dir = tempdir().expect("Failed to create temp dir");
        let output_path = dir.path().join("test.artrie");
        let ngram_dir = dir.path().join("ngrams");
        std::fs::create_dir(&ngram_dir).expect("Failed to create ngram dir");

        // Create a mock file with data from multiple years
        let file_path = ngram_dir.join("googlebooks-eng-all-1gram-20200217-a.gz");
        create_mock_ngram_file(
            &file_path,
            &[
                ("apple", 1990, 1000, 100),
                ("apple", 2000, 2000, 200),
                ("apple", 2010, 3000, 300),
                ("ant", 1990, 500, 50),
                ("ant", 2000, 600, 60),
            ],
        );

        // Import with year range filter (only 2000-2010)
        let config = GoogleBooksConfig::builder()
            .language("en")
            .orders(1..=1)
            .min_count(1)
            .year_range(2000, 2010)
            .output_path(output_path)
            .build()
            .expect("Failed to build config");

        let mut importer = GoogleBooksImporter::new(config).expect("Failed to create importer");
        let result = importer.import_files(&ngram_dir, |_| {});

        assert!(result.is_ok());
        // The year filtering should have excluded 1990 data
    }

    #[test]
    fn test_min_count_filtering() {
        let dir = tempdir().expect("Failed to create temp dir");
        let output_path = dir.path().join("test.artrie");
        let ngram_dir = dir.path().join("ngrams");
        std::fs::create_dir(&ngram_dir).expect("Failed to create ngram dir");

        // Create a mock file with varying counts
        let file_path = ngram_dir.join("googlebooks-eng-all-1gram-20200217-b.gz");
        create_mock_ngram_file(
            &file_path,
            &[
                ("big", 2000, 100000, 5000),    // High count
                ("bear", 2000, 50000, 2500),    // Medium count
                ("bxyz", 2000, 10, 2),          // Low count (below default threshold)
            ],
        );

        // Import with min_count=40 (Google's default)
        let config = GoogleBooksConfig::builder()
            .language("en")
            .orders(1..=1)
            .min_count(40)
            .output_path(output_path)
            .build()
            .expect("Failed to build config");

        let mut importer = GoogleBooksImporter::new(config).expect("Failed to create importer");
        let result = importer.import_files(&ngram_dir, |_| {});

        assert!(result.is_ok());
        // "bxyz" should have been filtered out due to low count
    }

    #[test]
    fn test_pos_tag_filtering() {
        let dir = tempdir().expect("Failed to create temp dir");
        let output_path = dir.path().join("test.artrie");
        let ngram_dir = dir.path().join("ngrams");
        std::fs::create_dir(&ngram_dir).expect("Failed to create ngram dir");

        // Create a mock file with POS-tagged and regular n-grams
        let file_path = ngram_dir.join("googlebooks-eng-all-1gram-20200217-c.gz");
        create_mock_ngram_file(
            &file_path,
            &[
                ("cat", 2000, 50000, 2500),
                ("cat_NOUN", 2000, 45000, 2300),     // POS tag
                ("car", 2000, 40000, 2000),
                ("the_DET", 2000, 100000, 5000),     // POS tag
            ],
        );

        // Import with POS tag filtering enabled
        let mut config = GoogleBooksConfig::builder()
            .language("en")
            .orders(1..=1)
            .min_count(1)
            .output_path(output_path)
            .build()
            .expect("Failed to build config");
        config.skip_pos_tags = true;

        let mut importer = GoogleBooksImporter::new(config).expect("Failed to create importer");
        let result = importer.import_files(&ngram_dir, |_| {});

        assert!(result.is_ok());
        // POS-tagged n-grams should have been filtered out
    }

    #[test]
    fn test_checkpoint_save_and_load() {
        let dir = tempdir().expect("Failed to create temp dir");
        let output_path = dir.path().join("test.artrie");

        let config = GoogleBooksConfig::builder()
            .language("en")
            .orders(1..=1)
            .output_path(output_path.clone())
            .build()
            .expect("Failed to build config");

        let mut importer = GoogleBooksImporter::new(config.clone()).expect("Failed to create importer");

        // Save checkpoint (now saved to trie, not JSON file)
        importer.save_checkpoint().expect("Failed to save checkpoint");

        // Load checkpoint from trie
        let loaded = {
            let trie = importer.trie.read();
            ImportCheckpoint::load_from_trie(&*trie)
                .expect("Failed to load checkpoint from trie")
                .expect("Checkpoint should exist in trie")
        };

        // v2 format: order_progress is a HashMap, completed_orders() is a method
        assert!(loaded.order_progress.is_empty());  // Fresh checkpoint has no progress
        assert!(loaded.completed_orders().is_empty());  // No orders completed yet
    }
}
