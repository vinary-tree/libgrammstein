//! Worker pool and per-prefix-file processing pipeline.
//!
//! This module owns the async machinery used by the importer to fan prefix
//! files out to a bounded pool of concurrent download/parse workers:
//!
//! - `Job` / `JobResult` / `RequestDebugInfo` — work items + debug telemetry.
//! - `WorkerSharedState` — per-import shared state for the persistent-worker
//!   `worker_task` architecture.
//! - `PrefixProcessingContext` — per-order shared state for the
//!   prefix-file architecture (`process_prefix_file{,_cached}`).
//! - `process_aggregated_stream` — drains a streaming parse into the
//!   storage layer with chunked transactions.
//! - `process_single_attempt` / `process_single_attempt_cached` — perform
//!   a single download+parse attempt with non-blocking retry semantics.
//! - `worker_task` — the long-lived per-worker loop the persistent-worker
//!   architecture pins to each slot.
//! - `process_prefix_file` / `process_prefix_file_cached` — single-shot
//!   prefix-file processing used by the parallel-import path.

#![cfg(feature = "google-books")]

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::super::config::GoogleBooksConfig;
use super::super::reader::ReaderError;
use super::super::storage::{NgramStorage, StoragePrefixTx};
use super::super::task_manager::RetryAfter;
use super::{
    cleanup_cache_file, download_to_cache, extract_retry_after, is_retryable_error,
    store_ngram_shared, ImportError, WorkerUpdate, COUNTER_BATCH_SIZE,
};

/// Maximum retry attempts for transient failures.
#[cfg(feature = "google-books")]
pub(super) const MAX_RETRIES: u8 = 5;

/// Initial backoff delay in milliseconds (doubles each retry).
#[cfg(feature = "google-books")]
pub(super) const INITIAL_BACKOFF_MS: u64 = 1000;

/// A job for the worker pool to process.
#[cfg(feature = "google-books")]
#[derive(Clone)]
pub(super) struct Job {
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
    pub(super) fn new(url: impl Into<Arc<str>>, prefix: impl Into<Arc<str>>, order: u8) -> Self {
        Self {
            url: url.into(),
            prefix: prefix.into(),
            order,
            attempt: 0,
            backoff_ms: INITIAL_BACKOFF_MS,
            ready_at: None, // Ready immediately
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
}

/// Result of a job processing attempt.
///
/// Always includes the order and prefix so the main task knows which prefix
/// succeeded or failed. This enables the main task to mark failed prefixes
/// in the checkpoint and continue with other prefixes instead of aborting.
#[cfg(feature = "google-books")]
#[derive(Debug)]
pub(super) struct JobResult {
    /// N-gram order for this job (1-5).
    pub(super) order: u8,
    /// The prefix that was processed (e.g., "th", "to").
    pub(super) prefix: Arc<str>,
    /// The result: n-gram count on success, or error details on failure.
    pub(super) outcome: JobOutcome,
}

/// Outcome of processing a single prefix file.
#[cfg(feature = "google-books")]
#[derive(Debug)]
pub(super) enum JobOutcome {
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
    /// Time taken for the request in milliseconds.
    response_time_ms: u64,
    /// Error message.
    error_message: String,
}

#[cfg(feature = "google-books")]
impl RequestDebugInfo {
    /// Create debug info from an error and request timing.
    fn from_error(url: &str, error: &ImportError, response_time: Duration) -> Self {
        let status_code = match error {
            ImportError::Reader(e) => {
                // Try to extract HTTP info from the error message
                let msg = e.to_string();
                if msg.contains("404") {
                    Some(404)
                } else if msg.contains("429") {
                    Some(429)
                } else if msg.contains("500") {
                    Some(500)
                } else if msg.contains("503") {
                    Some(503)
                } else {
                    None
                }
            }
            _ => None,
        };

        Self {
            url: url.to_string(),
            status_code,
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
pub(super) enum PrefixOutcome {
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
pub(super) struct WorkerSharedState {
    pub(super) config: GoogleBooksConfig,
    pub(super) storage: Arc<NgramStorage>,
    pub(super) total_ngrams: Arc<AtomicU64>,
    pub(super) unique_ngrams: Arc<AtomicU64>,
    pub(super) progress_tx: tokio::sync::mpsc::Sender<WorkerUpdate>,
    pub(super) paused: Arc<AtomicBool>,
    /// Current number of jobs in the queue (for all-deferred detection)
    pub(super) queue_size: Arc<AtomicUsize>,
    /// Per-worker packed stats for non-blocking, race-free sampling.
    /// Each AtomicU64 packs: upper 32 bits = total n-grams, lower 32 bits = unique n-grams.
    /// Single atomic ensures both counts are read/written atomically together.
    /// Maximum workers supported: length of this Vec.
    pub(super) worker_stats: Vec<AtomicU64>,
    /// Shared HTTP client for connection pooling and HTTP/2 multiplexing.
    /// Creating one client and sharing it across workers avoids the concurrency
    /// amplification bug where each worker creates independent connection pools,
    /// causing Google to see a spike in connections and trigger rate limiting.
    pub(super) http_client: reqwest::Client,
}

/// Shared context for the per-prefix-file processing path.
///
/// This struct holds everything that `process_prefix_file` and
/// `process_prefix_file_cached` need across all concurrent invocations within
/// a single order's import. It is constructed once per order at the call site
/// and shared via `Arc` to every spawned future.
///
/// Separation of concerns: distinct from `WorkerSharedState` (which serves the
/// persistent-worker `worker_task` architecture) because the prefix-file
/// architecture has its own concerns — a worker-ID claim pool and an Optional
/// progress channel (the worker-task path always has progress; the prefix-file
/// path may be invoked headless). The two structs share a conceptual core
/// (config, storage, counters, http_client) which a future refactor may
/// extract into a common base type.
#[cfg(feature = "google-books")]
pub(super) struct PrefixProcessingContext {
    pub(super) config: GoogleBooksConfig,
    pub(super) storage: Arc<NgramStorage>,
    pub(super) total_ngrams: Arc<AtomicU64>,
    pub(super) unique_ngrams: Arc<AtomicU64>,
    pub(super) progress_tx: Option<tokio::sync::mpsc::Sender<WorkerUpdate>>,
    /// Shared HTTP client — created once per order so all spawned futures
    /// reuse a single connection pool (avoids the concurrency-amplification
    /// rate-limiting bug previously caused by per-call `Client::builder()`).
    pub(super) http_client: reqwest::Client,
    /// Worker-ID claim channel: claimed when a future starts, returned when
    /// it finishes, ensuring each concurrent worker has a unique ID for
    /// display purposes.
    pub(super) worker_id_pool_tx: tokio::sync::mpsc::Sender<usize>,
    pub(super) worker_id_pool_rx: Arc<tokio::sync::Mutex<tokio::sync::mpsc::Receiver<usize>>>,
}

/// Shared helper: consume a stream of aggregated n-grams into the storage
/// transaction, with periodic chunked commits to bound per-transaction memory.
///
/// Used by both `process_prefix_file` (HTTP-streamed) and
/// `process_prefix_file_cached` (locally-cached file). The two paths differ only
/// in how the stream is produced; the per-record processing — SET-semantics
/// insert, chunk-commit at `ctx.config.tx_chunk_size`, periodic progress
/// emission, abort-on-error, final commit — is identical.
///
/// On success, returns the total n-grams processed and commits the final chunk
/// (marking the prefix complete + persisting checkpoint state). On error,
/// aborts the transaction; buffered uncommitted n-grams are discarded.
///
/// SET semantics + chunked commits = idempotent crash recovery: re-running the
/// prefix re-inserts the same values, and unchecked-pointed chunks are lost on
/// crash, so the prefix is just re-imported from scratch on resume.
#[cfg(feature = "google-books")]
async fn process_aggregated_stream<S>(
    stream: S,
    mut tx: StoragePrefixTx,
    ctx: &Arc<PrefixProcessingContext>,
    prefix: &str,
    order: u8,
    worker_id: usize,
    source_label: &str,
) -> Result<u64, ImportError>
where
    S: tokio_stream::Stream<Item = Result<super::super::aggregator::AggregatedNgram, ReaderError>>,
{
    use tokio_stream::StreamExt;

    tokio::pin!(stream);

    const NGRAM_PROGRESS_INTERVAL: u64 = 50_000;
    let mut count = 0u64;
    let mut chunk_count = 0u64;
    let mut stream_err: Option<ImportError> = None;
    let tx_chunk_size = ctx.config.tx_chunk_size;

    while let Some(result) = stream.next().await {
        let agg = match result {
            Ok(agg) => agg,
            Err(e) => {
                stream_err = Some(e.into());
                break;
            }
        };

        // Insert into transaction (SET semantics, not increment)
        // tx_insert_ngram splits to SmallVec internally, avoiding heap alloc
        if let Err(e) = ctx
            .storage
            .tx_insert_ngram(&mut tx, &agg.ngram, agg.total_count)
        {
            stream_err = Some(e.into());
            break;
        }
        count += 1;
        chunk_count += 1;

        // Chunked commit: bound per-transaction memory for large files
        if tx_chunk_size > 0 && chunk_count >= tx_chunk_size {
            match ctx
                .storage
                .commit_and_renew_prefix_tx(&mut tx, prefix, order)
            {
                Ok(committed) => {
                    log::trace!(
                        "Worker {}: committed chunk for {} '{}' ({} n-grams)",
                        worker_id,
                        source_label,
                        prefix,
                        committed
                    );
                    chunk_count = 0;
                }
                Err(e) => {
                    stream_err = Some(e.into());
                    break;
                }
            }
        }

        // Emit periodic progress for TUI display
        if count % NGRAM_PROGRESS_INTERVAL == 0 {
            if let Some(ref ptx) = ctx.progress_tx {
                let _ = ptx.try_send(WorkerUpdate::NgramProgress {
                    worker_id,
                    ngram_count: count,
                });
            }
        }
    }

    if let Some(e) = stream_err {
        if let Err(abort_err) = ctx.storage.abort_prefix_tx(tx) {
            log::warn!(
                "Worker {}: failed to abort transaction for {} '{}': {}",
                worker_id,
                source_label,
                prefix,
                abort_err
            );
        }
        return Err(e);
    }

    // Commit the final chunk and mark prefix as complete
    let committed = ctx.storage.commit_prefix_tx(tx)?;
    ctx.total_ngrams.fetch_add(count, Ordering::Relaxed);
    ctx.unique_ngrams
        .fetch_add(committed as u64, Ordering::Relaxed);
    log::trace!(
        "Worker {}: committed {} '{}' with {} n-grams ({} inserted)",
        worker_id,
        source_label,
        prefix,
        count,
        committed
    );
    Ok(count)
}

/// Process a single job attempt (no retry loop - single attempt only).
///
/// This helper extracts the core processing logic from worker_task to enable
/// non-blocking retry with DelayQueue.
///
/// ## Transaction-Based Atomicity (Sharded Mode)
///
/// For sharded storage, this function uses document transactions to ensure
/// idempotent imports:
///
/// 1. Begin a transaction before processing n-grams
/// 2. Buffer all n-grams in the transaction using SET semantics
/// 3. Commit atomically after all n-grams are processed
/// 4. On error, abort the transaction (buffered n-grams are discarded)
///
/// This prevents double-counting when an import is interrupted and resumed:
/// uncommitted transactions are discarded on recovery, and re-processing
/// simply SETs the same values again (idempotent).
///
/// Per-worker stats are updated continuously via packed atomics for race-free
/// sampling by the stats sampler task. No batching or progress channel sends
/// are needed - the stats sampler reads per-worker counters every 3 seconds.
#[cfg(feature = "google-books")]
async fn process_single_attempt(
    job: &Job,
    shared: &WorkerSharedState,
    worker_id: usize,
) -> Result<u64, ImportError> {
    use super::super::reader::HttpNgramReader;
    use tokio_stream::StreamExt;

    // Add small random delay to stagger connection starts (reduces rate limiting)
    let jitter_ms = rand::random::<u64>() % 500;
    tokio::time::sleep(Duration::from_millis(jitter_ms)).await;

    let mut reader = HttpNgramReader::with_options(
        &job.url,
        shared.config.skip_pos_tags,
        shared.config.min_count,
    );

    // Use the shared HTTP client for connection pooling and HTTP/2 multiplexing
    let stream = reader
        .stream_aggregated_with_client(shared.config.year_range, Some(shared.http_client.clone()));
    tokio::pin!(stream);

    // Local counters for this job (packed into per-worker atomic for race-free sampling)
    let mut count = 0u64;

    // Try to begin a transaction for atomic, idempotent import (sharded mode only)
    let maybe_tx = shared.storage.begin_prefix_tx(&job.prefix, job.order)?;

    // Process based on whether we have a transaction
    let tx_chunk_size = shared.config.tx_chunk_size;
    let result = if let Some(mut tx) = maybe_tx {
        // Sharded mode: use transaction for atomic import with chunking.
        // All tx operations are in a single async block for clean ownership.
        let tx_result: Result<u64, ImportError> = async {
            let mut chunk_count = 0u64;
            let mut stream_err: Option<ImportError> = None;

            while let Some(result) = stream.next().await {
                let agg = match result {
                    Ok(agg) => agg,
                    Err(e) => {
                        stream_err = Some(e.into());
                        break;
                    }
                };

                // Insert into transaction (SET semantics, not increment)
                // tx_insert_ngram splits to SmallVec internally, avoiding heap alloc
                if let Err(e) = shared
                    .storage
                    .tx_insert_ngram(&mut tx, &agg.ngram, agg.total_count)
                {
                    stream_err = Some(e.into());
                    break;
                }
                count += 1;
                chunk_count += 1;

                // Chunked commit: bound per-transaction memory for large files
                if tx_chunk_size > 0 && chunk_count >= tx_chunk_size {
                    match shared
                        .storage
                        .commit_and_renew_prefix_tx(&mut tx, &job.prefix, job.order)
                    {
                        Ok(committed) => {
                            log::trace!(
                                "Worker {}: committed chunk for prefix '{}' ({} n-grams)",
                                worker_id,
                                job.prefix,
                                committed
                            );
                            chunk_count = 0;
                        }
                        Err(e) => {
                            stream_err = Some(e.into());
                            break;
                        }
                    }
                }

                // Update per-worker atomic with count (for progress display)
                if worker_id < shared.worker_stats.len() {
                    let packed = (count as u64) << 32;
                    shared.worker_stats[worker_id].store(packed, Ordering::Relaxed);
                }
            }

            if let Some(e) = stream_err {
                // Abort the transaction - buffered n-grams are discarded
                if let Err(abort_err) = shared.storage.abort_prefix_tx(tx) {
                    log::warn!(
                        "Worker {}: failed to abort transaction for prefix '{}': {}",
                        worker_id,
                        job.prefix,
                        abort_err
                    );
                }
                return Err(e);
            }

            // Commit the final chunk and mark prefix as complete
            let committed = shared.storage.commit_prefix_tx(tx)?;
            log::trace!(
                "Worker {}: committed prefix '{}' with {} n-grams",
                worker_id,
                job.prefix,
                committed
            );
            Ok(count)
        }
        .await;

        tx_result
    } else {
        // Single-trie mode: use original increment-based approach
        // (No transaction support - caller must handle resume correctly)
        let mut unique_count = 0u64;

        while let Some(result) = stream.next().await {
            let agg = result?;
            let storage_result = store_ngram_shared(&agg.ngram, agg.total_count, &shared.storage)?;
            count += 1;
            if storage_result.is_new {
                unique_count += 1;
            }

            // Update per-worker atomic with packed counts (race-free, no batching needed)
            if worker_id < shared.worker_stats.len() {
                let packed = ((count as u64) << 32) | (unique_count as u64 & 0xFFFFFFFF);
                shared.worker_stats[worker_id].store(packed, Ordering::Relaxed);
            }
        }

        // Update unique_ngrams counter for single-trie mode
        if unique_count > 0 {
            shared
                .unique_ngrams
                .fetch_add(unique_count, Ordering::Relaxed);
        }

        Ok(count)
    };

    // Final flush to global counters (for checkpoint persistence)
    if let Ok(ngram_count) = result {
        shared
            .total_ngrams
            .fetch_add(ngram_count, Ordering::Relaxed);
    }

    // Reset per-worker stats after job completion (so next job starts fresh)
    if worker_id < shared.worker_stats.len() {
        shared.worker_stats[worker_id].store(0, Ordering::Relaxed);
    }

    result
}

/// Process a single job attempt using cached file mode.
///
/// 1. Compute cache path from config
/// 2. If cached file exists → skip download
/// 3. Else → download raw .gz to cache
/// 4. Stream from cached file via `stream_aggregated_from_cached_file`
/// 5. Process n-grams (same tx/non-tx logic as `process_single_attempt`)
/// 6. On success: delete cached file
/// 7. On error: delete cached file + .downloading remnant (will re-download on retry)
#[cfg(feature = "google-books")]
async fn process_single_attempt_cached(
    job: &Job,
    shared: &WorkerSharedState,
    worker_id: usize,
) -> Result<u64, ImportError> {
    use super::super::reader::stream_aggregated_from_cached_file;
    use tokio_stream::StreamExt;

    // Compute cache path
    let cache_path = shared
        .config
        .cache_file_path(job.order, &job.prefix)
        .ok_or_else(|| {
            ImportError::Config(format!(
                "Unknown language '{}' for cache file path",
                shared.config.language
            ))
        })?;

    // Download to cache (skips if already cached)
    download_to_cache(&job.url, &cache_path, &shared.http_client).await?;

    // Stream from cached file
    let stream = stream_aggregated_from_cached_file(
        &cache_path,
        shared.config.year_range,
        shared.config.skip_pos_tags,
        shared.config.min_count,
    );
    tokio::pin!(stream);

    // Local counters for this job
    let mut count = 0u64;

    // Try to begin a transaction for atomic, idempotent import (sharded mode only)
    let maybe_tx = shared.storage.begin_prefix_tx(&job.prefix, job.order)?;

    // Process based on whether we have a transaction
    let tx_chunk_size = shared.config.tx_chunk_size;
    let result = if let Some(mut tx) = maybe_tx {
        // Sharded mode: use transaction for atomic import with chunking.
        let tx_result: Result<u64, ImportError> = async {
            let mut chunk_count = 0u64;
            let mut stream_err: Option<ImportError> = None;

            while let Some(result) = stream.next().await {
                let agg = match result {
                    Ok(agg) => agg,
                    Err(e) => {
                        stream_err = Some(e.into());
                        break;
                    }
                };
                if let Err(e) = shared
                    .storage
                    .tx_insert_ngram(&mut tx, &agg.ngram, agg.total_count)
                {
                    stream_err = Some(e.into());
                    break;
                }
                count += 1;
                chunk_count += 1;

                // Chunked commit: bound per-transaction memory for large files
                if tx_chunk_size > 0 && chunk_count >= tx_chunk_size {
                    match shared
                        .storage
                        .commit_and_renew_prefix_tx(&mut tx, &job.prefix, job.order)
                    {
                        Ok(committed) => {
                            log::trace!(
                                "Worker {}: committed chunk for cached prefix '{}' ({} n-grams)",
                                worker_id,
                                job.prefix,
                                committed
                            );
                            chunk_count = 0;
                        }
                        Err(e) => {
                            stream_err = Some(e.into());
                            break;
                        }
                    }
                }

                if worker_id < shared.worker_stats.len() {
                    let packed = (count as u64) << 32;
                    shared.worker_stats[worker_id].store(packed, Ordering::Relaxed);
                }
            }

            if let Some(e) = stream_err {
                if let Err(abort_err) = shared.storage.abort_prefix_tx(tx) {
                    log::warn!(
                        "Worker {}: failed to abort transaction for prefix '{}': {}",
                        worker_id,
                        job.prefix,
                        abort_err
                    );
                }
                return Err(e);
            }

            let committed = shared.storage.commit_prefix_tx(tx)?;
            log::trace!(
                "Worker {}: committed cached prefix '{}' with {} n-grams",
                worker_id,
                job.prefix,
                committed
            );
            Ok(count)
        }
        .await;

        tx_result
    } else {
        // Single-trie mode: use original increment-based approach
        let mut unique_count = 0u64;

        while let Some(result) = stream.next().await {
            let agg = result?;
            let storage_result = store_ngram_shared(&agg.ngram, agg.total_count, &shared.storage)?;
            count += 1;
            if storage_result.is_new {
                unique_count += 1;
            }

            if worker_id < shared.worker_stats.len() {
                let packed = ((count as u64) << 32) | (unique_count as u64 & 0xFFFFFFFF);
                shared.worker_stats[worker_id].store(packed, Ordering::Relaxed);
            }
        }

        if unique_count > 0 {
            shared
                .unique_ngrams
                .fetch_add(unique_count, Ordering::Relaxed);
        }

        Ok(count)
    };

    // Clean up cached file on both success and error
    // On success: no longer needed. On error: will re-download on retry.
    cleanup_cache_file(&cache_path).await;

    // Final flush to global counters
    if let Ok(ngram_count) = result {
        shared
            .total_ngrams
            .fetch_add(ngram_count, Ordering::Relaxed);
    }

    // Reset per-worker stats after job completion
    if worker_id < shared.worker_stats.len() {
        shared.worker_stats[worker_id].store(0, Ordering::Relaxed);
    }

    result
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
pub(super) async fn worker_task(
    worker_id: usize,
    job_rx: async_channel::Receiver<Job>,
    job_tx: async_channel::Sender<Job>,
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

        // Get next job from queue (no mutex needed - async_channel receiver is Clone)
        let job = tokio::select! {
            biased;
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    log::debug!("Worker {} received shutdown signal while waiting for job", worker_id);
                    break;
                }
                continue;
            }
            result = job_rx.recv() => result.ok(),
        };

        let Some(job) = job else {
            // Queue closed with no more jobs - but we might have deferred jobs pending.
            // If all jobs were deferred (waiting on retry backoff), we need to check.
            if consecutive_deferred > 0 {
                log::debug!(
                    "Worker {} queue closed with {} deferred jobs pending",
                    worker_id,
                    consecutive_deferred
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
                        // Add per-worker jitter to prevent thundering herd when all workers
                        // wake up simultaneously after all-deferred sleep
                        let jitter = Duration::from_millis(
                            (worker_id as u64 * 100) + (rand::random::<u64>() % 500),
                        );
                        let staggered_wait = wait + jitter;
                        log::debug!(
                            "Worker {} blocking {}ms (+{}ms jitter) - all {} jobs deferred",
                            worker_id,
                            wait.as_millis(),
                            jitter.as_millis(),
                            queue_size
                        );
                        tokio::time::sleep(staggered_wait).await;
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

        // NOTE: We do NOT decrement queue_size here. The new accounting model:
        // - queue_size represents "jobs remaining to complete"
        // - Decrement ONLY when a job is finished (success, skipped, or failed permanently)
        // - Never decrement on job pickup (avoids phantom jobs from deferred requeues)
        // - Never increment on retry (job was never "completed", so nothing to restore)

        // ===== DEFER-AND-CONTINUE: Check if target shard is syncing =====
        // If the shard that would store this job's n-grams is currently being synced
        // (as part of a parallel checkpoint), defer the job and pick up the next one.
        // This prevents workers from blocking on a syncing shard.
        //
        // Key points:
        // - We do NOT increment attempt count (this isn't an error/retry)
        // - Small delay (50ms) prevents busy-spin while still being responsive
        // - Leverages existing all-deferred starvation prevention
        //
        // Formally verified in formal/tla/AsyncShardSync.tla
        if shared
            .storage
            .is_prefix_shard_syncing(&job.prefix, job.order)
        {
            // Shard is syncing - defer without incrementing retry count
            let deferred_job = Job {
                url: Arc::clone(&job.url),
                prefix: Arc::clone(&job.prefix),
                order: job.order,
                attempt: job.attempt,       // NO increment (not an error)
                backoff_ms: job.backoff_ms, // NO change
                ready_at: Some(Instant::now() + Duration::from_millis(50)), // Small delay
            };

            log::trace!(
                "Worker {} deferring {} (order {}) - shard syncing",
                worker_id,
                job.prefix,
                job.order
            );

            let _ = job_tx.send(deferred_job).await; // Back to primary queue
            consecutive_deferred += 1;

            // Use existing starvation prevention mechanism
            let queue_size = shared.queue_size.load(Ordering::SeqCst);
            if queue_size > 0 && consecutive_deferred >= queue_size {
                // All jobs deferred (all targeting syncing shards) - wait briefly
                let jitter =
                    Duration::from_millis((worker_id as u64 * 10) + (rand::random::<u64>() % 100));
                log::debug!(
                    "Worker {} blocking {}ms - all {} jobs targeting syncing shards",
                    worker_id,
                    jitter.as_millis(),
                    queue_size
                );
                tokio::time::sleep(jitter).await;
                consecutive_deferred = 0;
            }
            continue;
        }

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
            attempt: job.attempt,
        });

        // Single attempt - no blocking retry loop
        // On retryable error, requeue with ready_at set and pick up next job
        let start_time = Instant::now();
        let result = if shared.config.cache_files {
            process_single_attempt_cached(&job, &shared, worker_id).await
        } else {
            process_single_attempt(&job, &shared, worker_id).await
        };
        let elapsed = start_time.elapsed();

        match result {
            Ok(count) => {
                // Job completed successfully - decrement queue size
                shared.queue_size.fetch_sub(1, Ordering::SeqCst);

                // Success - send completion update and result
                let _ = shared.progress_tx.try_send(WorkerUpdate::Finished {
                    worker_id,
                    order: job.order,
                    prefix: job.prefix.clone(),
                    ngram_count: count,
                    duration: elapsed,
                });
                let job_result = JobResult {
                    order: job.order,
                    prefix: job.prefix,
                    outcome: JobOutcome::Success { ngram_count: count },
                };
                if result_tx.send(job_result).await.is_err() {
                    // Main task dropped, exit worker
                    let _ = worker_exit_tx.send(worker_id).await;
                    let _ = shared
                        .progress_tx
                        .try_send(WorkerUpdate::Exited { worker_id });
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
                let delay_ms = retry_job
                    .ready_at
                    .map(|ra| {
                        ra.saturating_duration_since(std::time::Instant::now())
                            .as_millis() as u64
                    })
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
                    if retry_after.is_some() {
                        " (from Retry-After header)"
                    } else {
                        ""
                    },
                    debug_info.url,
                    debug_info.error_message,
                    debug_info.status_code,
                    debug_info.response_time_ms,
                );

                // Emit deferred event (using Retrying for UI compatibility)
                let _ = shared.progress_tx.try_send(WorkerUpdate::Retrying {
                    worker_id,
                    order: retry_job.order,
                    prefix: Arc::clone(&retry_job.prefix),
                    attempt: retry_job.attempt as u32,
                    error: Arc::from(e.to_string()),
                });

                // Requeue with ready_at set - will be picked up after delay
                // NOTE: Do NOT increment queue_size here. The job was never "completed"
                // so it still counts as a pending job in the logical queue.
                let _ = job_tx.send(retry_job).await;

                // Worker immediately picks up next job (non-blocking)
            }
            Err(error) => {
                // Non-retryable error or max retries exceeded - skip for this session
                // Job completed (with error) - decrement queue size
                shared.queue_size.fetch_sub(1, Ordering::SeqCst);

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
                        let _ = shared
                            .progress_tx
                            .try_send(WorkerUpdate::Exited { worker_id });
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
                        let _ = shared
                            .progress_tx
                            .try_send(WorkerUpdate::Exited { worker_id });
                        return;
                    }
                }
            }
        }
    }

    // Notify main task that this worker is exiting (for active worker tracking)
    let _ = worker_exit_tx.send(worker_id).await;

    // Emit exited event so TUI can remove the worker from display
    let _ = shared
        .progress_tx
        .try_send(WorkerUpdate::Exited { worker_id });
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
/// * `ctx` - Shared processing context (config, storage, counters, http client,
///   progress channel, worker-ID pool). Constructed once per order at the call
///   site and shared via `Arc` to every spawned future.
/// * `url` - URL of the prefix file to download
/// * `prefix` - The prefix being downloaded (e.g., "th", "to")
/// * `order` - N-gram order (1-5)
/// * `attempt` - Current retry attempt (0 = first attempt)
/// * `backoff_ms` - Backoff delay in ms if this attempt fails (for next retry)
#[cfg(feature = "google-books")]
pub(super) async fn process_prefix_file(
    ctx: Arc<PrefixProcessingContext>,
    url: Arc<str>,
    prefix: Arc<str>,
    order: u8,
    attempt: u8,
    backoff_ms: u64,
) -> PrefixOutcome {
    use super::super::reader::HttpNgramReader;
    use tokio_stream::StreamExt;

    // Claim a worker ID from the pool - this blocks until a slot is available.
    // This ensures each concurrent worker has a unique ID for display purposes.
    let worker_id = {
        let mut rx = ctx.worker_id_pool_rx.lock().await;
        rx.recv().await.expect("Worker ID pool closed unexpectedly")
    };

    // Helper to return worker ID to pool (used on both success and error)
    let return_worker_id = |pool_tx: tokio::sync::mpsc::Sender<usize>, id: usize| async move {
        let _ = pool_tx.send(id).await;
    };

    // Send "Started" update (always include attempt for retry tracking)
    // Using try_send for backpressure - dropping updates is acceptable for progress
    if let Some(ref tx) = ctx.progress_tx {
        let _ = tx.try_send(WorkerUpdate::Started {
            worker_id,
            order,
            prefix: Arc::clone(&prefix),
            attempt,
        });
    }

    // Branch to cached processing if enabled
    if ctx.config.cache_files {
        let outcome =
            process_prefix_file_cached(&ctx, worker_id, url, prefix, order, attempt, backoff_ms)
                .await;

        // Return worker ID to pool
        return_worker_id(ctx.worker_id_pool_tx.clone(), worker_id).await;
        return outcome;
    }

    // Add small random delay to stagger connection starts (reduces rate limiting)
    let jitter_ms = rand::random::<u64>() % 500;
    tokio::time::sleep(Duration::from_millis(jitter_ms)).await;

    // Track processing time (after jitter delay)
    let start_time = Instant::now();

    // Single attempt processing with transaction-based atomicity
    let result: Result<u64, ImportError> = async {
        let mut reader =
            HttpNgramReader::with_options(&url, ctx.config.skip_pos_tags, ctx.config.min_count);

        // Stream n-grams instead of buffering entire file in memory.
        // This is critical for large 2-gram files (50-100M n-grams, 6-8GB).
        let stream = reader.stream_aggregated(ctx.config.year_range);

        // Try to begin a transaction for atomic, idempotent import (sharded mode only)
        let maybe_tx = ctx.storage.begin_prefix_tx(&prefix, order)?;

        if let Some(tx) = maybe_tx {
            // Sharded mode: delegate chunked-tx body to shared helper
            process_aggregated_stream(stream, tx, &ctx, &prefix, order, worker_id, "prefix").await
        } else {
            // Single-trie mode: use original increment-based approach
            // Local counters for batched atomic updates (reduces cache-line bouncing)
            tokio::pin!(stream);
            const NGRAM_PROGRESS_INTERVAL: u64 = 50_000;
            let mut local_total: u64 = 0;
            let mut local_unique: u64 = 0;

            let mut count = 0u64;
            while let Some(result) = stream.next().await {
                let agg = result?;
                let storage_result = store_ngram_shared(&agg.ngram, agg.total_count, &ctx.storage)?;
                count += 1;
                local_total += 1;
                if storage_result.is_new {
                    local_unique += 1;
                }

                // Batch flush atomic counters every COUNTER_BATCH_SIZE n-grams
                if local_total >= COUNTER_BATCH_SIZE {
                    ctx.total_ngrams.fetch_add(local_total, Ordering::Relaxed);
                    if local_unique > 0 {
                        ctx.unique_ngrams.fetch_add(local_unique, Ordering::Relaxed);
                    }
                    local_total = 0;
                    local_unique = 0;
                }

                // Emit periodic progress for TUI display
                if count % NGRAM_PROGRESS_INTERVAL == 0 {
                    if let Some(ref tx) = ctx.progress_tx {
                        let _ = tx.try_send(WorkerUpdate::NgramProgress {
                            worker_id,
                            ngram_count: count,
                        });
                    }
                }
            }

            // Flush remaining counts
            if local_total > 0 {
                ctx.total_ngrams.fetch_add(local_total, Ordering::Relaxed);
            }
            if local_unique > 0 {
                ctx.unique_ngrams.fetch_add(local_unique, Ordering::Relaxed);
            }

            Ok(count)
        }
    }
    .await;

    // Return worker ID to pool before returning result
    return_worker_id(ctx.worker_id_pool_tx.clone(), worker_id).await;

    let elapsed = start_time.elapsed();

    match result {
        Ok(count) => {
            // Send "Finished" update
            if let Some(ref tx) = ctx.progress_tx {
                let _ = tx.try_send(WorkerUpdate::Finished {
                    worker_id,
                    order,
                    prefix: Arc::clone(&prefix),
                    ngram_count: count,
                    duration: elapsed,
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
                prefix,
                order,
                attempt + 1,
                e
            );
            if let Some(ref tx) = ctx.progress_tx {
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
            }
        }
        Err(e) => {
            // Non-retryable error or max retries exceeded
            tracing::warn!(
                "Prefix '{}' (order {}) failed permanently after {} attempts: {}",
                prefix,
                order,
                attempt + 1,
                e
            );
            PrefixOutcome::Failed {
                prefix,
                error: e,
                attempts: (attempt + 1) as u32,
            }
        }
    }
}

/// Inner implementation for cached prefix file processing.
///
/// Called from `process_prefix_file` when `ctx.config.cache_files` is true.
/// Downloads the raw `.gz` file to a local cache, then streams from the
/// cached file. Cleans up the cache on both success and error.
///
/// The caller (`process_prefix_file`) is responsible for claiming the
/// `worker_id` from the pool and returning it after this function returns —
/// this function only needs the already-claimed `worker_id` for logging and
/// progress emission. All shared dependencies (storage, config, http_client,
/// counters, progress channel) come from `ctx`.
#[cfg(feature = "google-books")]
pub(super) async fn process_prefix_file_cached(
    ctx: &Arc<PrefixProcessingContext>,
    worker_id: usize,
    url: Arc<str>,
    prefix: Arc<str>,
    order: u8,
    attempt: u8,
    backoff_ms: u64,
) -> PrefixOutcome {
    use super::super::reader::stream_aggregated_from_cached_file;
    use tokio_stream::StreamExt;

    // Compute cache path
    let cache_path = match ctx.config.cache_file_path(order, &prefix) {
        Some(p) => p,
        None => {
            return PrefixOutcome::Failed {
                prefix,
                error: ImportError::Config(format!(
                    "Unknown language '{}' for cache file path",
                    ctx.config.language
                )),
                attempts: (attempt + 1) as u32,
            };
        }
    };

    // Reuse the shared HTTP client (single connection pool across all spawned
    // futures for this order's import — avoids the concurrency-amplification
    // rate-limiting bug previously caused by per-call `Client::builder()`).
    // Cloning is cheap: `reqwest::Client` is internally an `Arc`.
    let client = ctx.http_client.clone();

    // Download to cache (skips if already cached)
    if let Err(e) = download_to_cache(&url, &cache_path, &client).await {
        // Check if retryable
        if attempt < MAX_RETRIES && is_retryable_error(&e) {
            let next_backoff_ms = backoff_ms * 2;
            if let Some(ref tx) = ctx.progress_tx {
                let _ = tx.try_send(WorkerUpdate::Deferred {
                    worker_id,
                    order,
                    prefix: Arc::clone(&prefix),
                    attempt: (attempt + 1) as u32,
                    delay_seconds: backoff_ms / 1000,
                    error: Arc::from(e.to_string()),
                });
            }
            return PrefixOutcome::Deferred {
                url,
                prefix,
                order,
                attempt: attempt + 1,
                backoff_ms: next_backoff_ms,
            };
        }
        return PrefixOutcome::Failed {
            prefix,
            error: e,
            attempts: (attempt + 1) as u32,
        };
    }

    // Track processing time
    let start_time = Instant::now();

    // Stream from cached file
    let result: Result<u64, ImportError> = async {
        let stream = stream_aggregated_from_cached_file(
            &cache_path,
            ctx.config.year_range,
            ctx.config.skip_pos_tags,
            ctx.config.min_count,
        );

        let maybe_tx = ctx.storage.begin_prefix_tx(&prefix, order)?;

        if let Some(tx) = maybe_tx {
            // Sharded mode: delegate chunked-tx body to shared helper
            process_aggregated_stream(stream, tx, &ctx, &prefix, order, worker_id, "cached prefix")
                .await
        } else {
            // Single-trie mode
            tokio::pin!(stream);
            const NGRAM_PROGRESS_INTERVAL: u64 = 50_000;
            let mut local_total: u64 = 0;
            let mut local_unique: u64 = 0;
            let mut count = 0u64;

            while let Some(result) = stream.next().await {
                let agg = result?;
                let storage_result = store_ngram_shared(&agg.ngram, agg.total_count, &ctx.storage)?;
                count += 1;
                local_total += 1;
                if storage_result.is_new {
                    local_unique += 1;
                }

                if local_total >= COUNTER_BATCH_SIZE {
                    ctx.total_ngrams.fetch_add(local_total, Ordering::Relaxed);
                    if local_unique > 0 {
                        ctx.unique_ngrams.fetch_add(local_unique, Ordering::Relaxed);
                    }
                    local_total = 0;
                    local_unique = 0;
                }

                if count % NGRAM_PROGRESS_INTERVAL == 0 {
                    if let Some(ref ptx) = ctx.progress_tx {
                        let _ = ptx.try_send(WorkerUpdate::NgramProgress {
                            worker_id,
                            ngram_count: count,
                        });
                    }
                }
            }

            if local_total > 0 {
                ctx.total_ngrams.fetch_add(local_total, Ordering::Relaxed);
            }
            if local_unique > 0 {
                ctx.unique_ngrams.fetch_add(local_unique, Ordering::Relaxed);
            }

            Ok(count)
        }
    }
    .await;

    // Clean up cached file on both success and error
    cleanup_cache_file(&cache_path).await;

    let elapsed = start_time.elapsed();

    match result {
        Ok(count) => {
            if let Some(ref tx) = ctx.progress_tx {
                let _ = tx.try_send(WorkerUpdate::Finished {
                    worker_id,
                    order,
                    prefix: Arc::clone(&prefix),
                    ngram_count: count,
                    duration: elapsed,
                });
            }
            PrefixOutcome::Success {
                prefix,
                ngram_count: count,
            }
        }
        Err(e) if attempt < MAX_RETRIES && is_retryable_error(&e) => {
            let next_backoff_ms = backoff_ms * 2;
            if let Some(ref tx) = ctx.progress_tx {
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
            }
        }
        Err(e) => PrefixOutcome::Failed {
            prefix,
            error: e,
            attempts: (attempt + 1) as u32,
        },
    }
}
