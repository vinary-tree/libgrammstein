//! Google Books N-gram importer.
//!
//! Orchestrates the import process from Google Books N-grams into a PersistentARTrie,
//! with checkpoint/resume support for long-running imports.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use libdictenstein::persistent_artrie::PersistentARTrie;
use parking_lot::RwLock;

use super::sharding::{MergeCoordinator, MknAggregator};
use super::storage::{NgramStorage, StoragePrefixTx};
use crate::ngram::vocabulary::{
    decode_ngram_key_bytes, encode_indices_to_key_bytes,
    open_or_create_concurrent_vocabulary_lockfree_with_capacity,
};


use super::aggregator::YearAggregator;
use super::checkpoint::{
    CheckpointError, ImportCheckpoint, MknPhase, TrieCheckpointStorage,
};
use super::config::GoogleBooksConfig;
use super::events::{ImportCommand, ImportEvent, LogLevel};
use super::languages::{get_file_url, get_metadata, get_prefixes, is_supported};
use super::reader::{FileNgramReader, ReaderError};
use super::state_machine::CleanupResources;
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

/// Estimate the number of unique vocabulary terms for a given configuration.
///
/// The vocabulary size approximates the number of unique words in the corpus.
/// For Google Books, the 1-gram count is a good proxy since each unique 1-gram
/// is a unique word. Higher-order n-grams share the same vocabulary, so
/// `config.order` is intentionally not used here — only language and min_count
/// influence the estimate.
///
/// The min_count filter is applied since rare words below the threshold
/// are never inserted into the vocabulary.
fn estimate_vocabulary_size(config: &GoogleBooksConfig) -> usize {
    // Base vocabulary sizes by language (unique 1-gram count)
    let base_vocab = match config.language.as_str() {
        "en" | "eng" => 13_000_000usize,
        _ => 5_000_000usize,
    };

    // Apply min_count filter: higher thresholds prune rare words
    let factor = match config.min_count {
        0..=1 => 1.0,
        2..=10 => 0.4,
        11..=40 => 0.2,
        41..=100 => 0.1,
        _ => 0.05,
    };

    (base_vocab as f64 * factor) as usize
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
    // Store using ngram-string API (splits to SmallVec internally, avoiding heap alloc)
    let is_new = storage.store_ngram(ngram, count)?;

    Ok(NgramStorageResult { is_new })
}

/// Legacy version for direct trie access (used during migration).
#[allow(dead_code)]
fn store_ngram_shared_legacy(
    ngram: &str,
    count: u64,
    trie: &Arc<RwLock<PersistentARTrie<u64>>>,
) -> Result<NgramStorageResult, ImportError> {
    let mut trie_guard = trie.write();
    let is_new = trie_guard.get_value_bytes(ngram.as_bytes()).is_none();
    trie_guard.increment_bytes(ngram.as_bytes(), count as i64).map_err(|e| {
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

impl TrieCheckpointStorage for PersistentARTrie<u64> {
    type Error = TrieCheckpointError;

    fn store_checkpoint_u64(&mut self, key: &str, value: u64) -> Result<(), Self::Error> {
        self.upsert_bytes(key.as_bytes(), value)
            .map_err(|e| TrieCheckpointError::TrieError(e.to_string()))?;
        Ok(())
    }

    fn load_checkpoint_u64(&self, key: &str) -> Result<Option<u64>, Self::Error> {
        Ok(self.get_value_bytes(key.as_bytes()))
    }

    fn delete_checkpoint_key(&mut self, key: &str) -> Result<bool, Self::Error> {
        Ok(self.remove(key))
    }

    fn delete_checkpoint_prefix(&mut self, prefix: &str) -> Result<usize, Self::Error> {
        Ok(self.remove_prefix(prefix.as_bytes()))
    }

    fn iter_checkpoint_prefix(&self, prefix: &str) -> Result<Vec<(String, u64)>, Self::Error> {
        match self.iter_prefix_with_values(prefix.as_bytes()) {
            Some(iter) => Ok(iter
                .map(|(k, v)| (String::from_utf8_lossy(&k).into_owned(), v))
                .collect()),
            None => Ok(Vec::new()),
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
    /// Per-worker packed stats for non-blocking, race-free sampling.
    /// Each AtomicU64 packs: upper 32 bits = total n-grams, lower 32 bits = unique n-grams.
    /// Single atomic ensures both counts are read/written atomically together.
    /// Maximum workers supported: length of this Vec.
    worker_stats: Vec<AtomicU64>,
    /// Shared HTTP client for connection pooling and HTTP/2 multiplexing.
    /// Creating one client and sharing it across workers avoids the concurrency
    /// amplification bug where each worker creates independent connection pools,
    /// causing Google to see a spike in connections and trigger rate limiting.
    http_client: reqwest::Client,
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
    S: tokio_stream::Stream<Item = Result<super::aggregator::AggregatedNgram, ReaderError>>,
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
        if let Err(e) = ctx.storage.tx_insert_ngram(&mut tx, &agg.ngram, agg.total_count) {
            stream_err = Some(e.into());
            break;
        }
        count += 1;
        chunk_count += 1;

        // Chunked commit: bound per-transaction memory for large files
        if tx_chunk_size > 0 && chunk_count >= tx_chunk_size {
            match ctx.storage.commit_and_renew_prefix_tx(&mut tx, prefix, order) {
                Ok(committed) => {
                    log::trace!(
                        "Worker {}: committed chunk for {} '{}' ({} n-grams)",
                        worker_id, source_label, prefix, committed
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
                worker_id, source_label, prefix, abort_err
            );
        }
        return Err(e);
    }

    // Commit the final chunk and mark prefix as complete
    let committed = ctx.storage.commit_prefix_tx(tx)?;
    ctx.total_ngrams.fetch_add(count, Ordering::Relaxed);
    ctx.unique_ngrams.fetch_add(committed as u64, Ordering::Relaxed);
    log::trace!(
        "Worker {}: committed {} '{}' with {} n-grams ({} inserted)",
        worker_id, source_label, prefix, count, committed
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

    // Use the shared HTTP client for connection pooling and HTTP/2 multiplexing
    let stream = reader.stream_aggregated_with_client(
        shared.config.year_range,
        Some(shared.http_client.clone()),
    );
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
                    Err(e) => { stream_err = Some(e.into()); break; }
                };

                // Insert into transaction (SET semantics, not increment)
                // tx_insert_ngram splits to SmallVec internally, avoiding heap alloc
                if let Err(e) = shared.storage.tx_insert_ngram(&mut tx, &agg.ngram, agg.total_count) {
                    stream_err = Some(e.into());
                    break;
                }
                count += 1;
                chunk_count += 1;

                // Chunked commit: bound per-transaction memory for large files
                if tx_chunk_size > 0 && chunk_count >= tx_chunk_size {
                    match shared.storage.commit_and_renew_prefix_tx(&mut tx, &job.prefix, job.order) {
                        Ok(committed) => {
                            log::trace!(
                                "Worker {}: committed chunk for prefix '{}' ({} n-grams)",
                                worker_id, job.prefix, committed
                            );
                            chunk_count = 0;
                        }
                        Err(e) => { stream_err = Some(e.into()); break; }
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
                        worker_id, job.prefix, abort_err
                    );
                }
                return Err(e);
            }

            // Commit the final chunk and mark prefix as complete
            let committed = shared.storage.commit_prefix_tx(tx)?;
            log::trace!(
                "Worker {}: committed prefix '{}' with {} n-grams",
                worker_id, job.prefix, committed
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
            let storage_result = store_ngram_shared(
                &agg.ngram,
                agg.total_count,
                &shared.storage,
            )?;
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
            shared.unique_ngrams.fetch_add(unique_count, Ordering::Relaxed);
        }

        Ok(count)
    };

    // Final flush to global counters (for checkpoint persistence)
    if let Ok(ngram_count) = result {
        shared.total_ngrams.fetch_add(ngram_count, Ordering::Relaxed);
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
    use super::reader::stream_aggregated_from_cached_file;
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
                    Err(e) => { stream_err = Some(e.into()); break; }
                };
                if let Err(e) = shared.storage.tx_insert_ngram(&mut tx, &agg.ngram, agg.total_count) {
                    stream_err = Some(e.into());
                    break;
                }
                count += 1;
                chunk_count += 1;

                // Chunked commit: bound per-transaction memory for large files
                if tx_chunk_size > 0 && chunk_count >= tx_chunk_size {
                    match shared.storage.commit_and_renew_prefix_tx(&mut tx, &job.prefix, job.order) {
                        Ok(committed) => {
                            log::trace!(
                                "Worker {}: committed chunk for cached prefix '{}' ({} n-grams)",
                                worker_id, job.prefix, committed
                            );
                            chunk_count = 0;
                        }
                        Err(e) => { stream_err = Some(e.into()); break; }
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
                        worker_id, job.prefix, abort_err
                    );
                }
                return Err(e);
            }

            let committed = shared.storage.commit_prefix_tx(tx)?;
            log::trace!(
                "Worker {}: committed cached prefix '{}' with {} n-grams",
                worker_id, job.prefix, committed
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
            let storage_result = store_ngram_shared(
                &agg.ngram,
                agg.total_count,
                &shared.storage,
            )?;
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
            shared.unique_ngrams.fetch_add(unique_count, Ordering::Relaxed);
        }

        Ok(count)
    };

    // Clean up cached file on both success and error
    // On success: no longer needed. On error: will re-download on retry.
    cleanup_cache_file(&cache_path).await;

    // Final flush to global counters
    if let Ok(ngram_count) = result {
        shared.total_ngrams.fetch_add(ngram_count, Ordering::Relaxed);
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
async fn worker_task(
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
                        // Add per-worker jitter to prevent thundering herd when all workers
                        // wake up simultaneously after all-deferred sleep
                        let jitter = Duration::from_millis(
                            (worker_id as u64 * 100) + (rand::random::<u64>() % 500)
                        );
                        let staggered_wait = wait + jitter;
                        log::debug!(
                            "Worker {} blocking {}ms (+{}ms jitter) - all {} jobs deferred",
                            worker_id, wait.as_millis(), jitter.as_millis(), queue_size
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
        if shared.storage.is_prefix_shard_syncing(&job.prefix, job.order) {
            // Shard is syncing - defer without incrementing retry count
            let deferred_job = Job {
                url: Arc::clone(&job.url),
                prefix: Arc::clone(&job.prefix),
                order: job.order,
                attempt: job.attempt,        // NO increment (not an error)
                backoff_ms: job.backoff_ms,  // NO change
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
                let jitter = Duration::from_millis(
                    (worker_id as u64 * 10) + (rand::random::<u64>() % 100)
                );
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
/// * `ctx` - Shared processing context (config, storage, counters, http client,
///   progress channel, worker-ID pool). Constructed once per order at the call
///   site and shared via `Arc` to every spawned future.
/// * `url` - URL of the prefix file to download
/// * `prefix` - The prefix being downloaded (e.g., "th", "to")
/// * `order` - N-gram order (1-5)
/// * `attempt` - Current retry attempt (0 = first attempt)
/// * `backoff_ms` - Backoff delay in ms if this attempt fails (for next retry)
#[cfg(feature = "google-books")]
async fn process_prefix_file(
    ctx: Arc<PrefixProcessingContext>,
    url: Arc<str>,
    prefix: Arc<str>,
    order: u8,
    attempt: u8,
    backoff_ms: u64,
) -> PrefixOutcome {
    use super::reader::HttpNgramReader;
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
        let outcome = process_prefix_file_cached(
            &ctx,
            worker_id,
            url,
            prefix,
            order,
            attempt,
            backoff_ms,
        )
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
        let mut reader = HttpNgramReader::with_options(
            &url,
            ctx.config.skip_pos_tags,
            ctx.config.min_count,
        );

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
                let storage_result = store_ngram_shared(
                    &agg.ngram,
                    agg.total_count,
                    &ctx.storage,
                )?;
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
                prefix, order, attempt + 1, e
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
async fn process_prefix_file_cached(
    ctx: &Arc<PrefixProcessingContext>,
    worker_id: usize,
    url: Arc<str>,
    prefix: Arc<str>,
    order: u8,
    attempt: u8,
    backoff_ms: u64,
) -> PrefixOutcome {
    use super::reader::stream_aggregated_from_cached_file;
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
                error: e,
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
            process_aggregated_stream(stream, tx, &ctx, &prefix, order, worker_id, "cached prefix").await
        } else {
            // Single-trie mode
            tokio::pin!(stream);
            const NGRAM_PROGRESS_INTERVAL: u64 = 50_000;
            let mut local_total: u64 = 0;
            let mut local_unique: u64 = 0;
            let mut count = 0u64;

            while let Some(result) = stream.next().await {
                let agg = result?;
                let storage_result = store_ngram_shared(
                    &agg.ngram,
                    agg.total_count,
                    &ctx.storage,
                )?;
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
                error: e,
            }
        }
        Err(e) => {
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
        /// Retry attempt number (0 = first attempt, 1+ = retry).
        attempt: u8,
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
        /// Time taken to process this file.
        duration: Duration,
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
        /// N-gram order being retried.
        order: u8,
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

    /// Storage error.
    #[error("Storage error: {0}")]
    Storage(#[from] super::storage::StorageError),
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
    /// Can be single-trie (original behavior) or sharded storage. The storage
    /// also owns the checkpoint-metadata trie (see
    /// `NgramStorage::checkpoint_trie`), so the importer no longer needs a
    /// separate `trie` field.
    storage: Arc<NgramStorage>,

    /// Lock-free overlay flush threshold (entries per shard).
    ///
    /// When a shard's lock-free entry count exceeds this threshold, its
    /// overlay is flushed to the persistent trie. This bounds memory usage
    /// during high-parallelism imports where millions of entries can
    /// accumulate in lock-free overlays between checkpoints.
    ///
    /// Default: auto-scaled based on `parallel_downloads` (50K for >=8
    /// workers, 100K otherwise).
    lockfree_flush_threshold: u64,
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

        // Estimate vocabulary size (unique words) for pre-allocation
        let estimated_vocab = estimate_vocabulary_size(&config);
        log::info!("Estimated vocabulary size: {}", estimated_vocab);

        // Create or open lock-free concurrent vocabulary for compact encoding
        // Pre-sizes the lock-free layer to avoid DashMap/Vec resize spikes
        let vocabulary_path = config.vocabulary_path();
        log::info!("Using vocabulary at {:?}", vocabulary_path);
        let vocabulary = open_or_create_concurrent_vocabulary_lockfree_with_capacity(
            &vocabulary_path,
            estimated_vocab,
        ).map_err(|e| {
            ImportError::Trie(format!("Failed to create/open vocabulary: {}", e))
        })?;

        // Create storage backend with vocabulary for compact encoding
        let storage = NgramStorage::resume_or_start_with_vocabulary(
            &config,
            estimated_ngrams,
            Some(vocabulary),
        ).map_err(|e| {
            ImportError::Trie(format!("Failed to create storage: {}", e))
        })?;

        // Log storage mode and vocabulary status
        if storage.is_sharded() {
            log::info!("Using sharded storage with vocabulary-indexed encoding");
        } else {
            log::info!("Using single-trie storage with vocabulary-indexed encoding");
        }

        // (The checkpoint-metadata trie is now owned by NgramStorage; see
        // NgramStorage::checkpoint_trie. The importer no longer maintains
        // its own auxiliary trie.)

        // Auto-scale flush threshold: lower for high parallelism to bound memory
        let lockfree_flush_threshold = if config.parallel_downloads >= 8 {
            50_000
        } else {
            100_000
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
            lockfree_flush_threshold,
        })
    }

    /// Set the lock-free overlay flush threshold (entries per shard).
    ///
    /// This overrides the auto-scaled default. Lower values use less memory
    /// but flush more frequently (slightly reducing throughput). Higher values
    /// use more memory but flush less often.
    ///
    /// Typical values:
    /// - 10_000–25_000: Very memory-constrained environments
    /// - 50_000: Default for >=8 parallel workers
    /// - 100_000: Default for <8 parallel workers
    /// - 200_000+: Large-memory systems with fast storage
    pub fn set_lockfree_flush_threshold(&mut self, threshold: u64) {
        self.lockfree_flush_threshold = threshold;
        log::info!("Lock-free flush threshold set to {} entries per shard", threshold);
    }

    /// Get the current lock-free overlay flush threshold.
    pub fn lockfree_flush_threshold(&self) -> u64 {
        self.lockfree_flush_threshold
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
    ///
    /// **Safety Check**: If a checkpoint exists but the vocabulary WAL is
    /// unexpectedly large (> 1MB), this indicates a previous checkpoint
    /// didn't properly flush the vocabulary. A warning is logged.
    pub fn resume_or_start(config: GoogleBooksConfig) -> Result<Self, ImportError> {
        let checkpoint_path = config.output_path.with_extension("checkpoint.json");
        let vocabulary_path = config.vocabulary_path();

        // Check for vocabulary WAL inconsistency before proceeding
        Self::check_vocabulary_wal_consistency(&vocabulary_path, &checkpoint_path);

        // First, create the importer to get access to the trie
        let mut importer = Self::new(config)?;

        // Try to load checkpoint from the storage's checkpoint trie first
        // (more reliable than the JSON fallback).
        let trie_checkpoint = importer.storage.load_import_checkpoint()?;

        if let Some(checkpoint) = trie_checkpoint {
            log::info!(
                "Resuming from trie checkpoint: {} orders in progress, {} total prefixes completed",
                checkpoint.orders_in_progress().len(),
                checkpoint.total_completed_prefix_count()
            );

            importer.checkpoint = checkpoint;

            // Recover in-progress prefixes as failed for retry (crash recovery).
            // This aligns with CheckpointStateMachine.tla CrashRecoverySound property:
            // on resume, in-progress prefixes must be moved to failed state since
            // they may have partial data that needs cleanup before retry.
            for order in importer.config.orders.clone() {
                let in_progress = importer.checkpoint.in_progress_prefixes(order);
                if !in_progress.is_empty() {
                    log::warn!(
                        "Order {}: recovering {} in-progress prefixes as failed for retry: {:?}",
                        order,
                        in_progress.len(),
                        in_progress
                    );
                    importer.checkpoint.recover_in_progress_as_failed(order);
                }
            }

            // CRITICAL: Reconcile importer checkpoint with shard state.
            // Verify that prefixes marked complete in the importer checkpoint
            // actually have data in the shards. If not, mark them for retry.
            // This handles the case where the importer checkpoint was saved but
            // shard data was lost (e.g., due to OS buffer cache not being flushed).
            if let Some(coordinator) = importer.storage.as_sharded() {
                let mut reconciled_count = 0usize;

                for order in importer.config.orders.clone() {
                    // Get completed prefixes from shard state (authoritative)
                    let shard_completed = coordinator.completed_prefixes_for_order(order);

                    // Get completed prefixes from importer checkpoint
                    let importer_completed: Vec<String> = importer
                        .checkpoint
                        .order_progress
                        .get(&order)
                        .map(|p| p.completed_prefixes().cloned().collect())
                        .unwrap_or_default();

                    // Check each prefix marked complete in importer checkpoint
                    for prefix in importer_completed {
                        if !shard_completed.contains(&prefix) {
                            log::warn!(
                                "Order {}: prefix '{}' marked complete in importer checkpoint but \
                                 not found in shard state - marking for retry",
                                order,
                                prefix
                            );
                            // Mark as failed so it will be retried
                            importer.checkpoint.fail_prefix(order, &prefix);
                            reconciled_count += 1;
                        }
                    }
                }

                if reconciled_count > 0 {
                    log::warn!(
                        "Reconciliation: {} prefixes marked for retry due to missing shard data",
                        reconciled_count
                    );
                }
            }

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

            // Recover in-progress prefixes as failed for retry (crash recovery).
            // This aligns with CheckpointStateMachine.tla CrashRecoverySound property:
            // on resume, in-progress prefixes must be moved to failed state since
            // they may have partial data that needs cleanup before retry.
            for order in importer.config.orders.clone() {
                let in_progress = importer.checkpoint.in_progress_prefixes(order);
                if !in_progress.is_empty() {
                    log::warn!(
                        "Order {}: recovering {} in-progress prefixes as failed for retry: {:?}",
                        order,
                        in_progress.len(),
                        in_progress
                    );
                    importer.checkpoint.recover_in_progress_as_failed(order);
                }
            }

            // CRITICAL: Reconcile importer checkpoint with shard state.
            // (Same logic as trie checkpoint case above)
            if let Some(coordinator) = importer.storage.as_sharded() {
                let mut reconciled_count = 0usize;

                for order in importer.config.orders.clone() {
                    let shard_completed = coordinator.completed_prefixes_for_order(order);

                    let importer_completed: Vec<String> = importer
                        .checkpoint
                        .order_progress
                        .get(&order)
                        .map(|p| p.completed_prefixes().cloned().collect())
                        .unwrap_or_default();

                    for prefix in importer_completed {
                        if !shard_completed.contains(&prefix) {
                            log::warn!(
                                "Order {}: prefix '{}' marked complete in importer checkpoint but \
                                 not found in shard state - marking for retry",
                                order,
                                prefix
                            );
                            importer.checkpoint.fail_prefix(order, &prefix);
                            reconciled_count += 1;
                        }
                    }
                }

                if reconciled_count > 0 {
                    log::warn!(
                        "Reconciliation: {} prefixes marked for retry due to missing shard data",
                        reconciled_count
                    );
                }
            }

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
            importer
                .storage
                .save_import_checkpoint_async(&importer.checkpoint)
                .map_err(|e| {
                    ImportError::Trie(format!("Failed to migrate checkpoint to trie: {}", e))
                })?;

            return Ok(importer);
        }

        // No checkpoint exists - fresh start
        Ok(importer)
    }

    /// Check for vocabulary WAL consistency issues.
    ///
    /// If a checkpoint exists but the vocabulary WAL is unexpectedly large,
    /// this indicates a previous checkpoint didn't properly flush the vocabulary.
    /// This can lead to index inconsistency on resume.
    ///
    /// **Warning threshold**: 1 MB (WAL files should be ~64 bytes when checkpointed)
    fn check_vocabulary_wal_consistency(
        vocabulary_path: &Path,
        checkpoint_path: &Path,
    ) {
        // Only check if a checkpoint exists (indicating a resume scenario)
        let checkpoint_trie_path = checkpoint_path.with_extension("checkpoint.artrie");
        let has_checkpoint = checkpoint_path.exists() || checkpoint_trie_path.exists();

        if !has_checkpoint {
            return; // Fresh start, no need to check
        }

        // Check vocabulary WAL size
        let vocab_wal_path = vocabulary_path.with_extension("vocab.wal");
        let vocab_wal_path2 = {
            let mut p = vocabulary_path.to_path_buf();
            p.set_extension("wal");
            p
        };

        // Try both possible WAL paths
        for wal_path in [vocab_wal_path, vocab_wal_path2] {
            if wal_path.exists() {
                if let Ok(metadata) = std::fs::metadata(&wal_path) {
                    let size = metadata.len();
                    const WARNING_THRESHOLD: u64 = 1_000_000; // 1 MB

                    if size > WARNING_THRESHOLD {
                        log::warn!(
                            "VOCABULARY WAL INCONSISTENCY DETECTED: {} is {} bytes",
                            wal_path.display(),
                            size
                        );
                        log::warn!(
                            "This indicates a previous checkpoint did not properly flush the vocabulary."
                        );
                        log::warn!(
                            "Resume may result in index inconsistency and duplicated n-gram counts."
                        );
                        log::warn!(
                            "Consider starting a fresh import or manually checkpointing the vocabulary."
                        );
                    }
                }
            }
        }
    }

    /// Signal the importer to stop gracefully.
    pub fn interrupt(&self) {
        self.interrupted.store(true, Ordering::Release);
    }

    /// Check if import was interrupted.
    pub fn is_interrupted(&self) -> bool {
        self.interrupted.load(Ordering::Acquire)
    }

    /// Get filtered prefixes for a given order, respecting the config's prefix filter.
    ///
    /// If `config.prefix` is set, returns only that prefix (if valid for this order).
    /// Otherwise returns all prefixes for the order.
    fn get_filtered_prefixes(&self, order: u8) -> Vec<String> {
        let all_prefixes = get_prefixes(order);
        match &self.config.prefix {
            Some(p) => {
                if all_prefixes.contains(p) {
                    vec![p.clone()]
                } else {
                    vec![] // Invalid prefix for this order - skip silently
                }
            }
            None => all_prefixes,
        }
    }

    /// Save current checkpoint.
    ///
    /// This persists both the trie data (via WAL checkpoint) and the import
    /// progress. The checkpoint data is stored in both:
    /// 1. The trie itself (with reserved key namespace for atomic consistency)
    /// 2. A JSON file (for backwards compatibility and easy inspection)
    ///
    /// The trie checkpoint truncates the WAL to prevent unbounded growth.
    ///
    /// **IMPORTANT**: This checkpoints both vocabulary and n-gram shards to ensure
    /// consistency on resume. The order of operations is:
    ///
    /// 1. Sync atomic counters from checkpoint stats
    /// 2. Sync and checkpoint vocabulary WAL
    /// 3. Sync and checkpoint n-gram shard WALs
    /// 4. Save checkpoint metadata to trie
    /// 5. Checkpoint metadata trie
    ///
    /// Without vocabulary checkpointing, an interrupted import can result in lost
    /// vocabulary mappings, causing the resumed import to re-index words with
    /// different indices.
    ///
    /// Without shard checkpointing, n-grams in shard WALs are replayed on resume,
    /// causing counts to double (since `increment()` accumulates values).
    pub fn save_checkpoint(&mut self) -> Result<(), ImportError> {
        self.save_checkpoint_with_parallelism(Self::DEFAULT_CHECKPOINT_PARALLELISM)
    }

    /// Default number of shards to sync in parallel during checkpoint.
    /// Set to 8 for good SSD performance without overwhelming I/O.
    const DEFAULT_CHECKPOINT_PARALLELISM: usize = 8;

    /// Save checkpoint with configurable parallelism for shard syncing.
    ///
    /// This is the core checkpoint implementation that:
    /// 1. Syncs atomic counters from checkpoint stats
    /// 2. Syncs and checkpoints vocabulary WAL (synchronous, single resource)
    /// 3. Syncs n-gram shard WALs in parallel
    /// 4. Checkpoints n-gram shards
    /// 5. Saves checkpoint metadata to trie
    /// 6. Checkpoints metadata trie
    ///
    /// Workers can continue on non-syncing shards during step 3, enabling
    /// non-blocking checkpoints that don't stall the entire import.
    ///
    /// # Arguments
    ///
    /// * `max_concurrent_syncs` - Maximum shards to sync in parallel.
    ///   Recommended: 8 for SSDs, 2 for HDDs.
    ///
    /// # Performance
    ///
    /// With 100 shards @ 50ms each:
    /// - Sequential: ~5000ms total blocking
    /// - Parallel (8 concurrent): ~625ms + workers continue on other shards
    pub fn save_checkpoint_with_parallelism(
        &mut self,
        max_concurrent_syncs: usize,
    ) -> Result<(), ImportError> {
        // Sync atomic counters FROM checkpoint stats (source of truth).
        // The checkpoint.add_ngrams() method maintains accurate counts incrementally.
        // We sync the atomics from checkpoint to keep real-time display consistent.
        self.total_ngrams.store(self.checkpoint.stats.ngrams_processed, Ordering::Relaxed);
        self.unique_ngrams.store(self.checkpoint.stats.unique_ngrams, Ordering::Relaxed);
        self.checkpoint.stats.elapsed_seconds = self.start_time.elapsed().as_secs();

        // CRITICAL: Merge vocabulary lock-free layer and rotate WAL FIRST to ensure
        // vocabulary indices are durable before the checkpoint marks prefixes as
        // completed. This prevents the bug where vocabulary entries are in the WAL
        // (not persisted) when the checkpoint claims prefixes are done, leading to
        // index inconsistency on resume.
        //
        // Uses merge_and_rotate_vocabulary_wal() instead of the previous
        // sync_vocabulary() + rotate_vocabulary_wal() pair. Both of those methods
        // called merge_into() internally, causing two back-to-back HashMap rebuilds
        // of the vocabulary's reverse_index (~3.42 GB transient spike for 5.8M words).
        // The combined method does a single merge, halving the peak memory usage.
        self.storage.merge_and_rotate_vocabulary_wal().map_err(|e| {
            ImportError::Trie(format!("Failed to merge and rotate vocabulary WAL: {}", e))
        })?;

        // CRITICAL: Sync and checkpoint n-gram shards to prevent WAL replay on resume.
        // Without this, n-grams written to shard WALs before a checkpoint are replayed
        // on resume, causing counts to double (since increment() accumulates values).
        //
        // Use parallel sync for non-blocking operation:
        // - Workers can continue on shards that aren't syncing
        // - Only workers targeting a syncing shard defer their job
        // - Formally verified in formal/tla/AsyncShardSync.tla
        self.storage.sync_parallel(max_concurrent_syncs).map_err(|e| {
            ImportError::Trie(format!("Failed to sync storage: {}", e))
        })?;
        self.storage.checkpoint_parallel(max_concurrent_syncs).map_err(|e| {
            ImportError::Trie(format!("Failed to checkpoint storage: {}", e))
        })?;

        // Save checkpoint to the storage's metadata trie AFTER syncing all
        // data. `save_import_checkpoint` writes the checkpoint keys then
        // flushes the trie (truncating its WAL), keeping data and progress
        // tracking consistent.
        self.storage
            .save_import_checkpoint(&self.checkpoint)
            .map_err(|e| {
                ImportError::Trie(format!("Failed to save checkpoint to trie: {}", e))
            })?;

        log::debug!("Checkpoint saved: {}", self.checkpoint.progress_summary());
        Ok(())
    }

    /// Save checkpoint using async WAL sync.
    ///
    /// This is the recommended checkpoint method for high-throughput workloads.
    /// It provides the same durability guarantees as `save_checkpoint()` but
    /// with minimal blocking:
    ///
    /// 1. Vocabulary checkpoint (synchronous - single resource)
    /// 2. Start async sync on all dirty shards (fast WAL rotation)
    /// 3. Wait for all syncs in parallel
    /// 4. Finish checkpoint (truncate WALs with bounded parallelism)
    ///
    /// # Performance
    ///
    /// With 100 shards at 50ms fsync each:
    /// - `save_checkpoint()`: ~5000ms blocking (sequential)
    /// - `save_checkpoint_async()`: ~50ms rotation + parallel wait
    pub fn save_checkpoint_async(&mut self) -> Result<(), ImportError> {
        self.save_checkpoint_async_with_events(None)
    }

    /// Save checkpoint with optional progress events.
    ///
    /// This variant accepts an optional broadcast sender for emitting
    /// `CheckpointProgress` events during the checkpoint operation.
    pub fn save_checkpoint_async_with_events(
        &mut self,
        event_tx: Option<&tokio::sync::broadcast::Sender<super::events::ImportEvent>>,
    ) -> Result<(), ImportError> {
        // Sync atomic counters FROM checkpoint stats (source of truth).
        self.total_ngrams.store(self.checkpoint.stats.ngrams_processed, Ordering::Relaxed);
        self.unique_ngrams.store(self.checkpoint.stats.unique_ngrams, Ordering::Relaxed);
        self.checkpoint.stats.elapsed_seconds = self.start_time.elapsed().as_secs();

        // CRITICAL: Rotate vocabulary WAL FIRST to ensure vocabulary indices are
        // durable before the checkpoint marks prefixes as completed.
        //
        // Note: We use rotate_vocabulary_wal() instead of checkpoint_vocabulary() to
        // avoid file bloat from repeated full trie serialization. WAL replay provides
        // crash recovery.
        self.storage.rotate_vocabulary_wal().map_err(|e| {
            ImportError::Trie(format!("Failed to rotate vocabulary WAL: {}", e))
        })?;

        // Start async checkpoint - this rotates WALs and returns immediately
        let handle = self.storage.checkpoint_async().map_err(|e| {
            ImportError::Trie(format!("Failed to start async checkpoint: {}", e))
        })?;

        log::debug!(
            "Async checkpoint initiated: {} resources rotating",
            handle.count()
        );

        // Wait for all syncs to complete using parallel waiting for sharded storage.
        // This reduces wait time from O(n) to O(1) for n shards by waiting on all
        // shard sync handles concurrently rather than sequentially.
        handle.wait_all_parallel().map_err(|e| {
            ImportError::Trie(format!("Async checkpoint sync failed: {}", e))
        })?;

        // Finish checkpoint - truncate WALs with bounded I/O parallelism
        // Create a progress callback that emits CheckpointProgress events
        let progress_callback: Option<Box<dyn Fn(usize, usize) + Send + Sync>> = event_tx.map(|tx| {
            let tx = tx.clone();
            Box::new(move |processed: usize, total: usize| {
                let percent = if total > 0 {
                    (processed as f32 / total as f32) * 100.0
                } else {
                    100.0
                };
                let _ = tx.send(super::events::ImportEvent::CheckpointProgress {
                    shards_processed: processed,
                    total_shards: total,
                    percent_complete: percent,
                });
            }) as Box<dyn Fn(usize, usize) + Send + Sync>
        });

        self.storage.checkpoint_async_finish_with_progress(Self::DEFAULT_CHECKPOINT_PARALLELISM, progress_callback).map_err(|e| {
            ImportError::Trie(format!("Failed to finish async checkpoint: {}", e))
        })?;

        // Save checkpoint metadata AFTER syncing all data
        // This ensures consistency between data and progress tracking.
        self.storage
            .save_import_checkpoint(&self.checkpoint)
            .map_err(|e| {
                ImportError::Trie(format!("Failed to save checkpoint to trie: {}", e))
            })?;

        log::debug!("Async checkpoint saved: {}", self.checkpoint.progress_summary());
        Ok(())
    }

    /// Delete checkpoint file and trie-based checkpoint data (call after successful completion).
    pub fn cleanup_checkpoint(&mut self) -> Result<(), ImportError> {
        // Delete JSON checkpoint
        ImportCheckpoint::delete(&self.checkpoint_path)?;

        // Delete trie-based checkpoint data via the storage's API
        self.storage.delete_import_checkpoint().map_err(|e| {
            ImportError::Trie(format!("Failed to delete checkpoint from trie: {}", e))
        })?;

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

            let prefixes = self.get_filtered_prefixes(order);
            if prefixes.is_empty() {
                // Prefix filter didn't match any valid prefix for this order
                log::debug!(
                    "Prefix filter {:?} not valid for order {}, skipping",
                    self.config.prefix,
                    order
                );
                continue;
            }
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

                // Mark completion in storage layer (important for sharded storage)
                if let Err(e) = self.storage.mark_prefix_completed(prefix, order) {
                    log::warn!("Failed to mark prefix {} as completed in storage: {}", prefix, e);
                }

                self.checkpoint.add_ngrams(order, ngrams_in_file);
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

                // Flush lock-free overlays for shards exceeding threshold
                // (lightweight: only acquires write locks on over-threshold shards)
                if let Err(e) = self.storage.flush_lockfree_over_threshold(self.lockfree_flush_threshold) {
                    log::warn!("Lock-free flush failed: {}", e);
                }

                // Save checkpoint periodically (async for better throughput)
                let checkpoint_interval: usize = if self.config.parallel_downloads >= 8 { 5 } else { 10 };
                if (idx + 1) % checkpoint_interval == 0 {
                    self.save_checkpoint_async()?;
                }
            }

            self.checkpoint.complete_order(order)?;
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
    ///             WorkerUpdate::Retrying { worker_id, order, prefix, attempt, error } => {
    ///                 println!("[{}] Retry {} (order {}): {} - {}", worker_id, attempt, order, prefix, error);
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

            let prefixes = self.get_filtered_prefixes(order);
            if prefixes.is_empty() {
                // Prefix filter didn't match any valid prefix for this order
                log::debug!(
                    "Prefix filter {:?} not valid for order {}, skipping",
                    self.config.prefix,
                    order
                );
                continue;
            }
            let total_files = prefixes.len() as u32;

            // Filter to only prefixes that need processing
            let pending_prefixes: Vec<String> = prefixes
                .iter()
                .filter(|p| self.checkpoint.needs_prefix(order, p))
                .cloned()
                .collect();

            if pending_prefixes.is_empty() {
                log::info!("Order {} already complete", order);
                self.checkpoint.complete_order(order)?;
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

            // Build one shared HTTP client for this order's import. All spawned
            // futures clone it (cheap — internally an Arc) so they share a single
            // connection pool. This avoids the concurrency-amplification rate-
            // limiting bug previously caused by per-call `Client::builder()` in
            // the cached path.
            let http_client = reqwest::Client::builder()
                .timeout(Duration::from_secs(300))
                .connect_timeout(Duration::from_secs(30))
                .read_timeout(Duration::from_secs(60))
                .pool_max_idle_per_host(4)
                .user_agent("Mozilla/5.0 (compatible; libgrammstein/0.1; +https://github.com/vinary-tree/libgrammstein)")
                .build()
                .expect("Failed to build shared HTTP client for prefix-file path");

            // Assemble shared context for all spawned prefix-file futures
            let prefix_ctx = Arc::new(PrefixProcessingContext {
                config: config.clone(),
                storage: Arc::clone(&storage),
                total_ngrams: Arc::clone(&total_ngrams),
                unique_ngrams: Arc::clone(&unique_ngrams),
                progress_tx: worker_updates.clone(),
                http_client,
                worker_id_pool_tx: worker_id_pool_tx.clone(),
                worker_id_pool_rx: Arc::clone(&worker_id_pool_rx),
            });

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
                            Arc::clone(&prefix_ctx),
                            url,
                            prefix,
                            order,
                            0,                    // First attempt
                            INITIAL_BACKOFF_MS,   // Initial backoff
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

                        // Mark completion in storage layer (important for sharded storage)
                        if let Err(e) = self.storage.mark_prefix_completed(&prefix, order) {
                            log::warn!("Failed to mark prefix {} as completed in storage: {}", prefix, e);
                        }

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

                // Flush lock-free overlays for shards exceeding threshold
                if let Err(e) = self.storage.flush_lockfree_over_threshold(self.lockfree_flush_threshold) {
                    log::warn!("Lock-free flush failed: {}", e);
                }

                // Save checkpoint periodically (async for better throughput)
                let checkpoint_interval: u32 = if self.config.parallel_downloads >= 8 { 5 } else { 10 };
                if completed_in_order % checkpoint_interval == 0 {
                    self.save_checkpoint_async()?;
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

                // Create futures for deferred items — reuse the same shared context
                let retry_futures: Vec<_> = deferred_items
                    .drain(..)
                    .map(|(url, prefix, o, attempt, backoff_ms)| {
                        process_prefix_file(
                            Arc::clone(&prefix_ctx),
                            url,
                            prefix,
                            o,
                            attempt,
                            backoff_ms,
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

                            // Mark completion in storage layer (important for sharded storage)
                            if let Err(e) = self.storage.mark_prefix_completed(&prefix, order) {
                                log::warn!("Failed to mark prefix {} as completed in storage: {}", prefix, e);
                            }

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

            self.checkpoint.complete_order(order)?;
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
        keep_shards: bool,
    ) -> Result<ImportStats, ImportError> {
        use futures::stream::StreamExt;
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
        let mut order_files_skipped: std::collections::HashMap<u8, u64> =
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

            let prefixes = self.get_filtered_prefixes(order);
            if prefixes.is_empty() {
                // Prefix filter didn't match any valid prefix for this order
                log::debug!(
                    "Prefix filter {:?} not valid for order {}, skipping",
                    self.config.prefix,
                    order
                );
                continue;
            }
            let total_files = prefixes.len() as u64;
            order_total_files.insert(order, total_files);

            // Filter to only prefixes that need processing
            let pending_count = prefixes
                .iter()
                .filter(|p| self.checkpoint.needs_prefix(order, p))
                .count() as u64;

            if pending_count == 0 {
                log::info!("Order {} already complete", order);
                self.checkpoint.complete_order(order)?;
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

            // Emit initial OrderProgress with checkpoint state for resume.
            // This ensures the TUI displays correct progress immediately on resume
            // rather than showing 0 until the first file completes.
            if already_completed > 0 {
                let order_ngrams = self.checkpoint.stats.ngrams_by_order[(order - 1) as usize];
                let _ = event_tx.send(ImportEvent::OrderProgress {
                    order,
                    files_completed: already_completed,
                    total_files,
                    ngrams_processed: order_ngrams,
                    is_complete: false, // We wouldn't be here if complete (pending_count > 0)
                    files_succeeded: already_completed,
                    files_skipped: 0, // On resume we don't know which were skipped vs succeeded
                });
            }

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
                // Most updates map to a single event, but some need multiple events
                match update {
                    WorkerUpdate::Started { worker_id, order, prefix, attempt } => {
                        // Always emit WorkerStarted
                        let _ = event_tx_worker.send(ImportEvent::WorkerStarted {
                            worker_id,
                            order,
                            prefix: prefix.to_string(),
                        });
                        // If this is a retry attempt, also emit DeferredRetryStarted
                        // to decrement the backoff queue counter
                        if attempt > 0 {
                            let _ = event_tx_worker.send(ImportEvent::DeferredRetryStarted {
                                prefix: prefix.to_string(),
                                order,
                            });
                        }
                    }
                    WorkerUpdate::Finished { worker_id, order, prefix, ngram_count, duration } => {
                        let _ = event_tx_worker.send(ImportEvent::WorkerFinished {
                            worker_id,
                            order,
                            prefix: prefix.to_string(),
                            ngram_count,
                            duration,
                        });
                    }
                    WorkerUpdate::NgramProgress { worker_id, ngram_count } => {
                        let _ = event_tx_worker.send(ImportEvent::WorkerNgramProgress {
                            worker_id,
                            ngram_count,
                        });
                    }
                    WorkerUpdate::Retrying { worker_id, order, prefix, attempt, error } => {
                        // Emit WorkerRetrying for TUI worker status display
                        let _ = event_tx_worker.send(ImportEvent::WorkerRetrying {
                            worker_id,
                            prefix: prefix.to_string(),
                            attempt,
                            max_attempts: MAX_RETRIES as u32,
                            error: error.to_string(),
                        });
                        // Also emit DeferredRetry to track backoff queue count
                        let _ = event_tx_worker.send(ImportEvent::DeferredRetry {
                            prefix: prefix.to_string(),
                            attempt,
                            order,
                        });
                    }
                    WorkerUpdate::Deferred { worker_id, order, prefix, attempt, delay_seconds: _, error } => {
                        // Emit WorkerRetrying for TUI worker status display
                        let _ = event_tx_worker.send(ImportEvent::WorkerRetrying {
                            worker_id,
                            prefix: prefix.to_string(),
                            attempt,
                            max_attempts: MAX_RETRIES as u32,
                            error: error.to_string(),
                        });
                        // Also emit DeferredRetry to track backoff queue count
                        let _ = event_tx_worker.send(ImportEvent::DeferredRetry {
                            prefix: prefix.to_string(),
                            attempt,
                            order,
                        });
                    }
                    WorkerUpdate::Exited { worker_id } => {
                        let _ = event_tx_worker.send(ImportEvent::WorkerExited { worker_id });
                    }
                }
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
        // Use async_channel for lock-free MPMC queue - each worker gets a clone of the receiver
        // This eliminates the Tokio Mutex bottleneck that caused all workers to synchronize
        let (job_tx, job_rx) = async_channel::bounded::<Job>(
            total_pending as usize + failed_retry_count + requeue_capacity + 1
        );
        // Note: job_rx is Clone - no Arc<Mutex<...>> wrapper needed

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

            let prefixes = self.get_filtered_prefixes(order);
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

        // Pre-allocate per-worker stats atomics for race-free sampling.
        // Use 2x parallel_downloads to handle dynamic spawning without reallocation.
        let max_workers = parallel_downloads * 2;
        let worker_stats: Vec<AtomicU64> = (0..max_workers)
            .map(|_| AtomicU64::new(0))
            .collect();

        // Create shared HTTP client with connection pooling for all workers.
        // This prevents the concurrency amplification bug where each worker creating
        // independent clients causes Google to see a spike in connections.
        // - pool_max_idle_per_host: Allow connection reuse with reasonable pool size
        // - HTTP/2 multiplexing will automatically combine requests on shared connections
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))        // 5 minute total timeout
            .connect_timeout(Duration::from_secs(30)) // 30 second connection timeout
            .read_timeout(Duration::from_secs(60))    // 60 second read timeout
            .pool_max_idle_per_host(4)                // Allow connection reuse
            .user_agent("Mozilla/5.0 (compatible; libgrammstein/0.1; +https://github.com/vinary-tree/libgrammstein)")
            .build()
            .expect("Failed to build shared HTTP client");

        // Create shared state for workers
        let shared_state = Arc::new(WorkerSharedState {
            config: config.clone(),
            storage: Arc::clone(&storage),
            total_ngrams: Arc::clone(&total_ngrams),
            unique_ngrams: Arc::clone(&unique_ngrams),
            progress_tx: worker_tx.clone(),
            paused: Arc::clone(&paused),
            queue_size: Arc::clone(&queue_size),
            worker_stats,
            http_client,
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
        // Each worker gets a clone of the async_channel receiver (no mutex needed)
        for worker_id in 0..parallel_downloads {
            let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
            let handle = tokio::spawn(worker_task(
                worker_id,
                job_rx.clone(),
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

        // Spawn periodic stats emitter task (3 second interval)
        // Samples per-worker packed atomics for race-free, synchronized statistics.
        // This ensures the TUI receives real-time updates even when no files are completing.
        let stats_event_tx = event_tx.clone();
        let stats_shared_state = Arc::clone(&shared_state);
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

                // Sample all per-worker counters (non-blocking, race-free reads)
                // Each packed atomic: upper 32 bits = total, lower 32 bits = unique
                let mut live_total_ngrams = 0u64;
                let mut live_unique_ngrams = 0u64;

                for (worker_id, worker_stat) in stats_shared_state.worker_stats.iter().enumerate() {
                    let packed = worker_stat.load(Ordering::Relaxed);
                    let ngrams = packed >> 32;
                    let unique = packed & 0xFFFFFFFF;
                    live_total_ngrams += ngrams;
                    live_unique_ngrams += unique;

                    // Send per-worker progress event for TUI worker display
                    if ngrams > 0 {
                        let _ = stats_event_tx.send(ImportEvent::WorkerNgramProgress {
                            worker_id,
                            ngram_count: ngrams,
                        });
                    }
                }

                // Combine live in-progress counts with global completed counts
                let completed_total = stats_shared_state.total_ngrams.load(Ordering::Relaxed);
                let completed_unique = stats_shared_state.unique_ngrams.load(Ordering::Relaxed);
                let total = completed_total + live_total_ngrams;
                let unique = completed_unique + live_unique_ngrams;

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

        // Emit phase change: now importing n-grams
        let _ = event_tx.send(ImportEvent::PhaseChanged {
            phase: "Importing N-grams".to_string(),
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
                                         job_rx: &async_channel::Receiver<Job>,
                                         job_tx: &async_channel::Sender<Job>,
                                         shared_state: &Arc<WorkerSharedState>,
                                         result_tx: &tokio::sync::mpsc::Sender<JobResult>,
                                         worker_exit_tx: &tokio::sync::mpsc::Sender<usize>,
                                         event_tx: &tokio::sync::broadcast::Sender<ImportEvent>|
         -> usize {
            let current_count = worker_handles.len();
            let mut spawned = 0usize;

            if target > current_count {
                // Spawn additional workers immediately (each worker gets a clone of the receiver)
                for _ in 0..(target - current_count) {
                    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
                    let handle = tokio::spawn(worker_task(
                        *next_worker_id,
                        job_rx.clone(),
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

                    // Signal all workers to shutdown
                    signal_all_shutdown(&worker_shutdown_txs);

                    // Wait for ALL workers to fully exit before checkpointing.
                    // This ensures no vocabulary writes can occur after checkpoint.
                    //
                    // IMPORTANT: Draining results is NOT sufficient because a worker
                    // can send its result while still holding the vocabulary write lock.
                    // We must wait for worker_exit_rx notifications which are sent
                    // AFTER the worker has fully terminated.
                    log::info!(
                        "Cancellation: waiting for {} active workers to exit...",
                        active_workers
                    );

                    while active_workers > 0 {
                        tokio::select! {
                            biased;

                            // Track worker exits (highest priority)
                            Some(exited_worker_id) = worker_exit_rx.recv() => {
                                active_workers = active_workers.saturating_sub(1);
                                worker_handles.remove(&exited_worker_id);
                                worker_shutdown_txs.remove(&exited_worker_id);
                                log::debug!(
                                    "Cancellation: worker {} exited, {} remaining",
                                    exited_worker_id,
                                    active_workers
                                );
                            }

                            // Drain results concurrently to prevent channel backpressure
                            Some(_job_result) = result_rx.recv() => {
                                results_received += 1;
                            }

                            // Timeout safety net (shouldn't happen in normal operation)
                            _ = tokio::time::sleep(Duration::from_secs(30)) => {
                                log::error!(
                                    "Cancellation: timeout waiting for {} workers to exit, \
                                     proceeding with checkpoint anyway",
                                    active_workers
                                );
                                break;
                            }
                        }
                    }

                    log::info!("Cancellation: all workers exited, saving checkpoint");

                    // NOW safe to checkpoint - no more vocabulary writes can occur
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

                    // DISABLED: We intentionally do NOT save a checkpoint on each
                    // worker exit. When all 12 workers exit simultaneously (end of
                    // import), the original code below caused 12× redundant
                    // checkpoint saves (each taking ~30s for vocabulary merge +
                    // shard sync), adding ~6 minutes of blocking I/O. The periodic
                    // checkpoint (line ~5317) and the "Final checkpoint save"
                    // (line ~4382) already ensure durability.
                    //
                    // // Save checkpoint when workers exit to preserve progress
                    // if let Err(e) = self.save_checkpoint() {
                    //     log::error!("Checkpoint save failed on worker exit: {}", e);
                    // }
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
                            *order_files_completed.entry(result_order).or_insert(0) += 1;
                            *order_files_skipped.entry(result_order).or_insert(0) += 1;

                            // Emit per-order progress event (so TUI updates immediately)
                            let order_done = order_files_completed.get(&result_order).copied().unwrap_or(0);
                            let order_skipped = order_files_skipped.get(&result_order).copied().unwrap_or(0);
                            let order_total = order_total_files.get(&result_order).copied().unwrap_or(0);
                            let order_ngrams = self.checkpoint.stats.ngrams_by_order[(result_order - 1) as usize];
                            let order_pending = jobs_per_order.get(&result_order).copied().unwrap_or(0);
                            let order_already_complete = order_total - order_pending;

                            let _ = event_tx.send(ImportEvent::OrderProgress {
                                order: result_order,
                                files_completed: order_done,
                                total_files: order_total,
                                ngrams_processed: order_ngrams,
                                is_complete: order_done >= order_pending,
                                files_succeeded: order_done - order_skipped + order_already_complete,
                                files_skipped: order_skipped,
                            });

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
                            *order_files_completed.entry(result_order).or_insert(0) += 1;
                            *order_files_skipped.entry(result_order).or_insert(0) += 1;

                            // Emit per-order progress event (so TUI updates immediately)
                            let order_done = order_files_completed.get(&result_order).copied().unwrap_or(0);
                            let order_skipped = order_files_skipped.get(&result_order).copied().unwrap_or(0);
                            let order_total = order_total_files.get(&result_order).copied().unwrap_or(0);
                            let order_ngrams = self.checkpoint.stats.ngrams_by_order[(result_order - 1) as usize];
                            let order_pending = jobs_per_order.get(&result_order).copied().unwrap_or(0);
                            let order_already_complete = order_total - order_pending;

                            let _ = event_tx.send(ImportEvent::OrderProgress {
                                order: result_order,
                                files_completed: order_done,
                                total_files: order_total,
                                ngrams_processed: order_ngrams,
                                is_complete: order_done >= order_pending,
                                files_succeeded: order_done - order_skipped + order_already_complete,
                                files_skipped: order_skipped,
                            });

                            // Continue to next result
                            continue;
                        }
                    };

                    // Update per-order progress tracking (success case)
                    *order_files_completed.entry(result_order).or_insert(0) += 1;
                    self.checkpoint.complete_prefix(result_order, &prefix);

                    // Mark completion in storage layer (important for sharded storage)
                    if let Err(e) = self.storage.mark_prefix_completed(&prefix, result_order) {
                        log::warn!("Failed to mark prefix {} as completed in storage: {}", prefix, e);
                    }

                    self.checkpoint.add_ngrams(result_order, ngrams_in_file);
                    self.checkpoint.stats.ngrams_by_order[(result_order - 1) as usize] += ngrams_in_file;
                    files_completed.fetch_add(1, Ordering::Relaxed);

                    // Emit per-order progress event
                    let order_done = order_files_completed.get(&result_order).copied().unwrap_or(0);
                    let order_skipped = order_files_skipped.get(&result_order).copied().unwrap_or(0);
                    let order_total = order_total_files.get(&result_order).copied().unwrap_or(0);
                    let order_ngrams = self.checkpoint.stats.ngrams_by_order[(result_order - 1) as usize];
                    let order_pending = jobs_per_order.get(&result_order).copied().unwrap_or(0);
                    let order_already_complete = order_total - order_pending;

                    // Order is complete when all files have been processed (success + fail + skip)
                    // Note: order_done now includes all outcomes; failed prefixes will be retried next run
                    let is_order_complete = order_done >= order_pending;

                    let _ = event_tx.send(ImportEvent::OrderProgress {
                        order: result_order,
                        files_completed: order_done,
                        total_files: order_total,
                        ngrams_processed: order_ngrams,
                        is_complete: is_order_complete,
                        files_succeeded: order_done - order_skipped + order_already_complete,
                        files_skipped: order_skipped,
                    });

                    // Check if order is now complete
                    if is_order_complete && !self.checkpoint.is_order_complete(result_order) {
                        self.checkpoint.complete_order(result_order)?;
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
                        if order_skipped > 0 {
                            log::warn!(
                                "Order {} completed with {} failed prefixes (will be retried on next run): {} n-grams in {:?}",
                                result_order,
                                order_skipped,
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

                    // Flush lock-free overlays for shards exceeding threshold
                    if let Err(e) = self.storage.flush_lockfree_over_threshold(self.lockfree_flush_threshold) {
                        log::warn!("Lock-free flush failed: {}", e);
                    }

                    // Save checkpoint periodically (async for better throughput)
                    let checkpoint_interval: u64 = if self.config.parallel_downloads >= 8 { 5 } else { 10 };
                    if files_completed.load(Ordering::Relaxed) % checkpoint_interval == 0 {
                        if let Err(e) = self.save_checkpoint_async_with_events(Some(&event_tx)) {
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

        // ====================================================================
        // CLEANUP: Use CleanupGuard for deterministic LIFO cleanup order.
        // See state_machine.rs for detailed explanation of why order matters.
        // ====================================================================
        //
        // Workers hold Arc<WorkerSharedState> references. The shared_state
        // contains progress_tx, which keeps the worker_converter channel open.
        // CleanupGuard ensures proper cleanup order:
        //   1. Signal shutdown -> 2. Wait workers -> 3. Drop shared_state
        //   -> 4. Drop channels -> 5. Wait converter -> 6. Abort stats
        //   -> 7. Abort command handler

        // Emit phase change: entering cleanup phase
        let _ = event_tx.send(ImportEvent::PhaseChanged {
            phase: "Cleaning Up".to_string(),
        });

        // Build cleanup resources and execute cleanup guard (LIFO order guaranteed)
        let cleanup_resources = CleanupResources::new()
            .with_worker_handles(worker_handles)
            .with_worker_shutdown_txs(worker_shutdown_txs)
            .with_shared_state(shared_state)
            .with_result_tx(result_tx)
            .with_worker_exit_tx(worker_exit_tx)
            .with_worker_converter(worker_converter)
            .with_stats_task(stats_task)
            .with_command_handler(command_handler);

        let cleanup_guard = cleanup_resources.into_cleanup_guard();
        cleanup_guard.cleanup().await;

        // Allow TUI to catch up with cleanup events before sending post-cleanup phases
        // This prevents broadcast channel lagging from dropping PhaseChanged events
        tokio::task::yield_now().await;

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

        // Emit ImportCompleted event (n-gram collection done)
        let collection_duration = self.start_time.elapsed();
        let total = self.total_ngrams.load(Ordering::Relaxed);
        log::debug!("[IMPORTER] Cleanup complete, sending ImportCompleted");
        let _ = event_tx.send(ImportEvent::ImportCompleted {
            total_ngrams: total,
            duration: collection_duration,
        });

        // Yield to the event loop before starting finalization.
        // This allows pending signals (Ctrl+C) to be processed before we enter
        // the synchronous finalization phase (MKN computation + merge). Without
        // this yield, the tokio runtime may not process SIGINT handlers because
        // the synchronous work monopolizes the runtime thread.
        tokio::task::yield_now().await;

        // Emit phase change: now computing MKN statistics
        log::debug!("[IMPORTER] Sending PhaseChanged: 'Computing MKN Statistics'");
        let _ = event_tx.send(ImportEvent::PhaseChanged {
            phase: "Computing MKN Statistics".to_string(),
        });

        // Finalize: compute MKN stats, sync storage, and return final stats
        let import_stats = self.finalize_with_events(&event_tx)?;

        // Emit phase change: now merging shards
        log::debug!("[IMPORTER] MKN complete, sending PhaseChanged: 'Merging Shards'");
        let _ = event_tx.send(ImportEvent::PhaseChanged {
            phase: "Merging Shards".to_string(),
        });

        // Merge shards if using sharded storage
        let merge_performed = self.merge_shards(keep_shards, &event_tx).await?;

        // Emit AllWorkCompleted event (triggers completion dialog)
        let total_duration = self.start_time.elapsed();
        log::debug!("[IMPORTER] Merge complete, sending AllWorkCompleted");
        let _ = event_tx.send(ImportEvent::AllWorkCompleted {
            total_ngrams: import_stats.total_ngrams,
            total_duration,
            shards_kept: keep_shards || !merge_performed,
        });

        Ok(import_stats)
    }

    /// Process a single local file.
    ///
    /// For single-trie mode, uses file transactions with INCREMENT semantics
    /// to ensure atomic per-file processing and correct cross-file count
    /// accumulation. For sharded mode, uses direct storage calls (each prefix
    /// file is complete, so SET semantics are appropriate).
    fn process_file(&mut self, path: &Path) -> Result<u64, ImportError> {
        // Use transactions for single-trie mode, direct calls for sharded
        if !self.storage.is_sharded() {
            return self.process_file_with_transaction(path);
        }

        // Sharded mode: use direct storage calls (existing behavior)
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

    /// Process a single local file using file transactions (single-trie mode).
    ///
    /// Uses INCREMENT semantics for cross-file count accumulation with
    /// atomic per-file commit/rollback.
    fn process_file_with_transaction(&self, path: &Path) -> Result<u64, ImportError> {
        let file_id = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        // Begin file transaction
        let mut tx = self.storage.begin_file_tx(file_id)
            .map_err(|e| ImportError::Trie(format!("Failed to begin file tx: {}", e)))?;

        let result = self.process_file_inner(&mut tx, path);

        match result {
            Ok(ngrams_in_file) => {
                // Commit atomically
                self.storage.commit_file_tx(tx)
                    .map_err(|e| ImportError::Trie(format!("Failed to commit file tx: {}", e)))?;

                self.total_ngrams.fetch_add(ngrams_in_file, Ordering::Relaxed);
                Ok(ngrams_in_file)
            }
            Err(e) => {
                // Abort on error - discard partial work
                let _ = self.storage.abort_file_tx(tx);
                Err(e)
            }
        }
    }

    /// Inner file processing that operates on a transaction.
    fn process_file_inner(
        &self,
        tx: &mut super::storage::StorageFileTx,
        path: &Path,
    ) -> Result<u64, ImportError> {
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
                // Use tx_increment for INCREMENT semantics
                self.storage.tx_increment_ngram(tx, &aggregated.ngram, aggregated.total_count)
                    .map_err(|e| ImportError::Trie(format!("Failed to increment ngram: {}", e)))?;
                ngrams_in_file += 1;
            }
        }

        // Flush final n-gram
        if let Some(aggregated) = aggregator.flush() {
            self.storage.tx_increment_ngram(tx, &aggregated.ngram, aggregated.total_count)
                .map_err(|e| ImportError::Trie(format!("Failed to increment ngram: {}", e)))?;
            ngrams_in_file += 1;
        }

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
        self.finalize_with_events_inner(None)
    }

    /// Finalize import with event emission for TUI progress updates.
    fn finalize_with_events(
        &mut self,
        event_tx: &tokio::sync::broadcast::Sender<ImportEvent>,
    ) -> Result<ImportStats, ImportError> {
        self.finalize_with_events_inner(Some(event_tx))
    }

    /// Inner finalize implementation that optionally emits events.
    fn finalize_with_events_inner(
        &mut self,
        event_tx: Option<&tokio::sync::broadcast::Sender<ImportEvent>>,
    ) -> Result<ImportStats, ImportError> {
        log::info!("Finalizing import...");

        // DISABLED: We skip the redundant sync() + sync_vocabulary() + checkpoint()
        // calls that previously existed here. The "Final checkpoint save"
        // (save_checkpoint_with_parallelism) has already:
        // - Merged the vocabulary lock-free layer and rotated WAL
        // - Synced all n-gram shards in parallel
        // - Checkpointed all shards
        // No new data has been written between that checkpoint and this point.
        //
        // // IMPORTANT: Sync and checkpoint FIRST to ensure all data is persisted
        // // before computing MKN stats. MKN uses discover_shard_files() which reads
        // // from disk, so data must be flushed first.
        // log::info!("Syncing storage to disk...");
        // self.storage.sync().map_err(|e| {
        //     ImportError::Trie(format!("Failed to sync storage: {}", e))
        // })?;
        // self.storage.sync_vocabulary().map_err(|e| {
        //     ImportError::Trie(format!("Failed to sync vocabulary: {}", e))
        // })?;
        // log::info!("Creating storage checkpoint...");
        // self.storage.checkpoint().map_err(|e| {
        //     ImportError::Trie(format!("Failed to checkpoint storage: {}", e))
        // })?;
        //
        // We do perform a final vocabulary compaction (checkpoint_vocabulary)
        // which re-serializes the entire vocabulary trie to minimize recovery
        // time. This is only done once at finalize, not during periodic
        // checkpoints (which use WAL rotation for bloat-free durability).
        log::info!("Final vocabulary compaction...");
        self.storage.checkpoint_vocabulary().map_err(|e| {
            ImportError::Trie(format!("Failed to checkpoint vocabulary: {}", e))
        })?;

        // Now compute MKN stats (has access to all flushed shard data)
        self.compute_mkn_stats_with_events(event_tx)?;

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

    /// Merge shards into the final output file (sharded storage only).
    ///
    /// This method performs post-import merge of shards into a single trie file.
    /// It emits progress events for the TUI and optionally cleans up shard files.
    ///
    /// # Arguments
    ///
    /// * `keep_shards` - If true, preserve shard files after merge
    /// * `event_tx` - Broadcast sender for TUI progress events
    ///
    /// # Returns
    ///
    /// Returns `true` if merge was performed, `false` if not using sharded storage.
    async fn merge_shards(
        &self,
        keep_shards: bool,
        event_tx: &tokio::sync::broadcast::Sender<ImportEvent>,
    ) -> Result<bool, ImportError> {
        // Check if we're using sharded storage
        let coordinator = match self.storage.as_sharded() {
            Some(c) => c,
            None => {
                log::info!("Not using sharded storage, skipping merge phase");
                return Ok(false);
            }
        };

        let shard_count = coordinator.open_shard_keys().len();
        let estimated_ngrams = coordinator.total_entry_count();

        if shard_count == 0 {
            log::warn!("No shards to merge");
            return Ok(false);
        }

        log::info!("Starting merge of {} shards (~{} n-grams)", shard_count, estimated_ngrams);

        // Emit MergeStarted event
        log::debug!("[IMPORTER] Sending MergeStarted: shard_count={}, estimated_ngrams={}", shard_count, estimated_ngrams);
        let _ = event_tx.send(ImportEvent::MergeStarted {
            shard_count,
            estimated_ngrams,
        });

        // Create merge coordinator
        let merger = MergeCoordinator::new(coordinator);

        // Merge to the output trie
        let merge_start = Instant::now();
        let merge_result = merger.merge_to_trie(&self.config.output_path, |progress| {
            let _ = event_tx.send(ImportEvent::MergeProgress {
                shards_processed: progress.total_shards - progress.shards_remaining,
                total_shards: progress.total_shards,
                ngrams_merged: progress.ngrams_merged,
                percent_complete: progress.percent_complete,
            });
        });

        match merge_result {
            Ok(stats) => {
                let merge_duration = merge_start.elapsed();
                log::info!(
                    "Merge completed: {} n-grams, {} bytes in {:.1}s",
                    stats.total_ngrams,
                    stats.bytes_written,
                    merge_duration.as_secs_f64()
                );

                // Emit MergeCompleted event
                log::debug!("[IMPORTER] Sending MergeCompleted: total_ngrams={}, bytes_written={}", stats.total_ngrams, stats.bytes_written);
                let _ = event_tx.send(ImportEvent::MergeCompleted {
                    total_ngrams: stats.total_ngrams,
                    bytes_written: stats.bytes_written,
                    duration: merge_duration,
                });

                // Clean up shards if requested
                if !keep_shards {
                    self.cleanup_shards(shard_count, event_tx)?;
                }

                Ok(true)
            }
            Err(e) => {
                log::error!("Merge failed: {}", e);
                let _ = event_tx.send(ImportEvent::MergeFailed {
                    error: e.to_string(),
                });
                Err(ImportError::Trie(format!("Merge failed: {}", e)))
            }
        }
    }

    /// Clean up shard files after successful merge.
    fn cleanup_shards(
        &self,
        shard_count: usize,
        event_tx: &tokio::sync::broadcast::Sender<ImportEvent>,
    ) -> Result<(), ImportError> {
        log::info!("Cleaning up {} shard files...", shard_count);

        // Emit cleanup started event
        let _ = event_tx.send(ImportEvent::ShardCleanupStarted { shard_count });

        // Get the shard directory from coordinator
        let coordinator = self.storage.as_sharded().ok_or_else(|| {
            ImportError::Trie("Expected sharded storage for cleanup".to_string())
        })?;

        let shard_dir = coordinator.config().shard_dir.clone();

        // Count files and bytes before deletion
        let mut shards_deleted = 0usize;
        let mut bytes_freed = 0u64;

        // Read the shard directory and delete shard files
        if shard_dir.exists() {
            match std::fs::read_dir(&shard_dir) {
                Ok(entries) => {
                    for entry in entries.filter_map(|e| e.ok()) {
                        let path = entry.path();
                        // Delete shard trie files (*.artrie) and WAL files (*.wal)
                        if let Some(ext) = path.extension() {
                            if ext == "artrie" || ext == "wal" {
                                if let Ok(metadata) = std::fs::metadata(&path) {
                                    bytes_freed += metadata.len();
                                }
                                if std::fs::remove_file(&path).is_ok() {
                                    shards_deleted += 1;
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    log::warn!("Failed to read shard directory for cleanup: {}", e);
                }
            }

            // Delete wal_archive directory if it exists
            let wal_archive_dir = shard_dir.join("wal_archive");
            if wal_archive_dir.exists() && wal_archive_dir.is_dir() {
                // Calculate size of files in wal_archive
                if let Ok(entries) = std::fs::read_dir(&wal_archive_dir) {
                    for entry in entries.filter_map(|e| e.ok()) {
                        if let Ok(meta) = entry.metadata() {
                            bytes_freed += meta.len();
                        }
                    }
                }
                if std::fs::remove_dir_all(&wal_archive_dir).is_ok() {
                    log::info!("Deleted wal_archive directory");
                }
            }
        }

        log::info!(
            "Cleanup complete: deleted {} shard files, freed {} bytes",
            shards_deleted,
            bytes_freed
        );

        // Emit cleanup completed event
        let _ = event_tx.send(ImportEvent::ShardCleanupCompleted {
            shards_deleted,
            bytes_freed,
        });

        Ok(())
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
        self.compute_mkn_stats_with_events(None)
    }

    /// Compute MKN stats with optional event emission for TUI progress.
    fn compute_mkn_stats_with_events(
        &mut self,
        event_tx: Option<&tokio::sync::broadcast::Sender<ImportEvent>>,
    ) -> Result<(), ImportError> {
        if self.checkpoint.mkn_phase == MknPhase::Complete {
            log::info!("MKN statistics already computed");
            return Ok(());
        }

        log::info!("Computing MKN statistics (post-processing)...");
        let mkn_start = std::time::Instant::now();
        let estimated_ngrams = self.total_ngrams.load(Ordering::Relaxed);

        // Emit MknStarted event
        let source = if self.storage.is_sharded() { "shards" } else { "single_trie" };
        if let Some(tx) = event_tx {
            log::debug!("[IMPORTER] Sending MknStarted: source={}, estimated_ngrams={}", source, estimated_ngrams);
            let _ = tx.send(ImportEvent::MknStarted {
                source: source.to_string(),
                estimated_ngrams,
            });
        }

        let result = if self.storage.is_sharded() {
            // Sharded mode: use MknAggregator which iterates over all shards
            self.compute_mkn_stats_sharded_with_events(event_tx)
        } else {
            // Single-trie mode: iterate over trie and store stats inline
            self.compute_mkn_stats_single_trie_with_events(event_tx)
        };

        match result {
            Ok((continuation_entries, frequency_entries)) => {
                self.checkpoint.mkn_phase = MknPhase::Complete;
                self.save_checkpoint()?;

                let duration = mkn_start.elapsed();
                log::info!("MKN statistics computed successfully in {:.1}s", duration.as_secs_f64());

                // Emit MknCompleted event
                if let Some(tx) = event_tx {
                    log::debug!("[IMPORTER] Sending MknCompleted: continuation={}, frequency={}", continuation_entries, frequency_entries);
                    let _ = tx.send(ImportEvent::MknCompleted {
                        continuation_entries,
                        frequency_entries,
                        duration,
                    });
                }

                Ok(())
            }
            Err(e) => {
                // Emit MknFailed event
                if let Some(tx) = event_tx {
                    let _ = tx.send(ImportEvent::MknFailed {
                        error: e.to_string(),
                    });
                }
                Err(e)
            }
        }
    }

    /// Compute MKN stats for sharded storage using MknAggregator.
    fn compute_mkn_stats_sharded(&self) -> Result<(), ImportError> {
        self.compute_mkn_stats_sharded_with_events(None).map(|_| ())
    }

    /// Compute MKN stats for sharded storage with optional event emission.
    ///
    /// Returns (continuation_entries, frequency_entries) counts on success.
    fn compute_mkn_stats_sharded_with_events(
        &self,
        event_tx: Option<&tokio::sync::broadcast::Sender<ImportEvent>>,
    ) -> Result<(u64, u64), ImportError> {
        let coordinator = self.storage.as_sharded().ok_or_else(|| {
            ImportError::Trie("Expected sharded storage".to_string())
        })?;

        // Phase 1: Compute MKN statistics (parallel over shards via rayon)
        log::info!("MKN Phase 1: Computing statistics across shards...");
        if let Some(tx) = event_tx {
            let _ = tx.send(ImportEvent::MknProgress {
                phase: 1,
                total_phases: 2,
                items_processed: 0,
                total_items: self.total_ngrams.load(Ordering::Relaxed),
                percent_complete: 0.0,
            });
        }

        let aggregator = MknAggregator::new(coordinator)
            .with_cancellation_flag(&self.interrupted);
        let mkn_stats = aggregator.compute_all().map_err(|e| {
            ImportError::Trie(format!("Failed to compute MKN statistics: {}", e))
        })?;

        if let Some(tx) = event_tx {
            let _ = tx.send(ImportEvent::MknProgress {
                phase: 1,
                total_phases: 2,
                items_processed: self.total_ngrams.load(Ordering::Relaxed),
                total_items: self.total_ngrams.load(Ordering::Relaxed),
                percent_complete: 50.0,
            });
        }

        // Phase 2: Write MKN statistics to trie
        log::info!("MKN Phase 2: Writing statistics to MKN trie...");
        let mkn_path = self.config.output_path.with_extension("mkn.artrie");
        log::info!("Saving MKN statistics to {:?}...", mkn_path);

        let mkn_trie = PersistentARTrie::create(&mkn_path).map_err(|e| {
            ImportError::Trie(format!("Failed to create MKN trie: {}", e))
        })?;
        let mkn_trie = Arc::new(RwLock::new(mkn_trie));

        let mut continuation_entries = 0u64;
        let mut frequency_entries = 0u64;

        {
            let mut trie = mkn_trie.write();

            // Store frequency counts for each order
            for (order, counts) in mkn_stats.frequency_counts.iter().enumerate() {
                let prefix = format!("\x00order{}\x00", order);
                trie.upsert_bytes(format!("{}n1", prefix).as_bytes(), counts.n1).map_err(|e| {
                    ImportError::Trie(format!("Failed to write n1: {}", e))
                })?;
                trie.upsert_bytes(format!("{}n2", prefix).as_bytes(), counts.n2).map_err(|e| {
                    ImportError::Trie(format!("Failed to write n2: {}", e))
                })?;
                trie.upsert_bytes(format!("{}n3", prefix).as_bytes(), counts.n3).map_err(|e| {
                    ImportError::Trie(format!("Failed to write n3: {}", e))
                })?;
                trie.upsert_bytes(format!("{}n4", prefix).as_bytes(), counts.n4).map_err(|e| {
                    ImportError::Trie(format!("Failed to write n4: {}", e))
                })?;
                trie.upsert_bytes(format!("{}total_unique", prefix).as_bytes(), counts.total_unique).map_err(|e| {
                    ImportError::Trie(format!("Failed to write total_unique: {}", e))
                })?;
                trie.upsert_bytes(format!("{}total_count", prefix).as_bytes(), counts.total_count).map_err(|e| {
                    ImportError::Trie(format!("Failed to write total_count: {}", e))
                })?;
                frequency_entries += 6;
            }

            // Store continuation counts for each order
            for (order, conts) in mkn_stats.continuation_counts.iter().enumerate() {
                // Store predecessor counts (N1+(•w) - unique predecessors for each context)
                for (context, count) in &conts.predecessor_counts {
                    let mut key = format!("\x00N1+predecessor\x00{}\x00", order).into_bytes();
                    key.extend_from_slice(context);
                    trie.upsert_bytes(&key, *count).map_err(|e| {
                        ImportError::Trie(format!("Failed to write predecessor count: {}", e))
                    })?;
                    continuation_entries += 1;
                }

                // Store successor counts (N1+(w•) - unique successors for each context)
                for (context, count) in &conts.successor_counts {
                    let mut key = format!("\x00N1+successor\x00{}\x00", order).into_bytes();
                    key.extend_from_slice(context);
                    trie.upsert_bytes(&key, *count).map_err(|e| {
                        ImportError::Trie(format!("Failed to write successor count: {}", e))
                    })?;
                    continuation_entries += 1;
                }
            }

            // Checkpoint to persist
            trie.checkpoint().map_err(|e| {
                ImportError::Trie(format!("Failed to checkpoint MKN trie: {}", e))
            })?;
        }

        if let Some(tx) = event_tx {
            let _ = tx.send(ImportEvent::MknProgress {
                phase: 2,
                total_phases: 2,
                items_processed: continuation_entries + frequency_entries,
                total_items: continuation_entries + frequency_entries,
                percent_complete: 100.0,
            });
        }

        log::info!(
            "MKN statistics saved: {} orders with frequency and continuation counts",
            mkn_stats.max_order
        );

        Ok((continuation_entries, frequency_entries))
    }

    /// Compute MKN stats for single-trie storage (original behavior).
    fn compute_mkn_stats_single_trie(&self) -> Result<(), ImportError> {
        self.compute_mkn_stats_single_trie_with_events(None).map(|_| ())
    }

    /// Compute MKN stats for single-trie storage with optional event emission.
    ///
    /// Returns (continuation_entries, frequency_entries) counts on success.
    ///
    /// N-grams are stored as varint-encoded byte keys (LEB128 encoding).
    /// This function decodes them to extract word indices for
    /// computing predecessor and successor contexts.
    fn compute_mkn_stats_single_trie_with_events(
        &self,
        event_tx: Option<&tokio::sync::broadcast::Sender<ImportEvent>>,
    ) -> Result<(u64, u64), ImportError> {
        // Collect unique (suffix, prefix) and (context, following) pairs
        // using HashSets for deduplication.
        //
        // We store contexts as varint-encoded keys (same format as n-gram keys)
        // and use u64 indices for efficient comparison.
        use std::collections::HashSet;
        let mut continuation_pairs: HashSet<(Vec<u8>, u64)> = HashSet::new();
        let mut unique_cont_pairs: HashSet<(Vec<u8>, u64)> = HashSet::new();

        // Frequency count accumulators
        let mut n1 = 0u64;
        let mut n2 = 0u64;
        let mut n3 = 0u64;
        let mut n4 = 0u64;
        let mut total_unique = 0u64;
        let mut total_count = 0u64;

        let estimated_ngrams = self.total_ngrams.load(Ordering::Relaxed);

        // Phase 1: Iterate all n-grams, collect pairs and compute frequency counts
        log::info!("Phase 1: Collecting continuation pairs and frequency counts from n-grams...");
        if let Some(tx) = event_tx {
            let _ = tx.send(ImportEvent::MknProgress {
                phase: 1,
                total_phases: 3,
                items_processed: 0,
                total_items: estimated_ngrams,
                percent_complete: 0.0,
            });
        }

        {
            // Single-trie MKN: iterate the n-gram data living in the
            // checkpoint trie (which IS the data trie in single-trie mode).
            let trie_arc = self.storage.checkpoint_trie();
            let trie = trie_arc.read();
            // Collect all entries first to avoid lifetime issues with borrowed iterator
            let entries: Vec<(Vec<u8>, u64)> = trie
                .iter_prefix_with_values(b"")
                .map(|iter| iter.collect())
                .unwrap_or_default();
            drop(trie);
            for (ngram, count) in entries {
                    // Skip metadata keys (they start with \x00)
                    if ngram.starts_with(&[0x00]) {
                        continue;
                    }

                    // Accumulate frequency counts
                    total_unique += 1;
                    total_count += count;
                    match count {
                        1 => n1 += 1,
                        2 => n2 += 1,
                        3 => n3 += 1,
                        4 => n4 += 1,
                        _ => {}
                    }

                    // Decode varint-encoded key to word indices
                    let indices = decode_ngram_key_bytes(&ngram);
                    if indices.len() >= 2 {
                        // MKN Pass 1: continuation counts (suffix → unique prefixes)
                        // e.g., indices [0, 1, 2] → prefix=0, suffix=encode([1, 2])
                        let prefix = indices[0];
                        let suffix = encode_indices_to_key_bytes(&indices[1..]);
                        continuation_pairs.insert((suffix, prefix));

                        // MKN Pass 2: unique continuations (context → unique following)
                        // e.g., indices [0, 1, 2] → context=encode([0, 1]), following=2
                        let context = encode_indices_to_key_bytes(&indices[..indices.len() - 1]);
                        let following = indices[indices.len() - 1];
                        unique_cont_pairs.insert((context, following));
                    }
                }
        }

        if let Some(tx) = event_tx {
            let _ = tx.send(ImportEvent::MknProgress {
                phase: 1,
                total_phases: 3,
                items_processed: estimated_ngrams,
                total_items: estimated_ngrams,
                percent_complete: 33.0,
            });
        }

        log::info!(
            "Collected {} continuation pairs and {} unique continuation pairs",
            continuation_pairs.len(),
            unique_cont_pairs.len()
        );
        log::info!(
            "Frequency counts: n1={}, n2={}, n3={}, n4={}, total_unique={}, total_count={}",
            n1, n2, n3, n4, total_unique, total_count
        );

        // Phase 2: Compute and write continuation counts
        log::info!("Phase 2: Writing continuation statistics to trie...");
        if let Some(tx) = event_tx {
            let _ = tx.send(ImportEvent::MknProgress {
                phase: 2,
                total_phases: 3,
                items_processed: 0,
                total_items: continuation_pairs.len() as u64 + unique_cont_pairs.len() as u64,
                percent_complete: 33.0,
            });
        }

        let mut continuation_entries = 0u64;
        {
            // Single-trie MKN writes to the data trie (same as checkpoint
            // trie in single-trie mode).
            let trie_arc = self.storage.checkpoint_trie();
            let mut trie = trie_arc.write();

            // Count unique prefixes per suffix (N1+(suffix))
            let mut suffix_counts: std::collections::HashMap<Vec<u8>, u64> =
                std::collections::HashMap::new();
            for (suffix, _prefix) in &continuation_pairs {
                *suffix_counts.entry(suffix.clone()).or_insert(0) += 1;
            }

            // Write continuation counts
            for (suffix, count) in &suffix_counts {
                let mut count_key = b"\x00N1+\x00".to_vec();
                count_key.extend_from_slice(suffix);
                trie.upsert_bytes(&count_key, *count).map_err(|e| {
                    ImportError::Trie(format!(
                        "Failed to write MKN continuation count: {}",
                        e
                    ))
                })?;
                continuation_entries += 1;
            }

            // Count unique following words per context (N1+prefix(context))
            let mut context_counts: std::collections::HashMap<Vec<u8>, u64> =
                std::collections::HashMap::new();
            for (context, _following) in &unique_cont_pairs {
                *context_counts.entry(context.clone()).or_insert(0) += 1;
            }

            // Write unique continuation counts
            for (context, count) in &context_counts {
                let mut count_key = b"\x00N1+prefix\x00".to_vec();
                count_key.extend_from_slice(context);
                trie.upsert_bytes(&count_key, *count).map_err(|e| {
                    ImportError::Trie(format!(
                        "Failed to write MKN unique continuation count: {}",
                        e
                    ))
                })?;
                continuation_entries += 1;
            }
        }

        if let Some(tx) = event_tx {
            let _ = tx.send(ImportEvent::MknProgress {
                phase: 2,
                total_phases: 3,
                items_processed: continuation_entries,
                total_items: continuation_entries,
                percent_complete: 66.0,
            });
        }

        // Phase 3: Write frequency counts to trie
        log::info!("Phase 3: Writing frequency statistics to trie...");
        if let Some(tx) = event_tx {
            let _ = tx.send(ImportEvent::MknProgress {
                phase: 3,
                total_phases: 3,
                items_processed: 0,
                total_items: 6,
                percent_complete: 66.0,
            });
        }

        let mut frequency_entries = 0u64;
        {
            let trie_arc = self.storage.checkpoint_trie();
            let mut trie = trie_arc.write();

            trie.upsert_bytes(b"\x00mkn\x00n1", n1).map_err(|e| {
                ImportError::Trie(format!("Failed to write MKN n1: {}", e))
            })?;
            frequency_entries += 1;

            trie.upsert_bytes(b"\x00mkn\x00n2", n2).map_err(|e| {
                ImportError::Trie(format!("Failed to write MKN n2: {}", e))
            })?;
            frequency_entries += 1;

            trie.upsert_bytes(b"\x00mkn\x00n3", n3).map_err(|e| {
                ImportError::Trie(format!("Failed to write MKN n3: {}", e))
            })?;
            frequency_entries += 1;

            trie.upsert_bytes(b"\x00mkn\x00n4", n4).map_err(|e| {
                ImportError::Trie(format!("Failed to write MKN n4: {}", e))
            })?;
            frequency_entries += 1;

            trie.upsert_bytes(b"\x00mkn\x00total_unique", total_unique).map_err(|e| {
                ImportError::Trie(format!("Failed to write MKN total_unique: {}", e))
            })?;
            frequency_entries += 1;

            trie.upsert_bytes(b"\x00mkn\x00total_count", total_count).map_err(|e| {
                ImportError::Trie(format!("Failed to write MKN total_count: {}", e))
            })?;
            frequency_entries += 1;
        }

        if let Some(tx) = event_tx {
            let _ = tx.send(ImportEvent::MknProgress {
                phase: 3,
                total_phases: 3,
                items_processed: frequency_entries,
                total_items: frequency_entries,
                percent_complete: 100.0,
            });
        }

        Ok((continuation_entries, frequency_entries))
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

impl Drop for GoogleBooksImporter {
    /// Best-effort WAL rotation on drop.
    ///
    /// This is a safety net to ensure vocabulary data is durable even if the
    /// normal checkpoint path is bypassed (e.g., panic, unexpected exit).
    /// Uses rotate_vocabulary_wal() to avoid file bloat; WAL replay provides
    /// crash recovery on restart.
    fn drop(&mut self) {
        if let Err(e) = self.storage.rotate_vocabulary_wal() {
            log::error!("Failed to rotate vocabulary WAL on drop: {}", e);
        }
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
    importer: GoogleBooksImporter,
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
        if let Some(importer) = importer_clone.try_lock() {
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

// ============================================================================
// Periodic Checkpoint Support with Lock-Free Cron Scheduler
// ============================================================================

/// Shared state for periodic checkpoint tasks (lock-free reads).
///
/// This struct enables the cron scheduler to perform checkpoints without
/// holding locks on the importer. It uses atomic types and ArcSwap for
/// lock-free reads, with locking only during actual I/O operations.
#[cfg(feature = "google-books")]
pub struct CheckpointState {
    /// Current n-gram count (atomic).
    pub ngrams_processed: AtomicU64,
    /// Current unique n-gram count (atomic).
    pub unique_ngrams: AtomicU64,
    /// Storage handle (Arc - read-only from cron thread). The checkpoint
    /// trie is owned by the storage; access it via
    /// `storage.checkpoint_trie()` or the high-level
    /// `save_import_checkpoint` methods.
    pub storage: Arc<NgramStorage>,
    /// Checkpoint data (swapped atomically via ArcSwap).
    pub checkpoint: arc_swap::ArcSwap<ImportCheckpoint>,
    /// Flag indicating checkpoint in progress (atomic).
    pub checkpoint_in_progress: AtomicBool,
    /// Start time for elapsed time calculation.
    pub start_time: Instant,
}

#[cfg(feature = "google-books")]
impl CheckpointState {
    /// Perform a checkpoint (called from cron thread).
    ///
    /// Uses RwLock only for actual I/O - all state reads are lock-free.
    pub fn perform_checkpoint(&self) -> Result<(), ImportError> {
        // Set in-progress flag (atomic)
        if self.checkpoint_in_progress.swap(true, Ordering::AcqRel) {
            // Already in progress - skip
            log::debug!("Checkpoint already in progress, skipping");
            return Ok(());
        }

        // Read current state (atomic loads - no locks)
        let ngrams = self.ngrams_processed.load(Ordering::Acquire);
        let unique = self.unique_ngrams.load(Ordering::Acquire);

        // Load checkpoint atomically
        let checkpoint_guard = self.checkpoint.load();
        let mut checkpoint = (**checkpoint_guard).clone();
        checkpoint.stats.ngrams_processed = ngrams;
        checkpoint.stats.unique_ngrams = unique;
        checkpoint.stats.elapsed_seconds = self.start_time.elapsed().as_secs();

        // Perform I/O (this is where we need locks)
        // Uses merge_and_rotate_vocabulary_wal() for single merge + WAL rotation
        // instead of the previous sync_vocabulary() + rotate_vocabulary_wal() pair
        // which caused two back-to-back HashMap rebuilds (~3.42 GB transient spike).
        log::debug!("Periodic checkpoint: merging vocabulary and rotating WAL...");
        self.storage.merge_and_rotate_vocabulary_wal().map_err(|e| {
            self.checkpoint_in_progress.store(false, Ordering::Release);
            ImportError::Trie(format!("Failed to merge and rotate vocabulary WAL: {}", e))
        })?;

        log::debug!("Periodic checkpoint: syncing shards...");
        self.storage.sync_parallel(8).map_err(|e| {
            self.checkpoint_in_progress.store(false, Ordering::Release);
            ImportError::Trie(format!("Failed to sync storage: {}", e))
        })?;
        self.storage.checkpoint_parallel(8).map_err(|e| {
            self.checkpoint_in_progress.store(false, Ordering::Release);
            ImportError::Trie(format!("Failed to checkpoint storage: {}", e))
        })?;

        log::debug!("Periodic checkpoint: saving metadata...");
        if let Err(e) = self.storage.save_import_checkpoint(&checkpoint) {
            self.checkpoint_in_progress.store(false, Ordering::Release);
            return Err(ImportError::Trie(format!(
                "Failed to save checkpoint to trie: {}",
                e
            )));
        }

        // Store updated checkpoint (atomic swap)
        self.checkpoint.store(Arc::new(checkpoint));

        // Clear in-progress flag (atomic)
        self.checkpoint_in_progress.store(false, Ordering::Release);

        log::info!("Periodic checkpoint completed: {} n-grams", ngrams);
        Ok(())
    }

    /// Check if a checkpoint is currently in progress.
    pub fn is_checkpoint_in_progress(&self) -> bool {
        self.checkpoint_in_progress.load(Ordering::Acquire)
    }
}

/// Run import with graceful shutdown handling and periodic checkpointing.
///
/// This version uses a lock-free cron scheduler to perform periodic checkpoints
/// every 5 seconds (configurable), ensuring that progress is not lost when the
/// import is interrupted between file completions.
///
/// # Lock-Free Design
///
/// - **Task submission**: Lock-free MPSC channel (crossbeam-channel)
/// - **Termination signal**: AtomicBool
/// - **Statistics**: AtomicU64 counters
/// - **Checkpoint state reads**: ArcSwap + AtomicU64
/// - **Only blocking during I/O**: RwLock only used during actual file writes
///
/// # Arguments
///
/// * `importer` - The Google Books importer instance
/// * `progress` - Progress callback for status updates
/// * `checkpoint_interval_ms` - Interval between periodic checkpoints (default: 5000ms)
///
/// # Example
///
/// ```ignore
/// let importer = GoogleBooksImporter::resume_or_start(config).await?;
/// let stats = run_import_with_periodic_checkpoints(
///     importer,
///     |progress| println!("{:?}", progress),
///     5000, // Checkpoint every 5 seconds
/// ).await?;
/// ```
#[cfg(feature = "google-books")]
pub async fn run_import_with_periodic_checkpoints<F>(
    mut importer: GoogleBooksImporter,
    progress: F,
    checkpoint_interval_ms: u64,
) -> Result<ImportStats, ImportError>
where
    F: FnMut(ImportProgress) + Send + 'static,
{
    use crate::util::cron::{spawn_cron_with_interval, TaskMetadata};
    use std::sync::atomic::Ordering as AtomicOrdering;

    let terminating = Arc::new(AtomicBool::new(false));

    // Create shared checkpoint state (lock-free reads). The checkpoint trie
    // is owned by the storage now — no separate `trie` field to clone here.
    let checkpoint_state = Arc::new(CheckpointState {
        ngrams_processed: AtomicU64::new(importer.total_ngrams.load(Ordering::Relaxed)),
        unique_ngrams: AtomicU64::new(importer.unique_ngrams.load(Ordering::Relaxed)),
        storage: Arc::clone(&importer.storage),
        checkpoint: arc_swap::ArcSwap::from_pointee(importer.checkpoint.clone()),
        checkpoint_in_progress: AtomicBool::new(false),
        start_time: importer.start_time,
    });

    // Start cron state machine with 50ms poll interval for responsive shutdown
    let (cron_handle, cron_thread, cron_stats, _cron_ready) =
        spawn_cron_with_interval(Arc::clone(&terminating), 50);

    // Schedule periodic checkpoints
    let checkpoint_state_for_cron = Arc::clone(&checkpoint_state);
    let checkpoint_interval = checkpoint_interval_ms;
    cron_handle.schedule_recurring(
        checkpoint_interval,
        checkpoint_interval,
        "periodic-checkpoint",
        move || {
            match checkpoint_state_for_cron.perform_checkpoint() {
                Ok(()) => true,
                Err(e) => {
                    log::error!("Periodic checkpoint failed: {}", e);
                    // Return true to keep rescheduling - transient errors should not stop checkpoints
                    true
                }
            }
        },
    );

    // Wrap importer in Arc<Mutex> for sharing with shutdown handler
    let importer_ref = Arc::new(parking_lot::Mutex::new(importer));
    let importer_for_shutdown = Arc::clone(&importer_ref);
    let checkpoint_state_for_shutdown = Arc::clone(&checkpoint_state);
    let terminating_for_shutdown = Arc::clone(&terminating);

    // Spawn shutdown handler with user-visible status messages
    let shutdown_handle = tokio::spawn(async move {
        shutdown_signal().await;

        // Display prominent shutdown message
        eprintln!();
        log::warn!("╔══════════════════════════════════════════════════════════╗");
        log::warn!("║  Shutdown signal received - saving progress...           ║");
        log::warn!("║  Please wait for checkpoint to complete.                 ║");
        log::warn!("║  Press Ctrl+C again to force quit (may lose progress).   ║");
        log::warn!("╚══════════════════════════════════════════════════════════╝");

        // Check if checkpoint is in progress
        if checkpoint_state_for_shutdown.is_checkpoint_in_progress() {
            log::info!("Waiting for in-progress checkpoint to complete...");
        }

        // Signal termination and interrupt importer
        terminating_for_shutdown.store(true, AtomicOrdering::Release);
        if let Some(importer) = importer_for_shutdown.try_lock() {
            importer.interrupt();
        }
    });

    // Run import
    let result = {
        let mut importer = importer_ref.lock();

        // Wrap progress callback to update checkpoint state atomics
        let checkpoint_state_for_progress = Arc::clone(&checkpoint_state);
        let mut user_progress = progress;
        let progress_wrapper = move |p: ImportProgress| {
            // Update atomics for cron thread
            checkpoint_state_for_progress
                .ngrams_processed
                .store(p.total_ngrams, AtomicOrdering::Release);
            // Call user's progress callback
            user_progress(p);
        };

        importer.import_http(progress_wrapper).await
    };

    // Signal termination to stop cron scheduler
    terminating.store(true, AtomicOrdering::Release);

    // Wait for cron manager to stop
    log::info!("Stopping periodic checkpoint scheduler...");
    if let Err(e) = cron_thread.join() {
        log::warn!("Cron thread panicked: {:?}", e);
    }

    let stats = cron_stats;
    log::info!(
        "Cron manager stopped. Tasks executed: {}, failed: {}, panicked: {}",
        stats.tasks_executed.load(AtomicOrdering::Relaxed),
        stats.tasks_failed.load(AtomicOrdering::Relaxed),
        stats.tasks_panicked.load(AtomicOrdering::Relaxed)
    );

    // Final checkpoint with detailed status
    log::info!("╔══════════════════════════════════════════════════════════╗");
    log::info!("║  Saving final checkpoint and flushing data to disk...    ║");
    log::info!("╚══════════════════════════════════════════════════════════╝");

    log::info!("  → Syncing vocabulary WAL...");
    log::info!("  → Syncing n-gram shards...");
    log::info!("  → Writing checkpoint metadata...");

    let checkpoint_start = Instant::now();
    {
        let mut importer = importer_ref.lock();
        if let Err(e) = importer.save_checkpoint() {
            log::error!("Final checkpoint failed: {}", e);
        } else {
            let elapsed = checkpoint_start.elapsed();
            log::info!(
                "  ✓ Checkpoint saved successfully in {:.2}s",
                elapsed.as_secs_f64()
            );
        }
    }

    log::info!("╔══════════════════════════════════════════════════════════╗");
    log::info!("║  Shutdown complete. Safe to exit.                        ║");
    log::info!("╚══════════════════════════════════════════════════════════╝");

    // Cancel shutdown handler if import completed normally
    shutdown_handle.abort();

    result
}

/// Default checkpoint interval for periodic checkpoints (5 seconds).
pub const DEFAULT_CHECKPOINT_INTERVAL_MS: u64 = 5000;


// Cache-file download/cleanup helpers used by `--cache-files` mode.
#[cfg(feature = "google-books")]
mod cache;

#[cfg(feature = "google-books")]
use cache::{cleanup_cache_file, download_to_cache};

#[cfg(test)]
mod tests;
