//! Google Books N-gram importer.
//!
//! Orchestrates the import process from Google Books N-grams into a PersistentARTrie,
//! with checkpoint/resume support for long-running imports.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use libdictenstein::persistent_artrie::PersistentARTrie;
use parking_lot::RwLock;

use super::storage::NgramStorage;
use crate::ngram::vocabulary::open_or_create_concurrent_vocabulary_lockfree_with_capacity;

use super::aggregator::YearAggregator;
use super::checkpoint::{CheckpointError, ImportCheckpoint, TrieCheckpointStorage};
use super::config::GoogleBooksConfig;
use super::languages::{get_prefixes, is_supported};
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
            0,             // Order 0 (unused)
            13_000_000,    // 1-grams: ~13M
            314_000_000,   // 2-grams: ~314M
            977_000_000,   // 3-grams: ~977M
            1_313_000_000, // 4-grams: ~1.3B
            1_176_000_000, // 5-grams: ~1.2B
        ],
        _ => &[
            0,           // Order 0 (unused)
            5_000_000,   // 1-grams (estimate for non-English)
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
    trie_guard
        .increment_bytes(ngram.as_bytes(), count as i64)
        .map_err(|e| ImportError::Trie(format!("Failed to store ngram '{}': {}", ngram, e)))?;
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

    fn store_checkpoint_u64(&self, key: &str, value: u64) -> Result<(), Self::Error> {
        self.upsert_bytes(key.as_bytes(), value)
            .map_err(|e| TrieCheckpointError::TrieError(e.to_string()))?;
        Ok(())
    }

    fn load_checkpoint_u64(&self, key: &str) -> Result<Option<u64>, Self::Error> {
        Ok(self.get_value_bytes(key.as_bytes()))
    }

    fn delete_checkpoint_key(&self, key: &str) -> Result<bool, Self::Error> {
        Ok(self.remove(key))
    }

    fn delete_checkpoint_prefix(&self, prefix: &str) -> Result<usize, Self::Error> {
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
        )
        .map_err(|e| ImportError::Trie(format!("Failed to create/open vocabulary: {}", e)))?;

        // Create storage backend with vocabulary for compact encoding
        let storage = NgramStorage::resume_or_start_with_vocabulary(
            &config,
            estimated_ngrams,
            Some(vocabulary),
        )
        .map_err(|e| ImportError::Trie(format!("Failed to create storage: {}", e)))?;

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
        log::info!(
            "Lock-free flush threshold set to {} entries per shard",
            threshold
        );
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
            importer
                .unique_ngrams
                .store(importer.checkpoint.stats.unique_ngrams, Ordering::Relaxed);

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
            importer
                .unique_ngrams
                .store(importer.checkpoint.stats.unique_ngrams, Ordering::Relaxed);

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
    fn check_vocabulary_wal_consistency(vocabulary_path: &Path, checkpoint_path: &Path) {
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

        self.total_ngrams
            .fetch_add(ngrams_in_file, Ordering::Relaxed);
        Ok(ngrams_in_file)
    }

    /// Process a single local file using file transactions (single-trie mode).
    ///
    /// Uses INCREMENT semantics for cross-file count accumulation with
    /// atomic per-file commit/rollback.
    fn process_file_with_transaction(&self, path: &Path) -> Result<u64, ImportError> {
        let file_id = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        // Begin file transaction
        let mut tx = self
            .storage
            .begin_file_tx(file_id)
            .map_err(|e| ImportError::Trie(format!("Failed to begin file tx: {}", e)))?;

        let result = self.process_file_inner(&mut tx, path);

        match result {
            Ok(ngrams_in_file) => {
                // Commit atomically
                self.storage
                    .commit_file_tx(tx)
                    .map_err(|e| ImportError::Trie(format!("Failed to commit file tx: {}", e)))?;

                self.total_ngrams
                    .fetch_add(ngrams_in_file, Ordering::Relaxed);
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
                self.storage
                    .tx_increment_ngram(tx, &aggregated.ngram, aggregated.total_count)
                    .map_err(|e| ImportError::Trie(format!("Failed to increment ngram: {}", e)))?;
                ngrams_in_file += 1;
            }
        }

        // Flush final n-gram
        if let Some(aggregated) = aggregator.flush() {
            self.storage
                .tx_increment_ngram(tx, &aggregated.ngram, aggregated.total_count)
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
        let is_new = self
            .storage
            .store(ngram, count)
            .map_err(|e| ImportError::Trie(format!("Failed to store ngram '{}': {}", ngram, e)))?;

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
//
// `await_holding_lock` is allowed here: the importer guard is intentionally
// held across `import_http().await` so the shutdown handler's `try_lock()`
// observes the import as in-progress (guard held) rather than racing it.
#[cfg(feature = "google-books")]
#[allow(clippy::await_holding_lock)]
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
        // The importer guard is intentionally held across `import_http().await`:
        // the spawned shutdown handler uses `try_lock()` and must observe the
        // import as in-progress (guard held) rather than racing it.
        let mut importer = importer_ref.lock();
        importer.import_http(progress).await
    };

    // Cancel shutdown handler if import completed normally
    shutdown_handle.abort();

    result
}

/// Default checkpoint interval for periodic checkpoints (5 seconds).
pub const DEFAULT_CHECKPOINT_INTERVAL_MS: u64 = 5000;

#[cfg(feature = "google-books")]
mod checkpoint_ops;
mod cron;
pub use cron::{run_import_with_periodic_checkpoints, CheckpointState};
mod finalize;
#[cfg(feature = "google-books")]
mod import_ops;
#[cfg(feature = "google-books")]
mod mkn;
#[cfg(feature = "google-books")]
mod worker_pool;

// Cache-file download/cleanup helpers used by `--cache-files` mode.
#[cfg(feature = "google-books")]
mod cache;

#[cfg(feature = "google-books")]
use cache::{cleanup_cache_file, download_to_cache};

#[cfg(test)]
mod tests;
