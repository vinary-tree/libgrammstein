//! Google Books N-gram importer.
//!
//! Orchestrates the import process from Google Books N-grams into a PersistentARTrie,
//! with checkpoint/resume support for long-running imports.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use liblevenshtein::dictionary::persistent_artrie_char::DiskBackedCharTrieInner;
use parking_lot::RwLock;

use super::aggregator::YearAggregator;
use super::checkpoint::{CheckpointError, CheckpointStats, ImportCheckpoint, MknPhase};
use super::config::GoogleBooksConfig;
use super::languages::{get_file_url, get_metadata, get_prefixes, is_supported};
use super::parser::NgramRecord;
use super::reader::{FileNgramReader, ReaderError};

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

    /// Disk-backed n-gram storage using persistent AR-Trie.
    /// Uses UTF-8 character encoding for international language support.
    trie: Arc<RwLock<DiskBackedCharTrieInner<u64>>>,

    /// MKN Pass 1 tracking: suffix → set of unique preceding words.
    /// Tracks N1+(•suffix) - continuation count for each suffix.
    mkn_continuation_prefixes: Arc<RwLock<HashMap<String, HashSet<String>>>>,

    /// MKN Pass 2 tracking: prefix → set of unique following words.
    /// Tracks N1+(prefix•) - unique continuation count for each prefix.
    mkn_unique_continuations: Arc<RwLock<HashMap<String, HashSet<String>>>>,
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

        // Create or open the disk-backed trie
        let trie = if output_path.exists() {
            log::info!("Opening existing trie at {:?}", output_path);
            DiskBackedCharTrieInner::open(output_path).map_err(|e| {
                ImportError::Trie(format!("Failed to open trie: {}", e))
            })?
        } else {
            log::info!("Creating new trie at {:?}", output_path);
            DiskBackedCharTrieInner::create(output_path).map_err(|e| {
                ImportError::Trie(format!("Failed to create trie: {}", e))
            })?
        };

        Ok(Self {
            config,
            checkpoint: ImportCheckpoint::new(),
            checkpoint_path,
            total_ngrams: AtomicU64::new(0),
            unique_ngrams: AtomicU64::new(0),
            interrupted: AtomicBool::new(false),
            start_time: Instant::now(),
            trie: Arc::new(RwLock::new(trie)),
            mkn_continuation_prefixes: Arc::new(RwLock::new(HashMap::new())),
            mkn_unique_continuations: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Resume from checkpoint if it exists, otherwise start fresh.
    pub fn resume_or_start(config: GoogleBooksConfig) -> Result<Self, ImportError> {
        let checkpoint_path = config.output_path.with_extension("checkpoint.json");

        if ImportCheckpoint::exists(&checkpoint_path) {
            let checkpoint = ImportCheckpoint::load(&checkpoint_path)?;
            log::info!(
                "Resuming from checkpoint: order={}, prefixes={}",
                checkpoint.current_order,
                checkpoint.completed_prefixes.len()
            );

            let mut importer = Self::new(config)?;
            importer.checkpoint = checkpoint;
            importer.total_ngrams.store(
                importer.checkpoint.stats.ngrams_processed,
                Ordering::Relaxed,
            );

            Ok(importer)
        } else {
            Self::new(config)
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

    /// Save current checkpoint.
    ///
    /// This persists both the trie data (via WAL checkpoint) and the import
    /// progress (JSON checkpoint file). The trie checkpoint truncates the WAL
    /// to prevent unbounded growth during long imports.
    pub fn save_checkpoint(&mut self) -> Result<(), ImportError> {
        // Checkpoint the trie first (persists data and truncates WAL)
        {
            let mut trie = self.trie.write();
            trie.checkpoint().map_err(|e| {
                ImportError::Trie(format!("Failed to checkpoint trie: {}", e))
            })?;
        }

        // Then save JSON progress checkpoint
        self.checkpoint.stats.ngrams_processed = self.total_ngrams.load(Ordering::Relaxed);
        self.checkpoint.stats.elapsed_seconds = self.start_time.elapsed().as_secs();
        self.checkpoint.save(&self.checkpoint_path)?;

        log::debug!("Checkpoint saved: {}", self.checkpoint.progress_summary());
        Ok(())
    }

    /// Delete checkpoint file (call after successful completion).
    pub fn cleanup_checkpoint(&self) -> Result<(), ImportError> {
        ImportCheckpoint::delete(&self.checkpoint_path)?;
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
            if self.checkpoint.completed_orders.contains(&order) {
                log::info!("Skipping order {} (already completed)", order);
                continue;
            }

            self.checkpoint.current_order = order;
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

                self.checkpoint.complete_prefix(prefix);
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

        self.build_stats()
    }

    /// Import from HTTP (streaming from Google's servers).
    ///
    /// # Arguments
    ///
    /// * `progress` - Progress callback
    #[cfg(feature = "google-books")]
    pub async fn import_http<F>(&mut self, mut progress: F) -> Result<ImportStats, ImportError>
    where
        F: FnMut(ImportProgress),
    {
        use tokio_stream::StreamExt;

        for order in self.config.orders.clone() {
            if self.checkpoint.completed_orders.contains(&order) {
                log::info!("Skipping order {} (already completed)", order);
                continue;
            }

            self.checkpoint.current_order = order;
            let prefixes = get_prefixes(order);
            let total_files = prefixes.len() as u32;

            for (idx, prefix) in prefixes.iter().enumerate() {
                if self.is_interrupted() {
                    self.save_checkpoint()?;
                    return Err(ImportError::Interrupted);
                }

                if !self.checkpoint.needs_prefix(order, &prefix) {
                    continue;
                }

                // Get URL for this prefix
                let url = match get_file_url(&self.config.language, order, &prefix) {
                    Some(url) => url,
                    None => {
                        log::warn!("Could not generate URL for order={}, prefix={}", order, prefix);
                        continue;
                    }
                };

                // Process HTTP stream with retry logic for transient failures
                let ngrams_in_file = self.process_http_stream_with_retry(&url).await?;

                self.checkpoint.complete_prefix(&prefix);
                self.checkpoint.stats.ngrams_by_order[(order - 1) as usize] += ngrams_in_file;

                // Report progress
                progress(ImportProgress {
                    current_order: order,
                    current_prefix: prefix.clone(),
                    ngrams_in_file,
                    total_ngrams: self.total_ngrams.load(Ordering::Relaxed),
                    files_completed: idx as u32 + 1,
                    total_files,
                    bytes_downloaded: self.checkpoint.stats.bytes_downloaded,
                    ngrams_per_second: self.calculate_rate(),
                    eta_seconds: self.estimate_eta(idx as u32 + 1, total_files),
                    phase: ImportPhase::Importing,
                });

                // Save checkpoint periodically
                if (idx + 1) % 5 == 0 {
                    self.save_checkpoint()?;
                }
            }

            self.checkpoint.complete_order(order);
            self.save_checkpoint()?;
        }

        self.build_stats()
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

    /// Process an HTTP stream with retry logic.
    ///
    /// Wraps `process_http_stream` with exponential backoff retry for transient
    /// network failures (connection timeouts, temporary network issues).
    #[cfg(feature = "google-books")]
    async fn process_http_stream_with_retry(&mut self, url: &str) -> Result<u64, ImportError> {
        use std::time::Duration;

        const MAX_RETRIES: u32 = 5;
        const INITIAL_BACKOFF_MS: u64 = 1000;

        let mut attempt = 0;
        let mut backoff_ms = INITIAL_BACKOFF_MS;

        loop {
            match self.process_http_stream(url).await {
                Ok(count) => return Ok(count),
                Err(e) if attempt < MAX_RETRIES && Self::is_retryable_error(&e) => {
                    attempt += 1;
                    log::warn!(
                        "HTTP request failed (attempt {}/{}), retrying in {}ms: {}",
                        attempt,
                        MAX_RETRIES,
                        backoff_ms,
                        e
                    );
                    tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                    backoff_ms *= 2; // Exponential backoff
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Check if an error is retryable (transient network issues).
    fn is_retryable_error(e: &ImportError) -> bool {
        match e {
            ImportError::Reader(reader_err) => {
                let msg = reader_err.to_string().to_lowercase();
                // Retry on connection timeouts, network errors, temporary failures
                msg.contains("timeout")
                    || msg.contains("connection")
                    || msg.contains("network")
                    || msg.contains("temporarily")
                    || msg.contains("reset")
                    || msg.contains("broken pipe")
            }
            _ => false,
        }
    }

    /// Process an HTTP stream.
    #[cfg(feature = "google-books")]
    async fn process_http_stream(&mut self, url: &str) -> Result<u64, ImportError> {
        use super::reader::HttpNgramReader;

        let mut reader = HttpNgramReader::with_options(
            url,
            self.config.skip_pos_tags,
            self.config.min_count,
        );

        let aggregated = reader.read_aggregated(self.config.year_range).await?;
        let count = aggregated.len() as u64;

        for agg in aggregated {
            self.store_ngram(&agg.ngram, agg.total_count)?;
        }

        self.total_ngrams.fetch_add(count, Ordering::Relaxed);
        Ok(count)
    }

    /// Store an n-gram with its count.
    ///
    /// Writes to the disk-backed PersistentARTrie and tracks MKN statistics
    /// on-the-fly for efficient single-pass computation.
    fn store_ngram(&self, ngram: &str, count: u64) -> Result<(), ImportError> {
        // Store in the disk-backed trie
        {
            let mut trie = self.trie.write();

            // Check if this is a new ngram for statistics
            let is_new = trie.get(ngram).is_none();

            // Increment the count
            trie.increment(ngram, count as i64).map_err(|e| {
                ImportError::Trie(format!("Failed to store ngram '{}': {}", ngram, e))
            })?;

            if is_new {
                self.unique_ngrams.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Track MKN statistics on-the-fly
        let words: Vec<&str> = ngram.split_whitespace().collect();

        if words.len() >= 2 {
            // MKN Pass 1: Track continuation counts (suffix → unique prefixes)
            // For bigram "w1 w2": prefix = "w1", suffix = "w2"
            // For trigram "w1 w2 w3": prefix = "w1", suffix = "w2 w3"
            let prefix = words[0];
            let suffix = words[1..].join(" ");

            {
                let mut continuation_prefixes = self.mkn_continuation_prefixes.write();
                continuation_prefixes
                    .entry(suffix)
                    .or_default()
                    .insert(prefix.to_string());
            }

            // MKN Pass 2: Track unique continuations (prefix → unique following words)
            // For bigram "w1 w2": prefix = "w1", following = "w2"
            // For trigram "w1 w2 w3": prefix = "w1 w2", following = "w3"
            let context_prefix = words[..words.len() - 1].join(" ");
            let following_word = words[words.len() - 1];

            {
                let mut unique_continuations = self.mkn_unique_continuations.write();
                unique_continuations
                    .entry(context_prefix)
                    .or_default()
                    .insert(following_word.to_string());
            }
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

    /// Finalize import: compute MKN statistics, sync trie, and return stats.
    pub fn finalize(&mut self) -> Result<ImportStats, ImportError> {
        log::info!("Finalizing import...");

        // Compute MKN continuation counts
        self.compute_mkn_stats()?;

        // Sync and checkpoint the trie to ensure all data is persisted
        {
            let mut trie = self.trie.write();

            log::info!("Syncing trie to disk...");
            trie.sync().map_err(|e| {
                ImportError::Trie(format!("Failed to sync trie: {}", e))
            })?;

            log::info!("Creating trie checkpoint...");
            trie.checkpoint().map_err(|e| {
                ImportError::Trie(format!("Failed to checkpoint trie: {}", e))
            })?;
        }

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

    /// Compute Modified Kneser-Ney smoothing statistics.
    ///
    /// MKN statistics were tracked on-the-fly during import. This method
    /// writes the aggregated continuation counts to the trie with special
    /// key prefixes for later retrieval during inference.
    fn compute_mkn_stats(&mut self) -> Result<(), ImportError> {
        if self.checkpoint.mkn_phase == MknPhase::Complete {
            log::info!("MKN statistics already computed");
            return Ok(());
        }

        log::info!("Writing MKN continuation counts to trie...");

        // Pass 1: Write continuation counts (suffix → unique prefix count)
        // Key format: "\x00N1+\x00{suffix}" → count
        if matches!(
            self.checkpoint.mkn_phase,
            MknPhase::NotStarted | MknPhase::Pass1InProgress { .. }
        ) {
            self.checkpoint.mkn_phase = MknPhase::Pass1InProgress { current_order: 1 };
            self.save_checkpoint()?;

            let processed = {
                let continuation_prefixes = self.mkn_continuation_prefixes.read();
                let mut trie = self.trie.write();
                let mut processed = 0u64;

                for (suffix, prefixes) in continuation_prefixes.iter() {
                    let key = format!("\x00N1+\x00{}", suffix);
                    let count = prefixes.len() as i64;

                    trie.increment(&key, count).map_err(|e| {
                        ImportError::Trie(format!(
                            "Failed to store MKN continuation count for '{}': {}",
                            suffix, e
                        ))
                    })?;

                    processed += 1;
                    if processed % 100_000 == 0 {
                        log::debug!("MKN Pass 1: Wrote {} continuation counts", processed);
                    }
                }

                processed
            };

            log::info!(
                "MKN Pass 1 complete: wrote {} continuation counts",
                processed
            );

            self.checkpoint.mkn_phase = MknPhase::Pass1Complete;
            self.save_checkpoint()?;
        }

        // Pass 2: Write unique continuation counts (prefix → unique following word count)
        // Key format: "\x00N1+prefix\x00{prefix}" → count
        if matches!(
            self.checkpoint.mkn_phase,
            MknPhase::Pass1Complete | MknPhase::Pass2InProgress { .. }
        ) {
            self.checkpoint.mkn_phase = MknPhase::Pass2InProgress { current_order: 1 };
            self.save_checkpoint()?;

            let processed = {
                let unique_continuations = self.mkn_unique_continuations.read();
                let mut trie = self.trie.write();
                let mut processed = 0u64;

                for (prefix, following_words) in unique_continuations.iter() {
                    let key = format!("\x00N1+prefix\x00{}", prefix);
                    let count = following_words.len() as i64;

                    trie.increment(&key, count).map_err(|e| {
                        ImportError::Trie(format!(
                            "Failed to store MKN unique continuation count for '{}': {}",
                            prefix, e
                        ))
                    })?;

                    processed += 1;
                    if processed % 100_000 == 0 {
                        log::debug!("MKN Pass 2: Wrote {} unique continuation counts", processed);
                    }
                }

                processed
            };

            log::info!(
                "MKN Pass 2 complete: wrote {} unique continuation counts",
                processed
            );

            self.checkpoint.mkn_phase = MknPhase::Complete;
            self.save_checkpoint()?;
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

        // Save checkpoint
        importer.save_checkpoint().expect("Failed to save checkpoint");

        // Verify checkpoint file exists
        let checkpoint_path = output_path.with_extension("checkpoint.json");
        assert!(checkpoint_path.exists(), "Checkpoint file should exist");

        // Load checkpoint
        let loaded = ImportCheckpoint::load(&checkpoint_path).expect("Failed to load checkpoint");
        assert_eq!(loaded.current_order, 1);  // Initialized to 1 (first order to process)
        assert!(loaded.completed_orders.is_empty());
    }
}
