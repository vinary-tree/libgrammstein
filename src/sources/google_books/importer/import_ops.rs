//! Top-level import drivers: HTTP (with three progress variants) and
//! local-file modes.
//!
//! - `import_files` — drives an import from already-downloaded local files.
//! - `import_http` / `import_http_with_progress` — drive an HTTP-fetched
//!   import using the [`super::worker_pool`] machinery.
//! - `import_http_reactive` — the event-driven variant used by the TUI.
//!
//! All variants share the same overall shape: iterate the requested orders,
//! fan prefix files out to concurrent workers, persist checkpoints, then
//! return the per-order aggregate stats.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::super::checkpoint::MknPhase;
use super::super::config::GoogleBooksConfig;
use super::super::events::{ImportCommand, ImportEvent, LogLevel};
use super::super::languages::{get_file_url, get_metadata, get_prefixes, is_supported};
use super::super::state_machine::CleanupResources;
use super::worker_pool::{
    process_prefix_file, worker_task, Job, JobOutcome, JobResult, PrefixOutcome,
    PrefixProcessingContext, WorkerSharedState, INITIAL_BACKOFF_MS, MAX_RETRIES,
};
use super::{
    GoogleBooksImporter, ImportError, ImportPhase, ImportProgress, ImportStats, WorkerUpdate,
};

impl GoogleBooksImporter {
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
}
