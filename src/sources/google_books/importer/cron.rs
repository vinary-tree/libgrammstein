//! Lock-free cron scheduler glue for periodic checkpointing.
//!
//! `CheckpointState` holds atomic + ArcSwap-backed views of the in-flight
//! import progress so the cron thread can perform durable checkpoints
//! without contending with the workers' RwLock guards.
//!
//! `run_import_with_periodic_checkpoints` wires the cron scheduler up to
//! the importer's main loop: every `checkpoint_interval_ms` it pulls the
//! current ArcSwap and triggers a checkpoint flush.

#![cfg(feature = "google-books")]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use super::super::checkpoint::ImportCheckpoint;
use super::super::storage::NgramStorage;
use super::{shutdown_signal, GoogleBooksImporter, ImportError, ImportProgress, ImportStats};

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
        self.storage
            .merge_and_rotate_vocabulary_wal()
            .map_err(|e| {
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
