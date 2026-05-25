//! Checkpoint persistence: save / save-async / cleanup.
//!
//! These methods drive the durable persistence of the importer's
//! `ImportCheckpoint` plus the underlying vocabulary and n-gram shard
//! storage. They're invoked at periodic intervals from the import drivers
//! in [`super::import_ops`] and at the end of import from
//! [`super::finalize`].

use std::sync::atomic::Ordering;

use super::super::checkpoint::ImportCheckpoint;
use super::super::events as gb_events;
use super::super::events::{ImportEvent, LogLevel};
use super::{GoogleBooksImporter, ImportError};

impl GoogleBooksImporter {
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
        self.total_ngrams
            .store(self.checkpoint.stats.ngrams_processed, Ordering::Relaxed);
        self.unique_ngrams
            .store(self.checkpoint.stats.unique_ngrams, Ordering::Relaxed);
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
        self.storage
            .merge_and_rotate_vocabulary_wal()
            .map_err(|e| {
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
        self.storage
            .sync_parallel(max_concurrent_syncs)
            .map_err(|e| ImportError::Trie(format!("Failed to sync storage: {}", e)))?;
        self.storage
            .checkpoint_parallel(max_concurrent_syncs)
            .map_err(|e| ImportError::Trie(format!("Failed to checkpoint storage: {}", e)))?;

        // Save checkpoint to the storage's metadata trie AFTER syncing all
        // data. `save_import_checkpoint` writes the checkpoint keys then
        // flushes the trie (truncating its WAL), keeping data and progress
        // tracking consistent.
        self.storage
            .save_import_checkpoint(&self.checkpoint)
            .map_err(|e| ImportError::Trie(format!("Failed to save checkpoint to trie: {}", e)))?;

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
        event_tx: Option<&tokio::sync::broadcast::Sender<gb_events::ImportEvent>>,
    ) -> Result<(), ImportError> {
        // Sync atomic counters FROM checkpoint stats (source of truth).
        self.total_ngrams
            .store(self.checkpoint.stats.ngrams_processed, Ordering::Relaxed);
        self.unique_ngrams
            .store(self.checkpoint.stats.unique_ngrams, Ordering::Relaxed);
        self.checkpoint.stats.elapsed_seconds = self.start_time.elapsed().as_secs();

        // CRITICAL: Rotate vocabulary WAL FIRST to ensure vocabulary indices are
        // durable before the checkpoint marks prefixes as completed.
        //
        // Note: We use rotate_vocabulary_wal() instead of checkpoint_vocabulary() to
        // avoid file bloat from repeated full trie serialization. WAL replay provides
        // crash recovery.
        self.storage
            .rotate_vocabulary_wal()
            .map_err(|e| ImportError::Trie(format!("Failed to rotate vocabulary WAL: {}", e)))?;

        // Start async checkpoint - this rotates WALs and returns immediately
        let handle = self
            .storage
            .checkpoint_async()
            .map_err(|e| ImportError::Trie(format!("Failed to start async checkpoint: {}", e)))?;

        log::debug!(
            "Async checkpoint initiated: {} resources rotating",
            handle.count()
        );

        // Wait for all syncs to complete using parallel waiting for sharded storage.
        // This reduces wait time from O(n) to O(1) for n shards by waiting on all
        // shard sync handles concurrently rather than sequentially.
        handle
            .wait_all_parallel()
            .map_err(|e| ImportError::Trie(format!("Async checkpoint sync failed: {}", e)))?;

        // Finish checkpoint - truncate WALs with bounded I/O parallelism
        // Create a progress callback that emits CheckpointProgress events
        let progress_callback: Option<Box<dyn Fn(usize, usize) + Send + Sync>> =
            event_tx.map(|tx| {
                let tx = tx.clone();
                Box::new(move |processed: usize, total: usize| {
                    let percent = if total > 0 {
                        (processed as f32 / total as f32) * 100.0
                    } else {
                        100.0
                    };
                    let _ = tx.send(gb_events::ImportEvent::CheckpointProgress {
                        shards_processed: processed,
                        total_shards: total,
                        percent_complete: percent,
                    });
                }) as Box<dyn Fn(usize, usize) + Send + Sync>
            });

        self.storage
            .checkpoint_async_finish_with_progress(
                Self::DEFAULT_CHECKPOINT_PARALLELISM,
                progress_callback,
            )
            .map_err(|e| ImportError::Trie(format!("Failed to finish async checkpoint: {}", e)))?;

        // Save checkpoint metadata AFTER syncing all data
        // This ensures consistency between data and progress tracking.
        self.storage
            .save_import_checkpoint(&self.checkpoint)
            .map_err(|e| ImportError::Trie(format!("Failed to save checkpoint to trie: {}", e)))?;

        log::debug!(
            "Async checkpoint saved: {}",
            self.checkpoint.progress_summary()
        );
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
}
