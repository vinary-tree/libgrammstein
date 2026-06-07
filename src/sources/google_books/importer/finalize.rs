//! Import finalization: MKN compute, shard merge, cleanup, and final stats.
//!
//! Called after all prefix-files have been streamed in. Walks the storage
//! through a deterministic teardown:
//!
//! 1. Compute MKN statistics ([`super::mkn`]).
//! 2. Merge shards into the final output trie when sharding was used.
//! 3. Optionally delete the per-shard files if the caller requested cleanup.
//! 4. Persist the final checkpoint and build the user-facing `ImportStats`.

use std::time::Instant;

use super::super::events::ImportEvent;
use super::super::sharding::MergeCoordinator;
use super::{GoogleBooksImporter, ImportError, ImportStats};

impl GoogleBooksImporter {
    /// Finalize import: compute MKN statistics, sync storage, and return stats.
    pub fn finalize(&mut self) -> Result<ImportStats, ImportError> {
        self.finalize_with_events_inner(None)
    }

    /// Finalize import with event emission for TUI progress updates.
    pub(super) fn finalize_with_events(
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
        self.storage
            .checkpoint_vocabulary()
            .map_err(|e| ImportError::Trie(format!("Failed to checkpoint vocabulary: {}", e)))?;

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
    pub(super) async fn merge_shards(
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

        log::info!(
            "Starting merge of {} shards (~{} n-grams)",
            shard_count,
            estimated_ngrams
        );

        // Emit MergeStarted event
        log::debug!(
            "[IMPORTER] Sending MergeStarted: shard_count={}, estimated_ngrams={}",
            shard_count,
            estimated_ngrams
        );
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
                log::debug!(
                    "[IMPORTER] Sending MergeCompleted: total_ngrams={}, bytes_written={}",
                    stats.total_ngrams,
                    stats.bytes_written
                );
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
        let coordinator = self
            .storage
            .as_sharded()
            .ok_or_else(|| ImportError::Trie("Expected sharded storage for cleanup".to_string()))?;

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
}
