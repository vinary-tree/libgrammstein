//! Sync / checkpoint / flush operations across all shards.
//!
//! These are the operations the importer's periodic-checkpoint cron and the
//! finalize path invoke to drive durable state changes. They include the
//! lock-free-overlay flushes, the per-shard sync/checkpoint loops, and the
//! parallel variants that use the coordinator's persistent rayon pool.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use rayon::prelude::*;

use super::super::shard::ShardSyncState;
use super::{CheckpointHandle, CoordinatorError, CoordinatorResult, ShardCoordinator, ShardKey};

impl ShardCoordinator {
    /// Checkpoint shards whose lock-free overlay exceeds the entry threshold.
    ///
    /// Bounds per-shard overlay memory during high-parallelism imports: a shard over
    /// `threshold` is checkpointed (serializing its overlay snapshot to the on-disk
    /// image, which also reclaims overlay memory). Under the overlay-default write
    /// mode `flush_lockfree()` IS a `checkpoint()`, and it runs under a shared
    /// `shard.read()` guard so it does not stall the lock-free `increment_cas` writers.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Maximum lock-free entries per shard before checkpointing.
    ///
    /// # Returns
    ///
    /// The number of shards that were checkpointed.
    pub fn flush_lockfree_over_threshold(&self, threshold: u64) -> CoordinatorResult<usize> {
        let mut flushed = 0;
        let mut errors = Vec::new();

        for entry in self.shards.iter() {
            let key = entry.key().clone();
            let shard = entry.value();

            // Fast read-lock check (no contention with workers)
            let needs_flush = shard.read().lockfree_entry_count() > threshold;

            if needs_flush {
                // `flush_lockfree()` is `&self` (overlay checkpoint), so a shared read
                // guard suffices — workers keep writing during the checkpoint.
                let guard = shard.read();
                if let Err(e) = guard.flush_lockfree() {
                    errors.push(format!("Shard {}: {}", key, e));
                } else {
                    flushed += 1;
                }
            }
        }

        if errors.is_empty() {
            Ok(flushed)
        } else {
            Err(CoordinatorError::Checkpoint(errors.join("; ")))
        }
    }

    /// Get total lock-free entries across all shards.
    ///
    /// This is an approximate count useful for monitoring memory pressure
    /// from accumulated lock-free overlay data.
    pub fn total_lockfree_entries(&self) -> u64 {
        self.shards
            .iter()
            .map(|e| e.value().read().lockfree_entry_count())
            .sum()
    }

    /// Checkpoint all open shards (persist the overlay snapshot + retain WAL).
    ///
    /// Uses a blocking `shard.read()`: it coexists with concurrent lock-free
    /// `increment_cas` writers (the overlay snapshot is an immutable RCU point-in-time,
    /// so checkpointing no longer stalls writers) yet still waits for any exclusive
    /// writer rather than skipping a shard — a skipped shard would leave uncheckpointed
    /// data only in its WAL and double-count on resume (the checkpoint-resume bug class,
    /// `docs/debugging/checkpoint-resume-bug.md`).
    pub fn checkpoint_all(&self) -> CoordinatorResult<()> {
        let mut errors = Vec::new();

        for entry in self.shards.iter() {
            let key = entry.key().clone();
            let shard = entry.value();

            let guard = shard.read();
            if let Err(e) = guard.checkpoint() {
                errors.push(format!("Shard {}: {}", key, e));
            }
        }

        if errors.is_empty() {
            // Observability for the peak-heap validation (33.79 GB → < 16 GB): confirm the
            // resident-overlay budget is reclaiming cold nodes during checkpoints. Computed
            // only when debug logging is on, so it stays zero-cost on the hot path.
            if log::log_enabled!(log::Level::Debug) {
                let evict = self.aggregate_eviction_stats();
                log::debug!(
                    "checkpoint_all: {} shards; eviction nodes_evicted={} bytes_freed={} resident_bytes={}",
                    self.shards.len(),
                    evict.nodes_evicted,
                    evict.bytes_freed,
                    evict.resident_bytes
                );
            }
            Ok(())
        } else {
            Err(CoordinatorError::Checkpoint(errors.join("; ")))
        }
    }

    /// Sync all open shards (persist the overlay snapshot).
    pub fn sync_all(&self) -> CoordinatorResult<()> {
        let mut errors = Vec::new();

        for entry in self.shards.iter() {
            let key = entry.key().clone();
            let shard = entry.value();

            // Blocking `shard.read()`: coexists with lock-free writers but still waits
            // for any exclusive writer, so no shard is silently skipped (which would
            // double-count on WAL replay at resume).
            let guard = shard.read();
            if let Err(e) = guard.sync() {
                errors.push(format!("Shard {}: {}", key, e));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(CoordinatorError::Checkpoint(errors.join("; ")))
        }
    }

    /// Check if a specific shard is currently syncing.
    ///
    /// Workers can use this to defer writes to non-syncing shards,
    /// implementing the "defer-and-continue" pattern.
    ///
    /// # Arguments
    ///
    /// * `key` - The shard key to check
    ///
    /// # Returns
    ///
    /// `true` if the shard is currently syncing, `false` otherwise.
    /// Returns `false` if the shard doesn't exist (not yet created).
    pub fn is_shard_syncing(&self, key: &ShardKey) -> bool {
        self.shards
            .get(key)
            .map(|s| s.read().is_syncing())
            .unwrap_or(false)
    }

    /// Wait for a shard to finish syncing (with timeout).
    ///
    /// Returns immediately if the shard is not syncing.
    ///
    /// # Arguments
    ///
    /// * `key` - The shard key to wait on
    /// * `timeout` - Maximum time to wait
    ///
    /// # Returns
    ///
    /// `Ok(())` if sync completed or shard wasn't syncing,
    /// `Err(ShardError::SyncTimeout)` if timeout elapsed.
    pub fn wait_if_syncing(&self, key: &ShardKey, timeout: Duration) -> CoordinatorResult<()> {
        if let Some(shard) = self.shards.get(key) {
            let guard = shard.read();
            if guard.is_syncing() {
                guard.wait_for_sync(timeout)?;
            }
        }
        Ok(())
    }

    /// Sync all dirty shards in parallel using rayon.
    ///
    /// This method enables non-blocking checkpoint operations:
    /// - Workers can continue on non-syncing shards
    /// - Only workers targeting a syncing shard need to defer
    ///
    /// # Arguments
    ///
    /// * `max_concurrent` - Maximum number of shards to sync concurrently.
    ///   Higher values increase I/O parallelism but may overwhelm slow disks.
    ///   Recommended: 4-8 for SSDs, 1-2 for HDDs.
    ///
    /// # Returns
    ///
    /// Number of shards synced, or error if any sync failed.
    ///
    /// # Thread Safety
    ///
    /// Uses `sync_tracked()` which employs CAS to ensure only one sync
    /// operation per shard at a time. Formally verified in TLA+ spec.
    pub fn sync_all_parallel(&self, max_concurrent: usize) -> CoordinatorResult<usize> {
        // Collect dirty shard keys (read locks only, very fast)
        let dirty_shards: Vec<ShardKey> = self
            .shards
            .iter()
            .filter(|e| e.value().read().sync_state() == ShardSyncState::Dirty)
            .map(|e| e.key().clone())
            .collect();

        if dirty_shards.is_empty() {
            return Ok(0);
        }

        log::debug!(
            "Parallel sync: {} dirty shards with max {} concurrent",
            dirty_shards.len(),
            max_concurrent
        );

        // Parallel sync using the persistent rayon thread pool (built once
        // on first use, reused for the coordinator's lifetime).
        let pool = self.get_or_build_parallel_pool(max_concurrent.min(dirty_shards.len()))?;

        let errors: Vec<(ShardKey, String)> = pool.install(|| {
            dirty_shards
                .par_iter()
                .filter_map(|key| {
                    let shard = self.shards.get(key)?;

                    // Try to start sync (CAS: Dirty -> Syncing)
                    // If this returns false, another thread started the sync
                    {
                        let guard = shard.read();
                        if !guard.sync_coordinator().try_start_sync() {
                            return None; // Already syncing or clean
                        }
                    }

                    // Perform the actual sync (overlay snapshot; `sync()` is `&self`,
                    // so a shared read guard suffices and workers keep writing).
                    let guard = shard.read();
                    match guard.sync() {
                        Ok(()) => {
                            // Success: mark clean
                            let lsn = guard.stats().write_count.load(Ordering::Relaxed);
                            guard.sync_coordinator().complete_sync(lsn);
                            None
                        }
                        Err(e) => {
                            // Failure: mark failed
                            guard.sync_coordinator().fail_sync(&e.to_string());
                            Some((key.clone(), e.to_string()))
                        }
                    }
                })
                .collect()
        });

        let synced_count = dirty_shards.len() - errors.len();

        if errors.is_empty() {
            log::debug!("Parallel sync completed: {} shards synced", synced_count);
            Ok(synced_count)
        } else {
            let error_messages: Vec<String> = errors
                .iter()
                .map(|(k, e)| format!("{}: {}", k, e))
                .collect();
            Err(CoordinatorError::ParallelSyncFailed {
                errors: error_messages.join("; "),
            })
        }
    }

    /// Start async sync on all dirty shards (non-blocking).
    ///
    /// This initiates WAL segment rotation on each dirty shard:
    /// - Each shard rotates its WAL to a new segment (O(1) operation)
    /// - New writes go to the new segment immediately
    /// - Background threads sync the old segments
    /// - Workers can continue without waiting for fsync
    ///
    /// # Returns
    ///
    /// A `CheckpointHandle` that can be used to:
    /// - Check if all syncs completed (`all_synced()`)
    /// - Wait for all syncs to complete (`wait_all()`)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Start async sync on all dirty shards
    /// let handle = coordinator.sync_all_async()?;
    ///
    /// // Workers continue immediately...
    /// process_more_data();
    ///
    /// // When durability is needed
    /// handle.wait_all()?;
    /// ```
    pub fn sync_all_async(&self) -> CoordinatorResult<CheckpointHandle> {
        let mut handles = Vec::new();

        for entry in self.shards.iter() {
            let shard = entry.value();
            let guard = shard.read();

            // Only sync shards that have dirty data
            if guard.is_dirty() {
                if let Some(handle) = guard.sync_async()? {
                    handles.push(handle);
                }
            }
        }

        log::debug!(
            "Async sync initiated: {} shards rotating WAL",
            handles.len()
        );

        Ok(CheckpointHandle { handles })
    }

    /// Check if any shard is currently syncing.
    ///
    /// This can be used by workers to decide whether to defer operations
    /// during a checkpoint.
    pub fn any_shard_syncing(&self) -> bool {
        self.shards
            .iter()
            .any(|entry| entry.value().read().is_syncing())
    }

    /// Coordinated async checkpoint for maximum throughput.
    ///
    /// This is the recommended checkpoint method for high-throughput workloads.
    /// It provides the same durability guarantees as `coordinated_checkpoint()`
    /// but with minimal blocking:
    ///
    /// 1. **Fast rotation (~1ms total)**: Rotate all shards to new WAL segments
    /// 2. **Non-blocking**: Return immediately - workers continue writing
    /// 3. **Background sync**: Old segments sync in background threads
    /// 4. **Explicit wait**: Call `handle.wait_all()` when durability is needed
    ///
    /// # Performance
    ///
    /// With 100 shards at 50ms fsync each:
    /// - `coordinated_checkpoint()`: ~5000ms blocking (sequential)
    /// - `coordinated_checkpoint_parallel(8)`: ~625ms blocking
    /// - `coordinated_checkpoint_async()`: ~1-10ms, then workers continue
    ///
    /// The async checkpoint provides **40-50x less blocking** than sequential,
    /// and **~60x less blocking** than parallel checkpoint.
    ///
    /// # Usage Pattern
    ///
    /// ```ignore
    /// // Periodic checkpoint during import
    /// let handle = coordinator.coordinated_checkpoint_async()?;
    ///
    /// // Workers continue immediately on new WAL segments
    /// // ... more imports ...
    ///
    /// // At end of batch or before reporting progress
    /// handle.wait_all()?;
    ///
    /// // Now safe to update checkpoint metadata
    /// coordinator.coordinated_checkpoint_finish(8)?;
    /// ```
    pub fn coordinated_checkpoint_async(&self) -> CoordinatorResult<CheckpointHandle> {
        let start = Instant::now();

        // Start async sync on all dirty shards
        let handle = self.sync_all_async()?;

        let elapsed = start.elapsed();
        log::debug!(
            "Async checkpoint rotation completed: {} shards in {:.2}ms",
            handle.count(),
            elapsed.as_secs_f64() * 1000.0
        );

        Ok(handle)
    }

    /// Finish a checkpoint after async sync is complete.
    ///
    /// Call this after `coordinated_checkpoint_async().wait_all()` to:
    /// 1. Truncate the WALs (fast - data already synced)
    /// 2. Update global checkpoint state
    /// 3. Save checkpoint metadata
    ///
    /// This is the "commit" phase of an async checkpoint.
    ///
    /// # Arguments
    ///
    /// * `max_concurrent` - Maximum shards to persist in parallel.
    ///   Recommended: 4-8 for NVMe SSDs, 2-4 for SATA SSDs, 1-2 for HDDs.
    ///
    /// # Parallelization
    ///
    /// Uses a bounded rayon thread pool to checkpoint shards in parallel.
    /// Since the expensive fsync was already done during the async phase,
    /// this is primarily truncating WALs and updating state, which is fast
    /// (~1ms per shard). Bounded parallelism prevents overwhelming disk I/O
    /// when many dirty shards need to persist simultaneously.
    pub fn coordinated_checkpoint_finish(&self, max_concurrent: usize) -> CoordinatorResult<()> {
        self.coordinated_checkpoint_finish_with_progress(max_concurrent, None::<fn(usize, usize)>)
    }

    /// Finish a checkpoint with progress reporting.
    ///
    /// Same as `coordinated_checkpoint_finish()` but emits `CheckpointProgress` events
    /// through the provided channel.
    ///
    /// # Arguments
    ///
    /// * `max_concurrent` - Maximum shards to persist in parallel.
    /// * `progress_callback` - Optional callback invoked after each shard completes.
    ///   Receives (shards_processed, total_shards).
    pub fn coordinated_checkpoint_finish_with_progress<F>(
        &self,
        max_concurrent: usize,
        progress_callback: Option<F>,
    ) -> CoordinatorResult<()>
    where
        F: Fn(usize, usize) + Send + Sync,
    {
        let start = Instant::now();

        // Collect shard keys to process in parallel
        let shard_keys: Vec<ShardKey> = self.shards.iter().map(|e| e.key().clone()).collect();
        let total_shards = shard_keys.len();

        // Reuse the coordinator's persistent rayon pool for I/O parallelism
        // instead of building a fresh one on every call.
        let pool = self.get_or_build_parallel_pool(max_concurrent.min(shard_keys.len()).max(1))?;

        // Atomic counter for progress tracking
        let shards_processed = AtomicUsize::new(0);

        // Parallel checkpoint (truncate WALs) - collect state updates
        // OPTIMIZATION: Skip shards that are already clean (no dirty data) to avoid
        // expensive persist_to_disk() calls. Clean shards still contribute their state
        // to the checkpoint metadata.
        let results: Vec<Result<(ShardKey, u64, u64, HashSet<String>, Option<String>), String>> =
            pool.install(|| {
                shard_keys
                    .par_iter()
                    .filter_map(|key| {
                        let shard = self.shards.get(key)?;

                        // Check if shard is clean with a read lock first (fast, non-blocking)
                        let result = {
                            let guard = shard.read();
                            if !guard.is_dirty() {
                                // Shard is clean - still collect its state for checkpoint metadata
                                let state = guard.checkpoint_state();
                                Some(Ok((
                                    key.clone(),
                                    guard.len() as u64,
                                    state.ngrams_processed,
                                    state.completed_prefixes.clone(),
                                    state.current_prefix.clone(),
                                )))
                            } else {
                                None
                            }
                        };

                        let result = result.unwrap_or_else(|| {
                            // Shard is dirty - checkpoint it. `checkpoint()` is `&self`
                            // (overlay snapshot), so a shared read guard suffices and
                            // workers keep writing during the checkpoint.
                            let guard = shard.read();

                            // Checkpoint (retain WAL) - fast since data already synced
                            if let Err(e) = guard.checkpoint() {
                                return Err(format!("Shard {}: {}", key, e));
                            }

                            // Collect state for global checkpoint update (done outside parallel section
                            // to avoid lock contention on checkpoint_manager)
                            let state = guard.checkpoint_state();
                            Ok((
                                key.clone(),
                                guard.len() as u64,
                                state.ngrams_processed,
                                state.completed_prefixes.clone(),
                                state.current_prefix.clone(),
                            ))
                        });

                        // Update progress counter and emit event
                        let processed = shards_processed.fetch_add(1, Ordering::Relaxed) + 1;
                        if let Some(ref callback) = progress_callback {
                            callback(processed, total_shards);
                        }

                        Some(result)
                    })
                    .collect()
            });

        // Separate errors from successful results
        let mut errors = Vec::new();
        let mut shard_states = Vec::new();

        for result in results {
            match result {
                Ok(state) => shard_states.push(state),
                Err(e) => errors.push(e),
            }
        }

        // Update global checkpoint sequentially (single lock, fast operations)
        if let Some(ref manager) = self.checkpoint_manager {
            let mut mgr = manager.lock();
            let ckpt = mgr.checkpoint_mut();
            let now = std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);

            for (key, entry_count, ngrams_processed, completed_prefixes, current_prefix) in
                shard_states
            {
                // Get shard path for the record
                let path = self.config.shard_path(&key.as_file_stem());
                let record = ckpt.get_or_create_shard(&key, &path);
                record.entry_count = entry_count;
                record.ngrams_processed = ngrams_processed;
                record.completed_prefixes = completed_prefixes;
                record.current_prefix = current_prefix;
                record.last_checkpoint_time = now;
            }

            // Save global checkpoint
            mgr.save()?;
        }

        let elapsed = start.elapsed();
        log::debug!(
            "Async checkpoint finish completed: {} shards in {:.2}ms",
            self.shards.len(),
            elapsed.as_secs_f64() * 1000.0
        );

        if errors.is_empty() {
            Ok(())
        } else {
            Err(CoordinatorError::Checkpoint(errors.join("; ")))
        }
    }

    /// Perform a coordinated checkpoint with parallel WAL flushing.
    ///
    /// This method provides the same guarantees as `coordinated_checkpoint()`
    /// but with significantly better performance for large shard counts:
    ///
    /// 1. Vocabulary checkpoint (synchronous - single resource)
    /// 2. Parallel WAL sync via `sync_all_parallel()`
    /// 3. Sequential state collection from all shards (read locks, fast)
    /// 4. Sequential WAL checkpoint/truncate (write locks)
    /// 5. Atomic global checkpoint JSON save
    ///
    /// Workers can continue on non-syncing shards during step 2, only
    /// blocking when they need to access a shard that is currently syncing.
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
    pub fn coordinated_checkpoint_parallel(
        &self,
        max_concurrent_syncs: usize,
    ) -> CoordinatorResult<()> {
        let start = Instant::now();

        // Step 1: Parallel WAL sync
        let synced_count = self.sync_all_parallel(max_concurrent_syncs)?;

        // Step 2: Sequential state collection and checkpoint
        let mut errors = Vec::new();

        for entry in self.shards.iter() {
            let key = entry.key().clone();
            let shard = entry.value();

            // `checkpoint()` is `&self` (overlay snapshot), so a shared read guard
            // suffices and workers keep writing during the checkpoint.
            let guard = shard.read();

            // Checkpoint (retain WAL) - this is fast since data is already synced
            if let Err(e) = guard.checkpoint() {
                errors.push(format!("Shard {}: {}", key, e));
                continue;
            }

            // Update global checkpoint with shard state
            if let Some(ref manager) = self.checkpoint_manager {
                let mut mgr = manager.lock();
                let ckpt = mgr.checkpoint_mut();
                let state = guard.checkpoint_state();

                let record = ckpt.get_or_create_shard(&key, guard.path());
                record.entry_count = guard.len() as u64;
                record.ngrams_processed = state.ngrams_processed;
                record.completed_prefixes = state.completed_prefixes.clone();
                record.current_prefix = state.current_prefix.clone();
                record.last_checkpoint_time = std::time::SystemTime::now()
                    .duration_since(std::time::SystemTime::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
            }
        }

        // Step 3: Save global checkpoint
        if let Some(ref manager) = self.checkpoint_manager {
            manager.lock().save()?;
        }

        let elapsed = start.elapsed();
        log::debug!(
            "Parallel checkpoint completed: {} shards synced, {} total shards, {}ms elapsed",
            synced_count,
            self.shards.len(),
            elapsed.as_millis()
        );

        if errors.is_empty() {
            Ok(())
        } else {
            Err(CoordinatorError::Checkpoint(errors.join("; ")))
        }
    }

    /// Retry failed syncs (after `sync_all_parallel()` returned errors).
    ///
    /// This resets shards from SyncFailed to Dirty so they can be synced again.
    pub fn retry_failed_syncs(&self) -> usize {
        let mut retried = 0;
        for entry in self.shards.iter() {
            let guard = entry.value().read();
            if guard.sync_coordinator().retry_sync() {
                retried += 1;
            }
        }
        retried
    }
}
