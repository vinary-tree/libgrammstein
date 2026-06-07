//! Import-progress / checkpoint metadata operations on the coordinator.
//!
//! These methods coordinate the global ImportCheckpoint's view of which
//! prefixes are in-progress / completed and provide ergonomic accessors for
//! recovery checks and arbitrary checkpoint-metadata key/value storage.

use std::collections::HashSet;
use std::sync::atomic::Ordering;

use super::super::checkpoint::{ImportPhase, ImportState};
use super::{CoordinatorError, CoordinatorResult, ShardCoordinator, ShardKey};

impl ShardCoordinator {
    // ========== Checkpoint Coordination Methods ==========

    /// Check if recovery is needed from a previous interrupted import.
    pub fn needs_recovery(&self) -> bool {
        if let Some(ref manager) = self.checkpoint_manager {
            manager.lock().needs_recovery()
        } else {
            false
        }
    }

    /// Get the current import state.
    pub fn import_state(&self) -> Option<ImportState> {
        self.checkpoint_manager
            .as_ref()
            .map(|mgr| mgr.lock().checkpoint().import_state.clone())
    }

    /// Start a new import (sets checkpoint state to InProgress).
    pub fn start_import(&self) -> CoordinatorResult<()> {
        if let Some(ref manager) = self.checkpoint_manager {
            let mut mgr = manager.lock();

            // Check if recovery is needed
            if mgr.needs_recovery() {
                return Err(CoordinatorError::RecoveryRequired);
            }

            mgr.checkpoint_mut().start_import();
            mgr.save()?;
        }
        Ok(())
    }

    /// Resume an interrupted import.
    pub fn resume_import(&self) -> CoordinatorResult<()> {
        if let Some(ref manager) = self.checkpoint_manager {
            let mut mgr = manager.lock();
            mgr.resume();
            mgr.save()?;
        }
        Ok(())
    }

    /// Set the current import phase.
    pub fn set_import_phase(&self, phase: ImportPhase) -> CoordinatorResult<()> {
        if let Some(ref manager) = self.checkpoint_manager {
            let mut mgr = manager.lock();
            mgr.checkpoint_mut().set_phase(phase);
            mgr.maybe_save()?;
        }
        Ok(())
    }

    /// Mark import as completed.
    pub fn complete_import(&self) -> CoordinatorResult<()> {
        if let Some(ref manager) = self.checkpoint_manager {
            let mut mgr = manager.lock();
            let total = self.stats.total_ngrams.load(Ordering::Relaxed);
            let unique = self.stats.unique_ngrams.load(Ordering::Relaxed);
            mgr.checkpoint_mut().complete_import(total, unique);
            mgr.save()?;
        }
        Ok(())
    }

    /// Mark import as failed.
    pub fn fail_import(&self, error: impl Into<String>) -> CoordinatorResult<()> {
        if let Some(ref manager) = self.checkpoint_manager {
            let mut mgr = manager.lock();
            mgr.checkpoint_mut().fail_import(error);
            mgr.save()?;
        }
        Ok(())
    }

    /// Get all completed prefixes across all shards.
    pub fn all_completed_prefixes(&self) -> HashSet<String> {
        if let Some(ref manager) = self.checkpoint_manager {
            manager.lock().checkpoint().all_completed_prefixes()
        } else {
            // Fall back to querying shards directly
            self.shards
                .iter()
                .flat_map(|e| {
                    e.value()
                        .read()
                        .checkpoint_state()
                        .completed_prefixes
                        .iter()
                        .cloned()
                        .collect::<Vec<_>>()
                })
                .collect()
        }
    }

    /// Check if a prefix has been completed.
    pub fn is_prefix_completed(&self, prefix: &str) -> bool {
        self.all_completed_prefixes().contains(prefix)
    }

    /// Get completed prefixes for a specific n-gram order.
    ///
    /// Returns all prefixes that have been marked complete in shards
    /// associated with the given order.
    pub fn completed_prefixes_for_order(&self, order: u8) -> HashSet<String> {
        if let Some(ref manager) = self.checkpoint_manager {
            manager
                .lock()
                .checkpoint()
                .completed_prefixes_for_order(order)
        } else {
            // Fall back to querying shards directly
            self.shards
                .iter()
                .filter(|e| {
                    // Only include shards for this order or shards that contain all orders
                    e.key().order.is_none() || e.key().order == Some(order)
                })
                .flat_map(|e| {
                    e.value()
                        .read()
                        .checkpoint_state()
                        .completed_prefixes
                        .iter()
                        .cloned()
                        .collect::<Vec<_>>()
                })
                .collect()
        }
    }

    /// Mark a prefix as currently being processed.
    pub fn set_current_prefix(
        &self,
        shard_key: &ShardKey,
        prefix: Option<&str>,
    ) -> CoordinatorResult<()> {
        // Update shard's checkpoint state
        if let Some(shard) = self.shards.get(shard_key) {
            let mut guard = shard.write();
            guard.set_current_prefix(prefix);
        }

        // Update global checkpoint
        if let Some(ref manager) = self.checkpoint_manager {
            let mut mgr = manager.lock();
            mgr.checkpoint_mut().set_current_prefix(shard_key, prefix);
            mgr.maybe_save()?;
        }

        Ok(())
    }

    /// Mark a prefix as completed in a shard and update global checkpoint.
    pub fn mark_prefix_completed(
        &self,
        shard_key: &ShardKey,
        prefix: &str,
    ) -> CoordinatorResult<()> {
        // Update shard's checkpoint state
        if let Some(shard) = self.shards.get(shard_key) {
            let mut guard = shard.write();
            guard.complete_prefix(prefix)?;
        }

        // Update global checkpoint
        if let Some(ref manager) = self.checkpoint_manager {
            let mut mgr = manager.lock();
            mgr.checkpoint_mut().complete_prefix(shard_key, prefix);
            mgr.maybe_save()?;
        }

        Ok(())
    }

    /// Perform a coordinated checkpoint of all shards and the global state.
    ///
    /// This checkpoints each shard individually, then updates and saves the
    /// global checkpoint atomically.
    pub fn coordinated_checkpoint(&self) -> CoordinatorResult<()> {
        // First, checkpoint all open shards
        let mut errors = Vec::new();

        for entry in self.shards.iter() {
            let key = entry.key().clone();
            let shard = entry.value();

            // Blocking `shard.read()`: coexists with lock-free writers but waits for
            // any exclusive writer, so no shard is silently skipped (which would
            // double-count on WAL replay at resume).
            let guard = shard.read();

            // Update shard checkpoint
            if let Err(e) = guard.checkpoint() {
                errors.push(format!("Shard {}: {}", key, e));
                continue;
            }

            // Update global checkpoint with shard state
            if let Some(ref manager) = self.checkpoint_manager {
                let mut mgr = manager.lock();
                let ckpt = mgr.checkpoint_mut();
                let state = guard.checkpoint_state();

                // Create or update shard record
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

        // Save global checkpoint
        if let Some(ref manager) = self.checkpoint_manager {
            manager.lock().save()?;
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(CoordinatorError::Checkpoint(errors.join("; ")))
        }
    }

    /// Perform automatic checkpoint if enough time has passed.
    ///
    /// Returns `true` if a checkpoint was performed.
    pub fn maybe_checkpoint(&self) -> CoordinatorResult<bool> {
        if let Some(ref manager) = self.checkpoint_manager {
            let mut mgr = manager.lock();
            if mgr.maybe_save()? {
                drop(mgr);
                // Do a full coordinated checkpoint
                self.checkpoint_all()?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Set metadata in the global checkpoint.
    pub fn set_checkpoint_metadata(
        &self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> CoordinatorResult<()> {
        if let Some(ref manager) = self.checkpoint_manager {
            let mut mgr = manager.lock();
            mgr.checkpoint_mut().set_metadata(key, value);
            mgr.maybe_save()?;
        }
        Ok(())
    }

    /// Get metadata from the global checkpoint.
    pub fn get_checkpoint_metadata(&self, key: &str) -> Option<String> {
        self.checkpoint_manager
            .as_ref()
            .and_then(|mgr| mgr.lock().checkpoint().get_metadata(key).cloned())
    }

    /// Get a summary of the checkpoint state for logging.
    pub fn checkpoint_summary(&self) -> Option<super::super::checkpoint::CheckpointSummary> {
        self.checkpoint_manager
            .as_ref()
            .map(|mgr| mgr.lock().checkpoint().summary())
    }
}
