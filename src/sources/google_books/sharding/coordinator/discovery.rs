//! Shard discovery, shutdown, and summary operations.

use std::collections::HashMap;
use std::sync::atomic::Ordering;

use super::{CoordinatorResult, ShardCoordinator, ShardKey, ShardSummary};

impl ShardCoordinator {
    /// Close all shards (checkpoint and remove from memory).
    pub fn close_all(&self) -> CoordinatorResult<()> {
        self.shutdown
            .store(true, std::sync::atomic::Ordering::SeqCst);

        // Checkpoint all shards
        self.checkpoint_all()?;

        // Clear shards (Drop will handle cleanup)
        self.shards.clear();

        if let Some(ref lru) = self.lru_tracker {
            lru.lock().clear();
        }

        self.stats.active_shards.store(0, Ordering::Relaxed);

        Ok(())
    }

    /// Get total entry count across all open shards.
    pub fn total_entry_count(&self) -> u64 {
        self.shards
            .iter()
            .map(|entry| entry.value().read().len() as u64)
            .sum()
    }

    /// Iterate over all shard keys that have been opened.
    pub fn open_shard_keys(&self) -> Vec<ShardKey> {
        self.shards.iter().map(|e| e.key().clone()).collect()
    }

    /// Get all shard file paths in the shard directory.
    pub fn discover_shard_files(&self) -> CoordinatorResult<Vec<(ShardKey, std::path::PathBuf)>> {
        let mut shards = Vec::new();

        for entry in std::fs::read_dir(&self.config.shard_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|e| e.to_str()) == Some("artrie") {
                // Extract shard key from filename: shard_XX.artrie
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Some(prefix) = stem.strip_prefix("shard_") {
                        let key = ShardKey::new(prefix);
                        shards.push((key, path));
                    }
                }
            }
        }

        Ok(shards)
    }

    /// Mark a prefix as completed in its corresponding shard.
    pub fn complete_prefix(&self, file_prefix: &str, order: u8) -> CoordinatorResult<()> {
        let key = super::super::routing::shard_key_for_file_prefix(
            file_prefix,
            order,
            &self.config.granularity,
        );

        if let Some(shard) = self.shards.get(&key) {
            let mut guard = shard.write();
            guard.complete_prefix(file_prefix)?;
        }

        Ok(())
    }

    /// Get a summary of all shards for checkpoint purposes.
    pub fn shard_summaries(&self) -> HashMap<ShardKey, ShardSummary> {
        self.shards
            .iter()
            .map(|entry| {
                let key = entry.key().clone();
                let guard = entry.value().read();
                let summary = ShardSummary {
                    path: guard.path().to_path_buf(),
                    entry_count: guard.len() as u64,
                    completed_prefixes: guard
                        .checkpoint_state()
                        .completed_prefixes
                        .iter()
                        .cloned()
                        .collect(),
                    ngrams_processed: guard.checkpoint_state().ngrams_processed,
                };
                (key, summary)
            })
            .collect()
    }
}
