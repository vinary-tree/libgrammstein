//! Unified query interface over sharded n-gram storage.
//!
//! This module provides `ShardedTrieView` which enables transparent read access
//! across all shards without needing to know the underlying shard distribution.
//!
//! # Features
//!
//! - **Direct lookup**: Query any n-gram regardless of which shard contains it
//! - **Prefix iteration**: Iterate all n-grams with a given prefix across shards
//! - **Statistics**: Aggregate statistics across all shards
//! - **Lazy loading**: Only opens shards when needed for queries

use super::coordinator::ShardCoordinator;
use super::routing::{compute_shard_key, ngram_order, ShardKey};
use std::collections::BTreeMap;

/// Read-only view over sharded n-gram storage.
///
/// Provides transparent query access across all shards.
pub struct ShardedTrieView<'a> {
    /// Reference to the coordinator.
    coordinator: &'a ShardCoordinator,
}

impl<'a> ShardedTrieView<'a> {
    /// Create a new view over the coordinator.
    pub fn new(coordinator: &'a ShardCoordinator) -> Self {
        Self { coordinator }
    }

    /// Get the count for an n-gram.
    ///
    /// Automatically routes to the correct shard.
    pub fn get(&self, ngram: &str) -> Option<u64> {
        self.coordinator.get(ngram)
    }

    /// Check if an n-gram exists.
    pub fn contains(&self, ngram: &str) -> bool {
        self.coordinator.contains(ngram)
    }

    /// Get the total entry count across all shards.
    pub fn len(&self) -> u64 {
        self.coordinator.total_entry_count()
    }

    /// Check if all shards are empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get all n-grams with a specific prefix.
    ///
    /// This operation may need to query multiple shards depending on the prefix
    /// and the sharding granularity.
    ///
    /// # Arguments
    ///
    /// * `prefix` - The prefix to search for (e.g., "the|" for all bigrams starting with "the")
    ///
    /// # Returns
    ///
    /// A vector of (n-gram, count) pairs sorted by n-gram.
    pub fn prefix_search(&self, prefix: &str) -> Vec<(String, u64)> {
        // Determine which shard(s) to query based on prefix
        let config = self.coordinator.config();
        let granularity = &config.granularity;

        // If prefix is long enough, we can route to a specific shard
        let first_word = prefix.split('|').next().unwrap_or("");
        let prefix_chars: String = first_word
            .chars()
            .filter(|c| c.is_alphabetic())
            .flat_map(|c| c.to_lowercase())
            .collect();

        // Estimate order from prefix
        let order = ngram_order(prefix).max(1);
        let required_len = granularity.prefix_len_for_order(order);

        if prefix_chars.len() >= required_len {
            // We can route to a specific shard
            let shard_key = compute_shard_key(prefix, order, granularity);
            self.prefix_search_shard(&shard_key, prefix)
        } else {
            // Need to search multiple shards
            self.prefix_search_all_shards(prefix)
        }
    }

    /// Search for prefix in a specific shard.
    fn prefix_search_shard(&self, key: &ShardKey, prefix: &str) -> Vec<(String, u64)> {
        if let Ok(shard) = self.coordinator.get_or_create_shard(key) {
            let guard = shard.read();
            match guard.iter_with_counts() {
                Ok(iter) => iter.filter(|(ngram, _)| ngram.starts_with(prefix)).collect(),
                Err(e) => {
                    log::warn!("Failed to iterate shard {}: {}", key, e);
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        }
    }

    /// Search all open shards for prefix.
    fn prefix_search_all_shards(&self, prefix: &str) -> Vec<(String, u64)> {
        let mut results: BTreeMap<String, u64> = BTreeMap::new();

        for key in self.coordinator.open_shard_keys() {
            if let Ok(shard) = self.coordinator.get_or_create_shard(&key) {
                let guard = shard.read();
                match guard.iter_with_counts() {
                    Ok(iter) => {
                        for (ngram, count) in iter {
                            if ngram.starts_with(prefix) {
                                results.insert(ngram, count);
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!("Failed to iterate shard {}: {}", key, e);
                    }
                }
            }
        }

        results.into_iter().collect()
    }

    /// Get aggregate statistics across all shards.
    pub fn stats(&self) -> ViewStats {
        let shard_count = self.coordinator.open_shard_count();
        let coordinator_stats = self.coordinator.stats();

        ViewStats {
            shard_count,
            total_ngrams: coordinator_stats.total_ngrams.load(std::sync::atomic::Ordering::Relaxed),
            unique_ngrams: coordinator_stats.unique_ngrams.load(std::sync::atomic::Ordering::Relaxed),
            total_entries: self.len(),
        }
    }

    /// Get the shard key for an n-gram (for debugging/analysis).
    pub fn shard_key_for(&self, ngram: &str) -> ShardKey {
        self.coordinator.route_ngram(ngram)
    }

    /// Get the distribution of entries across shards.
    ///
    /// Returns a map from shard key to entry count.
    pub fn shard_distribution(&self) -> BTreeMap<String, u64> {
        let mut distribution = BTreeMap::new();

        for key in self.coordinator.open_shard_keys() {
            if let Ok(shard) = self.coordinator.get_or_create_shard(&key) {
                let guard = shard.read();
                distribution.insert(key.to_string(), guard.len() as u64);
            }
        }

        distribution
    }

    /// Iterate over all n-grams across all shards.
    ///
    /// **Warning**: This may be very slow for large datasets as it needs to
    /// iterate through all shards. Use with caution.
    ///
    /// # Returns
    ///
    /// An iterator over (n-gram, count) pairs. Order is not guaranteed.
    /// Shards that fail to iterate are logged and skipped.
    pub fn iter_all(&self) -> impl Iterator<Item = (String, u64)> + '_ {
        let keys = self.coordinator.open_shard_keys();

        keys.into_iter().flat_map(move |key| {
            if let Ok(shard) = self.coordinator.get_or_create_shard(&key) {
                let guard = shard.read();
                match guard.iter_with_counts() {
                    Ok(iter) => iter.collect::<Vec<_>>(),
                    Err(e) => {
                        log::warn!("Failed to iterate shard {}: {}", key, e);
                        Vec::new()
                    }
                }
            } else {
                Vec::new()
            }
        })
    }

    /// Get the top N most frequent n-grams across all shards.
    ///
    /// **Warning**: This requires iterating through all shards and may be slow.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of top n-grams to return
    ///
    /// # Returns
    ///
    /// A vector of (n-gram, count) pairs sorted by count (descending).
    pub fn top_n(&self, n: usize) -> Vec<(String, u64)> {
        use std::collections::BinaryHeap;
        use std::cmp::Reverse;

        // Use a min-heap to efficiently track top N
        let mut heap: BinaryHeap<Reverse<(u64, String)>> = BinaryHeap::new();

        for (ngram, count) in self.iter_all() {
            if heap.len() < n {
                heap.push(Reverse((count, ngram)));
            } else if let Some(Reverse((min_count, _))) = heap.peek() {
                if count > *min_count {
                    heap.pop();
                    heap.push(Reverse((count, ngram)));
                }
            }
        }

        // Convert to sorted vector (descending by count)
        let mut result: Vec<_> = heap
            .into_iter()
            .map(|Reverse((count, ngram))| (ngram, count))
            .collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result
    }
}

/// Statistics from the view.
#[derive(Clone, Debug)]
pub struct ViewStats {
    /// Number of open shards.
    pub shard_count: usize,

    /// Total n-grams processed (including duplicates).
    pub total_ngrams: u64,

    /// Unique n-grams stored.
    pub unique_ngrams: u64,

    /// Total entries across all shards.
    pub total_entries: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::config::{ShardConfig, ShardGranularity};
    use tempfile::TempDir;

    fn create_test_coordinator() -> (TempDir, ShardCoordinator) {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let config = ShardConfig::new(dir.path().join("shards"))
            .with_granularity(ShardGranularity::TwoChar);

        let coordinator = ShardCoordinator::new(config).expect("Failed to create coordinator");

        // Add test data
        coordinator.store_ngram("the|quick", 100).expect("store");
        coordinator.store_ngram("the|slow", 50).expect("store");
        coordinator.store_ngram("apple|pie", 30).expect("store");
        coordinator.store_ngram("apple|cider", 20).expect("store");
        coordinator.store_ngram("zebra|crossing", 10).expect("store");

        (dir, coordinator)
    }

    #[test]
    fn test_view_basic_queries() {
        let (_dir, coordinator) = create_test_coordinator();
        let view = ShardedTrieView::new(&coordinator);

        assert_eq!(view.get("the|quick"), Some(100));
        assert_eq!(view.get("apple|pie"), Some(30));
        assert_eq!(view.get("nonexistent"), None);

        assert!(view.contains("the|quick"));
        assert!(!view.contains("nonexistent"));
    }

    #[test]
    fn test_view_prefix_search() {
        let (_dir, coordinator) = create_test_coordinator();
        let view = ShardedTrieView::new(&coordinator);

        let results = view.prefix_search("the|");
        assert_eq!(results.len(), 2);

        // Should contain both "the|quick" and "the|slow"
        let ngrams: Vec<_> = results.iter().map(|(n, _)| n.as_str()).collect();
        assert!(ngrams.contains(&"the|quick"));
        assert!(ngrams.contains(&"the|slow"));

        let results = view.prefix_search("apple|");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_view_stats() {
        let (_dir, coordinator) = create_test_coordinator();
        let view = ShardedTrieView::new(&coordinator);

        let stats = view.stats();
        assert_eq!(stats.total_entries, 5);
        assert!(stats.shard_count > 0);
    }

    #[test]
    fn test_view_distribution() {
        let (_dir, coordinator) = create_test_coordinator();
        let view = ShardedTrieView::new(&coordinator);

        let dist = view.shard_distribution();

        // Should have multiple shards (th, ap, ze)
        assert!(dist.len() >= 2);

        // "th" shard should have 2 entries
        assert_eq!(dist.get("th"), Some(&2));

        // "ap" shard should have 2 entries
        assert_eq!(dist.get("ap"), Some(&2));
    }

    #[test]
    fn test_view_top_n() {
        let (_dir, coordinator) = create_test_coordinator();
        let view = ShardedTrieView::new(&coordinator);

        let top = view.top_n(3);

        assert_eq!(top.len(), 3);
        // Highest should be "the|quick" with 100
        assert_eq!(top[0], ("the|quick".to_string(), 100));
        // Second should be "the|slow" with 50
        assert_eq!(top[1], ("the|slow".to_string(), 50));
    }

    #[test]
    fn test_view_shard_key() {
        let (_dir, coordinator) = create_test_coordinator();
        let view = ShardedTrieView::new(&coordinator);

        let key = view.shard_key_for("the|quick");
        assert_eq!(key.prefix, "th");

        let key = view.shard_key_for("apple|pie");
        assert_eq!(key.prefix, "ap");
    }
}
