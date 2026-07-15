//! Unified **byte-native** query interface over sharded n-gram storage.
//!
//! This module provides `ShardedTrieView`, a cross-shard read utility over the raw
//! term-id byte keys the shards actually store (concatenated LEB128 varints; see
//! `crate::ngram::vocabulary`). It operates entirely on `&[u8]` keys — there is no
//! delimited-string surface, so it is collision-free (a token containing `'|'` is
//! part of one term-id, never a boundary).
//!
//! # Features
//!
//! - **Prefix iteration**: iterate all n-grams whose term-id byte key starts with a
//!   given varint byte prefix, across shards ([`ShardedTrieView::prefix_search`]).
//! - **Statistics**: aggregate statistics across all shards ([`ShardedTrieView::stats`]).
//! - **Distribution / top-N / full iteration**: byte-key aggregates over open shards.
//! - **Lazy loading**: only opens shards when needed for queries.
//!
//! Point lookups by a known key route through the coordinator's byte API
//! (`ShardCoordinator::get_in_shard` with `compute_shard_key_from_token`) where the
//! first token's characters are in scope; the view deliberately exposes only the
//! cross-shard aggregates that do not require a vocabulary.

use super::coordinator::ShardCoordinator;
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

    /// Get the total entry count across all shards.
    pub fn len(&self) -> u64 {
        self.coordinator.total_entry_count()
    }

    /// Check if all shards are empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get all n-grams whose term-id byte key starts with `prefix`.
    ///
    /// `prefix` is a **term-id varint byte prefix** — a leading slice of a
    /// concatenated-LEB128 n-gram key (e.g. the encoding of the first one or two
    /// term-ids), NOT delimited text. Because the first token's characters (needed
    /// to route to a single shard) are not recoverable from a bare varint prefix
    /// without the vocabulary, this scans every open shard and keeps the entries
    /// whose key `starts_with(prefix)`. This is collision-free by construction: a
    /// varint boundary is unambiguous, unlike a `'|'` join.
    ///
    /// # Arguments
    ///
    /// * `prefix` - Term-id varint byte prefix (e.g. `encode_varint(id("the"))` for
    ///   all n-grams beginning with the term "the").
    ///
    /// # Returns
    ///
    /// A vector of (term-id byte key, count) pairs sorted by key.
    pub fn prefix_search(&self, prefix: &[u8]) -> Vec<(Vec<u8>, u64)> {
        self.prefix_search_all_shards(prefix)
    }

    /// Search all open shards for a term-id byte prefix.
    fn prefix_search_all_shards(&self, prefix: &[u8]) -> Vec<(Vec<u8>, u64)> {
        let mut results: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

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
            total_ngrams: coordinator_stats
                .total_ngrams
                .load(std::sync::atomic::Ordering::Relaxed),
            unique_ngrams: coordinator_stats
                .unique_ngrams
                .load(std::sync::atomic::Ordering::Relaxed),
            total_entries: self.len(),
        }
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
    pub fn iter_all(&self) -> impl Iterator<Item = (Vec<u8>, u64)> + '_ {
        let keys = self.coordinator.open_shard_keys();

        keys.into_iter().flat_map(move |key| {
            if let Ok(shard) = self.coordinator.get_or_create_shard(&key) {
                let guard = shard.read();
                match guard.iter_with_counts() {
                    Ok(entries) => entries,
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
    pub fn top_n(&self, n: usize) -> Vec<(Vec<u8>, u64)> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        // Use a min-heap to efficiently track top N
        let mut heap: BinaryHeap<Reverse<(u64, Vec<u8>)>> = BinaryHeap::new();

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
        result.sort_by_key(|r| std::cmp::Reverse(r.1));
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
    use super::super::config::{ShardConfig, ShardGranularity};
    use super::super::routing::compute_shard_key_from_token;
    use super::*;
    use crate::ngram::vocabulary::{create_vocabulary, encode_varint, SharedVocabARTrie};
    use tempfile::TempDir;

    /// Store one n-gram exactly as the importer does: route by the first token's
    /// characters + sequence length, key by concatenated LEB128 term-id varints.
    fn store_ngram(
        coordinator: &ShardCoordinator,
        vocab: &SharedVocabARTrie,
        tokens: &[&str],
        count: u64,
    ) {
        let ids: Vec<u64> = tokens
            .iter()
            .map(|t| vocab.as_ref().insert(t).expect("vocab insert"))
            .collect();
        let shard_key = compute_shard_key_from_token(
            tokens[0],
            tokens.len() as u8,
            &coordinator.config().granularity,
        );
        let mut key = Vec::with_capacity(ids.len() * 2);
        for id in &ids {
            encode_varint(*id, &mut key);
        }
        coordinator
            .store_in_shard(&shard_key, &key, count)
            .expect("store_in_shard");
    }

    /// Recompute the term-id byte key for an already-interned token sequence.
    fn ngram_key(vocab: &SharedVocabARTrie, tokens: &[&str]) -> Vec<u8> {
        let mut key = Vec::with_capacity(tokens.len() * 2);
        for t in tokens {
            let id = vocab.as_ref().get_index(t).expect("token already in vocab");
            encode_varint(id, &mut key);
        }
        key
    }

    /// Look up a token sequence via the coordinator's byte API (first-token route
    /// + term-id key), the read path the view's aggregates complement.
    fn get_ngram(
        coordinator: &ShardCoordinator,
        vocab: &SharedVocabARTrie,
        tokens: &[&str],
    ) -> Option<u64> {
        let shard_key = compute_shard_key_from_token(
            tokens[0],
            tokens.len() as u8,
            &coordinator.config().granularity,
        );
        coordinator.get_in_shard(&shard_key, &ngram_key(vocab, tokens))
    }

    fn create_test_coordinator() -> (TempDir, ShardCoordinator, SharedVocabARTrie) {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let config =
            ShardConfig::new(dir.path().join("shards")).with_granularity(ShardGranularity::TwoChar);

        let coordinator = ShardCoordinator::new(config).expect("Failed to create coordinator");
        let vocab = create_vocabulary(&dir.path().join("vocab")).expect("vocabulary");

        // Term-id-encoded bigrams (routing prefixes: th, th, ap, ap, ze).
        store_ngram(&coordinator, &vocab, &["the", "quick"], 100);
        store_ngram(&coordinator, &vocab, &["the", "slow"], 50);
        store_ngram(&coordinator, &vocab, &["apple", "pie"], 30);
        store_ngram(&coordinator, &vocab, &["apple", "cider"], 20);
        store_ngram(&coordinator, &vocab, &["zebra", "crossing"], 10);

        (dir, coordinator, vocab)
    }

    #[test]
    fn test_view_basic_queries() {
        let (_dir, coordinator, vocab) = create_test_coordinator();
        // Point lookups go through the coordinator's byte API; the view provides
        // the cross-shard aggregates.
        assert_eq!(
            get_ngram(&coordinator, &vocab, &["the", "quick"]),
            Some(100)
        );
        assert_eq!(get_ngram(&coordinator, &vocab, &["apple", "pie"]), Some(30));
        assert_eq!(
            get_ngram(&coordinator, &vocab, &["zebra", "crossing"]),
            Some(10)
        );

        let view = ShardedTrieView::new(&coordinator);
        assert!(!view.is_empty());
        assert_eq!(view.len(), 5);
    }

    #[test]
    fn test_view_prefix_search() {
        let (_dir, coordinator, vocab) = create_test_coordinator();
        let view = ShardedTrieView::new(&coordinator);

        // A term-id varint byte prefix: the leading varint of "the" matches both
        // ["the","quick"] and ["the","slow"] (collision-free — a varint boundary).
        let the_prefix = ngram_key(&vocab, &["the"]);
        let results = view.prefix_search(&the_prefix);
        assert_eq!(results.len(), 2);

        let keys: Vec<_> = results.iter().map(|(k, _)| k.as_slice()).collect();
        assert!(keys.contains(&ngram_key(&vocab, &["the", "quick"]).as_slice()));
        assert!(keys.contains(&ngram_key(&vocab, &["the", "slow"]).as_slice()));

        let apple_prefix = ngram_key(&vocab, &["apple"]);
        let results = view.prefix_search(&apple_prefix);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_view_stats() {
        let (_dir, coordinator, _vocab) = create_test_coordinator();
        let view = ShardedTrieView::new(&coordinator);

        let stats = view.stats();
        assert_eq!(stats.total_entries, 5);
        assert!(stats.shard_count > 0);
    }

    #[test]
    fn test_view_distribution() {
        let (_dir, coordinator, _vocab) = create_test_coordinator();
        let view = ShardedTrieView::new(&coordinator);

        let dist = view.shard_distribution();

        // Should have multiple shards (th, ap, ze).
        assert!(dist.len() >= 2);

        // "th" shard should have 2 entries (the/quick, the/slow).
        assert_eq!(dist.get("th"), Some(&2));

        // "ap" shard should have 2 entries (apple/pie, apple/cider).
        assert_eq!(dist.get("ap"), Some(&2));
    }

    #[test]
    fn test_view_top_n() {
        let (_dir, coordinator, vocab) = create_test_coordinator();
        let view = ShardedTrieView::new(&coordinator);

        let top = view.top_n(3);

        assert_eq!(top.len(), 3);
        // Highest is ["the","quick"] with 100.
        assert_eq!(top[0], (ngram_key(&vocab, &["the", "quick"]), 100));
        // Second is ["the","slow"] with 50.
        assert_eq!(top[1], (ngram_key(&vocab, &["the", "slow"]), 50));
    }
}
