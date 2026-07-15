//! N-gram → shard routing and direct store/get operations (byte-native).
//!
//! The methods here fan **term-id byte keys** (concatenated LEB128 varints; see
//! `crate::ngram::vocabulary`) out to the right shard based on the configured
//! [`ShardGranularity`] (prefix-based or hash-based) and delegate the actual
//! storage to the per-shard `ShardHandle`. Routing consults only the **first
//! token's characters** and the **sequence length**
//! ([`compute_shard_key_from_token`]) — never a delimited string and never a
//! term-id *value*. The coordinator is vocabulary-free: callers that hold the
//! vocabulary (`NgramStorage::store_*` / `get_*`, `ShardedView`, the importer's
//! per-record fallback) encode tokens to term-ids and pass the byte key here.

use std::sync::atomic::Ordering;

use super::super::routing::compute_shard_key_from_token;
use super::{CoordinatorResult, ShardCoordinator, ShardKey};

impl ShardCoordinator {
    /// Compute the shard key from tokens (for vocabulary-indexed encoding).
    ///
    /// Routes on the first token's characters + sequence length, so the n-gram's
    /// stored key can be a concatenated LEB128 varint **term-id** byte sequence
    /// (not a delimited string). The routing decision is independent of the
    /// term-id *values*, so it is stable across vocabularies.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The n-gram tokens (e.g., ["the", "quick", "brown"])
    ///
    /// # Returns
    ///
    /// A `ShardKey` identifying which shard should store this n-gram.
    pub fn route_tokens(&self, tokens: &[&str]) -> ShardKey {
        let first_token = tokens.first().copied().unwrap_or("");
        let order = tokens.len() as u8;
        compute_shard_key_from_token(first_token, order, &self.config.granularity)
    }

    /// Store an encoded n-gram key in a specific shard.
    ///
    /// Use this when you have already computed the shard key and the encoded
    /// n-gram key (e.g., for vocabulary-indexed encoding).
    ///
    /// # Arguments
    ///
    /// * `shard_key` - The shard key (from `route_tokens`)
    /// * `encoded_key` - The encoded n-gram key to store
    /// * `count` - The count to add
    ///
    /// # Returns
    ///
    /// `true` if this was a new n-gram, `false` if it already existed.
    pub fn store_in_shard(
        &self,
        shard_key: &ShardKey,
        encoded_key: &[u8],
        count: u64,
    ) -> CoordinatorResult<bool> {
        let shard = self.get_or_create_shard(shard_key)?;

        // Lock-free path: shared read guard + CAS increment
        let guard = shard.read();
        let was_new = guard.increment_lockfree(encoded_key, count)?;

        if was_new {
            self.stats.unique_ngrams.fetch_add(1, Ordering::Relaxed);
        }
        self.stats.total_ngrams.fetch_add(count, Ordering::Relaxed);

        Ok(was_new)
    }

    /// Store multiple n-grams to the same shard efficiently.
    ///
    /// All n-grams must route to the same shard. This is more efficient
    /// than calling `store_ngram` repeatedly because it only acquires
    /// the write lock once.
    ///
    /// # Arguments
    ///
    /// * `key` - The shard key (all n-grams must route to this shard)
    /// * `ngrams` - Iterator of (ngram, count) pairs
    ///
    /// # Returns
    ///
    /// Number of new (unique) n-grams stored.
    pub fn store_ngrams_batch<'a, I>(&self, key: &ShardKey, ngrams: I) -> CoordinatorResult<u64>
    where
        I: Iterator<Item = (&'a [u8], u64)>,
    {
        let shard = self.get_or_create_shard(key)?;

        // Lock-free path: shared read guard + CAS increments
        let guard = shard.read();

        let mut new_count = 0u64;
        let mut total_count = 0u64;

        for (ngram, count) in ngrams {
            if guard.increment_lockfree(ngram, count)? {
                new_count += 1;
            }
            total_count += count;
        }

        self.stats.record_ngrams(total_count, new_count);

        Ok(new_count)
    }

    /// Get the count for an encoded key in a specific shard.
    ///
    /// Use this when you have already computed the shard key and the encoded
    /// n-gram key (e.g., for vocabulary-indexed encoding).
    ///
    /// # Arguments
    ///
    /// * `shard_key` - The shard key (from `route_tokens`)
    /// * `encoded_key` - The encoded n-gram key to look up
    pub fn get_in_shard(&self, shard_key: &ShardKey, encoded_key: &[u8]) -> Option<u64> {
        if let Some(shard) = self.shards.get(shard_key) {
            let guard = shard.read();
            return guard.get(encoded_key);
        }

        // Shard not loaded - check if file exists
        let path = self.config.shard_path(&shard_key.as_file_stem());
        if path.exists() {
            // Load shard and query
            if let Ok(shard) = self.get_or_create_shard(shard_key) {
                let guard = shard.read();
                return guard.get(encoded_key);
            }
        }

        None
    }
}
