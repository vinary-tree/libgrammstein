//! Integration tests for the Google Books sharding module.
//!
//! These tests verify the end-to-end functionality of the sharded n-gram
//! storage system, including:
//! - Shard creation and routing
//! - N-gram storage and retrieval
//! - Checkpoint and recovery
//! - MKN statistics computation
//! - Merge to output trie
//!
//! # Term-id / first-token routing
//!
//! N-grams are stored exactly as the importer stores them: routed by the FIRST
//! TOKEN's characters + sequence length ([`compute_shard_key_from_token`]) and
//! keyed by a concatenated LEB128 varint **term-id** byte sequence
//! (`vocabulary::encode_varint`). There is no delimited-string representation
//! anywhere — a token containing `'|'` is one term-id, never a boundary. Point
//! lookups go through `ShardCoordinator::get_in_shard`; cross-shard aggregates go
//! through the byte-native [`ShardedTrieView`].

#![cfg(feature = "google-books")]

use std::sync::Arc;
use std::thread;

use libdictenstein::persistent_artrie::PersistentARTrie;
use libgrammstein::ngram::vocabulary::{
    create_vocabulary, encode_indices_to_key_bytes, encode_varint, SharedVocabARTrie,
};
use libgrammstein::sources::google_books::sharding::{
    compute_shard_key_from_token, MergeCoordinator, MknAggregator, ShardConfig, ShardCoordinator,
    ShardGranularity, ShardKey, ShardedTrieView,
};
use tempfile::TempDir;

/// Concatenated LEB128 term-id byte key for a token sequence. `insert` is
/// idempotent, so recomputing a key for an already-stored n-gram is stable.
fn ngram_key(vocab: &SharedVocabARTrie, tokens: &[&str]) -> Vec<u8> {
    let mut key = Vec::with_capacity(tokens.len() * 2);
    for token in tokens {
        let id = vocab.as_ref().insert(token).expect("vocab insert");
        encode_varint(id, &mut key);
    }
    key
}

/// The shard a token sequence routes to — the first token's characters +
/// sequence length under the coordinator's configured granularity.
fn route(coordinator: &ShardCoordinator, tokens: &[&str]) -> ShardKey {
    compute_shard_key_from_token(
        tokens[0],
        tokens.len() as u8,
        &coordinator.config().granularity,
    )
}

/// Store one n-gram the importer's way (first-token route + term-id key).
fn store_ngram(
    coordinator: &ShardCoordinator,
    vocab: &SharedVocabARTrie,
    tokens: &[&str],
    count: u64,
) {
    coordinator
        .store_in_shard(
            &route(coordinator, tokens),
            &ngram_key(vocab, tokens),
            count,
        )
        .expect("store_in_shard");
}

/// Look up one n-gram the importer's way.
fn get_ngram(
    coordinator: &ShardCoordinator,
    vocab: &SharedVocabARTrie,
    tokens: &[&str],
) -> Option<u64> {
    coordinator.get_in_shard(&route(coordinator, tokens), &ngram_key(vocab, tokens))
}

/// Create a test coordinator with sample data (term-id keys, first-token routing).
fn create_coordinator_with_data() -> (TempDir, ShardCoordinator, SharedVocabARTrie) {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let config =
        ShardConfig::new(dir.path().join("shards")).with_granularity(ShardGranularity::TwoChar);

    let coordinator = ShardCoordinator::new(config).expect("Failed to create coordinator");
    let vocab = create_vocabulary(&dir.path().join("vocab")).expect("vocabulary");

    // Sample n-grams simulating Google Books data.
    // 1-grams
    for (word, count) in &[
        ("the", 1_000_000u64),
        ("a", 500_000),
        ("is", 250_000),
        ("of", 200_000),
        ("and", 180_000),
        ("to", 170_000),
        ("in", 160_000),
        ("that", 150_000),
    ] {
        store_ngram(&coordinator, &vocab, &[word], *count);
    }

    // 2-grams
    for (first, second, count) in &[
        ("the", "quick", 50_000u64),
        ("the", "slow", 25_000),
        ("the", "big", 40_000),
        ("a", "small", 30_000),
        ("a", "large", 28_000),
        ("is", "very", 22_000),
        ("of", "the", 100_000),
        ("in", "the", 90_000),
    ] {
        store_ngram(&coordinator, &vocab, &[first, second], *count);
    }

    // 3-grams
    for (first, second, third, count) in &[
        ("the", "quick", "brown", 10_000u64),
        ("the", "slow", "green", 5_000),
        ("of", "the", "united", 8_000),
        ("in", "the", "world", 7_000),
    ] {
        store_ngram(&coordinator, &vocab, &[first, second, third], *count);
    }

    (dir, coordinator, vocab)
}

// =============================================================================
// Basic Workflow Tests
// =============================================================================

#[test]
fn test_full_workflow() {
    let (dir, coordinator, vocab) = create_coordinator_with_data();

    // 1. Verify storage
    assert!(get_ngram(&coordinator, &vocab, &["the"]).is_some());
    assert!(get_ngram(&coordinator, &vocab, &["the", "quick"]).is_some());
    assert!(get_ngram(&coordinator, &vocab, &["the", "quick", "brown"]).is_some());

    // 2. Verify counts
    assert_eq!(get_ngram(&coordinator, &vocab, &["the"]), Some(1_000_000));
    assert_eq!(
        get_ngram(&coordinator, &vocab, &["the", "quick"]),
        Some(50_000)
    );
    assert_eq!(
        get_ngram(&coordinator, &vocab, &["the", "quick", "brown"]),
        Some(10_000)
    );

    // 3. Cross-shard aggregates via the byte-native ShardedTrieView
    let view = ShardedTrieView::new(&coordinator);
    assert!(!view.is_empty());
    assert_eq!(view.len(), 20);

    // 4. Prefix search by a term-id byte prefix (all n-grams whose first token is
    //    "the": the unigram plus every "the …" bi/trigram — collision-free).
    let the_prefix = ngram_key(&vocab, &["the"]);
    let the_ngrams = view.prefix_search(&the_prefix);
    assert!(the_ngrams.len() >= 3); // the, the|quick, the|slow, the|big, the|quick|brown, the|slow|green

    // 5. Top N — "the" (1,000,000) is the most frequent.
    let top = view.top_n(3);
    assert_eq!(top.len(), 3);
    assert_eq!(top[0].0, ngram_key(&vocab, &["the"]));
    assert_eq!(top[0].1, 1_000_000);

    // 6. Compute MKN statistics
    let aggregator = MknAggregator::new(&coordinator);
    let stats = aggregator
        .compute_all()
        .expect("Failed to compute MKN stats");

    // Should have counts for orders 1-3
    assert!(stats.frequency_counts[1].total_unique > 0);
    assert!(stats.frequency_counts[2].total_unique > 0);
    assert!(stats.frequency_counts[3].total_unique > 0);

    // 7. Merge to trie
    let output_path = dir.path().join("merged.artrie");
    let merger = MergeCoordinator::new(&coordinator);
    let merge_stats = merger
        .merge_to_trie(&output_path, |_| {})
        .expect("Failed to merge");

    assert!(output_path.exists());
    assert!(merge_stats.total_ngrams > 0);

    // 8. Verify merged trie (term-id byte keys)
    let (merged, _) =
        PersistentARTrie::<u64>::open_with_recovery(&output_path).expect("Failed to open");
    assert_eq!(
        merged.get_value_bytes(&ngram_key(&vocab, &["the"])),
        Some(1_000_000)
    );
    assert_eq!(
        merged.get_value_bytes(&ngram_key(&vocab, &["the", "quick"])),
        Some(50_000)
    );
}

// =============================================================================
// Checkpoint and Recovery Tests
// =============================================================================

#[test]
fn test_checkpoint_and_recovery() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let shard_path = dir.path().join("shards");
    // The vocabulary lives across both phases (its in-memory term-ids are stable),
    // so the recomputed keys match the ones written before the reopen.
    let vocab = create_vocabulary(&dir.path().join("vocab")).expect("vocabulary");

    // Phase 1: Create coordinator and store data
    {
        let config = ShardConfig::new(&shard_path).with_granularity(ShardGranularity::TwoChar);

        let coordinator =
            ShardCoordinator::new_with_checkpoints(config).expect("Failed to create coordinator");

        store_ngram(&coordinator, &vocab, &["apple", "pie"], 100);
        store_ngram(&coordinator, &vocab, &["banana", "split"], 50);

        coordinator.checkpoint_all().expect("Failed to checkpoint");
    }

    // Phase 2: Resume and verify data persists (get_in_shard lazily opens shards).
    {
        let config = ShardConfig::new(&shard_path).with_granularity(ShardGranularity::TwoChar);

        let coordinator =
            ShardCoordinator::resume_or_start(config).expect("Failed to resume coordinator");

        assert_eq!(
            get_ngram(&coordinator, &vocab, &["apple", "pie"]),
            Some(100)
        );
        assert_eq!(
            get_ngram(&coordinator, &vocab, &["banana", "split"]),
            Some(50)
        );
    }
}

// =============================================================================
// Shard Routing Tests
// =============================================================================

#[test]
fn test_shard_routing() {
    let (_dir, coordinator, _vocab) = create_coordinator_with_data();

    // Routing consistency — all "the …" n-grams route to the "th" shard because
    // routing is a function of the FIRST TOKEN's characters.
    let key1 = route(&coordinator, &["the", "quick"]);
    let key2 = route(&coordinator, &["the", "slow"]);
    let key3 = route(&coordinator, &["the", "big"]);

    assert_eq!(key1.prefix, "th");
    assert_eq!(key2.prefix, "th");
    assert_eq!(key3.prefix, "th");

    // Different first token → different shard.
    let key4 = route(&coordinator, &["apple", "pie"]);
    assert_eq!(key4.prefix, "ap");
    assert_ne!(key1.prefix, key4.prefix);
}

#[test]
fn test_shard_distribution() {
    let (_dir, coordinator, _vocab) = create_coordinator_with_data();
    let view = ShardedTrieView::new(&coordinator);

    let dist = view.shard_distribution();

    // Should have multiple shards
    assert!(dist.len() >= 2);

    // "th" shard should have entries
    assert!(dist.contains_key("th"));
    assert!(*dist.get("th").unwrap() > 0);
}

// =============================================================================
// MKN Statistics Tests
// =============================================================================

#[test]
fn test_mkn_discount_computation() {
    let (_dir, coordinator, _vocab) = create_coordinator_with_data();
    let aggregator = MknAggregator::new(&coordinator);

    let discounts = aggregator
        .compute_discounts_only()
        .expect("Failed to compute discounts");

    // Should have discount params for orders 1-5
    assert!(discounts.len() >= 4);

    // Discount values should be in reasonable range
    for (order, d) in discounts.iter().enumerate().skip(1) {
        if order <= 3 {
            assert!(
                d.d1 >= 0.0 && d.d1 <= 1.0,
                "D1 out of range for order {}",
                order
            );
            assert!(
                d.d2 >= 0.0 && d.d2 <= 2.0,
                "D2 out of range for order {}",
                order
            );
            assert!(
                d.d3_plus >= 0.0 && d.d3_plus <= 3.0,
                "D3+ out of range for order {}",
                order
            );
        }
    }
}

#[test]
fn test_mkn_continuation_counts() {
    let (_dir, coordinator, vocab) = create_coordinator_with_data();
    let aggregator = MknAggregator::new(&coordinator).with_continuations();

    let stats = aggregator
        .compute_all()
        .expect("Failed to compute MKN stats");

    // For bigrams, continuation contexts are varint-encoded term-id keys.
    let cont = &stats.continuation_counts[2];

    // Context "the" (a single term-id) has successors quick, slow, big.
    let the_id = vocab
        .as_ref()
        .get_index("the")
        .expect("'the' should be in vocab");
    let the_context = encode_indices_to_key_bytes(&[the_id]);
    if let Some(count) = cont.successor_counts.get(&the_context) {
        assert!(*count >= 3, "Expected at least 3 successors for 'the'");
    }
}

// =============================================================================
// Merge Tests
// =============================================================================

#[test]
fn test_merge_to_memory() {
    let (_dir, coordinator, vocab) = create_coordinator_with_data();
    let merger = MergeCoordinator::new(&coordinator);

    let merged = merger.merge_to_memory().expect("Failed to merge");

    // Should have all entries
    assert!(merged.len() >= 20); // 8 + 8 + 4 n-grams

    // Verify some entries (term-id byte keys)
    assert_eq!(merged.get(&ngram_key(&vocab, &["the"])), Some(&1_000_000));
    assert_eq!(
        merged.get(&ngram_key(&vocab, &["the", "quick"])),
        Some(&50_000)
    );
}

#[test]
fn test_merge_preserves_counts() {
    let (dir, coordinator, vocab) = create_coordinator_with_data();
    let merger = MergeCoordinator::new(&coordinator);

    let output_path = dir.path().join("merged_counts.artrie");
    merger
        .merge_to_trie(&output_path, |_| {})
        .expect("Failed to merge");

    let (merged, _) =
        PersistentARTrie::<u64>::open_with_recovery(&output_path).expect("Failed to open");

    // Verify all counts are preserved
    assert_eq!(
        merged.get_value_bytes(&ngram_key(&vocab, &["the"])),
        Some(1_000_000)
    );
    assert_eq!(
        merged.get_value_bytes(&ngram_key(&vocab, &["a"])),
        Some(500_000)
    );
    assert_eq!(
        merged.get_value_bytes(&ngram_key(&vocab, &["the", "quick"])),
        Some(50_000)
    );
    assert_eq!(
        merged.get_value_bytes(&ngram_key(&vocab, &["the", "quick", "brown"])),
        Some(10_000)
    );
}

// =============================================================================
// Parallel Storage Tests
// =============================================================================

#[test]
fn test_parallel_storage() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let config = ShardConfig::new(dir.path().join("shards"))
        .with_granularity(ShardGranularity::TwoChar)
        .with_max_open_shards(100);

    let coordinator =
        Arc::new(ShardCoordinator::new(config).expect("Failed to create coordinator"));
    // SharedVocabARTrie is a lock-free concurrent vocabulary (Arc), safe to share.
    let vocab = create_vocabulary(&dir.path().join("vocab")).expect("vocabulary");

    // Spawn multiple threads to store n-grams concurrently
    let mut handles = Vec::new();

    for thread_id in 0..4 {
        let coord = Arc::clone(&coordinator);
        let vocab = vocab.clone();
        handles.push(thread::spawn(move || {
            for i in 0..100 {
                let prefix = match thread_id {
                    0 => "aa",
                    1 => "bb",
                    2 => "cc",
                    _ => "dd",
                };
                let first = format!("{}word{}", prefix, i);
                store_ngram(&coord, &vocab, &[first.as_str(), "suffix"], 1);
            }
        }));
    }

    // Wait for all threads
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Verify all entries were stored
    assert_eq!(coordinator.total_entry_count(), 400);

    // Verify entries from each prefix
    assert!(get_ngram(&coordinator, &vocab, &["aaword0", "suffix"]).is_some());
    assert!(get_ngram(&coordinator, &vocab, &["bbword0", "suffix"]).is_some());
    assert!(get_ngram(&coordinator, &vocab, &["ccword0", "suffix"]).is_some());
    assert!(get_ngram(&coordinator, &vocab, &["ddword0", "suffix"]).is_some());
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_empty_coordinator() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let config =
        ShardConfig::new(dir.path().join("shards")).with_granularity(ShardGranularity::TwoChar);

    let coordinator = ShardCoordinator::new(config).expect("Failed to create coordinator");
    let vocab = create_vocabulary(&dir.path().join("vocab")).expect("vocabulary");

    // Empty coordinator should return None for queries
    assert_eq!(get_ngram(&coordinator, &vocab, &["nonexistent"]), None);
    assert_eq!(coordinator.total_entry_count(), 0);

    // View should be empty
    let view = ShardedTrieView::new(&coordinator);
    assert!(view.is_empty());
    assert_eq!(view.len(), 0);
}

#[test]
fn test_duplicate_storage() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let config =
        ShardConfig::new(dir.path().join("shards")).with_granularity(ShardGranularity::TwoChar);

    let coordinator = ShardCoordinator::new(config).expect("Failed to create coordinator");
    let vocab = create_vocabulary(&dir.path().join("vocab")).expect("vocabulary");

    // Store same n-gram multiple times
    store_ngram(&coordinator, &vocab, &["test", "word"], 10);
    store_ngram(&coordinator, &vocab, &["test", "word"], 20);
    store_ngram(&coordinator, &vocab, &["test", "word"], 30);

    // Count should be cumulative
    assert_eq!(get_ngram(&coordinator, &vocab, &["test", "word"]), Some(60));
}

#[test]
fn test_special_characters() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let config =
        ShardConfig::new(dir.path().join("shards")).with_granularity(ShardGranularity::TwoChar);

    let coordinator = ShardCoordinator::new(config).expect("Failed to create coordinator");
    let vocab = create_vocabulary(&dir.path().join("vocab")).expect("vocabulary");

    // Store n-grams with special characters (common in Google Books). Each token
    // is carried intact as a term-id — apostrophes and periods never split.
    store_ngram(&coordinator, &vocab, &["don't"], 1000);
    store_ngram(&coordinator, &vocab, &["it's", "good"], 500);
    store_ngram(&coordinator, &vocab, &["U.S.A."], 300);

    // Verify storage
    assert_eq!(get_ngram(&coordinator, &vocab, &["don't"]), Some(1000));
    assert_eq!(
        get_ngram(&coordinator, &vocab, &["it's", "good"]),
        Some(500)
    );
    assert_eq!(get_ngram(&coordinator, &vocab, &["U.S.A."]), Some(300));
}
