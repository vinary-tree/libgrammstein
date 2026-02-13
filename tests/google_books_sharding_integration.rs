//! Integration tests for the Google Books sharding module.
//!
//! These tests verify the end-to-end functionality of the sharded n-gram
//! storage system, including:
//! - Shard creation and routing
//! - N-gram storage and retrieval
//! - Checkpoint and recovery
//! - MKN statistics computation
//! - Merge to output trie

#![cfg(feature = "google-books")]

use libgrammstein::sources::google_books::sharding::{
    MergeCoordinator, MknAggregator, ShardConfig, ShardCoordinator, ShardGranularity,
    ShardedTrieView,
};
use libdictenstein::persistent_artrie::PersistentARTrie;
use tempfile::TempDir;

/// Create a test coordinator with sample data.
fn create_coordinator_with_data() -> (TempDir, ShardCoordinator) {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let config = ShardConfig::new(dir.path().join("shards"))
        .with_granularity(ShardGranularity::TwoChar);

    let coordinator = ShardCoordinator::new(config).expect("Failed to create coordinator");

    // Add sample n-grams simulating Google Books data
    // 1-grams
    for (word, count) in &[
        ("the", 1_000_000),
        ("a", 500_000),
        ("is", 250_000),
        ("of", 200_000),
        ("and", 180_000),
        ("to", 170_000),
        ("in", 160_000),
        ("that", 150_000),
    ] {
        coordinator
            .store_ngram(word, *count)
            .expect("Failed to store 1-gram");
    }

    // 2-grams
    for (ngram, count) in &[
        ("the|quick", 50_000),
        ("the|slow", 25_000),
        ("the|big", 40_000),
        ("a|small", 30_000),
        ("a|large", 28_000),
        ("is|very", 22_000),
        ("of|the", 100_000),
        ("in|the", 90_000),
    ] {
        coordinator
            .store_ngram(ngram, *count)
            .expect("Failed to store 2-gram");
    }

    // 3-grams
    for (ngram, count) in &[
        ("the|quick|brown", 10_000),
        ("the|slow|green", 5_000),
        ("of|the|united", 8_000),
        ("in|the|world", 7_000),
    ] {
        coordinator
            .store_ngram(ngram, *count)
            .expect("Failed to store 3-gram");
    }

    (dir, coordinator)
}

// =============================================================================
// Basic Workflow Tests
// =============================================================================

#[test]
fn test_full_workflow() {
    let (dir, coordinator) = create_coordinator_with_data();

    // 1. Verify storage
    assert!(coordinator.contains("the"));
    assert!(coordinator.contains("the|quick"));
    assert!(coordinator.contains("the|quick|brown"));

    // 2. Verify counts
    assert_eq!(coordinator.get("the"), Some(1_000_000));
    assert_eq!(coordinator.get("the|quick"), Some(50_000));
    assert_eq!(coordinator.get("the|quick|brown"), Some(10_000));

    // 3. Query via ShardedTrieView
    let view = ShardedTrieView::new(&coordinator);
    assert_eq!(view.get("the"), Some(1_000_000));
    assert!(view.len() > 0);

    // 4. Prefix search
    let the_bigrams = view.prefix_search(b"the|");
    assert!(the_bigrams.len() >= 3); // "the|quick", "the|slow", "the|big"

    // 5. Top N
    let top = view.top_n(3);
    assert_eq!(top.len(), 3);
    assert_eq!(top[0].0, b"the".to_vec()); // Most frequent

    // 6. Compute MKN statistics
    let aggregator = MknAggregator::new(&coordinator);
    let stats = aggregator.compute_all().expect("Failed to compute MKN stats");

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

    // 8. Verify merged trie
    let (merged, _) =
        PersistentARTrie::<u64>::open_with_recovery(&output_path).expect("Failed to open");
    assert_eq!(merged.get_value_bytes(b"the"), Some(1_000_000));
    assert_eq!(merged.get_value_bytes(b"the|quick"), Some(50_000));
}

// =============================================================================
// Checkpoint and Recovery Tests
// =============================================================================

#[test]
fn test_checkpoint_and_recovery() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let shard_path = dir.path().join("shards");

    // Phase 1: Create coordinator and store data
    {
        let config = ShardConfig::new(&shard_path).with_granularity(ShardGranularity::TwoChar);

        let coordinator =
            ShardCoordinator::new_with_checkpoints(config).expect("Failed to create coordinator");

        coordinator.store_ngram("apple|pie", 100).expect("store");
        coordinator.store_ngram("banana|split", 50).expect("store");

        coordinator
            .checkpoint_all()
            .expect("Failed to checkpoint");
    }

    // Phase 2: Resume and verify data persists
    {
        let config = ShardConfig::new(&shard_path).with_granularity(ShardGranularity::TwoChar);

        let coordinator =
            ShardCoordinator::resume_or_start(config).expect("Failed to resume coordinator");

        assert_eq!(coordinator.get("apple|pie"), Some(100));
        assert_eq!(coordinator.get("banana|split"), Some(50));
    }
}

// =============================================================================
// Shard Routing Tests
// =============================================================================

#[test]
fn test_shard_routing() {
    let (_dir, coordinator) = create_coordinator_with_data();

    // Test routing consistency
    let key1 = coordinator.route_ngram("the|quick");
    let key2 = coordinator.route_ngram("the|slow");
    let key3 = coordinator.route_ngram("the|big");

    // All "the" prefixed n-grams should route to same shard
    assert_eq!(key1.prefix, "th");
    assert_eq!(key2.prefix, "th");
    assert_eq!(key3.prefix, "th");

    // Different prefix should route differently
    let key4 = coordinator.route_ngram("apple|pie");
    assert_eq!(key4.prefix, "ap");
    assert_ne!(key1.prefix, key4.prefix);
}

#[test]
fn test_shard_distribution() {
    let (_dir, coordinator) = create_coordinator_with_data();
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
    let (_dir, coordinator) = create_coordinator_with_data();
    let aggregator = MknAggregator::new(&coordinator);

    let discounts = aggregator
        .compute_discounts_only()
        .expect("Failed to compute discounts");

    // Should have discount params for orders 1-5
    assert!(discounts.len() >= 4);

    // Discount values should be in reasonable range
    for (order, d) in discounts.iter().enumerate().skip(1) {
        if order <= 3 {
            assert!(d.d1 >= 0.0 && d.d1 <= 1.0, "D1 out of range for order {}", order);
            assert!(d.d2 >= 0.0 && d.d2 <= 2.0, "D2 out of range for order {}", order);
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
    let (_dir, coordinator) = create_coordinator_with_data();
    let aggregator = MknAggregator::new(&coordinator).with_continuations();

    let stats = aggregator.compute_all().expect("Failed to compute MKN stats");

    // For bigrams, should have continuation counts
    let cont = &stats.continuation_counts[2];

    // "the" should have multiple successors ("quick", "slow", "big")
    if let Some(count) = cont.successor_counts.get(&b"the"[..]) {
        assert!(*count >= 3, "Expected at least 3 successors for 'the'");
    }
}

// =============================================================================
// Merge Tests
// =============================================================================

#[test]
fn test_merge_to_memory() {
    let (_dir, coordinator) = create_coordinator_with_data();
    let merger = MergeCoordinator::new(&coordinator);

    let merged = merger.merge_to_memory().expect("Failed to merge");

    // Should have all entries
    assert!(merged.len() >= 20); // 8 + 8 + 4 n-grams

    // Verify some entries
    assert_eq!(merged.get(&b"the"[..]), Some(&1_000_000));
    assert_eq!(merged.get(&b"the|quick"[..]), Some(&50_000));
}

#[test]
fn test_merge_preserves_counts() {
    let (dir, coordinator) = create_coordinator_with_data();
    let merger = MergeCoordinator::new(&coordinator);

    let output_path = dir.path().join("merged_counts.artrie");
    merger
        .merge_to_trie(&output_path, |_| {})
        .expect("Failed to merge");

    let (merged, _) =
        PersistentARTrie::<u64>::open_with_recovery(&output_path).expect("Failed to open");

    // Verify all counts are preserved
    assert_eq!(merged.get_value_bytes(b"the"), Some(1_000_000));
    assert_eq!(merged.get_value_bytes(b"a"), Some(500_000));
    assert_eq!(merged.get_value_bytes(b"the|quick"), Some(50_000));
    assert_eq!(merged.get_value_bytes(b"the|quick|brown"), Some(10_000));
}

// =============================================================================
// Parallel Storage Tests
// =============================================================================

#[test]
fn test_parallel_storage() {
    use std::sync::Arc;
    use std::thread;

    let dir = TempDir::new().expect("Failed to create temp dir");
    let config = ShardConfig::new(dir.path().join("shards"))
        .with_granularity(ShardGranularity::TwoChar)
        .with_max_open_shards(100);

    let coordinator = Arc::new(ShardCoordinator::new(config).expect("Failed to create coordinator"));

    // Spawn multiple threads to store n-grams concurrently
    let mut handles = Vec::new();

    for thread_id in 0..4 {
        let coord = Arc::clone(&coordinator);
        handles.push(thread::spawn(move || {
            for i in 0..100 {
                let prefix = match thread_id {
                    0 => "aa",
                    1 => "bb",
                    2 => "cc",
                    _ => "dd",
                };
                let ngram = format!("{}word{}|suffix", prefix, i);
                coord.store_ngram(&ngram, 1).expect("Failed to store");
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
    assert!(coordinator.contains("aaword0|suffix"));
    assert!(coordinator.contains("bbword0|suffix"));
    assert!(coordinator.contains("ccword0|suffix"));
    assert!(coordinator.contains("ddword0|suffix"));
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_empty_coordinator() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let config = ShardConfig::new(dir.path().join("shards"))
        .with_granularity(ShardGranularity::TwoChar);

    let coordinator = ShardCoordinator::new(config).expect("Failed to create coordinator");

    // Empty coordinator should return None for queries
    assert_eq!(coordinator.get("nonexistent"), None);
    assert!(!coordinator.contains("anything"));
    assert_eq!(coordinator.total_entry_count(), 0);

    // View should be empty
    let view = ShardedTrieView::new(&coordinator);
    assert!(view.is_empty());
    assert_eq!(view.len(), 0);
}

#[test]
fn test_duplicate_storage() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let config = ShardConfig::new(dir.path().join("shards"))
        .with_granularity(ShardGranularity::TwoChar);

    let coordinator = ShardCoordinator::new(config).expect("Failed to create coordinator");

    // Store same n-gram multiple times
    coordinator.store_ngram("test|word", 10).expect("store");
    coordinator.store_ngram("test|word", 20).expect("store");
    coordinator.store_ngram("test|word", 30).expect("store");

    // Count should be cumulative
    assert_eq!(coordinator.get("test|word"), Some(60));
}

#[test]
fn test_special_characters() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let config = ShardConfig::new(dir.path().join("shards"))
        .with_granularity(ShardGranularity::TwoChar);

    let coordinator = ShardCoordinator::new(config).expect("Failed to create coordinator");

    // Store n-grams with special characters (common in Google Books)
    coordinator.store_ngram("don't", 1000).expect("store");
    coordinator.store_ngram("it's|good", 500).expect("store");
    coordinator.store_ngram("U.S.A.", 300).expect("store");

    // Verify storage
    assert_eq!(coordinator.get("don't"), Some(1000));
    assert_eq!(coordinator.get("it's|good"), Some(500));
    assert_eq!(coordinator.get("U.S.A."), Some(300));
}
