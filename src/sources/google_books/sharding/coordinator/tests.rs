//! Unit tests for the shard coordinator.

use super::super::{ImportState, ShardGranularity};
use super::*;
use tempfile::TempDir;

#[test]
fn test_coordinator_create_and_store() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let config = ShardConfig::new(dir.path().join("shards"));

    let coordinator = ShardCoordinator::new(config).expect("Failed to create coordinator");

    // Store some n-grams
    let was_new = coordinator
        .store_ngram("the|quick", 10)
        .expect("Failed to store");
    assert!(was_new);

    let was_new = coordinator
        .store_ngram("the|quick", 5)
        .expect("Failed to store");
    assert!(!was_new);

    let was_new = coordinator
        .store_ngram("apple|pie", 3)
        .expect("Failed to store");
    assert!(was_new);

    // Query
    assert_eq!(coordinator.get("the|quick"), Some(15));
    assert_eq!(coordinator.get("apple|pie"), Some(3));
    assert_eq!(coordinator.get("nonexistent"), None);

    // Stats
    assert_eq!(coordinator.stats().unique_ngrams.load(Ordering::Relaxed), 2);
    assert_eq!(coordinator.stats().total_ngrams.load(Ordering::Relaxed), 18);
}

#[test]
fn test_coordinator_routing() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let config =
        ShardConfig::new(dir.path().join("shards")).with_granularity(ShardGranularity::TwoChar);

    let coordinator = ShardCoordinator::new(config).expect("Failed to create coordinator");

    // Store n-grams that should go to different shards
    coordinator
        .store_ngram("the|quick", 1)
        .expect("Failed to store");
    coordinator
        .store_ngram("apple|pie", 1)
        .expect("Failed to store");
    coordinator
        .store_ngram("zebra|crossing", 1)
        .expect("Failed to store");

    // Should have 3 different shards open
    assert_eq!(coordinator.open_shard_count(), 3);

    // Verify routing
    assert_eq!(coordinator.route_ngram("the|quick").prefix, "th");
    assert_eq!(coordinator.route_ngram("apple|pie").prefix, "ap");
    assert_eq!(coordinator.route_ngram("zebra|crossing").prefix, "ze");
}

#[test]
fn test_coordinator_batch_store() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    // Use TwoChar granularity for prefix-based routing tests
    let config =
        ShardConfig::new(dir.path().join("shards")).with_granularity(ShardGranularity::TwoChar);

    let coordinator = ShardCoordinator::new(config).expect("Failed to create coordinator");

    let key = ShardKey::new("th");
    let ngrams: Vec<(&[u8], u64)> = vec![
        (b"the|quick" as &[u8], 10u64),
        (b"the|slow" as &[u8], 5),
        (b"this|is" as &[u8], 3),
    ];

    let new_count = coordinator
        .store_ngrams_batch(&key, ngrams.into_iter())
        .expect("Failed to batch store");

    assert_eq!(new_count, 3);
    assert_eq!(coordinator.get("the|quick"), Some(10));
    assert_eq!(coordinator.get("the|slow"), Some(5));
    assert_eq!(coordinator.get("this|is"), Some(3));
}

#[test]
fn test_coordinator_checkpoint() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let config = ShardConfig::new(dir.path().join("shards"));

    {
        let coordinator = ShardCoordinator::new(config.clone()).expect("Failed to create");
        coordinator
            .store_ngram("the|quick", 10)
            .expect("Failed to store");
        coordinator.checkpoint_all().expect("Failed to checkpoint");
    }

    // Reopen and verify
    {
        let coordinator = ShardCoordinator::open(config).expect("Failed to open");
        // Need to explicitly open the shard since we didn't query it yet
        let key = ShardKey::new("th");
        let _ = coordinator.get_or_create_shard(&key);
        assert_eq!(coordinator.get("the|quick"), Some(10));
    }
}

#[test]
fn test_coordinator_with_global_checkpoint() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    // Use TwoChar granularity for prefix-based routing tests
    let config =
        ShardConfig::new(dir.path().join("shards")).with_granularity(ShardGranularity::TwoChar);

    {
        let coordinator =
            ShardCoordinator::new_with_checkpoints(config.clone()).expect("Failed to create");

        // Start import
        coordinator.start_import().expect("Failed to start import");
        // Verify import is in progress
        if let Some(state) = coordinator.import_state() {
            assert!(matches!(state, ImportState::InProgress { .. }));
        } else {
            panic!("Expected import state to be set");
        }

        // Store some data
        coordinator
            .store_ngram("the|quick", 10)
            .expect("Failed to store");

        // Mark prefix as in-progress
        let key = ShardKey::new("th");
        coordinator
            .set_current_prefix(&key, Some("th"))
            .expect("Failed to set prefix");

        // Set metadata
        coordinator
            .set_checkpoint_metadata("language", "en")
            .expect("Failed to set metadata");

        // Perform coordinated checkpoint
        coordinator
            .coordinated_checkpoint()
            .expect("Failed to checkpoint");

        // Mark prefix as completed
        coordinator
            .mark_prefix_completed(&key, "th")
            .expect("Failed to complete prefix");

        // Complete import
        coordinator
            .complete_import()
            .expect("Failed to complete import");
    }

    // Reopen and verify
    {
        let coordinator = ShardCoordinator::open_with_checkpoints(config).expect("Failed to open");

        // Should not need recovery (import completed successfully)
        assert!(!coordinator.needs_recovery());

        // Check metadata persisted
        assert_eq!(
            coordinator.get_checkpoint_metadata("language"),
            Some("en".to_string())
        );

        // Check completed prefixes
        assert!(coordinator.is_prefix_completed("th"));
    }
}

#[test]
fn test_coordinator_recovery_detection() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let config = ShardConfig::new(dir.path().join("shards"));

    // Simulate interrupted import
    {
        let coordinator =
            ShardCoordinator::new_with_checkpoints(config.clone()).expect("Failed to create");

        coordinator.start_import().expect("Failed to start import");
        coordinator
            .store_ngram("the|quick", 10)
            .expect("Failed to store");

        let key = ShardKey::new("th");
        coordinator
            .set_current_prefix(&key, Some("th"))
            .expect("Failed to set prefix");

        // Checkpoint but don't complete import - simulates crash
        coordinator
            .coordinated_checkpoint()
            .expect("Failed to checkpoint");

        // Drop coordinator without completing import
    }

    // Reopen should detect recovery needed
    {
        let coordinator =
            ShardCoordinator::open_with_checkpoints(config.clone()).expect("Failed to open");

        assert!(coordinator.needs_recovery());

        // Starting a new import should fail
        assert!(coordinator.start_import().is_err());

        // Resume import
        coordinator.resume_import().expect("Failed to resume");
        assert!(!coordinator.needs_recovery());

        // Now we can complete
        coordinator.complete_import().expect("Failed to complete");
    }
}

#[test]
fn test_parallel_sync() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let config =
        ShardConfig::new(dir.path().join("shards")).with_granularity(ShardGranularity::TwoChar);

    let coordinator = ShardCoordinator::new(config).expect("Failed to create coordinator");

    // Store n-grams in multiple shards
    coordinator
        .store_ngram("the|quick", 10)
        .expect("Failed to store");
    coordinator
        .store_ngram("apple|pie", 5)
        .expect("Failed to store");
    coordinator
        .store_ngram("zebra|crossing", 3)
        .expect("Failed to store");

    // Should have 3 shards open
    assert_eq!(coordinator.open_shard_count(), 3);

    // All shards should be dirty after writes
    for key in coordinator.open_shard_keys() {
        let shard = coordinator
            .get_or_create_shard(&key)
            .expect("Failed to get shard");
        assert!(shard.read().is_dirty(), "Shard {} should be dirty", key);
    }

    // Parallel sync should sync all dirty shards
    let synced = coordinator
        .sync_all_parallel(4)
        .expect("Failed to parallel sync");
    assert_eq!(synced, 3, "Should have synced 3 shards");

    // All shards should be clean after sync
    for key in coordinator.open_shard_keys() {
        let shard = coordinator
            .get_or_create_shard(&key)
            .expect("Failed to get shard");
        assert!(
            !shard.read().is_dirty(),
            "Shard {} should be clean after sync",
            key
        );
    }

    // Syncing again should return 0 (nothing to sync)
    let synced = coordinator
        .sync_all_parallel(4)
        .expect("Failed to parallel sync");
    assert_eq!(synced, 0, "Should have synced 0 shards (all clean)");
}

#[test]
fn test_is_shard_syncing() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let config =
        ShardConfig::new(dir.path().join("shards")).with_granularity(ShardGranularity::TwoChar);

    let coordinator = ShardCoordinator::new(config).expect("Failed to create coordinator");

    // Store some data
    coordinator
        .store_ngram("the|quick", 10)
        .expect("Failed to store");

    let key = ShardKey::new("th");

    // Not syncing initially
    assert!(!coordinator.is_shard_syncing(&key));

    // Manually start sync on the shard
    {
        let shard = coordinator
            .get_or_create_shard(&key)
            .expect("Failed to get shard");
        let guard = shard.read();
        assert!(guard.sync_coordinator().try_start_sync());
    }

    // Now should report syncing
    assert!(coordinator.is_shard_syncing(&key));

    // Complete sync
    {
        let shard = coordinator
            .get_or_create_shard(&key)
            .expect("Failed to get shard");
        let guard = shard.read();
        guard.sync_coordinator().complete_sync(42);
    }

    // Not syncing anymore
    assert!(!coordinator.is_shard_syncing(&key));
}

#[test]
fn test_parallel_checkpoint() {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let config =
        ShardConfig::new(dir.path().join("shards")).with_granularity(ShardGranularity::TwoChar);

    let coordinator = ShardCoordinator::new_with_checkpoints(config.clone())
        .expect("Failed to create coordinator");

    // Store n-grams in multiple shards
    coordinator
        .store_ngram("the|quick", 10)
        .expect("Failed to store");
    coordinator
        .store_ngram("apple|pie", 5)
        .expect("Failed to store");
    coordinator
        .store_ngram("zebra|crossing", 3)
        .expect("Failed to store");

    // Parallel checkpoint
    coordinator
        .coordinated_checkpoint_parallel(4)
        .expect("Failed to parallel checkpoint");

    // All shards should be clean
    for key in coordinator.open_shard_keys() {
        let shard = coordinator
            .get_or_create_shard(&key)
            .expect("Failed to get shard");
        assert!(
            !shard.read().is_dirty(),
            "Shard {} should be clean after checkpoint",
            key
        );
    }

    // Verify data survives checkpoint
    assert_eq!(coordinator.get("the|quick"), Some(10));
    assert_eq!(coordinator.get("apple|pie"), Some(5));
    assert_eq!(coordinator.get("zebra|crossing"), Some(3));
}

// ---- Lock-free overlay flush threshold ----

#[test]
fn test_flush_lockfree_over_threshold_skips_under_threshold() {
    let dir = TempDir::new().expect("tempdir");
    let config =
        ShardConfig::new(dir.path().join("shards")).with_granularity(ShardGranularity::TwoChar);
    let coordinator = ShardCoordinator::new(config).expect("create coordinator");

    // Populate two shards, each with 5 entries
    for i in 0..5 {
        coordinator
            .store_ngram(&format!("th|w{}", i), 1)
            .expect("store");
        coordinator
            .store_ngram(&format!("ap|w{}", i), 1)
            .expect("store");
    }

    // Threshold of 10 — neither shard is over
    let flushed = coordinator
        .flush_lockfree_over_threshold(10)
        .expect("flush");
    assert_eq!(
        flushed, 0,
        "no shards should be flushed when all are under threshold"
    );
    assert_eq!(coordinator.total_lockfree_entries(), 10);
}

#[test]
fn test_flush_lockfree_over_threshold_flushes_over() {
    let dir = TempDir::new().expect("tempdir");
    let config =
        ShardConfig::new(dir.path().join("shards")).with_granularity(ShardGranularity::TwoChar);
    let coordinator = ShardCoordinator::new(config).expect("create coordinator");

    // Populate one shard with 15 entries (over threshold), another with 5
    for i in 0..15 {
        coordinator
            .store_ngram(&format!("th|w{}", i), 1)
            .expect("store");
    }
    for i in 0..5 {
        coordinator
            .store_ngram(&format!("ap|w{}", i), 1)
            .expect("store");
    }

    let flushed = coordinator
        .flush_lockfree_over_threshold(10)
        .expect("flush");
    assert_eq!(flushed, 1, "only the over-threshold shard should flush");

    // The over-threshold shard's count is reset; the other still has 5
    assert_eq!(coordinator.total_lockfree_entries(), 5);
}

#[test]
fn test_total_lockfree_entries_aggregates() {
    let dir = TempDir::new().expect("tempdir");
    let config =
        ShardConfig::new(dir.path().join("shards")).with_granularity(ShardGranularity::TwoChar);
    let coordinator = ShardCoordinator::new(config).expect("create coordinator");

    // Three shards with 5/10/15 entries
    for i in 0..5 {
        coordinator
            .store_ngram(&format!("th|w{}", i), 1)
            .expect("store");
    }
    for i in 0..10 {
        coordinator
            .store_ngram(&format!("ap|w{}", i), 1)
            .expect("store");
    }
    for i in 0..15 {
        coordinator
            .store_ngram(&format!("ze|w{}", i), 1)
            .expect("store");
    }

    assert_eq!(coordinator.total_lockfree_entries(), 30);
}

// ---- commit_chunk_tx ----

#[test]
fn test_commit_chunk_tx_does_not_mark_complete() {
    // commit_chunk_tx commits the buffered n-grams but must NOT mark the
    // prefix as complete — that's commit_prefix_tx's job.
    let dir = TempDir::new().expect("tempdir");
    let config =
        ShardConfig::new(dir.path().join("shards")).with_granularity(ShardGranularity::TwoChar);
    let coordinator = ShardCoordinator::new(config).expect("create coordinator");

    let shard_key = ShardKey::new("th");
    let mut tx = coordinator
        .begin_prefix_tx(&shard_key, "th")
        .expect("begin_prefix_tx");

    // Insert 5 entries
    for i in 0..5 {
        let key = format!("the|w{}", i);
        coordinator.tx_insert(&mut tx, key.as_bytes(), 100 + i as u64);
    }

    let committed = coordinator
        .commit_chunk_tx(&mut tx)
        .expect("commit_chunk_tx");
    assert_eq!(committed, 5);

    // Stats reflect the commit
    assert!(coordinator.stats().unique_ngrams.load(Ordering::Relaxed) >= 5);

    // But the prefix is NOT yet marked complete on the shard
    let shard = coordinator
        .get_or_create_shard(&shard_key)
        .expect("get shard");
    assert!(
        !shard
            .read()
            .checkpoint_state()
            .completed_prefixes
            .contains("th"),
        "commit_chunk_tx must NOT mark the prefix as complete"
    );

    // Abort so the test cleans up cleanly (tx.inner was renewed)
    coordinator.abort_prefix_tx(tx).expect("abort");
}
