//! Multi-shard Google Books import micro-benchmark + eviction observability.
//!
//! Streams `N` distinct synthetic n-grams through the `ShardCoordinator` import path
//! (the same path the real Google Books import uses) with the resident-overlay budget
//! enabled vs disabled, periodically checkpointing, and reports:
//!
//!   - import throughput (n-grams/s) routed across all shards;
//!   - the import-wide `aggregate_eviction_stats()` — `nodes_evicted` (how much of the
//!     resident overlay the budget reclaims) and `resident_bytes` (the live overlay
//!     heap gauge). These are the observables the 33.79 GB → < 16 GB peak-heap
//!     validation reads.
//!
//! This is scaffolding for the deferred end-to-end peak-RSS validation. Process RSS
//! (`VmHWM`) is deliberately NOT reported here: mimalloc (optimization #1) retains
//! freed pages, so RSS lags the live heap and would understate the reduction —
//! `nodes_evicted` is the reliable live-heap reclamation observable. The real-corpus
//! peak-RSS gate is run separately (see `docs/architecture/memory-optimization.md`).
//!
//! Eviction-threshold note (why a small-`N` run reports `nodes_evicted = 0`):
//! `nodes_evicted` only becomes nonzero once a single shard's resident overlay
//! exceeds the per-shard floor `MIN_PER_SHARD_OVERLAY_BUDGET_BYTES` (64 MiB) — any
//! smaller `[budget_bytes]` is floored up to it by `overlay_eviction_config`. With
//! the default ~32 hash shards that is ~2 GiB resident (on the order of 8M+ synthetic
//! n-grams) before the first eviction fires, so smoke-scale runs exercise only the
//! `resident_bytes` gauge (nonzero under a budget, `0` when unbudgeted because
//! eviction is then unarmed) — verified: a 1000-n-gram run reports
//! `resident_bytes = 262274` budgeted vs `0` unbudgeted, `nodes_evicted = 0` in both.
//! The reclamation signal itself appears only at the real-corpus scale this scaffolds.
//!
//! Run: `cargo bench --features google-books --bench google_books_import [N] [budget_bytes]`.

use libgrammstein::sources::google_books::sharding::{ShardConfig, ShardCoordinator};
use std::time::Instant;
use tempfile::TempDir;

/// Distinct first-tokens so n-grams route across many shards, mirroring the prefix
/// spread of a real import (routing hashes the first token to a shard).
const PREFIXES: &[&str] = &[
    "the", "and", "for", "you", "was", "with", "his", "her", "had", "not", "but", "all", "one",
    "out", "are", "who", "she", "him", "has", "can",
];

/// Stream `n` synthetic n-grams through the coordinator with the given overlay budget,
/// checkpointing every 100K, and report throughput + the import-wide eviction stats.
fn run(label: &str, budget: Option<usize>, n: u64) {
    let dir = TempDir::new().expect("tempdir");
    let config = ShardConfig::new(dir.path()).with_overlay_budget_bytes(budget);
    let coordinator = ShardCoordinator::new(config).expect("create coordinator");

    let t0 = Instant::now();
    for i in 0..n {
        let prefix = PREFIXES[(i as usize) % PREFIXES.len()];
        let ngram = format!("{prefix} w{:08}", i);
        coordinator.store_ngram(&ngram, 1).expect("store ngram");
        if i % 100_000 == 99_999 {
            coordinator.checkpoint_all().expect("periodic checkpoint");
        }
    }
    let ingest = t0.elapsed();

    let tc = Instant::now();
    coordinator.checkpoint_all().expect("final checkpoint");
    let ckpt = tc.elapsed();

    let evict = coordinator.aggregate_eviction_stats();
    let rate = n as f64 / ingest.as_secs_f64();
    println!(
        "{label:<22} | {n} ngrams in {ingest:>9.2?} ({rate:>10.0} ngrams/s) | \
         final ckpt {ckpt:>9.2?} | shards = {:>4} | nodes_evicted = {:>10} | resident_bytes = {}",
        coordinator.open_shard_count(),
        evict.nodes_evicted,
        evict.resident_bytes,
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: u64 = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let budget: Option<usize> = args.next().and_then(|s| s.parse().ok());

    println!(
        "Google Books import benchmark — {n} synthetic n-grams across {} shard prefixes",
        PREFIXES.len()
    );
    run("budget OFF (unbounded)", None, n);
    run("budget ON", budget.or(Some(64 * 1024 * 1024)), n);
}
