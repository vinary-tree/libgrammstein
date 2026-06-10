//! Overlay-heap eviction micro-benchmark (the OOM fix, optimization #15).
//!
//! Inserts `N` distinct synthetic n-grams into a single shard with the
//! lock-free-overlay budget **disabled** vs **enabled** (4 MiB), reporting:
//!
//!   - ingest throughput (n-grams/s) — does arming eviction regress writes?
//!   - the final-checkpoint latency — does the budget tail add eviction cost?
//!   - `nodes_evicted` — how much of the resident overlay the budget reclaims.
//!
//! `nodes_evicted` is the reliable RAM-reclamation observable (each evicted node
//! drops its `Arc`). Process RSS (`VmHWM`) is deliberately NOT reported: mimalloc
//! (optimization #1) retains freed pages, so RSS lags the live heap and would
//! understate the reduction — the live-heap reclamation is what `nodes_evicted`
//! captures. The end-to-end peak-RSS gate runs on a real corpus (see
//! `docs/architecture/memory-optimization.md`).
//!
//! Run: `cargo bench --features google-books --bench overlay_eviction [N]`.

use libdictenstein::persistent_artrie::eviction::EvictionConfig;
use libgrammstein::sources::google_books::sharding::{ShardHandle, ShardKey};
use std::time::Instant;
use tempfile::TempDir;

/// Insert `n` distinct n-grams into one shard, checkpointing every 100K, and
/// report ingest throughput + final-checkpoint latency + reclamation.
fn run(label: &str, budget: Option<usize>, n: u64) {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().join("bench.artrie");
    let shard = ShardHandle::create(ShardKey::new("th"), &path).expect("create shard");
    shard
        .arm_eviction(budget.map(|b| EvictionConfig {
            resident_budget_bytes: Some(b),
            ..EvictionConfig::without_memory_monitor()
        }))
        .expect("arm eviction");

    let t0 = Instant::now();
    for i in 0..n {
        shard
            .increment_lockfree(format!("th|w{:08}", i).as_bytes(), 1)
            .expect("increment");
        if i % 100_000 == 99_999 {
            shard.checkpoint().expect("periodic checkpoint");
        }
    }
    let ingest = t0.elapsed();

    let tc = Instant::now();
    shard.checkpoint().expect("final checkpoint");
    let ckpt = tc.elapsed();

    let stats = shard.eviction_stats();
    let rate = n as f64 / ingest.as_secs_f64();
    println!(
        "{label:<22} | {n} ngrams in {ingest:>9.2?} ({rate:>10.0} ngrams/s) | \
         final ckpt {ckpt:>9.2?} | nodes_evicted = {}",
        stats.nodes_evicted
    );

    // Sanity: the run must be lossless regardless of eviction.
    assert_eq!(shard.get(b"th|w00000000"), Some(1), "first key lost");
    assert_eq!(
        shard.get(format!("th|w{:08}", n - 1).as_bytes()),
        Some(1),
        "last key lost"
    );
}

fn main() {
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);

    println!("overlay-eviction bench: {n} synthetic n-grams, single shard\n");
    println!(
        "{:<22} | {:<46} | {:<16} | reclamation",
        "config", "ingest", "checkpoint"
    );
    run("eviction OFF", None, n);
    run("eviction ON (4 MiB)", Some(4 * 1024 * 1024), n);
}
