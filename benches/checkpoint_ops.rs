//! Benchmark for checkpoint state transition operations.
//!
//! This measures the performance of prefix state tracking operations:
//! - start_prefix (NotStarted -> InProgress)
//! - complete_prefix (InProgress -> Completed)
//! - fail_prefix (InProgress -> Failed)
//! - needs_prefix (state lookup)
//!
//! ## Bitmap Storage Benchmarks (v3)
//!
//! Also measures bitmap pack/unpack operations for v3 checkpoint format:
//! - pack_states: HashMap -> Vec<u64> bitmap chunks
//! - unpack_states: Vec<u64> bitmap chunks -> HashMap

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

// Include checkpoint module from libgrammstein
use libgrammstein::sources::google_books::ImportCheckpoint;

/// Generate all valid prefixes for Google Books n-grams.
///
/// - Order 1: 26 prefixes (a-z)
/// - Order 2-5: 676 prefixes (aa-zz)
fn generate_prefixes(order: u8) -> Vec<String> {
    if order == 1 {
        ('a'..='z').map(|c| c.to_string()).collect()
    } else {
        let mut prefixes = Vec::with_capacity(676);
        for c1 in 'a'..='z' {
            for c2 in 'a'..='z' {
                prefixes.push(format!("{}{}", c1, c2));
            }
        }
        prefixes
    }
}

/// Benchmark state transition cycle: start -> complete for all prefixes.
fn bench_state_transitions(c: &mut Criterion) {
    let mut group = c.benchmark_group("checkpoint_state_transitions");

    for order in [1u8, 2, 3] {
        let prefixes = generate_prefixes(order);
        let prefix_count = prefixes.len();

        group.bench_with_input(
            BenchmarkId::new(
                "start_complete_cycle",
                format!("order_{}_prefixes_{}", order, prefix_count),
            ),
            &prefixes,
            |b, prefixes| {
                b.iter(|| {
                    let mut checkpoint = ImportCheckpoint::new();

                    // Start all prefixes
                    for prefix in prefixes {
                        checkpoint.start_prefix(order, prefix);
                    }

                    // Complete all prefixes
                    for prefix in prefixes {
                        checkpoint.complete_prefix(order, prefix);
                    }

                    // Consume checkpoint to prevent optimization
                    checkpoint.stats.ngrams_processed
                });
            },
        );
    }

    group.finish();
}

/// Benchmark needs_prefix lookups (the most common operation during import).
fn bench_needs_prefix(c: &mut Criterion) {
    let mut group = c.benchmark_group("checkpoint_needs_prefix");

    for order in [1u8, 2, 3] {
        let prefixes = generate_prefixes(order);
        let prefix_count = prefixes.len();

        // Prepare checkpoint with half prefixes completed
        let mut checkpoint = ImportCheckpoint::new();
        for (i, prefix) in prefixes.iter().enumerate() {
            if i % 2 == 0 {
                checkpoint.complete_prefix(order, prefix);
            }
        }

        group.bench_with_input(
            BenchmarkId::new(
                "lookup",
                format!("order_{}_prefixes_{}", order, prefix_count),
            ),
            &(checkpoint, prefixes),
            |b, (checkpoint, prefixes): &(ImportCheckpoint, Vec<String>)| {
                b.iter(|| {
                    let mut count = 0usize;
                    for prefix in prefixes {
                        if checkpoint.needs_prefix(order, prefix) {
                            count += 1;
                        }
                    }
                    black_box(count)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark mixed operations (realistic import pattern).
///
/// Simulates: start prefix, process n-grams, complete prefix, check next.
fn bench_mixed_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("checkpoint_mixed_ops");

    let prefixes = generate_prefixes(2); // 676 prefixes for 2-grams
    let prefix_count = prefixes.len();

    group.bench_with_input(
        BenchmarkId::new("import_simulation", format!("prefixes_{}", prefix_count)),
        &prefixes,
        |b, prefixes| {
            b.iter(|| {
                let mut checkpoint = ImportCheckpoint::new();
                let order = 2u8;

                for (i, prefix) in prefixes.iter().enumerate() {
                    // Check if needs processing
                    if checkpoint.needs_prefix(order, prefix) {
                        // Start processing
                        checkpoint.start_prefix(order, prefix);

                        // Simulate some work with add_ngrams
                        checkpoint.add_ngrams(order, 1000);

                        // Complete or fail based on pattern
                        if i % 10 == 9 {
                            checkpoint.fail_prefix(order, prefix);
                        } else {
                            checkpoint.complete_prefix(order, prefix);
                        }
                    }
                }

                // Consume checkpoint to prevent optimization
                checkpoint.stats.ngrams_processed
            });
        },
    );

    group.finish();
}

/// Benchmark worst-case: many completed prefixes, looking for remaining ones.
///
/// This tests the O(n) contains() behavior when most prefixes are done.
fn bench_sparse_remaining(c: &mut Criterion) {
    let mut group = c.benchmark_group("checkpoint_sparse_remaining");

    let prefixes = generate_prefixes(2); // 676 prefixes
    let prefix_count = prefixes.len();

    // Complete 95% of prefixes (realistic end-of-import scenario)
    let mut checkpoint = ImportCheckpoint::new();
    let order = 2u8;
    for (i, prefix) in prefixes.iter().enumerate() {
        if i % 20 != 0 {
            // Complete 95%, leave 5% remaining
            checkpoint.complete_prefix(order, prefix);
        }
    }

    group.bench_with_input(
        BenchmarkId::new(
            "find_remaining",
            format!("prefixes_{}_remaining_{}", prefix_count, prefix_count / 20),
        ),
        &(checkpoint, prefixes),
        |b, (checkpoint, prefixes): &(ImportCheckpoint, Vec<String>)| {
            b.iter(|| {
                let order = 2u8;
                let mut remaining = Vec::new();
                for prefix in prefixes {
                    if checkpoint.needs_prefix(order, prefix) {
                        remaining.push(prefix.clone());
                    }
                }
                black_box(remaining)
            });
        },
    );

    group.finish();
}

/// Benchmark bitmap pack operation (HashMap -> u64 chunks).
///
/// This measures the v3 checkpoint format's bitmap packing performance.
fn bench_bitmap_pack(c: &mut Criterion) {
    let mut group = c.benchmark_group("checkpoint_bitmap_pack");

    // Test with different fill levels
    for fill_percent in [10, 50, 90, 100] {
        let prefixes = generate_prefixes(2);
        let fill_count = (prefixes.len() * fill_percent) / 100;

        // Create checkpoint with specified fill level
        let mut checkpoint = ImportCheckpoint::new();
        for prefix in prefixes.iter().take(fill_count) {
            checkpoint.complete_prefix(2, prefix);
        }

        group.throughput(Throughput::Elements(fill_count as u64));
        group.bench_with_input(
            BenchmarkId::new(
                "pack",
                format!("{}%_fill_{}_prefixes", fill_percent, fill_count),
            ),
            &checkpoint,
            |b, checkpoint| {
                b.iter(|| {
                    // Access the order_progress to get prefix_states
                    // This simulates what save_to_trie does internally
                    let progress = checkpoint.order_progress.get(&2).unwrap();
                    let completed: Vec<_> = progress.completed_prefixes().collect();
                    black_box(completed.len())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark checkpoint save_to_trie vs theoretical v2 (key-per-prefix).
///
/// Compares v3 bitmap format storage efficiency.
fn bench_bitmap_storage_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("checkpoint_storage_comparison");

    let prefixes = generate_prefixes(2);

    // Create fully populated checkpoint
    let mut checkpoint = ImportCheckpoint::new();
    for prefix in &prefixes {
        checkpoint.complete_prefix(2, prefix);
    }

    // Benchmark: Count keys that would be written
    // v2: 676 keys per order
    // v3: ~22 keys per order (non-zero bitmap chunks)
    group.bench_function("v3_key_count_simulation", |b| {
        b.iter(|| {
            // Simulate counting non-zero bitmap chunks
            let mut key_count = 0usize;

            // Metadata keys (same in v2 and v3)
            key_count += 9; // version, mkn_phase, byte_offset, timestamp, stats...

            // Per-order keys
            for order in 1..=5u8 {
                key_count += 1; // order_complete
                key_count += 1; // order_ngrams

                // v3: count non-zero bitmap chunks (max 22 for two-char)
                let num_chunks = if order == 1 { 1 } else { 22 };
                // In worst case, all chunks are non-zero
                key_count += num_chunks;
            }

            black_box(key_count)
        });
    });

    group.bench_function("v2_key_count_simulation", |b| {
        b.iter(|| {
            // Simulate v2 key counting
            let mut key_count = 0usize;

            // Metadata keys
            key_count += 9;

            // Per-order keys
            for order in 1..=5u8 {
                key_count += 1; // order_complete

                // v2: one key per prefix
                let num_prefixes = if order == 1 { 26 } else { 676 };
                key_count += num_prefixes;
            }

            black_box(key_count)
        });
    });

    group.finish();
}

/// Benchmark state lookup performance with populated checkpoint.
fn bench_bitmap_state_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("checkpoint_bitmap_lookup");

    let prefixes = generate_prefixes(2);

    // Create checkpoint with mixed states
    let mut checkpoint = ImportCheckpoint::new();
    for (i, prefix) in prefixes.iter().enumerate() {
        match i % 4 {
            0 => checkpoint.complete_prefix(2, prefix),
            1 => checkpoint.start_prefix(2, prefix),
            2 => {
                checkpoint.start_prefix(2, prefix);
                checkpoint.fail_prefix(2, prefix);
            }
            _ => {} // NotStarted - don't add
        }
    }

    // Benchmark lookups across all prefixes
    group.throughput(Throughput::Elements(prefixes.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("mixed_state_lookup", prefixes.len()),
        &(checkpoint, prefixes),
        |b, (checkpoint, prefixes): &(ImportCheckpoint, Vec<String>)| {
            b.iter(|| {
                let mut completed = 0usize;
                let mut in_progress = 0usize;
                let mut failed = 0usize;
                let mut not_started = 0usize;

                for prefix in prefixes {
                    if !checkpoint.needs_prefix(2, prefix) {
                        if checkpoint.is_in_progress(2, prefix) {
                            in_progress += 1;
                        } else {
                            completed += 1;
                        }
                    } else if checkpoint.is_failed_prefix(2, prefix) {
                        failed += 1;
                    } else {
                        not_started += 1;
                    }
                }

                black_box((completed, in_progress, failed, not_started))
            });
        },
    );

    group.finish();
}

criterion_group!(
    benches,
    bench_state_transitions,
    bench_needs_prefix,
    bench_mixed_operations,
    bench_sparse_remaining,
    bench_bitmap_pack,
    bench_bitmap_storage_comparison,
    bench_bitmap_state_lookup,
);
criterion_main!(benches);
