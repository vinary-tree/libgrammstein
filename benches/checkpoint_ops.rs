//! Benchmark for checkpoint state transition operations.
//!
//! This measures the performance of prefix state tracking operations:
//! - start_prefix (NotStarted -> InProgress)
//! - complete_prefix (InProgress -> Completed)
//! - fail_prefix (InProgress -> Failed)
//! - needs_prefix (state lookup)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

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
            BenchmarkId::new("start_complete_cycle", format!("order_{}_prefixes_{}", order, prefix_count)),
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
            BenchmarkId::new("lookup", format!("order_{}_prefixes_{}", order, prefix_count)),
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
        if i % 20 != 0 {  // Complete 95%, leave 5% remaining
            checkpoint.complete_prefix(order, prefix);
        }
    }

    group.bench_with_input(
        BenchmarkId::new("find_remaining", format!("prefixes_{}_remaining_{}", prefix_count, prefix_count / 20)),
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

criterion_group!(
    benches,
    bench_state_transitions,
    bench_needs_prefix,
    bench_mixed_operations,
    bench_sparse_remaining,
);
criterion_main!(benches);
