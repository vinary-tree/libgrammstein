//! Benchmark for MKN frequency count aggregation.
//!
//! Tests the performance of parallel frequency count computation
//! across sharded n-gram storage.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use libgrammstein::sources::google_books::sharding::{
    ShardConfig, ShardCoordinator, ShardGranularity,
};
use tempfile::TempDir;

/// Create test coordinator with synthetic data.
fn create_test_coordinator(num_ngrams: usize) -> (TempDir, ShardCoordinator) {
    let dir = TempDir::new().expect("Failed to create temp dir");
    let config = ShardConfig::new(dir.path().join("shards"))
        .with_granularity(ShardGranularity::TwoChar)
        .with_max_open_shards(100);

    let coordinator = ShardCoordinator::new(config).expect("Failed to create coordinator");

    // Generate synthetic n-grams with Zipf-like distribution
    for i in 0..num_ngrams {
        // 1-grams
        let word = format!("word{}", i % 1000);
        let count = ((num_ngrams - i) / 100 + 1) as u64; // Zipf-like
        coordinator.store_ngram(&word, count).expect("store");

        // 2-grams (every 5th)
        if i % 5 == 0 {
            let bigram = format!("{}|suffix{}", word, i % 100);
            coordinator
                .store_ngram(&bigram, count / 2 + 1)
                .expect("store");
        }

        // 3-grams (every 20th)
        if i % 20 == 0 {
            let trigram = format!("prefix{}|{}|end", i % 50, word);
            coordinator
                .store_ngram(&trigram, count / 3 + 1)
                .expect("store");
        }
    }

    (dir, coordinator)
}

fn bench_frequency_counts(c: &mut Criterion) {
    let mut group = c.benchmark_group("mkn_frequency_counts");

    // Test with different data sizes
    for num_ngrams in [1000, 5000, 10000] {
        let (_dir, coordinator) = create_test_coordinator(num_ngrams);

        group.bench_with_input(
            BenchmarkId::new("compute", num_ngrams),
            &coordinator,
            |b, coord| {
                use libgrammstein::sources::google_books::sharding::mkn::MknAggregator;
                let aggregator = MknAggregator::new(coord);
                b.iter(|| black_box(aggregator.compute_frequency_counts().expect("compute")));
            },
        );
    }

    group.finish();
}

fn bench_discount_params(c: &mut Criterion) {
    use libgrammstein::sources::google_books::sharding::mkn::{DiscountParams, FrequencyCounts};

    let mut group = c.benchmark_group("mkn_discount_params");

    // Typical corpus statistics (Zipf distribution: n1 >= n2 >= n3 >= n4)
    let zipf_counts = FrequencyCounts {
        n1: 10000,
        n2: 5000,
        n3: 2500,
        n4: 1250,
        total_unique: 20000,
        total_count: 100000,
    };

    // Atypical distribution (non-Zipf)
    let non_zipf_counts = FrequencyCounts {
        n1: 1000,
        n2: 5000, // n2 > n1
        n3: 2500,
        n4: 1250,
        total_unique: 10000,
        total_count: 50000,
    };

    group.bench_function("from_counts_zipf", |b| {
        b.iter(|| black_box(DiscountParams::from_counts(&zipf_counts)))
    });

    group.bench_function("from_counts_non_zipf", |b| {
        b.iter(|| black_box(DiscountParams::from_counts(&non_zipf_counts)))
    });

    group.finish();
}

fn bench_frequency_merge(c: &mut Criterion) {
    use libgrammstein::sources::google_books::sharding::mkn::FrequencyCounts;

    let mut group = c.benchmark_group("mkn_frequency_merge");

    // Create multiple frequency count structs to merge
    let counts: Vec<FrequencyCounts> = (0..100)
        .map(|i| FrequencyCounts {
            n1: 100 + i,
            n2: 50 + i / 2,
            n3: 25 + i / 4,
            n4: 12 + i / 8,
            total_unique: 200 + i,
            total_count: 1000 + i * 10,
        })
        .collect();

    group.bench_function("merge_100_sequential", |b| {
        b.iter(|| {
            let mut result = FrequencyCounts::default();
            for c in &counts {
                result.merge(c);
            }
            black_box(result)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_frequency_counts,
    bench_discount_params,
    bench_frequency_merge,
);
criterion_main!(benches);
