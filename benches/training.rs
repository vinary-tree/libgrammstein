//! N-gram training-pipeline benchmarks.
//!
//! Measures the full count-and-build cycle of `TrainerBuilder::train` over a
//! synthetic corpus of varying sizes, with a parameter sweep over n-gram
//! order. Throughput here dominates real-corpus import time.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use libgrammstein::corpus::PlaintextReader;
use libgrammstein::ngram::{NgramEntry, TrainerBuilder};
use liblevenshtein::dictionary::pathmap::PathMapDictionary;
use std::io::Write;
use tempfile::TempDir;

/// Write a deterministic synthetic corpus of `sentences` lines to `path`.
/// Each sentence is 8-12 words from a fixed pool, producing realistic-ish
/// vocab and n-gram density.
fn write_synthetic_corpus(path: &std::path::Path, sentences: usize) {
    let pool = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "a", "cat", "sat", "on",
        "mat", "and", "watched", "bird", "in", "garden", "she", "opened", "door", "walked", "into",
        "room", "they", "ran", "across", "field", "through", "woods", "I", "made", "cup", "of",
        "tea", "sat", "down", "to", "think", "he", "picked", "up", "book", "started", "read",
        "closed", "his", "eyes", "listened", "rain", "drove", "along", "road", "past", "old",
        "church",
    ];
    let mut file = std::fs::File::create(path).expect("create corpus");
    // Simple deterministic mixer — no rand crate dependency in benches.
    let mut state: u64 = 0xCAFEBABEDEADBEEF;
    for _ in 0..sentences {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let len = 8 + (state >> 60) as usize; // 8..=23 words
        let mut words = Vec::with_capacity(len);
        for _ in 0..len {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            words.push(pool[(state as usize) % pool.len()]);
        }
        writeln!(file, "{}.", words.join(" ")).expect("write");
    }
}

fn training_benchmark(c: &mut Criterion) {
    // Sweep across corpus sizes at a fixed order (3-gram).
    let mut group = c.benchmark_group("train_by_corpus_size");
    for sentences in [100usize, 500, 2000] {
        let tmp = TempDir::new().expect("tempdir");
        let path = tmp.path().join("corpus.txt");
        write_synthetic_corpus(&path, sentences);
        let bytes = std::fs::metadata(&path).expect("metadata").len();
        group.throughput(Throughput::Bytes(bytes));
        group.bench_with_input(BenchmarkId::from_parameter(sentences), &path, |b, p| {
            b.iter(|| {
                let reader = PlaintextReader::from_file(p).expect("reader");
                let dictionary = PathMapDictionary::<NgramEntry>::new();
                let model = TrainerBuilder::new(dictionary)
                    .order(3)
                    .train(reader)
                    .expect("training");
                black_box(model)
            });
        });
    }
    group.finish();

    // Sweep across n-gram order at a fixed corpus size (500 sentences).
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("corpus.txt");
    write_synthetic_corpus(&path, 500);
    let mut group = c.benchmark_group("train_by_order");
    for order in [2usize, 3, 4, 5] {
        group.bench_with_input(BenchmarkId::from_parameter(order), &order, |b, &o| {
            b.iter(|| {
                let reader = PlaintextReader::from_file(&path).expect("reader");
                let dictionary = PathMapDictionary::<NgramEntry>::new();
                let model = TrainerBuilder::new(dictionary)
                    .order(o)
                    .train(reader)
                    .expect("training");
                black_box(model)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, training_benchmark);
criterion_main!(benches);
