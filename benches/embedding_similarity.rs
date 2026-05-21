//! Subword embedding similarity benchmarks.
//!
//! Measures `similarity`, `word_vector`, and `most_similar` on a
//! `SubwordEmbedding` trained from a small synthetic corpus via the real
//! `EmbeddingTrainerBuilder` pipeline. The benchmarks cover three regimes:
//! in-vocab single-pair similarity (cache hot path), OOV word resolution
//! (subword-hash composition), and the `most_similar` k-NN search
//! (linear-scan over vocab × dim).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use libgrammstein::corpus::PlaintextReader;
use libgrammstein::embedding::{EmbeddingTrainerBuilder, SubwordEmbedding};
use std::io::Write;
use tempfile::TempDir;

/// Write a deterministic synthetic corpus and train a `SubwordEmbedding`
/// against it. Vocabulary is bounded by the unique tokens emitted.
fn build_test_model(unique_tokens: usize, dim: usize) -> (TempDir, SubwordEmbedding) {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("corpus.txt");
    let mut file = std::fs::File::create(&path).expect("create corpus");

    // Emit each token in many co-occurrence contexts so the trainer has
    // enough signal to assign distinct vectors.
    let mut state: u64 = 0xC0FFEE_DEADBEEF;
    for _ in 0..(unique_tokens * 50) {
        let mut sentence = Vec::with_capacity(10);
        for _ in 0..10 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let idx = (state as usize) % unique_tokens;
            sentence.push(format!("word{}", idx));
        }
        writeln!(file, "{}.", sentence.join(" ")).expect("write");
    }
    drop(file);

    let reader = PlaintextReader::from_file(&path).expect("reader");
    let model = EmbeddingTrainerBuilder::new()
        .dim(dim)
        .window_size(2)
        .min_count(1)
        .epochs(1)
        .train(reader)
        .expect("training");
    (tmp, model)
}

fn embedding_similarity_benchmark(c: &mut Criterion) {
    let (_tmp, model) = build_test_model(500, 64);

    // In-vocab pairwise similarity (cosine over two precomputed vectors).
    c.bench_function("similarity_in_vocab", |b| {
        b.iter(|| {
            let s = model.similarity(black_box("word42"), black_box("word123"));
            black_box(s)
        });
    });

    // word_vector lookup (cache hit path after first call).
    c.bench_function("word_vector_in_vocab", |b| {
        b.iter(|| {
            let v = model.word_vector(black_box("word42"));
            black_box(v)
        });
    });

    // OOV similarity: both words are out of vocab, resolved via subword hashes.
    c.bench_function("similarity_oov", |b| {
        b.iter(|| {
            let s = model.similarity(black_box("frabjous"), black_box("snorgleblat"));
            black_box(s)
        });
    });

    // most_similar(k) sweep: O(vocab_size × dim) cosine over all entries.
    let mut group = c.benchmark_group("most_similar_k");
    for k in [1, 5, 10, 25] {
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            b.iter(|| {
                let res = model.most_similar(black_box("word42"), k);
                black_box(res)
            });
        });
    }
    group.finish();

    // Vocab-size scaling: how does most_similar(10) scale with vocab size?
    let mut group = c.benchmark_group("most_similar_vocab_scaling");
    for vs in [100usize, 500, 2000] {
        let (_tmp_inner, m) = build_test_model(vs, 64);
        group.bench_with_input(BenchmarkId::from_parameter(vs), &vs, |b, _| {
            b.iter(|| {
                let res = m.most_similar(black_box("word42"), 10);
                black_box(res)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, embedding_similarity_benchmark);
criterion_main!(benches);
