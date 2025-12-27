//! Embedding similarity benchmarks.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn embedding_similarity_benchmark(c: &mut Criterion) {
    // TODO: Implement once SubwordEmbedding is complete
    c.bench_function("placeholder", |b| {
        b.iter(|| black_box(1 + 1))
    });
}

criterion_group!(benches, embedding_similarity_benchmark);
criterion_main!(benches);
