//! N-gram query benchmarks.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn ngram_query_benchmark(c: &mut Criterion) {
    // TODO: Implement once NgramModel training is complete
    c.bench_function("placeholder", |b| {
        b.iter(|| black_box(1 + 1))
    });
}

criterion_group!(benches, ngram_query_benchmark);
criterion_main!(benches);
