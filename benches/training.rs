//! Training benchmarks.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn training_benchmark(c: &mut Criterion) {
    // TODO: Implement once training pipeline is complete
    c.bench_function("placeholder", |b| {
        b.iter(|| black_box(1 + 1))
    });
}

criterion_group!(benches, training_benchmark);
criterion_main!(benches);
