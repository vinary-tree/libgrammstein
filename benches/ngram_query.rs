//! N-gram query benchmarks.
//!
//! Measures `log_prob` (word given context) and sentence-level scoring on a
//! small trained `NgramModel`. The model is built once from a synthetic
//! corpus before each benchmark group so the criterion measurements only
//! cover query cost, not training.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use libdictenstein::dynamic_dawg::DynamicDawg;
use libgrammstein::corpus::PlaintextReader;
use libgrammstein::ngram::store::TermIdStore;
use libgrammstein::ngram::{NgramEntry, NgramModel, TrainerBuilder};
use std::io::Write;
use tempfile::TempDir;

/// Build a small NgramModel from a deterministic synthetic corpus.
///
/// The corpus repeats short sentences so the trained trie has ~50 unique
/// 1-grams and ~150 unique 3-grams — small enough for sub-millisecond
/// queries while still exercising the smoothing / backoff path.
fn build_test_model() -> (TempDir, NgramModel<TermIdStore<DynamicDawg<NgramEntry>>>) {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("corpus.txt");
    let mut file = std::fs::File::create(&path).expect("create corpus");

    // ~5KB of varied 6-word sentences. Real-ish English so backoff matters.
    let sentences = [
        "the quick brown fox jumps over the lazy dog",
        "a brown dog runs quickly through the garden",
        "the lazy fox does not jump over the wall",
        "she opened the door and walked into the room",
        "they ran across the field and into the woods",
        "he picked up the book and started to read",
        "the cat watched the bird from the window sill",
        "we drove along the road past the old church",
        "I made a cup of tea and sat down to think",
        "he closed his eyes and listened to the rain",
    ];
    for _ in 0..40 {
        for s in &sentences {
            writeln!(file, "{}.", s).expect("write");
        }
    }
    drop(file);

    let reader = PlaintextReader::from_file(&path).expect("reader");
    let dictionary = DynamicDawg::<NgramEntry>::new();
    let model = TrainerBuilder::new(dictionary)
        .order(3)
        .train(reader)
        .expect("training");
    (tmp, model)
}

fn ngram_query_benchmark(c: &mut Criterion) {
    let (_tmp, model) = build_test_model();

    // log_prob: word given a 2-word context (3-gram lookup).
    c.bench_function("log_prob_3gram_in_vocab", |b| {
        b.iter(|| {
            // Words & context are in vocab — exercises the direct-hit path.
            let lp = model.log_prob(black_box("dog"), black_box(&["brown", "the"]));
            black_box(lp)
        });
    });

    // log_prob with OOV word: exercises backoff to lower-order smoothed prob.
    c.bench_function("log_prob_oov_word", |b| {
        b.iter(|| {
            let lp = model.log_prob(black_box("flibbertigibbet"), black_box(&["the"]));
            black_box(lp)
        });
    });

    // log_prob with OOV context: exercises full backoff to unigram + smoothing.
    c.bench_function("log_prob_oov_context", |b| {
        b.iter(|| {
            let lp = model.log_prob(black_box("dog"), black_box(&["frabjous", "snorgleblat"]));
            black_box(lp)
        });
    });

    // Sentence-length sweep: log_prob of multi-word phrases.
    let mut group = c.benchmark_group("log_prob_phrase");
    let phrases: &[(&str, Vec<&str>)] = &[
        ("short", vec!["the", "quick", "brown", "fox"]),
        (
            "medium",
            vec!["the", "quick", "brown", "fox", "jumps", "over", "the"],
        ),
        (
            "long",
            vec![
                "she", "opened", "the", "door", "and", "walked", "into", "the", "room",
            ],
        ),
    ];
    for (label, words) in phrases {
        group.bench_with_input(BenchmarkId::from_parameter(label), words, |b, ws| {
            b.iter(|| {
                let mut total = 0.0;
                for i in 1..ws.len() {
                    let ctx = &ws[..i];
                    total += model.log_prob(black_box(ws[i]), black_box(ctx));
                }
                black_box(total)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, ngram_query_benchmark);
criterion_main!(benches);
