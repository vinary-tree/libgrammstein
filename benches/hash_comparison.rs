//! Benchmark comparing hash implementations on real vocabulary data.
//!
//! Compares:
//! - Current hybrid (gxhash + FNV-1a)
//! - Proposed hybrid (gxhash + xxh3 for short inputs)
//! - xxhash3-only
//!
//! Uses n-gram words from `bak/english.vocab.artrie` for realistic benchmarking.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use libgrammstein::ngram::vocabulary::open_vocabulary;
use libgrammstein::util::hash::{fnv1a, safe_hash, safe_hash_with_seed, GXHASH_MIN_SIZE};
use std::path::Path;
use xxhash_rust::xxh3::{xxh3_64, xxh3_64_with_seed};

// ============================================================================
// Load vocabulary words
// ============================================================================

fn load_vocabulary_words() -> Vec<String> {
    let vocab_path = Path::new("bak/english.vocab.artrie");
    if !vocab_path.exists() {
        eprintln!(
            "Warning: {} not found, using synthetic data",
            vocab_path.display()
        );
        return generate_synthetic_words(10000);
    }

    let vocab = open_vocabulary(vocab_path).expect("Failed to open vocabulary");

    // Use O(1) get_term() lookups to collect all words. Hold the read guard
    // for the duration of the iteration to amortize lock-acquisition cost.
    let guard = vocab.read();
    let count = guard.len() as u64;
    (1..=count).filter_map(|i| guard.get_term(i)).collect()
}

fn generate_synthetic_words(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| {
            // Mix of short and long words
            if i % 4 == 0 {
                format!("w{}", i) // Very short
            } else if i % 4 == 1 {
                format!("word{}", i) // Medium
            } else if i % 4 == 2 {
                format!("longword{}", i) // Longer
            } else {
                format!("verylongwordindeed{}", i) // Long (>16 bytes)
            }
        })
        .collect()
}

// ============================================================================
// Hash implementations
// ============================================================================

/// Current hybrid implementation (gxhash for ≥16 bytes, FNV-1a otherwise)
#[inline]
fn hash_current_hybrid(bytes: &[u8]) -> u64 {
    safe_hash(bytes)
}

/// Current hybrid with seed
#[inline]
fn hash_current_hybrid_with_seed(bytes: &[u8], seed: u64) -> u64 {
    safe_hash_with_seed(bytes, seed)
}

/// Proposed hybrid: xxh3 for <16 bytes, gxhash for ≥16 bytes
#[inline]
fn hash_proposed_hybrid(bytes: &[u8]) -> u64 {
    if bytes.len() >= GXHASH_MIN_SIZE {
        gxhash::gxhash64(bytes, 0)
    } else {
        xxh3_64(bytes)
    }
}

/// Proposed hybrid with seed
#[inline]
fn hash_proposed_hybrid_with_seed(bytes: &[u8], seed: u64) -> u64 {
    if bytes.len() >= GXHASH_MIN_SIZE {
        gxhash::gxhash64(bytes, seed as i64)
    } else {
        xxh3_64_with_seed(bytes, seed)
    }
}

/// xxhash3-only (simplest option)
#[inline]
fn hash_xxh3(bytes: &[u8]) -> u64 {
    xxh3_64(bytes)
}

/// xxhash3 with seed
#[inline]
fn hash_xxh3_with_seed(bytes: &[u8], seed: u64) -> u64 {
    xxh3_64_with_seed(bytes, seed)
}

/// Direct gxhash (for ≥16 byte inputs only)
#[inline]
fn hash_gxhash_direct(bytes: &[u8]) -> u64 {
    gxhash::gxhash64(bytes, 0)
}

/// Direct FNV-1a (for reference)
#[inline]
fn hash_fnv1a_direct(bytes: &[u8]) -> u64 {
    fnv1a(bytes)
}

// ============================================================================
// Benchmarks
// ============================================================================

fn bench_single_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_single");

    // Test different input sizes
    let test_inputs: &[(&[u8], &str)] = &[
        (b"the", "3-byte"),
        (b"hello", "5-byte"),
        (b"international", "13-byte"),
        (b"0123456789abcdef", "16-byte"),
        (b"this is a longer string", "23-byte"),
        (
            b"this is a much longer string that exceeds typical word lengths",
            "63-byte",
        ),
    ];

    for &(input, label) in test_inputs {
        let is_long = input.len() >= GXHASH_MIN_SIZE;

        // Proposed hybrid (xxh3 + gxhash)
        group.bench_with_input(
            BenchmarkId::new("proposed_hybrid", label),
            input,
            |b, bytes| b.iter(|| black_box(hash_proposed_hybrid(black_box(bytes)))),
        );

        // xxh3-only
        group.bench_with_input(BenchmarkId::new("xxh3_only", label), input, |b, bytes| {
            b.iter(|| black_box(hash_xxh3(black_box(bytes))))
        });

        // Current hybrid for reference
        group.bench_with_input(
            BenchmarkId::new("current_hybrid", label),
            input,
            |b, bytes| b.iter(|| black_box(hash_current_hybrid(black_box(bytes)))),
        );

        // Only benchmark direct gxhash for safe inputs (≥16 bytes)
        if is_long {
            group.bench_with_input(
                BenchmarkId::new("gxhash_direct", label),
                input,
                |b, bytes| b.iter(|| black_box(hash_gxhash_direct(black_box(bytes)))),
            );
        }

        // FNV-1a for reference on short inputs
        if !is_long {
            group.bench_with_input(BenchmarkId::new("fnv1a", label), input, |b, bytes| {
                b.iter(|| black_box(hash_fnv1a_direct(black_box(bytes))))
            });
        }
    }

    group.finish();
}

fn bench_vocabulary_hash(c: &mut Criterion) {
    let words = load_vocabulary_words();

    if words.is_empty() {
        eprintln!("No vocabulary words loaded, skipping vocabulary benchmark");
        return;
    }

    let word_bytes: Vec<&[u8]> = words.iter().map(|s| s.as_bytes()).collect();

    // Categorize by length for analysis
    let short_words: Vec<&[u8]> = word_bytes
        .iter()
        .copied()
        .filter(|b| b.len() < GXHASH_MIN_SIZE)
        .collect();
    let long_words: Vec<&[u8]> = word_bytes
        .iter()
        .copied()
        .filter(|b| b.len() >= GXHASH_MIN_SIZE)
        .collect();

    println!(
        "\nVocabulary stats: {} total words, {} short (<16B), {} long (≥16B)",
        words.len(),
        short_words.len(),
        long_words.len()
    );

    // =========================================================================
    // Main comparison: xxh3-only vs proposed hybrid (xxh3+gxhash)
    // =========================================================================
    let mut group = c.benchmark_group("xxh3_vs_proposed_hybrid");
    group.throughput(Throughput::Elements(words.len() as u64));

    // All vocabulary words
    group.bench_function("xxh3_only_all", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for bytes in &word_bytes {
                sum = sum.wrapping_add(hash_xxh3(black_box(bytes)));
            }
            black_box(sum)
        })
    });

    group.bench_function("proposed_hybrid_all", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for bytes in &word_bytes {
                sum = sum.wrapping_add(hash_proposed_hybrid(black_box(bytes)));
            }
            black_box(sum)
        })
    });

    // Short words only (<16 bytes) - both use xxh3, should be identical
    if !short_words.is_empty() {
        group.throughput(Throughput::Elements(short_words.len() as u64));

        group.bench_function("xxh3_only_short", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for bytes in &short_words {
                    sum = sum.wrapping_add(hash_xxh3(black_box(bytes)));
                }
                black_box(sum)
            })
        });

        group.bench_function("proposed_hybrid_short", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for bytes in &short_words {
                    sum = sum.wrapping_add(hash_proposed_hybrid(black_box(bytes)));
                }
                black_box(sum)
            })
        });
    }

    // Long words only (≥16 bytes) - xxh3 vs gxhash
    if !long_words.is_empty() {
        group.throughput(Throughput::Elements(long_words.len() as u64));

        group.bench_function("xxh3_only_long", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for bytes in &long_words {
                    sum = sum.wrapping_add(hash_xxh3(black_box(bytes)));
                }
                black_box(sum)
            })
        });

        group.bench_function("proposed_hybrid_long", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for bytes in &long_words {
                    sum = sum.wrapping_add(hash_proposed_hybrid(black_box(bytes)));
                }
                black_box(sum)
            })
        });
    }

    group.finish();

    // =========================================================================
    // Reference: current hybrid for comparison
    // =========================================================================
    let mut group = c.benchmark_group("current_hybrid_reference");
    group.throughput(Throughput::Elements(words.len() as u64));

    group.bench_function("current_hybrid_all", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for bytes in &word_bytes {
                sum = sum.wrapping_add(hash_current_hybrid(black_box(bytes)));
            }
            black_box(sum)
        })
    });

    group.finish();
}

fn bench_hash_with_seed(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_with_seed");

    // Typical n-gram key sizes
    let test_inputs: &[(&[u8], &str)] = &[
        (b"the", "3-byte"),
        (b"hello world", "11-byte"),
        (b"this is a longer key", "20-byte"),
    ];

    let seed = 0x12345678u64;

    for &(input, label) in test_inputs {
        group.bench_with_input(
            BenchmarkId::new("hybrid_seeded", label),
            input,
            |b, bytes| b.iter(|| black_box(hash_current_hybrid_with_seed(black_box(bytes), seed))),
        );

        group.bench_with_input(BenchmarkId::new("xxh3_seeded", label), input, |b, bytes| {
            b.iter(|| black_box(hash_xxh3_with_seed(black_box(bytes), seed)))
        });
    }

    group.finish();
}

fn bench_ngram_keys(c: &mut Criterion) {
    let words = load_vocabulary_words();

    if words.len() < 100 {
        eprintln!("Not enough vocabulary words for n-gram benchmark");
        return;
    }

    // Generate realistic n-gram keys by concatenating vocabulary words
    let ngram_keys: Vec<String> = (0..10000)
        .map(|i| {
            let order = 1 + (i % 5); // 1 to 5-grams
            let mut key = String::new();
            for j in 0..order {
                if j > 0 {
                    key.push(' ');
                }
                key.push_str(&words[(i + j) % words.len()]);
            }
            key
        })
        .collect();

    let ngram_bytes: Vec<&[u8]> = ngram_keys.iter().map(|s| s.as_bytes()).collect();

    // Analyze n-gram key lengths
    let short_keys: usize = ngram_bytes
        .iter()
        .filter(|b| b.len() < GXHASH_MIN_SIZE)
        .count();
    let long_keys: usize = ngram_bytes
        .iter()
        .filter(|b| b.len() >= GXHASH_MIN_SIZE)
        .count();
    let avg_len: f64 =
        ngram_bytes.iter().map(|b| b.len()).sum::<usize>() as f64 / ngram_bytes.len() as f64;

    println!(
        "\nN-gram key stats: {} total, {} short (<16B), {} long (≥16B), avg len: {:.1}B",
        ngram_keys.len(),
        short_keys,
        long_keys,
        avg_len
    );

    let mut group = c.benchmark_group("ngram_keys_xxh3_vs_proposed");
    group.throughput(Throughput::Elements(ngram_keys.len() as u64));

    group.bench_function("xxh3_only", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for bytes in &ngram_bytes {
                sum = sum.wrapping_add(hash_xxh3(black_box(bytes)));
            }
            black_box(sum)
        })
    });

    group.bench_function("proposed_hybrid", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for bytes in &ngram_bytes {
                sum = sum.wrapping_add(hash_proposed_hybrid(black_box(bytes)));
            }
            black_box(sum)
        })
    });

    group.bench_function("current_hybrid", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for bytes in &ngram_bytes {
                sum = sum.wrapping_add(hash_current_hybrid(black_box(bytes)));
            }
            black_box(sum)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_hash,
    bench_vocabulary_hash,
    bench_hash_with_seed,
    bench_ngram_keys,
);
criterion_main!(benches);
