//! Benchmark for varint encoding/decoding candidates.
//!
//! Compares:
//! - Baseline: Current scalar LEB128 implementation
//! - varint-simd: SIMD-accelerated LEB128
//! - vu128: Prefix varint (length determinable from first byte)
//! - stream-vbyte: Separated control/data streams for bulk SIMD

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use libgrammstein::ngram::{decode_varint, encode_varint};
use stream_vbyte::scalar::Scalar;

// ============================================================================
// Test Data Generation
// ============================================================================

/// Generate test values with realistic vocabulary index distribution.
///
/// Zipf distribution: most words have low indices, few have high indices.
fn generate_zipf_indices(count: usize) -> Vec<u64> {
    // Simulates vocabulary where:
    // - 50% of lookups hit indices 0-127 (1 byte)
    // - 30% hit indices 128-16383 (2 bytes)
    // - 15% hit indices 16384-2097151 (3 bytes)
    // - 5% hit larger indices
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        let r = (i * 17 + 31) % 100; // Pseudo-random distribution
        let value = match r {
            0..=49 => (i % 128) as u64,              // 1-byte
            50..=79 => 128 + (i % 16256) as u64,     // 2-byte
            80..=94 => 16384 + (i % 2080767) as u64, // 3-byte
            _ => 2097152 + (i % 100000) as u64,      // 4+ byte
        };
        values.push(value);
    }
    values
}

/// Generate n-gram key sizes (1-5 indices per key).
fn generate_ngram_sizes(count: usize) -> Vec<usize> {
    (0..count).map(|i| 1 + (i % 5)).collect()
}

// ============================================================================
// Baseline: Current Scalar LEB128
// ============================================================================

fn encode_baseline(values: &[u64]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(values.len() * 2);
    for &v in values {
        encode_varint(v, &mut buf);
    }
    buf
}

fn decode_baseline(bytes: &[u8]) -> Vec<u64> {
    let mut result = Vec::new();
    let mut offset = 0;
    while offset < bytes.len() {
        if let Some((value, consumed)) = decode_varint(&bytes[offset..]) {
            result.push(value);
            offset += consumed;
        } else {
            break;
        }
    }
    result
}

// ============================================================================
// varint-simd
// ============================================================================

fn encode_varint_simd_bulk(values: &[u64]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(values.len() * 10);
    for &v in values {
        let (encoded, len) = varint_simd::encode::<u64>(v);
        buf.extend_from_slice(&encoded[..len as usize]);
    }
    buf
}

fn decode_varint_simd_bulk(bytes: &[u8]) -> Vec<u64> {
    let mut result = Vec::new();
    let mut offset = 0;
    while offset < bytes.len() {
        // varint-simd handles padding internally for slices < 16 bytes
        if let Ok((value, len)) = varint_simd::decode::decode::<u64>(&bytes[offset..]) {
            result.push(value);
            offset += len as usize;
        } else {
            break;
        }
    }
    result
}

// ============================================================================
// vu128
// ============================================================================

fn encode_vu128_bulk(values: &[u64]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(values.len() * 9);
    for &v in values {
        let mut tmp = [0u8; 9];
        let len = vu128::encode_u64(&mut tmp, v);
        buf.extend_from_slice(&tmp[..len]);
    }
    buf
}

fn decode_vu128_bulk(bytes: &[u8]) -> Vec<u64> {
    let mut result = Vec::new();
    let mut offset = 0;
    while offset < bytes.len() {
        // vu128::decode_u64 requires exactly 9 bytes
        if bytes.len() - offset >= 9 {
            let slice: &[u8; 9] = bytes[offset..offset + 9].try_into().expect("slice len");
            let (value, len) = vu128::decode_u64(slice);
            if len == 0 {
                break;
            }
            result.push(value);
            offset += len;
        } else {
            // Handle tail bytes - copy to padded buffer
            let mut padded = [0u8; 9];
            let remaining = bytes.len() - offset;
            padded[..remaining].copy_from_slice(&bytes[offset..]);
            let (value, len) = vu128::decode_u64(&padded);
            if len == 0 || len > remaining {
                break;
            }
            result.push(value);
            offset += len;
        }
    }
    result
}

// ============================================================================
// stream-vbyte (u32 only, for comparison)
// ============================================================================

/// Calculate the maximum encoded size for stream-vbyte.
fn stream_vbyte_max_size(count: usize) -> usize {
    // Control bytes: 1 byte per 4 values (2 bits each)
    // Data bytes: up to 4 bytes per value
    let control_bytes = (count + 3) / 4;
    let data_bytes = count * 4;
    control_bytes + data_bytes
}

fn encode_stream_vbyte(values: &[u32]) -> Vec<u8> {
    let max_size = stream_vbyte_max_size(values.len());
    let mut output = vec![0u8; max_size];
    let encoded_len = stream_vbyte::encode::encode::<Scalar>(values, &mut output);
    output.truncate(encoded_len);
    output
}

fn decode_stream_vbyte(bytes: &[u8], count: usize) -> Vec<u32> {
    let mut output = vec![0u32; count];
    stream_vbyte::decode::decode::<Scalar>(bytes, count, &mut output);
    output
}

// ============================================================================
// Benchmarks
// ============================================================================

fn bench_single_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("varint_single_encode");

    // Test different value sizes
    let test_values: &[(u64, &str)] = &[
        (50, "1-byte"),
        (5000, "2-byte"),
        (500000, "3-byte"),
        (50000000, "4-byte"),
    ];

    for &(value, label) in test_values {
        group.bench_with_input(BenchmarkId::new("baseline", label), &value, |b, &v| {
            let mut buf = Vec::with_capacity(10);
            b.iter(|| {
                buf.clear();
                encode_varint(black_box(v), &mut buf);
                buf.len()
            });
        });

        group.bench_with_input(BenchmarkId::new("varint-simd", label), &value, |b, &v| {
            b.iter(|| {
                let (encoded, len) = varint_simd::encode::<u64>(black_box(v));
                black_box((encoded, len))
            });
        });

        group.bench_with_input(BenchmarkId::new("vu128", label), &value, |b, &v| {
            let mut buf = [0u8; 9];
            b.iter(|| {
                let len = vu128::encode_u64(&mut buf, black_box(v));
                black_box(len)
            });
        });
    }

    group.finish();
}

fn bench_single_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("varint_single_decode");

    let test_values: &[(u64, &str)] = &[
        (50, "1-byte"),
        (5000, "2-byte"),
        (500000, "3-byte"),
        (50000000, "4-byte"),
    ];

    for &(value, label) in test_values {
        // Prepare encoded data
        let mut baseline_buf = Vec::new();
        encode_varint(value, &mut baseline_buf);
        // Pad for SIMD (varint-simd prefers 16+ bytes for optimal SIMD path)
        let mut simd_buf = baseline_buf.clone();
        simd_buf.resize(16, 0);

        let mut vu128_buf = [0u8; 9];
        let _ = vu128::encode_u64(&mut vu128_buf, value);

        group.bench_with_input(
            BenchmarkId::new("baseline", label),
            &baseline_buf,
            |b, buf| {
                b.iter(|| black_box(decode_varint(black_box(buf.as_slice()))));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("varint-simd", label),
            &simd_buf,
            |b, buf| {
                b.iter(|| {
                    black_box(varint_simd::decode::decode::<u64>(black_box(
                        buf.as_slice(),
                    )))
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("vu128", label), &vu128_buf, |b, buf| {
            b.iter(|| black_box(vu128::decode_u64(black_box(buf))));
        });
    }

    group.finish();
}

fn bench_bulk_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("varint_bulk_encode");

    for count in [100, 1000, 10000, 100000] {
        let values = generate_zipf_indices(count);
        let values_u32: Vec<u32> = values.iter().map(|&v| v as u32).collect();

        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(BenchmarkId::new("baseline", count), &values, |b, vals| {
            b.iter(|| black_box(encode_baseline(black_box(vals))));
        });

        group.bench_with_input(
            BenchmarkId::new("varint-simd", count),
            &values,
            |b, vals| {
                b.iter(|| black_box(encode_varint_simd_bulk(black_box(vals))));
            },
        );

        group.bench_with_input(BenchmarkId::new("vu128", count), &values, |b, vals| {
            b.iter(|| black_box(encode_vu128_bulk(black_box(vals))));
        });

        group.bench_with_input(
            BenchmarkId::new("stream-vbyte", count),
            &values_u32,
            |b, vals| {
                b.iter(|| black_box(encode_stream_vbyte(black_box(vals))));
            },
        );
    }

    group.finish();
}

fn bench_bulk_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("varint_bulk_decode");

    for count in [100, 1000, 10000, 100000] {
        let values = generate_zipf_indices(count);
        let values_u32: Vec<u32> = values.iter().map(|&v| v as u32).collect();

        // Pre-encode data
        let baseline_encoded = encode_baseline(&values);
        let varint_simd_encoded = encode_varint_simd_bulk(&values);
        let vu128_encoded = encode_vu128_bulk(&values);
        let stream_vbyte_encoded = encode_stream_vbyte(&values_u32);

        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(
            BenchmarkId::new("baseline", count),
            &baseline_encoded,
            |b, bytes| {
                b.iter(|| black_box(decode_baseline(black_box(bytes))));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("varint-simd", count),
            &varint_simd_encoded,
            |b, bytes| {
                b.iter(|| black_box(decode_varint_simd_bulk(black_box(bytes))));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("vu128", count),
            &vu128_encoded,
            |b, bytes| {
                b.iter(|| black_box(decode_vu128_bulk(black_box(bytes))));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("stream-vbyte", count),
            &(&stream_vbyte_encoded, count),
            |b, (bytes, cnt)| {
                b.iter(|| black_box(decode_stream_vbyte(black_box(bytes), *cnt)));
            },
        );
    }

    group.finish();
}

fn bench_ngram_key_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ngram_key_simulation");

    // Simulate encoding/decoding n-gram keys (1-5 indices each)
    let num_ngrams = 10000;
    let indices = generate_zipf_indices(num_ngrams * 3); // Average ~3 indices per ngram
    let ngram_sizes = generate_ngram_sizes(num_ngrams);

    // Build ngram index groups
    let mut ngrams: Vec<Vec<u64>> = Vec::with_capacity(num_ngrams);
    let mut idx = 0;
    for &size in &ngram_sizes {
        let end = (idx + size).min(indices.len());
        ngrams.push(indices[idx..end].to_vec());
        idx = end;
    }

    group.throughput(Throughput::Elements(num_ngrams as u64));

    group.bench_function("baseline_encode_ngrams", |b| {
        b.iter(|| {
            let mut buf = Vec::with_capacity(num_ngrams * 6);
            for ngram in &ngrams {
                let start = buf.len();
                for &idx in ngram {
                    encode_varint(idx, &mut buf);
                }
                black_box(buf.len() - start);
            }
            buf.len()
        });
    });

    group.bench_function("vu128_encode_ngrams", |b| {
        b.iter(|| {
            let mut buf = Vec::with_capacity(num_ngrams * 6);
            let mut tmp = [0u8; 9];
            for ngram in &ngrams {
                let start = buf.len();
                for &idx in ngram {
                    let len = vu128::encode_u64(&mut tmp, idx);
                    buf.extend_from_slice(&tmp[..len]);
                }
                black_box(buf.len() - start);
            }
            buf.len()
        });
    });

    // Pre-encode for decode benchmarks
    let mut baseline_encoded: Vec<Vec<u8>> = Vec::with_capacity(num_ngrams);
    let mut vu128_encoded: Vec<Vec<u8>> = Vec::with_capacity(num_ngrams);

    for ngram in &ngrams {
        let mut buf = Vec::new();
        for &idx in ngram {
            encode_varint(idx, &mut buf);
        }
        baseline_encoded.push(buf);

        let mut buf = Vec::new();
        let mut tmp = [0u8; 9];
        for &idx in ngram {
            let len = vu128::encode_u64(&mut tmp, idx);
            buf.extend_from_slice(&tmp[..len]);
        }
        vu128_encoded.push(buf);
    }

    group.bench_function("baseline_decode_ngrams", |b| {
        b.iter(|| {
            let mut total = 0u64;
            for encoded in &baseline_encoded {
                let decoded = decode_baseline(encoded);
                total += decoded.len() as u64;
            }
            black_box(total)
        });
    });

    group.bench_function("vu128_decode_ngrams", |b| {
        b.iter(|| {
            let mut total = 0u64;
            for encoded in &vu128_encoded {
                let decoded = decode_vu128_bulk(encoded);
                total += decoded.len() as u64;
            }
            black_box(total)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_encode,
    bench_single_decode,
    bench_bulk_encode,
    bench_bulk_decode,
    bench_ngram_key_simulation,
);
criterion_main!(benches);
