# N-gram Trie Storage Backends

This document describes the trie-based dictionary backends available for n-gram storage in libgrammstein.

## Overview

libgrammstein stores n-grams in trie structures provided by `liblevenshtein`. The choice of backend affects:

- **Memory usage**: How compactly n-grams are stored
- **Query speed**: Lookup latency for probability queries
- **Mutability**: Whether updates are allowed after construction
- **Thread safety**: Concurrent read/write capabilities

## Available Backends

### DynamicDawgChar (Recommended Default)

A Directed Acyclic Word Graph with UTF-8 support (4-byte nodes).

```rust
use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
use libgrammstein::ngram::{TrainerBuilder, NgramEntry};

let model = TrainerBuilder::new(DynamicDawgChar::new())
    .order(3)
    .train(&corpus)?;
```

**Characteristics**:

| Property | Value |
|----------|-------|
| Node size | 4 bytes (UTF-8) |
| Compression | Good (suffix sharing) |
| Concurrent writes | Yes (atomic) |
| SIMD optimization | Available |
| Bloom filter | Available |

**Best for**:
- General-purpose language models
- Multilingual text (CJK, Arabic, Cyrillic)
- Models that need incremental updates
- Parallel training

### DoubleArrayTrieChar (Production Read-Only)

A Double-Array Trie with UTF-8 support for maximum query speed.

```rust
use liblevenshtein::dictionary::double_array_trie_char::DoubleArrayTrieChar;

// Build from existing data
let trie = DoubleArrayTrieChar::from_iter(ngrams.iter().map(|(k, v)| (k.as_str(), v)));
```

**Characteristics**:

| Property | Value |
|----------|-------|
| Node size | 4 bytes (UTF-8) |
| Compression | Excellent |
| Concurrent writes | No (immutable) |
| Query speed | Fastest |
| Build time | Longer |

**Best for**:
- Production deployment (read-only)
- Maximum query throughput
- Memory-constrained environments
- Static dictionaries

### PathMapDictionary

Lock-free concurrent hash map with path-based keys.

```rust
use liblevenshtein::dictionary::path_map::PathMapDictionary;

let model = TrainerBuilder::new(PathMapDictionary::new())
    .order(3)
    .train(&corpus)?;
```

**Characteristics**:

| Property | Value |
|----------|-------|
| Storage | Hash-based |
| Compression | Poor |
| Concurrent writes | Yes (lock-free) |
| Prefix queries | Limited |
| Random access | Excellent |

**Best for**:
- High write throughput
- When prefix iteration not needed
- Testing and development

## Memory Comparison

For a 10M n-gram model:

| Backend | Memory | Relative |
|---------|--------|----------|
| DynamicDawgChar | ~400 MB | 1.0x |
| DoubleArrayTrieChar | ~250 MB | 0.6x |
| PathMapDictionary | ~800 MB | 2.0x |

## Query Performance

Benchmark results (1M random lookups):

| Backend | Time | Queries/sec |
|---------|------|-------------|
| DoubleArrayTrieChar | 45ms | 22M |
| DynamicDawgChar | 68ms | 15M |
| PathMapDictionary | 120ms | 8M |

## Thread Safety

### DynamicDawgChar

```rust
use std::sync::Arc;
use rayon::prelude::*;

let model = Arc::new(trained_model);

// Safe concurrent reads
let scores: Vec<f64> = queries.par_iter()
    .map(|q| model.log_prob(&q.word, &q.context))
    .collect();
```

### DoubleArrayTrieChar

```rust
// Fully thread-safe for reads (immutable after construction)
let model = Arc::new(model);

// Can be shared freely
handles.iter().for_each(|h| {
    let m = Arc::clone(&model);
    thread::spawn(move || m.log_prob("word", &["context"]));
});
```

## Serialization

All backends support serialization via serde:

```rust
// Save
model.save("model.bin")?;

// Load with specific backend
let model: NgramModel<DynamicDawgChar<NgramEntry>> =
    NgramModel::load("model.bin")?;
```

### Portable Format

For cross-backend compatibility:

```rust
// Save as portable
model.save_portable("model.portable.bin")?;

// Load with any backend
let model = NgramModel::load_portable(
    "model.portable.bin",
    || DoubleArrayTrieChar::new()
)?;
```

## Choosing a Backend

```
                    ┌─────────────────────────────────┐
                    │     Need concurrent writes?     │
                    └───────────────┬─────────────────┘
                                    │
              ┌──────────yes────────┴────────no───────┐
              │                                        │
              ▼                                        ▼
    ┌─────────────────┐                    ┌──────────────────┐
    │  DynamicDawgChar │                    │  Need fastest    │
    │  (or PathMap)    │                    │  queries?        │
    └─────────────────┘                    └────────┬─────────┘
                                                    │
                                     ┌──────yes─────┴─────no──────┐
                                     │                             │
                                     ▼                             ▼
                          ┌──────────────────┐          ┌──────────────────┐
                          │DoubleArrayTrieChar│          │  DynamicDawgChar │
                          └──────────────────┘          └──────────────────┘
```

## Implementation Details

### N-gram Key Encoding

N-grams are stored as space-separated strings:

```
"the" -> unigram
"the quick" -> bigram
"the quick brown" -> trigram
```

### NgramEntry Structure

```rust
pub struct NgramEntry {
    count: u64,                    // Raw occurrence count
    continuation_count: u32,       // For MKN smoothing
    cached_log_prob: Option<f64>,  // Lazily computed
}
```

### Backoff Lookup

```rust
// Query: P(fox | the quick brown)
// Tries: "the quick brown fox" -> "quick brown fox" -> "brown fox" -> "fox"

fn log_prob(&self, word: &str, context: &[&str]) -> f64 {
    for n in (0..=context.len()).rev() {
        let key = build_key(&context[context.len()-n..], word);
        if let Some(entry) = self.dictionary.get(&key) {
            return compute_mkn_prob(entry, n);
        }
    }
    self.unk_log_prob()
}
```

## See Also

- [Query API](query-api.md) - Query methods reference
- [Backend Selection](../../integration/liblevenshtein/backend-selection.md) - Selection guide
- [Threading Model](../../architecture/threading.md) - Concurrency details
