# Backend Selection Guide

This document provides guidance on selecting the appropriate liblevenshtein dictionary backend for libgrammstein language models.

## Available Backends

| Backend | Node Size | Concurrent Writes | Best For |
|---------|-----------|-------------------|----------|
| `DynamicDawgChar` | 4 bytes (UTF-8) | Yes (atomic) | General purpose |
| `DoubleArrayTrieChar` | 4 bytes (UTF-8) | No (immutable) | Production read-only |
| `PathMapDictionary` | N/A (hash) | Yes (lock-free) | Fast updates |
| `PersistentARTrieChar` | 4 bytes (UTF-8) | Yes (atomic) | Dictionary extraction |

## Decision Flowchart

```
                    ┌────────────────────────────────┐
                    │   Need concurrent writes?      │
                    └───────────────┬────────────────┘
                                    │
            ┌───────── yes ─────────┴──────── no ───────────┐
            │                                                │
            ▼                                                ▼
   ┌────────────────────┐                      ┌────────────────────┐
   │  Need prefix/       │                      │   Read-only        │
   │  fuzzy search?      │                      │   after build?     │
   └─────────┬──────────┘                      └─────────┬──────────┘
             │                                            │
    ┌── yes ─┴── no ──┐                        ┌── yes ──┴── no ──┐
    │                  │                        │                   │
    ▼                  ▼                        ▼                   ▼
┌──────────┐    ┌──────────┐           ┌──────────────┐    ┌──────────────┐
│ Dynamic  │    │ PathMap  │           │ DoubleArray  │    │ Dynamic      │
│ DawgChar │    │ Dict     │           │ TrieChar     │    │ DawgChar     │
└──────────┘    └──────────┘           └──────────────┘    └──────────────┘
```

## Detailed Comparison

### DynamicDawgChar

**Description**: Directed Acyclic Word Graph with UTF-8 support

```rust
use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
use libgrammstein::ngram::TrainerBuilder;

let model = TrainerBuilder::new(DynamicDawgChar::new())
    .order(3)
    .train(&corpus)?;
```

**Pros**:
- Good compression (suffix sharing)
- Supports concurrent updates (atomic)
- SIMD optimizations available
- Bloom filter acceleration
- Prefix and fuzzy search

**Cons**:
- Slower queries than DoubleArrayTrie
- More complex construction

**Best for**:
- General-purpose language models
- Models requiring incremental updates
- Multilingual text (CJK, Arabic, etc.)

### DoubleArrayTrieChar

**Description**: Double-Array Trie optimized for fast lookup

```rust
use liblevenshtein::dictionary::double_array_trie_char::DoubleArrayTrieChar;

// Build from sorted data
let trie = DoubleArrayTrieChar::from_sorted_iter(
    ngrams.iter().map(|(k, v)| (k.as_str(), v.clone()))
);
```

**Pros**:
- Fastest lookup performance
- Most compact memory representation
- Cache-friendly traversal
- Excellent for production deployment

**Cons**:
- Immutable after construction
- Slower to build
- Must build from sorted input

**Best for**:
- Production models (read-only)
- Memory-constrained environments
- High-throughput inference

### PathMapDictionary

**Description**: Lock-free concurrent hash map

```rust
use liblevenshtein::dictionary::path_map::PathMapDictionary;

let model = TrainerBuilder::new(PathMapDictionary::new())
    .order(3)
    .train(&corpus)?;
```

**Pros**:
- O(1) average lookup
- Lock-free concurrent writes
- Simple semantics
- Fast individual operations

**Cons**:
- No prefix search
- No fuzzy matching
- Higher memory usage
- No compression

**Best for**:
- Caching
- High-write scenarios
- When prefix search not needed

### PersistentARTrieChar

**Description**: Atomic Reference Trie for concurrent updates

```rust
use liblevenshtein::dictionary::persistent_ar_trie_char::PersistentARTrieChar;

let extractor = WordExtractor::new(PersistentARTrieChar::new());
```

**Pros**:
- Concurrent atomic updates
- Prefix search support
- Good for incremental building
- Persistent (immutable nodes)

**Cons**:
- Higher memory than DAWG
- Slower queries than DoubleArray

**Best for**:
- Dictionary extraction (word counting)
- Incremental model building
- Parallel training

## Use Case Recommendations

### Language Model Training

```rust
// Recommended: DynamicDawgChar
let model = TrainerBuilder::new(DynamicDawgChar::new())
    .order(3)
    .train(&corpus)?;
```

Rationale: Good balance of compression, concurrent writes, and query speed.

### Production Deployment

```rust
// Recommended: DoubleArrayTrieChar
let model: NgramModel<DoubleArrayTrieChar<_>> = model.to_double_array()?;

// Or load directly
let model = NgramModel::load_with_backend(
    "model.bin",
    || DoubleArrayTrieChar::new()
)?;
```

Rationale: Fastest queries, smallest memory footprint.

### Dictionary Extraction

```rust
// Recommended: PersistentARTrieChar
let extractor = WordExtractor::new(PersistentARTrieChar::new());

sentences.par_iter().for_each(|s| {
    extractor.process_sentence(s);  // Concurrent safe
});
```

Rationale: Optimized for concurrent incremental updates.

### Embedding Cache

```rust
// Recommended: PathMapDictionary
let cache: PathMapDictionary<Array1<f32>> = PathMapDictionary::new();
```

Rationale: O(1) lookup, concurrent updates, no prefix search needed.

### Hybrid Training

```rust
// Train with DynamicDawgChar
let model = TrainerBuilder::new(DynamicDawgChar::new())
    .order(3)
    .train(&corpus)?;

// Convert for production
let production = model.to_backend(|| DoubleArrayTrieChar::new())?;
```

## Performance Benchmarks

### Query Latency (1M lookups)

| Backend | Time | Queries/sec |
|---------|------|-------------|
| DoubleArrayTrieChar | 42ms | 24M |
| DynamicDawgChar | 68ms | 15M |
| PathMapDictionary | 55ms | 18M |

### Memory Usage (1M entries)

| Backend | Memory |
|---------|--------|
| DoubleArrayTrieChar | 35 MB |
| DynamicDawgChar | 55 MB |
| PathMapDictionary | 95 MB |

### Concurrent Write Throughput

| Backend | Writes/sec (4 threads) |
|---------|------------------------|
| DynamicDawgChar | 2.1M |
| PathMapDictionary | 3.8M |
| PersistentARTrieChar | 1.8M |

## Feature Matrix

| Feature | DynamicDawg | DoubleArray | PathMap | PersistentAR |
|---------|-------------|-------------|---------|--------------|
| UTF-8 | ✓ | ✓ | ✓ | ✓ |
| Concurrent write | ✓ | ✗ | ✓ | ✓ |
| Prefix search | ✓ | ✓ | ✗ | ✓ |
| Fuzzy search | ✓ | ✓ | ✗ | ✓ |
| Compression | Good | Best | None | Medium |
| Build speed | Medium | Slow | Fast | Medium |
| Query speed | Good | Best | Good | Medium |

## Migration Guide

### From DynamicDawg to DoubleArray

```rust
// Load DynamicDawg model
let dawg_model: NgramModel<DynamicDawgChar<_>> = NgramModel::load("model.bin")?;

// Convert to DoubleArray
let da_model = dawg_model.to_backend(|| DoubleArrayTrieChar::new())?;

// Save with new backend
da_model.save("model_da.bin")?;
```

### Portable Format

For cross-backend compatibility:

```rust
// Save as portable (backend-agnostic)
model.save_portable("model.portable.bin")?;

// Load with any backend
let model = NgramModel::load_portable(
    "model.portable.bin",
    || DoubleArrayTrieChar::new()  // or any backend
)?;
```

## See Also

- [PathMap Synergy](pathmap-synergy.md) - PathMap integration details
- [Trie Storage](../../components/ngram/trie-storage.md) - Storage comparison
- [Threading Model](../../architecture/threading.md) - Concurrency patterns
