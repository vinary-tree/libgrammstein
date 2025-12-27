# liblevenshtein Integration

This document explains how libgrammstein uses [liblevenshtein-rust](https://github.com/f1r3fly-io/liblevenshtein-rust) for efficient N-gram storage and retrieval.

## Overview

libgrammstein stores N-grams in liblevenshtein's trie dictionaries. This provides:

- **Efficient storage**: Shared prefixes stored once
- **Fast lookup**: O(k) where k = key length
- **Prefix matching**: Find all N-grams with common prefix (for backoff)
- **Multiple backends**: Choose based on use case

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            libgrammstein                                     │
│                                                                             │
│   NgramModel<D>                                                             │
│        │                                                                    │
│        │ uses MutableMappedDictionary<Value = NgramEntry>                   │
│        ▼                                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                          liblevenshtein-rust                                 │
│                                                                             │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────────┐   │
│   │ DynamicDawgChar │   │ PathMapDictionary│   │ DoubleArrayTrieChar   │   │
│   │                 │   │                 │   │                       │   │
│   │ - Mutable       │   │ - Distributed   │   │ - Static              │   │
│   │ - Serializable  │   │ - Structural    │   │ - Fastest reads       │   │
│   │ - General use   │   │   sharing       │   │ - Read-only after     │   │
│   │                 │   │ - Copy-on-write │   │   construction        │   │
│   └─────────────────┘   └─────────────────┘   └─────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## MutableMappedDictionary Trait

liblevenshtein provides the `MutableMappedDictionary` trait for key-value storage:

```rust
// From liblevenshtein/src/dictionary/mod.rs

/// Dictionary with mutable value storage
pub trait MutableMappedDictionary<Value>: Dictionary {
    /// Insert a key-value pair
    fn insert_with_value(&mut self, key: &str, value: Value);

    /// Update existing or insert new
    fn update_or_insert<F>(&mut self, key: &str, default: Value, update: F)
    where
        F: FnOnce(&mut Value);

    /// Get value by key
    fn get_value(&self, key: &str) -> Option<&Value>;

    /// Get mutable value by key
    fn get_value_mut(&mut self, key: &str) -> Option<&mut Value>;
}
```

libgrammstein's `NgramEntry` is stored as the value type:

```rust
#[derive(Clone, Debug, Default)]
pub struct NgramEntry {
    pub count: u64,              // Raw corpus count
    pub continuation_count: u32, // Unique preceding contexts
    pub unique_continuations: u32, // Unique following words
}

// Used with:
// NgramModel<DynamicDawgChar<NgramEntry>>
// NgramModel<PathMapDictionary<NgramEntry>>
```

## N-gram Key Encoding

N-grams are encoded as pipe-separated strings:

```
N-gram: ["the", "quick", "brown"]
Key:    "the|quick|brown"

Unigram: ["fox"]
Key:     "fox"

Bigram: ["brown", "fox"]
Key:    "brown|fox"
```

This encoding enables:
- **Prefix matching**: All N-grams starting with "the|quick" share a prefix
- **Compact storage**: Common prefixes stored once in the trie
- **Efficient backoff**: Traverse from longer to shorter contexts

```rust
impl<D: MutableMappedDictionary<Value = NgramEntry>> NgramModel<D> {
    /// Build key for an N-gram
    fn build_key(&self, context: &[&str], word: &str) -> String {
        let mut key = String::with_capacity(
            context.iter().map(|s| s.len() + 1).sum::<usize>() + word.len()
        );

        for (i, token) in context.iter().enumerate() {
            if i > 0 {
                key.push('|');
            }
            key.push_str(token);
        }

        if !context.is_empty() {
            key.push('|');
        }
        key.push_str(word);

        key
    }
}
```

## Dictionary Backends

### DynamicDawgChar (Recommended for Training)

Mutable DAWG (Directed Acyclic Word Graph) supporting updates:

```rust
use liblevenshtein::dictionary::DynamicDawgChar;
use libgrammstein::ngram::NgramEntry;

// Create mutable dictionary
let mut dictionary = DynamicDawgChar::<NgramEntry>::new();

// Insert N-grams during training
dictionary.update_or_insert(
    "the|quick|brown",
    NgramEntry::default(),
    |entry| entry.count += 1,
);

// Query
if let Some(entry) = dictionary.get_value("the|quick|brown") {
    println!("Count: {}", entry.count);
}
```

**Characteristics**:
- Supports incremental updates
- Serializable with serde
- O(k) lookup where k = key length
- Optional SIMD and bloom filter optimizations
- Best for training and general use

### PathMapDictionary (For Distributed Systems)

Persistent dictionary with structural sharing:

```rust
use liblevenshtein::dictionary::PathMapDictionary;
use libgrammstein::ngram::NgramEntry;

// Create PathMap-backed dictionary
let dictionary = PathMapDictionary::<NgramEntry>::new();

// Insert creates new version (copy-on-write)
let dictionary = dictionary.insert("the|quick", NgramEntry { count: 100, .. });

// Structural sharing: old and new versions share unchanged nodes
```

**Characteristics**:
- Copy-on-write semantics
- Structural sharing (memory efficient for snapshots)
- Designed for F1R3FLY.io distributed storage
- No serde support (uses PathMap serialization)
- Best for lling-llang integration with shared lattice infrastructure

### DoubleArrayTrieChar (For Production Inference)

Static, highly optimized trie:

```rust
use liblevenshtein::dictionary::DoubleArrayTrieChar;
use libgrammstein::ngram::NgramEntry;

// Build from existing data (one-time)
let entries: Vec<(String, NgramEntry)> = training_data.into_iter().collect();
let dictionary = DoubleArrayTrieChar::from_pairs(entries);

// Very fast queries
if let Some(entry) = dictionary.get_value("the|quick|brown") {
    println!("Count: {}", entry.count);
}

// No updates after construction
// dictionary.insert(...);  // Would not compile
```

**Characteristics**:
- Fastest read performance (~30ns per lookup)
- Static (read-only after construction)
- Compact memory layout
- Best for production inference with pre-trained models

## Backend Selection Guide

| Use Case | Recommended Backend | Reason |
|----------|---------------------|--------|
| Training | `DynamicDawgChar` | Supports updates, serializable |
| lling-llang integration | `PathMapDictionary` | Structural sharing, distributed |
| Production inference | `DoubleArrayTrieChar` | Fastest reads |
| Model save/load | `DynamicDawgChar` | Serde support |
| Development/testing | `DynamicDawgChar` | Flexible, debuggable |

## Type Aliases

libgrammstein provides convenient type aliases:

```rust
// For training and save/load workflows
pub type SerializableNgramModel = NgramModel<DynamicDawgChar<NgramEntry>>;

// For lling-llang integration
pub type PathMapNgramModel = NgramModel<PathMapDictionary<NgramEntry>>;

// For production inference
pub type StaticNgramModel = NgramModel<DoubleArrayTrieChar<NgramEntry>>;
```

## Training with DynamicDawgChar

```rust
use liblevenshtein::dictionary::DynamicDawgChar;
use libgrammstein::ngram::{TrainerBuilder, NgramEntry};
use libgrammstein::corpus::PlaintextReader;

// Create mutable dictionary
let dictionary = DynamicDawgChar::<NgramEntry>::new();

// Train model
let model = TrainerBuilder::new(dictionary)
    .order(5)
    .min_count(2)
    .train(&PlaintextReader::from_file("corpus.txt")?)?;

// Save trained model
model.save("ngram_model.bin")?;

// Load later
let model: NgramModel<DynamicDawgChar<NgramEntry>> = NgramModel::load("ngram_model.bin")?;
```

## Using PathMapDictionary with lling-llang

```rust
use liblevenshtein::dictionary::PathMapDictionary;
use libgrammstein::ngram::{TrainerBuilder, NgramEntry};

// Create PathMap-backed dictionary
let dictionary = PathMapDictionary::<NgramEntry>::new();

// Train model (creates new versions via copy-on-write)
let model = TrainerBuilder::new(dictionary)
    .order(5)
    .train(&reader)?;

// Use with lling-llang
let lm_layer = LanguageModelLayer::new(Box::new(model), 1.0);
```

## Converting Between Backends

Convert from training backend to production backend:

```rust
impl NgramModel<DynamicDawgChar<NgramEntry>> {
    /// Convert to static DoubleArrayTrie for faster inference
    pub fn to_static(&self) -> NgramModel<DoubleArrayTrieChar<NgramEntry>> {
        // Extract all key-value pairs
        let pairs: Vec<_> = self.dictionary
            .iter_with_values()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();

        // Build static trie
        let static_dict = DoubleArrayTrieChar::from_pairs(pairs);

        NgramModel {
            dictionary: Arc::new(static_dict),
            smoothing: self.smoothing.clone(),
            vocab_size: self.vocab_size,
            ..
        }
    }
}
```

## DictZipper for Backoff

liblevenshtein's `DictZipper` enables efficient prefix navigation for backoff:

```rust
use liblevenshtein::dictionary::DictZipper;

impl<D: MutableMappedDictionary<Value = NgramEntry>> NgramModel<D> {
    /// Navigate from 5-gram to 4-gram to 3-gram efficiently
    fn backoff_lookup(&self, context: &[&str], word: &str) -> Option<&NgramEntry> {
        // Start with full context
        let mut zipper = DictZipper::new(&self.dictionary);

        // Try to descend through context
        for token in context {
            if !zipper.descend_str(token) {
                return None;  // Context not found
            }
            if !zipper.descend('|') {
                return None;
            }
        }

        // Try to reach word
        if zipper.descend_str(word) {
            zipper.value()
        } else {
            None
        }
    }
}
```

## Thread Safety

All liblevenshtein dictionary backends support concurrent access:

| Backend | Thread Safety |
|---------|---------------|
| `DynamicDawgChar` | `Send + Sync` (wrap in `RwLock` for mutation) |
| `PathMapDictionary` | `Send + Sync` (immutable, copy-on-write) |
| `DoubleArrayTrieChar` | `Send + Sync` (immutable) |

libgrammstein wraps dictionaries in `Arc` for shared ownership:

```rust
pub struct NgramModel<D: MutableMappedDictionary<Value = NgramEntry>> {
    dictionary: Arc<D>,  // Shared across threads
    // ...
}
```

## Performance Characteristics

| Operation | DynamicDawgChar | PathMapDictionary | DoubleArrayTrieChar |
|-----------|-----------------|-------------------|---------------------|
| Lookup | ~100ns | ~80ns | ~30ns |
| Insert | ~200ns | ~300ns (COW) | N/A (static) |
| Memory | Moderate | Higher (structural sharing) | Compact |
| Serialization | Serde | PathMap native | Serde |

## Memory Layout

```
DynamicDawgChar<NgramEntry>
├── root: NodeId
├── nodes: Vec<DawgNode>
│   └── DawgNode { edges: SmallVec<(char, NodeId)>, value: Option<NgramEntry> }
└── bloom_filter: Option<BloomFilter>  // Optional optimization

PathMapDictionary<NgramEntry>
├── root: PathMapNodeRef
└── store: Arc<PathMapStore>
    └── Content-addressed storage with structural sharing

DoubleArrayTrieChar<NgramEntry>
├── base: Vec<i32>      // Base array
├── check: Vec<i32>     // Check array
├── values: Vec<Option<NgramEntry>>  // Values at terminal nodes
└── tail: Vec<String>   // Suffix compression
```

## Example: Custom Dictionary Configuration

```rust
use liblevenshtein::dictionary::{DynamicDawgChar, DynamicDawgConfig};

// Configure DAWG with optimizations
let config = DynamicDawgConfig {
    enable_bloom_filter: true,
    bloom_false_positive_rate: 0.01,
    enable_simd: true,
    initial_capacity: 1_000_000,  // Expected N-gram count
};

let dictionary = DynamicDawgChar::<NgramEntry>::with_config(config);

// Train with optimized dictionary
let model = TrainerBuilder::new(dictionary)
    .order(5)
    .train(&reader)?;
```

## Next Steps

- [Backend Selection](backend-selection.md): Detailed comparison
- [N-gram Overview](../../components/ngram/overview.md): N-gram model details
- [Architecture Overview](../../architecture/overview.md): System design
- [lling-llang Integration](../lling-llang/overview.md): WFST integration
