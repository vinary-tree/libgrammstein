# PathMap Synergy

This document describes how libgrammstein integrates with liblevenshtein's PathMap for efficient dictionary operations.

## Overview

PathMap provides lock-free concurrent hash map functionality optimized for string keys:

```rust
use liblevenshtein::dictionary::path_map::PathMapDictionary;

let mut map: PathMapDictionary<u64> = PathMapDictionary::new();

// Lock-free insertion
map.insert("hello", 42);

// O(1) lookup
let value = map.get("hello");
```

## Integration Points

### N-gram Storage

PathMap can be used as alternative n-gram backend:

```rust
use libgrammstein::ngram::TrainerBuilder;
use liblevenshtein::dictionary::path_map::PathMapDictionary;

let model = TrainerBuilder::new(PathMapDictionary::new())
    .order(3)
    .train(&corpus)?;
```

**Characteristics**:
- O(1) average lookup
- Lock-free concurrent writes
- Higher memory than tries
- No prefix iteration

### Embedding Cache

PathMap for word vector caching:

```rust
pub struct SubwordEmbedding {
    word_vectors: PathMapDictionary<Array1<f32>>,
    // ...
}

impl SubwordEmbedding {
    pub fn word_vector(&self, word: &str) -> Array1<f32> {
        if let Some(vec) = self.word_vectors.get(word) {
            return vec.clone();
        }

        // Compute and cache
        let vec = self.compute_vector(word);
        self.word_vectors.insert(word.to_string(), vec.clone());
        vec
    }
}
```

### Dictionary Index

Efficient word → metadata mapping:

```rust
pub struct SpellingDictionary {
    trie: DoubleArrayTrieChar<WordEntry>,
    index: PathMapDictionary<usize>,  // word → trie position
}

impl SpellingDictionary {
    pub fn frequency(&self, word: &str) -> Option<u64> {
        // O(1) lookup via PathMap
        let idx = self.index.get(word)?;
        Some(self.entries[*idx].frequency)
    }
}
```

## Concurrent Operations

### Lock-Free Insertion

```rust
use std::thread;
use std::sync::Arc;

let map = Arc::new(PathMapDictionary::new());

let handles: Vec<_> = (0..4).map(|t| {
    let map = Arc::clone(&map);
    thread::spawn(move || {
        for i in 0..1000 {
            map.insert(format!("word_{}_{}", t, i), i as u64);
        }
    })
}).collect();

for h in handles {
    h.join().unwrap();
}

assert_eq!(map.len(), 4000);
```

### Concurrent Read/Write

```rust
use rayon::prelude::*;

// Safe concurrent access
sentences.par_iter().for_each(|sentence| {
    for word in tokenize(sentence) {
        // Concurrent increment
        map.modify(&word, |v| v.map(|x| x + 1).unwrap_or(1));
    }
});
```

## Performance Characteristics

### Operation Complexity

| Operation | PathMap | DynamicDawgChar | DoubleArrayTrie |
|-----------|---------|-----------------|-----------------|
| Insert | O(1) avg | O(k) | N/A (immutable) |
| Lookup | O(1) avg | O(k) | O(k) |
| Delete | O(1) avg | O(k) | N/A |
| Prefix search | N/A | O(k + m) | O(k + m) |
| Memory | Higher | Medium | Lower |

Where k = key length, m = matches.

### Benchmark Results

For 1M word dictionary:

| Operation | PathMap | DoubleArrayTrie |
|-----------|---------|-----------------|
| Insert (all) | 150ms | N/A |
| Lookup (1M) | 45ms | 42ms |
| Memory | 80MB | 35MB |

### When to Use PathMap

**Prefer PathMap when**:
- Need concurrent writes
- Prefix search not required
- Fast individual lookups needed
- Memory is not constrained

**Prefer Trie when**:
- Read-only after construction
- Need prefix/autocomplete
- Memory is constrained
- Need fuzzy matching with liblevenshtein

## Hybrid Approach

Combine PathMap with Trie for best of both:

```rust
pub struct HybridDictionary {
    /// Static dictionary (DoubleArrayTrie)
    base: DoubleArrayTrieChar<WordEntry>,

    /// Dynamic additions (PathMap)
    additions: PathMapDictionary<WordEntry>,
}

impl HybridDictionary {
    pub fn contains(&self, word: &str) -> bool {
        self.base.contains(word) || self.additions.contains_key(word)
    }

    pub fn add(&self, word: &str, entry: WordEntry) {
        // New words go to PathMap
        self.additions.insert(word.to_string(), entry);
    }

    pub fn compact(&mut self) {
        // Periodically merge into new trie
        let combined = self.base.iter()
            .chain(self.additions.iter())
            .collect::<Vec<_>>();

        self.base = DoubleArrayTrieChar::from_sorted_iter(combined);
        self.additions.clear();
    }
}
```

## N-gram Model Integration

### Training with PathMap

```rust
// PathMap for flexible training
let model = TrainerBuilder::new(PathMapDictionary::new())
    .order(3)
    .train(&corpus)?;

// Convert to DoubleArrayTrie for production
let production_model = model.to_double_array()?;
```

### Incremental Updates

```rust
pub struct IncrementalNgramModel {
    base: NgramModel<DoubleArrayTrieChar<NgramEntry>>,
    updates: NgramModel<PathMapDictionary<NgramEntry>>,
}

impl IncrementalNgramModel {
    pub fn log_prob(&self, word: &str, context: &[&str]) -> f64 {
        // Check updates first (more recent data)
        if let Some(prob) = self.updates.try_log_prob(word, context) {
            return prob;
        }
        self.base.log_prob(word, context)
    }

    pub fn update(&self, sentence: &[&str]) {
        // Add to PathMap-based model
        self.updates.add_sentence(sentence);
    }
}
```

## Memory Management

### Sharding

For very large dictionaries:

```rust
pub struct ShardedPathMap {
    shards: Vec<PathMapDictionary<WordEntry>>,
    num_shards: usize,
}

impl ShardedPathMap {
    fn shard_for(&self, key: &str) -> usize {
        let hash = hash_string(key);
        hash as usize % self.num_shards
    }

    pub fn get(&self, key: &str) -> Option<&WordEntry> {
        self.shards[self.shard_for(key)].get(key)
    }

    pub fn insert(&self, key: String, value: WordEntry) {
        self.shards[self.shard_for(&key)].insert(key, value);
    }
}
```

### Eviction

For caches with size limits:

```rust
pub struct BoundedPathMap {
    map: PathMapDictionary<(WordEntry, Instant)>,
    max_size: usize,
}

impl BoundedPathMap {
    pub fn insert(&self, key: String, value: WordEntry) {
        if self.map.len() >= self.max_size {
            self.evict_oldest();
        }
        self.map.insert(key, (value, Instant::now()));
    }

    fn evict_oldest(&self) {
        // Find and remove oldest entry
        if let Some((oldest, _)) = self.map.iter()
            .min_by_key(|(_, (_, time))| *time)
        {
            self.map.remove(&oldest);
        }
    }
}
```

## Best Practices

1. **Use for dynamic data**: PathMap excels at concurrent updates

2. **Convert for production**: Use DoubleArrayTrie for read-heavy workloads

3. **Combine wisely**: Hybrid approach for best of both worlds

4. **Consider memory**: PathMap uses more memory than tries

5. **Leverage concurrency**: PathMap is optimized for parallel access

## See Also

- [Backend Selection](backend-selection.md) - Choosing the right backend
- [Trie Storage](../../components/ngram/trie-storage.md) - Trie comparison
- [Threading Model](../../architecture/threading.md) - Concurrency patterns
