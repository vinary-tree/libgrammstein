# Threading Model

This document describes the concurrency and thread-safety model used in libgrammstein.

## Overview

libgrammstein is designed for concurrent access:

- **Training**: Parallel corpus processing with Rayon
- **Queries**: Thread-safe model queries
- **Caching**: Lock-free concurrent caches

## Thread Safety Guarantees

### Models

| Type | Thread-Safe | Notes |
|------|-------------|-------|
| `NgramModel<D>` | Yes (if D is) | Queries are safe |
| `SubwordEmbedding` | Yes | Cache uses DashMap |
| `HybridLanguageModel<D>` | Yes | Uses Mutex for LRU cache |

### Dictionary Backends

| Backend | Thread-Safe | Concurrent Writes |
|---------|-------------|-------------------|
| `DynamicDawgChar` | Yes | Yes (atomic) |
| `PathMapDictionary` | Yes | Yes (lock-free) |
| `DoubleArrayTrieChar` | Yes | No (immutable) |

## Training Parallelism

### N-gram Training

```rust
// Training uses Rayon for parallel processing
sentences.par_chunks(batch_size).for_each(|batch| {
    for sentence in batch {
        // Tokenize
        let tokens = tokenizer.words(sentence);

        // Extract and count n-grams
        for ngram in extract_ngrams(&tokens, order) {
            // Atomic insertion into dictionary
            dictionary.insert_or_increment(&ngram);
        }
    }
});
```

The dictionary backend must support concurrent writes:

```rust
// DynamicDawgChar uses atomic operations
impl<V> MutableMappedDictionary for DynamicDawgChar<V>
where
    V: Default + Clone + Send + Sync,
{
    fn insert_with_value(&self, key: &str, value: V) {
        // Uses atomic CAS for lock-free insertion
        self.root.insert_atomic(key, value);
    }
}
```

### Embedding Training

Embedding training is currently sequential within epochs but can process sentences in parallel:

```rust
// Future: Parallel skip-gram updates with gradient accumulation
sentences.par_iter().for_each(|sentence| {
    let local_gradients = compute_gradients(sentence);
    // Accumulate gradients atomically
    global_gradients.add_atomic(&local_gradients);
});

// Apply accumulated gradients
model.apply_gradients(&global_gradients);
```

## Query Concurrency

### N-gram Queries

N-gram queries are read-only and fully thread-safe:

```rust
use std::thread;
use std::sync::Arc;

let model = Arc::new(trained_model);

let handles: Vec<_> = (0..4).map(|_| {
    let model = Arc::clone(&model);
    thread::spawn(move || {
        // Safe concurrent queries
        model.log_prob("fox", &["quick", "brown"])
    })
}).collect();

for handle in handles {
    let _ = handle.join();
}
```

### Embedding Queries

Embedding queries use a thread-safe cache:

```rust
// DashMap provides lock-free concurrent access
pub struct SubwordEmbedding {
    // ...
    cache: Arc<DashMap<String, Array1<f32>>>,
}

impl SubwordEmbedding {
    pub fn word_vector(&self, word: &str) -> Array1<f32> {
        // Check cache (lock-free read)
        if let Some(cached) = self.cache.get(word) {
            return cached.clone();
        }

        // Compute vector
        let vector = self.compute_vector(word);

        // Store in cache (lock-free write)
        if self.cache.len() < self.max_cache_size {
            self.cache.insert(word.to_string(), vector.clone());
        }

        vector
    }
}
```

### Hybrid Queries

Hybrid model uses a Mutex-protected LRU cache:

```rust
pub struct HybridLanguageModel<D> {
    // ...
    cache: Mutex<LruCache<CacheKey, f64>>,
}

impl<D> HybridLanguageModel<D> {
    pub fn score(&self, word: &str, context: &[&str]) -> f64 {
        let cache_key = CacheKey { word, context };

        // Check cache (acquires lock briefly)
        if let Ok(mut cache) = self.cache.lock() {
            if let Some(&score) = cache.get(&cache_key) {
                return score;
            }
        }

        // Compute score (no lock held)
        let score = self.compute_score(word, context);

        // Update cache (acquires lock briefly)
        if let Ok(mut cache) = self.cache.lock() {
            cache.put(cache_key, score);
        }

        score
    }
}
```

## Rayon Thread Pool

### Configuration

```rust
// Configure global thread pool
rayon::ThreadPoolBuilder::new()
    .num_threads(16)
    .thread_name(|i| format!("grammstein-worker-{}", i))
    .build_global()
    .expect("Failed to build thread pool");
```

### Best Practices

1. **Batch Size**: Larger batches reduce synchronization overhead
2. **Work Stealing**: Rayon automatically balances load
3. **Avoid Contention**: Use thread-local accumulators when possible

```rust
// Good: Thread-local accumulation
let total: u64 = sentences.par_iter()
    .map(|s| count_ngrams(s))  // Thread-local
    .sum();                     // Reduce at end

// Avoid: Frequent atomic updates
sentences.par_iter().for_each(|s| {
    global_counter.fetch_add(1, Ordering::Relaxed);  // Contention
});
```

## Atomic Operations

### NgramEntry Updates

```rust
#[derive(Default)]
pub struct NgramEntry {
    count: AtomicU64,
    continuation_count: AtomicU32,
}

impl NgramEntry {
    pub fn increment(&self) {
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }
}
```

### Memory Ordering

| Ordering | Use Case |
|----------|----------|
| `Relaxed` | Counters (order doesn't matter) |
| `Acquire/Release` | Initialization checks |
| `SeqCst` | Rarely needed |

## Send and Sync

### Model Bounds

```rust
// NgramModel is Send + Sync if D is
impl<D> Send for NgramModel<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + Send,
{}

impl<D> Sync for NgramModel<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + Sync,
{}
```

### Usage with Async

```rust
use tokio::task;

// Models can be shared across async tasks
let model = Arc::new(trained_model);

let tasks: Vec<_> = queries.iter().map(|q| {
    let model = Arc::clone(&model);
    task::spawn_blocking(move || {
        model.log_prob(&q.word, &q.context)
    })
}).collect();

let results = futures::future::join_all(tasks).await;
```

## Performance Considerations

### Cache Sizing

| Workload | Recommended Cache Size |
|----------|----------------------|
| Interactive REPL | 10,000 |
| Batch scoring | 50,000 |
| Web server | 100,000+ |

### Lock Contention

Monitor lock contention for hybrid model cache:

```rust
// If contention is high, consider:
// 1. Increase cache size
// 2. Use sharded LRU
// 3. Remove caching for batch workloads

hybrid.clear_cache();  // For batch processing
```

### Thread Pool Sizing

```rust
// Rule of thumb
let num_threads = num_cpus::get();

// For I/O bound (HTTP streaming)
let num_threads = num_cpus::get() * 2;

// For pure CPU bound (inference)
let num_threads = num_cpus::get();
```

## Thread-Local Storage

For accumulating statistics during training:

```rust
use std::cell::RefCell;

thread_local! {
    static LOCAL_COUNTS: RefCell<HashMap<String, u64>> = RefCell::new(HashMap::new());
}

// Accumulate locally
LOCAL_COUNTS.with(|counts| {
    *counts.borrow_mut().entry(key).or_insert(0) += 1;
});

// Merge at end of parallel section
let global_counts = merge_thread_local_counts();
```

## Debugging Concurrency Issues

### Enable Thread Sanitizer

```bash
RUSTFLAGS="-Z sanitizer=thread" cargo test --target x86_64-unknown-linux-gnu
```

### Common Issues

1. **Data Race**: Use atomic types or locks
2. **Deadlock**: Avoid nested locks, use try_lock
3. **Memory Ordering**: Prefer Relaxed for counters

## See Also

- [Data Flow](data-flow.md) - How data moves through the system
- [Training Guide](../training/ngram.md) - Training with parallelism
- [Large Corpora](../training/large-corpora.md) - Scaling training
