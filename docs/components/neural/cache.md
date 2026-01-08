# Neural Caching

The neural module provides caching mechanisms for efficient inference and embedding reuse.

## Cache Types

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Caching Architecture                             │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Transformer Inference                          │  │
│  │                                                                   │  │
│  │  ┌─────────────┐     ┌─────────────────┐     ┌────────────────┐  │  │
│  │  │  KvCache    │     │ SlidingWindow   │     │ LayerCache     │  │  │
│  │  │             │     │ Cache           │     │                │  │  │
│  │  │ Multi-layer │     │ Bounded memory  │     │ Per-layer K/V  │  │  │
│  │  │ K/V storage │     │ for long seqs   │     │ tensors        │  │  │
│  │  └─────────────┘     └─────────────────┘     └────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Embedding Layer                                │  │
│  │                                                                   │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │                   EmbeddingCache                             │  │  │
│  │  │                                                              │  │  │
│  │  │  Lock-free DashMap with LRU eviction                        │  │  │
│  │  │  Key: text hash → Value: Arc<Vec<f32>>                      │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## KV Cache

The `KvCache` stores key-value tensors from transformer attention layers for incremental inference.

### Why KV Caching?

Without caching, each new token requires recomputing attention over all previous tokens:

```
Without cache (O(n²) attention per token):
Token 1: Compute attention [1]
Token 2: Compute attention [1, 2]
Token 3: Compute attention [1, 2, 3]
...

With cache (O(n) attention per token):
Token 1: Compute attention [1], cache K₁, V₁
Token 2: Load K₁, V₁ from cache, compute [2], cache K₂, V₂
Token 3: Load K₁₂, V₁₂ from cache, compute [3], cache K₃, V₃
...
```

### Configuration

```rust
use libgrammstein::neural::{CacheConfig, KvCache};
use candle_core::DType;

let config = CacheConfig {
    max_seq_len: 8192,     // Maximum sequence length
    num_layers: 12,         // Number of transformer layers
    num_heads: 12,          // Number of attention heads
    head_dim: 64,           // Dimension per head (768 / 12)
    dtype: DType::F32,      // Data type for tensors
};

let cache = KvCache::new(&config);
```

### Layer Cache Operations

```rust
use libgrammstein::neural::LayerCache;

// Get cache for specific layer
let layer_cache = cache.layer(0);

// Update with new key-value tensors
let new_keys = /* Tensor */;
let new_values = /* Tensor */;
cache.update_layer(0, &new_keys, &new_values)?;

// Get current sequence length in cache
let seq_len = cache.seq_len();

// Clear all cached values
cache.clear();
```

### Pre-allocation

For better performance, pre-allocate cache tensors:

```rust
// Pre-allocate for expected sequence length
cache.preallocate(1024)?;  // Pre-allocate for 1024 tokens
```

## Sliding Window Cache

For very long sequences, the `SlidingWindowCache` maintains bounded memory:

```
Sliding Window (size=1024):

Full sequence: [t₁, t₂, t₃, ..., t₁₀₀₀, t₁₀₀₁, ..., t₂₀₀₀]

Cache contents at t=2000:
[t₁₀₀₁, t₁₀₀₂, ..., t₂₀₀₀]  (most recent 1000 tokens)

Older tokens (t₁ - t₁₀₀₀) are evicted.
```

### Configuration

```rust
use libgrammstein::neural::SlidingWindowCache;

let window_size = 1024;  // Keep last 1024 tokens
let sliding_cache = SlidingWindowCache::new(&config, window_size);
```

### Operations

```rust
// Update with automatic eviction
sliding_cache.update_layer(layer_idx, &keys, &values)?;

// Get current window size
let current_len = sliding_cache.seq_len();

// Get configured window size
let max_window = sliding_cache.window_size();

// Clear cache
sliding_cache.clear();
```

### Memory Comparison

| Cache Type | Memory for 8K tokens | Memory for 32K tokens |
|------------|---------------------|----------------------|
| KvCache | ~800 MB | ~3.2 GB |
| SlidingWindowCache (1K) | ~100 MB | ~100 MB |
| SlidingWindowCache (4K) | ~400 MB | ~400 MB |

## Embedding Cache

The `EmbeddingCache` stores computed embeddings for text reuse.

### Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        EmbeddingCache                                    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    DashMap (lock-free)                          │    │
│  │                                                                 │    │
│  │  Key: u64 (text hash)  →  Value: Arc<Vec<f32>> (embedding)     │    │
│  │                                                                 │    │
│  │  Concurrent reads: ✓  Concurrent writes: ✓  No global lock: ✓  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    LRU Eviction Queue                           │    │
│  │                                                                 │    │
│  │  VecDeque<u64>: [oldest_key, ..., newest_key]                  │    │
│  │                                                                 │    │
│  │  On insert: push to back                                        │    │
│  │  On capacity: pop from front, remove from map                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Configuration

```rust
use libgrammstein::neural::EmbeddingCache;

// Create cache with capacity
let cache = EmbeddingCache::new(10000);  // Cache up to 10k embeddings
```

### Operations

```rust
use std::sync::Arc;

// Get embedding (returns None if not cached)
let text = "Hello, world!";
if let Some(embedding) = cache.get(text) {
    println!("Cache hit! {} dimensions", embedding.len());
}

// Insert embedding
let embedding = vec![0.1, 0.2, 0.3, /* ... */];
cache.insert(text, Arc::new(embedding));

// Check cache state
println!("Cache size: {}", cache.len());
println!("Cache empty: {}", cache.is_empty());

// Clear cache
cache.clear();
```

### Thread Safety

The cache uses `DashMap` for lock-free concurrent access:

```rust
use std::sync::Arc;
use std::thread;

let cache = Arc::new(EmbeddingCache::new(10000));

// Multiple threads can read and write concurrently
let handles: Vec<_> = (0..4).map(|i| {
    let cache = Arc::clone(&cache);
    thread::spawn(move || {
        for j in 0..100 {
            let text = format!("text_{}_{}", i, j);

            // Check cache
            if cache.get(&text).is_none() {
                // Compute and insert
                let embedding = Arc::new(vec![0.0; 768]);
                cache.insert(&text, embedding);
            }
        }
    })
}).collect();

for handle in handles {
    handle.join().unwrap();
}
```

### Zero-Copy Sharing

Embeddings are wrapped in `Arc` for efficient sharing:

```rust
// Insert returns Arc
let embedding = Arc::new(vec![0.1, 0.2, 0.3]);
cache.insert("text", Arc::clone(&embedding));

// Get returns Arc (no copy)
let cached = cache.get("text").unwrap();
assert!(Arc::ptr_eq(&embedding, &cached));  // Same memory
```

## Cache Integration in Embedder

The `ModernBertEmbedder` automatically manages embedding cache:

```rust
use libgrammstein::neural::{ModernBertEmbedder, EmbeddingConfig};

let config = EmbeddingConfig {
    cache_size: 10000,  // Enable caching
    ..Default::default()
};

let embedder = ModernBertEmbedder::new(config)?;

// First call: computes embedding
let emb1 = embedder.embed_query("Hello")?;

// Second call: returns cached embedding
let emb2 = embedder.embed_query("Hello")?;

// Check cache statistics
let (hits, misses) = embedder.cache_stats();
println!("Hits: {}, Misses: {}", hits, misses);  // Hits: 1, Misses: 1

// Clear cache manually
embedder.clear_cache();
```

## Memory Management

### Estimating Cache Memory

```
EmbeddingCache memory ≈ capacity × (embedding_dim × 4 bytes + overhead)

For 10,000 embeddings at 768 dimensions:
Memory ≈ 10,000 × (768 × 4 + 64) ≈ 31 MB

For 100,000 embeddings:
Memory ≈ 100,000 × (768 × 4 + 64) ≈ 310 MB
```

### KV Cache Memory

```
KvCache memory ≈ 2 × num_layers × seq_len × hidden_size × dtype_size

For ModernBERT-base at seq_len=8192, F32:
Memory ≈ 2 × 12 × 8192 × 768 × 4 ≈ 600 MB
```

## Best Practices

### 1. Enable Caching for Repeated Queries

```rust
let config = EmbeddingConfig {
    cache_size: 10000,  // Common queries will be cached
    ..Default::default()
};
```

### 2. Use Sliding Window for Long Sequences

```rust
// For sequences > 4K tokens
let cache = SlidingWindowCache::new(&config, 2048);
```

### 3. Pre-allocate When Sequence Length is Known

```rust
let cache = KvCache::new(&config);
cache.preallocate(expected_seq_len)?;
```

### 4. Clear Cache Periodically in Long-Running Services

```rust
// Clear every N requests to prevent memory bloat
if request_count % 10000 == 0 {
    embedder.clear_cache();
}
```

## See Also

- [Overview](overview.md) - Neural module introduction
- [Model](model.md) - ModernBERT model details
- [Embedder](embedder.md) - Embedding with caching
