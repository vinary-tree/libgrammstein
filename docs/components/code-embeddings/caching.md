# Embedding Cache

libgrammstein provides a built-in thread-safe cache for code embeddings to avoid redundant computation.

## Why Cache Embeddings?

Embedding computation is expensive:
- Model inference takes 20-50ms per code snippet
- Many workflows re-embed the same code repeatedly
- Caching provides instant retrieval for repeated lookups

## Cache Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Embedding Cache                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Request: embed_code("fn main() {}", Rust)                             │
│       │                                                                  │
│       ▼                                                                  │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                     Cache Lookup                                 │   │
│   │                                                                  │   │
│   │   key = hash(code + language)                                   │   │
│   │         = hash("fn main() {}" + "rust")                         │   │
│   │         = 0x7a3b2c1d...                                         │   │
│   │                                                                  │   │
│   │   ┌─────────────────────────────────────┐                       │   │
│   │   │        DashMap<u64, Arc<[f32]>>     │                       │   │
│   │   │   ┌───────────────────────────────┐ │                       │   │
│   │   │   │ 0x7a3b... → [0.1, 0.2, ...]  │ │  ← HIT: Return        │   │
│   │   │   │ 0x3c4d... → [0.5, 0.3, ...]  │ │                       │   │
│   │   │   │ 0x9e8f... → [0.2, 0.7, ...]  │ │                       │   │
│   │   │   └───────────────────────────────┘ │                       │   │
│   │   └─────────────────────────────────────┘                       │   │
│   │                                                                  │   │
│   │   MISS: Run inference → Store → Return                          │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Configuration

### CodeEmbeddingCacheConfig

```rust
pub struct CodeEmbeddingCacheConfig {
    /// Maximum number of embeddings to cache.
    pub max_entries: usize,

    /// Whether to hash code for cache keys (saves memory for long code).
    pub hash_keys: bool,
}

impl Default for CodeEmbeddingCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10000,
            hash_keys: true,
        }
    }
}
```

### Enabling Cache

Caching is enabled by default for all embedders:

```rust
use libgrammstein::neural::code::{CodeT5Embedder, CodeT5Config, CodeEmbeddingCacheConfig};

// Default: caching enabled with 10,000 entries
let embedder = CodeT5Embedder::from_directory("/path/to/model")?;

// Custom cache size
let config = CodeT5Config {
    cache_config: Some(CodeEmbeddingCacheConfig {
        max_entries: 50000,  // Cache more embeddings
        hash_keys: true,
    }),
    ..CodeT5Config::codet5p_110m_embedding("/path/to/model")
};
let embedder = CodeT5Embedder::load(config)?;
```

### Disabling Cache

For memory-constrained environments or when code never repeats:

```rust
let config = CodeT5Config {
    cache_config: None,  // Disable caching
    ..Default::default()
};
```

## Cache Operations

### Checking Cache Status

```rust
// Get number of cached embeddings
if let Some(size) = embedder.cache_stats() {
    println!("Cached embeddings: {}", size);
}
```

### Clearing Cache

```rust
// Clear all cached embeddings
embedder.clear_cache();
```

### Cache Hit Demonstration

```rust
use std::time::Instant;

let code = "fn factorial(n: u32) -> u32 { if n <= 1 { 1 } else { n * factorial(n - 1) } }";

// First call: cache miss, runs inference
let start = Instant::now();
let _ = embedder.embed_code(code, CodeLanguage::Rust)?;
let first_time = start.elapsed();
println!("First call: {:?}", first_time);  // ~25ms

// Second call: cache hit, instant
let start = Instant::now();
let _ = embedder.embed_code(code, CodeLanguage::Rust)?;
let second_time = start.elapsed();
println!("Second call: {:?}", second_time);  // ~10µs

println!("Speedup: {:.0}x", first_time.as_nanos() as f64 / second_time.as_nanos() as f64);
```

## Memory Management

### Memory Usage

| Entries | Dimension | Approximate Memory |
|---------|-----------|-------------------|
| 10,000 | 256 | ~10 MB |
| 10,000 | 768 | ~30 MB |
| 50,000 | 768 | ~150 MB |
| 100,000 | 768 | ~300 MB |

Calculation: `entries × dimension × 4 bytes × overhead`

### Eviction Policy

When the cache reaches capacity, a random entry is evicted:

```rust
// Internal eviction logic
if self.cache.len() >= self.config.max_entries {
    // Remove a random entry (simple, constant-time eviction)
    if let Some(entry) = self.cache.iter().next() {
        let key = *entry.key();
        drop(entry);
        self.cache.remove(&key);
    }
}
```

This simple policy provides:
- O(1) eviction time
- No LRU overhead
- Good enough for most workloads

## Key Hashing

The cache key combines code content and language:

```rust
fn compute_key(&self, code: &str, language: CodeLanguage) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = gxhash::GxHasher::default();  // Fast non-cryptographic hash
    code.hash(&mut hasher);
    language.hash(&mut hasher);
    hasher.finish()
}
```

**Why hash?**
- Code can be kilobytes long
- Hash is constant 8 bytes
- Fast comparison

**Collision probability:**
- 64-bit hash → 2^64 possible values
- For 100,000 entries: ~0.0000003% collision chance
- Practically negligible

## Thread Safety

The cache uses `DashMap` for lock-free concurrent access:

```rust
use std::sync::Arc;
use std::thread;

let embedder = Arc::new(CodeT5Embedder::from_directory("/path")?);

// Multiple threads can read/write cache safely
let handles: Vec<_> = (0..8).map(|i| {
    let emb = Arc::clone(&embedder);
    thread::spawn(move || {
        for j in 0..100 {
            let code = format!("def func_{}_{}: pass", i, j);
            let _ = emb.embed_code(&code, CodeLanguage::Python);
        }
    })
}).collect();

for handle in handles {
    handle.join().unwrap();
}

// Check total cached
println!("Final cache size: {:?}", embedder.cache_stats());
```

## Shared Cache

Multiple embedders can share a single cache:

```rust
pub struct SharedCodeEmbeddingCache {
    cache: CodeEmbeddingCache,
}

impl SharedCodeEmbeddingCache {
    pub fn new(config: CodeEmbeddingCacheConfig) -> Self {
        Self {
            cache: CodeEmbeddingCache::new(config),
        }
    }

    pub fn get_or_compute<E: CodeEmbedder>(
        &self,
        embedder: &E,
        code: &str,
        language: CodeLanguage,
    ) -> Result<Vec<f32>> {
        // Check cache
        if let Some(embedding) = self.cache.get(code, language) {
            return Ok(embedding.to_vec());
        }

        // Compute and cache
        let embedding = embedder.embed_code(code, language)?;
        self.cache.insert(code, language, embedding.clone());
        Ok(embedding)
    }
}
```

## CodeEmbeddingCache API

```rust
pub struct CodeEmbeddingCache {
    cache: dashmap::DashMap<u64, Arc<[f32]>>,
    config: CodeEmbeddingCacheConfig,
}

impl CodeEmbeddingCache {
    /// Create a new cache with the given configuration.
    pub fn new(config: CodeEmbeddingCacheConfig) -> Self;

    /// Get an embedding from the cache.
    pub fn get(&self, code: &str, language: CodeLanguage) -> Option<Arc<[f32]>>;

    /// Insert an embedding into the cache.
    pub fn insert(&self, code: &str, language: CodeLanguage, embedding: Vec<f32>);

    /// Clear the cache.
    pub fn clear(&self);

    /// Get the number of cached embeddings.
    pub fn len(&self) -> usize;

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool;
}
```

## Best Practices

### 1. Size Cache Appropriately

```rust
// For code search index (embed once, query many times)
let config = CodeEmbeddingCacheConfig {
    max_entries: 100000,  // Large cache
    hash_keys: true,
};

// For interactive use (variety of inputs)
let config = CodeEmbeddingCacheConfig {
    max_entries: 5000,  // Moderate cache
    hash_keys: true,
};

// For streaming/batch processing (each code seen once)
let config: Option<CodeEmbeddingCacheConfig> = None;  // Disable
```

### 2. Warm Up Cache

Pre-populate cache for known code:

```rust
fn warm_up_cache(embedder: &dyn CodeEmbedder, codes: &[&str]) -> Result<()> {
    for code in codes {
        embedder.embed_code(code, CodeLanguage::Unknown)?;
    }
    Ok(())
}
```

### 3. Monitor Cache Efficiency

```rust
struct CacheMetrics {
    hits: AtomicU64,
    misses: AtomicU64,
}

impl CacheMetrics {
    fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }
}
```

### 4. Consider Language in Cache Key

The cache already considers language:

```rust
// These are different cache entries
embedder.embed_code("print(x)", CodeLanguage::Python)?;
embedder.embed_code("print(x)", CodeLanguage::JavaScript)?;
```

## See Also

- [Overview](overview.md) - Code embeddings introduction
- [CodeT5+](codet5.md) - Model with caching
- [UniXcoder](unixcoder.md) - Model with caching
- [Ensemble](ensemble.md) - Multi-model ensembles
