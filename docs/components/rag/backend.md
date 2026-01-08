# Retrieval Backends

The RAG module supports pluggable retrieval backends for different scale and performance requirements.

## Backend Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    RetrievalBackend Trait                                │
│                                                                          │
│  pub trait RetrievalBackend {                                           │
│      fn add(&mut self, id: DocumentId, embedding: &[f32]) -> Result<()>;│
│      fn query(&self, embedding: &[f32], top_k: usize)                   │
│          -> Vec<(DocumentId, f32)>;                                     │
│      fn len(&self) -> usize;                                            │
│      fn embedding_dim(&self) -> usize;                                  │
│      fn save(&self, path: &Path) -> Result<()>;                         │
│      fn load(path: &Path, dim: usize) -> Result<Self>;                  │
│      fn clear(&mut self);                                               │
│      fn contains(&self, id: DocumentId) -> bool;                        │
│      fn remove(&mut self, id: DocumentId) -> Result<bool>;              │
│  }                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
                    │                              │
                    ▼                              ▼
     ┌──────────────────────────┐    ┌──────────────────────────┐
     │   ExactCosineBackend     │    │     HnswBackend          │
     │                          │    │                          │
     │   • Dense matrix storage │    │   • HNSW graph structure │
     │   • BLAS acceleration    │    │   • Approximate search   │
     │   • Exact results        │    │   • Sublinear query time │
     │   • O(n) query          │    │   • O(log n) query       │
     │   • Best < 1M docs      │    │   • Best > 1M docs       │
     └──────────────────────────┘    └──────────────────────────┘
```

## Backend Selection

| Documents | Recommended Backend | Query Time | Recall |
|-----------|-------------------|------------|--------|
| < 100K | `ExactCosineBackend` | ~10ms | 100% |
| 100K - 1M | `ExactCosineBackend` | ~100ms | 100% |
| > 1M | `HnswBackend` | ~1ms | ~95% |

```rust
use libgrammstein::rag::BackendType;

// Choose based on expected size
let backend_type = if estimated_docs > 1_000_000 {
    BackendType::Hnsw
} else {
    BackendType::ExactCosine  // Default
};
```

## ExactCosineBackend

BLAS-accelerated dense retrieval with exact results.

### How It Works

```
Embeddings stored as matrix:
┌─────────────────────────────────────────────┐
│          Doc 0: [e₀₀, e₀₁, ..., e₀₇₆₇]      │
│          Doc 1: [e₁₀, e₁₁, ..., e₁₇₆₇]      │
│          Doc 2: [e₂₀, e₂₁, ..., e₂₇₆₇]      │
│          ...                                 │
│          Doc n: [eₙ₀, eₙ₁, ..., eₙ₇₆₇]      │
└─────────────────────────────────────────────┘

Query: q = [q₀, q₁, ..., q₇₆₇]

Scores = Embeddings @ q  (matrix-vector multiply)
       = [score₀, score₁, ..., scoreₙ]

(Pre-normalized embeddings: dot product = cosine similarity)
```

### Configuration

```rust
use libgrammstein::rag::ExactCosineBackend;

// Create with embedding dimension
let backend = ExactCosineBackend::new(768);

// Create with pre-allocated capacity
let backend = ExactCosineBackend::with_capacity(768, 100_000);
```

### Operations

```rust
use libgrammstein::rag::{DocumentId, RetrievalBackend};

let mut backend = ExactCosineBackend::new(768);

// Add document embedding
backend.add(DocumentId::new(0), &embedding)?;

// Query for similar documents
let results = backend.query(&query_embedding, 10);
for (doc_id, score) in results {
    println!("Doc {}: {:.4}", doc_id.0, score);
}

// Check status
println!("Documents: {}", backend.len());
println!("Embedding dim: {}", backend.embedding_dim());

// Remove document
backend.remove(DocumentId::new(0))?;

// Clear all
backend.clear();
```

### Persistence

```rust
use std::path::Path;

// Save to directory
backend.save(Path::new("./index/backend"))?;

// Load from directory
let loaded = ExactCosineBackend::load(Path::new("./index/backend"), 768)?;
```

File structure:
```
backend/
├── embeddings.bin   # Matrix data (n × 768 × f32)
└── doc_ids.bin      # Document ID mapping
```

### Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Add | O(1) amortized | Matrix resize occasionally |
| Query | O(n × d) | n=docs, d=768 |
| Remove | O(n) | Rebuilds matrix |
| Contains | O(n) | Linear scan |

### Memory Usage

```
Memory ≈ n × d × 4 bytes

For 1M documents at 768 dimensions:
Memory ≈ 1,000,000 × 768 × 4 = 3 GB
```

## HnswBackend

Hierarchical Navigable Small World graphs for approximate nearest neighbor search.

### How It Works

```
HNSW Graph Structure:
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer 2 (sparse):    A ────────────────── B                             │
│                      │                    │                             │
│ Layer 1 (medium):    A ─── C ─── D ───── B ─── E                       │
│                      │    /│     │\      │    /                        │
│ Layer 0 (dense):     A ─ C ─ F ─ D ─ G ─ B ─ E ─ H                     │
│                      │  / \   / \   / \  │ / \ │                       │
│                      ... more nodes ...                                 │
└─────────────────────────────────────────────────────────────────────────┘

Query traversal:
1. Start at entry point in top layer
2. Greedily move to closer neighbors
3. Descend to next layer when stuck
4. Return k nearest from bottom layer
```

### Configuration

```rust
use libgrammstein::rag::{HnswBackend, HnswConfig};

let config = HnswConfig {
    // Neighbors to consider during construction (higher = better quality)
    ef_construction: 200,

    // Max neighbors per node (higher = better quality, more memory)
    m: 16,

    // Neighbors to check during search (higher = better recall, slower)
    ef_search: 100,
};

let backend = HnswBackend::new(768, config);
```

### Configuration Guidelines

| Use Case | ef_construction | m | ef_search |
|----------|-----------------|---|-----------|
| Speed priority | 100 | 8 | 50 |
| Balanced | 200 | 16 | 100 |
| Quality priority | 400 | 32 | 200 |

### Lazy Building

The HNSW index builds lazily on first query:

```rust
let mut backend = HnswBackend::new(768, config);

// Add documents (stored as pending)
for (id, emb) in documents {
    backend.add(id, &emb)?;  // Fast, no index built yet
}

// First query triggers index build
let results = backend.query(&query, 10);  // Index built here
```

### Automatic Rebuilds

The index rebuilds periodically during ingestion:

```rust
// Rebuilds every 10,000 additions
for (i, (id, emb)) in documents.iter().enumerate() {
    backend.add(*id, emb)?;
    // Automatic rebuild at 10k, 20k, 30k, ...
}
```

### Thread Safety

```rust
use std::sync::Arc;
use std::thread;

// HnswBackend uses interior mutability (RwLock)
let backend = Arc::new(HnswBackend::new(768, config));

// Multiple readers can query concurrently
let handles: Vec<_> = queries.iter().map(|q| {
    let backend = Arc::clone(&backend);
    let q = q.clone();
    thread::spawn(move || backend.query(&q, 10))
}).collect();
```

### Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Add | O(log n) amortized | Pending until build |
| Query | O(log n) | After index built |
| Build | O(n log n) | Triggered automatically |
| Remove | O(n) | Requires rebuild |

### Memory Usage

```
Memory ≈ n × (d × 4 + M × 4 × avg_layers) bytes

For 10M documents at 768 dimensions, M=16:
Memory ≈ 10M × (768 × 4 + 16 × 4 × 1.5) ≈ 31 GB
```

## Utility Functions

The backend module provides vector utilities:

```rust
use libgrammstein::rag::backend::{normalize_embedding, dot_product, cosine_similarity};

// Normalize to unit length
let normalized = normalize_embedding(&embedding);
// ||normalized|| = 1.0

// Dot product
let dp = dot_product(&a, &b);

// Cosine similarity
let sim = cosine_similarity(&a, &b);  // -1.0 to 1.0
```

## Switching Backends

The `RagIndex` is generic over the backend type:

```rust
use libgrammstein::rag::{RagIndex, RagIndexConfig, ExactCosineBackend, HnswBackend};

// Exact backend (default)
let index_exact: RagIndex<ExactCosineBackend> = RagIndex::new(config.clone());

// HNSW backend
let index_hnsw: RagIndex<HnswBackend> = RagIndex::with_backend(
    config,
    HnswBackend::new(768, HnswConfig::default()),
);
```

## Error Handling

```rust
use libgrammstein::rag::RagError;

match backend.add(id, &embedding) {
    Ok(()) => println!("Added successfully"),
    Err(RagError::IndexError(msg)) => {
        // Dimension mismatch, capacity exceeded, etc.
        eprintln!("Index error: {}", msg);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## Best Practices

### 1. Pre-allocate for Known Size

```rust
// If you know approximate document count
let backend = ExactCosineBackend::with_capacity(768, 500_000);
```

### 2. Batch Additions Before Queries

```rust
// Add all documents first
for doc in documents {
    backend.add(doc.id, &doc.embedding)?;
}

// Then query (triggers HNSW build once)
let results = backend.query(&query, 10);
```

### 3. Tune HNSW for Your Use Case

```rust
// For speed-critical applications
let config = HnswConfig {
    ef_construction: 100,
    m: 8,
    ef_search: 50,
};

// For recall-critical applications
let config = HnswConfig {
    ef_construction: 400,
    m: 32,
    ef_search: 200,
};
```

## See Also

- [Overview](overview.md) - RAG module introduction
- [Index](index.md) - Using backends with RagIndex
- [Embedder](../neural/embedder.md) - Generating embeddings
