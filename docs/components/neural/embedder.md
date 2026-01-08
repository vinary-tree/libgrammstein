# ModernBERT Embedder

The `ModernBertEmbedder` generates dense vector embeddings for documents and queries using ModernBERT.

## What are Embeddings?

Embeddings are dense vector representations that capture semantic meaning:

```
Text: "Machine learning is a branch of AI"
            │
            ▼
   ┌─────────────────┐
   │  ModernBERT     │
   │  Embedder       │
   └────────┬────────┘
            │
            ▼
Embedding: [0.23, -0.45, 0.12, ..., 0.67]  (768 dimensions)
```

Similar texts produce similar embeddings, enabling:
- **Semantic search**: Find relevant documents by meaning, not keywords
- **Clustering**: Group similar documents together
- **Classification**: Use embeddings as features for ML models

## Embedder Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      ModernBertEmbedder                                  │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    ModernBertModel (Arc)                          │  │
│  │                    [shared across threads]                        │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                 │                                        │
│         ┌───────────────────────┼───────────────────────┐               │
│         │                       │                       │               │
│         ▼                       ▼                       ▼               │
│  ┌─────────────┐        ┌─────────────┐        ┌─────────────┐         │
│  │ embed_query │        │embed_document│       │ embed_batch │         │
│  │             │        │             │        │             │         │
│  │ Optimized   │        │ Full text   │        │ Parallel    │         │
│  │ for search  │        │ embedding   │        │ processing  │         │
│  └──────┬──────┘        └──────┬──────┘        └──────┬──────┘         │
│         │                      │                      │                 │
│         └──────────────────────┼──────────────────────┘                 │
│                                │                                        │
│                                ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                     Pooling Strategy                              │  │
│  │                                                                   │  │
│  │  ┌─────────┐     ┌──────────────┐     ┌──────────────┐          │  │
│  │  │  CLS    │     │ MeanPooling  │     │ MaxPooling   │          │  │
│  │  │ [CLS]   │     │ avg(tokens)  │     │ max(tokens)  │          │  │
│  │  │ token   │     │              │     │              │          │  │
│  │  └─────────┘     └──────────────┘     └──────────────┘          │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                │                                        │
│                                ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Normalization (optional)                       │  │
│  │                    ||embedding|| = 1.0                            │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                │                                        │
│                                ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    EmbeddingCache (optional)                      │  │
│  │                    Lock-free DashMap                              │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Configuration

```rust
use libgrammstein::neural::{EmbeddingConfig, ModernBertConfig, PoolingStrategy};

let config = EmbeddingConfig {
    // Model configuration
    model_config: ModernBertConfig::default(),

    // How to aggregate token embeddings
    pooling: PoolingStrategy::MeanPooling,

    // Normalize to unit length for cosine similarity
    normalize: true,

    // Cache size (0 to disable)
    cache_size: 10000,

    // Batch size for parallel processing
    batch_size: 32,
};
```

### Pooling Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `Cls` | Use [CLS] token embedding | Classification tasks |
| `MeanPooling` | Average all token embeddings | Semantic similarity (default) |
| `MaxPooling` | Max across all token embeddings | Keyword-focused tasks |

## Creating an Embedder

### From Configuration

```rust
use libgrammstein::neural::{ModernBertEmbedder, EmbeddingConfig};

let config = EmbeddingConfig::default();
let embedder = ModernBertEmbedder::new(config)?;
```

### From Existing Model

```rust
use std::sync::Arc;
use libgrammstein::neural::{ModernBertModel, ModernBertEmbedder, EmbeddingConfig};

let model = Arc::new(ModernBertModel::load(&model_config)?);
let embedder = ModernBertEmbedder::from_model(model, EmbeddingConfig::default());
```

## Embedding Documents

### Single Document

```rust
let text = "Machine learning is a subfield of artificial intelligence...";
let embedding = embedder.embed_document(text)?;
// Returns: Vec<f32> with 768 dimensions
```

### Batch Documents

```rust
let documents = vec![
    "First document text...",
    "Second document text...",
    "Third document text...",
];

let embeddings = embedder.embed_batch(&documents)?;
// Returns: Vec<Vec<f32>> with 768 dimensions each
```

## Embedding Queries

Queries are typically shorter and optimized differently:

```rust
let query = "What is machine learning?";
let embedding = embedder.embed_query(query)?;
```

## Similarity Computation

### Cosine Similarity

Embeddings are normalized by default, so dot product equals cosine similarity:

```rust
let doc_embedding = embedder.embed_document("Machine learning...")?;
let query_embedding = embedder.embed_query("What is ML?")?;

let similarity = embedder.cosine_similarity(&query_embedding, &doc_embedding);
// Returns: f32 in range [-1.0, 1.0], higher = more similar
```

### Manual Normalization

```rust
let unnormalized = vec![0.5, 0.3, 0.8];
let normalized = ModernBertEmbedder::normalize(&unnormalized);
// ||normalized|| = 1.0
```

## Text Truncation

Long texts are automatically truncated to fit the model's context:

```rust
// Truncate to approximately max_tokens
let truncated = embedder.truncate_text(long_text, 512);
```

The truncation uses an estimate of ~4 characters per token.

## Caching

### Embedding Cache

The embedder maintains an optional LRU cache for repeated embeddings:

```rust
// Get cache statistics
let (hits, misses) = embedder.cache_stats();
println!("Cache hit rate: {:.1}%", 100.0 * hits as f32 / (hits + misses) as f32);

// Clear cache
embedder.clear_cache();
```

### Cache Configuration

```rust
let config = EmbeddingConfig {
    cache_size: 50000,  // Cache up to 50k embeddings
    ..Default::default()
};

// Disable caching
let config = EmbeddingConfig {
    cache_size: 0,
    ..Default::default()
};
```

## Batch Document Embedder

For processing large corpora with progress tracking:

```rust
use libgrammstein::neural::{BatchDocumentEmbedder, DocumentEmbedding};

let batch_embedder = BatchDocumentEmbedder::new(embedder);

let documents = vec![
    ("doc1", "First document content..."),
    ("doc2", "Second document content..."),
];

let embeddings: Vec<DocumentEmbedding> = batch_embedder.embed_documents(
    documents.iter().map(|(id, text)| (*id, None, *text)),
)?;

for emb in embeddings {
    println!("Doc {}: {} dimensions", emb.document_id, emb.embedding.len());
}
```

## Thread Safety

The embedder supports concurrent access without locks:

```rust
use std::sync::Arc;
use rayon::prelude::*;

let embedder = Arc::new(ModernBertEmbedder::new(config)?);

let embeddings: Vec<_> = documents
    .par_iter()
    .map(|doc| {
        embedder.embed_document(doc).expect("embedding failed")
    })
    .collect();
```

Key design features:
- `ModernBertModel` wrapped in `Arc` for shared ownership
- All embedding methods use `&self` (immutable reference)
- `EmbeddingCache` uses lock-free `DashMap`

## Embedding Dimensions

| Model | Embedding Size |
|-------|----------------|
| ModernBERT-base | 768 |

Get the dimension programmatically:

```rust
let dim = embedder.embedding_dim();  // 768
```

## Error Handling

```rust
use libgrammstein::neural::NeuralError;

match embedder.embed_document(text) {
    Ok(embedding) => {
        println!("Embedded to {} dimensions", embedding.len());
    }
    Err(NeuralError::Tokenization(msg)) => {
        eprintln!("Failed to tokenize: {}", msg);
    }
    Err(NeuralError::Inference(msg)) => {
        eprintln!("Model inference failed: {}", msg);
    }
    Err(e) => {
        eprintln!("Other error: {}", e);
    }
}
```

## Best Practices

### 1. Batch Processing

Batch embedding is more efficient than individual calls:

```rust
// Efficient: batch processing
let embeddings = embedder.embed_batch(&documents)?;

// Less efficient: individual calls
let embeddings: Vec<_> = documents
    .iter()
    .map(|d| embedder.embed_document(d))
    .collect::<Result<_, _>>()?;
```

### 2. Reuse Embedder

Create once, use many times:

```rust
// Good: reuse embedder
let embedder = ModernBertEmbedder::new(config)?;
for doc in documents {
    embedder.embed_document(doc)?;
}

// Bad: recreate embedder
for doc in documents {
    let embedder = ModernBertEmbedder::new(config)?;  // Reloads model!
    embedder.embed_document(doc)?;
}
```

### 3. Enable Caching for Repeated Queries

```rust
let config = EmbeddingConfig {
    cache_size: 10000,  // Cache frequent queries
    ..Default::default()
};
```

## See Also

- [Overview](overview.md) - Neural module introduction
- [Model](model.md) - ModernBERT model details
- [RAG Builder](../rag/builder.md) - Using embedder in RAG pipeline
- [Cache](cache.md) - Caching strategies
