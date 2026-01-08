# RAG Retriever

The `Retriever` provides a high-level interface for querying RAG indices with text queries.

## Retriever Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Retriever<B>                                     │
│                                                                          │
│  Text Query: "What is machine learning?"                                │
│       │                                                                  │
│       ▼                                                                  │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    ModernBertEmbedder                             │  │
│  │                                                                   │  │
│  │  Text → Tokenize → Transform → Pool → Normalize → [f32; 768]     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│       │                                                                  │
│       │ Query Embedding                                                  │
│       ▼                                                                  │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    RagIndex<B>                                    │  │
│  │                                                                   │  │
│  │  query(embedding, top_k) → Vec<(DocumentMeta, score)>            │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│       │                                                                  │
│       │ Raw Results                                                      │
│       ▼                                                                  │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Filtering                                      │  │
│  │                                                                   │  │
│  │  • min_similarity threshold                                       │  │
│  │  • include_explicit_synopsis                                      │  │
│  │  • include_generated_synopsis                                     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│       │                                                                  │
│       ▼                                                                  │
│  Vec<RetrievalResult>                                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Configuration

```rust
use libgrammstein::rag::RetrievalConfig;

let config = RetrievalConfig {
    // Number of results to return
    top_k: 10,

    // Minimum similarity threshold (0.0 to 1.0)
    min_similarity: 0.0,

    // Include documents with explicit (author-provided) synopses
    include_explicit_synopsis: true,

    // Include documents with generated synopses
    include_generated_synopsis: true,
};
```

## Creating a Retriever

```rust
use std::sync::Arc;
use libgrammstein::rag::{RagIndex, Retriever, RetrievalConfig, ExactCosineBackend};
use libgrammstein::neural::{ModernBertEmbedder, EmbeddingConfig};

// Load or create index
let index: RagIndex<ExactCosineBackend> = RagIndex::load("./index")?;

// Create embedder for query encoding
let embedder = ModernBertEmbedder::new(EmbeddingConfig::default())?;

// Create retriever
let config = RetrievalConfig::default();
let retriever = Retriever::new(Arc::new(index), embedder, config);
```

## Querying

### Text Query

```rust
let results = retriever.query("What is machine learning?")?;

for result in &results {
    println!("{}. {} (score: {:.2})",
        result.rank,
        result.display_title(),
        result.score
    );
    println!("   {}", result.synopsis);
}
```

### Pre-computed Embedding Query

```rust
// When you already have the embedding
let embedding = embedder.embed_query("What is ML?")?;
let results = retriever.query_with_embedding(&embedding)?;
```

## RetrievalResult

Each result contains document information and scoring:

```rust
pub struct RetrievalResult {
    /// Document URI
    pub uri: String,

    /// Document title (if available)
    pub title: Option<String>,

    /// Document synopsis
    pub synopsis: String,

    /// Whether synopsis is explicit (author-provided)
    pub synopsis_is_explicit: bool,

    /// Similarity score (0.0 to 1.0)
    pub score: f32,

    /// Rank in results (1 = best)
    pub rank: usize,
}
```

### Display Helpers

```rust
for result in results {
    // Use title if available, otherwise URI
    let title = result.display_title();

    // Format score
    println!("{}: {:.2}", title, result.score);

    // Check synopsis type
    if result.synopsis_is_explicit {
        println!("(Author synopsis)");
    } else {
        println!("(Generated synopsis)");
    }
}
```

## Filtering

### Minimum Similarity

```rust
let config = RetrievalConfig {
    min_similarity: 0.5,  // Only return results with score >= 0.5
    ..Default::default()
};
```

### Synopsis Type Filtering

```rust
// Only explicit synopses (high-quality metadata)
let config = RetrievalConfig {
    include_explicit_synopsis: true,
    include_generated_synopsis: false,
    ..Default::default()
};

// Only generated synopses (for testing summarizer)
let config = RetrievalConfig {
    include_explicit_synopsis: false,
    include_generated_synopsis: true,
    ..Default::default()
};
```

## Dynamic Configuration

Update configuration at runtime:

```rust
let mut retriever = Retriever::new(index, embedder, config);

// Update config
retriever.set_config(RetrievalConfig {
    top_k: 20,
    min_similarity: 0.3,
    ..Default::default()
});

// Get current config
let config = retriever.config();
println!("top_k: {}", config.top_k);
```

## Batch Retrieval

For multiple queries:

```rust
use libgrammstein::rag::BatchRetriever;

let batch_retriever = BatchRetriever::new(retriever);

let queries = vec![
    "What is machine learning?",
    "How do neural networks work?",
    "What is deep learning?",
];

let all_results = batch_retriever.query_batch(&queries)?;

for (query, results) in queries.iter().zip(all_results.iter()) {
    println!("Query: {}", query);
    for result in results {
        println!("  - {}: {:.2}", result.display_title(), result.score);
    }
}
```

## Result Formatting

Pretty-print results:

```rust
use libgrammstein::rag::format_results;

let results = retriever.query("What is ML?")?;
let formatted = format_results(&results);
println!("{}", formatted);
```

Output:
```
1. [0.95] Introduction to Machine Learning
   URI: file:///docs/intro.md
   Synopsis (explicit): Overview of ML concepts and applications

2. [0.82] Neural Networks Guide
   URI: file:///docs/nn.md
   Synopsis (generated): Neural networks are computing systems...
```

## Accessing Components

```rust
// Get index reference
let index = retriever.index();
println!("Index size: {}", index.len());

// Get embedder reference
let embedder = retriever.embedder();

// Get mutable embedder (e.g., for cache clearing)
let embedder_mut = retriever.embedder_mut();
embedder_mut.clear_cache();
```

## Creating Results from Metadata

For custom result construction:

```rust
use libgrammstein::rag::RetrievalResult;

let result = RetrievalResult::from_meta(&document_meta, score, rank);
```

## Thread Safety

The retriever uses shared index via `Arc`:

```rust
use std::sync::Arc;
use std::thread;

// Index is shared (read-only)
let index = Arc::new(RagIndex::load("./index")?);

// Multiple retrievers can share the same index
let retriever1 = Retriever::new(Arc::clone(&index), embedder1, config);
let retriever2 = Retriever::new(Arc::clone(&index), embedder2, config);
```

## Error Handling

```rust
use libgrammstein::rag::RagError;

match retriever.query(query_text) {
    Ok(results) => {
        for result in results {
            println!("{}: {:.2}", result.display_title(), result.score);
        }
    }
    Err(RagError::EmbeddingError(msg)) => {
        eprintln!("Failed to embed query: {}", msg);
    }
    Err(e) => eprintln!("Query error: {}", e),
}
```

## Best Practices

### 1. Reuse Retriever

```rust
// Good: reuse retriever
let retriever = Retriever::new(index, embedder, config);
for query in queries {
    let results = retriever.query(query)?;
}

// Bad: recreate retriever
for query in queries {
    let retriever = Retriever::new(index.clone(), embedder.clone(), config);
    let results = retriever.query(query)?;
}
```

### 2. Use Batch for Multiple Queries

```rust
// Efficient for multiple queries
let batch = BatchRetriever::new(retriever);
let all_results = batch.query_batch(&queries)?;
```

### 3. Set Appropriate Thresholds

```rust
// For strict relevance
let config = RetrievalConfig {
    min_similarity: 0.7,
    top_k: 5,
    ..Default::default()
};

// For broad exploration
let config = RetrievalConfig {
    min_similarity: 0.0,
    top_k: 50,
    ..Default::default()
};
```

### 4. Cache Query Embeddings

The embedder has built-in caching. For repeated queries:

```rust
let embedder = ModernBertEmbedder::new(EmbeddingConfig {
    cache_size: 1000,  // Cache common queries
    ..Default::default()
})?;
```

## See Also

- [Overview](overview.md) - RAG module introduction
- [Index](index.md) - RagIndex operations
- [Embedder](../neural/embedder.md) - Query embedding
- [Document](document.md) - Result metadata
